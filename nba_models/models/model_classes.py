"""
Portable Model Classes for NBA Betting Model

This module contains minimal class definitions for unpickling trained models.
These classes are designed to work WITHOUT the heavy training dependencies
(XGBoost, LightGBM, CatBoost) at import time - the actual model instances
inside are already trained and only need scikit-learn for predict().

The issue: When models are trained via `python train_complete_balldontlie.py`,
pickle saves class references as `__main__.ClassName`. When loading on Railway,
Python tries to find these classes in `uvicorn.__main__` which fails.

This module provides the class definitions so unpickling works correctly.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import contextlib


# =============================================================================
# SMART FILLNA FOR INFERENCE
# =============================================================================

def smart_fillna(df: pd.DataFrame, game_date: str = None) -> pd.DataFrame:
    """
    Apply sensible defaults for missing values during inference.

    Simplified version that doesn't require the training module.
    Uses NBA-realistic defaults for common feature patterns.
    """
    result = df.copy()

    # Feature-specific defaults based on NBA averages
    defaults = {
        # Scoring
        'avg_pts': 15.0, 'avg_min': 25.0, 'avg_reb': 5.0, 'avg_ast': 3.0,
        'avg_threes': 1.5, 'avg_pra': 23.0,
        # Efficiency
        'fg_pct': 0.46, 'fg3_pct': 0.36, 'ft_pct': 0.78,
        'efg_pct': 0.54, 'ts_pct': 0.57,
        # Team ratings
        'off_rating': 114.0, 'def_rating': 114.0, 'net_rating': 0.0,
        'pace': 100.0, 'ortg': 114.0, 'drtg': 114.0,
        # Elo
        'elo': 1500.0, 'elo_diff': 0.0,
        # Usage
        'usg_pct': 0.20, 'usage_rate': 0.20,
        # Other
        'days_rest': 1.0, 'is_home': 0.5, 'is_back_to_back': 0.0,
        'games_played': 40.0,
    }

    for col in result.columns:
        if result[col].isna().any():
            # Check for exact match
            if col in defaults:
                result[col] = result[col].fillna(defaults[col])
            # Check for pattern matches
            elif 'pts' in col.lower() or 'point' in col.lower():
                result[col] = result[col].fillna(15.0)
            elif 'reb' in col.lower():
                result[col] = result[col].fillna(5.0)
            elif 'ast' in col.lower():
                result[col] = result[col].fillna(3.0)
            elif 'min' in col.lower():
                result[col] = result[col].fillna(25.0)
            elif 'pct' in col.lower() or 'rate' in col.lower():
                result[col] = result[col].fillna(0.5)
            elif 'rating' in col.lower():
                result[col] = result[col].fillna(114.0)
            elif 'elo' in col.lower():
                result[col] = result[col].fillna(0.0 if 'diff' in col.lower() else 1500.0)
            else:
                result[col] = result[col].fillna(0.0)

    return result


# =============================================================================
# PROP ENSEMBLE MODEL
# =============================================================================

class PropEnsembleModel:
    """
    Ensemble model for player prop predictions.

    This is the inference-only version. The actual sub-models (XGBoost, LightGBM,
    etc.) are already trained and stored inside self.models dict.
    """

    def __init__(self, prop_type: str = None, optimized_params: dict | None = None):
        self.prop_type = prop_type
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_metrics = {}
        self.cv_scores = {}
        self.meta_model = None
        self.over_under_classifier = None
        self.optimized_params = optimized_params
        self.model_weights = {}

    def predict(self, features: dict, prop_line: float = None) -> dict:
        """Make a prediction with the ensemble."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = smart_fillna(X[self.feature_names])
        X_scaled = self.scaler.transform(X)

        # Get base model predictions
        base_preds = []
        individual_preds = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[0]
                base_preds.append(pred)
                individual_preds[name] = pred
            except Exception:
                pass

        if not base_preds:
            raise ValueError("No base models available for prediction")

        # Weighted average prediction
        ensemble_pred = 0.0
        if hasattr(self, 'model_weights') and self.model_weights:
            for name, pred in individual_preds.items():
                weight = self.model_weights.get(name, 1.0 / len(individual_preds))
                ensemble_pred += weight * pred
        else:
            ensemble_pred = float(np.mean(base_preds))

        result = {
            'predicted_value': ensemble_pred,
            'prop_type': self.prop_type,
            'individual_predictions': individual_preds,
            'model_agreement': 1 - (np.std(base_preds) / max(np.mean(base_preds), 1)),
        }

        if prop_line is not None:
            result['prop_line'] = prop_line
            result['prediction'] = 'over' if ensemble_pred > prop_line else 'under'
            result['edge'] = ensemble_pred - prop_line
            result['edge_pct'] = (ensemble_pred - prop_line) / prop_line if prop_line > 0 else 0

            # Over/under probability
            if self.over_under_classifier is not None:
                try:
                    residual_features = np.array([[ensemble_pred, abs(ensemble_pred - prop_line)]])
                    proba = self.over_under_classifier.predict_proba(residual_features)[0]
                    result['over_probability'] = float(proba[1])
                    result['under_probability'] = float(proba[0])
                except Exception:
                    result['over_probability'] = 0.5 + (result['edge_pct'] * 2)
                    result['over_probability'] = max(0.3, min(0.7, result['over_probability']))
                    result['under_probability'] = 1 - result['over_probability']

        return result

    def _init_base_models(self, optimized_params=None):
        """Placeholder for training - not needed for inference."""
        pass


# =============================================================================
# QUANTILE PROP MODEL
# =============================================================================

class QuantilePropModel:
    """
    Quantile regression model for uncertainty estimation in prop predictions.

    Predicts the full distribution of possible outcomes by training separate
    models at different quantiles (10th, 25th, 50th, 75th, 90th percentiles).
    """

    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

    def __init__(self, prop_type: str = None):
        self.prop_type = prop_type
        self.quantile_models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_metrics = {}

    def predict_distribution(self, features: dict) -> dict[float, float]:
        """Predict the full distribution of outcomes."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = smart_fillna(X[self.feature_names])
        X_scaled = self.scaler.transform(X)

        return {q: float(model.predict(X_scaled)[0])
                for q, model in self.quantile_models.items()}

    def predict_over_probability(self, features: dict, line: float) -> float:
        """
        Estimate probability of actual value being OVER the line.
        Uses linear interpolation between quantile predictions.
        """
        dist = self.predict_distribution(features)

        # Sort quantiles
        qs = sorted(dist.keys())
        vals = [dist[q] for q in qs]

        # If line is below all predictions, high probability of over
        if line <= vals[0]:
            return 0.90 + 0.10 * (vals[0] - line) / max(vals[0], 1)

        # If line is above all predictions, low probability of over
        if line >= vals[-1]:
            return 0.10 * max(0, vals[-1] - line + 5) / 5

        # Interpolate between quantiles
        for i in range(len(qs) - 1):
            if vals[i] <= line <= vals[i + 1]:
                frac = (line - vals[i]) / (vals[i + 1] - vals[i] + 0.001)
                prob_over = (1 - qs[i]) - frac * (qs[i + 1] - qs[i])
                return max(0.05, min(0.95, prob_over))

        return 0.50

    def get_confidence_interval(self, features: dict) -> dict:
        """Get 80% confidence interval for prediction."""
        dist = self.predict_distribution(features)
        return {
            'lower_10': dist.get(0.10, 0),
            'lower_25': dist.get(0.25, 0),
            'median': dist.get(0.50, 0),
            'upper_75': dist.get(0.75, 0),
            'upper_90': dist.get(0.90, 0),
        }


# =============================================================================
# POSITION-AWARE PROP ENSEMBLE
# =============================================================================

class PositionAwarePropEnsemble:
    """
    Position-specific models for better predictions.

    Trains separate models for guards, forwards, and centers because:
    - Centers have very different rebound distributions than guards
    - Guards have different assist patterns than forwards
    """

    POSITION_GROUPS = {
        'guards': ['is_guard'],
        'forwards': ['is_forward'],
        'centers': ['is_center'],
    }

    MIN_SAMPLES_PER_POSITION = 500

    def __init__(self, prop_type: str = None):
        self.prop_type = prop_type
        self.position_models = {}  # {position: PropEnsembleModel}
        self.general_model = None
        self.position_metrics = {}
        self.is_fitted = False
        self.training_metrics = {}

    def _get_position_group(self, features: dict) -> str:
        """Determine position group from features."""
        if features.get('is_center', 0) == 1:
            return 'centers'
        if features.get('is_forward', 0) == 1:
            return 'forwards'
        return 'guards'

    def predict(self, features: dict, prop_line: float = None) -> dict:
        """Make prediction using position-appropriate model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        position = self._get_position_group(features)

        # Use position-specific model if available
        if position in self.position_models:
            model = self.position_models[position]
            result = model.predict(features, prop_line)
            result['model_type'] = f'position_{position}'
        else:
            result = self.general_model.predict(features, prop_line)
            result['model_type'] = 'general_fallback'

        return result


# =============================================================================
# SPREAD ENSEMBLE WRAPPER
# =============================================================================

class SpreadEnsembleWrapper:
    """Wrapper for spread ensemble prediction that can be pickled."""

    def __init__(self, models=None, weights=None, scaler=None, feature_names=None, metrics=None):
        self.models = models or {}
        self.weights = weights or {}
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.training_metrics = metrics or {}

    def predict(self, X):
        """Make spread prediction."""
        X_arr = np.array(X)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1, -1)
        X_scaled = self.scaler.transform(X_arr)
        pred = np.zeros(X_scaled.shape[0])
        for name, model in self.models.items():
            pred += self.weights.get(name, 0) * model.predict(X_scaled)
        return pred


# =============================================================================
# ENSEMBLE MONEYLINE WRAPPER
# =============================================================================

class EnsembleMoneylineWrapper:
    """Wrapper for moneyline ensemble prediction."""

    def __init__(self, models=None, weights=None, scaler=None, feature_names=None, metrics=None):
        self.models = models or {}
        self.weights = weights or {}
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.training_metrics = metrics or {}

    def predict(self, X):
        """Make moneyline prediction."""
        X_arr = np.array(X)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1, -1)
        X_scaled = self.scaler.transform(X_arr)
        pred = np.zeros(X_scaled.shape[0])
        for name, model in self.models.items():
            pred += self.weights.get(name, 0) * model.predict(X_scaled)
        return pred

    def predict_proba(self, X):
        """Make probability prediction if available."""
        X_arr = np.array(X)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1, -1)
        X_scaled = self.scaler.transform(X_arr)

        # Average probability predictions from models that support it
        probas = []
        for _name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                with contextlib.suppress(Exception):
                    probas.append(model.predict_proba(X_scaled))

        if probas:
            return np.mean(probas, axis=0)
        # Fallback to logistic sigmoid of prediction
        pred = self.predict(X)
        proba = 1 / (1 + np.exp(-pred))
        return np.column_stack([1 - proba, proba])
