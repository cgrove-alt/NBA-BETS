"""
Minutes Oracle - Main Prediction Class

Predicts minutes distribution using quantile regression.
Returns percentiles: 10th, 25th, 50th (median), 75th, 90th

Usage:
    predictor = MinutesPredictor.load('models/minutes_oracle.pkl')

    minutes_dist = predictor.predict(
        player_id=203999,  # Jokic
        game_context={
            'vegas_spread': -8.5,
            'vegas_total': 225.5,
            'opponent_team_id': 1610612738,
            'is_home': True,
            'is_back_to_back': False,
            'days_rest': 2,
        }
    )

    # Returns:
    # {
    #     'p10': 28.2,   # Floor (blowout scenario)
    #     'p25': 31.5,
    #     'p50': 34.1,   # Most likely
    #     'p75': 36.8,
    #     'p90': 39.5,   # Ceiling (OT/close game)
    #     'expected': 33.8,  # Weighted mean
    #     'uncertainty': 'medium',  # low/medium/high
    # }
"""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any, Union
from dataclasses import dataclass, field
import warnings

# Suppress LightGBM warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Using sklearn fallback.")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .minutes_features import (
    MinutesFeatureGenerator,
    MINUTES_FEATURE_NAMES,
    features_to_array,
)


@dataclass
class MinutesDistribution:
    """Data class for minutes prediction output."""
    p10: float
    p25: float
    p50: float  # Median
    p75: float
    p90: float
    expected: float  # Weighted mean
    uncertainty: str  # 'low', 'medium', 'high'
    spread: float  # p90 - p10
    player_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'p10': round(self.p10, 1),
            'p25': round(self.p25, 1),
            'p50': round(self.p50, 1),
            'p75': round(self.p75, 1),
            'p90': round(self.p90, 1),
            'expected': round(self.expected, 1),
            'uncertainty': self.uncertainty,
            'spread': round(self.spread, 1),
            'player_id': self.player_id,
        }


class MinutesPredictor:
    """
    Predicts minutes distribution using quantile regression.

    Uses LightGBM with quantile loss for each percentile.
    Falls back to sklearn GradientBoostingRegressor if LightGBM unavailable.
    """

    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
    QUANTILE_NAMES = ['p10', 'p25', 'p50', 'p75', 'p90']

    # Interval scaling factor to widen prediction intervals for better coverage
    # 1.0 = no scaling, 1.1 = 10% wider intervals
    INTERVAL_SCALE_FACTOR = 1.15  # Widen by 15% to improve coverage from 75% to ~80%

    def __init__(self,
                 feature_names: list[str] | None = None,
                 model_params: dict | None = None,
                 interval_scale: float | None = None):
        """
        Initialize the predictor.

        Args:
            feature_names: List of feature names in order
            model_params: LightGBM/sklearn parameters
            interval_scale: Scale factor for widening prediction intervals (default: 1.15)
        """
        self.feature_names = feature_names or MINUTES_FEATURE_NAMES
        self.model_params = model_params or self._default_params()
        self.interval_scale = interval_scale if interval_scale is not None else self.INTERVAL_SCALE_FACTOR

        # One model per quantile
        self.models: dict[str, Any] = {}

        # Training metadata
        self.trained = False
        self.training_samples = 0
        self.training_date = None
        self.feature_importances: dict[str, float] = {}

        # Feature generator (optional, for convenience methods)
        self._feature_generator: MinutesFeatureGenerator | None = None

    def _default_params(self) -> dict:
        """Default model parameters optimized for minutes prediction."""
        if HAS_LIGHTGBM:
            return {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1,
            }
        else:
            return {
                'n_estimators': 150,
                'max_depth': 5,
                'learning_rate': 0.1,
                'min_samples_leaf': 20,
                'subsample': 0.8,
                'random_state': 42,
            }

    def train(self,
              X: Union[np.ndarray, pd.DataFrame],
              y: Union[np.ndarray, pd.Series],
              sample_weights: np.ndarray | None = None,
              validation_split: float = 0.2,
              verbose: bool = True) -> dict[str, float]:
        """
        Train quantile regression models.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target minutes values (n_samples,)
            sample_weights: Optional sample weights for time decay
            validation_split: Fraction for validation
            verbose: Print training progress

        Returns:
            Dictionary of validation metrics
        """
        from datetime import datetime

        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Remove any NaN/inf values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        if sample_weights is not None:
            sample_weights = sample_weights[mask]

        if verbose:
            print(f"Training Minutes Oracle on {len(X)} samples...")

        # Temporal split (use most recent data for validation)
        n_samples = len(X)
        split_idx = int(n_samples * (1 - validation_split))

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if sample_weights is not None:
            w_train = sample_weights[:split_idx]
        else:
            w_train = None

        metrics = {}

        # Train a model for each quantile
        for q, q_name in zip(self.QUANTILES, self.QUANTILE_NAMES, strict=False):
            if verbose:
                print(f"  Training {q_name} ({q:.0%}) model...")

            if HAS_LIGHTGBM:
                model = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=q,
                    **self.model_params
                )
            elif HAS_SKLEARN:
                model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=q,
                    **self.model_params
                )
            else:
                raise ImportError("Neither LightGBM nor sklearn available")

            # Train
            if HAS_LIGHTGBM and w_train is not None:
                model.fit(X_train, y_train, sample_weight=w_train)
            else:
                model.fit(X_train, y_train)

            self.models[q_name] = model

            # Validation predictions
            val_preds = model.predict(X_val)

            # Calculate quantile loss (pinball loss)
            errors = y_val - val_preds
            quantile_loss = np.mean(np.maximum(q * errors, (q - 1) * errors))
            metrics[f'{q_name}_loss'] = quantile_loss

            # Calculate coverage (what % of actuals are below this quantile)
            coverage = np.mean(y_val <= val_preds)
            metrics[f'{q_name}_coverage'] = coverage

        # Calculate overall metrics
        p50_preds = self.models['p50'].predict(X_val)
        metrics['median_rmse'] = np.sqrt(np.mean((y_val - p50_preds) ** 2))
        metrics['median_mae'] = np.mean(np.abs(y_val - p50_preds))

        # Coverage of p10-p90 interval
        p10_preds = self.models['p10'].predict(X_val)
        p90_preds = self.models['p90'].predict(X_val)
        metrics['p10_p90_coverage'] = np.mean((y_val >= p10_preds) & (y_val <= p90_preds))

        # Feature importances (from median model)
        if HAS_LIGHTGBM and hasattr(self.models['p50'], 'feature_importances_'):
            importances = self.models['p50'].feature_importances_
            self.feature_importances = dict(zip(self.feature_names, importances, strict=False))

        # Update metadata
        self.trained = True
        self.training_samples = n_samples
        self.training_date = datetime.now().isoformat()

        if verbose:
            print("\nTraining complete!")
            print(f"  Median RMSE: {metrics['median_rmse']:.2f} minutes")
            print(f"  Median MAE: {metrics['median_mae']:.2f} minutes")
            print(f"  P10-P90 Coverage: {metrics['p10_p90_coverage']:.1%}")
            print(f"  P50 Calibration: {metrics['p50_coverage']:.1%} (target: 50%)")

        return metrics

    def predict(self,
                features: Union[dict, np.ndarray, pd.DataFrame],
                player_id: int | None = None) -> MinutesDistribution:
        """
        Predict minutes distribution for a single player.

        Args:
            features: Feature dict, array, or DataFrame row
            player_id: Optional player ID for tracking

        Returns:
            MinutesDistribution object
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Convert features to array
        if isinstance(features, dict):
            X = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        elif isinstance(features, pd.DataFrame):
            X = features[self.feature_names].values
        else:
            X = np.atleast_2d(features)

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Get predictions for each quantile
        predictions = {}
        for q_name in self.QUANTILE_NAMES:
            pred = self.models[q_name].predict(X)[0]
            # Clamp to reasonable range
            predictions[q_name] = float(np.clip(pred, 0, 53))  # 53 = max in OT

        # Apply interval scaling to widen prediction intervals for better coverage
        predictions = self._apply_interval_scaling(predictions)

        # Ensure monotonicity (p10 <= p25 <= p50 <= p75 <= p90)
        predictions = self._ensure_monotonic(predictions)

        # Calculate expected value (weighted mean)
        expected = self._calculate_expected(predictions)

        # Calculate uncertainty
        spread = predictions['p90'] - predictions['p10']
        uncertainty = self._classify_uncertainty(spread)

        return MinutesDistribution(
            p10=predictions['p10'],
            p25=predictions['p25'],
            p50=predictions['p50'],
            p75=predictions['p75'],
            p90=predictions['p90'],
            expected=expected,
            uncertainty=uncertainty,
            spread=spread,
            player_id=player_id,
        )

    def predict_batch(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      player_ids: list[int] | None = None) -> list[MinutesDistribution]:
        """
        Predict minutes distribution for multiple players.

        Args:
            X: Feature matrix (n_samples, n_features)
            player_ids: Optional list of player IDs

        Returns:
            List of MinutesDistribution objects
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n_samples = X.shape[0]

        # Get predictions for all quantiles
        all_preds = {}
        for q_name in self.QUANTILE_NAMES:
            preds = self.models[q_name].predict(X)
            all_preds[q_name] = np.clip(preds, 0, 53)

        # Build results
        results = []
        for i in range(n_samples):
            preds = {q_name: all_preds[q_name][i] for q_name in self.QUANTILE_NAMES}
            preds = self._apply_interval_scaling(preds)
            preds = self._ensure_monotonic(preds)

            spread = preds['p90'] - preds['p10']

            dist = MinutesDistribution(
                p10=preds['p10'],
                p25=preds['p25'],
                p50=preds['p50'],
                p75=preds['p75'],
                p90=preds['p90'],
                expected=self._calculate_expected(preds),
                uncertainty=self._classify_uncertainty(spread),
                spread=spread,
                player_id=player_ids[i] if player_ids else None,
            )
            results.append(dist)

        return results

    def _apply_interval_scaling(self, preds: dict[str, float]) -> dict[str, float]:
        """
        Apply interval scaling to widen prediction intervals for better coverage.

        Expands the distance from median (p50) for outer quantiles.
        A scale of 1.15 widens the p10-p90 interval by ~15%.
        """
        if self.interval_scale == 1.0:
            return preds

        p50 = preds['p50']

        # Scale distances from median
        scaled = {'p50': p50}

        # For lower quantiles, push them further below median
        scaled['p10'] = p50 - (p50 - preds['p10']) * self.interval_scale
        scaled['p25'] = p50 - (p50 - preds['p25']) * self.interval_scale

        # For upper quantiles, push them further above median
        scaled['p75'] = p50 + (preds['p75'] - p50) * self.interval_scale
        scaled['p90'] = p50 + (preds['p90'] - p50) * self.interval_scale

        # Clamp to valid range
        for key in scaled:
            scaled[key] = float(np.clip(scaled[key], 0, 53))

        return scaled

    def _ensure_monotonic(self, preds: dict[str, float]) -> dict[str, float]:
        """Ensure quantile predictions are monotonically increasing."""
        sorted_vals = sorted([preds['p10'], preds['p25'], preds['p50'], preds['p75'], preds['p90']])
        return {
            'p10': sorted_vals[0],
            'p25': sorted_vals[1],
            'p50': sorted_vals[2],
            'p75': sorted_vals[3],
            'p90': sorted_vals[4],
        }

    def _calculate_expected(self, preds: dict[str, float]) -> float:
        """
        Calculate expected minutes as weighted average of quantiles.

        Uses trapezoidal approximation of the distribution.
        """
        # Weights that approximate the mean from quantiles
        # Based on trapezoidal rule for quantile function
        weights = {
            'p10': 0.15,
            'p25': 0.20,
            'p50': 0.30,
            'p75': 0.20,
            'p90': 0.15,
        }

        return sum(preds[q] * w for q, w in weights.items())

    def _classify_uncertainty(self, spread: float) -> str:
        """
        Classify prediction uncertainty based on p10-p90 spread.

        Args:
            spread: p90 - p10 in minutes

        Returns:
            'low', 'medium', or 'high'
        """
        if spread < 6:
            return 'low'    # Very consistent player/situation
        elif spread < 10:
            return 'medium'  # Normal variance
        else:
            return 'high'   # High risk (blowout potential, rotation flux)

    def save(self, filepath: Union[str, Path]):
        """Save the trained model to a pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'models': self.models,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'interval_scale': self.interval_scale,
            'trained': self.trained,
            'training_samples': self.training_samples,
            'training_date': self.training_date,
            'feature_importances': self.feature_importances,
            'version': '1.1.0',  # Updated for interval scaling
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MinutesPredictor':
        """Load a trained model from a pickle file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        predictor = cls(
            feature_names=state.get('feature_names', MINUTES_FEATURE_NAMES),
            model_params=state.get('model_params'),
            interval_scale=state.get('interval_scale', cls.INTERVAL_SCALE_FACTOR),
        )

        predictor.models = state['models']
        predictor.trained = state.get('trained', True)
        predictor.training_samples = state.get('training_samples', 0)
        predictor.training_date = state.get('training_date')
        predictor.feature_importances = state.get('feature_importances', {})

        return predictor

    def get_feature_importance(self, top_n: int = 20) -> list[tuple]:
        """Get top N most important features."""
        if not self.feature_importances:
            return []

        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:top_n]

    def get_model_info(self) -> dict[str, Any]:
        """Get model metadata and configuration."""
        return {
            'trained': self.trained,
            'training_samples': self.training_samples,
            'training_date': self.training_date,
            'n_features': len(self.feature_names),
            'quantiles': self.QUANTILES,
            'model_type': 'LightGBM' if HAS_LIGHTGBM else 'sklearn',
            'version': '1.0.0',
        }


# Convenience function for quick predictions
def predict_minutes(player_id: int,
                    game_context: dict,
                    model_path: str = 'models/minutes_oracle.pkl') -> dict:
    """
    Quick prediction function.

    Args:
        player_id: Player ID
        game_context: Game context dict
        model_path: Path to trained model

    Returns:
        Minutes distribution dict
    """
    predictor = MinutesPredictor.load(model_path)

    # Generate features
    feature_gen = MinutesFeatureGenerator()
    features = feature_gen.generate_features(
        player_id=player_id,
        team_id=game_context.get('team_id', 0),
        opponent_team_id=game_context.get('opponent_team_id', 0),
        game_date=game_context.get('game_date', ''),
        game_context=game_context,
    )

    result = predictor.predict(features, player_id=player_id)
    return result.to_dict()
