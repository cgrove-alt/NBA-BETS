"""
NBA Betting Model Trainer

Implements machine learning models for NBA betting predictions:
- Logistic Regression for moneyline (win probability)
- Support Vector Machines for spread predictions
- Random Forest for player props
- Parlay probability calculator
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM (optional but recommended)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Run: pip install xgboost")

# Import calibration module for probability calibration
try:
    from calibration import ModelCalibrator, calibrate_moneyline_probability
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError) as e:
    HAS_LIGHTGBM = False
    # OSError can occur if libomp is missing on macOS
    # print("LightGBM not available (may need libomp on macOS)")
    pass

# Model save directory
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


class BaseModelTrainer:
    """Base class for model trainers with common functionality."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_metrics = {}

    def preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Preprocess features with scaling.

        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler

        Returns:
            Scaled numpy array
        """
        # Store feature names
        if fit:
            self.feature_names = list(X.columns)

        # Handle missing values
        X_clean = X.fillna(0)

        # Scale features
        if fit:
            return self.scaler.fit_transform(X_clean)
        return self.scaler.transform(X_clean)

    def save_model(self, filepath: Optional[Path] = None):
        """Save model, scaler, and metadata to disk."""
        if filepath is None:
            filepath = MODEL_DIR / f"{self.model_name}.pkl"

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "model_name": self.model_name,
            "saved_at": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath: Optional[Path] = None):
        """Load model, scaler, and metadata from disk."""
        if filepath is None:
            filepath = MODEL_DIR / f"{self.model_name}.pkl"

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.training_metrics = model_data["training_metrics"]
        self.is_fitted = True

        print(f"Model loaded from {filepath}")
        return self

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        if not self.is_fitted:
            return {}

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_).flatten()
        else:
            return {}

        return dict(zip(self.feature_names, importance))


class PropModelWrapper:
    """
    Wrapper class for player prop models trained with train_complete_balldontlie.py.
    Compatible with the existing app.py model loading system.
    """

    def __init__(self, model=None, scaler=None, feature_names=None, training_metrics=None, prop_type="points"):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.training_metrics = training_metrics or {}
        self.prop_type = prop_type
        self.is_fitted = True
        self.model_name = f"player_{prop_type}"

    def predict(self, features: Dict, prop_line: float = None) -> Dict:
        """Make a prediction - compatible with app.py interface."""
        import numpy as np
        import pandas as pd

        X = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)

        predicted = float(self.model.predict(X_scaled)[0])

        result = {
            "predicted_value": predicted,
            "prop_type": self.prop_type,
        }

        if prop_line is not None:
            result["prop_line"] = prop_line
            result["prediction"] = "over" if predicted > prop_line else "under"
            result["edge"] = predicted - prop_line
            result["confidence"] = abs(predicted - prop_line) / max(prop_line, 1)

        return result


class EnsembleMoneylineWrapper:
    """
    Wrapper class that makes ensemble models compatible with
    the existing app.py model loading system.

    This class mimics the interface of BaseModelTrainer.
    """

    def __init__(self, models=None, model_weights=None, scaler=None, feature_names=None, training_metrics=None):
        self.models = models or {}
        self.model_weights = model_weights or {}
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.training_metrics = training_metrics or {}
        self.is_fitted = True
        self.model_name = "moneyline_ensemble"

    def predict(self, features: Dict) -> Dict:
        """Make a prediction - compatible with app.py interface."""
        import numpy as np
        import pandas as pd

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        X_clean = X.fillna(0)
        X_scaled = self.scaler.transform(X_clean)

        # Ensemble prediction
        probs = np.zeros((1, 2))
        for name, model in self.models.items():
            model_probs = model.predict_proba(X_scaled)
            probs += self.model_weights[name] * model_probs

        home_prob = float(np.clip(probs[0, 1], 0.0, 1.0))
        away_prob = float(np.clip(probs[0, 0], 0.0, 1.0))

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(max(home_prob, away_prob)),
        }


class MoneylineModel(BaseModelTrainer):
    """
    Logistic Regression model for moneyline predictions.

    Predicts probability of home team winning.
    """

    def __init__(self):
        super().__init__("moneyline_logistic_regression")
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )

    def prepare_training_data(self, games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from historical games.

        Args:
            games_data: List of game dictionaries with features and outcomes

        Returns:
            Tuple of (features DataFrame, labels array) - SORTED CHRONOLOGICALLY
        """
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("moneyline_features", {})
            outcome = game.get("home_win", None)

            if features and outcome is not None:
                # Remove non-numeric and identifier fields
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
                }
                features_list.append(numeric_features)
                labels.append(1 if outcome else 0)
                game_dates.append(game.get("game_date", "1900-01-01"))

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        # TimeSeriesSplit REQUIRES chronological ordering to work correctly
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_time_series_cv: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the moneyline prediction model.

        Args:
            X: Feature DataFrame (MUST be sorted by date for time-series CV)
            y: Target labels (1 for home win, 0 for away win)
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            use_time_series_cv: Use time-series walk-forward validation (recommended)

        Returns:
            Dictionary with training metrics
        """
        if use_time_series_cv:
            # TIME-SERIES WALK-FORWARD VALIDATION
            # This is CRITICAL for sports betting - prevents look-ahead bias
            # Data must be sorted chronologically (oldest first)
            n_samples = len(X)
            test_samples = int(n_samples * test_size)

            # Use last test_size% as held-out test set
            X_train = X.iloc[:-test_samples]
            X_test = X.iloc[-test_samples:]
            y_train = y[:-test_samples]
            y_test = y[-test_samples:]

            print(f"\n  Using TIME-SERIES validation (walk-forward)")
            print(f"  Train: games 0-{len(X_train)-1}, Test: games {len(X_train)}-{n_samples-1}")
        else:
            # Legacy random split (NOT recommended for time-series data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            print(f"\n  Using RANDOM split (not recommended for time-series)")

        # Preprocess
        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        # Cross-validation with TimeSeriesSplit for proper evaluation
        if use_time_series_cv:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=tscv)
        else:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)

        # Train on full training set
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Predictions on held-out test set
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        self.training_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "validation_type": "time_series" if use_time_series_cv else "random",
        }

        print(f"\nMoneyline Model Training Results:")
        print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
        print(f"  Precision: {self.training_metrics['precision']:.4f}")
        print(f"  Recall: {self.training_metrics['recall']:.4f}")
        print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        print(f"  CV Score: {self.training_metrics['cv_mean']:.4f} (+/- {self.training_metrics['cv_std']:.4f})")

        return self.training_metrics

    def predict(self, features: Dict) -> Dict[str, float]:
        """
        Predict home team win probability.

        Args:
            features: Moneyline features dictionary

        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        # Prepare features
        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])

        # Ensure all expected features are present
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        # Scale and predict
        X_scaled = self.preprocess_features(X, fit=False)
        prob = self.model.predict_proba(X_scaled)[0]

        # CRITICAL: Ensure probabilities are valid (0.0 to 1.0)
        home_prob = float(np.clip(prob[1], 0.0, 1.0))
        away_prob = float(np.clip(prob[0], 0.0, 1.0))

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(np.clip(max(prob), 0.0, 1.0)),
        }


class SpreadModel(BaseModelTrainer):
    """
    Support Vector Machine model for spread predictions.

    Predicts point differential (positive = home team wins by that margin).
    """

    def __init__(self, use_classifier: bool = False):
        model_name = "spread_svm_classifier" if use_classifier else "spread_svm_regressor"
        super().__init__(model_name)
        self.use_classifier = use_classifier

        if use_classifier:
            # SVC for classifying if spread is covered
            self.model = SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            )
        else:
            # SVR for predicting actual point spread
            self.model = SVR(
                kernel="rbf",
                C=1.0,
                epsilon=0.1,
            )

    def prepare_training_data(
        self,
        games_data: List[Dict],
        spread_line: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from historical games.

        Args:
            games_data: List of game dictionaries with features and outcomes
            spread_line: If using classifier, the spread line to evaluate

        Returns:
            Tuple of (features DataFrame, labels array) - SORTED CHRONOLOGICALLY
        """
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("spread_features", {})
            actual_diff = game.get("point_differential", None)  # home - away

            if features and actual_diff is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in [
                        "home_team_id", "away_team_id", "injury_details"
                    ]
                }
                features_list.append(numeric_features)
                game_dates.append(game.get("game_date", "1900-01-01"))

                if self.use_classifier and spread_line is not None:
                    # 1 if home covers spread, 0 otherwise
                    labels.append(1 if actual_diff > spread_line else 0)
                else:
                    labels.append(actual_diff)

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_time_series_cv: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the spread prediction model.

        Args:
            X: Feature DataFrame (MUST be sorted by date for time-series CV)
            y: Target values (point diff or cover labels)
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            use_time_series_cv: Use time-series walk-forward validation (recommended)

        Returns:
            Dictionary with training metrics
        """
        # Split data using time-series or random approach
        if use_time_series_cv:
            # TIME-SERIES WALK-FORWARD VALIDATION
            n_samples = len(X)
            test_samples = int(n_samples * test_size)
            X_train = X.iloc[:-test_samples]
            X_test = X.iloc[-test_samples:]
            y_train = y[:-test_samples]
            y_test = y[-test_samples:]
            print(f"\n  Using TIME-SERIES validation (walk-forward)")
        elif self.use_classifier:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        # Preprocess
        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        # Cross-validation with TimeSeriesSplit or standard K-fold
        if use_time_series_cv:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=tscv)
        else:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)

        # Train
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Predictions
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        if self.use_classifier:
            self.training_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "validation_type": "time_series" if use_time_series_cv else "random",
            }
            print(f"\nSpread Classifier Training Results:")
            print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
            print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        else:
            self.training_metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "validation_type": "time_series" if use_time_series_cv else "random",
            }
            print(f"\nSpread Regressor Training Results:")
            print(f"  RMSE: {self.training_metrics['rmse']:.2f} points")
            print(f"  MAE: {self.training_metrics['mae']:.2f} points")
            print(f"  R2: {self.training_metrics['r2']:.4f}")

        print(f"  CV Score: {self.training_metrics['cv_mean']:.4f} (+/- {self.training_metrics['cv_std']:.4f})")

        return self.training_metrics

    def predict(self, features: Dict, spread_line: Optional[float] = None) -> Dict[str, Any]:
        """
        Predict spread outcome.

        Args:
            features: Spread features dictionary
            spread_line: The betting line to evaluate

        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in [
                "home_team_id", "away_team_id", "injury_details"
            ]
        }

        X = pd.DataFrame([numeric_features])

        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)

        if self.use_classifier:
            prob = self.model.predict_proba(X_scaled)[0]
            return {
                "cover_probability": float(np.clip(prob[1], 0.0, 1.0)),
                "no_cover_probability": float(np.clip(prob[0], 0.0, 1.0)),
                "prediction": "cover" if prob[1] > 0.5 else "no_cover",
                "confidence": float(np.clip(max(prob), 0.0, 1.0)),
            }
        else:
            predicted_diff = self.model.predict(X_scaled)[0]

            # CRITICAL FIX: Clip spread prediction to realistic NBA range
            # NBA games are never decided by more than ~50 points, typical range is -20 to +20
            predicted_diff = float(np.clip(predicted_diff, -30.0, 30.0))

            result = {
                "predicted_spread": predicted_diff,
                "predicted_winner": "home" if predicted_diff > 0 else "away",
                "predicted_margin": abs(predicted_diff),
            }

            if spread_line is not None:
                result["covers_spread"] = predicted_diff > spread_line
                result["spread_line"] = spread_line
                # Edge is capped to realistic values (-20 to +20)
                result["edge"] = float(np.clip(predicted_diff - spread_line, -20.0, 20.0))

            return result


class PlayerPropModel(BaseModelTrainer):
    """
    Random Forest model for player prop predictions.

    Predicts various player statistics (points, rebounds, assists, etc.).
    """

    def __init__(self, prop_type: str = "points", use_classifier: bool = False):
        """
        Initialize player prop model.

        Args:
            prop_type: Type of prop ("points", "rebounds", "assists", "threes", "pra")
            use_classifier: If True, classifies over/under; if False, predicts value
        """
        self.prop_type = prop_type
        self.use_classifier = use_classifier

        model_name = f"player_{prop_type}_{'classifier' if use_classifier else 'regressor'}"
        super().__init__(model_name)

        if use_classifier:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )

    def prepare_training_data(
        self,
        player_data: List[Dict],
        prop_line: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from historical player games.

        Args:
            player_data: List of player game dictionaries with features and outcomes
            prop_line: If using classifier, the prop line to evaluate

        Returns:
            Tuple of (features DataFrame, labels array)
        """
        # Map prop type to feature key
        prop_feature_map = {
            "points": "points_features",
            "rebounds": "rebounds_features",
            "assists": "assists_features",
            "threes": "threes_features",
            "pra": "pra_features",
        }

        # Map prop type to actual stat key
        stat_key_map = {
            "points": "pts",
            "rebounds": "reb",
            "assists": "ast",
            "threes": "fg3_made",
            "pra": "pra",  # pts + reb + ast
        }

        feature_key = prop_feature_map.get(self.prop_type, "points_features")
        stat_key = stat_key_map.get(self.prop_type, "pts")

        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in player_data:
            features = game.get(feature_key, {})
            actual_value = game.get("actual_stats", {}).get(stat_key, None)

            # Handle PRA calculation
            if self.prop_type == "pra" and actual_value is None:
                stats = game.get("actual_stats", {})
                pts = stats.get("pts", 0) or 0
                reb = stats.get("reb", 0) or 0
                ast = stats.get("ast", 0) or 0
                actual_value = pts + reb + ast

            if features and actual_value is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k != "player_id"
                }
                features_list.append(numeric_features)
                game_dates.append(game.get("game_date", "1900-01-01"))

                if self.use_classifier and prop_line is not None:
                    labels.append(1 if actual_value > prop_line else 0)
                else:
                    labels.append(actual_value)

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
        tune_hyperparameters: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the player prop model.

        Args:
            X: Feature DataFrame
            y: Target values
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            tune_hyperparameters: Whether to perform grid search

        Returns:
            Dictionary with training metrics
        """
        # Split data
        if self.use_classifier:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        # Preprocess
        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        # Hyperparameter tuning
        if tune_hyperparameters:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            }
            grid_search = GridSearchCV(
                self.model, param_grid, cv=cv_folds, scoring="neg_mean_squared_error" if not self.use_classifier else "accuracy",
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)

            # Train
            self.model.fit(X_train_scaled, y_train)

        self.is_fitted = True

        # Predictions
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        if self.use_classifier:
            self.training_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "cv_mean": cv_scores.mean() if not tune_hyperparameters else 0,
                "cv_std": cv_scores.std() if not tune_hyperparameters else 0,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
            print(f"\n{self.prop_type.title()} Prop Classifier Training Results:")
            print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
            print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        else:
            self.training_metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "cv_mean": cv_scores.mean() if not tune_hyperparameters else 0,
                "cv_std": cv_scores.std() if not tune_hyperparameters else 0,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
            print(f"\n{self.prop_type.title()} Prop Regressor Training Results:")
            print(f"  RMSE: {self.training_metrics['rmse']:.2f}")
            print(f"  MAE: {self.training_metrics['mae']:.2f}")
            print(f"  R2: {self.training_metrics['r2']:.4f}")

        return self.training_metrics

    def predict(self, features: Dict, prop_line: Optional[float] = None) -> Dict[str, Any]:
        """
        Predict player prop outcome.

        Args:
            features: Player prop features dictionary
            prop_line: The betting line to evaluate

        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k != "player_id"
        }

        X = pd.DataFrame([numeric_features])

        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)

        if self.use_classifier:
            prob = self.model.predict_proba(X_scaled)[0]
            return {
                "over_probability": prob[1],
                "under_probability": prob[0],
                "prediction": "over" if prob[1] > 0.5 else "under",
                "confidence": max(prob),
                "prop_type": self.prop_type,
            }
        else:
            predicted_value = self.model.predict(X_scaled)[0]
            result = {
                "predicted_value": predicted_value,
                "prop_type": self.prop_type,
            }

            if prop_line is not None:
                result["prop_line"] = prop_line
                result["prediction"] = "over" if predicted_value > prop_line else "under"
                result["edge"] = predicted_value - prop_line

            return result


class XGBoostMoneylineModel(BaseModelTrainer):
    """
    XGBoost model for moneyline predictions.

    XGBoost typically outperforms logistic regression for complex patterns.
    Requires: pip install xgboost
    """

    def __init__(self):
        super().__init__("moneyline_xgboost")
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
        )

    def prepare_training_data(self, games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data (same as MoneylineModel) - SORTED CHRONOLOGICALLY."""
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("moneyline_features", {})
            outcome = game.get("home_win", None)

            if features and outcome is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
                }
                features_list.append(numeric_features)
                labels.append(1 if outcome else 0)
                game_dates.append(game.get("game_date", "1900-01-01"))

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_time_series_cv: bool = True,
    ) -> Dict[str, Any]:
        """Train the XGBoost moneyline model with time-series validation."""
        if use_time_series_cv:
            # TIME-SERIES WALK-FORWARD VALIDATION
            n_samples = len(X)
            test_samples = int(n_samples * test_size)
            X_train = X.iloc[:-test_samples]
            X_test = X.iloc[-test_samples:]
            y_train = y[:-test_samples]
            y_test = y[-test_samples:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        self.training_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        print(f"\nXGBoost Moneyline Model Training Results:")
        print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        print(f"  CV Score: {self.training_metrics['cv_mean']:.4f} (+/- {self.training_metrics['cv_std']:.4f})")

        return self.training_metrics

    def predict(self, features: Dict, calibrate: bool = True) -> Dict[str, float]:
        """
        Predict home team win probability.

        Args:
            features: Feature dictionary for the matchup
            calibrate: Whether to apply probability calibration (default: True)
                       Calibration improves betting accuracy by ensuring
                       predicted probabilities match actual win rates.

        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)
        prob = self.model.predict_proba(X_scaled)[0]

        home_prob = float(np.clip(prob[1], 0.0, 1.0))
        away_prob = float(np.clip(prob[0], 0.0, 1.0))

        # Apply probability calibration if available and requested
        if calibrate and HAS_CALIBRATION:
            try:
                home_prob = calibrate_moneyline_probability(home_prob)
                away_prob = 1.0 - home_prob  # Ensure probabilities sum to 1
            except Exception:
                pass  # Use uncalibrated if calibration fails

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(np.clip(max(home_prob, away_prob), 0.0, 1.0)),
            "is_calibrated": calibrate and HAS_CALIBRATION,
        }


class LightGBMSpreadModel(BaseModelTrainer):
    """
    LightGBM model for spread predictions.

    LightGBM is faster and often more accurate than traditional methods.
    Requires: pip install lightgbm
    """

    def __init__(self):
        super().__init__("spread_lightgbm")
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

    def prepare_training_data(self, games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data for spread prediction - SORTED CHRONOLOGICALLY."""
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("spread_features", {})
            actual_diff = game.get("point_differential", None)

            if features and actual_diff is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id", "injury_details"]
                }
                features_list.append(numeric_features)
                labels.append(actual_diff)
                game_dates.append(game.get("game_date", "1900-01-01"))

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train the LightGBM spread model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds,
                                     scoring='neg_mean_squared_error')

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test_scaled)

        self.training_metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "cv_mean_rmse": np.sqrt(-cv_scores.mean()),
            "cv_std_rmse": np.sqrt(cv_scores.std()),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        print(f"\nLightGBM Spread Model Training Results:")
        print(f"  RMSE: {self.training_metrics['rmse']:.2f} points")
        print(f"  MAE: {self.training_metrics['mae']:.2f} points")
        print(f"  R2: {self.training_metrics['r2']:.4f}")
        print(f"  CV RMSE: {self.training_metrics['cv_mean_rmse']:.2f}")

        return self.training_metrics

    def predict(self, features: Dict, spread_line: Optional[float] = None) -> Dict[str, Any]:
        """Predict spread outcome."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id", "injury_details"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)
        predicted_diff = self.model.predict(X_scaled)[0]

        # Clip to realistic NBA range
        predicted_diff = float(np.clip(predicted_diff, -30.0, 30.0))

        result = {
            "predicted_spread": predicted_diff,
            "predicted_winner": "home" if predicted_diff > 0 else "away",
            "predicted_margin": abs(predicted_diff),
        }

        if spread_line is not None:
            result["covers_spread"] = predicted_diff > spread_line
            result["spread_line"] = spread_line
            result["edge"] = float(np.clip(predicted_diff - spread_line, -20.0, 20.0))

        return result


class EnsembleMoneylineModel(BaseModelTrainer):
    """
    Ensemble model combining multiple classifiers for moneyline predictions.

    Uses stacking with Logistic Regression, Random Forest, and Gradient Boosting.
    Optional: XGBoost and LightGBM if installed.
    """

    def __init__(self):
        super().__init__("moneyline_ensemble")

        # Base estimators
        estimators = [
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ]

        if HAS_XGBOOST:
            estimators.append(('xgb', xgb.XGBClassifier(
                n_estimators=100, max_depth=6, random_state=42,
                use_label_encoder=False, eval_metric='logloss'
            )))

        if HAS_LIGHTGBM:
            estimators.append(('lgb', lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, random_state=42, verbose=-1
            )))

        # Stacking classifier with logistic regression as final estimator
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
            n_jobs=-1,
        )

    def prepare_training_data(self, games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data - SORTED CHRONOLOGICALLY."""
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("moneyline_features", {})
            outcome = game.get("home_win", None)

            if features and outcome is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
                }
                features_list.append(numeric_features)
                labels.append(1 if outcome else 0)
                game_dates.append(game.get("game_date", "1900-01-01"))

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_time_series_cv: bool = True,
    ) -> Dict[str, Any]:
        """Train the ensemble model with walk-forward validation."""
        if use_time_series_cv:
            # TIME-SERIES WALK-FORWARD VALIDATION
            n_samples = len(X)
            test_samples = int(n_samples * test_size)
            X_train = X.iloc[:-test_samples]
            X_test = X.iloc[-test_samples:]
            y_train = y[:-test_samples]
            y_test = y[-test_samples:]
            print(f"\n  Using TIME-SERIES validation (walk-forward)")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        print("Training ensemble model (this may take a few minutes)...")
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        self.training_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "num_base_estimators": len(self.model.estimators_),
            "validation_type": "time_series" if use_time_series_cv else "random",
        }

        print(f"\nEnsemble Moneyline Model Training Results:")
        print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        print(f"  Base Estimators: {self.training_metrics['num_base_estimators']}")

        return self.training_metrics

    def predict(self, features: Dict) -> Dict[str, float]:
        """Predict home team win probability."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)
        prob = self.model.predict_proba(X_scaled)[0]

        home_prob = float(np.clip(prob[1], 0.0, 1.0))
        away_prob = float(np.clip(prob[0], 0.0, 1.0))

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(np.clip(max(prob), 0.0, 1.0)),
        }


class TunedEnsembleMoneylineModel(BaseModelTrainer):
    """
    Ensemble model with hyperparameter tuning for moneyline predictions.

    Uses RandomizedSearchCV to find optimal hyperparameters for each base model
    before combining them in a stacking ensemble.
    """

    # Hyperparameter search spaces for each model type
    PARAM_GRIDS = {
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        'gb': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'min_samples_split': [2, 5, 10],
        },
        'xgb': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9, 1.0],
        },
        'lgb': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [15, 31, 63],
        },
    }

    def __init__(self, n_iter: int = 20, cv_folds: int = 3):
        """
        Initialize the tuned ensemble model.

        Args:
            n_iter: Number of parameter combinations to try in RandomizedSearchCV
            cv_folds: Number of cross-validation folds for tuning
        """
        super().__init__("moneyline_ensemble_tuned")
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.best_params = {}
        self.model = None

    def _tune_model(self, model, param_grid, X_train, y_train, model_name: str):
        """Tune a single model using RandomizedSearchCV with TimeSeriesSplit."""
        print(f"    Tuning {model_name}...")

        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=min(self.n_iter, 10),  # Limit iterations per model
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)

        print(f"      Best {model_name} score: {search.best_score_:.4f}")
        print(f"      Best params: {search.best_params_}")

        self.best_params[model_name] = search.best_params_
        return search.best_estimator_

    def prepare_training_data(self, games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data - SORTED CHRONOLOGICALLY for time-series validation."""
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("moneyline_features", {})
            outcome = game.get("home_win", None)

            if features and outcome is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
                }
                features_list.append(numeric_features)
                labels.append(1 if outcome else 0)
                game_dates.append(game.get("game_date", "1900-01-01"))

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        tune_hyperparameters: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the tuned ensemble model.

        Args:
            X: Feature DataFrame (MUST be sorted by date)
            y: Target labels
            test_size: Proportion for test set
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Dictionary with training metrics
        """
        # TIME-SERIES SPLIT
        n_samples = len(X)
        test_samples = int(n_samples * test_size)
        X_train = X.iloc[:-test_samples]
        X_test = X.iloc[-test_samples:]
        y_train = y[:-test_samples]
        y_test = y[-test_samples:]

        print(f"\n  Training TunedEnsembleMoneylineModel")
        print(f"  Train: {len(X_train)} games, Test: {len(X_test)} games")

        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        # Build estimators (tuned if requested)
        estimators = []

        # Logistic Regression (minimal tuning needed)
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        estimators.append(('lr', lr))

        if tune_hyperparameters:
            print("  Tuning hyperparameters (this may take a while)...")

            # Tune Random Forest
            rf = RandomForestClassifier(random_state=42)
            tuned_rf = self._tune_model(rf, self.PARAM_GRIDS['rf'], X_train_scaled, y_train, 'rf')
            estimators.append(('rf', tuned_rf))

            # Tune Gradient Boosting
            gb = GradientBoostingClassifier(random_state=42)
            tuned_gb = self._tune_model(gb, self.PARAM_GRIDS['gb'], X_train_scaled, y_train, 'gb')
            estimators.append(('gb', tuned_gb))

            # Tune XGBoost if available
            if HAS_XGBOOST:
                xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
                tuned_xgb = self._tune_model(xgb_model, self.PARAM_GRIDS['xgb'], X_train_scaled, y_train, 'xgb')
                estimators.append(('xgb', tuned_xgb))

            # Tune LightGBM if available
            if HAS_LIGHTGBM:
                lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                tuned_lgb = self._tune_model(lgb_model, self.PARAM_GRIDS['lgb'], X_train_scaled, y_train, 'lgb')
                estimators.append(('lgb', tuned_lgb))
        else:
            # Use default parameters
            estimators.append(('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)))
            estimators.append(('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)))
            if HAS_XGBOOST:
                estimators.append(('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')))
            if HAS_LIGHTGBM:
                estimators.append(('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)))

        # Create stacking classifier with XGBoost meta-learner (better than LR)
        if HAS_XGBOOST:
            final_estimator = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
        else:
            final_estimator = LogisticRegression(max_iter=1000, C=0.1)

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=TimeSeriesSplit(n_splits=3),  # Time-series CV for stacking
            n_jobs=-1,
        )

        print("  Training final stacked ensemble...")
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        self.training_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "num_base_estimators": len(estimators),
            "tuned": tune_hyperparameters,
            "best_params": self.best_params,
        }

        print(f"\n  Tuned Ensemble Training Results:")
        print(f"    Accuracy: {self.training_metrics['accuracy']:.4f}")
        print(f"    F1 Score: {self.training_metrics['f1']:.4f}")
        print(f"    Base Estimators: {len(estimators)}")

        return self.training_metrics

    def predict(self, features: Dict) -> Dict[str, float]:
        """Predict home team win probability."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)
        prob = self.model.predict_proba(X_scaled)[0]

        home_prob = float(np.clip(prob[1], 0.0, 1.0))
        away_prob = float(np.clip(prob[0], 0.0, 1.0))

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(np.clip(max(prob), 0.0, 1.0)),
        }


class TotalsModel(BaseModelTrainer):
    """
    Model for predicting game totals (over/under).

    Predicts total combined points and evaluates against betting lines.
    """

    def __init__(self, use_classifier: bool = False):
        model_name = "totals_classifier" if use_classifier else "totals_regressor"
        super().__init__(model_name)
        self.use_classifier = use_classifier

        if use_classifier:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
            )
        else:
            if HAS_LIGHTGBM:
                self.model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1,
                )
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                )

    def prepare_training_data(
        self,
        games_data: List[Dict],
        total_line: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data for totals prediction - SORTED CHRONOLOGICALLY.

        Args:
            games_data: List of game dictionaries with features and outcomes
            total_line: If using classifier, the total line to evaluate

        Returns:
            Tuple of (features DataFrame, labels array)
        """
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            # Get spread features and add pace-related features
            features = game.get("spread_features", {}).copy()

            # Add totals-specific features if available
            totals_features = game.get("totals_features", {})
            features.update(totals_features)

            home_score = game.get("home_score", None)
            away_score = game.get("away_score", None)

            if features and home_score is not None and away_score is not None:
                total_points = home_score + away_score

                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
                }
                features_list.append(numeric_features)
                game_dates.append(game.get("game_date", "1900-01-01"))

                if self.use_classifier and total_line is not None:
                    labels.append(1 if total_points > total_line else 0)
                else:
                    labels.append(total_points)

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train the totals model."""
        if self.use_classifier:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test_scaled)

        if self.use_classifier:
            self.training_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
            print(f"\nTotals Classifier Training Results:")
            print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
            print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        else:
            self.training_metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
            print(f"\nTotals Regressor Training Results:")
            print(f"  RMSE: {self.training_metrics['rmse']:.2f} points")
            print(f"  MAE: {self.training_metrics['mae']:.2f} points")
            print(f"  R2: {self.training_metrics['r2']:.4f}")

        print(f"  CV Score: {self.training_metrics['cv_mean']:.4f} (+/- {self.training_metrics['cv_std']:.4f})")
        return self.training_metrics

    def predict(self, features: Dict, total_line: Optional[float] = None) -> Dict[str, Any]:
        """
        Predict total points.

        Args:
            features: Game features dictionary
            total_line: The betting line to evaluate

        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)

        if self.use_classifier:
            prob = self.model.predict_proba(X_scaled)[0]
            return {
                "over_probability": float(np.clip(prob[1], 0.0, 1.0)),
                "under_probability": float(np.clip(prob[0], 0.0, 1.0)),
                "prediction": "over" if prob[1] > 0.5 else "under",
                "confidence": float(np.clip(max(prob), 0.0, 1.0)),
            }
        else:
            predicted_total = self.model.predict(X_scaled)[0]

            # Clip to realistic NBA total range (160-280)
            predicted_total = float(np.clip(predicted_total, 160.0, 280.0))

            result = {
                "predicted_total": predicted_total,
            }

            if total_line is not None:
                result["total_line"] = total_line
                result["prediction"] = "over" if predicted_total > total_line else "under"
                result["edge"] = float(np.clip(predicted_total - total_line, -30.0, 30.0))

                # Convert edge to probability estimate
                edge_in_points = predicted_total - total_line
                # Roughly: 10 points edge = ~70% probability
                prob = 1 / (1 + np.exp(-edge_in_points / 10.0))
                result["over_probability"] = float(np.clip(prob, 0.0, 1.0))
                result["under_probability"] = float(np.clip(1.0 - prob, 0.0, 1.0))

            return result


class ParlayCalculator:
    """
    Calculator for parlay probabilities and expected values.

    Combines multiple individual bet predictions into parlay analysis.
    """

    def __init__(self):
        self.models = {}

    def add_model(self, model_type: str, model: BaseModelTrainer):
        """Add a trained model for parlay calculations."""
        self.models[model_type] = model

    def calculate_parlay_probability(self, legs: List[Dict]) -> Dict[str, Any]:
        """
        Calculate combined probability for a parlay.

        Args:
            legs: List of parlay legs, each with:
                - type: "moneyline", "spread", or "prop"
                - features: Feature dictionary for prediction
                - selection: "home", "away", "over", "under", "cover"
                - line: Betting line (for spread/props)

        Returns:
            Dictionary with parlay analysis
        """
        individual_probs = []
        leg_details = []

        for leg in legs:
            leg_type = leg.get("type")
            features = leg.get("features", {})
            selection = leg.get("selection")
            line = leg.get("line")

            if leg_type == "moneyline" and "moneyline" in self.models:
                pred = self.models["moneyline"].predict(features)
                if selection == "home":
                    prob = pred["home_win_probability"]
                else:
                    prob = pred["away_win_probability"]

            elif leg_type == "spread" and "spread" in self.models:
                pred = self.models["spread"].predict(features, spread_line=line)
                if self.models["spread"].use_classifier:
                    prob = pred["cover_probability"] if selection == "cover" else pred["no_cover_probability"]
                else:
                    # Convert regression to probability estimate
                    edge = pred.get("edge", 0)
                    prob = self._edge_to_probability(edge)

            elif leg_type == "prop":
                prop_type = leg.get("prop_type", "points")
                model_key = f"prop_{prop_type}"
                if model_key in self.models:
                    pred = self.models[model_key].predict(features, prop_line=line)
                    if self.models[model_key].use_classifier:
                        prob = pred["over_probability"] if selection == "over" else pred["under_probability"]
                    else:
                        edge = pred.get("edge", 0)
                        prob = self._edge_to_probability(edge) if selection == "over" else self._edge_to_probability(-edge)
                else:
                    prob = 0.5  # Default if model not available

            else:
                prob = 0.5  # Default probability

            individual_probs.append(prob)
            leg_details.append({
                "type": leg_type,
                "selection": selection,
                "probability": prob,
                "line": line,
            })

        # Calculate combined probability (independent events)
        # Ensure all individual probs are valid before multiplication
        valid_probs = [float(np.clip(p, 0.0, 1.0)) for p in individual_probs]
        combined_prob = float(np.prod(valid_probs))
        # Combined probability must be between 0 and 1
        combined_prob = float(np.clip(combined_prob, 0.0, 1.0))

        return {
            "combined_probability": combined_prob,
            "individual_legs": leg_details,
            "num_legs": len(legs),
            "implied_odds": self._probability_to_american_odds(combined_prob),
        }

    def calculate_expected_value(
        self,
        parlay_prob: float,
        odds: float,
        stake: float = 100,
    ) -> Dict[str, float]:
        """
        Calculate expected value for a parlay.

        Args:
            parlay_prob: Combined probability of winning
            odds: American odds offered
            stake: Bet amount

        Returns:
            Dictionary with EV analysis
        """
        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

        # Calculate potential profit and EV
        potential_profit = stake * (decimal_odds - 1)
        ev = (parlay_prob * potential_profit) - ((1 - parlay_prob) * stake)
        ev_percentage = (ev / stake) * 100

        # Calculate implied probability from odds
        implied_prob = 1 / decimal_odds

        return {
            "expected_value": ev,
            "ev_percentage": ev_percentage,
            "potential_profit": potential_profit,
            "stake": stake,
            "model_probability": parlay_prob,
            "implied_probability": implied_prob,
            "edge": parlay_prob - implied_prob,
            "recommendation": "bet" if ev > 0 else "pass",
        }

    def _edge_to_probability(self, edge: float, scale: float = 5.0) -> float:
        """
        Convert point edge to probability using sigmoid.

        Args:
            edge: Point edge (positive = favorable)
            scale: Scaling factor (default 5.0 means ~73% prob for 5-point edge)

        Returns:
            Probability between 0.0 and 1.0
        """
        # Clip edge to prevent overflow in exp
        edge = float(np.clip(edge, -50.0, 50.0))
        prob = 1 / (1 + np.exp(-edge / scale))
        # Ensure output is valid probability
        return float(np.clip(prob, 0.0, 1.0))

    def _probability_to_american_odds(self, prob: float) -> float:
        """Convert probability to American odds."""
        if prob <= 0:
            return 0
        if prob >= 1:
            return -10000

        if prob >= 0.5:
            return -100 * prob / (1 - prob)
        else:
            return 100 * (1 - prob) / prob


class ModelTrainingPipeline:
    """
    Complete training pipeline for all NBA betting models.
    """

    def __init__(self, season: str = "2025-26"):
        self.season = season
        self.models = {}

    def train_all_models(
        self,
        games_data: List[Dict],
        player_data: Optional[List[Dict]] = None,
        save_models: bool = True,
        use_ensemble: bool = True,
        use_tuned_ensemble: bool = True,
    ) -> Dict[str, Dict]:
        """
        Train all models with provided data.

        Args:
            games_data: Historical game data with features and outcomes
            player_data: Historical player data with features and outcomes
            save_models: Whether to save models after training
            use_ensemble: If True, use Ensemble model for maximum accuracy (default: True)
            use_tuned_ensemble: If True with use_ensemble, use TunedEnsembleMoneylineModel
                               with GridSearchCV hyperparameter optimization (default: True)

        Returns:
            Dictionary with all training metrics
        """
        results = {}

        # Train Moneyline Model - Use Tuned Ensemble for best accuracy
        print("\n" + "=" * 50)
        if use_ensemble and use_tuned_ensemble and (HAS_XGBOOST or HAS_LIGHTGBM):
            print("Training Moneyline Model (TUNED ENSEMBLE - OPTIMIZED HYPERPARAMETERS)")
            print("  Components: LR + RF + GradientBoosting + XGBoost + LightGBM")
            print("  Using GridSearchCV for hyperparameter optimization")
            print("=" * 50)
            moneyline_model = TunedEnsembleMoneylineModel(n_iter=30, cv_folds=5)
        elif use_ensemble and (HAS_XGBOOST or HAS_LIGHTGBM):
            print("Training Moneyline Model (ENSEMBLE - MAXIMUM ACCURACY)")
            print("  Components: LR + RF + GradientBoosting + XGBoost + LightGBM")
            print("=" * 50)
            moneyline_model = EnsembleMoneylineModel()
        elif HAS_XGBOOST:
            print("Training Moneyline Model (XGBoost)")
            print("=" * 50)
            moneyline_model = XGBoostMoneylineModel()
        else:
            print("Training Moneyline Model (Logistic Regression)")
            print("=" * 50)
            moneyline_model = MoneylineModel()
        X_ml, y_ml = moneyline_model.prepare_training_data(games_data)
        if len(X_ml) > 0:
            results["moneyline"] = moneyline_model.train(X_ml, y_ml)
            self.models["moneyline"] = moneyline_model
            if save_models:
                moneyline_model.save_model()

        # Train Spread Model (Regressor)
        print("\n" + "=" * 50)
        print("Training Spread Model (SVM Regressor)")
        print("=" * 50)
        spread_model = SpreadModel(use_classifier=False)
        X_sp, y_sp = spread_model.prepare_training_data(games_data)
        if len(X_sp) > 0:
            results["spread"] = spread_model.train(X_sp, y_sp)
            self.models["spread"] = spread_model
            if save_models:
                spread_model.save_model()

        # Train Player Prop Models
        if player_data:
            prop_types = ["points", "rebounds", "assists", "threes", "pra"]
            for prop_type in prop_types:
                print("\n" + "=" * 50)
                print(f"Training {prop_type.title()} Prop Model (Random Forest)")
                print("=" * 50)

                prop_model = PlayerPropModel(prop_type=prop_type, use_classifier=False)
                X_prop, y_prop = prop_model.prepare_training_data(player_data)

                if len(X_prop) > 0:
                    results[f"prop_{prop_type}"] = prop_model.train(X_prop, y_prop)
                    self.models[f"prop_{prop_type}"] = prop_model
                    if save_models:
                        prop_model.save_model()

        return results

    def load_all_models(self) -> Dict[str, BaseModelTrainer]:
        """Load all saved models."""
        model_files = list(MODEL_DIR.glob("*.pkl"))

        for filepath in model_files:
            model_name = filepath.stem

            try:
                # First, try direct loading to check for ensemble/wrapper or prop models
                with open(filepath, "rb") as f:
                    model_data = pickle.load(f)

                # Check if this is a prop model FIRST (before ensemble check)
                # Prop models have 'prop_type' key OR are named 'player_*'
                if "prop_type" in model_data or model_name.startswith("player_"):
                    prop_wrapper = PropModelWrapper(
                        model=model_data.get("model"),
                        scaler=model_data.get("scaler"),
                        feature_names=model_data.get("feature_names", []),
                        training_metrics=model_data.get("training_metrics", {}),
                        prop_type=model_data.get("prop_type", model_name.replace("player_", "")),
                    )
                    self.models[model_name] = prop_wrapper
                    continue

                # Check if this is an ensemble wrapper model (has predict method on the 'model' key)
                model_obj = model_data.get("model")
                if model_obj is not None and hasattr(model_obj, "predict") and hasattr(model_obj, "models"):
                    # This is our EnsembleMoneylineWrapper
                    wrapper = model_obj
                    wrapper.scaler = model_data.get("scaler", wrapper.scaler if hasattr(wrapper, 'scaler') else None)
                    wrapper.feature_names = model_data.get("feature_names", wrapper.feature_names if hasattr(wrapper, 'feature_names') else [])
                    wrapper.training_metrics = model_data.get("training_metrics", {})
                    wrapper.is_fitted = True
                    self.models[model_name] = wrapper
                    continue

            except Exception:
                pass  # Fall through to legacy loading

            if "moneyline" in model_name:
                # Try XGBoost first if available, fall back to Logistic Regression
                if "xgboost" in model_name.lower() and HAS_XGBOOST:
                    model = XGBoostMoneylineModel()
                elif HAS_XGBOOST:
                    # For generic moneyline models, try XGBoost
                    try:
                        model = XGBoostMoneylineModel()
                    except Exception:
                        model = MoneylineModel()
                else:
                    model = MoneylineModel()
            elif "spread" in model_name:
                use_classifier = "classifier" in model_name
                model = SpreadModel(use_classifier=use_classifier)
            elif "player_" in model_name:
                # Extract prop type
                parts = model_name.split("_")
                prop_type = parts[1] if len(parts) > 1 else "points"
                use_classifier = "classifier" in model_name
                model = PlayerPropModel(prop_type=prop_type, use_classifier=use_classifier)
            else:
                continue

            try:
                model.load_model(filepath)
                self.models[model_name] = model
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        return self.models

    def get_parlay_calculator(self) -> ParlayCalculator:
        """Get parlay calculator with loaded models."""
        calculator = ParlayCalculator()
        for model_name, model in self.models.items():
            calculator.add_model(model_name, model)
        return calculator


def create_sample_training_data() -> Tuple[List[Dict], List[Dict]]:
    """
    Create sample training data structure for demonstration.

    In production, this would be replaced with actual historical data.

    Returns:
        Tuple of (games_data, player_data)
    """
    # Sample structure for games data
    games_data_sample = {
        "moneyline_features": {
            "season_win_pct_diff": 0.1,
            "recent_win_pct_diff": 0.2,
            "net_rating_diff": 3.5,
            "off_rating_diff": 2.0,
            "def_rating_diff": -1.5,
            "home_streak": 3,
            "away_streak": -2,
            "h2h_home_win_pct": 0.6,
        },
        "spread_features": {
            "season_win_pct_diff": 0.1,
            "expected_point_diff": 5.0,
            "plus_minus_diff": 4.2,
            "net_rating_diff": 3.5,
        },
        "home_win": True,
        "point_differential": 8,
    }

    # Sample structure for player data
    player_data_sample = {
        "points_features": {
            "season_pts_avg": 25.3,
            "recent_pts_avg": 28.1,
            "pts_trend": 2.8,
            "opp_def_rating": 112.5,
        },
        "actual_stats": {
            "pts": 27,
            "reb": 8,
            "ast": 6,
        },
    }

    print("\nSample training data structure created.")
    print("In production, replace with actual historical data.")

    return [games_data_sample], [player_data_sample]


if __name__ == "__main__":
    print("NBA Betting Model Trainer")
    print("=" * 50)
    print("\nUsage:")
    print("""
# Initialize pipeline
pipeline = ModelTrainingPipeline(season="2025-26")

# Train all models with historical data
# games_data: List of dicts with moneyline_features, spread_features, outcomes
# player_data: List of dicts with prop features and actual stats
results = pipeline.train_all_models(games_data, player_data)

# Load saved models
models = pipeline.load_all_models()

# Make predictions
moneyline_pred = models["moneyline"].predict(features)
spread_pred = models["spread"].predict(features, spread_line=-3.5)

# Calculate parlay probabilities
calculator = pipeline.get_parlay_calculator()
parlay = calculator.calculate_parlay_probability([
    {"type": "moneyline", "features": ml_features, "selection": "home"},
    {"type": "spread", "features": sp_features, "selection": "cover", "line": -3.5},
])
ev = calculator.calculate_expected_value(parlay["combined_probability"], odds=+250)
""")

    # Create sample data structure
    games_sample, player_sample = create_sample_training_data()
    print("\nSample games data structure:")
    print(json.dumps(games_sample[0], indent=2, default=str))
