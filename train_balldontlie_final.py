"""
NBA Model Training with Balldontlie.io Data - Standalone Version

This standalone script trains models using pure sklearn classifiers
to avoid XGBoost/sklearn version compatibility issues.

Usage:
    python3 train_balldontlie_final.py
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings('ignore')

# Model save directory
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load environment
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key not in os.environ:
                        os.environ[key] = value

_load_env()


def load_training_data(cache_file: str) -> List[Dict]:
    """Load cached training data."""
    if not Path(cache_file).exists():
        print(f"Error: Cache file not found: {cache_file}")
        print("Run train_with_balldontlie.py first to collect data.")
        sys.exit(1)

    with open(cache_file) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} training examples from {cache_file}")
    return data


def prepare_moneyline_data(games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare data for moneyline model training."""
    features_list = []
    labels = []

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

    X = pd.DataFrame(features_list)
    y = np.array(labels)
    return X, y


def prepare_spread_data(games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare data for spread model training."""
    features_list = []
    labels = []

    for game in games_data:
        features = game.get("spread_features", {})
        diff = game.get("point_differential", None)

        if features and diff is not None:
            numeric_features = {
                k: v for k, v in features.items()
                if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
            }
            features_list.append(numeric_features)
            labels.append(diff)

    X = pd.DataFrame(features_list)
    y = np.array(labels)
    return X, y


class SimpleMoneylineModel:
    """Simple moneyline model using Gradient Boosting."""

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_metrics = {}

    def train(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """Train the model."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Store feature names
        self.feature_names = list(X.columns)

        # Clean and scale
        X_train_clean = X_train.fillna(0)
        X_test_clean = X_test.fillna(0)

        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        X_test_scaled = self.scaler.transform(X_test_clean)

        # Train
        print("  Training Gradient Boosting Classifier...")
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        self.training_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        return self.training_metrics

    def predict(self, features: Dict) -> Dict:
        """Make a prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

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

        prob = self.model.predict_proba(X_scaled)[0]
        home_prob = float(np.clip(prob[1], 0.0, 1.0))
        away_prob = float(np.clip(prob[0], 0.0, 1.0))

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(max(prob)),
        }

    def save(self, filepath: Path):
        """Save model to disk."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "model_name": "moneyline_gradient_boosting",
            "saved_at": datetime.now().isoformat(),
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved to {filepath}")


class SimpleSpreadModel:
    """Simple spread model using Gradient Boosting Regressor."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_metrics = {}

    def train(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """Train the model."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Store feature names
        self.feature_names = list(X.columns)

        # Clean and scale
        X_train_clean = X_train.fillna(0)
        X_test_clean = X_test.fillna(0)

        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        X_test_scaled = self.scaler.transform(X_test_clean)

        # Train
        print("  Training Gradient Boosting Regressor...")
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        self.training_metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        return self.training_metrics

    def predict(self, features: Dict, spread_line: float = None) -> Dict:
        """Make a prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

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

        predicted_diff = float(np.clip(self.model.predict(X_scaled)[0], -30.0, 30.0))

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

    def save(self, filepath: Path):
        """Save model to disk."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "model_name": "spread_gradient_boosting",
            "saved_at": datetime.now().isoformat(),
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved to {filepath}")


# Import the wrapper from model_trainer so pickle can find it
from model_trainer import EnsembleMoneylineWrapper


class EnsembleMoneylineModel:
    """Ensemble model combining multiple classifiers."""

    def __init__(self):
        self.models = {
            'lr': LogisticRegression(max_iter=1000, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        }
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_metrics = {}
        self.model_weights = {'lr': 0.2, 'rf': 0.4, 'gb': 0.4}  # Weighted average

    def train(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """Train all models in the ensemble."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Store feature names
        self.feature_names = list(X.columns)

        # Clean and scale
        X_train_clean = X_train.fillna(0)
        X_test_clean = X_test.fillna(0)

        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        X_test_scaled = self.scaler.transform(X_test_clean)

        # Train each model
        model_accuracies = {}
        for name, model in self.models.items():
            print(f"  Training {name.upper()}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            model_accuracies[name] = acc
            print(f"    {name.upper()} Accuracy: {acc:.4f}")

        self.is_fitted = True

        # Ensemble prediction
        y_pred_ensemble = self._ensemble_predict(X_test_scaled)

        self.training_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_ensemble),
            "precision": precision_score(y_test, y_pred_ensemble),
            "recall": recall_score(y_test, y_pred_ensemble),
            "f1": f1_score(y_test, y_pred_ensemble),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "individual_accuracies": model_accuracies,
        }

        return self.training_metrics

    def _ensemble_predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Make ensemble prediction using weighted average of probabilities."""
        probs = np.zeros((X_scaled.shape[0], 2))

        for name, model in self.models.items():
            model_probs = model.predict_proba(X_scaled)
            probs += self.model_weights[name] * model_probs

        return (probs[:, 1] > 0.5).astype(int)

    def _ensemble_predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get ensemble probabilities."""
        probs = np.zeros((X_scaled.shape[0], 2))

        for name, model in self.models.items():
            model_probs = model.predict_proba(X_scaled)
            probs += self.model_weights[name] * model_probs

        return probs

    def predict(self, features: Dict) -> Dict:
        """Make a prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

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

        prob = self._ensemble_predict_proba(X_scaled)[0]
        home_prob = float(np.clip(prob[1], 0.0, 1.0))
        away_prob = float(np.clip(prob[0], 0.0, 1.0))

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(max(prob)),
        }

    def save(self, filepath: Path):
        """Save ensemble model to disk in a format compatible with model_trainer.py."""
        # Create a wrapper instance that can be loaded by the app
        wrapper = EnsembleMoneylineWrapper(
            models=self.models,
            model_weights=self.model_weights,
            scaler=self.scaler,
            feature_names=self.feature_names,
            training_metrics=self.training_metrics,
        )

        # Save in the format expected by BaseModelTrainer.load_model()
        data = {
            "model": wrapper,  # The wrapper acts as the model
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "model_name": "moneyline_ensemble",
            "saved_at": datetime.now().isoformat(),
            # Also save the components for direct loading
            "_ensemble_models": self.models,
            "_ensemble_weights": self.model_weights,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved to {filepath}")


def main():
    print("="*60)
    print("NBA MODEL TRAINING - BALLDONTLIE DATA")
    print("="*60)

    # Load cached data - try larger dataset first
    cache_files = [
        "data/balldontlie_cache/training_data_2021_2022_2023_2024.json",
        "data/balldontlie_cache/training_data_2023_2024.json",
    ]

    training_data = None
    for cache_file in cache_files:
        if Path(cache_file).exists():
            training_data = load_training_data(cache_file)
            break

    if training_data is None:
        print("Error: No cached training data found.")
        print("Run train_with_balldontlie.py first to collect data.")
        sys.exit(1)

    # Data statistics
    home_wins = sum(1 for t in training_data if t.get('home_win', False))
    avg_diff = np.mean([t.get('point_differential', 0) for t in training_data])

    print(f"\nData Statistics:")
    print(f"  Total games: {len(training_data)}")
    print(f"  Home win rate: {home_wins / len(training_data):.1%}")
    print(f"  Avg point differential: {avg_diff:+.1f}")

    # Prepare data
    print("\nPreparing training data...")
    X_ml, y_ml = prepare_moneyline_data(training_data)
    X_sp, y_sp = prepare_spread_data(training_data)

    print(f"  Moneyline features: {X_ml.shape[1]}")
    print(f"  Spread features: {X_sp.shape[1]}")

    # Train Ensemble Moneyline Model
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE MONEYLINE MODEL")
    print("  (Logistic Regression + Random Forest + Gradient Boosting)")
    print("="*60)

    ensemble_model = EnsembleMoneylineModel()
    ml_metrics = ensemble_model.train(X_ml, y_ml)

    print(f"\n  Ensemble Results:")
    print(f"    Accuracy: {ml_metrics['accuracy']:.4f}")
    print(f"    Precision: {ml_metrics['precision']:.4f}")
    print(f"    Recall: {ml_metrics['recall']:.4f}")
    print(f"    F1 Score: {ml_metrics['f1']:.4f}")

    # Save moneyline model
    ensemble_model.save(MODEL_DIR / "moneyline_ensemble.pkl")

    # Also train and save the simpler model for compatibility
    print("\n" + "="*60)
    print("TRAINING SINGLE MONEYLINE MODEL (Gradient Boosting)")
    print("="*60)

    simple_ml = SimpleMoneylineModel()
    simple_ml_metrics = simple_ml.train(X_ml, y_ml)

    print(f"\n  Results:")
    print(f"    Accuracy: {simple_ml_metrics['accuracy']:.4f}")
    print(f"    F1 Score: {simple_ml_metrics['f1']:.4f}")

    simple_ml.save(MODEL_DIR / "moneyline_gradient_boosting.pkl")

    # Train Spread Model
    print("\n" + "="*60)
    print("TRAINING SPREAD MODEL (Gradient Boosting Regressor)")
    print("="*60)

    spread_model = SimpleSpreadModel()
    sp_metrics = spread_model.train(X_sp, y_sp)

    print(f"\n  Results:")
    print(f"    RMSE: {sp_metrics['rmse']:.2f} points")
    print(f"    MAE: {sp_metrics['mae']:.2f} points")
    print(f"    R2: {sp_metrics['r2']:.4f}")

    spread_model.save(MODEL_DIR / "spread_svm_regressor.pkl")

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nModels saved to 'models/' directory:")
    print("  - moneyline_ensemble.pkl (LR + RF + GB ensemble)")
    print("  - moneyline_gradient_boosting.pkl (single GB classifier)")
    print("  - spread_svm_regressor.pkl (GB regressor for spread)")

    print(f"\nFinal Performance Summary:")
    print(f"  Moneyline Ensemble Accuracy: {ml_metrics['accuracy']:.2%}")
    print(f"  Spread Model RMSE: {sp_metrics['rmse']:.2f} points")

    print("\n" + "="*60)
    print("Run 'python3 app.py' to use trained models for predictions")
    print("="*60)


if __name__ == "__main__":
    main()
