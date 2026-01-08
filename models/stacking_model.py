"""
Stacking Model Framework for NBA Predictions

Two-layer stacking architecture for improved predictions:
- Level 1: Diverse base models (XGBoost, LightGBM, RandomForest, Ridge)
- Level 2: Meta-learner to combine predictions

This framework supports both classification (moneyline) and regression (props/spread).
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings

warnings.filterwarnings('ignore')

# Optional imports for advanced models
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False
    print("LightGBM not available")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class StackingClassifier:
    """
    Two-layer stacking classifier for binary classification (e.g., moneyline).

    Architecture:
    - Level 1: XGBoost, LightGBM, RandomForest, GradientBoosting
    - Level 2: LogisticRegression meta-learner

    Uses out-of-fold predictions to train meta-learner (prevents leakage).
    """

    def __init__(self, n_folds: int = 5, use_proba: bool = True, verbose: bool = True):
        """
        Args:
            n_folds: Number of CV folds for OOF predictions
            use_proba: Use probability predictions (recommended for meta-learner)
            verbose: Print progress
        """
        self.n_folds = n_folds
        self.use_proba = use_proba
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.base_models = {}
        self.meta_model = None
        self.feature_names = []
        self.is_fitted = False

    def _build_base_models(self) -> Dict[str, Any]:
        """Build diverse base classifiers."""
        models = {}

        # Gradient Boosting (sklearn)
        models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

        # Random Forest
        models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        # XGBoost
        if HAS_XGBOOST:
            models['xgb'] = XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )

        # LightGBM
        if HAS_LIGHTGBM:
            models['lgbm'] = LGBMClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            )

        # CatBoost
        if HAS_CATBOOST:
            models['catboost'] = CatBoostClassifier(
                iterations=150,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )

        return models

    def _build_meta_model(self) -> LogisticRegression:
        """Build meta-learner (logistic regression)."""
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'StackingClassifier':
        """
        Fit the stacking classifier.

        Uses K-fold CV to generate OOF predictions for meta-learner training.
        """
        if self.verbose:
            print("Training StackingClassifier...")

        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        self.base_models = self._build_base_models()

        if self.verbose:
            print(f"  Base models: {list(self.base_models.keys())}")

        n_samples = len(y)
        n_models = len(self.base_models)

        # OOF predictions (probabilities or class predictions)
        if self.use_proba:
            oof_predictions = np.zeros((n_samples, n_models))
        else:
            oof_predictions = np.zeros((n_samples, n_models))

        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for model_idx, (name, model) in enumerate(self.base_models.items()):
            if self.verbose:
                print(f"  Training {name}...", end=" ")

            oof_pred = np.zeros(n_samples)

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
                X_train_fold = X_scaled[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X_scaled[val_idx]

                fold_model = type(model)(**model.get_params())
                fold_model.fit(X_train_fold, y_train_fold)

                if self.use_proba:
                    oof_pred[val_idx] = fold_model.predict_proba(X_val_fold)[:, 1]
                else:
                    oof_pred[val_idx] = fold_model.predict(X_val_fold)

            oof_predictions[:, model_idx] = oof_pred

            # Retrain on full data
            model.fit(X_scaled, y)

            if self.verbose:
                if self.use_proba:
                    oof_logloss = log_loss(y, oof_pred)
                    print(f"OOF LogLoss: {oof_logloss:.4f}")
                else:
                    oof_acc = accuracy_score(y, oof_pred.round())
                    print(f"OOF Accuracy: {oof_acc:.4f}")

        # Train meta-learner
        if self.verbose:
            print("  Training meta-learner...")

        self.meta_model = self._build_meta_model()
        self.meta_model.fit(oof_predictions, y)

        # Calculate stacked model metrics
        meta_pred_proba = self.meta_model.predict_proba(oof_predictions)[:, 1]
        meta_pred = self.meta_model.predict(oof_predictions)

        stacked_acc = accuracy_score(y, meta_pred)
        stacked_logloss = log_loss(y, meta_pred_proba)
        stacked_brier = brier_score_loss(y, meta_pred_proba)

        if self.verbose:
            print(f"\n  Stacked Model Metrics:")
            print(f"    Accuracy:  {stacked_acc:.4f}")
            print(f"    Log Loss:  {stacked_logloss:.4f}")
            print(f"    Brier:     {stacked_brier:.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        X_scaled = self.scaler.transform(X_aligned)

        base_predictions = np.zeros((len(X), len(self.base_models)))
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            if self.use_proba:
                base_predictions[:, model_idx] = model.predict_proba(X_scaled)[:, 1]
            else:
                base_predictions[:, model_idx] = model.predict(X_scaled)

        return self.meta_model.predict(base_predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        X_scaled = self.scaler.transform(X_aligned)

        base_predictions = np.zeros((len(X), len(self.base_models)))
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            if self.use_proba:
                base_predictions[:, model_idx] = model.predict_proba(X_scaled)[:, 1]
            else:
                base_predictions[:, model_idx] = model.predict(X_scaled)

        return self.meta_model.predict_proba(base_predictions)

    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'base_models': self.base_models,
                'meta_model': self.meta_model,
                'feature_names': self.feature_names,
                'use_proba': self.use_proba,
                'is_fitted': self.is_fitted,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'StackingClassifier':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls()
        model.scaler = data['scaler']
        model.base_models = data['base_models']
        model.meta_model = data['meta_model']
        model.feature_names = data['feature_names']
        model.use_proba = data['use_proba']
        model.is_fitted = data['is_fitted']
        return model


class StackingRegressor:
    """
    Two-layer stacking regressor for continuous predictions (e.g., spread, props).

    Architecture:
    - Level 1: XGBoost, LightGBM, RandomForest, Ridge
    - Level 2: Ridge/ElasticNet meta-learner
    """

    def __init__(self, n_folds: int = 5, verbose: bool = True):
        self.n_folds = n_folds
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.base_models = {}
        self.meta_model = None
        self.feature_names = []
        self.is_fitted = False

    def _build_base_models(self) -> Dict[str, Any]:
        """Build diverse base regressors."""
        models = {}

        # Ridge
        models['ridge'] = Ridge(alpha=1.0)

        # Gradient Boosting
        models['gb'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

        # Random Forest
        models['rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        # XGBoost
        if HAS_XGBOOST:
            models['xgb'] = XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0
            )

        # LightGBM
        if HAS_LIGHTGBM:
            models['lgbm'] = LGBMRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            )

        # CatBoost
        if HAS_CATBOOST:
            models['catboost'] = CatBoostRegressor(
                iterations=150,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )

        return models

    def _build_meta_model(self) -> ElasticNet:
        """Build meta-learner."""
        return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'StackingRegressor':
        """Fit the stacking regressor."""
        if self.verbose:
            print("Training StackingRegressor...")

        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        self.base_models = self._build_base_models()

        if self.verbose:
            print(f"  Base models: {list(self.base_models.keys())}")

        n_samples = len(y)
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models))

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for model_idx, (name, model) in enumerate(self.base_models.items()):
            if self.verbose:
                print(f"  Training {name}...", end=" ")

            oof_pred = np.zeros(n_samples)

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
                X_train_fold = X_scaled[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X_scaled[val_idx]

                fold_model = type(model)(**model.get_params())
                fold_model.fit(X_train_fold, y_train_fold)
                oof_pred[val_idx] = fold_model.predict(X_val_fold)

            oof_predictions[:, model_idx] = oof_pred
            model.fit(X_scaled, y)

            if self.verbose:
                oof_rmse = np.sqrt(mean_squared_error(y, oof_pred))
                print(f"OOF RMSE: {oof_rmse:.3f}")

        # Train meta-learner
        if self.verbose:
            print("  Training meta-learner...")

        self.meta_model = self._build_meta_model()
        self.meta_model.fit(oof_predictions, y)

        meta_pred = self.meta_model.predict(oof_predictions)
        stacked_rmse = np.sqrt(mean_squared_error(y, meta_pred))
        stacked_mae = mean_absolute_error(y, meta_pred)
        stacked_r2 = r2_score(y, meta_pred)

        if self.verbose:
            print(f"\n  Stacked Model Metrics:")
            print(f"    RMSE: {stacked_rmse:.3f}")
            print(f"    MAE:  {stacked_mae:.3f}")
            print(f"    R²:   {stacked_r2:.3f}")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        X_scaled = self.scaler.transform(X_aligned)

        base_predictions = np.zeros((len(X), len(self.base_models)))
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            base_predictions[:, model_idx] = model.predict(X_scaled)

        return self.meta_model.predict(base_predictions)

    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'base_models': self.base_models,
                'meta_model': self.meta_model,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'StackingRegressor':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls()
        model.scaler = data['scaler']
        model.base_models = data['base_models']
        model.meta_model = data['meta_model']
        model.feature_names = data['feature_names']
        model.is_fitted = data['is_fitted']
        return model


# Factory function
def create_stacking_model(task: str = 'classification', **kwargs):
    """
    Create appropriate stacking model for task.

    Args:
        task: 'classification' for moneyline, 'regression' for spread/props
        **kwargs: Additional arguments passed to model

    Returns:
        StackingClassifier or StackingRegressor
    """
    if task == 'classification':
        return StackingClassifier(**kwargs)
    elif task == 'regression':
        return StackingRegressor(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")
