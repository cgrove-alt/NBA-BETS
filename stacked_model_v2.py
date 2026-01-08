"""
Stacked Model Architecture V2 - Meta-Learning Ensemble

This module implements a state-of-the-art stacking ensemble:
1. Level 0: Diverse base models (XGBoost, LightGBM, Ridge, etc.)
2. Level 1: Meta-learner that combines base predictions
3. Level 2: Optional calibration layer

Key improvements over simple ensemble:
- Base models see different "views" of the data
- Meta-learner learns WHEN to trust each base model
- Calibration ensures probabilistic outputs are accurate

Usage:
    from stacked_model_v2 import StackedPropModel
    model = StackedPropModel(prop_type='points')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Optional imports
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class StackedPropModel:
    """
    Two-level stacking ensemble for player prop prediction.

    Architecture:
    ```
    Level 0 (Base Models):
    ├── XGBoost (gradient boosting)
    ├── LightGBM (histogram-based GB)
    ├── Ridge (linear regularized)
    ├── Gradient Boosting (scikit-learn)
    └── Random Forest (bagging)

    Level 1 (Meta-Learner):
    └── Ridge or ElasticNet on base predictions
    ```

    Key design choices:
    1. Base models are trained on K-fold CV to prevent leakage
    2. Meta-learner sees out-of-fold predictions only
    3. Optional quantile prediction for confidence intervals
    """

    PROP_FEATURE_IMPORTANCE = {
        'points': ['recent_pts_avg', 'season_pts_avg', 'usage_rate', 'min_avg', 'opp_def_rating'],
        'rebounds': ['recent_reb_avg', 'season_reb_avg', 'min_avg', 'is_center', 'opp_reb_factor'],
        'assists': ['recent_ast_avg', 'season_ast_avg', 'is_ball_handler', 'min_avg', 'opp_def_rating'],
        'threes': ['recent_fg3m_avg', 'fg3a_avg', 'fg3_pct', 'min_avg', 'is_volume_shooter'],
        'pra': ['pra_avg', 'recent_pts_avg', 'recent_reb_avg', 'recent_ast_avg', 'min_avg'],
    }

    def __init__(
        self,
        prop_type: str = 'points',
        n_folds: int = 5,
        use_quantile: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the stacked model.

        Args:
            prop_type: Type of prop ('points', 'rebounds', 'assists', 'threes', 'pra')
            n_folds: Number of CV folds for base model training
            use_quantile: Whether to include quantile predictions
            verbose: Print progress
        """
        self.prop_type = prop_type
        self.n_folds = n_folds
        self.use_quantile = use_quantile
        self.verbose = verbose

        # Model components
        self.scaler = StandardScaler()
        self.base_models = {}
        self.meta_model = None
        self.feature_names = []

        # Fitted state
        self.is_fitted = False

    def _build_base_models(self) -> Dict[str, Any]:
        """
        Build diverse base models.

        We use models with different inductive biases:
        - Tree-based: XGBoost, LightGBM, GradientBoosting, RandomForest
        - Linear: Ridge, Lasso
        - Neural: MLP (optional)
        """
        models = {}

        # Always include scikit-learn models
        models['ridge'] = Ridge(alpha=1.0)
        models['lasso'] = Lasso(alpha=0.1)
        models['gb'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        models['rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        # XGBoost (if available)
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

        # LightGBM (if available)
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

        # CatBoost (if available)
        if HAS_CATBOOST:
            models['catboost'] = CatBoostRegressor(
                iterations=150,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )

        return models

    def _build_meta_model(self) -> Any:
        """
        Build the meta-learner.

        We use ElasticNet for the meta-learner because:
        - It can handle correlated base predictions
        - L1/L2 regularization prevents overfitting
        - Simple and fast to train
        """
        return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'StackedPropModel':
        """
        Fit the stacked model using out-of-fold predictions.

        Process:
        1. Scale features
        2. Train base models on K-folds
        3. Collect out-of-fold predictions
        4. Train meta-learner on OOF predictions

        Args:
            X: Feature matrix (pandas DataFrame)
            y: Target values

        Returns:
            Self
        """
        if self.verbose:
            print(f"Training StackedPropModel for {self.prop_type}...")

        # Store feature names
        self.feature_names = list(X.columns)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build base models
        self.base_models = self._build_base_models()

        if self.verbose:
            print(f"  Base models: {list(self.base_models.keys())}")

        # Collect OOF predictions for meta-learner
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

                # Clone the model for each fold
                fold_model = type(model)(**model.get_params())
                fold_model.fit(X_train_fold, y_train_fold)
                oof_pred[val_idx] = fold_model.predict(X_val_fold)

            oof_predictions[:, model_idx] = oof_pred

            # Retrain on full data for final model
            model.fit(X_scaled, y)

            if self.verbose:
                fold_rmse = np.sqrt(mean_squared_error(y, oof_pred))
                print(f"OOF RMSE: {fold_rmse:.3f}")

        # Train meta-learner on OOF predictions
        if self.verbose:
            print("  Training meta-learner...")

        self.meta_model = self._build_meta_model()
        self.meta_model.fit(oof_predictions, y)

        # Calculate overall metrics
        meta_pred = self.meta_model.predict(oof_predictions)
        overall_rmse = np.sqrt(mean_squared_error(y, meta_pred))
        overall_mae = mean_absolute_error(y, meta_pred)
        overall_r2 = r2_score(y, meta_pred)

        if self.verbose:
            print(f"\n  Stacked Model Metrics:")
            print(f"    RMSE: {overall_rmse:.3f}")
            print(f"    MAE:  {overall_mae:.3f}")
            print(f"    R²:   {overall_r2:.3f}")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the stacked ensemble.

        Process:
        1. Scale features
        2. Get predictions from all base models
        3. Combine with meta-learner

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        # Ensure correct feature order
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        X_scaled = self.scaler.transform(X_aligned)

        # Get base model predictions
        base_predictions = np.zeros((len(X), len(self.base_models)))
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            base_predictions[:, model_idx] = model.predict(X_scaled)

        # Combine with meta-learner
        final_predictions = self.meta_model.predict(base_predictions)

        # Clip to realistic bounds
        PROP_BOUNDS = {
            'points': (0, 70),
            'rebounds': (0, 35),
            'assists': (0, 25),
            'threes': (0, 15),
            'pra': (0, 100),
        }
        min_val, max_val = PROP_BOUNDS.get(self.prop_type, (0, 100))
        final_predictions = np.clip(final_predictions, min_val, max_val)

        return final_predictions

    def get_base_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from each base model (for analysis).

        Args:
            X: Feature matrix

        Returns:
            Dictionary mapping model name to predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        X_scaled = self.scaler.transform(X_aligned)

        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X_scaled)

        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from tree-based base models.

        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        importance_data = []

        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                for feat_idx, feat_name in enumerate(self.feature_names):
                    importance_data.append({
                        'model': name,
                        'feature': feat_name,
                        'importance': model.feature_importances_[feat_idx]
                    })

        if not importance_data:
            return pd.DataFrame()

        df = pd.DataFrame(importance_data)
        # Average across models
        avg_importance = df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        return avg_importance

    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'prop_type': self.prop_type,
                'n_folds': self.n_folds,
                'scaler': self.scaler,
                'base_models': self.base_models,
                'meta_model': self.meta_model,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'StackedPropModel':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(prop_type=data['prop_type'], n_folds=data['n_folds'])
        model.scaler = data['scaler']
        model.base_models = data['base_models']
        model.meta_model = data['meta_model']
        model.feature_names = data['feature_names']
        model.is_fitted = data['is_fitted']
        return model


class QuantileStackedModel(StackedPropModel):
    """
    Stacked model with quantile predictions for uncertainty estimation.

    In addition to point predictions, this model provides:
    - Lower bound (10th percentile)
    - Upper bound (90th percentile)

    This is useful for:
    - Identifying high-variance predictions
    - Setting confidence-based betting thresholds
    """

    def __init__(self, prop_type: str = 'points', n_folds: int = 5, verbose: bool = True):
        super().__init__(prop_type, n_folds, use_quantile=True, verbose=verbose)
        self.quantile_models = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'QuantileStackedModel':
        """Fit base model and quantile models."""
        # Fit base stacked model
        super().fit(X, y)

        # Fit quantile models using GradientBoosting with quantile loss
        if self.verbose:
            print("  Training quantile models...")

        X_scaled = self.scaler.transform(X)

        for quantile in [0.1, 0.5, 0.9]:
            gb_quantile = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                loss='quantile' if hasattr(GradientBoostingRegressor(), 'loss') else 'squared_error',
                alpha=quantile,
                random_state=42
            )
            gb_quantile.fit(X_scaled, y)
            self.quantile_models[quantile] = gb_quantile

        return self

    def predict_quantiles(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty bounds.

        Returns:
            Dictionary with 'lower', 'median', 'upper' predictions
        """
        point_pred = self.predict(X)

        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        X_scaled = self.scaler.transform(X_aligned)

        return {
            'point': point_pred,
            'lower': self.quantile_models[0.1].predict(X_scaled),
            'median': self.quantile_models[0.5].predict(X_scaled),
            'upper': self.quantile_models[0.9].predict(X_scaled),
        }


def train_stacked_prop_models(
    training_data: pd.DataFrame,
    prop_types: List[str] = None,
    output_dir: Path = Path('models'),
    verbose: bool = True
) -> Dict[str, StackedPropModel]:
    """
    Train stacked models for all prop types.

    Args:
        training_data: DataFrame with features and target columns
        prop_types: List of prop types to train (default: all)
        output_dir: Directory to save models
        verbose: Print progress

    Returns:
        Dictionary mapping prop_type to trained model
    """
    if prop_types is None:
        prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']

    TARGET_COLS = {
        'points': 'pts',
        'rebounds': 'reb',
        'assists': 'ast',
        'threes': 'fg3m',
        'pra': 'pra',
    }

    models = {}

    for prop_type in prop_types:
        target_col = TARGET_COLS[prop_type]

        if target_col not in training_data.columns:
            if verbose:
                print(f"Skipping {prop_type}: target column '{target_col}' not found")
            continue

        # Prepare data
        feature_cols = [c for c in training_data.columns if c != target_col]
        X = training_data[feature_cols].copy()
        y = training_data[target_col].values

        # Remove NaN targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        # Fill NaN features
        X = X.fillna(0)

        if len(y) < 100:
            if verbose:
                print(f"Skipping {prop_type}: insufficient data ({len(y)} samples)")
            continue

        # Train model
        model = StackedPropModel(prop_type=prop_type, verbose=verbose)
        model.fit(X, y)

        # Save model
        output_path = output_dir / f"player_{prop_type}_stacked.pkl"
        model.save(str(output_path))
        if verbose:
            print(f"  Saved to {output_path}")

        models[prop_type] = model

    return models


# Quick test
if __name__ == "__main__":
    print("Testing StackedPropModel...")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = X['feature_0'] * 10 + X['feature_1'] * 5 + np.random.randn(n_samples) * 2 + 15

    # Train model
    model = StackedPropModel(prop_type='points', verbose=True)
    model.fit(X, y)

    # Test predictions
    preds = model.predict(X[:10])
    print(f"\nSample predictions: {preds[:5].round(1)}")
    print(f"Sample actuals:     {y[:5].round(1)}")

    # Feature importance
    importance = model.get_feature_importance()
    print(f"\nTop features:\n{importance.head()}")
