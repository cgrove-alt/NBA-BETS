"""
Stacking Meta-Learner Module for NBA Prediction Models

This module implements a sophisticated stacking ensemble approach that combines
predictions from multiple base models using a meta-learner. The stacking approach
uses out-of-fold (OOF) predictions to prevent data leakage and overfitting.

Key Features:
- Out-of-fold prediction generation using TimeSeriesSplit or KFold
  * Each base model is cloned and retrained on K-1 folds
  * Predictions are made on the held-out fold (prevents data leakage)
- Multiple meta-learner options (XGBoost, Neural Network, Ridge Regression)
- Context feature integration for enhanced prediction (12 contextual features)
- Sample weight support with time-decay (prioritizes recent games)
- Uncertainty quantification via prediction variance
- Temporal discipline to prevent future data leakage

Expected Impact:
- Provides rigorous ensemble learning without overfitting
- Actual performance improvement depends on base model diversity
- Must be validated through backtesting on held-out data
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StackingMetaLearner:
    """
    Stacking ensemble meta-learner that combines multiple base model predictions.

    The stacking approach works in two stages:
    1. Base models generate out-of-fold predictions on training data
    2. Meta-learner trains on OOF predictions to learn optimal combination

    This prevents overfitting by ensuring the meta-learner never sees
    predictions from models trained on the same data.
    """

    def __init__(
        self,
        base_models: List[Any],
        meta_learner_type: str = 'xgboost',
        cv_folds: int = 5,
        time_series_split: bool = True,
        regularization_strength: float = 1.0,
        random_state: int = 42,
        task_type: str = 'regression'
    ):
        """
        Initialize the Stacking Meta-Learner.

        Parameters:
        -----------
        base_models : List[Any]
            List of trained base models (e.g., XGBoost, LightGBM, etc.)
            Each model should have .predict() method
        meta_learner_type : str
            Type of meta-learner: 'xgboost', 'neural_network', or 'ridge'
        cv_folds : int
            Number of cross-validation folds for OOF prediction generation
        time_series_split : bool
            If True, use TimeSeriesSplit (respects temporal order)
            If False, use standard KFold (for non-temporal data)
        regularization_strength : float
            L2 regularization strength (alpha for Ridge, lambda for XGBoost)
        random_state : int
            Random seed for reproducibility
        task_type : str
            'regression' or 'classification' - determines how to generate predictions
        """
        self.base_models = base_models
        self.meta_learner_type = meta_learner_type
        self.cv_folds = cv_folds
        self.time_series_split = time_series_split
        self.regularization_strength = regularization_strength
        self.random_state = random_state
        self.task_type = task_type

        # Initialize components
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.context_scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.context_feature_names = None

        # Performance tracking
        self.oof_scores = {}
        self.base_model_weights = None

        logger.info(f"Initialized StackingMetaLearner with {len(base_models)} base models")
        logger.info(f"Meta-learner type: {meta_learner_type}, CV folds: {cv_folds}, Task: {task_type}")

    def _initialize_meta_learner(self, n_features: int):
        """
        Initialize the meta-learner based on specified type.

        Parameters:
        -----------
        n_features : int
            Number of input features (base model predictions + context features)
        """
        if self.meta_learner_type == 'xgboost':
            # XGBoost with strong regularization to prevent overfitting
            # For classification, use logistic objective to predict probabilities
            if self.task_type == 'classification':
                objective = 'reg:logistic'  # Predicts probabilities (0-1)
            else:
                objective = 'reg:squarederror'

            self.meta_learner = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,  # Shallow trees prevent overfitting
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=self.regularization_strength,  # L2 regularization
                random_state=self.random_state,
                objective=objective,
                verbosity=0
            )
            logger.info(f"Initialized XGBoost meta-learner with regularization (task: {self.task_type})")

        elif self.meta_learner_type == 'neural_network':
            # Multi-layer perceptron with dropout for regularization
            self.meta_learner = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                solver='adam',
                alpha=self.regularization_strength,  # L2 regularization
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=False
            )
            logger.info("Initialized Neural Network meta-learner (32-16 architecture)")

        elif self.meta_learner_type == 'ridge':
            # Ridge regression with polynomial features
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
            self.meta_learner = Ridge(
                alpha=self.regularization_strength,
                random_state=self.random_state
            )
            logger.info("Initialized Ridge meta-learner with polynomial features")

        else:
            raise ValueError(f"Unknown meta_learner_type: {self.meta_learner_type}")

    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate out-of-fold predictions from base models.

        This is the core of the stacking approach. For each fold:
        1. Train base models on K-1 folds
        2. Generate predictions on the held-out fold
        3. Combine OOF predictions from all folds

        This ensures no data leakage: the meta-learner never sees
        predictions from models trained on the same data.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples,)
        sample_weights : np.ndarray, optional
            Sample weights for training (n_samples,)

        Returns:
        --------
        oof_predictions : np.ndarray
            Out-of-fold predictions (n_samples, n_base_models)
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models))

        # Setup cross-validation strategy
        if self.time_series_split:
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            logger.info(f"Using TimeSeriesSplit with {self.cv_folds} folds")
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            logger.info(f"Using KFold with {self.cv_folds} folds")

        # Generate OOF predictions for each base model
        for model_idx, base_model in enumerate(self.base_models):
            model_name = base_model.__class__.__name__
            logger.info(f"Generating OOF predictions for model {model_idx + 1}/{n_models}: {model_name}")

            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Handle sample weights if provided
                if sample_weights is not None:
                    weights_train = sample_weights[train_idx]
                else:
                    weights_train = None

                # CRITICAL: Clone the base model to prevent data leakage
                # Each fold must train on ONLY the training data for that fold
                try:
                    cloned_model = clone(base_model)

                    # Train cloned model on training fold only
                    if weights_train is not None:
                        # Check if model supports sample_weight
                        if hasattr(cloned_model, 'fit'):
                            fit_params = cloned_model.fit.__code__.co_varnames
                            if 'sample_weight' in fit_params:
                                cloned_model.fit(X_train, y_train, sample_weight=weights_train)
                            else:
                                logger.warning(f"{model_name} does not support sample_weight, training without weights")
                                cloned_model.fit(X_train, y_train)
                        else:
                            cloned_model.fit(X_train, y_train)
                    else:
                        cloned_model.fit(X_train, y_train)

                    # Generate predictions on validation fold (model has NOT seen this data)
                    # For classification, use predict_proba to get probabilities
                    if self.task_type == 'classification' and hasattr(cloned_model, 'predict_proba'):
                        # Get probability of positive class (index 1)
                        val_predictions = cloned_model.predict_proba(X_val)[:, 1]
                    else:
                        val_predictions = cloned_model.predict(X_val)

                    # Store OOF predictions
                    oof_predictions[val_idx, model_idx] = val_predictions

                    # Calculate fold score
                    fold_rmse = np.sqrt(np.mean((y_val - val_predictions) ** 2))
                    fold_scores.append(fold_rmse)

                except Exception as e:
                    logger.error(f"Error generating OOF predictions for {model_name} on fold {fold_idx}: {e}")
                    logger.error(f"Exception details: {str(e)}")
                    # Re-raise the exception to fail fast rather than silently producing bad results
                    raise RuntimeError(f"Failed to generate OOF predictions for {model_name} on fold {fold_idx}") from e

            # Store average performance
            avg_rmse = np.mean(fold_scores)
            self.oof_scores[model_name] = avg_rmse
            logger.info(f"  {model_name} OOF RMSE: {avg_rmse:.4f}")

        # CRITICAL: After generating OOF predictions, train each base model on the FULL training set
        # This is necessary so the models can be used for final predictions
        logger.info("\nTraining base models on full training set...")
        for model_idx, base_model in enumerate(self.base_models):
            model_name = base_model.__class__.__name__
            try:
                if sample_weights is not None:
                    # Check if model supports sample_weight
                    if hasattr(base_model, 'fit'):
                        fit_params = base_model.fit.__code__.co_varnames
                        if 'sample_weight' in fit_params:
                            base_model.fit(X, y, sample_weight=sample_weights)
                        else:
                            base_model.fit(X, y)
                    else:
                        base_model.fit(X, y)
                else:
                    base_model.fit(X, y)
                logger.info(f"  Trained {model_name} on full dataset ({len(X)} samples)")
            except Exception as e:
                logger.error(f"Error training {model_name} on full dataset: {e}")
                raise

        return oof_predictions

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context_features: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None
    ):
        """
        Fit the stacking meta-learner.

        Process:
        1. Generate OOF predictions from base models
        2. Combine with context features
        3. Train meta-learner on combined features

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix for base models (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples,)
        context_features : np.ndarray, optional
            Additional context features (n_samples, n_context_features)
            Examples: days_rest_diff, pace, injury_count, etc.
        sample_weights : np.ndarray, optional
            Sample weights for training (n_samples,)
            Recommended: Time-decay weights (recent games weighted higher)
        """
        logger.info("=" * 80)
        logger.info("Starting Stacking Meta-Learner Training")
        logger.info("=" * 80)

        # Validate inputs
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")

        if context_features is not None and len(context_features) != len(X):
            raise ValueError(f"context_features must have same length as X: {len(context_features)} vs {len(X)}")

        # Step 1: Generate out-of-fold predictions
        logger.info("\nStep 1: Generating out-of-fold predictions...")
        oof_predictions = self._generate_oof_predictions(X, y, sample_weights)
        logger.info(f"OOF predictions shape: {oof_predictions.shape}")

        # Step 2: Combine with context features
        if context_features is not None:
            logger.info("\nStep 2: Adding context features...")
            # Normalize context features
            context_features_scaled = self.context_scaler.fit_transform(context_features)

            # Store context feature names for feature importance
            n_context = context_features.shape[1]
            self.context_feature_names = [f"Context_{i+1}" for i in range(n_context)]

            # Combine OOF predictions with context features
            meta_features = np.hstack([oof_predictions, context_features_scaled])
            logger.info(f"Combined features shape: {meta_features.shape}")
            logger.info(f"  - Base model predictions: {oof_predictions.shape[1]}")
            logger.info(f"  - Context features: {context_features_scaled.shape[1]}")
        else:
            meta_features = oof_predictions
            self.context_feature_names = None
            logger.info("\nNo context features provided, using only base model predictions")

        # Step 3: Normalize meta-features
        logger.info("\nStep 3: Normalizing meta-features...")
        meta_features_scaled = self.scaler.fit_transform(meta_features)

        # Step 4: Initialize and train meta-learner
        logger.info("\nStep 4: Training meta-learner...")
        self._initialize_meta_learner(n_features=meta_features_scaled.shape[1])

        # Apply polynomial features for Ridge regression
        if self.meta_learner_type == 'ridge':
            meta_features_scaled = self.poly_features.fit_transform(meta_features_scaled)
            logger.info(f"Applied polynomial features, new shape: {meta_features_scaled.shape}")

        # Train meta-learner with sample weights if provided
        if sample_weights is not None:
            # XGBoost, sklearn models, and most ML libraries support sample_weight
            try:
                self.meta_learner.fit(meta_features_scaled, y, sample_weight=sample_weights)
                logger.info(f"Training meta-learner with sample weights (mean weight: {np.mean(sample_weights):.4f})")
            except TypeError as e:
                # Fallback if sample_weight not supported
                logger.warning(f"Meta-learner does not support sample_weight: {e}")
                logger.warning("Training without sample weights")
                self.meta_learner.fit(meta_features_scaled, y)
        else:
            self.meta_learner.fit(meta_features_scaled, y)
            logger.info("Training meta-learner without sample weights")

        # Step 5: Calculate base model weights (for interpretability)
        self._calculate_base_model_weights(oof_predictions, y)

        # Mark as fitted
        self.is_fitted = True

        # Calculate final training performance
        final_predictions = self.predict(X, context_features)
        final_rmse = np.sqrt(np.mean((y - final_predictions) ** 2))

        logger.info("\n" + "=" * 80)
        logger.info(f"Training Complete! Final RMSE: {final_rmse:.4f}")
        logger.info("=" * 80)

        return self

    def predict(
        self,
        X: np.ndarray,
        context_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate predictions using the stacking ensemble.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        context_features : np.ndarray, optional
            Context features (n_samples, n_context_features)

        Returns:
        --------
        predictions : np.ndarray
            Final predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")

        # Step 1: Get base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for idx, model in enumerate(self.base_models):
            # For classification, use predict_proba to get probabilities
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                base_predictions[:, idx] = model.predict_proba(X)[:, 1]
            else:
                base_predictions[:, idx] = model.predict(X)

        # Step 2: Combine with context features if provided
        if context_features is not None:
            context_scaled = self.context_scaler.transform(context_features)
            meta_features = np.hstack([base_predictions, context_scaled])
        else:
            meta_features = base_predictions

        # Step 3: Normalize
        meta_features_scaled = self.scaler.transform(meta_features)

        # Step 4: Apply polynomial features for Ridge
        if self.meta_learner_type == 'ridge':
            meta_features_scaled = self.poly_features.transform(meta_features_scaled)

        # Step 5: Generate final predictions
        predictions = self.meta_learner.predict(meta_features_scaled)

        return predictions

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        context_features: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.

        Uncertainty is quantified as the variance of base model predictions.
        Higher variance = higher uncertainty = lower confidence.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        context_features : np.ndarray, optional
            Context features (n_samples, n_context_features)

        Returns:
        --------
        predictions : np.ndarray
            Final predictions (n_samples,)
        confidence_scores : np.ndarray
            Confidence scores 0-100 (n_samples,)
            100 = highest confidence, 0 = lowest confidence
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")

        # Get base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for idx, model in enumerate(self.base_models):
            # For classification, use predict_proba to get probabilities
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                base_predictions[:, idx] = model.predict_proba(X)[:, 1]
            else:
                base_predictions[:, idx] = model.predict(X)

        # Calculate prediction variance (uncertainty)
        prediction_variance = np.var(base_predictions, axis=1)
        prediction_mean = np.mean(base_predictions, axis=1)

        # Calculate confidence score
        # Confidence = 100 * (1 - min(std_dev / mean, 1.0))
        # High variance relative to mean = low confidence
        std_dev = np.sqrt(prediction_variance)
        confidence_scores = 100 * (1 - np.minimum(std_dev / (np.abs(prediction_mean) + 1e-6), 1.0))
        confidence_scores = np.clip(confidence_scores, 0, 100)

        # Get final predictions from meta-learner
        predictions = self.predict(X, context_features)

        return predictions, confidence_scores

    def _calculate_base_model_weights(self, oof_predictions: np.ndarray, y: np.ndarray):
        """
        Calculate effective weights of base models for interpretability.

        Uses inverse RMSE as weight (better models get higher weight).
        This is for reporting only; actual combination is learned by meta-learner.

        Parameters:
        -----------
        oof_predictions : np.ndarray
            Out-of-fold predictions (n_samples, n_base_models)
        y : np.ndarray
            True target values (n_samples,)
        """
        n_models = oof_predictions.shape[1]
        rmse_scores = []

        for idx in range(n_models):
            rmse = np.sqrt(np.mean((y - oof_predictions[:, idx]) ** 2))
            rmse_scores.append(rmse)

        # Calculate weights as inverse RMSE (normalized)
        inverse_rmse = 1.0 / np.array(rmse_scores)
        weights = inverse_rmse / np.sum(inverse_rmse)

        self.base_model_weights = weights

        logger.info("\nBase Model Performance (OOF):")
        for idx, (model, weight, rmse) in enumerate(zip(self.base_models, weights, rmse_scores)):
            model_name = model.__class__.__name__
            logger.info(f"  {model_name}: RMSE={rmse:.4f}, Weight={weight:.4f}")

    def get_base_model_weights(self) -> Dict[str, float]:
        """
        Get the effective weights of base models.

        Returns:
        --------
        weights : dict
            Dictionary mapping model names to weights
        """
        if self.base_model_weights is None:
            raise ValueError("Model must be fitted first")

        weights = {}
        for idx, model in enumerate(self.base_models):
            model_name = model.__class__.__name__
            weights[model_name] = float(self.base_model_weights[idx])

        return weights

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from meta-learner (if supported).

        Returns:
        --------
        importance : dict or None
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if self.meta_learner_type == 'xgboost':
            importance_scores = self.meta_learner.feature_importances_

            # Create feature names
            feature_names = [f"BaseModel_{i+1}" for i in range(len(self.base_models))]
            if self.context_feature_names is not None:
                feature_names.extend(self.context_feature_names)

            importance = dict(zip(feature_names, importance_scores))
            return importance
        else:
            logger.warning(f"Feature importance not available for {self.meta_learner_type}")
            return None


def calculate_time_decay_weights(
    dates: Union[List[datetime], np.ndarray],
    half_life_days: int = 180,
    reference_date: Optional[datetime] = None
) -> np.ndarray:
    """
    Calculate time-decay sample weights for training.

    Recent games are weighted higher than older games using exponential decay.
    Formula: weight = 0.5 ^ (days_ago / half_life_days)

    Parameters:
    -----------
    dates : list or array
        Game dates
    half_life_days : int
        Number of days for weight to decay to 50% (default: 180 = 6 months)
    reference_date : datetime, optional
        Reference date for calculating decay (default: today)

    Returns:
    --------
    weights : np.ndarray
        Sample weights (n_samples,)

    Example:
    --------
    >>> dates = [datetime(2024, 1, 1), datetime(2024, 6, 1), datetime(2024, 11, 1)]
    >>> weights = calculate_time_decay_weights(dates, half_life_days=180)
    >>> # Game from Nov 2024 gets weight ~1.0
    >>> # Game from Jun 2024 gets weight ~0.5
    >>> # Game from Jan 2024 gets weight ~0.25
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Convert to datetime if needed
    if isinstance(dates, pd.Series):
        dates = dates.tolist()

    if isinstance(dates[0], str):
        dates = [datetime.strptime(d, '%Y-%m-%d') if isinstance(d, str) else d for d in dates]

    # Calculate days ago
    days_ago = np.array([(reference_date - d).days for d in dates])

    # Calculate exponential decay weights
    weights = 0.5 ** (days_ago / half_life_days)

    # Normalize to sum to 1
    weights = weights / np.sum(weights) * len(weights)

    return weights


# Example usage and testing
if __name__ == "__main__":
    print("Testing StackingMetaLearner...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y_true = 25 + 2 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2]
    y = y_true + np.random.randn(n_samples) * 2

    # Create mock base models
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet

    base_models = [
        RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y),
        GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X, y),
        ElasticNet(random_state=42).fit(X, y)
    ]

    # Create context features
    context_features = np.random.randn(n_samples, 12)

    # Calculate time-decay weights
    dates = [datetime.now() - timedelta(days=i) for i in range(n_samples)]
    sample_weights = calculate_time_decay_weights(dates, half_life_days=180)

    # Initialize and train stacking meta-learner
    stacker = StackingMetaLearner(
        base_models=base_models,
        meta_learner_type='xgboost',
        cv_folds=5,
        time_series_split=True
    )

    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    context_train = context_features[:train_size]
    context_test = context_features[train_size:]
    weights_train = sample_weights[:train_size]

    # Fit
    stacker.fit(X_train, y_train, context_features=context_train, sample_weights=weights_train)

    # Predict with uncertainty
    predictions, confidence = stacker.predict_with_uncertainty(X_test, context_features=context_test)

    # Evaluate
    test_rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    print(f"\nTest RMSE: {test_rmse:.4f}")
    print(f"Average confidence: {np.mean(confidence):.2f}")

    # Get base model weights
    weights = stacker.get_base_model_weights()
    print("\nBase Model Weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")

    print("\nStackingMetaLearner test completed successfully!")
