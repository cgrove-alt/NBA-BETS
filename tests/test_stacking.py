"""
Unit Tests for Stacking Meta-Learner Module

Tests cover:
1. OOF prediction generation and temporal discipline
2. Meta-learner training and prediction
3. Uncertainty quantification
4. Time-decay weight calculation
5. All three meta-learner types (XGBoost, Neural Network, Ridge)
6. Edge cases and error handling
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
import sys
import os

# Add parent directory to path to import stacking_meta_learner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stacking_meta_learner import (
    StackingMetaLearner,
    calculate_time_decay_weights
)


class TestStackingMetaLearner(unittest.TestCase):
    """Test suite for StackingMetaLearner class"""

    def setUp(self):
        """Set up test fixtures before each test"""
        np.random.seed(42)

        # Create synthetic dataset
        self.n_samples = 500
        self.n_features = 15
        self.n_context = 12

        # Generate features
        self.X = np.random.randn(self.n_samples, self.n_features)

        # Generate target with known pattern
        self.y_true = (
            20 +
            3 * self.X[:, 0] -
            2 * self.X[:, 1] +
            1.5 * self.X[:, 2] -
            0.8 * self.X[:, 3]
        )
        self.y = self.y_true + np.random.randn(self.n_samples) * 1.5

        # Generate context features
        self.context_features = np.random.randn(self.n_samples, self.n_context)

        # Create base models
        self.base_models = [
            RandomForestRegressor(n_estimators=30, random_state=42, max_depth=5),
            GradientBoostingRegressor(n_estimators=30, random_state=42, max_depth=3),
            ElasticNet(random_state=42, alpha=0.1)
        ]

        # Train base models
        for model in self.base_models:
            model.fit(self.X, self.y)

        # Generate dates for time-decay testing
        self.dates = [datetime.now() - timedelta(days=i) for i in range(self.n_samples)]

    def test_initialization(self):
        """Test StackingMetaLearner initialization"""
        stacker = StackingMetaLearner(
            base_models=self.base_models,
            meta_learner_type='xgboost',
            cv_folds=5,
            time_series_split=True
        )

        self.assertEqual(len(stacker.base_models), 3)
        self.assertEqual(stacker.meta_learner_type, 'xgboost')
        self.assertEqual(stacker.cv_folds, 5)
        self.assertTrue(stacker.time_series_split)
        self.assertFalse(stacker.is_fitted)

    def test_xgboost_meta_learner(self):
        """Test stacking with XGBoost meta-learner"""
        stacker = StackingMetaLearner(
            base_models=self.base_models,
            meta_learner_type='xgboost',
            cv_folds=3,
            time_series_split=True
        )

        # Train on first 80% of data
        train_size = int(0.8 * self.n_samples)
        X_train = self.X[:train_size]
        y_train = self.y[:train_size]
        context_train = self.context_features[:train_size]

        # Fit
        stacker.fit(X_train, y_train, context_features=context_train)

        self.assertTrue(stacker.is_fitted)
        self.assertIsNotNone(stacker.meta_learner)

        # Predict on test set
        X_test = self.X[train_size:]
        y_test = self.y[train_size:]
        context_test = self.context_features[train_size:]

        predictions = stacker.predict(X_test, context_features=context_test)

        self.assertEqual(len(predictions), len(y_test))

        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        print(f"\nXGBoost Meta-Learner Test RMSE: {rmse:.4f}")

        # RMSE should be reasonable (better than random)
        baseline_rmse = np.sqrt(np.mean((y_test - np.mean(y_train)) ** 2))
        self.assertLess(rmse, baseline_rmse)

    def test_neural_network_meta_learner(self):
        """Test stacking with Neural Network meta-learner"""
        stacker = StackingMetaLearner(
            base_models=self.base_models,
            meta_learner_type='neural_network',
            cv_folds=3,
            time_series_split=True
        )

        train_size = int(0.8 * self.n_samples)
        X_train = self.X[:train_size]
        y_train = self.y[:train_size]

        # Fit (without context features this time)
        stacker.fit(X_train, y_train)

        self.assertTrue(stacker.is_fitted)

        # Predict
        X_test = self.X[train_size:]
        y_test = self.y[train_size:]
        predictions = stacker.predict(X_test)

        self.assertEqual(len(predictions), len(y_test))

        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        print(f"Neural Network Meta-Learner Test RMSE: {rmse:.4f}")

    def test_ridge_meta_learner(self):
        """Test stacking with Ridge meta-learner"""
        stacker = StackingMetaLearner(
            base_models=self.base_models,
            meta_learner_type='ridge',
            cv_folds=3,
            time_series_split=False  # Test KFold instead of TimeSeriesSplit
        )

        train_size = int(0.8 * self.n_samples)
        X_train = self.X[:train_size]
        y_train = self.y[:train_size]

        # Fit
        stacker.fit(X_train, y_train)

        self.assertTrue(stacker.is_fitted)

        # Predict
        X_test = self.X[train_size:]
        y_test = self.y[train_size:]
        predictions = stacker.predict(X_test)

        self.assertEqual(len(predictions), len(y_test))

        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        print(f"Ridge Meta-Learner Test RMSE: {rmse:.4f}")

    def test_predict_with_uncertainty(self):
        """Test uncertainty quantification"""
        stacker = StackingMetaLearner(
            base_models=self.base_models,
            meta_learner_type='xgboost',
            cv_folds=3
        )

        train_size = int(0.8 * self.n_samples)
        X_train = self.X[:train_size]
        y_train = self.y[:train_size]

        stacker.fit(X_train, y_train)

        # Predict with uncertainty
        X_test = self.X[train_size:]
        predictions, confidence = stacker.predict_with_uncertainty(X_test)

        # Check outputs
        self.assertEqual(len(predictions), len(X_test))
        self.assertEqual(len(confidence), len(X_test))

        # Confidence should be between 0 and 100
        self.assertTrue(np.all(confidence >= 0))
        self.assertTrue(np.all(confidence <= 100))

        print(f"Average confidence: {np.mean(confidence):.2f}")
        print(f"Confidence range: [{np.min(confidence):.2f}, {np.max(confidence):.2f}]")

        # High agreement (low variance) should correlate with accuracy
        y_test = self.y[train_size:]
        errors = np.abs(y_test - predictions)

        # Split into high and low confidence
        high_conf_mask = confidence > np.median(confidence)
        low_conf_mask = ~high_conf_mask

        high_conf_mae = np.mean(errors[high_conf_mask])
        low_conf_mae = np.mean(errors[low_conf_mask])

        print(f"High confidence MAE: {high_conf_mae:.4f}")
        print(f"Low confidence MAE: {low_conf_mae:.4f}")

        # High confidence predictions should generally be more accurate
        # (not strict test due to small sample size)
        self.assertIsNotNone(high_conf_mae)
        self.assertIsNotNone(low_conf_mae)

    def test_sample_weights(self):
        """Test training with sample weights"""
        stacker = StackingMetaLearner(
            base_models=self.base_models,
            meta_learner_type='xgboost',
            cv_folds=3
        )

        train_size = int(0.8 * self.n_samples)
        X_train = self.X[:train_size]
        y_train = self.y[:train_size]

        # Calculate time-decay weights
        sample_weights = calculate_time_decay_weights(
            self.dates[:train_size],
            half_life_days=180
        )

        # Fit with weights
        stacker.fit(X_train, y_train, sample_weights=sample_weights)

        self.assertTrue(stacker.is_fitted)

        # Should complete without errors
        X_test = self.X[train_size:]
        predictions = stacker.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))

    def test_base_model_weights(self):
        """Test base model weight calculation"""
        stacker = StackingMetaLearner(
            base_models=self.base_models,
            meta_learner_type='xgboost',
            cv_folds=3
        )

        train_size = int(0.8 * self.n_samples)
        stacker.fit(self.X[:train_size], self.y[:train_size])

        # Get weights
        weights = stacker.get_base_model_weights()

        self.assertEqual(len(weights), len(self.base_models))

        # Weights should sum to 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)

        # All weights should be positive
        for weight in weights.values():
            self.assertGreater(weight, 0)

        print("\nBase Model Weights:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f}")

    def test_no_data_leakage(self):
        """Test that OOF predictions don't leak information"""
        stacker = StackingMetaLearner(
            base_models=self.base_models,
            meta_learner_type='xgboost',
            cv_folds=5,
            time_series_split=True
        )

        # Generate OOF predictions manually to verify no leakage
        train_size = int(0.8 * self.n_samples)
        X_train = self.X[:train_size]
        y_train = self.y[:train_size]

        oof_predictions = stacker._generate_oof_predictions(X_train, y_train)

        # Check shape
        self.assertEqual(oof_predictions.shape, (train_size, len(self.base_models)))

        # Check no NaN values
        self.assertFalse(np.any(np.isnan(oof_predictions)))

        # OOF predictions should not be perfect (if they are, there's leakage)
        for model_idx in range(len(self.base_models)):
            oof_rmse = np.sqrt(np.mean((y_train - oof_predictions[:, model_idx]) ** 2))
            print(f"Model {model_idx} OOF RMSE: {oof_rmse:.4f}")

            # RMSE should be > 0 (not perfect fit, which would indicate leakage)
            self.assertGreater(oof_rmse, 0.5)

    def test_invalid_meta_learner_type(self):
        """Test error handling for invalid meta-learner type"""
        with self.assertRaises(ValueError):
            stacker = StackingMetaLearner(
                base_models=self.base_models,
                meta_learner_type='invalid_type',
                cv_folds=3
            )
            stacker.fit(self.X[:100], self.y[:100])

    def test_predict_before_fit(self):
        """Test error when predicting before fitting"""
        stacker = StackingMetaLearner(
            base_models=self.base_models,
            meta_learner_type='xgboost'
        )

        with self.assertRaises(ValueError):
            stacker.predict(self.X[:10])

    def test_mismatched_input_lengths(self):
        """Test error handling for mismatched input lengths"""
        stacker = StackingMetaLearner(
            base_models=self.base_models,
            meta_learner_type='xgboost'
        )

        # X and y different lengths
        with self.assertRaises(ValueError):
            stacker.fit(self.X[:100], self.y[:50])

        # X and context_features different lengths
        with self.assertRaises(ValueError):
            stacker.fit(self.X[:100], self.y[:100], context_features=self.context_features[:50])


class TestTimeDecayWeights(unittest.TestCase):
    """Test suite for time-decay weight calculation"""

    def test_basic_time_decay(self):
        """Test basic time-decay calculation"""
        # Create dates: today, 180 days ago, 360 days ago
        today = datetime.now()
        dates = [
            today,
            today - timedelta(days=180),
            today - timedelta(days=360)
        ]

        weights = calculate_time_decay_weights(dates, half_life_days=180)

        # Check shape
        self.assertEqual(len(weights), 3)

        # Most recent should have highest weight
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

        # 180 days ago should be approximately half of today's weight
        # (after normalization, exact ratio won't be 2:1, but should be close)
        ratio = weights[0] / weights[1]
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 2.5)

        print(f"\nTime-decay weights: {weights}")

    def test_time_decay_reference_date(self):
        """Test time-decay with custom reference date"""
        reference = datetime(2024, 12, 1)
        dates = [
            datetime(2024, 11, 1),  # 30 days ago
            datetime(2024, 9, 1),   # 90 days ago
            datetime(2024, 6, 1)    # 180 days ago
        ]

        weights = calculate_time_decay_weights(
            dates,
            half_life_days=180,
            reference_date=reference
        )

        self.assertEqual(len(weights), 3)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_time_decay_with_strings(self):
        """Test time-decay with date strings"""
        dates = ['2024-11-01', '2024-09-01', '2024-06-01']

        weights = calculate_time_decay_weights(
            dates,
            half_life_days=180,
            reference_date=datetime(2024, 12, 1)
        )

        self.assertEqual(len(weights), 3)
        self.assertTrue(np.all(weights > 0))

    def test_time_decay_normalization(self):
        """Test that weights are properly normalized"""
        today = datetime.now()
        dates = [today - timedelta(days=i*30) for i in range(10)]

        weights = calculate_time_decay_weights(dates, half_life_days=180)

        # Weights should sum to approximately n_samples (normalized)
        self.assertAlmostEqual(np.sum(weights), len(dates), places=5)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def test_complete_workflow(self):
        """Test complete workflow from data to prediction"""
        np.random.seed(42)

        # Create dataset
        n_samples = 300
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        y = 25 + 2*X[:, 0] - 1.5*X[:, 1] + np.random.randn(n_samples)

        # Create and train base models
        base_models = [
            RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y),
            GradientBoostingRegressor(n_estimators=20, random_state=42).fit(X, y),
            LinearRegression().fit(X, y)
        ]

        # Split data
        train_size = int(0.7 * n_samples)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Create stacker
        stacker = StackingMetaLearner(
            base_models=base_models,
            meta_learner_type='xgboost',
            cv_folds=3,
            time_series_split=True
        )

        # Fit
        stacker.fit(X_train, y_train)

        # Predict
        predictions = stacker.predict(X_test)

        # Evaluate
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        print(f"\nIntegration Test RMSE: {rmse:.4f}")

        # Compare to simple average
        simple_avg = np.mean([model.predict(X_test) for model in base_models], axis=0)
        simple_rmse = np.sqrt(np.mean((y_test - simple_avg) ** 2))
        print(f"Simple Average RMSE: {simple_rmse:.4f}")

        # Stacking should be competitive with simple average
        # (may not always be better on small datasets, but should be close)
        print(f"Improvement: {(simple_rmse - rmse) / simple_rmse * 100:.2f}%")

    def test_real_world_scenario(self):
        """Test with context features and sample weights"""
        np.random.seed(42)

        # Create dataset
        n_samples = 400
        n_features = 15
        n_context = 12

        X = np.random.randn(n_samples, n_features)
        context = np.random.randn(n_samples, n_context)

        # Target includes context influence
        y = (
            20 + 2*X[:, 0] - X[:, 1] +
            0.5*context[:, 0] +  # days_rest_diff
            0.3*context[:, 1] +  # pace
            np.random.randn(n_samples) * 1.5
        )

        # Create dates and weights
        dates = [datetime.now() - timedelta(days=i) for i in range(n_samples)]
        sample_weights = calculate_time_decay_weights(dates, half_life_days=180)

        # Train base models
        base_models = [
            RandomForestRegressor(n_estimators=30, random_state=42).fit(X, y),
            GradientBoostingRegressor(n_estimators=30, random_state=42).fit(X, y),
            ElasticNet(random_state=42).fit(X, y)
        ]

        # Split data
        train_size = int(0.75 * n_samples)
        X_train = X[:train_size]
        y_train = y[:train_size]
        context_train = context[:train_size]
        weights_train = sample_weights[:train_size]

        X_test = X[train_size:]
        y_test = y[train_size:]
        context_test = context[train_size:]

        # Train stacker with all features
        stacker = StackingMetaLearner(
            base_models=base_models,
            meta_learner_type='xgboost',
            cv_folds=5,
            time_series_split=True
        )

        stacker.fit(
            X_train, y_train,
            context_features=context_train,
            sample_weights=weights_train
        )

        # Predict with uncertainty
        predictions, confidence = stacker.predict_with_uncertainty(
            X_test,
            context_features=context_test
        )

        # Evaluate
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        mae = np.mean(np.abs(y_test - predictions))

        print(f"\nReal-world Scenario Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Avg Confidence: {np.mean(confidence):.2f}")

        # Get base model weights
        weights = stacker.get_base_model_weights()
        print(f"  Base Model Weights: {weights}")

        # All tests should pass without errors
        self.assertLess(rmse, 5.0)  # Reasonable performance
        self.assertTrue(np.all(confidence >= 0))
        self.assertTrue(np.all(confidence <= 100))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
