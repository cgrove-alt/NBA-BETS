"""
Integration tests for StackingMetaLearner with all model types.

Tests that verify the complete integration of StackingMetaLearner
across all prediction models: Moneyline, Spread, Player Props, and Quantile Props.
"""

import numpy as np
import pandas as pd
from model_trainer import (
    EnsembleMoneylineModel,
    LightGBMSpreadModel,
    PlayerPropModel,
    QuantilePropModel,
    HAS_STACKING_META_LEARNER,
)


def generate_context_features(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic context features (12 features)."""
    np.random.seed(seed)
    return np.random.randn(n_samples, 12)


def generate_sample_weights(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic sample weights with time decay."""
    np.random.seed(seed)
    # Simulate time decay: more recent samples get higher weights
    weights = np.exp(np.linspace(-2, 0, n_samples))
    return weights / weights.sum() * n_samples


def test_ensemble_moneyline_with_stacking():
    """Test EnsembleMoneylineModel with StackingMetaLearner."""
    print("\n" + "="*80)
    print("TEST 1: EnsembleMoneylineModel with StackingMetaLearner")
    print("="*80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 25

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f"feature_{i}" for i in range(n_features)])
    # Binary classification: home win (1) or away win (0)
    y = np.random.randint(0, 2, n_samples)

    context_features = generate_context_features(n_samples)
    sample_weights = generate_sample_weights(n_samples)

    # Test with stacking
    model = EnsembleMoneylineModel(use_stacking=True)

    if not HAS_STACKING_META_LEARNER:
        print("⚠️  StackingMetaLearner not available - skipping stacking test")
        return

    metrics = model.train(
        X, y,
        test_size=0.2,
        cv_folds=3,
        use_time_series_cv=True,
        context_features=context_features,
        sample_weights=sample_weights
    )

    print("\n✓ Training completed successfully")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Using StackingMetaLearner: {metrics.get('using_stacking_meta_learner', False)}")

    # Test prediction
    test_features = {f"feature_{i}": np.random.randn() for i in range(n_features)}
    test_context = generate_context_features(1)

    prediction = model.predict(test_features, context_features=test_context)
    print("\n✓ Prediction successful")
    print(f"  Home win probability: {prediction['home_win_probability']:.4f}")
    print(f"  Away win probability: {prediction['away_win_probability']:.4f}")

    # Test predict_with_confidence
    prediction_conf, confidence = model.predict_with_confidence(test_features, context_features=test_context)
    print("\n✓ Confidence prediction successful")
    print(f"  Confidence score: {confidence:.2f}/100")

    assert model.stacking_ensemble is not None, "Stacking ensemble should be initialized"
    assert metrics['using_stacking_meta_learner'] is True, "Should be using StackingMetaLearner"
    print("\n✅ EnsembleMoneylineModel test PASSED")


def test_spread_model_with_stacking():
    """Test LightGBMSpreadModel with StackingMetaLearner."""
    print("\n" + "="*80)
    print("TEST 2: LightGBMSpreadModel with StackingMetaLearner")
    print("="*80)

    # Generate synthetic data
    np.random.seed(43)
    n_samples = 500
    n_features = 25

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f"feature_{i}" for i in range(n_features)])
    # Regression: spread differential (-30 to +30)
    y = np.random.randn(n_samples) * 10

    context_features = generate_context_features(n_samples)
    sample_weights = generate_sample_weights(n_samples)

    # Test with stacking
    model = LightGBMSpreadModel(use_stacking=True)

    if not HAS_STACKING_META_LEARNER:
        print("⚠️  StackingMetaLearner not available - skipping stacking test")
        return

    metrics = model.train(
        X, y,
        test_size=0.2,
        cv_folds=3,
        use_time_series_cv=True,
        context_features=context_features,
        sample_weights=sample_weights
    )

    print("\n✓ Training completed successfully")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Using StackingMetaLearner: {metrics.get('using_stacking_meta_learner', False)}")

    # Test prediction
    test_features = {f"feature_{i}": np.random.randn() for i in range(n_features)}
    test_context = generate_context_features(1)

    prediction = model.predict(test_features, spread_line=-5.5, context_features=test_context)
    print("\n✓ Prediction successful")
    print(f"  Predicted spread: {prediction['predicted_spread']:.2f}")
    print(f"  Covers spread: {prediction['covers_spread']}")
    print(f"  Edge: {prediction['edge']:.2f}")

    assert model.stacking_ensemble is not None, "Stacking ensemble should be initialized"
    assert metrics['using_stacking_meta_learner'] is True, "Should be using StackingMetaLearner"
    print("\n✅ LightGBMSpreadModel test PASSED")


def test_player_prop_regressor_with_stacking():
    """Test PlayerPropModel (regression) with StackingMetaLearner."""
    print("\n" + "="*80)
    print("TEST 3: PlayerPropModel (Regression) with StackingMetaLearner")
    print("="*80)

    # Generate synthetic data
    np.random.seed(44)
    n_samples = 400
    n_features = 20

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f"feature_{i}" for i in range(n_features)])
    # Regression: player points (0-50)
    y = np.random.poisson(25, n_samples).astype(float)

    context_features = generate_context_features(n_samples)
    sample_weights = generate_sample_weights(n_samples)

    # Test with stacking
    model = PlayerPropModel(prop_type="points", use_classifier=False, use_stacking=True)

    if not HAS_STACKING_META_LEARNER:
        print("⚠️  StackingMetaLearner not available - skipping stacking test")
        return

    metrics = model.train(
        X, y,
        test_size=0.2,
        cv_folds=3,
        use_time_series_cv=True,
        context_features=context_features,
        sample_weights=sample_weights
    )

    print("\n✓ Training completed successfully")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Using StackingMetaLearner: {metrics.get('using_stacking_meta_learner', False)}")

    # Test prediction
    test_features = {f"feature_{i}": np.random.randn() for i in range(n_features)}
    test_context = generate_context_features(1)

    prediction = model.predict(test_features, prop_line=24.5, context_features=test_context)
    print("\n✓ Prediction successful")
    print(f"  Predicted value: {prediction['predicted_value']:.2f}")
    print(f"  Prediction: {prediction['prediction']}")
    print(f"  Edge: {prediction['edge']:.2f}")

    assert model.stacking_ensemble is not None, "Stacking ensemble should be initialized"
    assert metrics['using_stacking_meta_learner'] is True, "Should be using StackingMetaLearner"
    print("\n✅ PlayerPropModel (Regression) test PASSED")


def test_player_prop_classifier_with_stacking():
    """Test PlayerPropModel (classification) with StackingMetaLearner."""
    print("\n" + "="*80)
    print("TEST 4: PlayerPropModel (Classification) with StackingMetaLearner")
    print("="*80)

    # Generate synthetic data
    np.random.seed(45)
    n_samples = 400
    n_features = 20

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f"feature_{i}" for i in range(n_features)])
    # Binary classification: over (1) or under (0)
    y = np.random.randint(0, 2, n_samples)

    context_features = generate_context_features(n_samples)
    sample_weights = generate_sample_weights(n_samples)

    # Test with stacking
    model = PlayerPropModel(prop_type="points", use_classifier=True, use_stacking=True)

    if not HAS_STACKING_META_LEARNER:
        print("⚠️  StackingMetaLearner not available - skipping stacking test")
        return

    metrics = model.train(
        X, y,
        test_size=0.2,
        cv_folds=3,
        use_time_series_cv=True,
        context_features=context_features,
        sample_weights=sample_weights
    )

    print("\n✓ Training completed successfully")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Using StackingMetaLearner: {metrics.get('using_stacking_meta_learner', False)}")

    # Test prediction
    test_features = {f"feature_{i}": np.random.randn() for i in range(n_features)}
    test_context = generate_context_features(1)

    prediction = model.predict(test_features, context_features=test_context)
    print("\n✓ Prediction successful")
    print(f"  Over probability: {prediction['over_probability']:.4f}")
    print(f"  Under probability: {prediction['under_probability']:.4f}")
    print(f"  Prediction: {prediction['prediction']}")

    assert model.stacking_ensemble is not None, "Stacking ensemble should be initialized"
    assert metrics['using_stacking_meta_learner'] is True, "Should be using StackingMetaLearner"
    print("\n✅ PlayerPropModel (Classification) test PASSED")


def test_quantile_prop_model_with_stacking():
    """Test QuantilePropModel with StackingMetaLearner."""
    print("\n" + "="*80)
    print("TEST 5: QuantilePropModel with StackingMetaLearner")
    print("="*80)

    # Generate synthetic data
    np.random.seed(46)
    n_samples = 400
    n_features = 20

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f"feature_{i}" for i in range(n_features)])
    # Regression: player points (0-50)
    y = np.random.poisson(25, n_samples).astype(float)

    context_features = generate_context_features(n_samples)
    sample_weights = generate_sample_weights(n_samples)

    # Test with stacking
    model = QuantilePropModel(prop_type="points", use_stacking=True)

    if not HAS_STACKING_META_LEARNER:
        print("⚠️  StackingMetaLearner not available - skipping stacking test")
        return

    metrics = model.train(
        X, y,
        test_size=0.2,
        cv_folds=3,
        use_time_series_cv=True,
        context_features=context_features,
        sample_weights=sample_weights
    )

    print("\n✓ Training completed successfully")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Quantile crossings: {metrics['quantile_crossings']}")
    print(f"  Using StackingMetaLearner: {metrics.get('using_stacking_meta_learner', False)}")

    # Test prediction
    test_features = {f"feature_{i}": np.random.randn() for i in range(n_features)}
    test_context = generate_context_features(1)

    prediction = model.predict(test_features, prop_line=24.5, context_features=test_context)
    print("\n✓ Prediction successful")
    print(f"  Predicted value (q50): {prediction['predicted_value']:.2f}")
    print(f"  Q45: {prediction['q45']:.2f}")
    print(f"  Q50: {prediction['q50']:.2f}")
    print(f"  Q55: {prediction['q55']:.2f}")
    print(f"  Prediction spread: {prediction['prediction_spread']:.2f}")
    print(f"  Over probability: {prediction['over_probability']:.4f}")

    assert len(model.stacking_ensembles) > 0, "Stacking ensemble should be initialized"
    assert metrics['using_stacking_meta_learner'] is True, "Should be using StackingMetaLearner"
    print("\n✅ QuantilePropModel test PASSED")


def test_backward_compatibility():
    """Test that models work without stacking (backward compatibility)."""
    print("\n" + "="*80)
    print("TEST 6: Backward Compatibility (No Stacking)")
    print("="*80)

    # Generate synthetic data
    np.random.seed(47)
    n_samples = 200
    n_features = 20

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f"feature_{i}" for i in range(n_features)])
    y_classification = np.random.randint(0, 2, n_samples)
    y_regression = np.random.randn(n_samples) * 10

    # Test models WITHOUT stacking (should use standard sklearn models)
    model1 = EnsembleMoneylineModel(use_stacking=False)
    model1.train(X, y_classification, test_size=0.2, cv_folds=3)
    assert model1.stacking_ensemble is None, "Should not use stacking"
    print("✓ EnsembleMoneylineModel works without stacking")

    model2 = LightGBMSpreadModel(use_stacking=False)
    model2.train(X, y_regression, test_size=0.2, cv_folds=3)
    assert model2.stacking_ensemble is None, "Should not use stacking"
    print("✓ LightGBMSpreadModel works without stacking")

    model3 = PlayerPropModel(use_stacking=False)
    model3.train(X, y_regression, test_size=0.2, cv_folds=3)
    assert model3.stacking_ensemble is None, "Should not use stacking"
    print("✓ PlayerPropModel works without stacking")

    model4 = QuantilePropModel(use_stacking=False)
    model4.train(X, y_regression, test_size=0.2, cv_folds=3)
    assert len(model4.stacking_ensembles) == 0, "Should not use stacking"
    print("✓ QuantilePropModel works without stacking")

    print("\n✅ Backward compatibility test PASSED")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("STACKING META-LEARNER INTEGRATION TEST SUITE")
    print("="*80)

    if not HAS_STACKING_META_LEARNER:
        print("\n❌ ERROR: StackingMetaLearner not available!")
        print("Please ensure stacking_meta_learner.py is in the same directory.")
        exit(1)

    try:
        test_ensemble_moneyline_with_stacking()
        test_spread_model_with_stacking()
        test_player_prop_regressor_with_stacking()
        test_player_prop_classifier_with_stacking()
        test_quantile_prop_model_with_stacking()
        test_backward_compatibility()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✅")
        print("="*80)
        print("\nSummary:")
        print("  ✓ EnsembleMoneylineModel with StackingMetaLearner")
        print("  ✓ LightGBMSpreadModel with StackingMetaLearner")
        print("  ✓ PlayerPropModel (Regression) with StackingMetaLearner")
        print("  ✓ PlayerPropModel (Classification) with StackingMetaLearner")
        print("  ✓ QuantilePropModel with StackingMetaLearner")
        print("  ✓ Backward compatibility without stacking")
        print("\nTask 1.5 is complete and ready for validation!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
