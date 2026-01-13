"""
Test script to verify StackingMetaLearner integration with EnsembleMoneylineModel
"""

import numpy as np
import pandas as pd
from model_trainer import EnsembleMoneylineModel

def test_ensemble_with_stacking():
    """Test EnsembleMoneylineModel with stacking enabled."""
    print("=" * 60)
    print("Testing EnsembleMoneylineModel with Stacking Integration")
    print("=" * 60)

    # Create synthetic training data
    np.random.seed(42)
    n_samples = 500
    n_features = 20

    # Generate random features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Generate binary labels (home team wins)
    y = np.random.randint(0, 2, n_samples)

    # Generate context features (12 features as per spec)
    context_features = np.random.randn(n_samples, 12)

    # Generate sample weights (time-decay)
    days_ago = np.arange(n_samples, 0, -1)
    sample_weights = 0.5 ** (days_ago / 180.0)

    print("\n1. Testing model initialization...")
    try:
        # Initialize with stacking enabled
        model = EnsembleMoneylineModel(use_stacking=True)
        print("✓ Model initialized with use_stacking=True")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

    print("\n2. Testing training WITHOUT context features (standard mode)...")
    try:
        metrics = model.train(X, y, test_size=0.2, use_time_series_cv=True)
        print(f"✓ Training completed (standard mode)")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - F1 Score: {metrics['f1']:.4f}")
        print(f"  - Using StackingMetaLearner: {metrics.get('using_stacking_meta_learner', False)}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False

    print("\n3. Testing training WITH context features (stacking mode)...")
    try:
        model2 = EnsembleMoneylineModel(use_stacking=True)
        metrics = model2.train(
            X, y,
            test_size=0.2,
            use_time_series_cv=True,
            context_features=context_features,
            sample_weights=sample_weights
        )
        print(f"✓ Training completed (stacking mode)")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - F1 Score: {metrics['f1']:.4f}")
        print(f"  - Using StackingMetaLearner: {metrics.get('using_stacking_meta_learner', False)}")
    except Exception as e:
        print(f"✗ Training with context failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n4. Testing prediction...")
    try:
        # Create a test feature dictionary
        test_features = {f"feature_{i}": float(np.random.randn()) for i in range(n_features)}

        # Predict without context
        pred1 = model.predict(test_features)
        print("✓ Prediction without context:")
        print(f"  - Home Win Prob: {pred1['home_win_probability']:.4f}")
        print(f"  - Predicted Winner: {pred1['predicted_winner']}")

        # Predict with context (if stacking is available)
        if model2.stacking_ensemble is not None:
            test_context = np.random.randn(1, 12)
            pred2 = model2.predict(test_features, context_features=test_context)
            print("✓ Prediction with context:")
            print(f"  - Home Win Prob: {pred2['home_win_probability']:.4f}")
            print(f"  - Predicted Winner: {pred2['predicted_winner']}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n5. Testing predict_with_confidence...")
    try:
        test_features = {f"feature_{i}": float(np.random.randn()) for i in range(n_features)}

        # Standard model
        pred, confidence = model.predict_with_confidence(test_features)
        print("✓ Confidence prediction (standard):")
        print(f"  - Home Win Prob: {pred['home_win_probability']:.4f}")
        print(f"  - Confidence Score: {confidence:.2f}/100")

        # Determine tier
        if confidence >= 90:
            tier = "Elite"
        elif confidence >= 75:
            tier = "Strong"
        elif confidence >= 60:
            tier = "Moderate"
        else:
            tier = "Weak"
        print(f"  - Edge Quality Tier: {tier}")

        # Stacking model (if available)
        if model2.stacking_ensemble is not None:
            test_context = np.random.randn(1, 12)
            pred2, confidence2 = model2.predict_with_confidence(test_features, context_features=test_context)
            print("✓ Confidence prediction (stacking):")
            print(f"  - Home Win Prob: {pred2['home_win_probability']:.4f}")
            print(f"  - Confidence Score: {confidence2:.2f}/100")

            if confidence2 >= 90:
                tier = "Elite"
            elif confidence2 >= 75:
                tier = "Strong"
            elif confidence2 >= 60:
                tier = "Moderate"
            else:
                tier = "Weak"
            print(f"  - Edge Quality Tier: {tier}")

    except Exception as e:
        print(f"✗ Confidence prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed! Stacking integration successful.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_ensemble_with_stacking()
    exit(0 if success else 1)
