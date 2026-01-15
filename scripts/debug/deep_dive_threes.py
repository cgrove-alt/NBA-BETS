"""
Deep Dive: Which Model Is Actually Being Used?

The comprehensive_backtest.py loads models in this order:
1. player_{prop}_stacking.pkl (highest priority)
2. player_{prop}_ensemble.pkl (fallback)
3. player_{prop}.pkl (legacy fallback)

Let's verify which threes model is loaded and why it's failing.
"""

import pickle
from pathlib import Path

MODEL_DIR = Path("models")

def check_model_priority():
    """Check which model would be loaded for threes."""
    print("="*60)
    print("MODEL LOADING PRIORITY CHECK")
    print("="*60)

    prop_type = "threes"

    # Check in priority order
    models_to_check = [
        f"player_{prop_type}_stacking.pkl",
        f"player_{prop_type}_ensemble.pkl",
        f"player_{prop_type}.pkl"
    ]

    for i, model_file in enumerate(models_to_check, 1):
        model_path = MODEL_DIR / model_file
        exists = model_path.exists()
        size = model_path.stat().st_size / 1024 if exists else 0

        status = "✓ EXISTS" if exists else "✗ MISSING"
        priority = f"Priority {i}"

        print(f"\n{priority}: {model_file}")
        print(f"  Status: {status}")
        if exists:
            print(f"  Size: {size:.1f} KB")

            # Load and analyze
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                if isinstance(model_data, dict) and 'feature_names' in model_data:
                    num_features = len(model_data['feature_names'])
                    print(f"  Features: {num_features}")

                    # Check for 3PM-specific features
                    fg3_features = [f for f in model_data['feature_names']
                                   if 'fg3' in f.lower() or 'three' in f.lower()]
                    print(f"  3PM-specific features: {len(fg3_features)}")

                    if i == 1 and exists:
                        print(f"\n  ⚠️  THIS MODEL WILL BE LOADED BY comprehensive_backtest.py")
                        print(f"      Feature count: {num_features} (should be 100+)")
                        if num_features < 20:
                            print(f"      🔴 CRITICAL: Very few features! Model likely broken.")
            except Exception as e:
                print(f"  Error loading: {e}")

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    stacking_path = MODEL_DIR / "player_threes_stacking.pkl"
    ensemble_path = MODEL_DIR / "player_threes_ensemble.pkl"

    if stacking_path.exists():
        with open(stacking_path, 'rb') as f:
            stacking_data = pickle.load(f)

        stacking_features = len(stacking_data.get('feature_names', []))

        print(f"\n✗ PROBLEM IDENTIFIED:")
        print(f"  player_threes_stacking.pkl exists but only has {stacking_features} features")
        print(f"  This model takes priority over player_threes_ensemble.pkl (150 features)")
        print(f"\n💡 SOLUTION OPTIONS:")
        print(f"  Option A: Delete player_threes_stacking.pkl to use ensemble")
        print(f"  Option B: Retrain stacking model with proper features")
        print(f"  Option C: Rename ensemble to stacking after validation")

        if ensemble_path.exists():
            with open(ensemble_path, 'rb') as f:
                ensemble_data = pickle.load(f)
            ensemble_features = len(ensemble_data.get('feature_names', []))

            print(f"\n📊 MODEL COMPARISON:")
            print(f"  Stacking: {stacking_features} features (LOADED BY BACKTEST)")
            print(f"  Ensemble: {ensemble_features} features (IGNORED)")
            print(f"\n  The ensemble model has {ensemble_features - stacking_features} more features!")
            print(f"  This likely explains the negative R².")

if __name__ == "__main__":
    check_model_priority()
