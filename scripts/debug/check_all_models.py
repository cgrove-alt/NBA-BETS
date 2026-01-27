"""
Check all prop type models for the same issue.
"""

import pickle
from pathlib import Path

MODEL_DIR = Path("models")
PROP_TYPES = ['points', 'rebounds', 'assists', 'threes', 'pra']

def check_all_prop_models():
    """Check which models would be loaded for each prop type."""
    print("="*70)
    print("CHECKING ALL PROP TYPE MODELS")
    print("="*70)

    issues_found = []

    for prop_type in PROP_TYPES:
        print(f"\n{prop_type.upper()}:")
        print("-" * 40)

        # Check in priority order
        stacking = MODEL_DIR / f"player_{prop_type}_stacking.pkl"
        ensemble = MODEL_DIR / f"player_{prop_type}_ensemble.pkl"
        legacy = MODEL_DIR / f"player_{prop_type}.pkl"

        loaded_model = None
        loaded_features = 0

        if stacking.exists():
            try:
                with open(stacking, 'rb') as f:
                    data = pickle.load(f)
                loaded_features = len(data.get('feature_names', []))
                loaded_model = "stacking"
                print(f"  ✓ Will load: {stacking.name} ({loaded_features} features)")
            except Exception as e:
                print(f"  ✗ Error loading stacking: {e}")

        elif ensemble.exists():
            try:
                with open(ensemble, 'rb') as f:
                    data = pickle.load(f)
                loaded_features = len(data.get('feature_names', []))
                loaded_model = "ensemble"
                print(f"  ✓ Will load: {ensemble.name} ({loaded_features} features)")
            except Exception as e:
                print(f"  ✗ Error loading ensemble: {e}")

        elif legacy.exists():
            try:
                with open(legacy, 'rb') as f:
                    data = pickle.load(f)
                loaded_features = len(data.get('feature_names', []))
                loaded_model = "legacy"
                print(f"  ✓ Will load: {legacy.name} ({loaded_features} features)")
            except Exception as e:
                print(f"  ✗ Error loading legacy: {e}")
        else:
            print("  ✗ NO MODEL FOUND!")
            issues_found.append(f"{prop_type}: No model found")
            continue

        # Check if there's a better model available
        if loaded_model == "stacking" and ensemble.exists():
            try:
                with open(ensemble, 'rb') as f:
                    ensemble_data = pickle.load(f)
                ensemble_features = len(ensemble_data.get('feature_names', []))

                if loaded_features < 20 and ensemble_features > 100:
                    print(f"  ⚠️  ISSUE: Stacking has {loaded_features} features, ensemble has {ensemble_features}")
                    print("      Stacking model likely incomplete!")
                    issues_found.append(
                        f"{prop_type}: Broken stacking model ({loaded_features} features) "
                        f"blocking ensemble ({ensemble_features} features)"
                    )
            except:
                pass

        # Sanity check
        if loaded_features < 20:
            print(f"  🔴 WARNING: Only {loaded_features} features - model may be incomplete")
            if f"{prop_type}: Broken stacking" not in str(issues_found):
                issues_found.append(f"{prop_type}: Very few features ({loaded_features})")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if issues_found:
        print("\n🔴 ISSUES FOUND:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("\n✅ All models look good!")

if __name__ == "__main__":
    check_all_prop_models()
