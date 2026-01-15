"""
Diagnostic Script for Three-Point Model Investigation

This script analyzes why the threes model has negative R² (-0.568)
and provides actionable insights for fixing it.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Paths
MODEL_DIR = Path("models")
BACKTEST_FILE = Path("backtest_results_2025.json")

def load_model(model_path):
    """Load a pickled model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def analyze_model_structure(model_data, model_name):
    """Analyze the structure and components of a model."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {model_name}")
    print('='*60)

    if isinstance(model_data, dict):
        print(f"Model Type: Dictionary-based")
        print(f"Keys: {list(model_data.keys())}")

        # Check for ensemble structure
        if 'models' in model_data or 'base_models' in model_data:
            base_models = model_data.get('models') or model_data.get('base_models')
            print(f"\nBase Models ({len(base_models)}):")
            for name, model in base_models.items():
                print(f"  - {name}: {type(model).__name__}")

            if 'meta_model' in model_data:
                meta = model_data['meta_model']
                print(f"Meta Model: {type(meta).__name__ if meta else 'None (weighted average)'}")

            if 'model_weights' in model_data:
                print(f"\nModel Weights:")
                for name, weight in model_data.get('model_weights', {}).items():
                    print(f"  {name}: {weight:.4f}")

        # Feature information
        if 'feature_names' in model_data:
            features = model_data['feature_names']
            print(f"\nTotal Features: {len(features)}")

            # Categorize features
            three_pt_features = [f for f in features if 'fg3' in f.lower() or 'three' in f.lower()]
            shooting_features = [f for f in features if any(x in f.lower() for x in ['fg3', 'three', 'pct', 'makes', 'attempts'])]

            print(f"\n3-Point Specific Features ({len(three_pt_features)}):")
            for f in sorted(three_pt_features):
                print(f"  - {f}")

            print(f"\nShooting-Related Features ({len(shooting_features)}):")
            for f in sorted(shooting_features):
                print(f"  - {f}")

        # Check for feature importance
        if 'feature_importance' in model_data:
            print("\nFeature Importance Available: YES")
            importances = model_data['feature_importance']
            if isinstance(importances, dict):
                top_10 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
                print("Top 10 Features:")
                for feat, imp in top_10:
                    print(f"  {feat}: {imp:.4f}")
        else:
            print("\nFeature Importance Available: NO")
    else:
        print(f"Model Type: {type(model_data).__name__}")

def analyze_predictions_from_backtest():
    """Analyze three-point predictions from backtest results."""
    print(f"\n{'='*60}")
    print("ANALYZING BACKTEST PREDICTIONS")
    print('='*60)

    # Load training data to get actual predictions
    training_data_files = list(Path("training_data").glob("*.json"))
    if not training_data_files:
        print("No training data found for prediction analysis")
        return

    # Look for player prediction data
    player_data_file = Path("training_data/player_data_20251212_214916.json")
    if player_data_file.exists():
        with open(player_data_file) as f:
            player_data = json.load(f)

        if player_data:
            print(f"Player predictions found: {len(player_data)}")

            # Analyze three-point predictions
            threes_preds = []
            for record in player_data:
                if 'threes' in record or 'fg3m' in record:
                    threes_preds.append(record)

            print(f"Three-point predictions: {len(threes_preds)}")
        else:
            print("Player data file is empty")
    else:
        print(f"Player data file not found: {player_data_file}")

def calculate_baseline_metrics():
    """Calculate what 'predicting the mean' would give us."""
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON")
    print('='*60)

    # Simulate NBA three-point distribution
    # Average NBA player makes ~1.0 threes per game with high variance
    # Distribution is roughly: 0 (40%), 1 (25%), 2 (18%), 3 (10%), 4+ (7%)

    print("\nNBA Three-Point Distribution (typical):")
    print("  0 threes: ~40% of games")
    print("  1 three:  ~25% of games")
    print("  2 threes: ~18% of games")
    print("  3 threes: ~10% of games")
    print("  4+ threes: ~7% of games")
    print("\nMean: ~1.2 threes/game")
    print("Std Dev: ~1.4 threes")
    print("\nChallenge: Predicting 0 vs 1 vs 2+ is inherently difficult")
    print("because three-point shooting has high variance even for good shooters.")

def generate_recommendations():
    """Generate specific recommendations for fixing the threes model."""
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR FIXING THREES MODEL")
    print('='*60)

    print("\n1. FEATURE ENGINEERING IMPROVEMENTS:")
    print("   Current Issues:")
    print("   - May lack specialized 3PM features")
    print("   - Missing shooting context (volume, consistency, hot/cold streaks)")
    print("   - Not accounting for opponent 3P defense properly")
    print("\n   Add These Features:")
    print("   ✓ fg3a_per_min - Attempts per minute (accounts for playing time)")
    print("   ✓ fg3a_consistency - How consistent is attempt rate?")
    print("   ✓ regressed_fg3_pct - Bayesian-adjusted shooting %")
    print("   ✓ is_volume_shooter - Flag for players with 5+ attempts/game")
    print("   ✓ fg3_hot_streak - Made 40%+ in last 3 games")
    print("   ✓ fg3_cold_streak - Made <30% in last 3 games")
    print("   ✓ home_fg3_pct vs away_fg3_pct - Splits")
    print("   ✓ opp_fg3_pct_allowed - Opponent's 3P defense")
    print("   ✓ expected_fg3m - Attempts × Percentage = expected makes")

    print("\n2. MODEL ARCHITECTURE:")
    print("   Current: Ensemble of regressors")
    print("   Problem: Three-point makes are count data (0, 1, 2, 3...)")
    print("\n   Consider:")
    print("   ✓ Poisson Regression - Natural for count data")
    print("   ✓ Zero-Inflated Poisson - Handles many 0s")
    print("   ✓ Ordinal Regression - Treat as ordered categories")
    print("   ✓ Separate binary classifier: Will they make ANY threes?")
    print("     Then predict count GIVEN they make at least one")

    print("\n3. DATA QUALITY:")
    print("   ✓ Filter out players with <2 attempts/game in training")
    print("   ✓ Don't predict threes for non-shooters (centers, etc.)")
    print("   ✓ Weight recent games more heavily (3PM trends change fast)")

    print("\n4. PREDICTION BOUNDS:")
    print("   ✓ Clamp predictions to [0, 12] range")
    print("   ✓ Round to nearest 0.5 for realistic values")
    print("   ✓ Add uncertainty flags for low-volume shooters")

    print("\n5. EVALUATION STRATEGY:")
    print("   ✓ Don't just use R² - it's poor for count data")
    print("   ✓ Use MAE (Mean Absolute Error) as primary metric")
    print("   ✓ Track calibration: Do players predicted 2.0 average ~2.0?")
    print("   ✓ Separate metrics for volume shooters vs role players")

def main():
    """Main diagnostic routine."""
    print("="*60)
    print("THREE-POINT MODEL DIAGNOSTIC REPORT")
    print("="*60)
    print("\nGoal: Understand why threes R² = -0.568 (worse than baseline)")
    print("and provide actionable fixes.\n")

    # Analyze all threes models
    threes_models = [
        "player_threes.pkl",
        "player_threes_enhanced.pkl",
        "player_threes_ensemble.pkl",
        "player_threes_stacking.pkl",
        "player_threes_quantile.pkl"
    ]

    for model_file in threes_models:
        model_path = MODEL_DIR / model_file
        if model_path.exists():
            try:
                model_data = load_model(model_path)
                analyze_model_structure(model_data, model_file)
            except Exception as e:
                print(f"\nERROR loading {model_file}: {e}")
        else:
            print(f"\n{model_file}: NOT FOUND")

    # Analyze predictions
    analyze_predictions_from_backtest()

    # Baseline comparison
    calculate_baseline_metrics()

    # Generate recommendations
    generate_recommendations()

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print('='*60)
    print("\n1. Review feature list in models (see above)")
    print("2. Check if specialized 3PM features exist")
    print("3. If missing, implement new feature engineering")
    print("4. Consider Poisson regression for count data")
    print("5. Retrain with improved features")
    print("6. Re-run backtest to validate improvements")
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
