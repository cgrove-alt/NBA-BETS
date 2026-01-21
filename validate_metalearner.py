#!/usr/bin/env python3
"""
Meta-Learner A/B Test: Stacking vs Weighted Averaging
Validates if meta-learner upgrade improves RMSE over baseline.
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

def load_model(model_path):
    """Load a trained model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def compare_prediction_methods(model, test_predictions):
    """
    Compare meta-learner stacking vs weighted averaging.

    Args:
        model: Loaded ensemble model
        test_predictions: Dict with actual values and base predictions

    Returns:
        dict: Metrics for both methods
    """
    results = {
        'meta_learner': {'predictions': [], 'actuals': []},
        'weighted_avg': {'predictions': [], 'actuals': []}
    }

    for item in test_predictions:
        actual = item['actual']
        base_preds = item['base_predictions']  # List of predictions from base models

        # Method 1: Meta-learner stacking (current)
        if hasattr(model, 'meta_model') and model.meta_model is not None:
            stacked = np.array([base_preds])
            meta_pred = float(model.meta_model.predict(stacked)[0])
        else:
            meta_pred = float(np.mean(base_preds))

        # Method 2: Weighted averaging (baseline)
        if hasattr(model, 'model_weights') and model.model_weights:
            weighted_pred = 0.0
            for i, pred in enumerate(base_preds):
                weight = list(model.model_weights.values())[i] if i < len(model.model_weights) else 1.0/len(base_preds)
                weighted_pred += weight * pred
        else:
            weighted_pred = float(np.mean(base_preds))

        results['meta_learner']['predictions'].append(meta_pred)
        results['meta_learner']['actuals'].append(actual)
        results['weighted_avg']['predictions'].append(weighted_pred)
        results['weighted_avg']['actuals'].append(actual)

    # Calculate metrics
    metrics = {}
    for method in ['meta_learner', 'weighted_avg']:
        preds = np.array(results[method]['predictions'])
        actuals = np.array(results[method]['actuals'])

        metrics[method] = {
            'rmse': float(np.sqrt(mean_squared_error(actuals, preds))),
            'mae': float(mean_absolute_error(actuals, preds)),
            'r2': float(r2_score(actuals, preds)),
            'bias': float(np.mean(preds - actuals))
        }

    # Calculate improvement
    rmse_improvement = ((metrics['weighted_avg']['rmse'] - metrics['meta_learner']['rmse'])
                        / metrics['weighted_avg']['rmse'] * 100)

    return {
        'metrics': metrics,
        'rmse_improvement_pct': rmse_improvement,
        'sample_size': len(test_predictions)
    }

def main():
    print("=" * 70)
    print("META-LEARNER A/B TEST: Stacking vs Weighted Averaging")
    print("=" * 70)

    # Load backtest results with base predictions
    backtest_file = 'backtest_results_2025_quick.json'

    if not Path(backtest_file).exists():
        print(f"ERROR: {backtest_file} not found")
        print("Run comprehensive_backtest.py first to generate test data")
        sys.exit(1)

    with open(backtest_file, 'r') as f:
        backtest_data = json.load(f)

    if 'predictions' not in backtest_data or len(backtest_data['predictions']) == 0:
        print("ERROR: No predictions found in backtest file")
        sys.exit(1)

    print(f"\nLoaded {len(backtest_data['predictions'])} predictions from backtest")

    # Load models for each prop type
    prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']

    all_results = {}

    for prop_type in prop_types:
        model_path = Path(f"models/prop_{prop_type}.pkl")

        if not model_path.exists():
            print(f"\n⚠️  Skipping {prop_type}: Model not found")
            continue

        print(f"\n--- Testing {prop_type.upper()} ---")

        # Load model
        model = load_model(model_path)

        # Filter predictions for this prop type
        prop_predictions = [p for p in backtest_data['predictions']
                           if p.get('prop_type', '').lower() == prop_type]

        if len(prop_predictions) == 0:
            print(f"  No predictions found for {prop_type}")
            continue

        # Compare methods
        results = compare_prediction_methods(model, prop_predictions)
        all_results[prop_type] = results

        # Print results
        print(f"  Sample size: {results['sample_size']}")
        print(f"\n  Meta-Learner (Stacking):")
        print(f"    RMSE: {results['metrics']['meta_learner']['rmse']:.3f}")
        print(f"    MAE:  {results['metrics']['meta_learner']['mae']:.3f}")
        print(f"    R²:   {results['metrics']['meta_learner']['r2']:.3f}")

        print(f"\n  Weighted Averaging (Baseline):")
        print(f"    RMSE: {results['metrics']['weighted_avg']['rmse']:.3f}")
        print(f"    MAE:  {results['metrics']['weighted_avg']['mae']:.3f}")
        print(f"    R²:   {results['metrics']['weighted_avg']['r2']:.3f}")

        improvement = results['rmse_improvement_pct']
        symbol = "✅" if improvement > 0 else "❌"
        print(f"\n  {symbol} RMSE Improvement: {improvement:+.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    improvements = [r['rmse_improvement_pct'] for r in all_results.values()]
    avg_improvement = np.mean(improvements) if improvements else 0

    print(f"\nAverage RMSE Improvement: {avg_improvement:+.2f}%")
    print(f"Props tested: {len(all_results)}")

    if avg_improvement > 2.0:
        print("\n✅ META-LEARNER IS VALIDATED - Keep stacking")
        print("   Improvement exceeds 2% threshold")
    elif avg_improvement > 0:
        print("\n⚠️  META-LEARNER SHOWS MARGINAL IMPROVEMENT")
        print(f"   {avg_improvement:.2f}% < 2% threshold - Consider reverting")
    else:
        print("\n❌ META-LEARNER IS WORSE THAN BASELINE")
        print("   REVERT to weighted averaging immediately")

    # Save results
    output_file = 'metalearner_validation.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()
