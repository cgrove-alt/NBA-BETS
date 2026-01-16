"""
FIX #2: Recalculate Proper Bias Corrections
============================================

After DNP fix, recalculate bias corrections from the full dataset.

Previous issue:
- Corrections based on 100-sample subset
- Overall bias still -1.174 after first correction
- Points overcorrected (+0.46), PRA severely under (-2.97)

New approach:
- Use FULL backtest results (all predictions)
- Calculate prop-specific bias accurately
- Apply iterative correction if needed
- Target: Overall bias < |0.5|, per-prop bias < |0.5|
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List

def load_backtest_results(results_file: str = "backtest_results_2025.json") -> Dict:
    """Load backtest results from JSON file."""
    path = Path(results_file)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(path, 'r') as f:
        data = json.load(f)

    return data


def calculate_bias_from_raw_predictions(predictions: List[Dict]) -> Dict:
    """
    Calculate prop-specific bias from raw predictions.

    Bias = mean(predicted - actual)
    Correction = -bias (to offset the bias)
    """
    by_prop = defaultdict(list)

    for pred in predictions:
        prop_type = pred['prop_type']
        predicted = pred['predicted']
        actual = pred['actual']
        error = predicted - actual

        by_prop[prop_type].append({
            'error': error,
            'predicted': predicted,
            'actual': actual
        })

    results = {}
    for prop_type, pred_list in sorted(by_prop.items()):
        errors = [p['error'] for p in pred_list]

        bias = np.mean(errors)
        mae = np.mean([abs(e) for e in errors])
        rmse = np.sqrt(np.mean([e**2 for e in errors]))

        # Correction is negative of bias
        correction = -bias

        results[prop_type] = {
            'count': len(pred_list),
            'current_bias': round(bias, 3),
            'mae': round(mae, 3),
            'rmse': round(rmse, 3),
            'recommended_correction': round(correction, 3)
        }

    # Calculate overall
    all_errors = []
    for pred_list in by_prop.values():
        all_errors.extend([p['error'] for p in pred_list])

    results['overall'] = {
        'count': len(all_errors),
        'current_bias': round(np.mean(all_errors), 3),
        'mae': round(np.mean([abs(e) for e in all_errors]), 3),
        'rmse': round(np.sqrt(np.mean([e**2 for e in all_errors])), 3)
    }

    return results


def check_if_raw_predictions_available(results_file: str = "backtest_results_2025.json") -> bool:
    """Check if raw predictions are available in the results file."""
    data = load_backtest_results(results_file)
    return 'raw_predictions' in data or 'predictions' in data


def extract_raw_predictions(results_file: str = "backtest_results_2025.json") -> List[Dict]:
    """Extract raw prediction data from backtest results."""
    data = load_backtest_results(results_file)

    # Check for raw predictions
    if 'raw_predictions' in data:
        return data['raw_predictions']

    if 'predictions' in data:
        return data['predictions']

    # If not available, need to reconstruct from summary stats
    # This is less accurate but can work as fallback
    print("⚠️  Warning: Raw predictions not available. Using summary statistics.")
    print("   This is less accurate. Consider re-running backtest with raw prediction export.")

    return None


def generate_bias_correction_code(corrections: Dict) -> str:
    """Generate Python code for BIAS_CORRECTIONS dict."""
    lines = ["    BIAS_CORRECTIONS = {"]

    for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
        if prop_type in corrections:
            corr = corrections[prop_type]['recommended_correction']
            bias = corrections[prop_type]['current_bias']
            lines.append(f"        '{prop_type}': {corr:.3f},  # Fix bias of {bias:.3f}")

    lines.append("    }")

    return '\n'.join(lines)


def main():
    print("="*60)
    print("FIX #2: RECALCULATE BIAS CORRECTIONS")
    print("="*60)

    # Load backtest results
    results_file = "backtest_results_2025.json"
    print(f"\nLoading results from: {results_file}")

    if not Path(results_file).exists():
        print(f"❌ Error: {results_file} not found!")
        print("   Run comprehensive_backtest.py first to generate results.")
        return

    # Check if raw predictions available
    has_raw = check_if_raw_predictions_available(results_file)

    if not has_raw:
        print("\n❌ Error: Raw predictions not found in results file!")
        print("   The backtest results file needs to include raw prediction data.")
        print("   This was likely stripped to save space.")
        print("\n   Options:")
        print("   1. Re-run comprehensive_backtest.py (will regenerate with raw data)")
        print("   2. Use phase2_backtest.json which has raw predictions")

        # Try fallback to phase2_backtest.json
        fallback_file = "backtest_results/phase2_backtest.json"
        if Path(fallback_file).exists():
            print(f"\n   Using fallback: {fallback_file}")
            results_file = fallback_file
        else:
            return

    # Extract raw predictions
    print("\nExtracting raw predictions...")
    raw_preds = extract_raw_predictions(results_file)

    if not raw_preds:
        print("❌ Failed to extract predictions")
        return

    print(f"  Loaded {len(raw_preds)} predictions")

    # Calculate bias
    print("\nCalculating prop-specific bias...")
    bias_analysis = calculate_bias_from_raw_predictions(raw_preds)

    # Display results
    print("\n" + "="*60)
    print("BIAS ANALYSIS RESULTS")
    print("="*60)

    print(f"\nOVERALL:")
    overall = bias_analysis['overall']
    print(f"  Predictions: {overall['count']}")
    print(f"  Bias:        {overall['current_bias']:+.3f} {'✅ MET' if abs(overall['current_bias']) < 0.5 else '❌ NOT MET'} (target: <|0.5|)")
    print(f"  MAE:         {overall['mae']:.3f}")
    print(f"  RMSE:        {overall['rmse']:.3f}")

    print(f"\nPER-PROP TYPE:")
    for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
        if prop_type not in bias_analysis:
            continue

        stats = bias_analysis[prop_type]
        status = "✅ MET" if abs(stats['current_bias']) < 0.5 else "❌ NOT MET"
        print(f"\n  {prop_type.upper()}:")
        print(f"    Count:      {stats['count']}")
        print(f"    Bias:       {stats['current_bias']:+.3f} {status}")
        print(f"    Correction: {stats['recommended_correction']:+.3f}")
        print(f"    RMSE:       {stats['rmse']:.3f}")

    # Generate correction code
    print("\n" + "="*60)
    print("RECOMMENDED BIAS_CORRECTIONS CODE")
    print("="*60)
    print()
    code = generate_bias_correction_code(bias_analysis)
    print(code)

    # Save results
    output_file = Path("backtest_results/fix2_bias_corrections.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(bias_analysis, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Check if corrections meet targets
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    issues = []

    if abs(overall['current_bias']) >= 0.5:
        issues.append(f"Overall bias {overall['current_bias']:+.3f} >= 0.5")

    for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
        if prop_type not in bias_analysis:
            continue
        stats = bias_analysis[prop_type]
        if abs(stats['current_bias']) >= 0.5:
            issues.append(f"{prop_type} bias {stats['current_bias']:+.3f} >= 0.5")

    if issues:
        print("\n❌ TARGETS NOT MET:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n⚠️  Note: These corrections assume bias is LINEAR.")
        print("   If bias is non-linear (e.g., varies by player tier), may need")
        print("   more sophisticated correction (quantile-based, player-specific, etc.)")
    else:
        print("\n✅ ALL TARGETS MET!")
        print("   Apply the BIAS_CORRECTIONS code above to comprehensive_backtest.py")


if __name__ == "__main__":
    main()
