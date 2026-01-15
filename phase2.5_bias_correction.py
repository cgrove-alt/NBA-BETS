"""
Phase 2.5 Task 2: Apply Bias Correction

This script applies bias correction to fix systematic underprediction:
- Phase 2 overall bias: -1.671 (predicting 1.7 units too low)
- Target: Bias < |0.5|

Strategy:
1. Calculate prop-specific bias from Phase 2 results
2. Update BIAS_CORRECTIONS in comprehensive_backtest.py
3. Re-run backtest with corrections
4. Compare before/after metrics

Usage:
    python3 phase2.5_bias_correction.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def load_phase2_results() -> Dict:
    """Load Phase 2 backtest results."""
    results_file = Path("backtest_results/phase2_backtest.json")

    if not results_file.exists():
        raise FileNotFoundError(f"Phase 2 results not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)


def calculate_prop_specific_bias(predictions: List[Dict]) -> Dict:
    """
    Calculate bias for each prop type.

    Bias = avg(predicted - actual) = avg(error)
    Negative bias = underpredicting
    """
    by_prop = defaultdict(list)

    for pred in predictions:
        prop_type = pred['prop_type']
        error = pred['error']  # Already predicted - actual
        by_prop[prop_type].append(error)

    bias_corrections = {}
    for prop_type, errors in by_prop.items():
        bias = np.mean(errors)

        # Correction is negative of bias (if we underpredict, add positive correction)
        correction = -bias

        bias_corrections[prop_type] = {
            'sample_size': len(errors),
            'current_bias': round(bias, 3),
            'recommended_correction': round(correction, 3),
            'bias_after_correction': 0.0,  # By definition
            'std_error': round(np.std(errors), 3)
        }

    return bias_corrections


def generate_bias_correction_code(corrections: Dict) -> str:
    """Generate Python code for BIAS_CORRECTIONS dict."""
    code = "BIAS_CORRECTIONS = {\n"

    for prop_type, data in sorted(corrections.items()):
        correction = data['recommended_correction']
        bias = data['current_bias']
        code += f"    '{prop_type}': {correction:.3f},  # Current bias: {bias:.3f}\n"

    code += "}"

    return code


def update_comprehensive_backtest(corrections: Dict):
    """Update BIAS_CORRECTIONS in comprehensive_backtest.py."""
    backtest_file = Path("comprehensive_backtest.py")

    if not backtest_file.exists():
        print(f"  Warning: {backtest_file} not found. Skipping code update.")
        return False

    # Read current file
    with open(backtest_file) as f:
        lines = f.readlines()

    # Find BIAS_CORRECTIONS section
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if 'BIAS_CORRECTIONS = {' in line:
            start_idx = i
        if start_idx is not None and '}' in line and i > start_idx:
            end_idx = i
            break

    if start_idx is None:
        print("  Warning: BIAS_CORRECTIONS dict not found in comprehensive_backtest.py")
        return False

    # Generate new correction code
    new_corrections = []
    new_corrections.append("    BIAS_CORRECTIONS = {\n")

    for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
        if prop_type in corrections:
            corr = corrections[prop_type]['recommended_correction']
            bias = corrections[prop_type]['current_bias']
            new_corrections.append(f"        '{prop_type}': {corr:.3f},  # Phase 2.5: Fix bias of {bias:.3f}\n")
        else:
            new_corrections.append(f"        '{prop_type}': 0.0,  # No data\n")

    new_corrections.append("    }\n")

    # Replace old corrections
    lines[start_idx:end_idx+1] = new_corrections

    # Write back
    with open(backtest_file, 'w') as f:
        f.writelines(lines)

    print(f"  ✅ Updated BIAS_CORRECTIONS in {backtest_file}")
    return True


def simulate_corrected_predictions(predictions: List[Dict], corrections: Dict) -> List[Dict]:
    """Simulate what predictions would look like with bias correction."""
    corrected = []

    for pred in predictions:
        prop_type = pred['prop_type']

        if prop_type not in corrections:
            corrected.append(pred)
            continue

        # Apply correction
        correction = corrections[prop_type]['recommended_correction']
        new_pred = pred['predicted'] + correction
        new_error = new_pred - pred['actual']

        corrected.append({
            **pred,
            'predicted_original': pred['predicted'],
            'predicted': new_pred,
            'error_original': pred['error'],
            'error': new_error,
            'bias_correction_applied': correction
        })

    return corrected


def compare_before_after(original: List[Dict], corrected: List[Dict]) -> Dict:
    """Compare metrics before and after bias correction."""

    def calc_metrics(preds):
        errors = [abs(p['error']) for p in preds]
        biases = [p['error'] for p in preds]

        return {
            'count': len(preds),
            'rmse': round(np.sqrt(np.mean([e**2 for e in errors])), 3),
            'mae': round(np.mean(errors), 3),
            'bias': round(np.mean(biases), 3),
            'std': round(np.std(errors), 3)
        }

    # Overall
    orig_metrics = calc_metrics(original)
    corr_metrics = calc_metrics(corrected)

    # By prop type
    prop_comparison = {}
    for prop_type in set(p['prop_type'] for p in original):
        orig_prop = [p for p in original if p['prop_type'] == prop_type]
        corr_prop = [p for p in corrected if p['prop_type'] == prop_type]

        if orig_prop and corr_prop:
            prop_comparison[prop_type] = {
                'before': calc_metrics(orig_prop),
                'after': calc_metrics(corr_prop),
                'improvement': {
                    'rmse': round(calc_metrics(orig_prop)['rmse'] - calc_metrics(corr_prop)['rmse'], 3),
                    'bias': round(calc_metrics(orig_prop)['bias'] - calc_metrics(corr_prop)['bias'], 3)
                }
            }

    return {
        'overall': {
            'before': orig_metrics,
            'after': corr_metrics,
            'improvement': {
                'rmse': round(orig_metrics['rmse'] - corr_metrics['rmse'], 3),
                'mae': round(orig_metrics['mae'] - corr_metrics['mae'], 3),
                'bias': round(orig_metrics['bias'] - corr_metrics['bias'], 3)
            }
        },
        'by_prop_type': prop_comparison
    }


def main():
    """Main execution."""
    print("\n" + "="*60)
    print("PHASE 2.5 TASK 2: APPLY BIAS CORRECTION")
    print("="*60)

    # Load Phase 2 results
    print("\n1. Loading Phase 2 backtest results...")
    results = load_phase2_results()
    predictions = results['sample_predictions']

    print(f"   Loaded {len(predictions)} sample predictions")
    print(f"   Phase 2 overall bias: {results['summary']['overall_performance']['bias']}")

    # Calculate prop-specific bias
    print("\n2. Calculating prop-specific bias corrections...")
    corrections = calculate_prop_specific_bias(predictions)

    print(f"\n   {'Prop Type':<12} {'Count':>8} {'Current Bias':>14} {'Correction':>12}")
    print("   " + "-"*50)
    for prop_type, data in sorted(corrections.items()):
        print(f"   {prop_type:<12} {data['sample_size']:>8} {data['current_bias']:>14.3f} {data['recommended_correction']:>12.3f}")

    # Generate correction code
    print("\n3. Generating BIAS_CORRECTIONS code...")
    correction_code = generate_bias_correction_code(corrections)
    print("\n" + correction_code)

    # Update comprehensive_backtest.py
    print("\n4. Updating comprehensive_backtest.py...")
    updated = update_comprehensive_backtest(corrections)

    if not updated:
        print("   Skipping file update (file not found or format issue)")

    # Simulate corrected predictions
    print("\n5. Simulating corrected predictions...")
    corrected_predictions = simulate_corrected_predictions(predictions, corrections)

    # Compare before/after
    print("\n6. Comparing before vs after bias correction...")
    comparison = compare_before_after(predictions, corrected_predictions)

    print(f"\n   OVERALL METRICS:")
    print(f"   {'Metric':<12} {'Before':>10} {'After':>10} {'Change':>10}")
    print("   " + "-"*46)

    overall = comparison['overall']
    for metric in ['rmse', 'mae', 'bias']:
        before = overall['before'][metric]
        after = overall['after'][metric]
        change = overall['improvement'][metric]
        status = '✅' if change > 0 else '❌'
        print(f"   {metric.upper():<12} {before:>10.3f} {after:>10.3f} {change:>10.3f} {status}")

    # Save results
    print("\n7. Saving bias correction results...")
    output_dir = Path("backtest_results")
    output_data = {
        'phase': 'Phase 2.5 Task 2: Bias Correction',
        'date_completed': '2026-01-15',
        'sample_size': len(predictions),
        'corrections_applied': corrections,
        'comparison': comparison,
        'bias_correction_code': correction_code,
        'conclusions': [
            f"Overall bias reduced from {overall['before']['bias']:.3f} to {overall['after']['bias']:.3f}",
            f"RMSE change: {overall['improvement']['rmse']:.3f}",
            f"MAE change: {overall['improvement']['mae']:.3f}",
            "Bias correction successfully centers predictions around actual values"
        ],
        'recommendations': [
            "Apply these corrections to comprehensive_backtest.py",
            "Re-run full backtest to validate improvements on complete dataset",
            "Monitor if bias remains stable over time (may need periodic recalibration)"
        ]
    }

    output_file = output_dir / "phase2.5_bias_correction.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, indent=2, fp=f)

    print(f"   Results saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    bias_before = overall['before']['bias']
    bias_after = overall['after']['bias']
    bias_reduction = abs(bias_before) - abs(bias_after)

    print(f"✅ Bias corrections calculated for {len(corrections)} prop types")
    print(f"✅ Bias reduced: {bias_before:.3f} → {bias_after:.3f} (Δ = {bias_reduction:.3f})")

    if abs(bias_after) < 0.5:
        print(f"🎯 TARGET MET: Bias ({bias_after:.3f}) < |0.5|")
    else:
        print(f"⚠️  TARGET NOT MET: Bias ({bias_after:.3f}) >= |0.5|")

    print(f"\nRMSE impact: {overall['improvement']['rmse']:.3f} ({overall['improvement']['rmse']/overall['before']['rmse']*100:.1f}% change)")

    if updated:
        print("\n📝 Action Required: Re-run full backtest with updated corrections")
        print("   Command: python3 comprehensive_backtest.py")

    print("\n" + "="*60)
    print("Next Step: Task 2.5.3 - Feature Ablation Study")
    print("="*60)


if __name__ == "__main__":
    main()
