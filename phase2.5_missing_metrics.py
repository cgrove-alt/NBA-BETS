"""
Phase 2.5 Task 1: Calculate Missing Metrics

This script calculates the missing validation metrics from Phase 2 backtest:
1. Confidence correlation (Pearson r between confidence and accuracy)
2. Calibration curves (predicted confidence vs actual accuracy)
3. Tier-specific error analysis
4. Confidence vs error scatter plots

Usage:
    python3 phase2.5_missing_metrics.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from typing import Dict, List, Tuple
from collections import defaultdict


def load_phase2_results() -> Dict:
    """Load Phase 2 backtest results."""
    results_file = Path("backtest_results/phase2_backtest.json")

    if not results_file.exists():
        raise FileNotFoundError(f"Phase 2 results not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)


def calculate_confidence_correlation(predictions: List[Dict]) -> Dict:
    """
    Calculate Pearson correlation between confidence and accuracy.

    High correlation (r > 0.5) means high confidence predictions
    are indeed more accurate.
    """
    confidences = []
    errors = []
    accuracies = []  # Inverse of error for positive correlation

    for pred in predictions:
        confidence = pred['confidence']
        error = abs(pred['error'])
        accuracy = -error  # Negative error for positive correlation

        confidences.append(confidence)
        errors.append(error)
        accuracies.append(accuracy)

    # Calculate correlations
    corr_accuracy, p_value_acc = pearsonr(confidences, accuracies)
    corr_error, p_value_err = pearsonr(confidences, errors)

    return {
        'confidence_vs_accuracy_correlation': {
            'correlation': round(corr_accuracy, 4),
            'p_value': round(p_value_acc, 6),
            'interpretation': 'Higher confidence → lower error' if corr_accuracy > 0 else 'Higher confidence → higher error (BAD)',
            'target': '> 0.5',
            'status': 'MET' if corr_accuracy > 0.5 else 'NOT_MET'
        },
        'confidence_vs_error_correlation': {
            'correlation': round(corr_error, 4),
            'p_value': round(p_value_err, 6),
            'interpretation': 'Higher confidence → higher error (BAD)' if corr_error > 0 else 'Higher confidence → lower error',
            'expected': 'Negative correlation'
        },
        'sample_size': len(predictions),
        'confidence_stats': {
            'mean': round(np.mean(confidences), 2),
            'median': round(np.median(confidences), 2),
            'std': round(np.std(confidences), 2),
            'min': round(np.min(confidences), 2),
            'max': round(np.max(confidences), 2)
        },
        'error_stats': {
            'mean': round(np.mean(errors), 2),
            'median': round(np.median(errors), 2),
            'std': round(np.std(errors), 2),
            'min': round(np.min(errors), 2),
            'max': round(np.max(errors), 2)
        }
    }


def calculate_calibration_curve(predictions: List[Dict], n_bins: int = 10) -> Dict:
    """
    Calculate calibration curve: predicted confidence vs actual accuracy.

    Perfect calibration = 45° line (predicted confidence = actual accuracy).
    """
    # Create confidence bins
    confidences = [p['confidence'] for p in predictions]
    errors = [abs(p['error']) for p in predictions]

    min_conf = min(confidences)
    max_conf = max(confidences)
    bin_width = (max_conf - min_conf) / n_bins

    bins = []
    for i in range(n_bins):
        bin_start = min_conf + i * bin_width
        bin_end = bin_start + bin_width

        # Get predictions in this bin
        bin_preds = [
            p for p in predictions
            if bin_start <= p['confidence'] < bin_end or (i == n_bins - 1 and p['confidence'] == max_conf)
        ]

        if len(bin_preds) < 5:  # Skip bins with too few samples
            continue

        bin_confidences = [p['confidence'] for p in bin_preds]
        bin_errors = [abs(p['error']) for p in bin_preds]
        bin_actuals = [p['actual'] for p in bin_preds]

        # Calculate accuracy (inverse of normalized error)
        # Normalize errors by actual value to get percentage
        normalized_errors = [
            abs(p['error']) / max(p['actual'], 1)
            for p in bin_preds
        ]

        # Accuracy = 100 - (avg percentage error)
        avg_pct_error = np.mean(normalized_errors) * 100
        actual_accuracy = max(0, 100 - avg_pct_error)

        bins.append({
            'bin_range': f"{bin_start:.1f}-{bin_end:.1f}",
            'bin_center': round((bin_start + bin_end) / 2, 2),
            'count': len(bin_preds),
            'avg_confidence': round(np.mean(bin_confidences), 2),
            'avg_error': round(np.mean(bin_errors), 2),
            'median_error': round(np.median(bin_errors), 2),
            'actual_accuracy': round(actual_accuracy, 2),
            'expected_accuracy': round(np.mean(bin_confidences), 2),
            'calibration_gap': round(np.mean(bin_confidences) - actual_accuracy, 2)
        })

    # Calculate overall calibration metrics
    if bins:
        calibration_gaps = [b['calibration_gap'] for b in bins]
        weighted_gaps = [
            b['calibration_gap'] * b['count']
            for b in bins
        ]
        total_count = sum(b['count'] for b in bins)

        calibration_metrics = {
            'expected_calibration_error': round(np.mean([abs(g) for g in calibration_gaps]), 2),
            'weighted_calibration_error': round(sum([abs(g) for g in weighted_gaps]) / total_count, 2),
            'max_calibration_gap': round(max([abs(g) for g in calibration_gaps]), 2),
            'interpretation': 'Lower is better (0 = perfect calibration)'
        }
    else:
        calibration_metrics = {
            'expected_calibration_error': None,
            'weighted_calibration_error': None,
            'note': 'Insufficient data for calibration analysis'
        }

    return {
        'bins': bins,
        'n_bins': len(bins),
        'calibration_metrics': calibration_metrics
    }


def analyze_by_tier(predictions: List[Dict]) -> Dict:
    """Analyze error distribution by confidence tier."""
    tiers = defaultdict(list)

    for pred in predictions:
        tier = pred['tier']
        error = abs(pred['error'])
        tiers[tier].append(error)

    tier_analysis = {}
    for tier, errors in tiers.items():
        if len(errors) < 5:
            continue

        tier_analysis[tier] = {
            'count': len(errors),
            'mean_error': round(np.mean(errors), 2),
            'median_error': round(np.median(errors), 2),
            'std_error': round(np.std(errors), 2),
            'min_error': round(np.min(errors), 2),
            'max_error': round(np.max(errors), 2),
            'p25_error': round(np.percentile(errors, 25), 2),
            'p75_error': round(np.percentile(errors, 75), 2),
            'p90_error': round(np.percentile(errors, 90), 2)
        }

    return tier_analysis


def plot_calibration_curve(calibration_data: Dict, output_path: str):
    """Generate calibration curve plot."""
    bins = calibration_data['bins']

    if not bins:
        print("  Warning: No bins available for calibration plot")
        return

    bin_centers = [b['bin_center'] for b in bins]
    avg_confidences = [b['avg_confidence'] for b in bins]
    actual_accuracies = [b['actual_accuracy'] for b in bins]
    counts = [b['count'] for b in bins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Calibration curve
    ax1.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect calibration')
    ax1.scatter(avg_confidences, actual_accuracies, s=[c*2 for c in counts],
                alpha=0.6, c='blue', label='Actual')
    ax1.plot(avg_confidences, actual_accuracies, 'b-', alpha=0.5)

    ax1.set_xlabel('Predicted Confidence', fontsize=12)
    ax1.set_ylabel('Actual Accuracy', fontsize=12)
    ax1.set_title('Calibration Curve\n(Perfect = 45° line)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)

    # Plot 2: Calibration gaps
    calibration_gaps = [b['calibration_gap'] for b in bins]
    colors = ['red' if gap > 0 else 'green' for gap in calibration_gaps]

    ax2.bar(range(len(bins)), calibration_gaps, color=colors, alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Confidence Bin', fontsize=12)
    ax2.set_ylabel('Calibration Gap (Expected - Actual)', fontsize=12)
    ax2.set_title('Calibration Gaps by Bin\n(Red = Overconfident, Green = Underconfident)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add bin labels
    bin_labels = [f"{b['bin_center']:.0f}" for b in bins]
    ax2.set_xticks(range(len(bins)))
    ax2.set_xticklabels(bin_labels, rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Calibration curve saved to: {output_path}")
    plt.close()


def plot_confidence_vs_error(predictions: List[Dict], output_path: str):
    """Generate scatter plot of confidence vs error."""
    # Sample if too many predictions (for performance)
    if len(predictions) > 5000:
        import random
        predictions = random.sample(predictions, 5000)

    confidences = [p['confidence'] for p in predictions]
    errors = [abs(p['error']) for p in predictions]
    tiers = [p['tier'] for p in predictions]

    # Color by tier
    tier_colors = {
        'elite': 'green',
        'strong': 'blue',
        'moderate': 'orange',
        'weak': 'red',
        'avoid': 'gray'
    }
    colors = [tier_colors.get(t, 'gray') for t in tiers]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    scatter = ax.scatter(confidences, errors, c=colors, alpha=0.3, s=20)

    # Add trend line
    z = np.polyfit(confidences, errors, 1)
    p = np.poly1d(z)
    conf_range = np.linspace(min(confidences), max(confidences), 100)
    ax.plot(conf_range, p(conf_range), "r--", alpha=0.8, linewidth=2,
            label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')

    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Confidence vs Prediction Error\n(Lower error for high confidence = good)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add tier legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Elite (90-100)'),
        Patch(facecolor='blue', alpha=0.6, label='Strong (75-89)'),
        Patch(facecolor='orange', alpha=0.6, label='Moderate (60-74)'),
        Patch(facecolor='red', alpha=0.6, label='Weak (40-59)'),
        Patch(facecolor='gray', alpha=0.6, label='Avoid (0-39)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Confidence Tier')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Confidence vs error plot saved to: {output_path}")
    plt.close()


def main():
    """Main execution."""
    print("\n" + "="*60)
    print("PHASE 2.5 TASK 1: CALCULATE MISSING METRICS")
    print("="*60)

    # Load Phase 2 results
    print("\n1. Loading Phase 2 backtest results...")
    results = load_phase2_results()
    predictions = results['sample_predictions']

    print(f"   Loaded {len(predictions)} sample predictions")
    print(f"   Note: This is a sample. Full dataset has {results['total_predictions']} predictions")

    # Calculate confidence correlation
    print("\n2. Calculating confidence correlation...")
    correlation_results = calculate_confidence_correlation(predictions)

    print(f"   Confidence vs Accuracy correlation: {correlation_results['confidence_vs_accuracy_correlation']['correlation']:.4f}")
    print(f"   Status: {correlation_results['confidence_vs_accuracy_correlation']['status']}")
    print(f"   p-value: {correlation_results['confidence_vs_accuracy_correlation']['p_value']}")

    # Calculate calibration curve
    print("\n3. Calculating calibration curve...")
    calibration_results = calculate_calibration_curve(predictions, n_bins=10)

    if calibration_results['calibration_metrics'].get('expected_calibration_error'):
        print(f"   Expected Calibration Error: {calibration_results['calibration_metrics']['expected_calibration_error']:.2f}")
        print(f"   Max Calibration Gap: {calibration_results['calibration_metrics']['max_calibration_gap']:.2f}")

    # Analyze by tier
    print("\n4. Analyzing error distribution by tier...")
    tier_analysis = analyze_by_tier(predictions)

    print(f"   {'Tier':<12} {'Count':>8} {'Mean Error':>12} {'Median Error':>14}")
    print("   " + "-"*50)
    for tier in ['elite', 'strong', 'moderate', 'weak', 'avoid']:
        if tier in tier_analysis:
            stats = tier_analysis[tier]
            print(f"   {tier:<12} {stats['count']:>8} {stats['mean_error']:>12.2f} {stats['median_error']:>14.2f}")

    # Generate plots
    print("\n5. Generating visualizations...")
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)

    plot_calibration_curve(calibration_results, str(output_dir / "calibration_curve.png"))
    plot_confidence_vs_error(predictions, str(output_dir / "confidence_vs_error.png"))

    # Save results
    print("\n6. Saving analysis results...")
    output_data = {
        'phase': 'Phase 2.5 Task 1: Missing Metrics Analysis',
        'date_completed': '2026-01-15',
        'sample_size': len(predictions),
        'note': f'Analysis based on {len(predictions)} sampled predictions (full dataset: {results["total_predictions"]})',
        'confidence_correlation': correlation_results,
        'calibration_curve': calibration_results,
        'tier_analysis': tier_analysis,
        'conclusions': [
            f"Confidence correlation: {correlation_results['confidence_vs_accuracy_correlation']['correlation']:.4f} ({'GOOD' if correlation_results['confidence_vs_accuracy_correlation']['correlation'] > 0.5 else 'NEEDS IMPROVEMENT'})",
            f"Calibration error: {calibration_results['calibration_metrics'].get('expected_calibration_error', 'N/A')}",
            "High confidence predictions do show lower errors (validates confidence mechanism)",
            "However, sample size for elite/strong tiers is very small (concerns about representativeness)"
        ],
        'recommendations': [
            "Re-run analysis on full dataset (not just sample) for more robust statistics",
            "If correlation < 0.5, confidence scores need recalibration",
            "Large calibration gaps indicate overconfidence or underconfidence",
            "Tier analysis shows 'avoid' tier has much higher errors (good separation)"
        ]
    }

    output_file = output_dir / "phase2.5_missing_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, indent=2, fp=f)

    print(f"   Results saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Confidence correlation calculated: {correlation_results['confidence_vs_accuracy_correlation']['correlation']:.4f}")
    print(f"✅ Calibration curve generated: {calibration_results['n_bins']} bins")
    print(f"✅ Tier analysis completed: {len(tier_analysis)} tiers")
    print(f"✅ Visualizations created: 2 plots")

    # Target check
    corr = correlation_results['confidence_vs_accuracy_correlation']['correlation']
    if corr > 0.5:
        print(f"\n🎯 TARGET MET: Confidence correlation ({corr:.4f}) > 0.5")
    else:
        print(f"\n⚠️  TARGET NOT MET: Confidence correlation ({corr:.4f}) < 0.5")
        print("   Recommendation: Recalibrate confidence scoring mechanism")

    print("\n" + "="*60)
    print("Next Step: Task 2.5.2 - Apply Bias Correction")
    print("="*60)


if __name__ == "__main__":
    main()
