"""
Calculate Missing Confidence Metrics
====================================

Calculates:
1. Confidence correlation (Pearson r between confidence and accuracy)
2. Calibration curve (expected vs actual accuracy by confidence bin)
3. Expected Calibration Error (ECE)
4. Confidence vs error scatter analysis

Target: Confidence correlation r > 0.5
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def load_phase2_results():
    """Load Phase 2 backtest results."""
    with open('backtest_results/phase2_backtest.json', 'r') as f:
        return json.load(f)


def calculate_confidence_correlation(predictions):
    """
    Calculate Pearson correlation between confidence and accuracy.

    Higher confidence should correlate with lower error (higher accuracy).
    """
    confidences = []
    errors = []

    for pred in predictions:
        if 'confidence' not in pred or pred['confidence'] is None:
            continue

        confidence = pred['confidence']
        error = abs(pred['predicted'] - pred['actual'])

        confidences.append(confidence)
        errors.append(error)

    if len(confidences) < 10:
        return None

    # We want negative correlation (high confidence → low error)
    # So we'll correlate confidence with negative error (accuracy)
    accuracies = [-e for e in errors]

    corr, p_value = pearsonr(confidences, accuracies)

    return {
        'correlation': round(corr, 4),
        'p_value': round(p_value, 6),
        'n': len(confidences),
        'mean_confidence': round(np.mean(confidences), 2),
        'mean_error': round(np.mean(errors), 3),
        'status': 'MET' if corr > 0.5 else 'NOT_MET',
        'target': 0.5
    }


def calculate_calibration_curve(predictions, n_bins=10):
    """
    Calculate calibration curve.

    Bins predictions by confidence, compares expected accuracy to actual.
    """
    # Group by confidence bins
    bins = defaultdict(list)

    for pred in predictions:
        if 'confidence' not in pred or pred['confidence'] is None:
            continue

        confidence = pred['confidence']
        error = abs(pred['predicted'] - pred['actual'])

        # Bin by confidence decile
        bin_idx = min(int(confidence / 10), n_bins - 1)
        bins[bin_idx].append({
            'confidence': confidence,
            'error': error,
            'predicted': pred['predicted'],
            'actual': pred['actual']
        })

    calibration = []
    for bin_idx in sorted(bins.keys()):
        bin_preds = bins[bin_idx]

        avg_confidence = np.mean([p['confidence'] for p in bin_preds])
        avg_error = np.mean([p['error'] for p in bin_preds])

        # Normalize confidence to 0-1 scale for comparison with accuracy
        expected_accuracy = avg_confidence / 100.0

        # Actual accuracy: what percentage of predictions were "close"?
        # Define "close" as within 3 points for all props
        close_preds = sum(1 for p in bin_preds if p['error'] <= 3.0)
        actual_accuracy = close_preds / len(bin_preds) if bin_preds else 0

        calibration.append({
            'bin': bin_idx,
            'confidence_range': f'{bin_idx*10}-{(bin_idx+1)*10}',
            'count': len(bin_preds),
            'avg_confidence': round(avg_confidence, 2),
            'avg_error': round(avg_error, 3),
            'expected_accuracy': round(expected_accuracy, 3),
            'actual_accuracy': round(actual_accuracy, 3),
            'calibration_gap': round(expected_accuracy - actual_accuracy, 3)
        })

    # Calculate Expected Calibration Error (ECE)
    total_preds = sum(len(bins[b]) for b in bins)
    ece = sum(
        len(bins[c['bin']]) / total_preds * abs(c['calibration_gap'])
        for c in calibration
    )

    return {
        'calibration_curve': calibration,
        'expected_calibration_error': round(ece, 4),
        'n_bins': n_bins,
        'total_predictions': total_preds
    }


def analyze_confidence_by_tier(predictions):
    """Analyze confidence vs error by tier."""
    by_tier = defaultdict(list)

    for pred in predictions:
        if 'tier' not in pred or 'confidence' not in pred:
            continue

        tier = pred['tier']
        confidence = pred['confidence']
        error = abs(pred['predicted'] - pred['actual'])

        by_tier[tier].append({
            'confidence': confidence,
            'error': error
        })

    tier_stats = {}
    for tier, preds in by_tier.items():
        if not preds:
            continue

        confidences = [p['confidence'] for p in preds]
        errors = [p['error'] for p in preds]

        tier_stats[tier] = {
            'count': len(preds),
            'avg_confidence': round(np.mean(confidences), 2),
            'std_confidence': round(np.std(confidences), 2),
            'avg_error': round(np.mean(errors), 3),
            'median_error': round(np.median(errors), 3),
            '90th_pct_error': round(np.percentile(errors, 90), 3)
        }

    return tier_stats


def plot_calibration_curve(calibration_data, output_file='backtest_results/calibration_curve.png'):
    """Generate calibration curve plot."""
    curve = calibration_data['calibration_curve']

    confidences = [c['avg_confidence'] / 100 for c in curve]
    expected = [c['expected_accuracy'] for c in curve]
    actual = [c['actual_accuracy'] for c in curve]

    plt.figure(figsize=(10, 8))

    # Plot calibration
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    plt.plot(confidences, actual, 'bo-', label='Actual Calibration', linewidth=2, markersize=8)

    # Add error bars
    counts = [c['count'] for c in curve]
    sizes = [min(c / max(counts) * 500, 500) for c in counts]
    plt.scatter(confidences, actual, s=sizes, alpha=0.3, color='blue')

    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Actual Accuracy (% within 3 points)', fontsize=12)
    plt.title(f'Confidence Calibration Curve\nECE: {calibration_data["expected_calibration_error"]:.4f}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    # Add annotations
    for conf, acc, count in zip(confidences, actual, counts):
        if count > 100:
            plt.annotate(f'n={count}', (conf, acc), xytext=(5, 5),
                        textcoords='offset points', fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'  Calibration curve saved to: {output_file}')


def plot_confidence_vs_error(predictions, output_file='backtest_results/confidence_vs_error.png'):
    """Generate confidence vs error scatter plot."""
    confidences = []
    errors = []
    tiers = []

    tier_colors = {
        'elite': 'gold',
        'strong': 'green',
        'moderate': 'blue',
        'weak': 'orange',
        'avoid': 'red'
    }

    for pred in predictions:
        if 'confidence' not in pred or pred['confidence'] is None:
            continue

        confidences.append(pred['confidence'])
        errors.append(abs(pred['predicted'] - pred['actual']))
        tiers.append(pred.get('tier', 'unknown'))

    plt.figure(figsize=(12, 8))

    # Plot by tier
    for tier in ['elite', 'strong', 'moderate', 'weak', 'avoid']:
        tier_conf = [c for c, t in zip(confidences, tiers) if t == tier]
        tier_err = [e for e, t in zip(errors, tiers) if t == tier]

        if tier_conf:
            plt.scatter(tier_conf, tier_err, alpha=0.3, s=10,
                       color=tier_colors.get(tier, 'gray'), label=tier.capitalize())

    # Add trend line
    z = np.polyfit(confidences, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(confidences), max(confidences), 100)
    plt.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.4f})')

    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Confidence vs Prediction Error\n(Lower error = better)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Calculate correlation
    corr, _ = pearsonr(confidences, [-e for e in errors])
    plt.text(0.02, 0.98, f'Correlation: {corr:.4f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'  Confidence vs error plot saved to: {output_file}')


def main():
    print('='*70)
    print('CALCULATING MISSING CONFIDENCE METRICS')
    print('='*70)

    # Load data
    print('\nLoading Phase 2 backtest results...')
    data = load_phase2_results()

    # Get sample predictions (or all if available)
    if 'sample_predictions' in data:
        predictions = data['sample_predictions']
        print(f'  Using sample predictions: {len(predictions)}')
    else:
        print('  ❌ No predictions found in file')
        return

    # 1. Confidence Correlation
    print('\n' + '='*70)
    print('1. CONFIDENCE CORRELATION')
    print('='*70)

    corr_result = calculate_confidence_correlation(predictions)
    if corr_result:
        status_icon = '✅ MET' if corr_result['status'] == 'MET' else '❌ NOT MET'
        print(f'\nCorrelation: {corr_result["correlation"]} {status_icon}')
        print(f'  Target: > {corr_result["target"]}')
        print(f'  P-value: {corr_result["p_value"]}')
        print(f'  Sample size: {corr_result["n"]}')
        print(f'  Mean confidence: {corr_result["mean_confidence"]}')
        print(f'  Mean error: {corr_result["mean_error"]}')
    else:
        print('  ❌ Could not calculate (insufficient data)')

    # 2. Calibration Curve
    print('\n' + '='*70)
    print('2. CALIBRATION CURVE & ECE')
    print('='*70)

    calibration = calculate_calibration_curve(predictions)
    print(f'\nExpected Calibration Error (ECE): {calibration["expected_calibration_error"]:.4f}')
    print(f'Total predictions analyzed: {calibration["total_predictions"]}')
    print('\nCalibration by confidence bin:')
    print(f'  {\"Bin\":12s} {\"Count\":>6s} {\"Avg Conf\":>9s} {\"Avg Err\":>9s} {\"Expected\":>9s} {\"Actual\":>9s} {\"Gap\":>9s}')
    print(f'  {\"-\"*12} {\"------\"} {\"--------\"} {\"--------\"} {\"--------\"} {\"--------\"} {\"--------\"}')

    for c in calibration['calibration_curve']:
        print(f'  {c[\"confidence_range\"]:12s} {c[\"count\"]:6d} {c[\"avg_confidence\"]:9.2f} '
              f'{c[\"avg_error\"]:9.3f} {c[\"expected_accuracy\"]:9.3f} '
              f'{c[\"actual_accuracy\"]:9.3f} {c[\"calibration_gap\"]:+9.3f}')

    # 3. Tier Analysis
    print('\n' + '='*70)
    print('3. CONFIDENCE BY TIER')
    print('='*70)

    tier_stats = analyze_confidence_by_tier(predictions)
    print(f'\n{\"Tier\":10s} {\"Count\":>7s} {\"Avg Conf\":>9s} {\"Std Conf\":>9s} {\"Avg Err\":>9s} {\"Med Err\":>9s} {\"90th %\":>9s}')
    print(f'{\"-\"*10} {\"-------\"} {\"--------\"} {\"--------\"} {\"--------\"} {\"--------\"} {\"--------\"}')

    for tier in ['elite', 'strong', 'moderate', 'weak', 'avoid']:
        if tier in tier_stats:
            s = tier_stats[tier]
            print(f'{tier:10s} {s[\"count\"]:7d} {s[\"avg_confidence\"]:9.2f} '
                  f'{s[\"std_confidence\"]:9.2f} {s[\"avg_error\"]:9.3f} '
                  f'{s[\"median_error\"]:9.3f} {s[\"90th_pct_error\"]:9.3f}')

    # Generate plots
    print('\n' + '='*70)
    print('GENERATING PLOTS')
    print('='*70)

    plot_calibration_curve(calibration)
    plot_confidence_vs_error(predictions)

    # Save results
    output = {
        'confidence_correlation': corr_result,
        'calibration': calibration,
        'tier_analysis': tier_stats,
        'summary': {
            'correlation_met': corr_result['status'] == 'MET' if corr_result else False,
            'calibration_quality': 'Poor' if calibration['expected_calibration_error'] > 0.1 else 'Good',
            'recommendations': []
        }
    }

    # Add recommendations
    if corr_result and corr_result['correlation'] < 0.5:
        output['summary']['recommendations'].append(
            f'Confidence correlation ({corr_result[\"correlation\"]}) below target (0.5). '
            'Consider recalibrating confidence thresholds or using different confidence metric.'
        )

    if calibration['expected_calibration_error'] > 0.1:
        output['summary']['recommendations'].append(
            f'ECE ({calibration[\"expected_calibration_error\"]:.4f}) indicates poor calibration. '
            'Model is over/under-confident. Consider calibration methods like Platt scaling or isotonic regression.'
        )

    output_file = Path('backtest_results/confidence_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\n  Results saved to: {output_file}')

    # Final summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)

    if corr_result:
        status = '✅ MET' if corr_result['status'] == 'MET' else '❌ NOT MET'
        print(f'\nConfidence Correlation: {corr_result[\"correlation\"]} {status}')

    cal_status = '✅ Good' if calibration['expected_calibration_error'] < 0.1 else '❌ Poor'
    print(f'Calibration (ECE): {calibration["expected_calibration_error"]:.4f} {cal_status}')

    if output['summary']['recommendations']:
        print('\nRecommendations:')
        for i, rec in enumerate(output['summary']['recommendations'], 1):
            print(f'  {i}. {rec}')

    print('\n' + '='*70)


if __name__ == '__main__':
    main()
