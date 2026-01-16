"""
Track Phase 2.5 Target Achievement
==================================

Monitors all 4 Phase 2 targets and reports current status:
1. Overall RMSE < 5.0
2. Overall Bias < |0.5|
3. Elite+Strong ≥ 10%
4. Confidence correlation r > 0.5

This script provides a single dashboard to track progress through Phase 2.5.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr


def load_backtest_results():
    """Load Phase 2 backtest results."""
    results_file = Path('backtest_results/phase2_backtest.json')

    if not results_file.exists():
        print(f"❌ Backtest results not found: {results_file}")
        return None

    with open(results_file, 'r') as f:
        return json.load(f)


def check_rmse_target(results):
    """Check RMSE target (< 5.0)."""
    if 'overall_metrics' not in results:
        return None

    overall_rmse = results['overall_metrics'].get('rmse')

    if overall_rmse is None:
        return None

    target_met = overall_rmse < 5.0
    status_icon = '✅' if target_met else '❌'

    return {
        'target': '< 5.0',
        'actual': overall_rmse,
        'met': target_met,
        'status_icon': status_icon,
        'gap': overall_rmse - 5.0
    }


def check_bias_target(results):
    """Check bias target (< |0.5|)."""
    if 'overall_metrics' not in results:
        return None

    overall_bias = results['overall_metrics'].get('bias')

    if overall_bias is None:
        return None

    target_met = abs(overall_bias) < 0.5
    status_icon = '✅' if target_met else '❌'

    return {
        'target': '< |0.5|',
        'actual': overall_bias,
        'met': target_met,
        'status_icon': status_icon,
        'gap': abs(overall_bias) - 0.5
    }


def check_elite_strong_target(results):
    """Check Elite+Strong target (≥ 10%)."""
    if 'confidence_distribution' not in results:
        return None

    distribution = results['confidence_distribution']

    elite_count = distribution.get('Elite (90-100)', 0)
    strong_count = distribution.get('Strong (75-89)', 0)
    total_count = distribution.get('total', 0)

    if total_count == 0:
        return None

    elite_strong_pct = 100 * (elite_count + strong_count) / total_count

    target_met = elite_strong_pct >= 10.0
    status_icon = '✅' if target_met else '❌'

    return {
        'target': '≥ 10%',
        'actual': elite_strong_pct,
        'met': target_met,
        'status_icon': status_icon,
        'gap': elite_strong_pct - 10.0,
        'elite_count': elite_count,
        'strong_count': strong_count,
        'total_count': total_count
    }


def check_confidence_correlation_target(results):
    """Check confidence correlation target (r > 0.5)."""
    # Extract confidence scores and accuracy
    confidence_scores = []
    accuracies = []

    for game in results.get('games', []):
        for pred in game.get('predictions', []):
            if 'confidence' in pred and pred['confidence'] is not None:
                actual = pred.get('actual')
                predicted = pred.get('predicted')

                if actual is not None and predicted is not None:
                    confidence_scores.append(pred['confidence'])

                    # Compute accuracy (inverse of relative error)
                    if actual == 0:
                        accuracy = 1.0 if predicted < 2 else 0.0
                    else:
                        relative_error = abs(predicted - actual) / actual
                        accuracy = max(0, 1 - relative_error)

                    accuracies.append(accuracy)

    if len(confidence_scores) < 2:
        return None

    # Calculate Pearson correlation
    r, p_value = pearsonr(confidence_scores, accuracies)

    target_met = r > 0.5
    status_icon = '✅' if target_met else '❌'

    return {
        'target': '> 0.5',
        'actual': r,
        'met': target_met,
        'status_icon': status_icon,
        'gap': r - 0.5,
        'p_value': p_value,
        'n_samples': len(confidence_scores)
    }


def print_dashboard(targets):
    """Print dashboard of all targets."""
    print("="*70)
    print("PHASE 2.5 TARGET DASHBOARD")
    print("="*70)

    print(f"\n{'Target':<30s} {'Current':<15s} {'Goal':<15s} {'Status':<10s}")
    print("-"*70)

    # Target 1: RMSE
    if targets['rmse']:
        t = targets['rmse']
        print(f"{'1. Overall RMSE':<30s} {t['actual']:<15.3f} {t['target']:<15s} {t['status_icon']:<10s}")
        if not t['met']:
            print(f"   Gap: {t['gap']:+.3f} (need to reduce by {abs(t['gap']):.3f})")

    # Target 2: Bias
    if targets['bias']:
        t = targets['bias']
        print(f"{'2. Overall Bias':<30s} {t['actual']:<15.3f} {t['target']:<15s} {t['status_icon']:<10s}")
        if not t['met']:
            print(f"   Gap: {t['gap']:+.3f}")

    # Target 3: Elite+Strong
    if targets['elite_strong']:
        t = targets['elite_strong']
        print(f"{'3. Elite+Strong %':<30s} {t['actual']:<15.2f} {t['target']:<15s} {t['status_icon']:<10s}")
        if not t['met']:
            print(f"   Gap: {t['gap']:+.2f}% (need {abs(t['gap']):.2f}% more)")
        print(f"   Distribution: Elite={t['elite_count']}, Strong={t['strong_count']}, Total={t['total_count']}")

    # Target 4: Confidence Correlation
    if targets['confidence_correlation']:
        t = targets['confidence_correlation']
        print(f"{'4. Confidence Correlation':<30s} {t['actual']:<15.4f} {t['target']:<15s} {t['status_icon']:<10s}")
        if not t['met']:
            print(f"   Gap: {t['gap']:+.4f} (p={t['p_value']:.4e}, n={t['n_samples']})")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    targets_met = sum([
        1 if targets['rmse'] and targets['rmse']['met'] else 0,
        1 if targets['bias'] and targets['bias']['met'] else 0,
        1 if targets['elite_strong'] and targets['elite_strong']['met'] else 0,
        1 if targets['confidence_correlation'] and targets['confidence_correlation']['met'] else 0
    ])

    total_targets = 4
    progress_pct = 100 * targets_met / total_targets

    print(f"\nTargets Met: {targets_met}/{total_targets} ({progress_pct:.0f}%)")

    if targets_met == total_targets:
        print("\n🎉 ALL TARGETS MET! Phase 2 complete - ready for Phase 3!")
    elif targets_met >= 3:
        print("\n⚠️  Almost there! One more target to go.")
    elif targets_met >= 2:
        print("\n⚠️  Partial success. Continue with solution path.")
    else:
        print("\n❌ Multiple targets not met. Continue systematic fixes.")

    # Phase 3 readiness
    print("\n" + "="*70)
    print("PHASE 3 READINESS")
    print("="*70)

    critical_blockers = []

    if targets['elite_strong'] and not targets['elite_strong']['met']:
        critical_blockers.append("Elite+Strong < 10% - cannot implement selective betting")

    if targets['confidence_correlation'] and not targets['confidence_correlation']['met']:
        critical_blockers.append("Low confidence correlation - cannot trust confidence scores")

    if critical_blockers:
        print("\n🚫 BLOCKED - Cannot proceed to Phase 3:")
        for blocker in critical_blockers:
            print(f"  - {blocker}")
    else:
        print("\n✅ READY - Can proceed to Phase 3 (betting strategy)")


def main():
    print("Loading backtest results...\n")
    results = load_backtest_results()

    if not results:
        print("Cannot load backtest results. Run phase2_backtest_with_confidence.py first.")
        return

    # Check all targets
    targets = {
        'rmse': check_rmse_target(results),
        'bias': check_bias_target(results),
        'elite_strong': check_elite_strong_target(results),
        'confidence_correlation': check_confidence_correlation_target(results)
    }

    # Print dashboard
    print_dashboard(targets)

    # Save status
    output = {
        'timestamp': str(Path('backtest_results/phase2_backtest.json').stat().st_mtime),
        'targets': {
            'rmse': targets['rmse'],
            'bias': targets['bias'],
            'elite_strong': targets['elite_strong'],
            'confidence_correlation': targets['confidence_correlation']
        },
        'targets_met': sum([1 for t in targets.values() if t and t['met']]),
        'total_targets': 4
    }

    output_file = Path('backtest_results/phase2_target_status.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nStatus saved to: {output_file}")


if __name__ == '__main__':
    main()
