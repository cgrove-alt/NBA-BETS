#!/usr/bin/env python3
"""
Backtest: Minutes Oracle Impact on Prop Predictions

Compares prop predictions with and without the minutes oracle adjustment
to measure its impact on accuracy.

Usage:
    python3 scripts/backtest_minutes_impact.py
    python3 scripts/backtest_minutes_impact.py --games 50
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import norm


def get_prop_std_dev(prop_type: str) -> float:
    """Mirrored from daily_predictions.py."""
    PROP_STD_DEVS = {
        'points': 5.5,
        'rebounds': 7.0,
        'assists': 2.5,
        'threes': 1.8,
        'pra': 9.0,
    }
    return PROP_STD_DEVS.get(prop_type.lower(), 5.0)


def simulate_adjustment(predicted_value, avg_minutes, predicted_minutes_p50, line, prop_type):
    """Simulate the minutes oracle adjustment on a single prediction.

    Returns (adjusted_value, adjusted_over_prob, adjusted_edge) or originals if no adjustment.
    """
    std = get_prop_std_dev(prop_type)

    # Original
    orig_z = (predicted_value - line) / std
    orig_over_prob = float(norm.cdf(orig_z))
    orig_edge = (orig_over_prob - 0.524) * 100

    if avg_minutes <= 10 or predicted_minutes_p50 <= 0:
        return predicted_value, orig_over_prob, orig_edge, False

    rate = predicted_value / avg_minutes
    adjusted_value = rate * predicted_minutes_p50

    # Check if adjustment is meaningful (>1%)
    if abs(adjusted_value - predicted_value) / max(abs(predicted_value), 0.1) <= 0.01:
        return predicted_value, orig_over_prob, orig_edge, False

    adj_z = (adjusted_value - line) / std
    adj_over_prob = float(norm.cdf(adj_z))
    adj_edge = (adj_over_prob - 0.524) * 100

    return adjusted_value, adj_over_prob, adj_edge, True


def run_synthetic_backtest(n_games=100):
    """Run a synthetic backtest to demonstrate minutes oracle impact.

    Generates realistic player scenarios and shows how minutes adjustment
    would change predictions vs actual outcomes.
    """
    np.random.seed(42)

    print("=" * 70)
    print("  MINUTES ORACLE IMPACT — SYNTHETIC BACKTEST")
    print("=" * 70)
    print(f"\n  Simulating {n_games} player-prop predictions...\n")

    results = {
        'baseline': {'correct': 0, 'total': 0, 'mae': [], 'edges': []},
        'adjusted': {'correct': 0, 'total': 0, 'mae': [], 'edges': []},
    }

    adjustments_applied = 0

    for i in range(n_games):
        # Generate realistic player scenario
        prop_type = np.random.choice(['points', 'rebounds', 'assists'])

        if prop_type == 'points':
            true_avg_minutes = np.random.uniform(25, 38)
            per_min_rate = np.random.uniform(0.5, 1.0)  # pts per minute
            line = np.random.uniform(15, 35)
        elif prop_type == 'rebounds':
            true_avg_minutes = np.random.uniform(25, 38)
            per_min_rate = np.random.uniform(0.1, 0.4)
            line = np.random.uniform(4, 12)
        else:  # assists
            true_avg_minutes = np.random.uniform(25, 38)
            per_min_rate = np.random.uniform(0.1, 0.3)
            line = np.random.uniform(3, 10)

        # Historical average prediction (baseline — assumes avg minutes)
        model_avg_minutes = true_avg_minutes + np.random.normal(0, 1.5)
        predicted_value = per_min_rate * model_avg_minutes

        # Actual game: minutes vary due to game context
        # Simulate blowout/close game affecting minutes
        spread = np.random.uniform(-15, 15)
        minutes_impact = 0
        if abs(spread) > 10:
            minutes_impact = -np.random.uniform(2, 6)  # Blowout reduces minutes
        elif abs(spread) < 3:
            minutes_impact = np.random.uniform(0, 2)  # Close game may increase

        actual_minutes = true_avg_minutes + minutes_impact + np.random.normal(0, 3)
        actual_minutes = max(10, min(48, actual_minutes))

        # Minutes oracle prediction (approximate)
        oracle_predicted_minutes = actual_minutes + np.random.normal(0, 2.5)
        oracle_predicted_minutes = max(10, min(48, oracle_predicted_minutes))

        # Actual stat outcome
        actual_value = per_min_rate * actual_minutes + np.random.normal(0, get_prop_std_dev(prop_type) * 0.5)
        actual_over = actual_value > line

        # Baseline prediction (no minutes adjustment)
        std = get_prop_std_dev(prop_type)
        baseline_z = (predicted_value - line) / std
        baseline_over_prob = float(norm.cdf(baseline_z))
        baseline_pick_over = baseline_over_prob > 0.5
        baseline_correct = baseline_pick_over == actual_over

        results['baseline']['correct'] += int(baseline_correct)
        results['baseline']['total'] += 1
        results['baseline']['mae'].append(abs(predicted_value - actual_value))
        results['baseline']['edges'].append(abs(baseline_over_prob - 0.524) * 100)

        # Adjusted prediction (with minutes oracle)
        adjusted_value, adj_over_prob, adj_edge, was_adjusted = simulate_adjustment(
            predicted_value, model_avg_minutes, oracle_predicted_minutes, line, prop_type
        )

        adj_pick_over = adj_over_prob > 0.5
        adj_correct = adj_pick_over == actual_over

        results['adjusted']['correct'] += int(adj_correct)
        results['adjusted']['total'] += 1
        results['adjusted']['mae'].append(abs(adjusted_value - actual_value))
        results['adjusted']['edges'].append(abs(adj_edge))

        if was_adjusted:
            adjustments_applied += 1

    # Print results
    print("-" * 70)
    print(f"  {'Metric':<35} {'Baseline':>12} {'+ Minutes':>12} {'Delta':>10}")
    print("-" * 70)

    b_acc = results['baseline']['correct'] / results['baseline']['total'] * 100
    a_acc = results['adjusted']['correct'] / results['adjusted']['total'] * 100
    print(f"  {'Accuracy (over/under)':<35} {b_acc:>11.1f}% {a_acc:>11.1f}% {a_acc-b_acc:>+9.1f}%")

    b_mae = np.mean(results['baseline']['mae'])
    a_mae = np.mean(results['adjusted']['mae'])
    print(f"  {'MAE (mean absolute error)':<35} {b_mae:>12.2f} {a_mae:>12.2f} {a_mae-b_mae:>+10.2f}")

    b_edge = np.mean(results['baseline']['edges'])
    a_edge = np.mean(results['adjusted']['edges'])
    print(f"  {'Avg Edge Magnitude':<35} {b_edge:>11.1f}% {a_edge:>11.1f}% {a_edge-b_edge:>+9.1f}%")

    print("-" * 70)
    print(f"  Adjustments applied: {adjustments_applied}/{n_games} ({adjustments_applied/n_games*100:.0f}%)")
    print()

    # Blowout-specific analysis
    print("  BLOWOUT SCENARIO ANALYSIS (|spread| > 10)")
    print("-" * 70)
    # Re-run with only blowout games
    np.random.seed(42)
    blowout_baseline_correct = 0
    blowout_adjusted_correct = 0
    blowout_count = 0

    for i in range(n_games):
        prop_type = np.random.choice(['points', 'rebounds', 'assists'])
        if prop_type == 'points':
            true_avg_minutes = np.random.uniform(25, 38)
            per_min_rate = np.random.uniform(0.5, 1.0)
            line = np.random.uniform(15, 35)
        elif prop_type == 'rebounds':
            true_avg_minutes = np.random.uniform(25, 38)
            per_min_rate = np.random.uniform(0.1, 0.4)
            line = np.random.uniform(4, 12)
        else:
            true_avg_minutes = np.random.uniform(25, 38)
            per_min_rate = np.random.uniform(0.1, 0.3)
            line = np.random.uniform(3, 10)

        model_avg_minutes = true_avg_minutes + np.random.normal(0, 1.5)
        predicted_value = per_min_rate * model_avg_minutes

        spread = np.random.uniform(-15, 15)
        if abs(spread) <= 10:
            continue  # Skip non-blowout games

        blowout_count += 1
        minutes_impact = -np.random.uniform(2, 6)
        actual_minutes = true_avg_minutes + minutes_impact + np.random.normal(0, 3)
        actual_minutes = max(10, min(48, actual_minutes))

        oracle_predicted_minutes = actual_minutes + np.random.normal(0, 2.5)
        oracle_predicted_minutes = max(10, min(48, oracle_predicted_minutes))

        actual_value = per_min_rate * actual_minutes + np.random.normal(0, get_prop_std_dev(prop_type) * 0.5)
        actual_over = actual_value > line

        std = get_prop_std_dev(prop_type)
        baseline_z = (predicted_value - line) / std
        baseline_over_prob = float(norm.cdf(baseline_z))
        baseline_pick_over = baseline_over_prob > 0.5
        blowout_baseline_correct += int(baseline_pick_over == actual_over)

        adjusted_value, adj_over_prob, _, _ = simulate_adjustment(
            predicted_value, model_avg_minutes, oracle_predicted_minutes, line, prop_type
        )
        adj_pick_over = adj_over_prob > 0.5
        blowout_adjusted_correct += int(adj_pick_over == actual_over)

    if blowout_count > 0:
        b_acc = blowout_baseline_correct / blowout_count * 100
        a_acc = blowout_adjusted_correct / blowout_count * 100
        print(f"  Blowout games: {blowout_count}")
        print(f"  Baseline accuracy:  {b_acc:.1f}%")
        print(f"  Adjusted accuracy:  {a_acc:.1f}%")
        print(f"  Improvement:        {a_acc - b_acc:+.1f}%")
    else:
        print("  No blowout games in sample")

    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    overall_delta = (results['adjusted']['correct'] / results['adjusted']['total'] -
                     results['baseline']['correct'] / results['baseline']['total']) * 100
    mae_improvement = np.mean(results['baseline']['mae']) - np.mean(results['adjusted']['mae'])
    print(f"  Accuracy change: {overall_delta:+.1f}%")
    print(f"  MAE improvement: {mae_improvement:+.2f} (lower is better)")
    if overall_delta >= 0 and mae_improvement >= 0:
        print("  Status: POSITIVE — Minutes oracle improves predictions")
    elif mae_improvement >= 0:
        print("  Status: MIXED — MAE improved but accuracy unchanged")
    else:
        print("  Status: INVESTIGATE — Adjustment may need tuning")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backtest minutes oracle impact")
    parser.add_argument('--games', type=int, default=500,
                        help="Number of simulated predictions (default: 500)")
    args = parser.parse_args()

    run_synthetic_backtest(n_games=args.games)
