#!/usr/bin/env python3
"""
Simple comparison of Task 3.1 backtest vs Phase 2 baseline
"""

import json

def main():
    # Load files
    with open('backtest_results/phase2_backtest.json') as f:
        baseline = json.load(f)

    with open('backtest_results_2025.json') as f:
        new = json.load(f)

    # Extract metrics
    baseline_props = baseline['summary']['by_prop_type_filtered']
    new_props = new['metrics']

    print("=" * 70)
    print("TASK 3.1 BACKTEST ANALYSIS: Player Impact Metrics Integration")
    print("=" * 70)
    print()

    # Compare each prop type
    improvements = []

    print(f"{'Prop Type':<12} {'Metric':<8} {'Baseline':<10} {'New':<10} {'Change':<10} {'Status'}")
    print("-" * 70)

    for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
        if prop_type not in new_props:
            continue

        baseline_prop = baseline_props.get(prop_type, {})
        new_prop = new_props[prop_type]

        # RMSE comparison
        if 'rmse' in baseline_prop and 'rmse' in new_prop:
            b_rmse = baseline_prop['rmse']
            n_rmse = new_prop['rmse']
            improvement = ((b_rmse - n_rmse) / b_rmse) * 100

            improvements.append(improvement)

            status = "✅" if improvement > 0 else "❌"

            print(f"{prop_type:<12} {'RMSE':<8} {b_rmse:<10.4f} {n_rmse:<10.4f} {improvement:>+9.2f}% {status}")

        # MAE comparison
        if 'mae' in baseline_prop and 'mae' in new_prop:
            b_mae = baseline_prop['mae']
            n_mae = new_prop['mae']
            improvement_mae = ((b_mae - n_mae) / b_mae) * 100

            status = "✅" if improvement_mae > 0 else "❌"

            print(f"{prop_type:<12} {'MAE':<8} {b_mae:<10.4f} {n_mae:<10.4f} {improvement_mae:>+9.2f}% {status}")

        # R² comparison
        if 'r2' in baseline_prop and 'r2' in new_prop:
            b_r2 = baseline_prop['r2']
            n_r2 = new_prop['r2']
            improvement_r2 = ((n_r2 - b_r2) / abs(b_r2)) * 100 if b_r2 != 0 else 0

            status = "✅" if improvement_r2 > 0 else "❌"

            print(f"{prop_type:<12} {'R²':<8} {b_r2:<10.4f} {n_r2:<10.4f} {improvement_r2:>+9.2f}% {status}")

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"Average RMSE Improvement: {avg_improvement:+.2f}%")
        print(f"Prop Types Tested: {len(set([p for p in ['points', 'rebounds', 'assists', 'threes', 'pra'] if p in new_props]))}")
        print()

        if avg_improvement >= 5.0:
            print("✅ TARGET MET (≥5% RMSE improvement)")
            print()
            print("RECOMMENDATION: Mark Task 3.1 as COMPLETE")
        elif avg_improvement >= 2.0:
            print("✅ SUCCESS CRITERIA MET (≥2% RMSE improvement)")
            print()
            print("RECOMMENDATION: Mark Task 3.1 as COMPLETE")
        elif avg_improvement >= 0:
            print("⚠️  MARGINAL IMPROVEMENT (<2% RMSE improvement)")
            print()
            print("RECOMMENDATION: Investigate further or document as partial success")
        else:
            print("❌ REGRESSION (negative RMSE improvement)")
            print()
            print("RECOMMENDATION: Investigate why features hurt performance")

        print()
        print("=" * 70)
    else:
        print("No improvements calculated")

if __name__ == '__main__':
    main()
