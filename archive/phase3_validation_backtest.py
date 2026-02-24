"""
Phase 3 Validation Backtest - January 2025

Quick validation of Phase 3 features on recent data (Jan 2025).
This validates:
1. Quantile predictions work correctly
2. Kelly bet sizing functions
3. Portfolio management and stop-loss
4. Confidence calibration

For full 2-season analysis, see phase3_comprehensive_backtest.py

Usage:
    python3 phase3_validation_backtest.py
"""

import json

# Use the existing comprehensive backtest infrastructure
from phase3_comprehensive_backtest import Phase3Backtester, RESULTS_DIR

def main():
    """Run validation backtest on January 2025 data."""

    print("\n" + "="*80)
    print("PHASE 3 VALIDATION BACKTEST - JANUARY 2025")
    print("="*80)
    print("\nThis is a quick validation run on recent data.")
    print("For full 2-season analysis, use phase3_comprehensive_backtest.py")
    print("="*80 + "\n")

    # Initialize backtester for 2024-25 season
    backtester = Phase3Backtester(season=2025)

    # Run backtest for January 1-14, 2025 (recent data we likely have cached)
    results = backtester.run_comprehensive_backtest(
        start_date="2025-01-01",
        end_date="2025-01-14"
    )

    # Save results
    output_file = RESULTS_DIR / "phase3_validation_jan2025.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved validation results to: {output_file}")

    # Print key findings
    print("\n" + "="*80)
    print("KEY VALIDATION FINDINGS")
    print("="*80)

    betting = results.get('betting_performance', {})
    overall = results.get('overall_performance', {})
    elite_strong = results.get('elite_strong_performance', {})

    print(f"\n✓ Quantile Models: {len(backtester.quantile_models)} loaded")
    print(f"✓ Total Predictions: {results.get('total_predictions', 0):,}")
    print(f"✓ Elite+Strong Tier: {elite_strong.get('count', 0):,} ({elite_strong.get('percentage', 0):.1f}%)")

    print(f"\n✓ RMSE: {overall.get('rmse', 0):.3f} (Target: < 4.8)")
    print(f"✓ Elite+Strong RMSE: {elite_strong.get('rmse', 0):.3f}")

    print(f"\n✓ Bets Placed: {betting.get('total_bets', 0)}")
    print(f"✓ Win Rate: {betting.get('win_rate', 0):.1f}%")
    print(f"✓ ROI: {betting.get('roi', 0):.2f}%")
    print(f"✓ Sharpe Ratio: {betting.get('sharpe_ratio', 0):.2f}")

    # Check Phase 3 targets
    targets = results.get('phase3_targets', {})
    print("\nPHASE 3 TARGETS (on validation data):")
    targets_met = 0
    total_targets = 0
    for target_name, target_data in targets.items():
        if isinstance(target_data, dict) and 'met' in target_data:
            status = "✓" if target_data['met'] else "✗"
            print(f"  {status} {target_name}: {target_data['actual']} (target: {target_data['target']})")
            total_targets += 1
            if target_data['met']:
                targets_met += 1

    print(f"\n{targets_met}/{total_targets} targets met on validation data")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\nNOTE: This is a small validation sample (2 weeks).")
    print("For robust evaluation, run full 2-season backtest:")
    print("  python3 phase3_comprehensive_backtest.py")
    print("="*80 + "\n")

    return results

if __name__ == "__main__":
    main()
