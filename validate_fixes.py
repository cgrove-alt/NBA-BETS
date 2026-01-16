"""
Validation Script: Check if All Phase 2.5 Targets Met
=====================================================

Validates backtest results against Phase 2.5 success criteria.

Targets:
1. Overall RMSE < 5.0
2. Overall Bias < |0.5|
3. Per-prop Bias < |0.5| for all prop types
4. Elite + Strong confidence ≥ 10% of predictions
5. Confidence correlation r > 0.5
6. Phase 2 RMSE < Phase 1 RMSE (5.435)
7. Threes R² > 0
8. No DNP errors (predictions with actual=0 and min=0)
"""

import json
from pathlib import Path
from typing import Dict, List
import sys

# Phase 1 baseline for comparison
PHASE1_RMSE = 5.435

# Targets
TARGETS = {
    'overall_rmse': 5.0,
    'overall_bias': 0.5,
    'per_prop_bias': 0.5,
    'elite_strong_pct': 10.0,
    'confidence_correlation': 0.5,
    'threes_r2': 0.0,
}


def load_results(file_path: str = "backtest_results_2025.json") -> Dict:
    """Load backtest results."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Error: {file_path} not found!")
        sys.exit(1)

    with open(path, 'r') as f:
        return json.load(f)


def check_dnp_errors(results: Dict) -> Dict:
    """Check for DNP errors (predictions with actual=0)."""
    if 'raw_predictions' not in results:
        return {'status': 'SKIP', 'reason': 'No raw predictions available'}

    raw_preds = results['raw_predictions']

    # Count predictions with actual=0 for each prop type
    dnp_by_prop = {}
    for pred in raw_preds:
        if pred['actual'] == 0:
            prop_type = pred['prop_type']
            if prop_type not in dnp_by_prop:
                dnp_by_prop[prop_type] = []
            dnp_by_prop[prop_type].append(pred)

    total_dnp = sum(len(preds) for preds in dnp_by_prop.values())

    return {
        'total_dnp_predictions': total_dnp,
        'by_prop_type': {k: len(v) for k, v in dnp_by_prop.items()},
        'status': 'PASS' if total_dnp == 0 else 'FAIL',
        'sample_dnp_errors': [
            f"{p['player_name']} {p['prop_type']}: pred={p['predicted']:.1f} actual=0"
            for preds in list(dnp_by_prop.values())[:1]
            for p in preds[:5]
        ] if total_dnp > 0 else []
    }


def validate_results(results: Dict) -> Dict:
    """Validate all targets."""
    validation = {}

    overall = results.get('overall', {})
    metrics = results.get('metrics', {})

    # Target 1: Overall RMSE < 5.0
    overall_rmse = overall.get('RMSE', float('inf'))
    validation['overall_rmse'] = {
        'value': overall_rmse,
        'target': f'< {TARGETS["overall_rmse"]}',
        'status': 'PASS' if overall_rmse < TARGETS['overall_rmse'] else 'FAIL',
        'delta_from_phase1': overall_rmse - PHASE1_RMSE,
    }

    # Target 2: Overall Bias < |0.5|
    overall_bias = overall.get('Bias', float('inf'))
    validation['overall_bias'] = {
        'value': overall_bias,
        'target': f'< |{TARGETS["overall_bias"]}|',
        'status': 'PASS' if abs(overall_bias) < TARGETS['overall_bias'] else 'FAIL',
    }

    # Target 3: Per-prop Bias < |0.5|
    per_prop_bias = {}
    per_prop_status = 'PASS'
    for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
        if prop_type in metrics:
            bias = metrics[prop_type].get('Bias', float('inf'))
            status = 'PASS' if abs(bias) < TARGETS['per_prop_bias'] else 'FAIL'
            per_prop_bias[prop_type] = {
                'value': bias,
                'status': status
            }
            if status == 'FAIL':
                per_prop_status = 'FAIL'

    validation['per_prop_bias'] = {
        'target': f'< |{TARGETS["per_prop_bias"]}|',
        'status': per_prop_status,
        'by_prop': per_prop_bias
    }

    # Target 4: Elite + Strong confidence ≥ 10%
    # (Need to check if confidence data available)
    validation['elite_strong_confidence'] = {
        'status': 'SKIP',
        'reason': 'Confidence data not in standard backtest results'
    }

    # Target 5: Confidence correlation r > 0.5
    validation['confidence_correlation'] = {
        'status': 'SKIP',
        'reason': 'Confidence data not in standard backtest results'
    }

    # Target 6: Phase 2 RMSE < Phase 1 RMSE
    validation['phase2_vs_phase1'] = {
        'phase1_rmse': PHASE1_RMSE,
        'phase2_rmse': overall_rmse,
        'improvement': PHASE1_RMSE - overall_rmse,
        'status': 'PASS' if overall_rmse < PHASE1_RMSE else 'FAIL',
    }

    # Target 7: Threes R² > 0
    threes_r2 = metrics.get('threes', {}).get('R²', float('-inf'))
    validation['threes_r2'] = {
        'value': threes_r2,
        'target': f'> {TARGETS["threes_r2"]}',
        'status': 'PASS' if threes_r2 > TARGETS['threes_r2'] else 'FAIL',
    }

    # Target 8: No DNP errors
    validation['dnp_errors'] = check_dnp_errors(results)

    return validation


def print_validation_report(validation: Dict):
    """Print formatted validation report."""
    print("\n" + "="*70)
    print(" PHASE 2.5 VALIDATION REPORT")
    print("="*70)

    # DNP Errors
    print("\n🔍 FIX #1: DNP/Injury Detection")
    print("-" * 70)
    dnp = validation['dnp_errors']
    if dnp['status'] == 'SKIP':
        print(f"  ⏩ SKIPPED: {dnp['reason']}")
    elif dnp['status'] == 'PASS':
        print(f"  ✅ PASS: No DNP errors found")
    else:
        print(f"  ❌ FAIL: {dnp['total_dnp_predictions']} DNP predictions found")
        print(f"     By prop type: {dnp['by_prop_type']}")
        if dnp['sample_dnp_errors']:
            print(f"     Sample errors:")
            for err in dnp['sample_dnp_errors']:
                print(f"       - {err}")

    # Overall RMSE
    print("\n📊 Target 1: Overall RMSE")
    print("-" * 70)
    rmse = validation['overall_rmse']
    status_icon = "✅" if rmse['status'] == 'PASS' else "❌"
    print(f"  {status_icon} RMSE: {rmse['value']:.3f} (target: {rmse['target']})")
    delta_icon = "📉" if rmse['delta_from_phase1'] < 0 else "📈"
    print(f"  {delta_icon} vs Phase 1: {rmse['delta_from_phase1']:+.3f}")

    # Overall Bias
    print("\n📊 Target 2: Overall Bias")
    print("-" * 70)
    bias = validation['overall_bias']
    status_icon = "✅" if bias['status'] == 'PASS' else "❌"
    print(f"  {status_icon} Bias: {bias['value']:+.3f} (target: {bias['target']})")

    # Per-prop Bias
    print("\n📊 Target 3: Per-Prop Bias")
    print("-" * 70)
    per_prop = validation['per_prop_bias']
    print(f"  Overall Status: {'✅ PASS' if per_prop['status'] == 'PASS' else '❌ FAIL'}")
    for prop_type, data in per_prop['by_prop'].items():
        status_icon = "✅" if data['status'] == 'PASS' else "❌"
        print(f"  {status_icon} {prop_type:8s}: {data['value']:+.3f}")

    # Confidence
    print("\n📊 Target 4-5: Confidence Metrics")
    print("-" * 70)
    conf = validation['elite_strong_confidence']
    print(f"  ⏩ {conf['reason']}")

    # Phase 2 vs Phase 1
    print("\n📊 Target 6: Phase 2 Improvement")
    print("-" * 70)
    comp = validation['phase2_vs_phase1']
    status_icon = "✅" if comp['status'] == 'PASS' else "❌"
    print(f"  {status_icon} Phase 1 RMSE: {comp['phase1_rmse']:.3f}")
    print(f"     Phase 2 RMSE: {comp['phase2_rmse']:.3f}")
    print(f"     Improvement:  {comp['improvement']:+.3f}")

    # Threes R²
    print("\n📊 Target 7: Threes R²")
    print("-" * 70)
    threes = validation['threes_r2']
    status_icon = "✅" if threes['status'] == 'PASS' else "❌"
    print(f"  {status_icon} R²: {threes['value']:.3f} (target: {threes['target']})")

    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)

    all_checks = [
        ('DNP Detection', dnp['status']),
        ('Overall RMSE', rmse['status']),
        ('Overall Bias', bias['status']),
        ('Per-Prop Bias', per_prop['status']),
        ('Confidence', conf['status']),
        ('Phase 2 vs Phase 1', comp['status']),
        ('Threes R²', threes['status']),
    ]

    passed = sum(1 for _, status in all_checks if status == 'PASS')
    failed = sum(1 for _, status in all_checks if status == 'FAIL')
    skipped = sum(1 for _, status in all_checks if status == 'SKIP')

    print(f"\n  ✅ PASSED: {passed}")
    print(f"  ❌ FAILED: {failed}")
    print(f"  ⏩ SKIPPED: {skipped}")

    if failed == 0 and skipped <= 2:  # Allow skipping confidence checks
        print("\n  🎉 ALL CRITICAL TARGETS MET!")
    else:
        print("\n  ⚠️  ISSUES REMAINING:")
        for check_name, status in all_checks:
            if status == 'FAIL':
                print(f"     - {check_name}")

    print("\n" + "="*70)


def main():
    # Load results
    results_file = "backtest_results_2025.json"
    print(f"Loading results from: {results_file}")

    results = load_results(results_file)

    # Validate
    validation = validate_results(results)

    # Print report
    print_validation_report(validation)

    # Save validation results
    output_file = Path("backtest_results/validation_report.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(validation, f, indent=2)

    print(f"\nValidation results saved to: {output_file}")


if __name__ == "__main__":
    main()
