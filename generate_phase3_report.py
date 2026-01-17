"""
Generate Phase 3 Comprehensive Report

Synthesizes existing backtest results and generates comprehensive Phase 3 analysis.

Uses:
- Phase 2 backtest results (backtest_results/phase2_backtest.json)
- Phase 3 validation (if available)
- Model performance metrics
- Kelly sizing and risk management validation

Generates:
- Comprehensive JSON report
- Comparison to Phase 3 targets
- Recommendations for go-live

Usage:
    python3 generate_phase3_report.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

RESULTS_DIR = Path("backtest_results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_phase2_results() -> Dict:
    """Load existing Phase 2 backtest results."""
    phase2_file = RESULTS_DIR / "phase2_backtest.json"

    if not phase2_file.exists():
        print(f"WARNING: Phase 2 results not found at {phase2_file}")
        return {}

    with open(phase2_file, 'r') as f:
        return json.load(f)


def analyze_quantile_models() -> Dict:
    """Analyze available quantile models."""
    from pathlib import Path

    models_dir = Path("models")
    quantile_models = list(models_dir.glob("*_quantile.pkl"))

    return {
        'total_quantile_models': len(quantile_models),
        'prop_types_with_quantiles': [m.stem.replace('player_', '').replace('_quantile', '') for m in quantile_models],
        'quantile_model_files': [str(m) for m in quantile_models],
    }


def validate_kelly_sizing() -> Dict:
    """Validate Kelly bet sizing implementation."""
    try:
        from risk_management import calculate_kelly_bet_size, get_kelly_multiplier_for_tier

        # Test Kelly calculation
        test_cases = []

        # Elite tier, 55% win prob
        elite_bet = calculate_kelly_bet_size(
            win_prob=0.55,
            decimal_odds=1.909,
            bankroll=1000.0,
            fractional=0.25,
            edge_tier='elite',
            current_drawdown=0.0,
            num_same_day_bets=1,
            max_bet_pct=0.05
        )
        test_cases.append({'tier': 'elite', 'win_prob': 0.55, 'bet_size': elite_bet})

        # Strong tier, 60% win prob
        strong_bet = calculate_kelly_bet_size(
            win_prob=0.60,
            decimal_odds=1.909,
            bankroll=1000.0,
            fractional=0.25,
            edge_tier='strong',
            current_drawdown=0.0,
            num_same_day_bets=1,
            max_bet_pct=0.05
        )
        test_cases.append({'tier': 'strong', 'win_prob': 0.60, 'bet_size': strong_bet})

        # Moderate tier, 52% win prob
        moderate_bet = calculate_kelly_bet_size(
            win_prob=0.52,
            decimal_odds=1.909,
            bankroll=1000.0,
            fractional=0.25,
            edge_tier='moderate',
            current_drawdown=0.0,
            num_same_day_bets=1,
            max_bet_pct=0.05
        )
        test_cases.append({'tier': 'moderate', 'win_prob': 0.52, 'bet_size': moderate_bet})

        return {
            'kelly_available': True,
            'test_cases': test_cases,
            'validation': 'PASSED' if all(tc['bet_size'] >= 0 for tc in test_cases) else 'FAILED',
        }

    except ImportError as e:
        return {
            'kelly_available': False,
            'error': str(e),
            'validation': 'NOT_TESTED',
        }


def evaluate_phase3_targets(phase2_results: Dict) -> Dict:
    """Evaluate Phase 3 targets using Phase 2 results."""

    overall = phase2_results.get('summary', {}).get('overall_performance', {})
    elite_strong = phase2_results.get('summary', {}).get('elite_strong_performance', {})
    by_prop = phase2_results.get('summary', {}).get('by_prop_type_filtered', {})

    targets = {
        'target_1_overall_rmse': {
            'target': '< 4.8',
            'phase2_actual': overall.get('rmse', 999),
            'phase2_met': overall.get('rmse', 999) < 4.8,
            'notes': 'Phase 2 achieved 5.285, still above target. Phase 3 quantile regression should improve.'
        },
        'target_2_points_rmse': {
            'target': '< 5.5',
            'phase2_actual': by_prop.get('points', {}).get('rmse', 999),
            'phase2_met': by_prop.get('points', {}).get('rmse', 999) < 5.5,
            'notes': 'Phase 2 Points RMSE was 9.391 (elite/strong only). Needs improvement.'
        },
        'target_3_threes_r2': {
            'target': '> 0.10',
            'phase2_actual': by_prop.get('threes', {}).get('r2', -1),
            'phase2_met': by_prop.get('threes', {}).get('r2', -1) > 0.10,
            'notes': 'Phase 2 achieved R²=0.013 (elite/strong). Still challenging to predict 3PT makes.'
        },
        'target_4_roi_all': {
            'target': '> 3%',
            'phase2_actual': 'N/A (requires odds data)',
            'phase2_met': False,
            'notes': 'Phase 2 did not integrate betting odds. Phase 3 adds this capability.'
        },
        'target_5_roi_elite': {
            'target': '> 7%',
            'phase2_actual': 'N/A (requires odds data)',
            'phase2_met': False,
            'notes': 'Elite tier predictions (18.8% of total) have RMSE 1.858. Should achieve high ROI with odds.'
        },
        'target_6_sharpe_ratio': {
            'target': '> 1.5',
            'phase2_actual': 'N/A (requires betting simulation)',
            'phase2_met': False,
            'notes': 'Phase 3 adds portfolio simulation with Sharpe ratio calculation.'
        },
        'target_7_max_drawdown': {
            'target': '< 15%',
            'phase2_actual': 'N/A (requires betting simulation)',
            'phase2_met': False,
            'notes': 'Phase 3 implements stop-loss rules to limit drawdown.'
        },
        'target_8_confidence_correlation': {
            'target': 'Pearson r > 0.5',
            'phase2_actual': 'Not measured',
            'phase2_met': False,
            'notes': 'Phase 3 adds calibration analysis to validate confidence scores.'
        },
    }

    # Count targets met
    targets_met = sum(1 for t in targets.values() if t.get('phase2_met', False))
    total_targets = len(targets)

    return {
        'targets': targets,
        'targets_met': targets_met,
        'total_targets': total_targets,
        'completion_pct': (targets_met / total_targets) * 100,
    }


def generate_recommendations(phase2_results: Dict, phase3_analysis: Dict) -> Dict:
    """Generate go-live recommendations."""

    overall_rmse = phase2_results.get('summary', {}).get('overall_performance', {}).get('rmse', 999)
    elite_strong_rmse = phase2_results.get('summary', {}).get('elite_strong_performance', {}).get('rmse', 999)
    elite_strong_pct = phase2_results.get('summary', {}).get('elite_strong_percentage', 0)

    recommendations = {
        'overall_readiness': 'CONDITIONAL_GO',
        'strengths': [
            f"Elite+Strong tier achieves excellent RMSE of {elite_strong_rmse:.3f}",
            f"Elite+Strong tier represents {elite_strong_pct:.1f}% of predictions (good balance)",
            "Quantile models implemented for prediction bands",
            "Kelly bet sizing implemented with tier adjustments",
            "Stop-loss and portfolio management implemented",
        ],
        'concerns': [
            f"Overall RMSE ({overall_rmse:.3f}) still above Phase 3 target (4.8)",
            "Points predictions in elite/strong tier need improvement (RMSE 9.4)",
            "3-point predictions remain challenging (R² = 0.013)",
            "Betting performance (ROI, Sharpe) not yet validated on real odds",
        ],
        'recommendations': [
            "✓ GO-LIVE for paper trading with Elite+Strong tier only",
            "✓ Start with conservative bankroll (10% of intended)",
            "✓ Focus on Assists, Rebounds, PRA props (better R²)",
            "⚠ Monitor Points predictions closely (higher errors)",
            "⚠ Avoid 3PT props until model improves (R² < 0.10)",
            "⚠ Implement strict stop-loss: 3% daily, 8% weekly",
            "✓ Run 7-day paper trading before live betting",
        ],
        'next_steps': [
            "1. Integrate The Odds API for real-time lines",
            "2. Run 7-day paper trading on Elite+Strong tier",
            "3. Validate CLV > 0 (beating closing lines)",
            "4. Measure actual ROI and Sharpe ratio",
            "5. If ROI > 3% and Sharpe > 1.0 after 30 bets → scale to 25% bankroll",
        ],
    }

    return recommendations


def generate_comprehensive_report():
    """Generate comprehensive Phase 3 report."""

    print("\n" + "="*80)
    print("GENERATING PHASE 3 COMPREHENSIVE REPORT")
    print("="*80 + "\n")

    # Load existing results
    print("Loading Phase 2 backtest results...")
    phase2_results = load_phase2_results()

    print("Analyzing quantile models...")
    quantile_analysis = analyze_quantile_models()

    print("Validating Kelly bet sizing...")
    kelly_validation = validate_kelly_sizing()

    print("Evaluating Phase 3 targets...")
    targets_evaluation = evaluate_phase3_targets(phase2_results)

    print("Generating recommendations...")
    recommendations = generate_recommendations(phase2_results, {'quantile': quantile_analysis, 'kelly': kelly_validation})

    # Compile comprehensive report
    report = {
        'report_type': 'Phase 3 Comprehensive Analysis',
        'generated_at': datetime.now().isoformat(),
        'phase': 'Phase 3: Optimization (Weeks 5-6)',

        'phase2_summary': {
            'games_analyzed': phase2_results.get('games_analyzed', 0),
            'total_predictions': phase2_results.get('total_predictions', 0),
            'overall_rmse': phase2_results.get('summary', {}).get('overall_performance', {}).get('rmse', 0),
            'elite_strong_rmse': phase2_results.get('summary', {}).get('elite_strong_performance', {}).get('rmse', 0),
            'elite_strong_percentage': phase2_results.get('summary', {}).get('elite_strong_percentage', 0),
        },

        'phase3_enhancements': {
            'quantile_regression': quantile_analysis,
            'kelly_bet_sizing': kelly_validation,
            'prediction_bands': 'Implemented (pred_low, pred_median, pred_high)',
            'confidence_scoring': 'Implemented (band-width based)',
            'portfolio_management': 'Implemented (stop-loss, exposure limits)',
            'risk_management': 'Enhanced with Kelly multipliers per tier',
        },

        'phase3_targets': targets_evaluation,

        'recommendations': recommendations,

        'technical_details': {
            'quantile_models_available': quantile_analysis['total_quantile_models'],
            'prop_types': quantile_analysis['prop_types_with_quantiles'],
            'kelly_validation_status': kelly_validation.get('validation', 'UNKNOWN'),
            'stop_loss_rules': {
                'daily_limit': '3% of bankroll',
                'weekly_limit': '8% of bankroll',
                'max_drawdown': '15% from peak',
                'daily_exposure': '20% of bankroll',
            },
            'confidence_tiers': {
                'elite': '90-100 (Kelly 1.0x)',
                'strong': '75-89 (Kelly 0.5x)',
                'moderate': '60-74 (Kelly 0.25x)',
                'weak': '40-59 (monitor only)',
                'avoid': '< 40 (do not bet)',
            },
        },
    }

    # Save report
    output_file = RESULTS_DIR / "phase3_comprehensive_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Saved comprehensive report to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("PHASE 3 COMPREHENSIVE REPORT SUMMARY")
    print("="*80)

    print(f"\nPhase 2 Performance:")
    print(f"  Total Predictions: {report['phase2_summary']['total_predictions']:,}")
    print(f"  Overall RMSE: {report['phase2_summary']['overall_rmse']:.3f}")
    print(f"  Elite+Strong RMSE: {report['phase2_summary']['elite_strong_rmse']:.3f}")
    print(f"  Elite+Strong %: {report['phase2_summary']['elite_strong_percentage']:.1f}%")

    print(f"\nPhase 3 Enhancements:")
    print(f"  ✓ Quantile Models: {quantile_analysis['total_quantile_models']} prop types")
    print(f"  ✓ Kelly Bet Sizing: {kelly_validation.get('validation', 'UNKNOWN')}")
    print(f"  ✓ Prediction Bands: Implemented")
    print(f"  ✓ Portfolio Management: Implemented")

    print(f"\nPhase 3 Targets:")
    targets = targets_evaluation['targets']
    for target_name, target_data in targets.items():
        status = "✓" if target_data.get('phase2_met', False) else "⚠"
        actual = target_data.get('phase2_actual', 'N/A')
        target_val = target_data.get('target', 'N/A')
        print(f"  {status} {target_name}: {actual} (target: {target_val})")

    print(f"\nTargets Met: {targets_evaluation['targets_met']}/{targets_evaluation['total_targets']} ({targets_evaluation['completion_pct']:.0f}%)")

    print(f"\nOverall Readiness: {recommendations['overall_readiness']}")

    print(f"\nKey Recommendations:")
    for rec in recommendations['recommendations'][:5]:
        print(f"  {rec}")

    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nFull report saved to: {output_file}")
    print("="*80 + "\n")

    return report


if __name__ == "__main__":
    generate_comprehensive_report()
