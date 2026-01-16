#!/usr/bin/env python3
"""
Task 3.1 Backtest Results Analysis

This script analyzes the backtest results to determine if the new player impact
metrics (player_impact_metric, opponent_def_impact) improved model performance.

Success Criteria:
- Primary: ≥2% RMSE improvement (Target: ≥5%)
- Secondary: No regression in other metrics (MAE, R²)
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


def load_json_results(filepath: str) -> Dict[str, Any]:
    """Load backtest results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_improvement(baseline: float, new: float) -> float:
    """Calculate percentage improvement (positive = better)."""
    if baseline == 0:
        return 0.0
    return ((baseline - new) / baseline) * 100


def analyze_prop_type(prop_name: str, baseline_data: Dict, new_data: Dict) -> Dict[str, Any]:
    """Analyze improvements for a specific prop type."""
    results = {
        'prop_type': prop_name,
        'metrics': {}
    }

    # Extract metrics
    for metric in ['rmse', 'mae', 'r2', 'accuracy']:
        baseline_val = baseline_data.get(metric)
        new_val = new_data.get(metric)

        if baseline_val is not None and new_val is not None:
            # For RMSE and MAE, lower is better
            if metric in ['rmse', 'mae']:
                improvement = calculate_improvement(baseline_val, new_val)
            # For R² and accuracy, higher is better
            else:
                improvement = -calculate_improvement(baseline_val, new_val)

            results['metrics'][metric] = {
                'baseline': baseline_val,
                'new': new_val,
                'improvement_pct': improvement,
                'improved': improvement > 0
            }

    return results


def format_results_table(analysis_results: Dict[str, Any]) -> str:
    """Format analysis results as a markdown table."""
    lines = []
    lines.append("## Backtest Results Comparison")
    lines.append("")
    lines.append("| Prop Type | Metric | Baseline | New | Improvement | Status |")
    lines.append("|-----------|--------|----------|-----|-------------|--------|")

    for prop_name, prop_data in analysis_results.items():
        if prop_name == 'summary':
            continue

        for metric_name, metric_data in prop_data['metrics'].items():
            baseline = metric_data['baseline']
            new = metric_data['new']
            improvement = metric_data['improvement_pct']
            improved = metric_data['improved']

            status = "✅" if improved else "❌"

            lines.append(
                f"| {prop_name:9s} | {metric_name:6s} | "
                f"{baseline:8.4f} | {new:8.4f} | "
                f"{improvement:+7.2f}% | {status:6s} |"
            )

    return "\n".join(lines)


def generate_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate overall summary of improvements."""
    rmse_improvements = []
    mae_improvements = []

    for prop_name, prop_data in analysis_results.items():
        if prop_name == 'summary':
            continue

        if 'rmse' in prop_data['metrics']:
            rmse_improvements.append(prop_data['metrics']['rmse']['improvement_pct'])

        if 'mae' in prop_data['metrics']:
            mae_improvements.append(prop_data['metrics']['mae']['improvement_pct'])

    avg_rmse_improvement = sum(rmse_improvements) / len(rmse_improvements) if rmse_improvements else 0
    avg_mae_improvement = sum(mae_improvements) / len(mae_improvements) if mae_improvements else 0

    # Success criteria check
    success_criteria_met = avg_rmse_improvement >= 2.0
    target_met = avg_rmse_improvement >= 5.0

    return {
        'avg_rmse_improvement': avg_rmse_improvement,
        'avg_mae_improvement': avg_mae_improvement,
        'rmse_improvements': rmse_improvements,
        'mae_improvements': mae_improvements,
        'success_criteria_met': success_criteria_met,
        'target_met': target_met,
        'num_prop_types': len(rmse_improvements)
    }


def generate_markdown_report(analysis_results: Dict[str, Any], summary: Dict[str, Any]) -> str:
    """Generate complete markdown report."""
    lines = []

    lines.append("# Task 3.1 Backtest Results Analysis")
    lines.append("")
    lines.append(f"**Date**: {os.popen('date').read().strip()}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"**Average RMSE Improvement**: {summary['avg_rmse_improvement']:+.2f}%")
    lines.append(f"**Average MAE Improvement**: {summary['avg_mae_improvement']:+.2f}%")
    lines.append(f"**Prop Types Tested**: {summary['num_prop_types']}")
    lines.append("")

    if summary['target_met']:
        lines.append("✅ **TARGET MET** (≥5% RMSE improvement)")
    elif summary['success_criteria_met']:
        lines.append("✅ **SUCCESS CRITERIA MET** (≥2% RMSE improvement)")
    else:
        lines.append("❌ **SUCCESS CRITERIA NOT MET** (<2% RMSE improvement)")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed results table
    lines.append(format_results_table(analysis_results))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Individual prop type breakdown
    lines.append("## Detailed Breakdown by Prop Type")
    lines.append("")

    for prop_name, prop_data in analysis_results.items():
        if prop_name == 'summary':
            continue

        lines.append(f"### {prop_name.upper()}")
        lines.append("")

        for metric_name, metric_data in prop_data['metrics'].items():
            improvement = metric_data['improvement_pct']
            status = "✅ Improved" if metric_data['improved'] else "❌ Regressed"

            lines.append(f"**{metric_name.upper()}**: {improvement:+.2f}% {status}")
            lines.append(f"- Baseline: {metric_data['baseline']:.4f}")
            lines.append(f"- New: {metric_data['new']:.4f}")
            lines.append("")

    lines.append("---")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    if summary['success_criteria_met']:
        lines.append("### ✅ Task 3.1 Complete")
        lines.append("")
        lines.append("The player impact metrics integration successfully improved model performance:")
        lines.append("")
        lines.append("1. **Mark task as complete** in plan.md")
        lines.append("2. **Update documentation** with actual performance gains")
        lines.append("3. **Proceed to Task 3.2** (Quantile Regression)")
        lines.append("")
        lines.append("**Key Learnings:**")
        lines.append("- RAPTOR metrics provide valuable predictive signal")
        lines.append("- Team enrichment strategy effective")
        lines.append("- Feature integration successful")
    else:
        lines.append("### ⚠️ Task 3.1 Incomplete - Further Investigation Needed")
        lines.append("")
        lines.append("The impact metrics did not meet success criteria. Consider:")
        lines.append("")
        lines.append("1. **Feature Importance Analysis**: Check if models are using impact features")
        lines.append("2. **Feature Engineering**: Try interaction features (impact × usage rate)")
        lines.append("3. **Data Quality**: Verify RAPTOR data relevance for 2024-25 season")
        lines.append("4. **Alternative Metrics**: Explore paid data sources (BBall-Index, etc.)")
        lines.append("5. **Model Retraining**: Retrain models from scratch with new features")
        lines.append("")
        lines.append("**Options:**")
        lines.append("- Option A: Investigate and iterate (4-8 hours)")
        lines.append("- Option B: Document as-is, proceed to Task 3.2")
        lines.append("- Option C: Defer to future sprint, mark as partial completion")

    lines.append("")

    return "\n".join(lines)


def main():
    """Main analysis routine."""
    # Baseline from Phase 2
    baseline_file = Path('backtest_results/phase2_backtest.json')

    # New results from Task 3.1 backtest (in current directory)
    new_file = Path('backtest_results_2025.json')

    if not baseline_file.exists():
        print(f"Error: Baseline file not found: {baseline_file}")
        return

    if not new_file.exists():
        print(f"Error: New results file not found: {new_file}")
        return

    print(f"Analyzing backtest results:")
    print(f"  Baseline: {baseline_file}")
    print(f"  New:      {new_file}")
    print()

    # Load results
    try:
        baseline_data = load_json_results(baseline_file)
    except FileNotFoundError:
        print(f"Warning: Baseline file not found, will compare to zero baseline")
        baseline_data = {}

    new_data = load_json_results(new_file)

    # Analyze each prop type
    analysis_results = {}

    for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra', 'minutes']:
        if prop_type in new_data:
            baseline_prop = baseline_data.get(prop_type, {})
            new_prop = new_data[prop_type]

            analysis_results[prop_type] = analyze_prop_type(
                prop_type, baseline_prop, new_prop
            )

    # Generate summary
    summary = generate_summary(analysis_results)
    analysis_results['summary'] = summary

    # Generate markdown report
    report = generate_markdown_report(analysis_results, summary)

    # Save report
    report_file = Path('.zenflow/tasks/model-improvements-v2-3065/task_3.1_backtest_analysis.md')
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"✓ Analysis complete!")
    print(f"  Report saved to: {report_file}")
    print()
    print("=" * 60)
    print(report)
    print("=" * 60)

    # Save JSON results
    json_file = report_file.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n✓ JSON results saved to: {json_file}")


if __name__ == '__main__':
    main()
