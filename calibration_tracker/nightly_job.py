#!/usr/bin/env python3
"""
Nightly Calibration Job

Run this script after all games complete (~1am ET) to:
1. Fetch actual stats for completed games
2. Match predictions to outcomes
3. Update calibration adjustments
4. Generate daily report

Usage:
    python -m calibration_tracker.nightly_job [--date YYYY-MM-DD]

Schedule with cron:
    0 1 * * * cd /path/to/project && python -m calibration_tracker.nightly_job
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta

from .calibration_service import CalibrationService
from .weekly_report import WeeklyReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_nightly_job(game_date: str = None, verbose: bool = False):
    """
    Run the nightly calibration job.

    Args:
        game_date: Date to process (YYYY-MM-DD), defaults to yesterday
        verbose: Print detailed output
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default to yesterday
    if not game_date:
        game_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print("=" * 60)
    print(f"NIGHTLY CALIBRATION JOB - {game_date}")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()

    try:
        # Initialize service
        service = CalibrationService()

        # Run the job
        results = service.run_nightly_job(game_date=game_date)

        # Print results
        print("RESULTS:")
        print("-" * 40)

        outcomes = results['steps'].get('outcomes', {})
        print(f"Predictions matched: {outcomes.get('matched', 0)}")
        print(f"Predictions not found: {outcomes.get('not_found', 0)}")
        print(f"DNP (Did Not Play): {outcomes.get('dnp', 0)}")
        print(f"Errors: {outcomes.get('errors', 0)}")
        print()

        print(f"Calibration adjustments generated: {results['steps'].get('adjustments_generated', 0)}")
        print(f"Daily report saved: {results['steps'].get('report_saved', False)}")
        print()

        # Print summary
        summary = results.get('summary', {})
        if summary.get('overall_hit_rate'):
            print(f"Overall hit rate: {summary['overall_hit_rate']:.1%}")
        print()

        # Get and print calibration report summary
        report = service.get_calibration_report(days=30)
        print("CALIBRATION SUMMARY (Last 30 Days):")
        print("-" * 40)
        print(f"Total predictions: {report['overall']['predictions']}")
        print(f"Hit rate: {report['overall']['hit_rate']}")
        print(f"CLV: {report['overall']['clv_avg']}")
        print(f"ROI estimate: {report['overall']['roi_estimate']}")
        print()

        # Print recommendations
        if report.get('recommendations'):
            print("RECOMMENDATIONS:")
            print("-" * 40)
            for rec in report['recommendations'][:5]:
                print(f"  - {rec}")
            print()

        # Print active adjustments
        adjustments = service.get_active_adjustments()
        if adjustments:
            print("ACTIVE ADJUSTMENTS:")
            print("-" * 40)
            for adj in adjustments[:10]:
                print(f"  {adj.dimension}:{adj.dimension_value}: "
                      f"bias={adj.bias:+.2f}, adj={adj.adjustment:+.2f}")
            print()

        # Weekly report: generate on Mondays
        today = datetime.now()
        if today.weekday() == 0:  # Monday
            print("\nMONDAY — Generating weekly report...")
            try:
                weekly_gen = WeeklyReportGenerator(service.db)
                weekly = weekly_gen.generate_weekly_report()
                print(f"Weekly report generated: {weekly['week_start']} to {weekly['week_ending']}")
                print(f"  Predictions: {weekly['total_predictions']}")
                print(f"  Hit Rate: {weekly['overall_hit_rate']:.1%}" if weekly['overall_hit_rate'] else "  Hit Rate: N/A")
                print(f"  ECE: {weekly['ece']:.4f}")
            except Exception as e:
                logger.warning(f"Weekly report generation failed: {e}")
                print(f"  WARNING: Weekly report failed: {e}")

        print("=" * 60)
        print(f"Completed at: {results.get('completed_at', datetime.now().isoformat())}")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Nightly job failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Run nightly calibration job')
    parser.add_argument(
        '--date', '-d',
        type=str,
        default=None,
        help='Date to process (YYYY-MM-DD), defaults to yesterday'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output'
    )

    args = parser.parse_args()

    sys.exit(run_nightly_job(game_date=args.date, verbose=args.verbose))


if __name__ == '__main__':
    main()
