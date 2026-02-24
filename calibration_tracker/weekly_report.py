"""
Weekly Report Generator - Auto-Generated Weekly Performance Reports

Generates structured weekly reports every Monday with:
- Period summary (dates, total predictions, record, ROI, CLV)
- By-prop-type breakdown (hit rate, bias, edge quality)
- By-player-tier breakdown
- ECE score and trend
- Top/worst performing segments
- Active adjustments summary
- Actionable recommendations
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from .database import CalibrationDatabase
from .bias_analyzer import BiasAnalyzer, BiasReport
from .calibration_adjuster import CalibrationAdjuster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeeklyReportGenerator:
    """
    Generate weekly calibration performance reports.

    Reports are auto-generated every Monday, covering the prior Mon-Sun period.
    """

    def __init__(self, db: CalibrationDatabase = None):
        self.db = db or CalibrationDatabase()
        self.analyzer = BiasAnalyzer(self.db)
        self.adjuster = CalibrationAdjuster(self.db)

    def generate_weekly_report(self, week_ending: str = None) -> dict:
        """
        Generate a weekly performance report.

        Args:
            week_ending: The Sunday that ends the week (YYYY-MM-DD).
                         Defaults to most recent Sunday.

        Returns:
            Structured weekly report dict, also saved to database.
        """
        # Determine the week boundaries
        if week_ending:
            end_date = datetime.strptime(week_ending, '%Y-%m-%d')
        else:
            today = datetime.now()
            # Find most recent Sunday
            days_since_sunday = (today.weekday() + 1) % 7
            end_date = today - timedelta(days=days_since_sunday)

        start_date = end_date - timedelta(days=6)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        logger.info(f"Generating weekly report for {start_str} to {end_str}")

        # Run bias analysis for the week
        report = self.analyzer.analyze(start_date=start_str, end_date=end_str)

        # Get previous week for comparison
        prev_end = start_date - timedelta(days=1)
        prev_start = prev_end - timedelta(days=6)
        prev_report = self.analyzer.analyze(
            start_date=prev_start.strftime('%Y-%m-%d'),
            end_date=prev_end.strftime('%Y-%m-%d'),
        )

        # Build the weekly report
        weekly = {
            'week_ending': end_str,
            'week_start': start_str,
            'generated_at': datetime.now().isoformat(),

            # Period summary
            'total_predictions': report.total_predictions,
            'matched_predictions': report.matched_predictions,
            'overall_hit_rate': report.overall_hit_rate,
            'overall_clv': report.overall_clv,
            'overall_roi': report.overall_roi,

            # ECE
            'ece': report.ece,
            'calibration_bins': report.calibration_bins,

            # ECE trend (this week vs last week)
            'ece_trend': {
                'current': report.ece,
                'previous': prev_report.ece,
                'change': round(report.ece - prev_report.ece, 4),
                'improved': report.ece < prev_report.ece if prev_report.total_predictions > 0 else None,
            },

            # By prop type
            'by_prop_type': {
                k: v.to_dict() for k, v in report.by_prop_type.items()
            },

            # By player tier
            'by_player_tier': {
                k: v.to_dict() for k, v in report.by_player_tier.items()
            },

            # By position
            'by_position': {
                k: v.to_dict() for k, v in report.by_position.items()
            },

            # Top and worst segments
            'top_segments': self._get_top_segments(report),
            'worst_segments': self._get_worst_segments(report),

            # Active adjustments
            'active_adjustments': self._get_adjustments_summary(),

            # Recommendations
            'recommendations': report.recommendations,

            # Comparison to previous week
            'vs_previous_week': {
                'hit_rate_change': round(
                    report.overall_hit_rate - prev_report.overall_hit_rate, 4
                ) if prev_report.total_predictions > 0 else None,
                'roi_change': round(
                    report.overall_roi - prev_report.overall_roi, 4
                ) if prev_report.total_predictions > 0 else None,
                'prediction_count_change': (
                    report.total_predictions - prev_report.total_predictions
                ),
            },
        }

        # Save to database
        self.db.insert_weekly_report(end_str, weekly)
        logger.info(
            f"Weekly report saved: {report.total_predictions} predictions, "
            f"{report.overall_hit_rate:.1%} hit rate, ECE={report.ece:.4f}"
        )

        return weekly

    def _get_top_segments(self, report: BiasReport, limit: int = 5) -> list[dict]:
        """Get top performing segments across all dimensions."""
        segments = []

        all_dimensions = [
            ('prop_type', report.by_prop_type),
            ('position', report.by_position),
            ('player_tier', report.by_player_tier),
            ('minutes_bucket', report.by_minutes_bucket),
            ('game_type', report.by_game_type),
        ]

        for dim_name, dim_data in all_dimensions:
            for value, analysis in dim_data.items():
                if analysis.sample_size >= self.analyzer.MIN_SAMPLE_SIZE:
                    segments.append({
                        'dimension': dim_name,
                        'value': value,
                        'hit_rate': analysis.hit_rate,
                        'sample_size': analysis.sample_size,
                        'edge_quality': analysis.edge_quality,
                        'roi': analysis.roi_estimate,
                    })

        # Sort by hit rate descending
        segments.sort(key=lambda x: x['hit_rate'], reverse=True)
        return segments[:limit]

    def _get_worst_segments(self, report: BiasReport, limit: int = 5) -> list[dict]:
        """Get worst performing segments across all dimensions."""
        segments = []

        all_dimensions = [
            ('prop_type', report.by_prop_type),
            ('position', report.by_position),
            ('player_tier', report.by_player_tier),
            ('minutes_bucket', report.by_minutes_bucket),
            ('game_type', report.by_game_type),
        ]

        for dim_name, dim_data in all_dimensions:
            for value, analysis in dim_data.items():
                if analysis.sample_size >= self.analyzer.MIN_SAMPLE_SIZE:
                    segments.append({
                        'dimension': dim_name,
                        'value': value,
                        'hit_rate': analysis.hit_rate,
                        'sample_size': analysis.sample_size,
                        'edge_quality': analysis.edge_quality,
                        'roi': analysis.roi_estimate,
                    })

        # Sort by hit rate ascending (worst first)
        segments.sort(key=lambda x: x['hit_rate'])
        return segments[:limit]

    def _get_adjustments_summary(self) -> list[dict]:
        """Get summary of active calibration adjustments."""
        adjustments = self.adjuster.get_all_active_adjustments()
        return [
            {
                'dimension': adj.dimension,
                'value': adj.dimension_value,
                'bias': round(adj.bias, 2),
                'adjustment': round(adj.adjustment, 2),
                'confidence_multiplier': adj.confidence_multiplier,
                'sample_size': adj.sample_size,
            }
            for adj in adjustments
        ]
