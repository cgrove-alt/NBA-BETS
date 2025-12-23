"""
Drift Detector for Continuous Learning

Monitors model performance and detects drift by:
1. Comparing current win rate to historical baseline
2. Calculating Expected Calibration Error (ECE)
3. Tracking ROI trends
4. Triggering alerts when thresholds are exceeded
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prop_tracker import PropTracker


@dataclass
class DriftAlert:
    """Represents a drift detection alert."""
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    recommendation: str
    metric_value: float
    threshold_value: float
    detected_at: str = None

    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            'type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'recommendation': self.recommendation,
            'metric_value': self.metric_value,
            'threshold_value': self.threshold_value,
            'detected_at': self.detected_at,
        }


class DriftDetector:
    """Monitors model performance and detects drift."""

    # Configurable thresholds
    DEFAULT_THRESHOLDS = {
        'accuracy_drop': 0.05,           # 5% accuracy drop triggers alert
        'accuracy_critical': 0.10,       # 10% drop is critical
        'calibration_ece': 0.08,         # ECE > 8% triggers recalibration
        'calibration_critical': 0.15,    # ECE > 15% is critical
        'roi_threshold': -0.05,          # -5% ROI triggers review
        'roi_critical': -0.10,           # -10% ROI is critical
        'min_samples': 50,               # Minimum samples for drift detection
        'min_samples_warning': 20,       # Warn if samples below this
    }

    def __init__(self, prop_tracker: PropTracker = None,
                 baseline_metrics: Dict = None,
                 thresholds: Dict = None):
        """Initialize drift detector.

        Args:
            prop_tracker: PropTracker instance for accessing performance data
            baseline_metrics: Historical baseline metrics to compare against
            thresholds: Custom thresholds (merged with defaults)
        """
        self.prop_tracker = prop_tracker or PropTracker()
        self.baseline = baseline_metrics or {
            'win_rate': 0.52,  # 52% baseline (slightly above coin flip)
            'avg_confidence': 65.0,
            'avg_edge': 5.0,
        }
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._alert_history = []

    def check_drift(self, lookback_days: int = 7) -> Dict:
        """Check for performance drift.

        Args:
            lookback_days: Number of days to analyze

        Returns:
            Dict with drift analysis results
        """
        summary = self.prop_tracker.get_performance_summary(days=lookback_days)
        calibration = self.prop_tracker.get_calibration_data(days=lookback_days)

        alerts = []
        total_predictions = summary.get('total_predictions', 0)

        # Check sample size
        if total_predictions < self.thresholds['min_samples_warning']:
            alerts.append(DriftAlert(
                alert_type='low_sample_size',
                severity='low',
                message=f"Only {total_predictions} predictions in last {lookback_days} days",
                recommendation='Accumulate more data before making drift decisions',
                metric_value=total_predictions,
                threshold_value=self.thresholds['min_samples'],
            ))

        # Only check drift metrics if we have enough samples
        if total_predictions >= self.thresholds['min_samples']:
            # Check win rate vs baseline
            current_accuracy = summary.get('win_rate', 0)
            baseline_accuracy = self.baseline.get('win_rate', 0.52)
            accuracy_drop = baseline_accuracy - current_accuracy

            if accuracy_drop > self.thresholds['accuracy_critical']:
                alerts.append(DriftAlert(
                    alert_type='accuracy_drift',
                    severity='critical',
                    message=f"Critical accuracy drop: {current_accuracy:.1%} vs baseline {baseline_accuracy:.1%}",
                    recommendation='Immediately investigate and consider model retraining',
                    metric_value=current_accuracy,
                    threshold_value=baseline_accuracy - self.thresholds['accuracy_critical'],
                ))
            elif accuracy_drop > self.thresholds['accuracy_drop']:
                alerts.append(DriftAlert(
                    alert_type='accuracy_drift',
                    severity='high',
                    message=f"Win rate dropped: {current_accuracy:.1%} vs baseline {baseline_accuracy:.1%}",
                    recommendation='Consider model retraining',
                    metric_value=current_accuracy,
                    threshold_value=baseline_accuracy - self.thresholds['accuracy_drop'],
                ))

            # Check calibration
            calibration_error = self._calculate_ece(calibration)

            if calibration_error > self.thresholds['calibration_critical']:
                alerts.append(DriftAlert(
                    alert_type='calibration_drift',
                    severity='critical',
                    message=f"Critical calibration error: ECE = {calibration_error:.3f}",
                    recommendation='Urgent: Recalibrate confidence scores',
                    metric_value=calibration_error,
                    threshold_value=self.thresholds['calibration_critical'],
                ))
            elif calibration_error > self.thresholds['calibration_ece']:
                alerts.append(DriftAlert(
                    alert_type='calibration_drift',
                    severity='medium',
                    message=f"Expected Calibration Error: {calibration_error:.3f}",
                    recommendation='Recalibrate confidence scores',
                    metric_value=calibration_error,
                    threshold_value=self.thresholds['calibration_ece'],
                ))

            # Check by confidence level for miscalibration
            by_confidence = summary.get('by_confidence', {})
            for level, stats in by_confidence.items():
                if stats.get('total', 0) >= 10:
                    expected_rate = {'high': 0.75, 'medium': 0.60, 'low': 0.50}.get(level, 0.5)
                    actual_rate = stats.get('win_rate', 0)

                    if abs(expected_rate - actual_rate) > 0.15:
                        alerts.append(DriftAlert(
                            alert_type='confidence_miscalibration',
                            severity='medium',
                            message=f"{level.upper()} confidence: {actual_rate:.1%} actual vs {expected_rate:.0%} expected",
                            recommendation=f'Adjust {level} confidence thresholds',
                            metric_value=actual_rate,
                            threshold_value=expected_rate,
                        ))

        # Store alerts in history
        for alert in alerts:
            self._alert_history.append(alert.to_dict())

        # Calculate drift score (0-100, higher = more drift)
        drift_score = self._calculate_drift_score(alerts)

        return {
            'has_drift': len([a for a in alerts if a.severity in ('high', 'critical')]) > 0,
            'drift_score': drift_score,
            'alerts': [a.to_dict() for a in alerts],
            'metrics': summary,
            'calibration_error': self._calculate_ece(calibration),
            'sample_size': total_predictions,
            'lookback_days': lookback_days,
            'analysis_timestamp': datetime.now().isoformat(),
        }

    def _calculate_ece(self, calibration_data: List[Dict]) -> float:
        """Calculate Expected Calibration Error.

        ECE measures how well-calibrated the confidence scores are.
        Lower is better (0 = perfect calibration).

        Args:
            calibration_data: List of calibration bucket data

        Returns:
            ECE value (0.0 to 1.0)
        """
        if not calibration_data:
            return 0.0

        total_samples = sum(b.get('total', 0) for b in calibration_data)
        if total_samples == 0:
            return 0.0

        ece = 0.0
        for bucket in calibration_data:
            weight = bucket.get('total', 0) / total_samples
            predicted_rate = bucket.get('predicted_win_rate', 0.5)
            actual_rate = bucket.get('actual_win_rate', 0.5)
            error = abs(predicted_rate - actual_rate)
            ece += weight * error

        return ece

    def _calculate_drift_score(self, alerts: List[DriftAlert]) -> float:
        """Calculate overall drift score from alerts.

        Args:
            alerts: List of drift alerts

        Returns:
            Drift score 0-100 (higher = more drift)
        """
        severity_weights = {
            'low': 5,
            'medium': 15,
            'high': 30,
            'critical': 50,
        }

        score = sum(severity_weights.get(a.severity, 0) for a in alerts)
        return min(100, score)

    def get_trend(self, metric: str = 'win_rate', periods: int = 4,
                  period_days: int = 7) -> Dict:
        """Get trend data for a specific metric over time.

        Args:
            metric: Metric to track ('win_rate', 'avg_confidence', etc.)
            periods: Number of periods to analyze
            period_days: Days per period

        Returns:
            Dict with trend analysis
        """
        trend_data = []

        for i in range(periods):
            # Calculate date range for this period
            end_days = i * period_days
            start_days = (i + 1) * period_days

            # Get summary for this period
            # Note: This is a simplification - in production, you'd want
            # to query the database with specific date ranges
            summary = self.prop_tracker.get_performance_summary(days=start_days)

            if summary:
                value = summary.get(metric, 0)
                trend_data.append({
                    'period': i + 1,
                    'value': value,
                    'total': summary.get('total_predictions', 0),
                })

        # Calculate trend direction
        if len(trend_data) >= 2:
            recent = trend_data[0].get('value', 0)
            older = trend_data[-1].get('value', 0)

            if older > 0:
                change_pct = ((recent - older) / older) * 100
            else:
                change_pct = 0

            direction = 'improving' if change_pct > 5 else ('declining' if change_pct < -5 else 'stable')
        else:
            change_pct = 0
            direction = 'insufficient_data'

        return {
            'metric': metric,
            'periods': trend_data,
            'change_percent': round(change_pct, 1),
            'direction': direction,
        }

    def get_alert_history(self, limit: int = 20) -> List[Dict]:
        """Get recent alert history.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        return self._alert_history[-limit:]

    def should_retrain(self, lookback_days: int = 7) -> Dict:
        """Determine if model retraining is recommended.

        Args:
            lookback_days: Days to analyze

        Returns:
            Dict with retraining recommendation
        """
        drift_result = self.check_drift(lookback_days)

        # Count serious alerts
        critical_alerts = sum(1 for a in drift_result['alerts'] if a['severity'] == 'critical')
        high_alerts = sum(1 for a in drift_result['alerts'] if a['severity'] == 'high')

        # Retraining decision logic
        should_retrain = (
            critical_alerts >= 1 or
            high_alerts >= 2 or
            drift_result.get('drift_score', 0) >= 50
        )

        return {
            'should_retrain': should_retrain,
            'urgency': 'immediate' if critical_alerts >= 1 else ('high' if should_retrain else 'none'),
            'reasons': [a['message'] for a in drift_result['alerts'] if a['severity'] in ('high', 'critical')],
            'drift_score': drift_result.get('drift_score', 0),
        }

    def update_baseline(self, new_baseline: Dict):
        """Update baseline metrics.

        Args:
            new_baseline: New baseline values
        """
        self.baseline.update(new_baseline)

    def print_report(self, lookback_days: int = 7):
        """Print a formatted drift detection report."""
        result = self.check_drift(lookback_days)

        print(f"\n{'='*60}")
        print(f"DRIFT DETECTION REPORT - Last {lookback_days} Days")
        print(f"{'='*60}")

        print(f"\nDrift Score: {result['drift_score']}/100")
        print(f"Sample Size: {result['sample_size']} predictions")
        print(f"Calibration Error (ECE): {result['calibration_error']:.3f}")

        if result['alerts']:
            print(f"\nAlerts ({len(result['alerts'])}):")
            for alert in result['alerts']:
                severity_icons = {'low': '⚪', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
                icon = severity_icons.get(alert['severity'], '⚪')
                print(f"  {icon} [{alert['severity'].upper()}] {alert['message']}")
                print(f"     → {alert['recommendation']}")
        else:
            print("\n✅ No significant drift detected")

        retrain = self.should_retrain(lookback_days)
        print(f"\nRetraining Recommendation: {'YES' if retrain['should_retrain'] else 'NO'}")
        if retrain['urgency'] != 'none':
            print(f"  Urgency: {retrain['urgency'].upper()}")

        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # CLI for drift detection
    import argparse

    parser = argparse.ArgumentParser(description="Detect model drift")
    parser.add_argument("--days", type=int, default=7, help="Days to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    detector = DriftDetector()
    detector.print_report(args.days)
