"""
Continuous Learning Orchestrator

Coordinates the continuous learning pipeline:
1. Scheduled settlement of predictions
2. Periodic drift detection
3. Automated retraining when performance degrades
4. Status reporting and alerting
"""

import sys
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, List

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prop_tracker import PropTracker
from .settlement_service import SettlementService
from .drift_detector import DriftDetector
from .model_registry import ModelRegistry
from .incremental_trainer import IncrementalTrainer

# Try to import APScheduler for background jobs
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False
    print("Note: APScheduler not installed. Install with: pip install apscheduler")


class ContinuousLearningOrchestrator:
    """Coordinates the continuous learning pipeline."""

    DEFAULT_CONFIG = {
        'settlement_interval_hours': 6,        # Settle predictions every 6 hours
        'drift_check_interval_hours': 24,      # Check drift daily
        'retrain_check_interval_hours': 168,   # Check if retraining needed weekly
        'retrain_threshold_samples': 200,      # Min samples to trigger retrain
        'min_days_between_retrain': 7,         # Minimum days between retrains
        'auto_retrain': True,                  # Automatically retrain on drift
    }

    def __init__(
        self,
        prop_tracker: PropTracker = None,
        settlement_service: SettlementService = None,
        drift_detector: DriftDetector = None,
        model_registry: ModelRegistry = None,
        incremental_trainer: IncrementalTrainer = None,
        config: Dict = None
    ):
        """Initialize the orchestrator.

        Args:
            prop_tracker: PropTracker instance
            settlement_service: SettlementService instance
            drift_detector: DriftDetector instance
            model_registry: ModelRegistry instance
            incremental_trainer: IncrementalTrainer instance
            config: Configuration dict (merged with defaults)
        """
        self.prop_tracker = prop_tracker or PropTracker()
        self.settlement = settlement_service or SettlementService(self.prop_tracker)
        self.drift = drift_detector or DriftDetector(self.prop_tracker)
        self.registry = model_registry or ModelRegistry()
        self.trainer = incremental_trainer or IncrementalTrainer(
            self.prop_tracker, self.registry
        )

        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # Scheduler
        self.scheduler = BackgroundScheduler() if HAS_SCHEDULER else None
        self._is_running = False

        # State tracking
        self._last_settlement = None
        self._last_drift_check = None
        self._last_retrain = None
        self._event_history = []

        # Callbacks for alerting
        self._alert_callbacks: List[Callable] = []

    def start(self):
        """Start the continuous learning scheduler."""
        if not HAS_SCHEDULER:
            print("Cannot start scheduler: APScheduler not installed")
            return False

        if self._is_running:
            print("Orchestrator already running")
            return True

        # Schedule settlement every N hours
        self.scheduler.add_job(
            self.run_settlement,
            'interval',
            hours=self.config['settlement_interval_hours'],
            id='settlement',
            name='Settle Predictions'
        )

        # Schedule drift check daily
        self.scheduler.add_job(
            self.run_drift_check,
            'interval',
            hours=self.config['drift_check_interval_hours'],
            id='drift_check',
            name='Check Drift'
        )

        # Schedule retrain check weekly
        self.scheduler.add_job(
            self.run_retrain_check,
            'interval',
            hours=self.config['retrain_check_interval_hours'],
            id='retrain_check',
            name='Check Retraining'
        )

        self.scheduler.start()
        self._is_running = True

        self._log_event('started', 'Continuous Learning Orchestrator started')
        print("Continuous Learning Orchestrator started")
        return True

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler and self._is_running:
            self.scheduler.shutdown(wait=False)
            self._is_running = False
            self._log_event('stopped', 'Orchestrator stopped')
            print("Orchestrator stopped")

    def run_settlement(self) -> Dict:
        """Settle unsettled predictions.

        Returns:
            Settlement result dict
        """
        self._log_event('settlement_start', 'Starting settlement')

        try:
            # Settle yesterday's games and any older pending
            result = self.settlement.settle_all_pending(days_back=7)
            self._last_settlement = datetime.now()

            self._log_event('settlement_complete', f"Settled {result['total_settled']} predictions")

            return result

        except Exception as e:
            self._log_event('settlement_error', str(e))
            return {'error': str(e)}

    def run_drift_check(self) -> Dict:
        """Check for model drift and handle accordingly.

        Returns:
            Drift check result
        """
        self._log_event('drift_check_start', 'Starting drift check')

        try:
            drift_result = self.drift.check_drift(lookback_days=7)
            self._last_drift_check = datetime.now()

            # Handle alerts
            if drift_result['has_drift']:
                for alert in drift_result['alerts']:
                    self._handle_alert(alert)

                # Auto-retrain if configured and drift detected
                if self.config['auto_retrain']:
                    retrain_rec = self.drift.should_retrain()
                    if retrain_rec['should_retrain']:
                        self._log_event('auto_retrain_triggered',
                                       f"Auto-retraining triggered: {retrain_rec['reasons']}")
                        self.trigger_retraining()

            self._log_event('drift_check_complete',
                           f"Drift score: {drift_result['drift_score']}, Alerts: {len(drift_result['alerts'])}")

            return drift_result

        except Exception as e:
            self._log_event('drift_check_error', str(e))
            return {'error': str(e)}

    def run_retrain_check(self) -> Dict:
        """Check if models should be retrained based on new data.

        Returns:
            Retrain check result
        """
        self._log_event('retrain_check_start', 'Checking retraining status')

        try:
            status = self.trainer.get_training_status()

            models_to_retrain = [
                prop_type for prop_type, info in status.items()
                if info['should_retrain']
            ]

            result = {
                'status': status,
                'models_to_retrain': models_to_retrain,
            }

            if models_to_retrain:
                self._log_event('retrain_recommended',
                               f"Retraining recommended for: {models_to_retrain}")

            return result

        except Exception as e:
            self._log_event('retrain_check_error', str(e))
            return {'error': str(e)}

    def trigger_retraining(self, prop_types: List[str] = None) -> Dict:
        """Trigger model retraining.

        Args:
            prop_types: Specific prop types to retrain (default: all eligible)

        Returns:
            Retraining result dict
        """
        # Check minimum time between retrains
        if self._last_retrain:
            days_since = (datetime.now() - self._last_retrain).days
            if days_since < self.config['min_days_between_retrain']:
                return {
                    'skipped': True,
                    'reason': f"Only {days_since} days since last retrain (min: {self.config['min_days_between_retrain']})"
                }

        self._log_event('retraining_start', f"Starting retraining: {prop_types or 'all'}")

        try:
            if prop_types:
                results = {}
                for prop_type in prop_types:
                    version = self.trainer.retrain_prop_model(prop_type)
                    if version:
                        results[prop_type] = version
            else:
                results = self.trainer.retrain_all_prop_models()

            self._last_retrain = datetime.now()

            self._log_event('retraining_complete', f"Retrained {len(results)} models")

            return {
                'success': True,
                'models_retrained': results,
            }

        except Exception as e:
            self._log_event('retraining_error', str(e))
            return {'error': str(e)}

    def _handle_alert(self, alert: Dict):
        """Handle a drift alert.

        Args:
            alert: Alert dict from DriftDetector
        """
        self._log_event(f"alert_{alert['type']}", alert['message'])

        # Call registered alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")

    def register_alert_callback(self, callback: Callable):
        """Register a callback function for alerts.

        Args:
            callback: Function that takes an alert dict
        """
        self._alert_callbacks.append(callback)

    def _log_event(self, event_type: str, message: str):
        """Log an event.

        Args:
            event_type: Type of event
            message: Event message
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'message': message,
        }
        self._event_history.append(event)

        # Keep last 1000 events
        if len(self._event_history) > 1000:
            self._event_history = self._event_history[-1000:]

        print(f"[{event['timestamp']}] {event_type}: {message}")

    def get_status(self) -> Dict:
        """Get current status of the continuous learning system.

        Returns:
            Status dict with all component states
        """
        settlement_status = self.settlement.get_settlement_status()
        drift_result = self.drift.check_drift(lookback_days=7)
        training_status = self.trainer.get_training_status()

        return {
            'orchestrator': {
                'is_running': self._is_running,
                'last_settlement': self._last_settlement.isoformat() if self._last_settlement else None,
                'last_drift_check': self._last_drift_check.isoformat() if self._last_drift_check else None,
                'last_retrain': self._last_retrain.isoformat() if self._last_retrain else None,
            },
            'settlement': settlement_status,
            'drift': {
                'has_drift': drift_result['has_drift'],
                'drift_score': drift_result['drift_score'],
                'alerts_count': len(drift_result['alerts']),
                'calibration_error': drift_result.get('calibration_error', 0),
            },
            'training': training_status,
            'models': {
                model_type: self.registry.get_active_model(model_type)
                for model_type in self.registry.get_model_types()
            },
            'recent_events': self._event_history[-10:],
        }

    def get_event_history(self, limit: int = 50) -> List[Dict]:
        """Get recent event history.

        Args:
            limit: Maximum events to return

        Returns:
            List of recent events
        """
        return self._event_history[-limit:]

    def run_full_cycle(self) -> Dict:
        """Run a complete cycle manually (useful for testing).

        Returns:
            Results from all operations
        """
        results = {
            'settlement': self.run_settlement(),
            'drift_check': self.run_drift_check(),
            'retrain_check': self.run_retrain_check(),
        }

        # Trigger retraining if recommended
        if results['drift_check'].get('has_drift'):
            retrain_rec = self.drift.should_retrain()
            if retrain_rec['should_retrain']:
                results['retraining'] = self.trigger_retraining()

        return results

    def print_status(self):
        """Print a formatted status report."""
        status = self.get_status()

        print(f"\n{'='*60}")
        print(f"CONTINUOUS LEARNING SYSTEM STATUS")
        print(f"{'='*60}")

        orch = status['orchestrator']
        print(f"\nOrchestrator:")
        print(f"  Running: {'Yes' if orch['is_running'] else 'No'}")
        print(f"  Last Settlement: {orch['last_settlement'] or 'Never'}")
        print(f"  Last Drift Check: {orch['last_drift_check'] or 'Never'}")
        print(f"  Last Retrain: {orch['last_retrain'] or 'Never'}")

        print(f"\nSettlement:")
        print(f"  Pending: {status['settlement']['total_pending']} predictions")
        by_date = status['settlement'].get('by_date', {})
        if by_date:
            for date, count in sorted(by_date.items())[:5]:
                print(f"    {date}: {count}")

        drift = status['drift']
        print(f"\nDrift Detection:")
        print(f"  Has Drift: {'Yes' if drift['has_drift'] else 'No'}")
        print(f"  Drift Score: {drift['drift_score']}/100")
        print(f"  Calibration Error: {drift['calibration_error']:.3f}")
        print(f"  Active Alerts: {drift['alerts_count']}")

        print(f"\nTraining Status:")
        for prop_type, info in status['training'].items():
            retrain = 'Yes' if info['should_retrain'] else 'No'
            print(f"  {prop_type}: {info['samples']} samples (retrain: {retrain})")

        print(f"\nRecent Events:")
        for event in status['recent_events'][-5:]:
            print(f"  [{event['timestamp']}] {event['type']}: {event['message'][:50]}")

        print(f"\n{'='*60}\n")


def create_orchestrator() -> ContinuousLearningOrchestrator:
    """Convenience function to create a fully configured orchestrator."""
    return ContinuousLearningOrchestrator()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Continuous Learning Orchestrator")
    parser.add_argument("--start", action="store_true", help="Start the scheduler")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--settle", action="store_true", help="Run settlement now")
    parser.add_argument("--drift", action="store_true", help="Run drift check now")
    parser.add_argument("--retrain", action="store_true", help="Run retraining now")
    parser.add_argument("--cycle", action="store_true", help="Run full cycle")

    args = parser.parse_args()

    orchestrator = ContinuousLearningOrchestrator()

    if args.status:
        orchestrator.print_status()
    elif args.settle:
        result = orchestrator.run_settlement()
        print(json.dumps(result, indent=2, default=str))
    elif args.drift:
        result = orchestrator.run_drift_check()
        print(json.dumps(result, indent=2, default=str))
    elif args.retrain:
        result = orchestrator.trigger_retraining()
        print(json.dumps(result, indent=2, default=str))
    elif args.cycle:
        result = orchestrator.run_full_cycle()
        print(json.dumps(result, indent=2, default=str))
    elif args.start:
        if orchestrator.start():
            print("Press Ctrl+C to stop...")
            try:
                import time
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                orchestrator.stop()
        else:
            print("Failed to start orchestrator")
    else:
        parser.print_help()
