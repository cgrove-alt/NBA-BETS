"""
Model Performance Watchdog Agent

Continuously monitors model health, detects drift and degradation,
and triggers corrective action before problems compound.
Wraps DriftDetector from continuous_learning/drift_detector.py.

Trigger: Daily at 1:30 AM ET (30 min after postgame).
         Weekly deeper analysis on Mondays.
"""

import os
import glob
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents.core.agent_base import AgentBase

logger = logging.getLogger(__name__)

# Model staleness threshold (days)
MAX_MODEL_AGE_DAYS = 14


class ModelWatchdogAgent(AgentBase):
    """
    Model Performance Watchdog Agent.

    Wraps DriftDetector with LLM reasoning to interpret drift signals,
    distinguish model issues from market regime changes, and recommend
    retraining when statistically justified.
    """

    AGENT_NAME = 'watchdog'
    DAILY_TOKEN_BUDGET = 30_000
    MAX_EXECUTION_SECONDS = 300

    def __init__(self, target_date: str = None, lookback_days: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.target_date = target_date or datetime.now().strftime('%Y-%m-%d')
        self.lookback_days = lookback_days
        self._drift_detector = None

    def _get_drift_detector(self):
        """Lazy-init DriftDetector."""
        if self._drift_detector is None:
            from continuous_learning.drift_detector import DriftDetector
            self._drift_detector = DriftDetector()
        return self._drift_detector

    def _load_system_prompt(self) -> str:
        """Load the version-controlled system prompt."""
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'prompts', 'watchdog.md'
        )
        try:
            with open(prompt_path) as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"System prompt not found at {prompt_path}, using default")
            return self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return (
            "You are the Model Performance Watchdog Agent for an NBA betting model. "
            "Interpret drift signals, distinguish model issues from market regime changes, "
            "and recommend retraining only when statistically justified. "
            "Be conservative with critical alerts. "
            "Output valid JSON with health_assessment, recommended_actions, and reasoning."
        )

    def _check_model_staleness(self) -> dict:
        """Check if model files are too old."""
        models_dir = Path(os.path.dirname(os.path.dirname(__file__))) / '..' / 'models'
        models_dir = models_dir.resolve()

        staleness = {
            'models_found': 0,
            'oldest_age_days': 0,
            'newest_age_days': 0,
            'is_stale': False,
            'model_files': [],
        }

        pkl_files = list(models_dir.glob('*.pkl'))
        if not pkl_files:
            staleness['is_stale'] = True
            return staleness

        now = datetime.now().timestamp()
        ages = []

        for f in pkl_files:
            try:
                age_days = (now - f.stat().st_mtime) / 86400
                ages.append(age_days)
                staleness['model_files'].append({
                    'name': f.name,
                    'age_days': round(age_days, 1),
                })
            except OSError:
                continue

        if ages:
            staleness['models_found'] = len(ages)
            staleness['oldest_age_days'] = round(max(ages), 1)
            staleness['newest_age_days'] = round(min(ages), 1)
            staleness['is_stale'] = max(ages) > MAX_MODEL_AGE_DAYS

        return staleness

    def _determine_health_status(self, drift_score: float) -> str:
        """Map drift score to health status."""
        if drift_score < 20:
            return 'healthy'
        elif drift_score <= 50:
            return 'degraded'
        else:
            return 'critical'

    def _interpret_with_llm(self, drift_result: dict, retrain_result: dict,
                            staleness: dict, postgame_msgs: list) -> dict:
        """
        Call LLM to interpret drift signals.

        Falls back to deterministic summary if LLM unavailable.
        """
        system_prompt = self._load_system_prompt()

        user_message = json.dumps({
            'task': 'Assess model health and recommend actions',
            'drift_analysis': drift_result,
            'retraining_assessment': retrain_result,
            'model_staleness': staleness,
            'recent_postgame_feedback': [
                m.payload for m in postgame_msgs[:5]
            ] if postgame_msgs else [],
            'check_date': self.target_date,
        }, indent=2, default=str)

        response = self.call_llm(system_prompt, user_message, max_tokens=2048)

        if not response:
            return {}

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"[{self.AGENT_NAME}] LLM returned invalid JSON")
            return {}

    def run(self) -> dict:
        """
        Core watchdog monitoring logic.

        1. Run DriftDetector.check_drift()
        2. Run DriftDetector.should_retrain()
        3. Check model file staleness
        4. Read recent postgame feedback
        5. If concerning, call LLM to interpret
        6. Return structured health report
        """
        logger.info(f"[{self.AGENT_NAME}] Running health check for {self.target_date}")

        drift_detector = self._get_drift_detector()

        # Step 1: Check drift
        try:
            drift_result = drift_detector.check_drift(lookback_days=self.lookback_days)
        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] check_drift() failed: {e}")
            drift_result = {
                'has_drift': False,
                'drift_score': 0,
                'alerts': [],
                'metrics': {},
                'calibration_error': 0,
                'sample_size': 0,
            }

        # Step 2: Retraining recommendation
        try:
            retrain_result = drift_detector.should_retrain(lookback_days=self.lookback_days)
        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] should_retrain() failed: {e}")
            retrain_result = {
                'should_retrain': False,
                'urgency': 'none',
                'reasons': [],
                'drift_score': 0,
            }

        # Step 3: Model staleness
        staleness = self._check_model_staleness()

        # Step 4: Read postgame feedback from bus
        postgame_msgs = self.get_messages(event_type='results_analyzed')

        # Step 5: Determine health status
        drift_score = drift_result.get('drift_score', 0)
        health_status = self._determine_health_status(drift_score)

        # Add staleness alert if needed
        alerts = drift_result.get('alerts', [])
        if staleness['is_stale']:
            alerts.append({
                'type': 'model_staleness',
                'severity': 'high',
                'message': f"Models are {staleness['oldest_age_days']} days old (max: {MAX_MODEL_AGE_DAYS})",
                'recommendation': 'retrain',
            })
            if health_status == 'healthy':
                health_status = 'degraded'

        # Step 6: LLM interpretation for concerning states
        llm_interpretation = {}
        has_serious = any(
            a.get('severity') in ('high', 'critical') for a in alerts
        )
        if has_serious:
            llm_interpretation = self._interpret_with_llm(
                drift_result, retrain_result, staleness, postgame_msgs
            )

        # Build metrics snapshot
        metrics = drift_result.get('metrics', {})
        metrics_snapshot = {
            'drift_score': drift_score,
            'calibration_error': drift_result.get('calibration_error', 0),
            'sample_size': drift_result.get('sample_size', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_predictions': metrics.get('total_predictions', 0),
        }

        # Determine model version from newest pkl file
        model_version = 'unknown'
        if staleness.get('model_files'):
            newest = min(staleness['model_files'], key=lambda x: x['age_days'])
            model_version = newest['name'].replace('.pkl', '')

        reasoning = llm_interpretation.get('reasoning', '') or (
            f"Health: {health_status}. Drift score: {drift_score}/100. "
            f"{len(alerts)} alerts. "
            f"Retrain recommended: {retrain_result.get('should_retrain', False)}."
        )

        return {
            'check_date': self.target_date,
            'model_version': model_version,
            'health_status': health_status,
            'metrics_snapshot': metrics_snapshot,
            'alerts': alerts,
            'retraining_recommendation': {
                'recommended': retrain_result.get('should_retrain', False),
                'reason': '; '.join(retrain_result.get('reasons', [])) or 'No retraining needed',
                'priority': retrain_result.get('urgency', 'none'),
            },
            'model_staleness': staleness,
            'reasoning': reasoning,
        }

    def report(self, run_output: dict):
        """Send health_check message to all agents."""
        health_status = run_output.get('health_status', 'healthy')

        # Map health status to message priority
        priority_map = {
            'critical': 'urgent',
            'degraded': 'high',
            'healthy': 'low',
        }
        priority = priority_map.get(health_status, 'normal')

        self.send_message(
            recipient='all',
            event_type='health_check',
            payload={
                'check_date': run_output.get('check_date'),
                'health_status': health_status,
                'metrics_snapshot': run_output.get('metrics_snapshot', {}),
                'alerts': run_output.get('alerts', []),
                'retraining_recommendation': run_output.get('retraining_recommendation', {}),
            },
            priority=priority,
        )
