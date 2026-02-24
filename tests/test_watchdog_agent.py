"""
Tests for the Model Watchdog Agent.

Mocks: DriftDetector, call_llm.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.core.message_bus import InMemoryMessageBus
from agents.core.guardrails import Guardrails
from agents.watchdog.watchdog_agent import ModelWatchdogAgent


# =============================================================================
# Helpers
# =============================================================================

def _make_healthy_drift():
    """DriftDetector.check_drift() result — healthy state."""
    return {
        'has_drift': False,
        'drift_score': 5,
        'alerts': [],
        'metrics': {
            'win_rate': 0.54,
            'total_predictions': 120,
        },
        'calibration_error': 0.03,
        'sample_size': 120,
        'lookback_days': 7,
    }


def _make_critical_drift():
    """DriftDetector.check_drift() result — critical state."""
    return {
        'has_drift': True,
        'drift_score': 65,
        'alerts': [
            {
                'type': 'accuracy_drift',
                'severity': 'critical',
                'message': 'Critical accuracy drop: 42% vs baseline 52%',
                'recommendation': 'Immediately investigate and consider model retraining',
                'metric_value': 0.42,
                'threshold_value': 0.42,
            },
            {
                'type': 'calibration_drift',
                'severity': 'high',
                'message': 'Expected Calibration Error: 0.12',
                'recommendation': 'Recalibrate confidence scores',
                'metric_value': 0.12,
                'threshold_value': 0.08,
            },
        ],
        'metrics': {
            'win_rate': 0.42,
            'total_predictions': 80,
        },
        'calibration_error': 0.12,
        'sample_size': 80,
        'lookback_days': 7,
    }


def _make_healthy_retrain():
    return {
        'should_retrain': False,
        'urgency': 'none',
        'reasons': [],
        'drift_score': 5,
    }


def _make_urgent_retrain():
    return {
        'should_retrain': True,
        'urgency': 'immediate',
        'reasons': ['Critical accuracy drop: 42% vs baseline 52%'],
        'drift_score': 65,
    }


VALID_LLM_RESPONSE = json.dumps({
    'health_assessment': 'critical',
    'recommended_actions': [
        {
            'action': 'retrain',
            'priority': 'urgent',
            'rationale': 'Sustained accuracy below 45% for 7 days.',
        }
    ],
    'root_cause_hypothesis': 'Roster changes mid-season altered team dynamics.',
    'reasoning': 'Model performance has degraded significantly. Retraining recommended.',
})


@pytest.fixture
def memory_bus():
    return InMemoryMessageBus()


@pytest.fixture
def sqlite_guardrails(tmp_path):
    return Guardrails(pg_conn=None, sqlite_path=str(tmp_path / "test.db"))


@pytest.fixture
def watchdog_agent(memory_bus, sqlite_guardrails):
    agent = ModelWatchdogAgent(
        target_date='2026-02-24',
        lookback_days=7,
        message_bus=memory_bus,
        guardrails=sqlite_guardrails,
        shadow_mode=False,
    )
    return agent


# =============================================================================
# Tests
# =============================================================================

class TestModelWatchdogAgent:

    def test_run_calls_drift_detector(self, watchdog_agent):
        """DriftDetector.check_drift() is called."""
        mock_detector = MagicMock()
        mock_detector.check_drift.return_value = _make_healthy_drift()
        mock_detector.should_retrain.return_value = _make_healthy_retrain()
        watchdog_agent._drift_detector = mock_detector

        result = watchdog_agent.run()

        mock_detector.check_drift.assert_called_once_with(lookback_days=7)
        mock_detector.should_retrain.assert_called_once()
        assert result['check_date'] == '2026-02-24'

    def test_health_status_healthy(self, watchdog_agent):
        """Low drift score -> healthy (or degraded if models are stale on disk)."""
        mock_detector = MagicMock()
        mock_detector.check_drift.return_value = _make_healthy_drift()
        mock_detector.should_retrain.return_value = _make_healthy_retrain()
        watchdog_agent._drift_detector = mock_detector

        # Mock staleness check to avoid depending on real model files
        with patch.object(watchdog_agent, '_check_model_staleness', return_value={
            'models_found': 3, 'oldest_age_days': 5.0, 'newest_age_days': 2.0,
            'is_stale': False, 'model_files': [{'name': 'ensemble.pkl', 'age_days': 2.0}],
        }):
            result = watchdog_agent.run()

        assert result['health_status'] == 'healthy'
        assert result['retraining_recommendation']['recommended'] is False

    def test_health_status_critical(self, watchdog_agent):
        """High drift score -> critical."""
        mock_detector = MagicMock()
        mock_detector.check_drift.return_value = _make_critical_drift()
        mock_detector.should_retrain.return_value = _make_urgent_retrain()
        watchdog_agent._drift_detector = mock_detector

        with patch.object(watchdog_agent, 'call_llm', return_value=VALID_LLM_RESPONSE):
            result = watchdog_agent.run()

        assert result['health_status'] == 'critical'
        assert result['retraining_recommendation']['recommended'] is True

    def test_retraining_recommendation(self, watchdog_agent):
        """should_retrain() result included in output."""
        mock_detector = MagicMock()
        mock_detector.check_drift.return_value = _make_healthy_drift()
        mock_detector.should_retrain.return_value = _make_urgent_retrain()
        watchdog_agent._drift_detector = mock_detector

        result = watchdog_agent.run()

        assert result['retraining_recommendation']['recommended'] is True
        assert result['retraining_recommendation']['priority'] == 'immediate'

    def test_llm_interprets_drift(self, watchdog_agent):
        """When critical alerts exist, call_llm is invoked."""
        mock_detector = MagicMock()
        mock_detector.check_drift.return_value = _make_critical_drift()
        mock_detector.should_retrain.return_value = _make_urgent_retrain()
        watchdog_agent._drift_detector = mock_detector

        call_count = []

        def capture_call(system_prompt, user_message, **kwargs):
            call_count.append(1)
            return VALID_LLM_RESPONSE

        with patch.object(watchdog_agent, 'call_llm', side_effect=capture_call):
            result = watchdog_agent.run()

        assert len(call_count) == 1
        assert 'degraded' in result['reasoning'] or 'critical' in result['reasoning'].lower() or 'Retraining' in result['reasoning']

    def test_report_sends_health_check(self, watchdog_agent, memory_bus):
        """health_check broadcast to all."""
        run_output = {
            'check_date': '2026-02-24',
            'health_status': 'healthy',
            'metrics_snapshot': {},
            'alerts': [],
            'retraining_recommendation': {'recommended': False},
        }

        watchdog_agent.report(run_output)

        all_msgs = memory_bus.receive('all', event_type='health_check')
        assert len(all_msgs) >= 1

    def test_report_priority_escalation(self, watchdog_agent, memory_bus):
        """critical -> urgent priority on messages."""
        run_output = {
            'check_date': '2026-02-24',
            'health_status': 'critical',
            'metrics_snapshot': {},
            'alerts': [{'severity': 'critical', 'message': 'test'}],
            'retraining_recommendation': {'recommended': True},
        }

        watchdog_agent.report(run_output)

        all_msgs = memory_bus.receive('all', event_type='health_check')
        assert len(all_msgs) >= 1
        assert all_msgs[0].priority == 'urgent'

    def test_handles_no_data(self, watchdog_agent):
        """No performance data -> graceful result (no crash)."""
        mock_detector = MagicMock()
        mock_detector.check_drift.side_effect = Exception("No data available")
        mock_detector.should_retrain.side_effect = Exception("No data available")
        watchdog_agent._drift_detector = mock_detector

        # Mock staleness to isolate from real model files
        with patch.object(watchdog_agent, '_check_model_staleness', return_value={
            'models_found': 0, 'oldest_age_days': 0, 'newest_age_days': 0,
            'is_stale': True, 'model_files': [],
        }):
            result = watchdog_agent.run()

        # With stale models but no drift data, status should be degraded (stale models)
        assert result['health_status'] in ('healthy', 'degraded')
        assert result['retraining_recommendation']['recommended'] is False
        # Verify it didn't crash despite both calls raising exceptions
        assert 'check_date' in result
