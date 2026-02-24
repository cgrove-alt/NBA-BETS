"""
Tests for the Post-Game Analysis Agent.

Mocks: CalibrationService, BiasAnalyzer, call_llm.
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
from agents.postgame.postgame_agent import PostGameAnalysisAgent, PROP_STD_DEVS


# =============================================================================
# Fixtures
# =============================================================================

def _make_mock_predictions():
    """Create mock prediction+outcome records."""
    return [
        {
            'id': 1, 'player_name': 'LeBron James', 'prop_type': 'points',
            'predicted_value': 28.0, 'actual_value': 27.0, 'hit': 1,
            'confidence': 0.65, 'clv': 0.5, 'prop_line': 26.5,
            'minutes_predicted': 35, 'actual_minutes': 36,
            'is_home': True, 'opponent': 'BOS', 'spread': -3.5,
        },
        {
            'id': 2, 'player_name': 'Stephen Curry', 'prop_type': 'points',
            'predicted_value': 30.0, 'actual_value': 15.0, 'hit': 0,
            'confidence': 0.70, 'clv': -0.3, 'prop_line': 28.5,
            'minutes_predicted': 34, 'actual_minutes': 22,
            'is_home': False, 'opponent': 'LAL', 'spread': 3.5,
        },
        {
            'id': 3, 'player_name': 'Nikola Jokic', 'prop_type': 'rebounds',
            'predicted_value': 12.0, 'actual_value': 5.0, 'hit': 0,
            'confidence': 0.60, 'clv': 0.1, 'prop_line': 11.5,
            'minutes_predicted': 33, 'actual_minutes': 30,
            'is_home': True, 'opponent': 'MIA', 'spread': -5.0,
        },
        {
            'id': 4, 'player_name': 'Jayson Tatum', 'prop_type': 'assists',
            'predicted_value': 5.5, 'actual_value': 6.0, 'hit': 1,
            'confidence': 0.55, 'clv': 0.2, 'prop_line': 5.5,
            'minutes_predicted': 36, 'actual_minutes': 37,
            'is_home': False, 'opponent': 'CLE', 'spread': 1.5,
        },
    ]


VALID_MISS_ANALYSIS = json.dumps({
    'root_cause': 'data_issue',
    'explanation': 'Player was in foul trouble and played only 22 minutes vs predicted 34.',
    'recommended_action': 'Improve foul trouble detection in minutes prediction model.',
})


@pytest.fixture
def memory_bus():
    return InMemoryMessageBus()


@pytest.fixture
def sqlite_guardrails(tmp_path):
    return Guardrails(pg_conn=None, sqlite_path=str(tmp_path / "test.db"))


@pytest.fixture
def postgame_agent(memory_bus, sqlite_guardrails):
    """Create a PostGameAnalysisAgent with mocked dependencies."""
    agent = PostGameAnalysisAgent(
        target_date='2026-02-23',
        message_bus=memory_bus,
        guardrails=sqlite_guardrails,
        shadow_mode=False,
    )
    return agent


def _make_mock_service(predictions=None):
    """Create a mock CalibrationService."""
    mock = MagicMock()
    mock.run_nightly_job.return_value = {
        'steps': {
            'outcomes': {'matched': 4, 'not_found': 0, 'dnp': 0, 'errors': 0},
            'adjustments_generated': 3,
            'report_saved': True,
        },
    }
    mock.db.get_predictions_with_outcomes.return_value = predictions if predictions is not None else _make_mock_predictions()

    mock_report = MagicMock()
    mock_report.to_dict.return_value = {
        'by_prop_type': {},
        'by_position': {},
        'by_game_type': {},
        'by_player_tier': {},
    }
    mock.analyze_biases.return_value = mock_report

    return mock


# =============================================================================
# Tests
# =============================================================================

class TestPostGameAgent:

    def test_run_calls_nightly_job(self, postgame_agent):
        """CalibrationService.run_nightly_job is called."""
        mock_service = _make_mock_service()
        postgame_agent._calibration_service = mock_service

        with patch.object(postgame_agent, 'call_llm', return_value=VALID_MISS_ANALYSIS):
            result = postgame_agent.run()

        mock_service.run_nightly_job.assert_called_once_with(game_date='2026-02-23')

    def test_identifies_large_misses(self, postgame_agent):
        """Misses > 2 std devs identified."""
        predictions = _make_mock_predictions()
        large_misses = postgame_agent._identify_large_misses(predictions)

        # Curry: predicted 30, actual 15, diff=15, std=5.5, threshold=11. 15 > 11 -> large miss
        # Jokic: predicted 12, actual 5, diff=7, std=3.5, threshold=7. 7 >= 7 -> borderline
        assert len(large_misses) >= 1
        assert large_misses[0]['player_name'] == 'Stephen Curry'

    def test_caps_claude_at_10(self, postgame_agent):
        """Max 10 miss analyses per run."""
        # Create 15 large misses
        many_predictions = []
        for i in range(15):
            many_predictions.append({
                'id': i, 'player_name': f'Player {i}', 'prop_type': 'points',
                'predicted_value': 30.0, 'actual_value': 5.0,  # Huge miss
                'hit': 0, 'confidence': 0.6, 'clv': -0.5,
                'minutes_predicted': 34, 'actual_minutes': 20,
            })

        mock_service = _make_mock_service(predictions=many_predictions)
        postgame_agent._calibration_service = mock_service

        call_count = 0
        def count_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return VALID_MISS_ANALYSIS

        with patch.object(postgame_agent, 'call_llm', side_effect=count_calls):
            result = postgame_agent.run()

        assert call_count <= 10
        assert len(result['miss_analysis']) <= 10

    def test_parse_miss_analysis(self, postgame_agent):
        """Valid miss analysis JSON parsed."""
        miss = {
            'player_name': 'Test Player', 'prop_type': 'points',
            'predicted_value': 30, 'actual_value': 10,
            'minutes_predicted': 34, 'actual_minutes': 20,
            'std_devs_off': 3.6, 'confidence': 0.65,
        }

        with patch.object(postgame_agent, 'call_llm', return_value=VALID_MISS_ANALYSIS):
            result = postgame_agent._analyze_miss_with_llm(miss)

        assert result['root_cause'] == 'data_issue'
        assert 'foul trouble' in result['explanation']

    def test_pattern_min_sample_30(self, postgame_agent):
        """Patterns require 30+ samples."""
        bias_dict = {
            'by_prop_type': {
                'points': {'sample_size': 10, 'bias': 5.0, 'hit_rate': 0.4},  # Too few
                'rebounds': {'sample_size': 50, 'bias': 3.5, 'hit_rate': 0.45},  # Enough
            },
            'by_position': {},
            'by_game_type': {},
            'by_player_tier': {},
        }

        patterns = postgame_agent._extract_pattern_flags(bias_dict)

        # Should only flag rebounds (50 samples) not points (10 samples)
        assert len(patterns) >= 1
        flagged_values = [p['value'] for p in patterns]
        assert 'rebounds' in flagged_values
        assert 'points' not in flagged_values

    def test_report_sends_to_watchdog(self, postgame_agent, memory_bus):
        """results_analyzed sent to watchdog."""
        run_output = {
            'slate_date': '2026-02-23',
            'results_summary': {'total_bets': 4, 'wins': 2, 'losses': 2},
            'miss_analysis': [],
            'model_feedback': [],
            'pattern_flags': [],
            'reasoning': 'test',
        }

        postgame_agent.report(run_output)

        watchdog_msgs = memory_bus.receive('watchdog', event_type='results_analyzed')
        assert len(watchdog_msgs) >= 1

    def test_report_sends_to_briefing(self, postgame_agent, memory_bus):
        """results_analyzed sent to briefing."""
        run_output = {
            'slate_date': '2026-02-23',
            'results_summary': {'total_bets': 4, 'wins': 2, 'losses': 2},
            'miss_analysis': [],
            'model_feedback': [],
            'pattern_flags': [],
            'reasoning': 'test',
        }

        postgame_agent.report(run_output)

        briefing_msgs = memory_bus.receive('briefing', event_type='results_analyzed')
        assert len(briefing_msgs) >= 1

    def test_handles_no_predictions(self, memory_bus, sqlite_guardrails):
        """No predictions -> empty result, no crash."""
        agent = PostGameAnalysisAgent(
            target_date='2026-02-23',
            message_bus=memory_bus,
            guardrails=sqlite_guardrails,
        )
        mock_service = _make_mock_service(predictions=[])
        agent._calibration_service = mock_service

        result = agent.run()

        assert result['results_summary']['total_bets'] == 0
        assert result['miss_analysis'] == []
        assert result['pattern_flags'] == []
