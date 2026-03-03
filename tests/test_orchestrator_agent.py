"""
Tests for the Prediction Orchestrator Agent.

Mocks: daily_predictions (load_models, analyze_game), BalldontlieAPI, call_llm.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.core.message_bus import InMemoryMessageBus, Message
from agents.core.guardrails import Guardrails
from agents.orchestrator.orchestrator_agent import PredictionOrchestratorAgent


# =============================================================================
# Helpers
# =============================================================================

def _make_mock_prediction(game_id='game_001', home='LAL', away='BOS'):
    """Create a mock analyze_game() result."""
    return {
        'game_id': game_id,
        'home_team': home,
        'away_team': away,
        'game_time': '7:30 PM',
        'spread': {
            'line': '-3.5',
            'predicted_spread': -5.2,
            'edge': 4.8,
            'spread_edge_pct': 4.8,
            'signal': 'BET',
            'confidence': 'high',
        },
        'player_props': [
            {
                'player_name': 'LeBron James',
                'stat_type': 'points',
                'prop_line': 24.5,
                'predicted_value': 27.1,
                'pick': 'Over',
                'edge': 6.1,
                'edge_pct': 6.1,
                'signal': 'BET',
                'bet_recommendation': 'BET',
                'confidence': 'medium',
                'team': home,
            },
            {
                'player_name': 'Anthony Davis',
                'stat_type': 'rebounds',
                'prop_line': 10.5,
                'predicted_value': 11.8,
                'pick': 'Over',
                'edge': 5.5,
                'edge_pct': 5.5,
                'signal': 'BET',
                'bet_recommendation': 'BET',
                'confidence': 'medium',
                'team': home,
            },
            {
                'player_name': 'Austin Reaves',
                'stat_type': 'assists',
                'prop_line': 5.5,
                'predicted_value': 6.3,
                'pick': 'Over',
                'edge': 4.0,
                'edge_pct': 4.0,
                'signal': 'BET',
                'bet_recommendation': 'BET',
                'confidence': 'low',
                'team': home,
            },
        ],
    }


def _make_intel_message(bus, game_id='game_001', confidence='high'):
    """Send an intel_ready message to the bus."""
    msg = Message.create(
        sender='pregame',
        recipient='orchestrator',
        event_type='intel_ready',
        payload={
            'game_id': game_id,
            'overall_game_confidence': confidence,
            'injury_impact': {},
        },
    )
    bus.send(msg)


@pytest.fixture
def memory_bus():
    return InMemoryMessageBus()


@pytest.fixture
def sqlite_guardrails(tmp_path):
    return Guardrails(pg_conn=None, sqlite_path=str(tmp_path / "test.db"))


@pytest.fixture
def orchestrator_agent(memory_bus, sqlite_guardrails):
    return PredictionOrchestratorAgent(
        target_date='2026-02-24',
        message_bus=memory_bus,
        guardrails=sqlite_guardrails,
        shadow_mode=False,
    )


# =============================================================================
# Tests
# =============================================================================

class TestPredictionOrchestratorAgent:

    def test_run_produces_predictions(self, orchestrator_agent):
        """Output has predictions list."""
        mock_prediction = _make_mock_prediction()

        with patch.object(orchestrator_agent, '_run_predictions', return_value=[mock_prediction]):
            result = orchestrator_agent.run()

        assert 'predictions' in result
        assert len(result['predictions']) == 1
        assert result['slate_date'] == '2026-02-24'

    def test_reads_intel_ready_messages(self, orchestrator_agent, memory_bus):
        """Consumes intel_ready from pregame."""
        _make_intel_message(memory_bus, game_id='game_001', confidence='high')
        mock_prediction = _make_mock_prediction()

        with patch.object(orchestrator_agent, '_run_predictions', return_value=[mock_prediction]):
            result = orchestrator_agent.run()

        assert result['intel_context']['intel_messages'] >= 1

    def test_confidence_downgrade_low_intel(self, orchestrator_agent, memory_bus):
        """Low lineup confidence -> confidence downgrade."""
        _make_intel_message(memory_bus, game_id='game_001', confidence='low')
        mock_prediction = _make_mock_prediction()

        with patch.object(orchestrator_agent, '_run_predictions', return_value=[mock_prediction]):
            result = orchestrator_agent.run()

        # Check that at least one prop was downgraded
        props = result['predictions'][0]['player_props']
        downgraded = [p for p in props if p.get('intel_adjustment', '') == 'downgraded (low lineup confidence)']
        assert len(downgraded) > 0

    def test_correlation_check_multi_props(self, orchestrator_agent):
        """3+ BET signals on same team triggers correlation warning."""
        mock_prediction = _make_mock_prediction()
        # Already has 3 BET props on LAL

        with patch.object(orchestrator_agent, '_run_predictions', return_value=[mock_prediction]):
            result = orchestrator_agent.run()

        assert len(result['correlation_warnings']) >= 1
        assert result['correlation_warnings'][0]['team'] == 'LAL'

    def test_bankroll_cap_enforced(self, orchestrator_agent):
        """Total exposure never exceeds 10% cap."""
        # Create predictions with very high edges to push exposure
        predictions = []
        for i in range(5):
            pred = _make_mock_prediction(game_id=f'game_{i:03d}')
            for prop in pred['player_props']:
                prop['edge'] = 20.0  # Very high edge → high unit sizing
                prop['edge_pct'] = 20.0
            predictions.append(pred)

        with patch.object(orchestrator_agent, '_run_predictions', return_value=predictions):
            result = orchestrator_agent.run()

        assert result['total_exposure']['total_units'] <= 10.0

    def test_report_sends_predictions_published(self, orchestrator_agent, memory_bus):
        """predictions_published to briefing and all."""
        run_output = {
            'slate_date': '2026-02-24',
            'predictions': [_make_mock_prediction()],
            'correlation_warnings': [],
            'total_exposure': {'total_units': 3.0, 'total_pct': 3.0},
        }

        orchestrator_agent.report(run_output)

        briefing_msgs = memory_bus.receive('briefing', event_type='predictions_published')
        assert len(briefing_msgs) >= 1

        all_msgs = memory_bus.receive('all', event_type='predictions_published')
        assert len(all_msgs) >= 1

    def test_shadow_mode_no_messages(self, memory_bus, sqlite_guardrails):
        """Shadow mode works."""
        agent = PredictionOrchestratorAgent(
            target_date='2026-02-24',
            message_bus=memory_bus,
            guardrails=sqlite_guardrails,
            shadow_mode=True,
        )

        with patch.object(agent, '_run_predictions', return_value=[_make_mock_prediction()]):
            result = agent.execute()

        assert result.messages_sent == 0

    def test_handles_no_games(self, orchestrator_agent):
        """No games -> empty predictions, no crash."""
        with patch.object(orchestrator_agent, '_run_predictions', return_value=[]):
            result = orchestrator_agent.run()

        assert result['predictions'] == []
        assert result['total_exposure']['total_units'] == 0
