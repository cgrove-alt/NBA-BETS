"""
Tests for the Daily Briefing Agent.

Mocks: call_llm, message bus messages.
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
from agents.briefing.briefing_agent import DailyBriefingAgent


# =============================================================================
# Helpers
# =============================================================================

def _send_predictions_message(bus):
    """Send a predictions_published message to the bus."""
    msg = Message.create(
        sender='predictor',
        recipient='briefing',
        event_type='predictions_published',
        payload={
            'slate_date': '2026-02-24',
            'predictions': [{
                'game_id': 'game_001',
                'home_team': 'LAL',
                'away_team': 'BOS',
                'spread': {
                    'line': '-3.5',
                    'edge': 4.8,
                    'signal': 'BET',
                    'confidence': 'HIGH',
                    'recommended_units': 1.5,
                },
                'player_props': [{
                    'player_name': 'LeBron James',
                    'stat_type': 'points',
                    'prop_line': 24.5,
                    'pick': 'Over',
                    'edge': 6.1,
                    'signal': 'BET',
                    'bet_recommendation': 'BET',
                    'confidence': 'MEDIUM',
                    'recommended_units': 1.0,
                }],
            }],
            'total_exposure': {'total_units': 2.5, 'total_pct': 2.5},
            'correlation_warnings': [],
            'bet_count': 2,
            'lean_count': 0,
        },
    )
    bus.send(msg)


def _send_results_message(bus):
    """Send a results_analyzed message to the bus."""
    msg = Message.create(
        sender='postgame',
        recipient='briefing',
        event_type='results_analyzed',
        payload={
            'slate_date': '2026-02-23',
            'results_summary': {
                'total_bets': 8,
                'wins': 5,
                'losses': 3,
                'roi_today': '+3.2%',
                'clv_average': '+0.8',
            },
            'pattern_flags': [],
        },
    )
    bus.send(msg)


def _send_odds_message(bus):
    """Send an odds_alert message to the bus."""
    msg = Message.create(
        sender='odds_monitor',
        recipient='briefing',
        event_type='odds_alert',
        payload={
            'date': '2026-02-24',
            'steam_count': 1,
            'stale_count': 0,
            'notable_movements': [{'reasoning': 'Sharp money on LAL spread'}],
            'reasoning': 'One steam move detected on LAL game.',
        },
    )
    bus.send(msg)


def _send_health_message(bus):
    """Send a health_check message to the bus."""
    msg = Message.create(
        sender='watchdog',
        recipient='all',
        event_type='health_check',
        payload={
            'check_date': '2026-02-24',
            'health_status': 'healthy',
            'metrics_snapshot': {'drift_score': 5},
            'alerts': [],
            'retraining_recommendation': {'recommended': False},
        },
    )
    bus.send(msg)


VALID_LLM_RESPONSE = json.dumps({
    'sections': {
        'yesterday_recap': {
            'record': '5-3',
            'roi': '+3.2%',
            'pnl': '+$160',
            'clv_summary': 'Beat closing line by 0.8 points on average',
            'notable': 'Strong day with 62.5% win rate.',
        },
        'today_plays': [{
            'pick': 'LAL -3.5',
            'units': 1.5,
            'edge': '4.8%',
            'confidence': 'HIGH',
            'signal': 'BET',
            'reasoning': 'Model sees rest advantage plus sharp money aligned.',
        }],
        'bankroll': {
            'current': 'See dashboard',
            'today_exposure': '2.5u (2.5%)',
            'season_pnl': 'See dashboard',
        },
        'alerts': ['All systems healthy'],
        'market_intel': ['Sharp money on LAL spread'],
    },
    'formatted_text': 'NBA MODEL DAILY BRIEFING...',
    'reasoning': 'Briefing generated with full agent context.',
})


@pytest.fixture
def memory_bus():
    return InMemoryMessageBus()


@pytest.fixture
def sqlite_guardrails(tmp_path):
    return Guardrails(pg_conn=None, sqlite_path=str(tmp_path / "test.db"))


@pytest.fixture
def briefing_agent(memory_bus, sqlite_guardrails):
    agent = DailyBriefingAgent(
        target_date='2026-02-24',
        message_bus=memory_bus,
        guardrails=sqlite_guardrails,
        shadow_mode=False,
    )
    return agent


# =============================================================================
# Tests
# =============================================================================

class TestDailyBriefingAgent:

    def test_run_reads_all_agent_messages(self, briefing_agent, memory_bus):
        """Reads from predictor, postgame, odds_monitor, watchdog."""
        _send_predictions_message(memory_bus)
        _send_results_message(memory_bus)
        _send_odds_message(memory_bus)
        _send_health_message(memory_bus)

        with patch.object(briefing_agent, 'call_llm', return_value=VALID_LLM_RESPONSE):
            result = briefing_agent.run()

        assert 'predictions' in result['data_sources']
        assert 'yesterday_results' in result['data_sources']
        assert 'odds_intel' in result['data_sources']
        assert 'health_check' in result['data_sources']

    def test_output_has_all_sections(self, briefing_agent, memory_bus):
        """Contains yesterday_recap, today_plays, bankroll, alerts, market_intel."""
        _send_predictions_message(memory_bus)
        _send_results_message(memory_bus)

        with patch.object(briefing_agent, 'call_llm', return_value=VALID_LLM_RESPONSE):
            result = briefing_agent.run()

        sections = result['sections']
        assert 'yesterday_recap' in sections
        assert 'today_plays' in sections
        assert 'bankroll' in sections
        assert 'alerts' in sections
        assert 'market_intel' in sections

    def test_llm_generates_formatted_briefing(self, briefing_agent, memory_bus):
        """call_llm produces formatted text."""
        _send_predictions_message(memory_bus)

        call_count = []

        def capture_call(system_prompt, user_message, **kwargs):
            call_count.append(1)
            return VALID_LLM_RESPONSE

        with patch.object(briefing_agent, 'call_llm', side_effect=capture_call):
            result = briefing_agent.run()

        assert len(call_count) == 1
        assert result['formatted_text'] != ''

    def test_fallback_without_llm(self, briefing_agent, memory_bus):
        """Template-based briefing when LLM unavailable."""
        _send_predictions_message(memory_bus)
        _send_results_message(memory_bus)

        with patch.object(briefing_agent, 'call_llm', return_value=''):
            result = briefing_agent.run()

        assert 'formatted_text' in result
        assert 'DAILY BRIEFING' in result['formatted_text']
        assert 'deterministic' in result['reasoning'].lower() or 'LLM unavailable' in result['reasoning']

    def test_handles_missing_agent_data(self, briefing_agent, memory_bus):
        """If some agents haven't run, still generates briefing."""
        # Only send predictions, no results or health
        _send_predictions_message(memory_bus)

        with patch.object(briefing_agent, 'call_llm', return_value=''):
            result = briefing_agent.run()

        assert 'formatted_text' in result
        sections = result['sections']
        assert sections['yesterday_recap']['record'] == 'N/A'
        assert 'predictions' in result['data_sources']

    def test_report_sends_briefing_ready(self, briefing_agent, memory_bus):
        """briefing_ready broadcast."""
        run_output = {
            'briefing_date': '2026-02-24',
            'formatted_text': 'Test briefing',
            'sections': {},
        }

        briefing_agent.report(run_output)

        all_msgs = memory_bus.receive('all', event_type='briefing_ready')
        assert len(all_msgs) >= 1
        assert all_msgs[0].payload['formatted_text'] == 'Test briefing'

    def test_shadow_mode_no_messages(self, memory_bus, sqlite_guardrails):
        """Shadow mode works."""
        agent = DailyBriefingAgent(
            target_date='2026-02-24',
            message_bus=memory_bus,
            guardrails=sqlite_guardrails,
            shadow_mode=True,
        )

        with patch.object(agent, 'call_llm', return_value=VALID_LLM_RESPONSE):
            result = agent.execute()

        assert result.messages_sent == 0

    def test_handles_empty_bus(self, briefing_agent):
        """No messages on bus -> minimal briefing, no crash."""
        with patch.object(briefing_agent, 'call_llm', return_value=''):
            result = briefing_agent.run()

        assert 'formatted_text' in result
        assert result['data_sources'] == []
        sections = result['sections']
        assert sections['yesterday_recap']['record'] == 'N/A'
        assert sections['today_plays'] == []
