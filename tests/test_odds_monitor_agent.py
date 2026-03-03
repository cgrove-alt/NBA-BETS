"""
Tests for the Odds Monitoring Agent.

Mocks: MarketMonitor, call_llm.
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
from agents.odds_monitor.odds_monitor_agent import OddsMonitorAgent


# =============================================================================
# Helpers
# =============================================================================

def _make_steam_alert():
    """Create a mock SteamAlert-like object."""
    alert = MagicMock()
    alert.game_id = 'game_001'
    alert.market = 'spread'
    alert.side = 'home'
    alert.direction = 'down'
    alert.confidence = 0.8
    alert.leader_book = 'pinnacle'
    alert.leader_move = 0.04
    alert.leader_current_prob = 0.55
    alert.laggard_books = [{'book': 'draftkings', 'odds': -105, 'edge': 0.035}]
    return alert


def _make_stale_line():
    """Create a mock StaleLine-like object."""
    stale = MagicMock()
    stale.game_id = 'game_002'
    stale.book = 'caesars'
    stale.market = 'moneyline'
    stale.side = 'away'
    stale.edge = 0.03
    stale.book_odds = 150
    stale.book_implied_prob = 0.40
    stale.consensus_prob = 0.43
    return stale


VALID_LLM_RESPONSE = json.dumps({
    'notable_movements': [
        {
            'event_type': 'steam_move',
            'game_id': 'game_001',
            'market': 'spread',
            'recommendation': 're-evaluate',
            'sharp_money_assessment': 'confirmed_sharp',
            'confidence': 0.8,
            'reasoning': 'Pinnacle moved first, DraftKings still lagging.',
        }
    ],
    'overall_market_assessment': 'Sharp action on home side.',
    'reasoning': 'Clear steam move pattern with 4% probability shift at sharp books.',
})


@pytest.fixture
def memory_bus():
    return InMemoryMessageBus()


@pytest.fixture
def sqlite_guardrails(tmp_path):
    return Guardrails(pg_conn=None, sqlite_path=str(tmp_path / "test.db"))


@pytest.fixture
def odds_agent(memory_bus, sqlite_guardrails):
    return OddsMonitorAgent(
        target_date='2026-02-24',
        message_bus=memory_bus,
        guardrails=sqlite_guardrails,
        shadow_mode=False,
    )


# =============================================================================
# Tests
# =============================================================================

class TestOddsMonitorAgent:

    def test_run_calls_market_monitor(self, odds_agent):
        """MarketMonitor.check_once() is called."""
        mock_monitor = MagicMock()
        mock_monitor.check_once.return_value = {'steam': [], 'stale': []}
        odds_agent._market_monitor = mock_monitor

        result = odds_agent.run()

        mock_monitor.check_once.assert_called_once()
        assert result['target_date'] == '2026-02-24'

    def test_output_contains_steam_and_stale(self, odds_agent):
        """Output has steam_alerts and stale_lines keys."""
        mock_monitor = MagicMock()
        steam = _make_steam_alert()
        stale = _make_stale_line()
        mock_monitor.check_once.return_value = {'steam': [steam], 'stale': [stale]}
        odds_agent._market_monitor = mock_monitor

        with patch.object(odds_agent, 'call_llm', return_value=VALID_LLM_RESPONSE):
            result = odds_agent.run()

        assert 'steam_alerts' in result
        assert 'stale_lines' in result
        assert len(result['steam_alerts']) == 1
        assert len(result['stale_lines']) == 1

    def test_llm_interprets_steam_moves(self, odds_agent):
        """When steam alerts exist, call_llm is invoked."""
        mock_monitor = MagicMock()
        mock_monitor.check_once.return_value = {
            'steam': [_make_steam_alert()],
            'stale': [],
        }
        odds_agent._market_monitor = mock_monitor

        call_args = []

        def capture_call(system_prompt, user_message, **kwargs):
            call_args.append(user_message)
            return VALID_LLM_RESPONSE

        with patch.object(odds_agent, 'call_llm', side_effect=capture_call):
            result = odds_agent.run()

        assert len(call_args) == 1
        assert 'game_001' in call_args[0]
        assert len(result['notable_movements']) > 0

    def test_fallback_without_llm(self, odds_agent):
        """Without LLM, deterministic alerts still returned."""
        mock_monitor = MagicMock()
        mock_monitor.check_once.return_value = {
            'steam': [_make_steam_alert()],
            'stale': [_make_stale_line()],
        }
        odds_agent._market_monitor = mock_monitor

        with patch.object(odds_agent, 'call_llm', return_value=''):
            result = odds_agent.run()

        assert 'steam_alerts' in result
        assert 'notable_movements' in result
        assert 'Deterministic fallback' in result['reasoning']

    def test_report_sends_odds_alert_to_orchestrator(self, odds_agent, memory_bus):
        """odds_alert messages sent to orchestrator for notable steam moves."""
        run_output = {
            'target_date': '2026-02-24',
            'steam_alerts': [{
                'game_id': 'game_001',
                'market': 'spread',
                'side': 'home',
                'laggard_books': [{'book': 'draftkings', 'edge': 0.035}],
            }],
            'stale_lines': [],
            'notable_movements': [],
            'reasoning': 'test',
        }

        odds_agent.report(run_output)

        orchestrator_msgs = memory_bus.receive('orchestrator', event_type='odds_alert')
        assert len(orchestrator_msgs) >= 1
        assert orchestrator_msgs[0].priority == 'high'

    def test_report_sends_to_briefing(self, odds_agent, memory_bus):
        """odds_alert sent to briefing."""
        run_output = {
            'target_date': '2026-02-24',
            'steam_alerts': [],
            'stale_lines': [],
            'notable_movements': [],
            'reasoning': 'test',
        }

        odds_agent.report(run_output)

        briefing_msgs = memory_bus.receive('briefing', event_type='odds_alert')
        assert len(briefing_msgs) >= 1

    def test_shadow_mode_no_messages(self, memory_bus, sqlite_guardrails):
        """Shadow mode suppresses all messages."""
        agent = OddsMonitorAgent(
            target_date='2026-02-24',
            message_bus=memory_bus,
            guardrails=sqlite_guardrails,
            shadow_mode=True,
        )
        mock_monitor = MagicMock()
        mock_monitor.check_once.return_value = {
            'steam': [_make_steam_alert()],
            'stale': [],
        }
        agent._market_monitor = mock_monitor

        with patch.object(agent, 'call_llm', return_value=VALID_LLM_RESPONSE):
            result = agent.execute()

        assert result.messages_sent == 0

    def test_handles_no_market_data(self, odds_agent):
        """Empty odds -> empty result, no crash."""
        mock_monitor = MagicMock()
        mock_monitor.check_once.return_value = {'steam': [], 'stale': []}
        odds_agent._market_monitor = mock_monitor

        result = odds_agent.run()

        assert result['steam_alerts'] == []
        assert result['stale_lines'] == []
        assert result['notable_movements'] == []
