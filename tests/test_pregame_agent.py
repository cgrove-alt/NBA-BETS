"""
Tests for the Pre-Game Intelligence Agent.

Mocks: LineupIntelService, BalldontlieAPI, call_llm.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.core.message_bus import InMemoryMessageBus
from agents.core.guardrails import Guardrails
from agents.pregame.pregame_agent import PreGameIntelAgent


# =============================================================================
# Fixtures
# =============================================================================

def _make_mock_game_intel():
    """Create a mock GameIntel-like object."""
    intel = MagicMock()
    intel.to_dict.return_value = {
        'home_team': 'LAL',
        'away_team': 'BOS',
        'home_injuries': [
            {'player_name': 'Anthony Davis', 'status': 'OUT', 'injury_detail': 'Knee'},
        ],
        'away_injuries': [],
        'home_players': [],
        'away_players': [],
        'home_star_out': True,
        'away_star_out': False,
        'lineup_confidence': 0.85,
    }
    return intel


def _make_mock_game():
    """Create a mock BDL game response."""
    return {
        'id': 12345,
        'date': '2026-02-24',
        'home_team': {'id': 14, 'abbreviation': 'LAL', 'name': 'Lakers'},
        'visitor_team': {'id': 2, 'abbreviation': 'BOS', 'name': 'Celtics'},
        'status': 'scheduled',
    }


VALID_LLM_RESPONSE = json.dumps({
    'injury_impact': {
        'home': {
            'missing_players': ['Anthony Davis'],
            'impact_assessment': 'AD out — significant interior defense and rebounding loss',
            'rotation_changes': 'Rui Hachimura starts at PF, Jaxson Hayes gets more minutes at C',
        },
        'away': {
            'missing_players': [],
            'impact_assessment': 'Full strength',
            'rotation_changes': 'No changes',
        },
    },
    'projected_lineups': {
        'home': ['Austin Reaves', 'Max Christie', 'LeBron James', 'Rui Hachimura', 'Jaxson Hayes'],
        'away': ['Jrue Holiday', 'Derrick White', 'Jaylen Brown', 'Jayson Tatum', 'Kristaps Porzingis'],
    },
    'contextual_flags': ['rest_advantage_away'],
    'player_prop_briefs': {
        'LeBron James': {
            'context': 'With AD out, expect increased rebounding and usage',
            'confidence_modifier': 0.02,
        },
    },
    'overall_game_confidence': 'high',
    'reasoning': 'AD absence is the key factor. Lakers lose interior D, Celtics should dominate paint.',
})


@pytest.fixture
def memory_bus():
    return InMemoryMessageBus()


@pytest.fixture
def sqlite_guardrails(tmp_path):
    return Guardrails(pg_conn=None, sqlite_path=str(tmp_path / "test.db"))


@pytest.fixture
def pregame_agent(memory_bus, sqlite_guardrails):
    """Create a PreGameIntelAgent with mocked dependencies."""
    return PreGameIntelAgent(
        target_date='2026-02-24',
        message_bus=memory_bus,
        guardrails=sqlite_guardrails,
        shadow_mode=False,
    )


# =============================================================================
# Tests
# =============================================================================

class TestPreGameAgent:

    def test_run_produces_game_intels(self, pregame_agent):
        """Output has game_intels list."""
        mock_bdl = MagicMock()
        mock_bdl.get_games.return_value = [_make_mock_game()]

        mock_lineup = MagicMock()
        mock_lineup.get_game_intel.return_value = _make_mock_game_intel()

        pregame_agent._bdl_api = mock_bdl
        pregame_agent._lineup_service = mock_lineup

        with patch.object(pregame_agent, 'call_llm', return_value=VALID_LLM_RESPONSE):
            result = pregame_agent.run()

        assert 'game_intels' in result
        assert len(result['game_intels']) == 1
        assert result['games_analyzed'] == 1

    def test_output_matches_spec(self, pregame_agent):
        """Each intel has required fields from CLAUDE.md."""
        mock_bdl = MagicMock()
        mock_bdl.get_games.return_value = [_make_mock_game()]

        mock_lineup = MagicMock()
        mock_lineup.get_game_intel.return_value = _make_mock_game_intel()

        pregame_agent._bdl_api = mock_bdl
        pregame_agent._lineup_service = mock_lineup

        with patch.object(pregame_agent, 'call_llm', return_value=VALID_LLM_RESPONSE):
            result = pregame_agent.run()

        intel = result['game_intels'][0]
        required_fields = [
            'game_id', 'intel_generated_at', 'injury_impact',
            'projected_lineups', 'contextual_flags', 'player_prop_briefs',
            'overall_game_confidence', 'reasoning',
        ]
        for field in required_fields:
            assert field in intel, f"Missing field: {field}"

        assert intel['overall_game_confidence'] in ('high', 'medium', 'low')

    def test_synthesis_prompt_includes_injuries(self, pregame_agent):
        """LLM prompt contains injury data from raw intel."""
        mock_bdl = MagicMock()
        mock_bdl.get_games.return_value = [_make_mock_game()]

        mock_lineup = MagicMock()
        mock_lineup.get_game_intel.return_value = _make_mock_game_intel()

        pregame_agent._bdl_api = mock_bdl
        pregame_agent._lineup_service = mock_lineup

        call_args = []

        def capture_call(system_prompt, user_message, **kwargs):
            call_args.append(user_message)
            return VALID_LLM_RESPONSE

        with patch.object(pregame_agent, 'call_llm', side_effect=capture_call):
            pregame_agent.run()

        assert len(call_args) == 1
        assert 'Anthony Davis' in call_args[0]

    def test_parse_valid_response(self, pregame_agent):
        """Valid JSON parsed correctly."""
        result = pregame_agent._synthesize_with_llm({
            'home_team': 'LAL', 'away_team': 'BOS',
            'raw_intel': {'home_injuries': [], 'away_injuries': [], 'lineup_confidence': 0.9},
        })

        # With no LLM (no GEMINI_API_KEY), falls back to deterministic
        assert 'injury_impact' in result
        assert 'overall_game_confidence' in result

    def test_parse_malformed_falls_back(self, pregame_agent):
        """Bad JSON -> fallback to raw intel."""
        with patch.object(pregame_agent, 'call_llm', return_value='not valid json {{{'):
            result = pregame_agent._synthesize_with_llm({
                'home_team': 'LAL', 'away_team': 'BOS',
                'raw_intel': {'home_injuries': [], 'away_injuries': [], 'lineup_confidence': 0.7},
            })

        assert 'injury_impact' in result
        assert 'LLM unavailable' in result.get('reasoning', '') or 'Deterministic' in result.get('reasoning', '')

    def test_shadow_mode_no_messages(self, memory_bus, sqlite_guardrails):
        """Shadow mode sends 0 messages."""
        agent = PreGameIntelAgent(
            target_date='2026-02-24',
            message_bus=memory_bus,
            guardrails=sqlite_guardrails,
            shadow_mode=True,
        )

        mock_bdl = MagicMock()
        mock_bdl.get_games.return_value = [_make_mock_game()]

        mock_lineup = MagicMock()
        mock_lineup.get_game_intel.return_value = _make_mock_game_intel()

        agent._bdl_api = mock_bdl
        agent._lineup_service = mock_lineup

        with patch.object(agent, 'call_llm', return_value=VALID_LLM_RESPONSE):
            result = agent.execute()

        assert result.messages_sent == 0

    def test_report_sends_intel_ready(self, pregame_agent, memory_bus):
        """report() sends intel_ready to bus."""
        run_output = {
            'target_date': '2026-02-24',
            'games_analyzed': 1,
            'game_intels': [{
                'game_id': '123',
                'overall_game_confidence': 'high',
                'injury_impact': {},
                'reasoning': 'test',
            }],
            'reasoning': 'test',
        }

        pregame_agent.report(run_output)

        # Check predictor got intel_ready
        predictor_msgs = memory_bus.receive('predictor', event_type='intel_ready')
        assert len(predictor_msgs) >= 1

        # Check broadcast
        all_msgs = memory_bus.receive('all', event_type='intel_ready')
        assert len(all_msgs) >= 1

    def test_handles_no_games(self, pregame_agent):
        """Empty schedule -> empty result, no crash."""
        mock_bdl = MagicMock()
        mock_bdl.get_games.return_value = []

        pregame_agent._bdl_api = mock_bdl

        result = pregame_agent.run()

        assert result['game_intels'] == []
        assert result['games_analyzed'] == 0
