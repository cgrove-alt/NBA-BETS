"""
Pre-Game Intelligence Agent

Gathers, synthesizes, and interprets all context that affects a game
before predictions are generated. Wraps lineup_intel/ with Claude-powered
reasoning for injury cascade analysis, lineup uncertainty assessment,
and player prop context generation.

Trigger: 6-8 hours before first game, re-run 2 hours before tip-off.
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from agents.core.agent_base import AgentBase

logger = logging.getLogger(__name__)


class PreGameIntelAgent(AgentBase):
    """
    Pre-Game Intelligence Agent.

    Wraps LineupIntelService with LLM reasoning to produce
    game intel matching the CLAUDE.md spec.
    """

    AGENT_NAME = 'pregame'
    DAILY_TOKEN_BUDGET = 80_000
    MAX_EXECUTION_SECONDS = 600

    def __init__(self, target_date: str = None, **kwargs):
        super().__init__(**kwargs)
        self.target_date = target_date or datetime.now().strftime('%Y-%m-%d')
        self._lineup_service = None
        self._bdl_api = None

    def _get_lineup_service(self):
        """Lazy-init LineupIntelService."""
        if self._lineup_service is None:
            from lineup_intel import LineupIntelService
            self._lineup_service = LineupIntelService()
        return self._lineup_service

    def _get_bdl_api(self):
        """Lazy-init BalldontlieAPI."""
        if self._bdl_api is None:
            from nba_data.sources.balldontlie_api import BalldontlieAPI
            api_key = os.environ.get('BALLDONTLIE_API_KEY', '')
            self._bdl_api = BalldontlieAPI(api_key=api_key)
        return self._bdl_api

    def _load_system_prompt(self) -> str:
        """Load the version-controlled system prompt."""
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'prompts', 'pregame.md'
        )
        try:
            with open(prompt_path) as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"System prompt not found at {prompt_path}, using default")
            return self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return (
            "You are the Pre-Game Intelligence Agent for an NBA betting model. "
            "Analyze the provided game data and produce structured JSON intel. "
            "Focus on: injury cascade effects, lineup uncertainty, schedule spots, "
            "and player prop context. Never fabricate data. Flag uncertainty explicitly. "
            "Output valid JSON matching the schema provided."
        )

    def _get_schedule_context(self, team_games: list, team_abbr: str, game_date: str) -> dict:
        """Analyze schedule context (B2B, rest days, etc.)."""
        context = {
            'is_back_to_back': False,
            'days_rest': None,
            'games_in_5_days': 0,
            'is_road_trip': False,
        }

        try:
            target = datetime.strptime(game_date, '%Y-%m-%d').date()
            recent_dates = []

            for g in team_games:
                gd = g.get('date', '')
                if isinstance(gd, str) and len(gd) >= 10:
                    try:
                        d = datetime.strptime(gd[:10], '%Y-%m-%d').date()
                        if d < target:
                            recent_dates.append(d)
                    except ValueError:
                        pass

            if recent_dates:
                recent_dates.sort(reverse=True)
                last_game = recent_dates[0]
                days_rest = (target - last_game).days - 1
                context['days_rest'] = days_rest
                context['is_back_to_back'] = (days_rest == 0)

                # Games in last 5 days
                five_ago = target - timedelta(days=5)
                context['games_in_5_days'] = sum(1 for d in recent_dates if d >= five_ago)

        except Exception as e:
            logger.warning(f"Schedule context analysis failed for {team_abbr}: {e}")

        return context

    def _synthesize_with_llm(self, game_data: dict) -> dict:
        """
        Call LLM to synthesize raw game data into structured intel.

        Falls back to raw intel if LLM is unavailable or returns bad JSON.
        """
        system_prompt = self._load_system_prompt()

        user_message = json.dumps({
            'task': 'Analyze this game and produce pre-game intelligence',
            'game': game_data,
        }, indent=2, default=str)

        response = self.call_llm(system_prompt, user_message, max_tokens=4096)

        if not response:
            return self._fallback_intel(game_data)

        try:
            parsed = json.loads(response)
            # Validate required fields
            required = ['injury_impact', 'overall_game_confidence', 'reasoning']
            if all(k in parsed for k in required):
                return parsed
            else:
                logger.warning(f"[{self.AGENT_NAME}] LLM response missing required fields, using fallback")
                return self._fallback_intel(game_data)
        except json.JSONDecodeError:
            logger.warning(f"[{self.AGENT_NAME}] LLM returned invalid JSON, using fallback")
            return self._fallback_intel(game_data)

    def _fallback_intel(self, game_data: dict) -> dict:
        """Generate deterministic intel when LLM is unavailable."""
        home = game_data.get('home_team', 'UNK')
        away = game_data.get('away_team', 'UNK')
        raw_intel = game_data.get('raw_intel', {})

        # Build from raw LineupIntelService data
        home_injuries = raw_intel.get('home_injuries', [])
        away_injuries = raw_intel.get('away_injuries', [])

        home_out = [inj for inj in home_injuries if inj.get('status') == 'OUT']
        away_out = [inj for inj in away_injuries if inj.get('status') == 'OUT']

        confidence = 'medium'
        if raw_intel.get('lineup_confidence', 0) > 0.8:
            confidence = 'high'
        elif raw_intel.get('lineup_confidence', 0) < 0.5:
            confidence = 'low'

        flags = []
        schedule = game_data.get('schedule_context', {})
        if schedule.get('is_back_to_back'):
            flags.append('back_to_back')
        if schedule.get('days_rest', 1) >= 3:
            flags.append('well_rested')

        return {
            'injury_impact': {
                'home': {
                    'missing_players': [p.get('player_name', '') for p in home_out],
                    'impact_assessment': f"{len(home_out)} players out" if home_out else "Full strength",
                    'rotation_changes': 'Unknown — LLM unavailable',
                },
                'away': {
                    'missing_players': [p.get('player_name', '') for p in away_out],
                    'impact_assessment': f"{len(away_out)} players out" if away_out else "Full strength",
                    'rotation_changes': 'Unknown — LLM unavailable',
                },
            },
            'projected_lineups': raw_intel.get('projected_lineups', {}),
            'contextual_flags': flags,
            'player_prop_briefs': {},
            'overall_game_confidence': confidence,
            'reasoning': f"Deterministic fallback (LLM unavailable). {home} vs {away}. "
                         f"Home injuries: {len(home_out)} out. Away injuries: {len(away_out)} out.",
        }

    def run(self) -> dict:
        """
        Core pre-game intelligence gathering.

        1. Fetch today's games
        2. For each game, gather raw intel via LineupIntelService
        3. Synthesize with LLM reasoning
        4. Return structured intel per game
        """
        logger.info(f"[{self.AGENT_NAME}] Running for date: {self.target_date}")

        # Fetch today's games
        try:
            bdl = self._get_bdl_api()
            games = bdl.get_games(dates=[self.target_date])
        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] Failed to fetch games: {e}")
            games = []

        if not games:
            logger.info(f"[{self.AGENT_NAME}] No games scheduled for {self.target_date}")
            return {
                'target_date': self.target_date,
                'game_intels': [],
                'games_analyzed': 0,
                'reasoning': f"No games scheduled for {self.target_date}",
            }

        lineup_service = self._get_lineup_service()
        game_intels = []

        for game in games:
            try:
                game_intel = self._analyze_game(game, lineup_service)
                game_intels.append(game_intel)
            except Exception as e:
                logger.error(f"[{self.AGENT_NAME}] Failed to analyze game {game.get('id', '?')}: {e}")
                game_intels.append({
                    'game_id': game.get('id', 'unknown'),
                    'error': str(e),
                    'overall_game_confidence': 'low',
                    'reasoning': f"Analysis failed: {e}",
                })

        return {
            'target_date': self.target_date,
            'game_intels': game_intels,
            'games_analyzed': len(game_intels),
            'reasoning': f"Analyzed {len(game_intels)} games for {self.target_date}",
        }

    def _analyze_game(self, game: dict, lineup_service) -> dict:
        """Analyze a single game."""
        # Extract team info
        home_team = game.get('home_team', {})
        away_team = game.get('visitor_team', {})
        home_abbr = home_team.get('abbreviation', 'UNK')
        away_abbr = away_team.get('abbreviation', 'UNK')
        game_id = game.get('id', 'unknown')

        logger.info(f"[{self.AGENT_NAME}] Analyzing: {away_abbr} @ {home_abbr}")

        # Get raw intel from LineupIntelService
        try:
            raw_intel = lineup_service.get_game_intel(home_abbr, away_abbr, game_date=self.target_date)
            raw_intel_dict = raw_intel.to_dict() if hasattr(raw_intel, 'to_dict') else {}
        except Exception as e:
            logger.warning(f"[{self.AGENT_NAME}] LineupIntelService failed for {home_abbr} vs {away_abbr}: {e}")
            raw_intel_dict = {}

        # Get schedule context
        schedule_home = {}
        schedule_away = {}
        try:
            bdl = self._get_bdl_api()
            # Get recent games for schedule context
            home_team_id = home_team.get('id')
            away_team_id = away_team.get('id')
            if home_team_id:
                recent_home = bdl.get_games(team_ids=[home_team_id])
                schedule_home = self._get_schedule_context(recent_home, home_abbr, self.target_date)
            if away_team_id:
                recent_away = bdl.get_games(team_ids=[away_team_id])
                schedule_away = self._get_schedule_context(recent_away, away_abbr, self.target_date)
        except Exception as e:
            logger.warning(f"[{self.AGENT_NAME}] Schedule context failed: {e}")

        # Prepare data for LLM synthesis
        game_data = {
            'game_id': str(game_id),
            'home_team': home_abbr,
            'away_team': away_abbr,
            'game_date': self.target_date,
            'raw_intel': raw_intel_dict,
            'schedule_context': {
                'home': schedule_home,
                'away': schedule_away,
            },
        }

        # Synthesize with LLM
        synthesized = self._synthesize_with_llm(game_data)

        # Build final output matching CLAUDE.md spec
        return {
            'game_id': str(game_id),
            'intel_generated_at': datetime.now(timezone.utc).isoformat(),
            'home_team': home_abbr,
            'away_team': away_abbr,
            'injury_impact': synthesized.get('injury_impact', {}),
            'projected_lineups': synthesized.get('projected_lineups', {}),
            'contextual_flags': synthesized.get('contextual_flags', []),
            'player_prop_briefs': synthesized.get('player_prop_briefs', {}),
            'overall_game_confidence': synthesized.get('overall_game_confidence', 'medium'),
            'reasoning': synthesized.get('reasoning', ''),
        }

    def report(self, run_output: dict):
        """Send intel_ready messages for each game."""
        game_intels = run_output.get('game_intels', [])

        for intel in game_intels:
            if intel.get('error'):
                continue

            # Send to future Prediction Orchestrator
            self.send_message(
                recipient='orchestrator',
                event_type='intel_ready',
                payload=intel,
                priority='normal',
            )

        # Send summary to broadcast
        self.send_message(
            recipient='all',
            event_type='intel_ready',
            payload={
                'date': run_output.get('target_date'),
                'games_analyzed': run_output.get('games_analyzed', 0),
                'summary': run_output.get('reasoning', ''),
            },
            priority='normal',
        )
