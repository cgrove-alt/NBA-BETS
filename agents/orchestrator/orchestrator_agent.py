"""
Prediction Orchestrator Agent

Coordinates the full prediction pipeline and makes the final call on
which predictions to publish. Wraps daily_predictions.py with judgment
from pre-game intel and odds monitoring.

Does NOT duplicate daily_predictions.py. Instead wraps it: reads intel
messages, invokes the existing pipeline, then adds a judgment layer
(confidence adjustments, correlation checks, bankroll sizing, conflict
resolution).

Trigger: 11:30 AM ET (30 min after pregame's first run).
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

from agents.core.agent_base import AgentBase

logger = logging.getLogger(__name__)

# Bankroll constraints (from CLAUDE.md)
MAX_SINGLE_BET_PCT = 0.03      # 3% of bankroll
MAX_DAILY_EXPOSURE_PCT = 0.10   # 10% of bankroll
MAX_CORRELATED_PCT = 0.05       # 5% on same game
CORRELATION_THRESHOLD = 3       # Trigger correlation check at 3+ BET signals on same team


class PredictionOrchestratorAgent(AgentBase):
    """
    Prediction Orchestrator Agent.

    Wraps the daily_predictions pipeline with intel-based judgment:
    confidence adjustments, correlation checks, bankroll sizing,
    and conflict resolution.
    """

    AGENT_NAME = 'predictor'
    DAILY_TOKEN_BUDGET = 60_000
    MAX_EXECUTION_SECONDS = 900

    def __init__(self, target_date: str = None, **kwargs):
        super().__init__(**kwargs)
        self.target_date = target_date or datetime.now().strftime('%Y-%m-%d')
        self._models = None

    def _load_system_prompt(self) -> str:
        """Load the version-controlled system prompt."""
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'prompts', 'orchestrator.md'
        )
        try:
            with open(prompt_path) as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"System prompt not found at {prompt_path}, using default")
            return self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return (
            "You are the Prediction Orchestrator Agent for an NBA betting model. "
            "Resolve conflicts between model and market signals, assess correlation risk, "
            "and justify confidence adjustments. Conservative — never publish on stale data. "
            "Output valid JSON with adjustments, correlation_warnings, conflict_resolutions, and reasoning."
        )

    def _load_models(self) -> dict:
        """Lazy-load prediction models."""
        if self._models is None:
            try:
                from nba_models.inference.daily_predictions import load_models
                self._models = load_models()
            except Exception as e:
                logger.error(f"[{self.AGENT_NAME}] Failed to load models: {e}")
                self._models = {}
        return self._models

    def _run_predictions(self) -> list:
        """
        Run the existing daily_predictions pipeline.

        Returns list of game analyses from analyze_game().
        """
        models = self._load_models()
        if not models:
            logger.warning(f"[{self.AGENT_NAME}] No models loaded, cannot generate predictions")
            return []

        try:
            from nba_data.sources.balldontlie_api import BalldontlieAPI
            api = BalldontlieAPI()
            games = api.get_games(dates=[self.target_date])
        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] Failed to fetch games: {e}")
            return []

        if not games:
            logger.info(f"[{self.AGENT_NAME}] No games for {self.target_date}")
            return []

        # Fetch odds
        odds_data = {}
        try:
            odds_list = api.get_betting_odds(date=self.target_date)
            preferred_vendors = ['fanduel', 'draftkings', 'betmgm', 'caesars']
            for odds in odds_list:
                game_id = odds.get('game_id')
                vendor = odds.get('vendor', '').lower()
                if game_id not in odds_data or vendor in preferred_vendors:
                    odds_data[game_id] = odds
        except Exception as e:
            logger.warning(f"[{self.AGENT_NAME}] Failed to fetch odds: {e}")

        # Run analyze_game for each game
        from nba_models.inference.daily_predictions import analyze_game

        results = []
        for game in games:
            game_id = game.get('id')
            game_odds = odds_data.get(game_id, {})
            try:
                analysis = analyze_game(game, game_odds, models)
                results.append(analysis)
            except Exception as e:
                logger.error(f"[{self.AGENT_NAME}] analyze_game failed for {game_id}: {e}")
                results.append({
                    'game_id': game_id,
                    'home_team': game.get('home_team', {}).get('abbreviation', '?'),
                    'away_team': game.get('visitor_team', {}).get('abbreviation', '?'),
                    'error': str(e),
                })

        return results

    def _apply_intel_adjustments(self, predictions: list, intel_msgs: list) -> list:
        """
        Adjust prediction confidence based on pre-game intel.

        - low game confidence → downgrade one tier
        - high game confidence + confirmed lineup → maintain
        """
        # Build lookup: game_id -> intel
        intel_by_game = {}
        for msg in intel_msgs:
            game_id = str(msg.payload.get('game_id', ''))
            if game_id:
                intel_by_game[game_id] = msg.payload

        for pred in predictions:
            game_id = str(pred.get('game_id', ''))
            intel = intel_by_game.get(game_id, {})

            if not intel:
                continue

            game_confidence = intel.get('overall_game_confidence', 'medium')

            # Adjust player prop confidences
            for prop in pred.get('player_props', []):
                current_conf = prop.get('confidence', 'medium')

                if game_confidence == 'low':
                    # Downgrade one tier
                    downgrade = {'high': 'medium', 'medium': 'low', 'low': 'low'}
                    prop['confidence'] = downgrade.get(current_conf, current_conf)
                    prop['intel_adjustment'] = 'downgraded (low lineup confidence)'
                elif game_confidence == 'high':
                    prop['intel_adjustment'] = 'maintained (high confidence intel)'
                else:
                    prop['intel_adjustment'] = 'none'

        return predictions

    def _check_correlations(self, predictions: list) -> list:
        """
        Check for dangerous correlation clusters.

        If 3+ BET signals on the same team, flag for review.
        """
        warnings = []

        # Group BET signals by team
        team_bets = defaultdict(list)
        for pred in predictions:
            for prop in pred.get('player_props', []):
                signal = prop.get('signal', prop.get('bet_recommendation', 'PASS'))
                if signal == 'BET':
                    team = prop.get('team', pred.get('home_team', ''))
                    team_bets[team].append(prop)

        for team, bets in team_bets.items():
            if len(bets) >= CORRELATION_THRESHOLD:
                warnings.append({
                    'team': team,
                    'num_bets': len(bets),
                    'players': [b.get('player_name', '?') for b in bets],
                    'action': f"Review {len(bets)} correlated BET signals on {team}",
                })

        return warnings

    def _apply_correlation_downgrades(self, predictions: list, warnings: list) -> list:
        """Downgrade excess correlated bets from BET to LEAN."""
        teams_to_check = {w['team'] for w in warnings}

        for pred in predictions:
            for prop in pred.get('player_props', []):
                signal = prop.get('signal', prop.get('bet_recommendation', 'PASS'))
                team = prop.get('team', pred.get('home_team', ''))

                if team in teams_to_check and signal == 'BET':
                    edge = prop.get('edge', prop.get('edge_pct', 0))
                    # Keep the highest-edge bets, downgrade others
                    if edge < 6.0:  # Below high-confidence threshold
                        prop['signal'] = 'LEAN'
                        prop['bet_recommendation'] = 'LEAN'
                        prop['correlation_downgrade'] = True

        return predictions

    def _apply_bankroll_sizing(self, predictions: list) -> dict:
        """
        Apply quarter-Kelly bankroll sizing.

        Returns exposure summary.
        """
        total_units = 0.0
        game_exposure = defaultdict(float)

        for pred in predictions:
            game_id = pred.get('game_id', '')

            # Spread bet sizing
            spread = pred.get('spread', {})
            spread_signal = spread.get('signal', 'PASS')
            if spread_signal in ('BET', 'LEAN'):
                units = self._size_bet(
                    spread.get('edge', spread.get('spread_edge_pct', 0)),
                    spread_signal,
                )
                spread['recommended_units'] = units
                total_units += units
                game_exposure[game_id] += units

            # Prop bet sizing
            for prop in pred.get('player_props', []):
                signal = prop.get('signal', prop.get('bet_recommendation', 'PASS'))
                if signal in ('BET', 'LEAN'):
                    units = self._size_bet(
                        prop.get('edge', prop.get('edge_pct', 0)),
                        signal,
                    )
                    prop['recommended_units'] = units
                    total_units += units
                    game_exposure[game_id] += units

        # Enforce daily cap (10% = 10 units since 1u = 1%)
        max_daily = 10.0
        if total_units > max_daily:
            scale = max_daily / total_units
            self._scale_all_units(predictions, scale)
            total_units = max_daily

        return {
            'total_units': round(total_units, 2),
            'total_pct': round(total_units, 2),  # 1u = 1%
            'game_exposure': {k: round(v, 2) for k, v in game_exposure.items()},
            'capped': total_units >= max_daily,
        }

    def _size_bet(self, edge: float, signal: str) -> float:
        """Calculate unit size for a bet using quarter-Kelly."""
        if edge <= 0:
            return 0.0

        # Quarter-Kelly: edge% / 4, with signal modifier
        base_units = edge / 4.0

        if signal == 'LEAN':
            base_units *= 0.5  # Half size for LEAN

        # Cap at 3 units (3% of bankroll)
        return min(round(base_units, 2), 3.0)

    def _scale_all_units(self, predictions: list, scale: float):
        """Scale all recommended_units by a factor."""
        for pred in predictions:
            spread = pred.get('spread', {})
            if 'recommended_units' in spread:
                spread['recommended_units'] = round(spread['recommended_units'] * scale, 2)

            for prop in pred.get('player_props', []):
                if 'recommended_units' in prop:
                    prop['recommended_units'] = round(prop['recommended_units'] * scale, 2)

    def _resolve_conflicts_with_llm(self, predictions: list, odds_msgs: list) -> dict:
        """
        Use LLM to resolve conflicts between model and sharp money.

        Only called when there are actual conflicts to resolve.
        """
        # Identify games where sharp money disagrees with model
        conflicts = []
        for msg in odds_msgs:
            payload = msg.payload
            if payload.get('event_type') == 'steam_move':
                game_id = payload.get('game_id', '')
                # Find matching prediction
                for pred in predictions:
                    if str(pred.get('game_id', '')) == str(game_id):
                        conflicts.append({
                            'game_id': game_id,
                            'model_prediction': pred.get('spread', {}),
                            'sharp_signal': payload,
                        })
                        break

        if not conflicts:
            return {'conflict_resolutions': [], 'reasoning': 'No model-market conflicts detected.'}

        system_prompt = self._load_system_prompt()

        user_message = json.dumps({
            'task': 'Resolve conflicts between model predictions and sharp money signals',
            'conflicts': conflicts,
        }, indent=2, default=str)

        response = self.call_llm(system_prompt, user_message, max_tokens=2048)

        if not response:
            return {
                'conflict_resolutions': [{
                    'game_id': c['game_id'],
                    'resolution': 'Unable to resolve — defaulting to model prediction',
                } for c in conflicts],
                'reasoning': 'LLM unavailable for conflict resolution.',
            }

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                'conflict_resolutions': [],
                'reasoning': 'LLM returned invalid JSON for conflict resolution.',
            }

    def run(self) -> dict:
        """
        Core orchestration logic.

        1. Read intel + odds messages
        2. Run prediction pipeline
        3. Apply confidence adjustments
        4. Check correlations
        5. Resolve conflicts with LLM if needed
        6. Apply bankroll sizing
        7. Return final predictions
        """
        logger.info(f"[{self.AGENT_NAME}] Orchestrating predictions for {self.target_date}")

        # Step 1: Read messages from other agents
        intel_msgs = self.get_messages(event_type='intel_ready')
        odds_msgs = self.get_messages(event_type='odds_alert')

        logger.info(
            f"[{self.AGENT_NAME}] Context: {len(intel_msgs)} intel messages, "
            f"{len(odds_msgs)} odds alerts"
        )

        # Step 2: Run prediction pipeline
        predictions = self._run_predictions()

        if not predictions:
            return {
                'slate_date': self.target_date,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'predictions': [],
                'correlation_warnings': [],
                'total_exposure': {'total_units': 0, 'total_pct': 0},
                'reasoning': f"No games or predictions for {self.target_date}",
            }

        # Step 3: Apply intel-based confidence adjustments
        predictions = self._apply_intel_adjustments(predictions, intel_msgs)

        # Step 4: Check and manage correlations
        correlation_warnings = self._check_correlations(predictions)
        if correlation_warnings:
            logger.info(f"[{self.AGENT_NAME}] {len(correlation_warnings)} correlation warnings")
            predictions = self._apply_correlation_downgrades(predictions, correlation_warnings)

        # Step 5: Resolve model vs market conflicts
        conflict_result = {}
        if odds_msgs:
            conflict_result = self._resolve_conflicts_with_llm(predictions, odds_msgs)

        # Step 6: Apply bankroll sizing
        exposure = self._apply_bankroll_sizing(predictions)

        # Build final output
        return {
            'slate_date': self.target_date,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'predictions': predictions,
            'correlation_warnings': correlation_warnings,
            'conflict_resolutions': conflict_result.get('conflict_resolutions', []),
            'total_exposure': exposure,
            'intel_context': {
                'intel_messages': len(intel_msgs),
                'odds_alerts': len(odds_msgs),
            },
            'reasoning': (
                f"Orchestrated {len(predictions)} game predictions for {self.target_date}. "
                f"Intel: {len(intel_msgs)} messages. Odds: {len(odds_msgs)} alerts. "
                f"Correlations: {len(correlation_warnings)} warnings. "
                f"Exposure: {exposure['total_units']}u ({exposure['total_pct']}%)."
            ),
        }

    def report(self, run_output: dict):
        """Send predictions_published to briefing and all."""
        predictions = run_output.get('predictions', [])

        # Summary for briefing
        bet_count = 0
        lean_count = 0
        for pred in predictions:
            for prop in pred.get('player_props', []):
                signal = prop.get('signal', prop.get('bet_recommendation', 'PASS'))
                if signal == 'BET':
                    bet_count += 1
                elif signal == 'LEAN':
                    lean_count += 1

            spread_signal = pred.get('spread', {}).get('signal', 'PASS')
            if spread_signal == 'BET':
                bet_count += 1
            elif spread_signal == 'LEAN':
                lean_count += 1

        # Send to briefing with full predictions
        self.send_message(
            recipient='briefing',
            event_type='predictions_published',
            payload={
                'slate_date': run_output.get('slate_date'),
                'predictions': predictions,
                'total_exposure': run_output.get('total_exposure', {}),
                'correlation_warnings': run_output.get('correlation_warnings', []),
                'bet_count': bet_count,
                'lean_count': lean_count,
            },
            priority='normal',
        )

        # Send summary broadcast
        self.send_message(
            recipient='all',
            event_type='predictions_published',
            payload={
                'slate_date': run_output.get('slate_date'),
                'games_count': len(predictions),
                'bet_count': bet_count,
                'lean_count': lean_count,
                'total_exposure': run_output.get('total_exposure', {}),
            },
            priority='normal',
        )
