"""
Odds Monitoring Agent

Tracks line movements in real-time and interprets what the market is
telling us. Wraps MarketMonitor from nba_betting/odds/market_microstructure.py
with LLM reasoning for sharp vs public money classification and RLM detection.

Trigger: Every 15 minutes during game hours (8 AM–11 PM ET).
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from agents.core.agent_base import AgentBase

logger = logging.getLogger(__name__)

# Edge threshold for flagging notable events to the orchestrator
NOTABLE_EDGE_THRESHOLD = 0.025  # 2.5%


class OddsMonitorAgent(AgentBase):
    """
    Odds Monitoring Agent.

    Wraps MarketMonitor with LLM reasoning to interpret steam moves,
    stale lines, and sharp money signals.
    """

    AGENT_NAME = 'odds_monitor'
    DAILY_TOKEN_BUDGET = 40_000
    MAX_EXECUTION_SECONDS = 180

    def __init__(self, target_date: str = None, **kwargs):
        super().__init__(**kwargs)
        self.target_date = target_date or datetime.now().strftime('%Y-%m-%d')
        self._market_monitor = None

    def _get_market_monitor(self):
        """Lazy-init MarketMonitor."""
        if self._market_monitor is None:
            from nba_betting.odds.market_microstructure import MarketMonitor
            self._market_monitor = MarketMonitor()
        return self._market_monitor

    def _load_system_prompt(self) -> str:
        """Load the version-controlled system prompt."""
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'prompts', 'odds_monitor.md'
        )
        try:
            with open(prompt_path) as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"System prompt not found at {prompt_path}, using default")
            return self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return (
            "You are the Odds Monitoring Agent for an NBA betting model. "
            "Analyze steam moves and stale lines to classify sharp vs public action, "
            "detect reverse line movement, and rate signal confidence. "
            "Output valid JSON with recommendation, sharp_money_assessment, and reasoning."
        )

    def _serialize_alert(self, alert) -> dict:
        """Convert a SteamAlert or StaleLine to a serializable dict."""
        if hasattr(alert, '__dict__'):
            d = {}
            for k, v in alert.__dict__.items():
                if not k.startswith('_'):
                    d[k] = v
            return d
        return str(alert)

    def _interpret_with_llm(self, steam_alerts: list, stale_lines: list) -> dict:
        """
        Call LLM to interpret market signals.

        Falls back to deterministic summary if LLM unavailable.
        """
        system_prompt = self._load_system_prompt()

        user_message = json.dumps({
            'task': 'Interpret these market signals for an NBA betting model',
            'steam_alerts': [self._serialize_alert(a) for a in steam_alerts],
            'stale_lines': [self._serialize_alert(s) for s in stale_lines],
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str)

        response = self.call_llm(system_prompt, user_message, max_tokens=2048)

        if not response:
            return self._fallback_interpretation(steam_alerts, stale_lines)

        try:
            parsed = json.loads(response)
            if 'reasoning' in parsed:
                return parsed
            return self._fallback_interpretation(steam_alerts, stale_lines)
        except json.JSONDecodeError:
            logger.warning(f"[{self.AGENT_NAME}] LLM returned invalid JSON, using fallback")
            return self._fallback_interpretation(steam_alerts, stale_lines)

    def _fallback_interpretation(self, steam_alerts: list, stale_lines: list) -> dict:
        """Deterministic interpretation when LLM is unavailable."""
        notable = []

        for alert in steam_alerts:
            confidence = getattr(alert, 'confidence', 0)
            notable.append({
                'event_type': 'steam_move',
                'game_id': getattr(alert, 'game_id', 'unknown'),
                'market': getattr(alert, 'market', ''),
                'side': getattr(alert, 'side', ''),
                'confidence': confidence,
                'recommendation': 're-evaluate' if confidence > 0.7 else 'hold',
                'sharp_money_assessment': 'likely_sharp' if confidence > 0.5 else 'unclear',
            })

        for stale in stale_lines:
            edge = getattr(stale, 'edge', 0)
            notable.append({
                'event_type': 'stale_line',
                'game_id': getattr(stale, 'game_id', 'unknown'),
                'book': getattr(stale, 'book', ''),
                'market': getattr(stale, 'market', ''),
                'edge': edge,
                'recommendation': 'urgent_review' if edge > 0.05 else 'hold',
            })

        return {
            'notable_movements': notable,
            'reasoning': (
                f"Deterministic fallback (LLM unavailable). "
                f"Found {len(steam_alerts)} steam alerts and {len(stale_lines)} stale lines."
            ),
        }

    def run(self) -> dict:
        """
        Core odds monitoring logic.

        1. Call MarketMonitor.check_once() for steam + stale detection
        2. If notable events, call LLM to interpret
        3. Return structured results
        """
        logger.info(f"[{self.AGENT_NAME}] Running odds check for {self.target_date}")

        monitor = self._get_market_monitor()

        try:
            check_result = monitor.check_once()
        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] MarketMonitor.check_once() failed: {e}")
            return {
                'target_date': self.target_date,
                'poll_timestamp': datetime.now(timezone.utc).isoformat(),
                'steam_alerts': [],
                'stale_lines': [],
                'notable_movements': [],
                'reasoning': f"MarketMonitor failed: {e}",
            }

        steam_alerts = check_result.get('steam', [])
        stale_lines = check_result.get('stale', [])

        logger.info(
            f"[{self.AGENT_NAME}] Found {len(steam_alerts)} steam alerts, "
            f"{len(stale_lines)} stale lines"
        )

        # Determine if LLM interpretation is needed
        has_notable = len(steam_alerts) > 0 or any(
            getattr(s, 'edge', 0) > NOTABLE_EDGE_THRESHOLD for s in stale_lines
        )

        if has_notable:
            interpretation = self._interpret_with_llm(steam_alerts, stale_lines)
        else:
            interpretation = {
                'notable_movements': [],
                'reasoning': 'No notable market events detected.',
            }

        return {
            'target_date': self.target_date,
            'poll_timestamp': datetime.now(timezone.utc).isoformat(),
            'steam_alerts': [self._serialize_alert(a) for a in steam_alerts],
            'stale_lines': [self._serialize_alert(s) for s in stale_lines],
            'notable_movements': interpretation.get('notable_movements', []),
            'reasoning': interpretation.get('reasoning', ''),
        }

    def report(self, run_output: dict):
        """Send odds_alert messages to predictor and briefing."""
        steam_alerts = run_output.get('steam_alerts', [])
        stale_lines = run_output.get('stale_lines', [])
        notable = run_output.get('notable_movements', [])

        # Send high-priority alerts to predictor for steam moves
        for alert in steam_alerts:
            edge = 0
            laggards = alert.get('laggard_books', [])
            if laggards:
                edge = max(l.get('edge', 0) for l in laggards)

            if edge > NOTABLE_EDGE_THRESHOLD:
                self.send_message(
                    recipient='orchestrator',
                    event_type='odds_alert',
                    payload={
                        'event_type': 'steam_move',
                        'game_id': alert.get('game_id', ''),
                        'market': alert.get('market', ''),
                        'side': alert.get('side', ''),
                        'edge': edge,
                        'details': alert,
                    },
                    priority='high',
                )

        # Send summary to briefing
        self.send_message(
            recipient='briefing',
            event_type='odds_alert',
            payload={
                'date': run_output.get('target_date'),
                'steam_count': len(steam_alerts),
                'stale_count': len(stale_lines),
                'notable_movements': notable,
                'reasoning': run_output.get('reasoning', ''),
            },
            priority='normal',
        )
