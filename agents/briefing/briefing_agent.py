"""
Daily Briefing Agent

Synthesizes outputs from all other agents into a single, clear daily
briefing for Colin. This is the system's voice — no jargon, no code,
just actionable intelligence.

Trigger: Noon + 6 PM ET (after orchestrator, then 1hr before typical tip-off).
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from agents.core.agent_base import AgentBase

logger = logging.getLogger(__name__)


class DailyBriefingAgent(AgentBase):
    """
    Daily Briefing Agent.

    Most LLM-heavy agent. No existing service to wrap — entirely new.
    Reads messages from all other agents and synthesizes a human-readable
    briefing for Colin.
    """

    AGENT_NAME = 'briefing'
    DAILY_TOKEN_BUDGET = 80_000
    MAX_EXECUTION_SECONDS = 300

    def __init__(self, target_date: str = None, **kwargs):
        super().__init__(**kwargs)
        self.target_date = target_date or datetime.now().strftime('%Y-%m-%d')

    def _load_system_prompt(self) -> str:
        """Load the version-controlled system prompt."""
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'prompts', 'briefing.md'
        )
        try:
            with open(prompt_path) as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"System prompt not found at {prompt_path}, using default")
            return self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return (
            "You are the Daily Briefing Agent for an NBA betting model. "
            "Synthesize all agent outputs into a clear, plain-language briefing "
            "for a non-technical audience. Include: yesterday's results, today's plays, "
            "bankroll status, system health, and market intel. "
            "Output valid JSON with sections and formatted_text."
        )

    def _query_yesterday_results(self) -> Optional[dict]:
        """Direct DB fallback for yesterday's record when message bus has no data."""
        yesterday = (datetime.strptime(self.target_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            from agents.core.db_queries import query_yesterday_record
            return query_yesterday_record(yesterday)
        except Exception:
            logger.warning(f"[{self.AGENT_NAME}] Failed to query yesterday's record from DB")
            return None

    def _gather_context(self) -> dict:
        """
        Read all messages from other agents.

        Any agent may not have run yet — handle missing data gracefully.
        """
        context = {
            'predictions': None,
            'yesterday_results': None,
            'odds_intel': None,
            'health_check': None,
        }

        # From orchestrator: today's predictions
        pred_msgs = self.get_messages(event_type='predictions_published')
        if pred_msgs:
            # Take the most recent
            context['predictions'] = pred_msgs[-1].payload

        # From postgame: yesterday's results (message bus first, DB fallback)
        results_msgs = self.get_messages(event_type='results_analyzed')
        if results_msgs:
            context['yesterday_results'] = results_msgs[-1].payload
        else:
            context['yesterday_results'] = self._query_yesterday_results()

        # From odds monitor: market intelligence
        odds_msgs = self.get_messages(event_type='odds_alert')
        if odds_msgs:
            # Combine all recent odds alerts
            context['odds_intel'] = [m.payload for m in odds_msgs]

        # From watchdog: system health
        health_msgs = self.get_messages(event_type='health_check')
        if health_msgs:
            context['health_check'] = health_msgs[-1].payload

        return context

    def _build_sections_from_context(self, context: dict) -> dict:
        """Build briefing sections from available context (deterministic fallback)."""
        return {
            'yesterday_recap': self._build_yesterday_section(context.get('yesterday_results')),
            'today_plays': self._build_today_section(context.get('predictions')),
            'bankroll': self._build_bankroll_section(context.get('predictions')),
            'alerts': self._build_alerts_section(context.get('health_check')),
            'market_intel': self._build_market_section(context.get('odds_intel')),
        }

    def _build_yesterday_section(self, results: dict) -> dict:
        """Build yesterday's recap section.

        Handles two formats:
        - Message bus format: results_summary.wins/losses/roi_today/clv_average
        - DB format (from query_yesterday_record): overall.wins/losses/hit_rate, by_bet_type, etc.
        """
        if not results:
            return {
                'record': 'N/A',
                'roi': 'N/A',
                'pnl': 'N/A',
                'hit_rate': 'N/A',
                'clv_summary': 'No data available',
                'notable': 'No results data from post-game analysis',
            }

        # DB format: has 'overall' key
        if 'overall' in results:
            o = results['overall']
            cs = results.get('clv_summary')
            if cs:
                clv_str = f"CLV: {cs['avg_clv']:+.1f} avg | {cs['positive_clv_rate']:.0f}% positive CLV rate"
            else:
                clv_str = 'N/A'

            return {
                'record': f"{o['wins']}-{o['losses']}-{o['pushes']}",
                'roi': f"{o['roi']:+.1f}%" if o['roi'] else 'N/A',
                'pnl': f"${o['profit']:+,.0f}" if o.get('profit') else 'N/A',
                'hit_rate': f"{o['hit_rate']:.1f}%",
                'clv_summary': clv_str,
                'notable': '',
                'by_bet_type': results.get('by_bet_type', {}),
                'by_confidence': results.get('by_confidence', {}),
                'source': results.get('source', 'unknown'),
            }

        # Message bus format
        summary = results.get('results_summary', {})
        return {
            'record': f"{summary.get('wins', 0)}-{summary.get('losses', 0)}",
            'roi': summary.get('roi_today', 'N/A'),
            'pnl': summary.get('roi_today', 'N/A'),
            'clv_summary': f"CLV: {summary.get('clv_average', 'N/A')}",
            'notable': '',
        }

    def _build_today_section(self, predictions: dict) -> list:
        """Build today's plays section."""
        if not predictions:
            return []

        plays = []
        for pred in predictions.get('predictions', []):
            # Spread bet
            spread = pred.get('spread', {})
            spread_signal = spread.get('signal', 'PASS')
            if spread_signal in ('BET', 'LEAN'):
                plays.append({
                    'pick': f"{pred.get('home_team', '?')} {spread.get('line', '')}",
                    'units': spread.get('recommended_units', 1.0),
                    'edge': f"{spread.get('edge', spread.get('spread_edge_pct', 0)):.1f}%",
                    'confidence': spread.get('confidence', 'MEDIUM').upper(),
                    'signal': spread_signal,
                    'reasoning': spread.get('reasoning', ''),
                })

            # Player props
            for prop in pred.get('player_props', []):
                signal = prop.get('signal', prop.get('bet_recommendation', 'PASS'))
                if signal in ('BET', 'LEAN'):
                    pick_dir = prop.get('pick', 'Over')
                    line = prop.get('prop_line', prop.get('line', ''))
                    stat = prop.get('stat_type', prop.get('prop_type', ''))
                    player = prop.get('player_name', '?')

                    plays.append({
                        'pick': f"{player} {pick_dir} {line} {stat}",
                        'units': prop.get('recommended_units', 0.5),
                        'edge': f"{prop.get('edge', prop.get('edge_pct', 0)):.1f}%",
                        'confidence': prop.get('confidence', 'MEDIUM').upper(),
                        'signal': signal,
                        'reasoning': prop.get('reasoning', ''),
                    })

        return plays

    def _build_bankroll_section(self, predictions: dict) -> dict:
        """Build bankroll section."""
        if not predictions:
            return {
                'current': 'N/A',
                'today_exposure': '0u (0%)',
                'season_pnl': 'N/A',
            }

        exposure = predictions.get('total_exposure', {})
        return {
            'current': 'See dashboard',
            'today_exposure': f"{exposure.get('total_units', 0)}u ({exposure.get('total_pct', 0)}%)",
            'season_pnl': 'See dashboard',
        }

    def _build_alerts_section(self, health: dict) -> list:
        """Build alerts section."""
        if not health:
            return ['System health data unavailable — watchdog may not have run yet']

        alerts = []
        status = health.get('health_status', 'unknown')

        if status == 'healthy':
            return ['All systems healthy']

        for alert in health.get('alerts', []):
            severity = alert.get('severity', 'info')
            message = alert.get('message', alert.get('details', ''))
            alerts.append(f"[{severity.upper()}] {message}")

        retraining = health.get('retraining_recommendation', {})
        if retraining.get('recommended'):
            alerts.append(f"Model retraining recommended: {retraining.get('reason', '')}")

        return alerts or [f"System status: {status}"]

    def _build_market_section(self, odds_intel: list) -> list:
        """Build market intel section."""
        if not odds_intel:
            return ['No market intelligence available']

        intel_items = []
        for alert in odds_intel:
            reasoning = alert.get('reasoning', '')
            steam_count = alert.get('steam_count', 0)
            stale_count = alert.get('stale_count', 0)

            if steam_count > 0:
                intel_items.append(f"{steam_count} steam move(s) detected")
            if stale_count > 0:
                intel_items.append(f"{stale_count} stale line(s) found")

            notable = alert.get('notable_movements', [])
            for mov in notable[:3]:
                if isinstance(mov, dict):
                    intel_items.append(mov.get('reasoning', str(mov)))

            if reasoning and reasoning not in intel_items:
                intel_items.append(reasoning)

        return intel_items or ['No notable market movements']

    def _format_briefing_text(self, sections: dict) -> str:
        """Format sections into ASCII briefing matching CLAUDE.md spec."""
        lines = []
        date_str = self.target_date

        lines.append('=' * 55)
        lines.append(f'  NBA MODEL DAILY BRIEFING — {date_str}')
        lines.append('=' * 55)

        # Yesterday's Results
        recap = sections.get('yesterday_recap', {})
        lines.append('')
        lines.append('YESTERDAY\'S RESULTS')
        hit_rate_str = f" ({recap['hit_rate']})" if recap.get('hit_rate') and recap['hit_rate'] != 'N/A' else ''
        pnl_str = f" | {recap['pnl']}" if recap.get('pnl') and recap['pnl'] != 'N/A' else ''
        roi_str = f" | ROI: {recap.get('roi', 'N/A')}" if recap.get('roi') and recap['roi'] != 'N/A' else ''
        lines.append(f"  Record: {recap.get('record', 'N/A')}{hit_rate_str}{pnl_str}{roi_str}")

        # By bet type (only if data exists from DB format)
        by_type = recap.get('by_bet_type', {})
        if by_type:
            lines.append('')
            lines.append('  By Type:')
            sorted_types = sorted(by_type.items(), key=lambda x: x[1].get('total', 0), reverse=True)
            max_name_len = max(len(name) for name, _ in sorted_types) if sorted_types else 0
            for name, stats in sorted_types:
                if stats.get('total', 0) == 0:
                    continue
                pad = ' ' * (max_name_len - len(name) + 1)
                lines.append(f"    {name}{pad}{stats['wins']}-{stats['losses']}  {stats.get('hit_rate', 0):.1f}%")

        # By confidence (only if data exists from DB format)
        by_conf = recap.get('by_confidence', {})
        non_empty_conf = {k: v for k, v in by_conf.items() if v.get('total', 0) > 0}
        if non_empty_conf:
            lines.append('')
            lines.append('  By Confidence:')
            tier_labels = {
                'high': 'High (\u226560)',
                'medium': 'Medium (55-59)',
                'low': 'Low (<55)',
            }
            for tier in ('high', 'medium', 'low'):
                if tier in non_empty_conf:
                    s = non_empty_conf[tier]
                    label = tier_labels[tier]
                    lines.append(f"    {label:<16}{s['wins']}-{s['losses']}  {s.get('hit_rate', 0):.1f}%")

        # CLV line
        clv_str = recap.get('clv_summary', '')
        if clv_str and clv_str != 'N/A':
            lines.append('')
            lines.append(f"  {clv_str}")

        if recap.get('notable'):
            lines.append(f"  {recap['notable']}")

        # Today's Plays
        plays = sections.get('today_plays', [])
        lines.append('')
        lines.append(f"TODAY'S PLAYS ({len(plays)} recommended)")
        if plays:
            for play in plays:
                signal_icon = {'BET': 'BET', 'LEAN': 'LEAN'}.get(play.get('signal', ''), '---')
                lines.append(
                    f"  [{signal_icon}] {play.get('pick', '?')} "
                    f"({play.get('units', 0)}u) — "
                    f"{play.get('edge', '?')} edge, {play.get('confidence', '?')} confidence"
                )
        else:
            lines.append('  No actionable plays identified')

        # Bankroll
        bankroll = sections.get('bankroll', {})
        lines.append('')
        lines.append('BANKROLL')
        lines.append(f"  Today's exposure: {bankroll.get('today_exposure', 'N/A')}")

        # Alerts
        alerts = sections.get('alerts', [])
        lines.append('')
        lines.append('ALERTS')
        for alert in alerts:
            lines.append(f"  {alert}")

        # Market Intel
        market = sections.get('market_intel', [])
        lines.append('')
        lines.append('MARKET INTEL')
        for item in market[:5]:
            lines.append(f"  {item}")

        lines.append('')
        lines.append('=' * 55)

        return '\n'.join(lines)

    def _synthesize_with_llm(self, context: dict, fallback_sections: dict) -> dict:
        """
        Call LLM to produce a polished briefing.

        Falls back to deterministic sections if LLM unavailable.
        """
        system_prompt = self._load_system_prompt()

        user_message = json.dumps({
            'task': 'Generate the daily briefing for Colin',
            'briefing_date': self.target_date,
            'context': {
                'predictions': context.get('predictions'),
                'yesterday_results': context.get('yesterday_results'),
                'odds_intel': context.get('odds_intel'),
                'health_check': context.get('health_check'),
            },
        }, indent=2, default=str)

        response = self.call_llm(system_prompt, user_message, max_tokens=4096)

        if not response:
            return {
                'sections': fallback_sections,
                'formatted_text': self._format_briefing_text(fallback_sections),
                'reasoning': 'Generated using deterministic template (LLM unavailable).',
            }

        try:
            parsed = json.loads(response)
            if 'sections' in parsed or 'formatted_text' in parsed:
                # Ensure formatted_text exists
                if 'formatted_text' not in parsed:
                    parsed['formatted_text'] = self._format_briefing_text(
                        parsed.get('sections', fallback_sections)
                    )
                return parsed
            return {
                'sections': fallback_sections,
                'formatted_text': self._format_briefing_text(fallback_sections),
                'reasoning': 'LLM response missing required fields, using fallback.',
            }
        except json.JSONDecodeError:
            logger.warning(f"[{self.AGENT_NAME}] LLM returned invalid JSON, using fallback")
            return {
                'sections': fallback_sections,
                'formatted_text': self._format_briefing_text(fallback_sections),
                'reasoning': 'LLM returned invalid JSON, using template.',
            }

    def run(self) -> dict:
        """
        Core briefing generation logic.

        1. Gather context from all agent messages
        2. Build deterministic sections as fallback
        3. Call LLM for polished output
        4. Return structured briefing
        """
        logger.info(f"[{self.AGENT_NAME}] Generating briefing for {self.target_date}")

        # Step 1: Gather context
        context = self._gather_context()

        data_sources = []
        if context['predictions']:
            data_sources.append('predictions')
        if context['yesterday_results']:
            data_sources.append('yesterday_results')
        if context['odds_intel']:
            data_sources.append('odds_intel')
        if context['health_check']:
            data_sources.append('health_check')

        logger.info(f"[{self.AGENT_NAME}] Available data: {data_sources}")

        # Step 2: Build deterministic fallback sections
        fallback_sections = self._build_sections_from_context(context)

        # Step 3: Synthesize with LLM
        result = self._synthesize_with_llm(context, fallback_sections)

        return {
            'briefing_date': self.target_date,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'sections': result.get('sections', fallback_sections),
            'formatted_text': result.get('formatted_text', ''),
            'yesterday_record': context.get('yesterday_results'),
            'data_sources': data_sources,
            'reasoning': result.get('reasoning', ''),
        }

    def report(self, run_output: dict):
        """Send briefing_ready to all agents + push via Pushover."""
        self.send_message(
            recipient='all',
            event_type='briefing_ready',
            payload={
                'briefing_date': run_output.get('briefing_date'),
                'formatted_text': run_output.get('formatted_text', ''),
                'sections': run_output.get('sections', {}),
            },
            priority='normal',
        )

        # Push notification — failure never marks the run as failed
        try:
            from agents.core.notifications import send_briefing

            plays = run_output.get('sections', {}).get('today_plays', [])
            play_count = len(plays) if isinstance(plays, list) else 0

            send_briefing(
                formatted_text=run_output.get('formatted_text', ''),
                briefing_date=run_output.get('briefing_date', self.target_date),
                play_count=play_count,
            )
        except Exception as e:
            logger.warning(f"[{self.AGENT_NAME}] Push notification failed (non-fatal): {e}")
