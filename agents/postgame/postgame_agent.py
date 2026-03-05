"""
Post-Game Analysis Agent

Reviews every prediction against actual results after games conclude.
Wraps calibration_tracker/ with Claude-powered reasoning for root cause
analysis of misses and pattern detection.

Trigger: After all games complete (~1 AM ET).
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from agents.core.agent_base import AgentBase
from nba_betting.constants import PROP_STD_DEVS, DEFAULT_PROP_STD_DEV as DEFAULT_STD_DEV

logger = logging.getLogger(__name__)

# Standard deviations for identifying large misses (per prop type)
PROP_STD_DEVS = {
    'points': 6.5,
    'rebounds': 3.1,
    'assists': 2.2,
    'threes': 1.6,
    'pra': 8.5,
}
DEFAULT_STD_DEV = 5.0


class PostGameAnalysisAgent(AgentBase):
    """
    Post-Game Analysis Agent.

    Wraps CalibrationService nightly job with LLM reasoning
    to analyze misses, detect patterns, and generate model feedback.
    """

    AGENT_NAME = 'postgame'
    DAILY_TOKEN_BUDGET = 60_000
    MAX_EXECUTION_SECONDS = 600
    MAX_MISS_ANALYSES = 10
    MIN_PATTERN_SAMPLES = 30

    def __init__(self, target_date: str = None, **kwargs):
        super().__init__(**kwargs)
        if target_date:
            self.target_date = target_date
        else:
            self.target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        self._calibration_service = None

    def _get_calibration_service(self):
        """Lazy-init CalibrationService."""
        if self._calibration_service is None:
            from calibration_tracker import CalibrationService
            self._calibration_service = CalibrationService()
        return self._calibration_service

    def _load_system_prompt(self) -> str:
        """Load the version-controlled system prompt."""
        import os
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'prompts', 'postgame.md'
        )
        try:
            with open(prompt_path) as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"System prompt not found at {prompt_path}, using default")
            return self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return (
            "You are the Post-Game Analysis Agent for an NBA betting model. "
            "Analyze prediction misses and classify root causes. "
            "Categories: data_issue, model_limitation, feature_gap, normal_variance. "
            "Be honest — not every loss is a bug. Distinguish bad predictions from bad luck. "
            "Output valid JSON with root_cause, explanation, and recommended_action."
        )

    def _identify_large_misses(self, predictions_with_outcomes: list) -> list:
        """
        Find predictions that missed by > 2 standard deviations.

        Returns list of miss records sorted by miss magnitude (largest first).
        """
        large_misses = []

        for record in predictions_with_outcomes:
            prop_type = record.get('prop_type', '')
            predicted = record.get('predicted_value')
            actual = record.get('actual_value')

            if predicted is None or actual is None:
                continue

            std_dev = PROP_STD_DEVS.get(prop_type, DEFAULT_STD_DEV)
            error = abs(predicted - actual)
            threshold = 2 * std_dev

            if error > threshold:
                large_misses.append({
                    **record,
                    'error_magnitude': error,
                    'std_devs_off': round(error / std_dev, 2),
                    'threshold': threshold,
                })

        # Sort by magnitude (largest first)
        large_misses.sort(key=lambda x: x['error_magnitude'], reverse=True)
        return large_misses

    def _analyze_miss_with_llm(self, miss: dict) -> dict:
        """
        Call LLM to analyze why a specific prediction missed.

        Falls back to deterministic classification on LLM failure.
        """
        system_prompt = self._load_system_prompt()

        user_message = json.dumps({
            'task': 'Analyze why this prediction missed and classify the root cause',
            'prediction': {
                'player_name': miss.get('player_name', 'Unknown'),
                'prop_type': miss.get('prop_type', ''),
                'predicted_value': miss.get('predicted_value'),
                'actual_value': miss.get('actual_value'),
                'prop_line': miss.get('prop_line'),
                'predicted_minutes': miss.get('minutes_predicted'),
                'actual_minutes': miss.get('actual_minutes'),
                'confidence': miss.get('confidence'),
                'is_home': miss.get('is_home'),
                'opponent': miss.get('opponent', ''),
                'spread': miss.get('spread'),
                'error_std_devs': miss.get('std_devs_off'),
            },
        }, indent=2, default=str)

        response = self.call_llm(system_prompt, user_message, max_tokens=1024)

        if not response:
            return self._fallback_miss_analysis(miss)

        try:
            parsed = json.loads(response)
            required = ['root_cause', 'explanation']
            if all(k in parsed for k in required):
                # Validate root_cause category
                valid_causes = ['data_issue', 'model_limitation', 'feature_gap', 'normal_variance']
                if parsed['root_cause'] not in valid_causes:
                    parsed['root_cause'] = 'normal_variance'
                return parsed
            return self._fallback_miss_analysis(miss)
        except json.JSONDecodeError:
            return self._fallback_miss_analysis(miss)

    def _fallback_miss_analysis(self, miss: dict) -> dict:
        """Deterministic miss classification when LLM is unavailable."""
        predicted_minutes = miss.get('minutes_predicted')
        actual_minutes = miss.get('actual_minutes')

        # Simple heuristic: if minutes were way off, it's likely a data issue
        if predicted_minutes and actual_minutes:
            minutes_diff = abs(predicted_minutes - actual_minutes)
            if minutes_diff > 10:
                return {
                    'root_cause': 'data_issue',
                    'explanation': f"Minutes prediction was off by {minutes_diff:.0f} min "
                                   f"(predicted {predicted_minutes:.0f}, actual {actual_minutes:.0f}). "
                                   f"Likely late scratch or unexpected rotation change.",
                    'recommended_action': 'Improve minutes prediction or injury monitoring timeliness.',
                }

        std_devs = miss.get('std_devs_off', 0)
        if std_devs > 4:
            return {
                'root_cause': 'model_limitation',
                'explanation': f"Prediction was {std_devs:.1f} std devs off. "
                               f"Extreme miss suggests model doesn't capture this scenario well.",
                'recommended_action': 'Review feature coverage for this game type.',
            }

        return {
            'root_cause': 'normal_variance',
            'explanation': f"Miss of {std_devs:.1f} std devs. Within range of normal variance "
                           f"for NBA player performance.",
            'recommended_action': 'No action — normal variance.',
        }

    def _extract_pattern_flags(self, bias_report_dict: dict) -> list:
        """Extract significant patterns from bias analysis (30+ sample minimum)."""
        patterns = []

        dimensions_to_check = [
            ('by_prop_type', 'Prop type'),
            ('by_position', 'Position'),
            ('by_game_type', 'Game type'),
            ('by_player_tier', 'Player tier'),
        ]

        for dim_key, dim_label in dimensions_to_check:
            dimension_data = bias_report_dict.get(dim_key, {})
            for value, analysis in dimension_data.items():
                sample_size = analysis.get('sample_size', 0)
                if sample_size < self.MIN_PATTERN_SAMPLES:
                    continue

                bias = analysis.get('bias', 0)
                hit_rate = analysis.get('hit_rate', 0.5)

                # Flag significant biases
                if abs(bias) > 2.0:
                    direction = 'over-predicting' if bias > 0 else 'under-predicting'
                    patterns.append({
                        'dimension': dim_label,
                        'value': value,
                        'pattern': f"Model is {direction} {dim_label.lower()} '{value}' "
                                   f"by {abs(bias):.1f} points (n={sample_size}, "
                                   f"hit_rate={hit_rate:.1%})",
                        'severity': 'high' if abs(bias) > 4.0 else 'medium',
                        'sample_size': sample_size,
                    })

        return patterns

    def _settle_paper_trades(self) -> dict:
        """Settle paper trades for target_date and the day before.

        Settles two days to handle late-finishing games (e.g., overtime,
        West Coast games ending after midnight ET).

        Returns:
            Dict with settlement counts per date.
        """
        from datetime import date as _date

        results = {}
        try:
            from nba_betting.settle_trades import settle_date

            # Settle target_date and the day before
            target = datetime.strptime(self.target_date, '%Y-%m-%d').date()
            dates_to_settle = [
                (target - timedelta(days=1)).isoformat(),
                target.isoformat(),
            ]

            for d in dates_to_settle:
                try:
                    count = settle_date(d)
                    results[d] = count
                    if count > 0:
                        logger.info(f"[{self.AGENT_NAME}] Settled {count} paper trades for {d}")
                except Exception as e:
                    logger.warning(f"[{self.AGENT_NAME}] Settlement failed for {d}: {e}")
                    results[d] = 0

        except ImportError:
            logger.warning(f"[{self.AGENT_NAME}] settle_trades module not available")
        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] Settlement error: {e}")

        return results

    def run(self) -> dict:
        """
        Core post-game analysis.

        0. Settle paper trades (grade predictions against actual outcomes)
        1. Run deterministic nightly job (outcome matching, adjustments)
        2. Identify large misses
        3. Analyze top misses with LLM
        4. Detect patterns from bias analysis
        5. Return structured analysis
        """
        logger.info(f"[{self.AGENT_NAME}] Running for date: {self.target_date}")

        # Step 0: Settle paper trades before analysis
        settlement_results = self._settle_paper_trades()

        service = self._get_calibration_service()

        # Step 1: Run deterministic nightly job
        logger.info(f"[{self.AGENT_NAME}] Running nightly calibration job...")
        try:
            nightly_results = service.run_nightly_job(game_date=self.target_date)
        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] Nightly job failed: {e}")
            nightly_results = {'steps': {}, 'error': str(e)}

        # Step 2: Get predictions with outcomes for this date
        try:
            predictions = service.db.get_predictions_with_outcomes(
                start_date=self.target_date,
                end_date=self.target_date,
            )
        except Exception as e:
            logger.warning(f"[{self.AGENT_NAME}] Failed to get predictions: {e}")
            predictions = []

        if not predictions:
            logger.info(f"[{self.AGENT_NAME}] No predictions found for {self.target_date}")
            return {
                'slate_date': self.target_date,
                'analyzed_at': datetime.now(timezone.utc).isoformat(),
                'results_summary': {
                    'total_bets': 0, 'wins': 0, 'losses': 0,
                    'roi_today': 'N/A', 'clv_average': 'N/A',
                },
                'miss_analysis': [],
                'pattern_flags': [],
                'model_feedback': [],
                'reasoning': f"No predictions found for {self.target_date}",
            }

        # Compute summary
        total = len(predictions)
        wins = sum(1 for p in predictions if p.get('hit') == 1)
        losses = total - wins
        clv_values = [p.get('clv', 0) for p in predictions if p.get('clv') is not None]
        clv_avg = sum(clv_values) / len(clv_values) if clv_values else 0

        # Step 3: Identify large misses
        large_misses = self._identify_large_misses(predictions)
        logger.info(f"[{self.AGENT_NAME}] Found {len(large_misses)} large misses")

        # Step 4: Analyze top misses with LLM (capped at MAX_MISS_ANALYSES)
        miss_analyses = []
        for miss in large_misses[:self.MAX_MISS_ANALYSES]:
            analysis = self._analyze_miss_with_llm(miss)
            miss_analyses.append({
                'prediction_id': miss.get('id', miss.get('prediction_id')),
                'player_name': miss.get('player_name', 'Unknown'),
                'prop_type': miss.get('prop_type', ''),
                'predicted': miss.get('predicted_value'),
                'actual': miss.get('actual_value'),
                'miss_magnitude': 'large',
                'std_devs_off': miss.get('std_devs_off'),
                'root_cause': analysis.get('root_cause', 'normal_variance'),
                'explanation': analysis.get('explanation', ''),
                'recommended_action': analysis.get('recommended_action', ''),
            })

        # Step 5: Pattern detection from bias analysis
        pattern_flags = []
        try:
            bias_report = service.analyze_biases(
                start_date=self.target_date,
                end_date=self.target_date,
            )
            bias_dict = bias_report.to_dict() if hasattr(bias_report, 'to_dict') else {}
            pattern_flags = self._extract_pattern_flags(bias_dict)
        except Exception as e:
            logger.warning(f"[{self.AGENT_NAME}] Bias analysis failed: {e}")

        # Step 6: Model feedback
        model_feedback = []
        root_cause_counts = {}
        for ma in miss_analyses:
            rc = ma['root_cause']
            root_cause_counts[rc] = root_cause_counts.get(rc, 0) + 1

        for rc, count in root_cause_counts.items():
            if rc != 'normal_variance' and count >= 2:
                model_feedback.append({
                    'category': rc,
                    'count': count,
                    'recommendation': f"{count} misses classified as '{rc}' — investigate systematically.",
                })

        roi_today = f"+{(wins/total - 0.524) * 100:.1f}%" if total > 0 else 'N/A'

        return {
            'slate_date': self.target_date,
            'analyzed_at': datetime.now(timezone.utc).isoformat(),
            'results_summary': {
                'total_bets': total,
                'wins': wins,
                'losses': losses,
                'roi_today': roi_today,
                'clv_average': f"{clv_avg:+.2f}",
            },
            'miss_analysis': miss_analyses,
            'pattern_flags': pattern_flags,
            'model_feedback': model_feedback,
            'settlement_results': settlement_results,
            'nightly_job_results': nightly_results.get('steps', {}),
            'reasoning': (
                f"Analyzed {total} predictions for {self.target_date}. "
                f"Record: {wins}-{losses}. "
                f"{len(large_misses)} large misses found, {len(miss_analyses)} analyzed. "
                f"{len(pattern_flags)} patterns flagged."
            ),
        }

    def report(self, run_output: dict):
        """Send results_analyzed messages to watchdog and briefing."""
        # Send to future Watchdog agent
        self.send_message(
            recipient='watchdog',
            event_type='results_analyzed',
            payload={
                'slate_date': run_output.get('slate_date'),
                'results_summary': run_output.get('results_summary', {}),
                'miss_analysis': run_output.get('miss_analysis', []),
                'model_feedback': run_output.get('model_feedback', []),
            },
            priority='normal',
        )

        # Send to future Briefing agent
        self.send_message(
            recipient='briefing',
            event_type='results_analyzed',
            payload={
                'slate_date': run_output.get('slate_date'),
                'results_summary': run_output.get('results_summary', {}),
                'pattern_flags': run_output.get('pattern_flags', []),
            },
            priority='normal',
        )
