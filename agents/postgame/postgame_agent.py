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

    DRIFT_WARN_THRESHOLD = 1.0  # points of systematic bias that triggers a warning

    def _check_calibration_drift(self, predictions: list) -> list:
        """Check whether predicted vs actual bias has drifted beyond threshold.

        Computes mean signed error (predicted - actual) per prop type using the
        current date's predictions. Logs a WARNING when any prop type exceeds
        DRIFT_WARN_THRESHOLD so that the on-call pipeline can react before the
        bias compounds across multiple days.

        Args:
            predictions: List of prediction dicts with 'prop_type',
                'predicted_value', and 'actual_value' keys.

        Returns:
            List of drift alert dicts (empty when everything is in range).
        """
        from collections import defaultdict

        buckets: dict = defaultdict(list)
        for rec in predictions:
            prop_type = rec.get('prop_type', '')
            predicted = rec.get('predicted_value')
            actual = rec.get('actual_value')
            if predicted is None or actual is None or not prop_type:
                continue
            buckets[prop_type].append(predicted - actual)

        alerts = []
        for prop_type, errors in buckets.items():
            if len(errors) < 5:
                continue  # too few samples to be meaningful
            mean_bias = sum(errors) / len(errors)
            if abs(mean_bias) > self.DRIFT_WARN_THRESHOLD:
                direction = 'over-predicting' if mean_bias > 0 else 'under-predicting'
                logger.warning(
                    "[%s] CALIBRATION DRIFT: %s %s by %.2f points "
                    "(n=%d, threshold=%.1f) — consider triggering recalibration",
                    self.AGENT_NAME, prop_type, direction, abs(mean_bias),
                    len(errors), self.DRIFT_WARN_THRESHOLD,
                )
                alerts.append({
                    'prop_type': prop_type,
                    'mean_bias': round(mean_bias, 3),
                    'n_samples': len(errors),
                    'direction': direction,
                })
            else:
                logger.debug(
                    "[%s] calibration OK: %s bias=%.2f (n=%d)",
                    self.AGENT_NAME, prop_type, mean_bias, len(errors),
                )

        return alerts

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

    def _settle_tracked_bets(self) -> dict:
        """Settle tracked_bets by matching against settled paper_trades.

        After paper_trades are graded, transfers the results into tracked_bets
        so CLV tracking and bet-level P&L are complete.

        Matches on: event_date + player name present in selection + prop stat.

        Returns:
            Dict with 'settled', 'unmatched', 'errors' counts.
        """
        results = {'settled': 0, 'unmatched': 0, 'errors': 0}

        try:
            from nba_betting.edge.bet_tracker import BetTracker, BetStatus
            from nba_betting.paper_trading import PaperTrader
            import os

            trader = PaperTrader()
            tracker = BetTracker(db_path=os.path.join("data", "bet_tracking.db"))

            target = datetime.strptime(self.target_date, '%Y-%m-%d').date()
            dates_to_settle = [
                (target - timedelta(days=1)).isoformat(),
                target.isoformat(),
            ]

            for game_date in dates_to_settle:
                # Get settled paper trades for this date
                daily_report = trader.get_daily_report(game_date)
                if not daily_report['predictions']:
                    continue

                # Build lookup: (player_lower, prop_type_lower, direction_lower) -> result
                trade_lookup = {}
                for trade in daily_report['predictions']:
                    if trade.get('result') is None:
                        continue
                    key = (
                        trade['player_name'].lower(),
                        trade['prop_type'].lower(),
                        (trade.get('direction') or 'over').lower(),
                    )
                    trade_lookup[key] = {
                        'result': trade['result'],   # 'hit', 'miss', 'push'
                        'actual_value': trade.get('actual_value'),
                    }

                if not trade_lookup:
                    continue

                # Get all pending tracked bets and filter by event_date
                try:
                    pending_bets = tracker.get_pending_bets()
                except Exception as e:
                    logger.warning(
                        f"[{self.AGENT_NAME}] Could not fetch pending bets for {game_date}: {e}"
                    )
                    continue

                for bet in pending_bets:
                    # Filter to bets whose event_date matches this game_date
                    if bet.event_date is None:
                        continue
                    if bet.event_date.strftime('%Y-%m-%d') != game_date:
                        continue

                    # Parse selection: "{player} {stat} {pick} {line}"
                    selection = bet.selection or ''
                    parts = selection.split()

                    # Find OVER/UNDER keyword and its index
                    pick = None
                    pick_idx = None
                    for i, part in enumerate(parts):
                        if part.upper() in ('OVER', 'UNDER'):
                            pick = part.lower()
                            pick_idx = i
                            break

                    if pick is None or pick_idx < 2:
                        results['unmatched'] += 1
                        continue

                    # stat is the word immediately before the direction keyword
                    stat = parts[pick_idx - 1].lower()
                    player_name = ' '.join(parts[:pick_idx - 1]).lower()

                    # Exact match first
                    matched = trade_lookup.get((player_name, stat, pick))

                    # Fuzzy fallback: check if the paper trade player_name is
                    # a substring of the parsed player_name (handles truncation)
                    if matched is None:
                        for (tp, ts, td), tdata in trade_lookup.items():
                            if ts == stat and td == pick and (
                                tp in player_name or player_name in tp
                            ):
                                matched = tdata
                                break

                    if matched is None:
                        results['unmatched'] += 1
                        continue

                    trade_result = matched['result']
                    actual_value = matched.get('actual_value')

                    if trade_result == 'push':
                        status = BetStatus.PUSH
                    elif trade_result == 'hit':
                        status = BetStatus.WON
                    else:
                        status = BetStatus.LOST

                    try:
                        tracker.settle_bet(
                            bet_id=bet.bet_id,
                            status=status,
                            actual_result=str(actual_value) if actual_value is not None else trade_result,
                        )
                        results['settled'] += 1
                    except Exception as e:
                        logger.warning(
                            f"[{self.AGENT_NAME}] Failed to settle bet {bet.bet_id}: {e}"
                        )
                        results['errors'] += 1

        except ImportError as e:
            logger.warning(f"[{self.AGENT_NAME}] tracked_bets settlement skipped (import error): {e}")
        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] _settle_tracked_bets error: {e}")

        if results['settled'] > 0 or results['errors'] > 0:
            logger.info(
                f"[{self.AGENT_NAME}] tracked_bets: settled={results['settled']}, "
                f"unmatched={results['unmatched']}, errors={results['errors']}"
            )
        return results

    def _update_bankroll_state(self, game_date: str) -> dict:
        """Update bankroll_state and bankroll_daily_pl with today's P&L.

        Reads settled paper_trade P&L for should_bet=True bets, then persists
        the updated bankroll balance to the DB so BankrollManager has accurate
        state on the next run.

        Args:
            game_date: Date string (YYYY-MM-DD) to aggregate P&L for.

        Returns:
            Dict with pnl, bankroll_before, bankroll_after, num_bets, wins, losses.
            Empty dict if nothing to update or DB unavailable.
        """
        try:
            from agents.core.connections import get_postgres_connection
            conn = get_postgres_connection()
            if conn is None:
                logger.debug(f"[{self.AGENT_NAME}] No PostgreSQL connection; skipping bankroll update")
                return {}

            cur = conn.cursor()

            # Aggregate P&L from settled paper_trades for should_bet=True bets
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE result = 'hit')  AS wins,
                    COUNT(*) FILTER (WHERE result = 'miss') AS losses,
                    COUNT(*) FILTER (WHERE result IN ('hit', 'miss')) AS total,
                    COALESCE(SUM(profit_loss), 0)           AS total_pnl,
                    COALESCE(SUM(bet_size), 0)              AS total_staked
                FROM paper_trades
                WHERE game_date = %s
                  AND should_bet = TRUE
                  AND result IS NOT NULL
            """, (game_date,))
            row = cur.fetchone()

            if row is None or (row[2] or 0) == 0:
                cur.close()
                return {}

            wins, losses, total, total_pnl, total_staked = (
                int(row[0] or 0),
                int(row[1] or 0),
                int(row[2] or 0),
                float(row[3]),
                float(row[4]),
            )

            # Load the most recent bankroll balance
            cur.execute(
                "SELECT amount FROM bankroll_state ORDER BY updated_at DESC LIMIT 1"
            )
            bankroll_row = cur.fetchone()
            current_bankroll = float(bankroll_row[0]) if bankroll_row else 1000.0
            new_bankroll = current_bankroll + total_pnl

            # Append a new bankroll_state snapshot
            cur.execute(
                "INSERT INTO bankroll_state (amount, updated_at) VALUES (%s, NOW())",
                (new_bankroll,),
            )

            # Upsert daily P&L summary (idempotent — re-running is safe)
            cur.execute("""
                INSERT INTO bankroll_daily_pl
                    (date, starting_bankroll, ending_bankroll, total_staked,
                     total_returned, profit_loss, num_bets, num_wins, num_losses)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
                    ending_bankroll = EXCLUDED.ending_bankroll,
                    total_staked    = EXCLUDED.total_staked,
                    total_returned  = EXCLUDED.total_returned,
                    profit_loss     = EXCLUDED.profit_loss,
                    num_bets        = EXCLUDED.num_bets,
                    num_wins        = EXCLUDED.num_wins,
                    num_losses      = EXCLUDED.num_losses
            """, (
                game_date,
                current_bankroll,
                new_bankroll,
                total_staked,
                total_staked + total_pnl,   # total_returned
                total_pnl,
                total,
                wins,
                losses,
            ))

            conn.commit()
            cur.close()

            logger.info(
                f"[{self.AGENT_NAME}] Bankroll updated for {game_date}: "
                f"${current_bankroll:.2f} → ${new_bankroll:.2f} "
                f"(P&L: ${total_pnl:+.2f}, {wins}W-{losses}L)"
            )

            return {
                'date': game_date,
                'pnl': total_pnl,
                'bankroll_before': current_bankroll,
                'bankroll_after': new_bankroll,
                'num_bets': total,
                'wins': wins,
                'losses': losses,
            }

        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] Bankroll update failed for {game_date}: {e}")
            return {}

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

        # Step 0: Settle paper trades, then sync tracked_bets and bankroll
        settlement_results = self._settle_paper_trades()
        tracked_bet_results = self._settle_tracked_bets()
        bankroll_update = self._update_bankroll_state(self.target_date)

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

        # Check for calibration drift and log warnings
        drift_alerts = self._check_calibration_drift(predictions)

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
            'drift_alerts': drift_alerts,
            'settlement_results': settlement_results,
            'tracked_bets_settled': tracked_bet_results,
            'bankroll_update': bankroll_update,
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
