"""
CLV Bridge — Connect prediction pipeline to BetTracker for CLV tracking.

Records BET/LEAN signals as TrackedBets so closing odds can be captured
and Closing Line Value computed. This is the #1 metric for long-term
model quality per CLAUDE.md.

Usage:
    from nba_betting.edge.clv_bridge import record_predictions_as_bets

    count = record_predictions_as_bets(all_player_props, "2026-02-23")
"""

import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy-init singleton to avoid repeated DB connections
_tracker_instance = None


def _get_tracker():
    """Get or create BetTracker singleton."""
    global _tracker_instance
    if _tracker_instance is None:
        from nba_betting.edge.bet_tracker import BetTracker
        db_path = os.path.join("data", "bet_tracking.db")
        os.makedirs("data", exist_ok=True)
        _tracker_instance = BetTracker(db_path=db_path)
    return _tracker_instance


def record_predictions_as_bets(
    predictions: list[dict],
    game_date: str,
) -> int:
    """
    Record BET and LEAN signal predictions as TrackedBets for CLV tracking.

    Only records predictions with signal = 'BET' or 'LEAN'.
    PASS and FADE signals are skipped.

    Args:
        predictions: List of prediction dicts from the pipeline
        game_date: Date string (YYYY-MM-DD)

    Returns:
        Number of bets recorded
    """
    from nba_betting.edge.bet_tracker import TrackedBet, BetType

    tracker = _get_tracker()
    recorded = 0

    for pred in predictions:
        signal = pred.get('signal', pred.get('bet_recommendation', 'PASS'))
        if signal not in ('BET', 'LEAN'):
            continue

        player = pred.get('player', '')
        stat = pred.get('stat', '')
        line = pred.get('line', 0)
        pick = pred.get('pick', 'OVER')
        american_odds = pred.get('american_odds', -110)
        over_prob = pred.get('over_prob', 0.5)
        model_prob = pred.get('model_probability', over_prob if pick == 'OVER' else 1.0 - over_prob)
        edge = pred.get('edge', 0)
        game = pred.get('game', '')

        # Generate unique bet ID
        bet_id = f"{game_date}_{player[:15]}_{stat}_{pick}".replace(' ', '_')

        try:
            bet = TrackedBet(
                bet_id=bet_id,
                placed_at=datetime.now(),
                sport="NBA",
                bet_type=BetType.PLAYER_PROP,
                event_name=game,
                event_date=datetime.strptime(game_date, "%Y-%m-%d"),
                selection=f"{player} {stat} {pick} {line}",
                odds=float(american_odds),
                stake=0.0,  # No actual stake — tracking for CLV only
                model_probability=model_prob,
                opening_odds=float(american_odds),
                notes=f"signal={signal} edge={edge:.1f}%",
                tags=[signal, stat.lower(), pick.lower()],
            )
            tracker.record_bet(bet)
            recorded += 1
        except Exception as e:
            logger.debug(f"Failed to record bet for {player} {stat}: {e}")
            continue

    return recorded


def update_closing_odds(bet_id: str, closing_odds: float) -> bool:
    """
    Update closing odds on a TrackedBet for CLV computation.

    Works with both PostgreSQL (tracked_bets) and SQLite (bets) backends
    depending on which the tracker is configured to use.

    Args:
        bet_id: The bet ID to update
        closing_odds: Closing American odds

    Returns:
        True if updated successfully
    """
    tracker = _get_tracker()

    if tracker._use_postgres:
        try:
            cur = tracker._pg_conn.cursor()
            cur.execute(
                "UPDATE tracked_bets SET closing_odds = %s WHERE bet_id = %s",
                (closing_odds, bet_id)
            )
            updated = cur.rowcount > 0
            cur.close()
            return updated
        except Exception as e:
            logger.error(f"Failed to update closing odds for {bet_id} (PG): {e}")
            return False
    else:
        import sqlite3
        try:
            conn = sqlite3.connect(tracker.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE bets SET closing_odds = ? WHERE bet_id = ?",
                (closing_odds, bet_id)
            )
            updated = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return updated
        except Exception as e:
            logger.error(f"Failed to update closing odds for {bet_id} (SQLite): {e}")
            return False


def update_closing_odds_for_date(game_date: str, closing_odds_map: dict[str, float]) -> int:
    """
    Batch update closing odds for all bets on a date.

    Args:
        game_date: Date string (YYYY-MM-DD)
        closing_odds_map: Mapping of bet_id -> closing_odds

    Returns:
        Number of bets updated
    """
    updated = 0
    for bet_id, odds in closing_odds_map.items():
        if update_closing_odds(bet_id, odds):
            updated += 1
    return updated
