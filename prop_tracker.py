"""
Prop Prediction Tracker

Tracks player prop predictions vs actual outcomes for calibration and performance analysis.
Stores predictions before games and settles them after games complete.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid


class PropTracker:
    """Track and analyze player prop prediction performance."""

    def __init__(self, db_path: str = None):
        """Initialize prop tracker with SQLite database.

        Args:
            db_path: Path to SQLite database file. Defaults to prop_predictions.db
        """
        if db_path is None:
            db_path = str(Path(__file__).parent / "prop_predictions.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prop_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    game_id TEXT,
                    game_date TEXT,
                    player_id INTEGER,
                    player_name TEXT,
                    team_abbrev TEXT,
                    opponent_abbrev TEXT,
                    prop_type TEXT,
                    predicted_value REAL,
                    market_line REAL,
                    pick TEXT,
                    edge_pct REAL,
                    confidence REAL,
                    opp_def_rating REAL,
                    opp_adjustment REAL,
                    actual_value REAL,
                    hit INTEGER,
                    created_at TEXT,
                    settled_at TEXT,
                    is_settled INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_game_date ON prop_predictions(game_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_player_id ON prop_predictions(player_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_is_settled ON prop_predictions(is_settled)
            """)
            conn.commit()

    def record_prediction(
        self,
        game_id: str,
        game_date: str,
        player_id: int,
        player_name: str,
        team_abbrev: str,
        opponent_abbrev: str,
        prop_type: str,
        predicted_value: float,
        market_line: float,
        pick: str,
        edge_pct: float,
        confidence: float,
        opp_def_rating: float = None,
        opp_adjustment: float = None,
    ) -> str:
        """Record a new prop prediction.

        Args:
            game_id: Unique game identifier
            game_date: Date of the game (YYYY-MM-DD)
            player_id: Player ID
            player_name: Player's full name
            team_abbrev: Player's team abbreviation
            opponent_abbrev: Opponent team abbreviation
            prop_type: Type of prop (points, rebounds, assists, 3pm, pra)
            predicted_value: Model's predicted value
            market_line: Betting line
            pick: OVER, UNDER, or -
            edge_pct: Calculated edge percentage
            confidence: Confidence score (0-100)
            opp_def_rating: Opponent defensive rating (optional)
            opp_adjustment: Opponent adjustment applied (optional)

        Returns:
            prediction_id: Unique ID for this prediction
        """
        prediction_id = str(uuid.uuid4())[:8]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO prop_predictions (
                    prediction_id, game_id, game_date, player_id, player_name,
                    team_abbrev, opponent_abbrev, prop_type, predicted_value,
                    market_line, pick, edge_pct, confidence, opp_def_rating,
                    opp_adjustment, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id, game_id, game_date, player_id, player_name,
                team_abbrev, opponent_abbrev, prop_type, predicted_value,
                market_line, pick, edge_pct, confidence, opp_def_rating,
                opp_adjustment, datetime.now().isoformat()
            ))
            conn.commit()

        return prediction_id

    def settle_prediction(
        self,
        prediction_id: str,
        actual_value: float,
    ) -> bool:
        """Settle a prediction with actual outcome.

        Args:
            prediction_id: ID of the prediction to settle
            actual_value: Actual stat value from the game

        Returns:
            True if prediction was found and settled
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get the prediction
            cursor = conn.execute(
                "SELECT market_line, pick FROM prop_predictions WHERE prediction_id = ?",
                (prediction_id,)
            )
            row = cursor.fetchone()
            if not row:
                return False

            market_line, pick = row

            # Determine if pick hit
            hit = 0
            if pick == "OVER" and actual_value > market_line:
                hit = 1
            elif pick == "UNDER" and actual_value < market_line:
                hit = 1
            elif pick == "-":
                hit = -1  # No pick made

            conn.execute("""
                UPDATE prop_predictions
                SET actual_value = ?, hit = ?, settled_at = ?, is_settled = 1
                WHERE prediction_id = ?
            """, (actual_value, hit, datetime.now().isoformat(), prediction_id))
            conn.commit()

        return True

    def settle_game_predictions(
        self,
        game_id: str,
        player_stats: Dict[int, Dict],
    ) -> int:
        """Settle all predictions for a completed game.

        Args:
            game_id: Game ID to settle
            player_stats: Dict mapping player_id to their stats {pts, reb, ast, fg3m}

        Returns:
            Number of predictions settled
        """
        settled = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT prediction_id, player_id, prop_type
                   FROM prop_predictions
                   WHERE game_id = ? AND is_settled = 0""",
                (game_id,)
            )
            predictions = cursor.fetchall()

            for pred_id, player_id, prop_type in predictions:
                stats = player_stats.get(player_id, {})
                if not stats:
                    continue

                # Map prop type to stat key
                stat_map = {
                    "points": "pts",
                    "rebounds": "reb",
                    "assists": "ast",
                    "3pm": "fg3m",
                    "pra": None,  # Calculated
                }

                if prop_type == "pra":
                    actual = (
                        (stats.get("pts", 0) or 0) +
                        (stats.get("reb", 0) or 0) +
                        (stats.get("ast", 0) or 0)
                    )
                else:
                    stat_key = stat_map.get(prop_type)
                    if stat_key:
                        actual = stats.get(stat_key, 0) or 0
                    else:
                        continue

                if self.settle_prediction(pred_id, actual):
                    settled += 1

        return settled

    def get_unsettled_predictions(
        self,
        game_date: str = None,
    ) -> List[Dict]:
        """Get all unsettled predictions.

        Args:
            game_date: Optional filter by game date

        Returns:
            List of unsettled prediction records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if game_date:
                cursor = conn.execute(
                    "SELECT * FROM prop_predictions WHERE is_settled = 0 AND game_date = ?",
                    (game_date,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM prop_predictions WHERE is_settled = 0"
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_predictions_for_game(self, game_id: str) -> List[Dict]:
        """Get all predictions for a specific game.

        Args:
            game_id: Game ID to look up

        Returns:
            List of prediction dicts with player_id, prop_type, predicted_value, etc.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT player_id, player_name, team_abbrev, prop_type,
                       predicted_value, market_line, pick, confidence, edge_pct,
                       actual_value, hit, is_settled
                FROM prop_predictions
                WHERE game_id = ?
            """, (game_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_performance_summary(
        self,
        days: int = 30,
        min_confidence: float = 0,
    ) -> Dict:
        """Get performance summary for settled predictions.

        Args:
            days: Number of days to look back
            min_confidence: Minimum confidence filter

        Returns:
            Performance metrics dictionary
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            # Overall stats
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN hit = 0 THEN 1 ELSE 0 END) as losses,
                    AVG(confidence) as avg_confidence,
                    AVG(edge_pct) as avg_edge
                FROM prop_predictions
                WHERE is_settled = 1 AND hit >= 0
                  AND game_date >= ?
                  AND confidence >= ?
            """, (cutoff_date, min_confidence))
            overall = cursor.fetchone()

            # By prop type
            cursor = conn.execute("""
                SELECT
                    prop_type,
                    COUNT(*) as total,
                    SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as wins,
                    AVG(confidence) as avg_confidence
                FROM prop_predictions
                WHERE is_settled = 1 AND hit >= 0
                  AND game_date >= ?
                  AND confidence >= ?
                GROUP BY prop_type
            """, (cutoff_date, min_confidence))
            by_prop = {row[0]: {"total": row[1], "wins": row[2], "avg_conf": row[3]}
                       for row in cursor.fetchall()}

            # By confidence bucket
            cursor = conn.execute("""
                SELECT
                    CASE
                        WHEN confidence >= 75 THEN 'high'
                        WHEN confidence >= 60 THEN 'medium'
                        ELSE 'low'
                    END as bucket,
                    COUNT(*) as total,
                    SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as wins
                FROM prop_predictions
                WHERE is_settled = 1 AND hit >= 0
                  AND game_date >= ?
                  AND confidence >= ?
                GROUP BY bucket
            """, (cutoff_date, min_confidence))
            by_confidence = {row[0]: {"total": row[1], "wins": row[2],
                                      "win_rate": row[2] / row[1] if row[1] > 0 else 0}
                            for row in cursor.fetchall()}

        total = overall[0] or 0
        wins = overall[1] or 0
        losses = overall[2] or 0

        return {
            "period_days": days,
            "total_predictions": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0,
            "avg_confidence": overall[3] or 0,
            "avg_edge": overall[4] or 0,
            "by_prop_type": by_prop,
            "by_confidence": by_confidence,
        }

    def get_calibration_data(self, days: int = 30) -> List[Dict]:
        """Get calibration data - confidence vs actual win rate.

        Args:
            days: Number of days to analyze

        Returns:
            List of calibration buckets
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    CAST(confidence / 10 AS INTEGER) * 10 as conf_bucket,
                    COUNT(*) as total,
                    SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as wins,
                    AVG(confidence) as avg_confidence
                FROM prop_predictions
                WHERE is_settled = 1 AND hit >= 0
                  AND game_date >= ?
                GROUP BY conf_bucket
                ORDER BY conf_bucket
            """, (cutoff_date,))

            return [{
                "confidence_bucket": f"{row[0]}-{row[0]+9}",
                "predicted_win_rate": (row[3] or 50) / 100,  # Convert confidence to rate
                "actual_win_rate": row[2] / row[1] if row[1] > 0 else 0,
                "total": row[1],
                "wins": row[2],
            } for row in cursor.fetchall()]

    def print_performance_report(self, days: int = 30):
        """Print a formatted performance report."""
        summary = self.get_performance_summary(days)
        calibration = self.get_calibration_data(days)

        print(f"\n{'='*60}")
        print(f"PROP PREDICTION PERFORMANCE REPORT - Last {days} Days")
        print(f"{'='*60}")

        print(f"\nOverall Performance:")
        print(f"  Total Predictions: {summary['total_predictions']}")
        print(f"  Wins: {summary['wins']}")
        print(f"  Losses: {summary['losses']}")
        print(f"  Win Rate: {summary['win_rate']*100:.1f}%")
        print(f"  Avg Confidence: {summary['avg_confidence']:.1f}")
        print(f"  Avg Edge: {summary['avg_edge']:.1f}%")

        print(f"\nBy Prop Type:")
        for prop_type, stats in summary['by_prop_type'].items():
            win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {prop_type.upper()}: {stats['wins']}/{stats['total']} ({win_rate*100:.1f}%)")

        print(f"\nBy Confidence Level:")
        for level, stats in summary['by_confidence'].items():
            print(f"  {level.upper()}: {stats['wins']}/{stats['total']} ({stats['win_rate']*100:.1f}%)")

        print(f"\nCalibration (Predicted vs Actual Win Rate):")
        print(f"  {'Confidence':>12} | {'Predicted':>10} | {'Actual':>10} | {'N':>6}")
        print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
        for bucket in calibration:
            print(f"  {bucket['confidence_bucket']:>12} | "
                  f"{bucket['predicted_win_rate']*100:>9.1f}% | "
                  f"{bucket['actual_win_rate']*100:>9.1f}% | "
                  f"{bucket['total']:>6}")

        print(f"\n{'='*60}\n")


    def get_player_performance(self, min_predictions: int = 20, days: int = 60) -> Dict:
        """Get player-level prediction performance for blacklist/whitelist.

        Players with <30% win rate on 20+ predictions should be blacklisted.
        Players with >60% win rate on 20+ predictions can be whitelisted.

        Args:
            min_predictions: Minimum predictions to include player
            days: Number of days to look back

        Returns:
            Dict with blacklist and whitelist player IDs
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    player_id,
                    player_name,
                    COUNT(*) as total,
                    SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as wins
                FROM prop_predictions
                WHERE is_settled = 1 AND hit >= 0
                  AND game_date >= ?
                GROUP BY player_id, player_name
                HAVING COUNT(*) >= ?
                ORDER BY total DESC
            """, (cutoff_date, min_predictions))

            blacklist = []  # Players with <30% win rate
            whitelist = []  # Players with >60% win rate
            all_players = []

            for row in cursor.fetchall():
                player_id, player_name, total, wins = row
                win_rate = wins / total if total > 0 else 0

                player_data = {
                    'player_id': player_id,
                    'player_name': player_name,
                    'total': total,
                    'wins': wins,
                    'win_rate': win_rate,
                }
                all_players.append(player_data)

                # Blacklist: consistently bad predictions
                if win_rate < 0.30:
                    blacklist.append(player_id)

                # Whitelist: consistently good predictions
                elif win_rate > 0.60:
                    whitelist.append(player_id)

            return {
                'blacklist': blacklist,
                'whitelist': whitelist,
                'all_players': all_players,
                'blacklist_count': len(blacklist),
                'whitelist_count': len(whitelist),
            }

    def get_blacklisted_players(self, min_predictions: int = 20, days: int = 60) -> List[int]:
        """Get list of player IDs that should be blacklisted (skipped).

        Args:
            min_predictions: Minimum predictions to qualify
            days: Number of days to look back

        Returns:
            List of player IDs to skip
        """
        performance = self.get_player_performance(min_predictions, days)
        return performance['blacklist']


# Convenience function for quick reporting
def print_prop_report(days: int = 30, db_path: str = None):
    """Print prop performance report."""
    tracker = PropTracker(db_path)
    tracker.print_performance_report(days)


if __name__ == "__main__":
    # Example usage
    tracker = PropTracker()

    # Record some test predictions
    pred_id = tracker.record_prediction(
        game_id="12345",
        game_date="2024-12-16",
        player_id=1001,
        player_name="Test Player",
        team_abbrev="LAL",
        opponent_abbrev="BOS",
        prop_type="points",
        predicted_value=25.5,
        market_line=24.5,
        pick="OVER",
        edge_pct=4.1,
        confidence=72,
        opp_def_rating=112.5,
        opp_adjustment=1.2,
    )
    print(f"Recorded prediction: {pred_id}")

    # Get unsettled
    unsettled = tracker.get_unsettled_predictions()
    print(f"Unsettled predictions: {len(unsettled)}")

    # Print report
    tracker.print_performance_report()
