"""
Paper Trading / Forward Validation System

Tracks ALL predictions (not just recommended bets) and grades them after
games complete. Simulates P&L using recommended bet sizes at the actual
odds available at prediction time.

PostgreSQL-primary with SQLite-fallback pattern (same as PropTracker).
Production uses PostgreSQL (paper_trades table from migration 009).
Local dev/tests fall back to SQLite (data/paper_trades.db).
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

try:
    from agents.core.connections import get_postgres_connection
except (ImportError, TypeError):
    def get_postgres_connection():
        return None

logger = logging.getLogger(__name__)

# Standard paper bet size used when should_bet=True but no explicit bet_size is stored.
# Historically the pipeline logged bet_size=0 even for recommended bets because
# suggested_bet_size was null; this constant ensures P&L is always computable.
DEFAULT_PAPER_BET = 10.0


def _american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds.

    Args:
        american_odds: American-format odds (e.g., -110, +150).

    Returns:
        Decimal odds (e.g., 1.909, 2.5).
    """
    if american_odds is None or american_odds == 0:
        american_odds = -110
    if american_odds >= 100:
        return (american_odds / 100.0) + 1.0
    return (100.0 / abs(american_odds)) + 1.0


class PaperTrader:
    """Forward validation system — tracks ALL predictions vs actual outcomes.

    Stores every prediction (not just recommended bets) and grades them
    after games complete. Simulates P&L using recommended bet sizes at
    the actual odds available at prediction time.

    Uses PostgreSQL (paper_trades table) with SQLite fallback.
    """

    def __init__(self, db_path: str = None, pg_conn=None):
        """Initialize with PostgreSQL primary, SQLite fallback.

        Args:
            db_path: Path to SQLite database. Defaults to data/paper_trades.db.
            pg_conn: Optional existing PostgreSQL connection.
        """
        self._use_postgres = False
        self._pg_conn = None

        conn = pg_conn or get_postgres_connection()
        if conn is not None:
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'paper_trades'
                    )
                """)
                exists = cur.fetchone()[0]
                cur.close()
                if exists:
                    self._use_postgres = True
                    self._pg_conn = conn
                    logger.info("PaperTrader using PostgreSQL (paper_trades)")
                else:
                    logger.warning(
                        "PostgreSQL available but paper_trades table missing — "
                        "falling back to SQLite"
                    )
            except Exception as e:
                logger.warning(f"PostgreSQL verification failed: {e} — falling back to SQLite")

        if not self._use_postgres:
            if db_path is None:
                db_path = str(
                    Path(__file__).resolve().parent.parent / "data" / "paper_trades.db"
                )
            self.db_path = db_path
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()
            logger.info(f"PaperTrader using SQLite: {self.db_path}")
        else:
            self.db_path = None

    def _init_sqlite(self):
        """Create paper_trades table in SQLite if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    game_date TEXT NOT NULL,
                    game_id TEXT,
                    player_name TEXT NOT NULL,
                    prop_type TEXT NOT NULL,
                    line REAL NOT NULL,
                    direction TEXT NOT NULL,
                    predicted_value REAL,
                    over_prob REAL,
                    edge REAL,
                    true_ev REAL,
                    should_bet INTEGER DEFAULT 0,
                    bet_size REAL DEFAULT 0,
                    over_odds INTEGER,
                    under_odds INTEGER,
                    confidence REAL,
                    tier TEXT,
                    actual_value REAL,
                    result TEXT,
                    profit_loss REAL,
                    settled_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_paper_trades_date ON paper_trades(game_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_paper_trades_prop ON paper_trades(prop_type)"
            )
            conn.commit()

    def log_prediction(self, prediction: dict) -> str:
        """Record a single prediction to the database.

        Accepts both canonical format (player_name, prop_type, direction) and
        pipeline format (player, stat, pick) — normalizes internally.

        Args:
            prediction: Dict with prediction data.

        Returns:
            trade_id: Unique identifier for this trade.

        Raises:
            ValueError: If required keys (player_name/player, prop_type/stat) are missing.
        """
        player_name = prediction.get("player_name") or prediction.get("player", "")
        prop_type = prediction.get("prop_type") or prediction.get("stat", "")
        if not player_name or not prop_type:
            raise ValueError(
                f"Missing required keys: player_name={player_name!r}, prop_type={prop_type!r}"
            )

        game_date = prediction.get("game_date", "")
        game_id = prediction.get("game_id") or prediction.get("game", "")
        line = float(prediction.get("line", 0))
        direction = prediction.get("direction") or prediction.get("pick", "over")
        direction = direction.lower()
        predicted_value = prediction.get("predicted_value")
        if predicted_value is not None:
            predicted_value = float(predicted_value)
        over_prob = prediction.get("over_prob")
        if over_prob is not None:
            over_prob = float(over_prob)
        edge = prediction.get("edge")
        if edge is not None:
            edge = float(edge)
        true_ev = prediction.get("true_ev") or prediction.get("ev") or prediction.get("ev_per_dollar")
        if true_ev is not None:
            true_ev = float(true_ev)

        should_bet = prediction.get("should_bet")
        if should_bet is None:
            signal = prediction.get("signal") or prediction.get("bet_recommendation", "PASS")
            should_bet = signal in ("BET",)

        bet_size = float(prediction.get("bet_size") or prediction.get("suggested_bet_size", 0) or 0)
        # If this is a recommended bet but no size was provided, use the default paper bet.
        # The pipeline historically left suggested_bet_size=None for all BET-signal props.
        if should_bet and not bet_size:
            bet_size = DEFAULT_PAPER_BET

        over_odds = prediction.get("over_odds")
        if over_odds is not None:
            over_odds = int(over_odds)
        under_odds = prediction.get("under_odds")
        if under_odds is not None:
            under_odds = int(under_odds)

        confidence = prediction.get("confidence") or prediction.get("confidence_score")
        if confidence is not None:
            confidence = float(confidence)
        tier = prediction.get("tier") or prediction.get("edge_quality_tier", "")

        trade_id = f"{game_date}_{player_name}_{prop_type}".replace(" ", "_").lower()

        if self._use_postgres:
            try:
                cur = self._pg_conn.cursor()
                cur.execute("""
                    INSERT INTO paper_trades (
                        trade_id, game_date, game_id, player_name, prop_type,
                        line, direction, predicted_value, over_prob, edge,
                        true_ev, should_bet, bet_size, over_odds, under_odds,
                        confidence, tier, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trade_id) DO NOTHING
                """, (
                    trade_id, game_date, game_id, player_name, prop_type,
                    line, direction, predicted_value, over_prob, edge,
                    true_ev, should_bet, bet_size, over_odds, under_odds,
                    confidence, tier, datetime.now().isoformat(),
                ))
                cur.close()
            except Exception as e:
                logger.error(f"PostgreSQL log_prediction failed: {e}")
                raise
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO paper_trades (
                        trade_id, game_date, game_id, player_name, prop_type,
                        line, direction, predicted_value, over_prob, edge,
                        true_ev, should_bet, bet_size, over_odds, under_odds,
                        confidence, tier, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id, game_date, game_id, player_name, prop_type,
                    line, direction, predicted_value, over_prob, edge,
                    true_ev, int(bool(should_bet)), bet_size, over_odds, under_odds,
                    confidence, tier, datetime.now().isoformat(),
                ))
                conn.commit()

        return trade_id

    def log_predictions_batch(self, predictions: list, game_date: str) -> int:
        """Record all predictions from a daily run.

        Args:
            predictions: List of prediction dicts from the daily pipeline.
            game_date: Date string (YYYY-MM-DD).

        Returns:
            Number of predictions successfully logged.
        """
        count = 0
        for pred in predictions:
            try:
                if "game_date" not in pred:
                    pred["game_date"] = game_date
                self.log_prediction(pred)
                count += 1
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping bad prediction: {e}")
            except Exception as e:
                logger.warning(f"Failed to log prediction: {e}")
        return count

    def settle_trades(self, game_date: str, actual_stats: dict = None) -> int:
        """Grade predictions for a given date using actual results.

        For each unsettled trade on the given date:
        1. Look up the actual stat value from actual_stats
        2. Compare actual vs line to determine over/under/push
        3. Calculate P&L for recommended bets

        Args:
            game_date: Date to settle (YYYY-MM-DD).
            actual_stats: Dict mapping (player_name, prop_type) to actual_value.

        Returns:
            Number of trades settled.
        """
        if actual_stats is None:
            return 0

        settled = 0

        if self._use_postgres:
            try:
                cur = self._pg_conn.cursor()
                cur.execute("""
                    SELECT id, trade_id, player_name, prop_type, line, direction,
                           should_bet, bet_size, over_odds, under_odds
                    FROM paper_trades
                    WHERE game_date = %s AND result IS NULL
                """, (game_date,))
                rows = cur.fetchall()

                for row in rows:
                    (row_id, trade_id, player_name, prop_type, line, direction,
                     should_bet, bet_size, over_odds, under_odds) = row

                    key = (player_name, prop_type)
                    if key not in actual_stats:
                        continue

                    actual_value = float(actual_stats[key])
                    result, profit = self._compute_settlement(
                        actual_value, line, direction, should_bet, bet_size,
                        over_odds, under_odds
                    )
                    # Also write back the effective bet_size so total_wagered queries work.
                    # Bets logged before the DEFAULT_PAPER_BET fix have bet_size=0.
                    effective_bet = float(bet_size or 0)
                    if should_bet and not effective_bet:
                        effective_bet = DEFAULT_PAPER_BET

                    cur.execute("""
                        UPDATE paper_trades
                        SET actual_value = %s, result = %s, profit_loss = %s,
                            bet_size = %s, settled_at = %s
                        WHERE id = %s
                    """, (actual_value, result, profit, effective_bet,
                          datetime.now().isoformat(), row_id))
                    settled += 1

                cur.close()
            except Exception as e:
                logger.error(f"PostgreSQL settle_trades failed: {e}")
                raise
        else:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, trade_id, player_name, prop_type, line, direction,
                           should_bet, bet_size, over_odds, under_odds
                    FROM paper_trades
                    WHERE game_date = ? AND result IS NULL
                """, (game_date,))
                rows = cursor.fetchall()

                for row in rows:
                    (row_id, trade_id, player_name, prop_type, line, direction,
                     should_bet, bet_size, over_odds, under_odds) = row

                    key = (player_name, prop_type)
                    if key not in actual_stats:
                        continue

                    actual_value = float(actual_stats[key])
                    result, profit = self._compute_settlement(
                        actual_value, line, direction, should_bet, bet_size,
                        over_odds, under_odds
                    )
                    effective_bet = float(bet_size or 0)
                    if should_bet and not effective_bet:
                        effective_bet = DEFAULT_PAPER_BET

                    conn.execute("""
                        UPDATE paper_trades
                        SET actual_value = ?, result = ?, profit_loss = ?,
                            bet_size = ?, settled_at = ?
                        WHERE id = ?
                    """, (actual_value, result, profit, effective_bet,
                          datetime.now().isoformat(), row_id))
                    settled += 1

                conn.commit()

        return settled

    @staticmethod
    def _compute_settlement(
        actual_value: float,
        line: float,
        direction: str,
        should_bet: bool,
        bet_size: float,
        over_odds: int,
        under_odds: int,
    ) -> tuple:
        """Compute result and P&L for a single trade.

        Returns:
            (result, profit_loss) tuple.
        """
        if actual_value == line:
            return ("push", 0.0)

        over_hit = actual_value > line
        dir_lower = (direction or "over").lower()

        if (dir_lower == "over" and over_hit) or (dir_lower == "under" and not over_hit):
            result = "hit"
        else:
            result = "miss"

        profit = 0.0
        if should_bet:
            # Use stored bet_size, falling back to DEFAULT_PAPER_BET for bets that were
            # logged before the default was introduced (bet_size was always 0 then).
            effective_bet = bet_size if (bet_size and bet_size > 0) else DEFAULT_PAPER_BET
            odds = over_odds if dir_lower == "over" else under_odds
            decimal_odds = _american_to_decimal(odds)
            if result == "hit":
                profit = effective_bet * (decimal_odds - 1)
            else:
                profit = -effective_bet

        return (result, profit)

    def backfill_profit_loss(self) -> dict:
        """Recompute profit_loss and bet_size for all settled bets that have no P&L.

        Targets rows where:
          - result IN ('hit', 'miss')   — already settled
          - should_bet = TRUE           — recommended bet
          - profit_loss IS NULL OR profit_loss = 0  — P&L never computed

        Uses DEFAULT_PAPER_BET ($10) and stored odds (defaulting to -110 if null).

        Returns:
            Dict with updated_count and error (if any).
        """
        updated = 0

        if self._use_postgres:
            try:
                cur = self._pg_conn.cursor()
                cur.execute("""
                    SELECT id, direction, result, bet_size, over_odds, under_odds
                    FROM paper_trades
                    WHERE result IN ('hit', 'miss')
                      AND should_bet = TRUE
                      AND (profit_loss IS NULL OR profit_loss = 0)
                """)
                rows = cur.fetchall()

                for (row_id, direction, result, bet_size, over_odds, under_odds) in rows:
                    dir_lower = (direction or "over").lower()
                    effective_bet = float(bet_size or 0) or DEFAULT_PAPER_BET
                    odds = over_odds if dir_lower == "over" else under_odds
                    decimal_odds = _american_to_decimal(odds)

                    if result == "hit":
                        profit = effective_bet * (decimal_odds - 1)
                    else:
                        profit = -effective_bet

                    cur.execute("""
                        UPDATE paper_trades
                        SET profit_loss = %s, bet_size = %s
                        WHERE id = %s
                    """, (profit, effective_bet, row_id))
                    updated += 1

                cur.close()
                logger.info(f"backfill_profit_loss: updated {updated} rows in PostgreSQL")
            except Exception as e:
                logger.error(f"backfill_profit_loss failed: {e}")
                return {"updated_count": updated, "error": str(e)}
        else:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    rows = conn.execute("""
                        SELECT id, direction, result, bet_size, over_odds, under_odds
                        FROM paper_trades
                        WHERE result IN ('hit', 'miss')
                          AND should_bet = 1
                          AND (profit_loss IS NULL OR profit_loss = 0)
                    """).fetchall()

                    for (row_id, direction, result, bet_size, over_odds, under_odds) in rows:
                        dir_lower = (direction or "over").lower()
                        effective_bet = float(bet_size or 0) or DEFAULT_PAPER_BET
                        odds = over_odds if dir_lower == "over" else under_odds
                        decimal_odds = _american_to_decimal(odds)

                        if result == "hit":
                            profit = effective_bet * (decimal_odds - 1)
                        else:
                            profit = -effective_bet

                        conn.execute("""
                            UPDATE paper_trades
                            SET profit_loss = ?, bet_size = ?
                            WHERE id = ?
                        """, (profit, effective_bet, row_id))
                        updated += 1

                    conn.commit()
                logger.info(f"backfill_profit_loss: updated {updated} rows in SQLite")
            except Exception as e:
                logger.error(f"backfill_profit_loss (SQLite) failed: {e}")
                return {"updated_count": updated, "error": str(e)}

        return {"updated_count": updated}

    def get_summary(self, days: int = None) -> dict:
        """Return comprehensive paper trading performance summary.

        Args:
            days: If provided, limit to last N days.

        Returns:
            Dict with total_predictions, settled_predictions, unsettled_predictions,
            overall_accuracy, recommended_bets, recommended_accuracy, total_wagered,
            total_profit, roi, brier_score, by_prop_type, by_confidence_tier,
            by_edge_bucket.
        """
        empty = {
            "total_predictions": 0, "settled_predictions": 0,
            "unsettled_predictions": 0, "overall_accuracy": 0.0,
            "recommended_bets": 0, "recommended_accuracy": 0.0,
            "total_wagered": 0.0, "total_profit": 0.0, "roi": 0.0,
            "brier_score": 0.0, "by_prop_type": {},
            "by_confidence_tier": {}, "by_edge_bucket": {},
        }

        if self._use_postgres:
            return self._get_summary_pg(days, empty)
        return self._get_summary_sqlite(days, empty)

    def _get_summary_sqlite(self, days: int, empty: dict) -> dict:
        date_clause = ""
        date_params: tuple = ()
        if days is not None:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            date_clause = "AND game_date >= ?"
            date_params = (cutoff,)

        with sqlite3.connect(self.db_path) as conn:
            # Overall counts
            row = conn.execute(f"""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN result IS NOT NULL THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result IS NULL THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result = 'hit' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result IN ('hit', 'miss') THEN 1 ELSE 0 END),
                    SUM(CASE WHEN should_bet = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN should_bet = 1 AND result = 'hit' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN should_bet = 1 AND result IN ('hit', 'miss') THEN 1 ELSE 0 END),
                    SUM(CASE WHEN should_bet = 1 AND result IS NOT NULL THEN bet_size ELSE 0 END),
                    SUM(CASE WHEN should_bet = 1 AND result IS NOT NULL THEN COALESCE(profit_loss, 0) ELSE 0 END)
                FROM paper_trades
                WHERE 1=1 {date_clause}
            """, date_params).fetchone()

            total = row[0] or 0
            settled = row[1] or 0
            unsettled = row[2] or 0
            hits = row[3] or 0
            decided = row[4] or 0
            rec_bets = row[5] or 0
            rec_hits = row[6] or 0
            rec_decided = row[7] or 0
            total_wagered = row[8] or 0.0
            total_profit = row[9] or 0.0

            overall_accuracy = hits / decided if decided > 0 else 0.0
            rec_accuracy = rec_hits / rec_decided if rec_decided > 0 else 0.0
            roi = total_profit / total_wagered if total_wagered > 0 else 0.0

            # Brier score: mean((over_prob - actual_binary)^2)
            brier_row = conn.execute(f"""
                SELECT AVG(
                    (COALESCE(over_prob, 0.5) - CASE WHEN actual_value > line THEN 1.0 ELSE 0.0 END)
                    * (COALESCE(over_prob, 0.5) - CASE WHEN actual_value > line THEN 1.0 ELSE 0.0 END)
                )
                FROM paper_trades
                WHERE result IS NOT NULL AND result != 'push' {date_clause}
            """, date_params).fetchone()
            brier_score = brier_row[0] if brier_row[0] is not None else 0.0

            # By prop type
            by_prop_type = {}
            prop_rows = conn.execute(f"""
                SELECT
                    prop_type,
                    COUNT(*),
                    SUM(CASE WHEN result = 'hit' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result IN ('hit', 'miss') THEN 1 ELSE 0 END),
                    SUM(CASE WHEN should_bet = 1 AND result IS NOT NULL THEN bet_size ELSE 0 END),
                    SUM(CASE WHEN should_bet = 1 AND result IS NOT NULL THEN COALESCE(profit_loss, 0) ELSE 0 END)
                FROM paper_trades
                WHERE result IS NOT NULL AND result != 'push' {date_clause}
                GROUP BY prop_type
            """, date_params).fetchall()
            for prow in prop_rows:
                pt, cnt, pt_hits, pt_decided, pt_wagered, pt_profit = prow
                pt_hits = pt_hits or 0
                pt_decided = pt_decided or 0
                pt_wagered = pt_wagered or 0.0
                pt_profit = pt_profit or 0.0
                by_prop_type[pt] = {
                    "count": cnt,
                    "accuracy": pt_hits / pt_decided if pt_decided > 0 else 0.0,
                    "roi": pt_profit / pt_wagered if pt_wagered > 0 else 0.0,
                }

            # By confidence tier
            by_confidence_tier = {}
            tier_rows = conn.execute(f"""
                SELECT
                    COALESCE(tier, 'unknown'),
                    COUNT(*),
                    SUM(CASE WHEN result = 'hit' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result IN ('hit', 'miss') THEN 1 ELSE 0 END)
                FROM paper_trades
                WHERE result IS NOT NULL AND result != 'push' {date_clause}
                GROUP BY tier
            """, date_params).fetchall()
            for trow in tier_rows:
                t_name, t_cnt, t_hits, t_decided = trow
                t_hits = t_hits or 0
                t_decided = t_decided or 0
                by_confidence_tier[t_name] = {
                    "count": t_cnt,
                    "accuracy": t_hits / t_decided if t_decided > 0 else 0.0,
                }

            # By edge bucket
            by_edge_bucket = {}
            edge_rows = conn.execute(f"""
                SELECT
                    CASE
                        WHEN ABS(COALESCE(edge, 0)) < 2 THEN '0-2'
                        WHEN ABS(COALESCE(edge, 0)) < 5 THEN '2-5'
                        ELSE '5+'
                    END as bucket,
                    COUNT(*),
                    SUM(CASE WHEN result = 'hit' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result IN ('hit', 'miss') THEN 1 ELSE 0 END)
                FROM paper_trades
                WHERE result IS NOT NULL AND result != 'push' {date_clause}
                GROUP BY bucket
            """, date_params).fetchall()
            for erow in edge_rows:
                e_name, e_cnt, e_hits, e_decided = erow
                e_hits = e_hits or 0
                e_decided = e_decided or 0
                by_edge_bucket[e_name] = {
                    "count": e_cnt,
                    "accuracy": e_hits / e_decided if e_decided > 0 else 0.0,
                }

        return {
            "total_predictions": total,
            "settled_predictions": settled,
            "unsettled_predictions": unsettled,
            "overall_accuracy": overall_accuracy,
            "recommended_bets": rec_bets,
            "recommended_accuracy": rec_accuracy,
            "total_wagered": total_wagered,
            "total_profit": total_profit,
            "roi": roi,
            "brier_score": brier_score,
            "by_prop_type": by_prop_type,
            "by_confidence_tier": by_confidence_tier,
            "by_edge_bucket": by_edge_bucket,
        }

    def _get_summary_pg(self, days: int, empty: dict) -> dict:
        date_clause = ""
        date_params: tuple = ()
        if days is not None:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            date_clause = "AND game_date >= %s"
            date_params = (cutoff,)

        try:
            cur = self._pg_conn.cursor()

            cur.execute(f"""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN result IS NOT NULL THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result IS NULL THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result = 'hit' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result IN ('hit', 'miss') THEN 1 ELSE 0 END),
                    SUM(CASE WHEN should_bet = TRUE THEN 1 ELSE 0 END),
                    SUM(CASE WHEN should_bet = TRUE AND result = 'hit' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN should_bet = TRUE AND result IN ('hit', 'miss') THEN 1 ELSE 0 END),
                    SUM(CASE WHEN should_bet = TRUE AND result IS NOT NULL THEN bet_size ELSE 0 END),
                    SUM(CASE WHEN should_bet = TRUE AND result IS NOT NULL THEN COALESCE(profit_loss, 0) ELSE 0 END)
                FROM paper_trades
                WHERE 1=1 {date_clause}
            """, date_params)
            row = cur.fetchone()

            total = row[0] or 0
            settled = row[1] or 0
            unsettled = row[2] or 0
            hits = row[3] or 0
            decided = row[4] or 0
            rec_bets = row[5] or 0
            rec_hits = row[6] or 0
            rec_decided = row[7] or 0
            total_wagered = float(row[8] or 0)
            total_profit = float(row[9] or 0)

            overall_accuracy = hits / decided if decided > 0 else 0.0
            rec_accuracy = rec_hits / rec_decided if rec_decided > 0 else 0.0
            roi = total_profit / total_wagered if total_wagered > 0 else 0.0

            cur.execute(f"""
                SELECT AVG(
                    (COALESCE(over_prob, 0.5) - CASE WHEN actual_value > line THEN 1.0 ELSE 0.0 END)
                    * (COALESCE(over_prob, 0.5) - CASE WHEN actual_value > line THEN 1.0 ELSE 0.0 END)
                )
                FROM paper_trades
                WHERE result IS NOT NULL AND result != 'push' {date_clause}
            """, date_params)
            brier_row = cur.fetchone()
            brier_score = float(brier_row[0]) if brier_row[0] is not None else 0.0

            cur.execute(f"""
                SELECT
                    prop_type,
                    COUNT(*),
                    SUM(CASE WHEN result = 'hit' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result IN ('hit', 'miss') THEN 1 ELSE 0 END),
                    SUM(CASE WHEN should_bet = TRUE AND result IS NOT NULL THEN bet_size ELSE 0 END),
                    SUM(CASE WHEN should_bet = TRUE AND result IS NOT NULL THEN COALESCE(profit_loss, 0) ELSE 0 END)
                FROM paper_trades
                WHERE result IS NOT NULL AND result != 'push' {date_clause}
                GROUP BY prop_type
            """, date_params)
            by_prop_type = {}
            for prow in cur.fetchall():
                pt, cnt, pt_hits, pt_decided, pt_wagered, pt_profit = prow
                pt_hits = pt_hits or 0
                pt_decided = pt_decided or 0
                pt_wagered = float(pt_wagered or 0)
                pt_profit = float(pt_profit or 0)
                by_prop_type[pt] = {
                    "count": cnt,
                    "accuracy": pt_hits / pt_decided if pt_decided > 0 else 0.0,
                    "roi": pt_profit / pt_wagered if pt_wagered > 0 else 0.0,
                }

            cur.execute(f"""
                SELECT
                    COALESCE(tier, 'unknown'),
                    COUNT(*),
                    SUM(CASE WHEN result = 'hit' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result IN ('hit', 'miss') THEN 1 ELSE 0 END)
                FROM paper_trades
                WHERE result IS NOT NULL AND result != 'push' {date_clause}
                GROUP BY tier
            """, date_params)
            by_confidence_tier = {}
            for trow in cur.fetchall():
                t_name, t_cnt, t_hits, t_decided = trow
                t_hits = t_hits or 0
                t_decided = t_decided or 0
                by_confidence_tier[t_name] = {
                    "count": t_cnt,
                    "accuracy": t_hits / t_decided if t_decided > 0 else 0.0,
                }

            cur.execute(f"""
                SELECT
                    CASE
                        WHEN ABS(COALESCE(edge, 0)) < 2 THEN '0-2'
                        WHEN ABS(COALESCE(edge, 0)) < 5 THEN '2-5'
                        ELSE '5+'
                    END as bucket,
                    COUNT(*),
                    SUM(CASE WHEN result = 'hit' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN result IN ('hit', 'miss') THEN 1 ELSE 0 END)
                FROM paper_trades
                WHERE result IS NOT NULL AND result != 'push' {date_clause}
                GROUP BY bucket
            """, date_params)
            by_edge_bucket = {}
            for erow in cur.fetchall():
                e_name, e_cnt, e_hits, e_decided = erow
                e_hits = e_hits or 0
                e_decided = e_decided or 0
                by_edge_bucket[e_name] = {
                    "count": e_cnt,
                    "accuracy": e_hits / e_decided if e_decided > 0 else 0.0,
                }

            cur.close()

            return {
                "total_predictions": total,
                "settled_predictions": settled,
                "unsettled_predictions": unsettled,
                "overall_accuracy": overall_accuracy,
                "recommended_bets": rec_bets,
                "recommended_accuracy": rec_accuracy,
                "total_wagered": total_wagered,
                "total_profit": total_profit,
                "roi": roi,
                "brier_score": brier_score,
                "by_prop_type": by_prop_type,
                "by_confidence_tier": by_confidence_tier,
                "by_edge_bucket": by_edge_bucket,
            }
        except Exception as e:
            logger.error(f"PostgreSQL get_summary failed: {e}")
            raise

    def get_daily_report(self, game_date: str) -> dict:
        """Return results for a specific date.

        Args:
            game_date: Date string (YYYY-MM-DD).

        Returns:
            Dict with date, predictions list, accuracy, and P&L.
        """
        predictions = []
        hits = 0
        decided = 0
        profit = 0.0

        if self._use_postgres:
            try:
                cur = self._pg_conn.cursor()
                cur.execute("""
                    SELECT trade_id, player_name, prop_type, line, direction,
                           predicted_value, over_prob, edge, should_bet, bet_size,
                           over_odds, under_odds, confidence, tier,
                           actual_value, result, profit_loss
                    FROM paper_trades
                    WHERE game_date = %s
                    ORDER BY player_name, prop_type
                """, (game_date,))
                columns = [desc[0] for desc in cur.description]
                for row in cur.fetchall():
                    pred = dict(zip(columns, row, strict=False))
                    predictions.append(pred)
                    if pred["result"] in ("hit", "miss"):
                        decided += 1
                        if pred["result"] == "hit":
                            hits += 1
                    if pred["profit_loss"] is not None:
                        profit += pred["profit_loss"]
                cur.close()
            except Exception as e:
                logger.error(f"PostgreSQL get_daily_report failed: {e}")
                raise
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT trade_id, player_name, prop_type, line, direction,
                           predicted_value, over_prob, edge, should_bet, bet_size,
                           over_odds, under_odds, confidence, tier,
                           actual_value, result, profit_loss
                    FROM paper_trades
                    WHERE game_date = ?
                    ORDER BY player_name, prop_type
                """, (game_date,))
                for row in cursor.fetchall():
                    pred = dict(row)
                    predictions.append(pred)
                    if pred["result"] in ("hit", "miss"):
                        decided += 1
                        if pred["result"] == "hit":
                            hits += 1
                    if pred["profit_loss"] is not None:
                        profit += pred["profit_loss"]

        return {
            "date": game_date,
            "predictions": predictions,
            "total": len(predictions),
            "settled": decided,
            "accuracy": hits / decided if decided > 0 else 0.0,
            "profit_loss": profit,
        }

    def get_streak_info(self) -> dict:
        """Return current win/loss streak on recommended bets.

        Returns:
            Dict with current_streak (positive=wins, negative=losses),
            longest_win_streak, longest_loss_streak.
        """
        results = []

        if self._use_postgres:
            try:
                cur = self._pg_conn.cursor()
                cur.execute("""
                    SELECT result
                    FROM paper_trades
                    WHERE should_bet = TRUE AND result IN ('hit', 'miss')
                    ORDER BY game_date DESC, id DESC
                """)
                results = [r[0] for r in cur.fetchall()]
                cur.close()
            except Exception as e:
                logger.error(f"PostgreSQL get_streak_info failed: {e}")
                raise
        else:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT result
                    FROM paper_trades
                    WHERE should_bet = 1 AND result IN ('hit', 'miss')
                    ORDER BY game_date DESC, id DESC
                """)
                results = [r[0] for r in cursor.fetchall()]

        if not results:
            return {
                "current_streak": 0,
                "longest_win_streak": 0,
                "longest_loss_streak": 0,
            }

        # Current streak (from most recent)
        current_streak = 0
        first_result = results[0]
        for r in results:
            if r == first_result:
                current_streak += 1
            else:
                break
        if first_result == "miss":
            current_streak = -current_streak

        # Longest streaks
        longest_win = 0
        longest_loss = 0
        streak = 0
        prev = None
        for r in results:
            if r == prev:
                streak += 1
            else:
                streak = 1
                prev = r
            if r == "hit":
                longest_win = max(longest_win, streak)
            else:
                longest_loss = max(longest_loss, streak)

        return {
            "current_streak": current_streak,
            "longest_win_streak": longest_win,
            "longest_loss_streak": longest_loss,
        }
