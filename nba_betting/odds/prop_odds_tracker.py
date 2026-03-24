"""
Player Prop Odds Tracker — Phase 4.3: Line Movement Tracking

Stores point-in-time player prop odds snapshots from multiple sportsbooks
and computes line movement signals for each prediction.

Line Movement Signals:
    CONFIRMS_MODEL  — The line moved in the direction our model predicted.
                      (e.g., model says OVER, line rose → books agree → stronger bet)
    WARNS_MODEL     — The line moved against our model's prediction.
                      (e.g., model says OVER, line fell → sharp money disagrees)
    NEUTRAL         — No significant movement, or insufficient historical data.

Storage:
    PostgreSQL primary (prop_odds_snapshots table from migration 011).
    SQLite fallback for local development.

Usage:
    tracker = PropOddsTracker()
    # During odds_tracker_service runs:
    tracker.store_snapshot(game_date, player_name, prop_type, book, line, over_odds, under_odds)
    # In daily_predictions.py:
    signal = tracker.get_movement_signal(game_date, player_name, prop_type, pick)
"""

from __future__ import annotations

import sqlite3
import logging
import os
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Default SQLite path (local dev / fallback)
DEFAULT_SQLITE_PATH = "prop_odds.db"

# Movement threshold: changes smaller than this are noise
MOVEMENT_THRESHOLD = 0.5  # half-point minimum to call it significant


# ---------------------------------------------------------------------------
# PostgreSQL connection helper (optional dependency)
# ---------------------------------------------------------------------------
def _get_postgres_connection():
    """Return a live psycopg2 connection using DATABASE_URL, or None."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    try:
        import psycopg2
        return psycopg2.connect(db_url)
    except Exception as exc:
        logger.debug("PropOddsTracker: PostgreSQL unavailable (%s), using SQLite", exc)
        return None


# ---------------------------------------------------------------------------
# PropOddsTracker
# ---------------------------------------------------------------------------
class PropOddsTracker:
    """
    Tracks player prop line movements across sportsbooks.

    Persists odds snapshots to PostgreSQL (primary) or SQLite (fallback).
    Used by:
      - odds_tracker_service.py — stores snapshots every N minutes
      - daily_predictions.py   — reads movement signals + stores daily snapshot
    """

    def __init__(self, sqlite_path: str = DEFAULT_SQLITE_PATH):
        self._sqlite_path = sqlite_path
        self._pg_conn = None
        self._use_postgres = False
        self._setup_backend()

    # ------------------------------------------------------------------
    # Backend setup
    # ------------------------------------------------------------------
    def _setup_backend(self) -> None:
        """Connect to PostgreSQL or initialise SQLite schema."""
        conn = _get_postgres_connection()
        if conn is not None:
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1 FROM prop_odds_snapshots LIMIT 0")
                cur.close()
                self._pg_conn = conn
                self._use_postgres = True
                logger.info("PropOddsTracker: using PostgreSQL")
                return
            except Exception as exc:
                logger.warning(
                    "PropOddsTracker: prop_odds_snapshots table not ready (%s), "
                    "using SQLite — run migration 011 to enable PostgreSQL storage", exc
                )
                try:
                    conn.rollback()
                except Exception:
                    pass

        # Fallback: local SQLite
        self._init_sqlite()
        logger.info("PropOddsTracker: using SQLite at %s", self._sqlite_path)

    def _init_sqlite(self) -> None:
        """Create the prop_odds_snapshots table in SQLite if it doesn't exist."""
        with self._sqlite_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prop_odds_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_date TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    prop_type TEXT NOT NULL,
                    book_name TEXT NOT NULL,
                    line REAL NOT NULL,
                    over_odds INTEGER,
                    under_odds INTEGER,
                    implied_prob_over REAL,
                    timestamp TEXT NOT NULL,
                    is_opening INTEGER DEFAULT 0,
                    UNIQUE(game_date, player_name, prop_type, book_name, timestamp)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pos_player_prop_date
                    ON prop_odds_snapshots(player_name, prop_type, game_date, timestamp)
            """)
            conn.commit()

    @contextmanager
    def _sqlite_conn(self):
        conn = sqlite3.connect(self._sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Core write operation
    # ------------------------------------------------------------------
    def store_snapshot(
        self,
        game_date: str | date,
        player_name: str,
        prop_type: str,
        book_name: str,
        line: float,
        over_odds: int = -110,
        under_odds: int = -110,
        is_opening: bool = False,
        timestamp: datetime | None = None,
    ) -> None:
        """Store a single prop odds snapshot.

        Args:
            game_date: Game date (YYYY-MM-DD string or date object).
            player_name: Player's full name (as returned by The Odds API).
            prop_type: 'points', 'rebounds', 'assists', or 'pra'.
            book_name: Sportsbook key (e.g. 'draftkings', 'fanduel').
            line: The betting line (e.g. 24.5).
            over_odds: American odds for the over (e.g. -110).
            under_odds: American odds for the under (e.g. -115).
            is_opening: True if this is the first snapshot of the day.
            timestamp: Override timestamp (defaults to now).
        """
        if isinstance(game_date, date):
            game_date = game_date.isoformat()
        ts = (timestamp or datetime.now()).isoformat()
        implied = _remove_vig_over(over_odds, under_odds)

        if self._use_postgres:
            self._pg_store(game_date, player_name, prop_type, book_name,
                           line, over_odds, under_odds, implied, ts, is_opening)
        else:
            self._sqlite_store(game_date, player_name, prop_type, book_name,
                               line, over_odds, under_odds, implied, ts, is_opening)

    def store_snapshots_bulk(
        self,
        game_date: str | date,
        props: list[dict],
        is_opening: bool = False,
        timestamp: datetime | None = None,
    ) -> int:
        """Store multiple prop snapshots in one call.

        Args:
            game_date: Game date.
            props: List of dicts with keys: player_name, prop_type, book_name,
                   line, over_odds, under_odds.
            is_opening: Mark all snapshots as opening odds.
            timestamp: Override timestamp.

        Returns:
            Number of snapshots successfully stored.
        """
        stored = 0
        ts = timestamp or datetime.now()
        for p in props:
            try:
                self.store_snapshot(
                    game_date=game_date,
                    player_name=p["player_name"],
                    prop_type=p["prop_type"],
                    book_name=p["book_name"],
                    line=p["line"],
                    over_odds=p.get("over_odds", -110),
                    under_odds=p.get("under_odds", -110),
                    is_opening=is_opening,
                    timestamp=ts,
                )
                stored += 1
            except Exception as exc:
                logger.debug("PropOddsTracker.store_snapshot failed for %s %s: %s",
                             p.get("player_name"), p.get("prop_type"), exc)
        return stored

    # ------------------------------------------------------------------
    # Core read operations
    # ------------------------------------------------------------------
    def get_snapshots(
        self,
        game_date: str | date,
        player_name: str,
        prop_type: str,
        lookback_hours: int = 24,
    ) -> list[dict]:
        """Fetch all stored snapshots for a player prop on a given date.

        Returns snapshots ordered by timestamp ascending (oldest first).
        """
        if isinstance(game_date, date):
            game_date = game_date.isoformat()
        cutoff = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()

        if self._use_postgres:
            return self._pg_fetch_snapshots(game_date, player_name, prop_type, cutoff)
        return self._sqlite_fetch_snapshots(game_date, player_name, prop_type, cutoff)

    def get_opening_line(
        self,
        game_date: str | date,
        player_name: str,
        prop_type: str,
    ) -> float | None:
        """Return the opening line for a player prop (earliest snapshot of the day)."""
        snapshots = self.get_snapshots(game_date, player_name, prop_type, lookback_hours=36)
        if not snapshots:
            return None
        # Prefer is_opening=True; otherwise use oldest
        opening_snaps = [s for s in snapshots if s.get("is_opening")]
        if opening_snaps:
            return opening_snaps[0]["line"]
        return snapshots[0]["line"]

    def get_current_line(
        self,
        game_date: str | date,
        player_name: str,
        prop_type: str,
    ) -> float | None:
        """Return the most recent line for a player prop."""
        snapshots = self.get_snapshots(game_date, player_name, prop_type)
        if not snapshots:
            return None
        return snapshots[-1]["line"]

    def get_line_movement(
        self,
        game_date: str | date,
        player_name: str,
        prop_type: str,
    ) -> dict | None:
        """Calculate line movement from opening to current.

        Returns:
            {
                "opening_line": float,
                "current_line": float,
                "movement": float,        # current - opening (positive = line went up)
                "opening_timestamp": str,
                "current_timestamp": str,
                "num_snapshots": int,
            }
            or None if fewer than 2 snapshots exist.
        """
        snapshots = self.get_snapshots(game_date, player_name, prop_type, lookback_hours=36)
        if len(snapshots) < 2:
            return None

        opening = snapshots[0]
        current = snapshots[-1]
        movement = current["line"] - opening["line"]

        return {
            "opening_line": opening["line"],
            "current_line": current["line"],
            "movement": round(movement, 2),
            "opening_timestamp": opening.get("timestamp", ""),
            "current_timestamp": current.get("timestamp", ""),
            "num_snapshots": len(snapshots),
        }

    def get_movement_signal(
        self,
        game_date: str | date,
        player_name: str,
        prop_type: str,
        pick: str,
    ) -> str:
        """Compute the line movement signal for a model prediction.

        Logic:
            - If model says OVER and line went UP → CONFIRMS_MODEL (books moving same way)
            - If model says OVER and line went DOWN → WARNS_MODEL (smart money disagrees)
            - Vice versa for UNDER picks
            - Movements < MOVEMENT_THRESHOLD are considered noise → NEUTRAL

        Args:
            game_date: Game date.
            player_name: Player name.
            prop_type: Prop category.
            pick: "OVER", "UNDER", or "-".

        Returns:
            "CONFIRMS_MODEL", "WARNS_MODEL", or "NEUTRAL".
        """
        if pick not in ("OVER", "UNDER"):
            return "NEUTRAL"

        movement_data = self.get_line_movement(game_date, player_name, prop_type)
        if movement_data is None:
            return "NEUTRAL"

        movement = movement_data["movement"]
        if abs(movement) < MOVEMENT_THRESHOLD:
            return "NEUTRAL"

        # Line moved UP: books think the OVER is more likely → OVER pick confirmed
        # Line moved DOWN: books think the UNDER is more likely → UNDER pick confirmed
        if pick == "OVER":
            return "CONFIRMS_MODEL" if movement > 0 else "WARNS_MODEL"
        else:  # UNDER
            return "CONFIRMS_MODEL" if movement < 0 else "WARNS_MODEL"

    # ------------------------------------------------------------------
    # Per-book aggregation for dashboard
    # ------------------------------------------------------------------
    def get_book_comparison(
        self,
        game_date: str | date,
        player_name: str,
        prop_type: str,
    ) -> list[dict]:
        """Return the most recent snapshot per sportsbook for line shopping display.

        Returns:
            List of dicts: [{book, line, over_odds, under_odds, implied_prob_over}]
            sorted by over_odds descending (best value first).
        """
        snapshots = self.get_snapshots(game_date, player_name, prop_type)
        latest_per_book: dict[str, dict] = {}
        for s in snapshots:
            book = s.get("book_name", "unknown")
            latest_per_book[book] = s  # last write wins (ordered oldest→newest)

        rows = []
        for book, s in latest_per_book.items():
            rows.append({
                "book": book,
                "line": s.get("line"),
                "over_odds": s.get("over_odds"),
                "under_odds": s.get("under_odds"),
                "implied_prob_over": s.get("implied_prob_over"),
            })
        rows.sort(key=lambda x: (x.get("over_odds") or -999), reverse=True)
        return rows

    # ------------------------------------------------------------------
    # PostgreSQL helpers
    # ------------------------------------------------------------------
    def _pg_store(self, game_date, player_name, prop_type, book_name,
                  line, over_odds, under_odds, implied, ts, is_opening) -> None:
        try:
            cur = self._pg_conn.cursor()
            cur.execute("""
                INSERT INTO prop_odds_snapshots
                    (game_date, player_name, prop_type, book_name, line,
                     over_odds, under_odds, implied_prob_over, timestamp, is_opening)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_date, player_name, prop_type, book_name, timestamp)
                DO NOTHING
            """, (game_date, player_name, prop_type, book_name, line,
                  over_odds, under_odds, implied, ts, is_opening))
            self._pg_conn.commit()
            cur.close()
        except Exception as exc:
            logger.warning("PropOddsTracker PG store error: %s", exc)
            try:
                self._pg_conn.rollback()
            except Exception:
                pass

    def _pg_fetch_snapshots(self, game_date, player_name, prop_type, cutoff) -> list[dict]:
        try:
            cur = self._pg_conn.cursor()
            cur.execute("""
                SELECT game_date, player_name, prop_type, book_name, line,
                       over_odds, under_odds, implied_prob_over, timestamp, is_opening
                FROM prop_odds_snapshots
                WHERE game_date = %s
                  AND player_name = %s
                  AND prop_type = %s
                  AND timestamp >= %s
                ORDER BY timestamp ASC
            """, (game_date, player_name, prop_type, cutoff))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
            cur.close()
            return rows
        except Exception as exc:
            logger.warning("PropOddsTracker PG fetch error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------
    def _sqlite_store(self, game_date, player_name, prop_type, book_name,
                      line, over_odds, under_odds, implied, ts, is_opening) -> None:
        try:
            with self._sqlite_conn() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO prop_odds_snapshots
                        (game_date, player_name, prop_type, book_name, line,
                         over_odds, under_odds, implied_prob_over, timestamp, is_opening)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (game_date, player_name, prop_type, book_name, line,
                      over_odds, under_odds, implied, ts, int(is_opening)))
                conn.commit()
        except Exception as exc:
            logger.warning("PropOddsTracker SQLite store error: %s", exc)

    def _sqlite_fetch_snapshots(self, game_date, player_name, prop_type, cutoff) -> list[dict]:
        try:
            with self._sqlite_conn() as conn:
                cur = conn.execute("""
                    SELECT game_date, player_name, prop_type, book_name, line,
                           over_odds, under_odds, implied_prob_over, timestamp,
                           CAST(is_opening AS BOOLEAN) as is_opening
                    FROM prop_odds_snapshots
                    WHERE game_date = ?
                      AND player_name = ?
                      AND prop_type = ?
                      AND timestamp >= ?
                    ORDER BY timestamp ASC
                """, (game_date, player_name, prop_type, cutoff))
                return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.warning("PropOddsTracker SQLite fetch error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Module-level convenience — singleton used by odds_tracker_service
# ---------------------------------------------------------------------------
_tracker: PropOddsTracker | None = None


def get_prop_tracker() -> PropOddsTracker:
    """Return the module-level PropOddsTracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = PropOddsTracker()
    return _tracker


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------
def _remove_vig_over(over_odds: int, under_odds: int) -> float:
    """Return the vig-free implied probability for the OVER side."""
    def _raw(o: int) -> float:
        return 100.0 / (o + 100.0) if o >= 0 else abs(o) / (abs(o) + 100.0)

    raw_over = _raw(over_odds)
    raw_under = _raw(under_odds)
    total = raw_over + raw_under
    return round(raw_over / total, 4) if total > 0 else 0.5
