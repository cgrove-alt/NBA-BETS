"""
CLV (Closing Line Value) Analyzer

Analyzes Closing Line Value to determine model sharpness.
CLV measures whether we consistently get better prices than the closing line.
Positive CLV over a large sample = the model is genuinely sharp.

Uses PostgreSQL (tracked_bets table) with SQLite (bets table) fallback,
matching the same dual-DB pattern as PropTracker and BetTracker.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

try:
    from agents.core.connections import get_postgres_connection
except (ImportError, TypeError):
    def get_postgres_connection():
        """Stub when PostgreSQL agent connections unavailable."""
        return

from nba_betting.odds.devig import american_to_implied

logger = logging.getLogger(__name__)


class CLVAnalyzer:
    """Analyze Closing Line Value (CLV) to determine model sharpness.

    CLV measures whether we consistently get better prices than the closing line.
    Positive CLV over a large sample = the model is genuinely sharp.

    Uses PostgreSQL (tracked_bets table) with SQLite (bets table) fallback.
    """

    def __init__(self, db_path: str | None = None, pg_conn=None):
        """Initialize with PostgreSQL primary, SQLite fallback.

        Args:
            db_path: Path to SQLite database. Defaults to data/bet_tracking.db
            pg_conn: Optional existing PostgreSQL connection.
        """
        self._use_postgres = False
        self._pg_conn = None

        # Try PostgreSQL first
        conn = pg_conn or get_postgres_connection()
        if conn is not None:
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'tracked_bets'
                    )
                """)
                exists = cur.fetchone()[0]
                cur.close()
                if exists:
                    self._use_postgres = True
                    self._pg_conn = conn
                    logger.info("CLVAnalyzer using PostgreSQL (tracked_bets)")
                else:
                    logger.warning(
                        "PostgreSQL available but tracked_bets table missing "
                        "-- falling back to SQLite"
                    )
            except Exception as e:
                logger.warning(
                    "PostgreSQL verification failed: %s -- falling back to SQLite", e
                )

        # Fall back to SQLite
        if not self._use_postgres:
            if db_path is None:
                db_path = str(Path("data") / "bet_tracking.db")
            self.db_path = db_path
            # Ensure data directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._ensure_sqlite_table()
            logger.info("CLVAnalyzer using SQLite: %s", self.db_path)
        else:
            self.db_path = None

    def _ensure_sqlite_table(self):
        """Ensure the SQLite bets table exists (BetTracker creates it, but be safe).

        Returns:
            None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS bets (
                        bet_id TEXT PRIMARY KEY,
                        placed_at TEXT NOT NULL,
                        sport TEXT DEFAULT 'NBA',
                        bet_type TEXT NOT NULL,
                        sportsbook TEXT,
                        event_id TEXT,
                        event_name TEXT,
                        event_date TEXT,
                        selection TEXT NOT NULL,
                        odds REAL NOT NULL,
                        stake REAL NOT NULL,
                        potential_payout REAL,
                        model_probability REAL,
                        implied_probability REAL,
                        edge REAL,
                        opening_odds REAL,
                        closing_odds REAL,
                        line_movement REAL DEFAULT 0,
                        status TEXT DEFAULT 'pending',
                        actual_result TEXT,
                        pnl REAL DEFAULT 0,
                        settled_at TEXT,
                        notes TEXT,
                        tags TEXT,
                        parlay_legs TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            logger.warning("SQLite table creation failed: %s", e)

    def _fetch_bets_with_clv(self, days: int | None = None) -> list:
        """Fetch all bets that have both opening_odds and closing_odds.

        Args:
            days: If provided, only fetch bets from the last N days.

        Returns:
            List of dicts with keys: bet_id, opening_odds, closing_odds,
            result, selection, placed_at, tags, status
        """
        cutoff = None
        if days is not None:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        rows = []

        if self._use_postgres:
            try:
                cur = self._pg_conn.cursor()
                if cutoff:
                    cur.execute("""
                        SELECT bet_id, opening_odds, closing_odds, status,
                               selection, placed_at, tags
                        FROM tracked_bets
                        WHERE opening_odds IS NOT NULL
                          AND closing_odds IS NOT NULL
                          AND placed_at >= %s
                        ORDER BY placed_at DESC
                    """, (cutoff,))
                else:
                    cur.execute("""
                        SELECT bet_id, opening_odds, closing_odds, status,
                               selection, placed_at, tags
                        FROM tracked_bets
                        WHERE opening_odds IS NOT NULL
                          AND closing_odds IS NOT NULL
                        ORDER BY placed_at DESC
                    """)
                for row in cur.fetchall():
                    bet_id, opening, closing, status, selection, placed_at, tags = row
                    opening_impl = american_to_implied(float(opening))
                    closing_impl = american_to_implied(float(closing))
                    clv = closing_impl - opening_impl
                    rows.append({
                        "bet_id": bet_id,
                        "opening_odds": float(opening),
                        "closing_odds": float(closing),
                        "opening_implied": opening_impl,
                        "closing_implied": closing_impl,
                        "clv": clv,
                        "status": status,
                        "selection": selection or "",
                        "placed_at": str(placed_at) if placed_at else "",
                        "tags": tags if isinstance(tags, list) else [],
                    })
                cur.close()
            except Exception as e:
                logger.error("PostgreSQL _fetch_bets_with_clv failed: %s", e)
                return []
        else:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    if cutoff:
                        cursor = conn.execute("""
                            SELECT bet_id, opening_odds, closing_odds, status,
                                   selection, placed_at, tags
                            FROM bets
                            WHERE opening_odds IS NOT NULL
                              AND closing_odds IS NOT NULL
                              AND placed_at >= ?
                            ORDER BY placed_at DESC
                        """, (cutoff,))
                    else:
                        cursor = conn.execute("""
                            SELECT bet_id, opening_odds, closing_odds, status,
                                   selection, placed_at, tags
                            FROM bets
                            WHERE opening_odds IS NOT NULL
                              AND closing_odds IS NOT NULL
                            ORDER BY placed_at DESC
                        """)
                    for row in cursor.fetchall():
                        opening = float(row["opening_odds"])
                        closing = float(row["closing_odds"])
                        opening_impl = american_to_implied(opening)
                        closing_impl = american_to_implied(closing)
                        clv = closing_impl - opening_impl

                        # Parse tags from JSON string
                        raw_tags = row["tags"]
                        if isinstance(raw_tags, str):
                            try:
                                parsed_tags = json.loads(raw_tags)
                            except (json.JSONDecodeError, ValueError):
                                parsed_tags = []
                        else:
                            parsed_tags = raw_tags if raw_tags else []

                        rows.append({
                            "bet_id": row["bet_id"],
                            "opening_odds": opening,
                            "closing_odds": closing,
                            "opening_implied": opening_impl,
                            "closing_implied": closing_impl,
                            "clv": clv,
                            "status": row["status"] or "pending",
                            "selection": row["selection"] or "",
                            "placed_at": row["placed_at"] or "",
                            "tags": parsed_tags if isinstance(parsed_tags, list) else [],
                        })
            except sqlite3.Error as e:
                logger.error("SQLite _fetch_bets_with_clv failed: %s", e)
                return []

        return rows

    def _extract_prop_type(self, tags: list) -> str:
        """Extract the prop type from bet tags.

        Tags typically contain [signal, stat_type, direction] e.g. ['BET', 'points', 'over'].

        Args:
            tags: List of tag strings from the bet.

        Returns:
            Prop type string (e.g. 'points', 'rebounds') or 'unknown'.
        """
        prop_types = {"points", "rebounds", "assists", "pra", "3pm", "steals", "blocks"}
        for tag in tags:
            if isinstance(tag, str) and tag.lower() in prop_types:
                return tag.lower()
        return "unknown"

    def _extract_direction(self, tags: list, selection: str) -> str:
        """Extract bet direction (over/under) from tags or selection.

        Args:
            tags: List of tag strings from the bet.
            selection: Selection string (e.g. "LeBron James points OVER 28.5").

        Returns:
            'over', 'under', or 'unknown'.
        """
        for tag in tags:
            if isinstance(tag, str) and tag.lower() in ("over", "under"):
                return tag.lower()
        sel_lower = selection.lower()
        if "over" in sel_lower:
            return "over"
        if "under" in sel_lower:
            return "under"
        return "unknown"

    def _is_settled(self, status: str) -> bool:
        """Check if a bet status counts as settled.

        Args:
            status: Bet status string.

        Returns:
            True if the bet has a final result.
        """
        return status in ("won", "lost", "push")

    def _is_win(self, status: str) -> bool:
        """Check if a bet status is a win.

        Args:
            status: Bet status string.

        Returns:
            True if the bet was won.
        """
        return status == "won"

    def get_clv_summary(self, days: int | None = None) -> dict:
        """Return comprehensive CLV summary statistics.

        Args:
            days: If provided, limit analysis to last N days.

        Returns:
            Dict with keys:
                total_bets (int): Total bets with CLV data
                settled_bets (int): Bets with a result
                avg_clv (float): Mean CLV across all bets with closing odds
                avg_clv_7d (float): Mean CLV over last 7 days
                avg_clv_30d (float): Mean CLV over last 30 days
                median_clv (float): Median CLV
                positive_clv_rate (float): Fraction of bets with positive CLV
                clv_by_prop_type (dict): avg CLV keyed by prop type
                clv_by_direction (dict): avg CLV for 'over' vs 'under'
                win_rate_positive_clv (float): win rate when CLV > 0
                win_rate_negative_clv (float): win rate when CLV < 0
                sharp_rating (str): 'sharp', 'marginal', 'not_sharp', or 'insufficient_data'
        """
        empty_result = {
            "total_bets": 0,
            "settled_bets": 0,
            "avg_clv": 0.0,
            "avg_clv_7d": 0.0,
            "avg_clv_30d": 0.0,
            "median_clv": 0.0,
            "positive_clv_rate": 0.0,
            "clv_by_prop_type": {},
            "clv_by_direction": {},
            "win_rate_positive_clv": 0.0,
            "win_rate_negative_clv": 0.0,
            "sharp_rating": "insufficient_data",
        }

        bets = self._fetch_bets_with_clv(days=days)
        if not bets:
            return empty_result

        total_bets = len(bets)
        clv_values = [b["clv"] for b in bets]

        # Settled bets
        settled = [b for b in bets if self._is_settled(b["status"])]
        settled_bets = len(settled)

        # Overall CLV stats
        avg_clv = sum(clv_values) / total_bets if total_bets > 0 else 0.0
        median_clv = median(clv_values) if clv_values else 0.0
        positive_count = sum(1 for c in clv_values if c > 0)
        positive_clv_rate = positive_count / total_bets if total_bets > 0 else 0.0

        # 7-day and 30-day rolling CLV
        bets_7d = self._fetch_bets_with_clv(days=7)
        clv_7d = [b["clv"] for b in bets_7d]
        avg_clv_7d = sum(clv_7d) / len(clv_7d) if clv_7d else 0.0

        bets_30d = self._fetch_bets_with_clv(days=30)
        clv_30d = [b["clv"] for b in bets_30d]
        avg_clv_30d = sum(clv_30d) / len(clv_30d) if clv_30d else 0.0

        # CLV by prop type
        clv_by_prop = {}
        for b in bets:
            prop_type = self._extract_prop_type(b["tags"])
            if prop_type not in clv_by_prop:
                clv_by_prop[prop_type] = []
            clv_by_prop[prop_type].append(b["clv"])
        clv_by_prop_type = {
            k: sum(v) / len(v) for k, v in clv_by_prop.items() if v
        }

        # CLV by direction
        clv_by_dir = {}
        for b in bets:
            direction = self._extract_direction(b["tags"], b["selection"])
            if direction not in clv_by_dir:
                clv_by_dir[direction] = []
            clv_by_dir[direction].append(b["clv"])
        clv_by_direction = {
            k: sum(v) / len(v) for k, v in clv_by_dir.items() if v
        }

        # Win rate conditioned on CLV sign
        positive_clv_settled = [
            b for b in settled if b["clv"] > 0
        ]
        negative_clv_settled = [
            b for b in settled if b["clv"] <= 0
        ]
        positive_wins = sum(1 for b in positive_clv_settled if self._is_win(b["status"]))
        negative_wins = sum(1 for b in negative_clv_settled if self._is_win(b["status"]))
        win_rate_positive_clv = (
            positive_wins / len(positive_clv_settled)
            if positive_clv_settled else 0.0
        )
        win_rate_negative_clv = (
            negative_wins / len(negative_clv_settled)
            if negative_clv_settled else 0.0
        )

        # Sharp rating
        _, sharp_rating = self._compute_sharp_rating(avg_clv, total_bets)

        return {
            "total_bets": total_bets,
            "settled_bets": settled_bets,
            "avg_clv": round(avg_clv, 6),
            "avg_clv_7d": round(avg_clv_7d, 6),
            "avg_clv_30d": round(avg_clv_30d, 6),
            "median_clv": round(median_clv, 6),
            "positive_clv_rate": round(positive_clv_rate, 4),
            "clv_by_prop_type": {k: round(v, 6) for k, v in clv_by_prop_type.items()},
            "clv_by_direction": {k: round(v, 6) for k, v in clv_by_direction.items()},
            "win_rate_positive_clv": round(win_rate_positive_clv, 4),
            "win_rate_negative_clv": round(win_rate_negative_clv, 4),
            "sharp_rating": sharp_rating,
        }

    def _compute_sharp_rating(self, avg_clv: float, total_bets: int) -> tuple:
        """Compute sharp rating based on CLV and sample size.

        Args:
            avg_clv: Average CLV value.
            total_bets: Number of bets analyzed.

        Returns:
            Tuple of (is_sharp: bool, rating: str).
        """
        if total_bets < 100:
            return False, "insufficient_data"
        if avg_clv > 0.01:
            return True, "sharp"
        if avg_clv > 0:
            return False, "marginal"
        return False, "not_sharp"

    def get_clv_timeseries(self, days: int = 30) -> list:
        """Return daily aggregated CLV values for charting.

        Args:
            days: Number of days to include.

        Returns:
            List of dicts with keys: date, avg_clv, count, cumulative_clv
        """
        bets = self._fetch_bets_with_clv(days=days)
        if not bets:
            return []

        # Group by date
        daily = {}
        for b in bets:
            placed = b["placed_at"]
            if not placed:
                continue
            # Extract date portion (handle both ISO datetime and date-only)
            date_str = placed[:10]
            if date_str not in daily:
                daily[date_str] = []
            daily[date_str].append(b["clv"])

        # Sort by date and compute cumulative
        sorted_dates = sorted(daily.keys())
        result = []
        cumulative = 0.0
        for date_str in sorted_dates:
            day_clvs = daily[date_str]
            avg_clv = sum(day_clvs) / len(day_clvs) if day_clvs else 0.0
            cumulative += sum(day_clvs)
            result.append({
                "date": date_str,
                "avg_clv": round(avg_clv, 6),
                "count": len(day_clvs),
                "cumulative_clv": round(cumulative, 6),
            })

        return result

    def is_model_sharp(self) -> tuple:
        """Determine if the model is genuinely sharp based on CLV data.

        Rules:
        - avg_clv > 0.01 over 100+ bets = sharp
        - avg_clv > 0 but < 0.01 over 100+ bets = marginal
        - avg_clv <= 0 over 100+ bets = not sharp
        - Less than 100 bets = insufficient data

        Returns:
            Tuple of (is_sharp: bool, explanation: str)
        """
        bets = self._fetch_bets_with_clv()
        total = len(bets)

        if total < 100:
            return (
                False,
                f"Insufficient data: only {total} bets with CLV data (need 100+)",
            )

        clv_values = [b["clv"] for b in bets]
        avg_clv = sum(clv_values) / total
        positive_rate = sum(1 for c in clv_values if c > 0) / total

        if avg_clv > 0.01:
            return (
                True,
                f"Sharp: avg CLV of {avg_clv:.4f} over {total} bets "
                f"({positive_rate * 100:.1f}% positive CLV rate)",
            )

        if avg_clv > 0:
            return (
                False,
                f"Marginal: avg CLV of {avg_clv:.4f} over {total} bets "
                f"-- positive but below 1% threshold "
                f"({positive_rate * 100:.1f}% positive CLV rate)",
            )

        return (
            False,
            f"Not sharp: avg CLV of {avg_clv:.4f} over {total} bets "
            f"-- not consistently beating closing lines "
            f"({positive_rate * 100:.1f}% positive CLV rate)",
        )
