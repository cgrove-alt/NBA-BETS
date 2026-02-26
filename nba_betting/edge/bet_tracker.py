"""
Bet Tracking System

Comprehensive bet tracking and performance analysis for NBA betting.
Records all bets, tracks outcomes, calculates performance metrics,
and provides insights for continuous improvement.

Features:
1. Bet recording with full metadata
2. Automatic outcome tracking
3. ROI and P&L calculations
4. CLV (Closing Line Value) tracking
5. Performance by bet type, sport, sportsbook
6. Streak tracking and bankroll management
7. Export capabilities

Supports PostgreSQL (primary, via Railway) with SQLite fallback.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np

# PostgreSQL connection (optional — falls back to SQLite)
try:
    from agents.core.connections import get_postgres_connection
except (ImportError, TypeError):
    def get_postgres_connection():
        return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import portfolio optimizer for covariance-aware bet sizing
try:
    from portfolio_optimizer import (
        PortfolioOptimizer,
        BetType as PortfolioBetType,
        calculate_covariance,
        optimize_portfolio_kelly,
    )
    HAS_PORTFOLIO_OPTIMIZER = True
except (ImportError, TypeError):
    HAS_PORTFOLIO_OPTIMIZER = False
    logger.info("Portfolio optimizer not available. Bet sizing will use simple Kelly.")


class BetStatus(Enum):
    """Bet status states."""
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    PUSH = "push"
    VOID = "void"
    CASHED_OUT = "cashed_out"


class BetType(Enum):
    """Types of bets."""
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"
    PARLAY = "parlay"
    TEASER = "teaser"
    FUTURES = "futures"


@dataclass
class TrackedBet:
    """Complete bet record with all metadata."""
    bet_id: str
    placed_at: datetime
    sport: str = "NBA"
    bet_type: BetType = BetType.MONEYLINE
    sportsbook: str = ""

    # Event info
    event_id: str = ""
    event_name: str = ""  # e.g., "Lakers vs Celtics"
    event_date: datetime | None = None

    # Bet details
    selection: str = ""  # e.g., "Lakers ML", "Over 220.5"
    odds: float = -110  # American odds
    stake: float = 0.0
    potential_payout: float = 0.0

    # Model predictions
    model_probability: float = 0.5
    implied_probability: float = 0.5
    edge: float = 0.0

    # Market data
    opening_odds: float | None = None
    closing_odds: float | None = None
    line_movement: float = 0.0  # How much the line moved

    # Outcome
    status: BetStatus = BetStatus.PENDING
    actual_result: str | None = None  # e.g., "Lakers 112-105"
    pnl: float = 0.0
    settled_at: datetime | None = None

    # Additional metadata
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    parlay_legs: list[dict] = field(default_factory=list)  # For parlay bets

    def to_dict(self) -> dict:
        d = asdict(self)
        d['bet_type'] = self.bet_type.value
        d['status'] = self.status.value
        d['placed_at'] = self.placed_at.isoformat() if self.placed_at else None
        d['event_date'] = self.event_date.isoformat() if self.event_date else None
        d['settled_at'] = self.settled_at.isoformat() if self.settled_at else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrackedBet":
        d = d.copy()
        d['bet_type'] = BetType(d['bet_type']) if isinstance(d['bet_type'], str) else d['bet_type']
        d['status'] = BetStatus(d['status']) if isinstance(d['status'], str) else d['status']
        d['placed_at'] = datetime.fromisoformat(d['placed_at']) if d.get('placed_at') else datetime.now()
        d['event_date'] = datetime.fromisoformat(d['event_date']) if d.get('event_date') else None
        d['settled_at'] = datetime.fromisoformat(d['settled_at']) if d.get('settled_at') else None
        d['tags'] = d.get('tags', [])
        d['parlay_legs'] = d.get('parlay_legs', [])
        return cls(**d)

    def calculate_pnl(self) -> float:
        """Calculate P&L based on status and odds."""
        if self.status == BetStatus.WON:
            if self.odds > 0:
                self.pnl = self.stake * (self.odds / 100)
            else:
                self.pnl = self.stake * (100 / abs(self.odds))
        elif self.status == BetStatus.LOST:
            self.pnl = -self.stake
        elif self.status in [BetStatus.PUSH, BetStatus.VOID]:
            self.pnl = 0.0
        elif self.status == BetStatus.CASHED_OUT:
            # pnl should be set manually for cash out
            pass
        return self.pnl

    def closing_line_value(self) -> float | None:
        """
        Calculate Closing Line Value.

        CLV = Closing implied probability - Our implied probability

        Positive CLV means we beat the closing line.
        """
        if self.closing_odds is None:
            return None

        our_implied = self.implied_probability
        closing_implied = self._odds_to_prob(self.closing_odds)

        return closing_implied - our_implied

    @staticmethod
    def _odds_to_prob(american_odds: float) -> float:
        """Convert American odds to implied probability."""
        if american_odds >= 100:
            return 100 / (american_odds + 100)
        return abs(american_odds) / (abs(american_odds) + 100)


@dataclass
class PerformanceMetrics:
    """Performance metrics over a time period."""
    period_start: datetime
    period_end: datetime
    total_bets: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pushes: int = 0
    total_staked: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    avg_odds: float = 0.0
    avg_stake: float = 0.0
    avg_edge: float = 0.0
    avg_clv: float | None = None
    max_win: float = 0.0
    max_loss: float = 0.0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_bets": self.total_bets,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "total_pushes": self.total_pushes,
            "total_staked": self.total_staked,
            "total_pnl": self.total_pnl,
            "roi": self.roi,
            "win_rate": self.win_rate,
            "avg_odds": self.avg_odds,
            "avg_stake": self.avg_stake,
            "avg_edge": self.avg_edge,
            "avg_clv": self.avg_clv,
            "max_win": self.max_win,
            "max_loss": self.max_loss,
            "longest_win_streak": self.longest_win_streak,
            "longest_loss_streak": self.longest_loss_streak,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Performance Summary ({self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')})
{'=' * 60}
Record: {self.total_wins}W - {self.total_losses}L - {self.total_pushes}P ({self.total_bets} total)
Win Rate: {self.win_rate:.1f}%
Total Staked: ${self.total_staked:,.2f}
Total P&L: ${self.total_pnl:+,.2f}
ROI: {self.roi:+.2f}%

Avg Stake: ${self.avg_stake:.2f}
Avg Odds: {self.avg_odds:+.0f}
Avg Edge: {self.avg_edge:.2f}%
Avg CLV: {self.avg_clv:.4f if self.avg_clv else 'N/A'}

Best Win: ${self.max_win:,.2f}
Worst Loss: ${self.max_loss:,.2f}
Best Streak: {self.longest_win_streak}W
Worst Streak: {self.longest_loss_streak}L

Profit Factor: {self.profit_factor:.2f}
Sharpe Ratio: {self.sharpe_ratio:.2f}
{'=' * 60}
"""


# Column order for the PostgreSQL tracked_bets table, matching the migration schema.
_PG_COLUMNS = [
    'bet_id', 'placed_at', 'sport', 'bet_type', 'sportsbook',
    'event_id', 'event_name', 'event_date', 'selection', 'odds',
    'stake', 'potential_payout', 'model_probability', 'implied_probability',
    'edge', 'opening_odds', 'closing_odds', 'line_movement', 'status',
    'actual_result', 'pnl', 'settled_at', 'notes', 'tags', 'parlay_legs',
    'created_at',
]


class BetTracker:
    """
    Main bet tracking system with PostgreSQL primary / SQLite fallback.

    Provides comprehensive bet recording, tracking, and analysis.
    """

    def __init__(self, db_path: str = "bets.db", pg_conn=None):
        """
        Initialize bet tracker.

        Tries PostgreSQL first (via pg_conn or get_postgres_connection()).
        Falls back to SQLite at db_path if PG is unavailable.

        Args:
            db_path: Path to SQLite database (fallback)
            pg_conn: Optional existing psycopg2 connection
        """
        self.db_path = db_path
        self._use_postgres = False
        self._pg_conn = None

        # Try PostgreSQL first
        conn = pg_conn or get_postgres_connection()
        if conn is not None:
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'tracked_bets');"
                )
                exists = cur.fetchone()[0]
                cur.close()
                if exists:
                    self._use_postgres = True
                    self._pg_conn = conn
                    logger.info("BetTracker: using PostgreSQL (tracked_bets table)")
                else:
                    logger.warning("BetTracker: tracked_bets table not found in PostgreSQL, falling back to SQLite")
                    self._init_database()
            except Exception as e:
                logger.warning(f"BetTracker: PostgreSQL probe failed ({e}), falling back to SQLite")
                self._init_database()
        else:
            self._init_database()

    # ------------------------------------------------------------------
    # Database initialisation (SQLite only — PG uses migrations)
    # ------------------------------------------------------------------

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
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

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bets_placed_at ON bets(placed_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bets_bet_type ON bets(bet_type)
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    # ------------------------------------------------------------------
    # record_bet
    # ------------------------------------------------------------------

    def record_bet(self, bet: TrackedBet) -> str:
        """
        Record a new bet.

        Args:
            bet: TrackedBet object

        Returns:
            bet_id
        """
        # Calculate potential payout
        if bet.odds > 0:
            bet.potential_payout = bet.stake + bet.stake * (bet.odds / 100)
        else:
            bet.potential_payout = bet.stake + bet.stake * (100 / abs(bet.odds))

        # Calculate implied probability (skip if already set with devigged value)
        raw_implied = bet._odds_to_prob(bet.odds)
        if bet.implied_probability == 0.5 and bet.edge == 0.0:
            bet.implied_probability = raw_implied
            bet.edge = bet.model_probability - bet.implied_probability

        if self._use_postgres:
            self._record_bet_pg(bet)
        else:
            self._record_bet_sqlite(bet)

        logger.info(f"Recorded bet {bet.bet_id}: {bet.selection} @ {bet.odds} for ${bet.stake}")
        return bet.bet_id

    def _record_bet_pg(self, bet: TrackedBet) -> None:
        """Insert or upsert a bet into PostgreSQL tracked_bets."""
        cur = self._pg_conn.cursor()
        cur.execute("""
            INSERT INTO tracked_bets (
                bet_id, placed_at, sport, bet_type, sportsbook,
                event_id, event_name, event_date, selection, odds,
                stake, potential_payout, model_probability, implied_probability,
                edge, opening_odds, closing_odds, line_movement, status,
                actual_result, pnl, settled_at, notes, tags, parlay_legs
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (bet_id) DO UPDATE SET
                placed_at = EXCLUDED.placed_at,
                sport = EXCLUDED.sport,
                bet_type = EXCLUDED.bet_type,
                sportsbook = EXCLUDED.sportsbook,
                event_id = EXCLUDED.event_id,
                event_name = EXCLUDED.event_name,
                event_date = EXCLUDED.event_date,
                selection = EXCLUDED.selection,
                odds = EXCLUDED.odds,
                stake = EXCLUDED.stake,
                potential_payout = EXCLUDED.potential_payout,
                model_probability = EXCLUDED.model_probability,
                implied_probability = EXCLUDED.implied_probability,
                edge = EXCLUDED.edge,
                opening_odds = EXCLUDED.opening_odds,
                closing_odds = EXCLUDED.closing_odds,
                line_movement = EXCLUDED.line_movement,
                status = EXCLUDED.status,
                actual_result = EXCLUDED.actual_result,
                pnl = EXCLUDED.pnl,
                settled_at = EXCLUDED.settled_at,
                notes = EXCLUDED.notes,
                tags = EXCLUDED.tags,
                parlay_legs = EXCLUDED.parlay_legs
        """, (
            bet.bet_id,
            bet.placed_at,
            bet.sport,
            bet.bet_type.value,
            bet.sportsbook,
            bet.event_id,
            bet.event_name,
            bet.event_date,
            bet.selection,
            bet.odds,
            bet.stake,
            bet.potential_payout,
            bet.model_probability,
            bet.implied_probability,
            bet.edge,
            bet.opening_odds,
            bet.closing_odds,
            bet.line_movement,
            bet.status.value,
            bet.actual_result,
            bet.pnl,
            bet.settled_at,
            bet.notes,
            json.dumps(bet.tags),
            json.dumps(bet.parlay_legs),
        ))
        cur.close()

    def _record_bet_sqlite(self, bet: TrackedBet) -> None:
        """Insert or replace a bet in SQLite bets table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO bets (
                bet_id, placed_at, sport, bet_type, sportsbook,
                event_id, event_name, event_date, selection, odds,
                stake, potential_payout, model_probability, implied_probability,
                edge, opening_odds, closing_odds, line_movement, status,
                actual_result, pnl, settled_at, notes, tags, parlay_legs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bet.bet_id,
            bet.placed_at.isoformat(),
            bet.sport,
            bet.bet_type.value,
            bet.sportsbook,
            bet.event_id,
            bet.event_name,
            bet.event_date.isoformat() if bet.event_date else None,
            bet.selection,
            bet.odds,
            bet.stake,
            bet.potential_payout,
            bet.model_probability,
            bet.implied_probability,
            bet.edge,
            bet.opening_odds,
            bet.closing_odds,
            bet.line_movement,
            bet.status.value,
            bet.actual_result,
            bet.pnl,
            bet.settled_at.isoformat() if bet.settled_at else None,
            bet.notes,
            json.dumps(bet.tags),
            json.dumps(bet.parlay_legs),
        ))

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # settle_bet
    # ------------------------------------------------------------------

    def settle_bet(
        self,
        bet_id: str,
        status: Union[BetStatus, str],
        actual_result: str = None,
        closing_odds: float = None
    ) -> TrackedBet:
        """
        Settle a bet with outcome.

        Args:
            bet_id: Bet identifier
            status: Won, lost, push, etc.
            actual_result: Description of actual result
            closing_odds: Closing line odds for CLV

        Returns:
            Updated TrackedBet
        """
        if isinstance(status, str):
            status = BetStatus(status)

        bet = self.get_bet(bet_id)
        if not bet:
            raise ValueError(f"Bet not found: {bet_id}")

        bet.status = status
        bet.actual_result = actual_result
        bet.settled_at = datetime.now()
        bet.closing_odds = closing_odds

        # Calculate P&L
        bet.calculate_pnl()

        # Update line movement if closing odds provided
        if closing_odds and bet.opening_odds:
            bet.line_movement = closing_odds - bet.opening_odds

        if self._use_postgres:
            self._settle_bet_pg(bet)
        else:
            self._settle_bet_sqlite(bet)

        logger.info(f"Settled bet {bet_id}: {status.value}, P&L: ${bet.pnl:+.2f}")
        return bet

    def _settle_bet_pg(self, bet: TrackedBet) -> None:
        """Update a settled bet in PostgreSQL tracked_bets."""
        cur = self._pg_conn.cursor()
        cur.execute("""
            UPDATE tracked_bets SET
                status = %s,
                actual_result = %s,
                pnl = %s,
                settled_at = %s,
                closing_odds = %s,
                line_movement = %s
            WHERE bet_id = %s
        """, (
            bet.status.value,
            bet.actual_result,
            bet.pnl,
            bet.settled_at,
            bet.closing_odds,
            bet.line_movement,
            bet.bet_id,
        ))
        cur.close()

    def _settle_bet_sqlite(self, bet: TrackedBet) -> None:
        """Update a settled bet in SQLite bets table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE bets SET
                status = ?,
                actual_result = ?,
                pnl = ?,
                settled_at = ?,
                closing_odds = ?,
                line_movement = ?
            WHERE bet_id = ?
        """, (
            bet.status.value,
            bet.actual_result,
            bet.pnl,
            bet.settled_at.isoformat(),
            bet.closing_odds,
            bet.line_movement,
            bet.bet_id
        ))

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # get_bet
    # ------------------------------------------------------------------

    def get_bet(self, bet_id: str) -> TrackedBet | None:
        """Get a single bet by ID."""
        if self._use_postgres:
            return self._get_bet_pg(bet_id)
        return self._get_bet_sqlite(bet_id)

    def _get_bet_pg(self, bet_id: str) -> TrackedBet | None:
        """Fetch a single bet from PostgreSQL tracked_bets."""
        cur = self._pg_conn.cursor()
        cur.execute("SELECT * FROM tracked_bets WHERE bet_id = %s", (bet_id,))
        row = cur.fetchone()
        if row is None:
            cur.close()
            return None
        columns = [desc[0] for desc in cur.description]
        cur.close()
        return self._row_to_bet_pg(columns, row)

    def _get_bet_sqlite(self, bet_id: str) -> TrackedBet | None:
        """Fetch a single bet from SQLite bets table."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM bets WHERE bet_id = ?", (bet_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_bet(dict(row))
        return None

    # ------------------------------------------------------------------
    # query helpers
    # ------------------------------------------------------------------

    def get_pending_bets(self) -> list[TrackedBet]:
        """Get all pending (unsettled) bets."""
        return self._query_bets("status = ?", (BetStatus.PENDING.value,))

    def get_bets_by_date(
        self,
        start_date: datetime,
        end_date: datetime = None
    ) -> list[TrackedBet]:
        """Get bets within date range."""
        if end_date is None:
            end_date = datetime.now()

        return self._query_bets(
            "placed_at >= ? AND placed_at <= ?",
            (start_date.isoformat(), end_date.isoformat())
        )

    def get_bets_by_type(self, bet_type: BetType) -> list[TrackedBet]:
        """Get bets of specific type."""
        return self._query_bets("bet_type = ?", (bet_type.value,))

    def get_bets_by_sportsbook(self, sportsbook: str) -> list[TrackedBet]:
        """Get bets from specific sportsbook."""
        return self._query_bets("sportsbook = ?", (sportsbook,))

    def _query_bets(self, where_clause: str, params: tuple) -> list[TrackedBet]:
        """
        Query bets with a WHERE clause.

        The where_clause should use '?' placeholders (SQLite style). When
        running against PostgreSQL the placeholders and table name are
        automatically translated.
        """
        if self._use_postgres:
            return self._query_bets_pg(where_clause, params)
        return self._query_bets_sqlite(where_clause, params)

    def _query_bets_pg(self, where_clause: str, params: tuple) -> list[TrackedBet]:
        """Run a SELECT on PostgreSQL tracked_bets."""
        # Translate SQLite '?' placeholders to PG '%s'
        pg_where = where_clause.replace('?', '%s')
        cur = self._pg_conn.cursor()
        cur.execute(
            f"SELECT * FROM tracked_bets WHERE {pg_where} ORDER BY placed_at DESC",
            params,
        )
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        return [self._row_to_bet_pg(columns, row) for row in rows]

    def _query_bets_sqlite(self, where_clause: str, params: tuple) -> list[TrackedBet]:
        """Run a SELECT on SQLite bets table."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM bets WHERE {where_clause} ORDER BY placed_at DESC", params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_bet(dict(row)) for row in rows]

    # ------------------------------------------------------------------
    # row-to-bet conversion
    # ------------------------------------------------------------------

    def _row_to_bet_pg(self, columns: list[str], row: tuple) -> TrackedBet:
        """Convert a PostgreSQL row (tuple + column names) to TrackedBet."""
        d = dict(zip(columns, row, strict=False))

        # PG returns datetime objects directly for TIMESTAMP columns, but
        # placed_at / event_date / settled_at may also come back as strings
        # if the connection doesn't auto-cast.  Normalise them.
        placed_at = d.get('placed_at')
        if isinstance(placed_at, str):
            placed_at = datetime.fromisoformat(placed_at)

        event_date = d.get('event_date')
        if isinstance(event_date, str):
            event_date = datetime.fromisoformat(event_date)

        settled_at = d.get('settled_at')
        if isinstance(settled_at, str):
            settled_at = datetime.fromisoformat(settled_at)

        # PG JSONB columns come back as native Python objects (list/dict)
        tags_raw = d.get('tags')
        if isinstance(tags_raw, str):
            tags_raw = json.loads(tags_raw) if tags_raw else []
        elif tags_raw is None:
            tags_raw = []

        parlay_raw = d.get('parlay_legs')
        if isinstance(parlay_raw, str):
            parlay_raw = json.loads(parlay_raw) if parlay_raw else []
        elif parlay_raw is None:
            parlay_raw = []

        return TrackedBet(
            bet_id=d['bet_id'],
            placed_at=placed_at,
            sport=d.get('sport') or 'NBA',
            bet_type=BetType(d['bet_type']),
            sportsbook=d.get('sportsbook') or '',
            event_id=d.get('event_id') or '',
            event_name=d.get('event_name') or '',
            event_date=event_date,
            selection=d['selection'],
            odds=d['odds'],
            stake=d['stake'],
            potential_payout=d.get('potential_payout') or 0.0,
            model_probability=d.get('model_probability') or 0.5,
            implied_probability=d.get('implied_probability') or 0.5,
            edge=d.get('edge') or 0.0,
            opening_odds=d.get('opening_odds'),
            closing_odds=d.get('closing_odds'),
            line_movement=d.get('line_movement') or 0.0,
            status=BetStatus(d['status']),
            actual_result=d.get('actual_result'),
            pnl=d.get('pnl') or 0.0,
            settled_at=settled_at,
            notes=d.get('notes') or '',
            tags=tags_raw,
            parlay_legs=parlay_raw,
        )

    def _row_to_bet(self, row: dict) -> TrackedBet:
        """Convert a SQLite Row dict to TrackedBet."""
        return TrackedBet(
            bet_id=row['bet_id'],
            placed_at=datetime.fromisoformat(row['placed_at']),
            sport=row['sport'] or 'NBA',
            bet_type=BetType(row['bet_type']),
            sportsbook=row['sportsbook'] or '',
            event_id=row['event_id'] or '',
            event_name=row['event_name'] or '',
            event_date=datetime.fromisoformat(row['event_date']) if row['event_date'] else None,
            selection=row['selection'],
            odds=row['odds'],
            stake=row['stake'],
            potential_payout=row['potential_payout'] or 0.0,
            model_probability=row['model_probability'] or 0.5,
            implied_probability=row['implied_probability'] or 0.5,
            edge=row['edge'] or 0.0,
            opening_odds=row['opening_odds'],
            closing_odds=row['closing_odds'],
            line_movement=row['line_movement'] or 0.0,
            status=BetStatus(row['status']),
            actual_result=row['actual_result'],
            pnl=row['pnl'] or 0.0,
            settled_at=datetime.fromisoformat(row['settled_at']) if row['settled_at'] else None,
            notes=row['notes'] or '',
            tags=json.loads(row['tags']) if row['tags'] else [],
            parlay_legs=json.loads(row['parlay_legs']) if row['parlay_legs'] else [],
        )

    # ------------------------------------------------------------------
    # Performance analytics
    # ------------------------------------------------------------------

    def calculate_performance(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        bet_type: BetType = None,
        sportsbook: str = None
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics for given period and filters.

        Args:
            start_date: Start of period (default: 30 days ago)
            end_date: End of period (default: now)
            bet_type: Filter by bet type
            sportsbook: Filter by sportsbook

        Returns:
            PerformanceMetrics object
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        # Build query
        conditions = ["placed_at >= ?", "placed_at <= ?", "status != 'pending'"]
        params = [start_date.isoformat(), end_date.isoformat()]

        if bet_type:
            conditions.append("bet_type = ?")
            params.append(bet_type.value)
        if sportsbook:
            conditions.append("sportsbook = ?")
            params.append(sportsbook)

        bets = self._query_bets(" AND ".join(conditions), tuple(params))

        # Calculate metrics
        metrics = PerformanceMetrics(period_start=start_date, period_end=end_date)

        if not bets:
            return metrics

        metrics.total_bets = len(bets)
        metrics.total_wins = sum(1 for b in bets if b.status == BetStatus.WON)
        metrics.total_losses = sum(1 for b in bets if b.status == BetStatus.LOST)
        metrics.total_pushes = sum(1 for b in bets if b.status == BetStatus.PUSH)

        metrics.total_staked = sum(b.stake for b in bets)
        metrics.total_pnl = sum(b.pnl for b in bets)

        if metrics.total_staked > 0:
            metrics.roi = (metrics.total_pnl / metrics.total_staked) * 100

        decisions = metrics.total_wins + metrics.total_losses
        if decisions > 0:
            metrics.win_rate = (metrics.total_wins / decisions) * 100

        metrics.avg_odds = np.mean([b.odds for b in bets])
        metrics.avg_stake = np.mean([b.stake for b in bets])
        metrics.avg_edge = np.mean([b.edge for b in bets if b.edge]) * 100

        # CLV
        clv_values = [b.closing_line_value() for b in bets if b.closing_line_value() is not None]
        if clv_values:
            metrics.avg_clv = np.mean(clv_values)

        # Max win/loss
        pnls = [b.pnl for b in bets if b.pnl != 0]
        if pnls:
            metrics.max_win = max(pnls) if max(pnls) > 0 else 0
            metrics.max_loss = min(pnls) if min(pnls) < 0 else 0

        # Streaks
        metrics.longest_win_streak, metrics.longest_loss_streak = self._calculate_streaks(bets)

        # Profit factor
        wins_pnl = sum(b.pnl for b in bets if b.pnl > 0)
        losses_pnl = abs(sum(b.pnl for b in bets if b.pnl < 0))
        if losses_pnl > 0:
            metrics.profit_factor = wins_pnl / losses_pnl

        # Sharpe ratio (simplified daily)
        metrics.sharpe_ratio = self._calculate_sharpe(bets)

        return metrics

    def _calculate_streaks(self, bets: list[TrackedBet]) -> tuple[int, int]:
        """Calculate longest winning and losing streaks."""
        max_win, max_loss = 0, 0
        current_win, current_loss = 0, 0

        for bet in sorted(bets, key=lambda x: x.placed_at):
            if bet.status == BetStatus.WON:
                current_win += 1
                current_loss = 0
                max_win = max(max_win, current_win)
            elif bet.status == BetStatus.LOST:
                current_loss += 1
                current_win = 0
                max_loss = max(max_loss, current_loss)

        return max_win, max_loss

    def _calculate_sharpe(self, bets: list[TrackedBet]) -> float:
        """Calculate Sharpe ratio."""
        daily_pnl: dict[str, float] = {}
        for bet in bets:
            day = bet.placed_at.strftime("%Y-%m-%d")
            daily_pnl[day] = daily_pnl.get(day, 0) + bet.pnl

        if len(daily_pnl) < 2:
            return 0.0

        returns = list(daily_pnl.values())
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return > 0:
            return (mean_return / std_return) * np.sqrt(252)  # Annualized
        return 0.0

    def get_bankroll_history(
        self,
        start_date: datetime = None,
        initial_bankroll: float = 10000.0
    ) -> list[tuple[datetime, float]]:
        """
        Get bankroll over time.

        Args:
            start_date: Start tracking from this date
            initial_bankroll: Starting bankroll amount

        Returns:
            List of (datetime, bankroll) tuples
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)

        bets = self.get_bets_by_date(start_date)
        bets = [b for b in bets if b.status != BetStatus.PENDING]
        bets.sort(key=lambda x: x.settled_at or x.placed_at)

        history = [(start_date, initial_bankroll)]
        bankroll = initial_bankroll

        for bet in bets:
            bankroll += bet.pnl
            timestamp = bet.settled_at or bet.placed_at
            history.append((timestamp, bankroll))

        return history

    def export_to_csv(self, filepath: str, start_date: datetime = None) -> str:
        """Export bets to CSV file."""
        import csv

        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)

        bets = self.get_bets_by_date(start_date)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'bet_id', 'placed_at', 'sport', 'bet_type', 'sportsbook',
                'event_name', 'selection', 'odds', 'stake', 'model_probability',
                'implied_probability', 'edge', 'status', 'pnl', 'closing_odds',
                'clv', 'notes'
            ])
            writer.writeheader()

            for bet in bets:
                writer.writerow({
                    'bet_id': bet.bet_id,
                    'placed_at': bet.placed_at.isoformat(),
                    'sport': bet.sport,
                    'bet_type': bet.bet_type.value,
                    'sportsbook': bet.sportsbook,
                    'event_name': bet.event_name,
                    'selection': bet.selection,
                    'odds': bet.odds,
                    'stake': bet.stake,
                    'model_probability': bet.model_probability,
                    'implied_probability': bet.implied_probability,
                    'edge': bet.edge,
                    'status': bet.status.value,
                    'pnl': bet.pnl,
                    'closing_odds': bet.closing_odds,
                    'clv': bet.closing_line_value(),
                    'notes': bet.notes,
                })

        logger.info(f"Exported {len(bets)} bets to {filepath}")
        return filepath

    def get_performance_by_edge_range(self) -> dict[str, PerformanceMetrics]:
        """Analyze performance grouped by edge ranges."""
        edge_ranges = [
            ("0-2%", 0, 0.02),
            ("2-5%", 0.02, 0.05),
            ("5-10%", 0.05, 0.10),
            ("10%+", 0.10, 1.0),
        ]

        results = {}
        for name, min_edge, max_edge in edge_ranges:
            bets = self._query_bets(
                "edge >= ? AND edge < ? AND status != 'pending'",
                (min_edge, max_edge)
            )
            if bets:
                metrics = PerformanceMetrics(
                    period_start=min(b.placed_at for b in bets),
                    period_end=max(b.placed_at for b in bets)
                )
                metrics.total_bets = len(bets)
                metrics.total_wins = sum(1 for b in bets if b.status == BetStatus.WON)
                metrics.total_losses = sum(1 for b in bets if b.status == BetStatus.LOST)
                metrics.total_staked = sum(b.stake for b in bets)
                metrics.total_pnl = sum(b.pnl for b in bets)
                if metrics.total_staked > 0:
                    metrics.roi = (metrics.total_pnl / metrics.total_staked) * 100
                decisions = metrics.total_wins + metrics.total_losses
                if decisions > 0:
                    metrics.win_rate = (metrics.total_wins / decisions) * 100
                results[name] = metrics

        return results

    def optimize_pending_stakes(
        self,
        bankroll: float = 1000,
        min_edge: float = 0.02
    ) -> dict:
        """
        Optimize stake sizing for pending bets using portfolio optimization.

        Uses covariance-aware Kelly criterion to size bets accounting for
        correlations between same-game bets.

        Args:
            bankroll: Total bankroll for sizing
            min_edge: Minimum edge required (default 2%)

        Returns:
            Dictionary with optimized stakes for each bet
        """
        if not HAS_PORTFOLIO_OPTIMIZER:
            logger.warning("Portfolio optimizer not available")
            return {}

        # Get pending bets
        pending_bets = self.get_pending_bets()
        if not pending_bets:
            return {'bets': [], 'total_stake': 0, 'message': 'No pending bets'}

        # Convert to format for optimizer
        bets_for_optimizer = []
        for bet in pending_bets:
            if bet.edge < min_edge:
                continue

            bets_for_optimizer.append({
                'bet_id': bet.bet_id,
                'game_id': bet.event_id,
                'bet_type': bet.bet_type.value,
                'selection': bet.selection,
                'odds': int(bet.odds),
                'probability': bet.model_probability,
                'edge': bet.edge,
                'team': bet.event_name.split(' vs ')[0] if ' vs ' in bet.event_name else None,
                'side': 'home' if 'home' in bet.selection.lower() else (
                    'away' if 'away' in bet.selection.lower() else (
                        'over' if 'over' in bet.selection.lower() else 'under'
                    )
                ),
            })

        if not bets_for_optimizer:
            return {
                'bets': [],
                'total_stake': 0,
                'message': f'No bets with edge >= {min_edge:.1%}'
            }

        try:
            # Run optimization
            result = optimize_portfolio_kelly(
                bets_for_optimizer,
                bankroll=bankroll
            )

            # Map back to bet IDs
            optimized = {
                'bets': [],
                'total_stake': result.get('total_stake', 0),
                'expected_return': result.get('expected_return', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
            }

            for opt_bet in result.get('bets', []):
                if opt_bet.get('final_stake', 0) > 0:
                    optimized['bets'].append({
                        'selection': opt_bet['selection'],
                        'recommended_stake': opt_bet['final_stake'],
                        'kelly_fraction': opt_bet['kelly_fraction'],
                        'edge': opt_bet['edge'],
                        'odds': opt_bet['odds'],
                    })

            return optimized

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {'bets': [], 'total_stake': 0, 'message': str(e)}

    def get_correlation_matrix(self) -> np.ndarray | None:
        """
        Get correlation matrix for pending bets.

        Useful for understanding how bets are related.
        """
        if not HAS_PORTFOLIO_OPTIMIZER:
            return None

        pending_bets = self.get_pending_bets()
        if len(pending_bets) < 2:
            return None

        bets_data = []
        for bet in pending_bets:
            bets_data.append({
                'game_id': bet.event_id,
                'bet_type': bet.bet_type.value,
                'team': bet.event_name.split(' vs ')[0] if ' vs ' in bet.event_name else None,
                'probability': bet.model_probability,
            })

        try:
            return calculate_covariance(bets_data)
        except Exception:
            return None

    def print_summary(self) -> None:
        """Print comprehensive summary of betting performance."""
        # Overall performance
        overall = self.calculate_performance(
            start_date=datetime.now() - timedelta(days=365)
        )
        print(overall.summary())

        # By bet type
        print("\nPerformance by Bet Type:")
        print("-" * 40)
        for bet_type in BetType:
            metrics = self.calculate_performance(
                start_date=datetime.now() - timedelta(days=365),
                bet_type=bet_type
            )
            if metrics.total_bets > 0:
                print(f"{bet_type.value:15} | {metrics.total_bets:4} bets | "
                      f"{metrics.win_rate:5.1f}% WR | {metrics.roi:+6.2f}% ROI | "
                      f"${metrics.total_pnl:+8.2f}")

        # By edge range
        print("\nPerformance by Edge Range:")
        print("-" * 40)
        edge_analysis = self.get_performance_by_edge_range()
        for edge_range, metrics in edge_analysis.items():
            print(f"{edge_range:10} | {metrics.total_bets:4} bets | "
                  f"{metrics.win_rate:5.1f}% WR | {metrics.roi:+6.2f}% ROI | "
                  f"${metrics.total_pnl:+8.2f}")

        # CLV Analytics
        print("\nCLV (Closing Line Value) Analytics:")
        print("-" * 40)
        clv_analytics = self.get_clv_analytics()
        if clv_analytics.get("overall_avg_clv") is not None:
            print(f"Overall Avg CLV: {clv_analytics['overall_avg_clv']:.4f}")
            print(f"Positive CLV %:  {clv_analytics['positive_clv_pct']:.1%}")
            print(f"Sample Size:     {clv_analytics['sample_size']}")
            if clv_analytics.get("clv_roi_correlation") is not None:
                print(f"CLV-ROI Corr:    {clv_analytics['clv_roi_correlation']:.3f}")
        else:
            print("No CLV data available (closing odds not recorded)")

    def get_clv_analytics(self) -> dict:
        """
        Get comprehensive CLV (Closing Line Value) analytics.

        CLV is the most important metric for validating a betting edge.
        Positive CLV over time indicates you're consistently getting
        better odds than the market settles on.

        Returns:
            Dictionary with:
            - overall_avg_clv: Average CLV across all bets
            - positive_clv_pct: Percentage of bets with positive CLV
            - sample_size: Number of bets with CLV data
            - by_bet_type: CLV breakdown by bet type
            - clv_roi_correlation: Correlation between CLV and ROI
            - clv_distribution: Distribution of CLV values
        """
        # Get all settled bets with closing odds
        bets = self._query_bets(
            "status != 'pending' AND closing_odds IS NOT NULL",
            ()
        )

        result = {
            "overall_avg_clv": None,
            "positive_clv_pct": None,
            "sample_size": 0,
            "by_bet_type": {},
            "clv_roi_correlation": None,
            "clv_distribution": {
                "very_positive": 0,   # CLV > 0.05
                "positive": 0,         # CLV 0.01-0.05
                "neutral": 0,          # CLV -0.01 to 0.01
                "negative": 0,         # CLV -0.05 to -0.01
                "very_negative": 0,    # CLV < -0.05
            },
            "clv_vs_outcome": {
                "positive_clv_wins": 0,
                "positive_clv_losses": 0,
                "negative_clv_wins": 0,
                "negative_clv_losses": 0,
            },
        }

        if not bets:
            return result

        # Calculate CLV for each bet
        clv_values = []
        rois = []

        for bet in bets:
            clv = bet.closing_line_value()
            if clv is not None:
                clv_values.append(clv)

                # Track ROI for correlation
                if bet.stake > 0:
                    rois.append(bet.pnl / bet.stake)

                # Categorize CLV
                if clv > 0.05:
                    result["clv_distribution"]["very_positive"] += 1
                elif clv > 0.01:
                    result["clv_distribution"]["positive"] += 1
                elif clv > -0.01:
                    result["clv_distribution"]["neutral"] += 1
                elif clv > -0.05:
                    result["clv_distribution"]["negative"] += 1
                else:
                    result["clv_distribution"]["very_negative"] += 1

                # Track CLV vs outcome
                if clv > 0:
                    if bet.status == BetStatus.WON:
                        result["clv_vs_outcome"]["positive_clv_wins"] += 1
                    elif bet.status == BetStatus.LOST:
                        result["clv_vs_outcome"]["positive_clv_losses"] += 1
                else:
                    if bet.status == BetStatus.WON:
                        result["clv_vs_outcome"]["negative_clv_wins"] += 1
                    elif bet.status == BetStatus.LOST:
                        result["clv_vs_outcome"]["negative_clv_losses"] += 1

        if not clv_values:
            return result

        result["sample_size"] = len(clv_values)
        result["overall_avg_clv"] = float(np.mean(clv_values))
        result["positive_clv_pct"] = sum(1 for c in clv_values if c > 0) / len(clv_values)

        # CLV by bet type
        clv_by_type = {}
        for bet_type in BetType:
            type_bets = [b for b in bets if b.bet_type == bet_type]
            type_clvs = [b.closing_line_value() for b in type_bets
                        if b.closing_line_value() is not None]
            if type_clvs:
                clv_by_type[bet_type.value] = {
                    "avg_clv": float(np.mean(type_clvs)),
                    "positive_pct": sum(1 for c in type_clvs if c > 0) / len(type_clvs),
                    "sample_size": len(type_clvs),
                }
        result["by_bet_type"] = clv_by_type

        # CLV vs ROI correlation
        if len(clv_values) >= 10 and len(rois) == len(clv_values):
            try:
                correlation = np.corrcoef(clv_values, rois)[0, 1]
                if not np.isnan(correlation):
                    result["clv_roi_correlation"] = float(correlation)
            except Exception:
                pass

        return result


def create_tracker(db_path: str = "bets.db", pg_conn=None) -> BetTracker:
    """Create a new bet tracker instance."""
    return BetTracker(db_path=db_path, pg_conn=pg_conn)


def quick_record(
    selection: str,
    odds: float,
    stake: float,
    model_prob: float,
    bet_type: str = "moneyline",
    sportsbook: str = "",
    event_name: str = "",
    tracker: BetTracker = None
) -> TrackedBet:
    """
    Quick helper to record a bet.

    Args:
        selection: What you're betting on
        odds: American odds
        stake: Amount wagered
        model_prob: Model's win probability
        bet_type: Type of bet
        sportsbook: Which book
        event_name: Game/event name
        tracker: Optional existing tracker

    Returns:
        TrackedBet object
    """
    if tracker is None:
        tracker = BetTracker()

    bet = TrackedBet(
        bet_id=f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{selection[:10]}",
        placed_at=datetime.now(),
        bet_type=BetType(bet_type),
        sportsbook=sportsbook,
        event_name=event_name,
        selection=selection,
        odds=odds,
        stake=stake,
        model_probability=model_prob,
    )

    tracker.record_bet(bet)
    return bet


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Bet Tracking System Demo")
    print("=" * 60)

    tracker = BetTracker("demo_bets.db")

    # Record some sample bets
    sample_bets = [
        {"selection": "Lakers ML", "odds": -150, "stake": 100, "model_prob": 0.62, "won": True},
        {"selection": "Celtics +3.5", "odds": -110, "stake": 110, "model_prob": 0.55, "won": True},
        {"selection": "Warriors ML", "odds": +140, "stake": 75, "model_prob": 0.48, "won": False},
        {"selection": "Bucks -5.5", "odds": -110, "stake": 100, "model_prob": 0.58, "won": True},
        {"selection": "Over 225.5", "odds": -110, "stake": 50, "model_prob": 0.53, "won": False},
    ]

    print("\nRecording sample bets...")
    for i, bet_data in enumerate(sample_bets):
        bet = TrackedBet(
            bet_id=f"demo_{i}_{datetime.now().strftime('%H%M%S')}",
            placed_at=datetime.now() - timedelta(days=len(sample_bets) - i),
            selection=bet_data["selection"],
            odds=bet_data["odds"],
            stake=bet_data["stake"],
            model_probability=bet_data["model_prob"],
            bet_type=BetType.MONEYLINE if "ML" in bet_data["selection"] else BetType.SPREAD if "+" in bet_data["selection"] or "-" in bet_data["selection"] and "Over" not in bet_data["selection"] else BetType.TOTAL,
        )
        tracker.record_bet(bet)

        # Settle bet
        status = BetStatus.WON if bet_data["won"] else BetStatus.LOST
        tracker.settle_bet(bet.bet_id, status)

    # Print summary
    print("\n")
    tracker.print_summary()

    # Cleanup demo db
    import os
    os.remove("demo_bets.db")
