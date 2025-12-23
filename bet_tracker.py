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
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    event_date: Optional[datetime] = None

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
    opening_odds: Optional[float] = None
    closing_odds: Optional[float] = None
    line_movement: float = 0.0  # How much the line moved

    # Outcome
    status: BetStatus = BetStatus.PENDING
    actual_result: Optional[str] = None  # e.g., "Lakers 112-105"
    pnl: float = 0.0
    settled_at: Optional[datetime] = None

    # Additional metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    parlay_legs: List[Dict] = field(default_factory=list)  # For parlay bets

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['bet_type'] = self.bet_type.value
        d['status'] = self.status.value
        d['placed_at'] = self.placed_at.isoformat() if self.placed_at else None
        d['event_date'] = self.event_date.isoformat() if self.event_date else None
        d['settled_at'] = self.settled_at.isoformat() if self.settled_at else None
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "TrackedBet":
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

    def closing_line_value(self) -> Optional[float]:
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
        else:
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
    avg_clv: Optional[float] = None
    max_win: float = 0.0
    max_loss: float = 0.0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0

    def to_dict(self) -> Dict:
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


class BetTracker:
    """
    Main bet tracking system with SQLite backend.

    Provides comprehensive bet recording, tracking, and analysis.
    """

    def __init__(self, db_path: str = "bets.db"):
        """
        Initialize bet tracker.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
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

    def record_bet(self, bet: TrackedBet) -> str:
        """
        Record a new bet.

        Args:
            bet: TrackedBet object

        Returns:
            bet_id
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate potential payout
        if bet.odds > 0:
            bet.potential_payout = bet.stake + bet.stake * (bet.odds / 100)
        else:
            bet.potential_payout = bet.stake + bet.stake * (100 / abs(bet.odds))

        # Calculate implied probability
        bet.implied_probability = bet._odds_to_prob(bet.odds)
        bet.edge = bet.model_probability - bet.implied_probability

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

        logger.info(f"Recorded bet {bet.bet_id}: {bet.selection} @ {bet.odds} for ${bet.stake}")
        return bet.bet_id

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

        # Save updates
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
            bet_id
        ))

        conn.commit()
        conn.close()

        logger.info(f"Settled bet {bet_id}: {status.value}, P&L: ${bet.pnl:+.2f}")
        return bet

    def get_bet(self, bet_id: str) -> Optional[TrackedBet]:
        """Get a single bet by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM bets WHERE bet_id = ?", (bet_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_bet(dict(row))
        return None

    def get_pending_bets(self) -> List[TrackedBet]:
        """Get all pending (unsettled) bets."""
        return self._query_bets("status = ?", (BetStatus.PENDING.value,))

    def get_bets_by_date(
        self,
        start_date: datetime,
        end_date: datetime = None
    ) -> List[TrackedBet]:
        """Get bets within date range."""
        if end_date is None:
            end_date = datetime.now()

        return self._query_bets(
            "placed_at >= ? AND placed_at <= ?",
            (start_date.isoformat(), end_date.isoformat())
        )

    def get_bets_by_type(self, bet_type: BetType) -> List[TrackedBet]:
        """Get bets of specific type."""
        return self._query_bets("bet_type = ?", (bet_type.value,))

    def get_bets_by_sportsbook(self, sportsbook: str) -> List[TrackedBet]:
        """Get bets from specific sportsbook."""
        return self._query_bets("sportsbook = ?", (sportsbook,))

    def _query_bets(self, where_clause: str, params: tuple) -> List[TrackedBet]:
        """Query bets with WHERE clause."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM bets WHERE {where_clause} ORDER BY placed_at DESC", params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_bet(dict(row)) for row in rows]

    def _row_to_bet(self, row: Dict) -> TrackedBet:
        """Convert database row to TrackedBet."""
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

    def _calculate_streaks(self, bets: List[TrackedBet]) -> Tuple[int, int]:
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

    def _calculate_sharpe(self, bets: List[TrackedBet]) -> float:
        """Calculate Sharpe ratio."""
        daily_pnl: Dict[str, float] = {}
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
    ) -> List[Tuple[datetime, float]]:
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

    def get_performance_by_edge_range(self) -> Dict[str, PerformanceMetrics]:
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


def create_tracker(db_path: str = "bets.db") -> BetTracker:
    """Create a new bet tracker instance."""
    return BetTracker(db_path)


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
