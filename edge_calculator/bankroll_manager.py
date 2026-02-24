"""
Bankroll Manager - Track Bankroll, Exposure, and Risk Limits

Responsibilities:
- Track current bankroll and daily P/L
- Enforce exposure limits by prop type and game
- Detect correlated bets (same game, same player)
- Provide risk warnings
"""

import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import sqlite3
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PendingBet:
    """A pending bet that hasn't been settled."""
    bet_id: str
    timestamp: str
    player_name: str
    player_id: Optional[int]
    prop_type: str
    pick: str  # OVER or UNDER
    line: float
    odds: int
    stake: float
    units: float
    game_id: Optional[int] = None
    game_date: str = ""
    team: str = ""
    opponent: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SettledBet:
    """A settled bet with outcome."""
    bet_id: str
    pending_bet: PendingBet
    actual_value: float
    result: str  # win, loss, push
    profit_loss: float
    settled_at: str

    def to_dict(self) -> dict:
        return {
            'bet_id': self.bet_id,
            'pending_bet': self.pending_bet.to_dict(),
            'actual_value': self.actual_value,
            'result': self.result,
            'profit_loss': self.profit_loss,
            'settled_at': self.settled_at,
        }


@dataclass
class ExposureLimit:
    """Exposure limit configuration."""
    max_total_exposure: float = 0.20  # 20% of bankroll total
    max_per_prop_type: float = 0.20  # 20% per prop type
    max_per_game: float = 0.10  # 10% per game
    max_per_player: float = 0.05  # 5% per player
    max_correlated: float = 0.15  # 15% for correlated bets


class ExposureTracker:
    """Track current exposure across dimensions."""

    def __init__(self, bankroll: float):
        """
        Initialize exposure tracker.

        Args:
            bankroll: Current bankroll
        """
        self.bankroll = bankroll

        # Exposure by dimension
        self.by_prop_type: dict[str, float] = {}
        self.by_game: dict[str, float] = {}
        self.by_player: dict[str, float] = {}
        self.by_team: dict[str, float] = {}

        # Pending bets
        self.pending_bets: list[PendingBet] = []

        # Total exposure
        self.total_exposure: float = 0.0

    def add_bet(self, bet: PendingBet):
        """
        Add a pending bet and update exposure.

        Args:
            bet: PendingBet to add
        """
        self.pending_bets.append(bet)

        # Update exposure tracking
        self.total_exposure += bet.stake

        # By prop type
        if bet.prop_type not in self.by_prop_type:
            self.by_prop_type[bet.prop_type] = 0.0
        self.by_prop_type[bet.prop_type] += bet.stake

        # By game
        game_key = f"{bet.game_date}_{bet.team}_{bet.opponent}"
        if game_key not in self.by_game:
            self.by_game[game_key] = 0.0
        self.by_game[game_key] += bet.stake

        # By player
        player_key = str(bet.player_id or bet.player_name)
        if player_key not in self.by_player:
            self.by_player[player_key] = 0.0
        self.by_player[player_key] += bet.stake

        # By team
        if bet.team not in self.by_team:
            self.by_team[bet.team] = 0.0
        self.by_team[bet.team] += bet.stake

    def remove_bet(self, bet_id: str):
        """Remove a settled bet from tracking."""
        for i, bet in enumerate(self.pending_bets):
            if bet.bet_id == bet_id:
                # Remove exposure
                self.total_exposure -= bet.stake
                self.by_prop_type[bet.prop_type] -= bet.stake
                game_key = f"{bet.game_date}_{bet.team}_{bet.opponent}"
                self.by_game[game_key] -= bet.stake
                player_key = str(bet.player_id or bet.player_name)
                self.by_player[player_key] -= bet.stake
                self.by_team[bet.team] -= bet.stake

                # Remove bet
                self.pending_bets.pop(i)
                break

    def get_exposure_fraction(self, dimension: str, key: str) -> float:
        """Get exposure as fraction of bankroll."""
        if dimension == 'total':
            return self.total_exposure / self.bankroll if self.bankroll > 0 else 0
        elif dimension == 'prop_type':
            return self.by_prop_type.get(key, 0) / self.bankroll if self.bankroll > 0 else 0
        elif dimension == 'game':
            return self.by_game.get(key, 0) / self.bankroll if self.bankroll > 0 else 0
        elif dimension == 'player':
            return self.by_player.get(key, 0) / self.bankroll if self.bankroll > 0 else 0
        elif dimension == 'team':
            return self.by_team.get(key, 0) / self.bankroll if self.bankroll > 0 else 0
        return 0

    def get_summary(self) -> dict:
        """Get exposure summary."""
        return {
            'total_exposure': self.total_exposure,
            'total_exposure_pct': self.total_exposure / self.bankroll if self.bankroll > 0 else 0,
            'pending_bets': len(self.pending_bets),
            'by_prop_type': dict(self.by_prop_type),
            'by_game': dict(self.by_game),
            'by_team': dict(self.by_team),
        }


class BankrollManager:
    """
    Manage bankroll, track bets, and enforce risk limits.
    """

    def __init__(
        self,
        initial_bankroll: float,
        db_path: str = "data/bankroll.db",
        exposure_limits: ExposureLimit = None,
    ):
        """
        Initialize bankroll manager.

        Args:
            initial_bankroll: Starting bankroll
            db_path: Path to SQLite database
            exposure_limits: Exposure limit configuration
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.limits = exposure_limits or ExposureLimit()

        # Initialize database
        self._init_db()

        # Load or initialize bankroll
        self.bankroll = self._load_bankroll() or initial_bankroll
        if self._load_bankroll() is None:
            self._save_bankroll(initial_bankroll)

        # Initialize exposure tracker
        self.exposure = ExposureTracker(self.bankroll)

        # Load pending bets
        self._load_pending_bets()

        logger.info(f"BankrollManager initialized with ${self.bankroll:.2f}")

    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Bankroll table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bankroll (
                    id INTEGER PRIMARY KEY,
                    amount REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Pending bets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_bets (
                    bet_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    player_id INTEGER,
                    prop_type TEXT NOT NULL,
                    pick TEXT NOT NULL,
                    line REAL NOT NULL,
                    odds INTEGER NOT NULL,
                    stake REAL NOT NULL,
                    units REAL NOT NULL,
                    game_id INTEGER,
                    game_date TEXT,
                    team TEXT,
                    opponent TEXT
                )
            """)

            # Settled bets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settled_bets (
                    bet_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    player_id INTEGER,
                    prop_type TEXT NOT NULL,
                    pick TEXT NOT NULL,
                    line REAL NOT NULL,
                    odds INTEGER NOT NULL,
                    stake REAL NOT NULL,
                    units REAL NOT NULL,
                    game_date TEXT,
                    actual_value REAL NOT NULL,
                    result TEXT NOT NULL,
                    profit_loss REAL NOT NULL,
                    settled_at TEXT NOT NULL
                )
            """)

            # Daily P/L table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_pl (
                    date TEXT PRIMARY KEY,
                    starting_bankroll REAL,
                    ending_bankroll REAL,
                    total_staked REAL,
                    total_returned REAL,
                    profit_loss REAL,
                    num_bets INTEGER,
                    num_wins INTEGER,
                    num_losses INTEGER
                )
            """)

    def _load_bankroll(self) -> Optional[float]:
        """Load bankroll from database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT amount FROM bankroll ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return row['amount'] if row else None

    def _save_bankroll(self, amount: float):
        """Save bankroll to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO bankroll (amount, updated_at) VALUES (?, ?)",
                (amount, datetime.now().isoformat())
            )

    def _load_pending_bets(self):
        """Load pending bets from database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM pending_bets")
            for row in cursor.fetchall():
                bet = PendingBet(
                    bet_id=row['bet_id'],
                    timestamp=row['timestamp'],
                    player_name=row['player_name'],
                    player_id=row['player_id'],
                    prop_type=row['prop_type'],
                    pick=row['pick'],
                    line=row['line'],
                    odds=row['odds'],
                    stake=row['stake'],
                    units=row['units'],
                    game_id=row['game_id'],
                    game_date=row['game_date'],
                    team=row['team'],
                    opponent=row['opponent'],
                )
                self.exposure.add_bet(bet)

    def check_limits(
        self,
        stake: float,
        prop_type: str,
        game_key: str,
        player_key: str,
    ) -> tuple[bool, list[str]]:
        """
        Check if a bet would violate exposure limits.

        Args:
            stake: Proposed stake amount
            prop_type: Prop type
            game_key: Game identifier
            player_key: Player identifier

        Returns:
            Tuple of (is_allowed, list of warnings)
        """
        warnings = []
        is_allowed = True

        # Check total exposure
        new_total = (self.exposure.total_exposure + stake) / self.bankroll
        if new_total > self.limits.max_total_exposure:
            warnings.append(
                f"Total exposure would exceed {self.limits.max_total_exposure:.0%} "
                f"({new_total:.1%} after bet)"
            )
            is_allowed = False

        # Check prop type exposure
        current_prop = self.exposure.by_prop_type.get(prop_type, 0)
        new_prop = (current_prop + stake) / self.bankroll
        if new_prop > self.limits.max_per_prop_type:
            warnings.append(
                f"{prop_type} exposure would exceed {self.limits.max_per_prop_type:.0%} "
                f"({new_prop:.1%} after bet)"
            )
            is_allowed = False

        # Check game exposure
        current_game = self.exposure.by_game.get(game_key, 0)
        new_game = (current_game + stake) / self.bankroll
        if new_game > self.limits.max_per_game:
            warnings.append(
                f"Game exposure would exceed {self.limits.max_per_game:.0%} "
                f"({new_game:.1%} after bet)"
            )
            is_allowed = False

        # Check player exposure
        current_player = self.exposure.by_player.get(player_key, 0)
        new_player = (current_player + stake) / self.bankroll
        if new_player > self.limits.max_per_player:
            warnings.append(
                f"Player exposure would exceed {self.limits.max_per_player:.0%} "
                f"({new_player:.1%} after bet)"
            )
            is_allowed = False

        return is_allowed, warnings

    def check_correlation(
        self,
        game_key: str,
        team: str,
    ) -> list[str]:
        """
        Check for correlation warnings (multiple bets same game).

        Args:
            game_key: Game identifier
            team: Team abbreviation

        Returns:
            List of correlation warnings
        """
        warnings = []

        # Check if we already have bets on this game
        game_bets = [b for b in self.exposure.pending_bets
                     if f"{b.game_date}_{b.team}_{b.opponent}" == game_key]

        if game_bets:
            warnings.append(
                f"CORRELATION: Already have {len(game_bets)} bet(s) on this game"
            )

            # Check same team
            same_team = [b for b in game_bets if b.team == team]
            if same_team:
                warnings.append(
                    f"CORRELATION: Already betting on {team} in this game "
                    f"({[b.player_name for b in same_team]})"
                )

        return warnings

    def place_bet(
        self,
        player_name: str,
        prop_type: str,
        pick: str,
        line: float,
        odds: int,
        stake: float,
        units: float,
        player_id: int = None,
        game_id: int = None,
        game_date: str = None,
        team: str = "",
        opponent: str = "",
        force: bool = False,
    ) -> tuple[Optional[str], list[str]]:
        """
        Place a bet and track it.

        Args:
            player_name: Player name
            prop_type: Prop type
            pick: OVER or UNDER
            line: Betting line
            odds: American odds
            stake: Stake amount
            units: Stake in units
            player_id: Player ID
            game_id: Game ID
            game_date: Game date
            team: Player's team
            opponent: Opponent team
            force: Force bet even if limits exceeded

        Returns:
            Tuple of (bet_id or None, list of warnings)
        """
        game_date = game_date or datetime.now().strftime('%Y-%m-%d')
        game_key = f"{game_date}_{team}_{opponent}"
        player_key = str(player_id or player_name)

        warnings = []

        # Check limits
        is_allowed, limit_warnings = self.check_limits(stake, prop_type, game_key, player_key)
        warnings.extend(limit_warnings)

        # Check correlations
        corr_warnings = self.check_correlation(game_key, team)
        warnings.extend(corr_warnings)

        if not is_allowed and not force:
            logger.warning(f"Bet rejected due to limit violations: {warnings}")
            return None, warnings

        # Generate bet ID
        bet_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{player_name[:10]}_{prop_type}"

        # Create pending bet
        bet = PendingBet(
            bet_id=bet_id,
            timestamp=datetime.now().isoformat(),
            player_name=player_name,
            player_id=player_id,
            prop_type=prop_type,
            pick=pick,
            line=line,
            odds=odds,
            stake=stake,
            units=units,
            game_id=game_id,
            game_date=game_date,
            team=team,
            opponent=opponent,
        )

        # Save to database
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pending_bets
                (bet_id, timestamp, player_name, player_id, prop_type, pick, line,
                 odds, stake, units, game_id, game_date, team, opponent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bet.bet_id, bet.timestamp, bet.player_name, bet.player_id,
                bet.prop_type, bet.pick, bet.line, bet.odds, bet.stake,
                bet.units, bet.game_id, bet.game_date, bet.team, bet.opponent,
            ))

        # Update exposure
        self.exposure.add_bet(bet)

        logger.info(f"Placed bet {bet_id}: {player_name} {pick} {line} {prop_type} (${stake:.2f})")

        return bet_id, warnings

    def settle_bet(
        self,
        bet_id: str,
        actual_value: float,
    ) -> Optional[SettledBet]:
        """
        Settle a pending bet.

        Args:
            bet_id: Bet ID to settle
            actual_value: Actual stat value

        Returns:
            SettledBet or None if not found
        """
        # Find pending bet
        pending = None
        for bet in self.exposure.pending_bets:
            if bet.bet_id == bet_id:
                pending = bet
                break

        if not pending:
            logger.warning(f"Bet {bet_id} not found in pending bets")
            return None

        # Determine result
        if pending.pick == 'OVER':
            won = actual_value > pending.line
            pushed = actual_value == pending.line
        else:  # UNDER
            won = actual_value < pending.line
            pushed = actual_value == pending.line

        if pushed:
            result = 'push'
            profit_loss = 0.0
        elif won:
            result = 'win'
            # Calculate profit from odds
            if pending.odds < 0:
                profit_loss = pending.stake * (100 / abs(pending.odds))
            else:
                profit_loss = pending.stake * (pending.odds / 100)
        else:
            result = 'loss'
            profit_loss = -pending.stake

        # Update bankroll
        self.bankroll += profit_loss
        self._save_bankroll(self.bankroll)

        # Create settled bet
        settled = SettledBet(
            bet_id=bet_id,
            pending_bet=pending,
            actual_value=actual_value,
            result=result,
            profit_loss=profit_loss,
            settled_at=datetime.now().isoformat(),
        )

        # Save to database
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert into settled
            cursor.execute("""
                INSERT INTO settled_bets
                (bet_id, timestamp, player_name, player_id, prop_type, pick, line,
                 odds, stake, units, game_date, actual_value, result, profit_loss, settled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pending.bet_id, pending.timestamp, pending.player_name, pending.player_id,
                pending.prop_type, pending.pick, pending.line, pending.odds,
                pending.stake, pending.units, pending.game_date,
                actual_value, result, profit_loss, settled.settled_at,
            ))

            # Remove from pending
            cursor.execute("DELETE FROM pending_bets WHERE bet_id = ?", (bet_id,))

        # Update exposure tracker
        self.exposure.remove_bet(bet_id)
        self.exposure.bankroll = self.bankroll

        logger.info(f"Settled {bet_id}: {result} (${profit_loss:+.2f}), Bankroll: ${self.bankroll:.2f}")

        return settled

    def get_daily_stats(self, date: str = None) -> dict:
        """Get statistics for a specific date."""
        date = date or datetime.now().strftime('%Y-%m-%d')

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM settled_bets WHERE game_date = ?
            """, (date,))
            bets = cursor.fetchall()

        if not bets:
            return {
                'date': date,
                'num_bets': 0,
                'wins': 0,
                'losses': 0,
                'pushes': 0,
                'total_staked': 0,
                'profit_loss': 0,
                'roi': 0,
            }

        total_staked = sum(b['stake'] for b in bets)
        profit_loss = sum(b['profit_loss'] for b in bets)
        wins = sum(1 for b in bets if b['result'] == 'win')
        losses = sum(1 for b in bets if b['result'] == 'loss')
        pushes = sum(1 for b in bets if b['result'] == 'push')

        return {
            'date': date,
            'num_bets': len(bets),
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'total_staked': total_staked,
            'profit_loss': profit_loss,
            'roi': profit_loss / total_staked if total_staked > 0 else 0,
        }

    def get_bankroll_history(self, days: int = 30) -> list[dict]:
        """Get bankroll history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT amount, updated_at FROM bankroll
                ORDER BY id DESC LIMIT ?
            """, (days,))
            return [{'amount': r['amount'], 'date': r['updated_at']} for r in cursor.fetchall()]

    def get_status(self) -> dict:
        """Get current bankroll status."""
        return {
            'bankroll': self.bankroll,
            'pending_bets': len(self.exposure.pending_bets),
            'total_exposure': self.exposure.total_exposure,
            'exposure_pct': self.exposure.total_exposure / self.bankroll if self.bankroll > 0 else 0,
            'exposure_summary': self.exposure.get_summary(),
        }


if __name__ == "__main__":
    # Test bankroll manager
    manager = BankrollManager(initial_bankroll=1000)

    print("=" * 60)
    print("BANKROLL MANAGER TEST")
    print("=" * 60)

    print(f"\nInitial bankroll: ${manager.bankroll:.2f}")

    # Place some bets
    bet1_id, warnings1 = manager.place_bet(
        player_name="LeBron James",
        prop_type="points",
        pick="OVER",
        line=26.5,
        odds=-110,
        stake=20.0,
        units=2.0,
        team="LAL",
        opponent="BOS",
        game_date="2024-01-15",
    )
    print(f"\nBet 1 placed: {bet1_id}")
    if warnings1:
        print(f"  Warnings: {warnings1}")

    bet2_id, warnings2 = manager.place_bet(
        player_name="Giannis Antetokounmpo",
        prop_type="points",
        pick="OVER",
        line=29.5,
        odds=-110,
        stake=25.0,
        units=2.5,
        team="MIL",
        opponent="PHI",
        game_date="2024-01-15",
    )
    print(f"Bet 2 placed: {bet2_id}")

    # Check exposure
    print(f"\nExposure Summary:")
    status = manager.get_status()
    print(f"  Total exposure: ${status['total_exposure']:.2f} ({status['exposure_pct']:.1%})")
    print(f"  Pending bets: {status['pending_bets']}")

    # Try to place correlated bet
    bet3_id, warnings3 = manager.place_bet(
        player_name="Anthony Davis",
        prop_type="rebounds",
        pick="OVER",
        line=11.5,
        odds=-110,
        stake=15.0,
        units=1.5,
        team="LAL",
        opponent="BOS",
        game_date="2024-01-15",
    )
    print(f"\nBet 3 (same game): {bet3_id}")
    if warnings3:
        print(f"  Warnings: {warnings3}")

    # Settle bets
    print("\nSettling bets...")
    settled1 = manager.settle_bet(bet1_id, actual_value=29)  # Win
    print(f"  {bet1_id}: {settled1.result} (${settled1.profit_loss:+.2f})")

    settled2 = manager.settle_bet(bet2_id, actual_value=28)  # Loss
    print(f"  {bet2_id}: {settled2.result} (${settled2.profit_loss:+.2f})")

    settled3 = manager.settle_bet(bet3_id, actual_value=13)  # Win
    print(f"  {bet3_id}: {settled3.result} (${settled3.profit_loss:+.2f})")

    print(f"\nFinal bankroll: ${manager.bankroll:.2f}")
    print(f"Pending bets: {len(manager.exposure.pending_bets)}")
