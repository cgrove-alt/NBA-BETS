"""
Risk Management Module for NBA Betting Model

Provides capital preservation and drawdown protection mechanisms.
Critical for long-term betting success - without proper risk management,
even a winning strategy can lead to ruin due to variance.

Features:
1. Drawdown Protection - halt or reduce betting during drawdowns
2. Daily/Weekly Loss Limits - circuit breakers for bad streaks
3. Dynamic Position Sizing - adjust stakes based on risk conditions
4. Bankroll Management - track and protect capital
5. Recovery Mode - graduated return to full stakes after drawdowns

Key Principles:
- Preserve capital first, maximize returns second
- Reduce stakes during drawdowns (anti-martingale)
- Use circuit breakers to prevent catastrophic losses
- Graduated recovery prevents over-betting after losses
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# RISK OF RUIN CALCULATIONS
# =============================================================================

def calculate_risk_of_ruin(
    win_probability: float,
    win_payout: float,
    loss_amount: float,
    bankroll_units: float,
    ruin_threshold_units: float = 0
) -> float:
    """
    Calculate probability of going bust (Risk of Ruin).

    Uses the closed-form solution for simplified RoR calculation.
    For more accurate results with varying bet sizes, use Monte Carlo simulation.

    Formula: RoR = ((1-p)/p * (a/b))^n
    Where:
        p = win probability
        a = profit on win
        b = loss on loss
        n = number of units to lose to reach ruin

    Args:
        win_probability: Probability of winning each bet (0-1)
        win_payout: Amount won on a win (profit, not total return)
        loss_amount: Amount lost on a loss
        bankroll_units: Current bankroll in betting units
        ruin_threshold_units: Bankroll level considered "ruin" (default 0)

    Returns:
        Probability of ruin (0-1)
    """
    if win_probability <= 0 or win_probability >= 1:
        return 0.0 if win_probability >= 1 else 1.0

    if win_payout <= 0 or loss_amount <= 0:
        return 0.0

    p = win_probability
    q = 1 - p

    # Expected value per bet
    ev = p * win_payout - q * loss_amount

    # If EV is negative or zero, eventual ruin is certain (or very high)
    if ev <= 0:
        return 1.0

    # Units we can lose before reaching ruin
    units_to_ruin = bankroll_units - ruin_threshold_units

    if units_to_ruin <= 0:
        return 1.0

    # Simplified formula (assumes fixed bet size)
    # RoR = ((q/p) * (win_payout/loss_amount)) ^ units_to_ruin
    # Only valid when EV > 0

    ratio = (q / p) * (loss_amount / win_payout)

    if ratio >= 1:
        # Edge not large enough to overcome variance
        return 1.0

    # Calculate RoR
    try:
        ror = ratio ** units_to_ruin
    except OverflowError:
        ror = 0.0

    return min(1.0, max(0.0, ror))


def calculate_risk_of_ruin_monte_carlo(
    win_probability: float,
    bet_size_fraction: float,
    decimal_odds: float,
    num_simulations: int = 10000,
    num_bets: int = 1000,
    ruin_threshold_pct: float = 0.0
) -> Dict:
    """
    Calculate Risk of Ruin using Monte Carlo simulation.

    More accurate than closed-form when bet sizes vary or for complex scenarios.

    Args:
        win_probability: Probability of winning each bet
        bet_size_fraction: Bet size as fraction of bankroll
        decimal_odds: Decimal odds (e.g., 1.91 for -110)
        num_simulations: Number of Monte Carlo paths to simulate
        num_bets: Number of bets in each simulation
        ruin_threshold_pct: Bankroll percentage considered ruin

    Returns:
        Dict with RoR statistics and distribution
    """
    np.random.seed(42)  # For reproducibility

    starting_bankroll = 1.0  # Normalized to 1
    ruin_threshold = starting_bankroll * ruin_threshold_pct

    ruined_count = 0
    final_bankrolls = []
    max_drawdowns = []

    for _ in range(num_simulations):
        bankroll = starting_bankroll
        peak = starting_bankroll
        max_dd = 0

        for _ in range(num_bets):
            # Calculate bet size (fraction of current bankroll)
            bet_size = bankroll * bet_size_fraction

            # Simulate outcome
            if np.random.random() < win_probability:
                # Win
                profit = bet_size * (decimal_odds - 1)
                bankroll += profit
            else:
                # Lose
                bankroll -= bet_size

            # Track peak and drawdown
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            # Check for ruin
            if bankroll <= ruin_threshold:
                ruined_count += 1
                break

        final_bankrolls.append(bankroll)
        max_drawdowns.append(max_dd)

    ror = ruined_count / num_simulations
    final_arr = np.array(final_bankrolls)
    dd_arr = np.array(max_drawdowns)

    return {
        "risk_of_ruin": ror,
        "risk_of_ruin_pct": ror * 100,
        "mean_final_bankroll": np.mean(final_arr),
        "median_final_bankroll": np.median(final_arr),
        "std_final_bankroll": np.std(final_arr),
        "mean_max_drawdown": np.mean(dd_arr),
        "percentile_5_final": np.percentile(final_arr, 5),
        "percentile_95_final": np.percentile(final_arr, 95),
        "simulations": num_simulations,
        "bets_per_sim": num_bets,
    }


# =============================================================================
# CLV (CLOSING LINE VALUE) TRACKING
# =============================================================================

class CLVTracker:
    """
    Track and analyze Closing Line Value (CLV).

    CLV is the single best predictor of long-term betting profitability.
    If you consistently beat the closing line, you have a real edge.
    """

    def __init__(self):
        self.bets: List[Dict] = []

    def record_bet(
        self,
        bet_id: str,
        selection: str,
        opening_odds: float,
        bet_odds: float,
        closing_odds: float,
        won: bool = None
    ) -> Dict:
        """
        Record a bet with opening, bet, and closing odds.

        Args:
            bet_id: Unique bet identifier
            selection: What was bet on
            opening_odds: Odds when market opened (American)
            bet_odds: Odds when bet was placed (American)
            closing_odds: Final odds before game start (American)
            won: Whether bet won (can be None if pending)

        Returns:
            Dict with CLV analysis for this bet
        """
        # Convert to probabilities
        def american_to_prob(odds):
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)

        bet_prob = american_to_prob(bet_odds)
        closing_prob = american_to_prob(closing_odds)
        opening_prob = american_to_prob(opening_odds)

        # CLV = closing probability - bet probability
        # Positive CLV = you got better odds than closing
        clv = closing_prob - bet_prob
        clv_vs_opening = opening_prob - bet_prob

        # Line movement direction
        line_movement = closing_prob - opening_prob

        bet_record = {
            "bet_id": bet_id,
            "selection": selection,
            "opening_odds": opening_odds,
            "bet_odds": bet_odds,
            "closing_odds": closing_odds,
            "opening_prob": opening_prob,
            "bet_prob": bet_prob,
            "closing_prob": closing_prob,
            "clv": clv,
            "clv_vs_opening": clv_vs_opening,
            "line_movement": line_movement,
            "won": won,
            "timestamp": datetime.now().isoformat(),
        }

        self.bets.append(bet_record)

        return bet_record

    def get_clv_summary(self) -> Dict:
        """Get summary statistics of CLV performance."""
        if not self.bets:
            return {"error": "No bets recorded"}

        clv_values = [b["clv"] for b in self.bets]
        clv_vs_opening = [b["clv_vs_opening"] for b in self.bets]

        # Split by result
        won_bets = [b for b in self.bets if b.get("won") is True]
        lost_bets = [b for b in self.bets if b.get("won") is False]

        return {
            "total_bets": len(self.bets),
            "avg_clv": np.mean(clv_values),
            "avg_clv_pct": np.mean(clv_values) * 100,
            "median_clv": np.median(clv_values),
            "std_clv": np.std(clv_values),
            "positive_clv_count": sum(1 for c in clv_values if c > 0),
            "positive_clv_pct": sum(1 for c in clv_values if c > 0) / len(clv_values) * 100,
            "avg_clv_vs_opening": np.mean(clv_vs_opening),
            "avg_clv_winners": np.mean([b["clv"] for b in won_bets]) if won_bets else None,
            "avg_clv_losers": np.mean([b["clv"] for b in lost_bets]) if lost_bets else None,
        }

    def is_profitable_long_term(self) -> Tuple[bool, str]:
        """
        Assess if betting strategy shows signs of long-term profitability.

        Based on CLV analysis rather than short-term results.

        Returns:
            Tuple of (is_profitable, explanation)
        """
        if len(self.bets) < 50:
            return (None, f"Insufficient data: {len(self.bets)} bets. Need 50+ for assessment.")

        summary = self.get_clv_summary()

        avg_clv = summary["avg_clv"]
        positive_clv_pct = summary["positive_clv_pct"]

        # Criteria for likely profitable strategy:
        # 1. Average CLV > 1% (consistently beating closing line)
        # 2. >55% of bets have positive CLV

        if avg_clv > 0.01 and positive_clv_pct > 55:
            return (True, f"Strong positive CLV: avg {avg_clv:.2%}, {positive_clv_pct:.0f}% positive. "
                         "Strategy shows signs of genuine edge.")
        elif avg_clv > 0 and positive_clv_pct > 50:
            return (True, f"Marginal positive CLV: avg {avg_clv:.2%}, {positive_clv_pct:.0f}% positive. "
                         "Strategy may have edge but needs more data to confirm.")
        elif avg_clv < -0.02:
            return (False, f"Significant negative CLV: avg {avg_clv:.2%}. "
                          "Market is pricing better - strategy likely unprofitable long-term.")
        else:
            return (None, f"CLV near zero: avg {avg_clv:.2%}. "
                         "No clear edge detected - may be positive or negative variance.")


class RiskLevel(Enum):
    """Risk levels for betting operations."""
    NORMAL = "normal"           # Full stakes allowed
    CAUTION = "caution"         # Reduced stakes (75%)
    WARNING = "warning"         # Heavily reduced stakes (50%)
    CRITICAL = "critical"       # Minimal stakes (25%)
    HALT = "halt"               # No betting allowed


class HaltReason(Enum):
    """Reasons for halting betting."""
    NONE = "none"
    DAILY_LIMIT = "daily_limit_exceeded"
    WEEKLY_LIMIT = "weekly_limit_exceeded"
    DRAWDOWN_LIMIT = "drawdown_limit_exceeded"
    LOSING_STREAK = "losing_streak"
    MANUAL_HALT = "manual_halt"


@dataclass
class RiskStatus:
    """Current risk status for betting operations."""
    risk_level: RiskLevel
    stake_multiplier: float  # 0.0 to 1.0
    halt_reason: HaltReason
    message: str
    current_drawdown_pct: float
    daily_pnl_pct: float
    weekly_pnl_pct: float
    current_streak: int  # Negative = losing streak
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['risk_level'] = self.risk_level.value
        d['halt_reason'] = self.halt_reason.value
        d['timestamp'] = self.timestamp.isoformat()
        return d

    def is_betting_allowed(self) -> bool:
        return self.risk_level != RiskLevel.HALT


@dataclass
class BankrollSnapshot:
    """Point-in-time snapshot of bankroll."""
    timestamp: datetime
    balance: float
    peak_balance: float
    drawdown_pct: float
    daily_pnl: float
    weekly_pnl: float


class DrawdownProtection:
    """
    Drawdown protection system.

    Implements graduated stake reduction during drawdowns to protect capital.
    Based on professional trading risk management principles.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,
        max_weekly_loss_pct: float = 0.15,
        drawdown_halt_pct: float = 0.25,
        losing_streak_halt: int = 8
    ):
        """
        Args:
            max_daily_loss_pct: Maximum daily loss as % of bankroll (default 5%)
            max_weekly_loss_pct: Maximum weekly loss as % of bankroll (default 15%)
            drawdown_halt_pct: Drawdown % to halt betting (default 25%)
            losing_streak_halt: Number of consecutive losses to halt (default 8)
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.drawdown_halt_pct = drawdown_halt_pct
        self.losing_streak_halt = losing_streak_halt

        # Graduated stake multipliers based on drawdown
        self.drawdown_tiers = {
            0.00: 1.00,  # 0-10% drawdown: full stakes
            0.10: 0.75,  # 10-20% drawdown: 75% stakes
            0.20: 0.50,  # 20-30% drawdown: 50% stakes
            0.30: 0.25,  # 30%+ drawdown: 25% stakes
        }

    def get_stake_multiplier(self, drawdown_pct: float) -> float:
        """
        Get stake multiplier based on current drawdown.

        Args:
            drawdown_pct: Current drawdown as decimal (0.15 = 15%)

        Returns:
            Stake multiplier (0.0 to 1.0)
        """
        if drawdown_pct >= self.drawdown_halt_pct:
            return 0.0

        multiplier = 1.0
        for threshold, mult in sorted(self.drawdown_tiers.items()):
            if drawdown_pct >= threshold:
                multiplier = mult

        return multiplier

    def check_limits(
        self,
        current_drawdown_pct: float,
        daily_pnl_pct: float,
        weekly_pnl_pct: float,
        current_streak: int = 0
    ) -> RiskStatus:
        """
        Check all risk limits and return current status.

        Args:
            current_drawdown_pct: Current drawdown from peak
            daily_pnl_pct: Today's P&L as % of bankroll
            weekly_pnl_pct: This week's P&L as % of bankroll
            current_streak: Current win/loss streak (negative = losing)

        Returns:
            RiskStatus with current risk level and recommendations
        """
        halt_reason = HaltReason.NONE
        risk_level = RiskLevel.NORMAL
        message = "Normal operations"

        # Check drawdown limit (most critical)
        if current_drawdown_pct >= self.drawdown_halt_pct:
            return RiskStatus(
                risk_level=RiskLevel.HALT,
                stake_multiplier=0.0,
                halt_reason=HaltReason.DRAWDOWN_LIMIT,
                message=f"HALT: Drawdown {current_drawdown_pct:.1%} exceeds limit {self.drawdown_halt_pct:.1%}",
                current_drawdown_pct=current_drawdown_pct,
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                current_streak=current_streak
            )

        # Check daily loss limit
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            return RiskStatus(
                risk_level=RiskLevel.HALT,
                stake_multiplier=0.0,
                halt_reason=HaltReason.DAILY_LIMIT,
                message=f"HALT: Daily loss {daily_pnl_pct:.1%} exceeds limit {-self.max_daily_loss_pct:.1%}",
                current_drawdown_pct=current_drawdown_pct,
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                current_streak=current_streak
            )

        # Check weekly loss limit
        if weekly_pnl_pct <= -self.max_weekly_loss_pct:
            return RiskStatus(
                risk_level=RiskLevel.HALT,
                stake_multiplier=0.0,
                halt_reason=HaltReason.WEEKLY_LIMIT,
                message=f"HALT: Weekly loss {weekly_pnl_pct:.1%} exceeds limit {-self.max_weekly_loss_pct:.1%}",
                current_drawdown_pct=current_drawdown_pct,
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                current_streak=current_streak
            )

        # Check losing streak
        if current_streak <= -self.losing_streak_halt:
            return RiskStatus(
                risk_level=RiskLevel.HALT,
                stake_multiplier=0.0,
                halt_reason=HaltReason.LOSING_STREAK,
                message=f"HALT: Losing streak of {-current_streak} exceeds limit {self.losing_streak_halt}",
                current_drawdown_pct=current_drawdown_pct,
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                current_streak=current_streak
            )

        # Get stake multiplier based on drawdown
        stake_multiplier = self.get_stake_multiplier(current_drawdown_pct)

        # Determine risk level based on conditions
        if current_drawdown_pct >= 0.20:
            risk_level = RiskLevel.CRITICAL
            message = f"CRITICAL: High drawdown ({current_drawdown_pct:.1%}). Stakes at {stake_multiplier:.0%}"
        elif current_drawdown_pct >= 0.10:
            risk_level = RiskLevel.WARNING
            message = f"WARNING: Elevated drawdown ({current_drawdown_pct:.1%}). Stakes at {stake_multiplier:.0%}"
        elif daily_pnl_pct <= -0.03 or weekly_pnl_pct <= -0.08:
            risk_level = RiskLevel.CAUTION
            message = f"CAUTION: Approaching limits. Daily: {daily_pnl_pct:.1%}, Weekly: {weekly_pnl_pct:.1%}"
        elif current_streak <= -4:
            risk_level = RiskLevel.CAUTION
            stake_multiplier = min(stake_multiplier, 0.75)
            message = f"CAUTION: Losing streak of {-current_streak}. Stakes reduced."

        return RiskStatus(
            risk_level=risk_level,
            stake_multiplier=stake_multiplier,
            halt_reason=halt_reason,
            message=message,
            current_drawdown_pct=current_drawdown_pct,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl_pct=weekly_pnl_pct,
            current_streak=current_streak
        )


class BankrollManager:
    """
    Comprehensive bankroll management system.

    Tracks bankroll, calculates drawdowns, enforces limits,
    and provides position sizing recommendations.
    """

    def __init__(
        self,
        initial_bankroll: float,
        max_daily_loss_pct: float = 0.05,
        max_weekly_loss_pct: float = 0.15,
        max_drawdown_pct: float = 0.25,
        max_single_bet_pct: float = 0.05,
        losing_streak_halt: int = 8
    ):
        """
        Args:
            initial_bankroll: Starting bankroll amount
            max_daily_loss_pct: Maximum daily loss as % of bankroll
            max_weekly_loss_pct: Maximum weekly loss as % of bankroll
            max_drawdown_pct: Maximum drawdown to halt betting
            max_single_bet_pct: Maximum single bet as % of bankroll
            losing_streak_halt: Number of consecutive losses to halt
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll
        self.max_single_bet_pct = max_single_bet_pct

        # Initialize drawdown protection
        self.drawdown_protection = DrawdownProtection(
            max_daily_loss_pct=max_daily_loss_pct,
            max_weekly_loss_pct=max_weekly_loss_pct,
            drawdown_halt_pct=max_drawdown_pct,
            losing_streak_halt=losing_streak_halt
        )

        # Tracking
        self.bet_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}  # date -> pnl
        self.snapshots: List[BankrollSnapshot] = []
        self.current_streak: int = 0
        self.manual_halt: bool = False

    @property
    def current_drawdown_pct(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_bankroll <= 0:
            return 0.0
        return (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll

    def get_daily_pnl_pct(self) -> float:
        """Get today's P&L as percentage of start-of-day bankroll."""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_pnl = self.daily_pnl.get(today, 0.0)

        # Estimate start-of-day bankroll
        sod_bankroll = self.current_bankroll - daily_pnl
        if sod_bankroll <= 0:
            return 0.0

        return daily_pnl / sod_bankroll

    def get_weekly_pnl_pct(self) -> float:
        """Get this week's P&L as percentage of start-of-week bankroll."""
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())

        weekly_pnl = 0.0
        for date_str, pnl in self.daily_pnl.items():
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if date >= week_start:
                    weekly_pnl += pnl
            except ValueError:
                continue

        # Estimate start-of-week bankroll
        sow_bankroll = self.current_bankroll - weekly_pnl
        if sow_bankroll <= 0:
            return 0.0

        return weekly_pnl / sow_bankroll

    def update_bankroll(self, pnl: float, bet_won: bool) -> None:
        """
        Update bankroll after a bet settles.

        Args:
            pnl: Profit/loss from the bet
            bet_won: Whether the bet won
        """
        self.current_bankroll += pnl

        # Update peak
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll

        # Update daily tracking
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_pnl[today] = self.daily_pnl.get(today, 0.0) + pnl

        # Update streak
        if bet_won:
            if self.current_streak > 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
        else:
            if self.current_streak < 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1

        # Log bet
        self.bet_history.append({
            "timestamp": datetime.now().isoformat(),
            "pnl": pnl,
            "won": bet_won,
            "bankroll_after": self.current_bankroll,
            "drawdown_pct": self.current_drawdown_pct
        })

        # Create snapshot
        self.snapshots.append(BankrollSnapshot(
            timestamp=datetime.now(),
            balance=self.current_bankroll,
            peak_balance=self.peak_bankroll,
            drawdown_pct=self.current_drawdown_pct,
            daily_pnl=self.daily_pnl.get(today, 0.0),
            weekly_pnl=sum(self.daily_pnl.values())  # Approximation
        ))

    def get_risk_status(self) -> RiskStatus:
        """Get current risk status."""
        if self.manual_halt:
            return RiskStatus(
                risk_level=RiskLevel.HALT,
                stake_multiplier=0.0,
                halt_reason=HaltReason.MANUAL_HALT,
                message="HALT: Manual halt active",
                current_drawdown_pct=self.current_drawdown_pct,
                daily_pnl_pct=self.get_daily_pnl_pct(),
                weekly_pnl_pct=self.get_weekly_pnl_pct(),
                current_streak=self.current_streak
            )

        return self.drawdown_protection.check_limits(
            current_drawdown_pct=self.current_drawdown_pct,
            daily_pnl_pct=self.get_daily_pnl_pct(),
            weekly_pnl_pct=self.get_weekly_pnl_pct(),
            current_streak=self.current_streak
        )

    def calculate_position_size(
        self,
        kelly_fraction: float,
        edge_quality_score: float = 100.0,
        confidence: str = "medium"
    ) -> float:
        """
        Calculate recommended position size.

        Args:
            kelly_fraction: Kelly criterion stake as fraction of bankroll
            edge_quality_score: Edge quality score (0-100)
            confidence: Confidence level ('high', 'medium', 'low')

        Returns:
            Recommended stake amount
        """
        risk_status = self.get_risk_status()

        if not risk_status.is_betting_allowed():
            logger.warning(f"Betting halted: {risk_status.message}")
            return 0.0

        # Start with Kelly fraction
        base_stake_pct = kelly_fraction

        # Apply risk multiplier from drawdown protection
        adjusted_stake_pct = base_stake_pct * risk_status.stake_multiplier

        # Apply edge quality adjustment
        quality_multiplier = edge_quality_score / 100.0
        adjusted_stake_pct *= quality_multiplier

        # Apply confidence adjustment
        confidence_multipliers = {
            "high": 1.0,
            "medium": 0.6,
            "low": 0.3
        }
        adjusted_stake_pct *= confidence_multipliers.get(confidence, 0.5)

        # Cap at maximum single bet percentage
        adjusted_stake_pct = min(adjusted_stake_pct, self.max_single_bet_pct)

        # Calculate actual stake
        stake = self.current_bankroll * adjusted_stake_pct

        # Ensure minimum viable bet (avoid dust bets)
        min_bet = 1.0  # $1 minimum
        if stake < min_bet:
            stake = 0.0

        return round(stake, 2)

    def set_manual_halt(self, halt: bool = True) -> None:
        """Set or clear manual halt."""
        self.manual_halt = halt
        if halt:
            logger.warning("Manual halt activated")
        else:
            logger.info("Manual halt cleared")

    def get_stats(self) -> Dict:
        """Get comprehensive bankroll statistics."""
        total_bets = len(self.bet_history)
        wins = sum(1 for b in self.bet_history if b['won'])
        losses = total_bets - wins

        return {
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": self.current_bankroll,
            "peak_bankroll": self.peak_bankroll,
            "current_drawdown_pct": self.current_drawdown_pct,
            "total_pnl": self.current_bankroll - self.initial_bankroll,
            "total_pnl_pct": (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll,
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_bets if total_bets > 0 else 0.0,
            "current_streak": self.current_streak,
            "daily_pnl_pct": self.get_daily_pnl_pct(),
            "weekly_pnl_pct": self.get_weekly_pnl_pct(),
            "risk_status": self.get_risk_status().to_dict()
        }

    def save(self, filepath: str = "bankroll_state.json") -> None:
        """Save bankroll state to disk."""
        state = {
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": self.current_bankroll,
            "peak_bankroll": self.peak_bankroll,
            "daily_pnl": self.daily_pnl,
            "current_streak": self.current_streak,
            "manual_halt": self.manual_halt,
            "bet_history": self.bet_history[-100:],  # Keep last 100 bets
            "saved_at": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Bankroll state saved to {filepath}")

    def load(self, filepath: str = "bankroll_state.json") -> "BankrollManager":
        """Load bankroll state from disk."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.initial_bankroll = state.get("initial_bankroll", self.initial_bankroll)
            self.current_bankroll = state.get("current_bankroll", self.current_bankroll)
            self.peak_bankroll = state.get("peak_bankroll", self.peak_bankroll)
            self.daily_pnl = state.get("daily_pnl", {})
            self.current_streak = state.get("current_streak", 0)
            self.manual_halt = state.get("manual_halt", False)
            self.bet_history = state.get("bet_history", [])

            logger.info(f"Bankroll state loaded from {filepath}")

        except FileNotFoundError:
            logger.warning(f"No saved state found at {filepath}")

        return self


class DynamicKellyCalculator:
    """
    Dynamic Kelly Criterion calculator with risk adjustments.

    Standard Kelly can be too aggressive. This implementation provides:
    1. Fractional Kelly (typically 25% of full Kelly)
    2. Drawdown-adjusted Kelly
    3. Edge-quality adjusted Kelly
    4. Correlation adjustment for multiple same-day bets
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_kelly_bet: float = 0.05,
        correlation_factor: float = 0.15
    ):
        """
        Args:
            kelly_fraction: Fraction of full Kelly to use (default 25%)
            max_kelly_bet: Maximum bet as fraction of bankroll
            correlation_factor: Assumed correlation between same-day NBA bets
        """
        self.kelly_fraction = kelly_fraction
        self.max_kelly_bet = max_kelly_bet
        self.correlation_factor = correlation_factor

    def calculate_kelly(
        self,
        win_probability: float,
        decimal_odds: float
    ) -> float:
        """
        Calculate full Kelly criterion stake.

        Kelly formula: f* = (bp - q) / b
        Where:
            b = decimal odds - 1 (net odds)
            p = probability of winning
            q = probability of losing (1-p)

        Args:
            win_probability: Probability of winning (0-1)
            decimal_odds: Decimal odds (e.g., 2.0 for even money)

        Returns:
            Kelly stake as fraction of bankroll
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0

        if decimal_odds <= 1:
            return 0.0

        b = decimal_odds - 1  # Net odds
        p = win_probability
        q = 1 - p

        kelly = (b * p - q) / b

        # Kelly can be negative if edge is negative
        return max(0.0, kelly)

    def calculate_uncertainty_adjusted_kelly(
        self,
        win_probability: float,
        decimal_odds: float,
        probability_std: float = 0.05
    ) -> float:
        """
        Calculate Kelly with uncertainty adjustment for probability estimation error.

        When you're uncertain about your probability estimate, you should bet less
        than full Kelly suggests. This accounts for estimation error.

        Based on: "Kelly Criterion with Uncertainty" by Ed Thorp

        Args:
            win_probability: Estimated probability of winning
            decimal_odds: Decimal odds
            probability_std: Standard deviation of probability estimate
                           (how uncertain are you? 0.05 = fairly confident)

        Returns:
            Uncertainty-adjusted Kelly fraction
        """
        # Base Kelly
        base_kelly = self.calculate_kelly(win_probability, decimal_odds)

        if base_kelly <= 0 or probability_std <= 0:
            return base_kelly

        # Adjust for uncertainty
        # The more uncertain, the more we reduce Kelly
        # Formula: adjusted_kelly = kelly * (1 - variance/edge_squared)
        # where variance = probability_std^2

        b = decimal_odds - 1  # Net odds
        implied_prob = 1 / decimal_odds
        edge = win_probability - implied_prob

        if edge <= 0:
            return 0.0

        variance = probability_std ** 2
        edge_squared = edge ** 2

        # Uncertainty penalty (bounded to not go negative)
        uncertainty_factor = max(0.25, 1 - (variance / (edge_squared + 0.001)))

        adjusted_kelly = base_kelly * uncertainty_factor

        return max(0.0, adjusted_kelly)

    def calculate_dynamic_kelly(
        self,
        win_probability: float,
        decimal_odds: float,
        edge_quality_score: float = 100.0,
        current_drawdown: float = 0.0,
        num_same_day_bets: int = 1,
        probability_std: float = None
    ) -> Dict:
        """
        Calculate dynamically adjusted Kelly stake.

        IMPROVEMENTS over basic Kelly:
        1. Fractional Kelly (25% by default) for safety
        2. Edge quality adjustment - reduce stakes for lower quality edges
        3. Drawdown adjustment - reduce stakes during drawdowns
        4. Correlation adjustment - account for multiple same-day bets
        5. Uncertainty adjustment - account for probability estimation error

        Args:
            win_probability: Probability of winning (0-1)
            decimal_odds: Decimal odds
            edge_quality_score: Edge quality score (0-100)
            current_drawdown: Current drawdown as decimal
            num_same_day_bets: Number of bets placed today
            probability_std: Standard deviation of probability estimate
                           (optional - if provided, applies uncertainty adjustment)

        Returns:
            Dict with kelly calculations and adjustments
        """
        # Calculate base Kelly (with optional uncertainty adjustment)
        if probability_std is not None and probability_std > 0:
            full_kelly = self.calculate_uncertainty_adjusted_kelly(
                win_probability, decimal_odds, probability_std
            )
        else:
            full_kelly = self.calculate_kelly(win_probability, decimal_odds)

        # Apply fractional Kelly
        fractional_kelly = full_kelly * self.kelly_fraction

        # Edge quality adjustment (lower quality = smaller bet)
        quality_multiplier = edge_quality_score / 100.0
        quality_adjusted = fractional_kelly * quality_multiplier

        # Drawdown adjustment (higher drawdown = smaller bet)
        drawdown_multiplier = max(0.25, 1.0 - current_drawdown * 2)
        drawdown_adjusted = quality_adjusted * drawdown_multiplier

        # Correlation adjustment for same-day bets
        # NBA games have ~15-20% correlation due to shared factors
        if num_same_day_bets > 1:
            correlation_adj = 1.0 - (self.correlation_factor * (num_same_day_bets - 1))
            correlation_adjusted = drawdown_adjusted * max(0.25, correlation_adj)
        else:
            correlation_adjusted = drawdown_adjusted

        # Cap at maximum
        final_kelly = min(correlation_adjusted, self.max_kelly_bet)

        return {
            "full_kelly": full_kelly,
            "fractional_kelly": fractional_kelly,
            "quality_adjusted": quality_adjusted,
            "drawdown_adjusted": drawdown_adjusted,
            "final_kelly": final_kelly,
            "adjustments": {
                "kelly_fraction": self.kelly_fraction,
                "quality_multiplier": quality_multiplier,
                "drawdown_multiplier": drawdown_multiplier,
                "correlation_adjustment": 1.0 - self.correlation_factor * max(0, num_same_day_bets - 1)
            }
        }


def calculate_recommended_stake(
    bankroll: float,
    win_probability: float,
    decimal_odds: float,
    edge_quality_score: float = 100.0,
    current_drawdown: float = 0.0,
    confidence: str = "medium",
    kelly_fraction: float = 0.25
) -> Dict:
    """
    Convenience function to calculate recommended stake.

    Args:
        bankroll: Current bankroll
        win_probability: Model's win probability
        decimal_odds: Decimal odds for the bet
        edge_quality_score: Edge quality score (0-100)
        current_drawdown: Current drawdown as decimal
        confidence: 'high', 'medium', or 'low'
        kelly_fraction: Kelly fraction to use

    Returns:
        Dict with stake recommendation and details
    """
    kelly_calc = DynamicKellyCalculator(kelly_fraction=kelly_fraction)

    kelly_result = kelly_calc.calculate_dynamic_kelly(
        win_probability=win_probability,
        decimal_odds=decimal_odds,
        edge_quality_score=edge_quality_score,
        current_drawdown=current_drawdown
    )

    # Apply confidence adjustment
    confidence_multipliers = {"high": 1.0, "medium": 0.6, "low": 0.3}
    confidence_mult = confidence_multipliers.get(confidence, 0.5)

    final_fraction = kelly_result["final_kelly"] * confidence_mult
    recommended_stake = bankroll * final_fraction

    # Round to reasonable amount
    recommended_stake = round(recommended_stake, 2)

    return {
        "recommended_stake": recommended_stake,
        "stake_fraction": final_fraction,
        "kelly_details": kelly_result,
        "confidence_multiplier": confidence_mult,
        "bankroll": bankroll
    }


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Risk Management Demo")
    print("=" * 60)

    # Create bankroll manager
    manager = BankrollManager(
        initial_bankroll=10000.0,
        max_daily_loss_pct=0.05,
        max_weekly_loss_pct=0.15,
        max_drawdown_pct=0.25
    )

    print(f"\nInitial bankroll: ${manager.current_bankroll:,.2f}")

    # Simulate some bets
    bets = [
        (100, True),   # Win $100
        (-110, False), # Lose $110
        (150, True),   # Win $150
        (-110, False), # Lose $110
        (-110, False), # Lose $110
        (200, True),   # Win $200
    ]

    for pnl, won in bets:
        manager.update_bankroll(pnl, won)
        status = manager.get_risk_status()
        print(f"\nBet result: {'Won' if won else 'Lost'} ${abs(pnl)}")
        print(f"  Bankroll: ${manager.current_bankroll:,.2f}")
        print(f"  Drawdown: {manager.current_drawdown_pct:.1%}")
        print(f"  Risk Level: {status.risk_level.value}")
        print(f"  Stake Multiplier: {status.stake_multiplier:.0%}")

    # Test position sizing
    print("\n" + "=" * 60)
    print("Position Sizing Demo")
    print("=" * 60)

    result = calculate_recommended_stake(
        bankroll=10000,
        win_probability=0.55,
        decimal_odds=1.91,  # -110 odds
        edge_quality_score=80,
        current_drawdown=0.05,
        confidence="medium"
    )

    print(f"\nFor a bet with 55% probability at -110 odds:")
    print(f"  Recommended stake: ${result['recommended_stake']:,.2f}")
    print(f"  Stake fraction: {result['stake_fraction']:.2%}")
    print(f"  Full Kelly would be: {result['kelly_details']['full_kelly']:.2%}")

    print("\n" + "=" * 60)
    print("Stats Summary")
    print("=" * 60)
    stats = manager.get_stats()
    for key, value in stats.items():
        if key != "risk_status":
            print(f"  {key}: {value}")
