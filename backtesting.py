"""
Backtesting Framework for NBA Betting Model

Provides proper time-series validation to evaluate betting model performance.
Key principle: NEVER use future data to predict past events (look-ahead bias).

Features:
1. Walk-Forward Validation - train on past, test on future
2. Betting Simulation - realistic P&L tracking with juice/vig
3. Closing Line Value (CLV) - compare your bets to closing odds
4. Comprehensive metrics - ROI, win rate, max drawdown, Sharpe ratio

Critical for betting because:
- Past performance doesn't guarantee future results
- But proper backtesting helps identify real edges vs luck
- Accounts for betting costs (juice/vig)
"""

import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import warnings

# Import edge quality scoring for dynamic Kelly
try:
    from edge_quality import (
        EdgeQualityScorer, DynamicKellyCalculator, EdgeQualityResult,
        EdgeTier, american_to_decimal, decimal_to_implied_prob
    )
    HAS_EDGE_QUALITY = True
except ImportError:
    HAS_EDGE_QUALITY = False
    print("Warning: edge_quality module not found. Dynamic Kelly disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BetType(Enum):
    """Types of bets supported."""
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"


class BetOutcome(Enum):
    """Possible bet outcomes."""
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    PENDING = "pending"


@dataclass
class Bet:
    """Represents a single bet."""
    bet_id: str
    game_id: str
    bet_type: BetType
    selection: str  # e.g., "LAL", "Over", "LeBron James Points"
    odds: float  # American odds (e.g., -110, +150)
    stake: float  # Amount wagered
    predicted_probability: float  # Model's probability
    implied_probability: float  # Odds-implied probability
    edge: float  # predicted - implied
    placed_at: datetime
    closing_odds: Optional[float] = None  # For CLV calculation
    outcome: BetOutcome = BetOutcome.PENDING
    pnl: float = 0.0  # Profit/loss
    actual_result: Optional[Any] = None  # Actual game/stat result

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['bet_type'] = self.bet_type.value
        d['outcome'] = self.outcome.value
        d['placed_at'] = self.placed_at.isoformat()
        return d

    @staticmethod
    def american_to_decimal(american_odds: float) -> float:
        """Convert American odds to decimal odds."""
        if american_odds >= 100:
            return 1 + (american_odds / 100)
        else:
            return 1 + (100 / abs(american_odds))

    @staticmethod
    def american_to_implied_prob(american_odds: float) -> float:
        """Convert American odds to implied probability."""
        if american_odds >= 100:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    def calculate_pnl(self, won: bool, pushed: bool = False) -> float:
        """Calculate profit/loss for this bet."""
        if pushed:
            self.outcome = BetOutcome.PUSH
            self.pnl = 0.0
        elif won:
            self.outcome = BetOutcome.WIN
            decimal_odds = self.american_to_decimal(self.odds)
            self.pnl = self.stake * (decimal_odds - 1)
        else:
            self.outcome = BetOutcome.LOSS
            self.pnl = -self.stake
        return self.pnl

    def closing_line_value(self) -> Optional[float]:
        """
        Calculate Closing Line Value (CLV).

        CLV = Our implied probability - Closing implied probability

        Positive CLV = we got a better line than the market closed at
        This is the single best predictor of long-term betting success.
        """
        if self.closing_odds is None:
            return None

        our_implied = self.american_to_implied_prob(self.odds)
        closing_implied = self.american_to_implied_prob(self.closing_odds)

        return closing_implied - our_implied


@dataclass
class BacktestPeriod:
    """Results for a single backtest period."""
    start_date: datetime
    end_date: datetime
    num_bets: int = 0
    num_wins: int = 0
    num_losses: int = 0
    num_pushes: int = 0
    total_staked: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    avg_odds: float = 0.0
    avg_edge: float = 0.0
    avg_clv: Optional[float] = None
    bets: List[Bet] = field(default_factory=list)

    def calculate_metrics(self) -> None:
        """Calculate aggregate metrics from bets."""
        if not self.bets:
            return

        self.num_bets = len(self.bets)
        self.num_wins = sum(1 for b in self.bets if b.outcome == BetOutcome.WIN)
        self.num_losses = sum(1 for b in self.bets if b.outcome == BetOutcome.LOSS)
        self.num_pushes = sum(1 for b in self.bets if b.outcome == BetOutcome.PUSH)

        self.total_staked = sum(b.stake for b in self.bets)
        self.total_pnl = sum(b.pnl for b in self.bets)

        self.roi = (self.total_pnl / self.total_staked * 100) if self.total_staked > 0 else 0.0
        self.win_rate = (self.num_wins / (self.num_wins + self.num_losses) * 100) if (self.num_wins + self.num_losses) > 0 else 0.0

        self.avg_odds = np.mean([b.odds for b in self.bets])
        self.avg_edge = np.mean([b.edge for b in self.bets])

        clv_values = [b.closing_line_value() for b in self.bets if b.closing_line_value() is not None]
        self.avg_clv = np.mean(clv_values) if clv_values else None

    def to_dict(self) -> Dict:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "num_bets": self.num_bets,
            "num_wins": self.num_wins,
            "num_losses": self.num_losses,
            "num_pushes": self.num_pushes,
            "total_staked": self.total_staked,
            "total_pnl": self.total_pnl,
            "roi": self.roi,
            "win_rate": self.win_rate,
            "avg_odds": self.avg_odds,
            "avg_edge": self.avg_edge,
            "avg_clv": self.avg_clv,
        }


@dataclass
class BacktestResult:
    """Complete backtest results."""
    model_name: str
    bet_type: BetType
    start_date: datetime
    end_date: datetime
    periods: List[BacktestPeriod] = field(default_factory=list)

    # Aggregate metrics
    total_bets: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_staked: float = 0.0
    total_pnl: float = 0.0
    overall_roi: float = 0.0
    overall_win_rate: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0

    # Quality metrics
    avg_clv: Optional[float] = None
    periods_profitable: int = 0
    longest_winning_streak: int = 0
    longest_losing_streak: int = 0

    # Calibration & EV metrics
    ece: Optional[float] = None  # Expected Calibration Error
    avg_ev: Optional[float] = None  # Average Expected Value per bet

    def calculate_aggregate_metrics(self) -> None:
        """Calculate aggregate metrics from all periods."""
        if not self.periods:
            return

        # Flatten all bets
        all_bets = []
        for period in self.periods:
            all_bets.extend(period.bets)

        if not all_bets:
            return

        self.total_bets = len(all_bets)
        self.total_wins = sum(1 for b in all_bets if b.outcome == BetOutcome.WIN)
        self.total_losses = sum(1 for b in all_bets if b.outcome == BetOutcome.LOSS)
        self.total_staked = sum(b.stake for b in all_bets)
        self.total_pnl = sum(b.pnl for b in all_bets)

        self.overall_roi = (self.total_pnl / self.total_staked * 100) if self.total_staked > 0 else 0.0
        self.overall_win_rate = (self.total_wins / (self.total_wins + self.total_losses) * 100) if (self.total_wins + self.total_losses) > 0 else 0.0

        # Calculate drawdown
        self._calculate_drawdown(all_bets)

        # Calculate Sharpe ratio (daily returns)
        self._calculate_sharpe(all_bets)

        # Profit factor
        total_wins_pnl = sum(b.pnl for b in all_bets if b.pnl > 0)
        total_losses_pnl = abs(sum(b.pnl for b in all_bets if b.pnl < 0))
        self.profit_factor = total_wins_pnl / total_losses_pnl if total_losses_pnl > 0 else float('inf')

        # CLV
        clv_values = [b.closing_line_value() for b in all_bets if b.closing_line_value() is not None]
        self.avg_clv = np.mean(clv_values) if clv_values else None

        # Periods profitable
        self.periods_profitable = sum(1 for p in self.periods if p.total_pnl > 0)

        # Streaks
        self._calculate_streaks(all_bets)

        # ECE and EV
        self._calculate_calibration_metrics(all_bets)

    def _calculate_drawdown(self, bets: List[Bet]) -> None:
        """Calculate maximum drawdown."""
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for bet in sorted(bets, key=lambda x: x.placed_at):
            cumulative += bet.pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown

        self.max_drawdown = max_dd
        self.max_drawdown_pct = (max_dd / peak * 100) if peak > 0 else 0.0

    def _calculate_sharpe(self, bets: List[Bet]) -> None:
        """Calculate Sharpe ratio using daily returns."""
        # Group by day
        daily_pnl: Dict[str, float] = {}
        for bet in bets:
            day = bet.placed_at.strftime("%Y-%m-%d")
            daily_pnl[day] = daily_pnl.get(day, 0.0) + bet.pnl

        if len(daily_pnl) < 2:
            self.sharpe_ratio = 0.0
            return

        returns = list(daily_pnl.values())
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        # Annualized (assuming ~300 betting days per year)
        if std_return > 0:
            self.sharpe_ratio = (mean_return / std_return) * np.sqrt(300)
        else:
            self.sharpe_ratio = 0.0

    def _calculate_streaks(self, bets: List[Bet]) -> None:
        """Calculate longest winning and losing streaks."""
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for bet in sorted(bets, key=lambda x: x.placed_at):
            if bet.outcome == BetOutcome.WIN:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif bet.outcome == BetOutcome.LOSS:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)

        self.longest_winning_streak = max_win_streak
        self.longest_losing_streak = max_loss_streak

    def _calculate_calibration_metrics(self, bets: List[Bet]) -> None:
        """
        Calculate Expected Calibration Error (ECE) and Average Expected Value.

        ECE measures how well-calibrated the model's probabilities are.
        A perfectly calibrated model has ECE = 0.

        EV (Expected Value) measures the average expected profit per bet
        based on model probability vs implied probability.
        """
        if not bets:
            self.ece = None
            self.avg_ev = None
            return

        # Calculate ECE using binned probabilities
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)

        predicted_probs = np.array([b.predicted_probability for b in bets])
        actual_outcomes = np.array([1 if b.outcome == BetOutcome.WIN else 0 for b in bets])

        ece_sum = 0.0
        total_samples = len(bets)

        for i in range(n_bins):
            mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])
            bin_count = np.sum(mask)

            if bin_count > 0:
                bin_confidence = np.mean(predicted_probs[mask])
                bin_accuracy = np.mean(actual_outcomes[mask])
                ece_sum += bin_count * abs(bin_accuracy - bin_confidence)

        self.ece = ece_sum / total_samples if total_samples > 0 else None

        # Calculate Average EV (Expected Value per bet)
        # EV = predicted_prob * potential_win - (1 - predicted_prob) * stake
        # Simplified: EV = edge * stake (where edge = pred_prob - implied_prob)
        total_ev = sum(b.edge * b.stake for b in bets)
        self.avg_ev = total_ev / len(bets) if bets else None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"BACKTEST RESULTS: {self.model_name} ({self.bet_type.value})",
            f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            "=" * 60,
            "",
            "PERFORMANCE:",
            f"  Total Bets: {self.total_bets}",
            f"  Record: {self.total_wins}W - {self.total_losses}L",
            f"  Win Rate: {self.overall_win_rate:.1f}%",
            f"  Total Staked: ${self.total_staked:,.2f}",
            f"  Total P&L: ${self.total_pnl:,.2f}",
            f"  ROI: {self.overall_roi:+.2f}%",
            "",
            "RISK METRICS:",
            f"  Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.1f}%)",
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"  Profit Factor: {self.profit_factor:.2f}",
            "",
            "QUALITY METRICS:",
            f"  Avg CLV: {self.avg_clv:.4f}" if self.avg_clv else "  Avg CLV: N/A",
            f"  Profitable Periods: {self.periods_profitable}/{len(self.periods)}",
            f"  Best Win Streak: {self.longest_winning_streak}",
            f"  Worst Loss Streak: {self.longest_losing_streak}",
            "",
        ]

        # Interpretation
        lines.append("INTERPRETATION:")
        if self.overall_roi > 5:
            lines.append("  - Strong positive ROI - potential edge exists")
        elif self.overall_roi > 0:
            lines.append("  - Slightly positive ROI - marginal edge")
        else:
            lines.append("  - Negative ROI - no edge detected")

        if self.avg_clv and self.avg_clv > 0.02:
            lines.append("  - Positive CLV - beating the market consistently")
        elif self.avg_clv and self.avg_clv > 0:
            lines.append("  - Slightly positive CLV - slight market advantage")
        elif self.avg_clv:
            lines.append("  - Negative CLV - market is pricing better")

        if self.sharpe_ratio > 1.5:
            lines.append("  - Excellent risk-adjusted returns")
        elif self.sharpe_ratio > 0.5:
            lines.append("  - Decent risk-adjusted returns")
        else:
            lines.append("  - Poor risk-adjusted returns")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "bet_type": self.bet_type.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_bets": self.total_bets,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "total_staked": self.total_staked,
            "total_pnl": self.total_pnl,
            "overall_roi": self.overall_roi,
            "overall_win_rate": self.overall_win_rate,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "avg_clv": self.avg_clv,
            "periods_profitable": self.periods_profitable,
            "longest_winning_streak": self.longest_winning_streak,
            "longest_losing_streak": self.longest_losing_streak,
            "ece": self.ece,  # Expected Calibration Error
            "avg_ev": self.avg_ev,  # Average Expected Value per bet
            "periods": [p.to_dict() for p in self.periods],
        }


class WalkForwardValidator:
    """
    Walk-Forward Validation for Time-Series Data

    Trains on expanding or rolling window, tests on future data.
    Prevents look-ahead bias in model evaluation.

    Example with expanding window:
    - Train on games 1-100, test on 101-120
    - Train on games 1-120, test on 121-140
    - Train on games 1-140, test on 141-160
    etc.
    """

    def __init__(
        self,
        train_model_fn: Callable,
        predict_fn: Callable,
        min_train_size: int = 100,
        test_size: int = 20,
        expanding: bool = True,
        step_size: int = None
    ):
        """
        Args:
            train_model_fn: Function to train model, signature: (train_data) -> model
            predict_fn: Function to make predictions, signature: (model, test_data) -> predictions
            min_train_size: Minimum number of samples for initial training
            test_size: Number of samples in each test set
            expanding: If True, use expanding window. If False, use rolling window.
            step_size: How many samples to move forward each iteration (default: test_size)
        """
        self.train_model_fn = train_model_fn
        self.predict_fn = predict_fn
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.expanding = expanding
        self.step_size = step_size or test_size

    def validate(
        self,
        data: List[Dict],
        date_key: str = "date"
    ) -> List[Tuple[Any, List[Dict], List[Dict], np.ndarray]]:
        """
        Perform walk-forward validation.

        Args:
            data: List of samples (dicts) sorted by date
            date_key: Key for date field in data dicts

        Returns:
            List of (model, train_data, test_data, predictions) tuples
        """
        # Sort by date
        data = sorted(data, key=lambda x: x[date_key])
        n = len(data)

        if n < self.min_train_size + self.test_size:
            raise ValueError(f"Not enough data. Need at least {self.min_train_size + self.test_size}, got {n}")

        results = []
        train_end = self.min_train_size

        while train_end + self.test_size <= n:
            # Define train and test sets
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - self.min_train_size)

            train_data = data[train_start:train_end]
            test_data = data[train_end:train_end + self.test_size]

            logger.info(f"Walk-forward: train[{train_start}:{train_end}], test[{train_end}:{train_end + self.test_size}]")

            # Train model
            model = self.train_model_fn(train_data)

            # Make predictions
            predictions = self.predict_fn(model, test_data)

            results.append((model, train_data, test_data, predictions))

            train_end += self.step_size

        logger.info(f"Walk-forward validation complete: {len(results)} periods")
        return results


class BettingSimulator:
    """
    Simulates betting with realistic conditions.

    Features:
    - Accounts for juice/vig
    - Multiple staking strategies (flat, Kelly, percentage)
    - Tracks bankroll over time
    - Calculates CLV when closing odds available
    """

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        staking_strategy: str = "flat",  # flat, kelly, percentage
        flat_stake: float = 100.0,
        kelly_fraction: float = 0.25,  # Fraction of Kelly to use
        percentage_stake: float = 0.02,  # 2% of bankroll
        min_edge: float = 0.02,  # Minimum edge to place bet
        max_stake_pct: float = 0.05,  # Max stake as % of bankroll
    ):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.staking_strategy = staking_strategy
        self.flat_stake = flat_stake
        self.kelly_fraction = kelly_fraction
        self.percentage_stake = percentage_stake
        self.min_edge = min_edge
        self.max_stake_pct = max_stake_pct

        self.bets: List[Bet] = []
        self.bankroll_history: List[Tuple[datetime, float]] = [(datetime.now(), initial_bankroll)]

    def calculate_stake(
        self,
        probability: float,
        odds: float
    ) -> float:
        """Calculate stake based on strategy."""
        if self.staking_strategy == "flat":
            stake = self.flat_stake

        elif self.staking_strategy == "kelly":
            # Kelly criterion
            decimal_odds = Bet.american_to_decimal(odds)
            b = decimal_odds - 1  # Net odds
            p = probability
            q = 1 - p

            kelly = (b * p - q) / b if b > 0 else 0
            kelly = max(0, kelly)  # No negative stakes

            # Apply Kelly fraction (quarter Kelly is common)
            stake = self.bankroll * kelly * self.kelly_fraction

        elif self.staking_strategy == "percentage":
            stake = self.bankroll * self.percentage_stake

        else:
            stake = self.flat_stake

        # Apply maximum stake limit
        max_stake = self.bankroll * self.max_stake_pct
        stake = min(stake, max_stake, self.bankroll)

        return round(stake, 2)

    def should_bet(
        self,
        predicted_prob: float,
        odds: float
    ) -> bool:
        """Determine if we should place this bet."""
        implied_prob = Bet.american_to_implied_prob(odds)
        edge = predicted_prob - implied_prob
        return edge >= self.min_edge

    def place_bet(
        self,
        game_id: str,
        bet_type: BetType,
        selection: str,
        odds: float,
        predicted_prob: float,
        placed_at: datetime = None,
        closing_odds: float = None
    ) -> Optional[Bet]:
        """
        Place a bet if it meets criteria.

        Returns:
            Bet object if placed, None otherwise
        """
        implied_prob = Bet.american_to_implied_prob(odds)
        edge = predicted_prob - implied_prob

        if not self.should_bet(predicted_prob, odds):
            return None

        stake = self.calculate_stake(predicted_prob, odds)

        if stake < 1:  # Minimum $1 bet
            return None

        bet = Bet(
            bet_id=f"{game_id}_{bet_type.value}_{len(self.bets)}",
            game_id=game_id,
            bet_type=bet_type,
            selection=selection,
            odds=odds,
            stake=stake,
            predicted_probability=predicted_prob,
            implied_probability=implied_prob,
            edge=edge,
            placed_at=placed_at or datetime.now(),
            closing_odds=closing_odds
        )

        self.bets.append(bet)
        return bet

    def settle_bet(
        self,
        bet_id: str,
        won: bool,
        pushed: bool = False,
        actual_result: Any = None
    ) -> float:
        """
        Settle a bet and update bankroll.

        Returns:
            P&L for this bet
        """
        bet = next((b for b in self.bets if b.bet_id == bet_id), None)
        if not bet:
            raise ValueError(f"Bet not found: {bet_id}")

        bet.actual_result = actual_result
        pnl = bet.calculate_pnl(won, pushed)

        self.bankroll += pnl
        self.bankroll_history.append((datetime.now(), self.bankroll))

        return pnl

    def simulate_from_predictions(
        self,
        predictions: List[Dict],
        results: List[Dict]
    ) -> List[Bet]:
        """
        Simulate betting from a list of predictions and results.

        Args:
            predictions: List of prediction dicts with keys:
                - game_id, bet_type, selection, odds, predicted_prob, date, closing_odds (optional)
            results: List of result dicts with keys:
                - game_id, winner (or actual_value for props), home_score, away_score

        Returns:
            List of settled bets
        """
        # Create result lookup
        result_lookup = {r["game_id"]: r for r in results}

        for pred in predictions:
            bet = self.place_bet(
                game_id=pred["game_id"],
                bet_type=BetType(pred.get("bet_type", "moneyline")),
                selection=pred["selection"],
                odds=pred["odds"],
                predicted_prob=pred["predicted_prob"],
                placed_at=pred.get("date", datetime.now()) if isinstance(pred.get("date"), datetime) else datetime.now(),
                closing_odds=pred.get("closing_odds")
            )

            if bet and pred["game_id"] in result_lookup:
                result = result_lookup[pred["game_id"]]
                won = self._determine_winner(bet, result)
                self.settle_bet(bet.bet_id, won, actual_result=result)

        return self.bets

    def _determine_winner(self, bet: Bet, result: Dict) -> bool:
        """Determine if bet won based on result."""
        if bet.bet_type == BetType.MONEYLINE:
            return bet.selection == result.get("winner")
        elif bet.bet_type == BetType.SPREAD:
            # For spread, selection should include the spread number
            # e.g., "LAL -5.5" or "BOS +3.5"
            actual_diff = result.get("home_score", 0) - result.get("away_score", 0)
            # This is simplified - real implementation would parse the spread
            return bet.selection == result.get("spread_winner")
        elif bet.bet_type == BetType.TOTAL:
            total = result.get("home_score", 0) + result.get("away_score", 0)
            total_line = result.get("total_line", 0)
            if "Over" in bet.selection:
                return total > total_line
            else:
                return total < total_line
        elif bet.bet_type == BetType.PLAYER_PROP:
            actual_value = result.get("actual_value", 0)
            prop_line = result.get("prop_line", 0)
            if "Over" in bet.selection:
                return actual_value > prop_line
            else:
                return actual_value < prop_line
        return False

    def get_results(self, model_name: str, bet_type: BetType) -> BacktestResult:
        """Get backtest results."""
        if not self.bets:
            return BacktestResult(
                model_name=model_name,
                bet_type=bet_type,
                start_date=datetime.now(),
                end_date=datetime.now()
            )

        # Create single period with all bets
        period = BacktestPeriod(
            start_date=min(b.placed_at for b in self.bets),
            end_date=max(b.placed_at for b in self.bets),
            bets=self.bets.copy()
        )
        period.calculate_metrics()

        result = BacktestResult(
            model_name=model_name,
            bet_type=bet_type,
            start_date=period.start_date,
            end_date=period.end_date,
            periods=[period]
        )
        result.calculate_aggregate_metrics()

        return result


class DynamicKellyBettingSimulator(BettingSimulator):
    """
    Enhanced betting simulator with dynamic Kelly criterion and edge quality scoring.

    Improvements over basic Kelly:
    - Adjusts stake based on edge quality (ensemble agreement, line movement, etc.)
    - Reduces stakes during drawdowns
    - Accounts for losing streaks
    - Considers recent performance and volatility
    """

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        base_kelly_fraction: float = 0.25,
        min_edge: float = 0.02,
        max_stake_pct: float = 0.05,
        min_stake_pct: float = 0.005,
        enable_drawdown_protection: bool = True,
        enable_edge_quality: bool = True,
    ):
        """
        Initialize dynamic Kelly simulator.

        Args:
            initial_bankroll: Starting bankroll
            base_kelly_fraction: Base fraction of Kelly (0.25 = quarter Kelly)
            min_edge: Minimum edge to consider betting
            max_stake_pct: Maximum stake as percentage of bankroll
            min_stake_pct: Minimum stake as percentage of bankroll
            enable_drawdown_protection: Reduce stakes during drawdowns
            enable_edge_quality: Use edge quality scoring for stake sizing
        """
        super().__init__(
            initial_bankroll=initial_bankroll,
            staking_strategy="kelly",
            kelly_fraction=base_kelly_fraction,
            min_edge=min_edge,
            max_stake_pct=max_stake_pct,
        )

        self.base_kelly_fraction = base_kelly_fraction
        self.min_stake_pct = min_stake_pct
        self.enable_drawdown_protection = enable_drawdown_protection
        self.enable_edge_quality = enable_edge_quality and HAS_EDGE_QUALITY

        # Edge quality scorer
        if self.enable_edge_quality:
            self.edge_scorer = EdgeQualityScorer()
            self.kelly_calculator = DynamicKellyCalculator(
                base_kelly_fraction=base_kelly_fraction,
                max_bet_pct=max_stake_pct,
                min_bet_pct=min_stake_pct,
            )

        # Tracking for dynamic adjustments
        self.peak_bankroll = initial_bankroll
        self.consecutive_losses = 0
        self.recent_results: List[bool] = []  # Last 20 bets

    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_bankroll <= 0:
            return 0.0
        return max(0, (self.peak_bankroll - self.bankroll) / self.peak_bankroll)

    def get_recent_win_rate(self, window: int = 20) -> float:
        """Get win rate over last N bets."""
        if not self.recent_results:
            return 0.5
        recent = self.recent_results[-window:]
        return sum(recent) / len(recent) if recent else 0.5

    def calculate_stake(
        self,
        probability: float,
        odds: float,
        edge_quality_result: Optional['EdgeQualityResult'] = None,
    ) -> float:
        """
        Calculate stake with dynamic Kelly adjustments.

        Args:
            probability: Model's win probability
            odds: American odds
            edge_quality_result: Optional pre-computed edge quality

        Returns:
            Stake amount in dollars
        """
        # Basic validation
        implied_prob = Bet.american_to_implied_prob(odds)
        edge = probability - implied_prob

        if edge < self.min_edge:
            return 0.0

        # Use edge quality system if available
        if self.enable_edge_quality and edge_quality_result is not None:
            decimal_odds = Bet.american_to_decimal(odds)
            result = self.kelly_calculator.calculate_bet_size(
                bankroll=self.bankroll,
                probability=probability,
                decimal_odds=decimal_odds,
                edge_quality=edge_quality_result,
                current_drawdown=self.get_current_drawdown(),
                recent_win_rate=self.get_recent_win_rate(),
                consecutive_losses=self.consecutive_losses,
            )

            if result['should_bet']:
                return round(result['bet_amount'], 2)
            return 0.0

        # Fallback to adjusted Kelly without edge quality
        decimal_odds = Bet.american_to_decimal(odds)
        b = decimal_odds - 1
        p = probability
        q = 1 - p

        kelly = (b * p - q) / b if b > 0 else 0
        kelly = max(0, kelly)

        # Base Kelly fraction
        adjusted_kelly = kelly * self.base_kelly_fraction

        # Drawdown adjustment
        if self.enable_drawdown_protection:
            drawdown = self.get_current_drawdown()
            if drawdown >= 0.30:
                adjusted_kelly *= 0.25
            elif drawdown >= 0.20:
                adjusted_kelly *= 0.50
            elif drawdown >= 0.10:
                adjusted_kelly *= 0.75

            # Losing streak adjustment
            if self.consecutive_losses >= 5:
                adjusted_kelly *= 0.50
            elif self.consecutive_losses >= 3:
                adjusted_kelly *= 0.75

        # Apply min/max constraints
        stake_pct = max(self.min_stake_pct, min(self.max_stake_pct, adjusted_kelly))
        stake = self.bankroll * stake_pct

        return round(stake, 2)

    def place_bet_with_quality(
        self,
        game_id: str,
        bet_type: BetType,
        selection: str,
        odds: float,
        predicted_prob: float,
        individual_model_probs: Optional[Dict[str, float]] = None,
        raw_probability: Optional[float] = None,
        opening_odds: Optional[float] = None,
        public_betting_pct: Optional[float] = None,
        is_reverse_line_movement: bool = False,
        is_steam_move: bool = False,
        games_played: int = 30,
        is_home: bool = True,
        injury_impact: float = 0.0,
        travel_fatigue: float = 0.0,
        placed_at: datetime = None,
        closing_odds: float = None,
    ) -> Optional[Tuple[Bet, 'EdgeQualityResult']]:
        """
        Place a bet with full edge quality evaluation.

        Returns:
            Tuple of (Bet, EdgeQualityResult) if placed, None otherwise
        """
        implied_prob = Bet.american_to_implied_prob(odds)
        edge = predicted_prob - implied_prob

        if edge < self.min_edge:
            return None

        # Evaluate edge quality
        edge_quality = None
        if self.enable_edge_quality:
            edge_quality = self.edge_scorer.evaluate_edge(
                model_probability=predicted_prob,
                implied_probability=implied_prob,
                individual_model_predictions=individual_model_probs,
                raw_probability=raw_probability,
                opening_odds=opening_odds,
                current_odds=odds,
                public_betting_pct=public_betting_pct,
                is_reverse_line_movement=is_reverse_line_movement,
                is_steam_move=is_steam_move,
                games_played=games_played,
                home_away="home" if is_home else "away",
                injury_impact_score=injury_impact,
                travel_fatigue_score=travel_fatigue,
            )

            # Skip if edge quality is too low
            if edge_quality.tier == EdgeTier.AVOID:
                return None

        # Calculate stake
        stake = self.calculate_stake(predicted_prob, odds, edge_quality)

        if stake < 1:  # Minimum $1 bet
            return None

        bet = Bet(
            bet_id=f"{game_id}_{bet_type.value}_{len(self.bets)}",
            game_id=game_id,
            bet_type=bet_type,
            selection=selection,
            odds=odds,
            stake=stake,
            predicted_probability=predicted_prob,
            implied_probability=implied_prob,
            edge=edge,
            placed_at=placed_at or datetime.now(),
            closing_odds=closing_odds
        )

        self.bets.append(bet)
        return (bet, edge_quality)

    def settle_bet(self, bet: Bet, result: Dict) -> bool:
        """
        Settle bet and update tracking metrics.

        Returns:
            True if bet won, False otherwise
        """
        won = self._determine_outcome(bet, result)
        pushed = self._is_push(bet, result)

        if pushed:
            bet.calculate_pnl(won=False, pushed=True)
        else:
            bet.calculate_pnl(won=won)
            self.bankroll += bet.pnl

            # Update tracking
            if won:
                self.consecutive_losses = 0
                self.recent_results.append(True)
            else:
                self.consecutive_losses += 1
                self.recent_results.append(False)

            # Keep only last 50 results
            if len(self.recent_results) > 50:
                self.recent_results = self.recent_results[-50:]

            # Update peak bankroll
            if self.bankroll > self.peak_bankroll:
                self.peak_bankroll = self.bankroll

        self.bankroll_history.append((bet.placed_at, self.bankroll))
        return won

    def get_dynamic_metrics(self) -> Dict:
        """Get metrics specific to dynamic Kelly."""
        base_metrics = {}

        # Basic stats
        settled = [b for b in self.bets if b.outcome != BetOutcome.PENDING]
        if not settled:
            return base_metrics

        wins = [b for b in settled if b.outcome == BetOutcome.WIN]
        losses = [b for b in settled if b.outcome == BetOutcome.LOSS]

        base_metrics['total_bets'] = len(settled)
        base_metrics['wins'] = len(wins)
        base_metrics['losses'] = len(losses)
        base_metrics['win_rate'] = len(wins) / len(settled) if settled else 0

        # P&L metrics
        total_pnl = sum(b.pnl for b in settled)
        total_staked = sum(b.stake for b in settled)
        base_metrics['total_pnl'] = total_pnl
        base_metrics['total_staked'] = total_staked
        base_metrics['roi'] = total_pnl / total_staked if total_staked > 0 else 0

        # Drawdown
        base_metrics['current_drawdown'] = self.get_current_drawdown()
        base_metrics['peak_bankroll'] = self.peak_bankroll
        base_metrics['current_bankroll'] = self.bankroll

        # Calculate max drawdown from history
        peak = self.initial_bankroll
        max_dd = 0
        for _, br in self.bankroll_history:
            if br > peak:
                peak = br
            dd = (peak - br) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        base_metrics['max_drawdown'] = max_dd

        # Streak info
        base_metrics['consecutive_losses'] = self.consecutive_losses
        base_metrics['recent_win_rate'] = self.get_recent_win_rate()

        # Average stake as % of bankroll
        stakes_pct = [b.stake / self.initial_bankroll for b in settled]
        base_metrics['avg_stake_pct'] = np.mean(stakes_pct) if stakes_pct else 0
        base_metrics['max_stake_pct'] = max(stakes_pct) if stakes_pct else 0

        return base_metrics


class ModelBacktester:
    """
    High-level interface for backtesting betting models.

    Combines walk-forward validation with betting simulation.
    """

    def __init__(
        self,
        model_name: str = "default",
        initial_bankroll: float = 10000.0,
        staking_strategy: str = "kelly",
        min_edge: float = 0.02
    ):
        self.model_name = model_name
        self.initial_bankroll = initial_bankroll
        self.staking_strategy = staking_strategy
        self.min_edge = min_edge

    def backtest_moneyline(
        self,
        games: List[Dict],
        model_predict_fn: Callable,
        model_train_fn: Callable = None,
        walk_forward: bool = False
    ) -> BacktestResult:
        """
        Backtest moneyline predictions.

        Args:
            games: List of game dicts with keys:
                - game_id, date, home_team, away_team, home_odds, away_odds,
                  home_score, away_score, features
            model_predict_fn: Function that takes features and returns (home_prob, away_prob)
            model_train_fn: Optional function to train model (for walk-forward)
            walk_forward: Whether to use walk-forward validation

        Returns:
            BacktestResult object
        """
        simulator = BettingSimulator(
            initial_bankroll=self.initial_bankroll,
            staking_strategy=self.staking_strategy,
            min_edge=self.min_edge
        )

        # Sort by date
        games = sorted(games, key=lambda g: g["date"])

        for game in games:
            # Make prediction
            features = game.get("features", {})
            home_prob, away_prob = model_predict_fn(features)

            # Determine actual winner
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)
            home_won = home_score > away_score

            # Try to place home bet
            home_bet = simulator.place_bet(
                game_id=game["game_id"],
                bet_type=BetType.MONEYLINE,
                selection=game["home_team"],
                odds=game.get("home_odds", -110),
                predicted_prob=home_prob,
                placed_at=game["date"] if isinstance(game["date"], datetime) else datetime.now(),
                closing_odds=game.get("closing_home_odds")
            )

            if home_bet:
                simulator.settle_bet(
                    home_bet.bet_id,
                    won=home_won,
                    actual_result={"home_score": home_score, "away_score": away_score}
                )

            # Try to place away bet
            away_bet = simulator.place_bet(
                game_id=game["game_id"] + "_away",
                bet_type=BetType.MONEYLINE,
                selection=game["away_team"],
                odds=game.get("away_odds", -110),
                predicted_prob=away_prob,
                placed_at=game["date"] if isinstance(game["date"], datetime) else datetime.now(),
                closing_odds=game.get("closing_away_odds")
            )

            if away_bet:
                simulator.settle_bet(
                    away_bet.bet_id,
                    won=not home_won,
                    actual_result={"home_score": home_score, "away_score": away_score}
                )

        return simulator.get_results(self.model_name, BetType.MONEYLINE)

    def backtest_spread(
        self,
        games: List[Dict],
        model_predict_fn: Callable
    ) -> BacktestResult:
        """
        Backtest spread predictions.

        Args:
            games: List of game dicts with spread_line, home_odds_spread, etc.
            model_predict_fn: Function that takes features and returns (predicted_spread, cover_prob)
        """
        simulator = BettingSimulator(
            initial_bankroll=self.initial_bankroll,
            staking_strategy=self.staking_strategy,
            min_edge=self.min_edge
        )

        games = sorted(games, key=lambda g: g["date"])

        for game in games:
            features = game.get("features", {})
            predicted_spread, home_cover_prob = model_predict_fn(features)

            # Calculate actual result
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)
            spread_line = game.get("spread_line", 0)
            actual_margin = home_score - away_score
            home_covered = actual_margin > -spread_line

            # Place bet on home covering
            bet = simulator.place_bet(
                game_id=game["game_id"],
                bet_type=BetType.SPREAD,
                selection=f"{game['home_team']} {spread_line:+.1f}",
                odds=game.get("home_odds_spread", -110),
                predicted_prob=home_cover_prob,
                placed_at=game["date"] if isinstance(game["date"], datetime) else datetime.now()
            )

            if bet:
                pushed = abs(actual_margin + spread_line) < 0.5  # Push on exact spread
                simulator.settle_bet(
                    bet.bet_id,
                    won=home_covered and not pushed,
                    pushed=pushed
                )

        return simulator.get_results(self.model_name, BetType.SPREAD)

    def save_results(self, result: BacktestResult, directory: str = "backtest_results") -> str:
        """Save backtest results to file."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        filename = f"{result.model_name}_{result.bet_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = path / filename

        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")
        return str(filepath)


# Convenience functions
def quick_backtest(
    predictions: List[Dict],
    results: List[Dict],
    model_name: str = "model"
) -> BacktestResult:
    """
    Quick backtest from predictions and results.

    Args:
        predictions: List of {"game_id", "selection", "odds", "predicted_prob"}
        results: List of {"game_id", "winner"}

    Returns:
        BacktestResult
    """
    simulator = BettingSimulator(
        staking_strategy="flat",
        flat_stake=100.0
    )
    simulator.simulate_from_predictions(predictions, results)
    return simulator.get_results(model_name, BetType.MONEYLINE)


if __name__ == "__main__":
    # Demo with synthetic data
    print("=" * 60)
    print("Backtesting Framework Demo")
    print("=" * 60)

    # Generate synthetic games
    np.random.seed(42)
    n_games = 200

    games = []
    for i in range(n_games):
        # Simulate game with slight home advantage
        home_true_prob = np.clip(np.random.normal(0.55, 0.10), 0.30, 0.70)
        home_won = np.random.random() < home_true_prob

        # Simulate market odds (market is efficient but not perfect)
        market_prob = home_true_prob + np.random.normal(0, 0.03)
        market_prob = np.clip(market_prob, 0.35, 0.65)

        # Convert to American odds
        if market_prob > 0.5:
            home_odds = -100 * market_prob / (1 - market_prob)
            away_odds = 100 * (1 - market_prob) / market_prob
        else:
            home_odds = 100 * (1 - market_prob) / market_prob
            away_odds = -100 * market_prob / (1 - market_prob)

        games.append({
            "game_id": f"game_{i}",
            "date": datetime(2024, 1, 1) + timedelta(days=i // 5),
            "home_team": "HOM",
            "away_team": "AWY",
            "home_odds": home_odds,
            "away_odds": away_odds,
            "home_score": 100 + int(np.random.normal(10, 5)) if home_won else 95 + int(np.random.normal(5, 5)),
            "away_score": 95 + int(np.random.normal(5, 5)) if home_won else 100 + int(np.random.normal(10, 5)),
            "features": {"home_true_prob": home_true_prob},  # Cheating a bit for demo
        })

    # Define model that uses true probability (simulating a good model)
    def model_predict(features):
        # Add some noise to simulate model error
        true_prob = features.get("home_true_prob", 0.5)
        predicted_prob = true_prob + np.random.normal(0, 0.05)
        predicted_prob = np.clip(predicted_prob, 0.01, 0.99)
        return predicted_prob, 1 - predicted_prob

    # Run backtest
    backtester = ModelBacktester(
        model_name="demo_model",
        initial_bankroll=10000.0,
        staking_strategy="kelly",
        min_edge=0.03
    )

    result = backtester.backtest_moneyline(games, model_predict)

    # Print results
    print(result.summary())

    # Save results
    filepath = backtester.save_results(result)
    print(f"\nResults saved to: {filepath}")


# =============================================================================
# PRODUCTION UPGRADE: Walk-Forward Prop Backtester with Real Odds
# =============================================================================

class WalkForwardPropBacktester:
    """
    Production-grade walk-forward backtester for player props with periodic retraining.

    Key Features:
    1. Walk-forward validation: trains on past, predicts on future
    2. Periodic retraining: retrains model every N games to adapt to player changes
    3. Real odds: uses actual market odds for ROI calculation, not fixed -110
    4. Line-aware: evaluates at actual betting lines, not predicted values

    This is the gold standard for evaluating prop model profitability.
    """

    def __init__(
        self,
        initial_train_size: int = 500,
        retrain_frequency: int = 100,
        test_batch_size: int = 50,
        min_edge: float = 0.03,
        min_confidence: float = 0.55,
        use_expanding_window: bool = True,
    ):
        """
        Initialize walk-forward backtester.

        Args:
            initial_train_size: Minimum samples before first prediction
            retrain_frequency: Retrain model every N predictions
            test_batch_size: Number of predictions per walk-forward step
            min_edge: Minimum edge to place a bet
            min_confidence: Minimum P(Over) or P(Under) to bet
            use_expanding_window: If True, use all past data; if False, rolling window
        """
        self.initial_train_size = initial_train_size
        self.retrain_frequency = retrain_frequency
        self.test_batch_size = test_batch_size
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.use_expanding_window = use_expanding_window

        # Results storage
        self.predictions = []
        self.bets_placed = []
        self.training_history = []

    def calculate_roi_with_real_odds(
        self,
        predicted_prob: float,
        actual_hit: bool,
        american_odds: float
    ) -> Tuple[float, float]:
        """
        Calculate profit/loss using real American odds.

        Args:
            predicted_prob: Model's P(Over)
            actual_hit: Whether the over hit
            american_odds: Actual betting odds (e.g., -110, +100)

        Returns:
            Tuple of (profit_loss, roi) where stake is normalized to 1.0
        """
        # Convert American odds to decimal
        if american_odds > 0:
            decimal_odds = 1 + (american_odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(american_odds))

        # Profit/loss for $1 stake
        if actual_hit:
            profit = decimal_odds - 1  # Won: get back stake + profit
        else:
            profit = -1  # Lost: lose stake

        roi = profit * 100  # As percentage

        return profit, roi

    def should_bet(
        self,
        predicted_prob: float,
        american_odds: float,
        selection: str = "over"
    ) -> Tuple[bool, float]:
        """
        Determine if we should place a bet based on edge.

        Args:
            predicted_prob: P(Over) from model
            american_odds: Odds for the selection
            selection: "over" or "under"

        Returns:
            Tuple of (should_bet, edge)
        """
        # Get probability for the selection we're considering
        if selection == "over":
            model_prob = predicted_prob
        else:
            model_prob = 1 - predicted_prob

        # Calculate implied probability from odds
        if american_odds > 0:
            implied_prob = 100 / (american_odds + 100)
        else:
            implied_prob = abs(american_odds) / (abs(american_odds) + 100)

        # Edge is model_prob - implied_prob
        edge = model_prob - implied_prob

        # Should bet if edge > threshold and confidence is high enough
        should = (
            edge >= self.min_edge and
            model_prob >= self.min_confidence
        )

        return should, edge

    def backtest_props(
        self,
        player_data: List[Dict],
        prop_type: str = "points",
        train_model_fn: Callable = None,
        predict_fn: Callable = None,
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest on player prop data.

        Args:
            player_data: List of game dicts with:
                - game_date: Date of game
                - features: Feature dict for prediction
                - actual_value: Actual stat achieved
                - prop_line: Betting line (if available)
                - over_odds: Odds for over (default -110)
                - under_odds: Odds for under (default -110)
            prop_type: Type of prop being tested
            train_model_fn: Function(data) -> model to train model
            predict_fn: Function(model, features, line) -> P(Over)

        Returns:
            Comprehensive backtest results dict
        """
        # Sort by date
        data = sorted(player_data, key=lambda x: x.get("game_date", "1900-01-01"))

        if len(data) < self.initial_train_size + self.test_batch_size:
            raise ValueError(
                f"Not enough data. Need {self.initial_train_size + self.test_batch_size}, got {len(data)}"
            )

        results = {
            "prop_type": prop_type,
            "total_samples": len(data),
            "predictions": [],
            "bets": [],
            "training_periods": [],
            "metrics": {},
        }

        current_model = None
        predictions_since_retrain = 0
        train_end = self.initial_train_size

        print(f"\nWalk-Forward Backtest: {prop_type}")
        print(f"Total samples: {len(data)}")
        print(f"Initial training size: {self.initial_train_size}")
        print(f"Retrain every: {self.retrain_frequency} predictions")
        print("=" * 60)

        while train_end < len(data):
            # Determine training window
            if self.use_expanding_window:
                train_start = 0
            else:
                train_start = max(0, train_end - self.initial_train_size)

            train_data = data[train_start:train_end]

            # Retrain if needed
            if current_model is None or predictions_since_retrain >= self.retrain_frequency:
                if train_model_fn is not None:
                    print(f"  Training on {len(train_data)} samples (dates: {train_data[0].get('game_date')} to {train_data[-1].get('game_date')})")
                    current_model = train_model_fn(train_data)
                    predictions_since_retrain = 0

                    results["training_periods"].append({
                        "train_start": train_data[0].get("game_date"),
                        "train_end": train_data[-1].get("game_date"),
                        "train_size": len(train_data),
                    })

            # Determine test window
            test_end = min(train_end + self.test_batch_size, len(data))
            test_data = data[train_end:test_end]

            # Make predictions on test data
            for sample in test_data:
                features = sample.get("features", {})
                actual_value = sample.get("actual_value")
                prop_line = sample.get("prop_line")
                over_odds = sample.get("over_odds", -110)
                under_odds = sample.get("under_odds", -110)
                game_date = sample.get("game_date")

                # Skip if no line available
                if prop_line is None or actual_value is None:
                    continue

                # Get prediction
                if predict_fn is not None and current_model is not None:
                    predicted_over_prob = predict_fn(current_model, features, prop_line)
                else:
                    # Default: 50/50
                    predicted_over_prob = 0.5

                # Determine actual outcome
                actual_over = actual_value > prop_line

                # Record prediction
                prediction = {
                    "game_date": game_date,
                    "prop_line": prop_line,
                    "actual_value": actual_value,
                    "actual_over": actual_over,
                    "predicted_over_prob": predicted_over_prob,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                }
                results["predictions"].append(prediction)

                # Evaluate betting decisions
                for selection in ["over", "under"]:
                    odds = over_odds if selection == "over" else under_odds
                    prob = predicted_over_prob if selection == "over" else (1 - predicted_over_prob)

                    should_bet, edge = self.should_bet(
                        predicted_over_prob, odds, selection
                    )

                    if should_bet:
                        actual_hit = (selection == "over" and actual_over) or \
                                    (selection == "under" and not actual_over)

                        profit, roi = self.calculate_roi_with_real_odds(
                            prob, actual_hit, odds
                        )

                        bet = {
                            "game_date": game_date,
                            "selection": selection,
                            "prop_line": prop_line,
                            "odds": odds,
                            "predicted_prob": prob,
                            "edge": edge,
                            "actual_hit": actual_hit,
                            "profit": profit,
                            "roi": roi,
                        }
                        results["bets"].append(bet)

                predictions_since_retrain += 1

            # Move forward
            train_end = test_end

        # Calculate summary metrics
        results["metrics"] = self._calculate_metrics(results)

        print("\nBacktest Complete!")
        print(f"Total predictions: {len(results['predictions'])}")
        print(f"Total bets placed: {len(results['bets'])}")
        self._print_metrics(results["metrics"])

        return results

    def _calculate_metrics(self, results: Dict) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics."""
        predictions = results["predictions"]
        bets = results["bets"]

        if not predictions:
            return {"error": "No predictions"}

        # Prediction accuracy
        correct_over = sum(1 for p in predictions if p["predicted_over_prob"] > 0.5 and p["actual_over"])
        correct_under = sum(1 for p in predictions if p["predicted_over_prob"] <= 0.5 and not p["actual_over"])
        total_predictions = len(predictions)
        overall_accuracy = (correct_over + correct_under) / total_predictions if total_predictions > 0 else 0

        # Calibration: compare predicted probs to actual hit rates
        from collections import defaultdict
        calibration_buckets = defaultdict(lambda: {"predicted": 0, "actual": 0, "count": 0})

        for p in predictions:
            bucket = int(p["predicted_over_prob"] * 10) / 10  # Round to 0.1
            calibration_buckets[bucket]["predicted"] += p["predicted_over_prob"]
            calibration_buckets[bucket]["actual"] += 1 if p["actual_over"] else 0
            calibration_buckets[bucket]["count"] += 1

        # Betting metrics
        if bets:
            total_staked = len(bets)  # Each bet is $1
            total_profit = sum(b["profit"] for b in bets)
            wins = sum(1 for b in bets if b["actual_hit"])
            losses = len(bets) - wins

            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            win_rate = (wins / len(bets) * 100) if bets else 0

            # Average edge on bets placed
            avg_edge = np.mean([b["edge"] for b in bets])

            # Edge when winning vs losing
            winning_bets = [b for b in bets if b["actual_hit"]]
            losing_bets = [b for b in bets if not b["actual_hit"]]
            avg_edge_winners = np.mean([b["edge"] for b in winning_bets]) if winning_bets else 0
            avg_edge_losers = np.mean([b["edge"] for b in losing_bets]) if losing_bets else 0

            # CLV approximation (edge - actual)
            # Not exact without closing odds, but edge serves as proxy

            # Sharpe ratio (annualized, assuming ~1000 bets/year)
            daily_returns = []
            current_day = None
            day_profit = 0
            for b in bets:
                if b["game_date"] != current_day:
                    if current_day is not None:
                        daily_returns.append(day_profit)
                    current_day = b["game_date"]
                    day_profit = 0
                day_profit += b["profit"]
            if day_profit != 0:
                daily_returns.append(day_profit)

            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe = 0

            betting_metrics = {
                "total_bets": len(bets),
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "roi": roi,
                "avg_edge": avg_edge,
                "avg_edge_winners": avg_edge_winners,
                "avg_edge_losers": avg_edge_losers,
                "sharpe_ratio": sharpe,
            }
        else:
            betting_metrics = {
                "total_bets": 0,
                "message": "No bets placed (edge threshold not met)"
            }

        return {
            "prediction_accuracy": overall_accuracy,
            "total_predictions": total_predictions,
            "calibration": dict(calibration_buckets),
            "betting": betting_metrics,
            "training_periods": len(results["training_periods"]),
        }

    def _print_metrics(self, metrics: Dict):
        """Print formatted metrics."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print(f"\nPrediction Performance:")
        print(f"  Accuracy: {metrics['prediction_accuracy']:.1%}")
        print(f"  Total Predictions: {metrics['total_predictions']}")
        print(f"  Training Periods: {metrics['training_periods']}")

        if "betting" in metrics and metrics["betting"].get("total_bets", 0) > 0:
            b = metrics["betting"]
            print(f"\nBetting Performance:")
            print(f"  Total Bets: {b['total_bets']}")
            print(f"  Win Rate: {b['win_rate']:.1f}%")
            print(f"  ROI: {b['roi']:+.2f}%")
            print(f"  Total Profit: ${b['total_profit']:+.2f} (per $1 bets)")
            print(f"  Avg Edge: {b['avg_edge']:.1%}")
            print(f"  Sharpe Ratio: {b['sharpe_ratio']:.2f}")

            # Interpretation
            print("\nInterpretation:")
            if b["roi"] > 5:
                print("  ✓ Strong positive ROI - model has edge")
            elif b["roi"] > 0:
                print("  ~ Marginal positive ROI - possible edge")
            else:
                print("  ✗ Negative ROI - no profitable edge")

            if b["win_rate"] > 52.4:  # Breakeven at -110
                print("  ✓ Win rate above breakeven")
            else:
                print("  ✗ Win rate below breakeven")

        print("=" * 60)


def calculate_real_odds_roi(
    predictions: List[Dict],
    default_odds: float = -110
) -> Dict[str, float]:
    """
    Calculate ROI using real market odds from predictions.

    Each prediction dict should have:
    - predicted_prob: Model probability
    - actual_hit: Whether bet won
    - odds: American odds (optional, defaults to -110)

    Returns:
        Dict with ROI metrics
    """
    total_staked = 0
    total_returned = 0
    wins = 0
    losses = 0

    for pred in predictions:
        odds = pred.get("odds", default_odds)
        actual_hit = pred.get("actual_hit", False)

        # Convert to decimal for payout calculation
        if odds > 0:
            decimal_odds = 1 + (odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(odds))

        stake = 1.0  # Normalize to $1 bets
        total_staked += stake

        if actual_hit:
            total_returned += stake * decimal_odds
            wins += 1
        else:
            losses += 1

    profit = total_returned - total_staked
    roi = (profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    return {
        "total_staked": total_staked,
        "total_returned": total_returned,
        "profit": profit,
        "roi": roi,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
    }
