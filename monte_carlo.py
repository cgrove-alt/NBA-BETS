"""
Monte Carlo Simulation for NBA Betting Model

Provides statistical confidence intervals for performance metrics through simulation:
- ROI confidence intervals (95%, 99%)
- Probability of ruin calculation
- Maximum drawdown percentiles
- Time to double bankroll estimates
- Expected bankroll distribution

This helps quantify uncertainty in backtesting results and set realistic expectations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings


@dataclass
class BetSimulation:
    """Represents a single bet in simulation."""
    probability: float  # Model's estimated win probability
    odds: float         # Decimal odds
    edge: float         # Expected edge (prob - implied_prob)
    stake_pct: float    # Stake as percentage of bankroll


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    # ROI metrics
    roi_mean: float
    roi_median: float
    roi_std: float
    roi_95_ci: Tuple[float, float]
    roi_99_ci: Tuple[float, float]

    # Bankroll metrics
    final_bankroll_mean: float
    final_bankroll_median: float
    final_bankroll_95_ci: Tuple[float, float]

    # Drawdown metrics
    max_drawdown_mean: float
    max_drawdown_median: float
    max_drawdown_95_percentile: float
    max_drawdown_99_percentile: float

    # Risk metrics
    probability_of_ruin: float  # P(bankroll < ruin_threshold)
    probability_of_profit: float  # P(final_bankroll > initial)
    probability_of_doubling: float  # P(final_bankroll > 2 * initial)

    # Time metrics
    expected_bets_to_double: Optional[float]
    expected_bets_to_ruin: Optional[float]

    # Distribution data for visualization
    roi_distribution: np.ndarray
    final_bankroll_distribution: np.ndarray
    max_drawdown_distribution: np.ndarray

    # Simulation parameters
    n_simulations: int
    n_bets_per_simulation: int


class MonteCarloSimulator:
    """
    Monte Carlo simulation for betting performance analysis.

    Simulates thousands of possible outcomes to estimate:
    - Expected returns and variance
    - Probability of various outcomes (profit, ruin, doubling)
    - Confidence intervals for key metrics
    """

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        ruin_threshold: float = 0.1,  # Ruin if bankroll falls to 10% of initial
        n_simulations: int = 10000,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize simulator.

        Args:
            initial_bankroll: Starting bankroll
            ruin_threshold: Fraction of bankroll below which is considered ruin
            n_simulations: Number of simulation runs
            random_seed: Random seed for reproducibility
        """
        self.initial_bankroll = initial_bankroll
        self.ruin_threshold = ruin_threshold
        self.n_simulations = n_simulations

        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_single_path(
        self,
        bets: List[BetSimulation],
        track_drawdown: bool = True,
    ) -> Dict:
        """
        Simulate a single betting path.

        Args:
            bets: List of bets to simulate
            track_drawdown: Whether to track drawdown during simulation

        Returns:
            Dict with final bankroll, max drawdown, etc.
        """
        bankroll = self.initial_bankroll
        peak_bankroll = self.initial_bankroll
        max_drawdown = 0.0
        ruin_bet = None
        double_bet = None

        for i, bet in enumerate(bets):
            # Check if ruined
            if bankroll < self.initial_bankroll * self.ruin_threshold:
                if ruin_bet is None:
                    ruin_bet = i
                break

            # Simulate bet outcome
            won = np.random.random() < bet.probability

            stake = bankroll * bet.stake_pct
            if won:
                profit = stake * (bet.odds - 1)
            else:
                profit = -stake

            bankroll += profit

            # Track peak and drawdown
            if track_drawdown:
                if bankroll > peak_bankroll:
                    peak_bankroll = bankroll
                    # Check if doubled
                    if peak_bankroll >= 2 * self.initial_bankroll and double_bet is None:
                        double_bet = i

                current_dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                max_drawdown = max(max_drawdown, current_dd)

        # Calculate ROI
        total_staked = sum(self.initial_bankroll * b.stake_pct for b in bets[:ruin_bet or len(bets)])
        pnl = bankroll - self.initial_bankroll
        roi = pnl / total_staked if total_staked > 0 else 0

        return {
            'final_bankroll': bankroll,
            'max_drawdown': max_drawdown,
            'roi': roi,
            'pnl': pnl,
            'ruined': ruin_bet is not None,
            'doubled': double_bet is not None,
            'bets_to_ruin': ruin_bet,
            'bets_to_double': double_bet,
        }

    def run_simulation(
        self,
        bets: List[BetSimulation],
        n_bets_per_sim: Optional[int] = None,
        resample_bets: bool = True,
    ) -> MonteCarloResult:
        """
        Run full Monte Carlo simulation.

        Args:
            bets: List of bets (will be resampled for each simulation if resample_bets=True)
            n_bets_per_sim: Number of bets per simulation (defaults to len(bets))
            resample_bets: Whether to resample bets with replacement

        Returns:
            MonteCarloResult with comprehensive statistics
        """
        n_bets = n_bets_per_sim or len(bets)

        # Storage for results
        final_bankrolls = []
        max_drawdowns = []
        rois = []
        ruined_count = 0
        profit_count = 0
        doubled_count = 0
        bets_to_ruin = []
        bets_to_double = []

        for _ in range(self.n_simulations):
            # Resample bets if requested
            if resample_bets:
                sim_bets = [bets[i] for i in np.random.choice(len(bets), size=n_bets, replace=True)]
            else:
                sim_bets = bets[:n_bets]

            # Run single simulation
            result = self.simulate_single_path(sim_bets)

            final_bankrolls.append(result['final_bankroll'])
            max_drawdowns.append(result['max_drawdown'])
            rois.append(result['roi'])

            if result['ruined']:
                ruined_count += 1
                if result['bets_to_ruin'] is not None:
                    bets_to_ruin.append(result['bets_to_ruin'])
            if result['final_bankroll'] > self.initial_bankroll:
                profit_count += 1
            if result['doubled']:
                doubled_count += 1
                if result['bets_to_double'] is not None:
                    bets_to_double.append(result['bets_to_double'])

        # Convert to arrays
        final_bankrolls = np.array(final_bankrolls)
        max_drawdowns = np.array(max_drawdowns)
        rois = np.array(rois)

        # Calculate statistics
        return MonteCarloResult(
            # ROI metrics
            roi_mean=np.mean(rois),
            roi_median=np.median(rois),
            roi_std=np.std(rois),
            roi_95_ci=(np.percentile(rois, 2.5), np.percentile(rois, 97.5)),
            roi_99_ci=(np.percentile(rois, 0.5), np.percentile(rois, 99.5)),

            # Bankroll metrics
            final_bankroll_mean=np.mean(final_bankrolls),
            final_bankroll_median=np.median(final_bankrolls),
            final_bankroll_95_ci=(np.percentile(final_bankrolls, 2.5), np.percentile(final_bankrolls, 97.5)),

            # Drawdown metrics
            max_drawdown_mean=np.mean(max_drawdowns),
            max_drawdown_median=np.median(max_drawdowns),
            max_drawdown_95_percentile=np.percentile(max_drawdowns, 95),
            max_drawdown_99_percentile=np.percentile(max_drawdowns, 99),

            # Risk metrics
            probability_of_ruin=ruined_count / self.n_simulations,
            probability_of_profit=profit_count / self.n_simulations,
            probability_of_doubling=doubled_count / self.n_simulations,

            # Time metrics
            expected_bets_to_double=np.mean(bets_to_double) if bets_to_double else None,
            expected_bets_to_ruin=np.mean(bets_to_ruin) if bets_to_ruin else None,

            # Distributions
            roi_distribution=rois,
            final_bankroll_distribution=final_bankrolls,
            max_drawdown_distribution=max_drawdowns,

            # Parameters
            n_simulations=self.n_simulations,
            n_bets_per_simulation=n_bets,
        )

    def simulate_from_parameters(
        self,
        win_probability: float,
        odds: float,
        stake_pct: float,
        n_bets: int,
    ) -> MonteCarloResult:
        """
        Run simulation with uniform bet parameters.

        Useful for theoretical analysis without historical data.

        Args:
            win_probability: Model's win probability
            odds: Decimal odds for all bets
            stake_pct: Stake as percentage of bankroll
            n_bets: Number of bets per simulation

        Returns:
            MonteCarloResult
        """
        implied_prob = 1 / odds
        edge = win_probability - implied_prob

        # Create uniform bets
        bets = [
            BetSimulation(
                probability=win_probability,
                odds=odds,
                edge=edge,
                stake_pct=stake_pct,
            )
            for _ in range(n_bets)
        ]

        return self.run_simulation(bets, resample_bets=False)

    def simulate_from_backtest(
        self,
        backtest_results: List[Dict],
        n_bets_per_sim: Optional[int] = None,
    ) -> MonteCarloResult:
        """
        Run simulation based on historical backtest results.

        Args:
            backtest_results: List of dicts with keys:
                - probability: Model's win probability
                - odds: Decimal odds
                - stake_pct: Stake as percentage of bankroll
            n_bets_per_sim: Number of bets per simulation

        Returns:
            MonteCarloResult
        """
        bets = []
        for result in backtest_results:
            prob = result.get('probability', result.get('predicted_probability', 0.5))
            odds = result.get('decimal_odds', result.get('odds', 2.0))
            stake = result.get('stake_pct', result.get('stake', 100) / self.initial_bankroll)

            if odds < 1:
                # Convert American to decimal if needed
                if odds >= 100:
                    odds = 1 + (odds / 100)
                else:
                    odds = 1 + (100 / abs(odds))

            implied_prob = 1 / odds if odds > 0 else 0.5
            edge = prob - implied_prob

            bets.append(BetSimulation(
                probability=prob,
                odds=odds,
                edge=edge,
                stake_pct=stake,
            ))

        return self.run_simulation(bets, n_bets_per_sim)

    def kelly_sensitivity_analysis(
        self,
        win_probability: float,
        odds: float,
        kelly_fractions: List[float] = None,
        n_bets: int = 500,
    ) -> Dict[str, MonteCarloResult]:
        """
        Analyze sensitivity of results to Kelly fraction.

        Args:
            win_probability: Model's win probability
            odds: Decimal odds
            kelly_fractions: List of Kelly fractions to test
            n_bets: Number of bets per simulation

        Returns:
            Dict mapping Kelly fraction to MonteCarloResult
        """
        if kelly_fractions is None:
            kelly_fractions = [0.10, 0.25, 0.50, 0.75, 1.0]

        results = {}

        # Calculate full Kelly
        b = odds - 1
        p = win_probability
        q = 1 - p
        full_kelly = (b * p - q) / b if b > 0 else 0

        for fraction in kelly_fractions:
            stake_pct = full_kelly * fraction
            stake_pct = max(0.001, min(0.10, stake_pct))  # Bound stake

            result = self.simulate_from_parameters(
                win_probability=win_probability,
                odds=odds,
                stake_pct=stake_pct,
                n_bets=n_bets,
            )
            results[f"kelly_{fraction}"] = result

        return results

    def edge_sensitivity_analysis(
        self,
        base_probability: float,
        odds: float,
        stake_pct: float,
        edge_adjustments: List[float] = None,
        n_bets: int = 500,
    ) -> Dict[str, MonteCarloResult]:
        """
        Analyze sensitivity of results to edge estimation error.

        Important for understanding impact of miscalibration.

        Args:
            base_probability: Model's estimated probability
            odds: Decimal odds
            stake_pct: Stake as percentage
            edge_adjustments: List of adjustments to true probability
            n_bets: Number of bets per simulation

        Returns:
            Dict mapping edge adjustment to MonteCarloResult
        """
        if edge_adjustments is None:
            edge_adjustments = [-0.05, -0.02, 0, 0.02, 0.05]

        results = {}

        for adj in edge_adjustments:
            true_prob = base_probability + adj
            true_prob = max(0.01, min(0.99, true_prob))

            # Create bets with model probability but true probability for outcomes
            implied_prob = 1 / odds
            edge = base_probability - implied_prob

            bets = [
                BetSimulation(
                    probability=true_prob,  # True probability for simulation
                    odds=odds,
                    edge=edge,  # Edge based on model probability
                    stake_pct=stake_pct,
                )
                for _ in range(n_bets)
            ]

            result = self.run_simulation(bets, resample_bets=False)
            adj_label = f"edge_{adj:+.0%}" if adj != 0 else "edge_baseline"
            results[adj_label] = result

        return results


def calculate_var_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall).

    Args:
        returns: Array of returns/PnL
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        (VaR, CVaR) tuple
    """
    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean() if np.any(returns <= var) else var
    return var, cvar


def print_monte_carlo_report(result: MonteCarloResult) -> str:
    """Generate a formatted report from Monte Carlo results."""
    lines = [
        "=" * 60,
        "MONTE CARLO SIMULATION RESULTS",
        "=" * 60,
        "",
        f"Simulations: {result.n_simulations:,}",
        f"Bets per simulation: {result.n_bets_per_simulation:,}",
        "",
        "--- ROI Analysis ---",
        f"Mean ROI: {result.roi_mean:.2%}",
        f"Median ROI: {result.roi_median:.2%}",
        f"Std Dev: {result.roi_std:.2%}",
        f"95% CI: [{result.roi_95_ci[0]:.2%}, {result.roi_95_ci[1]:.2%}]",
        f"99% CI: [{result.roi_99_ci[0]:.2%}, {result.roi_99_ci[1]:.2%}]",
        "",
        "--- Bankroll Analysis ---",
        f"Mean Final Bankroll: ${result.final_bankroll_mean:,.2f}",
        f"Median Final Bankroll: ${result.final_bankroll_median:,.2f}",
        f"95% CI: [${result.final_bankroll_95_ci[0]:,.2f}, ${result.final_bankroll_95_ci[1]:,.2f}]",
        "",
        "--- Drawdown Analysis ---",
        f"Mean Max Drawdown: {result.max_drawdown_mean:.1%}",
        f"Median Max Drawdown: {result.max_drawdown_median:.1%}",
        f"95th Percentile: {result.max_drawdown_95_percentile:.1%}",
        f"99th Percentile: {result.max_drawdown_99_percentile:.1%}",
        "",
        "--- Risk Analysis ---",
        f"Probability of Profit: {result.probability_of_profit:.1%}",
        f"Probability of Doubling: {result.probability_of_doubling:.1%}",
        f"Probability of Ruin: {result.probability_of_ruin:.1%}",
        "",
        "--- Time Analysis ---",
    ]

    if result.expected_bets_to_double is not None:
        lines.append(f"Expected Bets to Double: {result.expected_bets_to_double:.0f}")
    else:
        lines.append("Expected Bets to Double: N/A (never doubled in simulations)")

    if result.expected_bets_to_ruin is not None:
        lines.append(f"Expected Bets to Ruin: {result.expected_bets_to_ruin:.0f}")
    else:
        lines.append("Expected Bets to Ruin: N/A (never ruined in simulations)")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Create simulator
    simulator = MonteCarloSimulator(
        initial_bankroll=10000,
        n_simulations=10000,
        random_seed=42
    )

    # Simulate with typical betting parameters
    # Assume 55% win rate at -110 odds with quarter Kelly
    win_prob = 0.55
    decimal_odds = 1.909  # -110 in decimal
    implied_prob = 1 / decimal_odds  # 0.524

    # Calculate Kelly stake
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p
    full_kelly = (b * p - q) / b
    stake_pct = full_kelly * 0.25  # Quarter Kelly

    print(f"Win Probability: {win_prob:.1%}")
    print(f"Implied Probability: {implied_prob:.1%}")
    print(f"Edge: {(win_prob - implied_prob):.1%}")
    print(f"Full Kelly: {full_kelly:.2%}")
    print(f"Quarter Kelly Stake: {stake_pct:.2%}")
    print()

    # Run simulation
    result = simulator.simulate_from_parameters(
        win_probability=win_prob,
        odds=decimal_odds,
        stake_pct=stake_pct,
        n_bets=500,
    )

    print(print_monte_carlo_report(result))

    # Kelly sensitivity analysis
    print("\n" + "=" * 60)
    print("KELLY FRACTION SENSITIVITY ANALYSIS")
    print("=" * 60)

    kelly_results = simulator.kelly_sensitivity_analysis(
        win_probability=win_prob,
        odds=decimal_odds,
        n_bets=500,
    )

    print(f"\n{'Kelly Fraction':<15} {'Mean ROI':<12} {'Max DD 95%':<12} {'P(Ruin)':<10}")
    print("-" * 50)
    for name, res in kelly_results.items():
        fraction = float(name.split("_")[1])
        print(f"{fraction:<15.2f} {res.roi_mean:<12.1%} {res.max_drawdown_95_percentile:<12.1%} {res.probability_of_ruin:<10.1%}")

    # Edge sensitivity analysis
    print("\n" + "=" * 60)
    print("EDGE SENSITIVITY ANALYSIS")
    print("=" * 60)
    print("(What if true win rate differs from model estimate?)")

    edge_results = simulator.edge_sensitivity_analysis(
        base_probability=win_prob,
        odds=decimal_odds,
        stake_pct=stake_pct,
        n_bets=500,
    )

    print(f"\n{'True Edge Adj':<15} {'Mean ROI':<12} {'P(Profit)':<12} {'P(Ruin)':<10}")
    print("-" * 50)
    for name, res in sorted(edge_results.items()):
        print(f"{name:<15} {res.roi_mean:<12.1%} {res.probability_of_profit:<12.1%} {res.probability_of_ruin:<10.1%}")
