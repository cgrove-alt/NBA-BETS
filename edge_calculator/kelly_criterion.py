"""
Kelly Criterion - Optimal Bet Sizing

The Kelly Criterion determines optimal bet size to maximize long-term growth.

Full Kelly: f = (bp - q) / b
where:
  b = decimal odds - 1 (profit per unit staked)
  p = probability of winning
  q = probability of losing (1 - p)

We use FRACTIONAL Kelly (25-50%) to reduce variance.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BetSize:
    """Result of Kelly criterion calculation."""

    # Input values
    win_probability: float
    decimal_odds: float
    bankroll: float
    kelly_fraction: float

    # Calculated values
    full_kelly: float = 0.0  # Full Kelly fraction of bankroll
    fractional_kelly: float = 0.0  # Adjusted Kelly fraction
    bet_amount: float = 0.0  # Actual bet amount
    bet_units: float = 0.0  # Bet in units (1 unit = 1% of bankroll)

    # Constraints applied
    min_bet_applied: bool = False
    max_bet_applied: bool = False
    should_bet: bool = True

    # Expected outcomes
    expected_growth: float = 0.0  # Expected bankroll growth rate
    risk_of_ruin: float = 0.0  # Approximate risk of ruin

    def to_dict(self) -> dict:
        return {
            'win_probability': round(self.win_probability, 4),
            'decimal_odds': round(self.decimal_odds, 3),
            'bankroll': round(self.bankroll, 2),
            'kelly_fraction': self.kelly_fraction,
            'full_kelly': round(self.full_kelly, 4),
            'fractional_kelly': round(self.fractional_kelly, 4),
            'bet_amount': round(self.bet_amount, 2),
            'bet_units': round(self.bet_units, 2),
            'should_bet': self.should_bet,
            'min_bet_applied': self.min_bet_applied,
            'max_bet_applied': self.max_bet_applied,
            'expected_growth': round(self.expected_growth, 6),
        }


class KellyCriterion:
    """
    Kelly Criterion bet sizing calculator.

    Uses fractional Kelly to reduce variance while maintaining
    positive expected growth.
    """

    # Default settings
    DEFAULT_KELLY_FRACTION = 0.35  # 35% of full Kelly
    DEFAULT_MAX_BET_FRACTION = 0.05  # 5% of bankroll max
    DEFAULT_MIN_BET_FRACTION = 0.005  # 0.5% of bankroll min

    # Confidence tier settings
    CONFIDENCE_TIERS = {
        'strong': {
            'min_edge': 0.05,
            'kelly_fraction': 0.50,
            'max_units': 3.0,
        },
        'moderate': {
            'min_edge': 0.03,
            'kelly_fraction': 0.35,
            'max_units': 2.0,
        },
        'marginal': {
            'min_edge': 0.02,
            'kelly_fraction': 0.25,
            'max_units': 1.0,
        },
        'pass': {
            'min_edge': 0.0,
            'kelly_fraction': 0.0,
            'max_units': 0.0,
        },
    }

    def __init__(
        self,
        kelly_fraction: float = None,
        max_bet_fraction: float = None,
        min_bet_fraction: float = None,
        unit_size: float = 0.01,  # 1 unit = 1% of bankroll
    ):
        """
        Initialize Kelly criterion calculator.

        Args:
            kelly_fraction: Fraction of full Kelly to use (default 0.35)
            max_bet_fraction: Maximum bet as fraction of bankroll (default 0.05)
            min_bet_fraction: Minimum bet as fraction of bankroll (default 0.005)
            unit_size: Size of one unit as fraction of bankroll (default 0.01)
        """
        self.kelly_fraction = kelly_fraction or self.DEFAULT_KELLY_FRACTION
        self.max_bet_fraction = max_bet_fraction or self.DEFAULT_MAX_BET_FRACTION
        self.min_bet_fraction = min_bet_fraction or self.DEFAULT_MIN_BET_FRACTION
        self.unit_size = unit_size

    def get_tier_settings(self, edge: float) -> dict:
        """
        Get confidence tier settings based on edge.

        Args:
            edge: Edge as decimal (e.g., 0.05 for 5%)

        Returns:
            Tier settings dict
        """
        if edge >= self.CONFIDENCE_TIERS['strong']['min_edge']:
            return self.CONFIDENCE_TIERS['strong']
        elif edge >= self.CONFIDENCE_TIERS['moderate']['min_edge']:
            return self.CONFIDENCE_TIERS['moderate']
        elif edge >= self.CONFIDENCE_TIERS['marginal']['min_edge']:
            return self.CONFIDENCE_TIERS['marginal']
        else:
            return self.CONFIDENCE_TIERS['pass']

    def calculate_full_kelly(
        self,
        win_probability: float,
        decimal_odds: float,
    ) -> float:
        """
        Calculate full Kelly fraction.

        f = (bp - q) / b

        where:
          b = decimal_odds - 1 (profit per unit)
          p = win_probability
          q = 1 - p

        Args:
            win_probability: Probability of winning (0-1)
            decimal_odds: Decimal odds (e.g., 1.909 for -110)

        Returns:
            Full Kelly fraction (can be negative if no edge)
        """
        b = decimal_odds - 1  # Profit per unit staked
        p = win_probability
        q = 1 - p

        if b <= 0:
            return 0.0

        return (b * p - q) / b


    def calculate(
        self,
        win_probability: float,
        decimal_odds: float,
        bankroll: float,
        edge: float = None,
        kelly_fraction_override: float = None,
    ) -> BetSize:
        """
        Calculate optimal bet size using fractional Kelly.

        Args:
            win_probability: Model's win probability (0-1)
            decimal_odds: Decimal odds
            bankroll: Current bankroll
            edge: Edge for tier determination (optional)
            kelly_fraction_override: Override the Kelly fraction

        Returns:
            BetSize with recommended bet amount
        """
        # Calculate full Kelly
        full_kelly = self.calculate_full_kelly(win_probability, decimal_odds)

        # Determine Kelly fraction based on edge tier
        if kelly_fraction_override is not None:
            kelly_frac = kelly_fraction_override
        elif edge is not None:
            tier = self.get_tier_settings(edge)
            kelly_frac = tier['kelly_fraction']
        else:
            kelly_frac = self.kelly_fraction

        # Apply fractional Kelly
        fractional_kelly = full_kelly * kelly_frac

        # Initialize result
        result = BetSize(
            win_probability=win_probability,
            decimal_odds=decimal_odds,
            bankroll=bankroll,
            kelly_fraction=kelly_frac,
            full_kelly=full_kelly,
            fractional_kelly=fractional_kelly,
        )

        # No bet if Kelly is negative or zero
        if fractional_kelly <= 0:
            result.should_bet = False
            result.bet_amount = 0
            result.bet_units = 0
            return result

        # Calculate bet amount
        bet_amount = bankroll * fractional_kelly

        # Apply maximum bet constraint
        max_bet = bankroll * self.max_bet_fraction
        if edge is not None:
            tier = self.get_tier_settings(edge)
            max_units = tier['max_units']
            tier_max = bankroll * self.unit_size * max_units
            max_bet = min(max_bet, tier_max)

        if bet_amount > max_bet:
            bet_amount = max_bet
            result.max_bet_applied = True

        # Apply minimum bet constraint
        min_bet = bankroll * self.min_bet_fraction
        if bet_amount < min_bet:
            # Check if we should bet at all
            if bet_amount < min_bet * 0.5:
                # Too small, don't bet
                result.should_bet = False
                result.bet_amount = 0
                result.bet_units = 0
                return result
            else:
                # Round up to minimum
                bet_amount = min_bet
                result.min_bet_applied = True

        # Calculate units
        unit_value = bankroll * self.unit_size
        bet_units = bet_amount / unit_value if unit_value > 0 else 0

        # Calculate expected growth rate (Kelly growth formula)
        # g = p * log(1 + f*b) + q * log(1 - f)
        # Simplified approximation for small f
        expected_growth = win_probability * (decimal_odds - 1) * fractional_kelly - \
                         (1 - win_probability) * fractional_kelly

        result.bet_amount = bet_amount
        result.bet_units = bet_units
        result.expected_growth = expected_growth
        result.should_bet = True

        return result

    def calculate_from_american(
        self,
        win_probability: float,
        american_odds: int,
        bankroll: float,
        edge: float = None,
    ) -> BetSize:
        """
        Calculate bet size from American odds.

        Args:
            win_probability: Model's win probability (0-1)
            american_odds: American odds (e.g., -110, +150)
            bankroll: Current bankroll
            edge: Edge for tier determination

        Returns:
            BetSize
        """
        # Convert American to decimal
        if american_odds < 0:
            decimal_odds = 1 + (100 / abs(american_odds))
        else:
            decimal_odds = 1 + (american_odds / 100)

        return self.calculate(win_probability, decimal_odds, bankroll, edge)

    def optimal_growth_rate(
        self,
        win_probability: float,
        decimal_odds: float,
    ) -> float:
        """
        Calculate optimal growth rate under full Kelly.

        This is the maximum achievable long-term growth rate.

        Args:
            win_probability: Win probability
            decimal_odds: Decimal odds

        Returns:
            Optimal growth rate per bet
        """
        import math

        p = win_probability
        q = 1 - p
        b = decimal_odds - 1

        if b <= 0 or p <= 0 or q <= 0:
            return 0.0

        # Full Kelly fraction
        f = (b * p - q) / b

        if f <= 0:
            return 0.0

        # Growth rate: g = p * log(1 + f*b) + q * log(1 - f)
        try:
            growth = p * math.log(1 + f * b) + q * math.log(1 - f)
        except ValueError:
            growth = 0.0

        return growth

    def simulate_kelly(
        self,
        win_probability: float,
        decimal_odds: float,
        bankroll: float,
        kelly_fraction: float,
        num_bets: int = 100,
        num_simulations: int = 1000,
    ) -> dict:
        """
        Simulate Kelly betting to estimate outcomes.

        Args:
            win_probability: Win probability
            decimal_odds: Decimal odds
            bankroll: Starting bankroll
            kelly_fraction: Kelly fraction to use
            num_bets: Number of bets per simulation
            num_simulations: Number of simulations

        Returns:
            Dict with simulation statistics
        """
        import random
        import numpy as np

        final_bankrolls = []

        for _ in range(num_simulations):
            current_bankroll = bankroll

            for _ in range(num_bets):
                if current_bankroll <= 0:
                    break

                # Calculate bet size
                full_kelly = self.calculate_full_kelly(win_probability, decimal_odds)
                bet_fraction = full_kelly * kelly_fraction
                bet_fraction = max(0, min(bet_fraction, self.max_bet_fraction))
                bet_amount = current_bankroll * bet_fraction

                # Simulate outcome
                if random.random() < win_probability:
                    # Win
                    profit = bet_amount * (decimal_odds - 1)
                    current_bankroll += profit
                else:
                    # Lose
                    current_bankroll -= bet_amount

            final_bankrolls.append(current_bankroll)

        # Calculate statistics
        final_array = np.array(final_bankrolls)

        return {
            'starting_bankroll': bankroll,
            'mean_final': np.mean(final_array),
            'median_final': np.median(final_array),
            'std_final': np.std(final_array),
            'min_final': np.min(final_array),
            'max_final': np.max(final_array),
            'bust_rate': np.mean(final_array <= 0),
            'double_rate': np.mean(final_array >= 2 * bankroll),
            'kelly_fraction': kelly_fraction,
            'num_bets': num_bets,
            'num_simulations': num_simulations,
        }


# Convenience function
def calculate_bet_size(
    win_probability: float,
    american_odds: int,
    bankroll: float,
    edge: float = None,
) -> BetSize:
    """
    Quick bet size calculation.

    Args:
        win_probability: Win probability (0-1)
        american_odds: American odds
        bankroll: Current bankroll
        edge: Edge for tier determination

    Returns:
        BetSize
    """
    kelly = KellyCriterion()
    return kelly.calculate_from_american(win_probability, american_odds, bankroll, edge)


if __name__ == "__main__":
    # Test Kelly criterion
    kelly = KellyCriterion()

    print("=" * 60)
    print("KELLY CRITERION TEST")
    print("=" * 60)

    # Test scenarios
    print("\nBet Size Calculations:")

    test_cases = [
        (0.55, -110, 1000, 0.03),  # 55% at -110, moderate edge
        (0.58, -110, 1000, 0.05),  # 58% at -110, strong edge
        (0.52, -110, 1000, 0.02),  # 52% at -110, marginal edge
        (0.60, +100, 1000, 0.10),  # 60% at +100, very strong
        (0.50, -110, 1000, 0.00),  # 50% at -110, no edge
    ]

    for prob, odds, bankroll, edge in test_cases:
        result = kelly.calculate_from_american(prob, odds, bankroll, edge)
        print(f"\n  Win Prob: {prob:.0%}, Odds: {odds:+d}, Edge: {edge:.0%}")
        print(f"    Full Kelly: {result.full_kelly:.2%}")
        print(f"    Fractional Kelly ({result.kelly_fraction:.0%}): {result.fractional_kelly:.2%}")
        print(f"    Bet Amount: ${result.bet_amount:.2f}")
        print(f"    Bet Units: {result.bet_units:.1f}u")
        print(f"    Should Bet: {result.should_bet}")

    # Test tier settings
    print("\n\nConfidence Tiers:")
    for edge in [0.02, 0.03, 0.05, 0.08]:
        tier = kelly.get_tier_settings(edge)
        print(f"  Edge {edge:.0%}: Kelly {tier['kelly_fraction']:.0%}, Max {tier['max_units']}u")

    # Simulate to show variance reduction
    print("\n\nSimulation (1000 trials, 100 bets each):")
    for fraction in [1.0, 0.5, 0.35, 0.25]:
        sim = kelly.simulate_kelly(
            win_probability=0.55,
            decimal_odds=1.909,  # -110
            bankroll=1000,
            kelly_fraction=fraction,
            num_bets=100,
            num_simulations=1000,
        )
        print(f"\n  Kelly Fraction: {fraction:.0%}")
        print(f"    Mean Final: ${sim['mean_final']:.0f}")
        print(f"    Median Final: ${sim['median_final']:.0f}")
        print(f"    Bust Rate: {sim['bust_rate']:.1%}")
        print(f"    Double Rate: {sim['double_rate']:.1%}")
