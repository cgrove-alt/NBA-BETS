"""
Edge Calculator - Core Edge and Expected Value Calculations

Calculates:
- Edge: model_probability - implied_probability
- Expected Value (EV): edge * potential_profit
- True probability accounting for vig
"""

import logging
from dataclasses import dataclass
from typing import Optional

from scipy.stats import norm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EdgeResult:
    """Result of edge calculation."""

    # Input values
    model_probability: float  # Model's predicted probability (0-1)
    american_odds: int  # American odds (e.g., -110, +150)
    stake: float = 1.0  # Assumed stake for EV calculation

    # Calculated values
    implied_probability: float = 0.0
    no_vig_probability: float = 0.0
    edge: float = 0.0
    edge_percentage: float = 0.0
    expected_value: float = 0.0
    ev_per_dollar: float = 0.0
    decimal_odds: float = 0.0
    potential_profit: float = 0.0

    # Classification
    has_edge: bool = False
    edge_quality: str = "none"  # none, marginal, moderate, strong

    def to_dict(self) -> dict:
        return {
            'model_probability': round(self.model_probability, 4),
            'american_odds': self.american_odds,
            'implied_probability': round(self.implied_probability, 4),
            'no_vig_probability': round(self.no_vig_probability, 4),
            'edge': round(self.edge, 4),
            'edge_percentage': round(self.edge_percentage, 2),
            'expected_value': round(self.expected_value, 4),
            'ev_per_dollar': round(self.ev_per_dollar, 4),
            'decimal_odds': round(self.decimal_odds, 3),
            'potential_profit': round(self.potential_profit, 2),
            'has_edge': self.has_edge,
            'edge_quality': self.edge_quality,
        }


class EdgeCalculator:
    """
    Calculate betting edge and expected value.

    Edge = Model Probability - Implied Probability
    EV = Edge * Potential Profit - (1 - Model Probability) * Stake

    Standard sportsbook vig is ~4.5% (both sides at -110)
    """

    # Edge quality thresholds
    EDGE_THRESHOLDS = {
        'strong': 0.05,      # 5%+ edge
        'moderate': 0.03,    # 3-5% edge
        'marginal': 0.02,    # 2-3% edge
        'none': 0.0,         # <2% edge
    }

    # Standard odds for vig calculation
    STANDARD_VIG_ODDS = -110  # Most prop bets

    def __init__(self, min_edge_threshold: float = 0.03):
        """
        Initialize edge calculator.

        Args:
            min_edge_threshold: Minimum edge to consider (default 3%)
        """
        self.min_edge_threshold = min_edge_threshold

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """
        Convert American odds to decimal odds.

        -110 → 1.909
        +150 → 2.500

        Args:
            american_odds: American odds (e.g., -110, +150)

        Returns:
            Decimal odds
        """
        if american_odds < 0:
            # Favorite: divide 100 by absolute value, add 1
            return 1 + (100 / abs(american_odds))
        else:
            # Underdog: divide by 100, add 1
            return 1 + (american_odds / 100)

    @staticmethod
    def american_to_implied_probability(american_odds: int) -> float:
        """
        Convert American odds to implied probability.

        Includes vig, so probabilities won't sum to 100%.

        -110 → 52.38%
        +150 → 40.00%

        Args:
            american_odds: American odds

        Returns:
            Implied probability (0-1)
        """
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)

    @staticmethod
    def calculate_no_vig_probability(
        american_odds: int,
        opposite_odds: int = None
    ) -> float:
        """
        Calculate true probability removing the vig.

        If opposite_odds not provided, assumes standard -110 on other side.

        Args:
            american_odds: Odds for this side
            opposite_odds: Odds for opposite side (optional)

        Returns:
            No-vig probability (0-1)
        """
        if opposite_odds is None:
            # Assume standard -110 for other side
            opposite_odds = -110

        # Get implied probabilities for both sides
        implied_this = EdgeCalculator.american_to_implied_probability(american_odds)
        implied_other = EdgeCalculator.american_to_implied_probability(opposite_odds)

        # Total implied probability (includes vig)
        total_implied = implied_this + implied_other

        # Remove vig by normalizing
        return implied_this / total_implied


    @staticmethod
    def calculate_vig(
        odds_side_a: int,
        odds_side_b: int = None
    ) -> float:
        """
        Calculate the vig (juice) on a market.

        Standard -110/-110 = 4.55% vig

        Args:
            odds_side_a: Odds for side A
            odds_side_b: Odds for side B (defaults to -110)

        Returns:
            Vig as percentage (e.g., 0.0455 for 4.55%)
        """
        if odds_side_b is None:
            odds_side_b = -110

        implied_a = EdgeCalculator.american_to_implied_probability(odds_side_a)
        implied_b = EdgeCalculator.american_to_implied_probability(odds_side_b)

        total = implied_a + implied_b
        return total - 1.0


    def classify_edge(self, edge: float) -> str:
        """
        Classify edge quality.

        Args:
            edge: Edge as decimal (e.g., 0.05 for 5%)

        Returns:
            Edge quality string
        """
        if edge >= self.EDGE_THRESHOLDS['strong']:
            return 'strong'
        elif edge >= self.EDGE_THRESHOLDS['moderate']:
            return 'moderate'
        elif edge >= self.EDGE_THRESHOLDS['marginal']:
            return 'marginal'
        else:
            return 'none'

    def calculate_edge(
        self,
        model_probability: float,
        american_odds: int = -110,
        opposite_odds: int = None,
        stake: float = 1.0,
    ) -> EdgeResult:
        """
        Calculate edge and expected value for a bet.

        Args:
            model_probability: Model's predicted win probability (0-1)
            american_odds: American odds for this bet
            opposite_odds: American odds for opposite side (for vig calc)
            stake: Stake amount for EV calculation

        Returns:
            EdgeResult with all calculations
        """
        # Convert odds
        decimal_odds = self.american_to_decimal(american_odds)
        implied_prob = self.american_to_implied_probability(american_odds)
        no_vig_prob = self.calculate_no_vig_probability(american_odds, opposite_odds)

        # Calculate edge
        edge = model_probability - implied_prob
        model_probability - no_vig_prob

        # Calculate potential profit
        potential_profit = stake * (decimal_odds - 1)

        # Calculate expected value
        # EV = P(win) * profit - P(lose) * stake
        ev = (model_probability * potential_profit) - ((1 - model_probability) * stake)
        ev_per_dollar = ev / stake

        # Classify edge
        has_edge = edge >= self.min_edge_threshold
        edge_quality = self.classify_edge(edge)

        return EdgeResult(
            model_probability=model_probability,
            american_odds=american_odds,
            stake=stake,
            implied_probability=implied_prob,
            no_vig_probability=no_vig_prob,
            edge=edge,
            edge_percentage=edge * 100,
            expected_value=ev,
            ev_per_dollar=ev_per_dollar,
            decimal_odds=decimal_odds,
            potential_profit=potential_profit,
            has_edge=has_edge,
            edge_quality=edge_quality,
        )

    # Prop-specific standard deviations (aligned with daily_predictions.py PROP_STD_DEVS)
    PROP_STD_DEVS = {
        'points': 5.5,
        'rebounds': 7.0,
        'assists': 2.5,
        'threes': 1.8,
        'pra': 9.0,
    }
    DEFAULT_PROP_STD_DEV = 5.0

    def calculate_edge_from_prediction(
        self,
        predicted_value: float,
        prop_line: float,
        american_odds: int = -110,
        model_confidence: float = None,
        prop_type: str = None,
    ) -> EdgeResult:
        """
        Calculate edge from a prediction value vs line.

        Uses norm.cdf with prop-specific standard deviations for accurate
        probability conversion, matching the approach in daily_predictions.py.

        Args:
            predicted_value: Model's predicted value
            prop_line: The betting line
            american_odds: Odds for the over
            model_confidence: Optional confidence from model (0-100)
            prop_type: Prop category ('points', 'rebounds', 'assists', 'threes', 'pra')
                       Used to select calibrated std dev. Falls back to default if None.

        Returns:
            EdgeResult
        """
        diff = predicted_value - prop_line

        # Use calibrated prop-specific std dev for norm.cdf conversion
        std_dev = self.PROP_STD_DEVS.get(
            prop_type.lower() if prop_type else '', self.DEFAULT_PROP_STD_DEV
        )
        model_prob = float(norm.cdf(diff / std_dev))

        # If model confidence provided, blend it in
        if model_confidence is not None:
            conf_prob = model_confidence / 100
            model_prob = 0.7 * model_prob + 0.3 * conf_prob

        # Clamp to valid range
        model_prob = max(0.05, min(0.95, model_prob))

        return self.calculate_edge(
            model_probability=model_prob,
            american_odds=american_odds,
        )

    def break_even_probability(self, american_odds: int) -> float:
        """
        Calculate break-even probability for given odds.

        At -110, need to win 52.38% to break even.

        Args:
            american_odds: American odds

        Returns:
            Break-even probability
        """
        return self.american_to_implied_probability(american_odds)

    def required_win_rate(self, american_odds: int, target_roi: float = 0.0) -> float:
        """
        Calculate required win rate for target ROI.

        Args:
            american_odds: American odds
            target_roi: Target ROI (0.0 for break-even, 0.05 for 5% profit)

        Returns:
            Required win rate
        """
        decimal_odds = self.american_to_decimal(american_odds)

        # ROI = (Win% * Profit - Lose% * Stake) / Stake
        # ROI = Win% * (Decimal - 1) - (1 - Win%)
        # ROI = Win% * Decimal - Win% - 1 + Win%
        # ROI = Win% * Decimal - 1
        # Win% = (ROI + 1) / Decimal

        return (target_roi + 1) / decimal_odds


def devig_probability(over_odds: float, under_odds: float) -> tuple[float, float]:
    """Remove vig from odds to get true implied probabilities.

    Args:
        over_odds: American odds for over (e.g., -110)
        under_odds: American odds for under (e.g., -110)

    Returns:
        Tuple of (no_vig_over_prob, no_vig_under_prob)
    """
    def american_to_implied(odds):
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    raw_over = american_to_implied(over_odds)
    raw_under = american_to_implied(under_odds)
    total = raw_over + raw_under  # This is > 1.0 due to vig

    if total == 0:
        return 0.5, 0.5

    return raw_over / total, raw_under / total


# Convenience function
def calculate_edge(
    model_probability: float,
    american_odds: int = -110,
    min_edge: float = 0.03,
) -> EdgeResult:
    """
    Quick edge calculation.

    Args:
        model_probability: Win probability (0-1)
        american_odds: American odds
        min_edge: Minimum edge threshold

    Returns:
        EdgeResult
    """
    calc = EdgeCalculator(min_edge_threshold=min_edge)
    return calc.calculate_edge(model_probability, american_odds)


if __name__ == "__main__":
    # Test edge calculator
    calc = EdgeCalculator()

    print("=" * 60)
    print("EDGE CALCULATOR TEST")
    print("=" * 60)

    # Test odds conversions
    print("\nOdds Conversions:")
    for odds in [-110, -150, +150, -200, +200]:
        decimal = calc.american_to_decimal(odds)
        implied = calc.american_to_implied_probability(odds)
        no_vig = calc.calculate_no_vig_probability(odds)
        print(f"  {odds:+4d} → Decimal: {decimal:.3f}, Implied: {implied:.1%}, No-vig: {no_vig:.1%}")

    # Test vig calculation
    print(f"\nVig at -110/-110: {calc.calculate_vig(-110, -110):.2%}")
    print(f"Vig at -105/-115: {calc.calculate_vig(-105, -115):.2%}")

    # Test edge calculation
    print("\nEdge Calculations:")

    test_cases = [
        (0.55, -110),  # 55% model prob vs -110
        (0.58, -110),  # 58% model prob vs -110
        (0.52, -110),  # 52% model prob vs -110 (marginal)
        (0.60, +100),  # 60% model prob vs +100
        (0.45, +120),  # 45% model prob vs +120 (underdog bet)
    ]

    for model_prob, odds in test_cases:
        result = calc.calculate_edge(model_prob, odds)
        print(f"\n  Model: {model_prob:.0%}, Odds: {odds:+d}")
        print(f"    Implied: {result.implied_probability:.1%}")
        print(f"    Edge: {result.edge_percentage:+.1f}%")
        print(f"    EV/dollar: ${result.ev_per_dollar:+.3f}")
        print(f"    Quality: {result.edge_quality}")
        print(f"    Has Edge: {result.has_edge}")

    # Test prediction-based edge
    print("\nPrediction-Based Edge:")
    result = calc.calculate_edge_from_prediction(
        predicted_value=28.2,
        prop_line=26.5,
        american_odds=-110,
        model_confidence=65,
    )
    print("  Predicted: 28.2 vs Line: 26.5")
    print(f"  Model Probability: {result.model_probability:.1%}")
    print(f"  Edge: {result.edge_percentage:+.1f}%")
    print(f"  EV/dollar: ${result.ev_per_dollar:+.3f}")
