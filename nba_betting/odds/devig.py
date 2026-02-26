"""
No-Vig Probability Calculation Module

Removes the bookmaker's vig (overround) from odds to compute true implied probabilities.
This is CRITICAL for accurate edge calculation — without devigging, edges are systematically
understated by the vig margin (~2-5%).

Methods:
- Multiplicative (basic): Divide each raw prob by total overround
- Power method: Better for balanced markets
- Shin method: Best for unbalanced markets (favorites vs longshots)
"""

import numpy as np
from typing import Optional


def american_to_implied(odds: float) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def decimal_to_implied(odds: float) -> float:
    """Convert decimal odds to raw implied probability."""
    if odds <= 0:
        return 0.5
    return 1.0 / odds


def implied_to_american(prob: float) -> float:
    """Convert implied probability to American odds."""
    if prob <= 0 or prob >= 1:
        return -110  # Default
    if prob >= 0.5:
        return -100 * prob / (1 - prob)
    return 100 * (1 - prob) / prob


def multiplicative_devig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """
    Basic multiplicative devig — divide each probability by the overround.

    Works well for balanced markets (both sides near 50%).

    Args:
        prob_a: Raw implied probability for side A
        prob_b: Raw implied probability for side B

    Returns:
        Tuple of (no_vig_prob_a, no_vig_prob_b)
    """
    total = prob_a + prob_b
    if total == 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def power_devig(prob_a: float, prob_b: float, max_iter: int = 100) -> tuple[float, float]:
    """
    Power method devig — finds the exponent k such that prob_a^k + prob_b^k = 1.

    Better than multiplicative for balanced markets. Converges to the same
    result as multiplicative when the market is perfectly balanced.

    Args:
        prob_a: Raw implied probability for side A
        prob_b: Raw implied probability for side B
        max_iter: Maximum iterations for binary search

    Returns:
        Tuple of (no_vig_prob_a, no_vig_prob_b)
    """
    if prob_a <= 0 or prob_b <= 0:
        return multiplicative_devig(prob_a, prob_b)

    # Binary search for k
    lo, hi = 0.001, 10.0
    for _ in range(max_iter):
        k = (lo + hi) / 2
        total = prob_a ** k + prob_b ** k
        if total > 1.0:
            lo = k
        else:
            hi = k
        if abs(total - 1.0) < 1e-10:
            break

    return prob_a ** k, prob_b ** k


def shin_devig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """
    Shin method devig — accounts for the favorite-longshot bias.

    Based on Hyun Song Shin's model that explains why favorites are
    underpriced and longshots overpriced. Best for unbalanced markets.

    Args:
        prob_a: Raw implied probability for side A
        prob_b: Raw implied probability for side B

    Returns:
        Tuple of (no_vig_prob_a, no_vig_prob_b)
    """
    total = prob_a + prob_b
    if total <= 1.0:
        return prob_a, prob_b  # No vig to remove

    # Shin's formula: z = (total - 1) / (n - 1) where n = number of outcomes
    z = (total - 1.0) / 1.0  # For 2-way market, n=2, denominator = 1

    # Shin-adjusted probabilities
    def shin_adjust(p, z_val):
        return (np.sqrt(z_val ** 2 + 4 * (1 - z_val) * p ** 2 / total) - z_val) / (2 * (1 - z_val))

    try:
        adj_a = shin_adjust(prob_a, z)
        adj_b = shin_adjust(prob_b, z)
        # Normalize to sum to 1
        total_adj = adj_a + adj_b
        if total_adj > 0:
            return adj_a / total_adj, adj_b / total_adj
    except (ValueError, ZeroDivisionError):
        pass

    return multiplicative_devig(prob_a, prob_b)


def devig_american_odds(
    odds_a: float,
    odds_b: float,
    method: str = "multiplicative",
) -> tuple[float, float]:
    """
    Remove vig from a pair of American odds.

    This is the primary function to use for edge calculation.

    Args:
        odds_a: American odds for side A (e.g., -110 for over)
        odds_b: American odds for side B (e.g., -110 for under)
        method: Devig method - "multiplicative", "power", or "shin"

    Returns:
        Tuple of (no_vig_prob_a, no_vig_prob_b) summing to 1.0

    Example:
        >>> devig_american_odds(-110, -110)
        (0.5, 0.5)  # Fair 50/50 after removing vig

        >>> devig_american_odds(-150, +130)
        (0.5357, 0.4643)  # True implied probabilities
    """
    prob_a = american_to_implied(odds_a)
    prob_b = american_to_implied(odds_b)

    methods = {
        "multiplicative": multiplicative_devig,
        "power": power_devig,
        "shin": shin_devig,
    }

    devig_fn = methods.get(method, multiplicative_devig)
    return devig_fn(prob_a, prob_b)


def calculate_true_edge(
    model_probability: float,
    market_odds_over: float,
    market_odds_under: float,
    side: str = "over",
    devig_method: str = "multiplicative",
) -> dict:
    """
    Calculate the true edge after removing vig.

    Args:
        model_probability: Model's estimated probability (0-1)
        market_odds_over: American odds for over side
        market_odds_under: American odds for under side
        side: Which side we're betting ("over" or "under")
        devig_method: Method for removing vig

    Returns:
        Dictionary with edge metrics:
        - true_edge: Edge against no-vig line (THIS is what matters)
        - vigged_edge: Edge against vigged line (misleadingly lower)
        - no_vig_prob: The true implied probability
        - vig_margin: How much vig the book is charging
        - expected_value: Expected value per $1 wagered
    """
    no_vig_over, no_vig_under = devig_american_odds(
        market_odds_over, market_odds_under, devig_method
    )

    vigged_over = american_to_implied(market_odds_over)
    vigged_under = american_to_implied(market_odds_under)

    if side == "over":
        no_vig_prob = no_vig_over
        vigged_prob = vigged_over
        decimal_odds = 1 / vigged_over if vigged_over > 0 else 2.0
    else:
        no_vig_prob = no_vig_under
        vigged_prob = vigged_under
        decimal_odds = 1 / vigged_under if vigged_under > 0 else 2.0

    true_edge = model_probability - no_vig_prob
    vigged_edge = model_probability - vigged_prob
    vig_margin = (vigged_over + vigged_under) - 1.0

    # Expected value: prob * (payout) - (1-prob) * (stake)
    # = prob * (decimal_odds - 1) - (1 - prob)
    ev = model_probability * (decimal_odds - 1) - (1 - model_probability)

    return {
        "true_edge": true_edge,
        "vigged_edge": vigged_edge,
        "no_vig_prob": no_vig_prob,
        "vigged_prob": vigged_prob,
        "vig_margin": vig_margin,
        "expected_value": ev,
        "decimal_odds": decimal_odds,
        "is_profitable": true_edge > 0 and ev > 0,
    }


# Convenience aliases
remove_vig = devig_american_odds
get_fair_odds = devig_american_odds
