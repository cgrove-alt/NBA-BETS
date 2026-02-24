"""
Edge Calculator - Convert Predictions to Actionable Bet Recommendations

This module provides:
1. EdgeCalculator - Calculate edge and expected value
2. KellyCriterion - Optimal bet sizing with fractional Kelly
3. BetRecommender - Final recommendations with reasoning
4. BankrollManager - Track bankroll and exposure limits

Usage:
    from edge_calculator import BetRecommender

    recommender = BetRecommender(bankroll=1000)
    recommendations = recommender.analyze_props(props_data)

    for rec in recommendations:
        print(f"{rec['player']} {rec['pick']} {rec['line']}: {rec['suggested_units']}u")
"""

from .edge_calculator import EdgeCalculator, EdgeResult
from .kelly_criterion import KellyCriterion, BetSize
from .bankroll_manager import BankrollManager, ExposureTracker
from .bet_recommender import BetRecommender, BetRecommendation, ConfidenceTier

__all__ = [
    'EdgeCalculator',
    'EdgeResult',
    'KellyCriterion',
    'BetSize',
    'BankrollManager',
    'ExposureTracker',
    'BetRecommender',
    'BetRecommendation',
    'ConfidenceTier',
]
