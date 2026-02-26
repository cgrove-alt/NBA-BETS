"""
nba_betting.odds — Odds utilities and market data.

Modules:
- devig: No-vig probability calculation (multiplicative, power, Shin methods)
- betting_market_features: Market feature extraction
- closing_odds_scheduler: Closing odds data scheduling
- market_microstructure: Market microstructure analysis
- odds_tracker_service: Real-time odds tracking service
"""

from .devig import (
    american_to_implied,
    decimal_to_implied,
    implied_to_american,
    multiplicative_devig,
    power_devig,
    shin_devig,
    devig_american_odds,
    calculate_true_edge,
    remove_vig,
    get_fair_odds,
)

__all__ = [
    "american_to_implied",
    "decimal_to_implied",
    "implied_to_american",
    "multiplicative_devig",
    "power_devig",
    "shin_devig",
    "devig_american_odds",
    "calculate_true_edge",
    "remove_vig",
    "get_fair_odds",
]
