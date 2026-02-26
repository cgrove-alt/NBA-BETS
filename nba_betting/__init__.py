"""
nba_betting — Core betting logic library for the NBA prediction model.

Sub-packages:
- odds:     Odds conversion, devigging, and market data utilities
- signals:  Confidence scoring and betting signal generation
- edge:     Edge quality metrics, bet tracking, and CLV analysis
- bankroll: Kelly sizing, portfolio optimization, and risk management

New modules (v3 improvements):
- bet_filter:          Smart bet selection — only bet when edge + confidence justify it
- prediction_pipeline: Unified orchestrator (calibrate → filter → size)
"""

from .odds import devig_american_odds, calculate_true_edge
from .signals import ConfidenceEngine
from .bet_filter import should_bet, calculate_bet_size, get_edge_tier
from .prediction_pipeline import calibrate_probability, evaluate_bet, evaluate_bets_batch

__all__ = [
    # Legacy exports
    "devig_american_odds",
    "calculate_true_edge",
    "ConfidenceEngine",
    # Improvement 2: bet filter
    "should_bet",
    "calculate_bet_size",
    "get_edge_tier",
    # Improvement 5: prediction pipeline
    "calibrate_probability",
    "evaluate_bet",
    "evaluate_bets_batch",
]
