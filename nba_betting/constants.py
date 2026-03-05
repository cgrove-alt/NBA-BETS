"""
Canonical constants for the NBA betting model.

Single source of truth for shared constants used across the prediction pipeline,
edge calculator, agents, and backtesting. All modules must import from here —
never redefine these constants locally.

IMPORTANT: When updating PROP_STD_DEVS, update via empirical analysis of actual
game-by-game stat distributions, not by feel. The values here are calibrated from
NBA historical data. After each model retrain, re-run:

    python3 scripts/calibrate_quantile_decompression.py

to check if std devs have drifted.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Prop-type standard deviations
# ---------------------------------------------------------------------------
# These represent the empirical game-to-game standard deviation of each stat
# category in the NBA. They are used to convert point-differential predictions
# into over/under probabilities via norm.cdf(diff / std_dev).
#
# Derivation:
#   points   — NBA box-score data; mean player-game std dev across starters
#   rebounds — Corrected 2026-02-26 from 7.0 → 3.1 after empirical analysis
#              (old value inflated Z-scores 2×, leading to 76.7% avg over_prob)
#   assists  — Calibrated from 3 seasons of player-game logs
#   threes   — Discrete distribution; Poisson approx ~1.6
#   pra      — Composite: sqrt(6.5²+3.1²+2.2²) ≈ 7.5, inflated for correlation
#
# Do NOT change these without re-running the calibration script.
PROP_STD_DEVS: dict[str, float] = {
    'points':   6.5,   # Empirically-derived from NBA historical data
    'rebounds': 3.1,   # Corrected from 7.0 on 2026-02-26 (was inflating Z-scores ~2x)
    'assists':  2.2,   # Calibrated from 3 seasons of player-game logs
    'threes':   1.6,   # Calibrated; Poisson approximation
    'pra':      8.5,   # Points + Rebounds + Assists combined variance
}

DEFAULT_PROP_STD_DEV: float = 5.0  # fallback when prop type is unknown

# ---------------------------------------------------------------------------
# Edge quality tiers
# ---------------------------------------------------------------------------
# Used consistently by edge_calculator, bet_filter, and daily_predictions.
# Represents the minimum edge (model_prob - implied_prob) to classify a bet.
EDGE_QUALITY_THRESHOLDS: dict[str, float] = {
    'elite':    0.20,   # ≥20% edge — very rare, bet max Kelly
    'strong':   0.12,   # ≥12% edge — high confidence
    'moderate': 0.06,   # ≥6% edge  — worth betting
    'low':      0.03,   # ≥3% edge  — marginal
    # below 3% → 'noise' — do not bet
}

# ---------------------------------------------------------------------------
# Break-even implied probability at standard -110 juice
# ---------------------------------------------------------------------------
BREAK_EVEN_PROB_110: float = 52.38 / 100  # 110 / (110 + 100)

# ---------------------------------------------------------------------------
# Minimum samples for statistically meaningful backtesting
# ---------------------------------------------------------------------------
MIN_BACKTEST_SAMPLES: int = 50

# ---------------------------------------------------------------------------
# Backtest sanity thresholds
# ---------------------------------------------------------------------------
# Values exceeding these almost certainly indicate data leakage or bugs.
# Professional sports bettors achieve 2-8% ROI, 54-57% win rates long-term.
BACKTEST_SANITY: dict[str, float] = {
    'max_roi':          15.0,   # > 15% ROI on test set → leakage red flag
    'max_win_rate':     60.0,   # > 60% win rate at -110 → near-impossible
    'max_sharpe':        3.0,   # > 3.0 Sharpe → hedge-fund tier (unrealistic)
    'max_profit_factor': 3.0,   # Gross profits / gross losses
    'max_training_roi': 50.0,   # Training-set ROI > 50% → train/test leak
    'max_streak_pct':    0.15,  # Win streak > 15% of total bets → suspicious
}

# ---------------------------------------------------------------------------
# Kelly sizing fractions by confidence tier
# ---------------------------------------------------------------------------
KELLY_FRACTIONS: dict[str, float] = {
    'elite':    0.50,   # 50% fractional Kelly for elite edges
    'strong':   0.35,   # 35% fractional Kelly
    'moderate': 0.25,   # 25% fractional Kelly
    'low':      0.00,   # Do not bet marginal edges
}

MAX_BET_FRACTION: float = 0.03    # 3% of bankroll maximum per bet
MIN_BET_FRACTION: float = 0.005   # 0.5% of bankroll minimum

# ---------------------------------------------------------------------------
# Quantile decompression defaults
# ---------------------------------------------------------------------------
# When models/quantile_decompression.json is not present (e.g., first run),
# these defaults are used. Regenerate the JSON after each model retrain:
#
#     python3 scripts/calibrate_quantile_decompression.py
#
# mean_gap = average (predicted_median - line); negative = under-prediction
# slope    = regression slope of predicted_median on line; < 1.0 = compression
# mean_line = average prop line across players (used for slope correction)
QUANTILE_DECOMPRESSION_DEFAULTS: dict[str, dict[str, float]] = {
    'points':   {'slope': 0.724, 'mean_gap': -3.15, 'mean_line': 19.9},
    'rebounds': {'slope': 0.805, 'mean_gap':  0.00, 'mean_line':  5.1},
    'assists':  {'slope': 0.644, 'mean_gap':  0.38, 'mean_line':  4.1},
    'threes':   {'slope': 0.850, 'mean_gap':  0.00, 'mean_line':  2.5},
    'pra':      {'slope': 0.800, 'mean_gap': -1.00, 'mean_line': 30.0},
}

QUANTILE_TARGET_SLOPE: float = 0.85  # target after decompression

# ---------------------------------------------------------------------------
# Exposure / correlation limits
# ---------------------------------------------------------------------------
MAX_TOTAL_EXPOSURE: float  = 0.20   # 20% of bankroll total at risk
MAX_GAME_EXPOSURE: float   = 0.10   # 10% on any single game
MAX_PLAYER_EXPOSURE: float = 0.05   # 5% on any single player
MAX_PROP_TYPE_EXPOSURE: float = 0.20  # 20% per prop category
MAX_CORRELATED_EXPOSURE: float = 0.15  # 15% correlated bets (same game)
