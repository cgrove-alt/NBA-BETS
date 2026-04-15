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
    'points':   6.16,  # From backtest RMSE: sqrt(6.31² - 1.38²) = 6.16
    'rebounds': 2.67,  # From backtest RMSE: 2.67 (bias ≈ 0, so RMSE ≈ σ)
    'assists':  1.95,  # From backtest RMSE: sqrt(2.83² - 2.05²) = 1.95
    'threes':   1.36,  # From backtest RMSE: sqrt(1.44² - 0.48²) = 1.36
    'pra':      7.97,  # From backtest RMSE: sqrt(7.98² - 0.45²) = 7.97
}

DEFAULT_PROP_STD_DEV: float = 5.0  # fallback when prop type is unknown

# ---------------------------------------------------------------------------
# Per-prop bias corrections — Re-enabled (Phase 1 fixes, 2026-03-31)
# ---------------------------------------------------------------------------
# Values are the NEGATIVE of the measured aggregate bias from the OOS
# walk-forward backtest (data/backtest_results/oos_walkforward_results.json,
# run 2026-03-22, 2 windows, 33K+ predictions).
#
# Applied as: z_score = (predicted_value + bias_fix - line) / sigma
# A positive model bias (over-predicts) requires a negative correction.
#
# Aggregate biases from the JSON:
#   points:   +0.774  →  correction: -0.774
#   rebounds: +0.259  →  correction: -0.259
#   assists:  +0.049  →  correction: -0.049  (small, near-zero)
#   threes:   not measured (disabled prop)   →  0.0
#   pra:      +0.814  →  correction: -0.814
PROP_BIAS_CORRECTION: dict[str, float] = {
    'points':   -0.774,   # OOS aggregate bias: +0.774 — model over-predicts points
    'rebounds':  -0.259,  # OOS aggregate bias: +0.259 — model over-predicts rebounds
    'assists':   -0.049,  # OOS aggregate bias: +0.049 — near-zero, small correction
    'threes':    0.0,     # Not in OOS walkforward (disabled prop); no correction applied
    'pra':       -0.814,  # OOS aggregate bias: +0.814 — model over-predicts pra
}

# ---------------------------------------------------------------------------
# Disabled prop types (no demonstrated model edge)
# ---------------------------------------------------------------------------
# Post-statistical-analysis configuration (2026-03-22).
# Bootstrap significance: rebounds p=0.027 (significant), PRA p=0.068 (marginal).
# Points p=0.463 — no edge, disabled.
# Assists: 45.3% hit rate (30-day, 2026-04-15 analysis) — disabled.
#   Bias swings both over and under; no fixable directional correction.
#   Paper trading enforcement bug fixed 2026-04-15: api.py best-bets and
#   paper trading loggers now check DISABLED_PROPS before serving/logging.
# Threes too stochastic. Spread worse than market.
#
# ENFORCEMENT: bet_filter.py, prediction_pipeline.py, and backend/api.py
# (best-bets endpoint + both paper trading loggers) all check this list.
DISABLED_PROPS: list[str] = ['points', 'assists', 'threes', 'spread']

# ---------------------------------------------------------------------------
# Probability clamping — safety floor/ceiling for ALL probability outputs
# ---------------------------------------------------------------------------
# Applied after every norm.cdf call, isotonic calibration, and ensemble output
# to prevent degenerate values (0.0/1.0) from flowing into Kelly sizing.
# These are the single source of truth — import and use everywhere.
PROB_CLAMP_MIN: float = 0.05
PROB_CLAMP_MAX: float = 0.95

# ---------------------------------------------------------------------------
# Probability-edge tiers (Phase 1.2 — bet selection filter)
# ---------------------------------------------------------------------------
# Edge = model_prob - BREAK_EVEN_PROB_110 (not market implied prob).
# "7% edge" means the model predicts P(win) = 52.38% + 7% = 59.38%.
# These tiers replace the ratio-to-threshold system for clearer interpretation.
PROB_EDGE_HIGH: float   = 0.07   # >7% above breakeven → "High" confidence tier
PROB_EDGE_MEDIUM: float = 0.05   # 5–7% above breakeven → "Medium" confidence tier
PROB_EDGE_LOW: float    = 0.03   # 3–5% above breakeven → "Low" confidence tier
# below 3% → noise — do not bet

# Minimum tier required to generate a bet recommendation.
# Change to 'medium' or 'low' to increase bet volume at lower conviction.
MIN_BET_TIER: str = 'high'

# ---------------------------------------------------------------------------
# Spread model betting flag (Phase 3.1)
# ---------------------------------------------------------------------------
# The spread model has RMSE 14.2 vs market RMSE of 12-13 — it is worse than
# the market and should NOT generate active picks.  However, the spread model
# output is still useful as a *feature* for the moneyline model (it encodes
# point-differential information that complements win-probability estimation).
#
# SPREAD_BETTING_ENABLED = False  → no spread bets appear in recommendations
# SPREAD_AS_ML_FEATURE   = True   → predicted_spread is injected into
#                                    moneyline features as 'model_spread_pred'
SPREAD_BETTING_ENABLED: bool = False
SPREAD_AS_ML_FEATURE: bool = True

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
# Kelly sizing (Phase 1.3)
# ---------------------------------------------------------------------------
# Quarter-Kelly is the single fractional multiplier applied to full Kelly.
# The bet tier (high/medium/low) determines WHETHER to bet at all;
# sizing always uses DEFAULT_KELLY_FRACTION × full_kelly × bankroll.
#
# Formula: actual_bet = DEFAULT_KELLY_FRACTION × ((b×p − q) / b) × bankroll
#   where b = decimal_odds − 1, p = win_prob, q = 1 − p
#
# Rationale: quarter-Kelly minimises risk of ruin while preserving ~75% of
# the theoretical growth rate of full Kelly (Kelly, 1956; Thorp, 2008).
DEFAULT_KELLY_FRACTION: float = 0.25   # Quarter-Kelly — reduces variance vs full Kelly

# Uniform Kelly fractions: all active tiers use the same DEFAULT_KELLY_FRACTION.
# Tier only gates entry, not sizing. This prevents the old system where
# moderate-tier bets used 0.0625× full Kelly (too small to be actionable).
KELLY_FRACTIONS: dict[str, float] = {
    'high':     DEFAULT_KELLY_FRACTION,   # Phase 1.2 tier: >7% prob edge
    'medium':   DEFAULT_KELLY_FRACTION,   # Phase 1.2 tier: 5–7% prob edge
    'low':      0.00,                     # Phase 1.2 tier: 3–5% — excluded by MIN_BET_TIER
    'elite':    DEFAULT_KELLY_FRACTION,   # Legacy tier alias
    'strong':   DEFAULT_KELLY_FRACTION,   # Legacy tier alias
    'moderate': DEFAULT_KELLY_FRACTION,   # Legacy tier alias
    'weak':     0.00,                     # Legacy: below threshold
    'avoid':    0.00,                     # Legacy: below threshold
}

MAX_BET_FRACTION: float = 0.05    # 5% of bankroll hard cap per bet
MIN_BET_FRACTION: float = 0.005   # 0.5% of bankroll minimum (avoid dust bets)

# ---------------------------------------------------------------------------
# Quantile decompression defaults — DISABLED (Fix 2.2)
# ---------------------------------------------------------------------------
# Removed: decompression was patching a symptom (quantile compression from
# sklearn GBR with 80+ noisy features). Fix 1.1 (feature reduction) and
# Fix 1.3 (LightGBM quantile loss) address the root cause.
# Identity transform: slope=1.0, mean_gap=0.0 → no decompression applied.
QUANTILE_DECOMPRESSION_DEFAULTS: dict[str, dict[str, float]] = {
    'points':   {'slope': 1.0, 'mean_gap': 0.0, 'mean_line': 19.9},
    'rebounds': {'slope': 1.0, 'mean_gap': 0.0, 'mean_line':  5.1},
    'assists':  {'slope': 1.0, 'mean_gap': 0.0, 'mean_line':  4.1},
    'threes':   {'slope': 1.0, 'mean_gap': 0.0, 'mean_line':  2.5},
    'pra':      {'slope': 1.0, 'mean_gap': 0.0, 'mean_line': 30.0},
}

QUANTILE_TARGET_SLOPE: float = 1.0  # identity (no decompression)

# ---------------------------------------------------------------------------
# Exposure / correlation limits
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Phase 4: Odds API integration thresholds
# ---------------------------------------------------------------------------
# Minimum edge over market implied probability to flag a bet recommendation.
# model_prob > implied_prob + MIN_EDGE_OVER_IMPLIED → flag as edge bet.
# Configurable: increase to reduce false positives, decrease to surface more bets.
MIN_EDGE_OVER_IMPLIED: float = 0.03   # 3% edge above vig-free implied probability

# Minimum EV per dollar staked to flag a bet (positive EV threshold).
MIN_EV_PER_DOLLAR: float = 0.02      # $0.02 EV per $1 staked (2 cents on the dollar)

# All sportsbooks queried for line shopping (The Odds API bookmaker keys).
# DraftKings and FanDuel are first-tier; rest provide additional coverage.
LINE_SHOP_BOOKS: list[str] = [
    "draftkings", "fanduel", "betmgm", "caesars", "betrivers", "pointsbet",
]

# Line movement thresholds for prop signals
# A movement larger than this (in absolute line units) is considered significant.
PROP_LINE_MOVEMENT_THRESHOLD: float = 0.5   # 0.5 points movement is notable

MAX_TOTAL_EXPOSURE: float  = 0.20   # 20% of bankroll total at risk
MAX_GAME_EXPOSURE: float   = 0.10   # 10% on any single game
MAX_PLAYER_EXPOSURE: float = 0.05   # 5% on any single player
MAX_PROP_TYPE_EXPOSURE: float = 0.20  # 20% per prop category
MAX_CORRELATED_EXPOSURE: float = 0.15  # 15% correlated bets (same game)
