# SPREAD MODEL STATUS: DISABLED
# Last evaluation: 2026-02-26
# Spread RMSE: 14.2 (market baseline: ~12-13)
# Decision: Disabled — RMSE above market, no positive ATS edge
"""
Smart bet selection filter for NBA-BETS.
Only recommends bets where the model has a meaningful edge.

Improvement 2: Bet Selection Filter
- Filters out props with no model edge (e.g. threes)
- Enforces minimum edge thresholds per prop type
- Requires sufficient player sample size (10+ games)
- Enforces minimum calibrated confidence
"""

import numpy as np
from nba_betting.constants import (
    DISABLED_PROPS,
    PROB_EDGE_HIGH,
    PROB_EDGE_MEDIUM,
    PROB_EDGE_LOW,
    MIN_BET_TIER,
    BREAK_EVEN_PROB_110,
    DEFAULT_KELLY_FRACTION,
    MAX_BET_FRACTION,
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_EDGE_THRESHOLDS = {
    'points': 3.0,     # Need 3+ point edge for points props
    'rebounds': 2.0,   # Need 2+ rebound edge
    'assists': 2.0,    # Need 2+ assist edge (14% hit rate demands higher bar)
    'threes': 999,     # Effectively disabled (R²=0.31, no edge vs naive baseline)
    'pra': 4.0,        # Need 4+ PRA edge
    'spread': 999,      # Effectively disabled (RMSE 14.2, above market ~12-13)
    'moneyline': 0.05, # Need 5% probability edge
}

# EV-based thresholds (used when real odds are available)
# These take priority over stat-based edge thresholds
MIN_EV_THRESHOLDS = {
    'points': 0.03,     # 3% minimum EV
    'rebounds': 0.03,
    'assists': 0.03,
    'threes': 999,      # Still disabled
    'pra': 0.03,
    'spread': 0.04,     # Higher threshold — model is weaker here
    'moneyline': 0.04,
}

MIN_GAMES_PLAYED = 10  # Minimum games for reliable player predictions
MIN_CONFIDENCE = 0.62  # Minimum calibrated probability to bet

# DISABLED_PROPS imported from nba_betting.constants (single source of truth)


# ---------------------------------------------------------------------------
# Phase 1.2: Probability-edge tier classification
# ---------------------------------------------------------------------------

def get_bet_confidence_tier(
    model_prob: float,
    breakeven_prob: float = BREAK_EVEN_PROB_110,
    min_bet_tier: str = MIN_BET_TIER,
) -> tuple[str, float, bool]:
    """
    Classify a calibrated win probability into a confidence tier.

    The tier reflects how far the model's probability exceeds the sportsbook
    break-even, NOT an absolute probability threshold.

    Tiers at -110 odds (breakeven ≈ 52.38%):
        High   — prob_edge > 7%  (model_prob > ~59.4%) → bet
        Medium — prob_edge > 5%  (model_prob > ~57.4%) → bet if MIN_BET_TIER ≤ 'medium'
        Low    — prob_edge > 3%  (model_prob > ~55.4%) → bet if MIN_BET_TIER = 'low'
        Noise  — below 3% edge  → do not bet

    Args:
        model_prob:     Calibrated win probability (0–1). Must be clamped to
                        [PROB_CLAMP_MIN, PROB_CLAMP_MAX] before calling.
        breakeven_prob: Implied breakeven probability from the offered odds.
        min_bet_tier:   Minimum tier for a positive bet decision.

    Returns:
        Tuple of (tier, prob_edge, should_bet).
    """
    prob_edge = float(model_prob) - float(breakeven_prob)

    if prob_edge >= PROB_EDGE_HIGH:
        tier = 'high'
    elif prob_edge >= PROB_EDGE_MEDIUM:
        tier = 'medium'
    elif prob_edge >= PROB_EDGE_LOW:
        tier = 'low'
    else:
        tier = 'noise'

    tier_rank = {'high': 3, 'medium': 2, 'low': 1, 'noise': 0}
    should = tier_rank.get(tier, 0) >= tier_rank.get(min_bet_tier, 3)
    return tier, prob_edge, should


# ---------------------------------------------------------------------------
# Core filter function
# ---------------------------------------------------------------------------

def should_bet(prop_type: str, predicted_value: float, line_value: float,
               confidence: float = None, games_played: int = None,
               is_over: bool = True, true_ev: float = None,
               min_edge_pct: float = None):
    """
    Determine if a prediction warrants a bet.

    Args:
        prop_type:       One of 'points', 'rebounds', 'assists', 'threes',
                         'pra', 'spread', 'moneyline'.
        predicted_value: Model's predicted value for the stat / margin.
        line_value:      Market line (over/under or spread number).
        confidence:      Calibrated win probability from the model (0-1).
                         If provided, the Phase 1.2 probability-edge tier check
                         is applied in addition to stat-based edge thresholds.
        games_played:    Number of games the player has played this season.
                         If None, sample size check is skipped.
        is_over:         True  → we are betting the OVER / home ATS.
                         False → we are betting the UNDER / away ATS.
        true_ev:         True expected value from devigged odds (0-1 scale).
                         If provided, EV gate takes priority over stat-based edge.
        min_edge_pct:    Minimum probability edge above breakeven required to bet.
                         If None, uses MIN_BET_TIER from constants (default 'high'
                         → requires PROB_EDGE_HIGH = 0.07 = 7% above breakeven).

    Returns:
        Tuple of (should_bet: bool, reason: str, edge: float)
    """
    # Check if prop type is disabled
    if prop_type in DISABLED_PROPS:
        return False, f"Prop type '{prop_type}' is disabled (no model edge)", 0.0

    # Check sample size
    if games_played is not None and games_played < MIN_GAMES_PLAYED:
        return False, (
            f"Insufficient sample: {games_played} games < {MIN_GAMES_PLAYED} minimum"
        ), 0.0

    # Calculate edge (positive = favours our bet direction)
    if is_over:
        edge = predicted_value - line_value
    else:
        edge = line_value - predicted_value

    # When EV is available, it takes priority over stat-based edge
    if true_ev is not None:
        ev_threshold = MIN_EV_THRESHOLDS.get(prop_type, 0.03)
        if true_ev < ev_threshold:
            return False, (
                f"True EV {true_ev:.1%} below minimum {ev_threshold:.0%} for {prop_type}"
            ), edge

    # Check minimum edge threshold
    threshold = MIN_EDGE_THRESHOLDS.get(prop_type, 2.0)
    if abs(edge) < threshold:
        return False, (
            f"Edge {edge:.2f} below threshold {threshold} for {prop_type}"
        ), edge

    # Check confidence
    if confidence is not None and confidence < MIN_CONFIDENCE:
        return False, (
            f"Confidence {confidence:.3f} below minimum {MIN_CONFIDENCE}"
        ), edge

    # Phase 1.2: Probability-edge tier gate.
    # When calibrated confidence is available, also check that the model's
    # win probability exceeds the sportsbook break-even by the required margin.
    if confidence is not None:
        _tier, _prob_edge, _passes = get_bet_confidence_tier(
            model_prob=confidence,
            breakeven_prob=BREAK_EVEN_PROB_110,
            min_bet_tier=MIN_BET_TIER,
        )
        # Apply explicit min_edge_pct override when provided
        if min_edge_pct is not None:
            _passes = _prob_edge >= float(min_edge_pct)
            _required = float(min_edge_pct)
        else:
            _required = {
                'high': PROB_EDGE_HIGH,
                'medium': PROB_EDGE_MEDIUM,
                'low': PROB_EDGE_LOW,
            }.get(MIN_BET_TIER, PROB_EDGE_HIGH)

        if not _passes:
            return False, (
                f"Probability edge {_prob_edge:.3f} < required {_required:.3f} "
                f"(tier: {_tier}, confidence: {confidence:.3f}, "
                f"breakeven: {BREAK_EVEN_PROB_110:.4f})"
            ), edge

    tier = get_edge_tier(edge, prop_type)
    conf_str = f"{confidence:.3f}" if confidence is not None else "N/A"
    return True, f"Edge={edge:.2f}, Confidence={conf_str}, Tier={tier}", edge


# ---------------------------------------------------------------------------
# Bet sizing
# ---------------------------------------------------------------------------

def calculate_bet_size(edge: float, confidence: float, bankroll: float,
                       prop_type: str = 'points',
                       max_bet_pct: float = MAX_BET_FRACTION,
                       kelly_fraction: float = DEFAULT_KELLY_FRACTION,
                       decimal_odds: float = 1.909) -> float:
    """
    Calculate recommended bet size using fractional Kelly criterion (Phase 1.3).

    Formula: actual_bet = kelly_fraction × full_kelly × bankroll
    where full_kelly = (b×p − q) / b, b = decimal_odds − 1, q = 1 − p.

    Args:
        edge:           Predicted edge (|predicted - line|). Used for logging only.
        confidence:     Calibrated win probability (0–1). Must be > 0.5 to bet.
        bankroll:       Current bankroll in dollars.
        prop_type:      Prop type (informational only).
        max_bet_pct:    Hard cap as fraction of bankroll (default: MAX_BET_FRACTION = 5%).
        kelly_fraction: Fractional Kelly multiplier (default: DEFAULT_KELLY_FRACTION = 0.25,
                        i.e. quarter-Kelly). Configurable for tuning.
        decimal_odds:   Decimal odds for the bet (default 1.909 = -110 American).

    Returns:
        Recommended bet amount in dollars (0 if Kelly is non-positive).
    """
    if confidence <= 0.5:
        return 0.0

    b = decimal_odds - 1  # Net profit per unit staked
    if b <= 0:
        # decimal_odds <= 1.0 means no profit or a loss on a win — invalid for Kelly
        return 0.0
    p = float(confidence)
    q = 1.0 - p

    # Full Kelly: f* = (b×p − q) / b
    kelly_full = (b * p - q) / b

    if kelly_full <= 0:
        return 0.0

    # Fractional Kelly (Phase 1.3: actual_bet = kelly_fraction × full_kelly × bankroll)
    bet_fraction = kelly_full * float(kelly_fraction)

    # Hard cap per bet (Phase 1.3: default 5%)
    bet_fraction = min(bet_fraction, float(max_bet_pct))

    return round(bankroll * bet_fraction, 2)


# ---------------------------------------------------------------------------
# Edge tier classification
# ---------------------------------------------------------------------------

def get_edge_tier(edge: float, prop_type: str) -> str:
    """
    Classify edge quality into tiers relative to the threshold for that prop.

    Tiers:
        elite    — edge ≥ 2.5× threshold
        strong   — edge ≥ 1.5× threshold
        moderate — edge ≥ 1.0× threshold
        weak     — edge ≥ 0.5× threshold
        no_bet   — below threshold

    Returns:
        Tier string.
    """
    threshold = MIN_EDGE_THRESHOLDS.get(prop_type, 2.0)
    if threshold <= 0 or threshold >= 999:
        return 'no_bet'

    ratio = abs(edge) / threshold

    if ratio >= 2.5:
        return 'elite'
    elif ratio >= 1.5:
        return 'strong'
    elif ratio >= 1.0:
        return 'moderate'
    elif ratio >= 0.5:
        return 'weak'
    else:
        return 'no_bet'
