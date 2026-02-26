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

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_EDGE_THRESHOLDS = {
    'points': 2.0,     # Need 2+ point edge for points props
    'rebounds': 1.0,   # Need 1+ rebound edge
    'assists': 0.8,    # Need 0.8+ assist edge
    'threes': 999,     # Effectively disabled (R²=0.31, no edge vs naive baseline)
    'pra': 3.0,        # Need 3+ PRA edge
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
MIN_CONFIDENCE = 0.58  # Minimum calibrated probability to bet

DISABLED_PROPS = ['threes', 'spread']  # Props where model has no demonstrated edge


# ---------------------------------------------------------------------------
# Core filter function
# ---------------------------------------------------------------------------

def should_bet(prop_type: str, predicted_value: float, line_value: float,
               confidence: float = None, games_played: int = None,
               is_over: bool = True, true_ev: float = None):
    """
    Determine if a prediction warrants a bet.

    Args:
        prop_type:       One of 'points', 'rebounds', 'assists', 'threes',
                         'pra', 'spread', 'moneyline'.
        predicted_value: Model's predicted value for the stat / margin.
        line_value:      Market line (over/under or spread number).
        confidence:      Calibrated win probability from the model (0-1).
                         If None, confidence check is skipped.
        games_played:    Number of games the player has played this season.
                         If None, sample size check is skipped.
        is_over:         True  → we are betting the OVER / home ATS.
                         False → we are betting the UNDER / away ATS.
        true_ev:         True expected value from devigged odds (0-1 scale).
                         If provided, EV gate takes priority over stat-based edge.

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

    tier = get_edge_tier(edge, prop_type)
    conf_str = f"{confidence:.3f}" if confidence is not None else "N/A"
    return True, f"Edge={edge:.2f}, Confidence={conf_str}, Tier={tier}", edge


# ---------------------------------------------------------------------------
# Bet sizing
# ---------------------------------------------------------------------------

def calculate_bet_size(edge: float, confidence: float, bankroll: float,
                       prop_type: str = 'points', max_bet_pct: float = 0.03,
                       kelly_fraction: float = 0.25) -> float:
    """
    Calculate recommended bet size using fractional Kelly criterion.

    Args:
        edge:           Predicted edge (|predicted - line|).
        confidence:     Calibrated win probability (0-1).
        bankroll:       Current bankroll in dollars.
        prop_type:      Prop type (used for logging only).
        max_bet_pct:    Hard cap on bet as fraction of bankroll (default 3%).
        kelly_fraction: What fraction of full Kelly to use (default 25%).

    Returns:
        Recommended bet amount in dollars (0 if Kelly is non-positive).
    """
    if confidence <= 0.5:
        return 0.0

    # Standard -110 odds → decimal odds = 1.909
    odds_decimal = 1.909
    b = odds_decimal - 1  # Net profit per unit staked = 0.909
    p = confidence
    q = 1 - p

    # Full Kelly: f* = (b*p - q) / b
    kelly_full = (b * p - q) / b

    if kelly_full <= 0:
        return 0.0

    # Fractional Kelly
    bet_fraction = kelly_full * kelly_fraction

    # Hard cap at max_bet_pct
    bet_fraction = min(bet_fraction, max_bet_pct)

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
