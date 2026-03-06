"""
Smart bet selection filter for NBA-BETS.
Only recommends bets where the model has a meaningful edge.

Bet Selection Filter
- Enforces minimum edge thresholds per prop type
- Requires sufficient player sample size (10+ games)
- Enforces minimum calibrated confidence

THREES RE-ENABLED (2026-03-06):
  Threes now use Poisson CDF instead of Normal CDF for over probability.
  Poisson is the correct distribution for discrete, low-count stats like
  3-pointers (mean ~2.5).  The R²=0.31 figure was for regression accuracy,
  not for over/under prediction — the relevant metric is calibrated edge, and
  Poisson-based probabilities are well-calibrated for this stat.
  Minimum EV threshold raised to 4% (vs 3% for other props) to compensate for
  higher variance, and Kelly fraction is halved for threes bets.

SPREAD MODEL STATUS: DISABLED
  Last evaluation: 2026-02-26
  Spread RMSE: 14.2 (market baseline: ~12-13)
  Decision: Disabled — RMSE above market, no positive ATS edge
"""

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_EDGE_THRESHOLDS = {
    'points': 2.0,      # Need 2+ point edge for points props
    'rebounds': 1.0,    # Need 1+ rebound edge
    'assists': 0.8,     # Need 0.8+ assist edge
    'threes': 0.5,      # Re-enabled 2026-03-06 with Poisson probabilities
    'pra': 3.0,         # Need 3+ PRA edge
    'spread': 999,      # Effectively disabled (RMSE 14.2, above market ~12-13)
    'moneyline': 0.05,  # Need 5% probability edge
}

# EV-based thresholds (used when real odds are available)
# These take priority over stat-based edge thresholds
MIN_EV_THRESHOLDS = {
    'points': 0.03,     # 3% minimum EV
    'rebounds': 0.03,
    'assists': 0.03,
    'threes': 0.04,     # Higher threshold for threes (higher variance bet)
    'pra': 0.03,
    'spread': 0.04,     # Higher threshold — model is weaker here
    'moneyline': 0.04,
}

MIN_GAMES_PLAYED = 10  # Minimum games for reliable player predictions
MIN_CONFIDENCE = 0.58  # Minimum calibrated probability to bet

# Kelly fraction multiplier per prop type (applied on top of tier-based fraction).
# Threes have high variance (Poisson σ/μ = 1 for pure Poisson but actual NBA data
# shows overdispersion), so we use 50% of the base Kelly to manage variance.
KELLY_PROP_MULTIPLIERS: dict[str, float] = {
    'points': 1.0,
    'rebounds': 1.0,
    'assists': 1.0,
    'threes': 0.5,   # High-variance discrete stat — half Kelly
    'pra': 1.0,
    'spread': 0.0,   # Disabled
    'moneyline': 1.0,
}

DISABLED_PROPS = ['spread']  # Props where model has no demonstrated edge


# ---------------------------------------------------------------------------
# Core filter function
# ---------------------------------------------------------------------------

def should_bet(prop_type: str, predicted_value: float, line_value: float,
               confidence: float = None, games_played: int = None,
               is_over: bool = True, true_ev: float = None,
               overdispersion: float = None, ewma_value: float = None,
               poisson_rate: float = None):
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
        overdispersion:  Poisson overdispersion ratio (threes only). High values
                         (>3.5) indicate very inconsistent 3-point shooting.
        ewma_value:      Exponentially-weighted recent stat average. Used for
                         direction alignment check (don't bet over on players
                         whose EWMA is trending below the line).
        poisson_rate:    Estimated Poisson rate (lambda) for threes. Provides an
                         independent check on the model's predicted value.

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

    ptype = prop_type.lower() if prop_type else ''

    # Threes-specific guards (use Poisson rate and overdispersion)
    if ptype == 'threes':
        # Reject highly overdispersed shooters — Poisson approximation breaks down
        if overdispersion is not None and overdispersion > 3.5:
            return False, (
                f"Threes overdispersion too high ({overdispersion:.1f}×) — "
                "inconsistent shooter, Poisson edge unreliable"
            ), edge

        # Poisson rate must clearly support bet direction (independent of model)
        if poisson_rate is not None:
            if is_over and poisson_rate <= line_value:
                return False, (
                    f"Poisson rate ({poisson_rate:.2f}) does not support OVER "
                    f"vs line ({line_value:.1f})"
                ), edge
            elif not is_over and poisson_rate >= line_value:
                return False, (
                    f"Poisson rate ({poisson_rate:.2f}) does not support UNDER "
                    f"vs line ({line_value:.1f})"
                ), edge

    # EWMA direction alignment check (applies to all props).
    # If the EWMA is trending opposite to our bet direction, reduce confidence.
    # We don't veto the bet but we require a higher edge to compensate.
    ewma_misaligned = False
    if ewma_value is not None:
        ewma_edge = ewma_value - line_value if is_over else line_value - ewma_value
        if ewma_edge < 0:
            ewma_misaligned = True

    # When EV is available, it takes priority over stat-based edge
    if true_ev is not None:
        ev_threshold = MIN_EV_THRESHOLDS.get(prop_type, 0.03)
        # Increase threshold if EWMA is misaligned
        if ewma_misaligned:
            ev_threshold = ev_threshold * 1.5
        if true_ev < ev_threshold:
            return False, (
                f"True EV {true_ev:.1%} below {'adjusted ' if ewma_misaligned else ''}"
                f"minimum {ev_threshold:.0%} for {prop_type}"
            ), edge

    # Check minimum edge threshold
    threshold = MIN_EDGE_THRESHOLDS.get(prop_type, 2.0)
    # Increase required edge if EWMA is misaligned
    if ewma_misaligned:
        threshold = threshold * 1.5
    if abs(edge) < threshold:
        return False, (
            f"Edge {edge:.2f} below {'adjusted ' if ewma_misaligned else ''}"
            f"threshold {threshold:.2f} for {prop_type}"
        ), edge

    # Check confidence
    if confidence is not None and confidence < MIN_CONFIDENCE:
        return False, (
            f"Confidence {confidence:.3f} below minimum {MIN_CONFIDENCE}"
        ), edge

    tier = get_edge_tier(edge, prop_type)
    conf_str = f"{confidence:.3f}" if confidence is not None else "N/A"
    ewma_note = " (EWMA misaligned — higher threshold required)" if ewma_misaligned else ""
    return True, f"Edge={edge:.2f}, Confidence={conf_str}, Tier={tier}{ewma_note}", edge


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
        prop_type:      Prop type — used to apply per-prop Kelly multiplier.
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

    # Apply prop-type-specific Kelly multiplier (e.g., 0.5× for threes)
    prop_multiplier = KELLY_PROP_MULTIPLIERS.get(prop_type.lower(), 1.0)
    adjusted_kelly_fraction = kelly_fraction * prop_multiplier

    # Fractional Kelly
    bet_fraction = kelly_full * adjusted_kelly_fraction

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
