"""
Unified prediction pipeline for NBA-BETS.
Orchestrates: model prediction → probability calibration → bet filtering → Kelly sizing.

Improvement 5: Ties improvements 1-4 together into a single callable interface.

Usage:
    from nba_betting.prediction_pipeline import evaluate_bet

    result = evaluate_bet(
        prop_type='points',
        predicted=28.5,
        line=26.5,
        raw_confidence=0.72,
        games_played=45,
        bankroll=1000.0,
    )
    if result['should_bet']:
        print(f"Bet ${result['bet_size']} on {result['direction']} @ {result['confidence']:.1%}")
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from nba_betting.constants import (
    DISABLED_PROPS,
    KELLY_FRACTIONS,
    MAX_BET_FRACTION,
    DEFAULT_KELLY_FRACTION,
    PROB_CLAMP_MIN,
    PROB_CLAMP_MAX,
    PROB_EDGE_HIGH,
    PROB_EDGE_MEDIUM,
    PROB_EDGE_LOW,
    MIN_BET_TIER,
    BREAK_EVEN_PROB_110,
)

try:
    from nba_betting.odds.devig import american_to_implied, multiplicative_devig
except ImportError:
    def american_to_implied(odds):
        """Convert American odds to raw implied probability (fallback)."""
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)

    def multiplicative_devig(p1, p2):
        """Basic multiplicative devig — divide each prob by the overround (fallback)."""
        total = p1 + p2
        if total == 0:
            return 0.5, 0.5
        return p1 / total, p2 / total

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

# Temperature scaling for probability calibration (Improvement 1)
CALIBRATION_TEMPERATURE = 2.0

# Minimum edges required to place a bet (Improvement 2)
MIN_EDGE = {
    'points': 3.0,
    'rebounds': 2.0,
    'assists': 2.0,
    'pra': 4.0,
    'moneyline': 0.05,
}

# Maximum credible edge — reject bets where model disagrees with line by
# more than this amount. Edges this large usually indicate the model is
# exploiting bench player averaging, not finding genuine matchup edge.
MAX_EDGE = {
    'points': 6.0,    # ±6 pts is a 1-sigma event for most players
    'rebounds': 3.0,
    'assists': 2.5,
    'pra': 8.0,
    'moneyline': 0.15,
}

# DISABLED_PROPS imported from nba_betting.constants (single source of truth)

# Bet sizing constraints (Phase 1.3)
MIN_GAMES = 10          # Minimum player sample size
MIN_CONFIDENCE = 0.65   # Minimum calibrated win probability (raised from 0.62)
MAX_BET_PCT = MAX_BET_FRACTION   # 5% hard cap per bet (from constants)
KELLY_FRACTION = DEFAULT_KELLY_FRACTION  # Quarter-Kelly (from constants)

# Confidence reliability controls
# Early-season samples are noisier; shrink probabilities toward 50% until
# enough games are observed to trust distribution-based confidence.
CONFIDENCE_SHRINK_HALF_LIFE_GAMES = 40.0
CONFIDENCE_SHRINK_MIN_FACTOR = 0.35

# Minimum true EV required when real odds are available.
# Tighter thresholds on weaker/market-efficient markets reduce false positives.
# Raised to 5% to ensure genuine edge over market — low thresholds pass
# bets that are just noise.
MIN_TRUE_EV = {
    'points': 0.05,
    'rebounds': 0.05,
    'assists': 0.05,
    'pra': 0.05,
    'moneyline': 0.04,
    'spread': 0.04,
}

# Minimum probability edge (model_prob - breakeven) for each tier (Phase 1.2).
# These come from constants.py and are the single source of truth.
# Exposed here as locals for use within this module.
_PROB_EDGE_THRESHOLDS = {
    'high':   PROB_EDGE_HIGH,    # 0.07
    'medium': PROB_EDGE_MEDIUM,  # 0.05
    'low':    PROB_EDGE_LOW,     # 0.03
}


# ---------------------------------------------------------------------------
# Odds conversion
# ---------------------------------------------------------------------------

def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds.

    Args:
        odds: American odds (e.g., -110, +150).

    Returns:
        Decimal odds (e.g., 1.909, 2.5).
    """
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


# ---------------------------------------------------------------------------
# Phase 1.2: Probability-edge tier classification
# ---------------------------------------------------------------------------

def get_prob_edge_tier(
    model_prob: float,
    breakeven_prob: float = BREAK_EVEN_PROB_110,
    min_bet_tier: str = MIN_BET_TIER,
) -> tuple[str, float, bool]:
    """
    Classify a bet by its probability edge above the breakeven point.

    The "edge" here is how much the model's win probability exceeds the
    sportsbook's implied breakeven — NOT the difference between predicted
    stat value and the line.

    Tiers (at -110 odds, breakeven ≈ 52.38%):
        High   — model_prob > breakeven + 7%  (i.e., >59.38%)
        Medium — model_prob > breakeven + 5%  (i.e., >57.38%)
        Low    — model_prob > breakeven + 3%  (i.e., >55.38%)
        None   — below Low threshold → do not bet

    Args:
        model_prob:     Calibrated win probability from the model (0–1).
        breakeven_prob: Implied breakeven probability at the offered odds.
                        Defaults to BREAK_EVEN_PROB_110 (52.38%).
        min_bet_tier:   Minimum tier to consider as a "should_bet" decision.
                        Defaults to MIN_BET_TIER ('high').

    Returns:
        Tuple of (tier: str, prob_edge: float, should_bet: bool)
        where tier is 'high' | 'medium' | 'low' | 'noise',
        prob_edge is model_prob - breakeven_prob (can be negative),
        and should_bet reflects whether tier meets min_bet_tier.
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

    # Determine if this tier meets the minimum betting threshold
    tier_rank = {'high': 3, 'medium': 2, 'low': 1, 'noise': 0}
    min_rank = tier_rank.get(min_bet_tier, 3)
    should_bet_flag = tier_rank.get(tier, 0) >= min_rank

    return tier, prob_edge, should_bet_flag


# ---------------------------------------------------------------------------
# Step 1: Probability calibration
# ---------------------------------------------------------------------------

def calibrate_probability(raw_prob: float, temperature: float = CALIBRATION_TEMPERATURE) -> float:
    """
    Apply temperature scaling to a raw probability.

    NOTE: The over_under_classifier is DISABLED. Quantile model probabilities
    (from norm.cdf on quantile spread) are already calibrated — callers should
    pass pre_calibrated=True to evaluate_bet() to skip this function.

    This function is still used as a fallback when raw classifier output is
    provided without pre-calibration.

    Args:
        raw_prob:    Raw probability (0–1).
        temperature: Softening factor.  2.0 is the default.

    Returns:
        Calibrated probability clipped to [0.05, 0.95].
    """
    raw_prob = float(np.clip(raw_prob, 0.01, 0.99))
    logit = np.log(raw_prob / (1 - raw_prob))
    calibrated_logit = logit / temperature
    calibrated = 1 / (1 + np.exp(-calibrated_logit))
    return float(np.clip(calibrated, 0.05, 0.95))


def _sample_size_reliability_factor(
    games_played: int | None,
    half_life_games: float = CONFIDENCE_SHRINK_HALF_LIFE_GAMES,
    min_factor: float = CONFIDENCE_SHRINK_MIN_FACTOR,
) -> float:
    """Map sample size to confidence reliability factor in [min_factor, 1]."""
    if games_played is None:
        return 1.0

    gp = max(0.0, float(games_played))
    if gp <= 0:
        return float(np.clip(min_factor, 0.0, 1.0))

    reliability = 1.0 - float(np.exp(-gp / max(1.0, half_life_games)))
    factor = min_factor + (1.0 - min_factor) * reliability
    return float(np.clip(factor, min_factor, 1.0))


def apply_sample_size_confidence_shrink(
    probability: float,
    games_played: int | None,
) -> float:
    """Shrink confidence toward 50% when sample size is small."""
    p = float(np.clip(probability, 0.05, 0.95))
    factor = _sample_size_reliability_factor(games_played)
    adjusted = 0.5 + (p - 0.5) * factor
    return float(np.clip(adjusted, 0.05, 0.95))


# ---------------------------------------------------------------------------
# Step 2-4: Full pipeline
# ---------------------------------------------------------------------------

def evaluate_bet(
    prop_type: str,
    predicted: float,
    line: float,
    raw_confidence: float | None = None,
    games_played: int | None = None,
    bankroll: float = 1000.0,
    over_odds: int | None = None,
    under_odds: int | None = None,
    pre_calibrated: bool = False,
    min_edge_pct: float | None = None,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
) -> dict:
    """
    Full prediction pipeline: calibrate → filter → size.

    Args:
        prop_type:       One of 'points', 'rebounds', 'assists', 'threes',
                         'pra', 'spread', 'moneyline'.
        predicted:       Model's predicted stat value or margin.
        line:            Market line (over/under number or spread line).
        raw_confidence:  Raw probability from over_under_classifier (0–1).
                         If None, confidence is estimated from edge size.
        games_played:    How many games the player has played this season.
                         If None, sample-size check is skipped.
        bankroll:        Current bankroll in dollars (used for Kelly sizing).
        over_odds:       American odds for the over side (e.g., -115).
                         If None, EV calculation is skipped.
        under_odds:      American odds for the under side (e.g., -105).
                         If None (but over_odds given), raw implied prob is used.
        pre_calibrated:  If True, skip temperature scaling and use raw_confidence
                         directly. Use when confidence comes from a well-calibrated
                         source (e.g., quantile model interpolation).
        min_edge_pct:    Minimum probability edge (model_prob - breakeven_prob)
                         required to generate a bet, as a decimal (e.g. 0.07 for
                         7%). Overrides MIN_BET_TIER from constants if provided.
                         When None, the tier system from constants.MIN_BET_TIER is
                         used (default 'high', i.e. 0.07).
        kelly_fraction:  Fractional Kelly multiplier. Defaults to quarter-Kelly
                         (DEFAULT_KELLY_FRACTION = 0.25). Formula:
                         bet = kelly_fraction × ((b×p−q)/b) × bankroll.

    Returns:
        dict with keys:
            should_bet          (bool)  — whether to place a bet
            direction           (str)   — 'over' or 'under'
            edge                (float) — absolute |predicted - line|
            signed_edge         (float) — predicted - line (positive = over)
            confidence          (float) — calibrated win probability
            bet_size            (float) — recommended bet amount in dollars
            tier                (str)   — 'high' | 'medium' | 'low' | 'noise' | 'no_bet'
            prob_edge_tier      (str)   — same as tier using Phase 1.2 prob-edge system
            prob_edge           (float) — model_prob - breakeven_prob
            reason              (str)   — human-readable explanation
            true_ev             (float | None) — true expected value (when odds provided)
            ev_edge             (float | None) — model prob minus market implied prob
            market_implied_prob (float | None) — devigged market probability
            best_odds           (int | None)   — American odds for the bet side
    """
    # Gate 0: Input validation
    if line is None or line <= 0:
        return {
            'should_bet': False,
            'direction': 'over',
            'edge': 0.0,
            'signed_edge': 0.0,
            'confidence': 0.5,
            'confidence_reliability': None,
            'bet_size': 0.0,
            'tier': 'no_bet',
            'prob_edge_tier': 'noise',
            'prob_edge': 0.0,
            'reason': f'Invalid line ({line}) — skipping edge calculation',
            'true_ev': None,
            'ev_edge': None,
            'market_implied_prob': None,
            'best_odds': None,
        }
    # Gate 0b: NaN/None predicted value guard
    if predicted is None or (isinstance(predicted, float) and np.isnan(predicted)):
        return {
            'should_bet': False,
            'direction': 'over',
            'edge': 0.0,
            'signed_edge': 0.0,
            'confidence': 0.5,
            'confidence_reliability': None,
            'bet_size': 0.0,
            'tier': 'no_bet',
            'prob_edge_tier': 'noise',
            'prob_edge': 0.0,
            'reason': f'Predicted value is None or NaN — skipping',
            'true_ev': None,
            'ev_edge': None,
            'market_implied_prob': None,
            'best_odds': None,
        }

    signed_edge = predicted - line
    abs_edge = abs(signed_edge)

    # Direction determination:
    # When pre-calibrated probability AND real odds are available, compare
    # model probability against MARKET probability. This is the correct
    # approach: bet over when the model thinks over is more likely than
    # the market does, bet under when the model thinks under is more likely.
    # This naturally generates both over AND under bets.
    if (pre_calibrated and raw_confidence is not None
            and over_odds is not None and under_odds is not None):
        raw_over_implied = american_to_implied(over_odds)
        raw_under_implied = american_to_implied(under_odds)
        _nv_over, _nv_under = multiplicative_devig(raw_over_implied, raw_under_implied)
        # Compare model's P(over) against market's devigged P(over)
        direction = 'over' if raw_confidence > _nv_over else 'under'
    elif pre_calibrated and raw_confidence is not None:
        direction = 'over' if raw_confidence > 0.5 else 'under'
    else:
        direction = 'over' if signed_edge > 0 else 'under'

    result = {
        'should_bet': False,
        'direction': direction,
        'edge': abs_edge,
        'signed_edge': signed_edge,
        'confidence': 0.5,
        'confidence_reliability': None,
        'bet_size': 0.0,
        'tier': 'no_bet',
        'prob_edge_tier': 'noise',
        'prob_edge': 0.0,
        'reason': '',
        'true_ev': None,
        'ev_edge': None,
        'market_implied_prob': None,
        'best_odds': None,
    }

    # ---------- Gate 1: Disabled props ----------
    if prop_type in DISABLED_PROPS:
        result['reason'] = (
            f"'{prop_type}' betting disabled — no demonstrated model edge "
            f"(R²=0.31, RMSE matches naive baseline)"
        )
        return result

    # ---------- Gate 2: Sample size ----------
    if games_played is not None and games_played < MIN_GAMES:
        result['reason'] = (
            f"Only {games_played} games played (need {MIN_GAMES}+) — "
            f"insufficient sample for reliable prediction"
        )
        return result

    # ---------- Gate 3: Edge thresholding ----------
    # When pre-calibrated probability AND real odds are available, use
    # PROBABILITY-BASED edge (model_prob - market_implied_prob). This avoids
    # the systematic bias where abs(predicted - line) is dominated by
    # season_avg >> sportsbook_line rather than genuine model skill.
    #
    # When only flat odds are available (simulated backtests), fall back to
    # the point-prediction edge (abs(predicted - line)).
    use_prob_edge = (
        pre_calibrated
        and raw_confidence is not None
        and over_odds is not None
        and under_odds is not None
    )

    if not use_prob_edge:
        # Fallback: point-prediction edge for simulated-line backtests
        threshold = MIN_EDGE.get(prop_type, 2.0)
        if abs_edge < threshold:
            result['reason'] = (
                f"Edge {abs_edge:.2f} < threshold {threshold} for {prop_type}"
            )
            return result

        max_threshold = MAX_EDGE.get(prop_type, 6.0)
        if abs_edge > max_threshold:
            result['reason'] = (
                f"Edge {abs_edge:.2f} > max credible {max_threshold} for {prop_type} "
                f"— likely averaging artifact, not genuine edge"
            )
            return result

    # If using probability edge, skip the point-based gates entirely.
    # The EV gate (Gate 5) below will handle probability-based filtering
    # using the calibrated over_prob vs devigged market implied prob.

    # ---------- Step: Calibrate probability ----------
    if raw_confidence is not None and pre_calibrated:
        # Already calibrated (e.g., from quantile model interpolation)
        confidence = float(np.clip(raw_confidence, 0.05, 0.95))
    elif raw_confidence is not None:
        confidence = calibrate_probability(raw_confidence)
    else:
        # Estimate confidence from edge when no classifier output available.
        # Edge of 1× threshold → ~0.60 confidence; 2× threshold → ~0.70.
        confidence = 0.5 + min(abs_edge / (threshold * 4), 0.20)
        confidence = float(np.clip(confidence, 0.50, 0.70))

    reliability_factor = _sample_size_reliability_factor(games_played)
    confidence = apply_sample_size_confidence_shrink(confidence, games_played)

    result['confidence'] = confidence
    result['confidence_reliability'] = reliability_factor

    # ---------- Gate 4: Minimum confidence / prob-edge (Phase 1.2) ----------
    if use_prob_edge:
        # In probability-edge mode, the model's absolute confidence is biased
        # by survivorship (always P(over) > 0.5). Instead of checking absolute
        # confidence > 0.65, check that the model's probability for the chosen
        # direction exceeds the BREAK-EVEN probability by at least min_edge_pct.
        # Under bets remain unprofitable (46.3% win rate) even with the
        # half-offset distribution correction.
        if direction == 'under':
            result['reason'] = (
                "Under bets disabled — model trained on season-avg proxies, "
                "not sportsbook lines; cannot identify mispriced unders"
            )
            return result

        # Classify by probability edge above breakeven (Phase 1.2 tier system).
        # min_edge_pct overrides MIN_BET_TIER when explicitly provided.
        _prob_tier, _prob_edge_val, _tier_passes = get_prob_edge_tier(
            model_prob=confidence,
            breakeven_prob=BREAK_EVEN_PROB_110,
            min_bet_tier=MIN_BET_TIER,
        )
        result['prob_edge_tier'] = _prob_tier
        result['prob_edge'] = _prob_edge_val

        # Apply explicit min_edge_pct override when caller provides one
        if min_edge_pct is not None:
            _tier_passes = _prob_edge_val >= float(min_edge_pct)

        if not _tier_passes:
            _required = min_edge_pct if min_edge_pct is not None else _PROB_EDGE_THRESHOLDS.get(MIN_BET_TIER, PROB_EDGE_HIGH)
            result['reason'] = (
                f"Probability edge {_prob_edge_val:.3f} below minimum {_required:.3f} "
                f"(tier: {_prob_tier}, model: {confidence:.1%}, "
                f"breakeven: {BREAK_EVEN_PROB_110:.1%})"
            )
            return result
    else:
        # Populate prob_edge fields before early return so diagnostics are accurate
        _prob_tier, _prob_edge_val, _ = get_prob_edge_tier(confidence)
        result['prob_edge_tier'] = _prob_tier
        result['prob_edge'] = _prob_edge_val

        if confidence < MIN_CONFIDENCE:
            result['reason'] = (
                f"Calibrated confidence {confidence:.3f} < minimum {MIN_CONFIDENCE}"
            )
            return result

    # ---------- Step: Tier classification (legacy + Phase 1.2) ----------
    # Use the Phase 1.2 prob-edge tier as the primary tier label.
    # Legacy tiers (elite/strong/moderate) are computed for backward compat.
    if use_prob_edge:
        tier = result['prob_edge_tier']  # 'high' | 'medium' | 'low' | 'noise'
    else:
        threshold = MIN_EDGE.get(prop_type, 2.0)
        ratio = abs_edge / threshold if threshold > 0 else 0
        if ratio >= 2.5:
            tier = 'elite'
        elif ratio >= 1.5:
            tier = 'strong'
        elif ratio >= 1.0:
            tier = 'moderate'
        else:
            tier = 'weak'

    result['tier'] = tier

    # ---------- EV calculation using real odds ----------
    ev_edge = None
    market_implied_prob = None
    true_ev = None
    best_odds = None
    model_prob = confidence if direction == 'over' else (1 - confidence)

    if over_odds is not None and under_odds is not None:
        raw_over_implied = american_to_implied(over_odds)
        raw_under_implied = american_to_implied(under_odds)
        nv_over, nv_under = multiplicative_devig(raw_over_implied, raw_under_implied)

        if direction == 'over':
            market_implied_prob = nv_over
            best_odds = over_odds
        else:
            market_implied_prob = nv_under
            best_odds = under_odds

        ev_edge = model_prob - market_implied_prob
        decimal_odds = american_to_decimal(best_odds)
        true_ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
    elif over_odds is not None:
        raw_implied = american_to_implied(over_odds)
        if direction == 'over':
            market_implied_prob = raw_implied
            best_odds = over_odds
            ev_edge = model_prob - market_implied_prob
            decimal_odds = american_to_decimal(best_odds)
            true_ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
        else:
            # UNDER direction but only over_odds available — can't compute
            # reliable EV. Set informational fields only; leave best_odds/true_ev
            # as None so Kelly falls back to default -110 and EV gate is skipped.
            market_implied_prob = 1 - raw_implied
            ev_edge = model_prob - market_implied_prob

    result['true_ev'] = true_ev
    result['ev_edge'] = ev_edge
    result['market_implied_prob'] = market_implied_prob
    result['best_odds'] = best_odds

    # ---------- Gate 5: Minimum EV (when odds available) ----------
    min_ev = MIN_TRUE_EV.get(prop_type, 0.03)
    if true_ev is not None and true_ev < min_ev:
        result['reason'] = (
            f"True EV {true_ev:.1%} below minimum {min_ev:.0%} "
            f"(model: {model_prob:.1%} vs market: {market_implied_prob:.1%})"
        )
        return result

    # ---------- Step: Kelly bet sizing ----------
    if best_odds is not None:
        odds_decimal = american_to_decimal(best_odds)
    else:
        odds_decimal = 1.909  # Default -110

    b = odds_decimal - 1
    p = confidence if direction == 'over' else (1 - confidence)
    q = 1 - p

    kelly_full = (b * p - q) / b
    if kelly_full <= 0:
        result['reason'] = (
            f"Kelly criterion non-positive ({kelly_full:.4f}) — edge not sufficient "
            f"at these odds (confidence={confidence:.3f})"
        )
        return result

    # ---------- Step: Kelly bet sizing (Phase 1.3) ----------
    # Formula: actual_bet = kelly_fraction × full_kelly × bankroll
    # kelly_fraction defaults to DEFAULT_KELLY_FRACTION (quarter-Kelly = 0.25).
    # All active tiers use the same kelly_fraction; the tier gates entry only.
    # Hard cap at MAX_BET_FRACTION (5%) prevents catastrophic single-bet loss.
    effective_kelly_fraction = float(kelly_fraction)
    bet_fraction = min(kelly_full * effective_kelly_fraction, MAX_BET_FRACTION)
    bet_size = round(bankroll * bet_fraction, 2)

    # ---------- All gates passed ----------
    result['should_bet'] = True
    result['bet_size'] = bet_size
    _edge_label = f"{abs_edge:.2f}" if not use_prob_edge else f"{_prob_edge_val:.1%} above BE"
    result['reason'] = (
        f"{tier.upper()} edge: {_edge_label} ({direction}) | "
        f"confidence: {confidence:.1%} | kelly: {effective_kelly_fraction:.2f}× | "
        f"bet: ${bet_size:.2f}"
    )

    return result


# ---------------------------------------------------------------------------
# Convenience: batch evaluation
# ---------------------------------------------------------------------------

def evaluate_bets_batch(
    predictions: list[dict],
    bankroll: float = 1000.0,
    min_edge_pct: float | None = None,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
) -> list[dict]:
    """
    Evaluate a list of prediction dicts through the full pipeline.

    Each input dict must have keys: prop_type, predicted, line.
    Optional keys: raw_confidence, games_played, over_odds, under_odds,
                   pre_calibrated, min_edge_pct, kelly_fraction.

    Args:
        predictions:  List of prediction dicts.
        bankroll:     Bankroll for Kelly sizing (applied to all bets).
        min_edge_pct: Override minimum probability edge for all bets.
        kelly_fraction: Kelly fraction for all bets (default quarter-Kelly).

    Returns:
        List of evaluate_bet() result dicts (augmented with original input).
    """
    results = []
    for pred in predictions:
        ev = evaluate_bet(
            prop_type=pred['prop_type'],
            predicted=pred['predicted'],
            line=pred['line'],
            raw_confidence=pred.get('raw_confidence'),
            games_played=pred.get('games_played'),
            bankroll=bankroll,
            over_odds=pred.get('over_odds'),
            under_odds=pred.get('under_odds'),
            pre_calibrated=pred.get('pre_calibrated', False),
            min_edge_pct=pred.get('min_edge_pct', min_edge_pct),
            kelly_fraction=pred.get('kelly_fraction', kelly_fraction),
        )
        ev['input'] = pred
        results.append(ev)
    return results
