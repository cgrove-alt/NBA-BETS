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
    'points': 2.0,
    'rebounds': 1.0,
    'assists': 0.8,
    'threes': 999,     # Disabled — no demonstrated model edge
    'pra': 3.0,
    'spread': 2.5,
    'moneyline': 0.05,
}

# Disabled prop types (Improvement 2)
DISABLED_PROPS = ['threes']

# Bet sizing constraints (Improvement 2)
MIN_GAMES = 10          # Minimum player sample size
MIN_CONFIDENCE = 0.58   # Minimum calibrated win probability
MAX_BET_PCT = 0.03      # Hard cap: 3% of bankroll per bet
KELLY_FRACTION = 0.25   # Quarter-Kelly for conservative sizing


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
# Step 1: Probability calibration
# ---------------------------------------------------------------------------

def calibrate_probability(raw_prob: float, temperature: float = CALIBRATION_TEMPERATURE) -> float:
    """
    Apply temperature scaling to a raw classifier probability.

    The over_under_classifier in player_*_ensemble.pkl returns extreme
    values (0.0 or 1.0) because it was never calibrated after training.
    Temperature scaling > 1 softens extremes toward 0.5, preventing
    Kelly sizing from allocating max bankroll on every prop.

    Args:
        raw_prob:    Raw probability from classifier (0–1).
        temperature: Softening factor.  2.0 is a sensible default.

    Returns:
        Calibrated probability clipped to [0.05, 0.95].
    """
    raw_prob = float(np.clip(raw_prob, 0.01, 0.99))
    logit = np.log(raw_prob / (1 - raw_prob))
    calibrated_logit = logit / temperature
    calibrated = 1 / (1 + np.exp(-calibrated_logit))
    return float(np.clip(calibrated, 0.05, 0.95))


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

    Returns:
        dict with keys:
            should_bet          (bool)  — whether to place a bet
            direction           (str)   — 'over' or 'under'
            edge                (float) — absolute |predicted - line|
            signed_edge         (float) — predicted - line (positive = over)
            confidence          (float) — calibrated win probability
            bet_size            (float) — recommended bet amount in dollars
            tier                (str)   — 'elite' | 'strong' | 'moderate' | 'weak' | 'no_bet'
            reason              (str)   — human-readable explanation
            true_ev             (float | None) — true expected value (when odds provided)
            ev_edge             (float | None) — model prob minus market implied prob
            market_implied_prob (float | None) — devigged market probability
            best_odds           (int | None)   — American odds for the bet side
    """
    signed_edge = predicted - line
    abs_edge = abs(signed_edge)
    direction = 'over' if signed_edge > 0 else 'under'

    result = {
        'should_bet': False,
        'direction': direction,
        'edge': abs_edge,
        'signed_edge': signed_edge,
        'confidence': 0.5,
        'bet_size': 0.0,
        'tier': 'no_bet',
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

    # ---------- Gate 3: Minimum edge ----------
    threshold = MIN_EDGE.get(prop_type, 2.0)
    if abs_edge < threshold:
        result['reason'] = (
            f"Edge {abs_edge:.2f} < threshold {threshold} for {prop_type}"
        )
        return result

    # ---------- Step: Calibrate probability ----------
    if raw_confidence is not None:
        confidence = calibrate_probability(raw_confidence)
    else:
        # Estimate confidence from edge when no classifier output available.
        # Edge of 1× threshold → ~0.60 confidence; 2× threshold → ~0.70.
        confidence = 0.5 + min(abs_edge / (threshold * 4), 0.20)
        confidence = float(np.clip(confidence, 0.50, 0.70))

    result['confidence'] = confidence

    # ---------- Gate 4: Minimum confidence ----------
    if confidence < MIN_CONFIDENCE:
        result['reason'] = (
            f"Calibrated confidence {confidence:.3f} < minimum {MIN_CONFIDENCE}"
        )
        return result

    # ---------- Step: Tier classification ----------
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
        else:
            market_implied_prob = 1 - raw_implied
            best_odds = over_odds
        ev_edge = model_prob - market_implied_prob
        decimal_odds = american_to_decimal(best_odds)
        true_ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)

    result['true_ev'] = true_ev
    result['ev_edge'] = ev_edge
    result['market_implied_prob'] = market_implied_prob
    result['best_odds'] = best_odds

    # ---------- Gate 5: Minimum EV (when odds available) ----------
    MIN_EV = 0.03
    if true_ev is not None and true_ev < MIN_EV:
        result['reason'] = (
            f"True EV {true_ev:.1%} below minimum {MIN_EV:.0%} "
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

    bet_fraction = min(kelly_full * KELLY_FRACTION, MAX_BET_PCT)
    bet_size = round(bankroll * bet_fraction, 2)

    # ---------- All gates passed ----------
    result['should_bet'] = True
    result['bet_size'] = bet_size
    result['reason'] = (
        f"{tier.upper()} edge: {abs_edge:.2f} ({direction}) | "
        f"confidence: {confidence:.1%} | bet: ${bet_size:.2f}"
    )

    return result


# ---------------------------------------------------------------------------
# Convenience: batch evaluation
# ---------------------------------------------------------------------------

def evaluate_bets_batch(predictions: list[dict], bankroll: float = 1000.0) -> list[dict]:
    """
    Evaluate a list of prediction dicts through the full pipeline.

    Each input dict must have keys: prop_type, predicted, line.
    Optional keys: raw_confidence, games_played.

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
        )
        ev['input'] = pred
        results.append(ev)
    return results
