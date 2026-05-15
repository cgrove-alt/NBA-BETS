"""Portfolio-level exposure caps for daily prediction batches.

The per-bet Kelly sizing in prediction_pipeline.py treats each bet in isolation.
That is correct for the marginal bet but produces concentrated exposure when
multiple high-edge bets land on the same game, same player, or same prop type
— all of which are heavily correlated.

This module applies portfolio-level caps after per-bet sizing:

  - MAX_TOTAL_EXPOSURE      total bankroll at risk across all bets
  - MAX_GAME_EXPOSURE       cap per NBA game (same-game player props correlate)
  - MAX_PLAYER_EXPOSURE     cap per player (same player props strongly correlate)
  - MAX_PROP_TYPE_EXPOSURE  cap per prop category (e.g., total PRA exposure)
  - MAX_CORRELATED_EXPOSURE cap on the union of game + prop-type buckets
                            (catches the "10 PRA bets on tonight's high-pace game"
                             pattern that would otherwise pass all per-bucket caps)

Algorithm: greedy by edge. Sort the candidate bets descending by their reported
edge metric. Walk the list and admit each bet if and only if it fits within
every cap. Bets that don't fit are marked with a 'cap_rejected' reason so they
remain visible in the output but won't be staked.

Caps come from nba_betting.constants — the single source of truth. The function
does not redefine them locally.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from nba_betting.constants import (
    MAX_CORRELATED_EXPOSURE,
    MAX_GAME_EXPOSURE,
    MAX_PLAYER_EXPOSURE,
    MAX_PROP_TYPE_EXPOSURE,
    MAX_TOTAL_EXPOSURE,
)


def _bet_size_fraction(bet: dict) -> float:
    """Extract the bet size as a fraction of bankroll (0-1).

    daily_predictions.py always stores `suggested_bet_size` as a percent of
    bankroll (0-100): see the kelly_full * 0.25 * 100 expressions throughout
    predict_player_prop. A 0.5% bet is stored as 0.5, a 5% bet as 5.0. We
    therefore divide every value by 100 — the previous heuristic ("values <=1
    are fractions") would have admitted a 0.5% bet at 50% bankroll, blowing
    portfolio caps immediately. Hard-cap at 1.0 so a corrupt input >100 can't
    blow up downstream math.
    """
    raw = bet.get('suggested_bet_size', 0.0)
    if raw is None:
        return 0.0
    try:
        pct = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if pct <= 0:
        return 0.0
    fraction = pct / 100.0
    return min(fraction, 1.0)


def _edge_score(bet: dict) -> float:
    """Best-available edge proxy for ranking bets.

    Prefer EV per dollar when present (most honest), fall back to over/under
    edge magnitude, then signed edge. Missing → 0.
    """
    for key in ('ev_per_dollar', 'over_edge', 'edge', 'prob_edge'):
        val = bet.get(key)
        if val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        return abs(v) if key in ('over_edge', 'edge') else v
    return 0.0


def _bet_key(bet: dict) -> tuple[str, str, str]:
    """(game_id, player_key, prop_type) — used for caps."""
    game = bet.get('game') or bet.get('game_id') or 'unknown_game'
    player = (
        bet.get('player_id')
        or bet.get('player')
        or bet.get('player_name')
        or 'unknown_player'
    )
    prop = (bet.get('stat') or bet.get('prop_type') or 'unknown_prop').lower()
    return str(game), str(player), prop


def apply_exposure_caps(
    bets: list[dict],
    *,
    max_total: float = MAX_TOTAL_EXPOSURE,
    max_game: float = MAX_GAME_EXPOSURE,
    max_player: float = MAX_PLAYER_EXPOSURE,
    max_prop_type: float = MAX_PROP_TYPE_EXPOSURE,
    max_correlated: float = MAX_CORRELATED_EXPOSURE,
) -> dict[str, Any]:
    """Filter a list of bet dicts against portfolio exposure caps.

    Mutates each input dict in place — admitted bets gain `cap_admitted=True`,
    rejected bets gain `cap_admitted=False` and `cap_rejected_reason`. The bet
    size of admitted bets is unchanged; rejected bets keep their original size
    in the dict for visibility but should be staked at 0.

    Only bets with a positive `suggested_bet_size` AND a `bet_recommendation`
    of 'BET' (case-insensitive) are eligible for admission. Everything else
    passes through with `cap_admitted=None`.

    Returns a summary dict with counts and per-bucket exposures, suitable for
    logging.
    """
    summary = {
        'eligible': 0,
        'admitted': 0,
        'rejected': 0,
        'total_exposure': 0.0,
        'rejections_by_reason': defaultdict(int),
    }

    # Partition: only 'BET' recommendations with positive size are eligible.
    candidates: list[tuple[float, int, dict]] = []
    for idx, bet in enumerate(bets):
        rec = str(bet.get('bet_recommendation', '')).upper()
        size = _bet_size_fraction(bet)
        if rec != 'BET' or size <= 0:
            bet['cap_admitted'] = None
            continue
        summary['eligible'] += 1
        # Negative idx breaks ties in favor of original ordering (stable sort).
        candidates.append((_edge_score(bet), -idx, bet))

    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)

    # Running exposure buckets
    total_exp = 0.0
    by_game: dict[str, float] = defaultdict(float)
    by_player: dict[str, float] = defaultdict(float)
    by_prop: dict[str, float] = defaultdict(float)
    by_correlated: dict[tuple[str, str], float] = defaultdict(float)

    for _edge, _idx, bet in candidates:
        size = _bet_size_fraction(bet)
        game, player, prop = _bet_key(bet)

        rejection = None
        if total_exp + size > max_total:
            rejection = f'total_exposure_cap ({max_total:.0%})'
        elif by_game[game] + size > max_game:
            rejection = f'game_exposure_cap ({max_game:.0%}) on {game}'
        elif by_player[player] + size > max_player:
            rejection = f'player_exposure_cap ({max_player:.0%}) on {player}'
        elif by_prop[prop] + size > max_prop_type:
            rejection = f'prop_type_exposure_cap ({max_prop_type:.0%}) on {prop}'
        elif by_correlated[(game, prop)] + size > max_correlated:
            rejection = (
                f'correlated_exposure_cap ({max_correlated:.0%}) on '
                f'{game}/{prop}'
            )

        if rejection:
            bet['cap_admitted'] = False
            bet['cap_rejected_reason'] = rejection
            summary['rejected'] += 1
            summary['rejections_by_reason'][rejection.split(' ')[0]] += 1
            continue

        bet['cap_admitted'] = True
        total_exp += size
        by_game[game] += size
        by_player[player] += size
        by_prop[prop] += size
        by_correlated[(game, prop)] += size
        summary['admitted'] += 1

    summary['total_exposure'] = round(total_exp, 4)
    summary['rejections_by_reason'] = dict(summary['rejections_by_reason'])
    return summary


__all__ = ['apply_exposure_caps']
