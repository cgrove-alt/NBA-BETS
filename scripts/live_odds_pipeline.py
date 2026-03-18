#!/usr/bin/env python3
"""
Live Odds Pipeline (Fix 6.2)

Real-time odds integration for NBA player props:
1. Fetch props 2-3 hours before game from Balldontlie API / The Odds API
2. Run model predictions for each available prop
3. Compare to devigged market prices
4. Only bet when edge > vig (true EV > 0)

This script is designed to run on a schedule (e.g., cron at 4pm ET for 7pm games).

Usage:
    PYTHONPATH=. python3 scripts/live_odds_pipeline.py
    PYTHONPATH=. python3 scripts/live_odds_pipeline.py --date 2026-03-18
    PYTHONPATH=. python3 scripts/live_odds_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = os.environ.get(
    "NBA_BETS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "nba_models", "training"))

ET = ZoneInfo('America/New_York')
logger = logging.getLogger(__name__)

# Minimum true EV to recommend a bet (after devigging)
MIN_TRUE_EV = 0.03  # 3%


def fetch_todays_props(api, target_date: str) -> list[dict]:
    """Fetch all player props for today's games from the API.

    Returns list of dicts with: game_id, player_name, player_id,
    prop_type, line, over_odds, under_odds, sportsbook.
    """
    from nba_models.inference.daily_predictions import get_player_props_for_game

    # Get today's games
    games = api.get_games(dates=[target_date])
    if not games:
        logger.info("No games found for %s", target_date)
        return []

    all_props = []
    for game in games:
        game_id = game.get('id')
        if not game_id:
            continue

        home = game.get('home_team', {}).get('abbreviation', '?')
        away = game.get('visitor_team', {}).get('abbreviation', '?')
        game_label = f"{away}@{home}"

        try:
            props = get_player_props_for_game(api, game_id)
            for player_id, prop_data in props.items():
                for prop_entry in prop_data if isinstance(prop_data, list) else [prop_data]:
                    all_props.append({
                        'game_id': game_id,
                        'game_label': game_label,
                        'player_id': player_id,
                        'player_name': prop_entry.get('player_name', f'Player {player_id}'),
                        'prop_type': prop_entry.get('stat_type', 'points'),
                        'line': prop_entry.get('line', 0),
                        'over_odds': prop_entry.get('over_odds', -110),
                        'under_odds': prop_entry.get('under_odds', -110),
                        'sportsbook': prop_entry.get('sportsbook', 'unknown'),
                    })
        except Exception as e:
            logger.warning("Failed to fetch props for game %s: %s", game_id, e)

    logger.info("Fetched %d player props across %d games", len(all_props), len(games))
    return all_props


def evaluate_props(props: list[dict], models: dict) -> list[dict]:
    """Run model predictions against live odds and compute true EV.

    Returns list of evaluated props with model predictions and EV.
    """
    from nba_betting.prediction_pipeline import evaluate_bet
    from nba_betting.constants import DISABLED_PROPS
    from nba_models.inference.daily_predictions import predict_player_prop

    evaluated = []

    for prop in props:
        prop_type = prop['prop_type'].lower()

        # Skip disabled props
        if prop_type in DISABLED_PROPS:
            continue

        line = prop['line']
        if line <= 0:
            continue

        # Get model prediction
        try:
            pred = predict_player_prop(
                player_name=prop['player_name'],
                player_id=prop['player_id'],
                prop_type=prop_type,
                line=line,
                opponent='',
                opponent_id=0,
                models=models,
                use_api_features=True,
                american_odds=prop['over_odds'],
                under_odds=prop['under_odds'],
            )

            if pred is None or pred.get('predicted_value') is None:
                continue

            # Evaluate via pipeline with real odds
            ev_result = evaluate_bet(
                prop_type=prop_type,
                predicted=pred['predicted_value'],
                line=line,
                raw_confidence=pred.get('over_prob'),
                games_played=None,
                bankroll=1000.0,
                over_odds=prop['over_odds'],
                under_odds=prop['under_odds'],
                pre_calibrated=True,
            )

            result = {
                **prop,
                'predicted_value': pred['predicted_value'],
                'over_prob': pred.get('over_prob', 0.5),
                'direction': ev_result['direction'],
                'edge': ev_result['edge'],
                'confidence': ev_result['confidence'],
                'true_ev': ev_result.get('true_ev'),
                'ev_edge': ev_result.get('ev_edge'),
                'market_implied_prob': ev_result.get('market_implied_prob'),
                'should_bet': ev_result['should_bet'],
                'bet_size': ev_result['bet_size'],
                'tier': ev_result['tier'],
                'reason': ev_result['reason'],
            }
            evaluated.append(result)

        except Exception as e:
            logger.debug("Failed to evaluate %s %s: %s", prop['player_name'], prop_type, e)

    return evaluated


def filter_actionable(evaluated: list[dict]) -> list[dict]:
    """Filter to only actionable bets: positive EV and should_bet=True."""
    actionable = []
    for ev in evaluated:
        true_ev = ev.get('true_ev')
        if ev['should_bet'] and true_ev is not None and true_ev >= MIN_TRUE_EV:
            actionable.append(ev)

    # Sort by true_ev descending
    actionable.sort(key=lambda x: x.get('true_ev', 0), reverse=True)
    return actionable


def print_results(actionable: list[dict], evaluated: list[dict]) -> None:
    """Print formatted results."""
    print("=" * 80)
    print("   LIVE ODDS PIPELINE — Actionable Bets")
    print("=" * 80)
    print(f"  Props evaluated:  {len(evaluated)}")
    print(f"  Actionable bets:  {len(actionable)}")
    print()

    if not actionable:
        print("  No actionable bets found. All edges below threshold.")
        print("=" * 80)
        return

    print(f"  {'Player':<20} {'Prop':<8} {'Line':>5} {'Pred':>6} {'Dir':<6} "
          f"{'EV':>6} {'Edge':>6} {'Tier':<8} {'Book':<10}")
    print(f"  {'-'*20} {'-'*8} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*10}")

    for bet in actionable:
        ev_pct = bet.get('true_ev', 0) * 100
        print(
            f"  {bet['player_name'][:20]:<20} {bet['prop_type']:<8} "
            f"{bet['line']:>5.1f} {bet['predicted_value']:>6.1f} "
            f"{bet['direction']:<6} {ev_pct:>+5.1f}% "
            f"{bet['edge']:>5.2f} {bet['tier']:<8} {bet['sportsbook'][:10]:<10}"
        )

    total_ev = sum(b.get('true_ev', 0) for b in actionable) / len(actionable) * 100
    print()
    print(f"  Average true EV: {total_ev:+.1f}%")
    print("=" * 80)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Live Odds Pipeline")
    parser.add_argument("--date", default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Don't record bets, just evaluate")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    target_date = args.date or datetime.now(ET).strftime('%Y-%m-%d')
    logger.info("Running live odds pipeline for %s", target_date)

    # Initialize API
    from balldontlie_api import BalldontlieAPI
    api = BalldontlieAPI()

    # Load models
    from nba_models.inference.daily_predictions import load_models
    models = load_models()

    # Fetch props
    props = fetch_todays_props(api, target_date)
    if not props:
        print("No props available. Games may not have started or API may be down.")
        return 0

    # Evaluate
    evaluated = evaluate_props(props, models)
    actionable = filter_actionable(evaluated)

    # Print
    print_results(actionable, evaluated)

    # Record to CLV tracker (unless dry run)
    if not args.dry_run and actionable:
        try:
            from nba_betting.edge.clv_bridge import record_predictions_as_bets
            formatted = []
            for bet in actionable:
                formatted.append({
                    'player': bet['player_name'],
                    'stat': bet['prop_type'],
                    'line': bet['line'],
                    'pick': bet['direction'].upper(),
                    'over_prob': bet['over_prob'],
                    'american_odds': bet['over_odds'] if bet['direction'] == 'over' else bet['under_odds'],
                    'over_odds': bet['over_odds'],
                    'under_odds': bet['under_odds'],
                    'signal': 'BET',
                    'game': bet['game_label'],
                    'ev_per_dollar': bet.get('true_ev', 0),
                })
            count = record_predictions_as_bets(formatted, target_date)
            logger.info("Recorded %d bets to CLV tracker", count)
        except Exception as e:
            logger.warning("CLV recording failed: %s", e)

    # Save JSON
    out_path = args.output or os.path.join(ROOT, "data", "live_odds", f"{target_date}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "date": target_date,
            "props_fetched": len(props),
            "props_evaluated": len(evaluated),
            "actionable_bets": len(actionable),
            "bets": actionable,
        }, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
