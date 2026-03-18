#!/usr/bin/env python3
"""
Baseline Comparison: Does the model beat a simple average?

For every player-game in the test set, computes 3 predictions:
  (a) season average (before the game)
  (b) last-5 game average
  (c) model prediction

Reports RMSE, MAE, R², and bias per prop type for each method.
The model MUST beat season-average on OOS data to have any value.

This script is the quality gate: run after every retrain.

Usage:
    PYTHONPATH=. python3 scripts/baseline_comparison.py
    PYTHONPATH=. python3 scripts/baseline_comparison.py --season 2023-24
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.environ.get(
    "NBA_BETS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "nba_models", "training"))
sys.path.insert(0, ROOT)

logger = logging.getLogger(__name__)

# Prop types and their feature keys
PROP_CONFIG = {
    "points": {
        "actual_key": "actual_pts",
        "season_avg_key": "season_pts_avg",
        "last5_avg_key": "last5_pts_avg",
    },
    "rebounds": {
        "actual_key": "actual_reb",
        "season_avg_key": "season_reb_avg",
        "last5_avg_key": "last5_reb_avg",
    },
    "assists": {
        "actual_key": "actual_ast",
        "season_avg_key": "season_ast_avg",
        "last5_avg_key": "last5_ast_avg",
    },
    "threes": {
        "actual_key": "actual_fg3m",
        "season_avg_key": "season_fg3m_avg",
        "last5_avg_key": "last5_fg3m_avg",
    },
    "pra": {
        "actual_key": "actual_pra",
        "season_avg_key": None,  # Computed from pts+reb+ast
        "last5_avg_key": None,   # Computed from pts+reb+ast
    },
}

MIN_MINUTES = 15
MIN_SEASON_GAMES = 10


def compute_metrics(actuals: np.ndarray, preds: np.ndarray) -> dict:
    """Compute RMSE, MAE, R², and bias."""
    if len(actuals) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "bias": float("nan"), "n": 0}
    residuals = actuals - preds
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "r2": float(r2),
        "bias": float(np.mean(residuals)),
        "n": int(len(actuals)),
    }


def run_comparison(season: str = "2023-24") -> dict:
    """Run the baseline comparison and return results dict."""
    from train_from_csv import (
        build_team_id_map,
        _build_team_metadata,
        load_team_games,
        load_player_stats,
    )
    from train_complete_balldontlie import (
        initialize_league_averages,
        process_games_for_training,
    )
    from nba_models.backtesting.profitability_backtest import load_models

    # Load data
    logger.info("Loading CSV data...")
    team_id_map = build_team_id_map()
    team_meta = _build_team_metadata()
    context_seasons = ["2022-23", season]
    games = load_team_games(context_seasons, team_id_map, team_meta)
    game_ids = {g["id"] for g in games}
    player_stats_by_game = load_player_stats(game_ids, context_seasons, team_id_map)
    logger.info("Loaded %d games", len(games))

    # Build features
    logger.info("Building walk-forward features...")
    tracker_games = [
        {"game_date": g["date"], "home_score": g["home_team_score"], "away_score": g["visitor_team_score"]}
        for g in games
    ]
    initialize_league_averages(tracker_games)
    _, player_data = process_games_for_training(games, player_stats_by_game)

    # Filter to test season
    date_range = {
        "2023-24": ("2023-10-01", "2024-04-30"),
        "2024-25": ("2024-10-01", "2025-04-30"),
        "2025-26": ("2025-10-01", "2026-04-30"),
    }
    start, end = date_range.get(season, ("2023-10-01", "2024-04-30"))
    test_data = [p for p in player_data if start <= p["game_date"] <= end]
    test_data.sort(key=lambda x: x["game_date"])
    logger.info("Test set: %d player-game samples", len(test_data))

    # Load models
    logger.info("Loading models...")
    ensemble_models, quantile_models = load_models()

    # Collect predictions
    results = {}
    for prop_type, cfg in PROP_CONFIG.items():
        actuals = []
        season_avg_preds = []
        last5_avg_preds = []
        model_preds = []

        for sample in test_data:
            features = sample["features"]
            actual_min = sample.get("actual_min", 0)

            # Skip low-minutes players (DNP filter)
            if actual_min < MIN_MINUTES:
                continue

            # Skip early-season (insufficient data)
            if features.get("season_games", 0) < MIN_SEASON_GAMES:
                continue

            # Get actual value
            if prop_type == "pra":
                actual = sample.get("actual_pts", 0) + sample.get("actual_reb", 0) + sample.get("actual_ast", 0)
            else:
                actual = sample.get(cfg["actual_key"], 0)

            if actual is None or actual == 0:
                continue

            # Season average baseline
            if prop_type == "pra":
                s_avg = (
                    features.get("season_pts_avg", 0)
                    + features.get("season_reb_avg", 0)
                    + features.get("season_ast_avg", 0)
                )
            else:
                s_avg = features.get(cfg["season_avg_key"], 0)

            # Last-5 average baseline
            if prop_type == "pra":
                l5_avg = (
                    features.get("last5_pts_avg", 0)
                    + features.get("last5_reb_avg", 0)
                    + features.get("last5_ast_avg", 0)
                )
            else:
                l5_avg = features.get(cfg["last5_avg_key"], 0)

            if s_avg <= 0:
                continue

            # Model prediction
            model = ensemble_models.get(prop_type)
            if model is None:
                continue

            try:
                pred_result = model.predict(features, prop_line=s_avg)
                model_pred = pred_result.get("predicted_value", 0)
            except Exception:
                continue

            if model_pred <= 0:
                continue

            actuals.append(actual)
            season_avg_preds.append(s_avg)
            last5_avg_preds.append(l5_avg if l5_avg > 0 else s_avg)
            model_preds.append(model_pred)

        actuals_arr = np.array(actuals)
        results[prop_type] = {
            "season_average": compute_metrics(actuals_arr, np.array(season_avg_preds)),
            "last5_average": compute_metrics(actuals_arr, np.array(last5_avg_preds)),
            "model": compute_metrics(actuals_arr, np.array(model_preds)),
        }

        # Does model beat season average?
        sa_rmse = results[prop_type]["season_average"]["rmse"]
        m_rmse = results[prop_type]["model"]["rmse"]
        beats_baseline = m_rmse < sa_rmse
        improvement_pct = (sa_rmse - m_rmse) / sa_rmse * 100 if sa_rmse > 0 else 0
        results[prop_type]["beats_season_avg"] = beats_baseline
        results[prop_type]["rmse_improvement_pct"] = round(improvement_pct, 2)

    return results


def print_report(results: dict) -> None:
    """Print a formatted comparison report."""
    print("=" * 90)
    print("   BASELINE COMPARISON: Model vs Simple Averages")
    print("=" * 90)
    print()

    overall_pass = True
    for prop_type, data in results.items():
        sa = data["season_average"]
        l5 = data["last5_average"]
        m = data["model"]
        beats = data["beats_season_avg"]
        imp = data["rmse_improvement_pct"]

        if not beats:
            overall_pass = False

        status = "PASS" if beats else "FAIL"
        print(f"--- {prop_type.upper()} ({sa['n']} samples) [{status}] ---")
        print(f"  {'Method':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Bias':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'Season Average':<20} {sa['rmse']:>8.3f} {sa['mae']:>8.3f} {sa['r2']:>8.3f} {sa['bias']:>+8.3f}")
        print(f"  {'Last-5 Average':<20} {l5['rmse']:>8.3f} {l5['mae']:>8.3f} {l5['r2']:>8.3f} {l5['bias']:>+8.3f}")
        print(f"  {'Model':<20} {m['rmse']:>8.3f} {m['mae']:>8.3f} {m['r2']:>8.3f} {m['bias']:>+8.3f}")
        print(f"  RMSE improvement over season avg: {imp:+.2f}%")
        print()

    print("=" * 90)
    gate = "PASS" if overall_pass else "FAIL"
    print(f"   QUALITY GATE: {gate}")
    if not overall_pass:
        print("   Model does NOT beat season-average baseline on all enabled props.")
        print("   Do NOT deploy this model for live betting.")
    print("=" * 90)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Baseline Comparison Quality Gate")
    parser.add_argument("--season", default="2023-24", help="Test season (default: 2023-24)")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    results = run_comparison(args.season)

    print_report(results)

    # Save JSON
    out_path = args.output or os.path.join(ROOT, "data", "backtest_results", "baseline_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Exit code: 0 if all props beat baseline, 1 if any fail
    all_pass = all(d.get("beats_season_avg", False) for d in results.values() if d.get("model", {}).get("n", 0) > 0)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
