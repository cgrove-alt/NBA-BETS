#!/usr/bin/env python3
"""
Out-of-Sample (OOS) Backtest: Train on historical seasons, test on holdout.

Trains models on specified seasons (e.g., 2020-2022), saves them to a
separate directory (models/holdout/), then runs the profitability backtest
against a holdout season (e.g., 2023-24) using those models.

This gives honest, unbiased performance estimates — unlike the in-sample
backtest where models were trained on the same data they were tested on.

Usage:
    PYTHONPATH=. python3 nba_models/backtesting/oos_backtest.py
    PYTHONPATH=. python3 nba_models/backtesting/oos_backtest.py --skip-retrain
    PYTHONPATH=. python3 nba_models/backtesting/oos_backtest.py --train-seasons 2020-21 2021-22 2022-23
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.environ.get(
    "NBA_BETS_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "nba_models", "training"))
sys.path.insert(0, ROOT)

logger = logging.getLogger(__name__)

# Module-level progress tracker (read by API status endpoint)
_oos_progress: dict = {}

HOLDOUT_DIR = Path(ROOT) / "models" / "holdout"


def train_holdout_models(train_seasons: list[str]) -> dict:
    """Phase A: Train models on specified seasons, save to holdout directory.

    Returns training metrics dict.
    """
    from train_from_csv import (
        build_team_id_map,
        _build_team_metadata,
        load_team_games,
        load_player_stats,
    )
    from train_complete_balldontlie import (
        initialize_league_averages,
        process_games_for_training,
        train_all_models,
    )

    _oos_progress.update({
        "phase": "training",
        "step": "loading_data",
        "train_seasons": train_seasons,
    })

    logger.info("OOS Phase A: Training on seasons %s", train_seasons)
    logger.info("  Holdout model dir: %s", HOLDOUT_DIR)

    # Load training data
    team_id_map = build_team_id_map()
    team_meta = _build_team_metadata()
    games = load_team_games(train_seasons, team_id_map, team_meta)
    game_ids = {g["id"] for g in games}
    player_stats_by_game = load_player_stats(game_ids, train_seasons, team_id_map)

    total_records = sum(len(v) for v in player_stats_by_game.values())
    logger.info("  Training data: %d games, %d player-game records", len(games), total_records)

    _oos_progress["step"] = "building_features"

    # Initialize league averages and process features
    tracker_games = [
        {
            "game_date": g["date"],
            "home_score": g["home_team_score"],
            "away_score": g["visitor_team_score"],
        }
        for g in games
    ]
    initialize_league_averages(tracker_games)
    team_data, player_data = process_games_for_training(games, player_stats_by_game)

    logger.info("  Processed: %d team samples, %d player samples", len(team_data), len(player_data))

    _oos_progress["step"] = "training_models"

    # Train all models, saving to holdout directory
    results = train_all_models(
        team_data=team_data,
        player_data=player_data,
        use_time_decay=True,
        time_decay_half_life=180,
        use_ensemble_props=True,
        model_dir=HOLDOUT_DIR,
    )

    _oos_progress["step"] = "training_complete"
    logger.info("OOS Phase A complete: models saved to %s", HOLDOUT_DIR)
    return results


def run_oos_backtest(
    train_seasons: list[str],
    test_season: str = "2023-24",
    skip_retrain: bool = False,
    bankroll: float = 1000.0,
) -> dict | None:
    """Run the full OOS backtest: train (Phase A) + simulate (Phase B).

    Args:
        train_seasons: Seasons to train on (e.g., ["2020-21", "2021-22", "2022-23"])
        test_season: Season to test on (holdout)
        skip_retrain: If True, skip training and use existing holdout models
        bankroll: Starting bankroll for simulation

    Returns:
        Backtest results dict, or None on failure.
    """
    from nba_models.backtesting.profitability_backtest import run_backtest, PROP_TYPES

    _oos_progress.update({
        "type": "oos",
        "train_seasons": train_seasons,
        "test_season": test_season,
        "started": time.time(),
    })

    # Phase A: Train holdout models
    training_metrics = None
    if skip_retrain and HOLDOUT_DIR.exists() and any(HOLDOUT_DIR.glob("*.pkl")):
        logger.info("Skipping retrain — using existing holdout models in %s", HOLDOUT_DIR)
        _oos_progress["phase"] = "skipped_training"
    else:
        t0 = time.time()
        training_metrics = train_holdout_models(train_seasons)
        train_time = time.time() - t0
        logger.info("Training took %.1f seconds", train_time)
        _oos_progress["train_time_sec"] = round(train_time)

    # Phase B: Run simulation using holdout models
    _oos_progress.update({"phase": "simulation", "step": "starting_backtest"})

    args = argparse.Namespace(bankroll=bankroll, season=test_season)
    results = run_backtest(args, model_dir=HOLDOUT_DIR)

    if results:
        # Tag results as OOS
        results["backtest_type"] = "out_of_sample"
        results["train_seasons"] = train_seasons
        results["test_season"] = test_season
        if training_metrics:
            results["training_metrics_summary"] = {
                k: v for k, v in training_metrics.items()
                if isinstance(v, (int, float, str))
            }
        # Update caveats for OOS
        results["caveats"] = [
            f"Models trained on {', '.join(train_seasons)} (out-of-sample). "
            f"Test season: {test_season}. Results are unbiased estimates.",
            "Prop lines are simulated (70% season avg + 30% recent avg, "
            "rounded to 0.5). Real lines may differ.",
        ]

    _oos_progress.update({"phase": "complete"})
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="NBA-BETS Out-of-Sample Backtest")
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        default=["2020-21", "2021-22", "2022-23"],
        help="Seasons to train on (default: 2020-21 2021-22 2022-23)",
    )
    parser.add_argument(
        "--test-season",
        default="2023-24",
        help="Season to test on (default: 2023-24)",
    )
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Skip training if holdout models already exist",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll (default: 1000)",
    )
    args = parser.parse_args()

    results = run_oos_backtest(
        train_seasons=args.train_seasons,
        test_season=args.test_season,
        skip_retrain=args.skip_retrain,
        bankroll=args.bankroll,
    )

    if results and "error" not in results:
        logger.info("OOS backtest completed successfully!")
        return 0

    logger.error("OOS backtest failed!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
