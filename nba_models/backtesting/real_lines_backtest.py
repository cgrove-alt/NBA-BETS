#!/usr/bin/env python3
"""Real-Lines Profitability Backtest

Uses actual sportsbook lines from The Odds API and real player outcomes
from the BallDontLie API to produce honest P&L results for the 2024-25
NBA season.

Key differences from the simulated-lines backtest:
  - Real prop lines from DraftKings/FanDuel (not simulated from averages)
  - Real American odds per prop (not flat -110)
  - Real player outcomes from BDL API for settlement
  - True EV calculation using devigged market probabilities

Walk-forward guarantee: features for each game are computed using ONLY
data from games played BEFORE that date.

Data requirements (run these fetchers first):
  python nba_models/backtesting/fetch_historical_lines.py --season 2024-25 --resume
  python nba_models/backtesting/fetch_player_stats.py --season 2024 --force

Usage:
    PYTHONPATH=. python3 nba_models/backtesting/real_lines_backtest.py
    PYTHONPATH=. python3 nba_models/backtesting/real_lines_backtest.py --bankroll 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
_progress: dict = {}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INITIAL_BANKROLL = 1000.0
PROP_TYPES = ["points", "rebounds", "assists", "pra"]

# Default season config — overridden by --season flag
TEST_SEASON_LABEL = "2024-25"
TEST_SEASON_INT = 2024

# Season configs: test_season → {context_seasons, bdl_stats_file, test_date_range}
SEASON_CONFIGS = {
    "2023-24": {
        "test_int": 2023,
        "context_seasons": ["2022-23", "2023-24"],
        "test_start": "2023-10-01",
        "test_end": "2024-06-30",
        "oos_train_seasons": ["2021-22", "2022-23"],
        "oos_train_date_end": "2023-06-30",
    },
    "2024-25": {
        "test_int": 2024,
        "context_seasons": ["2023-24", "2024-25"],
        "test_start": "2024-10-01",
        "test_end": "2025-06-30",
        "oos_train_seasons": ["2021-22", "2022-23", "2023-24"],
        "oos_train_date_end": "2024-06-30",
    },
}

CONTEXT_SEASONS = ["2023-24", "2024-25"]  # Default, overridden at runtime
LINES_DIR = os.path.join(ROOT, "data", "historical_lines")
OUTPUT_DIR = os.path.join(ROOT, "data", "backtest_results")
MIN_MINUTES = 15
MIN_SEASON_GAMES = 10


# ---------------------------------------------------------------------------
# Data loading: BDL player stats → pipeline format
# ---------------------------------------------------------------------------
def load_bdl_player_stats(
    csv_games: list[dict],
    team_meta: dict,
    team_id_map: dict,
    season_int: int | None = None,
) -> dict[int, list[dict]]:
    """Load BDL player stats and convert to process_games_for_training format.

    Reads the enhanced BDL cache (player_stats_{season}.json + _meta.json),
    maps BDL game IDs to NBA.com game IDs by matching date + teams,
    and converts player stats to the pipeline format.

    Args:
        csv_games: Games loaded from CSV (with NBA.com IDs).
        team_meta: Team metadata from _build_team_metadata().
        team_id_map: NBA.com team_id → compact team_id.
        season_int: Season start year (e.g., 2023 for 2023-24). Defaults to TEST_SEASON_INT.

    Returns:
        player_stats_by_game: dict mapping NBA.com game_id → [player stat dicts]
    """
    _season = season_int if season_int is not None else TEST_SEASON_INT
    stats_path = os.path.join(LINES_DIR, f"player_stats_{_season}.json")
    meta_path = os.path.join(LINES_DIR, f"player_stats_{_season}_meta.json")

    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"BDL stats cache not found: {stats_path}\n"
            f"Run: BALLDONTLIE_API_KEY=<key> python nba_models/backtesting/"
            f"fetch_player_stats.py --season {_season} --force"
        )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"BDL metadata not found: {meta_path}\n"
            f"Re-run fetch_player_stats.py with --force to generate metadata."
        )

    logger.info("Loading BDL player stats from %s", stats_path)
    with open(stats_path) as f:
        bdl_stats = json.load(f)  # str(BDL game_id) → [player stats]
    with open(meta_path) as f:
        bdl_meta = json.load(f)  # str(BDL game_id) → {date, teams, ...}

    total_records = sum(len(v) for v in bdl_stats.values())
    logger.info(
        "BDL cache: %d player-game records across %d games",
        total_records,
        len(bdl_stats),
    )

    # 1. Build BDL team_id → abbreviation mapping from player stats
    bdl_team_map: dict[int, str] = {}
    for game_players in bdl_stats.values():
        for p in game_players:
            tid = p.get("team_id")
            abbrev = p.get("team_abbreviation", "")
            if tid and abbrev:
                bdl_team_map[tid] = abbrev

    logger.info("BDL team map: %d teams — %s", len(bdl_team_map), sorted(bdl_team_map.values()))

    # 2. Build abbreviation → compact team_id from team_meta
    abbrev_to_compact: dict[str, int] = {}
    for nba_team_id, meta in team_meta.items():
        abbrev = meta.get("abbreviation", "")
        compact_id = team_id_map.get(nba_team_id)
        if abbrev and compact_id:
            abbrev_to_compact[abbrev] = compact_id

    logger.info("Abbreviation → compact map: %d teams", len(abbrev_to_compact))

    # 3. Build (date, home_abbrev, away_abbrev) → NBA.com game_id from CSV games
    csv_game_lookup: dict[tuple[str, str, str], int] = {}
    for g in csv_games:
        date = g["date"]
        home_abbrev = g["home_team"]["abbreviation"]
        away_abbrev = g["visitor_team"]["abbreviation"]
        csv_game_lookup[(date, home_abbrev, away_abbrev)] = g["id"]

    # 4. Map BDL game_id → NBA.com game_id
    bdl_to_nba: dict[str, int] = {}
    unmatched = 0

    for bdl_gid, meta in bdl_meta.items():
        game_date = str(meta.get("date", ""))[:10]
        home_bdl_tid = meta.get("home_team_id")
        away_bdl_tid = meta.get("visitor_team_id")

        home_abbrev = bdl_team_map.get(home_bdl_tid, "")
        away_abbrev = bdl_team_map.get(away_bdl_tid, "")

        nba_gid = csv_game_lookup.get((game_date, home_abbrev, away_abbrev))
        if nba_gid is None:
            # Try reverse (BDL might swap home/away)
            nba_gid = csv_game_lookup.get((game_date, away_abbrev, home_abbrev))

        if nba_gid is not None:
            bdl_to_nba[bdl_gid] = nba_gid
        else:
            unmatched += 1

    logger.info(
        "Game ID mapping: %d matched, %d unmatched",
        len(bdl_to_nba),
        unmatched,
    )

    # 5. Convert BDL player stats to pipeline format
    player_stats_by_game: dict[int, list[dict]] = defaultdict(list)
    converted = 0
    skipped_no_map = 0

    for bdl_gid, players in bdl_stats.items():
        nba_gid = bdl_to_nba.get(bdl_gid)
        if nba_gid is None:
            skipped_no_map += len(players)
            continue

        for p in players:
            abbrev = p.get("team_abbreviation", "")
            compact_tid = abbrev_to_compact.get(abbrev, 0)

            first = p.get("first_name", "")
            last = p.get("last_name", "")
            if not first and not last:
                # Fall back to splitting player_name
                parts = p.get("player_name", "").split(" ", 1)
                first = parts[0] if parts else ""
                last = parts[1] if len(parts) > 1 else ""

            stat_dict = {
                "player": {
                    "id": p.get("player_id", 0),
                    "first_name": first,
                    "last_name": last,
                    "position": p.get("position", ""),
                },
                "team": {
                    "id": compact_tid,
                    "abbreviation": abbrev,
                },
                "game": {
                    "id": nba_gid,
                },
                "min": p.get("min", "0"),
                "pts": p.get("pts", 0) or 0,
                "reb": p.get("reb", 0) or 0,
                "ast": p.get("ast", 0) or 0,
                "stl": p.get("stl", 0) or 0,
                "blk": p.get("blk", 0) or 0,
                "turnover": p.get("turnover", 0) or 0,
                "pf": p.get("pf", 0) or 0,
                "fgm": p.get("fgm", 0) or 0,
                "fga": p.get("fga", 0) or 0,
                "fg3m": p.get("fg3m", 0) or 0,
                "fg3a": p.get("fg3a", 0) or 0,
                "ftm": p.get("ftm", 0) or 0,
                "fta": p.get("fta", 0) or 0,
                "oreb": p.get("oreb", 0) or 0,
                "dreb": p.get("dreb", 0) or 0,
                "fg_pct": p.get("fg_pct", 0.0) or 0.0,
                "fg3_pct": p.get("fg3_pct", 0.0) or 0.0,
                "ft_pct": p.get("ft_pct", 0.0) or 0.0,
            }
            player_stats_by_game[nba_gid].append(stat_dict)
            converted += 1

    logger.info(
        "Converted %d BDL records to pipeline format (%d skipped, no game match)",
        converted,
        skipped_no_map,
    )
    return dict(player_stats_by_game)


# ---------------------------------------------------------------------------
# Data loading: historical sportsbook lines
# ---------------------------------------------------------------------------
def load_historical_lines() -> dict[str, list[dict]]:
    """Load all cached historical lines, indexed by game date.

    Returns:
        Dict mapping date string → list of game dicts, each containing
        player_props with real sportsbook lines and odds.
    """
    lines_by_date: dict[str, list[dict]] = {}
    lines_dir = Path(LINES_DIR)

    json_files = sorted(lines_dir.glob("20*.json"))
    total_games = 0
    total_props = 0

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        date = data.get("date", jf.stem)
        games = data.get("games", [])
        if games:
            lines_by_date[date] = games
            total_games += len(games)
            total_props += sum(len(g.get("player_props", [])) for g in games)

    logger.info(
        "Loaded historical lines: %d dates, %d games, %d props",
        len(lines_by_date),
        total_games,
        total_props,
    )
    return lines_by_date


# ---------------------------------------------------------------------------
# Model loading (reuse from profitability_backtest)
# ---------------------------------------------------------------------------
def load_models(model_dir: Path | None = None) -> tuple[dict, dict]:
    """Load ensemble + quantile models for each prop type.

    Returns:
        (ensemble_models, quantile_models) — both keyed by prop type.
    """
    import pickle
    from train_complete_balldontlie import PropEnsembleModel, QuantilePropModel

    ensemble_models: dict = {}
    quantile_models: dict = {}
    mdir = model_dir or Path(ROOT) / "models"

    for prop_type in PROP_TYPES:
        # Ensemble model
        epath = mdir / f"player_{prop_type}_ensemble.pkl"
        if epath.exists():
            try:
                model = PropEnsembleModel.load(epath)
                ensemble_models[prop_type] = model
                logger.info("Loaded ensemble %s", epath.name)
            except Exception as exc:
                logger.warning("Failed to load ensemble %s: %s", epath, exc)

        # Quantile model
        qpath = mdir / f"player_{prop_type}_quantile.pkl"
        if qpath.exists():
            try:
                qmodel = QuantilePropModel.load(qpath)
                quantile_models[prop_type] = qmodel
                logger.info("Loaded quantile %s", qpath.name)
            except Exception:
                try:
                    with open(qpath, "rb") as f:
                        data = pickle.load(f)
                    if isinstance(data, dict) and "model" in data:
                        quantile_models[prop_type] = data["model"]
                        logger.info("Loaded quantile (wrapped) %s", qpath.name)
                except Exception as exc2:
                    logger.warning("Failed to load quantile %s: %s", qpath, exc2)

    return ensemble_models, quantile_models


# ---------------------------------------------------------------------------
# Player name matching
# ---------------------------------------------------------------------------
def normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching."""
    return (
        name.lower()
        .strip()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("  ", " ")
    )


def build_player_feature_index(
    player_data: list[dict],
) -> dict[tuple[str, str], dict]:
    """Index player features by (normalized_name, game_date).

    Args:
        player_data: Output of process_games_for_training (player samples).

    Returns:
        Dict mapping (normalized_name, date) → player sample dict.
    """
    index: dict[tuple[str, str], dict] = {}
    for sample in player_data:
        name = normalize_name(sample.get("player_name", ""))
        date = sample["game_date"]
        index[(name, date)] = sample
    return index


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------
def run_backtest(args: argparse.Namespace, model_dir: Path | None = None, season_cfg: dict | None = None) -> dict | None:
    """Execute the real-lines walk-forward backtest.

    Args:
        args: Namespace with bankroll, etc.
        model_dir: Custom model directory. If None, uses default models/.
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
        MinutesPredictionModel,
    )
    from nba_betting.prediction_pipeline import evaluate_bet

    # ── 1. Load team game data ─────────────────────────────────────────
    logger.info("Step 1/6: Loading team game data ...")
    team_id_map = build_team_id_map()
    team_meta = _build_team_metadata()

    games = load_team_games(CONTEXT_SEASONS, team_id_map, team_meta)
    logger.info("Loaded %d team games for %s", len(games), CONTEXT_SEASONS)

    # ── 2. Load player stats (BDL for 2024-25) ────────────────────────
    logger.info("Step 2/6: Loading BDL player stats ...")
    player_stats_by_game = load_bdl_player_stats(
        games, team_meta, team_id_map, season_int=TEST_SEASON_INT
    )

    total_records = sum(len(v) for v in player_stats_by_game.values())
    logger.info(
        "Player stats: %d games, %d records",
        len(player_stats_by_game),
        total_records,
    )

    # ── 3. Build walk-forward features ─────────────────────────────────
    logger.info("Step 3/6: Building walk-forward features ...")
    tracker_games = [
        {
            "game_date": g["date"],
            "home_score": g["home_team_score"],
            "away_score": g["visitor_team_score"],
        }
        for g in games
    ]
    initialize_league_averages(tracker_games)

    _, player_data = process_games_for_training(games, player_stats_by_game)

    # Inject predicted_minutes and prop_line_vs_recent to match training features
    _model_dir = model_dir or Path(ROOT) / "models"
    _min_path = _model_dir / "player_minutes_model.pkl"
    if _min_path.exists():
        logger.info("Injecting predicted_minutes from minutes model...")
        _min_model = MinutesPredictionModel.load(_min_path)
        _X_all = pd.DataFrame([d["features"] for d in player_data])
        try:
            _batch = _min_model.predict_batch(_X_all)
            _min_preds = _batch[0] if isinstance(_batch, tuple) else _batch
            if _min_preds is not None and len(_min_preds) == len(player_data):
                for _i, _d in enumerate(player_data):
                    _d["features"]["predicted_minutes"] = float(_min_preds[_i])
                logger.info("Injected predicted_minutes (mean=%.1f)", _min_preds.mean())
        except Exception as _e:
            logger.warning("Failed to inject predicted_minutes: %s", _e)

    for _d in player_data:
        _f = _d["features"]
        for _sa, _rec in [
            ("season_pts_avg", "recent_pts_avg"),
            ("season_reb_avg", "recent_reb_avg"),
            ("season_ast_avg", "recent_ast_avg"),
        ]:
            _f.setdefault("prop_line_vs_recent", (_f.get(_sa, 0) or 0) - (_f.get(_rec, 0) or 0))

    # Filter to test season using season config
    _test_start = season_cfg["test_start"]
    _test_end = season_cfg["test_end"]
    test_data = [
        p
        for p in player_data
        if _test_start <= p["game_date"] <= _test_end
    ]
    test_data.sort(key=lambda x: x["game_date"])
    logger.info(
        "Feature data: %d total samples, %d in test window",
        len(player_data),
        len(test_data),
    )

    # Build player feature lookup index
    feature_index = build_player_feature_index(test_data)
    logger.info("Player feature index: %d entries", len(feature_index))

    # ── 4. Load historical lines ───────────────────────────────────────
    logger.info("Step 4/6: Loading historical sportsbook lines ...")
    lines_by_date = load_historical_lines()

    # ── 5. Load or train models ────────────────────────────────────────
    oos_mode = getattr(args, "oos", False)

    # Use season config or defaults
    if season_cfg is None:
        season_cfg = SEASON_CONFIGS.get(TEST_SEASON_LABEL, SEASON_CONFIGS["2024-25"])

    if oos_mode:
        # TRUE OUT-OF-SAMPLE: train fresh models on data BEFORE test season.
        logger.info("Step 5/6: OOS MODE — training fresh models for %s ...", TEST_SEASON_LABEL)
        from nba_models.backtesting.oos_walkforward_backtest import train_fresh_models

        # Load CSV player stats for training seasons
        train_seasons = season_cfg["oos_train_seasons"]
        csv_games = load_team_games(train_seasons, team_id_map, team_meta)
        csv_game_ids = {g["id"] for g in csv_games}
        csv_player_stats = load_player_stats(csv_game_ids, train_seasons, team_id_map)

        logger.info("  Training on seasons %s: %d games", train_seasons, len(csv_games))

        # Build training features from CSV data
        _tracker_train = [
            {"game_date": g["date"], "home_score": g["home_team_score"],
             "away_score": g["visitor_team_score"]}
            for g in csv_games
        ]
        initialize_league_averages(_tracker_train)
        _, train_player_data = process_games_for_training(csv_games, csv_player_stats)
        logger.info("  Training samples: %d", len(train_player_data))

        # Train fresh models
        train_date_end = season_cfg["oos_train_date_end"]
        ensemble_models, quantile_models = train_fresh_models(
            train_player_data, train_date_end
        )

        if not ensemble_models:
            logger.error("No models trained — cannot run backtest.")
            return None
        logger.info("  Trained fresh models: %s", list(ensemble_models.keys()))

        # Re-initialize league averages for the full context (train + test)
        initialize_league_averages(tracker_games)
    else:
        logger.info("Step 5/6: Loading pre-trained models ...")
        ensemble_models, quantile_models = load_models(model_dir=model_dir)
        if not ensemble_models and not quantile_models:
            logger.error("No models loaded — cannot run backtest.")
            return None

    logger.info("Ensemble: %s | Quantile: %s",
                list(ensemble_models.keys()), list(quantile_models.keys()))

    # ── 6. Simulate trades using real lines ────────────────────────────
    logger.info("Step 6/6: Running walk-forward P&L simulation with real lines ...")

    bankroll = INITIAL_BANKROLL
    trades: list[dict] = []
    daily_bankroll: dict[str, float] = {}
    diag = defaultdict(int)

    # Naive baseline: bet "over" on every eligible prop where season_avg > line
    naive_trades: list[dict] = []
    naive_bankroll = INITIAL_BANKROLL

    sim_start = _time.time()
    sorted_dates = sorted(lines_by_date.keys())
    _progress.update({
        "phase": "simulation",
        "current": 0,
        "total": len(sorted_dates),
        "trades": 0,
        "started": sim_start,
    })

    for date_idx, game_date in enumerate(sorted_dates):
        games_on_date = lines_by_date[game_date]
        diag["dates_processed"] += 1

        for game in games_on_date:
            props = game.get("player_props", [])
            if not props:
                continue

            diag["games_with_props"] += 1

            for prop in props:
                player_name = prop.get("player_name", "")
                prop_type = prop.get("prop_type", "")
                prop_line = prop.get("line", 0)
                over_odds = prop.get("over_odds")
                under_odds = prop.get("under_odds")

                if not player_name or not prop_type or prop_line <= 0:
                    diag["skip_invalid_prop"] += 1
                    continue

                if prop_type not in PROP_TYPES:
                    diag["skip_unknown_prop_type"] += 1
                    continue

                diag["total_props_seen"] += 1

                # Find matching player features
                norm_name = normalize_name(player_name)
                sample = feature_index.get((norm_name, game_date))

                if sample is None:
                    diag["skip_no_features"] += 1
                    continue

                features = sample["features"]
                games_played = features.get("season_games", 0)
                actual_min = sample.get("actual_min", 0)

                if actual_min < MIN_MINUTES:
                    diag["skip_low_minutes"] += 1
                    continue

                if games_played < MIN_SEASON_GAMES:
                    diag["skip_low_games"] += 1
                    continue

                # Skip bench players: require season avg minutes >= 25
                # Bench player lines are set far below their averages,
                # creating false edge that doesn't reflect genuine prediction skill.
                season_min_avg = features.get("season_min_avg", 0) or 0
                if season_min_avg < 25:
                    diag["skip_bench_player"] += 1
                    continue

                diag[f"eligible_{prop_type}"] += 1

                # --- Naive baseline: bet over if season_avg > line ---
                sa_key_map = {
                    "points": "season_pts_avg",
                    "rebounds": "season_reb_avg",
                    "assists": "season_ast_avg",
                    "pra": None,
                }
                sa_key = sa_key_map.get(prop_type)
                if sa_key:
                    naive_season_avg = features.get(sa_key, 0) or 0
                else:
                    naive_season_avg = (
                        (features.get("season_pts_avg", 0) or 0)
                        + (features.get("season_reb_avg", 0) or 0)
                        + (features.get("season_ast_avg", 0) or 0)
                    )

                actual_map_naive = {
                    "points": sample.get("actual_pts", 0),
                    "rebounds": sample.get("actual_reb", 0),
                    "assists": sample.get("actual_ast", 0),
                    "pra": sample.get("actual_pra", 0),
                }
                naive_actual = actual_map_naive.get(prop_type, 0)

                if naive_season_avg > prop_line and naive_actual != prop_line:
                    naive_won = naive_actual > prop_line
                    naive_bet_size = 30.0
                    if over_odds and over_odds != 0:
                        if over_odds > 0:
                            naive_payout = naive_bet_size * over_odds / 100.0
                        else:
                            naive_payout = naive_bet_size * 100.0 / abs(over_odds)
                    else:
                        naive_payout = naive_bet_size * 100.0 / 110.0
                    naive_pnl = naive_payout if naive_won else -naive_bet_size
                    naive_bankroll += naive_pnl
                    naive_trades.append({
                        "won": naive_won,
                        "pnl": naive_pnl,
                        "bet_size": naive_bet_size,
                        "date": game_date,
                        "prop_type": prop_type,
                    })

                # Model prediction
                if prop_type not in ensemble_models:
                    diag[f"skip_no_model_{prop_type}"] += 1
                    continue

                model = ensemble_models[prop_type]
                try:
                    prediction = model.predict(features, prop_line=prop_line)
                except Exception as exc:
                    diag[f"predict_error_{prop_type}"] += 1
                    if diag[f"predict_error_{prop_type}"] <= 3:
                        logger.warning("Predict error (%s): %s", prop_type, exc)
                    continue

                predicted_value = prediction.get("predicted_value", 0)

                # If model was trained in residual mode, add season avg back (no offset)
                if getattr(model, '_residual_mode', False):
                    sa_col = getattr(model, '_season_avg_col', None)
                    if sa_col and prop_type != 'pra':
                        predicted_value = features.get(sa_col, 0) + predicted_value
                    elif prop_type == 'pra':
                        predicted_value = (
                            features.get('season_pts_avg', 0)
                            + features.get('season_reb_avg', 0)
                            + features.get('season_ast_avg', 0)
                        ) + predicted_value

                # Get calibrated over-probability from quantile model
                over_prob = None
                use_pre_calibrated = False
                if prop_type in quantile_models:
                    try:
                        over_prob = quantile_models[prop_type].predict_over_probability(
                            features, prop_line
                        )
                        use_pre_calibrated = True
                    except Exception:
                        diag[f"quantile_err_{prop_type}"] += 1

                # Pipeline evaluation with REAL odds
                ev_result = evaluate_bet(
                    prop_type=prop_type,
                    predicted=predicted_value,
                    line=prop_line,
                    raw_confidence=over_prob,
                    games_played=games_played,
                    bankroll=INITIAL_BANKROLL,  # Flat sizing
                    over_odds=over_odds,
                    under_odds=under_odds,
                    pre_calibrated=use_pre_calibrated,
                )

                if not ev_result["should_bet"]:
                    reason = ev_result.get("reason", "")
                    if "disabled" in reason.lower():
                        diag[f"reject_disabled_{prop_type}"] += 1
                    elif "games played" in reason.lower():
                        diag[f"reject_sample_{prop_type}"] += 1
                    elif "edge" in reason.lower() and "threshold" in reason.lower():
                        diag[f"reject_edge_{prop_type}"] += 1
                    elif "confidence" in reason.lower():
                        diag[f"reject_confidence_{prop_type}"] += 1
                    elif "ev" in reason.lower():
                        diag[f"reject_ev_{prop_type}"] += 1
                    elif "kelly" in reason.lower():
                        diag[f"reject_kelly_{prop_type}"] += 1
                    else:
                        diag[f"reject_other_{prop_type}"] += 1
                    continue

                # Actual outcome from walk-forward data
                actual_map = {
                    "points": sample.get("actual_pts", 0),
                    "rebounds": sample.get("actual_reb", 0),
                    "assists": sample.get("actual_ast", 0),
                    "pra": sample.get("actual_pra", 0),
                }
                actual_value = actual_map.get(prop_type, 0)

                direction = ev_result["direction"]

                # Push (actual == line) → no action
                if actual_value == prop_line:
                    diag["push"] += 1
                    continue

                won = (
                    actual_value > prop_line
                    if direction == "over"
                    else actual_value < prop_line
                )

                bet_size = ev_result["bet_size"]
                if bet_size <= 0:
                    continue

                # P&L using REAL odds
                best_odds = ev_result.get("best_odds")
                if best_odds is not None and best_odds != 0:
                    if best_odds > 0:
                        payout_ratio = best_odds / 100.0
                    else:
                        payout_ratio = 100.0 / abs(best_odds)
                else:
                    payout_ratio = 100.0 / 110.0  # Fallback -110

                pnl = bet_size * payout_ratio if won else -bet_size
                bankroll += pnl

                trades.append({
                    "date": game_date,
                    "player": player_name,
                    "prop_type": prop_type,
                    "prop_line": prop_line,
                    "predicted": round(predicted_value, 2),
                    "actual": actual_value,
                    "direction": direction,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                    "best_odds": best_odds,
                    "edge": round(ev_result["edge"], 2),
                    "confidence": round(ev_result["confidence"], 4),
                    "ev_edge": round(ev_result.get("ev_edge") or 0, 4),
                    "true_ev": round(ev_result.get("true_ev") or 0, 4),
                    "tier": ev_result["tier"],
                    "bet_size": round(bet_size, 2),
                    "won": bool(won),
                    "pnl": round(pnl, 2),
                    "bankroll": round(bankroll, 2),
                    "bookmaker": prop.get("bookmaker", ""),
                })

                # Debug: first 3 trades per prop type
                debug_key = f"_logged_{prop_type}"
                if diag.get(debug_key, 0) < 3:
                    diag[debug_key] = diag.get(debug_key, 0) + 1
                    logger.info(
                        "TRADE %s | %s %s | line=%.1f pred=%.1f actual=%s | "
                        "odds=%s/%s | dir=%s won=%s pnl=$%.2f",
                        game_date, player_name, prop_type, prop_line,
                        predicted_value, actual_value, over_odds, under_odds,
                        direction, won, pnl,
                    )

        # End-of-day bankroll
        daily_bankroll[game_date] = round(bankroll, 2)

        if (date_idx + 1) % 10 == 0:
            elapsed = _time.time() - sim_start
            rate = (date_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(sorted_dates) - date_idx - 1) / rate if rate > 0 else 0
            logger.info(
                "  %d/%d dates | %d trades | bankroll=$%.2f | ETA %.0fs",
                date_idx + 1, len(sorted_dates), len(trades), bankroll, eta,
            )
            _progress.update({
                "current": date_idx + 1,
                "trades": len(trades),
                "rate": round(rate, 1),
                "eta_sec": round(eta),
            })

    _progress.update({
        "phase": "report",
        "current": len(sorted_dates),
        "total": len(sorted_dates),
        "trades": len(trades),
    })

    # Diagnostics
    logger.info("=== DIAGNOSTIC COUNTERS ===")
    for key in sorted(diag):
        if not key.startswith("_"):
            logger.info("  %s: %s", key, diag[key])

    logger.info(
        "Simulation complete: %d trades, final bankroll=$%.2f",
        len(trades), bankroll,
    )

    # ── Naive baseline summary ────────────────────────────────────────
    if naive_trades:
        naive_total = len(naive_trades)
        naive_wins = sum(1 for t in naive_trades if t["won"])
        naive_wagered = sum(t["bet_size"] for t in naive_trades)
        naive_pnl_total = sum(t["pnl"] for t in naive_trades)
        naive_wr = naive_wins / naive_total if naive_total > 0 else 0
        naive_roi = naive_pnl_total / naive_wagered if naive_wagered > 0 else 0
        logger.info("=== NAIVE BASELINE (bet over when season_avg > line) ===")
        logger.info("  Trades: %d | Win Rate: %.1f%% | ROI: %+.2f%% | P&L: $%+.2f",
                     naive_total, naive_wr * 100, naive_roi * 100, naive_pnl_total)

    # ── Report ─────────────────────────────────────────────────────────
    results = generate_report(trades, daily_bankroll)
    if results:
        # Add naive baseline to results
        if naive_trades:
            results["naive_baseline"] = {
                "trades": naive_total,
                "wins": naive_wins,
                "win_rate": round(naive_wr, 4),
                "total_wagered": round(naive_wagered, 2),
                "total_pnl": round(naive_pnl_total, 2),
                "roi": round(naive_roi, 4),
            }

        results["diagnostics"] = {
            k: v for k, v in diag.items() if not k.startswith("_")
        }

        # Re-generate text report now that naive_baseline is included
        report = _format_text_report(results)
        txt_path = os.path.join(OUTPUT_DIR, "real_lines_backtest_report.txt")
        with open(txt_path, "w") as f:
            f.write(report)

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    trades: list[dict], daily_bankroll: dict[str, float]
) -> dict:
    """Build JSON results and text report."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not trades:
        logger.warning("No trades to report!")
        return {"error": "No trades generated"}

    df = pd.DataFrame(trades)

    # Overall metrics
    total = len(df)
    wins = int(df["won"].sum())
    losses = total - wins
    win_rate = wins / total
    total_wagered = float(df["bet_size"].sum())
    total_pnl = float(df["pnl"].sum())
    roi = total_pnl / total_wagered if total_wagered > 0 else 0
    final_bankroll = float(df["bankroll"].iloc[-1])

    # Max drawdown
    bk = df["bankroll"].values
    peak = np.maximum.accumulate(bk)
    dd_pct = (peak - bk) / np.where(peak > 0, peak, 1)
    max_dd_pct = float(np.max(dd_pct))
    max_dd_dollar = float(np.max(peak - bk))

    # Annualised Sharpe
    daily_pnl = df.groupby("date")["pnl"].sum()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    avg_bet = float(df["bet_size"].mean())

    # By prop type
    by_prop: dict = {}
    for pt in PROP_TYPES:
        sub = df[df["prop_type"] == pt]
        if sub.empty:
            continue
        pw = int(sub["won"].sum())
        pt_total = len(sub)
        pt_wag = float(sub["bet_size"].sum())
        pt_pnl = float(sub["pnl"].sum())
        by_prop[pt] = {
            "trades": pt_total,
            "wins": pw,
            "losses": pt_total - pw,
            "win_rate": round(pw / pt_total, 4),
            "total_wagered": round(pt_wag, 2),
            "total_pnl": round(pt_pnl, 2),
            "roi": round(pt_pnl / pt_wag, 4) if pt_wag > 0 else 0,
            "avg_edge": round(float(sub["edge"].mean()), 2),
            "avg_confidence": round(float(sub["confidence"].mean()), 4),
            "avg_ev_edge": round(float(sub["ev_edge"].mean()), 4),
        }

    # Monthly breakdown
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    monthly: dict = {}
    for month, grp in df.groupby("month"):
        mw = int(grp["won"].sum())
        mwag = float(grp["bet_size"].sum())
        mpnl = float(grp["pnl"].sum())
        monthly[str(month)] = {
            "trades": len(grp),
            "wins": mw,
            "win_rate": round(mw / len(grp), 4),
            "pnl": round(mpnl, 2),
            "roi": round(mpnl / mwag, 4) if mwag > 0 else 0,
        }

    # By tier
    by_tier: dict = {}
    for tier, grp in df.groupby("tier"):
        tw = int(grp["won"].sum())
        twag = float(grp["bet_size"].sum())
        tpnl = float(grp["pnl"].sum())
        by_tier[str(tier)] = {
            "trades": len(grp),
            "wins": tw,
            "win_rate": round(tw / len(grp), 4),
            "pnl": round(tpnl, 2),
            "roi": round(tpnl / twag, 4) if twag > 0 else 0,
        }

    # Odds distribution
    if "best_odds" in df.columns:
        odds_stats = {
            "mean_over_odds": round(float(df["over_odds"].dropna().mean()), 1),
            "mean_under_odds": round(float(df["under_odds"].dropna().mean()), 1),
            "pct_plus_odds": round(
                float((df["best_odds"].dropna() > 0).mean()), 4
            ),
        }
    else:
        odds_stats = {}

    # Assemble
    results = {
        "backtest_type": "real_lines",
        "backtest_date": datetime.now().isoformat(),
        "test_season": TEST_SEASON_LABEL,
        "initial_bankroll": INITIAL_BANKROLL,
        "final_bankroll": round(final_bankroll, 2),
        "data_sources": {
            "lines": "The Odds API (historical player props)",
            "odds": "Real American odds from DraftKings/FanDuel",
            "outcomes": "BallDontLie API (actual player box scores)",
            "features": "Walk-forward from BDL game data",
        },
        "caveats": [
            "Models trained on 2020-2023 CSV data, tested on 2024-25 BDL data. "
            "Different data source may introduce feature distribution shift.",
            "Player features built from 2024-25 only (no cross-season carryover). "
            "Early-season predictions may be less reliable.",
            "Historical lines from pre-game snapshots (~1hr before tipoff). "
            "Actual available lines at bet time may have differed.",
            "Flat bet sizing (initial bankroll for all Kelly calculations) "
            "to prevent compounding effects.",
        ],
        "summary": {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "total_wagered": round(total_wagered, 2),
            "total_pnl": round(total_pnl, 2),
            "roi": round(roi, 4),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd_pct, 4),
            "max_drawdown_dollar": round(max_dd_dollar, 2),
            "avg_bet_size": round(avg_bet, 2),
        },
        "odds_distribution": odds_stats,
        "by_prop_type": by_prop,
        "monthly_breakdown": monthly,
        "by_tier": by_tier,
        "bankroll_curve": dict(sorted(daily_bankroll.items())),
    }

    # Bias diagnostics — detect systematic issues
    over_trades = df[df["direction"] == "over"]
    under_trades = df[df["direction"] == "under"]
    results["bias_diagnostics"] = {
        "over_pct": round(len(over_trades) / total, 4) if total > 0 else 0,
        "under_pct": round(len(under_trades) / total, 4) if total > 0 else 0,
        "over_win_rate": round(float(over_trades["won"].mean()), 4) if len(over_trades) > 0 else 0,
        "under_win_rate": round(float(under_trades["won"].mean()), 4) if len(under_trades) > 0 else 0,
        "avg_predicted": round(float(df["predicted"].mean()), 2),
        "avg_line": round(float(df["prop_line"].mean()), 2),
        "avg_actual": round(float(df["actual"].mean()), 2),
        "pred_minus_line": round(float((df["predicted"] - df["prop_line"]).mean()), 2),
        "actual_minus_line": round(float((df["actual"] - df["prop_line"]).mean()), 2),
        "actual_over_line_pct": round(float((df["actual"] > df["prop_line"]).mean()), 4),
        "unique_players": int(df["player"].nunique()),
    }

    # Save trade log for analysis
    trade_log_path = os.path.join(OUTPUT_DIR, "real_lines_trade_log.json")
    with open(trade_log_path, "w") as f:
        json.dump(trades, f, indent=2)
    logger.info("Trade log → %s (%d trades)", trade_log_path, len(trades))

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "real_lines_backtest_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("JSON → %s", json_path)

    # Save text report
    report = _format_text_report(results)
    txt_path = os.path.join(OUTPUT_DIR, "real_lines_backtest_report.txt")
    with open(txt_path, "w") as f:
        f.write(report)
    logger.info("Text → %s", txt_path)

    # Bankroll chart
    try:
        _save_bankroll_chart(df, daily_bankroll)
    except Exception as exc:
        logger.warning("Chart skipped: %s", exc)

    print(report)
    return results


def _format_text_report(results: dict) -> str:
    s = results["summary"]
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("   NBA-BETS REAL-LINES PROFITABILITY BACKTEST  |  2024-25 SEASON")
    lines.append("=" * 80)
    lines.append(f"  Run Date:          {results['backtest_date'][:19]}")
    lines.append(f"  Test Season:       {results['test_season']}")
    lines.append(f"  Initial Bankroll:  ${results['initial_bankroll']:.2f}")
    lines.append(f"  Final Bankroll:    ${results['final_bankroll']:.2f}")
    lines.append("  Lines Source:      Real sportsbook (DraftKings/FanDuel)")
    lines.append("  Outcomes Source:   BallDontLie API")
    lines.append("")

    lines.append("--- SUMMARY " + "-" * 67)
    lines.append(f"  Total Trades:      {s['total_trades']}")
    lines.append(f"  Record:            {s['wins']}W - {s['losses']}L")
    lines.append(f"  Win Rate:          {s['win_rate']:.1%}")
    lines.append(f"  Total Wagered:     ${s['total_wagered']:,.2f}")
    lines.append(f"  Total P&L:         ${s['total_pnl']:+,.2f}")
    lines.append(f"  ROI:               {s['roi']:+.2%}")
    lines.append(f"  Sharpe Ratio:      {s['sharpe_ratio']:.3f}")
    lines.append(f"  Max Drawdown:      {s['max_drawdown_pct']:.1%} (${s['max_drawdown_dollar']:,.2f})")
    lines.append(f"  Avg Bet Size:      ${s['avg_bet_size']:.2f}")

    # Sanity flags
    lines.append("")
    if abs(s["roi"]) > 0.15:
        lines.append("  *** WARNING: |ROI| > 15% — check for data leakage or overfitting")
    if s["win_rate"] > 0.60:
        lines.append("  *** WARNING: Win rate > 60% — suspiciously high for real lines")
    if s["total_trades"] < 50:
        lines.append("  *** WARNING: < 50 trades — insufficient sample size")
    if abs(s["roi"]) <= 0.05 and s["total_trades"] >= 100:
        lines.append("  REALISTIC: ROI within +/-5% — consistent with market efficiency")

    # Odds info
    odds = results.get("odds_distribution", {})
    if odds:
        lines.append("")
        lines.append("--- ODDS DISTRIBUTION " + "-" * 57)
        lines.append(f"  Avg Over Odds:     {odds.get('mean_over_odds', 'N/A')}")
        lines.append(f"  Avg Under Odds:    {odds.get('mean_under_odds', 'N/A')}")
        lines.append(f"  % Plus Money:      {odds.get('pct_plus_odds', 0):.1%}")

    lines.append("")
    lines.append("--- BY PROP TYPE " + "-" * 62)
    lines.append(
        f"  {'Prop':<10} {'Trades':>7} {'W-L':>10} {'Win%':>7} "
        f"{'P&L':>10} {'ROI':>8} {'AvgEdge':>8} {'EvEdge':>8}"
    )
    lines.append(
        f"  {'-' * 10} {'-' * 7} {'-' * 10} {'-' * 7} "
        f"{'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8}"
    )
    for pt, p in results["by_prop_type"].items():
        wl = f"{p['wins']}-{p['losses']}"
        lines.append(
            f"  {pt:<10} {p['trades']:>7} {wl:>10} {p['win_rate']:>6.1%} "
            f"${p['total_pnl']:>+9,.2f} {p['roi']:>+7.2%} "
            f"{p['avg_edge']:>7.2f} {p.get('avg_ev_edge', 0):>+7.4f}"
        )

    lines.append("")
    lines.append("--- MONTHLY BREAKDOWN " + "-" * 58)
    lines.append(f"  {'Month':<10} {'Trades':>7} {'Win%':>7} {'P&L':>10} {'ROI':>8}")
    lines.append(f"  {'-' * 10} {'-' * 7} {'-' * 7} {'-' * 10} {'-' * 8}")
    for month, m in sorted(results["monthly_breakdown"].items()):
        lines.append(
            f"  {month:<10} {m['trades']:>7} {m['win_rate']:>6.1%} "
            f"${m['pnl']:>+9,.2f} {m['roi']:>+7.2%}"
        )

    lines.append("")
    lines.append("--- BY TIER " + "-" * 67)
    lines.append(f"  {'Tier':<10} {'Trades':>7} {'Win%':>7} {'P&L':>10} {'ROI':>8}")
    lines.append(f"  {'-' * 10} {'-' * 7} {'-' * 7} {'-' * 10} {'-' * 8}")
    for tier, t in sorted(results["by_tier"].items()):
        lines.append(
            f"  {tier:<10} {t['trades']:>7} {t['win_rate']:>6.1%} "
            f"${t['pnl']:>+9,.2f} {t['roi']:>+7.2%}"
        )

    lines.append("")
    # Naive baseline comparison
    nb = results.get("naive_baseline", {})
    if nb:
        lines.append("--- NAIVE BASELINE (bet over when season_avg > line) " + "-" * 27)
        lines.append(f"  Trades:            {nb.get('trades', 0)}")
        lines.append(f"  Win Rate:          {nb.get('win_rate', 0):.1%}")
        lines.append(f"  ROI:               {nb.get('roi', 0):+.2%}")
        lines.append(f"  P&L:               ${nb.get('total_pnl', 0):+,.2f}")
        model_roi = s.get("roi", 0)
        naive_roi = nb.get("roi", 0)
        lift = model_roi - naive_roi
        lines.append(f"  Model lift vs naive: {lift:+.2%}")
        if lift < 0.02:
            lines.append("  *** Model does NOT beat naive baseline by 2%+ — no genuine edge ***")
        else:
            lines.append(f"  Model adds {lift:.1%} genuine edge over naive strategy")
        lines.append("")

    # Bias diagnostics
    bd = results.get("bias_diagnostics", {})
    if bd:
        lines.append("--- BIAS DIAGNOSTICS " + "-" * 58)
        lines.append(f"  Direction split:    {bd.get('over_pct', 0):.1%} over / {bd.get('under_pct', 0):.1%} under")
        lines.append(f"  Over win rate:      {bd.get('over_win_rate', 0):.1%}")
        lines.append(f"  Under win rate:     {bd.get('under_win_rate', 0):.1%}")
        lines.append(f"  Avg predicted:      {bd.get('avg_predicted', 0):.2f}")
        lines.append(f"  Avg line:           {bd.get('avg_line', 0):.2f}")
        lines.append(f"  Avg actual:         {bd.get('avg_actual', 0):.2f}")
        lines.append(f"  Pred - Line:        {bd.get('pred_minus_line', 0):+.2f}")
        lines.append(f"  Actual - Line:      {bd.get('actual_minus_line', 0):+.2f}")
        lines.append(f"  Actual > Line:      {bd.get('actual_over_line_pct', 0):.1%}")
        lines.append(f"  Unique players:     {bd.get('unique_players', 0)}")
        lines.append("")

    lines.append("--- CAVEATS " + "-" * 67)
    for caveat in results.get("caveats", []):
        lines.append(f"  * {caveat}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def _save_bankroll_chart(
    df: pd.DataFrame, daily_bankroll: dict[str, float]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates = sorted(daily_bankroll.keys())
    values = [daily_bankroll[d] for d in dates]
    date_objs = pd.to_datetime(dates)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(date_objs, values, "b-", linewidth=1.5, label="Bankroll")
    ax1.axhline(
        y=INITIAL_BANKROLL,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Start (${INITIAL_BANKROLL:,.0f})",
    )
    ax1.fill_between(
        date_objs,
        INITIAL_BANKROLL,
        values,
        where=[v >= INITIAL_BANKROLL for v in values],
        alpha=0.15,
        color="green",
    )
    ax1.fill_between(
        date_objs,
        INITIAL_BANKROLL,
        values,
        where=[v < INITIAL_BANKROLL for v in values],
        alpha=0.15,
        color="red",
    )
    ax1.set_title(
        "NBA-BETS Real-Lines Backtest — 2024-25 Season",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylabel("Bankroll ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    daily_pnl = df.groupby("date")["pnl"].sum()
    daily_dates = pd.to_datetime(daily_pnl.index)
    colors = ["green" if p >= 0 else "red" for p in daily_pnl.values]
    ax2.bar(daily_dates, daily_pnl.values, color=colors, alpha=0.6, width=1)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("Daily P&L ($)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "real_lines_bankroll_curve.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info("Chart → %s", chart_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="NBA-BETS Real-Lines Profitability Backtest"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll (default: 1000)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Custom model directory (default: models/)",
    )
    parser.add_argument(
        "--oos",
        action="store_true",
        help="True OOS mode: train fresh models on pre-test-season data only",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2024-25",
        choices=list(SEASON_CONFIGS.keys()),
        help="Test season (default: 2024-25)",
    )
    args = parser.parse_args()

    global INITIAL_BANKROLL, TEST_SEASON_LABEL, TEST_SEASON_INT, CONTEXT_SEASONS
    INITIAL_BANKROLL = args.bankroll

    # Apply season config
    season_cfg = SEASON_CONFIGS[args.season]
    TEST_SEASON_LABEL = args.season
    TEST_SEASON_INT = season_cfg["test_int"]
    CONTEXT_SEASONS = season_cfg["context_seasons"]

    model_dir = Path(args.model_dir) if args.model_dir else None
    results = run_backtest(args, model_dir=model_dir, season_cfg=season_cfg)

    if results and "error" not in results:
        logger.info("Backtest completed successfully!")
        return 0

    logger.error("Backtest failed!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
