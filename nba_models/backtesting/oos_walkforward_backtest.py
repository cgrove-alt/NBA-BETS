#!/usr/bin/env python3
"""
True Out-of-Sample Walk-Forward Backtest

Unlike profitability_backtest.py (which uses models trained on data including
the test season), this script trains FRESH models on historical data and tests
on a held-out future season. This is the honest way to evaluate.

Walk-forward windows:
  Window 1: Train 2021-22, 2022-23 -> Test 2023-24
  Window 2: Train 2022-23, 2023-24 -> Test 2024-25

Usage:
    PYTHONPATH=. python3 nba_models/backtesting/oos_walkforward_backtest.py
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROP_TYPES = ["points", "rebounds", "assists", "pra"]
STANDARD_ODDS = -110
INITIAL_BANKROLL = 1000.0
MIN_MINUTES = 15
OUTPUT_DIR = os.path.join(ROOT, "data", "backtest_results")

# Walk-forward windows: train on N seasons, test on N+1
WALK_FORWARD_WINDOWS = [
    {
        "name": "Window 1",
        "train_seasons": ["2021-22", "2022-23"],
        "test_season": "2023-24",
        "test_date_start": "2023-10-01",
        "test_date_end": "2024-06-30",
    },
    {
        "name": "Window 2",
        "train_seasons": ["2022-23", "2023-24"],
        "test_season": "2024-25",
        "test_date_start": "2024-10-01",
        "test_date_end": "2025-06-30",
    },
]

# Season date boundaries for splitting training vs test data
SEASON_DATE_RANGES = {
    "2021-22": ("2021-10-01", "2022-06-30"),
    "2022-23": ("2022-10-01", "2023-06-30"),
    "2023-24": ("2023-10-01", "2024-06-30"),
    "2024-25": ("2024-10-01", "2025-06-30"),
}


# ---------------------------------------------------------------------------
# Prop-line simulation (Fix 3.1: decorrelated season-avg-only)
# ---------------------------------------------------------------------------
def simulate_prop_line(features: dict, prop_type: str) -> float:
    """Simulate a sportsbook prop line using season average only.

    Decorrelated from model's recent-form features (Fix 3.1).
    """
    if prop_type == "pra":
        season_val = (
            features.get("season_pts_avg", 0)
            + features.get("season_reb_avg", 0)
            + features.get("season_ast_avg", 0)
        )
    else:
        key_map = {
            "points": "season_pts_avg",
            "rebounds": "season_reb_avg",
            "assists": "season_ast_avg",
        }
        season_val = features.get(key_map.get(prop_type, "season_pts_avg"), 0)

    if season_val <= 0:
        return 0.0
    return round(season_val * 2) / 2  # nearest 0.5


# ---------------------------------------------------------------------------
# Real sportsbook lines loader
# ---------------------------------------------------------------------------
def load_real_lines_index(date_start: str, date_end: str, data_dir: str = None) -> dict:
    """Load real sportsbook prop lines from historical_lines/ for a date range.

    Returns a nested dict: {game_date: {player_name_lower: {prop_type: line}}}
    Falls back gracefully if files are missing.
    """
    if data_dir is None:
        data_dir = os.path.join(ROOT, "data", "historical_lines")

    index: dict = {}
    from datetime import date, timedelta

    try:
        start = date.fromisoformat(date_start)
        end = date.fromisoformat(date_end)
    except ValueError:
        logger.warning("load_real_lines_index: invalid date range %s – %s", date_start, date_end)
        return index

    current = start
    loaded = 0
    while current <= end:
        path = os.path.join(data_dir, f"{current.isoformat()}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    day_data = json.load(f)
                day_key = current.isoformat()
                day_index: dict = {}
                for game in day_data.get("games", []):
                    for prop in game.get("player_props", []):
                        pname = prop.get("player_name", "").lower().strip()
                        ptype = prop.get("prop_type", "").lower().strip()
                        line = prop.get("line")
                        if pname and ptype and line is not None and line > 0:
                            if pname not in day_index:
                                day_index[pname] = {}
                            # Keep first occurrence (typically best bookmaker)
                            if ptype not in day_index[pname]:
                                day_index[pname][ptype] = float(line)
                if day_index:
                    index[day_key] = day_index
                    loaded += 1
            except Exception as exc:
                logger.debug("Could not load real lines for %s: %s", current.isoformat(), exc)
        current += timedelta(days=1)

    logger.info(
        "Loaded real sportsbook lines for %d / %d dates (%s – %s)",
        loaded,
        (end - start).days + 1,
        date_start,
        date_end,
    )
    return index


# ---------------------------------------------------------------------------
# Inject prop_line_vs_recent feature (mirrors train_all_models logic)
# ---------------------------------------------------------------------------
PROP_LINE_SEASON_AVG_MAP = {
    "points": ("season_pts_avg", "recent_pts_avg"),
    "rebounds": ("season_reb_avg", "recent_reb_avg"),
    "assists": ("season_ast_avg", "recent_ast_avg"),
    "pra": None,
}

SEASON_AVG_COL = {
    "points": "season_pts_avg",
    "rebounds": "season_reb_avg",
    "assists": "season_ast_avg",
    "pra": None,
}


def _inject_prop_line_features(X_df: pd.DataFrame, prop_name: str) -> pd.DataFrame:
    """Inject prop_line_vs_recent into feature DataFrame for training."""
    X_out = X_df.copy()
    mapping = PROP_LINE_SEASON_AVG_MAP.get(prop_name)

    if mapping is not None:
        season_col, recent_col = mapping
        line_proxy = X_out[season_col].fillna(0)
        X_out["prop_line_vs_recent"] = line_proxy - X_out[recent_col].fillna(
            X_out[season_col]
        )
    else:
        # PRA
        line_proxy = (
            X_out["season_pts_avg"].fillna(0)
            + X_out["season_reb_avg"].fillna(0)
            + X_out["season_ast_avg"].fillna(0)
        )
        X_out["prop_line_vs_recent"] = line_proxy - (
            X_out["recent_pts_avg"].fillna(0)
            + X_out["recent_reb_avg"].fillna(0)
            + X_out["recent_ast_avg"].fillna(0)
        )
    return X_out


# ---------------------------------------------------------------------------
# Train fresh models on training-season data only
# ---------------------------------------------------------------------------
def train_fresh_models(
    player_data: list[dict],
    train_date_end: str,
) -> tuple[dict, dict]:
    """Train PropEnsembleModel and QuantilePropModel for each prop type.

    Uses ONLY player_data samples with game_date <= train_date_end.

    Returns:
        (ensemble_models, quantile_models) keyed by prop type.
    """
    from train_complete_balldontlie import (
        PropEnsembleModel,
        QuantilePropModel,
        MinutesPredictionModel,
        REDUCED_FEATURES,
        smart_fillna,
        calculate_time_decay_weights,
    )

    # Filter to training data only
    train_data = [p for p in player_data if p["game_date"] <= train_date_end]
    train_data.sort(key=lambda x: x["game_date"])

    if not train_data:
        logger.error("No training data before %s", train_date_end)
        return {}, {}

    logger.info("  Training data: %d samples (up to %s)", len(train_data), train_date_end)

    X_all = pd.DataFrame([d["features"] for d in train_data])

    # Train minutes model on first 60% of train data (matching main pipeline)
    # and inject predicted_minutes into ALL player_data (train + test)
    min_train_end = int(len(train_data) * 0.6)
    y_minutes = np.array([d.get("actual_min", 10.0) for d in train_data])
    min_model = MinutesPredictionModel()
    try:
        min_model.train(
            X_all.iloc[:min_train_end], y_minutes[:min_train_end],
        )
        # Predict for all data (train + future test)
        X_full = pd.DataFrame([d["features"] for d in player_data])
        batch = min_model.predict_batch(X_full)
        min_preds = batch[0] if isinstance(batch, tuple) else batch
        if min_preds is not None and len(min_preds) == len(player_data):
            for i, d in enumerate(player_data):
                d["features"]["predicted_minutes"] = float(min_preds[i])
            # Re-create X_all with injected feature
            X_all = pd.DataFrame([d["features"] for d in train_data])
            logger.info("  Injected predicted_minutes (mean=%.1f)", min_preds.mean())
    except Exception as e:
        logger.warning("  Minutes model failed: %s", e)

    # Time-decay weights
    player_dates = [d.get("game_date", "") for d in train_data]
    time_weights = calculate_time_decay_weights(player_dates, half_life_days=180)
    outlier_weights = np.array([d.get("sample_weight", 1.0) for d in train_data])
    sample_weights = outlier_weights * time_weights

    prop_types = [
        ("points", "actual_pts"),
        ("rebounds", "actual_reb"),
        ("assists", "actual_ast"),
        ("pra", "actual_pra"),
    ]

    ensemble_models = {}
    quantile_models = {}

    for prop_name, target_col in prop_types:
        logger.info("  Training %s models...", prop_name)

        y_raw = np.array([d[target_col] for d in train_data])
        X_with_line = _inject_prop_line_features(X_all, prop_name)

        # Residual target (Fix 1.4)
        sa_col = SEASON_AVG_COL.get(prop_name)
        if sa_col is not None:
            season_avgs = X_with_line[sa_col].fillna(0).values
        else:
            season_avgs = (
                X_with_line["season_pts_avg"].fillna(0).values
                + X_with_line["season_reb_avg"].fillna(0).values
                + X_with_line["season_ast_avg"].fillna(0).values
            )
        y_residual = y_raw - season_avgs

        # Reduced features (Fix 1.1)
        reduced_cols = REDUCED_FEATURES.get(prop_name, [])
        available_cols = [c for c in reduced_cols if c in X_with_line.columns]
        X_reduced = X_with_line[available_cols] if available_cols else X_with_line

        # --- Ensemble model ---
        try:
            prop_model = PropEnsembleModel(prop_name)
            prop_model._residual_mode = True
            prop_model._season_avg_col = sa_col
            metrics = prop_model.train(
                X_reduced, y_residual,
                dates=player_dates,
                sample_weights=sample_weights,
            )
            ensemble_models[prop_name] = prop_model
            logger.info(
                "    %s ensemble: RMSE=%.3f, MAE=%.3f, R2=%.4f",
                prop_name, metrics["ensemble_rmse"], metrics["ensemble_mae"],
                metrics["ensemble_r2"],
            )
        except Exception as exc:
            logger.warning("    Failed to train ensemble for %s: %s", prop_name, exc)

        # --- Quantile model (trains on raw target, not residual) ---
        try:
            q_model = QuantilePropModel(prop_name)
            # Compute calibration lines (season averages)
            _q_sa_col = SEASON_AVG_COL.get(prop_name)
            if _q_sa_col is not None:
                _cal_lines = X_with_line[_q_sa_col].fillna(0).values
            else:
                _cal_lines = (
                    X_with_line['season_pts_avg'].fillna(0).values
                    + X_with_line['season_reb_avg'].fillna(0).values
                    + X_with_line['season_ast_avg'].fillna(0).values
                )
            q_model.train(X_reduced, y_raw, sample_weights=sample_weights,
                          calibration_lines=_cal_lines)
            # Compute and store survivorship offset
            _valid = _cal_lines > 0
            q_model._survivorship_offset = float(
                np.mean(y_raw[_valid] - _cal_lines[_valid])
            )
            quantile_models[prop_name] = q_model
            logger.info("    %s quantile: trained OK", prop_name)
        except Exception as exc:
            logger.warning("    Failed to train quantile for %s: %s", prop_name, exc)

    return ensemble_models, quantile_models


# ---------------------------------------------------------------------------
# Evaluate a single window
# ---------------------------------------------------------------------------
def evaluate_window(window: dict) -> dict | None:
    """Run one walk-forward window: train on train_seasons, test on test_season.

    Returns a results dict with per-prop and aggregate metrics.
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
    )
    from nba_betting.prediction_pipeline import evaluate_bet

    name = window["name"]
    train_seasons = window["train_seasons"]
    test_season = window["test_season"]
    test_start = window["test_date_start"]
    test_end = window["test_date_end"]

    # Determine training cutoff: day before test season starts
    train_date_end = SEASON_DATE_RANGES[train_seasons[-1]][1]

    logger.info("=" * 70)
    logger.info("%s: Train on %s, Test on %s", name, train_seasons, test_season)
    logger.info("=" * 70)

    # --- 1. Load data (train + test seasons for feature context) ---
    all_seasons = train_seasons + [test_season]
    logger.info("Step 1: Loading data for seasons %s", all_seasons)

    team_id_map = build_team_id_map()
    team_meta = _build_team_metadata()
    games = load_team_games(all_seasons, team_id_map, team_meta)
    game_ids = {g["id"] for g in games}
    player_stats_by_game = load_player_stats(game_ids, all_seasons, team_id_map)

    total_records = sum(len(v) for v in player_stats_by_game.values())
    logger.info("  Loaded %d games, %d player-game records (CSV)", len(games), total_records)

    # If CSV player stats are sparse (e.g., 2024-25 not in CSVs), supplement
    # with BDL player stats cache. This enables Window 2 (test on 2024-25).
    BDL_STATS_PATH = os.path.join(ROOT, "data", "historical_lines", "player_stats_2024.json")
    BDL_META_PATH = os.path.join(ROOT, "data", "historical_lines", "player_stats_2024_meta.json")
    if os.path.exists(BDL_STATS_PATH) and os.path.exists(BDL_META_PATH):
        from nba_models.backtesting.real_lines_backtest import load_bdl_player_stats
        try:
            bdl_stats = load_bdl_player_stats(games, team_meta, team_id_map)
            bdl_records = sum(len(v) for v in bdl_stats.values())
            if bdl_records > 0:
                # Merge: BDL stats fill in games not covered by CSV
                for gid, players in bdl_stats.items():
                    if gid not in player_stats_by_game:
                        player_stats_by_game[gid] = players
                new_total = sum(len(v) for v in player_stats_by_game.values())
                logger.info(
                    "  Supplemented with BDL data: %d → %d player-game records (+%d)",
                    total_records, new_total, new_total - total_records,
                )
        except Exception as exc:
            logger.warning("  Failed to load BDL supplement: %s", exc)

    # --- 2. Build walk-forward features ---
    logger.info("Step 2: Building walk-forward features...")
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

    logger.info("  Total player-game samples: %d", len(player_data))

    # --- 3. Train fresh models on training data ONLY ---
    logger.info("Step 3: Training fresh models (train seasons only)...")
    t0 = time.time()
    ensemble_models, quantile_models = train_fresh_models(player_data, train_date_end)
    train_time = time.time() - t0
    logger.info("  Training took %.1f seconds", train_time)

    if not ensemble_models:
        logger.error("  No models trained -- skipping window")
        return None

    # --- 4. Evaluate on test season (frozen models) ---
    logger.info("Step 4: Evaluating on test season %s (frozen models)...", test_season)
    test_data = [
        p for p in player_data
        if test_start <= p["game_date"] <= test_end
    ]
    test_data.sort(key=lambda x: x["game_date"])
    logger.info("  Test set: %d player-game samples", len(test_data))

    # Load real sportsbook lines for the test window
    real_lines_index = load_real_lines_index(test_start, test_end)
    real_line_hits = 0
    real_line_misses = 0

    if not test_data:
        logger.error("  No test data in date range %s to %s", test_start, test_end)
        return None

    # Collect predictions and actuals for metrics
    prop_results: dict[str, dict] = {pt: defaultdict(list) for pt in PROP_TYPES}
    trade_log: list[dict] = []
    diag = defaultdict(int)

    for i, sample in enumerate(test_data):
        features = sample["features"]
        player_name = sample.get("player_name", "Unknown")
        game_date = sample["game_date"]
        actual_min = sample.get("actual_min", 0)
        games_played = features.get("season_games", 0)

        diag["total_samples"] += 1

        if actual_min < MIN_MINUTES:
            diag["skipped_low_minutes"] += 1
            continue

        # Skip bench players: require season avg minutes >= 25
        season_min_avg = features.get("season_min_avg", 0) or 0
        if season_min_avg < 25:
            diag["skipped_bench_player"] += 1
            continue

        player_name_lower = player_name.lower().strip()
        day_lines = real_lines_index.get(game_date, {})
        player_lines = day_lines.get(player_name_lower, {})

        for prop_type in PROP_TYPES:
            if prop_type not in ensemble_models:
                continue

            # Prefer real sportsbook line; fall back to season-average proxy
            real_line = player_lines.get(prop_type)
            if real_line and real_line > 0:
                prop_line = real_line
                real_line_hits += 1
            else:
                prop_line = simulate_prop_line(features, prop_type)
                real_line_misses += 1

            if prop_line <= 0:
                diag[f"skip_zero_line_{prop_type}"] += 1
                continue

            # Actual value
            actual_map = {
                "points": sample.get("actual_pts", 0),
                "rebounds": sample.get("actual_reb", 0),
                "assists": sample.get("actual_ast", 0),
                "pra": sample.get("actual_pra", 0),
            }
            actual_value = actual_map[prop_type]

            # Model prediction (frozen)
            model = ensemble_models[prop_type]
            try:
                prediction = model.predict(features, prop_line=prop_line)
            except Exception:
                diag[f"predict_error_{prop_type}"] += 1
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

            # Record for regression metrics
            prop_results[prop_type]["predicted"].append(predicted_value)
            prop_results[prop_type]["actual"].append(actual_value)
            prop_results[prop_type]["line"].append(prop_line)

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
                    pass

            # Evaluate via betting pipeline
            ev_result = evaluate_bet(
                prop_type=prop_type,
                predicted=predicted_value,
                line=prop_line,
                raw_confidence=over_prob,
                games_played=games_played,
                bankroll=INITIAL_BANKROLL,
                over_odds=STANDARD_ODDS,
                under_odds=STANDARD_ODDS,
                pre_calibrated=use_pre_calibrated,
            )

            if not ev_result["should_bet"]:
                continue

            # Determine outcome
            if actual_value == prop_line:
                continue  # push

            direction = ev_result["direction"]
            won = (
                actual_value > prop_line if direction == "over"
                else actual_value < prop_line
            )
            bet_size = ev_result["bet_size"]
            if bet_size <= 0:
                continue

            pnl = bet_size * (100.0 / 110.0) if won else -bet_size

            trade_log.append({
                "date": game_date,
                "player": player_name,
                "prop_type": prop_type,
                "line": prop_line,
                "predicted": round(predicted_value, 2),
                "actual": actual_value,
                "direction": direction,
                "won": bool(won),
                "pnl": round(pnl, 2),
                "bet_size": round(bet_size, 2),
            })

        if (i + 1) % 500 == 0:
            logger.info("  Processed %d/%d test samples", i + 1, len(test_data))

    # --- 5. Compute metrics ---
    total_line_lookups = real_line_hits + real_line_misses
    if total_line_lookups > 0:
        logger.info(
            "  Real sportsbook lines: %d / %d prop lookups used real lines (%.1f%% coverage)",
            real_line_hits,
            total_line_lookups,
            100.0 * real_line_hits / total_line_lookups,
        )
    else:
        logger.info("  Real sportsbook lines: none available — using season-average proxies")

    logger.info("Step 5: Computing metrics...")
    window_metrics: dict = {
        "window": name,
        "train_seasons": train_seasons,
        "test_season": test_season,
        "train_time_sec": round(train_time, 1),
        "test_samples": len(test_data),
        "diagnostics": dict(diag),
        "by_prop_type": {},
    }

    for prop_type in PROP_TYPES:
        pr = prop_results[prop_type]
        if not pr["actual"]:
            continue

        actual = np.array(pr["actual"])
        predicted = np.array(pr["predicted"])
        lines = np.array(pr["line"])

        n = len(actual)
        rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
        mae = float(np.mean(np.abs(predicted - actual)))
        bias = float(np.mean(predicted - actual))

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Baseline RMSE (season avg as predictor)
        baseline_rmse = float(np.sqrt(np.mean((lines - actual) ** 2)))

        # Directional accuracy: did model predict correct side of line?
        model_direction = predicted > lines
        actual_direction = actual > lines
        mask = actual != lines  # exclude pushes
        if mask.sum() > 0:
            directional_acc = float(
                np.mean(model_direction[mask] == actual_direction[mask])
            )
        else:
            directional_acc = 0.5

        window_metrics["by_prop_type"][prop_type] = {
            "n_predictions": n,
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "r2": round(r2, 4),
            "bias": round(bias, 3),
            "baseline_rmse": round(baseline_rmse, 3),
            "directional_accuracy": round(directional_acc, 4),
        }

    # Trade-level metrics
    if trade_log:
        df_trades = pd.DataFrame(trade_log)
        total_trades = len(df_trades)
        wins = int(df_trades["won"].sum())
        total_wagered = float(df_trades["bet_size"].sum())
        total_pnl = float(df_trades["pnl"].sum())
        roi = total_pnl / total_wagered if total_wagered > 0 else 0.0

        window_metrics["betting"] = {
            "total_trades": total_trades,
            "wins": wins,
            "losses": total_trades - wins,
            "win_rate": round(wins / total_trades, 4) if total_trades > 0 else 0,
            "total_wagered": round(total_wagered, 2),
            "total_pnl": round(total_pnl, 2),
            "roi": round(roi, 4),
        }

        # Per-prop betting
        for prop_type in PROP_TYPES:
            sub = df_trades[df_trades["prop_type"] == prop_type]
            if sub.empty:
                continue
            pw = int(sub["won"].sum())
            pt_total = len(sub)
            pt_wag = float(sub["bet_size"].sum())
            pt_pnl = float(sub["pnl"].sum())
            if prop_type in window_metrics["by_prop_type"]:
                window_metrics["by_prop_type"][prop_type].update({
                    "trades": pt_total,
                    "trade_wins": pw,
                    "trade_win_rate": round(pw / pt_total, 4) if pt_total > 0 else 0,
                    "trade_roi": round(pt_pnl / pt_wag, 4) if pt_wag > 0 else 0,
                })
    else:
        window_metrics["betting"] = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_wagered": 0,
            "total_pnl": 0,
            "roi": 0,
        }

    return window_metrics


# ---------------------------------------------------------------------------
# Aggregate results across windows
# ---------------------------------------------------------------------------
def aggregate_results(window_results: list[dict]) -> dict:
    """Combine metrics across walk-forward windows."""
    agg: dict = {
        "run_date": datetime.now().isoformat(),
        "n_windows": len(window_results),
        "windows": window_results,
        "aggregate": {"by_prop_type": {}},
    }

    # Aggregate per-prop metrics (weighted by n_predictions)
    for prop_type in PROP_TYPES:
        total_n = 0
        weighted_rmse_sq = 0.0
        weighted_mae = 0.0
        weighted_r2 = 0.0
        weighted_bias = 0.0
        weighted_dir_acc = 0.0
        total_trades = 0
        total_trade_wins = 0

        for w in window_results:
            pm = w.get("by_prop_type", {}).get(prop_type)
            if not pm:
                continue
            n = pm["n_predictions"]
            total_n += n
            weighted_rmse_sq += pm["rmse"] ** 2 * n
            weighted_mae += pm["mae"] * n
            weighted_r2 += pm["r2"] * n
            weighted_bias += pm["bias"] * n
            weighted_dir_acc += pm["directional_accuracy"] * n
            total_trades += pm.get("trades", 0)
            total_trade_wins += pm.get("trade_wins", 0)

        if total_n == 0:
            continue

        agg["aggregate"]["by_prop_type"][prop_type] = {
            "n_predictions": total_n,
            "rmse": round(float(np.sqrt(weighted_rmse_sq / total_n)), 3),
            "mae": round(weighted_mae / total_n, 3),
            "r2": round(weighted_r2 / total_n, 4),
            "bias": round(weighted_bias / total_n, 3),
            "directional_accuracy": round(weighted_dir_acc / total_n, 4),
            "total_trades": total_trades,
            "trade_win_rate": (
                round(total_trade_wins / total_trades, 4) if total_trades > 0 else 0
            ),
        }

    # Aggregate betting
    total_trades = sum(w["betting"]["total_trades"] for w in window_results)
    total_wins = sum(w["betting"]["wins"] for w in window_results)
    total_wagered = sum(w["betting"]["total_wagered"] for w in window_results)
    total_pnl = sum(w["betting"]["total_pnl"] for w in window_results)

    agg["aggregate"]["betting"] = {
        "total_trades": total_trades,
        "wins": total_wins,
        "losses": total_trades - total_wins,
        "win_rate": round(total_wins / total_trades, 4) if total_trades > 0 else 0,
        "total_wagered": round(total_wagered, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(total_pnl / total_wagered, 4) if total_wagered > 0 else 0,
    }

    return agg


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------
def print_report(results: dict) -> str:
    """Format and print a human-readable report. Returns the report string."""
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("   TRUE OUT-OF-SAMPLE WALK-FORWARD BACKTEST")
    lines.append("=" * 80)
    lines.append(f"  Run Date: {results['run_date'][:19]}")
    lines.append(f"  Windows:  {results['n_windows']}")
    lines.append("")

    # Per-window detail
    for w in results["windows"]:
        lines.append("-" * 80)
        lines.append(
            f"  {w['window']}: Train {w['train_seasons']} -> Test {w['test_season']}"
        )
        lines.append(f"  Test samples: {w['test_samples']}")
        lines.append(f"  Training time: {w['train_time_sec']}s")
        lines.append("")

        lines.append(
            f"  {'Prop':<10} {'N':>6} {'RMSE':>7} {'MAE':>7} {'R2':>8} "
            f"{'Bias':>7} {'BasRMSE':>8} {'DirAcc':>7} {'Trades':>7} {'WinRate':>8} {'ROI':>8}"
        )
        lines.append(
            f"  {'-'*10} {'-'*6} {'-'*7} {'-'*7} {'-'*8} "
            f"{'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*8}"
        )
        for pt, pm in w.get("by_prop_type", {}).items():
            trades = pm.get("trades", 0)
            wr = pm.get("trade_win_rate", 0)
            roi = pm.get("trade_roi", 0)
            lines.append(
                f"  {pt:<10} {pm['n_predictions']:>6} {pm['rmse']:>7.3f} "
                f"{pm['mae']:>7.3f} {pm['r2']:>8.4f} {pm['bias']:>7.3f} "
                f"{pm['baseline_rmse']:>8.3f} {pm['directional_accuracy']:>6.1%} "
                f"{trades:>7} {wr:>7.1%} {roi:>+7.2%}"
            )

        b = w.get("betting", {})
        lines.append("")
        lines.append(
            f"  Betting: {b.get('total_trades', 0)} trades, "
            f"{b.get('wins', 0)}W-{b.get('losses', 0)}L, "
            f"Win Rate {b.get('win_rate', 0):.1%}, "
            f"P&L ${b.get('total_pnl', 0):+,.2f}, "
            f"ROI {b.get('roi', 0):+.2%}"
        )
        lines.append("")

    # Aggregate
    lines.append("=" * 80)
    lines.append("  AGGREGATE ACROSS ALL WINDOWS")
    lines.append("=" * 80)

    agg = results.get("aggregate", {})
    lines.append(
        f"  {'Prop':<10} {'N':>6} {'RMSE':>7} {'MAE':>7} {'R2':>8} "
        f"{'Bias':>7} {'DirAcc':>7} {'Trades':>7} {'WinRate':>8}"
    )
    lines.append(
        f"  {'-'*10} {'-'*6} {'-'*7} {'-'*7} {'-'*8} "
        f"{'-'*7} {'-'*7} {'-'*7} {'-'*8}"
    )
    for pt, pm in agg.get("by_prop_type", {}).items():
        lines.append(
            f"  {pt:<10} {pm['n_predictions']:>6} {pm['rmse']:>7.3f} "
            f"{pm['mae']:>7.3f} {pm['r2']:>8.4f} {pm['bias']:>7.3f} "
            f"{pm['directional_accuracy']:>6.1%} "
            f"{pm.get('total_trades', 0):>7} {pm.get('trade_win_rate', 0):>7.1%}"
        )

    ab = agg.get("betting", {})
    lines.append("")
    lines.append(
        f"  Overall Betting: {ab.get('total_trades', 0)} trades, "
        f"{ab.get('wins', 0)}W-{ab.get('losses', 0)}L, "
        f"Win Rate {ab.get('win_rate', 0):.1%}, "
        f"P&L ${ab.get('total_pnl', 0):+,.2f}, "
        f"ROI {ab.get('roi', 0):+.2%}"
    )

    # Sanity checks
    lines.append("")
    lines.append("--- SANITY CHECKS " + "-" * 61)
    roi_val = ab.get("roi", 0)
    wr_val = ab.get("win_rate", 0)
    if abs(roi_val) > 0.15:
        lines.append("  WARNING: |ROI| > 15% -- investigate for residual leakage")
    if wr_val > 0.60:
        lines.append("  WARNING: Win rate > 60% -- suspiciously high for OOS")
    if ab.get("total_trades", 0) < 50:
        lines.append("  WARNING: < 50 trades -- insufficient sample size")
    for pt, pm in agg.get("by_prop_type", {}).items():
        if pm.get("r2", 0) < 0:
            lines.append(f"  NOTE: {pt} R2 < 0 -- model worse than mean predictor")

    lines.append("")
    lines.append("--- METHODOLOGY " + "-" * 63)
    lines.append(
        "  * Models trained FRESH on training seasons only (no test data)."
    )
    lines.append(
        "  * Features are walk-forward safe (point-in-time calculators)."
    )
    lines.append(
        "  * Prop lines use real sportsbook odds when available (data/historical_lines/);"
    )
    lines.append(
        "    season-average proxy used as fallback (Fix 3.1)."
    )
    lines.append(
        "  * Models are frozen during test evaluation (no retraining)."
    )
    lines.append(
        "  * This is the honest evaluation. Compare with profitability_backtest.py"
    )
    lines.append(
        "    (in-sample model) to measure the optimism gap."
    )
    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Starting True OOS Walk-Forward Backtest")
    t_start = time.time()

    window_results: list[dict] = []

    for window in WALK_FORWARD_WINDOWS:
        try:
            result = evaluate_window(window)
            if result is not None:
                window_results.append(result)
            else:
                logger.warning("Window %s returned no results", window["name"])
        except Exception as exc:
            logger.error("Window %s failed: %s", window["name"], exc, exc_info=True)

    if not window_results:
        logger.error("No windows completed successfully!")
        return 1

    # Aggregate
    results = aggregate_results(window_results)
    results["total_runtime_sec"] = round(time.time() - t_start, 1)

    # Save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "oos_walkforward_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", json_path)

    # Save text report
    report = print_report(results)
    txt_path = os.path.join(OUTPUT_DIR, "oos_walkforward_report.txt")
    with open(txt_path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", txt_path)

    logger.info("Total runtime: %.1f seconds", results["total_runtime_sec"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
