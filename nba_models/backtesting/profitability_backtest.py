#!/usr/bin/env python3
"""
Profitability Backtest: Walk-Forward P&L Simulation

Replays the 2023-24 NBA season using real trained models, the evaluate_bet()
pipeline, standard -110 odds, and quarter-Kelly sizing.

Walk-forward guarantee: features for each game are computed using ONLY data
from games before that date (via process_games_for_training's point-in-time
calculators).  The trained model weights, however, were fit on data that
includes 2023-24 — so this is an "in-sample features, in-sample model"
backtest.  Results will be optimistic; treat as an upper bound.

Outputs:
  data/backtest_results/profitability_backtest_results.json
  data/backtest_results/profitability_backtest_report.txt
  data/backtest_results/bankroll_curve.png

Usage:
    PYTHONPATH=. python3 nba_models/backtesting/profitability_backtest.py
    PYTHONPATH=. python3 nba_models/backtesting/profitability_backtest.py --bankroll 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
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
INITIAL_BANKROLL = 1000.0
PROP_TYPES = ["points", "rebounds", "assists", "pra"]
STANDARD_ODDS = -110
TEST_SEASON = "2023-24"
# Need 2+ prior seasons for rolling-average features to stabilise
CONTEXT_SEASONS = ["2021-22", "2022-23", "2023-24"]
OUTPUT_DIR = os.path.join(ROOT, "data", "backtest_results")
MIN_MINUTES = 15  # Skip garbage-time-only players


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models() -> dict:
    """Load trained PropEnsembleModel (or fallback) for each prop type."""
    models: dict = {}
    model_dir = Path(ROOT) / "models"

    for prop_type in PROP_TYPES:
        # Prefer ensemble → quantile → standard
        candidates = [
            model_dir / f"player_{prop_type}_ensemble.pkl",
            model_dir / f"player_{prop_type}_quantile.pkl",
            model_dir / f"player_{prop_type}.pkl",
        ]
        for path in candidates:
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        models[prop_type] = pickle.load(f)
                    logger.info("Loaded %s for %s", path.name, prop_type)
                    break
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", path, exc)

        if prop_type not in models:
            logger.warning("No model found for %s — skipping", prop_type)

    return models


# ---------------------------------------------------------------------------
# Prop-line simulation
# ---------------------------------------------------------------------------
def simulate_prop_line(features: dict, prop_type: str) -> float:
    """Simulate a sportsbook prop line.

    Books anchor to the player's season average and nudge toward recent form.
    We use 70/30 season/recent weighting, rounded to the nearest 0.5.
    """
    key_map = {
        "points": ("season_pts_avg", "recent_pts_avg"),
        "rebounds": ("season_reb_avg", "recent_reb_avg"),
        "assists": ("season_ast_avg", "recent_ast_avg"),
        "pra": (None, "pra_avg"),  # no single season-pra key
    }

    season_key, recent_key = key_map.get(prop_type, ("season_pts_avg", "recent_pts_avg"))

    if prop_type == "pra":
        season_val = (
            features.get("season_pts_avg", 0)
            + features.get("season_reb_avg", 0)
            + features.get("season_ast_avg", 0)
        )
    else:
        season_val = features.get(season_key, 0)

    recent_val = features.get(recent_key, season_val)

    if season_val <= 0:
        return 0.0

    line = 0.70 * season_val + 0.30 * recent_val
    return round(line * 2) / 2  # nearest 0.5


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------
def run_backtest(args: argparse.Namespace) -> dict | None:
    """Execute the walk-forward backtest and return results dict."""

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

    # ── 1. Load CSV data ──────────────────────────────────────────────────
    logger.info("Step 1/5: Loading CSV data …")
    team_id_map = build_team_id_map()
    team_meta = _build_team_metadata()

    games = load_team_games(CONTEXT_SEASONS, team_id_map, team_meta)
    game_ids = {g["id"] for g in games}
    player_stats_by_game = load_player_stats(game_ids, CONTEXT_SEASONS, team_id_map)

    total_records = sum(len(v) for v in player_stats_by_game.values())
    logger.info("Loaded %d games, %d player-game records", len(games), total_records)

    # ── 2. Build walk-forward features ────────────────────────────────────
    logger.info("Step 2/5: Building walk-forward features …")
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

    # Filter to test season (Oct 2023 – Apr 2024)
    test_data = [
        p
        for p in player_data
        if "2023-10-01" <= p["game_date"] <= "2024-04-30"
    ]
    test_data.sort(key=lambda x: x["game_date"])
    logger.info("Test set: %d player-game samples in %s", len(test_data), TEST_SEASON)

    # ── 3. Load models ────────────────────────────────────────────────────
    logger.info("Step 3/5: Loading trained models …")
    models = load_models()
    if not models:
        logger.error("No models loaded — cannot run backtest.")
        return None
    logger.info("Models available: %s", list(models.keys()))

    # ── 4. Simulate trades ────────────────────────────────────────────────
    logger.info("Step 4/5: Running walk-forward P&L simulation …")

    bankroll = INITIAL_BANKROLL
    trades: list[dict] = []
    daily_bankroll: dict[str, float] = {}

    # Diagnostic counters
    diag = defaultdict(int)

    for i, sample in enumerate(test_data):
        game_date = sample["game_date"]
        features = sample["features"]
        player_name = sample.get("player_name", "Unknown")
        games_played = features.get("season_games", 0)
        actual_min = sample.get("actual_min", 0)

        diag["total_samples"] += 1

        if actual_min < MIN_MINUTES:
            diag["skipped_low_minutes"] += 1
            continue

        diag["eligible_samples"] += 1

        for prop_type in PROP_TYPES:
            if prop_type not in models:
                diag[f"skip_no_model_{prop_type}"] += 1
                continue

            prop_line = simulate_prop_line(features, prop_type)
            if prop_line <= 0:
                diag[f"skip_zero_line_{prop_type}"] += 1
                continue

            diag[f"predictions_attempted_{prop_type}"] += 1

            # Model prediction
            model = models[prop_type]
            try:
                prediction = model.predict(features, prop_line=prop_line)
            except Exception as exc:
                diag[f"predict_error_{prop_type}"] += 1
                if diag[f"predict_error_{prop_type}"] <= 3:
                    logger.warning("Model predict error (%s): %s", prop_type, exc)
                continue

            predicted_value = prediction.get("predicted_value", 0)
            over_prob = prediction.get("over_probability")
            edge = abs(predicted_value - prop_line)

            diag[f"predictions_ok_{prop_type}"] += 1
            if edge > 0:
                diag[f"nonzero_edge_{prop_type}"] += 1

            # Log first 3 predictions per prop type for debugging
            debug_key = f"_logged_{prop_type}"
            if diag.get(debug_key, 0) < 3:
                diag[debug_key] = diag.get(debug_key, 0) + 1
                logger.info(
                    "SAMPLE %s | %s %s | line=%.1f pred=%.1f edge=%.2f over_p=%s gp=%d",
                    game_date, player_name, prop_type, prop_line,
                    predicted_value, edge, over_prob, games_played,
                )

            # Pipeline evaluation
            ev_result = evaluate_bet(
                prop_type=prop_type,
                predicted=predicted_value,
                line=prop_line,
                raw_confidence=over_prob,
                games_played=games_played,
                bankroll=bankroll,
                over_odds=STANDARD_ODDS,
                under_odds=STANDARD_ODDS,
            )

            if not ev_result["should_bet"]:
                reason = ev_result.get("reason", "unknown")
                # Categorise rejection
                if "disabled" in reason.lower():
                    diag[f"reject_disabled_{prop_type}"] += 1
                elif "games played" in reason.lower() or "sample" in reason.lower():
                    diag[f"reject_sample_size_{prop_type}"] += 1
                elif "edge" in reason.lower() and "threshold" in reason.lower():
                    diag[f"reject_low_edge_{prop_type}"] += 1
                elif "confidence" in reason.lower():
                    diag[f"reject_low_confidence_{prop_type}"] += 1
                elif "ev" in reason.lower():
                    diag[f"reject_low_ev_{prop_type}"] += 1
                elif "kelly" in reason.lower():
                    diag[f"reject_kelly_{prop_type}"] += 1
                else:
                    diag[f"reject_other_{prop_type}"] += 1
                continue

            # Actual outcome
            actual_map = {
                "points": sample.get("actual_pts", 0),
                "rebounds": sample.get("actual_reb", 0),
                "assists": sample.get("actual_ast", 0),
                "pra": sample.get("actual_pra", 0),
            }
            actual_value = actual_map.get(prop_type, 0)

            direction = ev_result["direction"]
            won = (
                actual_value > prop_line
                if direction == "over"
                else actual_value < prop_line
            )

            # Push (actual == line) → skip
            if actual_value == prop_line:
                continue

            bet_size = ev_result["bet_size"]
            if bet_size <= 0:
                continue

            # P&L at -110 odds
            pnl = bet_size * (100.0 / 110.0) if won else -bet_size
            bankroll += pnl

            trades.append(
                {
                    "date": game_date,
                    "player": player_name,
                    "prop_type": prop_type,
                    "prop_line": prop_line,
                    "predicted": round(predicted_value, 2),
                    "actual": actual_value,
                    "direction": direction,
                    "edge": round(ev_result["edge"], 2),
                    "confidence": round(ev_result["confidence"], 4),
                    "tier": ev_result["tier"],
                    "bet_size": round(bet_size, 2),
                    "won": bool(won),
                    "pnl": round(pnl, 2),
                    "bankroll": round(bankroll, 2),
                }
            )

        # Record end-of-day bankroll
        daily_bankroll[game_date] = round(bankroll, 2)

        if (i + 1) % 5000 == 0:
            logger.info(
                "  %d/%d samples | %d trades | bankroll=$%.2f",
                i + 1,
                len(test_data),
                len(trades),
                bankroll,
            )

    # Log diagnostics
    logger.info("=== DIAGNOSTIC COUNTERS ===")
    for key in sorted(diag):
        logger.info("  %s: %d", key, diag[key])

    logger.info(
        "Simulation complete: %d trades, final bankroll=$%.2f",
        len(trades),
        bankroll,
    )

    # ── 5. Report ─────────────────────────────────────────────────────────
    logger.info("Step 5/5: Generating report …")
    results = generate_report(trades, daily_bankroll)
    if results:
        results["diagnostics"] = dict(diag)
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(trades: list[dict], daily_bankroll: dict[str, float]) -> dict:
    """Build JSON results, text report, and bankroll chart."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not trades:
        logger.warning("No trades to report!")
        return {"error": "No trades generated"}

    df = pd.DataFrame(trades)

    # ── Overall metrics ───────────────────────────────────────────────────
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

    # Annualised Sharpe (daily P&L)
    daily_pnl = df.groupby("date")["pnl"].sum()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    avg_bet = float(df["bet_size"].mean())

    # ── By prop type ──────────────────────────────────────────────────────
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
        }

    # ── Monthly breakdown ─────────────────────────────────────────────────
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

    # ── By tier ───────────────────────────────────────────────────────────
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

    # ── Assemble ──────────────────────────────────────────────────────────
    results = {
        "backtest_date": datetime.now().isoformat(),
        "test_season": TEST_SEASON,
        "initial_bankroll": INITIAL_BANKROLL,
        "final_bankroll": round(final_bankroll, 2),
        "caveats": [
            "Model weights were trained on data including the test season "
            "(in-sample model). Features are walk-forward safe (point-in-time). "
            "Treat ROI as an upper-bound estimate.",
            "Prop lines are simulated (70% season avg + 30% recent avg, "
            "rounded to 0.5). Real lines may differ.",
            "OT-normalised actuals used for settlement (matches training). "
            "Real books settle on raw stats.",
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
        "by_prop_type": by_prop,
        "monthly_breakdown": monthly,
        "by_tier": by_tier,
        "bankroll_curve": dict(sorted(daily_bankroll.items())),
    }

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "profitability_backtest_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("JSON → %s", json_path)

    # Save text report
    report = _format_text_report(results)
    txt_path = os.path.join(OUTPUT_DIR, "profitability_backtest_report.txt")
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


# ---------------------------------------------------------------------------
# Text report formatter
# ---------------------------------------------------------------------------
def _format_text_report(results: dict) -> str:
    s = results["summary"]
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("   NBA-BETS PROFITABILITY BACKTEST  |  2023-24 SEASON")
    lines.append("=" * 80)
    lines.append(f"  Run Date:          {results['backtest_date'][:19]}")
    lines.append(f"  Test Season:       {results['test_season']}")
    lines.append(f"  Initial Bankroll:  ${results['initial_bankroll']:.2f}")
    lines.append(f"  Final Bankroll:    ${results['final_bankroll']:.2f}")
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
        lines.append("  WARNING: |ROI| > 15% — likely data leakage or overfitting")
    if s["win_rate"] > 0.60:
        lines.append("  WARNING: Win rate > 60% — suspiciously high")
    if s["total_trades"] < 50:
        lines.append("  WARNING: < 50 trades — insufficient sample size")

    lines.append("")
    lines.append("--- BY PROP TYPE " + "-" * 62)
    lines.append(
        f"  {'Prop':<10} {'Trades':>7} {'W-L':>10} {'Win%':>7} "
        f"{'P&L':>10} {'ROI':>8} {'AvgEdge':>8} {'AvgConf':>8}"
    )
    lines.append(f"  {'-' * 10} {'-' * 7} {'-' * 10} {'-' * 7} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8}")
    for pt, p in results["by_prop_type"].items():
        wl = f"{p['wins']}-{p['losses']}"
        lines.append(
            f"  {pt:<10} {p['trades']:>7} {wl:>10} {p['win_rate']:>6.1%} "
            f"${p['total_pnl']:>+9,.2f} {p['roi']:>+7.2%} "
            f"{p['avg_edge']:>7.2f} {p['avg_confidence']:>7.1%}"
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
    lines.append("--- CAVEATS " + "-" * 67)
    for caveat in results.get("caveats", []):
        lines.append(f"  * {caveat}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bankroll chart
# ---------------------------------------------------------------------------
def _save_bankroll_chart(df: pd.DataFrame, daily_bankroll: dict[str, float]) -> None:
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

    # ── Bankroll curve ────────────────────────────────────────────────────
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
        "NBA-BETS Profitability Backtest — 2023-24 Season",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylabel("Bankroll ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # ── Daily P&L bars ────────────────────────────────────────────────────
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
    chart_path = os.path.join(OUTPUT_DIR, "bankroll_curve.png")
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

    parser = argparse.ArgumentParser(description="NBA-BETS Profitability Backtest")
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll (default: 1000)",
    )
    parser.add_argument(
        "--season",
        default="2023-24",
        help="Test season label (default: 2023-24)",
    )
    args = parser.parse_args()

    global INITIAL_BANKROLL, TEST_SEASON
    INITIAL_BANKROLL = args.bankroll
    TEST_SEASON = args.season

    results = run_backtest(args)

    if results and "error" not in results:
        logger.info("Backtest completed successfully!")
        return 0

    logger.error("Backtest failed!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
