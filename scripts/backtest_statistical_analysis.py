#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis of Backtest Results

Reads the real-lines trade log and computes:
1. Bootstrap confidence intervals on ROI, win rate, Sharpe
2. P-value: probability that true ROI ≤ 0
3. Per-player P&L concentration (Herfindahl index)
4. Temporal stability (monthly CIs, rolling performance, regime detection)

Usage:
    python scripts/backtest_statistical_analysis.py
    python scripts/backtest_statistical_analysis.py --trade-log data/backtest_results/real_lines_trade_log.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------

def bootstrap_statistic(
    data: np.ndarray,
    stat_fn,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        stat_fn: Function that computes the statistic (e.g., np.mean)
        n_resamples: Number of bootstrap resamples
        ci: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Dict with: observed, ci_lower, ci_upper, std_error, p_value_positive
    """
    rng = np.random.RandomState(seed)
    observed = float(stat_fn(data))
    n = len(data)

    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = data[rng.randint(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)

    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(boot_stats, alpha * 100))
    ci_upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    std_error = float(np.std(boot_stats))

    # P-value: fraction of bootstrap samples with stat ≤ 0
    p_value = float(np.mean(boot_stats <= 0))

    return {
        "observed": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
        "p_value_positive": p_value,
    }


# ---------------------------------------------------------------------------
# ROI computation
# ---------------------------------------------------------------------------

def compute_roi(pnl: np.ndarray, bet_sizes: np.ndarray) -> float:
    """Compute ROI = total_pnl / total_wagered."""
    total_wagered = bet_sizes.sum()
    if total_wagered == 0:
        return 0.0
    return float(pnl.sum() / total_wagered)


def compute_sharpe(daily_pnl: np.ndarray) -> float:
    """Compute annualized Sharpe ratio from daily P&L."""
    if len(daily_pnl) < 2 or daily_pnl.std() == 0:
        return 0.0
    return float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(trade_log_path: str) -> dict:
    """Run full statistical analysis on trade log."""
    with open(trade_log_path) as f:
        trades = json.load(f)

    if not trades:
        print("ERROR: No trades in log")
        return {}

    n = len(trades)
    print(f"Loaded {n} trades from {trade_log_path}")
    print()

    pnl = np.array([t["pnl"] for t in trades])
    bet_sizes = np.array([t["bet_size"] for t in trades])
    won = np.array([t["won"] for t in trades], dtype=float)
    prop_types = [t["prop_type"] for t in trades]

    results = {"n_trades": n}

    # ===================================================================
    # 1. BOOTSTRAP CONFIDENCE INTERVALS
    # ===================================================================
    print("=" * 70)
    print("  BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples)")
    print("=" * 70)
    print()

    # ROI bootstrap
    # Resample trade indices, compute ROI on each resample
    def roi_stat(indices):
        return compute_roi(pnl[indices], bet_sizes[indices])

    indices = np.arange(n)
    roi_boot = bootstrap_statistic(
        indices,
        roi_stat,
        n_resamples=10000,
    )
    results["roi"] = roi_boot

    print(f"  ROI:       {roi_boot['observed']:+.2%}  "
          f"95% CI: [{roi_boot['ci_lower']:+.2%}, {roi_boot['ci_upper']:+.2%}]  "
          f"p(ROI≤0): {roi_boot['p_value_positive']:.3f}")

    # Win rate bootstrap
    wr_boot = bootstrap_statistic(won, np.mean, n_resamples=10000)
    results["win_rate"] = wr_boot

    print(f"  Win Rate:  {wr_boot['observed']:.1%}  "
          f"95% CI: [{wr_boot['ci_lower']:.1%}, {wr_boot['ci_upper']:.1%}]  "
          f"p(WR≤50%): {float(np.mean(np.array([np.mean(won[np.random.RandomState(42+i).randint(0, n, n)]) for i in range(10000)]) <= 0.5)):.3f}")

    # Sharpe bootstrap
    daily_pnl_map = defaultdict(float)
    for t in trades:
        daily_pnl_map[t["date"]] += t["pnl"]
    daily_pnl_arr = np.array(list(daily_pnl_map.values()))

    sharpe_boot = bootstrap_statistic(
        daily_pnl_arr, compute_sharpe, n_resamples=10000
    )
    results["sharpe"] = sharpe_boot

    print(f"  Sharpe:    {sharpe_boot['observed']:.2f}  "
          f"95% CI: [{sharpe_boot['ci_lower']:.2f}, {sharpe_boot['ci_upper']:.2f}]")

    # Breakeven probability
    breakeven_odds = -107  # Average odds from the backtest
    breakeven_wr = abs(breakeven_odds) / (abs(breakeven_odds) + 100) if breakeven_odds < 0 else 100 / (breakeven_odds + 100)
    p_above_breakeven = float(np.mean(
        np.array([np.mean(won[np.random.RandomState(42 + i).randint(0, n, n)])
                  for i in range(10000)]) > breakeven_wr
    ))
    print(f"  P(WR > breakeven {breakeven_wr:.1%}): {p_above_breakeven:.3f}")
    results["p_above_breakeven"] = p_above_breakeven

    print()

    # ===================================================================
    # 2. PER-PLAYER CONCENTRATION
    # ===================================================================
    print("=" * 70)
    print("  PER-PLAYER P&L CONCENTRATION")
    print("=" * 70)
    print()

    player_pnl = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
    for t in trades:
        p = t["player"]
        player_pnl[p]["pnl"] += t["pnl"]
        player_pnl[p]["trades"] += 1
        if t["won"]:
            player_pnl[p]["wins"] += 1

    sorted_players = sorted(player_pnl.items(), key=lambda x: -x[1]["pnl"])
    total_pnl = pnl.sum()

    print("  Top 10 Winners:")
    for rank, (name, stats) in enumerate(sorted_players[:10], 1):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        pct = stats["pnl"] / abs(total_pnl) * 100 if total_pnl != 0 else 0
        print(f"    {rank:2d}. {name:<25s}  ${stats['pnl']:+8.2f}  "
              f"{stats['trades']:3d} trades  {wr:.0f}% WR  ({pct:+.1f}% of total)")

    print()
    print("  Top 10 Losers:")
    for rank, (name, stats) in enumerate(sorted_players[-10:], 1):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        print(f"    {rank:2d}. {name:<25s}  ${stats['pnl']:+8.2f}  "
              f"{stats['trades']:3d} trades  {wr:.0f}% WR")

    # Herfindahl index of P&L concentration
    abs_pnls = np.array([abs(s["pnl"]) for s in player_pnl.values()])
    if abs_pnls.sum() > 0:
        shares = abs_pnls / abs_pnls.sum()
        hhi = float(np.sum(shares ** 2))
    else:
        hhi = 0.0

    # Concentration metrics
    top_5_pnl = sum(s["pnl"] for _, s in sorted_players[:5])
    top_10_pnl = sum(s["pnl"] for _, s in sorted_players[:10])
    top_20_pnl = sum(s["pnl"] for _, s in sorted_players[:20])

    print()
    print(f"  Unique players: {len(player_pnl)}")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    print(f"  Top 5 players:  ${top_5_pnl:+.2f} ({top_5_pnl / abs(total_pnl) * 100 if total_pnl else 0:+.1f}% of total)")
    print(f"  Top 10 players: ${top_10_pnl:+.2f} ({top_10_pnl / abs(total_pnl) * 100 if total_pnl else 0:+.1f}% of total)")
    print(f"  Top 20 players: ${top_20_pnl:+.2f} ({top_20_pnl / abs(total_pnl) * 100 if total_pnl else 0:+.1f}% of total)")
    print(f"  Herfindahl Index (HHI): {hhi:.4f}", end="")
    if hhi < 0.05:
        print(" (DIVERSE — edge is broadly spread)")
    elif hhi < 0.15:
        print(" (MODERATE concentration)")
    else:
        print(" (CONCENTRATED — edge depends on few players)")

    results["concentration"] = {
        "unique_players": len(player_pnl),
        "hhi": round(hhi, 4),
        "top_5_pnl_pct": round(top_5_pnl / abs(total_pnl) * 100, 1) if total_pnl else 0,
        "top_10_pnl_pct": round(top_10_pnl / abs(total_pnl) * 100, 1) if total_pnl else 0,
        "top_20_pnl_pct": round(top_20_pnl / abs(total_pnl) * 100, 1) if total_pnl else 0,
    }

    print()

    # ===================================================================
    # 3. TEMPORAL STABILITY
    # ===================================================================
    print("=" * 70)
    print("  TEMPORAL STABILITY")
    print("=" * 70)
    print()

    # Monthly breakdown with bootstrap CI
    monthly = defaultdict(lambda: {"pnl": [], "won": [], "bet_sizes": []})
    for t in trades:
        month = t["date"][:7]  # YYYY-MM
        monthly[month]["pnl"].append(t["pnl"])
        monthly[month]["won"].append(float(t["won"]))
        monthly[month]["bet_sizes"].append(t["bet_size"])

    print(f"  {'Month':<10s}  {'Trades':>6s}  {'Win%':>6s}  {'ROI':>8s}  {'95% CI ROI':>20s}  {'P&L':>10s}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*20}  {'-'*10}")

    monthly_rois = []
    monthly_results = {}
    for month in sorted(monthly.keys()):
        m = monthly[month]
        m_pnl = np.array(m["pnl"])
        m_won = np.array(m["won"])
        m_bets = np.array(m["bet_sizes"])
        m_n = len(m_pnl)

        m_roi = compute_roi(m_pnl, m_bets)
        m_wr = np.mean(m_won)
        monthly_rois.append(m_roi)

        # Bootstrap CI on monthly ROI
        m_indices = np.arange(m_n)
        def m_roi_fn(idx, _pnl=m_pnl, _bets=m_bets):
            return compute_roi(_pnl[idx], _bets[idx])
        m_boot = bootstrap_statistic(m_indices, m_roi_fn, n_resamples=5000)

        print(f"  {month:<10s}  {m_n:>6d}  {m_wr:>5.1%}  {m_roi:>+7.2%}  "
              f"[{m_boot['ci_lower']:>+7.2%}, {m_boot['ci_upper']:>+7.2%}]  "
              f"${m_pnl.sum():>+9.2f}")

        monthly_results[month] = {
            "trades": m_n,
            "win_rate": round(float(m_wr), 4),
            "roi": round(m_roi, 4),
            "roi_ci_lower": round(m_boot["ci_lower"], 4),
            "roi_ci_upper": round(m_boot["ci_upper"], 4),
            "pnl": round(float(m_pnl.sum()), 2),
        }

    results["monthly"] = monthly_results

    # T-test: is mean monthly ROI significantly > 0?
    monthly_rois_arr = np.array(monthly_rois)
    if len(monthly_rois_arr) > 1 and monthly_rois_arr.std() > 0:
        t_stat = monthly_rois_arr.mean() / (monthly_rois_arr.std() / np.sqrt(len(monthly_rois_arr)))
        # One-sided p-value from t-distribution approximation
        # For small samples, use bootstrap instead
        monthly_mean_boot = bootstrap_statistic(monthly_rois_arr, np.mean, n_resamples=10000)
        print()
        print(f"  Mean monthly ROI: {monthly_rois_arr.mean():+.2%} ± {monthly_rois_arr.std():.2%}")
        print(f"  T-statistic: {t_stat:.2f}")
        print(f"  P(mean ROI ≤ 0): {monthly_mean_boot['p_value_positive']:.3f}")
        results["monthly_t_stat"] = round(float(t_stat), 2)
        results["monthly_p_value"] = monthly_mean_boot["p_value_positive"]

    # Rolling 30-trade performance
    print()
    print("  Rolling 30-trade windows:")
    window_size = 30
    if n >= window_size:
        rolling_wrs = []
        rolling_rois = []
        for start in range(0, n - window_size + 1, window_size // 2):
            end = start + window_size
            w_pnl = pnl[start:end]
            w_won = won[start:end]
            w_bets = bet_sizes[start:end]
            r_wr = float(w_won.mean())
            r_roi = compute_roi(w_pnl, w_bets)
            rolling_wrs.append(r_wr)
            rolling_rois.append(r_roi)

        rolling_wrs = np.array(rolling_wrs)
        rolling_rois = np.array(rolling_rois)
        pct_positive_windows = float(np.mean(rolling_rois > 0))

        print(f"    Windows: {len(rolling_wrs)}")
        print(f"    WR range: {rolling_wrs.min():.1%} — {rolling_wrs.max():.1%}")
        print(f"    ROI range: {rolling_rois.min():+.1%} — {rolling_rois.max():+.1%}")
        print(f"    % windows with positive ROI: {pct_positive_windows:.1%}")

        results["rolling"] = {
            "n_windows": len(rolling_wrs),
            "wr_min": round(float(rolling_wrs.min()), 4),
            "wr_max": round(float(rolling_wrs.max()), 4),
            "roi_min": round(float(rolling_rois.min()), 4),
            "roi_max": round(float(rolling_rois.max()), 4),
            "pct_positive_windows": round(pct_positive_windows, 4),
        }

    # Regime detection: first half vs second half
    print()
    half = n // 2
    first_half_roi = compute_roi(pnl[:half], bet_sizes[:half])
    second_half_roi = compute_roi(pnl[half:], bet_sizes[half:])
    first_half_wr = float(won[:half].mean())
    second_half_wr = float(won[half:].mean())

    print(f"  First half ({half} trades):  ROI={first_half_roi:+.2%}, WR={first_half_wr:.1%}")
    print(f"  Second half ({n - half} trades): ROI={second_half_roi:+.2%}, WR={second_half_wr:.1%}")

    if second_half_roi > first_half_roi:
        print("  Trend: IMPROVING (second half outperforms first)")
    elif second_half_roi < first_half_roi - 0.03:
        print("  Trend: DEGRADING (second half underperforms by 3%+)")
    else:
        print("  Trend: STABLE")

    results["regime"] = {
        "first_half_roi": round(first_half_roi, 4),
        "second_half_roi": round(second_half_roi, 4),
        "first_half_wr": round(first_half_wr, 4),
        "second_half_wr": round(second_half_wr, 4),
    }

    # ===================================================================
    # 4. PER-PROP BREAKDOWN
    # ===================================================================
    print()
    print("=" * 70)
    print("  PER-PROP TYPE BOOTSTRAP")
    print("=" * 70)
    print()

    for pt in sorted(set(prop_types)):
        pt_mask = np.array([t["prop_type"] == pt for t in trades])
        pt_pnl = pnl[pt_mask]
        pt_bets = bet_sizes[pt_mask]
        pt_won = won[pt_mask]
        pt_n = pt_mask.sum()

        if pt_n < 10:
            print(f"  {pt}: {pt_n} trades (insufficient for bootstrap)")
            continue

        pt_indices = np.arange(pt_n)
        def pt_roi_fn(idx, _pnl=pt_pnl, _bets=pt_bets):
            return compute_roi(_pnl[idx], _bets[idx])

        pt_boot = bootstrap_statistic(pt_indices, pt_roi_fn, n_resamples=10000)

        print(f"  {pt:<12s}  {pt_n:>4d} trades  "
              f"ROI={pt_boot['observed']:+.2%}  "
              f"95% CI: [{pt_boot['ci_lower']:+.2%}, {pt_boot['ci_upper']:+.2%}]  "
              f"p(ROI≤0): {pt_boot['p_value_positive']:.3f}  "
              f"WR={pt_won.mean():.1%}")

        results[f"prop_{pt}"] = {
            "n_trades": int(pt_n),
            "roi": pt_boot,
            "win_rate": round(float(pt_won.mean()), 4),
        }

    print()

    # ===================================================================
    # VERDICT
    # ===================================================================
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    issues = []
    strengths = []

    if roi_boot["p_value_positive"] < 0.05:
        strengths.append(f"ROI is statistically significant (p={roi_boot['p_value_positive']:.3f})")
    else:
        issues.append(f"ROI is NOT statistically significant (p={roi_boot['p_value_positive']:.3f})")

    if roi_boot["ci_lower"] > 0:
        strengths.append(f"95% CI for ROI is entirely positive: [{roi_boot['ci_lower']:+.2%}, {roi_boot['ci_upper']:+.2%}]")
    else:
        issues.append(f"95% CI for ROI includes zero: [{roi_boot['ci_lower']:+.2%}, {roi_boot['ci_upper']:+.2%}]")

    if hhi < 0.05:
        strengths.append(f"Edge is diversified across players (HHI={hhi:.4f})")
    elif hhi > 0.15:
        issues.append(f"Edge is concentrated in few players (HHI={hhi:.4f})")

    if "rolling" in results and results["rolling"]["pct_positive_windows"] > 0.6:
        strengths.append(f"{results['rolling']['pct_positive_windows']:.0%} of rolling windows are profitable")
    elif "rolling" in results:
        issues.append(f"Only {results['rolling']['pct_positive_windows']:.0%} of rolling windows are profitable")

    for s in strengths:
        print(f"  ✓ {s}")
    for i in issues:
        print(f"  ✗ {i}")

    print()

    # Save results
    out_path = os.path.join(ROOT, "data", "backtest_results", "statistical_analysis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest Statistical Analysis")
    parser.add_argument(
        "--trade-log",
        default=os.path.join(ROOT, "data", "backtest_results", "real_lines_trade_log.json"),
        help="Path to trade log JSON",
    )
    args = parser.parse_args()

    if not os.path.exists(args.trade_log):
        print(f"ERROR: Trade log not found: {args.trade_log}")
        print("Run the real-lines backtest first:")
        print("  PYTHONPATH=. python3 nba_models/backtesting/real_lines_backtest.py --oos")
        return 1

    run_analysis(args.trade_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
