#!/usr/bin/env python3
"""
CLV Weekly Report (Fix 3.3)

Generates a Closing Line Value report from tracked bets.
CLV is the #1 predictor of long-term sports betting profitability.

Integrates with:
  - nba_betting.edge.clv_analyzer.CLVAnalyzer
  - nba_betting.edge.clv_bridge (records opening odds)
  - nba_betting.odds.closing_odds_scheduler (captures closing odds)

Usage:
    PYTHONPATH=. python3 scripts/clv_report.py
    PYTHONPATH=. python3 scripts/clv_report.py --days 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = os.environ.get(
    "NBA_BETS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
sys.path.insert(0, ROOT)

logger = logging.getLogger(__name__)


def generate_clv_report(days: int = 7) -> dict:
    """Generate CLV report for the last N days."""
    from nba_betting.edge.clv_analyzer import CLVAnalyzer

    analyzer = CLVAnalyzer()
    summary = analyzer.get_clv_summary(days=days)

    if not summary:
        return {"error": "No CLV data available", "days": days}

    # Augment with sharpness assessment
    # is_model_sharp() returns (bool, explanation_str) tuple, no params
    sharp_result = analyzer.is_model_sharp()
    is_sharp = sharp_result[0] if isinstance(sharp_result, tuple) else sharp_result

    report = {
        "report_date": datetime.now().isoformat(),
        "period_days": days,
        "summary": summary,
        "is_model_sharp": is_sharp,
        "assessment": _assess_clv(summary, is_sharp),
    }

    return report


def _assess_clv(summary: dict, is_sharp: bool) -> str:
    """Generate human-readable CLV assessment."""
    total = summary.get("total_bets", 0)
    settled = summary.get("settled_bets", 0)
    avg_clv = summary.get("avg_clv", 0)
    positive_rate = summary.get("positive_clv_rate", 0)

    if total == 0:
        return "No bets tracked. Ensure CLV bridge is recording predictions."

    if settled < 20:
        return (
            f"Only {settled} settled bets with CLV data — need 20+ for "
            "meaningful analysis. Continue tracking."
        )

    lines = []
    if avg_clv > 0:
        lines.append(f"Positive avg CLV ({avg_clv:+.2f}%) — model gets better prices than close.")
    else:
        lines.append(f"Negative avg CLV ({avg_clv:+.2f}%) — model gets worse prices than close.")

    if positive_rate > 0.55:
        lines.append(f"Strong positive CLV rate ({positive_rate:.0%}) — consistently sharp.")
    elif positive_rate > 0.50:
        lines.append(f"Marginally positive CLV rate ({positive_rate:.0%}) — borderline sharp.")
    else:
        lines.append(f"Negative CLV rate ({positive_rate:.0%}) — model is not sharp.")

    # Per-prop breakdown
    by_prop = summary.get("clv_by_prop_type", {})
    if by_prop:
        sharp_props = [p for p, v in by_prop.items() if v.get("avg_clv", 0) > 0]
        if sharp_props:
            lines.append(f"Sharp on: {', '.join(sharp_props)}")
        weak_props = [p for p, v in by_prop.items() if v.get("avg_clv", 0) < 0]
        if weak_props:
            lines.append(f"Not sharp on: {', '.join(weak_props)}")

    # Win rate comparison
    wr_pos = summary.get("win_rate_positive_clv", 0)
    wr_neg = summary.get("win_rate_negative_clv", 0)
    if wr_pos > 0 and wr_neg > 0:
        lines.append(
            f"Win rate +CLV: {wr_pos:.0%} vs -CLV: {wr_neg:.0%} "
            f"(gap: {(wr_pos - wr_neg):.0%})"
        )

    if is_sharp:
        lines.append("VERDICT: Model is SHARP — continue live betting.")
    else:
        lines.append("VERDICT: Model is NOT sharp — paper trade only.")

    return " ".join(lines)


def print_report(report: dict) -> None:
    """Print formatted CLV report."""
    print("=" * 70)
    print(f"   CLV REPORT — Last {report.get('period_days', 7)} Days")
    print("=" * 70)

    if "error" in report:
        print(f"  {report['error']}")
        print("=" * 70)
        return

    s = report.get("summary", {})
    print(f"  Total bets tracked:    {s.get('total_bets', 0)}")
    print(f"  Settled with CLV:      {s.get('settled_bets', 0)}")
    print(f"  Average CLV:           {s.get('avg_clv', 0):+.2f}%")
    print(f"  Median CLV:            {s.get('median_clv', 0):+.2f}%")
    print(f"  Positive CLV rate:     {s.get('positive_clv_rate', 0):.0%}")
    print(f"  7-day CLV:             {s.get('clv_7d', 0):+.2f}%")
    print(f"  30-day CLV:            {s.get('clv_30d', 0):+.2f}%")
    print()

    # By prop type
    by_prop = s.get("clv_by_prop_type", {})
    if by_prop:
        print("  --- By Prop Type ---")
        print(f"  {'Prop':<12} {'Avg CLV':>8} {'Pos Rate':>9} {'Count':>6}")
        for prop, data in sorted(by_prop.items()):
            print(
                f"  {prop:<12} {data.get('avg_clv', 0):>+7.2f}% "
                f"{data.get('positive_rate', 0):>8.0%} "
                f"{data.get('count', 0):>6}"
            )
        print()

    # Sharp rating
    rating = s.get("sharp_rating", "unknown")
    print(f"  Sharp Rating:          {rating}")
    print(f"  Model Sharp:           {'YES' if report.get('is_model_sharp') else 'NO'}")
    print()

    print("  --- Assessment ---")
    print(f"  {report.get('assessment', 'N/A')}")
    print("=" * 70)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="CLV Weekly Report")
    parser.add_argument("--days", type=int, default=7, help="Look-back period in days")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    report = generate_clv_report(args.days)
    print_report(report)

    # Save JSON
    out_path = args.output or os.path.join(ROOT, "data", "backtest_results", "clv_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
