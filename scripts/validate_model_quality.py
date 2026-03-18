#!/usr/bin/env python3
"""
Model Quality Gate: Validates retrained models meet minimum quality thresholds.

Run after every retrain. If any enabled prop type fails, the model should NOT
be deployed for live betting.

Thresholds:
  - R² > 0.02 (must beat the mean)
  - RMSE < season-average RMSE (must beat simplest baseline)
  - |Bias| < 1.0 (predictions not systematically off)

Usage:
    PYTHONPATH=. python3 scripts/validate_model_quality.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

ROOT = os.environ.get(
    "NBA_BETS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "nba_models", "training"))
sys.path.insert(0, ROOT)

from nba_betting.constants import DISABLED_PROPS

logger = logging.getLogger(__name__)

# Quality thresholds
MIN_R2 = 0.02
MAX_ABS_BIAS = 1.0
MIN_SAMPLES = 100


def validate(results_path: str | None = None) -> tuple[bool, list[str]]:
    """Validate model quality from baseline comparison results.

    Returns:
        (passed, messages) — True if all enabled props pass quality gates.
    """
    path = results_path or os.path.join(ROOT, "data", "backtest_results", "baseline_comparison.json")
    if not os.path.exists(path):
        return False, [f"Baseline comparison results not found at {path}. Run baseline_comparison.py first."]

    with open(path) as f:
        results = json.load(f)

    messages = []
    all_passed = True

    for prop_type, data in results.items():
        if prop_type in DISABLED_PROPS:
            messages.append(f"  {prop_type}: SKIPPED (disabled)")
            continue

        model = data.get("model", {})
        sa = data.get("season_average", {})
        n = model.get("n", 0)

        if n < MIN_SAMPLES:
            messages.append(f"  {prop_type}: SKIP (only {n} samples, need {MIN_SAMPLES})")
            continue

        r2 = model.get("r2", -999)
        rmse = model.get("rmse", 999)
        bias = model.get("bias", 999)
        sa_rmse = sa.get("rmse", 999)
        beats = data.get("beats_season_avg", False)

        failures = []
        if r2 < MIN_R2:
            failures.append(f"R²={r2:.4f} < {MIN_R2}")
        if not beats:
            failures.append(f"RMSE={rmse:.3f} >= season_avg RMSE={sa_rmse:.3f}")
        if abs(bias) > MAX_ABS_BIAS:
            failures.append(f"|bias|={abs(bias):.3f} > {MAX_ABS_BIAS}")

        if failures:
            all_passed = False
            messages.append(f"  {prop_type}: FAIL — {'; '.join(failures)}")
        else:
            messages.append(f"  {prop_type}: PASS (R²={r2:.4f}, RMSE={rmse:.3f}, bias={bias:+.3f})")

    return all_passed, messages


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    passed, messages = validate()

    print("=" * 60)
    print("   MODEL QUALITY GATE")
    print("=" * 60)
    for msg in messages:
        print(msg)
    print("=" * 60)

    if passed:
        print("RESULT: PASS — model meets quality thresholds")
        return 0
    else:
        print("RESULT: FAIL — do NOT deploy this model")
        return 1


if __name__ == "__main__":
    sys.exit(main())
