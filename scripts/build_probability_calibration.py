"""
Build empirical probability calibration from backtest data.

For each enabled prop type, fits an isotonic regression mapping
raw model probability → actual hit rate. This corrects any remaining
nonlinear distortion after sigma+bias corrections.

Usage:
    python3 scripts/build_probability_calibration.py

Outputs:
    models/probability_calibrators/{prop_type}_isotonic.pkl
    models/probability_calibrators/{prop_type}_lookup.json
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nba_betting.constants import (
    PROP_BIAS_CORRECTION,
    PROP_STD_DEVS,
    DEFAULT_PROP_STD_DEV,
)


def load_backtest_predictions() -> list[dict]:
    """Load raw predictions from backtest archive."""
    backtest_path = PROJECT_ROOT / "archive" / "backtest_results_2025.json"
    if not backtest_path.exists():
        print(f"ERROR: Backtest file not found: {backtest_path}")
        sys.exit(1)
    with open(backtest_path) as f:
        data = json.load(f)
    return data.get("raw_predictions", [])


def compute_corrected_over_prob(
    predicted: float, actual: float, prop_type: str
) -> tuple[float, int]:
    """Compute corrected over_prob using new sigma+bias, and whether over hit.

    We use 'actual' as a proxy for the line (what the sportsbook would have set),
    since the backtest doesn't include the betting line. In practice, the line
    tracks the true mean closely, so actual serves as a reasonable proxy for
    calibrating the probability mapping.

    Actually, we use the predicted value as the model output and compute the
    z-score relative to a synthetic line. Since we want to calibrate the
    relationship between model probability and actual outcome, we use the
    player's season-level mean as a proxy for the line.

    Simplified approach: use the midpoint between predicted and actual as a
    rough line proxy, then compute over_prob.

    Even simpler: The model outputs a predicted value. The "line" in production
    comes from sportsbooks. For calibration we need (model_prob, outcome) pairs.
    We can reconstruct model_prob from the backtest using the corrected formula,
    treating 'predicted' as the model output and computing over_prob against
    varying hypothetical lines.

    Best approach for calibration: for each prediction, compute what over_prob
    WOULD have been at a range of lines near the actual, and record whether
    over actually hit. This creates more training data and better captures the
    full probability spectrum.
    """
    sigma = PROP_STD_DEVS.get(prop_type.lower(), DEFAULT_PROP_STD_DEV)
    bias_fix = PROP_BIAS_CORRECTION.get(prop_type.lower(), 0.0)

    # Use actual as the "line" — this is the ground truth value
    # Model predicts `predicted`, corrected to `predicted + bias_fix`
    # Z-score = (predicted + bias_fix - line) / sigma
    line = actual  # actual outcome serves as proxy line
    z_score = (predicted + bias_fix - line) / sigma
    over_prob = float(norm.cdf(z_score))

    # The outcome: did the actual exceed the line?
    # Since line == actual here, this is always 50/50 by construction.
    # This won't work — we need a different approach.

    # CORRECT APPROACH: Use percentile-based binning.
    # For each prediction, compute over_prob at a set of representative lines,
    # and check whether actual > line for each.
    return over_prob, 1 if actual > predicted else 0


def build_calibration_data(
    predictions: list[dict], prop_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build (model_prob, hit_over) arrays for isotonic regression.

    For each prediction, we generate multiple (prob, outcome) pairs by
    computing over_prob at several synthetic lines around the prediction.
    """
    model_probs = []
    outcomes = []

    sigma = PROP_STD_DEVS.get(prop_type.lower(), DEFAULT_PROP_STD_DEV)
    bias_fix = PROP_BIAS_CORRECTION.get(prop_type.lower(), 0.0)

    prop_preds = [p for p in predictions if p["prop_type"] == prop_type]
    if not prop_preds:
        return np.array([]), np.array([])

    # Collect all actuals to build a representative line distribution
    all_actuals = np.array([p["actual"] for p in prop_preds])
    # Use percentiles as synthetic lines
    percentile_lines = np.percentile(all_actuals, np.arange(10, 91, 5))

    for pred in prop_preds:
        predicted = pred["predicted"]
        actual = pred["actual"]

        # For each synthetic line, compute model probability and check outcome
        for line in percentile_lines:
            z_score = (predicted + bias_fix - line) / sigma
            model_prob = float(norm.cdf(z_score))
            hit_over = 1 if actual > line else 0

            # Skip probabilities at extreme ends (not useful for calibration)
            if 0.02 < model_prob < 0.98:
                model_probs.append(model_prob)
                outcomes.append(hit_over)

    return np.array(model_probs), np.array(outcomes)


def compute_ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 20) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_prob = probs[mask].mean()
        bin_outcome = outcomes[mask].mean()
        ece += mask.sum() * abs(bin_prob - bin_outcome)
    return ece / len(probs) if len(probs) > 0 else 0.0


def cross_validate_calibration(
    model_probs: np.ndarray,
    outcomes: np.ndarray,
    n_folds: int = 5,
) -> tuple[float, float]:
    """Run k-fold cross-validation on isotonic calibration.

    Returns (mean_held_out_ece, std_held_out_ece).
    """
    n = len(model_probs)
    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    fold_size = n // n_folds
    held_out_eces = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso.fit(model_probs[train_idx], outcomes[train_idx])

        calibrated_test = iso.predict(model_probs[test_idx])
        fold_ece = compute_ece(calibrated_test, outcomes[test_idx])
        held_out_eces.append(fold_ece)

    return float(np.mean(held_out_eces)), float(np.std(held_out_eces))


def main():
    print("=" * 60)
    print("Building Empirical Probability Calibration")
    print("=" * 60)

    predictions = load_backtest_predictions()
    print(f"Loaded {len(predictions)} raw predictions")

    output_dir = PROJECT_ROOT / "models" / "probability_calibrators"
    output_dir.mkdir(parents=True, exist_ok=True)

    enabled_props = ["points", "rebounds", "pra"]

    for prop_type in enabled_props:
        print(f"\n--- {prop_type.upper()} ---")

        model_probs, outcomes = build_calibration_data(predictions, prop_type)
        if len(model_probs) < 100:
            print(f"  SKIP: Only {len(model_probs)} data points (need 100+)")
            continue

        print(f"  Data points: {len(model_probs)}")

        # ECE before calibration (raw model probabilities)
        ece_before = compute_ece(model_probs, outcomes)
        print(f"  ECE before calibration: {ece_before:.4f}")

        # 5-fold cross-validation to measure out-of-sample improvement
        cv_mean_ece, cv_std_ece = cross_validate_calibration(model_probs, outcomes)
        print(f"  CV held-out ECE: {cv_mean_ece:.4f} ± {cv_std_ece:.4f}")

        # Only save calibrator if held-out ECE improves over raw
        if cv_mean_ece >= ece_before:
            print(
                f"  SKIP SAVE: Held-out ECE ({cv_mean_ece:.4f}) >= raw ECE ({ece_before:.4f}). "
                "Calibration does not generalize — would overfit."
            )
            continue

        print(
            f"  CV improvement: {(1 - cv_mean_ece / max(ece_before, 1e-9)) * 100:.1f}% "
            "(held-out, not in-sample)"
        )

        # Train final calibrator on full data for production use
        iso_reg = IsotonicRegression(
            y_min=0.01, y_max=0.99, out_of_bounds="clip"
        )
        iso_reg.fit(model_probs, outcomes)

        # Report in-sample ECE for reference (expected near-zero for isotonic)
        calibrated = iso_reg.predict(model_probs)
        ece_insample = compute_ece(calibrated, outcomes)
        print(f"  In-sample ECE: {ece_insample:.4f} (expected ~0 for isotonic)")

        # Save isotonic model
        pkl_path = output_dir / f"{prop_type}_isotonic.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(iso_reg, f)
        print(f"  Saved: {pkl_path}")

        # Save JSON lookup table at 1% increments
        lookup = {}
        for pct in range(1, 100):
            raw_prob = pct / 100.0
            cal_prob = float(iso_reg.predict([raw_prob])[0])
            lookup[str(pct)] = round(cal_prob, 4)

        json_path = output_dir / f"{prop_type}_lookup.json"
        with open(json_path, "w") as f:
            json.dump(lookup, f, indent=2)
        print(f"  Saved: {json_path}")

    print("\n" + "=" * 60)
    print("Calibration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
