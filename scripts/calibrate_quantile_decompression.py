"""
Auto-calibrate quantile decompression constants after model retraining.

This script MUST be run after every model retrain to update the
quantile decompression constants used in daily_predictions.py.

It measures the empirical relationship between quantile model predictions
and the actual prop lines (regression-to-mean compression) using back-
test / calibration data stored in the local SQLite or PostgreSQL database.

Usage:
    python3 scripts/calibrate_quantile_decompression.py
    python3 scripts/calibrate_quantile_decompression.py --dry-run   # print only
    python3 scripts/calibrate_quantile_decompression.py --min-samples 100

Output:
    models/quantile_decompression.json  (also printed to stdout)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Canonical defaults — used when calibration data is insufficient
from nba_betting.constants import (
    QUANTILE_DECOMPRESSION_DEFAULTS,
    QUANTILE_TARGET_SLOPE,
    DEFAULT_PROP_STD_DEV,
)

PROP_TYPES = ['points', 'rebounds', 'assists', 'threes', 'pra']
OUTPUT_PATH = Path("models/quantile_decompression.json")
MIN_SAMPLES_DEFAULT = 50


def _load_calibration_rows(prop_type: str, min_date: str = "2024-10-01") -> list[dict]:
    """Load prediction rows with actual outcomes from the database.

    Returns list of dicts: {predicted_value, actual_value, line, prop_type}.
    """
    rows = []

    # Try PostgreSQL first
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url:
        try:
            import psycopg2
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT predicted_value, actual_value, line
                FROM predictions_history
                WHERE prop_type = %s
                  AND actual_value IS NOT NULL
                  AND predicted_value IS NOT NULL
                  AND line IS NOT NULL
                  AND game_date >= %s
                ORDER BY game_date DESC
                LIMIT 2000
                """,
                (prop_type, min_date),
            )
            for predicted, actual, line in cur.fetchall():
                rows.append({'predicted': float(predicted), 'actual': float(actual), 'line': float(line)})
            cur.close()
            conn.close()
            logger.info("  Loaded %d rows for '%s' from PostgreSQL", len(rows), prop_type)
            return rows
        except Exception as exc:
            logger.warning("  PostgreSQL load failed for '%s': %s", prop_type, exc)

    # Fallback: local SQLite
    for db_path in ["prop_predictions.db", "data/calibration.db", "nba_betting.db"]:
        p = Path(db_path)
        if not p.exists():
            continue
        try:
            import sqlite3
            conn = sqlite3.connect(p)
            cur = conn.cursor()
            # Try multiple column name conventions
            for table in ("prop_predictions", "predictions_history", "calibration_records"):
                try:
                    cur.execute(
                        f"""
                        SELECT predicted_value, actual_value, line
                        FROM {table}
                        WHERE prop_type = ?
                          AND actual_value IS NOT NULL
                          AND predicted_value IS NOT NULL
                          AND line IS NOT NULL
                          AND game_date >= ?
                        ORDER BY game_date DESC
                        LIMIT 2000
                        """,
                        (prop_type, min_date),
                    )
                    for predicted, actual, line in cur.fetchall():
                        rows.append({'predicted': float(predicted), 'actual': float(actual), 'line': float(line)})
                    if rows:
                        break
                except sqlite3.OperationalError:
                    continue
            conn.close()
            if rows:
                logger.info("  Loaded %d rows for '%s' from %s", len(rows), prop_type, db_path)
                return rows
        except Exception as exc:
            logger.warning("  SQLite load from %s failed: %s", db_path, exc)

    return rows


def _compute_decompression_params(
    rows: list[dict],
    prop_type: str,
    min_samples: int = MIN_SAMPLES_DEFAULT,
) -> dict:
    """Compute slope (regression-to-mean) and mean_gap (level bias) for a prop type.

    Uses linear regression of predicted_median on prop_line to measure compression.
    Uses mean residual (actual - predicted) as the level bias.

    Returns a dict compatible with QUANTILE_DECOMPRESSION_DEFAULTS.
    """
    if len(rows) < min_samples:
        logger.warning(
            "  Insufficient data for '%s' (%d rows, need %d). Using canonical defaults.",
            prop_type, len(rows), min_samples,
        )
        return QUANTILE_DECOMPRESSION_DEFAULTS[prop_type].copy()

    predictions = np.array([r['predicted'] for r in rows])
    actuals     = np.array([r['actual']    for r in rows])
    lines       = np.array([r['line']      for r in rows])

    # Outlier removal (> 4 std devs from mean actual)
    actual_std = np.std(actuals)
    actual_mean = np.mean(actuals)
    mask = np.abs(actuals - actual_mean) < 4 * actual_std
    predictions, actuals, lines = predictions[mask], actuals[mask], lines[mask]

    if len(predictions) < min_samples:
        logger.warning(
            "  After outlier removal, insufficient data for '%s' (%d rows). Using defaults.",
            prop_type, len(predictions),
        )
        return QUANTILE_DECOMPRESSION_DEFAULTS[prop_type].copy()

    # Regression of predicted on line: predicted ≈ slope * line + intercept
    # slope < 1.0 means regression to mean (high-line players under-predicted)
    if np.std(lines) < 0.01:
        slope = 1.0
    else:
        slope, _ = np.polyfit(lines, predictions, 1)
        slope = float(slope)

    # Load the currently-live mean_gap so we can undo the level fix before
    # measuring residuals.  Stored predictions = raw_model_output - old_mean_gap
    # (because level_fix = -mean_gap is added during decompress).  Adding
    # old_mean_gap back recovers the raw quantile model output, breaking the
    # circular dependency where re-calibration always saw the post-correction
    # residual and converged to the wrong value each retrain cycle.
    try:
        with open(OUTPUT_PATH) as _f:
            _cur = json.load(_f)
        _old_mean_gap = float(_cur.get(prop_type, {}).get('mean_gap', 0.0))
    except (FileNotFoundError, json.JSONDecodeError):
        _old_mean_gap = 0.0
    raw_predictions = predictions + _old_mean_gap

    # mean_gap sign convention: positive = model over-predicts (raw > actual).
    # level_fix = -mean_gap therefore subtracts from predictions to correct
    # overestimation.  Old convention (actual - predicted) had the wrong sign
    # and caused level_fix to amplify over-prediction instead of damping it.
    residuals = raw_predictions - actuals
    mean_gap = float(np.mean(residuals))

    # Median line (needed for slope correction formula in decompress_quantile_prediction)
    mean_line = float(np.median(lines))

    logger.info(
        "  '%s': n=%d  slope=%.3f  mean_gap=%+.3f  mean_line=%.1f",
        prop_type, len(predictions), slope, mean_gap, mean_line,
    )

    return {
        'slope': round(slope, 4),
        'mean_gap': round(mean_gap, 4),
        'mean_line': round(mean_line, 2),
        '_n': len(predictions),
        '_computed_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    }


def run_full_calibration(min_samples: int = MIN_SAMPLES_DEFAULT, dry_run: bool = False) -> dict:
    """Compute and save quantile decompression constants for all prop types.

    Args:
        min_samples: Minimum rows required per prop type before trusting computed params.
        dry_run: If True, print results but do NOT write to disk.

    Returns:
        Dict mapping prop_type → params.
    """
    output: dict = {}
    any_from_data = False

    for prop_type in PROP_TYPES:
        logger.info("Calibrating '%s'...", prop_type)
        rows = _load_calibration_rows(prop_type)
        params = _compute_decompression_params(rows, prop_type, min_samples)
        output[prop_type] = params
        if '_n' in params:
            any_from_data = True

    if not any_from_data:
        logger.warning(
            "No calibration data found in any database. "
            "Output will use canonical defaults unchanged. "
            "Run after at least one prediction cycle with outcome settlement."
        )

    output['_comment'] = "Quantile decompression constants. Regenerate after each retrain."
    output['_last_updated'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    if dry_run:
        print(json.dumps(output, indent=2))
        logger.info("Dry run — NOT writing to disk.")
        return output

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Safety: never write stale placeholder values (slope=0.7 for ALL props)
    real_slopes = [output[p]['slope'] for p in PROP_TYPES if p in output]
    if real_slopes and len({round(s, 2) for s in real_slopes}) == 1:
        logger.warning(
            "All slopes identical (%.2f) — this looks like uninitialized defaults. "
            "Preserving existing file if present.",
            real_slopes[0],
        )
        if OUTPUT_PATH.exists():
            logger.info("Keeping existing %s unchanged.", OUTPUT_PATH)
            return output

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info("Saved to %s", OUTPUT_PATH)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate quantile decompression constants")
    parser.add_argument("--dry-run", action="store_true", help="Print results without saving")
    parser.add_argument(
        "--min-samples", type=int, default=MIN_SAMPLES_DEFAULT,
        help=f"Minimum rows per prop type (default {MIN_SAMPLES_DEFAULT})",
    )
    args = parser.parse_args()

    result = run_full_calibration(min_samples=args.min_samples, dry_run=args.dry_run)
    print("\nFinal decompression constants:")
    for prop, params in result.items():
        if prop.startswith('_'):
            continue
        print(f"  {prop:10s}  slope={params.get('slope', '?'):.4f}  "
              f"mean_gap={params.get('mean_gap', '?'):+.4f}  "
              f"mean_line={params.get('mean_line', '?'):.1f}"
              + (f"  (n={params['_n']})" if '_n' in params else "  (default)"))
