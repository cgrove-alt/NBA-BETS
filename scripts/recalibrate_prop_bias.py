"""Recalibrate PROP_BIAS_CORRECTION from the latest OOS walk-forward backtest.

Reads `data/backtest_results/oos_walkforward_results.json`, extracts the
aggregate per-prop bias measured across the full prediction sample (not the
betting sample — selection bias makes the betting-sample bias unusable for
this purpose), and writes the negated values to
`models/calibration/prop_bias_correction.json`.

Run this after every model retrain so the additive bias corrections stay in
sync with current model behavior:

    python3 scripts/recalibrate_prop_bias.py

Use --dry-run to preview the changes without writing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OOS_PATH = _REPO_ROOT / 'data' / 'backtest_results' / 'oos_walkforward_results.json'
_OUT_PATH = _REPO_ROOT / 'models' / 'calibration' / 'prop_bias_correction.json'

# Props to track — extend if we add new prop types.
_PROPS = ['points', 'rebounds', 'assists', 'threes', 'pra']

# Sanity bounds: bias corrections outside this range likely indicate a
# broken backtest run (model totally miscalibrated) rather than real signal.
# We refuse to write values outside this range and keep the previous JSON.
_MAX_ABS_CORRECTION = 5.0
_MIN_N_PREDICTIONS = 1000  # require at least this many samples per prop


def _load_oos_results() -> dict:
    if not _OOS_PATH.exists():
        raise SystemExit(
            f"OOS walk-forward results not found at {_OOS_PATH}. "
            "Run nba_models/backtesting/oos_walkforward_backtest.py first."
        )
    with open(_OOS_PATH, encoding='utf-8') as f:
        return json.load(f)


def _compute_corrections(oos: dict) -> tuple[dict[str, float], dict[str, int]]:
    """Return (corrections, sample_sizes). Correction = -bias from aggregate."""
    aggregate = oos.get('aggregate', {})
    by_prop = aggregate.get('by_prop_type', {})

    corrections: dict[str, float] = {}
    sample_sizes: dict[str, int] = {}

    for prop in _PROPS:
        stats = by_prop.get(prop)
        if not stats:
            # Disabled or untracked prop — fall back to 0.0 (no correction)
            corrections[prop] = 0.0
            sample_sizes[prop] = 0
            continue

        bias = stats.get('bias')
        n_pred = int(stats.get('n_predictions', 0))

        if bias is None:
            corrections[prop] = 0.0
            sample_sizes[prop] = n_pred
            continue

        # Reject if sample too small — keep 0 correction (model has no data here)
        if n_pred < _MIN_N_PREDICTIONS:
            print(
                f"  WARN: {prop} has only {n_pred} predictions "
                f"(min {_MIN_N_PREDICTIONS}) — setting correction to 0.0"
            )
            corrections[prop] = 0.0
            sample_sizes[prop] = n_pred
            continue

        correction = -float(bias)

        # Sanity-check magnitude
        if abs(correction) > _MAX_ABS_CORRECTION:
            print(
                f"  WARN: {prop} correction {correction:+.3f} exceeds "
                f"sanity bound ±{_MAX_ABS_CORRECTION}. Refusing to apply — "
                f"check for backtest bugs. Setting to 0.0."
            )
            corrections[prop] = 0.0
        else:
            corrections[prop] = round(correction, 4)
        sample_sizes[prop] = n_pred

    return corrections, sample_sizes


def _load_existing() -> dict | None:
    if not _OUT_PATH.exists():
        return None
    try:
        with open(_OUT_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show the new corrections without writing the file.',
    )
    parser.add_argument(
        '--force-sign-flip', action='store_true',
        help=(
            'Write the JSON even when a sign-flip is detected. Without this '
            'flag, sign flips abort with exit code 2 so a downstream pipeline '
            'fails loudly instead of silently flipping model behavior.'
        ),
    )
    args = parser.parse_args(argv)

    oos = _load_oos_results()
    corrections, sample_sizes = _compute_corrections(oos)

    existing = _load_existing()
    existing_corr = existing.get('corrections', {}) if existing else {}

    print(f"OOS source:     {_OOS_PATH.relative_to(_REPO_ROOT)}")
    print(f"OOS run date:   {oos.get('run_date', '(unknown)')}")
    print(f"Windows:        {oos.get('n_windows', '?')}")
    print()
    print(f"{'prop':<10} {'old':>9} {'new':>9}   delta    n_preds  flag")
    print('-' * 60)
    sign_flips: list[str] = []
    for prop in _PROPS:
        old = existing_corr.get(prop, 0.0)
        new = corrections[prop]
        delta = new - old
        # A sign-flip (excluding cases where one side is zero) often signals
        # label corruption or backwards-loaded features from a bad retrain.
        # Flag it loudly so the operator catches it before re-deploying.
        is_sign_flip = (
            abs(old) > 0.1 and abs(new) > 0.1 and (old * new) < 0
        )
        flag = '!! SIGN FLIP' if is_sign_flip else ''
        if is_sign_flip:
            sign_flips.append(f"{prop}: {old:+.3f} → {new:+.3f}")
        print(
            f"{prop:<10} {old:+9.4f} {new:+9.4f}   {delta:+7.4f}   "
            f"{sample_sizes[prop]:>7}  {flag}"
        )

    if sign_flips:
        print()
        print("WARNING: bias correction sign flips detected for:")
        for entry in sign_flips:
            print(f"  - {entry}")
        print(
            "A sign flip after a retrain usually means the model is now "
            "biased in the opposite direction — investigate before applying. "
            "Common causes: label leakage, swapped train/test split, baseline "
            "computed against wrong stat. Use --dry-run to inspect; pass "
            "--force-sign-flip to write anyway."
        )
        if not getattr(args, 'force_sign_flip', False):
            print("\nNot writing JSON because of sign flips. Re-run with "
                  "--force-sign-flip to override.")
            return 2

    if args.dry_run:
        print("\nDry run — no file written.")
        return 0

    payload = {
        "_doc": (
            "Per-prop additive bias correction. Values are the NEGATIVE of "
            "the measured aggregate bias on the full prediction sample (not "
            "betting sample) from the most recent OOS walk-forward backtest. "
            "Applied as: z_score = (predicted_value + correction - line) / "
            "sigma. Regenerate with: python3 scripts/recalibrate_prop_bias.py"
        ),
        "_source": str(_OOS_PATH.relative_to(_REPO_ROOT)),
        "_source_run_date": oos.get('run_date'),
        "_aggregate_n_predictions_by_prop": sample_sizes,
        "_generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "_method": "negative of aggregate.by_prop_type.{prop}.bias",
        "corrections": corrections,
    }

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')

    print(f"\nWrote {_OUT_PATH.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
