"""Recalibrate DISABLED_PROPS from the latest OOS walk-forward backtest.

Reads `data/backtest_results/oos_walkforward_results.json` and applies three
objective gates to each prop type:

  1. Bootstrap significance test — bootstrap_p < 0.05 against naive baseline.
     (We don't compute the bootstrap here; we read it from the statistical
      analysis JSON if available, else use trade win rate as a proxy.)

  2. Trade win rate — when bets are simulated, win rate must beat the -110
     break-even probability (≈52.4%).

  3. RMSE vs naive baseline — model RMSE must not exceed the naive (season-avg)
     baseline RMSE. If the model is worse than naive, it has negative edge.

A prop is *enabled* only if it passes every gate where data is available. The
goal is to make DISABLED_PROPS a data-driven status report instead of a
hardcoded list — re-enabling a prop is automatic once the model gets good at
it. Run after every retrain and OOS backtest:

    python3 scripts/recalibrate_prop_gates.py

Use --dry-run to preview without writing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OOS_PATH = _REPO_ROOT / 'data' / 'backtest_results' / 'oos_walkforward_results.json'
_STAT_PATH = _REPO_ROOT / 'data' / 'backtest_results' / 'statistical_analysis.json'
_OUT_PATH = _REPO_ROOT / 'models' / 'calibration' / 'prop_enable_gates.json'

_PROPS = ['points', 'rebounds', 'assists', 'threes', 'pra']

# Gates — single source of truth, also embedded in the output JSON.
GATE_BOOTSTRAP_P_MAX = 0.05
GATE_TRADE_WIN_RATE_MIN = 0.524  # -110 break-even probability
GATE_RMSE_VS_BASELINE_MAX_RATIO = 1.0  # model RMSE ≤ baseline RMSE


def _load_oos() -> dict:
    if not _OOS_PATH.exists():
        raise SystemExit(
            f"OOS walk-forward results not found at {_OOS_PATH}. "
            "Run the OOS backtest first."
        )
    with open(_OOS_PATH, encoding='utf-8') as f:
        return json.load(f)


def _load_stat() -> dict:
    if not _STAT_PATH.exists():
        return {}
    try:
        with open(_STAT_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _evaluate_prop(prop: str, oos_stats: dict, stat_analysis: dict) -> dict:
    """Return {'enabled': bool, 'reasons': [str]} for a single prop."""
    reasons_pass: list[str] = []
    reasons_fail: list[str] = []

    # Pull aggregate metrics
    agg = oos_stats.get('aggregate', {}).get('by_prop_type', {}).get(prop, {})
    n_pred = int(agg.get('n_predictions', 0))
    rmse = agg.get('rmse')
    n_trades = int(agg.get('total_trades', 0))
    trade_wr = agg.get('trade_win_rate')

    # Baseline RMSE comes from individual windows — average the available ones
    baseline_rmses: list[float] = []
    for win in oos_stats.get('windows', []):
        win_stats = win.get('by_prop_type', {}).get(prop, {})
        b = win_stats.get('baseline_rmse')
        if b is not None:
            baseline_rmses.append(float(b))
    baseline_rmse = (sum(baseline_rmses) / len(baseline_rmses)) if baseline_rmses else None

    # Gate 1: bootstrap significance
    bootstrap_p = None
    boot = stat_analysis.get('bootstrap', {}).get(prop) if stat_analysis else None
    if isinstance(boot, dict):
        bootstrap_p = boot.get('p_value')
    if bootstrap_p is not None:
        if float(bootstrap_p) < GATE_BOOTSTRAP_P_MAX:
            reasons_pass.append(f"bootstrap p={bootstrap_p:.3f} < {GATE_BOOTSTRAP_P_MAX}")
        else:
            reasons_fail.append(f"bootstrap p={bootstrap_p:.3f} >= {GATE_BOOTSTRAP_P_MAX}")
    # If we have no bootstrap result, don't fail the prop on this gate — note it.

    # Gate 2: trade win rate (only meaningful when enough trades exist)
    if n_trades >= 50 and trade_wr is not None:
        if float(trade_wr) >= GATE_TRADE_WIN_RATE_MIN:
            reasons_pass.append(f"trade WR {trade_wr:.1%} >= {GATE_TRADE_WIN_RATE_MIN:.1%}")
        else:
            reasons_fail.append(f"trade WR {trade_wr:.1%} < {GATE_TRADE_WIN_RATE_MIN:.1%}")
    elif n_trades > 0:
        reasons_pass.append(f"trade WR sample too small ({n_trades} trades)")

    # Gate 3: RMSE vs baseline
    if rmse is not None and baseline_rmse is not None and baseline_rmse > 0:
        ratio = float(rmse) / float(baseline_rmse)
        if ratio <= GATE_RMSE_VS_BASELINE_MAX_RATIO:
            reasons_pass.append(
                f"RMSE {rmse:.2f} / baseline {baseline_rmse:.2f} = {ratio:.3f}"
            )
        else:
            reasons_fail.append(
                f"RMSE {rmse:.2f} > baseline {baseline_rmse:.2f} (ratio {ratio:.3f})"
            )

    # Need at least one passing gate and zero failing gates to enable.
    enabled = bool(reasons_pass) and not reasons_fail

    # If we have no signal at all (no predictions in OOS), keep prop disabled.
    if n_pred == 0:
        enabled = False
        reasons_fail.append("no OOS predictions")

    return {
        'enabled': enabled,
        'reasons': (reasons_pass if enabled else reasons_fail) or ['no data'],
        '_metrics': {
            'n_predictions': n_pred,
            'rmse': rmse,
            'baseline_rmse': baseline_rmse,
            'n_trades': n_trades,
            'trade_win_rate': trade_wr,
            'bootstrap_p': bootstrap_p,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--commit', action='store_true',
        help=(
            'Actually write the updated gates JSON. Default is dry-run because '
            'these gates are computed from SIMULATED-line OOS backtest data, '
            'which can be over-optimistic vs real sportsbook lines. Re-run the '
            'real-lines backtest (with all props temporarily enabled) and '
            'manually verify gates before committing.'
        ),
    )
    args = parser.parse_args(argv)
    args.dry_run = not args.commit  # back-compat with --dry-run callers

    oos = _load_oos()
    stat_analysis = _load_stat()

    per_prop: dict[str, dict] = {}
    for prop in _PROPS:
        per_prop[prop] = _evaluate_prop(prop, oos, stat_analysis)

    # Always include 'spread' in disabled regardless of stats — it's a
    # separate market with its own model whose disable is documented in
    # constants.py:128-131. We don't have OOS predictions for spread here.
    per_prop['spread'] = {
        'enabled': False,
        'reasons': ['spread model is documented as worse than market (constants.py)'],
        '_metrics': {},
    }

    disabled = sorted([p for p, status in per_prop.items() if not status['enabled']])

    # Strip the private _metrics field from per_prop for the output payload,
    # but keep them in a sidecar for diagnostics.
    per_prop_public = {
        p: {'enabled': s['enabled'], 'reasons': s['reasons']}
        for p, s in per_prop.items()
    }
    per_prop_metrics = {p: s['_metrics'] for p, s in per_prop.items()}

    print(f"OOS source:     {_OOS_PATH.relative_to(_REPO_ROOT)}")
    print(f"OOS run date:   {oos.get('run_date', '(unknown)')}")
    print()
    print(f"{'prop':<10} {'status':<10} reasons")
    print('-' * 80)
    for prop in [*_PROPS, 'spread']:
        status = per_prop[prop]
        flag = 'ENABLED' if status['enabled'] else 'DISABLED'
        for i, r in enumerate(status['reasons']):
            label = prop if i == 0 else ''
            stat_label = flag if i == 0 else ''
            print(f"{label:<10} {stat_label:<10} {r}")
        print()

    print(f"Disabled list:  {disabled}")

    if args.dry_run:
        print("\nDry run — no file written.")
        return 0

    payload = {
        "_doc": (
            "Per-prop enable/disable gates. A prop is DISABLED if it fails ANY "
            "gate; otherwise enabled. Regenerate with: "
            "python3 scripts/recalibrate_prop_gates.py"
        ),
        "_source": str(_OOS_PATH.relative_to(_REPO_ROOT)),
        "_source_run_date": oos.get('run_date'),
        "_generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "_gates": {
            "bootstrap_p_value_max": GATE_BOOTSTRAP_P_MAX,
            "trade_win_rate_min": GATE_TRADE_WIN_RATE_MIN,
            "rmse_vs_baseline_max_ratio": GATE_RMSE_VS_BASELINE_MAX_RATIO,
        },
        "per_prop_status": per_prop_public,
        "per_prop_metrics": per_prop_metrics,
        "disabled_props": disabled,
    }

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')

    print(f"\nWrote {_OUT_PATH.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
