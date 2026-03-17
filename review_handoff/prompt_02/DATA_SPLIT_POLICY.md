# Data Split Policy — NBA-BETS

**Version:** 1.0.0

---

## D1. Definitions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_window_days` | 365 | Days of training data per window |
| `val_window_days` | 30 | Validation window for hyperparameter selection |
| `test_window_days` | 30 | Test window for evaluation (NEVER seen during training) |
| `retrain_cadence_days` | 14 | How often to retrain and advance the window |
| `mode` | `rolling` | `rolling` (fixed window size) or `expanding` (growing train set) |

### Season Boundary Handling

- NBA regular season: approx Oct 22 – Apr 13.
- Playoffs: Apr 14 – Jun 20.
- Off-season: Jun 21 – Oct 21.
- **Rule:** Off-season gap is skipped. When advancing windows across the off-season, the next window starts at the following season opener.
- **Rule:** Playoff data MAY be included in training but MUST be flagged `is_playoff = true` in features.

### New Players / Trades / Role Changes

- **New player:** If `games_played < 10`, the prediction pipeline rejects the bet via Gate 2 (`MIN_GAMES = 10`, see `nba_betting/prediction_pipeline.py:L64`). This is correct behavior.
- **Traded player:** Features must reflect only the player's stats with their CURRENT team post-trade. Historical stats from previous team are included but should carry a `team_changed` flag.
- **Role change:** No special handling required beyond the rolling window naturally picking up new usage patterns.

### Lookahead Leakage Prevention

**ABSOLUTE RULES:**
1. For any test sample with `game_date = D`, ALL training data must have `game_date < D`.
2. Model weights must be fitted ONLY on data with `game_date < val_start_date`.
3. Validation data must have `game_date < test_start_date`.
4. Feature computation must use only data available BEFORE the game (`*_before_date()` functions).
5. Bias corrections, calibration parameters, and decompression constants must be fitted ONLY on train+val data, NEVER on test data.

**Current violations:**
- `profitability_backtest.py:L11`: "model weights were fit on data that includes 2023-24" (the test period).
- `nba_betting/constants.py:L50-56`: `PROP_BIAS_CORRECTION` derived from "67K-prediction backtest" which includes training data.
- `comprehensive_backtest.py:L1296-1302`: `BIAS_CORRECTIONS` fitted on the same data they are applied to.

---

## D2. Pseudocode

```python
from datetime import date, timedelta
from typing import List, Tuple

def generate_windows(
    start_date: date,
    end_date: date,
    train_window_days: int = 365,
    val_window_days: int = 30,
    test_window_days: int = 30,
    retrain_cadence_days: int = 14,
    mode: str = "rolling",  # "rolling" or "expanding"
) -> List[Tuple[date, date, date, date, date, date]]:
    """
    Generate (train_start, train_end, val_start, val_end, test_start, test_end) tuples.

    In rolling mode, train_start advances with each window.
    In expanding mode, train_start is always start_date.

    Returns list of 6-tuples.
    """
    windows = []
    cursor = start_date + timedelta(days=train_window_days + val_window_days)

    while cursor + timedelta(days=test_window_days) <= end_date:
        test_start = cursor
        test_end = cursor + timedelta(days=test_window_days)
        val_start = test_start - timedelta(days=val_window_days)
        val_end = test_start

        if mode == "rolling":
            train_start = val_start - timedelta(days=train_window_days)
        else:  # expanding
            train_start = start_date
        train_end = val_start

        # Skip off-season gaps (Jun 21 - Oct 21)
        # If test window falls entirely in off-season, skip to next season
        if _is_offseason(test_start) and _is_offseason(test_end):
            cursor += timedelta(days=retrain_cadence_days)
            continue

        windows.append((train_start, train_end, val_start, val_end, test_start, test_end))
        cursor += timedelta(days=retrain_cadence_days)

    return windows


def _is_offseason(d: date) -> bool:
    """NBA off-season: roughly Jun 21 - Oct 21."""
    month_day = (d.month, d.day)
    return (6, 21) <= month_day <= (10, 21)


def run_walk_forward_backtest(
    windows: List[Tuple],
    model_factory,       # callable that trains a model given data
    feature_generator,   # callable that generates features for a game
    line_source: str,    # path to historical_lines dir
    settlement_source,   # callable that fetches actual stats
    output_dir: str,
) -> dict:
    """
    Execute walk-forward backtest across all windows.

    For each window:
    1. Train model on [train_start, train_end)
    2. Validate on [val_start, val_end) — used for hyperparameter selection only
    3. Evaluate on [test_start, test_end) — record per-bet results
    4. Save bet log and summary per window

    Returns aggregated results dict.
    """
    all_bets = []
    for train_start, train_end, val_start, val_end, test_start, test_end in windows:
        # 1. Train
        model = model_factory(train_start, train_end)

        # 2. Validate (optional — for tuning only)
        val_metrics = evaluate_window(model, val_start, val_end, line_source, settlement_source)

        # 3. Test
        test_bets = evaluate_window(model, test_start, test_end, line_source, settlement_source)

        # 4. Record
        for bet in test_bets:
            bet['train_window'] = f"{train_start}/{train_end}"
            bet['test_window'] = f"{test_start}/{test_end}"
            all_bets.append(bet)

    return {
        'total_bets': len(all_bets),
        'bets': all_bets,
        'windows': len(windows),
    }
```

---

## D3. Example CLI Commands (Future-Facing)

These commands describe intended behavior. Scripts DO NOT exist yet.

```bash
# Generate walk-forward windows for 2024-25 season
python -m nba_models.evaluation.split_policy \
    --start-date 2023-10-22 \
    --end-date 2025-04-13 \
    --train-days 365 \
    --val-days 30 \
    --test-days 30 \
    --retrain-cadence 14 \
    --mode rolling \
    --output review_handoff/prompt_02/reports/windows.json

# Run walk-forward backtest
python -m nba_models.evaluation.walk_forward \
    --windows review_handoff/prompt_02/reports/windows.json \
    --lines-dir data/historical_lines/ \
    --model-dir models/ \
    --output-dir review_handoff/prompt_02/reports/
```

Expected outputs:
- `review_handoff/prompt_02/reports/per_bet_log.jsonl`
- `review_handoff/prompt_02/reports/summary.json`
- `review_handoff/prompt_02/reports/clv_by_month.csv`
