# NBA-BETS Model Improvements v3

**Date:** 2026-02-25  
**Status:** All 6 improvements implemented and tested

---

## Summary

Six targeted improvements were made to address the root causes of poor model performance identified in the diagnostic analysis: RMSE=14.20 for spreads, win_prob=1.0 from the over/under classifier, and zero edge in threes predictions.

---

## Improvement 1: Over/Under Classifier Calibration (Fixed)

**Problem:** The `over_under_classifier` stored in `player_*_ensemble.pkl` returned extreme probabilities (0.0 or 1.0) because it was never calibrated after training. This caused Kelly sizing to bet the maximum allowed on every single prop.

**Fix:** Added Platt-style temperature scaling (temperature=2.0) in three places:

| File | Change |
|------|--------|
| `nba_models/models/model_classes.py` | Added `calibrate_probability()` function + applied it to `over_probability` in `PropEnsembleModel.predict()` |
| `nba_models/backtesting/comprehensive_backtest.py` | Added `calibrate_probability()` as a module-level utility function |
| `nba_models/training/train_complete_balldontlie.py` | Applied inline temperature scaling in `PropEnsembleModel.predict()` |

**Effect:**
- `raw_prob=0.0` → `calibrated=0.09` (previously caused $0 Kelly, now sensible)
- `raw_prob=1.0` → `calibrated=0.91` (previously caused max Kelly, now capped)
- `raw_prob=0.5` → `calibrated=0.50` (unchanged)

---

## Improvement 2: Smart Bet Selection Filter

**Problem:** The model bet on every prediction regardless of edge quality. Threes were included despite having zero model edge (R²=0.31, RMSE identical to naive season-average baseline).

**New File:** `nba_betting/bet_filter.py`

**Key thresholds:**

| Prop Type | Min Edge | Notes |
|-----------|----------|-------|
| points    | 2.0      |       |
| rebounds  | 1.0      |       |
| assists   | 0.8      |       |
| **threes**| **999**  | **Effectively disabled** |
| pra       | 3.0      |       |
| spread    | 2.5      |       |
| moneyline | 0.05     | 5% probability edge |

Additional gates:
- Minimum 10 games played (sample size)
- Minimum calibrated confidence of 0.58

**Functions:**
- `should_bet(prop_type, predicted_value, line_value, confidence, games_played)` → `(bool, reason, edge)`
- `calculate_bet_size(edge, confidence, bankroll, ...)` → dollar amount via quarter-Kelly
- `get_edge_tier(edge, prop_type)` → `'elite' | 'strong' | 'moderate' | 'weak' | 'no_bet'`

---

## Improvement 3: Opponent-Adjusted Features

**Problem:** Player prop models used only player-level features; no opponent defensive strength was encoded beyond league-average defaults.

**Modified File:** `nba_models/backtesting/comprehensive_backtest.py` — `get_player_features_before_date()` method

**New features added** (computed point-in-time, using only past games):

| Feature | Description |
|---------|-------------|
| `opp_pts_allowed_avg` | Avg points per player allowed by opponent in recent games |
| `opp_reb_allowed_avg` | Avg rebounds per player allowed by opponent |
| `opp_ast_allowed_avg` | Avg assists per player allowed by opponent |
| `opp_pts_factor` | Ratio vs league average (>1.0 = soft defense) |
| `opp_reb_factor` | Ratio vs league average |
| `opp_ast_factor` | Ratio vs league average |

**Note:** Currently-trained models ignore unknown feature columns (gracefully via `X[feature_names]` with fillna). The next retrain will incorporate these features into model training.

---

## Improvement 4: Spread Model Regularization

**Problem:** Spread model RMSE = 14.20, worse than Vegas (~12-13). Root cause: XGBoost and LightGBM were overfitting with deep trees and low regularization.

**Modified File:** `nba_models/training/train_complete_balldontlie.py` — spread model instantiation section

**XGBoost changes:**

| Parameter | Before | After |
|-----------|--------|-------|
| `n_estimators` | 200 | 500 |
| `max_depth` | 6 | 4 |
| `learning_rate` | 0.1 | 0.03 |
| `min_child_weight` | 3 | 10 |
| `subsample` | 0.8 | 0.7 |
| `colsample_bytree` | 0.8 | 0.7 |
| `reg_alpha` | 0.1 | 1.0 |
| `reg_lambda` | 1.0 | 5.0 |

**LightGBM changes:**

| Parameter | Before | After |
|-----------|--------|-------|
| `n_estimators` | 200 | 500 |
| `max_depth` | 8 | 4 |
| `learning_rate` | 0.1 | 0.03 |
| `num_leaves` | 31 | 15 |
| `min_child_samples` | 20 | 30 |
| `subsample` | 0.8 | 0.7 |
| `colsample_bytree` | 0.8 | 0.7 |
| `reg_alpha` | 0.1 | 1.0 |
| `reg_lambda` | 0.1 | 5.0 |

**Effect:** Next retrain should bring RMSE closer to Vegas baseline (~12-13) by reducing overfit variance.

---

## Improvement 5: Unified Prediction Pipeline

**New File:** `nba_betting/prediction_pipeline.py`

A single function `evaluate_bet()` that orchestrates the full pipeline:

```
model prediction
    ↓
calibrate_probability() — temperature scaling on raw classifier output
    ↓
DISABLED_PROPS gate — threes and any other zero-edge props
    ↓
sample size gate — MIN_GAMES = 10
    ↓
minimum edge gate — prop-specific thresholds
    ↓
confidence gate — MIN_CONFIDENCE = 0.58
    ↓
Kelly sizing — quarter-Kelly, hard cap at 3% bankroll
    ↓
Result dict with: should_bet, direction, edge, confidence, bet_size, tier, reason
```

**Usage:**
```python
from nba_betting.prediction_pipeline import evaluate_bet

result = evaluate_bet(
    prop_type='points',
    predicted=28.5,
    line=26.0,
    raw_confidence=0.72,
    games_played=35,
    bankroll=1000.0,
)
# result = {'should_bet': True, 'tier': 'moderate', 'bet_size': 30.0, ...}
```

---

## Improvement 6: Main Prediction Entry Point Integration

**Modified File:** `nba_models/inference/daily_predictions.py`

**Changes:**
1. Added graceful import of `bet_filter` and `prediction_pipeline` with `HAS_BET_FILTER` flag
2. Applied `evaluate_bet()` after all predictions are generated in `predict_player_prop()`
3. Filter overrides `bet_recommendation='PASS'` and `suggested_bet_size=0` when the filter rejects
4. Added `bet_filter`, `bet_filter_passed`, `bet_filter_tier`, `bet_filter_reason` keys to the returned prediction dict

**Backward compatibility:** If `nba_betting` package is unavailable, falls back to `HAS_BET_FILTER=False` with no-op stubs. Existing behavior is preserved.

---

## Package Updates

**Modified File:** `nba_betting/__init__.py`

Added exports:
```python
from .bet_filter import should_bet, calculate_bet_size, get_edge_tier
from .prediction_pipeline import calibrate_probability, evaluate_bet, evaluate_bets_batch
```

---

## Files Changed

| File | Type | Description |
|------|------|-------------|
| `nba_models/models/model_classes.py` | Modified | Added `calibrate_probability()`, applied to over_under_classifier output |
| `nba_models/backtesting/comprehensive_backtest.py` | Modified | Added `calibrate_probability()`, added opponent-adjusted features to `get_player_features_before_date()` |
| `nba_models/training/train_complete_balldontlie.py` | Modified | Fixed over_under calibration, increased spread model regularization |
| `nba_models/inference/daily_predictions.py` | Modified | Integrated bet filter and prediction pipeline |
| `nba_betting/bet_filter.py` | **New** | Smart bet selection filter |
| `nba_betting/prediction_pipeline.py` | **New** | Unified calibrate→filter→size pipeline |
| `nba_betting/__init__.py` | Modified | Exported new modules |

---

## Expected Impact

| Issue | Before | After (next retrain/run) |
|-------|--------|--------------------------|
| Over/under classifier | win_prob=1.0 constantly | Calibrated to 0.05–0.95 range |
| Bet selection | Bets on everything | Only high-edge, high-confidence bets |
| Threes betting | Active (no edge) | Disabled |
| Spread RMSE | 14.20 | Target ≤ 13.0 after retrain |
| Opponent defense | Not modeled | 6 new features ready for next retrain |
| Kelly sizing | Max bets on every game | Proportional quarter-Kelly sizing |
