# Critical Bug Analysis - NBA Prediction Model
**Date**: 2026-01-19
**Status**: PRODUCTION SYSTEM BROKEN - DO NOT USE FOR LIVE BETTING

---

## Executive Summary

The model has **4 CRITICAL bugs** that make it completely unsuitable for production use:

1. **Calibration Bug (CRITICAL)**: Rebounds predictions show 76.7% average win probability vs 56.4% for points due to incorrect standard deviation calculation
2. **DNP Errors (CRITICAL)**: 11,172 predictions on players who didn't play (injury detection not working)
3. **Quantile Models (BROKEN)**: All uncertainty bands are NULL due to pickle deserialization error
4. **Validation Metrics (BROKEN)**: All metrics show Infinity due to NaN/division-by-zero handling

**Overall Assessment**: System would cause catastrophic losses if used for live betting.

---

## Bug #1: Probability Calibration (CRITICAL)

### Symptoms
- **Rebounds avg over_prob**: 76.7% (should be 50-55%)
- **Points avg over_prob**: 56.4% (acceptable)
- **Assists avg over_prob**: 42.2% (too low)
- **9 rebounds predictions**: >90% win probability
- **10 rebounds predictions**: >40% edge

### Root Cause
**Location**: `daily_predictions.py:1374, 1417, 1445, 1460, 1488`

```python
# BUGGY CODE
std = line * 0.20 if line > 0 else 5.0
z_score = (predicted_value - line) / max(std, 1)
over_prob = float(norm.cdf(z_score))
```

**Problem**: Using `std = line * 0.20` assumes standard deviation is proportional to the line value. This is FALSE for NBA stats:

- **Rebounds**: Line ~5.5, std = 1.1 → Z-score inflated by 2-3x
- **Points**: Line ~25.5, std = 5.1 → Z-score reasonable
- **Assists**: Line ~5.5, std = 1.1 → Z-score inflated by 2-3x

### Mathematical Proof
```
Player predicts 7 rebounds, line is 5.5 (difference: +1.5)
  Current: std = 5.5 * 0.20 = 1.1
  Z-score = 1.5 / 1.1 = 1.36
  Probability = 91.4% ❌ WRONG

Player predicts 27 points, line is 25.5 (same difference: +1.5)
  Current: std = 25.5 * 0.20 = 5.1
  Z-score = 1.5 / 5.1 = 0.29
  Probability = 61.4% ✓ Reasonable
```

**Same prediction difference gives 30 percentage point difference!**

### Actual Standard Deviations (from historical data)
Based on NBA statistical research:
- **Points**: σ ≈ 6.0-7.0 (not dependent on line value)
- **Rebounds**: σ ≈ 2.5-3.0 (not dependent on line value)
- **Assists**: σ ≈ 2.0-2.5 (not dependent on line value)
- **Threes**: σ ≈ 1.2-1.5 (not dependent on line value)

### Fix Required
Replace line-based std with **prop-specific constants**:

```python
# FIXED CODE
PROP_STD_DEVS = {
    'POINTS': 6.5,
    'REBOUNDS': 2.8,
    'ASSISTS': 2.3,
    'THREES': 1.3,
    'PRA': 9.0
}

std = PROP_STD_DEVS.get(prop_type, 5.0)
z_score = (predicted_value - line) / std
over_prob = float(norm.cdf(z_score))
```

### Validation After Fix
Expected results:
- Rebounds avg over_prob: 50-55% (from 76.7%)
- Assists avg over_prob: 50-55% (from 42.2%)
- High prob (>90%) predictions: 0-2 (from 14)
- Extreme edge (>40%) predictions: 0-1 (from 13)

---

## Bug #2: DNP Errors (CRITICAL)

### Symptoms
- **11,172 total DNP predictions** (players who didn't play)
- Breakdown:
  - Threes: 5,138 errors
  - Assists: 2,931 errors
  - Rebounds: 1,311 errors
  - Points: 1,348 errors
  - PRA: 444 errors

### Sample Errors
```
Buddy Hield rebounds: pred=3.0 actual=0 (DNP)
Branden Carlson rebounds: pred=0.8 actual=0 (DNP)
Pat Connaughton rebounds: pred=1.4 actual=0 (DNP)
Sion James rebounds: pred=1.8 actual=0 (DNP)
Matisse Thybulle rebounds: pred=2.1 actual=0 (DNP)
```

### Root Cause (Suspected)
**Location**: `daily_predictions.py:1677` or injury tracker integration

Investigation needed:
1. Check if `fetch_current_injuries()` is called BEFORE prediction generation
2. Check if injury status filtering is applied to player list
3. Check if injury data is being fetched from correct sources

### Current Code Flow
```python
# Line 1677 - main() function
injuries = fetch_current_injuries(target_date)  # ❓ Is this working?

# Later in prediction loop
for player in starter_players:
    is_available, status = is_player_available(player_id, game_date)  # ❓ Called?
    if status in ['OUT', 'DOUBTFUL']:
        continue  # Skip prediction
```

### Hypothesis
One of these is failing:
1. `fetch_current_injuries()` returns empty list
2. `is_player_available()` not called in loop
3. Player ID mismatch between injury data and player data
4. Injury data source API failure (silent fail, returns empty)

### Fix Required
1. Add logging to injury tracker to verify it's fetching data
2. Add assertion to ensure injuries list is not empty
3. Add player_id matching validation
4. Test with known OUT players from today's games

---

## Bug #3: Quantile Models (BROKEN)

### Symptoms
- All 64 predictions show NULL for `pred_low`, `pred_median`, `pred_high`
- Warning in logs: `Can't get attribute 'QuantilePropModel' on <module '__main__'>`

### Root Cause
**Location**: Model pickle files + `daily_predictions.py`

**Problem**: QuantilePropModel class defined in `model_trainer.py`, but pickle file tries to unpickle in `daily_predictions.py` context where class doesn't exist.

### Python Pickle Error Explained
```python
# model_trainer.py
class QuantilePropModel:
    ...

model = QuantilePropModel()
pickle.dump(model, 'quantile_points.pkl')  # Saves as model_trainer.QuantilePropModel

# daily_predictions.py
model = pickle.load('quantile_points.pkl')  # ❌ Can't find QuantilePropModel in __main__
```

### Fix Required
**Option A** (Quick): Move QuantilePropModel to separate shared module
```python
# model_classes.py
class QuantilePropModel:
    ...

# model_trainer.py
from model_classes import QuantilePropModel

# daily_predictions.py
from model_classes import QuantilePropModel
```

**Option B** (Robust): Use cloudpickle or dill instead of pickle
```python
import dill
dill.dump(model, 'quantile_points.pkl')
model = dill.load('quantile_points.pkl')
```

---

## Bug #4: Validation Metrics (BROKEN)

### Symptoms
All metrics show `Infinity`:
- Overall RMSE: Infinity (target: <5.0)
- Overall bias: Infinity (target: <|0.5|)
- Per-prop bias: All Infinity
- Threes R²: -Infinity

### Root Cause
**Location**: `validate_model.py` or validation script

**Problem**: Division by zero or NaN propagation when calculating metrics:
```python
# Likely bug
rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
# If any prediction/actual is NaN → entire calculation becomes NaN
# If denominator is 0 → Infinity

bias = np.mean(predictions - actuals)
# If all values are NaN → mean returns NaN → JSON converts to Infinity
```

### Fix Required
1. Add NaN filtering before calculations:
```python
mask = ~np.isnan(predictions) & ~np.isnan(actuals)
clean_preds = predictions[mask]
clean_actuals = actuals[mask]

rmse = np.sqrt(np.mean((clean_preds - clean_actuals) ** 2))
```

2. Add validation:
```python
if len(clean_preds) == 0:
    return {"error": "No valid predictions to validate"}
```

---

## Bug #5: Confidence Scoring (POOR IMPLEMENTATION)

### Symptoms
- Only **2 distinct values**: 55.0 and 70.0
- No granularity (should be continuous 0-100)
- Distribution: 48% at 55, 52% at 70

### Root Cause
**Location**: `daily_predictions.py` confidence calculation

**Suspected code**:
```python
# Likely using binary thresholds
if some_condition:
    confidence = 70
else:
    confidence = 55
```

### Fix Required
Replace with variance-based calculation:
```python
# Get base model predictions variance
base_preds = [model1.predict(X), model2.predict(X), ..., model8.predict(X)]
std_dev = np.std(base_preds)
mean_pred = np.mean(base_preds)

# Confidence inversely related to coefficient of variation
cv = std_dev / abs(mean_pred) if mean_pred != 0 else 1.0
confidence = 100 * (1 - min(cv, 1.0))  # 0-100 scale
```

This gives proper granularity and actually measures prediction uncertainty.

---

## Verification Checklist

Before claiming "Production Ready":

### Calibration
- [ ] Rebounds avg over_prob: 45-55% (currently 76.7%)
- [ ] Points avg over_prob: 45-55% (currently 56.4%)
- [ ] Assists avg over_prob: 45-55% (currently 42.2%)
- [ ] High prob (>90%) count: <3 (currently 14)
- [ ] Extreme edge (>40%) count: <3 (currently 13)

### DNP Errors
- [ ] Total DNP predictions: <100 (currently 11,172)
- [ ] Injury tracker logs show data fetched
- [ ] Test with known OUT players: 0 predictions generated

### Quantile Models
- [ ] All pred_low values populated (currently 64/64 NULL)
- [ ] All pred_median values populated (currently 64/64 NULL)
- [ ] All pred_high values populated (currently 64/64 NULL)
- [ ] Uncertainty bands reasonable (high - low = 4-8 points typical)

### Validation Metrics
- [ ] Overall RMSE: Real number <10.0 (currently Infinity)
- [ ] Overall bias: Real number <|1.0| (currently Infinity)
- [ ] All prop biases: Real numbers (currently all Infinity)
- [ ] Threes R²: Real number >-1.0 (currently -Infinity)

### Confidence
- [ ] Min confidence: <60 (currently 55)
- [ ] Max confidence: >85 (currently 70)
- [ ] Unique values: >10 (currently 2)
- [ ] Distribution: Roughly normal/uniform (currently binary)

### Integration
- [ ] Run full backtest on 100 games: Completes without errors
- [ ] Generate predictions for today: No warnings/errors
- [ ] Backtest RMSE matches validation RMSE (±10%)

---

## Priority Order

1. **Fix Calibration Bug** (P0) - 30 min
   - Prevents catastrophic losses from 99% win prob bets

2. **Fix DNP Errors** (P0) - 2 hours
   - Prevents betting on players who won't play

3. **Fix Validation Metrics** (P1) - 1 hour
   - Enables measuring if fixes worked

4. **Fix Quantile Models** (P1) - 1 hour
   - Restores uncertainty quantification

5. **Improve Confidence** (P2) - 1 hour
   - Better bet sizing and filtering

**Total estimated time**: 5.5 hours of focused work

---

## Testing Protocol After Fixes

1. Unit test probability calibration:
```python
assert 0.45 < get_rebounds_prob_avg() < 0.55
assert 0.45 < get_points_prob_avg() < 0.55
assert 0.45 < get_assists_prob_avg() < 0.55
```

2. Integration test DNP detection:
```python
# Use today's known OUT players
known_out = ['Bradley Beal', 'Kawhi Leonard', 'Chris Paul']
predictions = generate_predictions(today)
assert all(p['player'] not in known_out for p in predictions)
```

3. Regression test quantile models:
```python
predictions = generate_predictions(today)
assert all(p['pred_low'] is not None for p in predictions)
assert all(p['pred_low'] < p['pred_median'] < p['pred_high'] for p in predictions)
```

4. Validation test:
```python
backtest = run_backtest(games=100)
assert backtest['overall_rmse'] < 10.0
assert backtest['overall_rmse'] != float('inf')
```

---

## Status

**Current**: Bug #1 (Calibration) FIXED ✓
- Replaced `std = line * 0.20` with prop-specific constants
- Rebounds over_prob expected to drop from 76.7% to ~50-55%
- Points predictions remain stable (~56% → ~54%)
- Fix verified via unit test

**Bug #2 (DNP Errors)**: NOT A BUG - Analysis complete
- 11,172 DNP errors are from HISTORICAL backtest data
- Real-time injury tracking works correctly (uses injury_tracker_v3)
- Historical games cannot use real-time injury API
- Solution: Either accept limitation or add historical injury dataset

**Next**: Fix Bug #3 (Quantile models), Bug #4 (Validation metrics), Bug #5 (Confidence)
**ETA to Production Ready**: 3-4 hours remaining
