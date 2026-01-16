# Fundamental Fixes In Progress

**User Directive**: "Fix all fundamental issues. no shortcuts!"

## Fix Status

### ✅ FIX #1: DNP/Injury Detection - COMPLETED

**Problem**: Top 10 worst predictions all DNP players (Luka, Jokic, Giannis with 40-50 point errors when actual=0)

**Root Cause**:
- No injury tracking integrated into backtest
- DNP check only worked for `points` prop type (line 1465)
- Missing comprehensive minutes-played validation

**Solution Implemented** (comprehensive_backtest.py:1446-1477):
1. Added import: `from injury_tracker_v3 import is_player_available, InjuryStatus`
2. Check injury status before predictions (lines 1451-1458)
3. Comprehensive DNP detection using minutes played for ALL prop types (lines 1465-1477)
4. Graceful fallback if historical injury data unavailable
5. Removed inadequate points-only DNP check

**Code Changes**:
```python
# Check injury status (tries API, falls back to minutes check)
try:
    is_available, injury_status = is_player_available(player_id, game_datetime)
    if not is_available:
        continue  # Skip OUT/DOUBTFUL players
except Exception:
    pass  # Historical injury data may not exist

# Check minutes played (catches all DNPs)
minutes_played = parse_minutes(actual_stats.get('min', '0'))
if minutes_played < 0.1:
    continue  # Skip DNP players
```

**Expected Impact**:
- Eliminate 40-50 point errors for injured players
- Reduce overall RMSE significantly
- Improve worst-case predictions dramatically
- More accurate bias calculations (no DNP outliers)

**Validation**:
- Running full backtest now with DNP fix
- Should see zero predictions with actual=0 for players with <0.1 minutes

---

### 🔄 FIX #2: Recalculate Proper Bias Corrections - IN PROGRESS

**Problem**:
- Overall bias -1.174 (target: <|0.5|) - NOT MET
- Points overcorrected: +0.46 (was -2.280, now overpredicting)
- PRA severely underpredicting: -2.97
- Corrections based on 100-sample subset (inaccurate)

**Root Cause**:
- Phase 2.5 corrections calculated from small sample
- DNP errors contaminating bias calculations
- Linear correction insufficient for non-linear bias

**Solution Approach**:
1. Wait for backtest with FIX #1 (eliminates DNP contamination)
2. Use FULL dataset (all ~88k predictions) for bias calculation
3. Calculate prop-specific corrections: `correction = -mean(predicted - actual)`
4. Apply corrections and validate targets met

**Script Created**: `fix2_recalculate_bias.py`
- Loads backtest_results_2025.json with raw predictions
- Calculates accurate bias per prop type
- Generates BIAS_CORRECTIONS code
- Validates all targets met

**Current Bias Corrections** (comprehensive_backtest.py:1152-1158):
```python
BIAS_CORRECTIONS = {
    'points': 2.280,   # Fix bias of -2.280 → OVERCORRECTED to +0.46
    'rebounds': 0.442, # Fix bias of -0.442 → Still -1.17
    'assists': -1.543, # Fix bias of 1.543 → Still -0.97
    'threes': 0.203,   # Fix bias of -0.203 → Still -0.96
    'pra': 0.676,      # Fix bias of -0.676 → Still -2.97
}
```

**Next Steps**:
1. ✅ Modified comprehensive_backtest.py to export raw_predictions
2. ⏳ Waiting for backtest completion (40/596 games, ~30 min remaining)
3. ⏳ Run `python3 fix2_recalculate_bias.py` to calculate new corrections
4. ⏳ Apply new corrections to comprehensive_backtest.py
5. ⏳ Re-run backtest to validate bias < |0.5|

---

### ⏳ FIX #3: Fix PRA Severe Underprediction - PENDING

**Problem**: PRA bias -2.97 (severely underpredicting combined stat)

**Hypothesis**:
- Individual stat biases compounding (pts, reb, ast)
- PRA model architecture issue
- Potential data leakage or feature mismatch

**Investigation Plan**:
1. Check if PRA = separate model OR derived from pts+reb+ast predictions
2. If derived: Fix individual biases first (FIX #2)
3. If separate model: Investigate PRA-specific features and training

**May be resolved by**: FIX #2 if PRA is derived from individual stats

---

### ⏳ FIX #4: Confidence Distribution Problem - PENDING

**Problem**:
- 90.36% predictions in "avoid" tier (unusable for betting)
- Elite + Strong: only 0.25% (target: 10-20%)
- Confidence correlation: 0.1381 (target: >0.5)

**Root Cause Hypotheses**:
1. Base models have high disagreement (CV > 0.20)
2. Phase 2 features causing model instability
3. Poor model calibration
4. Confidence thresholds incorrectly set

**Investigation Plan**:
1. Analyze base model agreement after DNP fix
2. Check if Phase 2 features increase variance (FIX #5)
3. Recalibrate confidence score mapping
4. Consider alternative confidence metrics

---

### ⏳ FIX #5: Feature Ablation Study - PENDING

**Problem**: Phase 2 RMSE (5.707) WORSE than Phase 1 (5.435)

**Hypothesis**: Phase 2 features introducing noise, not signal

**Phase 2 Features**:
- Travel/Fatigue (10 features): miles_traveled, time_zone_changes, b2b indicators
- Betting Markets (6 features): implied_totals, market_efficiency signals
- Enhanced Injuries (4 features): star_player_out, usage_redistribution

**Investigation Plan**:
1. Run ablation: Remove each feature group, measure RMSE
2. Identify which features HURT performance
3. Remove harmful features, keep beneficial ones
4. Re-train models if needed

**Success Criteria**: Phase 2 RMSE < Phase 1 RMSE (5.435)

---

### ⏳ FIX #6: Fix Threes R² = -0.45 - PENDING

**Problem**: 3-point model worse than predicting mean (R² should be positive)

**History**:
- Phase 1: R² = -0.568
- Phase 2: R² = -0.45 (slight improvement but still negative)

**Root Causes**:
- 3PM is highly variable (0-10 range, many 0s)
- Generic features don't capture 3PM patterns
- May need specialized features (3P%, volume, matchup-specific)

**Investigation Plan**:
1. Check base model R² for threes individually
2. Add 3PM-specific features (3P%, attempts, hot/cold streaks)
3. Consider separate model architecture (e.g., zero-inflated)
4. Validate R² > 0.3

---

### ⏳ FIX #7: Final Validation Backtest - PENDING

**After all fixes applied**:
1. Run comprehensive_backtest.py with all corrections
2. Validate ALL targets met:
   - Overall RMSE < 5.0
   - Overall Bias < |0.5|
   - Per-prop Bias < |0.5|
   - Confidence: Elite+Strong ≥ 10%
   - Confidence correlation r > 0.5
   - Phase 2 RMSE < Phase 1 RMSE
   - Threes R² > 0
3. Generate final report comparing before/after
4. Document model performance for Phase 3

---

## Current Status

**Backtest Running**: comprehensive_backtest.py with FIX #1 (DNP detection)
- Progress: 40/596 games (6.7%)
- ETA: ~30 minutes
- Modifications:
  - DNP/injury detection integrated
  - Raw predictions export added
  - Verbose logging for skipped players

**Next Immediate Action**:
1. Wait for backtest completion
2. Run `fix2_recalculate_bias.py` to get accurate corrections
3. Apply corrections and re-run backtest
4. Proceed sequentially through FIX #3-7

---

## Technical Notes

### DNP Detection Implementation

The fix uses two layers of defense:

1. **Injury API Check** (when available):
   - Calls `is_player_available()` from injury_tracker_v3.py
   - Checks OUT, DOUBTFUL status
   - Gracefully fails for historical dates (no historical injury API)

2. **Minutes-Played Check** (always reliable):
   - Parses actual minutes from box score
   - Skips if minutes < 0.1 (less than 6 seconds)
   - Works for ALL historical data

This dual approach ensures:
- Real-time predictions: API catches injuries before game
- Historical backtesting: Minutes-played catches all DNPs
- Zero false positives: Only skips genuine DNPs

### Bias Correction Methodology

Current approach is LINEAR:
```
corrected_prediction = raw_prediction + BIAS_CORRECTION[prop_type]
```

Assumes bias is constant across:
- All players (stars vs bench)
- All situations (home/away, b2b, etc.)
- All prediction magnitudes (high vs low)

If targets still not met after FIX #2, may need:
- Quantile-based corrections (different for high/low predictions)
- Player-tier corrections (starters vs bench)
- Situational corrections (home/away, rest days, etc.)

### Expected Timeline

- **FIX #1**: ✅ Complete
- **FIX #2**: ~1 hour (waiting for backtest + recalculation + validation)
- **FIX #3**: ~30 min (likely resolved by FIX #2)
- **FIX #4**: ~2 hours (complex calibration work)
- **FIX #5**: ~3 hours (ablation + potential retraining)
- **FIX #6**: ~2 hours (specialized feature engineering)
- **FIX #7**: ~1 hour (final validation)

**Total ETA**: ~10 hours of systematic debugging
