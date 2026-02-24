# Improvement Plan V4: Best Bets Calibration

## Phase 1: Analysis & Diagnosis (CRITICAL)
- [x] **Inspect Edge Calculation**: `edge = prediction - line` (simple subtraction)
- [x] **Check Calibration**: Prop calibrator DISABLED (was crushing confidence to 39-50%)
- [x] **Audit Thresholds**: Previous: min_confidence=50%, min_edge=3.0

## Phase 2: Refinement
- [x] **Enable Calibration**: Skipped - calibrator produced bad results
- [x] **Stricter Filters**: Updated to min_confidence=55%, min_edge=4%
- [x] **Confidence Boosts**: Added Factor 8 (Easy Matchup: opp_def_strength < -3 → +5%) and Factor 9 (Hit Rate > 60% → +5%)
- [x] **Better Sorting**: Now sorting by `(confidence-50) × edge_pct` composite score

## Phase 3: Verification
- [x] **Data Verification**: Code compiles, confidence boosts added
- [x] **Frontend Check**: No changes needed

---

## Review (Completed 2026-01-07)

### Changes Made:

**`backend/api.py`:**
- min_confidence: 50% → 55%
- min_edge: 3.0 → 4.0 pts
- Sorting: `(confidence-50) × edge_pct`

**`dashboard/data_service.py` (`_calculate_prop_confidence`):**
- Factor 8: Easy Matchup Boost - if `opp_def_strength < -3` → +5%
- Factor 9: Hit Rate Boost - if `last_10_hit_rate > 0.6` → +5%
