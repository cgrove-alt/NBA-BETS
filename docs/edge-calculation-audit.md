# Edge Calculation Audit — Phase 2, Step 1

**Date:** 2026-02-23
**Status:** Fixed and tested

## The Bug

**Location:** `nba_models/inference/daily_predictions.py` line 1136 (the production prediction path)

```python
# BUGGY (before):
spread_edge_points = predicted_spread - market_spread

# FIXED (after):
home_cover_threshold = -market_spread
spread_edge_points = predicted_spread - home_cover_threshold  # = predicted_spread + market_spread
```

### Convention

| Variable | Meaning | Example |
|---|---|---|
| `predicted_spread` | Home margin. +10 = home wins by 10. | +4.6 |
| `market_spread` | Home spread. -12 = home favored by 12. | -12.0 |
| `home_cover_threshold` | Points home needs to win by to cover. `-market_spread`. | 12.0 |

### Why It Was Wrong

When home is a -12 favorite and the model predicts home wins by 4.6:

| Formula | Calculation | Result | Side | Correct? |
|---|---|---|---|---|
| **Buggy** | `4.6 - (-12) = 16.6` | +16.6 | HOME covers | NO |
| **Fixed** | `4.6 + (-12) = -7.4` | -7.4 | AWAY covers | YES |

The model says home wins by only 4.6, but the market requires them to win by 12+. Home clearly doesn't cover. The old formula subtracted a negative market spread, producing a double-positive that always favored home.

### Secondary Bug: `abs()` on Line 1137

The old code applied `abs()` to `spread_edge_points` before passing to `get_spread_cover_probability()`, then tried to flip the probability in the `else` branch with `cover_prob = 1 - cover_prob`. This was both confusing and incorrect — it computed the probability of home covering, then subtracted from 1, instead of directly computing the probability of the recommended side covering.

**Fix:** Each branch now passes the absolute edge to `get_spread_cover_probability()` directly.

### The Fix Already Existed

`app.py:58-116` (`determine_spread_bet_side`) had the correct formula since the college basketball model audit. But `daily_predictions.py` — the actual production prediction path — was never updated.

## Correct Formula (from app.py)

```python
home_cover_threshold = -market_spread    # Points home needs to win by
edge = predicted_spread - home_cover_threshold  # = predicted_spread + market_spread

if edge > 0:
    # Home covers
    bet_side = home, cover_prob = norm.cdf(edge / 13.0)
else:
    # Away covers
    bet_side = away, cover_prob = norm.cdf(abs(edge) / 13.0)

edge_pct = (cover_prob - 0.524) * 100  # vs -110 break-even
```

## Edge Calculator Module Fix

`edge_calculator/edge_calculator.py` `calculate_edge_from_prediction()` used a hardcoded linear approximation (`diff * 0.04`) to convert prediction differences to probabilities. This was replaced with `norm.cdf(diff / std_dev)` using prop-specific standard deviations from `PROP_STD_DEVS`, matching the production path in `daily_predictions.py`.

## Test Coverage

**File:** `tests/test_edge_calculations.py` — 44 tests

| Test Class | Count | Coverage |
|---|---|---|
| `TestSpreadEdgeFormula` | 14 | All 7 plan scenarios + direction, symmetry, blowout, vig hurdle |
| `TestCrossCodepathConsistency` | 10 | Verifies daily_predictions.py matches app.py for 10 scenarios |
| `TestPropEdge` | 7 | Over/under, all prop types, std dev effects |
| `TestEdgeCalculatorModule` | 4 | norm.cdf usage, prop_type routing, backward compat |
| `TestMoneylineEdge` | 4 | Favorite/underdog/no edge/negative edge |

### Test Scenarios (Spread)

| # | Case | predicted | market | Side | Edge |
|---|------|-----------|--------|------|------|
| 1 | Home fav covers | +15 | -12 | home | 3.0 |
| 2 | Home fav doesn't cover | +4.6 | -12 | away | 7.4 |
| 3 | Home dog covers | -2.5 | +5.5 | home | 3.0 |
| 4 | Home dog doesn't cover | -8 | +5.5 | away | 2.5 |
| 5 | Pick'em | +3 | 0 | home | 3.0 |
| 6 | Zero edge | +7 | -7 | away | 0.0 |
| 7 | CLAUDE.md case | -2.5 | +5.5 | home | 3.0 |

## Impact Assessment

The buggy formula systematically over-recommended home-side bets and under-recommended away-side bets whenever the home team was favored. For games where the model predicted a lower margin than the market line (the most common scenario), the bug would recommend HOME when AWAY was the correct play.

This means past "negative edge" recommendations on the away side were likely actually positive-edge bets that were missed. A backtest comparing old vs new formula on historical data is needed to quantify the exact number of missed opportunities.

## Files Changed

| File | Change |
|---|---|
| `nba_models/inference/daily_predictions.py` | Fixed spread edge formula (lines 1131-1157) |
| `edge_calculator/edge_calculator.py` | Replaced linear 4% hack with norm.cdf + PROP_STD_DEVS |
| `tests/test_edge_calculations.py` | New: 44 regression tests |
| `docs/edge-calculation-audit.md` | This document |
