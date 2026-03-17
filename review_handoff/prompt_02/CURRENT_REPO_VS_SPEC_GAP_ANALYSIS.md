# Current Repo vs Evaluation Spec — Gap Analysis

**Repo HEAD:** `2b81b31b17b12926fcec5180b6af836f61a9f2ca`

---

## Gate 1: No Simulated Lines

| Aspect | Status | Evidence |
|--------|--------|----------|
| Profitability backtest | **FAIL** | `nba_models/backtesting/profitability_backtest.py:L134-164` — `simulate_prop_line()` generates lines from features |
| Real-lines backtest | **PASS** | `nba_models/backtesting/real_lines_backtest.py:L249-280` — `load_historical_lines()` reads from `data/historical_lines/` |
| Comprehensive backtest | **FAIL** | No historical line lookup; uses feature-derived lines |
| Daily predictions (live) | **PASS** | `nba_models/inference/daily_predictions.py` fetches real lines from BallDontLie API |

**Required changes:** Profitability backtest and comprehensive backtest must use `load_historical_lines()` or be relabeled RESEARCH-ONLY. A `real_lines_backtest.py` exists but is not the default evaluation path.

---

## Gate 2: Decision-Time Line and Odds Present

| Aspect | Status | Evidence |
|--------|--------|----------|
| Historical lines data | **PARTIAL** | `data/historical_lines/*.json` has `snapshot_timestamp` per game and per-prop `bookmaker` + `over_odds`/`under_odds`. 166 files covering 2024-10-22 to 2025-04-xx. |
| Profitability backtest output | **FAIL** | Uses simulated lines; no `snapshot_timestamp` or `book` in output |
| Real-lines backtest output | **PARTIAL** | Uses real lines but does not store `snapshot_timestamp` or `book` in per-bet output |
| Daily predictions CSV | **FAIL** | `daily_predictions.py:L3318-3376` — CSV has `line_source` and `line_vendor` but no `snapshot_timestamp` |

**Required changes:** Per-bet records must include `snapshot_timestamp` and `book` fields.

---

## Gate 3: Closing Line and Odds for CLV

| Aspect | Status | Evidence |
|--------|--------|----------|
| Historical lines data | **BLOCKER — NO** | `data/historical_lines/2024-12-25.json` contains ONE snapshot per game (`snapshot_timestamp: "2024-12-25T16:00:00Z"`). No `closing_line`, `closing_odds`, or second snapshot exists. |
| Any closing line capture code | **NO** | Searched: `grep -r "closing_line\|closing_odds" nba_models/ nba_betting/ edge_calculator/` — no storage/capture logic found. `nba_betting/odds/betting_market_features.py` has CLV feature computation but does not persist closing data. |
| CLV computation | **NO** | No function computes CLV from stored closing lines. |

**BLOCKER:** Closing lines are not captured or stored anywhere. This blocks ALL production-like evaluation.

---

## Gate 4: No Test-Period Data in Training

| Aspect | Status | Evidence |
|--------|--------|----------|
| Profitability backtest | **FAIL** | `profitability_backtest.py:L11` — docstring admits in-sample model. `L61-63`: `TEST_SEASON = "2023-24"`, `CONTEXT_SEASONS = ["2022-23", "2023-24"]`. |
| Real-lines backtest | **FAIL** | `real_lines_backtest.py:L818` — caveat note mentions "Historical lines from pre-game snapshots" but uses same models trained on overlapping data. No walk-forward retraining. |
| Weekly retrain workflow | **FAIL** | `.github/workflows/weekly-retrain.yml` trains on all available data without temporal holdout. |
| Model artifacts | **FAIL** | `models/*.pkl` do not store `train_window_end`. Cannot verify temporal separation. |

**Required changes:** Implement walk-forward retraining per DATA_SPLIT_POLICY.md. Add `train_window_start`/`train_window_end` to all artifacts.

---

## Gate 5: Real Odds or Research-Only Label

| Aspect | Status | Evidence |
|--------|--------|----------|
| Profitability backtest | **FAIL** | `profitability_backtest.py:L60`: `STANDARD_ODDS = -110` hardcoded for all bets. Not labeled research-only in outputs. |
| Real-lines backtest | **PASS** | `real_lines_backtest.py:L249-280` loads real per-prop odds from historical data |
| Daily predictions | **PASS** | Fetches real odds from BallDontLie/Odds API |

**Required changes:** Profitability backtest outputs must carry `realism_level: "RESEARCH-ONLY"` watermark.

---

## Gate 6: Settlement Supports VOID for DNP

| Aspect | Status | Evidence |
|--------|--------|----------|
| `settle_trades.py` | **FAIL** | `nba_betting/settle_trades.py:L15-91` — no minutes check, no void concept |
| `settlement_service.py` | **FAIL** | `continuous_learning/settlement_service.py` — no DNP check |
| `paper_trading.py` | **FAIL** | `nba_betting/paper_trading.py:L261` — `settle_trades()` has no void logic |
| Profitability backtest | **PARTIAL** | `profitability_backtest.py:L278` skips `actual_min < 15` (filters out DNP and low-minute) but does not label them VOID — just skips silently |

**Required changes:** Settlement must check for 0 minutes → VOID.

---

## Gate 7: Bet Record Schema Completeness

| Aspect | Status | Evidence |
|--------|--------|----------|
| Prediction CSV output | **FAIL** | `daily_predictions.py:L3318-3376` outputs 37 columns. Missing from spec: `event_id`, `snapshot_timestamp`, `book` (as bet-level field), `closing_line`, `closing_odds`, `market_implied_probability`, `vig_adjusted_probability`, `raw_edge`, `vig_adjusted_edge`, `accepted`, `result`, `CLV`, `PnL`, `stake`, `artifact_version`, `git_sha`, `realism_level` |
| Backtest output | **FAIL** | `profitability_backtest.py:L406-424` produces per-trade dicts with ~13 fields. Missing most spec fields. |
| Real-lines backtest output | **FAIL** | Similar gaps to profitability backtest |

**Required changes:** Implement BET_RECORD_SCHEMA.md as canonical output format.

---

## Gate 8: Artifact Metadata Completeness

| Aspect | Status | Evidence |
|--------|--------|----------|
| Model save format | **FAIL** | `train_complete_balldontlie.py:L5763-5781` saves `saved_at` but NOT `git_sha`, `train_window_start`, `train_window_end`, `training_samples`, `feature_schema_version`, `artifact_version` |
| Model registry | **PARTIAL** | `continuous_learning/model_registry.py` has `ModelVersion` class with some metadata, but not fully conformant |

**Required changes:** Add all MODEL_ARTIFACT_SCHEMA.md fields to training output.

---

## Additional Spec Requirements

### Walk-Forward Retraining
**Status:** NO
**Evidence:** No walk-forward retraining infrastructure exists. `profitability_backtest.py` uses a single pre-trained model. `real_lines_backtest.py` also uses a single pre-trained model.

### CLV as First-Class Metric
**Status:** NO
**Evidence:** No CLV computation exists in any backtest output. `nba_betting/odds/betting_market_features.py` has CLV feature logic but it's for model features, not for evaluating model quality.

### Bias Correction Temporal Safety
**Status:** FAIL
**Evidence:**
- `nba_betting/constants.py:L50-56`: `PROP_BIAS_CORRECTION` derived from "67K-prediction backtest" — temporal provenance unclear
- `comprehensive_backtest.py:L1296-1302`: `BIAS_CORRECTIONS` appear to be fitted on test data

### Evaluation Mode Labels
**Status:** NO
**Evidence:** No output from any backtest or prediction pipeline includes a `realism_level` field.

---

## Summary of Blockers

| ID | Blocker | Severity |
|----|---------|----------|
| B1 | No closing line data anywhere in repo | CRITICAL — blocks all CLV and production-like evaluation |
| B2 | No walk-forward retraining infrastructure | CRITICAL — blocks temporal separation guarantee |
| B3 | No bet record conforming to canonical schema | HIGH — blocks standardized evaluation |
| B4 | No artifact metadata conforming to schema | HIGH — blocks provenance tracking |
| B5 | Settlement has no VOID for DNP | MEDIUM — distorts backtest results |
| B6 | No realism_level field in any output | MEDIUM — allows misleading claims |
