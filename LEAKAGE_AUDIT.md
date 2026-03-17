# Leakage Audit — NBA-BETS

## CRITICAL Issues

### L-01: In-Sample Model Backtest (CRITICAL)

**Severity:** CRITICAL
**Files:** `nba_models/backtesting/profitability_backtest.py` (lines 8-12)
**Description:** The profitability backtest uses models trained on data that INCLUDES the 2023-24 test season. The docstring explicitly states: "The trained model weights were fit on data that includes 2023-24 — so this is an 'in-sample features, in-sample model' backtest."
**Why this is leakage:** The model has already seen the outcomes it is being tested on during training. Walk-forward feature computation (point-in-time) does NOT fix this — the model weights themselves encode knowledge of the test period outcomes.
**Affected outputs:** ALL backtest ROI, win rate, Sharpe, and profitability claims from profitability_backtest.py.
**Likely effect:** Reported ROI could be inflated by 5-20% or more. Any positive ROI claim from this backtest is unverifiable.
**Permanent fix:** Implement proper temporal holdout: train on seasons <= 2022-23, test on 2023-24. OR implement rolling walk-forward where models are retrained periodically using only past data.
**Required tests:** Test that no training sample has a date >= earliest test sample date.

### L-02: Simulated Prop Lines in Backtest (CRITICAL)

**Severity:** CRITICAL
**Files:** `nba_models/backtesting/profitability_backtest.py`, function `simulate_prop_line()` (lines 134-164)
**Description:** Backtests use simulated prop lines computed as `0.70 * season_avg + 0.30 * recent_avg`, rounded to nearest 0.5. These are NOT real sportsbook lines.
**Why this is leakage:** The simulated lines are computed from the same player statistics that the model uses as features. The model is essentially predicting against its own inputs, not against an independent market. Real sportsbook lines incorporate information the model does not have (injuries, matchup context, sharp money, public betting %). Simulated lines are systematically easier to beat.
**Affected outputs:** ALL profitability backtest results.
**Likely effect:** Could inflate edge by 2-5 points per prediction. Win rate inflation of 3-8%.
**Permanent fix:** Use historical sportsbook lines from `data/historical_lines/` or an API. If historical lines are unavailable for the test period, the backtest cannot make profitability claims — only accuracy claims.
**Required tests:** Assert that backtest prop lines come from external source, not model features.

### L-03: Post-Hoc Bias Corrections in Comprehensive Backtest (CRITICAL)

**Severity:** CRITICAL
**Files:** `nba_models/backtesting/comprehensive_backtest.py` (lines 1294-1302)
**Description:** Hard-coded bias corrections: points +1.728, rebounds +1.608, assists -0.534, threes +1.161, pra +3.647. Comment says "set to 0 now that feature mismatch is fixed" but values are non-zero.
**Why this is leakage:** Fitting corrections on test data and applying them to the same test data is circular. The corrections perfectly compensate for systematic error on the data they were derived from, but provide zero evidence of out-of-sample performance.
**Affected outputs:** Comprehensive backtest accuracy metrics (RMSE, MAE, bias).
**Likely effect:** Makes bias appear near-zero when actual model bias is 0.5-3.6 points.
**Permanent fix:** If bias corrections are needed, fit them on a separate calibration set BEFORE the test period. Never fit corrections on test data.
**Required tests:** Test that bias corrections are derived from calibration data, not test data.

### L-04: Bias Corrections in Production (CRITICAL)

**Severity:** CRITICAL
**Files:** `nba_betting/constants.py` (lines 46-56), `nba_models/inference/daily_predictions.py` (line 2294)
**Description:** `PROP_BIAS_CORRECTION` values (points +1.38, assists +2.05, etc.) are added to predictions before computing z-scores and over/under probabilities.
**Why this is leakage:** These corrections were derived from backtest data that includes the training period (L-01). They bake in-sample calibration into production predictions. If the model's systematic bias changes (due to retraining, roster changes, rule changes), these fixed corrections become stale and harmful.
**Affected outputs:** ALL production predictions and edges.
**Likely effect:** Currently inflates apparent edge accuracy. Will degrade over time as conditions drift.
**Permanent fix:** Replace with dynamically computed bias from a rolling window of recent SETTLED predictions (out-of-sample only). Track bias drift in calibration_tracker.
**Required tests:** Test that bias corrections are updated after each retrain, not hard-coded.

## HIGH Issues

### L-05: Quantile Decompression Parameters (HIGH)

**Severity:** HIGH
**Files:** `nba_betting/constants.py` (lines 126-132)
**Description:** `QUANTILE_DECOMPRESSION_DEFAULTS` (slope, mean_gap, mean_line per prop type) are used to decompress quantile predictions. These parameters were fitted on historical data that may overlap with training data.
**Why this is leakage:** The decompression corrects for regression-to-mean compression using parameters fit on potentially training-contaminated data.
**Affected outputs:** All quantile-based predictions and probabilities in production.
**Likely effect:** Moderate — recalibrated after each retrain via `scripts/calibrate_quantile_decompression.py`, reducing staleness.
**Permanent fix:** Ensure `calibrate_quantile_decompression.py` uses ONLY out-of-sample predictions for fitting.
**Required tests:** Test that decompression parameters are not fit on training data.

### L-06: Empirical Probability Calibration (HIGH)

**Severity:** HIGH
**Files:** `nba_models/inference/daily_predictions.py`, function `apply_empirical_calibration()`
**Description:** Isotonic regression calibrators in `models/probability_calibrators/` map raw over_prob to calibrated over_prob. Built by `scripts/build_probability_calibration.py`.
**Why this is leakage:** If these calibrators are fit on in-sample predictions (likely given L-01), they encode knowledge of outcomes the model already trained on.
**Affected outputs:** All calibrated probabilities in production.
**Likely effect:** Makes probabilities appear well-calibrated when they may not be on truly unseen data.
**Permanent fix:** Fit calibrators exclusively on out-of-sample predictions from walk-forward validation.
**Required tests:** Verify calibrator training uses only held-out predictions.

### L-07: Feature-Derived Confidence Multipliers (HIGH)

**Severity:** HIGH
**Files:** `nba_models/inference/daily_predictions.py` (lines 2446-2463)
**Description:** Confidence multipliers per prop type (assists: 4.9, points: 1.6, rebounds: 3.9, threes: 8.4, pra: 1.3) were "calibrated" from a "61K prediction backtest."
**Why this is leakage:** These multipliers were tuned to match backtest accuracy, which is in-sample (L-01). They will not generalize.
**Affected outputs:** Confidence scores, bet filtering, bet sizing.
**Likely effect:** Over-confident or under-confident predictions.
**Permanent fix:** Derive from out-of-sample walk-forward results only.

## MEDIUM Issues

### L-08: Training Data Includes Test Season Context (MEDIUM)

**Severity:** MEDIUM
**Files:** `nba_models/backtesting/profitability_backtest.py` (line 63)
**Description:** `CONTEXT_SEASONS = ["2022-23", "2023-24"]` — both seasons loaded for feature computation. Features are point-in-time safe, but the model was trained on both.
**Permanent fix:** Same as L-01.

### L-09: No Temporal Holdout Enforcement in CI (MEDIUM)

**Severity:** MEDIUM
**Files:** `.github/workflows/weekly-retrain.yml`
**Description:** The daily retrain workflow trains on all available data without explicitly holding out recent games for validation. No automated check that test data was excluded from training.
**Permanent fix:** Add explicit temporal holdout (e.g., last 30 days never in training). Add CI check.

### L-10: Settlement Uses Same-Day Stats (MEDIUM)

**Severity:** MEDIUM
**Files:** `nba_betting/settle_trades.py`
**Description:** Settlement fetches actual stats from Balldontlie API. No check that predictions were made BEFORE games started. If predictions are re-generated mid-game and settled, that is leakage.
**Permanent fix:** Add `predicted_at` timestamp. Settlement must verify `predicted_at < game_start_time`.

## LOW Issues

### L-11: OT-Normalized Stats in Training (LOW)

**Severity:** LOW
**Files:** `nba_models/training/train_complete_balldontlie.py`
**Description:** Training uses raw box-score stats including overtime. Most sportsbooks settle player props on regulation stats only.
**Permanent fix:** Document settlement rules. If regulation-only, normalize training labels.

### L-12: Feature Imputation with League Averages (LOW)

**Severity:** LOW
**Files:** Training and inference `smart_fillna()` functions
**Description:** Missing features filled with league averages. If averages include future games, minor temporal leakage.
**Permanent fix:** Compute league averages from past data only.
