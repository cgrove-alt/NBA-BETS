# Train / Inference / Backtest Parity Audit — NBA-BETS

## Executive Summary

There are significant parity issues across the three code paths. The most critical: the backtest uses simulated lines while production uses real lines; the backtest applies no post-prediction adjustments while production applies 5+ adjustment layers; and feature generation follows different code paths in training vs inference.

---

## Feature Definitions

### Training Features
**Source:** `nba_models/training/train_complete_balldontlie.py`, function `process_games_for_training()` (line 2965)
**Generator:** `nba_data/transformers/feature_engineering.py`
**Feature count:** ~150 per player-game sample
**Point-in-time:** YES

### Inference Features
**Source:** `nba_models/inference/daily_predictions.py`, function `get_cached_features()` (line 1002), `generate_complete_prop_features()` (line 764)
**Generator:** `nba_data/transformers/feature_generator.py`
**Feature count:** ~150
**Point-in-time:** YES (inherently, for live inference)

### Backtest Features
**Profitability backtest:** Uses `process_games_for_training()` — SAME as training.
**Comprehensive backtest:** Uses `get_player_features_before_date()` — CUSTOM implementation with 43 DEFAULT feature values.

### PARITY ISSUE P-01: Feature Generator Drift (CRITICAL)

**Description:** Training uses `nba_data/transformers/feature_engineering.py` while inference uses `nba_data/transformers/feature_generator.py`. These are two separate files with potentially different computation logic.
**Risk:** Subtle differences in rolling window calculations, default values, or feature naming cause train/serve skew.
**Evidence:** Inference code (daily_predictions.py line 2128) builds `X = pd.DataFrame([{k: features.get(k, np.nan) for k in feature_names}])`, falling back to NaN for any feature not generated. If inference generates a slightly different feature name, it silently gets NaN-filled.
**Permanent fix:** Unify feature generation into a single code path used by training, inference, AND backtesting.

### PARITY ISSUE P-02: Comprehensive Backtest Feature Defaults (HIGH)

**Description:** `comprehensive_backtest.py` defines 43 NBA-realistic default feature values (lines 43-89). Training and production use `smart_fillna()` with potentially different defaults.
**Risk:** Backtest uses optimistic defaults that make predictions look better than with real missing data.
**Permanent fix:** All three paths must use the same `smart_fillna()` function with same defaults.

---

## Target Construction

### Training
- Props: `actual_pts`, `actual_reb`, `actual_ast`, `actual_fg3m`, `actual_pra` — raw box-score stats (including OT)

### Inference
- Regression value -> z-score -> `norm.cdf()` -> over_prob
- Uses `PROP_BIAS_CORRECTION` (+1.38 pts, +2.05 ast, etc.) before z-score computation

### Backtest
- Profitability: Uses same actuals as training
- Comprehensive: Uses API box-score stats + its OWN different bias corrections

### PARITY ISSUE P-03: Bias Correction Applied Only in Production (CRITICAL)

**Description:** Production inference applies `PROP_BIAS_CORRECTION` to shift predictions before computing z-scores. The profitability backtest does NOT apply these corrections. The comprehensive backtest applies DIFFERENT corrections (points +1.728 vs production +1.38).
**Risk:** Three different bias-handling approaches across three code paths.
**Permanent fix:** Either remove bias corrections entirely and fix the model, OR apply the same corrections in all paths.

---

## Calibration

### Training
- Isotonic calibration fitted during training
- Saved with model artifact

### Inference (5 calibration layers)
1. Temperature scaling (T=2.0) in `prediction_pipeline.py`
2. Empirical isotonic calibration via `apply_empirical_calibration()`
3. Sample size shrinkage via `apply_sample_size_confidence_shrink()`
4. Quantile decompression via `decompress_quantile_prediction()`
5. CalibrationAdjuster via `calibration_tracker/`

### Backtest (Profitability)
- Uses `QuantilePropModel.predict_over_probability()` with `pre_calibrated=True`
- None of the 5 production calibration layers applied

### PARITY ISSUE P-04: Calibration Pipeline Mismatch (CRITICAL)

**Description:** Production applies 5 calibration/adjustment layers. Backtest applies only quantile model probability. These are fundamentally different prediction paths.
**Risk:** Backtest results cannot predict production behavior.
**Permanent fix:** Backtest must use the EXACT same `predict_player_prop()` function as production.

---

## Edge Calculation

### Inference
`_calculate_prop_edge()` in daily_predictions.py — probability-based: `(model_prob - devigged_implied_prob) * 100`

### Backtest (Profitability)
`evaluate_bet()` in prediction_pipeline.py — stat-point-based: `abs(predicted - line)`

### PARITY ISSUE P-05: Edge Computation Differs (HIGH)

**Description:** Production computes edge as probability difference. Backtest computes edge as stat-point difference. These are fundamentally different metrics.
**Risk:** A bet classified as "elite" in the backtest might be "weak" in production.
**Permanent fix:** Use the same edge computation everywhere. Probability-based with devigging is correct for betting.

---

## Spread Sign Conventions

### PARITY ISSUE P-06: Spread Sign Inconsistency (MEDIUM)

**Description:** Training uses `home - away` (positive = home won by X). Sportsbook data uses negative = home favored. Previous production bug.
**Current mitigation:** Spread betting is DISABLED.
**Permanent fix:** Resolve sign convention before re-enabling spread betting.

---

## Availability Gating

### PARITY ISSUE P-07: Availability Asymmetry (HIGH)

| Gate | Inference | Profitability BT | Comprehensive BT |
|------|-----------|-----------------|-------------------|
| Injury status check | OUT/DOUBTFUL skip | No | No |
| Uncertainty flag | QUESTIONABLE/GTD flagged | No | No |
| Minutes oracle adjustment | Yes | No | Optional |
| Injury boost | Yes | No | No |
| Minutes threshold | None | 15 min | 0.1 min |
| Sample size gate | 10+ games | Via evaluate_bet() | No |

**Risk:** Production has sophisticated availability gating. Backtests use only post-hoc minutes filtering. Backtests benefit from perfect hindsight on who actually played.
**Permanent fix:** Incorporate historical injury reports into backtests.

---

## Model Artifact Usage

### PARITY ISSUE P-08: Model Loading Path Divergence (MEDIUM)

**Description:** Inference handles 4+ model formats with fallbacks in `predict_player_prop()`. Backtest uses class-based `.load()` methods from `PropEnsembleModel`.
**Risk:** If artifact format changes, one path may work while the other breaks silently.
**Permanent fix:** Unify model loading into single `load_model(path, model_type)` function.

---

## Bet Sizing

### PARITY ISSUE P-09: Flat vs Compounding Kelly (LOW)

**Description:** Backtest uses INITIAL_BANKROLL for all Kelly calculations (flat sizing, intentionally conservative). Production uses actual bankroll (compounding).

---

## Settlement

### PARITY: Settlement logic is consistent across paths.

- Production: Balldontlie API actuals, maps prop_type to stat field
- Backtest: Training data actuals, same mapping
- Push: actual == line = skip (no P&L)
- P&L: won = `bet_size * (100/110)`, lost = `-bet_size`

---

## Summary Matrix

| Aspect | Train | Inference | Profitability BT | Comprehensive BT | Parity |
|--------|-------|-----------|-------------------|-------------------|--------|
| Feature gen code path | feature_engineering.py | feature_generator.py | feature_engineering.py | custom | FAIL |
| Feature defaults | smart_fillna | smart_fillna | smart_fillna | 43 hardcoded | FAIL |
| Bias correction | None | PROP_BIAS_CORRECTION | None | Own corrections | FAIL |
| Calibration layers | Isotonic (training) | 5 layers | Quantile only | None | FAIL |
| Edge calc method | N/A | prob-based (devig) | stat-point-based | N/A | FAIL |
| Prop lines | N/A | Real (API) | Simulated | Simulated | FAIL |
| Odds | N/A | Real (API) | Fixed -110 | N/A | FAIL |
| Availability | N/A | Injury tracker + oracle | Post-hoc minutes | Post-hoc minutes | FAIL |
| Model loading | pickle.dump | 4+ format handlers | class.load() | class.load() | WARN |
| Settlement | N/A | API actuals | Training actuals | API actuals | OK |
| Kelly sizing | N/A | Tiered, real odds | Flat, -110 | N/A | WARN |
