# Test Coverage Gap Audit — NBA-BETS

## Current Test Inventory

**Location:** `tests/` (46 files, ~14,559 lines)
**Runner:** pytest with coverage reporting (via `pytest.ini`)
**CI:** GitHub Actions `quality-checks.yml` runs `pytest tests/ -v --cov=backend -k "not slow" --maxfail=5`

## What Has Tests

| Component | Test File(s) | Coverage Quality |
|-----------|-------------|-----------------|
| Agent framework | `test_agent_framework.py` + 6 agent-specific tests | **Good** |
| Stacking models | `test_stacking.py`, `test_stacking_integration.py`, `test_stacking_context_features.py` | **Good** |
| Confidence scoring | `test_confidence_scoring.py`, `test_confidence_formulas.py` | **Moderate** |
| Edge calculations | `test_edge_calculations.py`, `test_phase4_edge_reframing.py` | **Moderate** — edge has 4 implementations |
| Risk management | `test_risk_management.py` | **Good** |
| Data validation | `test_data_validation.py` | **Good** |
| Betting features | `test_betting_features.py` | **Good** |
| Advanced stats | `test_advanced_stats.py`, `test_four_factors.py` | **Good** |
| Deployment config | `test_deployment_config.py` | **Good** |
| Canonical constants | `test_canonical_constants.py` | **Good** |
| Calibration | `test_phase5_calibration.py` | **Moderate** |

## CRITICAL Gaps (No Tests)

### G-01: No Test for Train/Inference Feature Parity

**Risk:** Feature engineering uses different code paths for training (`feature_engineering.py`) vs inference (`feature_generator.py`). No test verifies both produce identical feature vectors for the same input.
**Impact:** Silent train/serve skew could make all predictions systematically wrong.
**Required test:** Given a player-game input, assert that training feature generator and inference feature generator produce identical feature dictionaries.

### G-02: No Test for Backtest/Production Prediction Parity

**Risk:** Profitability backtest and production inference apply different calibration, bias corrections, and adjustments. No test verifies equivalence.
**Impact:** Backtest results are unreliable predictors of production behavior.
**Required test:** Given identical model and features, assert that backtest and production prediction paths produce the same output.

### G-03: No Test for Temporal Leakage in Training

**Risk:** No automated test ensures training data excludes future information.
**Impact:** Undetected temporal leakage invalidates all profitability claims.
**Required test:** Assert for every training sample: all features computed from data with dates strictly before the sample's game date.

### G-04: No Test for Settlement Correctness

**Risk:** `settle_trades.py` maps prop types to stat fields but has no tests.
**Impact:** Settlement bugs could reverse win/loss grading.
**Required test:** Given known player stats, assert correct win/loss/push grading for each prop type and direction (over/under).

### G-05: No End-to-End Prediction Pipeline Test

**Risk:** No test runs full pipeline from features -> model -> edge -> filter -> sizing -> settlement.
**Impact:** Integration bugs between stages go undetected.
**Required test:** Mock full pipeline with known inputs, verify complete output chain.

### G-06: No Test for Model Artifact Compatibility

**Risk:** Pickle artifacts loaded by different code paths with different format expectations.
**Impact:** After retraining, models could fail to load silently, causing fallback to wrong predictions.
**Required test:** After each retrain, verify every .pkl loads successfully in all consuming code paths.

### G-07: No Regression Test for Spread Sign Convention

**Risk:** Previous production bug where spread sign was inverted. No regression test prevents recurrence.
**Impact:** Spread bets placed backwards.
**Required test:** Assert positive spread in training = home favored, verify sportsbook convention conversion.

## HIGH Gaps

### G-08: No Test for Bias Correction Application

`PROP_BIAS_CORRECTION` applied in production but not backtest. No test verifies correct application.
**Required test:** Assert corrections applied exactly once, correct direction, correct stage.

### G-09: No Test for Devigging Correctness

Devigging implemented in multiple places. No test verifies math.
**Required test:** Assert devig(-110, -110) -> (0.5, 0.5), devig(-150, +130) -> known values.

### G-10: No Test for Kelly Sizing Bounds

Kelly has min (0.5%) and max (3%) bounds plus tier fractions. No test verifies bounds.
**Required test:** Assert bet_size always within [0.5%, 3%] for all tiers.

### G-11: No Test for Disabled Props Actually Disabled

`DISABLED_PROPS = ['threes', 'spread', 'assists']` but no test verifies filtering in all code paths.
**Required test:** Assert evaluate_bet() returns should_bet=False for all disabled props.

## MEDIUM Gaps

### G-12: No Test for Minutes Oracle Feature Generation

No test that `MinutesFeatureGenerator.generate_features()` produces correct 38 features.

### G-13: No Test for Injury Impact Calculation

`InjuryImpactCalculator` has complex star vs role player impact logic, no dedicated test.

### G-14: No Test for Quantile Decompression

`decompress_quantile_prediction()` applies slope/gap corrections, no test for math correctness.

### G-15: No Test for Paper Trading Void on DNP

No test that paper trades are voided (not counted as losses) when player has 0 minutes.

## Misleading Test Coverage

### G-16: CI Coverage Only Measures `backend/`

**Issue:** `pytest.ini` configures coverage for `backend/` only. CI reports show 0% for ALL of: `nba_models/`, `nba_betting/`, `nba_data/`, `edge_calculator/`, `minutes_oracle/`, `calibration_tracker/`.
**Impact:** False sense of coverage. Real critical code unmeasured.
**Fix:** Expand coverage to all Python packages.

### G-17: Tests Use Mocks That May Diverge from Reality

Many tests mock API responses and model outputs. If real APIs change schema or models change output format, tests still pass but production breaks.
**Fix:** Add integration tests that hit real APIs (marked `slow`/`integration`).

## Required Regression Tests (Based on Known Production Bugs)

1. **Spread sign convention** — previous bug, no regression test
2. **Injury boost type conversion** — previous bug (string -> bool), partially fixed
3. **Frontend-backend prediction mismatch** — previous bug, threshold alignment
4. **Git push conflict in CI** — previous bug in predict-daily.yml
5. **DATABASE_URL missing in workflow steps** — previous bug, needs test
