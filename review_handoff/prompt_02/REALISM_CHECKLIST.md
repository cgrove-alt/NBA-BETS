# Realism Checklist — NBA-BETS

**Version:** 1.0.0

ALL gates must PASS for a run to be labeled `MARKET-REALISTIC` or `PRODUCTION-LIKE`.
Any gate failure downgrades the run to `RESEARCH-ONLY`.

---

## Gate 1: No Simulated Lines

**Rule:** `decision_line` must come from an external sportsbook source, NOT from model features or player averages.

**Pass condition:** No bet record in the log has `decision_line` computed via `simulate_prop_line()` or equivalent.

**Current violation:**
- `nba_models/backtesting/profitability_backtest.py:L134-164` — `simulate_prop_line()` computes lines as `0.70 * season_avg + 0.30 * recent_avg`.

**Assertion location:** `tests/eval_spec_tests/test_gate_01_no_simulated_lines.py`

**Implementation location:** MISSING — needs new validation function in `nba_models/evaluation/realism_gates.py`

---

## Gate 2: Decision-Time Line and Odds Present

**Rule:** Every accepted bet must have non-null `decision_line`, `decision_odds`, `snapshot_timestamp`, and `book`.

**Pass condition:** For all records where `accepted = true`: `decision_line IS NOT NULL AND decision_odds IS NOT NULL AND snapshot_timestamp IS NOT NULL AND book IS NOT NULL`.

**Current violation:**
- `nba_models/inference/daily_predictions.py` outputs predictions to CSV (`L3318-3376`) but does not store `snapshot_timestamp` or `book` per bet. Lines are stored but may be simulated during backtests.

**Assertion location:** `tests/eval_spec_tests/test_gate_02_decision_time_line_present.py`

**Implementation location:** MISSING — `nba_models/evaluation/realism_gates.py`

---

## Gate 3: Closing Line and Odds Present for CLV

**Rule:** For CLV to be computed, `closing_line` and `closing_odds` must be captured after game start.

**Pass condition:** At least 90% of accepted bet records have non-null `closing_line`.

**Current violation:**
- **BLOCKER:** `data/historical_lines/*.json` contains only a single snapshot per game. No closing line concept exists. Key `closing_line` absent from all 166 files.
- No code captures closing lines. No database table stores closing lines.

**Assertion location:** `tests/eval_spec_tests/test_gate_03_closing_line_present.py`

**Implementation location:** MISSING — requires new data ingestion job and schema addition.

---

## Gate 4: No Test-Period Data in Training

**Rule:** Model weights used for test-period predictions must be trained ONLY on data with `game_date < test_start_date`.

**Pass condition:** `artifact_metadata.train_window_end < test_window_start` for the artifact used.

**Current violation:**
- `profitability_backtest.py:L11`: "model weights were fit on data that includes 2023-24" — the test period.
- `.github/workflows/weekly-retrain.yml`: trains on all available data with no holdout.

**Assertion location:** `tests/eval_spec_tests/test_gate_04_no_test_period_training.py`

**Implementation location:** MISSING — requires artifact metadata schema enforcement.

---

## Gate 5: Real Odds Present OR Run Labeled Research-Only

**Rule:** If `decision_odds` is a hardcoded constant (e.g., -110 for all bets), the run MUST be labeled `RESEARCH-ONLY`.

**Pass condition:** `decision_odds` values show variance across bets (not all identical), OR `realism_level = "RESEARCH-ONLY"`.

**Current violation:**
- `profitability_backtest.py:L60`: `STANDARD_ODDS = -110` used for all bets.
- `profitability_backtest.py:L356-357`: `over_odds=STANDARD_ODDS, under_odds=STANDARD_ODDS`.

**Assertion location:** `tests/eval_spec_tests/test_gate_05_real_odds_or_research.py`

---

## Gate 6: Settlement Supports VOID for DNP

**Rule:** If a player has 0 minutes played, the bet result MUST be `void` with `PnL = 0`.

**Pass condition:** Settlement function checks minutes played and assigns `result = "void"` for DNP.

**Current violation:**
- `nba_betting/settle_trades.py:L15-91`: No check for minutes played. No void concept.
- `continuous_learning/settlement_service.py`: No DNP check.

**Assertion location:** `tests/eval_spec_tests/test_gate_06_settlement_void_dnp.py`

**Implementation location:** `nba_betting/settle_trades.py` — add minutes check.

---

## Gate 7: Bet Record Schema Completeness

**Rule:** Every bet record must have all REQUIRED fields non-null as defined in BET_RECORD_SCHEMA.md.

**Pass condition:** Schema validation passes for all records.

**Current violation:**
- Current prediction output CSV (`daily_predictions.py:L3318-3376`) has 37 columns but is missing: `event_id`, `snapshot_timestamp`, `book`, `closing_line`, `closing_odds`, `market_implied_probability`, `vig_adjusted_probability`, `raw_edge`, `vig_adjusted_edge`, `accepted`, `result`, `CLV`, `PnL`, `stake`, `artifact_version`, `git_sha`, `realism_level`.

**Assertion location:** `tests/eval_spec_tests/test_gate_07_schema_completeness.py`

---

## Gate 8: Artifact Metadata Completeness

**Rule:** Every model artifact used in evaluation must have metadata conforming to MODEL_ARTIFACT_SCHEMA.md.

**Pass condition:** `artifact_version`, `git_sha`, `train_window_start`, `train_window_end`, `training_timestamp`, `training_samples`, `feature_names` all present.

**Current violation:**
- Model artifacts (e.g., `models/moneyline_ensemble.pkl`) contain `saved_at` but NOT `git_sha`, `train_window_start`, `train_window_end`, or `training_samples`.

**Assertion location:** `tests/eval_spec_tests/test_gate_08_artifact_metadata.py`
