# Prioritized Execution Plan — NBA-BETS

## Phase 0: Emergency Fixes (Week 1)

### 0.1 Add Model Quality Gate to Retrain Workflow
**Priority:** P0
**Files:** `.github/workflows/weekly-retrain.yml`
**Action:** After retrain, run quick validation on last 14 days of settled predictions. If accuracy degrades >5%, reject model and alert.
**Why first:** Without this, every daily retrain risks silently deploying a broken model.

### 0.2 Fix In-Sample Backtest
**Priority:** P0
**Files:** `nba_models/backtesting/profitability_backtest.py`
**Action:** Implement proper temporal holdout. Train on seasons <= 2022-23, test on 2023-24. This is the MINIMUM change needed to produce trustworthy profitability numbers.
**Blocked by:** Nothing
**Unblocks:** All downstream profitability claims

## Phase 1: Trustworthy Evaluation (Weeks 1-2)

### 1.1 Replace Simulated Lines with Historical Lines
**Priority:** P0
**Files:** `nba_models/backtesting/profitability_backtest.py`
**Action:** Load real sportsbook lines from `data/historical_lines/` instead of `simulate_prop_line()`. For dates without historical lines, use backfill from Odds API.
**Why:** Without real lines, profitability backtest is meaningless.

### 1.2 Remove Post-Hoc Bias Corrections from Comprehensive Backtest
**Priority:** P1
**Files:** `nba_models/backtesting/comprehensive_backtest.py`
**Action:** Remove `BIAS_CORRECTIONS` dict (lines 1296-1302). Report raw model accuracy.

### 1.3 Unify Feature Generation Paths
**Priority:** P1
**Files:** `nba_data/transformers/feature_engineering.py`, `nba_data/transformers/feature_generator.py`, `nba_models/inference/daily_predictions.py`
**Action:** Create a single `generate_features(player, game_context, point_in_time_date)` function used by training, inference, AND backtesting. Delete duplicate feature generators.

### 1.4 Align Calibration Across Paths
**Priority:** P1
**Files:** `nba_models/inference/daily_predictions.py`, `nba_models/backtesting/profitability_backtest.py`
**Action:** Backtest must apply the SAME prediction pipeline as production (including all calibration layers). Either simplify production to match backtest, or make backtest call the production prediction function.

### 1.5 Add Critical Regression Tests
**Priority:** P1
**Files:** `tests/`
**Action:** Add tests for:
- Train/inference feature parity (G-01)
- Settlement correctness (G-04)
- Spread sign convention (G-07)
- Disabled props actually disabled (G-11)
- Devigging correctness (G-09)

## Phase 2: Market Realism (Weeks 2-3)

### 2.1 Implement CLV Tracking
**Priority:** P1
**Files:** New: `nba_betting/clv_tracker.py`
**Action:** After each game, compute CLV = closing_line - prediction_line. Track rolling CLV by prop type. This is the primary metric for determining if the model has genuine edge.

### 2.2 Use Real Odds in Backtest
**Priority:** P1
**Files:** `nba_models/backtesting/profitability_backtest.py`
**Action:** Replace fixed -110 with actual odds from historical data or API. Update devigging to use real odds.

### 2.3 Add Decision-Time Line Capture
**Priority:** P2
**Files:** `nba_models/inference/daily_predictions.py`, migrations
**Action:** Store `prediction_timestamp`, `line_at_prediction_time`, and later `closing_line`. Add migration for new columns.

### 2.4 Handle OT Settlement Correctly
**Priority:** P2
**Files:** Training data, settlement
**Action:** Document settlement rules. Add `includes_ot` flag. If regulation-only settlement, adjust training labels.

## Phase 3: Availability Hardening (Weeks 3-4)

### 3.1 Align Minute Thresholds
**Priority:** P2
**Files:** Backtest files, constants
**Action:** Use consistent minimum minutes threshold across all code paths (recommend 10 min).

### 3.2 Add Scratch/Void Handling to Settlement
**Priority:** P2
**Files:** `nba_betting/settle_trades.py`, `nba_betting/paper_trading.py`
**Action:** If player has 0 minutes in settlement, mark as VOID (no P&L), not loss.

### 3.3 Integrate Injury Data into Backtests
**Priority:** P2
**Files:** Backtest files
**Action:** Use historical injury reports to simulate availability gating in backtests.

## Phase 4: Artifact and Deployment (Weeks 4-5)

### 4.1 Add Model Provenance to Artifacts
**Priority:** P2
**Files:** Training script
**Action:** Store git_commit, training_data_hash, date_range, hyperparameters in every .pkl.

### 4.2 Move Models Out of Git
**Priority:** P3
**Files:** Workflows, deployment
**Action:** Store models in cloud storage. Track metadata in git. Reduce repo bloat.

### 4.3 Expand Test Coverage to Critical Paths
**Priority:** P2
**Files:** `pytest.ini`, `tests/`
**Action:** Expand coverage to `nba_models/`, `nba_betting/`, `edge_calculator/`.

### 4.4 Route Model Saves Through Registry
**Priority:** P3
**Files:** Training script, `continuous_learning/model_registry.py`
**Action:** All model saves go through registry. Enables rollback and version comparison.

## Phase 5: Code Cleanup (Weeks 5-6)

### 5.1 Remove Dead Root-Level Scripts
**Priority:** P3
**Action:** Delete or archive ~40 dead/legacy root-level Python files.

### 5.2 Consolidate Edge Calculation
**Priority:** P3
**Action:** Edge calculated in 4 places. Consolidate to single function.

### 5.3 Remove Dashboard (Legacy Dash)
**Priority:** P3
**Action:** `dashboard/` fully replaced by React frontend. Archive it.

### 5.4 Dynamic Bias Correction
**Priority:** P2
**Files:** `nba_betting/constants.py`, new calibration pipeline
**Action:** Replace hard-coded `PROP_BIAS_CORRECTION` with dynamically computed values from rolling window of settled predictions. Update after each settlement run.

---

## Success Criteria

After completing Phases 0-2:
1. Backtest uses real sportsbook lines
2. Backtest uses proper temporal holdout (no in-sample model)
3. Production and backtest use identical prediction pipeline
4. CLV is tracked for every prediction
5. Model quality gate prevents deploying worse models
6. Critical regression tests exist for known production bugs

**Expected outcome:** Reported profitability will likely DROP significantly (possibly to breakeven or negative) once leakage and market-realism issues are fixed. This is the correct outcome. If the model still shows positive CLV after these fixes, you have a genuinely profitable system.
