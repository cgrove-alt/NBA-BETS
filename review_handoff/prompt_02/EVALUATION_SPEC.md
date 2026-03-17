# Canonical Evaluation Specification — NBA-BETS

**Version:** 1.0.0
**Date:** 2026-03-17
**Status:** DRAFT — requires implementation

---

## C1. Evaluation Modes

### Mode 1: RESEARCH-ONLY Backtest

**Purpose:** Explore model quality, feature engineering, stat prediction accuracy.
**Allowed:**
- Simulated prop lines (e.g., 70/30 season/recent blend)
- Fixed -110 odds on both sides
- In-sample model weights (train and test overlap temporally)
- Hardcoded bias corrections
- Imputed league-average features

**Forbidden:**
- Labeling results as "profitable" or "production-ready"
- Using results to justify bet sizing or capital allocation
- Omitting the RESEARCH-ONLY watermark from any output

**Permitted claims:** "Model RMSE is X", "Directional accuracy is Y%", "Feature importance ranking"
**Prohibited claims:** "ROI is X%", "System is profitable", "Edge is X%"
**Required watermark:** Every output file MUST contain header: `REALISM_LEVEL: RESEARCH-ONLY — NOT PROFITABILITY EVIDENCE`

**Current repo status:**
- `nba_models/backtesting/profitability_backtest.py` uses simulated lines (L134 `simulate_prop_line()`), fixed -110 odds (L60), and in-sample model (L11 docstring admits this).
- This backtest MUST be relabeled RESEARCH-ONLY.

### Mode 2: MARKET-REALISTIC Historical Simulation

**Purpose:** Estimate real-world profitability using historical sportsbook data.
**Allowed:**
- Real sportsbook lines from `data/historical_lines/` at decision time
- Real odds from the same source
- Walk-forward model retraining (no test-period data in training)
- Settlement against raw box-score stats matching sportsbook rules

**Forbidden:**
- Simulated lines
- Fixed -110 odds (unless odds data genuinely missing AND run is downgraded to research-only)
- In-sample model weights
- Post-hoc bias corrections fitted on test data
- Omitting CLV computation when closing lines are available

**Permitted claims:** "Estimated ROI on historical data is X% ± CI", "CLV is Y cents/bet"
**Prohibited claims:** "System IS profitable" (requires paper-trading confirmation)
**Required watermark:** `REALISM_LEVEL: MARKET-REALISTIC — HISTORICAL SIMULATION`

**Gate conditions:** Must pass ALL gates in REALISM_CHECKLIST.md (Gates 1-8).

**Current repo status:** This mode DOES NOT EXIST in the repo. No backtest uses real historical lines for player props.

### Mode 3: Paper-Trading / Shadow Evaluation

**Purpose:** Forward-looking validation on live predictions vs live outcomes.
**Allowed:**
- Real-time predictions generated before game start
- Real lines captured at decision time
- Real settlement from box scores
- Tracking of closing lines for CLV

**Forbidden:**
- Retroactive prediction generation
- Editing predictions after game start
- Settling against non-final box scores

**Permitted claims:** "Forward CLV over N bets is X", "Live win rate is Y%"
**Prohibited claims:** "System is profitable" (requires sufficient sample size per PROMOTION_CRITERIA)
**Required watermark:** `REALISM_LEVEL: PAPER-TRADING — FORWARD VALIDATION`

**Current repo status:**
- `nba_betting/paper_trading.py` and `nba_betting/settle_trades.py` provide infrastructure.
- Missing: decision-time line capture, closing-line capture, CLV computation.
- Predictions are generated daily via `.github/workflows/predict-daily.yml` (L40-48).

### Mode 4: Production Model Comparison

**Purpose:** Compare candidate model vs incumbent using same live bet stream.
**Allowed:**
- A/B shadow predictions on same games
- Identical line/odds data for both models
- CLV comparison as primary metric

**Forbidden:**
- Comparing models trained on different data windows without disclosure
- Cherry-picking date ranges
- Using different settlement rules for each model

**Permitted claims:** "Model B has +X CLV improvement over Model A on N shared bets"
**Required watermark:** `REALISM_LEVEL: PRODUCTION-COMPARISON`

**Current repo status:** No comparison infrastructure exists.

---

## C2. Walk-Forward Retraining

**Policy:** See DATA_SPLIT_POLICY.md for full specification.

**Summary:**
- Rolling window: train on most recent T days, validate on next V days, test on next E days.
- Retrain cadence: every 14 days (adjustable).
- ABSOLUTE RULE: No game played on or after `test_start_date` may appear in training data.
- Season boundaries: Season start = October 22 (typical). Prior-season data allowed in training but weighted by time decay.

**Current repo violations:**
- `nba_models/backtesting/profitability_backtest.py:L61-63`: `TEST_SEASON = "2023-24"`, `CONTEXT_SEASONS = ["2022-23", "2023-24"]` — model trained on test season.
- `.github/workflows/weekly-retrain.yml`: Trains on all available data without holdout.

---

## C3. Market Data Requirements

**Per-bet minimum fields:** See BET_RECORD_SCHEMA.md for full schema.

**Critical requirements:**
1. `decision_line` and `decision_odds` MUST come from a real sportsbook snapshot, NOT from model features.
2. `snapshot_timestamp` MUST be recorded at the time the line/odds were observed.
3. `closing_line` and `closing_odds` MUST be captured after game start for CLV computation.
4. If `closing_line` is NULL, the bet record MUST be flagged `clv_available = false` and the run CANNOT be labeled PRODUCTION-LIKE.

**Current repo status:**
- `data/historical_lines/*.json` contains `snapshot_timestamp` per game and player prop lines with bookmaker odds.
- **BLOCKER:** No closing line concept exists in the data. Files contain a single snapshot, not opening/closing pairs.
- `nba_models/inference/daily_predictions.py:L84` imports `PROP_BIAS_CORRECTION` — bias corrections applied before edge, not tracked in output.

---

## C4. Settlement Rules

**Rule 1:** Settlement MUST use raw sportsbook-equivalent stats.
- Points, rebounds, assists, threes: regulation + OT (matching most US sportsbooks for player props).
- If a specific sportsbook uses regulation-only, that MUST be documented and the settlement function parameterized.

**Rule 2:** Push handling: `actual == line` → result = PUSH, PnL = 0.
- Current: `nba_models/backtesting/profitability_backtest.py:L388-390` skips pushes (correct).

**Rule 3:** DNP/Scratch → result = VOID, PnL = 0.
- If player has 0 minutes played, bet is VOID.
- Current: `nba_betting/settle_trades.py:L15-20` does NOT check for DNP. No void logic exists.

**Rule 4:** Settlement data source: BallDontLie API box scores (current: `settle_trades.py:L38`), only for games with status "Final" (L51-54).

---

## C5. No Simulated Lines Rule

**ABSOLUTE RULE:** Any evaluation run that uses lines generated from model features (e.g., `simulate_prop_line()` at `profitability_backtest.py:L134`) MUST be labeled `REALISM_LEVEL: RESEARCH-ONLY`.

**Enforcement:** Gate 1 in REALISM_CHECKLIST.md.

**Current violations:**
- `profitability_backtest.py:L134-164`: `simulate_prop_line()` generates lines as `0.70 * season_avg + 0.30 * recent_avg`.
- `comprehensive_backtest.py`: Also uses simulated lines (no historical line lookup).

---

## C6. JSON Schemas

### Per-Bet Record Schema

```json
{
  "event_id": {"type": "string", "required": true},
  "game_id": {"type": "integer", "required": true},
  "player_id": {"type": "integer", "required": false},
  "market_type": {"type": "string", "enum": ["player_points", "player_rebounds", "player_assists", "player_threes", "player_pra", "moneyline", "spread", "total"], "required": true},
  "side": {"type": "string", "enum": ["over", "under", "home", "away"], "required": true},
  "decision_timestamp": {"type": "string", "format": "ISO8601", "required": true},
  "snapshot_timestamp": {"type": "string", "format": "ISO8601", "required": true},
  "decision_line": {"type": "number", "required": true},
  "decision_odds": {"type": "integer", "required": true, "description": "American odds"},
  "book": {"type": "string", "required": true},
  "closing_line": {"type": "number", "required": false},
  "closing_odds": {"type": "integer", "required": false},
  "model_fair_probability": {"type": "number", "required": true, "minimum": 0, "maximum": 1},
  "market_implied_probability": {"type": "number", "required": true, "minimum": 0, "maximum": 1},
  "vig_adjusted_probability": {"type": "number", "required": true, "minimum": 0, "maximum": 1},
  "raw_edge": {"type": "number", "required": true},
  "vig_adjusted_edge": {"type": "number", "required": true},
  "calibrated_probability": {"type": "number", "required": false, "minimum": 0, "maximum": 1},
  "uncertainty_score": {"type": "number", "required": false},
  "availability_flags": {"type": "string", "required": false, "enum": ["available", "questionable", "gtd", "doubtful", "out", "unknown"]},
  "accepted": {"type": "boolean", "required": true},
  "pass_reason": {"type": "string", "required": false, "description": "Reason if rejected"},
  "result": {"type": "string", "enum": ["win", "lose", "push", "void", "pending"], "required": true},
  "CLV": {"type": "number", "required": false, "description": "closing_line - decision_line, positive = got better line"},
  "PnL": {"type": "number", "required": true},
  "stake": {"type": "number", "required": true},
  "artifact_version": {"type": "string", "required": true},
  "git_sha": {"type": "string", "required": false},
  "realism_level": {"type": "string", "enum": ["RESEARCH-ONLY", "MARKET-REALISTIC", "PAPER-TRADING", "PRODUCTION-COMPARISON"], "required": true}
}
```

### Model Artifact Metadata Schema

```json
{
  "artifact_version": {"type": "string", "required": true, "description": "Unique version ID, e.g. 20260317_083000"},
  "git_sha": {"type": "string", "required": true},
  "model_family": {"type": "string", "required": true, "enum": ["ensemble", "quantile", "position_aware", "minutes_oracle", "moneyline", "spread"]},
  "target_market_type": {"type": "string", "required": true},
  "train_window_start": {"type": "string", "format": "date", "required": true},
  "train_window_end": {"type": "string", "format": "date", "required": true},
  "feature_schema_version": {"type": "string", "required": true},
  "calibration_version": {"type": "string", "required": false},
  "data_snapshot_id": {"type": "string", "required": false},
  "training_timestamp": {"type": "string", "format": "ISO8601", "required": true},
  "hyperparams_hash": {"type": "string", "required": false},
  "training_samples": {"type": "integer", "required": true},
  "validation_metrics": {"type": "object", "required": false}
}
```

---

## C7. Realism Level Header

Every evaluation output file (JSON, CSV, JSONL, report) MUST include a top-level field:
```
realism_level: "RESEARCH-ONLY" | "MARKET-REALISTIC" | "PAPER-TRADING" | "PRODUCTION-COMPARISON" | "INVALID"
```

**Gate conditions controlling the label:**
- PRODUCTION-LIKE requires ALL gates in REALISM_CHECKLIST.md to PASS.
- MARKET-REALISTIC requires Gates 1-6 to PASS.
- RESEARCH-ONLY: any run that fails Gate 1 or Gate 4.
- INVALID: any run that fails Gate 7 (schema completeness).

---

## C8. Implementation Hooks

Functions to implement (signatures only — do NOT implement in this prompt):

```python
def run_walk_forward_backtest(
    model_dir: str,
    historical_lines_dir: str,
    train_window_days: int,
    val_window_days: int,
    test_window_days: int,
    retrain_cadence_days: int,
    settlement_source: str,  # "balldontlie" | "csv"
) -> BacktestResult: ...

def save_bet_log(
    bets: list[BetRecord],
    output_path: str,
    realism_level: str,
) -> None: ...

def compute_clv(
    decision_line: float,
    closing_line: float,
    side: str,
) -> float: ...

def validate_realism_gates(
    bet_log_path: str,
    artifact_metadata_path: str,
) -> RealismCheckResult: ...

def generate_evaluation_report(
    bet_log_path: str,
    realism_level: str,
    output_dir: str,
) -> ReportBundle: ...
```

**Likely files to edit during implementation:**
- `nba_models/backtesting/profitability_backtest.py` — replace simulated lines, add walk-forward retraining
- `nba_models/inference/daily_predictions.py` — add decision-time line capture to prediction output
- `nba_betting/settle_trades.py` — add DNP/void logic
- NEW: `nba_models/evaluation/bet_record.py` — BetRecord dataclass
- NEW: `nba_models/evaluation/realism_gates.py` — gate validation
- NEW: `nba_models/evaluation/clv.py` — CLV computation
- NEW: `nba_models/evaluation/report_generator.py` — canonical report generation
