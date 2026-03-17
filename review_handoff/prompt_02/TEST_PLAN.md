# Test Plan — Evaluation Spec

## Implemented Unit Tests

| Test File | Gate | Asserts | Expected Result |
|-----------|------|---------|-----------------|
| `test_gate_01_no_simulated_lines.py` | Gate 1 | profitability_backtest uses simulated lines without RESEARCH-ONLY label; real_lines_backtest uses historical data | FAIL (violation documented) |
| `test_gate_02_decision_time_line_present.py` | Gate 2 | historical_lines/ exists, has snapshot_timestamp, has bookmaker+odds | PASS (data exists) |
| `test_gate_03_closing_line_present.py` | Gate 3 | closing_line field exists in historical data; compute_clv function exists | FAIL (BLOCKER: no closing lines) |
| `test_gate_04_no_test_period_training.py` | Gate 4 | profitability_backtest admits in-sample without label; model artifacts have train_window metadata | FAIL (in-sample + missing metadata) |
| `test_gate_05_real_odds_or_research.py` | Gate 5 | profitability_backtest uses fixed -110 without label; real_lines_backtest uses real odds | FAIL (no label on fixed odds) |
| `test_gate_06_settlement_void_dnp.py` | Gate 6 | settle_trades.py has void/DNP logic; paper_trading.py has void logic | FAIL (no void logic) |
| `test_gate_07_schema_completeness.py` | Gate 7 | prediction CSV has all required BET_RECORD_SCHEMA fields; canonical JSONL exists | FAIL (missing fields) |
| `test_gate_08_artifact_metadata.py` | Gate 8 | model pkl files have git_sha, train_window_*, training_samples, feature_names | FAIL (missing metadata) |

## Required Integration Tests (Not Yet Implemented)

| Test | Description | Priority |
|------|-------------|----------|
| `test_walk_forward_windows` | Verify generate_windows() produces valid non-overlapping temporal windows | HIGH |
| `test_settlement_correctness` | Given known box scores, verify win/lose/push/void grading | HIGH |
| `test_clv_computation` | Given decision_line and closing_line, verify CLV calculation | HIGH |
| `test_bet_record_roundtrip` | Write and read a BetRecord, verify all fields preserved | MEDIUM |
| `test_realism_gate_validator` | Run validate_realism_gates() on sample data, verify pass/fail | MEDIUM |
