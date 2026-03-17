# Final Report — Prompt 02

## Executive Summary

This prompt defines the canonical evaluation framework for NBA-BETS: 4 evaluation modes, 8 realism gates, per-bet record and artifact schemas, data split policy, and reporting standards. Every gate has a corresponding regression test. Gap analysis against the current repo reveals 6 blockers, the most critical being: **no closing line data exists anywhere in the repo**, which makes CLV computation impossible and blocks all production-like evaluation. The existing profitability backtest uses simulated lines, fixed -110 odds, and in-sample models — it must be relabeled RESEARCH-ONLY. A `real_lines_backtest.py` exists that uses historical sportsbook lines but still lacks walk-forward retraining and closing-line CLV.

---

## Verdict

**STOP AND FIX BLOCKERS**

---

## Prioritized Next Steps

| # | Action | Why | Files | Effort |
|---|--------|-----|-------|--------|
| 1 | **Capture closing lines** — Add data ingestion job that snapshots lines at game start | Blocks ALL CLV computation and production-like evaluation (Gate 3) | NEW: `scripts/capture_closing_lines.py`, `data/closing_lines/` | LARGE |
| 2 | **Backfill closing lines** — Query Odds API for historical closing data (2024-25 season) | Enables retroactive CLV on existing predictions | NEW: `scripts/backfill_closing_lines.py` | MEDIUM |
| 3 | **Label profitability_backtest.py as RESEARCH-ONLY** — Add realism_level header to all outputs | Gates 1, 4, 5 violations | `nba_models/backtesting/profitability_backtest.py` | SMALL |
| 4 | **Implement walk-forward retraining** — Per DATA_SPLIT_POLICY.md | Gate 4: no test-period training | `nba_models/backtesting/profitability_backtest.py`, NEW: `nba_models/evaluation/split_policy.py` | LARGE |
| 5 | **Add DNP/void to settlement** — Check minutes=0 → result=void | Gate 6 | `nba_betting/settle_trades.py:L15-91`, `nba_betting/paper_trading.py:L261` | SMALL |
| 6 | **Add artifact metadata** — git_sha, train_window, training_samples to model pkl | Gate 8 | `nba_models/training/train_complete_balldontlie.py:L5763-5781` | SMALL |
| 7 | **Implement BetRecord dataclass** — Conform to BET_RECORD_SCHEMA.md | Gate 7 | NEW: `nba_models/evaluation/bet_record.py` | MEDIUM |
| 8 | **Implement CLV computation** — `compute_clv(decision_line, closing_line, side)` | Required for all non-research evaluation | NEW: `nba_models/evaluation/clv.py` | SMALL |
| 9 | **Implement realism gate validator** — `validate_realism_gates()` that checks all 8 gates on a bet log | Automated enforcement | NEW: `nba_models/evaluation/realism_gates.py` | MEDIUM |
| 10 | **Add decision-time capture** — Store snapshot_timestamp, book, and odds per bet in predictions | Gate 2 completeness | `nba_models/inference/daily_predictions.py:L3318-3376` | MEDIUM |
| 11 | **Make real_lines_backtest the default** — Promote over profitability_backtest | Gates 1, 5 | `.github/workflows/` and documentation | SMALL |
| 12 | **Expand CI coverage** — Add `nba_models/`, `nba_betting/` to pytest coverage config | Visibility | `pytest.ini:L30` | SMALL |

---

## Data Availability Check

| Data Type | Status | Evidence |
|-----------|--------|----------|
| Historical decision-time lines | **AVAILABLE** | `data/historical_lines/` — 166 files, 2024-10-22 to 2025-04-xx, with `snapshot_timestamp`, `bookmaker`, `over_odds`, `under_odds` per prop |
| Historical closing lines | **MISSING — BLOCKER** | No `closing_line`, `closing_odds`, or second snapshot exists in any file. See MISSING_DATA.md. |
| Real player outcomes | **AVAILABLE** | BallDontLie API via `settle_trades.py:L38` |
| Model artifacts | **AVAILABLE** | `models/*.pkl` — 39 files. Missing metadata fields. |
