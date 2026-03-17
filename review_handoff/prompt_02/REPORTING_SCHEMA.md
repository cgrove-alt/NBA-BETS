# Reporting Schema — NBA-BETS

**Version:** 1.0.0

---

## Canonical Output Format: JSONL (per-bet log)

**File:** `reports/per_bet_log.jsonl`
**Format:** One JSON object per line, conforming to BET_RECORD_SCHEMA.md.
**Rationale:** JSONL is streamable, appendable, and parseable by pandas/polars.

---

## Summary Report

**File:** `reports/summary.json`

```json
{
  "realism_level": "MARKET-REALISTIC",
  "generated_at": "2026-03-17T14:00:00Z",
  "git_sha": "2b81b31b",
  "artifact_version": "20260310_080000",
  "test_window": "2025-01-01/2025-03-15",
  "total_bets_evaluated": 1200,
  "total_bets_accepted": 340,
  "total_bets_rejected": 860,
  "results": {
    "wins": 185,
    "losses": 140,
    "pushes": 10,
    "voids": 5,
    "pending": 0
  },
  "win_rate": 0.5692,
  "roi_pct": 4.2,
  "total_wagered": 3400.00,
  "total_pnl": 142.80,
  "sharpe_ratio": 1.45,
  "max_drawdown_pct": 8.3,
  "avg_clv": 0.35,
  "clv_positive_pct": 0.58,
  "brier_score": 0.245,
  "ece": 0.032,
  "gates_passed": ["gate_1", "gate_2", "gate_4", "gate_5", "gate_6", "gate_7", "gate_8"],
  "gates_failed": ["gate_3"],
  "warnings": ["CLV incomplete: closing lines available for only 0% of bets"]
}
```

---

## Monthly CLV Table

**File:** `reports/clv_by_month.csv`

| Column | Type | Description |
|--------|------|-------------|
| `month` | string | YYYY-MM |
| `bets` | integer | Number of accepted bets |
| `avg_clv` | number | Average CLV in line-points |
| `clv_positive_pct` | number | % of bets with positive CLV |
| `avg_pnl_per_bet` | number | Average PnL per bet |
| `roi_pct` | number | ROI for the month |

---

## Per-Market CLV Table

**File:** `reports/clv_by_market.csv`

| Column | Type | Description |
|--------|------|-------------|
| `market_type` | string | e.g., `player_points` |
| `bets` | integer | Number of accepted bets |
| `avg_clv` | number | Average CLV |
| `win_rate` | number | Win rate |
| `roi_pct` | number | ROI |
| `avg_edge` | number | Average vig-adjusted edge at decision time |

---

## Drawdown Table

**File:** `reports/drawdown.csv`

| Column | Type | Description |
|--------|------|-------------|
| `date` | string | YYYY-MM-DD |
| `cumulative_pnl` | number | Running PnL |
| `peak_pnl` | number | All-time high PnL to date |
| `drawdown` | number | `cumulative_pnl - peak_pnl` |
| `drawdown_pct` | number | `drawdown / peak_pnl` (or vs initial bankroll) |

---

## Calibration Summary

**File:** `reports/calibration_summary.json`

```json
{
  "brier_score": 0.245,
  "ece": 0.032,
  "mce": 0.085,
  "bins": [
    {"bin_center": 0.55, "predicted_avg": 0.55, "actual_avg": 0.53, "count": 120},
    {"bin_center": 0.65, "predicted_avg": 0.64, "actual_avg": 0.62, "count": 85},
    {"bin_center": 0.75, "predicted_avg": 0.74, "actual_avg": 0.71, "count": 40}
  ]
}
```

---

## Promotion Scorecard

**File:** `reports/promotion_scorecard.json`

```json
{
  "candidate_model": "20260310_080000",
  "incumbent_model": "20260224_080000",
  "test_window": "2025-01-01/2025-03-15",
  "metrics": {
    "holdout_roi_pct": 4.2,
    "holdout_clv_avg": 0.35,
    "max_drawdown_pct": 8.3,
    "ece": 0.032,
    "monthly_roi_std": 3.1,
    "min_monthly_roi": -2.5,
    "bet_count": 340,
    "clv_positive_months": 3,
    "clv_total_months": 3,
    "all_gates_passed": false,
    "gates_failed": ["gate_3"],
    "unresolved_audit_issues": 7
  },
  "promotion_decision": "HOLD — Gate 3 (closing lines) not passed",
  "realism_level": "RESEARCH-ONLY"
}
```

---

## Example CLI Commands (Future-Facing)

```bash
# Generate per-bet log from walk-forward backtest
# (script does not exist yet — to be implemented)
python -m nba_models.evaluation.report_generator \
    --bet-log review_handoff/prompt_02/reports/per_bet_log.jsonl \
    --output-dir review_handoff/prompt_02/reports/

# Validate realism gates on a bet log
python -m nba_models.evaluation.realism_gates \
    --bet-log review_handoff/prompt_02/reports/per_bet_log.jsonl \
    --artifact-metadata models/registry.json \
    --output review_handoff/prompt_02/reports/gate_results.json
```
