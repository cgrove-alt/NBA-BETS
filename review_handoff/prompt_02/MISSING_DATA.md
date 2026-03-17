# MISSING DATA — Closing Lines

**Status:** BLOCKER — prevents all production-like evaluation and CLV computation.

---

## Evidence

### Files Inspected

1. **`data/historical_lines/2024-10-22.json`**
   - Top-level keys: `date`, `games`, `api_credits_used`, `fetched_at`
   - `fetched_at`: `"2026-02-28T18:31:42.792368+00:00"` (fetched retroactively, not at game time)
   - Game-level keys: `bdl_game_id`, `odds_api_event_id`, `home_team`, `away_team`, `home_abbrev`, `away_abbrev`, `commence_time`, `snapshot_timestamp`, `player_props`
   - `snapshot_timestamp`: single value per game (e.g., `"2024-10-22T23:00:00Z"`)
   - **NO `closing_line`, `closing_odds`, `closing_timestamp` keys**

2. **`data/historical_lines/2024-12-25.json`**
   - Same structure. `snapshot_timestamp: "2024-12-25T16:00:00Z"` (~1hr before game)
   - `player_props` is a list of objects with keys: `player_name`, `prop_type`, `line`, `bookmaker`, `over_odds`, `under_odds`
   - **NO closing concept at any level**

3. **`data/historical_lines/2025-02-01.json`**
   - Same structure, same missing fields.

### Codebase Search

```
grep -r "closing_line\|closing_odds" nba_models/ nba_betting/ edge_calculator/ data/
```
Results: No file stores or computes closing lines for evaluation purposes. `nba_betting/odds/betting_market_features.py` has CLV-related feature generation code but it computes features for model input, not for evaluation output.

---

## What This Blocks

1. **CLV computation** — Cannot compute `closing_line - decision_line` without closing lines.
2. **MARKET-REALISTIC evaluation** — Gate 3 in REALISM_CHECKLIST.md requires closing lines for ≥90% of bets.
3. **PRODUCTION-LIKE label** — Cannot be assigned without CLV evidence.
4. **Model comparison** — CLV is the primary metric for comparing model quality. Without it, only noisy win-rate comparisons are possible.

---

## Required Fix

### Option A: Capture Closing Lines Prospectively (Recommended)
- Add a data ingestion job that captures lines at two times:
  1. Decision time (already done in historical_lines, ~1hr pre-game)
  2. At or just before game start (the "closing line")
- Store both snapshots per game/prop
- Requires: The Odds API subscription (already used) + cron job timed to game start

### Option B: Backfill from Odds API Historical Data
- The Odds API may offer historical odds retrieval for past events
- Would allow retroactive CLV computation for the 2024-25 season
- Requires: API access verification + backfill script

### Option C: Accept RESEARCH-ONLY Status
- All current evaluation remains labeled RESEARCH-ONLY
- No profitability claims
- Focus on model accuracy metrics only (RMSE, directional accuracy, calibration)

---

## Verdict Impact

Because closing lines are missing, the FINAL_REPORT verdict MUST be: **STOP AND FIX BLOCKERS**.
