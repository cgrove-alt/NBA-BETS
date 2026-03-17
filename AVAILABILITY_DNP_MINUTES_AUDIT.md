# Availability / DNP / Minutes Audit — NBA-BETS

## System Components

### 1. Injury Tracking

| Component | File | Status |
|-----------|------|--------|
| `InjuryStatus` enum | `nba_data/sources/injury_fetcher.py` | **Fully Wired** — OUT(0.0), DOUBTFUL(0.25), QUESTIONABLE(0.50), GTD(0.50), PROBABLE(0.75), AVAILABLE(1.0) |
| `InjuryReport` dataclass | `nba_data/sources/injury_fetcher.py` | **Fully Wired** — tracks player, status, type, impact stats |
| `InjuryFetcher` class | `nba_data/sources/injury_fetcher.py` | **Fully Wired** — Balldontlie API primary |
| `InjuryTrackerV3` class | `nba_data/sources/injury_tracker_v3.py` | **Fully Wired** — RotoWire primary, NBA.com fallback, PostgreSQL storage |
| `LineupIntelService` | `lineup_intel/lineup_intel_service.py` | **Partially Wired** — infrastructure exists but unclear if actively called |
| Injury scraper | `lineup_intel/injury_scraper.py` | **Partially Wired** — web scraping, may break with site changes |

### 2. DNP Handling

| Context | Logic | Status |
|---------|-------|--------|
| **Production inference** | Hard filter: skip OUT/DOUBTFUL before any prediction (daily_predictions.py line 3018) | **Fully Wired** |
| **Profitability backtest** | `actual_min < 15` -> skip (line 278) | **Partially Wired** — uses post-hoc minutes |
| **Comprehensive backtest** | `minutes_played < 0.1` -> skip (line 1590) | **Partially Wired** — same issue |
| **Training data** | `mins < 5` -> skip in minutes oracle; no explicit DNP filter in main training | **Partially Wired** |
| **Settlement** | No DNP check — if player did not play, prediction is not voided | **NOT Wired** |

### 3. Questionable / Doubtful / Out Handling

| Status | Production | Backtest | Training |
|--------|------------|----------|----------|
| OUT | Skip entirely | Post-hoc skip via minutes | No special handling |
| DOUBTFUL | Skip entirely | Post-hoc skip via minutes | No special handling |
| QUESTIONABLE | Flag HIGH_UNCERTAINTY | Not modeled | No special handling |
| GTD | Flag HIGH_UNCERTAINTY | Not modeled | No special handling |
| PROBABLE | No flag | Not modeled | No special handling |

### 4. Lineup Uncertainty

**What exists:** `LineupIntelService` with starter detection, `PlayerIntel` dataclass with starter confidence.
**What is wired:** Starter detection via `avg_mins >= 28` in minutes features.
**What is dead:** `lineup_tracker.py` exists but no evidence of active use.
**What is misleading:** Feature `is_starter` derived from minutes average, not actual lineup data.

### 5. Minutes Projections

| Component | File | Status |
|-----------|------|--------|
| `MinutesFeatureGenerator` | `minutes_oracle/minutes_features.py` | **Fully Wired** — 38 features |
| `MinutesPredictor` | `minutes_oracle/minutes_predictor.py` | **Fully Wired** — quantile regression (p10-p90) |
| `MinutesAwarePropPredictor` | `minutes_oracle/integration.py` | **Partially Wired** — exists but daily_predictions.py uses its own logic |
| Minutes adjustment | `daily_predictions.py` (lines 2298-2332) | **Fully Wired** — caps adjustment to +/-15% |

### 6. Issues

#### A-01: Dual Minutes Adjustment Paths (MEDIUM)

`minutes_oracle/integration.py` has `MinutesAwarePropPredictor`. SEPARATELY, `daily_predictions.py` has its own minutes adjustment logic (lines 2298-2332). Unclear which path runs.
**Risk:** Double-adjustment or inconsistency.
**Fix:** Consolidate to single path.

#### A-02: Inconsistent Low-Minute Thresholds (HIGH)

Three different thresholds across code paths:
- Training (minutes oracle): < 5 min excluded
- Profitability backtest: < 15 min excluded
- Comprehensive backtest: < 0.1 min excluded (DNP only)
- Production: No explicit threshold

**Risk:** Profitability backtest's 15-minute threshold silently excludes ~15-20% of player-games (bench players 10-14 min). This makes backtest look better because bench players are harder to predict. Production will generate predictions for these players that the backtest never evaluated.
**Fix:** Align thresholds. Use consistent cutoff (recommend 10 min) or let model learn the variance.

#### A-03: No Scratch/Void Handling (HIGH)

**What exists:** Nothing.
**Risk:** Bets placed on late-scratched players void at sportsbook but are not tracked as voids in the system. Paper trades for scratched players are silently lost.
**Fix:** Check actual minutes in settlement. If minutes = 0, mark as VOID, not loss.

#### A-04: No Return-From-Injury Modeling (MEDIUM)

**What exists:** Nothing explicit.
**Risk:** Players returning from injury often play restricted minutes. Model has no feature for "games since return from injury" or "is on minutes restriction."
**Impact:** Predictions for returning players overestimate based on pre-injury stats.
**Fix:** Add `games_since_injury_return` feature and `is_minutes_restricted` flag.

#### A-05: Role Instability Not Detected (MEDIUM)

**What exists:** `rotation_spot` feature computed from SEASON averages.
**What is missing:** No detection of role changes mid-season (trade, lineup change). Feature uses stale data after roster moves.
**Fix:** Compute `rotation_spot` from recent games (last 5-10) rather than full season.

### 7. Backtest vs Production Availability Comparison

| Filter | Profitability BT | Comprehensive BT | Production |
|--------|-----------------|-------------------|------------|
| Minutes threshold | 15 min | 0.1 min | None |
| Injury status | No | No | Yes (OUT/DOUBTFUL skip) |
| Uncertainty flag | No | No | Yes (GTD/QUESTIONABLE) |
| Minutes oracle | No | Optional | Yes |
| Injury boost | No | No | Yes |
| Sample size | 10+ games | No | 10+ games |

### 8. Summary: What Works vs What is Broken

| Category | Fully Wired | Partially Wired | Dead / Missing |
|----------|-------------|-----------------|----------------|
| OUT/DOUBTFUL filter | Production | Backtest (via minutes) | Training |
| QUESTIONABLE/GTD flag | Production | | Backtest, Training |
| Minutes prediction | Production | | Backtest |
| Minutes adjustment | Production (2 paths) | | Backtest |
| Injury boost | Production | | Backtest |
| Scratch handling | | | All paths |
| Return-from-injury | | | All paths |
| Role change detection | | | All paths |
| Late scratch voiding | | | Settlement |
| Consistent minute thresholds | | | All paths |
