# Bet Record Schema — NBA-BETS

**Version:** 1.0.0

---

## Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event_id` | string | YES | Unique bet identifier (UUID or composite key) |
| `game_id` | integer | YES | BallDontLie game ID |
| `player_id` | integer | NO | BallDontLie player ID (null for team-level bets) |
| `player_name` | string | NO | Human-readable player name |
| `market_type` | string (enum) | YES | One of: `player_points`, `player_rebounds`, `player_assists`, `player_threes`, `player_pra`, `moneyline`, `spread`, `total` |
| `side` | string (enum) | YES | One of: `over`, `under`, `home`, `away` |
| `decision_timestamp` | string (ISO8601) | YES | When the prediction/decision was generated |
| `snapshot_timestamp` | string (ISO8601) | YES | When the line/odds were observed from sportsbook |
| `decision_line` | number | YES | The prop line at decision time |
| `decision_odds` | integer | YES | American odds at decision time |
| `book` | string | YES | Sportsbook name (e.g., `draftkings`, `fanduel`) |
| `closing_line` | number | NO | Prop line at game start (for CLV). NULL if not captured. |
| `closing_odds` | integer | NO | American odds at game start. NULL if not captured. |
| `model_fair_probability` | number [0,1] | YES | Model's estimated true probability of the chosen side |
| `market_implied_probability` | number [0,1] | YES | Raw implied probability from decision_odds |
| `vig_adjusted_probability` | number [0,1] | YES | No-vig (devigged) market probability |
| `raw_edge` | number | YES | `model_fair_probability - market_implied_probability` |
| `vig_adjusted_edge` | number | YES | `model_fair_probability - vig_adjusted_probability` |
| `calibrated_probability` | number [0,1] | NO | Post-calibration probability (if calibration applied) |
| `predicted_value` | number | NO | Model's predicted stat value (e.g., 24.5 points) |
| `uncertainty_score` | number | NO | Uncertainty metric (e.g., quantile spread) |
| `availability_flags` | string (enum) | NO | One of: `available`, `questionable`, `gtd`, `doubtful`, `out`, `unknown` |
| `accepted` | boolean | YES | Whether the bet passed all pipeline gates |
| `pass_reason` | string | NO | If rejected, the reason (e.g., "edge below threshold", "disabled prop") |
| `result` | string (enum) | YES | One of: `win`, `lose`, `push`, `void`, `pending` |
| `actual_value` | number | NO | Actual stat value from box score |
| `CLV` | number | NO | `closing_line - decision_line` (for over bets; sign-adjusted for under) |
| `PnL` | number | YES | Profit/loss in dollars. VOID and PUSH = 0. |
| `stake` | number | YES | Amount wagered in dollars |
| `artifact_version` | string | YES | Model artifact version that generated the prediction |
| `git_sha` | string | NO | Git commit hash of the code that generated the prediction |
| `realism_level` | string (enum) | YES | One of: `RESEARCH-ONLY`, `MARKET-REALISTIC`, `PAPER-TRADING`, `PRODUCTION-COMPARISON` |
| `train_window` | string | NO | Train window used (e.g., "2023-10-22/2024-10-21") |

---

## Example Instance

```json
{
  "event_id": "20260315-dal-bos-luka-pts-over",
  "game_id": 1045231,
  "player_id": 666786,
  "player_name": "Luka Doncic",
  "market_type": "player_points",
  "side": "over",
  "decision_timestamp": "2026-03-15T14:00:00Z",
  "snapshot_timestamp": "2026-03-15T13:45:00Z",
  "decision_line": 28.5,
  "decision_odds": -115,
  "book": "fanduel",
  "closing_line": 29.5,
  "closing_odds": -110,
  "model_fair_probability": 0.62,
  "market_implied_probability": 0.535,
  "vig_adjusted_probability": 0.512,
  "raw_edge": 0.085,
  "vig_adjusted_edge": 0.108,
  "calibrated_probability": 0.60,
  "predicted_value": 31.2,
  "uncertainty_score": 0.15,
  "availability_flags": "available",
  "accepted": true,
  "pass_reason": null,
  "result": "win",
  "actual_value": 33.0,
  "CLV": -1.0,
  "PnL": 8.70,
  "stake": 10.00,
  "artifact_version": "20260310_080000",
  "git_sha": "a1b2c3d4",
  "realism_level": "PAPER-TRADING",
  "train_window": "2024-10-22/2025-10-21"
}
```
