# Market Realism Audit — NBA-BETS

## CRITICAL Issues

### MR-01: Simulated Prop Lines (CRITICAL)

**Location:** `nba_models/backtesting/profitability_backtest.py`, `simulate_prop_line()` (lines 134-164)
**Behavior:** Prop lines are generated as `0.70 * season_avg + 0.30 * recent_avg`, rounded to 0.5.
**Why unrealistic:** Real sportsbook lines incorporate: sharp money flow, injury news, rest/schedule context, correlated market information, public betting percentages, and algorithmic line-making. The simulated lines are trivially predictable from model features. A model that beats its own input-derived lines proves nothing about beating a sportsbook.
**Acceptable for:** Nothing. This invalidates profitability claims entirely.
**Permanent fix:** Use historical sportsbook lines from `data/historical_lines/` (170+ files exist). For the 2023-24 backtest period, acquire historical lines from The Odds API or another provider.

### MR-02: Fixed -110 Odds Assumption (CRITICAL)

**Location:** `nba_models/backtesting/profitability_backtest.py` (line 60), `nba_betting/bet_filter.py` (line 145), `nba_betting/prediction_pipeline.py` (line 369)
**Behavior:** All backtests and the fallback bet sizing assume standard -110 odds on both sides.
**Why unrealistic:** Real player prop odds range from -140 to +120. The vig on player props is typically 5-10%, not the 4.5% of -110/-110. Many props are -115/-115 or worse. Enhanced odds promotions create -110 but are not the norm.
**Acceptable for:** Research approximation only.
**Permanent fix:** Use actual odds from historical data or API. The devigging code already supports real odds (in production). Backtests must use real odds to compute real EV.

### MR-03: No Timestamped Decision-Time Line Capture (HIGH)

**Location:** System-wide
**Behavior:** Production predictions are generated at a single point in time (daily at 10 AM ET via cron). There is no record of what the lines were at the exact moment the prediction was made vs what they were at game time.
**Why unrealistic:** Lines move throughout the day. A prediction generated at 10 AM against a 10 AM line may no longer have edge by game time (7 PM). Without decision-time line capture, you cannot compute true CLV (Closing Line Value), which is the gold standard for measuring betting edge.
**Permanent fix:**
1. Store `prediction_timestamp` and `line_at_prediction_time` alongside each prediction.
2. Store `closing_line` after game starts.
3. Compute CLV = `closing_line - prediction_line` for each prediction.
4. Track CLV as primary model quality metric.

### MR-04: No Opening / Mid / Close Line Tracking in Backtest (HIGH)

**Location:** `nba_models/backtesting/profitability_backtest.py`
**Behavior:** Backtests use a single simulated line per prediction. No opening line, no closing line, no line movement data.
**Why unrealistic:** Real profitable betting requires getting the best available line. A model might show +EV at opening lines but -EV at closing lines (or vice versa). Without this distinction, backtest results are meaningless for practical betting.
**Permanent fix:** Incorporate `data/historical_lines/` with timestamps into backtests.

## HIGH Issues

### MR-05: CLV Not Tracked (HIGH)

**Location:** System-wide
**Behavior:** CLV (Closing Line Value) infrastructure exists in code (`nba_betting/odds/betting_market_features.py`) as a feature, but CLV is NOT computed for production predictions as a model quality metric.
**Why unrealistic:** CLV is the single most important metric for evaluating a betting model's true edge. Win rate and ROI are noisy; CLV is the signal. Without CLV tracking, you cannot distinguish skill from variance.
**Permanent fix:** After each game, compare prediction line to closing line. Track rolling CLV by prop type.

### MR-06: Stale-Line Blindness (HIGH)

**Location:** `nba_models/inference/daily_predictions.py`
**Behavior:** Lines are fetched once during daily prediction run. If lines move significantly between prediction time and bet execution, the model may be betting into stale edges that no longer exist.
**Why unrealistic:** Sharp money moves lines within minutes. A 3-point edge at 10 AM may be a 0-point edge by noon.
**Permanent fix:**
1. Record timestamp of line fetch.
2. Before execution, re-check current lines.
3. Add staleness check: reject bets where line has moved against the prediction by >50% of the edge.

### MR-07: OT Settlement Ambiguity (HIGH)

**Location:** Training data, settlement logic
**Behavior:** Training uses raw box-score stats (including OT). Settlement in `settle_trades.py` also uses raw stats. Most sportsbooks settle player props based on REGULATION stats only (some books vary).
**Why unrealistic:** A player who scores 18 points in regulation and 5 in OT (23 total) would be settled as UNDER 20.5 by most books but OVER by this system.
**Affected games:** ~2-3% of NBA games go to OT.
**Permanent fix:**
1. Document which sportsbook's rules are used.
2. Add regulation-only stats to training data where possible.
3. Add `includes_ot` flag to each game in training data.

### MR-08: No Book-Level Data in Backtest (HIGH)

**Location:** `nba_models/backtesting/profitability_backtest.py`
**Behavior:** Backtests assume a single line and single odds. In reality, different sportsbooks offer different lines and odds.
**Why unrealistic:** Best-line shopping across 8+ books can improve EV by 1-3%. Assuming a single book understates the opportunity (or overstates it if the model's edge is only at the best available book).
**Permanent fix:** When using real lines, include multi-book data and compute results for both "average book" and "best available line."

## MEDIUM Issues

### MR-09: No Bet Execution Simulation (MEDIUM)

**Location:** Backtests assume instant execution at posted odds
**Behavior:** In real betting, lines move. There is slippage between the line you target and the line you get. Large bets move markets.
**Why unrealistic:** Particularly relevant for low-liquidity player prop markets.
**Permanent fix:** Add 0.5-1 point slippage to backtest lines, or apply a vig penalty.

### MR-10: Kelly Sizing Uses Confidence, Not True Probability (MEDIUM)

**Location:** `nba_betting/prediction_pipeline.py` (lines 371-386)
**Behavior:** Kelly criterion uses the model's calibrated confidence as the win probability. If calibration is wrong (L-06, L-07), Kelly sizing will be wrong.
**Why unrealistic:** Kelly is highly sensitive to probability accuracy. A 5% error in probability can lead to 2-3x over-betting.
**Permanent fix:** Use quarter-Kelly (already done) AND verify calibration on out-of-sample data before trusting Kelly sizing.

### MR-11: Hidden Exclusion of Hard Cases (MEDIUM)

**Location:** `nba_betting/constants.py` (line 61), `nba_betting/bet_filter.py`
**Behavior:** Threes, spread, and assists are completely DISABLED (`DISABLED_PROPS`). This means the model never bets on its worst-performing props.
**Why misleading:** Backtest results only reflect the model's BEST prop types. The overall model may not have a genuine edge — it just avoids its worst predictions. If a bettor were using this system, they'd see "7% ROI" but only on cherry-picked prop types.
**Permanent fix:** Report full-portfolio results (all prop types) alongside filtered results. Make it clear which results reflect selective betting.

### MR-12: Missing Scratch Handling (MEDIUM)

**Location:** Backtests and production
**Behavior:** If a player is scratched (late scratch after prediction), there's no automatic void/cancellation logic. The prediction remains in the system.
**Why unrealistic:** Real sportsbooks void bets when players don't play. The model should track void rates and account for them.
**Permanent fix:** Check actual minutes played in settlement. If minutes = 0, mark as VOID (no P&L), not as loss.

## LOW Issues

### MR-13: No Public Betting Data (LOW)

**Location:** System-wide
**Behavior:** No integration with public betting percentage data (e.g., Action Network, Pregame.com).
**Why relevant:** Sharp/public split is valuable for detecting "value" (when sharps disagree with public).
**Permanent fix:** Low priority. Could add as feature for future model improvement.

### MR-14: No Alternate Lines (LOW)

**Location:** System-wide
**Behavior:** Model only considers the primary line. Sportsbooks offer alternate lines (e.g., points over 22.5 at +150 vs primary over 20.5 at -110).
**Why relevant:** Sometimes better EV exists at alternate lines.
**Permanent fix:** Low priority. Would require API support for alternate markets.
