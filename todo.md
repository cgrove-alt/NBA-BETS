# NBA Moneyline & Spread Model Improvement Plan

## Current State Analysis

### Model Architecture
- **Moneyline**: Weighted ensemble (XGBoost 18%, LightGBM 15%, GB 15%, RF 15%, MLP 12%, SVM 10%, LR 8%, CatBoost 12%)
- **Spread**: Ensemble regression with inverse-RMSE weighting, XGBoost meta-learner option
- **Training**: Time-decay weighting (180-day half-life), stratified train-test split

### Current Features (Team Models)
```
- season_win_pct_diff, recent_win_pct_diff
- pts_avg_diff, recent_pts_diff
- off_rating_diff, def_rating_diff, net_rating_diff
- location_win_pct_diff, home_advantage_factor
- home/away individual ratings and win percentages
- expected_point_diff, plus_minus_diff
```

### Key Gaps Identified
1. **No travel/rest features in team model training** - feature_engineering.py has `calculate_travel_fatigue`, `is_back_to_back`, `travel_fatigue_score`, but these are NOT being used in `process_games_for_training()`
2. **No Elo ratings** - simple but powerful (65-68% accuracy alone)
3. **No pace-adjusted predictions** - affects total points and spread variance
4. **No player impact metrics** - DARKO, EPM, RAPTOR not integrated
5. **No betting market features** - opening line, line movement, RLM
6. **Calibration not optimized** - research shows calibration > accuracy for ROI

---

## Improvement Recommendations (Priority Ordered)

### TIER 1: High Impact, Easy Implementation

- [ ] **1.1 Add Travel/Fatigue Features to Team Model**
  - Back-to-back indicator (1.5-2 point impact)
  - Rest days differential
  - Travel distance from last game
  - Time zone differential
  - Road B2B vs rested is worth ~2-4 points
  - **File**: `train_complete_balldontlie.py` line 2363 (team_features dict)

- [ ] **1.2 Implement Simple Elo Ratings**
  - Start at 1500, K-factor ~20
  - Home court advantage: +70 Elo points
  - FiveThirtyEight methodology proven at 65-68% accuracy
  - Update after each game, use rating difference as feature
  - **Effort**: ~100 lines of code

- [ ] **1.3 Optimize for Calibration, Not Just Accuracy**
  - Research shows 70% higher ROI with calibration-optimized models
  - Use Brier score as primary metric, not accuracy
  - Implement isotonic regression post-processing
  - **File**: `calibration.py` already has infrastructure

### TIER 2: High Impact, Medium Effort

- [ ] **2.1 Add Pace-Adjusted Spread Predictions**
  - Calculate expected possessions: `(Pace_home + Pace_away) / 2`
  - High-pace matchups = more variance in spread
  - Low-pace = grind-out games, tighter spreads
  - Adjust spread confidence based on pace match

- [ ] **2.2 Integrate Player Impact Metrics**
  - DARKO projections (free, daily updated): https://apanalytics.shinyapps.io/DARKO/
  - Calculate injury-adjusted team ratings
  - `Team Rating = Sum(Player_DARKO * Expected_Minutes) / Total_Minutes`
  - Critical for injury news edge

- [ ] **2.3 Implement Stacked Meta-Learner**
  - Current: weighted averaging
  - Better: MLP or XGBoost meta-learner on base model predictions
  - Use out-of-fold predictions to train meta-learner (prevents leakage)
  - Research shows 2-4% accuracy improvement

- [ ] **2.4 Add Quantile Regression for Spreads**
  - Predict 10th, 25th, 50th, 75th, 90th percentiles
  - Only bet when prediction interval is narrow (high confidence)
  - Handles high-variance matchups better
  - **File**: Already have `QuantilePropModel`, adapt for spreads

### TIER 3: Medium Impact, Higher Effort

- [ ] **3.1 Betting Market Features**
  - Opening line vs current line (line movement)
  - Movement direction + magnitude
  - Reverse line movement (sharp money indicator)
  - Public bet percentage (if available via odds_fetcher)

- [ ] **3.2 Recency Bias Exploitation**
  - Track blowout/close game flags
  - Market overreacts to recent blowouts
  - Fade teams coming off big wins against weak opponents
  - Back teams after close losses to quality opponents

- [ ] **3.3 Lineup-Based Adjustments**
  - Use actual expected lineups (from injury/rest data)
  - Calculate on-court rating vs off-court
  - Minutes-weighted player contributions
  - **Requires**: Real-time lineup data source

- [ ] **3.4 Advanced Four Factors Integration**
  - eFG% (Effective Field Goal %)
  - TOV% (Turnover Rate)
  - ORB% (Offensive Rebound %)
  - FT Rate (Free Throw Rate)
  - All should be point-in-time (before game)

### TIER 4: Edge Cases & Market Inefficiencies

- [ ] **4.1 Schedule Spot Analysis**
  - Lookahead spots (big game coming, trap game today)
  - Letdown spots (after emotional wins)
  - Sandwich games (between two tough opponents)
  - Long road trip fatigue (3+ games)

- [ ] **4.2 West Coast Morning Games**
  - Teams from PST playing early EST games struggle
  - Circadian rhythm disruption
  - ~1 point penalty for West teams at 12pm EST

- [ ] **4.3 Altitude Effects**
  - Denver (+5280 ft) creates conditioning advantage
  - Utah (+4200 ft) similar but less severe
  - Visiting teams suffer in 4th quarter at altitude

---

## Expected Performance Improvements

| Improvement | MAE Reduction | ATS Accuracy Gain |
|-------------|---------------|-------------------|
| Travel/B2B features | 0.3-0.5 pts | +1-2% |
| Elo ratings | 0.5-1.0 pts | +2-3% |
| Calibration optimization | - | +1-2% (ROI +20%) |
| Pace adjustment | 0.2-0.4 pts | +0.5-1% |
| Player impact metrics | 0.5-1.0 pts | +1-2% |
| Stacked meta-learner | 0.3-0.5 pts | +1-2% |

**Realistic ceiling**: 54-55% ATS (52.4% breaks even at -110)

---

## Implementation Order

### Phase 1: Quick Wins (This Week)
1. Add travel/fatigue features to team_features dict
2. Implement Elo ratings
3. Optimize for Brier score

### Phase 2: Core Improvements (Next 2 Sprints)
1. Pace-adjusted predictions
2. Stacked meta-learner architecture
3. Quantile regression for spreads

### Phase 3: Advanced Features (Ongoing)
1. DARKO/EPM integration
2. Line movement features
3. Schedule spot analysis

---

## Key Research Sources

1. [Stacked ensemble model for NBA (Scientific Reports, 2025)](https://www.nature.com/articles/s41598-025-13657-1)
2. [Calibration vs Accuracy for betting ROI (2024)](https://www.sciencedirect.com/science/article/pii/S266682702400015X) - **CRITICAL**: 70% higher ROI with calibration-focused
3. [FiveThirtyEight NBA methodology](https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/)
4. [DARKO Plus-Minus](https://www.nbastuffer.com/analytics101/darko-daily-plus-minus/)
5. [XGBoost + SHAP for NBA (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11265715/)

---

## Review Section

### Implementation Summary (December 26, 2025)

All Phase 1 and Phase 2 improvements have been implemented:

#### Completed Improvements

- [x] **1.1 Travel/Fatigue Features** - Added 20+ new features to team model:
  - `home_days_rest`, `away_days_rest`, `rest_advantage`
  - `home_is_b2b`, `away_is_b2b`, `b2b_disadvantage`
  - `home_travel_distance`, `away_travel_distance`, `travel_distance_diff`
  - `timezone_advantage`, `altitude_disadvantage`
  - `road_b2b_vs_rested`, `away_tired_traveler`

- [x] **1.2 Elo Ratings** - Implemented `EloRatingSystem` class:
  - K-factor: 20.0, Home advantage: +100 Elo points
  - Margin-of-victory multiplier for rating adjustments
  - Point-in-time ratings for training (no data leakage)
  - Features: `home_elo`, `away_elo`, `elo_diff`, `elo_win_prob`, `elo_spread`

- [x] **1.3 Calibration Optimization** - Added isotonic regression:
  - Compares calibrated vs uncalibrated Brier scores
  - Auto-selects best approach
  - Saves calibrator with model if improvement found

- [x] **2.1 Pace-Adjusted Predictions** - Added pace features:
  - `home_pace`, `away_pace`, `expected_pace`, `pace_diff`

- [x] **2.3 Stacked Meta-Learner** - Implemented logistic regression meta-learner:
  - Trains on base model predictions
  - Compares against weighted average
  - Auto-selects best approach

- [x] **2.4 Quantile Regression for Spreads** - Added `spread_quantile.pkl`:
  - 5 quantile models (10%, 25%, 50%, 75%, 90%)
  - 80% prediction interval coverage: 75% achieved
  - Average interval width: 32.2 points

#### Training Results (Dec 26, 2025) - Post Phase 3

**Moneyline Model:**
- Accuracy: 63.37% (+0.5% improvement)
- Brier Score: 0.2292
- Best individual: MLP (65.08%), SVM (63.37%)
- Stacking: Did not improve (weighted avg better)
- Calibration: Did not improve (uncalibrated better)

**Spread Model:**
- RMSE: 14.49 points (improved from 14.51)
- MAE: 11.36 points (improved from 11.40)
- R²: 0.1995
- Quantile coverage (80%): 75%

**Player Props:**
- Points: RMSE=6.39, R²=0.463
- Rebounds: RMSE=2.52, R²=0.294 (position-aware)
- Assists: RMSE=1.92, R²=0.427 (position-aware)
- Threes: RMSE=1.32, R²=0.269
- PRA: RMSE=8.13, R²=0.517

#### New Features Added (70+ total)
```
ELO: home_elo, away_elo, elo_diff, elo_win_prob, elo_spread
REST: home_days_rest, away_days_rest, rest_advantage, home_is_b2b, away_is_b2b, b2b_disadvantage
TRAVEL: home_travel_distance, away_travel_distance, travel_distance_diff
TIMEZONE: home_timezone_change, away_timezone_change, timezone_advantage
ALTITUDE: home_altitude_disadvantage, away_altitude_disadvantage
FATIGUE: home_travel_fatigue, away_travel_fatigue, fatigue_advantage, away_coast_to_coast
PACE: home_pace, away_pace, expected_pace, pace_diff
SITUATIONAL: road_b2b_vs_rested, away_tired_traveler
SCHEDULE_SPOTS: home_letdown_spot, away_letdown_spot, home_trap_game, away_trap_game
              home_sandwich_game, away_sandwich_game, home_road_trip_fatigue, away_road_trip_fatigue
              home_revenge_game, away_revenge_game, home_long_homestand, away_long_homestand
              home_early_season, away_early_season, schedule_spot_advantage
LINE_MOVEMENT: spread_movement, spread_movement_abs, spread_moved_toward_home/away
              total_movement, model_vs_market_spread, model_disagrees_spread, large_spread_move
```

#### Phase 3 Completed (Dec 26, 2025)
- [x] Integrate DARKO/EPM player impact metrics - Created `player_impact_fetcher.py`
- [x] Add betting line movement features - Added `calculate_line_movement_features()` function
- [x] Implement schedule spot analysis - Added `analyze_schedule_spots()` with letdown, trap, sandwich, revenge, road trip fatigue detection
- [x] Run comprehensive backtest with new models - Completed with improved results

#### Summary of Phase 3 Improvements

1. **Player Impact Fetcher** (`player_impact_fetcher.py`):
   - `PlayerImpactFetcher` class for fetching EPM/DARKO metrics
   - `STAR_PLAYER_IMPACTS` dictionary with 40+ star players
   - `calculate_injury_adjustment()` for injury-based spread adjustments

2. **Schedule Spot Analysis** (`analyze_schedule_spots()`):
   - Letdown spots (after big wins vs elite teams)
   - Trap games (weak opponent before tough game)
   - Sandwich games (between two elite opponents)
   - Road trip fatigue (3rd+ road game)
   - Revenge games (recent close loss to opponent)
   - Long homestand complacency (4th+ home game)
   - Early season variance flag
   - Combined schedule_spot_score

3. **Line Movement Features** (`calculate_line_movement_features()`):
   - Spread movement tracking (opening vs current)
   - Total movement tracking
   - Model vs market disagreement detection
   - Steam move indicators (large moves)
   - Ready for live prediction integration

#### Tier 1 Architectural Upgrades (Dec 26, 2025)

- [x] **1. Quantile Regression for Player Props** (`model_trainer.py`)
  - Added `QuantilePropModel` class using GradientBoostingRegressor with quantile loss
  - Trains 3 models: 0.45, 0.50 (median), 0.55 quantiles
  - Generates implied Over/Under probabilities from quantile positions
  - More accurate for betting than simple mean prediction

- [x] **2. Dynamic Imputation with Rolling Averages** (`train_complete_balldontlie.py`)
  - Added `DynamicLeagueAverages` class
  - Calculates rolling 7-day league averages for each stat
  - Adapts to season-long scoring trends automatically
  - Replaces static defaults (114.0 for ratings) with dynamic values

- [x] **3. Neural Network in Ensemble** (`model_trainer.py`)
  - Added MLPClassifier to `EnsembleMoneylineModel`
  - Config: hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.0001
  - Early stopping enabled with validation fraction 0.1
  - Captures non-linear patterns that tree-based models miss

- [x] **4. Automated Feature Selection with RFECV** (`feature_engineering.py`)
  - Added `FeatureSelector` class using sklearn's RFECV
  - Uses XGBoost as base estimator for feature importance
  - Automatically identifies and drops 0-importance features
  - Saves selected features to JSON for consistent training/inference

#### Next Steps (Future Enhancements)
- [ ] Integrate live line movement data from odds_fetcher during predictions
- [ ] Add public betting percentage data (if available)
- [ ] Implement trap game detection with actual future schedule data
- [ ] Add player-level injury impact adjustments to props models
- [ ] Track Closing Line Value (CLV) for bet quality validation
