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

#### Training Results (Dec 26, 2025)

**Moneyline Model:**
- Accuracy: 62.86% (vs previous baseline)
- Brier Score: 0.2263
- Best individual: SVM (65.59%), LR (65.08%)
- Stacking: Did not improve (weighted avg better)
- Calibration: Did not improve (uncalibrated better)

**Spread Model:**
- RMSE: 14.51 points
- MAE: 11.40 points
- R²: 0.1963
- Quantile coverage (80%): 75%

**Player Props:**
- Points: RMSE=6.39, R²=0.463
- Rebounds: RMSE=2.52, R²=0.294 (position-aware)
- Assists: RMSE=1.92, R²=0.427 (position-aware)
- Threes: RMSE=1.32, R²=0.269
- PRA: Training completed

#### New Features Added (50+ total)
```
ELO: home_elo, away_elo, elo_diff, elo_win_prob, elo_spread
REST: home_days_rest, away_days_rest, rest_advantage, home_is_b2b, away_is_b2b, b2b_disadvantage
TRAVEL: home_travel_distance, away_travel_distance, travel_distance_diff
TIMEZONE: home_timezone_change, away_timezone_change, timezone_advantage
ALTITUDE: home_altitude_disadvantage, away_altitude_disadvantage
FATIGUE: home_travel_fatigue, away_travel_fatigue, fatigue_advantage, away_coast_to_coast
PACE: home_pace, away_pace, expected_pace, pace_diff
SITUATIONAL: road_b2b_vs_rested, away_tired_traveler
```

#### Next Steps (Phase 3)
- [ ] Integrate DARKO/EPM player impact metrics
- [ ] Add betting line movement features
- [ ] Implement schedule spot analysis
- [ ] Run comprehensive backtest with new models
