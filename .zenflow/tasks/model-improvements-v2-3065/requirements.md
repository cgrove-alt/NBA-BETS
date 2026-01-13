# Product Requirements Document (PRD)
# NBA Prediction Model v2 - Path to SOTA Performance

**Document Version**: 1.0
**Date**: January 13, 2026
**Author**: AI Senior ML Engineer & NBA Analytics Expert
**Objective**: Transform the current NBA prediction model into the most accurate NBA betting model ever created

---

## Executive Summary

This PRD outlines the requirements for upgrading the existing NBA prediction model from its current state (R² = 0.68, player props RMSE = 5.4) to state-of-the-art (SOTA) performance. The model currently employs an 8-model ensemble with stacking architecture, comprehensive feature engineering, and proper temporal validation. However, systematic analysis has identified critical gaps in feature engineering, injury detection, and meta-learning sophistication that prevent optimal accuracy.

**Key Performance Gaps Identified**:
- **Threes Model**: R² = -0.568 (performs worse than baseline)
- **PRA Model**: RMSE = 8.469 (poor composite prediction)
- **Points Model**: RMSE = 6.757 with -1.518 bias (systematic under-prediction)
- **Injury Detection**: 161 DNP (Did Not Play) players missed in Jan 7, 2026 predictions
- **Feature Coverage**: Missing Dean Oliver's Four Factors (the "holy grail" of basketball analytics)

**Success Criteria**: Achieve industry-leading accuracy benchmarks:
- Player Props RMSE: < 4.5 (from 5.4)
- Points R²: > 0.55 (from 0.38)
- Threes R²: > 0.20 (from -0.57)
- PRA RMSE: < 7.0 (from 8.47)
- Injury Detection Accuracy: > 95% (from ~70%)
- Betting ROI: > 5% (calibrated probabilities)

---

## 1. Current State Analysis

### 1.1 Model Architecture (Current)

**Moneyline (Win Probability)**:
- **Ensemble Approach**: Weighted averaging of 8 models
  - XGBoost (18%), LightGBM (15%), GradientBoosting (15%), RandomForest (15%), MLP (12%), CatBoost (12%), SVM (10%), Logistic Regression (8%)
- **Meta-Learner**: Simple weighted averaging with inverse-RMSE weights
- **Calibration**: Platt Scaling, Isotonic Regression, Temperature Scaling, Beta Calibration

**Spread Predictions**:
- Ensemble regression (SVR, RandomForest, GradientBoosting)
- Meta-learner: Ridge or XGBoost
- Normal distribution CDF for spread-edge-to-probability conversion

**Player Props** (Points, Rebounds, Assists, Threes, PRA):
- **Stacking Model v2**:
  - Level 0: XGBoost, LightGBM, Ridge, Lasso, GradientBoosting, RandomForest, optional CatBoost
  - Level 1: ElasticNet meta-learner on out-of-fold predictions
- Alternative models: Enhanced, Quantile (10th/50th/90th percentiles)

### 1.2 Feature Engineering (Current)

**Team-Level Features**:
✅ Implemented:
- Win percentage differentials (season, recent)
- Points differential, offensive/defensive/net ratings
- Home/away splits, home advantage factors
- Expected point differential, plus-minus
- ELO ratings (team strength)

❌ Missing Critical Features:
- **Dean Oliver's Four Factors** (eFG%, TOV%, ORB%, FT/FGA) - 80% of game outcomes
- **Travel/Rest Features**: Days rest, back-to-back games, travel distance, altitude adjustment
- **Pace-Adjusted Metrics**: Possessions per game impacts spread variance
- **Betting Market Features**: Opening lines, line movement, Reverse Line Movement (RLM), consensus odds
- **Player Impact Metrics**: DARKO, EPM, RAPTOR (star player value)
- **Momentum Interactions**: "3rd game in 4 nights on road" scenarios

**Player-Level Features**:
✅ Implemented:
- Season/recent averages (points, rebounds, assists, threes)
- Usage rates, minutes played
- Position-specific factors
- Opponent defensive context
- Head-to-head history
- Recent form (5-game, 10-game)
- Injury impact adjustment

❌ Missing Critical Features:
- **Real-time injury feeds** (currently delayed, causing 161 DNP misses)
- **Usage redistribution modeling** (when star players are out)
- **Matchup-specific metrics** (player X vs zone defense)
- **Fatigue modeling** (back-to-back, minutes load)
- **Shot quality metrics** (expected FG% from tracking data)

### 1.3 Data Sources

**Primary**: Balldontlie API (600 req/min, preferred for speed)
**Fallback**: NBA API (nba_api Python package)
**Live Odds**: Multi-sportsbook odds fetcher
**Tracking Data**: Play-by-play, shot charts via NBA CDN

**Issue**: No real-time injury API integration (manual scraping, delayed updates)

### 1.4 Performance Metrics (Backtest Oct 21 - Dec 12, 2025)

| Prop Type | Count | RMSE | MAE | R² | Bias |
|-----------|-------|------|-----|-----|------|
| Points | 7,078 | 6.757 | 5.072 | 0.381 | -1.518 |
| Rebounds | 7,097 | 2.543 | 1.944 | 0.364 | 0.161 |
| Assists | 6,007 | 2.035 | 1.441 | 0.324 | -0.464 |
| Threes | 4,578 | 1.700 | 1.268 | **-0.568** | -0.920 |
| PRA | 7,685 | 8.469 | 6.668 | 0.513 | -0.375 |
| **Overall** | **32,445** | **5.435** | **3.557** | **0.681** | **-0.601** |

**Critical Failures (Jan 7, 2026)**:
- 46 data errors (avg error: 15.2 points)
- 176 feature errors (avg error: 7.0 points)
- 613 variance errors (avg error: 1.9 points)
- 34 model errors (avg error: 13.9 points)
- **161 injury errors** (DNP players not detected)
- Notable misses: SGA (30.5pt error), Luka Doncic (27.6pt error)

---

## 2. Requirements for SOTA Model

### 2.1 Feature Engineering Requirements

#### FR-1: Implement Dean Oliver's Four Factors
**Priority**: P0 (Critical)
**Impact**: 2-4 percentage point accuracy improvement
**Rationale**: Research shows Four Factors explain 80% of game outcomes

**Acceptance Criteria**:
- Calculate for each team:
  - **eFG%** (Effective Field Goal %) = (FG + 0.5 × 3PM) / FGA
  - **TOV%** (Turnover Rate) = TOV / (FGA + 0.44 × FTA + TOV)
  - **ORB%** (Offensive Rebound %) = ORB / (ORB + Opp DRB)
  - **FT/FGA** (Free Throw Rate) = FT / FGA
- Generate rolling averages (5-game, 10-game, season)
- Calculate differentials (Team A - Team B) for each factor
- Add interaction terms (e.g., eFG% × TOV% differential)
- Temporal discipline: Use only data before game_date
- Integration: Add to `advanced_stats_v2.py` module
- Validation: Backtest shows ≥1% RMSE reduction

**Technical Notes**:
- Data source: Balldontlie `/stats` endpoint provides necessary box score stats
- Computation: Can be calculated from existing FG, FGA, 3PM, FT, FTA, ORB, DRB, TOV fields
- Storage: Add 12 new columns to feature matrix (4 factors × 3 windows)

---

#### FR-2: Add Travel and Fatigue Features
**Priority**: P0 (Critical)
**Impact**: ~2 points per game on back-to-backs, altitude adjustments
**Rationale**: Back-to-back games correlate with -2.1 point differential; Denver altitude worth +1.5 pts

**Acceptance Criteria**:
- **Days Rest**: Calculate days since last game (0 = back-to-back)
- **Back-to-Back Detection**: Binary flag for consecutive games
- **Travel Distance**: Haversine formula using arena coordinates (already in `NBA_ARENA_DATA`)
- **Altitude Adjustment**: Denver (5280ft) and Utah (4200ft) require fatigue multipliers
- **Timezone Crossing**: Count zones crossed (affects circadian rhythm)
- **Schedule Density**: "3rd game in 4 nights" scenario detection
- **Road Trip Length**: Number of consecutive away games
- Differentials: Calculate for both teams (home rest advantage)

**Technical Notes**:
- Arena data exists in `feature_engineering.py:86-120` with coordinates, altitude, timezone
- Integration point: `calculate_rest_and_fatigue()` function at line 2800+
- Validation: Denver home games should show +1.5 spread adjustment

---

#### FR-3: Integrate Player Impact Metrics (DARKO/EPM/RAPTOR)
**Priority**: P1 (High)
**Impact**: 5-8% accuracy improvement for player prop models
**Rationale**: Advanced metrics capture player value beyond box score stats

**Acceptance Criteria**:
- Fetch DARKO DPM (Daily Plus-Minus) from external API or scraping
- Fetch ESPN's EPM (Estimated Plus-Minus) or use FiveThirtyEight's RAPTOR
- Add to player feature set:
  - Player's impact metric (standardized -10 to +10 scale)
  - Team impact when player is on/off court
  - Opponent defensive impact against player's position
- Cache metrics daily (updated once per 24 hours)
- Fallback: If unavailable, use BPM (Box Plus-Minus) from Basketball Reference

**Technical Notes**:
- New module: `player_impact_fetcher.py`
- Integration: Add to `generate_*_prop_features()` functions
- Data freshness: Update nightly, not per-game (reduces API calls)

---

#### FR-4: Real-Time Injury Detection System
**Priority**: P0 (Critical)
**Impact**: Eliminates 161+ DNP prediction errors
**Rationale**: Jan 7, 2026 failures were predominantly injury-related

**Acceptance Criteria**:
- **Primary Source**: Integrate RotoWire injury API (paid, real-time) or FantasyLabs
- **Fallback Source**: Scrape NBA.com/injuries or ESPN injury reports
- **Update Frequency**: Every 15 minutes during game days (2 hours pre-tipoff)
- **Data Points**:
  - Injury status: OUT, DOUBTFUL, QUESTIONABLE, GTD (Game-Time Decision)
  - Injury type: Knee, ankle, back, illness, rest
  - Last update timestamp
- **Usage Redistribution**:
  - When top-3 scorer is OUT, distribute 70% of usage to teammates by role
  - Explicit binary flag: `star_player_out` for models
- **Alert System**: Flag predictions as "HIGH UNCERTAINTY" if player status is GTD
- **Validation**: Zero DNP players in predictions (100% detection rate)

**Technical Notes**:
- New module: `injury_tracker_v3.py` (upgrade from `injury_impact_v2.py`)
- Integration: Pre-prediction check in `daily_predictions.py:500+`
- Database: Cache injury status in SQLite for historical analysis
- Edge case: Handle late scratches (< 30 min before tipoff) with prediction withdrawal

---

#### FR-5: Betting Market Features
**Priority**: P1 (High)
**Impact**: 3-5% ROI improvement via Closing Line Value (CLV)
**Rationale**: Market consensus contains wisdom of crowds; line movement signals sharp action

**Acceptance Criteria**:
- **Opening Line**: First odds posted (typically -110/-110 for spread)
- **Closing Line**: Final odds before tipoff
- **Line Movement**: Closing - Opening (e.g., spread moved from -5 to -7)
- **Reverse Line Movement (RLM)**: Line moves against majority of bets (sharp money indicator)
- **Consensus Odds**: Average across 10+ sportsbooks
- **Steam Moves**: Rapid line movement (>1.5 points in <5 minutes)
- **Bet Percentage**: Public % on each side (contrarian indicator)
- Data freshness: Update every 5 minutes during game day

**Technical Notes**:
- Extend `odds_fetcher.py` to track line history (not just current odds)
- Store in time-series DB (InfluxDB or PostgreSQL with TimescaleDB)
- Integration: Add to `generate_game_features()` as 5 new columns
- Validation: RLM detection should flag games with >60% bets on one side, line moves opposite

---

#### FR-6: Pace-Adjusted Metrics
**Priority**: P2 (Medium)
**Impact**: Reduces spread variance by 10-15%
**Rationale**: High-pace games (100+ possessions) have higher variance; affects confidence bounds

**Acceptance Criteria**:
- Calculate team pace: Possessions per 48 minutes
  - Formula: 48 × ((Possessions) / (Minutes))
  - Possessions = 0.5 × ((FGA + 0.4 × FTA - 1.07 × (ORB / (ORB + Opp DRB)) × (FGA - FG) + TOV) + (Opp FGA + 0.4 × Opp FTA - ...))
- Adjust all per-game stats to per-100 possessions:
  - Points per 100 = Points × (100 / Pace)
- Calculate expected game pace: Average of both teams' pace
- Variance adjustment: High pace (>102) = 1.15× confidence interval width
- Add pace differential as feature (Team A pace - Team B pace)

**Technical Notes**:
- Integration: `advanced_stats_v2.py` module
- Affects spread predictions: Wider confidence intervals for high-pace games
- Validation: Fast-paced teams (e.g., Pacers, Suns) should have wider prediction bands

---

### 2.2 Model Architecture Requirements

#### MR-1: Upgrade Meta-Learner from Weighted Averaging to Neural Network
**Priority**: P0 (Critical)
**Impact**: 2-4% accuracy improvement via context-aware model selection
**Rationale**: Current weighted averaging can't learn WHEN to trust each base model

**Acceptance Criteria**:
- Replace simple weighted averaging with **Stacking Generalization**
- **Level 0 (Base Models)**: Keep existing 8 models (XGBoost, LightGBM, etc.)
- **Level 1 (Meta-Learner)**:
  - Option A: Small neural network (2 hidden layers, 32-16 neurons, ReLU activation)
  - Option B: XGBoost (simpler, faster, often as good)
  - Option C: Logistic Regression with polynomial features (interactions between base models)
- **Input to Meta-Learner**:
  - Base model predictions (8 values)
  - Contextual features (12 values): days_rest, pace, injury_count, etc.
- **Training Protocol**:
  - K-fold cross-validation (K=5) with TimeSeriesSplit
  - Out-of-fold (OOF) predictions to prevent leakage
  - Meta-learner trains on OOF predictions + context
- **Validation**: Backtest ROI improves by ≥1.5 percentage points

**Technical Notes**:
- Modify `model_trainer.py:3105` (`EnsembleMoneylineModel` class)
- Use `sklearn.ensemble.StackingClassifier` for moneyline
- Use `sklearn.ensemble.StackingRegressor` for props
- Hyperparameter tuning: Optuna for meta-learner architecture search
- Prevent overfitting: L2 regularization (alpha=0.01) for neural network

**Design Decision**:
- **Assumption**: Start with XGBoost meta-learner (faster training, easier to interpret)
- If XGBoost doesn't yield >1% improvement, upgrade to neural network
- Rationale: XGBoost captures non-linear interactions without neural network complexity

---

#### MR-2: Implement Model Confidence Scoring
**Priority**: P1 (High)
**Impact**: 70% higher ROI when filtering low-confidence bets
**Rationale**: Not all predictions are equal; should avoid betting on uncertain games

**Acceptance Criteria**:
- **Confidence Score** (0-100):
  - Based on agreement among base models (low variance = high confidence)
  - Inverse of prediction standard deviation across ensemble
  - Formula: `confidence = 100 × (1 - min(std_dev / mean, 1))`
- **Edge Quality Tiers**:
  - **Elite** (90-100): Bet 1.0× Kelly
  - **Strong** (75-89): Bet 0.5× Kelly
  - **Moderate** (60-74): Bet 0.25× Kelly
  - **Weak** (40-59): Monitor only
  - **Avoid** (<40): Do not bet
- **Uncertainty Flags**:
  - `HIGH_UNCERTAINTY` if key player is GTD (game-time decision)
  - `DATA_INCOMPLETE` if ≥3 features are missing
- Output: Add `confidence_score` and `edge_quality_tier` to predictions CSV

**Technical Notes**:
- Implementation: `edge_quality.py` already exists (line 918)
- Integration: `daily_predictions.py:1200+` adds confidence to output
- Validation: Filter backtest to only bet on Elite+Strong tiers, ROI should increase

---

#### MR-3: Quantile Regression for Confidence Intervals
**Priority**: P2 (Medium)
**Impact**: Better risk management via prediction bands
**Rationale**: Point estimates miss distribution shape; need percentiles for bet sizing

**Acceptance Criteria**:
- Train **quantile regressors** for 10th, 50th (median), 90th percentiles
- Use LightGBM with quantile loss function:
  - `objective='quantile'`, `alpha=0.1` for 10th percentile
  - `alpha=0.5` for median
  - `alpha=0.9` for 90th percentile
- Output predictions with uncertainty bands:
  - `prediction_low` (10th percentile)
  - `prediction_median` (50th percentile)
  - `prediction_high` (90th percentile)
- **Bet Sizing Logic**:
  - Wide bands (high - low > 8 pts) → Reduce bet size by 50%
  - Narrow bands (high - low < 3 pts) → Increase confidence score by 10%
- Validation: Empirical coverage should match theoretical (10% below low, 10% above high)

**Technical Notes**:
- Already partially implemented: `QuantilePropModel` in `model_trainer.py:1818`
- Extend to all prop types (currently only player props)
- Integration: Add columns to output CSV: `pred_low`, `pred_median`, `pred_high`

---

### 2.3 Data Quality and Validation Requirements

#### DQ-1: Temporal Leakage Prevention (Mandatory)
**Priority**: P0 (Critical)
**Impact**: Prevents artificially inflated backtest performance
**Rationale**: Using future data in training = unrealistic performance estimates

**Acceptance Criteria**:
- **All** feature generation functions must accept `game_date` parameter
- When backtesting, NEVER use data from:
  - The target game itself
  - Any game after `game_date`
- Use temporal-safe functions:
  - `fetch_team_statistics_before_date()` instead of `fetch_team_statistics()`
  - `fetch_player_stats_before_date_auto()` instead of `fetch_player_stats()`
- **Validation Protocol**:
  - Automated test: Fetch features for a historical game (e.g., Oct 25, 2025)
  - Assert: No feature uses data from Oct 26+
  - Run on 100 random historical games
- **Sanity Checks**:
  - If backtest ROI > 15% → Flag as potential leakage
  - If win rate > 60% → Flag as potential leakage
  - If Sharpe ratio > 3.0 → Flag as potential leakage

**Technical Notes**:
- Already implemented in `feature_engineering.py` (lines 13-43)
- Enforcement: Add unit tests in `tests/test_temporal_discipline.py`
- CI/CD: Run temporal leakage tests on every commit

---

#### DQ-2: Data Completeness Validation
**Priority**: P1 (High)
**Impact**: Prevents missing data from causing poor predictions
**Rationale**: 176 "feature errors" on Jan 7, 2026 suggest incomplete data

**Acceptance Criteria**:
- **Pre-Prediction Validation**:
  - Check required fields are non-null for each team/player
  - If ≥3 team features are missing → Skip game (flag as `DATA_INCOMPLETE`)
  - If player has <3 recent games → Use season average with `low_confidence` flag
- **Required Team Features** (must exist):
  - `win_pct`, `pts_avg`, `opp_pts_avg`, `days_rest`, `home_win_pct`
- **Required Player Features**:
  - `pts_avg`, `min_avg`, `usage_rate`, `injury_status`
- **Fallback Values**:
  - Missing `days_rest` → Default to 1 (not 0, which implies back-to-back)
  - Missing `win_pct` → Use league average (0.500)
  - Missing advanced stats → Use positional average
- **Alert System**: Log warnings to `data_quality_report_{date}.json` daily

**Technical Notes**:
- New module: `data_validator.py`
- Integration: Run validation before `daily_predictions.py` generates outputs
- Thresholds: If >10% of games have incomplete data, halt predictions and alert

---

#### DQ-3: Outlier Detection and Handling
**Priority**: P2 (Medium)
**Impact**: Prevents extreme values from skewing predictions
**Rationale**: SGA's 30.5pt error suggests model can't handle outlier performances

**Acceptance Criteria**:
- **Training Data Cleaning**:
  - Remove games with impossible values (e.g., 200 points, -50 rebounds)
  - Cap outliers at 99th percentile during training (Winsorization)
- **Prediction Clipping**:
  - Player points: Min = 0, Max = 60
  - Rebounds: Min = 0, Max = 25
  - Assists: Min = 0, Max = 20
  - Threes: Min = 0, Max = 12
- **Anomaly Detection**:
  - If prediction is >3 standard deviations from player's season average → Flag as `OUTLIER_RISK`
  - Reduce confidence score by 20% for outlier predictions
- **Post-Prediction Audit**:
  - Daily report: "Predictions in 95th+ percentile" for manual review

**Technical Notes**:
- Implementation: Add to `model_trainer.py` in `BaseModelTrainer._preprocess_features()`
- Validation: Check distribution of predictions matches historical distribution

---

### 2.4 System Performance Requirements

#### SR-1: Prediction Generation Speed
**Priority**: P1 (High)
**Impact**: Enables real-time betting (odds change rapidly)
**Rationale**: Must generate predictions before lines close

**Acceptance Criteria**:
- **Latency Targets**:
  - Generate predictions for all games in 1 day: < 5 minutes
  - Single game prediction (on-demand): < 10 seconds
  - Real-time injury update triggers re-prediction: < 30 seconds
- **Optimization Strategies**:
  - Cache team statistics (refresh every 6 hours)
  - Parallelize model inference (8 base models in parallel)
  - Use lightweight models for low-edge games (faster inference)
- **Monitoring**: Log prediction generation time to CloudWatch/DataDog

**Technical Notes**:
- Current bottleneck: API calls to Balldontlie (600/min limit)
- Solution: Batch API calls, use `concurrent.futures` for parallelization
- Alternative: Pre-fetch data for all games at 9 AM daily

---

#### SR-2: Model Retraining Frequency
**Priority**: P1 (High)
**Impact**: Keeps model current with league trends
**Rationale**: NBA meta-game shifts (e.g., 3-point revolution) require adaptation

**Acceptance Criteria**:
- **Full Retraining**: Every 14 days (2 weeks)
  - Retrain all base models + meta-learner
  - Use all data from last 2 seasons
  - Duration: < 4 hours
- **Incremental Updates**: Every 3 days
  - Update only last layer (meta-learner) with recent 100 games
  - Duration: < 15 minutes
- **Drift Detection**: Daily
  - Monitor prediction error vs actual results
  - If RMSE increases >10% for 3 consecutive days → Trigger full retrain
- **A/B Testing**:
  - Deploy new model to 20% of predictions for 3 days
  - If performance ≥ current model → Promote to 100%

**Technical Notes**:
- Already implemented: `continuous_learning/` module
- Enhancement: Add automated drift detection alerts (Slack/email)
- Infrastructure: Use Railway scheduled jobs or GitHub Actions for retraining

---

### 2.5 User Interface and Output Requirements

#### UI-1: Prediction Output Format
**Priority**: P0 (Critical)
**Impact**: Usability for betting decisions
**Rationale**: Must be actionable for daily use

**Acceptance Criteria**:
- **CSV Output** (`predictions_{date}.csv`):
  - Columns: `game_id`, `game_time`, `home_team`, `away_team`, `prediction_type` (moneyline/spread/prop)
  - Prediction: `predicted_value`, `confidence_score`, `edge_quality_tier`
  - Uncertainty: `pred_low`, `pred_median`, `pred_high` (10th/50th/90th percentiles)
  - Context: `key_injuries`, `days_rest_diff`, `pace_projection`, `line_movement`
  - Recommendation: `bet_recommendation` (BET/MONITOR/AVOID), `suggested_bet_size` (Kelly %)
- **JSON API** (for frontend dashboard):
  - Same data as CSV, structured as JSON
  - Endpoint: `/api/predictions/{date}`
- **Alerts**:
  - Flag games with `HIGH_VALUE` (edge > 5%, confidence > 80%)
  - Flag games with `HIGH_UNCERTAINTY` (player GTD, low data quality)

**Technical Notes**:
- Implementation: `daily_predictions.py:1799` already generates CSV
- Enhancement: Add JSON export and API endpoint in `backend/api.py`

---

#### UI-2: Backtesting Reports
**Priority**: P1 (High)
**Impact**: Trust in model via transparent performance metrics
**Rationale**: Users need proof model works before risking capital

**Acceptance Criteria**:
- **Automated Backtest**: Run monthly (full season replay)
- **Report Sections**:
  1. **Overall Performance**: ROI, win rate, Sharpe ratio, max drawdown
  2. **By Bet Type**: Moneyline, spread, player props breakdown
  3. **By Confidence Tier**: Elite, Strong, Moderate performance comparison
  4. **Calibration Plots**: Predicted probability vs actual outcome frequency
  5. **Worst Misses**: Top 20 largest errors with root cause analysis
  6. **Drift Analysis**: Model accuracy trend over time
- **Format**: HTML report with interactive Plotly charts
- **Distribution**: Email to stakeholders, uploaded to S3 bucket

**Technical Notes**:
- Current: `comprehensive_backtest.py` generates JSON
- Enhancement: Add HTML report generation using Jinja2 templates
- Visualization: Use Plotly for interactive charts (reliability diagrams, ROI curves)

---

### 2.6 Risk Management and Bankroll Requirements

#### RM-1: Kelly Criterion Bet Sizing
**Priority**: P0 (Critical)
**Impact**: Maximizes long-term growth while controlling risk
**Rationale**: Optimal bet sizing is as important as accurate predictions

**Acceptance Criteria**:
- **Kelly Formula**: `f* = (bp - q) / b`
  - Where: `b` = decimal odds - 1, `p` = win probability, `q` = 1 - p
- **Fractional Kelly**: Use 1/4 Kelly for conservative growth
  - Full Kelly can lead to >20% drawdowns
  - Quarter Kelly: smoother equity curve, lower volatility
- **Bet Size Caps**:
  - Single bet maximum: 5% of bankroll
  - Daily total exposure: 20% of bankroll
  - Correlated bets: If 2+ bets on same game, halve each bet size
- **Dynamic Adjustment**:
  - After 10% bankroll drawdown → Reduce to 1/8 Kelly until recovery
  - After 3 consecutive losing days → Reduce bet sizes by 50%
- **Integration**: Calculate `suggested_bet_size` in prediction output

**Technical Notes**:
- Implementation: `portfolio_optimizer.py` already exists
- Enhancement: Add correlation matrix for same-game parlays
- Validation: Backtest with Kelly vs flat betting → Kelly should have higher Sharpe ratio

---

#### RM-2: Stop-Loss and Drawdown Management
**Priority**: P1 (High)
**Impact**: Prevents catastrophic losses during cold streaks
**Rationale**: Even best models have variance; need guardrails

**Acceptance Criteria**:
- **Daily Stop-Loss**: If down >3% of bankroll in a day → Stop betting
- **Weekly Stop-Loss**: If down >8% in a week → Pause betting, investigate
- **Maximum Drawdown**: If down >15% from peak → Halt all betting, retrain model
- **Recovery Protocol**:
  - After stop-loss trigger, resume with 1/2 normal bet sizes for 20 bets
  - If positive after 20 bets → Return to normal sizing
- **Alerts**: Immediate notification (email/SMS) when stop-loss triggered

**Technical Notes**:
- New module: `risk_management.py`
- Integration: Check before placing each bet in betting bot
- Monitoring: Track bankroll in database, log all stop-loss events

---

## 3. Non-Functional Requirements

### 3.1 Reliability
- **Uptime**: 99.5% availability during NBA season (Oct-Jun)
- **Fault Tolerance**: If primary API (Balldontlie) fails, automatically fallback to NBA API
- **Data Validation**: Zero critical errors (e.g., negative probabilities, null predictions)

### 3.2 Scalability
- **Handle Load**: 15 games/day × 10 players/team = 300 player prop predictions + 30 team predictions
- **Future-Proof**: Architecture should support adding NFL, MLB models without rewrite

### 3.3 Maintainability
- **Code Quality**: 80%+ test coverage for critical modules (feature engineering, model training)
- **Documentation**: Docstrings for all public functions, README with setup instructions
- **Logging**: Structured logging (JSON) with correlation IDs for debugging

### 3.4 Security
- **API Keys**: Store in environment variables (`.env` file), never commit to Git
- **Data Privacy**: No PII (personally identifiable information) storage
- **Access Control**: API endpoints require authentication (JWT tokens)

---

## 4. Success Metrics and Validation

### 4.1 Model Performance Metrics (Primary)

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Player Props Overall R² | 0.681 | 0.750 | Backtest (2 seasons) |
| Points RMSE | 6.757 | < 5.5 | Backtest (2 seasons) |
| Threes R² | -0.568 | > 0.10 | Backtest (2 seasons) |
| PRA RMSE | 8.469 | < 7.0 | Backtest (2 seasons) |
| Injury Detection Rate | ~70% | > 95% | Manual audit (100 games) |
| Betting ROI (All Bets) | TBD | > 3% | Live betting (30 days) |
| Betting ROI (Elite Tier) | TBD | > 7% | Live betting (30 days) |

### 4.2 System Performance Metrics (Secondary)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Prediction Generation Time | < 5 min for all games | CloudWatch logs |
| Model Retraining Time | < 4 hours (full), < 15 min (incremental) | CI/CD pipeline |
| API Latency (p95) | < 500ms | Application monitoring |
| Data Completeness | > 98% | Daily validation reports |

### 4.3 Risk Metrics (Tertiary)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Maximum Drawdown | < 15% | Bankroll tracking |
| Sharpe Ratio | > 1.5 | Monthly backtests |
| Closing Line Value (CLV) | > 0 (beating closing line) | Compare predictions to closing odds |

---

## 5. Implementation Phases and Priorities

### Phase 1: Foundation (Week 1-2)
**Goal**: Fix critical issues, establish baseline
- ✅ **FR-4**: Real-time injury detection (eliminates 161 DNP errors)
- ✅ **FR-1**: Implement Four Factors (2-4% accuracy gain)
- ✅ **DQ-1**: Enforce temporal discipline (prevent leakage)
- ✅ **MR-1**: Upgrade to stacking meta-learner (2-4% accuracy gain)
- ✅ **Validation**: Run comprehensive backtest, ensure no regression

### Phase 2: Enhancement (Week 3-4)
**Goal**: Add advanced features, improve edge
- ✅ **FR-2**: Travel and fatigue features (~2 pts impact)
- ✅ **FR-5**: Betting market features (RLM, line movement)
- ✅ **MR-2**: Model confidence scoring (70% higher ROI)
- ✅ **RM-1**: Kelly criterion bet sizing
- ✅ **Validation**: Backtest shows ROI > 3%

### Phase 3: Optimization (Week 5-6)
**Goal**: Fine-tune, integrate advanced analytics
- ✅ **FR-3**: Player impact metrics (DARKO/EPM)
- ✅ **FR-6**: Pace-adjusted metrics
- ✅ **MR-3**: Quantile regression confidence intervals
- ✅ **RM-2**: Stop-loss and drawdown management
- ✅ **Validation**: Backtest shows ROI > 5%, Sharpe > 1.5

### Phase 4: Productionization (Week 7-8)
**Goal**: Deploy, monitor, iterate
- ✅ **SR-1**: Optimize prediction speed (< 5 min)
- ✅ **SR-2**: Automated retraining pipeline
- ✅ **UI-1**: Enhanced prediction outputs (JSON API)
- ✅ **UI-2**: Automated backtesting reports
- ✅ **Go-Live**: Deploy to production, start live betting with 10% bankroll

---

## 6. Assumptions and Open Questions

### Assumptions
1. **Data Access**: Balldontlie API remains available at 600 req/min
2. **Computational Resources**: Railway or Vercel can handle 4-hour retraining jobs
3. **Paid Data**: Budget available for RotoWire injury API (~$100/month)
4. **Historical Data**: 2+ seasons of game data available for training
5. **Meta-Learner**: XGBoost will outperform simple weighted averaging (if not, pivot to neural network)

### Open Questions (Need User Clarification)
1. **Budget for Paid APIs**:
   - RotoWire injury API: $100/month
   - Betting odds API (premium): $50/month
   - Player impact metrics (DARKO): Free (scraping) or paid API?
   - **Decision Needed**: Approve budget or use free alternatives?

2. **Live Betting Strategy**:
   - Start with paper trading (simulated bets) or real money?
   - If real money, what's the initial bankroll? (Recommendation: $5,000-$10,000)
   - **Decision Needed**: Paper trading for 30 days, then transition to live?

3. **Retraining Infrastructure**:
   - Railway scheduled jobs sufficient or need dedicated ML platform (AWS SageMaker)?
   - **Decision Needed**: Stick with Railway or migrate to AWS?

4. **Performance Reporting**:
   - Who receives backtest reports? (Email list, Slack channel?)
   - How often? (Weekly, monthly?)
   - **Decision Needed**: Set up email distribution list?

5. **Scope of "Most Accurate Ever"**:
   - Benchmark against public models (FiveThirtyEight, ESPN) or private sharp bettors?
   - Focus on player props (current strength) or expand to moneyline/spread?
   - **Decision Needed**: Define "SOTA" benchmark criteria?

---

## 7. Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Data API Rate Limits** | Medium | High | Implement caching, batch requests, use multiple APIs |
| **Injury API Unavailable** | Low | High | Build web scraper as fallback, use multiple sources |
| **Model Overfitting** | Medium | Medium | Strict temporal validation, out-of-sample testing, cross-validation |
| **Poor Four Factors Integration** | Low | Medium | Start with simple features, validate incrementally |
| **Slow Retraining (>4 hrs)** | Medium | Low | Use Dask for distributed training, optimize hyperparameters |
| **Live Betting Losses** | Medium | High | Start with paper trading, strict stop-loss rules, Kelly sizing |
| **Market Efficiency (No Edge)** | Medium | High | Focus on props (less efficient) vs spreads (more efficient) |

---

## 8. Dependencies and Integrations

### External APIs
- **Balldontlie**: Primary data source (600 req/min, free tier)
- **NBA API**: Fallback data source (slower, no rate limit)
- **RotoWire or FantasyLabs**: Real-time injury data (paid)
- **Odds API**: Multi-sportsbook odds (The Odds API, paid tier)
- **DARKO/EPM**: Player impact metrics (scraping or paid API)

### Internal Modules (Existing)
- `feature_engineering.py` (4801 lines) - Team/player features
- `model_trainer.py` (4507 lines) - Model classes
- `train_complete_balldontlie.py` (5992 lines) - Training pipeline
- `daily_predictions.py` (1799 lines) - Prediction generation
- `backtesting.py` (2013 lines) - Walk-forward validation

### Internal Modules (New/Modified)
- `advanced_stats_v2.py` - Four Factors calculation (**NEW**)
- `injury_tracker_v3.py` - Real-time injury detection (**NEW**)
- `player_impact_fetcher.py` - DARKO/EPM integration (**NEW**)
- `data_validator.py` - Pre-prediction validation (**NEW**)
- `risk_management.py` - Stop-loss, drawdown management (**NEW**)
- `stacking_meta_learner.py` - Neural network meta-learner (**NEW**)

---

## 9. Acceptance Criteria Summary

**The model is ready for production when ALL of the following are met**:

### Model Performance
- ✅ Player Props Overall R² ≥ 0.75
- ✅ Points RMSE < 5.5
- ✅ Threes R² > 0.10 (fixed from negative)
- ✅ PRA RMSE < 7.0
- ✅ Injury Detection Rate > 95%

### System Validation
- ✅ Comprehensive backtest (2 seasons) shows ROI > 3%
- ✅ Elite tier predictions show ROI > 7%
- ✅ Temporal leakage tests pass (100 random games)
- ✅ Data completeness > 98% (daily validation)
- ✅ Prediction generation < 5 minutes (all games)

### Risk Management
- ✅ Kelly criterion bet sizing implemented
- ✅ Stop-loss rules automated
- ✅ Maximum drawdown < 15% in backtest
- ✅ Sharpe ratio > 1.5

### Documentation
- ✅ All new functions have docstrings
- ✅ README updated with setup instructions
- ✅ Backtest report generated and reviewed
- ✅ API documentation for prediction endpoints

### User Approval
- ✅ User reviews backtest report and approves performance
- ✅ User approves Phase 4 go-live for paper trading
- ✅ After 30 days paper trading, user approves live betting

---

## 10. Conclusion

This PRD outlines a comprehensive roadmap to transform the existing NBA prediction model from its current state (R² = 0.68, RMSE = 5.4) to state-of-the-art performance (R² > 0.75, RMSE < 5.0, ROI > 5%). The approach is grounded in:

1. **Basketball Analytics Best Practices**: Dean Oliver's Four Factors, pace adjustments, player impact metrics
2. **Machine Learning Rigor**: Stacking ensembles, temporal validation, out-of-fold predictions
3. **Betting Industry Standards**: Kelly criterion, CLV tracking, confidence-based bet sizing
4. **Risk Management**: Stop-loss rules, drawdown limits, fractional Kelly

**Key Success Factors**:
- Fix injury detection immediately (biggest current weakness)
- Implement Four Factors (highest ROI feature engineering)
- Upgrade meta-learner from averaging to stacking (architectural improvement)
- Enforce temporal discipline religiously (prevent leakage)
- Validate relentlessly (comprehensive backtesting before deployment)

**Next Steps**:
1. User reviews and approves PRD
2. Technical specification document created
3. Implementation begins with Phase 1 (Foundation)
4. Weekly progress reviews with backtest performance updates

This model has the foundational infrastructure to achieve SOTA performance. Execution of the prioritized requirements will unlock the full potential.
