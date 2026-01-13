# Technical Specification Document (TSD)
# NBA Prediction Model v2 - Path to SOTA Performance

**Document Version**: 1.0
**Date**: January 13, 2026
**Technology Stack**: Python 3.x, scikit-learn, XGBoost, LightGBM, CatBoost, FastAPI, Railway, Vercel
**Data Sources**: Balldontlie API, NBA API, The Odds API

---

## 1. Technical Context

### 1.1 Current Architecture Overview

The NBA prediction system is a sophisticated ensemble-based ML platform with the following components:

**Core Modules** (Analysis from codebase):
- `model_trainer.py` (4,507 lines) - Ensemble models with calibration
- `feature_engineering.py` (4,801 lines) - Temporal-safe feature generation
- `advanced_stats_v2.py` (616 lines) - Four Factors calculator (Dean Oliver)
- `injury_impact_v2.py` - Usage redistribution and star player tracking
- `models/stacking_model.py` - Two-layer stacking architecture
- `daily_predictions.py` (1,799 lines) - Production prediction pipeline
- `comprehensive_backtest.py` - Walk-forward validation framework
- `continuous_learning/` - Drift detection, incremental training, model registry

**Current Model Architecture**:
1. **Moneyline (Win Probability)**:
   - 8-model ensemble: XGBoost (18%), LightGBM (15%), GradientBoosting (15%), RandomForest (15%), MLP (12%), CatBoost (12%), SVM (10%), LogisticRegression (8%)
   - Meta-learner: Weighted averaging (inverse-RMSE weights)
   - Calibration: Platt Scaling, Isotonic Regression, Temperature Scaling, Beta Calibration

2. **Player Props** (Points, Rebounds, Assists, Threes, PRA):
   - Stacking Model v2 (already implemented in `stacking_model.py`)
   - Level 0: XGBoost, LightGBM, Ridge, Lasso, GradientBoosting, RandomForest, optional CatBoost
   - Level 1: ElasticNet meta-learner trained on out-of-fold predictions

3. **Spread Predictions**:
   - Ensemble regression: SVR, RandomForest, GradientBoosting
   - Meta-learner: Ridge or XGBoost

**Key Strengths** (Already Implemented):
✅ Temporal discipline enforced (`fetch_*_before_date()` functions)
✅ Stacking architecture for player props
✅ Four Factors calculator (Dean Oliver)
✅ Injury impact module with usage redistribution
✅ Arena data with coordinates, altitude, timezone (for travel features)
✅ Continuous learning framework (drift detection, model registry)
✅ Comprehensive backtesting with walk-forward validation

**Critical Gaps** (From Requirements Analysis):
❌ Real-time injury detection (causing 161 DNP errors)
❌ Betting market features (line movement, RLM)
❌ Player impact metrics (DARKO/EPM/RAPTOR)
❌ Travel/fatigue features (only arena data exists, not used)
❌ Pace-adjusted metrics (formula exists but not integrated)
❌ Confidence scoring system for predictions
❌ Quantile regression for uncertainty bands
❌ Risk management automation (Kelly criterion, stop-loss)

---

## 2. Implementation Approach

### 2.1 Design Philosophy

**Guiding Principles**:
1. **Extend, Don't Rebuild**: Leverage existing robust infrastructure (stacking models, temporal discipline, calibration)
2. **Incremental Validation**: Each feature addition must show measurable improvement in backtest
3. **Production-First**: All features must work in real-time prediction pipeline (`daily_predictions.py`)
4. **Temporal Safety**: Zero tolerance for data leakage (enforce `game_date` parameters)
5. **Fail-Safe Defaults**: Missing data should gracefully degrade, not crash predictions

**Implementation Strategy**:
- **Phase 1 (Foundation)**: Fix critical bugs (injury detection), add high-ROI features (Four Factors integration)
- **Phase 2 (Enhancement)**: Add advanced features (travel, betting markets, confidence scoring)
- **Phase 3 (Optimization)**: Fine-tune models (quantile regression, player impact metrics)
- **Phase 4 (Production)**: Optimize performance, automate retraining, deploy monitoring

---

## 3. Source Code Structure Changes

### 3.1 New Modules

#### 3.1.1 `injury_tracker_v3.py` - Real-Time Injury Detection
**Purpose**: Replace delayed injury detection with real-time multi-source injury feeds
**Location**: `/injury_tracker_v3.py`
**Dependencies**: `requests`, `beautifulsoup4`, `data_fetcher.py`

**Architecture**:
```python
class InjuryTracker:
    """
    Multi-source injury tracker with 15-minute refresh cycle.

    Data Sources (priority order):
    1. RotoWire API (paid, most reliable)
    2. NBA.com/injuries (scraping fallback)
    3. ESPN injury report (scraping fallback)
    4. Balldontlie injuries endpoint (free, lower quality)

    Caching: SQLite database with 15-minute TTL
    """

    def __init__(self, sources=['rotowire', 'nba_com', 'espn', 'balldontlie']):
        self.sources = sources
        self.cache_db = 'injury_cache.db'
        self.refresh_interval = 900  # 15 minutes

    def get_injury_status(self, player_id: int, as_of_date: str = None) -> Dict:
        """
        Get player injury status with confidence score.

        Returns:
            {
                'status': 'OUT' | 'DOUBTFUL' | 'QUESTIONABLE' | 'GTD' | 'ACTIVE',
                'injury_type': 'knee' | 'ankle' | 'rest' | 'illness' | None,
                'confidence': 0.0-1.0,  # Agreement across sources
                'last_update': datetime,
                'source': 'rotowire' | 'nba_com' | 'composite'
            }
        """

    def get_team_injury_report(self, team_id: int, as_of_date: str = None) -> List[Dict]:
        """Get all injuries for a team."""

    def calculate_usage_redistribution(self, team_id: int, injured_players: List[int]) -> Dict:
        """
        Model how injured player's usage redistributes to teammates.

        Returns:
            {
                player_id: {
                    'usage_boost': 0.05,  # +5% usage rate
                    'minutes_boost': 3.2,  # +3.2 minutes
                    'role_change': 'backup_PG' -> 'starting_PG'
                }
            }
        """
```

**Integration Points**:
- `daily_predictions.py:500` - Pre-prediction validation check
- `feature_engineering.py:generate_*_prop_features()` - Add injury flags and usage boosts
- Database: SQLite cache at `data/injury_cache.db`

**Success Criteria**:
- Zero DNP (Did Not Play) players in predictions
- 95%+ accuracy on injury status detection
- < 30 second latency for injury status lookup

---

#### 3.1.2 `betting_market_features.py` - Line Movement and Market Intelligence
**Purpose**: Track odds changes, reverse line movement, consensus for betting edge
**Location**: `/betting_market_features.py`
**Dependencies**: `odds_fetcher.py` (existing), PostgreSQL/TimescaleDB

**Architecture**:
```python
class BettingMarketAnalyzer:
    """
    Track and analyze betting market signals.

    Features Generated:
    - Opening line (first posted odds)
    - Closing line (final odds before tipoff)
    - Line movement (closing - opening)
    - Reverse Line Movement (line moves opposite to bet percentage)
    - Steam moves (rapid line changes >1.5 pts in <5 min)
    - Consensus odds (average across 10+ books)
    """

    def __init__(self, db_connection: str = 'postgresql://...'):
        self.db = db_connection
        self.sportsbooks = [
            'DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'BetRivers',
            'PointsBet', 'Barstool', 'WynnBET', 'Unibet', 'Bet365'
        ]

    def track_line_history(self, game_id: str, market: str = 'spread') -> pd.DataFrame:
        """
        Retrieve time-series of odds for a game.

        Returns DataFrame with columns:
        - timestamp, sportsbook, home_line, away_line, home_price, away_price
        """

    def detect_reverse_line_movement(self, game_id: str) -> Dict:
        """
        Detect RLM: Line moves against majority of bets (sharp money indicator).

        Example: 70% of bets on Lakers, but line moves from LAL -5 to LAL -3
        This suggests sharp bettors are loading up on the opponent.

        Returns:
            {
                'is_rlm': True,
                'bet_percentage_home': 0.30,  # 30% on home team
                'line_movement': -2.0,  # Line moved 2 pts toward away team
                'sharp_side': 'away',  # Sharp money on away team
                'confidence': 0.85
            }
        """

    def calculate_market_features(self, game_id: str, as_of_time: datetime = None) -> Dict:
        """
        Generate betting market features for model input.

        Returns:
            {
                'opening_spread': -5.0,
                'closing_spread': -7.0,
                'line_movement': -2.0,
                'is_rlm': True,
                'steam_move_count': 2,
                'consensus_spread': -6.8,
                'spread_variance': 1.2,  # Disagreement across books
                'bet_pct_favorite': 0.68
            }
        """
```

**Database Schema** (PostgreSQL with TimescaleDB):
```sql
CREATE TABLE odds_history (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50),
    sportsbook VARCHAR(50),
    market VARCHAR(20),  -- 'spread', 'moneyline', 'total'
    timestamp TIMESTAMP,
    home_line NUMERIC,
    away_line NUMERIC,
    home_price INTEGER,  -- American odds (e.g., -110)
    away_price INTEGER
);

CREATE INDEX idx_odds_game_time ON odds_history(game_id, timestamp);
SELECT create_hypertable('odds_history', 'timestamp');  -- TimescaleDB
```

**Integration Points**:
- `feature_engineering.py:generate_game_features()` - Add 8 new betting market columns
- `odds_fetcher.py` - Extend to log history (currently only fetches current odds)
- Scheduled task: Poll odds every 5 minutes during game day (9 AM - 11 PM EST)

**Success Criteria**:
- RLM detection accuracy > 80% (validated against known sharp action)
- Betting market features show >1% improvement in backtest ROI
- < 500ms latency for market feature retrieval

---

#### 3.1.3 `player_impact_metrics.py` - DARKO/EPM/RAPTOR Integration
**Purpose**: Fetch advanced player impact metrics (beyond box score stats)
**Location**: `/player_impact_metrics.py`
**Dependencies**: `requests`, `beautifulsoup4`

**Architecture**:
```python
class PlayerImpactFetcher:
    """
    Fetch and cache advanced player impact metrics.

    Metrics:
    - DARKO DPM (Daily Plus-Minus): darko.fyi
    - ESPN EPM (Estimated Plus-Minus): espn.com/nba/stats
    - FiveThirtyEight RAPTOR: fivethirtyeight.com
    - Basketball Reference BPM (Box Plus-Minus): basketball-reference.com

    Caching: Daily refresh (metrics update once per day)
    """

    def __init__(self, cache_dir: str = 'data/impact_metrics'):
        self.cache_dir = cache_dir
        self.metrics_cache = {}  # player_id -> metrics dict
        self.last_update = None

    def fetch_darko_dpm(self, season: str = '2025-26') -> pd.DataFrame:
        """
        Scrape DARKO DPM from darko.fyi (free, publicly available).

        Returns DataFrame:
        - player_name, player_id, dpm, dpm_offense, dpm_defense, minutes
        """

    def fetch_espn_epm(self, season: str = '2025-26') -> pd.DataFrame:
        """Fetch ESPN's EPM (web scraping)."""

    def fetch_raptor(self, season: str = '2025-26') -> pd.DataFrame:
        """Fetch FiveThirtyEight RAPTOR (CSV download)."""

    def get_player_impact(self, player_id: int, metric: str = 'auto') -> float:
        """
        Get player's impact metric (standardized -10 to +10 scale).

        Priority: DARKO > EPM > RAPTOR > BPM
        """

    def calculate_team_impact_sum(self, team_id: int, player_ids: List[int]) -> float:
        """Sum of team's active players' impact metrics."""
```

**Integration Points**:
- `feature_engineering.py:generate_*_prop_features()` - Add `player_impact_score` column
- `feature_engineering.py:generate_game_features()` - Add `team_impact_diff` for spreads
- Scheduled task: Update metrics daily at 6 AM EST

**Success Criteria**:
- 95%+ player coverage (all rotation players have metrics)
- Player prop RMSE improves by 3-5%
- Update latency < 2 hours after new data published

---

#### 3.1.4 `confidence_scoring.py` - Prediction Confidence and Edge Quality
**Purpose**: Assign confidence scores and edge quality tiers to predictions
**Location**: `/confidence_scoring.py`
**Dependencies**: `numpy`, `pandas`

**Architecture**:
```python
class ConfidenceScorer:
    """
    Calculate prediction confidence based on ensemble agreement.

    Confidence Factors:
    1. Ensemble variance (low variance = high confidence)
    2. Data completeness (missing features reduce confidence)
    3. Injury uncertainty (GTD players reduce confidence)
    4. Historical accuracy for similar matchups
    """

    def calculate_ensemble_confidence(self, base_predictions: np.ndarray) -> float:
        """
        Confidence from base model agreement.

        confidence = 100 × (1 - min(std_dev / mean, 1))

        Example:
        - Predictions: [24.2, 24.5, 24.1, 24.4, 24.3] → std=0.15, mean=24.3 → conf=99.4
        - Predictions: [20.1, 26.5, 22.8, 28.3, 21.2] → std=3.5, mean=23.8 → conf=85.3
        """

    def adjust_for_data_quality(self, confidence: float, missing_features: int) -> float:
        """
        Reduce confidence for incomplete data.

        Penalty: -10 points per missing critical feature
        """

    def adjust_for_injury_uncertainty(self, confidence: float, gtd_count: int) -> float:
        """
        Reduce confidence when key players are game-time decisions.

        Penalty: -15 points per GTD player in top-5 usage
        """

    def assign_edge_quality_tier(self, confidence: float, predicted_edge: float) -> str:
        """
        Categorize prediction quality for bet sizing.

        Tiers:
        - ELITE (90-100): High confidence, large edge → Bet 1.0× Kelly
        - STRONG (75-89): Good confidence, moderate edge → Bet 0.5× Kelly
        - MODERATE (60-74): Uncertain, small edge → Bet 0.25× Kelly
        - WEAK (40-59): Low confidence → Monitor only, no bet
        - AVOID (<40): Very low confidence → Do not bet
        """

    def calculate_prediction_confidence(
        self,
        base_predictions: np.ndarray,
        missing_features: int = 0,
        gtd_count: int = 0,
        historical_accuracy: float = None
    ) -> Dict:
        """
        Master confidence calculation.

        Returns:
            {
                'confidence_score': 87.3,
                'edge_quality_tier': 'STRONG',
                'recommended_bet_size': 0.5,  # Fractional Kelly
                'uncertainty_flags': ['GTD_PLAYER_PRESENT']
            }
        """
```

**Integration Points**:
- `daily_predictions.py:1200` - Add confidence columns to output CSV
- `portfolio_optimizer.py` - Use confidence for bet sizing (existing module, extend)

**Success Criteria**:
- Elite tier bets show >7% ROI (vs 3% for all bets)
- Confidence score correlates with actual accuracy (r > 0.6)
- Low-confidence bets (<60) are correctly flagged 80%+ of time

---

#### 3.1.5 `risk_management.py` - Kelly Criterion and Stop-Loss Automation
**Purpose**: Automate bankroll management and risk controls
**Location**: `/risk_management.py`
**Dependencies**: `numpy`, `pandas`

**Architecture**:
```python
class RiskManager:
    """
    Automated risk management for betting operations.

    Features:
    - Kelly Criterion bet sizing
    - Daily/weekly stop-loss enforcement
    - Drawdown monitoring
    - Correlation adjustment for same-game bets
    """

    def __init__(self, bankroll: float, kelly_fraction: float = 0.25):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction  # Conservative 1/4 Kelly
        self.peak_bankroll = bankroll
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.stop_loss_triggered = False

    def calculate_kelly_bet_size(
        self,
        win_probability: float,
        decimal_odds: float,
        confidence_score: float = 1.0
    ) -> float:
        """
        Kelly Criterion: f* = (bp - q) / b

        Where:
        - b = decimal_odds - 1
        - p = win_probability
        - q = 1 - p

        Returns: Bet size as fraction of bankroll (e.g., 0.03 = 3%)
        """

    def check_stop_loss(self) -> Dict:
        """
        Enforce stop-loss rules.

        Rules:
        - Daily stop-loss: -3% of bankroll in a day → STOP
        - Weekly stop-loss: -8% in a week → PAUSE, investigate
        - Maximum drawdown: -15% from peak → HALT, retrain model

        Returns:
            {
                'should_stop': True/False,
                'reason': 'DAILY_STOP_LOSS',
                'current_drawdown': 0.035,
                'recovery_protocol': 'HALF_SIZE_FOR_20_BETS'
            }
        """

    def adjust_for_correlation(self, bets: List[Dict]) -> List[Dict]:
        """
        Reduce bet sizes for correlated bets (same game).

        Example: Betting on both Lakers spread and Lakers player prop
        → Reduce each bet size by 50% to account for correlation
        """

    def apply_bet_size_caps(self, bet_size: float) -> float:
        """
        Enforce bet size limits.

        Caps:
        - Single bet max: 5% of bankroll
        - Daily total exposure: 20% of bankroll
        """
```

**Integration Points**:
- `daily_predictions.py` - Calculate suggested_bet_size for each prediction
- Betting bot (if exists) - Check stop-loss before placing bets
- Database: Log bankroll history for drawdown tracking

**Success Criteria**:
- Kelly-sized bets show higher Sharpe ratio than flat betting
- Stop-loss prevents >15% drawdown in backtests
- Correlation adjustment reduces simultaneous loss frequency

---

### 3.2 Modified Modules

#### 3.2.1 `feature_engineering.py` - Travel and Fatigue Features
**Changes**: Extend existing `calculate_rest_and_fatigue()` function
**Location**: `feature_engineering.py:2800+`

**Current Implementation** (Line 86-120):
- Arena data exists with coordinates, altitude, timezone
- Function stub exists but not fully implemented

**New Implementation**:
```python
def calculate_rest_and_fatigue(
    team_id: int,
    game_date: str,
    is_home: bool = True,
    opponent_id: int = None
) -> Dict:
    """
    Calculate travel and fatigue features.

    Features Generated:
    1. days_rest: Days since last game (0 = back-to-back)
    2. is_back_to_back: Binary flag
    3. travel_distance: Miles traveled using Haversine formula
    4. altitude_change: Feet elevation change (Denver = +5280)
    5. timezone_crossed: Number of time zones crossed
    6. schedule_density: Games in last 5 days
    7. road_trip_length: Consecutive away games
    8. fatigue_score: Composite fatigue metric (0-10)

    Returns:
        {
            'days_rest': 2,
            'is_back_to_back': False,
            'travel_distance': 1453.2,  # miles
            'altitude_change': 5280,  # going to Denver
            'timezone_crossed': 2,
            'schedule_density': 3,  # 3 games in 5 days
            'road_trip_length': 2,  # 2nd game of road trip
            'fatigue_score': 6.2  # moderate fatigue
        }
    """
```

**Integration**:
- Call in `generate_game_features()` for both home and away teams
- Add differential features: `fatigue_diff = home_fatigue - away_fatigue`

---

#### 3.2.2 `advanced_stats_v2.py` - Pace-Adjusted Metrics
**Changes**: Add pace calculation and per-100-possession adjustments
**Location**: `advanced_stats_v2.py:FourFactorsCalculator`

**New Methods**:
```python
def calculate_pace(self, team_stats: Dict, opp_stats: Dict = None) -> float:
    """
    Calculate team pace (possessions per 48 minutes).

    Pace = 48 × (Team Poss + Opp Poss) / (2 × Minutes)
    """

def adjust_for_pace(self, stat_value: float, team_pace: float, league_avg_pace: float = 100.0) -> float:
    """
    Convert per-game stat to per-100-possessions.

    Adjusted = stat_value × (100 / team_pace)
    """

def calculate_expected_game_pace(self, team1_id: int, team2_id: int, game_date: str) -> Dict:
    """
    Predict game pace based on both teams' tendencies.

    Returns:
        {
            'expected_pace': 102.3,
            'variance_multiplier': 1.15,  # High pace = wider spread
            'pace_differential': 5.2  # Team1 pace - Team2 pace
        }
    """
```

**Integration**:
- Add to `generate_game_features()` - Include `expected_pace`, `pace_diff` columns
- Use `variance_multiplier` in spread predictions for confidence intervals

---

#### 3.2.3 `model_trainer.py` - Quantile Regression Enhancements
**Changes**: Extend existing `QuantilePropModel` to all prediction types
**Location**: `model_trainer.py:1818` (QuantilePropModel class)

**Enhancements**:
```python
class QuantilePropModel:
    """
    Multi-quantile regression for uncertainty quantification.

    Trains 3 models:
    - 10th percentile (conservative lower bound)
    - 50th percentile (median prediction)
    - 90th percentile (optimistic upper bound)
    """

    def predict_with_intervals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return predictions with uncertainty bands.

        Returns DataFrame with columns:
        - pred_low (10th percentile)
        - pred_median (50th percentile)
        - pred_high (90th percentile)
        - pred_range (high - low)
        """
```

**Integration**:
- Call in `daily_predictions.py` for all prop predictions
- Add bet sizing logic: Wide ranges (>8 pts) → reduce bet size by 50%

---

## 4. Data Model / API / Interface Changes

### 4.1 Database Schema Changes

#### 4.1.1 New Table: `injury_status`
```sql
CREATE TABLE injury_status (
    id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    status VARCHAR(20),  -- OUT, DOUBTFUL, QUESTIONABLE, GTD, ACTIVE
    injury_type VARCHAR(50),
    source VARCHAR(50),
    confidence NUMERIC(3,2),
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, game_date, source)
);
```

#### 4.1.2 New Table: `odds_history` (TimescaleDB)
```sql
CREATE TABLE odds_history (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) NOT NULL,
    sportsbook VARCHAR(50) NOT NULL,
    market VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    home_line NUMERIC,
    away_line NUMERIC,
    home_price INTEGER,
    away_price INTEGER
);

SELECT create_hypertable('odds_history', 'timestamp');
```

#### 4.1.3 New Table: `player_impact_metrics`
```sql
CREATE TABLE player_impact_metrics (
    id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    season VARCHAR(10) NOT NULL,
    metric_name VARCHAR(20) NOT NULL,
    metric_value NUMERIC(5,2),
    last_update DATE,
    UNIQUE(player_id, season, metric_name)
);
```

### 4.2 Enhanced Prediction Output Format

**CSV Columns**:
```csv
game_id,game_time,home_team,away_team,prediction_type,predicted_value,
confidence_score,edge_quality_tier,pred_low,pred_median,pred_high,
suggested_bet_size,bet_recommendation,key_injuries,days_rest_diff,
pace_projection,line_movement,uncertainty_flags
```

---

## 5. Delivery Phases (Incremental, Testable Milestones)

### Phase 1: Foundation (Week 1-2)
**Goal**: Fix critical issues, establish baseline

**Tasks**:
1. Injury Tracker v3 (5 days)
2. Four Factors Integration (3 days)
3. Temporal Discipline Audit (2 days)
4. Meta-Learner Upgrade (2 days)

**Success Criteria**:
- Zero DNP players ✅
- Player props R² ≥ 0.70 ✅
- Points RMSE ≤ 6.5 ✅

### Phase 2: Enhancement (Week 3-4)
**Goal**: Add features with proven ROI

**Tasks**:
1. Travel and Fatigue Features (4 days)
2. Betting Market Features (5 days)
3. Confidence Scoring (3 days)
4. Kelly Criterion Bet Sizing (2 days)

**Success Criteria**:
- Backtest ROI ≥ 3.5% ✅
- Elite tier ROI ≥ 7% ✅

### Phase 3: Optimization (Week 5-6)
**Goal**: Integrate advanced analytics

**Tasks**:
1. Player Impact Metrics (4 days)
2. Pace-Adjusted Metrics (2 days)
3. Quantile Regression (3 days)
4. Stop-Loss Automation (2 days)

**Success Criteria**:
- Player props R² ≥ 0.75 ✅
- Points RMSE < 5.5 ✅
- Backtest ROI ≥ 5% ✅

### Phase 4: Productionization (Week 7-8)
**Goal**: Deploy and monitor

**Tasks**:
1. Performance Optimization (3 days)
2. Retraining Automation (3 days)
3. API Endpoints (2 days)
4. Backtesting Reports (2 days)
5. Go-Live Preparation (2 days)

**Success Criteria**:
- Prediction latency < 5 min ✅
- API latency p95 < 500ms ✅
- Automated retraining operational ✅

---

## 6. Verification Approach

### 6.1 Unit Testing

**Test Coverage**: 80%+ for critical modules

**Key Test Suites**:
- `tests/test_temporal_discipline.py` - Temporal leakage detection
- `tests/test_injury_tracker.py` - Injury detection accuracy
- `tests/test_features.py` - Feature engineering validation
- `tests/test_confidence.py` - Confidence scoring

### 6.2 Backtest Validation

**Methodology**: Walk-forward validation with temporal splits

**Test Periods**:
- Training: 2023-24 + 2024-25 (Oct-Mar)
- Validation: 2024-25 (Apr-Jun)
- Test: 2025-26 (Oct-Dec)

**Validation Metrics**:
| Metric | Baseline | Target |
|--------|----------|--------|
| Player Props R² | 0.681 | ≥ 0.750 |
| Points RMSE | 6.757 | < 5.5 |
| Threes R² | -0.568 | > 0.10 |
| PRA RMSE | 8.469 | < 7.0 |
| Betting ROI | TBD | > 3% |
| Elite ROI | TBD | > 7% |
| Sharpe Ratio | TBD | > 1.5 |

---

## 7. Technical Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API Rate Limits | Low | High | Multiple sources, caching |
| Injury Scraping Breaks | Medium | High | Multi-source, monitoring |
| Model Overfitting | Medium | High | Temporal validation, OOS testing |
| Slow Retraining | Medium | Low | Distributed training, optimization |
| Market Inefficiency | Medium | High | Focus on props, monitor CLV |

---

## 8. Dependencies and External Integrations

### 8.1 External APIs

| API | Purpose | Rate Limit | Cost | Fallback |
|-----|---------|-----------|------|----------|
| Balldontlie | Primary data | 600 req/min | Free | NBA API |
| RotoWire | Injury data | 1000/day | $100/mo | Scraping |
| The Odds API | Betting odds | 500/day | $50/mo | Manual |
| DARKO | Player impact | N/A | Free | EPM |

### 8.2 Infrastructure

- PostgreSQL with TimescaleDB (odds history)
- Redis (optional caching)
- Railway (backend hosting)
- Vercel (frontend hosting)

---

## 9. Acceptance Criteria Summary

### Phase 1 (Foundation)
- ✅ Zero DNP players in predictions
- ✅ Four Factors show ≥1% RMSE reduction
- ✅ Temporal leakage tests pass

### Phase 2 (Enhancement)
- ✅ Denver home advantage validated (+1.5 pts)
- ✅ Elite tier ROI > 7%
- ✅ Kelly Sharpe > flat betting

### Phase 3 (Optimization)
- ✅ Player props R² ≥ 0.75
- ✅ All prop targets met
- ✅ ROI ≥ 5%, Sharpe > 1.5

### Phase 4 (Production)
- ✅ Prediction latency < 5 min
- ✅ API operational
- ✅ Automated retraining
- ✅ Zero downtime

---

## 10. Conclusion

This technical specification provides a detailed implementation roadmap to achieve SOTA NBA prediction performance. The approach leverages existing infrastructure while systematically addressing critical gaps identified in the requirements analysis.

**Key Technical Decisions**:
1. Extend existing stacking architecture (proven effective)
2. Multi-source injury feeds for 95%+ accuracy
3. PostgreSQL + TimescaleDB for odds history
4. XGBoost meta-learner (start simple, upgrade if needed)
5. Conservative 1/4 Kelly for risk management

**Next Steps**:
1. Review and approve specification
2. Set up infrastructure (PostgreSQL, API keys)
3. Begin Phase 1 implementation
4. Run comprehensive backtest after each phase
5. Deploy to production after validation

The model is architecturally sound and positioned for industry-leading performance through systematic execution of this phased implementation plan.
