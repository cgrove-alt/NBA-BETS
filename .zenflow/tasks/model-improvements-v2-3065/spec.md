# Technical Specification
# NBA Prediction Model v2 - Implementation Details

**Document Version**: 1.0
**Date**: January 13, 2026
**Based On**: requirements.md (PRD)
**Target**: Achieve 5%+ ROI with positive CLV, beating professional sharp benchmarks

---

## 1. Technical Context

### 1.1 Technology Stack

**Core ML Stack**:
- Python 3.9+
- scikit-learn 1.3.0+ (base models, preprocessing)
- XGBoost 2.0.0+ (gradient boosting)
- LightGBM 4.0.0+ (fast gradient boosting)
- CatBoost 1.2.0+ (categorical boosting)
- NumPy 1.24.0+ / Pandas 2.0.0+ (data processing)
- SciPy 1.11.0+ (statistical functions)

**Data & APIs**:
- Balldontlie API (GOAT tier - unlimited rate limits)
- The Odds API (100k subscription - historical odds, line movements)
- requests 2.31.0+ (HTTP client)
- aiohttp 3.9.0+ (async API calls for performance)

**Backend & Deployment**:
- FastAPI 0.104.0+ (REST API)
- Uvicorn 0.24.0+ (ASGI server)
- PostgreSQL 15+ (predictions storage, historical results)
- Railway (compute platform - scheduled jobs, API hosting)

**Utilities**:
- APScheduler 3.10.0+ (scheduled model retraining)
- python-dotenv 1.0.0+ (environment variables)
- Plotly 5.18.0+ (backtesting visualizations)

### 1.2 Current Codebase Structure

```
/
├── feature_engineering.py       # 4801 lines - Team/player features
├── model_trainer.py            # 4507 lines - Model classes (8 models)
├── train_complete_balldontlie.py # 5992 lines - Training pipeline
├── daily_predictions.py        # 1799 lines - Prediction generation
├── data_fetcher.py             # 2254 lines - API wrapper (NBA + Balldontlie)
├── balldontlie_api.py          # 1071 lines - Balldontlie-specific
├── stacked_model_v2.py         # Stacking ensemble (Level 0 + Level 1)
├── backtesting.py              # 2013 lines - Walk-forward validation
├── comprehensive_backtest.py   # 1598 lines - Full season replay
├── edge_quality.py             # 918 lines - Confidence scoring
├── calibration.py              # Probability calibration methods
│
├── models/                     # 462MB - Trained model artifacts
│   ├── moneyline_ensemble.pkl
│   ├── spread_ensemble.pkl
│   ├── player_{prop}_stacking.pkl (points, rebounds, assists, threes, pra)
│   └── calibration/            # Calibration metadata
│
├── continuous_learning/
│   ├── drift_detector.py
│   ├── incremental_trainer.py
│   └── model_registry.py
│
├── backend/
│   ├── api.py                  # FastAPI endpoints
│   └── schemas.py              # Pydantic models
│
└── backtest_results/
    └── backtest_results_2025.json
```

### 1.3 Current Performance Baseline

From `backtest_results_2025.json` (Oct 21 - Dec 12, 2025):
- **Overall R²**: 0.681
- **Overall RMSE**: 5.435
- **Points RMSE**: 6.757 (Target: <5.5)
- **Threes R²**: -0.568 (Target: >0.10) ← **Critical failure**
- **PRA RMSE**: 8.469 (Target: <7.0)

---

## 2. Implementation Approach

### 2.1 Architecture Philosophy

**Incremental Enhancement, Not Rewrite**:
- Preserve existing model classes (XGBoost, LightGBM, etc.)
- Add new feature modules alongside existing `feature_engineering.py`
- Upgrade meta-learner from weighted averaging to stacking
- Maintain temporal discipline (all changes must respect `game_date` parameter)

**Risk Mitigation**:
- Each phase includes backtest validation (no regression allowed)
- A/B testing: Deploy new features to 20% of predictions, compare performance
- Rollback plan: Keep old model artifacts in `models/backup/` directory

**Performance First**:
- Cache API responses (6-hour TTL for team stats, 24-hour for season averages)
- Parallel API calls using `asyncio` (Balldontlie supports concurrency)
- Lazy loading: Only compute expensive features (tracking data) if confidence is borderline

---

## 3. Source Code Structure Changes

### 3.1 New Modules to Create

#### **Module 1: `advanced_stats_v2.py`**
**Purpose**: Calculate Dean Oliver's Four Factors
**Location**: Root directory
**Size Estimate**: ~800 lines

**Public Functions**:
```python
def calculate_four_factors(
    team_id: int,
    game_date: datetime,
    window: str = "season"  # "season", "L5", "L10"
) -> Dict[str, float]:
    """
    Calculate Dean Oliver's Four Factors for a team.

    Returns:
        {
            "efg_pct": float,       # Effective FG% = (FG + 0.5*3PM) / FGA
            "tov_pct": float,       # Turnover Rate = TOV / (FGA + 0.44*FTA + TOV)
            "orb_pct": float,       # Off Rebound % = ORB / (ORB + Opp_DRB)
            "ft_rate": float,       # Free Throw Rate = FT / FGA
        }
    """

def calculate_four_factors_differential(
    home_id: int,
    away_id: int,
    game_date: datetime
) -> Dict[str, float]:
    """
    Calculate Four Factors differential (home - away).

    Returns 12 features:
        - efg_diff_season, efg_diff_L5, efg_diff_L10
        - tov_diff_season, tov_diff_L5, tov_diff_L10
        - orb_diff_season, orb_diff_L5, orb_diff_L10
        - ftr_diff_season, ftr_diff_L5, ftr_diff_L10
    """

def calculate_pace(
    team_id: int,
    game_date: datetime,
    window: str = "season"
) -> float:
    """
    Calculate possessions per 48 minutes.

    Formula:
        48 * (Possessions / Minutes)
        Possessions ≈ 0.5 * ((FGA + 0.4*FTA - 1.07*ORB_factor*(FGA-FG) + TOV) +
                             (Opp_FGA + 0.4*Opp_FTA - ... ))
    """

def adjust_for_pace(
    stat_value: float,
    team_pace: float,
    per_100: bool = True
) -> float:
    """Convert per-game stat to per-100 possessions."""
    if per_100:
        return stat_value * (100.0 / team_pace)
    return stat_value
```

**Data Sources**:
- Balldontlie `/stats` endpoint: Provides FG, FGA, 3PM, FT, FTA, ORB, DRB, TOV
- Use `fetch_team_statistics_before_date()` for temporal safety
- Cache results: 6-hour TTL for current season, permanent for completed seasons

**Integration Points**:
- Called by `feature_engineering.py::generate_game_features()`
- Adds 12 new columns to feature matrix
- Used by all models (moneyline, spread, props)

**Validation**:
- Unit test: Calculate Four Factors for Warriors on 2024-03-15, compare to Basketball-Reference.com
- Backtest: Train model with/without Four Factors, expect ≥1% RMSE reduction

---

#### **Module 2: `injury_tracker_v3.py`**
**Purpose**: Real-time injury detection and usage redistribution
**Location**: Root directory
**Size Estimate**: ~600 lines

**Public Functions**:
```python
def fetch_current_injuries(
    date: datetime = None
) -> List[Dict]:
    """
    Fetch all NBA injuries for a given date.

    Returns list of:
        {
            "player_id": int,
            "player_name": str,
            "team_id": int,
            "status": str,  # "OUT", "DOUBTFUL", "QUESTIONABLE", "GTD"
            "injury_type": str,  # "Knee", "Ankle", "Rest", etc.
            "last_update": datetime,
        }
    """

def is_player_available(
    player_id: int,
    game_date: datetime
) -> Tuple[bool, str]:
    """
    Check if player is available to play.

    Returns:
        (is_available: bool, status: str)
        - is_available: True if status in ["", "PROBABLE", "AVAILABLE"]
        - status: Current injury status
    """

def calculate_usage_redistribution(
    team_id: int,
    injured_player_id: int,
    game_date: datetime
) -> Dict[int, float]:
    """
    When star player is out, redistribute usage to teammates.

    Logic:
        1. Get injured player's usage rate
        2. Identify top 5 teammates by minutes played
        3. Distribute 70% of usage proportionally by role:
            - Primary scorer gets 30%
            - Secondary scorer gets 25%
            - Third option gets 15%
            - Remaining 2 players split 30%

    Returns:
        {player_id: additional_usage_rate, ...}
    """

def detect_star_player_out(
    team_id: int,
    game_date: datetime
) -> Tuple[bool, Optional[str]]:
    """
    Check if team's top-3 scorer is out.

    Returns:
        (star_out: bool, player_name: str or None)
    """

class InjuryCache:
    """
    In-memory cache for injury data (15-minute TTL during game days).
    Reduces API calls while maintaining freshness.
    """
    def __init__(self, ttl_minutes: int = 15):
        ...

    def get_injuries(self, date: datetime) -> Optional[List[Dict]]:
        ...

    def set_injuries(self, date: datetime, injuries: List[Dict]):
        ...
```

**Data Sources**:
1. **Primary**: Scrape NBA.com/injuries (free, 5-minute delay)
   - URL: `https://www.nba.com/stats/injuries`
   - Parse HTML table with BeautifulSoup
2. **Fallback**: ESPN injury report
   - URL: `https://www.espn.com/nba/injuries`
3. **Future Upgrade**: RotoWire API if free scraping proves unreliable

**Scraping Strategy**:
```python
# Pseudo-code for NBA.com scraper
def scrape_nba_injuries() -> List[Dict]:
    response = requests.get("https://www.nba.com/stats/injuries")
    soup = BeautifulSoup(response.text, 'html.parser')

    injuries = []
    for row in soup.find_all('tr', class_='injury-row'):
        player_name = row.find('td', class_='player').text
        team = row.find('td', class_='team').text
        status = row.find('td', class_='status').text  # "Out", "Questionable", etc.
        injury_type = row.find('td', class_='injury').text

        player_id = get_player_id(player_name)  # Lookup in database
        team_id = get_team_id(team)

        injuries.append({
            "player_id": player_id,
            "team_id": team_id,
            "status": status.upper(),
            "injury_type": injury_type,
            "last_update": datetime.now(),
        })

    return injuries
```

**Integration Points**:
- Called by `daily_predictions.py` BEFORE generating predictions (line ~500)
- Pre-flight check: If player status is "OUT", skip that player's prop predictions
- Add binary feature to team models: `star_player_out` (0 or 1)
- Enhance player prop features with `usage_boost` (e.g., +5% if star teammate is out)

**Error Handling**:
- If scraping fails, fallback to cached injuries from previous fetch (max 2 hours old)
- If no cache available, log warning and proceed (mark predictions as `DATA_INCOMPLETE`)
- Alert system: Send Slack/email if injury data is >30 minutes stale during game day

**Database Schema** (PostgreSQL for historical tracking):
```sql
CREATE TABLE injuries (
    id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    status VARCHAR(20),  -- OUT, DOUBTFUL, QUESTIONABLE, GTD
    injury_type VARCHAR(100),
    detected_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(player_id, game_date)
);

CREATE INDEX idx_injuries_date ON injuries(game_date);
CREATE INDEX idx_injuries_player ON injuries(player_id, game_date);
```

**Validation**:
- Manual audit: Review 100 games from Dec 2025, verify 0 DNP players in predictions
- Backtest: Compare predictions with/without injury detection, expect elimination of 161+ DNP errors

---

#### **Module 3: `stacking_meta_learner.py`**
**Purpose**: Replace weighted averaging with intelligent meta-learner
**Location**: Root directory
**Size Estimate**: ~500 lines

**Architecture**:
```
Level 0 (Base Models):
├── XGBoost
├── LightGBM
├── GradientBoosting
├── RandomForest
├── MLP
├── CatBoost
├── SVM
└── Logistic Regression

↓ (Out-of-fold predictions + context features)

Level 1 (Meta-Learner):
└── XGBoost Meta-Learner
    Input: [8 base predictions + 12 context features]
    Output: Final prediction
```

**Context Features** (12 features passed to meta-learner):
1. `days_rest_diff` - Rest advantage (home - away)
2. `pace_combined` - Expected game pace
3. `injury_count_home` - Number of injured players (home)
4. `injury_count_away` - Number of injured players (away)
5. `star_player_out_home` - Binary flag
6. `star_player_out_away` - Binary flag
7. `line_movement` - Closing line - Opening line
8. `reverse_line_movement` - Binary flag (line moved against public)
9. `prediction_variance` - Std dev of base model predictions (high = uncertainty)
10. `home_advantage` - Home court factor
11. `travel_distance_away` - Miles traveled by away team
12. `back_to_back_away` - Binary flag

**Public Functions**:
```python
class StackingMetaLearner:
    """
    Two-level stacking ensemble with context-aware meta-learner.
    """

    def __init__(
        self,
        base_models: List[BaseEstimator],
        meta_learner_type: str = "xgboost",  # "xgboost", "neural_net", "logistic"
        cv_folds: int = 5,
        time_series_split: bool = True
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of scikit-learn compatible models
            meta_learner_type: Type of meta-learner ("xgboost" recommended)
            cv_folds: Number of cross-validation folds for OOF predictions
            time_series_split: Use TimeSeriesSplit (True) or KFold (False)
        """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context_features: np.ndarray = None,
        sample_weights: np.ndarray = None
    ):
        """
        Train the stacking ensemble.

        Process:
            1. Split data into K folds (time-series aware)
            2. For each fold:
                - Train base models on K-1 folds
                - Generate predictions on held-out fold (OOF predictions)
            3. Combine all OOF predictions (no leakage)
            4. Train meta-learner on [OOF predictions + context features]
            5. Retrain base models on full dataset (for final predictions)

        Args:
            X: Training features (team/player stats)
            y: Target values
            context_features: Context for meta-learner (days_rest, pace, etc.)
            sample_weights: Time-decay weights (recent games weighted higher)
        """

    def predict(
        self,
        X: np.ndarray,
        context_features: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate predictions using stacked ensemble.

        Process:
            1. Get predictions from all base models
            2. Combine with context features
            3. Pass to meta-learner for final prediction
        """

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        context_features: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence intervals.

        Returns:
            (predictions, std_dev)
            - std_dev calculated from variance of base model predictions
        """

    def get_base_model_weights(self) -> Dict[str, float]:
        """
        Extract learned importance of each base model.

        For XGBoost meta-learner, use feature importance.
        Shows which models meta-learner trusts most.
        """
```

**Meta-Learner Options**:

**Option A: XGBoost (Recommended)**
```python
from xgboost import XGBRegressor, XGBClassifier

meta_learner = XGBRegressor(
    n_estimators=100,
    max_depth=3,  # Shallow to prevent overfitting
    learning_rate=0.05,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```
**Pros**: Fast, handles non-linear interactions, built-in feature importance
**Cons**: Can overfit if not regularized properly

**Option B: Neural Network (If XGBoost insufficient)**
```python
from sklearn.neural_network import MLPRegressor

meta_learner = MLPRegressor(
    hidden_layer_sizes=(32, 16),  # 2 hidden layers
    activation='relu',
    alpha=0.01,  # L2 regularization
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42
)
```
**Pros**: Can learn complex interactions between base models
**Cons**: Slower training, harder to interpret, requires more data

**Option C: Ridge Regression with Polynomial Features (Fallback)**
```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True)
meta_learner = Ridge(alpha=1.0)
```
**Pros**: Fast, simple, less prone to overfitting
**Cons**: Limited to quadratic interactions

**Decision Logic**:
1. Start with **XGBoost** (Option A)
2. If backtest shows <1% improvement → Try **Neural Network** (Option B)
3. If overfitting detected (train accuracy >> test accuracy) → Use **Ridge** (Option C)

**Integration Points**:
- Modify `model_trainer.py:3105` (`EnsembleMoneylineModel` class)
- Replace `_combine_predictions()` method with `StackingMetaLearner.predict()`
- Modify `train_complete_balldontlie.py` to pass context features during training

**Validation**:
- Backtest: Compare ROI with old weighted averaging vs new stacking
- Target: ≥2% accuracy improvement or ≥1.5 percentage points ROI increase
- Feature importance analysis: Verify meta-learner learns reasonable patterns (e.g., trusts XGBoost more in high-pace games)

---

#### **Module 4: `travel_fatigue.py`**
**Purpose**: Calculate travel distance, rest days, back-to-back detection
**Location**: Root directory
**Size Estimate**: ~400 lines

**Public Functions**:
```python
def calculate_travel_distance(
    from_team: str,  # "LAL", "BOS", etc.
    to_team: str,
    from_game_date: datetime = None  # If None, use team's home arena
) -> float:
    """
    Calculate travel distance using Haversine formula.

    Returns:
        Distance in miles
    """

def get_days_rest(
    team_id: int,
    game_date: datetime
) -> int:
    """
    Calculate days since team's last game.

    Returns:
        0 = back-to-back, 1 = played yesterday, 2+ = normal rest
    """

def detect_schedule_density(
    team_id: int,
    game_date: datetime
) -> Dict[str, Any]:
    """
    Detect fatigue scenarios: "3rd game in 4 nights", "4 in 5", etc.

    Returns:
        {
            "games_in_last_3_days": int,
            "games_in_last_5_days": int,
            "is_3_in_4": bool,
            "is_4_in_5": bool,
            "consecutive_road_games": int,
        }
    """

def calculate_altitude_adjustment(
    team_id: int,
    game_team_id: int,
    is_home: bool
) -> float:
    """
    Adjust for altitude (Denver at 5280ft, Utah at 4200ft).

    Logic:
        - If away team playing in Denver: -1.5 point adjustment
        - If away team playing in Utah: -1.0 point adjustment
        - If home team at altitude: +1.5 or +1.0 point adjustment

    Returns:
        Adjustment in points (positive = advantage, negative = disadvantage)
    """

def calculate_timezone_crossings(
    from_team: str,
    to_team: str
) -> int:
    """
    Count timezone crossings (affects circadian rhythm).

    Returns:
        Number of timezones crossed (0-3)
    """
```

**Data Source**:
- Arena data already exists in `feature_engineering.py:86-120` (`NBA_ARENA_DATA`)
- Contains: coordinates (lat/lon), altitude, timezone for all 30 teams

**Formulas**:

**Haversine Distance**:
```python
def haversine(lat1, lon1, lat2, lon2):
    R = 3959  # Earth radius in miles

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c
```

**Back-to-Back Impact** (from research):
- Back-to-back games: -2.1 points expected performance
- 3-in-4 nights: -1.5 points
- 4-in-5 nights: -2.5 points

**Integration Points**:
- Add to `feature_engineering.py::generate_game_features()`
- New columns: `days_rest_home`, `days_rest_away`, `travel_distance_away`, `altitude_adj_home`, `is_3_in_4_home`, `is_3_in_4_away`
- Total: 10 new features

**Validation**:
- Statistical test: Back-to-back games should show -2 ± 0.5 point differential in historical data
- Backtest: Verify Denver home games have +1.5 point adjustment

---

#### **Module 5: `betting_market_features.py`**
**Purpose**: Track line movements, RLM, consensus odds
**Location**: Root directory
**Size Estimate**: ~700 lines

**Public Functions**:
```python
def fetch_opening_line(
    game_id: int,
    market: str = "spreads"  # "spreads", "totals", "h2h"
) -> Optional[float]:
    """
    Fetch the first odds posted (typically 2-3 days before game).

    Uses The Odds API historical endpoint.

    Returns:
        Opening line value (e.g., -5.5 for spread)
    """

def fetch_closing_line(
    game_id: int,
    market: str = "spreads"
) -> Optional[float]:
    """
    Fetch final odds before tipoff (within 5 minutes of game start).
    """

def calculate_line_movement(
    game_id: int,
    market: str = "spreads"
) -> Dict[str, float]:
    """
    Calculate line movement from opening to closing.

    Returns:
        {
            "opening_line": float,
            "closing_line": float,
            "movement": float,  # closing - opening
            "movement_direction": str,  # "up", "down", "stable"
        }
    """

def detect_reverse_line_movement(
    game_id: int,
    market: str = "spreads"
) -> bool:
    """
    Detect RLM: Line moves opposite to public betting percentage.

    Example:
        - 70% of bets on Lakers -5
        - But line moves to Lakers -3.5
        - This is RLM (sharp money on opposite side)

    Data source: The Odds API may not provide bet percentages directly.
    Heuristic: If line moves >1.5 points without injury news, flag as potential RLM.

    Returns:
        True if RLM detected
    """

def calculate_consensus_odds(
    game_id: int,
    market: str = "spreads"
) -> float:
    """
    Average odds across all available sportsbooks.

    Uses The Odds API to fetch from 10+ books:
        DraftKings, FanDuel, BetMGM, Caesars, etc.

    Returns:
        Consensus line (mean of all books)
    """

def detect_steam_move(
    game_id: int,
    market: str = "spreads",
    lookback_minutes: int = 15
) -> bool:
    """
    Detect steam move: Rapid line movement (>1.5 points in <5 minutes).

    Indicates synchronized sharp action across books.

    Returns:
        True if steam move detected in last `lookback_minutes`
    """

class OddsTracker:
    """
    Background service to track odds every 5 minutes during game day.
    Stores time-series in PostgreSQL for historical analysis.
    """

    def __init__(self, update_interval_minutes: int = 5):
        ...

    async def fetch_and_store_odds(self):
        """
        Fetch current odds from The Odds API, store in database.
        Runs continuously during NBA season (Oct-Jun).
        """

    def get_odds_history(self, game_id: int) -> pd.DataFrame:
        """
        Retrieve historical odds for a game (for line movement analysis).

        Returns:
            DataFrame with columns: timestamp, spread, total, h2h, book_name
        """
```

**Database Schema** (PostgreSQL for odds time-series):
```sql
CREATE TABLE odds_history (
    id SERIAL PRIMARY KEY,
    game_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    book_name VARCHAR(50),
    market VARCHAR(20),  -- "spreads", "totals", "h2h"
    home_odds FLOAT,
    away_odds FLOAT,
    home_line FLOAT,     -- For spreads (e.g., -5.5)
    away_line FLOAT,
    total FLOAT,         -- For totals (e.g., 220.5)
    INDEX(game_id, timestamp),
    INDEX(timestamp)
);

-- For quick lookups
CREATE INDEX idx_odds_game_market ON odds_history(game_id, market, timestamp DESC);
```

**The Odds API Integration**:
```python
import requests

def fetch_odds_from_api(sport: str = "basketball_nba") -> List[Dict]:
    """
    Fetch current odds from The Odds API (100k subscription).

    Endpoint: https://api.the-odds-api.com/v4/sports/{sport}/odds
    """
    api_key = os.getenv("ODDS_API_KEY")

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals,h2h",
        "oddsFormat": "american",
        "bookmakers": "fanduel,draftkings,betmgm,caesars,pointsbet,betrivers,unibet,wynnbet,barstool,espnbet"
    }

    response = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
        params=params
    )

    return response.json()
```

**Integration Points**:
- Background job (APScheduler): Run `OddsTracker.fetch_and_store_odds()` every 5 minutes (8 AM - 11 PM during game days)
- Add to `feature_engineering.py::generate_game_features()`
- New columns: `opening_line`, `closing_line`, `line_movement`, `rlm_flag`, `consensus_odds`, `steam_move_flag`
- Total: 6 new features

**Validation**:
- Historical test: Verify RLM games show higher sharp bettor win rate (expected: 55-60% vs 50-52% baseline)
- Backtest: Model with market features should show improved Closing Line Value (CLV > 0)

---

### 3.2 Modifications to Existing Modules

#### **Modification 1: `feature_engineering.py`**
**Line**: ~2800 (in `generate_game_features()` function)

**Current Code** (approximate):
```python
def generate_game_features(home_id, away_id, game_date=None):
    features = {}

    # Existing features: win_pct, pts_avg, etc.
    home_stats = fetch_team_statistics_before_date(home_id, game_date)
    away_stats = fetch_team_statistics_before_date(away_id, game_date)

    features['win_pct_diff'] = home_stats['win_pct'] - away_stats['win_pct']
    features['pts_avg_diff'] = home_stats['pts_avg'] - away_stats['pts_avg']
    # ... more features

    return features
```

**New Code** (additions):
```python
from advanced_stats_v2 import calculate_four_factors_differential, calculate_pace
from travel_fatigue import get_days_rest, calculate_travel_distance, calculate_altitude_adjustment, detect_schedule_density
from betting_market_features import calculate_line_movement, detect_reverse_line_movement, calculate_consensus_odds

def generate_game_features(home_id, away_id, game_date=None):
    features = {}

    # Existing features (unchanged)
    home_stats = fetch_team_statistics_before_date(home_id, game_date)
    away_stats = fetch_team_statistics_before_date(away_id, game_date)
    features['win_pct_diff'] = home_stats['win_pct'] - away_stats['win_pct']
    # ... more existing features

    # === NEW: Four Factors (12 features) ===
    four_factors = calculate_four_factors_differential(home_id, away_id, game_date)
    features.update(four_factors)  # efg_diff_season, efg_diff_L5, etc.

    # === NEW: Pace Features (3 features) ===
    home_pace = calculate_pace(home_id, game_date)
    away_pace = calculate_pace(away_id, game_date)
    features['pace_home'] = home_pace
    features['pace_away'] = away_pace
    features['pace_combined'] = (home_pace + away_pace) / 2.0

    # === NEW: Travel & Fatigue (10 features) ===
    features['days_rest_home'] = get_days_rest(home_id, game_date)
    features['days_rest_away'] = get_days_rest(away_id, game_date)
    features['days_rest_diff'] = features['days_rest_home'] - features['days_rest_away']

    # Travel distance for away team (home team = 0)
    features['travel_distance_away'] = calculate_travel_distance(
        from_team=get_team_abbr(away_id),  # Need last game location
        to_team=get_team_abbr(home_id),
        from_game_date=game_date - timedelta(days=features['days_rest_away'])
    )

    features['altitude_adj_home'] = calculate_altitude_adjustment(home_id, away_id, is_home=True)

    home_density = detect_schedule_density(home_id, game_date)
    away_density = detect_schedule_density(away_id, game_date)
    features['is_3_in_4_home'] = int(home_density['is_3_in_4'])
    features['is_3_in_4_away'] = int(away_density['is_3_in_4'])
    features['consecutive_road_games_away'] = away_density['consecutive_road_games']

    # === NEW: Injury Features (4 features) ===
    from injury_tracker_v3 import detect_star_player_out
    star_out_home, _ = detect_star_player_out(home_id, game_date)
    star_out_away, _ = detect_star_player_out(away_id, game_date)
    features['star_player_out_home'] = int(star_out_home)
    features['star_player_out_away'] = int(star_out_away)

    injury_count_home = len(fetch_current_injuries(game_date, team_id=home_id))
    injury_count_away = len(fetch_current_injuries(game_date, team_id=away_id))
    features['injury_count_home'] = injury_count_home
    features['injury_count_away'] = injury_count_away

    # === NEW: Betting Market Features (6 features) ===
    # Only available for live predictions (not historical backtesting without odds history)
    if game_date is None or game_date >= datetime.now():
        game_id = get_game_id(home_id, away_id, game_date)  # Helper function

        line_data = calculate_line_movement(game_id, market="spreads")
        features['opening_line'] = line_data.get('opening_line', 0.0)
        features['closing_line'] = line_data.get('closing_line', 0.0)
        features['line_movement'] = line_data.get('movement', 0.0)
        features['rlm_flag'] = int(detect_reverse_line_movement(game_id))
        features['consensus_odds'] = calculate_consensus_odds(game_id, market="spreads")
        features['steam_move_flag'] = int(detect_steam_move(game_id))
    else:
        # Historical games: Odds data may not be available
        # Use default values (or fetch from stored odds_history table)
        features['opening_line'] = 0.0
        features['closing_line'] = 0.0
        features['line_movement'] = 0.0
        features['rlm_flag'] = 0
        features['consensus_odds'] = 0.0
        features['steam_move_flag'] = 0

    return features
```

**Total New Features**: 41 (12 Four Factors + 3 Pace + 10 Travel/Fatigue + 4 Injury + 6 Market + 6 existing)

**Impact**: Feature count increases from ~35 to ~76 (rough estimate)

---

#### **Modification 2: `model_trainer.py` - Upgrade Ensemble**
**Class**: `EnsembleMoneylineModel` (line ~3105)

**Current Code** (simplified):
```python
class EnsembleMoneylineModel(BaseModelTrainer):
    def __init__(self):
        self.models = {
            'xgb': XGBClassifier(...),
            'lgb': LGBMClassifier(...),
            'gb': GradientBoostingClassifier(...),
            # ... 5 more models
        }
        self.weights = None  # Inverse-RMSE weights

    def train(self, X_train, y_train):
        # Train each model independently
        for name, model in self.models.items():
            model.fit(X_train, y_train)

        # Calculate inverse-RMSE weights
        self.weights = self._calculate_weights(X_val, y_val)

    def predict(self, X):
        # Get predictions from all models
        predictions = [model.predict_proba(X)[:, 1] for model in self.models.values()]

        # Weighted average
        final_pred = np.average(predictions, axis=0, weights=self.weights)
        return final_pred
```

**New Code** (with stacking):
```python
from stacking_meta_learner import StackingMetaLearner

class EnsembleMoneylineModel(BaseModelTrainer):
    def __init__(self, use_stacking=True):
        self.base_models = [
            ('xgb', XGBClassifier(...)),
            ('lgb', LGBMClassifier(...)),
            ('gb', GradientBoostingClassifier(...)),
            ('rf', RandomForestClassifier(...)),
            ('mlp', MLPClassifier(...)),
            ('catboost', CatBoostClassifier(...)),
            ('svm', SVC(probability=True, ...)),
            ('lr', LogisticRegression(...)),
        ]

        self.use_stacking = use_stacking
        if use_stacking:
            self.ensemble = StackingMetaLearner(
                base_models=[model for name, model in self.base_models],
                meta_learner_type='xgboost',
                cv_folds=5,
                time_series_split=True
            )
        else:
            # Fallback to old weighted averaging (for A/B testing)
            self.weights = None

    def train(self, X_train, y_train, context_features=None, sample_weights=None):
        if self.use_stacking:
            # Train with stacking ensemble
            self.ensemble.fit(
                X_train,
                y_train,
                context_features=context_features,  # days_rest, pace, injury_count, etc.
                sample_weights=sample_weights
            )
        else:
            # Old weighted averaging approach (unchanged)
            for name, model in self.base_models:
                model.fit(X_train, y_train)
            self.weights = self._calculate_weights(X_val, y_val)

    def predict(self, X, context_features=None):
        if self.use_stacking:
            return self.ensemble.predict(X, context_features=context_features)
        else:
            # Old weighted averaging
            predictions = [model.predict_proba(X)[:, 1] for _, model in self.base_models]
            return np.average(predictions, axis=0, weights=self.weights)

    def predict_with_confidence(self, X, context_features=None):
        """
        Generate predictions with confidence score.

        Confidence = inverse of prediction variance across base models.
        """
        if self.use_stacking:
            preds, std_dev = self.ensemble.predict_with_uncertainty(X, context_features)
            confidence = 100 * (1 - np.minimum(std_dev / np.maximum(preds, 0.01), 1.0))
            return preds, confidence
        else:
            # For weighted averaging, calculate variance manually
            predictions = np.array([model.predict_proba(X)[:, 1] for _, model in self.base_models])
            preds = np.average(predictions, axis=0, weights=self.weights)
            std_dev = np.std(predictions, axis=0)
            confidence = 100 * (1 - np.minimum(std_dev / np.maximum(preds, 0.01), 1.0))
            return preds, confidence
```

**Integration**:
- Add `use_stacking=True` parameter when instantiating `EnsembleMoneylineModel` in `train_complete_balldontlie.py`
- Pass `context_features` during training (extract from main feature matrix)
- Repeat for `SpreadModel` and player prop models

---

#### **Modification 3: `train_complete_balldontlie.py` - Training Pipeline**
**Location**: Main training loop (line ~4000+)

**Current Code** (simplified):
```python
def train_all_models():
    # Fetch historical games
    games = fetch_historical_games(start_date, end_date)

    # Generate features for each game
    X_train = []
    y_train = []
    for game in games:
        features = generate_game_features(game['home_id'], game['away_id'], game['date'])
        X_train.append(features)
        y_train.append(game['home_won'])  # Binary target

    # Train moneyline model
    moneyline_model = EnsembleMoneylineModel()
    moneyline_model.train(X_train, y_train)

    # Save model
    joblib.dump(moneyline_model, 'models/moneyline_ensemble.pkl')
```

**New Code** (with context features):
```python
def train_all_models():
    # Fetch historical games
    games = fetch_historical_games(start_date, end_date)

    # Generate features for each game
    X_train = []
    y_train = []
    context_train = []  # NEW: Context features for meta-learner
    sample_weights = []  # Time-decay weights

    for game in games:
        # Generate full feature set (now includes Four Factors, travel, etc.)
        features = generate_game_features(game['home_id'], game['away_id'], game['date'])
        X_train.append(features)
        y_train.append(game['home_won'])

        # === NEW: Extract context features for meta-learner ===
        context = [
            features['days_rest_diff'],
            features['pace_combined'],
            features['injury_count_home'],
            features['injury_count_away'],
            features['star_player_out_home'],
            features['star_player_out_away'],
            features['line_movement'],
            features['rlm_flag'],
            0.0,  # prediction_variance (filled during training)
            features.get('home_advantage', 3.0),  # Default home advantage
            features['travel_distance_away'],
            int(features['days_rest_away'] == 0),  # back_to_back_away
        ]
        context_train.append(context)

        # Time-decay weights (180-day half-life)
        days_ago = (datetime.now() - game['date']).days
        weight = 0.5 ** (days_ago / 180.0)
        sample_weights.append(weight)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    context_train = np.array(context_train)
    sample_weights = np.array(sample_weights)

    # === Train moneyline model with stacking ===
    moneyline_model = EnsembleMoneylineModel(use_stacking=True)
    moneyline_model.train(
        X_train,
        y_train,
        context_features=context_train,
        sample_weights=sample_weights
    )

    # Save model
    joblib.dump(moneyline_model, 'models/moneyline_ensemble_v2.pkl')

    # === Validation: Compare with old model ===
    old_model = joblib.load('models/moneyline_ensemble.pkl')
    new_accuracy = evaluate_model(moneyline_model, X_test, y_test, context_test)
    old_accuracy = evaluate_model(old_model, X_test, y_test)

    print(f"Old Model Accuracy: {old_accuracy:.4f}")
    print(f"New Model Accuracy: {new_accuracy:.4f}")
    print(f"Improvement: {(new_accuracy - old_accuracy)*100:.2f}%")

    # Only replace old model if new model is better
    if new_accuracy > old_accuracy:
        print("✅ New model is better! Deploying...")
        os.rename('models/moneyline_ensemble.pkl', 'models/backup/moneyline_ensemble_old.pkl')
        os.rename('models/moneyline_ensemble_v2.pkl', 'models/moneyline_ensemble.pkl')
    else:
        print("⚠️ New model is not better. Keeping old model.")
```

---

#### **Modification 4: `daily_predictions.py` - Add Injury Check**
**Location**: Before generating predictions (line ~500)

**New Code** (add at start of prediction loop):
```python
from injury_tracker_v3 import fetch_current_injuries, is_player_available

def generate_daily_predictions(date=None):
    if date is None:
        date = datetime.now()

    # === NEW: Fetch injuries BEFORE predictions ===
    print(f"Fetching injury data for {date.strftime('%Y-%m-%d')}...")
    injuries = fetch_current_injuries(date)
    print(f"Found {len(injuries)} injured players")

    # Build lookup dict for fast access
    injury_lookup = {inj['player_id']: inj['status'] for inj in injuries}

    # Fetch today's games
    games = fetch_todays_games(date)

    all_predictions = []
    for game in games:
        home_id = game['home_team_id']
        away_id = game['away_team_id']

        # Generate team features (moneyline, spread)
        team_features = generate_game_features(home_id, away_id, date)

        # Predict moneyline
        ml_model = load_model('moneyline_ensemble.pkl')
        ml_pred, ml_confidence = ml_model.predict_with_confidence([team_features])

        # Generate player prop predictions
        home_roster = fetch_team_roster(home_id)
        away_roster = fetch_team_roster(away_id)

        for player in home_roster + away_roster:
            player_id = player['id']

            # === NEW: Check injury status ===
            if player_id in injury_lookup:
                status = injury_lookup[player_id]
                if status in ["OUT", "DOUBTFUL"]:
                    print(f"⚠️ Skipping {player['name']} - Status: {status}")
                    continue  # Don't predict for unavailable players
                elif status in ["QUESTIONABLE", "GTD"]:
                    print(f"⚠️ Warning: {player['name']} is {status} - Low confidence prediction")
                    # Continue but flag prediction
                    uncertainty_flag = "HIGH_UNCERTAINTY"
                else:
                    uncertainty_flag = None
            else:
                uncertainty_flag = None

            # Generate player prop features
            prop_features = generate_points_prop_features(player_id, home_id, away_id, date)

            # Predict
            points_model = load_model('player_points_stacking.pkl')
            pred = points_model.predict([prop_features])[0]

            all_predictions.append({
                'player_name': player['name'],
                'team': player['team'],
                'prop': 'points',
                'prediction': pred,
                'confidence': ml_confidence[0],
                'uncertainty_flag': uncertainty_flag,
            })

    # Save predictions to CSV
    df = pd.DataFrame(all_predictions)
    df.to_csv(f'predictions/predictions_{date.strftime("%Y%m%d")}.csv', index=False)

    return df
```

---

## 4. Data Model / API / Interface Changes

### 4.1 Feature Matrix Schema

**Current**: ~35 features per game
**New**: ~76 features per game

**New Feature Groups**:
1. **Four Factors** (12): efg_diff_season, efg_diff_L5, efg_diff_L10, tov_diff_season, ..., ftr_diff_L10
2. **Pace** (3): pace_home, pace_away, pace_combined
3. **Travel/Fatigue** (10): days_rest_home, days_rest_away, days_rest_diff, travel_distance_away, altitude_adj_home, is_3_in_4_home, is_3_in_4_away, consecutive_road_games_away, timezone_crossings, ...
4. **Injury** (4): star_player_out_home, star_player_out_away, injury_count_home, injury_count_away
5. **Betting Market** (6): opening_line, closing_line, line_movement, rlm_flag, consensus_odds, steam_move_flag

### 4.2 Prediction Output Schema

**Current CSV Output**:
```
game_id, home_team, away_team, prediction, confidence
```

**New CSV Output** (enhanced):
```csv
game_id,game_time,home_team,away_team,prediction_type,predicted_value,confidence_score,edge_quality_tier,pred_low,pred_median,pred_high,key_injuries,days_rest_diff,pace_projection,line_movement,bet_recommendation,suggested_bet_size,uncertainty_flags
123,2026-01-15 19:00,LAL,BOS,moneyline,0.65,85,Elite,0.60,0.65,0.70,"",2,101.5,+1.5,BET,2.5%,""
123,2026-01-15 19:00,LAL,BOS,spread,-4.5,82,Strong,-6.2,-4.5,-2.8,"",2,101.5,+1.5,BET,2.0%,""
123,2026-01-15 19:00,LAL,BOS,player_points_LeBron,26.8,78,Strong,22.5,26.8,31.2,"",2,101.5,+1.5,BET,1.5%,""
```

**New Columns**:
- `pred_low`, `pred_median`, `pred_high`: 10th/50th/90th percentile predictions
- `key_injuries`: Comma-separated list of injured star players
- `days_rest_diff`: Rest advantage (home - away)
- `pace_projection`: Expected game pace
- `line_movement`: Closing line - Opening line
- `bet_recommendation`: "BET", "MONITOR", "AVOID"
- `suggested_bet_size`: Kelly % (e.g., "2.5%" = bet 2.5% of bankroll)
- `uncertainty_flags`: "HIGH_UNCERTAINTY", "DATA_INCOMPLETE", etc.

### 4.3 Database Schema Additions

**Table: `odds_history`** (new)
```sql
CREATE TABLE odds_history (
    id SERIAL PRIMARY KEY,
    game_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    book_name VARCHAR(50),
    market VARCHAR(20),
    home_odds FLOAT,
    away_odds FLOAT,
    home_line FLOAT,
    away_line FLOAT,
    total FLOAT
);
CREATE INDEX idx_odds_game ON odds_history(game_id, timestamp DESC);
```

**Table: `injuries`** (new)
```sql
CREATE TABLE injuries (
    id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    status VARCHAR(20),
    injury_type VARCHAR(100),
    detected_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(player_id, game_date)
);
CREATE INDEX idx_injuries_date ON injuries(game_date);
```

**Table: `predictions_history`** (existing, add columns)
```sql
ALTER TABLE predictions_history
ADD COLUMN confidence_score FLOAT,
ADD COLUMN edge_quality_tier VARCHAR(20),
ADD COLUMN pred_low FLOAT,
ADD COLUMN pred_high FLOAT,
ADD COLUMN uncertainty_flags VARCHAR(100);
```

### 4.4 API Endpoints (FastAPI)

**New Endpoint: `/api/predictions/{date}`**
```python
@app.get("/api/predictions/{date}")
async def get_predictions(date: str):
    """
    Fetch predictions for a specific date.

    Args:
        date: YYYY-MM-DD format

    Returns:
        JSON array of predictions with confidence, edge quality, etc.
    """
    predictions = load_predictions_from_csv(f"predictions/predictions_{date}.csv")
    return predictions.to_dict(orient='records')
```

**New Endpoint: `/api/injuries/{date}`**
```python
@app.get("/api/injuries/{date}")
async def get_injuries(date: str = None):
    """
    Fetch current injury report.

    Returns:
        JSON array of injured players with status
    """
    from injury_tracker_v3 import fetch_current_injuries

    if date:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
    else:
        date_obj = datetime.now()

    injuries = fetch_current_injuries(date_obj)
    return injuries
```

**New Endpoint: `/api/line-movement/{game_id}`**
```python
@app.get("/api/line-movement/{game_id}")
async def get_line_movement(game_id: int):
    """
    Fetch line movement history for a game.

    Returns:
        JSON with opening_line, closing_line, movement, RLM flag
    """
    from betting_market_features import calculate_line_movement

    line_data = calculate_line_movement(game_id, market="spreads")
    return line_data
```

---

## 5. Delivery Phases (Incremental, Testable Milestones)

### Phase 1: Foundation (Weeks 1-2) - Critical Fixes

**Goal**: Fix critical issues preventing accurate predictions

**Tasks**:
1. ✅ Create `injury_tracker_v3.py` module
   - Build NBA.com injury scraper
   - Implement `fetch_current_injuries()` and `is_player_available()`
   - Add PostgreSQL `injuries` table
   - **Test**: Manual audit of 100 recent games, verify 0 DNP players missed

2. ✅ Create `advanced_stats_v2.py` module
   - Implement Four Factors calculations
   - Add rolling averages (season, L5, L10)
   - **Test**: Compare calculations to Basketball-Reference.com for 10 random teams

3. ✅ Integrate Four Factors into `feature_engineering.py`
   - Modify `generate_game_features()` to call `calculate_four_factors_differential()`
   - **Test**: Verify 12 new columns added to feature matrix

4. ✅ Create `stacking_meta_learner.py` module
   - Implement `StackingMetaLearner` class with XGBoost meta-learner
   - Add OOF (out-of-fold) prediction logic
   - **Test**: Unit test with synthetic data, verify no leakage

5. ✅ Upgrade `model_trainer.py` ensemble classes
   - Modify `EnsembleMoneylineModel` to use `StackingMetaLearner`
   - Add `use_stacking` parameter for A/B testing
   - **Test**: Train on 100 games, verify model trains without errors

6. ✅ Modify `train_complete_balldontlie.py` training pipeline
   - Extract context features for meta-learner
   - Pass to `StackingMetaLearner.fit()`
   - **Test**: Full training run on 2 seasons, compare accuracy to baseline

7. ✅ Add injury check to `daily_predictions.py`
   - Call `fetch_current_injuries()` before predictions
   - Skip players with "OUT" or "DOUBTFUL" status
   - **Test**: Generate predictions for today, verify no OUT players predicted

8. ✅ **Validation: Run comprehensive backtest**
   - Backtest 2024-25 season (Oct - Apr)
   - Compare metrics to baseline (backtest_results_2025.json)
   - **Success Criteria**:
     - Overall RMSE: < 5.3 (from 5.4)
     - Points RMSE: < 6.5 (from 6.8)
     - Threes R²: > -0.4 (from -0.57)
     - Zero DNP errors (from 161)

**Deliverables**:
- 3 new Python modules (injury_tracker_v3.py, advanced_stats_v2.py, stacking_meta_learner.py)
- Modified feature_engineering.py, model_trainer.py, train_complete_balldontlie.py, daily_predictions.py
- Backtest report showing improvement over baseline
- PostgreSQL `injuries` table populated

**Risk**: If Four Factors don't improve accuracy by ≥1%, investigate feature importance to diagnose

---

### Phase 2: Enhancement (Weeks 3-4) - Advanced Features

**Goal**: Add travel, betting market features, confidence scoring

**Tasks**:
1. ✅ Create `travel_fatigue.py` module
   - Implement distance calculations (Haversine formula)
   - Add `get_days_rest()`, `detect_schedule_density()`, `calculate_altitude_adjustment()`
   - **Test**: Verify Denver home games show +1.5 pt adjustment

2. ✅ Integrate travel features into `feature_engineering.py`
   - Add 10 new columns (days_rest, travel_distance, altitude, etc.)
   - **Test**: Backtest shows back-to-back games correlate with -2 pts

3. ✅ Create `betting_market_features.py` module
   - Integrate The Odds API for opening/closing lines
   - Implement `calculate_line_movement()`, `detect_reverse_line_movement()`
   - Add PostgreSQL `odds_history` table
   - **Test**: Verify RLM detection on 50 historical games

4. ✅ Set up `OddsTracker` background job
   - APScheduler job to fetch odds every 5 minutes
   - Store in PostgreSQL `odds_history` table
   - **Test**: Run for 1 day, verify odds are captured

5. ✅ Integrate market features into `feature_engineering.py`
   - Add 6 new columns (opening_line, line_movement, rlm_flag, etc.)
   - **Test**: Verify features populated for live games

6. ✅ Implement confidence scoring in `model_trainer.py`
   - Modify `predict_with_confidence()` to calculate variance-based confidence
   - **Test**: High-agreement predictions should have confidence > 80%

7. ✅ Add confidence and edge quality to `daily_predictions.py` output
   - Calculate edge quality tiers (Elite, Strong, Moderate, Weak, Avoid)
   - Add to CSV output
   - **Test**: Generate predictions, verify confidence and tier columns exist

8. ✅ **Validation: Run comprehensive backtest with filters**
   - Backtest 2024-25 season
   - Filter to only bet on Elite + Strong tiers
   - **Success Criteria**:
     - Overall RMSE: < 5.0 (from 5.3)
     - ROI (Elite tier): > 5%
     - Positive CLV (beat closing line on average)

**Deliverables**:
- 2 new Python modules (travel_fatigue.py, betting_market_features.py)
- PostgreSQL `odds_history` table populated
- APScheduler background job for odds tracking
- Enhanced prediction CSV with confidence and edge quality
- Backtest report showing ROI > 3%

---

### Phase 3: Optimization (Weeks 5-6) - Fine-Tuning

**Goal**: Integrate player impact metrics, quantile regression, risk management

**Tasks**:
1. ✅ Create `player_impact_fetcher.py` module
   - Scrape or fetch DARKO DPM, ESPN EPM, or FiveThirtyEight RAPTOR
   - Cache daily (24-hour TTL)
   - **Test**: Verify impact metrics fetched for all starters

2. ✅ Integrate player impact into player prop features
   - Modify `generate_points_prop_features()` to include impact metric
   - **Test**: Backtest player props, expect ≥5% RMSE reduction

3. ✅ Implement quantile regression in `model_trainer.py`
   - Add `QuantilePropModel` for 10th/50th/90th percentiles
   - Train for all prop types
   - **Test**: Empirical coverage matches theoretical (10% below low, 10% above high)

4. ✅ Add prediction bands to `daily_predictions.py` output
   - Columns: pred_low, pred_median, pred_high
   - **Test**: Wide bands (>8 pts) should correlate with low confidence

5. ✅ Create `risk_management.py` module
   - Implement Kelly criterion bet sizing
   - Add stop-loss rules (daily, weekly, max drawdown)
   - **Test**: Backtest with Kelly vs flat betting, verify higher Sharpe ratio

6. ✅ Add bet sizing to `daily_predictions.py` output
   - Calculate suggested_bet_size using Kelly criterion
   - Apply confidence-based adjustments (Elite = 1.0x Kelly, Strong = 0.5x, etc.)
   - **Test**: Verify bet sizes sum to <20% of bankroll per day

7. ✅ **Validation: Full end-to-end backtest**
   - Backtest 2 seasons (2023-24, 2024-25)
   - Apply Kelly bet sizing with stop-loss rules
   - **Success Criteria**:
     - Overall RMSE: < 4.8
     - Points RMSE: < 5.5
     - Threes R²: > 0.10
     - ROI (All bets): > 3%
     - ROI (Elite tier): > 7%
     - Sharpe ratio: > 1.5
     - Max drawdown: < 15%

**Deliverables**:
- 2 new Python modules (player_impact_fetcher.py, risk_management.py)
- Quantile regression models for all prop types
- Enhanced prediction CSV with bet sizing and prediction bands
- Comprehensive backtest report (2 seasons, ROI, Sharpe, drawdown)

---

### Phase 4: Productionization (Weeks 7-8) - Deployment

**Goal**: Deploy to production, set up monitoring, start live betting

**Tasks**:
1. ✅ Optimize prediction generation speed
   - Profile `daily_predictions.py`, identify bottlenecks
   - Add caching for team statistics (6-hour TTL)
   - Parallelize API calls with `asyncio`
   - **Test**: Generate all predictions for 15 games in < 5 minutes

2. ✅ Set up automated retraining pipeline
   - Railway scheduled job: Full retrain every 14 days
   - Incremental update every 3 days (meta-learner only)
   - **Test**: Trigger manual retrain, verify completes in < 4 hours

3. ✅ Implement drift detection in `continuous_learning/drift_detector.py`
   - Monitor RMSE daily
   - Alert if RMSE increases >10% for 3 consecutive days
   - **Test**: Simulate drift with synthetic data, verify alert triggers

4. ✅ Create HTML backtesting report
   - Use Jinja2 templates for HTML generation
   - Add Plotly charts (ROI curve, calibration plot, reliability diagram)
   - **Test**: Generate report for 2024-25 season, verify visualizations

5. ✅ Set up FastAPI endpoints
   - `/api/predictions/{date}` - Fetch predictions
   - `/api/injuries/{date}` - Fetch injury report
   - `/api/line-movement/{game_id}` - Fetch line movement
   - **Test**: Hit each endpoint, verify JSON response

6. ✅ Deploy to Railway
   - Push code to GitHub
   - Configure Railway scheduled jobs (prediction generation, retraining)
   - Set up PostgreSQL database
   - **Test**: Generate predictions on Railway, verify output

7. ✅ **Paper Trading (Week 8)**
   - Track hypothetical bets for 7 days
   - Compare predictions to actual outcomes
   - Calculate ROI, Sharpe, max drawdown
   - **Success Criteria**: ROI > 3%, confidence matches actual win rate

8. ✅ **Go-Live (End of Week 8)**
   - Start live betting with 10% of intended bankroll (e.g., $500)
   - Strict stop-loss rules (3% daily, 8% weekly)
   - Daily monitoring of bankroll, ROI, CLV
   - **Success Criteria**: Positive ROI after 30 bets, positive CLV

**Deliverables**:
- Optimized prediction pipeline (< 5 min for all games)
- Railway deployment with scheduled jobs
- HTML backtesting reports
- FastAPI endpoints for predictions, injuries, line movement
- 7-day paper trading results
- Live betting dashboard (optional)

---

## 6. Verification Approach

### 6.1 Unit Tests

**Test Coverage Target**: 80%+ for critical modules

**Key Test Cases**:

1. **`advanced_stats_v2.py`**:
   - `test_four_factors_calculation()`: Compare to Basketball-Reference for 10 teams
   - `test_pace_calculation()`: Verify formula correctness
   - `test_temporal_discipline()`: Ensure no future data used

2. **`injury_tracker_v3.py`**:
   - `test_scraper()`: Verify scraping returns valid data structure
   - `test_dnp_detection()`: Mock injured player, verify `is_player_available()` returns False
   - `test_usage_redistribution()`: Verify usage sums to 100%

3. **`stacking_meta_learner.py`**:
   - `test_oof_predictions()`: Verify no leakage (OOF predictions don't use holdout set for training)
   - `test_meta_learner_training()`: Train on synthetic data, verify convergence
   - `test_confidence_calculation()`: High variance → low confidence

4. **`travel_fatigue.py`**:
   - `test_haversine_distance()`: Compare to known distances (LAL → BOS = ~2600 miles)
   - `test_back_to_back_detection()`: Verify correct identification
   - `test_altitude_adjustment()`: Denver home games should show +1.5 pts

5. **`betting_market_features.py`**:
   - `test_line_movement_calculation()`: Mock opening/closing lines, verify delta
   - `test_rlm_detection()`: Mock scenario with line moving opposite to public
   - `test_odds_api_integration()`: Hit The Odds API (in staging environment), verify response

**Run Tests**:
```bash
pytest tests/ --cov=. --cov-report=html
```

### 6.2 Integration Tests

**Test Scenarios**:

1. **End-to-End Prediction Generation**:
   - Generate predictions for a historical date (e.g., 2025-11-15)
   - Verify output CSV contains all expected columns
   - Verify no DNP players in output
   - Assert: RMSE on that date < 6.0

2. **Temporal Leakage Check**:
   - Select 100 random historical games
   - For each game, generate features using `game_date` parameter
   - Verify no features use data from after `game_date`
   - Use data inspection: Check max date in fetched statistics

3. **API Integration**:
   - Hit `/api/predictions/2026-01-13` endpoint
   - Verify JSON response contains predictions with confidence scores
   - Hit `/api/injuries/2026-01-13`
   - Verify JSON response contains injured players

4. **Background Jobs**:
   - Trigger `OddsTracker.fetch_and_store_odds()` manually
   - Verify odds stored in PostgreSQL `odds_history` table
   - Check: All games for today have ≥5 sportsbook entries

### 6.3 Backtesting Validation

**Backtest Protocol**:

1. **Historical Replay**:
   - Use `comprehensive_backtest.py` to replay 2024-25 season
   - Walk-forward validation: Train on games before date, test on games after
   - No lookahead bias (enforce temporal discipline)

2. **Metrics to Track**:
   - **Accuracy Metrics**: RMSE, MAE, R², Bias for each prop type
   - **Betting Metrics**: ROI, Win Rate, Sharpe Ratio, Max Drawdown
   - **Calibration**: Brier Score, Expected Calibration Error (ECE)
   - **Market Metrics**: Closing Line Value (CLV), Reverse Line Movement Win Rate

3. **Sanity Checks**:
   - ROI > 15% → Flag as potential leakage
   - Win Rate > 60% → Flag as unrealistic
   - Sharpe Ratio > 3.0 → Investigate
   - If any flag triggers, audit for temporal leakage

4. **Comparison to Baseline**:
   - Load `backtest_results_2025.json` (baseline)
   - Run backtest with new model
   - Calculate delta for each metric
   - **Success Criteria**: New model shows ≥1% RMSE improvement OR ≥1.5 pp ROI increase

**Backtest Report Format** (JSON):
```json
{
  "backtest_date": "2026-01-13",
  "season": "2024-25",
  "games_processed": 1230,
  "start_date": "2024-10-22",
  "end_date": "2025-04-13",
  "model_version": "v2.0_stacking",
  "metrics": {
    "overall": {
      "count": 45000,
      "rmse": 4.85,
      "mae": 3.21,
      "r2": 0.758,
      "bias": -0.15
    },
    "points": {"rmse": 5.42, "r2": 0.512, ...},
    "threes": {"rmse": 1.52, "r2": 0.125, ...},
    ...
  },
  "betting_results": {
    "total_bets": 2500,
    "wins": 1375,
    "losses": 1125,
    "win_rate": 0.55,
    "roi": 0.047,
    "sharpe_ratio": 1.68,
    "max_drawdown": 0.128,
    "closing_line_value": 0.025
  },
  "by_tier": {
    "elite": {"bets": 450, "roi": 0.082, "win_rate": 0.592},
    "strong": {"bets": 850, "roi": 0.051, "win_rate": 0.561},
    ...
  }
}
```

### 6.4 Live Validation (Paper Trading)

**Week 8 Protocol**:

1. **Daily Predictions**:
   - Generate predictions each morning (9 AM)
   - Save to `predictions/predictions_{date}.csv`
   - Do NOT place real bets

2. **Outcome Tracking**:
   - After games finish, scrape actual results
   - Compare to predictions, calculate RMSE
   - Track hypothetical bankroll (starting $5,000)

3. **Daily Review**:
   - Which predictions were accurate?
   - Which predictions missed badly? (error > 10 pts)
   - Were injured players detected correctly?
   - Did confidence scores match actual accuracy?

4. **Week-End Report**:
   - Total bets: X
   - Win rate: Y%
   - ROI: Z%
   - Sharpe ratio: W
   - Worst miss: Player A, error = N pts
   - **Go/No-Go Decision**: If ROI > 3% and CLV > 0, approve live betting

### 6.5 Continuous Monitoring (Post-Deployment)

**Daily Checks**:
- RMSE: Should be < 5.5 for player props
- DNP errors: Should be 0
- API failures: Should be < 1% of calls
- Prediction generation time: Should be < 5 minutes

**Weekly Checks**:
- ROI: Should be positive (>0%)
- Closing Line Value: Should be positive
- Max drawdown: Should be < 15%
- Drift detection: If RMSE increases >10%, trigger retrain

**Monthly Checks**:
- Full backtest on last 30 days
- Compare to sharp bettor benchmarks (55-58% ATS, 3-5% ROI)
- Feature importance analysis: Are new features being used?
- Model calibration: Are probabilities accurate?

---

## 7. Technical Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Four Factors don't improve accuracy** | Low | Medium | Validate with research papers; if <1% improvement, investigate feature importance |
| **XGBoost meta-learner overfits** | Medium | Medium | Use strong regularization (alpha=0.1, lambda=1.0); shallow trees (max_depth=3); cross-validation |
| **Injury scraping breaks (NBA.com changes HTML)** | Medium | High | Monitor scraper daily; add ESPN as fallback; budget for RotoWire API upgrade |
| **The Odds API rate limits exceeded** | Low | Medium | 100k subscription should be sufficient (~300 calls/day); add caching with 5-min TTL |
| **Prediction generation too slow (>5 min)** | Medium | Low | Profile code, add caching for team stats; parallelize API calls with asyncio |
| **Model retraining takes >4 hours** | Low | Low | Optimize hyperparameters; use Dask for distributed training if needed |
| **Backtest shows no improvement over baseline** | Medium | High | Investigate feature importance; verify temporal discipline; check for data quality issues |
| **Live betting loses money (negative ROI)** | Medium | High | Start with 10% bankroll; strict stop-loss rules (3% daily, 8% weekly); revert to old model if needed |
| **Temporal leakage in new features** | Low | Critical | Automated tests for every feature function; audit 100 random games |
| **Deployment failures on Railway** | Low | Medium | Test in staging environment; use Docker for reproducibility; monitor with Sentry |

---

## 8. Dependencies and Prerequisites

### 8.1 External Dependencies (Already Secured ✅)
- ✅ Balldontlie API (GOAT tier - unlimited rate limits)
- ✅ The Odds API (100k subscription - historical odds, line movements)

### 8.2 Additional Dependencies (Nice-to-Have)
- ⚠️ RotoWire Injury API (~$100/month) - **Start with free scraping, upgrade if needed**
- 🔄 DARKO DPM / ESPN EPM / FiveThirtyEight RAPTOR - **Scraping (free) or paid API**

### 8.3 Infrastructure (Already Secured ✅)
- ✅ Railway (compute, scheduled jobs, PostgreSQL)
- ✅ GitHub (version control)
- ✅ Vercel (frontend dashboard, if needed)

### 8.4 Python Packages (Install)
```bash
pip install scikit-learn==1.3.0 xgboost==2.0.0 lightgbm==4.0.0 catboost==1.2.0
pip install numpy==1.24.0 pandas==2.0.0 scipy==1.11.0
pip install requests==2.31.0 aiohttp==3.9.0
pip install fastapi==0.104.0 uvicorn==0.24.0 pydantic==2.5.0
pip install apscheduler==3.10.0 python-dotenv==1.0.0
pip install plotly==5.18.0 jinja2==3.1.2
pip install beautifulsoup4==4.12.0 lxml==4.9.0  # For web scraping
pip install psycopg2-binary==2.9.0  # PostgreSQL driver
pip install pytest==7.4.0 pytest-cov==4.1.0  # Testing
```

---

## 9. Success Criteria Summary

**Phase 1 (Foundation) - Week 2**:
- ✅ Zero DNP (Did Not Play) errors in predictions
- ✅ Overall RMSE < 5.3 (from 5.4)
- ✅ Threes R² > -0.4 (from -0.57)
- ✅ Backtest shows no regression

**Phase 2 (Enhancement) - Week 4**:
- ✅ Overall RMSE < 5.0
- ✅ ROI (Elite tier) > 5%
- ✅ Positive CLV (beat closing line)
- ✅ Confidence scores correlate with actual accuracy

**Phase 3 (Optimization) - Week 6**:
- ✅ Overall RMSE < 4.8
- ✅ Points RMSE < 5.5
- ✅ Threes R² > 0.10
- ✅ ROI (All bets) > 3%
- ✅ ROI (Elite tier) > 7%
- ✅ Sharpe ratio > 1.5
- ✅ Max drawdown < 15%

**Phase 4 (Production) - Week 8**:
- ✅ Paper trading ROI > 3% (7 days)
- ✅ Prediction generation < 5 minutes
- ✅ Live betting with 10% bankroll deployed
- ✅ Positive ROI after 30 live bets

**Long-Term (Month 3+)**:
- ✅ Sustained ROI > 5% over 1000+ bets
- ✅ Positive CLV consistently
- ✅ Sharpe ratio > 1.5
- ✅ Beat professional sharp benchmarks (55-58% ATS, 5-8% ROI)

---

## 10. Next Steps

1. ✅ **User Reviews and Approves This Spec** → Proceed to Planning step
2. ✅ Create detailed implementation plan (break down tasks, estimate hours)
3. ✅ Set up development environment (install dependencies, PostgreSQL)
4. ✅ Begin Phase 1: Foundation (injury detection, Four Factors, stacking meta-learner)
5. ✅ Run first backtest, compare to baseline
6. ✅ Iterate based on results, proceed to Phase 2

**Estimated Timeline**: 8 weeks (2 weeks per phase)
**Estimated Effort**: ~200-250 hours (full-time equivalent)

---

**End of Technical Specification**
