# Betting Market Features - Quick Reference Guide

## Setup

### 1. Install Dependencies
```bash
# Already included in project
pip install numpy sqlite3
```

### 2. Set API Key
```bash
export THE_ODDS_API_KEY="your_odds_api_key_here"
```

### 3. Initialize Database
```python
from betting_market_features import BettingMarketFeatures

tracker = BettingMarketFeatures()
# Database automatically initialized at odds_history.db
```

---

## Basic Usage

### Fetch Current Odds
```python
from betting_market_features import BettingMarketFeatures

tracker = BettingMarketFeatures()

# Fetch and cache odds (respects 5-minute TTL)
odds = tracker.fetch_current_odds()
print(f"Fetched odds for {len(odds)} games")
```

### Store Odds in Database
```python
# Store current odds
count = tracker.fetch_and_store_odds()
print(f"Stored {count} odds snapshots")

# Mark as opening line (run at 9 AM daily)
count = tracker.fetch_and_store_odds(mark_as_opening=True)

# Mark as closing line (run at game time)
count = tracker.fetch_and_store_odds(mark_as_closing=True)
```

### Generate Features for ML Model
```python
# Main method for prediction pipeline
features = tracker.get_market_features(
    game_id="abc123",
    home_team="Los Angeles Lakers",
    away_team="Boston Celtics"
)

print(features)
# Output:
# {
#     'opening_line': -5.0,
#     'closing_line': -7.0,
#     'line_movement': -2.0,
#     'rlm_flag': True,
#     'consensus_odds': -110,
#     'steam_move_flag': False
# }
```

---

## Advanced Usage

### Calculate Line Movement
```python
# Spread movement
movement = tracker.calculate_line_movement(game_id, 'spread')
print(f"Spread moved {movement} points")  # -2.0 = moved toward away

# Totals movement
movement = tracker.calculate_line_movement(game_id, 'totals')
print(f"Total moved {movement} points")  # +1.5 = moved higher

# Moneyline movement (probability)
movement = tracker.calculate_line_movement(game_id, 'moneyline')
print(f"Home probability changed {movement:.1%}")  # -0.05 = -5%
```

### Detect Reverse Line Movement (RLM)
```python
# Basic RLM detection (heuristic)
rlm = tracker.detect_reverse_line_movement(game_id, 'spread')
if rlm:
    print("⚠️ RLM detected - sharp money opposite to public")

# With public betting data (if available)
rlm = tracker.detect_reverse_line_movement(
    game_id,
    'spread',
    public_betting_pct=0.75  # 75% of public on home
)
if rlm:
    print("⚠️ RLM: Public on home, but line moved away")
```

### Detect Steam Moves
```python
# Check for rapid line movement in last 15 minutes
steam = tracker.detect_steam_move(game_id, 'spread', lookback_minutes=15)
if steam:
    print("🔥 STEAM MOVE: Sharp money moving line rapidly")

# Custom time window
steam = tracker.detect_steam_move(game_id, 'totals', lookback_minutes=30)
```

### Calculate Consensus Odds
```python
# Get fair market odds from multiple sportsbooks
consensus = tracker.calculate_consensus_odds(game_id, 'spread')

if consensus:
    print(f"Consensus line: {consensus['consensus_line']}")
    print(f"Consensus odds: {consensus['consensus_odds']}")
    print(f"Based on {consensus['num_books']} sportsbooks")
```

---

## Background Job Setup (APScheduler)

### Continuous Odds Monitoring
```python
from apscheduler.schedulers.background import BackgroundScheduler
from betting_market_features import OddsTracker

# Initialize tracker
tracker = OddsTracker(
    api_key="your_key",
    update_interval_minutes=5
)

# Setup scheduler
scheduler = BackgroundScheduler()

# Fetch odds every 5 minutes during game days
scheduler.add_job(
    tracker.fetch_and_store_odds,
    'interval',
    minutes=5,
    start_date='08:00',  # 8 AM
    end_date='23:00'     # 11 PM
)

# Mark opening lines at 9 AM
def mark_opening():
    tracker.features.fetch_and_store_odds(mark_as_opening=True)

scheduler.add_job(
    mark_opening,
    'cron',
    hour=9,
    minute=0
)

# Start monitoring
scheduler.start()
print("Odds monitoring started")

# ... your app runs ...

# Stop when done
scheduler.shutdown()
```

---

## Integration with Prediction Pipeline

### In feature_engineering.py
```python
from betting_market_features import BettingMarketFeatures

# Initialize once (module-level or class member)
betting_tracker = BettingMarketFeatures()

def generate_game_features(game_id, home_team, away_team, game_date):
    # ... existing feature generation ...

    # Add betting market features
    market_features = betting_tracker.get_market_features(
        game_id, home_team, away_team
    )

    features.update({
        'opening_line': market_features['opening_line'],
        'closing_line': market_features['closing_line'],
        'line_movement': market_features['line_movement'],
        'rlm_flag': int(market_features['rlm_flag']),  # Convert to 0/1
        'consensus_odds': market_features['consensus_odds'],
        'steam_move_flag': int(market_features['steam_move_flag'])
    })

    return features
```

### In daily_predictions.py
```python
from betting_market_features import BettingMarketFeatures

def generate_daily_predictions():
    tracker = BettingMarketFeatures()

    # Fetch and store current odds
    tracker.fetch_and_store_odds()

    # Generate predictions for each game
    for game in today_games:
        # ... existing prediction code ...

        # Add market features
        market_features = tracker.get_market_features(
            game['id'],
            game['home_team'],
            game['away_team']
        )

        # Include in prediction output
        prediction['market_features'] = market_features

        # Flag suspicious lines
        if market_features['rlm_flag']:
            prediction['alert'] = "RLM detected - proceed with caution"
        if market_features['steam_move_flag']:
            prediction['alert'] = "Steam move detected - sharp action"
```

---

## Utility Functions

### Odds Conversion
```python
tracker = BettingMarketFeatures()

# American odds → Probability
prob = tracker._american_to_prob(-150)
print(f"Probability: {prob:.1%}")  # 60.0%

prob = tracker._american_to_prob(+200)
print(f"Probability: {prob:.1%}")  # 33.3%

# Probability → American odds
odds = tracker._prob_to_american(0.60)
print(f"American odds: {odds:+d}")  # -150

odds = tracker._prob_to_american(0.333)
print(f"American odds: {odds:+d}")  # +200
```

---

## Database Queries (Direct Access)

### Query Odds History
```python
# Get recent odds for a game
history = tracker.db.get_odds_history(
    game_id,
    market='spread',
    lookback_minutes=60
)

for snap in history:
    print(f"{snap['timestamp']}: {snap['book_name']} @ {snap['home_line']}")
```

### Query Opening/Closing Lines
```python
# Get opening line
opening = tracker.db.get_opening_line(game_id, 'spread')
if opening:
    print(f"Opening: {opening['home_line']} ({opening['book_name']})")

# Get closing line
closing = tracker.db.get_closing_line(game_id, 'spread')
if closing:
    print(f"Closing: {closing['home_line']} ({closing['book_name']})")
```

---

## Configuration

### Thresholds (Customize in betting_market_features.py)
```python
STEAM_THRESHOLD_POINTS = 1.5   # Spread/total movement (points)
STEAM_THRESHOLD_ML = 0.03      # Moneyline movement (3%)
STEAM_TIME_WINDOW = 900        # 15 minutes (seconds)
RLM_THRESHOLD = 0.02           # 2% probability movement
UPDATE_INTERVAL_SECONDS = 300  # 5 minutes
```

### Database Location
```python
# Default location
tracker = BettingMarketFeatures(db_path="odds_history.db")

# Custom location
tracker = BettingMarketFeatures(db_path="/data/odds/nba_odds.db")
```

---

## Error Handling

### Graceful Degradation
```python
# Module handles missing data gracefully
features = tracker.get_market_features(game_id, home_team, away_team)

# If no data available, returns defaults:
# {
#     'opening_line': 0.0,
#     'closing_line': 0.0,
#     'line_movement': 0.0,
#     'rlm_flag': False,
#     'consensus_odds': -110,
#     'steam_move_flag': False
# }
```

### API Failures
```python
# Cached data is used if API fails
odds = tracker.fetch_current_odds()

# Check cache timestamp
if tracker._cache_timestamp:
    age = (datetime.now() - tracker._cache_timestamp).seconds
    print(f"Using cached data from {age} seconds ago")
```

---

## Testing

### Run Unit Tests
```bash
python3 tests/test_betting_features.py
```

### Run Manual Test
```bash
python3 betting_market_features.py
```

Expected output:
```
======================================================================
BETTING MARKET FEATURES TEST
======================================================================

1. DATABASE INITIALIZATION
----------------------------------------
Database path: odds_history.db
Schema created successfully

2. FETCH CURRENT ODDS
----------------------------------------
Fetched odds for 8 games
...
```

---

## Troubleshooting

### No API Key Error
```
Warning: No API key provided. Set THE_ODDS_API_KEY or pass api_key parameter.
```
**Solution**: Set environment variable or pass to constructor
```python
tracker = BettingMarketFeatures(api_key="your_key")
```

### Import Errors
```
Warning: odds_fetcher.py not found. Some features may be limited.
```
**Solution**: Module works standalone, but integrates better with existing code

### Empty Features
**Problem**: All features return default values
**Solution**: Ensure odds have been fetched and stored with opening/closing marks
```python
tracker.fetch_and_store_odds(mark_as_opening=True)
# Wait for game time
tracker.fetch_and_store_odds(mark_as_closing=True)
# Now features will have real values
```

---

## Best Practices

1. **Fetch Opening Lines**: Run at 9 AM daily with `mark_as_opening=True`
2. **Fetch Closing Lines**: Run 5-10 minutes before game time with `mark_as_closing=True`
3. **Update Frequency**: Every 5 minutes during game days (8 AM - 11 PM)
4. **Monitor API Usage**: The Odds API has request limits (check usage in response headers)
5. **Database Maintenance**: Periodically clean old odds (>7 days) to save space
6. **Cache Strategy**: 5-minute TTL balances freshness with API limits

---

## Example Workflow

### Daily Prediction Generation
```python
from betting_market_features import BettingMarketFeatures
from datetime import datetime

tracker = BettingMarketFeatures()

# Morning: Mark opening lines
if datetime.now().hour == 9:
    tracker.fetch_and_store_odds(mark_as_opening=True)
    print("✓ Opening lines captured")

# Throughout day: Update every 5 minutes
tracker.fetch_and_store_odds()

# Before games: Mark closing lines
for game in games_starting_soon:
    tracker.fetch_and_store_odds(mark_as_closing=True)
    print(f"✓ Closing lines captured for {game['home_team']} vs {game['away_team']}")

# Generate predictions with market features
for game in today_games:
    features = tracker.get_market_features(
        game['id'],
        game['home_team'],
        game['away_team']
    )

    # Make prediction with enhanced features
    prediction = model.predict(features)

    # Alert on suspicious market activity
    if features['rlm_flag'] or features['steam_move_flag']:
        print(f"⚠️ Market alert for {game['home_team']} vs {game['away_team']}")
```

---

## Support

For issues or questions:
- Check unit tests for usage examples
- Review `betting_market_features.py` docstrings
- See completion summary: `.zenflow/tasks/model-improvements-v2-3065/task_2.2_completion_summary.md`
