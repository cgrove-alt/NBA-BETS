# Lineup Intel - NBA Lineup and Injury Intelligence

Real-time lineup and injury tracking for NBA betting.

## Components

### 1. InjuryScraper (`injury_scraper.py`)
Multi-source injury report fetcher:
- **Balldontlie API** (primary) - Structured, reliable
- **ESPN Injuries** (secondary) - Web scraping fallback

```python
from lineup_intel import InjuryScraper

scraper = InjuryScraper()
injuries = scraper.fetch_all_injuries()  # All league injuries
team_injuries = scraper.get_team_injuries("LAL")  # Team-specific
out_players = scraper.get_unavailable_players()  # Just OUT/DOUBTFUL
```

### 2. LineupTracker (`lineup_tracker.py`)
Starting lineup prediction and tracking:
- Historical starter analysis (Balldontlie)
- ESPN depth charts

```python
from lineup_intel import LineupTracker

tracker = LineupTracker()
lineup = tracker.get_lineup("LAL")
print(f"Confirmed: {lineup.is_confirmed}")
for starter in lineup.starters:
    print(f"{starter.position} {starter.player_name}")
```

### 3. NewsMonitor (`news_monitor.py`)
Breaking news detection for lineup changes:
- Injury updates
- Surprise scratches
- Load management decisions

```python
from lineup_intel import NewsMonitor

monitor = NewsMonitor()
alerts = monitor.get_critical_alerts()  # Star player alerts only
```

### 4. LineupIntelService (`lineup_intel_service.py`)
Main service combining all components:

```python
from lineup_intel import LineupIntelService

service = LineupIntelService()

# Full game intelligence
game = service.get_game_intel("LAL", "BOS")
print(f"Home star out: {game.home_star_out}")
print(f"Away star out: {game.away_star_out}")
print(f"Injury edge: {game.injury_edge}")

# Player-specific intelligence
player = service.get_player_intel("LeBron James", "LAL")
print(f"Status: {player.injury_status.value}")
print(f"Expected minutes: {player.expected_minutes}")
print(f"Availability: {player.availability_probability:.0%}")
```

## Integration with Minutes Oracle

```python
from lineup_intel.integration import LineupAwarePredictor

predictor = LineupAwarePredictor()

# Get minutes prediction with lineup context
pred = predictor.get_player_minutes_prediction(
    player_name="LeBron James",
    team="LAL",
    opponent_team="BOS"
)
print(f"Expected: {pred['expected_minutes']:.1f} min")
print(f"Range: {pred['p10']:.1f} - {pred['p90']:.1f}")

# Get confidence adjustment for prop bet
adj = predictor.get_prop_confidence_adjustment(
    player_name="LeBron James",
    team="LAL",
    base_confidence=65.0
)
print(f"Adjusted: {adj['adjusted_confidence']:.1f}")
print(f"Skip bet: {adj['skip_bet']}")
```

## Data Sources

| Source | Data Type | Update Frequency | Reliability |
|--------|-----------|------------------|-------------|
| Balldontlie API | Injuries, Stats | Real-time | High |
| ESPN | Injuries, Depth Charts | 15-30 min | Medium |

## Injury Statuses

- **OUT**: Player will not play (0% availability)
- **DOUBTFUL**: Unlikely to play (25% availability)
- **QUESTIONABLE**: Uncertain (50% availability)
- **PROBABLE**: Likely to play (85% availability)
- **GTD**: Game-time decision (50% availability)

## Alert Severity Levels

- **CRITICAL**: Star player affected
- **HIGH**: Key rotation player
- **MEDIUM**: Role player or uncertain status
- **LOW**: Minor impact

## Integration with DataService

Add to `data_service.py`:

```python
# Import
try:
    from lineup_intel import LineupIntelService
    LINEUP_INTEL_AVAILABLE = True
except ImportError:
    LineupIntelService = None
    LINEUP_INTEL_AVAILABLE = False

# In __init__
self._lineup_intel = None
if LINEUP_INTEL_AVAILABLE:
    self._lineup_intel = LineupIntelService()

# In _get_player_predictions
if self._lineup_intel:
    intel = self._lineup_intel.get_player_intel(player_name, team)

    # Skip if OUT
    if intel.injury_status.value == 'Out':
        return None

    # Adjust minutes
    minutes *= intel.minutes_multiplier
```
