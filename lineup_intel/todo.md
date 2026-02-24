# Lineup Intel Module - Build Summary

## Completed Tasks

- [x] Create `lineup_intel/` directory structure
- [x] Build `injury_scraper.py` - Multi-source injury fetcher
  - Balldontlie API integration (primary)
  - ESPN web scraping (secondary)
  - InjuryStatus enum with availability probabilities
  - PlayerInjury dataclass with minutes multipliers
- [x] Build `lineup_tracker.py` - Starting lineup prediction
  - Historical starter analysis via Balldontlie
  - ESPN depth chart scraping
  - StarterInfo and LineupConfirmation dataclasses
- [x] Build `news_monitor.py` - Breaking news detection
  - ESPN news scraping
  - Alert classification (CRITICAL/HIGH/MEDIUM/LOW)
  - Minutes impact estimation
- [x] Build `lineup_intel_service.py` - Main integration service
  - GameIntel - Complete game lineup status
  - PlayerIntel - Individual player status
  - Injury edge calculation
  - Star player detection
- [x] Build `integration.py` - Minutes Oracle integration
  - LineupAwarePredictor class
  - Confidence adjustment calculation
  - Skip bet recommendations
- [x] Create `README.md` - Usage documentation
- [x] Fix Balldontlie API parameter issues

## Test Results

```
Found 211 injured players across the league
- Balldontlie: 100 injuries
- ESPN: 111 injuries (unique after merge: 211)

Star player OUT detection: Working
- Jayson Tatum (BOS): OUT
- Jaylen Brown (BOS): OUT
- Austin Reaves (LAL): OUT

Injury edge calculation: Working
- Example: LAL vs BOS shows "home" edge (Boston more injured)
```

## Review

### What Was Built
The Lineup Intel module provides real-time NBA lineup and injury intelligence for betting decisions. It combines data from multiple sources (Balldontlie API, ESPN) to give a comprehensive view of player availability.

### Key Features
1. **Multi-source injury aggregation** - Pulls from Balldontlie API and ESPN, merging and deduplicating
2. **Injury status classification** - OUT, DOUBTFUL, QUESTIONABLE, PROBABLE, GTD with availability percentages
3. **Minutes impact estimation** - Players with injuries often have reduced minutes even when playing
4. **Star player detection** - Critical alerts when star players are affected
5. **Confidence adjustment** - Reduces prop bet confidence based on lineup uncertainty
6. **Integration ready** - Works with existing Minutes Oracle and DataService

### Architecture
```
lineup_intel/
├── __init__.py              # Module exports
├── injury_scraper.py        # Multi-source injury fetching
├── lineup_tracker.py        # Starting lineup prediction
├── news_monitor.py          # Breaking news detection
├── lineup_intel_service.py  # Main service integration
├── integration.py           # Minutes Oracle integration
└── README.md                # Documentation
```

### Integration Points
- **Minutes Oracle**: Adjust predicted minutes based on injury status
- **DataService**: Skip OUT players, adjust confidence
- **Prop Predictions**: Factor in lineup uncertainty

### Limitations
1. ESPN depth charts don't include player stats (no minutes data)
2. Historical starter analysis requires Balldontlie All-Star tier
3. News parsing is keyword-based (could miss some alerts)

### Future Improvements
- Add more news sources (Twitter/X, FantasyLabs)
- Improve player name matching
- Add coach tendencies integration
- Real-time lineup confirmation tracking
