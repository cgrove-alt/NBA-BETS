# NBA Model V3 "The Oracle" - Execution Tasks

## Phase 0: Granular Data Acquisition (The Missing Attribute)
- [ ] **Data Pipeline Infrastructure**
    - [ ] Create `tracking_data.py`
    - [ ] Implement `fetch_pbp_historical(game_id)` (Deep, slow for training)
    - [ ] Implement `fetch_pbp_live(game_id)` (Fast, light for inference)
    - [ ] Implement `fetch_shot_chart(game_id)` using `nba_api`
- [ ] **Data Processing**
    - [ ] Build `PBPParser` to convert text play-by-play into `Possession` objects
    - [ ] Create `ShotAtlas` (Heatmap of player efficiencies by (X,Y) zone)
    - [ ] Implement `RotationTracker` (Derive substitution matrix from PBP)

## Phase 1: Simulation Engine V3 (Tracking-Based)
- [ ] **Upgrade `simulation_engine.py`**
    - [ ] Refactor `PlayerStats` to `PlayerTrackingStats` (Include Zone Shooting %)
    - [ ] Update `_simulate_shot` to use `ShotAtlas` probabilities instead of season FG%
    - [ ] Update `_select_shooter` to use `RotationTracker` for realistic lineups
- [ ] **Validation**
    - [ ] Verify V3 calibration against historical tracking data

## Phase 2: Market Microstructure V3 (Latency-Optimized)
- [ ] **Speed Upgrades**
    - [ ] Refactor `OddsMonitor` to use multi-threaded polling
    - [ ] Implement "Heartbeat" mechanism for steam detection (< 1s latency)

## Phase 3: Qualitative V3 (Context-Aware)
- [ ] **LLM Integration**
    - [ ] Integrate `injury_report` into `RotationTracker` (Remove injured players from rotation)

---
## Archive: V2 Implementation (Completed Jan 2026)
- [x] Phase 1: Simulation Engine (Base Monte Carlo)
- [x] Phase 2: Market Microstructure (Base Odds)
- [x] Phase 3: Qualitative Intelligence (Base Sentiment)
- [x] Phase 4: Portfolio Optimization (Base Kelly)
