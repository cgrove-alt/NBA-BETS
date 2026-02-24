-- Prop prediction tracking table (migrated from nba_betting/edge/prop_predictions.db SQLite)
-- Used by: nba_betting/edge/prop_tracker.py

CREATE TABLE IF NOT EXISTS prop_prediction_tracking (
    prediction_id VARCHAR(20) PRIMARY KEY,
    game_id VARCHAR(50),
    game_date DATE,
    player_id INTEGER,
    player_name VARCHAR(100),
    team_abbrev VARCHAR(10),
    opponent_abbrev VARCHAR(10),
    prop_type VARCHAR(20),
    predicted_value REAL,
    market_line REAL,
    pick VARCHAR(10),
    edge_pct REAL,
    confidence REAL,
    opp_def_rating REAL,
    opp_adjustment REAL,
    actual_value REAL,
    hit INTEGER,
    created_at TIMESTAMP,
    settled_at TIMESTAMP,
    is_settled BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ppt_game_date ON prop_prediction_tracking(game_date);
CREATE INDEX IF NOT EXISTS idx_ppt_player_id ON prop_prediction_tracking(player_id);
CREATE INDEX IF NOT EXISTS idx_ppt_is_settled ON prop_prediction_tracking(is_settled);
