-- Injury tracking table (migrated from data/injuries.db SQLite)
-- Used by: nba_data/sources/injury_tracker_v3.py

CREATE TABLE IF NOT EXISTS injury_reports (
    id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    team VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL,
    injury_type VARCHAR(100),
    injury_detail TEXT,
    game_date DATE,
    last_update TIMESTAMP NOT NULL,
    source VARCHAR(50) NOT NULL,
    UNIQUE(player_id, game_date)
);

CREATE INDEX IF NOT EXISTS idx_injury_player_date ON injury_reports(player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_injury_game_date ON injury_reports(game_date);
