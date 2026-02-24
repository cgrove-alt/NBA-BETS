-- Odds history tables (migrated from odds_history.db SQLite)
-- Used by: nba_betting/odds/betting_market_features.py
-- Table names prefixed with "tracked_" to avoid collision with 001_initial_schema.sql

-- Game metadata
CREATE TABLE IF NOT EXISTS tracked_games (
    game_id VARCHAR(100) PRIMARY KEY,
    home_team VARCHAR(50) NOT NULL,
    away_team VARCHAR(50) NOT NULL,
    commence_time TIMESTAMP NOT NULL,
    sport VARCHAR(50) DEFAULT 'basketball_nba',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Point-in-time odds snapshots per sportsbook
CREATE TABLE IF NOT EXISTS tracked_odds_history (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(100) NOT NULL REFERENCES tracked_games(game_id),
    timestamp TIMESTAMP NOT NULL,
    book_name VARCHAR(50) NOT NULL,
    market VARCHAR(30) NOT NULL,
    home_odds REAL,
    away_odds REAL,
    home_line REAL,
    away_line REAL,
    total REAL,
    over_odds REAL,
    under_odds REAL,
    is_opening BOOLEAN DEFAULT FALSE,
    is_closing BOOLEAN DEFAULT FALSE,
    UNIQUE(game_id, timestamp, book_name, market)
);

CREATE INDEX IF NOT EXISTS idx_toh_game_market ON tracked_odds_history(game_id, market, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_toh_timestamp ON tracked_odds_history(timestamp DESC);

-- Line movement analysis summaries
CREATE TABLE IF NOT EXISTS tracked_line_movements (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(100) NOT NULL REFERENCES tracked_games(game_id),
    market VARCHAR(30) NOT NULL,
    opening_line REAL,
    closing_line REAL,
    movement REAL,
    opening_time TIMESTAMP,
    closing_time TIMESTAMP,
    num_moves INTEGER,
    max_move REAL,
    rlm_detected BOOLEAN DEFAULT FALSE,
    steam_detected BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(game_id, market)
);

CREATE INDEX IF NOT EXISTS idx_tlm_game ON tracked_line_movements(game_id);
