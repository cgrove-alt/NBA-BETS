-- NBA Betting Model - PostgreSQL Schema Migration
-- Version: 001 - Initial Schema
-- Date: 2026-01-19
-- Description: Creates all tables for NBA betting model (teams, players, games, odds, injuries, predictions)

-- ============================================================================
-- TEAMS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    nba_id INTEGER UNIQUE NOT NULL,
    abbreviation VARCHAR(3) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    city VARCHAR(50),
    conference VARCHAR(4),
    division VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_teams_nba_id ON teams(nba_id);
CREATE INDEX IF NOT EXISTS idx_teams_abbreviation ON teams(abbreviation);

-- ============================================================================
-- PLAYERS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    nba_id INTEGER UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    team_id INTEGER REFERENCES teams(id),
    position VARCHAR(10),
    height VARCHAR(10),
    weight INTEGER,
    jersey_number INTEGER,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_players_nba_id ON players(nba_id);
CREATE INDEX IF NOT EXISTS idx_players_team_id ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);

-- ============================================================================
-- GAMES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS games (
    id SERIAL PRIMARY KEY,
    nba_id INTEGER UNIQUE NOT NULL,
    home_team_id INTEGER REFERENCES teams(id) NOT NULL,
    away_team_id INTEGER REFERENCES teams(id) NOT NULL,
    game_date DATE NOT NULL,
    season INTEGER NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    status VARCHAR(20) DEFAULT 'scheduled',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_games_nba_id ON games(nba_id);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_home_team ON games(home_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_games_away_team ON games(away_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);

-- ============================================================================
-- PLAYER GAME STATS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS player_game_stats (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id) NOT NULL,
    player_id INTEGER REFERENCES players(id) NOT NULL,
    team_id INTEGER REFERENCES teams(id) NOT NULL,
    minutes INTEGER,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    fgm INTEGER,
    fga INTEGER,
    fg3m INTEGER,
    fg3a INTEGER,
    ftm INTEGER,
    fta INTEGER,
    oreb INTEGER,
    dreb INTEGER,
    pf INTEGER,
    plus_minus INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_player_stats_game ON player_game_stats(game_id);
CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_game_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_player_stats_team ON player_game_stats(team_id);

-- ============================================================================
-- INJURIES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS injuries (
    id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    status VARCHAR(20),
    injury_type VARCHAR(100),
    detected_at TIMESTAMP DEFAULT NOW(),
    source VARCHAR(50),
    UNIQUE(player_id, game_date)
);

CREATE INDEX IF NOT EXISTS idx_injuries_date ON injuries(game_date);
CREATE INDEX IF NOT EXISTS idx_injuries_player ON injuries(player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_injuries_status ON injuries(status);

-- ============================================================================
-- ODDS HISTORY TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS odds_history (
    id SERIAL PRIMARY KEY,
    game_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    book_name VARCHAR(50),
    market VARCHAR(20),
    home_odds REAL,
    away_odds REAL,
    home_line REAL,
    away_line REAL,
    total REAL,
    over_odds REAL,
    under_odds REAL,
    source VARCHAR(50) DEFAULT 'the_odds_api',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_odds_game_market ON odds_history(game_id, market, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_odds_timestamp ON odds_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_odds_book ON odds_history(book_name, timestamp);

-- ============================================================================
-- PREDICTIONS HISTORY TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS predictions_history (
    id SERIAL PRIMARY KEY,
    game_id INTEGER,
    player_id INTEGER,
    prediction_date DATE NOT NULL,
    prediction_type VARCHAR(20) NOT NULL,
    prop_type VARCHAR(20),
    predicted_value REAL NOT NULL,
    pred_low REAL,
    pred_median REAL,
    pred_high REAL,
    confidence_score INTEGER,
    edge_quality_tier VARCHAR(20),
    suggested_bet_size REAL,
    bet_recommendation VARCHAR(20),
    actual_value REAL,
    result VARCHAR(10),
    error REAL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions_history(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions_history(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions_history(player_id, prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions_history(prediction_type, prop_type);

-- ============================================================================
-- BETTING HISTORY TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS betting_history (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions_history(id),
    game_id INTEGER,
    player_id INTEGER,
    bet_date DATE NOT NULL,
    bet_type VARCHAR(20) NOT NULL,
    prop_type VARCHAR(20),
    stake REAL NOT NULL,
    odds REAL NOT NULL,
    predicted_value REAL,
    actual_value REAL,
    result VARCHAR(10),
    profit_loss REAL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_betting_date ON betting_history(bet_date);
CREATE INDEX IF NOT EXISTS idx_betting_prediction ON betting_history(prediction_id);
CREATE INDEX IF NOT EXISTS idx_betting_result ON betting_history(result);

-- ============================================================================
-- MODEL METADATA TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    trained_at TIMESTAMP DEFAULT NOW(),
    training_samples INTEGER,
    validation_rmse REAL,
    validation_r2 REAL,
    validation_roi REAL,
    active BOOLEAN DEFAULT false,
    model_path VARCHAR(255),
    hyperparameters JSONB,
    feature_importance JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_active ON model_metadata(active, model_type);
CREATE INDEX IF NOT EXISTS idx_model_name_version ON model_metadata(model_name, model_version);

-- ============================================================================
-- RETRAINING HISTORY TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS retraining_history (
    id SERIAL PRIMARY KEY,
    retrain_type VARCHAR(20) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    status VARCHAR(20) NOT NULL,
    games_fetched INTEGER,
    old_model_rmse REAL,
    new_model_rmse REAL,
    performance_change REAL,
    deployed BOOLEAN DEFAULT false,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retrain_date ON retraining_history(started_at);
CREATE INDEX IF NOT EXISTS idx_retrain_status ON retraining_history(status);

-- ============================================================================
-- DATA QUALITY CHECKS
-- ============================================================================

-- Add constraints
ALTER TABLE games ADD CONSTRAINT chk_games_score_positive
    CHECK (home_score IS NULL OR home_score >= 0);
ALTER TABLE games ADD CONSTRAINT chk_games_date_reasonable
    CHECK (game_date >= '2000-01-01');

ALTER TABLE player_game_stats ADD CONSTRAINT chk_stats_minutes_valid
    CHECK (minutes IS NULL OR (minutes >= 0 AND minutes <= 60));
ALTER TABLE player_game_stats ADD CONSTRAINT chk_stats_points_valid
    CHECK (points IS NULL OR points >= 0);

ALTER TABLE predictions_history ADD CONSTRAINT chk_pred_confidence_valid
    CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 100));

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE teams IS 'NBA teams master data';
COMMENT ON TABLE players IS 'NBA players master data with current team';
COMMENT ON TABLE games IS 'NBA games schedule and results';
COMMENT ON TABLE player_game_stats IS 'Player box score statistics by game';
COMMENT ON TABLE injuries IS 'Player injury reports by date (from NBA.com, ESPN)';
COMMENT ON TABLE odds_history IS 'Betting odds history from multiple books (5-min intervals)';
COMMENT ON TABLE predictions_history IS 'Model predictions with confidence and quantile bands';
COMMENT ON TABLE betting_history IS 'Actual bets placed with results and P&L';
COMMENT ON TABLE model_metadata IS 'Trained model versions with performance metrics';
COMMENT ON TABLE retraining_history IS 'Model retraining attempts and results';

-- ============================================================================
-- GRANT PERMISSIONS (adjust username as needed)
-- ============================================================================

-- For Railway, this is handled automatically via DATABASE_URL
-- If deploying elsewhere, grant permissions:
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO nba_betting_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO nba_betting_user;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

SELECT 'Schema migration 001 completed successfully!' AS status;
