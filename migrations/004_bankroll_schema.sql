-- Bankroll tracking tables (migrated from data/bankroll.db SQLite)
-- Used by: edge_calculator/bankroll_manager.py

-- Current bankroll state
CREATE TABLE IF NOT EXISTS bankroll_state (
    id SERIAL PRIMARY KEY,
    amount REAL NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Pending (unsettled) bets with exposure
CREATE TABLE IF NOT EXISTS bankroll_pending_bets (
    bet_id VARCHAR(100) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    player_id INTEGER,
    prop_type VARCHAR(20) NOT NULL,
    pick VARCHAR(10) NOT NULL,
    line REAL NOT NULL,
    odds INTEGER NOT NULL,
    stake REAL NOT NULL,
    units REAL NOT NULL,
    game_id INTEGER,
    game_date DATE,
    team VARCHAR(10),
    opponent VARCHAR(10)
);

-- Settled bets with P&L
CREATE TABLE IF NOT EXISTS bankroll_settled_bets (
    bet_id VARCHAR(100) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    player_id INTEGER,
    prop_type VARCHAR(20) NOT NULL,
    pick VARCHAR(10) NOT NULL,
    line REAL NOT NULL,
    odds INTEGER NOT NULL,
    stake REAL NOT NULL,
    units REAL NOT NULL,
    game_date DATE,
    actual_value REAL NOT NULL,
    result VARCHAR(10) NOT NULL,
    profit_loss REAL NOT NULL,
    settled_at TIMESTAMP NOT NULL
);

-- Daily P&L aggregation
CREATE TABLE IF NOT EXISTS bankroll_daily_pl (
    date DATE PRIMARY KEY,
    starting_bankroll REAL,
    ending_bankroll REAL,
    total_staked REAL,
    total_returned REAL,
    profit_loss REAL,
    num_bets INTEGER,
    num_wins INTEGER,
    num_losses INTEGER
);
