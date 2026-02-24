-- Bet tracking / CLV tables (migrated from data/bet_tracking.db SQLite)
-- Used by: nba_betting/edge/bet_tracker.py, nba_betting/edge/clv_bridge.py

CREATE TABLE IF NOT EXISTS tracked_bets (
    bet_id VARCHAR(200) PRIMARY KEY,
    placed_at TIMESTAMP NOT NULL,
    sport VARCHAR(20) DEFAULT 'NBA',
    bet_type VARCHAR(30) NOT NULL,
    sportsbook VARCHAR(50),
    event_id VARCHAR(100),
    event_name VARCHAR(200),
    event_date TIMESTAMP,
    selection VARCHAR(200) NOT NULL,
    odds REAL NOT NULL,
    stake REAL NOT NULL,
    potential_payout REAL,
    model_probability REAL,
    implied_probability REAL,
    edge REAL,
    opening_odds REAL,
    closing_odds REAL,
    line_movement REAL DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    actual_result TEXT,
    pnl REAL DEFAULT 0,
    settled_at TIMESTAMP,
    notes TEXT,
    tags JSONB,
    parlay_legs JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tracked_bets_placed_at ON tracked_bets(placed_at);
CREATE INDEX IF NOT EXISTS idx_tracked_bets_status ON tracked_bets(status);
CREATE INDEX IF NOT EXISTS idx_tracked_bets_bet_type ON tracked_bets(bet_type);
