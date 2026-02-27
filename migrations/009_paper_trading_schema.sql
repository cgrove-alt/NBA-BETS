-- Paper Trading / Forward Validation
-- Stores ALL predictions (not just recommended bets) for forward validation

CREATE TABLE IF NOT EXISTS paper_trades (
    id SERIAL PRIMARY KEY,
    trade_id TEXT UNIQUE NOT NULL,
    game_date DATE NOT NULL,
    game_id TEXT,
    player_name TEXT NOT NULL,
    prop_type TEXT NOT NULL,
    line FLOAT NOT NULL,
    direction TEXT NOT NULL,
    predicted_value FLOAT,
    over_prob FLOAT,
    edge FLOAT,
    true_ev FLOAT,
    should_bet BOOLEAN DEFAULT FALSE,
    bet_size FLOAT DEFAULT 0,
    over_odds INTEGER,
    under_odds INTEGER,
    confidence FLOAT,
    tier TEXT,
    actual_value FLOAT,
    result TEXT,
    profit_loss FLOAT,
    settled_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_date ON paper_trades(game_date);
CREATE INDEX IF NOT EXISTS idx_paper_trades_unsettled ON paper_trades(result) WHERE result IS NULL;
CREATE INDEX IF NOT EXISTS idx_paper_trades_prop ON paper_trades(prop_type);
CREATE INDEX IF NOT EXISTS idx_paper_trades_should_bet ON paper_trades(should_bet) WHERE should_bet = TRUE;
