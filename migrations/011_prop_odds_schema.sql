-- Migration 011: Player Prop Odds Snapshots
-- Purpose: Store point-in-time prop odds from multiple sportsbooks for line movement tracking.
-- Used by: nba_betting/odds/prop_odds_tracker.py (PropOddsTracker class)
--
-- This table powers Phase 4.3 (Line Movement Tracking) by recording prop lines
-- at multiple timestamps throughout the day so we can detect:
--   - Opening line vs current line movement
--   - Confirmation signals (line moved toward model prediction)
--   - Warning signals (line moved against model prediction)

CREATE TABLE IF NOT EXISTS prop_odds_snapshots (
    id SERIAL PRIMARY KEY,
    game_date DATE NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    prop_type VARCHAR(20) NOT NULL,     -- 'points', 'rebounds', 'assists', 'pra'
    book_name VARCHAR(50) NOT NULL,     -- 'draftkings', 'fanduel', 'betmgm', etc.
    line REAL NOT NULL,                 -- The betting line (e.g. 24.5)
    over_odds INTEGER,                  -- American odds for the over (e.g. -110)
    under_odds INTEGER,                 -- American odds for the under (e.g. -110)
    implied_prob_over REAL,             -- Vig-free implied probability for the over
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    is_opening BOOLEAN DEFAULT FALSE,   -- True for the first snapshot of the day
    UNIQUE(game_date, player_name, prop_type, book_name, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_pos_player_prop_date
    ON prop_odds_snapshots(player_name, prop_type, game_date, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pos_date_ts
    ON prop_odds_snapshots(game_date, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pos_book
    ON prop_odds_snapshots(book_name, game_date);

COMMENT ON TABLE prop_odds_snapshots IS
    'Point-in-time player prop odds from multiple sportsbooks. '
    'Used for line movement tracking and confirmation/warning signals.';

SELECT 'Schema migration 011 completed successfully!' AS status;
