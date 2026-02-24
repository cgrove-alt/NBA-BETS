-- NBA Betting Model - PostgreSQL Schema Migration
-- Version: 001 - Initial Schema
-- Date: 2026-01-19 (revised 2026-02-24)
-- Description: Creates tables actually used by production PostgreSQL code.
--
-- NOTE: Many services (odds tracker, retraining scheduler, calibration tracker)
-- use local SQLite databases. Only tables written to by code that references
-- DATABASE_URL are defined here. Additional tables can be added as services
-- migrate from SQLite to PostgreSQL.

-- ============================================================================
-- PREDICTIONS HISTORY TABLE
-- Used by: nba_models/inference/daily_predictions.py (lines 2630-2708)
-- Stores daily prop predictions with edges, confidence, and bet signals.
-- ============================================================================
CREATE TABLE IF NOT EXISTS predictions_history (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    game VARCHAR(100),
    player_name VARCHAR(100) NOT NULL,
    team VARCHAR(10),
    prop_type VARCHAR(20) NOT NULL,
    prediction FLOAT NOT NULL,
    pred_low FLOAT,
    pred_median FLOAT,
    pred_high FLOAT,
    line FLOAT NOT NULL,
    over_prob FLOAT,
    edge FLOAT,
    confidence_score FLOAT NOT NULL,
    edge_quality_tier VARCHAR(20),
    suggested_bet_size FLOAT,
    bet_recommendation VARCHAR(20),
    pick VARCHAR(10),
    uncertainty_flag VARCHAR(50),
    injury_boost BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(date, player_name, prop_type)
);

CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions_history(date);
CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions_history(player_name, date);
CREATE INDEX IF NOT EXISTS idx_predictions_prop ON predictions_history(prop_type, date);
CREATE INDEX IF NOT EXISTS idx_predictions_edge ON predictions_history(edge_quality_tier, date);

-- ============================================================================
-- DATA QUALITY CHECKS
-- ============================================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_pred_confidence_valid'
    ) THEN
        ALTER TABLE predictions_history ADD CONSTRAINT chk_pred_confidence_valid
            CHECK (confidence_score >= 0 AND confidence_score <= 100);
    END IF;
END $$;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE predictions_history IS 'Daily prop predictions with edges, confidence, and bet signals. Written by daily_predictions.py.';

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
SELECT 'Schema migration 001 completed successfully!' AS status;
