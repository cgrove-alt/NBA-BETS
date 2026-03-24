-- Migration 012: Add Phase 4 odds fields to predictions_history
-- Purpose: Store implied probability, EV, best-book, and line movement data
--          alongside each daily prediction so the API can surface them.
-- Used by: nba_models/inference/daily_predictions.py (INSERT), dashboard/data_service.py (SELECT)

DO $$
BEGIN
    -- 4.1: Implied probability (vig-free, from The Odds API devigging)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'predictions_history' AND column_name = 'implied_probability'
    ) THEN
        ALTER TABLE predictions_history ADD COLUMN implied_probability FLOAT;
        COMMENT ON COLUMN predictions_history.implied_probability IS
            'Vig-free implied probability from the market odds for the recommended side.';
    END IF;

    -- 4.1: Expected value per dollar staked
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'predictions_history' AND column_name = 'ev_per_dollar'
    ) THEN
        ALTER TABLE predictions_history ADD COLUMN ev_per_dollar FLOAT;
        COMMENT ON COLUMN predictions_history.ev_per_dollar IS
            'EV per dollar wagered: (model_prob * payout) - (1 - model_prob). Positive = value bet.';
    END IF;

    -- 4.1: American odds used for EV calculation
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'predictions_history' AND column_name = 'best_odds'
    ) THEN
        ALTER TABLE predictions_history ADD COLUMN best_odds INTEGER;
        COMMENT ON COLUMN predictions_history.best_odds IS
            'Best available American odds for the recommended side across all sportsbooks.';
    END IF;

    -- 4.2: Sportsbook with the best available odds (line shopping winner)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'predictions_history' AND column_name = 'best_book'
    ) THEN
        ALTER TABLE predictions_history ADD COLUMN best_book VARCHAR(50);
        COMMENT ON COLUMN predictions_history.best_book IS
            'Sportsbook offering the best odds for the recommended pick (line shopping result).';
    END IF;

    -- 4.3: Line movement signal relative to model prediction
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'predictions_history' AND column_name = 'line_movement_signal'
    ) THEN
        ALTER TABLE predictions_history ADD COLUMN line_movement_signal VARCHAR(20);
        COMMENT ON COLUMN predictions_history.line_movement_signal IS
            'CONFIRMS_MODEL: line moved in our favor. WARNS_MODEL: line moved against us. NEUTRAL: no significant movement.';
    END IF;

    -- 4.3: Opening line for movement comparison
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'predictions_history' AND column_name = 'opening_line'
    ) THEN
        ALTER TABLE predictions_history ADD COLUMN opening_line FLOAT;
        COMMENT ON COLUMN predictions_history.opening_line IS
            'Opening line at market open, for line movement calculation.';
    END IF;

    -- 4.2: Over odds from the recommended sportsbook
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'predictions_history' AND column_name = 'over_odds'
    ) THEN
        ALTER TABLE predictions_history ADD COLUMN over_odds INTEGER;
    END IF;

    -- 4.2: Under odds from the recommended sportsbook
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'predictions_history' AND column_name = 'under_odds'
    ) THEN
        ALTER TABLE predictions_history ADD COLUMN under_odds INTEGER;
    END IF;
END $$;

SELECT 'Schema migration 012 completed successfully!' AS status;
