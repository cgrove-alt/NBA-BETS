-- Migration 010: Add line_source and line_vendor columns to predictions_history
-- These columns are written by daily_predictions.py but were never formally
-- added to the schema, causing UndefinedColumn errors on fresh deployments.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'predictions_history' AND column_name = 'line_source'
    ) THEN
        ALTER TABLE predictions_history ADD COLUMN line_source VARCHAR(50);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'predictions_history' AND column_name = 'line_vendor'
    ) THEN
        ALTER TABLE predictions_history ADD COLUMN line_vendor VARCHAR(50);
    END IF;
END $$;

SELECT 'Schema migration 010 completed successfully!' AS status;
