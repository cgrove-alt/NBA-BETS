-- Calibration database tables (migrated from data/calibration.db SQLite)
-- Used by: calibration_tracker/database.py

-- Predictions: every prediction made with full context
CREATE TABLE IF NOT EXISTS calibration_predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    game_date DATE NOT NULL,
    game_id INTEGER,

    -- Player info
    player_id INTEGER NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    team VARCHAR(10) NOT NULL,
    opponent VARCHAR(10) NOT NULL,
    position VARCHAR(10),

    -- Prediction details
    prop_type VARCHAR(20) NOT NULL,
    predicted_value REAL NOT NULL,
    prop_line REAL NOT NULL,
    predicted_over_prob REAL,
    confidence REAL,
    edge REAL,

    -- Minutes prediction
    minutes_predicted REAL,
    minutes_p10 REAL,
    minutes_p90 REAL,
    minutes_uncertainty VARCHAR(20),

    -- Game context
    is_home BOOLEAN,
    spread REAL,
    total REAL,
    is_favorite BOOLEAN,
    is_back_to_back BOOLEAN,
    days_rest INTEGER,

    -- Player context
    season_avg REAL,
    recent_avg REAL,
    vs_opponent_avg REAL,

    -- Model info
    model_version VARCHAR(50),
    features_hash VARCHAR(64),

    -- Status
    status VARCHAR(20) DEFAULT 'pending',

    UNIQUE(player_id, game_date, prop_type)
);

CREATE INDEX IF NOT EXISTS idx_cal_pred_game_date ON calibration_predictions(game_date);
CREATE INDEX IF NOT EXISTS idx_cal_pred_player ON calibration_predictions(player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_cal_pred_prop_type ON calibration_predictions(prop_type);
CREATE INDEX IF NOT EXISTS idx_cal_pred_status ON calibration_predictions(status);

-- Outcomes: actual results matched to predictions
CREATE TABLE IF NOT EXISTS calibration_outcomes (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES calibration_predictions(id),
    recorded_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Actual results
    actual_value REAL NOT NULL,
    actual_minutes REAL,

    -- Result classification
    result VARCHAR(10) NOT NULL,
    hit BOOLEAN NOT NULL,
    error REAL,

    -- Line movement
    closing_line REAL,
    clv REAL,

    -- Additional context
    game_score_diff INTEGER,
    player_started BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_cal_out_prediction ON calibration_outcomes(prediction_id);

-- Calibration adjustments: active bias corrections by dimension
CREATE TABLE IF NOT EXISTS calibration_adjustments (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    valid_from TIMESTAMP NOT NULL,
    valid_until TIMESTAMP,

    -- Dimension
    dimension VARCHAR(50) NOT NULL,
    dimension_value VARCHAR(50) NOT NULL,

    -- Adjustment
    bias REAL NOT NULL,
    adjustment REAL NOT NULL,
    confidence_multiplier REAL DEFAULT 1.0,

    -- Evidence
    sample_size INTEGER NOT NULL,
    hit_rate REAL,
    avg_error REAL,
    std_error REAL,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,

    UNIQUE(dimension, dimension_value, valid_from)
);

CREATE INDEX IF NOT EXISTS idx_cal_adj_active ON calibration_adjustments(is_active, dimension);

-- Daily reports: daily performance summaries
CREATE TABLE IF NOT EXISTS calibration_daily_reports (
    id SERIAL PRIMARY KEY,
    report_date DATE NOT NULL UNIQUE,
    generated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    total_predictions INTEGER,
    matched_predictions INTEGER,
    overall_hit_rate REAL,
    overall_clv REAL,

    report_json JSONB NOT NULL,

    status VARCHAR(20) DEFAULT 'complete'
);

-- Weekly reports: weekly performance summaries
CREATE TABLE IF NOT EXISTS calibration_weekly_reports (
    id SERIAL PRIMARY KEY,
    week_ending DATE NOT NULL UNIQUE,
    generated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    total_predictions INTEGER,
    matched_predictions INTEGER,
    overall_hit_rate REAL,
    overall_clv REAL,
    overall_roi REAL,
    ece REAL,
    report_json JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'complete'
);
