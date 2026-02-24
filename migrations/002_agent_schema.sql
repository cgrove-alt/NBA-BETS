-- NBA Betting Model - Agent Schema Migration
-- Version: 002 - Agent Infrastructure
-- Date: 2026-02-24
-- Description: Creates tables for agent framework (token budgets, run history, registry)

-- ============================================================================
-- AGENT TOKEN BUDGETS
-- ============================================================================
CREATE TABLE IF NOT EXISTS agent_token_budgets (
    agent_name VARCHAR(50) PRIMARY KEY,
    daily_limit INTEGER DEFAULT 50000,
    used_today INTEGER DEFAULT 0,
    reset_date DATE DEFAULT CURRENT_DATE
);

-- ============================================================================
-- AGENT RUNS (execution history)
-- ============================================================================
CREATE TABLE IF NOT EXISTS agent_runs (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(50) NOT NULL,
    run_id VARCHAR(36) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status VARCHAR(20) NOT NULL,
    success BOOLEAN,
    tokens_used INTEGER DEFAULT 0,
    execution_seconds REAL,
    messages_sent INTEGER DEFAULT 0,
    errors JSONB,
    payload JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_name_date ON agent_runs(agent_name, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status);

-- ============================================================================
-- AGENT REGISTRY (agent metadata and status)
-- ============================================================================
CREATE TABLE IF NOT EXISTS agent_registry (
    agent_name VARCHAR(50) PRIMARY KEY,
    agent_class VARCHAR(100) NOT NULL,
    schedule VARCHAR(100),
    enabled BOOLEAN DEFAULT true,
    status VARCHAR(20) DEFAULT 'idle',
    last_run_at TIMESTAMP,
    last_result JSONB,
    registered_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE agent_token_budgets IS 'Daily token budgets per agent for LLM API cost control';
COMMENT ON TABLE agent_runs IS 'Execution history for all agent runs with timing, tokens, and errors';
COMMENT ON TABLE agent_registry IS 'Central registry of agents with schedule and status';

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

SELECT 'Schema migration 002 (agent infrastructure) completed successfully!' AS status;
