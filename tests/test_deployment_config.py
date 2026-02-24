#!/usr/bin/env python3
"""
Deployment Configuration Tests

Verifies all deployment files are properly configured for Railway.
Uses pytest assertions so failures are actually caught by the test runner.

Run: python3 -m pytest tests/test_deployment_config.py -v
"""

import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_railway_toml_exists():
    """railway.toml exists with required sections."""
    toml_path = Path("railway.toml")
    assert toml_path.exists(), "railway.toml not found"

    content = toml_path.read_text()

    for req in ["[build]", "[deploy]", "startCommand", "healthcheckPath"]:
        assert req in content, f"Missing required section: {req}"


def test_migration_script_exists():
    """Initial migration script exists with required tables."""
    migration_path = Path("migrations/001_initial_schema.sql")
    assert migration_path.exists(), "migrations/001_initial_schema.sql not found"

    content = migration_path.read_text()

    required_tables = [
        "CREATE TABLE IF NOT EXISTS predictions_history",
    ]
    for table in required_tables:
        assert table in content, f"Missing table creation: {table}"


def test_env_example_exists():
    """.env.example exists with all required variables."""
    env_path = Path(".env.example")
    assert env_path.exists(), ".env.example not found"

    content = env_path.read_text()

    required_vars = [
        "BALLDONTLIE_API_KEY",
        "DATABASE_URL",
        "THE_ODDS_API_KEY",
        "AUTH_ENABLED",
        "JWT_SECRET_KEY",
    ]
    for var in required_vars:
        assert var in content, f"Missing environment variable: {var}"


def test_requirements_txt():
    """Root requirements.txt has all necessary packages."""
    req_path = Path("requirements.txt")
    assert req_path.exists(), "requirements.txt not found"

    content = req_path.read_text()

    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "catboost",
        "xgboost",
        "requests",
        "apscheduler",
        "plotly",
        "jinja2",
    ]
    for pkg in required_packages:
        assert pkg in content, f"Missing package: {pkg}"


def test_api_structure():
    """FastAPI app exists with required endpoints."""
    from backend.api import app

    assert app is not None, "FastAPI app not found"

    routes = [route.path for route in app.routes if hasattr(route, "path")]

    required_routes = [
        "/api/health",
        "/api/predictions/{date}",
        "/api/injuries/{date}",
        "/api/backtest/latest",
    ]
    for route in required_routes:
        assert route in routes, f"Missing route: {route}"


def test_scheduled_scripts_exist():
    """Scheduled job scripts exist."""
    required_scripts = [
        "daily_predictions.py",
        "odds_tracker_service.py",
        "scheduled_retraining.py",
    ]
    for script in required_scripts:
        assert Path(script).exists(), f"Missing script: {script}"


def test_deployment_docs():
    """Deployment documentation exists."""
    assert Path("docs/railway-setup.md").exists(), "docs/railway-setup.md not found"


def test_verify_script():
    """System verification script exists."""
    assert Path("scripts/verify_system.py").exists(), "scripts/verify_system.py not found"


def test_agent_scheduler_exists():
    """Agent scheduler daemon script exists."""
    assert Path("agents/core/agent_runner.py").exists(), "agents/core/agent_runner.py not found"


def test_env_agent_vars():
    """.env.example includes agent-related environment variables."""
    env_path = Path(".env.example")
    assert env_path.exists()

    content = env_path.read_text()

    agent_vars = ["REDIS_URL", "GEMINI_API_KEY"]
    for var in agent_vars:
        assert var in content, f"Missing agent env var: {var}"


def test_agent_prompts_exist():
    """All agent prompt files exist."""
    prompts_dir = Path("agents/prompts")
    assert prompts_dir.exists(), "agents/prompts/ directory not found"

    required_prompts = [
        "pregame.md",
        "postgame.md",
        "odds_monitor.md",
        "orchestrator.md",
        "watchdog.md",
        "briefing.md",
    ]
    for prompt in required_prompts:
        assert (prompts_dir / prompt).exists(), f"Missing agent prompt: {prompt}"


def test_migration_runner_importable():
    """Migration runner script can be imported."""
    from scripts.run_migrations import run_migrations
    assert callable(run_migrations)
