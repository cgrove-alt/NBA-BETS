from __future__ import annotations

import json

from backend import api as backend_api


def _clear_runtime_env(monkeypatch):
    for name in [
        "RAILWAY_ENVIRONMENT",
        "RAILWAY_SERVICE_NAME",
        "RAILWAY_SERVICE",
        "DATABASE_URL",
        "REDIS_URL",
        "BALLDONTLIE_API_KEY",
        "THE_ODDS_API_KEY",
        "GEMINI_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_environment_report_defaults_to_local_api(monkeypatch):
    _clear_runtime_env(monkeypatch)

    report = backend_api._build_environment_report()

    assert report["is_railway"] is False
    assert report["service_name"] == "nba-betting-api"
    assert report["shared_env"]["DATABASE_URL"]["present"] is False
    assert "nba-daily-predictions" in report["service_requirements"]


def test_railway_missing_required_envs_are_fatal(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "nba-betting-api")

    report = backend_api._build_environment_report()
    issues, warnings = backend_api._summarize_health_issues(
        models_loaded=True,
        db_connected=False,
        redis_connected=False,
        environment=report,
    )

    assert any("missing required env vars: DATABASE_URL" in issue for issue in issues)
    assert any("missing recommended env vars" in warning for warning in warnings)


def test_health_check_includes_environment_matrix(monkeypatch):
    _clear_runtime_env(monkeypatch)

    response = backend_api.health_check()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "healthy"
    assert payload["environment"]["service_name"] == "nba-betting-api"
    assert "service_requirements" in payload["environment"]
    assert "environment_summary" in payload["checks"]
    assert payload["issues"] == []
