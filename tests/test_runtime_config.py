from __future__ import annotations

from datetime import date

from dashboard.data_service import DataService
from nba_data.transformers.feature_engineering import (
    InjuryReportManager,
    PlayerPropFeatureGenerator,
)
from nba_data.utils.runtime_config import (
    get_data_dir,
    get_historical_csv_dir,
    get_live_seasons_dir,
    resolve_nba_season,
)


def test_resolve_nba_season_uses_regular_season_boundary():
    assert resolve_nba_season(as_of=date(2026, 3, 5)) == "2025-26"
    assert resolve_nba_season(as_of=date(2026, 10, 21)) == "2026-27"


def test_resolve_nba_season_honors_env_override(monkeypatch):
    monkeypatch.setenv("NBA_BETS_SEASON", "2099-00")
    assert resolve_nba_season() == "2099-00"


def test_runtime_paths_point_to_repo_data():
    data_dir = get_data_dir()
    assert data_dir.name == "data"
    assert get_historical_csv_dir().exists()
    assert get_live_seasons_dir().exists()


def test_feature_generators_default_to_active_season():
    expected = resolve_nba_season()
    assert PlayerPropFeatureGenerator().season == expected
    assert InjuryReportManager().season == expected


def test_data_service_uses_active_season(monkeypatch):
    expected = resolve_nba_season()

    monkeypatch.setattr(DataService, "_initialize", lambda self: None)
    DataService._instance = None
    DataService._initialized = False

    service = DataService()

    assert service._active_season == expected
    assert service._prop_feature_generator.season == expected

    DataService._instance = None
    DataService._initialized = False
