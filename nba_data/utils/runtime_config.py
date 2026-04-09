from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path


def get_repo_root() -> Path:
    """Return the repository root regardless of the caller's package depth."""
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return the canonical data directory, with optional environment override."""
    override = os.getenv("NBA_BETS_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return get_repo_root() / "data"


def get_historical_csv_dir() -> Path:
    """Return the canonical historical CSV dataset directory."""
    return get_data_dir() / "NBA-Data-2010-2024-main"


def get_live_seasons_dir() -> Path:
    """Return the canonical live seasons cache directory."""
    return get_data_dir() / "live_seasons"


def resolve_nba_season(
    season: str | None = None,
    *,
    as_of: date | datetime | None = None,
) -> str:
    """Resolve the active NBA season from an explicit value, env override, or date."""
    if season:
        return season

    override = os.getenv("NBA_BETS_SEASON")
    if override:
        return override

    if as_of is None:
        as_of = datetime.now()

    if isinstance(as_of, datetime):
        as_of = as_of.date()

    start_year = as_of.year if as_of.month >= 10 else as_of.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


DEFAULT_NBA_SEASON = resolve_nba_season()
