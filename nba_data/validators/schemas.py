"""
Pydantic schemas for API response validation.

Validates structure and ranges of data from BallDontLie and TheOddsAPI.
Uses permissive validation: logs warnings on unexpected shapes but returns
data anyway to avoid breaking existing callers.
"""

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# BallDontLie API Schemas
# =============================================================================

class BDLTeam(BaseModel):
    """BallDontLie team object."""
    id: int
    conference: str | None = None
    division: str | None = None
    city: str | None = None
    name: str | None = None
    full_name: str | None = None
    abbreviation: str | None = None


class BDLPlayer(BaseModel):
    """BallDontLie player object."""
    id: int
    first_name: str | None = None
    last_name: str | None = None
    position: str | None = None
    height: str | None = None
    weight: str | None = None
    jersey_number: str | None = None
    college: str | None = None
    country: str | None = None
    draft_year: int | None = None
    draft_round: int | None = None
    draft_number: int | None = None
    team: BDLTeam | None = None


class BDLPlayerStats(BaseModel):
    """BallDontLie player game stats."""
    id: int | None = None
    min: str | None = None
    pts: int | None = None
    reb: int | None = None
    ast: int | None = None
    stl: int | None = None
    blk: int | None = None
    turnover: int | None = None
    fgm: int | None = None
    fga: int | None = None
    fg_pct: float | None = None
    fg3m: int | None = None
    fg3a: int | None = None
    fg3_pct: float | None = None
    ftm: int | None = None
    fta: int | None = None
    ft_pct: float | None = None
    oreb: int | None = None
    dreb: int | None = None
    pf: int | None = None
    player: BDLPlayer | None = None
    team: BDLTeam | None = None
    game: dict | None = None

    @field_validator('pts', 'reb', 'ast', 'stl', 'blk', mode='before')
    @classmethod
    def non_negative_stat(cls, v):
        if v is not None and v < 0:
            logger.warning(f"Negative stat value: {v}")
        return v


class BDLGameResponse(BaseModel):
    """BallDontLie game object."""
    id: int
    date: str | None = None
    season: int | None = None
    status: str | None = None
    home_team: BDLTeam | None = None
    visitor_team: BDLTeam | None = None
    home_team_score: int | None = None
    visitor_team_score: int | None = None


class BDLPaginatedResponse(BaseModel):
    """Wrapper for paginated BDL responses."""
    data: list = Field(default_factory=list)
    meta: dict | None = None


# =============================================================================
# TheOddsAPI Schemas
# =============================================================================

class OddsAPIOutcome(BaseModel):
    """Single outcome in a market."""
    name: str
    price: int  # American odds
    point: float | None = None  # Spread or total line

    @field_validator('price')
    @classmethod
    def odds_in_range(cls, v):
        if abs(v) > 10000:
            logger.warning(f"Extreme odds value: {v}")
        return v


class OddsAPIMarket(BaseModel):
    """Market (spread, totals, h2h) for a bookmaker."""
    key: str  # "spreads", "totals", "h2h"
    last_update: str | None = None
    outcomes: list[OddsAPIOutcome] = Field(default_factory=list)


class OddsAPIBookmaker(BaseModel):
    """Bookmaker with their markets."""
    key: str  # "draftkings", "fanduel", etc.
    title: str | None = None
    last_update: str | None = None
    markets: list[OddsAPIMarket] = Field(default_factory=list)


class OddsAPIGame(BaseModel):
    """Full game with odds from TheOddsAPI."""
    id: str | None = None
    sport_key: str | None = None
    sport_title: str | None = None
    commence_time: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    bookmakers: list[OddsAPIBookmaker] = Field(default_factory=list)


class ValidatedOdds(BaseModel):
    """Cleaned and validated odds for a game, with range checks."""
    home_team: str
    away_team: str
    spread: float | None = None
    total: float | None = None
    home_ml: int | None = None
    away_ml: int | None = None
    bookmaker: str | None = None
    fetched_at: str | None = None

    @field_validator('spread')
    @classmethod
    def spread_in_range(cls, v):
        if v is not None and abs(v) > 30:
            logger.warning(f"Extreme spread value: {v}")
        return v

    @field_validator('total')
    @classmethod
    def total_in_range(cls, v):
        if v is not None and (v < 150 or v > 300):
            logger.warning(f"Unusual total value: {v}")
        return v

    @field_validator('home_ml', 'away_ml')
    @classmethod
    def moneyline_nonzero(cls, v):
        if v is not None and v == 0:
            logger.warning("Moneyline of 0 is invalid")
        return v


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_bdl_response(data: dict | list, context: str = "") -> bool:
    """
    Validate a BallDontLie API response shape.
    Returns True if valid, False if unexpected. Logs warnings, never raises.
    """
    if data is None:
        logger.warning(f"BDL response is None ({context})")
        return False

    if isinstance(data, dict):
        if 'data' in data:
            if not isinstance(data['data'], list):
                logger.warning(f"BDL 'data' field is not a list ({context}): {type(data['data'])}")
                return False
        return True

    if isinstance(data, list):
        return True

    logger.warning(f"BDL unexpected response type ({context}): {type(data)}")
    return False


def validate_odds_response(data: dict | list, context: str = "") -> bool:
    """
    Validate a TheOddsAPI response shape.
    Returns True if valid, False if unexpected. Logs warnings, never raises.
    """
    if data is None:
        logger.warning(f"OddsAPI response is None ({context})")
        return False

    if not isinstance(data, (dict, list)):
        logger.warning(f"OddsAPI unexpected response type ({context}): {type(data)}")
        return False

    return True
