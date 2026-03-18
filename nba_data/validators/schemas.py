"""
Pydantic schemas for API response validation.

Validates structure and ranges of data from BallDontLie and TheOddsAPI.
Uses permissive validation: logs warnings on unexpected shapes but returns
data anyway to avoid breaking existing callers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# BallDontLie API Schemas
# =============================================================================

class BDLTeam(BaseModel):
    """BallDontLie team object."""
    id: int
    conference: Optional[str] = None
    division: Optional[str] = None
    city: Optional[str] = None
    name: Optional[str] = None
    full_name: Optional[str] = None
    abbreviation: Optional[str] = None


class BDLPlayer(BaseModel):
    """BallDontLie player object."""
    id: int
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    position: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    jersey_number: Optional[str] = None
    college: Optional[str] = None
    country: Optional[str] = None
    draft_year: Optional[int] = None
    draft_round: Optional[int] = None
    draft_number: Optional[int] = None
    team: Optional[BDLTeam] = None


class BDLPlayerStats(BaseModel):
    """BallDontLie player game stats."""
    id: Optional[int] = None
    min: Optional[str] = None
    pts: Optional[int] = None
    reb: Optional[int] = None
    ast: Optional[int] = None
    stl: Optional[int] = None
    blk: Optional[int] = None
    turnover: Optional[int] = None
    fgm: Optional[int] = None
    fga: Optional[int] = None
    fg_pct: Optional[float] = None
    fg3m: Optional[int] = None
    fg3a: Optional[int] = None
    fg3_pct: Optional[float] = None
    ftm: Optional[int] = None
    fta: Optional[int] = None
    ft_pct: Optional[float] = None
    oreb: Optional[int] = None
    dreb: Optional[int] = None
    pf: Optional[int] = None
    player: Optional[BDLPlayer] = None
    team: Optional[BDLTeam] = None
    game: Optional[dict] = None

    @field_validator('pts', 'reb', 'ast', 'stl', 'blk', mode='before')
    @classmethod
    def non_negative_stat(cls, v):
        if v is not None and v < 0:
            logger.warning(f"Negative stat value: {v}")
        return v


class BDLGameResponse(BaseModel):
    """BallDontLie game object."""
    id: int
    date: Optional[str] = None
    season: Optional[int] = None
    status: Optional[str] = None
    home_team: Optional[BDLTeam] = None
    visitor_team: Optional[BDLTeam] = None
    home_team_score: Optional[int] = None
    visitor_team_score: Optional[int] = None


class BDLPaginatedResponse(BaseModel):
    """Wrapper for paginated BDL responses."""
    data: list = Field(default_factory=list)
    meta: Optional[dict] = None


# =============================================================================
# TheOddsAPI Schemas
# =============================================================================

class OddsAPIOutcome(BaseModel):
    """Single outcome in a market."""
    name: str
    price: int  # American odds
    point: Optional[float] = None  # Spread or total line

    @field_validator('price')
    @classmethod
    def odds_in_range(cls, v):
        if abs(v) > 10000:
            logger.warning(f"Extreme odds value: {v}")
        return v


class OddsAPIMarket(BaseModel):
    """Market (spread, totals, h2h) for a bookmaker."""
    key: str  # "spreads", "totals", "h2h"
    last_update: Optional[str] = None
    outcomes: List[OddsAPIOutcome] = Field(default_factory=list)


class OddsAPIBookmaker(BaseModel):
    """Bookmaker with their markets."""
    key: str  # "draftkings", "fanduel", etc.
    title: Optional[str] = None
    last_update: Optional[str] = None
    markets: List[OddsAPIMarket] = Field(default_factory=list)


class OddsAPIGame(BaseModel):
    """Full game with odds from TheOddsAPI."""
    id: Optional[str] = None
    sport_key: Optional[str] = None
    sport_title: Optional[str] = None
    commence_time: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    bookmakers: List[OddsAPIBookmaker] = Field(default_factory=list)


class ValidatedOdds(BaseModel):
    """Cleaned and validated odds for a game, with range checks."""
    home_team: str
    away_team: str
    spread: Optional[float] = None
    total: Optional[float] = None
    home_ml: Optional[int] = None
    away_ml: Optional[int] = None
    bookmaker: Optional[str] = None
    fetched_at: Optional[str] = None

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

def validate_bdl_response(data: Union[dict, list], context: str = "") -> bool:
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


def validate_odds_response(data: Union[dict, list], context: str = "") -> bool:
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
