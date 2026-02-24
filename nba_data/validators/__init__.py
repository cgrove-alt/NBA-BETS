"""
Data Validation Layer — Phase 2, Step 3

Provides:
1. Pydantic schemas for API response validation (BallDontLie, TheOddsAPI)
2. Data freshness tracking for prediction transparency
3. Null/NaN checks for required features
"""

from .schemas import (
    BDLPlayerStats,
    BDLGameResponse,
    OddsAPIGame,
    OddsAPIBookmaker,
    ValidatedOdds,
)
from .freshness import DataFreshness
from .null_checks import validate_features, REQUIRED_GAME_FEATURES, REQUIRED_PROP_FEATURES

__all__ = [
    'BDLPlayerStats',
    'BDLGameResponse',
    'OddsAPIGame',
    'OddsAPIBookmaker',
    'ValidatedOdds',
    'DataFreshness',
    'validate_features',
    'REQUIRED_GAME_FEATURES',
    'REQUIRED_PROP_FEATURES',
]
