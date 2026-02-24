"""
Tests for data validation layer (Phase 2, Step 3).

Covers:
- Pydantic schema validation (valid/invalid responses)
- Data freshness tracking (stale/fresh detection)
- Null check validation (missing/NaN required features)
- Utility functions
"""

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Schema Validation Tests
# =============================================================================

class TestBDLSchemas:
    """Test BallDontLie API response schemas."""

    def test_valid_player_stats(self):
        from nba_data.validators.schemas import BDLPlayerStats
        stats = BDLPlayerStats(
            id=1, pts=25, reb=10, ast=5, min="34:20",
            stl=2, blk=1, fg_pct=0.45
        )
        assert stats.pts == 25
        assert stats.reb == 10

    def test_player_stats_with_nulls(self):
        """Schema should accept missing optional fields."""
        from nba_data.validators.schemas import BDLPlayerStats
        stats = BDLPlayerStats(id=1)
        assert stats.pts is None
        assert stats.reb is None

    def test_game_response(self):
        from nba_data.validators.schemas import BDLGameResponse
        game = BDLGameResponse(
            id=12345,
            date="2026-02-23",
            season=2025,
            status="Final",
            home_team_score=110,
            visitor_team_score=105,
        )
        assert game.id == 12345
        assert game.home_team_score == 110

    def test_validate_bdl_response_valid_dict(self):
        from nba_data.validators.schemas import validate_bdl_response
        assert validate_bdl_response({'data': [1, 2, 3]}) is True

    def test_validate_bdl_response_valid_list(self):
        from nba_data.validators.schemas import validate_bdl_response
        assert validate_bdl_response([1, 2, 3]) is True

    def test_validate_bdl_response_none(self):
        from nba_data.validators.schemas import validate_bdl_response
        assert validate_bdl_response(None) is False

    def test_validate_bdl_response_string(self):
        from nba_data.validators.schemas import validate_bdl_response
        assert validate_bdl_response("not a dict") is False

    def test_validate_bdl_response_bad_data_field(self):
        from nba_data.validators.schemas import validate_bdl_response
        assert validate_bdl_response({'data': "not a list"}) is False


class TestOddsSchemas:
    """Test TheOddsAPI response schemas."""

    def test_valid_odds_outcome(self):
        from nba_data.validators.schemas import OddsAPIOutcome
        outcome = OddsAPIOutcome(name="Boston Celtics", price=-110, point=-5.5)
        assert outcome.price == -110
        assert outcome.point == -5.5

    def test_validated_odds_normal(self):
        from nba_data.validators.schemas import ValidatedOdds
        odds = ValidatedOdds(
            home_team="BOS", away_team="LAL",
            spread=-5.5, total=220.5,
            home_ml=-200, away_ml=170,
        )
        assert odds.spread == -5.5
        assert odds.total == 220.5

    def test_validated_odds_extreme_spread_warns(self, caplog):
        """Spread > 30 should log a warning but still validate."""
        import logging
        from nba_data.validators.schemas import ValidatedOdds
        with caplog.at_level(logging.WARNING):
            odds = ValidatedOdds(
                home_team="BOS", away_team="LAL",
                spread=-35.0,
            )
        assert odds.spread == -35.0
        assert "Extreme spread" in caplog.text

    def test_validated_odds_unusual_total_warns(self, caplog):
        """Total outside 150-300 should warn."""
        import logging
        from nba_data.validators.schemas import ValidatedOdds
        with caplog.at_level(logging.WARNING):
            odds = ValidatedOdds(
                home_team="BOS", away_team="LAL",
                total=350.0,
            )
        assert odds.total == 350.0
        assert "Unusual total" in caplog.text

    def test_validate_odds_response_valid(self):
        from nba_data.validators.schemas import validate_odds_response
        assert validate_odds_response([{'id': 'abc'}]) is True
        assert validate_odds_response({'data': 123}) is True

    def test_validate_odds_response_invalid(self):
        from nba_data.validators.schemas import validate_odds_response
        assert validate_odds_response(None) is False
        assert validate_odds_response("string") is False


# =============================================================================
# Freshness Tracking Tests
# =============================================================================

class TestDataFreshness:
    """Test DataFreshness tracking."""

    def test_initial_state(self):
        from nba_data.validators.freshness import DataFreshness
        f = DataFreshness()
        assert f.odds_fetched_at is None
        assert f.stats_fetched_at is None
        assert f.injuries_fetched_at is None

    def test_record_timestamps(self):
        from nba_data.validators.freshness import DataFreshness
        f = DataFreshness()
        f.record_odds_fetch()
        f.record_stats_fetch()
        f.record_injuries_fetch()
        assert f.odds_fetched_at is not None
        assert f.stats_fetched_at is not None
        assert f.injuries_fetched_at is not None

    def test_fresh_data_not_stale(self):
        from nba_data.validators.freshness import DataFreshness
        f = DataFreshness()
        f.record_odds_fetch()
        f.record_stats_fetch()
        f.record_injuries_fetch()
        assert f.is_stale() is False

    def test_stale_odds_detection(self):
        from nba_data.validators.freshness import DataFreshness
        f = DataFreshness()
        f.odds_fetched_at = datetime.now() - timedelta(seconds=600)
        f.record_stats_fetch()
        f.record_injuries_fetch()
        assert f.is_stale(max_odds_age_sec=300) is True

    def test_stale_stats_detection(self):
        from nba_data.validators.freshness import DataFreshness
        f = DataFreshness()
        f.record_odds_fetch()
        f.stats_fetched_at = datetime.now() - timedelta(minutes=120)
        f.record_injuries_fetch()
        assert f.is_stale(max_stats_age_min=60) is True

    def test_unfetched_is_not_stale(self):
        """Data that was never fetched should not be reported as stale.
        The stale_sources method should report it as 'never fetched'."""
        from nba_data.validators.freshness import DataFreshness
        f = DataFreshness()
        # Never fetched, not stale (stale = fetched but too old)
        assert f.is_stale() is False

    def test_stale_sources_lists_issues(self):
        from nba_data.validators.freshness import DataFreshness
        f = DataFreshness()
        sources = f.stale_sources()
        assert any('never fetched' in s for s in sources)

    def test_to_dict_structure(self):
        from nba_data.validators.freshness import DataFreshness
        f = DataFreshness()
        f.record_odds_fetch()
        d = f.to_dict()
        assert 'odds_fetched_at' in d
        assert 'odds_age_seconds' in d
        assert 'is_stale' in d
        assert 'checked_at' in d
        assert d['odds_age_seconds'] >= 0

    def test_to_dict_unfetched(self):
        from nba_data.validators.freshness import DataFreshness
        f = DataFreshness()
        d = f.to_dict()
        assert d['odds_fetched_at'] is None
        assert d['odds_age_seconds'] == -1


# =============================================================================
# Null/NaN Check Tests
# =============================================================================

class TestNullChecks:
    """Test feature null/NaN validation."""

    def test_valid_features_pass(self):
        from nba_data.validators.null_checks import validate_features
        features = {
            'net_rating_diff': 5.0,
            'win_pct_diff': 0.1,
            'off_rating_diff': 3.0,
            'def_rating_diff': -2.0,
            'pace_diff': 1.5,
        }
        is_valid, issues = validate_features(features)
        assert is_valid is True
        assert issues == []

    def test_missing_feature_detected(self):
        from nba_data.validators.null_checks import validate_features
        features = {'net_rating_diff': 5.0}  # Missing others
        is_valid, issues = validate_features(features)
        assert is_valid is False
        assert any("missing" in i for i in issues)

    def test_none_value_detected(self):
        from nba_data.validators.null_checks import validate_features
        features = {
            'net_rating_diff': None,
            'win_pct_diff': 0.1,
            'off_rating_diff': 3.0,
            'def_rating_diff': -2.0,
            'pace_diff': 1.5,
        }
        is_valid, issues = validate_features(features)
        assert is_valid is False
        assert any("null/NaN" in i for i in issues)

    def test_nan_value_detected(self):
        from nba_data.validators.null_checks import validate_features
        features = {
            'net_rating_diff': float('nan'),
            'win_pct_diff': 0.1,
            'off_rating_diff': 3.0,
            'def_rating_diff': -2.0,
            'pace_diff': 1.5,
        }
        is_valid, issues = validate_features(features)
        assert is_valid is False
        assert any("null/NaN" in i for i in issues)

    def test_inf_value_detected(self):
        from nba_data.validators.null_checks import validate_features
        features = {
            'net_rating_diff': float('inf'),
            'win_pct_diff': 0.1,
            'off_rating_diff': 3.0,
            'def_rating_diff': -2.0,
            'pace_diff': 1.5,
        }
        is_valid, issues = validate_features(features)
        assert is_valid is False

    def test_custom_required_features(self):
        from nba_data.validators.null_checks import validate_features
        features = {'avg_minutes': 32.5, 'avg_stat': 24.0}
        is_valid, issues = validate_features(
            features, required=['avg_minutes', 'avg_stat']
        )
        assert is_valid is True

    def test_count_valid_features(self):
        from nba_data.validators.null_checks import count_valid_features
        features = {
            'a': 1.0,
            'b': None,
            'c': float('nan'),
            'd': 5.0,
            'e': float('inf'),
        }
        valid, total = count_valid_features(features)
        assert total == 5
        assert valid == 2  # Only 'a' and 'd'

    def test_empty_features(self):
        from nba_data.validators.null_checks import validate_features
        is_valid, issues = validate_features({})
        assert is_valid is False

    def test_all_features_present_with_zero_values(self):
        """Zero is a valid value, not missing."""
        from nba_data.validators.null_checks import validate_features
        features = {
            'net_rating_diff': 0.0,
            'win_pct_diff': 0.0,
            'off_rating_diff': 0.0,
            'def_rating_diff': 0.0,
            'pace_diff': 0.0,
        }
        is_valid, issues = validate_features(features)
        assert is_valid is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestValidatorsPackage:
    """Test that the validators package imports work correctly."""

    def test_package_imports(self):
        from nba_data.validators import (
            BDLPlayerStats,
            BDLGameResponse,
            OddsAPIGame,
            OddsAPIBookmaker,
            ValidatedOdds,
            DataFreshness,
            validate_features,
            REQUIRED_GAME_FEATURES,
            REQUIRED_PROP_FEATURES,
        )
        # All imports succeed
        assert BDLPlayerStats is not None
        assert DataFreshness is not None
        assert len(REQUIRED_GAME_FEATURES) > 0
        assert len(REQUIRED_PROP_FEATURES) > 0
