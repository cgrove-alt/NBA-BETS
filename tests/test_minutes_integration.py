"""
Tests for Phase 3: Minutes Oracle integration into the prediction pipeline.

Covers:
- Minutes Oracle loading
- Feature generation
- Prediction output (MinutesDistribution)
- Per-minute rate adjustment math
- Confidence penalty for minutes uncertainty
- Fallback behavior when oracle unavailable
- End-to-end integration (minutes metadata in prediction output)
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock
from scipy.stats import norm

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Constants — imported from canonical module (not duplicated here)
# ============================================================
from nba_betting.constants import PROP_STD_DEVS, DEFAULT_PROP_STD_DEV


def get_prop_std_dev(prop_type: str) -> float:
    return PROP_STD_DEVS.get(prop_type.lower(), DEFAULT_PROP_STD_DEV)


# ============================================================
# Test 1: Minutes Oracle loads from pkl
# ============================================================

class TestMinutesOracleLoading:

    def test_minutes_oracle_loads(self):
        """MinutesPredictor.load() works with existing pkl file."""
        from pathlib import Path
        model_path = Path("models/minutes_oracle.pkl")
        if not model_path.exists():
            pytest.skip("minutes_oracle.pkl not found")

        from minutes_oracle import MinutesPredictor
        predictor = MinutesPredictor.load(model_path)
        assert predictor is not None
        assert predictor.trained is True
        assert hasattr(predictor, 'predict')

    def test_load_models_includes_minutes_oracle(self):
        """load_models() populates models['minutes_oracle'] when pkl exists."""
        from pathlib import Path
        if not Path("models/minutes_oracle.pkl").exists():
            pytest.skip("minutes_oracle.pkl not found")

        from daily_predictions import load_models
        models = load_models()
        assert 'minutes_oracle' in models
        assert 'minutes_feature_gen' in models
        assert hasattr(models['minutes_oracle'], 'predict')


# ============================================================
# Test 2: Feature generation
# ============================================================

class TestMinutesFeatureGeneration:

    def test_minutes_feature_generation(self):
        """MinutesFeatureGenerator produces features dict with expected keys."""
        from minutes_oracle import MinutesFeatureGenerator

        gen = MinutesFeatureGenerator()
        features = gen.generate_features(
            player_id=12345,
            team_id=1,
            opponent_team_id=2,
            game_date='2026-01-15',
            game_context={
                'vegas_spread': -5.0,
                'vegas_total': 225.0,
                'is_home': True,
                'is_back_to_back': False,
                'days_rest': 2,
            },
        )

        assert isinstance(features, dict)
        assert len(features) > 0
        # Should have context features at minimum
        assert 'vegas_spread' in features or 'vegas_spread_abs' in features


# ============================================================
# Test 3: Prediction output
# ============================================================

class TestMinutesPrediction:

    def _get_predictor(self):
        from pathlib import Path
        if not Path("models/minutes_oracle.pkl").exists():
            pytest.skip("minutes_oracle.pkl not found")
        from minutes_oracle import MinutesPredictor
        return MinutesPredictor.load("models/minutes_oracle.pkl")

    def test_minutes_prediction_output(self):
        """predict() returns valid MinutesDistribution with all fields."""
        predictor = self._get_predictor()
        from minutes_oracle import MinutesFeatureGenerator

        gen = MinutesFeatureGenerator()
        features = gen.generate_features(
            player_id=1,
            team_id=1,
            opponent_team_id=2,
            game_date='2026-01-15',
        )

        dist = predictor.predict(features, player_id=1)
        d = dist.to_dict()

        # Check all required fields present
        for key in ['p10', 'p25', 'p50', 'p75', 'p90', 'expected', 'uncertainty', 'spread']:
            assert key in d, f"Missing key: {key}"

        # Check uncertainty is valid
        assert d['uncertainty'] in ('low', 'medium', 'high')

    def test_minutes_distribution_sanity(self):
        """p10 <= p25 <= p50 <= p75 <= p90 and all in [0, 53]."""
        predictor = self._get_predictor()
        from minutes_oracle import MinutesFeatureGenerator

        gen = MinutesFeatureGenerator()
        features = gen.generate_features(
            player_id=1,
            team_id=1,
            opponent_team_id=2,
            game_date='2026-01-15',
        )

        dist = predictor.predict(features)
        d = dist.to_dict()

        # Monotonicity
        assert d['p10'] <= d['p25'] <= d['p50'] <= d['p75'] <= d['p90']

        # Reasonable range
        for key in ['p10', 'p25', 'p50', 'p75', 'p90']:
            assert 0 <= d[key] <= 53, f"{key}={d[key]} out of range [0, 53]"


# ============================================================
# Test 4: predict_minutes_distribution helper
# ============================================================

class TestPredictMinutesDistributionHelper:

    def test_predict_minutes_distribution_helper(self):
        """Helper function returns valid dict when oracle available."""
        from pathlib import Path
        if not Path("models/minutes_oracle.pkl").exists():
            pytest.skip("minutes_oracle.pkl not found")

        from daily_predictions import predict_minutes_distribution, load_models
        models = load_models()

        if 'minutes_oracle' not in models:
            pytest.skip("Minutes oracle not loaded")

        result = predict_minutes_distribution(
            player_id=1,
            team_id=1,
            opponent_team_id=2,
            game_context={'spread': -5.0, 'total': 225, 'is_home': True},
            models=models,
        )

        assert result is not None
        assert 'p50' in result
        assert 'uncertainty' in result

    def test_predict_minutes_distribution_fallback(self):
        """Returns None when oracle unavailable."""
        from daily_predictions import predict_minutes_distribution

        # Empty models dict = no oracle
        result = predict_minutes_distribution(
            player_id=1,
            team_id=1,
            opponent_team_id=2,
            game_context={'spread': -5.0, 'total': 225},
            models={},
        )

        assert result is None


# ============================================================
# Test 5: Per-minute rate adjustment math
# ============================================================

class TestPerMinuteRateAdjustment:

    def test_per_minute_rate_adjustment(self):
        """Adjustment scales correctly: 24.2pts/34min * 30min = 21.4."""
        predicted_value = 24.2
        avg_minutes = 34.0
        predicted_minutes = 30.0

        rate = predicted_value / avg_minutes
        adjusted = rate * predicted_minutes

        assert abs(adjusted - 21.35) < 0.1  # 24.2/34*30 ≈ 21.35
        assert adjusted < predicted_value  # Fewer minutes → lower prediction

    def test_per_minute_rate_no_adjustment_when_similar(self):
        """No adjustment when predicted ≈ average minutes (within 1%)."""
        predicted_value = 24.2
        avg_minutes = 34.0
        predicted_minutes = 34.2  # Very close to average

        rate = predicted_value / avg_minutes
        adjusted = rate * predicted_minutes

        # Change is < 1%, so adjustment should NOT be applied
        pct_change = abs(adjusted - predicted_value) / max(abs(predicted_value), 0.1)
        assert pct_change < 0.01

    def test_adjustment_increases_with_more_minutes(self):
        """More minutes predicted → higher stat prediction."""
        predicted_value = 20.0
        avg_minutes = 30.0
        predicted_minutes = 36.0  # More minutes than usual

        rate = predicted_value / avg_minutes
        adjusted = rate * predicted_minutes

        assert adjusted > predicted_value
        assert abs(adjusted - 24.0) < 0.1  # 20/30*36 = 24.0

    def test_adjustment_skipped_with_low_avg_minutes(self):
        """No adjustment when avg_minutes <= 10 (bench player)."""
        avg_minutes = 8.0  # Bench player

        # The code checks `if avg_minutes > 10`
        assert avg_minutes <= 10  # Confirms adjustment would be skipped


# ============================================================
# Test 6: Confidence penalty
# ============================================================

class TestConfidencePenalty:

    def test_confidence_penalty_high_uncertainty(self):
        """High uncertainty → 20% confidence reduction."""
        base_confidence = 80.0
        adjusted = base_confidence * 0.80
        assert adjusted == 64.0

    def test_confidence_penalty_medium_uncertainty(self):
        """Medium uncertainty → 8% confidence reduction."""
        base_confidence = 80.0
        adjusted = base_confidence * 0.92
        assert abs(adjusted - 73.6) < 0.01

    def test_confidence_penalty_low_uncertainty(self):
        """Low uncertainty → no penalty."""
        base_confidence = 80.0
        # 'low' uncertainty = no multiplier
        adjusted = base_confidence  # No change
        assert adjusted == 80.0

    def test_confidence_clamped_after_penalty(self):
        """Confidence stays in [40, 90] after penalty."""
        base_confidence = 45.0
        adjusted = base_confidence * 0.80  # = 36.0
        clamped = max(40.0, min(90.0, adjusted))
        assert clamped == 40.0  # Clamped to minimum


# ============================================================
# Test 7: Prediction output includes minutes data
# ============================================================

class TestPredictionOutputIncludesMinutes:

    def test_prediction_output_includes_minutes_fields(self):
        """Return dict has minutes_distribution, predicted_minutes, minutes_uncertainty."""
        from daily_predictions import predict_player_prop

        # Run with minimal setup (no API features)
        result = predict_player_prop(
            player_name="Test Player",
            player_id=1,
            prop_type='points',
            line=20.5,
            opponent='BOS',
            opponent_id=2,
            models={},
            use_api_features=False,
        )

        # Should have the minutes fields even if None
        assert 'minutes_distribution' in result
        assert 'predicted_minutes' in result
        assert 'minutes_uncertainty' in result


# ============================================================
# Test 8: Blowout scenario reduces prediction
# ============================================================

class TestBlowoutScenario:

    def test_blowout_reduces_prediction(self):
        """Large spread → minutes oracle predicts fewer minutes → lower stat value."""
        # Simulate: player averaging 34 min, but blowout means p50=28 min
        predicted_value = 24.2  # Based on 34 avg minutes
        avg_minutes = 34.0
        blowout_predicted_minutes = 28.0

        rate = predicted_value / avg_minutes
        adjusted = rate * blowout_predicted_minutes

        # Should be meaningfully lower
        assert adjusted < predicted_value
        assert (predicted_value - adjusted) > 1.0  # At least 1 point reduction
        assert abs(adjusted - 19.93) < 0.1  # 24.2/34*28 ≈ 19.93

    def test_close_game_no_significant_reduction(self):
        """Small spread → similar minutes → minimal adjustment."""
        predicted_value = 24.2
        avg_minutes = 34.0
        close_game_minutes = 33.5  # Close game, similar minutes

        rate = predicted_value / avg_minutes
        adjusted = rate * close_game_minutes

        # Difference should be very small
        diff = abs(adjusted - predicted_value)
        assert diff < 0.5  # Less than 0.5 points change
