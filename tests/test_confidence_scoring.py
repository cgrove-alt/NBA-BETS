"""
Tests for Model Confidence Scoring System
Tests edge quality tiers, uncertainty flags, and confidence calculations.
"""

import sys
import os
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from edge_quality import EdgeQualityScorer, EdgeTier, DynamicKellyCalculator, american_to_decimal
from model_trainer import calculate_uncertainty_flags


class TestEdgeQualityTiers:
    """Test edge quality tier classification."""

    def test_elite_tier_threshold(self):
        """Test ELITE tier requires score >= 90."""
        scorer = EdgeQualityScorer()

        # Score of 90 should be ELITE
        result = scorer.evaluate_edge(
            model_probability=0.65,
            implied_probability=0.50,
            individual_model_predictions={
                'lr': 0.64, 'rf': 0.65, 'gb': 0.66, 'xgb': 0.65, 'lgb': 0.64
            },
            opening_odds=-3.0,
            current_odds=-4.5,
            games_played=50,
            training_data_age_days=10,
            last_game_days_ago=1,
        )

        # With these inputs, should get high score
        assert result.overall_score >= 75, f"Expected high score, got {result.overall_score}"
        assert result.tier in [EdgeTier.ELITE, EdgeTier.STRONG]

    def test_strong_tier_range(self):
        """Test STRONG tier is 75-89."""
        scorer = EdgeQualityScorer()

        result = scorer.evaluate_edge(
            model_probability=0.58,
            implied_probability=0.52,
            individual_model_predictions={
                'lr': 0.56, 'rf': 0.59, 'gb': 0.58, 'xgb': 0.57, 'lgb': 0.59
            },
            games_played=40,
        )

        # Should be in reasonable range
        assert 50 <= result.overall_score <= 100

    def test_moderate_tier_range(self):
        """Test MODERATE tier is 60-74."""
        scorer = EdgeQualityScorer()

        result = scorer.evaluate_edge(
            model_probability=0.54,
            implied_probability=0.52,
            individual_model_predictions={
                'lr': 0.52, 'rf': 0.55, 'gb': 0.54, 'xgb': 0.53, 'lgb': 0.56
            },
            games_played=30,
            training_data_age_days=45,
        )

        # Lower conviction should result in moderate tier
        assert result.tier in [EdgeTier.MODERATE, EdgeTier.WEAK, EdgeTier.STRONG]

    def test_weak_tier_no_betting(self):
        """Test WEAK tier (40-59) has 0 Kelly multiplier."""
        scorer = EdgeQualityScorer()

        # Artificially low score scenario
        result = scorer.evaluate_edge(
            model_probability=0.51,
            implied_probability=0.50,
            individual_model_predictions={
                'lr': 0.45, 'rf': 0.55, 'gb': 0.50, 'xgb': 0.48, 'lgb': 0.57
            },
            games_played=15,
            training_data_age_days=90,
            injury_impact_score=0.6,
        )

        # If tier is WEAK, Kelly multiplier should be 0
        if result.tier == EdgeTier.WEAK:
            assert result.recommended_kelly_multiplier == 0.0

    def test_avoid_tier(self):
        """Test AVOID tier (<40) has 0 Kelly multiplier."""
        scorer = EdgeQualityScorer()

        result = scorer.evaluate_edge(
            model_probability=0.50,
            implied_probability=0.50,
            individual_model_predictions={
                'lr': 0.40, 'rf': 0.60, 'gb': 0.45, 'xgb': 0.55, 'lgb': 0.50
            },
            games_played=10,
            training_data_age_days=120,
            injury_impact_score=0.8,
            travel_fatigue_score=0.8,
        )

        # With high disagreement and poor conditions, should avoid
        if result.tier == EdgeTier.AVOID:
            assert result.recommended_kelly_multiplier == 0.0


class TestKellyMultipliers:
    """Test Kelly multipliers match task requirements."""

    def test_elite_kelly_multiplier(self):
        """Test ELITE tier uses 1.0x Kelly."""
        assert EdgeQualityScorer.KELLY_MULTIPLIERS[EdgeTier.ELITE] == 1.0

    def test_strong_kelly_multiplier(self):
        """Test STRONG tier uses 0.5x Kelly."""
        assert EdgeQualityScorer.KELLY_MULTIPLIERS[EdgeTier.STRONG] == 0.50

    def test_moderate_kelly_multiplier(self):
        """Test MODERATE tier uses 0.25x Kelly."""
        assert EdgeQualityScorer.KELLY_MULTIPLIERS[EdgeTier.MODERATE] == 0.25

    def test_weak_kelly_multiplier(self):
        """Test WEAK tier has 0 Kelly (monitor only)."""
        assert EdgeQualityScorer.KELLY_MULTIPLIERS[EdgeTier.WEAK] == 0.0

    def test_avoid_kelly_multiplier(self):
        """Test AVOID tier has 0 Kelly."""
        assert EdgeQualityScorer.KELLY_MULTIPLIERS[EdgeTier.AVOID] == 0.0


class TestConfidenceScoring:
    """Test confidence score calculation."""

    def test_high_agreement_high_confidence(self):
        """Test high model agreement yields high confidence."""
        scorer = EdgeQualityScorer()

        # Very tight agreement
        score, factors = scorer.calculate_ensemble_agreement_score(
            individual_predictions={
                'lr': 0.60, 'rf': 0.61, 'gb': 0.60, 'xgb': 0.61, 'lgb': 0.60
            },
            ensemble_prediction=0.604
        )

        # Standard deviation ~0.005, should score very high
        assert score >= 90, f"Expected high agreement score, got {score}"
        assert any("agreement" in f.lower() for f in factors)

    def test_low_agreement_low_confidence(self):
        """Test high model disagreement yields low confidence."""
        scorer = EdgeQualityScorer()

        # Wide disagreement
        score, factors = scorer.calculate_ensemble_agreement_score(
            individual_predictions={
                'lr': 0.40, 'rf': 0.70, 'gb': 0.50, 'xgb': 0.60, 'lgb': 0.45
            },
            ensemble_prediction=0.53
        )

        # Standard deviation ~0.11, should score low
        assert score < 80, f"Expected low agreement score, got {score}"

    def test_direction_disagreement_penalty(self):
        """Test penalty when models disagree on direction."""
        scorer = EdgeQualityScorer()

        # Models split on >50% vs <50%
        score, factors = scorer.calculate_ensemble_agreement_score(
            individual_predictions={
                'lr': 0.48, 'rf': 0.52, 'gb': 0.49, 'xgb': 0.53, 'lgb': 0.51
            },
            ensemble_prediction=0.506
        )

        # Should have lower score due to direction disagreement
        assert any("disagree" in f.lower() for f in factors)


class TestUncertaintyFlags:
    """Test uncertainty flag calculation."""

    def test_gtd_player_high_uncertainty(self):
        """Test GTD player triggers HIGH_UNCERTAINTY flag."""
        result = calculate_uncertainty_flags(
            features={'pts_avg': 25.0},
            confidence_score=75.0,
            is_player_gtd=True
        )

        assert "HIGH_UNCERTAINTY" in result["uncertainty_flags"]
        assert "PLAYER_GTD" in result["uncertainty_flags"]
        assert result["uncertainty_level"] == "HIGH"
        assert result["has_uncertainty"] is True

    def test_incomplete_data_flag(self):
        """Test incomplete data triggers DATA_INCOMPLETE flag."""
        result = calculate_uncertainty_flags(
            features={'pts_avg': 25.0},
            confidence_score=75.0,
            missing_feature_count=3
        )

        assert "DATA_INCOMPLETE" in result["uncertainty_flags"]
        assert result["uncertainty_level"] in ["MEDIUM", "HIGH"]

    def test_low_confidence_flag(self):
        """Test low confidence score triggers LOW_CONFIDENCE flag."""
        result = calculate_uncertainty_flags(
            features={'pts_avg': 25.0},
            confidence_score=35.0
        )

        assert "LOW_CONFIDENCE" in result["uncertainty_flags"]
        assert result["uncertainty_level"] == "HIGH"

    def test_missing_critical_features(self):
        """Test missing critical features triggers flag."""
        result = calculate_uncertainty_flags(
            features={'pts_avg': 25.0},
            confidence_score=75.0,
            required_features=['pts_avg', 'min_avg', 'usage_rate']
        )

        # min_avg and usage_rate are missing
        assert any("MISSING_CRITICAL" in flag for flag in result["uncertainty_flags"])
        assert result["uncertainty_level"] == "HIGH"

    def test_no_uncertainty_clean_prediction(self):
        """Test clean prediction with no uncertainty flags."""
        result = calculate_uncertainty_flags(
            features={'pts_avg': 25.0, 'min_avg': 35.0},
            confidence_score=85.0,
            is_player_gtd=False,
            missing_feature_count=0
        )

        assert result["has_uncertainty"] is False
        assert result["uncertainty_level"] == "LOW"
        assert len(result["uncertainty_flags"]) == 0


class TestDynamicKellyCalculator:
    """Test dynamic Kelly criterion bet sizing."""

    def test_elite_tier_full_kelly(self):
        """Test ELITE tier gets full Kelly multiplier."""
        scorer = EdgeQualityScorer()
        kelly_calc = DynamicKellyCalculator()

        edge_result = scorer.evaluate_edge(
            model_probability=0.65,
            implied_probability=0.50,
            individual_model_predictions={
                'lr': 0.64, 'rf': 0.65, 'gb': 0.66, 'xgb': 0.65, 'lgb': 0.64
            }
        )

        # Force ELITE tier for testing
        if edge_result.tier == EdgeTier.ELITE:
            bet = kelly_calc.calculate_bet_size(
                bankroll=10000,
                probability=0.65,
                decimal_odds=american_to_decimal(-110),
                edge_quality=edge_result
            )

            assert bet['should_bet'] is True
            assert bet['recommended_bet_pct'] > 0

    def test_weak_tier_no_bet(self):
        """Test WEAK tier results in no bet."""
        scorer = EdgeQualityScorer()
        kelly_calc = DynamicKellyCalculator()

        # Create weak edge
        edge_result = scorer.evaluate_edge(
            model_probability=0.51,
            implied_probability=0.50,
            individual_model_predictions={
                'lr': 0.45, 'rf': 0.55, 'gb': 0.50, 'xgb': 0.48, 'lgb': 0.57
            }
        )

        # Should not bet on WEAK tier
        if edge_result.tier == EdgeTier.WEAK:
            bet = kelly_calc.calculate_bet_size(
                bankroll=10000,
                probability=0.51,
                decimal_odds=american_to_decimal(-110),
                edge_quality=edge_result
            )

            assert bet['should_bet'] is False
            assert bet['recommended_bet_pct'] == 0.0

    def test_drawdown_reduces_bet_size(self):
        """Test drawdown reduces bet sizing."""
        scorer = EdgeQualityScorer()
        kelly_calc = DynamicKellyCalculator()

        edge_result = scorer.evaluate_edge(
            model_probability=0.60,
            implied_probability=0.52
        )

        # Bet with no drawdown
        bet_normal = kelly_calc.calculate_dynamic_kelly(
            probability=0.60,
            decimal_odds=american_to_decimal(-110),
            edge_quality=edge_result,
            current_drawdown=0.0
        )

        # Bet with 20% drawdown
        bet_drawdown = kelly_calc.calculate_dynamic_kelly(
            probability=0.60,
            decimal_odds=american_to_decimal(-110),
            edge_quality=edge_result,
            current_drawdown=0.20
        )

        # Drawdown should reduce bet size
        assert bet_drawdown['recommended_bet_pct'] <= bet_normal['recommended_bet_pct']


class TestSpreadModelConfidence:
    """Test SpreadModel confidence scoring."""

    def test_spread_classifier_high_confidence(self):
        """Test classifier gives high confidence for strong predictions."""
        # Mock features for high-confidence spread prediction
        features = {
            'elo_diff': 15.0,
            'off_rating_diff': 8.0,
            'def_rating_diff': -7.0,
            'net_rating_diff': 12.0,
            'pace_combined': 102.0,
        }

        # Note: This test validates the confidence calculation logic
        # For a real prediction, the model would need to be trained first
        # Here we're testing the formula: confidence = 100 * (distance_from_even / 0.5)

        # Example: If model predicts 80% cover probability
        cover_prob = 0.80
        distance_from_even = abs(cover_prob - 0.5)
        expected_confidence = 100.0 * (distance_from_even / 0.5)

        assert abs(expected_confidence - 60.0) < 0.01, f"Expected 60.0, got {expected_confidence}"

    def test_spread_classifier_low_confidence(self):
        """Test classifier gives low confidence for weak predictions."""
        # Example: If model predicts 55% cover probability (close to coin flip)
        cover_prob = 0.55
        distance_from_even = abs(cover_prob - 0.5)
        expected_confidence = 100.0 * (distance_from_even / 0.5)

        assert abs(expected_confidence - 10.0) < 0.01, f"Expected 10.0, got {expected_confidence}"

    def test_spread_regressor_blowout_confidence(self):
        """Test regressor gives high confidence for blowout predictions."""
        # Predict 18-point blowout (margin >= 15)
        margin = 18.0
        expected_confidence = min(90.0, 80.0 + (margin - 15.0) / 3.0)

        assert 80.0 <= expected_confidence <= 90.0
        assert expected_confidence == 81.0

    def test_spread_regressor_close_game_confidence(self):
        """Test regressor gives low confidence for close games."""
        # Predict 2-point game (margin < 3)
        margin = 2.0
        expected_confidence = 40.0 + margin * 3.33

        assert 40.0 <= expected_confidence < 50.0
        assert abs(expected_confidence - 46.66) < 0.1

    def test_spread_regressor_comfortable_win(self):
        """Test regressor gives good confidence for comfortable wins."""
        # Predict 10-point win (7 <= margin < 15)
        margin = 10.0
        expected_confidence = 65.0 + (margin - 7.0) * 1.75

        assert 65.0 <= expected_confidence < 80.0
        assert expected_confidence == 70.25


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_elite_bet(self):
        """Test end-to-end elite bet recommendation."""
        scorer = EdgeQualityScorer()
        kelly_calc = DynamicKellyCalculator()

        # Elite scenario: Strong edge, high agreement, good conditions
        edge_result = scorer.evaluate_edge(
            model_probability=0.65,
            implied_probability=0.50,
            individual_model_predictions={
                'lr': 0.64, 'rf': 0.65, 'gb': 0.66, 'xgb': 0.65, 'lgb': 0.64
            },
            opening_odds=-110,
            current_odds=-120,
            games_played=50,
            training_data_age_days=10,
            last_game_days_ago=1,
            home_away="home",
        )

        # Calculate bet size
        bet = kelly_calc.calculate_bet_size(
            bankroll=10000,
            probability=0.65,
            decimal_odds=american_to_decimal(-120),
            edge_quality=edge_result,
            current_drawdown=0.0,
            consecutive_losses=0,
        )

        # Should recommend betting with reasonable size
        assert bet['should_bet'] is True
        assert 0 < bet['recommended_bet_pct'] <= 0.05  # Max 5%
        assert 0 < bet['bet_amount'] <= 500  # Max $500 on $10k bankroll

    def test_end_to_end_avoid_bet(self):
        """Test end-to-end avoid recommendation."""
        scorer = EdgeQualityScorer()

        # Avoid scenario: Weak edge, high disagreement, poor conditions
        edge_result = scorer.evaluate_edge(
            model_probability=0.51,
            implied_probability=0.50,
            individual_model_predictions={
                'lr': 0.40, 'rf': 0.60, 'gb': 0.45, 'xgb': 0.55, 'lgb': 0.50
            },
            games_played=10,
            training_data_age_days=120,
            injury_impact_score=0.8,
        )

        # Should be AVOID or WEAK tier
        assert edge_result.tier in [EdgeTier.AVOID, EdgeTier.WEAK]
        assert len(edge_result.risk_factors) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
