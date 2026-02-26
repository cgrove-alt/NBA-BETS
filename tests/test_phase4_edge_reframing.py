"""
Phase 4 Tests — Edge-Focused Prediction Reframing

Tests cover:
1. _calculate_prop_edge() — OVER/UNDER edge, coin flip, legacy compat, different odds
2. get_signal_from_edge() — BET/LEAN/PASS/FADE signal mapping
3. predict_player_prop() — enriched output dict, backward compat
4. CLV bridge — recording BET signals, skipping PASS
5. Prediction logging — CalibrationService integration
6. Fallback when EdgeCalculator unavailable
"""

import os
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Test 1-6: _calculate_prop_edge
# ============================================================

class TestCalculatePropEdge:
    """Tests for the _calculate_prop_edge helper."""

    def test_calculate_prop_edge_over(self):
        """over_prob=0.6 → pick=OVER, positive over_edge."""
        from nba_models.inference.daily_predictions import _calculate_prop_edge

        result = _calculate_prop_edge(0.6, -110)
        assert result['pick'] == 'OVER'
        assert result['over_edge'] > 0
        assert result['edge'] > 0
        assert result['model_probability'] == pytest.approx(0.6, abs=0.01)

    def test_calculate_prop_edge_under(self):
        """over_prob=0.4 → pick=UNDER, positive under_edge."""
        from nba_models.inference.daily_predictions import _calculate_prop_edge

        result = _calculate_prop_edge(0.4, -110)
        assert result['pick'] == 'UNDER'
        assert result['under_edge'] > 0
        assert result['edge'] > 0

    def test_calculate_prop_edge_coin_flip(self):
        """over_prob=0.5 → near-zero edge both sides."""
        from nba_models.inference.daily_predictions import _calculate_prop_edge

        result = _calculate_prop_edge(0.5, -110)
        assert abs(result['over_edge']) < 5
        assert abs(result['under_edge']) < 5
        # Both edges should be negative (vig eats them)
        assert result['over_edge'] < 0
        assert result['under_edge'] < 0

    def test_edge_at_minus_110_matches_legacy(self):
        """At -110 odds, new calc should be close to old (over_prob-0.524)*100."""
        from nba_models.inference.daily_predictions import _calculate_prop_edge

        for prob in [0.55, 0.60, 0.65, 0.70]:
            result = _calculate_prop_edge(prob, -110)
            legacy_edge = (prob - 0.524) * 100  # old formula
            # The over_edge should be close to legacy (within 0.5%)
            assert abs(result['over_edge'] - legacy_edge) < 0.5, \
                f"prob={prob}: new={result['over_edge']:.2f} vs legacy={legacy_edge:.2f}"

    def test_edge_at_different_odds(self):
        """Harder odds (-120) should give less edge than standard (-110) for same prob."""
        from nba_models.inference.daily_predictions import _calculate_prop_edge

        prob = 0.58
        result_110 = _calculate_prop_edge(prob, -110)
        result_120 = _calculate_prop_edge(prob, -120)

        # At -120, implied probability is higher (~54.5% vs 52.4%), so edge is smaller
        assert result_120['over_edge'] < result_110['over_edge']

    def test_over_under_edge_sum_negative(self):
        """over_edge + under_edge < 0 due to vig."""
        from nba_models.inference.daily_predictions import _calculate_prop_edge

        result = _calculate_prop_edge(0.55, -110)
        total = result['over_edge'] + result['under_edge']
        assert total < 0, f"Expected negative sum (vig), got {total}"


# ============================================================
# Test 7-10: get_signal_from_edge
# ============================================================

class TestGetSignalFromEdge:
    """Tests for signal classification."""

    def test_signal_bet(self):
        """Strong edge quality → BET."""
        from nba_models.inference.daily_predictions import get_signal_from_edge

        assert get_signal_from_edge(6.0, 'strong') == 'BET'
        assert get_signal_from_edge(3.5, 'moderate') == 'BET'

    def test_signal_lean(self):
        """Marginal edge quality → LEAN."""
        from nba_models.inference.daily_predictions import get_signal_from_edge

        assert get_signal_from_edge(2.5, 'marginal') == 'LEAN'

    def test_signal_pass(self):
        """Low edge, no quality → PASS."""
        from nba_models.inference.daily_predictions import get_signal_from_edge

        assert get_signal_from_edge(1.0, 'none') == 'PASS'
        assert get_signal_from_edge(0.5, None) == 'PASS'

    def test_signal_fade(self):
        """Large negative edge → FADE."""
        from nba_models.inference.daily_predictions import get_signal_from_edge

        assert get_signal_from_edge(-7.0, 'none') == 'FADE'
        assert get_signal_from_edge(-6.0, None) == 'FADE'


# ============================================================
# Test 11-12: Output dict validation
# ============================================================

class TestOutputDict:
    """Tests for enriched prediction output."""

    def _make_prediction(self, **kwargs):
        """Helper to call predict_player_prop with minimal args."""
        from nba_models.inference.daily_predictions import predict_player_prop, load_models

        defaults = {
            'player_name': 'Test Player',
            'player_id': 999,
            'prop_type': 'points',
            'line': 25.5,
            'opponent': 'BOS',
            'opponent_id': 1,
            'models': {},
            'use_api_features': False,
        }
        defaults.update(kwargs)
        return predict_player_prop(**defaults)

    def test_output_has_new_fields(self):
        """Return dict includes Phase 4 fields: pick, over_edge, under_edge, edge_quality, american_odds."""
        result = self._make_prediction()

        required_fields = [
            'pick', 'over_edge', 'under_edge', 'edge_quality',
            'american_odds', 'signal', 'under_prob', 'ev_per_dollar',
            'implied_probability', 'model_probability', 'has_edge',
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # Verify pick is OVER or UNDER
        assert result['pick'] in ('OVER', 'UNDER')
        # Verify signal is valid
        assert result['signal'] in ('BET', 'LEAN', 'PASS', 'FADE')

    def test_output_backward_compat(self):
        """Legacy fields edge_quality_tier, bet_recommendation still present."""
        result = self._make_prediction()

        assert 'edge_quality_tier' in result
        assert 'bet_recommendation' in result
        assert 'over_prob' in result
        assert 'edge' in result
        assert 'confidence_score' in result
        # signal and bet_recommendation should match
        assert result['signal'] == result['bet_recommendation']


# ============================================================
# Test 13-14: CLV Bridge
# ============================================================

class TestCLVBridge:
    """Tests for the CLV bridge recording."""

    def test_clv_bridge_records_bets(self):
        """BET signal predictions get recorded as TrackedBets."""
        # Use temp DB to avoid polluting real data
        import nba_betting.edge.clv_bridge as bridge

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_bets.db")

            # Monkey-patch the tracker to use temp DB
            from nba_betting.edge.bet_tracker import BetTracker
            old_tracker = bridge._tracker_instance
            bridge._tracker_instance = BetTracker(db_path)

            try:
                predictions = [
                    {
                        'player': 'LeBron James',
                        'stat': 'POINTS',
                        'line': 26.5,
                        'pick': 'OVER',
                        'signal': 'BET',
                        'american_odds': -110,
                        'over_prob': 0.60,
                        'model_probability': 0.60,
                        'edge': 7.6,
                        'game': 'BOS@LAL',
                    },
                ]
                count = bridge.record_predictions_as_bets(predictions, "2026-02-23")
                assert count == 1

                # Verify bet exists in DB
                bets = bridge._tracker_instance.get_pending_bets()
                assert len(bets) >= 1
                assert 'LeBron' in bets[0].selection
            finally:
                bridge._tracker_instance = old_tracker

    def test_clv_bridge_skips_pass(self):
        """PASS signal predictions are NOT tracked."""
        import nba_betting.edge.clv_bridge as bridge

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_bets.db")

            from nba_betting.edge.bet_tracker import BetTracker
            old_tracker = bridge._tracker_instance
            bridge._tracker_instance = BetTracker(db_path)

            try:
                predictions = [
                    {
                        'player': 'Bench Player',
                        'stat': 'POINTS',
                        'line': 8.5,
                        'pick': 'OVER',
                        'signal': 'PASS',
                        'american_odds': -110,
                        'over_prob': 0.51,
                        'edge': 0.5,
                        'game': 'BOS@LAL',
                    },
                    {
                        'player': 'Bad Bet',
                        'stat': 'ASSISTS',
                        'line': 5.5,
                        'pick': 'UNDER',
                        'signal': 'FADE',
                        'american_odds': -110,
                        'over_prob': 0.35,
                        'edge': -8.0,
                        'game': 'BOS@LAL',
                    },
                ]
                count = bridge.record_predictions_as_bets(predictions, "2026-02-23")
                assert count == 0
            finally:
                bridge._tracker_instance = old_tracker


# ============================================================
# Test 15: Prediction logging
# ============================================================

class TestPredictionLogging:
    """Tests for CalibrationService prediction logging."""

    def test_prediction_logging(self):
        """CalibrationService.log_prediction records a prediction."""
        try:
            from calibration_tracker import CalibrationService
        except ImportError:
            pytest.skip("CalibrationService not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_calibration.db")
            service = CalibrationService(db_path=db_path)

            pred_id = service.log_prediction(
                player_id=2544,
                player_name="LeBron James",
                team="LAL",
                opponent="BOS",
                game_date="2026-02-23",
                prop_type="points",
                predicted_value=27.5,
                prop_line=26.5,
                predicted_over_prob=0.58,
                confidence=65.0,
                edge=5.8,
            )

            assert pred_id > 0


# ============================================================
# Test 16: EdgeCalculator fallback
# ============================================================

class TestEdgeCalculatorFallback:
    """Tests for fallback behavior when EdgeCalculator is unavailable."""

    def test_edge_calculator_fallback(self):
        """When EdgeCalculator unavailable, falls back to legacy formula."""
        import nba_models.inference.daily_predictions as dp

        # Save original state
        original_flag = dp.HAS_EDGE_CALCULATOR

        try:
            # Force fallback mode
            dp.HAS_EDGE_CALCULATOR = False

            result = dp._calculate_prop_edge(0.58, -110)

            # Should still produce valid results
            assert result['pick'] in ('OVER', 'UNDER')
            assert 'over_edge' in result
            assert 'under_edge' in result
            assert 'edge_quality' in result
            assert result['pick'] == 'OVER'  # 0.58 > 0.5

            # Legacy formula: (0.58 - 110/210) * 100 = 5.619...
            # (110/210 = 0.52381, not 0.524 as previously rounded)
            assert abs(result['over_edge'] - 5.619) < 0.05

        finally:
            dp.HAS_EDGE_CALCULATOR = original_flag


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
