"""
Phase 5 Tests — Calibration Feedback Loop

Tests cover:
1. BiasAnalyzer - player_tier classification, ECE computation
2. CalibrationAdjuster - apply_adjustments with player_tier, generate_adjustments
3. CalibrationService - log/retrieve, nightly job, calibrated prediction
4. Live integration - predict_player_prop output includes calibration fields
5. WeeklyReportGenerator - report generation with ECE and player_tier
6. Database - weekly_reports table, adjustment round-trip
"""

import os
import sys
import tempfile
import json
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration_tracker.database import CalibrationDatabase
from calibration_tracker.bias_analyzer import BiasAnalyzer, BiasReport, DimensionAnalysis
from calibration_tracker.calibration_adjuster import CalibrationAdjuster, CalibrationAdjustment
from calibration_tracker.calibration_service import CalibrationService
from calibration_tracker.weekly_report import WeeklyReportGenerator


# ============================================================
# Helpers
# ============================================================

def _make_temp_db():
    """Create a temporary CalibrationDatabase."""
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmp.close()
    return CalibrationDatabase(db_path=tmp.name)


def _seed_predictions(db, n=100, hit_rate=0.55):
    """
    Seed the DB with n predictions + outcomes for testing.

    Creates records with varied prop_types, positions, minutes, etc.
    """
    from datetime import datetime, timedelta
    import random

    prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']
    positions = ['guard', 'forward', 'center']
    random.seed(42)

    for i in range(n):
        game_date = (datetime.now() - timedelta(days=random.randint(1, 28))).strftime('%Y-%m-%d')
        prop_type = prop_types[i % len(prop_types)]
        position = positions[i % len(positions)]
        minutes_predicted = random.choice([18, 22, 26, 30, 34, 38])
        confidence = random.uniform(45, 85)
        predicted_value = random.uniform(10, 35)
        prop_line = predicted_value + random.uniform(-3, 3)
        over_prob = random.uniform(0.4, 0.7)
        spread = random.uniform(-10, 10)

        pred_id = db.insert_prediction({
            'game_date': game_date,
            'player_id': 1000 + i,
            'player_name': f'Player_{i}',
            'team': 'TST',
            'opponent': 'OPP',
            'position': position,
            'prop_type': prop_type,
            'predicted_value': predicted_value,
            'prop_line': prop_line,
            'predicted_over_prob': over_prob,
            'confidence': confidence,
            'edge': abs(over_prob - 0.524) * 100,
            'minutes_predicted': minutes_predicted,
            'is_home': random.choice([0, 1]),
            'spread': spread,
            'total': 220,
            'is_back_to_back': random.choice([0, 1]),
        })

        # Determine hit
        hit = 1 if random.random() < hit_rate else 0
        actual_value = predicted_value + random.uniform(-5, 5)
        error = predicted_value - actual_value

        # Determine over/under result
        if actual_value > prop_line:
            result = 'over'
        elif actual_value < prop_line:
            result = 'under'
        else:
            result = 'push'

        db.insert_outcome({
            'prediction_id': pred_id,
            'actual_value': actual_value,
            'actual_minutes': minutes_predicted + random.uniform(-5, 5),
            'result': result,
            'hit': hit,
            'error': error,
            'clv': random.uniform(-0.02, 0.03),
        })


# ============================================================
# Test 1-6: BiasAnalyzer
# ============================================================

class TestBiasAnalyzer:
    """Tests for BiasAnalyzer player_tier and ECE features."""

    def test_player_tier_classification(self):
        """Star/starter/role_player thresholds work correctly."""
        db = _make_temp_db()
        analyzer = BiasAnalyzer(db)

        assert analyzer._classify_player_tier(35) == 'star'
        assert analyzer._classify_player_tier(32) == 'star'
        assert analyzer._classify_player_tier(28) == 'starter'
        assert analyzer._classify_player_tier(24) == 'starter'
        assert analyzer._classify_player_tier(20) == 'role_player'
        assert analyzer._classify_player_tier(15) == 'role_player'
        assert analyzer._classify_player_tier(None) == 'unknown'

    def test_player_tier_in_report(self):
        """by_player_tier is populated in BiasReport."""
        db = _make_temp_db()
        _seed_predictions(db, n=100)
        analyzer = BiasAnalyzer(db)

        report = analyzer.analyze()

        assert hasattr(report, 'by_player_tier')
        assert len(report.by_player_tier) > 0

        # Check that known tiers exist
        tier_keys = set(report.by_player_tier.keys())
        # At least one of the tiers should be present
        assert tier_keys & {'star', 'starter', 'role_player'}

        # Each entry should be a DimensionAnalysis
        for tier, analysis in report.by_player_tier.items():
            assert isinstance(analysis, DimensionAnalysis)
            assert analysis.dimension == 'player_tier'

    def test_ece_perfect_calibration(self):
        """ECE=0 when predictions perfectly match outcomes."""
        db = _make_temp_db()
        analyzer = BiasAnalyzer(db)

        # Create records where predicted_over_prob perfectly matches hit rate
        # All in 0.6 bin, all hit
        records = [
            {'predicted_over_prob': 0.6, 'hit': 1} for _ in range(50)
        ] + [
            {'predicted_over_prob': 0.6, 'hit': 0} for _ in range(34)  # ~60% hit rate for 0.6 prob
        ]
        # 50 / 84 ≈ 0.595 which is close to 0.6

        result = analyzer.compute_ece(records)
        assert 'ece' in result
        assert 'bin_data' in result
        # ECE should be small (not exactly 0 due to binning)
        assert result['ece'] < 0.05

    def test_ece_poor_calibration(self):
        """ECE>0 when predictions don't match outcomes."""
        db = _make_temp_db()
        analyzer = BiasAnalyzer(db)

        # Predict 0.9 probability but only hit 10% of the time
        records = [
            {'predicted_over_prob': 0.9, 'hit': 0} for _ in range(90)
        ] + [
            {'predicted_over_prob': 0.9, 'hit': 1} for _ in range(10)
        ]

        result = analyzer.compute_ece(records)
        assert result['ece'] > 0.5  # Very poor calibration

    def test_ece_in_report(self):
        """ece field is present in BiasReport after analyze()."""
        db = _make_temp_db()
        _seed_predictions(db, n=50)
        analyzer = BiasAnalyzer(db)

        report = analyzer.analyze()

        assert hasattr(report, 'ece')
        assert isinstance(report.ece, float)
        assert hasattr(report, 'calibration_bins')
        assert isinstance(report.calibration_bins, list)

        # Check to_dict includes ece
        d = report.to_dict()
        assert 'ece' in d
        assert 'calibration_bins' in d

    def test_analyze_empty_data(self):
        """Empty records produce zero-value report, no crash."""
        db = _make_temp_db()
        analyzer = BiasAnalyzer(db)

        report = analyzer.analyze()

        assert report.total_predictions == 0
        assert report.overall_hit_rate == 0.0
        assert report.ece == 0.0
        assert report.calibration_bins == []
        assert len(report.by_player_tier) == 0


# ============================================================
# Test 7-13: CalibrationAdjuster
# ============================================================

class TestCalibrationAdjuster:
    """Tests for CalibrationAdjuster with player_tier support."""

    def test_apply_adjustments_basic(self):
        """Value adjustment applied correctly."""
        db = _make_temp_db()
        adjuster = CalibrationAdjuster(db)

        # Insert an active adjustment
        db.insert_adjustment({
            'dimension': 'prop_type',
            'dimension_value': 'points',
            'bias': 2.0,
            'adjustment': -2.0,
            'confidence_multiplier': 1.0,
            'sample_size': 100,
            'hit_rate': 0.55,
            'avg_error': 2.0,
            'std_error': 3.0,
        })

        result = adjuster.apply_adjustments(
            predicted_value=25.0,
            confidence=65.0,
            prop_type='points',
        )

        assert result['adjusted_value'] == 23.0  # 25 + (-2.0)
        assert result['total_value_adjustment'] == -2.0
        assert len(result['adjustments_applied']) == 1

    def test_apply_adjustments_with_player_tier(self):
        """Player tier dimension applied at 0.4 weight."""
        db = _make_temp_db()
        adjuster = CalibrationAdjuster(db)

        # Insert player_tier adjustment
        db.insert_adjustment({
            'dimension': 'player_tier',
            'dimension_value': 'star',
            'bias': 3.0,
            'adjustment': -3.0,
            'confidence_multiplier': 1.1,
            'sample_size': 80,
            'hit_rate': 0.58,
        })

        result = adjuster.apply_adjustments(
            predicted_value=30.0,
            confidence=70.0,
            prop_type='points',
            player_tier='star',
        )

        # Adjustment = -3.0 * 0.4 = -1.2
        assert result['adjusted_value'] == pytest.approx(28.8, abs=0.1)
        assert result['total_value_adjustment'] == pytest.approx(-1.2, abs=0.1)

        # Confidence multiplier: (1.1 - 1) * 0.4 + 1 = 1.04
        assert result['adjusted_confidence'] == pytest.approx(70.0 * 1.04, abs=0.5)

        # Check adjustment was logged
        tier_adj = [a for a in result['adjustments_applied'] if a['dimension'] == 'player_tier']
        assert len(tier_adj) == 1
        assert tier_adj[0]['value'] == 'star'

    def test_apply_adjustments_no_active(self):
        """No adjustments in DB, prediction unchanged."""
        db = _make_temp_db()
        adjuster = CalibrationAdjuster(db)

        result = adjuster.apply_adjustments(
            predicted_value=25.0,
            confidence=65.0,
            prop_type='points',
        )

        assert result['adjusted_value'] == 25.0
        assert result['adjusted_confidence'] == 65.0
        assert result['total_value_adjustment'] == 0
        assert result['adjustments_applied'] == []

    def test_confidence_clamped(self):
        """Confidence stays in [0, 100] after adjustments."""
        db = _make_temp_db()
        adjuster = CalibrationAdjuster(db)

        # Insert adjustment with extreme confidence multiplier
        db.insert_adjustment({
            'dimension': 'prop_type',
            'dimension_value': 'points',
            'bias': 1.0,
            'adjustment': -1.0,
            'confidence_multiplier': 2.0,  # Would double confidence
            'sample_size': 100,
        })

        result = adjuster.apply_adjustments(
            predicted_value=25.0,
            confidence=95.0,
            prop_type='points',
        )

        assert result['adjusted_confidence'] <= 100
        assert result['adjusted_confidence'] >= 0

    def test_generate_adjustments_min_sample(self):
        """Skips dimensions with < 50 samples."""
        db = _make_temp_db()
        # Seed only 20 predictions (below MIN_SAMPLE_SIZE of 50)
        _seed_predictions(db, n=20)
        adjuster = CalibrationAdjuster(db)

        adjustments = adjuster.generate_adjustments(save_to_db=False)

        # With only 20 total predictions, each dimension bucket will have even fewer
        # so no adjustments should be generated
        assert len(adjustments) == 0

    def test_generate_adjustments_min_bias(self):
        """Skips biases < 0.5 magnitude."""
        db = _make_temp_db()
        # Seed enough data for analysis
        _seed_predictions(db, n=200, hit_rate=0.524)
        adjuster = CalibrationAdjuster(db)

        adjustments = adjuster.generate_adjustments(save_to_db=False)

        # Any generated adjustment must have bias >= 0.5
        for adj in adjustments:
            assert abs(adj.bias) >= adjuster.MIN_BIAS_THRESHOLD

    def test_should_skip_bet(self):
        """Negative hit rate segment returns skip=True."""
        db = _make_temp_db()
        adjuster = CalibrationAdjuster(db)

        # Insert adjustment with terrible hit rate
        db.insert_adjustment({
            'dimension': 'prop_type',
            'dimension_value': 'threes',
            'bias': 2.0,
            'adjustment': -2.0,
            'confidence_multiplier': 0.8,
            'sample_size': 60,
            'hit_rate': 0.45,  # Below 0.48 threshold
        })

        should_skip, reason = adjuster.should_skip_bet('threes')
        assert should_skip is True
        assert 'threes' in reason.lower()
        assert 'negative' in reason.lower()


# ============================================================
# Test 14-17: CalibrationService
# ============================================================

class TestCalibrationService:
    """Tests for CalibrationService facade."""

    def test_log_and_retrieve(self):
        """Log prediction, retrieve by ID."""
        tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        tmp.close()
        service = CalibrationService(db_path=tmp.name)

        pred_id = service.log_prediction(
            player_id=123,
            player_name='Test Player',
            team='TST',
            opponent='OPP',
            game_date='2024-01-15',
            prop_type='points',
            predicted_value=25.0,
            prop_line=24.5,
            predicted_over_prob=0.58,
            confidence=65.0,
            edge=3.0,
        )

        assert pred_id is not None
        assert pred_id > 0

        # Retrieve
        pred = service.db.get_prediction(pred_id)
        assert pred is not None
        assert pred['player_name'] == 'Test Player'
        assert pred['prop_type'] == 'points'
        assert pred['predicted_value'] == 25.0

    def test_record_outcome(self):
        """Record outcome, hit/error computed correctly."""
        tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        tmp.close()
        service = CalibrationService(db_path=tmp.name)

        pred_id = service.log_prediction(
            player_id=456,
            player_name='Scorer',
            team='TST',
            opponent='OPP',
            game_date='2024-01-15',
            prop_type='points',
            predicted_value=28.0,
            prop_line=26.5,
            predicted_over_prob=0.6,
            confidence=70.0,
            edge=4.0,
        )

        out_id = service.record_outcome(
            prediction_id=pred_id,
            actual_value=30.0,
            actual_minutes=35.0,
        )

        assert out_id is not None

        # Check outcome stored
        outcome = service.db.get_outcome(pred_id)
        assert outcome is not None
        assert outcome['actual_value'] == 30.0
        # hit/error are computed internally from prediction + actual_value
        assert outcome['hit'] in (0, 1)
        assert outcome['error'] is not None

        # Check prediction marked as matched
        pred = service.db.get_prediction(pred_id)
        assert pred['status'] == 'matched'

    def test_run_nightly_job(self):
        """Full nightly flow runs without error."""
        tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        tmp.close()
        service = CalibrationService(db_path=tmp.name)

        # Seed some data
        _seed_predictions(service.db, n=50)

        # Run nightly job (won't fetch real outcomes, but should not crash)
        results = service.run_nightly_job(game_date='2024-01-15')

        assert results is not None
        assert 'steps' in results
        assert 'summary' in results

    def test_create_calibrated_prediction(self):
        """Facade produces calibrated output with player_tier."""
        tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        tmp.close()
        service = CalibrationService(db_path=tmp.name)

        result = service.create_calibrated_prediction(
            player={
                'id': 100,
                'name': 'Star Player',
                'position': 'PG',
                'team': 'TST',
                'projected_minutes': 35.0,
            },
            prop_type='points',
            raw_prediction=28.0,
            prop_line=27.0,
            raw_confidence=70.0,
            game_context={
                'opponent': 'OPP',
                'game_date': '2024-01-15',
                'is_home': True,
                'spread': -5.0,
                'total': 225.0,
                'is_back_to_back': False,
            },
            log_prediction=False,
        )

        assert 'raw_prediction' in result
        assert 'calibrated_prediction' in result
        assert 'calibrated_confidence' in result
        assert 'classification' in result
        # Without any adjustments in DB, calibrated should equal raw
        assert result['calibrated_prediction'] == result['raw_prediction']


# ============================================================
# Test 18-20: Live Integration (daily_predictions.py)
# ============================================================

class TestLiveIntegration:
    """Tests for calibration wired into predict_player_prop."""

    def _make_minimal_models(self):
        """Create minimal model dict that won't crash predict_player_prop."""
        return {}

    def test_predict_with_calibration(self):
        """predict_player_prop output includes calibration_adjustment field."""
        from nba_models.inference.daily_predictions import predict_player_prop

        result = predict_player_prop(
            player_name='Test Player',
            player_id=999,
            prop_type='points',
            line=25.5,
            opponent='BOS',
            opponent_id=1,
            models=self._make_minimal_models(),
            use_api_features=False,
            player_position='G',
        )

        assert 'calibration_adjustment' in result
        assert 'calibration_applied' in result

    def test_predict_without_calibration(self):
        """When no adjustments exist, prediction unchanged by calibration."""
        from nba_models.inference.daily_predictions import predict_player_prop

        result = predict_player_prop(
            player_name='Test Player',
            player_id=999,
            prop_type='rebounds',
            line=8.5,
            opponent='LAL',
            opponent_id=2,
            models=self._make_minimal_models(),
            use_api_features=False,
        )

        # With no adjustments in DB, calibration_adjustment should be 0
        assert result['calibration_adjustment'] == 0

    def test_calibration_never_blocks(self):
        """Even if CalibrationAdjuster crashes, prediction still returns."""
        from nba_models.inference.daily_predictions import predict_player_prop
        import nba_models.inference.daily_predictions as dp

        # Save original and temporarily set a broken adjuster
        original = dp._calibration_adjuster
        dp._calibration_adjuster = "not_a_real_adjuster"  # Will crash on .apply_adjustments()

        try:
            result = predict_player_prop(
                player_name='Test Player',
                player_id=999,
                prop_type='assists',
                line=6.5,
                opponent='GSW',
                opponent_id=3,
                models=self._make_minimal_models(),
                use_api_features=False,
            )

            # Should still get a valid result
            assert result is not None
            assert 'over_prob' in result
            assert 'edge' in result
        finally:
            # Restore original
            dp._calibration_adjuster = original


# ============================================================
# Test 21-23: Weekly Report
# ============================================================

class TestWeeklyReport:
    """Tests for WeeklyReportGenerator."""

    def test_generate_report(self):
        """Weekly report generates and stores to DB."""
        db = _make_temp_db()
        _seed_predictions(db, n=100)
        gen = WeeklyReportGenerator(db)

        report = gen.generate_weekly_report()

        assert report is not None
        assert 'week_ending' in report
        assert 'week_start' in report
        assert 'total_predictions' in report
        assert 'overall_hit_rate' in report
        assert 'recommendations' in report

        # Check it was saved to DB
        stored = db.get_weekly_report(report['week_ending'])
        assert stored is not None

    def test_report_includes_ece(self):
        """ECE present in weekly report."""
        db = _make_temp_db()
        _seed_predictions(db, n=100)
        gen = WeeklyReportGenerator(db)

        report = gen.generate_weekly_report()

        assert 'ece' in report
        assert isinstance(report['ece'], float)
        assert 'ece_trend' in report
        assert 'current' in report['ece_trend']
        assert 'previous' in report['ece_trend']
        assert 'calibration_bins' in report

    def test_report_includes_player_tier(self):
        """Player tier breakdown in report."""
        db = _make_temp_db()
        _seed_predictions(db, n=100)
        gen = WeeklyReportGenerator(db)

        report = gen.generate_weekly_report()

        assert 'by_player_tier' in report
        assert isinstance(report['by_player_tier'], dict)


# ============================================================
# Test 24-25: Database
# ============================================================

class TestDatabase:
    """Tests for database weekly_reports table and adjustment operations."""

    def test_weekly_reports_table(self):
        """weekly_reports table exists and accepts inserts."""
        db = _make_temp_db()

        report_id = db.insert_weekly_report('2024-01-14', {
            'total_predictions': 50,
            'matched_predictions': 45,
            'overall_hit_rate': 0.55,
            'overall_clv': 0.02,
            'overall_roi': 0.035,
            'ece': 0.08,
            'summary': 'Test weekly report',
        })

        assert report_id is not None

        # Retrieve
        stored = db.get_weekly_report('2024-01-14')
        assert stored is not None
        assert stored['total_predictions'] == 50
        assert stored['overall_hit_rate'] == 0.55
        assert stored['ece'] == 0.08

        # Check report JSON parsed
        assert 'report' in stored
        assert stored['report']['summary'] == 'Test weekly report'

    def test_insert_and_get_adjustment(self):
        """Round-trip insert/get adjustment works."""
        db = _make_temp_db()

        db.insert_adjustment({
            'dimension': 'player_tier',
            'dimension_value': 'star',
            'bias': 1.5,
            'adjustment': -1.5,
            'confidence_multiplier': 1.05,
            'sample_size': 75,
            'hit_rate': 0.57,
            'avg_error': 1.5,
            'std_error': 3.2,
        })

        adj = db.get_adjustment('player_tier', 'star')
        assert adj is not None
        assert adj['dimension'] == 'player_tier'
        assert adj['dimension_value'] == 'star'
        assert adj['bias'] == 1.5
        assert adj['adjustment'] == -1.5
        assert adj['confidence_multiplier'] == 1.05
        assert adj['is_active'] == 1
