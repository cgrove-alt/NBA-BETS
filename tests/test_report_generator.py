"""
Unit tests for report_generator.py

Tests HTML report generation with Plotly visualizations.
"""

import json
import math
import os
import sys
from pathlib import Path
import pytest
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report_generator import (
    load_backtest_results,
    safe_get,
    create_roi_curve,
    create_calibration_plot,
    create_tier_performance_chart,
    create_prop_type_comparison,
    create_worst_misses_table,
    generate_html_report
)


class TestSafeGet:
    """Test safe_get utility function."""

    def test_existing_key(self):
        data = {'key': 'value'}
        assert safe_get(data, 'key') == 'value'

    def test_missing_key_default(self):
        data = {}
        assert safe_get(data, 'key', 'default') == 'default'

    def test_none_value(self):
        data = {'key': None}
        assert safe_get(data, 'key', 'default') == 'default'

    def test_nan_value(self):
        data = {'key': float('nan')}
        assert safe_get(data, 'key', 'default') == 'default'

    def test_zero_value(self):
        data = {'key': 0}
        assert safe_get(data, 'key', 'default') == 0


class TestROICurve:
    """Test ROI curve chart generation."""

    def test_with_bet_history(self):
        betting_data = {
            'roi': 5.5,
            'bet_history': [
                {'date': '2025-01-01', 'profit': 10, 'amount': 100},
                {'date': '2025-01-02', 'profit': -5, 'amount': 100},
                {'date': '2025-01-03', 'profit': 15, 'amount': 100}
            ]
        }
        fig = create_roi_curve(betting_data)
        assert fig is not None
        assert len(fig.data) == 1  # One trace
        assert fig.data[0].name == 'Cumulative ROI'

    def test_without_bet_history(self):
        betting_data = {'roi': 5.5}
        fig = create_roi_curve(betting_data)
        assert fig is not None
        assert len(fig.data) == 1  # One bar
        assert fig.data[0].y[0] == 5.5

    def test_empty_betting_data(self):
        betting_data = {}
        fig = create_roi_curve(betting_data)
        assert fig is not None


class TestCalibrationPlot:
    """Test calibration plot generation."""

    def test_basic_calibration(self):
        predictions = [
            {'confidence': 80, 'error': 2.0},
            {'confidence': 85, 'error': 1.5},
            {'confidence': 90, 'error': 1.0},
            {'confidence': 50, 'error': 8.0},
            {'confidence': 55, 'error': 7.5}
        ]
        fig = create_calibration_plot(predictions)
        assert fig is not None
        assert len(fig.data) == 2  # Perfect line + actual

    def test_empty_predictions(self):
        predictions = []
        fig = create_calibration_plot(predictions)
        assert fig is not None

    def test_all_high_confidence(self):
        predictions = [
            {'confidence': 95, 'error': 1.0},
            {'confidence': 90, 'error': 2.0},
            {'confidence': 92, 'error': 1.5}
        ]
        fig = create_calibration_plot(predictions)
        assert fig is not None


class TestTierPerformanceChart:
    """Test tier performance chart generation."""

    def test_all_tiers(self):
        tier_data = {
            'elite': {'rmse': 3.5, 'count': 100},
            'strong': {'rmse': 4.2, 'count': 200},
            'moderate': {'rmse': 5.8, 'count': 150},
            'weak': {'rmse': 8.5, 'count': 50}
        }
        fig = create_tier_performance_chart(tier_data)
        assert fig is not None
        assert len(fig.data) == 2  # RMSE + Count

    def test_missing_tiers(self):
        tier_data = {
            'strong': {'rmse': 4.2, 'count': 200},
            'weak': {'rmse': 8.5, 'count': 50}
        }
        fig = create_tier_performance_chart(tier_data)
        assert fig is not None

    def test_empty_tier_data(self):
        tier_data = {}
        fig = create_tier_performance_chart(tier_data)
        assert fig is not None


class TestPropTypeComparison:
    """Test prop type comparison chart generation."""

    def test_all_props(self):
        prop_data = {
            'points': {'rmse': 6.5, 'r2': 0.15},
            'rebounds': {'rmse': 3.0, 'r2': 0.25},
            'assists': {'rmse': 3.5, 'r2': -0.10},
            'threes': {'rmse': 1.8, 'r2': -0.55}
        }
        fig = create_prop_type_comparison(prop_data)
        assert fig is not None
        assert len(fig.data) == 2  # RMSE + R²

    def test_missing_metrics(self):
        prop_data = {
            'points': {'rmse': 6.5},
            'rebounds': {'r2': 0.25}
        }
        fig = create_prop_type_comparison(prop_data)
        assert fig is not None

    def test_empty_prop_data(self):
        prop_data = {}
        fig = create_prop_type_comparison(prop_data)
        assert fig is not None


class TestWorstMissesTable:
    """Test worst misses HTML table generation."""

    def test_basic_table(self):
        predictions = [
            {
                'player': 'LeBron James',
                'prop_type': 'points',
                'predicted': 28.5,
                'actual': 15.0,
                'error': 13.5,
                'confidence': 85,
                'tier': 'strong',
                'game_date': '2025-01-15'
            },
            {
                'player': 'Stephen Curry',
                'prop_type': 'threes',
                'predicted': 5.5,
                'actual': 1.0,
                'error': 4.5,
                'confidence': 70,
                'tier': 'moderate',
                'game_date': '2025-01-15'
            }
        ]
        html = create_worst_misses_table(predictions, top_n=2)
        assert '<table' in html
        assert 'LeBron James' in html
        assert 'Stephen Curry' in html
        assert '13.5' in html or '13.50' in html

    def test_empty_predictions(self):
        html = create_worst_misses_table([])
        assert '<table' in html

    def test_top_n_limit(self):
        predictions = [
            {'player': f'Player{i}', 'error': float(i), 'predicted': 10, 'actual': 10-i}
            for i in range(30)
        ]
        html = create_worst_misses_table(predictions, top_n=10)
        assert '<table' in html
        # Should only have 10 rows (plus header)
        row_count = html.count('<tr>') - 1  # Subtract header
        assert row_count == 10


class TestGenerateHTMLReport:
    """Test full HTML report generation."""

    @pytest.fixture
    def sample_backtest_file(self):
        """Create a temporary backtest JSON file."""
        data = {
            'season_2025_26': {
                'phase': 'Phase 3: Test',
                'date_completed': '2026-01-19',
                'total_predictions': 100,
                'overall_performance': {
                    'count': 100,
                    'rmse': 4.5,
                    'mae': 3.2,
                    'bias': 1.1
                },
                'tier_performance': {
                    'strong': {'count': 60, 'rmse': 3.8, 'mae': 2.5, 'bias': 0.8}
                },
                'prop_type_performance': {
                    'points': {'count': 50, 'rmse': 5.0, 'r2': 0.10, 'mae': 3.5, 'bias': 1.2}
                },
                'betting_performance': {
                    'total_bets': 20,
                    'wins': 12,
                    'losses': 8,
                    'pushes': 0,
                    'win_rate': 60.0,
                    'roi': 5.5,
                    'total_wagered': 1000,
                    'total_profit': 55,
                    'final_bankroll': 1055,
                    'peak_bankroll': 1055,
                    'max_drawdown': 2.5,
                    'sharpe_ratio': 1.8
                },
                'calibration': {
                    'confidence_accuracy_correlation': 0.65,
                    'avg_confidence_all': 75.0
                },
                'elite_strong_performance': {
                    'count': 60,
                    'rmse': 3.8,
                    'mae': 2.5,
                    'bias': 0.8,
                    'percentage': 60.0
                },
                'sample_predictions': [
                    {
                        'player': 'Test Player',
                        'prop_type': 'points',
                        'predicted': 25.0,
                        'actual': 20.0,
                        'error': 5.0,
                        'confidence': 80,
                        'tier': 'strong',
                        'game_date': '2025-01-15'
                    }
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            return f.name

    def test_full_report_generation(self, sample_backtest_file):
        """Test complete report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_report.html')

            result_path = generate_html_report(sample_backtest_file, output_path)

            assert os.path.exists(result_path)
            assert result_path == output_path

            # Read and verify HTML content
            with open(result_path) as f:
                html = f.read()

            assert '<!DOCTYPE html>' in html
            assert 'NBA Prediction Model' in html
            assert 'Phase 3: Test' in html
            assert '5.50%' in html  # ROI
            assert '60.00%' in html  # Win rate
            assert 'Test Player' in html

        # Cleanup
        os.unlink(sample_backtest_file)

    def test_auto_output_path(self, sample_backtest_file):
        """Test automatic output path generation."""
        result_path = generate_html_report(sample_backtest_file)

        assert os.path.exists(result_path)
        assert 'backtest_reports' in result_path
        assert result_path.endswith('.html')

        # Cleanup
        os.unlink(sample_backtest_file)
        os.unlink(result_path)

    def test_season_selection(self, sample_backtest_file):
        """Test that correct season data is selected."""
        result_path = generate_html_report(sample_backtest_file)
        assert os.path.exists(result_path)

        with open(result_path) as f:
            html = f.read()

        assert 'Season 2025 26' in html  # Season key formatted

        # Cleanup
        os.unlink(sample_backtest_file)
        os.unlink(result_path)


class TestLoadBacktestResults:
    """Test loading backtest JSON files."""

    def test_load_valid_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'test': 'data'}, f)
            temp_file = f.name

        result = load_backtest_results(temp_file)
        assert result == {'test': 'data'}

        os.unlink(temp_file)

    def test_load_invalid_file(self):
        with pytest.raises(FileNotFoundError):
            load_backtest_results('/nonexistent/file.json')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
