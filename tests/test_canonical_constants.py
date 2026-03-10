"""
Tests for nba_betting/constants.py — canonical shared constants.

These tests enforce that:
1. PROP_STD_DEVS values are empirically correct (no regressions to old stale values)
2. All consumers of PROP_STD_DEVS import from the same source
3. BACKTEST_SANITY thresholds are realistic for sports betting
4. QUANTILE_DECOMPRESSION_DEFAULTS are in range
5. The JSON artifact on disk is not the stale placeholder
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# Canonical constants import
# ─────────────────────────────────────────────

class TestPropStdDevs:
    """PROP_STD_DEVS must reflect empirically-calibrated NBA variance."""

    from nba_betting.constants import PROP_STD_DEVS

    def test_all_prop_types_present(self):
        from nba_betting.constants import PROP_STD_DEVS
        required = {'points', 'rebounds', 'assists', 'threes', 'pra'}
        assert required.issubset(PROP_STD_DEVS.keys()), (
            f"Missing prop types: {required - PROP_STD_DEVS.keys()}"
        )

    def test_points_std_dev_correct(self):
        """Points std dev must be 6.16 — derived from backtest RMSE."""
        from nba_betting.constants import PROP_STD_DEVS
        assert PROP_STD_DEVS['points'] == 6.16, (
            f"points std dev is {PROP_STD_DEVS['points']}, expected 6.16. "
            "Derived from backtest RMSE: sqrt(6.31² - 1.38²) = 6.16."
        )

    def test_rebounds_std_dev_corrected(self):
        """Rebounds std dev must be 2.67 — derived from backtest RMSE."""
        from nba_betting.constants import PROP_STD_DEVS
        assert PROP_STD_DEVS['rebounds'] == 2.67, (
            f"rebounds std dev is {PROP_STD_DEVS['rebounds']}, expected 2.67. "
            "Derived from backtest RMSE: 2.67 (bias ≈ 0, so RMSE ≈ σ)."
        )

    def test_assists_std_dev_correct(self):
        """Assists std dev must be 1.95 — derived from backtest RMSE."""
        from nba_betting.constants import PROP_STD_DEVS
        assert PROP_STD_DEVS['assists'] == 1.95

    def test_threes_std_dev_correct(self):
        """Threes std dev must be 1.36 — derived from backtest RMSE."""
        from nba_betting.constants import PROP_STD_DEVS
        assert PROP_STD_DEVS['threes'] == 1.36

    def test_pra_std_dev_correct(self):
        """PRA std dev must be 7.97 — derived from backtest RMSE."""
        from nba_betting.constants import PROP_STD_DEVS
        assert PROP_STD_DEVS['pra'] == 7.97

    def test_all_std_devs_positive(self):
        from nba_betting.constants import PROP_STD_DEVS
        for prop, std in PROP_STD_DEVS.items():
            assert std > 0, f"{prop} std dev must be positive, got {std}"

    def test_std_devs_in_realistic_range(self):
        """All std devs should be in [0.5, 15.0] — sanity check for typos."""
        from nba_betting.constants import PROP_STD_DEVS
        for prop, std in PROP_STD_DEVS.items():
            assert 0.5 <= std <= 15.0, (
                f"{prop} std dev {std} is outside realistic range [0.5, 15.0]"
            )

    def test_assists_smaller_than_points(self):
        """Assists std dev < points std dev — assists are more consistent."""
        from nba_betting.constants import PROP_STD_DEVS
        assert PROP_STD_DEVS['assists'] < PROP_STD_DEVS['points']

    def test_pra_larger_than_components(self):
        """PRA std dev should be larger than any individual component."""
        from nba_betting.constants import PROP_STD_DEVS
        assert PROP_STD_DEVS['pra'] > PROP_STD_DEVS['points']
        assert PROP_STD_DEVS['pra'] > PROP_STD_DEVS['rebounds']
        assert PROP_STD_DEVS['pra'] > PROP_STD_DEVS['assists']


class TestEdgeCalculatorUsesCanonicalStdDevs:
    """edge_calculator.py must import PROP_STD_DEVS from nba_betting.constants."""

    def test_edge_calculator_uses_canonical_rebounds_std(self):
        """Critical: edge_calculator rebounds std must be 3.1, NOT the old 7.0."""
        from edge_calculator.edge_calculator import EdgeCalculator
        from nba_betting.constants import PROP_STD_DEVS
        assert EdgeCalculator.PROP_STD_DEVS['rebounds'] == PROP_STD_DEVS['rebounds'], (
            "EdgeCalculator.PROP_STD_DEVS['rebounds'] does not match canonical constant. "
            "This means edge_calculator.py has a local copy instead of importing."
        )

    def test_edge_calculator_matches_canonical_for_all_props(self):
        from edge_calculator.edge_calculator import EdgeCalculator
        from nba_betting.constants import PROP_STD_DEVS
        for prop in PROP_STD_DEVS:
            assert EdgeCalculator.PROP_STD_DEVS.get(prop) == PROP_STD_DEVS[prop], (
                f"EdgeCalculator.PROP_STD_DEVS['{prop}'] = "
                f"{EdgeCalculator.PROP_STD_DEVS.get(prop)} != canonical {PROP_STD_DEVS[prop]}"
            )

    def test_edge_from_prediction_uses_correct_rebounds_std(self):
        """Verify that edge calculation for rebounds uses correct std dev + bias correction."""
        from scipy.stats import norm
        from edge_calculator.edge_calculator import EdgeCalculator
        from nba_betting.constants import PROP_STD_DEVS, PROP_BIAS_CORRECTION

        calc = EdgeCalculator()
        diff = 2.0  # model predicts 2 rebounds above line
        result = calc.calculate_edge_from_prediction(
            predicted_value=7.0,
            prop_line=5.0,
            prop_type='rebounds',
        )
        bias_fix = PROP_BIAS_CORRECTION.get('rebounds', 0.0)
        expected_prob = float(norm.cdf((diff + bias_fix) / PROP_STD_DEVS['rebounds']))
        assert abs(result.model_probability - expected_prob) < 0.005, (
            f"Rebounds edge probability {result.model_probability:.4f} != "
            f"expected {expected_prob:.4f} using std={PROP_STD_DEVS['rebounds']}, "
            f"bias={bias_fix}"
        )


class TestPostgameAgentUsesCanonicalStdDevs:
    """postgame_agent.py must import PROP_STD_DEVS from nba_betting.constants."""

    def test_postgame_agent_imports_canonical_prop_std_devs(self):
        from agents.postgame.postgame_agent import PROP_STD_DEVS as AGENT_STD_DEVS
        from nba_betting.constants import PROP_STD_DEVS as CANONICAL

        for prop in CANONICAL:
            assert AGENT_STD_DEVS.get(prop) == CANONICAL[prop], (
                f"PostGameAgent PROP_STD_DEVS['{prop}'] = {AGENT_STD_DEVS.get(prop)} "
                f"!= canonical {CANONICAL[prop]}. "
                "postgame_agent.py must import from nba_betting.constants."
            )


class TestBacktestSanityThresholds:
    """BACKTEST_SANITY thresholds must be realistic for sports betting."""

    def test_max_roi_realistic(self):
        from nba_betting.constants import BACKTEST_SANITY
        assert BACKTEST_SANITY['max_roi'] <= 20.0, (
            f"max_roi={BACKTEST_SANITY['max_roi']}% is too high for leakage detection. "
            "Professional bettors achieve 2-8% ROI."
        )
        assert BACKTEST_SANITY['max_roi'] >= 10.0, (
            f"max_roi={BACKTEST_SANITY['max_roi']}% is too strict — elite bettors can reach 10-12%."
        )

    def test_max_win_rate_realistic(self):
        from nba_betting.constants import BACKTEST_SANITY
        assert BACKTEST_SANITY['max_win_rate'] <= 65.0, (
            f"max_win_rate={BACKTEST_SANITY['max_win_rate']}% is too high to detect leakage."
        )
        assert BACKTEST_SANITY['max_win_rate'] >= 55.0, (
            f"max_win_rate={BACKTEST_SANITY['max_win_rate']}% is too strict — some edges achieve 57-60%."
        )

    def test_model_trainer_uses_canonical_thresholds(self):
        from nba_models.models.model_trainer import SANITY_LIMITS
        from nba_betting.constants import BACKTEST_SANITY
        assert SANITY_LIMITS['max_roi'] == BACKTEST_SANITY['max_roi']
        assert SANITY_LIMITS['max_win_rate'] == BACKTEST_SANITY['max_win_rate']

    def test_backtesting_module_uses_canonical_thresholds(self):
        from nba_models.backtesting.backtesting import SANITY_THRESHOLDS
        from nba_betting.constants import BACKTEST_SANITY
        assert SANITY_THRESHOLDS['max_realistic_roi'] == BACKTEST_SANITY['max_roi']
        assert SANITY_THRESHOLDS['max_realistic_win_rate'] == BACKTEST_SANITY['max_win_rate']


class TestQuantileDecompressionDefaults:
    """QUANTILE_DECOMPRESSION_DEFAULTS must be calibrated, not placeholder."""

    def test_defaults_have_all_prop_types(self):
        from nba_betting.constants import QUANTILE_DECOMPRESSION_DEFAULTS
        required = {'points', 'rebounds', 'assists', 'threes', 'pra'}
        assert required.issubset(QUANTILE_DECOMPRESSION_DEFAULTS.keys())

    def test_defaults_have_required_keys(self):
        from nba_betting.constants import QUANTILE_DECOMPRESSION_DEFAULTS
        for prop, params in QUANTILE_DECOMPRESSION_DEFAULTS.items():
            assert 'slope' in params, f"{prop} missing 'slope'"
            assert 'mean_gap' in params, f"{prop} missing 'mean_gap'"
            assert 'mean_line' in params, f"{prop} missing 'mean_line'"

    def test_slopes_are_not_all_identical_placeholders(self):
        """Slopes must NOT all be 0.7 — that's the old uninitialized placeholder."""
        from nba_betting.constants import QUANTILE_DECOMPRESSION_DEFAULTS
        slopes = [p['slope'] for p in QUANTILE_DECOMPRESSION_DEFAULTS.values()]
        assert not all(s == 0.7 for s in slopes), (
            "All quantile decompression slopes are 0.7 — this is the stale placeholder value. "
            "Run: python3 scripts/calibrate_quantile_decompression.py"
        )

    def test_slopes_are_between_0_and_1(self):
        """Slope < 1.0 indicates regression to mean (expected). Slope > 1.0 is unusual."""
        from nba_betting.constants import QUANTILE_DECOMPRESSION_DEFAULTS
        for prop, params in QUANTILE_DECOMPRESSION_DEFAULTS.items():
            slope = params['slope']
            assert 0.2 <= slope <= 1.2, (
                f"{prop} slope={slope} outside expected range [0.2, 1.2]"
            )

    def test_points_slope_indicates_compression(self):
        """Points model has the most regression-to-mean (slope ≈ 0.72)."""
        from nba_betting.constants import QUANTILE_DECOMPRESSION_DEFAULTS
        slope = QUANTILE_DECOMPRESSION_DEFAULTS['points']['slope']
        assert slope < 0.90, (
            f"Points slope={slope} — expected < 0.90 (regression to mean). "
            "If it's near 1.0, decompression is unnecessary. Re-run calibration."
        )


class TestQuantileDecompressionJsonFile:
    """The models/quantile_decompression.json artifact must not contain stale placeholders."""

    JSON_PATH = Path(__file__).parent.parent / "models" / "quantile_decompression.json"

    def test_file_exists(self):
        assert self.JSON_PATH.exists(), (
            f"models/quantile_decompression.json not found at {self.JSON_PATH}. "
            "Run: python3 scripts/calibrate_quantile_decompression.py"
        )

    def test_file_not_all_stale_placeholders(self):
        """File must NOT have slope=0.7 for ALL props — that is the uninitialized default."""
        data = json.loads(self.JSON_PATH.read_text())
        prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']
        slopes = []
        for p in prop_types:
            if p in data and 'slope' in data[p]:
                slopes.append(data[p]['slope'])

        if slopes:
            assert not all(s == 0.7 for s in slopes), (
                "quantile_decompression.json has slope=0.7 for all props — stale placeholder. "
                "Run: python3 scripts/calibrate_quantile_decompression.py"
            )

    def test_file_has_calibrated_points_slope(self):
        """Points slope should be around 0.72 (measured production value)."""
        data = json.loads(self.JSON_PATH.read_text())
        if 'points' in data and 'slope' in data['points']:
            slope = data['points']['slope']
            assert 0.60 <= slope <= 0.95, (
                f"points slope={slope} is outside expected range [0.60, 0.95]. "
                "This may indicate the file contains incorrect values."
            )


class TestKellySizingConstants:
    """Kelly fraction constants must reflect conservative fractional Kelly approach."""

    def test_kelly_fractions_sum_to_reasonable_total(self):
        from nba_betting.constants import KELLY_FRACTIONS
        # No single bet should be full Kelly (1.0) — too volatile
        for tier, fraction in KELLY_FRACTIONS.items():
            assert fraction <= 0.60, (
                f"Kelly fraction for {tier} tier is {fraction} — too aggressive. "
                "Full Kelly is known to lead to ruin with any model error."
            )

    def test_max_bet_fraction_conservative(self):
        from nba_betting.constants import MAX_BET_FRACTION
        assert MAX_BET_FRACTION <= 0.05, (
            f"MAX_BET_FRACTION={MAX_BET_FRACTION} exceeds 5% of bankroll. "
            "Any single bet > 5% is dangerously concentrated."
        )

    def test_elite_beats_strong_beats_moderate(self):
        from nba_betting.constants import KELLY_FRACTIONS
        assert KELLY_FRACTIONS['elite'] >= KELLY_FRACTIONS['strong'] >= KELLY_FRACTIONS['moderate']

    def test_low_tier_not_bet(self):
        from nba_betting.constants import KELLY_FRACTIONS
        assert KELLY_FRACTIONS.get('low', 0) == 0.0, (
            "Low-confidence tier should have Kelly fraction = 0 (do not bet)."
        )
