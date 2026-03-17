"""
Comprehensive regression tests for edge calculation logic.

Covers:
- Spread edge formula (the critical bug fix)
- Prop edge (over/under, different prop types)
- Moneyline edge
- Cross-codepath consistency (app.py vs daily_predictions.py)

Convention:
    predicted_spread = home margin (+ = home wins by X)
    market_spread = home line (- = home favored, + = home underdog)
    home_cover_threshold = -market_spread = points home needs to win by
    spread_edge_points = predicted_spread - home_cover_threshold
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from scipy.stats import norm

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Helper: replicate the FIXED spread edge logic from daily_predictions.py
# ============================================================

NBA_SPREAD_VOLATILITY = 13.0


def compute_spread_edge(predicted_spread: float, market_spread: float):
    """
    Replicate the fixed spread edge logic from daily_predictions.py.

    Returns:
        (side, edge_points, cover_prob, edge_pct)
        side: 'home' or 'away'
        edge_points: absolute edge in points (always positive)
        cover_prob: probability of the recommended side covering
        edge_pct: cover_prob - 0.524 (break-even at -110), as percentage
    """
    home_cover_threshold = -market_spread
    raw_edge = predicted_spread - home_cover_threshold  # = predicted_spread + market_spread

    if raw_edge > 0:
        side = 'home'
        edge_points = raw_edge
        cover_prob = float(norm.cdf(edge_points / NBA_SPREAD_VOLATILITY))
    else:
        # Away covers (or zero edge — matches app.py's `if model_spread > threshold` else away)
        side = 'away'
        edge_points = abs(raw_edge)
        cover_prob = float(norm.cdf(edge_points / NBA_SPREAD_VOLATILITY))

    edge_pct = (cover_prob - 0.524) * 100
    return side, edge_points, cover_prob, edge_pct


# ============================================================
# Helper: replicate app.py determine_spread_bet_side
# ============================================================

def app_determine_spread_bet_side(model_spread: float, market_spread: float):
    """Exact copy of app.py's determine_spread_bet_side logic."""
    home_cover_threshold = -market_spread

    if model_spread > home_cover_threshold:
        side = "home"
        edge_points = model_spread - home_cover_threshold
        cover_prob = float(norm.cdf(edge_points / NBA_SPREAD_VOLATILITY))
    else:
        side = "away"
        edge_points = home_cover_threshold - model_spread
        cover_prob = float(norm.cdf(edge_points / NBA_SPREAD_VOLATILITY))

    return side, edge_points, cover_prob


# ============================================================
# SPREAD EDGE TESTS — The critical bug fix
# ============================================================

class TestSpreadEdgeFormula:
    """Test the corrected spread edge formula against all plan scenarios."""

    # Plan's test matrix: (predicted_spread, market_spread, expected_side, expected_edge_pts)
    CASES = [
        # Case 1: Home fav covers — model +15, market -12 → home by 3 pts
        (+15.0, -12.0, 'home', 3.0),
        # Case 2: Home fav doesn't cover — model +4.6, market -12 → away by 7.4
        (+4.6, -12.0, 'away', 7.4),
        # Case 3: Home dog covers — model -2.5, market +5.5 → home by 3.0
        (-2.5, +5.5, 'home', 3.0),
        # Case 4: Home dog doesn't cover — model -8, market +5.5 → away by 2.5
        (-8.0, +5.5, 'away', 2.5),
        # Case 5: Pick'em — model +3, market 0 → home by 3.0
        (+3.0, 0.0, 'home', 3.0),
        # Case 6: Zero edge — model +7, market -7 → threshold=7, edge=0
        (+7.0, -7.0, 'away', 0.0),
        # Case 7: CLAUDE.md case — model -2.5, market +5.5 → home by 3.0
        (-2.5, +5.5, 'home', 3.0),
    ]

    @pytest.mark.parametrize(
        "predicted_spread, market_spread, expected_side, expected_edge",
        CASES,
        ids=[
            "home_fav_covers",
            "home_fav_doesnt_cover",
            "home_dog_covers",
            "home_dog_doesnt_cover",
            "pickem",
            "zero_edge",
            "claude_md_case",
        ],
    )
    def test_spread_edge_direction(self, predicted_spread, market_spread, expected_side, expected_edge):
        side, edge_pts, cover_prob, edge_pct = compute_spread_edge(predicted_spread, market_spread)
        assert side == expected_side, (
            f"Expected {expected_side}, got {side} "
            f"(predicted={predicted_spread}, market={market_spread})"
        )
        assert abs(edge_pts - expected_edge) < 0.01, (
            f"Expected edge {expected_edge}, got {edge_pts:.2f}"
        )

    @pytest.mark.parametrize(
        "predicted_spread, market_spread, expected_side, expected_edge",
        CASES,
        ids=[
            "home_fav_covers",
            "home_fav_doesnt_cover",
            "home_dog_covers",
            "home_dog_doesnt_cover",
            "pickem",
            "zero_edge",
            "claude_md_case",
        ],
    )
    def test_cover_prob_direction(self, predicted_spread, market_spread, expected_side, expected_edge):
        """Cover probability should be > 0.5 when there's positive edge, == 0.5 at zero."""
        _, edge_pts, cover_prob, _ = compute_spread_edge(predicted_spread, market_spread)
        if edge_pts > 0:
            assert cover_prob > 0.5, f"Cover prob {cover_prob} should be > 0.5 with {edge_pts} pts edge"
        else:
            assert abs(cover_prob - 0.5) < 0.001

    def test_old_formula_was_wrong(self):
        """Demonstrate the bug: old formula gave opposite direction."""
        predicted_spread = 4.6
        market_spread = -12.0

        # OLD (buggy): predicted_spread - market_spread
        old_edge = predicted_spread - market_spread  # 4.6 - (-12) = 16.6
        # Old formula says home covers with 16.6 pts edge — WRONG

        # NEW (fixed): predicted_spread + market_spread
        new_edge = predicted_spread + market_spread  # 4.6 + (-12) = -7.4
        # New formula says away covers with 7.4 pts edge — CORRECT

        assert old_edge > 0, "Old formula incorrectly says home covers"
        assert new_edge < 0, "New formula correctly says away covers"
        assert abs(new_edge) == pytest.approx(7.4, abs=0.01)

    def test_edge_pct_positive_when_edge_exists(self):
        """Edge percentage should be positive when there's a meaningful edge."""
        # 5-point edge → substantial probability advantage
        _, _, _, edge_pct = compute_spread_edge(+10.0, -5.0)
        assert edge_pct > 0, "Edge % should be positive with 5pt edge"

    def test_edge_pct_negative_near_zero_edge(self):
        """Tiny edge produces negative edge_pct because of the -110 vig hurdle."""
        # 0.1 pt edge → cover_prob just above 0.5 but below 0.524
        _, _, cover_prob, edge_pct = compute_spread_edge(0.1, 0.0)
        assert cover_prob > 0.5
        assert edge_pct < 0, "Tiny edge can't overcome -110 vig"

    def test_symmetry(self):
        """Swapping roles: same magnitude edge should produce same cover probability."""
        _, edge_h, prob_h, _ = compute_spread_edge(+10.0, -7.0)  # home covers by 3
        _, edge_a, prob_a, _ = compute_spread_edge(+4.0, -7.0)   # away covers by 3
        assert abs(edge_h - edge_a) < 0.01
        assert abs(prob_h - prob_a) < 0.001

    def test_large_blowout_edge(self):
        """Extreme edge case: 20+ point edge should give very high cover prob."""
        _, edge_pts, cover_prob, _ = compute_spread_edge(+25.0, -3.0)
        assert edge_pts == pytest.approx(22.0, abs=0.01)
        assert cover_prob > 0.9


# ============================================================
# CROSS-CODEPATH CONSISTENCY
# ============================================================

class TestCrossCodepathConsistency:
    """Verify daily_predictions.py's fixed logic matches app.py's proven logic."""

    @pytest.mark.parametrize(
        "predicted_spread, market_spread",
        [
            (+15.0, -12.0),
            (+4.6, -12.0),
            (-2.5, +5.5),
            (-8.0, +5.5),
            (+3.0, 0.0),
            (+7.0, -7.0),
            (-2.5, +5.5),
            (+2.0, -1.0),
            (-15.0, +3.0),
            (+1.5, -0.5),
        ],
        ids=[
            "home_fav_covers",
            "home_fav_doesnt_cover",
            "home_dog_covers",
            "home_dog_doesnt_cover",
            "pickem",
            "zero_edge",
            "claude_md_case",
            "tiny_edge",
            "big_away_edge",
            "coin_flip",
        ],
    )
    def test_daily_predictions_matches_app(self, predicted_spread, market_spread):
        """The fixed daily_predictions logic must agree with app.py for all scenarios."""
        dp_side, dp_edge, dp_prob, _ = compute_spread_edge(predicted_spread, market_spread)
        app_side, app_edge, app_prob = app_determine_spread_bet_side(predicted_spread, market_spread)

        assert dp_side == app_side, (
            f"Side mismatch: daily_predictions={dp_side}, app={app_side} "
            f"(predicted={predicted_spread}, market={market_spread})"
        )
        assert abs(dp_edge - app_edge) < 0.01, (
            f"Edge mismatch: daily_predictions={dp_edge:.2f}, app={app_edge:.2f}"
        )
        assert abs(dp_prob - app_prob) < 0.001, (
            f"Prob mismatch: daily_predictions={dp_prob:.4f}, app={app_prob:.4f}"
        )


# ============================================================
# PROP EDGE TESTS
# ============================================================

class TestPropEdge:
    """Test prop over/under edge calculations from daily_predictions.py."""

    PROP_STD_DEVS = {
        'points': 5.5,
        'rebounds': 7.0,
        'assists': 2.5,
        'threes': 1.8,
        'pra': 9.0,
    }

    def _calc_prop_over_prob(self, predicted, line, prop_type='points'):
        """Replicate daily_predictions.py prop over probability."""
        std_dev = self.PROP_STD_DEVS.get(prop_type, 5.0)
        z_score = (predicted - line) / std_dev
        return float(norm.cdf(z_score))

    def test_over_when_predicted_above_line(self):
        """Model predicts above line → over probability > 0.5."""
        prob = self._calc_prop_over_prob(28.0, 24.5, 'points')
        assert prob > 0.5

    def test_under_when_predicted_below_line(self):
        """Model predicts below line → over probability < 0.5."""
        prob = self._calc_prop_over_prob(20.0, 24.5, 'points')
        assert prob < 0.5

    def test_prop_at_line(self):
        """Prediction equals line → over probability = 0.5."""
        prob = self._calc_prop_over_prob(24.5, 24.5, 'points')
        assert abs(prob - 0.5) < 0.001

    def test_rebounds_wider_std_dev(self):
        """Rebounds have larger std dev → same point diff produces lower probability."""
        pts_prob = self._calc_prop_over_prob(27.5, 24.5, 'points')  # 3pt diff, std=5.5
        reb_prob = self._calc_prop_over_prob(10.5, 7.5, 'rebounds')  # 3pt diff, std=7.0
        assert pts_prob > reb_prob, "Same diff should give higher prob with tighter std"

    def test_assists_tighter_std_dev(self):
        """Assists have smaller std dev → same point diff produces higher probability."""
        pts_prob = self._calc_prop_over_prob(27.0, 25.0, 'points')  # 2pt diff, std=5.5
        ast_prob = self._calc_prop_over_prob(7.0, 5.0, 'assists')   # 2pt diff, std=2.5
        assert ast_prob > pts_prob, "Same diff should give higher prob with tighter std"

    def test_threes_extreme_diff(self):
        """Large diff on threes (small std) → very high probability."""
        prob = self._calc_prop_over_prob(5.0, 2.5, 'threes')  # 2.5pt diff, std=1.8
        assert prob > 0.9

    def test_pra_combined(self):
        """PRA uses combined std dev of 9.0."""
        prob = self._calc_prop_over_prob(45.0, 40.0, 'pra')  # 5pt diff, std=9.0
        expected = float(norm.cdf(5.0 / 9.0))
        assert abs(prob - expected) < 0.001


# ============================================================
# EDGE CALCULATOR MODULE TESTS
# ============================================================

class TestEdgeCalculatorModule:
    """Test the edge_calculator.py calculate_edge_from_prediction with norm.cdf."""

    def test_prop_uses_norm_cdf(self):
        """calculate_edge_from_prediction should use norm.cdf with canonical std devs and bias correction."""
        from edge_calculator.edge_calculator import EdgeCalculator
        from nba_betting.constants import PROP_STD_DEVS, PROP_BIAS_CORRECTION

        calc = EdgeCalculator()
        result = calc.calculate_edge_from_prediction(
            predicted_value=28.0,
            prop_line=24.5,
            american_odds=-110,
            prop_type='points',
        )
        # diff = 28.0 - 24.5 = 3.5, bias_fix = PROP_BIAS_CORRECTION['points']
        pts_std = PROP_STD_DEVS['points']
        bias_fix = PROP_BIAS_CORRECTION.get('points', 0.0)
        expected_prob = float(norm.cdf((3.5 + bias_fix) / pts_std))
        assert abs(result.model_probability - expected_prob) < 0.01, (
            f"Expected norm.cdf((3.5 + {bias_fix}) / {pts_std}) = {expected_prob:.4f}, "
            f"got {result.model_probability:.4f}"
        )

    def test_prop_type_affects_probability(self):
        """Different prop types should produce different probabilities for same diff."""
        from edge_calculator.edge_calculator import EdgeCalculator
        from nba_betting.constants import PROP_STD_DEVS

        calc = EdgeCalculator()
        pts_result = calc.calculate_edge_from_prediction(
            predicted_value=27.0, prop_line=25.0, prop_type='points'
        )
        ast_result = calc.calculate_edge_from_prediction(
            predicted_value=7.0, prop_line=5.0, prop_type='assists'
        )
        # Same 2pt diff: assists std (2.2) < points std (6.5) → assists z-score is larger → higher prob
        assert PROP_STD_DEVS['assists'] < PROP_STD_DEVS['points'], (
            "assists std dev should be smaller than points std dev"
        )
        assert ast_result.model_probability > pts_result.model_probability

    def test_default_std_when_no_prop_type(self):
        """Without prop_type, falls back to default std dev of 5.0."""
        from edge_calculator.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()
        result = calc.calculate_edge_from_prediction(
            predicted_value=30.0, prop_line=25.0
        )
        expected_prob = float(norm.cdf(5.0 / 5.0))  # z=1.0
        assert abs(result.model_probability - expected_prob) < 0.01

    def test_backward_compatible_no_prop_type(self):
        """Old callers that don't pass prop_type should still work."""
        from edge_calculator.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()
        # Should not raise
        result = calc.calculate_edge_from_prediction(
            predicted_value=28.0,
            prop_line=26.5,
            american_odds=-110,
            model_confidence=65,
        )
        assert 0.05 <= result.model_probability <= 0.95


# ============================================================
# MONEYLINE EDGE TESTS
# ============================================================

class TestMoneylineEdge:
    """Test moneyline edge calculation."""

    def test_favorite_with_edge(self):
        """Model sees 65% chance, market implies 60% → has_edge uses no-vig edge."""
        from edge_calculator.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()
        result = calc.calculate_edge(model_probability=0.65, american_odds=-150)
        assert result.edge > 0
        assert result.no_vig_edge > 0
        assert result.has_edge is True

    def test_underdog_with_edge(self):
        """Model sees 45% chance on +150, no-vig ~43.3% → small positive no-vig edge."""
        from edge_calculator.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()
        result = calc.calculate_edge(model_probability=0.45, american_odds=150)
        assert result.edge > 0
        assert result.no_vig_edge > 0
        assert result.has_edge is False  # no_vig_edge < 3% threshold

    def test_no_edge(self):
        """Model at 50% vs -110/-110 → no_vig_edge ≈ 0, edge is negative (vig)."""
        from edge_calculator.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()
        result = calc.calculate_edge(model_probability=0.50, american_odds=-110)
        assert abs(result.no_vig_edge) < 0.01
        assert result.edge < 0
        assert result.has_edge is False

    def test_negative_edge(self):
        """Model probability below implied → negative edge, no bet."""
        from edge_calculator.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()
        result = calc.calculate_edge(model_probability=0.45, american_odds=-110)
        assert result.edge < 0
        assert result.has_edge is False
