"""
Comprehensive tests for the 2026-03-06 model improvement audit.

Tests cover:
1. Poisson-based over/under probability for threes
2. EWMA feature generation
3. Re-enabled threes model with proper calibration
4. EWMA direction alignment check in bet filter
5. Poisson overdispersion guard in bet filter
6. Edge quality prop-variance adjustment
7. Team stats calculator EWMA and momentum features
8. Kelly criterion prop-type multipliers
9. Edge calculator Poisson distribution support
"""

from __future__ import annotations

import numpy as np
import pytest


# ============================================================================
# 1. Poisson-based over/under probability
# ============================================================================

class TestOverProbability:
    """Tests for the over_probability() function in daily_predictions.py."""

    def _get_over_prob(self):
        """Import over_probability lazily to avoid heavy module-level imports."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from nba_models.inference.daily_predictions import over_probability
        return over_probability

    def test_threes_poisson_over_probability_basic(self):
        """P(X > 2.5 | mu=3) should be > 0.5."""
        over_probability = self._get_over_prob()
        prob = over_probability(3.0, 2.5, 'threes')
        assert prob > 0.5, f"Expected P(OVER 2.5 | mu=3) > 0.5, got {prob:.3f}"

    def test_threes_poisson_under_probability_basic(self):
        """P(X > 2.5 | mu=1) should be < 0.5 — under is likely."""
        over_probability = self._get_over_prob()
        prob = over_probability(1.0, 2.5, 'threes')
        assert prob < 0.5, f"Expected P(OVER 2.5 | mu=1) < 0.5, got {prob:.3f}"

    def test_threes_poisson_symmetry_near_50(self):
        """P(X > line | mu=line) should be close to 0.5 for non-integer lines."""
        over_probability = self._get_over_prob()
        # For mu=2.5, P(X >= 3) using Poisson
        from scipy.stats import poisson
        expected = 1.0 - poisson.cdf(2, 2.5)
        prob = over_probability(2.5, 2.5, 'threes')
        assert abs(prob - expected) < 0.01

    def test_threes_integer_line_tie_handling(self):
        """For integer lines, ties are split 50/50."""
        over_probability = self._get_over_prob()
        from scipy.stats import poisson
        # P(X > 3 | mu=3) vs P(X >= 3 | mu=3) should differ by pmf(3)/2
        prob_integer = over_probability(3.0, 3.0, 'threes')
        # Should equal P(X >= 4) + 0.5 * P(X == 3)
        expected = 1.0 - poisson.cdf(2, 3.0) - poisson.pmf(3, 3.0) * 0.5
        assert abs(prob_integer - expected) < 0.01

    def test_points_uses_normal_cdf(self):
        """Points prop should still use Normal CDF (not Poisson)."""
        over_probability = self._get_over_prob()
        from scipy.stats import norm
        from nba_betting.constants import PROP_STD_DEVS
        std = PROP_STD_DEVS['points']
        expected = norm.cdf((25.0 - 23.5) / std)
        prob = over_probability(25.0, 23.5, 'points')
        assert abs(prob - expected) < 0.01

    def test_probability_clamped_to_valid_range(self):
        """All over probabilities should be in [0.05, 0.95]."""
        over_probability = self._get_over_prob()
        for ptype in ['points', 'rebounds', 'assists', 'threes', 'pra']:
            prob_high = over_probability(100.0, 0.1, ptype)
            prob_low = over_probability(0.0, 100.0, ptype)
            assert 0.05 <= prob_high <= 0.95, f"{ptype}: prob={prob_high}"
            assert 0.05 <= prob_low <= 0.95, f"{ptype}: prob={prob_low}"


# ============================================================================
# 2. Bet filter improvements
# ============================================================================

class TestBetFilterImprovements:
    """Tests for the updated should_bet() function."""

    def _should_bet(self, **kwargs):
        from nba_betting.bet_filter import should_bet
        return should_bet(**kwargs)

    # ----- Threes re-enabled -----

    def test_threes_enabled_with_good_edge(self):
        """Threes should be bettable with a clear Poisson edge."""
        bet, reason, edge = self._should_bet(
            prop_type='threes',
            predicted_value=3.5,
            line_value=2.5,
            confidence=0.63,
            games_played=15,
            is_over=True,
            poisson_rate=3.5,
        )
        assert bet, f"Expected threes bet to be accepted: {reason}"

    def test_threes_disabled_without_poisson_support(self):
        """Threes OVER should be rejected if Poisson rate < line."""
        bet, reason, edge = self._should_bet(
            prop_type='threes',
            predicted_value=3.2,
            line_value=2.5,
            confidence=0.63,
            games_played=15,
            is_over=True,
            poisson_rate=2.0,  # Rate below line
        )
        assert not bet, f"Expected threes OVER rejected when Poisson rate < line: {reason}"

    def test_threes_high_overdispersion_rejected(self):
        """Very overdispersed threes shooter should be rejected."""
        bet, reason, edge = self._should_bet(
            prop_type='threes',
            predicted_value=3.5,
            line_value=2.5,
            confidence=0.70,
            games_played=20,
            is_over=True,
            overdispersion=4.5,  # Very high
        )
        assert not bet, f"Expected high-overdispersion threes rejected: {reason}"

    def test_threes_moderate_overdispersion_allowed(self):
        """Moderate overdispersion (< 3.5) should not block the bet."""
        bet, reason, edge = self._should_bet(
            prop_type='threes',
            predicted_value=3.5,
            line_value=2.5,
            confidence=0.63,
            games_played=15,
            is_over=True,
            overdispersion=2.0,
        )
        assert bet, f"Moderate overdispersion should be allowed: {reason}"

    # ----- EWMA direction alignment -----

    def test_ewma_misalignment_raises_threshold(self):
        """Bet with EWMA below line should require higher edge."""
        # Without EWMA: edge of 1.0 should fail (threshold = 0.5 for threes but let's use points)
        # Edge = 2.1 for points should pass normally
        bet_normal, _, _ = self._should_bet(
            prop_type='points',
            predicted_value=25.1,  # edge = 2.1 (above 2.0 threshold)
            line_value=23.0,
            confidence=0.63,
            games_played=15,
            is_over=True,
        )
        # With EWMA below line: same bet should fail (threshold becomes 3.0)
        bet_misaligned, reason, _ = self._should_bet(
            prop_type='points',
            predicted_value=25.1,  # edge = 2.1
            line_value=23.0,
            confidence=0.63,
            games_played=15,
            is_over=True,
            ewma_value=22.0,  # EWMA below line — trending down
        )
        assert bet_normal, "Normal bet should pass"
        assert not bet_misaligned, f"Misaligned EWMA should raise threshold: {reason}"

    def test_ewma_aligned_passes_normal_threshold(self):
        """When EWMA is aligned with bet direction, normal threshold applies."""
        bet, reason, _ = self._should_bet(
            prop_type='points',
            predicted_value=25.1,  # edge = 2.1
            line_value=23.0,
            confidence=0.63,
            games_played=15,
            is_over=True,
            ewma_value=25.5,  # EWMA above line — aligned with OVER
        )
        assert bet, f"EWMA-aligned bet should pass: {reason}"

    # ----- Spread still disabled -----

    def test_spread_still_disabled(self):
        """Spread should remain disabled."""
        bet, reason, edge = self._should_bet(
            prop_type='spread',
            predicted_value=5.0,
            line_value=3.0,
            confidence=0.65,
            games_played=30,
            is_over=True,
        )
        assert not bet, f"Spread should be disabled: {reason}"


# ============================================================================
# 3. Kelly bet sizing with prop-type multipliers
# ============================================================================

class TestKellyPropMultipliers:
    """Tests for prop-type-aware Kelly sizing."""

    def test_threes_kelly_is_half_of_points_kelly(self):
        """Threes should get 0.5x Kelly multiplier compared to points.

        Uses a very low confidence so neither bet hits the max_bet_pct cap,
        ensuring we can verify the raw multiplier relationship.
        """
        from nba_betting.bet_filter import calculate_bet_size

        bankroll = 1000.0
        # Low confidence → small Kelly → stays well below the 3% cap
        confidence = 0.515
        edge = 2.0

        points_bet = calculate_bet_size(edge, confidence, bankroll,
                                        prop_type='points', kelly_fraction=0.25)
        threes_bet = calculate_bet_size(edge, confidence, bankroll,
                                        prop_type='threes', kelly_fraction=0.25)

        # Neither bet should hit the 3% hard cap (they should be small)
        assert points_bet < 30.0, f"Points bet hit cap: {points_bet}"
        # Threes should be exactly half of points (0.5x multiplier)
        assert abs(threes_bet - points_bet * 0.5) < 0.01, (
            f"Expected threes bet ({threes_bet:.2f}) = 0.5 × points bet ({points_bet:.2f})"
        )

    def test_points_and_rebounds_have_same_kelly(self):
        """Points and rebounds have 1.0x multiplier — same Kelly."""
        from nba_betting.bet_filter import calculate_bet_size

        bankroll = 1000.0
        confidence = 0.62
        edge = 2.0

        pts_bet = calculate_bet_size(edge, confidence, bankroll, prop_type='points')
        reb_bet = calculate_bet_size(edge, confidence, bankroll, prop_type='rebounds')

        assert abs(pts_bet - reb_bet) < 0.01

    def test_spread_kelly_is_zero(self):
        """Spread (disabled) should have 0x multiplier."""
        from nba_betting.bet_filter import calculate_bet_size

        bet = calculate_bet_size(5.0, 0.70, 1000.0, prop_type='spread')
        assert bet == 0.0


# ============================================================================
# 4. Edge calculator Poisson support
# ============================================================================

class TestEdgeCalculatorPoisson:
    """Tests for Poisson distribution in EdgeCalculator."""

    def test_threes_uses_poisson_not_normal(self):
        """Threes edge calculation should produce different results than normal CDF."""
        from edge_calculator.edge_calculator import EdgeCalculator
        from scipy.stats import norm, poisson

        calc = EdgeCalculator()

        # For predicted=3.0, line=2.5, prop_type='threes'
        result_threes = calc.calculate_edge_from_prediction(3.0, 2.5, prop_type='threes')

        # Calculate what normal CDF would give
        from nba_betting.constants import PROP_STD_DEVS
        std = PROP_STD_DEVS['threes']
        normal_prob = norm.cdf((3.0 - 2.5) / std)

        # Poisson probability
        poisson_prob = 1.0 - poisson.cdf(2, 3.0)

        # The edge calculator should use Poisson, not Normal
        assert abs(result_threes.model_probability - poisson_prob) < 0.01, (
            f"EdgeCalculator for threes should use Poisson. "
            f"Got {result_threes.model_probability:.3f}, "
            f"expected Poisson={poisson_prob:.3f} (Normal={normal_prob:.3f})"
        )

    def test_points_still_uses_normal(self):
        """Points edge calculation should still use Normal CDF."""
        from edge_calculator.edge_calculator import EdgeCalculator
        from scipy.stats import norm
        from nba_betting.constants import PROP_STD_DEVS

        calc = EdgeCalculator()
        result = calc.calculate_edge_from_prediction(25.0, 23.5, prop_type='points')
        std = PROP_STD_DEVS['points']
        expected = np.clip(norm.cdf((25.0 - 23.5) / std), 0.05, 0.95)
        assert abs(result.model_probability - expected) < 0.01


# ============================================================================
# 5. Edge quality prop-variance adjustment
# ============================================================================

class TestEdgeQualityVarianceAdjustment:
    """Tests for the new prop-variance score in EdgeQualityScorer."""

    def _make_scorer(self):
        from nba_betting.edge.edge_quality import EdgeQualityScorer
        return EdgeQualityScorer()

    def test_high_overdispersion_lowers_score(self):
        """Very high 3PM overdispersion should lower the quality score."""
        scorer = self._make_scorer()

        adj_low, _ = scorer.calculate_prop_variance_adjustment(
            prop_type='threes', overdispersion=2.0)
        adj_high, _ = scorer.calculate_prop_variance_adjustment(
            prop_type='threes', overdispersion=5.0)

        assert adj_high < adj_low, (
            f"High overdispersion ({adj_high:.1f}) should score lower "
            f"than moderate ({adj_low:.1f})"
        )

    def test_low_overdispersion_bonus(self):
        """Low 3PM overdispersion (consistent shooter) should add positive score."""
        scorer = self._make_scorer()
        adj, factors = scorer.calculate_prop_variance_adjustment(
            prop_type='threes', overdispersion=1.2)
        assert adj > 0, f"Expected positive adjustment for consistent shooter: {adj}"
        assert any('consistent' in f.lower() for f in factors)

    def test_high_cv_penalty(self):
        """High coefficient of variation should penalise any prop type."""
        scorer = self._make_scorer()
        adj_low, _ = scorer.calculate_prop_variance_adjustment(
            prop_type='points', coefficient_of_variation=0.2)
        adj_high, _ = scorer.calculate_prop_variance_adjustment(
            prop_type='points', coefficient_of_variation=1.5)
        assert adj_high < adj_low

    def test_evaluate_edge_accepts_prop_type_params(self):
        """evaluate_edge should accept and use prop_type and variance params."""
        scorer = self._make_scorer()
        result = scorer.evaluate_edge(
            model_probability=0.62,
            implied_probability=0.52,
            prop_type='threes',
            poisson_overdispersion=4.0,
            coefficient_of_variation=0.8,
        )
        # With high overdispersion, should get a penalty in detailed breakdown
        assert 'prop_variance_adjustment' in result.detailed_breakdown
        assert result.detailed_breakdown['prop_variance_adjustment'] < 0


# ============================================================================
# 6. Team stats calculator EWMA and momentum features
# ============================================================================

class TestTeamStatsCalculatorEWMA:
    """Tests for EWMA and momentum features in TeamStatsCalculator."""

    def _make_calculator(self):
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nba_models', 'training'))
        from train_complete_balldontlie import TeamStatsCalculator
        return TeamStatsCalculator(window=5)

    def _add_games(self, calc, team_id=1, opp_id=2, scores=None):
        """Add a list of (home_score, away_score) game results."""
        if scores is None:
            scores = [(110, 105), (120, 100), (95, 115), (108, 108), (125, 90), (102, 99)]
        for i, (home, away) in enumerate(scores):
            game = {
                'date': f'2025-01-{10 + i:02d}',
                'home_team': {'id': team_id},
                'visitor_team': {'id': opp_id},
                'home_team_score': home,
                'visitor_team_score': away,
            }
            calc.add_game(game)

    def test_ewma_point_diff_present(self):
        """get_team_stats_before_date should return ewma_point_diff."""
        calc = self._make_calculator()
        self._add_games(calc)
        stats = calc.get_team_stats_before_date(1, '2025-01-20')
        assert stats is not None
        assert 'ewma_point_diff' in stats

    def test_net_rating_momentum_present(self):
        """net_rating_momentum (recent - season) should be present."""
        calc = self._make_calculator()
        self._add_games(calc)
        stats = calc.get_team_stats_before_date(1, '2025-01-20')
        assert 'net_rating_momentum' in stats

    def test_point_diff_std_present(self):
        """point_diff_std should be present."""
        calc = self._make_calculator()
        self._add_games(calc)
        stats = calc.get_team_stats_before_date(1, '2025-01-20')
        assert 'point_diff_std' in stats
        assert stats['point_diff_std'] >= 0

    def test_ewma_more_weight_on_recent(self):
        """EWMA should reflect recent performance more than simple average.

        We add games in chronological order. The TeamStatsCalculator sorts games
        by date descending (most recent first) when computing stats, so the MOST
        RECENTLY ADDED games are games[-1] after add_game loop, but FIRST in the
        sorted list (recent[0]).

        Test: first 3 games are big wins, last 3 are big losses (declining team).
        EWMA should be more negative than the simple window average because it
        weights the most-recent (losing) games more.
        """
        calc = self._make_calculator()
        # First 3 games added = early games: big wins (+30 diff)
        # Last 3 games added = recent games: big losses (-30 diff)
        scores = [
            (130, 100), (125, 95), (128, 98),   # early: +30 average diff
            (90, 120), (88, 118), (85, 115),     # recent: -30 average diff
        ]
        self._add_games(calc, scores=scores)
        stats = calc.get_team_stats_before_date(1, '2025-01-20')
        assert stats is not None
        # With EWMA giving most weight to the 3 recent losses (−30 diff each),
        # ewma_point_diff should be more negative than the simple average of all 5 recent games.
        # Simple average of all 5 (window=5) would blend old wins and recent losses.
        # EWMA focuses more on the most recent losses.
        assert stats['ewma_point_diff'] < stats['net_rating'], (
            f"EWMA ({stats['ewma_point_diff']:.1f}) should be lower than "
            f"simple recent avg ({stats['net_rating']:.1f}) when team is declining"
        )


# ============================================================================
# 7. Feature generator EWMA and Poisson features
# ============================================================================

class TestFeatureGeneratorNewFeatures:
    """Tests for new EWMA and Poisson features in canonical feature generator."""

    def _make_generator(self):
        from nba_data.transformers.feature_generator import PlayerFeatureGenerator
        return PlayerFeatureGenerator()

    def _make_games(self, n=12, pts=None, fg3m=None):
        """Create (date, stats) tuples."""
        from datetime import datetime, timedelta
        base = datetime(2025, 1, 1)
        games = []
        for i in range(n):
            d = (base + timedelta(days=i)).strftime('%Y-%m-%d')
            stats = {
                'pts': (pts[i] if pts else 20 + (i % 5)),
                'reb': 5, 'ast': 4,
                'fg3m': (fg3m[i] if fg3m else 2 + (i % 3)),
                'fg3a': 6, 'fgm': 8, 'fga': 18, 'fta': 4,
                'turnover': 2, 'min': 32
            }
            games.append((d, stats))
        return games

    def test_ewma_features_present(self):
        """EWMA features should be in the output."""
        gen = self._make_generator()
        games = self._make_games()
        features = gen.generate_features(games, '2025-01-15')
        assert features is not None
        assert 'pts_ewma' in features
        assert 'reb_ewma' in features
        assert 'ast_ewma' in features
        assert 'fg3m_ewma' in features

    def test_poisson_rate_positive(self):
        """Poisson rate should be non-negative."""
        gen = self._make_generator()
        games = self._make_games()
        features = gen.generate_features(games, '2025-01-15')
        assert features is not None
        assert 'poisson_rate' in features
        assert features['poisson_rate'] >= 0

    def test_overdispersion_positive(self):
        """Overdispersion should be positive."""
        gen = self._make_generator()
        games = self._make_games()
        features = gen.generate_features(games, '2025-01-15')
        assert features is not None
        assert 'fg3m_overdispersion' in features
        assert features['fg3m_overdispersion'] >= 0

    def test_ewma_gives_more_weight_to_recent(self):
        """EWMA of recent cold streak should be below simple average.

        _make_games() creates games in chronological order (earliest first).
        generate_features() sorts them descending (most recent first), so the
        LAST games added become recent[0] (highest weight in EWMA).

        Test: first 8 games = 25 pts (hot), last 4 games = 10 pts (cold streak).
        After descending sort: recent[0..3] = 10 pts, recent[4..9] = 25 pts.
        EWMA should be closer to 10 than the simple mean of recent (10 games).
        """
        gen = self._make_generator()
        # First 8 games chronologically: 25 pts (these will be at tail of sorted list)
        # Last 4 games chronologically: 10 pts (these will be at head = recent[0..3])
        pts = [25] * 8 + [10] * 4
        games = self._make_games(n=12, pts=pts)
        features = gen.generate_features(games, '2025-01-15')
        assert features is not None
        # recent_pts_avg = mean of last 10 games (sorted desc):
        #   recent[0..3] = 10 pts, recent[4..9] = 25 pts → avg ≈ (4*10 + 6*25)/10 = 19
        # EWMA weights recent[0..3] (cold games) more heavily → should be < 19
        assert features['pts_ewma'] < features['recent_pts_avg'], (
            f"EWMA ({features['pts_ewma']:.1f}) should be below simple avg "
            f"({features['recent_pts_avg']:.1f}) during a cold streak"
        )

    def test_hot_cold_score_positive_for_hot_player(self):
        """Hot streak: fg3m_hot_cold_score should be positive."""
        gen = self._make_generator()
        # First 9 games: 1 three, last 3 games: 4 threes (hot)
        fg3m = [1] * 9 + [4] * 3
        games = self._make_games(n=12, fg3m=fg3m)
        features = gen.generate_features(games, '2025-01-15')
        assert features is not None
        assert 'fg3m_hot_cold_score' in features
        assert features['fg3m_hot_cold_score'] > 0, (
            f"Hot streak: expected positive hot_cold_score, "
            f"got {features['fg3m_hot_cold_score']:.2f}"
        )

    def test_opp_fg3_pct_default_present(self):
        """Opponent 3PT defense features should have neutral defaults."""
        gen = self._make_generator()
        games = self._make_games()
        features = gen.generate_features(games, '2025-01-15')
        assert features is not None
        assert 'opp_fg3_pct_allowed' in features
        assert abs(features['opp_fg3_pct_allowed'] - 0.36) < 0.01  # League average


# ============================================================================
# 8. Integration: over_probability for threes vs edge calculator consistency
# ============================================================================

class TestProbabilityConsistency:
    """Tests that over_probability and EdgeCalculator are consistent for threes."""

    def test_threes_probability_matches_between_modules(self):
        """over_probability() and EdgeCalculator should produce matching threes probs."""
        from nba_models.inference.daily_predictions import over_probability
        from edge_calculator.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()

        for mu, line in [(2.0, 1.5), (3.0, 2.5), (1.5, 2.0), (4.0, 3.5)]:
            daily_pred_prob = over_probability(mu, line, 'threes')
            edge_result = calc.calculate_edge_from_prediction(mu, line, prop_type='threes')

            assert abs(daily_pred_prob - edge_result.model_probability) < 0.01, (
                f"mu={mu}, line={line}: "
                f"daily_pred={daily_pred_prob:.3f} vs edge_calc={edge_result.model_probability:.3f}"
            )
