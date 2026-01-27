"""
Shared Feature Generator - Single Source of Truth

This module contains the DEFINITIVE feature generation logic used by:
1. Training scripts (train_complete_balldontlie.py, etc.)
2. Backtest script (comprehensive_backtest.py)
3. Production prediction (daily_predictions.py)

CRITICAL: Any changes to features MUST be made here and nowhere else.
This ensures 100% consistency between training and prediction.

Created: 2026-01-14
Purpose: Fix feature mismatch bug discovered during Phase 1
"""

import numpy as np


class PlayerFeatureGenerator:
    """
    Generates all 150 features for player prop predictions.

    This is the canonical feature generation logic. Training and prediction
    MUST use this exact implementation to ensure consistency.
    """

    def __init__(self):
        """Initialize feature generator."""
        pass

    def generate_features(
        self,
        player_games: list[tuple[str, dict]],  # [(date, stats), ...]
        game_date: str,
        opponent_id: int = None,
        is_home: bool = True,
        player_position: str = 'F',
        position_defense_features: dict = None,
        team_pace: float = 100.0,
        opp_pace: float = 100.0,
    ) -> dict[str, float] | None:
        """
        Generate all 150 features for a player for a specific game.

        Args:
            player_games: List of (date, stats) tuples for player's history
            game_date: Date of game to predict
            opponent_id: Opponent team ID
            is_home: Whether playing at home
            player_position: Player's position (G, F, C)
            position_defense_features: Opponent defense vs position
            team_pace: Team's pace
            opp_pace: Opponent's pace

        Returns:
            Dictionary with all 150 features, or None if insufficient data
        """

        # Filter to games before this date
        games = [(d, s) for d, s in player_games if d < game_date]
        if len(games) < 3:  # Need minimum games
            return None

        # Sort by date descending (most recent first)
        games.sort(key=lambda x: x[0], reverse=True)

        # Define windows
        window = 10
        recent = games[:window]
        last_5 = games[:5]
        last_3 = games[:3]

        # Calculate days rest
        try:
            from datetime import datetime
            current_date = datetime.strptime(game_date, "%Y-%m-%d")
            last_game_date = datetime.strptime(games[0][0], "%Y-%m-%d")
            days_rest = (current_date - last_game_date).days
        except:
            days_rest = 2

        # Helper functions
        def parse_min(min_val):
            if isinstance(min_val, (int, float)):
                return float(min_val)
            if not min_val:
                return 0.0
            try:
                if ':' in str(min_val):
                    parts = str(min_val).split(':')
                    return float(parts[0]) + float(parts[1]) / 60
                return float(min_val)
            except:
                return 0.0

        def get_stat(s, key):
            return s.get(key, 0) or 0

        # Extract stats
        pts = [get_stat(s, 'pts') for _, s in recent]
        reb = [get_stat(s, 'reb') for _, s in recent]
        ast = [get_stat(s, 'ast') for _, s in recent]
        fg3m = [get_stat(s, 'fg3m') for _, s in recent]
        mins = [parse_min(s.get('min', 0)) for _, s in recent]

        # Season stats
        season_pts = [get_stat(s, 'pts') for _, s in games]
        season_reb = [get_stat(s, 'reb') for _, s in games]
        season_ast = [get_stat(s, 'ast') for _, s in games]
        season_fg3m = [get_stat(s, 'fg3m') for _, s in games]
        season_mins = [parse_min(s.get('min', 0)) for _, s in games]

        # Calculate averages
        season_pts_avg = np.mean(season_pts)
        season_reb_avg = np.mean(season_reb)
        season_ast_avg = np.mean(season_ast)
        season_fg3m_avg = np.mean(season_fg3m)
        season_min_avg = np.mean(season_mins)

        # Build features dictionary
        features = {}

        # ==========================================
        # BASIC FEATURES (1-50)
        # ==========================================

        # Season averages (1-6)
        features['season_games'] = len(games)
        features['season_pts_avg'] = season_pts_avg
        features['season_reb_avg'] = season_reb_avg
        features['season_ast_avg'] = season_ast_avg
        features['season_fg3m_avg'] = season_fg3m_avg
        features['season_min_avg'] = season_min_avg

        # Recent averages (7-17)
        features['recent_pts_avg'] = np.mean(pts)
        features['recent_pts_std'] = np.std(pts) if len(pts) > 1 else 0
        features['recent_pts_min'] = np.min(pts) if pts else 0
        features['recent_pts_max'] = np.max(pts) if pts else 0
        features['recent_reb_avg'] = np.mean(reb)
        features['recent_reb_std'] = np.std(reb) if len(reb) > 1 else 0
        features['recent_ast_avg'] = np.mean(ast)
        features['recent_ast_std'] = np.std(ast) if len(ast) > 1 else 0
        features['recent_fg3m_avg'] = np.mean(fg3m)
        features['recent_fg3m_std'] = np.std(fg3m) if len(fg3m) > 1 else 0
        features['recent_min_avg'] = np.mean(mins)

        # Minutes features (18-20)
        last5_mins = [parse_min(s.get('min', 0)) for _, s in last_5]
        features['min_trend'] = np.mean(last5_mins) - np.mean(mins) if mins else 0
        features['min_consistency'] = 1 - (np.std(mins) / np.mean(mins)) if np.mean(mins) > 0 else 0
        features['last5_min_avg'] = np.mean(last5_mins)

        # Last 5 games (21-24)
        features['last5_pts_avg'] = np.mean([get_stat(s, 'pts') for _, s in last_5])
        features['last5_reb_avg'] = np.mean([get_stat(s, 'reb') for _, s in last_5])
        features['last5_ast_avg'] = np.mean([get_stat(s, 'ast') for _, s in last_5])
        features['last5_fg3m_avg'] = np.mean([get_stat(s, 'fg3m') for _, s in last_5])

        # Last 3 games (25-29)
        features['last3_pts_avg'] = np.mean([get_stat(s, 'pts') for _, s in last_3])
        features['last3_reb_avg'] = np.mean([get_stat(s, 'reb') for _, s in last_3])
        features['last3_ast_avg'] = np.mean([get_stat(s, 'ast') for _, s in last_3])
        features['last3_fg3m_avg'] = np.mean([get_stat(s, 'fg3m') for _, s in last_3])
        features['last3_min_avg'] = np.mean([parse_min(s.get('min', 0)) for _, s in last_3])

        # Trends (30-33)
        features['pts_trend'] = features['last5_pts_avg'] - features['recent_pts_avg']
        features['reb_trend'] = features['last5_reb_avg'] - features['recent_reb_avg']
        features['ast_trend'] = features['last5_ast_avg'] - features['recent_ast_avg']
        features['fg3m_trend'] = features['last5_fg3m_avg'] - features['recent_fg3m_avg']

        # Season variance (34-37)
        features['season_pts_std'] = np.std(season_pts) if len(season_pts) > 1 else 0
        features['season_reb_std'] = np.std(season_reb) if len(season_reb) > 1 else 0
        features['season_ast_std'] = np.std(season_ast) if len(season_ast) > 1 else 0
        features['season_fg3m_std'] = np.std(season_fg3m) if len(season_fg3m) > 1 else 0

        # Combined PRA stats (38-40)
        features['pra_avg'] = np.mean([get_stat(s, 'pts') + get_stat(s, 'reb') + get_stat(s, 'ast') for _, s in recent])
        features['pra_std'] = np.std([get_stat(s, 'pts') + get_stat(s, 'reb') + get_stat(s, 'ast') for _, s in recent]) if len(recent) > 1 else 0
        features['last3_pra_avg'] = np.mean([get_stat(s, 'pts') + get_stat(s, 'reb') + get_stat(s, 'ast') for _, s in last_3])

        # Efficiency stats (41-45)
        features['ts_pct'] = self._calc_ts_pct(recent)
        features['efg_pct'] = self._calc_efg_pct(recent)
        features['usage_rate'] = self._calc_usage_rate(recent)
        features['fg3_rate'] = self._calc_fg3_rate(recent)
        features['fta_rate'] = self._calc_fta_rate(recent)

        # Advanced stats (46-48)
        features['bpm'] = self._calc_bpm(recent)
        features['assist_rate'] = self._calc_assist_rate(recent)
        features['rebound_rate'] = self._calc_rebound_rate(recent)

        # Rest features (49-50)
        features['days_rest'] = days_rest
        features['is_back_to_back'] = 1 if days_rest == 1 else 0

        # ==========================================
        # 3-POINT FEATURES (51-66)
        # ==========================================

        features['fg3_pct'] = self._calc_fg3_pct(recent)
        features['last5_fg3_pct'] = self._calc_fg3_pct(last_5)
        features['fg3_pct_variance'] = self._calc_fg3_variance(games)

        streak_features = self._calc_fg3_streak_features(games)
        features.update(streak_features)

        # Specialized 3PM features
        three_pm_features = self._calc_three_pm_features(recent, games, mins)
        features.update(three_pm_features)

        # ==========================================
        # POSITION/ROLE FEATURES (67-75)
        # ==========================================

        position_features = self._infer_position_features(
            season_pts_avg, season_reb_avg, season_ast_avg, season_min_avg
        )
        features.update(position_features)

        # ==========================================
        # OPPONENT FEATURES (76-88)
        # ==========================================

        # Use defaults (can be enhanced with real opponent stats)
        features['opp_def_rating'] = 114.0
        features['opp_off_rating'] = 114.0
        features['opp_net_rating'] = 0.0
        features['opp_pts_allowed'] = 114.0
        features['opp_pts_allowed_recent'] = 114.0
        features['opp_pts_allowed_std'] = 5.0
        features['opp_pace'] = opp_pace
        features['opp_pace_season'] = opp_pace
        features['opp_def_strength'] = 0.0
        features['opp_reb_factor'] = 1.0
        features['opp_location_def'] = 114.0
        features['opp_win_pct'] = 0.5
        features['opp_recent_win_pct'] = 0.5

        # ==========================================
        # GAME CONTEXT (89-91)
        # ==========================================

        features['is_home'] = 1 if is_home else 0
        features['team_pace'] = team_pace
        features['team_off_rating'] = 114.0

        # ==========================================
        # POSITION DEFENSE FEATURES (92-108)
        # ==========================================

        if position_defense_features:
            features.update(position_defense_features)
        else:
            # Defaults
            for pos in ['guards', 'forwards', 'centers']:
                features[f'opp_pts_allowed_to_{pos}'] = 14.0
                features[f'opp_reb_allowed_to_{pos}'] = 5.0
                features[f'opp_ast_allowed_to_{pos}'] = 3.0
                features[f'opp_fg3m_allowed_to_{pos}'] = 1.2
            features['opp_pts_vs_pos_diff'] = 0.0
            features['opp_reb_vs_pos_diff'] = 0.0
            features['opp_ast_vs_pos_diff'] = 0.0
            features['opp_fg3m_vs_pos_diff'] = 0.0
            features['opp_pts_vs_pos_std'] = 5.0

        # ==========================================
        # ADVANCED FEATURES (109-150) - 42 features
        # ==========================================

        # Pace adjustments (10 features)
        expected_game_pace = (team_pace + opp_pace) / 2.0
        league_avg_pace = 100.0
        pace_vs_average = (expected_game_pace - league_avg_pace) / league_avg_pace
        pace_multiplier = expected_game_pace / league_avg_pace

        features['expected_game_pace'] = expected_game_pace
        features['pace_vs_average'] = pace_vs_average
        features['pace_multiplier'] = pace_multiplier
        features['pace_pts_adjustment'] = (pace_multiplier - 1.0) * season_pts_avg * 0.5
        features['pace_reb_adjustment'] = (pace_multiplier - 1.0) * season_reb_avg * 0.3
        features['pace_ast_adjustment'] = (pace_multiplier - 1.0) * season_ast_avg * 0.4
        features['pace_fg3_adjustment'] = (pace_multiplier - 1.0) * season_fg3m_avg * 0.4
        features['is_high_pace_game'] = 1 if expected_game_pace > 102 else 0
        features['is_low_pace_game'] = 1 if expected_game_pace < 98 else 0
        features['total_multiplier'] = pace_multiplier

        # Regression adjustments (13 features)
        sample_weight = min(1.0, len(games) / 20.0)
        league_avg_pts, league_avg_reb, league_avg_ast, league_avg_fg3m = 14.0, 5.0, 3.0, 1.2

        pts_regressed = sample_weight * season_pts_avg + (1 - sample_weight) * league_avg_pts
        reb_regressed = sample_weight * season_reb_avg + (1 - sample_weight) * league_avg_reb
        ast_regressed = sample_weight * season_ast_avg + (1 - sample_weight) * league_avg_ast
        fg3_regressed = sample_weight * season_fg3m_avg + (1 - sample_weight) * league_avg_fg3m

        features['pts_regressed_estimate'] = pts_regressed
        features['pts_regression_adjustment'] = pts_regressed - season_pts_avg
        features['pts_deviation_from_mean'] = features['recent_pts_avg'] - season_pts_avg
        features['pts_variance_penalty'] = -(features['season_pts_std'] / season_pts_avg) * 2.0 if season_pts_avg > 0 else 0

        features['reb_regression_adjustment'] = reb_regressed - season_reb_avg
        features['reb_deviation_from_mean'] = features['recent_reb_avg'] - season_reb_avg
        features['reb_variance_penalty'] = -(features['season_reb_std'] / season_reb_avg) * 1.5 if season_reb_avg > 0 else 0

        features['ast_regression_adjustment'] = ast_regressed - season_ast_avg
        features['ast_deviation_from_mean'] = features['recent_ast_avg'] - season_ast_avg
        features['ast_variance_penalty'] = -(features['season_ast_std'] / season_ast_avg) * 1.5 if season_ast_avg > 0 else 0

        features['fg3_regression_adjustment'] = fg3_regressed - season_fg3m_avg
        features['fg3_deviation_from_mean'] = features['recent_fg3m_avg'] - season_fg3m_avg
        features['fg3_variance_penalty'] = -(features['season_fg3m_std'] / season_fg3m_avg) * 2.0 if season_fg3m_avg > 0 else 0

        # Recency ratios (4 features)
        features['pts_recency_ratio'] = features['recent_pts_avg'] / season_pts_avg if season_pts_avg > 0 else 1.0
        features['reb_recency_ratio'] = features['recent_reb_avg'] / season_reb_avg if season_reb_avg > 0 else 1.0
        features['ast_recency_ratio'] = features['recent_ast_avg'] / season_ast_avg if season_ast_avg > 0 else 1.0
        features['fg3_recency_ratio'] = features['recent_fg3m_avg'] / season_fg3m_avg if season_fg3m_avg > 0 else 1.0

        # Per-100-possession (3 features)
        features['pts_per_100_poss'] = (season_pts_avg / expected_game_pace) * 100 if expected_game_pace > 0 else season_pts_avg
        features['reb_per_100_poss'] = (season_reb_avg / expected_game_pace) * 100 if expected_game_pace > 0 else season_reb_avg
        features['ast_per_100_poss'] = (season_ast_avg / expected_game_pace) * 100 if expected_game_pace > 0 else season_ast_avg

        # Minutes projections (4 features)
        features['minutes_cv'] = features['season_pts_std'] / season_min_avg if season_min_avg > 0 else 0
        features['minutes_recency_ratio'] = features['recent_min_avg'] / season_min_avg if season_min_avg > 0 else 1.0
        features['expected_min_reduction'] = 0.0
        features['projected_min_factor'] = 1.0

        # Vegas/Total features (5 features - defaults)
        features['vegas_total'] = 220.0
        features['total_vs_average'] = 0.0
        features['total_pts_boost'] = 0.0
        features['is_high_total_game'] = 0
        features['is_low_total_game'] = 0

        # Blowout/Spread features (3 features - defaults)
        features['spread_magnitude'] = 0.0
        features['blowout_probability'] = 0.0
        features['is_likely_blowout'] = 0

        return features

    # ==========================================
    # HELPER METHODS
    # ==========================================

    def _calc_ts_pct(self, games) -> float:
        """Calculate True Shooting %."""
        total_pts = sum((s.get('pts', 0) or 0) for _, s in games)
        total_fga = sum((s.get('fga', 0) or 0) for _, s in games)
        total_fta = sum((s.get('fta', 0) or 0) for _, s in games)
        tsa = 2 * (total_fga + 0.44 * total_fta)
        return round(total_pts / tsa, 3) if tsa > 0 else 0.55

    def _calc_efg_pct(self, games) -> float:
        """Calculate Effective FG%."""
        total_fgm = sum((s.get('fgm', 0) or 0) for _, s in games)
        total_fg3m = sum((s.get('fg3m', 0) or 0) for _, s in games)
        total_fga = sum((s.get('fga', 0) or 0) for _, s in games)
        return round((total_fgm + 0.5 * total_fg3m) / total_fga, 3) if total_fga > 0 else 0.50

    def _calc_usage_rate(self, games) -> float:
        """Calculate approximate usage rate."""
        def parse_min(m):
            if isinstance(m, (int, float)):
                return float(m)
            if not m:
                return 0.0
            try:
                if ':' in str(m):
                    parts = str(m).split(':')
                    return float(parts[0]) + float(parts[1]) / 60
                return float(m)
            except:
                return 0.0

        total_fga = sum((s.get('fga', 0) or 0) for _, s in games)
        total_fta = sum((s.get('fta', 0) or 0) for _, s in games)
        total_tov = sum((s.get('turnover', 0) or 0) for _, s in games)
        total_min = sum(parse_min(s.get('min', 0)) for _, s in games)
        return round((total_fga + 0.44 * total_fta + total_tov) / total_min, 3) if total_min > 0 else 0.2

    def _calc_fg3_rate(self, games) -> float:
        """Calculate 3PT attempt rate."""
        total_fg3a = sum((s.get('fg3a', 0) or 0) for _, s in games)
        total_fga = sum((s.get('fga', 0) or 0) for _, s in games)
        return round(total_fg3a / total_fga, 3) if total_fga > 0 else 0.35

    def _calc_fta_rate(self, games) -> float:
        """Calculate free throw rate."""
        total_fta = sum((s.get('fta', 0) or 0) for _, s in games)
        total_fga = sum((s.get('fga', 0) or 0) for _, s in games)
        return round(total_fta / total_fga, 3) if total_fga > 0 else 0.25

    def _calc_bpm(self, games) -> float:
        """Calculate simplified Box Plus/Minus."""
        # Implementation from comprehensive_backtest.py
        if not games:
            return 0.0

        def parse_min(m):
            if isinstance(m, (int, float)):
                return float(m)
            if not m:
                return 0.0
            try:
                if ':' in str(m):
                    parts = str(m).split(':')
                    return float(parts[0]) + float(parts[1]) / 60
                return float(m)
            except:
                return 0.0

        total_pts = sum((s.get('pts', 0) or 0) for _, s in games)
        total_reb = sum((s.get('reb', 0) or 0) for _, s in games)
        total_ast = sum((s.get('ast', 0) or 0) for _, s in games)
        total_stl = sum((s.get('stl', 0) or 0) for _, s in games)
        total_blk = sum((s.get('blk', 0) or 0) for _, s in games)
        total_tov = sum((s.get('turnover', 0) or 0) for _, s in games)
        total_min = sum(parse_min(s.get('min', 0)) for _, s in games)

        if total_min < 10:
            return 0.0

        per36 = 36.0 / (total_min / len(games))
        bpm = ((total_pts + total_reb * 0.8 + total_ast * 1.1 +
                total_stl * 2.0 + total_blk * 1.5 - total_tov) / len(games) * per36 - 15) / 5
        return round(np.clip(bpm, -10, 10), 2)

    def _calc_assist_rate(self, games) -> float:
        """Calculate assists per 36 minutes."""
        if not games:
            return 4.0

        def parse_min(m):
            if isinstance(m, (int, float)):
                return float(m)
            if not m:
                return 0.0
            try:
                if ':' in str(m):
                    parts = str(m).split(':')
                    return float(parts[0]) + float(parts[1]) / 60
                return float(m)
            except:
                return 0.0

        total_ast = sum((s.get('ast', 0) or 0) for _, s in games)
        total_min = sum(parse_min(s.get('min', 0)) for _, s in games)
        if total_min > 0:
            return round(total_ast / total_min * 36, 2)
        return 4.0

    def _calc_rebound_rate(self, games) -> float:
        """Calculate rebounds per 36 minutes."""
        if not games:
            return 6.0

        def parse_min(m):
            if isinstance(m, (int, float)):
                return float(m)
            if not m:
                return 0.0
            try:
                if ':' in str(m):
                    parts = str(m).split(':')
                    return float(parts[0]) + float(parts[1]) / 60
                return float(m)
            except:
                return 0.0

        total_reb = sum((s.get('reb', 0) or 0) for _, s in games)
        total_min = sum(parse_min(s.get('min', 0)) for _, s in games)
        if total_min > 0:
            return round(total_reb / total_min * 36, 2)
        return 6.0

    def _calc_fg3_pct(self, games) -> float:
        """Calculate 3-point shooting percentage."""
        total_fg3m = sum((s.get('fg3m', 0) or 0) for _, s in games)
        total_fg3a = sum((s.get('fg3a', 0) or 0) for _, s in games)
        if total_fg3a > 0:
            return round(total_fg3m / total_fg3a, 3)
        return 0.36

    def _calc_fg3_variance(self, games) -> float:
        """Calculate variance in 3-point shooting."""
        game_fg3_pcts = []
        for _, g in games:
            fg3a = g.get('fg3a', 0) or 0
            fg3m = g.get('fg3m', 0) or 0
            if fg3a >= 3:
                game_fg3_pcts.append(fg3m / fg3a)
        if len(game_fg3_pcts) >= 3:
            return round(float(np.var(game_fg3_pcts)), 4)
        return 0.1

    def _calc_fg3_streak_features(self, games) -> dict[str, float]:
        """Calculate hot/cold streak features for 3PM."""
        if len(games) < 3:
            return {'fg3_hot_streak': 0, 'fg3_cold_streak': 0, 'fg3_momentum': 0.0}

        recent_fg3_pcts = []
        for _, g in games[:5]:
            fg3a = g.get('fg3a', 0) or 0
            fg3m = g.get('fg3m', 0) or 0
            if fg3a >= 2:
                recent_fg3_pcts.append(fg3m / fg3a)

        if not recent_fg3_pcts:
            return {'fg3_hot_streak': 0, 'fg3_cold_streak': 0, 'fg3_momentum': 0.0}

        hot_streak = sum(1 for p in recent_fg3_pcts[:3] if p >= 0.40) >= 2
        cold_streak = sum(1 for p in recent_fg3_pcts[:3] if p <= 0.30) >= 2

        if len(recent_fg3_pcts) >= 3:
            momentum = np.polyfit(range(len(recent_fg3_pcts)), recent_fg3_pcts, 1)[0]
        else:
            momentum = 0.0

        return {
            'fg3_hot_streak': 1 if hot_streak else 0,
            'fg3_cold_streak': 1 if cold_streak else 0,
            'fg3_momentum': round(float(momentum), 4),
        }

    def _calc_three_pm_features(self, recent, all_games, mins) -> dict[str, float]:
        """Calculate specialized 3PM prediction features."""
        LEAGUE_AVG_FG3_PCT = 0.36

        if not recent:
            return {
                'fg3a_per_min': 0.15,
                'fg3a_avg': 4.5,
                'fg3a_std': 2.0,
                'fg3a_consistency': 0.7,
                'regressed_fg3_pct': LEAGUE_AVG_FG3_PCT,
                'expected_fg3m': 1.5,
                'fg3_makes_std': 1.0,
                'fg3_attempt_trend': 0.0,
                'is_volume_shooter': 0,
                'shooting_confidence': 0.5,
            }

        fg3a_values = [(s.get('fg3a', 0) or 0) for _, s in recent]
        fg3m_values = [(s.get('fg3m', 0) or 0) for _, s in recent]

        fg3a_avg = np.mean(fg3a_values) if fg3a_values else 0
        fg3a_std = np.std(fg3a_values) if len(fg3a_values) > 1 else 2.0
        fg3m_std = np.std(fg3m_values) if len(fg3m_values) > 1 else 1.0

        total_mins = sum(mins)
        total_fg3a = sum(fg3a_values)
        fg3a_per_min = (total_fg3a / total_mins) if total_mins > 0 else 0.15

        fg3a_consistency = 1 - (fg3a_std / max(fg3a_avg, 1)) if fg3a_avg > 0 else 0.5
        fg3a_consistency = max(0.3, min(1.0, fg3a_consistency))

        raw_fg3_pct = (sum(fg3m_values) / sum(fg3a_values)) if sum(fg3a_values) > 0 else LEAGUE_AVG_FG3_PCT

        total_attempts_season = sum((s.get('fg3a', 0) or 0) for _, s in all_games)
        regression_weight = min(1.0, total_attempts_season / 250)
        regressed_fg3_pct = regression_weight * raw_fg3_pct + (1 - regression_weight) * LEAGUE_AVG_FG3_PCT

        expected_fg3m = fg3a_avg * regressed_fg3_pct

        if len(fg3a_values) >= 3:
            last3_fg3a = np.mean(fg3a_values[:3])
            fg3_attempt_trend = last3_fg3a - fg3a_avg
        else:
            fg3_attempt_trend = 0.0

        is_volume_shooter = 1 if fg3a_avg >= 5 else 0
        sample_factor = min(1.0, len(all_games) / 20)
        shooting_confidence = sample_factor * fg3a_consistency

        return {
            'fg3a_per_min': round(fg3a_per_min, 4),
            'fg3a_avg': round(fg3a_avg, 2),
            'fg3a_std': round(fg3a_std, 2),
            'fg3a_consistency': round(fg3a_consistency, 3),
            'regressed_fg3_pct': round(regressed_fg3_pct, 4),
            'expected_fg3m': round(expected_fg3m, 2),
            'fg3_makes_std': round(fg3m_std, 2),
            'fg3_attempt_trend': round(fg3_attempt_trend, 2),
            'is_volume_shooter': is_volume_shooter,
            'shooting_confidence': round(shooting_confidence, 3),
        }

    def _infer_position_features(self, pts_avg, reb_avg, ast_avg, min_avg) -> dict[str, float]:
        """Infer position and role features from stats."""
        if reb_avg > 7 and ast_avg < 3:
            is_center, is_forward, is_guard = 1, 0, 0
        elif ast_avg > 5 and reb_avg < 5:
            is_center, is_forward, is_guard = 0, 0, 1
        elif reb_avg > 5:
            is_center, is_forward, is_guard = 0, 1, 0
        else:
            is_center, is_forward, is_guard = 0, 0, 1

        is_starter = 1 if min_avg >= 25 else 0
        is_star = 1 if pts_avg >= 20 else 0
        is_high_volume = 1 if pts_avg >= 15 else 0
        is_ball_handler = 1 if ast_avg >= 4 else 0

        pos_reb_factor = 1.3 if is_center else (1.0 if is_forward else 0.7)
        pos_ast_factor = 1.3 if is_guard else (0.9 if is_forward else 0.6)

        return {
            'is_guard': is_guard,
            'is_forward': is_forward,
            'is_center': is_center,
            'is_starter': is_starter,
            'is_star': is_star,
            'is_high_volume': is_high_volume,
            'is_ball_handler': is_ball_handler,
            'pos_reb_factor': pos_reb_factor,
            'pos_ast_factor': pos_ast_factor,
        }


# Global singleton instance
_feature_generator = PlayerFeatureGenerator()

def generate_player_features(*args, **kwargs):
    """
    Public API for feature generation.
    Use this function in all training and prediction code.
    """
    return _feature_generator.generate_features(*args, **kwargs)
