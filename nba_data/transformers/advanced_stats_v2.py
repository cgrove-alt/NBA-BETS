"""
Advanced Statistics Module V2 - Dean Oliver's Four Factors

Dean Oliver identified the Four Factors that win basketball games:
1. eFG% (Effective Field Goal Percentage) - Shooting efficiency
2. TOV% (Turnover Rate) - Ball security
3. ORB% (Offensive Rebound Percentage) - Second chance opportunities
4. FT/FGA (Free Throw Rate) - Getting to the line

This module calculates:
- Team-level Four Factors
- Rolling 5/10 game differentials
- Matchup-specific factors
- Style clash indicators

Usage:
    from advanced_stats_v2 import FourFactorsCalculator
    calc = FourFactorsCalculator()
    calc.add_game(team_id, game_date, stats)
    features = calc.get_four_factors_before_date(team_id, game_date)
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict


class FourFactorsCalculator:
    """
    Calculate Dean Oliver's Four Factors for NBA teams.

    The Four Factors explain ~90% of wins/losses:
    1. Shooting (40% weight) - eFG%
    2. Turnovers (25% weight) - TOV%
    3. Rebounding (20% weight) - ORB%
    4. Free Throws (15% weight) - FT/FGA

    Reference: Basketball on Paper by Dean Oliver (2004)
    """

    # Factor weights per Dean Oliver
    FACTOR_WEIGHTS = {
        'efg_pct': 0.40,
        'tov_pct': 0.25,
        'orb_pct': 0.20,
        'ft_rate': 0.15,
    }

    # League averages (2025-26 season estimates)
    LEAGUE_AVG = {
        'efg_pct': 0.530,     # ~53% eFG league-wide
        'tov_pct': 0.130,     # ~13% turnover rate
        'orb_pct': 0.260,     # ~26% offensive rebound rate
        'ft_rate': 0.215,     # ~0.215 FT/FGA ratio
        'pace': 100.0,        # Possessions per 48 minutes
        'off_rating': 114.0,  # Points per 100 possessions
        'def_rating': 114.0,  # Points allowed per 100 possessions
    }

    def __init__(self):
        # Structure: team_id -> [(game_date, stats_dict), ...]
        self.team_games = defaultdict(list)
        # Structure: team_id -> opponent_id -> [(game_date, stats_dict), ...]
        self.matchup_history = defaultdict(lambda: defaultdict(list))

    def _estimate_possessions(self, team_stats: dict, opp_stats: dict = None) -> float:
        """
        Estimate possessions using the official NBA formula.

        Possessions ≈ FGA + 0.4*FTA - 1.07*(ORB/(ORB+DRB_opp))*(FGA-FGM) + TOV

        Simplified when opponent stats not available:
        Possessions ≈ FGA - ORB + TOV + 0.44*FTA
        """
        fga = team_stats.get('fga', 0) or 0
        fta = team_stats.get('fta', 0) or 0
        orb = team_stats.get('oreb', 0) or team_stats.get('orb', 0) or 0
        tov = team_stats.get('tov', 0) or team_stats.get('turnover', 0) or 0
        fgm = team_stats.get('fgm', 0) or 0

        if opp_stats:
            drb_opp = opp_stats.get('dreb', 0) or opp_stats.get('drb', 0) or 0
            total_reb = orb + drb_opp
            orb_factor = (orb / total_reb) if total_reb > 0 else 0.26
            poss = fga + 0.4 * fta - 1.07 * orb_factor * (fga - fgm) + tov
        else:
            # Simplified formula
            poss = fga - orb + tov + 0.44 * fta

        return max(poss, 1.0)  # Avoid division by zero

    def calculate_efg_pct(self, stats: dict) -> float:
        """
        Calculate Effective Field Goal Percentage.

        eFG% = (FGM + 0.5 * FG3M) / FGA

        This weights 3-pointers as worth 1.5 times a 2-pointer.
        """
        fgm = stats.get('fgm', 0) or 0
        fg3m = stats.get('fg3m', 0) or 0
        fga = stats.get('fga', 0) or 0

        if fga == 0:
            return self.LEAGUE_AVG['efg_pct']

        return (fgm + 0.5 * fg3m) / fga

    def calculate_tov_pct(self, stats: dict) -> float:
        """
        Calculate Turnover Rate.

        TOV% = TOV / (FGA + 0.44*FTA + TOV)

        Lower is better - represents turnovers per play.
        """
        tov = stats.get('tov', 0) or stats.get('turnover', 0) or 0
        fga = stats.get('fga', 0) or 0
        fta = stats.get('fta', 0) or 0

        plays = fga + 0.44 * fta + tov
        if plays == 0:
            return self.LEAGUE_AVG['tov_pct']

        return tov / plays

    def calculate_orb_pct(self, stats: dict, opp_stats: dict = None) -> float:
        """
        Calculate Offensive Rebound Percentage.

        ORB% = ORB / (ORB + Opp_DRB)

        If opponent stats unavailable, use league average DRB.
        """
        orb = stats.get('oreb', 0) or stats.get('orb', 0) or 0

        if opp_stats:
            drb_opp = opp_stats.get('dreb', 0) or opp_stats.get('drb', 0) or 0
        else:
            # Estimate: opponent gets ~74% of defensive rebounds
            total_reb = stats.get('reb', 0) or 0
            drb_opp = total_reb * 0.74 if total_reb > 0 else 35

        total = orb + drb_opp
        if total == 0:
            return self.LEAGUE_AVG['orb_pct']

        return orb / total

    def calculate_ft_rate(self, stats: dict) -> float:
        """
        Calculate Free Throw Rate.

        FT Rate = FTA / FGA

        Higher is better - represents ability to get to the line.
        """
        fta = stats.get('fta', 0) or 0
        fga = stats.get('fga', 0) or 0

        if fga == 0:
            return self.LEAGUE_AVG['ft_rate']

        return fta / fga

    def calculate_four_factors(self, stats: dict, opp_stats: dict = None) -> dict[str, float]:
        """
        Calculate all Four Factors for a single game.

        Args:
            stats: Team stats dictionary
            opp_stats: Opponent stats dictionary (optional, for ORB% accuracy)

        Returns:
            Dictionary with all four factors
        """
        return {
            'efg_pct': round(self.calculate_efg_pct(stats), 4),
            'tov_pct': round(self.calculate_tov_pct(stats), 4),
            'orb_pct': round(self.calculate_orb_pct(stats, opp_stats), 4),
            'ft_rate': round(self.calculate_ft_rate(stats), 4),
        }

    def add_game(self, team_id: int, game_date: str, stats: dict,
                 opponent_id: int = None, opp_stats: dict = None):
        """
        Add a game's stats for a team.

        Args:
            team_id: Team identifier
            game_date: Date in YYYY-MM-DD format
            stats: Team stats dictionary with fgm, fga, fg3m, fta, orb/oreb, tov, etc.
                   If 'poss' key is present in stats, it will be used directly.
                   Otherwise, possessions will be estimated using the NBA formula.
            opponent_id: Opponent team ID (optional)
            opp_stats: Opponent stats (optional)

        Note:
            The stats dict is spread into the game data, so any additional keys
            (e.g., 'poss', 'minutes_played') will be preserved for later use.
        """
        # Calculate Four Factors
        four_factors = self.calculate_four_factors(stats, opp_stats)

        # Calculate additional stats
        pts = stats.get('pts', 0) or 0
        poss = self._estimate_possessions(stats, opp_stats)

        game_data = {
            'date': game_date,
            'pts': pts,
            'poss': poss,
            'off_rating': (pts / poss * 100) if poss > 0 else 100,
            **four_factors,
            **stats,  # Include raw stats for later use
        }

        self.team_games[team_id].append((game_date, game_data))

        # Sort by date
        self.team_games[team_id].sort(key=lambda x: x[0])

        # Add to matchup history if opponent known
        if opponent_id:
            self.matchup_history[team_id][opponent_id].append((game_date, game_data))
            self.matchup_history[team_id][opponent_id].sort(key=lambda x: x[0])

    def get_four_factors_before_date(self, team_id: int, game_date: str,
                                      window: int = 10, min_games: int = 3) -> "dict | None":
        """
        Get Four Factors features using only games BEFORE game_date.

        This is the CRITICAL temporal-safe function for training/prediction.

        Args:
            team_id: Team identifier
            game_date: Target date (exclusive)
            window: Number of recent games to consider (default 10)
            min_games: Minimum games required (default 3)

        Returns:
            Dictionary with Four Factors features or None if insufficient data
        """
        if team_id not in self.team_games:
            return None

        # Filter to games before target date
        games = [(d, s) for d, s in self.team_games[team_id] if d < game_date]

        if len(games) < min_games:
            return self._get_default_features()

        # Get recent games
        recent = games[-window:]
        last_5 = games[-5:] if len(games) >= 5 else games
        last_3 = games[-3:] if len(games) >= 3 else games
        season = games  # All available games

        # Calculate rolling averages
        features = {}

        # Season averages
        for stat in ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate', 'off_rating']:
            values = [g[1].get(stat, self.LEAGUE_AVG.get(stat, 0)) for g in season]
            features[f'season_{stat}'] = round(np.mean(values), 4)

        # Recent (last 10) averages
        for stat in ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate', 'off_rating']:
            values = [g[1].get(stat, self.LEAGUE_AVG.get(stat, 0)) for g in recent]
            features[f'recent_{stat}'] = round(np.mean(values), 4)

        # Last 5 games
        for stat in ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate', 'off_rating']:
            values = [g[1].get(stat, self.LEAGUE_AVG.get(stat, 0)) for g in last_5]
            features[f'last5_{stat}'] = round(np.mean(values), 4)

        # Last 3 games
        for stat in ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate', 'off_rating']:
            values = [g[1].get(stat, self.LEAGUE_AVG.get(stat, 0)) for g in last_3]
            features[f'last3_{stat}'] = round(np.mean(values), 4)

        # Trends (last 5 vs season)
        for stat in ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate']:
            trend = features[f'last5_{stat}'] - features[f'season_{stat}']
            features[f'{stat}_trend'] = round(trend, 4)

        # Variance (consistency indicator)
        for stat in ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate']:
            values = [g[1].get(stat, self.LEAGUE_AVG.get(stat, 0)) for g in recent]
            features[f'{stat}_std'] = round(np.std(values), 4) if len(values) > 1 else 0.0

        # Differential from league average
        for stat in ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate']:
            diff = features[f'season_{stat}'] - self.LEAGUE_AVG[stat]
            features[f'{stat}_vs_league'] = round(diff, 4)

        # Composite Four Factor Score (weighted)
        # Note: For TOV%, lower is better, so we invert it
        # Avoid division by zero
        tov_pct_safe = features['season_tov_pct'] if features['season_tov_pct'] > 0 else self.LEAGUE_AVG['tov_pct']

        composite = (
            self.FACTOR_WEIGHTS['efg_pct'] * (features['season_efg_pct'] / self.LEAGUE_AVG['efg_pct']) +
            self.FACTOR_WEIGHTS['tov_pct'] * (self.LEAGUE_AVG['tov_pct'] / tov_pct_safe) +
            self.FACTOR_WEIGHTS['orb_pct'] * (features['season_orb_pct'] / self.LEAGUE_AVG['orb_pct']) +
            self.FACTOR_WEIGHTS['ft_rate'] * (features['season_ft_rate'] / self.LEAGUE_AVG['ft_rate'])
        )
        features['four_factor_composite'] = round(composite, 4)

        return features

    def get_matchup_four_factors(self, team_id: int, opponent_id: int,
                                  game_date: str, min_games: int = 1) -> "dict | None":
        """
        Get Four Factors for head-to-head matchup history.

        Args:
            team_id: Team identifier
            opponent_id: Opponent identifier
            game_date: Target date (exclusive)
            min_games: Minimum H2H games required

        Returns:
            Dictionary with matchup-specific features or None
        """
        if opponent_id not in self.matchup_history[team_id]:
            return None

        # Filter to games before target date
        h2h_games = [
            (d, s) for d, s in self.matchup_history[team_id][opponent_id]
            if d < game_date
        ]

        if len(h2h_games) < min_games:
            return None

        features = {}

        # H2H averages
        for stat in ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate', 'off_rating']:
            values = [g[1].get(stat, self.LEAGUE_AVG.get(stat, 0)) for g in h2h_games]
            features[f'h2h_{stat}'] = round(np.mean(values), 4)

        features['h2h_games'] = len(h2h_games)

        return features

    def calculate_four_factor_differential(
        self,
        home_id: int,
        away_id: int,
        game_date: str
    ) -> "dict | None":
        """
        Calculate Four Factor differential between two teams.

        This is the key input for spread/moneyline predictions.

        Args:
            home_id: Home team ID
            away_id: Away team ID
            game_date: Target date

        Returns:
            Dictionary with differential features or None
        """
        home_factors = self.get_four_factors_before_date(home_id, game_date)
        away_factors = self.get_four_factors_before_date(away_id, game_date)

        if not home_factors or not away_factors:
            return None

        features = {}

        # Raw differentials (home - away)
        for stat in ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate', 'off_rating']:
            home_val = home_factors.get(f'season_{stat}', self.LEAGUE_AVG.get(stat, 0))
            away_val = away_factors.get(f'season_{stat}', self.LEAGUE_AVG.get(stat, 0))
            features[f'{stat}_diff'] = round(home_val - away_val, 4)

        # Composite differential
        features['four_factor_composite_diff'] = round(
            home_factors.get('four_factor_composite', 1.0) -
            away_factors.get('four_factor_composite', 1.0),
            4
        )

        # Include raw values for both teams
        for key, value in home_factors.items():
            features[f'home_{key}'] = value
        for key, value in away_factors.items():
            features[f'away_{key}'] = value

        return features

    def calculate_pace(self, team_id: int, game_date: str, window: str = 'season') -> float:
        """
        Calculate team's pace (possessions per 48 minutes).

        Args:
            team_id: Team identifier
            game_date: Target date (exclusive)
            window: 'season', 'last10', 'last5', or 'last3'

        Returns:
            Average pace for the specified window
        """
        if team_id not in self.team_games:
            return self.LEAGUE_AVG['pace']

        # Filter to games before target date
        games = [(d, s) for d, s in self.team_games[team_id] if d < game_date]

        if not games:
            return self.LEAGUE_AVG['pace']

        # Select window
        if window == 'last10':
            games = games[-10:]
        elif window == 'last5':
            games = games[-5:]
        elif window == 'last3':
            games = games[-3:]
        # else: use all games (season)

        # Calculate average pace (possessions already normalized to 48 minutes)
        paces = [g[1].get('poss', 100) for g in games]
        return round(np.mean(paces), 2) if paces else self.LEAGUE_AVG['pace']

    def adjust_for_pace(self, stat_value: float, team_pace: float,
                       per_100: bool = True) -> float:
        """
        Adjust a stat for pace to normalize comparisons.

        Args:
            stat_value: Raw stat value (e.g., points, rebounds)
            team_pace: Team's pace (possessions per 48 min)
            per_100: If True, return per-100 possessions; if False, return per-game

        Returns:
            Pace-adjusted stat value
        """
        if team_pace <= 0:
            team_pace = self.LEAGUE_AVG['pace']

        if per_100:
            # Convert to per 100 possessions
            return round(stat_value / team_pace * 100, 2)
        # Adjust to league-average pace
        return round(stat_value / team_pace * self.LEAGUE_AVG['pace'], 2)

    def _get_default_features(self) -> dict:
        """Return default features when insufficient data."""
        features = {}

        for prefix in ['season', 'recent', 'last5', 'last3']:
            for stat in ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate', 'off_rating']:
                features[f'{prefix}_{stat}'] = self.LEAGUE_AVG.get(stat, 0)

        for stat in ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate']:
            features[f'{stat}_trend'] = 0.0
            features[f'{stat}_std'] = 0.0
            features[f'{stat}_vs_league'] = 0.0

        features['four_factor_composite'] = 1.0

        return features


class StyleClashCalculator:
    """
    Calculate style clash features between teams.

    Identifies matchup advantages based on playing style:
    - Pace mismatch (fast vs slow teams)
    - 3PT heavy vs interior defense
    - Size mismatches
    - Transition offense vs half-court defense
    """

    STYLE_THRESHOLDS = {
        'fast_pace': 102.0,       # High pace threshold
        'slow_pace': 98.0,        # Low pace threshold
        'three_heavy': 0.40,      # 3PA/FGA ratio
        'interior_heavy': 0.30,   # Paint FGA ratio (estimated)
    }

    def __init__(self):
        self.team_styles = defaultdict(dict)

    def calculate_team_style(self, team_id: int, games: list[tuple[str, dict]]) -> dict:
        """
        Identify team's playing style from recent games.

        Returns style indicators and classifications.
        """
        if not games:
            return self._get_default_style()

        # Calculate style metrics
        paces = []
        fg3a_rates = []
        ft_rates = []
        off_ratings = []

        for _, stats in games:
            fga = stats.get('fga', 0) or 0
            fg3a = stats.get('fg3a', 0) or 0
            fta = stats.get('fta', 0) or 0
            poss = stats.get('poss', 100) or 100
            pts = stats.get('pts', 0) or 0

            if fga > 0:
                fg3a_rates.append(fg3a / fga)
                ft_rates.append(fta / fga)
            paces.append(poss)  # Possessions already normalized to 48 minutes
            if poss > 0:
                off_ratings.append(pts / poss * 100)

        style = {
            'avg_pace': np.mean(paces) if paces else 100,
            'avg_fg3a_rate': np.mean(fg3a_rates) if fg3a_rates else 0.35,
            'avg_ft_rate': np.mean(ft_rates) if ft_rates else 0.20,
            'avg_off_rating': np.mean(off_ratings) if off_ratings else 114,
            'pace_std': np.std(paces) if len(paces) > 1 else 3.0,
        }

        # Style classifications
        style['is_fast_paced'] = 1 if style['avg_pace'] > self.STYLE_THRESHOLDS['fast_pace'] else 0
        style['is_slow_paced'] = 1 if style['avg_pace'] < self.STYLE_THRESHOLDS['slow_pace'] else 0
        style['is_three_heavy'] = 1 if style['avg_fg3a_rate'] > self.STYLE_THRESHOLDS['three_heavy'] else 0
        style['is_physical'] = 1 if style['avg_ft_rate'] > 0.25 else 0

        return style

    def calculate_style_clash(self, home_style: dict, away_style: dict) -> dict:
        """
        Calculate style clash features between two teams.

        Identifies potential advantages/disadvantages from style differences.
        """
        features = {}

        # Pace mismatch
        pace_diff = home_style['avg_pace'] - away_style['avg_pace']
        features['pace_mismatch'] = round(pace_diff, 2)
        features['pace_mismatch_abs'] = round(abs(pace_diff), 2)

        # Fast vs slow clash
        features['fast_vs_slow'] = (
            1 if home_style['is_fast_paced'] and away_style['is_slow_paced'] else
            -1 if home_style['is_slow_paced'] and away_style['is_fast_paced'] else 0
        )

        # 3PT style differential
        fg3_diff = home_style['avg_fg3a_rate'] - away_style['avg_fg3a_rate']
        features['three_pt_style_diff'] = round(fg3_diff, 4)

        # Physical style differential
        ft_diff = home_style['avg_ft_rate'] - away_style['avg_ft_rate']
        features['physical_style_diff'] = round(ft_diff, 4)

        # Offensive efficiency differential
        off_diff = home_style['avg_off_rating'] - away_style['avg_off_rating']
        features['off_rating_diff'] = round(off_diff, 2)

        # Expected game pace (average of both)
        features['expected_pace'] = round(
            (home_style['avg_pace'] + away_style['avg_pace']) / 2, 1
        )

        # Style compatibility score (0-1, higher = more similar styles)
        pace_sim = 1 - abs(pace_diff) / 10
        fg3_sim = 1 - abs(fg3_diff) / 0.2
        ft_sim = 1 - abs(ft_diff) / 0.15
        features['style_compatibility'] = round(
            max(0, (pace_sim + fg3_sim + ft_sim) / 3), 3
        )

        return features

    def _get_default_style(self) -> dict:
        """Return default style when insufficient data."""
        return {
            'avg_pace': 100.0,
            'avg_fg3a_rate': 0.35,
            'avg_ft_rate': 0.20,
            'avg_off_rating': 114.0,
            'pace_std': 3.0,
            'is_fast_paced': 0,
            'is_slow_paced': 0,
            'is_three_heavy': 0,
            'is_physical': 0,
        }


def generate_advanced_game_features(
    four_factors_calc: FourFactorsCalculator,
    style_calc: StyleClashCalculator,
    home_id: int,
    away_id: int,
    game_date: str
) -> dict:
    """
    Generate all advanced features for a game prediction.

    Combines:
    - Four Factors differential
    - Style clash features
    - Matchup history

    Args:
        four_factors_calc: FourFactorsCalculator instance
        style_calc: StyleClashCalculator instance
        home_id: Home team ID
        away_id: Away team ID
        game_date: Target date

    Returns:
        Complete feature dictionary
    """
    features = {}

    # Get Four Factors differential
    ff_diff = four_factors_calc.calculate_four_factor_differential(
        home_id, away_id, game_date
    )
    if ff_diff:
        features.update(ff_diff)

    # Get team styles
    home_games = four_factors_calc.team_games.get(home_id, [])
    away_games = four_factors_calc.team_games.get(away_id, [])

    home_recent = [(d, s) for d, s in home_games if d < game_date][-10:]
    away_recent = [(d, s) for d, s in away_games if d < game_date][-10:]

    home_style = style_calc.calculate_team_style(home_id, home_recent)
    away_style = style_calc.calculate_team_style(away_id, away_recent)

    # Add style features
    for key, val in home_style.items():
        features[f'home_{key}'] = val
    for key, val in away_style.items():
        features[f'away_{key}'] = val

    # Get style clash
    clash = style_calc.calculate_style_clash(home_style, away_style)
    features.update(clash)

    # Get H2H history
    h2h = four_factors_calc.get_matchup_four_factors(home_id, away_id, game_date)
    if h2h:
        features.update(h2h)

    return features


# Utility function for quick integration
def add_four_factors_to_features(existing_features: dict, team_stats: dict) -> dict:
    """
    Quick utility to add Four Factors to an existing feature dictionary.

    Args:
        existing_features: Existing feature dict
        team_stats: Raw team stats with fgm, fga, fg3m, fta, orb, tov

    Returns:
        Enhanced feature dictionary
    """
    calc = FourFactorsCalculator()
    factors = calc.calculate_four_factors(team_stats)

    # Add with prefix to avoid conflicts
    enhanced = existing_features.copy()
    for key, val in factors.items():
        enhanced[f'game_{key}'] = val

    return enhanced
