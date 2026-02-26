"""
Feature engineering for the Minutes Oracle.

Generates features for predicting player minutes distribution:
- Historical baseline features (season avg, recent avg, trend)
- Game context features (spread, total, home/away, rest)
- Rotation context features (injuries, depth, starter status)
- Coach tendency features (starter mins, blowout patterns)
- Situational features (competitiveness, foul rate, clutch usage)
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Any
from collections import defaultdict

from .coach_tendencies import (
    get_coach_tendency,
    get_blowout_minutes_factor,
    get_b2b_minutes_factor,
    TEAM_IDS,
    TEAM_ID_TO_ABBREV,
)


# Position groups for depth scoring
POSITION_GROUPS = {
    'G': ['PG', 'SG', 'G', 'G-F', 'Point Guard', 'Shooting Guard', 'Guard'],
    'F': ['SF', 'PF', 'F', 'F-G', 'F-C', 'Small Forward', 'Power Forward', 'Forward'],
    'C': ['C', 'C-F', 'Center'],
}


def normalize_position(position: str) -> str:
    """Normalize position to G/F/C."""
    if not position:
        return 'F'  # Default to forward

    position = position.upper().strip()

    for group, positions in POSITION_GROUPS.items():
        if position in [p.upper() for p in positions]:
            return group

    # Handle combo positions
    if 'G' in position:
        return 'G'
    if 'F' in position:
        return 'F'
    if 'C' in position:
        return 'C'

    return 'F'


class MinutesFeatureGenerator:
    """
    Generates features for minutes prediction.

    Features are organized into categories:
    1. Historical baseline - Player's minutes history
    2. Game context - Pre-game situational factors
    3. Rotation context - Team composition and injuries
    4. Coach tendencies - Coach-specific patterns
    5. Situational - Game competitiveness, foul trouble, etc.
    """

    def __init__(self, api_client=None):
        """
        Initialize the feature generator.

        Args:
            api_client: API client for data fetching (e.g., BalldontlieAPI)
        """
        self.api_client = api_client

        # Cache for expensive lookups
        self._player_cache: dict[int, dict] = {}
        self._team_cache: dict[int, dict] = {}
        self._roster_cache: dict[tuple[int, str], list[dict]] = {}

    def generate_features(self,
                          player_id: int,
                          team_id: int,
                          opponent_team_id: int,
                          game_date: str,
                          game_context: dict | None = None,
                          player_game_logs: list[dict] | None = None,
                          team_roster: list[dict] | None = None,
                          injured_players: list[int] | None = None) -> dict[str, float]:
        """
        Generate all features for minutes prediction.

        Args:
            player_id: Player ID
            team_id: Player's team ID
            opponent_team_id: Opponent team ID
            game_date: Game date (YYYY-MM-DD)
            game_context: Pre-game context dict with keys:
                - vegas_spread: Point spread (negative = favorite)
                - vegas_total: Over/under total
                - is_home: True if home game
                - is_back_to_back: True if B2B
                - days_rest: Days since last game
            player_game_logs: List of player's recent game logs
            team_roster: Team roster with player info
            injured_players: List of injured player IDs on team

        Returns:
            Feature dictionary
        """
        features = {}

        # Set defaults for game context
        game_context = game_context or {}
        vegas_spread = game_context.get('vegas_spread', 0.0)
        vegas_total = game_context.get('vegas_total', 220.0)
        is_home = game_context.get('is_home', True)
        is_back_to_back = game_context.get('is_back_to_back', False)
        days_rest = game_context.get('days_rest', 1)

        # 1. Historical baseline features
        baseline_features = self._generate_baseline_features(
            player_game_logs or [],
            game_date
        )
        features.update(baseline_features)

        # 2. Game context features
        context_features = self._generate_context_features(
            vegas_spread=vegas_spread,
            vegas_total=vegas_total,
            is_home=is_home,
            is_back_to_back=is_back_to_back,
            days_rest=days_rest,
            opponent_team_id=opponent_team_id
        )
        features.update(context_features)

        # 3. Rotation context features
        rotation_features = self._generate_rotation_features(
            player_id=player_id,
            team_id=team_id,
            team_roster=team_roster or [],
            injured_players=injured_players or [],
            player_game_logs=player_game_logs or []
        )
        features.update(rotation_features)

        # 4. Coach tendency features
        coach_features = self._generate_coach_features(
            team_id=team_id,
            vegas_spread=vegas_spread,
            is_back_to_back=is_back_to_back,
            is_starter=features.get('is_starter', 1)
        )
        features.update(coach_features)

        # 5. Situational features
        situational_features = self._generate_situational_features(
            player_game_logs=player_game_logs or [],
            vegas_spread=vegas_spread,
            vegas_total=vegas_total,
            is_starter=features.get('is_starter', 1)
        )
        features.update(situational_features)

        return features

    def _generate_baseline_features(self,
                                    game_logs: list[dict],
                                    game_date: str) -> dict[str, float]:
        """
        Generate historical baseline features.

        Features:
        - season_min_avg: Season average minutes
        - recent_min_avg: Last 5 games average
        - recent_min_std: Last 5 games standard deviation
        - min_trend: Trend in minutes (increasing/decreasing)
        - games_played: Sample size indicator
        - last3_min_avg: Last 3 games average
        - min_consistency: 1 / (1 + CV) where CV = std/mean
        """
        features = {
            'season_min_avg': 28.0,  # Default for new/unknown players
            'recent_min_avg': 28.0,
            'recent_min_std': 4.0,
            'min_trend': 0.0,
            'games_played': 0,
            'last3_min_avg': 28.0,
            'min_consistency': 0.7,
            'min_floor': 20.0,
            'min_ceiling': 36.0,
        }

        if not game_logs:
            return features

        # Filter to games before target date and with meaningful minutes
        try:
            target_date = datetime.strptime(game_date, '%Y-%m-%d')
        except (ValueError, TypeError):
            target_date = datetime.now()

        valid_logs = []
        for log in game_logs:
            # Extract minutes
            mins = self._extract_minutes(log)
            if mins is None or mins < 5:  # Skip DNP/garbage time
                continue

            # Check date if available
            log_date_str = log.get('game', {}).get('date') or log.get('date')
            if log_date_str:
                try:
                    log_date = datetime.strptime(log_date_str[:10], '%Y-%m-%d')
                    if log_date >= target_date:
                        continue
                except (ValueError, TypeError):
                    pass

            valid_logs.append({
                'minutes': mins,
                'date': log_date_str or ''  # Ensure never None for sorting
            })

        if not valid_logs:
            return features

        # Sort by date (most recent first) - use empty string for missing dates
        valid_logs.sort(key=lambda x: x.get('date') or '', reverse=True)

        # All minutes values
        all_mins = [log['minutes'] for log in valid_logs]
        features['games_played'] = len(all_mins)
        features['season_min_avg'] = np.mean(all_mins)

        # Recent games (last 5)
        recent_logs = valid_logs[:5]
        recent_mins = [log['minutes'] for log in recent_logs]
        features['recent_min_avg'] = np.mean(recent_mins)
        features['recent_min_std'] = np.std(recent_mins) if len(recent_mins) > 1 else 3.0

        # Last 3 games
        last3_mins = [log['minutes'] for log in valid_logs[:3]]
        features['last3_min_avg'] = np.mean(last3_mins) if last3_mins else features['recent_min_avg']

        # Trend: recent avg vs season avg
        if features['season_min_avg'] > 0:
            features['min_trend'] = features['recent_min_avg'] - features['season_min_avg']

        # Consistency score (higher = more consistent)
        if features['recent_min_avg'] > 0:
            cv = features['recent_min_std'] / features['recent_min_avg']
            features['min_consistency'] = 1.0 / (1.0 + cv)

        # Floor and ceiling from historical data
        features['min_floor'] = np.percentile(all_mins, 10) if len(all_mins) >= 5 else features['season_min_avg'] - 8
        features['min_ceiling'] = np.percentile(all_mins, 90) if len(all_mins) >= 5 else features['season_min_avg'] + 8

        return features

    def _generate_context_features(self,
                                   vegas_spread: float,
                                   vegas_total: float,
                                   is_home: bool,
                                   is_back_to_back: bool,
                                   days_rest: int,
                                   opponent_team_id: int) -> dict[str, float]:
        """
        Generate game context features.

        Features:
        - vegas_spread_abs: Absolute value of spread (blowout indicator)
        - vegas_total: Over/under (pace indicator)
        - is_home: Home/away flag
        - is_back_to_back: B2B flag
        - days_rest: Days since last game
        - is_favorite: 1 if team is favored
        - blowout_risk: Probability of blowout based on spread
        - expected_pace_factor: Expected pace vs league average
        """
        features = {
            'vegas_spread_abs': abs(vegas_spread),
            'vegas_spread': vegas_spread,
            'vegas_total': vegas_total,
            'is_home': 1 if is_home else 0,
            'is_back_to_back': 1 if is_back_to_back else 0,
            'days_rest': min(days_rest, 7),  # Cap at 7
        }

        # Is team favored (negative spread = favored at home)
        if is_home:
            features['is_favorite'] = 1 if vegas_spread < 0 else 0
        else:
            features['is_favorite'] = 1 if vegas_spread > 0 else 0

        # Blowout risk (higher spread = more blowout risk)
        spread_abs = abs(vegas_spread)
        if spread_abs < 5:
            features['blowout_risk'] = 0.1
        elif spread_abs < 8:
            features['blowout_risk'] = 0.2
        elif spread_abs < 12:
            features['blowout_risk'] = 0.35
        else:
            features['blowout_risk'] = 0.5

        # Expected pace factor (220 = league average total)
        features['expected_pace_factor'] = vegas_total / 220.0

        # Rest advantage
        features['rest_advantage'] = 1 if days_rest >= 2 else 0

        return features

    def _generate_rotation_features(self,
                                    player_id: int,
                                    team_id: int,
                                    team_roster: list[dict],
                                    injured_players: list[int],
                                    player_game_logs: list[dict]) -> dict[str, float]:
        """
        Generate rotation context features.

        Features:
        - teammates_injured_minutes: Total minutes of injured teammates
        - position_depth_score: How deep is team at this position (1-5)
        - is_starter: Is player a confirmed starter
        - rotation_spot: Player's rotation position (1=star, 2=starter, 3=rotation, 4=deep bench)
        - injury_boost_factor: Expected minutes boost from injuries
        """
        features = {
            'teammates_injured_minutes': 0.0,
            'position_depth_score': 3.0,  # Average depth
            'is_starter': 1,  # Assume starter if unknown
            'rotation_spot': 2.0,
            'injury_boost_factor': 1.0,
        }

        # Determine if player is a starter based on recent minutes
        if player_game_logs:
            recent_mins = [self._extract_minutes(log) for log in player_game_logs[:5]]
            recent_mins = [m for m in recent_mins if m is not None]
            if recent_mins:
                avg_mins = np.mean(recent_mins)
                if avg_mins >= 28:
                    features['is_starter'] = 1
                    features['rotation_spot'] = 1.5 if avg_mins >= 32 else 2.0
                elif avg_mins >= 18:
                    features['is_starter'] = 0
                    features['rotation_spot'] = 3.0
                else:
                    features['is_starter'] = 0
                    features['rotation_spot'] = 4.0

        # Calculate injured teammates' minutes impact
        if team_roster and injured_players:
            player_position = None
            injured_same_pos_mins = 0.0
            total_injured_mins = 0.0

            # Find player's position
            for player in team_roster:
                pid = player.get('id') or player.get('player_id')
                if pid == player_id:
                    player_position = normalize_position(
                        player.get('position', '') or player.get('pos', '')
                    )
                    break

            # Calculate injured teammates' minutes
            for player in team_roster:
                pid = player.get('id') or player.get('player_id')
                if pid in injured_players and pid != player_id:
                    injured_mins = player.get('season_min_avg', 0) or player.get('mpg', 0) or 20
                    total_injured_mins += injured_mins

                    # Check if same position group
                    pos = normalize_position(
                        player.get('position', '') or player.get('pos', '')
                    )
                    if pos == player_position:
                        injured_same_pos_mins += injured_mins

            features['teammates_injured_minutes'] = total_injured_mins

            # Injury boost factor (more minutes available if teammates out)
            # Assume ~40% of injured players' minutes get redistributed to same position
            if player_position and features['is_starter']:
                minutes_available = injured_same_pos_mins * 0.4
                features['injury_boost_factor'] = 1.0 + (minutes_available / 48.0)

        # Position depth score
        if team_roster:
            player_position = None
            for player in team_roster:
                pid = player.get('id') or player.get('player_id')
                if pid == player_id:
                    player_position = normalize_position(
                        player.get('position', '') or player.get('pos', '')
                    )
                    break

            if player_position:
                # Count players at same position
                same_pos_count = sum(
                    1 for p in team_roster
                    if normalize_position(p.get('position', '') or p.get('pos', '')) == player_position
                    and p.get('id', p.get('player_id')) not in injured_players
                )
                # Depth score: 1 = very thin, 5 = very deep
                features['position_depth_score'] = min(5, max(1, same_pos_count))

        return features

    def _generate_coach_features(self,
                                 team_id: int,
                                 vegas_spread: float,
                                 is_back_to_back: bool,
                                 is_starter: int) -> dict[str, float]:
        """
        Generate coach tendency features.

        Features:
        - coach_starter_min_avg: Coach's average starter minutes
        - coach_bench_min_avg: Coach's average bench minutes
        - coach_blowout_factor: Expected minutes reduction in blowout
        - coach_b2b_factor: Expected minutes reduction on B2B
        - coach_variance_score: How variable are coach's rotations (0-1)
        """
        coach = get_coach_tendency(team_id=team_id)

        features = {
            'coach_starter_min_avg': coach.starter_min_avg,
            'coach_bench_min_avg': coach.bench_min_avg,
            'coach_blowout_pull_lead': coach.blowout_pull_lead,
            'coach_blowout_pull_deficit': coach.blowout_pull_deficit,
        }

        # Variance score (low=0.2, medium=0.5, high=0.8)
        variance_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        features['coach_variance_score'] = variance_map.get(coach.min_variance, 0.5)

        # Expected blowout factor
        is_favored = vegas_spread < -3 if vegas_spread else False
        features['coach_blowout_factor'] = get_blowout_minutes_factor(
            team_id=team_id,
            expected_margin=abs(vegas_spread),
            is_winning=is_favored
        )

        # B2B factor
        if is_back_to_back:
            features['coach_b2b_factor'] = get_b2b_minutes_factor(team_id=team_id)
        else:
            features['coach_b2b_factor'] = 1.0

        # Expected minutes baseline from coach
        if is_starter:
            features['coach_expected_mins'] = coach.starter_min_avg
        else:
            features['coach_expected_mins'] = coach.bench_min_avg

        return features

    def _generate_situational_features(self,
                                       player_game_logs: list[dict],
                                       vegas_spread: float,
                                       vegas_total: float,
                                       is_starter: int) -> dict[str, float]:
        """
        Generate situational features.

        Features:
        - projected_game_competitiveness: How close will game be (0-1)
        - player_foul_rate: Historical fouls per 36 minutes
        - clutch_player_flag: Does player play in close games
        - overtime_likelihood: Probability of overtime
        - blowout_minutes_penalty: Expected minutes lost to blowout
        """
        features = {
            'projected_game_competitiveness': 0.5,
            'player_foul_rate': 3.0,  # League average ~3 fouls per 36
            'clutch_player_flag': 1 if is_starter else 0,
            'overtime_likelihood': 0.06,  # ~6% of games go to OT
            'blowout_minutes_penalty': 0.0,
        }

        # Game competitiveness (inverse of spread)
        spread_abs = abs(vegas_spread)
        if spread_abs < 3:
            features['projected_game_competitiveness'] = 0.9
            features['overtime_likelihood'] = 0.10
        elif spread_abs < 6:
            features['projected_game_competitiveness'] = 0.7
            features['overtime_likelihood'] = 0.07
        elif spread_abs < 10:
            features['projected_game_competitiveness'] = 0.5
            features['overtime_likelihood'] = 0.04
        else:
            features['projected_game_competitiveness'] = 0.3
            features['overtime_likelihood'] = 0.02

        # Blowout minutes penalty for starters
        if is_starter and spread_abs >= 10:
            # Expected 3-5 minutes lost in big blowouts
            features['blowout_minutes_penalty'] = (spread_abs - 10) * 0.3
            features['blowout_minutes_penalty'] = min(5.0, features['blowout_minutes_penalty'])

        # Calculate foul rate from game logs
        if player_game_logs:
            fouls = []
            minutes = []
            for log in player_game_logs[:10]:
                pf = log.get('pf') or log.get('fouls') or log.get('personal_fouls', 0)
                mins = self._extract_minutes(log)
                if mins and mins > 10 and pf is not None:
                    fouls.append(pf)
                    minutes.append(mins)

            if fouls and minutes and sum(minutes) > 0:
                total_fouls = sum(fouls)
                total_mins = sum(minutes)
                features['player_foul_rate'] = (total_fouls / total_mins) * 36

        # Clutch player flag (plays heavy minutes in close games)
        # If starter with high minutes, assume clutch
        if is_starter:
            features['clutch_player_flag'] = 1
        else:
            features['clutch_player_flag'] = 0

        return features

    def _extract_minutes(self, game_log: dict) -> float | None:
        """Extract minutes from a game log dict, handling various formats."""
        # Try common field names
        for field in ['min', 'mins', 'minutes', 'mp', 'time_played']:
            val = game_log.get(field)
            if val is not None:
                return self._parse_minutes(val)

        return None

    def _parse_minutes(self, value: Any) -> float | None:
        """Parse minutes from various formats (int, float, string 'MM:SS')."""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            value = value.strip()
            if not value or value == '--':
                return None

            # Handle MM:SS format
            if ':' in value:
                try:
                    parts = value.split(':')
                    mins = int(parts[0])
                    secs = int(parts[1]) if len(parts) > 1 else 0
                    return mins + secs / 60.0
                except (ValueError, IndexError):
                    return None

            # Try direct conversion
            try:
                return float(value)
            except ValueError:
                return None

        return None

    def generate_features_batch(self,
                                 players: list[dict],
                                 game_context: dict,
                                 game_date: str) -> list[dict]:
        """
        Generate features for multiple players efficiently.

        Args:
            players: List of player dicts with 'player_id', 'team_id', 'game_logs', etc.
            game_context: Shared game context
            game_date: Game date

        Returns:
            List of feature dictionaries
        """
        results = []
        for player in players:
            features = self.generate_features(
                player_id=player.get('player_id'),
                team_id=player.get('team_id'),
                opponent_team_id=game_context.get('opponent_team_id'),
                game_date=game_date,
                game_context=game_context,
                player_game_logs=player.get('game_logs', []),
                team_roster=player.get('team_roster', []),
                injured_players=player.get('injured_players', [])
            )
            features['player_id'] = player.get('player_id')
            results.append(features)

        return results


# Feature list for model training (in order)
MINUTES_FEATURE_NAMES = [
    # Baseline features
    'season_min_avg',
    'recent_min_avg',
    'recent_min_std',
    'min_trend',
    'games_played',
    'last3_min_avg',
    'min_consistency',
    'min_floor',
    'min_ceiling',

    # Context features
    'vegas_spread_abs',
    'vegas_spread',
    'vegas_total',
    'is_home',
    'is_back_to_back',
    'days_rest',
    'is_favorite',
    'blowout_risk',
    'expected_pace_factor',
    'rest_advantage',

    # Rotation features
    'teammates_injured_minutes',
    'position_depth_score',
    'is_starter',
    'rotation_spot',
    'injury_boost_factor',

    # Coach features
    'coach_starter_min_avg',
    'coach_bench_min_avg',
    'coach_blowout_pull_lead',
    'coach_blowout_pull_deficit',
    'coach_variance_score',
    'coach_blowout_factor',
    'coach_b2b_factor',
    'coach_expected_mins',

    # Situational features
    'projected_game_competitiveness',
    'player_foul_rate',
    'clutch_player_flag',
    'overtime_likelihood',
    'blowout_minutes_penalty',
]


def features_to_array(features: dict[str, float],
                      feature_names: list[str] = MINUTES_FEATURE_NAMES) -> np.ndarray:
    """Convert feature dict to numpy array in correct order."""
    return np.array([features.get(name, 0.0) for name in feature_names])


def features_to_dataframe(features_list: list[dict[str, float]],
                          feature_names: list[str] = MINUTES_FEATURE_NAMES):
    """Convert list of feature dicts to pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame(features_list)[feature_names]
