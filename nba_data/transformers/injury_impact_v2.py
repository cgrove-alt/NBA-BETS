"""
Injury Impact Analysis V2 - Star Player & Usage Impact

This module calculates the impact of injuries on:
1. Team performance (spread/moneyline)
2. Player props (usage redistribution)

Key Features:
- star_player_out: Binary flag for star player absence
- usage_lost: Percentage of team's usage that's missing
- rebounds_available: Extra rebounding opportunity when big out
- assists_opportunity: Playmaking opportunity when PG out

Usage:
    from injury_impact_v2 import InjuryImpactCalculator
    calc = InjuryImpactCalculator()
    calc.update_player_usage(player_id, stats)
    impact = calc.calculate_injury_impact(team_id, injuries, game_date)
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict


class PlayerUsageTracker:
    """
    Track player usage and role metrics over time.

    Usage Rate = (FGA + 0.44*FTA + TOV) / (Minutes * Team Possessions)

    Higher usage = more important to team's offense.
    """

    STAR_PLAYER_THRESHOLDS = {
        'pts_avg': 20.0,      # 20+ PPG = star
        'usage_rate': 0.25,   # 25%+ usage = high volume
        'min_avg': 32.0,      # 32+ MPG = starter
    }

    POSITION_GROUPS = {
        'G': ['PG', 'SG', 'G', 'G-F', 'PG-SG'],
        'F': ['F', 'SF', 'PF', 'F-G', 'F-C', 'SF-PF'],
        'C': ['C', 'C-F', 'C-PF', 'PF-C'],
    }

    def __init__(self):
        # Structure: player_id -> [(game_date, stats_dict), ...]
        self.player_games = defaultdict(list)
        # Structure: player_id -> info dict
        self.player_info = {}
        # Structure: team_id -> set of player_ids
        self.team_rosters = defaultdict(set)

    def add_player_game(self, player_id: int, game_date: str, stats: dict,
                        player_info: dict = None, team_id: int = None):
        """
        Add a game's stats for a player.

        Args:
            player_id: Player identifier
            game_date: Date in YYYY-MM-DD format
            stats: Player stats (pts, reb, ast, fga, fta, min, etc.)
            player_info: Optional player info (name, position)
            team_id: Team identifier
        """
        # Parse minutes
        min_val = stats.get('min', 0)
        if isinstance(min_val, str) and ':' in min_val:
            parts = min_val.split(':')
            minutes = float(parts[0]) + float(parts[1]) / 60
        else:
            minutes = float(min_val) if min_val else 0

        # Calculate usage rate (simplified)
        fga = stats.get('fga', 0) or 0
        fta = stats.get('fta', 0) or 0
        tov = stats.get('tov', 0) or stats.get('turnover', 0) or 0
        pts = stats.get('pts', 0) or 0
        reb = stats.get('reb', 0) or 0
        ast = stats.get('ast', 0) or 0
        fg3m = stats.get('fg3m', 0) or 0

        # Usage per minute (normalized)
        plays = fga + 0.44 * fta + tov
        usage_per_min = plays / max(minutes, 1)

        game_data = {
            'date': game_date,
            'pts': pts,
            'reb': reb,
            'ast': ast,
            'fg3m': fg3m,
            'min': minutes,
            'fga': fga,
            'fta': fta,
            'tov': tov,
            'usage_per_min': usage_per_min,
            'pra': pts + reb + ast,
        }

        self.player_games[player_id].append((game_date, game_data))
        self.player_games[player_id].sort(key=lambda x: x[0])

        # Update player info
        if player_info:
            self.player_info[player_id] = {
                'name': player_info.get('name', f"Player {player_id}"),
                'first_name': player_info.get('first_name', ''),
                'last_name': player_info.get('last_name', ''),
                'position': player_info.get('position', 'F'),
            }

        # Update roster
        if team_id:
            self.team_rosters[team_id].add(player_id)

    def get_player_metrics_before_date(self, player_id: int, game_date: str,
                                        window: int = 10) -> dict | None:
        """
        Get player metrics using only games BEFORE game_date.

        Args:
            player_id: Player identifier
            game_date: Target date (exclusive)
            window: Number of recent games

        Returns:
            Dictionary with player metrics or None
        """
        if player_id not in self.player_games:
            return None

        games = [(d, s) for d, s in self.player_games[player_id] if d < game_date]

        if len(games) < 3:
            return None

        recent = games[-window:]

        pts = [g[1]['pts'] for g in recent]
        reb = [g[1]['reb'] for g in recent]
        ast = [g[1]['ast'] for g in recent]
        mins = [g[1]['min'] for g in recent]
        usage = [g[1]['usage_per_min'] for g in recent]

        metrics = {
            'pts_avg': np.mean(pts),
            'reb_avg': np.mean(reb),
            'ast_avg': np.mean(ast),
            'min_avg': np.mean(mins),
            'usage_rate': np.mean(usage) * 48 * 0.2,  # Approximate team usage
            'games_played': len(games),
        }

        # Classify player role
        metrics['is_star'] = (
            metrics['pts_avg'] >= self.STAR_PLAYER_THRESHOLDS['pts_avg'] or
            metrics['usage_rate'] >= self.STAR_PLAYER_THRESHOLDS['usage_rate']
        )
        metrics['is_starter'] = metrics['min_avg'] >= self.STAR_PLAYER_THRESHOLDS['min_avg']

        # Position
        info = self.player_info.get(player_id, {})
        metrics['position'] = info.get('position', 'F')
        metrics['name'] = info.get('name', f"Player {player_id}")

        return metrics

    def get_team_usage_distribution(self, team_id: int, game_date: str) -> dict:
        """
        Get usage distribution for a team.

        Returns dict mapping player_id -> usage share.
        """
        if team_id not in self.team_rosters:
            return {}

        usage_dist = {}
        total_usage = 0

        for player_id in self.team_rosters[team_id]:
            metrics = self.get_player_metrics_before_date(player_id, game_date)
            if metrics:
                usage = metrics['usage_rate'] * metrics['min_avg']
                usage_dist[player_id] = usage
                total_usage += usage

        # Normalize to percentages
        if total_usage > 0:
            usage_dist = {pid: u / total_usage for pid, u in usage_dist.items()}

        return usage_dist


class InjuryImpactCalculator:
    """
    Calculate the impact of injuries on team and player performance.

    Key metrics:
    - Team-level: Expected point differential impact
    - Player-level: Usage redistribution opportunity
    """

    # Position importance weights for team impact
    POSITION_WEIGHTS = {
        'G': {'scoring': 0.35, 'playmaking': 0.40, 'rebounding': 0.10, 'defense': 0.15},
        'F': {'scoring': 0.30, 'playmaking': 0.20, 'rebounding': 0.25, 'defense': 0.25},
        'C': {'scoring': 0.20, 'playmaking': 0.10, 'rebounding': 0.40, 'defense': 0.30},
    }

    # Impact multipliers by star level
    STAR_IMPACT = {
        'superstar': 6.0,   # Top 10 player: ~6 point swing
        'star': 4.0,        # All-star: ~4 point swing
        'starter': 2.0,     # Quality starter: ~2 point swing
        'rotation': 0.5,    # Rotation player: minimal impact
        'bench': 0.0,       # Deep bench: no impact
    }

    def __init__(self, usage_tracker: PlayerUsageTracker = None):
        self.usage_tracker = usage_tracker or PlayerUsageTracker()

    def classify_player_tier(self, metrics: dict) -> str:
        """
        Classify player into tier based on metrics.

        Returns: 'superstar', 'star', 'starter', 'rotation', or 'bench'
        """
        pts = metrics.get('pts_avg', 0)
        mins = metrics.get('min_avg', 0)
        usage = metrics.get('usage_rate', 0)

        if pts >= 25 or (pts >= 22 and usage >= 0.28):
            return 'superstar'
        if pts >= 18 or (pts >= 15 and usage >= 0.22):
            return 'star'
        if mins >= 28 and pts >= 10:
            return 'starter'
        if mins >= 15:
            return 'rotation'
        return 'bench'

    def calculate_injury_impact(
        self,
        team_id: int,
        injured_player_ids: list[int],
        game_date: str
    ) -> dict:
        """
        Calculate the combined impact of injuries on a team.

        Args:
            team_id: Team identifier
            injured_player_ids: List of injured player IDs
            game_date: Game date for temporal safety

        Returns:
            Dictionary with injury impact features
        """
        features = {
            'num_players_out': len(injured_player_ids),
            'star_player_out': 0,
            'starter_out': 0,
            'usage_lost': 0.0,
            'pts_lost': 0.0,
            'reb_lost': 0.0,
            'ast_lost': 0.0,
            'expected_point_impact': 0.0,
            'guard_out': 0,
            'forward_out': 0,
            'center_out': 0,
        }

        if not injured_player_ids:
            return features

        # Get team usage distribution
        usage_dist = self.usage_tracker.get_team_usage_distribution(team_id, game_date)

        for player_id in injured_player_ids:
            metrics = self.usage_tracker.get_player_metrics_before_date(
                player_id, game_date
            )

            if not metrics:
                continue

            # Classify tier
            tier = self.classify_player_tier(metrics)

            # Update counts
            if tier in ['superstar', 'star']:
                features['star_player_out'] = 1
            if tier in ['superstar', 'star', 'starter']:
                features['starter_out'] += 1

            # Usage lost
            features['usage_lost'] += usage_dist.get(player_id, 0.15)

            # Stats lost
            features['pts_lost'] += metrics['pts_avg']
            features['reb_lost'] += metrics['reb_avg']
            features['ast_lost'] += metrics['ast_avg']

            # Point impact
            features['expected_point_impact'] += self.STAR_IMPACT.get(tier, 0)

            # Position tracking
            pos = metrics.get('position', 'F')
            if pos in ['PG', 'SG', 'G', 'G-F']:
                features['guard_out'] += 1
            elif pos in ['C', 'C-F']:
                features['center_out'] += 1
            else:
                features['forward_out'] += 1

        return features

    def calculate_player_opportunity(
        self,
        player_id: int,
        team_id: int,
        injured_player_ids: list[int],
        game_date: str
    ) -> dict:
        """
        Calculate the opportunity boost for a player given team injuries.

        When teammates are out, remaining players get:
        - More minutes
        - Higher usage
        - More shot attempts

        Args:
            player_id: Target player ID
            team_id: Team ID
            injured_player_ids: List of injured teammates
            game_date: Game date

        Returns:
            Dictionary with opportunity boost features
        """
        features = {
            'usage_boost': 0.0,
            'minutes_boost': 0.0,
            'rebounds_opportunity': 0.0,
            'assists_opportunity': 0.0,
            'shots_opportunity': 0.0,
            'primary_option_boost': 0,
        }

        player_metrics = self.usage_tracker.get_player_metrics_before_date(
            player_id, game_date
        )

        if not player_metrics:
            return features

        # Calculate what's being lost
        injury_impact = self.calculate_injury_impact(team_id, injured_player_ids, game_date)

        usage_lost = injury_impact['usage_lost']
        pts_lost = injury_impact['pts_lost']
        reb_lost = injury_impact['reb_lost']
        ast_lost = injury_impact['ast_lost']

        # Player's share of the redistribution
        # Higher usage players absorb more
        player_metrics.get('usage_rate', 0.15)

        # Redistribution assumes top players get ~40% of lost usage
        # Remaining is spread among others
        player_tier = self.classify_player_tier(player_metrics)

        if player_tier in ['superstar', 'star']:
            redistribution_share = 0.40
        elif player_tier == 'starter':
            redistribution_share = 0.25
        elif player_tier == 'rotation':
            redistribution_share = 0.15
        else:
            redistribution_share = 0.05

        # Apply redistribution
        features['usage_boost'] = round(usage_lost * redistribution_share, 3)
        features['minutes_boost'] = round(usage_lost * redistribution_share * 10, 1)  # ~10 min per 100% usage

        # Stat-specific opportunities
        # Points: Scale by player's scoring ability
        if player_metrics['pts_avg'] >= 15:
            features['shots_opportunity'] = round(pts_lost * redistribution_share / 2, 1)

        # Rebounds: More opportunity if center/forward out
        if injury_impact['center_out'] > 0 or injury_impact['forward_out'] > 0:
            features['rebounds_opportunity'] = round(reb_lost * redistribution_share, 1)

        # Assists: More opportunity if guard out
        if injury_impact['guard_out'] > 0:
            features['assists_opportunity'] = round(ast_lost * redistribution_share, 1)

        # Primary option boost (when top scorer out)
        if injury_impact['star_player_out'] and player_tier in ['star', 'starter']:
            features['primary_option_boost'] = 1

        return features


class TeamInjuryManager:
    """
    High-level manager for tracking and calculating injury impacts.

    Provides easy interface for model integration.
    """

    def __init__(self):
        self.usage_tracker = PlayerUsageTracker()
        self.impact_calc = InjuryImpactCalculator(self.usage_tracker)
        # Track current injuries: team_id -> set of player_ids
        self.current_injuries = defaultdict(set)

    def add_player_game(self, player_id: int, game_date: str, stats: dict,
                        player_info: dict = None, team_id: int = None):
        """Add a game for a player (passthrough to usage tracker)."""
        self.usage_tracker.add_player_game(
            player_id, game_date, stats, player_info, team_id
        )

    def set_injuries(self, team_id: int, injured_player_ids: list[int]):
        """Set current injuries for a team."""
        self.current_injuries[team_id] = set(injured_player_ids)

    def get_team_injury_features(self, team_id: int, game_date: str) -> dict:
        """
        Get injury-related features for a team.

        Returns features like:
        - star_player_out
        - usage_lost
        - expected_point_impact
        """
        injured = list(self.current_injuries.get(team_id, []))
        return self.impact_calc.calculate_injury_impact(team_id, injured, game_date)

    def get_player_injury_boost(self, player_id: int, team_id: int, game_date: str) -> dict:
        """
        Get opportunity boost for a player given team injuries.

        Returns features like:
        - usage_boost
        - rebounds_opportunity
        - primary_option_boost
        """
        injured = list(self.current_injuries.get(team_id, []))
        return self.impact_calc.calculate_player_opportunity(
            player_id, team_id, injured, game_date
        )


def generate_injury_features(
    manager: TeamInjuryManager,
    home_id: int,
    away_id: int,
    game_date: str
) -> dict:
    """
    Generate all injury-related features for a game.

    Args:
        manager: TeamInjuryManager instance
        home_id: Home team ID
        away_id: Away team ID
        game_date: Game date

    Returns:
        Dictionary with injury features for both teams
    """
    features = {}

    # Home team injuries
    home_impact = manager.get_team_injury_features(home_id, game_date)
    for key, val in home_impact.items():
        features[f'home_{key}'] = val

    # Away team injuries
    away_impact = manager.get_team_injury_features(away_id, game_date)
    for key, val in away_impact.items():
        features[f'away_{key}'] = val

    # Differential features
    features['injury_advantage'] = (
        away_impact['expected_point_impact'] - home_impact['expected_point_impact']
    )
    features['star_out_diff'] = (
        away_impact['star_player_out'] - home_impact['star_player_out']
    )

    return features


def generate_player_prop_features(
    manager: TeamInjuryManager,
    player_id: int,
    team_id: int,
    opponent_id: int,
    game_date: str
) -> dict:
    """
    Generate injury-related features for player prop prediction.

    Args:
        manager: TeamInjuryManager instance
        player_id: Target player ID
        team_id: Player's team ID
        opponent_id: Opponent team ID
        game_date: Game date

    Returns:
        Dictionary with player-specific injury features
    """
    features = {}

    # Teammate injuries (opportunity)
    boost = manager.get_player_injury_boost(player_id, team_id, game_date)
    for key, val in boost.items():
        features[f'teammate_{key}'] = val

    # Opponent injuries (matchup advantage)
    opp_impact = manager.get_team_injury_features(opponent_id, game_date)
    features['opp_star_out'] = opp_impact['star_player_out']
    features['opp_center_out'] = opp_impact['center_out']
    features['opp_guard_out'] = opp_impact['guard_out']
    features['opp_def_weakened'] = opp_impact['expected_point_impact']

    return features
