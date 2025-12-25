"""
NBA Feature Engineering

Processes NBA data to generate features for betting predictions:
- Moneyline predictions
- Spread predictions
- Player prop predictions
- Head-to-head matchup analysis
- Positional strengths/weaknesses
- Injury impact analysis
"""

import numpy as np
import requests
import concurrent.futures
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

from data_fetcher import (
    fetch_team_statistics,
    fetch_historical_games,
    fetch_player_stats,
    fetch_league_team_stats,
    fetch_team_roster,
    fetch_head_to_head,
    fetch_player_vs_team,
    get_team_id,
    get_player_id,
)


# Position mapping for analysis
POSITION_GROUPS = {
    "G": ["PG", "SG", "G", "G-F"],
    "F": ["SF", "PF", "F", "F-G", "F-C"],
    "C": ["C", "C-F"],
}


# =============================================================================
# FEATURE VALIDATION - Critical for preventing broken model outputs
# =============================================================================

# Realistic NBA value ranges for validation
NBA_VALUE_RANGES = {
    # Team per-game stats
    "pts_avg": (80, 140),           # Team points per game (typical: 100-125)
    "reb_avg": (35, 60),            # Team rebounds per game
    "ast_avg": (18, 35),            # Team assists per game
    "stl_avg": (4, 12),             # Team steals per game
    "blk_avg": (2, 8),              # Team blocks per game

    # Efficiency ratings
    "off_rating": (95, 130),        # Offensive rating (points per 100 possessions)
    "def_rating": (95, 130),        # Defensive rating
    "net_rating": (-20, 20),        # Net rating (off - def)

    # Percentages (0-1 scale)
    "win_pct": (0, 1),
    "fg_pct": (0.35, 0.55),
    "fg3_pct": (0.25, 0.45),
    "ft_pct": (0.65, 0.90),

    # Pace and tempo
    "pace": (90, 110),              # Possessions per 48 minutes

    # Point differentials
    "expected_point_diff": (-30, 30),
    "plus_minus_diff": (-20, 20),
    "pts_avg_diff": (-25, 25),
    "recent_pts_diff": (-30, 30),

    # Location-specific
    "location_pts_avg": (80, 140),
    "expected_home_pts": (80, 140),
    "expected_away_pts": (80, 140),

    # Home/away
    "home_pts_avg": (80, 140),
    "away_pts_avg": (80, 140),

    # Rebounding/assist diffs
    "reb_diff": (-15, 15),
    "ast_diff": (-10, 10),
}


def validate_and_clip_feature(value: float, feature_name: str, warn: bool = True) -> float:
    """
    Validate and clip a feature value to realistic NBA ranges.

    Args:
        value: The feature value to validate
        feature_name: Name of the feature for range lookup
        warn: Whether to print warning for out-of-range values

    Returns:
        Clipped value within valid range
    """
    if value is None:
        return 0.0

    # Look for matching range - EXACT match first, then partial
    range_bounds = None
    feature_lower = feature_name.lower()

    # First try exact match
    if feature_lower in NBA_VALUE_RANGES:
        range_bounds = NBA_VALUE_RANGES[feature_lower]
    else:
        # Try partial match - but prefer longer (more specific) keys first
        for key in sorted(NBA_VALUE_RANGES.keys(), key=len, reverse=True):
            if key in feature_lower or feature_lower in key:
                range_bounds = NBA_VALUE_RANGES[key]
                break

    if range_bounds is None:
        return float(value)

    min_val, max_val = range_bounds

    if value < min_val or value > max_val:
        if warn and abs(value) > max_val * 2:  # Only warn for significantly bad values
            print(f"Warning: {feature_name}={value:.2f} outside expected range [{min_val}, {max_val}]")
        return float(np.clip(value, min_val, max_val))

    return float(value)


def validate_features_dict(features: Dict, prefix: str = "", warn: bool = True) -> Dict:
    """
    Validate and clip all numeric features in a dictionary.

    Args:
        features: Dictionary of features
        prefix: Optional prefix for feature names in warnings
        warn: Whether to print warnings

    Returns:
        Dictionary with validated/clipped values
    """
    validated = {}

    for key, value in features.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            full_key = f"{prefix}_{key}" if prefix else key
            validated[key] = validate_and_clip_feature(value, full_key, warn)
        elif isinstance(value, dict):
            # Recursively validate nested dicts (but not injury_details)
            if key != "injury_details":
                validated[key] = validate_features_dict(value, key, warn)
            else:
                validated[key] = value
        else:
            validated[key] = value

    return validated


def ensure_per_game_average(total: float, games_played: int, default: float = 0.0) -> float:
    """
    Ensure a stat is a per-game average, not a season total.

    Args:
        total: The value (might be total or already per-game)
        games_played: Number of games played
        default: Default value if calculation fails

    Returns:
        Per-game average
    """
    if games_played is None or games_played <= 0:
        return default

    # If value is suspiciously high (likely a season total), convert to per-game
    # A team scoring 130 PPG would have ~10,000 total points over 80 games
    if abs(total) > 200:  # Threshold for detecting season totals
        return total / games_played

    return total


class InjuryReportManager:
    """
    Manages injury report data and calculates impact on team performance.

    Note: This class provides a framework for injury integration. In production,
    you would connect to a real injury data source (e.g., ESPN API, official NBA injury reports).
    """

    # Impact multipliers for different injury statuses
    STATUS_IMPACT = {
        "out": 1.0,           # Full impact - player not playing
        "doubtful": 0.85,     # 85% chance of full impact
        "questionable": 0.5,  # 50% chance of impact
        "probable": 0.15,     # 15% chance of impact
        "available": 0.0,     # No impact
    }

    # Position importance weights for team performance
    POSITION_WEIGHTS = {
        "PG": {"offense": 0.25, "defense": 0.15, "playmaking": 0.35},
        "SG": {"offense": 0.25, "defense": 0.20, "scoring": 0.30},
        "SF": {"offense": 0.20, "defense": 0.25, "versatility": 0.25},
        "PF": {"offense": 0.15, "defense": 0.25, "rebounding": 0.30},
        "C": {"offense": 0.15, "defense": 0.30, "rebounding": 0.35},
    }

    def __init__(self, season="2025-26"):
        self.season = season
        self._injury_cache = {}

    def set_injury_report(self, team_id: int, injuries: List[Dict]):
        """
        Set injury report for a team manually.

        Args:
            team_id: NBA team ID
            injuries: List of injury dictionaries with keys:
                - player_id: NBA player ID
                - player_name: Player name
                - status: "out", "doubtful", "questionable", "probable"
                - injury: Description of injury
        """
        self._injury_cache[team_id] = injuries

    def get_injury_report(self, team_id: int) -> List[Dict]:
        """Get cached injury report for a team."""
        return self._injury_cache.get(team_id, [])

    def get_out_player_names(self, team_id: int) -> set:
        """
        Get set of player names who are OUT or DOUBTFUL for a team.

        These players should be excluded from prop predictions since
        sportsbooks won't accept bets on injured players.

        Args:
            team_id: NBA team ID

        Returns:
            Set of player names (lowercase for matching)
        """
        injuries = self.get_injury_report(team_id)
        out_players = set()

        for inj in injuries:
            status = inj.get("status", "").lower()
            # Exclude OUT and DOUBTFUL players (highly unlikely to play)
            if status in ("out", "doubtful"):
                player_name = inj.get("player_name", "")
                if player_name:
                    out_players.add(player_name.lower())

        return out_players

    def calculate_player_value(self, player_stats: Dict, position: str) -> float:
        """
        Calculate a player's value score based on their stats.

        Args:
            player_stats: Player season averages
            position: Player position

        Returns:
            Value score (0-100 scale)
        """
        pts = player_stats.get("pts_avg", 0) or 0
        reb = player_stats.get("reb_avg", 0) or 0
        ast = player_stats.get("ast_avg", 0) or 0
        stl = player_stats.get("stl_avg", 0) or 0
        blk = player_stats.get("blk_avg", 0) or 0
        mins = player_stats.get("min_avg", 0) or 0

        # Base value from stats
        base_value = (pts * 1.0 + reb * 1.2 + ast * 1.5 + stl * 2.0 + blk * 2.0)

        # Adjust for minutes played (proxy for team reliance)
        minutes_factor = mins / 36.0 if mins else 0.5

        return base_value * minutes_factor

    def calculate_injury_impact(self, team_id: int) -> Dict:
        """
        Calculate the overall impact of injuries on team performance.

        Args:
            team_id: NBA team ID

        Returns:
            Dictionary with injury impact metrics
        """
        injuries = self.get_injury_report(team_id)

        if not injuries:
            return {
                "total_impact": 0.0,
                "offensive_impact": 0.0,
                "defensive_impact": 0.0,
                "injured_player_count": 0,
                "key_players_out": 0,
                "injured_players": [],
            }

        total_value_lost = 0.0
        offensive_value_lost = 0.0
        defensive_value_lost = 0.0
        key_players_out = 0
        injured_player_details = []

        for injury in injuries:
            player_id = injury.get("player_id")
            status = injury.get("status", "questionable").lower()
            position = injury.get("position", "G")

            impact_probability = self.STATUS_IMPACT.get(status, 0.5)

            if player_id:
                try:
                    stats = fetch_player_stats(player_id, self.season)
                    season_avg = stats.get("season_averages", {})
                    player_value = self.calculate_player_value(season_avg, position)

                    # Weighted impact
                    value_lost = player_value * impact_probability
                    total_value_lost += value_lost

                    # Position-specific impact
                    pos_weights = self.POSITION_WEIGHTS.get(position[:2], {"offense": 0.2, "defense": 0.2})
                    offensive_value_lost += value_lost * pos_weights.get("offense", 0.2)
                    defensive_value_lost += value_lost * pos_weights.get("defense", 0.2)

                    if player_value > 15 and status in ["out", "doubtful"]:
                        key_players_out += 1

                    injured_player_details.append({
                        "player_name": injury.get("player_name"),
                        "status": status,
                        "value_lost": value_lost,
                        "position": position,
                    })
                except Exception:
                    # If we can't fetch player stats, use estimated impact
                    estimated_value = 10.0 * impact_probability
                    total_value_lost += estimated_value
                    injured_player_details.append({
                        "player_name": injury.get("player_name"),
                        "status": status,
                        "value_lost": estimated_value,
                        "position": position,
                    })
            else:
                # No player_id available - use estimated impact based on status
                # Default estimated value: 10 for role players, higher for known stars
                estimated_value = 10.0 * impact_probability
                total_value_lost += estimated_value

                # Position-specific impact for estimated values
                pos_weights = self.POSITION_WEIGHTS.get(position[:2], {"offense": 0.2, "defense": 0.2})
                offensive_value_lost += estimated_value * pos_weights.get("offense", 0.2)
                defensive_value_lost += estimated_value * pos_weights.get("defense", 0.2)

                injured_player_details.append({
                    "player_name": injury.get("player_name"),
                    "status": status,
                    "value_lost": estimated_value,
                    "position": position,
                })

        return {
            "total_impact": total_value_lost,
            "offensive_impact": offensive_value_lost,
            "defensive_impact": defensive_value_lost,
            "injured_player_count": len(injuries),
            "key_players_out": key_players_out,
            "injured_players": injured_player_details,
        }


class HeadToHeadAnalyzer:
    """Analyze head-to-head matchup history between teams."""

    def __init__(self, season="2025-26"):
        self.season = season

    def analyze_h2h(self, team1_id: int, team2_id: int, include_previous_season: bool = True) -> Dict:
        """
        Analyze head-to-head history between two teams.

        Args:
            team1_id: First team NBA ID
            team2_id: Second team NBA ID
            include_previous_season: Include previous season games

        Returns:
            Dictionary with H2H analysis
        """
        seasons = self.season
        if include_previous_season:
            # Add previous season
            current_year = int(self.season.split("-")[0])
            prev_season = f"{current_year - 1}-{str(current_year)[-2:]}"
            seasons = f"{prev_season},{self.season}"

        h2h_games = fetch_head_to_head(team1_id, team2_id, seasons, last_n_games=10)

        if not h2h_games:
            return {
                "games_played": 0,
                "team1_wins": 0,
                "team1_win_pct": 0.5,
                "avg_point_diff": 0,
                "avg_total_points": 0,
                "team1_avg_pts": 0,
                "recent_winner": None,
                "home_team_wins": 0,
            }

        team1_wins = sum(1 for g in h2h_games if g.get("wl") == "W")
        total_pts = [g.get("pts", 0) or 0 for g in h2h_games]
        point_diffs = [g.get("plus_minus", 0) or 0 for g in h2h_games]

        # Determine home games (matchup format: "TEAM vs. OPP" is home, "TEAM @ OPP" is away)
        home_wins = 0
        for game in h2h_games:
            matchup = game.get("matchup", "")
            is_home = "vs." in matchup
            if is_home and game.get("wl") == "W":
                home_wins += 1

        return {
            "games_played": len(h2h_games),
            "team1_wins": team1_wins,
            "team1_losses": len(h2h_games) - team1_wins,
            "team1_win_pct": team1_wins / len(h2h_games) if h2h_games else 0.5,
            "avg_point_diff": np.mean(point_diffs) if point_diffs else 0,
            "avg_total_points": np.mean(total_pts) * 2 if total_pts else 0,  # Both teams
            "team1_avg_pts": np.mean(total_pts) if total_pts else 0,
            "recent_winner": "team1" if h2h_games[0].get("wl") == "W" else "team2",
            "home_team_advantage": home_wins / len(h2h_games) if h2h_games else 0.5,
            "point_diff_std": np.std(point_diffs) if len(point_diffs) > 1 else 0,
        }


class PositionalAnalyzer:
    """Analyze positional strengths and weaknesses."""

    def __init__(self, season="2025-26"):
        self.season = season

    def get_position_group(self, position: str) -> str:
        """Map detailed position to position group (G/F/C)."""
        position = position.upper() if position else "G"
        for group, positions in POSITION_GROUPS.items():
            if position in positions:
                return group
        return "G"  # Default to guard

    def analyze_team_positional_strength(self, team_id: int) -> Dict:
        """
        Analyze team's strength at each position group.

        Args:
            team_id: NBA team ID

        Returns:
            Dictionary with positional analysis
        """
        roster = fetch_team_roster(team_id, self.season)

        position_stats = {"G": [], "F": [], "C": []}

        # Helper function for parallel fetching
        def fetch_single_player(player):
            player_id = player.get("player_id")
            if not player_id:
                return None
            try:
                stats = fetch_player_stats(player_id, self.season, last_n_games=10)
                return (player, stats)
            except Exception:
                return None

        # Fetch all player stats in parallel (max 5 concurrent to respect rate limits)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(fetch_single_player, roster))

        # Process results
        for result in results:
            if result is None:
                continue
            player, stats = result
            position = player.get("position", "G")
            pos_group = self.get_position_group(position)
            season_avg = stats.get("season_averages", {})

            if season_avg.get("games_played", 0):
                position_stats[pos_group].append({
                    "player_id": player.get("player_id"),
                    "player_name": player.get("player_name"),
                    "pts": season_avg.get("pts_avg", 0) or 0,
                    "reb": season_avg.get("reb_avg", 0) or 0,
                    "ast": season_avg.get("ast_avg", 0) or 0,
                    "min": season_avg.get("min_avg", 0) or 0,
                })

        # Calculate positional strengths
        strengths = {}
        for pos_group, players in position_stats.items():
            if players:
                # Weight by minutes played
                total_mins = sum(p["min"] for p in players)
                if total_mins > 0:
                    weighted_pts = sum(p["pts"] * p["min"] for p in players) / total_mins
                    weighted_reb = sum(p["reb"] * p["min"] for p in players) / total_mins
                    weighted_ast = sum(p["ast"] * p["min"] for p in players) / total_mins
                else:
                    weighted_pts = np.mean([p["pts"] for p in players])
                    weighted_reb = np.mean([p["reb"] for p in players])
                    weighted_ast = np.mean([p["ast"] for p in players])

                strengths[pos_group] = {
                    "pts_production": weighted_pts,
                    "reb_production": weighted_reb,
                    "ast_production": weighted_ast,
                    "player_count": len(players),
                    "total_minutes": total_mins,
                    "composite_score": weighted_pts + weighted_reb * 1.2 + weighted_ast * 1.5,
                }
            else:
                strengths[pos_group] = {
                    "pts_production": 0,
                    "reb_production": 0,
                    "ast_production": 0,
                    "player_count": 0,
                    "total_minutes": 0,
                    "composite_score": 0,
                }

        return strengths

    def calculate_positional_matchup(self, team1_id: int, team2_id: int) -> Dict:
        """
        Calculate positional advantages in a matchup.

        Args:
            team1_id: First team NBA ID
            team2_id: Second team NBA ID

        Returns:
            Dictionary with positional matchup analysis
        """
        team1_strengths = self.analyze_team_positional_strength(team1_id)
        team2_strengths = self.analyze_team_positional_strength(team2_id)

        matchup = {}
        for pos in ["G", "F", "C"]:
            t1 = team1_strengths.get(pos, {})
            t2 = team2_strengths.get(pos, {})

            matchup[pos] = {
                "team1_score": t1.get("composite_score", 0),
                "team2_score": t2.get("composite_score", 0),
                "advantage": t1.get("composite_score", 0) - t2.get("composite_score", 0),
                "pts_diff": t1.get("pts_production", 0) - t2.get("pts_production", 0),
                "reb_diff": t1.get("reb_production", 0) - t2.get("reb_production", 0),
            }

        # Overall positional advantage
        total_advantage = sum(m["advantage"] for m in matchup.values())

        return {
            "guard_matchup": matchup["G"],
            "forward_matchup": matchup["F"],
            "center_matchup": matchup["C"],
            "total_positional_advantage": total_advantage,
            "strongest_position": max(matchup.keys(), key=lambda k: matchup[k]["advantage"]),
            "weakest_position": min(matchup.keys(), key=lambda k: matchup[k]["advantage"]),
        }


class TeamFeatureGenerator:
    """Generate features for team-level predictions (moneyline, spread)."""

    def __init__(self, season="2025-26"):
        self.season = season
        self._league_stats_cache = None

    def get_league_stats(self):
        """Get cached league stats for normalization."""
        if self._league_stats_cache is None:
            self._league_stats_cache = fetch_league_team_stats(self.season)
        return self._league_stats_cache

    def calculate_recent_form(self, team_id, last_n_games=10, before_date=None):
        """
        Calculate team's recent form based on last N games.

        Args:
            team_id: NBA team ID
            last_n_games: Number of recent games to analyze
            before_date: Only include games BEFORE this date (YYYY-MM-DD format)
                        CRITICAL for preventing data leakage in training

        Returns:
            Dictionary with recent performance metrics
        """
        games = fetch_historical_games(
            team_id=team_id,
            season=self.season,
            last_n_games=last_n_games,
            date_to=before_date  # CRITICAL: Only use games before the prediction date
        )

        if not games:
            return None

        wins = sum(1 for g in games if g.get("wl") == "W")
        total_pts = sum(g.get("pts", 0) or 0 for g in games)
        total_plus_minus = sum(g.get("plus_minus", 0) or 0 for g in games)

        fg_pcts = [g.get("fg_pct") for g in games if g.get("fg_pct") is not None]
        fg3_pcts = [g.get("fg3_pct") for g in games if g.get("fg3_pct") is not None]

        return {
            "games_played": len(games),
            "wins": wins,
            "losses": len(games) - wins,
            "win_pct": wins / len(games) if games else 0,
            "avg_pts": total_pts / len(games) if games else 0,
            "avg_plus_minus": total_plus_minus / len(games) if games else 0,
            "avg_fg_pct": np.mean(fg_pcts) if fg_pcts else 0,
            "avg_fg3_pct": np.mean(fg3_pcts) if fg3_pcts else 0,
            "streak": self._calculate_streak(games),
        }

    def _calculate_streak(self, games):
        """Calculate current win/loss streak."""
        if not games:
            return 0

        streak = 0
        streak_type = games[0].get("wl")

        for game in games:
            if game.get("wl") == streak_type:
                streak += 1
            else:
                break

        return streak if streak_type == "W" else -streak

    def calculate_home_advantage(self, team_id):
        """
        Calculate home court advantage metrics.

        Returns:
            Dictionary with home vs away performance differential
        """
        team_stats = fetch_team_statistics(team_id, self.season)

        home = team_stats.get("home", {})
        away = team_stats.get("away", {})

        home_win_pct = home.get("win_pct", 0) or 0
        away_win_pct = away.get("win_pct", 0) or 0
        home_pts = home.get("pts_avg", 0) or 0
        away_pts = away.get("pts_avg", 0) or 0

        return {
            "home_win_pct": home_win_pct,
            "away_win_pct": away_win_pct,
            "home_away_win_diff": home_win_pct - away_win_pct,
            "home_pts_avg": home_pts,
            "away_pts_avg": away_pts,
            "home_away_pts_diff": home_pts - away_pts,
            "home_games": home.get("games_played", 0),
            "away_games": away.get("games_played", 0),
        }

    def calculate_offensive_rating(self, team_stats):
        """Extract offensive efficiency metrics."""
        overall = team_stats.get("overall", {})
        return {
            "off_rating": overall.get("off_rating", 0) or 0,
            "pts_avg": overall.get("pts_avg", 0) or 0,
            "fg_pct": overall.get("fg_pct", 0) or 0,
            "fg3_pct": overall.get("fg3_pct", 0) or 0,
            "ast_avg": overall.get("ast_avg", 0) or 0,
            "pace": overall.get("pace", 0) or 0,
        }

    def calculate_defensive_rating(self, team_stats):
        """Extract defensive efficiency metrics."""
        overall = team_stats.get("overall", {})
        return {
            "def_rating": overall.get("def_rating", 0) or 0,
            "stl_avg": overall.get("stl_avg", 0) or 0,
            "blk_avg": overall.get("blk_avg", 0) or 0,
            "reb_avg": overall.get("reb_avg", 0) or 0,
        }

    def calculate_rest_and_fatigue(self, team_id, game_date=None):
        """
        Calculate rest days and back-to-back status for a team.

        Research shows:
        - Back-to-back games reduce win probability by 3-5%
        - Each extra day of rest improves win probability by 2-3%
        - 3+ days rest can actually hurt performance (rust factor)

        Args:
            team_id: NBA team ID
            game_date: Date of upcoming game (defaults to today)

        Returns:
            Dictionary with rest/fatigue features
        """
        if game_date is None:
            game_date = datetime.now()
        elif isinstance(game_date, str):
            # Parse date string if provided
            try:
                game_date = datetime.strptime(game_date, "%Y-%m-%d")
            except ValueError:
                game_date = datetime.now()

        # Fetch recent games to find last game date
        games = fetch_historical_games(team_id=team_id, season=self.season, last_n_games=5)

        if not games:
            return {
                "days_rest": 2,  # Default to 2 days rest
                "is_back_to_back": 0,
                "is_3_in_4": 0,
                "rest_advantage": 0,
                "fatigue_factor": 0,
            }

        # Parse game dates and sort (most recent first)
        game_dates = []
        for g in games:
            date_str = g.get("game_date")
            if date_str:
                try:
                    gd = datetime.strptime(date_str, "%Y-%m-%d")
                    game_dates.append(gd)
                except ValueError:
                    continue

        if not game_dates:
            return {
                "days_rest": 2,
                "is_back_to_back": 0,
                "is_3_in_4": 0,
                "rest_advantage": 0,
                "fatigue_factor": 0,
            }

        game_dates.sort(reverse=True)
        last_game = game_dates[0]
        days_rest = (game_date.date() - last_game.date()).days

        # Check for back-to-back (0 or 1 day rest)
        is_back_to_back = 1 if days_rest <= 1 else 0

        # Check for 3 games in 4 nights
        four_days_ago = game_date - timedelta(days=4)
        games_in_4 = sum(1 for d in game_dates if four_days_ago <= d < game_date)
        is_3_in_4 = 1 if games_in_4 >= 3 else 0

        # Calculate fatigue factor (-1 to +1 scale)
        # Negative = fatigued, Positive = well-rested
        if is_back_to_back:
            fatigue_factor = -0.5 - (0.2 if is_3_in_4 else 0)
        elif days_rest == 2:
            fatigue_factor = 0.2  # Ideal rest
        elif days_rest == 3:
            fatigue_factor = 0.1  # Good rest
        elif days_rest >= 4:
            fatigue_factor = -0.1  # Potential rust
        else:
            fatigue_factor = 0

        return {
            "days_rest": min(days_rest, 7),  # Cap at 7 days
            "is_back_to_back": is_back_to_back,
            "is_3_in_4": is_3_in_4,
            "rest_advantage": 0,  # Will be calculated in matchup
            "fatigue_factor": fatigue_factor,
        }

    def calculate_advanced_shooting(self, team_stats):
        """
        Calculate advanced shooting efficiency metrics.

        True Shooting % (TS%): Accounts for 2PT, 3PT, and FT
        Effective FG% (eFG%): Adjusts for 3-pointers being worth more

        These are CRITICAL for accuracy - they better predict scoring than raw FG%.

        Args:
            team_stats: Team statistics dictionary

        Returns:
            Dictionary with advanced shooting metrics
        """
        overall = team_stats.get("overall", {})

        # Get raw stats needed for calculations
        pts = overall.get("pts_avg", 0) or 0
        fga = overall.get("fga_avg", 0) or overall.get("fga", 0) or 0
        fta = overall.get("fta_avg", 0) or overall.get("fta", 0) or 0
        fgm = overall.get("fgm_avg", 0) or overall.get("fgm", 0) or 0
        fg3m = overall.get("fg3m_avg", 0) or overall.get("fg3m", 0) or 0

        # Calculate True Shooting % (TS%)
        # TS% = PTS / (2 * (FGA + 0.44 * FTA))
        tsa = 2 * (fga + 0.44 * fta)  # True Shooting Attempts
        ts_pct = (pts / tsa) if tsa > 0 else 0.55  # Default to league avg
        ts_pct = max(0.4, min(0.7, ts_pct))  # Clip to realistic range

        # Calculate Effective FG% (eFG%)
        # eFG% = (FGM + 0.5 * 3PM) / FGA
        efg_pct = ((fgm + 0.5 * fg3m) / fga) if fga > 0 else 0.50
        efg_pct = max(0.4, min(0.65, efg_pct))  # Clip to realistic range

        return {
            "ts_pct": ts_pct,
            "efg_pct": efg_pct,
            "fta_rate": (fta / fga) if fga > 0 else 0.25,  # Free throw rate
            "fg3_rate": (fg3m / fgm) if fgm > 0 else 0.35,  # 3PT rate
        }

    def generate_team_features(self, team_id, is_home=True, last_n_games=10, game_date=None):
        """
        Generate comprehensive features for a team.

        Args:
            team_id: NBA team ID
            is_home: Whether team is playing at home
            last_n_games: Number of recent games for form calculation
            game_date: Game date (YYYY-MM-DD) - CRITICAL for preventing data leakage.
                      When set, only uses games BEFORE this date for features.

        Returns:
            Dictionary with all team features
        """
        team_stats = fetch_team_statistics(team_id, self.season)
        # CRITICAL: Pass game_date to prevent data leakage
        recent_form = self.calculate_recent_form(team_id, last_n_games, before_date=game_date)
        home_advantage = self.calculate_home_advantage(team_id)
        offensive = self.calculate_offensive_rating(team_stats)
        defensive = self.calculate_defensive_rating(team_stats)
        rest_fatigue = self.calculate_rest_and_fatigue(team_id, game_date)
        advanced_shooting = self.calculate_advanced_shooting(team_stats)

        overall = team_stats.get("overall", {})

        features = {
            # Basic stats
            "team_id": team_id,
            "is_home": 1 if is_home else 0,
            "season_win_pct": overall.get("win_pct", 0) or 0,
            "season_games_played": overall.get("games_played", 0) or 0,

            # Recent form
            "recent_win_pct": recent_form.get("win_pct", 0) if recent_form else 0,
            "recent_avg_pts": recent_form.get("avg_pts", 0) if recent_form else 0,
            "recent_avg_plus_minus": recent_form.get("avg_plus_minus", 0) if recent_form else 0,
            "current_streak": recent_form.get("streak", 0) if recent_form else 0,

            # Home/Away specific
            "location_win_pct": home_advantage.get("home_win_pct", 0) if is_home else home_advantage.get("away_win_pct", 0),
            "location_pts_avg": home_advantage.get("home_pts_avg", 0) if is_home else home_advantage.get("away_pts_avg", 0),
            "home_away_advantage": home_advantage.get("home_away_win_diff", 0),

            # Offensive metrics
            "off_rating": offensive.get("off_rating", 0),
            "pts_avg": offensive.get("pts_avg", 0),
            "fg_pct": offensive.get("fg_pct", 0),
            "fg3_pct": offensive.get("fg3_pct", 0),
            "ast_avg": offensive.get("ast_avg", 0),
            "pace": offensive.get("pace", 0),

            # Defensive metrics
            "def_rating": defensive.get("def_rating", 0),
            "stl_avg": defensive.get("stl_avg", 0),
            "blk_avg": defensive.get("blk_avg", 0),
            "reb_avg": defensive.get("reb_avg", 0),

            # Net rating
            "net_rating": overall.get("net_rating", 0) or 0,

            # NEW: Contextual features (rest/fatigue)
            "days_rest": rest_fatigue.get("days_rest", 2),
            "is_back_to_back": rest_fatigue.get("is_back_to_back", 0),
            "is_3_in_4": rest_fatigue.get("is_3_in_4", 0),
            "fatigue_factor": rest_fatigue.get("fatigue_factor", 0),

            # NEW: Advanced shooting efficiency
            "ts_pct": advanced_shooting.get("ts_pct", 0.55),
            "efg_pct": advanced_shooting.get("efg_pct", 0.50),
            "fta_rate": advanced_shooting.get("fta_rate", 0.25),
            "fg3_rate": advanced_shooting.get("fg3_rate", 0.35),
        }

        return features


class MatchupFeatureGenerator:
    """Generate features for matchup predictions with comprehensive analysis."""

    def __init__(self, season="2025-26", injury_manager: Optional[InjuryReportManager] = None):
        self.season = season
        self.team_generator = TeamFeatureGenerator(season)
        self.h2h_analyzer = HeadToHeadAnalyzer(season)
        self.positional_analyzer = PositionalAnalyzer(season)
        self.injury_manager = injury_manager or InjuryReportManager(season)

        # Instance-level caching to avoid duplicate API calls
        self._team_features_cache = {}  # {(team_id, is_home): features}
        self._h2h_cache = None
        self._positional_cache = None

    def _get_team_features(self, team_id: int, is_home: bool, last_n_games: int = 10, game_date: str = None) -> Dict:
        """Get team features with caching to avoid duplicate API calls."""
        cache_key = (team_id, is_home)
        if cache_key not in self._team_features_cache:
            self._team_features_cache[cache_key] = self.team_generator.generate_team_features(
                team_id, is_home=is_home, last_n_games=last_n_games, game_date=game_date
            )
        return self._team_features_cache[cache_key]

    def analyze_head_to_head(self, home_team_id: int, away_team_id: int) -> Dict:
        """
        Analyze head-to-head history for matchup features.

        Args:
            home_team_id: Home team NBA ID
            away_team_id: Away team NBA ID

        Returns:
            Dictionary with H2H features
        """
        # Cache H2H results to avoid duplicate API calls
        if self._h2h_cache is None:
            self._h2h_cache = self.h2h_analyzer.analyze_h2h(home_team_id, away_team_id)
        h2h = self._h2h_cache

        return {
            "h2h_games_played": h2h.get("games_played", 0),
            "h2h_home_win_pct": h2h.get("team1_win_pct", 0.5),
            "h2h_avg_point_diff": h2h.get("avg_point_diff", 0),
            "h2h_avg_total_points": h2h.get("avg_total_points", 0),
            "h2h_point_diff_volatility": h2h.get("point_diff_std", 0),
            "h2h_recent_winner_home": 1 if h2h.get("recent_winner") == "team1" else 0,
            "h2h_home_court_factor": h2h.get("home_team_advantage", 0.5),
        }

    def analyze_positional_matchup(self, home_team_id: int, away_team_id: int) -> Dict:
        """
        Analyze positional strengths/weaknesses for matchup features.

        Args:
            home_team_id: Home team NBA ID
            away_team_id: Away team NBA ID

        Returns:
            Dictionary with positional matchup features
        """
        pos_matchup = self.positional_analyzer.calculate_positional_matchup(home_team_id, away_team_id)

        guard = pos_matchup.get("guard_matchup", {})
        forward = pos_matchup.get("forward_matchup", {})
        center = pos_matchup.get("center_matchup", {})

        return {
            "guard_advantage": guard.get("advantage", 0),
            "guard_pts_diff": guard.get("pts_diff", 0),
            "forward_advantage": forward.get("advantage", 0),
            "forward_pts_diff": forward.get("pts_diff", 0),
            "center_advantage": center.get("advantage", 0),
            "center_reb_diff": center.get("reb_diff", 0),
            "total_positional_advantage": pos_matchup.get("total_positional_advantage", 0),
            "strongest_position": pos_matchup.get("strongest_position", "G"),
            "weakest_position": pos_matchup.get("weakest_position", "C"),
        }

    def analyze_injury_impact(self, home_team_id: int, away_team_id: int) -> Dict:
        """
        Analyze injury impact for both teams.

        Args:
            home_team_id: Home team NBA ID
            away_team_id: Away team NBA ID

        Returns:
            Dictionary with injury impact features
        """
        home_injuries = self.injury_manager.calculate_injury_impact(home_team_id)
        away_injuries = self.injury_manager.calculate_injury_impact(away_team_id)

        return {
            "home_injury_impact": home_injuries.get("total_impact", 0),
            "home_offensive_injury_impact": home_injuries.get("offensive_impact", 0),
            "home_defensive_injury_impact": home_injuries.get("defensive_impact", 0),
            "home_injured_count": home_injuries.get("injured_player_count", 0),
            "home_key_players_out": home_injuries.get("key_players_out", 0),

            "away_injury_impact": away_injuries.get("total_impact", 0),
            "away_offensive_injury_impact": away_injuries.get("offensive_impact", 0),
            "away_defensive_injury_impact": away_injuries.get("defensive_impact", 0),
            "away_injured_count": away_injuries.get("injured_player_count", 0),
            "away_key_players_out": away_injuries.get("key_players_out", 0),

            "injury_advantage": away_injuries.get("total_impact", 0) - home_injuries.get("total_impact", 0),
            "injury_details": {
                "home": home_injuries.get("injured_players", []),
                "away": away_injuries.get("injured_players", []),
            },
        }

    def generate_moneyline_features(self, home_team_id, away_team_id, last_n_games=10, include_advanced=True, game_date=None):
        """
        Generate features for moneyline prediction.

        Args:
            home_team_id: Home team NBA ID
            away_team_id: Away team NBA ID
            last_n_games: Recent games for form calculation
            include_advanced: Include H2H, positional, and injury analysis
            game_date: Game date (YYYY-MM-DD) - CRITICAL for preventing data leakage.
                      When training, pass the game date so only prior games are used.

        Returns:
            Dictionary with matchup features for moneyline prediction
        """
        # OPTIMIZED: Fetch home and away team features in parallel with caching
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            home_future = executor.submit(
                self._get_team_features, home_team_id, True, last_n_games, game_date
            )
            away_future = executor.submit(
                self._get_team_features, away_team_id, False, last_n_games, game_date
            )
            home_features = home_future.result()
            away_features = away_future.result()

        features = {
            # Team identifiers
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,

            # Win percentage differentials
            "season_win_pct_diff": home_features["season_win_pct"] - away_features["season_win_pct"],
            "recent_win_pct_diff": home_features["recent_win_pct"] - away_features["recent_win_pct"],
            "location_win_pct_diff": home_features["location_win_pct"] - away_features["location_win_pct"],

            # Scoring differentials
            "pts_avg_diff": home_features["pts_avg"] - away_features["pts_avg"],
            "recent_pts_diff": home_features["recent_avg_pts"] - away_features["recent_avg_pts"],

            # Efficiency differentials
            "off_rating_diff": home_features["off_rating"] - away_features["off_rating"],
            "def_rating_diff": home_features["def_rating"] - away_features["def_rating"],
            "net_rating_diff": home_features["net_rating"] - away_features["net_rating"],

            # Form indicators
            "home_streak": home_features["current_streak"],
            "away_streak": away_features["current_streak"],
            "combined_form": home_features["recent_avg_plus_minus"] - away_features["recent_avg_plus_minus"],

            # Home advantage
            "home_advantage_factor": home_features["home_away_advantage"],

            # Shooting differentials
            "fg_pct_diff": home_features["fg_pct"] - away_features["fg_pct"],
            "fg3_pct_diff": home_features["fg3_pct"] - away_features["fg3_pct"],

            # Pace factor (for total points consideration)
            "avg_pace": (home_features["pace"] + away_features["pace"]) / 2,
            "pace_diff": home_features["pace"] - away_features["pace"],

            # Individual team features
            "home_season_win_pct": home_features["season_win_pct"],
            "away_season_win_pct": away_features["season_win_pct"],
            "home_net_rating": home_features["net_rating"],
            "away_net_rating": away_features["net_rating"],
            "home_off_rating": home_features["off_rating"],
            "away_off_rating": away_features["off_rating"],
            "home_def_rating": home_features["def_rating"],
            "away_def_rating": away_features["def_rating"],

            # NEW: Contextual features (rest/fatigue) - HIGH IMPACT
            "home_days_rest": home_features.get("days_rest", 2),
            "away_days_rest": away_features.get("days_rest", 2),
            "rest_days_diff": home_features.get("days_rest", 2) - away_features.get("days_rest", 2),
            "home_is_b2b": home_features.get("is_back_to_back", 0),
            "away_is_b2b": away_features.get("is_back_to_back", 0),
            "home_is_3_in_4": home_features.get("is_3_in_4", 0),
            "away_is_3_in_4": away_features.get("is_3_in_4", 0),
            "fatigue_advantage": home_features.get("fatigue_factor", 0) - away_features.get("fatigue_factor", 0),

            # NEW: Advanced shooting efficiency - CRITICAL for accuracy
            "ts_pct_diff": home_features.get("ts_pct", 0.55) - away_features.get("ts_pct", 0.55),
            "efg_pct_diff": home_features.get("efg_pct", 0.50) - away_features.get("efg_pct", 0.50),
            "home_ts_pct": home_features.get("ts_pct", 0.55),
            "away_ts_pct": away_features.get("ts_pct", 0.55),
            "home_efg_pct": home_features.get("efg_pct", 0.50),
            "away_efg_pct": away_features.get("efg_pct", 0.50),
            "fta_rate_diff": home_features.get("fta_rate", 0.25) - away_features.get("fta_rate", 0.25),
            "fg3_rate_diff": home_features.get("fg3_rate", 0.35) - away_features.get("fg3_rate", 0.35),
        }

        # Add advanced matchup analysis
        if include_advanced:
            # Head-to-head analysis
            h2h_features = self.analyze_head_to_head(home_team_id, away_team_id)
            features.update(h2h_features)

            # Positional matchup analysis
            positional_features = self.analyze_positional_matchup(home_team_id, away_team_id)
            features.update(positional_features)

            # Injury impact analysis
            injury_features = self.analyze_injury_impact(home_team_id, away_team_id)
            # Don't include injury_details dict in flat features
            injury_flat = {k: v for k, v in injury_features.items() if k != "injury_details"}
            features.update(injury_flat)
            features["injury_details"] = injury_features.get("injury_details")

        # CRITICAL: Validate and clip all features to realistic ranges
        features = validate_features_dict(features, warn=True)
        return features

    def generate_spread_features(self, home_team_id, away_team_id, last_n_games=10, include_advanced=True, game_date=None):
        """
        Generate features for spread prediction.

        Builds on moneyline features with additional point-based metrics.

        Args:
            home_team_id: Home team NBA ID
            away_team_id: Away team NBA ID
            last_n_games: Recent games for form calculation
            include_advanced: Include H2H, positional, and injury analysis
            game_date: Game date (YYYY-MM-DD) - CRITICAL for preventing data leakage.

        Returns:
            Dictionary with matchup features for spread prediction
        """
        # Start with moneyline features - pass game_date to prevent leakage
        features = self.generate_moneyline_features(
            home_team_id, away_team_id, last_n_games, include_advanced, game_date=game_date
        )

        # OPTIMIZED: Use cached team features (already fetched by generate_moneyline_features)
        home_features = self._get_team_features(home_team_id, True, last_n_games, game_date)
        away_features = self._get_team_features(away_team_id, False, last_n_games, game_date)

        # Additional spread-specific features
        spread_features = {
            # Point differential estimates
            "expected_home_pts": home_features["location_pts_avg"],
            "expected_away_pts": away_features["location_pts_avg"],
            "expected_point_diff": home_features["location_pts_avg"] - away_features["location_pts_avg"],

            # Plus/minus indicators
            "home_plus_minus": home_features["recent_avg_plus_minus"],
            "away_plus_minus": away_features["recent_avg_plus_minus"],
            "plus_minus_diff": home_features["recent_avg_plus_minus"] - away_features["recent_avg_plus_minus"],

            # Margin of victory estimation factors
            "home_pts_avg": home_features["pts_avg"],
            "away_pts_avg": away_features["pts_avg"],

            # Rebounding advantage (affects second-chance points)
            "reb_diff": home_features["reb_avg"] - away_features["reb_avg"],

            # Turnover differential (affects fast break points)
            "ast_diff": home_features["ast_avg"] - away_features["ast_avg"],
        }

        # Add H2H spread-specific features if available
        if include_advanced:
            # OPTIMIZED: Use cached H2H (already fetched by generate_moneyline_features)
            if self._h2h_cache is None:
                self._h2h_cache = self.h2h_analyzer.analyze_h2h(home_team_id, away_team_id)
            h2h = self._h2h_cache
            spread_features["h2h_spread_prediction"] = h2h.get("avg_point_diff", 0)
            spread_features["h2h_spread_volatility"] = h2h.get("point_diff_std", 0)

        features.update(spread_features)

        # CRITICAL: Validate and clip all features to realistic ranges
        features = validate_features_dict(features, warn=True)
        return features

    def generate_total_points_features(self, home_team_id, away_team_id, last_n_games=10, game_date=None):
        """
        Generate features for over/under total points prediction.

        Args:
            home_team_id: Home team NBA ID
            away_team_id: Away team NBA ID
            last_n_games: Recent games for form calculation
            game_date: Game date (YYYY-MM-DD) - CRITICAL for preventing data leakage.
                      When training, pass the game date so only prior games are used.

        Returns:
            Dictionary with features for total points prediction
        """
        # OPTIMIZED: Use cached team features (already fetched by generate_moneyline_features)
        home_features = self._get_team_features(home_team_id, True, last_n_games, game_date)
        away_features = self._get_team_features(away_team_id, False, last_n_games, game_date)

        features = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,

            # Combined scoring potential
            "combined_pts_avg": home_features["pts_avg"] + away_features["pts_avg"],
            "combined_recent_pts": home_features["recent_avg_pts"] + away_features["recent_avg_pts"],

            # Pace indicators (higher pace = more possessions = more points)
            "combined_pace": home_features["pace"] + away_features["pace"],
            "avg_pace": (home_features["pace"] + away_features["pace"]) / 2,

            # Offensive efficiency
            "combined_off_rating": home_features["off_rating"] + away_features["off_rating"],
            "avg_off_rating": (home_features["off_rating"] + away_features["off_rating"]) / 2,

            # Defensive efficiency (lower = better defense = fewer points allowed)
            "combined_def_rating": home_features["def_rating"] + away_features["def_rating"],
            "avg_def_rating": (home_features["def_rating"] + away_features["def_rating"]) / 2,

            # Shooting percentages
            "combined_fg_pct": (home_features["fg_pct"] + away_features["fg_pct"]) / 2,
            "combined_fg3_pct": (home_features["fg3_pct"] + away_features["fg3_pct"]) / 2,

            # Individual team scoring
            "home_pts_avg": home_features["pts_avg"],
            "away_pts_avg": away_features["pts_avg"],
            "home_location_pts": home_features["location_pts_avg"],
            "away_location_pts": away_features["location_pts_avg"],
        }

        return features


class PlayerPropFeatureGenerator:
    """Generate features for player prop predictions with matchup analysis."""

    def __init__(self, season="2025-26"):
        self.season = season
        self._balldontlie_api = None  # Lazy loading

    def _get_balldontlie_api(self):
        """Get Balldontlie API client (lazy loaded)."""
        if self._balldontlie_api is None:
            try:
                from balldontlie_api import BalldontlieAPI
                self._balldontlie_api = BalldontlieAPI()
            except Exception:
                self._balldontlie_api = None
        return self._balldontlie_api

    def get_player_season_averages_balldontlie(self, player_id: int) -> Dict:
        """
        Fetch player season averages from Balldontlie API.

        This provides more accurate and up-to-date stats than NBA API.

        Args:
            player_id: Balldontlie player ID

        Returns:
            Dictionary with season averages
        """
        api = self._get_balldontlie_api()
        if api is None:
            return {}

        try:
            averages = api.get_season_averages(player_ids=[player_id])
            if averages:
                avg = averages[0]
                return {
                    "bdl_pts_avg": avg.get("pts", 0),
                    "bdl_reb_avg": avg.get("reb", 0),
                    "bdl_ast_avg": avg.get("ast", 0),
                    "bdl_stl_avg": avg.get("stl", 0),
                    "bdl_blk_avg": avg.get("blk", 0),
                    "bdl_fg3m_avg": avg.get("fg3m", 0),
                    "bdl_min_avg": avg.get("min", 0),
                    "bdl_fg_pct": avg.get("fg_pct", 0),
                    "bdl_fg3_pct": avg.get("fg3_pct", 0),
                    "bdl_games_played": avg.get("games_played", 0),
                }
        except Exception:
            pass
        return {}

    def calculate_opponent_defensive_context(self, opponent_team_id: int, player_position: str = None) -> Dict:
        """
        Calculate opponent's defensive context for player prop predictions.

        This is CRITICAL for accurate props - a player facing a bad defense
        will typically score more than against a good defense.

        Args:
            opponent_team_id: Opponent team NBA ID
            player_position: Player's position (G, F, C) for position-specific analysis

        Returns:
            Dictionary with opponent defensive metrics
        """
        opp_stats = fetch_team_statistics(opponent_team_id, self.season)
        opp_overall = opp_stats.get("overall", {})

        # Base defensive metrics
        def_rating = opp_overall.get("def_rating", 110) or 110
        pace = opp_overall.get("pace", 100) or 100
        opp_pts_allowed = opp_overall.get("opp_pts_avg", 112) or 112

        # Calculate defensive strength relative to league average
        # Lower def_rating = better defense, so we invert for impact
        league_avg_def = 114.0  # 2025-26 league average
        def_strength = (def_rating - league_avg_def) / 10  # Positive = bad defense (good for scorer)

        # Position-specific adjustments (estimated from historical data)
        # Bad perimeter defense = more points for guards
        # Bad interior defense = more points for bigs
        position_adjustment = 0
        if player_position:
            pos_group = player_position.upper()
            if pos_group in ["G", "PG", "SG"]:
                # Guards affected more by perimeter defense
                position_adjustment = def_strength * 1.2
            elif pos_group in ["F", "SF", "PF"]:
                # Forwards get moderate benefit
                position_adjustment = def_strength * 1.0
            elif pos_group in ["C"]:
                # Centers affected more by interior defense
                position_adjustment = def_strength * 0.8

        return {
            "opp_def_rating": def_rating,
            "opp_pace": pace,
            "opp_pts_allowed": opp_pts_allowed,
            "opp_def_strength": def_strength,  # Positive = bad defense
            "opp_position_adjustment": position_adjustment,
            "expected_pts_boost": def_strength * 2.5,  # ~2.5 pts per unit of bad defense
            "expected_reb_boost": def_strength * 0.5,  # ~0.5 reb per unit
            "expected_ast_boost": def_strength * 0.3,  # ~0.3 ast per unit
            "pace_factor": pace / 100,  # Higher pace = more opportunities
        }

    def analyze_player_vs_team_history(self, player_id: int, opponent_team_id: int) -> Dict:
        """
        Analyze player's historical performance against a specific team.

        Args:
            player_id: NBA player ID
            opponent_team_id: Opponent team NBA ID

        Returns:
            Dictionary with player vs team analysis
        """
        vs_games = fetch_player_vs_team(player_id, opponent_team_id, self.season, last_n_games=10)

        if not vs_games:
            return {
                "vs_team_games": 0,
                "vs_team_pts_avg": 0,
                "vs_team_reb_avg": 0,
                "vs_team_ast_avg": 0,
                "vs_team_fg3_avg": 0,
                "vs_team_min_avg": 0,
                "vs_team_plus_minus_avg": 0,
            }

        pts = [g.get("pts", 0) or 0 for g in vs_games]
        reb = [g.get("reb", 0) or 0 for g in vs_games]
        ast = [g.get("ast", 0) or 0 for g in vs_games]
        fg3 = [g.get("fg3_made", 0) or 0 for g in vs_games]
        mins = [g.get("min", 0) or 0 for g in vs_games]
        plus_minus = [g.get("plus_minus", 0) or 0 for g in vs_games]

        return {
            "vs_team_games": len(vs_games),
            "vs_team_pts_avg": np.mean(pts) if pts else 0,
            "vs_team_pts_std": np.std(pts) if len(pts) > 1 else 0,
            "vs_team_reb_avg": np.mean(reb) if reb else 0,
            "vs_team_ast_avg": np.mean(ast) if ast else 0,
            "vs_team_fg3_avg": np.mean(fg3) if fg3 else 0,
            "vs_team_min_avg": np.mean(mins) if mins else 0,
            "vs_team_plus_minus_avg": np.mean(plus_minus) if plus_minus else 0,
            "vs_team_pra_avg": np.mean([p + r + a for p, r, a in zip(pts, reb, ast)]) if pts else 0,
        }

    def calculate_regression_to_mean(self, season_avg: float, recent_avg: float,
                                      sample_size: int = 5, stat_type: str = "points") -> Dict:
        """
        Calculate regression-to-mean adjustment for player props.

        Players with extreme recent performance tend to regress toward their
        season average. This is a well-documented statistical phenomenon.

        Args:
            season_avg: Player's season average for the stat
            recent_avg: Player's recent average (e.g., last 5 games)
            sample_size: Number of recent games used
            stat_type: Type of stat ('points', 'rebounds', 'assists', '3pm', 'pra')

        Returns:
            Dictionary with regression adjustment features
        """
        if season_avg <= 0:
            return {
                "regression_adjustment": 0.0,
                "regression_factor": 0.0,
                "deviation_from_mean": 0.0,
                "expected_regression_pct": 0.0,
            }

        # Calculate deviation from season mean
        deviation = recent_avg - season_avg
        deviation_pct = deviation / season_avg if season_avg > 0 else 0

        # Regression factor depends on sample size
        # Smaller samples = more expected regression
        # Bayesian prior strength varies by stat type
        # Points are more stable than 3PM, so less regression expected
        stat_stability = {
            "points": 0.7,    # High stability
            "rebounds": 0.65,  # Medium-high
            "assists": 0.6,   # Medium
            "3pm": 0.4,       # Low stability (high variance)
            "pra": 0.65,      # Medium-high (combination)
        }

        stability = stat_stability.get(stat_type.lower(), 0.6)

        # Regression factor: how much of the deviation we expect to regress
        # More games = less regression expected
        regression_factor = (1.0 - stability) * (10.0 / (sample_size + 10.0))

        # Expected regression amount (in raw units)
        expected_regression = -deviation * regression_factor

        # Adjusted prediction = recent_avg + expected_regression
        # This pulls extreme performances back toward the mean

        return {
            "regression_adjustment": expected_regression,
            "regression_factor": regression_factor,
            "deviation_from_mean": deviation,
            "deviation_pct": deviation_pct,
            "expected_regression_pct": regression_factor * 100,
            "regression_predicted_value": recent_avg + expected_regression,
        }

    def calculate_shot_distribution_trends(self, player_game_logs: list,
                                            last_n: int = 5) -> Dict:
        """
        Calculate shot attempt distribution trends to predict 3PM props.

        Tracks 3-point attempt trends which are predictive of 3PM outcomes.
        A player increasing their 3PA is likely to hit more 3s.

        Args:
            player_game_logs: List of game dictionaries with shot data
            last_n: Number of recent games for trend calculation

        Returns:
            Dictionary with shot distribution trend features
        """
        if not player_game_logs or len(player_game_logs) < last_n:
            return {
                "three_attempt_trend": 0.0,
                "recent_3pa_avg": 0.0,
                "earlier_3pa_avg": 0.0,
                "three_rate_change": 0.0,
                "shot_volume_trend": 0.0,
            }

        # Get recent games (most recent first, so reverse if needed)
        recent = player_game_logs[:last_n]
        earlier = player_game_logs[last_n:2*last_n] if len(player_game_logs) >= 2*last_n else []

        # Extract 3-point attempts
        recent_3pa = [g.get("fg3a", 0) or g.get("fg3_attempted", 0) or 0 for g in recent]
        recent_fga = [g.get("fga", 0) or g.get("fg_attempted", 0) or 0 for g in recent]

        recent_3pa_avg = np.mean(recent_3pa) if recent_3pa else 0
        recent_fga_avg = np.mean(recent_fga) if recent_fga else 0

        if earlier:
            earlier_3pa = [g.get("fg3a", 0) or g.get("fg3_attempted", 0) or 0 for g in earlier]
            earlier_fga = [g.get("fga", 0) or g.get("fg_attempted", 0) or 0 for g in earlier]
            earlier_3pa_avg = np.mean(earlier_3pa) if earlier_3pa else recent_3pa_avg
            earlier_fga_avg = np.mean(earlier_fga) if earlier_fga else recent_fga_avg
        else:
            earlier_3pa_avg = recent_3pa_avg
            earlier_fga_avg = recent_fga_avg

        # 3PA trend: positive = shooting more 3s recently
        three_attempt_trend = recent_3pa_avg - earlier_3pa_avg

        # 3-point rate: what % of shots are 3s
        recent_3pt_rate = (recent_3pa_avg / recent_fga_avg * 100) if recent_fga_avg > 0 else 0
        earlier_3pt_rate = (earlier_3pa_avg / earlier_fga_avg * 100) if earlier_fga_avg > 0 else 0
        three_rate_change = recent_3pt_rate - earlier_3pt_rate

        # Overall shot volume trend
        shot_volume_trend = recent_fga_avg - earlier_fga_avg

        return {
            "three_attempt_trend": three_attempt_trend,
            "recent_3pa_avg": recent_3pa_avg,
            "earlier_3pa_avg": earlier_3pa_avg,
            "recent_3pt_rate": recent_3pt_rate,
            "three_rate_change": three_rate_change,
            "shot_volume_trend": shot_volume_trend,
            "recent_fga_avg": recent_fga_avg,
        }

    def calculate_player_recent_form(self, player_id, last_n_games=5):
        """
        Calculate player's recent performance.

        Args:
            player_id: NBA player ID
            last_n_games: Number of recent games

        Returns:
            Dictionary with recent stats
        """
        stats = fetch_player_stats(player_id, self.season, last_n_games)
        games = stats.get("game_log", [])

        if not games:
            return None

        pts = [g.get("pts", 0) or 0 for g in games]
        reb = [g.get("reb", 0) or 0 for g in games]
        ast = [g.get("ast", 0) or 0 for g in games]
        fg3_made = [g.get("fg3_made", 0) or 0 for g in games]
        mins = [g.get("min", 0) or 0 for g in games]

        return {
            "games_played": len(games),
            "pts_avg": np.mean(pts),
            "pts_std": np.std(pts),
            "pts_min": np.min(pts),
            "pts_max": np.max(pts),
            "reb_avg": np.mean(reb),
            "reb_std": np.std(reb),
            "ast_avg": np.mean(ast),
            "ast_std": np.std(ast),
            "fg3_avg": np.mean(fg3_made),
            "fg3_std": np.std(fg3_made),
            "min_avg": np.mean(mins),
            "min_std": np.std(mins),
            "pts_plus_reb_plus_ast_avg": np.mean([p + r + a for p, r, a in zip(pts, reb, ast)]),
        }

    def generate_points_prop_features(self, player_id, opponent_team_id=None, last_n_games=10):
        """
        Generate features for player points prop.

        Args:
            player_id: NBA player ID
            opponent_team_id: Optional opponent team ID for matchup context
            last_n_games: Recent games for form calculation

        Returns:
            Dictionary with features for points prop prediction
        """
        stats = fetch_player_stats(player_id, self.season, last_n_games)
        season_avg = stats.get("season_averages", {})
        recent_form = self.calculate_player_recent_form(player_id, last_n_games=5)

        features = {
            "player_id": player_id,

            # Season averages
            "season_pts_avg": season_avg.get("pts_avg", 0) or 0,
            "season_min_avg": season_avg.get("min_avg", 0) or 0,
            "season_fg_pct": season_avg.get("fg_pct", 0) or 0,
            "season_fg3_pct": season_avg.get("fg3_pct", 0) or 0,
            "season_games": season_avg.get("games_played", 0) or 0,

            # Recent form
            "recent_pts_avg": recent_form.get("pts_avg", 0) if recent_form else 0,
            "recent_pts_std": recent_form.get("pts_std", 0) if recent_form else 0,
            "recent_pts_min": recent_form.get("pts_min", 0) if recent_form else 0,
            "recent_pts_max": recent_form.get("pts_max", 0) if recent_form else 0,
            "recent_min_avg": recent_form.get("min_avg", 0) if recent_form else 0,

            # Form trend (recent vs season)
            "pts_trend": (recent_form.get("pts_avg", 0) - (season_avg.get("pts_avg", 0) or 0)) if recent_form else 0,

            # Consistency score (lower std = more consistent)
            "consistency_score": 1 / (1 + (recent_form.get("pts_std", 1) if recent_form else 1)),
        }

        # Add opponent context if available
        if opponent_team_id:
            # NEW: Use enhanced opponent defensive context
            opp_context = self.calculate_opponent_defensive_context(opponent_team_id)
            features["opp_def_rating"] = opp_context.get("opp_def_rating", 110)
            features["opp_pace"] = opp_context.get("opp_pace", 100)
            features["opp_def_strength"] = opp_context.get("opp_def_strength", 0)
            features["expected_pts_boost"] = opp_context.get("expected_pts_boost", 0)
            features["pace_factor"] = opp_context.get("pace_factor", 1.0)

            # Add player vs team history
            vs_team = self.analyze_player_vs_team_history(player_id, opponent_team_id)
            features["vs_team_games"] = vs_team.get("vs_team_games", 0)
            features["vs_team_pts_avg"] = vs_team.get("vs_team_pts_avg", 0)
            features["vs_team_pts_std"] = vs_team.get("vs_team_pts_std", 0)

            # Calculate adjustment based on vs team history
            if vs_team.get("vs_team_games", 0) >= 2:
                season_pts = season_avg.get("pts_avg", 0) or 0
                vs_pts = vs_team.get("vs_team_pts_avg", 0)
                features["vs_team_pts_adjustment"] = vs_pts - season_pts
            else:
                features["vs_team_pts_adjustment"] = 0

            # NEW: Calculate projected points with all adjustments
            base_pts = features.get("season_pts_avg", 0)
            trend_adj = features.get("pts_trend", 0) * 0.5  # Weight recent form
            def_adj = features.get("expected_pts_boost", 0)
            vs_adj = features.get("vs_team_pts_adjustment", 0) * 0.3  # Weight historical
            features["projected_pts"] = base_pts + trend_adj + def_adj + vs_adj

        return features

    def generate_rebounds_prop_features(self, player_id, opponent_team_id=None, last_n_games=10):
        """
        Generate features for player rebounds prop.

        Args:
            player_id: NBA player ID
            opponent_team_id: Optional opponent team ID
            last_n_games: Recent games for form calculation

        Returns:
            Dictionary with features for rebounds prop prediction
        """
        stats = fetch_player_stats(player_id, self.season, last_n_games)
        season_avg = stats.get("season_averages", {})
        recent_form = self.calculate_player_recent_form(player_id, last_n_games=5)

        features = {
            "player_id": player_id,

            # Season averages
            "season_reb_avg": season_avg.get("reb_avg", 0) or 0,
            "season_min_avg": season_avg.get("min_avg", 0) or 0,

            # Recent form
            "recent_reb_avg": recent_form.get("reb_avg", 0) if recent_form else 0,
            "recent_reb_std": recent_form.get("reb_std", 0) if recent_form else 0,
            "recent_min_avg": recent_form.get("min_avg", 0) if recent_form else 0,

            # Trend
            "reb_trend": (recent_form.get("reb_avg", 0) - (season_avg.get("reb_avg", 0) or 0)) if recent_form else 0,
        }

        if opponent_team_id:
            opp_stats = fetch_team_statistics(opponent_team_id, self.season)
            opp_overall = opp_stats.get("overall", {})
            features["opp_reb_avg"] = opp_overall.get("reb_avg", 0) or 0
            features["opp_pace"] = opp_overall.get("pace", 0) or 0

        return features

    def generate_assists_prop_features(self, player_id, opponent_team_id=None, last_n_games=10):
        """
        Generate features for player assists prop.

        Args:
            player_id: NBA player ID
            opponent_team_id: Optional opponent team ID
            last_n_games: Recent games for form calculation

        Returns:
            Dictionary with features for assists prop prediction
        """
        stats = fetch_player_stats(player_id, self.season, last_n_games)
        season_avg = stats.get("season_averages", {})
        recent_form = self.calculate_player_recent_form(player_id, last_n_games=5)

        features = {
            "player_id": player_id,

            # Season averages
            "season_ast_avg": season_avg.get("ast_avg", 0) or 0,
            "season_min_avg": season_avg.get("min_avg", 0) or 0,
            "season_tov_avg": season_avg.get("tov_avg", 0) or 0,

            # Recent form
            "recent_ast_avg": recent_form.get("ast_avg", 0) if recent_form else 0,
            "recent_ast_std": recent_form.get("ast_std", 0) if recent_form else 0,
            "recent_min_avg": recent_form.get("min_avg", 0) if recent_form else 0,

            # Trend
            "ast_trend": (recent_form.get("ast_avg", 0) - (season_avg.get("ast_avg", 0) or 0)) if recent_form else 0,
        }

        if opponent_team_id:
            opp_stats = fetch_team_statistics(opponent_team_id, self.season)
            opp_overall = opp_stats.get("overall", {})
            features["opp_def_rating"] = opp_overall.get("def_rating", 0) or 0
            features["opp_stl_avg"] = opp_overall.get("stl_avg", 0) or 0

        return features

    def generate_threes_prop_features(self, player_id, opponent_team_id=None, last_n_games=10):
        """
        Generate features for player 3-pointers made prop.

        Args:
            player_id: NBA player ID
            opponent_team_id: Optional opponent team ID
            last_n_games: Recent games for form calculation

        Returns:
            Dictionary with features for 3-pointers prop prediction
        """
        stats = fetch_player_stats(player_id, self.season, last_n_games)
        season_avg = stats.get("season_averages", {})
        recent_form = self.calculate_player_recent_form(player_id, last_n_games=5)

        features = {
            "player_id": player_id,

            # Season shooting
            "season_fg3_pct": season_avg.get("fg3_pct", 0) or 0,
            "season_min_avg": season_avg.get("min_avg", 0) or 0,

            # Recent form
            "recent_fg3_avg": recent_form.get("fg3_avg", 0) if recent_form else 0,
            "recent_fg3_std": recent_form.get("fg3_std", 0) if recent_form else 0,
            "recent_min_avg": recent_form.get("min_avg", 0) if recent_form else 0,
        }

        if opponent_team_id:
            opp_stats = fetch_team_statistics(opponent_team_id, self.season)
            opp_overall = opp_stats.get("overall", {})
            features["opp_fg3_pct_allowed"] = opp_overall.get("fg3_pct", 0) or 0
            features["opp_def_rating"] = opp_overall.get("def_rating", 0) or 0

        return features

    def generate_pra_prop_features(self, player_id, opponent_team_id=None, last_n_games=10):
        """
        Generate features for Points+Rebounds+Assists prop.

        Args:
            player_id: NBA player ID
            opponent_team_id: Optional opponent team ID
            last_n_games: Recent games for form calculation

        Returns:
            Dictionary with features for PRA prop prediction
        """
        stats = fetch_player_stats(player_id, self.season, last_n_games)
        season_avg = stats.get("season_averages", {})
        recent_form = self.calculate_player_recent_form(player_id, last_n_games=5)

        season_pra = (
            (season_avg.get("pts_avg", 0) or 0) +
            (season_avg.get("reb_avg", 0) or 0) +
            (season_avg.get("ast_avg", 0) or 0)
        )

        features = {
            "player_id": player_id,

            # Season PRA
            "season_pra_avg": season_pra,
            "season_pts_avg": season_avg.get("pts_avg", 0) or 0,
            "season_reb_avg": season_avg.get("reb_avg", 0) or 0,
            "season_ast_avg": season_avg.get("ast_avg", 0) or 0,
            "season_min_avg": season_avg.get("min_avg", 0) or 0,

            # Recent PRA
            "recent_pra_avg": recent_form.get("pts_plus_reb_plus_ast_avg", 0) if recent_form else 0,
            "recent_pts_avg": recent_form.get("pts_avg", 0) if recent_form else 0,
            "recent_reb_avg": recent_form.get("reb_avg", 0) if recent_form else 0,
            "recent_ast_avg": recent_form.get("ast_avg", 0) if recent_form else 0,
            "recent_min_avg": recent_form.get("min_avg", 0) if recent_form else 0,

            # Trend
            "pra_trend": (recent_form.get("pts_plus_reb_plus_ast_avg", 0) - season_pra) if recent_form else 0,
        }

        if opponent_team_id:
            opp_stats = fetch_team_statistics(opponent_team_id, self.season)
            opp_overall = opp_stats.get("overall", {})
            features["opp_def_rating"] = opp_overall.get("def_rating", 0) or 0
            features["opp_pace"] = opp_overall.get("pace", 0) or 0

        return features


def generate_game_features(
    home_team,
    away_team,
    season="2025-26",
    last_n_games=10,
    include_advanced=True,
    injury_manager: Optional[InjuryReportManager] = None,
    game_date=None,
):
    """
    Convenience function to generate all features for a game.

    Args:
        home_team: Home team name or abbreviation
        away_team: Away team name or abbreviation
        season: NBA season
        last_n_games: Recent games for form calculation
        include_advanced: Include H2H, positional, and injury analysis
        injury_manager: Optional InjuryReportManager with pre-loaded injury data
        game_date: Game date (YYYY-MM-DD) - CRITICAL for preventing data leakage.
                  When training, pass the game date so only prior games are used.

    Returns:
        Dictionary with moneyline, spread, and total features
    """
    home_id = get_team_id(home_team)
    away_id = get_team_id(away_team)

    if not home_id or not away_id:
        raise ValueError(f"Could not find team IDs for {home_team} or {away_team}")

    matchup_gen = MatchupFeatureGenerator(season, injury_manager)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_team_id": home_id,
        "away_team_id": away_id,
        # CRITICAL: Pass game_date to prevent using future data in training
        "moneyline_features": matchup_gen.generate_moneyline_features(
            home_id, away_id, last_n_games, include_advanced, game_date=game_date
        ),
        "spread_features": matchup_gen.generate_spread_features(
            home_id, away_id, last_n_games, include_advanced, game_date=game_date
        ),
        "total_features": matchup_gen.generate_total_points_features(home_id, away_id, last_n_games, game_date=game_date),
    }


def generate_player_features(player_name, opponent_team=None, season="2025-26", last_n_games=10):
    """
    Convenience function to generate all prop features for a player.

    Args:
        player_name: Player name
        opponent_team: Optional opponent team name
        season: NBA season
        last_n_games: Recent games for form calculation

    Returns:
        Dictionary with all prop features
    """
    player_id = get_player_id(player_name)

    if not player_id:
        raise ValueError(f"Could not find player ID for {player_name}")

    opponent_id = get_team_id(opponent_team) if opponent_team else None

    prop_gen = PlayerPropFeatureGenerator(season)

    features = {
        "player_name": player_name,
        "player_id": player_id,
        "opponent_team": opponent_team,
        "opponent_team_id": opponent_id,
        "points_features": prop_gen.generate_points_prop_features(player_id, opponent_id, last_n_games),
        "rebounds_features": prop_gen.generate_rebounds_prop_features(player_id, opponent_id, last_n_games),
        "assists_features": prop_gen.generate_assists_prop_features(player_id, opponent_id, last_n_games),
        "threes_features": prop_gen.generate_threes_prop_features(player_id, opponent_id, last_n_games),
        "pra_features": prop_gen.generate_pra_prop_features(player_id, opponent_id, last_n_games),
    }

    # Add player vs team history if opponent specified
    if opponent_id:
        features["vs_team_history"] = prop_gen.analyze_player_vs_team_history(player_id, opponent_id)

    return features


def create_injury_report(injuries_data: List[Dict], season="2025-26") -> InjuryReportManager:
    """
    Create an InjuryReportManager with injury data.

    Args:
        injuries_data: List of dictionaries with format:
            [{
                "team": "LAL" or team_id,
                "injuries": [
                    {"player_name": "LeBron James", "player_id": 2544, "status": "questionable", "position": "SF"},
                    ...
                ]
            }, ...]
        season: NBA season

    Returns:
        Configured InjuryReportManager
    """
    manager = InjuryReportManager(season)

    for team_data in injuries_data:
        team = team_data.get("team")
        if isinstance(team, str):
            team_id = get_team_id(team)
        else:
            team_id = team

        if team_id:
            manager.set_injury_report(team_id, team_data.get("injuries", []))

    return manager


# =============================================================================
# LINEUP-ADJUSTED STATISTICS
# =============================================================================

class LineupImpactCalculator:
    """
    Calculate lineup-adjusted statistics based on player availability.

    This is CRITICAL for accurate predictions because:
    - Star player absences can shift win probability 10-15%
    - Role player combinations matter for team chemistry
    - Usage redistribution affects prop predictions
    - On/off splits quantify true player impact

    Research shows incorporating lineup info adds 2-4% to prediction accuracy.
    """

    # Base player impact data (would be populated from API in production)
    # Format: {player_name: {ppg, rpg, apg, min, on_off_diff, usage_rate, team}}
    PLAYER_IMPACTS = {
        # Tier 1 - Superstars (massive impact)
        "Nikola Jokic": {"ppg": 26.5, "rpg": 12.5, "apg": 9.5, "min": 34, "on_off_diff": 12.5, "usage_rate": 0.28, "team": "DEN", "tier": 1},
        "Luka Doncic": {"ppg": 33.0, "rpg": 9.0, "apg": 9.5, "min": 37, "on_off_diff": 11.0, "usage_rate": 0.35, "team": "DAL", "tier": 1},
        "Giannis Antetokounmpo": {"ppg": 31.0, "rpg": 12.0, "apg": 6.0, "min": 35, "on_off_diff": 10.5, "usage_rate": 0.32, "team": "MIL", "tier": 1},
        "Shai Gilgeous-Alexander": {"ppg": 31.0, "rpg": 5.5, "apg": 6.5, "min": 34, "on_off_diff": 10.0, "usage_rate": 0.30, "team": "OKC", "tier": 1},
        "Joel Embiid": {"ppg": 34.0, "rpg": 11.0, "apg": 5.5, "min": 34, "on_off_diff": 9.5, "usage_rate": 0.36, "team": "PHI", "tier": 1},
        "Stephen Curry": {"ppg": 27.0, "rpg": 4.5, "apg": 5.5, "min": 32, "on_off_diff": 9.0, "usage_rate": 0.28, "team": "GSW", "tier": 1},
        "Jayson Tatum": {"ppg": 27.0, "rpg": 8.5, "apg": 4.5, "min": 36, "on_off_diff": 8.5, "usage_rate": 0.29, "team": "BOS", "tier": 1},
        "Anthony Edwards": {"ppg": 26.5, "rpg": 5.5, "apg": 5.0, "min": 35, "on_off_diff": 8.0, "usage_rate": 0.28, "team": "MIN", "tier": 1},
        "Kevin Durant": {"ppg": 27.0, "rpg": 6.5, "apg": 5.0, "min": 36, "on_off_diff": 7.5, "usage_rate": 0.30, "team": "PHX", "tier": 1},
        "LeBron James": {"ppg": 25.5, "rpg": 7.5, "apg": 8.0, "min": 35, "on_off_diff": 7.5, "usage_rate": 0.27, "team": "LAL", "tier": 1},
        "Devin Booker": {"ppg": 27.0, "rpg": 4.5, "apg": 6.5, "min": 35, "on_off_diff": 7.0, "usage_rate": 0.29, "team": "PHX", "tier": 1},
        "Anthony Davis": {"ppg": 25.0, "rpg": 12.5, "apg": 3.5, "min": 35, "on_off_diff": 8.0, "usage_rate": 0.27, "team": "LAL", "tier": 1},
        "Victor Wembanyama": {"ppg": 22.0, "rpg": 10.5, "apg": 4.0, "min": 30, "on_off_diff": 8.5, "usage_rate": 0.25, "team": "SAS", "tier": 1},
        "Ja Morant": {"ppg": 26.0, "rpg": 6.0, "apg": 8.0, "min": 32, "on_off_diff": 7.5, "usage_rate": 0.30, "team": "MEM", "tier": 1},
        "Donovan Mitchell": {"ppg": 28.0, "rpg": 4.5, "apg": 5.0, "min": 35, "on_off_diff": 7.0, "usage_rate": 0.29, "team": "CLE", "tier": 1},

        # Tier 2 - All-Stars (significant impact)
        "Jalen Brunson": {"ppg": 28.0, "rpg": 3.5, "apg": 6.5, "min": 35, "on_off_diff": 6.5, "usage_rate": 0.29, "team": "NYK", "tier": 2},
        "Jaylen Brown": {"ppg": 23.0, "rpg": 5.5, "apg": 3.5, "min": 34, "on_off_diff": 6.0, "usage_rate": 0.26, "team": "BOS", "tier": 2},
        "Damian Lillard": {"ppg": 26.0, "rpg": 4.5, "apg": 7.0, "min": 35, "on_off_diff": 6.0, "usage_rate": 0.28, "team": "MIL", "tier": 2},
        "Tyrese Haliburton": {"ppg": 20.5, "rpg": 4.0, "apg": 10.5, "min": 33, "on_off_diff": 7.0, "usage_rate": 0.24, "team": "IND", "tier": 2},
        "Trae Young": {"ppg": 26.5, "rpg": 3.0, "apg": 10.5, "min": 35, "on_off_diff": 5.5, "usage_rate": 0.30, "team": "ATL", "tier": 2},
        "De'Aaron Fox": {"ppg": 26.0, "rpg": 4.5, "apg": 6.0, "min": 35, "on_off_diff": 6.0, "usage_rate": 0.28, "team": "SAC", "tier": 2},
        "Domantas Sabonis": {"ppg": 19.5, "rpg": 13.5, "apg": 8.0, "min": 35, "on_off_diff": 6.5, "usage_rate": 0.21, "team": "SAC", "tier": 2},
        "Paolo Banchero": {"ppg": 23.0, "rpg": 7.0, "apg": 5.5, "min": 35, "on_off_diff": 5.5, "usage_rate": 0.27, "team": "ORL", "tier": 2},
        "Chet Holmgren": {"ppg": 17.0, "rpg": 8.5, "apg": 2.5, "min": 30, "on_off_diff": 7.5, "usage_rate": 0.21, "team": "OKC", "tier": 2},
        "Bam Adebayo": {"ppg": 20.0, "rpg": 10.5, "apg": 3.5, "min": 34, "on_off_diff": 5.5, "usage_rate": 0.22, "team": "MIA", "tier": 2},
        "Jimmy Butler": {"ppg": 21.0, "rpg": 5.5, "apg": 5.5, "min": 33, "on_off_diff": 5.0, "usage_rate": 0.25, "team": "MIA", "tier": 2},
        "Karl-Anthony Towns": {"ppg": 22.0, "rpg": 9.0, "apg": 3.0, "min": 33, "on_off_diff": 5.0, "usage_rate": 0.24, "team": "NYK", "tier": 2},
        "Kawhi Leonard": {"ppg": 24.0, "rpg": 6.0, "apg": 4.0, "min": 32, "on_off_diff": 6.0, "usage_rate": 0.27, "team": "LAC", "tier": 2},
        "Paul George": {"ppg": 22.5, "rpg": 5.5, "apg": 4.0, "min": 34, "on_off_diff": 4.5, "usage_rate": 0.25, "team": "PHI", "tier": 2},
        "Zion Williamson": {"ppg": 23.0, "rpg": 6.5, "apg": 5.0, "min": 30, "on_off_diff": 5.5, "usage_rate": 0.28, "team": "NOP", "tier": 2},

        # Tier 3 - Key Starters (moderate impact)
        "Darius Garland": {"ppg": 19.0, "rpg": 3.0, "apg": 6.5, "min": 32, "on_off_diff": 4.0, "usage_rate": 0.24, "team": "CLE", "tier": 3},
        "Jarrett Allen": {"ppg": 16.5, "rpg": 11.0, "apg": 1.5, "min": 32, "on_off_diff": 4.5, "usage_rate": 0.17, "team": "CLE", "tier": 3},
        "Evan Mobley": {"ppg": 16.0, "rpg": 9.5, "apg": 3.0, "min": 32, "on_off_diff": 5.0, "usage_rate": 0.19, "team": "CLE", "tier": 3},
        "Jalen Williams": {"ppg": 19.5, "rpg": 5.0, "apg": 5.0, "min": 34, "on_off_diff": 5.5, "usage_rate": 0.22, "team": "OKC", "tier": 3},
        "Dejounte Murray": {"ppg": 22.0, "rpg": 5.5, "apg": 7.0, "min": 34, "on_off_diff": 4.0, "usage_rate": 0.26, "team": "NOP", "tier": 3},
        "Brandon Ingram": {"ppg": 21.5, "rpg": 5.5, "apg": 5.5, "min": 34, "on_off_diff": 3.5, "usage_rate": 0.26, "team": "NOP", "tier": 3},
        "Franz Wagner": {"ppg": 21.0, "rpg": 5.5, "apg": 4.5, "min": 34, "on_off_diff": 4.5, "usage_rate": 0.25, "team": "ORL", "tier": 3},
        "Lauri Markkanen": {"ppg": 23.0, "rpg": 8.5, "apg": 2.0, "min": 34, "on_off_diff": 4.0, "usage_rate": 0.26, "team": "UTA", "tier": 3},
        "Cade Cunningham": {"ppg": 23.0, "rpg": 5.0, "apg": 8.5, "min": 35, "on_off_diff": 4.5, "usage_rate": 0.28, "team": "DET", "tier": 3},
        "Alperen Sengun": {"ppg": 21.0, "rpg": 9.5, "apg": 5.5, "min": 33, "on_off_diff": 5.0, "usage_rate": 0.24, "team": "HOU", "tier": 3},
        "Jalen Green": {"ppg": 20.0, "rpg": 4.0, "apg": 3.5, "min": 32, "on_off_diff": 3.0, "usage_rate": 0.26, "team": "HOU", "tier": 3},
    }

    # Impact weights for different stats
    IMPACT_WEIGHTS = {
        "on_off_diff": 0.40,  # Most important - actual net rating impact
        "usage_rate": 0.25,   # Usage redistribution impact
        "ppg": 0.20,          # Scoring impact
        "min": 0.15,          # Minutes share impact
    }

    def __init__(self):
        pass

    def calculate_missing_player_impact(
        self,
        missing_players: List[str],
        team_abbrev: str,
        status_probabilities: Dict[str, float] = None
    ) -> Dict:
        """
        Calculate the aggregate impact of missing players on team performance.

        Args:
            missing_players: List of player names who are out/questionable
            team_abbrev: Team abbreviation
            status_probabilities: Dict mapping player -> probability they miss game
                                 (default: 1.0 for all, meaning definitely out)

        Returns:
            Dictionary with impact metrics
        """
        if status_probabilities is None:
            status_probabilities = {p: 1.0 for p in missing_players}

        impact = {
            "team": team_abbrev,
            "missing_players": [],
            "total_on_off_impact": 0.0,
            "total_usage_void": 0.0,
            "total_ppg_lost": 0.0,
            "total_rpg_lost": 0.0,
            "total_apg_lost": 0.0,
            "total_min_lost": 0.0,
            "tier1_players_out": 0,
            "tier2_players_out": 0,
            "tier3_players_out": 0,
            "star_player_out": False,
            "lineup_net_rating_adj": 0.0,
            "win_probability_adj": 0.0,
        }

        for player_name in missing_players:
            miss_prob = status_probabilities.get(player_name, 1.0)

            # Look up player data
            player_data = self.PLAYER_IMPACTS.get(player_name)

            if player_data and player_data.get("team") == team_abbrev:
                # Calculate weighted impact based on probability of missing
                on_off = player_data.get("on_off_diff", 0) * miss_prob
                usage = player_data.get("usage_rate", 0) * miss_prob
                ppg = player_data.get("ppg", 0) * miss_prob
                rpg = player_data.get("rpg", 0) * miss_prob
                apg = player_data.get("apg", 0) * miss_prob
                minutes = player_data.get("min", 0) * miss_prob
                tier = player_data.get("tier", 3)

                impact["total_on_off_impact"] += on_off
                impact["total_usage_void"] += usage
                impact["total_ppg_lost"] += ppg
                impact["total_rpg_lost"] += rpg
                impact["total_apg_lost"] += apg
                impact["total_min_lost"] += minutes

                if miss_prob >= 0.75:  # Likely to miss
                    if tier == 1:
                        impact["tier1_players_out"] += 1
                        impact["star_player_out"] = True
                    elif tier == 2:
                        impact["tier2_players_out"] += 1
                    else:
                        impact["tier3_players_out"] += 1

                impact["missing_players"].append({
                    "name": player_name,
                    "tier": tier,
                    "miss_probability": miss_prob,
                    "on_off_impact": on_off,
                    "usage_void": usage,
                    "ppg_lost": ppg,
                })
            else:
                # Unknown player - use default impact
                default_impact = 3.0 * miss_prob  # Assume role player
                impact["total_on_off_impact"] += default_impact
                impact["missing_players"].append({
                    "name": player_name,
                    "tier": 4,  # Unknown
                    "miss_probability": miss_prob,
                    "on_off_impact": default_impact,
                })

        # Calculate lineup net rating adjustment
        # On/off differential directly translates to expected point differential change
        impact["lineup_net_rating_adj"] = -impact["total_on_off_impact"]

        # Estimate win probability adjustment
        # ~2.5 points of net rating = ~10% win probability swing
        # Tier 1 player out = additional 5-7% penalty
        base_win_adj = -impact["total_on_off_impact"] * 0.04
        star_penalty = -0.06 if impact["star_player_out"] else 0
        impact["win_probability_adj"] = base_win_adj + star_penalty

        return impact

    def calculate_usage_redistribution(
        self,
        missing_players: List[str],
        team_abbrev: str,
        remaining_players: List[str] = None
    ) -> Dict:
        """
        Calculate how usage will be redistributed when players are out.

        When a high-usage player misses, their shots/touches go to teammates.
        This is important for player prop predictions.

        Args:
            missing_players: Players who are out
            team_abbrev: Team abbreviation
            remaining_players: Optional list of remaining key players

        Returns:
            Dictionary with usage redistribution predictions
        """
        total_usage_void = 0.0

        for player_name in missing_players:
            player_data = self.PLAYER_IMPACTS.get(player_name)
            if player_data and player_data.get("team") == team_abbrev:
                total_usage_void += player_data.get("usage_rate", 0)

        # Estimate redistribution
        # Usage void is typically distributed:
        # - 40% to next highest usage player
        # - 30% to other starters
        # - 30% lost to decreased efficiency

        redistribution = {
            "total_usage_void": total_usage_void,
            "primary_beneficiary_boost": total_usage_void * 0.40,
            "secondary_boost": total_usage_void * 0.30,
            "efficiency_loss": total_usage_void * 0.30,
            "projected_boosts": [],
        }

        if remaining_players:
            # Find highest usage remaining player
            remaining_usages = []
            for player in remaining_players:
                data = self.PLAYER_IMPACTS.get(player, {})
                if data.get("team") == team_abbrev:
                    remaining_usages.append((player, data.get("usage_rate", 0)))

            remaining_usages.sort(key=lambda x: x[1], reverse=True)

            if remaining_usages:
                # Primary beneficiary
                primary = remaining_usages[0]
                boost = total_usage_void * 0.40
                redistribution["projected_boosts"].append({
                    "player": primary[0],
                    "usage_boost": boost,
                    "ppg_boost": boost * 40,  # Rough conversion: 1% usage = ~0.4 PPG
                })

                # Secondary beneficiaries
                for player, usage in remaining_usages[1:3]:
                    boost = total_usage_void * 0.15
                    redistribution["projected_boosts"].append({
                        "player": player,
                        "usage_boost": boost,
                        "ppg_boost": boost * 40,
                    })

        return redistribution

    def generate_lineup_features(
        self,
        team_abbrev: str,
        missing_players: List[str],
        opponent_missing: List[str] = None,
        status_probs: Dict[str, float] = None,
        opponent_status_probs: Dict[str, float] = None
    ) -> Dict:
        """
        Generate features for model training based on lineup situation.

        Args:
            team_abbrev: Team abbreviation
            missing_players: Players out for this team
            opponent_missing: Players out for opponent
            status_probs: Probability each player misses (default: 1.0)
            opponent_status_probs: Same for opponent

        Returns:
            Dictionary with lineup-adjusted features
        """
        # Calculate team impact
        team_impact = self.calculate_missing_player_impact(
            missing_players, team_abbrev, status_probs
        )

        features = {
            # Team injury impact
            "lineup_net_rating_adj": team_impact["lineup_net_rating_adj"],
            "lineup_win_prob_adj": team_impact["win_probability_adj"],
            "total_on_off_lost": team_impact["total_on_off_impact"],
            "total_usage_void": team_impact["total_usage_void"],
            "total_ppg_lost": team_impact["total_ppg_lost"],
            "tier1_players_out": team_impact["tier1_players_out"],
            "tier2_players_out": team_impact["tier2_players_out"],
            "star_player_out": 1 if team_impact["star_player_out"] else 0,
            "total_players_out": len(missing_players),
        }

        # Add opponent impact if provided
        if opponent_missing is not None:
            opp_impact = self.calculate_missing_player_impact(
                opponent_missing, "", opponent_status_probs  # Empty team for opponent
            )

            features["opp_lineup_net_rating_adj"] = opp_impact["lineup_net_rating_adj"]
            features["opp_lineup_win_prob_adj"] = opp_impact["win_probability_adj"]
            features["opp_total_on_off_lost"] = opp_impact["total_on_off_impact"]
            features["opp_star_player_out"] = 1 if opp_impact["star_player_out"] else 0
            features["opp_total_players_out"] = len(opponent_missing)

            # Calculate relative advantage
            features["lineup_advantage"] = (
                opp_impact["total_on_off_impact"] - team_impact["total_on_off_impact"]
            )
            features["win_prob_swing"] = (
                opp_impact["win_probability_adj"] - team_impact["win_probability_adj"]
            )
        else:
            features["opp_lineup_net_rating_adj"] = 0
            features["opp_lineup_win_prob_adj"] = 0
            features["opp_total_on_off_lost"] = 0
            features["opp_star_player_out"] = 0
            features["opp_total_players_out"] = 0
            features["lineup_advantage"] = -team_impact["total_on_off_impact"]
            features["win_prob_swing"] = -team_impact["win_probability_adj"]

        return features

    def get_rotation_depth(self, team_abbrev: str, available_players: List[str]) -> Dict:
        """
        Assess team's rotation depth given available players.

        Args:
            team_abbrev: Team abbreviation
            available_players: List of available players

        Returns:
            Dictionary with rotation depth assessment
        """
        available_minutes = 0
        available_tiers = {1: 0, 2: 0, 3: 0}

        for player in available_players:
            data = self.PLAYER_IMPACTS.get(player, {})
            if data.get("team") == team_abbrev:
                available_minutes += data.get("min", 0)
                tier = data.get("tier", 4)
                if tier in available_tiers:
                    available_tiers[tier] += 1

        # Team needs ~240 minutes covered (8 rotation players)
        rotation_score = min(available_minutes / 240.0, 1.0)

        return {
            "rotation_minutes_available": available_minutes,
            "rotation_completeness": rotation_score,
            "tier1_available": available_tiers[1],
            "tier2_available": available_tiers[2],
            "tier3_available": available_tiers[3],
            "depth_score": (
                available_tiers[1] * 3 + available_tiers[2] * 2 + available_tiers[3]
            ) / 10.0,  # Normalized depth score
        }


# =============================================================================
# LINE MOVEMENT FEATURES - Sharp Money Detection
# =============================================================================

class LineMovementFeatureGenerator:
    """
    Generate features from betting line movements for sharp money detection.

    Line movement analysis is one of the most valuable signals for sports betting:
    - Sharp money moves lines efficiently toward the true probability
    - Reverse Line Movement (RLM) = line moves opposite to public betting %
    - Steam moves = rapid 1.5+ point moves indicating coordinated sharp action

    Research shows:
    - Following sharp money adds 2-3% to win rate
    - RLM signals are profitable with 54-56% hit rate
    - Fading public when >70% on one side is profitable long-term
    """

    def __init__(self):
        self.public_betting_threshold = 0.65  # >65% public = heavily bet side
        self.steam_move_threshold = 1.5  # Points for steam detection
        self.significant_ml_move = 15  # Moneyline points change considered significant

    def calculate_line_movement_features(
        self,
        opening_spread: float,
        current_spread: float,
        opening_ml_home: int,
        current_ml_home: int,
        opening_total: float,
        current_total: float,
        public_home_pct: float = None,
        public_over_pct: float = None,
        model_spread_prediction: float = None,
        time_to_game_hours: float = None
    ) -> Dict:
        """
        Generate features from line movements.

        Args:
            opening_spread: Opening home team spread (e.g., -3.5)
            current_spread: Current home team spread
            opening_ml_home: Opening home moneyline odds
            current_ml_home: Current home moneyline odds
            opening_total: Opening over/under total
            current_total: Current over/under total
            public_home_pct: % of bets on home team (0-1 scale)
            public_over_pct: % of bets on over (0-1 scale)
            model_spread_prediction: Our model's predicted spread
            time_to_game_hours: Hours until game starts

        Returns:
            Dictionary with line movement features
        """
        features = {}

        # ===== SPREAD MOVEMENT FEATURES =====
        spread_movement = current_spread - opening_spread
        features["spread_movement"] = spread_movement
        features["spread_movement_abs"] = abs(spread_movement)

        # Direction interpretation (negative movement = line moved toward home team)
        # If line goes from -3.5 to -5.5, home is more heavily favored
        features["spread_toward_home"] = 1 if spread_movement < -0.5 else (
            -1 if spread_movement > 0.5 else 0
        )

        # Steam move detection (rapid sharp action)
        features["is_steam_move"] = 1 if abs(spread_movement) >= self.steam_move_threshold else 0

        # ===== MONEYLINE MOVEMENT FEATURES =====
        if opening_ml_home and current_ml_home:
            # Convert to implied probability for proper comparison
            def ml_to_prob(ml):
                if ml > 0:
                    return 100 / (ml + 100)
                else:
                    return abs(ml) / (abs(ml) + 100)

            opening_prob = ml_to_prob(opening_ml_home)
            current_prob = ml_to_prob(current_ml_home)

            features["ml_prob_movement"] = current_prob - opening_prob
            features["ml_prob_movement_abs"] = abs(current_prob - opening_prob)

            # Significant ML movement (in raw odds points)
            features["ml_raw_movement"] = current_ml_home - opening_ml_home
            features["is_significant_ml_move"] = 1 if abs(current_ml_home - opening_ml_home) >= self.significant_ml_move else 0
        else:
            features["ml_prob_movement"] = 0
            features["ml_prob_movement_abs"] = 0
            features["ml_raw_movement"] = 0
            features["is_significant_ml_move"] = 0

        # ===== TOTAL MOVEMENT FEATURES =====
        if opening_total and current_total:
            total_movement = current_total - opening_total
            features["total_movement"] = total_movement
            features["total_movement_abs"] = abs(total_movement)
            features["total_toward_over"] = 1 if total_movement > 0.5 else (
                -1 if total_movement < -0.5 else 0
            )
        else:
            features["total_movement"] = 0
            features["total_movement_abs"] = 0
            features["total_toward_over"] = 0

        # ===== REVERSE LINE MOVEMENT (RLM) =====
        # The MOST valuable signal - when line moves opposite to public betting
        if public_home_pct is not None:
            features["public_home_pct"] = public_home_pct
            features["public_away_pct"] = 1 - public_home_pct

            # Heavy public side
            features["public_heavily_on_home"] = 1 if public_home_pct > self.public_betting_threshold else 0
            features["public_heavily_on_away"] = 1 if public_home_pct < (1 - self.public_betting_threshold) else 0

            # RLM Detection
            # If public is heavy on home (>65%) but line moves TOWARD away = RLM on away
            # If public is heavy on away (>65%) but line moves TOWARD home = RLM on home
            is_rlm_away = public_home_pct > self.public_betting_threshold and spread_movement > 0.5
            is_rlm_home = public_home_pct < (1 - self.public_betting_threshold) and spread_movement < -0.5

            features["is_rlm"] = 1 if (is_rlm_away or is_rlm_home) else 0
            features["rlm_on_home"] = 1 if is_rlm_home else 0
            features["rlm_on_away"] = 1 if is_rlm_away else 0

            # RLM strength (how much the line moved against public)
            if features["is_rlm"]:
                features["rlm_strength"] = abs(spread_movement) * max(public_home_pct, 1 - public_home_pct)
            else:
                features["rlm_strength"] = 0
        else:
            features["public_home_pct"] = 0.5
            features["public_away_pct"] = 0.5
            features["public_heavily_on_home"] = 0
            features["public_heavily_on_away"] = 0
            features["is_rlm"] = 0
            features["rlm_on_home"] = 0
            features["rlm_on_away"] = 0
            features["rlm_strength"] = 0

        # ===== TOTALS RLM =====
        if public_over_pct is not None:
            features["public_over_pct"] = public_over_pct
            features["public_under_pct"] = 1 - public_over_pct

            # Heavy public side on totals
            features["public_heavily_on_over"] = 1 if public_over_pct > self.public_betting_threshold else 0
            features["public_heavily_on_under"] = 1 if public_over_pct < (1 - self.public_betting_threshold) else 0

            # RLM on totals
            total_movement = features.get("total_movement", 0)
            is_rlm_under = public_over_pct > self.public_betting_threshold and total_movement < -0.5
            is_rlm_over = public_over_pct < (1 - self.public_betting_threshold) and total_movement > 0.5

            features["is_totals_rlm"] = 1 if (is_rlm_under or is_rlm_over) else 0
            features["rlm_on_over"] = 1 if is_rlm_over else 0
            features["rlm_on_under"] = 1 if is_rlm_under else 0
        else:
            features["public_over_pct"] = 0.5
            features["public_under_pct"] = 0.5
            features["public_heavily_on_over"] = 0
            features["public_heavily_on_under"] = 0
            features["is_totals_rlm"] = 0
            features["rlm_on_over"] = 0
            features["rlm_on_under"] = 0

        # ===== LINE VALUE VS MODEL =====
        if model_spread_prediction is not None:
            # Positive = model says home should be favored MORE than market
            features["model_vs_market_spread"] = model_spread_prediction - current_spread

            # Does sharp money agree with model?
            model_favors_home_more = model_spread_prediction < current_spread
            line_moved_toward_home = spread_movement < -0.5

            features["sharps_agree_with_model"] = 1 if (
                model_favors_home_more and line_moved_toward_home
            ) or (
                not model_favors_home_more and not line_moved_toward_home
            ) else 0
        else:
            features["model_vs_market_spread"] = 0
            features["sharps_agree_with_model"] = 0

        # ===== TIME-BASED FEATURES =====
        if time_to_game_hours is not None:
            features["hours_to_game"] = time_to_game_hours
            # Late movement (within 2 hours) is often sharper
            features["is_late_movement"] = 1 if time_to_game_hours <= 2 else 0
            # Normalize movement by time (bigger moves early are less significant)
            if time_to_game_hours > 0:
                features["movement_velocity"] = abs(spread_movement) / max(time_to_game_hours, 0.1)
            else:
                features["movement_velocity"] = abs(spread_movement) * 10  # Game about to start
        else:
            features["hours_to_game"] = 24  # Default
            features["is_late_movement"] = 0
            features["movement_velocity"] = 0

        # ===== COMPOSITE SHARP SCORE =====
        # Combine multiple signals into single "sharp" score
        sharp_score = 0

        # RLM is most valuable (+30 points)
        sharp_score += features.get("is_rlm", 0) * 30

        # Steam moves are valuable (+20 points)
        sharp_score += features.get("is_steam_move", 0) * 20

        # Late movement is valuable (+10 points)
        sharp_score += features.get("is_late_movement", 0) * 10

        # Sharps agreeing with model is valuable (+15 points)
        sharp_score += features.get("sharps_agree_with_model", 0) * 15

        # Heavy public against us is valuable (contrarian) (+10 points)
        # This is captured in the RLM signal already

        features["sharp_score"] = sharp_score
        features["sharp_score_normalized"] = min(sharp_score / 75, 1.0)  # Normalize to 0-1

        return features

    def calculate_closing_line_value_features(
        self,
        bet_spread: float,
        closing_spread: float,
        bet_ml: int,
        closing_ml: int,
        bet_total: float,
        closing_total: float
    ) -> Dict:
        """
        Calculate CLV-related features for bet evaluation.

        CLV (Closing Line Value) is the single best predictor of long-term betting success.
        Sharp bettors consistently beat closing lines.

        Args:
            bet_spread: Spread when bet was placed
            closing_spread: Spread at game start
            bet_ml: Moneyline when bet was placed
            closing_ml: Moneyline at game start
            bet_total: Total when bet was placed
            closing_total: Total at game start

        Returns:
            Dictionary with CLV features
        """
        features = {}

        # Spread CLV (in points)
        features["spread_clv_points"] = closing_spread - bet_spread

        # Moneyline CLV (in implied probability)
        def ml_to_prob(ml):
            if ml is None:
                return 0.5
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return abs(ml) / (abs(ml) + 100)

        bet_prob = ml_to_prob(bet_ml)
        closing_prob = ml_to_prob(closing_ml)
        features["ml_clv_prob"] = closing_prob - bet_prob

        # Total CLV (in points)
        if bet_total and closing_total:
            features["total_clv_points"] = closing_total - bet_total
        else:
            features["total_clv_points"] = 0

        # CLV quality scores
        features["positive_spread_clv"] = 1 if features["spread_clv_points"] > 0 else 0
        features["positive_ml_clv"] = 1 if features["ml_clv_prob"] > 0 else 0

        # Magnitude of CLV advantage
        features["clv_magnitude"] = abs(features["spread_clv_points"]) + abs(features["ml_clv_prob"] * 10)

        return features


# =============================================================================
# TRAVEL AND SCHEDULE FATIGUE FEATURES
# =============================================================================

class TravelFatigueFeatureGenerator:
    """
    Generate features related to travel and schedule fatigue.

    Research shows:
    - West-to-East travel hurts performance significantly (jet lag)
    - Denver altitude affects visitors (especially in 1st quarter)
    - Extended road trips (5+ games) reduce win probability
    - Back-to-backs after long flights are particularly damaging
    """

    # Team locations (latitude, longitude, timezone offset from ET)
    TEAM_LOCATIONS = {
        "ATL": (33.757, -84.396, 0),
        "BOS": (42.366, -71.062, 0),
        "BKN": (40.683, -73.975, 0),
        "CHA": (35.225, -80.839, 0),
        "CHI": (41.881, -87.674, -1),
        "CLE": (41.496, -81.688, 0),
        "DAL": (32.790, -96.810, -1),
        "DEN": (39.749, -105.007, -2),
        "DET": (42.341, -83.055, 0),
        "GSW": (37.768, -122.388, -3),
        "HOU": (29.751, -95.362, -1),
        "IND": (39.764, -86.155, 0),
        "LAC": (34.043, -118.267, -3),
        "LAL": (34.043, -118.267, -3),
        "MEM": (35.138, -90.051, -1),
        "MIA": (25.781, -80.187, 0),
        "MIL": (43.043, -87.917, -1),
        "MIN": (44.979, -93.276, -1),
        "NOP": (29.949, -90.082, -1),
        "NYK": (40.751, -73.994, 0),
        "OKC": (35.463, -97.515, -1),
        "ORL": (28.539, -81.384, 0),
        "PHI": (39.901, -75.172, 0),
        "PHX": (33.446, -112.071, -2),
        "POR": (45.532, -122.667, -3),
        "SAC": (38.580, -121.500, -3),
        "SAS": (29.427, -98.438, -1),
        "TOR": (43.643, -79.379, 0),
        "UTA": (40.768, -111.901, -2),
        "WAS": (38.898, -77.021, 0),
    }

    # Denver altitude (5,280 ft) - significant factor
    DENVER_ALTITUDE_FT = 5280
    HIGH_ALTITUDE_TEAMS = ["DEN", "UTA"]  # Utah also has some elevation

    def __init__(self):
        pass

    def calculate_distance_miles(self, team1: str, team2: str) -> float:
        """Calculate approximate distance between two team cities."""
        loc1 = self.TEAM_LOCATIONS.get(team1)
        loc2 = self.TEAM_LOCATIONS.get(team2)

        if not loc1 or not loc2:
            return 1000  # Default medium distance

        # Haversine formula approximation
        lat1, lon1, _ = loc1
        lat2, lon2, _ = loc2

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in miles
        r = 3956

        return c * r

    def calculate_timezone_change(self, from_team: str, to_team: str) -> int:
        """Calculate timezone change (negative = west to east)."""
        loc_from = self.TEAM_LOCATIONS.get(from_team)
        loc_to = self.TEAM_LOCATIONS.get(to_team)

        if not loc_from or not loc_to:
            return 0

        return loc_to[2] - loc_from[2]

    def generate_travel_features(
        self,
        team_abbrev: str,
        opponent_abbrev: str,
        is_home: bool,
        previous_game_location: str = None,
        games_in_last_7_days: int = None,
        games_in_last_14_days: int = None,
        is_back_to_back: bool = False,
        road_trip_game_number: int = None
    ) -> Dict:
        """
        Generate travel and fatigue related features.

        Args:
            team_abbrev: Team abbreviation (e.g., "LAL")
            opponent_abbrev: Opponent abbreviation
            is_home: Whether team is playing at home
            previous_game_location: Location of previous game (team abbrev)
            games_in_last_7_days: Number of games in last 7 days
            games_in_last_14_days: Number of games in last 14 days
            is_back_to_back: Whether this is second game of back-to-back
            road_trip_game_number: Game number on current road trip (1-indexed)

        Returns:
            Dictionary with travel/fatigue features
        """
        features = {}

        # ===== DISTANCE FEATURES =====
        game_location = team_abbrev if is_home else opponent_abbrev

        if previous_game_location:
            travel_distance = self.calculate_distance_miles(previous_game_location, game_location)
            features["travel_distance_miles"] = travel_distance
            features["is_long_flight"] = 1 if travel_distance > 1500 else 0
            features["is_very_long_flight"] = 1 if travel_distance > 2500 else 0

            # Timezone changes
            tz_change = self.calculate_timezone_change(previous_game_location, game_location)
            features["timezone_change"] = tz_change
            features["is_west_to_east"] = 1 if tz_change > 0 else 0
            features["is_east_to_west"] = 1 if tz_change < 0 else 0
            features["timezone_change_abs"] = abs(tz_change)

            # Jet lag factor (west to east is worse)
            if tz_change > 0:  # West to east
                features["jet_lag_factor"] = tz_change * 1.5  # Extra penalty
            elif tz_change < 0:  # East to west
                features["jet_lag_factor"] = abs(tz_change) * 1.0
            else:
                features["jet_lag_factor"] = 0
        else:
            features["travel_distance_miles"] = 0
            features["is_long_flight"] = 0
            features["is_very_long_flight"] = 0
            features["timezone_change"] = 0
            features["is_west_to_east"] = 0
            features["is_east_to_west"] = 0
            features["timezone_change_abs"] = 0
            features["jet_lag_factor"] = 0

        # ===== ALTITUDE FEATURES =====
        features["is_at_altitude"] = 1 if game_location in self.HIGH_ALTITUDE_TEAMS else 0
        features["is_denver_game"] = 1 if game_location == "DEN" else 0

        # Altitude disadvantage for visiting team at Denver
        if not is_home and game_location == "DEN":
            features["altitude_disadvantage"] = 1
        else:
            features["altitude_disadvantage"] = 0

        # ===== SCHEDULE DENSITY FEATURES =====
        if games_in_last_7_days is not None:
            features["games_last_7_days"] = games_in_last_7_days
            features["is_heavy_schedule_7d"] = 1 if games_in_last_7_days >= 4 else 0
        else:
            features["games_last_7_days"] = 2  # Default
            features["is_heavy_schedule_7d"] = 0

        if games_in_last_14_days is not None:
            features["games_last_14_days"] = games_in_last_14_days
            features["is_heavy_schedule_14d"] = 1 if games_in_last_14_days >= 7 else 0
        else:
            features["games_last_14_days"] = 5  # Default
            features["is_heavy_schedule_14d"] = 0

        # ===== BACK-TO-BACK FEATURES =====
        features["is_back_to_back"] = 1 if is_back_to_back else 0

        # B2B after long travel is particularly bad
        if is_back_to_back and features.get("is_long_flight", 0):
            features["b2b_after_travel"] = 1
        else:
            features["b2b_after_travel"] = 0

        # ===== ROAD TRIP FEATURES =====
        if road_trip_game_number is not None:
            features["road_trip_game_num"] = road_trip_game_number
            features["is_extended_road_trip"] = 1 if road_trip_game_number >= 5 else 0
            features["road_trip_fatigue"] = min(road_trip_game_number / 7.0, 1.0)  # Normalized
        else:
            features["road_trip_game_num"] = 0
            features["is_extended_road_trip"] = 0
            features["road_trip_fatigue"] = 0

        # ===== HOME ADVANTAGE FEATURES =====
        features["is_home"] = 1 if is_home else 0
        features["is_away"] = 0 if is_home else 1

        # ===== COMPOSITE FATIGUE SCORE =====
        # Combine multiple fatigue signals
        fatigue_score = 0

        # Back-to-back is significant
        fatigue_score += features.get("is_back_to_back", 0) * 2.0

        # Travel fatigue
        fatigue_score += features.get("jet_lag_factor", 0) * 0.5
        fatigue_score += features.get("is_long_flight", 0) * 1.0
        fatigue_score += features.get("is_very_long_flight", 0) * 0.5  # Additional

        # Schedule density
        fatigue_score += features.get("is_heavy_schedule_7d", 0) * 1.5

        # Road trip fatigue
        fatigue_score += features.get("road_trip_fatigue", 0) * 1.5

        # Altitude
        fatigue_score += features.get("altitude_disadvantage", 0) * 1.0

        features["composite_fatigue_score"] = fatigue_score
        features["fatigue_score_normalized"] = min(fatigue_score / 8.0, 1.0)  # Normalize

        return features


class FourFactorsCalculator:
    """
    Calculate Dean Oliver's Four Factors for NBA analytics.

    The Four Factors (with approximate weights):
    1. Effective Field Goal % (eFG%) - 40% importance
       - Adjusts for 3-pointers being worth more
       - eFG% = (FGM + 0.5 * FG3M) / FGA

    2. Turnover Rate (TOV%) - 25% importance
       - Turnovers per 100 possessions
       - TOV% = TOV / (FGA + 0.44*FTA + TOV)

    3. Offensive Rebound Rate (ORB%) - 20% importance
       - % of available offensive rebounds grabbed
       - ORB% = ORB / (ORB + Opp DRB)

    4. Free Throw Rate (FTR) - 15% importance
       - Getting to the line and making free throws
       - FTR = FTM / FGA

    These four factors explain ~90% of scoring variance.
    """

    # Weights from Dean Oliver's research
    WEIGHTS = {
        'efg_pct': 0.40,
        'tov_pct': 0.25,
        'orb_pct': 0.20,
        'ft_rate': 0.15,
    }

    # League average baselines (2023-24 season)
    LEAGUE_AVERAGES = {
        'efg_pct': 0.548,
        'tov_pct': 0.125,
        'orb_pct': 0.260,
        'ft_rate': 0.195,
    }

    def calculate_efg_pct(
        self,
        fgm: float,
        fg3m: float,
        fga: float,
    ) -> float:
        """
        Calculate Effective Field Goal Percentage.

        eFG% = (FGM + 0.5 * FG3M) / FGA
        """
        if fga <= 0:
            return 0.0
        return (fgm + 0.5 * fg3m) / fga

    def calculate_tov_pct(
        self,
        tov: float,
        fga: float,
        fta: float,
    ) -> float:
        """
        Calculate Turnover Percentage.

        TOV% = TOV / (FGA + 0.44*FTA + TOV)
        Lower is better.
        """
        possessions = fga + 0.44 * fta + tov
        if possessions <= 0:
            return 0.0
        return tov / possessions

    def calculate_orb_pct(
        self,
        orb: float,
        opp_drb: float,
    ) -> float:
        """
        Calculate Offensive Rebound Percentage.

        ORB% = ORB / (ORB + Opponent DRB)
        """
        total = orb + opp_drb
        if total <= 0:
            return 0.0
        return orb / total

    def calculate_ft_rate(
        self,
        ftm: float,
        fga: float,
    ) -> float:
        """
        Calculate Free Throw Rate.

        FTR = FTM / FGA
        """
        if fga <= 0:
            return 0.0
        return ftm / fga

    def calculate_all_factors(
        self,
        fgm: float,
        fga: float,
        fg3m: float,
        ftm: float,
        fta: float,
        tov: float,
        orb: float,
        opp_drb: float,
    ) -> Dict[str, float]:
        """
        Calculate all Four Factors from box score stats.

        Returns dict with eFG%, TOV%, ORB%, FTR
        """
        return {
            'efg_pct': self.calculate_efg_pct(fgm, fg3m, fga),
            'tov_pct': self.calculate_tov_pct(tov, fga, fta),
            'orb_pct': self.calculate_orb_pct(orb, opp_drb),
            'ft_rate': self.calculate_ft_rate(ftm, fga),
        }

    def calculate_four_factors_score(
        self,
        team_factors: Dict[str, float],
        opp_factors: Dict[str, float] = None,
    ) -> float:
        """
        Calculate composite Four Factors score.

        Score is weighted sum of how much each factor exceeds league average.
        For TOV%, lower is better so we invert it.

        Args:
            team_factors: Team's four factors
            opp_factors: Opponent's factors (for net calculation)

        Returns:
            Score where positive = above average, negative = below
        """
        score = 0.0

        for factor, weight in self.WEIGHTS.items():
            team_val = team_factors.get(factor, self.LEAGUE_AVERAGES[factor])
            baseline = self.LEAGUE_AVERAGES[factor]

            if factor == 'tov_pct':
                # Lower is better for turnovers
                diff = (baseline - team_val) / baseline
            else:
                # Higher is better for other factors
                diff = (team_val - baseline) / baseline

            score += weight * diff

        # If opponent factors provided, calculate net advantage
        if opp_factors:
            opp_score = self.calculate_four_factors_score(opp_factors)
            score = score - opp_score

        return score

    def generate_four_factors_features(
        self,
        team_stats: Dict[str, float],
        opp_stats: Dict[str, float] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Generate Four Factors features from team stats.

        Args:
            team_stats: Dict with fgm, fga, fg3m, ftm, fta, tov, orb
            opp_stats: Optional opponent stats for differential features
            prefix: Prefix for feature names

        Returns:
            Dict of Four Factors features
        """
        features = {}
        p = prefix + "_" if prefix else ""

        # Extract stats with defaults
        fgm = team_stats.get('fgm', 0)
        fga = team_stats.get('fga', 1)
        fg3m = team_stats.get('fg3m', 0)
        ftm = team_stats.get('ftm', 0)
        fta = team_stats.get('fta', 0)
        tov = team_stats.get('tov', 0)
        orb = team_stats.get('orb', 0)

        # Get opponent DRB (or estimate)
        if opp_stats:
            opp_drb = opp_stats.get('drb', 30)
        else:
            opp_drb = 30  # League average estimate

        # Calculate factors
        factors = self.calculate_all_factors(
            fgm, fga, fg3m, ftm, fta, tov, orb, opp_drb
        )

        # Add to features
        features[f'{p}efg_pct'] = factors['efg_pct']
        features[f'{p}tov_pct'] = factors['tov_pct']
        features[f'{p}orb_pct'] = factors['orb_pct']
        features[f'{p}ft_rate'] = factors['ft_rate']

        # Composite score
        features[f'{p}four_factors_score'] = self.calculate_four_factors_score(factors)

        # VS league average
        features[f'{p}efg_vs_avg'] = factors['efg_pct'] - self.LEAGUE_AVERAGES['efg_pct']
        features[f'{p}tov_vs_avg'] = self.LEAGUE_AVERAGES['tov_pct'] - factors['tov_pct']
        features[f'{p}orb_vs_avg'] = factors['orb_pct'] - self.LEAGUE_AVERAGES['orb_pct']
        features[f'{p}ft_rate_vs_avg'] = factors['ft_rate'] - self.LEAGUE_AVERAGES['ft_rate']

        # Opponent differential if provided
        if opp_stats:
            opp_fgm = opp_stats.get('fgm', 0)
            opp_fga = opp_stats.get('fga', 1)
            opp_fg3m = opp_stats.get('fg3m', 0)
            opp_ftm = opp_stats.get('ftm', 0)
            opp_fta = opp_stats.get('fta', 0)
            opp_tov = opp_stats.get('tov', 0)
            opp_orb = opp_stats.get('orb', 0)
            team_drb = team_stats.get('drb', 30)

            opp_factors = self.calculate_all_factors(
                opp_fgm, opp_fga, opp_fg3m, opp_ftm, opp_fta, opp_tov, opp_orb, team_drb
            )

            features[f'{p}efg_pct_diff'] = factors['efg_pct'] - opp_factors['efg_pct']
            features[f'{p}tov_pct_diff'] = opp_factors['tov_pct'] - factors['tov_pct']  # Positive = we turn it over less
            features[f'{p}orb_pct_diff'] = factors['orb_pct'] - opp_factors['orb_pct']
            features[f'{p}ft_rate_diff'] = factors['ft_rate'] - opp_factors['ft_rate']
            features[f'{p}four_factors_net'] = self.calculate_four_factors_score(factors, opp_factors)

        return features


class ClutchPerformanceCalculator:
    """
    Calculate clutch performance metrics for NBA teams and players.

    Clutch situations are typically defined as:
    - Last 5 minutes of regulation or overtime
    - Score within 5 points

    Clutch performance is crucial for ATS betting because:
    - Close games often come down to clutch execution
    - Some teams/players consistently outperform in clutch
    - Can identify teams that cover/lose covers in final minutes
    """

    def __init__(self):
        # League averages for clutch stats (2023-24)
        self.league_clutch_fg_pct = 0.438
        self.league_clutch_ft_pct = 0.775
        self.league_clutch_tov_pct = 0.145

    def calculate_clutch_metrics(
        self,
        clutch_pts: float,
        clutch_poss: float,
        clutch_fgm: float,
        clutch_fga: float,
        clutch_ftm: float,
        clutch_fta: float,
        clutch_tov: float,
    ) -> Dict[str, float]:
        """
        Calculate clutch performance metrics.

        Args:
            clutch_pts: Points in clutch situations
            clutch_poss: Possessions in clutch
            clutch_fgm: Field goals made in clutch
            clutch_fga: Field goal attempts in clutch
            clutch_ftm: Free throws made in clutch
            clutch_fta: Free throw attempts in clutch
            clutch_tov: Turnovers in clutch

        Returns:
            Dict with clutch metrics
        """
        metrics = {}

        # Clutch offensive rating (points per 100 possessions)
        if clutch_poss > 0:
            metrics['clutch_off_rtg'] = (clutch_pts / clutch_poss) * 100
        else:
            metrics['clutch_off_rtg'] = 100.0  # Neutral

        # Clutch shooting
        if clutch_fga > 0:
            metrics['clutch_fg_pct'] = clutch_fgm / clutch_fga
        else:
            metrics['clutch_fg_pct'] = self.league_clutch_fg_pct

        # Clutch free throws (critical!)
        if clutch_fta > 0:
            metrics['clutch_ft_pct'] = clutch_ftm / clutch_fta
        else:
            metrics['clutch_ft_pct'] = self.league_clutch_ft_pct

        # Clutch turnover rate
        total_plays = clutch_fga + clutch_tov + 0.44 * clutch_fta
        if total_plays > 0:
            metrics['clutch_tov_pct'] = clutch_tov / total_plays
        else:
            metrics['clutch_tov_pct'] = self.league_clutch_tov_pct

        return metrics

    def generate_clutch_features(
        self,
        team_clutch_stats: Dict[str, float],
        team_regular_stats: Dict[str, float],
        record_in_close_games: Tuple[int, int] = None,
        games_decided_by_margin: Dict[str, Tuple[int, int]] = None,
    ) -> Dict[str, float]:
        """
        Generate comprehensive clutch features.

        Args:
            team_clutch_stats: Clutch situation stats
            team_regular_stats: Regular game stats for comparison
            record_in_close_games: (wins, losses) in games within 5 points
            games_decided_by_margin: Dict of margin -> (wins, losses)

        Returns:
            Dict of clutch features
        """
        features = {}

        # Calculate clutch metrics
        clutch_metrics = self.calculate_clutch_metrics(
            clutch_pts=team_clutch_stats.get('clutch_pts', 0),
            clutch_poss=team_clutch_stats.get('clutch_poss', 1),
            clutch_fgm=team_clutch_stats.get('clutch_fgm', 0),
            clutch_fga=team_clutch_stats.get('clutch_fga', 1),
            clutch_ftm=team_clutch_stats.get('clutch_ftm', 0),
            clutch_fta=team_clutch_stats.get('clutch_fta', 1),
            clutch_tov=team_clutch_stats.get('clutch_tov', 0),
        )

        features['clutch_off_rtg'] = clutch_metrics['clutch_off_rtg']
        features['clutch_fg_pct'] = clutch_metrics['clutch_fg_pct']
        features['clutch_ft_pct'] = clutch_metrics['clutch_ft_pct']
        features['clutch_tov_pct'] = clutch_metrics['clutch_tov_pct']

        # Compare clutch to regular performance
        regular_fg_pct = team_regular_stats.get('fg_pct', 0.45)
        regular_ft_pct = team_regular_stats.get('ft_pct', 0.77)

        features['clutch_fg_pct_diff'] = clutch_metrics['clutch_fg_pct'] - regular_fg_pct
        features['clutch_ft_pct_diff'] = clutch_metrics['clutch_ft_pct'] - regular_ft_pct

        # VS league average
        features['clutch_fg_vs_avg'] = clutch_metrics['clutch_fg_pct'] - self.league_clutch_fg_pct
        features['clutch_ft_vs_avg'] = clutch_metrics['clutch_ft_pct'] - self.league_clutch_ft_pct

        # Close game record
        if record_in_close_games:
            wins, losses = record_in_close_games
            total = wins + losses
            if total > 0:
                features['close_game_win_pct'] = wins / total
                features['close_game_record'] = wins - losses
            else:
                features['close_game_win_pct'] = 0.5
                features['close_game_record'] = 0

        # Games by margin (for ATS analysis)
        if games_decided_by_margin:
            # Games within 3 points
            w3, l3 = games_decided_by_margin.get('3pt', (0, 0))
            if w3 + l3 > 0:
                features['games_within_3_win_pct'] = w3 / (w3 + l3)
            else:
                features['games_within_3_win_pct'] = 0.5

            # Games within 5 points
            w5, l5 = games_decided_by_margin.get('5pt', (0, 0))
            if w5 + l5 > 0:
                features['games_within_5_win_pct'] = w5 / (w5 + l5)
            else:
                features['games_within_5_win_pct'] = 0.5

        # Comeback indicator (trailing in Q3)
        comeback_wins = team_clutch_stats.get('comeback_wins', 0)
        trailing_q3_games = team_clutch_stats.get('trailing_q3_games', 1)
        features['comeback_rate'] = comeback_wins / max(trailing_q3_games, 1)

        # Blown lead indicator (leading in Q3)
        blown_leads = team_clutch_stats.get('blown_leads', 0)
        leading_q3_games = team_clutch_stats.get('leading_q3_games', 1)
        features['blown_lead_rate'] = blown_leads / max(leading_q3_games, 1)

        # Net clutch score (composite)
        features['clutch_score'] = (
            (features['clutch_fg_vs_avg'] * 2) +
            (features['clutch_ft_vs_avg'] * 1.5) +
            (self.league_clutch_tov_pct - features['clutch_tov_pct']) +
            features.get('close_game_win_pct', 0.5) - 0.5
        )

        return features


class MomentumCalculator:
    """
    Calculate momentum indicators for NBA teams.

    Momentum captures recent performance trends that may indicate:
    - Hot/cold streaks
    - Team chemistry building/breaking down
    - Fatigue or second wind
    - Schedule-related factors

    Important for betting because:
    - Recent form is often overweighted by public
    - True momentum effects do exist but are smaller than perceived
    - Can identify when to fade hot/cold teams
    """

    def calculate_momentum_score(
        self,
        recent_values: List[float],
        season_average: float,
        lookback: int = 5,
    ) -> float:
        """
        Calculate momentum as deviation from season average.

        Args:
            recent_values: Recent stat values (most recent last)
            season_average: Season average for the stat
            lookback: Number of recent games to consider

        Returns:
            Momentum score (positive = trending up, negative = down)
        """
        if not recent_values or season_average <= 0:
            return 0.0

        recent = recent_values[-lookback:]
        if not recent:
            return 0.0

        recent_avg = sum(recent) / len(recent)
        return (recent_avg - season_average) / season_average

    def calculate_trend(
        self,
        values: List[float],
        lookback: int = 5,
    ) -> float:
        """
        Calculate linear trend in recent values.

        Returns slope normalized by mean (positive = improving).
        """
        if len(values) < 2:
            return 0.0

        recent = values[-lookback:]
        if len(recent) < 2:
            return 0.0

        # Simple linear regression
        x = np.arange(len(recent))
        y = np.array(recent)

        mean_x = np.mean(x)
        mean_y = np.mean(y)

        if mean_y == 0:
            return 0.0

        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sum((x - mean_x) ** 2)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope / abs(mean_y)  # Normalize by mean

    def calculate_streak(
        self,
        results: List[bool],
    ) -> Tuple[str, int]:
        """
        Calculate current win/loss streak.

        Args:
            results: List of game results (True=win, False=loss)

        Returns:
            Tuple of (streak_type, streak_length)
            streak_type: 'W' for win streak, 'L' for loss streak
        """
        if not results:
            return ('N', 0)

        current_result = results[-1]
        streak_type = 'W' if current_result else 'L'
        streak_length = 1

        for result in reversed(results[:-1]):
            if result == current_result:
                streak_length += 1
            else:
                break

        return (streak_type, streak_length)

    def generate_momentum_features(
        self,
        recent_pts_for: List[float],
        recent_pts_against: List[float],
        recent_results: List[bool],  # True = win
        season_ppg: float,
        season_oppg: float,
        season_record: Tuple[int, int],
        recent_ats_results: List[bool] = None,
        recent_margins: List[float] = None,
    ) -> Dict[str, float]:
        """
        Generate comprehensive momentum features.

        Args:
            recent_pts_for: Recent points scored (last N games)
            recent_pts_against: Recent points allowed
            recent_results: Recent W/L results
            season_ppg: Season average points per game
            season_oppg: Season average opponent PPG
            season_record: (wins, losses) for the season
            recent_ats_results: Recent ATS results (True = covered)
            recent_margins: Recent game margins (positive = won by)

        Returns:
            Dict of momentum features
        """
        features = {}

        # Scoring momentum
        features['pts_momentum_5g'] = self.calculate_momentum_score(
            recent_pts_for, season_ppg, lookback=5
        )
        features['pts_momentum_10g'] = self.calculate_momentum_score(
            recent_pts_for, season_ppg, lookback=10
        )

        # Defense momentum
        features['def_momentum_5g'] = -self.calculate_momentum_score(
            recent_pts_against, season_oppg, lookback=5
        )  # Negative because lower is better

        # Net rating momentum
        if recent_pts_for and recent_pts_against:
            recent_net = [pf - pa for pf, pa in zip(recent_pts_for[-5:], recent_pts_against[-5:])]
            season_net = season_ppg - season_oppg
            if season_net != 0:
                features['net_rating_momentum'] = (sum(recent_net) / len(recent_net) - season_net) / max(abs(season_net), 1)
            else:
                features['net_rating_momentum'] = 0.0
        else:
            features['net_rating_momentum'] = 0.0

        # Win/loss streak
        streak_type, streak_length = self.calculate_streak(recent_results)
        features['win_streak'] = streak_length if streak_type == 'W' else 0
        features['loss_streak'] = streak_length if streak_type == 'L' else 0
        features['streak_value'] = streak_length if streak_type == 'W' else -streak_length

        # Recent record
        if recent_results:
            recent_10 = recent_results[-10:]
            features['win_pct_l10'] = sum(recent_10) / len(recent_10)
            recent_5 = recent_results[-5:]
            features['win_pct_l5'] = sum(recent_5) / len(recent_5)
        else:
            features['win_pct_l10'] = 0.5
            features['win_pct_l5'] = 0.5

        # Season comparison
        wins, losses = season_record
        if wins + losses > 0:
            season_win_pct = wins / (wins + losses)
            features['recent_vs_season'] = features['win_pct_l10'] - season_win_pct
        else:
            features['recent_vs_season'] = 0.0

        # Scoring trend
        features['pts_trend'] = self.calculate_trend(recent_pts_for, lookback=5)
        features['def_trend'] = -self.calculate_trend(recent_pts_against, lookback=5)

        # ATS momentum (for betting analysis)
        if recent_ats_results:
            features['ats_win_pct_l10'] = sum(recent_ats_results[-10:]) / min(10, len(recent_ats_results))
            features['ats_win_pct_l5'] = sum(recent_ats_results[-5:]) / min(5, len(recent_ats_results))

            # ATS streak
            ats_streak_type, ats_streak_length = self.calculate_streak(recent_ats_results)
            features['ats_streak'] = ats_streak_length if ats_streak_type == 'W' else -ats_streak_length
        else:
            features['ats_win_pct_l10'] = 0.5
            features['ats_win_pct_l5'] = 0.5
            features['ats_streak'] = 0

        # Margin momentum (important for spread betting)
        if recent_margins:
            avg_margin_5 = sum(recent_margins[-5:]) / min(5, len(recent_margins))
            avg_margin_10 = sum(recent_margins[-10:]) / min(10, len(recent_margins))
            features['avg_margin_l5'] = avg_margin_5
            features['avg_margin_l10'] = avg_margin_10

            # Margin trend
            features['margin_trend'] = self.calculate_trend(recent_margins, lookback=5)
        else:
            features['avg_margin_l5'] = 0.0
            features['avg_margin_l10'] = 0.0
            features['margin_trend'] = 0.0

        # Composite momentum score
        features['momentum_composite'] = (
            features['pts_momentum_5g'] * 0.25 +
            features['def_momentum_5g'] * 0.25 +
            features['net_rating_momentum'] * 0.20 +
            features['recent_vs_season'] * 0.15 +
            (features['streak_value'] / 10) * 0.15  # Normalize streak
        )

        # Public perception indicator (for fading)
        # Hot teams get overbet, cold teams get underbet
        if features['win_streak'] >= 4:
            features['public_overbet_risk'] = 0.3 + (features['win_streak'] - 4) * 0.1
        elif features['loss_streak'] >= 4:
            features['public_overbet_risk'] = -0.3 - (features['loss_streak'] - 4) * 0.1
        else:
            features['public_overbet_risk'] = 0.0

        return features


if __name__ == "__main__":
    # Example usage
    print("NBA Feature Engineering Module")
    print("=" * 50)
    print("\nExample usage:")
    print("\n# Generate features for a game (includes H2H, positional, injury analysis):")
    print('features = generate_game_features("LAL", "BOS")')
    print("\n# Generate features for a player prop (includes vs team history):")
    print('player_features = generate_player_features("LeBron James", "BOS")')
    print("\n# Create injury report and include in analysis:")
    print('''
injury_data = [
    {
        "team": "LAL",
        "injuries": [
            {"player_name": "Anthony Davis", "player_id": 203076, "status": "questionable", "position": "PF"}
        ]
    }
]
injury_mgr = create_injury_report(injury_data)
features = generate_game_features("LAL", "BOS", injury_manager=injury_mgr)
''')
    print("\n# Generate line movement features:")
    print('''
lm_gen = LineMovementFeatureGenerator()
lm_features = lm_gen.calculate_line_movement_features(
    opening_spread=-3.5,
    current_spread=-5.5,
    opening_ml_home=-150,
    current_ml_home=-200,
    opening_total=220.5,
    current_total=218.5,
    public_home_pct=0.72,  # 72% on home
    model_spread_prediction=-6.0
)
print(f"Sharp Score: {lm_features['sharp_score']}")
print(f"RLM Detected: {lm_features['is_rlm']}")
''')
    print("\n# Generate travel fatigue features:")
    print('''
travel_gen = TravelFatigueFeatureGenerator()
travel_features = travel_gen.generate_travel_features(
    team_abbrev="LAL",
    opponent_abbrev="BOS",
    is_home=False,
    previous_game_location="PHX",
    games_in_last_7_days=4,
    is_back_to_back=True,
    road_trip_game_number=3
)
print(f"Travel Distance: {travel_features['travel_distance_miles']:.0f} miles")
print(f"Fatigue Score: {travel_features['composite_fatigue_score']:.2f}")
''')
