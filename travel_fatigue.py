"""
Travel Fatigue Module for NBA Predictions

This module calculates travel-related fatigue factors that impact player and team performance:
- Travel distance (Haversine formula)
- Days rest / back-to-back detection
- Schedule density (3-in-4, 4-in-5 nights)
- Altitude adjustments (Denver, Utah)
- Timezone crossings

Research-backed adjustments:
- Back-to-back games: -2.1 points expected performance
- 3-in-4 nights: -1.5 points
- 4-in-5 nights: -2.5 points
- Denver altitude (5,280 ft): +1.5 home advantage
- Utah altitude (4,327 ft): +1.0 home advantage

Usage:
    from travel_fatigue import TravelFatigueCalculator

    calc = TravelFatigueCalculator()
    features = calc.get_travel_features(
        team_id=1,
        game_date='2025-01-14',
        opponent_id=2,
        is_home=False,
        recent_games=[...]
    )
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from math import radians, cos, sin, asin, sqrt


# NBA Arena Data: coordinates, altitude (ft), timezone offset
NBA_ARENA_DATA = {
    # Atlantic Division
    'BOS': {'coords': (42.366, -71.062), 'altitude': 20, 'timezone': -5, 'name': 'Boston'},
    'BKN': {'coords': (40.683, -73.976), 'altitude': 30, 'timezone': -5, 'name': 'Brooklyn'},
    'NYK': {'coords': (40.751, -73.994), 'altitude': 33, 'timezone': -5, 'name': 'New York'},
    'PHI': {'coords': (39.901, -75.172), 'altitude': 39, 'timezone': -5, 'name': 'Philadelphia'},
    'TOR': {'coords': (43.643, -79.379), 'altitude': 249, 'timezone': -5, 'name': 'Toronto'},
    # Central Division
    'CHI': {'coords': (41.881, -87.674), 'altitude': 594, 'timezone': -6, 'name': 'Chicago'},
    'CLE': {'coords': (41.497, -81.688), 'altitude': 653, 'timezone': -5, 'name': 'Cleveland'},
    'DET': {'coords': (42.341, -83.055), 'altitude': 600, 'timezone': -5, 'name': 'Detroit'},
    'IND': {'coords': (39.764, -86.156), 'altitude': 715, 'timezone': -5, 'name': 'Indiana'},
    'MIL': {'coords': (43.045, -87.917), 'altitude': 617, 'timezone': -6, 'name': 'Milwaukee'},
    # Southeast Division
    'ATL': {'coords': (33.757, -84.396), 'altitude': 1050, 'timezone': -5, 'name': 'Atlanta'},
    'CHA': {'coords': (35.225, -80.839), 'altitude': 751, 'timezone': -5, 'name': 'Charlotte'},
    'MIA': {'coords': (25.781, -80.188), 'altitude': 10, 'timezone': -5, 'name': 'Miami'},
    'ORL': {'coords': (28.539, -81.384), 'altitude': 82, 'timezone': -5, 'name': 'Orlando'},
    'WAS': {'coords': (38.898, -77.021), 'altitude': 50, 'timezone': -5, 'name': 'Washington'},
    # Northwest Division
    'DEN': {'coords': (39.749, -105.008), 'altitude': 5280, 'timezone': -7, 'name': 'Denver'},  # HIGH ALTITUDE
    'MIN': {'coords': (44.979, -93.276), 'altitude': 830, 'timezone': -6, 'name': 'Minnesota'},
    'OKC': {'coords': (35.463, -97.515), 'altitude': 1201, 'timezone': -6, 'name': 'Oklahoma City'},
    'POR': {'coords': (45.532, -122.667), 'altitude': 77, 'timezone': -8, 'name': 'Portland'},
    'UTA': {'coords': (40.768, -111.901), 'altitude': 4327, 'timezone': -7, 'name': 'Utah'},  # HIGH ALTITUDE
    # Pacific Division
    'GSW': {'coords': (37.768, -122.388), 'altitude': 13, 'timezone': -8, 'name': 'Golden State'},
    'LAC': {'coords': (34.043, -118.267), 'altitude': 270, 'timezone': -8, 'name': 'LA Clippers'},
    'LAL': {'coords': (34.043, -118.267), 'altitude': 270, 'timezone': -8, 'name': 'LA Lakers'},
    'PHX': {'coords': (33.446, -112.071), 'altitude': 1086, 'timezone': -7, 'name': 'Phoenix'},
    'SAC': {'coords': (38.580, -121.500), 'altitude': 30, 'timezone': -8, 'name': 'Sacramento'},
    # Southwest Division
    'DAL': {'coords': (32.790, -96.810), 'altitude': 430, 'timezone': -6, 'name': 'Dallas'},
    'HOU': {'coords': (29.751, -95.362), 'altitude': 50, 'timezone': -6, 'name': 'Houston'},
    'MEM': {'coords': (35.138, -90.051), 'altitude': 337, 'timezone': -6, 'name': 'Memphis'},
    'NOP': {'coords': (29.949, -90.082), 'altitude': 3, 'timezone': -6, 'name': 'New Orleans'},
    'SAS': {'coords': (29.427, -98.437), 'altitude': 650, 'timezone': -6, 'name': 'San Antonio'},
}

# Team ID to abbreviation mapping (Balldontlie API)
TEAM_ID_TO_ABBREV = {
    1: 'ATL', 2: 'BOS', 3: 'BKN', 4: 'CHA', 5: 'CHI',
    6: 'CLE', 7: 'DAL', 8: 'DEN', 9: 'DET', 10: 'GSW',
    11: 'HOU', 12: 'IND', 13: 'LAC', 14: 'LAL', 15: 'MEM',
    16: 'MIA', 17: 'MIL', 18: 'MIN', 19: 'NOP', 20: 'NYK',
    21: 'OKC', 22: 'ORL', 23: 'PHI', 24: 'PHX', 25: 'POR',
    26: 'SAC', 27: 'SAS', 28: 'TOR', 29: 'UTA', 30: 'WAS'
}


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.

    Args:
        coord1: (latitude, longitude) of first point
        coord2: (latitude, longitude) of second point

    Returns:
        Distance in miles
    """
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Earth's radius in miles
    R = 3959
    return R * c


class TravelFatigueCalculator:
    """Calculate comprehensive travel fatigue features for NBA predictions."""

    def __init__(self):
        self.arena_data = NBA_ARENA_DATA
        self.team_id_map = TEAM_ID_TO_ABBREV

    def _get_team_abbrev(self, team_id: int) -> Optional[str]:
        """Convert team ID to abbreviation."""
        return self.team_id_map.get(team_id)

    def calculate_travel_distance(self, from_team: str, to_team: str) -> float:
        """
        Calculate travel distance between two NBA arenas.

        Args:
            from_team: Team abbreviation (e.g., 'LAL')
            to_team: Team abbreviation (e.g., 'BOS')

        Returns:
            Distance in miles
        """
        from_arena = self.arena_data.get(from_team)
        to_arena = self.arena_data.get(to_team)

        if not from_arena or not to_arena:
            return 0.0

        return haversine_distance(from_arena['coords'], to_arena['coords'])

    def get_days_rest(self, team_games: List[Dict], game_date: str) -> int:
        """
        Calculate days since team's last game.

        Args:
            team_games: List of recent games for team (sorted by date desc)
            game_date: Current game date (YYYY-MM-DD)

        Returns:
            Days since last game (0 = back-to-back, 1 = 1 day rest, etc.)
        """
        if not team_games:
            return 3  # Default: assume well-rested

        try:
            current_date = datetime.strptime(game_date, "%Y-%m-%d")
            last_game_date = datetime.strptime(team_games[0]['date'], "%Y-%m-%d")
            days_diff = (current_date - last_game_date).days
            return max(0, days_diff)
        except:
            return 3

    def detect_schedule_density(self, team_games: List[Dict], game_date: str) -> Dict[str, int]:
        """
        Detect compressed schedules (3-in-4 nights, 4-in-5 nights, etc.).

        Research shows these have significant impact:
        - 3 games in 4 nights: -1.5 points
        - 4 games in 5 nights: -2.5 points

        Args:
            team_games: List of recent games (sorted by date desc)
            game_date: Current game date (YYYY-MM-DD)

        Returns:
            Dict with schedule density flags
        """
        result = {
            'is_3_in_4': 0,  # 3 games in 4 nights (including current)
            'is_4_in_5': 0,  # 4 games in 5 nights
            'games_last_5_days': 0,  # Total games in last 5 days
            'games_last_7_days': 0,  # Total games in last 7 days
        }

        if not team_games:
            return result

        try:
            current_date = datetime.strptime(game_date, "%Y-%m-%d")

            # Count games in last N days
            games_4_days = []
            games_5_days = []
            games_7_days = []

            for game in team_games:
                game_date_obj = datetime.strptime(game['date'], "%Y-%m-%d")
                days_ago = (current_date - game_date_obj).days

                if days_ago <= 3:  # Last 4 days (0-3 days ago)
                    games_4_days.append(game)
                if days_ago <= 4:  # Last 5 days
                    games_5_days.append(game)
                if days_ago <= 6:  # Last 7 days
                    games_7_days.append(game)

            # Include current game in counts
            result['games_last_5_days'] = len(games_5_days) + 1
            result['games_last_7_days'] = len(games_7_days) + 1

            # 3 in 4: 2 games already + current = 3 games in 4 nights
            if len(games_4_days) >= 2:
                result['is_3_in_4'] = 1

            # 4 in 5: 3 games already + current = 4 games in 5 nights
            if len(games_5_days) >= 3:
                result['is_4_in_5'] = 1

        except Exception as e:
            pass

        return result

    def calculate_altitude_adjustment(self, team_abbrev: str, game_team_abbrev: str, is_home: bool) -> float:
        """
        Calculate altitude advantage/disadvantage.

        Research shows:
        - Denver (5,280 ft): +1.5 point home advantage
        - Utah (4,327 ft): +1.0 point home advantage
        - Takes 24-48 hours for visiting teams to acclimate

        Args:
            team_abbrev: Abbreviation of team playing (e.g., 'LAL')
            game_team_abbrev: Abbreviation of team hosting (e.g., 'DEN')
            is_home: Whether team is playing at home

        Returns:
            Altitude adjustment in points (positive = advantage, negative = disadvantage)
        """
        game_arena = self.arena_data.get(game_team_abbrev)
        if not game_arena:
            return 0.0

        altitude = game_arena['altitude']

        # High altitude cities (above 4000 ft)
        if altitude >= 5000:  # Denver
            return 1.5 if is_home else -1.5
        elif altitude >= 4000:  # Utah
            return 1.0 if is_home else -1.0
        else:
            return 0.0

    def calculate_timezone_crossings(self, from_team: str, to_team: str) -> int:
        """
        Calculate number of timezone crossings.

        Args:
            from_team: Team abbreviation (e.g., 'LAL')
            to_team: Team abbreviation (e.g., 'BOS')

        Returns:
            Number of timezone crossings (0-4)
        """
        from_arena = self.arena_data.get(from_team)
        to_arena = self.arena_data.get(to_team)

        if not from_arena or not to_arena:
            return 0

        return abs(to_arena['timezone'] - from_arena['timezone'])

    def get_travel_features(
        self,
        team_id: int,
        game_date: str,
        opponent_id: int,
        is_home: bool,
        team_games: List[Dict],
        opponent_games: List[Dict] = None
    ) -> Dict[str, float]:
        """
        Generate comprehensive travel fatigue features.

        Args:
            team_id: Team ID (Balldontlie API)
            game_date: Game date (YYYY-MM-DD)
            opponent_id: Opponent team ID
            is_home: Whether team is playing at home
            team_games: Recent games for team (sorted by date desc)
            opponent_games: Recent games for opponent (optional)

        Returns:
            Dictionary of travel features (18 features total)
        """
        team_abbrev = self._get_team_abbrev(team_id)
        opp_abbrev = self._get_team_abbrev(opponent_id)

        if not team_abbrev or not opp_abbrev:
            return self._get_default_features()

        # Get last game location
        last_location = team_abbrev if not team_games else self._get_game_location(team_games[0], team_id)
        current_location = team_abbrev if is_home else opp_abbrev

        # 1. Days rest and back-to-back
        days_rest = self.get_days_rest(team_games, game_date)
        is_back_to_back = 1 if days_rest == 0 else 0

        # 2. Schedule density
        schedule_density = self.detect_schedule_density(team_games, game_date)

        # 3. Travel distance
        travel_distance = self.calculate_travel_distance(last_location, current_location)

        # 4. Timezone crossings
        timezone_crossings = self.calculate_timezone_crossings(last_location, current_location)

        # 5. Altitude adjustment
        altitude_adjustment = self.calculate_altitude_adjustment(team_abbrev, current_location, is_home)

        # 6. Research-backed point adjustments
        expected_impact = 0.0
        if is_back_to_back:
            expected_impact -= 2.1
        if schedule_density['is_3_in_4']:
            expected_impact -= 1.5
        if schedule_density['is_4_in_5']:
            expected_impact -= 2.5

        # 7. Coast-to-coast flag (2000+ miles)
        is_coast_to_coast = 1 if travel_distance >= 2000 else 0

        # 8. Composite fatigue score (0-1 scale)
        distance_factor = min(travel_distance / 3000, 1.0) * 0.4
        tz_factor = min(timezone_crossings / 3, 1.0) * 0.25
        schedule_factor = (schedule_density['games_last_5_days'] - 1) / 4 * 0.35  # Normalize to 0-1

        fatigue_score = distance_factor + tz_factor + schedule_factor

        # Rest mitigates fatigue
        if days_rest >= 2:
            fatigue_score *= 0.6
        elif days_rest == 1:
            fatigue_score *= 0.85

        fatigue_score = min(fatigue_score, 1.0)

        return {
            # Rest features (3)
            'days_rest': days_rest,
            'is_back_to_back': is_back_to_back,
            'is_well_rested': 1 if days_rest >= 2 else 0,

            # Schedule density (4)
            'is_3_in_4': schedule_density['is_3_in_4'],
            'is_4_in_5': schedule_density['is_4_in_5'],
            'games_last_5_days': schedule_density['games_last_5_days'],
            'games_last_7_days': schedule_density['games_last_7_days'],

            # Travel distance (3)
            'travel_distance': round(travel_distance, 1),
            'is_coast_to_coast': is_coast_to_coast,
            'timezone_crossings': timezone_crossings,

            # Altitude (2)
            'altitude_adjustment': round(altitude_adjustment, 2),
            'playing_high_altitude': 1 if abs(altitude_adjustment) > 0 else 0,

            # Impact estimates (3)
            'expected_fatigue_impact': round(expected_impact, 2),
            'fatigue_score': round(fatigue_score, 3),
            'travel_fatigue_multiplier': round(1 + (expected_impact / 100), 4),  # e.g., -2.1 pts = 0.979x

            # Meta features (3)
            'is_long_road_trip': 1 if not is_home and travel_distance > 1500 else 0,
            'is_home_heavy_schedule': 1 if is_home and schedule_density['games_last_5_days'] >= 3 else 0,
            'rest_advantage': 0,  # Placeholder for differential vs opponent
        }

    def _get_game_location(self, game: Dict, team_id: int) -> str:
        """Determine which team's arena the game was played at."""
        # Assume game dict has 'home_team_id' or we infer from team_id
        if 'home_team_id' in game:
            location_id = game['home_team_id']
        else:
            # Default: assume road game
            location_id = team_id

        return self._get_team_abbrev(location_id) or 'UNK'

    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when data is unavailable."""
        return {
            'days_rest': 1,
            'is_back_to_back': 0,
            'is_well_rested': 0,
            'is_3_in_4': 0,
            'is_4_in_5': 0,
            'games_last_5_days': 1,
            'games_last_7_days': 1,
            'travel_distance': 0.0,
            'is_coast_to_coast': 0,
            'timezone_crossings': 0,
            'altitude_adjustment': 0.0,
            'playing_high_altitude': 0,
            'expected_fatigue_impact': 0.0,
            'fatigue_score': 0.0,
            'travel_fatigue_multiplier': 1.0,
            'is_long_road_trip': 0,
            'is_home_heavy_schedule': 0,
            'rest_advantage': 0,
        }


# Convenience functions for backward compatibility
def calculate_travel_distance(from_team: str, to_team: str) -> float:
    """Calculate travel distance between two teams."""
    calc = TravelFatigueCalculator()
    return calc.calculate_travel_distance(from_team, to_team)


def get_days_rest(team_id: int, game_date: str, team_games: List[Dict]) -> int:
    """Get days rest for a team."""
    calc = TravelFatigueCalculator()
    return calc.get_days_rest(team_games, game_date)


def detect_schedule_density(team_id: int, game_date: str, team_games: List[Dict]) -> Dict:
    """Detect compressed schedules."""
    calc = TravelFatigueCalculator()
    return calc.detect_schedule_density(team_games, game_date)


def calculate_altitude_adjustment(team_id: int, game_team_id: int, is_home: bool) -> float:
    """Calculate altitude advantage/disadvantage."""
    calc = TravelFatigueCalculator()
    team_abbrev = calc._get_team_abbrev(team_id)
    game_abbrev = calc._get_team_abbrev(game_team_id)
    return calc.calculate_altitude_adjustment(team_abbrev, game_abbrev, is_home)


def calculate_timezone_crossings(from_team: str, to_team: str) -> int:
    """Calculate timezone crossings."""
    calc = TravelFatigueCalculator()
    return calc.calculate_timezone_crossings(from_team, to_team)
