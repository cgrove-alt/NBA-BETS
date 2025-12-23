"""
NBA Injury Data Fetcher

Fetches real-time injury data from multiple sources to enhance prediction accuracy.
Injuries significantly impact team performance and betting lines.

Data Sources:
1. Official NBA Injury Report (via web scraping/API)
2. nbainjuries package (if available)
3. ESPN injury data (backup source)

Injury Statuses (Official NBA):
- Out: Player will not play
- Doubtful: Unlikely to play (25% chance)
- Questionable: Uncertain (50% chance)
- Probable: Likely to play (75% chance)
- Available: Player cleared to play
- GTD (Game Time Decision): Will be decided before tip-off
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InjuryStatus(Enum):
    """Official NBA injury status classifications."""
    OUT = "Out"
    DOUBTFUL = "Doubtful"
    QUESTIONABLE = "Questionable"
    PROBABLE = "Probable"
    GTD = "GTD"
    AVAILABLE = "Available"
    UNKNOWN = "Unknown"

    @classmethod
    def from_string(cls, status: str) -> "InjuryStatus":
        """Parse status string to enum."""
        status_lower = status.lower().strip()
        mapping = {
            "out": cls.OUT,
            "o": cls.OUT,
            "doubtful": cls.DOUBTFUL,
            "d": cls.DOUBTFUL,
            "questionable": cls.QUESTIONABLE,
            "q": cls.QUESTIONABLE,
            "probable": cls.PROBABLE,
            "p": cls.PROBABLE,
            "gtd": cls.GTD,
            "game time decision": cls.GTD,
            "available": cls.AVAILABLE,
            "day-to-day": cls.GTD,
            "day to day": cls.GTD,
        }
        return mapping.get(status_lower, cls.UNKNOWN)

    def availability_probability(self) -> float:
        """Return probability player will be available."""
        probs = {
            InjuryStatus.OUT: 0.0,
            InjuryStatus.DOUBTFUL: 0.25,
            InjuryStatus.QUESTIONABLE: 0.50,
            InjuryStatus.PROBABLE: 0.75,
            InjuryStatus.GTD: 0.50,
            InjuryStatus.AVAILABLE: 1.0,
            InjuryStatus.UNKNOWN: 0.50,
        }
        return probs.get(self, 0.50)


@dataclass
class InjuryReport:
    """Individual player injury report."""
    player_name: str
    player_id: Optional[str] = None
    team: str = ""
    team_id: Optional[str] = None
    status: InjuryStatus = InjuryStatus.UNKNOWN
    injury_type: str = ""  # e.g., "Knee", "Ankle", "Illness"
    injury_detail: str = ""  # e.g., "Left knee soreness"
    report_date: Optional[datetime] = None
    expected_return: Optional[str] = None  # e.g., "2-3 weeks"
    games_missed: int = 0
    source: str = ""

    # Player value metrics for impact calculation
    ppg: float = 0.0  # Points per game
    rpg: float = 0.0  # Rebounds per game
    apg: float = 0.0  # Assists per game
    minutes: float = 0.0  # Minutes per game

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['status'] = self.status.value
        d['report_date'] = self.report_date.isoformat() if self.report_date else None
        return d

    def availability_probability(self) -> float:
        """Get probability player will play."""
        return self.status.availability_probability()


@dataclass
class TeamInjuryImpact:
    """Calculated impact of injuries on a team."""
    team: str
    team_id: Optional[str] = None
    total_players_out: int = 0
    total_players_questionable: int = 0

    # Estimated stat impact (negative values = team plays worse)
    points_impact: float = 0.0
    rebounds_impact: float = 0.0
    assists_impact: float = 0.0
    minutes_impact: float = 0.0

    # Overall impact score (-1 to 0, where -1 is catastrophic)
    overall_impact: float = 0.0

    # Key player flags
    star_player_out: bool = False
    starting_pg_out: bool = False

    injuries: List[InjuryReport] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "team": self.team,
            "team_id": self.team_id,
            "total_players_out": self.total_players_out,
            "total_players_questionable": self.total_players_questionable,
            "points_impact": self.points_impact,
            "rebounds_impact": self.rebounds_impact,
            "assists_impact": self.assists_impact,
            "minutes_impact": self.minutes_impact,
            "overall_impact": self.overall_impact,
            "star_player_out": self.star_player_out,
            "starting_pg_out": self.starting_pg_out,
            "injuries": [inj.to_dict() for inj in self.injuries]
        }


# NBA team abbreviation mappings
NBA_TEAM_MAPPING = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

# Reverse mapping (full name to abbreviation)
NBA_TEAM_ABBREV = {v: k for k, v in NBA_TEAM_MAPPING.items()}
# Add ESPN alternate names
NBA_TEAM_ABBREV["LA Clippers"] = "LAC"
NBA_TEAM_ABBREV["LA Lakers"] = "LAL"


class InjuryFetcher:
    """
    Fetches and processes NBA injury data from multiple sources.

    Primary sources:
    1. ESPN NBA Injuries API
    2. CBS Sports injury data
    3. Official NBA injury report (via RotoWire)
    """

    # ESPN API endpoints
    ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"

    # RotoWire (aggregates official NBA injury report)
    ROTOWIRE_URL = "https://www.rotowire.com/basketball/nba-lineups.php"

    def __init__(self, cache_duration_minutes: int = 30):
        """
        Initialize the injury fetcher.

        Args:
            cache_duration_minutes: How long to cache injury data (default 30 min)
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._star_players = self._load_star_players()

    def _load_star_players(self) -> Dict[str, Dict]:
        """Load list of star players for impact calculations."""
        # Top 50 NBA players by impact (simplified list)
        # In production, this would come from a database
        return {
            "LeBron James": {"team": "LAL", "ppg": 25.5, "rpg": 7.5, "apg": 8.0, "min": 35.0, "tier": 1},
            "Stephen Curry": {"team": "GSW", "ppg": 26.0, "rpg": 4.5, "apg": 5.5, "min": 32.0, "tier": 1},
            "Kevin Durant": {"team": "PHX", "ppg": 27.0, "rpg": 6.5, "apg": 5.0, "min": 36.0, "tier": 1},
            "Giannis Antetokounmpo": {"team": "MIL", "ppg": 30.0, "rpg": 12.0, "apg": 6.0, "min": 35.0, "tier": 1},
            "Nikola Jokic": {"team": "DEN", "ppg": 26.0, "rpg": 12.5, "apg": 9.5, "min": 34.0, "tier": 1},
            "Luka Doncic": {"team": "DAL", "ppg": 33.0, "rpg": 9.0, "apg": 9.5, "min": 37.0, "tier": 1},
            "Joel Embiid": {"team": "PHI", "ppg": 34.0, "rpg": 11.0, "apg": 5.5, "min": 34.0, "tier": 1},
            "Jayson Tatum": {"team": "BOS", "ppg": 27.0, "rpg": 8.5, "apg": 4.5, "min": 36.0, "tier": 1},
            "Anthony Davis": {"team": "LAL", "ppg": 25.0, "rpg": 12.5, "apg": 3.5, "min": 35.0, "tier": 1},
            "Shai Gilgeous-Alexander": {"team": "OKC", "ppg": 31.0, "rpg": 5.5, "apg": 6.5, "min": 34.0, "tier": 1},
            "Damian Lillard": {"team": "MIL", "ppg": 26.0, "rpg": 4.5, "apg": 7.0, "min": 35.0, "tier": 1},
            "Jaylen Brown": {"team": "BOS", "ppg": 23.0, "rpg": 5.5, "apg": 3.5, "min": 34.0, "tier": 2},
            "Anthony Edwards": {"team": "MIN", "ppg": 26.5, "rpg": 5.5, "apg": 5.0, "min": 35.0, "tier": 1},
            "Devin Booker": {"team": "PHX", "ppg": 27.0, "rpg": 4.5, "apg": 6.5, "min": 35.0, "tier": 1},
            "Donovan Mitchell": {"team": "CLE", "ppg": 28.0, "rpg": 4.5, "apg": 5.0, "min": 35.0, "tier": 2},
            "Trae Young": {"team": "ATL", "ppg": 26.5, "rpg": 3.0, "apg": 10.5, "min": 35.0, "tier": 2},
            "Jimmy Butler": {"team": "MIA", "ppg": 21.0, "rpg": 5.5, "apg": 5.5, "min": 33.0, "tier": 2},
            "Kawhi Leonard": {"team": "LAC", "ppg": 24.0, "rpg": 6.0, "apg": 4.0, "min": 32.0, "tier": 2},
            "Paul George": {"team": "PHI", "ppg": 22.5, "rpg": 5.5, "apg": 4.0, "min": 34.0, "tier": 2},
            "Tyrese Haliburton": {"team": "IND", "ppg": 20.5, "rpg": 4.0, "apg": 10.5, "min": 33.0, "tier": 2},
            "Ja Morant": {"team": "MEM", "ppg": 26.0, "rpg": 6.0, "apg": 8.0, "min": 32.0, "tier": 1},
            "Zion Williamson": {"team": "NOP", "ppg": 23.0, "rpg": 6.5, "apg": 5.0, "min": 30.0, "tier": 2},
            "De'Aaron Fox": {"team": "SAC", "ppg": 26.0, "rpg": 4.5, "apg": 6.0, "min": 35.0, "tier": 2},
            "Domantas Sabonis": {"team": "SAC", "ppg": 19.5, "rpg": 13.5, "apg": 8.0, "min": 35.0, "tier": 2},
            "Karl-Anthony Towns": {"team": "NYK", "ppg": 22.0, "rpg": 9.0, "apg": 3.0, "min": 33.0, "tier": 2},
            "Jalen Brunson": {"team": "NYK", "ppg": 28.0, "rpg": 3.5, "apg": 6.5, "min": 35.0, "tier": 2},
            "Bam Adebayo": {"team": "MIA", "ppg": 20.0, "rpg": 10.5, "apg": 3.5, "min": 34.0, "tier": 2},
            "Victor Wembanyama": {"team": "SAS", "ppg": 22.0, "rpg": 10.5, "apg": 4.0, "min": 30.0, "tier": 1},
            "Chet Holmgren": {"team": "OKC", "ppg": 17.0, "rpg": 8.5, "apg": 2.5, "min": 30.0, "tier": 2},
            "Paolo Banchero": {"team": "ORL", "ppg": 23.0, "rpg": 7.0, "apg": 5.5, "min": 35.0, "tier": 2},
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        cached_time, _ = self._cache[cache_key]
        return datetime.now() - cached_time < self.cache_duration

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get cached data if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
        return None

    def _set_cache(self, cache_key: str, data: Any) -> None:
        """Set cache data."""
        self._cache[cache_key] = (datetime.now(), data)

    def fetch_espn_injuries(self) -> List[InjuryReport]:
        """
        Fetch injury data from ESPN API.

        Returns:
            List of InjuryReport objects
        """
        cache_key = "espn_injuries"
        cached = self._get_cached(cache_key)
        if cached:
            logger.info("Using cached ESPN injury data")
            return cached

        injuries = []

        try:
            response = requests.get(self.ESPN_INJURIES_URL, timeout=10)
            response.raise_for_status()
            data = response.json()

            # ESPN returns injuries grouped by team
            for team_data in data.get("injuries", []):
                # ESPN API has displayName directly on team_data (e.g., "Atlanta Hawks")
                team_name = team_data.get("displayName", "")
                team_id = team_data.get("id", "")
                # Look up abbreviation from full name
                team_abbrev = NBA_TEAM_ABBREV.get(team_name, "")

                for player_injury in team_data.get("injuries", []):
                    athlete = player_injury.get("athlete", {})
                    player_name = athlete.get("displayName", "")
                    player_id = athlete.get("id", "")

                    # Get injury details
                    injury_type = player_injury.get("type", {}).get("description", "")
                    injury_detail = player_injury.get("longComment", "") or player_injury.get("shortComment", "")
                    status_str = player_injury.get("status", "Unknown")

                    # Parse date
                    date_str = player_injury.get("date", "")
                    report_date = None
                    if date_str:
                        try:
                            report_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        except ValueError:
                            pass

                    # Get player stats if available
                    stats = self._star_players.get(player_name, {})

                    injury = InjuryReport(
                        player_name=player_name,
                        player_id=player_id,
                        team=team_abbrev or team_name,
                        team_id=team_id,
                        status=InjuryStatus.from_string(status_str),
                        injury_type=injury_type,
                        injury_detail=injury_detail,
                        report_date=report_date,
                        source="ESPN",
                        ppg=stats.get("ppg", 0.0),
                        rpg=stats.get("rpg", 0.0),
                        apg=stats.get("apg", 0.0),
                        minutes=stats.get("min", 0.0),
                    )
                    injuries.append(injury)

            logger.info(f"Fetched {len(injuries)} injuries from ESPN")
            self._set_cache(cache_key, injuries)
            return injuries

        except requests.RequestException as e:
            logger.error(f"Failed to fetch ESPN injuries: {e}")
            return []
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse ESPN injury data: {e}")
            return []

    def fetch_all_injuries(self) -> List[InjuryReport]:
        """
        Fetch injuries from all available sources and merge.

        Returns:
            Deduplicated list of InjuryReport objects
        """
        all_injuries = []
        seen_players = set()

        # Fetch from ESPN (primary source)
        espn_injuries = self.fetch_espn_injuries()
        for injury in espn_injuries:
            key = (injury.player_name.lower(), injury.team)
            if key not in seen_players:
                seen_players.add(key)
                all_injuries.append(injury)

        # Additional sources could be added here
        # cbs_injuries = self.fetch_cbs_injuries()
        # rotowire_injuries = self.fetch_rotowire_injuries()

        logger.info(f"Total unique injuries: {len(all_injuries)}")
        return all_injuries

    def get_team_injuries(self, team: str) -> List[InjuryReport]:
        """
        Get all injuries for a specific team.

        Args:
            team: Team abbreviation (e.g., "LAL") or full name

        Returns:
            List of injuries for the team
        """
        all_injuries = self.fetch_all_injuries()

        # Normalize team name
        team_upper = team.upper()
        team_full = NBA_TEAM_MAPPING.get(team_upper, team)
        team_abbrev = NBA_TEAM_ABBREV.get(team, team_upper)

        team_injuries = []
        for injury in all_injuries:
            injury_team = injury.team.upper()
            if (injury_team == team_upper or
                injury_team == team_abbrev or
                injury.team == team_full):
                team_injuries.append(injury)

        return team_injuries

    def calculate_team_impact(self, team: str) -> TeamInjuryImpact:
        """
        Calculate the overall impact of injuries on a team.

        Args:
            team: Team abbreviation or full name

        Returns:
            TeamInjuryImpact object with calculated metrics
        """
        injuries = self.get_team_injuries(team)

        impact = TeamInjuryImpact(
            team=team,
            injuries=injuries
        )

        total_ppg_lost = 0.0
        total_rpg_lost = 0.0
        total_apg_lost = 0.0
        total_min_lost = 0.0

        for injury in injuries:
            availability = injury.availability_probability()
            missing_prob = 1.0 - availability

            # Calculate expected stat loss
            if missing_prob > 0:
                # Use known stats or estimate
                ppg = injury.ppg if injury.ppg > 0 else 8.0  # Default for unknown players
                rpg = injury.rpg if injury.rpg > 0 else 3.0
                apg = injury.apg if injury.apg > 0 else 2.0
                minutes = injury.minutes if injury.minutes > 0 else 15.0

                total_ppg_lost += ppg * missing_prob
                total_rpg_lost += rpg * missing_prob
                total_apg_lost += apg * missing_prob
                total_min_lost += minutes * missing_prob

                if injury.status == InjuryStatus.OUT:
                    impact.total_players_out += 1
                elif injury.status in [InjuryStatus.QUESTIONABLE, InjuryStatus.DOUBTFUL, InjuryStatus.GTD]:
                    impact.total_players_questionable += 1

                # Check for star player
                if injury.player_name in self._star_players:
                    star_info = self._star_players[injury.player_name]
                    if star_info.get("tier") == 1 and missing_prob >= 0.75:
                        impact.star_player_out = True

        # Calculate impact as percentage of team production
        # Average NBA team: ~115 PPG, ~45 RPG, ~25 APG, 240 total minutes
        impact.points_impact = -total_ppg_lost / 115.0
        impact.rebounds_impact = -total_rpg_lost / 45.0
        impact.assists_impact = -total_apg_lost / 25.0
        impact.minutes_impact = -total_min_lost / 240.0

        # Overall impact weighted by importance
        # Points most important, then rebounds, then assists
        impact.overall_impact = (
            impact.points_impact * 0.5 +
            impact.rebounds_impact * 0.25 +
            impact.assists_impact * 0.15 +
            impact.minutes_impact * 0.10
        )

        # Clip to reasonable range
        impact.overall_impact = max(-1.0, min(0.0, impact.overall_impact))

        return impact

    def get_game_injury_summary(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, Any]:
        """
        Get injury summary for a specific game matchup.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Dictionary with injury impacts for both teams
        """
        home_impact = self.calculate_team_impact(home_team)
        away_impact = self.calculate_team_impact(away_team)

        # Calculate relative advantage
        # Positive = home team advantage, negative = away team advantage
        injury_advantage = away_impact.overall_impact - home_impact.overall_impact

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_impact": home_impact.to_dict(),
            "away_impact": away_impact.to_dict(),
            "injury_advantage_home": injury_advantage,
            "recommendation": self._get_injury_recommendation(home_impact, away_impact),
            "fetched_at": datetime.now().isoformat(),
        }

    def _get_injury_recommendation(
        self,
        home_impact: TeamInjuryImpact,
        away_impact: TeamInjuryImpact
    ) -> str:
        """Generate betting recommendation based on injuries."""
        advantage = away_impact.overall_impact - home_impact.overall_impact

        if home_impact.star_player_out and not away_impact.star_player_out:
            return f"STRONG FADE HOME: {home_impact.team} missing star player"
        elif away_impact.star_player_out and not home_impact.star_player_out:
            return f"STRONG LEAN HOME: {away_impact.team} missing star player"
        elif advantage > 0.10:
            return f"Lean HOME: {home_impact.team} has injury advantage"
        elif advantage < -0.10:
            return f"Lean AWAY: {away_impact.team} has injury advantage"
        else:
            return "NEUTRAL: No significant injury advantage"

    def get_spread_adjustment(
        self,
        home_team: str,
        away_team: str
    ) -> float:
        """
        Calculate spread adjustment based on injuries.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Spread adjustment (positive = favor home, negative = favor away)
        """
        home_impact = self.calculate_team_impact(home_team)
        away_impact = self.calculate_team_impact(away_team)

        # Each 0.10 impact roughly equals 1 point on the spread
        # Star player out can be worth 3-5 points alone
        adjustment = 0.0

        # Base adjustment from overall impact
        adjustment += (away_impact.overall_impact - home_impact.overall_impact) * 10.0

        # Additional adjustment for star players
        if home_impact.star_player_out:
            adjustment -= 4.0
        if away_impact.star_player_out:
            adjustment += 4.0

        # Clip to reasonable range (-10 to +10 points)
        return max(-10.0, min(10.0, adjustment))

    def format_injury_report(self, team: str = None) -> str:
        """
        Format a human-readable injury report.

        Args:
            team: Optional team to filter (None = all teams)

        Returns:
            Formatted string report
        """
        if team:
            injuries = self.get_team_injuries(team)
        else:
            injuries = self.fetch_all_injuries()

        if not injuries:
            return "No injuries reported."

        # Group by team
        by_team: Dict[str, List[InjuryReport]] = {}
        for injury in injuries:
            team_key = injury.team
            if team_key not in by_team:
                by_team[team_key] = []
            by_team[team_key].append(injury)

        lines = ["=" * 60, "NBA INJURY REPORT", f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "=" * 60]

        for team_name, team_injuries in sorted(by_team.items()):
            lines.append(f"\n{team_name}")
            lines.append("-" * 40)
            for injury in sorted(team_injuries, key=lambda x: x.status.availability_probability()):
                status_emoji = {
                    InjuryStatus.OUT: "[OUT]",
                    InjuryStatus.DOUBTFUL: "[DTD]",
                    InjuryStatus.QUESTIONABLE: "[Q]",
                    InjuryStatus.PROBABLE: "[P]",
                    InjuryStatus.GTD: "[GTD]",
                }.get(injury.status, "[?]")

                lines.append(f"  {status_emoji:6} {injury.player_name}")
                if injury.injury_detail:
                    lines.append(f"         {injury.injury_detail}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def create_default_fetcher() -> InjuryFetcher:
    """Create an InjuryFetcher with default settings."""
    return InjuryFetcher(cache_duration_minutes=30)


# Convenience functions for integration
def get_injuries_for_game(home_team: str, away_team: str) -> Dict[str, Any]:
    """
    Get injury summary for a game.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation

    Returns:
        Dictionary with injury data for both teams
    """
    fetcher = create_default_fetcher()
    return fetcher.get_game_injury_summary(home_team, away_team)


def get_spread_adjustment(home_team: str, away_team: str) -> float:
    """
    Get spread adjustment based on injuries.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation

    Returns:
        Spread adjustment in points
    """
    fetcher = create_default_fetcher()
    return fetcher.get_spread_adjustment(home_team, away_team)


def print_all_injuries() -> None:
    """Print formatted injury report for all teams."""
    fetcher = create_default_fetcher()
    print(fetcher.format_injury_report())


if __name__ == "__main__":
    # Example usage
    fetcher = InjuryFetcher()

    # Fetch all injuries
    print("Fetching NBA injuries...")
    injuries = fetcher.fetch_all_injuries()
    print(f"Found {len(injuries)} injuries\n")

    # Print formatted report
    print(fetcher.format_injury_report())

    # Example: Get impact for a specific game
    print("\n" + "=" * 60)
    print("GAME ANALYSIS: Lakers vs Celtics")
    print("=" * 60)

    summary = fetcher.get_game_injury_summary("LAL", "BOS")
    print(f"\nLakers injury impact: {summary['home_impact']['overall_impact']:.3f}")
    print(f"Celtics injury impact: {summary['away_impact']['overall_impact']:.3f}")
    print(f"Home advantage from injuries: {summary['injury_advantage_home']:.3f}")
    print(f"\nRecommendation: {summary['recommendation']}")

    spread_adj = fetcher.get_spread_adjustment("LAL", "BOS")
    print(f"Spread adjustment: {spread_adj:+.1f} points (+ = favor Lakers)")
