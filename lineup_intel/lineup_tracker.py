"""
Lineup Tracker - Confirmed Starting Lineup Intelligence

Tracks and monitors starting lineup confirmations from:
1. Balldontlie API (player stats endpoint shows starters)
2. ESPN Depth Charts
3. Historical patterns (when starters not yet confirmed)

Confirmed lineups typically become available:
- 6:00 PM ET on game days (official)
- 30-60 minutes before tip-off (final confirmation)
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import requests
from bs4 import BeautifulSoup

# Try Balldontlie API
try:
    from balldontlie_api import BalldontlieAPI
    BALLDONTLIE_AVAILABLE = True
except ImportError:
    BALLDONTLIE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Team mappings
NBA_TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


@dataclass
class StarterInfo:
    """Information about a starting player."""
    player_name: str
    player_id: Optional[str] = None
    team: str = ""
    position: str = ""  # PG, SG, SF, PF, C
    avg_minutes: float = 0.0
    avg_points: float = 0.0
    avg_rebounds: float = 0.0
    avg_assists: float = 0.0
    is_confirmed: bool = False
    confidence: float = 0.0  # 0-1, how confident in this starter

    def to_dict(self) -> dict:
        return {
            'player_name': self.player_name,
            'player_id': self.player_id,
            'team': self.team,
            'position': self.position,
            'avg_minutes': self.avg_minutes,
            'avg_points': self.avg_points,
            'avg_rebounds': self.avg_rebounds,
            'avg_assists': self.avg_assists,
            'is_confirmed': self.is_confirmed,
            'confidence': self.confidence,
        }


@dataclass
class LineupConfirmation:
    """Confirmed starting lineup for a team."""
    team: str
    team_id: Optional[str] = None
    game_date: str = ""
    opponent: str = ""
    starters: list[StarterInfo] = field(default_factory=list)
    is_confirmed: bool = False
    confirmation_time: Optional[datetime] = None
    source: str = ""
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            'team': self.team,
            'team_id': self.team_id,
            'game_date': self.game_date,
            'opponent': self.opponent,
            'starters': [s.to_dict() for s in self.starters],
            'is_confirmed': self.is_confirmed,
            'confirmation_time': self.confirmation_time.isoformat() if self.confirmation_time else None,
            'source': self.source,
            'last_updated': self.last_updated.isoformat(),
        }


class LineupTracker:
    """
    Track and predict starting lineups.

    Sources:
    1. Historical starting patterns (primary)
    2. ESPN depth charts (secondary)
    3. Live confirmations when available
    """

    ESPN_DEPTH_CHART_BASE = "https://www.espn.com/nba/team/depth/_/name"

    def __init__(self, cache_duration_minutes: int = 15):
        """
        Initialize lineup tracker.

        Args:
            cache_duration_minutes: How long to cache lineup data
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self._cache: dict[str, tuple[datetime, LineupConfirmation]] = {}

        # Balldontlie API for historical data
        self._balldontlie = None
        self._team_id_map = {}
        if BALLDONTLIE_AVAILABLE:
            try:
                self._balldontlie = BalldontlieAPI()
                self._build_team_mapping()
                logger.info("Balldontlie API initialized for lineup tracking")
            except Exception as e:
                logger.warning(f"Failed to init Balldontlie: {e}")

        # Expected starters based on historical data
        self._historical_starters: dict[str, list[StarterInfo]] = {}

    def _build_team_mapping(self):
        """Build team ID to abbreviation mapping."""
        if not self._balldontlie:
            return
        try:
            teams = self._balldontlie.get_teams()
            for team in teams:
                team_id = team.get("id")
                abbrev = team.get("abbreviation", "")
                if team_id and abbrev:
                    self._team_id_map[team_id] = abbrev
                    self._team_id_map[abbrev] = team_id  # Reverse mapping too
        except Exception as e:
            logger.warning(f"Failed to build team mapping: {e}")

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is valid."""
        if cache_key not in self._cache:
            return False
        cached_time, _ = self._cache[cache_key]
        return datetime.now() - cached_time < self.cache_duration

    def _analyze_historical_starters(self, team: str, recent_games: int = 10) -> list[StarterInfo]:
        """
        Analyze recent games to identify likely starters.

        Args:
            team: Team abbreviation
            recent_games: Number of games to analyze

        Returns:
            List of StarterInfo for likely starters
        """
        if not self._balldontlie:
            return []

        team_id = self._team_id_map.get(team.upper())
        if not team_id:
            return []

        try:
            # Get recent games for this team using current season
            current_year = datetime.now().year
            # NBA season spans two years - if before June, use previous year as season start
            season_year = current_year if datetime.now().month >= 10 else current_year - 1

            games = self._balldontlie.get_games(
                seasons=[season_year],
                team_ids=[team_id],
                per_page=recent_games
            )

            if not games:
                return []

            # Get player stats for these games
            game_ids = [g.get('id') for g in games if g.get('id')]

            # Track starter frequency
            starter_counts: dict[str, dict] = {}

            # Fetch stats for all games at once (more efficient)
            try:
                all_stats = self._balldontlie.get_player_stats(
                    game_ids=game_ids[:recent_games],
                    per_page=100
                )
            except Exception as e:
                logger.warning(f"Error fetching player stats: {e}")
                all_stats = []

            for stat in all_stats:
                player = stat.get('player', {})
                player_team_id = player.get('team_id')

                # Only count players from our team
                if player_team_id != team_id:
                    continue

                player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
                player_id = str(player.get('id', ''))

                # Check if starter (first appearance with significant minutes)
                minutes_str = stat.get('min', '0')
                if isinstance(minutes_str, str) and ':' in minutes_str:
                    mins = int(minutes_str.split(':')[0])
                else:
                    mins = int(float(minutes_str or 0))

                # Consider starter if played 25+ minutes
                if mins >= 25:
                    if player_name not in starter_counts:
                        starter_counts[player_name] = {
                            'player_id': player_id,
                            'games': 0,
                            'total_min': 0,
                            'total_pts': 0,
                            'total_reb': 0,
                            'total_ast': 0,
                            'position': player.get('position', 'F'),
                        }

                    starter_counts[player_name]['games'] += 1
                    starter_counts[player_name]['total_min'] += mins
                    starter_counts[player_name]['total_pts'] += stat.get('pts', 0) or 0
                    starter_counts[player_name]['total_reb'] += stat.get('reb', 0) or 0
                    starter_counts[player_name]['total_ast'] += stat.get('ast', 0) or 0

            # Build starter list (top 5 by games started)
            starters = []
            sorted_players = sorted(
                starter_counts.items(),
                key=lambda x: (x[1]['games'], x[1]['total_min']),
                reverse=True
            )

            for player_name, data in sorted_players[:5]:
                games_count = data['games']
                if games_count > 0:
                    starters.append(StarterInfo(
                        player_name=player_name,
                        player_id=data['player_id'],
                        team=team.upper(),
                        position=data['position'],
                        avg_minutes=data['total_min'] / games_count,
                        avg_points=data['total_pts'] / games_count,
                        avg_rebounds=data['total_reb'] / games_count,
                        avg_assists=data['total_ast'] / games_count,
                        is_confirmed=False,
                        confidence=min(0.95, games_count / recent_games),
                    ))

            return starters

        except Exception as e:
            logger.error(f"Error analyzing historical starters for {team}: {e}")
            return []

    def fetch_espn_depth_chart(self, team: str) -> list[StarterInfo]:
        """
        Scrape ESPN depth chart for a team.

        Args:
            team: Team abbreviation (e.g., "LAL")

        Returns:
            List of likely starters from depth chart
        """
        # ESPN team codes (some differ from NBA abbreviations)
        espn_team_codes = {
            "ATL": "atl", "BOS": "bos", "BKN": "bkn", "CHA": "cha",
            "CHI": "chi", "CLE": "cle", "DAL": "dal", "DEN": "den",
            "DET": "det", "GSW": "gs", "HOU": "hou", "IND": "ind",
            "LAC": "lac", "LAL": "lal", "MEM": "mem", "MIA": "mia",
            "MIL": "mil", "MIN": "min", "NOP": "no", "NYK": "ny",
            "OKC": "okc", "ORL": "orl", "PHI": "phi", "PHX": "phx",
            "POR": "por", "SAC": "sac", "SAS": "sa", "TOR": "tor",
            "UTA": "utah", "WAS": "wsh",
        }

        team_code = espn_team_codes.get(team.upper())
        if not team_code:
            logger.warning(f"Unknown team code for {team}")
            return []

        url = f"{self.ESPN_DEPTH_CHART_BASE}/{team_code}"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            starters = []
            # Find depth chart table
            tables = soup.find_all('table', class_='Table')

            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        # First cell is position, second is top player (starter)
                        position = cells[0].get_text(strip=True)
                        player_cell = cells[1]

                        # Get player name from link
                        player_link = player_cell.find('a')
                        if player_link:
                            player_name = player_link.get_text(strip=True)

                            # Skip header rows
                            if player_name.lower() in ['player', 'name', '']:
                                continue

                            starters.append(StarterInfo(
                                player_name=player_name,
                                team=team.upper(),
                                position=position,
                                is_confirmed=False,
                                confidence=0.75,  # Depth chart = moderate confidence
                            ))

            logger.info(f"Found {len(starters)} starters from ESPN depth chart for {team}")
            return starters[:5]  # Only top 5

        except requests.RequestException as e:
            logger.error(f"ESPN depth chart fetch error for {team}: {e}")
            return []
        except Exception as e:
            logger.error(f"ESPN depth chart parse error for {team}: {e}")
            return []

    def get_lineup(
        self,
        team: str,
        game_date: Optional[str] = None,
        opponent: Optional[str] = None,
        force_refresh: bool = False
    ) -> LineupConfirmation:
        """
        Get lineup for a team.

        Combines multiple sources:
        1. Check for confirmed lineup (live)
        2. Historical starter analysis
        3. ESPN depth chart

        Args:
            team: Team abbreviation
            game_date: Game date (YYYY-MM-DD), defaults to today
            opponent: Opponent team abbreviation
            force_refresh: Force refresh even if cached

        Returns:
            LineupConfirmation with starters
        """
        team_upper = team.upper()
        game_date = game_date or datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{team_upper}_{game_date}"

        if not force_refresh and self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]

        # Get starters from multiple sources
        all_starters: dict[str, StarterInfo] = {}

        # 1. Historical analysis (highest confidence)
        historical = self._analyze_historical_starters(team_upper)
        for starter in historical:
            all_starters[starter.player_name.lower()] = starter

        # 2. ESPN depth chart (fill in gaps)
        espn_starters = self.fetch_espn_depth_chart(team_upper)
        for starter in espn_starters:
            key = starter.player_name.lower()
            if key not in all_starters:
                all_starters[key] = starter
            else:
                # Merge position info if missing
                if not all_starters[key].position and starter.position:
                    all_starters[key].position = starter.position

        # Sort by confidence and take top 5
        sorted_starters = sorted(
            all_starters.values(),
            key=lambda x: (x.confidence, x.avg_minutes),
            reverse=True
        )[:5]

        # Determine if lineup is "confirmed"
        # Consider confirmed if all starters have >80% confidence
        is_confirmed = all(s.confidence >= 0.8 for s in sorted_starters) if sorted_starters else False

        lineup = LineupConfirmation(
            team=team_upper,
            team_id=str(self._team_id_map.get(team_upper, "")),
            game_date=game_date,
            opponent=opponent.upper() if opponent else "",
            starters=sorted_starters,
            is_confirmed=is_confirmed,
            confirmation_time=datetime.now() if is_confirmed else None,
            source="historical+espn",
            last_updated=datetime.now(),
        )

        self._cache[cache_key] = (datetime.now(), lineup)
        return lineup

    def get_expected_minutes(self, team: str, player_name: str) -> float:
        """
        Get expected minutes for a player based on lineup status.

        Args:
            team: Team abbreviation
            player_name: Player's full name

        Returns:
            Expected minutes (0 if not in lineup)
        """
        lineup = self.get_lineup(team)

        for starter in lineup.starters:
            if starter.player_name.lower() == player_name.lower():
                return starter.avg_minutes

            # Partial name match
            if player_name.lower().split()[-1] == starter.player_name.lower().split()[-1]:
                return starter.avg_minutes

        # Not in starting lineup - estimate bench minutes
        return 15.0  # Default bench minutes


# Singleton instance
_tracker_instance: Optional[LineupTracker] = None


def get_lineup_tracker() -> LineupTracker:
    """Get global LineupTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = LineupTracker()
    return _tracker_instance


if __name__ == "__main__":
    # Test the tracker
    tracker = LineupTracker()

    test_teams = ["LAL", "BOS", "GSW"]

    for team in test_teams:
        print(f"\n{'='*50}")
        print(f"LINEUP: {team}")
        print('='*50)

        lineup = tracker.get_lineup(team)
        print(f"Confirmed: {lineup.is_confirmed}")
        print(f"Source: {lineup.source}")
        print(f"\nExpected Starters:")

        for starter in lineup.starters:
            print(f"  {starter.position:3} {starter.player_name:25} "
                  f"({starter.avg_minutes:.1f} MPG, {starter.avg_points:.1f} PPG) "
                  f"[{starter.confidence*100:.0f}%]")
