"""
Injury Scraper - Multi-Source NBA Injury Report Fetcher

Data Sources:
1. Balldontlie API (primary - reliable, structured)
2. ESPN NBA Injuries page (secondary - web scraping)
3. NBA.com Official Injury Report (tertiary - released 5pm ET)

The NBA officially releases injury reports at 5pm ET on game days.
This scraper aggregates from multiple sources for redundancy.
"""

import re
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import requests
from bs4 import BeautifulSoup

# Try to import Balldontlie API
try:
    from balldontlie_api import BalldontlieAPI
    BALLDONTLIE_AVAILABLE = True
except ImportError:
    BALLDONTLIE_AVAILABLE = False

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
        if not status:
            return cls.UNKNOWN
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
            "game-time decision": cls.GTD,
            "available": cls.AVAILABLE,
            "day-to-day": cls.GTD,
            "day to day": cls.GTD,
            "expected to play": cls.PROBABLE,
            "not with team": cls.OUT,
            "suspended": cls.OUT,
        }
        return mapping.get(status_lower, cls.UNKNOWN)

    def availability_probability(self) -> float:
        """Return probability player will be available."""
        probs = {
            InjuryStatus.OUT: 0.0,
            InjuryStatus.DOUBTFUL: 0.25,
            InjuryStatus.QUESTIONABLE: 0.50,
            InjuryStatus.PROBABLE: 0.85,
            InjuryStatus.GTD: 0.50,
            InjuryStatus.AVAILABLE: 1.0,
            InjuryStatus.UNKNOWN: 0.75,  # Assume available if unknown
        }
        return probs.get(self, 0.50)

    def minutes_multiplier(self) -> float:
        """Return expected minutes multiplier (even if player plays)."""
        # Players coming off injury often play reduced minutes
        multipliers = {
            InjuryStatus.OUT: 0.0,
            InjuryStatus.DOUBTFUL: 0.0,
            InjuryStatus.QUESTIONABLE: 0.85,  # May be on minutes restriction
            InjuryStatus.PROBABLE: 0.95,
            InjuryStatus.GTD: 0.80,  # Often limited when GTD
            InjuryStatus.AVAILABLE: 1.0,
            InjuryStatus.UNKNOWN: 1.0,
        }
        return multipliers.get(self, 1.0)


@dataclass
class PlayerInjury:
    """Individual player injury report."""
    player_name: str
    player_id: str | None = None
    team: str = ""
    team_id: str | None = None
    status: InjuryStatus = InjuryStatus.UNKNOWN
    injury_type: str = ""  # e.g., "Knee", "Ankle", "Illness"
    injury_detail: str = ""  # e.g., "Left knee soreness"
    report_date: datetime | None = None
    expected_return: str | None = None
    source: str = ""
    confidence: float = 1.0  # How confident we are in this report

    # Calculated fields
    availability_prob: float = field(init=False)
    minutes_multiplier: float = field(init=False)

    def __post_init__(self):
        self.availability_prob = self.status.availability_probability()
        self.minutes_multiplier = self.status.minutes_multiplier()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['status'] = self.status.value
        d['report_date'] = self.report_date.isoformat() if self.report_date else None
        return d


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
NBA_TEAM_ABBREV = {v: k for k, v in NBA_TEAMS.items()}
# Add common variations
NBA_TEAM_ABBREV["LA Clippers"] = "LAC"
NBA_TEAM_ABBREV["LA Lakers"] = "LAL"


class InjuryScraper:
    """
    Multi-source injury report scraper.

    Fetches from:
    1. Balldontlie API (primary)
    2. ESPN injuries page (secondary)
    """

    ESPN_INJURIES_URL = "https://www.espn.com/nba/injuries"

    def __init__(self, cache_duration_minutes: int = 15):
        """
        Initialize injury scraper.

        Args:
            cache_duration_minutes: How long to cache injury data
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self._cache: dict[str, tuple[datetime, list[PlayerInjury]]] = {}

        # Initialize Balldontlie API if available
        self._balldontlie = None
        self._team_id_map = {}
        if BALLDONTLIE_AVAILABLE:
            try:
                self._balldontlie = BalldontlieAPI()
                self._build_team_id_mapping()
                logger.info("Balldontlie API initialized for injury scraping")
            except Exception as e:
                logger.warning(f"Failed to init Balldontlie API: {e}")

    def _build_team_id_mapping(self):
        """Build mapping of Balldontlie team IDs to abbreviations."""
        if not self._balldontlie:
            return
        try:
            teams = self._balldontlie.get_teams()
            for team in teams:
                team_id = team.get("id")
                abbrev = team.get("abbreviation", "")
                if team_id and abbrev:
                    self._team_id_map[team_id] = abbrev
        except Exception as e:
            logger.warning(f"Failed to build team mapping: {e}")

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid."""
        if cache_key not in self._cache:
            return False
        cached_time, _ = self._cache[cache_key]
        return datetime.now() - cached_time < self.cache_duration

    def _get_cached(self, cache_key: str) -> list[PlayerInjury] | None:
        """Get cached data if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
        return None

    def _set_cache(self, cache_key: str, data: list[PlayerInjury]):
        """Set cache data."""
        self._cache[cache_key] = (datetime.now(), data)

    def fetch_balldontlie_injuries(self) -> list[PlayerInjury]:
        """
        Fetch injuries from Balldontlie API.

        Returns:
            List of PlayerInjury objects
        """
        cache_key = "balldontlie"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug("Using cached Balldontlie injuries")
            return cached

        if not self._balldontlie:
            return []

        injuries = []
        try:
            injury_data = self._balldontlie.get_injuries()

            for record in injury_data:
                player = record.get("player", {})
                player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
                player_id = str(player.get("id", ""))

                team_id_int = player.get("team_id")
                team_abbrev = self._team_id_map.get(team_id_int, "")

                status_str = record.get("status", "Unknown")
                injury_detail = record.get("description", "") or record.get("comment", "")

                # Extract injury type from description
                injury_type = ""
                if injury_detail:
                    match = re.search(r'\(([a-zA-Z\-\s]+)\)', injury_detail)
                    if match:
                        injury_type = match.group(1).strip().title()

                # Parse date
                date_str = record.get("date") or record.get("updated_at", "")
                report_date = None
                if date_str:
                    try:
                        if "T" in date_str:
                            report_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        else:
                            report_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        pass

                injury = PlayerInjury(
                    player_name=player_name,
                    player_id=player_id,
                    team=team_abbrev,
                    team_id=str(team_id_int) if team_id_int else None,
                    status=InjuryStatus.from_string(status_str),
                    injury_type=injury_type,
                    injury_detail=injury_detail,
                    report_date=report_date,
                    source="Balldontlie",
                    confidence=0.95,  # High confidence - structured API
                )
                injuries.append(injury)

            logger.info(f"Fetched {len(injuries)} injuries from Balldontlie")
            self._set_cache(cache_key, injuries)
            return injuries

        except Exception as e:
            logger.error(f"Balldontlie injury fetch error: {e}")
            return []

    def fetch_espn_injuries(self) -> list[PlayerInjury]:
        """
        Scrape injuries from ESPN injuries page.

        Returns:
            List of PlayerInjury objects
        """
        cache_key = "espn"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug("Using cached ESPN injuries")
            return cached

        injuries = []
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            response = requests.get(self.ESPN_INJURIES_URL, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find injury tables (ESPN structure: team sections with injury rows)
            team_sections = soup.find_all('div', class_='ResponsiveTable')

            current_team = ""
            for section in team_sections:
                # Get team name from header
                header = section.find_previous('div', class_='Table__Title')
                if header:
                    team_text = header.get_text(strip=True)
                    # Extract team abbrev from full name
                    current_team = NBA_TEAM_ABBREV.get(team_text, "")
                    if not current_team:
                        # Try partial match
                        for full_name, abbrev in NBA_TEAM_ABBREV.items():
                            if full_name in team_text or team_text in full_name:
                                current_team = abbrev
                                break

                # Find injury rows
                rows = section.find_all('tr', class_='Table__TR')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        # Parse player name
                        name_cell = cells[0]
                        player_link = name_cell.find('a')
                        player_name = player_link.get_text(strip=True) if player_link else name_cell.get_text(strip=True)

                        # Parse status
                        status_cell = cells[1] if len(cells) > 1 else None
                        status_text = status_cell.get_text(strip=True) if status_cell else ""

                        # Parse injury detail
                        detail_cell = cells[2] if len(cells) > 2 else None
                        injury_detail = detail_cell.get_text(strip=True) if detail_cell else ""

                        if player_name and player_name.lower() not in ['name', 'player']:
                            injury = PlayerInjury(
                                player_name=player_name,
                                team=current_team,
                                status=InjuryStatus.from_string(status_text),
                                injury_detail=injury_detail,
                                report_date=datetime.now(),
                                source="ESPN",
                                confidence=0.85,  # Slightly lower - web scraping
                            )
                            injuries.append(injury)

            logger.info(f"Fetched {len(injuries)} injuries from ESPN")
            self._set_cache(cache_key, injuries)
            return injuries

        except requests.RequestException as e:
            logger.error(f"ESPN injury fetch error: {e}")
            return []
        except Exception as e:
            logger.error(f"ESPN parsing error: {e}")
            return []

    def fetch_all_injuries(self, force_refresh: bool = False) -> list[PlayerInjury]:
        """
        Fetch injuries from all sources and merge.

        Args:
            force_refresh: Force fetch even if cache is valid

        Returns:
            Deduplicated list of PlayerInjury objects
        """
        if force_refresh:
            self._cache.clear()

        all_injuries: dict[str, PlayerInjury] = {}

        # Fetch from Balldontlie (primary - higher confidence)
        bdl_injuries = self.fetch_balldontlie_injuries()
        for injury in bdl_injuries:
            key = (injury.player_name.lower(), injury.team)
            if key not in all_injuries:
                all_injuries[key] = injury

        # Fetch from ESPN (secondary)
        espn_injuries = self.fetch_espn_injuries()
        for injury in espn_injuries:
            key = (injury.player_name.lower(), injury.team)
            if key not in all_injuries:
                all_injuries[key] = injury
            else:
                # Merge - prefer newer or higher confidence
                existing = all_injuries[key]
                if injury.confidence > existing.confidence:
                    all_injuries[key] = injury
                elif injury.report_date and existing.report_date:
                    if injury.report_date > existing.report_date:
                        all_injuries[key] = injury

        result = list(all_injuries.values())
        logger.info(f"Total unique injuries: {len(result)}")
        return result

    def get_team_injuries(self, team: str) -> list[PlayerInjury]:
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
        team_abbrev = NBA_TEAM_ABBREV.get(team, team_upper)

        return [
            inj for inj in all_injuries
            if inj.team.upper() == team_upper or inj.team.upper() == team_abbrev
        ]

    def get_player_injury(self, player_name: str) -> PlayerInjury | None:
        """
        Get injury status for a specific player.

        Args:
            player_name: Player's full name

        Returns:
            PlayerInjury if found, None otherwise
        """
        all_injuries = self.fetch_all_injuries()
        player_lower = player_name.lower().strip()

        for injury in all_injuries:
            if injury.player_name.lower() == player_lower:
                return injury
            # Partial match (last name)
            if player_lower.split()[-1] == injury.player_name.lower().split()[-1]:
                return injury

        return None

    def get_unavailable_players(self) -> list[PlayerInjury]:
        """
        Get all players who are definitively OUT.

        Returns:
            List of players with OUT status
        """
        all_injuries = self.fetch_all_injuries()
        return [
            inj for inj in all_injuries
            if inj.status in [InjuryStatus.OUT, InjuryStatus.DOUBTFUL]
        ]

    def get_game_time_decisions(self) -> list[PlayerInjury]:
        """
        Get all players who are game-time decisions.

        Returns:
            List of GTD/Questionable players
        """
        all_injuries = self.fetch_all_injuries()
        return [
            inj for inj in all_injuries
            if inj.status in [InjuryStatus.GTD, InjuryStatus.QUESTIONABLE]
        ]


# Singleton instance for convenience
_scraper_instance: InjuryScraper | None = None


def get_injury_scraper() -> InjuryScraper:
    """Get global InjuryScraper instance."""
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = InjuryScraper()
    return _scraper_instance


if __name__ == "__main__":
    # Test the scraper
    scraper = InjuryScraper()

    print("Fetching NBA injuries...")
    injuries = scraper.fetch_all_injuries()
    print(f"\nFound {len(injuries)} injured players")

    # Group by status
    by_status: dict[str, list] = {}
    for inj in injuries:
        status_name = inj.status.value
        if status_name not in by_status:
            by_status[status_name] = []
        by_status[status_name].append(inj)

    print("\nBy Status:")
    for status, players in sorted(by_status.items()):
        print(f"  {status}: {len(players)} players")

    # Show some examples
    print("\nSample OUT players:")
    out_players = [i for i in injuries if i.status == InjuryStatus.OUT][:5]
    for p in out_players:
        print(f"  {p.player_name} ({p.team}): {p.injury_detail}")
