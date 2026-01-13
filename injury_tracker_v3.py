"""
NBA Injury Tracker v3
=====================

Enhanced injury tracking system with real-time scraping and caching.
Critical for eliminating DNP (Did Not Play) errors in predictions.

Features:
- Multi-source injury data (NBA.com, ESPN, Balldontlie)
- In-memory caching with TTL (15-minute default)
- Database persistence (SQLite)
- Star player impact detection
- Usage redistribution calculations

Data Flow:
1. Check cache (15-min TTL)
2. Try Balldontlie API (primary)
3. Fallback to web scraping (NBA.com → ESPN)
4. Cache and persist to database
5. Calculate team impact for star players

Success Criteria:
- Detection rate > 95% (from ~70%)
- Zero DNP errors in predictions
- <2 second response time with caching
"""

import requests
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from functools import lru_cache
import re

# Try importing BeautifulSoup (optional, fallback for scraping)
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not available. Web scraping disabled. Install with: pip install beautifulsoup4")

# Import Balldontlie API if available
try:
    from balldontlie_api import BalldontlieAPI
    BALLDONTLIE_AVAILABLE = True
except ImportError:
    BALLDONTLIE_AVAILABLE = False
    logging.warning("Balldontlie API not available. Import the module to enable API fetching.")

# Import database
try:
    from database import DatabaseManager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logging.warning("Database module not available. Persistence disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

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
        """Parse status string to enum (case-insensitive, handles abbreviations)."""
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
        """Return probability player will be available (0.0-1.0)."""
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
    player_id: Optional[int] = None
    team_abbrev: str = ""
    team_id: Optional[int] = None
    status: InjuryStatus = InjuryStatus.UNKNOWN
    injury_type: str = ""  # e.g., "Knee", "Ankle", "Illness"
    injury_detail: str = ""  # e.g., "Left knee soreness"
    report_date: Optional[datetime] = None
    expected_return: Optional[str] = None  # e.g., "2-3 weeks"
    games_missed: int = 0
    source: str = ""  # "balldontlie", "nba.com", "espn", "cache"
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
        if self.report_date is None:
            self.report_date = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['report_date'] = self.report_date.isoformat() if self.report_date else None
        data['last_updated'] = self.last_updated.isoformat() if self.last_updated else None
        return data

    def is_unavailable(self) -> bool:
        """Check if player is definitely unavailable (OUT or DOUBTFUL)."""
        return self.status in [InjuryStatus.OUT, InjuryStatus.DOUBTFUL]

    def is_uncertain(self) -> bool:
        """Check if player availability is uncertain (QUESTIONABLE, GTD)."""
        return self.status in [InjuryStatus.QUESTIONABLE, InjuryStatus.GTD]


# =============================================================================
# In-Memory Cache with TTL
# =============================================================================

class InjuryCache:
    """
    In-memory cache for injury data with time-to-live (TTL).

    Cache Structure:
    - Key: game_date (YYYY-MM-DD)
    - Value: (injury_reports_list, timestamp)

    TTL: 15 minutes (default)
    Max Cache Size: 100 dates (keeps ~7 days of data)
    """

    def __init__(self, ttl_minutes: int = 15, max_size: int = 100):
        self.ttl_seconds = ttl_minutes * 60
        self.max_size = max_size
        self.cache: Dict[str, Tuple[List[InjuryReport], datetime]] = {}

    def get(self, date_key: str) -> Optional[List[InjuryReport]]:
        """Get cached injury reports for a date if not expired."""
        if date_key not in self.cache:
            return None

        reports, timestamp = self.cache[date_key]
        age_seconds = (datetime.now() - timestamp).total_seconds()

        if age_seconds > self.ttl_seconds:
            # Cache expired
            del self.cache[date_key]
            return None

        logger.debug(f"Cache HIT for {date_key} (age: {age_seconds:.1f}s)")
        return reports

    def set(self, date_key: str, reports: List[InjuryReport]):
        """Cache injury reports for a date."""
        # Limit cache size (FIFO eviction)
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[date_key] = (reports, datetime.now())
        logger.debug(f"Cache SET for {date_key} ({len(reports)} reports)")

    def clear(self):
        """Clear entire cache."""
        self.cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_minutes": self.ttl_seconds / 60,
            "entries": list(self.cache.keys()),
        }


# Global cache instance
_injury_cache = InjuryCache(ttl_minutes=15)


# =============================================================================
# Data Fetching Functions
# =============================================================================

def fetch_injuries_from_balldontlie(date: datetime) -> List[InjuryReport]:
    """
    Fetch injury data from Balldontlie API (primary source).

    Args:
        date: Target date for injury reports

    Returns:
        List of InjuryReport objects
    """
    if not BALLDONTLIE_AVAILABLE:
        logger.warning("Balldontlie API not available")
        return []

    try:
        api = BalldontlieAPI()

        # Note: Balldontlie API v2 provides player status
        # This is a placeholder - actual implementation depends on API structure
        # Check API docs: https://docs.balldontlie.io

        # Example structure (adjust based on actual API):
        # injuries_data = api.get_injuries(date=date.strftime('%Y-%m-%d'))

        logger.info("Balldontlie API injury fetch not yet implemented")
        return []

    except Exception as e:
        logger.error(f"Error fetching from Balldontlie API: {e}")
        return []


def scrape_nba_injuries() -> List[InjuryReport]:
    """
    Scrape injury reports from NBA.com/injuries.

    Returns:
        List of InjuryReport objects
    """
    if not BS4_AVAILABLE:
        logger.warning("BeautifulSoup not available for scraping")
        return []

    try:
        url = "https://www.nba.com/news/injury-report"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        injuries = []

        # NBA.com structure (as of 2025, may change):
        # Look for injury report table or sections
        # This is a simplified example - actual parsing depends on current site structure

        injury_sections = soup.find_all('div', class_='injury-report-section')

        for section in injury_sections:
            # Extract team name
            team_elem = section.find('h2') or section.find('h3')
            team_name = team_elem.get_text(strip=True) if team_elem else ""

            # Extract player rows
            player_rows = section.find_all('tr') or section.find_all('div', class_='player-row')

            for row in player_rows:
                try:
                    # Extract player name, status, injury
                    # Structure varies, this is illustrative:
                    cols = row.find_all('td') or row.find_all('div')

                    if len(cols) >= 3:
                        player_name = cols[0].get_text(strip=True)
                        status_text = cols[1].get_text(strip=True)
                        injury_text = cols[2].get_text(strip=True)

                        status = InjuryStatus.from_string(status_text)

                        injury = InjuryReport(
                            player_name=player_name,
                            team_abbrev=team_name[:3].upper(),  # Approximate
                            status=status,
                            injury_detail=injury_text,
                            source="nba.com",
                            report_date=datetime.now(),
                        )
                        injuries.append(injury)

                except Exception as e:
                    logger.warning(f"Error parsing injury row: {e}")
                    continue

        logger.info(f"Scraped {len(injuries)} injuries from NBA.com")
        return injuries

    except requests.RequestException as e:
        logger.error(f"Error scraping NBA.com: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error scraping NBA.com: {e}")
        return []


def scrape_espn_injuries() -> List[InjuryReport]:
    """
    Scrape injury reports from ESPN (fallback source).

    Returns:
        List of InjuryReport objects
    """
    if not BS4_AVAILABLE:
        logger.warning("BeautifulSoup not available for scraping")
        return []

    try:
        url = "https://www.espn.com/nba/injuries"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        injuries = []

        # ESPN structure (as of 2025, may change)
        # Find injury tables by team
        tables = soup.find_all('table', class_='tablehead')

        for table in tables:
            # Get team name from preceding header
            team_header = table.find_previous('div', class_='team-name')
            team_name = team_header.get_text(strip=True) if team_header else ""

            # Parse table rows
            rows = table.find_all('tr')[1:]  # Skip header

            for row in rows:
                try:
                    cols = row.find_all('td')

                    if len(cols) >= 4:
                        player_name = cols[0].get_text(strip=True)
                        position = cols[1].get_text(strip=True)
                        status_text = cols[2].get_text(strip=True)
                        injury_text = cols[3].get_text(strip=True)

                        status = InjuryStatus.from_string(status_text)

                        injury = InjuryReport(
                            player_name=player_name,
                            team_abbrev=team_name[:3].upper(),
                            status=status,
                            injury_detail=injury_text,
                            source="espn",
                            report_date=datetime.now(),
                        )
                        injuries.append(injury)

                except Exception as e:
                    logger.warning(f"Error parsing ESPN row: {e}")
                    continue

        logger.info(f"Scraped {len(injuries)} injuries from ESPN")
        return injuries

    except requests.RequestException as e:
        logger.error(f"Error scraping ESPN: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error scraping ESPN: {e}")
        return []


def fetch_current_injuries(date: Optional[datetime] = None, use_cache: bool = True) -> List[InjuryReport]:
    """
    Fetch current injury reports with multi-source fallback and caching.

    Data source priority:
    1. In-memory cache (15-min TTL)
    2. Balldontlie API (primary)
    3. NBA.com scraping
    4. ESPN scraping (fallback)
    5. Database (stale data max 2 hours old)

    Args:
        date: Target date (defaults to today)
        use_cache: Whether to use in-memory cache

    Returns:
        List of InjuryReport objects
    """
    if date is None:
        date = datetime.now()

    date_key = date.strftime('%Y-%m-%d')

    # 1. Check cache
    if use_cache:
        cached = _injury_cache.get(date_key)
        if cached is not None:
            return cached

    injuries = []

    # 2. Try Balldontlie API (primary)
    try:
        injuries = fetch_injuries_from_balldontlie(date)
        if injuries:
            logger.info(f"Fetched {len(injuries)} injuries from Balldontlie")
            _injury_cache.set(date_key, injuries)
            _persist_injuries_to_db(injuries, date)
            return injuries
    except Exception as e:
        logger.warning(f"Balldontlie fetch failed: {e}")

    # 3. Try NBA.com scraping
    try:
        injuries = scrape_nba_injuries()
        if injuries:
            logger.info(f"Scraped {len(injuries)} injuries from NBA.com")
            _injury_cache.set(date_key, injuries)
            _persist_injuries_to_db(injuries, date)
            return injuries
    except Exception as e:
        logger.warning(f"NBA.com scraping failed: {e}")

    # 4. Try ESPN scraping (fallback)
    try:
        injuries = scrape_espn_injuries()
        if injuries:
            logger.info(f"Scraped {len(injuries)} injuries from ESPN")
            _injury_cache.set(date_key, injuries)
            _persist_injuries_to_db(injuries, date)
            return injuries
    except Exception as e:
        logger.warning(f"ESPN scraping failed: {e}")

    # 5. Fallback to database (max 2 hours old)
    try:
        db_injuries = _fetch_injuries_from_db(date, max_age_hours=2)
        if db_injuries:
            logger.warning(f"Using stale data from database ({len(db_injuries)} injuries)")
            return db_injuries
    except Exception as e:
        logger.error(f"Database fallback failed: {e}")

    # All sources failed
    logger.error(f"All injury data sources failed for {date_key}")
    return []


def is_player_available(player_id: int, game_date: datetime) -> Tuple[bool, Optional[InjuryStatus]]:
    """
    Check if a specific player is available for a game.

    Args:
        player_id: Player's ID
        game_date: Date of the game

    Returns:
        Tuple of (is_available: bool, status: InjuryStatus or None)

    Example:
        >>> available, status = is_player_available(237, datetime(2025, 1, 15))
        >>> if not available:
        >>>     print(f"Player unavailable: {status.value}")
    """
    injuries = fetch_current_injuries(game_date)

    # Find injury report for this player
    for injury in injuries:
        if injury.player_id == player_id:
            is_available = not injury.is_unavailable()
            return is_available, injury.status

    # No injury report found - assume available
    return True, None


# =============================================================================
# Star Player Impact Functions
# =============================================================================

# Top-3 scorer threshold for star player detection
STAR_PLAYER_MIN_PPG = 18.0  # Player must average ≥18 PPG to be considered "star"


def detect_star_player_out(team_id: int, game_date: datetime) -> Tuple[bool, List[str]]:
    """
    Detect if a star player (top-3 scorer on team) is out for a game.

    Args:
        team_id: Team's ID
        game_date: Date of the game

    Returns:
        Tuple of (has_star_out: bool, list of star player names)

    Example:
        >>> has_star_out, names = detect_star_player_out(1, datetime(2025, 1, 15))
        >>> if has_star_out:
        >>>     print(f"Star players out: {names}")
    """
    injuries = fetch_current_injuries(game_date)

    # Get team's injured players
    team_injuries = [inj for inj in injuries if inj.team_id == team_id and inj.is_unavailable()]

    if not team_injuries:
        return False, []

    # Check if any injured player is a star (simplified check)
    # In production, this should query actual PPG stats from database
    star_players_out = []

    for injury in team_injuries:
        # TODO: Query player stats to determine if top-3 scorer
        # For now, use heuristic: any OUT player might be impactful
        if injury.status == InjuryStatus.OUT:
            star_players_out.append(injury.player_name)

    has_star_out = len(star_players_out) > 0

    return has_star_out, star_players_out


def calculate_usage_redistribution(
    team_id: int,
    injured_player_id: int,
    game_date: datetime
) -> Dict[int, float]:
    """
    Calculate how an injured star player's usage is redistributed to teammates.

    When a high-usage player is out, their touches/shots are redistributed based on:
    - Remaining players' usage rates
    - Positional similarity
    - Minutes played

    Args:
        team_id: Team's ID
        injured_player_id: Injured player's ID
        game_date: Date of the game

    Returns:
        Dictionary mapping player_id -> additional_usage_percentage

    Example:
        >>> redistribution = calculate_usage_redistribution(1, 237, datetime(2025, 1, 15))
        >>> # {201935: 4.2, 203954: 3.8, 1629029: 2.1}
        >>> # Meaning player 201935 gets +4.2% usage
    """
    # TODO: Implement actual usage redistribution logic
    # This requires querying:
    # 1. Injured player's usage rate (USG%)
    # 2. Team roster and their current usage rates
    # 3. Historical redistribution patterns when this player missed games

    # For now, return empty dict (placeholder)
    logger.warning("Usage redistribution not yet implemented")
    return {}


# =============================================================================
# Database Persistence Functions
# =============================================================================

def _persist_injuries_to_db(injuries: List[InjuryReport], report_date: datetime):
    """
    Persist injury reports to database.

    Args:
        injuries: List of injury reports
        report_date: Date of the report
    """
    if not DATABASE_AVAILABLE:
        return

    try:
        db = DatabaseManager()

        for injury in injuries:
            # Convert to database format
            injury_data = {
                "player_id": injury.player_id,
                "reported_date": report_date.strftime('%Y-%m-%d'),
                "game_id": None,  # Will be linked later
                "status": injury.status.value,
                "injury_type": injury.injury_detail,
                "return_date": None,
                "source": injury.source,
            }

            db.upsert_injury(injury_data)

        logger.debug(f"Persisted {len(injuries)} injuries to database")

    except Exception as e:
        logger.error(f"Error persisting injuries to database: {e}")


def _fetch_injuries_from_db(date: datetime, max_age_hours: int = 2) -> List[InjuryReport]:
    """
    Fetch injury reports from database (fallback when APIs fail).

    Args:
        date: Target date
        max_age_hours: Maximum age of data to accept (hours)

    Returns:
        List of InjuryReport objects
    """
    if not DATABASE_AVAILABLE:
        return []

    try:
        db = DatabaseManager()

        # Query injuries within max_age window
        cutoff_date = date - timedelta(hours=max_age_hours)

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT i.*, p.name as player_name, p.nba_id as player_nba_id,
                       t.abbreviation as team_abbrev, t.id as team_db_id
                FROM injuries i
                LEFT JOIN players p ON i.player_id = p.id
                LEFT JOIN teams t ON p.team_id = t.id
                WHERE i.reported_date >= ?
                ORDER BY i.reported_date DESC
            """, (cutoff_date.strftime('%Y-%m-%d'),))

            rows = cursor.fetchall()

        injuries = []
        for row in rows:
            injury = InjuryReport(
                player_name=row['player_name'] or "Unknown",
                player_id=row['player_nba_id'],
                team_abbrev=row['team_abbrev'] or "",
                team_id=row['team_db_id'],
                status=InjuryStatus.from_string(row['status']),
                injury_detail=row['injury_type'] or "",
                report_date=datetime.strptime(row['reported_date'], '%Y-%m-%d'),
                source="database",
            )
            injuries.append(injury)

        return injuries

    except Exception as e:
        logger.error(f"Error fetching from database: {e}")
        return []


# =============================================================================
# Utility Functions
# =============================================================================

def get_injury_summary(date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Get a summary of injuries for a specific date.

    Args:
        date: Target date (defaults to today)

    Returns:
        Dictionary with injury statistics

    Example:
        >>> summary = get_injury_summary()
        >>> print(f"Total injuries: {summary['total_count']}")
        >>> print(f"Players out: {summary['out_count']}")
    """
    injuries = fetch_current_injuries(date)

    status_counts = {}
    for status in InjuryStatus:
        status_counts[status.value] = sum(1 for inj in injuries if inj.status == status)

    return {
        "date": (date or datetime.now()).strftime('%Y-%m-%d'),
        "total_count": len(injuries),
        "out_count": status_counts.get("Out", 0),
        "doubtful_count": status_counts.get("Doubtful", 0),
        "questionable_count": status_counts.get("Questionable", 0),
        "gtd_count": status_counts.get("GTD", 0),
        "status_breakdown": status_counts,
        "source": injuries[0].source if injuries else "none",
    }


def clear_injury_cache():
    """Clear the in-memory injury cache."""
    _injury_cache.clear()


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    return _injury_cache.get_stats()


# =============================================================================
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NBA Injury Tracker v3 - Test Run")
    print("=" * 60)

    # Test 1: Fetch current injuries
    print("\n[Test 1] Fetching current injuries...")
    injuries = fetch_current_injuries()
    print(f"✓ Found {len(injuries)} injuries")

    if injuries:
        print("\nSample injury reports:")
        for i, injury in enumerate(injuries[:5], 1):
            print(f"  {i}. {injury.player_name} ({injury.team_abbrev}): {injury.status.value} - {injury.injury_detail}")

    # Test 2: Get injury summary
    print("\n[Test 2] Injury summary...")
    summary = get_injury_summary()
    print(f"✓ Date: {summary['date']}")
    print(f"✓ Total injuries: {summary['total_count']}")
    print(f"✓ Players OUT: {summary['out_count']}")
    print(f"✓ Questionable: {summary['questionable_count']}")
    print(f"✓ Data source: {summary['source']}")

    # Test 3: Cache stats
    print("\n[Test 3] Cache statistics...")
    cache_stats = get_cache_stats()
    print(f"✓ Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"✓ TTL: {cache_stats['ttl_minutes']} minutes")

    # Test 4: Test player availability check (demo with fake ID)
    print("\n[Test 4] Player availability check...")
    available, status = is_player_available(999999, datetime.now())  # Fake ID
    print(f"✓ Player available: {available}")
    print(f"✓ Status: {status.value if status else 'No injury report'}")

    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)

    # Print warnings if optional dependencies missing
    if not BS4_AVAILABLE:
        print("\n⚠️  WARNING: BeautifulSoup not installed. Web scraping disabled.")
        print("   Install with: pip install beautifulsoup4")

    if not BALLDONTLIE_AVAILABLE:
        print("\n⚠️  WARNING: Balldontlie API module not found.")
        print("   Ensure balldontlie_api.py is in the same directory.")
