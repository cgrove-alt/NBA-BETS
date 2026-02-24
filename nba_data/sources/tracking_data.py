"""
NBA Tracking Data Fetcher

Fetches granular play-by-play data, shot charts, and tracking metrics
for enhanced simulation accuracy.

=============================================================================
PHASE 0: GRANULAR DATA ACQUISITION
=============================================================================
This module provides:
1. fetch_pbp_historical(game_id) - Deep, detailed PBP for training
2. fetch_pbp_live(game_id) - Fast, lightweight PBP for live inference
3. fetch_shot_chart(game_id) - Shot location data with zone efficiency

Data flows into:
- PBPParser: Converts raw PBP to Possession objects
- ShotAtlas: Zone-based shooting efficiency heatmaps
- RotationTracker: Substitution patterns and lineup matrices
=============================================================================
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any
from enum import Enum

import numpy as np

# NBA API imports
try:
    from nba_api.stats.endpoints import (
        playbyplayv2,
        shotchartdetail,
        boxscoretraditionalv2,
        boxscoreadvancedv2,
        leaguegamefinder,
    )
    from nba_api.stats.static import teams, players
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False
    print("Warning: nba_api not installed. Some features will be unavailable.")

import requests

# Rate limiting
API_DELAY = 0.6  # seconds between NBA API calls

# Cache directory
CACHE_DIR = Path(__file__).parent / ".tracking_cache"
CACHE_DIR.mkdir(exist_ok=True)


# =============================================================================
# DATA CLASSES
# =============================================================================

class PlayType(Enum):
    """Types of plays in NBA play-by-play."""
    FIELD_GOAL_MADE = 1
    FIELD_GOAL_MISSED = 2
    FREE_THROW = 3
    REBOUND = 4
    TURNOVER = 5
    FOUL = 6
    VIOLATION = 7
    SUBSTITUTION = 8
    TIMEOUT = 9
    JUMP_BALL = 10
    EJECTION = 11
    PERIOD_BEGIN = 12
    PERIOD_END = 13
    UNKNOWN = 0


@dataclass
class Play:
    """Single play from play-by-play data."""
    game_id: str
    event_num: int
    event_type: PlayType
    period: int
    game_clock: str  # "MM:SS" format
    home_score: int
    away_score: int

    # Player info
    player_id: int | None = None
    player_name: str | None = None
    team_id: int | None = None

    # Play details
    description: str = ""
    action_type: str = ""

    # Shot-specific
    shot_type: str | None = None  # "2PT", "3PT"
    shot_made: bool | None = None
    shot_distance: int | None = None
    shot_x: float | None = None
    shot_y: float | None = None

    # Assist/rebound info
    assist_player_id: int | None = None
    assist_player_name: str | None = None

    @property
    def seconds_remaining(self) -> float:
        """Convert game clock to seconds remaining in period."""
        if not self.game_clock or self.game_clock == "":
            return 0.0
        try:
            parts = self.game_clock.split(":")
            if len(parts) == 2:
                minutes, seconds = int(parts[0]), float(parts[1])
                return minutes * 60 + seconds
        except (ValueError, IndexError):
            return 0.0
        return 0.0

    @property
    def total_seconds_elapsed(self) -> float:
        """Total seconds elapsed in the game."""
        period_seconds = (self.period - 1) * 720  # 12 min quarters
        if self.period > 4:  # Overtime
            period_seconds = 4 * 720 + (self.period - 5) * 300  # 5 min OT
        return period_seconds + (720 - self.seconds_remaining if self.period <= 4 else 300 - self.seconds_remaining)


@dataclass
class Possession:
    """A single possession with all associated plays."""
    game_id: str
    possession_num: int
    period: int
    start_time: float  # seconds remaining
    end_time: float
    team_id: int
    team_name: str

    # Outcome
    points_scored: int = 0
    outcome: str = ""  # "made_2pt", "made_3pt", "missed", "turnover", "foul"

    # Plays in this possession
    plays: list[Play] = field(default_factory=list)

    # Key players
    shooter_id: int | None = None
    shooter_name: str | None = None
    assist_player_id: int | None = None

    @property
    def duration(self) -> float:
        """Duration of possession in seconds."""
        return self.start_time - self.end_time


@dataclass
class ShotLocation:
    """Shot with court location data."""
    game_id: str
    player_id: int
    player_name: str
    team_id: int
    period: int
    game_clock: str

    # Location (in feet from basket, basket at 0,0)
    x: float  # -25 to 25 (sideline to sideline)
    y: float  # -5 to 47 (baseline to far 3pt line)

    # Shot info
    shot_type: str  # "2PT Field Goal", "3PT Field Goal"
    shot_zone_basic: str  # "Restricted Area", "Mid-Range", "Above the Break 3", etc.
    shot_zone_area: str  # "Center(C)", "Left Side(L)", "Right Side(R)"
    shot_zone_range: str  # "Less Than 8 ft.", "8-16 ft.", "16-24 ft.", "24+ ft."
    shot_distance: int  # feet

    # Result
    made: bool

    # Context
    action_type: str = ""  # "Jump Shot", "Layup Shot", "Dunk Shot"
    shot_attempted_flag: int = 1

    @property
    def zone_key(self) -> str:
        """Unique key for this shot zone."""
        return f"{self.shot_zone_basic}_{self.shot_zone_area}_{self.shot_zone_range}"


@dataclass
class PlayerZoneStats:
    """Player shooting stats by zone."""
    player_id: int
    player_name: str

    # Zone -> (made, attempted) mapping
    zone_stats: dict[str, tuple[int, int]] = field(default_factory=dict)

    def add_shot(self, zone_key: str, made: bool):
        """Add a shot to zone stats."""
        if zone_key not in self.zone_stats:
            self.zone_stats[zone_key] = (0, 0)
        m, a = self.zone_stats[zone_key]
        self.zone_stats[zone_key] = (m + (1 if made else 0), a + 1)

    def get_zone_pct(self, zone_key: str) -> float:
        """Get shooting percentage for a zone."""
        if zone_key not in self.zone_stats:
            return 0.0
        made, attempted = self.zone_stats[zone_key]
        return made / attempted if attempted > 0 else 0.0

    def get_zone_volume(self, zone_key: str) -> int:
        """Get total attempts in a zone."""
        if zone_key not in self.zone_stats:
            return 0
        return self.zone_stats[zone_key][1]


# =============================================================================
# HISTORICAL FETCHERS (Deep, for training)
# =============================================================================

def fetch_pbp_historical(game_id: str, use_cache: bool = True, max_retries: int = 3) -> list[Play]:
    """
    Fetch detailed play-by-play data for a historical game.

    This is the DEEP fetcher - gets all details for training purposes.
    Slower but comprehensive.

    Uses NBA CDN as primary source (more reliable), falls back to stats.nba.com.

    Args:
        game_id: NBA game ID (e.g., "0022400001")
        use_cache: Whether to use cached data if available
        max_retries: Number of retry attempts

    Returns:
        List of Play objects with full detail
    """
    # Check cache first
    cache_file = CACHE_DIR / f"pbp_{game_id}.json"
    if use_cache and cache_file.exists():
        try:
            with open(cache_file) as f:
                cached_data = json.load(f)
            # Try CDN format first, then stats.nba.com format
            if 'game' in cached_data:
                return _parse_pbp_cdn(cached_data, game_id)
            return _parse_pbp_from_cache(cached_data, game_id)
        except (json.JSONDecodeError, KeyError):
            pass  # Cache corrupted, fetch fresh

    # Try NBA CDN first (more reliable, richer data)
    plays = _fetch_pbp_from_cdn(game_id, use_cache, cache_file)
    if plays:
        return plays

    # Fallback to stats.nba.com
    return _fetch_pbp_from_stats(game_id, use_cache, cache_file, max_retries)


def _fetch_pbp_from_cdn(game_id: str, use_cache: bool, cache_file: Path) -> list[Play]:
    """Fetch PBP from NBA CDN (nba.com/game endpoint data source)."""
    # CDN URL format: https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json
    cdn_url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.nba.com/',
    }

    try:
        time.sleep(API_DELAY)
        response = requests.get(cdn_url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Cache the raw response
        if use_cache and data:
            with open(cache_file, 'w') as f:
                json.dump(data, f)

        return _parse_pbp_cdn(data, game_id)

    except Exception:
        # CDN may not have all games, fall through to stats API
        return []


def _fetch_pbp_from_stats(game_id: str, use_cache: bool, cache_file: Path, max_retries: int) -> list[Play]:
    """Fetch PBP from stats.nba.com API."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
    }

    url = f"https://stats.nba.com/stats/playbyplayv2?GameID={game_id}&StartPeriod=0&EndPeriod=0"

    for attempt in range(max_retries):
        time.sleep(API_DELAY * (attempt + 1))
        try:
            response = requests.get(url, headers=headers, timeout=45)
            response.raise_for_status()
            data = response.json()

            if not data or 'resultSets' not in data:
                continue

            if use_cache:
                with open(cache_file, 'w') as f:
                    json.dump(data, f)

            return _parse_pbp_response(data, game_id)

        except requests.exceptions.Timeout:
            print(f"Timeout fetching PBP for {game_id} (attempt {attempt + 1}/{max_retries})")
            continue
        except requests.exceptions.RequestException as e:
            print(f"Request error for {game_id}: {e}")
            continue
        except Exception as e:
            print(f"Error fetching PBP for {game_id}: {e}")
            break

    return []


def _parse_pbp_cdn(data: dict, game_id: str) -> list[Play]:
    """Parse NBA CDN play-by-play format (richer structure)."""
    plays = []

    game_data = data.get('game', {})
    actions = game_data.get('actions', [])

    for action in actions:
        try:
            # Map action type to PlayType
            action_type = action.get('actionType', '').lower()
            event_type = _map_cdn_action_type(action_type)

            play = Play(
                game_id=game_id,
                event_num=action.get('actionNumber', 0),
                event_type=event_type,
                period=action.get('period', 1),
                game_clock=action.get('clock', ''),
                home_score=action.get('scoreHome', 0) or 0,
                away_score=action.get('scoreAway', 0) or 0,
                player_id=action.get('personId'),
                player_name=action.get('playerNameI', ''),
                team_id=action.get('teamId'),
                description=action.get('description', ''),
                action_type=action_type,
            )

            # Shot-specific data
            if action_type in ('2pt', '3pt', 'freethrow'):
                play.shot_made = action.get('shotResult') == 'Made'
                play.shot_type = '3PT' if action_type == '3pt' else '2PT'
                play.shot_distance = action.get('shotDistance')
                play.shot_x = action.get('x')
                play.shot_y = action.get('y')

                # Assist info
                if action.get('assistPersonId'):
                    play.assist_player_id = action.get('assistPersonId')
                    play.assist_player_name = action.get('assistPlayerNameInitial', '')

            plays.append(play)

        except (KeyError, TypeError):
            continue

    return plays


def _map_cdn_action_type(action_type: str) -> PlayType:
    """Map CDN action type string to PlayType enum."""
    action_type = action_type.lower()

    if '2pt' in action_type or '3pt' in action_type:
        # Check if made or missed from the full action
        return PlayType.FIELD_GOAL_MADE  # Will be refined by shot_made field
    if 'freethrow' in action_type:
        return PlayType.FREE_THROW
    if 'rebound' in action_type:
        return PlayType.REBOUND
    if 'turnover' in action_type:
        return PlayType.TURNOVER
    if 'foul' in action_type:
        return PlayType.FOUL
    if 'substitution' in action_type:
        return PlayType.SUBSTITUTION
    if 'timeout' in action_type:
        return PlayType.TIMEOUT
    if 'jumpball' in action_type:
        return PlayType.JUMP_BALL
    if 'period' in action_type:
        if 'start' in action_type:
            return PlayType.PERIOD_BEGIN
        if 'end' in action_type:
            return PlayType.PERIOD_END

    return PlayType.UNKNOWN


def _parse_pbp_response(data: dict, game_id: str) -> list[Play]:
    """Parse NBA API play-by-play response into Play objects."""
    plays = []

    result_sets = data.get('resultSets', [])
    if not result_sets:
        return plays

    # First result set contains play-by-play
    pbp_data = result_sets[0]
    headers = pbp_data.get('headers', [])
    rows = pbp_data.get('rowSet', [])

    # Create header index mapping
    h = {name: i for i, name in enumerate(headers)}

    for row in rows:
        try:
            event_type = _map_event_type(row[h.get('EVENTMSGTYPE', 0)] if 'EVENTMSGTYPE' in h else 0)

            play = Play(
                game_id=game_id,
                event_num=row[h.get('EVENTNUM', 0)] if 'EVENTNUM' in h else 0,
                event_type=event_type,
                period=row[h.get('PERIOD', 0)] if 'PERIOD' in h else 1,
                game_clock=row[h.get('PCTIMESTRING', '')] if 'PCTIMESTRING' in h else "",
                home_score=row[h.get('SCOREHOME', 0)] if 'SCOREHOME' in h else 0,
                away_score=row[h.get('SCOREAWAY', 0)] if 'SCOREAWAY' in h else 0,
                player_id=row[h.get('PLAYER1_ID')] if 'PLAYER1_ID' in h else None,
                player_name=row[h.get('PLAYER1_NAME', '')] if 'PLAYER1_NAME' in h else None,
                team_id=row[h.get('PLAYER1_TEAM_ID')] if 'PLAYER1_TEAM_ID' in h else None,
                description=row[h.get('HOMEDESCRIPTION', '')] or row[h.get('VISITORDESCRIPTION', '')] or row[h.get('NEUTRALDESCRIPTION', '')] or "",
            )

            # Parse shot details from description
            if event_type in (PlayType.FIELD_GOAL_MADE, PlayType.FIELD_GOAL_MISSED):
                play.shot_made = event_type == PlayType.FIELD_GOAL_MADE
                play.shot_type = "3PT" if "3PT" in play.description.upper() else "2PT"

                # Extract distance if available
                import re
                dist_match = re.search(r"(\d+)'", play.description)
                if dist_match:
                    play.shot_distance = int(dist_match.group(1))

                # Check for assist
                if 'PLAYER2_ID' in h and row[h['PLAYER2_ID']]:
                    play.assist_player_id = row[h['PLAYER2_ID']]
                    play.assist_player_name = row[h.get('PLAYER2_NAME', '')]

            plays.append(play)

        except (IndexError, KeyError, TypeError):
            continue  # Skip malformed rows

    return plays


def _parse_pbp_from_cache(data: dict, game_id: str) -> list[Play]:
    """Parse cached PBP data."""
    return _parse_pbp_response(data, game_id)


def _map_event_type(event_code: int) -> PlayType:
    """Map NBA API event type code to PlayType enum."""
    mapping = {
        1: PlayType.FIELD_GOAL_MADE,
        2: PlayType.FIELD_GOAL_MISSED,
        3: PlayType.FREE_THROW,
        4: PlayType.REBOUND,
        5: PlayType.TURNOVER,
        6: PlayType.FOUL,
        7: PlayType.VIOLATION,
        8: PlayType.SUBSTITUTION,
        9: PlayType.TIMEOUT,
        10: PlayType.JUMP_BALL,
        11: PlayType.EJECTION,
        12: PlayType.PERIOD_BEGIN,
        13: PlayType.PERIOD_END,
    }
    return mapping.get(event_code, PlayType.UNKNOWN)


# =============================================================================
# LIVE FETCHERS (Fast, for inference)
# =============================================================================

def fetch_pbp_live(game_id: str) -> list[Play]:
    """
    Fetch lightweight play-by-play for live game inference.

    This is the FAST fetcher - minimal processing, no caching.
    Used during live games for real-time updates.

    Args:
        game_id: NBA game ID

    Returns:
        List of Play objects (lightweight, last 50 plays only)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.nba.com/',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
    }

    url = f"https://stats.nba.com/stats/playbyplayv2?GameID={game_id}&StartPeriod=0&EndPeriod=0"

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        return _parse_pbp_live(data, game_id)

    except Exception as e:
        print(f"Error fetching live PBP for {game_id}: {e}")
        return []


def _parse_pbp_live(data: dict, game_id: str) -> list[Play]:
    """Fast parse for live PBP - minimal fields."""
    plays = []

    result_sets = data.get('resultSets', [])
    if not result_sets:
        return plays

    pbp_data = result_sets[0]
    headers = pbp_data.get('headers', [])
    rows = pbp_data.get('rowSet', [])

    h = {name: i for i, name in enumerate(headers)}

    # Only parse last 50 plays for live context
    recent_rows = rows[-50:] if len(rows) > 50 else rows

    for row in recent_rows:
        try:
            plays.append(Play(
                game_id=game_id,
                event_num=row[h.get('EVENTNUM', 0)] if 'EVENTNUM' in h else 0,
                event_type=_map_event_type(row[h.get('EVENTMSGTYPE', 0)] if 'EVENTMSGTYPE' in h else 0),
                period=row[h.get('PERIOD', 0)] if 'PERIOD' in h else 1,
                game_clock=row[h.get('PCTIMESTRING', '')] if 'PCTIMESTRING' in h else "",
                home_score=int(row[h.get('SCOREHOME')] or 0) if 'SCOREHOME' in h else 0,
                away_score=int(row[h.get('SCOREAWAY')] or 0) if 'SCOREAWAY' in h else 0,
                player_id=row[h.get('PLAYER1_ID')] if 'PLAYER1_ID' in h else None,
                team_id=row[h.get('PLAYER1_TEAM_ID')] if 'PLAYER1_TEAM_ID' in h else None,
                description=str(row[h.get('HOMEDESCRIPTION', '')] or row[h.get('VISITORDESCRIPTION', '')] or ""),
            ))
        except (IndexError, KeyError, TypeError):
            continue

    return plays


# =============================================================================
# SHOT CHART FETCHER
# =============================================================================

def fetch_shot_chart(game_id: str, use_cache: bool = True) -> list[ShotLocation]:
    """
    Fetch shot chart data with court locations.

    Returns all shots from a game with (x, y) coordinates and zone info.

    Args:
        game_id: NBA game ID
        use_cache: Whether to use cached data

    Returns:
        List of ShotLocation objects
    """
    if not HAS_NBA_API:
        print("Error: nba_api required for fetch_shot_chart")
        return []

    # Check cache
    cache_file = CACHE_DIR / f"shots_{game_id}.json"
    if use_cache and cache_file.exists():
        try:
            with open(cache_file) as f:
                cached_data = json.load(f)
            return _parse_shot_chart_from_cache(cached_data, game_id)
        except (json.JSONDecodeError, KeyError):
            pass

    # Fetch from API
    time.sleep(API_DELAY)
    try:
        shots = shotchartdetail.ShotChartDetail(
            game_id_nullable=game_id,
            team_id=0,  # All teams
            player_id=0,  # All players
            context_measure_simple='FGA',
            timeout=30
        )
        data = shots.get_dict()

        # Cache response
        if use_cache:
            with open(cache_file, 'w') as f:
                json.dump(data, f)

        return _parse_shot_chart_response(data, game_id)

    except Exception as e:
        print(f"Error fetching shot chart for {game_id}: {e}")
        return []


def _parse_shot_chart_response(data: dict, game_id: str) -> list[ShotLocation]:
    """Parse NBA API shot chart response."""
    shots = []

    result_sets = data.get('resultSets', [])
    if not result_sets:
        return shots

    shot_data = result_sets[0]
    headers = shot_data.get('headers', [])
    rows = shot_data.get('rowSet', [])

    h = {name: i for i, name in enumerate(headers)}

    for row in rows:
        try:
            shot = ShotLocation(
                game_id=game_id,
                player_id=row[h['PLAYER_ID']],
                player_name=row[h['PLAYER_NAME']],
                team_id=row[h['TEAM_ID']],
                period=row[h['PERIOD']],
                game_clock=row[h.get('GAME_CLOCK', '')] if 'GAME_CLOCK' in h else "",
                x=row[h['LOC_X']] / 10.0,  # Convert to feet
                y=row[h['LOC_Y']] / 10.0,
                shot_type=row[h['SHOT_TYPE']],
                shot_zone_basic=row[h['SHOT_ZONE_BASIC']],
                shot_zone_area=row[h['SHOT_ZONE_AREA']],
                shot_zone_range=row[h['SHOT_ZONE_RANGE']],
                shot_distance=row[h['SHOT_DISTANCE']],
                made=row[h['SHOT_MADE_FLAG']] == 1,
                action_type=row[h.get('ACTION_TYPE', '')] if 'ACTION_TYPE' in h else "",
            )
            shots.append(shot)

        except (IndexError, KeyError, TypeError):
            continue

    return shots


def _parse_shot_chart_from_cache(data: dict, game_id: str) -> list[ShotLocation]:
    """Parse cached shot chart data."""
    return _parse_shot_chart_response(data, game_id)


# =============================================================================
# BATCH HISTORICAL FETCHER
# =============================================================================

def fetch_season_games(season: str = "2024-25", team_id: int | None = None) -> list[str]:
    """
    Get list of game IDs for a season.

    Args:
        season: NBA season (e.g., "2024-25")
        team_id: Optional team to filter by

    Returns:
        List of game IDs
    """
    if not HAS_NBA_API:
        return []

    time.sleep(API_DELAY)
    try:
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            team_id_nullable=team_id,
            league_id_nullable="00",
            timeout=30
        )
        data = finder.get_dict()

        result_sets = data.get('resultSets', [])
        if not result_sets:
            return []

        games_data = result_sets[0]
        headers = games_data.get('headers', [])
        rows = games_data.get('rowSet', [])

        h = {name: i for i, name in enumerate(headers)}

        # Get unique game IDs
        game_ids = set()
        for row in rows:
            game_id = row[h['GAME_ID']]
            game_ids.add(game_id)

        return sorted(game_ids)

    except Exception as e:
        print(f"Error fetching season games: {e}")
        return []


def fetch_pbp_batch(
    game_ids: list[str],
    max_games: int = 100,
    progress_callback: Callable | None = None
) -> dict[str, list[Play]]:
    """
    Fetch PBP for multiple games (with rate limiting).

    Args:
        game_ids: List of game IDs to fetch
        max_games: Maximum games to fetch
        progress_callback: Optional callback(current, total) for progress

    Returns:
        Dict mapping game_id -> List[Play]
    """
    results = {}
    games_to_fetch = game_ids[:max_games]

    for i, game_id in enumerate(games_to_fetch):
        if progress_callback:
            progress_callback(i + 1, len(games_to_fetch))

        plays = fetch_pbp_historical(game_id, use_cache=True)
        if plays:
            results[game_id] = plays

        # Extra delay for batch to avoid rate limits
        time.sleep(API_DELAY * 1.5)

    return results


def fetch_shot_charts_batch(
    game_ids: list[str],
    max_games: int = 100,
    progress_callback: Callable | None = None
) -> dict[str, list[ShotLocation]]:
    """
    Fetch shot charts for multiple games.

    Args:
        game_ids: List of game IDs
        max_games: Maximum games to fetch
        progress_callback: Optional callback for progress

    Returns:
        Dict mapping game_id -> List[ShotLocation]
    """
    results = {}
    games_to_fetch = game_ids[:max_games]

    for i, game_id in enumerate(games_to_fetch):
        if progress_callback:
            progress_callback(i + 1, len(games_to_fetch))

        shots = fetch_shot_chart(game_id, use_cache=True)
        if shots:
            results[game_id] = shots

        time.sleep(API_DELAY * 1.5)

    return results


# =============================================================================
# DATA PROCESSING CLASSES
# =============================================================================

class PBPParser:
    """
    Parse play-by-play data into structured Possession objects.

    Converts raw Play events into possessions with:
    - Start/end times
    - Outcome (made shot, missed shot, turnover, foul)
    - Key players (shooter, assister, rebounder)
    - Points scored

    Usage:
        parser = PBPParser()
        possessions = parser.parse_game(plays)
    """

    # Events that end a possession
    POSSESSION_ENDING_EVENTS = {
        PlayType.FIELD_GOAL_MADE,
        PlayType.TURNOVER,
        PlayType.PERIOD_END,
    }

    # Events that might end possession (depends on context)
    CONDITIONAL_ENDING_EVENTS = {
        PlayType.FIELD_GOAL_MISSED,  # Ends if defensive rebound
        PlayType.FREE_THROW,  # Ends on last FT make or defensive rebound
    }

    def __init__(self):
        self.possessions: list[Possession] = []
        self._current_possession: dict | None = None
        self._possession_count = 0

    def parse_game(self, plays: list[Play]) -> list[Possession]:
        """
        Parse all plays from a game into possessions.

        Args:
            plays: List of Play objects from fetch_pbp_historical

        Returns:
            List of Possession objects
        """
        if not plays:
            return []

        self.possessions = []
        self._possession_count = 0
        self._current_possession = None

        game_id = plays[0].game_id

        for i, play in enumerate(plays):
            # Skip non-game events
            if play.event_type in (PlayType.PERIOD_BEGIN, PlayType.TIMEOUT,
                                   PlayType.EJECTION, PlayType.UNKNOWN):
                continue

            # Start new possession if needed
            if self._current_possession is None:
                self._start_possession(play, game_id)

            # Add play to current possession
            self._current_possession['plays'].append(play)

            # Check if possession ends
            if self._is_possession_end(play, plays, i):
                self._end_possession(play)

        # Close any remaining possession
        if self._current_possession:
            self._end_possession(plays[-1] if plays else None)

        return self.possessions

    def _start_possession(self, play: Play, game_id: str):
        """Start tracking a new possession."""
        self._possession_count += 1
        self._current_possession = {
            'game_id': game_id,
            'possession_num': self._possession_count,
            'period': play.period,
            'start_time': play.seconds_remaining,
            'team_id': play.team_id,
            'plays': [],
        }

    def _end_possession(self, final_play: Play | None):
        """End the current possession and create Possession object."""
        if not self._current_possession:
            return

        poss = self._current_possession
        plays = poss['plays']

        # Determine outcome and points
        outcome, points, shooter_id, shooter_name, assist_id = self._analyze_outcome(plays)

        # Get team name from plays
        team_name = ""
        for p in plays:
            if p.team_id == poss['team_id'] and p.player_name:
                team_name = p.player_name.split()[-1] if p.player_name else ""
                break

        possession = Possession(
            game_id=poss['game_id'],
            possession_num=poss['possession_num'],
            period=poss['period'],
            start_time=poss['start_time'],
            end_time=final_play.seconds_remaining if final_play else 0,
            team_id=poss['team_id'] or 0,
            team_name=team_name,
            points_scored=points,
            outcome=outcome,
            plays=plays,
            shooter_id=shooter_id,
            shooter_name=shooter_name,
            assist_player_id=assist_id,
        )

        self.possessions.append(possession)
        self._current_possession = None

    def _is_possession_end(self, play: Play, all_plays: list[Play], idx: int) -> bool:
        """Determine if this play ends the current possession."""
        # Definite possession enders
        if play.event_type in self.POSSESSION_ENDING_EVENTS:
            return True

        # Made field goal always ends possession
        if play.event_type == PlayType.FIELD_GOAL_MADE:
            return True

        # Missed shot - check for defensive rebound
        if play.event_type == PlayType.FIELD_GOAL_MISSED:
            # Look ahead for rebound
            for j in range(idx + 1, min(idx + 5, len(all_plays))):
                next_play = all_plays[j]
                if next_play.event_type == PlayType.REBOUND:
                    # Defensive rebound ends possession
                    if next_play.team_id != play.team_id:
                        return True
                    # Offensive rebound continues possession
                    return False
            # No rebound found, assume possession ended
            return True

        # Free throws - check if last FT
        if play.event_type == PlayType.FREE_THROW:
            desc = play.description.upper() if play.description else ""
            # Check for "X of Y" pattern
            if "OF 1" in desc or "1 OF 1" in desc:
                return True
            if "2 OF 2" in desc or "3 OF 3" in desc:
                return True
            # Technical FT
            if "TECHNICAL" in desc:
                return False  # Technical FTs don't end possession

        # Foul - depends on type (shooting foul vs offensive)
        if play.event_type == PlayType.FOUL:
            desc = play.description.upper() if play.description else ""
            if "OFFENSIVE" in desc or "OFF." in desc:
                return True

        return False

    def _analyze_outcome(self, plays: list[Play]) -> tuple[str, int, int | None, str | None, int | None]:
        """
        Analyze plays to determine possession outcome.

        Returns:
            (outcome, points, shooter_id, shooter_name, assist_player_id)
        """
        points = 0
        outcome = "unknown"
        shooter_id = None
        shooter_name = None
        assist_id = None

        for play in plays:
            # Made field goal
            if play.event_type == PlayType.FIELD_GOAL_MADE or (play.shot_made is True):
                if play.shot_type == "3PT":
                    points = 3
                    outcome = "made_3pt"
                else:
                    points = 2
                    outcome = "made_2pt"
                shooter_id = play.player_id
                shooter_name = play.player_name
                assist_id = play.assist_player_id
                break

            # Missed field goal
            if play.event_type == PlayType.FIELD_GOAL_MISSED or (play.shot_made is False and play.shot_type):
                outcome = "missed"
                shooter_id = play.player_id
                shooter_name = play.player_name

            # Turnover
            if play.event_type == PlayType.TURNOVER:
                outcome = "turnover"
                break

            # Free throws
            if play.event_type == PlayType.FREE_THROW:
                desc = play.description.upper() if play.description else ""
                if "MISS" not in desc:
                    points += 1
                outcome = "free_throws"
                shooter_id = play.player_id
                shooter_name = play.player_name

        return outcome, points, shooter_id, shooter_name, assist_id

    def get_possession_stats(self) -> dict[str, Any]:
        """Get summary statistics for parsed possessions."""
        if not self.possessions:
            return {}

        total = len(self.possessions)
        made_2pt = len([p for p in self.possessions if p.outcome == "made_2pt"])
        made_3pt = len([p for p in self.possessions if p.outcome == "made_3pt"])
        turnovers = len([p for p in self.possessions if p.outcome == "turnover"])
        missed = len([p for p in self.possessions if p.outcome == "missed"])

        total_points = sum(p.points_scored for p in self.possessions)
        avg_duration = np.mean([p.duration for p in self.possessions if p.duration > 0])

        return {
            'total_possessions': total,
            'made_2pt': made_2pt,
            'made_3pt': made_3pt,
            'turnovers': turnovers,
            'missed_shots': missed,
            'total_points': total_points,
            'points_per_possession': total_points / total if total > 0 else 0,
            'avg_possession_duration': avg_duration,
            'turnover_rate': turnovers / total if total > 0 else 0,
        }


class ShotAtlas:
    """
    Zone-based shooting efficiency heatmap for players.

    Creates a spatial representation of shooting efficiency that can be used
    to upgrade simulation accuracy from season FG% to zone-specific percentages.

    Zones are based on NBA shot chart zones:
    - Restricted Area (0-4 ft)
    - In The Paint (Non-RA) (4-14 ft)
    - Mid-Range (14-22 ft)
    - Above the Break 3 (22+ ft center)
    - Left/Right Corner 3 (22 ft corners)

    Usage:
        atlas = ShotAtlas()
        atlas.add_shots(shot_chart_data)
        efficiency = atlas.get_player_zone_efficiency(player_id)
    """

    # Standard NBA zones
    ZONES = [
        'Restricted Area',
        'In The Paint (Non-RA)',
        'Mid-Range',
        'Left Corner 3',
        'Right Corner 3',
        'Above the Break 3',
        'Backcourt',
    ]

    # Zone areas (left, center, right)
    ZONE_AREAS = ['Left Side(L)', 'Center(C)', 'Right Side(R)',
                  'Left Side Center(LC)', 'Right Side Center(RC)']

    def __init__(self):
        # player_id -> zone -> (made, attempted)
        self.player_zones: dict[int, dict[str, tuple[int, int]]] = {}
        # player_id -> player_name
        self.player_names: dict[int, str] = {}
        # team_id -> zone -> (made, attempted)
        self.team_zones: dict[int, dict[str, tuple[int, int]]] = {}
        # League averages by zone
        self.league_zones: dict[str, tuple[int, int]] = {}

    def add_shots(self, shots: list[ShotLocation]):
        """
        Add shot chart data to the atlas.

        Args:
            shots: List of ShotLocation objects
        """
        for shot in shots:
            self._add_shot(shot)

    def _add_shot(self, shot: ShotLocation):
        """Add a single shot to all tracking dicts."""
        zone = shot.shot_zone_basic
        player_id = shot.player_id
        team_id = shot.team_id
        made = 1 if shot.made else 0

        # Player zones
        if player_id not in self.player_zones:
            self.player_zones[player_id] = {}
            self.player_names[player_id] = shot.player_name

        if zone not in self.player_zones[player_id]:
            self.player_zones[player_id][zone] = (0, 0)

        m, a = self.player_zones[player_id][zone]
        self.player_zones[player_id][zone] = (m + made, a + 1)

        # Team zones
        if team_id not in self.team_zones:
            self.team_zones[team_id] = {}

        if zone not in self.team_zones[team_id]:
            self.team_zones[team_id][zone] = (0, 0)

        m, a = self.team_zones[team_id][zone]
        self.team_zones[team_id][zone] = (m + made, a + 1)

        # League zones
        if zone not in self.league_zones:
            self.league_zones[zone] = (0, 0)

        m, a = self.league_zones[zone]
        self.league_zones[zone] = (m + made, a + 1)

    def get_player_zone_efficiency(
        self,
        player_id: int,
        min_attempts: int = 3,
        use_league_fallback: bool = True
    ) -> dict[str, float]:
        """
        Get shooting efficiency by zone for a player.

        Args:
            player_id: Player ID
            min_attempts: Minimum attempts to return zone-specific %
            use_league_fallback: Use league average for zones with low attempts

        Returns:
            Dict mapping zone -> shooting percentage
        """
        result = {}
        player_data = self.player_zones.get(player_id, {})

        for zone in self.ZONES:
            if zone in player_data:
                made, attempted = player_data[zone]
                if attempted >= min_attempts:
                    result[zone] = made / attempted
                elif use_league_fallback and zone in self.league_zones:
                    lm, la = self.league_zones[zone]
                    result[zone] = lm / la if la > 0 else 0.0
            elif use_league_fallback and zone in self.league_zones:
                lm, la = self.league_zones[zone]
                result[zone] = lm / la if la > 0 else 0.0

        return result

    def get_player_shot_distribution(self, player_id: int) -> dict[str, float]:
        """
        Get shot distribution by zone for a player (where they shoot from).

        Args:
            player_id: Player ID

        Returns:
            Dict mapping zone -> percentage of total shots
        """
        player_data = self.player_zones.get(player_id, {})
        total_shots = sum(a for _, a in player_data.values())

        if total_shots == 0:
            return {}

        return {
            zone: attempted / total_shots
            for zone, (_, attempted) in player_data.items()
        }

    def get_team_zone_efficiency(self, team_id: int) -> dict[str, float]:
        """Get team shooting efficiency by zone."""
        team_data = self.team_zones.get(team_id, {})
        return {
            zone: made / attempted if attempted > 0 else 0.0
            for zone, (made, attempted) in team_data.items()
        }

    def get_league_averages(self) -> dict[str, float]:
        """Get league average shooting by zone."""
        return {
            zone: made / attempted if attempted > 0 else 0.0
            for zone, (made, attempted) in self.league_zones.items()
        }

    def get_hot_zones(self, player_id: int, threshold: float = 0.05) -> list[str]:
        """
        Get zones where player shoots above league average.

        Args:
            player_id: Player ID
            threshold: How much above league average to be considered "hot"

        Returns:
            List of zone names where player is hot
        """
        player_eff = self.get_player_zone_efficiency(player_id, use_league_fallback=False)
        league_eff = self.get_league_averages()

        hot_zones = []
        for zone, pct in player_eff.items():
            league_pct = league_eff.get(zone, 0)
            if pct > league_pct + threshold:
                hot_zones.append(zone)

        return hot_zones

    def get_cold_zones(self, player_id: int, threshold: float = 0.05) -> list[str]:
        """Get zones where player shoots below league average."""
        player_eff = self.get_player_zone_efficiency(player_id, use_league_fallback=False)
        league_eff = self.get_league_averages()

        cold_zones = []
        for zone, pct in player_eff.items():
            league_pct = league_eff.get(zone, 0)
            if pct < league_pct - threshold:
                cold_zones.append(zone)

        return cold_zones

    def to_simulation_input(self, player_id: int) -> dict[str, Any]:
        """
        Convert zone data to simulation engine input format.

        Returns dict compatible with PlayerTrackingStats.
        """
        zones = self.get_player_zone_efficiency(player_id)
        distribution = self.get_player_shot_distribution(player_id)

        return {
            'zone_fg_pct': zones,
            'zone_distribution': distribution,
            'hot_zones': self.get_hot_zones(player_id),
            'cold_zones': self.get_cold_zones(player_id),
        }


@dataclass
class LineupSpell:
    """A continuous period with the same 5 players on court."""
    team_id: int
    player_ids: tuple[int, ...]  # Sorted tuple of 5 player IDs
    period: int
    start_time: float  # seconds remaining
    end_time: float

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.start_time - self.end_time

    @property
    def minutes(self) -> float:
        """Duration in minutes."""
        return self.duration / 60.0


class RotationTracker:
    """
    Track player rotations and substitution patterns from PBP data.

    Derives:
    - Who plays with whom (lineup combinations)
    - Minutes per lineup
    - Substitution timing patterns
    - Starter vs bench identification

    Usage:
        tracker = RotationTracker()
        tracker.process_game(plays, home_team_id, away_team_id)
        lineups = tracker.get_lineups(team_id)
    """

    def __init__(self):
        # team_id -> list of LineupSpell
        self.lineup_spells: dict[int, list[LineupSpell]] = {}
        # team_id -> set of starter player_ids
        self.starters: dict[int, set] = {}
        # player_id -> total minutes
        self.player_minutes: dict[int, float] = {}
        # (player_id, player_id) -> minutes together
        self.player_pairs: dict[tuple[int, int], float] = {}
        # team_id -> current lineup (set of player_ids)
        self._current_lineups: dict[int, set] = {}
        # team_id -> when current lineup started
        self._lineup_start: dict[int, tuple[int, float]] = {}  # (period, time)

    def process_game(
        self,
        plays: list[Play],
        home_team_id: int,
        away_team_id: int
    ):
        """
        Process game PBP to extract rotation data.

        Args:
            plays: List of Play objects
            home_team_id: Home team ID
            away_team_id: Away team ID
        """
        self.lineup_spells = {home_team_id: [], away_team_id: []}
        self.starters = {home_team_id: set(), away_team_id: set()}
        self._current_lineups = {home_team_id: set(), away_team_id: set()}
        self._lineup_start = {}


        for play in plays:
            # Track period changes
            if play.event_type == PlayType.PERIOD_BEGIN:
                # Reset lineups at period start (starters may change)
                for team_id in [home_team_id, away_team_id]:
                    if self._current_lineups[team_id]:
                        self._end_lineup_spell(team_id, play.period, play.seconds_remaining)
                continue

            if play.event_type == PlayType.PERIOD_END:
                # End current lineup spells
                for team_id in [home_team_id, away_team_id]:
                    if self._current_lineups[team_id]:
                        self._end_lineup_spell(team_id, play.period, play.seconds_remaining)
                continue

            # Process substitutions
            if play.event_type == PlayType.SUBSTITUTION:
                self._process_substitution(play, home_team_id, away_team_id)
                continue

            # Track players in action (to identify who's on court)
            if play.player_id and play.team_id in [home_team_id, away_team_id]:
                team_id = play.team_id
                player_id = play.player_id

                # Add to current lineup if not already there
                if player_id not in self._current_lineups[team_id]:
                    self._current_lineups[team_id].add(player_id)

                    # If this is period 1 and early, they're a starter
                    if play.period == 1 and play.seconds_remaining > 660:  # First 60 seconds
                        self.starters[team_id].add(player_id)

                    # Start lineup spell if we have 5 players
                    if len(self._current_lineups[team_id]) == 5:
                        self._start_lineup_spell(team_id, play.period, play.seconds_remaining)

        # Calculate derived stats
        self._calculate_player_minutes()
        self._calculate_pair_minutes()

    def _process_substitution(self, play: Play, home_team_id: int, away_team_id: int):
        """Process a substitution event."""
        # Substitution description usually contains "SUB: X FOR Y"
        desc = play.description.upper() if play.description else ""
        team_id = play.team_id

        if team_id not in [home_team_id, away_team_id]:
            return

        # End current lineup spell
        if len(self._current_lineups[team_id]) == 5:
            self._end_lineup_spell(team_id, play.period, play.seconds_remaining)

        # Parse substitution - player entering is usually PLAYER1, exiting is PLAYER2
        # This varies by data source, so we also parse description
        if "SUB:" in desc or "SUBSTITUTION" in desc:
            # Try to parse from description
            pass  # Complex parsing, rely on player_id tracking instead

        # The play's player_id is typically the player entering
        if play.player_id:
            self._current_lineups[team_id].add(play.player_id)

        # For now, we'll let subsequent plays naturally rebuild the lineup
        # This is more robust than trying to parse complex substitution descriptions

    def _start_lineup_spell(self, team_id: int, period: int, time_remaining: float):
        """Start tracking a new lineup spell."""
        self._lineup_start[team_id] = (period, time_remaining)

    def _end_lineup_spell(self, team_id: int, period: int, time_remaining: float):
        """End current lineup spell and record it."""
        if team_id not in self._lineup_start:
            return

        start_period, start_time = self._lineup_start[team_id]

        # Only record if same period (cross-period is handled separately)
        if start_period == period and len(self._current_lineups[team_id]) == 5:
            lineup = tuple(sorted(self._current_lineups[team_id]))
            spell = LineupSpell(
                team_id=team_id,
                player_ids=lineup,
                period=period,
                start_time=start_time,
                end_time=time_remaining,
            )

            if spell.duration > 0:
                self.lineup_spells[team_id].append(spell)

        del self._lineup_start[team_id]

    def _calculate_player_minutes(self):
        """Calculate total minutes for each player."""
        self.player_minutes = {}

        for _team_id, spells in self.lineup_spells.items():
            for spell in spells:
                for player_id in spell.player_ids:
                    if player_id not in self.player_minutes:
                        self.player_minutes[player_id] = 0.0
                    self.player_minutes[player_id] += spell.minutes

    def _calculate_pair_minutes(self):
        """Calculate minutes each pair of players played together."""
        self.player_pairs = {}

        for _team_id, spells in self.lineup_spells.items():
            for spell in spells:
                # Generate all pairs from the 5-player lineup
                players = list(spell.player_ids)
                for i in range(len(players)):
                    for j in range(i + 1, len(players)):
                        pair = (min(players[i], players[j]), max(players[i], players[j]))
                        if pair not in self.player_pairs:
                            self.player_pairs[pair] = 0.0
                        self.player_pairs[pair] += spell.minutes

    def get_lineups(self, team_id: int) -> list[LineupSpell]:
        """Get all lineup spells for a team."""
        return self.lineup_spells.get(team_id, [])

    def get_lineup_minutes(self, team_id: int) -> dict[tuple[int, ...], float]:
        """
        Get total minutes for each unique lineup.

        Returns:
            Dict mapping lineup tuple -> total minutes
        """
        lineup_mins = {}

        for spell in self.lineup_spells.get(team_id, []):
            if spell.player_ids not in lineup_mins:
                lineup_mins[spell.player_ids] = 0.0
            lineup_mins[spell.player_ids] += spell.minutes

        return lineup_mins

    def get_most_used_lineups(self, team_id: int, n: int = 5) -> list[tuple[tuple[int, ...], float]]:
        """Get the N most-used lineups by minutes."""
        lineup_mins = self.get_lineup_minutes(team_id)
        sorted_lineups = sorted(lineup_mins.items(), key=lambda x: -x[1])
        return sorted_lineups[:n]

    def get_player_minutes(self, player_id: int) -> float:
        """Get total minutes for a player."""
        return self.player_minutes.get(player_id, 0.0)

    def get_pair_minutes(self, player1_id: int, player2_id: int) -> float:
        """Get minutes two players played together."""
        pair = (min(player1_id, player2_id), max(player1_id, player2_id))
        return self.player_pairs.get(pair, 0.0)

    def get_starters(self, team_id: int) -> set:
        """Get identified starters for a team."""
        return self.starters.get(team_id, set())

    def to_simulation_input(self, team_id: int) -> dict[str, Any]:
        """
        Convert rotation data to simulation engine input format.
        """
        lineup_mins = self.get_lineup_minutes(team_id)
        total_mins = sum(lineup_mins.values())

        # Convert to lineup probabilities
        lineup_probs = {
            lineup: mins / total_mins if total_mins > 0 else 0
            for lineup, mins in lineup_mins.items()
        }

        return {
            'lineup_probabilities': lineup_probs,
            'starters': list(self.starters.get(team_id, set())),
            'player_minutes': {
                pid: mins for pid, mins in self.player_minutes.items()
                if pid in set().union(*[set(l) for l in lineup_mins])
            },
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def build_player_zone_stats(shots: list[ShotLocation]) -> dict[int, PlayerZoneStats]:
    """
    Build zone shooting stats from shot chart data.

    Args:
        shots: List of ShotLocation objects

    Returns:
        Dict mapping player_id -> PlayerZoneStats
    """
    player_stats = {}

    for shot in shots:
        if shot.player_id not in player_stats:
            player_stats[shot.player_id] = PlayerZoneStats(
                player_id=shot.player_id,
                player_name=shot.player_name
            )

        player_stats[shot.player_id].add_shot(shot.zone_key, shot.made)

    return player_stats


def get_zone_efficiency_matrix(
    player_zone_stats: PlayerZoneStats,
    min_attempts: int = 5
) -> dict[str, float]:
    """
    Get efficiency by zone for a player.

    Args:
        player_zone_stats: Player's zone stats
        min_attempts: Minimum attempts to include zone

    Returns:
        Dict mapping zone_key -> shooting percentage
    """
    efficiency = {}

    for zone_key, (made, attempted) in player_zone_stats.zone_stats.items():
        if attempted >= min_attempts:
            efficiency[zone_key] = made / attempted

    return efficiency


def clear_cache(older_than_days: int = 30):
    """Clear cached data older than specified days."""
    cutoff = datetime.now().timestamp() - (older_than_days * 86400)

    cleared = 0
    for cache_file in CACHE_DIR.glob("*.json"):
        if cache_file.stat().st_mtime < cutoff:
            cache_file.unlink()
            cleared += 1

    print(f"Cleared {cleared} cached files older than {older_than_days} days")


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  NBA Tracking Data Fetcher - Test")
    print("=" * 60)

    # Test with a sample game (adjust game_id as needed)
    test_game_id = "0022400001"  # First game of 2024-25 season

    print(f"\n1. Fetching PBP for game {test_game_id}...")
    plays = fetch_pbp_historical(test_game_id)
    print(f"   Retrieved {len(plays)} plays")

    if plays:
        # Show some stats
        shots = [p for p in plays if p.event_type in (PlayType.FIELD_GOAL_MADE, PlayType.FIELD_GOAL_MISSED)]
        made = len([s for s in shots if s.shot_made])
        print(f"   Shots: {made}/{len(shots)} ({100*made/len(shots):.1f}%)" if shots else "   No shots found")

    print(f"\n2. Fetching shot chart for game {test_game_id}...")
    shot_chart = fetch_shot_chart(test_game_id)
    print(f"   Retrieved {len(shot_chart)} shots with location data")

    if shot_chart:
        # Build zone stats
        zone_stats = build_player_zone_stats(shot_chart)
        print(f"   Players with zone data: {len(zone_stats)}")

        # Show top shooter's zones
        if zone_stats:
            top_player = max(zone_stats.values(), key=lambda p: sum(s[1] for s in p.zone_stats.values()))
            print(f"   Most shots: {top_player.player_name}")
            for zone, (m, a) in sorted(top_player.zone_stats.items(), key=lambda x: -x[1][1])[:3]:
                print(f"     {zone}: {m}/{a} ({100*m/a:.1f}%)")

    print("\n" + "=" * 60)
    print("  Test complete!")
    print("=" * 60)
