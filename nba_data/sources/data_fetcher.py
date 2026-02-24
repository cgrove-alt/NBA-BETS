"""
NBA Data Fetcher

Fetches NBA schedules, historical game data, team statistics, and player stats.

=============================================================================
DATA SOURCES
=============================================================================
- Schedule: Balldontlie (primary), NBA API (fallback)
- Player Stats: Balldontlie (primary), NBA API (fallback)
- Team Stats: NBA API only (Balldontlie has no team stats endpoint)
- Clutch Stats: NBA API only (not available in Balldontlie)

Prefer *_auto() functions which automatically try Balldontlie first:
- fetch_player_stats_auto() - player game stats
- fetch_player_stats_before_date_auto() - temporal-safe player stats

=============================================================================
TEMPORAL DISCIPLINE
=============================================================================
When training ML models, it is CRITICAL to avoid temporal leakage - using data
from the future to predict the past. This module provides two approaches:

1. LEAKAGE-SAFE functions (use these for training):
   - fetch_team_statistics_before_date(team_id, season, before_date)
   - fetch_player_stats_before_date_auto(player_id, season, before_date)
   - fetch_head_to_head(..., date_to=game_date)
   - fetch_historical_games(..., date_to=game_date)

2. CURRENT-STATE functions (use only for live predictions):
   - fetch_team_statistics(team_id, season) - returns full-season stats
   - fetch_player_stats_auto(player_id, season) - returns full-season player stats
   - fetch_head_to_head(...) without date_to - includes all games

When building features for historical games during training, ALWAYS use the
leakage-safe variants with the game's date to ensure you only use data that
was available at the time of the game.
=============================================================================
"""

import json
import time
import threading
import hashlib
import functools
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar
from collections.abc import Callable

# Type variable for generic return types
T = TypeVar('T')

try:
    from nba_api.stats.endpoints import (
        scoreboardv2,
        leaguegamefinder,
        teamdashboardbygeneralsplits,
        playergamelog,
        commonteamroster,
        teamgamelog,
        playerdashboardbygeneralsplits,
        leaguedashteamstats,
        commonplayerinfo,
        # CLUTCH STATS: For crunch-time performance data
        leaguedashplayerclutch,
        leaguedashteamclutch,
    )
    from nba_api.stats.static import teams, players
    HAS_CLUTCH_ENDPOINTS = True
except ImportError:
    # Some endpoints may not be available in older nba_api versions
    try:
        from nba_api.stats.endpoints import (
            scoreboardv2,
            leaguegamefinder,
            teamdashboardbygeneralsplits,
            playergamelog,
            commonteamroster,
            teamgamelog,
            playerdashboardbygeneralsplits,
            leaguedashteamstats,
            commonplayerinfo,
        )
        from nba_api.stats.static import teams, players
        HAS_CLUTCH_ENDPOINTS = False
    except ImportError:
        print("Error: nba_api package not installed.")
        print("Install it with: pip install nba_api")
        exit(1)

# Rate limiting to avoid API throttling
API_DELAY = 0.4  # seconds between API calls (reduced from 0.6 for faster props loading)

# =============================================================================
# RELIABILITY: Retry, Caching, and Fallback System
# =============================================================================

# Cache configuration
CACHE_DIR = Path(__file__).parent / ".api_cache"
CACHE_TTL_SECONDS = 3600  # 1 hour TTL for cached responses
CACHE_ENABLED = True

# Retry configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 2.0
RETRY_INITIAL_DELAY = 1.0  # seconds


def _get_cache_path(cache_key: str) -> Path:
    """Get the file path for a cache entry."""
    CACHE_DIR.mkdir(exist_ok=True)
    # Create a safe filename from the cache key
    key_hash = hashlib.md5(cache_key.encode()).hexdigest()
    return CACHE_DIR / f"{key_hash}.json"


def _read_from_cache(cache_key: str) -> Any | None:
    """Read data from disk cache if it exists and hasn't expired."""
    if not CACHE_ENABLED:
        return None

    cache_path = _get_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path) as f:
            cached = json.load(f)

        # Check TTL
        cached_at = cached.get("cached_at", 0)
        if time.time() - cached_at > CACHE_TTL_SECONDS:
            # Cache expired, delete it
            cache_path.unlink(missing_ok=True)
            return None

        return cached.get("data")
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def _write_to_cache(cache_key: str, data: Any) -> None:
    """Write data to disk cache."""
    if not CACHE_ENABLED:
        return

    cache_path = _get_cache_path(cache_key)
    try:
        with open(cache_path, "w") as f:
            json.dump({
                "cached_at": time.time(),
                "cache_key": cache_key,
                "data": data,
            }, f)
    except (OSError, TypeError):
        pass  # Fail silently - caching is best-effort


def clear_cache(older_than_hours: float = 0) -> int:
    """
    Clear the API cache.

    Args:
        older_than_hours: Only clear entries older than this many hours.
                          If 0, clears all cache entries.

    Returns:
        Number of cache entries removed.
    """
    if not CACHE_DIR.exists():
        return 0

    removed = 0
    cutoff_time = time.time() - (older_than_hours * 3600) if older_than_hours > 0 else float('inf')

    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            if older_than_hours > 0:
                with open(cache_file) as f:
                    cached = json.load(f)
                    if cached.get("cached_at", 0) > cutoff_time:
                        continue  # Not old enough to remove

            cache_file.unlink()
            removed += 1
        except (OSError, json.JSONDecodeError):
            try:
                cache_file.unlink()
                removed += 1
            except OSError:
                pass

    return removed


def retry_with_backoff(
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
    initial_delay: float = RETRY_INITIAL_DELAY,
    exceptions: tuple = (Exception,),
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for delay after each failure
        initial_delay: Initial delay in seconds before first retry
        exceptions: Tuple of exception types to catch and retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        print(f"[Retry {attempt + 1}/{max_attempts}] {func.__name__} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        print(f"[Retry {attempt + 1}/{max_attempts}] {func.__name__} failed permanently: {e}")

            # All retries exhausted - re-raise the last exception
            raise last_exception

        return wrapper
    return decorator


def with_cache(cache_key_fn: Callable[..., str]):
    """
    Decorator that adds disk caching to a function.

    Args:
        cache_key_fn: Function that generates a cache key from the function arguments

    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            cache_key = cache_key_fn(*args, **kwargs)

            # Check cache first
            cached_data = _read_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

            # Call the actual function
            result = func(*args, **kwargs)

            # Cache the result (only if it's not None/empty)
            if result:
                _write_to_cache(cache_key, result)

            return result

        return wrapper
    return decorator


# Fallback API client (lazy-loaded)
_balldontlie_api = None


def _get_balldontlie_api():
    """Get or create the Balldontlie API client."""
    global _balldontlie_api
    if _balldontlie_api is None:
        try:
            from balldontlie_api import BalldontlieAPI
            api_key = os.environ.get("BALLDONTLIE_API_KEY")
            if api_key:
                _balldontlie_api = BalldontlieAPI(api_key=api_key)
            else:
                _balldontlie_api = False  # No API key, disable fallback
        except (ImportError, ValueError) as e:
            print(f"Balldontlie fallback unavailable: {e}")
            _balldontlie_api = False

    return _balldontlie_api if _balldontlie_api else None


def with_fallback(fallback_fn: Callable | None = None):
    """
    Decorator that provides fallback to Balldontlie API on failure.

    Args:
        fallback_fn: Optional function to call as fallback. If None, no fallback is used.

    Returns:
        Decorated function with fallback logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if fallback_fn is None:
                    raise

                print(f"[Fallback] {func.__name__} failed ({e}), trying fallback...")
                try:
                    return fallback_fn(*args, **kwargs)
                except Exception as fallback_error:
                    print(f"[Fallback] Fallback also failed: {fallback_error}")
                    raise e  # Re-raise original exception

        return wrapper
    return decorator


class ThreadSafeRateLimiter:
    """Thread-safe rate limiter that allows concurrent requests while respecting API limits."""

    def __init__(self, min_interval: float = 0.4):
        """
        Initialize rate limiter.

        Args:
            min_interval: Minimum seconds between API calls (globally coordinated)
        """
        self._lock = threading.Lock()
        self._last_call_time = 0.0
        self._min_interval = min_interval

    def wait(self):
        """Wait until it's safe to make another API call."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                time.sleep(sleep_time)
            self._last_call_time = time.time()


# Global rate limiter instance
_rate_limiter = ThreadSafeRateLimiter(API_DELAY)


def _fetch_todays_schedule_balldontlie():
    """Primary: Fetch today's schedule using Balldontlie API."""
    api = _get_balldontlie_api()
    if not api:
        raise RuntimeError("Balldontlie API not available for fallback")

    today = datetime.now().strftime("%Y-%m-%d")
    games = api.get_todays_games()

    # Convert Balldontlie format to NBA API format
    game_header = []
    line_score = []

    for game in games:
        game_id = str(game.get("id", ""))
        home_team = game.get("home_team", {})
        visitor_team = game.get("visitor_team", {})

        game_header.append({
            "GAME_ID": game_id,
            "GAME_STATUS_TEXT": game.get("status", ""),
            "GAME_DATE_EST": game.get("date", today),
            "HOME_TEAM_ID": home_team.get("id"),
            "VISITOR_TEAM_ID": visitor_team.get("id"),
            "ARENA_NAME": "",
            "LIVE_PERIOD": game.get("period", 0),
            "LIVE_PC_TIME": "",
            "NATL_TV_BROADCASTER_ABBREVIATION": "",
        })

        # Add LineScore entries for team info
        line_score.append({
            "GAME_ID": game_id,
            "TEAM_ID": home_team.get("id"),
            "TEAM_ABBREVIATION": home_team.get("abbreviation", ""),
            "TEAM_CITY_NAME": home_team.get("city", ""),
            "TEAM_NAME": home_team.get("name", ""),
            "PTS": game.get("home_team_score"),
        })
        line_score.append({
            "GAME_ID": game_id,
            "TEAM_ID": visitor_team.get("id"),
            "TEAM_ABBREVIATION": visitor_team.get("abbreviation", ""),
            "TEAM_CITY_NAME": visitor_team.get("city", ""),
            "TEAM_NAME": visitor_team.get("name", ""),
            "PTS": game.get("visitor_team_score"),
        })

    return {"GameHeader": game_header, "LineScore": line_score}, today


@retry_with_backoff(max_attempts=3, exceptions=(Exception,))
def _fetch_todays_schedule_nba_api():
    """Fetch today's NBA schedule from the NBA API with retry logic."""
    today = datetime.now().strftime("%Y-%m-%d")
    _rate_limiter.wait()
    scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
    games_data = scoreboard.get_normalized_dict()
    return games_data, today


def fetch_todays_schedule():
    """
    Fetch today's NBA schedule from Balldontlie API.

    RELIABILITY: Includes retry logic and fallback to NBA API.
    Primary: Balldontlie API (premium data source)
    Fallback: NBA API (scoreboardv2)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching NBA schedule for {today}...")

    # Primary: Balldontlie API
    try:
        return _fetch_todays_schedule_balldontlie()
    except Exception as e:
        print(f"[Fallback] Balldontlie API failed ({e}), trying NBA API...")
        try:
            return _fetch_todays_schedule_nba_api()
        except Exception as fallback_error:
            print(f"[Fallback] NBA API also failed: {fallback_error}")
            # Return empty schedule rather than crashing
            return {"GameHeader": [], "LineScore": []}, today


def parse_game_details(games_data):
    """Parse the raw API response to extract relevant game details."""
    games = []

    game_header = games_data.get("GameHeader", [])
    line_score = games_data.get("LineScore", [])

    # Build team ID to info lookup from nba_api teams
    nba_teams = teams.get_teams()
    team_id_lookup = {team['id']: team for team in nba_teams}

    # Create a lookup for team scores by game ID (for live/completed games)
    team_scores = {}
    for team in line_score:
        game_id = team.get("GAME_ID")
        if game_id not in team_scores:
            team_scores[game_id] = []
        team_scores[game_id].append({
            "team_id": team.get("TEAM_ID"),
            "team_abbreviation": team.get("TEAM_ABBREVIATION"),
            "team_city": team.get("TEAM_CITY_NAME"),
            "team_name": team.get("TEAM_NAME"),
            "pts": team.get("PTS"),
        })

    for game in game_header:
        game_id = game.get("GAME_ID")
        game_status = game.get("GAME_STATUS_TEXT", "")
        game_time = game.get("GAME_DATE_EST", "")

        home_team_id = game.get("HOME_TEAM_ID")
        visitor_team_id = game.get("VISITOR_TEAM_ID")

        # Get team details from line score (for live/completed games)
        home_team = None
        visitor_team = None

        for team in team_scores.get(game_id, []):
            if team["team_id"] == home_team_id:
                home_team = team
            elif team["team_id"] == visitor_team_id:
                visitor_team = team

        # Fallback to nba_api team lookup if LineScore is empty (games not started)
        home_team_static = team_id_lookup.get(home_team_id, {})
        visitor_team_static = team_id_lookup.get(visitor_team_id, {})

        game_info = {
            "game_id": game_id,
            "status": game_status,
            "game_time": game_time,
            "arena": game.get("ARENA_NAME"),
            "home_team": {
                "id": home_team_id,
                "abbreviation": home_team["team_abbreviation"] if home_team else home_team_static.get("abbreviation"),
                "city": home_team["team_city"] if home_team else home_team_static.get("city"),
                "name": home_team["team_name"] if home_team else home_team_static.get("nickname"),
                "score": home_team["pts"] if home_team else None,
            },
            "visitor_team": {
                "id": visitor_team_id,
                "abbreviation": visitor_team["team_abbreviation"] if visitor_team else visitor_team_static.get("abbreviation"),
                "city": visitor_team["team_city"] if visitor_team else visitor_team_static.get("city"),
                "name": visitor_team["team_name"] if visitor_team else visitor_team_static.get("nickname"),
                "score": visitor_team["pts"] if visitor_team else None,
            },
            "live_period": game.get("LIVE_PERIOD"),
            "live_pc_time": game.get("LIVE_PC_TIME"),
            "natl_tv_broadcaster": game.get("NATL_TV_BROADCASTER_ABBREVIATION"),
        }

        games.append(game_info)

    return games


def get_team_id(team_name_or_abbrev):
    """Get team ID from team name or abbreviation."""
    nba_teams = teams.get_teams()
    for team in nba_teams:
        if (team_name_or_abbrev.upper() == team['abbreviation'] or
            team_name_or_abbrev.lower() in team['full_name'].lower() or
            team_name_or_abbrev.lower() == team['nickname'].lower()):
            return team['id']
    return None


def get_team_abbrev(team_id: int) -> str | None:
    """Get team abbreviation from NBA team ID."""
    nba_teams = teams.get_teams()
    for team in nba_teams:
        if team['id'] == team_id:
            return team['abbreviation']
    return None


def get_player_id(player_name):
    """Get player ID from player name."""
    nba_players = players.get_players()
    for player in nba_players:
        if player_name.lower() in player['full_name'].lower():
            return player['id']
    return None


def _historical_games_cache_key(team_id=None, season="2025-26", last_n_games=None, date_from=None, date_to=None):
    """Generate cache key for historical games."""
    return f"historical_games:{team_id}:{season}:{last_n_games}:{date_from}:{date_to}"


@retry_with_backoff(max_attempts=3, exceptions=(Exception,))
def _fetch_historical_games_api(team_id=None, season="2025-26", date_from=None, date_to=None):
    """Raw API call with retry logic."""
    _rate_limiter.wait()
    game_finder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable="Regular Season",
        date_from_nullable=date_from,
        date_to_nullable=date_to,
    )
    games_df = game_finder.get_normalized_dict()
    return games_df.get("LeagueGameFinderResults", [])


def fetch_historical_games(team_id=None, season="2025-26", last_n_games=None, date_from=None, date_to=None):
    """
    Fetch historical game data for analysis.

    RELIABILITY: Includes retry logic with exponential backoff and disk caching.

    Args:
        team_id: Optional team ID to filter games
        season: NBA season (e.g., "2025-26")
        last_n_games: Limit to last N games
        date_from: Start date (MM/DD/YYYY format)
        date_to: End date (MM/DD/YYYY format)

    Returns:
        List of game dictionaries with detailed stats
    """
    # Check cache first
    cache_key = _historical_games_cache_key(team_id, season, last_n_games, date_from, date_to)
    cached = _read_from_cache(cache_key)
    if cached is not None:
        return cached

    # Fetch from API with retry
    try:
        games = _fetch_historical_games_api(team_id, season, date_from, date_to)
    except Exception as e:
        print(f"Warning: Failed to fetch historical games: {e}")
        return []

    parsed_games = []
    for game in games:
        if last_n_games and len(parsed_games) >= last_n_games:
            break

        parsed_game = {
            "game_id": game.get("GAME_ID"),
            "game_date": game.get("GAME_DATE"),
            "matchup": game.get("MATCHUP"),
            "wl": game.get("WL"),
            "team_id": game.get("TEAM_ID"),
            "team_abbreviation": game.get("TEAM_ABBREVIATION"),
            "pts": game.get("PTS"),
            "fg_pct": game.get("FG_PCT"),
            "fg3_pct": game.get("FG3_PCT"),
            "ft_pct": game.get("FT_PCT"),
            "reb": game.get("REB"),
            "ast": game.get("AST"),
            "stl": game.get("STL"),
            "blk": game.get("BLK"),
            "tov": game.get("TOV"),
            "plus_minus": game.get("PLUS_MINUS"),
            "min": game.get("MIN"),
        }
        parsed_games.append(parsed_game)

    # Cache successful result
    if parsed_games:
        _write_to_cache(cache_key, parsed_games)

    return parsed_games


def _team_stats_cache_key(team_id, season="2025-26"):
    """Generate cache key for team statistics."""
    return f"team_stats:{team_id}:{season}"


@retry_with_backoff(max_attempts=3, exceptions=(Exception,))
def _fetch_team_stats_api(team_id, season="2025-26"):
    """Raw API call for team stats with retry logic."""
    _rate_limiter.wait()
    team_stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
        team_id=team_id,
        season=season,
        season_type_all_star="Regular Season"
    )
    return team_stats.get_normalized_dict()


@retry_with_backoff(max_attempts=3, exceptions=(Exception,))
def _fetch_team_advanced_stats_api(team_id, season="2025-26"):
    """Raw API call for team advanced stats with retry logic."""
    _rate_limiter.wait()
    advanced_stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
        team_id=team_id,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced"
    )
    return advanced_stats.get_normalized_dict()


def fetch_team_statistics(team_id, season="2025-26"):
    """
    Fetch comprehensive team statistics.

    WARNING: This function returns CURRENT (full-season) stats, which may cause
    TEMPORAL LEAKAGE when used for training on historical games. For training,
    use fetch_team_statistics_before_date() instead.

    RELIABILITY: Includes retry logic with exponential backoff and disk caching.

    Args:
        team_id: NBA team ID
        season: NBA season (e.g., "2025-26")

    Returns:
        Dictionary with team statistics
    """
    # Check cache first
    cache_key = _team_stats_cache_key(team_id, season)
    cached = _read_from_cache(cache_key)
    if cached is not None:
        return cached

    # Fetch base stats with retry
    try:
        stats_dict = _fetch_team_stats_api(team_id, season)
    except Exception as e:
        print(f"Warning: Could not fetch team stats for {team_id}: {e}")
        stats_dict = {}

    overall = stats_dict.get("OverallTeamDashboard", [{}])[0] if stats_dict.get("OverallTeamDashboard") else {}
    home_away = stats_dict.get("LocationTeamDashboard", [])

    # Fetch advanced stats for ratings with retry
    try:
        advanced_dict = _fetch_team_advanced_stats_api(team_id, season)
        advanced_overall = advanced_dict.get("OverallTeamDashboard", [{}])[0] if advanced_dict.get("OverallTeamDashboard") else {}
    except Exception as e:
        print(f"Warning: Could not fetch advanced stats for team {team_id}: {e}")
        advanced_overall = {}

    home_stats = next((s for s in home_away if s.get("GROUP_VALUE") == "Home"), {})
    away_stats = next((s for s in home_away if s.get("GROUP_VALUE") == "Road"), {})

    # Calculate games played for dividing totals into averages
    gp = max(overall.get("GP") or 1, 1)  # Avoid division by zero

    result = {
        "team_id": team_id,
        "season": season,
        "overall": {
            "games_played": overall.get("GP"),
            "wins": overall.get("W"),
            "losses": overall.get("L"),
            "win_pct": overall.get("W_PCT"),
            # NBA API returns season TOTALS, divide by GP for per-game averages
            "pts_avg": (overall.get("PTS") or 0) / gp,
            "reb_avg": (overall.get("REB") or 0) / gp,
            "ast_avg": (overall.get("AST") or 0) / gp,
            "stl_avg": (overall.get("STL") or 0) / gp,
            "blk_avg": (overall.get("BLK") or 0) / gp,
            "tov_avg": (overall.get("TOV") or 0) / gp,
            "fg_pct": overall.get("FG_PCT"),
            "fg3_pct": overall.get("FG3_PCT"),
            "ft_pct": overall.get("FT_PCT"),
            "plus_minus": (overall.get("PLUS_MINUS") or 0) / gp,
            "off_rating": advanced_overall.get("OFF_RATING"),
            "def_rating": advanced_overall.get("DEF_RATING"),
            "net_rating": advanced_overall.get("NET_RATING"),
            "pace": advanced_overall.get("PACE"),
        },
        "home": {
            "games_played": home_stats.get("GP"),
            "wins": home_stats.get("W"),
            "losses": home_stats.get("L"),
            "win_pct": home_stats.get("W_PCT"),
            # PTS is total points, divide by GP for average
            "pts_avg": (home_stats.get("PTS") or 0) / max(home_stats.get("GP") or 1, 1),
            "plus_minus": home_stats.get("PLUS_MINUS"),
        },
        "away": {
            "games_played": away_stats.get("GP"),
            "wins": away_stats.get("W"),
            "losses": away_stats.get("L"),
            "win_pct": away_stats.get("W_PCT"),
            # PTS is total points, divide by GP for average
            "pts_avg": (away_stats.get("PTS") or 0) / max(away_stats.get("GP") or 1, 1),
            "plus_minus": away_stats.get("PLUS_MINUS"),
        },
    }

    # Cache successful result
    if result.get("overall", {}).get("games_played"):
        _write_to_cache(cache_key, result)

    return result


def fetch_team_statistics_before_date(team_id, season="2025-26", before_date=None):
    """
    TEMPORAL DISCIPLINE: Compute team stats from games BEFORE the specified date only.

    This function prevents temporal leakage by only using data that would have been
    available at the time of the game being predicted.

    Args:
        team_id: NBA team ID
        season: NBA season (e.g., "2025-26")
        before_date: Date string (YYYY-MM-DD format). Only include games BEFORE this date.
                     If None, returns empty/zero stats.

    Returns:
        Dictionary with team statistics (same structure as fetch_team_statistics)
    """
    if before_date is None:
        # No date specified - return empty stats to avoid leakage
        return {
            "team_id": team_id,
            "season": season,
            "overall": {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "win_pct": 0.0,
                "pts_avg": 0.0,
                "reb_avg": 0.0,
                "ast_avg": 0.0,
                "stl_avg": 0.0,
                "blk_avg": 0.0,
                "tov_avg": 0.0,
                "fg_pct": 0.0,
                "fg3_pct": 0.0,
                "ft_pct": 0.0,
                "plus_minus": 0.0,
                "off_rating": None,
                "def_rating": None,
                "net_rating": None,
                "pace": None,
            },
            "home": {
                "games_played": 0, "wins": 0, "losses": 0, "win_pct": 0.0,
                "pts_avg": 0.0, "plus_minus": 0.0,
            },
            "away": {
                "games_played": 0, "wins": 0, "losses": 0, "win_pct": 0.0,
                "pts_avg": 0.0, "plus_minus": 0.0,
            },
        }

    # Convert date to MM/DD/YYYY format for API
    try:
        date_obj = datetime.strptime(before_date, "%Y-%m-%d")
        # Use day before to get games STRICTLY before this date
        day_before = date_obj - timedelta(days=1)
        date_to = day_before.strftime("%m/%d/%Y")
    except ValueError:
        # If date parsing fails, return empty stats
        return fetch_team_statistics_before_date(team_id, season, None)

    # Fetch games using date_to filter
    games = fetch_historical_games(team_id=team_id, season=season, date_to=date_to)

    if not games:
        return fetch_team_statistics_before_date(team_id, season, None)

    # Calculate averages from game-by-game data
    total_games = len(games)
    home_games = []
    away_games = []

    for g in games:
        matchup = g.get("matchup", "")
        is_home = " vs. " in matchup
        if is_home:
            home_games.append(g)
        else:
            away_games.append(g)

    def calc_avg(game_list, key):
        values = [g.get(key) for g in game_list if g.get(key) is not None]
        return sum(values) / len(values) if values else 0.0

    def count_wins(game_list):
        return sum(1 for g in game_list if g.get("wl") == "W")

    # Overall stats
    overall_wins = count_wins(games)
    overall_losses = total_games - overall_wins

    # Home stats
    home_wins = count_wins(home_games)
    home_losses = len(home_games) - home_wins

    # Away stats
    away_wins = count_wins(away_games)
    away_losses = len(away_games) - away_wins

    return {
        "team_id": team_id,
        "season": season,
        "overall": {
            "games_played": total_games,
            "wins": overall_wins,
            "losses": overall_losses,
            "win_pct": overall_wins / total_games if total_games > 0 else 0.0,
            "pts_avg": calc_avg(games, "pts"),
            "reb_avg": calc_avg(games, "reb"),
            "ast_avg": calc_avg(games, "ast"),
            "stl_avg": calc_avg(games, "stl"),
            "blk_avg": calc_avg(games, "blk"),
            "tov_avg": calc_avg(games, "tov"),
            "fg_pct": calc_avg(games, "fg_pct"),
            "fg3_pct": calc_avg(games, "fg3_pct"),
            "ft_pct": calc_avg(games, "ft_pct"),
            "plus_minus": calc_avg(games, "plus_minus"),
            # Advanced stats not available from game logs - use None
            "off_rating": None,
            "def_rating": None,
            "net_rating": None,
            "pace": None,
        },
        "home": {
            "games_played": len(home_games),
            "wins": home_wins,
            "losses": home_losses,
            "win_pct": home_wins / len(home_games) if home_games else 0.0,
            "pts_avg": calc_avg(home_games, "pts"),
            "plus_minus": calc_avg(home_games, "plus_minus"),
        },
        "away": {
            "games_played": len(away_games),
            "wins": away_wins,
            "losses": away_losses,
            "win_pct": away_wins / len(away_games) if away_games else 0.0,
            "pts_avg": calc_avg(away_games, "pts"),
            "plus_minus": calc_avg(away_games, "plus_minus"),
        },
    }


def fetch_league_team_stats(season="2025-26"):
    """
    Fetch league-wide team statistics for ranking and comparison.

    Args:
        season: NBA season

    Returns:
        List of team stats dictionaries
    """
    _rate_limiter.wait()

    league_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame"
    )
    stats_dict = league_stats.get_normalized_dict()

    return stats_dict.get("LeagueDashTeamStats", [])


def _player_stats_cache_key(player_id, season="2025-26", last_n_games=None):
    """Generate cache key for player statistics."""
    return f"player_stats:{player_id}:{season}:{last_n_games}"


@retry_with_backoff(max_attempts=3, exceptions=(Exception,))
def _fetch_player_game_log_api(player_id, season="2025-26"):
    """Raw API call for player game log with retry logic."""
    _rate_limiter.wait()
    game_log = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season"
    )
    return game_log.get_normalized_dict()


@retry_with_backoff(max_attempts=3, exceptions=(Exception,))
def _fetch_player_dashboard_api(player_id, season="2025-26"):
    """Raw API call for player dashboard with retry logic."""
    _rate_limiter.wait()
    player_dashboard = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
        player_id=player_id,
        season=season,
        season_type_playoffs="Regular Season",
        per_mode_detailed="PerGame"
    )
    return player_dashboard.get_normalized_dict()


def fetch_player_stats(player_id, season="2025-26", last_n_games=None):
    """
    Fetch player statistics and game log.

    RELIABILITY: Includes retry logic with exponential backoff and disk caching.

    Args:
        player_id: NBA player ID
        season: NBA season (e.g., "2024-25")
        last_n_games: Optional limit to last N games

    Returns:
        Dictionary with player stats, game log, and last 5 game averages
    """
    # Check cache first
    cache_key = _player_stats_cache_key(player_id, season, last_n_games)
    cached = _read_from_cache(cache_key)
    if cached is not None:
        return cached

    # Get player game log with retry
    try:
        log_dict = _fetch_player_game_log_api(player_id, season)
        games = log_dict.get("PlayerGameLog", [])
    except Exception as e:
        print(f"Warning: Could not fetch game log for player {player_id}: {e}")
        games = []

    if last_n_games:
        games = games[:last_n_games]

    # Get player dashboard stats with retry
    try:
        dashboard_dict = _fetch_player_dashboard_api(player_id, season)
        overall = dashboard_dict.get("OverallPlayerDashboard", [{}])[0] if dashboard_dict.get("OverallPlayerDashboard") else {}
    except Exception as e:
        print(f"Warning: Could not fetch dashboard for player {player_id}: {e}")
        overall = {}

    parsed_games = []
    for game in games:
        parsed_games.append({
            "game_id": game.get("Game_ID"),
            "game_date": game.get("GAME_DATE"),
            "matchup": game.get("MATCHUP"),
            "wl": game.get("WL"),
            "min": game.get("MIN"),
            "pts": game.get("PTS"),
            "reb": game.get("REB"),
            "ast": game.get("AST"),
            "stl": game.get("STL"),
            "blk": game.get("BLK"),
            "tov": game.get("TOV"),
            "fg_made": game.get("FGM"),
            "fg_att": game.get("FGA"),
            "fg_pct": game.get("FG_PCT"),
            "fg3_made": game.get("FG3M"),
            "fg3_att": game.get("FG3A"),
            "fg3_pct": game.get("FG3_PCT"),
            "ft_made": game.get("FTM"),
            "ft_att": game.get("FTA"),
            "ft_pct": game.get("FT_PCT"),
            "plus_minus": game.get("PLUS_MINUS"),
        })

    # Calculate last 5 games averages for recent form
    last_5_averages = {}
    if len(parsed_games) >= 1:
        recent_games = parsed_games[:5]  # Game log is already sorted most recent first
        num_games = len(recent_games)
        last_5_averages = {
            "games_count": num_games,
            "pts_avg": sum(g.get("pts", 0) or 0 for g in recent_games) / num_games,
            "reb_avg": sum(g.get("reb", 0) or 0 for g in recent_games) / num_games,
            "ast_avg": sum(g.get("ast", 0) or 0 for g in recent_games) / num_games,
            "fg3_avg": sum(g.get("fg3_made", 0) or 0 for g in recent_games) / num_games,
            "min_avg": sum(g.get("min", 0) or 0 for g in recent_games) / num_games,
            "stl_avg": sum(g.get("stl", 0) or 0 for g in recent_games) / num_games,
            "blk_avg": sum(g.get("blk", 0) or 0 for g in recent_games) / num_games,
        }

    result = {
        "player_id": player_id,
        "season": season,
        "season_averages": {
            "games_played": overall.get("GP"),
            "min_avg": overall.get("MIN"),
            "pts_avg": overall.get("PTS"),
            "reb_avg": overall.get("REB"),
            "ast_avg": overall.get("AST"),
            "stl_avg": overall.get("STL"),
            "blk_avg": overall.get("BLK"),
            "tov_avg": overall.get("TOV"),
            "fg_pct": overall.get("FG_PCT"),
            "fg3_pct": overall.get("FG3_PCT"),
            "ft_pct": overall.get("FT_PCT"),
            "plus_minus": overall.get("PLUS_MINUS"),
        },
        "last_5_averages": last_5_averages,
        "game_log": parsed_games,
    }

    # Cache successful result
    if parsed_games:
        _write_to_cache(cache_key, result)

    return result


def fetch_player_stats_before_date(player_id, season="2025-26", before_date=None, last_n_games=None):
    """
    TEMPORAL DISCIPLINE: Fetch player statistics using ONLY games BEFORE the specified date.

    This is the leakage-safe version of fetch_player_stats(). Use this when training
    on historical games to ensure you don't use future data to make predictions.

    Args:
        player_id: NBA player ID
        season: NBA season (e.g., "2024-25")
        before_date: Date string (YYYY-MM-DD). Only include games BEFORE this date.
                     If None, returns empty data (safe default to prevent leakage).
        last_n_games: Optional limit to last N games before the date

    Returns:
        Dictionary with player stats computed only from games before before_date
    """
    # Safety: If no date provided, return empty data to prevent accidental leakage
    if before_date is None:
        return {
            "player_id": player_id,
            "season": season,
            "season_averages": {},
            "last_5_averages": {},
            "game_log": [],
            "games_used": 0,
            "temporal_cutoff": None,
        }

    _rate_limiter.wait()

    # Get player game log
    try:
        game_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season"
        )
        log_dict = game_log.get_normalized_dict()
        all_games = log_dict.get("PlayerGameLog", [])
    except Exception as e:
        print(f"Warning: Could not fetch game log for player {player_id}: {e}")
        return {
            "player_id": player_id,
            "season": season,
            "season_averages": {},
            "last_5_averages": {},
            "game_log": [],
            "games_used": 0,
            "temporal_cutoff": before_date,
        }

    # Filter games to only those BEFORE the specified date
    # Game log is sorted most recent first
    filtered_games = []
    for game in all_games:
        game_date_str = game.get("GAME_DATE", "")
        # GAME_DATE format varies: could be "DEC 25, 2024" or "2024-12-25"
        try:
            if "-" in game_date_str:
                # ISO format: YYYY-MM-DD
                game_date = game_date_str[:10]  # Take first 10 chars
            else:
                # Parse "DEC 25, 2024" format
                from datetime import datetime as dt
                parsed = dt.strptime(game_date_str, "%b %d, %Y")
                game_date = parsed.strftime("%Y-%m-%d")

            # Only include games BEFORE the cutoff date
            if game_date < before_date:
                filtered_games.append(game)
        except Exception:
            # If date parsing fails, skip game to be safe
            continue

    # Apply last_n_games limit if specified
    if last_n_games and len(filtered_games) > last_n_games:
        filtered_games = filtered_games[:last_n_games]

    # Parse games into standard format
    parsed_games = []
    for game in filtered_games:
        parsed_games.append({
            "game_id": game.get("Game_ID"),
            "game_date": game.get("GAME_DATE"),
            "matchup": game.get("MATCHUP"),
            "wl": game.get("WL"),
            "min": game.get("MIN"),
            "pts": game.get("PTS"),
            "reb": game.get("REB"),
            "ast": game.get("AST"),
            "stl": game.get("STL"),
            "blk": game.get("BLK"),
            "tov": game.get("TOV"),
            "fg_made": game.get("FGM"),
            "fg_att": game.get("FGA"),
            "fg_pct": game.get("FG_PCT"),
            "fg3_made": game.get("FG3M"),
            "fg3_att": game.get("FG3A"),
            "fg3_pct": game.get("FG3_PCT"),
            "ft_made": game.get("FTM"),
            "ft_att": game.get("FTA"),
            "ft_pct": game.get("FT_PCT"),
            "plus_minus": game.get("PLUS_MINUS"),
        })

    # Calculate season averages from filtered games (not from API which has full season)
    season_averages = {}
    if parsed_games:
        num_games = len(parsed_games)
        season_averages = {
            "games_played": num_games,
            "min_avg": sum(g.get("min", 0) or 0 for g in parsed_games) / num_games,
            "pts_avg": sum(g.get("pts", 0) or 0 for g in parsed_games) / num_games,
            "reb_avg": sum(g.get("reb", 0) or 0 for g in parsed_games) / num_games,
            "ast_avg": sum(g.get("ast", 0) or 0 for g in parsed_games) / num_games,
            "stl_avg": sum(g.get("stl", 0) or 0 for g in parsed_games) / num_games,
            "blk_avg": sum(g.get("blk", 0) or 0 for g in parsed_games) / num_games,
            "tov_avg": sum(g.get("tov", 0) or 0 for g in parsed_games) / num_games,
            "fg3_avg": sum(g.get("fg3_made", 0) or 0 for g in parsed_games) / num_games,
            "fg_pct": sum(g.get("fg_pct", 0) or 0 for g in parsed_games) / num_games,
            "fg3_pct": sum(g.get("fg3_pct", 0) or 0 for g in parsed_games) / num_games,
            "ft_pct": sum(g.get("ft_pct", 0) or 0 for g in parsed_games) / num_games,
            "plus_minus": sum(g.get("plus_minus", 0) or 0 for g in parsed_games) / num_games,
        }

    # Calculate last 5 games averages (from filtered data)
    last_5_averages = {}
    if len(parsed_games) >= 1:
        recent_games = parsed_games[:5]  # Most recent 5 games (before cutoff)
        num_recent = len(recent_games)
        last_5_averages = {
            "games_count": num_recent,
            "pts_avg": sum(g.get("pts", 0) or 0 for g in recent_games) / num_recent,
            "reb_avg": sum(g.get("reb", 0) or 0 for g in recent_games) / num_recent,
            "ast_avg": sum(g.get("ast", 0) or 0 for g in recent_games) / num_recent,
            "fg3_avg": sum(g.get("fg3_made", 0) or 0 for g in recent_games) / num_recent,
            "min_avg": sum(g.get("min", 0) or 0 for g in recent_games) / num_recent,
            "stl_avg": sum(g.get("stl", 0) or 0 for g in recent_games) / num_recent,
            "blk_avg": sum(g.get("blk", 0) or 0 for g in recent_games) / num_recent,
        }

    return {
        "player_id": player_id,
        "season": season,
        "season_averages": season_averages,
        "last_5_averages": last_5_averages,
        "game_log": parsed_games,
        "games_used": len(parsed_games),
        "temporal_cutoff": before_date,
    }


def fetch_team_roster(team_id, season="2025-26"):
    """
    Fetch team roster with player IDs.

    Args:
        team_id: NBA team ID
        season: NBA season

    Returns:
        List of player dictionaries
    """
    _rate_limiter.wait()

    roster = commonteamroster.CommonTeamRoster(
        team_id=team_id,
        season=season
    )
    roster_dict = roster.get_normalized_dict()

    players_list = []
    for player in roster_dict.get("CommonTeamRoster", []):
        players_list.append({
            "player_id": player.get("PLAYER_ID"),
            "player_name": player.get("PLAYER"),
            "position": player.get("POSITION"),
            "height": player.get("HEIGHT"),
            "weight": player.get("WEIGHT"),
            "age": player.get("AGE"),
            "experience": player.get("EXP"),
        })

    return players_list


def fetch_player_info(player_id):
    """
    Fetch detailed player information.

    Args:
        player_id: NBA player ID

    Returns:
        Dictionary with player info
    """
    _rate_limiter.wait()

    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    info_dict = player_info.get_normalized_dict()

    info = info_dict.get("CommonPlayerInfo", [{}])[0] if info_dict.get("CommonPlayerInfo") else {}

    return {
        "player_id": player_id,
        "first_name": info.get("FIRST_NAME"),
        "last_name": info.get("LAST_NAME"),
        "full_name": f"{info.get('FIRST_NAME', '')} {info.get('LAST_NAME', '')}".strip(),
        "team_id": info.get("TEAM_ID"),
        "team_name": info.get("TEAM_NAME"),
        "team_abbreviation": info.get("TEAM_ABBREVIATION"),
        "position": info.get("POSITION"),
        "height": info.get("HEIGHT"),
        "weight": info.get("WEIGHT"),
        "birth_date": info.get("BIRTHDATE"),
        "experience": info.get("SEASON_EXP"),
        "jersey": info.get("JERSEY"),
        "draft_year": info.get("DRAFT_YEAR"),
        "draft_round": info.get("DRAFT_ROUND"),
        "draft_number": info.get("DRAFT_NUMBER"),
    }


def fetch_head_to_head(team1_id, team2_id, season="2025-26", last_n_games=10, date_to=None):
    """
    Fetch head-to-head game history between two teams.

    TEMPORAL DISCIPLINE: Use date_to parameter to prevent temporal leakage
    when computing H2H features for historical games.

    Args:
        team1_id: First team NBA ID
        team2_id: Second team NBA ID
        season: NBA season (can include multiple seasons like "2023-24,2025-26")
        last_n_games: Maximum number of games to return
        date_to: Optional date cutoff (MM/DD/YYYY or YYYY-MM-DD format).
                 Only returns games BEFORE this date.

    Returns:
        List of head-to-head game results
    """
    _rate_limiter.wait()

    # Convert date format if needed (YYYY-MM-DD -> MM/DD/YYYY)
    date_to_api = None
    if date_to:
        try:
            if "-" in date_to and len(date_to) == 10:  # YYYY-MM-DD format
                date_obj = datetime.strptime(date_to, "%Y-%m-%d")
                # Use day before to get games STRICTLY before this date
                day_before = date_obj - timedelta(days=1)
                date_to_api = day_before.strftime("%m/%d/%Y")
            else:
                date_to_api = date_to
        except ValueError:
            pass  # Use None if parsing fails

    # Get team1's games
    game_finder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team1_id,
        vs_team_id_nullable=team2_id,
        season_nullable=season,
        season_type_nullable="Regular Season",
        date_to_nullable=date_to_api,
    )
    games_dict = game_finder.get_normalized_dict()
    games = games_dict.get("LeagueGameFinderResults", [])

    if last_n_games:
        games = games[:last_n_games]

    h2h_games = []
    for game in games:
        h2h_games.append({
            "game_id": game.get("GAME_ID"),
            "game_date": game.get("GAME_DATE"),
            "matchup": game.get("MATCHUP"),
            "team_id": game.get("TEAM_ID"),
            "wl": game.get("WL"),
            "pts": game.get("PTS"),
            "fg_pct": game.get("FG_PCT"),
            "fg3_pct": game.get("FG3_PCT"),
            "reb": game.get("REB"),
            "ast": game.get("AST"),
            "plus_minus": game.get("PLUS_MINUS"),
        })

    return h2h_games


def fetch_player_vs_team(player_id, opponent_team_id, season="2025-26", last_n_games=10):
    """
    Fetch player's performance history against a specific team.

    Args:
        player_id: NBA player ID
        opponent_team_id: Opponent team NBA ID
        season: NBA season
        last_n_games: Maximum games to return

    Returns:
        List of player game logs against the opponent
    """
    _rate_limiter.wait()

    game_log = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season"
    )
    log_dict = game_log.get_normalized_dict()
    all_games = log_dict.get("PlayerGameLog", [])

    # Filter for games against the specific opponent
    # Get opponent abbreviation
    nba_teams = teams.get_teams()
    opp_abbrev = None
    for team in nba_teams:
        if team['id'] == opponent_team_id:
            opp_abbrev = team['abbreviation']
            break

    vs_games = []
    for game in all_games:
        matchup = game.get("MATCHUP", "")
        if opp_abbrev and opp_abbrev in matchup:
            vs_games.append({
                "game_id": game.get("Game_ID"),
                "game_date": game.get("GAME_DATE"),
                "matchup": matchup,
                "wl": game.get("WL"),
                "min": game.get("MIN"),
                "pts": game.get("PTS"),
                "reb": game.get("REB"),
                "ast": game.get("AST"),
                "stl": game.get("STL"),
                "blk": game.get("BLK"),
                "fg_pct": game.get("FG_PCT"),
                "fg3_made": game.get("FG3M"),
                "fg3_pct": game.get("FG3_PCT"),
                "plus_minus": game.get("PLUS_MINUS"),
            })
            if last_n_games and len(vs_games) >= last_n_games:
                break

    return vs_games


def save_schedule_to_json(schedule, date, output_dir="."):
    """Save the parsed schedule to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / f"nba_schedule_{date}.json"

    output_data = {
        "date": date,
        "fetched_at": datetime.now().isoformat(),
        "game_count": len(schedule),
        "games": schedule,
    }

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Schedule saved to {filename}")
    return filename


# =============================================================================
# BALLDONTLIE API PRIMARY DATA LAYER
# =============================================================================
# These functions use Balldontlie API as primary source for faster, more
# reliable data fetching. Falls back to NBA API if needed.
# =============================================================================

def _parse_minutes(min_str) -> float:
    """Parse Balldontlie minutes string (e.g., '23:45') to float."""
    if min_str is None:
        return 0.0
    if isinstance(min_str, (int, float)):
        return float(min_str)
    if isinstance(min_str, str):
        try:
            if ':' in min_str:
                parts = min_str.split(':')
                return float(parts[0]) + float(parts[1]) / 60.0
            return float(min_str)
        except (ValueError, IndexError):
            return 0.0
    return 0.0


def _get_id_mapper():
    """Get or create the ID mapper instance."""
    global _id_mapper
    try:
        _id_mapper
    except NameError:
        _id_mapper = None

    if _id_mapper is None:
        try:
            from id_mapping import IDMapper
            _id_mapper = IDMapper()
        except ImportError:
            _id_mapper = False

    return _id_mapper if _id_mapper else None


def fetch_player_stats_bdl(
    player_id: int = None,
    player_name: str = None,
    season: int = None,
    last_n_games: int = None,
) -> dict:
    """
    Fetch player statistics using Balldontlie API (primary source).

    This is significantly faster than NBA API (600 req/min vs 5 req/min).

    Args:
        player_id: Balldontlie player ID (preferred)
        player_name: Player name (will be converted to BDL ID)
        season: Season year (e.g., 2024 for 2024-25 season)
        last_n_games: Limit to last N games

    Returns:
        Dictionary with player stats matching fetch_player_stats() format
    """
    api = _get_balldontlie_api()
    if not api:
        # Fall back to NBA API
        if player_name:
            nba_pid = get_player_id(player_name)
            if nba_pid:
                return fetch_player_stats(nba_pid, last_n_games=last_n_games)
        return {}

    # Convert player name to Balldontlie ID if needed
    bdl_player_id = player_id
    if bdl_player_id is None and player_name:
        mapper = _get_id_mapper()
        if mapper:
            bdl_player_id = mapper.get_player_id(player_name)

    if bdl_player_id is None:
        return {
            "player_id": player_id,
            "season": season,
            "season_averages": {},
            "last_5_averages": {},
            "game_log": [],
        }

    # Determine season
    if season is None:
        season = datetime.now().year if datetime.now().month > 9 else datetime.now().year - 1

    # Fetch all game stats for the season
    all_stats = api.get_player_stats_paginated(bdl_player_id, season)

    if not all_stats:
        return {
            "player_id": bdl_player_id,
            "season": season,
            "season_averages": {},
            "last_5_averages": {},
            "game_log": [],
        }

    # Sort by date descending (most recent first)
    all_stats.sort(
        key=lambda x: x.get("game", {}).get("date", ""),
        reverse=True
    )

    # Apply last_n_games limit
    games = all_stats[:last_n_games] if last_n_games else all_stats

    # Convert to standard format
    parsed_games = []
    for stat in games:
        game = stat.get("game", {})
        parsed_games.append({
            "game_id": game.get("id"),
            "game_date": game.get("date", "")[:10] if game.get("date") else "",
            "matchup": f"{game.get('visitor_team', {}).get('abbreviation', '')} @ {game.get('home_team', {}).get('abbreviation', '')}",
            "wl": None,  # Not directly available
            "min": _parse_minutes(stat.get("min")),
            "pts": stat.get("pts"),
            "reb": stat.get("reb"),
            "ast": stat.get("ast"),
            "stl": stat.get("stl"),
            "blk": stat.get("blk"),
            "tov": stat.get("turnover"),
            "fg_made": stat.get("fgm"),
            "fg_att": stat.get("fga"),
            "fg_pct": stat.get("fg_pct"),
            "fg3_made": stat.get("fg3m"),
            "fg3_att": stat.get("fg3a"),
            "fg3_pct": stat.get("fg3_pct"),
            "ft_made": stat.get("ftm"),
            "ft_att": stat.get("fta"),
            "ft_pct": stat.get("ft_pct"),
            "plus_minus": None,  # Not in BDL stats
        })

    # Calculate season averages
    num_games = len(parsed_games)
    season_averages = {}
    if num_games > 0:
        season_averages = {
            "games_played": num_games,
            "min_avg": sum(g.get("min") or 0 for g in parsed_games) / num_games,
            "pts_avg": sum(g.get("pts") or 0 for g in parsed_games) / num_games,
            "reb_avg": sum(g.get("reb") or 0 for g in parsed_games) / num_games,
            "ast_avg": sum(g.get("ast") or 0 for g in parsed_games) / num_games,
            "stl_avg": sum(g.get("stl") or 0 for g in parsed_games) / num_games,
            "blk_avg": sum(g.get("blk") or 0 for g in parsed_games) / num_games,
            "tov_avg": sum(g.get("tov") or 0 for g in parsed_games) / num_games,
            "fg_pct": sum(g.get("fg_pct") or 0 for g in parsed_games) / num_games,
            "fg3_pct": sum(g.get("fg3_pct") or 0 for g in parsed_games) / num_games,
            "ft_pct": sum(g.get("ft_pct") or 0 for g in parsed_games) / num_games,
        }

    # Calculate last 5 averages
    last_5_averages = {}
    if num_games >= 1:
        recent = parsed_games[:5]
        n = len(recent)
        last_5_averages = {
            "games_count": n,
            "pts_avg": sum(g.get("pts") or 0 for g in recent) / n,
            "reb_avg": sum(g.get("reb") or 0 for g in recent) / n,
            "ast_avg": sum(g.get("ast") or 0 for g in recent) / n,
            "fg3_avg": sum(g.get("fg3_made") or 0 for g in recent) / n,
            "min_avg": sum(g.get("min") or 0 for g in recent) / n,
            "stl_avg": sum(g.get("stl") or 0 for g in recent) / n,
            "blk_avg": sum(g.get("blk") or 0 for g in recent) / n,
        }

    return {
        "player_id": bdl_player_id,
        "season": season,
        "season_averages": season_averages,
        "last_5_averages": last_5_averages,
        "game_log": parsed_games,
    }


def fetch_player_stats_before_date_bdl(
    player_id: int = None,
    player_name: str = None,
    before_date: str = None,
    season: int = None,
    last_n_games: int = None,
) -> dict:
    """
    TEMPORAL SAFE: Fetch player stats using only games BEFORE the specified date.

    Uses Balldontlie API for speed (600 req/min).

    Args:
        player_id: Balldontlie player ID (preferred)
        player_name: Player name (will be converted to BDL ID)
        before_date: Date string (YYYY-MM-DD) - only include games before this
        season: Season year
        last_n_games: Limit to last N games before the date

    Returns:
        Dictionary with player stats (same format as fetch_player_stats)
    """
    # Safety: No date means return empty to prevent leakage
    if before_date is None:
        return {
            "player_id": player_id,
            "season": season,
            "season_averages": {},
            "last_5_averages": {},
            "game_log": [],
            "games_used": 0,
            "temporal_cutoff": None,
        }

    api = _get_balldontlie_api()
    if not api:
        # Fall back to NBA API
        if player_name:
            nba_pid = get_player_id(player_name)
            if nba_pid:
                season_str = f"{season}-{str(season+1)[-2:]}" if season else "2025-26"
                return fetch_player_stats_before_date(nba_pid, season_str, before_date, last_n_games)
        return {
            "player_id": player_id,
            "season": season,
            "season_averages": {},
            "last_5_averages": {},
            "game_log": [],
            "games_used": 0,
            "temporal_cutoff": before_date,
        }

    # Convert player name to Balldontlie ID if needed
    bdl_player_id = player_id
    if bdl_player_id is None and player_name:
        mapper = _get_id_mapper()
        if mapper:
            bdl_player_id = mapper.get_player_id(player_name)

    if bdl_player_id is None:
        return {
            "player_id": player_id,
            "season": season,
            "season_averages": {},
            "last_5_averages": {},
            "game_log": [],
            "games_used": 0,
            "temporal_cutoff": before_date,
        }

    # Determine season
    if season is None:
        season = datetime.now().year if datetime.now().month > 9 else datetime.now().year - 1

    # Fetch stats before date using Balldontlie's temporal method
    filtered_stats = api.get_player_stats_before_date(
        bdl_player_id,
        before_date,
        season,
        last_n_games
    )

    if not filtered_stats:
        return {
            "player_id": bdl_player_id,
            "season": season,
            "season_averages": {},
            "last_5_averages": {},
            "game_log": [],
            "games_used": 0,
            "temporal_cutoff": before_date,
        }

    # Convert to standard format
    parsed_games = []
    for stat in filtered_stats:
        game = stat.get("game", {})
        parsed_games.append({
            "game_id": game.get("id"),
            "game_date": game.get("date", "")[:10] if game.get("date") else "",
            "matchup": f"{game.get('visitor_team', {}).get('abbreviation', '')} @ {game.get('home_team', {}).get('abbreviation', '')}",
            "wl": None,
            "min": _parse_minutes(stat.get("min")),
            "pts": stat.get("pts"),
            "reb": stat.get("reb"),
            "ast": stat.get("ast"),
            "stl": stat.get("stl"),
            "blk": stat.get("blk"),
            "tov": stat.get("turnover"),
            "fg_made": stat.get("fgm"),
            "fg_att": stat.get("fga"),
            "fg_pct": stat.get("fg_pct"),
            "fg3_made": stat.get("fg3m"),
            "fg3_att": stat.get("fg3a"),
            "fg3_pct": stat.get("fg3_pct"),
            "ft_made": stat.get("ftm"),
            "ft_att": stat.get("fta"),
            "ft_pct": stat.get("ft_pct"),
            "plus_minus": None,
        })

    # Calculate averages
    num_games = len(parsed_games)
    season_averages = {}
    if num_games > 0:
        season_averages = {
            "games_played": num_games,
            "min_avg": sum(g.get("min") or 0 for g in parsed_games) / num_games,
            "pts_avg": sum(g.get("pts") or 0 for g in parsed_games) / num_games,
            "reb_avg": sum(g.get("reb") or 0 for g in parsed_games) / num_games,
            "ast_avg": sum(g.get("ast") or 0 for g in parsed_games) / num_games,
            "stl_avg": sum(g.get("stl") or 0 for g in parsed_games) / num_games,
            "blk_avg": sum(g.get("blk") or 0 for g in parsed_games) / num_games,
            "tov_avg": sum(g.get("tov") or 0 for g in parsed_games) / num_games,
            "fg3_avg": sum(g.get("fg3_made") or 0 for g in parsed_games) / num_games,
            "fg_pct": sum(g.get("fg_pct") or 0 for g in parsed_games) / num_games,
            "fg3_pct": sum(g.get("fg3_pct") or 0 for g in parsed_games) / num_games,
            "ft_pct": sum(g.get("ft_pct") or 0 for g in parsed_games) / num_games,
        }

    last_5_averages = {}
    if num_games >= 1:
        recent = parsed_games[:5]
        n = len(recent)
        last_5_averages = {
            "games_count": n,
            "pts_avg": sum(g.get("pts") or 0 for g in recent) / n,
            "reb_avg": sum(g.get("reb") or 0 for g in recent) / n,
            "ast_avg": sum(g.get("ast") or 0 for g in recent) / n,
            "fg3_avg": sum(g.get("fg3_made") or 0 for g in recent) / n,
            "min_avg": sum(g.get("min") or 0 for g in recent) / n,
            "stl_avg": sum(g.get("stl") or 0 for g in recent) / n,
            "blk_avg": sum(g.get("blk") or 0 for g in recent) / n,
        }

    return {
        "player_id": bdl_player_id,
        "season": season,
        "season_averages": season_averages,
        "last_5_averages": last_5_averages,
        "game_log": parsed_games,
        "games_used": num_games,
        "temporal_cutoff": before_date,
    }


def fetch_season_averages_bdl(
    player_ids: list = None,
    player_names: list = None,
    season: int = None,
) -> dict:
    """
    Fetch season averages for multiple players using Balldontlie API.

    Much faster than NBA API for batch lookups.

    Args:
        player_ids: List of Balldontlie player IDs
        player_names: List of player names (will be converted to IDs)
        season: Season year

    Returns:
        Dictionary mapping player_id -> season averages
    """
    api = _get_balldontlie_api()
    if not api:
        return {}

    # Convert names to IDs if needed
    bdl_ids = list(player_ids) if player_ids else []
    if player_names:
        mapper = _get_id_mapper()
        if mapper:
            for name in player_names:
                pid = mapper.get_player_id(name)
                if pid and pid not in bdl_ids:
                    bdl_ids.append(pid)

    if not bdl_ids:
        return {}

    # Determine season
    if season is None:
        season = datetime.now().year if datetime.now().month > 9 else datetime.now().year - 1

    # Fetch season averages from Balldontlie
    averages = api.get_season_averages(season=season, player_ids=bdl_ids)

    # Convert to dictionary keyed by player_id
    result = {}
    for avg in averages:
        pid = avg.get("player_id")
        if pid:
            result[pid] = {
                "games_played": avg.get("games_played"),
                "min_avg": avg.get("min"),
                "pts_avg": avg.get("pts"),
                "reb_avg": avg.get("reb"),
                "ast_avg": avg.get("ast"),
                "stl_avg": avg.get("stl"),
                "blk_avg": avg.get("blk"),
                "tov_avg": avg.get("turnover"),
                "fg_pct": avg.get("fg_pct"),
                "fg3_pct": avg.get("fg3_pct"),
                "ft_pct": avg.get("ft_pct"),
            }

    return result


def fetch_injuries_bdl(
    team_abbrev: str = None,
    player_names: list = None,
) -> list:
    """
    Fetch current NBA injuries using Balldontlie API.

    Args:
        team_abbrev: Filter by team abbreviation
        player_names: Filter by specific player names

    Returns:
        List of injury dictionaries with player info and status
    """
    api = _get_balldontlie_api()
    if not api:
        return []

    # Get team ID if filtering by team
    team_ids = None
    if team_abbrev:
        from id_mapping import TEAM_ABBREV_TO_BDL
        tid = TEAM_ABBREV_TO_BDL.get(team_abbrev.upper())
        if tid:
            team_ids = [tid]

    # Fetch injuries
    injuries = api.get_injuries(team_ids=team_ids)

    # Build team_id -> abbreviation map for reverse lookup
    from id_mapping import TEAM_ABBREV_TO_BDL
    bdl_to_abbrev = {v: k for k, v in TEAM_ABBREV_TO_BDL.items()}

    result = []
    for inj in injuries:
        player = inj.get("player", {})
        # team_id is directly in the player object, not a nested team dict
        team_id = player.get("team_id")
        team_abbrev = bdl_to_abbrev.get(team_id, "") if team_id else ""

        injury_info = {
            "player_id": player.get("id"),
            "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
            "team_id": team_id,
            "team_abbrev": team_abbrev,
            "status": inj.get("status", ""),
            "comment": inj.get("description", "") or inj.get("comment", ""),
            "date": inj.get("return_date", "") or inj.get("date", ""),
        }

        # Filter by player names if specified
        if player_names:
            if injury_info["player_name"].lower() not in [n.lower() for n in player_names]:
                continue

        result.append(injury_info)

    return result


def get_player_injury_status(player_name: str) -> dict:
    """
    Check if a specific player is injured.

    Args:
        player_name: Player full name

    Returns:
        Dictionary with injury status or None if healthy
    """
    injuries = fetch_injuries_bdl()

    name_lower = player_name.lower()
    for inj in injuries:
        if inj.get("player_name", "").lower() == name_lower:
            return inj

    return None


# =============================================================================
# UNIFIED DATA FUNCTIONS (Auto-select best source)
# =============================================================================

def fetch_player_stats_auto(
    player_id: int = None,
    player_name: str = None,
    season: str = None,
    last_n_games: int = None,
    prefer_bdl: bool = True,
) -> dict:
    """
    Fetch player stats, automatically choosing the best data source.

    Prefers Balldontlie (faster) but falls back to NBA API if needed.

    Args:
        player_id: Player ID (NBA API or Balldontlie depending on prefer_bdl)
        player_name: Player name (works with either source)
        season: Season (e.g., "2025-26" or 2025)
        last_n_games: Limit to last N games
        prefer_bdl: If True, try Balldontlie first

    Returns:
        Player stats dictionary
    """
    if prefer_bdl:
        # Try Balldontlie first
        # Pass player_id as-is (Balldontlie IDs can be large like 17896075)
        result = fetch_player_stats_bdl(
            player_id=player_id,
            player_name=player_name,
            last_n_games=last_n_games,
        )
        if result.get("game_log"):
            return result

    # Fall back to NBA API
    nba_id = player_id
    if player_name and not nba_id:
        nba_id = get_player_id(player_name)

    if nba_id:
        return fetch_player_stats(nba_id, season=season or "2025-26", last_n_games=last_n_games)

    return {}


def fetch_player_stats_before_date_auto(
    player_id: int = None,
    player_name: str = None,
    before_date: str = None,
    season: str = None,
    last_n_games: int = None,
    prefer_bdl: bool = True,
) -> dict:
    """
    TEMPORAL SAFE: Fetch player stats before a date, auto-selecting source.

    Args:
        player_id: Player ID
        player_name: Player name
        before_date: Date cutoff (YYYY-MM-DD)
        season: Season
        last_n_games: Limit games
        prefer_bdl: Prefer Balldontlie API

    Returns:
        Player stats dictionary with temporal safety
    """
    if before_date is None:
        return {
            "player_id": player_id,
            "season": season,
            "season_averages": {},
            "last_5_averages": {},
            "game_log": [],
            "games_used": 0,
            "temporal_cutoff": None,
        }

    if prefer_bdl:
        # Pass player_id as-is (Balldontlie IDs can be large like 17896075)
        result = fetch_player_stats_before_date_bdl(
            player_id=player_id,
            player_name=player_name,
            before_date=before_date,
            last_n_games=last_n_games,
        )
        if result.get("games_used", 0) > 0:
            return result

    # Fall back to NBA API
    nba_id = player_id
    if player_name and not nba_id:
        nba_id = get_player_id(player_name)

    if nba_id:
        return fetch_player_stats_before_date(
            nba_id,
            season=season or "2025-26",
            before_date=before_date,
            last_n_games=last_n_games
        )

    return {
        "player_id": player_id,
        "season": season,
        "season_averages": {},
        "last_5_averages": {},
        "game_log": [],
        "games_used": 0,
        "temporal_cutoff": before_date,
    }


# =============================================================================
# CLUTCH STATISTICS (1-3% accuracy impact)
# Clutch = last 5 minutes of 4th quarter or OT, score within 5 points
# =============================================================================

def fetch_team_clutch_stats(team_id: int, season: str = "2025-26") -> dict[str, Any]:
    """
    Fetch team clutch performance statistics.

    Clutch time is defined as the last 5 minutes of the 4th quarter or overtime,
    with the score margin within 5 points. This data is critical for predicting
    close game outcomes.

    Args:
        team_id: NBA team ID
        season: Season string (e.g., "2025-26")

    Returns:
        Dict with clutch statistics:
        - clutch_net_rating: Points per 100 possessions differential in clutch
        - clutch_win_pct: Win percentage in clutch situations
        - clutch_fg_pct: Field goal percentage in clutch
        - clutch_plus_minus: Plus/minus in clutch minutes
        - clutch_games: Number of clutch games played
    """
    if not HAS_CLUTCH_ENDPOINTS:
        return _get_default_clutch_stats()

    cache_key = f"team_clutch_{team_id}_{season}"
    cached = _read_from_cache(cache_key)
    if cached:
        return cached

    try:
        time.sleep(API_DELAY)

        clutch_data = leaguedashteamclutch.LeagueDashTeamClutch(
            season=season,
            season_type_all_star="Regular Season",
            clutch_time="Last 5 Minutes",
            ahead_behind="Ahead or Behind",
            point_diff=5,  # Within 5 points
        ).get_normalized_dict()

        # Find our team in the results
        team_stats = None
        for row in clutch_data.get('LeagueDashTeamClutch', []):
            if row.get('TEAM_ID') == team_id:
                team_stats = row
                break

        if not team_stats:
            return _get_default_clutch_stats()

        result = {
            "clutch_net_rating": team_stats.get('NET_RATING', 0.0),
            "clutch_win_pct": team_stats.get('W_PCT', 0.5),
            "clutch_fg_pct": team_stats.get('FG_PCT', 0.45),
            "clutch_plus_minus": team_stats.get('PLUS_MINUS', 0.0),
            "clutch_games": team_stats.get('GP', 0),
            "clutch_pts_per_game": team_stats.get('PTS', 0.0),
            "clutch_available": True,
        }

        _write_to_cache(cache_key, result)
        return result

    except Exception as e:
        print(f"[CLUTCH] Error fetching team {team_id} clutch stats: {e}")
        return _get_default_clutch_stats()


def fetch_player_clutch_stats(player_id: int, season: str = "2025-26") -> dict[str, Any]:
    """
    Fetch player clutch performance statistics.

    Identifies "closers" - players who perform well in clutch situations.

    Args:
        player_id: NBA player ID
        season: Season string

    Returns:
        Dict with player clutch statistics
    """
    if not HAS_CLUTCH_ENDPOINTS:
        return _get_default_player_clutch_stats()

    cache_key = f"player_clutch_{player_id}_{season}"
    cached = _read_from_cache(cache_key)
    if cached:
        return cached

    try:
        time.sleep(API_DELAY)

        clutch_data = leaguedashplayerclutch.LeagueDashPlayerClutch(
            season=season,
            season_type_all_star="Regular Season",
            clutch_time="Last 5 Minutes",
            ahead_behind="Ahead or Behind",
            point_diff=5,
        ).get_normalized_dict()

        # Find our player in the results
        player_stats = None
        for row in clutch_data.get('LeagueDashPlayerClutch', []):
            if row.get('PLAYER_ID') == player_id:
                player_stats = row
                break

        if not player_stats:
            return _get_default_player_clutch_stats()

        result = {
            "player_clutch_pts": player_stats.get('PTS', 0.0),
            "player_clutch_fg_pct": player_stats.get('FG_PCT', 0.0),
            "player_clutch_plus_minus": player_stats.get('PLUS_MINUS', 0.0),
            "player_clutch_minutes": player_stats.get('MIN', 0.0),
            "is_closer": player_stats.get('PLUS_MINUS', 0.0) > 2.0,  # +2 or better in clutch
            "clutch_available": True,
        }

        _write_to_cache(cache_key, result)
        return result

    except Exception as e:
        print(f"[CLUTCH] Error fetching player {player_id} clutch stats: {e}")
        return _get_default_player_clutch_stats()


def _get_default_clutch_stats() -> dict[str, Any]:
    """Return default/neutral clutch stats when data is unavailable."""
    return {
        "clutch_net_rating": 0.0,
        "clutch_win_pct": 0.5,
        "clutch_fg_pct": 0.45,
        "clutch_plus_minus": 0.0,
        "clutch_games": 0,
        "clutch_pts_per_game": 0.0,
        "clutch_available": False,
    }


def _get_default_player_clutch_stats() -> dict[str, Any]:
    """Return default/neutral player clutch stats when data is unavailable."""
    return {
        "player_clutch_pts": 0.0,
        "player_clutch_fg_pct": 0.0,
        "player_clutch_plus_minus": 0.0,
        "player_clutch_minutes": 0.0,
        "is_closer": False,
        "clutch_available": False,
    }


def fetch_team_clutch_differential(
    home_team_id: int,
    away_team_id: int,
    season: str = "2025-26"
) -> dict[str, float]:
    """
    Calculate clutch performance differential between two teams.

    This is the primary feature for game predictions - teams that perform
    better in clutch situations have an edge in close games.

    Args:
        home_team_id: Home team NBA ID
        away_team_id: Away team NBA ID
        season: Season string

    Returns:
        Dict with differential features:
        - clutch_net_rating_diff: Home - Away clutch net rating
        - clutch_win_pct_diff: Home - Away clutch win percentage
        - clutch_fg_pct_diff: Home - Away clutch FG%
        - home_clutch_edge: Boolean if home team has clutch advantage
    """
    home_clutch = fetch_team_clutch_stats(home_team_id, season)
    away_clutch = fetch_team_clutch_stats(away_team_id, season)

    net_diff = home_clutch['clutch_net_rating'] - away_clutch['clutch_net_rating']
    win_diff = home_clutch['clutch_win_pct'] - away_clutch['clutch_win_pct']
    fg_diff = home_clutch['clutch_fg_pct'] - away_clutch['clutch_fg_pct']

    return {
        "clutch_net_rating_diff": round(net_diff, 2),
        "clutch_win_pct_diff": round(win_diff, 3),
        "clutch_fg_pct_diff": round(fg_diff, 3),
        "home_clutch_edge": 1 if net_diff > 3.0 else 0,  # >3 net rating = meaningful edge
        "clutch_data_available": home_clutch['clutch_available'] and away_clutch['clutch_available'],
    }


def main():
    """Main function to fetch, parse, and save NBA schedule."""
    # Fetch today's schedule
    games_data, date = fetch_todays_schedule()

    # Parse game details
    schedule = parse_game_details(games_data)

    if not schedule:
        print(f"No games scheduled for {date}")
    else:
        print(f"Found {len(schedule)} game(s) scheduled for {date}")
        for game in schedule:
            home = game["home_team"]
            visitor = game["visitor_team"]
            print(f"  {visitor['abbreviation']} @ {home['abbreviation']} - {game['status']}")

    # Save to JSON
    save_schedule_to_json(schedule, date)


if __name__ == "__main__":
    main()
