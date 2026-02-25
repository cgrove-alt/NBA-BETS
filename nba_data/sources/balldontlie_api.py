"""
Balldontlie NBA API Integration (Paid Tiers)

Full-featured API client supporting Free, All-Star ($9.99), and GOAT ($39.99) tiers.
Get your API key at: https://app.balldontlie.io

GOAT tier provides:
- Live betting odds from 6+ sportsbooks
- Real-time box scores
- Season averages & advanced stats
- Player props
- Team standings & leaders

Usage:
    from balldontlie_api import BalldontlieAPI

    api = BalldontlieAPI(api_key="your-key")

    # Get today's odds
    odds = api.get_betting_odds()

    # Get live box scores
    live = api.get_live_box_scores()

    # Get season averages
    stats = api.get_season_averages(player_ids=[1, 2, 3])
"""

import os
import requests
import time
import hashlib
import threading
import logging
from datetime import datetime, timedelta
from typing import Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# =============================================================================
# RESPONSE CACHING
# =============================================================================
# Cache API responses to disk to reduce API calls and improve speed.
# Different TTLs for different data types.

CACHE_DIR = Path(__file__).parent / ".bdl_cache"
CACHE_TTL = {
    "live": 60,          # 1 minute for live data (odds, box scores)
    "daily": 1800,       # 30 minutes for daily data (schedules, props)
    "stats": 3600,       # 1 hour for player stats
    "historical": 86400, # 24 hours for historical data
    "static": 604800,    # 1 week for static data (teams, players list)
}


def _get_cache_key(endpoint: str, params: dict = None) -> str:
    """Generate a unique cache key for an API request."""
    params_str = json.dumps(params or {}, sort_keys=True)
    key = f"{endpoint}:{params_str}"
    return hashlib.md5(key.encode()).hexdigest()


def _get_cache_path(cache_key: str) -> Path:
    """Get the file path for a cache entry."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{cache_key}.json"


def _read_cache(endpoint: str, params: dict = None, ttl_type: str = "stats") -> Any | None:
    """Read data from cache if valid."""
    cache_key = _get_cache_key(endpoint, params)
    cache_path = _get_cache_path(cache_key)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path) as f:
            cached = json.load(f)

        # Check TTL
        cached_at = cached.get("cached_at", 0)
        ttl = CACHE_TTL.get(ttl_type, 3600)
        if time.time() - cached_at > ttl:
            cache_path.unlink(missing_ok=True)
            return None

        return cached.get("data")
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def _write_cache(endpoint: str, params: dict, data: Any) -> None:
    """Write data to cache."""
    if not data:
        return

    cache_key = _get_cache_key(endpoint, params)
    cache_path = _get_cache_path(cache_key)

    try:
        with open(cache_path, "w") as f:
            json.dump({
                "cached_at": time.time(),
                "endpoint": endpoint,
                "data": data,
            }, f)
    except (OSError, TypeError):
        pass


def clear_bdl_cache(older_than_hours: float = 0) -> int:
    """
    Clear the Balldontlie API cache.

    Args:
        older_than_hours: Only clear entries older than this. If 0, clear all.

    Returns:
        Number of entries removed.
    """
    if not CACHE_DIR.exists():
        return 0

    removed = 0
    cutoff = time.time() - (older_than_hours * 3600) if older_than_hours > 0 else float('inf')

    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            if older_than_hours > 0:
                with open(cache_file) as f:
                    cached = json.load(f)
                    if cached.get("cached_at", 0) > cutoff:
                        continue
            cache_file.unlink()
            removed += 1
        except (OSError, json.JSONDecodeError):
            try:
                cache_file.unlink()
                removed += 1
            except OSError:
                pass

    return removed


# Load .env file if exists
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key not in os.environ:
                        os.environ[key] = value

_load_env()

# API Configuration
BASE_URL_V1 = "https://api.balldontlie.io/v1"
BASE_URL_V2 = "https://api.balldontlie.io/v2"  # For newer endpoints like odds


class BalldontlieAPI:
    """
    Balldontlie NBA API client with paid tier support.

    Tiers:
    - Free: Teams, Players, Games (5 req/min)
    - All-Star ($9.99): + Player stats, Injuries (60 req/min)
    - GOAT ($39.99): + Odds, Box scores, Advanced stats (600 req/min)
    """

    # Rate limits by tier
    RATE_LIMITS = {
        "free": 5,      # 5 requests per minute
        "allstar": 60,  # 60 requests per minute
        "goat": 600,    # 600 requests per minute
    }

    def __init__(self, api_key: str | None = None, tier: str = "goat"):
        """
        Initialize the Balldontlie client.

        Args:
            api_key: API key from https://app.balldontlie.io
                     Falls back to BALLDONTLIE_API_KEY env variable
            tier: Your subscription tier ("free", "allstar", or "goat")
        """
        self.api_key = api_key or os.environ.get("BALLDONTLIE_API_KEY")
        self.tier = tier.lower()

        if not self.api_key:
            raise ValueError(
                "API key required. Set BALLDONTLIE_API_KEY environment variable "
                "or pass api_key parameter. Sign up at: https://app.balldontlie.io"
            )

        self.headers = {
            "Authorization": self.api_key,
            "Accept": "application/json",
        }

        # Rate limiting — thread-safe with 20% safety margin
        self.requests_per_minute = self.RATE_LIMITS.get(self.tier, 60)
        # Use 80% of allowed rate to leave headroom and avoid 429s
        effective_rate = self.requests_per_minute * 0.80
        self.min_delay = 60.0 / effective_rate
        self._last_request = 0
        self._rate_lock = threading.Lock()

    def _rate_limit(self):
        """Ensure we don't exceed rate limits. Thread-safe."""
        with self._rate_lock:
            now = time.time()
            elapsed = now - self._last_request
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
            self._last_request = time.time()

    def _get(self, endpoint: str, params: dict = None, version: int = 1,
             cache_ttl: str = None, _retry_count: int = 0) -> Any:
        """
        Make a GET request to the API with optional caching.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            version: API version (1 or 2)
            cache_ttl: Cache TTL type ("live", "daily", "stats", "historical", "static")
                       If None, no caching is used.
            _retry_count: Internal retry counter for 429 handling

        Returns:
            JSON response data
        """
        MAX_429_RETRIES = 3

        # Check cache first (if caching enabled for this request)
        if cache_ttl:
            cached = _read_cache(endpoint, params, cache_ttl)
            if cached is not None:
                return cached

        self._rate_limit()
        base_url = BASE_URL_V2 if version == 2 else BASE_URL_V1
        url = f"{base_url}/{endpoint}"

        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=30
            )

            if response.status_code == 401:
                raise ValueError("Invalid API key or insufficient permissions")
            if response.status_code == 403:
                raise ValueError("This endpoint requires a higher tier subscription")
            if response.status_code == 429:
                if _retry_count >= MAX_429_RETRIES:
                    logger.error(f"BDL rate limited {MAX_429_RETRIES} times on {endpoint}, giving up")
                    return None
                # Exponential backoff: 5s, 15s, 45s
                wait = 5 * (3 ** _retry_count)
                logger.warning(f"BDL rate limited on {endpoint} (attempt {_retry_count + 1}/{MAX_429_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
                return self._get(endpoint, params, version, cache_ttl, _retry_count + 1)
            if response.status_code != 200:
                logger.warning(f"BDL API error {response.status_code}: {response.text[:200]}")
                return None

            data = response.json()

            # Validate response shape (warn-only, don't break callers)
            try:
                from nba_data.validators.schemas import validate_bdl_response
                if not validate_bdl_response(data, context=endpoint):
                    import logging
                    logging.getLogger(__name__).warning(
                        f"BDL response validation warning for {endpoint}"
                    )
            except ImportError:
                pass

            # Cache successful response
            if cache_ttl and data:
                _write_cache(endpoint, params, data)

            return data

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    # ==================== FREE TIER ====================

    def get_teams(self) -> list[dict]:
        """Get all NBA teams (cached for 1 week)."""
        data = self._get("teams", cache_ttl="static")
        return data.get("data", []) if data else []

    def get_team(self, team_id: int) -> dict:
        """Get a specific team by ID (cached for 1 week)."""
        data = self._get(f"teams/{team_id}", cache_ttl="static")
        return data.get("data", {}) if data else {}

    def get_players(self, search: str = None, per_page: int = 100) -> list[dict]:
        """
        Get players.

        Args:
            search: Search by player name
            per_page: Results per page (max 100)

        Returns:
            List of player dictionaries
        """
        params = {"per_page": per_page}
        if search:
            params["search"] = search

        data = self._get("players", params)
        return data.get("data", []) if data else []

    def get_player(self, player_id: int) -> dict:
        """Get a specific player by ID (cached 1 week — player metadata rarely changes)."""
        data = self._get(f"players/{player_id}", cache_ttl="static")
        return data.get("data", {}) if data else {}

    def get_games(
        self,
        dates: list[str] = None,
        seasons: list[int] = None,
        team_ids: list[int] = None,
        per_page: int = 100,
    ) -> list[dict]:
        """
        Get games.

        Args:
            dates: List of dates (YYYY-MM-DD)
            seasons: List of season years (e.g., [2024, 2025])
            team_ids: Filter by team IDs
            per_page: Results per page

        Returns:
            List of game dictionaries
        """
        params = {"per_page": per_page}

        if dates:
            params["dates[]"] = dates
        if seasons:
            params["seasons[]"] = seasons
        if team_ids:
            params["team_ids[]"] = team_ids

        data = self._get("games", params)
        return data.get("data", []) if data else []

    def get_todays_games(self) -> list[dict]:
        """Get all games scheduled for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_games(dates=[today])

    def get_upcoming_games(
        self,
        team_id: int,
        from_date: str = None,
        days_ahead: int = 7
    ) -> list[dict]:
        """
        Get upcoming games for a specific team.

        Used for trap game detection - need to know who's playing next
        to detect lookahead spots (weak opponent before elite team).

        Args:
            team_id: Team ID to get upcoming games for
            from_date: Start date (YYYY-MM-DD), defaults to today
            days_ahead: Number of days to look ahead (default 7)

        Returns:
            List of upcoming games sorted by date

        Example:
            >>> api = BalldontlieAPI()
            >>> upcoming = api.get_upcoming_games(1610612738, days_ahead=7)  # BOS
            >>> for game in upcoming:
            ...     print(game['date'], game['home_team']['abbreviation'])
        """
        if from_date is None:
            from_date = datetime.now().strftime("%Y-%m-%d")

        # Generate list of dates to query
        try:
            start = datetime.strptime(from_date, "%Y-%m-%d")
        except ValueError:
            start = datetime.now()

        # Generate dates from tomorrow to days_ahead
        dates = [
            (start + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, days_ahead + 1)
        ]

        if not dates:
            return []

        # Fetch games for these dates and filter by team
        all_games = self.get_games(dates=dates, team_ids=[team_id])

        # Sort by date
        return sorted(
            all_games,
            key=lambda g: g.get('date', '') or g.get('datetime', '') or ''
        )


    # ==================== ALL-STAR TIER ($9.99) ====================

    def get_player_stats(
        self,
        player_ids: list[int] = None,
        game_ids: list[int] = None,
        dates: list[str] = None,
        seasons: list[int] = None,
        per_page: int = 100,
    ) -> list[dict]:
        """
        Get player game statistics.

        Requires: All-Star tier or higher

        Args:
            player_ids: Filter by player IDs
            game_ids: Filter by game IDs
            dates: Filter by dates
            seasons: Filter by seasons
            per_page: Results per page

        Returns:
            List of player stat lines
        """
        params = {"per_page": per_page}

        if player_ids:
            params["player_ids[]"] = player_ids
        if game_ids:
            params["game_ids[]"] = game_ids
        if dates:
            params["dates[]"] = dates
        if seasons:
            params["seasons[]"] = seasons

        data = self._get("stats", params)
        return data.get("data", []) if data else []

    def get_player_stats_paginated(
        self,
        player_id: int,
        season: int = None,
        max_pages: int = 10,
    ) -> list[dict]:
        """
        Fetch ALL player game stats with pagination.

        Requires: All-Star tier or higher.
        Cached for 1 hour (stats TTL) to avoid redundant fetches within a run.

        Args:
            player_id: Player ID to fetch stats for
            season: Season year (defaults to current)
            max_pages: Maximum pages to fetch (safety limit)

        Returns:
            List of all player stat lines for the season
        """
        if season is None:
            season = datetime.now().year if datetime.now().month > 9 else datetime.now().year - 1

        # Check composite cache for the full paginated result
        cache_endpoint = f"stats_paginated/{player_id}/{season}"
        cached = _read_cache(cache_endpoint, None, "stats")
        if cached is not None:
            return cached

        all_stats = []
        cursor = None
        page = 0

        while page < max_pages:
            params = {
                "player_ids[]": [player_id],
                "seasons[]": [season],
                "per_page": 100,
            }
            if cursor:
                params["cursor"] = cursor

            data = self._get("stats", params, cache_ttl="stats")
            if not data:
                break

            stats = data.get("data", [])
            if not stats:
                break

            all_stats.extend(stats)

            # Check for next page
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

            page += 1

        # Cache the full composite result
        if all_stats:
            _write_cache(cache_endpoint, None, all_stats)

        return all_stats

    def get_player_stats_before_date(
        self,
        player_id: int,
        before_date: str,
        season: int = None,
        last_n_games: int = None,
    ) -> list[dict]:
        """
        Fetch player stats for games BEFORE a specific date.

        This is critical for temporal safety during model training -
        ensures we only use data available before the prediction date.

        Requires: All-Star tier or higher

        Args:
            player_id: Player ID to fetch stats for
            before_date: Date string (YYYY-MM-DD) - only return games before this
            season: Season year (defaults to current)
            last_n_games: If specified, only return the N most recent games

        Returns:
            List of player stat lines before the date, sorted by date descending
        """
        # Get all stats for the season
        all_stats = self.get_player_stats_paginated(player_id, season)

        # Filter to games before the specified date
        cutoff = datetime.strptime(before_date, "%Y-%m-%d")
        filtered_stats = []

        for stat in all_stats:
            game = stat.get("game", {})
            game_date_str = game.get("date", "")
            if game_date_str:
                try:
                    # Handle ISO format dates
                    if "T" in game_date_str:
                        game_date_str = game_date_str.split("T")[0]
                    game_date = datetime.strptime(game_date_str, "%Y-%m-%d")
                    if game_date < cutoff:
                        filtered_stats.append(stat)
                except ValueError:
                    continue

        # Sort by date descending (most recent first)
        filtered_stats.sort(
            key=lambda x: x.get("game", {}).get("date", ""),
            reverse=True
        )

        # Optionally limit to last N games
        if last_n_games and len(filtered_stats) > last_n_games:
            filtered_stats = filtered_stats[:last_n_games]

        return filtered_stats

    def get_players_paginated(
        self,
        search: str = None,
        team_ids: list[int] = None,
        max_pages: int = 20,
    ) -> list[dict]:
        """
        Fetch ALL players with pagination.

        Args:
            search: Search by player name
            team_ids: Filter by team IDs
            max_pages: Maximum pages to fetch

        Returns:
            List of all matching players
        """
        all_players = []
        cursor = None
        page = 0

        while page < max_pages:
            params = {"per_page": 100}
            if search:
                params["search"] = search
            if team_ids:
                params["team_ids[]"] = team_ids
            if cursor:
                params["cursor"] = cursor

            data = self._get("players", params)
            if not data:
                break

            players = data.get("data", [])
            if not players:
                break

            all_players.extend(players)

            # Check for next page
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

            page += 1

        return all_players

    def get_all_active_players(self) -> list[dict]:
        """
        Fetch ALL active NBA players with pagination.

        Requires: All-Star tier or higher

        Returns:
            List of all active player dictionaries
        """
        all_players = []
        cursor = None

        while True:
            params = {"per_page": 100}
            if cursor:
                params["cursor"] = cursor

            data = self._get("players/active", params)
            if not data:
                break

            players = data.get("data", [])
            if not players:
                break

            all_players.extend(players)

            # Check for next page
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        return all_players

    def get_active_players(self) -> list[dict]:
        """
        Get all active NBA players.

        Requires: All-Star tier or higher
        """
        data = self._get("players/active")
        return data.get("data", []) if data else []

    def get_injuries(self, team_ids: list[int] = None, player_ids: list[int] = None) -> list[dict]:
        """
        Get current player injuries.

        Requires: All-Star tier or higher

        Args:
            team_ids: Filter by team IDs
            player_ids: Filter by player IDs

        Returns:
            List of injury reports
        """
        params = {"per_page": 100}
        if team_ids:
            params["team_ids[]"] = team_ids
        if player_ids:
            params["player_ids[]"] = player_ids

        data = self._get("player_injuries", params)
        return data.get("data", []) if data else []

    # ==================== GOAT TIER ($39.99) ====================

    def get_betting_odds(
        self,
        date: str = None,
        game_ids: list[int] = None,
    ) -> list[dict]:
        """
        Get betting odds from multiple sportsbooks.

        Requires: GOAT tier

        Returns odds from: DraftKings, FanDuel, Caesars, BetMGM, Bet365, Betway

        Args:
            date: Date in YYYY-MM-DD format (defaults to today)
            game_ids: List of specific game IDs

        Returns:
            List of odds data with spreads, moneylines, totals
        """
        # API requires at least one of: dates[] or game_ids[]
        if date is None and game_ids is None:
            date = datetime.now().strftime("%Y-%m-%d")

        params = {"per_page": 100}
        if date:
            params["dates[]"] = date  # API uses dates[] not date
        if game_ids:
            params["game_ids[]"] = game_ids

        # Use v2 API for odds endpoint
        data = self._get("odds", params, version=2)
        return data.get("data", []) if data else []

    def get_todays_odds(self) -> list[dict]:
        """Get betting odds for today's games."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_betting_odds(date=today)

    def get_live_box_scores(self) -> list[dict]:
        """
        Get live box scores for today's games.

        Requires: GOAT tier

        Returns real-time player and team stats for games in progress.
        """
        data = self._get("box_scores/live")
        return data.get("data", []) if data else []

    def get_box_score(self, game_id: int) -> dict:
        """
        Get box score for a specific game.

        Requires: GOAT tier

        Args:
            game_id: Game ID

        Returns:
            Box score with player stats
        """
        data = self._get(f"box_scores/{game_id}")
        return data.get("data", {}) if data else {}

    def get_season_averages(
        self,
        season: int = None,
        player_ids: list[int] = None,
    ) -> list[dict]:
        """
        Get season averages for players.

        Requires: GOAT tier

        Note: API only supports single player_id per request,
        so we loop through player_ids and combine results.

        Args:
            season: Season year (defaults to current)
            player_ids: List of player IDs

        Returns:
            List of player season averages
        """
        if season is None:
            season = datetime.now().year if datetime.now().month > 9 else datetime.now().year - 1

        results = []

        if player_ids:
            for pid in player_ids:
                params = {"season": season, "player_id": pid}
                data = self._get("season_averages", params, cache_ttl="stats")
                if data and data.get("data"):
                    results.extend(data.get("data", []))
        else:
            return []

        return results

    def get_standings(self, season: int = None) -> list[dict]:
        """
        Get current team standings.

        Requires: GOAT tier

        Args:
            season: Season year (defaults to current)

        Returns:
            List of team standings
        """
        if season is None:
            season = datetime.now().year if datetime.now().month > 9 else datetime.now().year - 1

        data = self._get("standings", {"season": season})
        return data.get("data", []) if data else []

    def get_leaders(
        self,
        stat: str = "pts",
        season: int = None,
        per_page: int = 25,
    ) -> list[dict]:
        """
        Get league leaders for a specific stat.

        Requires: GOAT tier

        Args:
            stat: Stat category (pts, ast, reb, stl, blk, etc.)
            season: Season year
            per_page: Number of leaders to return

        Returns:
            List of league leaders
        """
        if season is None:
            season = datetime.now().year if datetime.now().month > 9 else datetime.now().year - 1

        params = {"stat_type": stat, "season": season, "per_page": per_page}
        data = self._get("leaders", params)
        return data.get("data", []) if data else []

    def get_player_props(self, game_id: int) -> list[dict]:
        """
        Get player prop bets for a game.

        Requires: GOAT tier

        Args:
            game_id: Game ID

        Returns:
            List of player props with lines and odds
        """
        data = self._get("odds/player_props", params={"game_id": game_id}, version=2, cache_ttl="daily")
        return data.get("data", []) if data else []

    def get_advanced_stats(
        self,
        game_id: int = None,
        player_id: int = None,
    ) -> list[dict]:
        """
        Get advanced statistics.

        Requires: GOAT tier

        Args:
            game_id: Specific game ID
            player_id: Specific player ID

        Returns:
            List of advanced stats
        """
        params = {}
        if game_id:
            params["game_id"] = game_id
        if player_id:
            params["player_id"] = player_id

        data = self._get("stats/advanced", params)
        return data.get("data", []) if data else []


# ==================== CONVENIENCE FUNCTIONS ====================

def get_todays_nba_odds(api_key: str | None = None) -> list[dict]:
    """Quick function to get today's NBA betting odds."""
    api = BalldontlieAPI(api_key)
    return api.get_todays_odds()


def get_live_scores(api_key: str | None = None) -> list[dict]:
    """Quick function to get live NBA scores."""
    api = BalldontlieAPI(api_key)
    return api.get_live_box_scores()


def get_nba_injuries(api_key: str | None = None) -> list[dict]:
    """Quick function to get current NBA injuries."""
    api = BalldontlieAPI(api_key)
    return api.get_injuries()


# ==================== ODDS FORMATTING ====================

def format_odds_for_model(raw_odds: list[dict]) -> dict[str, dict]:
    """
    Format Balldontlie odds into our standard format.

    Args:
        raw_odds: Raw odds from API

    Returns:
        Dictionary keyed by "away@home" with odds data
    """
    formatted = {}

    for game_odds in raw_odds:
        game = game_odds.get("game", {})
        home_team = game.get("home_team", {}).get("abbreviation", "")
        away_team = game.get("visitor_team", {}).get("abbreviation", "")

        if not home_team or not away_team:
            continue

        key = f"{away_team}@{home_team}"

        # Extract odds from first available sportsbook
        odds_data = game_odds.get("odds", [])
        if not odds_data:
            continue

        # Use first sportsbook (usually DraftKings)
        book = odds_data[0]

        formatted[key] = {
            "moneyline": {
                "home": book.get("home_ml", -110),
                "away": book.get("away_ml", -110),
            },
            "spread": {
                "home_line": book.get("spread", -3.5),
                "home_odds": book.get("home_spread_odds", -110),
                "away_line": -book.get("spread", 3.5) if book.get("spread") else 3.5,
                "away_odds": book.get("away_spread_odds", -110),
            },
            "total": {
                "line": book.get("total", 220.0),
                "over_odds": book.get("over_odds", -110),
                "under_odds": book.get("under_odds", -110),
            },
            "sportsbook": book.get("vendor", "Unknown"),
            "game_id": game.get("id"),
            "game_time": game.get("date"),
        }

    return formatted


# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Balldontlie NBA API")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--odds", action="store_true", help="Get today's odds")
    parser.add_argument("--live", action="store_true", help="Get live scores")
    parser.add_argument("--injuries", action="store_true", help="Get injuries")
    parser.add_argument("--standings", action="store_true", help="Get standings")
    parser.add_argument("--games", action="store_true", help="Get today's games")
    parser.add_argument("--test", action="store_true", help="Test API connection")

    args = parser.parse_args()

    try:
        api = BalldontlieAPI(args.api_key)

        if args.test:
            print("=" * 60)
            print("Balldontlie API Test")
            print("=" * 60)

            # Test free tier
            print("\n1. Testing Teams (Free)...")
            teams = api.get_teams()
            if teams:
                print(f"   Found {len(teams)} teams")
            else:
                print("   Failed")

            # Test games
            print("\n2. Testing Today's Games (Free)...")
            games = api.get_todays_games()
            print(f"   Found {len(games)} games today")
            for g in games[:3]:
                home = g.get("home_team", {}).get("abbreviation", "")
                away = g.get("visitor_team", {}).get("abbreviation", "")
                status = g.get("status", "")
                print(f"   - {away} @ {home} ({status})")

            # Test paid tier - Injuries (All-Star)
            print("\n3. Testing Injuries (All-Star tier)...")
            try:
                injuries = api.get_injuries()
                print(f"   Found {len(injuries)} injuries")
            except ValueError as e:
                print(f"   {e}")

            # Test paid tier - Odds (GOAT)
            print("\n4. Testing Betting Odds (GOAT tier)...")
            try:
                odds = api.get_todays_odds()
                print(f"   Found odds for {len(odds)} games")
                if odds:
                    formatted = format_odds_for_model(odds)
                    for matchup, data in list(formatted.items())[:2]:
                        spread = data["spread"]["home_line"]
                        total = data["total"]["line"]
                        print(f"   - {matchup}: Spread {spread:+.1f}, O/U {total}")
            except ValueError as e:
                print(f"   {e}")

            # Test live box scores (GOAT)
            print("\n5. Testing Live Box Scores (GOAT tier)...")
            try:
                live = api.get_live_box_scores()
                print(f"   Found {len(live)} live games")
            except ValueError as e:
                print(f"   {e}")

            print("\n" + "=" * 60)
            print("Test complete!")
            print("=" * 60)

        elif args.odds:
            print("Today's NBA Betting Odds:")
            print("=" * 50)
            odds = api.get_todays_odds()
            formatted = format_odds_for_model(odds)

            for matchup, data in formatted.items():
                print(f"\n{matchup}")
                print(f"  Spread: {data['spread']['home_line']:+.1f} ({data['spread']['home_odds']})")
                print(f"  Total: {data['total']['line']} (O: {data['total']['over_odds']}, U: {data['total']['under_odds']})")
                print(f"  ML: Home {data['moneyline']['home']}, Away {data['moneyline']['away']}")
                print(f"  Book: {data['sportsbook']}")

        elif args.live:
            print("Live NBA Games:")
            print("=" * 50)
            live = api.get_live_box_scores()
            if not live:
                print("No games in progress")
            for game in live:
                home = game.get("home_team", {}).get("abbreviation", "")
                away = game.get("visitor_team", {}).get("abbreviation", "")
                home_score = game.get("home_team_score", 0)
                away_score = game.get("visitor_team_score", 0)
                print(f"  {away} {away_score} @ {home} {home_score}")

        elif args.injuries:
            print("Current NBA Injuries:")
            print("=" * 50)
            injuries = api.get_injuries()
            for inj in injuries[:15]:
                player = inj.get("player", {})
                name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
                status = inj.get("status", "")
                print(f"  {name}: {status}")

        elif args.standings:
            print("NBA Standings:")
            print("=" * 50)
            standings = api.get_standings()
            for team in sorted(standings, key=lambda x: x.get("wins", 0), reverse=True)[:10]:
                name = team.get("team", {}).get("abbreviation", "")
                wins = team.get("wins", 0)
                losses = team.get("losses", 0)
                print(f"  {name}: {wins}-{losses}")

        elif args.games:
            print("Today's NBA Games:")
            print("=" * 50)
            games = api.get_todays_games()
            for g in games:
                home = g.get("home_team", {}).get("abbreviation", "")
                away = g.get("visitor_team", {}).get("abbreviation", "")
                status = g.get("status", "")
                game_time = g.get("date", "")
                print(f"  {away} @ {home} - {status} ({game_time})")

        else:
            print("Use --help for options, or --test to test API connection")

    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo get an API key:")
        print("1. Visit https://app.balldontlie.io")
        print("2. Sign up and get your API key")
        print("3. Set BALLDONTLIE_API_KEY environment variable")
        print("4. For full features, upgrade to GOAT tier ($39.99/mo)")
