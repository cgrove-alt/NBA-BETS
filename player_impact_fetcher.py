"""
NBA Player Impact Metrics Fetcher

Fetches advanced player impact metrics for injury-adjusted team ratings:
- EPM (Estimated Plus-Minus) from Dunks & Threes
- Basic stats from nba_api as fallback

Usage:
    fetcher = PlayerImpactFetcher()
    epm_data = fetcher.get_epm_ratings()
    team_rating = fetcher.calculate_team_rating(team_players, injuries)
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Try importing nba_api for fallback
try:
    from nba_api.stats.endpoints import leaguedashplayerstats
    from nba_api.stats.static import players
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False
    print("Note: nba_api not installed. Install with: pip install nba_api")


# Cache directory for player stats
CACHE_DIR = Path("player_impact_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Cache expiry (24 hours)
CACHE_EXPIRY_HOURS = 24


class PlayerImpactFetcher:
    """
    Fetches and caches player impact metrics.

    Primary source: EPM from Dunks & Threes
    Fallback: Basic plus/minus from nba_api
    """

    def __init__(self):
        self.epm_cache: Dict[str, Dict] = {}
        self.basic_stats_cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached player data."""
        epm_cache_file = CACHE_DIR / "epm_cache.json"
        if epm_cache_file.exists():
            try:
                with open(epm_cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is still valid
                    cache_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
                    if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                        self.epm_cache = data.get('players', {})
                        print(f"Loaded {len(self.epm_cache)} players from EPM cache")
            except Exception as e:
                print(f"Error loading EPM cache: {e}")

    def _save_cache(self):
        """Save player data to cache."""
        epm_cache_file = CACHE_DIR / "epm_cache.json"
        try:
            with open(epm_cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'players': self.epm_cache
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving EPM cache: {e}")

    def fetch_epm_from_dunks_and_threes(self, season: int = 2025) -> Dict[str, Dict]:
        """
        Fetch EPM data from Dunks & Threes website.

        EPM (Estimated Plus-Minus) is one of the best modern impact metrics,
        estimating player contribution in points per 100 possessions.

        Args:
            season: Season year (e.g., 2025 for 2024-25 season)

        Returns:
            Dictionary mapping player name to stats
        """
        # Dunks & Threes doesn't have a public API, so we'll try scraping
        # or using their data export if available

        url = f"https://dunksandthrees.com/epm"

        try:
            # Try to fetch the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"Could not fetch EPM data: HTTP {response.status_code}")
                return {}

            # The actual data would need to be parsed from the HTML
            # For now, we'll note that this requires more complex scraping
            # and fall back to nba_api
            print("EPM scraping requires JavaScript rendering - using fallback")
            return {}

        except Exception as e:
            print(f"Error fetching EPM: {e}")
            return {}

    def fetch_basic_stats_from_nba_api(self, season: str = "2024-25") -> Dict[str, Dict]:
        """
        Fetch basic player stats from nba_api as fallback.

        Includes: PPG, RPG, APG, Plus/Minus, Minutes, etc.

        Args:
            season: Season string (e.g., "2024-25")

        Returns:
            Dictionary mapping player name to stats
        """
        if not HAS_NBA_API:
            print("nba_api not available")
            return {}

        print(f"Fetching player stats from nba_api for {season}...")

        try:
            # Add delay to avoid rate limiting
            time.sleep(1)

            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame'
            )

            df = stats.get_data_frames()[0]

            players_dict = {}
            for _, row in df.iterrows():
                player_name = row['PLAYER_NAME']
                players_dict[player_name] = {
                    'player_id': row['PLAYER_ID'],
                    'team': row['TEAM_ABBREVIATION'],
                    'games': row['GP'],
                    'minutes': row['MIN'],
                    'points': row['PTS'],
                    'rebounds': row['REB'],
                    'assists': row['AST'],
                    'plus_minus': row['PLUS_MINUS'],
                    'net_rating': row.get('NET_RATING', 0),
                    # Estimate impact using plus/minus and minutes
                    'estimated_impact': row['PLUS_MINUS'] / max(row['MIN'], 1) * 36  # Per-36 plus/minus
                }

            print(f"Fetched stats for {len(players_dict)} players")
            self.basic_stats_cache = players_dict
            return players_dict

        except Exception as e:
            print(f"Error fetching from nba_api: {e}")
            return {}

    def get_player_impact(self, player_name: str) -> Optional[Dict]:
        """
        Get impact metrics for a specific player.

        Args:
            player_name: Player's full name

        Returns:
            Dictionary with player impact metrics or None
        """
        # Check EPM cache first
        if player_name in self.epm_cache:
            return self.epm_cache[player_name]

        # Check basic stats cache
        if player_name in self.basic_stats_cache:
            return self.basic_stats_cache[player_name]

        # Try to fetch if caches are empty
        if not self.basic_stats_cache and HAS_NBA_API:
            self.fetch_basic_stats_from_nba_api()
            return self.basic_stats_cache.get(player_name)

        return None

    def calculate_team_rating_adjustment(
        self,
        team_abbrev: str,
        injured_players: List[str] = None,
        resting_players: List[str] = None
    ) -> float:
        """
        Calculate team rating adjustment based on player availability.

        Uses player impact metrics to estimate how much a team's rating
        should be adjusted when key players are out.

        Args:
            team_abbrev: Team abbreviation (e.g., "LAL")
            injured_players: List of injured player names
            resting_players: List of players resting

        Returns:
            Rating adjustment (negative = team weaker without players)
        """
        injured_players = injured_players or []
        resting_players = resting_players or []
        unavailable = set(injured_players + resting_players)

        if not unavailable:
            return 0.0

        total_adjustment = 0.0

        for player_name in unavailable:
            impact = self.get_player_impact(player_name)
            if impact:
                # Use estimated impact (points per 36 minutes equivalent)
                player_impact = impact.get('estimated_impact', 0)
                minutes = impact.get('minutes', 0)

                # Weight by minutes played (more minutes = bigger impact when out)
                minutes_weight = min(minutes / 36, 1.0)  # Cap at 1.0 for 36+ MPG players

                # Adjustment is negative (team gets worse when player is out)
                # Scale factor: 1 point of plus/minus ≈ 0.5 points of team rating
                adjustment = -abs(player_impact) * minutes_weight * 0.5
                total_adjustment += adjustment

        return round(total_adjustment, 2)

    def get_team_roster_impacts(self, team_abbrev: str) -> List[Dict]:
        """
        Get sorted list of players by impact for a team.

        Args:
            team_abbrev: Team abbreviation

        Returns:
            List of players sorted by impact (highest first)
        """
        team_players = []

        # Search in basic stats cache
        for name, stats in self.basic_stats_cache.items():
            if stats.get('team') == team_abbrev:
                team_players.append({
                    'name': name,
                    'minutes': stats.get('minutes', 0),
                    'impact': stats.get('estimated_impact', 0),
                    'points': stats.get('points', 0),
                    'plus_minus': stats.get('plus_minus', 0),
                })

        # Sort by estimated impact
        team_players.sort(key=lambda x: x['impact'], reverse=True)

        return team_players

    def refresh_data(self):
        """Refresh all player data from sources."""
        print("Refreshing player impact data...")

        # Try EPM first (preferred)
        epm_data = self.fetch_epm_from_dunks_and_threes()
        if epm_data:
            self.epm_cache = epm_data
            self._save_cache()
            return

        # Fall back to nba_api
        if HAS_NBA_API:
            self.fetch_basic_stats_from_nba_api()


# Simple impact estimates for star players when API unavailable
# These are rough estimates based on public data
STAR_PLAYER_IMPACTS = {
    # MVP-caliber players (4+ points of impact when out)
    "Nikola Jokic": 5.0,
    "Luka Doncic": 4.5,
    "Giannis Antetokounmpo": 4.5,
    "Joel Embiid": 4.5,
    "Shai Gilgeous-Alexander": 4.0,
    "Jayson Tatum": 4.0,
    "Anthony Davis": 4.0,
    "Kevin Durant": 4.0,
    "LeBron James": 4.0,
    "Stephen Curry": 4.0,

    # All-Star caliber (2-4 points)
    "Donovan Mitchell": 3.5,
    "Trae Young": 3.5,
    "Anthony Edwards": 3.5,
    "Devin Booker": 3.5,
    "Ja Morant": 3.5,
    "Damian Lillard": 3.5,
    "De'Aaron Fox": 3.0,
    "Tyrese Haliburton": 3.0,
    "Tyrese Maxey": 3.0,
    "Paolo Banchero": 3.0,
    "Chet Holmgren": 3.0,
    "Victor Wembanyama": 3.5,
    "Jalen Brunson": 3.0,
    "Domantas Sabonis": 3.0,
    "James Harden": 3.0,
    "Bradley Beal": 2.5,
    "Kawhi Leonard": 3.5,
    "Paul George": 3.0,
    "Jimmy Butler": 3.5,
    "Bam Adebayo": 3.0,
    "Karl-Anthony Towns": 3.0,
    "Lauri Markkanen": 2.5,

    # Quality starters (1-2 points)
    "Scottie Barnes": 2.5,
    "Franz Wagner": 2.5,
    "Evan Mobley": 2.5,
    "Jaren Jackson Jr.": 2.5,
    "Zion Williamson": 3.0,
    "Brandon Ingram": 2.5,
    "CJ McCollum": 2.0,
    "Dejounte Murray": 2.5,
    "Fred VanVleet": 2.0,
    "Jalen Williams": 2.5,
    "Alperen Sengun": 2.5,
    "Myles Turner": 2.0,
    "Pascal Siakam": 2.5,
    "OG Anunoby": 2.0,
}


def get_star_player_impact(player_name: str) -> float:
    """
    Get estimated impact for a star player from hardcoded list.

    Args:
        player_name: Player's full name

    Returns:
        Estimated impact in points (0 if unknown)
    """
    return STAR_PLAYER_IMPACTS.get(player_name, 0.0)


def calculate_injury_adjustment(injured_players: List[str]) -> float:
    """
    Calculate spread adjustment based on injured players.

    Args:
        injured_players: List of injured player names

    Returns:
        Spread adjustment (negative = team weaker)
    """
    total = 0.0
    for player in injured_players:
        impact = get_star_player_impact(player)
        total -= impact

    return round(total, 1)


if __name__ == "__main__":
    print("Player Impact Fetcher")
    print("=" * 50)

    fetcher = PlayerImpactFetcher()

    # Try to fetch data
    if HAS_NBA_API:
        print("\nFetching from nba_api...")
        fetcher.fetch_basic_stats_from_nba_api()

        # Show top players by impact
        print("\nTop 10 players by estimated impact:")
        all_players = list(fetcher.basic_stats_cache.items())
        all_players.sort(key=lambda x: x[1].get('estimated_impact', 0), reverse=True)

        for name, stats in all_players[:10]:
            print(f"  {name}: {stats.get('estimated_impact', 0):.1f} (PPG: {stats.get('points', 0):.1f})")

    # Test injury adjustment
    print("\nTest injury adjustments:")
    print(f"  Lakers without LeBron + AD: {calculate_injury_adjustment(['LeBron James', 'Anthony Davis']):.1f} pts")
    print(f"  Celtics without Tatum: {calculate_injury_adjustment(['Jayson Tatum']):.1f} pts")
    print(f"  Thunder without SGA: {calculate_injury_adjustment(['Shai Gilgeous-Alexander']):.1f} pts")
