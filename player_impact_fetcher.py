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
from bs4 import BeautifulSoup
import re

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

    Priority order:
    1. DARKO DPM (Daily Plus-Minus) from APAnalytics
    2. ESPN EPM (Estimated Plus-Minus)
    3. FiveThirtyEight RAPTOR
    4. Basic plus/minus from nba_api (fallback)

    All metrics standardized to -10 to +10 scale where:
    - +10 = MVP-level impact (top 1%)
    - +5 = All-Star level (top 10%)
    - 0 = Average starter
    - -5 = Below replacement level
    """

    def __init__(self):
        self.darko_cache: Dict[str, Dict] = {}
        self.epm_cache: Dict[str, Dict] = {}
        self.raptor_cache: Dict[str, Dict] = {}
        self.basic_stats_cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached player data from all sources."""
        cache_files = {
            'darko': CACHE_DIR / "darko_cache.json",
            'epm': CACHE_DIR / "epm_cache.json",
            'raptor': CACHE_DIR / "raptor_cache.json",
        }

        for source, cache_file in cache_files.items():
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        # Check if cache is still valid
                        cache_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
                        if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                            if source == 'darko':
                                self.darko_cache = data.get('players', {})
                                print(f"Loaded {len(self.darko_cache)} players from DARKO cache")
                            elif source == 'epm':
                                self.epm_cache = data.get('players', {})
                                print(f"Loaded {len(self.epm_cache)} players from EPM cache")
                            elif source == 'raptor':
                                self.raptor_cache = data.get('players', {})
                                print(f"Loaded {len(self.raptor_cache)} players from RAPTOR cache")
                except Exception as e:
                    print(f"Error loading {source} cache: {e}")

    def _save_cache(self, source: str = 'all'):
        """Save player data to cache."""
        cache_data = {
            'darko': (CACHE_DIR / "darko_cache.json", self.darko_cache),
            'epm': (CACHE_DIR / "epm_cache.json", self.epm_cache),
            'raptor': (CACHE_DIR / "raptor_cache.json", self.raptor_cache),
        }

        sources_to_save = [source] if source != 'all' else cache_data.keys()

        for src in sources_to_save:
            if src in cache_data:
                cache_file, cache_dict = cache_data[src]
                try:
                    with open(cache_file, 'w') as f:
                        json.dump({
                            'timestamp': datetime.now().isoformat(),
                            'players': cache_dict
                        }, f, indent=2)
                except Exception as e:
                    print(f"Error saving {src} cache: {e}")

    def _standardize_metric(self, value: float, metric_type: str) -> float:
        """
        Standardize different metrics to -10 to +10 scale.

        Args:
            value: Raw metric value
            metric_type: 'darko', 'epm', 'raptor', or 'plus_minus'

        Returns:
            Standardized value on -10 to +10 scale
        """
        # Conversion factors based on typical ranges
        # DARKO DPM: typically -8 to +8 (MVP level)
        # EPM: typically -7 to +7
        # RAPTOR: typically -8 to +8
        # Plus/Minus per 36: typically -10 to +10

        if metric_type == 'darko':
            # DARKO already close to our scale, cap at ±10
            return max(-10, min(10, value * 1.25))
        elif metric_type == 'epm':
            # EPM scale: expand slightly
            return max(-10, min(10, value * 1.4))
        elif metric_type == 'raptor':
            # RAPTOR scale: expand slightly
            return max(-10, min(10, value * 1.25))
        elif metric_type == 'plus_minus':
            # Plus/minus per 36 already roughly on scale
            return max(-10, min(10, value))
        else:
            return 0.0

    def fetch_darko_dpm(self, season: str = "2024-25") -> Dict[str, Dict]:
        """
        Fetch DARKO DPM (Daily Plus-Minus) from APAnalytics Shiny app.

        DARKO is one of the most advanced publicly available impact metrics,
        combining box score stats with play-by-play data.

        Args:
            season: Season string (e.g., "2024-25")

        Returns:
            Dictionary mapping player name to impact metrics
        """
        print(f"Fetching DARKO DPM for {season}...")

        # APAnalytics DARKO endpoint
        url = "https://apanalytics.shinyapps.io/DARKO/"

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            # Add delay to be respectful
            time.sleep(2)

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"Could not fetch DARKO data: HTTP {response.status_code}")
                return {}

            soup = BeautifulSoup(response.content, 'html.parser')

            # DARKO uses a data table - look for it
            # The actual scraping depends on the page structure
            # This is a best-effort implementation
            players_dict = {}

            # Try to find the data table
            tables = soup.find_all('table')

            if not tables:
                print("DARKO: No tables found on page (may require JavaScript)")
                return {}

            # Parse the main data table
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) < 2:
                    continue

                # Get headers
                headers_row = rows[0]
                headers = [th.get_text(strip=True).lower() for th in headers_row.find_all(['th', 'td'])]

                # Find relevant column indices
                name_idx = next((i for i, h in enumerate(headers) if 'player' in h or 'name' in h), None)
                dpm_idx = next((i for i, h in enumerate(headers) if 'dpm' in h or 'darko' in h), None)
                team_idx = next((i for i, h in enumerate(headers) if 'team' in h), None)

                if name_idx is None or dpm_idx is None:
                    continue

                # Parse data rows
                for row in rows[1:]:
                    cols = row.find_all('td')
                    if len(cols) <= max(name_idx, dpm_idx):
                        continue

                    try:
                        player_name = cols[name_idx].get_text(strip=True)
                        dpm_raw = cols[dpm_idx].get_text(strip=True)
                        team = cols[team_idx].get_text(strip=True) if team_idx and len(cols) > team_idx else 'UNK'

                        # Parse DPM value
                        dpm_value = float(dpm_raw)

                        # Standardize to -10 to +10 scale
                        standardized_impact = self._standardize_metric(dpm_value, 'darko')

                        players_dict[player_name] = {
                            'source': 'darko',
                            'raw_dpm': dpm_value,
                            'impact_metric': standardized_impact,
                            'team': team,
                            'season': season
                        }

                    except (ValueError, IndexError) as e:
                        continue

            if players_dict:
                print(f"Fetched DARKO data for {len(players_dict)} players")
                self.darko_cache = players_dict
                self._save_cache('darko')
                return players_dict
            else:
                print("DARKO: Could not parse data from page")
                return {}

        except Exception as e:
            print(f"Error fetching DARKO: {e}")
            return {}

    def fetch_espn_epm(self, season: int = 2025) -> Dict[str, Dict]:
        """
        Fetch ESPN EPM (Estimated Plus-Minus) data.

        EPM is ESPN's proprietary player impact metric that estimates
        a player's contribution per 100 possessions.

        Args:
            season: Season year (e.g., 2025 for 2024-25)

        Returns:
            Dictionary mapping player name to impact metrics
        """
        print(f"Fetching ESPN EPM for {season}...")

        # ESPN EPM stats page
        url = f"https://www.espn.com/nba/stats/player/_/season/{season}/seasontype/2"

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            time.sleep(2)

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"Could not fetch ESPN EPM: HTTP {response.status_code}")
                return {}

            soup = BeautifulSoup(response.content, 'html.parser')
            players_dict = {}

            # ESPN uses complex table structures
            # Look for player stats tables
            tables = soup.find_all('table', class_=re.compile('Table'))

            if not tables:
                print("ESPN: No stats tables found (may require JavaScript)")
                return {}

            # Parse tables for EPM-related data
            # Note: ESPN may not directly expose EPM on the main stats page
            # You may need to scrape from their RPM/EPM specific pages

            # Placeholder: In practice, ESPN's EPM data may require
            # API access or more complex scraping
            print("ESPN EPM: Direct access not available via web scraping")
            return {}

        except Exception as e:
            print(f"Error fetching ESPN EPM: {e}")
            return {}

    def fetch_fivethirtyeight_raptor(self, season: str = "2024-25") -> Dict[str, Dict]:
        """
        Fetch FiveThirtyEight RAPTOR ratings.

        RAPTOR (Robust Algorithm using Player Tracking and On/off Ratings)
        is FiveThirtyEight's player impact metric.

        Args:
            season: Season string (e.g., "2024-25")

        Returns:
            Dictionary mapping player name to impact metrics
        """
        print(f"Fetching FiveThirtyEight RAPTOR for {season}...")

        # FiveThirtyEight provides data on GitHub
        # For current season, they may have a live endpoint
        # Historical data: https://github.com/fivethirtyeight/data/tree/master/nba-raptor

        # Try GitHub raw file for latest season
        season_year = season.split('-')[0]
        url = f"https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/modern_RAPTOR_by_player.csv"

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            time.sleep(2)

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"Could not fetch RAPTOR data: HTTP {response.status_code}")
                return {}

            # Parse CSV data using pandas-style approach for robustness
            import csv
            from io import StringIO

            players_dict = {}
            csv_reader = csv.DictReader(StringIO(response.text))

            # Check if CSV has required columns
            fieldnames = csv_reader.fieldnames or []
            if not fieldnames:
                print("RAPTOR: No CSV headers found")
                return {}

            # Find column names (case-insensitive)
            fieldnames_lower = {f.lower(): f for f in fieldnames}

            name_col = next((fieldnames_lower[k] for k in fieldnames_lower if 'player' in k or 'name' in k), None)
            raptor_col = next((fieldnames_lower[k] for k in fieldnames_lower if 'raptor' in k), None)
            team_col = next((fieldnames_lower[k] for k in fieldnames_lower if 'team' in k), None)
            season_col = next((fieldnames_lower[k] for k in fieldnames_lower if 'season' in k), None)

            if not name_col or not raptor_col:
                print(f"RAPTOR: Required columns not found. Available: {fieldnames}")
                return {}

            # Parse data rows
            for row in csv_reader:
                try:
                    player_name = row[name_col].strip()
                    if not player_name:
                        continue

                    # Get RAPTOR value (may be in different columns)
                    raptor_raw = None
                    for possible_raptor_col in [raptor_col, 'raptor_total', 'raptor_offense', 'war_total']:
                        if possible_raptor_col in row:
                            try:
                                raptor_raw = float(row[possible_raptor_col])
                                break
                            except (ValueError, TypeError):
                                continue

                    if raptor_raw is None:
                        continue

                    team = row.get(team_col, 'UNK').strip() if team_col else 'UNK'
                    player_season = row.get(season_col, '').strip() if season_col else ''

                    # Filter to current season if specified
                    if season_col and season_year and season_year not in player_season:
                        continue

                    # Standardize to -10 to +10 scale
                    standardized_impact = self._standardize_metric(raptor_raw, 'raptor')

                    players_dict[player_name] = {
                        'source': 'raptor',
                        'raw_raptor': raptor_raw,
                        'impact_metric': standardized_impact,
                        'team': team,
                        'season': player_season
                    }

                except (ValueError, KeyError, TypeError) as e:
                    continue

            if players_dict:
                print(f"Fetched RAPTOR data for {len(players_dict)} players")
                self.raptor_cache = players_dict
                self._save_cache('raptor')
                return players_dict
            else:
                print("RAPTOR: Could not parse player data")
                return {}

        except Exception as e:
            print(f"Error fetching RAPTOR: {e}")
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
                # Calculate per-36 plus/minus as impact estimate
                per_36_plus_minus = row['PLUS_MINUS'] / max(row['MIN'], 1) * 36
                # Standardize to our scale
                impact_metric = self._standardize_metric(per_36_plus_minus, 'plus_minus')

                players_dict[player_name] = {
                    'source': 'nba_api',
                    'player_id': row['PLAYER_ID'],
                    'team': row['TEAM_ABBREVIATION'],
                    'games': row['GP'],
                    'minutes': row['MIN'],
                    'points': row['PTS'],
                    'rebounds': row['REB'],
                    'assists': row['AST'],
                    'plus_minus': row['PLUS_MINUS'],
                    'net_rating': row.get('NET_RATING', 0),
                    'estimated_impact': per_36_plus_minus,  # Raw per-36 for backward compatibility
                    'impact_metric': impact_metric  # Standardized metric
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

        Priority order:
        1. DARKO DPM (most advanced)
        2. ESPN EPM
        3. FiveThirtyEight RAPTOR
        4. nba_api basic stats (fallback)

        Args:
            player_name: Player's full name

        Returns:
            Dictionary with player impact metrics (with 'impact_metric' key) or None
        """
        # Priority 1: DARKO
        if player_name in self.darko_cache:
            return self.darko_cache[player_name]

        # Priority 2: ESPN EPM
        if player_name in self.epm_cache:
            return self.epm_cache[player_name]

        # Priority 3: FiveThirtyEight RAPTOR
        if player_name in self.raptor_cache:
            return self.raptor_cache[player_name]

        # Priority 4: Basic stats
        if player_name in self.basic_stats_cache:
            return self.basic_stats_cache[player_name]

        # Try to fetch if caches are empty
        if not self.darko_cache and not self.raptor_cache and not self.basic_stats_cache:
            self.refresh_data()
            # Try again after refresh
            return self.get_player_impact(player_name)

        return None

    def get_player_impact_metric(self, player_name: str) -> float:
        """
        Get standardized impact metric for a player (-10 to +10 scale).

        Args:
            player_name: Player's full name

        Returns:
            Impact metric value (0.0 if not found)
        """
        impact_data = self.get_player_impact(player_name)
        if impact_data:
            return impact_data.get('impact_metric', 0.0)
        return 0.0

    def get_team_impact_when_player_on_court(
        self,
        team_abbrev: str,
        player_name: str
    ) -> float:
        """
        Estimate team's net rating when specific player is on court.

        Args:
            team_abbrev: Team abbreviation
            player_name: Player name

        Returns:
            Estimated team net rating with player on court
        """
        player_impact = self.get_player_impact_metric(player_name)

        # Team base rating (assuming average team is 0)
        # Add player's impact
        return player_impact

    def get_opponent_defensive_impact_vs_position(
        self,
        opponent_team: str,
        position: str
    ) -> float:
        """
        Calculate opponent's defensive impact against a specific position.

        Args:
            opponent_team: Opponent team abbreviation
            position: Position ('G', 'F', 'C')

        Returns:
            Defensive impact (negative = strong defense, positive = weak defense)
        """
        # Get all players on opponent team
        team_defenders = []

        for cache in [self.darko_cache, self.epm_cache, self.raptor_cache, self.basic_stats_cache]:
            for name, stats in cache.items():
                if stats.get('team') == opponent_team:
                    impact = stats.get('impact_metric', 0.0)
                    team_defenders.append((name, impact))

        if not team_defenders:
            return 0.0

        # Sort by impact (best players first)
        team_defenders.sort(key=lambda x: x[1], reverse=True)

        # Top 3 defenders for position
        # In a full implementation, would filter by position matchup
        # For now, use top defenders overall
        top_defenders = team_defenders[:3]
        avg_defensive_impact = sum(impact for _, impact in top_defenders) / len(top_defenders)

        # Invert sign (positive defender impact = harder to score against)
        return -avg_defensive_impact * 0.3  # Scale down for prop adjustments

    def calculate_team_rating_adjustment(
        self,
        team_abbrev: str,
        injured_players: List[str] = None,
        resting_players: List[str] = None
    ) -> float:
        """
        Calculate team rating adjustment based on player availability.

        Uses advanced player impact metrics (DARKO/EPM/RAPTOR) to estimate
        how much a team's rating should be adjusted when key players are out.

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
            impact_data = self.get_player_impact(player_name)
            if impact_data:
                # Use standardized impact metric (-10 to +10 scale)
                player_impact = impact_data.get('impact_metric', 0)

                # Get minutes weight if available
                minutes = impact_data.get('minutes', 30)  # Default to 30 MPG
                minutes_weight = min(minutes / 36, 1.0)  # Cap at 1.0 for 36+ MPG players

                # Adjustment is negative (team gets worse when player is out)
                # Impact metric already on -10 to +10 scale
                # Scale by 0.5 for spread adjustment
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

        # Search in all caches (priority order)
        all_caches = [
            ('darko', self.darko_cache),
            ('epm', self.epm_cache),
            ('raptor', self.raptor_cache),
            ('basic', self.basic_stats_cache)
        ]

        seen_players = set()

        for source, cache in all_caches:
            for name, stats in cache.items():
                if stats.get('team') == team_abbrev and name not in seen_players:
                    seen_players.add(name)
                    team_players.append({
                        'name': name,
                        'source': source,
                        'minutes': stats.get('minutes', 0),
                        'impact': stats.get('impact_metric', stats.get('estimated_impact', 0)),
                        'points': stats.get('points', 0),
                        'plus_minus': stats.get('plus_minus', 0),
                    })

        # Sort by impact
        team_players.sort(key=lambda x: x['impact'], reverse=True)

        return team_players

    def refresh_data(self, season: str = "2024-25"):
        """
        Refresh all player data from sources.

        Tries in priority order:
        1. DARKO DPM
        2. FiveThirtyEight RAPTOR (more reliable than ESPN)
        3. nba_api basic stats (fallback)

        Args:
            season: Season string (e.g., "2024-25")
        """
        print("Refreshing player impact data...")

        # Try DARKO first (best metric)
        darko_data = self.fetch_darko_dpm(season)
        if darko_data:
            print(f"✓ Successfully loaded DARKO data for {len(darko_data)} players")

        # Try RAPTOR (reliable GitHub data)
        raptor_data = self.fetch_fivethirtyeight_raptor(season)
        if raptor_data:
            print(f"✓ Successfully loaded RAPTOR data for {len(raptor_data)} players")

        # Try ESPN EPM (often requires JS, may fail)
        season_year = int(season.split('-')[0])
        epm_data = self.fetch_espn_epm(season_year)
        if epm_data:
            print(f"✓ Successfully loaded ESPN EPM data for {len(epm_data)} players")

        # Fallback to nba_api
        if not darko_data and not raptor_data and not epm_data:
            if HAS_NBA_API:
                print("Falling back to nba_api basic stats...")
                self.fetch_basic_stats_from_nba_api(season)
            else:
                print("WARNING: No player impact data sources available!")

        print(f"Data refresh complete. Total unique players: {len(set(list(self.darko_cache.keys()) + list(self.raptor_cache.keys()) + list(self.epm_cache.keys()) + list(self.basic_stats_cache.keys())))}")


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


# Player defensive and offensive roles for prop adjustments
# Used to adjust props when opponent defensive specialists are out
PLAYER_DEFENSIVE_ROLES = {
    # Elite Perimeter Defenders - when out, opposing guards score more
    "Jrue Holiday": {"team": "BOS", "position": "G", "defensive_role": "perimeter", "offensive_role": "secondary", "impact_score": 2.5},
    "Marcus Smart": {"team": "MEM", "position": "G", "defensive_role": "perimeter", "offensive_role": "secondary", "impact_score": 1.5},
    "Derrick White": {"team": "BOS", "position": "G", "defensive_role": "perimeter", "offensive_role": "secondary", "impact_score": 2.0},
    "Alex Caruso": {"team": "OKC", "position": "G", "defensive_role": "perimeter", "offensive_role": "role_player", "impact_score": 1.5},
    "Matisse Thybulle": {"team": "POR", "position": "G", "defensive_role": "perimeter", "offensive_role": "role_player", "impact_score": 1.0},
    "Lu Dort": {"team": "OKC", "position": "G", "defensive_role": "perimeter", "offensive_role": "role_player", "impact_score": 1.5},
    "Herb Jones": {"team": "NOP", "position": "F", "defensive_role": "perimeter", "offensive_role": "role_player", "impact_score": 1.5},

    # Elite Wing Stoppers - when out, opposing forwards score more
    "OG Anunoby": {"team": "NYK", "position": "F", "defensive_role": "wing_stopper", "offensive_role": "secondary", "impact_score": 2.0},
    "Mikal Bridges": {"team": "NYK", "position": "F", "defensive_role": "wing_stopper", "offensive_role": "secondary", "impact_score": 2.0},
    "Aaron Gordon": {"team": "DEN", "position": "F", "defensive_role": "wing_stopper", "offensive_role": "secondary", "impact_score": 2.0},
    "Dillon Brooks": {"team": "HOU", "position": "F", "defensive_role": "wing_stopper", "offensive_role": "role_player", "impact_score": 1.5},
    "Andrew Wiggins": {"team": "GSW", "position": "F", "defensive_role": "wing_stopper", "offensive_role": "secondary", "impact_score": 2.0},

    # Elite Rim Protectors - when out, interior scoring increases
    "Rudy Gobert": {"team": "MIN", "position": "C", "defensive_role": "rim_protector", "offensive_role": "role_player", "impact_score": 3.0},
    "Anthony Davis": {"team": "LAL", "position": "C", "defensive_role": "rim_protector", "offensive_role": "primary_scorer", "impact_score": 4.0},
    "Evan Mobley": {"team": "CLE", "position": "C", "defensive_role": "rim_protector", "offensive_role": "secondary", "impact_score": 2.5},
    "Jaren Jackson Jr.": {"team": "MEM", "position": "C", "defensive_role": "rim_protector", "offensive_role": "secondary", "impact_score": 2.5},
    "Victor Wembanyama": {"team": "SAS", "position": "C", "defensive_role": "rim_protector", "offensive_role": "primary_scorer", "impact_score": 3.5},
    "Chet Holmgren": {"team": "OKC", "position": "C", "defensive_role": "rim_protector", "offensive_role": "secondary", "impact_score": 3.0},
    "Myles Turner": {"team": "IND", "position": "C", "defensive_role": "rim_protector", "offensive_role": "secondary", "impact_score": 2.0},
    "Brook Lopez": {"team": "MIL", "position": "C", "defensive_role": "rim_protector", "offensive_role": "role_player", "impact_score": 2.0},
    "Bam Adebayo": {"team": "MIA", "position": "C", "defensive_role": "versatile", "offensive_role": "secondary", "impact_score": 3.0},
    "Giannis Antetokounmpo": {"team": "MIL", "position": "F", "defensive_role": "versatile", "offensive_role": "primary_scorer", "impact_score": 4.5},
    "Draymond Green": {"team": "GSW", "position": "F", "defensive_role": "versatile", "offensive_role": "playmaker", "impact_score": 2.5},
    "Joel Embiid": {"team": "PHI", "position": "C", "defensive_role": "rim_protector", "offensive_role": "primary_scorer", "impact_score": 4.5},

    # Primary Playmakers - when out, teammate assists may increase
    "Nikola Jokic": {"team": "DEN", "position": "C", "defensive_role": None, "offensive_role": "playmaker", "impact_score": 5.0},
    "Luka Doncic": {"team": "DAL", "position": "G", "defensive_role": None, "offensive_role": "playmaker", "impact_score": 4.5},
    "Trae Young": {"team": "ATL", "position": "G", "defensive_role": None, "offensive_role": "playmaker", "impact_score": 3.5},
    "Tyrese Haliburton": {"team": "IND", "position": "G", "defensive_role": None, "offensive_role": "playmaker", "impact_score": 3.0},
    "Darius Garland": {"team": "CLE", "position": "G", "defensive_role": None, "offensive_role": "playmaker", "impact_score": 2.5},
    "LaMelo Ball": {"team": "CHA", "position": "G", "defensive_role": None, "offensive_role": "playmaker", "impact_score": 3.0},
    "James Harden": {"team": "LAC", "position": "G", "defensive_role": None, "offensive_role": "playmaker", "impact_score": 3.0},

    # Primary Scorers - when out, teammate scoring may increase
    "Stephen Curry": {"team": "GSW", "position": "G", "defensive_role": None, "offensive_role": "primary_scorer", "impact_score": 4.0},
    "Jayson Tatum": {"team": "BOS", "position": "F", "defensive_role": None, "offensive_role": "primary_scorer", "impact_score": 4.0},
    "Kevin Durant": {"team": "PHX", "position": "F", "defensive_role": None, "offensive_role": "primary_scorer", "impact_score": 4.0},
    "Shai Gilgeous-Alexander": {"team": "OKC", "position": "G", "defensive_role": "perimeter", "offensive_role": "primary_scorer", "impact_score": 4.0},
    "Donovan Mitchell": {"team": "CLE", "position": "G", "defensive_role": None, "offensive_role": "primary_scorer", "impact_score": 3.5},
    "Devin Booker": {"team": "PHX", "position": "G", "defensive_role": None, "offensive_role": "primary_scorer", "impact_score": 3.5},
    "Anthony Edwards": {"team": "MIN", "position": "G", "defensive_role": None, "offensive_role": "primary_scorer", "impact_score": 3.5},
    "Damian Lillard": {"team": "MIL", "position": "G", "defensive_role": None, "offensive_role": "primary_scorer", "impact_score": 3.5},
    "Jimmy Butler": {"team": "MIA", "position": "F", "defensive_role": "wing_stopper", "offensive_role": "primary_scorer", "impact_score": 3.5},
    "LeBron James": {"team": "LAL", "position": "F", "defensive_role": None, "offensive_role": "primary_scorer", "impact_score": 4.0},
}

# Injury boost constants for prop adjustments
PROP_INJURY_BOOST = {
    "perimeter_defender_out_guard_points": 0.08,   # +8% for guards
    "perimeter_defender_out_guard_threes": 0.10,   # +10% for 3PM
    "rim_protector_out_center_points": 0.10,       # +10% for centers
    "rim_protector_out_guard_points": 0.05,        # +5% guards attack paint
    "rim_protector_out_rebounds": 0.12,            # +12% rebounds
    "wing_stopper_out_forward_points": 0.08,       # +8% for forwards
    "primary_scorer_out_secondary": 0.10,          # +10% for secondary scorer
    "playmaker_out_secondary_assists": 0.15,       # +15% for secondary handler
}


def get_player_role(player_name: str) -> Optional[Dict]:
    """Get complete role information for a player."""
    return PLAYER_DEFENSIVE_ROLES.get(player_name)


def calculate_prop_injury_boost(
    player_position: str,
    prop_type: str,
    opponent_injured: List[str],
    teammate_injured: List[str] = None
) -> Dict:
    """
    Calculate prop prediction boost based on injuries.

    Args:
        player_position: G (guard), F (forward), or C (center)
        prop_type: points, rebounds, assists, threes, pra
        opponent_injured: List of injured opponent player names
        teammate_injured: List of injured teammate names

    Returns:
        Dictionary with boost_factor and reasons
    """
    boost_factor = 1.0
    reasons = []

    # Check opponent injuries
    for injured_player in opponent_injured:
        role_info = PLAYER_DEFENSIVE_ROLES.get(injured_player, {})
        defensive_role = role_info.get("defensive_role")

        if not defensive_role:
            continue

        # Perimeter defender out - guards score more
        if defensive_role == "perimeter" and player_position == "G":
            if prop_type in ["points", "pra"]:
                boost = PROP_INJURY_BOOST["perimeter_defender_out_guard_points"]
                boost_factor *= (1 + boost)
                reasons.append(f"{injured_player} (perimeter) out")
            elif prop_type == "threes":
                boost = PROP_INJURY_BOOST["perimeter_defender_out_guard_threes"]
                boost_factor *= (1 + boost)
                reasons.append(f"{injured_player} (perimeter) out")

        # Wing stopper out - forwards score more
        elif defensive_role == "wing_stopper" and player_position == "F":
            if prop_type in ["points", "pra"]:
                boost = PROP_INJURY_BOOST["wing_stopper_out_forward_points"]
                boost_factor *= (1 + boost)
                reasons.append(f"{injured_player} (wing stopper) out")

        # Rim protector out
        elif defensive_role in ["rim_protector", "versatile"]:
            if prop_type in ["points", "pra"] and player_position == "C":
                boost = PROP_INJURY_BOOST["rim_protector_out_center_points"]
                boost_factor *= (1 + boost)
                reasons.append(f"{injured_player} (rim protector) out")
            elif prop_type in ["rebounds", "pra"]:
                boost = PROP_INJURY_BOOST["rim_protector_out_rebounds"]
                boost_factor *= (1 + boost)
                reasons.append(f"{injured_player} (rim protector) out")

    # Check teammate injuries
    if teammate_injured:
        for injured_teammate in teammate_injured:
            role_info = PLAYER_DEFENSIVE_ROLES.get(injured_teammate, {})
            offensive_role = role_info.get("offensive_role")

            if offensive_role == "primary_scorer" and prop_type in ["points", "pra"]:
                boost = PROP_INJURY_BOOST["primary_scorer_out_secondary"]
                boost_factor *= (1 + boost)
                reasons.append(f"Teammate {injured_teammate} out (+usage)")

            elif offensive_role == "playmaker" and prop_type in ["assists", "pra"]:
                boost = PROP_INJURY_BOOST["playmaker_out_secondary_assists"]
                boost_factor *= (1 + boost)
                reasons.append(f"Playmaker {injured_teammate} out (+assists)")

    # Cap at +/- 15%
    boost_factor = max(0.85, min(1.15, boost_factor))

    return {
        "boost_factor": boost_factor,
        "reasons": reasons,
        "adjustment_pct": (boost_factor - 1) * 100
    }


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
