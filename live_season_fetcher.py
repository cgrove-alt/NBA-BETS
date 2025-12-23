"""
Live Season Fetcher

Fetches NBA games from missing seasons (2023-24, 2024-25, 2025-26) using APIs.
Uses Balldontlie API (faster, 100ms delay) with nba_api fallback.

Usage:
    python3 live_season_fetcher.py                    # Fetch all missing seasons
    python3 live_season_fetcher.py --season 2025     # Fetch specific season
    python3 live_season_fetcher.py --test            # Test with small sample
"""

import requests
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# API Configuration
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"
FAST_API_DELAY = 0.1  # 100ms between requests

# Try to import nba_api for fallback
try:
    from nba_api.stats.endpoints import leaguegamefinder
    from nba_api.stats.static import teams as nba_teams
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False

# Team abbreviation mappings
TEAM_ABBREV_MAP = {
    "NJN": "BKN",
    "NOH": "NOP",
    "NOK": "NOP",
    "SEA": "OKC",
    "VAN": "MEM",
    "CHH": "CHA",
    "PHO": "PHX",
}

# NBA Team IDs for nba_api
NBA_TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}


def fetch_season_from_nba_api(season: int) -> pd.DataFrame:
    """
    Fetch a full season from nba_api.

    Args:
        season: Season year (e.g., 2023 for 2023-24 season)

    Returns:
        DataFrame with game data (normalized columns)
    """
    if not HAS_NBA_API:
        print("nba_api not available")
        return pd.DataFrame()

    season_str = f"{season}-{str(season+1)[-2:]}"  # e.g., "2023-24"
    print(f"\nFetching {season_str} season from NBA API...")
    print("  (Using 600ms delay - this will take a few seconds)")

    try:
        time.sleep(0.6)  # Initial delay
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str,
            season_type_nullable="Regular Season",
            league_id_nullable="00"
        )
        games_df = finder.get_data_frames()[0]

        if games_df.empty:
            print(f"  No games found for {season_str}")
            return pd.DataFrame()

        # Add SEASON_YEAR column to match historical CSV format
        games_df["SEASON_YEAR"] = season_str

        print(f"  Found {len(games_df)} game records")
        return games_df

    except Exception as e:
        print(f"  Error fetching from NBA API: {e}")
        return pd.DataFrame()


def fetch_season_from_balldontlie(
    season: int,
    api_key: Optional[str] = None,
    max_pages: int = 50,
) -> pd.DataFrame:
    """
    Fetch a full season from Balldontlie API.

    Args:
        season: Season year (e.g., 2023 for 2023-24 season)
        api_key: Optional API key for higher rate limits
        max_pages: Maximum pages to fetch

    Returns:
        DataFrame with game data
    """
    season_str = f"{season}-{str(season+1)[-2:]}"
    print(f"\nFetching {season_str} season from Balldontlie API...")
    print(f"  (Using 100ms delay - fast!)")

    all_games = []
    url = f"{BALLDONTLIE_BASE}/games"

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = api_key

    params = {
        "per_page": 100,
        "seasons[]": season,
    }

    for page in range(1, max_pages + 1):
        params["cursor"] = page

        try:
            time.sleep(FAST_API_DELAY)
            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 401:
                print("  API key required - falling back to nba_api")
                return fetch_season_from_nba_api(season)

            if response.status_code == 429:
                print("  Rate limited - waiting 10 seconds...")
                time.sleep(10)
                continue

            if response.status_code != 200:
                print(f"  API error: {response.status_code}")
                break

            data = response.json()
            games = data.get("data", [])

            if not games:
                break

            all_games.extend(games)

            if page % 10 == 0:
                print(f"  Page {page}: {len(all_games)} total games")

            # Check if more pages
            meta = data.get("meta", {})
            if not meta.get("next_cursor"):
                break

        except Exception as e:
            print(f"  Error on page {page}: {e}")
            continue

    print(f"  Total games fetched: {len(all_games)}")

    if not all_games:
        print("  Balldontlie returned no data - trying nba_api fallback")
        return fetch_season_from_nba_api(season)

    # Convert to DataFrame
    rows = []
    for game in all_games:
        # Skip if game not completed
        if game.get("status") != "Final":
            continue

        home_team = game.get("home_team", {})
        away_team = game.get("visitor_team", {})

        home_abbrev = home_team.get("abbreviation", "")
        away_abbrev = away_team.get("abbreviation", "")
        home_score = game.get("home_team_score", 0)
        away_score = game.get("visitor_team_score", 0)

        if not all([home_abbrev, away_abbrev, home_score, away_score]):
            continue

        # Normalize team abbreviations
        home_abbrev = TEAM_ABBREV_MAP.get(home_abbrev, home_abbrev)
        away_abbrev = TEAM_ABBREV_MAP.get(away_abbrev, away_abbrev)

        game_date = game.get("date", "")[:10]  # YYYY-MM-DD

        # Create records for both teams (matching CSV format)
        # Home team record
        rows.append({
            "SEASON_YEAR": season_str,
            "TEAM_ABBREVIATION": home_abbrev,
            "GAME_ID": game.get("id", ""),
            "GAME_DATE": game_date,
            "MATCHUP": f"{home_abbrev} vs. {away_abbrev}",
            "WL": "W" if home_score > away_score else "L",
            "PTS": home_score,
            "PLUS_MINUS": home_score - away_score,
        })

        # Away team record
        rows.append({
            "SEASON_YEAR": season_str,
            "TEAM_ABBREVIATION": away_abbrev,
            "GAME_ID": game.get("id", ""),
            "GAME_DATE": game_date,
            "MATCHUP": f"{away_abbrev} @ {home_abbrev}",
            "WL": "W" if away_score > home_score else "L",
            "PTS": away_score,
            "PLUS_MINUS": away_score - home_score,
        })

    df = pd.DataFrame(rows)
    print(f"  Converted to {len(df)} team-game records")
    return df


def fetch_all_missing_seasons(
    seasons: List[int] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch all missing seasons.

    Args:
        seasons: List of season years (default: [2023, 2024, 2025])
        api_key: Optional Balldontlie API key

    Returns:
        Combined DataFrame with all seasons
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]  # 2023-24, 2024-25, 2025-26

    print("=" * 60)
    print("FETCHING MISSING NBA SEASONS")
    print("=" * 60)
    print(f"Seasons to fetch: {[f'{s}-{str(s+1)[-2:]}' for s in seasons]}")

    all_data = []

    for season in seasons:
        df = fetch_season_from_balldontlie(season, api_key)

        if not df.empty:
            all_data.append(df)
            print(f"  {season}-{str(season+1)[-2:]}: {len(df)} records")

    if not all_data:
        print("\nNo data fetched!")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(combined)} team-game records")
    print(f"{'=' * 60}")

    return combined


def save_to_csv(df: pd.DataFrame, output_path: str = None) -> str:
    """
    Save fetched data to CSV file.

    Args:
        df: DataFrame to save
        output_path: Optional output path

    Returns:
        Path to saved file
    """
    if output_path is None:
        data_dir = Path(__file__).parent / "data" / "live_seasons"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / f"live_seasons_{datetime.now().strftime('%Y%m%d')}.csv"

    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    return str(output_path)


def load_live_season_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load previously fetched live season data.

    Args:
        csv_path: Path to CSV file (optional, finds latest if not specified)

    Returns:
        DataFrame with live season data
    """
    if csv_path:
        return pd.read_csv(csv_path)

    # Find latest file
    data_dir = Path(__file__).parent / "data" / "live_seasons"
    if not data_dir.exists():
        return pd.DataFrame()

    csv_files = sorted(data_dir.glob("live_seasons_*.csv"), reverse=True)
    if not csv_files:
        return pd.DataFrame()

    latest = csv_files[0]
    print(f"Loading live data from: {latest}")
    return pd.read_csv(latest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NBA seasons from API")
    parser.add_argument("--season", type=int, help="Specific season year (e.g., 2025)")
    parser.add_argument("--seasons", type=str, help="Comma-separated seasons (e.g., 2023,2024,2025)")
    parser.add_argument("--api-key", type=str, help="Balldontlie API key")
    parser.add_argument("--test", action="store_true", help="Test mode (fetch less data)")
    parser.add_argument("--output", type=str, help="Output CSV path")

    args = parser.parse_args()

    # Determine seasons to fetch
    if args.season:
        seasons = [args.season]
    elif args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(",")]
    else:
        seasons = [2023, 2024, 2025]  # Default: all missing seasons

    print(f"\nFetching seasons: {seasons}")

    if args.test:
        print("TEST MODE: Fetching first 2 pages only")
        # Just test with one season
        df = fetch_season_from_balldontlie(seasons[0], args.api_key, max_pages=2)
    else:
        df = fetch_all_missing_seasons(seasons, args.api_key)

    if not df.empty:
        output_path = save_to_csv(df, args.output)

        # Show summary
        print("\nSeason Summary:")
        for season in df["SEASON_YEAR"].unique():
            count = len(df[df["SEASON_YEAR"] == season])
            print(f"  {season}: {count} records ({count // 2} games)")
