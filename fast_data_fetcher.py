"""
Fast NBA Data Fetcher

Uses multiple free APIs without aggressive rate limiting:
1. balldontlie.io (free tier with API key)
2. Direct requests to public endpoints

This is much faster than nba_api which has aggressive rate limiting.
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# API Configuration
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"
API_KEY = None  # Set this if you have a BALLDONTLIE API key

# Minimal rate limiting (much faster than nba_api)
FAST_API_DELAY = 0.1  # 100ms between requests


def set_api_key(key: str):
    """Set the BALLDONTLIE API key."""
    global API_KEY
    API_KEY = key


def _get_headers():
    """Get request headers."""
    headers = {"Accept": "application/json"}
    if API_KEY:
        headers["Authorization"] = API_KEY
    return headers


# NBA Team Data (static - no API call needed)
NBA_TEAMS = {
    "ATL": {"id": 1, "name": "Atlanta Hawks", "conference": "East"},
    "BOS": {"id": 2, "name": "Boston Celtics", "conference": "East"},
    "BKN": {"id": 3, "name": "Brooklyn Nets", "conference": "East"},
    "CHA": {"id": 4, "name": "Charlotte Hornets", "conference": "East"},
    "CHI": {"id": 5, "name": "Chicago Bulls", "conference": "East"},
    "CLE": {"id": 6, "name": "Cleveland Cavaliers", "conference": "East"},
    "DAL": {"id": 7, "name": "Dallas Mavericks", "conference": "West"},
    "DEN": {"id": 8, "name": "Denver Nuggets", "conference": "West"},
    "DET": {"id": 9, "name": "Detroit Pistons", "conference": "East"},
    "GSW": {"id": 10, "name": "Golden State Warriors", "conference": "West"},
    "HOU": {"id": 11, "name": "Houston Rockets", "conference": "West"},
    "IND": {"id": 12, "name": "Indiana Pacers", "conference": "East"},
    "LAC": {"id": 13, "name": "Los Angeles Clippers", "conference": "West"},
    "LAL": {"id": 14, "name": "Los Angeles Lakers", "conference": "West"},
    "MEM": {"id": 15, "name": "Memphis Grizzlies", "conference": "West"},
    "MIA": {"id": 16, "name": "Miami Heat", "conference": "East"},
    "MIL": {"id": 17, "name": "Milwaukee Bucks", "conference": "East"},
    "MIN": {"id": 18, "name": "Minnesota Timberwolves", "conference": "West"},
    "NOP": {"id": 19, "name": "New Orleans Pelicans", "conference": "West"},
    "NYK": {"id": 20, "name": "New York Knicks", "conference": "East"},
    "OKC": {"id": 21, "name": "Oklahoma City Thunder", "conference": "West"},
    "ORL": {"id": 22, "name": "Orlando Magic", "conference": "East"},
    "PHI": {"id": 23, "name": "Philadelphia 76ers", "conference": "East"},
    "PHX": {"id": 24, "name": "Phoenix Suns", "conference": "West"},
    "POR": {"id": 25, "name": "Portland Trail Blazers", "conference": "West"},
    "SAC": {"id": 26, "name": "Sacramento Kings", "conference": "West"},
    "SAS": {"id": 27, "name": "San Antonio Spurs", "conference": "West"},
    "TOR": {"id": 28, "name": "Toronto Raptors", "conference": "East"},
    "UTA": {"id": 29, "name": "Utah Jazz", "conference": "West"},
    "WAS": {"id": 30, "name": "Washington Wizards", "conference": "East"},
}

# Reverse lookup
TEAM_ID_TO_ABBREV = {v["id"]: k for k, v in NBA_TEAMS.items()}


def get_all_teams() -> List[Dict]:
    """Get all NBA teams (no API call needed)."""
    return [
        {"abbreviation": abbrev, "id": data["id"], "name": data["name"], "conference": data["conference"]}
        for abbrev, data in NBA_TEAMS.items()
    ]


def fetch_games_fast(
    start_date: str = None,
    end_date: str = None,
    season: int = 2024,
    per_page: int = 100,
    max_pages: int = 10,
) -> List[Dict]:
    """
    Fetch games quickly from balldontlie API.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        season: Season year (e.g., 2024 for 2024-25 season)
        per_page: Results per page (max 100)
        max_pages: Maximum pages to fetch

    Returns:
        List of game dictionaries
    """
    all_games = []

    # Build URL
    url = f"{BALLDONTLIE_BASE}/games"
    params = {
        "per_page": per_page,
    }

    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if season:
        params["seasons[]"] = season

    print(f"Fetching games from balldontlie API...")

    for page in range(1, max_pages + 1):
        params["cursor"] = page

        try:
            time.sleep(FAST_API_DELAY)
            response = requests.get(url, params=params, headers=_get_headers(), timeout=30)

            if response.status_code == 401:
                print("API key required for balldontlie. Using fallback method...")
                return fetch_games_fallback(start_date, end_date, season)

            if response.status_code != 200:
                print(f"API error: {response.status_code}")
                break

            data = response.json()
            games = data.get("data", [])

            if not games:
                break

            all_games.extend(games)
            print(f"  Page {page}: {len(games)} games (total: {len(all_games)})")

            # Check if more pages
            meta = data.get("meta", {})
            if not meta.get("next_cursor"):
                break

        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break

    return all_games


def fetch_games_fallback(
    start_date: str = None,
    end_date: str = None,
    season: int = 2024,
) -> List[Dict]:
    """
    Fallback: Use free-nba API (no auth required).
    """
    print("Using free-nba API fallback...")

    all_games = []
    base_url = "https://www.balldontlie.io/api/v1/games"  # Old free endpoint

    # Try the old free API
    params = {"per_page": 100}
    if season:
        params["seasons[]"] = season
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    for page in range(1, 20):
        params["page"] = page
        try:
            time.sleep(FAST_API_DELAY)
            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code != 200:
                print(f"Fallback API also requires auth. Status: {response.status_code}")
                break

            data = response.json()
            games = data.get("data", [])

            if not games:
                break

            all_games.extend(games)
            print(f"  Page {page}: {len(games)} games")

        except Exception as e:
            print(f"Error: {e}")
            break

    return all_games


def generate_synthetic_training_data(num_games: int = 1000) -> List[Dict]:
    """
    Generate high-quality synthetic training data based on realistic NBA statistics.

    This is useful when APIs are unavailable or rate-limited.
    Uses realistic distributions based on historical NBA data.

    CRITICAL: All values must be properly scaled (per-game averages, not totals)
    to match what feature_engineering.py produces.

    Realistic NBA ranges:
    - Points per game: 100-130
    - Win percentage: 0.20-0.80
    - Offensive/Defensive rating: 105-120
    - Net rating: -15 to +15
    - Pace: 95-105
    - Point differential: -15 to +15 typically
    """
    import random

    print(f"\nGenerating {num_games} synthetic games for training...")
    print("  Using realistic NBA per-game statistics...")

    teams = list(NBA_TEAMS.keys())
    games = []

    # 2025-26 season team strength ratings (realistic)
    team_ratings = {
        "BOS": 0.73, "OKC": 0.72, "CLE": 0.70, "DEN": 0.65, "MIN": 0.62,
        "NYK": 0.60, "MIL": 0.58, "PHX": 0.57, "DAL": 0.57, "MEM": 0.55,
        "LAC": 0.55, "SAC": 0.54, "MIA": 0.53, "LAL": 0.53, "IND": 0.52,
        "GSW": 0.50, "PHI": 0.48, "NOP": 0.48, "ORL": 0.50, "HOU": 0.47,
        "ATL": 0.45, "CHI": 0.43, "BKN": 0.40, "TOR": 0.38, "SAS": 0.35,
        "POR": 0.33, "UTA": 0.32, "WAS": 0.28, "CHA": 0.27, "DET": 0.25,
    }

    # Team-specific offensive/defensive tendencies
    team_offense_bias = {team: random.gauss(0, 0.03) for team in teams}
    team_defense_bias = {team: random.gauss(0, 0.03) for team in teams}
    team_pace_bias = {team: random.gauss(0, 2) for team in teams}

    base_date = datetime(2025, 10, 21)  # 2025-26 Season start

    for i in range(num_games):
        # Pick random teams
        home_team = random.choice(teams)
        away_team = random.choice([t for t in teams if t != home_team])

        # Get team ratings
        home_rating = team_ratings.get(home_team, 0.50)
        away_rating = team_ratings.get(away_team, 0.50)

        # Home court advantage (~3 points, ~60% home win rate)
        home_advantage_pts = 3.0

        # Calculate expected point differential based on ratings
        # Rating difference of 0.20 ~= 8 point advantage
        rating_diff = (home_rating - away_rating)
        expected_diff = rating_diff * 40 + home_advantage_pts  # Scaled appropriately

        # Clip expected diff to realistic range
        expected_diff = max(-25, min(25, expected_diff))

        # Add game-to-game variance (NBA games have ~12 point std dev)
        actual_diff = expected_diff + random.gauss(0, 12)
        actual_diff = max(-40, min(40, actual_diff))  # Clip extreme outliers

        # Generate final scores (NBA average ~112 ppg)
        avg_score = 112 + random.gauss(0, 5)
        home_score = int(avg_score + actual_diff / 2)
        away_score = int(avg_score - actual_diff / 2)

        # Ensure realistic score ranges (85-145)
        home_score = max(85, min(145, home_score))
        away_score = max(85, min(145, away_score))

        # No ties in NBA
        if home_score == away_score:
            home_score += random.choice([-1, 1])

        home_win = home_score > away_score
        point_diff = home_score - away_score

        # Generate realistic game date
        game_date = (base_date + timedelta(days=i // 12)).strftime("%Y-%m-%d")

        # =====================================================================
        # MONEYLINE FEATURES - All properly scaled
        # =====================================================================
        # Win percentages (0.0 to 1.0)
        home_season_win_pct = max(0.15, min(0.85, home_rating + random.gauss(0, 0.03)))
        away_season_win_pct = max(0.15, min(0.85, away_rating + random.gauss(0, 0.03)))
        home_recent_win_pct = max(0.0, min(1.0, home_rating + random.gauss(0, 0.1)))
        away_recent_win_pct = max(0.0, min(1.0, away_rating + random.gauss(0, 0.1)))
        home_location_win_pct = max(0.15, min(0.90, home_rating + 0.05 + random.gauss(0, 0.05)))
        away_location_win_pct = max(0.10, min(0.85, away_rating - 0.05 + random.gauss(0, 0.05)))

        # Efficiency ratings (100-125 range)
        base_off_rating = 112
        base_def_rating = 112
        home_off_rating = max(100, min(125, base_off_rating + (home_rating - 0.5) * 15 + team_offense_bias[home_team] * 50 + random.gauss(0, 2)))
        away_off_rating = max(100, min(125, base_off_rating + (away_rating - 0.5) * 15 + team_offense_bias[away_team] * 50 + random.gauss(0, 2)))
        home_def_rating = max(100, min(125, base_def_rating - (home_rating - 0.5) * 12 - team_defense_bias[home_team] * 50 + random.gauss(0, 2)))
        away_def_rating = max(100, min(125, base_def_rating - (away_rating - 0.5) * 12 - team_defense_bias[away_team] * 50 + random.gauss(0, 2)))

        home_net_rating = home_off_rating - home_def_rating
        away_net_rating = away_off_rating - away_def_rating

        # Points per game (100-130 range)
        home_pts_avg = max(100, min(130, 112 + (home_rating - 0.5) * 15 + random.gauss(0, 3)))
        away_pts_avg = max(100, min(130, 112 + (away_rating - 0.5) * 15 + random.gauss(0, 3)))

        # Pace (95-105)
        home_pace = max(95, min(108, 100 + team_pace_bias[home_team] + random.gauss(0, 2)))
        away_pace = max(95, min(108, 100 + team_pace_bias[away_team] + random.gauss(0, 2)))

        # Streaks (-10 to +10)
        home_streak = random.randint(-8, 10) if random.random() < 0.3 else random.randint(-3, 5)
        away_streak = random.randint(-8, 10) if random.random() < 0.3 else random.randint(-3, 5)

        # Shooting percentages (0.40-0.50 for FG%, 0.30-0.42 for 3P%)
        home_fg_pct = max(0.42, min(0.50, 0.46 + (home_rating - 0.5) * 0.03 + random.gauss(0, 0.01)))
        away_fg_pct = max(0.42, min(0.50, 0.46 + (away_rating - 0.5) * 0.03 + random.gauss(0, 0.01)))
        home_fg3_pct = max(0.32, min(0.42, 0.36 + random.gauss(0, 0.02)))
        away_fg3_pct = max(0.32, min(0.42, 0.36 + random.gauss(0, 0.02)))

        # Assists (22-30 per game)
        home_ast_avg = max(22, min(32, 25 + random.gauss(0, 2)))
        away_ast_avg = max(22, min(32, 25 + random.gauss(0, 2)))

        # Rebounds (42-50 per game)
        home_reb_avg = max(40, min(52, 45 + random.gauss(0, 2)))
        away_reb_avg = max(40, min(52, 45 + random.gauss(0, 2)))

        # Plus/minus (average point differential per game)
        home_plus_minus = (home_rating - 0.5) * 10 + random.gauss(0, 2)
        away_plus_minus = (away_rating - 0.5) * 10 + random.gauss(0, 2)

        moneyline_features = {
            # Win percentage differentials (all should be -1.0 to +1.0 range)
            "season_win_pct_diff": home_season_win_pct - away_season_win_pct,
            "recent_win_pct_diff": home_recent_win_pct - away_recent_win_pct,
            "location_win_pct_diff": home_location_win_pct - away_location_win_pct,

            # Scoring differentials (-25 to +25 points)
            "pts_avg_diff": home_pts_avg - away_pts_avg,
            "recent_pts_diff": (home_pts_avg - away_pts_avg) + random.gauss(0, 3),

            # Efficiency differentials (-20 to +20)
            "off_rating_diff": home_off_rating - away_off_rating,
            "def_rating_diff": home_def_rating - away_def_rating,
            "net_rating_diff": home_net_rating - away_net_rating,

            # Streaks
            "home_streak": home_streak,
            "away_streak": away_streak,
            "combined_form": home_plus_minus - away_plus_minus,

            # Home advantage factor (0.0 to 0.15)
            "home_advantage_factor": 0.06 + random.gauss(0, 0.02),

            # Shooting differentials (-0.08 to +0.08)
            "fg_pct_diff": home_fg_pct - away_fg_pct,
            "fg3_pct_diff": home_fg3_pct - away_fg3_pct,

            # Pace (95-105 average)
            "avg_pace": (home_pace + away_pace) / 2,
            "pace_diff": home_pace - away_pace,

            # Individual team stats
            "home_season_win_pct": home_season_win_pct,
            "away_season_win_pct": away_season_win_pct,
            "home_net_rating": home_net_rating,
            "away_net_rating": away_net_rating,
            "home_off_rating": home_off_rating,
            "away_off_rating": away_off_rating,
            "home_def_rating": home_def_rating,
            "away_def_rating": away_def_rating,
        }

        # =====================================================================
        # SPREAD FEATURES - Includes moneyline features plus spread-specific
        # =====================================================================
        spread_features = {
            **moneyline_features,  # Include all moneyline features

            # Point estimates (all per-game, 85-130 range)
            "expected_home_pts": max(95, min(130, home_pts_avg + random.gauss(0, 3))),
            "expected_away_pts": max(95, min(130, away_pts_avg + random.gauss(0, 3))),
            "expected_point_diff": max(-25, min(25, expected_diff + random.gauss(0, 2))),

            # Plus/minus (-15 to +15)
            "home_plus_minus": max(-15, min(15, home_plus_minus)),
            "away_plus_minus": max(-15, min(15, away_plus_minus)),
            "plus_minus_diff": max(-20, min(20, home_plus_minus - away_plus_minus)),

            # Team averages (per-game)
            "home_pts_avg": home_pts_avg,
            "away_pts_avg": away_pts_avg,

            # Rebounding differential (-10 to +10)
            "reb_diff": max(-10, min(10, home_reb_avg - away_reb_avg)),

            # Assist differential (-8 to +8)
            "ast_diff": max(-8, min(8, home_ast_avg - away_ast_avg)),
        }

        game_record = {
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_win": home_win,
            "point_differential": float(point_diff),
            "home_score": home_score,
            "away_score": away_score,
            "moneyline_features": moneyline_features,
            "spread_features": spread_features,
        }

        games.append(game_record)

    # Validate generated data
    valid_count = 0
    for g in games:
        sf = g.get("spread_features", {})
        # Check that key values are in reasonable ranges
        if (abs(sf.get("pts_avg_diff", 0)) < 30 and
            abs(sf.get("expected_point_diff", 0)) < 30 and
            80 < sf.get("home_pts_avg", 0) < 140 and
            80 < sf.get("away_pts_avg", 0) < 140):
            valid_count += 1

    print(f"Generated {len(games)} synthetic training games")
    print(f"  Valid games: {valid_count}/{len(games)} ({100*valid_count/len(games):.1f}%)")

    # Show sample statistics
    home_wins = sum(1 for g in games if g["home_win"])
    avg_diff = sum(g["point_differential"] for g in games) / len(games)
    print(f"  Home win rate: {100*home_wins/len(games):.1f}%")
    print(f"  Avg point differential: {avg_diff:.1f} (home team)")

    return games


def fetch_training_data_fast(
    num_games: int = 100,
    season: str = "2025-26",
    use_synthetic: bool = False,
) -> List[Dict]:
    """
    Main entry point for fast training data collection.

    Args:
        num_games: Number of games to fetch
        season: NBA season (e.g., "2025-26")
        use_synthetic: If True, generate synthetic data (fastest option)

    Returns:
        List of game dictionaries ready for training
    """
    if use_synthetic:
        return generate_synthetic_training_data(num_games)

    # Try balldontlie API
    season_year = int(season.split("-")[0])

    # Calculate date range for current season
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = f"{season_year}-10-01"

    games = fetch_games_fast(
        start_date=start_date,
        end_date=end_date,
        season=season_year,
        max_pages=num_games // 100 + 1,
    )

    if not games:
        print("\nAPI data unavailable. Generating synthetic data instead...")
        return generate_synthetic_training_data(num_games)

    # Convert to training format
    training_data = []
    for game in games[:num_games]:
        home_team = game.get("home_team", {})
        away_team = game.get("visitor_team", {})

        home_abbrev = home_team.get("abbreviation", "")
        away_abbrev = away_team.get("abbreviation", "")
        home_score = game.get("home_team_score", 0)
        away_score = game.get("visitor_team_score", 0)

        if not all([home_abbrev, away_abbrev, home_score, away_score]):
            continue

        training_data.append({
            "game_date": game.get("date", ""),
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "home_win": home_score > away_score,
            "point_differential": home_score - away_score,
            "home_score": home_score,
            "away_score": away_score,
            "moneyline_features": {},  # Will need to generate
            "spread_features": {},
        })

    return training_data


if __name__ == "__main__":
    # Test the fast fetcher
    print("Testing fast data fetcher...\n")

    # Generate synthetic data (fastest)
    games = generate_synthetic_training_data(50)

    print(f"\nSample game:")
    print(json.dumps(games[0], indent=2))

    # Show win distribution
    home_wins = sum(1 for g in games if g["home_win"])
    print(f"\nHome win rate: {home_wins}/{len(games)} ({home_wins/len(games):.1%})")
