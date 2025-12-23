"""
Kaggle/GitHub NBA Data Loader

Loads historical NBA data from CSV files (no rate limits!).
Data source: https://github.com/NocturneBear/NBA-Data-2010-2024

This provides 33,000+ games from 2010-2024 with full box scores.
Much faster and more reliable than API-based data collection.

Usage:
    from kaggle_data_loader import load_training_data_from_csv
    games = load_training_data_from_csv(seasons=["2022-23", "2023-24"])
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Data directory
DATA_DIR = Path(__file__).parent / "data" / "NBA-Data-2010-2024-main"

# Team abbreviation mappings (handle variations)
TEAM_ABBREV_MAP = {
    "NJN": "BKN",  # New Jersey Nets -> Brooklyn Nets
    "NOH": "NOP",  # New Orleans Hornets -> Pelicans
    "NOK": "NOP",  # Alternative
    "SEA": "OKC",  # Seattle -> OKC (historical)
    "VAN": "MEM",  # Vancouver -> Memphis (historical)
    "CHH": "CHA",  # Charlotte Hornets variations
    "PHO": "PHX",  # Phoenix variations
}


def normalize_team_abbrev(abbrev: str) -> str:
    """Normalize team abbreviations to current format."""
    return TEAM_ABBREV_MAP.get(abbrev, abbrev)


def load_raw_game_data(
    include_playoffs: bool = False,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load raw game data from CSV files.

    Args:
        include_playoffs: Whether to include playoff games
        data_dir: Optional custom data directory

    Returns:
        DataFrame with all game records
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Run: curl -L -o data/nba_data.zip 'https://github.com/NocturneBear/NBA-Data-2010-2024/archive/refs/heads/main.zip'"
        )

    print(f"Loading NBA data from: {data_dir}")

    # Load regular season data
    regular_season_file = data_dir / "regular_season_totals_2010_2024.csv"
    if not regular_season_file.exists():
        raise FileNotFoundError(f"Regular season data not found: {regular_season_file}")

    df = pd.read_csv(regular_season_file)
    print(f"  Loaded {len(df)} regular season game records")

    # Optionally add playoffs
    if include_playoffs:
        playoff_file = data_dir / "play_off_totals_2010_2024.csv"
        if playoff_file.exists():
            playoff_df = pd.read_csv(playoff_file)
            playoff_df["is_playoff"] = True
            df["is_playoff"] = False
            df = pd.concat([df, playoff_df], ignore_index=True)
            print(f"  Added {len(playoff_df)} playoff game records")

    return df


def process_games_to_matchups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert individual team game records to matchup records.

    Each game appears twice in the raw data (once per team).
    This combines them into single game records with home/away distinction.

    Args:
        df: Raw game data DataFrame

    Returns:
        DataFrame with one row per game (home team perspective)
    """
    print("Processing games into matchups...")

    # Parse game date (handle multiple formats)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")

    # Normalize team abbreviations
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].apply(normalize_team_abbrev)

    # Determine home/away from MATCHUP column
    # Format: "GSW @ POR" (away) or "GSW vs. POR" (home)
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs\\.", regex=True)

    # Extract opponent
    def extract_opponent(matchup: str) -> str:
        if "@" in matchup:
            return matchup.split("@")[1].strip()
        elif "vs." in matchup:
            return matchup.split("vs.")[1].strip()
        return ""

    df["OPPONENT"] = df["MATCHUP"].apply(extract_opponent)
    df["OPPONENT"] = df["OPPONENT"].apply(normalize_team_abbrev)

    # Group by GAME_ID to get both teams
    games = []
    game_ids = df["GAME_ID"].unique()

    for game_id in game_ids:
        game_rows = df[df["GAME_ID"] == game_id]

        if len(game_rows) != 2:
            continue

        home_row = game_rows[game_rows["IS_HOME"] == True]
        away_row = game_rows[game_rows["IS_HOME"] == False]

        if len(home_row) != 1 or len(away_row) != 1:
            continue

        home = home_row.iloc[0]
        away = away_row.iloc[0]

        games.append({
            "game_id": game_id,
            "game_date": home["GAME_DATE"],
            "season": home["SEASON_YEAR"],
            "home_team": home["TEAM_ABBREVIATION"],
            "away_team": away["TEAM_ABBREVIATION"],
            "home_score": home["PTS"],
            "away_score": away["PTS"],
            "home_win": home["WL"] == "W",
            "point_differential": home["PTS"] - away["PTS"],
            # Home team stats
            "home_fg_pct": home["FG_PCT"],
            "home_fg3_pct": home["FG3_PCT"],
            "home_ft_pct": home["FT_PCT"],
            "home_reb": home["REB"],
            "home_ast": home["AST"],
            "home_tov": home["TOV"],
            "home_stl": home["STL"],
            "home_blk": home["BLK"],
            "home_plus_minus": home["PLUS_MINUS"],
            # Away team stats
            "away_fg_pct": away["FG_PCT"],
            "away_fg3_pct": away["FG3_PCT"],
            "away_ft_pct": away["FT_PCT"],
            "away_reb": away["REB"],
            "away_ast": away["AST"],
            "away_tov": away["TOV"],
            "away_stl": away["STL"],
            "away_blk": away["BLK"],
            "away_plus_minus": away["PLUS_MINUS"],
        })

    result = pd.DataFrame(games)
    result = result.sort_values("game_date").reset_index(drop=True)

    print(f"  Processed {len(result)} unique games")
    return result


def calculate_rolling_stats(
    games_df: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """
    Calculate rolling team statistics for each game.

    Uses only past data to avoid look-ahead bias.

    Args:
        games_df: Processed games DataFrame
        window: Number of past games to use

    Returns:
        DataFrame with rolling stats added
    """
    print(f"Calculating rolling stats (window={window})...")

    games_df = games_df.copy()

    # Get unique teams
    all_teams = set(games_df["home_team"].unique()) | set(games_df["away_team"].unique())

    # Calculate rolling stats per team
    team_stats = {}

    for team in all_teams:
        # Get all games for this team (home or away)
        team_home = games_df[games_df["home_team"] == team].copy()
        team_away = games_df[games_df["away_team"] == team].copy()

        # Standardize column names for combining
        team_home["team"] = team
        team_home["is_home"] = True
        team_home["pts"] = team_home["home_score"]
        team_home["opp_pts"] = team_home["away_score"]
        team_home["won"] = team_home["home_win"]
        team_home["fg_pct"] = team_home["home_fg_pct"]
        team_home["fg3_pct"] = team_home["home_fg3_pct"]

        team_away["team"] = team
        team_away["is_home"] = False
        team_away["pts"] = team_away["away_score"]
        team_away["opp_pts"] = team_away["home_score"]
        team_away["won"] = ~team_away["home_win"]
        team_away["fg_pct"] = team_away["away_fg_pct"]
        team_away["fg3_pct"] = team_away["away_fg3_pct"]

        team_games = pd.concat([team_home, team_away]).sort_values("game_date")

        # Calculate rolling averages (shift by 1 to exclude current game)
        team_games["roll_pts_avg"] = team_games["pts"].shift(1).rolling(window, min_periods=3).mean()
        team_games["roll_opp_pts_avg"] = team_games["opp_pts"].shift(1).rolling(window, min_periods=3).mean()
        team_games["roll_win_pct"] = team_games["won"].shift(1).rolling(window, min_periods=3).mean()
        team_games["roll_fg_pct"] = team_games["fg_pct"].shift(1).rolling(window, min_periods=3).mean()
        team_games["roll_fg3_pct"] = team_games["fg3_pct"].shift(1).rolling(window, min_periods=3).mean()

        # Store by game_id
        for _, row in team_games.iterrows():
            game_id = row["game_id"]
            if game_id not in team_stats:
                team_stats[game_id] = {}

            prefix = "home" if row["is_home"] else "away"
            team_stats[game_id][f"{prefix}_roll_pts_avg"] = row["roll_pts_avg"]
            team_stats[game_id][f"{prefix}_roll_opp_pts_avg"] = row["roll_opp_pts_avg"]
            team_stats[game_id][f"{prefix}_roll_win_pct"] = row["roll_win_pct"]
            team_stats[game_id][f"{prefix}_roll_fg_pct"] = row["roll_fg_pct"]
            team_stats[game_id][f"{prefix}_roll_fg3_pct"] = row["roll_fg3_pct"]

    # Merge rolling stats back
    stats_df = pd.DataFrame.from_dict(team_stats, orient="index")
    stats_df.index.name = "game_id"
    stats_df = stats_df.reset_index()

    games_df = games_df.merge(stats_df, on="game_id", how="left")

    print(f"  Added rolling statistics for {len(all_teams)} teams")
    return games_df


def generate_training_features(games_df: pd.DataFrame) -> List[Dict]:
    """
    Generate training features from processed games data.

    Args:
        games_df: DataFrame with games and rolling stats

    Returns:
        List of training examples with features and outcomes
    """
    print("Generating training features...")

    training_data = []

    for _, game in games_df.iterrows():
        # Skip games without rolling stats (early season)
        if pd.isna(game.get("home_roll_win_pct")) or pd.isna(game.get("away_roll_win_pct")):
            continue

        # Calculate differentials
        win_pct_diff = game["home_roll_win_pct"] - game["away_roll_win_pct"]
        pts_avg_diff = game["home_roll_pts_avg"] - game["away_roll_pts_avg"]
        fg_pct_diff = game["home_roll_fg_pct"] - game["away_roll_fg_pct"]
        fg3_pct_diff = game["home_roll_fg3_pct"] - game["away_roll_fg3_pct"]

        # Approximate net rating from pts differential
        home_net = game["home_roll_pts_avg"] - game["home_roll_opp_pts_avg"]
        away_net = game["away_roll_pts_avg"] - game["away_roll_opp_pts_avg"]
        net_rating_diff = home_net - away_net

        # Features for moneyline prediction
        moneyline_features = {
            "season_win_pct_diff": win_pct_diff,
            "recent_win_pct_diff": win_pct_diff,  # Same as season for rolling
            "location_win_pct_diff": win_pct_diff + 0.05,  # Home advantage adjustment
            "pts_avg_diff": pts_avg_diff,
            "recent_pts_diff": pts_avg_diff,
            "off_rating_diff": pts_avg_diff * 1.1,  # Approximate
            "def_rating_diff": (game["home_roll_opp_pts_avg"] - game["away_roll_opp_pts_avg"]),
            "net_rating_diff": net_rating_diff,
            "home_streak": 0,  # Not available in this dataset
            "away_streak": 0,
            "combined_form": net_rating_diff,
            "home_advantage_factor": 0.06,
            "fg_pct_diff": fg_pct_diff,
            "fg3_pct_diff": fg3_pct_diff,
            "avg_pace": 100,  # Default pace
            "pace_diff": 0,
            "home_season_win_pct": game["home_roll_win_pct"],
            "away_season_win_pct": game["away_roll_win_pct"],
            "home_net_rating": home_net,
            "away_net_rating": away_net,
            "home_off_rating": game["home_roll_pts_avg"] * 1.1,
            "away_off_rating": game["away_roll_pts_avg"] * 1.1,
            "home_def_rating": game["home_roll_opp_pts_avg"] * 1.1,
            "away_def_rating": game["away_roll_opp_pts_avg"] * 1.1,
        }

        # Features for spread prediction (includes all moneyline features + more)
        spread_features = {
            **moneyline_features,
            "expected_home_pts": game["home_roll_pts_avg"],
            "expected_away_pts": game["away_roll_pts_avg"],
            "expected_point_diff": net_rating_diff + 3,  # Home advantage
            "home_plus_minus": home_net,
            "away_plus_minus": away_net,
            "plus_minus_diff": net_rating_diff,
            "home_pts_avg": game["home_roll_pts_avg"],
            "away_pts_avg": game["away_roll_pts_avg"],
            "reb_diff": 0,  # Not calculated
            "ast_diff": 0,
        }

        training_data.append({
            "game_id": game["game_id"],
            "game_date": game["game_date"].strftime("%Y-%m-%d") if hasattr(game["game_date"], "strftime") else str(game["game_date"]),
            "season": game["season"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_win": bool(game["home_win"]),
            "point_differential": float(game["point_differential"]),
            "home_score": int(game["home_score"]),
            "away_score": int(game["away_score"]),
            "moneyline_features": moneyline_features,
            "spread_features": spread_features,
        })

    print(f"  Generated {len(training_data)} training examples")
    return training_data


def load_training_data_from_csv(
    seasons: Optional[List[str]] = None,
    include_playoffs: bool = False,
    window: int = 10,
    data_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Main function to load training data from CSV files.

    Args:
        seasons: List of seasons to include (e.g., ["2022-23", "2023-24"])
                If None, uses all available seasons
        include_playoffs: Whether to include playoff games
        window: Rolling window size for stats calculation
        data_dir: Optional custom data directory

    Returns:
        List of training examples with features and outcomes
    """
    print("\n" + "="*60)
    print("Loading NBA Training Data from CSV")
    print("="*60)

    # Load raw data
    df = load_raw_game_data(include_playoffs=include_playoffs, data_dir=data_dir)

    # Filter by seasons if specified
    if seasons:
        df = df[df["SEASON_YEAR"].isin(seasons)]
        print(f"Filtered to seasons: {seasons}")
        print(f"  {len(df)} records remaining")

    # Process into game matchups
    games_df = process_games_to_matchups(df)

    # Calculate rolling stats
    games_df = calculate_rolling_stats(games_df, window=window)

    # Generate training features
    training_data = generate_training_features(games_df)

    # Summary statistics
    home_wins = sum(1 for g in training_data if g["home_win"])
    avg_diff = sum(g["point_differential"] for g in training_data) / len(training_data)

    print("\n" + "="*60)
    print("Data Loading Complete!")
    print("="*60)
    print(f"Total training examples: {len(training_data)}")
    print(f"Home win rate: {home_wins/len(training_data):.1%}")
    print(f"Average point differential: {avg_diff:+.1f}")
    if seasons:
        print(f"Seasons: {', '.join(seasons)}")
    else:
        print(f"Seasons: All available (2010-2024)")

    return training_data


def get_available_seasons(data_dir: Optional[Path] = None) -> List[str]:
    """Get list of available seasons in the dataset."""
    if data_dir is None:
        data_dir = DATA_DIR

    df = pd.read_csv(data_dir / "regular_season_totals_2010_2024.csv", usecols=["SEASON_YEAR"])
    seasons = sorted(df["SEASON_YEAR"].unique())
    return seasons


def load_live_season_data() -> pd.DataFrame:
    """
    Load live season data fetched from API.

    Returns:
        DataFrame with live season data (2023-24, 2024-25, 2025-26)
    """
    live_dir = Path(__file__).parent / "data" / "live_seasons"

    if not live_dir.exists():
        return pd.DataFrame()

    # Find latest file
    csv_files = sorted(live_dir.glob("live_seasons_*.csv"), reverse=True)
    if not csv_files:
        return pd.DataFrame()

    latest = csv_files[0]
    print(f"Loading live data from: {latest}")

    df = pd.read_csv(latest)

    # Ensure SEASON_YEAR column exists (nba_api uses it)
    if "SEASON_YEAR" not in df.columns and "SEASON_ID" in df.columns:
        # Convert SEASON_ID (22023) to SEASON_YEAR (2023-24)
        def season_id_to_year(sid):
            year = int(str(sid)[-4:])
            return f"{year}-{str(year+1)[-2:]}"
        df["SEASON_YEAR"] = df["SEASON_ID"].apply(season_id_to_year)

    return df


def load_training_data_with_live(
    csv_seasons: Optional[List[str]] = None,
    live_seasons: Optional[List[str]] = None,
    include_playoffs: bool = False,
    window: int = 10,
) -> List[Dict]:
    """
    Load training data from both CSV (historical) and live API data.

    Args:
        csv_seasons: Seasons to load from CSV (e.g., ["2021-22", "2022-23"])
                    Default: ["2021-22", "2022-23"]
        live_seasons: Seasons to load from live API data (e.g., ["2023-24", "2024-25", "2025-26"])
                     Default: All available in live data
        include_playoffs: Whether to include playoff games
        window: Rolling window size for stats calculation

    Returns:
        List of training examples with features and outcomes
    """
    print("\n" + "="*60)
    print("Loading NBA Training Data (CSV + Live)")
    print("="*60)

    # Default CSV seasons (historical data)
    if csv_seasons is None:
        csv_seasons = ["2023-24"]  # Most recent complete season in CSV data

    # Load historical CSV data
    print(f"\n--- Loading Historical CSV Data ---")
    print(f"Seasons: {csv_seasons}")
    csv_df = load_raw_game_data(include_playoffs=include_playoffs)
    csv_df = csv_df[csv_df["SEASON_YEAR"].isin(csv_seasons)]
    print(f"  Loaded {len(csv_df)} records from CSV")

    # Load live API data
    print(f"\n--- Loading Live API Data ---")
    live_df = load_live_season_data()

    if live_df.empty:
        print("  No live data found. Run: python3 live_season_fetcher.py")
        combined_df = csv_df
    else:
        if live_seasons:
            live_df = live_df[live_df["SEASON_YEAR"].isin(live_seasons)]

        print(f"  Loaded {len(live_df)} records from live API")
        print(f"  Live seasons: {live_df['SEASON_YEAR'].unique().tolist()}")

        # Ensure column compatibility
        common_cols = list(set(csv_df.columns) & set(live_df.columns))
        csv_df = csv_df[common_cols]
        live_df = live_df[common_cols]

        # Combine datasets
        combined_df = pd.concat([csv_df, live_df], ignore_index=True)

    print(f"\n--- Combined Dataset ---")
    print(f"Total records: {len(combined_df)}")

    # Process into game matchups
    games_df = process_games_to_matchups(combined_df)

    # Calculate rolling stats
    games_df = calculate_rolling_stats(games_df, window=window)

    # Generate training features
    training_data = generate_training_features(games_df)

    # Summary statistics
    home_wins = sum(1 for g in training_data if g["home_win"])
    avg_diff = sum(g["point_differential"] for g in training_data) / len(training_data) if training_data else 0

    print("\n" + "="*60)
    print("Data Loading Complete!")
    print("="*60)
    print(f"Total training examples: {len(training_data)}")
    print(f"Home win rate: {home_wins/len(training_data):.1%}" if training_data else "N/A")
    print(f"Average point differential: {avg_diff:+.1f}")

    # Show season breakdown
    seasons_in_data = set(g["season"] for g in training_data)
    print(f"Seasons included: {sorted(seasons_in_data)}")

    return training_data


if __name__ == "__main__":
    print("Testing Kaggle/GitHub NBA Data Loader\n")

    # Check available seasons
    try:
        seasons = get_available_seasons()
        print(f"Available seasons: {seasons}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo download data, run:")
        print("  mkdir -p data && cd data")
        print("  curl -L -o nba_data.zip 'https://github.com/NocturneBear/NBA-Data-2010-2024/archive/refs/heads/main.zip'")
        print("  unzip nba_data.zip")
        exit(1)

    # Load recent seasons
    training_data = load_training_data_from_csv(
        seasons=["2023-24"],
        include_playoffs=False,
        window=10,
    )

    # Show sample
    if training_data:
        print("\nSample training example:")
        sample = training_data[0]
        print(f"  {sample['away_team']} @ {sample['home_team']} ({sample['game_date']})")
        print(f"  Score: {sample['away_score']} - {sample['home_score']}")
        print(f"  Home win: {sample['home_win']}")
        print(f"  Point diff: {sample['point_differential']:+.0f}")
