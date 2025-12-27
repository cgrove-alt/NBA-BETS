"""
NBA Model Training Script

Fetches historical NBA game data and trains all ML models:
- Moneyline Model (Logistic Regression)
- Spread Model (SVM Regressor)
- Player Prop Models (Random Forest)
- XGBoost Moneyline Model
- Ensemble Models

Usage (RECOMMENDED: Use Kaggle/CSV data for best results):
    python3 train_models.py --kaggle --live   # BEST: CSV + live API data (all recent seasons)
    python3 train_models.py --kaggle          # CSV data only (2010-2023) - fast, reliable
    python3 train_models.py --kaggle --seasons 2021-22,2022-23  # Specific CSV seasons
    python3 train_models.py --use-database    # Use collected database data
    python3 train_models.py --fast            # Use synthetic data (for testing only)
    python3 train_models.py [--games 100]     # SLOW: Fetch from NBA API (rate limited, not recommended)

Data Source Hierarchy (fastest to slowest):
    1. Kaggle/CSV (--kaggle) - 33K+ games, instant loading, RECOMMENDED
    2. Database (--use-database) - Point-in-time safe, requires prior collection
    3. NBA API (default) - Rate limited, slow, use only for testing
"""

# API safeguard: Limit games when using slow API path
MAX_API_GAMES = 20  # Maximum games to fetch via API (prevents long waits)
API_WARNING_THRESHOLD = 50  # Warn if user requests more than this via API

import argparse
import json
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import our modules
from data_fetcher import (
    fetch_historical_games as get_team_game_log,
    fetch_player_stats,
    fetch_team_roster,
    get_team_id,
    API_DELAY,
)
from nba_api.stats.static import teams as nba_teams
from feature_engineering import generate_game_features, generate_player_features
from model_trainer import ModelTrainingPipeline

# Fast data fetcher (no rate limiting)
from fast_data_fetcher import generate_synthetic_training_data, fetch_training_data_fast

# Database support
try:
    from database import DatabaseManager
    from historical_data_collector import generate_training_features_from_db
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False

# Kaggle/CSV data support (no rate limits!)
try:
    from kaggle_data_loader import load_training_data_from_csv, load_training_data_with_live
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False

def get_all_teams():
    """Get all NBA teams."""
    return nba_teams.get_teams()

# Rate limiting
def rate_limit():
    time.sleep(API_DELAY)


def fetch_historical_games(
    num_games: int = 100,
    season: str = "2025-26",
) -> List[Dict]:
    """
    Fetch historical game data with outcomes.

    Args:
        num_games: Target number of games to fetch
        season: NBA season string

    Returns:
        List of game dictionaries with features and outcomes
    """
    print(f"\n{'='*60}")
    print(f"Fetching historical game data ({num_games} games target)")
    print(f"Season: {season}")
    print(f"{'='*60}\n")

    games_data = []
    teams = get_all_teams()
    rate_limit()

    if not teams:
        print("Error: Could not fetch team list")
        return []

    # Get a subset of teams to reduce API calls
    team_list = teams[:15]  # Top 15 teams

    games_processed = set()  # Track unique games by (team1, team2, date)

    for team in team_list:
        if len(games_data) >= num_games:
            break

        team_abbrev = team['abbreviation']
        team_id = team['id']
        print(f"\nFetching game log for {team_abbrev}...")

        try:
            game_log = get_team_game_log(team_id=team_id, season=season, last_n_games=20)
            rate_limit()

            if not game_log:
                print(f"  No games found for {team_abbrev}")
                continue

            for game in game_log[:10]:  # Process up to 10 games per team
                if len(games_data) >= num_games:
                    break

                # Parse game info (keys may be upper or lowercase depending on API)
                matchup = game.get('MATCHUP', '') or game.get('matchup', '')
                game_date = game.get('GAME_DATE', '') or game.get('game_date', '')

                # Determine home/away
                is_home = '@' not in matchup

                # Extract opponent
                if '@' in matchup:
                    opp_abbrev = matchup.split('@')[1].strip().split()[-1]
                else:
                    opp_abbrev = matchup.split('vs.')[1].strip().split()[-1] if 'vs.' in matchup else None

                if not opp_abbrev:
                    continue

                # Create unique game key
                game_key = tuple(sorted([team_abbrev, opp_abbrev]) + [game_date])
                if game_key in games_processed:
                    continue
                games_processed.add(game_key)

                # Determine home and away teams
                if is_home:
                    home_abbrev = team_abbrev
                    away_abbrev = opp_abbrev
                else:
                    home_abbrev = opp_abbrev
                    away_abbrev = team_abbrev

                # Get game outcome (keys may be upper or lowercase depending on API)
                wl = game.get('WL', '') or game.get('wl', '')
                pts = game.get('PTS', 0) or game.get('pts', 0) or 0

                # Calculate actual point differential (home - away perspective)
                plus_minus = game.get('PLUS_MINUS', 0) or game.get('plus_minus', 0) or 0
                if is_home:
                    home_pts = pts
                    away_pts = pts - plus_minus
                else:
                    away_pts = pts
                    home_pts = pts + plus_minus

                point_diff = home_pts - away_pts
                home_win = point_diff > 0

                print(f"  Processing: {away_abbrev} @ {home_abbrev} ({game_date})")

                try:
                    # Generate features for this game with point-in-time stats
                    # Pass as_of_date to ensure we only use data available before game
                    features = generate_game_features(
                        home_abbrev,
                        away_abbrev,
                        season=season,
                        include_advanced=False,  # Faster
                        game_date=game_date,  # CRITICAL: Prevents temporal leakage
                    )
                    rate_limit()

                    if features:
                        game_record = {
                            "game_date": game_date,
                            "home_team": home_abbrev,
                            "away_team": away_abbrev,
                            "home_win": home_win,
                            "point_differential": point_diff,
                            "home_score": home_pts,
                            "away_score": away_pts,
                            "moneyline_features": features.get("moneyline_features", {}),
                            "spread_features": features.get("spread_features", {}),
                        }
                        games_data.append(game_record)
                        print(f"    ✓ Added game: {away_abbrev} @ {home_abbrev} (diff: {int(point_diff):+d})")
                    else:
                        print(f"    ✗ No features generated")

                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    continue

        except Exception as e:
            print(f"  Error fetching {team_abbrev}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Total games fetched: {len(games_data)}")
    print(f"{'='*60}")

    return games_data


def fetch_player_data(
    num_players: int = 50,
    games_per_player: int = 10,
    season: str = "2025-26",
) -> List[Dict]:
    """
    Fetch historical player game data with outcomes.

    Args:
        num_players: Number of players to fetch
        games_per_player: Games per player
        season: NBA season string

    Returns:
        List of player game dictionaries with features and outcomes
    """
    print(f"\n{'='*60}")
    print(f"Fetching player data ({num_players} players)")
    print(f"{'='*60}\n")

    player_data = []
    teams = get_all_teams()
    rate_limit()

    players_processed = 0

    for team in teams[:10]:  # Process 10 teams
        if players_processed >= num_players:
            break

        team_abbrev = team['abbreviation']
        team_id = team['id']

        print(f"\nFetching roster for {team_abbrev}...")

        try:
            roster = get_team_roster(team_id, season=season)
            rate_limit()

            if not roster:
                continue

            # Get top players from roster
            for player in roster[:5]:  # Top 5 players per team
                if players_processed >= num_players:
                    break

                player_id = player.get('PLAYER_ID')
                player_name = player.get('PLAYER', 'Unknown')

                print(f"  Processing {player_name}...")

                try:
                    game_log = get_player_game_log(player_id, season=season, last_n_games=games_per_player)
                    rate_limit()

                    if not game_log:
                        continue

                    for game in game_log:
                        pts = game.get('PTS', 0) or 0
                        reb = game.get('REB', 0) or 0
                        ast = game.get('AST', 0) or 0
                        fg3m = game.get('FG3M', 0) or 0
                        min_played = game.get('MIN', 0) or 0

                        # Skip games with low minutes (likely injury/rest)
                        if min_played < 15:
                            continue

                        # Generate features
                        try:
                            prop_features = generate_player_prop_features(
                                player_id,
                                team_abbrev,  # Approximate
                                season=season,
                            )
                            rate_limit()

                            if prop_features:
                                player_record = {
                                    "player_id": player_id,
                                    "player_name": player_name,
                                    "game_date": game.get('GAME_DATE', ''),
                                    "points_features": prop_features.get("points", {}),
                                    "rebounds_features": prop_features.get("rebounds", {}),
                                    "assists_features": prop_features.get("assists", {}),
                                    "threes_features": prop_features.get("threes", {}),
                                    "pra_features": prop_features.get("pra", {}),
                                    "actual_stats": {
                                        "pts": pts,
                                        "reb": reb,
                                        "ast": ast,
                                        "fg3_made": fg3m,
                                        "pra": pts + reb + ast,
                                    },
                                }
                                player_data.append(player_record)

                        except Exception as e:
                            continue

                    players_processed += 1
                    print(f"    ✓ Added {len([p for p in player_data if p['player_id'] == player_id])} games")

                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    continue

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Total player game records: {len(player_data)}")
    print(f"{'='*60}")

    return player_data


def save_training_data(
    games_data: List[Dict],
    player_data: List[Dict],
    output_dir: str = "training_data",
):
    """Save training data to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    games_file = output_path / f"games_data_{timestamp}.json"
    players_file = output_path / f"player_data_{timestamp}.json"

    with open(games_file, 'w') as f:
        json.dump(games_data, f, indent=2, default=str)
    print(f"Games data saved to {games_file}")

    with open(players_file, 'w') as f:
        json.dump(player_data, f, indent=2, default=str)
    print(f"Player data saved to {players_file}")

    return games_file, players_file


def load_training_data(
    games_file: Optional[str] = None,
    players_file: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Load training data from JSON files."""
    games_data = []
    player_data = []

    if games_file and Path(games_file).exists():
        with open(games_file, 'r') as f:
            games_data = json.load(f)
        print(f"Loaded {len(games_data)} games from {games_file}")

    if players_file and Path(players_file).exists():
        with open(players_file, 'r') as f:
            player_data = json.load(f)
        print(f"Loaded {len(player_data)} player records from {players_file}")

    return games_data, player_data


def load_training_data_from_database(
    db_path: str = "nba_betting.db",
    seasons: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Load training data from SQLite database.

    Uses point-in-time stats to avoid look-ahead bias.

    Args:
        db_path: Path to database file
        seasons: Optional list of seasons to include (e.g., ["2022-23", "2023-24"])

    Returns:
        List of game dictionaries with features
    """
    if not HAS_DATABASE:
        print("Error: Database modules not available")
        return []

    db = DatabaseManager(Path(db_path))

    print(f"\nLoading training data from database: {db_path}")

    # Check if database has data
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM games WHERE status = 'completed'")
        game_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM team_stats_snapshot")
        snapshot_count = cursor.fetchone()[0]

    print(f"  Found {game_count} completed games")
    print(f"  Found {snapshot_count} team stats snapshots")

    if game_count == 0:
        print("\nNo games in database. Run historical_data_collector.py first.")
        return []

    # Generate features from database
    games_data = generate_training_features_from_db(db)

    if seasons:
        # Filter by season if specified
        games_data = [
            g for g in games_data
            if any(s in g.get("game_date", "") for s in seasons)
        ]

    print(f"  Generated {len(games_data)} training examples with features")

    return games_data


def load_training_data_from_kaggle(
    seasons: Optional[List[str]] = None,
    include_playoffs: bool = False,
    data_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Load training data from Kaggle/CSV files (NO RATE LIMITS!).

    This is the fastest and most reliable data source for historical games.
    Uses pre-downloaded CSV files with 33,000+ game records from 2010-2024.

    Args:
        seasons: List of seasons to include (e.g., ["2022-23", "2023-24"])
                 Defaults to ["2022-23", "2023-24"] if not specified
        include_playoffs: Whether to include playoff games
        data_dir: Directory containing CSV files (optional)

    Returns:
        List of game dictionaries with features
    """
    if not HAS_KAGGLE:
        print("Error: kaggle_data_loader module not available")
        return []

    if seasons is None:
        # Default to recent seasons with enough data
        seasons = ["2023-24"]

    print(f"\n{'='*60}")
    print("KAGGLE/CSV DATA MODE (No Rate Limits!)")
    print(f"{'='*60}")
    print(f"  Seasons: {', '.join(seasons)}")
    print(f"  Include playoffs: {include_playoffs}")

    games_data = load_training_data_from_csv(
        seasons=seasons,
        include_playoffs=include_playoffs,
        window=10,  # 10-game rolling window for stats
        data_dir=data_dir,
    )

    if games_data:
        # Calculate some stats
        home_wins = sum(1 for g in games_data if g.get("home_win", False))
        avg_diff = sum(g.get("point_differential", 0) for g in games_data) / len(games_data)

        print(f"\n  Loaded {len(games_data)} training examples")
        print(f"  Home win rate: {home_wins / len(games_data):.1%}")
        print(f"  Average point differential: {avg_diff:+.1f}")
    else:
        print("\n  No games loaded. Check data directory.")

    return games_data


def train_models(
    games_data: List[Dict],
    player_data: Optional[List[Dict]] = None,
    season: str = "2025-26",
    use_ensemble: bool = True,
) -> Dict:
    """
    Train all models with the provided data.

    Args:
        games_data: Historical game data
        player_data: Historical player data
        season: Season string
        use_ensemble: Use ensemble model for maximum accuracy (default: True)

    Returns:
        Training results dictionary
    """
    print(f"\n{'='*60}")
    print("TRAINING ML MODELS")
    print(f"{'='*60}")
    print(f"Games: {len(games_data)}")
    print(f"Player records: {len(player_data) if player_data else 0}")
    print(f"Using Ensemble: {use_ensemble}")

    if len(games_data) < 20:
        print("\nWarning: Less than 20 games. Models may not perform well.")
        print("Consider fetching more historical data.")

    pipeline = ModelTrainingPipeline(season=season)

    results = pipeline.train_all_models(
        games_data=games_data,
        player_data=player_data,
        save_models=True,
        use_ensemble=use_ensemble,
    )

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models trained and saved to 'models/' directory")

    # Print summary
    print("\nModel Performance Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        if 'accuracy' in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
        if 'rmse' in metrics:
            print(f"  RMSE: {metrics['rmse']:.2f}")
        if 'r2' in metrics:
            print(f"  R2: {metrics['r2']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train NBA betting models. Use --kaggle for best results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 train_models.py --kaggle --live    # RECOMMENDED: CSV + live data
  python3 train_models.py --kaggle           # CSV data only (fast)
  python3 train_models.py --use-database     # Pre-collected database
  python3 train_models.py --fast             # Synthetic data (testing)
        """
    )
    parser.add_argument("--games", type=int, default=50, help="Number of games (API mode only, max 20)")
    parser.add_argument("--players", type=int, default=20, help="Number of players (API mode only)")
    parser.add_argument("--season", type=str, default="2025-26", help="NBA season")
    parser.add_argument("--load-games", type=str, help="Load games from JSON file")
    parser.add_argument("--load-players", type=str, help="Load players from JSON file")
    parser.add_argument("--save-data", action="store_true", help="Save fetched data to JSON")
    parser.add_argument("--skip-players", action="store_true", help="Skip player prop training")
    parser.add_argument("--fast", action="store_true", help="Use fast synthetic data (no API calls)")
    parser.add_argument("--use-database", action="store_true", help="Load data from SQLite database")
    parser.add_argument("--db-path", type=str, default="nba_betting.db", help="Database file path")
    parser.add_argument("--seasons", type=str, help="Comma-separated seasons to use (e.g., 2023-24,2024-25)")
    parser.add_argument("--kaggle", action="store_true", help="Use Kaggle/CSV data (fastest, no rate limits)")
    parser.add_argument("--live", action="store_true", help="Include live API data (2024-25, 2025-26)")
    parser.add_argument("--include-playoffs", action="store_true", help="Include playoff games in training data")
    parser.add_argument("--data-dir", type=str, help="Directory containing CSV data files")
    parser.add_argument("--unsafe-api", action="store_true",
        help="Enable direct API training (NOT RECOMMENDED - temporal leakage risk)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("NBA MODEL TRAINING SCRIPT")
    print(f"{'='*60}")
    print(f"Season: {args.season}")
    print(f"Target games: {args.games}")
    print(f"Target players: {args.players}")

    # Load or fetch data
    if args.kaggle and args.live:
        print("\n*** KAGGLE + LIVE MODE: CSV + API data (BEST) ***")
        print("This combines historical CSV + live API data for maximum coverage!\n")
        csv_seasons = args.seasons.split(",") if args.seasons else ["2023-24"]
        games_data = load_training_data_with_live(
            csv_seasons=csv_seasons,
            live_seasons=None,  # Use all available live seasons
            include_playoffs=args.include_playoffs,
        )
        player_data = None
    elif args.kaggle:
        print("\n*** KAGGLE/CSV MODE: Using real historical data ***")
        print("This loads CSV data (2010-2023) - no rate limits!\n")
        seasons = args.seasons.split(",") if args.seasons else None
        games_data = load_training_data_from_kaggle(
            seasons=seasons,
            include_playoffs=args.include_playoffs,
            data_dir=args.data_dir,
        )
        player_data = None
    elif args.use_database:
        print("\n*** DATABASE MODE: Using collected historical data ***")
        seasons = args.seasons.split(",") if args.seasons else None
        games_data = load_training_data_from_database(args.db_path, seasons)
        player_data = None
    elif args.load_games or args.load_players:
        print("\nLoading data from files...")
        games_data, player_data = load_training_data(args.load_games, args.load_players)
    elif args.fast:
        print("\n*** FAST MODE: Using synthetic data (no API calls) ***")
        print("This generates realistic training data instantly.\n")
        games_data = generate_synthetic_training_data(args.games)
        player_data = None
    else:
        # API PATH - Rate limited and slow, not recommended for production training
        # TEMPORAL DISCIPLINE: Require explicit --unsafe-api flag
        if not args.unsafe_api:
            print("\n" + "=" * 70)
            print("ERROR: Direct API training requires --unsafe-api flag")
            print("=" * 70)
            print("\nDirect API training is disabled by default because it risks")
            print("temporal leakage (using future data to predict past games).")
            print("\nRECOMMENDED approaches (point-in-time safe):")
            print("  python3 train_models.py --kaggle --live   # Best: CSV + live data")
            print("  python3 train_models.py --kaggle          # CSV data only")
            print("  python3 train_models.py --use-database    # Pre-collected data")
            print("\nIf you understand the risks and need API mode anyway:")
            print("  python3 train_models.py --unsafe-api --games 20")
            print("=" * 70 + "\n")
            sys.exit(1)

        print("\n" + "=" * 70)
        print("WARNING: Using NBA API for data fetching (SLOW - rate limited)")
        print("CAUTION: API mode may cause temporal leakage in features!")
        print("=" * 70)
        print("\nRECOMMENDED: Use --kaggle flag for faster, more reliable training:")
        print("  python3 train_models.py --kaggle --live")
        print("\nProceeding with API fetch (this will take several minutes)...")
        print("=" * 70 + "\n")

        # Limit games when using API to prevent very long waits
        actual_games = min(args.games, MAX_API_GAMES)
        if args.games > MAX_API_GAMES:
            print(f"NOTE: Limiting API fetch to {MAX_API_GAMES} games (requested: {args.games})")
            print(f"      Use --kaggle for full historical data\n")

        # Fetch historical games
        games_data = fetch_historical_games(
            num_games=actual_games,
            season=args.season,
        )

        # Fetch player data (optional)
        if not args.skip_players:
            actual_players = min(args.players, MAX_API_GAMES)
            player_data = fetch_player_data(
                num_players=actual_players,
                season=args.season,
            )
        else:
            player_data = None

    # Save data if requested
    if args.save_data and games_data:
        save_training_data(games_data, player_data or [])

    # Train models
    if games_data:
        results = train_models(
            games_data=games_data,
            player_data=player_data,
            season=args.season,
        )

        print("\n" + "="*60)
        print("DONE! Models are now trained and ready.")
        print("Run 'python3 app.py' to use trained models for predictions.")
        print("="*60)
    else:
        print("\nNo data available for training.")
        print("Try running with --kaggle (recommended), --use-database, or --fast flag.")
        sys.exit(1)


if __name__ == "__main__":
    main()
