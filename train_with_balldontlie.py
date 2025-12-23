"""
NBA Model Training with Balldontlie.io Data

Comprehensive training script that fetches real historical NBA game data
from the Balldontlie.io API and trains ML models for betting predictions.

This script:
1. Fetches historical games from multiple seasons
2. Computes point-in-time team statistics (avoiding look-ahead bias)
3. Generates features for moneyline and spread predictions
4. Trains ensemble models with proper cross-validation

Usage:
    python3 train_with_balldontlie.py --seasons 2022,2023,2024 --min-games 500
    python3 train_with_balldontlie.py --quick  # Quick test with limited data
"""

import os
import sys
import json
import time
import pickle
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd

# Load environment variables
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

# Import balldontlie API
from balldontlie_api import BalldontlieAPI

# Import model trainer
from model_trainer import ModelTrainingPipeline


class BalldontlieDataCollector:
    """
    Collects and processes NBA data from the Balldontlie.io API
    for model training purposes.
    """

    def __init__(self, api_key: Optional[str] = None, tier: str = "goat"):
        """
        Initialize the data collector.

        Args:
            api_key: Balldontlie API key (falls back to env var)
            tier: API subscription tier ("free", "allstar", "goat")
        """
        self.api = BalldontlieAPI(api_key=api_key, tier=tier)
        self.teams_cache = {}
        self.players_cache = {}
        self.games_cache = {}
        self.stats_cache = {}

        # Cache directory
        self.cache_dir = Path("data/balldontlie_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_all_teams(self) -> List[Dict]:
        """Fetch and cache all NBA teams."""
        if not self.teams_cache:
            teams = self.api.get_teams()
            for team in teams:
                self.teams_cache[team['id']] = team
                # Also index by abbreviation
                self.teams_cache[team.get('abbreviation', '')] = team
        return list(t for t in self.teams_cache.values() if isinstance(t.get('id'), int))

    def get_team_by_id(self, team_id: int) -> Optional[Dict]:
        """Get team info by ID."""
        if not self.teams_cache:
            self.get_all_teams()
        return self.teams_cache.get(team_id)

    def fetch_season_games(
        self,
        season: int,
        per_page: int = 100,
        max_pages: int = 50,
    ) -> List[Dict]:
        """
        Fetch all games for a season.

        Args:
            season: Season year (e.g., 2024 for 2024-25 season)
            per_page: Results per page
            max_pages: Maximum pages to fetch

        Returns:
            List of game dictionaries
        """
        cache_file = self.cache_dir / f"games_{season}.json"

        # Check cache first
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
                if cached.get('complete', False):
                    print(f"  Loaded {len(cached['games'])} games from cache for {season}")
                    return cached['games']

        print(f"  Fetching games for {season} season...")
        all_games = []

        # Fetch games
        games = self.api.get_games(seasons=[season], per_page=per_page)
        all_games.extend(games)
        print(f"    Fetched {len(games)} games...")

        # The API returns paginated results, need to handle pagination
        # For simplicity, we'll fetch by date ranges

        # Season typically runs Oct-Apr of next year
        start_date = datetime(season, 10, 1)
        end_date = datetime(season + 1, 6, 30)

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            day_games = self.api.get_games(dates=[date_str])

            for game in day_games:
                # Check if already added
                game_id = game.get('id')
                if game_id and not any(g.get('id') == game_id for g in all_games):
                    all_games.append(game)

            current_date += timedelta(days=1)

            # Progress update every 30 days
            if (current_date - start_date).days % 30 == 0:
                print(f"    Progress: {current_date.strftime('%Y-%m-%d')}, {len(all_games)} games...")

        # Filter for completed games only
        completed_games = [
            g for g in all_games
            if g.get('status') == 'Final' and g.get('home_team_score', 0) > 0
        ]

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump({
                'season': season,
                'games': completed_games,
                'complete': True,
                'fetched_at': datetime.now().isoformat()
            }, f)

        print(f"    Completed: {len(completed_games)} games for {season}")
        return completed_games

    def fetch_games_date_range(
        self,
        start_date: str,
        end_date: str,
    ) -> List[Dict]:
        """
        Fetch games within a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of completed game dictionaries
        """
        all_games = []

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            day_games = self.api.get_games(dates=[date_str])

            # Only include completed games
            for game in day_games:
                if game.get('status') == 'Final' and game.get('home_team_score', 0) > 0:
                    all_games.append(game)

            current += timedelta(days=1)

            # Rate limit consideration
            time.sleep(0.1)

        return all_games

    def fetch_player_stats_for_game(self, game_id: int) -> List[Dict]:
        """
        Fetch player statistics for a specific game.

        Args:
            game_id: Balldontlie game ID

        Returns:
            List of player stat dictionaries
        """
        cache_key = f"stats_{game_id}"
        if cache_key in self.stats_cache:
            return self.stats_cache[cache_key]

        stats = self.api.get_player_stats(game_ids=[game_id], per_page=100)
        self.stats_cache[cache_key] = stats
        return stats


class TeamStatsCalculator:
    """
    Calculates rolling team statistics from game-by-game data.
    Implements point-in-time calculation to avoid look-ahead bias.
    """

    def __init__(self, window: int = 10):
        """
        Initialize the calculator.

        Args:
            window: Number of recent games for rolling averages
        """
        self.window = window
        self.team_games = defaultdict(list)  # team_id -> list of (date, game_stats)

    def add_game(self, game: Dict):
        """
        Add a game to the historical record.

        Args:
            game: Game dictionary from Balldontlie API
        """
        game_date = game.get('date', '')
        if isinstance(game_date, str) and 'T' in game_date:
            game_date = game_date.split('T')[0]

        home_team_id = game.get('home_team', {}).get('id')
        away_team_id = game.get('visitor_team', {}).get('id')

        home_score = game.get('home_team_score', 0)
        away_score = game.get('visitor_team_score', 0)

        if not all([home_team_id, away_team_id, game_date, home_score]):
            return

        # Home team stats
        home_stats = {
            'date': game_date,
            'opponent_id': away_team_id,
            'is_home': True,
            'pts': home_score,
            'pts_allowed': away_score,
            'win': home_score > away_score,
            'point_diff': home_score - away_score,
        }
        self.team_games[home_team_id].append((game_date, home_stats))

        # Away team stats
        away_stats = {
            'date': game_date,
            'opponent_id': home_team_id,
            'is_home': False,
            'pts': away_score,
            'pts_allowed': home_score,
            'win': away_score > home_score,
            'point_diff': away_score - home_score,
        }
        self.team_games[away_team_id].append((game_date, away_stats))

    def get_team_stats_before_date(
        self,
        team_id: int,
        date: str,
        min_games: int = 5,
    ) -> Optional[Dict]:
        """
        Get team statistics before a specific date (point-in-time).

        Args:
            team_id: Team ID
            date: Date to calculate stats before
            min_games: Minimum games required

        Returns:
            Dictionary with team statistics or None if insufficient data
        """
        if team_id not in self.team_games:
            return None

        # Get all games before the date
        games = [(d, s) for d, s in self.team_games[team_id] if d < date]

        if len(games) < min_games:
            return None

        # Sort by date (most recent first)
        games.sort(key=lambda x: x[0], reverse=True)

        # Take recent games for rolling stats
        recent_games = games[:self.window]
        all_season_games = games

        # Calculate rolling stats
        recent_pts = [g['pts'] for _, g in recent_games]
        recent_pts_allowed = [g['pts_allowed'] for _, g in recent_games]
        recent_wins = [g['win'] for _, g in recent_games]
        recent_point_diff = [g['point_diff'] for _, g in recent_games]

        # Season stats
        season_pts = [g['pts'] for _, g in all_season_games]
        season_wins = [g['win'] for _, g in all_season_games]
        season_point_diff = [g['point_diff'] for _, g in all_season_games]

        # Home/away splits
        home_games = [(d, g) for d, g in all_season_games if g['is_home']]
        away_games = [(d, g) for d, g in all_season_games if not g['is_home']]

        home_wins = sum(1 for _, g in home_games if g['win'])
        away_wins = sum(1 for _, g in away_games if g['win'])
        home_pts = [g['pts'] for _, g in home_games]
        away_pts = [g['pts'] for _, g in away_games]

        # Calculate current streak
        streak = 0
        if recent_games:
            streak_type = recent_games[0][1]['win']
            for _, g in recent_games:
                if g['win'] == streak_type:
                    streak += 1
                else:
                    break
            if not streak_type:
                streak = -streak

        return {
            # Season stats
            'season_games': len(all_season_games),
            'season_win_pct': sum(season_wins) / len(season_wins) if season_wins else 0.5,
            'season_pts_avg': np.mean(season_pts) if season_pts else 100,
            'season_pts_allowed_avg': np.mean([g['pts_allowed'] for _, g in all_season_games]) if all_season_games else 100,
            'season_point_diff': np.mean(season_point_diff) if season_point_diff else 0,

            # Recent form (rolling)
            'recent_games': len(recent_games),
            'recent_win_pct': np.mean(recent_wins) if recent_wins else 0.5,
            'recent_pts_avg': np.mean(recent_pts) if recent_pts else 100,
            'recent_pts_allowed_avg': np.mean(recent_pts_allowed) if recent_pts_allowed else 100,
            'recent_point_diff': np.mean(recent_point_diff) if recent_point_diff else 0,

            # Home/away
            'home_games': len(home_games),
            'away_games': len(away_games),
            'home_win_pct': home_wins / len(home_games) if home_games else 0.5,
            'away_win_pct': away_wins / len(away_games) if away_games else 0.5,
            'home_pts_avg': np.mean(home_pts) if home_pts else 100,
            'away_pts_avg': np.mean(away_pts) if away_pts else 100,

            # Momentum
            'current_streak': streak,

            # Efficiency estimates (simplified)
            'off_rating': np.mean(recent_pts) * 1.0 if recent_pts else 110,  # Simplified
            'def_rating': np.mean(recent_pts_allowed) * 1.0 if recent_pts_allowed else 110,
            'net_rating': np.mean(recent_point_diff) if recent_point_diff else 0,
        }


class FeatureGenerator:
    """
    Generates training features from team statistics.
    """

    def __init__(self, stats_calculator: TeamStatsCalculator):
        self.stats_calc = stats_calculator

    def generate_moneyline_features(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: str,
    ) -> Optional[Dict]:
        """
        Generate moneyline prediction features for a matchup.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            game_date: Game date

        Returns:
            Dictionary with features or None if insufficient data
        """
        home_stats = self.stats_calc.get_team_stats_before_date(home_team_id, game_date)
        away_stats = self.stats_calc.get_team_stats_before_date(away_team_id, game_date)

        if not home_stats or not away_stats:
            return None

        features = {
            # Win percentage differentials
            'season_win_pct_diff': home_stats['season_win_pct'] - away_stats['season_win_pct'],
            'recent_win_pct_diff': home_stats['recent_win_pct'] - away_stats['recent_win_pct'],

            # Scoring differentials
            'pts_avg_diff': home_stats['season_pts_avg'] - away_stats['season_pts_avg'],
            'recent_pts_diff': home_stats['recent_pts_avg'] - away_stats['recent_pts_avg'],

            # Efficiency differentials
            'off_rating_diff': home_stats['off_rating'] - away_stats['off_rating'],
            'def_rating_diff': home_stats['def_rating'] - away_stats['def_rating'],
            'net_rating_diff': home_stats['net_rating'] - away_stats['net_rating'],

            # Form indicators
            'home_streak': home_stats['current_streak'],
            'away_streak': away_stats['current_streak'],
            'combined_form': home_stats['recent_point_diff'] - away_stats['recent_point_diff'],

            # Location-specific
            'location_win_pct_diff': home_stats['home_win_pct'] - away_stats['away_win_pct'],
            'home_advantage_factor': home_stats['home_win_pct'] - home_stats['away_win_pct'],

            # Individual team stats (for model to learn patterns)
            'home_season_win_pct': home_stats['season_win_pct'],
            'away_season_win_pct': away_stats['season_win_pct'],
            'home_recent_win_pct': home_stats['recent_win_pct'],
            'away_recent_win_pct': away_stats['recent_win_pct'],
            'home_net_rating': home_stats['net_rating'],
            'away_net_rating': away_stats['net_rating'],
            'home_off_rating': home_stats['off_rating'],
            'away_off_rating': away_stats['off_rating'],
            'home_def_rating': home_stats['def_rating'],
            'away_def_rating': away_stats['def_rating'],
            'home_pts_avg': home_stats['season_pts_avg'],
            'away_pts_avg': away_stats['season_pts_avg'],

            # Season progression (games played can indicate team has settled)
            'home_games_played': home_stats['season_games'],
            'away_games_played': away_stats['season_games'],
        }

        return features

    def generate_spread_features(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: str,
    ) -> Optional[Dict]:
        """
        Generate spread prediction features for a matchup.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            game_date: Game date

        Returns:
            Dictionary with features or None if insufficient data
        """
        # Start with moneyline features
        features = self.generate_moneyline_features(home_team_id, away_team_id, game_date)

        if not features:
            return None

        home_stats = self.stats_calc.get_team_stats_before_date(home_team_id, game_date)
        away_stats = self.stats_calc.get_team_stats_before_date(away_team_id, game_date)

        # Add spread-specific features
        spread_features = {
            # Point differential expectations
            'expected_point_diff': home_stats['home_pts_avg'] - away_stats['away_pts_avg'],
            'plus_minus_diff': home_stats['recent_point_diff'] - away_stats['recent_point_diff'],

            # Location-adjusted scoring
            'expected_home_pts': home_stats['home_pts_avg'],
            'expected_away_pts': away_stats['away_pts_avg'],

            # Home/away scoring
            'home_plus_minus': home_stats['recent_point_diff'],
            'away_plus_minus': away_stats['recent_point_diff'],

            # Defense impact
            'home_pts_allowed': home_stats['recent_pts_allowed_avg'],
            'away_pts_allowed': away_stats['recent_pts_allowed_avg'],
        }

        features.update(spread_features)
        return features


def process_games_to_training_data(
    games: List[Dict],
    min_games_for_stats: int = 10,
    window: int = 10,
) -> List[Dict]:
    """
    Process raw games into training data with features.

    Args:
        games: List of game dictionaries from API
        min_games_for_stats: Minimum games needed for stats calculation
        window: Rolling window for recent stats

    Returns:
        List of training examples with features and outcomes
    """
    print(f"\nProcessing {len(games)} games into training data...")

    # Sort games by date
    games_sorted = sorted(games, key=lambda g: g.get('date', ''))

    # Initialize calculators
    stats_calc = TeamStatsCalculator(window=window)
    feature_gen = FeatureGenerator(stats_calc)

    training_data = []
    skipped_insufficient = 0

    for i, game in enumerate(games_sorted):
        game_date = game.get('date', '')
        if isinstance(game_date, str) and 'T' in game_date:
            game_date = game_date.split('T')[0]

        home_team = game.get('home_team', {})
        away_team = game.get('visitor_team', {})

        home_team_id = home_team.get('id')
        away_team_id = away_team.get('id')

        home_score = game.get('home_team_score', 0)
        away_score = game.get('visitor_team_score', 0)

        if not all([home_team_id, away_team_id, game_date, home_score]):
            continue

        # Generate features BEFORE adding this game to stats
        # (this ensures point-in-time correctness)
        ml_features = feature_gen.generate_moneyline_features(
            home_team_id, away_team_id, game_date
        )
        spread_features = feature_gen.generate_spread_features(
            home_team_id, away_team_id, game_date
        )

        # Now add the game to stats calculator for future games
        stats_calc.add_game(game)

        # Skip if insufficient historical data
        if not ml_features or not spread_features:
            skipped_insufficient += 1
            continue

        # Create training example
        training_example = {
            'game_id': game.get('id'),
            'game_date': game_date,
            'home_team': home_team.get('abbreviation', ''),
            'away_team': away_team.get('abbreviation', ''),
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,

            # Outcomes
            'home_win': home_score > away_score,
            'home_score': home_score,
            'away_score': away_score,
            'point_differential': home_score - away_score,

            # Features
            'moneyline_features': ml_features,
            'spread_features': spread_features,
        }

        training_data.append(training_example)

        # Progress update
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(games_sorted)} games, {len(training_data)} valid examples...")

    print(f"  Total valid training examples: {len(training_data)}")
    print(f"  Skipped (insufficient history): {skipped_insufficient}")

    return training_data


def fetch_and_prepare_training_data(
    seasons: List[int],
    api_key: Optional[str] = None,
    min_games: int = 100,
    cache_file: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch data from Balldontlie API and prepare for training.

    Args:
        seasons: List of season years to fetch
        api_key: API key (falls back to env var)
        min_games: Minimum total games required
        cache_file: Optional cache file path

    Returns:
        List of training examples
    """
    # Check cache first
    if cache_file and Path(cache_file).exists():
        print(f"Loading cached training data from {cache_file}")
        with open(cache_file) as f:
            cached = json.load(f)
            if len(cached) >= min_games:
                print(f"  Loaded {len(cached)} training examples from cache")
                return cached
            print(f"  Cache has only {len(cached)} examples, fetching more...")

    print("="*60)
    print("BALLDONTLIE.IO DATA COLLECTION")
    print("="*60)
    print(f"Seasons to fetch: {seasons}")

    collector = BalldontlieDataCollector(api_key=api_key)

    # First, get all teams
    print("\nFetching team data...")
    teams = collector.get_all_teams()
    print(f"  Found {len(teams)} NBA teams")

    # Fetch games for each season
    all_games = []
    for season in seasons:
        print(f"\nSeason {season}-{str(season+1)[-2:]}:")
        season_games = collector.fetch_season_games(season)
        all_games.extend(season_games)

    print(f"\nTotal games collected: {len(all_games)}")

    # Process into training data
    training_data = process_games_to_training_data(
        all_games,
        min_games_for_stats=10,
        window=10,
    )

    # Save to cache
    if cache_file:
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        print(f"\nSaved training data to {cache_file}")

    return training_data


def train_models_with_data(
    training_data: List[Dict],
    season: str = "2025-26",
    use_ensemble: bool = True,
) -> Dict:
    """
    Train all models with the prepared training data.

    Args:
        training_data: List of training examples
        season: Season string for model naming
        use_ensemble: Use ensemble model for maximum accuracy

    Returns:
        Dictionary with training results
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    print(f"Training examples: {len(training_data)}")
    print(f"Using ensemble: {use_ensemble}")

    # Analyze data
    home_wins = sum(1 for t in training_data if t.get('home_win', False))
    avg_diff = np.mean([t.get('point_differential', 0) for t in training_data])

    print(f"\nData Statistics:")
    print(f"  Home win rate: {home_wins / len(training_data):.1%}")
    print(f"  Average point differential: {avg_diff:+.1f}")

    # Initialize pipeline
    pipeline = ModelTrainingPipeline(season=season)

    # Train all models
    results = pipeline.train_all_models(
        games_data=training_data,
        player_data=None,  # No player props for now
        save_models=True,
        use_ensemble=use_ensemble,
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train NBA betting models with Balldontlie.io data"
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="2023,2024,2025",
        help="Comma-separated season years (e.g., 2023,2024,2025)"
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=500,
        help="Minimum number of training games"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use cached data if available, smaller dataset"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cache and fetch fresh data"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Balldontlie API key (falls back to BALLDONTLIE_API_KEY env var)"
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble model (use single XGBoost instead)"
    )

    args = parser.parse_args()

    # Parse seasons
    seasons = [int(s.strip()) for s in args.seasons.split(",")]

    if args.quick:
        seasons = [2025]  # Just current season for quick test
        args.min_games = 100

    print("="*60)
    print("NBA MODEL TRAINING WITH BALLDONTLIE.IO DATA")
    print("="*60)
    print(f"Seasons: {seasons}")
    print(f"Minimum games: {args.min_games}")
    print(f"Quick mode: {args.quick}")
    print(f"Use cache: {not args.no_cache}")
    print(f"Ensemble: {not args.no_ensemble}")

    # Cache file path
    cache_file = None
    if not args.no_cache:
        cache_file = f"data/balldontlie_cache/training_data_{'_'.join(map(str, seasons))}.json"

    # Fetch and prepare data
    training_data = fetch_and_prepare_training_data(
        seasons=seasons,
        api_key=args.api_key,
        min_games=args.min_games,
        cache_file=cache_file,
    )

    if len(training_data) < 20:
        print("\nError: Insufficient training data!")
        print("Try adding more seasons or reducing --min-games requirement.")
        sys.exit(1)

    # Train models
    results = train_models_with_data(
        training_data=training_data,
        season="2025-26",
        use_ensemble=not args.no_ensemble,
    )

    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nModel Performance Summary:")

    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        if 'accuracy' in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
        if 'rmse' in metrics:
            print(f"  RMSE: {metrics['rmse']:.2f} points")
        if 'r2' in metrics:
            print(f"  R2: {metrics['r2']:.4f}")
        if 'f1' in metrics:
            print(f"  F1: {metrics['f1']:.4f}")

    print("\n" + "="*60)
    print("Models saved to 'models/' directory")
    print("Run 'python3 app.py' to use trained models for predictions")
    print("="*60)


if __name__ == "__main__":
    main()
