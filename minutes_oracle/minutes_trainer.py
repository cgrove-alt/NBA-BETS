"""
Minutes Oracle Training Script

Trains the quantile regression model for minutes prediction using
historical game data from the Balldontlie API.

Usage:
    python -m minutes_oracle.minutes_trainer

    # Or with custom options:
    python -m minutes_oracle.minutes_trainer --seasons 2023 2024 2025 --output models/minutes_oracle.pkl
"""

from __future__ import annotations

import sys
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from minutes_oracle.minutes_predictor import MinutesPredictor
from minutes_oracle.minutes_features import (
    MinutesFeatureGenerator,
    MINUTES_FEATURE_NAMES,
    normalize_position,
)
from minutes_oracle.coach_tendencies import (
    CoachTendencyLearner,
    get_coach_tendency,
    COACH_TENDENCIES,
    TEAM_ID_TO_ABBREV,
)


# =============================================================================
# TRAINING DATA EXTRACTION
# =============================================================================

class MinutesTrainingDataExtractor:
    """
    Extracts training data for the Minutes Oracle from Balldontlie game data.

    Uses the same data pipeline as train_complete_balldontlie.py but focuses
    on minutes prediction features.
    """

    def __init__(self, min_games_history: int = 5, min_minutes_threshold: int = 10):
        """
        Initialize the extractor.

        Args:
            min_games_history: Minimum games of history required for a player
            min_minutes_threshold: Minimum minutes to include a game (filters DNP/garbage time)
        """
        self.min_games_history = min_games_history
        self.min_minutes_threshold = min_minutes_threshold

        # Player history tracking
        self.player_game_logs: dict[int, list[dict]] = defaultdict(list)
        self.player_info: dict[int, dict] = {}

        # Team tracking
        self.team_rosters: dict[int, list[dict]] = defaultdict(list)
        self.team_schedules: dict[int, list[dict]] = defaultdict(list)

        # Coach tendency learner
        self.coach_learner = CoachTendencyLearner()

        # Feature generator
        self.feature_gen = MinutesFeatureGenerator()

    def process_games(self,
                      games: list[dict],
                      player_stats_by_game: dict[int, list[dict]],
                      vegas_data: dict[int, dict] | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Process games into training data.

        Args:
            games: List of game dicts from Balldontlie API
            player_stats_by_game: Dict mapping game_id -> list of player stat dicts
            vegas_data: Optional dict mapping game_id -> vegas data (spread, total)

        Returns:
            Tuple of (features_df, targets, sample_weights)
        """
        print(f"Processing {len(games)} games for minutes training...")

        # Sort games chronologically
        games_sorted = sorted(games, key=lambda g: g.get('date', ''))

        training_examples = []
        skipped_no_history = 0
        skipped_low_minutes = 0
        skipped_no_stats = 0

        for i, game in enumerate(games_sorted):
            if i % 500 == 0:
                print(f"  Processing game {i+1}/{len(games_sorted)}...")

            game_id = game.get('id')
            game_date = game.get('date', '')
            if isinstance(game_date, str) and 'T' in game_date:
                game_date = game_date.split('T')[0]

            home_team = game.get('home_team', {})
            away_team = game.get('visitor_team', {})
            home_team_id = home_team.get('id')
            away_team_id = away_team.get('id')
            home_score = game.get('home_team_score', 0)
            away_score = game.get('visitor_team_score', 0)

            if not all([home_team_id, away_team_id, game_date]):
                continue

            # Get Vegas data if available
            vegas = vegas_data.get(game_id, {}) if vegas_data else {}
            vegas_spread = vegas.get('spread', 0.0)
            vegas_total = vegas.get('total', 220.0)

            # Calculate game context
            final_margin = home_score - away_score
            went_to_overtime = home_score > 0 and away_score > 0 and (
                max(home_score, away_score) > 130 or  # High score suggests OT
                abs(final_margin) <= 3  # Close game might have gone to OT
            )

            # Get player stats for this game
            game_player_stats = (
                player_stats_by_game.get(game_id, []) or
                player_stats_by_game.get(str(game_id), [])
            )

            if not game_player_stats:
                skipped_no_stats += 1
                continue

            # Track for coach tendency learning
            home_players = []
            away_players = []

            for ps in game_player_stats:
                player_id = ps.get('player', {}).get('id')
                if not player_id:
                    continue

                # Get actual minutes
                actual_min = self._parse_minutes(ps.get('min', '0'))

                # Determine player's team
                player_team = ps.get('team', {})
                player_team_id = player_team.get('id')
                is_home = player_team_id == home_team_id
                opponent_team_id = away_team_id if is_home else home_team_id

                # Track for coach learning
                player_entry = {
                    'player_id': player_id,
                    'minutes': actual_min,
                    'is_starter': actual_min >= 25,  # Heuristic: starters typically 25+ min
                }
                if is_home:
                    home_players.append(player_entry)
                else:
                    away_players.append(player_entry)

                # Skip if below minimum minutes threshold
                if actual_min < self.min_minutes_threshold:
                    skipped_low_minutes += 1
                    # Still add to history for future predictions
                    self._add_to_player_history(player_id, ps, game_date)
                    continue

                # Check if player has enough history
                history_count = len(self.player_game_logs[player_id])
                if history_count < self.min_games_history:
                    skipped_no_history += 1
                    self._add_to_player_history(player_id, ps, game_date)
                    continue

                # Calculate rest days
                days_rest = self._calculate_days_rest(player_team_id, game_date)
                is_b2b = days_rest == 1

                # Build game context
                game_context = {
                    'vegas_spread': vegas_spread if is_home else -vegas_spread,
                    'vegas_total': vegas_total,
                    'is_home': is_home,
                    'is_back_to_back': is_b2b,
                    'days_rest': days_rest,
                }

                # Generate features
                features = self.feature_gen.generate_features(
                    player_id=player_id,
                    team_id=player_team_id,
                    opponent_team_id=opponent_team_id,
                    game_date=game_date,
                    game_context=game_context,
                    player_game_logs=list(self.player_game_logs[player_id]),
                    team_roster=list(self.team_rosters.get(player_team_id, [])),
                    injured_players=[],  # Historical training - injury data not always available
                )

                # Calculate sample weight
                sample_weight = self._calculate_sample_weight(
                    game_date=game_date,
                    final_margin=abs(final_margin),
                    went_to_overtime=went_to_overtime
                )

                training_examples.append({
                    'features': features,
                    'target': actual_min,
                    'weight': sample_weight,
                    'player_id': player_id,
                    'game_id': game_id,
                    'game_date': game_date,
                })

                # Add to player history AFTER generating features (point-in-time)
                self._add_to_player_history(player_id, ps, game_date)

            # Update team schedules
            self._update_team_schedule(home_team_id, game)
            self._update_team_schedule(away_team_id, game)

            # Learn coach tendencies
            home_coach = self._get_coach_name(home_team_id)
            away_coach = self._get_coach_name(away_team_id)

            if home_coach and home_players:
                self.coach_learner.add_game(
                    coach_name=home_coach,
                    team_id=home_team_id,
                    player_minutes=home_players,
                    final_margin=final_margin,
                    is_back_to_back=self._calculate_days_rest(home_team_id, game_date) == 1,
                    went_to_overtime=went_to_overtime
                )

            if away_coach and away_players:
                self.coach_learner.add_game(
                    coach_name=away_coach,
                    team_id=away_team_id,
                    player_minutes=away_players,
                    final_margin=-final_margin,
                    is_back_to_back=self._calculate_days_rest(away_team_id, game_date) == 1,
                    went_to_overtime=went_to_overtime
                )

        print("\nExtraction complete:")
        print(f"  Training examples: {len(training_examples)}")
        print(f"  Skipped (no history): {skipped_no_history}")
        print(f"  Skipped (low minutes): {skipped_low_minutes}")
        print(f"  Skipped (no stats): {skipped_no_stats}")

        if not training_examples:
            raise ValueError("No training examples extracted!")

        # Convert to arrays
        feature_dicts = [ex['features'] for ex in training_examples]
        targets = np.array([ex['target'] for ex in training_examples])
        weights = np.array([ex['weight'] for ex in training_examples])

        # Create DataFrame
        features_df = pd.DataFrame(feature_dicts)

        # Ensure all expected columns exist
        for col in MINUTES_FEATURE_NAMES:
            if col not in features_df.columns:
                features_df[col] = 0.0

        # Reorder columns
        features_df = features_df[MINUTES_FEATURE_NAMES]

        return features_df, targets, weights

    def _parse_minutes(self, value: Any) -> float:
        """Parse minutes from various formats."""
        if value is None:
            return 0.0

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            value = value.strip()
            if not value or value == '--':
                return 0.0

            if ':' in value:
                try:
                    parts = value.split(':')
                    mins = int(parts[0])
                    secs = int(parts[1]) if len(parts) > 1 else 0
                    return mins + secs / 60.0
                except (ValueError, IndexError):
                    return 0.0

            try:
                return float(value)
            except ValueError:
                return 0.0

        return 0.0

    def _add_to_player_history(self, player_id: int, game_stats: dict, game_date: str):
        """Add a game to player's history."""
        log_entry = {
            'min': game_stats.get('min'),
            'pts': game_stats.get('pts', 0),
            'reb': game_stats.get('reb', 0),
            'ast': game_stats.get('ast', 0),
            'pf': game_stats.get('pf', 0),
            'date': game_date,
            'game': {'date': game_date},
        }
        self.player_game_logs[player_id].append(log_entry)

        # Store player info
        player_info = game_stats.get('player', {})
        if player_info and player_id not in self.player_info:
            self.player_info[player_id] = {
                'id': player_id,
                'name': f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}".strip(),
                'position': player_info.get('position', 'F'),
            }

    def _update_team_schedule(self, team_id: int, game: dict):
        """Update team's schedule history."""
        self.team_schedules[team_id].append({
            'date': game.get('date', ''),
            'game_id': game.get('id'),
        })

    def _calculate_days_rest(self, team_id: int, game_date: str) -> int:
        """Calculate days since last game for a team."""
        schedule = self.team_schedules.get(team_id, [])
        if not schedule:
            return 2  # Default

        try:
            current_date = datetime.strptime(game_date[:10], '%Y-%m-%d')
        except (ValueError, TypeError):
            return 2

        last_game = None
        for game in reversed(schedule):
            game_dt_str = game.get('date', '')
            if isinstance(game_dt_str, str) and game_dt_str:
                try:
                    game_dt = datetime.strptime(game_dt_str[:10], '%Y-%m-%d')
                    if game_dt < current_date:
                        last_game = game_dt
                        break
                except (ValueError, TypeError):
                    continue

        if last_game:
            return (current_date - last_game).days
        return 2

    def _get_coach_name(self, team_id: int) -> str | None:
        """Get coach name for a team."""
        from minutes_oracle.coach_tendencies import COACH_BY_TEAM_ID
        coach = COACH_BY_TEAM_ID.get(team_id)
        return coach.name if coach else None

    def _calculate_sample_weight(self,
                                  game_date: str,
                                  final_margin: float,
                                  went_to_overtime: bool) -> float:
        """
        Calculate sample weight for a training example.

        Applies:
        - Time decay (more recent games weighted higher)
        - Blowout down-weighting
        - OT game down-weighting
        """
        weight = 1.0

        # Time decay (half-life of 6 months)
        try:
            game_dt = datetime.strptime(game_date[:10], '%Y-%m-%d')
            days_ago = (datetime.now() - game_dt).days
            weight *= 0.5 ** (days_ago / 180)
        except (ValueError, TypeError):
            pass

        # Blowout down-weighting
        if final_margin >= 25:
            weight *= 0.5
        elif final_margin >= 20:
            weight *= 0.7

        # OT down-weighting (unusual minutes distribution)
        if went_to_overtime:
            weight *= 0.7

        return max(0.1, weight)  # Minimum weight

    def update_coach_tendencies(self, min_games: int = 20):
        """Update global coach tendencies from learned data."""
        updated = self.coach_learner.update_global_tendencies(min_games)
        print(f"Updated {updated} coach tendencies from historical data")
        return updated


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_minutes_oracle(
    seasons: list[int] = None,
    output_path: str = 'models/minutes_oracle.pkl',
    validation_split: float = 0.2,
    min_games_history: int = 5,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Train the Minutes Oracle model.

    Args:
        seasons: List of seasons to train on (e.g., [2023, 2024, 2025])
        output_path: Path to save the trained model
        validation_split: Fraction of data for validation
        min_games_history: Minimum games required for player history
        verbose: Print progress

    Returns:
        Dictionary of training metrics
    """
    # Import data collection from main training script
    try:
        from train_complete_balldontlie import ComprehensiveDataCollector
    except ImportError as e:
        print(f"Error: Could not import from train_complete_balldontlie.py: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)

    # Default seasons
    if seasons is None:
        seasons = [2023, 2024, 2025]  # 2023-24, 2024-25, 2025-26

    print("=" * 60)
    print("MINUTES ORACLE TRAINING")
    print("=" * 60)
    print(f"Seasons: {seasons}")
    print(f"Output: {output_path}")
    print(f"Min games history: {min_games_history}")
    print()

    # Step 1: Collect game data
    print("Step 1: Collecting game data...")
    collector = ComprehensiveDataCollector()

    all_games = []
    for season in seasons:
        print(f"  Season {season}-{str(season+1)[-2:]}...")
        games = collector.fetch_season_games(season)
        all_games.extend(games)
        print(f"    Found {len(games)} games")

    print(f"  Total games: {len(all_games)}")

    # Step 2: Fetch player stats
    print("\nStep 2: Fetching player statistics...")
    game_ids = [g.get('id') for g in all_games if g.get('id')]
    player_stats = collector.fetch_player_stats_for_games(game_ids)
    print(f"  Player stats for {len(player_stats)} games")

    # Step 3: Extract training data
    print("\nStep 3: Extracting training features...")
    extractor = MinutesTrainingDataExtractor(min_games_history=min_games_history)
    features_df, targets, weights = extractor.process_games(all_games, player_stats)

    print(f"\nTraining data shape: {features_df.shape}")
    print(f"Target range: {targets.min():.1f} - {targets.max():.1f} minutes")
    print(f"Target mean: {targets.mean():.1f} minutes")

    # Step 4: Update coach tendencies from data
    print("\nStep 4: Learning coach tendencies...")
    extractor.update_coach_tendencies(min_games=20)

    # Step 5: Train the model
    print("\nStep 5: Training quantile regression models...")
    predictor = MinutesPredictor()
    metrics = predictor.train(
        X=features_df,
        y=targets,
        sample_weights=weights,
        validation_split=validation_split,
        verbose=verbose
    )

    # Step 6: Save the model
    print(f"\nStep 6: Saving model to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictor.save(output_path)

    # Step 7: Print feature importance
    print("\nTop 15 Most Important Features:")
    for name, importance in predictor.get_feature_importance(top_n=15):
        print(f"  {name}: {importance:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: {output_path}")
    print(f"Training samples: {predictor.training_samples}")
    print("\nValidation Metrics:")
    print(f"  Median RMSE: {metrics['median_rmse']:.2f} minutes")
    print(f"  Median MAE: {metrics['median_mae']:.2f} minutes")
    print(f"  P10-P90 Coverage: {metrics['p10_p90_coverage']:.1%}")
    print(f"  P50 Calibration: {metrics['p50_coverage']:.1%} (target: 50%)")

    return metrics


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_minutes_oracle(
    model_path: str = 'models/minutes_oracle.pkl',
    test_seasons: list[int] = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Validate the trained Minutes Oracle on test data.

    Args:
        model_path: Path to trained model
        test_seasons: Seasons to validate on (defaults to current season)
        verbose: Print detailed results

    Returns:
        Dictionary of validation metrics
    """
    from train_complete_balldontlie import ComprehensiveDataCollector

    # Load model
    print("Loading model...")
    predictor = MinutesPredictor.load(model_path)

    # Default to current season for validation
    if test_seasons is None:
        test_seasons = [2025]

    # Collect test data
    print(f"Collecting test data for seasons {test_seasons}...")
    collector = ComprehensiveDataCollector()

    all_games = []
    for season in test_seasons:
        games = collector.fetch_season_games(season)
        all_games.extend(games)

    # Sort and take most recent 20%
    all_games = sorted(all_games, key=lambda g: g.get('date', ''))
    test_games = all_games[int(len(all_games) * 0.8):]

    print(f"Test games: {len(test_games)}")

    # Extract test features
    extractor = MinutesTrainingDataExtractor()
    game_ids = [g.get('id') for g in test_games if g.get('id')]
    player_stats = collector.fetch_player_stats_for_games(game_ids)

    # Process first to build history
    for game in sorted(all_games[:int(len(all_games) * 0.8)], key=lambda g: g.get('date', '')):
        game_id = game.get('id')
        game_date = game.get('date', '')[:10]
        for ps in player_stats.get(game_id, []) or player_stats.get(str(game_id), []):
            player_id = ps.get('player', {}).get('id')
            if player_id:
                extractor._add_to_player_history(player_id, ps, game_date)
                extractor._update_team_schedule(
                    ps.get('team', {}).get('id'),
                    game
                )

    features_df, targets, _ = extractor.process_games(test_games, player_stats)

    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict_batch(features_df)

    # Calculate metrics
    p50_preds = np.array([p.p50 for p in predictions])
    p10_preds = np.array([p.p10 for p in predictions])
    p90_preds = np.array([p.p90 for p in predictions])

    metrics = {
        'median_rmse': np.sqrt(np.mean((targets - p50_preds) ** 2)),
        'median_mae': np.mean(np.abs(targets - p50_preds)),
        'p50_calibration': np.mean(targets <= p50_preds),
        'p10_calibration': np.mean(targets <= p10_preds),
        'p90_calibration': np.mean(targets <= p90_preds),
        'p10_p90_coverage': np.mean((targets >= p10_preds) & (targets <= p90_preds)),
        'n_samples': len(targets),
    }

    # Calculate baseline (using season average as prediction)
    if 'season_min_avg' in features_df.columns:
        baseline_preds = features_df['season_min_avg'].values
        metrics['baseline_rmse'] = np.sqrt(np.mean((targets - baseline_preds) ** 2))
        metrics['rmse_improvement'] = (metrics['baseline_rmse'] - metrics['median_rmse']) / metrics['baseline_rmse']

    # Calculate metrics by spread bucket
    if 'vegas_spread_abs' in features_df.columns:
        spreads = features_df['vegas_spread_abs'].values
        for bucket_name, (low, high) in [('close', (0, 5)), ('medium', (5, 10)), ('blowout', (10, 100))]:
            mask = (spreads >= low) & (spreads < high)
            if mask.sum() > 10:
                bucket_rmse = np.sqrt(np.mean((targets[mask] - p50_preds[mask]) ** 2))
                metrics[f'rmse_{bucket_name}'] = bucket_rmse

    if verbose:
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(f"\nSamples: {metrics['n_samples']}")
        print("\nMedian Prediction:")
        print(f"  RMSE: {metrics['median_rmse']:.2f} minutes")
        print(f"  MAE: {metrics['median_mae']:.2f} minutes")
        if 'baseline_rmse' in metrics:
            print(f"  Baseline RMSE: {metrics['baseline_rmse']:.2f} minutes")
            print(f"  Improvement: {metrics['rmse_improvement']:.1%}")
        print("\nCalibration:")
        print(f"  P10: {metrics['p10_calibration']:.1%} (target: 10%)")
        print(f"  P50: {metrics['p50_calibration']:.1%} (target: 50%)")
        print(f"  P90: {metrics['p90_calibration']:.1%} (target: 90%)")
        print("\nCoverage:")
        print(f"  P10-P90: {metrics['p10_p90_coverage']:.1%} (target: 80%)")
        if 'rmse_close' in metrics:
            print("\nBy Game Type:")
            print(f"  Close (spread < 5): {metrics.get('rmse_close', 0):.2f} RMSE")
            print(f"  Medium (5-10): {metrics.get('rmse_medium', 0):.2f} RMSE")
            print(f"  Blowout (10+): {metrics.get('rmse_blowout', 0):.2f} RMSE")

    return metrics


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train the Minutes Oracle model')
    parser.add_argument('--seasons', type=int, nargs='+', default=[2023, 2024, 2025],
                        help='Seasons to train on (e.g., 2023 2024 2025)')
    parser.add_argument('--output', type=str, default='models/minutes_oracle.pkl',
                        help='Output path for trained model')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Fraction of data for validation')
    parser.add_argument('--min-games', type=int, default=5,
                        help='Minimum games of history required')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing model, skip training')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    if args.validate_only:
        validate_minutes_oracle(
            model_path=args.output,
            test_seasons=[max(args.seasons)],
            verbose=not args.quiet
        )
    else:
        train_minutes_oracle(
            seasons=args.seasons,
            output_path=args.output,
            validation_split=args.validation_split,
            min_games_history=args.min_games,
            verbose=not args.quiet
        )


if __name__ == '__main__':
    main()
