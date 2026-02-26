#!/usr/bin/env python3
"""
Comprehensive validation of the Minutes Oracle model.

Uses the same training data but with proper temporal split to get
accurate metrics by game type (close/medium/blowout).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from minutes_oracle.minutes_predictor import MinutesPredictor
from minutes_oracle.minutes_features import MINUTES_FEATURE_NAMES


def run_comprehensive_validation(model_path: str = 'models/minutes_oracle.pkl'):
    """Run validation with detailed metrics by game type."""

    print("=" * 60)
    print("MINUTES ORACLE COMPREHENSIVE VALIDATION")
    print("=" * 60)

    # Load the model
    print("\nLoading model...")
    predictor = MinutesPredictor.load(model_path)
    print(f"  Interval scale factor: {predictor.interval_scale}")

    # Re-extract training data to get validation set
    print("\nLoading training data...")
    try:
        from train_complete_balldontlie import ComprehensiveDataCollector
    except ImportError as e:
        print(f"Error importing: {e}")
        return None

    from minutes_oracle.minutes_trainer import MinutesTrainingDataExtractor

    # Collect data
    collector = ComprehensiveDataCollector()
    seasons = [2023, 2024, 2025]

    all_games = []
    for season in seasons:
        print(f"  Loading season {season}...")
        games = collector.fetch_season_games(season)
        all_games.extend(games)

    print(f"  Total games: {len(all_games)}")

    # Fetch player stats
    game_ids = [g.get('id') for g in all_games if g.get('id')]
    player_stats = collector.fetch_player_stats_for_games(game_ids)

    # Extract features
    print("\nExtracting features...")
    extractor = MinutesTrainingDataExtractor(min_games_history=5)
    features_df, targets, weights = extractor.process_games(all_games, player_stats)

    # Use last 20% as validation (temporal split)
    n_samples = len(features_df)
    split_idx = int(n_samples * 0.8)

    X_val = features_df.iloc[split_idx:]
    y_val = targets[split_idx:]

    print(f"\nValidation set: {len(X_val)} samples")

    # Get predictions
    print("Making predictions...")
    predictions = predictor.predict_batch(X_val)

    # Extract prediction arrays
    p10 = np.array([p.p10 for p in predictions])
    p25 = np.array([p.p25 for p in predictions])
    p50 = np.array([p.p50 for p in predictions])
    p75 = np.array([p.p75 for p in predictions])
    p90 = np.array([p.p90 for p in predictions])

    # Overall metrics
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    rmse = np.sqrt(np.mean((y_val - p50) ** 2))
    mae = np.mean(np.abs(y_val - p50))

    # Baseline (season average)
    baseline = X_val['season_min_avg'].values
    baseline_rmse = np.sqrt(np.mean((y_val - baseline) ** 2))
    improvement = (baseline_rmse - rmse) / baseline_rmse * 100

    print(f"\nOverall Performance ({len(y_val)} samples):")
    print(f"  Median RMSE: {rmse:.2f} min")
    print(f"  Median MAE: {mae:.2f} min")
    print(f"  Baseline RMSE: {baseline_rmse:.2f} min")
    print(f"  Improvement: {improvement:+.1f}%")

    # Calibration
    print("\nCalibration (% of actuals <= predicted quantile):")
    print(f"  P10: {np.mean(y_val <= p10)*100:.1f}% (target: 10%)")
    print(f"  P25: {np.mean(y_val <= p25)*100:.1f}% (target: 25%)")
    print(f"  P50: {np.mean(y_val <= p50)*100:.1f}% (target: 50%)")
    print(f"  P75: {np.mean(y_val <= p75)*100:.1f}% (target: 75%)")
    print(f"  P90: {np.mean(y_val <= p90)*100:.1f}% (target: 90%)")

    # Coverage
    p10_p90_cov = np.mean((y_val >= p10) & (y_val <= p90))
    p25_p75_cov = np.mean((y_val >= p25) & (y_val <= p75))
    print("\nInterval Coverage:")
    print(f"  P10-P90: {p10_p90_cov*100:.1f}% (target: 80%)")
    print(f"  P25-P75: {p25_p75_cov*100:.1f}% (target: 50%)")

    # By game type (using vegas_spread_abs)
    spreads = X_val['vegas_spread_abs'].values

    close_mask = spreads < 5
    medium_mask = (spreads >= 5) & (spreads < 10)
    blowout_mask = spreads >= 10

    print("\nBy Game Type (Vegas Spread):")

    if close_mask.sum() > 10:
        close_rmse = np.sqrt(np.mean((y_val[close_mask] - p50[close_mask]) ** 2))
        close_cov = np.mean((y_val[close_mask] >= p10[close_mask]) & (y_val[close_mask] <= p90[close_mask]))
        print(f"  Close games (spread < 5): {close_rmse:.2f} RMSE, {close_cov*100:.1f}% coverage ({close_mask.sum()} samples)")
    else:
        print(f"  Close games (spread < 5): Insufficient samples ({close_mask.sum()})")

    if medium_mask.sum() > 10:
        medium_rmse = np.sqrt(np.mean((y_val[medium_mask] - p50[medium_mask]) ** 2))
        medium_cov = np.mean((y_val[medium_mask] >= p10[medium_mask]) & (y_val[medium_mask] <= p90[medium_mask]))
        print(f"  Medium (spread 5-10): {medium_rmse:.2f} RMSE, {medium_cov*100:.1f}% coverage ({medium_mask.sum()} samples)")
    else:
        print(f"  Medium (spread 5-10): Insufficient samples ({medium_mask.sum()})")

    if blowout_mask.sum() > 10:
        blowout_rmse = np.sqrt(np.mean((y_val[blowout_mask] - p50[blowout_mask]) ** 2))
        blowout_cov = np.mean((y_val[blowout_mask] >= p10[blowout_mask]) & (y_val[blowout_mask] <= p90[blowout_mask]))
        print(f"  Blowout (spread > 10): {blowout_rmse:.2f} RMSE, {blowout_cov*100:.1f}% coverage ({blowout_mask.sum()} samples)")
    else:
        print(f"  Blowout (spread > 10): Insufficient samples ({blowout_mask.sum()})")

    # By player type (based on actual minutes)
    print("\nBy Player Type (Actual Minutes):")

    starter_mask = y_val >= 30
    rotation_mask = (y_val >= 20) & (y_val < 30)
    bench_mask = (y_val >= 10) & (y_val < 20)

    if starter_mask.sum() > 10:
        starter_rmse = np.sqrt(np.mean((y_val[starter_mask] - p50[starter_mask]) ** 2))
        starter_cov = np.mean((y_val[starter_mask] >= p10[starter_mask]) & (y_val[starter_mask] <= p90[starter_mask]))
        print(f"  Starters (30+ min): {starter_rmse:.2f} RMSE, {starter_cov*100:.1f}% coverage ({starter_mask.sum()} samples)")

    if rotation_mask.sum() > 10:
        rotation_rmse = np.sqrt(np.mean((y_val[rotation_mask] - p50[rotation_mask]) ** 2))
        rotation_cov = np.mean((y_val[rotation_mask] >= p10[rotation_mask]) & (y_val[rotation_mask] <= p90[rotation_mask]))
        print(f"  Rotation (20-30 min): {rotation_rmse:.2f} RMSE, {rotation_cov*100:.1f}% coverage ({rotation_mask.sum()} samples)")

    if bench_mask.sum() > 10:
        bench_rmse = np.sqrt(np.mean((y_val[bench_mask] - p50[bench_mask]) ** 2))
        bench_cov = np.mean((y_val[bench_mask] >= p10[bench_mask]) & (y_val[bench_mask] <= p90[bench_mask]))
        print(f"  Bench (10-20 min): {bench_rmse:.2f} RMSE, {bench_cov*100:.1f}% coverage ({bench_mask.sum()} samples)")

    # By uncertainty level
    print("\nBy Predicted Uncertainty:")
    uncertainties = [p.uncertainty for p in predictions]
    for level in ['low', 'medium', 'high']:
        mask = np.array([u == level for u in uncertainties])
        if mask.sum() > 10:
            level_rmse = np.sqrt(np.mean((y_val[mask] - p50[mask]) ** 2))
            level_cov = np.mean((y_val[mask] >= p10[mask]) & (y_val[mask] <= p90[mask]))
            print(f"  {level.capitalize()}: {level_rmse:.2f} RMSE, {level_cov*100:.1f}% coverage ({mask.sum()} samples)")

    # Prediction interval width
    avg_spread = np.mean(p90 - p10)
    print("\nPrediction Interval Width:")
    print(f"  Average P10-P90 spread: {avg_spread:.1f} min")

    print("\n" + "=" * 60)

    # Return metrics for further analysis
    return {
        'rmse': rmse,
        'mae': mae,
        'baseline_rmse': baseline_rmse,
        'improvement': improvement,
        'p10_p90_coverage': p10_p90_cov,
        'p50_calibration': np.mean(y_val <= p50),
        'n_samples': len(y_val),
    }


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/minutes_oracle.pkl'
    run_comprehensive_validation(model_path)
