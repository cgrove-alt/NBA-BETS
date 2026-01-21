#!/usr/bin/env python3
"""
Quick validation of Dean Oliver's Four Factors integration.

Tests that the new TOV%, ORB%, and FT Rate calculations work correctly
and produce reasonable values for NBA players.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_complete_balldontlie import PlayerStatsCalculator


def test_four_factors():
    """Test Four Factors calculations with sample data."""

    calc = PlayerStatsCalculator(window=10)

    # Sample game data for a typical NBA player
    # Example: Guard averaging 20pts, 4reb, 6ast, 2 tov
    sample_games = [
        ("2025-01-20", {
            'pts': 22, 'reb': 4, 'ast': 6, 'stl': 1, 'blk': 0,
            'fgm': 8, 'fga': 16, 'fg3m': 2, 'fg3a': 6,
            'ftm': 4, 'fta': 5, 'min': 32, 'turnover': 2,
            'oreb': 1, 'dreb': 3
        }),
        ("2025-01-18", {
            'pts': 18, 'reb': 5, 'ast': 7, 'stl': 2, 'blk': 0,
            'fgm': 7, 'fga': 14, 'fg3m': 1, 'fg3a': 4,
            'ftm': 3, 'fta': 4, 'min': 34, 'turnover': 3,
            'oreb': 2, 'dreb': 3
        }),
        ("2025-01-16", {
            'pts': 25, 'reb': 3, 'ast': 5, 'stl': 1, 'blk': 1,
            'fgm': 9, 'fga': 18, 'fg3m': 3, 'fg3a': 8,
            'ftm': 4, 'fta': 4, 'min': 36, 'turnover': 2,
            'oreb': 0, 'dreb': 3
        }),
        ("2025-01-14", {
            'pts': 19, 'reb': 6, 'ast': 8, 'stl': 0, 'blk': 0,
            'fgm': 7, 'fga': 15, 'fg3m': 2, 'fg3a': 5,
            'ftm': 3, 'fta': 3, 'min': 30, 'turnover': 1,
            'oreb': 1, 'dreb': 5
        }),
        ("2025-01-12", {
            'pts': 21, 'reb': 4, 'ast': 6, 'stl': 2, 'blk': 0,
            'fgm': 8, 'fga': 17, 'fg3m': 2, 'fg3a': 7,
            'ftm': 3, 'fta': 4, 'min': 33, 'turnover': 3,
            'oreb': 1, 'dreb': 3
        }),
    ]

    # Add games to calculator
    player_id = 12345
    for date, stats in sample_games:
        calc.add_game_stats(
            player_id=player_id,
            game_date=date,
            stats=stats,
            player_info={'position': 'G'}
        )

    # Get features
    features = calc.get_player_stats_before_date(
        player_id=player_id,
        date="2025-01-21",
        min_games=3
    )

    if not features:
        print("❌ FAILED: No features generated")
        return False

    # Check that Four Factors features exist
    four_factors_keys = ['efg_pct', 'tov_pct', 'orb_pct', 'ft_rate']
    missing = [k for k in four_factors_keys if k not in features]

    if missing:
        print(f"❌ FAILED: Missing Four Factors features: {missing}")
        return False

    # Validate ranges
    validations = [
        ('efg_pct', 0.40, 0.65, "Effective FG%"),
        ('tov_pct', 0.05, 0.25, "Turnover %"),
        ('orb_pct', 0.00, 0.40, "Offensive Rebound %"),
        ('ft_rate', 0.00, 0.60, "Free Throw Rate"),
        ('ts_pct', 0.40, 0.70, "True Shooting %"),
        ('fta_rate', 0.00, 0.60, "FTA Rate"),
    ]

    print("\n" + "="*60)
    print("FOUR FACTORS VALIDATION")
    print("="*60)

    all_valid = True
    for key, min_val, max_val, label in validations:
        value = features.get(key, -999)
        in_range = min_val <= value <= max_val
        status = "✅" if in_range else "❌"
        print(f"{status} {label:25s} = {value:.3f} (range: {min_val:.2f}-{max_val:.2f})")
        if not in_range:
            all_valid = False

    # Show sample of other features
    print("\n" + "="*60)
    print("OTHER KEY FEATURES")
    print("="*60)
    key_features = [
        'season_pts_avg', 'season_reb_avg', 'season_ast_avg',
        'last10_pts_avg', 'usage_rate', 'bpm'
    ]
    for key in key_features:
        value = features.get(key, -999)
        print(f"  {key:20s} = {value:.3f}")

    print("\n" + "="*60)
    print("CALCULATION DETAILS")
    print("="*60)

    # Calculate expected values manually
    total_fgm = sum(g['fgm'] for _, g in sample_games)
    total_fg3m = sum(g['fg3m'] for _, g in sample_games)
    total_fga = sum(g['fga'] for _, g in sample_games)
    total_ftm = sum(g['ftm'] for _, g in sample_games)
    total_fta = sum(g['fta'] for _, g in sample_games)
    total_tov = sum(g['turnover'] for _, g in sample_games)

    expected_efg = (total_fgm + 0.5 * total_fg3m) / total_fga
    expected_tov_pct = total_tov / (total_fga + 0.44 * total_fta + total_tov)
    expected_ft_rate = total_ftm / total_fga

    print(f"Expected eFG%:     {expected_efg:.3f}")
    print(f"Calculated eFG%:   {features['efg_pct']:.3f}")
    print(f"Match: {abs(expected_efg - features['efg_pct']) < 0.01}")
    print()
    print(f"Expected TOV%:     {expected_tov_pct:.3f}")
    print(f"Calculated TOV%:   {features['tov_pct']:.3f}")
    print(f"Match: {abs(expected_tov_pct - features['tov_pct']) < 0.01}")
    print()
    print(f"Expected FT Rate:  {expected_ft_rate:.3f}")
    print(f"Calculated FT Rate: {features['ft_rate']:.3f}")
    print(f"Match: {abs(expected_ft_rate - features['ft_rate']) < 0.01}")

    if all_valid:
        print("\n✅ ALL VALIDATIONS PASSED")
        return True
    else:
        print("\n❌ SOME VALIDATIONS FAILED")
        return False


if __name__ == '__main__':
    success = test_four_factors()
    sys.exit(0 if success else 1)
