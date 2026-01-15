"""
Feature Consistency Verification

This script checks if comprehensive_backtest.py generates all 150 features
that the ensemble models expect. Mismatches cause poor predictions.
"""

import pickle
import json
from pathlib import Path
from collections import defaultdict

MODEL_DIR = Path("models")
PROP_TYPES = ['points', 'rebounds', 'assists', 'threes', 'pra']

def load_ensemble_model(prop_type):
    """Load ensemble model and extract feature names."""
    model_path = MODEL_DIR / f"player_{prop_type}_ensemble.pkl"
    if not model_path.exists():
        return None, []

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    feature_names = model_data.get('feature_names', [])
    return model_data, feature_names

def get_backtest_features():
    """
    Extract features that comprehensive_backtest.py generates.
    Look at get_player_features_before_date() method.
    """
    # Read the backtest script and identify all features
    backtest_file = Path("comprehensive_backtest.py")

    if not backtest_file.exists():
        return []

    # This is the list of features from comprehensive_backtest.py
    # Lines 699-816 in get_player_features_before_date()

    backtest_features = [
        # Season averages (1-6)
        'season_games',
        'season_pts_avg',
        'season_reb_avg',
        'season_ast_avg',
        'season_fg3m_avg',
        'season_min_avg',

        # Recent averages (7-17)
        'recent_pts_avg',
        'recent_pts_std',
        'recent_pts_min',
        'recent_pts_max',
        'recent_reb_avg',
        'recent_reb_std',
        'recent_ast_avg',
        'recent_ast_std',
        'recent_fg3m_avg',
        'recent_fg3m_std',
        'recent_min_avg',

        # Minutes features (18-20)
        'min_trend',
        'min_consistency',
        'last5_min_avg',

        # Last 5 games (21-24)
        'last5_pts_avg',
        'last5_reb_avg',
        'last5_ast_avg',
        'last5_fg3m_avg',

        # Last 3 games (25-29)
        'last3_pts_avg',
        'last3_reb_avg',
        'last3_ast_avg',
        'last3_fg3m_avg',
        'last3_min_avg',

        # Trends (30-33)
        'pts_trend',
        'reb_trend',
        'ast_trend',
        'fg3m_trend',

        # Season variance (34-37)
        'season_pts_std',
        'season_reb_std',
        'season_ast_std',
        'season_fg3m_std',

        # Combined PRA stats (38-40)
        'pra_avg',
        'pra_std',
        'last3_pra_avg',

        # Efficiency stats (41-45)
        'ts_pct',
        'efg_pct',
        'usage_rate',
        'fg3_rate',
        'fta_rate',

        # Advanced stats (46-48)
        'bpm',
        'assist_rate',
        'rebound_rate',

        # Rest features (49-50)
        'days_rest',
        'is_back_to_back',

        # 3PM features (51-56)
        'fg3_pct',
        'last5_fg3_pct',
        'fg3_pct_variance',
        'fg3_hot_streak',
        'fg3_cold_streak',
        'fg3_momentum',

        # Specialized 3PM features (57-66)
        'fg3a_per_min',
        'fg3a_avg',
        'fg3a_std',
        'fg3a_consistency',
        'regressed_fg3_pct',
        'expected_fg3m',
        'fg3_makes_std',
        'fg3_attempt_trend',
        'is_volume_shooter',
        'shooting_confidence',

        # Position/role features (67-75)
        'is_guard',
        'is_forward',
        'is_center',
        'is_starter',
        'is_star',
        'is_high_volume',
        'is_ball_handler',
        'pos_reb_factor',
        'pos_ast_factor',

        # Opponent features (76-88)
        'opp_def_rating',
        'opp_off_rating',
        'opp_net_rating',
        'opp_pts_allowed',
        'opp_pts_allowed_recent',
        'opp_pts_allowed_std',
        'opp_pace',
        'opp_pace_season',
        'opp_def_strength',
        'opp_reb_factor',
        'opp_location_def',
        'opp_win_pct',
        'opp_recent_win_pct',

        # Game context (89-91)
        'is_home',
        'team_pace',
        'team_off_rating',

        # Position defense features (TIER 2.2) - added by position_defense_calc
        'opp_pts_allowed_to_guards',
        'opp_pts_allowed_to_forwards',
        'opp_pts_allowed_to_centers',
        'opp_reb_allowed_to_guards',
        'opp_reb_allowed_to_forwards',
        'opp_reb_allowed_to_centers',
        'opp_ast_allowed_to_guards',
        'opp_ast_allowed_to_forwards',
        'opp_ast_allowed_to_centers',
        'opp_fg3m_allowed_to_guards',
        'opp_fg3m_allowed_to_forwards',
        'opp_fg3m_allowed_to_centers',
        'opp_pts_vs_pos_diff',
        'opp_reb_vs_pos_diff',
        'opp_ast_vs_pos_diff',
        'opp_fg3m_vs_pos_diff',
        'opp_pts_vs_pos_std',
    ]

    return backtest_features

def compare_features():
    """Compare model features vs backtest features."""
    print("="*80)
    print("FEATURE CONSISTENCY VERIFICATION")
    print("="*80)

    backtest_features = set(get_backtest_features())
    print(f"\nBacktest generates: {len(backtest_features)} features")

    print("\n" + "="*80)
    print("CHECKING EACH PROP TYPE")
    print("="*80)

    all_issues = []

    for prop_type in PROP_TYPES:
        print(f"\n{prop_type.upper()}:")
        print("-" * 40)

        model_data, model_features = load_ensemble_model(prop_type)

        if not model_features:
            print(f"  ✗ Could not load model")
            continue

        model_feature_set = set(model_features)
        print(f"  Model expects: {len(model_features)} features")

        # Find mismatches
        missing_in_backtest = model_feature_set - backtest_features
        extra_in_backtest = backtest_features - model_feature_set

        if not missing_in_backtest and not extra_in_backtest:
            print(f"  ✓ Perfect match!")
        else:
            if missing_in_backtest:
                print(f"  ✗ Missing in backtest: {len(missing_in_backtest)} features")
                print(f"    First 10: {list(missing_in_backtest)[:10]}")
                all_issues.append({
                    'prop': prop_type,
                    'type': 'missing',
                    'count': len(missing_in_backtest),
                    'features': list(missing_in_backtest)
                })

            if extra_in_backtest:
                print(f"  ⚠  Extra in backtest: {len(extra_in_backtest)} features")
                print(f"    (These are ignored by model)")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if not all_issues:
        print("\n✅ ALL MODELS HAVE MATCHING FEATURES!")
        print("   Feature generation is consistent.")
        return True
    else:
        print(f"\n❌ FEATURE MISMATCHES FOUND!")
        print(f"   {len(all_issues)} prop types have missing features\n")

        # Show common missing features
        if all_issues:
            all_missing = set()
            for issue in all_issues:
                if issue['type'] == 'missing':
                    all_missing.update(issue['features'])

            if all_missing:
                print(f"Common missing features ({len(all_missing)}):")
                for feat in sorted(all_missing)[:20]:
                    print(f"  - {feat}")
                if len(all_missing) > 20:
                    print(f"  ... and {len(all_missing) - 20} more")

        return False

def main():
    """Run feature consistency check."""
    is_consistent = compare_features()

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)

    if is_consistent:
        print("\n✓ Feature generation is correct")
        print("  The poor performance is NOT due to missing features")
        print("\n  Other possible causes:")
        print("    1. Early season volatility (small sample)")
        print("    2. DNP predictions (injury tracker not integrated)")
        print("    3. Systematic bias (needs correction)")
        print("    4. Different dataset (100 vs 372 games)")
    else:
        print("\n✗ Feature mismatch detected!")
        print("  The backtest is NOT generating all features the models need")
        print("\n  This explains the poor performance!")
        print("\n  Action required:")
        print("    1. Update comprehensive_backtest.py to generate missing features")
        print("    2. OR retrain models with only the features backtest provides")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
