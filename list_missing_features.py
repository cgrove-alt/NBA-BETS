"""Get complete list of missing features."""

import pickle
from pathlib import Path

# Load one model to get all expected features
model_path = Path("models/player_points_ensemble.pkl")
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

all_features = model_data['feature_names']

# Backtest features (108)
backtest_features = [
    'season_games', 'season_pts_avg', 'season_reb_avg', 'season_ast_avg',
    'season_fg3m_avg', 'season_min_avg', 'recent_pts_avg', 'recent_pts_std',
    'recent_pts_min', 'recent_pts_max', 'recent_reb_avg', 'recent_reb_std',
    'recent_ast_avg', 'recent_ast_std', 'recent_fg3m_avg', 'recent_fg3m_std',
    'recent_min_avg', 'min_trend', 'min_consistency', 'last5_min_avg',
    'last5_pts_avg', 'last5_reb_avg', 'last5_ast_avg', 'last5_fg3m_avg',
    'last3_pts_avg', 'last3_reb_avg', 'last3_ast_avg', 'last3_fg3m_avg',
    'last3_min_avg', 'pts_trend', 'reb_trend', 'ast_trend', 'fg3m_trend',
    'season_pts_std', 'season_reb_std', 'season_ast_std', 'season_fg3m_std',
    'pra_avg', 'pra_std', 'last3_pra_avg', 'ts_pct', 'efg_pct', 'usage_rate',
    'fg3_rate', 'fta_rate', 'bpm', 'assist_rate', 'rebound_rate', 'days_rest',
    'is_back_to_back', 'fg3_pct', 'last5_fg3_pct', 'fg3_pct_variance',
    'fg3_hot_streak', 'fg3_cold_streak', 'fg3_momentum', 'fg3a_per_min',
    'fg3a_avg', 'fg3a_std', 'fg3a_consistency', 'regressed_fg3_pct',
    'expected_fg3m', 'fg3_makes_std', 'fg3_attempt_trend', 'is_volume_shooter',
    'shooting_confidence', 'is_guard', 'is_forward', 'is_center', 'is_starter',
    'is_star', 'is_high_volume', 'is_ball_handler', 'pos_reb_factor',
    'pos_ast_factor', 'opp_def_rating', 'opp_off_rating', 'opp_net_rating',
    'opp_pts_allowed', 'opp_pts_allowed_recent', 'opp_pts_allowed_std',
    'opp_pace', 'opp_pace_season', 'opp_def_strength', 'opp_reb_factor',
    'opp_location_def', 'opp_win_pct', 'opp_recent_win_pct', 'is_home',
    'team_pace', 'team_off_rating', 'opp_pts_allowed_to_guards',
    'opp_pts_allowed_to_forwards', 'opp_pts_allowed_to_centers',
    'opp_reb_allowed_to_guards', 'opp_reb_allowed_to_forwards',
    'opp_reb_allowed_to_centers', 'opp_ast_allowed_to_guards',
    'opp_ast_allowed_to_forwards', 'opp_ast_allowed_to_centers',
    'opp_fg3m_allowed_to_guards', 'opp_fg3m_allowed_to_forwards',
    'opp_fg3m_allowed_to_centers', 'opp_pts_vs_pos_diff',
    'opp_reb_vs_pos_diff', 'opp_ast_vs_pos_diff', 'opp_fg3m_vs_pos_diff',
    'opp_pts_vs_pos_std'
]

backtest_set = set(backtest_features)
missing = [f for f in all_features if f not in backtest_set]

print(f"Total features needed: {len(all_features)}")
print(f"Backtest provides: {len(backtest_features)}")
print(f"Missing: {len(missing)}\n")

print("MISSING FEATURES:")
print("="*60)
for i, feat in enumerate(sorted(missing), 1):
    print(f"{i:2}. {feat}")
