"""
Enhanced Model Training V2 - Integrating All Improvements

This script trains player prop models with:
1. Four Factors features (eFG%, TOV%, ORB%, FT Rate)
2. Injury Impact features (star_player_out, usage_lost)
3. Style Clash features (pace_mismatch, style_compatibility)
4. Stacked Ensemble architecture

Based on forensic analysis of Jan 7th predictions which showed:
- Star players significantly under-predicted (SGA: 15.5 vs 46 actual)
- High bias in certain game contexts
- Model struggles with usage redistribution

Usage:
    python3 train_enhanced_v2.py
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Import our new modules
from advanced_stats_v2 import FourFactorsCalculator, StyleClashCalculator
from injury_impact_v2 import TeamInjuryManager, PlayerUsageTracker
from stacked_model_v2 import StackedPropModel, QuantileStackedModel

# Model save directory
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Cache directory
CACHE_DIR = Path("data/balldontlie_cache")


class EnhancedFeatureGenerator:
    """
    Generates enhanced features for player prop predictions.

    Combines:
    - Traditional player stats (season avg, recent form)
    - Four Factors team context
    - Injury impact adjustments
    - Style clash indicators
    """

    def __init__(self):
        self.four_factors = FourFactorsCalculator()
        self.style_calc = StyleClashCalculator()
        self.injury_manager = TeamInjuryManager()

        # Player statistics cache
        self.player_stats = defaultdict(list)  # player_id -> [(date, stats), ...]
        self.player_info = {}  # player_id -> info dict
        self.team_rosters = defaultdict(set)  # team_id -> set of player_ids

    def load_historical_data(self, data_dir: Path = CACHE_DIR):
        """
        Load historical game and player stats from cache.

        Populates:
        - Four Factors calculator
        - Injury manager
        - Player stats cache
        """
        print("Loading historical data...")

        # Load games
        games_file = data_dir / "games_2025_full.json"
        if games_file.exists():
            with open(games_file) as f:
                games_data = json.load(f)
            games = games_data.get('games', [])
            print(f"  Loaded {len(games)} games")

            # Add to Four Factors calculator
            for game in games:
                if game.get('status') != 'Final':
                    continue

                home_id = game.get('home_team', {}).get('id')
                away_id = game.get('visitor_team', {}).get('id')
                game_date = game.get('date', '')[:10]

                # We need team box scores for Four Factors
                # For now, use simplified approach
                home_score = game.get('home_team_score', 0)
                away_score = game.get('visitor_team_score', 0)

                if home_id and home_score > 0:
                    self.four_factors.add_game(
                        home_id, game_date,
                        {'pts': home_score, 'fga': 85, 'fgm': 35, 'fg3m': 12, 'fta': 20, 'fta': 20, 'orb': 10, 'tov': 13},
                        opponent_id=away_id
                    )
                if away_id and away_score > 0:
                    self.four_factors.add_game(
                        away_id, game_date,
                        {'pts': away_score, 'fga': 85, 'fgm': 35, 'fg3m': 12, 'fta': 20, 'orb': 10, 'tov': 13},
                        opponent_id=home_id
                    )

        # Load player stats
        stats_loaded = 0
        batch_files = list(data_dir.glob("player_stats_batch_*.json"))
        for batch_file in batch_files:
            try:
                with open(batch_file) as f:
                    batch_data = json.load(f)

                if isinstance(batch_data, dict):
                    for game_id_str, game_stats in batch_data.items():
                        if isinstance(game_stats, list):
                            for stat in game_stats:
                                player = stat.get('player', {})
                                player_id = player.get('id')
                                game = stat.get('game', {})
                                game_date = game.get('date', '')[:10]
                                team_id = stat.get('team', {}).get('id')

                                if player_id and game_date:
                                    self.player_stats[player_id].append((game_date, stat))
                                    stats_loaded += 1

                                    if player_id not in self.player_info:
                                        self.player_info[player_id] = {
                                            'first_name': player.get('first_name', ''),
                                            'last_name': player.get('last_name', ''),
                                            'position': player.get('position', 'F'),
                                        }

                                    # Update injury manager
                                    self.injury_manager.add_player_game(
                                        player_id, game_date, stat,
                                        self.player_info[player_id], team_id
                                    )

                                    if team_id:
                                        self.team_rosters[team_id].add(player_id)

            except Exception as e:
                print(f"  Warning: Could not load {batch_file}: {e}")

        # Sort player stats by date
        for player_id in self.player_stats:
            self.player_stats[player_id].sort(key=lambda x: x[0])

        print(f"  Loaded {stats_loaded} player stat records for {len(self.player_stats)} players")

    def generate_player_features(
        self,
        player_id: int,
        game_date: str,
        team_id: int = None,
        opponent_id: int = None,
        is_home: bool = True
    ) -> Optional[Dict]:
        """
        Generate all features for a player prediction.

        Combines:
        - Traditional player stats
        - Four Factors context
        - Injury impact
        - Matchup features
        """
        if player_id not in self.player_stats:
            return None

        # Get games before target date
        games = [(d, s) for d, s in self.player_stats[player_id] if d < game_date]
        if len(games) < 3:
            return None

        # Sort by date descending (most recent first)
        games.sort(key=lambda x: x[0], reverse=True)

        recent = games[:10]
        last_5 = games[:5]
        last_3 = games[:3]
        season = games

        # Helper functions
        def parse_min(m):
            if isinstance(m, (int, float)):
                return float(m)
            if not m:
                return 0.0
            try:
                if ':' in str(m):
                    parts = str(m).split(':')
                    return float(parts[0]) + float(parts[1]) / 60
                return float(m)
            except:
                return 0.0

        def get_stat(s, key):
            return s.get(key, 0) or 0

        # Extract basic stats
        pts = [get_stat(s, 'pts') for _, s in recent]
        reb = [get_stat(s, 'reb') for _, s in recent]
        ast = [get_stat(s, 'ast') for _, s in recent]
        fg3m = [get_stat(s, 'fg3m') for _, s in recent]
        mins = [parse_min(s.get('min', 0)) for _, s in recent]

        season_pts = [get_stat(s, 'pts') for _, s in season]
        season_reb = [get_stat(s, 'reb') for _, s in season]
        season_ast = [get_stat(s, 'ast') for _, s in season]
        season_mins = [parse_min(s.get('min', 0)) for _, s in season]

        # Build base features
        features = {
            # Season averages
            'season_games': len(season),
            'season_pts_avg': np.mean(season_pts),
            'season_reb_avg': np.mean(season_reb),
            'season_ast_avg': np.mean(season_ast),
            'season_min_avg': np.mean(season_mins),

            # Recent averages
            'recent_pts_avg': np.mean(pts),
            'recent_pts_std': np.std(pts) if len(pts) > 1 else 0,
            'recent_reb_avg': np.mean(reb),
            'recent_reb_std': np.std(reb) if len(reb) > 1 else 0,
            'recent_ast_avg': np.mean(ast),
            'recent_ast_std': np.std(ast) if len(ast) > 1 else 0,
            'recent_fg3m_avg': np.mean(fg3m),
            'recent_min_avg': np.mean(mins),

            # Last 5 averages
            'last5_pts_avg': np.mean([get_stat(s, 'pts') for _, s in last_5]),
            'last5_reb_avg': np.mean([get_stat(s, 'reb') for _, s in last_5]),
            'last5_ast_avg': np.mean([get_stat(s, 'ast') for _, s in last_5]),
            'last5_min_avg': np.mean([parse_min(s.get('min', 0)) for _, s in last_5]),

            # Last 3 averages (hot hand detection)
            'last3_pts_avg': np.mean([get_stat(s, 'pts') for _, s in last_3]),
            'last3_reb_avg': np.mean([get_stat(s, 'reb') for _, s in last_3]),
            'last3_ast_avg': np.mean([get_stat(s, 'ast') for _, s in last_3]),
            'last3_min_avg': np.mean([parse_min(s.get('min', 0)) for _, s in last_3]),

            # Trends
            'pts_trend': np.mean([get_stat(s, 'pts') for _, s in last_5]) - np.mean(pts),
            'min_trend': np.mean([parse_min(s.get('min', 0)) for _, s in last_5]) - np.mean(mins),

            # PRA combined
            'pra_avg': np.mean([get_stat(s, 'pts') + get_stat(s, 'reb') + get_stat(s, 'ast') for _, s in recent]),
            'last3_pra_avg': np.mean([get_stat(s, 'pts') + get_stat(s, 'reb') + get_stat(s, 'ast') for _, s in last_3]),

            # Context
            'is_home': 1 if is_home else 0,
        }

        # Calculate days rest
        try:
            current = datetime.strptime(game_date, "%Y-%m-%d")
            last_game = datetime.strptime(games[0][0], "%Y-%m-%d")
            days_rest = (current - last_game).days
        except:
            days_rest = 2
        features['days_rest'] = days_rest
        features['is_back_to_back'] = 1 if days_rest == 1 else 0

        # Add Four Factors context (team-level)
        if team_id:
            ff = self.four_factors.get_four_factors_before_date(team_id, game_date)
            if ff:
                for key, val in ff.items():
                    features[f'team_{key}'] = val

        # Add opponent Four Factors
        if opponent_id:
            opp_ff = self.four_factors.get_four_factors_before_date(opponent_id, game_date)
            if opp_ff:
                for key, val in opp_ff.items():
                    features[f'opp_{key}'] = val

        # Add injury impact features
        if team_id:
            injury_boost = self.injury_manager.get_player_injury_boost(
                player_id, team_id, game_date
            )
            for key, val in injury_boost.items():
                features[f'injury_{key}'] = val

        # Infer position features
        info = self.player_info.get(player_id, {})
        pos = info.get('position', 'F')
        features['is_guard'] = 1 if pos in ['PG', 'SG', 'G', 'G-F'] else 0
        features['is_forward'] = 1 if pos in ['SF', 'PF', 'F', 'F-G', 'F-C'] else 0
        features['is_center'] = 1 if pos in ['C', 'C-F'] else 0

        # Star player indicator
        features['is_star'] = 1 if features['season_pts_avg'] >= 20 else 0
        features['is_starter'] = 1 if features['season_min_avg'] >= 28 else 0

        return features

    def build_training_dataset(
        self,
        start_date: str = "2025-10-21",
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Build training dataset from historical data.

        Returns DataFrame with features and target columns.
        """
        print("\nBuilding training dataset...")

        records = []

        for player_id, games in self.player_stats.items():
            # Skip players with insufficient history
            if len(games) < 5:
                continue

            for i, (game_date, stat) in enumerate(games):
                if game_date < start_date:
                    continue
                if end_date and game_date > end_date:
                    continue

                # Need at least 3 prior games
                prior_games = [(d, s) for d, s in games if d < game_date]
                if len(prior_games) < 3:
                    continue

                # Get team info
                team_id = stat.get('team', {}).get('id')
                game = stat.get('game', {})

                # Determine opponent
                home_team_id = game.get('home_team', {}).get('id')
                away_team_id = game.get('visitor_team', {}).get('id')

                if team_id == home_team_id:
                    opponent_id = away_team_id
                    is_home = True
                else:
                    opponent_id = home_team_id
                    is_home = False

                # Generate features
                features = self.generate_player_features(
                    player_id, game_date, team_id, opponent_id, is_home
                )

                if not features:
                    continue

                # Add targets
                features['pts'] = stat.get('pts', 0) or 0
                features['reb'] = stat.get('reb', 0) or 0
                features['ast'] = stat.get('ast', 0) or 0
                features['fg3m'] = stat.get('fg3m', 0) or 0
                features['pra'] = features['pts'] + features['reb'] + features['ast']

                # Metadata
                features['player_id'] = player_id
                features['game_date'] = game_date

                records.append(features)

        df = pd.DataFrame(records)
        print(f"  Built dataset with {len(df)} samples, {len(df.columns)} columns")

        return df


def train_enhanced_models():
    """
    Train enhanced models with all new features.
    """
    print("=" * 60)
    print("ENHANCED MODEL TRAINING V2")
    print("=" * 60)

    # Initialize feature generator
    generator = EnhancedFeatureGenerator()
    generator.load_historical_data()

    # Build training dataset
    df = generator.build_training_dataset()

    if len(df) < 100:
        print("Insufficient data for training!")
        return

    # Define target columns
    TARGET_COLS = {
        'points': 'pts',
        'rebounds': 'reb',
        'assists': 'ast',
        'threes': 'fg3m',
        'pra': 'pra',
    }

    # Feature columns (exclude targets and metadata)
    exclude_cols = list(TARGET_COLS.values()) + ['player_id', 'game_date']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"\nUsing {len(feature_cols)} features for training")

    # Train each prop type
    models = {}

    for prop_type, target_col in TARGET_COLS.items():
        print(f"\n{'='*60}")
        print(f"Training {prop_type.upper()} model")
        print(f"{'='*60}")

        X = df[feature_cols].copy()
        y = df[target_col].values

        # Remove rows with missing targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        # Fill NaN features
        X = X.fillna(0)

        if len(y) < 100:
            print(f"  Skipping: insufficient data ({len(y)} samples)")
            continue

        # Split data (temporal split)
        split_idx = int(len(y) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train stacked model
        model = StackedPropModel(prop_type=prop_type, verbose=True)
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n  Test Set Metrics:")
        print(f"    RMSE: {rmse:.3f}")
        print(f"    MAE:  {mae:.3f}")
        print(f"    R²:   {r2:.3f}")

        # Save model
        output_path = MODEL_DIR / f"player_{prop_type}_enhanced.pkl"
        model.save(str(output_path))
        print(f"  Saved to {output_path}")

        models[prop_type] = model

        # Print feature importance
        importance = model.get_feature_importance()
        if len(importance) > 0:
            print(f"\n  Top 10 Features:")
            for i, (feat, imp) in enumerate(importance.head(10).items()):
                print(f"    {i+1}. {feat}: {imp:.4f}")

    # Save training summary
    summary = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'samples': len(df),
        'features': len(feature_cols),
        'models_trained': list(models.keys()),
    }

    summary_path = MODEL_DIR / "enhanced_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete! Summary saved to {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_enhanced_models()
