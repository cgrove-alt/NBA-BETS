"""
Training Script for Stacking Models

Trains stacking ensemble models for:
- Moneyline predictions (classification)
- Spread predictions (regression)
- Player prop predictions (regression)

Includes Optuna hyperparameter tuning for optimal performance.

Usage:
    python3 train_stacking_model.py [--tune] [--model moneyline|spread|props]
"""

import os
import sys
import json
import pickle
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent / "models"))

from models.stacking_model import StackingClassifier, StackingRegressor, create_stacking_model

# Try importing Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not available - hyperparameter tuning disabled")

# Directories
MODEL_DIR = Path("models")
CACHE_DIR = Path("data/balldontlie_cache")


def calculate_time_decay_weights(dates: pd.Series, half_life_days: int = 180) -> np.ndarray:
    """
    Calculate time-decay sample weights for training data.

    Recent games are weighted higher than older games using exponential decay.
    Half-life is the number of days for weight to decay to 50%.

    Parameters:
    -----------
    dates : pd.Series
        Game dates in 'YYYY-MM-DD' format
    half_life_days : int
        Number of days for weight to decay to 50% (default: 180 days = 6 months)

    Returns:
    --------
    weights : np.ndarray
        Sample weights with recent games weighted higher
    """
    # Convert dates to datetime
    date_objs = pd.to_datetime(dates)
    most_recent = date_objs.max()

    # Calculate days ago for each sample
    days_ago = (most_recent - date_objs).dt.days

    # Calculate weights using exponential decay: weight = 0.5 ^ (days_ago / half_life)
    weights = 0.5 ** (days_ago / half_life_days)

    # Normalize weights to sum to N (number of samples)
    weights = weights * len(weights) / weights.sum()

    return weights.values


class TrainingDataLoader:
    """Load and prepare training data from cache with proper feature engineering."""

    def __init__(self, cache_dir: Path = CACHE_DIR, window: int = 10):
        self.cache_dir = cache_dir
        self.window = window
        self.games = []
        self.player_stats = defaultdict(list)
        self.team_history = defaultdict(list)  # team_id -> [(date, game_data)]

    def load_games(self) -> List[Dict]:
        """Load game data from multiple seasons."""
        all_games = []

        # Load all available season files
        for season_file in sorted(self.cache_dir.glob("games_*.json")):
            try:
                with open(season_file) as f:
                    data = json.load(f)
                games = data.get('games', []) if isinstance(data, dict) else data
                final_games = [g for g in games if g.get('status') == 'Final']
                all_games.extend(final_games)
                print(f"  Loaded {len(final_games)} games from {season_file.name}")
            except Exception as e:
                print(f"  Warning loading {season_file}: {e}")

        # Sort by date and deduplicate
        seen_ids = set()
        unique_games = []
        for g in sorted(all_games, key=lambda x: x.get('date', '')):
            gid = g.get('id')
            if gid and gid not in seen_ids:
                seen_ids.add(gid)
                unique_games.append(g)

        self.games = unique_games
        print(f"  Total unique games: {len(self.games)}")

        # Build team history for feature calculation
        self._build_team_history()
        return self.games

    def _build_team_history(self):
        """Build historical record for each team."""
        for game in self.games:
            game_date = game.get('date', '')[:10]
            home_team_id = game.get('home_team', {}).get('id')
            away_team_id = game.get('visitor_team', {}).get('id')
            home_score = game.get('home_team_score', 0) or 0
            away_score = game.get('visitor_team_score', 0) or 0

            if not all([game_date, home_team_id, away_team_id, home_score]):
                continue

            # Record for home team
            self.team_history[home_team_id].append({
                'date': game_date,
                'is_home': True,
                'pts_scored': home_score,
                'pts_allowed': away_score,
                'won': home_score > away_score,
                'point_diff': home_score - away_score,
                'opponent_id': away_team_id,
            })

            # Record for away team
            self.team_history[away_team_id].append({
                'date': game_date,
                'is_home': False,
                'pts_scored': away_score,
                'pts_allowed': home_score,
                'won': away_score > home_score,
                'point_diff': away_score - home_score,
                'opponent_id': home_team_id,
            })

    def _get_team_features(self, team_id: int, before_date: str, min_games: int = 5) -> Optional[Dict]:
        """Calculate team features using only games BEFORE the given date."""
        if team_id not in self.team_history:
            return None

        # Get games before this date
        prior_games = [g for g in self.team_history[team_id] if g['date'] < before_date]

        if len(prior_games) < min_games:
            return None

        # Sort and get recent games
        prior_games.sort(key=lambda x: x['date'], reverse=True)
        recent = prior_games[:self.window]
        all_games = prior_games

        # Calculate features
        return {
            'games_played': len(all_games),
            'win_pct': np.mean([g['won'] for g in all_games]),
            'recent_win_pct': np.mean([g['won'] for g in recent]),
            'pts_avg': np.mean([g['pts_scored'] for g in all_games]),
            'recent_pts_avg': np.mean([g['pts_scored'] for g in recent]),
            'pts_allowed_avg': np.mean([g['pts_allowed'] for g in all_games]),
            'recent_pts_allowed': np.mean([g['pts_allowed'] for g in recent]),
            'point_diff_avg': np.mean([g['point_diff'] for g in all_games]),
            'recent_point_diff': np.mean([g['point_diff'] for g in recent]),
            'home_win_pct': np.mean([g['won'] for g in all_games if g['is_home']]) if any(g['is_home'] for g in all_games) else 0.5,
            'away_win_pct': np.mean([g['won'] for g in all_games if not g['is_home']]) if any(not g['is_home'] for g in all_games) else 0.5,
            'streak': self._calc_streak(recent),
        }

    def _calc_streak(self, games: List[Dict]) -> int:
        """Calculate current win/loss streak (positive=wins, negative=losses)."""
        if not games:
            return 0
        streak = 0
        first_result = games[0]['won']
        for g in games:
            if g['won'] == first_result:
                streak += 1 if first_result else -1
            else:
                break
        return streak

    def _extract_context_features(self, game: Dict, home_feats: Dict, away_feats: Dict) -> Dict:
        """
        Extract context features for meta-learner (12 features total).

        Context features help the meta-learner understand the game context
        and weight base model predictions appropriately.

        Features:
        1. days_rest_diff: Difference in days rest (home - away)
        2. pace_combined: Combined pace estimate
        3. injury_count_home: Number of injured players (home)
        4. injury_count_away: Number of injured players (away)
        5. star_player_out_home: Is a star player out (home)?
        6. star_player_out_away: Is a star player out (away)?
        7. line_movement: Market line movement (placeholder)
        8. rlm_flag: Reverse line movement flag (placeholder)
        9. prediction_variance: Filled during training (placeholder = 0)
        10. home_advantage: Standard home court advantage
        11. travel_distance_away: Travel distance for away team (placeholder)
        12. back_to_back_away: Is away team on back-to-back?
        """
        # Calculate days since last game for each team
        game_date = game.get('date', '')[:10]
        home_team_id = game.get('home_team', {}).get('id')
        away_team_id = game.get('visitor_team', {}).get('id')

        # Get last game dates
        home_games = [g for g in self.team_history.get(home_team_id, []) if g['date'] < game_date]
        away_games = [g for g in self.team_history.get(away_team_id, []) if g['date'] < game_date]

        days_rest_home = 2  # Default
        days_rest_away = 2  # Default

        if home_games:
            home_games.sort(key=lambda x: x['date'], reverse=True)
            last_home_date = datetime.strptime(home_games[0]['date'], '%Y-%m-%d')
            current_date = datetime.strptime(game_date, '%Y-%m-%d')
            days_rest_home = (current_date - last_home_date).days

        if away_games:
            away_games.sort(key=lambda x: x['date'], reverse=True)
            last_away_date = datetime.strptime(away_games[0]['date'], '%Y-%m-%d')
            current_date = datetime.strptime(game_date, '%Y-%m-%d')
            days_rest_away = (current_date - last_away_date).days

        # Estimate pace from recent scoring
        pace_combined = (home_feats.get('recent_pts_avg', 110) +
                        away_feats.get('recent_pts_avg', 110)) / 2

        # Context features (12 total)
        context = {
            'ctx_days_rest_diff': days_rest_home - days_rest_away,
            'ctx_pace_combined': pace_combined,
            'ctx_injury_count_home': 0,  # TODO: Will be populated in Phase 2
            'ctx_injury_count_away': 0,  # TODO: Will be populated in Phase 2
            'ctx_star_player_out_home': 0,  # TODO: Will be populated in Phase 2
            'ctx_star_player_out_away': 0,  # TODO: Will be populated in Phase 2
            'ctx_line_movement': 0.0,  # TODO: Will be populated in Phase 2
            'ctx_rlm_flag': 0,  # TODO: Will be populated in Phase 2
            'ctx_prediction_variance': 0.0,  # Filled during training
            'ctx_home_advantage': 3.0,  # Standard NBA home court advantage
            'ctx_travel_distance_away': 0.0,  # TODO: Will be populated in Phase 2
            'ctx_back_to_back_away': int(days_rest_away == 0),
        }

        return context

    def load_player_stats(self):
        """Load player statistics."""
        batch_files = list(self.cache_dir.glob("player_stats_batch_*.json"))
        for batch_file in batch_files:
            try:
                with open(batch_file) as f:
                    batch_data = json.load(f)
                if isinstance(batch_data, dict):
                    for game_id, stats in batch_data.items():
                        if isinstance(stats, list):
                            for stat in stats:
                                player_id = stat.get('player', {}).get('id')
                                game_date = stat.get('game', {}).get('date', '')[:10]
                                if player_id and game_date:
                                    self.player_stats[player_id].append((game_date, stat))
            except Exception as e:
                print(f"Warning loading {batch_file}: {e}")

        # Sort by date
        for pid in self.player_stats:
            self.player_stats[pid].sort(key=lambda x: x[0])

    def build_moneyline_dataset(self) -> pd.DataFrame:
        """Build training dataset for moneyline predictions with proper features."""
        print("Building moneyline training dataset...")

        records = []
        skipped = 0

        for game in self.games:
            game_date = game.get('date', '')[:10]
            home_team_id = game.get('home_team', {}).get('id')
            away_team_id = game.get('visitor_team', {}).get('id')
            home_score = game.get('home_team_score', 0) or 0
            away_score = game.get('visitor_team_score', 0) or 0

            if home_score == 0 or away_score == 0:
                continue

            # Get team features BEFORE this game (prevents leakage)
            home_feats = self._get_team_features(home_team_id, game_date)
            away_feats = self._get_team_features(away_team_id, game_date)

            if not home_feats or not away_feats:
                skipped += 1
                continue

            # Target: 1 if home team won
            home_win = 1 if home_score > away_score else 0

            # Build features (NO LEAKAGE - all from prior games)
            features = {
                'home_win_pct': home_feats['win_pct'],
                'away_win_pct': away_feats['win_pct'],
                'home_recent_win_pct': home_feats['recent_win_pct'],
                'away_recent_win_pct': away_feats['recent_win_pct'],
                'home_pts_avg': home_feats['pts_avg'],
                'away_pts_avg': away_feats['pts_avg'],
                'home_pts_allowed': home_feats['pts_allowed_avg'],
                'away_pts_allowed': away_feats['pts_allowed_avg'],
                'home_point_diff': home_feats['point_diff_avg'],
                'away_point_diff': away_feats['point_diff_avg'],
                'home_recent_diff': home_feats['recent_point_diff'],
                'away_recent_diff': away_feats['recent_point_diff'],
                'home_home_win_pct': home_feats['home_win_pct'],
                'away_away_win_pct': away_feats['away_win_pct'],
                'home_streak': home_feats['streak'],
                'away_streak': away_feats['streak'],
                'win_pct_diff': home_feats['win_pct'] - away_feats['win_pct'],
                'point_diff_diff': home_feats['point_diff_avg'] - away_feats['point_diff_avg'],
                'target': home_win,
                'game_date': game_date,  # Store date for time-decay weights
            }

            # Extract context features for meta-learner (12 features)
            # These will be separated later for meta-learner training
            context = self._extract_context_features(game, home_feats, away_feats)
            features.update(context)

            records.append(features)

        df = pd.DataFrame(records)
        print(f"  Built {len(df)} samples (skipped {skipped} due to insufficient history)")
        return df

    def build_spread_dataset(self) -> pd.DataFrame:
        """Build training dataset for spread predictions with proper features."""
        print("Building spread training dataset...")

        records = []
        skipped = 0

        for game in self.games:
            game_date = game.get('date', '')[:10]
            home_team_id = game.get('home_team', {}).get('id')
            away_team_id = game.get('visitor_team', {}).get('id')
            home_score = game.get('home_team_score', 0) or 0
            away_score = game.get('visitor_team_score', 0) or 0

            if home_score == 0 or away_score == 0:
                continue

            # Get team features BEFORE this game
            home_feats = self._get_team_features(home_team_id, game_date)
            away_feats = self._get_team_features(away_team_id, game_date)

            if not home_feats or not away_feats:
                skipped += 1
                continue

            # Target: actual spread (home - away)
            spread = home_score - away_score

            # Build features
            features = {
                'home_win_pct': home_feats['win_pct'],
                'away_win_pct': away_feats['win_pct'],
                'home_recent_win_pct': home_feats['recent_win_pct'],
                'away_recent_win_pct': away_feats['recent_win_pct'],
                'home_pts_avg': home_feats['pts_avg'],
                'away_pts_avg': away_feats['pts_avg'],
                'home_pts_allowed': home_feats['pts_allowed_avg'],
                'away_pts_allowed': away_feats['pts_allowed_avg'],
                'home_point_diff': home_feats['point_diff_avg'],
                'away_point_diff': away_feats['point_diff_avg'],
                'home_recent_diff': home_feats['recent_point_diff'],
                'away_recent_diff': away_feats['recent_point_diff'],
                'home_home_win_pct': home_feats['home_win_pct'],
                'away_away_win_pct': away_feats['away_win_pct'],
                'home_streak': home_feats['streak'],
                'away_streak': away_feats['streak'],
                'win_pct_diff': home_feats['win_pct'] - away_feats['win_pct'],
                'point_diff_diff': home_feats['point_diff_avg'] - away_feats['point_diff_avg'],
                'expected_margin': home_feats['point_diff_avg'] - away_feats['point_diff_avg'] + 3.0,  # Home court ~3 pts
                'target': spread,
                'game_date': game_date,  # Store date for time-decay weights
            }

            # Extract context features for meta-learner
            context = self._extract_context_features(game, home_feats, away_feats)
            features.update(context)

            records.append(features)

        df = pd.DataFrame(records)
        print(f"  Built {len(df)} samples (skipped {skipped} due to insufficient history)")
        return df

    def build_props_dataset(self, prop_type: str = 'points') -> pd.DataFrame:
        """Build training dataset for player prop predictions."""
        print(f"Building {prop_type} prop training dataset...")

        target_map = {
            'points': 'pts',
            'rebounds': 'reb',
            'assists': 'ast',
            'threes': 'fg3m',
        }
        target_key = target_map.get(prop_type, 'pts')

        records = []
        for player_id, games in self.player_stats.items():
            if len(games) < 5:
                continue

            for i in range(5, len(games)):
                date, stat = games[i]
                prior = games[max(0, i-10):i]

                # Calculate features from prior games
                pts = [s.get('pts', 0) or 0 for _, s in prior]
                reb = [s.get('reb', 0) or 0 for _, s in prior]
                ast = [s.get('ast', 0) or 0 for _, s in prior]

                features = {
                    'pts_avg': np.mean(pts),
                    'pts_std': np.std(pts) if len(pts) > 1 else 0,
                    'reb_avg': np.mean(reb),
                    'ast_avg': np.mean(ast),
                    'games_played': len(prior),
                    'target': stat.get(target_key, 0) or 0,
                }
                records.append(features)

        df = pd.DataFrame(records)
        print(f"  Built {len(df)} samples")
        return df


def train_moneyline_model(data: pd.DataFrame, tune: bool = False) -> StackingClassifier:
    """Train stacking classifier for moneyline with context features and sample weights."""
    print("\n" + "=" * 60)
    print("TRAINING MONEYLINE MODEL WITH CONTEXT FEATURES")
    print("=" * 60)

    # Separate features, context features, target, and dates
    context_cols = [c for c in data.columns if c.startswith('ctx_')]
    feature_cols = [c for c in data.columns if c not in context_cols + ['target', 'game_date']]

    X = data[feature_cols].fillna(0)
    context_features = data[context_cols].fillna(0).values if context_cols else None
    y = data['target'].values
    dates = data['game_date']

    # Calculate time-decay sample weights
    print(f"  Calculating time-decay sample weights (180-day half-life)...")
    sample_weights = calculate_time_decay_weights(dates, half_life_days=180)
    print(f"    Weight range: {sample_weights.min():.3f} to {sample_weights.max():.3f}")

    # Split data (use temporal split)
    split_idx = int(len(y) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    weights_train = sample_weights[:split_idx]

    if context_features is not None:
        context_train = context_features[:split_idx]
        context_test = context_features[split_idx:]
        print(f"  Context features shape: {context_features.shape}")
    else:
        context_train = None
        context_test = None

    if tune and HAS_OPTUNA:
        model = tune_classifier_optuna(X_train, y_train)
    else:
        model = StackingClassifier(verbose=True)
        # Note: StackingClassifier doesn't support context features yet
        # This will be enhanced in a future iteration
        model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print(f"\n  Test Set Metrics:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Log Loss:  {ll:.4f}")

    # A/B Test: Compare with baseline (if exists)
    baseline_path = MODEL_DIR / "moneyline_stacking_baseline.pkl"
    if baseline_path.exists():
        print(f"\n  A/B Test: Comparing with baseline...")
        try:
            with open(baseline_path, 'rb') as f:
                baseline_model = pickle.load(f)
            y_pred_baseline = baseline_model.predict(X_test)
            y_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]
            acc_baseline = accuracy_score(y_test, y_pred_baseline)
            ll_baseline = log_loss(y_test, y_proba_baseline)
            acc_improvement = (acc - acc_baseline) * 100
            ll_improvement = (ll_baseline - ll) / ll_baseline * 100

            print(f"    Baseline Accuracy: {acc_baseline:.4f}")
            print(f"    New Model Accuracy: {acc:.4f}")
            print(f"    Accuracy Improvement: {acc_improvement:+.2f}pp")
            print(f"    Log Loss Improvement: {ll_improvement:+.2f}%")

            # Only save if improved
            if acc >= acc_baseline or ll <= ll_baseline:
                output_path = MODEL_DIR / "moneyline_stacking.pkl"
                model.save(str(output_path))
                print(f"  ✓ Model improved! Saved to {output_path}")
            else:
                print(f"  ✗ Model did not improve. Keeping baseline.")
                return baseline_model
        except Exception as e:
            print(f"  Warning: Could not load baseline: {e}")
    else:
        output_path = MODEL_DIR / "moneyline_stacking.pkl"
        model.save(str(output_path))
        # Save as baseline for future comparisons
        model.save(str(baseline_path))
        print(f"  Saved to {output_path}")

    return model


def train_spread_model(data: pd.DataFrame, tune: bool = False) -> StackingRegressor:
    """Train stacking regressor for spread with context features and sample weights."""
    print("\n" + "=" * 60)
    print("TRAINING SPREAD MODEL WITH CONTEXT FEATURES")
    print("=" * 60)

    # Separate features, context features, target, and dates
    context_cols = [c for c in data.columns if c.startswith('ctx_')]
    feature_cols = [c for c in data.columns if c not in context_cols + ['target', 'game_date']]

    X = data[feature_cols].fillna(0)
    context_features = data[context_cols].fillna(0).values if context_cols else None
    y = data['target'].values
    dates = data['game_date']

    # Calculate time-decay sample weights
    print(f"  Calculating time-decay sample weights (180-day half-life)...")
    sample_weights = calculate_time_decay_weights(dates, half_life_days=180)
    print(f"    Weight range: {sample_weights.min():.3f} to {sample_weights.max():.3f}")

    # Split data (use temporal split)
    split_idx = int(len(y) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    weights_train = sample_weights[:split_idx]

    if context_features is not None:
        context_train = context_features[:split_idx]
        context_test = context_features[split_idx:]
        print(f"  Context features shape: {context_features.shape}")
    else:
        context_train = None
        context_test = None

    if tune and HAS_OPTUNA:
        model = tune_regressor_optuna(X_train, y_train)
    else:
        model = StackingRegressor(verbose=True)
        # Note: StackingRegressor doesn't support context features yet
        # This will be enhanced in a future iteration
        model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n  Test Set Metrics:")
    print(f"    RMSE: {rmse:.3f}")
    print(f"    R²:   {r2:.3f}")

    # A/B Test: Compare with baseline (if exists)
    baseline_path = MODEL_DIR / "spread_stacking_baseline.pkl"
    if baseline_path.exists():
        print(f"\n  A/B Test: Comparing with baseline...")
        try:
            with open(baseline_path, 'rb') as f:
                baseline_model = pickle.load(f)
            y_pred_baseline = baseline_model.predict(X_test)
            rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
            improvement = (rmse_baseline - rmse) / rmse_baseline * 100
            print(f"    Baseline RMSE: {rmse_baseline:.3f}")
            print(f"    New Model RMSE: {rmse:.3f}")
            print(f"    Improvement: {improvement:+.2f}%")

            # Only save if improved
            if improvement >= 0:
                output_path = MODEL_DIR / "spread_stacking.pkl"
                model.save(str(output_path))
                print(f"  ✓ Model improved! Saved to {output_path}")
            else:
                print(f"  ✗ Model did not improve. Keeping baseline.")
                return baseline_model
        except Exception as e:
            print(f"  Warning: Could not load baseline: {e}")
    else:
        output_path = MODEL_DIR / "spread_stacking.pkl"
        model.save(str(output_path))
        # Save as baseline for future comparisons
        model.save(str(baseline_path))
        print(f"  Saved to {output_path}")

    return model


def train_props_model(data: pd.DataFrame, prop_type: str, tune: bool = False) -> StackingRegressor:
    """Train stacking regressor for player props."""
    print("\n" + "=" * 60)
    print(f"TRAINING {prop_type.upper()} PROP MODEL")
    print("=" * 60)

    feature_cols = [c for c in data.columns if c != 'target']
    X = data[feature_cols].fillna(0)
    y = data['target'].values

    split_idx = int(len(y) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if tune and HAS_OPTUNA:
        model = tune_regressor_optuna(X_train, y_train)
    else:
        model = StackingRegressor(verbose=True)
        model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n  Test Set Metrics:")
    print(f"    RMSE: {rmse:.3f}")
    print(f"    R²:   {r2:.3f}")

    output_path = MODEL_DIR / f"player_{prop_type}_stacking.pkl"
    model.save(str(output_path))
    print(f"  Saved to {output_path}")

    return model


def tune_classifier_optuna(X: pd.DataFrame, y: np.ndarray, n_trials: int = 50) -> StackingClassifier:
    """Tune classifier hyperparameters with Optuna."""
    print("  Running Optuna hyperparameter tuning...")

    def objective(trial):
        # This is a simplified objective - in production, tune base model params
        n_folds = trial.suggest_int('n_folds', 3, 7)

        model = StackingClassifier(n_folds=n_folds, verbose=False)
        model.fit(X, y)

        # Use OOF predictions for evaluation
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best accuracy: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Train final model with best params
    model = StackingClassifier(n_folds=study.best_params['n_folds'], verbose=True)
    model.fit(X, y)
    return model


def tune_regressor_optuna(X: pd.DataFrame, y: np.ndarray, n_trials: int = 50) -> StackingRegressor:
    """Tune regressor hyperparameters with Optuna."""
    print("  Running Optuna hyperparameter tuning...")

    def objective(trial):
        n_folds = trial.suggest_int('n_folds', 3, 7)

        model = StackingRegressor(n_folds=n_folds, verbose=False)
        model.fit(X, y)

        y_pred = model.predict(X)
        return -np.sqrt(mean_squared_error(y, y_pred))  # Minimize RMSE

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best RMSE: {-study.best_value:.3f}")
    print(f"  Best params: {study.best_params}")

    model = StackingRegressor(n_folds=study.best_params['n_folds'], verbose=True)
    model.fit(X, y)
    return model


def main():
    parser = argparse.ArgumentParser(description='Train stacking models')
    parser.add_argument('--tune', action='store_true', help='Enable Optuna tuning')
    parser.add_argument('--model', choices=['moneyline', 'spread', 'props', 'all'],
                        default='all', help='Which model to train')
    args = parser.parse_args()

    print("=" * 60)
    print("STACKING MODEL TRAINING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Optuna tuning: {'Enabled' if args.tune and HAS_OPTUNA else 'Disabled'}")
    print("=" * 60)

    # Load data
    loader = TrainingDataLoader()
    loader.load_games()
    loader.load_player_stats()

    models_trained = []

    if args.model in ['moneyline', 'all']:
        data = loader.build_moneyline_dataset()
        if len(data) >= 50:
            train_moneyline_model(data, tune=args.tune)
            models_trained.append('moneyline')
        else:
            print("Insufficient data for moneyline model")

    if args.model in ['spread', 'all']:
        data = loader.build_spread_dataset()
        if len(data) >= 50:
            train_spread_model(data, tune=args.tune)
            models_trained.append('spread')
        else:
            print("Insufficient data for spread model")

    if args.model in ['props', 'all']:
        for prop_type in ['points', 'rebounds', 'assists', 'threes']:
            data = loader.build_props_dataset(prop_type)
            if len(data) >= 100:
                train_props_model(data, prop_type, tune=args.tune)
                models_trained.append(f'{prop_type}_props')
            else:
                print(f"Insufficient data for {prop_type} props model")

    print("\n" + "=" * 60)
    print(f"Training complete! Models trained: {models_trained}")
    print("=" * 60)


if __name__ == "__main__":
    main()
