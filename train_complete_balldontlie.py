"""
Complete NBA Model Training with Balldontlie.io Data

This script trains ALL models using real data from Balldontlie.io:
- Moneyline predictions (ensemble model)
- Spread predictions
- Player prop predictions (points, rebounds, assists, threes, PRA)

Current Season: 2025-26 (started October 2025)

Usage:
    python3 train_complete_balldontlie.py
"""

import os
import sys
import json
import pickle
import warnings
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss,
    brier_score_loss,
)
from sklearn.isotonic import IsotonicRegression  # For probability calibration

# Try to import optional advanced libraries
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Note: CatBoost not installed. Install with: pip install catboost")

try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Note: XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except (ImportError, OSError) as e:
    HAS_LIGHTGBM = False
    print(f"Note: LightGBM not available ({type(e).__name__}). Install with: pip install lightgbm")

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Note: Optuna not installed. Install with: pip install optuna")

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import argparse

warnings.filterwarnings('ignore')

# Model save directory
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Cache directory
CACHE_DIR = Path("data/balldontlie_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Load environment
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

from balldontlie_api import BalldontlieAPI
from model_trainer import EnsembleMoneylineWrapper


# =============================================================================
# SMART IMPUTATION FOR MISSING VALUES
# =============================================================================

# Feature-specific defaults (much better than fillna(0))
FEATURE_DEFAULTS = {
    # Offensive/Defensive Ratings (league average ~114)
    'off_rating': 114.0,
    'def_rating': 114.0,
    'home_off_rating': 114.0,
    'away_off_rating': 114.0,
    'home_def_rating': 114.0,
    'away_def_rating': 114.0,
    'net_rating': 0.0,

    # Win percentages
    'win_pct': 0.5,
    'home_win_pct': 0.5,
    'away_win_pct': 0.5,
    'season_win_pct': 0.5,

    # Shooting percentages
    'fg_pct': 0.46,
    'fg3_pct': 0.36,
    'ft_pct': 0.78,
    'ts_pct': 0.57,
    'efg_pct': 0.53,

    # Per-game stats (league averages 2024-25)
    'pts_avg': 114.0,
    'reb_avg': 44.0,
    'ast_avg': 25.0,
    'stl_avg': 7.5,
    'blk_avg': 5.0,
    'tov_avg': 13.5,

    # Pace
    'pace': 100.0,

    # Rest days (reasonable default)
    'rest_days': 1.0,
    'home_rest_days': 1.0,
    'away_rest_days': 1.0,
}


def smart_fillna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply intelligent default values based on feature type.

    Much better than fillna(0) which gives wrong values for:
    - Ratings (league avg ~114, not 0)
    - Percentages (0.5 is neutral, not 0)
    - Per-game stats (league averages vary)
    """
    result = df.copy()

    for col in result.columns:
        if result[col].isna().any():
            # Check for exact matches first
            col_lower = col.lower()
            if col in FEATURE_DEFAULTS:
                result[col] = result[col].fillna(FEATURE_DEFAULTS[col])
            elif col_lower in FEATURE_DEFAULTS:
                result[col] = result[col].fillna(FEATURE_DEFAULTS[col_lower])
            # Check for partial matches
            elif 'off_rating' in col_lower or 'offensive_rating' in col_lower:
                result[col] = result[col].fillna(114.0)
            elif 'def_rating' in col_lower or 'defensive_rating' in col_lower:
                result[col] = result[col].fillna(114.0)
            elif 'net_rating' in col_lower:
                result[col] = result[col].fillna(0.0)
            elif 'rating' in col_lower:
                result[col] = result[col].fillna(114.0)
            elif 'win_pct' in col_lower or 'winpct' in col_lower:
                result[col] = result[col].fillna(0.5)
            elif 'pct' in col_lower or 'percentage' in col_lower:
                result[col] = result[col].fillna(0.5)
            elif 'pace' in col_lower:
                result[col] = result[col].fillna(100.0)
            elif '_diff' in col_lower or 'diff_' in col_lower:
                result[col] = result[col].fillna(0.0)
            elif 'rest' in col_lower:
                result[col] = result[col].fillna(1.0)
            elif 'pts' in col_lower or 'points' in col_lower:
                result[col] = result[col].fillna(114.0)
            elif 'reb' in col_lower or 'rebounds' in col_lower:
                result[col] = result[col].fillna(44.0)
            elif 'ast' in col_lower or 'assists' in col_lower:
                result[col] = result[col].fillna(25.0)
            else:
                # Fall back to column median or 0
                median = result[col].median()
                result[col] = result[col].fillna(median if pd.notna(median) else 0.0)

    return result


# =============================================================================
# EXPANDING WINDOW CROSS-VALIDATION
# =============================================================================

class ExpandingWindowCV:
    """
    Expanding window cross-validation for time series data.

    Unlike TimeSeriesSplit which uses fixed-size folds, this approach:
    1. Always includes ALL available history in each training fold
    2. Uses a fixed-size test window
    3. Better mimics real betting scenarios where we train on all past data

    Example with 1000 samples and 5 splits:
    - Fold 1: Train on [0:600], Test on [600:680]
    - Fold 2: Train on [0:680], Test on [680:760]
    - Fold 3: Train on [0:760], Test on [760:840]
    - Fold 4: Train on [0:840], Test on [840:920]
    - Fold 5: Train on [0:920], Test on [920:1000]
    """

    def __init__(self, n_splits: int = 5, min_train_size: int = 500, test_size: int = None):
        """
        Initialize the cross-validator.

        Args:
            n_splits: Number of CV splits
            min_train_size: Minimum samples in first training fold
            test_size: Fixed test size (if None, calculated automatically)
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size

    def split(self, X, y=None, groups=None):
        """
        Generate train/test indices for expanding window CV.

        Args:
            X: Feature array or DataFrame
            y: Target array (optional, for API compatibility)
            groups: Group labels (optional, for API compatibility)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        # Calculate test size if not specified
        if self.test_size is None:
            # Divide remaining samples after min_train equally among folds
            remaining = n_samples - self.min_train_size
            test_size = max(50, remaining // self.n_splits)
        else:
            test_size = self.test_size

        # Ensure we have enough samples
        if n_samples < self.min_train_size + test_size:
            raise ValueError(
                f"Not enough samples ({n_samples}) for min_train_size={self.min_train_size} "
                f"and test_size={test_size}"
            )

        for i in range(self.n_splits):
            # Training set: all data from start to train_end
            train_end = self.min_train_size + (i * test_size)
            if train_end >= n_samples:
                break

            # Test set: next test_size samples
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)

            if test_start >= n_samples:
                break

            train_indices = np.arange(train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits."""
        return self.n_splits


# =============================================================================
# HYPERPARAMETER TUNING FOR TEAM MODELS
# =============================================================================

def tune_moneyline_xgb(X_train, y_train, sample_weights=None, n_trials=50):
    """
    Hyperparameter tuning for XGBoost moneyline classifier using Optuna.

    Returns optimal parameters for the model.
    """
    if not HAS_OPTUNA or not HAS_XGBOOST:
        print("  Optuna or XGBoost not available, using defaults")
        return None

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
        }

        model = XGBClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        # Cross-validation with time series split
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            if sample_weights is not None:
                w_tr = sample_weights[train_idx]
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                model.fit(X_tr, y_tr)

            # Use negative log loss (lower is better)
            probs = model.predict_proba(X_val)[:, 1]
            score = log_loss(y_val, probs)
            scores.append(score)

        return np.mean(scores)  # Minimize log loss

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best log loss: {study.best_value:.4f}")
    return study.best_params


def tune_spread_xgb(X_train, y_train, sample_weights=None, n_trials=50):
    """
    Hyperparameter tuning for XGBoost spread regressor using Optuna.

    Returns optimal parameters for the model.
    """
    if not HAS_OPTUNA or not HAS_XGBOOST:
        print("  Optuna or XGBoost not available, using defaults")
        return None

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
        }

        model = XGBRegressor(
            **params,
            random_state=42,
            n_jobs=-1
        )

        # Cross-validation with time series split
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            if sample_weights is not None:
                w_tr = sample_weights[train_idx]
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                model.fit(X_tr, y_tr)

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(rmse)

        return np.mean(scores)  # Minimize RMSE

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best RMSE: {study.best_value:.4f}")
    return study.best_params


# =============================================================================
# BETTING ROI SCORER
# =============================================================================

def betting_roi_scorer(y_true, y_proba, vegas_implied_probs, min_edge=0.05):
    """
    Calculate betting ROI based on edge over Vegas.

    This measures what actually matters: BETTING PROFITABILITY.
    A model could be 55% accurate but unprofitable if it loses on high-confidence bets.

    Args:
        y_true: Actual outcomes (0/1 for moneyline, 1=home win)
        y_proba: Model predicted probabilities for home team
        vegas_implied_probs: Vegas implied probabilities for home team
        min_edge: Minimum edge required to place bet (default 5%)

    Returns:
        ROI as a decimal (0.08 = 8% profit)
    """
    total_wagered = 0
    total_profit = 0

    for actual, model_prob, vegas_prob in zip(y_true, y_proba, vegas_implied_probs):
        # Skip if vegas_prob is invalid
        if vegas_prob is None or vegas_prob <= 0 or vegas_prob >= 1:
            continue

        edge = model_prob - vegas_prob

        if edge >= min_edge:  # Only bet when we have edge on home team
            total_wagered += 1  # Unit bet

            # American odds calculation from implied probability
            # If prob < 0.5, odds are positive: (100/prob) - 100
            # If prob >= 0.5, odds are negative: -100/(1-prob)
            if actual == 1:  # Home team won
                payout = (1 / vegas_prob) - 1  # Approximate odds conversion
                total_profit += payout
            else:
                total_profit -= 1  # Lost the bet

        # Also check for edge betting against home team
        away_model_prob = 1 - model_prob
        away_vegas_prob = 1 - vegas_prob
        away_edge = away_model_prob - away_vegas_prob

        if away_edge >= min_edge:
            total_wagered += 1

            if actual == 0:  # Away team won
                payout = (1 / away_vegas_prob) - 1
                total_profit += payout
            else:
                total_profit -= 1

    if total_wagered == 0:
        return 0.0, 0

    return total_profit / total_wagered, total_wagered


# =============================================================================
# SPREAD ENSEMBLE WRAPPER (for pickling)
# =============================================================================

class SpreadEnsembleWrapper:
    """Wrapper for spread ensemble prediction that can be pickled."""
    def __init__(self, models, weights, scaler, feature_names, metrics):
        self.models = models
        self.weights = weights
        self.scaler = scaler
        self.feature_names = feature_names
        self.training_metrics = metrics

    def predict(self, X):
        X_arr = np.array(X)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1, -1)
        X_scaled = self.scaler.transform(X_arr)
        pred = np.zeros(X_scaled.shape[0])
        for name, model in self.models.items():
            pred += self.weights[name] * model.predict(X_scaled)
        return pred


# =============================================================================
# POSITION DEFENSE CALCULATOR (TIER 2.2)
# =============================================================================

class PositionDefenseCalculator:
    """
    Calculate team defensive efficiency by opponent position.

    Tracks how many points/rebounds/assists/threes each team allows
    to guards, forwards, and centers separately. This enables position-specific
    matchup analysis for more accurate prop predictions.
    """

    # Map detailed positions to G/F/C groups
    POSITION_GROUPS = {
        'G': ['G', 'PG', 'SG', 'G-F', 'PG-SG'],
        'F': ['F', 'SF', 'PF', 'F-G', 'F-C', 'SF-PF', 'PF-SF'],
        'C': ['C', 'C-F', 'C-PF', 'PF-C']
    }

    # League average stats by position (for fallback)
    LEAGUE_AVG = {
        'G': {'pts': 14.5, 'reb': 3.2, 'ast': 4.8, 'fg3m': 1.9},
        'F': {'pts': 12.8, 'reb': 5.4, 'ast': 2.1, 'fg3m': 1.2},
        'C': {'pts': 11.2, 'reb': 8.1, 'ast': 1.8, 'fg3m': 0.4},
    }

    def __init__(self):
        # Structure: {team_id: {date: {pos_group: [list of player stats dicts]}}}
        self.team_position_defense = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def _parse_minutes(self, min_str) -> float:
        """Parse minutes from string format (e.g., '35:24' or '35')."""
        if isinstance(min_str, (int, float)):
            return float(min_str)
        if not min_str:
            return 0.0
        try:
            if ':' in str(min_str):
                parts = str(min_str).split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_str)
        except:
            return 0.0

    def _get_position_group(self, position: str) -> str:
        """Map detailed position to G/F/C group."""
        if not position:
            return 'F'  # Default to Forward
        position = position.upper().strip()
        for group, positions in self.POSITION_GROUPS.items():
            if position in positions:
                return group
        # Try partial matches
        if 'G' in position:
            return 'G'
        if 'C' in position:
            return 'C'
        return 'F'  # Default

    def process_game(self, game_id: int, game_date: str, home_team_id: int,
                     away_team_id: int, player_stats: List[Dict]):
        """
        Process a game's player stats to update position defense data.

        For each player who scored X points against team Y, we track that
        team Y allowed X points to that player's position.

        Args:
            game_id: Balldontlie game ID
            game_date: Date string (YYYY-MM-DD)
            home_team_id: Home team ID
            away_team_id: Away team ID
            player_stats: List of player stat dicts from box score
        """
        for ps in player_stats:
            # Get player's team
            player_team_id = ps.get('team', {}).get('id') or ps.get('team_id')
            if not player_team_id:
                continue

            # Determine opponent team
            if player_team_id == home_team_id:
                opponent_id = away_team_id
            elif player_team_id == away_team_id:
                opponent_id = home_team_id
            else:
                continue  # Skip if can't determine teams

            # Get player position
            player_info = ps.get('player', {})
            position = player_info.get('position', '')
            pos_group = self._get_position_group(position)

            # Parse minutes - only track players with significant minutes
            minutes = self._parse_minutes(ps.get('min', '0'))
            if minutes < 10:
                continue  # Skip low-minute players for cleaner signal

            # Extract stats
            stat_record = {
                'pts': ps.get('pts', 0) or 0,
                'reb': ps.get('reb', 0) or 0,
                'ast': ps.get('ast', 0) or 0,
                'fg3m': ps.get('fg3m', 0) or 0,
                'min': minutes
            }

            # Track what the OPPONENT TEAM allowed to this position
            # opponent_id allowed these stats to pos_group on game_date
            self.team_position_defense[opponent_id][game_date][pos_group].append(stat_record)

    def get_position_defense_before_date(self, team_id: int, game_date: str,
                                          player_position: str,
                                          min_games: int = 5) -> Dict:
        """
        Get team's defensive stats vs a position BEFORE a specific date.

        This is point-in-time safe - only uses data available before game_date.

        Args:
            team_id: Team whose defense we're analyzing
            game_date: Date to get stats before (YYYY-MM-DD)
            player_position: Position of the player we're predicting for
            min_games: Minimum games needed for valid stats

        Returns:
            Dict with position-specific defensive features
        """
        pos_group = self._get_position_group(player_position)

        # Collect all stats from games BEFORE this date
        pts_list, reb_list, ast_list, fg3m_list = [], [], [], []

        team_data = self.team_position_defense.get(team_id, {})
        for date, pos_data in sorted(team_data.items()):
            if date >= game_date:
                break  # Point-in-time: only use earlier dates

            if pos_group in pos_data:
                for stat in pos_data[pos_group]:
                    pts_list.append(stat['pts'])
                    reb_list.append(stat['reb'])
                    ast_list.append(stat['ast'])
                    fg3m_list.append(stat['fg3m'])

        # Also get all positions for context
        all_pts_g, all_pts_f, all_pts_c = [], [], []
        all_reb_g, all_reb_f, all_reb_c = [], [], []
        all_ast_g, all_ast_f, all_ast_c = [], [], []
        all_fg3m_g, all_fg3m_f, all_fg3m_c = [], [], []

        for date, pos_data in sorted(team_data.items()):
            if date >= game_date:
                break
            for pg, stats in pos_data.items():
                for stat in stats:
                    if pg == 'G':
                        all_pts_g.append(stat['pts'])
                        all_reb_g.append(stat['reb'])
                        all_ast_g.append(stat['ast'])
                        all_fg3m_g.append(stat['fg3m'])
                    elif pg == 'F':
                        all_pts_f.append(stat['pts'])
                        all_reb_f.append(stat['reb'])
                        all_ast_f.append(stat['ast'])
                        all_fg3m_f.append(stat['fg3m'])
                    elif pg == 'C':
                        all_pts_c.append(stat['pts'])
                        all_reb_c.append(stat['reb'])
                        all_ast_c.append(stat['ast'])
                        all_fg3m_c.append(stat['fg3m'])

        # Calculate averages with fallbacks to league average
        def safe_mean(lst, default):
            return np.mean(lst) if len(lst) >= min_games else default

        def safe_std(lst, default=5.0):
            return np.std(lst) if len(lst) >= min_games else default

        league_avg = self.LEAGUE_AVG[pos_group]

        # Features for opponent defense vs ALL positions
        features = {
            # What opponent allows to Guards
            'opp_pts_allowed_to_guards': safe_mean(all_pts_g, self.LEAGUE_AVG['G']['pts']),
            'opp_reb_allowed_to_guards': safe_mean(all_reb_g, self.LEAGUE_AVG['G']['reb']),
            'opp_ast_allowed_to_guards': safe_mean(all_ast_g, self.LEAGUE_AVG['G']['ast']),
            'opp_fg3m_allowed_to_guards': safe_mean(all_fg3m_g, self.LEAGUE_AVG['G']['fg3m']),

            # What opponent allows to Forwards
            'opp_pts_allowed_to_forwards': safe_mean(all_pts_f, self.LEAGUE_AVG['F']['pts']),
            'opp_reb_allowed_to_forwards': safe_mean(all_reb_f, self.LEAGUE_AVG['F']['reb']),
            'opp_ast_allowed_to_forwards': safe_mean(all_ast_f, self.LEAGUE_AVG['F']['ast']),
            'opp_fg3m_allowed_to_forwards': safe_mean(all_fg3m_f, self.LEAGUE_AVG['F']['fg3m']),

            # What opponent allows to Centers
            'opp_pts_allowed_to_centers': safe_mean(all_pts_c, self.LEAGUE_AVG['C']['pts']),
            'opp_reb_allowed_to_centers': safe_mean(all_reb_c, self.LEAGUE_AVG['C']['reb']),
            'opp_ast_allowed_to_centers': safe_mean(all_ast_c, self.LEAGUE_AVG['C']['ast']),
            'opp_fg3m_allowed_to_centers': safe_mean(all_fg3m_c, self.LEAGUE_AVG['C']['fg3m']),

            # Variance in points allowed to this position (defensive consistency)
            'opp_pts_vs_pos_std': safe_std(pts_list, 5.0),
        }

        # Calculate matchup advantage: how much more/less does opponent allow vs league avg
        opp_pts_to_pos = safe_mean(pts_list, league_avg['pts'])
        features['opp_pts_vs_pos_diff'] = (opp_pts_to_pos - league_avg['pts']) / max(league_avg['pts'], 1)

        opp_reb_to_pos = safe_mean(reb_list, league_avg['reb'])
        features['opp_reb_vs_pos_diff'] = (opp_reb_to_pos - league_avg['reb']) / max(league_avg['reb'], 1)

        opp_ast_to_pos = safe_mean(ast_list, league_avg['ast'])
        features['opp_ast_vs_pos_diff'] = (opp_ast_to_pos - league_avg['ast']) / max(league_avg['ast'], 1)

        opp_fg3m_to_pos = safe_mean(fg3m_list, league_avg['fg3m'])
        features['opp_fg3m_vs_pos_diff'] = (opp_fg3m_to_pos - league_avg['fg3m']) / max(league_avg['fg3m'], 0.5)

        return features


# =============================================================================
# DATA COLLECTION
# =============================================================================

class ComprehensiveDataCollector:
    """
    Collects game data AND player statistics from Balldontlie.io API.
    """

    def __init__(self):
        self.api = BalldontlieAPI()
        self.teams_cache = {}
        self.players_cache = {}

    def get_all_teams(self) -> Dict[int, Dict]:
        """Fetch and cache all NBA teams."""
        if not self.teams_cache:
            teams = self.api.get_teams()
            for team in teams:
                self.teams_cache[team['id']] = team
        return self.teams_cache

    def fetch_season_games(self, season: int) -> List[Dict]:
        """
        Fetch all completed games for a season.

        Args:
            season: Season year (e.g., 2025 for 2025-26 season)
        """
        cache_file = CACHE_DIR / f"games_{season}_full.json"

        # Check cache
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
                if cached.get('complete', False) or cached.get('partial_through'):
                    print(f"  Loaded {len(cached['games'])} games from cache for {season}")
                    return cached['games']

        print(f"  Fetching games for {season} season from API...")
        all_games = []

        # Season typically runs Oct-Jun
        # For current season (2025), we're in December
        if season == 2025:
            # Current season - only fetch up to today
            start_date = datetime(2025, 10, 1)
            end_date = datetime.now()
        else:
            start_date = datetime(season, 10, 1)
            end_date = datetime(season + 1, 6, 30)

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")

            try:
                day_games = self.api.get_games(dates=[date_str])

                for game in day_games:
                    if game.get('status') == 'Final' and game.get('home_team_score', 0) > 0:
                        # Check if already added
                        game_id = game.get('id')
                        if not any(g.get('id') == game_id for g in all_games):
                            all_games.append(game)

            except Exception as e:
                print(f"    Error fetching {date_str}: {e}")

            current_date += timedelta(days=1)

            # Progress update
            if (current_date - start_date).days % 30 == 0:
                print(f"    Progress: {current_date.strftime('%Y-%m-%d')}, {len(all_games)} games...")

            # Rate limiting
            time.sleep(0.05)

        # Save to cache
        is_complete = season < 2025 or (season == 2025 and datetime.now().month > 6)
        with open(cache_file, 'w') as f:
            json.dump({
                'season': season,
                'games': all_games,
                'complete': is_complete,
                'partial_through': datetime.now().strftime('%Y-%m-%d') if not is_complete else None,
                'fetched_at': datetime.now().isoformat()
            }, f)

        print(f"    Completed: {len(all_games)} games for {season}")
        return all_games

    def fetch_player_stats_for_games(
        self,
        game_ids: List[int],
        batch_size: int = 25,
    ) -> Dict[int, List[Dict]]:
        """
        Fetch player statistics for multiple games with per-game caching.

        Uses individual cache files per game for resumable fetching.
        Falls back to box_score cache files if available.

        Args:
            game_ids: List of game IDs
            batch_size: Number of games per API request

        Returns:
            Dict mapping game_id -> list of player stats
        """
        all_stats = {}
        games_needing_fetch = []

        # First, load from individual cache files (box_score_*.json)
        for game_id in game_ids:
            cache_file = CACHE_DIR / f"box_score_{game_id}.json"
            player_stats_file = CACHE_DIR / f"player_stats_{game_id}.json"

            # Try player_stats cache first
            if player_stats_file.exists():
                try:
                    with open(player_stats_file) as f:
                        cached = json.load(f)
                        if cached:
                            all_stats[game_id] = cached
                            continue
                except (json.JSONDecodeError, IOError):
                    pass

            # Try box_score cache (has player stats embedded)
            # Box score format: {player_id: {player: {...}, pts, reb, ast, fg3m, min, team_id}}
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        box_score = json.load(f)

                        # Box score is keyed by player_id
                        if isinstance(box_score, dict) and box_score:
                            player_list = []
                            for player_id, p in box_score.items():
                                if isinstance(p, dict) and 'pts' in p:
                                    player_list.append({
                                        'id': p.get('player', {}).get('id', int(player_id)),
                                        'min': p.get('min', '0'),
                                        'pts': p.get('pts', 0),
                                        'reb': p.get('reb', 0),
                                        'ast': p.get('ast', 0),
                                        'fg3m': p.get('fg3m', 0),
                                        'fgm': p.get('fgm', 0),
                                        'fga': p.get('fga', 0),
                                        'ftm': p.get('ftm', 0),
                                        'fta': p.get('fta', 0),
                                        'stl': p.get('stl', 0),
                                        'blk': p.get('blk', 0),
                                        'turnover': p.get('turnover', 0),
                                        'oreb': p.get('oreb', 0),
                                        'dreb': p.get('dreb', 0),
                                        'player': p.get('player', {}),
                                        'team': {'id': p.get('team_id')},
                                        'game': {'id': game_id}
                                    })
                            if player_list:
                                all_stats[game_id] = player_list
                                continue
                except (json.JSONDecodeError, IOError):
                    pass

            # Need to fetch this game
            games_needing_fetch.append(game_id)

        cached_count = len(all_stats)
        print(f"  Loaded {cached_count} games from cache, need to fetch {len(games_needing_fetch)} games...")

        if not games_needing_fetch:
            return all_stats

        # Fetch missing games in batches
        for i in range(0, len(games_needing_fetch), batch_size):
            batch = games_needing_fetch[i:i + batch_size]

            try:
                stats = self.api.get_player_stats(game_ids=batch, per_page=100)

                # Group stats by game_id
                batch_stats = {}
                for stat in stats:
                    game_id = stat.get('game', {}).get('id')
                    if game_id:
                        if game_id not in batch_stats:
                            batch_stats[game_id] = []
                        batch_stats[game_id].append(stat)

                # Cache each game individually and add to results
                for game_id, game_stats in batch_stats.items():
                    cache_file = CACHE_DIR / f"player_stats_{game_id}.json"
                    with open(cache_file, 'w') as f:
                        json.dump(game_stats, f)
                    all_stats[game_id] = game_stats

            except Exception as e:
                print(f"    Error fetching batch {i//batch_size}: {e}")

            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(games_needing_fetch):
                fetched = min(i + batch_size, len(games_needing_fetch))
                print(f"    Progress: {fetched}/{len(games_needing_fetch)} games fetched (total: {len(all_stats)})")

            time.sleep(0.1)

        print(f"  Total games with player stats: {len(all_stats)}")
        return all_stats

    def fetch_season_averages(self, season: int, player_ids: List[int]) -> Dict[int, Dict]:
        """
        Fetch season averages for players.

        Args:
            season: Season year
            player_ids: List of player IDs

        Returns:
            Dict mapping player_id -> season averages
        """
        cache_file = CACHE_DIR / f"season_averages_{season}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
                # Convert string keys back to int
                return {int(k): v for k, v in cached.items()}

        print(f"  Fetching season averages for {len(player_ids)} players...")
        averages = {}

        # Batch requests
        batch_size = 25
        for i in range(0, len(player_ids), batch_size):
            batch = player_ids[i:i + batch_size]

            try:
                avgs = self.api.get_season_averages(player_ids=batch, season=season)

                for avg in avgs:
                    pid = avg.get('player_id')
                    if pid:
                        averages[pid] = avg

            except Exception as e:
                print(f"    Error fetching batch {i//batch_size}: {e}")

            time.sleep(0.1)

        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(averages, f)

        return averages


# =============================================================================
# FEATURE GENERATION
# =============================================================================

class TeamStatsCalculator:
    """Calculate rolling team statistics with point-in-time correctness."""

    def __init__(self, window: int = 10):
        self.window = window
        self.team_games = defaultdict(list)

    def add_game(self, game: Dict):
        """Add a game to the historical record."""
        game_date = game.get('date', '')
        if isinstance(game_date, str) and 'T' in game_date:
            game_date = game_date.split('T')[0]

        home_team_id = game.get('home_team', {}).get('id')
        away_team_id = game.get('visitor_team', {}).get('id')
        home_team_abbrev = game.get('home_team', {}).get('abbreviation', '')
        away_team_abbrev = game.get('visitor_team', {}).get('abbreviation', '')
        home_score = game.get('home_team_score', 0)
        away_score = game.get('visitor_team_score', 0)

        if not all([home_team_id, away_team_id, game_date, home_score]):
            return

        # Home team stats - venue is home team's arena
        self.team_games[home_team_id].append((game_date, {
            'opponent_id': away_team_id,
            'opponent_abbrev': away_team_abbrev,
            'venue_abbrev': home_team_abbrev,  # Game was at home team's venue
            'is_home': True,
            'pts': home_score,
            'pts_allowed': away_score,
            'win': home_score > away_score,
            'point_diff': home_score - away_score,
        }))

        # Away team stats - venue is home team's arena (where away team traveled to)
        self.team_games[away_team_id].append((game_date, {
            'opponent_id': home_team_id,
            'opponent_abbrev': home_team_abbrev,
            'venue_abbrev': home_team_abbrev,  # Game was at home team's venue (away traveled there)
            'is_home': False,
            'pts': away_score,
            'pts_allowed': home_score,
            'win': away_score > home_score,
            'point_diff': away_score - home_score,
        }))

    def get_team_stats_before_date(self, team_id: int, date: str, min_games: int = 5) -> Optional[Dict]:
        """Get team statistics before a specific date with enhanced defensive metrics."""
        if team_id not in self.team_games:
            return None

        games = [(d, s) for d, s in self.team_games[team_id] if d < date]
        if len(games) < min_games:
            return None

        games.sort(key=lambda x: x[0], reverse=True)
        recent = games[:self.window]
        all_games = games

        # Calculate points allowed stats for defensive rating
        pts_allowed_all = [g['pts_allowed'] for _, g in all_games]
        pts_allowed_recent = [g['pts_allowed'] for _, g in recent]

        # Calculate pace (possessions per game approximation)
        # Pace = (pts scored + pts allowed) / 2 - approximates possessions
        pace_all = [(g['pts'] + g['pts_allowed']) / 2 for _, g in all_games]
        pace_recent = [(g['pts'] + g['pts_allowed']) / 2 for _, g in recent]

        # Calculate rebound differential (if available, else estimate)
        # For now estimate based on point differential correlation
        reb_factor = np.mean([g['point_diff'] for _, g in recent]) * 0.15  # Rough correlation

        # Calculate stats
        return {
            'season_games': len(all_games),
            'season_win_pct': np.mean([g['win'] for _, g in all_games]),
            'season_pts_avg': np.mean([g['pts'] for _, g in all_games]),
            'recent_win_pct': np.mean([g['win'] for _, g in recent]),
            'recent_pts_avg': np.mean([g['pts'] for _, g in recent]),
            'recent_point_diff': np.mean([g['point_diff'] for _, g in recent]),
            'home_win_pct': np.mean([g['win'] for _, g in all_games if g['is_home']]) if any(g['is_home'] for _, g in all_games) else 0.5,
            'away_win_pct': np.mean([g['win'] for _, g in all_games if not g['is_home']]) if any(not g['is_home'] for _, g in all_games) else 0.5,
            'home_pts_avg': np.mean([g['pts'] for _, g in all_games if g['is_home']]) if any(g['is_home'] for _, g in all_games) else 100,
            'away_pts_avg': np.mean([g['pts'] for _, g in all_games if not g['is_home']]) if any(not g['is_home'] for _, g in all_games) else 100,
            'off_rating': np.mean([g['pts'] for _, g in recent]),
            'def_rating': np.mean([g['pts_allowed'] for _, g in recent]),
            'net_rating': np.mean([g['point_diff'] for _, g in recent]),
            # NEW: Enhanced defensive metrics for player props
            'pts_allowed_avg': np.mean(pts_allowed_all),
            'pts_allowed_recent': np.mean(pts_allowed_recent),
            'pts_allowed_std': np.std(pts_allowed_recent) if len(pts_allowed_recent) > 1 else 5.0,
            'pace': np.mean(pace_recent),
            'pace_season': np.mean(pace_all),
            'reb_diff_factor': reb_factor,
            # Defensive strength relative to league average (114 pts)
            'def_strength': (np.mean(pts_allowed_recent) - 114.0) / 10.0,  # Positive = bad defense
            # Home/away defensive splits
            'home_def_rating': np.mean([g['pts_allowed'] for _, g in all_games if g['is_home']]) if any(g['is_home'] for _, g in all_games) else 112,
            'away_def_rating': np.mean([g['pts_allowed'] for _, g in all_games if not g['is_home']]) if any(not g['is_home'] for _, g in all_games) else 114,
        }

    def get_last_game_info(self, team_id: int, before_date: str) -> Optional[Dict]:
        """
        Get information about a team's last game before a specific date.
        Used for travel/fatigue calculations.

        Returns:
            Dict with 'date', 'venue_abbrev', 'is_home', 'days_rest', 'is_back_to_back'
        """
        if team_id not in self.team_games:
            return None

        games = [(d, s) for d, s in self.team_games[team_id] if d < before_date]
        if not games:
            return None

        games.sort(key=lambda x: x[0], reverse=True)
        last_game_date, last_game_stats = games[0]

        # Calculate days rest
        try:
            current = datetime.strptime(before_date, "%Y-%m-%d")
            last = datetime.strptime(last_game_date, "%Y-%m-%d")
            days_rest = (current - last).days
        except:
            days_rest = 2  # Default

        return {
            'date': last_game_date,
            'venue_abbrev': last_game_stats.get('venue_abbrev', ''),
            'is_home': last_game_stats.get('is_home', False),
            'opponent_id': last_game_stats.get('opponent_id'),
            'opponent_abbrev': last_game_stats.get('opponent_abbrev', ''),
            'days_rest': days_rest,
            'is_back_to_back': days_rest <= 1,
        }


# =============================================================================
# ELO RATING SYSTEM
# =============================================================================

class EloRatingSystem:
    """
    Simple but effective Elo rating system for NBA teams.
    Based on FiveThirtyEight methodology - proven at 65-68% accuracy.
    """

    def __init__(self, k_factor: float = 20.0, home_advantage: float = 100.0,
                 initial_rating: float = 1500.0):
        """
        Args:
            k_factor: How quickly ratings adjust (higher = more reactive)
            home_advantage: Elo points added to home team's expected score
            initial_rating: Starting rating for all teams
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.ratings = defaultdict(lambda: initial_rating)
        self.rating_history = defaultdict(list)  # Track rating over time

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected win probability for team A vs team B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, home_team_id: int, away_team_id: int,
                       home_score: int, away_score: int, game_date: str):
        """
        Update Elo ratings after a game.

        Args:
            home_team_id: ID of home team
            away_team_id: ID of away team
            home_score: Points scored by home team
            away_score: Points scored by away team
            game_date: Date of game (for history tracking)
        """
        home_rating = self.ratings[home_team_id]
        away_rating = self.ratings[away_team_id]

        # Add home court advantage to home team's effective rating
        home_effective = home_rating + self.home_advantage

        # Calculate expected scores
        home_expected = self.expected_score(home_effective, away_rating)
        away_expected = 1.0 - home_expected

        # Actual outcome (1 for win, 0 for loss, 0.5 for tie)
        if home_score > away_score:
            home_actual, away_actual = 1.0, 0.0
        elif away_score > home_score:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5

        # Margin of victory multiplier (bigger wins = bigger rating changes)
        mov = abs(home_score - away_score)
        mov_multiplier = np.log(mov + 1) * 0.7 + 0.6  # Ranges from ~0.6 to ~2.5

        # Update ratings
        k_adjusted = self.k_factor * mov_multiplier
        self.ratings[home_team_id] += k_adjusted * (home_actual - home_expected)
        self.ratings[away_team_id] += k_adjusted * (away_actual - away_expected)

        # Track history
        self.rating_history[home_team_id].append((game_date, self.ratings[home_team_id]))
        self.rating_history[away_team_id].append((game_date, self.ratings[away_team_id]))

    def get_rating(self, team_id: int) -> float:
        """Get current Elo rating for a team."""
        return self.ratings[team_id]

    def get_rating_before_date(self, team_id: int, date: str) -> float:
        """Get team's Elo rating before a specific date (point-in-time)."""
        if team_id not in self.rating_history:
            return self.initial_rating

        history = self.rating_history[team_id]
        # Find last rating before this date
        rating_before = self.initial_rating
        for game_date, rating in history:
            if game_date < date:
                rating_before = rating
            else:
                break
        return rating_before

    def predict_win_probability(self, home_team_id: int, away_team_id: int,
                                 before_date: str = None) -> float:
        """
        Predict home team win probability.

        Args:
            home_team_id: ID of home team
            away_team_id: ID of away team
            before_date: If provided, use ratings as of this date

        Returns:
            Home team win probability (0.0 to 1.0)
        """
        if before_date:
            home_rating = self.get_rating_before_date(home_team_id, before_date)
            away_rating = self.get_rating_before_date(away_team_id, before_date)
        else:
            home_rating = self.ratings[home_team_id]
            away_rating = self.ratings[away_team_id]

        home_effective = home_rating + self.home_advantage
        return self.expected_score(home_effective, away_rating)

    def get_spread_prediction(self, home_team_id: int, away_team_id: int,
                               before_date: str = None) -> float:
        """
        Predict point spread (positive = home favored).

        Elo difference of 100 points ≈ 3 point spread
        This is calibrated to typical NBA scoring.
        """
        if before_date:
            home_rating = self.get_rating_before_date(home_team_id, before_date)
            away_rating = self.get_rating_before_date(away_team_id, before_date)
        else:
            home_rating = self.ratings[home_team_id]
            away_rating = self.ratings[away_team_id]

        # Add home court advantage
        elo_diff = (home_rating + self.home_advantage) - away_rating

        # Convert to spread: ~100 Elo = 3 points
        spread = elo_diff * 0.03
        return spread


# =============================================================================
# TIER 2.1: NBA ARENA DATA (Travel & Fatigue Features)
# =============================================================================

# Arena coordinates (lat, lon) and altitude (feet) for travel calculations
# Timezone as UTC offset during regular season (standard time)
NBA_ARENA_DATA = {
    # Atlantic Division
    'BOS': {'coords': (42.366, -71.062), 'altitude': 20, 'timezone': -5, 'name': 'TD Garden'},
    'BKN': {'coords': (40.683, -73.976), 'altitude': 30, 'timezone': -5, 'name': 'Barclays Center'},
    'NYK': {'coords': (40.751, -73.994), 'altitude': 33, 'timezone': -5, 'name': 'Madison Square Garden'},
    'PHI': {'coords': (39.901, -75.172), 'altitude': 39, 'timezone': -5, 'name': 'Wells Fargo Center'},
    'TOR': {'coords': (43.643, -79.379), 'altitude': 249, 'timezone': -5, 'name': 'Scotiabank Arena'},

    # Central Division
    'CHI': {'coords': (41.881, -87.674), 'altitude': 594, 'timezone': -6, 'name': 'United Center'},
    'CLE': {'coords': (41.497, -81.688), 'altitude': 653, 'timezone': -5, 'name': 'Rocket Mortgage FieldHouse'},
    'DET': {'coords': (42.341, -83.055), 'altitude': 600, 'timezone': -5, 'name': 'Little Caesars Arena'},
    'IND': {'coords': (39.764, -86.156), 'altitude': 715, 'timezone': -5, 'name': 'Gainbridge Fieldhouse'},
    'MIL': {'coords': (43.045, -87.917), 'altitude': 617, 'timezone': -6, 'name': 'Fiserv Forum'},

    # Southeast Division
    'ATL': {'coords': (33.757, -84.396), 'altitude': 1050, 'timezone': -5, 'name': 'State Farm Arena'},
    'CHA': {'coords': (35.225, -80.839), 'altitude': 751, 'timezone': -5, 'name': 'Spectrum Center'},
    'MIA': {'coords': (25.781, -80.188), 'altitude': 10, 'timezone': -5, 'name': 'Kaseya Center'},
    'ORL': {'coords': (28.539, -81.384), 'altitude': 82, 'timezone': -5, 'name': 'Amway Center'},
    'WAS': {'coords': (38.898, -77.021), 'altitude': 50, 'timezone': -5, 'name': 'Capital One Arena'},

    # Northwest Division
    'DEN': {'coords': (39.749, -105.008), 'altitude': 5280, 'timezone': -7, 'name': 'Ball Arena'},  # HIGH ALTITUDE!
    'MIN': {'coords': (44.979, -93.276), 'altitude': 830, 'timezone': -6, 'name': 'Target Center'},
    'OKC': {'coords': (35.463, -97.515), 'altitude': 1201, 'timezone': -6, 'name': 'Paycom Center'},
    'POR': {'coords': (45.532, -122.667), 'altitude': 77, 'timezone': -8, 'name': 'Moda Center'},
    'UTA': {'coords': (40.768, -111.901), 'altitude': 4327, 'timezone': -7, 'name': 'Delta Center'},  # HIGH ALTITUDE!

    # Pacific Division
    'GSW': {'coords': (37.768, -122.388), 'altitude': 13, 'timezone': -8, 'name': 'Chase Center'},
    'LAC': {'coords': (34.043, -118.267), 'altitude': 270, 'timezone': -8, 'name': 'Crypto.com Arena'},
    'LAL': {'coords': (34.043, -118.267), 'altitude': 270, 'timezone': -8, 'name': 'Crypto.com Arena'},
    'PHX': {'coords': (33.446, -112.071), 'altitude': 1086, 'timezone': -7, 'name': 'Footprint Center'},
    'SAC': {'coords': (38.580, -121.500), 'altitude': 30, 'timezone': -8, 'name': 'Golden 1 Center'},

    # Southwest Division
    'DAL': {'coords': (32.790, -96.810), 'altitude': 430, 'timezone': -6, 'name': 'American Airlines Center'},
    'HOU': {'coords': (29.751, -95.362), 'altitude': 50, 'timezone': -6, 'name': 'Toyota Center'},
    'MEM': {'coords': (35.138, -90.051), 'altitude': 337, 'timezone': -6, 'name': 'FedExForum'},
    'NOP': {'coords': (29.949, -90.082), 'altitude': 3, 'timezone': -6, 'name': 'Smoothie King Center'},
    'SAS': {'coords': (29.427, -98.437), 'altitude': 650, 'timezone': -6, 'name': 'Frost Bank Center'},
}

# Team abbreviation mappings (handle variations)
TEAM_ABBREV_MAP = {
    'NJN': 'BKN', 'SEA': 'OKC', 'VAN': 'MEM', 'NOH': 'NOP', 'NOK': 'NOP',
    'CHA': 'CHA', 'CHH': 'CHA',  # Handle Charlotte Hornets variations
}


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate great-circle distance between two points in miles.

    Args:
        coord1: (latitude, longitude) of first point
        coord2: (latitude, longitude) of second point

    Returns:
        Distance in miles
    """
    from math import radians, cos, sin, asin, sqrt

    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Earth's radius in miles
    r = 3956

    return c * r


def calc_travel_fatigue_features(last_game_team_abbrev: str, current_game_team_abbrev: str,
                                  days_rest: int, is_back_to_back: bool) -> Dict[str, float]:
    """
    TIER 2.1: Calculate travel-related fatigue features.

    Args:
        last_game_team_abbrev: Team abbreviation of venue for previous game
        current_game_team_abbrev: Team abbreviation of venue for current game
        days_rest: Days since last game
        is_back_to_back: Whether this is a back-to-back

    Returns:
        Dictionary of travel fatigue features
    """
    # Normalize team abbreviations
    last_team = TEAM_ABBREV_MAP.get(last_game_team_abbrev, last_game_team_abbrev)
    current_team = TEAM_ABBREV_MAP.get(current_game_team_abbrev, current_game_team_abbrev)

    # Default values
    result = {
        'travel_distance': 0.0,
        'timezone_change': 0,
        'altitude_change': 0,
        'altitude_disadvantage': 0.0,  # Negative effect when visiting high altitude
        'travel_fatigue_score': 0.0,
        'coast_to_coast': 0,  # 1 if cross-country trip
    }

    # Get arena data
    last_arena = NBA_ARENA_DATA.get(last_team)
    current_arena = NBA_ARENA_DATA.get(current_team)

    if not last_arena or not current_arena:
        return result

    # Calculate travel distance
    distance = haversine_distance(last_arena['coords'], current_arena['coords'])
    result['travel_distance'] = round(distance, 1)

    # Timezone change (westbound travel is harder to recover from)
    tz_change = current_arena['timezone'] - last_arena['timezone']
    result['timezone_change'] = tz_change

    # Altitude change (feet)
    alt_change = current_arena['altitude'] - last_arena['altitude']
    result['altitude_change'] = alt_change

    # High altitude disadvantage (Denver=5280ft, Utah=4327ft are significant)
    if current_arena['altitude'] >= 4000:
        result['altitude_disadvantage'] = min(current_arena['altitude'] / 10000, 0.5)  # Max 0.5 impact

    # Coast-to-coast indicator (>2000 miles)
    if distance > 2000:
        result['coast_to_coast'] = 1

    # Comprehensive fatigue score (0-1 scale)
    # Factors: distance, timezone, altitude, rest days
    distance_factor = min(distance / 3000, 1.0) * 0.4  # Max 40% from distance
    tz_factor = min(abs(tz_change) / 3, 1.0) * 0.25     # Max 25% from timezone
    alt_factor = result['altitude_disadvantage'] * 0.2  # Max 10% from altitude
    rest_factor = (0.3 if is_back_to_back else 0.15 if days_rest == 1 else 0) * 0.15

    fatigue_score = distance_factor + tz_factor + alt_factor + rest_factor

    # Rest mitigates fatigue
    if days_rest >= 2:
        fatigue_score *= 0.6  # 40% reduction with 2+ days rest
    elif days_rest == 1:
        fatigue_score *= 0.85  # 15% reduction with 1 day rest

    result['travel_fatigue_score'] = round(min(fatigue_score, 1.0), 3)

    return result


# =============================================================================
# SCHEDULE SPOT ANALYSIS
# =============================================================================

# Teams considered "elite" for lookahead/letdown detection
ELITE_TEAMS = {'BOS', 'DEN', 'MIL', 'PHX', 'LAL', 'GSW', 'MIA', 'PHI', 'CLE', 'OKC', 'MIN', 'NYK'}


def analyze_schedule_spots(
    team_id: int,
    team_abbrev: str,
    game_date: str,
    opponent_abbrev: str,
    team_calc: 'TeamStatsCalculator',
    is_home: bool,
    future_games: List[Dict] = None
) -> Dict[str, int]:
    """
    Analyze schedule-based situational spots that affect team performance.

    Schedule spots are proven market inefficiencies:
    - Letdown: After emotional/big win, team underperforms vs weaker opponent
    - Trap: Playing weak team before tough game (lookahead)
    - Sandwich: Between two tough opponents
    - Road trip fatigue: 3rd+ game on road trip
    - Revenge: Playing team that beat them recently

    Args:
        team_id: Team ID
        team_abbrev: Team abbreviation
        game_date: Current game date
        opponent_abbrev: Opponent abbreviation
        team_calc: TeamStatsCalculator for game history
        is_home: Whether team is home
        future_games: List of team's upcoming games (if available)

    Returns:
        Dictionary of schedule spot indicators (0 or 1)
    """
    spots = {
        'letdown_spot': 0,           # After big win, playing weaker team
        'trap_game': 0,              # Weak team before tough opponent (lookahead)
        'sandwich_game': 0,          # Between two tough games
        'road_trip_fatigue': 0,      # 3rd+ game of road trip
        'revenge_game': 0,           # Playing team that beat them recently
        'long_homestand': 0,         # 4th+ consecutive home game (complacency)
        'early_season_variance': 0,  # First 10 games of season (volatile)
        'schedule_spot_score': 0.0,  # Combined situational score
    }

    # Get team's recent game history
    last_game = team_calc.get_last_game_info(team_id, game_date)
    if not last_game:
        return spots

    # Check history of games
    if team_id not in team_calc.team_games:
        return spots

    recent_games = [(d, s) for d, s in team_calc.team_games[team_id] if d < game_date]
    recent_games.sort(key=lambda x: x[0], reverse=True)

    if not recent_games:
        return spots

    # === LETDOWN SPOT ===
    # After big win (15+ points), playing a weaker opponent
    if len(recent_games) >= 1:
        last_game_date, last_game_stats = recent_games[0]
        if last_game_stats.get('win', False) and last_game_stats.get('point_diff', 0) >= 15:
            # Was it against a good team? (elite opponent)
            last_opp = last_game_stats.get('opponent_abbrev', '')
            if last_opp in ELITE_TEAMS:
                # Now playing non-elite opponent - potential letdown
                if opponent_abbrev not in ELITE_TEAMS:
                    spots['letdown_spot'] = 1

    # === TRAP GAME ===
    # Weak opponent before elite opponent (if we have future schedule)
    # For training data, we can check if next game was vs elite team
    # This is tricky for point-in-time data, so we'll use a proxy

    # === ROAD TRIP FATIGUE ===
    # Count consecutive road games
    if not is_home:
        consecutive_road = 0
        for _, game_stats in recent_games[:5]:
            if not game_stats.get('is_home', True):
                consecutive_road += 1
            else:
                break
        if consecutive_road >= 2:  # This would be 3rd road game
            spots['road_trip_fatigue'] = 1

    # === LONG HOMESTAND (complacency) ===
    if is_home:
        consecutive_home = 0
        for _, game_stats in recent_games[:6]:
            if game_stats.get('is_home', False):
                consecutive_home += 1
            else:
                break
        if consecutive_home >= 3:  # This would be 4th home game
            spots['long_homestand'] = 1

    # === REVENGE GAME ===
    # Check if opponent beat this team in last 30 days
    for game_date_str, game_stats in recent_games[:15]:
        opp_abbrev = game_stats.get('opponent_abbrev', '')
        if opp_abbrev == opponent_abbrev:
            if not game_stats.get('win', True):  # Lost to this team
                # Check if it was a close game (more motivation)
                if abs(game_stats.get('point_diff', 0)) <= 10:
                    spots['revenge_game'] = 1
            break

    # === SANDWICH GAME ===
    # Between two elite opponents (need to check both prev and next)
    # For training, check if previous opponent was elite
    if len(recent_games) >= 1:
        prev_opp = recent_games[0][1].get('opponent_abbrev', '')
        if prev_opp in ELITE_TEAMS and opponent_abbrev not in ELITE_TEAMS:
            # If we knew next game was also elite, this would be stronger
            # Partial indicator
            spots['sandwich_game'] = 1 if prev_opp in ELITE_TEAMS else 0

    # === EARLY SEASON VARIANCE ===
    # First 10 games have more variance, models less accurate
    total_games = len(recent_games)
    if total_games <= 10:
        spots['early_season_variance'] = 1

    # === COMBINED SCORE ===
    # Weighted combination of spots
    spot_weights = {
        'letdown_spot': -1.5,        # Negative = underperform
        'trap_game': -1.0,           # Negative = underperform
        'sandwich_game': -0.5,       # Negative = underperform
        'road_trip_fatigue': -1.5,   # Negative = underperform
        'revenge_game': 1.0,         # Positive = extra motivation
        'long_homestand': -0.5,      # Negative = complacency
        'early_season_variance': 0,  # Neutral (just more variance)
    }

    score = sum(spots[k] * spot_weights[k] for k in spot_weights)
    spots['schedule_spot_score'] = round(score, 2)

    return spots


def calculate_line_movement_features(
    opening_spread: Optional[float] = None,
    current_spread: Optional[float] = None,
    opening_total: Optional[float] = None,
    current_total: Optional[float] = None,
    model_spread: Optional[float] = None,
    model_win_prob: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate betting line movement features for live predictions.

    Line movement features help identify:
    - Sharp vs public money (reverse line movement)
    - Market disagreement with model (potential edge)
    - Steam moves (rapid line change)

    Args:
        opening_spread: Opening spread for home team (negative = favorite)
        current_spread: Current spread for home team
        opening_total: Opening over/under total
        current_total: Current over/under total
        model_spread: Model's predicted spread
        model_win_prob: Model's predicted home win probability

    Returns:
        Dictionary of line movement features
    """
    features = {
        # Spread movement
        'spread_movement': 0.0,              # Points moved (positive = toward home)
        'spread_movement_abs': 0.0,          # Absolute movement
        'spread_moved_toward_home': 0,       # Binary: line moved toward home
        'spread_moved_toward_away': 0,       # Binary: line moved toward away

        # Total movement
        'total_movement': 0.0,               # Points moved (positive = up)
        'total_movement_abs': 0.0,           # Absolute movement
        'total_moved_up': 0,                 # Binary: total increased
        'total_moved_down': 0,               # Binary: total decreased

        # Model vs market disagreement
        'model_vs_market_spread': 0.0,       # Model spread - Market spread
        'model_disagrees_spread': 0,         # Binary: |disagreement| > 2 points
        'model_favors_home_more': 0,         # Binary: model more bullish on home

        # Steam move indicators
        'large_spread_move': 0,              # Binary: > 2 point move
        'large_total_move': 0,               # Binary: > 3 point move

        # Reverse line movement (RLM) proxy
        # RLM occurs when line moves opposite to public betting %
        # We can't directly calculate without public % data, but large moves
        # against strong teams often indicate sharp money
        'line_has_moved': 0,                 # Binary: any movement detected
    }

    # Calculate spread movement
    if opening_spread is not None and current_spread is not None:
        movement = opening_spread - current_spread  # Positive = moved toward home (home became more favored)
        features['spread_movement'] = movement
        features['spread_movement_abs'] = abs(movement)
        features['spread_moved_toward_home'] = 1 if movement > 0.5 else 0
        features['spread_moved_toward_away'] = 1 if movement < -0.5 else 0
        features['large_spread_move'] = 1 if abs(movement) >= 2.0 else 0
        features['line_has_moved'] = 1 if abs(movement) >= 0.5 else 0

    # Calculate total movement
    if opening_total is not None and current_total is not None:
        total_move = current_total - opening_total  # Positive = total increased
        features['total_movement'] = total_move
        features['total_movement_abs'] = abs(total_move)
        features['total_moved_up'] = 1 if total_move > 0.5 else 0
        features['total_moved_down'] = 1 if total_move < -0.5 else 0
        features['large_total_move'] = 1 if abs(total_move) >= 3.0 else 0
        if features['line_has_moved'] == 0:
            features['line_has_moved'] = 1 if abs(total_move) >= 0.5 else 0

    # Calculate model vs market disagreement
    if current_spread is not None and model_spread is not None:
        disagreement = model_spread - current_spread
        features['model_vs_market_spread'] = disagreement
        features['model_disagrees_spread'] = 1 if abs(disagreement) >= 2.0 else 0
        features['model_favors_home_more'] = 1 if disagreement > 0 else 0

    return features


class PlayerStatsCalculator:
    """Calculate rolling player statistics for prop predictions."""

    # Position encoding for features
    POSITION_GROUPS = {
        'G': ['PG', 'SG', 'G', 'G-F'],
        'F': ['SF', 'PF', 'F', 'F-G', 'F-C'],
        'C': ['C', 'C-F'],
    }

    def __init__(self, window: int = 10):
        self.window = window
        self.player_games = defaultdict(list)
        self.player_info = {}

    def _get_position_group(self, position: str) -> str:
        """Map detailed position to position group (G/F/C)."""
        if not position:
            return 'G'  # Default to guard
        position = position.upper()
        for group, positions in self.POSITION_GROUPS.items():
            if position in positions:
                return group
        # Handle edge cases
        if 'G' in position:
            return 'G'
        elif 'F' in position:
            return 'F'
        elif 'C' in position:
            return 'C'
        return 'G'  # Default

    def _encode_position(self, position: str) -> Dict[str, int]:
        """Encode position as binary features."""
        pos_group = self._get_position_group(position)
        return {
            'is_guard': 1 if pos_group == 'G' else 0,
            'is_forward': 1 if pos_group == 'F' else 0,
            'is_center': 1 if pos_group == 'C' else 0,
        }

    def add_game_stats(self, player_id: int, game_date: str, stats: Dict, player_info: Dict = None):
        """Add player game stats to the historical record."""
        if player_info:
            self.player_info[player_id] = player_info

        # Determine opponent_id correctly based on player's team
        game_info = stats.get('game', {})
        player_team_id = stats.get('team', {}).get('id')
        home_team_id = game_info.get('home_team', {}).get('id')
        visitor_team_id = game_info.get('visitor_team', {}).get('id')
        opponent_id = visitor_team_id if player_team_id == home_team_id else home_team_id

        self.player_games[player_id].append((game_date, {
            'pts': stats.get('pts', 0) or 0,
            'reb': stats.get('reb', 0) or 0,
            'ast': stats.get('ast', 0) or 0,
            'stl': stats.get('stl', 0) or 0,
            'blk': stats.get('blk', 0) or 0,
            'fg3m': stats.get('fg3m', 0) or 0,
            'fg3a': stats.get('fg3a', 0) or 0,
            'fgm': stats.get('fgm', 0) or 0,
            'fga': stats.get('fga', 0) or 0,
            'ftm': stats.get('ftm', 0) or 0,
            'fta': stats.get('fta', 0) or 0,
            'min': self._parse_minutes(stats.get('min', '0')),
            'turnover': stats.get('turnover', 0) or 0,
            'team_id': player_team_id,
            'opponent_id': opponent_id,
        }))

    def _parse_minutes(self, min_str) -> float:
        """Parse minutes from string format."""
        if isinstance(min_str, (int, float)):
            return float(min_str)
        if not min_str:
            return 0.0
        try:
            if ':' in str(min_str):
                parts = str(min_str).split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_str)
        except:
            return 0.0

    def get_player_stats_before_date(self, player_id: int, date: str, min_games: int = 3) -> Optional[Dict]:
        """Get player statistics before a specific date with enhanced features."""
        if player_id not in self.player_games:
            return None

        games = [(d, s) for d, s in self.player_games[player_id] if d < date]
        if len(games) < min_games:
            return None

        games.sort(key=lambda x: x[0], reverse=True)
        recent = games[:self.window]
        last_5 = games[:5]
        last_3 = games[:3]  # NEW: Very recent form

        # NEW: Calculate days since last game
        try:
            from datetime import datetime
            current_date = datetime.strptime(date, "%Y-%m-%d")
            last_game_date = datetime.strptime(games[0][0], "%Y-%m-%d")
            days_rest = (current_date - last_game_date).days
        except:
            days_rest = 2  # Default to 2 days if calculation fails

        # Calculate stats
        pts = [g['pts'] for _, g in recent]
        reb = [g['reb'] for _, g in recent]
        ast = [g['ast'] for _, g in recent]
        fg3m = [g['fg3m'] for _, g in recent]
        mins = [g['min'] for _, g in recent]

        # TIER 1.1: Get position and role features
        player_info = self.player_info.get(player_id, {})
        position = player_info.get('position', '')
        position_features = self._encode_position(position)

        # Calculate role features based on averages
        season_pts_avg = np.mean([g['pts'] for _, g in games])
        season_min_avg = np.mean([g['min'] for _, g in games])

        # Role features: is_starter (25+ min avg), is_star (20+ pts avg), is_high_volume (18+ FGA)
        is_starter = 1 if season_min_avg >= 25 else 0
        is_star = 1 if season_pts_avg >= 20 else 0
        # High volume scorer (takes many shots)
        total_fga = sum(g.get('fga', 0) for _, g in recent)
        avg_fga = total_fga / len(recent) if recent else 0
        is_high_volume = 1 if avg_fga >= 15 else 0
        # Ball handler (high assist rate for guards)
        is_ball_handler = 1 if position_features['is_guard'] and np.mean(ast) >= 5 else 0

        features = {
            # Season averages (all games)
            'season_games': len(games),
            'season_pts_avg': np.mean([g['pts'] for _, g in games]),
            'season_reb_avg': np.mean([g['reb'] for _, g in games]),
            'season_ast_avg': np.mean([g['ast'] for _, g in games]),
            'season_fg3m_avg': np.mean([g['fg3m'] for _, g in games]),
            'season_min_avg': np.mean([g['min'] for _, g in games]),

            # Recent averages (last N games)
            'recent_pts_avg': np.mean(pts),
            'recent_pts_std': np.std(pts) if len(pts) > 1 else 0,
            'recent_pts_min': np.min(pts),
            'recent_pts_max': np.max(pts),
            'recent_reb_avg': np.mean(reb),
            'recent_reb_std': np.std(reb) if len(reb) > 1 else 0,
            'recent_ast_avg': np.mean(ast),
            'recent_ast_std': np.std(ast) if len(ast) > 1 else 0,
            'recent_fg3m_avg': np.mean(fg3m),
            'recent_fg3m_std': np.std(fg3m) if len(fg3m) > 1 else 0,
            'recent_min_avg': np.mean(mins),

            # NEW: Minutes trend and consistency features
            'min_trend': np.mean([g['min'] for _, g in last_5]) - np.mean(mins) if mins else 0,
            'min_consistency': 1 - (np.std(mins) / np.mean(mins)) if np.mean(mins) > 0 else 0,
            'last5_min_avg': np.mean([g['min'] for _, g in last_5]),

            # Last 5 games (more recent form)
            'last5_pts_avg': np.mean([g['pts'] for _, g in last_5]),
            'last5_reb_avg': np.mean([g['reb'] for _, g in last_5]),
            'last5_ast_avg': np.mean([g['ast'] for _, g in last_5]),
            'last5_fg3m_avg': np.mean([g['fg3m'] for _, g in last_5]),

            # NEW: Last 3 games (very recent form - critical for prop betting)
            'last3_pts_avg': np.mean([g['pts'] for _, g in last_3]),
            'last3_reb_avg': np.mean([g['reb'] for _, g in last_3]),
            'last3_ast_avg': np.mean([g['ast'] for _, g in last_3]),
            'last3_fg3m_avg': np.mean([g['fg3m'] for _, g in last_3]),
            'last3_min_avg': np.mean([g['min'] for _, g in last_3]),

            # Trends
            'pts_trend': np.mean([g['pts'] for _, g in last_5]) - np.mean(pts),
            'reb_trend': np.mean([g['reb'] for _, g in last_5]) - np.mean(reb),
            'ast_trend': np.mean([g['ast'] for _, g in last_5]) - np.mean(ast),
            'fg3m_trend': np.mean([g['fg3m'] for _, g in last_5]) - np.mean(fg3m),

            # NEW: Season variance (consistency indicator)
            'season_pts_std': np.std([g['pts'] for _, g in games]) if len(games) > 1 else 0,
            'season_reb_std': np.std([g['reb'] for _, g in games]) if len(games) > 1 else 0,
            'season_ast_std': np.std([g['ast'] for _, g in games]) if len(games) > 1 else 0,
            'season_fg3m_std': np.std([g['fg3m'] for _, g in games]) if len(games) > 1 else 0,

            # Combined stats
            'pra_avg': np.mean([g['pts'] + g['reb'] + g['ast'] for _, g in recent]),
            'pra_std': np.std([g['pts'] + g['reb'] + g['ast'] for _, g in recent]) if len(recent) > 1 else 0,
            'last3_pra_avg': np.mean([g['pts'] + g['reb'] + g['ast'] for _, g in last_3]),

            # ADVANCED EFFICIENCY STATS (Phase 4 enhancement)
            # True Shooting % (TS%): PTS / (2 * (FGA + 0.44 * FTA))
            'ts_pct': self._calc_ts_pct(recent),
            # Effective FG% (eFG%): (FGM + 0.5 * 3PM) / FGA
            'efg_pct': self._calc_efg_pct(recent),
            # Usage Rate approximation: (FGA + 0.44*FTA + TOV) / minutes played
            'usage_rate': self._calc_usage_rate(recent),
            # Shooting volume
            'fg3_rate': self._calc_fg3_rate(recent),  # 3PA / FGA
            'fta_rate': self._calc_fta_rate(recent),  # FTA / FGA (free throw rate)

            # TIER 1.2: Advanced stats (BPM, assist rate, rebound rate)
            'bpm': self._calc_simplified_bpm(recent),  # Box Plus/Minus approximation
            'assist_rate': self._calc_assist_rate(recent),  # Assists per 36 min
            'rebound_rate': self._calc_rebound_rate(recent),  # Rebounds per 36 min

            # NEW: Rest days features
            'days_rest': days_rest,
            'is_back_to_back': 1 if days_rest == 1 else 0,

            # NEW PHASE 2: 3-POINTER SHOT QUALITY FEATURES
            # FG3% (3-point shooting accuracy)
            'fg3_pct': self._calc_fg3_pct(recent),
            # Recent 5-game FG3%
            'last5_fg3_pct': self._calc_fg3_pct(last_5),
            # FG3% variance (shooting consistency)
            'fg3_pct_variance': self._calc_fg3_variance(games),
            # Hot/cold streak detection
            **self._calc_fg3_streak_features(games),

            # TIER 1.3: Specialized 3PM features (shooting-specific)
            **self._calc_three_pm_specialized_features(recent, games, mins),
        }

        # TIER 1.1: Add position and role features
        features.update(position_features)  # is_guard, is_forward, is_center
        features['is_starter'] = is_starter
        features['is_star'] = is_star
        features['is_high_volume'] = is_high_volume
        features['is_ball_handler'] = is_ball_handler

        # Position-specific expected stats (league averages by position)
        # Guards: Higher assists, lower rebounds
        # Forwards: Balanced
        # Centers: Higher rebounds, lower assists
        pos_reb_factor = 1.3 if position_features['is_center'] else (1.0 if position_features['is_forward'] else 0.7)
        pos_ast_factor = 1.3 if position_features['is_guard'] else (0.9 if position_features['is_forward'] else 0.6)
        features['pos_reb_factor'] = pos_reb_factor
        features['pos_ast_factor'] = pos_ast_factor

        # CRITICAL NEW FEATURES: Recency ratio, variance penalty, minutes stability
        # These features are critical for predicting when players will under/over perform

        # Points recency ratio (< 1 = slumping, > 1 = hot)
        season_pts = features.get('season_pts_avg', 0)
        recent_pts = features.get('recent_pts_avg', 0)
        features['pts_recency_ratio'] = recent_pts / season_pts if season_pts > 0 else 1.0

        # Rebounds recency ratio
        season_reb = features.get('season_reb_avg', 0)
        recent_reb = features.get('recent_reb_avg', 0)
        features['reb_recency_ratio'] = recent_reb / season_reb if season_reb > 0 else 1.0

        # Assists recency ratio
        season_ast = features.get('season_ast_avg', 0)
        recent_ast = features.get('recent_ast_avg', 0)
        features['ast_recency_ratio'] = recent_ast / season_ast if season_ast > 0 else 1.0

        # 3PM recency ratio
        season_fg3 = features.get('season_fg3m_avg', 0)
        recent_fg3 = features.get('recent_fg3m_avg', 0)
        features['fg3_recency_ratio'] = recent_fg3 / season_fg3 if season_fg3 > 0 else 1.0

        # Variance penalties (higher = less predictable)
        features['pts_variance_penalty'] = features.get('season_pts_std', 0) / max(season_pts, 1)
        features['reb_variance_penalty'] = features.get('season_reb_std', 0) / max(season_reb, 1)
        features['ast_variance_penalty'] = features.get('season_ast_std', 0) / max(season_ast, 1)
        features['fg3_variance_penalty'] = features.get('season_fg3m_std', 0) / max(season_fg3, 0.5)

        # Minutes stability (coefficient of variation)
        season_min = features.get('season_min_avg', 0)
        recent_min = features.get('recent_min_avg', 0)
        features['minutes_cv'] = abs(recent_min - season_min) / max(season_min, 1)
        features['minutes_recency_ratio'] = recent_min / season_min if season_min > 0 else 1.0

        return features

    def _calc_ts_pct(self, games: List[Tuple[str, Dict]]) -> float:
        """Calculate True Shooting % from game list."""
        total_pts = sum(g['pts'] for _, g in games)
        total_fga = sum(g['fga'] for _, g in games)
        total_fta = sum(g.get('fta', 0) for _, g in games)
        tsa = 2 * (total_fga + 0.44 * total_fta)
        if tsa > 0:
            return round(total_pts / tsa, 3)
        return 0.55  # League average default

    def _calc_efg_pct(self, games: List[Tuple[str, Dict]]) -> float:
        """Calculate Effective FG% from game list."""
        total_fgm = sum(g['fgm'] for _, g in games)
        total_fg3m = sum(g['fg3m'] for _, g in games)
        total_fga = sum(g['fga'] for _, g in games)
        if total_fga > 0:
            return round((total_fgm + 0.5 * total_fg3m) / total_fga, 3)
        return 0.50  # League average default

    def _calc_usage_rate(self, games: List[Tuple[str, Dict]]) -> float:
        """Calculate simplified usage rate approximation."""
        total_fga = sum(g['fga'] for _, g in games)
        total_fta = sum(g.get('fta', 0) for _, g in games)
        total_tov = sum(g.get('turnover', 0) for _, g in games)
        total_min = sum(g['min'] for _, g in games)
        # Usage approximation: possessions used / minutes (normalized to ~0.25 avg)
        if total_min > 0:
            poss_used = total_fga + 0.44 * total_fta + total_tov
            # Normalize to roughly 0.20-0.35 range like real USG%
            usage = (poss_used / total_min) * 0.4
            return round(min(0.45, max(0.10, usage)), 3)
        return 0.22  # League average default

    def _calc_fg3_rate(self, games: List[Tuple[str, Dict]]) -> float:
        """Calculate 3-point attempt rate (3PA/FGA)."""
        total_fg3a = sum(g.get('fg3a', 0) for _, g in games)
        total_fga = sum(g['fga'] for _, g in games)
        if total_fga > 0:
            return round(total_fg3a / total_fga, 3)
        return 0.35  # League average default

    def _calc_fta_rate(self, games: List[Tuple[str, Dict]]) -> float:
        """Calculate free throw attempt rate (FTA/FGA)."""
        total_fta = sum(g.get('fta', 0) for _, g in games)
        total_fga = sum(g['fga'] for _, g in games)
        if total_fga > 0:
            return round(total_fta / total_fga, 3)
        return 0.25  # League average default

    def _calc_fg3_pct(self, games: List[Tuple[str, Dict]]) -> float:
        """Calculate 3-point shooting percentage (FG3M/FG3A)."""
        total_fg3m = sum(g.get('fg3m', 0) for _, g in games)
        total_fg3a = sum(g.get('fg3a', 0) for _, g in games)
        if total_fg3a > 0:
            return round(total_fg3m / total_fg3a, 3)
        return 0.36  # League average default

    def _calc_fg3_streak_features(self, games: List[Tuple[str, Dict]]) -> Dict[str, float]:
        """Detect hot/cold shooting streaks for 3-pointers."""
        if len(games) < 3:
            return {'fg3_hot_streak': 0, 'fg3_cold_streak': 0, 'fg3_momentum': 0.0}

        # Calculate game-by-game FG3%
        recent_fg3_pcts = []
        for _, g in games[:5]:  # Last 5 games
            fg3a = g.get('fg3a', 0) or 0
            fg3m = g.get('fg3m', 0) or 0
            if fg3a >= 2:  # Only count games with meaningful attempts
                recent_fg3_pcts.append(fg3m / fg3a)

        if not recent_fg3_pcts:
            return {'fg3_hot_streak': 0, 'fg3_cold_streak': 0, 'fg3_momentum': 0.0}

        # Hot streak: above 40% in last 3 games
        hot_streak = sum(1 for p in recent_fg3_pcts[:3] if p >= 0.40) >= 2
        # Cold streak: below 30% in last 3 games
        cold_streak = sum(1 for p in recent_fg3_pcts[:3] if p <= 0.30) >= 2

        # Momentum: trend direction (positive = improving)
        if len(recent_fg3_pcts) >= 3:
            momentum = np.polyfit(range(len(recent_fg3_pcts)), recent_fg3_pcts, 1)[0]
        else:
            momentum = 0.0

        return {
            'fg3_hot_streak': 1 if hot_streak else 0,
            'fg3_cold_streak': 1 if cold_streak else 0,
            'fg3_momentum': round(float(momentum), 4),
        }

    def _calc_fg3_variance(self, games: List[Tuple[str, Dict]]) -> float:
        """Calculate variance in 3-point shooting percentage (consistency indicator)."""
        game_fg3_pcts = []
        for _, g in games:
            fg3a = g.get('fg3a', 0) or 0
            fg3m = g.get('fg3m', 0) or 0
            if fg3a >= 3:  # Only count games with meaningful attempts
                game_fg3_pcts.append(fg3m / fg3a)
        if len(game_fg3_pcts) >= 3:
            return round(float(np.var(game_fg3_pcts)), 4)
        return 0.1  # Default variance

    def _calc_three_pm_specialized_features(self, recent: List[Tuple[str, Dict]],
                                            all_games: List[Tuple[str, Dict]],
                                            mins: List[float]) -> Dict[str, float]:
        """
        Calculate specialized features for 3PM prediction.

        The key insight is that 3PM = FG3A × FG3%, where:
        - FG3A (attempts) is fairly predictable based on minutes and shot distribution
        - FG3% is highly variable but regresses to mean over time

        We create features that capture:
        1. Expected 3PA (attempts per minute)
        2. Regressed FG3% (blended with league average)
        3. Expected 3PM (attempts × regressed %)
        4. Shooting consistency metrics
        """
        LEAGUE_AVG_FG3_PCT = 0.36  # NBA league average 3PT%

        if not recent:
            return {
                'fg3a_per_min': 0.15,  # Default ~4.5 attempts in 30 min
                'fg3a_avg': 4.5,
                'fg3a_std': 2.0,
                'fg3a_consistency': 0.7,
                'regressed_fg3_pct': LEAGUE_AVG_FG3_PCT,
                'expected_fg3m': 1.5,
                'fg3_makes_std': 1.0,
                'fg3_attempt_trend': 0.0,
                'is_volume_shooter': 0,
                'shooting_confidence': 0.5,
            }

        # FG3A (attempts) statistics
        fg3a_values = [g.get('fg3a', 0) or 0 for _, g in recent]
        fg3m_values = [g.get('fg3m', 0) or 0 for _, g in recent]

        fg3a_avg = np.mean(fg3a_values) if fg3a_values else 0
        fg3a_std = np.std(fg3a_values) if len(fg3a_values) > 1 else 2.0
        fg3m_avg = np.mean(fg3m_values) if fg3m_values else 0
        fg3m_std = np.std(fg3m_values) if len(fg3m_values) > 1 else 1.0

        # FG3A per minute (normalizes for playing time)
        total_mins = sum(mins)
        total_fg3a = sum(fg3a_values)
        fg3a_per_min = (total_fg3a / total_mins) if total_mins > 0 else 0.15

        # FG3A consistency (lower variance = more predictable)
        fg3a_consistency = 1 - (fg3a_std / max(fg3a_avg, 1)) if fg3a_avg > 0 else 0.5
        fg3a_consistency = max(0.3, min(1.0, fg3a_consistency))

        # Calculate raw FG3%
        raw_fg3_pct = (sum(fg3m_values) / sum(fg3a_values)) if sum(fg3a_values) > 0 else LEAGUE_AVG_FG3_PCT

        # Regressed FG3% - blend with league average based on sample size
        # The idea is: with few shots, regress more to league average
        total_attempts_season = sum(g.get('fg3a', 0) or 0 for _, g in all_games)
        regression_weight = min(1.0, total_attempts_season / 250)  # Full weight at 250 attempts
        regressed_fg3_pct = regression_weight * raw_fg3_pct + (1 - regression_weight) * LEAGUE_AVG_FG3_PCT

        # Expected 3PM = Expected FG3A × Regressed FG3%
        expected_fg3m = fg3a_avg * regressed_fg3_pct

        # FG3A trend (last 3 vs average)
        if len(fg3a_values) >= 3:
            last3_fg3a = np.mean(fg3a_values[:3])
            fg3_attempt_trend = last3_fg3a - fg3a_avg
        else:
            fg3_attempt_trend = 0.0

        # Volume shooter flag (takes >= 5 3PA per game on average)
        is_volume_shooter = 1 if fg3a_avg >= 5 else 0

        # Shooting confidence: combines sample size with consistency
        sample_factor = min(1.0, len(all_games) / 20)  # Full confidence at 20+ games
        shooting_confidence = sample_factor * fg3a_consistency

        return {
            'fg3a_per_min': round(fg3a_per_min, 4),
            'fg3a_avg': round(fg3a_avg, 2),
            'fg3a_std': round(fg3a_std, 2),
            'fg3a_consistency': round(fg3a_consistency, 3),
            'regressed_fg3_pct': round(regressed_fg3_pct, 4),
            'expected_fg3m': round(expected_fg3m, 2),
            'fg3_makes_std': round(fg3m_std, 2),
            'fg3_attempt_trend': round(fg3_attempt_trend, 2),
            'is_volume_shooter': is_volume_shooter,
            'shooting_confidence': round(shooting_confidence, 3),
        }

    def _calc_simplified_bpm(self, games: List[Tuple[str, Dict]]) -> float:
        """
        Calculate simplified Box Plus/Minus (BPM) approximation.
        BPM estimates a player's contribution per 100 possessions.
        League average is 0, stars typically range from +5 to +10.
        """
        if not games:
            return 0.0

        total_mins = sum(g['min'] for _, g in games)
        if total_mins < 50:  # Need meaningful sample
            return 0.0

        # Calculate per-36 minute rates
        per36_factor = 36 * len(games) / total_mins if total_mins > 0 else 0

        total_pts = sum(g['pts'] for _, g in games)
        total_reb = sum(g['reb'] for _, g in games)
        total_ast = sum(g['ast'] for _, g in games)
        total_stl = sum(g.get('stl', 0) for _, g in games)
        total_blk = sum(g.get('blk', 0) for _, g in games)
        total_tov = sum(g.get('turnover', 0) for _, g in games)
        total_fga = sum(g['fga'] for _, g in games)
        total_fgm = sum(g['fgm'] for _, g in games)

        # Per-36 stats
        pts_per36 = total_pts * per36_factor / len(games) if games else 0
        reb_per36 = total_reb * per36_factor / len(games) if games else 0
        ast_per36 = total_ast * per36_factor / len(games) if games else 0
        stl_per36 = total_stl * per36_factor / len(games) if games else 0
        blk_per36 = total_blk * per36_factor / len(games) if games else 0
        tov_per36 = total_tov * per36_factor / len(games) if games else 0

        # Shooting efficiency
        fg_pct = total_fgm / total_fga if total_fga > 0 else 0.45

        # Simplified BPM formula (based on Basketball-Reference methodology)
        # Weights approximate the value of each stat per 100 possessions
        bpm = (
            0.4 * (pts_per36 - 15)  # Points above average
            + 0.5 * (reb_per36 - 7)  # Rebounds above average
            + 0.7 * (ast_per36 - 4)  # Assists above average
            + 1.5 * (stl_per36 - 1)  # Steals above average
            + 1.0 * (blk_per36 - 0.5)  # Blocks above average
            - 1.0 * (tov_per36 - 2)  # Turnovers (negative value)
            + 5.0 * (fg_pct - 0.45)  # Shooting efficiency bonus
        )

        # Clamp to realistic range (-10 to +15)
        return round(max(-10, min(15, bpm)), 2)

    def _calc_assist_rate(self, games: List[Tuple[str, Dict]]) -> float:
        """
        Calculate assist rate (assists per 36 minutes).
        Normalizes assists to playing time for fair comparison.
        """
        if not games:
            return 4.0  # League average default

        total_ast = sum(g['ast'] for _, g in games)
        total_mins = sum(g['min'] for _, g in games)

        if total_mins > 0:
            ast_per36 = (total_ast / total_mins) * 36
            return round(ast_per36, 2)
        return 4.0  # League average default

    def _calc_rebound_rate(self, games: List[Tuple[str, Dict]]) -> float:
        """
        Calculate rebound rate (rebounds per 36 minutes).
        Normalizes rebounds to playing time for fair comparison.
        """
        if not games:
            return 7.0  # League average default

        total_reb = sum(g['reb'] for _, g in games)
        total_mins = sum(g['min'] for _, g in games)

        if total_mins > 0:
            reb_per36 = (total_reb / total_mins) * 36
            return round(reb_per36, 2)
        return 7.0  # League average default

    def get_player_vs_opponent_history(self, player_id: int, opponent_id: int, date: str, min_games: int = 2) -> Optional[Dict]:
        """
        Get player's historical performance against a specific opponent.

        Args:
            player_id: The player's ID
            opponent_id: The opponent team's ID
            date: Only consider games before this date
            min_games: Minimum number of games required

        Returns:
            Dictionary with matchup-specific stats, or None if insufficient data
        """
        if player_id not in self.player_games:
            return None

        # Filter games against the specific opponent before the given date
        games = [(d, s) for d, s in self.player_games[player_id]
                 if d < date and s.get('opponent_id') == opponent_id]

        if len(games) < min_games:
            return None

        return {
            'vs_opp_games': len(games),
            'vs_opp_pts_avg': np.mean([g['pts'] for _, g in games]),
            'vs_opp_reb_avg': np.mean([g['reb'] for _, g in games]),
            'vs_opp_ast_avg': np.mean([g['ast'] for _, g in games]),
            'vs_opp_fg3m_avg': np.mean([g['fg3m'] for _, g in games]),
            'vs_opp_min_avg': np.mean([g['min'] for _, g in games]),
            'vs_opp_pra_avg': np.mean([g['pts'] + g['reb'] + g['ast'] for _, g in games]),
        }


# =============================================================================
# NEW FEATURE HELPER FUNCTIONS
# =============================================================================

def calculate_blowout_risk_features(home_stats: Dict, away_stats: Dict, vegas_spread: float = None) -> Dict:
    """
    Calculate blowout risk features that predict when starters may get pulled early.

    Research shows:
    - Vegas spreads > 10 points have ~35% blowout chance
    - Spreads > 15 points have ~50% blowout chance
    - Blowouts typically result in 5-7 minute reduction for starters

    Args:
        home_stats: Home team statistics
        away_stats: Away team statistics
        vegas_spread: Vegas spread if available (positive = home favored)

    Returns:
        Dictionary of blowout risk features
    """
    # Estimate spread from team strength if Vegas line not available
    if vegas_spread is not None:
        spread_magnitude = abs(vegas_spread)
    else:
        # Estimate from net ratings: each point of net rating diff ~= 2.5 point spread
        home_net = home_stats.get('net_rating', 0) if home_stats else 0
        away_net = away_stats.get('net_rating', 0) if away_stats else 0
        net_diff = (home_net - away_net) + 3.5  # Home court advantage ~3.5 pts
        spread_magnitude = abs(net_diff) / 2.5 * 2.5  # Rough conversion

    # Blowout probability calculation
    # Based on historical data: P(blowout) = min(0.7, spread / 25)
    blowout_prob = min(0.70, max(0.05, spread_magnitude / 25))

    # Expected minutes reduction for starters in blowout scenario
    # Stars typically play 34-36 min, reduced to 26-30 in blowouts
    expected_min_reduction = blowout_prob * 7  # Up to 7 minutes reduction

    # Projected minutes factor (multiply by expected minutes)
    # 1.0 = full minutes, 0.82 = 18% reduction in blowout
    projected_min_factor = 1.0 - (blowout_prob * 0.20)

    return {
        'blowout_probability': round(blowout_prob, 3),
        'expected_min_reduction': round(expected_min_reduction, 2),
        'projected_min_factor': round(projected_min_factor, 3),
        'is_likely_blowout': 1 if spread_magnitude >= 12 else 0,
        'spread_magnitude': round(spread_magnitude, 1),
    }


def calculate_pace_adjusted_features(player_features: Dict, team_pace: float,
                                      opp_pace: float, league_avg_pace: float = 100.0) -> Dict:
    """
    Calculate pace-adjusted per-100-possession stats.

    This normalizes stats to account for different team tempos, which is critical
    because a fast-paced game (110 poss) has ~10% more scoring opportunities than
    a slow-paced game (100 poss).

    Args:
        player_features: Player's raw stat averages
        team_pace: Player's team pace (possessions per 48 min)
        opp_pace: Opponent's pace
        league_avg_pace: League average pace (default 100)

    Returns:
        Dictionary of pace-adjusted features
    """
    # Expected game pace (average of both teams)
    expected_game_pace = (team_pace + opp_pace) / 2

    # Pace multiplier relative to league average
    # > 1.0 means more possessions than average (boost to stats)
    pace_multiplier = expected_game_pace / league_avg_pace

    # Get player's raw averages
    pts_avg = player_features.get('season_pts_avg', 0) or 0
    reb_avg = player_features.get('season_reb_avg', 0) or 0
    ast_avg = player_features.get('season_ast_avg', 0) or 0
    fg3m_avg = player_features.get('season_fg3m_avg', 0) or 0
    min_avg = player_features.get('season_min_avg', 30) or 30

    # Estimate possessions played (player's share of team possessions)
    # Roughly: possessions = minutes * (team_pace / 48)
    player_poss = (min_avg / 48) * team_pace if team_pace > 0 else min_avg * 2

    # Per-100-possession stats (normalized)
    # This allows fair comparison across different team tempos
    poss_factor = 100 / max(player_poss, 50)  # Avoid division issues

    return {
        'expected_game_pace': round(expected_game_pace, 1),
        'pace_multiplier': round(pace_multiplier, 3),
        'pace_vs_average': round(expected_game_pace - league_avg_pace, 1),
        'is_high_pace_game': 1 if expected_game_pace >= 103 else 0,
        'is_low_pace_game': 1 if expected_game_pace <= 97 else 0,
        # Pace-adjusted predictions (expected boost/reduction from pace)
        'pace_pts_adjustment': round((pace_multiplier - 1) * pts_avg, 2),
        'pace_reb_adjustment': round((pace_multiplier - 1) * reb_avg * 0.5, 2),  # Rebounds less affected
        'pace_ast_adjustment': round((pace_multiplier - 1) * ast_avg, 2),
        'pace_fg3_adjustment': round((pace_multiplier - 1) * fg3m_avg, 2),
        # Per-100-possession normalized stats
        'pts_per_100_poss': round(pts_avg * poss_factor, 1),
        'reb_per_100_poss': round(reb_avg * poss_factor, 1),
        'ast_per_100_poss': round(ast_avg * poss_factor, 1),
    }


def detect_outlier_game(game_stats: Dict, player_avg: Dict, threshold: float = 2.5) -> Dict:
    """
    Detect if a game is an outlier based on player performance vs their averages.

    This helps identify:
    1. OT games (inflated stats due to extra playing time)
    2. Blowouts (deflated stats due to garbage time)
    3. Statistical anomalies (injury exits, foul trouble, etc.)

    Args:
        game_stats: Stats from this specific game
        player_avg: Player's season/recent averages
        threshold: Z-score threshold for outlier detection

    Returns:
        Dictionary with outlier flags and adjustment factors
    """
    outlier_flags = {
        'is_outlier': 0,
        'outlier_type': 'normal',
        'adjustment_factor': 1.0,
        'z_score_pts': 0.0,
        'z_score_min': 0.0,
    }

    # Get actual vs expected values
    actual_pts = game_stats.get('pts', 0) or 0
    actual_min = game_stats.get('min', 0) or 0

    # Handle string minutes like "32:15"
    if isinstance(actual_min, str):
        try:
            parts = actual_min.split(':')
            actual_min = float(parts[0]) + float(parts[1]) / 60 if len(parts) == 2 else float(parts[0])
        except:
            actual_min = 25

    expected_pts = player_avg.get('season_pts_avg', actual_pts) or actual_pts
    expected_min = player_avg.get('season_min_avg', actual_min) or actual_min
    pts_std = player_avg.get('season_pts_std', expected_pts * 0.3) or (expected_pts * 0.3)
    min_std = player_avg.get('recent_min_avg', expected_min * 0.15) or (expected_min * 0.15)

    # Calculate z-scores
    if pts_std > 0:
        z_score_pts = (actual_pts - expected_pts) / pts_std
    else:
        z_score_pts = 0

    if min_std > 0:
        z_score_min = (actual_min - expected_min) / max(min_std, 3)
    else:
        z_score_min = 0

    outlier_flags['z_score_pts'] = round(z_score_pts, 2)
    outlier_flags['z_score_min'] = round(z_score_min, 2)

    # Detect OT game (minutes significantly above expected)
    if actual_min > expected_min + 8:  # 8+ extra minutes suggests OT
        outlier_flags['is_outlier'] = 1
        outlier_flags['outlier_type'] = 'overtime'
        # OT inflates stats - apply reduction factor for future predictions
        outlier_flags['adjustment_factor'] = expected_min / max(actual_min, 1)

    # Detect blowout (minutes significantly below expected for starters)
    elif actual_min < expected_min - 10 and expected_min >= 25:  # 10+ fewer minutes
        outlier_flags['is_outlier'] = 1
        outlier_flags['outlier_type'] = 'blowout'
        # Blowout deflates stats - don't heavily weight this game
        outlier_flags['adjustment_factor'] = 0.7

    # Detect statistical anomaly (points way off from expected)
    elif abs(z_score_pts) > threshold:
        outlier_flags['is_outlier'] = 1
        outlier_flags['outlier_type'] = 'anomaly_high' if z_score_pts > 0 else 'anomaly_low'
        # Reduce weight of anomalous games in training
        outlier_flags['adjustment_factor'] = 0.8

    return outlier_flags


def calculate_vegas_total_features(vegas_total: float, player_features: Dict,
                                    league_avg_total: float = 225.0) -> Dict:
    """
    Use Vegas game total to improve individual prop predictions.

    Logic: Higher game totals = more possessions = more stats for everyone
    - Game total of 235 (10 above average) suggests ~4.4% more scoring
    - This impacts all prop types, not just points

    Args:
        vegas_total: Vegas over/under line for total points
        player_features: Player's stat averages
        league_avg_total: League average game total (default 225)

    Returns:
        Dictionary of Vegas total features
    """
    if vegas_total is None or vegas_total <= 0:
        return {
            'vegas_total': 225.0,
            'total_vs_average': 0.0,
            'total_multiplier': 1.0,
            'is_high_total_game': 0,
            'is_low_total_game': 0,
            'total_pts_boost': 0.0,
        }

    # Total multiplier (1.0 = average scoring environment)
    total_multiplier = vegas_total / league_avg_total
    total_vs_average = vegas_total - league_avg_total

    # Get player's scoring share approximation
    pts_avg = player_features.get('season_pts_avg', 15) or 15
    usage = player_features.get('usage_rate', 0.20) or 0.20

    # Expected points boost from high/low total
    # Higher usage players benefit more from high-scoring games
    total_pts_boost = (total_multiplier - 1) * pts_avg * (1 + usage)

    return {
        'vegas_total': round(vegas_total, 1),
        'total_vs_average': round(total_vs_average, 1),
        'total_multiplier': round(total_multiplier, 3),
        'is_high_total_game': 1 if vegas_total >= 235 else 0,
        'is_low_total_game': 1 if vegas_total <= 215 else 0,
        'total_pts_boost': round(total_pts_boost, 2),
    }


# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_games_for_training(games: List[Dict], player_stats_by_game: Dict[int, List[Dict]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Process games into training data for both team and player models.

    Returns:
        Tuple of (team_training_data, player_training_data)
    """
    print(f"\nProcessing {len(games)} games...")

    # Sort games by date
    games_sorted = sorted(games, key=lambda g: g.get('date', ''))

    team_calc = TeamStatsCalculator(window=10)
    player_calc = PlayerStatsCalculator(window=10)
    position_defense_calc = PositionDefenseCalculator()  # TIER 2.2: Track position-specific defense
    elo_system = EloRatingSystem(k_factor=20.0, home_advantage=100.0)  # NEW: Elo ratings

    team_data = []
    player_data = []
    skipped = 0

    for i, game in enumerate(games_sorted):
        game_date = game.get('date', '')
        if isinstance(game_date, str) and 'T' in game_date:
            game_date = game_date.split('T')[0]

        game_id = game.get('id')
        home_team = game.get('home_team', {})
        away_team = game.get('visitor_team', {})

        home_team_id = home_team.get('id')
        away_team_id = away_team.get('id')
        home_score = game.get('home_team_score', 0)
        away_score = game.get('visitor_team_score', 0)

        if not all([home_team_id, away_team_id, game_date, home_score]):
            continue

        # Get team features BEFORE adding this game (point-in-time)
        home_stats = team_calc.get_team_stats_before_date(home_team_id, game_date)
        away_stats = team_calc.get_team_stats_before_date(away_team_id, game_date)

        # Get Elo ratings BEFORE this game (point-in-time)
        home_elo = elo_system.get_rating_before_date(home_team_id, game_date)
        away_elo = elo_system.get_rating_before_date(away_team_id, game_date)
        elo_win_prob = elo_system.predict_win_probability(home_team_id, away_team_id, before_date=game_date)
        elo_spread = elo_system.get_spread_prediction(home_team_id, away_team_id, before_date=game_date)

        # Get last game info for travel/fatigue features
        home_team_abbrev = home_team.get('abbreviation', '')
        away_team_abbrev = away_team.get('abbreviation', '')

        home_last_game = team_calc.get_last_game_info(home_team_id, game_date)
        away_last_game = team_calc.get_last_game_info(away_team_id, game_date)

        # Calculate travel fatigue for home team (usually minimal - playing at home)
        home_travel_features = {'travel_distance': 0.0, 'timezone_change': 0, 'altitude_change': 0,
                                 'altitude_disadvantage': 0.0, 'travel_fatigue_score': 0.0, 'coast_to_coast': 0}
        home_days_rest = 2
        home_is_b2b = False
        if home_last_game:
            home_days_rest = home_last_game['days_rest']
            home_is_b2b = home_last_game['is_back_to_back']
            # Home team: travel from last game venue to home
            if home_last_game.get('venue_abbrev'):
                home_travel_features = calc_travel_fatigue_features(
                    home_last_game['venue_abbrev'], home_team_abbrev,
                    home_days_rest, home_is_b2b
                )

        # Calculate travel fatigue for away team (usually significant)
        away_travel_features = {'travel_distance': 0.0, 'timezone_change': 0, 'altitude_change': 0,
                                 'altitude_disadvantage': 0.0, 'travel_fatigue_score': 0.0, 'coast_to_coast': 0}
        away_days_rest = 2
        away_is_b2b = False
        if away_last_game:
            away_days_rest = away_last_game['days_rest']
            away_is_b2b = away_last_game['is_back_to_back']
            # Away team: travel from last game venue to this game (home team's arena)
            if away_last_game.get('venue_abbrev'):
                away_travel_features = calc_travel_fatigue_features(
                    away_last_game['venue_abbrev'], home_team_abbrev,
                    away_days_rest, away_is_b2b
                )

        # === NEW: SCHEDULE SPOT ANALYSIS ===
        # Analyze schedule-based situational spots for both teams
        home_schedule_spots = analyze_schedule_spots(
            team_id=home_team_id,
            team_abbrev=home_team_abbrev,
            game_date=game_date,
            opponent_abbrev=away_team_abbrev,
            team_calc=team_calc,
            is_home=True,
            future_games=None  # Historical training - no future data available
        )

        away_schedule_spots = analyze_schedule_spots(
            team_id=away_team_id,
            team_abbrev=away_team_abbrev,
            game_date=game_date,
            opponent_abbrev=home_team_abbrev,
            team_calc=team_calc,
            is_home=False,
            future_games=None  # Historical training - no future data available
        )

        # Add game to team calculator (AFTER getting point-in-time features)
        team_calc.add_game(game)

        # Update Elo ratings AFTER the game
        elo_system.update_ratings(home_team_id, away_team_id, home_score, away_score, game_date)

        # Process player stats for this game
        # Try both int and string keys for compatibility
        game_player_stats = player_stats_by_game.get(game_id, []) or player_stats_by_game.get(str(game_id), [])

        # TIER 2.2: Process game through position defense calculator
        # This tracks what each team allows to each position for future predictions
        if game_player_stats:
            position_defense_calc.process_game(
                game_id=game_id,
                game_date=game_date,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                player_stats=game_player_stats
            )

        for ps in game_player_stats:
            player_id = ps.get('player', {}).get('id')
            if not player_id:
                continue

            # Get player features BEFORE this game
            player_pre_stats = player_calc.get_player_stats_before_date(player_id, game_date)

            # Add to player calculator
            player_calc.add_game_stats(
                player_id,
                game_date,
                ps,
                ps.get('player', {})
            )

            # Create player training example if we have enough history
            if player_pre_stats and ps.get('min') and player_calc._parse_minutes(ps.get('min')) >= 10:
                actual_pts = ps.get('pts', 0) or 0
                actual_reb = ps.get('reb', 0) or 0
                actual_ast = ps.get('ast', 0) or 0
                actual_fg3m = ps.get('fg3m', 0) or 0
                actual_pra = actual_pts + actual_reb + actual_ast
                actual_min = player_calc._parse_minutes(ps.get('min', '0'))  # TIER 2.3: Track actual minutes

                # ============================================================
                # CONFIDENCE FIX: Detect outlier games and apply corrections
                # ============================================================

                # Detect if this is an OT or blowout game
                game_stats_for_outlier = {'pts': actual_pts, 'min': actual_min}
                outlier_info = detect_outlier_game(game_stats_for_outlier, player_pre_stats)

                # Calculate sample weight for this training example
                # Default weight is 1.0, reduced for outliers
                example_sample_weight = 1.0

                # OT NORMALIZATION: Scale stats to regulation-equivalent
                # If player played 53 minutes (5 min OT), normalize to 48 min equivalent
                if outlier_info['outlier_type'] == 'overtime' and actual_min > 48:
                    ot_factor = 48.0 / actual_min
                    actual_pts = round(actual_pts * ot_factor, 1)
                    actual_reb = round(actual_reb * ot_factor, 1)
                    actual_ast = round(actual_ast * ot_factor, 1)
                    actual_fg3m = round(actual_fg3m * ot_factor, 1)
                    actual_pra = actual_pts + actual_reb + actual_ast
                    # Also reduce weight since OT games are less representative
                    example_sample_weight *= 0.7

                # BLOWOUT DETECTION: Down-weight games with 20+ point differential
                point_diff = abs(home_score - away_score)
                if point_diff >= 25:
                    example_sample_weight *= 0.5  # Heavy blowout - low weight
                elif point_diff >= 20:
                    example_sample_weight *= 0.7  # Moderate blowout

                # Also apply adjustment factor from outlier detection
                example_sample_weight *= outlier_info['adjustment_factor']

                # ENHANCEMENT: Add opponent context features for better prop predictions
                player_team_id = ps.get('team', {}).get('id')
                is_home = player_team_id == home_team_id
                opponent_team_id = away_team_id if is_home else home_team_id
                opponent_stats = away_stats if is_home else home_stats

                # Create enhanced features with opponent context
                enhanced_features = player_pre_stats.copy() if player_pre_stats else {}

                # Add opponent defensive context (ENHANCED)
                if opponent_stats:
                    # Core defensive ratings
                    enhanced_features['opp_def_rating'] = opponent_stats.get('def_rating', 114) or 114
                    enhanced_features['opp_off_rating'] = opponent_stats.get('off_rating', 114) or 114
                    enhanced_features['opp_net_rating'] = opponent_stats.get('net_rating', 0) or 0

                    # Points allowed (key for points props)
                    enhanced_features['opp_pts_allowed'] = opponent_stats.get('pts_allowed_avg', 114) or 114
                    enhanced_features['opp_pts_allowed_recent'] = opponent_stats.get('pts_allowed_recent', 114) or 114
                    enhanced_features['opp_pts_allowed_std'] = opponent_stats.get('pts_allowed_std', 5.0) or 5.0

                    # Pace (affects all props - more possessions = more stats)
                    enhanced_features['opp_pace'] = opponent_stats.get('pace', 100) or 100
                    enhanced_features['opp_pace_season'] = opponent_stats.get('pace_season', 100) or 100

                    # Defensive strength relative to league average (114 pts)
                    opp_def = enhanced_features['opp_pts_allowed_recent']
                    enhanced_features['opp_def_strength'] = (opp_def - 114.0) / 10.0  # Positive = bad defense (good for props)

                    # Rebound context
                    enhanced_features['opp_reb_factor'] = opponent_stats.get('reb_diff_factor', 0) or 0

                    # Home/away defensive splits
                    if is_home:
                        # Player is home, opponent is away - use opponent's away defense
                        enhanced_features['opp_location_def'] = opponent_stats.get('away_def_rating', 114) or 114
                    else:
                        # Player is away, opponent is home - use opponent's home defense
                        enhanced_features['opp_location_def'] = opponent_stats.get('home_def_rating', 112) or 112

                    # Win percentage context (good teams play tighter defense)
                    enhanced_features['opp_win_pct'] = opponent_stats.get('season_win_pct', 0.5) or 0.5
                    enhanced_features['opp_recent_win_pct'] = opponent_stats.get('recent_win_pct', 0.5) or 0.5
                else:
                    # Default values when opponent stats unavailable
                    enhanced_features['opp_def_rating'] = 114
                    enhanced_features['opp_off_rating'] = 114
                    enhanced_features['opp_net_rating'] = 0
                    enhanced_features['opp_pts_allowed'] = 114
                    enhanced_features['opp_pts_allowed_recent'] = 114
                    enhanced_features['opp_pts_allowed_std'] = 5.0
                    enhanced_features['opp_pace'] = 100
                    enhanced_features['opp_pace_season'] = 100
                    enhanced_features['opp_def_strength'] = 0
                    enhanced_features['opp_reb_factor'] = 0
                    enhanced_features['opp_location_def'] = 114
                    enhanced_features['opp_win_pct'] = 0.5
                    enhanced_features['opp_recent_win_pct'] = 0.5

                # Add game context
                enhanced_features['is_home'] = 1 if is_home else 0

                # Player's team context (if available)
                player_team_stats = home_stats if is_home else away_stats
                if player_team_stats:
                    enhanced_features['team_pace'] = player_team_stats.get('pace', 100) or 100
                    enhanced_features['team_off_rating'] = player_team_stats.get('off_rating', 114) or 114
                else:
                    enhanced_features['team_pace'] = 100
                    enhanced_features['team_off_rating'] = 114

                # TIER 2.2: Add position-specific opponent defense features
                # Get player position from their info
                player_position = ps.get('player', {}).get('position', 'F')
                pos_defense_features = position_defense_calc.get_position_defense_before_date(
                    team_id=opponent_team_id,
                    game_date=game_date,
                    player_position=player_position
                )
                enhanced_features.update(pos_defense_features)

                # NEW: Add blowout risk features (predicts when starters get pulled early)
                blowout_features = calculate_blowout_risk_features(
                    home_stats=home_stats,
                    away_stats=away_stats,
                    vegas_spread=None  # Historical training - no Vegas data available
                )
                enhanced_features.update(blowout_features)

                # NEW: Add pace-adjusted features (normalizes for team tempo)
                team_pace = enhanced_features.get('team_pace', 100)
                opp_pace = enhanced_features.get('opp_pace', 100)
                pace_features = calculate_pace_adjusted_features(
                    player_features=enhanced_features,
                    team_pace=team_pace,
                    opp_pace=opp_pace,
                    league_avg_pace=100.0
                )
                enhanced_features.update(pace_features)

                # NEW: Vegas total features (placeholder for historical training)
                # In live predictions, this will use actual Vegas totals
                vegas_total_features = calculate_vegas_total_features(
                    vegas_total=None,  # Not available in historical data
                    player_features=enhanced_features,
                    league_avg_total=225.0
                )
                enhanced_features.update(vegas_total_features)

                # ============================================================
                # REGRESSION-TO-MEAN FEATURES
                # Hot streaks cool off, cold streaks warm up
                # ============================================================
                games_played = enhanced_features.get('games_played', 10)

                # Points regression
                pts_season = enhanced_features.get('season_pts_avg', 15)
                pts_recent = enhanced_features.get('recent_pts_avg', pts_season)
                pts_deviation = pts_recent - pts_season
                # More games = more stable, less regression needed
                regression_weight = 0.4 * (1 - min(games_played, 50) / 50)
                enhanced_features['pts_deviation_from_mean'] = round(pts_deviation, 2)
                enhanced_features['pts_regression_adjustment'] = round(-pts_deviation * regression_weight, 2)
                enhanced_features['pts_regressed_estimate'] = round(pts_recent - (pts_deviation * regression_weight), 2)

                # Rebounds regression
                reb_season = enhanced_features.get('season_reb_avg', 5)
                reb_recent = enhanced_features.get('recent_reb_avg', reb_season)
                reb_deviation = reb_recent - reb_season
                enhanced_features['reb_deviation_from_mean'] = round(reb_deviation, 2)
                enhanced_features['reb_regression_adjustment'] = round(-reb_deviation * regression_weight, 2)

                # Assists regression
                ast_season = enhanced_features.get('season_ast_avg', 3)
                ast_recent = enhanced_features.get('recent_ast_avg', ast_season)
                ast_deviation = ast_recent - ast_season
                enhanced_features['ast_deviation_from_mean'] = round(ast_deviation, 2)
                enhanced_features['ast_regression_adjustment'] = round(-ast_deviation * regression_weight, 2)

                # 3PM regression (higher variance stat, more regression)
                fg3_season = enhanced_features.get('season_fg3_avg', 1)
                fg3_recent = enhanced_features.get('recent_fg3_avg', fg3_season)
                fg3_deviation = fg3_recent - fg3_season
                enhanced_features['fg3_deviation_from_mean'] = round(fg3_deviation, 2)
                enhanced_features['fg3_regression_adjustment'] = round(-fg3_deviation * regression_weight * 1.2, 2)  # Extra regression for high-variance stat

                player_data.append({
                    'player_id': player_id,
                    'player_name': f"{ps.get('player', {}).get('first_name', '')} {ps.get('player', {}).get('last_name', '')}",
                    'game_date': game_date,
                    'team_id': player_team_id,
                    'opponent_team_id': opponent_team_id,
                    'features': enhanced_features,
                    'actual_pts': actual_pts,
                    'actual_reb': actual_reb,
                    'actual_ast': actual_ast,
                    'actual_fg3m': actual_fg3m,
                    'actual_pra': actual_pra,
                    'actual_min': actual_min,  # TIER 2.3: For minutes model
                    'sample_weight': example_sample_weight,  # NEW: For weighted training
                })

        # Skip team example if insufficient history
        if not home_stats or not away_stats:
            skipped += 1
            continue

        # Create team training example with ALL features
        team_features = {
            # === ORIGINAL FEATURES ===
            'season_win_pct_diff': home_stats['season_win_pct'] - away_stats['season_win_pct'],
            'recent_win_pct_diff': home_stats['recent_win_pct'] - away_stats['recent_win_pct'],
            'pts_avg_diff': home_stats['season_pts_avg'] - away_stats['season_pts_avg'],
            'recent_pts_diff': home_stats['recent_pts_avg'] - away_stats['recent_pts_avg'],
            'off_rating_diff': home_stats['off_rating'] - away_stats['off_rating'],
            'def_rating_diff': home_stats['def_rating'] - away_stats['def_rating'],
            'net_rating_diff': home_stats['net_rating'] - away_stats['net_rating'],
            'location_win_pct_diff': home_stats['home_win_pct'] - away_stats['away_win_pct'],
            'home_advantage_factor': home_stats['home_win_pct'] - home_stats['away_win_pct'],
            'home_season_win_pct': home_stats['season_win_pct'],
            'away_season_win_pct': away_stats['season_win_pct'],
            'home_net_rating': home_stats['net_rating'],
            'away_net_rating': away_stats['net_rating'],
            'home_off_rating': home_stats['off_rating'],
            'away_off_rating': away_stats['off_rating'],
            'home_def_rating': home_stats['def_rating'],
            'away_def_rating': away_stats['def_rating'],
            'home_pts_avg': home_stats['season_pts_avg'],
            'away_pts_avg': away_stats['season_pts_avg'],
            'home_games_played': home_stats['season_games'],
            'away_games_played': away_stats['season_games'],
            'expected_point_diff': home_stats['home_pts_avg'] - away_stats['away_pts_avg'],
            'plus_minus_diff': home_stats['recent_point_diff'] - away_stats['recent_point_diff'],

            # === NEW: ELO RATING FEATURES ===
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo,
            'elo_win_prob': elo_win_prob,  # Elo-based home win probability
            'elo_spread': elo_spread,  # Elo-based spread prediction

            # === NEW: REST & FATIGUE FEATURES ===
            'home_days_rest': home_days_rest,
            'away_days_rest': away_days_rest,
            'rest_advantage': home_days_rest - away_days_rest,  # Positive = home more rested
            'home_is_b2b': 1 if home_is_b2b else 0,
            'away_is_b2b': 1 if away_is_b2b else 0,
            'b2b_disadvantage': (1 if away_is_b2b else 0) - (1 if home_is_b2b else 0),  # Positive = away on B2B

            # === NEW: TRAVEL FEATURES ===
            'home_travel_distance': home_travel_features['travel_distance'],
            'away_travel_distance': away_travel_features['travel_distance'],
            'travel_distance_diff': away_travel_features['travel_distance'] - home_travel_features['travel_distance'],
            'home_timezone_change': abs(home_travel_features['timezone_change']),
            'away_timezone_change': abs(away_travel_features['timezone_change']),
            'timezone_advantage': abs(away_travel_features['timezone_change']) - abs(home_travel_features['timezone_change']),
            'home_altitude_disadvantage': home_travel_features['altitude_disadvantage'],
            'away_altitude_disadvantage': away_travel_features['altitude_disadvantage'],
            'home_travel_fatigue': home_travel_features['travel_fatigue_score'],
            'away_travel_fatigue': away_travel_features['travel_fatigue_score'],
            'fatigue_advantage': away_travel_features['travel_fatigue_score'] - home_travel_features['travel_fatigue_score'],
            'away_coast_to_coast': away_travel_features['coast_to_coast'],

            # === NEW: PACE FEATURES (for spread variance) ===
            'home_pace': home_stats.get('pace', 100),
            'away_pace': away_stats.get('pace', 100),
            'expected_pace': (home_stats.get('pace', 100) + away_stats.get('pace', 100)) / 2,
            'pace_diff': home_stats.get('pace', 100) - away_stats.get('pace', 100),

            # === NEW: COMBINED SITUATIONAL FEATURES ===
            # Road B2B vs rested home team is the worst spot (~2-4 points)
            'road_b2b_vs_rested': 1 if (away_is_b2b and home_days_rest >= 2) else 0,
            # Long travel + B2B is especially bad
            'away_tired_traveler': 1 if (away_is_b2b and away_travel_features['travel_distance'] > 1000) else 0,

            # === NEW: SCHEDULE SPOT FEATURES ===
            # Home team schedule spots
            'home_letdown_spot': home_schedule_spots['letdown_spot'],
            'home_trap_game': home_schedule_spots['trap_game'],
            'home_sandwich_game': home_schedule_spots['sandwich_game'],
            'home_road_trip_fatigue': home_schedule_spots['road_trip_fatigue'],
            'home_revenge_game': home_schedule_spots['revenge_game'],
            'home_long_homestand': home_schedule_spots['long_homestand'],
            'home_early_season': home_schedule_spots['early_season_variance'],
            'home_schedule_spot_score': home_schedule_spots['schedule_spot_score'],

            # Away team schedule spots
            'away_letdown_spot': away_schedule_spots['letdown_spot'],
            'away_trap_game': away_schedule_spots['trap_game'],
            'away_sandwich_game': away_schedule_spots['sandwich_game'],
            'away_road_trip_fatigue': away_schedule_spots['road_trip_fatigue'],
            'away_revenge_game': away_schedule_spots['revenge_game'],
            'away_long_homestand': away_schedule_spots['long_homestand'],
            'away_early_season': away_schedule_spots['early_season_variance'],
            'away_schedule_spot_score': away_schedule_spots['schedule_spot_score'],

            # Combined schedule spot advantage (positive = home team favored by spots)
            'schedule_spot_advantage': home_schedule_spots['schedule_spot_score'] - away_schedule_spots['schedule_spot_score'],

            # === LINE MOVEMENT FEATURES (for live predictions) ===
            # These are placeholders during training (no historical line data)
            # Filled in during live predictions with actual line movement
            'spread_movement': 0.0,
            'spread_movement_abs': 0.0,
            'spread_moved_toward_home': 0,
            'spread_moved_toward_away': 0,
            'total_movement': 0.0,
            'total_movement_abs': 0.0,
            'total_moved_up': 0,
            'total_moved_down': 0,
            'model_vs_market_spread': 0.0,
            'model_disagrees_spread': 0,
            'model_favors_home_more': 0,
            'large_spread_move': 0,
            'large_total_move': 0,
            'line_has_moved': 0,
        }

        team_data.append({
            'game_id': game_id,
            'game_date': game_date,
            'home_team': home_team.get('abbreviation', ''),
            'away_team': away_team.get('abbreviation', ''),
            'features': team_features,
            'home_win': home_score > away_score,
            'point_differential': home_score - away_score,
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(games_sorted)} games...")

    print(f"  Team examples: {len(team_data)}")
    print(f"  Player examples: {len(player_data)}")
    print(f"  Skipped (insufficient history): {skipped}")

    return team_data, player_data


# =============================================================================
# MODEL TRAINING
# =============================================================================

class PropModel:
    """Model for predicting player props with K-Fold cross-validation."""

    def __init__(self, prop_type: str, use_xgboost: bool = True):
        self.prop_type = prop_type
        self.use_xgboost = use_xgboost and HAS_XGBOOST

        # Use XGBoost if available for better performance
        # TUNED: Adjusted hyperparameters for better generalization
        if self.use_xgboost:
            self.model = XGBRegressor(
                n_estimators=300,           # More trees for better learning
                max_depth=5,                # Slightly shallower to prevent overfitting
                learning_rate=0.05,         # Slower learning rate
                min_child_weight=5,         # Regularization: min samples per leaf
                subsample=0.8,
                colsample_bytree=0.7,       # Use fewer features per tree
                reg_alpha=0.5,              # L1 regularization
                reg_lambda=2.0,             # L2 regularization
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                random_state=42,
            )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_metrics = {}
        self.cv_scores = {}

    def train(self, X: pd.DataFrame, y: np.ndarray, sample_weights: np.ndarray = None,
              test_size: float = 0.2, n_cv_folds: int = 5, use_time_series_cv: bool = True) -> Dict:
        """Train the prop model with K-Fold cross-validation (Phase 5).

        Args:
            X: Feature DataFrame
            y: Target values
            sample_weights: Optional sample weights for time decay
            test_size: Holdout test size
            n_cv_folds: Number of cross-validation folds
            use_time_series_cv: Use TimeSeriesSplit (True) or regular KFold (False)

        Returns:
            Training metrics dictionary
        """
        from sklearn.model_selection import TimeSeriesSplit, KFold

        self.feature_names = list(X.columns)
        X_filled = smart_fillna(X).values
        y_arr = np.array(y)

        # Create CV splitter
        if use_time_series_cv:
            cv = TimeSeriesSplit(n_splits=n_cv_folds)
            cv_name = "TimeSeriesSplit"
        else:
            cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            cv_name = "KFold"

        # Run cross-validation
        cv_rmse = []
        cv_mae = []
        cv_r2 = []

        print(f"    Running {cv_name} with {n_cv_folds} folds...")
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_filled)):
            X_cv_train, X_cv_val = X_filled[train_idx], X_filled[val_idx]
            y_cv_train, y_cv_val = y_arr[train_idx], y_arr[val_idx]

            # Get sample weights for this fold
            w_cv_train = sample_weights[train_idx] if sample_weights is not None else None

            # Fit scaler on training fold
            fold_scaler = StandardScaler()
            X_cv_train_scaled = fold_scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = fold_scaler.transform(X_cv_val)

            # Create fresh model for each fold
            if self.use_xgboost:
                fold_model = XGBRegressor(
                    n_estimators=150, max_depth=6, learning_rate=0.08,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    n_jobs=-1, verbosity=0
                )
            else:
                fold_model = GradientBoostingRegressor(
                    n_estimators=150, max_depth=5, learning_rate=0.08,
                    subsample=0.8, random_state=42
                )

            # Fit with sample weights if available
            if w_cv_train is not None:
                fold_model.fit(X_cv_train_scaled, y_cv_train, sample_weight=w_cv_train)
            else:
                fold_model.fit(X_cv_train_scaled, y_cv_train)

            # Evaluate on validation fold
            y_cv_pred = fold_model.predict(X_cv_val_scaled)
            fold_rmse = np.sqrt(mean_squared_error(y_cv_val, y_cv_pred))
            fold_mae = mean_absolute_error(y_cv_val, y_cv_pred)
            fold_r2 = r2_score(y_cv_val, y_cv_pred)

            cv_rmse.append(fold_rmse)
            cv_mae.append(fold_mae)
            cv_r2.append(fold_r2)

        # Store CV scores
        self.cv_scores = {
            'cv_type': cv_name,
            'n_folds': n_cv_folds,
            'rmse_mean': np.mean(cv_rmse),
            'rmse_std': np.std(cv_rmse),
            'mae_mean': np.mean(cv_mae),
            'mae_std': np.std(cv_mae),
            'r2_mean': np.mean(cv_r2),
            'r2_std': np.std(cv_r2),
            'fold_rmse': cv_rmse,
        }
        print(f"    CV RMSE: {self.cv_scores['rmse_mean']:.3f} ± {self.cv_scores['rmse_std']:.3f}")
        print(f"    CV R²: {self.cv_scores['r2_mean']:.4f} ± {self.cv_scores['r2_std']:.4f}")

        # Now train final model on full training data
        # FIXED: Use chronological split instead of random to prevent data leakage
        # Data should already be sorted by date, so we split at a percentage point
        split_idx = int(len(X_filled) * (1 - test_size))

        X_train = X_filled[:split_idx]
        X_test = X_filled[split_idx:]
        y_train = y_arr[:split_idx]
        y_test = y_arr[split_idx:]

        if sample_weights is not None:
            w_train = sample_weights[:split_idx]
        else:
            w_train = None

        # Fit final scaler and model
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if w_train is not None:
            self.model.fit(X_train_scaled, y_train, sample_weight=w_train)
        else:
            self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Final holdout evaluation
        y_pred = self.model.predict(X_test_scaled)

        self.training_metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'train_size': len(X_train),
            'test_size': len(X_test),
            # Include CV scores in metrics
            'cv_rmse_mean': self.cv_scores['rmse_mean'],
            'cv_rmse_std': self.cv_scores['rmse_std'],
            'cv_r2_mean': self.cv_scores['r2_mean'],
            'cv_r2_std': self.cv_scores['r2_std'],
            'model_type': 'XGBoost' if self.use_xgboost else 'GradientBoosting',
        }

        # Calculate dynamic bias correction from holdout predictions
        self._calculate_dynamic_bias(y_test, y_pred)

        return self.training_metrics

    def _calculate_dynamic_bias(self, y_actual: np.ndarray, y_predicted: np.ndarray):
        """
        Calculate dynamic bias correction from recent predictions.

        This replaces hard-coded bias corrections with data-driven values.
        Bias = mean(actual) - mean(predicted)
        Positive bias means model underpredicts, negative means overpredicts.
        """
        if len(y_actual) < 50:
            # Not enough data - use fallback defaults
            self.dynamic_bias = 0.0
            return

        bias = np.mean(y_actual) - np.mean(y_predicted)

        # Clip extreme bias values (model shouldn't be off by more than 3 points)
        self.dynamic_bias = np.clip(bias, -3.0, 3.0)

        print(f"    Dynamic bias correction for {self.prop_type}: {self.dynamic_bias:.3f}")

    # DEPRECATED: Legacy hard-coded bias corrections (kept for reference only)
    # These are now replaced by dynamic bias calculation in train()
    LEGACY_BIAS_CORRECTIONS = {
        'points': 0.94,      # Historical: Model underpredicted by ~0.94
        'rebounds': -0.21,   # Historical: Model slightly overpredicted
        'assists': 0.43,     # Historical: Model underpredicted by ~0.43
        'threes': 0.70,      # Historical: Model underpredicted by ~0.70
        'pra': -0.82,        # Historical: Model slightly overpredicted
    }

    def predict(self, features: Dict, prop_line: float = None, apply_bias_correction: bool = True) -> Dict:
        """Make a prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = smart_fillna(X[self.feature_names])
        X_scaled = self.scaler.transform(X)

        predicted = float(self.model.predict(X_scaled)[0])

        # Apply dynamic bias correction (calculated during training)
        if apply_bias_correction and hasattr(self, 'dynamic_bias') and self.dynamic_bias != 0:
            predicted += self.dynamic_bias

        result = {
            'predicted_value': predicted,
            'prop_type': self.prop_type,
        }

        if prop_line is not None:
            result['prop_line'] = prop_line
            result['prediction'] = 'over' if predicted > prop_line else 'under'
            result['edge'] = predicted - prop_line

        return result

    def save(self, filepath: Path):
        """Save model to disk with CV scores and dynamic bias."""
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'cv_scores': self.cv_scores,
            'prop_type': self.prop_type,
            'model_type': 'XGBoost' if self.use_xgboost else 'GradientBoosting',
            'dynamic_bias': getattr(self, 'dynamic_bias', 0.0),  # NEW: Save dynamic bias
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: Path) -> 'PropModel':
        """Load model from disk with dynamic bias support."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(data['prop_type'])
        model.model = data['model']
        model.scaler = data['scaler']
        model.feature_names = data['feature_names']
        model.training_metrics = data.get('training_metrics', {})
        model.cv_scores = data.get('cv_scores', {})
        model.dynamic_bias = data.get('dynamic_bias', 0.0)  # NEW: Load dynamic bias
        model.is_fitted = True
        return model


# =============================================================================
# MINUTES PREDICTION MODEL (TIER 2.3)
# =============================================================================

class MinutesPredictionModel:
    """
    Two-stage model for predicting player minutes.

    Stage 1: Binary classifier - Will player see significant minutes (>5)?
    Stage 2: Regression - If yes, how many minutes?

    This helps avoid predicting stats for DNP (Did Not Play) players.
    """

    def __init__(self):
        self.will_play_classifier = None  # Predicts if player plays (>5 min)
        self.minutes_regressor = None      # Predicts actual minutes
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_metrics = {}

        # Minutes-specific feature importance
        self.min_features_used = [
            'season_min_avg', 'recent_min_avg', 'last5_min_avg', 'last3_min_avg',
            'is_starter', 'is_star', 'days_rest', 'is_back_to_back',
            'is_home', 'min_consistency'
        ]

    def train(self, X: pd.DataFrame, y_minutes: np.ndarray,
              sample_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Train both stages of the minutes prediction model.

        Args:
            X: Feature matrix
            y_minutes: Actual minutes played
            sample_weights: Optional sample weights for time decay
        """
        print(f"  Training MinutesPredictionModel (TIER 2.3)...")

        # Store feature names
        self.feature_names = list(X.columns)

        # Handle missing values with smart imputation
        X_clean = smart_fillna(X).values
        X_scaled = self.scaler.fit_transform(X_clean)

        # Stage 1: Binary classification - will player play?
        # Define "playing" as >5 minutes (to filter out garbage time cameos)
        y_will_play = (y_minutes >= 5).astype(int)

        print(f"    Stage 1: Will-Play Classifier")
        n_played = y_will_play.sum()
        n_dnp = len(y_will_play) - n_played
        print(f"      Players who played (>5 min): {n_played} ({100*y_will_play.mean():.1f}%)")
        print(f"      Players who didn't play: {n_dnp}")

        # Check if we have both classes
        clf_accuracy, clf_precision, clf_recall, clf_f1 = 1.0, 1.0, 1.0, 1.0  # Defaults

        if n_dnp < 10:
            # Not enough DNP samples to train classifier
            # We'll use minutes regressor alone with a threshold
            print(f"      [SKIP] Not enough DNP samples to train classifier")
            print(f"      Will use minutes threshold (5 min) instead")
            self.will_play_classifier = None
            self.classifier_skipped = True
        else:
            self.classifier_skipped = False
            # Use XGBoost for classification
            if HAS_XGBOOST:
                self.will_play_classifier = XGBClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    min_child_weight=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=2.0,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss',
                )
            else:
                self.will_play_classifier = GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    min_samples_leaf=10,
                    random_state=42,
                )

            # Train classifier
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                X_scaled, y_will_play, test_size=0.2, random_state=42
            )

            if sample_weights is not None:
                sw_train, sw_test = train_test_split(sample_weights, test_size=0.2, random_state=42)
                self.will_play_classifier.fit(X_train_c, y_train_c, sample_weight=sw_train)
            else:
                self.will_play_classifier.fit(X_train_c, y_train_c)

            # Evaluate classifier
            y_pred_c = self.will_play_classifier.predict(X_test_c)
            y_prob_c = self.will_play_classifier.predict_proba(X_test_c)[:, 1]

            clf_accuracy = accuracy_score(y_test_c, y_pred_c)
            clf_precision = precision_score(y_test_c, y_pred_c, zero_division=0)
            clf_recall = recall_score(y_test_c, y_pred_c, zero_division=0)
            clf_f1 = f1_score(y_test_c, y_pred_c, zero_division=0)

            print(f"      Accuracy: {clf_accuracy:.4f}")
            print(f"      Precision: {clf_precision:.4f}")
            print(f"      Recall: {clf_recall:.4f}")
            print(f"      F1 Score: {clf_f1:.4f}")

        # Stage 2: Regression for actual minutes (only for players who played)
        print(f"    Stage 2: Minutes Regressor (players with >5 min)")

        # Filter to only players who played
        played_mask = y_minutes >= 5
        X_played = X_scaled[played_mask]
        y_played = y_minutes[played_mask]

        if sample_weights is not None:
            sw_played = sample_weights[played_mask]
        else:
            sw_played = None

        print(f"      Training samples: {len(y_played)}")
        print(f"      Minutes range: {y_played.min():.1f} - {y_played.max():.1f}")

        # Use XGBoost for regression
        if HAS_XGBOOST:
            self.minutes_regressor = XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.minutes_regressor = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                min_samples_leaf=5,
                random_state=42,
            )

        # Train regressor
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_played, y_played, test_size=0.2, random_state=42
        )

        if sw_played is not None:
            sw_train_r, _ = train_test_split(sw_played, test_size=0.2, random_state=42)
            self.minutes_regressor.fit(X_train_r, y_train_r, sample_weight=sw_train_r)
        else:
            self.minutes_regressor.fit(X_train_r, y_train_r)

        # Evaluate regressor
        y_pred_r = self.minutes_regressor.predict(X_test_r)

        reg_rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
        reg_mae = mean_absolute_error(y_test_r, y_pred_r)
        reg_r2 = r2_score(y_test_r, y_pred_r)

        print(f"      RMSE: {reg_rmse:.2f} minutes")
        print(f"      MAE: {reg_mae:.2f} minutes")
        print(f"      R²: {reg_r2:.4f}")

        # Store metrics
        self.training_metrics = {
            'classifier': {
                'accuracy': clf_accuracy,
                'precision': clf_precision,
                'recall': clf_recall,
                'f1': clf_f1,
            },
            'regressor': {
                'rmse': reg_rmse,
                'mae': reg_mae,
                'r2': reg_r2,
            },
            'n_samples': len(y_minutes),
            'n_played': len(y_played),
            'play_rate': y_will_play.mean(),
        }

        self.is_fitted = True
        return self.training_metrics

    def predict(self, X: np.ndarray) -> Tuple[float, float]:
        """
        Predict minutes for a player.

        Returns:
            Tuple of (predicted_minutes, play_probability)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_arr = np.array(X)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1, -1)

        X_scaled = self.scaler.transform(X_arr)

        # Get predicted minutes (if playing)
        raw_minutes = self.minutes_regressor.predict(X_scaled)[0]

        # Get probability of playing
        if self.will_play_classifier is not None:
            play_prob = self.will_play_classifier.predict_proba(X_scaled)[0, 1]
            # Blend: if low play probability, reduce predicted minutes
            predicted_minutes = raw_minutes * min(play_prob * 1.5, 1.0)
        else:
            # No classifier trained - use minutes threshold heuristic
            # If predicted minutes < 5, set low play probability
            play_prob = 1.0 if raw_minutes >= 5 else raw_minutes / 5.0
            predicted_minutes = raw_minutes

        # Clamp minutes to realistic range
        predicted_minutes = max(0, min(48, predicted_minutes))

        return predicted_minutes, play_prob

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict minutes for multiple players."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_arr = np.array(X)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1, -1)

        X_scaled = self.scaler.transform(X_arr)

        raw_minutes = self.minutes_regressor.predict(X_scaled)

        if self.will_play_classifier is not None:
            play_probs = self.will_play_classifier.predict_proba(X_scaled)[:, 1]
            # Blend based on play probability
            predicted_minutes = raw_minutes * np.minimum(play_probs * 1.5, 1.0)
        else:
            # No classifier - use minutes threshold heuristic
            play_probs = np.where(raw_minutes >= 5, 1.0, raw_minutes / 5.0)
            predicted_minutes = raw_minutes

        predicted_minutes = np.clip(predicted_minutes, 0, 48)

        return predicted_minutes, play_probs

    def save(self, filepath: Path):
        """Save the minutes prediction model."""
        data = {
            'will_play_classifier': self.will_play_classifier,
            'minutes_regressor': self.minutes_regressor,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'is_fitted': self.is_fitted,
            'classifier_skipped': getattr(self, 'classifier_skipped', False),
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"    Saved MinutesPredictionModel: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'MinutesPredictionModel':
        """Load a saved minutes prediction model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls()
        model.will_play_classifier = data['will_play_classifier']
        model.minutes_regressor = data['minutes_regressor']
        model.scaler = data['scaler']
        model.feature_names = data['feature_names']
        model.training_metrics = data['training_metrics']
        model.is_fitted = data['is_fitted']
        model.classifier_skipped = data.get('classifier_skipped', False)
        return model


class PropEnsembleModel:
    """
    Ensemble model for player prop predictions.

    Combines multiple algorithms for more robust predictions:
    - XGBoost (primary)
    - LightGBM (fast, handles categorical)
    - Random Forest (diverse)
    - Ridge Regression (linear baseline)

    Uses stacking with a meta-learner for final predictions.
    Also provides over/under probability estimates for betting.
    """

    def __init__(self, prop_type: str, optimized_params: Optional[Dict[str, Dict]] = None):
        self.prop_type = prop_type
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_metrics = {}
        self.cv_scores = {}
        self.meta_model = None  # For stacking
        self.over_under_classifier = None  # For probability estimates
        self.optimized_params = optimized_params  # Store for save/load

        # Initialize base models with optimized params if provided
        self._init_base_models(optimized_params)

    def _init_base_models(self, optimized_params: Optional[Dict[str, Dict]] = None):
        """Initialize the ensemble of base models with optional optimized parameters."""

        # Default parameters
        xgb_defaults = {
            'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
            'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.5, 'reg_lambda': 2.0, 'random_state': 42,
            'n_jobs': -1, 'verbosity': 0,
        }
        lgb_defaults = {
            'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
            'num_leaves': 31, 'min_child_samples': 20, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        rf_defaults = {
            'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 10,
            'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': 42,
            'n_jobs': -1,
        }
        ridge_defaults = {'alpha': 1.0, 'random_state': 42}
        gb_defaults = {
            'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05,
            'subsample': 0.8, 'random_state': 42,
        }
        # NEW: CatBoost defaults (good for handling categorical features like position)
        catboost_defaults = {
            'iterations': 300, 'depth': 5, 'learning_rate': 0.03,
            'l2_leaf_reg': 3.0, 'random_seed': 42,
            'verbose': False, 'thread_count': -1,
        }

        # Merge with optimized params if provided
        if optimized_params:
            xgb_params = {**xgb_defaults, **optimized_params.get('xgboost', {})}
            lgb_params = {**lgb_defaults, **optimized_params.get('lightgbm', {})}
            rf_params = {**rf_defaults, **optimized_params.get('random_forest', {})}
            ridge_params = {**ridge_defaults, **optimized_params.get('ridge', {})}
            gb_params = {**gb_defaults, **optimized_params.get('gradient_boosting', {})}
            catboost_params = {**catboost_defaults, **optimized_params.get('catboost', {})}
        else:
            xgb_params, lgb_params, rf_params = xgb_defaults, lgb_defaults, rf_defaults
            ridge_params, gb_params = ridge_defaults, gb_defaults
            catboost_params = catboost_defaults

        # XGBoost (primary)
        if HAS_XGBOOST:
            self.models['xgboost'] = XGBRegressor(**xgb_params)

        # LightGBM (fast)
        if HAS_LIGHTGBM:
            self.models['lightgbm'] = LGBMRegressor(**lgb_params)

        # NEW: CatBoost (excellent for categorical features like position, handles missing values)
        if HAS_CATBOOST:
            from catboost import CatBoostRegressor
            self.models['catboost'] = CatBoostRegressor(**catboost_params)

        # Random Forest (diverse)
        self.models['random_forest'] = RandomForestRegressor(**rf_params)

        # Ridge Regression (linear baseline)
        self.models['ridge'] = Ridge(**ridge_params)

        # Gradient Boosting (fallback if XGBoost unavailable, also provides diversity)
        if not HAS_XGBOOST:
            self.models['gradient_boosting'] = GradientBoostingRegressor(**gb_params)

        print(f"    Initialized ensemble with {len(self.models)} models: {list(self.models.keys())}")
        if optimized_params:
            print(f"    Using Optuna-optimized hyperparameters")

    def train(self, X: pd.DataFrame, y: np.ndarray, dates: List[str] = None,
              sample_weights: np.ndarray = None, test_size: float = 0.2) -> Dict:
        """
        Train the ensemble model.

        Args:
            X: Feature DataFrame
            y: Target values (actual points/rebounds/etc)
            dates: Game dates for chronological split
            sample_weights: Optional time-decay weights
            test_size: Holdout test size

        Returns:
            Training metrics dictionary
        """
        from sklearn.model_selection import TimeSeriesSplit

        self.feature_names = list(X.columns)
        X_filled = smart_fillna(X).values
        y_arr = np.array(y)

        # Chronological split (data assumed sorted by date)
        split_idx = int(len(X_filled) * (1 - test_size))
        X_train, X_test = X_filled[:split_idx], X_filled[split_idx:]
        y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]

        if sample_weights is not None:
            w_train = sample_weights[:split_idx]
        else:
            w_train = None

        # Fit scaler
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train each base model
        base_predictions_train = []
        base_predictions_test = []
        model_metrics = {}

        print(f"    Training {len(self.models)} base models...")
        for name, model in self.models.items():
            try:
                # Train with sample weights if supported
                if w_train is not None and hasattr(model, 'fit') and 'sample_weight' in str(model.fit.__code__.co_varnames):
                    model.fit(X_train_scaled, y_train, sample_weight=w_train)
                else:
                    model.fit(X_train_scaled, y_train)

                # Get predictions for stacking
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)

                base_predictions_train.append(train_pred)
                base_predictions_test.append(test_pred)

                # Calculate metrics for this model
                rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                mae = mean_absolute_error(y_test, test_pred)
                r2 = r2_score(y_test, test_pred)

                model_metrics[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
                print(f"      {name}: RMSE={rmse:.3f}, R²={r2:.4f}")

            except Exception as e:
                print(f"      Error training {name}: {e}")

        # Create stacked features (base model predictions)
        stacked_train = np.column_stack(base_predictions_train)
        stacked_test = np.column_stack(base_predictions_test)

        # Use inverse-RMSE weighted average instead of meta-learner
        # This avoids the issue of meta-learner overfitting to training predictions
        model_weights = {}
        total_inverse_rmse = 0
        for name, metrics in model_metrics.items():
            # Inverse RMSE weighting - better models get higher weight
            inv_rmse = 1.0 / (metrics['rmse'] + 0.1)  # +0.1 to avoid division issues
            model_weights[name] = inv_rmse
            total_inverse_rmse += inv_rmse

        # Normalize weights
        self.model_weights = {k: v / total_inverse_rmse for k, v in model_weights.items()}

        # Calculate weighted ensemble predictions
        ensemble_pred = np.zeros(len(y_test))
        for i, (name, _) in enumerate(self.models.items()):
            weight = self.model_weights.get(name, 1.0 / len(self.models))
            ensemble_pred += weight * base_predictions_test[i]

        # TIER 2.4: Upgraded meta-learner from Ridge to XGBoost for better stacking
        # XGBoost can learn non-linear combinations of base model predictions
        if HAS_XGBOOST:
            self.meta_model = XGBRegressor(
                n_estimators=50,        # Small ensemble for meta-learner
                max_depth=2,            # Very shallow - just learning model weights
                learning_rate=0.1,
                min_child_weight=5,     # Prevent overfitting
                subsample=0.8,
                colsample_bytree=1.0,   # Use all base model predictions
                reg_alpha=0.5,          # L1 regularization
                reg_lambda=1.0,         # L2 regularization
                random_state=42,
                n_jobs=-1,
            )
        else:
            # Fallback to Ridge if XGBoost not available
            self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(stacked_train, y_train)

        # Calculate ensemble metrics
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)

        print(f"    Ensemble: RMSE={ensemble_rmse:.3f}, MAE={ensemble_mae:.3f}, R²={ensemble_r2:.4f}")

        # Train over/under classifier for probability estimates
        self._train_over_under_classifier(X_train_scaled, y_train, X_test_scaled, y_test)

        self.is_fitted = True

        self.training_metrics = {
            'ensemble_rmse': ensemble_rmse,
            'ensemble_mae': ensemble_mae,
            'ensemble_r2': ensemble_r2,
            'model_metrics': model_metrics,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_models': len(self.models),
        }

        return self.training_metrics

    def _train_over_under_classifier(self, X_train, y_train, X_test, y_test):
        """Train a classifier for over/under probability estimates."""
        # We'll train on residuals relative to various line values
        # For now, use a simple approach: classify whether actual > predicted

        # Get ensemble predictions on training data
        base_preds_train = []
        for name, model in self.models.items():
            try:
                pred = model.predict(X_train)
                base_preds_train.append(pred)
            except:
                pass

        if not base_preds_train:
            return

        stacked_train = np.column_stack(base_preds_train)
        ensemble_pred_train = self.meta_model.predict(stacked_train)

        # Create binary labels: 1 if actual > predicted, 0 otherwise
        # This helps calibrate confidence
        residuals = y_train - ensemble_pred_train
        binary_labels = (residuals > 0).astype(int)

        # Simple logistic regression for probability calibration
        try:
            from sklearn.linear_model import LogisticRegression as LR
            # Use residual magnitude as feature
            residual_features = np.column_stack([
                ensemble_pred_train,
                np.abs(ensemble_pred_train - np.mean(y_train)),
            ])
            self.over_under_classifier = LR(random_state=42, max_iter=1000)
            self.over_under_classifier.fit(residual_features, binary_labels)
        except Exception as e:
            print(f"      Warning: Could not train over/under classifier: {e}")

    def predict(self, features: Dict, prop_line: float = None) -> Dict:
        """Make a prediction with the ensemble."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = smart_fillna(X[self.feature_names])
        X_scaled = self.scaler.transform(X)

        # Get base model predictions
        base_preds = []
        individual_preds = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[0]
                base_preds.append(pred)
                individual_preds[name] = pred
            except:
                pass

        if not base_preds:
            raise ValueError("No base models available for prediction")

        # Weighted average prediction (more robust than meta-learner)
        ensemble_pred = 0.0
        if hasattr(self, 'model_weights') and self.model_weights:
            for name, pred in individual_preds.items():
                weight = self.model_weights.get(name, 1.0 / len(individual_preds))
                ensemble_pred += weight * pred
        else:
            # Fallback to simple average
            ensemble_pred = float(np.mean(base_preds))

        result = {
            'predicted_value': ensemble_pred,
            'prop_type': self.prop_type,
            'individual_predictions': individual_preds,
            'model_agreement': 1 - (np.std(base_preds) / max(np.mean(base_preds), 1)),  # Higher = more agreement
        }

        if prop_line is not None:
            result['prop_line'] = prop_line
            result['prediction'] = 'over' if ensemble_pred > prop_line else 'under'
            result['edge'] = ensemble_pred - prop_line
            result['edge_pct'] = (ensemble_pred - prop_line) / prop_line if prop_line > 0 else 0

            # Over/under probability (if classifier available)
            if self.over_under_classifier is not None:
                try:
                    residual_features = np.array([[ensemble_pred, abs(ensemble_pred - prop_line)]])
                    proba = self.over_under_classifier.predict_proba(residual_features)[0]
                    result['over_probability'] = float(proba[1])
                    result['under_probability'] = float(proba[0])
                except:
                    result['over_probability'] = 0.5 + (result['edge_pct'] * 2)  # Simple estimate
                    result['over_probability'] = max(0.3, min(0.7, result['over_probability']))
                    result['under_probability'] = 1 - result['over_probability']

        return result

    def save(self, filepath: Path):
        """Save ensemble model to disk."""
        data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'meta_model': self.meta_model,
            'model_weights': getattr(self, 'model_weights', {}),
            'over_under_classifier': self.over_under_classifier,
            'prop_type': self.prop_type,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: Path) -> 'PropEnsembleModel':
        """Load ensemble model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(data['prop_type'])
        model.models = data['models']
        model.scaler = data['scaler']
        model.feature_names = data['feature_names']
        model.training_metrics = data.get('training_metrics', {})
        model.meta_model = data.get('meta_model')
        model.model_weights = data.get('model_weights', {})
        model.over_under_classifier = data.get('over_under_classifier')
        model.is_fitted = True

        return model


# =============================================================================
# QUANTILE REGRESSION FOR UNCERTAINTY ESTIMATION
# =============================================================================

class QuantilePropModel:
    """
    Quantile regression model for uncertainty estimation in prop predictions.

    Instead of predicting a single point estimate, this model predicts the
    full distribution of possible outcomes by training separate models at
    different quantiles (10th, 25th, 50th, 75th, 90th percentiles).

    This enables:
    1. Better probability estimation for OVER/UNDER decisions
    2. Confidence intervals for predictions
    3. Risk-adjusted betting recommendations
    """

    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

    def __init__(self, prop_type: str):
        self.prop_type = prop_type
        self.quantile_models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_metrics = {}

    def train(self, X: pd.DataFrame, y: np.ndarray,
              sample_weights: np.ndarray = None, test_size: float = 0.2) -> Dict:
        """
        Train quantile regression models at each percentile.

        Args:
            X: Feature DataFrame
            y: Target values
            sample_weights: Optional time-decay weights
            test_size: Holdout test size

        Returns:
            Dictionary of training metrics
        """
        print(f"\n    Training quantile regression for {self.prop_type}...")

        # Prepare data
        self.feature_names = list(X.columns)
        X_filled = smart_fillna(X)
        y_arr = np.array(y)

        # Chronological split
        split_idx = int(len(X_filled) * (1 - test_size))
        X_train, X_test = X_filled[:split_idx], X_filled[split_idx:]
        y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]

        # Fit scaler
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train a model for each quantile
        quantile_metrics = {}

        for q in self.QUANTILES:
            print(f"      Quantile {q:.0%}...")

            # Use GradientBoostingRegressor with quantile loss
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42,
            )

            model.fit(X_train_scaled, y_train)
            self.quantile_models[q] = model

            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)

            # Calculate pinball loss (proper scoring rule for quantiles)
            pinball_loss = self._pinball_loss(y_test, y_pred, q)
            quantile_metrics[f'q{int(q*100)}_pinball'] = pinball_loss

        self.is_fitted = True

        # Calculate coverage metrics
        y_pred_10 = self.quantile_models[0.10].predict(X_test_scaled)
        y_pred_90 = self.quantile_models[0.90].predict(X_test_scaled)

        # 80% prediction interval coverage
        in_interval = np.sum((y_test >= y_pred_10) & (y_test <= y_pred_90))
        coverage_80 = in_interval / len(y_test)

        self.training_metrics = {
            **quantile_metrics,
            'coverage_80': coverage_80,
            'train_size': len(X_train),
            'test_size': len(X_test),
        }

        print(f"      80% interval coverage: {coverage_80:.1%}")

        return self.training_metrics

    def _pinball_loss(self, y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
        """Calculate pinball (quantile) loss."""
        errors = y_true - y_pred
        return float(np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors)))

    def predict_distribution(self, features: Dict) -> Dict[float, float]:
        """
        Predict the full distribution of outcomes.

        Returns:
            Dictionary mapping quantile to predicted value
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = smart_fillna(X[self.feature_names])
        X_scaled = self.scaler.transform(X)

        return {q: float(model.predict(X_scaled)[0])
                for q, model in self.quantile_models.items()}

    def predict_over_probability(self, features: Dict, line: float) -> float:
        """
        Estimate probability of actual value being OVER the line.

        Uses linear interpolation between quantile predictions.

        Args:
            features: Player/game features
            line: The prop line to compare against

        Returns:
            Estimated probability (0-1) of OVER hitting
        """
        quantile_preds = self.predict_distribution(features)

        # Sort quantiles and predictions
        quantiles = sorted(quantile_preds.keys())
        predictions = [quantile_preds[q] for q in quantiles]

        # Find where the line falls in the distribution
        # If line < lowest prediction, high probability of OVER
        # If line > highest prediction, low probability of OVER
        if line <= predictions[0]:  # Below 10th percentile
            return 0.95  # Very high OVER probability

        if line >= predictions[-1]:  # Above 90th percentile
            return 0.05  # Very low OVER probability

        # Linear interpolation to find probability
        for i in range(len(predictions) - 1):
            if predictions[i] <= line <= predictions[i + 1]:
                # Interpolate probability between quantiles
                lower_q = quantiles[i]
                upper_q = quantiles[i + 1]
                lower_pred = predictions[i]
                upper_pred = predictions[i + 1]

                # Position within this interval
                if upper_pred == lower_pred:
                    pos = 0.5
                else:
                    pos = (line - lower_pred) / (upper_pred - lower_pred)

                # Interpolate the cumulative probability
                prob_below_line = lower_q + pos * (upper_q - lower_q)

                # OVER probability = 1 - P(below line)
                return 1 - prob_below_line

        return 0.50  # Default to 50% if something goes wrong

    def predict(self, features: Dict, prop_line: float = None) -> Dict:
        """
        Make a prediction with uncertainty estimates.

        Returns:
            Dictionary with median prediction and uncertainty metrics
        """
        quantile_preds = self.predict_distribution(features)

        result = {
            'predicted_value': quantile_preds[0.50],  # Median
            'prediction_10th': quantile_preds[0.10],
            'prediction_25th': quantile_preds[0.25],
            'prediction_75th': quantile_preds[0.75],
            'prediction_90th': quantile_preds[0.90],
            'prediction_std': (quantile_preds[0.75] - quantile_preds[0.25]) / 1.35,  # IQR-based std
            'prop_type': self.prop_type,
        }

        if prop_line is not None:
            result['prop_line'] = prop_line
            result['over_probability'] = self.predict_over_probability(features, prop_line)
            result['prediction'] = 'over' if result['over_probability'] > 0.5 else 'under'
            result['edge'] = result['predicted_value'] - prop_line

        return result

    def save(self, filepath: Path):
        """Save model to disk."""
        data = {
            'quantile_models': self.quantile_models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'prop_type': self.prop_type,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: Path) -> 'QuantilePropModel':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(data['prop_type'])
        model.quantile_models = data['quantile_models']
        model.scaler = data['scaler']
        model.feature_names = data['feature_names']
        model.training_metrics = data.get('training_metrics', {})
        model.is_fitted = True
        return model


# =============================================================================
# TIER 1.4: POSITION-AWARE PROP ENSEMBLE
# =============================================================================

class PositionAwarePropEnsemble:
    """
    TIER 1.4: Position-specific models for better predictions.

    Trains separate models for guards, forwards, and centers because:
    - Centers have very different rebound distributions than guards
    - Guards have different assist patterns than forwards
    - Position-specific models can capture these nuances better

    Falls back to general model if position-specific has insufficient data.
    """

    POSITION_GROUPS = {
        'guards': ['is_guard'],      # PG, SG
        'forwards': ['is_forward'],  # SF, PF
        'centers': ['is_center'],    # C
    }

    # Minimum samples required for position-specific model
    MIN_SAMPLES_PER_POSITION = 500

    def __init__(self, prop_type: str):
        self.prop_type = prop_type
        self.position_models = {}  # {position: PropEnsembleModel}
        self.general_model = None  # Fallback
        self.position_metrics = {}
        self.is_fitted = False
        self.training_metrics = {}

    def _get_position_group(self, features: Dict) -> str:
        """Determine position group from features."""
        if features.get('is_center', 0) == 1:
            return 'centers'
        elif features.get('is_forward', 0) == 1:
            return 'forwards'
        else:
            return 'guards'  # Default to guards

    def train(self, X: pd.DataFrame, y: np.ndarray, player_data: List[Dict],
              sample_weights: np.ndarray = None) -> Dict:
        """
        Train position-specific models.

        Args:
            X: Feature DataFrame
            y: Target values
            player_data: Original player data with position info in features
            sample_weights: Optional time-decay weights
        """
        print(f"\n    [TIER 1.4] Training position-specific {self.prop_type} models...")

        # Split data by position
        position_data = {
            'guards': {'X': [], 'y': [], 'weights': [], 'indices': []},
            'forwards': {'X': [], 'y': [], 'weights': [], 'indices': []},
            'centers': {'X': [], 'y': [], 'weights': [], 'indices': []},
        }

        for i, data_point in enumerate(player_data):
            features = data_point.get('features', {})
            position = self._get_position_group(features)

            position_data[position]['indices'].append(i)

        # Convert to proper format
        for pos in position_data:
            indices = position_data[pos]['indices']
            if indices:
                position_data[pos]['X'] = X.iloc[indices].copy()
                position_data[pos]['y'] = y[indices]
                if sample_weights is not None:
                    position_data[pos]['weights'] = sample_weights[indices]

        # Train position-specific models
        all_metrics = {}

        for position, data in position_data.items():
            n_samples = len(data['indices'])
            print(f"      {position.capitalize()}: {n_samples} samples")

            if n_samples >= self.MIN_SAMPLES_PER_POSITION:
                # Train position-specific model
                print(f"        Training position-specific model...")
                model = PropEnsembleModel(f"{self.prop_type}_{position}")

                weights = data['weights'] if len(data['weights']) > 0 else None
                metrics = model.train(
                    data['X'],
                    data['y'],
                    sample_weights=weights
                )

                self.position_models[position] = model
                self.position_metrics[position] = metrics
                all_metrics[position] = metrics

                print(f"        {position.capitalize()} R²: {metrics['ensemble_r2']:.4f}")
            else:
                print(f"        Insufficient samples, will use general model")

        # Train general model as fallback
        print(f"      Training general fallback model...")
        self.general_model = PropEnsembleModel(self.prop_type)
        general_metrics = self.general_model.train(X, y, sample_weights=sample_weights)
        all_metrics['general'] = general_metrics

        # Calculate weighted average metrics
        total_samples = len(y)
        weighted_r2 = 0
        weighted_rmse = 0

        for position, data in position_data.items():
            n_samples = len(data['indices'])
            weight = n_samples / total_samples

            if position in self.position_models:
                metrics = self.position_metrics[position]
            else:
                metrics = general_metrics

            weighted_r2 += weight * metrics['ensemble_r2']
            weighted_rmse += weight * metrics['ensemble_rmse']

        self.is_fitted = True
        self.training_metrics = {
            'weighted_r2': weighted_r2,
            'weighted_rmse': weighted_rmse,
            'general_r2': general_metrics['ensemble_r2'],
            'general_rmse': general_metrics['ensemble_rmse'],
            'position_metrics': self.position_metrics,
            'position_counts': {p: len(d['indices']) for p, d in position_data.items()},
            'models_trained': list(self.position_models.keys()),
            'ensemble_rmse': weighted_rmse,  # For compatibility
            'ensemble_r2': weighted_r2,      # For compatibility
            'ensemble_mae': general_metrics['ensemble_mae'],  # Approximate
            'n_models': sum(m.training_metrics.get('n_models', 3)
                          for m in self.position_models.values()) + general_metrics.get('n_models', 3),
        }

        print(f"      Position-aware R²: {weighted_r2:.4f} (vs general: {general_metrics['ensemble_r2']:.4f})")
        improvement = ((weighted_r2 - general_metrics['ensemble_r2']) / max(general_metrics['ensemble_r2'], 0.01)) * 100
        if improvement > 0:
            print(f"      Improvement: +{improvement:.1f}%")

        return self.training_metrics

    def predict(self, features: Dict, prop_line: float = None) -> Dict:
        """Make prediction using position-appropriate model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        position = self._get_position_group(features)

        # Use position-specific model if available
        if position in self.position_models:
            model = self.position_models[position]
            result = model.predict(features, prop_line)
            result['model_type'] = f'position_{position}'
        else:
            result = self.general_model.predict(features, prop_line)
            result['model_type'] = 'general_fallback'

        return result

    def save(self, filepath: Path):
        """Save position-aware ensemble to disk."""
        data = {
            'prop_type': self.prop_type,
            'position_models': {},
            'general_model': self.general_model,
            'position_metrics': self.position_metrics,
            'training_metrics': self.training_metrics,
        }

        # Save position models
        for position, model in self.position_models.items():
            data['position_models'][position] = model

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"      Saved position-aware model: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'PositionAwarePropEnsemble':
        """Load position-aware ensemble from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(data['prop_type'])
        model.position_models = data.get('position_models', {})
        model.general_model = data.get('general_model')
        model.position_metrics = data.get('position_metrics', {})
        model.training_metrics = data.get('training_metrics', {})
        model.is_fitted = True

        return model


# =============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# =============================================================================

class OptunaHyperparameterTuner:
    """
    Bayesian hyperparameter optimization using Optuna.

    Tunes XGBoost, Random Forest, Ridge, and optionally LightGBM
    for each prop type using cross-validation.
    """

    def __init__(self, n_trials: int = 50, cv_folds: int = 3, random_state: int = 42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state

    def _manual_cv_score(self, model, X, y):
        """
        Manual cross-validation to avoid sklearn 1.6+ compatibility issues with XGBoost.
        Uses TimeSeriesSplit and returns mean MSE.
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        mse_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = np.mean((y_val - y_pred) ** 2)
            mse_scores.append(mse)

        return np.mean(mse_scores)

    def _xgboost_objective(self, trial, X, y):
        """Objective function for XGBoost hyperparameter tuning."""
        params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 400),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2, log=True),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.01, 10.0, log=True),
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0,
        }

        model = XGBRegressor(**params)
        return self._manual_cv_score(model, X, y)

    def _lightgbm_objective(self, trial, X, y):
        """Objective function for LightGBM hyperparameter tuning."""
        params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 400),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 15, 63),
            'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 50),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0.01, 10.0, log=True),
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1,
        }

        model = LGBMRegressor(**params)
        return self._manual_cv_score(model, X, y)

    def _random_forest_objective(self, trial, X, y):
        """Objective function for Random Forest hyperparameter tuning."""
        params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 100, 400),
            'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', 0.5, 0.7]),
            'random_state': self.random_state,
            'n_jobs': -1,
        }

        model = RandomForestRegressor(**params)
        return self._manual_cv_score(model, X, y)

    def _ridge_objective(self, trial, X, y):
        """Objective function for Ridge hyperparameter tuning."""
        alpha = trial.suggest_float('ridge_alpha', 0.001, 100.0, log=True)

        model = Ridge(alpha=alpha, random_state=self.random_state)
        return self._manual_cv_score(model, X, y)

    def tune_all_models(self, X: np.ndarray, y: np.ndarray, prop_type: str) -> Dict[str, Dict]:
        """
        Run Optuna optimization for all models in the ensemble.

        Args:
            X: Feature array (already scaled)
            y: Target array
            prop_type: Name of prop type being optimized

        Returns:
            Dictionary of optimized parameters per model
        """
        if not HAS_OPTUNA:
            print("    Optuna not available, using default parameters")
            return {}

        print(f"\n{'='*60}")
        print(f"Tuning hyperparameters for {prop_type.upper()} model")
        print(f"{'='*60}")

        best_params = {}
        model_count = 3 if not HAS_LIGHTGBM else 4

        # XGBoost
        if HAS_XGBOOST:
            print(f"\n[1/{model_count}] Tuning XGBoost ({self.n_trials} trials)...")
            study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=self.random_state))
            study_xgb.optimize(lambda t: self._xgboost_objective(t, X, y),
                              n_trials=self.n_trials, show_progress_bar=True)
            best_params['xgboost'] = {k.replace('xgb_', ''): v for k, v in study_xgb.best_params.items()}
            print(f"   Best XGBoost MSE: {study_xgb.best_value:.4f}")

        # LightGBM
        if HAS_LIGHTGBM:
            print(f"\n[2/{model_count}] Tuning LightGBM ({self.n_trials} trials)...")
            study_lgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=self.random_state))
            study_lgb.optimize(lambda t: self._lightgbm_objective(t, X, y),
                              n_trials=self.n_trials, show_progress_bar=True)
            best_params['lightgbm'] = {k.replace('lgb_', ''): v for k, v in study_lgb.best_params.items()}
            print(f"   Best LightGBM MSE: {study_lgb.best_value:.4f}")

        # Random Forest
        rf_idx = 3 if HAS_LIGHTGBM else 2
        print(f"\n[{rf_idx}/{model_count}] Tuning Random Forest ({self.n_trials} trials)...")
        study_rf = optuna.create_study(direction='minimize', sampler=TPESampler(seed=self.random_state))
        study_rf.optimize(lambda t: self._random_forest_objective(t, X, y),
                         n_trials=self.n_trials, show_progress_bar=True)
        best_params['random_forest'] = {k.replace('rf_', ''): v for k, v in study_rf.best_params.items()}
        print(f"   Best Random Forest MSE: {study_rf.best_value:.4f}")

        # Ridge
        ridge_idx = 4 if HAS_LIGHTGBM else 3
        print(f"\n[{ridge_idx}/{model_count}] Tuning Ridge ({self.n_trials} trials)...")
        study_ridge = optuna.create_study(direction='minimize', sampler=TPESampler(seed=self.random_state))
        study_ridge.optimize(lambda t: self._ridge_objective(t, X, y),
                            n_trials=self.n_trials, show_progress_bar=True)
        best_params['ridge'] = {'alpha': study_ridge.best_params['ridge_alpha']}
        print(f"   Best Ridge MSE: {study_ridge.best_value:.4f}")

        return best_params


# =============================================================================
# PLAYER STATS COVERAGE UTILITIES
# =============================================================================

def check_player_stats_coverage(cache_dir: Path) -> Tuple[int, int, List[int]]:
    """
    Check player stats coverage and return missing game IDs.

    Args:
        cache_dir: Path to cache directory

    Returns:
        Tuple of (games_with_stats, total_games, missing_game_ids)
    """
    # Load games from all seasons - check multiple filename patterns
    all_games = []
    game_ids_seen = set()  # Avoid duplicates

    # Try different filename patterns that exist in the cache
    patterns = [
        "games_2021.json",
        "games_2022.json",
        "games_2023.json",
        "games_2023_full.json",
        "games_2024.json",
        "games_2024_full.json",
        "games_2025.json",
        "games_2025_full.json",
    ]

    for pattern in patterns:
        season_file = cache_dir / pattern
        if season_file.exists():
            try:
                with open(season_file, 'r') as f:
                    data = json.load(f)
                    # Handle both formats: dict with 'games' key or direct list
                    if isinstance(data, dict):
                        games = data.get('games', [])
                    else:
                        games = data
                    for game in games:
                        if isinstance(game, dict):
                            game_id = game.get('id')
                            if game_id and game_id not in game_ids_seen:
                                all_games.append(game)
                                game_ids_seen.add(game_id)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Warning: Could not read {pattern}: {e}")

    total_games = len(all_games)
    games_with_stats = 0
    missing_game_ids = []

    for game in all_games:
        game_id = game.get('id')
        if not game_id:
            continue

        cache_path = cache_dir / f"player_stats_{game_id}.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    stats = json.load(f)
                    if stats:  # Non-empty stats
                        games_with_stats += 1
                    else:
                        missing_game_ids.append(game_id)
            except:
                missing_game_ids.append(game_id)
        else:
            missing_game_ids.append(game_id)

    return games_with_stats, total_games, missing_game_ids


def fetch_missing_player_stats(
    api,
    cache_dir: Path,
    missing_game_ids: List[int],
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> int:
    """
    Fetch player stats for missing games with retry logic.

    Args:
        api: BallDontLie API instance
        cache_dir: Path to cache directory
        missing_game_ids: List of game IDs to fetch
        max_retries: Maximum retry attempts per game
        base_delay: Base delay for exponential backoff

    Returns:
        Number of games successfully fetched
    """
    fetched = 0
    failed = 0
    total = len(missing_game_ids)

    print(f"\nFetching player stats for {total} missing games...", flush=True)
    print("This may take a while due to rate limiting...", flush=True)
    print(f"Estimated time: {total * 0.15 / 60:.1f} minutes", flush=True)

    for i, game_id in enumerate(missing_game_ids):
        cache_path = cache_dir / f"player_stats_{game_id}.json"

        # Progress every 25 games (more frequent)
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%) - fetched: {fetched}, failed: {failed}", flush=True)

        # Retry logic with exponential backoff
        success = False
        for attempt in range(max_retries):
            try:
                stats = api.get_player_stats(game_ids=[game_id])

                # Cache the result
                with open(cache_path, 'w') as f:
                    json.dump(stats, f)

                fetched += 1
                success = True
                break  # Success, exit retry loop

            except Exception as e:
                delay = base_delay * (2 ** attempt)
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    # Write empty list to avoid re-fetching permanently failed games
                    with open(cache_path, 'w') as f:
                        json.dump([], f)
                    failed += 1

        # Rate limiting: 600 req/min = 10 req/sec, add buffer
        time.sleep(0.12)

    print(f"\n  Completed: {fetched}/{total} games fetched successfully, {failed} failed", flush=True)
    return fetched


def calculate_time_decay_weights(
    dates: List[str],
    half_life_days: int = 180,
    min_weight: float = 0.1
) -> np.ndarray:
    """
    Calculate time-decay weights for training samples.

    More recent games are weighted higher than older games.
    Uses exponential decay with configurable half-life.

    Args:
        dates: List of game dates (YYYY-MM-DD format)
        half_life_days: Days until weight drops to 50% (default 180 = 6 months)
        min_weight: Minimum weight for very old games

    Returns:
        Array of weights (same length as dates)
    """
    today = datetime.now()
    weights = []

    for date_str in dates:
        try:
            if isinstance(date_str, str):
                if 'T' in date_str:
                    date_str = date_str.split('T')[0]
                game_date = datetime.strptime(date_str, '%Y-%m-%d')
            else:
                game_date = date_str

            days_old = (today - game_date).days
            # Exponential decay: weight = exp(-days / (half_life / ln(2)))
            decay_constant = half_life_days / np.log(2)
            weight = np.exp(-days_old / decay_constant)
            weight = max(weight, min_weight)

        except (ValueError, TypeError):
            weight = min_weight

        weights.append(weight)

    return np.array(weights)


def train_all_models(
    team_data: List[Dict],
    player_data: List[Dict],
    use_time_decay: bool = True,
    time_decay_half_life: int = 180,
    use_ensemble_props: bool = True,  # Use ensemble models for props
    use_optuna: bool = False,  # NEW: Use Optuna hyperparameter optimization
    optuna_trials: int = 50,  # Number of Optuna trials per model
    tune_team_models: bool = False,  # NEW: Use Optuna for team model tuning
    team_tune_trials: int = 50,  # Number of Optuna trials for team models
) -> Dict:
    """
    Train all models with the prepared data.

    Args:
        team_data: List of team training samples
        player_data: List of player training samples
        use_time_decay: Whether to apply time-decay weighting
        time_decay_half_life: Half-life in days for weight decay
        use_optuna: Whether to use Optuna for hyperparameter optimization
        optuna_trials: Number of trials per model for Optuna
        tune_team_models: Whether to tune moneyline/spread team models
        team_tune_trials: Number of Optuna trials for team model tuning

    Returns:
        Dictionary of training metrics
    """
    results = {}

    # Calculate sample weights if using time decay
    sample_weights = None
    if use_time_decay and team_data:
        dates = [d.get('game_date', '') for d in team_data]
        sample_weights = calculate_time_decay_weights(dates, time_decay_half_life)
        print(f"\n[Time-Decay Weighting] Half-life: {time_decay_half_life} days")
        print(f"  Weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")
        print(f"  Avg weight: {sample_weights.mean():.3f}")

    # ==========================================================================
    # TEAM MODELS
    # ==========================================================================
    print("\n" + "="*60)
    print("TRAINING TEAM MODELS")
    print("="*60)

    # Prepare team data
    X_team = pd.DataFrame([d['features'] for d in team_data])
    y_ml = np.array([1 if d['home_win'] else 0 for d in team_data])
    y_spread = np.array([d['point_differential'] for d in team_data])

    # Train Ensemble Moneyline Model
    print("\n--- Enhanced Ensemble Moneyline Model ---")
    print("  Building diverse model ensemble for robust predictions...")

    # Track tuned params for later use
    xgb_ml_params = None
    xgb_sp_params = None

    # Build diverse ensemble with multiple model families
    models = {
        # Linear model (fast baseline)
        'lr': LogisticRegression(max_iter=1000, random_state=42, C=1.0),

        # Tree-based models (capture non-linear patterns)
        'rf': RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            min_samples_split=5, random_state=42
        ),

        # Neural network (learns complex feature interactions)
        'mlp': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        ),

        # SVM with RBF kernel (good for high-dimensional data)
        'svm': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Enable probability estimates
            random_state=42,
            cache_size=500
        ),
    }

    # Add XGBoost if available (often best single model)
    if HAS_XGBOOST:
        models['xgb'] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    # Add LightGBM if available (fast and accurate)
    if HAS_LIGHTGBM:
        models['lgb'] = LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    # Add CatBoost if available (handles categoricals well)
    if HAS_CATBOOST:
        models['catboost'] = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
            allow_writing_files=False
        )

    # Dynamic weight assignment based on available models
    # Weights based on typical performance and model diversity
    base_weights = {
        'lr': 0.08,       # Fast baseline, less weight
        'rf': 0.15,       # Good generalization
        'gb': 0.15,       # Strong performer
        'mlp': 0.12,      # Captures complex patterns
        'svm': 0.10,      # Different decision boundary
        'xgb': 0.18,      # Often best single model
        'lgb': 0.15,      # Fast and accurate
        'catboost': 0.12, # Robust to overfitting
    }

    # Filter to only available models and renormalize weights
    model_weights = {k: v for k, v in base_weights.items() if k in models}
    weight_sum = sum(model_weights.values())
    model_weights = {k: v / weight_sum for k, v in model_weights.items()}

    print(f"  Models in ensemble: {list(models.keys())}")
    print(f"  Weights: {', '.join([f'{k}={v:.2f}' for k, v in model_weights.items()])}")
    scaler_ml = StandardScaler()

    # Split data with stratification
    if sample_weights is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X_team, y_ml, sample_weights, test_size=0.2, random_state=42, stratify=y_ml
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_team, y_ml, test_size=0.2, random_state=42, stratify=y_ml
        )
        w_train = None

    feature_names = list(X_team.columns)

    X_train_scaled = scaler_ml.fit_transform(smart_fillna(X_train))
    X_test_scaled = scaler_ml.transform(smart_fillna(X_test))

    # Optuna hyperparameter tuning for XGBoost moneyline (if enabled)
    if tune_team_models and HAS_OPTUNA and HAS_XGBOOST:
        print("\n  [Optuna] Tuning XGBoost moneyline hyperparameters...")
        xgb_ml_params = tune_moneyline_xgb(
            X_train_scaled, y_train, sample_weights=w_train, n_trials=team_tune_trials
        )
        if xgb_ml_params:
            print(f"  Optimized params: {xgb_ml_params}")
            # Update XGBoost model with tuned params
            models['xgb'] = XGBClassifier(
                **xgb_ml_params,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )

    # Models that support sample_weight parameter
    SUPPORTS_SAMPLE_WEIGHT = {'lr', 'rf', 'gb', 'xgb', 'lgb', 'catboost'}

    individual_metrics = {}
    for name, model in models.items():
        print(f"  Training {name.upper()}...")
        try:
            # Use sample weights if available and supported
            if w_train is not None and name in SUPPORTS_SAMPLE_WEIGHT:
                model.fit(X_train_scaled, y_train, sample_weight=w_train)
            else:
                model.fit(X_train_scaled, y_train)

            # Calculate comprehensive metrics for each model
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            individual_metrics[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'log_loss': log_loss(y_test, y_proba),
                'brier_score': brier_score_loss(y_test, y_proba),
            }
            print(f"    {name.upper()} Accuracy: {individual_metrics[name]['accuracy']:.4f} | "
                  f"Brier: {individual_metrics[name]['brier_score']:.4f}")
        except Exception as e:
            print(f"    {name.upper()} training failed: {e}")
            # Remove failed model
            individual_metrics[name] = {'error': str(e)}

    # Remove failed models from ensemble
    models = {k: v for k, v in models.items() if 'error' not in individual_metrics.get(k, {})}
    model_weights = {k: v for k, v in model_weights.items() if k in models}
    weight_sum = sum(model_weights.values())
    if weight_sum > 0:
        model_weights = {k: v / weight_sum for k, v in model_weights.items()}

    # === STACKED META-LEARNER ===
    # Instead of fixed weights, train a meta-learner to combine base model predictions
    # This learns optimal non-linear combinations from the data
    print("\n  Training stacked meta-learner...")

    # Get base model predictions on training data (for meta-learner training)
    meta_features_train = np.column_stack([
        model.predict_proba(X_train_scaled)[:, 1] for model in models.values()
    ])
    meta_features_test = np.column_stack([
        model.predict_proba(X_test_scaled)[:, 1] for model in models.values()
    ])

    # Train meta-learner (logistic regression works well and is interpretable)
    meta_learner = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    meta_learner.fit(meta_features_train, y_train)

    # Get meta-learner predictions
    y_proba_stacked = meta_learner.predict_proba(meta_features_test)[:, 1]
    y_pred_stacked = (y_proba_stacked > 0.5).astype(int)

    # Also compute weighted average for comparison
    probs = np.zeros((len(X_test_scaled), 2))
    for name, model in models.items():
        probs += model_weights[name] * model.predict_proba(X_test_scaled)
    y_proba_weighted = probs[:, 1]

    # Compare stacked vs weighted
    brier_stacked = brier_score_loss(y_test, y_proba_stacked)
    brier_weighted = brier_score_loss(y_test, y_proba_weighted)
    acc_stacked = accuracy_score(y_test, y_pred_stacked)
    acc_weighted = accuracy_score(y_test, (y_proba_weighted > 0.5).astype(int))

    print(f"    Weighted Avg: Acc={acc_weighted:.4f}, Brier={brier_weighted:.4f}")
    print(f"    Stacked:      Acc={acc_stacked:.4f}, Brier={brier_stacked:.4f}")

    # Use stacked if it's better
    if brier_stacked < brier_weighted:
        print(f"    Using STACKED meta-learner (Brier improved by {(brier_weighted - brier_stacked) / brier_weighted * 100:.1f}%)")
        y_proba_ensemble = y_proba_stacked
        y_pred_ensemble = y_pred_stacked
        use_stacking = True
        # Get meta-learner coefficients (shows model importance)
        coefs = dict(zip(models.keys(), meta_learner.coef_[0]))
        print(f"    Meta-learner coefficients: {', '.join([f'{k}={v:.2f}' for k, v in coefs.items()])}")
    else:
        print(f"    Using WEIGHTED average (stacking didn't improve)")
        y_proba_ensemble = y_proba_weighted
        y_pred_ensemble = (y_proba_weighted > 0.5).astype(int)
        use_stacking = False
        meta_learner = None

    # === ISOTONIC CALIBRATION ===
    # Research shows calibration-optimized models have 70% higher ROI than accuracy-optimized
    # We use a held-out calibration set to prevent overfitting
    print("\n  Applying isotonic calibration for improved probability estimates...")

    # Get calibration probabilities on training set (using OOF predictions would be better but more complex)
    probs_train = np.zeros((len(X_train_scaled), 2))
    for name, model in models.items():
        probs_train += model_weights[name] * model.predict_proba(X_train_scaled)
    y_proba_train = probs_train[:, 1]

    # Fit isotonic calibration on training data
    isotonic_calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
    isotonic_calibrator.fit(y_proba_train, y_train)

    # Apply calibration to test predictions
    y_proba_calibrated = isotonic_calibrator.predict(y_proba_ensemble)

    # Compare Brier scores
    brier_uncalibrated = brier_score_loss(y_test, y_proba_ensemble)
    brier_calibrated = brier_score_loss(y_test, y_proba_calibrated)
    print(f"    Brier (uncalibrated): {brier_uncalibrated:.4f}")
    print(f"    Brier (calibrated):   {brier_calibrated:.4f}")
    if brier_calibrated < brier_uncalibrated:
        print(f"    Improvement: {(brier_uncalibrated - brier_calibrated) / brier_uncalibrated * 100:.1f}%")
        use_calibration = True
    else:
        print(f"    Calibration didn't improve Brier score, using uncalibrated predictions")
        use_calibration = False
        isotonic_calibrator = None

    # Use calibrated or uncalibrated based on which is better
    final_proba = y_proba_calibrated if use_calibration else y_proba_ensemble

    # Comprehensive ensemble metrics (using calibrated probabilities if better)
    ml_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_ensemble),
        'precision': precision_score(y_test, y_pred_ensemble, zero_division=0),
        'recall': recall_score(y_test, y_pred_ensemble, zero_division=0),
        'f1': f1_score(y_test, y_pred_ensemble, zero_division=0),
        'log_loss': log_loss(y_test, final_proba),  # Use calibrated if available
        'brier_score': brier_score_loss(y_test, final_proba),  # Use calibrated if available
        'brier_uncalibrated': brier_uncalibrated,
        'brier_calibrated': brier_calibrated if use_calibration else None,
        'calibration_used': use_calibration,
        'stacking_used': use_stacking,
        'individual_metrics': individual_metrics,
        'model_weights': model_weights,
        'n_models': len(models),
    }
    print(f"\n  Ensemble Results:")
    print(f"    Accuracy: {ml_metrics['accuracy']:.4f}")
    print(f"    Brier Score: {ml_metrics['brier_score']:.4f} (lower is better)")
    print(f"    Log Loss: {ml_metrics['log_loss']:.4f}")
    print(f"    Calibration: {'ENABLED' if use_calibration else 'disabled'}")

    # Calculate Betting ROI if we have Vegas implied probabilities
    if 'vegas_implied_home_prob' in X_test.columns:
        vegas_probs = X_test['vegas_implied_home_prob'].values
        valid_mask = (vegas_probs > 0) & (vegas_probs < 1)
        if valid_mask.sum() > 10:
            roi, n_bets = betting_roi_scorer(
                y_test[valid_mask],
                y_proba_ensemble[valid_mask],
                vegas_probs[valid_mask],
                min_edge=0.05
            )
            ml_metrics['betting_roi'] = roi
            ml_metrics['betting_n_bets'] = n_bets
            print(f"    Betting ROI (5% edge): {roi:+.1%} on {n_bets} hypothetical bets")
    else:
        print(f"    Betting ROI: N/A (no Vegas odds in training data)")

    results['moneyline'] = ml_metrics

    # Save ensemble model with calibrator and meta-learner
    wrapper = EnsembleMoneylineWrapper(
        models=models,
        model_weights=model_weights,
        scaler=scaler_ml,
        feature_names=feature_names,
        training_metrics=ml_metrics,
    )
    with open(MODEL_DIR / 'moneyline_ensemble.pkl', 'wb') as f:
        pickle.dump({
            'model': wrapper,
            'scaler': scaler_ml,
            'feature_names': feature_names,
            'training_metrics': ml_metrics,
            'isotonic_calibrator': isotonic_calibrator,  # Probability calibration
            'calibration_enabled': use_calibration,
            'meta_learner': meta_learner,  # Stacked meta-learner
            'stacking_enabled': use_stacking,
            'base_model_order': list(models.keys()),  # Order for meta-learner input
        }, f)
    print("  Saved: models/moneyline_ensemble.pkl")
    extras = []
    if use_stacking:
        extras.append("stacked meta-learner")
    if use_calibration:
        extras.append("isotonic calibration")
    if extras:
        print(f"    (with {' + '.join(extras)})")

    # Train Enhanced Spread Ensemble Model
    print("\n--- Enhanced Spread Ensemble Model ---")
    print("  Building diverse regression ensemble for spread predictions...")

    # Build spread regression ensemble
    spread_models = {
        # Linear models (capture baseline relationships)
        'ridge': Ridge(alpha=1.0, random_state=42),
        'lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000),
        'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),

        # Tree-based models (capture non-linear patterns)
        'rf': RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            min_samples_split=5, random_state=42
        ),
    }

    # Add XGBoost regressor if available
    if HAS_XGBOOST:
        spread_models['xgb'] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )

    # Add LightGBM regressor if available
    if HAS_LIGHTGBM:
        spread_models['lgb'] = LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    # Dynamic weights for spread models
    spread_base_weights = {
        'ridge': 0.08,
        'lasso': 0.05,
        'elasticnet': 0.07,
        'rf': 0.18,
        'gb': 0.18,
        'xgb': 0.22,
        'lgb': 0.22,
    }

    spread_weights = {k: v for k, v in spread_base_weights.items() if k in spread_models}
    sw_sum = sum(spread_weights.values())
    spread_weights = {k: v / sw_sum for k, v in spread_weights.items()}

    print(f"  Models in ensemble: {list(spread_models.keys())}")
    print(f"  Weights: {', '.join([f'{k}={v:.2f}' for k, v in spread_weights.items()])}")

    scaler_sp = StandardScaler()

    # Split data with sample weights if available
    if sample_weights is not None:
        X_train_sp, X_test_sp, y_train_sp, y_test_sp, w_train_sp, w_test_sp = train_test_split(
            X_team, y_spread, sample_weights, test_size=0.2, random_state=42
        )
    else:
        X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(
            X_team, y_spread, test_size=0.2, random_state=42
        )
        w_train_sp = None

    X_train_sp_scaled = scaler_sp.fit_transform(smart_fillna(X_train_sp))
    X_test_sp_scaled = scaler_sp.transform(smart_fillna(X_test_sp))

    # Optuna hyperparameter tuning for XGBoost spread (if enabled)
    if tune_team_models and HAS_OPTUNA and HAS_XGBOOST:
        print("\n  [Optuna] Tuning XGBoost spread hyperparameters...")
        xgb_sp_params = tune_spread_xgb(
            X_train_sp_scaled, y_train_sp, sample_weights=w_train_sp, n_trials=team_tune_trials
        )
        if xgb_sp_params:
            print(f"  Optimized params: {xgb_sp_params}")
            # Update XGBoost model with tuned params
            spread_models['xgb'] = XGBRegressor(
                **xgb_sp_params,
                random_state=42,
                n_jobs=-1
            )

    # Models that support sample_weight
    SPREAD_SUPPORTS_WEIGHT = {'rf', 'gb', 'xgb', 'lgb'}

    spread_individual_metrics = {}
    for name, model in spread_models.items():
        print(f"  Training {name.upper()}...")
        try:
            if w_train_sp is not None and name in SPREAD_SUPPORTS_WEIGHT:
                model.fit(X_train_sp_scaled, y_train_sp, sample_weight=w_train_sp)
            else:
                model.fit(X_train_sp_scaled, y_train_sp)

            y_pred = model.predict(X_test_sp_scaled)
            spread_individual_metrics[name] = {
                'rmse': np.sqrt(mean_squared_error(y_test_sp, y_pred)),
                'mae': mean_absolute_error(y_test_sp, y_pred),
                'r2': r2_score(y_test_sp, y_pred),
            }
            print(f"    {name.upper()} RMSE: {spread_individual_metrics[name]['rmse']:.2f} | "
                  f"MAE: {spread_individual_metrics[name]['mae']:.2f}")
        except Exception as e:
            print(f"    {name.upper()} training failed: {e}")
            spread_individual_metrics[name] = {'error': str(e)}

    # Remove failed models
    spread_models = {k: v for k, v in spread_models.items() if 'error' not in spread_individual_metrics.get(k, {})}
    spread_weights = {k: v for k, v in spread_weights.items() if k in spread_models}
    sw_sum = sum(spread_weights.values())
    if sw_sum > 0:
        spread_weights = {k: v / sw_sum for k, v in spread_weights.items()}

    # Ensemble prediction using weighted averaging
    y_pred_sp = np.zeros(len(X_test_sp_scaled))
    for name, model in spread_models.items():
        y_pred_sp += spread_weights[name] * model.predict(X_test_sp_scaled)

    sp_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test_sp, y_pred_sp)),
        'mae': mean_absolute_error(y_test_sp, y_pred_sp),
        'r2': r2_score(y_test_sp, y_pred_sp),
        'individual_metrics': spread_individual_metrics,
        'model_weights': spread_weights,
        'n_models': len(spread_models),
    }
    print(f"\n  Ensemble Results:")
    print(f"    RMSE: {sp_metrics['rmse']:.2f} points")
    print(f"    MAE: {sp_metrics['mae']:.2f} points")
    print(f"    R²: {sp_metrics['r2']:.4f}")
    results['spread'] = sp_metrics

    # Create wrapper for spread ensemble (using global class for pickling)
    spread_wrapper = SpreadEnsembleWrapper(
        models=spread_models,
        weights=spread_weights,
        scaler=scaler_sp,
        feature_names=feature_names,
        metrics=sp_metrics
    )

    with open(MODEL_DIR / 'spread_ensemble.pkl', 'wb') as f:
        pickle.dump({
            'model': spread_wrapper,
            'models': spread_models,
            'weights': spread_weights,
            'scaler': scaler_sp,
            'feature_names': feature_names,
            'training_metrics': sp_metrics,
        }, f)
    print("  Saved: models/spread_ensemble.pkl")

    # Also save backwards-compatible single model file
    with open(MODEL_DIR / 'spread_svm_regressor.pkl', 'wb') as f:
        pickle.dump({
            'model': spread_wrapper,
            'scaler': scaler_sp,
            'feature_names': feature_names,
            'training_metrics': sp_metrics,
        }, f)
    print("  Saved: models/spread_svm_regressor.pkl (backwards-compatible)")

    # === QUANTILE REGRESSION FOR SPREAD UNCERTAINTY ===
    # Predicts prediction intervals - only bet when interval is narrow (high confidence)
    print("\n--- Spread Quantile Model (Uncertainty Estimation) ---")
    try:
        # GradientBoostingRegressor is already imported at module level
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        quantile_models = {}

        for q in quantiles:
            qr_model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42
            )
            qr_model.fit(X_train_sp_scaled, y_train_sp)
            quantile_models[f'q{int(q*100)}'] = qr_model
            print(f"  Trained quantile {q:.0%} model")

        # Evaluate coverage on test set
        q10_pred = quantile_models['q10'].predict(X_test_sp_scaled)
        q90_pred = quantile_models['q90'].predict(X_test_sp_scaled)
        coverage_80 = np.mean((y_test_sp >= q10_pred) & (y_test_sp <= q90_pred))
        interval_width = np.mean(q90_pred - q10_pred)

        print(f"  80% Prediction Interval Coverage: {coverage_80:.1%} (target: 80%)")
        print(f"  Average Interval Width: {interval_width:.1f} points")

        # Save quantile models
        with open(MODEL_DIR / 'spread_quantile.pkl', 'wb') as f:
            pickle.dump({
                'quantile_models': quantile_models,
                'scaler': scaler_sp,
                'feature_names': feature_names,
                'coverage_80': coverage_80,
                'avg_interval_width': interval_width,
            }, f)
        print("  Saved: models/spread_quantile.pkl")

        results['spread_quantile'] = {
            'coverage_80': coverage_80,
            'avg_interval_width': interval_width,
            'quantiles': quantiles,
        }
    except Exception as e:
        print(f"  Quantile regression failed: {e}")

    # ==========================================================================
    # PLAYER PROP MODELS
    # ==========================================================================
    print("\n" + "="*60)
    print("TRAINING PLAYER PROP MODELS")
    print("="*60)

    if not player_data:
        print("  No player data available for prop models")
        return results

    # Prepare player data
    X_player = pd.DataFrame([d['features'] for d in player_data])
    player_feature_names = list(X_player.columns)

    # Calculate player-specific sample weights
    # Combines: 1) Time decay (recent games weighted more)
    #           2) Outlier/blowout weights (OT, blowout games weighted less)
    player_sample_weights = None

    # Start with outlier weights from training data (OT normalization, blowout detection)
    outlier_weights = np.array([d.get('sample_weight', 1.0) for d in player_data])

    if use_time_decay:
        player_dates = [d.get('game_date', '') for d in player_data]
        time_weights = calculate_time_decay_weights(player_dates, time_decay_half_life)
        # Combine time decay with outlier weights
        player_sample_weights = outlier_weights * time_weights
        print(f"\n[Player Sample Weighting]")
        print(f"  Time-decay range: {time_weights.min():.3f} - {time_weights.max():.3f}")
        print(f"  Outlier weight range: {outlier_weights.min():.3f} - {outlier_weights.max():.3f}")
        print(f"  Combined weight range: {player_sample_weights.min():.3f} - {player_sample_weights.max():.3f}")
        print(f"  Avg combined weight: {player_sample_weights.mean():.3f}")
        # Count how many samples were down-weighted due to outliers
        downweighted = np.sum(outlier_weights < 1.0)
        print(f"  Samples down-weighted (OT/blowout): {downweighted} ({downweighted/len(outlier_weights)*100:.1f}%)")
    else:
        player_sample_weights = outlier_weights
        print(f"\n[Player Outlier Weighting Only]")
        print(f"  Weight range: {player_sample_weights.min():.3f} - {player_sample_weights.max():.3f}")

    # ==========================================================================
    # TIER 2.3: MINUTES PREDICTION MODEL
    # ==========================================================================
    print("\n--- MINUTES Prediction Model (TIER 2.3) ---")

    # Get actual minutes - use 10 as default for samples missing actual_min
    y_minutes = np.array([d.get('actual_min', 10.0) for d in player_data])

    minutes_model = MinutesPredictionModel()
    min_metrics = minutes_model.train(X_player, y_minutes, sample_weights=player_sample_weights)

    # Save minutes model
    minutes_model.save(MODEL_DIR / 'player_minutes_model.pkl')
    print(f"  Saved: models/player_minutes_model.pkl")

    results['minutes_model'] = {
        'classifier_accuracy': min_metrics['classifier']['accuracy'],
        'classifier_f1': min_metrics['classifier']['f1'],
        'regressor_rmse': min_metrics['regressor']['rmse'],
        'regressor_r2': min_metrics['regressor']['r2'],
        'play_rate': min_metrics['play_rate'],
    }

    prop_types = [
        ('points', 'actual_pts'),
        ('rebounds', 'actual_reb'),
        ('assists', 'actual_ast'),
        ('threes', 'actual_fg3m'),
        ('pra', 'actual_pra'),
    ]

    # TIER 1.4: Props that benefit from position-specific models
    # Rebounds and assists vary significantly by position
    POSITION_AWARE_PROPS = ['rebounds', 'assists']

    for prop_name, target_col in prop_types:
        print(f"\n--- {prop_name.upper()} Prop Model ---")

        y = np.array([d[target_col] for d in player_data])

        if use_ensemble_props:
            # TIER 1.4: Use position-aware models for rebounds and assists
            use_position_aware = prop_name in POSITION_AWARE_PROPS

            if use_position_aware:
                print(f"  Using PositionAwarePropEnsemble (TIER 1.4)")

                prop_model = PositionAwarePropEnsemble(prop_name)
                metrics = prop_model.train(
                    X_player, y, player_data,
                    sample_weights=player_sample_weights
                )

                print(f"  Position-Aware RMSE: {metrics['ensemble_rmse']:.2f}")
                print(f"  Position-Aware R²: {metrics['ensemble_r2']:.4f}")

                # Save as position-aware model
                prop_model.save(MODEL_DIR / f'player_{prop_name}_position_aware.pkl')
                print(f"  Saved: models/player_{prop_name}_position_aware.pkl")

                # Also save regular ensemble for backward compatibility
                print(f"  Also training general ensemble for backward compatibility...")
                general_model = PropEnsembleModel(prop_name)
                general_metrics = general_model.train(X_player, y, sample_weights=player_sample_weights)
                general_model.save(MODEL_DIR / f'player_{prop_name}_ensemble.pkl')

                results[f'prop_{prop_name}'] = {
                    'rmse': metrics['ensemble_rmse'],
                    'mae': metrics['ensemble_mae'],
                    'r2': metrics['ensemble_r2'],
                    'model_type': 'position_aware',
                    'n_models': metrics.get('n_models', 4),
                    'position_metrics': metrics.get('position_metrics', {}),
                    'general_r2': metrics.get('general_r2', general_metrics['ensemble_r2']),
                    'position_improvement': metrics['ensemble_r2'] - metrics.get('general_r2', general_metrics['ensemble_r2']),
                }
            else:
                # Use standard ensemble model for other props
                print(f"  Using PropEnsembleModel (stacked ensemble)")

                # Optuna hyperparameter optimization if enabled
                optimized_params = None
                if use_optuna and HAS_OPTUNA:
                    print(f"  Running Optuna hyperparameter optimization ({optuna_trials} trials)...")
                    # Scale features for Optuna tuning
                    temp_scaler = StandardScaler()
                    X_scaled = temp_scaler.fit_transform(smart_fillna(X_player).values)

                    tuner = OptunaHyperparameterTuner(n_trials=optuna_trials, cv_folds=3)
                    optimized_params = tuner.tune_all_models(X_scaled, y, prop_name)

                    # Save optimized params for reproducibility
                    params_path = MODEL_DIR / f'{prop_name}_optuna_params.json'
                    with open(params_path, 'w') as f:
                        json.dump(optimized_params, f, indent=2)
                    print(f"  Saved optimized params: {params_path}")

                prop_model = PropEnsembleModel(prop_name, optimized_params=optimized_params)
                player_dates = [d.get('game_date', '') for d in player_data]
                metrics = prop_model.train(X_player, y, dates=player_dates, sample_weights=player_sample_weights)

                print(f"  Ensemble RMSE: {metrics['ensemble_rmse']:.2f}")
                print(f"  Ensemble MAE: {metrics['ensemble_mae']:.2f}")
                print(f"  Ensemble R²: {metrics['ensemble_r2']:.4f}")

                # Save as ensemble model
                prop_model.save(MODEL_DIR / f'player_{prop_name}_ensemble.pkl')
                print(f"  Saved: models/player_{prop_name}_ensemble.pkl")

                # Also save metrics for comparison
                results[f'prop_{prop_name}'] = {
                    'rmse': metrics['ensemble_rmse'],
                    'mae': metrics['ensemble_mae'],
                    'r2': metrics['ensemble_r2'],
                    'model_type': 'ensemble',
                    'n_models': metrics['n_models'],
                    'model_metrics': metrics.get('model_metrics', {}),
                    'optuna_optimized': optimized_params is not None,
                }
        else:
            # Use original single model
            prop_model = PropModel(prop_name)
            metrics = prop_model.train(X_player, y, sample_weights=player_sample_weights)

            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  R²: {metrics['r2']:.4f}")

            prop_model.save(MODEL_DIR / f'player_{prop_name}.pkl')
            print(f"  Saved: models/player_{prop_name}.pkl")

            results[f'prop_{prop_name}'] = metrics

    # ==========================================================================
    # QUANTILE MODELS FOR UNCERTAINTY ESTIMATION
    # These provide honest confidence through prediction intervals
    # ==========================================================================
    print("\n" + "="*60)
    print("TRAINING QUANTILE MODELS FOR UNCERTAINTY ESTIMATION")
    print("="*60)

    for prop_name, target_col in prop_types:
        print(f"\n--- {prop_name.upper()} Quantile Model ---")

        y = np.array([d[target_col] for d in player_data])

        try:
            quantile_model = QuantilePropModel(prop_name)
            q_metrics = quantile_model.train(X_player, y, sample_weights=player_sample_weights)

            # Save quantile model
            quantile_path = MODEL_DIR / f'player_{prop_name}_quantile.pkl'
            with open(quantile_path, 'wb') as f:
                pickle.dump({
                    'model': quantile_model,
                    'training_metrics': q_metrics,
                    'feature_names': player_feature_names,
                }, f)
            print(f"  Saved: models/player_{prop_name}_quantile.pkl")
            print(f"  Coverage (80%): {q_metrics['coverage_80']:.1%}")

            results[f'quantile_{prop_name}'] = {
                'coverage_80': q_metrics['coverage_80'],
                'pinball_losses': {k: v for k, v in q_metrics.items() if 'pinball' in k},
            }
        except Exception as e:
            print(f"  Error training quantile model for {prop_name}: {e}")
            results[f'quantile_{prop_name}'] = {'error': str(e)}

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("COMPLETE NBA MODEL TRAINING WITH BALLDONTLIE.IO")
    print("="*60)
    print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Current Season: 2025-26")
    print()

    collector = ComprehensiveDataCollector()

    # Fetch teams
    print("Fetching team data...")
    teams = collector.get_all_teams()
    print(f"  Found {len(teams)} teams")

    # Fetch games from multiple seasons (including current 2025-26)
    seasons = [2023, 2024, 2025]  # 2023-24, 2024-25, 2025-26 (current)
    all_games = []

    print("\nFetching game data...")
    for season in seasons:
        print(f"\nSeason {season}-{str(season+1)[-2:]}:")
        games = collector.fetch_season_games(season)
        all_games.extend(games)

    print(f"\nTotal games collected: {len(all_games)}")

    # Fetch player stats for all games
    print("\nFetching player statistics...")
    game_ids = [g.get('id') for g in all_games if g.get('id')]
    player_stats = collector.fetch_player_stats_for_games(game_ids)
    print(f"  Player stats for {len(player_stats)} games")

    # Process into training data
    team_data, player_data = process_games_for_training(all_games, player_stats)

    if len(team_data) < 50:
        print("\nError: Insufficient team training data")
        sys.exit(1)

    # Data statistics
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    home_wins = sum(1 for d in team_data if d['home_win'])
    print(f"Team games: {len(team_data)}")
    print(f"Home win rate: {home_wins / len(team_data):.1%}")
    print(f"Avg point differential: {np.mean([d['point_differential'] for d in team_data]):+.1f}")
    print(f"Player game stats: {len(player_data)}")
    if player_data:
        print(f"Avg points: {np.mean([d['actual_pts'] for d in player_data]):.1f}")
        print(f"Avg rebounds: {np.mean([d['actual_reb'] for d in player_data]):.1f}")
        print(f"Avg assists: {np.mean([d['actual_ast'] for d in player_data]):.1f}")

    # Train all models (with ensemble prop models by default)
    # Check for Optuna args from command line
    use_optuna = getattr(args, 'use_optuna', False) if args else False
    optuna_trials = getattr(args, 'optuna_trials', 50) if args else 50
    tune_team_models = getattr(args, 'tune_team_models', False) if args else False
    team_tune_trials = getattr(args, 'team_tune_trials', 50) if args else 50

    results = train_all_models(
        team_data,
        player_data,
        use_ensemble_props=True,
        use_optuna=use_optuna,
        optuna_trials=optuna_trials,
        tune_team_models=tune_team_models,
        team_tune_trials=team_tune_trials,
    )

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nModel Performance Summary:")

    if 'moneyline' in results:
        print(f"\n  Moneyline Ensemble: {results['moneyline']['accuracy']:.2%} accuracy")
    if 'spread' in results:
        print(f"  Spread: {results['spread']['rmse']:.2f} RMSE")

    for key, metrics in results.items():
        if key.startswith('prop_'):
            prop_name = key.replace('prop_', '').upper()
            optimized = " (Optuna)" if metrics.get('optuna_optimized') else ""
            print(f"  {prop_name} Props: {metrics['rmse']:.2f} RMSE, {metrics['r2']:.3f} R²{optimized}")

    print("\n" + "="*60)
    print("All models saved to 'models/' directory")
    print("Run 'python3 app.py' to use for predictions")
    print("="*60)


# Global args variable for use in main()
args = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NBA betting models")
    parser.add_argument('--fetch-missing', action='store_true',
                        help='Fetch missing player stats only (no training)')
    parser.add_argument('--use-optuna', action='store_true',
                        help='Use Optuna hyperparameter optimization for prop models')
    parser.add_argument('--optuna-trials', type=int, default=50,
                        help='Number of Optuna trials per model (default: 50)')
    parser.add_argument('--check-coverage', action='store_true',
                        help='Check player stats coverage and exit')
    parser.add_argument('--tune-team-models', action='store_true',
                        help='Enable Optuna hyperparameter tuning for moneyline/spread models')
    parser.add_argument('--team-tune-trials', type=int, default=50,
                        help='Number of Optuna trials for team model tuning (default: 50)')

    args = parser.parse_args()

    # Handle fetch-missing mode
    if args.fetch_missing:
        print("="*60)
        print("FETCHING MISSING PLAYER STATS")
        print("="*60)
        api = BalldontlieAPI()
        covered, total, missing = check_player_stats_coverage(CACHE_DIR)
        print(f"\nCurrent coverage: {covered}/{total} ({covered/total*100:.1f}%)")

        if missing:
            print(f"Missing games: {len(missing)}")
            fetched = fetch_missing_player_stats(api, CACHE_DIR, missing)
            new_covered, _, _ = check_player_stats_coverage(CACHE_DIR)
            print(f"\nNew coverage: {new_covered}/{total} ({new_covered/total*100:.1f}%)")
        else:
            print("All player stats already fetched!")
        sys.exit(0)

    # Handle check-coverage mode
    if args.check_coverage:
        covered, total, missing = check_player_stats_coverage(CACHE_DIR)
        print(f"Player stats coverage: {covered}/{total} ({covered/total*100:.1f}%)")
        print(f"Missing games: {len(missing)}")
        sys.exit(0)

    # Run full training
    main()
