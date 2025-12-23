"""
Comprehensive Backtesting for 2025-26 NBA Season

This script runs point-in-time predictions for all completed games in the
2025-26 season and compares predictions to actual results.

Usage:
    python3 comprehensive_backtest.py
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Directories
MODEL_DIR = Path("models")
CACHE_DIR = Path("data/balldontlie_cache")


class PositionDefenseCalculator:
    """
    Calculate team defensive efficiency by opponent position.
    Tracks how many points/rebounds/assists/threes each team allows
    to guards, forwards, and centers separately.

    This is a POINT-IN-TIME calculator - for backtesting, we only use
    data from games that occurred BEFORE the prediction date.
    """

    POSITION_GROUPS = {
        'G': ['G', 'PG', 'SG', 'G-F', 'PG-SG'],
        'F': ['F', 'SF', 'PF', 'F-G', 'F-C', 'SF-PF', 'PF-SF'],
        'C': ['C', 'C-F', 'C-PF', 'PF-C']
    }

    LEAGUE_AVG = {
        'G': {'pts': 14.5, 'reb': 3.2, 'ast': 4.8, 'fg3m': 1.9},
        'F': {'pts': 12.8, 'reb': 5.4, 'ast': 2.1, 'fg3m': 1.2},
        'C': {'pts': 11.2, 'reb': 8.1, 'ast': 1.8, 'fg3m': 0.4},
    }

    def __init__(self):
        # Structure: team_id -> date -> position_group -> stat -> list of values
        self.team_position_defense = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    def _parse_minutes(self, min_val) -> float:
        """Parse minutes from various formats."""
        if isinstance(min_val, (int, float)):
            return float(min_val)
        if not min_val:
            return 0.0
        try:
            min_str = str(min_val)
            if ':' in min_str:
                parts = min_str.split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_str)
        except:
            return 0.0

    def _get_position_group(self, position: str) -> str:
        """Map detailed position to G/F/C group."""
        if not position:
            return 'F'  # Default to forward
        position = position.upper().strip()
        for group, positions in self.POSITION_GROUPS.items():
            if position in positions:
                return group
        # Handle edge cases
        if 'G' in position:
            return 'G'
        if 'C' in position:
            return 'C'
        return 'F'  # Default

    def process_game(self, game_id: int, game_date: str, home_team_id: int,
                     away_team_id: int, player_stats: List[Dict]):
        """
        Process a game's box scores to update position defense stats.

        For each player who scored, we track what the OPPONENT allowed to that position.
        E.g., if a Guard on the home team scored 25 points, we add 25 to the away team's
        'points allowed to guards' list.
        """
        for ps in player_stats:
            # Get player's team to determine opponent
            player_team_id = ps.get('team_id') or ps.get('team', {}).get('id')
            if not player_team_id:
                continue

            # Determine opponent
            if player_team_id == home_team_id:
                opponent_id = away_team_id
            elif player_team_id == away_team_id:
                opponent_id = home_team_id
            else:
                continue

            # Get player position
            player = ps.get('player', {})
            position = player.get('position', '') or ps.get('position', '')
            pos_group = self._get_position_group(position)

            # Get stats
            mins = self._parse_minutes(ps.get('min', 0))
            if mins < 10:  # Only count players with significant minutes
                continue

            pts = ps.get('pts', 0) or 0
            reb = ps.get('reb', 0) or 0
            ast = ps.get('ast', 0) or 0
            fg3m = ps.get('fg3m', 0) or 0

            # Record what opponent allowed to this position
            self.team_position_defense[opponent_id][game_date][pos_group]['pts'].append(pts)
            self.team_position_defense[opponent_id][game_date][pos_group]['reb'].append(reb)
            self.team_position_defense[opponent_id][game_date][pos_group]['ast'].append(ast)
            self.team_position_defense[opponent_id][game_date][pos_group]['fg3m'].append(fg3m)

    def get_position_defense_before_date(self, team_id: int, game_date: str,
                                          player_position: str, min_games: int = 5) -> Dict[str, float]:
        """
        Get team's defensive stats vs position BEFORE game_date.
        Returns features for the model.
        """
        pos_group = self._get_position_group(player_position)

        # Aggregate all games before this date
        pts_allowed = {'G': [], 'F': [], 'C': []}
        reb_allowed = {'G': [], 'F': [], 'C': []}
        ast_allowed = {'G': [], 'F': [], 'C': []}
        fg3m_allowed = {'G': [], 'F': [], 'C': []}

        for date_str, pos_data in sorted(self.team_position_defense.get(team_id, {}).items()):
            if date_str >= game_date:
                break

            for pos in ['G', 'F', 'C']:
                if pos in pos_data:
                    pts_allowed[pos].extend(pos_data[pos].get('pts', []))
                    reb_allowed[pos].extend(pos_data[pos].get('reb', []))
                    ast_allowed[pos].extend(pos_data[pos].get('ast', []))
                    fg3m_allowed[pos].extend(pos_data[pos].get('fg3m', []))

        # Check if we have enough data for player's position
        if len(pts_allowed[pos_group]) < min_games:
            return self._get_default_features(pos_group)

        # Calculate averages for all positions
        features = {}
        for pos in ['G', 'F', 'C']:
            pos_name = {'G': 'guards', 'F': 'forwards', 'C': 'centers'}[pos]
            if pts_allowed[pos]:
                features[f'opp_pts_allowed_to_{pos_name}'] = np.mean(pts_allowed[pos])
                features[f'opp_reb_allowed_to_{pos_name}'] = np.mean(reb_allowed[pos]) if reb_allowed[pos] else self.LEAGUE_AVG[pos]['reb']
                features[f'opp_ast_allowed_to_{pos_name}'] = np.mean(ast_allowed[pos]) if ast_allowed[pos] else self.LEAGUE_AVG[pos]['ast']
                features[f'opp_fg3m_allowed_to_{pos_name}'] = np.mean(fg3m_allowed[pos]) if fg3m_allowed[pos] else self.LEAGUE_AVG[pos]['fg3m']
            else:
                features[f'opp_pts_allowed_to_{pos_name}'] = self.LEAGUE_AVG[pos]['pts']
                features[f'opp_reb_allowed_to_{pos_name}'] = self.LEAGUE_AVG[pos]['reb']
                features[f'opp_ast_allowed_to_{pos_name}'] = self.LEAGUE_AVG[pos]['ast']
                features[f'opp_fg3m_allowed_to_{pos_name}'] = self.LEAGUE_AVG[pos]['fg3m']

        # Calculate position-specific differences from league average
        league_avg = self.LEAGUE_AVG[pos_group]
        player_pos_name = {'G': 'guards', 'F': 'forwards', 'C': 'centers'}[pos_group]

        opp_pts = features.get(f'opp_pts_allowed_to_{player_pos_name}', league_avg['pts'])
        opp_reb = features.get(f'opp_reb_allowed_to_{player_pos_name}', league_avg['reb'])
        opp_ast = features.get(f'opp_ast_allowed_to_{player_pos_name}', league_avg['ast'])
        opp_fg3m = features.get(f'opp_fg3m_allowed_to_{player_pos_name}', league_avg['fg3m'])

        features['opp_pts_vs_pos_diff'] = (opp_pts - league_avg['pts']) / max(league_avg['pts'], 1)
        features['opp_reb_vs_pos_diff'] = (opp_reb - league_avg['reb']) / max(league_avg['reb'], 1)
        features['opp_ast_vs_pos_diff'] = (opp_ast - league_avg['ast']) / max(league_avg['ast'], 1)
        features['opp_fg3m_vs_pos_diff'] = (opp_fg3m - league_avg['fg3m']) / max(league_avg['fg3m'], 1)

        # Add variance for points (consistency measure)
        if pts_allowed[pos_group]:
            features['opp_pts_vs_pos_std'] = np.std(pts_allowed[pos_group])
        else:
            features['opp_pts_vs_pos_std'] = 5.0

        return features

    def _get_default_features(self, pos_group: str) -> Dict[str, float]:
        """Return default features when insufficient data."""
        features = {}
        for pos in ['G', 'F', 'C']:
            pos_name = {'G': 'guards', 'F': 'forwards', 'C': 'centers'}[pos]
            features[f'opp_pts_allowed_to_{pos_name}'] = self.LEAGUE_AVG[pos]['pts']
            features[f'opp_reb_allowed_to_{pos_name}'] = self.LEAGUE_AVG[pos]['reb']
            features[f'opp_ast_allowed_to_{pos_name}'] = self.LEAGUE_AVG[pos]['ast']
            features[f'opp_fg3m_allowed_to_{pos_name}'] = self.LEAGUE_AVG[pos]['fg3m']

        features['opp_pts_vs_pos_diff'] = 0.0
        features['opp_reb_vs_pos_diff'] = 0.0
        features['opp_ast_vs_pos_diff'] = 0.0
        features['opp_fg3m_vs_pos_diff'] = 0.0
        features['opp_pts_vs_pos_std'] = 5.0

        return features


@dataclass
class PropPrediction:
    """Single prop prediction result."""
    player_id: int
    player_name: str
    team: str
    prop_type: str
    predicted: float
    actual: float
    game_id: int
    game_date: str
    is_home: bool
    days_rest: int = 2

    @property
    def error(self) -> float:
        return self.predicted - self.actual

    @property
    def abs_error(self) -> float:
        return abs(self.error)

    @property
    def squared_error(self) -> float:
        return self.error ** 2


@dataclass
class BacktestResults:
    """Container for all backtest results."""
    predictions: List[PropPrediction] = field(default_factory=list)
    games_processed: int = 0
    games_with_errors: int = 0
    start_date: str = ""
    end_date: str = ""

    def add(self, pred: PropPrediction):
        self.predictions.append(pred)

    def get_by_prop_type(self, prop_type: str) -> List[PropPrediction]:
        return [p for p in self.predictions if p.prop_type == prop_type]

    def get_by_home_away(self, is_home: bool) -> List[PropPrediction]:
        return [p for p in self.predictions if p.is_home == is_home]

    def get_by_rest_days(self, min_days: int, max_days: int) -> List[PropPrediction]:
        return [p for p in self.predictions if min_days <= p.days_rest <= max_days]

    def calculate_metrics(self, preds: List[PropPrediction] = None, exclude_dnp: bool = True) -> Dict:
        """Calculate metrics for a set of predictions."""
        if preds is None:
            preds = self.predictions
        if not preds:
            return {}

        # Filter out DNP (Did Not Play) predictions where actual=0
        # These are usually injuries/rest that we can't predict
        if exclude_dnp:
            preds = [p for p in preds if p.actual > 0]

        if not preds:
            return {}

        actuals = [p.actual for p in preds]
        predicted = [p.predicted for p in preds]

        rmse = np.sqrt(mean_squared_error(actuals, predicted))
        mae = mean_absolute_error(actuals, predicted)
        r2 = r2_score(actuals, predicted) if len(preds) > 1 else 0

        # Bias (mean error)
        bias = np.mean([p.error for p in preds])

        return {
            'count': len(preds),
            'rmse': round(rmse, 3),
            'mae': round(mae, 3),
            'r2': round(r2, 3),
            'bias': round(bias, 3),
        }


class SeasonBacktester:
    """Run point-in-time backtesting on the 2025-26 season."""

    PROP_TYPES = ['points', 'rebounds', 'assists', 'threes', 'pra']
    PROP_STAT_MAP = {
        'points': 'pts',
        'rebounds': 'reb',
        'assists': 'ast',
        'threes': 'fg3m',
        'pra': 'pra',  # Computed: pts + reb + ast
    }

    def __init__(self, season: int = 2025):
        self.season = season
        self.models = {}
        self.player_calc = None
        self.team_calc = None

        # TIER 2.3: Minutes prediction model
        self.minutes_model = None

        # TIER 2.2: Position-specific defense calculator
        self.position_defense_calc = PositionDefenseCalculator()

        # Data storage
        self.games = []
        self.player_stats = defaultdict(list)  # player_id -> [(date, stats), ...]
        self.player_info = {}  # player_id -> name/team info
        self.game_box_scores = {}  # game_id -> {player_id: box_score}

    def load_models(self):
        """Load trained prop models from disk."""
        print("Loading models...")
        for prop_type in self.PROP_TYPES:
            # Try ensemble model first (Optuna-optimized), fallback to legacy model
            model_path = MODEL_DIR / f"player_{prop_type}_ensemble.pkl"
            if not model_path.exists():
                model_path = MODEL_DIR / f"player_{prop_type}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                self.models[prop_type] = data
                model_type = "ensemble" if "ensemble" in str(model_path) else "legacy"
                print(f"  Loaded {prop_type} model ({model_type})")
            else:
                print(f"  WARNING: {prop_type} model not found at {model_path}")

        # TIER 2.3: Load minutes prediction model
        minutes_path = MODEL_DIR / "player_minutes_model.pkl"
        if minutes_path.exists():
            with open(minutes_path, 'rb') as f:
                self.minutes_model = pickle.load(f)
            print(f"  Loaded minutes model (TIER 2.3)")
        else:
            print(f"  WARNING: Minutes model not found - predictions won't be minutes-adjusted")

    def load_games(self) -> List[Dict]:
        """Load 2025-26 season games from cache."""
        print(f"\nLoading games for {self.season}-{self.season + 1} season...")
        games_file = CACHE_DIR / f"games_{self.season}_full.json"

        if not games_file.exists():
            print(f"ERROR: Games file not found: {games_file}")
            return []

        with open(games_file) as f:
            data = json.load(f)

        # Filter to completed games only
        all_games = data.get('games', [])
        completed = [g for g in all_games if g.get('status') == 'Final']

        # Sort by date
        completed.sort(key=lambda g: g['date'])

        self.games = completed
        print(f"  Found {len(completed)} completed games")
        if completed:
            print(f"  Date range: {completed[0]['date']} to {completed[-1]['date']}")

        return completed

    def load_historical_player_stats(self):
        """Load player statistics from the training data cache."""
        print("\nLoading historical player statistics...")

        stats_loaded = 0

        # Load all player_stats batch files
        # Structure: {game_id: [list of player stat dicts]}
        batch_files = list(CACHE_DIR.glob("player_stats_batch_*.json"))
        for batch_file in batch_files:
            try:
                with open(batch_file) as f:
                    batch_data = json.load(f)

                if isinstance(batch_data, dict):
                    for game_id_str, game_stats in batch_data.items():
                        # game_stats is a list of player stat dicts for this game
                        if isinstance(game_stats, list):
                            for stat in game_stats:
                                player = stat.get('player', {})
                                player_id = player.get('id')
                                game = stat.get('game', {})
                                game_date = game.get('date', '')

                                if player_id and game_date:
                                    self.player_stats[player_id].append((game_date, stat))
                                    stats_loaded += 1

                                    if player_id not in self.player_info:
                                        self.player_info[player_id] = player

                                    # Also cache box score by game_id
                                    game_id = int(game_id_str)
                                    if game_id not in self.game_box_scores:
                                        self.game_box_scores[game_id] = {}
                                    self.game_box_scores[game_id][player_id] = {
                                        'player': player,
                                        'pts': stat.get('pts', 0) or 0,
                                        'reb': stat.get('reb', 0) or 0,
                                        'ast': stat.get('ast', 0) or 0,
                                        'fg3m': stat.get('fg3m', 0) or 0,
                                        'min': stat.get('min', '0'),
                                        'team_id': stat.get('team', {}).get('id'),
                                    }
            except Exception as e:
                print(f"  Warning: Could not load {batch_file}: {e}")

        # Also try loading individual season stats files
        for season in range(2020, 2026):
            season_file = CACHE_DIR / f"stats_{season}.json"
            if season_file.exists():
                try:
                    with open(season_file) as f:
                        stats_data = json.load(f)

                    for stat in stats_data:
                        player = stat.get('player', {})
                        player_id = player.get('id')
                        game = stat.get('game', {})
                        game_date = game.get('date', '')

                        if player_id and game_date:
                            self.player_stats[player_id].append((game_date, stat))
                            stats_loaded += 1

                            if player_id not in self.player_info:
                                self.player_info[player_id] = player
                except Exception as e:
                    print(f"  Warning: Could not load {season_file}: {e}")

        # Sort each player's games by date
        for player_id in self.player_stats:
            self.player_stats[player_id].sort(key=lambda x: x[0])

        print(f"  Loaded {stats_loaded} stat records for {len(self.player_stats)} players")
        print(f"  Cached box scores for {len(self.game_box_scores)} games")

    def fetch_box_scores_for_game(self, game: Dict) -> Dict[int, Dict]:
        """
        Fetch box scores for a specific game from cache or API.
        Returns dict of player_id -> box score stats.
        """
        game_id = game['id']

        # First check in-memory cache (populated by load_historical_player_stats)
        if game_id in self.game_box_scores:
            return self.game_box_scores[game_id]

        # Check if we have this game's box score in file cache
        box_file = CACHE_DIR / f"box_score_{game_id}.json"
        if box_file.exists():
            with open(box_file) as f:
                box_scores = json.load(f)
                # Convert string keys back to int
                return {int(k): v for k, v in box_scores.items()}

        # Try to get from API using the correct method
        try:
            from balldontlie_api import BalldontlieAPI
            api = BalldontlieAPI()

            # Use get_player_stats with game_ids filter
            stats = api.get_player_stats(game_ids=[game_id])

            box_scores = {}
            for stat in stats:
                player = stat.get('player', {})
                player_id = player.get('id')
                if player_id:
                    box_scores[player_id] = {
                        'player': player,
                        'pts': stat.get('pts', 0) or 0,
                        'reb': stat.get('reb', 0) or 0,
                        'ast': stat.get('ast', 0) or 0,
                        'fg3m': stat.get('fg3m', 0) or 0,
                        'min': stat.get('min', '0'),
                        'team_id': stat.get('team', {}).get('id'),
                    }

            # Cache in memory
            self.game_box_scores[game_id] = box_scores

            # Cache to file
            with open(box_file, 'w') as f:
                json.dump(box_scores, f)

            return box_scores
        except Exception as e:
            print(f"    Could not fetch box score for game {game_id}: {e}")
            return {}

    def get_player_features_before_date(self, player_id: int, game_date: str,
                                         opponent_id: int = None, is_home: bool = True,
                                         player_position: str = None) -> Optional[Dict]:
        """
        Generate features for a player using only data available before game_date.
        This is the CRITICAL point-in-time feature generation.

        IMPORTANT: Must generate ALL 91 features that models were trained on!
        """
        if player_id not in self.player_stats:
            return None

        # Get all games before this date
        games = [(d, s) for d, s in self.player_stats[player_id] if d < game_date]
        if len(games) < 3:  # Need minimum games
            return None

        # Sort by date descending (most recent first)
        games.sort(key=lambda x: x[0], reverse=True)

        window = 10
        recent = games[:window]
        last_5 = games[:5]
        last_3 = games[:3]

        # Calculate days rest
        try:
            current_date = datetime.strptime(game_date, "%Y-%m-%d")
            last_game_date = datetime.strptime(games[0][0], "%Y-%m-%d")
            days_rest = (current_date - last_game_date).days
        except:
            days_rest = 2

        # Helper to parse minutes
        def parse_min(min_val):
            if isinstance(min_val, (int, float)):
                return float(min_val)
            if not min_val:
                return 0.0
            try:
                if ':' in str(min_val):
                    parts = str(min_val).split(':')
                    return float(parts[0]) + float(parts[1]) / 60
                return float(min_val)
            except:
                return 0.0

        # Extract stats
        def get_stat(s, key):
            return s.get(key, 0) or 0

        pts = [get_stat(s, 'pts') for _, s in recent]
        reb = [get_stat(s, 'reb') for _, s in recent]
        ast = [get_stat(s, 'ast') for _, s in recent]
        fg3m = [get_stat(s, 'fg3m') for _, s in recent]
        mins = [parse_min(s.get('min', 0)) for _, s in recent]

        # Season stats for std calculations
        season_pts = [get_stat(s, 'pts') for _, s in games]
        season_reb = [get_stat(s, 'reb') for _, s in games]
        season_ast = [get_stat(s, 'ast') for _, s in games]
        season_fg3m = [get_stat(s, 'fg3m') for _, s in games]
        season_mins = [parse_min(s.get('min', 0)) for _, s in games]

        # Calculate averages
        season_pts_avg = np.mean(season_pts)
        season_reb_avg = np.mean(season_reb)
        season_ast_avg = np.mean(season_ast)
        season_fg3m_avg = np.mean(season_fg3m)
        season_min_avg = np.mean(season_mins)

        # Build features matching training script (all 91 features!)
        features = {
            # Season averages (1-6)
            'season_games': len(games),
            'season_pts_avg': season_pts_avg,
            'season_reb_avg': season_reb_avg,
            'season_ast_avg': season_ast_avg,
            'season_fg3m_avg': season_fg3m_avg,
            'season_min_avg': season_min_avg,

            # Recent averages (7-17)
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

            # Minutes features (18-20)
            'min_trend': np.mean([parse_min(s.get('min', 0)) for _, s in last_5]) - np.mean(mins) if mins else 0,
            'min_consistency': 1 - (np.std(mins) / np.mean(mins)) if np.mean(mins) > 0 else 0,
            'last5_min_avg': np.mean([parse_min(s.get('min', 0)) for _, s in last_5]),

            # Last 5 games (21-24)
            'last5_pts_avg': np.mean([get_stat(s, 'pts') for _, s in last_5]),
            'last5_reb_avg': np.mean([get_stat(s, 'reb') for _, s in last_5]),
            'last5_ast_avg': np.mean([get_stat(s, 'ast') for _, s in last_5]),
            'last5_fg3m_avg': np.mean([get_stat(s, 'fg3m') for _, s in last_5]),

            # Last 3 games (25-29) - NEW!
            'last3_pts_avg': np.mean([get_stat(s, 'pts') for _, s in last_3]),
            'last3_reb_avg': np.mean([get_stat(s, 'reb') for _, s in last_3]),
            'last3_ast_avg': np.mean([get_stat(s, 'ast') for _, s in last_3]),
            'last3_fg3m_avg': np.mean([get_stat(s, 'fg3m') for _, s in last_3]),
            'last3_min_avg': np.mean([parse_min(s.get('min', 0)) for _, s in last_3]),

            # Trends (30-33)
            'pts_trend': np.mean([get_stat(s, 'pts') for _, s in last_5]) - np.mean(pts),
            'reb_trend': np.mean([get_stat(s, 'reb') for _, s in last_5]) - np.mean(reb),
            'ast_trend': np.mean([get_stat(s, 'ast') for _, s in last_5]) - np.mean(ast),
            'fg3m_trend': np.mean([get_stat(s, 'fg3m') for _, s in last_5]) - np.mean(fg3m),

            # Season variance (34-37) - NEW!
            'season_pts_std': np.std(season_pts) if len(season_pts) > 1 else 0,
            'season_reb_std': np.std(season_reb) if len(season_reb) > 1 else 0,
            'season_ast_std': np.std(season_ast) if len(season_ast) > 1 else 0,
            'season_fg3m_std': np.std(season_fg3m) if len(season_fg3m) > 1 else 0,

            # Combined PRA stats (38-40)
            'pra_avg': np.mean([get_stat(s, 'pts') + get_stat(s, 'reb') + get_stat(s, 'ast') for _, s in recent]),
            'pra_std': np.std([get_stat(s, 'pts') + get_stat(s, 'reb') + get_stat(s, 'ast') for _, s in recent]) if len(recent) > 1 else 0,
            'last3_pra_avg': np.mean([get_stat(s, 'pts') + get_stat(s, 'reb') + get_stat(s, 'ast') for _, s in last_3]),

            # Efficiency stats (41-45)
            'ts_pct': self._calc_ts_pct(recent),
            'efg_pct': self._calc_efg_pct(recent),
            'usage_rate': self._calc_usage_rate(recent),
            'fg3_rate': self._calc_fg3_rate(recent),
            'fta_rate': self._calc_fta_rate(recent),

            # Advanced stats (46-48) - NEW!
            'bpm': self._calc_bpm(recent),
            'assist_rate': self._calc_assist_rate(recent),
            'rebound_rate': self._calc_rebound_rate(recent),

            # Rest features (49-50)
            'days_rest': days_rest,
            'is_back_to_back': 1 if days_rest == 1 else 0,

            # 3PM features (51-56)
            'fg3_pct': self._calc_fg3_pct(recent),
            'last5_fg3_pct': self._calc_fg3_pct(last_5),
            'fg3_pct_variance': self._calc_fg3_variance(games),
            **self._calc_fg3_streak_features(games),

            # Specialized 3PM features (57-66) - NEW!
            **self._calc_three_pm_features(recent, games, mins),

            # Position/role features (67-75) - Must be inferred from stats
            **self._infer_position_features(season_pts_avg, season_reb_avg, season_ast_avg, season_min_avg),

            # Opponent features (76-88) - Use league averages as defaults
            'opp_def_rating': 114.0,  # League average
            'opp_off_rating': 114.0,
            'opp_net_rating': 0.0,
            'opp_pts_allowed': 114.0,
            'opp_pts_allowed_recent': 114.0,
            'opp_pts_allowed_std': 5.0,
            'opp_pace': 100.0,
            'opp_pace_season': 100.0,
            'opp_def_strength': 0.0,
            'opp_reb_factor': 1.0,
            'opp_location_def': 114.0,
            'opp_win_pct': 0.5,
            'opp_recent_win_pct': 0.5,

            # Game context (89-91)
            'is_home': 1 if is_home else 0,
            'team_pace': 100.0,  # Default
            'team_off_rating': 114.0,  # Default
        }

        # TIER 2.2: Add position-specific opponent defense features
        if opponent_id and player_position:
            pos_defense_features = self.position_defense_calc.get_position_defense_before_date(
                team_id=opponent_id,
                game_date=game_date,
                player_position=player_position
            )
            features.update(pos_defense_features)
        else:
            # Use defaults if no opponent/position info
            features.update(self.position_defense_calc._get_default_features('F'))

        return features

    def _calc_bpm(self, games) -> float:
        """Calculate simplified Box Plus/Minus."""
        if not games:
            return 0.0

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

        total_pts = sum((s.get('pts', 0) or 0) for _, s in games)
        total_reb = sum((s.get('reb', 0) or 0) for _, s in games)
        total_ast = sum((s.get('ast', 0) or 0) for _, s in games)
        total_stl = sum((s.get('stl', 0) or 0) for _, s in games)
        total_blk = sum((s.get('blk', 0) or 0) for _, s in games)
        total_tov = sum((s.get('turnover', 0) or 0) for _, s in games)
        total_min = sum(parse_min(s.get('min', 0)) for _, s in games)

        if total_min < 10:
            return 0.0

        per36 = 36.0 / (total_min / len(games))
        bpm = ((total_pts + total_reb * 0.8 + total_ast * 1.1 +
                total_stl * 2.0 + total_blk * 1.5 - total_tov) / len(games) * per36 - 15) / 5
        return round(np.clip(bpm, -10, 10), 2)

    def _calc_assist_rate(self, games) -> float:
        """Calculate assists per 36 minutes."""
        if not games:
            return 4.0

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

        total_ast = sum((s.get('ast', 0) or 0) for _, s in games)
        total_min = sum(parse_min(s.get('min', 0)) for _, s in games)
        if total_min > 0:
            return round(total_ast / total_min * 36, 2)
        return 4.0

    def _calc_rebound_rate(self, games) -> float:
        """Calculate rebounds per 36 minutes."""
        if not games:
            return 6.0

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

        total_reb = sum((s.get('reb', 0) or 0) for _, s in games)
        total_min = sum(parse_min(s.get('min', 0)) for _, s in games)
        if total_min > 0:
            return round(total_reb / total_min * 36, 2)
        return 6.0

    def _calc_fg3_pct(self, games) -> float:
        """Calculate 3-point shooting percentage."""
        total_fg3m = sum((s.get('fg3m', 0) or 0) for _, s in games)
        total_fg3a = sum((s.get('fg3a', 0) or 0) for _, s in games)
        if total_fg3a > 0:
            return round(total_fg3m / total_fg3a, 3)
        return 0.36

    def _calc_fg3_variance(self, games) -> float:
        """Calculate variance in 3-point shooting."""
        game_fg3_pcts = []
        for _, g in games:
            fg3a = g.get('fg3a', 0) or 0
            fg3m = g.get('fg3m', 0) or 0
            if fg3a >= 3:
                game_fg3_pcts.append(fg3m / fg3a)
        if len(game_fg3_pcts) >= 3:
            return round(float(np.var(game_fg3_pcts)), 4)
        return 0.1

    def _calc_fg3_streak_features(self, games) -> Dict[str, float]:
        """Calculate hot/cold streak features for 3PM."""
        if len(games) < 3:
            return {'fg3_hot_streak': 0, 'fg3_cold_streak': 0, 'fg3_momentum': 0.0}

        recent_fg3_pcts = []
        for _, g in games[:5]:
            fg3a = g.get('fg3a', 0) or 0
            fg3m = g.get('fg3m', 0) or 0
            if fg3a >= 2:
                recent_fg3_pcts.append(fg3m / fg3a)

        if not recent_fg3_pcts:
            return {'fg3_hot_streak': 0, 'fg3_cold_streak': 0, 'fg3_momentum': 0.0}

        hot_streak = sum(1 for p in recent_fg3_pcts[:3] if p >= 0.40) >= 2
        cold_streak = sum(1 for p in recent_fg3_pcts[:3] if p <= 0.30) >= 2

        if len(recent_fg3_pcts) >= 3:
            momentum = np.polyfit(range(len(recent_fg3_pcts)), recent_fg3_pcts, 1)[0]
        else:
            momentum = 0.0

        return {
            'fg3_hot_streak': 1 if hot_streak else 0,
            'fg3_cold_streak': 1 if cold_streak else 0,
            'fg3_momentum': round(float(momentum), 4),
        }

    def _calc_three_pm_features(self, recent, all_games, mins) -> Dict[str, float]:
        """Calculate specialized 3PM prediction features."""
        LEAGUE_AVG_FG3_PCT = 0.36

        if not recent:
            return {
                'fg3a_per_min': 0.15,
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

        fg3a_values = [(s.get('fg3a', 0) or 0) for _, s in recent]
        fg3m_values = [(s.get('fg3m', 0) or 0) for _, s in recent]

        fg3a_avg = np.mean(fg3a_values) if fg3a_values else 0
        fg3a_std = np.std(fg3a_values) if len(fg3a_values) > 1 else 2.0
        fg3m_std = np.std(fg3m_values) if len(fg3m_values) > 1 else 1.0

        total_mins = sum(mins)
        total_fg3a = sum(fg3a_values)
        fg3a_per_min = (total_fg3a / total_mins) if total_mins > 0 else 0.15

        fg3a_consistency = 1 - (fg3a_std / max(fg3a_avg, 1)) if fg3a_avg > 0 else 0.5
        fg3a_consistency = max(0.3, min(1.0, fg3a_consistency))

        raw_fg3_pct = (sum(fg3m_values) / sum(fg3a_values)) if sum(fg3a_values) > 0 else LEAGUE_AVG_FG3_PCT

        total_attempts_season = sum((s.get('fg3a', 0) or 0) for _, s in all_games)
        regression_weight = min(1.0, total_attempts_season / 250)
        regressed_fg3_pct = regression_weight * raw_fg3_pct + (1 - regression_weight) * LEAGUE_AVG_FG3_PCT

        expected_fg3m = fg3a_avg * regressed_fg3_pct

        if len(fg3a_values) >= 3:
            last3_fg3a = np.mean(fg3a_values[:3])
            fg3_attempt_trend = last3_fg3a - fg3a_avg
        else:
            fg3_attempt_trend = 0.0

        is_volume_shooter = 1 if fg3a_avg >= 5 else 0
        sample_factor = min(1.0, len(all_games) / 20)
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

    def _infer_position_features(self, pts_avg, reb_avg, ast_avg, min_avg) -> Dict[str, float]:
        """Infer position and role features from stats (when position data unavailable)."""
        # Infer position from stats ratios
        # Centers: high reb, low ast
        # Guards: high ast, low reb
        # Forwards: balanced

        if reb_avg > 7 and ast_avg < 3:
            is_center, is_forward, is_guard = 1, 0, 0
        elif ast_avg > 5 and reb_avg < 5:
            is_center, is_forward, is_guard = 0, 0, 1
        elif reb_avg > 5:
            is_center, is_forward, is_guard = 0, 1, 0
        else:
            is_center, is_forward, is_guard = 0, 0, 1  # Default to guard

        # Role features
        is_starter = 1 if min_avg >= 25 else 0
        is_star = 1 if pts_avg >= 20 else 0
        is_high_volume = 1 if pts_avg >= 15 else 0
        is_ball_handler = 1 if ast_avg >= 4 else 0

        # Position-specific factors
        pos_reb_factor = 1.3 if is_center else (1.0 if is_forward else 0.7)
        pos_ast_factor = 1.3 if is_guard else (0.9 if is_forward else 0.6)

        return {
            'is_guard': is_guard,
            'is_forward': is_forward,
            'is_center': is_center,
            'is_starter': is_starter,
            'is_star': is_star,
            'is_high_volume': is_high_volume,
            'is_ball_handler': is_ball_handler,
            'pos_reb_factor': pos_reb_factor,
            'pos_ast_factor': pos_ast_factor,
        }

    def _calc_ts_pct(self, games) -> float:
        """Calculate True Shooting %."""
        total_pts = sum((s.get('pts', 0) or 0) for _, s in games)
        total_fga = sum((s.get('fga', 0) or 0) for _, s in games)
        total_fta = sum((s.get('fta', 0) or 0) for _, s in games)
        tsa = 2 * (total_fga + 0.44 * total_fta)
        return round(total_pts / tsa, 3) if tsa > 0 else 0.55

    def _calc_efg_pct(self, games) -> float:
        """Calculate Effective FG%."""
        total_fgm = sum((s.get('fgm', 0) or 0) for _, s in games)
        total_fg3m = sum((s.get('fg3m', 0) or 0) for _, s in games)
        total_fga = sum((s.get('fga', 0) or 0) for _, s in games)
        return round((total_fgm + 0.5 * total_fg3m) / total_fga, 3) if total_fga > 0 else 0.50

    def _calc_usage_rate(self, games) -> float:
        """Calculate approximate usage rate."""
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

        total_fga = sum((s.get('fga', 0) or 0) for _, s in games)
        total_fta = sum((s.get('fta', 0) or 0) for _, s in games)
        total_tov = sum((s.get('turnover', 0) or 0) for _, s in games)
        total_min = sum(parse_min(s.get('min', 0)) for _, s in games)
        return round((total_fga + 0.44 * total_fta + total_tov) / total_min, 3) if total_min > 0 else 0.2

    def _calc_fg3_rate(self, games) -> float:
        """Calculate 3PT attempt rate."""
        total_fg3a = sum((s.get('fg3a', 0) or 0) for _, s in games)
        total_fga = sum((s.get('fga', 0) or 0) for _, s in games)
        return round(total_fg3a / total_fga, 3) if total_fga > 0 else 0.35

    def _calc_fta_rate(self, games) -> float:
        """Calculate free throw rate."""
        total_fta = sum((s.get('fta', 0) or 0) for _, s in games)
        total_fga = sum((s.get('fga', 0) or 0) for _, s in games)
        return round(total_fta / total_fga, 3) if total_fga > 0 else 0.25

    # Bias corrections - set to 0 now that feature mismatch is fixed
    # These will be recalibrated after a clean backtest run
    BIAS_CORRECTIONS = {
        'points': 0.0,
        'rebounds': 0.0,
        'assists': 0.0,
        'threes': 0.0,
        'pra': 0.0,
    }

    def predict_minutes(self, features: Dict) -> Optional[float]:
        """
        TIER 2.3: Predict expected minutes for a player.
        Returns predicted minutes or None if model not available.
        """
        if self.minutes_model is None:
            return None

        try:
            model_data = self.minutes_model

            # Handle different model formats
            if isinstance(model_data, dict):
                # Check if it has the new format with regressor and scaler
                if 'minutes_regressor' in model_data:
                    regressor = model_data['minutes_regressor']
                    scaler = model_data.get('scaler')
                    feature_names = model_data.get('feature_names', [])

                    # Build feature vector
                    X = pd.DataFrame([features])
                    for col in feature_names:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[feature_names].fillna(0)

                    if scaler:
                        X_scaled = scaler.transform(X)
                    else:
                        X_scaled = X.values

                    predicted_min = float(regressor.predict(X_scaled)[0])
                elif 'model' in model_data:
                    # Legacy format
                    model = model_data['model']
                    scaler = model_data.get('scaler')
                    feature_names = model_data.get('feature_names', [])

                    X = pd.DataFrame([features])
                    for col in feature_names:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[feature_names].fillna(0)

                    if scaler:
                        X_scaled = scaler.transform(X)
                    else:
                        X_scaled = X.values

                    predicted_min = float(model.predict(X_scaled)[0])
                else:
                    return None
            else:
                # Assume it's a model object with predict method
                X = pd.DataFrame([features])
                predicted_min = float(model_data.predict(X)[0])

            # Clamp to realistic bounds
            return max(0, min(predicted_min, 48))

        except Exception as e:
            # Silently fail - minutes prediction is optional
            return None

    def predict(self, prop_type: str, features: Dict, apply_bias_correction: bool = True,
                predicted_minutes: Optional[float] = None) -> Optional[float]:
        """Make a prediction using the trained model."""
        if prop_type not in self.models:
            return None

        model_data = self.models[prop_type]

        # Calculate baseline from player's season averages (used for fallback)
        STAT_AVG_KEYS = {
            'points': 'season_pts_avg',
            'rebounds': 'season_reb_avg',
            'assists': 'season_ast_avg',
            'threes': 'season_fg3m_avg',
            'pra': 'pra_avg',
        }
        baseline = features.get(STAT_AVG_KEYS.get(prop_type, 'season_pts_avg'), 0)

        # Handle new PropEnsembleModel format (has 'models' key with base models + meta_model)
        if isinstance(model_data, dict) and 'models' in model_data and 'meta_model' in model_data:
            # New stacked ensemble format
            base_models = model_data['models']
            meta_model = model_data['meta_model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            model_weights = model_data.get('model_weights', {})

            # Build feature array in correct order
            X = pd.DataFrame([features])
            for col in feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_names].fillna(0)

            X_scaled = scaler.transform(X)

            # Get base model predictions
            base_preds = []
            for name, model in base_models.items():
                pred = model.predict(X_scaled)[0]
                base_preds.append(pred)

            # Use meta model for stacking or simple weighted average
            if meta_model is not None:
                meta_features = np.array(base_preds).reshape(1, -1)
                predicted = float(meta_model.predict(meta_features)[0])
            else:
                # Weighted average fallback
                weights = list(model_weights.values()) if model_weights else [1.0/len(base_preds)] * len(base_preds)
                predicted = float(np.average(base_preds, weights=weights))

        elif hasattr(model_data, 'predict'):
            # Object with predict method
            X = pd.DataFrame([features])
            predicted = float(model_data.predict(X)[0])
        else:
            # Legacy format with model/scaler/feature_names dict
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']

            # Build feature array in correct order
            X = pd.DataFrame([features])
            for col in feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_names].fillna(0)

            X_scaled = scaler.transform(X)
            predicted = float(model.predict(X_scaled)[0])

        # Apply bias correction
        if apply_bias_correction and prop_type in self.BIAS_CORRECTIONS:
            predicted += self.BIAS_CORRECTIONS[prop_type]

        # TIER 2.3: Minutes-based prediction scaling
        # If we have predicted minutes and they're low, scale down the prediction
        if predicted_minutes is not None and predicted_minutes < 20:
            # Player expected to play fewer than 20 minutes - scale predictions
            # Use player's typical minutes (from features) as baseline
            season_min_avg = features.get('season_min_avg', 25)
            if season_min_avg > 0:
                # Calculate ratio of expected minutes to typical minutes
                min_ratio = predicted_minutes / season_min_avg
                # Apply scaling: if expected to play 50% of normal minutes, scale down by 50%
                # But only scale down, never up
                if min_ratio < 1.0:
                    predicted = predicted * max(min_ratio, 0.1)  # Floor at 10%

        # CRITICAL: Clamp predictions to realistic bounds
        # NBA player stats cannot be negative and have practical maximums
        PROP_BOUNDS = {
            'points': (0, 70),      # Wilt scored 100, but realistic max ~70
            'rebounds': (0, 35),    # Record is 55, realistic max ~35
            'assists': (0, 30),     # Record is 30
            'threes': (0, 15),      # Record is 14
            'pra': (0, 100),        # Combined stat
        }

        if prop_type in PROP_BOUNDS:
            min_val, max_val = PROP_BOUNDS[prop_type]
            predicted = max(min_val, min(predicted, max_val))

        # CRITICAL: Baseline fallback for when model produces unreasonably low predictions
        # If the model predicts < 20% of player's season average, blend with baseline
        # This prevents stars like Luka getting predictions of 0 when they should be ~50 PRA
        if baseline > 5 and predicted < baseline * 0.2:
            # Model is predicting way too low - use weighted blend
            # Weight: 30% model, 70% baseline when prediction is extremely off
            blend_weight = 0.3
            predicted = blend_weight * predicted + (1 - blend_weight) * baseline

        return predicted

    def run_backtest(self) -> BacktestResults:
        """Run the full backtest on all 2025-26 season games."""
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE BACKTEST")
        print("="*60)

        # Load everything
        self.load_models()
        self.load_games()
        self.load_historical_player_stats()

        if not self.games:
            print("No games to backtest!")
            return BacktestResults()

        results = BacktestResults()
        results.start_date = self.games[0]['date']
        results.end_date = self.games[-1]['date']

        print(f"\nProcessing {len(self.games)} games...")

        for i, game in enumerate(self.games):
            game_id = game['id']
            game_date = game['date']
            home_team = game.get('home_team', {})
            away_team = game.get('visitor_team', {})

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(self.games)} games...")

            # Get box scores (actual results) for this game
            box_scores = self.fetch_box_scores_for_game(game)

            if not box_scores:
                # If no box scores, try to find in our player_stats cache
                for player_id, stats_list in self.player_stats.items():
                    for date, stat in stats_list:
                        if date == game_date:
                            stat_game = stat.get('game', {})
                            if stat_game.get('id') == game_id:
                                player = stat.get('player', {})
                                team = stat.get('team', {})
                                box_scores[player_id] = {
                                    'player': player,
                                    'pts': stat.get('pts', 0) or 0,
                                    'reb': stat.get('reb', 0) or 0,
                                    'ast': stat.get('ast', 0) or 0,
                                    'fg3m': stat.get('fg3m', 0) or 0,
                                    'min': stat.get('min', '0'),
                                    'team_id': team.get('id'),
                                }

            if not box_scores:
                results.games_with_errors += 1
                continue

            results.games_processed += 1

            # TIER 2.2: Process this game through position defense calculator
            # Convert box_scores dict to list format for process_game
            player_stats_list = []
            for pid, stats in box_scores.items():
                player_stats_list.append({
                    'player': stats.get('player', {}),
                    'team_id': stats.get('team_id'),
                    'pts': stats.get('pts', 0),
                    'reb': stats.get('reb', 0),
                    'ast': stats.get('ast', 0),
                    'fg3m': stats.get('fg3m', 0),
                    'min': stats.get('min', '0'),
                })
            self.position_defense_calc.process_game(
                game_id=game_id,
                game_date=game_date,
                home_team_id=home_team.get('id'),
                away_team_id=away_team.get('id'),
                player_stats=player_stats_list
            )

            # For each player in box score, generate prediction and compare
            for player_id, actual_stats in box_scores.items():
                player_name = actual_stats.get('player', {}).get('first_name', '') + ' ' + \
                             actual_stats.get('player', {}).get('last_name', 'Unknown')
                player_team_id = actual_stats.get('team_id')
                is_home = player_team_id == home_team.get('id')

                # Get player position for TIER 2.2 position defense features
                player_position = actual_stats.get('player', {}).get('position', 'F')

                # Get features using only pre-game data
                features = self.get_player_features_before_date(
                    player_id, game_date,
                    opponent_id=away_team.get('id') if is_home else home_team.get('id'),
                    is_home=is_home,
                    player_position=player_position
                )

                if not features:
                    continue

                # TIER 2.3: Predict expected minutes for this player
                predicted_minutes = self.predict_minutes(features)

                # Make predictions for each prop type
                for prop_type in self.PROP_TYPES:
                    pred_value = self.predict(prop_type, features, predicted_minutes=predicted_minutes)
                    if pred_value is None:
                        continue

                    # Get actual value
                    stat_key = self.PROP_STAT_MAP[prop_type]
                    if stat_key == 'pra':
                        actual_value = (actual_stats.get('pts', 0) or 0) + \
                                      (actual_stats.get('reb', 0) or 0) + \
                                      (actual_stats.get('ast', 0) or 0)
                    else:
                        actual_value = actual_stats.get(stat_key, 0) or 0

                    # Skip if player didn't play
                    if actual_value == 0 and prop_type == 'points':
                        continue

                    # Record prediction
                    pred = PropPrediction(
                        player_id=player_id,
                        player_name=player_name.strip(),
                        team=home_team.get('abbreviation', '?') if is_home else away_team.get('abbreviation', '?'),
                        prop_type=prop_type,
                        predicted=pred_value,
                        actual=actual_value,
                        game_id=game_id,
                        game_date=game_date,
                        is_home=is_home,
                        days_rest=features.get('days_rest', 2),
                    )
                    results.add(pred)

            # Add this game's stats to our history for future predictions
            for player_id, stats in box_scores.items():
                # Reconstruct the stat format
                stat_record = {
                    'pts': stats.get('pts', 0),
                    'reb': stats.get('reb', 0),
                    'ast': stats.get('ast', 0),
                    'fg3m': stats.get('fg3m', 0),
                    'min': stats.get('min', '0'),
                    'fgm': stats.get('fgm', 0),
                    'fga': stats.get('fga', 0),
                    'fta': stats.get('fta', 0),
                    'turnover': stats.get('turnover', 0),
                    'game': {'id': game_id, 'date': game_date},
                    'team': {'id': stats.get('team_id')},
                }
                self.player_stats[player_id].append((game_date, stat_record))
                # Keep sorted
                self.player_stats[player_id].sort(key=lambda x: x[0])

        return results

    def generate_report(self, results: BacktestResults):
        """Generate a detailed backtest report."""
        print("\n" + "="*60)
        print(f"2025-26 SEASON BACKTEST RESULTS")
        print("="*60)
        print(f"Games Analyzed: {results.games_processed}")
        print(f"Games with Errors: {results.games_with_errors}")
        print(f"Date Range: {results.start_date} to {results.end_date}")
        print(f"Total Predictions: {len(results.predictions)}")

        # Overall metrics
        print("\n--- OVERALL ACCURACY ---")
        overall = results.calculate_metrics()
        print(f"RMSE: {overall.get('rmse', 'N/A')}")
        print(f"MAE: {overall.get('mae', 'N/A')}")
        print(f"R²: {overall.get('r2', 'N/A')}")
        print(f"Bias: {overall.get('bias', 'N/A')}")

        # By prop type
        print("\n--- BY PROP TYPE ---")
        print(f"{'Type':<12} {'Count':>8} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Bias':>8}")
        print("-" * 56)
        for prop_type in self.PROP_TYPES:
            preds = results.get_by_prop_type(prop_type)
            if preds:
                m = results.calculate_metrics(preds)
                print(f"{prop_type:<12} {m.get('count', 0):>8} {m.get('rmse', 0):>8.2f} "
                      f"{m.get('mae', 0):>8.2f} {m.get('r2', 0):>8.2f} {m.get('bias', 0):>8.2f}")

        # By home/away
        print("\n--- BY HOME/AWAY ---")
        home_preds = results.get_by_home_away(True)
        away_preds = results.get_by_home_away(False)
        home_m = results.calculate_metrics(home_preds)
        away_m = results.calculate_metrics(away_preds)
        print(f"Home: RMSE={home_m.get('rmse', 'N/A')}, MAE={home_m.get('mae', 'N/A')}, Count={home_m.get('count', 0)}")
        print(f"Away: RMSE={away_m.get('rmse', 'N/A')}, MAE={away_m.get('mae', 'N/A')}, Count={away_m.get('count', 0)}")

        # By rest days
        print("\n--- BY REST DAYS ---")
        b2b_preds = results.get_by_rest_days(1, 1)
        normal_preds = results.get_by_rest_days(2, 3)
        rested_preds = results.get_by_rest_days(4, 30)

        if b2b_preds:
            b2b_m = results.calculate_metrics(b2b_preds)
            print(f"Back-to-back (1 day): RMSE={b2b_m.get('rmse', 'N/A')}, Count={b2b_m.get('count', 0)}")
        if normal_preds:
            normal_m = results.calculate_metrics(normal_preds)
            print(f"Normal rest (2-3 days): RMSE={normal_m.get('rmse', 'N/A')}, Count={normal_m.get('count', 0)}")
        if rested_preds:
            rested_m = results.calculate_metrics(rested_preds)
            print(f"Well rested (4+ days): RMSE={rested_m.get('rmse', 'N/A')}, Count={rested_m.get('count', 0)}")

        # Worst predictions
        print("\n--- TOP 10 WORST PREDICTIONS (by absolute error) ---")
        sorted_preds = sorted(results.predictions, key=lambda p: p.abs_error, reverse=True)[:10]
        for p in sorted_preds:
            print(f"  {p.player_name[:20]:<20} {p.prop_type:<10} Pred={p.predicted:>6.1f} "
                  f"Actual={p.actual:>6.1f} Error={p.error:>+7.1f}")

        # Players with highest average error
        print("\n--- PLAYERS WITH HIGHEST AVG ERROR (min 5 predictions) ---")
        player_errors = defaultdict(list)
        for p in results.predictions:
            player_errors[p.player_name].append(p.abs_error)

        player_avg_errors = []
        for name, errors in player_errors.items():
            if len(errors) >= 5:
                player_avg_errors.append((name, np.mean(errors), len(errors)))

        player_avg_errors.sort(key=lambda x: x[1], reverse=True)
        for name, avg_err, count in player_avg_errors[:10]:
            print(f"  {name:<25} Avg Error={avg_err:>6.2f} (n={count})")

        # Recommendations
        print("\n--- IMPROVEMENT RECOMMENDATIONS ---")

        # Check for prop-specific issues
        for prop_type in self.PROP_TYPES:
            preds = results.get_by_prop_type(prop_type)
            if preds:
                m = results.calculate_metrics(preds)
                if abs(m.get('bias', 0)) > 1:
                    direction = "over" if m['bias'] > 0 else "under"
                    print(f"  - {prop_type.upper()}: Model {direction}predicts by {abs(m['bias']):.1f} on average. "
                          f"Consider adjusting predictions down/up.")

        # Check rest days impact
        if b2b_preds and normal_preds:
            b2b_rmse = results.calculate_metrics(b2b_preds).get('rmse', 0)
            normal_rmse = results.calculate_metrics(normal_preds).get('rmse', 0)
            if b2b_rmse > normal_rmse * 1.2:
                print(f"  - BACK-TO-BACK: Model struggles on B2B games (RMSE {b2b_rmse:.2f} vs {normal_rmse:.2f}). "
                      f"Consider stronger B2B adjustment.")

        print("\n" + "="*60)


def main():
    """Main entry point."""
    backtester = SeasonBacktester(season=2025)
    results = backtester.run_backtest()
    backtester.generate_report(results)

    # Save results for further analysis
    output_file = Path("backtest_results_2025.json")
    output_data = {
        'games_processed': results.games_processed,
        'games_with_errors': results.games_with_errors,
        'start_date': results.start_date,
        'end_date': results.end_date,
        'total_predictions': len(results.predictions),
        'metrics': {
            prop_type: results.calculate_metrics(results.get_by_prop_type(prop_type))
            for prop_type in backtester.PROP_TYPES
        },
        'overall': results.calculate_metrics(),
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
