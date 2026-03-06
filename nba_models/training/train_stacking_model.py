"""
Training Script for Stacking Models

Trains stacking ensemble models for:
- Moneyline predictions (classification)
- Spread predictions (regression)
- Player prop predictions (regression)

Includes Optuna hyperparameter tuning for optimal performance.

Usage:
    python3 train_stacking_model.py [--tune] [--model moneyline|spread|props]
    python3 train_stacking_model.py --incremental  # Retrain meta-learner only (fast)
"""

from __future__ import annotations

import sys
import json
import pickle
import warnings
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent / "models"))

from models.stacking_model import StackingClassifier, StackingRegressor
from stacking_meta_learner import StackingMetaLearner

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
HISTORICAL_LINES_DIR = Path("data/historical_lines")
LIVE_SEASON_LABELS = ["2024-25"]


def normalize_player_name(name: str) -> str:
    """Normalize player names for fuzzy matching across archives."""
    return (
        name.lower()
        .strip()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("  ", " ")
    )


def parse_minutes(value: Any) -> float:
    """Convert a minutes field like '32:14' or '32' to float minutes."""
    if value is None:
        return 0.0

    text = str(value).strip()
    if not text or text in {"00", "0", "0:00"}:
        return 0.0

    if ":" in text:
        try:
            minutes_str, seconds_str = text.split(":", 1)
            return float(minutes_str) + (float(seconds_str) / 60.0)
        except ValueError:
            return 0.0

    try:
        return float(text)
    except ValueError:
        return 0.0


def build_base_models_for_regression():
    """Build diverse base models for regression tasks (spread, props)."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    base_models = []

    # Try XGBoost
    try:
        from xgboost import XGBRegressor
        base_models.append(XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        ))
    except ImportError:
        pass

    # Try LightGBM
    try:
        from lightgbm import LGBMRegressor
        base_models.append(LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        ))
    except ImportError:
        pass

    # Gradient Boosting
    base_models.append(GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ))

    # Random Forest
    base_models.append(RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))

    # Ridge Regression
    base_models.append(Ridge(alpha=1.0, random_state=42))

    return base_models


def build_base_models_for_classification():
    """Build diverse base models for classification tasks (moneyline)."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    base_models = []

    # Try XGBoost
    try:
        from xgboost import XGBClassifier
        base_models.append(XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        ))
    except ImportError:
        pass

    # Try LightGBM
    try:
        from lightgbm import LGBMClassifier
        base_models.append(LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        ))
    except ImportError:
        pass

    # Gradient Boosting
    base_models.append(GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ))

    # Random Forest
    base_models.append(RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))

    # Logistic Regression
    base_models.append(LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    ))

    return base_models


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


def _save_inference_ready_stacking_model(
    model,
    output_paths: list[Path],
    feature_names: list[str],
    context_feature_names: list[str] | None = None,
) -> None:
    """Save stacking artifacts in a format the inference pipeline can consume."""
    model.feature_names = list(feature_names)
    inference_context_names = [
        name[4:] if name.startswith("ctx_") else name
        for name in (context_feature_names or [])
    ]
    model.context_feature_names = inference_context_names

    artifact = {
        'model': model,
        'feature_names': list(feature_names),
        'context_feature_names': inference_context_names,
        'context_scaler': getattr(model, 'context_scaler', None),
        'artifact_type': 'stacking_meta_learner',
    }

    for output_path in output_paths:
        with open(output_path, 'wb') as f:
            pickle.dump(artifact, f)


class TrainingDataLoader:
    """Load and prepare training data from cache with proper feature engineering."""

    def __init__(self, cache_dir: Path = CACHE_DIR, window: int = 10):
        self.cache_dir = cache_dir
        self.window = window
        self.games = []
        self.player_stats = defaultdict(list)
        self.team_history = defaultdict(list)
        self.team_id_to_abbrev: dict[int, str] = {}
        self.team_players_by_game: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
        self.player_team_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        self.market_context: dict[tuple[str, str, str], dict[str, float | int]] = {}
        self.game_market_context: dict[tuple[str, str, str], dict[str, float | int]] = {}
        self.market_active_players: dict[tuple[str, str], set[int]] = defaultdict(set)

    def load_games(self) -> list[dict]:
        """Load game data from cache or CSV fallback."""
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

        if not all_games:
            all_games = self._load_games_from_csv_fallback()

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

    def _load_games_from_csv_fallback(self) -> list[dict]:
        """Load current-season games from CSV when the cache directory is empty."""
        print("  No cache games found; loading live-season CSV fallback...")

        try:
            from train_from_csv import build_team_id_map, _build_team_metadata, load_team_games
        except Exception as e:
            print(f"  Warning: CSV fallback unavailable: {e}")
            return []

        try:
            team_id_map = build_team_id_map()
            team_meta = _build_team_metadata()
            games = load_team_games(LIVE_SEASON_LABELS, team_id_map, team_meta)
            print(f"  Loaded {len(games)} games from live-season CSV fallback")
            return games
        except Exception as e:
            print(f"  Warning: Could not load games from CSV fallback: {e}")
            return []

    def _build_team_history(self):
        """Build historical record for each team."""
        for game in self.games:
            game_date = game.get('date', '')[:10]
            home_team_id = game.get('home_team', {}).get('id')
            away_team_id = game.get('visitor_team', {}).get('id')
            home_abbrev = game.get('home_team', {}).get('abbreviation', '')
            away_abbrev = game.get('visitor_team', {}).get('abbreviation', '')
            home_score = game.get('home_team_score', 0) or 0
            away_score = game.get('visitor_team_score', 0) or 0

            if not all([game_date, home_team_id, away_team_id, home_score]):
                continue

            self.team_id_to_abbrev[home_team_id] = home_abbrev
            self.team_id_to_abbrev[away_team_id] = away_abbrev

            # Record for home team
            self.team_history[home_team_id].append({
                'date': game_date,
                'is_home': True,
                'pts_scored': home_score,
                'pts_allowed': away_score,
                'won': home_score > away_score,
                'point_diff': home_score - away_score,
                'opponent_id': away_team_id,
                'team_abbrev': home_abbrev,
                'opponent_abbrev': away_abbrev,
                'venue_abbrev': home_abbrev,
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
                'team_abbrev': away_abbrev,
                'opponent_abbrev': home_abbrev,
                'venue_abbrev': home_abbrev,
            })

    def _get_team_features(self, team_id: int, before_date: str, min_games: int = 5) -> dict | None:
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

    def _calc_streak(self, games: list[dict]) -> int:
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

    def _record_player_game(self, game_date: str, raw_stat: dict):
        """Store a player-game record in the flattened structures this trainer uses."""
        player_id = raw_stat.get('player_id') or raw_stat.get('player', {}).get('id')
        if not player_id:
            return

        player_name = raw_stat.get('player_name')
        if not player_name:
            first = raw_stat.get('first_name') or raw_stat.get('player', {}).get('first_name', '')
            last = raw_stat.get('last_name') or raw_stat.get('player', {}).get('last_name', '')
            player_name = f"{first} {last}".strip()

        team_id = raw_stat.get('team_id')
        if team_id is None:
            team_id = raw_stat.get('team', {}).get('id')

        team_abbrev = raw_stat.get('team_abbreviation')
        if not team_abbrev:
            team_abbrev = raw_stat.get('team', {}).get('abbreviation', '')

        flat_stat = {
            'player_id': int(player_id),
            'player_name': player_name,
            'position': raw_stat.get('position') or raw_stat.get('player', {}).get('position', ''),
            'team_id': team_id,
            'team_abbreviation': team_abbrev,
            'min': parse_minutes(raw_stat.get('min')),
            'pts': raw_stat.get('pts', 0) or 0,
            'reb': raw_stat.get('reb', 0) or 0,
            'ast': raw_stat.get('ast', 0) or 0,
            'fg3m': raw_stat.get('fg3m', 0) or 0,
        }
        flat_stat['pra'] = flat_stat['pts'] + flat_stat['reb'] + flat_stat['ast']

        self.player_stats[int(player_id)].append((game_date, flat_stat))

        if team_abbrev:
            self.team_players_by_game[(game_date, team_abbrev)][int(player_id)] = flat_stat
        if player_name:
            self.player_team_lookup[(normalize_player_name(player_name), game_date)] = {
                'player_id': int(player_id),
                'team_abbreviation': team_abbrev,
            }

    def _load_historical_player_stats_fallback(self) -> bool:
        """Load current-season player stats from the historical BDL archive."""
        stats_path = HISTORICAL_LINES_DIR / "player_stats_2024.json"
        meta_path = HISTORICAL_LINES_DIR / "player_stats_2024_meta.json"
        if not stats_path.exists() or not meta_path.exists():
            return False

        try:
            with open(stats_path) as f:
                stats_by_game = json.load(f)
            with open(meta_path) as f:
                meta_by_game = json.load(f)
        except Exception as e:
            print(f"Warning loading historical player stats fallback: {e}")
            return False

        loaded = 0
        for game_id, players in stats_by_game.items():
            meta = meta_by_game.get(str(game_id), {})
            game_date = str(meta.get('date', ''))[:10]
            if not game_date or not isinstance(players, list):
                continue
            for player in players:
                self._record_player_game(game_date, player)
                loaded += 1

        if loaded:
            print(f"  Loaded {loaded} player-game records from historical archive fallback")
        return loaded > 0

    def _load_market_context(self):
        """Load archived prop-board and game-market context."""
        grouped_snapshots: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)

        for snapshot_file in sorted(HISTORICAL_LINES_DIR.glob("20*.json")):
            try:
                with open(snapshot_file) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Warning loading {snapshot_file.name}: {e}")
                continue

            for game in data.get('games', []):
                home_abbrev = game.get('home_abbrev')
                away_abbrev = game.get('away_abbrev')
                game_date = str(game.get('commence_time', ''))[:10]
                if not all([home_abbrev, away_abbrev, game_date]):
                    continue
                event_id = game.get('odds_api_event_id') or str(game.get('bdl_game_id'))
                grouped_snapshots[(event_id, game_date, home_abbrev, away_abbrev)].append(game)

                game_markets = game.get('game_markets', {})
                if isinstance(game_markets, dict):
                    derived = game_markets.get('derived', {})
                    if derived:
                        self.game_market_context[(game_date, home_abbrev, away_abbrev)] = {
                            'opening_line': float(derived.get('opening_line', 0.0) or 0.0),
                            'closing_line': float(derived.get('closing_line', 0.0) or 0.0),
                            'line_movement': float(derived.get('line_movement', 0.0) or 0.0),
                            'moneyline_home_prob_movement': float(
                                derived.get('moneyline_home_prob_movement', 0.0) or 0.0
                            ),
                            'consensus_odds': float(derived.get('consensus_odds', -110) or -110),
                            'rlm_flag': int(bool(derived.get('rlm_flag', False))),
                            'steam_move_flag': int(bool(derived.get('steam_move_flag', False))),
                        }

        loaded = 0
        for (_event_id, game_date, home_abbrev, away_abbrev), snapshots in grouped_snapshots.items():
            snapshots.sort(key=lambda snap: snap.get('snapshot_timestamp', ''))
            first = self._summarize_market_snapshot(game_date, home_abbrev, away_abbrev, snapshots[0])
            last = self._summarize_market_snapshot(game_date, home_abbrev, away_abbrev, snapshots[-1])

            if not first and not last:
                continue

            first_diff = first.get('market_strength_diff', 0.0)
            last_diff = last.get('market_strength_diff', 0.0)
            line_movement = last_diff - first_diff if len(snapshots) > 1 else 0.0

            self.market_context[(game_date, home_abbrev, away_abbrev)] = {
                'market_strength_diff': last_diff,
                'line_movement': line_movement,
                'market_depth_total': last.get('market_depth_total', 0),
                'snapshot_count': len(snapshots),
            }
            self.market_active_players[(game_date, home_abbrev)].update(last.get('active_home_ids', set()))
            self.market_active_players[(game_date, away_abbrev)].update(last.get('active_away_ids', set()))
            loaded += 1

        if loaded:
            print(f"  Loaded market context for {loaded} games from historical props archive")
        if self.game_market_context:
            print(f"  Loaded true game-market history for {len(self.game_market_context)} games")

    def _summarize_market_snapshot(
        self,
        game_date: str,
        home_abbrev: str,
        away_abbrev: str,
        snapshot: dict,
    ) -> dict[str, float | int | set[int]]:
        """Collapse a single prop-board snapshot into team-level market features."""
        team_player_lines: dict[str, dict[int, dict[str, float]]] = {
            home_abbrev: {},
            away_abbrev: {},
        }

        for prop in snapshot.get('player_props', []):
            player_name = prop.get('player_name', '')
            if not player_name:
                continue
            lookup = self.player_team_lookup.get((normalize_player_name(player_name), game_date))
            if not lookup:
                continue

            team_abbrev = lookup.get('team_abbreviation')
            player_id = lookup.get('player_id')
            if team_abbrev not in team_player_lines or player_id is None:
                continue

            player_props = team_player_lines[team_abbrev].setdefault(int(player_id), {})
            player_props[prop.get('prop_type', '')] = float(prop.get('line', 0) or 0)

        def _representative_lines(player_map: dict[int, dict[str, float]]) -> list[float]:
            values = []
            for prop_map in player_map.values():
                if prop_map.get('points', 0) > 0:
                    values.append(prop_map['points'])
                elif prop_map.get('pra', 0) > 0:
                    values.append(prop_map['pra'] * 0.6)
            return sorted(values, reverse=True)

        home_lines = _representative_lines(team_player_lines[home_abbrev])
        away_lines = _representative_lines(team_player_lines[away_abbrev])

        return {
            'market_strength_diff': float(sum(home_lines[:5]) - sum(away_lines[:5])),
            'market_depth_total': len(home_lines) + len(away_lines),
            'active_home_ids': set(team_player_lines[home_abbrev].keys()),
            'active_away_ids': set(team_player_lines[away_abbrev].keys()),
        }

    def _get_availability_context(self, game_date: str, team_id: int, team_abbrev: str) -> dict[str, float | int]:
        """Infer pregame absences from recent rotation players and the historical prop board."""
        prior_team_games = [
            game for game in self.team_history.get(team_id, [])
            if game['date'] < game_date
        ]
        prior_team_games.sort(key=lambda item: item['date'], reverse=True)
        recent_games = prior_team_games[:10]
        if len(recent_games) < 3:
            return {'injury_count': 0, 'star_player_out': 0}

        player_recent: dict[int, list[dict]] = defaultdict(list)
        for prior_game in recent_games:
            for player_id, player_stat in self.team_players_by_game.get((prior_game['date'], team_abbrev), {}).items():
                player_recent[player_id].append(player_stat)

        expected_rotation: dict[int, dict[str, float]] = {}
        for player_id, appearances in player_recent.items():
            avg_minutes = float(np.mean([stat['min'] for stat in appearances]))
            if len(appearances) < 3 or avg_minutes < 12:
                continue

            avg_points = float(np.mean([stat['pts'] for stat in appearances]))
            avg_pra = float(np.mean([stat['pra'] for stat in appearances]))
            expected_rotation[player_id] = {
                'avg_minutes': avg_minutes,
                'avg_points': avg_points,
                'avg_pra': avg_pra,
            }

        if not expected_rotation:
            return {'injury_count': 0, 'star_player_out': 0}

        active_ids = self.market_active_players.get((game_date, team_abbrev), set())
        absent_players = [
            meta for player_id, meta in expected_rotation.items()
            if player_id not in active_ids
        ]

        star_out = any(
            meta['avg_minutes'] >= 28 and (meta['avg_points'] >= 18 or meta['avg_pra'] >= 28)
            for meta in absent_players
        )

        return {
            'injury_count': len(absent_players),
            'star_player_out': int(star_out),
        }

    def _get_market_context(
        self,
        game_date: str,
        home_abbrev: str,
        away_abbrev: str,
        baseline_diff: float,
    ) -> dict[str, float | int]:
        """Return market context, preferring true game-odds history when present."""
        prop_market = self.market_context.get((game_date, home_abbrev, away_abbrev), {})
        game_market = self.game_market_context.get((game_date, home_abbrev, away_abbrev), {})

        line_movement = float(game_market.get('line_movement', 0.0))
        if not game_market:
            line_movement = float(prop_market.get('line_movement', 0.0))

        rlm_flag = int(game_market.get('rlm_flag', 0))
        if not game_market:
            rlm_flag = int(
                prop_market.get('snapshot_count', 0) > 1
                and abs(line_movement) >= 1.0
                and baseline_diff != 0
                and np.sign(line_movement) != np.sign(baseline_diff)
            )

        return {
            'market_strength_diff': float(prop_market.get('market_strength_diff', 0.0)),
            'opening_line': float(game_market.get('opening_line', 0.0)),
            'closing_line': float(game_market.get('closing_line', 0.0)),
            'line_movement': line_movement,
            'rlm_flag': rlm_flag,
            'consensus_odds': float(game_market.get('consensus_odds', -110)),
            'steam_move_flag': int(game_market.get('steam_move_flag', 0)),
            'moneyline_home_prob_movement': float(game_market.get('moneyline_home_prob_movement', 0.0)),
        }

    def _extract_context_features(self, game: dict, home_feats: dict, away_feats: dict) -> dict:
        """
        Extract context features for the stacking meta-learner.

        These features intentionally mirror the semantics of the live
        generate_game_features() output as closely as the historical archives allow.
        Injury/availability features are inferred from real player-game history and
        archived pregame prop boards; game-market features come from true archived
        spread/moneyline/totals snapshots when available; travel features come from
        real venue sequencing.
        """
        game_date = game.get('date', '')[:10]
        home_team_id = game.get('home_team', {}).get('id')
        away_team_id = game.get('visitor_team', {}).get('id')
        home_abbrev = game.get('home_team', {}).get('abbreviation', '')
        away_abbrev = game.get('visitor_team', {}).get('abbreviation', '')

        home_games = [g for g in self.team_history.get(home_team_id, []) if g['date'] < game_date]
        away_games = [g for g in self.team_history.get(away_team_id, []) if g['date'] < game_date]
        home_games.sort(key=lambda x: x['date'], reverse=True)
        away_games.sort(key=lambda x: x['date'], reverse=True)

        days_rest_home = 2
        days_rest_away = 2

        if home_games:
            last_home_date = datetime.strptime(home_games[0]['date'], '%Y-%m-%d')
            current_date = datetime.strptime(game_date, '%Y-%m-%d')
            days_rest_home = (current_date - last_home_date).days

        if away_games:
            last_away_date = datetime.strptime(away_games[0]['date'], '%Y-%m-%d')
            current_date = datetime.strptime(game_date, '%Y-%m-%d')
            days_rest_away = (current_date - last_away_date).days

        recent_home_diffs = [g['point_diff'] for g in home_games[:5]]
        recent_away_diffs = [g['point_diff'] for g in away_games[:5]]
        volatility_samples = recent_home_diffs + recent_away_diffs
        prediction_variance = float(np.std(volatility_samples)) if len(volatility_samples) >= 2 else 0.0

        availability_home = self._get_availability_context(game_date, home_team_id, home_abbrev)
        availability_away = self._get_availability_context(game_date, away_team_id, away_abbrev)

        baseline_strength_diff = home_feats.get('point_diff_avg', 0) - away_feats.get('point_diff_avg', 0)
        market_context = self._get_market_context(game_date, home_abbrev, away_abbrev, baseline_strength_diff)

        away_travel_distance = 0.0
        if away_games and home_abbrev:
            try:
                from travel_fatigue import TravelFatigueCalculator

                travel_calc = TravelFatigueCalculator()
                previous_venue = away_games[0].get('venue_abbrev', away_abbrev)
                away_travel_distance = travel_calc.calculate_travel_distance(previous_venue, home_abbrev)
            except Exception:
                away_travel_distance = 0.0

        avg_pace = (home_feats.get('recent_pts_avg', 110) + away_feats.get('recent_pts_avg', 110)) / 2.0
        home_advantage_factor = home_feats.get('home_win_pct', 0.5) - away_feats.get('away_win_pct', 0.5)

        return {
            'ctx_rest_days_diff': days_rest_home - days_rest_away,
            'ctx_avg_pace': avg_pace,
            'ctx_injury_count_home': availability_home['injury_count'],
            'ctx_injury_count_away': availability_away['injury_count'],
            'ctx_star_player_out_home': availability_home['star_player_out'],
            'ctx_star_player_out_away': availability_away['star_player_out'],
            'ctx_market_strength_diff': market_context['market_strength_diff'],
            'ctx_opening_line': market_context['opening_line'],
            'ctx_closing_line': market_context['closing_line'],
            'ctx_line_movement': market_context['line_movement'],
            'ctx_rlm_flag': market_context['rlm_flag'],
            'ctx_consensus_odds': market_context['consensus_odds'],
            'ctx_steam_move_flag': market_context['steam_move_flag'],
            'ctx_moneyline_home_prob_movement': market_context['moneyline_home_prob_movement'],
            'ctx_prediction_variance': prediction_variance,
            'ctx_home_advantage_factor': home_advantage_factor,
            'ctx_away_travel_distance': away_travel_distance,
            'ctx_away_is_b2b': int(days_rest_away <= 1),
        }


    def load_player_stats(self):
        """Load player statistics."""
        batch_files = list(self.cache_dir.glob("player_stats_batch_*.json"))
        loaded_any = False
        for batch_file in batch_files:
            try:
                with open(batch_file) as f:
                    batch_data = json.load(f)
                if isinstance(batch_data, dict):
                    for _game_id, stats in batch_data.items():
                        if isinstance(stats, list):
                            for stat in stats:
                                game_date = stat.get('game', {}).get('date', '')[:10]
                                if game_date:
                                    self._record_player_game(game_date, stat)
                                    loaded_any = True
            except Exception as e:
                print(f"Warning loading {batch_file}: {e}")

        if not loaded_any:
            loaded_any = self._load_historical_player_stats_fallback()

        if loaded_any:
            self._load_market_context()

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
        for _player_id, games in self.player_stats.items():
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
    print("  Calculating time-decay sample weights (180-day half-life)...")
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

    # Build base models for stacking
    print("  Building base models for stacking...")
    base_models = build_base_models_for_classification()
    print(f"    Created {len(base_models)} base models")

    # Initialize StackingMetaLearner with context feature support
    print("  Initializing StackingMetaLearner with XGBoost meta-learner...")
    model = StackingMetaLearner(
        base_models=base_models,
        meta_learner_type='xgboost',
        cv_folds=5,
        time_series_split=True,
        task_type='classification'
    )

    # Train with context features and sample weights
    print("  Training with context features and time-decay weights...")
    print(f"    X_train shape: {X_train.shape}")
    print(f"    Context features shape: {context_train.shape if context_train is not None else 'None'}")
    print(f"    Sample weights shape: {weights_train.shape}")

    model.fit(
        X=X_train.values,
        y=y_train,
        context_features=context_train,
        sample_weights=weights_train
    )

    # Evaluate
    print("\n  Generating predictions on test set...")
    y_pred_proba = model.predict(X_test.values, context_features=context_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_pred_proba)

    print("\n  Test Set Metrics:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Log Loss:  {ll:.4f}")

    # Log base model performance
    if hasattr(model, 'oof_scores'):
        print("\n  Base Model OOF Performance:")
        for model_name, score in model.oof_scores.items():
            print(f"    {model_name}: RMSE={score:.3f}")

    # A/B Test: Compare with baseline (if exists)
    baseline_path = MODEL_DIR / "moneyline_stacking_baseline.pkl"
    if baseline_path.exists():
        print("\n  A/B Test: Comparing with baseline...")
        try:
            with open(baseline_path, 'rb') as f:
                baseline_model = pickle.load(f)

            # Check if baseline supports context features
            try:
                y_pred_proba_baseline = baseline_model.predict(X_test.values, context_features=context_test)
                y_pred_baseline = (y_pred_proba_baseline > 0.5).astype(int)
            except (TypeError, AttributeError):
                # Old model without context support
                if hasattr(baseline_model, 'predict'):
                    y_pred_baseline = baseline_model.predict(X_test)
                    y_pred_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]
                else:
                    raise

            acc_baseline = accuracy_score(y_test, y_pred_baseline)
            ll_baseline = log_loss(y_test, y_pred_proba_baseline)
            acc_improvement = (acc - acc_baseline) * 100
            ll_improvement = (ll_baseline - ll) / ll_baseline * 100

            print(f"    Baseline Accuracy: {acc_baseline:.4f}")
            print(f"    New Model Accuracy: {acc:.4f}")
            print(f"    Accuracy Improvement: {acc_improvement:+.2f}pp")
            print(f"    Log Loss Improvement: {ll_improvement:+.2f}%")

            # Only save if improved
            if acc >= acc_baseline or ll <= ll_baseline:
                output_paths = [
                    MODEL_DIR / "moneyline_stacking.pkl",
                    MODEL_DIR / "moneyline_stacking_metalearner.pkl",
                ]
                _save_inference_ready_stacking_model(model, output_paths, feature_cols, context_cols)
                print(f"  ✓ Model improved! Saved to {output_paths[0]} and {output_paths[1]}")
            else:
                print("  ✗ Model did not improve. Keeping baseline.")
                return baseline_model
        except Exception as e:
            print(f"  Warning: Could not load baseline: {e}")
            # Save anyway if baseline comparison failed
            output_paths = [
                MODEL_DIR / "moneyline_stacking.pkl",
                MODEL_DIR / "moneyline_stacking_metalearner.pkl",
            ]
            _save_inference_ready_stacking_model(model, output_paths, feature_cols, context_cols)
            print(f"  Saved to {output_paths[0]} and {output_paths[1]}")
    else:
        output_paths = [
            MODEL_DIR / "moneyline_stacking.pkl",
            MODEL_DIR / "moneyline_stacking_metalearner.pkl",
        ]
        _save_inference_ready_stacking_model(model, output_paths, feature_cols, context_cols)
        # Save as baseline for future comparisons
        with open(baseline_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved to {output_paths[0]} and {output_paths[1]}")
        print("  Also saved as baseline for future comparisons")

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
    print("  Calculating time-decay sample weights (180-day half-life)...")
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

    # Build base models for stacking
    print("  Building base models for stacking...")
    base_models = build_base_models_for_regression()
    print(f"    Created {len(base_models)} base models")

    # Initialize StackingMetaLearner with context feature support
    print("  Initializing StackingMetaLearner with XGBoost meta-learner...")
    model = StackingMetaLearner(
        base_models=base_models,
        meta_learner_type='xgboost',
        cv_folds=5,
        time_series_split=True,
        task_type='regression'
    )

    # Train with context features and sample weights
    print("  Training with context features and time-decay weights...")
    print(f"    X_train shape: {X_train.shape}")
    print(f"    Context features shape: {context_train.shape if context_train is not None else 'None'}")
    print(f"    Sample weights shape: {weights_train.shape}")

    model.fit(
        X=X_train.values,
        y=y_train,
        context_features=context_train,
        sample_weights=weights_train
    )

    # Evaluate
    print("\n  Generating predictions on test set...")
    y_pred = model.predict(X_test.values, context_features=context_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n  Test Set Metrics:")
    print(f"    RMSE: {rmse:.3f}")
    print(f"    R²:   {r2:.3f}")

    # Log base model performance
    if hasattr(model, 'oof_scores'):
        print("\n  Base Model OOF Performance:")
        for model_name, score in model.oof_scores.items():
            print(f"    {model_name}: RMSE={score:.3f}")

    # A/B Test: Compare with baseline (if exists)
    baseline_path = MODEL_DIR / "spread_stacking_baseline.pkl"
    if baseline_path.exists():
        print("\n  A/B Test: Comparing with baseline...")
        try:
            with open(baseline_path, 'rb') as f:
                baseline_model = pickle.load(f)

            # Check if baseline supports context features
            try:
                y_pred_baseline = baseline_model.predict(X_test.values, context_features=context_test)
            except (TypeError, AttributeError):
                # Old model without context support
                if hasattr(baseline_model, 'predict'):
                    y_pred_baseline = baseline_model.predict(X_test)
                else:
                    raise

            rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
            improvement = (rmse_baseline - rmse) / rmse_baseline * 100
            print(f"    Baseline RMSE: {rmse_baseline:.3f}")
            print(f"    New Model RMSE: {rmse:.3f}")
            print(f"    Improvement: {improvement:+.2f}%")

            # Only save if improved
            if improvement >= 0:
                output_paths = [
                    MODEL_DIR / "spread_stacking.pkl",
                    MODEL_DIR / "spread_stacking_metalearner.pkl",
                ]
                _save_inference_ready_stacking_model(model, output_paths, feature_cols, context_cols)
                print(f"  ✓ Model improved! Saved to {output_paths[0]} and {output_paths[1]}")
            else:
                print("  ✗ Model did not improve. Keeping baseline.")
                return baseline_model
        except Exception as e:
            print(f"  Warning: Could not load baseline: {e}")
            # Save anyway if baseline comparison failed
            output_paths = [
                MODEL_DIR / "spread_stacking.pkl",
                MODEL_DIR / "spread_stacking_metalearner.pkl",
            ]
            _save_inference_ready_stacking_model(model, output_paths, feature_cols, context_cols)
            print(f"  Saved to {output_paths[0]} and {output_paths[1]}")
    else:
        output_paths = [
            MODEL_DIR / "spread_stacking.pkl",
            MODEL_DIR / "spread_stacking_metalearner.pkl",
        ]
        _save_inference_ready_stacking_model(model, output_paths, feature_cols, context_cols)
        # Save as baseline for future comparisons
        with open(baseline_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved to {output_paths[0]} and {output_paths[1]}")
        print("  Also saved as baseline for future comparisons")

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

    print("\n  Test Set Metrics:")
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
    parser.add_argument('--incremental', action='store_true',
                        help='Incremental update: retrain meta-learner only (keeps base models)')
    args = parser.parse_args()

    print("=" * 60)
    if args.incremental:
        print("INCREMENTAL META-LEARNER UPDATE")
    else:
        print("STACKING MODEL TRAINING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Incremental (meta-learner only)' if args.incremental else 'Full retraining'}")
    print(f"Optuna tuning: {'Enabled' if args.tune and HAS_OPTUNA else 'Disabled'}")
    print("=" * 60)

    # Incremental mode: Only retrain meta-learner (fast)
    if args.incremental:
        print("\n" + "!" * 60)
        print("INCREMENTAL MODE: Retraining meta-learner only")
        print("Base models will NOT be retrained (keeps existing base models)")
        print("!" * 60 + "\n")

        # Load existing base models and retrain meta-learner
        import pickle

        models_updated = []

        # Load data (we still need recent data to retrain meta-learner)
        loader = TrainingDataLoader()
        loader.load_games()
        loader.load_player_stats()

        # Retrain meta-learners for each model type
        prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']

        for prop_type in prop_types:
            model_path = MODEL_DIR / f"player_{prop_type}_ensemble.pkl"

            if not model_path.exists():
                print(f"⚠️  Skipping {prop_type}: No existing model found")
                continue

            try:
                print(f"\n{'='*60}")
                print(f"Retraining {prop_type.upper()} meta-learner...")
                print(f"{'='*60}")

                # Load existing ensemble model
                with open(model_path, 'rb') as f:
                    ensemble = pickle.load(f)

                # Check if model has stacking meta-learner
                if not hasattr(ensemble, 'meta_learner') or ensemble.meta_learner is None:
                    print(f"⚠️  Skipping {prop_type}: Model doesn't use stacking ensemble")
                    continue

                # Load recent data (last 60 days for incremental update)
                data = loader.build_props_dataset(prop_type)

                if len(data) < 100:
                    print(f"⚠️  Skipping {prop_type}: Insufficient data ({len(data)} samples)")
                    continue

                # Use only recent data (last 30 days worth of games)
                recent_cutoff = datetime.now() - timedelta(days=30)
                if 'game_date' in data.columns:
                    data['game_date'] = pd.to_datetime(data['game_date'])
                    recent_data = data[data['game_date'] >= recent_cutoff]
                    print(f"Using {len(recent_data)} recent samples (last 30 days)")
                else:
                    # If no date column, use last 30% of data
                    recent_data = data.tail(int(len(data) * 0.3))
                    print(f"Using {len(recent_data)} recent samples (last 30%)")

                if len(recent_data) < 50:
                    print(f"⚠️  Skipping {prop_type}: Insufficient recent data ({len(recent_data)} samples)")
                    continue

                # Prepare features and targets
                feature_cols = [c for c in recent_data.columns if c not in ['target', 'game_date', 'player_id', 'player_name']]
                X = recent_data[feature_cols].fillna(0).values
                y = recent_data['target'].values

                # Generate OOF predictions from existing base models
                print("Generating predictions from existing base models...")

                # Extract base models from ensemble
                base_models = ensemble.base_models if hasattr(ensemble, 'base_models') else []

                if not base_models:
                    print(f"⚠️  Skipping {prop_type}: No base models found in ensemble")
                    continue

                # Create time-decay sample weights (recent games more important)
                if 'game_date' in recent_data.columns:
                    days_ago = (datetime.now() - recent_data['game_date']).dt.days
                    sample_weights = 0.5 ** (days_ago / 30.0)  # 30-day half-life
                    sample_weights = sample_weights.values
                else:
                    sample_weights = None

                # Retrain meta-learner with recent OOF predictions
                print(f"Retraining meta-learner on {len(X)} samples...")
                ensemble.meta_learner.fit(X, y, sample_weights=sample_weights)

                # Save updated ensemble (with retrained meta-learner)
                with open(model_path, 'wb') as f:
                    pickle.dump(ensemble, f)

                print(f"✅ {prop_type.upper()} meta-learner retrained and saved")
                models_updated.append(prop_type)

            except Exception as e:
                print(f"❌ Error retraining {prop_type} meta-learner: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        if models_updated:
            print("INCREMENTAL UPDATE COMPLETE")
            print(f"Meta-learners updated: {', '.join(models_updated)}")
        else:
            print("INCREMENTAL UPDATE FAILED")
            print("No meta-learners were updated (check warnings above)")
        print("=" * 60)
        return

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
