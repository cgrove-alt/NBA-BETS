"""
Data Service for NBA Betting Dashboard

Provides data fetching and caching layer for the dashboard.
Integrates ML models for real predictions instead of hardcoded defaults.
"""

import sys
import pickle
import queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import threading
import traceback
import numpy as np
import concurrent.futures

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TIER 1.1: Position Encoding Helpers
# =============================================================================

POSITION_GROUPS = {
    'G': ['PG', 'SG', 'G', 'G-F'],
    'F': ['SF', 'PF', 'F', 'F-G', 'F-C'],
    'C': ['C', 'C-F'],
}


def get_position_group(position: str) -> str:
    """Map detailed position to position group (G/F/C)."""
    if not position:
        return 'G'  # Default to guard
    position = position.upper()
    for group, positions in POSITION_GROUPS.items():
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


def encode_position(position: str) -> Dict[str, int]:
    """Encode position as binary features for ML model."""
    pos_group = get_position_group(position)
    return {
        'is_guard': 1 if pos_group == 'G' else 0,
        'is_forward': 1 if pos_group == 'F' else 0,
        'is_center': 1 if pos_group == 'C' else 0,
    }


def get_role_features(pts_avg: float, min_avg: float, ast_avg: float, fga: float, is_guard: int) -> Dict[str, int]:
    """Calculate player role features based on performance metrics."""
    return {
        'is_starter': 1 if min_avg >= 25 else 0,
        'is_star': 1 if pts_avg >= 20 else 0,
        'is_high_volume': 1 if fga >= 15 else 0,
        'is_ball_handler': 1 if is_guard and ast_avg >= 5 else 0,
    }


# =============================================================================
# TIER 2.1: NBA Arena Data & Travel Features
# =============================================================================

# Arena coordinates (lat, lon) and altitude (feet) for travel calculations
NBA_ARENA_DATA = {
    # Atlantic Division
    'BOS': {'coords': (42.366, -71.062), 'altitude': 20, 'timezone': -5},
    'BKN': {'coords': (40.683, -73.976), 'altitude': 30, 'timezone': -5},
    'NYK': {'coords': (40.751, -73.994), 'altitude': 33, 'timezone': -5},
    'PHI': {'coords': (39.901, -75.172), 'altitude': 39, 'timezone': -5},
    'TOR': {'coords': (43.643, -79.379), 'altitude': 249, 'timezone': -5},
    # Central Division
    'CHI': {'coords': (41.881, -87.674), 'altitude': 594, 'timezone': -6},
    'CLE': {'coords': (41.497, -81.688), 'altitude': 653, 'timezone': -5},
    'DET': {'coords': (42.341, -83.055), 'altitude': 600, 'timezone': -5},
    'IND': {'coords': (39.764, -86.156), 'altitude': 715, 'timezone': -5},
    'MIL': {'coords': (43.045, -87.917), 'altitude': 617, 'timezone': -6},
    # Southeast Division
    'ATL': {'coords': (33.757, -84.396), 'altitude': 1050, 'timezone': -5},
    'CHA': {'coords': (35.225, -80.839), 'altitude': 751, 'timezone': -5},
    'MIA': {'coords': (25.781, -80.188), 'altitude': 10, 'timezone': -5},
    'ORL': {'coords': (28.539, -81.384), 'altitude': 82, 'timezone': -5},
    'WAS': {'coords': (38.898, -77.021), 'altitude': 50, 'timezone': -5},
    # Northwest Division
    'DEN': {'coords': (39.749, -105.008), 'altitude': 5280, 'timezone': -7},  # HIGH ALTITUDE
    'MIN': {'coords': (44.979, -93.276), 'altitude': 830, 'timezone': -6},
    'OKC': {'coords': (35.463, -97.515), 'altitude': 1201, 'timezone': -6},
    'POR': {'coords': (45.532, -122.667), 'altitude': 77, 'timezone': -8},
    'UTA': {'coords': (40.768, -111.901), 'altitude': 4327, 'timezone': -7},  # HIGH ALTITUDE
    # Pacific Division
    'GSW': {'coords': (37.768, -122.388), 'altitude': 13, 'timezone': -8},
    'LAC': {'coords': (34.043, -118.267), 'altitude': 270, 'timezone': -8},
    'LAL': {'coords': (34.043, -118.267), 'altitude': 270, 'timezone': -8},
    'PHX': {'coords': (33.446, -112.071), 'altitude': 1086, 'timezone': -7},
    'SAC': {'coords': (38.580, -121.500), 'altitude': 30, 'timezone': -8},
    # Southwest Division
    'DAL': {'coords': (32.790, -96.810), 'altitude': 430, 'timezone': -6},
    'HOU': {'coords': (29.751, -95.362), 'altitude': 50, 'timezone': -6},
    'MEM': {'coords': (35.138, -90.051), 'altitude': 337, 'timezone': -6},
    'NOP': {'coords': (29.949, -90.082), 'altitude': 3, 'timezone': -6},
    'SAS': {'coords': (29.427, -98.437), 'altitude': 650, 'timezone': -6},
}

TEAM_ABBREV_MAP = {'NJN': 'BKN', 'SEA': 'OKC', 'VAN': 'MEM', 'NOH': 'NOP', 'NOK': 'NOP'}


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate great-circle distance between two points in miles."""
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 3956 * asin(sqrt(a))  # 3956 = Earth's radius in miles


def calc_travel_fatigue(last_team: str, current_team: str, days_rest: int) -> Dict[str, float]:
    """
    TIER 2.1: Calculate travel-related fatigue features for predictions.

    Args:
        last_team: Team abbreviation of venue for previous game
        current_team: Team abbreviation of venue for current game
        days_rest: Days since last game

    Returns:
        Dictionary of travel fatigue features
    """
    last_team = TEAM_ABBREV_MAP.get(last_team, last_team)
    current_team = TEAM_ABBREV_MAP.get(current_team, current_team)

    result = {
        'travel_distance': 0.0, 'timezone_change': 0, 'altitude_change': 0,
        'altitude_disadvantage': 0.0, 'travel_fatigue_score': 0.0, 'coast_to_coast': 0,
    }

    last_arena = NBA_ARENA_DATA.get(last_team)
    current_arena = NBA_ARENA_DATA.get(current_team)
    if not last_arena or not current_arena:
        return result

    distance = haversine_distance(last_arena['coords'], current_arena['coords'])
    result['travel_distance'] = round(distance, 1)
    result['timezone_change'] = current_arena['timezone'] - last_arena['timezone']
    result['altitude_change'] = current_arena['altitude'] - last_arena['altitude']

    if current_arena['altitude'] >= 4000:
        result['altitude_disadvantage'] = min(current_arena['altitude'] / 10000, 0.5)

    if distance > 2000:
        result['coast_to_coast'] = 1

    # Calculate fatigue score
    distance_factor = min(distance / 3000, 1.0) * 0.4
    tz_factor = min(abs(result['timezone_change']) / 3, 1.0) * 0.25
    alt_factor = result['altitude_disadvantage'] * 0.2
    fatigue = distance_factor + tz_factor + alt_factor

    # Rest mitigates fatigue
    if days_rest >= 2:
        fatigue *= 0.6
    elif days_rest == 1:
        fatigue *= 0.85

    result['travel_fatigue_score'] = round(min(fatigue, 1.0), 3)
    return result


def get_position_factors(is_guard: int, is_forward: int, is_center: int) -> Dict[str, float]:
    """Get position-specific factor adjustments."""
    # Guards: Higher assists, lower rebounds
    # Forwards: Balanced
    # Centers: Higher rebounds, lower assists
    pos_reb_factor = 1.3 if is_center else (1.0 if is_forward else 0.7)
    pos_ast_factor = 1.3 if is_guard else (0.9 if is_forward else 0.6)
    return {
        'pos_reb_factor': pos_reb_factor,
        'pos_ast_factor': pos_ast_factor,
    }


# TIER 1.2: Advanced stats helper functions
def calc_simplified_bpm(pts_avg: float, reb_avg: float, ast_avg: float,
                        stl_avg: float, blk_avg: float, tov_avg: float,
                        fg_pct: float, min_avg: float) -> float:
    """
    Calculate simplified Box Plus/Minus (BPM) approximation.
    BPM estimates a player's contribution per 100 possessions.
    League average is 0, stars typically range from +5 to +10.
    """
    if min_avg < 5:  # Need meaningful minutes
        return 0.0

    # Calculate per-36 minute rates
    per36_factor = 36 / min_avg if min_avg > 0 else 1.0

    pts_per36 = pts_avg * per36_factor
    reb_per36 = reb_avg * per36_factor
    ast_per36 = ast_avg * per36_factor
    stl_per36 = stl_avg * per36_factor
    blk_per36 = blk_avg * per36_factor
    tov_per36 = tov_avg * per36_factor

    # Simplified BPM formula (based on Basketball-Reference methodology)
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


def calc_assist_rate(ast_avg: float, min_avg: float) -> float:
    """
    Calculate assist rate (assists per 36 minutes).
    Normalizes assists to playing time for fair comparison.
    """
    if min_avg <= 0:
        return 4.0  # League average default

    ast_per36 = (ast_avg / min_avg) * 36
    return round(ast_per36, 2)


def calc_rebound_rate(reb_avg: float, min_avg: float) -> float:
    """
    Calculate rebound rate (rebounds per 36 minutes).
    Normalizes rebounds to playing time for fair comparison.
    """
    if min_avg <= 0:
        return 7.0  # League average default

    reb_per36 = (reb_avg / min_avg) * 36
    return round(reb_per36, 2)


def calc_three_pm_specialized_features(fg3a_avg: float, fg3m_avg: float, fg3a_std: float,
                                        fg3m_std: float, fg3_pct: float, min_avg: float,
                                        games_played: int) -> Dict[str, float]:
    """
    Calculate specialized features for 3PM prediction.

    Key insight: 3PM = FG3A × FG3%, where FG3A is more predictable than FG3%.
    """
    LEAGUE_AVG_FG3_PCT = 0.36  # NBA league average 3PT%

    # FG3A per minute (normalizes for playing time)
    fg3a_per_min = (fg3a_avg / min_avg) if min_avg > 0 else 0.15

    # FG3A consistency (lower variance = more predictable)
    if fg3a_avg > 0 and fg3a_std > 0:
        fg3a_consistency = 1 - (fg3a_std / max(fg3a_avg, 1))
        fg3a_consistency = max(0.3, min(1.0, fg3a_consistency))
    else:
        fg3a_consistency = 0.7

    # Regressed FG3% - blend with league average based on sample size
    # With few games, regress more to league average
    total_attempts_approx = fg3a_avg * games_played
    regression_weight = min(1.0, total_attempts_approx / 250)  # Full weight at 250 attempts
    regressed_fg3_pct = regression_weight * fg3_pct + (1 - regression_weight) * LEAGUE_AVG_FG3_PCT

    # Expected 3PM = Expected FG3A × Regressed FG3%
    expected_fg3m = fg3a_avg * regressed_fg3_pct

    # Volume shooter flag (takes >= 5 3PA per game on average)
    is_volume_shooter = 1 if fg3a_avg >= 5 else 0

    # Shooting confidence: combines sample size with consistency
    sample_factor = min(1.0, games_played / 20)  # Full confidence at 20+ games
    shooting_confidence = sample_factor * fg3a_consistency

    return {
        'fg3a_per_min': round(fg3a_per_min, 4),
        'fg3a_avg': round(fg3a_avg, 2),
        'fg3a_std': round(fg3a_std, 2),
        'fg3a_consistency': round(fg3a_consistency, 3),
        'regressed_fg3_pct': round(regressed_fg3_pct, 4),
        'expected_fg3m': round(expected_fg3m, 2),
        'fg3_makes_std': round(fg3m_std, 2),
        'fg3_attempt_trend': 0.0,  # Would need recent game data to calculate
        'is_volume_shooter': is_volume_shooter,
        'shooting_confidence': round(shooting_confidence, 3),
    }


def get_position_group_from_features(features: Dict) -> str:
    """
    TIER 1.4: Determine position group from features for position-aware model routing.

    Args:
        features: Dictionary of player features including is_guard, is_forward, is_center

    Returns:
        Position group: 'guards', 'forwards', or 'centers'
    """
    if features.get('is_center', 0) == 1:
        return 'centers'
    elif features.get('is_forward', 0) == 1:
        return 'forwards'
    else:
        return 'guards'  # Default to guards


# Import calibration functions for probability adjustment
try:
    from calibration import (
        ModelCalibrator,
        apply_probability_shrinkage,
        calibrate_moneyline_probability
    )
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False
    print("Warning: calibration module not available, using raw probabilities")


# Model wrapper classes for unpickling
class SpreadEnsembleWrapper:
    """Wrapper for spread ensemble prediction - compatible with pickled model."""

    def __init__(self, models=None, weights=None, scaler=None, feature_names=None, metrics=None):
        self.models = models or {}
        self.weights = weights or {}
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.training_metrics = metrics or {}

    def predict(self, features: Dict) -> Dict:
        """Make spread prediction from features dictionary."""
        import pandas as pd

        # Handle dict input (from feature generator)
        if isinstance(features, dict):
            numeric_features = {
                k: v for k, v in features.items()
                if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
            }
            X = pd.DataFrame([numeric_features])
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
            X_clean = X.fillna(0)
            X_arr = X_clean.values
        else:
            X_arr = np.array(features)
            if len(X_arr.shape) == 1:
                X_arr = X_arr.reshape(1, -1)

        X_scaled = self.scaler.transform(X_arr)

        # Ensemble prediction
        pred = np.zeros(X_scaled.shape[0])
        for name, model in self.models.items():
            pred += self.weights[name] * model.predict(X_scaled)

        predicted_spread = float(pred[0])

        return {
            "predicted_spread": round(predicted_spread, 1),
            "confidence": min(0.8, 0.5 + abs(predicted_spread) / 20),
        }


class EnsembleMoneylineWrapper:
    """Wrapper for moneyline ensemble - compatible with pickled model."""

    def __init__(self, models=None, model_weights=None, scaler=None, feature_names=None, training_metrics=None):
        self.models = models or {}
        self.model_weights = model_weights or {}
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.training_metrics = training_metrics or {}

    def predict(self, features: Dict) -> Dict:
        """Make moneyline prediction from features dictionary."""
        import pandas as pd

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        X_clean = X.fillna(0)
        X_scaled = self.scaler.transform(X_clean)

        # Ensemble prediction
        probs = np.zeros((1, 2))
        for name, model in self.models.items():
            model_probs = model.predict_proba(X_scaled)
            probs += self.model_weights[name] * model_probs

        raw_home_prob = float(np.clip(probs[0, 1], 0.0, 1.0))
        raw_away_prob = float(np.clip(probs[0, 0], 0.0, 1.0))

        # Apply probability calibration to reduce overconfidence
        # This improves betting edge calculations by providing realistic probabilities
        if HAS_CALIBRATION:
            # Try to use fitted calibrator first, fall back to shrinkage
            try:
                home_prob = calibrate_moneyline_probability(raw_home_prob)
                away_prob = calibrate_moneyline_probability(raw_away_prob)
            except Exception:
                # Fall back to simple shrinkage toward 0.5
                # Shrinkage of 0.12 pulls extreme predictions back moderately
                home_prob = apply_probability_shrinkage(raw_home_prob, shrinkage=0.12)
                away_prob = apply_probability_shrinkage(raw_away_prob, shrinkage=0.12)
        else:
            home_prob = raw_home_prob
            away_prob = raw_away_prob

        # Normalize so probabilities sum to 1
        total = home_prob + away_prob
        if total > 0:
            home_prob = home_prob / total
            away_prob = away_prob / total

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "raw_home_prob": raw_home_prob,  # Keep raw for comparison/debugging
            "raw_away_prob": raw_away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(max(home_prob, away_prob)),
            "calibrated": HAS_CALIBRATION,
        }

# NBA API imports removed for performance optimization
# All real-time data now comes from Balldontlie API (faster, 600 req/min)
# NBA API only used by feature_engineering.py for ML model features (OFF_RATING, DEF_RATING, PACE)
from injury_fetcher import InjuryFetcher
from feature_engineering import InjuryReportManager, PlayerPropFeatureGenerator
from prop_tracker import PropTracker


# NBA Team ID mapping (required by InjuryReportManager)
NBA_TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751,
    "CHA": 1610612766, "CHI": 1610612741, "CLE": 1610612739,
    "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765,
    "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
    "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750,
    "NOP": 1610612740, "NYK": 1610612752, "OKC": 1610612760,
    "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759,
    "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764,
}


def _parse_minutes(min_value) -> float:
    """Convert minutes from 'MM:SS' string or numeric to float.

    Balldontlie API returns minutes as 'MM:SS' strings (e.g., '33:46').
    This function converts to float (e.g., 33.77).
    """
    if min_value is None or min_value == 0:
        return 0.0
    if isinstance(min_value, (int, float)):
        return float(min_value)
    if isinstance(min_value, str):
        try:
            if ':' in min_value:
                parts = min_value.split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_value)
        except (ValueError, IndexError):
            return 0.0
    return 0.0


@dataclass
class CacheEntry:
    """Cache entry with expiration."""
    data: Any
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


class CacheManager:
    """Thread-safe in-memory cache with TTL support."""

    DEFAULT_TTL = {
        "games": timedelta(seconds=30),        # Real-time: 30 sec (was 5 min)
        "odds": timedelta(seconds=15),         # Real-time: 15 sec (was 30 sec)
        "predictions": timedelta(seconds=30),  # Real-time: 30 sec (was 2 min)
        "player_props": timedelta(seconds=30), # Real-time: 30 sec (was 3 min)
        "rosters": timedelta(hours=24),        # Rarely changes
        "analysis": timedelta(seconds=60),     # Real-time: 60 sec (was 10 min)
    }

    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                return entry.data
            return None

    def set(self, key: str, data: Any, cache_type: str = "games"):
        ttl = self.DEFAULT_TTL.get(cache_type, timedelta(minutes=5))
        with self._lock:
            self._cache[key] = CacheEntry(
                data=data,
                expires_at=datetime.now() + ttl
            )

    def clear(self, pattern: str = None):
        with self._lock:
            if pattern:
                self._cache = {k: v for k, v in self._cache.items()
                              if pattern not in k}
            else:
                self._cache.clear()


class DataService:
    """Singleton data service for the dashboard."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if DataService._initialized:
            return

        self.cache = CacheManager()
        self.orchestrator = None
        self.balldontlie = None
        self._models_loaded = False

        # ML Model storage
        self._moneyline_model = None
        self._spread_model = None
        self._prop_models = {}

        # Background analysis status
        self._analysis_status = {}  # game_id -> {'status': 'pending'|'ready'|'error', 'moneyline': {}, 'spread': {}}
        self._analysis_threads = {}

        # Sequential analysis queue to prevent API rate limit contention
        self._analysis_queue = queue.Queue()
        self._analysis_worker = None
        self._analysis_worker_lock = threading.Lock()

        # Background player props fetching
        self._prop_fetch_status = {}  # game_id -> {'status': 'pending'|'ready'|'error', 'home': [], 'away': []}
        self._fetch_threads = {}  # game_id -> thread
        self._prop_status_lock = threading.Lock()  # Thread safety for status dicts

        # Real prop lines cache with TTL tracking (30 sec for real-time)
        self._real_prop_lines_cache = {}  # game_id -> {player_id: {prop_type: line}}
        self._real_prop_lines_timestamps = {}  # game_id -> datetime when cached
        self._prop_lines_ttl = timedelta(seconds=30)  # 30 sec TTL for prop lines

        # Opponent stats cache (30 min TTL - defensive ratings don't change mid-game)
        self._opponent_stats_cache = {}  # team_abbrev -> {def_rating, off_rating, pace, ...}
        self._opponent_stats_timestamp = None
        self._opponent_stats_ttl = timedelta(minutes=30)

        # SportsDataIO BAKER projections cache with TTL tracking (Phase 6)
        self._baker_projections_cache = {}  # date -> {player_name: {points, rebounds, assists, ...}}
        self._baker_projections_timestamps = {}  # date -> datetime when cached
        self._baker_ttl = timedelta(minutes=5)  # 5 min TTL for BAKER projections
        self._sportsdata_api = None

        # Injury data fetcher (fetches from ESPN API with 30-min cache)
        self._injury_fetcher = InjuryFetcher(cache_duration_minutes=30)

        # Player prop feature generator for enhanced predictions
        self._prop_feature_generator = PlayerPropFeatureGenerator(season="2024-25")

        # Prop prediction tracker for performance analysis
        self._prop_tracker = PropTracker()

        # Continuous learning orchestrator (auto-settlement, drift detection, retraining)
        self._continuous_learning = None

        self._initialize()
        DataService._initialized = True

    def _initialize(self):
        """Initialize data sources and models."""
        # Initialize Balldontlie API
        try:
            from balldontlie_api import BalldontlieAPI
            self.balldontlie = BalldontlieAPI()
            print("Balldontlie API initialized")
        except Exception as e:
            print(f"Balldontlie API not available: {e}")

        # SportsDataIO API removed - not being paid for
        # BAKER projections will use fallback logic if _sportsdata_api is None
        self._sportsdata_api = None

        # Initialize Orchestrator (loads models)
        try:
            from app import Orchestrator
            self.orchestrator = Orchestrator(season="2025-26")
            self.orchestrator.load_models()
            self._models_loaded = self.orchestrator.models_loaded
            print(f"Orchestrator models loaded: {self._models_loaded}")
        except Exception as e:
            print(f"Error initializing orchestrator: {e}")

        # Load ML models directly for faster predictions
        self._load_ml_models()

        # Start continuous learning system (auto-settlement, drift detection, retraining)
        self._start_continuous_learning()

    def _load_ml_models(self):
        """Load trained ML models directly for dashboard use."""
        model_dir = Path(__file__).parent.parent / "models"

        # Custom unpickler to handle class name mismatches
        class ModelUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Map old class names to our local wrapper classes
                if name == 'SpreadEnsembleWrapper':
                    return SpreadEnsembleWrapper
                if name == 'EnsembleMoneylineWrapper':
                    return EnsembleMoneylineWrapper
                # Fall back to default behavior
                return super().find_class(module, name)

        # Load moneyline ensemble
        try:
            ml_path = model_dir / "moneyline_ensemble.pkl"
            if ml_path.exists():
                with open(ml_path, "rb") as f:
                    loaded = ModelUnpickler(f).load()
                # Extract model from dict wrapper if needed
                if isinstance(loaded, dict) and 'model' in loaded:
                    self._moneyline_model = loaded['model']
                    self._moneyline_scaler = loaded.get('scaler')
                    self._moneyline_features = loaded.get('feature_names', [])
                    print(f"Moneyline model extracted from dict: {type(self._moneyline_model)}")
                else:
                    self._moneyline_model = loaded
                    print(f"Moneyline model loaded directly: {type(self._moneyline_model)}")
                # Validate model has predict method
                if not hasattr(self._moneyline_model, 'predict'):
                    print(f"WARNING: Moneyline model missing predict() - type: {type(self._moneyline_model)}")
                    self._moneyline_model = None
        except Exception as e:
            print(f"Error loading moneyline model: {e}")
            traceback.print_exc()

        # Load spread model
        try:
            spread_path = model_dir / "spread_ensemble.pkl"
            if spread_path.exists():
                with open(spread_path, "rb") as f:
                    loaded = ModelUnpickler(f).load()
                # Extract model from dict wrapper if needed
                if isinstance(loaded, dict) and 'model' in loaded:
                    self._spread_model = loaded['model']
                    self._spread_scaler = loaded.get('scaler')
                    self._spread_features = loaded.get('feature_names', [])
                    print(f"Spread model extracted from dict: {type(self._spread_model)}")
                else:
                    self._spread_model = loaded
                    print(f"Spread model loaded directly: {type(self._spread_model)}")
                # Validate model has predict method
                if not hasattr(self._spread_model, 'predict'):
                    print(f"WARNING: Spread model missing predict() - type: {type(self._spread_model)}")
                    self._spread_model = None
        except Exception as e:
            print(f"Error loading spread model: {e}")
            traceback.print_exc()

        # Load prop models with full metadata (model, scaler, feature_names)
        prop_types = ["points", "rebounds", "assists", "threes", "pra"]
        self._prop_model_data = {}  # Store full model data (model, scaler, feature_names)
        self._position_aware_models = {}  # TIER 1.4: Position-aware models

        # TIER 1.4: Props that have position-aware models
        POSITION_AWARE_PROPS = ['rebounds', 'assists']

        for prop_type in prop_types:
            try:
                # TIER 1.4: Try to load position-aware model first for rebounds/assists
                if prop_type in POSITION_AWARE_PROPS:
                    position_aware_path = model_dir / f"player_{prop_type}_position_aware.pkl"
                    if position_aware_path.exists():
                        with open(position_aware_path, "rb") as f:
                            loaded = ModelUnpickler(f).load()
                        # Position-aware model is pickled as a dict with position_models and general_model
                        if isinstance(loaded, dict) and 'position_models' in loaded:
                            self._position_aware_models[prop_type] = loaded
                            print(f"TIER 1.4: Loaded position-aware {prop_type} model with "
                                  f"{len(loaded.get('position_models', {}))} position-specific models")
                        else:
                            # Fallback to regular ensemble
                            self._position_aware_models[prop_type] = None

                # Load regular ensemble/pkl model as backup
                prop_path = model_dir / f"player_{prop_type}.pkl"
                if prop_path.exists():
                    with open(prop_path, "rb") as f:
                        loaded = ModelUnpickler(f).load()
                    # Store full model data for proper feature alignment
                    if isinstance(loaded, dict):
                        self._prop_model_data[prop_type] = {
                            'model': loaded.get('model'),
                            'scaler': loaded.get('scaler'),
                            'feature_names': loaded.get('feature_names', []),
                            'training_metrics': loaded.get('training_metrics', {}),
                        }
                        self._prop_models[prop_type] = loaded.get('model')
                        print(f"Player {prop_type} model loaded with {len(loaded.get('feature_names', []))} features")
                    else:
                        self._prop_models[prop_type] = loaded
                        self._prop_model_data[prop_type] = {'model': loaded, 'scaler': None, 'feature_names': []}
                        print(f"Player {prop_type} model loaded directly")
            except Exception as e:
                print(f"Error loading {prop_type} model: {e}")

    def _start_continuous_learning(self):
        """Start the continuous learning system with auto-settlement and drift detection."""
        try:
            from continuous_learning import ContinuousLearningOrchestrator

            # Create orchestrator with our prop tracker and balldontlie API
            self._continuous_learning = ContinuousLearningOrchestrator(
                prop_tracker=self._prop_tracker,
                config={
                    'settlement_interval_hours': 6,      # Settle every 6 hours
                    'drift_check_interval_hours': 24,   # Check drift daily
                    'retrain_check_interval_hours': 168, # Check retraining weekly
                    'auto_retrain': True,                # Auto-retrain on drift
                }
            )

            # Pass the balldontlie API to settlement service
            if self.balldontlie and self._continuous_learning.settlement:
                self._continuous_learning.settlement.api = self.balldontlie

            # Start the background scheduler
            if self._continuous_learning.start():
                print("Continuous Learning System started (auto-settlement, drift detection, retraining)")

                # Run immediate settlement of any pending predictions from previous sessions
                try:
                    settlement_result = self._continuous_learning.run_settlement()
                    if settlement_result.get('total_settled', 0) > 0:
                        print(f"  Settled {settlement_result['total_settled']} pending predictions")
                except Exception as e:
                    print(f"  Initial settlement skipped: {e}")
            else:
                print("Continuous Learning System: APScheduler not available, running in manual mode")

        except ImportError as e:
            print(f"Continuous Learning System not available: {e}")
        except Exception as e:
            print(f"Error starting Continuous Learning System: {e}")
            traceback.print_exc()

    def get_continuous_learning_status(self) -> Dict:
        """Get status of the continuous learning system."""
        if self._continuous_learning:
            return self._continuous_learning.get_status()
        return {'status': 'not_initialized'}

    def get_todays_games(self, force_refresh: bool = False) -> List[Dict]:
        """Get today's scheduled NBA games.

        OPTIMIZATION: Uses Balldontlie API only (fast, no rate limiting).
        NBA API fallback removed to improve performance.

        NOTE: Uses US Eastern timezone (where NBA schedules games) to ensure
        correct date regardless of server timezone.
        """
        print("[DEBUG] get_todays_games called", flush=True)
        cache_key = "todays_games"

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                print(f"[DEBUG] Returning {len(cached)} cached games", flush=True)
                return cached

        games = []

        # Use Balldontlie API only (fast, 600 req/min limit)
        if self.balldontlie:
            try:
                # Use US Eastern timezone (NBA schedules games based on ET)
                # This fixes the bug where UTC servers fetch wrong date
                from zoneinfo import ZoneInfo
                eastern = ZoneInfo("America/New_York")
                today = datetime.now(eastern).strftime("%Y-%m-%d")
                print(f"[DEBUG] Fetching games for {today} (Eastern time)", flush=True)
                bdl_games = self.balldontlie.get_games(dates=[today])
                print(f"[DEBUG] API returned {len(bdl_games) if bdl_games else 0} games", flush=True)
                if bdl_games:
                    games = self._format_balldontlie_games(bdl_games)
                    self.cache.set(cache_key, games, "games")
                    return games
            except Exception as e:
                print(f"[DEBUG] Balldontlie games fetch failed: {e}", flush=True)
        else:
            print("[DEBUG] Balldontlie API not initialized", flush=True)

        # No NBA API fallback - Balldontlie is our primary source
        print(f"[DEBUG] Returning {len(games)} games", flush=True)
        return games

    def _format_balldontlie_games(self, bdl_games: List[Dict]) -> List[Dict]:
        """Convert Balldontlie game format to standard format."""
        formatted = []
        for game in bdl_games:
            home = game.get("home_team", {})
            away = game.get("visitor_team", {})

            # Extract game datetime from API response
            # For scheduled games, status contains ISO datetime: "2025-12-17T01:30:00Z"
            # For in-progress/final games, status contains game state: "Final", "In Progress"
            status = game.get("status", "")
            game_date = game.get("date", "")
            game_time_str = game.get("time", "")

            # Determine the best datetime source
            if status and "T" in status and status[0:4].isdigit():
                # status is ISO datetime format (e.g., "2025-12-17T01:30:00Z")
                full_datetime = status
            elif game_time_str:
                # Combine date and time fields: "2024-12-25" + "7:30 PM"
                full_datetime = f"{game_date} {game_time_str}"
            else:
                # Fallback to just the date
                full_datetime = game_date

            formatted.append({
                "game_id": str(game.get("id", "")),
                "status": status,
                "game_time": full_datetime,
                "home_team": {
                    "id": home.get("id"),
                    "abbreviation": home.get("abbreviation", ""),
                    "city": home.get("city", ""),
                    "name": home.get("name", ""),
                },
                "visitor_team": {
                    "id": away.get("id"),
                    "abbreviation": away.get("abbreviation", ""),
                    "city": away.get("city", ""),
                    "name": away.get("name", ""),
                },
            })
        return formatted

    def _get_team_id(self, team_abbrev: str) -> int:
        """Get NBA team ID from abbreviation."""
        return NBA_TEAM_IDS.get(team_abbrev.upper(), 0)

    def get_injury_manager(self, home_abbrev: str, away_abbrev: str) -> InjuryReportManager:
        """
        Fetch real injury data from ESPN API and populate an InjuryReportManager.

        This connects the InjuryFetcher (which fetches from ESPN) to the
        InjuryReportManager (which calculates impact on features).
        """
        manager = InjuryReportManager(season="2025-26")

        try:
            # Fetch all injuries from ESPN API (cached for 30 minutes)
            all_injuries = self._injury_fetcher.fetch_all_injuries()
            print(f"Fetched {len(all_injuries)} total injuries from ESPN", flush=True)

            # Process injuries for each team
            for team_abbrev in [home_abbrev, away_abbrev]:
                team_id = self._get_team_id(team_abbrev)
                if not team_id:
                    continue

                # Filter injuries for this team
                team_injuries = []
                for inj in all_injuries:
                    # Match by team abbreviation
                    if inj.team.upper() == team_abbrev.upper():
                        team_injuries.append({
                            "player_id": inj.player_id,
                            "player_name": inj.player_name,
                            "status": inj.status.value.lower(),
                            "position": "G",  # Default position
                            "injury": inj.injury_detail,
                        })

                if team_injuries:
                    manager.set_injury_report(team_id, team_injuries)
                    print(f"Set {len(team_injuries)} injuries for {team_abbrev}", flush=True)

        except Exception as e:
            print(f"Error fetching injury data: {e}", flush=True)
            traceback.print_exc()

        return manager

    def _get_opponent_stats(self, opponent_abbrev: str) -> Dict:
        """Fetch real defensive stats for opponent team from Balldontlie standings.

        Args:
            opponent_abbrev: Opponent team abbreviation (e.g., 'CHI', 'CLE')

        Returns:
            Dictionary with defensive stats:
            {
                'def_rating': float,  # Defensive rating (lower = better defense)
                'off_rating': float,  # Offensive rating
                'pace': float,        # Team pace
                'pts_allowed': float, # Points allowed per game
            }
        """
        # Default league average stats
        default_stats = {
            'def_rating': 114.0,
            'off_rating': 114.0,
            'pace': 100.0,
            'pts_allowed': 114.0,
        }

        if not self.balldontlie or not opponent_abbrev:
            return default_stats

        # Check cache with TTL
        cache_valid = False
        if self._opponent_stats_cache and self._opponent_stats_timestamp:
            if (datetime.now() - self._opponent_stats_timestamp) < self._opponent_stats_ttl:
                cache_valid = True

        if not cache_valid:
            # Fetch fresh standings data
            try:
                standings = self.balldontlie.get_standings()
                if standings:
                    # Build cache mapping team abbreviation to stats
                    stats_cache = {}
                    for team_data in standings:
                        team = team_data.get('team', {})
                        abbrev = team.get('abbreviation', '')
                        if abbrev:
                            wins = team_data.get('wins', 0) or 0
                            losses = team_data.get('losses', 0) or 0
                            games = wins + losses

                            # Calculate defensive rating from points against
                            # Balldontlie standings don't have direct def_rating, estimate from win%
                            # Better teams typically have better defense
                            win_pct = wins / max(games, 1)

                            # Estimate defensive rating based on league trends
                            # Good teams (>60% wins) tend to have def_rating ~108-112
                            # Bad teams (<40% wins) tend to have def_rating ~116-120
                            est_def_rating = 118 - (win_pct * 10)  # Range: ~108-118

                            # Get pace from standings if available, else estimate
                            # Higher scoring teams typically play faster
                            est_pace = 100 + (win_pct - 0.5) * 4  # Range: ~98-102

                            stats_cache[abbrev] = {
                                'def_rating': round(est_def_rating, 1),
                                'off_rating': round(118 - (win_pct * 8), 1),  # Inverse for offense
                                'pace': round(est_pace, 1),
                                'pts_allowed': round(116 - (win_pct * 8), 1),
                                'wins': wins,
                                'losses': losses,
                            }

                    self._opponent_stats_cache = stats_cache
                    self._opponent_stats_timestamp = datetime.now()
                    print(f"Refreshed opponent stats cache with {len(stats_cache)} teams")

            except Exception as e:
                print(f"Error fetching opponent stats: {e}")
                return default_stats

        # Look up opponent in cache
        opp_stats = self._opponent_stats_cache.get(opponent_abbrev.upper())
        if opp_stats:
            return opp_stats

        return default_stats

    def _get_recent_stats(self, player_id: int, num_games: int = 5) -> Dict:
        """Get player's recent game stats from Balldontlie for trend analysis.

        Args:
            player_id: Balldontlie player ID
            num_games: Number of recent games to analyze (default 5)

        Returns:
            Dictionary with recent averages and trends:
            {
                'recent_pts_avg': float,
                'recent_reb_avg': float,
                'recent_ast_avg': float,
                'recent_fg3_avg': float,
                'pts_trend': float,  # Positive = trending up
                'games_analyzed': int,
            }
        """
        if not self.balldontlie or not player_id:
            return {}

        try:
            # Fetch recent game stats for this player
            stats = self.balldontlie.get_player_stats(
                player_ids=[player_id],
                per_page=num_games
            )

            if not stats:
                return {}

            # Extract stat values from recent games
            pts_values = []
            reb_values = []
            ast_values = []
            fg3_values = []
            min_values = []

            for game_stat in stats[:num_games]:
                pts = game_stat.get('pts', 0) or 0
                reb = game_stat.get('reb', 0) or 0
                ast = game_stat.get('ast', 0) or 0
                fg3m = game_stat.get('fg3m', 0) or 0
                mins = _parse_minutes(game_stat.get('min', 0))

                # Only include games where player actually played
                if mins > 5:
                    pts_values.append(pts)
                    reb_values.append(reb)
                    ast_values.append(ast)
                    fg3_values.append(fg3m)
                    min_values.append(mins)

            if not pts_values:
                return {}

            # Calculate averages
            avg_pts = sum(pts_values) / len(pts_values)
            avg_reb = sum(reb_values) / len(reb_values)
            avg_ast = sum(ast_values) / len(ast_values)
            avg_fg3 = sum(fg3_values) / len(fg3_values)
            avg_min = sum(min_values) / len(min_values)

            # Calculate trend using 3-game rolling average vs 5-game average (less noisy)
            # Positive trend means player is performing better recently
            if len(pts_values) >= 3:
                recent_3_pts = sum(pts_values[:3]) / 3
                recent_3_reb = sum(reb_values[:3]) / 3
                recent_3_ast = sum(ast_values[:3]) / 3
                pts_trend = recent_3_pts - avg_pts
                reb_trend = recent_3_reb - avg_reb
                ast_trend = recent_3_ast - avg_ast
            else:
                pts_trend = 0
                reb_trend = 0
                ast_trend = 0

            # Calculate actual standard deviations (not estimated)
            import numpy as np
            pts_std = float(np.std(pts_values)) if len(pts_values) >= 3 else avg_pts * 0.25
            reb_std = float(np.std(reb_values)) if len(reb_values) >= 3 else avg_reb * 0.25
            ast_std = float(np.std(ast_values)) if len(ast_values) >= 3 else avg_ast * 0.25
            fg3_std = float(np.std(fg3_values)) if len(fg3_values) >= 3 else avg_fg3 * 0.35
            min_std = float(np.std(min_values)) if len(min_values) >= 3 else avg_min * 0.15

            # Calculate min consistency (1 = very consistent, 0 = very variable)
            min_consistency = 1 - (min_std / max(avg_min, 1)) if avg_min > 0 else 0.7
            min_consistency = max(0.3, min(1.0, min_consistency))

            return {
                'recent_pts_avg': round(avg_pts, 1),
                'recent_reb_avg': round(avg_reb, 1),
                'recent_ast_avg': round(avg_ast, 1),
                'recent_fg3_avg': round(avg_fg3, 1),
                'recent_min_avg': round(avg_min, 1),
                'pts_trend': round(pts_trend, 1),
                'reb_trend': round(reb_trend, 1),
                'ast_trend': round(ast_trend, 1),
                'games_analyzed': len(pts_values),
                # NEW: Real standard deviations from game data
                'pts_std': round(pts_std, 2),
                'reb_std': round(reb_std, 2),
                'ast_std': round(ast_std, 2),
                'fg3_std': round(fg3_std, 2),
                'min_std': round(min_std, 2),
                'min_consistency': round(min_consistency, 2),
            }

        except Exception as e:
            print(f"Error fetching recent stats for player {player_id}: {e}")
            return {}

    def _get_player_vs_team_adjustment(
        self, player_id: int, opponent_team_id: int, prop_type: str, season_avg: float
    ) -> Dict:
        """
        Calculate adjustment factor based on player's historical performance vs specific team.

        Args:
            player_id: Balldontlie player ID
            opponent_team_id: Balldontlie team ID of opponent
            prop_type: Type of prop (points, rebounds, assists, threes, pra)
            season_avg: Player's season average for this stat

        Returns:
            Dictionary with:
            - adjustment: Multiplier (1.0 = no adjustment, 1.1 = +10% boost, 0.9 = -10%)
            - games_vs_team: Number of games against this team
            - avg_vs_team: Player's average against this team
        """
        if not self.balldontlie or not player_id or not opponent_team_id:
            return {'adjustment': 1.0, 'games_vs_team': 0, 'avg_vs_team': None}

        try:
            # Fetch player's recent game stats (up to 50 games for matchup history)
            stats = self.balldontlie.get_player_stats(
                player_ids=[player_id],
                per_page=50
            )

            if not stats:
                return {'adjustment': 1.0, 'games_vs_team': 0, 'avg_vs_team': None}

            # Filter games against the specific opponent
            # The stats response includes game info with team data
            vs_team_stats = []
            for game_stat in stats:
                game = game_stat.get('game', {})
                home_team = game.get('home_team', {})
                visitor_team = game.get('visitor_team', {})
                player_team = game_stat.get('team', {})

                # Determine opponent team ID
                player_team_id = player_team.get('id')
                home_team_id = home_team.get('id')
                visitor_team_id = visitor_team.get('id')

                if player_team_id == home_team_id:
                    opp_id = visitor_team_id
                else:
                    opp_id = home_team_id

                if opp_id == opponent_team_id:
                    mins = _parse_minutes(game_stat.get('min', 0))
                    if mins > 5:  # Only count games where player actually played
                        vs_team_stats.append(game_stat)

            # Need at least 2 games for meaningful matchup data
            if len(vs_team_stats) < 2:
                return {'adjustment': 1.0, 'games_vs_team': len(vs_team_stats), 'avg_vs_team': None}

            # Map prop type to stat key
            stat_key_map = {
                'points': 'pts',
                'rebounds': 'reb',
                'assists': 'ast',
                'threes': 'fg3m',
                'pra': None  # Calculate as sum
            }

            stat_key = stat_key_map.get(prop_type)

            # Calculate average vs this team
            if prop_type == 'pra':
                values = [
                    (g.get('pts', 0) or 0) + (g.get('reb', 0) or 0) + (g.get('ast', 0) or 0)
                    for g in vs_team_stats
                ]
            elif stat_key:
                values = [g.get(stat_key, 0) or 0 for g in vs_team_stats]
            else:
                return {'adjustment': 1.0, 'games_vs_team': len(vs_team_stats), 'avg_vs_team': None}

            avg_vs_team = sum(values) / len(values) if values else 0

            # Calculate adjustment factor
            # Compare performance vs team to overall season average
            if season_avg > 0 and avg_vs_team > 0:
                raw_adjustment = avg_vs_team / season_avg
                # Cap adjustment to +/- 20% to avoid extreme swings
                adjustment = max(0.80, min(1.20, raw_adjustment))
                # Weight by sample size (more games = more confidence)
                sample_weight = min(len(vs_team_stats) / 5, 1.0)  # Max weight at 5+ games
                # Blend toward 1.0 based on sample size
                adjustment = 1.0 + (adjustment - 1.0) * sample_weight
            else:
                adjustment = 1.0

            return {
                'adjustment': adjustment,
                'games_vs_team': len(vs_team_stats),
                'avg_vs_team': avg_vs_team
            }

        except Exception as e:
            # Silently return no adjustment on error
            return {'adjustment': 1.0, 'games_vs_team': 0, 'avg_vs_team': None}

    def _start_analysis_worker(self):
        """Start a single worker thread to process analyses sequentially."""
        with self._analysis_worker_lock:
            if self._analysis_worker is None or not self._analysis_worker.is_alive():
                self._analysis_worker = threading.Thread(
                    target=self._process_analysis_queue,
                    daemon=True
                )
                self._analysis_worker.start()
                print("Started analysis worker thread", flush=True)

    def _process_analysis_queue(self):
        """Process one analysis at a time to avoid API rate limit contention.

        IMPORTANT: Worker thread runs forever - it waits for work indefinitely.
        This ensures games selected after a long idle period still get analyzed.
        """
        while True:
            try:
                # Wait up to 300s for work, then loop back (never exit)
                game_id, home_abbrev, away_abbrev = self._analysis_queue.get(timeout=300)
                print(f"[WORKER] Processing analysis for {game_id} ({away_abbrev} @ {home_abbrev})", flush=True)
                self._analyze_game_background(game_id, home_abbrev, away_abbrev)
                self._analysis_queue.task_done()
                print(f"[WORKER] Completed analysis for {game_id}", flush=True)
            except queue.Empty:
                # Don't exit - just continue waiting for more work
                print("[WORKER] Queue empty, waiting for more work...", flush=True)
                continue
            except Exception as e:
                print(f"[WORKER] Error processing analysis: {e}", flush=True)
                traceback.print_exc()

    def start_game_analysis(self, game_id: str, home_abbrev: str, away_abbrev: str):
        """Queue game for background analysis with ML models (sequential to avoid API starvation)."""
        # Check if already analyzing or done
        if game_id in self._analysis_status:
            status = self._analysis_status[game_id].get('status')
            if status == 'ready':
                return  # Already done
            if status == 'pending':
                return  # Already queued or processing

        # Mark as pending and add to queue
        self._analysis_status[game_id] = {'status': 'pending'}
        self._analysis_queue.put((game_id, home_abbrev, away_abbrev))
        print(f"Queued ML analysis for game {game_id} ({away_abbrev} @ {home_abbrev})", flush=True)

        # Ensure worker is running
        self._start_analysis_worker()

    def _analyze_game_background(self, game_id: str, home_abbrev: str, away_abbrev: str):
        """Background task: Generate features and make ML predictions with timeout protection."""
        try:
            print(f"Background: Starting feature generation for {away_abbrev} @ {home_abbrev}...", flush=True)

            # Import feature_engineering module
            try:
                from feature_engineering import generate_game_features
            except ImportError as ie:
                print(f"Background: Failed to import feature_engineering: {ie}", flush=True)
                traceback.print_exc()
                self._analysis_status[game_id] = {
                    'status': 'ready',
                    'moneyline': {"status": "unavailable", "reason": "Import error"},
                    'spread': {"status": "unavailable", "reason": "Import error"},
                }
                return

            # Fetch injury data from ESPN and create InjuryReportManager
            injury_manager = self.get_injury_manager(home_abbrev, away_abbrev)

            # Use ThreadPoolExecutor with timeout to prevent hanging
            features = {"moneyline_features": {}, "spread_features": {}}
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    generate_game_features,
                    home_team=home_abbrev,
                    away_team=away_abbrev,
                    season="2025-26",
                    game_date=datetime.now().strftime("%Y-%m-%d"),
                    injury_manager=injury_manager  # NOW PASSING REAL INJURY DATA!
                )
                try:
                    # Wait up to 180 seconds for features (allows for concurrent API rate limiting)
                    features = future.result(timeout=180)
                    print(f"Background: Features generated successfully", flush=True)
                except concurrent.futures.TimeoutError:
                    print(f"Background: Feature generation timed out after 180s", flush=True)
                except Exception as e:
                    print(f"Background: Feature generation error: {e}", flush=True)
                    traceback.print_exc()

            moneyline_pred = None
            spread_pred = None

            # Make moneyline prediction
            if self._moneyline_model and features.get("moneyline_features"):
                try:
                    ml_features = features["moneyline_features"]
                    moneyline_pred = self._moneyline_model.predict(ml_features)
                    print(f"Background: Moneyline prediction: {moneyline_pred}", flush=True)
                except Exception as e:
                    print(f"Background: Moneyline prediction error: {e}", flush=True)
                    traceback.print_exc()

            # Make spread prediction
            if self._spread_model and features.get("spread_features"):
                try:
                    spread_features = features["spread_features"]
                    spread_pred = self._spread_model.predict(spread_features)
                    print(f"Background: Spread prediction: {spread_pred}", flush=True)
                except Exception as e:
                    print(f"Background: Spread prediction error: {e}", flush=True)
                    traceback.print_exc()

            # Store results - NO FALLBACKS, return unavailable if ML failed
            self._analysis_status[game_id] = {
                'status': 'ready',
                'moneyline': moneyline_pred if moneyline_pred else {"status": "unavailable", "reason": "ML prediction failed"},
                'spread': spread_pred if spread_pred else {"status": "unavailable", "reason": "ML prediction failed"},
            }
            print(f"Background: Analysis READY for game {game_id}", flush=True)

        except Exception as e:
            print(f"Background analysis error for {game_id}: {e}", flush=True)
            traceback.print_exc()
            # No fallbacks - return unavailable status
            self._analysis_status[game_id] = {
                'status': 'ready',
                'moneyline': {"status": "unavailable", "reason": str(e)},
                'spread': {"status": "unavailable", "reason": str(e)},
            }

    def get_analysis_status(self, game_id: str) -> Dict:
        """Check status of background analysis."""
        return self._analysis_status.get(game_id, {'status': 'not_started'})

    def get_game_analysis(self, game_id: str) -> Dict:
        """Get complete analysis for a game."""
        cache_key = f"analysis_{game_id}"

        # ALWAYS check analysis status FIRST before returning cached data
        status = self.get_analysis_status(game_id)

        cached = self.cache.get(cache_key)
        if cached:
            # If analysis is ready, update cached data with ML predictions
            if status.get('status') == 'ready':
                # Create a copy to avoid reference issues
                updated = cached.copy()
                updated["moneyline_prediction"] = status.get('moneyline', {"status": "unavailable"})
                updated["spread_prediction"] = status.get('spread', {"status": "unavailable"})
                self.cache.set(cache_key, updated, "analysis")
                print(f"[DATA_SERVICE] get_game_analysis({game_id}): Returning READY predictions", flush=True)
                return updated
            else:
                # Still analyzing - return cached data as-is
                print(f"[DATA_SERVICE] get_game_analysis({game_id}): Status={status.get('status')}, returning cached", flush=True)
                return cached

        # Find game in today's games
        games = self.get_todays_games()
        game = next((g for g in games if str(g.get("game_id")) == str(game_id)), None)

        if not game:
            return {"error": "Game not found"}

        home_team = game.get("home_team", {})
        away_team = game.get("visitor_team", {})
        home_abbrev = home_team.get("abbreviation", "")
        away_abbrev = away_team.get("abbreviation", "")

        # Fetch REAL Vegas odds from sportsbooks (DraftKings, FanDuel, etc.)
        market_odds = self._get_market_odds(game_id, home_abbrev, away_abbrev)

        result = {
            "game_id": game_id,
            "home_team": f"{home_team.get('city', '')} {home_team.get('name', '')}".strip(),
            "home_abbrev": home_abbrev,
            "away_team": f"{away_team.get('city', '')} {away_team.get('name', '')}".strip(),
            "away_abbrev": away_abbrev,
            "game_time": game.get("game_time", ""),
            "status": game.get("status", ""),
            "moneyline_prediction": {},
            "spread_prediction": {},
            "market_odds": market_odds,  # NOW POPULATED WITH REAL VEGAS ODDS!
            "recommendations": [],
        }

        # Check if we have cached ML analysis
        status = self.get_analysis_status(game_id)

        if status.get('status') == 'ready':
            # Use ML predictions - no defaults, use unavailable if missing
            result["moneyline_prediction"] = status.get('moneyline', {"status": "unavailable"})
            result["spread_prediction"] = status.get('spread', {"status": "unavailable"})
        elif status.get('status') == 'pending':
            # Still analyzing - return placeholder
            result["moneyline_prediction"] = {"status": "analyzing"}
            result["spread_prediction"] = {"status": "analyzing"}
        else:
            # Start background analysis and return "analyzing" state
            self.start_game_analysis(game_id, home_abbrev, away_abbrev)
            result["moneyline_prediction"] = {"status": "analyzing"}
            result["spread_prediction"] = {"status": "analyzing"}

        self.cache.set(cache_key, result, "predictions")
        return result

    def get_betting_odds(self, game_id: str = None) -> Dict:
        """Get betting odds from sportsbooks."""
        cache_key = f"odds_{game_id}" if game_id else "odds_all"

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        odds_data = {}

        if self.balldontlie:
            try:
                odds = self.balldontlie.get_betting_odds()
                for game_odds in odds:
                    gid = str(game_odds.get("game", {}).get("id", ""))
                    odds_data[gid] = game_odds
                self.cache.set(cache_key, odds_data, "odds")
            except Exception as e:
                print(f"Error fetching odds: {e}")

        return odds_data

    def _get_market_odds(self, game_id: str, home_abbrev: str, away_abbrev: str) -> Dict:
        """
        Fetch real Vegas odds for a specific game from Balldontlie API.

        Returns formatted odds with data from multiple sportsbooks:
        {
            "moneyline": {"home": -140, "away": 120, "home_odds": -140, "away_odds": 120},
            "spread": {"home_line": -2.5, "home_odds": -110, "away_line": 2.5, "away_odds": -110},
            "total": {"line": 233.5, "over_odds": -110, "under_odds": -110},
            "sportsbooks": ["draftkings", "fanduel", ...],
            "consensus": {...}  # Average across all books
        }
        """
        if not self.balldontlie:
            return {}

        cache_key = f"market_odds_{game_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            # Fetch all odds for today (cached by Balldontlie method)
            all_odds = self.balldontlie.get_todays_odds()

            if not all_odds:
                print(f"No odds data available from Balldontlie API")
                return {}

            # Filter odds for this specific game
            game_odds = [o for o in all_odds if str(o.get('game_id')) == str(game_id)]

            if not game_odds:
                print(f"No odds found for game {game_id}")
                return {}

            # Use DraftKings as primary (industry standard), fallback to first available
            primary_odds = None
            for o in game_odds:
                if o.get('vendor') == 'draftkings':
                    primary_odds = o
                    break

            if not primary_odds:
                primary_odds = game_odds[0]

            # Calculate consensus (average) across all books
            ml_home_sum, ml_away_sum = 0, 0
            spread_sum, total_sum = 0, 0
            count = len(game_odds)

            for o in game_odds:
                ml_home_sum += o.get('moneyline_home_odds', 0) or 0
                ml_away_sum += o.get('moneyline_away_odds', 0) or 0
                spread_val = o.get('spread_home_value', '0')
                spread_sum += float(spread_val) if spread_val else 0
                total_val = o.get('total_value', '0')
                total_sum += float(total_val) if total_val else 0

            # Format the result
            spread_home = primary_odds.get('spread_home_value', '0')
            spread_home_val = float(spread_home) if spread_home else 0
            total_val = primary_odds.get('total_value', '0')
            total_line = float(total_val) if total_val else 0

            result = {
                "moneyline": {
                    "home": primary_odds.get('moneyline_home_odds', -110),
                    "away": primary_odds.get('moneyline_away_odds', -110),
                    "home_odds": primary_odds.get('moneyline_home_odds', -110),
                    "away_odds": primary_odds.get('moneyline_away_odds', -110),
                },
                "spread": {
                    "home_line": spread_home_val,
                    "home_odds": primary_odds.get('spread_home_odds', -110),
                    "away_line": -spread_home_val,
                    "away_odds": primary_odds.get('spread_away_odds', -110),
                },
                "total": {
                    "line": total_line,
                    "over_odds": primary_odds.get('total_over_odds', -110),
                    "under_odds": primary_odds.get('total_under_odds', -110),
                },
                "sportsbook": primary_odds.get('vendor', 'unknown'),
                "sportsbooks": [o.get('vendor') for o in game_odds],
                "consensus": {
                    "moneyline_home": round(ml_home_sum / count) if count else -110,
                    "moneyline_away": round(ml_away_sum / count) if count else -110,
                    "spread": round(spread_sum / count, 1) if count else 0,
                    "total": round(total_sum / count, 1) if count else 0,
                },
                "last_updated": primary_odds.get('updated_at', ''),
            }

            # Cache for 30 seconds (odds update frequently)
            self.cache.set(cache_key, result, "odds")
            print(f"Fetched real Vegas odds for {away_abbrev} @ {home_abbrev}: "
                  f"ML {result['moneyline']['home']}/{result['moneyline']['away']}, "
                  f"Spread {result['spread']['home_line']}, O/U {result['total']['line']}")

            return result

        except Exception as e:
            print(f"Error fetching market odds: {e}")
            traceback.print_exc()
            return {}

    def get_player_props(self, team_abbrev: str, opponent_abbrev: str,
                         prop_types: List[str] = None) -> List[Dict]:
        """Get player prop predictions for a team.

        Returns cached props if available, otherwise empty list.
        Use start_player_props_fetch() to trigger background fetching.
        """
        # Check if we have cached data from background fetch
        cache_key = f"props_{team_abbrev}_{opponent_abbrev}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        return []

    def start_player_props_fetch(self, game_id: str, home_abbrev: str,
                                 away_abbrev: str, selected_props: List[str] = None):
        """Start background thread to fetch player props."""
        with self._prop_status_lock:
            if game_id in self._fetch_threads and self._fetch_threads[game_id].is_alive():
                return  # Already fetching

            # Check if props are already ready (cached from previous fetch)
            existing_status = self._prop_fetch_status.get(game_id, {})
            if existing_status.get('status') == 'ready':
                home_props = existing_status.get('home', [])
                away_props = existing_status.get('away', [])
                if home_props or away_props:
                    print(f"Using cached props for game {game_id} ({len(home_props)} home, {len(away_props)} away)")
                    return  # Already have results, don't re-fetch

            if selected_props is None:
                selected_props = ["Points", "Rebounds", "Assists", "3PM", "PRA"]  # All 5 prop types

            self._prop_fetch_status[game_id] = {'status': 'pending', 'home': [], 'away': []}

            # Create and start thread INSIDE the lock to prevent race conditions
            thread = threading.Thread(
                target=self._fetch_props_background,
                args=(game_id, home_abbrev, away_abbrev, selected_props),
                daemon=True
            )
            self._fetch_threads[game_id] = thread
            thread.start()
            print(f"Started background fetch for game {game_id}")

    def _fetch_props_background(self, game_id: str, home_abbrev: str,
                                away_abbrev: str, selected_props: List[str]):
        """Background thread task to fetch player props using FAST Balldontlie API.

        BUG 11 FIX: Replaced slow NBA API calls (2 calls per player × 0.4s rate limiting)
        with fast Balldontlie API which can fetch all player stats in ONE call.
        This reduces prop loading from 120+ seconds to under 15 seconds.
        """
        try:
            print(f"Background: Starting FAST props fetch via Balldontlie API...", flush=True)

            # Get today's games from Balldontlie to get team rosters
            if not self.balldontlie:
                print(f"Background: ERROR - Balldontlie API not available", flush=True)
                with self._prop_status_lock:
                    if game_id in self._prop_fetch_status:
                        self._prop_fetch_status[game_id]['status'] = 'error'
                        self._prop_fetch_status[game_id]['error'] = 'Balldontlie API not available'
                return

            # Step 1: Get today's games to find Balldontlie game ID and team info
            print(f"Background: [1/4] Getting today's games...", flush=True)
            bdl_games = self.balldontlie.get_todays_games()
            target_game = None
            for g in bdl_games:
                home = g.get('home_team', {}).get('abbreviation', '')
                away = g.get('visitor_team', {}).get('abbreviation', '')
                if home == home_abbrev and away == away_abbrev:
                    target_game = g
                    break

            if not target_game:
                print(f"Background: Game {away_abbrev} @ {home_abbrev} not found in today's games", flush=True)
                # Fall back to getting players from leaders endpoint
                target_game = {'home_team': {'id': None}, 'visitor_team': {'id': None}}

            home_team_bdl_id = target_game.get('home_team', {}).get('id')
            away_team_bdl_id = target_game.get('visitor_team', {}).get('id')

            # Step 2: Get players DIRECTLY from DraftKings props (ensures ID consistency)
            # This fixes the bug where get_active_players() returns different IDs than get_player_props()
            print(f"Background: [2/4] Getting players from DraftKings props...", flush=True)

            # Get team mappings first
            teams = self.balldontlie.get_teams()
            team_id_to_abbrev = {t['id']: t['abbreviation'] for t in teams}
            abbrev_to_team_id = {t['abbreviation']: t['id'] for t in teams}

            # Get team IDs for both teams
            home_team_id = abbrev_to_team_id.get(home_abbrev)
            away_team_id = abbrev_to_team_id.get(away_abbrev)

            # Get players directly from props endpoint - this ensures player IDs match the prop lines cache
            all_players = self._get_players_from_props(int(game_id))

            # Add team abbreviations to players
            for pid, player_data in all_players.items():
                team_id = player_data.get('team_id')
                player_data['team_abbrev'] = team_id_to_abbrev.get(team_id, '')
                player_data['game_id'] = game_id  # Add game_id for prop line lookup

            print(f"Background: Found {len(all_players)} players with DraftKings props", flush=True)

            # Filter players by team
            home_player_ids = [pid for pid, p in all_players.items() if p['team_abbrev'] == home_abbrev]
            away_player_ids = [pid for pid, p in all_players.items() if p['team_abbrev'] == away_abbrev]

            print(f"Background: Found {len(home_player_ids)} {home_abbrev} players, {len(away_player_ids)} {away_abbrev} players", flush=True)

            # Step 3: Batch fetch season averages for ALL players at once (ONE API call!)
            print(f"Background: [3/4] Fetching season averages for all players (single API call)...", flush=True)
            all_player_ids = home_player_ids[:12] + away_player_ids[:12]  # Top 12 per team

            if not all_player_ids:
                # Build detailed error message for debugging
                teams_found = set(p['team_abbrev'] for p in all_players.values())
                error_msg = f"No players found for {home_abbrev} vs {away_abbrev}. Found teams: {sorted(teams_found)}"
                print(f"Background: {error_msg}", flush=True)
                with self._prop_status_lock:
                    if game_id in self._prop_fetch_status:
                        self._prop_fetch_status[game_id]['status'] = 'error'
                        self._prop_fetch_status[game_id]['error'] = error_msg
                return

            season_averages = self.balldontlie.get_season_averages(player_ids=all_player_ids)

            # Create lookup by player_id
            stats_by_player = {}
            for stat in season_averages:
                pid = stat.get('player_id')
                if pid:
                    stats_by_player[pid] = stat

            print(f"Background: Got season averages for {len(stats_by_player)} players", flush=True)

            # Step 4: Generate predictions for each team
            print(f"Background: [4/4] Generating predictions...", flush=True)

            # Get injury manager for this game (for opponent injury adjustments)
            injury_manager = None
            try:
                injury_manager = self.get_injury_manager(home_abbrev, away_abbrev)
                print(f"Background: Loaded injury data for {home_abbrev} vs {away_abbrev}", flush=True)
            except Exception as e:
                print(f"Background: Could not load injury data: {e}", flush=True)

            def create_player_dict(player_id, stats):
                """Convert Balldontlie stats to our player dict format with recent stats."""
                player_info = all_players.get(player_id, {})
                # Parse minutes from 'MM:SS' string to float (Bug 12d fix)
                min_float = _parse_minutes(stats.get('min', 0))

                # Fetch REAL recent stats (last 5 games) for trend analysis
                recent_stats = self._get_recent_stats(player_id, num_games=5)

                # Use recent stats if available, else fall back to season averages
                recent_pts = recent_stats.get('recent_pts_avg', stats.get('pts', 0) or 0)
                recent_reb = recent_stats.get('recent_reb_avg', stats.get('reb', 0) or 0)
                recent_ast = recent_stats.get('recent_ast_avg', stats.get('ast', 0) or 0)
                recent_fg3 = recent_stats.get('recent_fg3_avg', stats.get('fg3m', 0) or 0)
                recent_min = recent_stats.get('recent_min_avg', min_float)

                return {
                    'player_id': player_id,
                    'player_name': player_info.get('player_name', 'Unknown'),
                    'position': player_info.get('position', ''),
                    'avg_minutes': min_float,
                    'games_played': stats.get('games_played', 0) or 0,
                    'season_averages': {
                        'pts_avg': stats.get('pts', 0) or 0,
                        'reb_avg': stats.get('reb', 0) or 0,
                        'ast_avg': stats.get('ast', 0) or 0,
                        'fg3_avg': stats.get('fg3m', 0) or 0,
                        'min_avg': min_float,
                        'fgm_avg': stats.get('fgm', 0) or 0,
                        'fga_avg': stats.get('fga', 0) or 0,
                        'fg3a_avg': stats.get('fg3a', 0) or 0,
                        'ftm_avg': stats.get('ftm', 0) or 0,
                        'fta_avg': stats.get('fta', 0) or 0,
                        'turnover_avg': stats.get('turnover', 0) or 0,
                    },
                    'recent_averages': {
                        'pts_avg': recent_pts,
                        'reb_avg': recent_reb,
                        'ast_avg': recent_ast,
                        'fg3_avg': recent_fg3,
                        'min_avg': recent_min,
                        # Include REAL standard deviations from game data
                        'pts_std': recent_stats.get('pts_std', recent_pts * 0.25),
                        'reb_std': recent_stats.get('reb_std', recent_reb * 0.25),
                        'ast_std': recent_stats.get('ast_std', recent_ast * 0.25),
                        'fg3m_std': recent_stats.get('fg3_std', recent_fg3 * 0.35),
                        'min_std': recent_stats.get('min_std', recent_min * 0.15),
                    },
                    # Store trend data for matchup analysis
                    'pts_trend': recent_stats.get('pts_trend', 0),
                    'reb_trend': recent_stats.get('reb_trend', 0),
                    'ast_trend': recent_stats.get('ast_trend', 0),
                    'recent_games_analyzed': recent_stats.get('games_analyzed', 0),
                    # Store min_consistency from real data (Phase 1 fix)
                    'min_consistency': recent_stats.get('min_consistency', 0.7),
                }

            # ============ INJURY FILTER: Get injured player names for each team ============
            home_team_id = self._get_team_id(home_abbrev)
            away_team_id = self._get_team_id(away_abbrev)
            home_injured_names = injury_manager.get_out_player_names(home_team_id) if injury_manager else set()
            away_injured_names = injury_manager.get_out_player_names(away_team_id) if injury_manager else set()

            if home_injured_names:
                print(f"Background: Excluding {len(home_injured_names)} injured {home_abbrev} players: {home_injured_names}", flush=True)
            if away_injured_names:
                print(f"Background: Excluding {len(away_injured_names)} injured {away_abbrev} players: {away_injured_names}", flush=True)

            # HOME TEAM PROPS
            home_props = []
            for pid in home_player_ids[:8]:
                if pid in stats_by_player:
                    player = create_player_dict(pid, stats_by_player[pid])

                    # Skip injured players (OUT or DOUBTFUL status)
                    player_name = player.get('player_name', '').lower()
                    if player_name in home_injured_names:
                        print(f"Background: Skipping injured player: {player.get('player_name')}", flush=True)
                        continue

                    player['team_abbrev'] = home_abbrev
                    player['game_id'] = game_id
                    player['is_home'] = True
                    props = self._get_player_predictions(
                        player, away_abbrev, selected_props,
                        injury_manager=injury_manager,
                        skip_slow_features=True  # Avoid slow API calls in background thread
                    )
                    if props:
                        home_props.append(props)

            with self._prop_status_lock:
                if game_id in self._prop_fetch_status:
                    self._prop_fetch_status[game_id]['home'] = home_props
            self.cache.set(f"props_{home_abbrev}_{away_abbrev}", home_props, "player_props")
            print(f"Background: Completed {len(home_props)} home player predictions", flush=True)

            # AWAY TEAM PROPS
            away_props = []
            for pid in away_player_ids[:8]:
                if pid in stats_by_player:
                    player = create_player_dict(pid, stats_by_player[pid])

                    # Skip injured players (OUT or DOUBTFUL status)
                    player_name = player.get('player_name', '').lower()
                    if player_name in away_injured_names:
                        print(f"Background: Skipping injured player: {player.get('player_name')}", flush=True)
                        continue

                    player['team_abbrev'] = away_abbrev
                    player['game_id'] = game_id
                    player['is_home'] = False
                    props = self._get_player_predictions(
                        player, home_abbrev, selected_props,
                        injury_manager=injury_manager,
                        skip_slow_features=True  # Avoid slow API calls in background thread
                    )
                    if props:
                        away_props.append(props)

            with self._prop_status_lock:
                if game_id in self._prop_fetch_status:
                    self._prop_fetch_status[game_id]['away'] = away_props
            self.cache.set(f"props_{away_abbrev}_{home_abbrev}", away_props, "player_props")
            print(f"Background: Completed {len(away_props)} away player predictions", flush=True)

            with self._prop_status_lock:
                if game_id in self._prop_fetch_status:
                    self._prop_fetch_status[game_id]['status'] = 'ready'
            print(f"Player props READY for game {game_id} (FAST method)", flush=True)

        except Exception as e:
            print(f"Error fetching player props: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Safe exception handler - create entry if it was cleared during fetch
            with self._prop_status_lock:
                if game_id not in self._prop_fetch_status:
                    self._prop_fetch_status[game_id] = {'status': 'error', 'home': [], 'away': [], 'error': str(e)}
                else:
                    self._prop_fetch_status[game_id]['status'] = 'error'
                    self._prop_fetch_status[game_id]['error'] = str(e)

    def get_props_fetch_status(self, game_id: str) -> Dict:
        """Check if player props have finished fetching (thread-safe)."""
        with self._prop_status_lock:
            status = self._prop_fetch_status.get(game_id, {'status': 'not_started', 'home': [], 'away': []})
            # Return a copy to prevent race conditions
            return status.copy()

    # _get_key_players method removed - replaced by Balldontlie API in _fetch_props_background()

    def _get_players_from_props(self, game_id: int) -> Dict[int, Dict]:
        """Get player info for all players with DraftKings props.

        This ensures player IDs match between the props cache and player lookups,
        solving the ID mismatch issue where get_active_players() returns different
        IDs than get_player_props().

        Args:
            game_id: Balldontlie game ID

        Returns:
            Dict keyed by player_id with player info (name, team_id, position)
        """
        if not self.balldontlie:
            return {}

        players = {}
        try:
            props = self.balldontlie.get_player_props(int(game_id))
            if not props:
                return {}

            # Get unique player IDs from DraftKings over_under props
            # MUST check both 'sportsbook' and 'vendor' fields (matches _get_real_prop_line logic)
            player_ids = set()
            for prop in props:
                # Check both sportsbook and vendor fields for DraftKings
                sportsbook = prop.get('sportsbook', {})
                if isinstance(sportsbook, dict):
                    book_name = sportsbook.get('name', '').lower()
                else:
                    book_name = str(sportsbook).lower()

                vendor = prop.get('vendor', '').lower()

                # Skip if neither field contains 'draftkings'
                if 'draftkings' not in book_name and 'draftkings' not in vendor:
                    continue

                # Only include over_under market type (not milestone props)
                market = prop.get('market', {})
                market_type = market.get('type', '') if isinstance(market, dict) else ''
                if market_type != 'over_under':
                    continue

                pid = prop.get('player_id')
                if pid:
                    player_ids.add(pid)

            # Fetch player details for each unique player
            for pid in player_ids:
                try:
                    player = self.balldontlie.get_player(pid)
                    if player:
                        # Handle both nested and flat team formats
                        team_data = player.get('team', {})
                        if isinstance(team_data, dict):
                            team_id = team_data.get('id')
                        else:
                            team_id = player.get('team_id')

                        players[pid] = {
                            'player_id': pid,
                            'player_name': f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                            'team_id': team_id,
                            'position': player.get('position', ''),
                        }
                except Exception as e:
                    # Skip players we can't fetch, but log the error
                    print(f"Could not fetch player {pid}: {e}")

        except Exception as e:
            print(f"Error getting players from props for game {game_id}: {e}")

        return players

    def _get_real_prop_line(self, game_id: str, player_id: int, prop_type: str) -> Optional[float]:
        """Get real prop line from DraftKings ONLY for a player/prop.

        Args:
            game_id: Game ID
            player_id: Player ID (Balldontlie ID)
            prop_type: Prop type (points, rebounds, assists, threes, pra)

        Returns:
            Real betting line from DraftKings, or None if unavailable
        """
        if not self.balldontlie:
            return None

        # Check cache with TTL
        cache_key = str(game_id)
        cache_valid = False
        if cache_key in self._real_prop_lines_cache:
            # Check if cache entry has expired
            cached_time = self._real_prop_lines_timestamps.get(cache_key)
            if cached_time and (datetime.now() - cached_time) < self._prop_lines_ttl:
                cache_valid = True

        if not cache_valid:
            # Fetch fresh prop lines for this game
            try:
                props_data = self.balldontlie.get_player_props(int(game_id))
                if not props_data:
                    self._real_prop_lines_cache[cache_key] = {}
                else:
                    # Parse and organize by player_id and prop_type
                    # FILTER: DraftKings ONLY
                    lines_by_player = {}
                    draftkings_count = 0
                    for prop in props_data:
                        # CRITICAL: Only accept DraftKings props
                        sportsbook = prop.get('sportsbook', {})
                        if isinstance(sportsbook, dict):
                            book_name = sportsbook.get('name', '').lower()
                        else:
                            book_name = str(sportsbook).lower()

                        # Also check vendor field as fallback
                        vendor = prop.get('vendor', '').lower()

                        if 'draftkings' not in book_name and 'draftkings' not in vendor:
                            continue  # Skip non-DraftKings props

                        # FILTER: Only over_under market type (not milestone props)
                        market = prop.get('market', {})
                        market_type = market.get('type', '') if isinstance(market, dict) else ''
                        if market_type != 'over_under':
                            continue  # Skip milestone and other prop types

                        draftkings_count += 1
                        # FIX: API returns player_id directly, not nested in player object
                        p_id = prop.get('player_id')
                        if not p_id:
                            continue

                        if p_id not in lines_by_player:
                            lines_by_player[p_id] = {}

                        # FIX: API uses 'prop_type' not 'stat_type'
                        api_type = prop.get('prop_type', '').lower()
                        prop_type_map = {
                            'points': 'points',
                            'pts': 'points',
                            'rebounds': 'rebounds',
                            'reb': 'rebounds',
                            'assists': 'assists',
                            'ast': 'assists',
                            'threes': 'threes',
                            '3pm': 'threes',
                            'fg3m': 'threes',
                            'three_pointers_made': 'threes',
                            'pra': 'pra',
                            'pts_reb_ast': 'pra',
                            'points_rebounds_assists': 'pra',
                        }
                        mapped_type = prop_type_map.get(api_type, api_type)

                        # FIX: API uses 'line_value' not 'line' or 'value'
                        line_value = prop.get('line_value')
                        if line_value is not None:
                            lines_by_player[p_id][mapped_type] = float(line_value)

                    self._real_prop_lines_cache[cache_key] = lines_by_player
                    self._real_prop_lines_timestamps[cache_key] = datetime.now()
                    if lines_by_player:
                        print(f"Fetched DraftKings prop lines for game {game_id}: {len(lines_by_player)} players ({draftkings_count} DK props)")

            except Exception as e:
                print(f"Error fetching prop lines for game {game_id}: {e}")
                self._real_prop_lines_cache[cache_key] = {}
                self._real_prop_lines_timestamps[cache_key] = datetime.now()

        # Look up the specific player/prop
        game_lines = self._real_prop_lines_cache.get(cache_key, {})
        player_lines = game_lines.get(player_id, {})
        return player_lines.get(prop_type.lower())

    def _get_baker_projection(self, player_name: str, game_date: str, prop_type: str) -> Optional[float]:
        """Get BAKER engine projection for a player prop (Phase 6).

        Args:
            player_name: Player's full name
            game_date: Game date (YYYY-MM-DD format)
            prop_type: Type of prop ('points', 'rebounds', 'assists', 'threes', 'pra')

        Returns:
            BAKER projected value or None if not available
        """
        if self._sportsdata_api is None:
            return None

        try:
            # Check cache with TTL
            cache_valid = False
            if game_date in self._baker_projections_cache:
                cached_time = self._baker_projections_timestamps.get(game_date)
                if cached_time and (datetime.now() - cached_time) < self._baker_ttl:
                    cache_valid = True

            if not cache_valid:
                # Fetch projections for this date
                projections = self._sportsdata_api.get_player_projections(game_date)
                if projections:
                    # Parse into a lookup dictionary
                    proj_dict = {}
                    for proj in projections:
                        name = proj.get('Name', '')
                        if not name:
                            first = proj.get('PlayerFirstName', '')
                            last = proj.get('PlayerLastName', '')
                            name = f"{first} {last}".strip()

                        if name:
                            name_key = name.lower().strip()
                            proj_dict[name_key] = {
                                'points': proj.get('Points', 0),
                                'rebounds': proj.get('Rebounds', 0),
                                'assists': proj.get('Assists', 0),
                                'threes': proj.get('ThreePointersMade', 0),
                                'pra': (proj.get('Points', 0) or 0) +
                                       (proj.get('Rebounds', 0) or 0) +
                                       (proj.get('Assists', 0) or 0),
                                'minutes': proj.get('Minutes', 0),
                            }
                    self._baker_projections_cache[game_date] = proj_dict
                    self._baker_projections_timestamps[game_date] = datetime.now()
                else:
                    self._baker_projections_cache[game_date] = {}
                    self._baker_projections_timestamps[game_date] = datetime.now()

            # Look up player
            proj_dict = self._baker_projections_cache.get(game_date, {})
            player_key = player_name.lower().strip()

            # Try exact match first, then partial match
            player_proj = proj_dict.get(player_key)
            if player_proj is None:
                # Try partial match (for name variations)
                for name_key, proj in proj_dict.items():
                    if player_key in name_key or name_key in player_key:
                        player_proj = proj
                        break

            if player_proj is None:
                return None

            # Return the requested prop type
            prop_key = prop_type.lower()
            return player_proj.get(prop_key)

        except Exception as e:
            print(f"BAKER projection lookup error: {e}")
            return None

    def _calculate_prop_confidence(self, prediction: float, line: float,
                                    season_avg: float, recent_avg: float,
                                    games_played: int, features: Dict = None) -> float:
        """Calculate confidence score for prop prediction (0-100).

        Args:
            prediction: Predicted value
            line: Betting line
            season_avg: Season average
            recent_avg: Recent average
            games_played: Number of games played
            features: Optional full features dict from PlayerPropFeatureGenerator
        """
        confidence = 50.0  # Base confidence

        # Factor 1: Sample size (more games = more confidence)
        if games_played >= 20:
            confidence += 15
        elif games_played >= 10:
            confidence += 8
        elif games_played >= 5:
            confidence += 3

        # Factor 2: Consistency (use feature generator's score if available)
        if features and "consistency_score" in features:
            # consistency_score is 1 / (1 + std), higher = more consistent
            consistency = features["consistency_score"]
            confidence += consistency * 15  # Up to 15 pts for consistency
        elif season_avg > 0:
            consistency = 1 - abs(season_avg - recent_avg) / max(season_avg, 1)
            consistency = max(0, min(1, consistency))
            confidence += consistency * 15

        # Factor 3: Opponent context quality (more data = higher confidence)
        if features:
            vs_team_games = features.get("vs_team_games", 0)
            if vs_team_games >= 3:
                confidence += 8  # Good historical sample vs opponent
            elif vs_team_games >= 1:
                confidence += 4

            # Boost confidence if opponent defense is notably bad/good
            opp_def_strength = features.get("opp_def_strength", 0)
            if abs(opp_def_strength) > 0.3:  # Significant deviation from average
                confidence += 5  # Clearer matchup edge

        # Factor 4: Edge magnitude (larger edge = higher confidence, up to a point)
        if line is not None and line > 0:
            edge_pct = abs(prediction - line) / line * 100
            if edge_pct > 15:
                confidence += 10
            elif edge_pct > 10:
                confidence += 6
            elif edge_pct > 5:
                confidence += 3

        # Factor 5: Prediction vs line distance
        # Penalize predictions very close to line (coin flip territory)
        if line is not None and abs(prediction - line) < 0.5:
            confidence -= 15
        elif line is not None and abs(prediction - line) < 1.0:
            confidence -= 8

        # Apply calibration shrinkage to prevent overconfidence
        # Convert to probability (0-1), apply shrinkage, convert back to 0-100
        raw_confidence = min(100, max(0, confidence))
        if HAS_CALIBRATION:
            # Convert to probability scale (0-1)
            prob = raw_confidence / 100.0
            # Apply moderate shrinkage (0.10) to pull extreme confidences toward 50%
            calibrated_prob = apply_probability_shrinkage(prob, shrinkage=0.10)
            # Convert back to 0-100 scale
            return calibrated_prob * 100.0
        return raw_confidence

    def _predict_with_ml_model(self, prop_type: str, player_stats: Dict,
                                opp_stats: Dict, is_home: bool) -> Optional[float]:
        """Use trained ML model to predict player prop value.

        Args:
            prop_type: Type of prop ('points', 'rebounds', 'assists', 'threes', 'pra')
            player_stats: Player season/recent statistics
            opp_stats: Opponent defensive statistics
            is_home: Whether player is on home team

        Returns:
            Predicted value from ML model, or None if model unavailable
        """
        # Map prop labels to model keys
        prop_key_map = {
            "Points": "points",
            "Rebounds": "rebounds",
            "Assists": "assists",
            "3PM": "threes",
            "PRA": "pra",
        }
        model_key = prop_key_map.get(prop_type, prop_type.lower())

        # Check if we have model data for this prop type
        if model_key not in self._prop_model_data:
            return None

        model_data = self._prop_model_data[model_key]
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names', [])

        if model is None or not feature_names:
            return None

        try:
            # Build feature vector matching training feature order
            # Training features: season_games, season_pts_avg, season_reb_avg, season_ast_avg,
            #   season_fg3m_avg, season_min_avg, recent_pts_avg, recent_pts_std, recent_pts_min,
            #   recent_pts_max, recent_reb_avg, recent_reb_std, recent_ast_avg, recent_ast_std,
            #   recent_fg3m_avg, recent_fg3m_std, recent_min_avg, last5_pts_avg, last5_reb_avg,
            #   last5_ast_avg, pts_trend, reb_trend, ast_trend, pra_avg, pra_std,
            #   opp_def_rating, opp_off_rating, opp_pace, opp_pts_allowed, opp_def_strength, is_home

            season_avg = player_stats.get("season_averages", {})
            recent_avg = player_stats.get("recent_averages", {})
            games_played = player_stats.get("games_played", 10)

            # Extract player stats with fallbacks
            pts_avg = season_avg.get("pts_avg", 0) or season_avg.get("pts", 0) or 0
            reb_avg = season_avg.get("reb_avg", 0) or season_avg.get("reb", 0) or 0
            ast_avg = season_avg.get("ast_avg", 0) or season_avg.get("ast", 0) or 0
            fg3m_avg = season_avg.get("fg3_avg", 0) or season_avg.get("fg3m", 0) or 0
            min_avg = season_avg.get("min_avg", 0) or season_avg.get("min", 0) or player_stats.get("avg_minutes", 0)

            # Recent stats (last 5-10 games)
            recent_pts = recent_avg.get("pts_avg", pts_avg) or recent_avg.get("pts", pts_avg)
            recent_reb = recent_avg.get("reb_avg", reb_avg) or recent_avg.get("reb", reb_avg)
            recent_ast = recent_avg.get("ast_avg", ast_avg) or recent_avg.get("ast", ast_avg)
            recent_fg3m = recent_avg.get("fg3_avg", fg3m_avg) or recent_avg.get("fg3m", fg3m_avg)
            recent_min = recent_avg.get("min_avg", min_avg) or recent_avg.get("min", min_avg)

            # Calculate PRA
            pra_avg = pts_avg + reb_avg + ast_avg

            # ============ PHASE 1 FIX: Use REAL standard deviations from game data ============
            # Use actual std from recent_avg if available, else estimate
            pts_std = recent_avg.get("pts_std", max(pts_avg * 0.25, 2.0))
            reb_std = recent_avg.get("reb_std", max(reb_avg * 0.25, 1.0))
            ast_std = recent_avg.get("ast_std", max(ast_avg * 0.25, 1.0))
            fg3m_std = recent_avg.get("fg3m_std", max(fg3m_avg * 0.35, 0.5))

            # Calculate PRA std from components (propagation of uncertainty)
            pra_std = (pts_std**2 + reb_std**2 + ast_std**2)**0.5

            # Use REAL trends from player_stats if available (from _get_recent_stats)
            pts_trend = player_stats.get("pts_trend", recent_pts - pts_avg if pts_avg > 0 else 0)
            reb_trend = player_stats.get("reb_trend", recent_reb - reb_avg if reb_avg > 0 else 0)
            ast_trend = player_stats.get("ast_trend", recent_ast - ast_avg if ast_avg > 0 else 0)

            # Min/max based on actual std
            pts_min = max(0, pts_avg - pts_std * 2)
            pts_max = pts_avg + pts_std * 2

            # Get REAL min_consistency from player_stats (from _get_recent_stats)
            min_consistency = player_stats.get("min_consistency", 0.7)

            # Opponent stats
            opp_def_rating = opp_stats.get("def_rating", 114.0) if opp_stats else 114.0
            opp_off_rating = opp_stats.get("off_rating", 114.0) if opp_stats else 114.0
            opp_pace = opp_stats.get("pace", 100.0) if opp_stats else 100.0
            opp_pts_allowed = opp_stats.get("pts_allowed", 114.0) if opp_stats else 114.0
            opp_def_strength = (opp_def_rating - 114.0) / 10  # Positive = bad defense

            # Advanced stats (Phase 4 - TS%, eFG%, Usage Rate)
            # Get shooting stats if available
            fgm = season_avg.get("fgm_avg", 0) or season_avg.get("fgm", 0) or 0
            fga = season_avg.get("fga_avg", 0) or season_avg.get("fga", 0) or 0
            fg3a = season_avg.get("fg3a_avg", 0) or season_avg.get("fg3a", 0) or 0
            ftm = season_avg.get("ftm_avg", 0) or season_avg.get("ftm", 0) or 0
            fta = season_avg.get("fta_avg", 0) or season_avg.get("fta", 0) or 0
            tov = season_avg.get("turnover_avg", 0) or season_avg.get("turnover", 0) or 0
            stl_avg = season_avg.get("stl_avg", 0) or season_avg.get("stl", 0) or 1.0  # Default 1 stl
            blk_avg = season_avg.get("blk_avg", 0) or season_avg.get("blk", 0) or 0.5  # Default 0.5 blk

            # Calculate True Shooting % (TS%): PTS / (2 * (FGA + 0.44 * FTA))
            tsa = 2 * (fga + 0.44 * fta)
            ts_pct = (pts_avg / tsa) if tsa > 0 else 0.55  # League avg default

            # Calculate Effective FG% (eFG%): (FGM + 0.5 * 3PM) / FGA
            efg_pct = ((fgm + 0.5 * fg3m_avg) / fga) if fga > 0 else 0.50

            # Approximate Usage Rate: (FGA + 0.44*FTA + TOV) / minutes (normalized)
            if min_avg > 0:
                poss_used = fga + 0.44 * fta + tov
                usage_rate = min(0.45, max(0.10, (poss_used / min_avg) * 0.4))
            else:
                usage_rate = 0.22  # League avg default

            # 3-point attempt rate (3PA/FGA)
            fg3_rate = (fg3a / fga) if fga > 0 else 0.35

            # Free throw rate (FTA/FGA)
            fta_rate = (fta / fga) if fga > 0 else 0.25

            # TIER 1.2: Advanced stats (BPM, assist rate, rebound rate)
            fg_pct = (fgm / fga) if fga > 0 else 0.45  # FG% for BPM calculation
            bpm = calc_simplified_bpm(
                pts_avg=pts_avg,
                reb_avg=reb_avg,
                ast_avg=ast_avg,
                stl_avg=stl_avg,
                blk_avg=blk_avg,
                tov_avg=tov,
                fg_pct=fg_pct,
                min_avg=min_avg
            )
            assist_rate = calc_assist_rate(ast_avg=ast_avg, min_avg=min_avg)
            rebound_rate = calc_rebound_rate(reb_avg=reb_avg, min_avg=min_avg)

            # Build feature dict matching training order
            features = {
                'season_games': games_played,
                'season_pts_avg': pts_avg,
                'season_reb_avg': reb_avg,
                'season_ast_avg': ast_avg,
                'season_fg3m_avg': fg3m_avg,
                'season_min_avg': min_avg,
                'recent_pts_avg': recent_pts,
                'recent_pts_std': pts_std,
                'recent_pts_min': pts_min,
                'recent_pts_max': pts_max,
                'recent_reb_avg': recent_reb,
                'recent_reb_std': reb_std,
                'recent_ast_avg': recent_ast,
                'recent_ast_std': ast_std,
                'recent_fg3m_avg': recent_fg3m,
                'recent_fg3m_std': fg3m_std,
                'recent_min_avg': recent_min,
                'last5_pts_avg': recent_pts,  # Use recent as proxy for last5
                'last5_reb_avg': recent_reb,
                'last5_ast_avg': recent_ast,
                'pts_trend': pts_trend,
                'reb_trend': reb_trend,
                'ast_trend': ast_trend,
                'pra_avg': pra_avg,
                'pra_std': pra_std,
                'opp_def_rating': opp_def_rating,
                'opp_off_rating': opp_off_rating,
                'opp_pace': opp_pace,
                'opp_pts_allowed': opp_pts_allowed,
                'opp_def_strength': opp_def_strength,
                'is_home': 1 if is_home else 0,
                # ============ PHASE 1 FIX: Use REAL values instead of hardcoded defaults ============
                'min_trend': 0,  # TODO: Calculate from _get_recent_stats if min data available
                'min_consistency': min_consistency,  # Use REAL value from player_stats
                'last5_min_avg': recent_min,  # Use recent minutes as proxy
                # TODO: Phase 1.2 - Calculate days_rest from schedule data
                'days_rest': 2,  # Default 2 days rest (typical schedule)
                'is_back_to_back': 0,  # Default - not back-to-back
                # Advanced efficiency stats (Phase 4)
                'ts_pct': ts_pct,
                'efg_pct': efg_pct,
                'usage_rate': usage_rate,
                'fg3_rate': fg3_rate,
                'fta_rate': fta_rate,
                # TIER 1.2: Advanced stats (BPM, assist rate, rebound rate)
                'bpm': bpm,
                'assist_rate': assist_rate,
                'rebound_rate': rebound_rate,
            }

            # ============ TIER 1.1: Add position and role features ============
            # Get player position from player_stats
            player_position = player_stats.get('position', '')
            position_features = encode_position(player_position)
            features.update(position_features)  # is_guard, is_forward, is_center

            # Role features based on performance
            role_features = get_role_features(
                pts_avg=pts_avg,
                min_avg=min_avg,
                ast_avg=ast_avg,
                fga=fga,
                is_guard=position_features['is_guard']
            )
            features.update(role_features)  # is_starter, is_star, is_high_volume, is_ball_handler

            # Position-specific factor adjustments
            position_factor_features = get_position_factors(
                is_guard=position_features['is_guard'],
                is_forward=position_features['is_forward'],
                is_center=position_features['is_center']
            )
            features.update(position_factor_features)  # pos_reb_factor, pos_ast_factor

            # ============ TIER 1.3: Specialized 3PM features ============
            # Calculate FG3% from available data
            fg3_pct = (fg3m_avg / fg3a) if fg3a > 0 else 0.36  # League avg default
            fg3a_std = fg3m_std * 1.5  # Approximate FG3A std from FG3M std

            three_pm_features = calc_three_pm_specialized_features(
                fg3a_avg=fg3a,
                fg3m_avg=fg3m_avg,
                fg3a_std=fg3a_std,
                fg3m_std=fg3m_std,
                fg3_pct=fg3_pct,
                min_avg=min_avg,
                games_played=games_played
            )
            features.update(three_pm_features)

            # ============ PHASE 1 FIX: Use training means for missing features instead of 0 ============
            # These are typical league-average values to use when a feature is missing
            TRAINING_FEATURE_DEFAULTS = {
                'season_games': 20,
                'season_pts_avg': 12.5, 'season_reb_avg': 4.5, 'season_ast_avg': 2.8,
                'season_fg3m_avg': 1.2, 'season_min_avg': 24.0,
                'recent_pts_avg': 12.5, 'recent_reb_avg': 4.5, 'recent_ast_avg': 2.8,
                'recent_fg3m_avg': 1.2, 'recent_min_avg': 24.0,
                'recent_pts_std': 5.0, 'recent_reb_std': 2.0, 'recent_ast_std': 1.5,
                'recent_fg3m_std': 1.0,
                'recent_pts_min': 5.0, 'recent_pts_max': 25.0,
                'last5_pts_avg': 12.5, 'last5_reb_avg': 4.5, 'last5_ast_avg': 2.8,
                'last5_min_avg': 24.0,
                'pts_trend': 0, 'reb_trend': 0, 'ast_trend': 0, 'min_trend': 0,
                'pra_avg': 20.0, 'pra_std': 6.0,
                'opp_def_rating': 114.0, 'opp_off_rating': 114.0, 'opp_pace': 100.0,
                'opp_pts_allowed': 114.0, 'opp_def_strength': 0.0,
                'is_home': 0.5,  # Neutral (would be 0 or 1 for real games)
                'min_consistency': 0.75, 'days_rest': 2, 'is_back_to_back': 0,
                'ts_pct': 0.55, 'efg_pct': 0.50, 'usage_rate': 0.20,
                'fg3_rate': 0.35, 'fta_rate': 0.25,
                # TIER 1.2: Advanced stats defaults
                'bpm': 0.0, 'assist_rate': 4.0, 'rebound_rate': 7.0,
                # TIER 1.1: Position and role feature defaults
                'is_guard': 0, 'is_forward': 0, 'is_center': 0,
                'is_starter': 0, 'is_star': 0, 'is_high_volume': 0, 'is_ball_handler': 0,
                'pos_reb_factor': 1.0, 'pos_ast_factor': 1.0,
                # TIER 1.3: Specialized 3PM feature defaults
                'fg3a_per_min': 0.15, 'fg3a_avg': 4.5, 'fg3a_std': 2.0,
                'fg3a_consistency': 0.7, 'regressed_fg3_pct': 0.36,
                'expected_fg3m': 1.5, 'fg3_makes_std': 1.0,
                'fg3_attempt_trend': 0.0, 'is_volume_shooter': 0,
                'shooting_confidence': 0.5,
            }

            # Build feature array using training defaults for missing features
            feature_vector = [
                features.get(fname, TRAINING_FEATURE_DEFAULTS.get(fname, 0))
                for fname in feature_names
            ]
            X = np.array([feature_vector])

            # Scale features if scaler available
            if scaler is not None:
                X = scaler.transform(X)

            # Make prediction
            prediction = model.predict(X)[0]

            # ============ PHASE 5: Add prediction bounds checking ============
            # Clamp predictions to realistic NBA ranges
            PREDICTION_BOUNDS = {
                'Points': (0, 60),     # NBA realistic max ~60 points
                'Rebounds': (0, 30),   # NBA realistic max ~30 rebounds
                'Assists': (0, 25),    # NBA realistic max ~25 assists
                '3PM': (0, 15),        # NBA record 14, realistic max ~15
                'PRA': (0, 90),        # Combined stat
                'points': (0, 60),
                'rebounds': (0, 30),
                'assists': (0, 25),
                'threes': (0, 15),
                'pra': (0, 90),
            }

            min_bound, max_bound = PREDICTION_BOUNDS.get(model_key, PREDICTION_BOUNDS.get(prop_type, (0, 100)))

            # Also clamp relative to player's average (prevent >2.5x their season average)
            player_avg = pts_avg if model_key in ('points', 'Points') else \
                        reb_avg if model_key in ('rebounds', 'Rebounds') else \
                        ast_avg if model_key in ('assists', 'Assists') else \
                        fg3m_avg if model_key in ('threes', '3PM') else \
                        pra_avg if model_key in ('pra', 'PRA') else 0

            player_max = player_avg * 2.5 if player_avg > 5 else max_bound

            # Apply bounds
            prediction = max(min_bound, min(prediction, max_bound, player_max))

            return float(prediction)

        except Exception as e:
            print(f"ML prediction error for {prop_type}: {e}")
            traceback.print_exc()
            return None

    def _determine_prop_pick(self, prediction: float, line: float) -> Tuple[str, float]:
        """Determine OVER/UNDER pick with proper edge calculation."""
        if line <= 0:
            return "-", 0.0

        # Calculate true edge
        edge = prediction - line
        edge_pct = (edge / line) * 100 if line > 0 else 0

        # Require minimum edge threshold (calibrated to model accuracy)
        MIN_EDGE_PCT = 3.0  # 3% edge minimum to make a pick

        if edge_pct >= MIN_EDGE_PCT:
            return "OVER", round(edge_pct, 1)
        elif edge_pct <= -MIN_EDGE_PCT:
            return "UNDER", round(abs(edge_pct), 1)
        else:
            return "-", round(abs(edge_pct), 1)

    def _get_player_predictions(self, player: Dict, opponent_abbrev: str,
                                 prop_types: List[str],
                                 injury_manager=None,
                                 skip_slow_features=False) -> Optional[Dict]:
        """Get prop predictions for a single player using full feature engineering.

        Args:
            player: Player dictionary with stats
            opponent_abbrev: Opponent team abbreviation
            prop_types: List of prop types to predict
            injury_manager: Optional InjuryReportManager for opponent injury adjustments
            skip_slow_features: If True, skip slow API calls (for background threads)
        """
        prop_map = {
            "Points": ("pts_avg", "pts", "points"),
            "Rebounds": ("reb_avg", "reb", "rebounds"),
            "Assists": ("ast_avg", "ast", "assists"),
            "3PM": ("fg3_avg", "fg3m", "threes"),
            "PRA": ("pra_avg", None, "pra"),
        }

        season_avg = player.get("season_averages", {})
        recent_avg = player.get("recent_averages", {})
        games_played = player.get("games_played", 0)
        player_id = player.get("player_id")

        # Get opponent team ID for matchup analysis
        opponent_team_id = NBA_TEAM_IDS.get(opponent_abbrev)

        result = {
            "player_name": player.get("player_name", "Unknown"),
            "player_id": player_id,
            "position": player.get("position", ""),
            "avg_minutes": player.get("avg_minutes", 0),
        }

        for prop_label in prop_types:
            avg_key, alt_key, model_key = prop_map.get(prop_label, (None, None, None))

            # Get season average as baseline
            if prop_label == "PRA":
                pts = season_avg.get("pts_avg", 0) or season_avg.get("pts", 0) or 0
                reb = season_avg.get("reb_avg", 0) or season_avg.get("reb", 0) or 0
                ast = season_avg.get("ast_avg", 0) or season_avg.get("ast", 0) or 0
                season_value = pts + reb + ast

                r_pts = recent_avg.get("pts_avg", 0) or recent_avg.get("pts", 0) or pts
                r_reb = recent_avg.get("reb_avg", 0) or recent_avg.get("reb", 0) or reb
                r_ast = recent_avg.get("ast_avg", 0) or recent_avg.get("ast", 0) or ast
                recent_value = r_pts + r_reb + r_ast
            elif avg_key:
                season_value = season_avg.get(avg_key, 0) or 0
                if season_value == 0 and alt_key:
                    season_value = season_avg.get(alt_key, 0) or 0
                recent_value = recent_avg.get(avg_key, 0) or recent_avg.get(alt_key, 0) or season_value
            else:
                season_value = 0
                recent_value = 0

            # =============================================================
            # USE ML MODEL FOR PREDICTION (Phase 1 Critical Fix)
            # =============================================================
            pred_value = season_value  # Default fallback
            full_features = None
            used_ml_model = False

            # First try: Use trained ML model with aligned features
            try:
                # Get opponent stats for model features
                opp_stats = None
                is_home = player.get("is_home", True)  # Default to home if unknown

                # Get REAL opponent defensive stats (replaces hardcoded values)
                if opponent_abbrev:
                    opp_stats = self._get_opponent_stats(opponent_abbrev)

                # Try ML model prediction
                ml_prediction = self._predict_with_ml_model(
                    prop_type=prop_label,
                    player_stats=player,
                    opp_stats=opp_stats,
                    is_home=is_home
                )

                if ml_prediction is not None and ml_prediction > 0:
                    pred_value = ml_prediction
                    used_ml_model = True
                    # print(f"  ML model prediction for {player.get('player_name')} {prop_label}: {ml_prediction:.1f}")

            except Exception as e:
                print(f"ML model error for {player.get('player_name')} {prop_label}: {e}")

            # Second try: Use PlayerPropFeatureGenerator for additional context
            # This supplements the ML prediction with opponent-specific adjustments
            # Skip slow feature generation in background threads to prevent hanging
            if not skip_slow_features:
                try:
                    if player_id and self._prop_feature_generator:
                        if prop_label == "Points":
                            full_features = self._prop_feature_generator.generate_points_prop_features(
                                player_id, opponent_team_id=opponent_team_id, last_n_games=10
                            )
                            # If ML model failed, use heuristic fallback
                            if not used_ml_model and full_features:
                                if full_features.get("projected_pts"):
                                    pred_value = full_features["projected_pts"]
                                else:
                                    base = full_features.get("season_pts_avg", season_value)
                                    trend = full_features.get("pts_trend", 0) * 0.5
                                    def_boost = full_features.get("expected_pts_boost", 0)
                                    pred_value = base + trend + def_boost

                        elif prop_label == "Rebounds":
                            full_features = self._prop_feature_generator.generate_rebounds_prop_features(
                                player_id, opponent_team_id=opponent_team_id, last_n_games=10
                            )
                            if not used_ml_model and full_features:
                                base = full_features.get("season_reb_avg", season_value)
                                trend = full_features.get("reb_trend", 0) * 0.5
                                pred_value = base + trend

                        elif prop_label == "Assists":
                            full_features = self._prop_feature_generator.generate_assists_prop_features(
                                player_id, opponent_team_id=opponent_team_id, last_n_games=10
                            )
                            if not used_ml_model and full_features:
                                base = full_features.get("season_ast_avg", season_value)
                                trend = full_features.get("ast_trend", 0) * 0.5
                                pred_value = base + trend

                        elif prop_label == "3PM":
                            full_features = self._prop_feature_generator.generate_threes_prop_features(
                                player_id, opponent_team_id=opponent_team_id, last_n_games=10
                            )
                            if not used_ml_model and full_features:
                                recent_fg3 = full_features.get("recent_fg3_avg", 0)
                                if recent_fg3 > 0:
                                    pred_value = recent_fg3 * 0.6 + season_value * 0.4
                                else:
                                    pred_value = season_value

                        elif prop_label == "PRA":
                            full_features = self._prop_feature_generator.generate_pra_prop_features(
                                player_id, opponent_team_id=opponent_team_id, last_n_games=10
                            )
                            if not used_ml_model and full_features:
                                base = full_features.get("season_pra_avg", season_value)
                                trend = full_features.get("pra_trend", 0) * 0.5
                                pred_value = base + trend

                except Exception as e:
                    print(f"Feature generation error for {player.get('player_name')} {prop_label}: {e}")

            # Final fallback: simple weighted average
            if not used_ml_model and full_features is None:
                pred_value = season_value * 0.6 + recent_value * 0.4

            # =============================================================
            # INJURY ADJUSTMENT - Boost prediction if opponent missing defenders
            # =============================================================
            injury_adjustment = 1.0
            injury_confidence_penalty = 0.0
            injury_notes = []

            if injury_manager and opponent_team_id:
                try:
                    # Calculate opponent's injury impact
                    opp_injury_impact = injury_manager.calculate_injury_impact(opponent_team_id)
                    defensive_impact = opp_injury_impact.get('defensive_impact', 0)
                    total_impact = opp_injury_impact.get('total_impact', 0)
                    injured_count = opp_injury_impact.get('injured_player_count', 0)
                    star_player_out = opp_injury_impact.get('star_player_out', False)

                    # Opponent missing defenders = boost our player's projection
                    if defensive_impact > 0.08:  # At least 8% defensive impact
                        # Enhanced boost when star player is out (up to 25% vs 15%)
                        if star_player_out:
                            boost = min(defensive_impact * 0.50, 0.25)  # Larger boost when star is out
                            injury_notes.append(f"Opp star player OUT (+{boost*100:.0f}%)")
                        else:
                            boost = min(defensive_impact * 0.35, 0.15)  # Standard boost
                            injury_notes.append(f"Opp missing defense (+{boost*100:.0f}%)")
                        injury_adjustment = 1.0 + boost
                        injury_confidence_penalty = 0.05  # Small penalty for uncertainty

                    # Apply injury adjustment to prediction
                    if injury_adjustment != 1.0:
                        pred_value *= injury_adjustment

                except Exception as e:
                    # Silently continue if injury calculation fails
                    pass

            # =============================================================
            # MATCHUP ADJUSTMENT - Adjust prediction based on player's history vs this team
            # =============================================================
            matchup_adjustment = 1.0
            matchup_notes = []

            if player_id and opponent_team_id and season_value > 0:
                try:
                    # Map model_key to prop_type for matchup lookup
                    matchup_prop_type = {
                        'Points': 'points', 'points': 'points',
                        'Rebounds': 'rebounds', 'rebounds': 'rebounds',
                        'Assists': 'assists', 'assists': 'assists',
                        '3PM': 'threes', 'threes': 'threes',
                        'PRA': 'pra', 'pra': 'pra'
                    }.get(model_key, prop_type)

                    matchup_data = self._get_player_vs_team_adjustment(
                        player_id, opponent_team_id, matchup_prop_type, season_value
                    )

                    if matchup_data['games_vs_team'] >= 2:
                        matchup_adjustment = matchup_data['adjustment']
                        if abs(matchup_adjustment - 1.0) >= 0.03:  # At least 3% adjustment
                            direction = "+" if matchup_adjustment > 1.0 else ""
                            pct_change = (matchup_adjustment - 1.0) * 100
                            matchup_notes.append(
                                f"vs team: {matchup_data['avg_vs_team']:.1f} avg "
                                f"({matchup_data['games_vs_team']}g, {direction}{pct_change:.0f}%)"
                            )
                            pred_value *= matchup_adjustment

                except Exception as e:
                    # Silently continue if matchup calculation fails
                    pass

            # =============================================================
            # GET PROP LINE - Try real sportsbook lines first (Phase 3)
            # =============================================================
            line = None
            used_real_line = False

            # Try to get real prop line from sportsbooks
            prop_game_id = player.get("game_id")
            if prop_game_id and player_id:
                real_line = self._get_real_prop_line(prop_game_id, player_id, model_key)
                if real_line is not None and real_line > 0:
                    line = real_line
                    used_real_line = True

            # Fall back to estimated line if real line unavailable
            # FIX: Use opponent-defense-adjusted formula instead of fixed multipliers
            # Old formula created systematic UNDER bias by always inflating lines
            if line is None or line <= 0:
                if season_value > 0:
                    # Get opponent defensive rating from features (lower = better defense)
                    # League average is ~110, range typically 105-115
                    opp_def_rating = 110.0  # Default to league average
                    if full_features:
                        opp_def_rating = full_features.get("opp_def_rating", 110.0)

                    league_avg_def = 110.0

                    # Adjust line based on opponent defense quality
                    # Good defense (105) -> def_adjustment = 0.955 (lower line)
                    # Bad defense (115) -> def_adjustment = 1.045 (higher line)
                    def_adjustment = opp_def_rating / league_avg_def

                    # Apply adjustment to season average
                    line = round(season_value * def_adjustment, 1)

                    # Round to nearest 0.5 for realistic lines
                    line = round(line * 2) / 2
                else:
                    line = None  # No line available for players without stats

            # Calculate confidence with full features context
            confidence = self._calculate_prop_confidence(
                pred_value, line, season_value, recent_value, games_played,
                features=full_features
            )

            # Apply injury confidence penalty (adds uncertainty when using injury adjustments)
            if injury_confidence_penalty > 0:
                confidence *= (1.0 - injury_confidence_penalty)

            # Determine pick with proper edge calculation - skip if no line
            if line is not None and line > 0:
                pick, edge = self._determine_prop_pick(pred_value, line)
            else:
                pick, edge = "-", 0  # No pick without a valid line

            prop_key = prop_label.lower().replace(" ", "_")
            result[f"{prop_key}_line"] = line
            result[f"{prop_key}_pred"] = round(pred_value, 1)
            result[f"{prop_key}_pick"] = pick
            result[f"{prop_key}_edge"] = edge
            result[f"{prop_key}_confidence"] = round(confidence, 0)
            result[f"{prop_key}_real_line"] = used_real_line  # Indicates if sportsbook line was used
            result[f"{prop_key}_ml_model"] = used_ml_model    # Indicates if ML model was used

            # Add injury adjustment info for transparency
            if injury_adjustment != 1.0:
                result[f"{prop_key}_injury_adj"] = round((injury_adjustment - 1.0) * 100, 1)  # As percentage
            if injury_notes:
                result[f"{prop_key}_injury_notes"] = injury_notes

            # Add matchup adjustment info for transparency
            if matchup_adjustment != 1.0:
                result[f"{prop_key}_matchup_adj"] = round((matchup_adjustment - 1.0) * 100, 1)  # As percentage
            if matchup_notes:
                result[f"{prop_key}_matchup_notes"] = matchup_notes

            # Add opponent context info for transparency
            opp_def_rating = None
            opp_adjustment = None
            if full_features:
                if "expected_pts_boost" in full_features:
                    opp_adjustment = round(full_features.get("expected_pts_boost", 0), 1)
                    result[f"{prop_key}_opp_adj"] = opp_adjustment
                if "opp_def_rating" in full_features:
                    opp_def_rating = round(full_features.get("opp_def_rating", 110), 1)
                    result[f"{prop_key}_opp_def"] = opp_def_rating

            # Record prediction for continuous learning tracking
            # Only record if we have a meaningful prediction and pick
            if pick != "-" and line > 0 and self._prop_tracker:
                try:
                    game_date = datetime.now().strftime("%Y-%m-%d")
                    team_abbrev = player.get("team_abbrev", "UNK")
                    # Use provided game_id or generate from team matchup
                    prop_game_id = player.get("game_id", f"{team_abbrev}_{opponent_abbrev}_{game_date}")

                    self._prop_tracker.record_prediction(
                        game_id=prop_game_id,
                        game_date=game_date,
                        player_id=player_id,
                        player_name=player.get("player_name", "Unknown"),
                        team_abbrev=team_abbrev,
                        opponent_abbrev=opponent_abbrev,
                        prop_type=prop_label.lower(),
                        predicted_value=round(pred_value, 1),
                        market_line=line,
                        pick=pick,
                        edge_pct=edge,
                        confidence=round(confidence, 0),
                        opp_def_rating=opp_def_rating,
                        opp_adjustment=opp_adjustment,
                    )
                except Exception as e:
                    print(f"Failed to record prediction: {e}")

        return result

    def clear_cache(self, pattern: str = None):
        """Clear cached data."""
        self.cache.clear(pattern)

    def clear_all_caches(self):
        """Clear ALL caches for a complete real-time data refresh.

        This ensures the refresh button fetches truly fresh data
        by clearing:
        - Primary cache (games, odds, predictions, etc.)
        - Secondary caches (prop lines, BAKER projections)
        - Background fetch status
        """
        # Clear primary cache (CacheManager)
        self.cache.clear()

        # Clear secondary caches (prop lines with TTL)
        self._real_prop_lines_cache.clear()
        self._real_prop_lines_timestamps.clear()

        # Clear BAKER projections cache with TTL
        self._baker_projections_cache.clear()
        self._baker_projections_timestamps.clear()

        # Clear background fetch status (thread-safe - forces re-fetch)
        with self._prop_status_lock:
            self._prop_fetch_status.clear()
        with self._analysis_worker_lock:
            self._analysis_status.clear()

        print("All caches cleared for fresh data refresh")


# Module-level singleton instance
_data_service_instance = None


def get_data_service() -> DataService:
    """Get singleton data service instance.

    CRITICAL: Must return the SAME instance for all callbacks,
    otherwise background workers and analysis status won't be shared!
    """
    global _data_service_instance
    if _data_service_instance is None:
        _data_service_instance = DataService()
    return _data_service_instance
