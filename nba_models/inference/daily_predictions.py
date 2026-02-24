"""
Daily NBA Predictions - Comprehensive Betting Analysis

Generates predictions for all bet types:
- Moneyline (win probability)
- Spread (cover probability vs market line)
- Player Props (points, rebounds, assists, threes for all starters)

Uses Balldontlie API (GOAT tier) for real betting lines.

Usage:
    python3 daily_predictions.py              # Today's predictions
    python3 daily_predictions.py --date 2026-01-05  # Specific date
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo
import numpy as np

# Eastern Time for date-sensitive operations
ET = ZoneInfo('America/New_York')

# Create logger instance
logger = logging.getLogger(__name__)

# Suppress logging noise
logging.disable(logging.WARNING)

# Import our modules
from balldontlie_api import BalldontlieAPI
from feature_engineering import generate_game_features, PlayerPropFeatureGenerator, InjuryReportManager, filter_features
from scipy.stats import norm
from data_fetcher import fetch_player_stats_bdl
from injury_tracker_v3 import fetch_current_injuries, InjuryStatus

# BUG FIX: Prop-specific standard deviations (not line-based)
# Previous bug: std = line * 0.20 caused massive miscalibration
# - Rebounds (line ~5): std=1.0 → Z-scores inflated 2-3x → 76.7% avg over_prob
# - Points (line ~25): std=5.0 → Z-scores reasonable → 56.4% avg over_prob
# Fix: Use empirically-derived prop-specific constants from NBA historical data
PROP_STD_DEVS = {
    'points': 5.5,      # Calibrated: 54.7% ✓ (target: 50±5%)
    'rebounds': 7.0,    # Tuned: 6.5 → 7.0 (was 55.2%, target: 50-55%)
    'assists': 2.5,     # Calibrated: 48.7% ✓ (target: 50±5%)
    'threes': 1.8,      # Calibrated from empirical data
    'pra': 9.0,         # Points + Rebounds + Assists combined variance
}

def get_prop_std_dev(prop_type: str) -> float:
    """
    Get empirically-derived standard deviation for prop type.

    These values are based on NBA historical data analysis and represent
    the typical game-to-game variance for each stat category.

    Returns:
        float: Standard deviation for calculating Z-scores
    """
    return PROP_STD_DEVS.get(prop_type.lower(), 5.0)

# Import performance optimizations (Task 4.1)
from prediction_optimizer import get_executor, warmup_cache, clear_cache

# Import prop injury boost calculation
try:
    from player_impact_fetcher import calculate_prop_injury_boost
    HAS_INJURY_BOOST = True
except ImportError:
    HAS_INJURY_BOOST = False
    def calculate_prop_injury_boost(*args, **kwargs):
        return {'boost_factor': 1.0, 'reasons': []}

# Minutes Oracle for distribution-based minutes prediction (Phase 3)
try:
    from minutes_oracle import MinutesPredictor, MinutesFeatureGenerator
    MINUTES_ORACLE_AVAILABLE = True
except ImportError:
    MinutesPredictor = None
    MinutesFeatureGenerator = None
    MINUTES_ORACLE_AVAILABLE = False

# Phase 4: Edge Calculator for proper edge/EV computation
try:
    from edge_calculator import EdgeCalculator
    HAS_EDGE_CALCULATOR = True
except ImportError:
    HAS_EDGE_CALCULATOR = False

# Phase 4: Calibration Service for prediction logging
try:
    from calibration_tracker import CalibrationService
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False

# Phase 5: Calibration adjuster for bias correction
try:
    from calibration_tracker import CalibrationAdjuster, CalibrationDatabase
    HAS_CALIBRATION_ADJUSTER = True
except ImportError:
    HAS_CALIBRATION_ADJUSTER = False

# Phase 5: Lazy-init singleton for calibration adjuster
_calibration_adjuster = None

def _get_calibration_adjuster():
    """Get or create the CalibrationAdjuster singleton."""
    global _calibration_adjuster
    if _calibration_adjuster is None and HAS_CALIBRATION_ADJUSTER:
        try:
            _calibration_adjuster = CalibrationAdjuster(CalibrationDatabase())
        except Exception:
            pass
    return _calibration_adjuster

# Import Kelly bet sizing from risk_management (Task 3.4)
try:
    from risk_management import calculate_kelly_bet_size, get_kelly_multiplier_for_tier
    HAS_KELLY_SIZING = True
except ImportError:
    HAS_KELLY_SIZING = False
    def calculate_kelly_bet_size(*args, **kwargs):
        return 0.0
    def get_kelly_multiplier_for_tier(*args, **kwargs):
        return 0.0

# Helper function to map confidence score to edge quality tier
def get_tier_from_confidence(confidence_score: float) -> str:
    """Map confidence score (0-100) to edge quality tier."""
    if confidence_score >= 90:
        return 'elite'
    if confidence_score >= 75:
        return 'strong'
    if confidence_score >= 60:
        return 'moderate'
    if confidence_score >= 40:
        return 'weak'
    return 'avoid'

# Phase 4: Edge-focused prop edge calculation
def _calculate_prop_edge(over_prob: float, american_odds: int = -110) -> dict:
    """
    Calculate edge for both OVER and UNDER sides of a prop bet.

    Uses EdgeCalculator when available, falls back to legacy formula.

    Args:
        over_prob: Model's probability of the OVER hitting (0-1)
        american_odds: American odds for the bet (default -110)

    Returns:
        Dict with over_edge, under_edge, pick, edge, edge_quality, ev_per_dollar,
        implied_probability, model_probability, has_edge
    """
    under_prob = 1.0 - over_prob

    if HAS_EDGE_CALCULATOR:
        calc = EdgeCalculator(min_edge_threshold=0.02)

        over_result = calc.calculate_edge(over_prob, american_odds)
        under_result = calc.calculate_edge(under_prob, american_odds)

        # Pick the side with more edge
        if over_result.edge >= under_result.edge:
            pick = 'OVER'
            best = over_result
        else:
            pick = 'UNDER'
            best = under_result

        return {
            'over_edge': over_result.edge_percentage,
            'under_edge': under_result.edge_percentage,
            'pick': pick,
            'edge': best.edge_percentage,
            'edge_quality': best.edge_quality,
            'ev_per_dollar': best.ev_per_dollar,
            'implied_probability': best.implied_probability,
            'model_probability': best.model_probability,
            'has_edge': best.has_edge,
        }
    else:
        # Legacy fallback: assumes -110 odds (implied 52.4%)
        over_edge = (over_prob - 0.524) * 100
        under_edge = (under_prob - 0.524) * 100

        if over_edge >= under_edge:
            pick = 'OVER'
            edge = over_edge
        else:
            pick = 'UNDER'
            edge = under_edge

        return {
            'over_edge': over_edge,
            'under_edge': under_edge,
            'pick': pick,
            'edge': edge,
            'edge_quality': 'strong' if edge >= 5 else 'moderate' if edge >= 3 else 'marginal' if edge >= 2 else 'none',
            'ev_per_dollar': edge / 100,
            'implied_probability': 0.524,
            'model_probability': over_prob if pick == 'OVER' else under_prob,
            'has_edge': edge >= 2,
        }


def get_signal_from_edge(edge: float, edge_quality: str = None) -> str:
    """
    Map edge magnitude to BET/LEAN/PASS/FADE signal per CLAUDE.md.

    Args:
        edge: Edge percentage (positive = value, negative = anti-value)
        edge_quality: Quality classification from EdgeCalculator

    Returns:
        Signal string: 'BET', 'LEAN', 'PASS', or 'FADE'
    """
    if edge_quality in ('strong', 'moderate'):
        return 'BET'
    if edge_quality == 'marginal':
        return 'LEAN'
    if edge < -5:
        return 'FADE'
    return 'PASS'


# Import training feature generator for accurate prop predictions
try:
    from train_complete_balldontlie import (
        PlayerStatsCalculator,
        PositionDefenseCalculator,
        TeamStatsCalculator,
        SpreadEnsembleWrapper,
        calculate_pace_adjusted_features,
        calculate_vegas_total_features,
        calculate_blowout_risk_features,
        analyze_schedule_spots,
    )
    HAS_TRAINING_FEATURES = True
except ImportError:
    HAS_TRAINING_FEATURES = False
    print("Note: Training feature generator not available. Using simplified features.")

# Import simulation engine for Monte Carlo predictions
try:
    from simulation_engine import (
        GameSimulator,
        GameSimulatorV3,
        TeamStats,
        PlayerStats,
        PlayerTrackingStats,
        create_player_from_dict,
        create_team_from_dict,
    )
    HAS_SIMULATION_ENGINE = True
except ImportError:
    HAS_SIMULATION_ENGINE = False

# Import V3 tracking data for enhanced simulation
try:
    from tracking_data import (
        ShotAtlas,
        RotationTracker,
        fetch_shot_chart,
        fetch_pbp_historical,
    )
    HAS_TRACKING_DATA = True
except ImportError:
    HAS_TRACKING_DATA = False


def fetch_team_tracking_data(team_id: int, n_games: int = 3) -> tuple[Optional['ShotAtlas'], Optional['RotationTracker']]:
    """
    Fetch tracking data for a team's recent games.

    Fetches shot charts and PBP data for the most recent N games,
    building a ShotAtlas and RotationTracker for zone-based simulation.

    Args:
        team_id: NBA team ID
        n_games: Number of recent games to fetch (default 3)

    Returns:
        Tuple of (ShotAtlas, RotationTracker) or (None, None) if unavailable
    """
    if not HAS_TRACKING_DATA:
        return None, None

    try:
        from tracking_data import (
            ShotAtlas, RotationTracker,
            fetch_shot_chart, fetch_pbp_historical,
            fetch_season_games
        )

        shot_atlas = ShotAtlas()
        rotation_tracker = RotationTracker()

        # Get recent game IDs for team
        game_ids = fetch_season_games(season="2025-26", team_id=team_id)
        recent_games = game_ids[-n_games:] if game_ids else []

        for game_id in recent_games:
            # Fetch shot chart
            shots = fetch_shot_chart(game_id, use_cache=True)
            if shots:
                shot_atlas.add_shots(shots)

            # Fetch PBP for rotations
            plays = fetch_pbp_historical(game_id, use_cache=True)
            if plays:
                # Process for rotations (we don't know opponent ID here, use 0)
                rotation_tracker.process_game(plays, team_id, 0)

        return shot_atlas, rotation_tracker

    except Exception:
        return None, None


# Import portfolio optimizer for bet sizing
try:
    from portfolio_optimizer import (
        PortfolioOptimizer,
        BetType as PortfolioBetType,
        calculate_covariance,
        optimize_portfolio_kelly,
    )
    HAS_PORTFOLIO_OPTIMIZER = True
except ImportError:
    HAS_PORTFOLIO_OPTIMIZER = False

# Import news sentiment for qualitative intelligence
try:
    from news_sentiment import SentimentPipeline
    HAS_SENTIMENT = True
except ImportError:
    HAS_SENTIMENT = False

# Import market microstructure for steam/stale line detection
try:
    from market_microstructure import (
        MarketMonitor,
        ConsensusCalculator,
        OddsFetcher as MarketOddsFetcher,
    )
    HAS_MARKET_MICRO = True
except ImportError:
    HAS_MARKET_MICRO = False

# Global feature generator for player props (lazy loaded)
_prop_feature_gen = None
_player_feature_cache = {}  # Cache player features to avoid redundant API calls
_id_mapper = None  # IDMapper for Balldontlie player/team ID lookups
_injury_manager = None  # InjuryReportManager for injury data
_player_stats_calc = None  # PlayerStatsCalculator for training-compatible features
_team_stats_calc = None  # TeamStatsCalculator for team features
_position_def_calc = None  # PositionDefenseCalculator for opponent features
_balldontlie_api = None  # Shared Balldontlie API instance

def get_feature_engine(season: str = "2025-26") -> PlayerPropFeatureGenerator:
    """Get or create the feature generator for player prop features."""
    global _prop_feature_gen
    if _prop_feature_gen is None:
        _prop_feature_gen = PlayerPropFeatureGenerator(season=season)
    return _prop_feature_gen

def get_id_mapper():
    """Get or create the ID mapper for Balldontlie lookups."""
    global _id_mapper
    if _id_mapper is None:
        try:
            from id_mapping import IDMapper
            _id_mapper = IDMapper()
        except ImportError:
            return None
    return _id_mapper

def get_player_name_from_bdl_id(bdl_player_id: int) -> str | None:
    """Get player name from Balldontlie player ID."""
    mapper = get_id_mapper()
    if mapper:
        return mapper.get_player_name(bdl_player_id)
    return None

def get_bdl_player_id(player_name: str) -> int | None:
    """Get Balldontlie player ID from player name (fast, cached)."""
    mapper = get_id_mapper()
    if mapper:
        return mapper.get_player_id(player_name)
    return None


def get_injury_manager(season: str = "2025-26") -> InjuryReportManager:
    """Get or create the injury manager for injury data."""
    global _injury_manager
    if _injury_manager is None:
        _injury_manager = InjuryReportManager(season=season)
        try:
            _injury_manager.fetch_all_injuries()
        except Exception:
            pass  # Continue without injuries if fetch fails
    return _injury_manager


def get_balldontlie_api() -> BalldontlieAPI | None:
    """Get or create shared Balldontlie API instance."""
    global _balldontlie_api
    if _balldontlie_api is None:
        api_key = os.getenv("BALLDONTLIE_API_KEY")
        if api_key:
            _balldontlie_api = BalldontlieAPI(api_key=api_key)
    return _balldontlie_api


# Cache for future games to avoid redundant API calls
_future_games_cache = {}

def get_future_games_for_team(team_id: int, game_date: str) -> list[dict]:
    """
    Get upcoming games for a team (cached).

    Used for trap game and sandwich game detection.

    Args:
        team_id: Team ID
        game_date: Current game date (YYYY-MM-DD)

    Returns:
        List of upcoming games for this team
    """
    cache_key = f"{team_id}_{game_date}"
    if cache_key in _future_games_cache:
        return _future_games_cache[cache_key]

    api = get_balldontlie_api()
    if not api:
        return []

    try:
        future_games = api.get_upcoming_games(team_id, game_date, days_ahead=7)
        _future_games_cache[cache_key] = future_games
        return future_games
    except Exception:
        return []


def get_player_stats_calculator():
    """Get or create PlayerStatsCalculator for training-compatible features."""
    global _player_stats_calc
    if _player_stats_calc is None and HAS_TRAINING_FEATURES:
        _player_stats_calc = PlayerStatsCalculator(window=10)
    return _player_stats_calc


def generate_complete_prop_features(
    player_id: int,
    player_name: str,
    opponent_team_id: int,
    is_home: bool = False,
    vegas_total: float = None,
) -> dict | None:
    """
    Generate ALL 150 features matching what the model was trained on.

    Uses the same PlayerStatsCalculator logic from training to ensure
    features match exactly.
    """
    if not HAS_TRAINING_FEATURES:
        return None

    api = get_balldontlie_api()
    if not api:
        return None

    # Fetch player's game history
    stats_data = fetch_player_stats_bdl(player_id=player_id)
    game_log = stats_data.get("game_log", [])

    if not game_log or len(game_log) < 3:
        return None

    # Get player info (position, etc.)
    mapper = get_id_mapper()
    position = "G"  # Default
    if mapper and mapper._all_players:
        for p in mapper._all_players:
            if p.get("id") == player_id:
                position = p.get("position", "G") or "G"
                break

    # Create a fresh PlayerStatsCalculator and populate with game data
    calc = PlayerStatsCalculator(window=10)

    # Add games to calculator in chronological order
    for game in sorted(game_log, key=lambda x: x.get("game_date", "")):
        game_date = game.get("game_date", "")
        if not game_date:
            continue

        # Format stats for calculator
        # NOTE: fetch_player_stats_bdl uses fg_made/fg_att format, not fgm/fga
        calc.add_game_stats(
            player_id=player_id,
            game_date=game_date,
            stats={
                'pts': game.get('pts', 0) or 0,
                'reb': game.get('reb', 0) or 0,
                'ast': game.get('ast', 0) or 0,
                'stl': game.get('stl', 0) or 0,
                'blk': game.get('blk', 0) or 0,
                'fg3m': game.get('fg3_made') or game.get('fg3m', 0) or 0,
                'fg3a': game.get('fg3_att') or game.get('fg3a', 0) or 0,
                'fgm': game.get('fg_made') or game.get('fgm', 0) or 0,
                'fga': game.get('fg_att') or game.get('fga', 0) or 0,
                'ftm': game.get('ft_made') or game.get('ftm', 0) or 0,
                'fta': game.get('ft_att') or game.get('fta', 0) or 0,
                'min': game.get('min', 0) or 0,
                'turnover': game.get('tov') or game.get('turnover', 0) or 0,
                'team': {'id': game.get('team_id')},
                'game': {
                    'home_team': {'id': 0},  # Simplified
                    'visitor_team': {'id': 0},
                },
            },
            player_info={'position': position}
        )

    # Get features using a future date to include all games
    future_date = "2099-12-31"
    base_features = calc.get_player_stats_before_date(
        player_id=player_id,
        date=future_date,
        min_games=3
    )

    if not base_features:
        return None

    # Add opponent features (use defaults for now, can be enhanced)
    opponent_features = {
        'opp_def_rating': 114.0,  # League average
        'opp_off_rating': 114.0,
        'opp_net_rating': 0.0,
        'opp_pts_allowed': 114.0,
        'opp_pts_allowed_recent': 114.0,
        'opp_pts_allowed_std': 8.0,
        'opp_pace': 100.0,
        'opp_pace_season': 100.0,
        'opp_def_strength': 0.0,
        'opp_reb_factor': 1.0,
        'opp_location_def': 0.0,
        'opp_win_pct': 0.5,
        'opp_recent_win_pct': 0.5,
        'is_home': 1 if is_home else 0,
        'team_pace': 100.0,
        'team_off_rating': 114.0,
        # Position-specific opponent allowed stats (use league averages)
        'opp_pts_allowed_to_guards': 18.0,
        'opp_reb_allowed_to_guards': 3.5,
        'opp_ast_allowed_to_guards': 5.5,
        'opp_fg3m_allowed_to_guards': 2.0,
        'opp_pts_allowed_to_forwards': 16.0,
        'opp_reb_allowed_to_forwards': 6.5,
        'opp_ast_allowed_to_forwards': 3.0,
        'opp_fg3m_allowed_to_forwards': 1.5,
        'opp_pts_allowed_to_centers': 14.0,
        'opp_reb_allowed_to_centers': 9.0,
        'opp_ast_allowed_to_centers': 2.5,
        'opp_fg3m_allowed_to_centers': 0.5,
        'opp_pts_vs_pos_std': 3.0,
        'opp_pts_vs_pos_diff': 0.0,
        'opp_reb_vs_pos_diff': 0.0,
        'opp_ast_vs_pos_diff': 0.0,
        'opp_fg3m_vs_pos_diff': 0.0,
    }
    base_features.update(opponent_features)

    # Add pace-adjusted features
    try:
        pace_features = calculate_pace_adjusted_features(
            player_features=base_features,
            team_pace=100.0,
            opponent_pace=100.0,
            vegas_spread=0.0
        )
        base_features.update(pace_features)
    except Exception:
        # Use defaults if calculation fails
        base_features.update({
            'blowout_probability': 0.1,
            'expected_min_reduction': 0.0,
            'projected_min_factor': 1.0,
            'is_likely_blowout': 0,
            'spread_magnitude': 0.0,
            'expected_game_pace': 100.0,
            'pace_multiplier': 1.0,
            'pace_vs_average': 0.0,
            'is_high_pace_game': 0,
            'is_low_pace_game': 0,
            'pace_pts_adjustment': 0.0,
            'pace_reb_adjustment': 0.0,
            'pace_ast_adjustment': 0.0,
            'pace_fg3_adjustment': 0.0,
            'pts_per_100_poss': base_features.get('season_pts_avg', 15) * 3.0,
            'reb_per_100_poss': base_features.get('season_reb_avg', 5) * 3.0,
            'ast_per_100_poss': base_features.get('season_ast_avg', 4) * 3.0,
        })

    # Add Vegas total features
    try:
        vegas_features = calculate_vegas_total_features(
            vegas_total=vegas_total or 225.0,
            player_features=base_features,
            is_starter=base_features.get('is_starter', 1)
        )
        base_features.update(vegas_features)
    except Exception:
        base_features.update({
            'vegas_total': 225.0,
            'total_vs_average': 0.0,
            'total_multiplier': 1.0,
            'is_high_total_game': 0,
            'is_low_total_game': 0,
            'total_pts_boost': 0.0,
        })

    # Add regression features
    base_features.update({
        'pts_deviation_from_mean': base_features.get('season_pts_avg', 15) - 15.0,
        'pts_regression_adjustment': 0.0,
        'pts_regressed_estimate': base_features.get('season_pts_avg', 15),
        'reb_deviation_from_mean': base_features.get('season_reb_avg', 5) - 5.0,
        'reb_regression_adjustment': 0.0,
        'ast_deviation_from_mean': base_features.get('season_ast_avg', 4) - 4.0,
        'ast_regression_adjustment': 0.0,
        'fg3_deviation_from_mean': base_features.get('season_fg3m_avg', 1.5) - 1.5,
        'fg3_regression_adjustment': 0.0,
    })

    return base_features


def apply_injury_adjustments(
    home_prob: float,
    away_prob: float,
    injury_features: dict
) -> tuple[float, float]:
    """
    Apply post-hoc injury adjustments to win probabilities.

    Since models weren't trained with injury data, we apply adjustments
    after prediction. Research shows star player injuries shift win prob 4-8%.
    """
    home_key_out = injury_features.get("home_key_players_out", 0)
    away_key_out = injury_features.get("away_key_players_out", 0)
    injury_advantage = injury_features.get("injury_advantage", 0)

    # Each key player out shifts probability ~4%
    adjustment = (away_key_out - home_key_out) * 0.04

    # Add injury advantage impact (scaled down since it's cumulative)
    if injury_advantage != 0:
        adjustment += injury_advantage * 0.002

    # Cap adjustment at +/- 15%
    adjustment = max(-0.15, min(0.15, adjustment))

    adjusted_home = home_prob + adjustment
    adjusted_home = max(0.1, min(0.9, adjusted_home))
    adjusted_away = 1 - adjusted_home

    return adjusted_home, adjusted_away

def get_cached_features(player_name: str, prop_type: str, opponent_id: int,
                        bdl_player_id: int = None, is_home: bool = False,
                        vegas_total: float = None) -> dict:
    """
    Get cached features or generate new ones using Balldontlie data.

    CRITICAL: Uses training-compatible feature generator to produce ALL 150 features
    that models expect. This fixes the feature mismatch bug where only 27 features
    were generated, causing garbage predictions (e.g., Brunson 16.8 instead of 27).

    Falls back to simplified generator if training features unavailable.

    NOTE: Uses in-memory cache (_player_feature_cache) for fast repeated access.
    Cache persists for the duration of the prediction run. (Task 4.1 optimization)
    """
    cache_key = f"{player_name}_{prop_type}_{opponent_id}"
    if cache_key in _player_feature_cache:
        return _player_feature_cache[cache_key]

    # Skip unknown players
    if not player_name or player_name.startswith("Player "):
        return None

    # Get Balldontlie player ID if not provided
    if bdl_player_id is None:
        bdl_player_id = get_bdl_player_id(player_name)

    if not bdl_player_id:
        return None

    # TRY TRAINING-COMPATIBLE FEATURES FIRST (generates all 150 features)
    if HAS_TRAINING_FEATURES:
        try:
            features = generate_complete_prop_features(
                player_id=bdl_player_id,
                player_name=player_name,
                opponent_team_id=opponent_id,
                is_home=is_home,
                vegas_total=vegas_total,
            )
            if features:
                _player_feature_cache[cache_key] = features
                return features
        except Exception:
            # Log but continue to fallback
            pass

    # FALLBACK: Use old feature generator (only 27 features - may cause issues)
    fe = get_feature_engine()
    try:
        if prop_type == 'points':
            features = fe.generate_points_prop_features(
                player_id=bdl_player_id,
                opponent_team_id=opponent_id,
            )
        elif prop_type == 'rebounds':
            features = fe.generate_rebounds_prop_features(
                player_id=bdl_player_id,
                opponent_team_id=opponent_id,
            )
        elif prop_type == 'assists':
            features = fe.generate_assists_prop_features(
                player_id=bdl_player_id,
                opponent_team_id=opponent_id,
            )
        else:
            features = fe.generate_points_prop_features(
                player_id=bdl_player_id,
                opponent_team_id=opponent_id,
            )

        if features:
            _player_feature_cache[cache_key] = features
            return features
    except Exception:
        pass

    return None

# Constants
NBA_SPREAD_VOLATILITY = 13.0  # Historical std dev of NBA margins
MODEL_DIR = Path("models")


def load_models() -> dict:
    """Load all prediction models."""
    models = {}

    # Moneyline - try meta-learner first, then stacking, then fall back to ensemble
    ml_path = MODEL_DIR / "moneyline_stacking_metalearner.pkl"
    if not ml_path.exists():
        ml_path = MODEL_DIR / "moneyline_stacking.pkl"
    if not ml_path.exists():
        ml_path = MODEL_DIR / "moneyline_ensemble.pkl"  # Fallback
    if ml_path.exists():
        try:
            with open(ml_path, 'rb') as f:
                data = pickle.load(f)
                models['moneyline'] = data.get('model', data) if isinstance(data, dict) else data
            print(f"    Loaded moneyline from {ml_path.name}")
        except Exception as e:
            print(f"    Warning: Could not load moneyline model: {e}")

    # Spread - try meta-learner first, then stacking, then fall back to ensemble
    spread_path = MODEL_DIR / "spread_stacking_metalearner.pkl"
    if not spread_path.exists():
        spread_path = MODEL_DIR / "spread_stacking.pkl"
    if not spread_path.exists():
        spread_path = MODEL_DIR / "spread_ensemble.pkl"  # Fallback
    if spread_path.exists():
        try:
            with open(spread_path, 'rb') as f:
                data = pickle.load(f)
                models['spread'] = data.get('model', data) if isinstance(data, dict) else data
            print(f"    Loaded spread from {spread_path.name}")
        except Exception as e:
            print(f"    Warning: Could not load spread model: {e}")

    # Player prop models - load available models
    for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
        # Try different model files in order of preference
        # IMPORTANT: Stacking models come first, then ensemble
        model_paths = [
            MODEL_DIR / f"player_{prop_type}_stacking.pkl",  # New stacking models
            MODEL_DIR / f"player_{prop_type}_ensemble.pkl",  # Ensemble with 150 features
            MODEL_DIR / f"player_{prop_type}_line_classifier.pkl",
            MODEL_DIR / f"player_{prop_type}_position_aware.pkl",
            MODEL_DIR / f"player_{prop_type}.pkl",  # Simple regressor (old, 27 features)
        ]

        # Load quantile model separately for prediction bands (Task 3.4)
        quantile_path = MODEL_DIR / f"player_{prop_type}_quantile.pkl"
        if quantile_path.exists():
            try:
                with open(quantile_path, 'rb') as f:
                    quantile_data = pickle.load(f)
                models[f'prop_{prop_type}_quantile'] = quantile_data
                print(f"    Loaded quantile model for {prop_type}")
            except Exception as e:
                print(f"    Warning: Could not load quantile model for {prop_type}: {e}")

        for path in model_paths:
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)

                    # Handle ensemble format (has 'models' and 'meta_model')
                    if isinstance(data, dict) and 'models' in data and 'meta_model' in data:
                        # Store full ensemble for proper prediction
                        models[f'prop_{prop_type}'] = {
                            'ensemble': True,
                            'models': data['models'],
                            'meta_model': data['meta_model'],
                            'model_weights': data.get('model_weights', {}),
                            'scaler': data.get('scaler'),
                            'feature_names': data.get('feature_names', []),
                            'over_under_classifier': data.get('over_under_classifier'),
                            'prop_type': prop_type,
                        }
                        break

                    # Handle dict format with single model, scaler, feature_names
                    if isinstance(data, dict):
                        model = data.get('model')
                        scaler = data.get('scaler')
                        feature_names = data.get('feature_names', [])

                        if model and hasattr(model, 'predict'):
                            models[f'prop_{prop_type}'] = {
                                'model': model,
                                'scaler': scaler,
                                'feature_names': feature_names,
                                'prop_type': prop_type,
                            }
                            break
                    elif hasattr(data, 'predict'):
                        models[f'prop_{prop_type}'] = data
                        break
                except Exception:
                    continue

    # Load Minutes Oracle (Phase 3)
    if MINUTES_ORACLE_AVAILABLE:
        minutes_path = MODEL_DIR / "minutes_oracle.pkl"
        if minutes_path.exists():
            try:
                models['minutes_oracle'] = MinutesPredictor.load(minutes_path)
                models['minutes_feature_gen'] = MinutesFeatureGenerator()
                print(f"    Loaded Minutes Oracle")
            except Exception as e:
                print(f"    Warning: Could not load Minutes Oracle: {e}")

    return models


def predict_minutes_distribution(
    player_id: int,
    team_id: int,
    opponent_team_id: int,
    game_context: dict,
    models: dict,
    game_date: str = None,
) -> dict:
    """Predict minutes distribution using the Minutes Oracle.

    Returns dict with p10/p25/p50/p75/p90/expected/uncertainty/spread,
    or a fallback dict based on historical average if oracle unavailable.
    """
    oracle = models.get('minutes_oracle')
    feature_gen = models.get('minutes_feature_gen')

    if oracle is None or feature_gen is None:
        return None

    try:
        if game_date is None:
            game_date = datetime.now(ET).strftime('%Y-%m-%d')

        # Map game_context keys to what MinutesFeatureGenerator expects
        oracle_context = {
            'vegas_spread': game_context.get('spread', 0),
            'vegas_total': game_context.get('total', 220),
            'is_home': game_context.get('is_home', True),
            'is_back_to_back': game_context.get('is_b2b', False),
            'days_rest': game_context.get('days_rest', 1),
        }

        features = feature_gen.generate_features(
            player_id=player_id,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            game_date=game_date,
            game_context=oracle_context,
        )

        dist = oracle.predict(features, player_id=player_id)
        return dist.to_dict()

    except Exception:
        return None


def get_spread_cover_probability(edge_points: float) -> float:
    """Convert point edge to cover probability using normal CDF."""
    return norm.cdf(edge_points / NBA_SPREAD_VOLATILITY)


def get_implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def predict_moneyline(features: dict, models: dict) -> tuple[float, float]:
    """Predict moneyline probabilities."""
    model = models.get('moneyline')
    if not model:
        return 0.5, 0.5

    # Extract key features
    feature_cols = ['net_rating_diff', 'win_pct_diff', 'pace_diff', 'off_rating_diff',
                   'def_rating_diff', 'home_win_streak', 'away_win_streak']

    X = np.array([[features.get(col, 0) for col in feature_cols]])

    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[0]
            home_prob = probs[1] if len(probs) > 1 else probs[0]
        else:
            # Fallback to simple formula
            net_rating_diff = features.get('net_rating_diff', 0)
            home_prob = 0.5 + (net_rating_diff * 0.02)
    except Exception:
        net_rating_diff = features.get('net_rating_diff', 0)
        home_prob = 0.5 + (net_rating_diff * 0.02)

    home_prob = max(0.1, min(0.9, home_prob))
    return home_prob, 1 - home_prob


def predict_spread(features: dict, models: dict) -> float:
    """Predict point spread (positive = home favored)."""
    model = models.get('spread')

    # Simple formula based on net rating
    net_rating_diff = features.get('net_rating_diff', 0)
    home_advantage = 3.0

    if model:
        try:
            feature_cols = ['net_rating_diff', 'off_rating_diff', 'def_rating_diff',
                          'pace_diff', 'expected_point_diff']
            X = np.array([[features.get(col, 0) for col in feature_cols]])
            return model.predict(X)[0]
        except Exception:
            pass

    # Fallback: net rating / 3 + home advantage
    return (net_rating_diff / 3.0) + home_advantage


def simulate_game_predictions(
    home_team_data: dict,
    away_team_data: dict,
    home_players: list[dict],
    away_players: list[dict],
    n_simulations: int = 1000
) -> dict | None:
    """
    Use Monte Carlo simulation for enhanced predictions.

    Provides more accurate probability distributions than regression.
    Captures pace effects, blowout scenarios, and player correlations.

    Args:
        home_team_data: Home team info dict
        away_team_data: Away team info dict
        home_players: List of home player stat dicts
        away_players: List of away player stat dicts
        n_simulations: Number of simulations (default 1000)

    Returns:
        Dictionary with simulation results or None if unavailable
    """
    if not HAS_SIMULATION_ENGINE:
        return None

    try:
        # Create team objects
        home_team = create_team_from_dict(home_team_data, home_players)
        away_team = create_team_from_dict(away_team_data, away_players)

        # Use V3 simulator with tracking data if available
        use_v3 = HAS_TRACKING_DATA and os.environ.get('USE_V3_SIMULATION', '1') == '1'

        if use_v3:
            simulator = GameSimulatorV3(home_team, away_team)

            # Try to load tracking data for enhanced accuracy
            shot_atlas = None
            rotation_tracker = None
            tracking_loaded = False

            try:
                from pathlib import Path
                import json
                cache_dir = Path(__file__).parent / ".tracking_cache"
                cache_dir.mkdir(exist_ok=True)

                # Create ShotAtlas and RotationTracker
                shot_atlas = ShotAtlas()
                rotation_tracker = RotationTracker()

                # Get team IDs for filtering
                home_team_id = home_team_data.get('id', 0)
                away_team_id = away_team_data.get('id', 0)

                # 1. Load cached shot data for BOTH teams specifically
                shots_loaded = 0
                for cache_file in cache_dir.glob("shots_*.json"):
                    try:
                        with open(cache_file) as f:
                            data = json.load(f)

                        # Parse shots
                        from tracking_data import _parse_shot_chart_response
                        game_id = cache_file.stem.replace('shots_', '')
                        shots = _parse_shot_chart_response(data, game_id)

                        if shots:
                            # Check if shots are from either team
                            game_team_ids = {s.team_id for s in shots}
                            if home_team_id in game_team_ids or away_team_id in game_team_ids:
                                shot_atlas.add_shots(shots)
                                shots_loaded += len(shots)
                    except Exception:
                        continue

                # 2. Load cached PBP for rotation tracking
                games_processed = 0
                for cache_file in list(cache_dir.glob("pbp_*.json"))[:10]:
                    try:
                        with open(cache_file) as f:
                            data = json.load(f)

                        # Parse PBP
                        from tracking_data import _parse_pbp_cdn, _parse_pbp_response
                        game_id = cache_file.stem.replace('pbp_', '')

                        if 'game' in data:
                            plays = _parse_pbp_cdn(data, game_id)
                        else:
                            plays = _parse_pbp_response(data, game_id)

                        if plays:
                            # Identify team IDs from plays
                            play_team_ids = {p.team_id for p in plays if p.team_id}
                            if home_team_id in play_team_ids or away_team_id in play_team_ids:
                                rotation_tracker.process_game(plays, home_team_id, away_team_id)
                                games_processed += 1
                    except Exception:
                        continue

                # 3. Load tracking data into simulator
                if shot_atlas.league_zones:
                    simulator.load_tracking_data(
                        shot_atlas=shot_atlas,
                        rotation_tracker=rotation_tracker if rotation_tracker.player_minutes else None
                    )
                    tracking_loaded = True

            except Exception:
                pass  # Fall back to V3 without tracking data

            results = simulator.run_simulation(n_simulations=n_simulations)
            source = 'monte_carlo_v3' + ('_tracking' if tracking_loaded else '')
        else:
            # Use standard simulator
            simulator = GameSimulator(home_team, away_team)
            results = simulator.run_simulation(n_simulations=n_simulations)
            source = 'monte_carlo'

        # Get betting probabilities
        return {
            'home_win_prob': results['home_win_prob'],
            'away_win_prob': results['away_win_prob'],
            'projected_home_score': results['home_score_mean'],
            'projected_away_score': results['away_score_mean'],
            'projected_margin': results['margin_mean'],
            'margin_std': results['margin_std'],
            'projected_total': results['total_mean'],
            'total_std': results['total_std'],
            'simulator': simulator,  # For prop calculations
            'source': source,
        }

    except Exception as e:
        print(f"  Simulation error: {e}")
        return None


def optimize_bet_portfolio(
    predictions: list[dict],
    bankroll: float = 1000,
) -> dict | None:
    """
    Optimize bet sizing across all predictions using portfolio optimization.

    Uses covariance-aware Kelly criterion to size bets accounting for
    correlations between same-game bets.

    Args:
        predictions: List of prediction dicts with edge/probability
        bankroll: Total bankroll

    Returns:
        Optimized portfolio or None if unavailable
    """
    if not HAS_PORTFOLIO_OPTIMIZER:
        return None

    try:
        optimizer = PortfolioOptimizer(bankroll=bankroll)

        for pred in predictions:
            if pred.get('edge', 0) < 0.02:  # Skip low-edge bets
                continue

            # Determine bet type
            bet_type_str = pred.get('type', 'moneyline')
            if bet_type_str == 'moneyline':
                bet_type = PortfolioBetType.MONEYLINE
            elif bet_type_str == 'spread':
                bet_type = PortfolioBetType.SPREAD
            elif bet_type_str == 'total':
                bet_type = PortfolioBetType.TOTAL
            else:
                bet_type = PortfolioBetType.PLAYER_PROP

            optimizer.add_bet(
                game_id=str(pred.get('game_id', '')),
                bet_type=bet_type,
                selection=pred.get('selection', ''),
                odds=pred.get('odds', -110),
                probability=pred.get('probability', 0.5),
                team=pred.get('team'),
                player=pred.get('player'),
                side=pred.get('side'),
            )

        result = optimizer.optimize()
        return result.to_dict()

    except Exception as e:
        print(f"  Portfolio optimization error: {e}")
        return None


def analyze_game(game: dict, odds: dict, models: dict) -> dict:
    """
    Analyze a single game with all bet types.

    Args:
        game: Game info from Balldontlie
        odds: Betting odds for this game
        models: Loaded prediction models

    Returns:
        Analysis dictionary with all predictions
    """
    home_team = game.get('home_team', {})
    away_team = game.get('visitor_team', {})

    home_abbrev = home_team.get('abbreviation', 'HOME')
    away_abbrev = away_team.get('abbreviation', 'AWAY')
    game_time = game.get('status', '')
    game_id = game.get('id')

    # Data freshness tracking (Phase 2, Step 3)
    try:
        from nba_data.validators.freshness import DataFreshness
        freshness = DataFreshness()
    except ImportError:
        freshness = None

    analysis = {
        'game_id': game_id,
        'home_team': home_abbrev,
        'away_team': away_abbrev,
        'game_time': game_time,
        'moneyline': {},
        'spread': {},
        'player_props': []
    }

    # Generate features with injury data
    injury_mgr = get_injury_manager()
    injury_features = {}
    injury_details = {'home': [], 'away': []}

    try:
        features = generate_game_features(
            home_abbrev, away_abbrev,
            season="2025-26",
            include_advanced=True,
            injury_manager=injury_mgr
        )
        if freshness:
            freshness.record_stats_fetch()
            freshness.record_injuries_fetch()
        ml_features = features.get('moneyline_features', {}) if features else {}

        # Extract injury features from moneyline_features (where they're stored)
        if ml_features:
            injury_features = {
                'home_injury_impact': ml_features.get('home_injury_impact', 0),
                'away_injury_impact': ml_features.get('away_injury_impact', 0),
                'home_key_players_out': ml_features.get('home_key_players_out', 0),
                'away_key_players_out': ml_features.get('away_key_players_out', 0),
                'injury_advantage': ml_features.get('injury_advantage', 0),
            }
            injury_details = ml_features.get('injury_details', {'home': [], 'away': []})
    except Exception:
        ml_features = {}

    if not ml_features:
        # Use basic defaults
        pass

    # Get schedule spot features (trap games, sandwich games, etc.)
    schedule_spots = {'home': {}, 'away': {}}
    if HAS_TRAINING_FEATURES:
        try:
            # Get game date for schedule analysis
            game_date_str = game.get('date', '') or datetime.now().strftime("%Y-%m-%d")
            if 'T' in game_date_str:
                game_date_str = game_date_str.split('T')[0]

            # Get team IDs
            home_team_id = home_team.get('id', 0)
            away_team_id = away_team.get('id', 0)

            # Create minimal TeamStatsCalculator for live predictions
            # (schedule spots analysis needs it but we don't have full history)
            team_calc = TeamStatsCalculator(window=10)

            # Get future games for each team
            home_future = get_future_games_for_team(home_team_id, game_date_str)
            away_future = get_future_games_for_team(away_team_id, game_date_str)

            # Analyze schedule spots for home team
            schedule_spots['home'] = analyze_schedule_spots(
                team_id=home_team_id,
                team_abbrev=home_abbrev,
                game_date=game_date_str,
                opponent_abbrev=away_abbrev,
                team_calc=team_calc,
                is_home=True,
                future_games=home_future
            )

            # Analyze schedule spots for away team
            schedule_spots['away'] = analyze_schedule_spots(
                team_id=away_team_id,
                team_abbrev=away_abbrev,
                game_date=game_date_str,
                opponent_abbrev=home_abbrev,
                team_calc=team_calc,
                is_home=False,
                future_games=away_future
            )
        except Exception:
            pass  # Continue without schedule spots if analysis fails

    analysis['schedule_spots'] = schedule_spots

    if not ml_features:
        # Use basic defaults
        ml_features = {'net_rating_diff': 0, 'win_pct_diff': 0}

    # Store injury info in analysis
    analysis['injury_features'] = injury_features
    analysis['injury_details'] = injury_details

    # Apply feature selection (if models/selected_features.json exists)
    ml_features = filter_features(ml_features)

    # Record odds freshness (odds were fetched by the caller before this)
    if freshness and odds:
        freshness.record_odds_fetch()

    # Moneyline prediction (with injury adjustments)
    home_prob, away_prob = predict_moneyline(ml_features, models)

    # Apply post-hoc injury adjustments
    if injury_features.get('home_key_players_out', 0) or injury_features.get('away_key_players_out', 0):
        home_prob, away_prob = apply_injury_adjustments(home_prob, away_prob, injury_features)

    # Get market odds
    home_ml_odds = odds.get('home_moneyline', -110)
    away_ml_odds = odds.get('away_moneyline', -110)
    home_implied = get_implied_probability(home_ml_odds)
    away_implied = get_implied_probability(away_ml_odds)

    analysis['moneyline'] = {
        'home_prob': home_prob,
        'away_prob': away_prob,
        'home_odds': home_ml_odds,
        'away_odds': away_ml_odds,
        'home_edge': (home_prob - home_implied) * 100,
        'away_edge': (away_prob - away_implied) * 100,
    }

    # Spread prediction
    predicted_spread = predict_spread(ml_features, models)
    market_spread = odds.get('spread', 0)  # Negative = home favored

    # FIXED: Use app.py's proven formula for spread edge calculation.
    # Convention: predicted_spread = home margin (+ = home wins by X)
    #             market_spread = home line (- = home favored, + = home underdog)
    # home_cover_threshold = -market_spread = points home must win by to cover
    # e.g., market_spread=-12 → threshold=12 (home must win by 12+)
    # e.g., market_spread=+5.5 → threshold=-5.5 (home can lose by up to 5)
    home_cover_threshold = -market_spread
    spread_edge_points = predicted_spread - home_cover_threshold  # = predicted_spread + market_spread

    if spread_edge_points > 0:
        # Home covers: model predicts home exceeds the threshold
        bet_side = f"{home_abbrev} {market_spread:+.1f}"
        cover_prob = get_spread_cover_probability(spread_edge_points)
    else:
        # Away covers: home doesn't meet the threshold
        bet_side = f"{away_abbrev} {-market_spread:+.1f}"
        cover_prob = get_spread_cover_probability(abs(spread_edge_points))

    # Phase 4: Use EdgeCalculator for spread edge when available
    spread_odds = odds.get('spread_home_odds', -110) if spread_edge_points > 0 else odds.get('spread_away_odds', -110)
    if HAS_EDGE_CALCULATOR:
        _spread_calc = EdgeCalculator(min_edge_threshold=0.02)
        _spread_result = _spread_calc.calculate_edge(cover_prob, spread_odds)
        edge_pct = _spread_result.edge_percentage
    else:
        edge_pct = (cover_prob - 0.524) * 100  # vs -110 implied 52.4%

    analysis['spread'] = {
        'predicted_spread': predicted_spread,
        'market_spread': market_spread,
        'spread_edge_points': abs(spread_edge_points),
        'cover_prob': cover_prob,
        'edge_pct': edge_pct,
        'bet_side': bet_side,
    }

    # Include data freshness metadata
    if freshness:
        analysis['data_freshness'] = freshness.to_dict()
        if freshness.is_stale():
            stale = freshness.stale_sources()
            import logging as _log
            _log.getLogger(__name__).warning(
                f"Stale data for {home_abbrev} vs {away_abbrev}: {', '.join(stale)}"
            )

    return analysis


def print_game_analysis(analysis: dict):
    """Print formatted game analysis."""
    home = analysis['home_team']
    away = analysis['away_team']
    time = analysis['game_time']

    print(f"\n{'='*65}")
    print(f"  {away} @ {home}  ({time})")
    print(f"{'='*65}")

    # Display injuries if any
    injury_details = analysis.get('injury_details', {})
    home_injured = injury_details.get('home', []) if isinstance(injury_details, dict) else []
    away_injured = injury_details.get('away', []) if isinstance(injury_details, dict) else []

    if home_injured or away_injured:
        print("\n  INJURIES:")
        if home_injured:
            print(f"    {home}:")
            for inj in home_injured[:5]:  # Limit to 5 per team
                player_name = inj.get('player_name', 'Unknown')
                status = inj.get('status', 'Unknown')
                print(f"      - {player_name} ({status})")
        if away_injured:
            print(f"    {away}:")
            for inj in away_injured[:5]:  # Limit to 5 per team
                player_name = inj.get('player_name', 'Unknown')
                status = inj.get('status', 'Unknown')
                print(f"      - {player_name} ({status})")

    # Moneyline
    ml = analysis['moneyline']
    home_prob = ml['home_prob']
    away_prob = ml['away_prob']
    home_edge = ml['home_edge']
    away_edge = ml['away_edge']

    ml_rec = ""
    if home_edge > 3:
        ml_rec = f">>> {home} ML"
    elif away_edge > 3:
        ml_rec = f">>> {away} ML"

    print("\n  MONEYLINE:")
    print(f"    {home}: {home_prob:.1%} (edge: {home_edge:+.1f}%)")
    print(f"    {away}: {away_prob:.1%} (edge: {away_edge:+.1f}%)")
    if ml_rec:
        print(f"    {ml_rec}")

    # Spread
    sp = analysis['spread']
    print("\n  SPREAD:")
    print(f"    Model: {home} {sp['predicted_spread']:+.1f}")
    print(f"    Market: {home} {sp['market_spread']:+.1f}")
    print(f"    Cover Prob: {sp['cover_prob']:.1%} | Edge: {sp['edge_pct']:+.1f}%")
    if abs(sp['edge_pct']) > 2:
        print(f"    >>> {sp['bet_side']}")

    # Player props (if any)
    props = analysis.get('player_props', [])
    if props:
        print("\n  PLAYER PROPS:")
        for prop in props:
            player = prop.get('player', 'Unknown')
            stat = prop.get('stat', '')
            line = prop.get('line', 0)
            over_prob = prop.get('over_prob', 0.5)
            edge = prop.get('edge', 0)
            predicted = prop.get('predicted_value')

            # Task 3.4: Display prediction bands and bet sizing
            pred_low = prop.get('pred_low')
            pred_median = prop.get('pred_median')
            pred_high = prop.get('pred_high')
            confidence = prop.get('confidence_score', 50)
            tier = prop.get('edge_quality_tier', 'moderate')
            bet_size = prop.get('suggested_bet_size', 0)
            recommendation = prop.get('bet_recommendation', 'MONITOR')

            direction = "Over" if over_prob > 0.5 else "Under"
            prob = over_prob if over_prob > 0.5 else (1 - over_prob)

            marker = "**" if abs(edge) > 5 else ("*" if abs(edge) > 3 else "")

            # Build display with prediction bands if available
            if pred_low is not None and pred_median is not None and pred_high is not None:
                pred_str = f"[{pred_low:.1f} | {pred_median:.1f} | {pred_high:.1f}]"
                print(f"    {player} {stat} {line}: {direction} {prob:.0%} ({edge:+.1f}%) {marker}")
                print(f"      Pred: {pred_str} | Conf: {confidence:.0f} ({tier.upper()}) | ${bet_size:.0f} ({recommendation})")
            elif predicted is not None:
                print(f"    {player} {stat} {line} (pred: {predicted:.1f}): {direction} {prob:.0%} ({edge:+.1f}%) {marker}")
                print(f"      Conf: {confidence:.0f} ({tier.upper()}) | ${bet_size:.0f} ({recommendation})")
            else:
                print(f"    {player} {stat} {line}: {direction} {prob:.0%} ({edge:+.1f}%) {marker}")


def get_player_props_for_game(api: BalldontlieAPI, game_id: int) -> dict[int, dict]:
    """
    Get player props from Balldontlie API for a game.

    Returns dict indexed by player_id with prop lines.
    """
    props_by_player = {}

    try:
        props = api.get_player_props(game_id)
        if not props:
            return props_by_player

        # Group by player and prop type, preferring DraftKings
        for prop in props:
            player_id = prop.get('player_id')
            prop_type = prop.get('prop_type', '').lower()
            vendor = prop.get('vendor', '').lower()
            line = prop.get('line_value')

            if not player_id or not prop_type or not line:
                continue

            try:
                line = float(line)
            except:
                continue

            # Skip milestone props (like "18+ points" which have high odds)
            market = prop.get('market', {})
            if market.get('type') == 'milestone':
                continue

            if player_id not in props_by_player:
                props_by_player[player_id] = {'player_id': player_id}

            # Store team_id if available
            team_id = prop.get('team_id')
            if team_id:
                props_by_player[player_id]['team_id'] = team_id

            # Store if not exists or vendor is preferred
            key = f'{prop_type}_line'
            if key not in props_by_player[player_id] or vendor in ['draftkings', 'fanduel']:
                props_by_player[player_id][key] = line
                props_by_player[player_id][f'{prop_type}_vendor'] = vendor

    except Exception as e:
        print(f"    Error fetching props: {e}")

    return props_by_player


def predict_player_prop(
    player_name: str,
    player_id: int,
    prop_type: str,
    line: float,
    opponent: str,
    opponent_id: int,
    models: dict,
    use_api_features: bool = False,  # Disable by default for speed
    player_position: str = None,  # Player position (G/F/C)
    opponent_injured: list[str] = None,  # Injured players on opponent
    teammate_injured: list[str] = None,  # Injured teammates
    team_id: int = None,  # Player's team ID (Phase 3: minutes oracle)
    game_context: dict = None,  # Game context for minutes oracle
    american_odds: int = -110,  # Phase 4: actual odds for edge calc
) -> dict:
    """
    Predict over/under probability for a player prop.

    Args:
        player_name: Player's name
        player_id: Player's ID
        prop_type: Type of prop (points, rebounds, assists, threes)
        line: The betting line
        opponent: Opponent team abbreviation
        opponent_id: Opponent team ID for matchup features
        models: Loaded models
        use_api_features: If True, fetch player stats from API (slow)
        player_position: Player's position (G/F/C) for injury impact
        opponent_injured: List of injured player names on opponent
        teammate_injured: List of injured teammate names

    Returns:
        Prediction dict with over_prob and edge
    """
    import pandas as pd

    over_prob = 0.5
    edge = 0.0
    predicted_value = None
    features = None  # Initialize for quantile model usage later

    model_data = models.get(f'prop_{prop_type}')

    if model_data and use_api_features:
        try:
            # Get cached features (or generate new ones) - use Balldontlie ID for fast lookup
            features = get_cached_features(player_name, prop_type, opponent_id, bdl_player_id=player_id)

            if features:
                # Handle ENSEMBLE format (multiple models with meta_model)
                if isinstance(model_data, dict) and model_data.get('ensemble'):
                    base_models = model_data['models']
                    meta_model = model_data['meta_model']
                    model_data.get('model_weights', {})
                    scaler = model_data.get('scaler')
                    feature_names = model_data.get('feature_names', [])

                    # Build feature array matching training features
                    X = pd.DataFrame([{k: features.get(k, 0) for k in feature_names}])
                    X = X[feature_names].fillna(0)

                    # Scale if scaler available
                    X_scaled = scaler.transform(X) if scaler is not None else X.values

                    # Get predictions from tree-based models only (ridge can have scaling issues)
                    tree_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
                    base_preds = []
                    for name in tree_models:
                        if name in base_models and base_models[name] is not None:
                            try:
                                pred = base_models[name].predict(X_scaled)[0]
                                if -50 < pred < 100:  # Sanity check
                                    base_preds.append(pred)
                            except Exception:
                                continue

                    # Compute weighted average if we have predictions
                    if base_preds:
                        predicted_value = float(np.mean(base_preds))
                    else:
                        # Fallback to season average from features
                        predicted_value = features.get('season_pts_avg', 15.0)

                    # Convert to probability using normal CDF
                    std = get_prop_std_dev(prop_type)  # FIX: Use prop-specific std
                    z_score = (predicted_value - line) / std
                    over_prob = float(norm.cdf(z_score))

                # Handle StackingRegressor format (has 'base_models' key)
                elif isinstance(model_data, dict) and 'base_models' in model_data and 'meta_model' in model_data:
                    base_models = model_data['base_models']
                    meta_model = model_data['meta_model']
                    scaler = model_data.get('scaler')
                    feature_names = model_data.get('feature_names', [])

                    # Build feature array matching training features
                    X = pd.DataFrame([{k: features.get(k, 0) for k in feature_names}])
                    X = X[feature_names].fillna(0)

                    # Scale if scaler available
                    X_scaled = scaler.transform(X) if scaler is not None else X.values

                    # Get base model predictions
                    base_preds = []
                    for name, model in base_models.items():
                        try:
                            pred = model.predict(X_scaled)[0]
                            if -50 < pred < 100:  # Sanity check
                                base_preds.append(pred)
                        except Exception:
                            continue

                    # Use meta model for stacking
                    if meta_model is not None and base_preds:
                        meta_features = np.array(base_preds).reshape(1, -1)
                        predicted_value = float(meta_model.predict(meta_features)[0])
                    elif base_preds:
                        predicted_value = float(np.mean(base_preds))
                    else:
                        predicted_value = features.get('season_pts_avg', 15.0)

                    # Convert to probability using normal CDF
                    std = get_prop_std_dev(prop_type)  # FIX: Use prop-specific std
                    z_score = (predicted_value - line) / std
                    over_prob = float(norm.cdf(z_score))

                # Handle dict format with single model
                elif isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                    scaler = model_data.get('scaler')
                    feature_names = model_data.get('feature_names', [])

                    # Build feature array matching training features
                    X = pd.DataFrame([{k: features.get(k, 0) for k in feature_names}])
                    for col in feature_names:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[feature_names].fillna(0)

                    # Scale if scaler available
                    X_scaled = scaler.transform(X) if scaler is not None else X.values

                    # Predict (regression model predicts stat value)
                    predicted_value = float(model.predict(X_scaled)[0])

                    # Convert to probability using normal CDF
                    std = get_prop_std_dev(prop_type)  # FIX: Use prop-specific std
                    z_score = (predicted_value - line) / std
                    over_prob = float(norm.cdf(z_score))

                # Handle model object with predict method
                elif hasattr(model_data, 'predict'):
                    result = model_data.predict(features, prop_line=line)

                    if 'over_probability' in result:
                        over_prob = result['over_probability']
                    elif 'predicted_value' in result:
                        predicted_value = result['predicted_value']
                        std = get_prop_std_dev(prop_type)  # FIX: Use prop-specific std
                        z_score = (predicted_value - line) / std
                        over_prob = float(norm.cdf(z_score))

        except Exception:
            pass  # Fall through to return defaults

    # Phase 3: Minutes Oracle adjustment — per-minute rate scaling
    minutes_dist = None
    if predicted_value is not None and game_context and models.get('minutes_oracle'):
        minutes_dist = predict_minutes_distribution(
            player_id=player_id,
            team_id=team_id or 0,
            opponent_team_id=game_context.get('opponent_team_id', 0),
            game_context=game_context,
            models=models,
        )

        if minutes_dist:
            # Get player's historical average minutes from features
            avg_minutes = 0
            if features:
                avg_minutes = features.get('season_min_avg', 0) or features.get('recent_min_avg', 0) or 0

            predicted_minutes = minutes_dist.get('p50', avg_minutes)

            # Only adjust if we have meaningful baseline and prediction
            if avg_minutes > 10 and predicted_minutes > 0:
                rate = predicted_value / avg_minutes
                adjusted_value = rate * predicted_minutes

                # Only apply if adjustment is meaningful (>1% change)
                if abs(adjusted_value - predicted_value) / max(abs(predicted_value), 0.1) > 0.01:
                    predicted_value = adjusted_value

                    # Recalculate probability with adjusted value
                    std = get_prop_std_dev(prop_type)
                    z_score = (predicted_value - line) / std
                    over_prob = float(norm.cdf(z_score))

    # Apply injury-based adjustments to predicted value
    injury_boost_info = {'boost_factor': 1.0, 'reasons': []}
    if predicted_value is not None and HAS_INJURY_BOOST:
        try:
            # Calculate injury boost based on position, prop type, and injuries
            injury_boost_info = calculate_prop_injury_boost(
                player_position=player_position or 'G',  # Default to guard
                prop_type=prop_type,
                opponent_injured=opponent_injured or [],
                teammate_injured=teammate_injured or []
            )

            boost_factor = injury_boost_info.get('boost_factor', 1.0)

            # Apply boost to predicted value (capped at ±15%)
            if boost_factor != 1.0:
                adjusted_value = predicted_value * boost_factor
                predicted_value = adjusted_value

                # Recalculate probability with adjusted value
                std = get_prop_std_dev(prop_type)  # FIX: Use prop-specific std
                z_score = (predicted_value - line) / std
                over_prob = float(norm.cdf(z_score))
        except Exception:
            pass  # Continue without injury adjustment if it fails

    # Phase 5: Apply calibration bias corrections BEFORE edge computation
    calibration_applied = {}
    if HAS_CALIBRATION_ADJUSTER:
        try:
            adjuster = _get_calibration_adjuster()
            if adjuster:
                # Classify player tier from minutes
                _mins = minutes_dist.get('p50') if minutes_dist else None
                if _mins and _mins >= 32:
                    _player_tier = 'star'
                elif _mins and _mins >= 24:
                    _player_tier = 'starter'
                else:
                    _player_tier = 'role_player'

                # Classify position group
                _pos_group = None
                if player_position:
                    if player_position.upper() in ('PG', 'SG', 'G', 'G-F'):
                        _pos_group = 'guard'
                    elif player_position.upper() in ('C', 'C-F'):
                        _pos_group = 'center'
                    else:
                        _pos_group = 'forward'

                # Classify minutes bucket
                _min_bucket = None
                if _mins:
                    if _mins >= 30:
                        _min_bucket = 'starter'
                    elif _mins >= 20:
                        _min_bucket = 'rotation'
                    else:
                        _min_bucket = 'bench'

                calibration_applied = adjuster.apply_adjustments(
                    predicted_value=predicted_value,
                    confidence=0.0,  # Confidence not yet computed; adjusted later
                    prop_type=prop_type.lower(),
                    position=_pos_group,
                    minutes_bucket=_min_bucket,
                    player_tier=_player_tier,
                )

                # Apply value correction
                if calibration_applied.get('total_value_adjustment', 0) != 0:
                    predicted_value = calibration_applied['adjusted_value']
                    # Recalculate over_prob with corrected value
                    std = get_prop_std_dev(prop_type)
                    z_score = (predicted_value - line) / std
                    over_prob = float(norm.cdf(z_score))
        except Exception:
            pass  # Never block predictions on calibration failure

    # Phase 4: Single edge computation after all adjustments (minutes + injury + calibration)
    edge_info = _calculate_prop_edge(over_prob, american_odds)
    edge = edge_info['edge']
    pick = edge_info['pick']

    # Task 3.4: Add prediction bands using quantile models
    pred_low = None
    pred_median = None
    pred_high = None
    confidence_score = 50.0  # Default moderate confidence
    edge_quality_tier = 'moderate'
    suggested_bet_size = 0.0
    bet_recommendation = 'PASS'  # Phase 4: default to PASS

    # Try to get quantile predictions for better risk assessment
    quantile_model_dict = models.get(f'prop_{prop_type}_quantile')

    if quantile_model_dict and features and use_api_features:
        try:
            import pandas as pd

            # BUG FIX: The dict structure is {'model': QuantilePropModel, 'feature_names': [...]}
            # Extract the QuantilePropModel object from the dict
            quantile_model_obj = None
            if isinstance(quantile_model_dict, dict) and 'model' in quantile_model_dict:
                quantile_model_obj = quantile_model_dict['model']

            # Check if we have a QuantilePropModel with quantile_models attribute
            if quantile_model_obj and hasattr(quantile_model_obj, 'quantile_models'):
                quantile_models = quantile_model_obj.quantile_models
                scaler = getattr(quantile_model_obj, 'scaler', None)
                feature_names = getattr(quantile_model_obj, 'feature_names', [])
            # Legacy format: dict with quantile_models directly
            elif isinstance(quantile_model_dict, dict) and 'quantile_models' in quantile_model_dict:
                quantile_models = quantile_model_dict['quantile_models']
                scaler = quantile_model_dict.get('scaler')
                feature_names = quantile_model_dict.get('feature_names', [])
            else:
                # Unable to extract quantile models
                quantile_models = None
                feature_names = []

            if quantile_models and feature_names:

                # Build feature array
                X = pd.DataFrame([{k: features.get(k, 0) for k in feature_names}])
                X = X[feature_names].fillna(0)

                # Scale if scaler available
                X_scaled = scaler.transform(X) if scaler is not None else X.values

                # Get predictions from all quantile models
                # BUG FIX: Use correct quantile keys (0.1, 0.5, 0.9 not 0.10, 0.50, 0.90)
                pred_low = float(quantile_models[0.1].predict(X_scaled)[0])
                pred_median = float(quantile_models[0.5].predict(X_scaled)[0])
                pred_high = float(quantile_models[0.9].predict(X_scaled)[0])

                # Use median as the primary prediction if we don't have one yet
                if predicted_value is None:
                    predicted_value = pred_median
        except Exception:
            # Silently fall back to defaults if quantile prediction fails
            pass

    # Calculate confidence score based on prediction band width (Task 2.4)
    if pred_low is not None and pred_high is not None and pred_median is not None:
        band_width = pred_high - pred_low
        # STATISTICALLY CALIBRATED confidence multipliers (61K prediction backtest)
        # Calibration target: confidence score ≈ accuracy - 3% (conservative)
        # Analysis: THREES had 80% confidence but only 67% accuracy (worst calibration)
        #           REBOUNDS needed +0.9 adjustment to reduce dominance (was 84% of high-conf bets)
        #           ASSISTS/POINTS/PRA need more aggressive multipliers for better bet flow
        multipliers = {
            'assists': 4.9,    # 73.2% accuracy → 70% target confidence (well-calibrated)
            'points': 1.6,     # 71.7% accuracy → 68% target confidence (more aggressive for bet flow)
            'rebounds': 3.9,   # 71.7% accuracy → 68% target confidence (reduced from 3.0 to fix dominance)
            'threes': 8.4,     # 67.1% accuracy → 64% target confidence (MAJOR fix: was massively overconfident)
            'pra': 1.3,        # 71.4% accuracy → 68% target confidence (more aggressive for combo stat)
        }
        multiplier = multipliers.get(prop_type.lower(), 3.0)  # Default 3.0

        # Formula: confidence = 90 - (band_width * multiplier), clamped to [40, 90]
        confidence_score = max(40.0, min(90.0, 90.0 - (band_width * multiplier)))
    elif predicted_value is not None:
        # BUG FIX: More granular confidence based on edge magnitude and prediction difference
        # Higher edge and larger prediction difference = higher confidence
        pred_diff_pct = abs(predicted_value - line) / max(line, 1.0) if line > 0 else 0
        edge_magnitude = abs(edge)

        # Combine edge and prediction difference for confidence score (0-100 scale)
        # Strong edge (>20%) AND large diff (>20%) = high confidence (~80-90)
        # Weak edge (<5%) OR small diff (<5%) = low confidence (~40-50)
        confidence_from_edge = min(edge_magnitude * 2, 50.0)  # 0-50 from edge
        confidence_from_diff = min(pred_diff_pct * 200, 50.0)  # 0-50 from diff
        confidence_score = 50.0 + (confidence_from_edge + confidence_from_diff) / 2

        # Clamp to reasonable range [40, 90]
        confidence_score = max(40.0, min(90.0, confidence_score))

    # Phase 3: Adjust confidence based on minutes uncertainty
    if minutes_dist:
        uncertainty = minutes_dist.get('uncertainty', 'medium')
        if uncertainty == 'high':
            confidence_score *= 0.80  # 20% penalty for high minutes uncertainty
        elif uncertainty == 'medium':
            confidence_score *= 0.92  # 8% penalty for medium uncertainty
        # 'low' = no penalty
        confidence_score = max(40.0, min(90.0, confidence_score))

    # Phase 5: Apply calibration confidence adjustment
    if calibration_applied and calibration_applied.get('adjustments_applied'):
        try:
            adjuster = _get_calibration_adjuster()
            if adjuster:
                conf_result = adjuster.apply_adjustments(
                    predicted_value=0,  # Ignored — only want confidence multiplier
                    confidence=confidence_score,
                    prop_type=prop_type.lower(),
                    position=_pos_group if HAS_CALIBRATION_ADJUSTER else None,
                    minutes_bucket=_min_bucket if HAS_CALIBRATION_ADJUSTER else None,
                    player_tier=_player_tier if HAS_CALIBRATION_ADJUSTER else None,
                )
                confidence_score = conf_result['adjusted_confidence']
                confidence_score = max(40.0, min(90.0, confidence_score))
        except Exception:
            pass  # Never block predictions on calibration failure

    # Calculate edge quality tier based on confidence score (Task 2.4)
    edge_quality_tier = get_tier_from_confidence(confidence_score)

    # Calculate Kelly bet size (Task 3.4)
    if HAS_KELLY_SIZING and abs(edge) > 2.0:  # Only bet if edge > 2%
        try:
            # Convert edge to probability advantage
            # over_prob already accounts for our edge
            win_prob = over_prob if over_prob > 0.5 else (1 - over_prob)

            # Assume -110 odds (decimal 1.909)
            decimal_odds = 1.909

            # Use a default $1000 bankroll for bet sizing (user can scale)
            default_bankroll = 1000.0

            suggested_bet_size = calculate_kelly_bet_size(
                win_prob=win_prob,
                decimal_odds=decimal_odds,
                bankroll=default_bankroll,
                fractional=0.25,  # Quarter Kelly for safety
                edge_tier=edge_quality_tier,
                current_drawdown=0.0,  # Assume no drawdown for daily predictions
                num_same_day_bets=1,   # Conservative default
                max_bet_pct=0.05       # Cap at 5% of bankroll
            )

            # Calculate bet size as percentage of bankroll for display
            (suggested_bet_size / default_bankroll) * 100

        except Exception:
            # Fall back to defaults
            pass

    # Phase 4: Signal classification using edge magnitude
    bet_recommendation = get_signal_from_edge(edge, edge_info.get('edge_quality'))

    return {
        'player': player_name,
        'player_id': player_id,
        'stat': prop_type.upper(),
        'line': line,
        'over_prob': over_prob,
        'under_prob': 1.0 - over_prob,
        'edge': edge,
        'predicted_value': predicted_value,
        'pred_low': pred_low,
        'pred_median': pred_median,
        'pred_high': pred_high,
        'confidence_score': confidence_score,
        'edge_quality_tier': edge_quality_tier,
        'suggested_bet_size': suggested_bet_size,
        'bet_recommendation': bet_recommendation,
        'injury_boost': injury_boost_info.get('boost_factor', 1.0),
        'injury_reasons': injury_boost_info.get('reasons', []),
        'minutes_distribution': minutes_dist,
        'predicted_minutes': minutes_dist.get('p50') if minutes_dist else None,
        'minutes_uncertainty': minutes_dist.get('uncertainty') if minutes_dist else None,
        # Phase 4: Edge-focused fields
        'pick': pick,
        'over_edge': edge_info['over_edge'],
        'under_edge': edge_info['under_edge'],
        'edge_quality': edge_info['edge_quality'],
        'ev_per_dollar': edge_info['ev_per_dollar'],
        'implied_probability': edge_info['implied_probability'],
        'model_probability': edge_info['model_probability'],
        'has_edge': edge_info['has_edge'],
        'american_odds': american_odds,
        'signal': bet_recommendation,
        # Phase 5: Calibration feedback loop
        'calibration_adjustment': calibration_applied.get('total_value_adjustment', 0),
        'calibration_applied': bool(calibration_applied.get('adjustments_applied')),
    }


def get_starters_for_game(api: BalldontlieAPI, game: dict) -> dict[str, list[dict]]:
    """Get starters for both teams from recent games."""
    home_team = game.get('home_team', {})
    away_team = game.get('visitor_team', {})

    starters = {
        'home': [],
        'away': []
    }

    # Try to get from season averages - top 5 by minutes
    try:
        for team_key, team in [('home', home_team), ('away', away_team)]:
            team_id = team.get('id')
            if team_id:
                # Get team roster and stats
                players = api.get_players(team_ids=[team_id])
                if players:
                    # Just take first 5 as approximation of starters
                    starters[team_key] = players[:5]
    except Exception:
        pass

    return starters


def main():
    """Main entry point for daily predictions."""
    import argparse

    parser = argparse.ArgumentParser(description="Daily NBA Predictions")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    parser.add_argument("--no-warmup", action="store_true", help="Skip cache warmup")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")
    args = parser.parse_args()

    target_date = args.date or datetime.now(ET).strftime("%Y-%m-%d")

    # Clear cache if requested
    if args.clear_cache:
        removed = clear_cache()
        _player_feature_cache.clear()  # Also clear in-memory cache
        print(f"Cleared {removed} disk cache entries + in-memory cache")

    print("=" * 65)
    print("  NBA BETTING MODEL - Daily Predictions")
    print("=" * 65)
    print(f"  Date: {target_date}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load models
    print("\n  Loading models...")
    start_time = time.time()
    models = load_models()
    print(f"  Loaded: {list(models.keys())}")
    print(f"  Model loading time: {time.time() - start_time:.1f}s")

    # Initialize Balldontlie API
    try:
        api = BalldontlieAPI()
        print("  Balldontlie API: Connected (GOAT tier)")
    except Exception as e:
        print(f"  Balldontlie API: Error - {e}")
        api = None

    # Get today's games
    print("\n  Fetching games...")
    games = []
    odds_data = {}

    # Fetch current injuries BEFORE generating predictions (Task 1.4)
    print("\n  Fetching injury reports...")
    try:
        target_date_dt = datetime.strptime(target_date, "%Y-%m-%d")
        current_injuries = fetch_current_injuries(target_date_dt)

        # Build lookup dict: {player_id: status}
        injury_lookup = {}
        for injury_report in current_injuries:
            if injury_report.player_id:
                injury_lookup[injury_report.player_id] = injury_report.status

        # Print summary
        out_count = sum(1 for inj in current_injuries if inj.status == InjuryStatus.OUT)
        doubtful_count = sum(1 for inj in current_injuries if inj.status == InjuryStatus.DOUBTFUL)
        questionable_count = sum(1 for inj in current_injuries if inj.status == InjuryStatus.QUESTIONABLE)
        print(f"  Found {len(current_injuries)} injured players: {out_count} OUT, {doubtful_count} DOUBTFUL, {questionable_count} QUESTIONABLE")
    except Exception as e:
        print(f"  Warning: Could not fetch injury data: {e}")
        injury_lookup = {}
        current_injuries = []

    if api:
        try:
            games = api.get_games(dates=[target_date])
            print(f"  Found {len(games)} games")

            # Get betting odds (returns list of odds from multiple vendors)
            odds_list = api.get_betting_odds(date=target_date)
            print(f"  Fetched {len(odds_list)} odds entries")

            # Index by game_id, preferring FanDuel
            preferred_vendors = ['fanduel', 'draftkings', 'betmgm', 'caesars']
            for odds in odds_list:
                game_id = odds.get('game_id')
                vendor = odds.get('vendor', '').lower()
                if not game_id:
                    continue

                # Only store if no odds yet, or this is a preferred vendor
                if game_id not in odds_data or vendor in preferred_vendors:
                    odds_data[game_id] = odds
        except Exception as e:
            print(f"  Error fetching games: {e}")

    # Parse odds into game-indexed dict with best values
    if odds_data:
        parsed_odds = {}
        for odds in odds_data.values() if isinstance(odds_data, dict) else odds_data:
            game_id = odds.get('game_id')
            if not game_id:
                continue

            # Parse spread (convert string to float)
            spread_val = odds.get('spread_home_value', '0')
            try:
                spread = float(spread_val) if spread_val else 0
            except:
                spread = 0

            # Get moneylines
            home_ml = odds.get('moneyline_home_odds', -110)
            away_ml = odds.get('moneyline_away_odds', -110)

            # Store parsed odds
            if game_id not in parsed_odds:
                parsed_odds[game_id] = {
                    'spread': -spread,  # Convert to home perspective (negative = home favored)
                    'home_moneyline': home_ml,
                    'away_moneyline': away_ml,
                    'total': float(odds.get('total_value', '220') or '220'),
                    'vendor': odds.get('vendor', 'unknown')
                }

        odds_data = parsed_odds
        print(f"  Parsed odds for {len(parsed_odds)} games")

    if not games:
        # Fallback to NBA API
        print("  Falling back to NBA Live API...")
        try:
            from nba_api.live.nba.endpoints import scoreboard
            board = scoreboard.ScoreBoard()
            nba_games = board.games.get_dict()

            for g in nba_games:
                games.append({
                    'id': g.get('gameId'),
                    'home_team': {'abbreviation': g.get('homeTeam', {}).get('teamTricode', 'HOME')},
                    'visitor_team': {'abbreviation': g.get('awayTeam', {}).get('teamTricode', 'AWAY')},
                    'status': g.get('gameStatusText', ''),
                })
            print(f"  Found {len(games)} games via NBA API")
        except Exception as e:
            print(f"  Error: {e}")

    if not games:
        print("\n  No games found for today.")
        return

    # TASK 4.1: Cache warmup - pre-fetch all team/player data in parallel
    # Also cache props data to avoid duplicate API calls
    props_cache = {}  # Cache props data for reuse in main loop

    if not args.no_warmup and api:
        team_ids = []
        player_ids_to_warm = []

        # Collect team IDs
        for game in games:
            home_id = game.get('home_team', {}).get('id')
            away_id = game.get('visitor_team', {}).get('id')
            if home_id:
                team_ids.append(home_id)
            if away_id:
                team_ids.append(away_id)

        team_ids = list(set(team_ids))  # Remove duplicates

        # Collect player IDs from all games (quickly)
        # OPTIMIZATION: Cache props data here to avoid fetching twice
        print("\n  Collecting player IDs for cache warmup...", end='', flush=True)
        for game in games:
            game_id = game.get('id')
            if game_id:
                try:
                    props_data = get_player_props_for_game(api, game_id)
                    if props_data:
                        # Cache props for later use in main loop
                        props_cache[game_id] = props_data

                        # Get players with significant lines (likely to be analyzed)
                        for pid, props in props_data.items():
                            if props.get('points_line', 0) >= 15:
                                player_ids_to_warm.append(pid)
                except Exception:
                    pass  # Continue if one game fails

        player_ids_to_warm = list(set(player_ids_to_warm))  # Remove duplicates
        print(f" {len(player_ids_to_warm)} players")

        if team_ids or player_ids_to_warm:
            warmup_cache(api, target_date, team_ids, player_ids_to_warm)

    # Analyze each game
    print("\n" + "=" * 65)
    print("  GAME PREDICTIONS")
    print("=" * 65)

    all_analyses = []
    all_player_props = []

    for game in games:
        game_id = game.get('id')
        odds = odds_data.get(game_id, {})

        # Set default odds if not available
        if not odds:
            odds = {
                'home_moneyline': -110,
                'away_moneyline': -110,
                'spread': -3.0,  # Default home favored by 3
            }

        analysis = analyze_game(game, odds, models)

        # Fetch player props for this game (limit to top players with points props)
        if api and game_id:
            home = analysis['home_team']
            away = analysis['away_team']
            print(f"\n  Analyzing {away}@{home} props...", end="", flush=True)

            # OPTIMIZATION: Use cached props if available (from warmup phase)
            props_data = props_cache.get(game_id)
            if not props_data:
                # Fall back to API call if not in cache
                props_data = get_player_props_for_game(api, game_id)

            if props_data:
                # Get player names from API
                list(props_data.keys())

                # Filter to players with points lines > 15 (likely starters/key players)
                key_players = {pid: props for pid, props in props_data.items()
                              if props.get('points_line', 0) >= 15}

                # Limit to top 10 players by points line
                sorted_players = sorted(key_players.items(),
                                       key=lambda x: x[1].get('points_line', 0), reverse=True)[:10]

                if sorted_players:
                    # Build player name mapping using IDMapper (cached, fast)
                    player_names = {}
                    mapper = get_id_mapper()

                    # Get names for all players in props
                    for player_id, _ in sorted_players:
                        if mapper:
                            name = mapper.get_player_name(player_id)
                            if name:
                                player_names[player_id] = name

                    # Fallback: try Balldontlie API for any missing names
                    missing_ids = [pid for pid, _ in sorted_players if pid not in player_names]
                    if missing_ids:
                        try:
                            for pid in missing_ids:
                                p = api.get_player(pid)
                                if p:
                                    fname = p.get('first_name', '')
                                    lname = p.get('last_name', '')
                                    if fname or lname:
                                        player_names[pid] = f"{fname} {lname}".strip()
                        except:
                            pass

                    # Get team IDs for opponent context
                    home_team = game.get('home_team', {})
                    away_team = game.get('visitor_team', {})
                    home_team_id = home_team.get('id')
                    away_team_id = away_team.get('id')

                    # Get injured players for injury boost calculation
                    # Note: We use injury_tracker_v3 for primary injury checking (ID-based)
                    # but also extract names from injury_details for boost calculation
                    injury_details = analysis.get('injury_details', {})
                    home_injured_names = []
                    away_injured_names = []

                    for inj in injury_details.get('home', []):
                        status = inj.get('status', '').upper()
                        player_inj_name = inj.get('player_name', '')
                        if status in ('OUT', 'DOUBTFUL'):
                            home_injured_names.append(player_inj_name)

                    for inj in injury_details.get('away', []):
                        status = inj.get('status', '').upper()
                        player_inj_name = inj.get('player_name', '')
                        if status in ('OUT', 'DOUBTFUL'):
                            away_injured_names.append(player_inj_name)

                    # TASK 4.1: Prepare player prop prediction tasks for parallel execution
                    prop_tasks = []

                    for player_id, props in sorted_players:
                        player_name = player_names.get(player_id, f"Player {player_id}")
                        player_team_id = props.get('team_id')

                        # CHECK INJURY STATUS using injury_tracker_v3 (Task 1.4)
                        uncertainty_flag = None
                        if player_id in injury_lookup:
                            status = injury_lookup[player_id]
                            if status in [InjuryStatus.OUT, InjuryStatus.DOUBTFUL]:
                                # Skip prediction for OUT or DOUBTFUL players
                                print(f"    Skipping {player_name} ({status.value})")
                                continue
                            if status in [InjuryStatus.QUESTIONABLE, InjuryStatus.GTD]:
                                uncertainty_flag = "HIGH_UNCERTAINTY"

                        # Get player metadata
                        bdl_stats_id = None
                        player_position = 'G'
                        if player_name and not player_name.startswith("Player"):
                            bdl_stats_id = get_bdl_player_id(player_name)
                            if mapper and mapper._all_players:
                                for p in mapper._all_players:
                                    if p.get('id') == bdl_stats_id:
                                        player_position = p.get('position', 'G') or 'G'
                                        break

                        # Determine opponent/teammate injuries
                        if player_team_id == home_team_id:
                            opponent_id = away_team_id
                            opponent_abbrev = analysis['away_team']
                            opponent_injured = away_injured_names
                            teammate_injured = home_injured_names
                        else:
                            opponent_id = home_team_id
                            opponent_abbrev = analysis['home_team']
                            opponent_injured = home_injured_names
                            teammate_injured = away_injured_names

                        # Add tasks for each prop type
                        for prop_type in ['points', 'rebounds', 'assists']:
                            line_key = f'{prop_type}_line'
                            if line_key in props:
                                line = props[line_key]
                                prop_tasks.append({
                                    'player_name': player_name,
                                    'player_id': bdl_stats_id or player_id,
                                    'prop_type': prop_type,
                                    'line': line,
                                    'opponent': opponent_abbrev,
                                    'opponent_id': opponent_id,
                                    'player_position': player_position,
                                    'opponent_injured': opponent_injured,
                                    'teammate_injured': teammate_injured,
                                    'uncertainty_flag': uncertainty_flag,
                                    'team_id': player_team_id,
                                    'game_context': {
                                        'spread': odds.get('spread', 0),
                                        'total': odds.get('total', 220),
                                        'is_home': player_team_id == home_team_id,
                                        'is_b2b': False,
                                        'days_rest': 1,
                                        'opponent_team_id': opponent_id,
                                    },
                                })

                    # TASK 4.1: Execute prop predictions in parallel
                    if prop_tasks:
                        executor = get_executor(max_workers=10)

                        def process_prop_task(task):
                            pred = predict_player_prop(
                                task['player_name'],
                                task['player_id'],
                                task['prop_type'],
                                task['line'],
                                task['opponent'],
                                task['opponent_id'],
                                models,
                                use_api_features=True,
                                player_position=task['player_position'],
                                opponent_injured=task['opponent_injured'],
                                teammate_injured=task['teammate_injured'],
                                team_id=task.get('team_id'),
                                game_context=task.get('game_context'),
                            )
                            if task['uncertainty_flag']:
                                pred['uncertainty_flag'] = task['uncertainty_flag']
                            return pred

                        # Execute in parallel
                        prop_predictions = executor.map(
                            process_prop_task,
                            prop_tasks,
                            desc="Props",
                            show_progress=False
                        )

                        # Add to results
                        for pred in prop_predictions:
                            if pred:
                                analysis['player_props'].append(pred)
                                all_player_props.append({
                                    'game': f"{analysis['away_team']}@{analysis['home_team']}",
                                    **pred
                                })

                    # Show completion
                    prop_count = len(analysis['player_props'])
                    edges = [p['edge'] for p in analysis['player_props'] if abs(p['edge']) > 3]
                    print(f" {prop_count} props analyzed, {len(edges)} edges found")

        all_analyses.append(analysis)
        print_game_analysis(analysis)

    # Phase 4: Log predictions to calibration tracker
    if HAS_CALIBRATION and all_player_props:
        try:
            _cal_service = CalibrationService()
            logged_count = 0
            for prop in all_player_props:
                try:
                    game_str = prop.get('game', '')
                    opponent = ''
                    if '@' in game_str:
                        parts = game_str.split('@')
                        opponent = parts[0]  # away team abbreviation

                    _cal_service.log_prediction(
                        player_id=prop.get('player_id', 0),
                        player_name=prop.get('player', ''),
                        team='',
                        opponent=opponent,
                        game_date=target_date,
                        prop_type=prop.get('stat', ''),
                        predicted_value=prop.get('predicted_value') or 0,
                        prop_line=prop.get('line', 0),
                        predicted_over_prob=prop.get('over_prob'),
                        confidence=prop.get('confidence_score'),
                        edge=prop.get('edge'),
                        minutes_predicted=prop.get('predicted_minutes'),
                        minutes_uncertainty=prop.get('minutes_uncertainty'),
                        is_home=prop.get('game_context', {}).get('is_home') if isinstance(prop.get('game_context'), dict) else None,
                        spread=prop.get('game_context', {}).get('spread') if isinstance(prop.get('game_context'), dict) else None,
                        total=prop.get('game_context', {}).get('total') if isinstance(prop.get('game_context'), dict) else None,
                    )
                    logged_count += 1
                except Exception:
                    continue
            print(f"\n  Logged {logged_count} predictions to calibration tracker")
        except Exception as e:
            print(f"\n  Warning: Prediction logging failed: {e}")

    # Phase 4: Record BET/LEAN signals to bet tracker for CLV
    try:
        from nba_betting.edge.clv_bridge import record_predictions_as_bets
        bet_count = record_predictions_as_bets(all_player_props, target_date)
        if bet_count > 0:
            print(f"  Recorded {bet_count} BET/LEAN signals to bet tracker")
    except ImportError:
        pass  # CLV bridge not yet available
    except Exception as e:
        print(f"  Warning: CLV recording failed: {e}")

    # Summary of best bets
    print("\n" + "=" * 65)
    print("  TOP RECOMMENDATIONS (>5% edge)")
    print("=" * 65)

    recommendations = []

    for a in all_analyses:
        home = a['home_team']
        away = a['away_team']

        # Moneyline edges
        if a['moneyline']['home_edge'] > 5:
            recommendations.append({
                'game': f"{away}@{home}",
                'bet': f"{home} ML",
                'prob': a['moneyline']['home_prob'],
                'edge': a['moneyline']['home_edge']
            })
        if a['moneyline']['away_edge'] > 5:
            recommendations.append({
                'game': f"{away}@{home}",
                'bet': f"{away} ML",
                'prob': a['moneyline']['away_prob'],
                'edge': a['moneyline']['away_edge']
            })

        # Spread edges
        if a['spread']['edge_pct'] > 5:
            recommendations.append({
                'game': f"{away}@{home}",
                'bet': a['spread']['bet_side'],
                'prob': a['spread']['cover_prob'],
                'edge': a['spread']['edge_pct']
            })

        # Player prop edges — Phase 4: use pick and signal fields
        for prop in a.get('player_props', []):
            signal = prop.get('signal', prop.get('bet_recommendation', 'PASS'))
            if signal in ('BET', 'LEAN'):
                direction = prop.get('pick', 'OVER' if prop['over_prob'] > 0.5 else 'UNDER')
                prob = prop.get('model_probability', prop['over_prob'] if prop['over_prob'] > 0.5 else (1 - prop['over_prob']))
                recommendations.append({
                    'game': f"{away}@{home}",
                    'bet': f"{prop['player']} {prop['stat']} {direction} {prop['line']}",
                    'prob': prob,
                    'edge': prop['edge'],
                    'signal': signal,
                })

    # Sort by edge
    recommendations.sort(key=lambda x: x['edge'], reverse=True)

    if recommendations:
        print()
        for i, rec in enumerate(recommendations[:10], 1):
            print(f"  {i}. {rec['game']}: {rec['bet']} ({rec['prob']:.0%}, edge: {rec['edge']:+.1f}%)")
    else:
        print("\n  No strong edges found today.")

    # Task 3.4: Export predictions to CSV with enhanced columns
    if all_player_props:
        try:
            import pandas as pd
            csv_filename = f"predictions_{target_date}.csv"

            # Build DataFrame with all enhanced columns
            csv_data = []
            for prop in all_player_props:
                # Extract team from game string (format: "AWAY@HOME")
                game_str = prop.get('game', '')
                team = ''
                if '@' in game_str:
                    # We don't know which team this player is on from the prop alone
                    # We'll need to extract it from context or leave empty
                    # For now, leave empty - frontend doesn't strictly need it
                    team = ''

                # Phase 4: Use pick directly from prediction output
                over_prob = prop.get('over_prob', 0.5)
                bet_rec = prop.get('bet_recommendation', 'PASS')
                pick = prop.get('pick', 'OVER' if over_prob > 0.5 else 'UNDER')

                row = {
                    'date': target_date,
                    'game': game_str,
                    'player_name': prop.get('player', ''),
                    'team': team,  # Added for API compatibility
                    'prop_type': prop.get('stat', ''),
                    'line': prop.get('line', 0),
                    'prediction': prop.get('predicted_value', ''),
                    'pred_low': prop.get('pred_low', ''),
                    'pred_median': prop.get('pred_median', ''),
                    'pred_high': prop.get('pred_high', ''),
                    'over_prob': over_prob,
                    'under_prob': prop.get('under_prob', 1.0 - over_prob),
                    'edge': prop.get('edge', 0),
                    'over_edge': prop.get('over_edge', 0),
                    'under_edge': prop.get('under_edge', 0),
                    'confidence_score': prop.get('confidence_score', 50),
                    'edge_quality_tier': prop.get('edge_quality_tier', 'moderate'),
                    'edge_quality': prop.get('edge_quality', ''),
                    'suggested_bet_size': prop.get('suggested_bet_size', 0),
                    'bet_recommendation': bet_rec,
                    'signal': prop.get('signal', bet_rec),
                    'pick': pick,
                    'american_odds': prop.get('american_odds', -110),
                    'ev_per_dollar': prop.get('ev_per_dollar', 0),
                    'has_edge': prop.get('has_edge', False),
                    'uncertainty_flag': prop.get('uncertainty_flag', ''),
                    'injury_boost': prop.get('injury_boost', 1.0),
                }
                csv_data.append(row)

            df = pd.DataFrame(csv_data)
            # Fill NaN values with empty strings to prevent JSON serialization issues in API
            df = df.fillna('')
            df.to_csv(csv_filename, index=False)
            print(f"\n  Predictions saved to: {csv_filename}")
            print(f"  Total props: {len(all_player_props)}")

            # RAILWAY FIX: Also save to PostgreSQL database (persists across deployments)
            try:
                import psycopg2
                import os
                database_url = os.getenv("DATABASE_URL")

                if database_url:
                    print("\n  Saving to database...")
                    conn = psycopg2.connect(database_url)
                    cursor = conn.cursor()

                    # Create table if not exists
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS predictions_history (
                            id SERIAL PRIMARY KEY,
                            date DATE NOT NULL,
                            game VARCHAR(100),
                            player_name VARCHAR(100) NOT NULL,
                            team VARCHAR(10),
                            prop_type VARCHAR(20) NOT NULL,
                            prediction FLOAT NOT NULL,
                            pred_low FLOAT,
                            pred_median FLOAT,
                            pred_high FLOAT,
                            line FLOAT NOT NULL,
                            over_prob FLOAT,
                            edge FLOAT,
                            confidence_score FLOAT NOT NULL,
                            edge_quality_tier VARCHAR(20),
                            suggested_bet_size FLOAT,
                            bet_recommendation VARCHAR(20),
                            pick VARCHAR(10),
                            uncertainty_flag VARCHAR(50),
                            injury_boost BOOLEAN,
                            created_at TIMESTAMP DEFAULT NOW(),
                            UNIQUE(date, player_name, prop_type)
                        )
                    """)
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions_history(date)")
                    conn.commit()

                    # Delete existing predictions for this date
                    cursor.execute("DELETE FROM predictions_history WHERE date = %s", (target_date,))
                    deleted_count = cursor.rowcount
                    print(f"  Cleared {deleted_count} old predictions for {target_date}")

                    # Insert new predictions
                    inserted_count = 0
                    for _, row in df.iterrows():
                        def safe_val(val):
                            return None if pd.isna(val) or val == '' else val

                        cursor.execute("""
                            INSERT INTO predictions_history (
                                date, game, player_name, team, prop_type,
                                prediction, pred_low, pred_median, pred_high,
                                line, over_prob, edge, confidence_score,
                                edge_quality_tier, suggested_bet_size, bet_recommendation,
                                pick, uncertainty_flag, injury_boost
                            ) VALUES (
                                %s, %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s,
                                %s, %s, %s
                            )
                        """, (
                            target_date,
                            safe_val(row.get('game')),
                            row['player_name'],
                            safe_val(row.get('team')),
                            row['prop_type'],
                            row['prediction'],
                            safe_val(row.get('pred_low')),
                            safe_val(row.get('pred_median')),
                            safe_val(row.get('pred_high')),
                            row['line'],
                            safe_val(row.get('over_prob')),
                            safe_val(row.get('edge')),
                            row['confidence_score'],
                            safe_val(row.get('edge_quality_tier')),
                            safe_val(row.get('suggested_bet_size')),
                            safe_val(row.get('bet_recommendation')),
                            safe_val(row.get('pick')),
                            safe_val(row.get('uncertainty_flag')),
                            safe_val(row.get('injury_boost'))
                        ))
                        inserted_count += 1

                    conn.commit()
                    conn.close()
                    print(f"  ✓ Saved {inserted_count} predictions to database!")
                else:
                    print("  ℹ️ DATABASE_URL not set - skipping database save (CSV only)")
            except Exception as db_error:
                print(f"  ⚠️ Database save failed: {db_error}")
                print("     CSV file still available as fallback")

            # Show summary by recommendation
            if 'bet_recommendation' in df.columns:
                rec_counts = df['bet_recommendation'].value_counts()
                print(f"  Recommendations: {rec_counts.to_dict()}")
        except Exception as e:
            print(f"\n  Warning: Could not save CSV: {e}")

    print("\n" + "=" * 65)
    print("  Note: Always verify with actual sportsbook odds before betting.")
    print("=" * 65)


if __name__ == "__main__":
    main()
