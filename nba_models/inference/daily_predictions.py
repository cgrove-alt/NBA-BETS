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

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import load_env  # noqa: F401  — load .env before any code reads os.environ
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

# Keep the historical logger name for backward compatibility with older
# integration scripts and operational log filters.
logger = logging.getLogger("daily_predictions")

# Suppress noisy third-party loggers (but keep our own warnings visible)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Import our modules
from balldontlie_api import BalldontlieAPI
from feature_engineering import generate_game_features, PlayerPropFeatureGenerator, InjuryReportManager, filter_features
from scipy.stats import norm
from data_fetcher import fetch_player_stats_bdl
from injury_tracker_v3 import (
    fetch_current_injuries as _fetch_current_injuries,
    is_player_available,
    InjuryStatus as _RawInjuryStatus,
)


class _CompatInjuryStatus(str):
    @property
    def value(self) -> str:
        return str(self)


class InjuryStatus:
    OUT = _CompatInjuryStatus(_RawInjuryStatus.OUT)
    DOUBTFUL = _CompatInjuryStatus(_RawInjuryStatus.DOUBTFUL)
    QUESTIONABLE = _CompatInjuryStatus(_RawInjuryStatus.QUESTIONABLE)
    GTD = _CompatInjuryStatus(_RawInjuryStatus.GTD)
    PROBABLE = _CompatInjuryStatus(_RawInjuryStatus.PROBABLE)
    ACTIVE = _CompatInjuryStatus(_RawInjuryStatus.ACTIVE)


def fetch_current_injuries(target_date=None):
    """Backward-compatible wrapper around the current-injury fetcher."""
    _ = target_date
    return _fetch_current_injuries()

# Required for pickle deserialization of quantile models
from nba_models.models.model_trainer import QuantilePropModel  # noqa: F401

# Canonical constants — single source of truth across the entire pipeline.
# Do NOT redefine PROP_STD_DEVS or quantile defaults locally; update nba_betting/constants.py.
from nba_betting.constants import (
    PROP_STD_DEVS,
    DEFAULT_PROP_STD_DEV,
    PROP_BIAS_CORRECTION,
    QUANTILE_DECOMPRESSION_DEFAULTS,
    QUANTILE_TARGET_SLOPE,
    PROB_CLAMP_MIN,   # Phase 1.1: probability safety floor (0.05)
    PROB_CLAMP_MAX,   # Phase 1.1: probability safety ceiling (0.95)
    SPREAD_BETTING_ENABLED,
    SPREAD_AS_ML_FEATURE,
)
from nba_models.models.model_classes import smart_fillna
from nba_models.inference.model_compat import (
    get_context_feature_names,
    get_feature_names,
    predict_binary_probability,
    predict_regression_value,
    prepare_loaded_model_artifact,
)

# ---------------------------------------------------------------------------
# Phase 3.3: Poisson probability + regression-to-mean for threes
# ---------------------------------------------------------------------------
try:
    from nba_models.models.poisson_prop_model import (
        compute_poisson_over_prob,
        detect_threes_streak,
    )
    from nba_betting.prop_config import get_prop_config as _get_prop_config
    HAS_POISSON_MODEL = True
except Exception as _pm_err:
    HAS_POISSON_MODEL = False
    logger.debug("Poisson model unavailable: %s", _pm_err)

# Phase 3.2: Dynamic ensemble weighting + per-model performance tracker
# ---------------------------------------------------------------------------
try:
    from nba_models.ensemble.dynamic_weighting import DynamicEnsembleWeighter
    from nba_models.ensemble.model_performance_tracker import ModelPerformanceTracker
    _WEIGHTER_PATH = Path(__file__).resolve().parents[2] / "data" / "model_performance" / "ensemble_weights.json"
    _ENSEMBLE_WEIGHTER: DynamicEnsembleWeighter = DynamicEnsembleWeighter.load(_WEIGHTER_PATH)
    _PERF_TRACKER: ModelPerformanceTracker = ModelPerformanceTracker()
    HAS_DYNAMIC_WEIGHTING = True
except Exception as _dw_err:
    HAS_DYNAMIC_WEIGHTING = False
    _ENSEMBLE_WEIGHTER = None  # type: ignore[assignment]
    _PERF_TRACKER = None  # type: ignore[assignment]
    logger.debug("Dynamic ensemble weighting unavailable: %s", _dw_err)


# ---------------------------------------------------------------------------
# Empirical probability calibration (Phase 4)
# ---------------------------------------------------------------------------
_CALIBRATORS: dict = {}


def _load_calibrator(prop_type: str):
    """Load cached isotonic calibrator for a prop type."""
    if prop_type in _CALIBRATORS:
        return _CALIBRATORS[prop_type]
    cal_dir = Path(__file__).resolve().parent.parent.parent / "models" / "probability_calibrators"
    pkl_path = cal_dir / f"{prop_type}_isotonic.pkl"
    json_path = cal_dir / f"{prop_type}_lookup.json"
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                _CALIBRATORS[prop_type] = ("pkl", pickle.load(f))
                return _CALIBRATORS[prop_type]
        except Exception:
            logger.warning("Failed to load isotonic calibrator for %s", prop_type)
    if json_path.exists():
        try:
            import json as _json
            with open(json_path) as f:
                _CALIBRATORS[prop_type] = ("json", _json.load(f))
                return _CALIBRATORS[prop_type]
        except Exception:
            logger.warning("Failed to load JSON calibrator for %s", prop_type)
    _CALIBRATORS[prop_type] = None
    return None


def apply_empirical_calibration(over_prob: float, prop_type: str) -> float:
    """Apply empirical isotonic calibration to raw over_prob.

    Phase 1.1: ALL outputs are clamped to [PROB_CLAMP_MIN, PROB_CLAMP_MAX]
    (i.e. [0.05, 0.95]) to prevent degenerate probabilities (0.0 or 1.0)
    from reaching the Kelly sizing formula, which would result in either
    zero bet size or infinite sizing recommendation.
    """
    # Phase 1.1: clamp input before calibration to avoid extrapolation issues
    over_prob = float(np.clip(over_prob, PROB_CLAMP_MIN, PROB_CLAMP_MAX))

    cal = _load_calibrator(prop_type.lower())
    if cal is None:
        return over_prob
    kind, obj = cal
    if kind == "pkl":
        try:
            calibrated = float(obj.predict([over_prob])[0])
        except Exception:
            logger.warning("Isotonic predict failed for %s", prop_type)
            calibrated = over_prob
    elif kind == "json":
        # JSON lookup table at 1% increments
        pct = max(1, min(99, int(round(over_prob * 100))))
        calibrated = float(obj.get(str(pct), over_prob))
    else:
        calibrated = over_prob

    # Phase 1.1: clamp output — isotonic regression can produce boundary values
    return float(np.clip(calibrated, PROB_CLAMP_MIN, PROB_CLAMP_MAX))


def load_quantile_decompression() -> dict:
    """Load quantile decompression constants from model artifact, falling back to defaults.

    Looks for models/quantile_decompression.json alongside the model files.
    Falls back to QUANTILE_DECOMPRESSION_DEFAULTS (from nba_betting.constants) if the file
    is not found or is unreadable.

    Re-run the calibration script and regenerate the JSON after every model retrain:
        python3 scripts/calibrate_quantile_decompression.py
    """
    import json
    from pathlib import Path
    filepath = Path("models/quantile_decompression.json")
    if filepath.exists():
        try:
            with open(filepath) as f:
                data = json.load(f)
            # Validate that the loaded data has the expected structure and is not stale placeholders
            expected_keys = {'points', 'rebounds', 'assists', 'threes', 'pra'}
            if expected_keys.issubset(data.keys()):
                # Detect stale placeholder: all slopes identical at 0.7 is the old default
                slopes = [data[k].get('slope', 0) for k in expected_keys]
                if len(set(slopes)) == 1 and slopes[0] == 0.7:
                    logger.warning(
                        "quantile_decompression.json contains stale placeholder values "
                        "(slope=0.7 for all props). Falling back to calibrated defaults. "
                        "Run: python3 scripts/calibrate_quantile_decompression.py"
                    )
                else:
                    return data
        except Exception as exc:
            logger.warning("Failed to load quantile_decompression.json: %s. Using defaults.", exc)
    return QUANTILE_DECOMPRESSION_DEFAULTS


QUANTILE_DECOMPRESSION = load_quantile_decompression()

def get_prop_std_dev(prop_type: str) -> float:
    """
    Get empirically-derived standard deviation for prop type.

    These values are based on NBA historical data analysis and represent
    the typical game-to-game variance for each stat category.

    Returns:
        float: Standard deviation for calculating Z-scores
    """
    return PROP_STD_DEVS.get(prop_type.lower(), DEFAULT_PROP_STD_DEV)


def compute_quantile_sigma(pred_low: float, pred_high: float, prop_type: str) -> float:
    """Derive player-specific sigma from quantile model's 10th-90th percentile spread.

    For normal distribution: P90 - P10 = 2 * 1.282 * sigma = 2.564 * sigma.
    Floor at 50% of fixed PROP_STD_DEVS to prevent overconfidence.
    """
    spread = pred_high - pred_low
    if spread <= 0:
        return get_prop_std_dev(prop_type)
    quantile_sigma = spread / 2.564
    min_sigma = get_prop_std_dev(prop_type) * 0.75  # Prevent overconfidence
    max_sigma = get_prop_std_dev(prop_type) * 1.50  # Prevent under-confidence
    return min(max(quantile_sigma, min_sigma), max_sigma)


def decompress_quantile_prediction(predicted_value: float, line: float, prop_type: str, player_season_avg: float = None) -> float:
    """Correct regression-to-mean compression in quantile model predictions.

    The quantile models predict with slope < 1.0 relative to the line, meaning
    they under-predict high-line players and over-predict low-line players. This
    is especially severe for POINTS (slope=0.724 → high scorers predicted 3-9
    points too low).

    Applies two corrections:
    1. Slope fix: adds back missing variation around the mean line
    2. Level fix: shifts the overall prediction level to center on the line

    These parameters are measured from production predictions and should be
    re-measured after every model retrain.
    """
    params = QUANTILE_DECOMPRESSION.get(prop_type.lower())
    if not params:
        return predicted_value

    current_slope = params['slope']
    mean_gap = params['mean_gap']
    mean_line = params['mean_line']

    # Skip correction if slope is already close to target
    slope_fix = QUANTILE_TARGET_SLOPE - current_slope
    if abs(slope_fix) < 0.01 and abs(mean_gap) < 0.1:
        return predicted_value

    level_fix = -mean_gap
    # Use player's season average as anchor when available (more accurate than
    # universal mean_line). Falls back to mean_line if no player-specific anchor.
    anchor = player_season_avg if player_season_avg is not None else mean_line
    return predicted_value + slope_fix * (line - anchor) + level_fix


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

# Fix 5.3: Critical dependencies — import with clear error messages instead of
# silent degradation. If these fail, the module still loads but logs a warning.
try:
    from edge_calculator import EdgeCalculator
    HAS_EDGE_CALCULATOR = True
except ImportError:
    HAS_EDGE_CALCULATOR = False
    logger.error(
        "CRITICAL: edge_calculator not importable. Edge calculations will use "
        "legacy fallback. Install edge_calculator or check PYTHONPATH."
    )

try:
    from calibration_tracker import CalibrationService
    HAS_CALIBRATION = True
except (ImportError, TypeError):
    HAS_CALIBRATION = False
    logger.error("CRITICAL: calibration_tracker not importable. Prediction logging disabled.")

try:
    from calibration_tracker import CalibrationAdjuster, CalibrationDatabase
    HAS_CALIBRATION_ADJUSTER = True
except (ImportError, TypeError):
    HAS_CALIBRATION_ADJUSTER = False

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

# Fix 5.3: Kelly sizing and bet filter — required but with fallback
try:
    from risk_management import calculate_kelly_bet_size, get_kelly_multiplier_for_tier
    HAS_KELLY_SIZING = True
except ImportError:
    HAS_KELLY_SIZING = False
    logger.error("CRITICAL: risk_management not importable. Kelly sizing disabled.")
    def calculate_kelly_bet_size(*args, **kwargs):
        return 0.0
    def get_kelly_multiplier_for_tier(*args, **kwargs):
        return 0.0

try:
    from nba_betting.bet_filter import should_bet as _should_bet, calculate_bet_size as _calc_bet_size
    from nba_betting.prediction_pipeline import evaluate_bet as _evaluate_bet
    HAS_BET_FILTER = True
except ImportError:
    HAS_BET_FILTER = False
    logger.error("CRITICAL: bet_filter/prediction_pipeline not importable. Bet filtering disabled.")
    def _should_bet(*args, **kwargs):
        return True, 'filter unavailable', 0.0

# Phase 4 Odds Integration: prop odds tracker for line movement signals
try:
    from nba_betting.odds.prop_odds_tracker import PropOddsTracker, get_prop_tracker
    HAS_PROP_TRACKER = True
except ImportError:
    HAS_PROP_TRACKER = False
    logger.warning("PropOddsTracker not importable — line movement signals disabled.")
    def _calc_bet_size(*args, **kwargs):
        return 0.0
    def _evaluate_bet(*args, **kwargs):
        return {'should_bet': True, 'tier': 'moderate', 'reason': 'filter unavailable'}

# Helper function to map confidence + edge to quality tier
def get_edge_quality_tier(confidence_score: float, edge: float) -> str:
    """Map confidence score (0-100) + edge magnitude to edge quality tier.

    Thresholds verified against the over_prob confidence formula (Phase 1 fix, 2026-03-31).
    With confidence = 40 + abs(over_prob - 0.5) * 100:
      confidence >= 65  →  over_prob >= 0.75  (meaningful model lean)
      confidence >= 70  →  over_prob >= 0.80  (strong model lean)
      confidence >= 75  →  over_prob >= 0.85  (very strong model lean)
    These thresholds correctly gate elite/strong tiers on high-probability predictions.
    """
    abs_edge = abs(edge)
    if abs_edge >= 20 and confidence_score >= 75:
        return 'elite'
    if abs_edge >= 12 and confidence_score >= 70:
        return 'strong'
    if abs_edge >= 8 and confidence_score >= 65:
        return 'moderate'
    if confidence_score >= 75 and abs_edge >= 5:
        return 'strong'
    if confidence_score >= 65 and abs_edge >= 5:
        return 'moderate'
    if abs_edge >= 5 and confidence_score >= 55:
        return 'weak'
    return 'avoid'


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds >= 100:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1

# Phase 4: Edge-focused prop edge calculation
def _calculate_prop_edge(over_prob: float, american_odds: int = -110, under_odds: int = None) -> dict:
    """
    Calculate edge for both OVER and UNDER sides of a prop bet.

    Uses EdgeCalculator when available, falls back to legacy formula.
    When both over_odds and under_odds are provided, uses no-vig devigging
    to compute edges against the true (vig-free) market probabilities.

    Args:
        over_prob: Model's probability of the OVER hitting (0-1)
        american_odds: American odds for the over (default -110)
        under_odds: American odds for the under (optional; enables devigging)

    Returns:
        Dict with over_edge, under_edge, pick, edge, edge_quality, ev_per_dollar,
        implied_probability, model_probability, has_edge
    """
    under_prob = 1.0 - over_prob

    if HAS_EDGE_CALCULATOR:
        calc = EdgeCalculator(min_edge_threshold=0.02)

        # Use devigged probabilities as the implied benchmark when both sides are known
        if under_odds is not None and under_odds != american_odds:
            try:
                from edge_calculator import devig_probability
                nv_over, nv_under = devig_probability(american_odds, under_odds)
                # Compute edge vs no-vig probability directly
                over_edge_val = (over_prob - nv_over) * 100
                under_edge_val = (under_prob - nv_under) * 100

                if over_edge_val >= under_edge_val:
                    pick = 'OVER'
                    edge = over_edge_val
                    best_odds = american_odds
                    model_prob = over_prob
                    implied_prob = nv_over
                else:
                    pick = 'UNDER'
                    edge = under_edge_val
                    best_odds = under_odds
                    model_prob = under_prob
                    implied_prob = nv_under

                # Compute EV using decimal odds
                decimal_odds = calc.american_to_decimal(best_odds)
                ev_per_dollar = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
                has_edge = edge / 100 >= calc.min_edge_threshold
                edge_quality = calc.classify_edge(edge / 100)

                return {
                    'over_edge': over_edge_val,
                    'under_edge': under_edge_val,
                    'pick': pick,
                    'edge': edge,
                    'edge_quality': edge_quality,
                    'ev_per_dollar': ev_per_dollar,
                    'implied_probability': implied_prob,
                    'model_probability': model_prob,
                    'has_edge': has_edge,
                }
            except Exception:
                pass  # Fall through to EdgeCalculator path

        over_result = calc.calculate_edge(over_prob, american_odds)
        _under_odds = under_odds if under_odds is not None else american_odds
        under_result = calc.calculate_edge(under_prob, _under_odds)

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
        # Legacy fallback: devig both sides using the actual odds provided
        def _american_to_raw(odds):
            if odds > 0:
                return 100 / (odds + 100)
            return abs(odds) / (abs(odds) + 100)

        def _american_to_decimal(odds):
            if odds > 0:
                return 1 + odds / 100
            return 1 + 100 / abs(odds)

        if under_odds is not None:
            # Proper devig when both sides are available
            raw_over_implied = _american_to_raw(american_odds)
            raw_under_implied = _american_to_raw(under_odds)
            total_implied = raw_over_implied + raw_under_implied
            if total_implied > 0:
                nv_over_implied = raw_over_implied / total_implied
                nv_under_implied = raw_under_implied / total_implied
            else:
                nv_over_implied = 0.5
                nv_under_implied = 0.5
            over_edge = (over_prob - nv_over_implied) * 100
            under_edge = (under_prob - nv_under_implied) * 100
        else:
            # Single-side fallback: use raw implied probability of the over side
            over_implied = _american_to_raw(american_odds)
            nv_over_implied = over_implied
            nv_under_implied = 1.0 - over_implied
            over_edge = (over_prob - over_implied) * 100
            under_edge = (under_prob - (1.0 - over_implied)) * 100

        if over_edge >= under_edge:
            pick = 'OVER'
            edge = over_edge
            model_prob = over_prob
            market_implied = nv_over_implied
            best_odds = american_odds
        else:
            pick = 'UNDER'
            edge = under_edge
            model_prob = under_prob
            market_implied = nv_under_implied
            best_odds = under_odds if under_odds is not None else american_odds

        # Proper EV: (model_prob * decimal_odds) - 1
        # When best_odds is available, use actual decimal odds.
        # When we only have over_odds but picked UNDER, estimate decimal from implied prob.
        if best_odds is not None and (under_odds is not None or pick == 'OVER'):
            decimal_odds = _american_to_decimal(best_odds)
        elif market_implied > 0 and market_implied < 1:
            # Estimate decimal odds from no-vig implied probability
            decimal_odds = 1.0 / market_implied
        else:
            decimal_odds = 1.909  # Default -110
        ev_per_dollar = (model_prob * decimal_odds) - 1

        return {
            'over_edge': over_edge,
            'under_edge': under_edge,
            'pick': pick,
            'edge': edge,
            'edge_quality': 'elite' if edge >= 20 else 'strong' if edge >= 12 else 'moderate' if edge >= 6 else 'weak' if edge >= 3 else 'avoid',
            'ev_per_dollar': ev_per_dollar,
            'implied_probability': market_implied,
            'model_probability': model_prob,
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
    if edge_quality == 'elite':
        return 'BET'
    if edge_quality == 'strong' and edge >= 8:
        return 'BET'
    if edge_quality in ('strong', 'moderate'):
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


def fetch_team_tracking_data(team_id: int, n_games: int = 3) -> tuple['ShotAtlas | None', 'RotationTracker | None']:
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
    opp_stats: dict = None,
    # Phase 2.1: Opponent schedule context
    opp_days_rest: int = 2,
    opp_is_b2b: bool = False,
    opp_def_tier: int = 2,
    # Phase 2.2: Player team game context
    travel_distance: float = 0.0,
    games_last_7: int = 3,
    season_phase: int = 1,
    is_b2b_home: bool = False,
    is_b2b_away: bool = False,
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
                # Phase 2.3: offensive/defensive rebound splits
                'oreb': game.get('oreb', 0) or 0,
                'dreb': game.get('dreb', 0) or 0,
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

    # Add opponent features — use real stats when opp_stats provided, else league avg
    _opp = opp_stats or {}
    _def_rating = _opp.get('def_rating', 114.0)
    _off_rating = _opp.get('off_rating', 114.0)
    _pace = _opp.get('pace', 100.0)
    _pts_allowed = _opp.get('pts_allowed', 114.0)
    _win_pct = _opp.get('win_pct', 0.5)

    # Scale position-specific stats by opponent defense relative to league average
    _def_scale = _def_rating / 114.0 if _def_rating > 0 else 1.0

    opponent_features = {
        'opp_def_rating': _def_rating,
        'opp_off_rating': _off_rating,
        'opp_net_rating': _off_rating - _def_rating,
        'opp_pts_allowed': _pts_allowed,
        'opp_pts_allowed_recent': _opp.get('pts_allowed_recent', _pts_allowed),
        'opp_pts_allowed_std': _opp.get('pts_allowed_std', 8.0),
        'opp_pace': _pace,
        'opp_pace_season': _opp.get('pace_season', _pace),
        'opp_def_strength': (_def_rating - 114.0) / 114.0,
        'opp_location_def': 0.0,
        'opp_win_pct': _win_pct,
        'opp_recent_win_pct': _opp.get('recent_win_pct', _win_pct),
        'is_home': 1 if is_home else 0,
        'team_pace': _opp.get('team_pace', _pace),
        'team_off_rating': _opp.get('team_off_rating', 114.0),
        # Position-specific: scale league averages by opponent defensive strength
        'opp_pts_allowed_to_guards': 18.0 * _def_scale,
        'opp_reb_allowed_to_guards': 3.5 * _def_scale,
        'opp_ast_allowed_to_guards': 5.5 * _def_scale,
        'opp_fg3m_allowed_to_guards': 2.0 * _def_scale,
        'opp_pts_allowed_to_forwards': 16.0 * _def_scale,
        'opp_reb_allowed_to_forwards': 6.5 * _def_scale,
        'opp_ast_allowed_to_forwards': 3.0 * _def_scale,
        'opp_fg3m_allowed_to_forwards': 1.5 * _def_scale,
        'opp_pts_allowed_to_centers': 14.0 * _def_scale,
        'opp_reb_allowed_to_centers': 9.0 * _def_scale,
        'opp_ast_allowed_to_centers': 2.5 * _def_scale,
        'opp_fg3m_allowed_to_centers': 0.5 * _def_scale,
        'opp_pts_vs_pos_std': 3.0,
        'opp_pts_vs_pos_diff': (_def_rating - 114.0) * 0.15,
        'opp_reb_vs_pos_diff': (_def_rating - 114.0) * 0.05,
        'opp_ast_vs_pos_diff': (_def_rating - 114.0) * 0.04,
        'opp_fg3m_vs_pos_diff': (_def_rating - 114.0) * 0.02,
        # Opponent-adjusted factors based on real defensive strength
        'opp_pts_allowed_avg': 15.0 * _def_scale,
        'opp_reb_allowed_avg': 5.5 * _def_scale,
        'opp_ast_allowed_avg': 3.5 * _def_scale,
        'opp_pts_factor': _def_scale,
        'opp_reb_factor': _def_scale,
        'opp_ast_factor': _def_scale,
        # Phase 2.1: Opponent schedule / fatigue
        'opp_days_rest': opp_days_rest,
        'opp_is_back_to_back': 1 if opp_is_b2b else 0,
        'opp_b2b_home': 0,
        'opp_b2b_away': 1 if (opp_is_b2b and not is_home) else 0,
        'opp_def_tier': opp_def_tier,
        'opp_is_elite_defense': 1 if opp_def_tier == 1 else 0,
        'opp_is_weak_defense': 1 if opp_def_tier == 3 else 0,
        # Phase 2.2: Game context
        'travel_distance': travel_distance,
        'games_last_7_days': games_last_7,
        'is_high_schedule_load': 1 if games_last_7 >= 4 else 0,
        'is_long_travel': 1 if travel_distance >= 1500 else 0,
        'season_phase': season_phase,
        'is_late_season': 1 if season_phase >= 2 else 0,
        'is_b2b_home': 1 if is_b2b_home else 0,
        'is_b2b_away': 1 if is_b2b_away else 0,
        # Phase 2.3: Opponent rebounding context
        'opp_oreb_pct': _opp.get('oreb_pct', 0.245),
        'opp_dreb_pct': _opp.get('dreb_pct', 0.755),
        'opp_oreb_avg': _opp.get('oreb_avg', 8.5),
        'opp_dreb_avg': _opp.get('dreb_avg', 26.0),
    }
    base_features.update(opponent_features)

    # Fill rest_advantage_vs_opp after days_rest is known from base_features
    _player_days_rest = base_features.get('days_rest', 2)
    base_features['rest_advantage_vs_opp'] = float(_player_days_rest - opp_days_rest)

    # Add pace-adjusted features with real pace data
    _team_pace = _opp.get('team_pace', _pace)
    _opp_pace = _pace
    try:
        pace_features = calculate_pace_adjusted_features(
            player_features=base_features,
            team_pace=_team_pace,
            opponent_pace=_opp_pace,
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
                        vegas_total: float = None, opp_stats: dict = None,
                        # Phase 2 context
                        opp_days_rest: int = 2, opp_is_b2b: bool = False,
                        opp_def_tier: int = 2,
                        travel_distance: float = 0.0, games_last_7: int = 3,
                        season_phase: int = 1, is_b2b_home: bool = False,
                        is_b2b_away: bool = False) -> dict:
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
                opp_stats=opp_stats,
                # Phase 2: game context passthrough
                opp_days_rest=opp_days_rest,
                opp_is_b2b=opp_is_b2b,
                opp_def_tier=opp_def_tier,
                travel_distance=travel_distance,
                games_last_7=games_last_7,
                season_phase=season_phase,
                is_b2b_home=is_b2b_home,
                is_b2b_away=is_b2b_away,
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

    # Moneyline - prefer ensemble (has feature_names + predict_proba), then stacking, then metalearner
    ml_path = MODEL_DIR / "moneyline_ensemble.pkl"
    if not ml_path.exists():
        ml_path = MODEL_DIR / "moneyline_stacking.pkl"
    if not ml_path.exists():
        ml_path = MODEL_DIR / "moneyline_stacking_metalearner.pkl"
    if ml_path.exists():
        try:
            with open(ml_path, 'rb') as f:
                data = pickle.load(f)
                models['moneyline'] = prepare_loaded_model_artifact(data)
            print(f"    Loaded moneyline from {ml_path.name}")
        except Exception as e:
            print(f"    Warning: Could not load moneyline model: {e}")

    # Spread - prefer ensemble (has feature_names + scaler), then stacking, then metalearner
    spread_path = MODEL_DIR / "spread_ensemble.pkl"
    if not spread_path.exists():
        spread_path = MODEL_DIR / "spread_stacking.pkl"
    if not spread_path.exists():
        spread_path = MODEL_DIR / "spread_stacking_metalearner.pkl"
    if spread_path.exists():
        try:
            with open(spread_path, 'rb') as f:
                data = pickle.load(f)
                models['spread'] = prepare_loaded_model_artifact(data)
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
                print("    Loaded Minutes Oracle")
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
    prop_features: dict = None,
) -> dict:
    """Predict minutes distribution using the Minutes Oracle.

    Returns dict with p10/p25/p50/p75/p90/expected/uncertainty/spread,
    or a fallback dict based on historical average if oracle unavailable.

    Args:
        prop_features: Feature dict from the prop prediction pipeline. Used to
            override the oracle's baseline features (season_min_avg, recent_min_avg,
            etc.) with real player data. Without this, the oracle uses hardcoded
            defaults and systematically under-predicts minutes.
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

        # Override baseline features with real player data from the prop pipeline.
        # The feature generator defaults to season_min_avg=28, games_played=0 when
        # no game logs are passed, which causes the oracle to systematically
        # under-predict minutes for all players.
        if prop_features:
            season_avg = prop_features.get('season_min_avg', 0)
            recent_avg = prop_features.get('recent_min_avg', 0)

            if season_avg > 0:
                features['season_min_avg'] = season_avg
                features['recent_min_avg'] = recent_avg or season_avg
                features['last3_min_avg'] = recent_avg or season_avg
                features['min_trend'] = (recent_avg - season_avg) if recent_avg > 0 else 0.0
                features['min_floor'] = season_avg - 6
                features['min_ceiling'] = season_avg + 6
                features['min_consistency'] = 0.85  # Known player with real data
                features['games_played'] = 50  # Signal that this is a real player

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


def preserve_model_context_features(
    filtered_features: dict,
    raw_features: dict,
    model: object | None,
) -> dict:
    """Reattach stacking context fields removed by feature-selection gating."""
    if not model:
        return filtered_features

    preserved = dict(filtered_features)
    for name in get_context_feature_names(model):
        candidates = [name]
        if name.startswith("ctx_"):
            candidates.append(name[4:])
        else:
            candidates.append(f"ctx_{name}")

        for candidate in candidates:
            if candidate in raw_features:
                preserved[candidate] = raw_features[candidate]
                break
    return preserved


def predict_moneyline(features: dict, models: dict) -> tuple[float, float]:
    """Predict moneyline probabilities."""
    model = models.get('moneyline')
    if not model:
        return 0.5, 0.5

    try:
        model_features = get_feature_names(model)
        if model_features:
            non_zero = sum(1 for col in model_features if features.get(col, 0))
            if non_zero < len(model_features) * 0.3:
                print(f"    WARNING: moneyline model has {non_zero}/{len(model_features)} non-zero features")

            home_prob = predict_binary_probability(model, features)
            if home_prob is None:
                raise ValueError("unsupported moneyline artifact format")
        else:
            print("    WARNING: moneyline model has no feature_names, using fallback")
            net_rating_diff = features.get('net_rating_diff', 0)
            home_prob = 0.5 + (net_rating_diff * 0.02)
    except Exception as e:
        print(f"    WARNING: moneyline model predict failed: {type(e).__name__}: {e}")
        net_rating_diff = features.get('net_rating_diff', 0)
        home_prob = 0.5 + (net_rating_diff * 0.02)

    home_prob = max(0.1, min(0.9, home_prob))
    return home_prob, 1 - home_prob


def predict_spread(features: dict, models: dict) -> float:
    """Predict point spread (positive = home favored)."""
    model = models.get('spread')

    # Heuristic fallback
    net_rating_diff = features.get('net_rating_diff', 0)
    home_advantage = 3.0

    if model:
        try:
            model_features = get_feature_names(model)
            if model_features:
                non_zero = sum(1 for col in model_features if features.get(col, 0))
                if non_zero < len(model_features) * 0.3:
                    print(f"    WARNING: spread model has {non_zero}/{len(model_features)} non-zero features")
                pred = predict_regression_value(model, features)
                if pred is None:
                    raise ValueError("unsupported spread artifact format")
                # Sanity check
                if -30 <= pred <= 30:
                    return pred
                else:
                    print(f"    WARNING: spread prediction {pred:.1f} outside [-30, 30], using fallback")
            else:
                print("    WARNING: spread model has no feature_names, using fallback")
        except Exception as e:
            print(f"    WARNING: spread model predict failed: {type(e).__name__}: {e}")

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

    spread_features = {}
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
        spread_features = features.get('spread_features', {}) if features else {}

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
    except Exception as e:
        import traceback
        print(f"    WARNING: generate_game_features() failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        ml_features = {}
        spread_features = {}

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

    if not spread_features:
        spread_features = ml_features.copy()

    # Store injury info in analysis
    analysis['injury_features'] = injury_features
    analysis['injury_details'] = injury_details

    # Apply feature selection (if models/selected_features.json exists), but keep
    # context features required by context-aware stacking artifacts.
    raw_ml_features = dict(ml_features)
    ml_features = filter_features(ml_features)
    ml_features = preserve_model_context_features(ml_features, raw_ml_features, models.get('moneyline'))

    # Record odds freshness (odds were fetched by the caller before this)
    if freshness and odds:
        freshness.record_odds_fetch()

    # Phase 3.1: Run spread prediction BEFORE moneyline so we can inject it
    # as a feature.  The spread model encodes point-differential information
    # that helps moneyline calibration even though spread betting is disabled.
    predicted_spread = predict_spread(spread_features if spread_features else ml_features, models)

    if SPREAD_AS_ML_FEATURE and predicted_spread is not None:
        ml_features['model_spread_pred'] = float(predicted_spread)

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

    # Spread computation was already run above (Phase 3.1 reorder)
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
    spread_disabled_note = "" if SPREAD_BETTING_ENABLED else " [BETTING DISABLED — used as ML feature]"
    print(f"\n  SPREAD:{spread_disabled_note}")
    print(f"    Model: {home} {sp['predicted_spread']:+.1f}")
    print(f"    Market: {home} {sp['market_spread']:+.1f}")
    print(f"    Cover Prob: {sp['cover_prob']:.1%} | Edge: {sp['edge_pct']:+.1f}%")
    if SPREAD_BETTING_ENABLED and abs(sp['edge_pct']) > 2:
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
    Uses positive over_under filter matching proven data_service.py approach.
    """
    props_by_player = {}

    try:
        raw_props = api.get_player_props(game_id)
        if not raw_props:
            print(f"    Props API returned empty for game {game_id}")
            return props_by_player

        # Diagnostic counters
        total = len(raw_props)
        filtered_market = 0
        filtered_missing = 0
        accepted = 0

        # Prop type normalization (matches data_service.py lines 2755-2769)
        prop_type_map = {
            'points': 'points', 'pts': 'points',
            'rebounds': 'rebounds', 'reb': 'rebounds',
            'assists': 'assists', 'ast': 'assists',
            'threes': 'threes', '3pm': 'threes', 'fg3m': 'threes',
            'three_pointers_made': 'threes',
            'pra': 'pra', 'pts_reb_ast': 'pra',
            'points_rebounds_assists': 'pra',
            'steals': 'steals', 'stl': 'steals',
            'blocks': 'blocks', 'blk': 'blocks',
        }

        for prop in raw_props:
            # POSITIVE filter: only over_under market type (matching data_service.py line 2655)
            market = prop.get('market', {})
            market_type = market.get('type', '') if isinstance(market, dict) else ''
            if market_type != 'over_under':
                filtered_market += 1
                continue

            player_id = prop.get('player_id')
            raw_prop_type = prop.get('prop_type', '').lower()
            line = prop.get('line_value')
            vendor = prop.get('vendor', '').lower()

            if not player_id or not raw_prop_type or line is None:
                filtered_missing += 1
                continue

            try:
                line = float(line)
            except (ValueError, TypeError):
                filtered_missing += 1
                continue

            # Normalize prop type name
            prop_type = prop_type_map.get(raw_prop_type, raw_prop_type)

            if player_id not in props_by_player:
                props_by_player[player_id] = {'player_id': player_id}

            # Store if not exists or vendor is preferred (DraftKings/FanDuel)
            key = f'{prop_type}_line'
            if key not in props_by_player[player_id] or vendor in ['draftkings', 'fanduel']:
                props_by_player[player_id][key] = line
                props_by_player[player_id][f'{prop_type}_vendor'] = vendor
                # Extract actual sportsbook odds from market data
                market_data = prop.get('market', {})
                if isinstance(market_data, dict):
                    props_by_player[player_id][f'{prop_type}_over_odds'] = market_data.get('over_odds', -110)
                    props_by_player[player_id][f'{prop_type}_under_odds'] = market_data.get('under_odds')

            accepted += 1

        print(f"    Props: {total} raw, {filtered_market} non-over_under, {filtered_missing} missing fields, {accepted} accepted, {len(props_by_player)} players")

        # Resolve team_id for each player using cached get_player() calls.
        # With cache_ttl="static" on get_player(), these are free after first run.
        for pid in list(props_by_player.keys()):
            try:
                player = api.get_player(pid)
                if player:
                    team_data = player.get('team', {})
                    if isinstance(team_data, dict):
                        props_by_player[pid]['team_id'] = team_data.get('id')
                    else:
                        props_by_player[pid]['team_id'] = player.get('team_id')
            except Exception:
                pass  # team_id is nice-to-have, not critical

    except Exception as e:
        print(f"    Error fetching props for game {game_id}: {e}")

    return props_by_player


def get_player_props_hybrid(
    game_id: int,
    prop_source: str,
    api: BalldontlieAPI | None,
    event_map: dict | None = None,
    prop_fetcher=None,
    id_mapper=None,
) -> tuple[dict[int, dict], str]:
    """Fetch player props using the configured source strategy.

    Args:
        game_id: Balldontlie game ID.
        prop_source: One of 'odds-api', 'balldontlie', 'hybrid'.
        api: BalldontlieAPI instance (for legacy fallback).
        event_map: Dict from PlayerPropFetcher.match_events_to_games().
        prop_fetcher: PlayerPropFetcher instance.
        id_mapper: IDMapper for name resolution.

    Returns:
        (props_dict, source_used) where source_used is 'odds-api' or 'balldontlie'.
    """
    props = {}
    source_used = "none"

    event_info = event_map.get(game_id) if event_map else None

    # Try Odds API first (if enabled and event matched)
    if prop_source in ("odds-api", "hybrid") and prop_fetcher and event_info:
        try:
            props = prop_fetcher.get_props_for_game(
                bdl_game_id=game_id,
                event_id=event_info["event_id"],
                id_mapper=id_mapper,
            )
            if props:
                source_used = "odds-api"
                print(f" [Odds API: {len(props)} players]", end="")
        except Exception as e:
            print(f" [Odds API error: {e}]", end="")

    # Fall back to Balldontlie (if enabled and Odds API didn't return data)
    if not props and prop_source in ("balldontlie", "hybrid") and api:
        try:
            props = get_player_props_for_game(api, game_id)
            if props:
                source_used = "balldontlie"
                print(f" [BDL fallback: {len(props)} players]", end="")
        except Exception as e:
            print(f" [BDL error: {e}]", end="")

    return props, source_used


# Mapping from prop type to its per-minute rate feature name. Kept at module
# scope so the test suite can target it without reimplementing the schema.
# Naming convention matches the rest of the codebase: recent_fg3m_* uses 'm'
# explicitly to mean "made" (as opposed to "attempted").
_PER_MIN_KEY_MAP = {
    'points':   'recent_pts_per_min',
    'rebounds': 'recent_reb_per_min',
    'assists':  'recent_ast_per_min',
    'threes':   'recent_fg3m_per_min',
}

# Blend weights for rate-based projection (audit 2026-05-15). Weight 0.6 toward
# the rate × predicted_minutes projection; the model retains 0.4 weight so
# non-minutes signals (matchup, pace, role) still influence the output. These
# are hand-picked and have NOT been backtested — listed here as named
# constants so they can be swept against historical data and rotated as one.
_RATE_PROJECTION_WEIGHT = 0.6
_MODEL_PROJECTION_WEIGHT = 0.4

# Cap on the magnitude of the adjustment, expressed as a fraction of
# predicted_value. The rate-based path uses a wider cap because it's a
# structural correction; the legacy heuristic uses a narrower nudge cap.
_RATE_PROJECTION_MAX_ADJ_FRAC = 0.35
_LEGACY_NUDGE_MAX_ADJ_FRAC = 0.15


def _compute_minutes_rate_adjustment(
    predicted_value: float,
    avg_minutes: float,
    predicted_minutes: float,
    prop_type: str,
    features: dict | None,
) -> tuple[float, str, float]:
    """Adjust a model prediction for tonight's expected minutes.

    Returns (adjusted_value, rate_source, rate). rate_source is one of:
      - 'recent_per_min': features carry recent_{stat}_per_min for this prop;
        adjusted = 0.6 * (rate * predicted_minutes) + 0.4 * predicted_value,
        clamped to ±35% of predicted_value.
      - 'legacy_pred_div_avg': falls back to legacy linear scaling
        (rate = predicted_value / avg_minutes), nudge capped at ±15%.

    Pure function so tests can drive every branch without touching the model.
    """
    # Branch 1: per-minute rate for direct-stat props.
    per_min_key = _PER_MIN_KEY_MAP.get(prop_type)
    rate: float = 0.0
    rate_source = 'legacy_pred_div_avg'
    if per_min_key and features and (features.get(per_min_key) or 0) > 0:
        rate = float(features[per_min_key])
        rate_source = 'recent_per_min'
    elif prop_type == 'pra' and features:
        pts_pm = float(features.get('recent_pts_per_min') or 0)
        reb_pm = float(features.get('recent_reb_per_min') or 0)
        ast_pm = float(features.get('recent_ast_per_min') or 0)
        total_pm = pts_pm + reb_pm + ast_pm
        if total_pm > 0:
            rate = total_pm
            rate_source = 'recent_per_min'

    if rate_source == 'legacy_pred_div_avg':
        # avg_minutes is the model's implicit pace baseline. Without that we
        # have nothing to scale against — return the model prediction unchanged.
        if avg_minutes <= 0:
            return predicted_value, rate_source, 0.0
        rate = predicted_value / avg_minutes
        minutes_delta = rate * (predicted_minutes - avg_minutes)
        adjusted = predicted_value + minutes_delta
        max_adj = abs(predicted_value) * _LEGACY_NUDGE_MAX_ADJ_FRAC
    else:
        rate_projection = rate * predicted_minutes
        adjusted = (
            _RATE_PROJECTION_WEIGHT * rate_projection
            + _MODEL_PROJECTION_WEIGHT * predicted_value
        )
        max_adj = abs(predicted_value) * _RATE_PROJECTION_MAX_ADJ_FRAC

    # Symmetric cap relative to predicted_value
    lo = predicted_value - max_adj
    hi = predicted_value + max_adj
    if adjusted < lo:
        adjusted = lo
    elif adjusted > hi:
        adjusted = hi
    return adjusted, rate_source, rate


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
    under_odds: int = None,  # Phase 4: under side odds for devigging
    opp_stats: dict = None,  # Real opponent defensive stats
    minutes_multiplier: float = 1.0,  # Lineup-intel cap (e.g., 0.65 = ≤65% of usual)
    availability_probability: float = 1.0,  # Lineup-intel availability (0-1)
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

    # FIX 5: Hard DNP guard — injury checks happen at the call-site loop, but this
    # second layer ensures no prediction is generated if the player name is empty
    # or clearly a placeholder (e.g., from a fallback path that bypassed the loop filter).
    if not player_name or player_name.strip() == '':
        return None

    over_prob = 0.5
    predicted_value = None
    ensemble_predicted_value = None
    features = None  # Initialize for quantile model usage later
    effective_sigma = get_prop_std_dev(prop_type)  # Overridden by quantile model if available
    quantile_sigma = None

    model_data = models.get(f'prop_{prop_type}')

    if model_data and use_api_features:
        try:
            # Get cached features (or generate new ones) - use Balldontlie ID for fast lookup
            # Extract Phase 2 game context when available
            _gc = game_context or {}
            _is_home = _gc.get('is_home', False)
            features = get_cached_features(
                player_name, prop_type, opponent_id,
                bdl_player_id=player_id,
                is_home=_is_home,
                opp_stats=opp_stats,
                # Phase 2.1: Opponent schedule
                opp_days_rest=_gc.get('opp_days_rest', 2),
                opp_is_b2b=bool(_gc.get('opp_is_back_to_back', False)),
                opp_def_tier=_gc.get('opp_def_tier', 2),
                # Phase 2.2: Player game context
                travel_distance=float(_gc.get('travel_distance', 0.0)),
                games_last_7=int(_gc.get('games_last_7_days', 3)),
                season_phase=int(_gc.get('season_phase', 1)),
                is_b2b_home=bool(_gc.get('is_b2b_home', False)),
                is_b2b_away=bool(_gc.get('is_b2b_away', False)),
            )

            if features:
                # Guard: verify features have minimum required fields.
                # The model needs season_avg, recent_avg, predicted_minutes etc.
                # If fallback features (only 27 fields) are used, key averages
                # may be missing → garbage predictions. Check for critical fields.
                _critical_keys = {'season_min_avg', 'days_rest', 'opp_def_rating'}
                _has_critical = sum(1 for k in _critical_keys if features.get(k) is not None)
                if _has_critical < 2:
                    logger.warning(
                        "Incomplete features for %s %s (%d fields, %d/3 critical keys "
                        "present) — skipping. Likely cause: circuit breaker open or "
                        "BDL throttling.",
                        player_name, prop_type, len(features), _has_critical,
                    )
                    return None

                # Feature-quality flag: derive the expected feature count from
                # the *model's own* feature_names so the threshold works after
                # feature-reduction retrains (Phase 1.1 cut some props to ~80
                # features). Mark degraded only when fewer than 75% of the
                # model's expected features are populated with non-None values.
                # Fallback to a generous expected count of 60 when the model
                # doesn't expose its feature list — keeps the warning useful
                # without false positives on legitimately compact models.
                _expected_features: list[str] = []
                if isinstance(model_data, dict):
                    _expected_features = list(model_data.get('feature_names') or [])
                _expected_count = max(60, len(_expected_features))
                _populated_count = sum(
                    1 for k in (_expected_features or features.keys())
                    if features.get(k) is not None
                )
                _coverage = _populated_count / _expected_count if _expected_count else 1.0
                _feature_quality = 'full' if _coverage >= 0.75 else 'degraded'
                if _feature_quality == 'degraded':
                    logger.warning(
                        "Degraded feature set for %s %s: %d/%d fields "
                        "populated (%.0f%% coverage, threshold 75%%)",
                        player_name, prop_type,
                        _populated_count, _expected_count, _coverage * 100,
                    )

                # Inject prop_line features. After the next retrain, models will no
                # longer include prop_line or prop_line_vs_season in their feature_names
                # and these values will be ignored. Until then, we keep injecting them
                # for backward compatibility with the currently deployed model.
                _season_avg_map = {
                    'points': ('season_pts_avg', 'recent_pts_avg'),
                    'rebounds': ('season_reb_avg', 'recent_reb_avg'),
                    'assists': ('season_ast_avg', 'recent_ast_avg'),
                    'threes': ('season_fg3m_avg', 'recent_fg3m_avg'),
                }
                _line_mapping = _season_avg_map.get(prop_type)
                if _line_mapping:
                    _s_avg = features.get(_line_mapping[0], line)
                    _r_avg = features.get(_line_mapping[1], _s_avg)
                else:
                    # PRA: sum of component season avgs
                    _s_avg = (features.get('season_pts_avg', 0) +
                              features.get('season_reb_avg', 0) +
                              features.get('season_ast_avg', 0))
                    _r_avg = (features.get('recent_pts_avg', 0) +
                              features.get('recent_reb_avg', 0) +
                              features.get('recent_ast_avg', 0))
                # Backward compat: keep prop_line for pre-retrain models
                features['prop_line'] = line
                features['prop_line_vs_season'] = line - _s_avg
                features['prop_line_vs_recent'] = line - _r_avg

                # Handle ENSEMBLE format (multiple models with meta_model)
                if isinstance(model_data, dict) and model_data.get('ensemble'):
                    base_models = model_data['models']
                    meta_model = model_data['meta_model']
                    # Phase 3.2: stored training-time weights serve as prior;
                    # dynamic weighter overrides them when enough history exists.
                    stored_weights = model_data.get('model_weights', {})
                    scaler = model_data.get('scaler')
                    feature_names = model_data.get('feature_names', [])

                    # Build feature array matching training features
                    X = pd.DataFrame([{k: features.get(k, np.nan) for k in feature_names}])
                    X = smart_fillna(X[feature_names])

                    # Scale if scaler available
                    X_scaled = scaler.transform(X) if scaler is not None else X.values

                    # Get predictions from tree-based models only (ridge can have scaling issues)
                    tree_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
                    base_preds_dict: dict = {}
                    for name in tree_models:
                        if name in base_models and base_models[name] is not None:
                            try:
                                pred = base_models[name].predict(X_scaled)[0]
                                if -50 < pred < 100:  # Sanity check
                                    base_preds_dict[name] = float(pred)
                            except Exception:
                                continue

                    # Phase 3.2: Compute weighted average using dynamic weights.
                    # Priority: dynamic (perf-based) > stored training weights > equal.
                    if base_preds_dict:
                        if HAS_DYNAMIC_WEIGHTING and _ENSEMBLE_WEIGHTER is not None:
                            dyn_weights = _ENSEMBLE_WEIGHTER.get_weights(
                                model_names=list(base_preds_dict.keys()),
                                prop_type=prop_type,
                                recent_predictions=base_preds_dict,
                            )
                            predicted_value = float(sum(
                                dyn_weights.get(n, 1.0 / len(base_preds_dict)) * v
                                for n, v in base_preds_dict.items()
                            ))
                            logger.debug(
                                "Dynamic weights for %s %s: %s",
                                player_name, prop_type,
                                {n: f"{w:.3f}" for n, w in dyn_weights.items()},
                            )
                        elif stored_weights:
                            # Use training-time weights if dynamic weighter unavailable
                            wt_total = sum(
                                stored_weights.get(n, 1.0) for n in base_preds_dict
                            )
                            if wt_total > 0:
                                predicted_value = float(sum(
                                    stored_weights.get(n, 1.0) * v / wt_total
                                    for n, v in base_preds_dict.items()
                                ))
                            else:
                                predicted_value = float(np.mean(list(base_preds_dict.values())))
                        else:
                            predicted_value = float(np.mean(list(base_preds_dict.values())))

                        # Phase 3.2: Log individual model predictions for accuracy tracking
                        if _PERF_TRACKER is not None:
                            try:
                                _PERF_TRACKER.log_predictions(
                                    date=datetime.now(ET).strftime('%Y-%m-%d'),
                                    player=player_name,
                                    prop_type=prop_type,
                                    line=line,
                                    predictions=base_preds_dict,
                                    ensemble_pred=predicted_value,
                                )
                            except Exception:
                                pass  # Logging failure must never crash inference
                    else:
                        # Fallback to season average from features
                        predicted_value = features.get('season_pts_avg', 15.0)

                    # Convert to probability using normal CDF
                    # Phase 1.1: clamp to [PROB_CLAMP_MIN, PROB_CLAMP_MAX]
                    std = get_prop_std_dev(prop_type)  # FIX: Use prop-specific std
                    z_score = (predicted_value + PROP_BIAS_CORRECTION.get(prop_type.lower(), 0.0) - line) / std
                    over_prob = float(np.clip(norm.cdf(z_score), PROB_CLAMP_MIN, PROB_CLAMP_MAX))

                # Handle StackingRegressor format (has 'base_models' key)
                elif isinstance(model_data, dict) and 'base_models' in model_data and 'meta_model' in model_data:
                    base_models = model_data['base_models']
                    meta_model = model_data['meta_model']
                    scaler = model_data.get('scaler')
                    feature_names = model_data.get('feature_names', [])

                    # Build feature array matching training features
                    X = pd.DataFrame([{k: features.get(k, np.nan) for k in feature_names}])
                    X = smart_fillna(X[feature_names])

                    # Scale if scaler available
                    X_scaled = scaler.transform(X) if scaler is not None else X.values

                    # Get base model predictions (preserve order for meta-model)
                    stacking_preds_dict: dict = {}
                    for name, model in base_models.items():
                        try:
                            pred = model.predict(X_scaled)[0]
                            if -50 < pred < 100:  # Sanity check
                                stacking_preds_dict[name] = float(pred)
                        except Exception:
                            continue

                    base_preds = list(stacking_preds_dict.values())

                    # Use meta model for stacking (true stacking — meta-learner
                    # learns optimal combination of base predictions)
                    if meta_model is not None and base_preds:
                        meta_features = np.array(base_preds).reshape(1, -1)
                        predicted_value = float(meta_model.predict(meta_features)[0])
                    elif base_preds:
                        # Phase 3.2: dynamic weights when meta-model absent
                        if HAS_DYNAMIC_WEIGHTING and _ENSEMBLE_WEIGHTER is not None and stacking_preds_dict:
                            dyn_weights = _ENSEMBLE_WEIGHTER.get_weights(
                                model_names=list(stacking_preds_dict.keys()),
                                prop_type=prop_type,
                                recent_predictions=stacking_preds_dict,
                            )
                            predicted_value = float(sum(
                                dyn_weights.get(n, 1.0 / len(stacking_preds_dict)) * v
                                for n, v in stacking_preds_dict.items()
                            ))
                        else:
                            predicted_value = float(np.mean(base_preds))
                    else:
                        predicted_value = features.get('season_pts_avg', 15.0)

                    # Convert to probability using normal CDF
                    # Phase 1.1: clamp to [PROB_CLAMP_MIN, PROB_CLAMP_MAX]
                    std = get_prop_std_dev(prop_type)  # FIX: Use prop-specific std
                    z_score = (predicted_value + PROP_BIAS_CORRECTION.get(prop_type.lower(), 0.0) - line) / std
                    over_prob = float(np.clip(norm.cdf(z_score), PROB_CLAMP_MIN, PROB_CLAMP_MAX))

                # Handle dict format with single model
                elif isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                    scaler = model_data.get('scaler')
                    feature_names = model_data.get('feature_names', [])

                    # Build feature array matching training features
                    X = pd.DataFrame([{k: features.get(k, np.nan) for k in feature_names}])
                    X = smart_fillna(X[feature_names])

                    # Scale if scaler available
                    X_scaled = scaler.transform(X) if scaler is not None else X.values

                    # Predict (regression model predicts stat value)
                    predicted_value = float(model.predict(X_scaled)[0])

                    # Convert to probability using normal CDF
                    # Phase 1.1: clamp to [PROB_CLAMP_MIN, PROB_CLAMP_MAX]
                    std = get_prop_std_dev(prop_type)  # FIX: Use prop-specific std
                    z_score = (predicted_value + PROP_BIAS_CORRECTION.get(prop_type.lower(), 0.0) - line) / std
                    over_prob = float(np.clip(norm.cdf(z_score), PROB_CLAMP_MIN, PROB_CLAMP_MAX))

                # Handle model object with predict method (PropEnsembleModel)
                elif hasattr(model_data, 'predict'):
                    result = model_data.predict(features, prop_line=line)

                    if 'predicted_value' in result:
                        predicted_value = result['predicted_value']

                        # Fix 1.4: If model was trained in residual mode,
                        # predicted_value is a residual — add season average.
                        if getattr(model_data, '_residual_mode', False):
                            sa_col = getattr(model_data, '_season_avg_col', None)
                            if sa_col and features:
                                season_avg = features.get(sa_col, 0)
                            elif prop_type == 'pra' and features:
                                season_avg = (
                                    features.get('season_pts_avg', 0)
                                    + features.get('season_reb_avg', 0)
                                    + features.get('season_ast_avg', 0)
                                )
                            else:
                                season_avg = 0
                            # DO NOT add residual_mean_offset back. The offset
                            # represents survivorship bias (+1.87 for points) that
                            # was removed during training. Adding it back makes
                            # predictions always exceed sportsbook lines → always over.
                            # The model's bias_correction already handles calibration.
                            predicted_value = season_avg + predicted_value

                    if 'over_probability' in result and not getattr(model_data, '_residual_mode', False):
                        over_prob = result['over_probability']
                    elif predicted_value is not None:
                        std = get_prop_std_dev(prop_type)
                        z_score = (predicted_value + PROP_BIAS_CORRECTION.get(prop_type.lower(), 0.0) - line) / std
                        over_prob = float(np.clip(norm.cdf(z_score), PROB_CLAMP_MIN, PROB_CLAMP_MAX))  # Phase 1.1

        except Exception:
            logger.warning("Ensemble prediction failed for %s %s", player_name, prop_type, exc_info=True)

    # Quantile model: run BEFORE adjustments so we can derive player-specific sigma
    pred_low = None
    pred_median = None
    pred_high = None
    quantile_model_dict = models.get(f'prop_{prop_type}_quantile')

    if quantile_model_dict and features and use_api_features:
        try:
            import pandas as pd

            # Extract QuantilePropModel from dict
            quantile_model_obj = None
            if isinstance(quantile_model_dict, dict) and 'model' in quantile_model_dict:
                quantile_model_obj = quantile_model_dict['model']

            if quantile_model_obj and hasattr(quantile_model_obj, 'quantile_models'):
                quantile_models = quantile_model_obj.quantile_models
                scaler = getattr(quantile_model_obj, 'scaler', None)
                feature_names = getattr(quantile_model_obj, 'feature_names', [])
            elif isinstance(quantile_model_dict, dict) and 'quantile_models' in quantile_model_dict:
                quantile_models = quantile_model_dict['quantile_models']
                scaler = quantile_model_dict.get('scaler')
                feature_names = quantile_model_dict.get('feature_names', [])
            else:
                quantile_models = None
                feature_names = []

            if quantile_models and feature_names:
                X = pd.DataFrame([{k: features.get(k, np.nan) for k in feature_names}])
                X = smart_fillna(X[feature_names])
                X_scaled = scaler.transform(X) if scaler is not None else X.values

                pred_low = float(quantile_models[0.1].predict(X_scaled)[0])
                pred_median = float(quantile_models[0.5].predict(X_scaled)[0])
                pred_high = float(quantile_models[0.9].predict(X_scaled)[0])

                # Use quantile median as primary prediction when available
                # Quantile median (slope=0.89) is far less compressed than ensemble (slope=0.62)
                # The ensemble still serves as fallback when quantile model isn't available
                ensemble_predicted_value = predicted_value
                if pred_median is not None:
                    predicted_value = pred_median

                # Correct regression-to-mean compression in quantile predictions.
                # POINTS slope=0.724 means high scorers are predicted 3-9 pts too low.
                # Use player's season average as decompression anchor (defined earlier)
                predicted_value = decompress_quantile_prediction(
                    predicted_value, line, prop_type,
                    player_season_avg=_s_avg if features else None
                )

                # Fix 2: Apply PROP_BIAS_CORRECTION to predicted_value so the
                # displayed prediction reflects the systematic over-prediction
                # correction, not just the z-score used for probability.
                _bias_corr = PROP_BIAS_CORRECTION.get(prop_type.lower(), 0.0)
                if _bias_corr != 0.0:
                    predicted_value += _bias_corr

                # Fix 3: Sanity clamp — predicted value shouldn't exceed 1.5x the
                # sportsbook line or 2x the player's season average (whichever is
                # larger).  Prevents absurd quantile outputs (e.g. 22 pts for an
                # 8.5-line player) that are a model quality issue, not real signal.
                _s_avg_val = _s_avg if _s_avg else None
                _max_reasonable = (
                    max(_s_avg_val * 2.0, line * 1.5) if _s_avg_val else line * 2.0
                )
                predicted_value = min(predicted_value, _max_reasonable)

                # Derive player-specific sigma from quantile spread
                quantile_sigma = compute_quantile_sigma(pred_low, pred_high, prop_type)
                effective_sigma = quantile_sigma

                # Use quantile model's interpolation-based probability instead of
                # norm.cdf heuristic. This gives principled P(X > line) from the
                # actual predicted distribution, not a Gaussian assumption.
                if quantile_model_obj and hasattr(quantile_model_obj, 'predict_over_probability'):
                    try:
                        over_prob = quantile_model_obj.predict_over_probability(features, line)
                    except Exception:
                        # Fall back to norm.cdf if quantile prob fails.
                        # PROP_BIAS_CORRECTION already applied to predicted_value (Fix 2),
                        # so omit bias_fix here to avoid double-counting.
                        z_score = (predicted_value - line) / effective_sigma
                        over_prob = float(np.clip(norm.cdf(z_score), PROB_CLAMP_MIN, PROB_CLAMP_MAX))  # Phase 1.1
        except Exception:
            logger.warning("Quantile model failed for %s %s", player_name, prop_type, exc_info=True)

    # Save original prediction for total adjustment cap
    original_predicted_value = predicted_value

    # Recalculate over_prob with quantile-derived sigma ONLY if quantile model
    # didn't already set it via predict_over_probability above
    if predicted_value is not None and not (quantile_model_dict and features and use_api_features):
        # PROP_BIAS_CORRECTION already applied to predicted_value (Fix 2), omit here.
        z_score = (predicted_value - line) / effective_sigma
        over_prob = float(np.clip(norm.cdf(z_score), PROB_CLAMP_MIN, PROB_CLAMP_MAX))  # Phase 1.1

    # Phase 3: Minutes Oracle adjustment — per-minute rate scaling
    minutes_dist = None
    if predicted_value is not None and game_context and models.get('minutes_oracle'):
        minutes_dist = predict_minutes_distribution(
            player_id=player_id,
            team_id=team_id or 0,
            opponent_team_id=game_context.get('opponent_team_id', 0),
            game_context=game_context,
            models=models,
            prop_features=features,
        )

        if minutes_dist:
            # Get player's historical average minutes from features
            avg_minutes = 0
            if features:
                avg_minutes = features.get('season_min_avg', 0) or features.get('recent_min_avg', 0) or 0

            predicted_minutes = minutes_dist.get('p50', avg_minutes)

            # Apply lineup-intel minutes_multiplier BEFORE the DNP gate so a
            # restricted player (e.g., 0.6 multiplier on a 30-min average) gets
            # the correct 18-minute projection instead of 30. The multiplier
            # only narrows minutes (never expands), and we floor it at 0.3 to
            # guard against bad scraper output.
            if 0.3 <= minutes_multiplier < 1.0:
                predicted_minutes = predicted_minutes * minutes_multiplier
                logger.info(
                    "Minutes restriction applied for %s %s: ×%.2f → %.1f min",
                    player_name, prop_type, minutes_multiplier, predicted_minutes,
                )

            # DNP filter: skip predictions for players predicted to play < 15 minutes
            if 0 < predicted_minutes < 15:
                return None

            # Only adjust if we have meaningful baseline and prediction
            if avg_minutes > 10 and predicted_minutes > 0:
                # Use post-decompression predicted_value for consistent rate calculation.
                # The rate should reflect the final predicted production level.
                #
                adjusted_value, rate_source, rate = _compute_minutes_rate_adjustment(
                    predicted_value=predicted_value,
                    avg_minutes=avg_minutes,
                    predicted_minutes=predicted_minutes,
                    prop_type=prop_type,
                    features=features,
                )

                # Only apply if adjustment is meaningful (>1% change)
                if abs(adjusted_value - predicted_value) / max(abs(predicted_value), 0.1) > 0.01:
                    logger.info(
                        "Minutes-rate adjustment for %s %s (%s): "
                        "%.2f -> %.2f (rate=%.4f, predicted_min=%.1f, avg_min=%.1f)",
                        player_name, prop_type, rate_source,
                        predicted_value, adjusted_value, rate,
                        predicted_minutes, avg_minutes,
                    )
                    predicted_value = adjusted_value

                    # Recalculate probability with adjusted value
                    # PROP_BIAS_CORRECTION already in predicted_value, omit bias_fix.
                    z_score = (predicted_value - line) / effective_sigma
                    over_prob = float(np.clip(norm.cdf(z_score), PROB_CLAMP_MIN, PROB_CLAMP_MAX))  # Phase 1.1

    # Apply lineup-intel minutes_multiplier in the no-minutes-oracle fallback.
    # The primary application happens above (inside `if minutes_dist:`) using
    # the predicted minutes. When the oracle is unavailable that block doesn't
    # run, so we have to scale predicted_value here directly — otherwise a
    # 0.6× minutes restriction silently produces a full season-average forecast.
    # Counting-stat props scale roughly linearly with minutes, so the same
    # multiplier applies to the value as to the minutes.
    if (predicted_value is not None
            and not minutes_dist
            and 0.3 <= minutes_multiplier < 1.0):
        predicted_value = predicted_value * minutes_multiplier
        z_score = (predicted_value - line) / effective_sigma
        over_prob = float(np.clip(norm.cdf(z_score), PROB_CLAMP_MIN, PROB_CLAMP_MAX))
        logger.info(
            "Minutes restriction (oracle-unavailable path) for %s %s: "
            "×%.2f → predicted_value %.2f",
            player_name, prop_type, minutes_multiplier, predicted_value,
        )

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
                # PROP_BIAS_CORRECTION already in predicted_value, omit bias_fix.
                z_score = (predicted_value - line) / effective_sigma
                over_prob = float(np.clip(norm.cdf(z_score), PROB_CLAMP_MIN, PROB_CLAMP_MAX))  # Phase 1.1
        except Exception:
            logger.warning("Injury boost failed for %s %s", player_name, prop_type, exc_info=True)

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
                    # PROP_BIAS_CORRECTION already in predicted_value, omit bias_fix.
                    z_score = (predicted_value - line) / effective_sigma
                    over_prob = float(np.clip(norm.cdf(z_score), PROB_CLAMP_MIN, PROB_CLAMP_MAX))  # Phase 1.1
        except Exception:
            logger.warning("Calibration adjustment failed for %s %s", player_name, prop_type, exc_info=True)

    # Total cap: all adjustments (minutes + injury + calibration) stay within ±25% of original
    if predicted_value is not None and original_predicted_value is not None and original_predicted_value != 0:
        max_total = abs(original_predicted_value) * 0.25
        lower = original_predicted_value - max_total
        upper = original_predicted_value + max_total
        if predicted_value < lower or predicted_value > upper:
            predicted_value = max(lower, min(upper, predicted_value))
            # PROP_BIAS_CORRECTION already in predicted_value, omit bias_fix.
            z_score = (predicted_value - line) / effective_sigma
            over_prob = float(np.clip(norm.cdf(z_score), PROB_CLAMP_MIN, PROB_CLAMP_MAX))  # Phase 1.1

    # Phase 3.3: Threes-specific enhancements before final calibration
    #   1. Sample size gate — skip if player has too few games or attempts
    #   2. Regression-to-mean — fade hot streaks, boost cold ones
    #   3. Poisson CDF — more accurate P(X > line) for count data
    threes_streak_info = {}
    if prop_type.lower() == 'threes' and predicted_value is not None:
        threes_cfg = _get_prop_config('threes') if HAS_POISSON_MODEL else None
        if threes_cfg is not None:
            # 1. Sample size gate
            _season_games_cnt = features.get('season_games', 0) if features else 0
            _fg3a_avg_val = features.get('fg3a_avg', 0.0) if features else 0.0
            if (_season_games_cnt < threes_cfg.min_sample_games
                    or _fg3a_avg_val < threes_cfg.min_fg3a):
                logger.debug(
                    "Threes gate: %s skipped (games=%d, fg3a_avg=%.1f)",
                    player_name, _season_games_cnt, _fg3a_avg_val,
                )
                return None  # Insufficient sample — do not generate a threes bet

            # 2. Regression-to-mean adjustment (streak detection)
            if features and HAS_POISSON_MODEL:
                try:
                    threes_streak_info = detect_threes_streak(features)
                    streak_fade = threes_streak_info.get('streak_fade', 0.0)
                    if streak_fade != 0.0:
                        predicted_value = predicted_value * (1.0 + streak_fade)
                        # PROP_BIAS_CORRECTION already in predicted_value, omit bias_fix.
                        z_score = (predicted_value - line) / effective_sigma
                        over_prob = float(norm.cdf(z_score))
                        logger.debug(
                            "Threes streak %s for %s: fade=%.1f%% → pred=%.2f",
                            threes_streak_info.get('streak_type'), player_name,
                            streak_fade * 100, predicted_value,
                        )
                except Exception:
                    logger.debug(
                        "Streak detection failed for %s", player_name, exc_info=True
                    )

            # 3. Poisson CDF override — more accurate for integer count data
            if features and HAS_POISSON_MODEL and threes_cfg.use_poisson:
                try:
                    lam = max(predicted_value, 0.01)
                    poisson_prob = compute_poisson_over_prob(lam, line)
                    # Blend: 70% Poisson, 30% Gaussian to hedge model error
                    over_prob = 0.70 * poisson_prob + 0.30 * over_prob
                    logger.debug(
                        "Threes Poisson: λ=%.2f line=%.1f P_poisson=%.3f P_blend=%.3f",
                        lam, line, poisson_prob, over_prob,
                    )
                except Exception:
                    logger.debug(
                        "Poisson CDF failed for %s", player_name, exc_info=True
                    )

    # Apply empirical probability calibration (isotonic regression from backtest)
    if over_prob is not None:
        over_prob = apply_empirical_calibration(over_prob, prop_type)

    # Phase 4: Single edge computation after all adjustments (minutes + injury + calibration)
    edge_info = _calculate_prop_edge(over_prob, american_odds, under_odds=under_odds)
    edge = edge_info['edge']
    pick = edge_info['pick']

    # Confidence scoring and bet sizing
    confidence_score = 50.0  # Default moderate confidence
    edge_quality_tier = 'moderate'
    suggested_bet_size = 0.0
    bet_recommendation = 'PASS'  # Phase 4: default to PASS

    # Calculate confidence score from over_prob distance from 50% (Phase 1 fix, 2026-03-31).
    # The old band_width formula was fragile: it required per-prop-type multipliers tuned
    # by hand on a specific backtest, and was based on the quantile band width which has
    # weak correlation with actual prediction accuracy.
    #
    # over_prob (already computed from the z-score above) is monotonically related to
    # prediction certainty — the further it is from 0.5, the more decisive the model is
    # about which side of the line the outcome will fall on.
    #
    # Mapping:
    #   over_prob = 0.50  →  confidence = 40.0  (no edge, model is uncertain)
    #   over_prob = 0.65  →  confidence = 55.0  (moderate lean)
    #   over_prob = 0.75  →  confidence = 65.0  (meaningful edge)
    #   over_prob = 0.85  →  confidence = 75.0  (strong edge)
    #   over_prob = 0.90  →  confidence = 80.0  (very strong edge)
    #   over_prob = 1.00  →  confidence = 90.0  (maximum certainty, theoretical)
    distance_from_50 = abs(over_prob - 0.5) * 2  # 0 to 1 scale
    confidence_score = max(40.0, min(90.0, 40.0 + distance_from_50 * 50.0))

    # Phase 3: Adjust confidence based on minutes uncertainty
    if minutes_dist:
        uncertainty = minutes_dist.get('uncertainty', 'medium')
        if uncertainty == 'high':
            confidence_score *= 0.80  # 20% penalty for high minutes uncertainty
        elif uncertainty == 'medium':
            confidence_score *= 0.92  # 8% penalty for medium uncertainty
        # 'low' = no penalty
        confidence_score = max(40.0, min(90.0, confidence_score))

    # Phase 5: Calibration confidence adjustment — DISABLED (Phase 1 fix, 2026-03-31).
    # The calibration adjuster was trained on the old band_width-based confidence signal.
    # After replacing that formula with the over_prob-based formula (Fix 2 above), the
    # adjuster's learned corrections are derived from a different input distribution and
    # would push confidence in the wrong direction. It must be retrained on the new signal
    # before being re-enabled.
    # if calibration_applied and calibration_applied.get('adjustments_applied'):
    #     ... (disabled — retrain adjuster on over_prob confidence before re-enabling)

    # Calculate edge quality tier based on confidence + edge magnitude (Task 2.4)
    edge_quality_tier = get_edge_quality_tier(confidence_score, edge)

    # Fix 5.1: Kelly sizing from calibrated quantile probability (over_prob).
    # over_prob comes from QuantilePropModel.predict_over_probability() or
    # norm.cdf(z_score) — both are calibrated. Do NOT use confidence_score
    # (which has r=0.10 correlation with accuracy) for bet sizing.
    #
    # Phase 1.1: Clamp over_prob to [PROB_CLAMP_MIN, PROB_CLAMP_MAX] before
    # deriving win_prob. calculate_kelly_bet_size() rejects win_prob ≥ 1 or ≤ 0
    # with a warning and returns 0.0 — this clamp prevents that silent failure.
    if abs(edge) > 2.0:
        over_prob = float(np.clip(over_prob, PROB_CLAMP_MIN, PROB_CLAMP_MAX))
        win_prob = over_prob if over_prob > 0.5 else (1 - over_prob)
        decimal_odds = american_to_decimal(american_odds)
        default_bankroll = 1000.0

        # Bug fix #12: Inline Kelly fallback so exceptions are handled properly.
        # HAS_KELLY_SIZING is always True after Fix 5.3 required imports,
        # but the function can still throw on unexpected inputs.
        try:
            suggested_bet_size = calculate_kelly_bet_size(
                win_prob=win_prob,
                decimal_odds=decimal_odds,
                bankroll=default_bankroll,
                fractional=0.25,
                edge_tier=edge_quality_tier,
                current_drawdown=0.0,
                num_same_day_bets=1,
                max_bet_pct=0.05,
            )
            suggested_bet_size = (suggested_bet_size / default_bankroll) * 100
        except Exception:
            # Fallback: manual quarter-Kelly
            b = decimal_odds - 1
            q = 1 - win_prob
            kelly_full = max(0, (b * win_prob - q) / b) if b > 0 else 0
            suggested_bet_size = min(kelly_full * 0.25 * 100, 5.0)

    # Phase 4: Signal classification using edge magnitude
    bet_recommendation = get_signal_from_edge(edge, edge_info.get('edge_quality'))

    # IMPROVEMENT 6: Apply smart bet filter (bet_filter.py / prediction_pipeline.py)
    # This overrides legacy bet sizing and adds filter metadata to the result.
    bet_filter_result = {}
    if HAS_BET_FILTER and predicted_value is not None:
        try:
            # Estimate games_played from season averages in features (if available)
            _season_games = None
            if features is not None and isinstance(features, dict):
                _season_games = features.get('season_games')

            # Gate: only bet on starter-level players (predicted minutes >= 25)
            _pred_mins = minutes_dist.get('p50') if minutes_dist else None
            if _pred_mins is not None and _pred_mins < 25:
                bet_recommendation = 'PASS'
                suggested_bet_size = 0.0
                bet_filter_result = {
                    'should_bet': False,
                    'reason': f'Predicted minutes {_pred_mins:.0f} < 25 (starter-level gate)',
                    'tier': 'no_bet',
                }
                _edge_val = abs(predicted_value - line) if predicted_value is not None else 0.0
                return {
                    'player_name': player_name,
                    'player_id': player_id,
                    'prop_type': prop_type,
                    'line': line,
                    'predicted_value': predicted_value,
                    'over_probability': over_prob,
                    'confidence_score': 50.0,
                    'edge_quality_tier': 'none',
                    'suggested_bet_size': 0.0,
                    'bet_recommendation': 'PASS',
                    'signal': 'PASS',
                    'edge': _edge_val,
                    'over_edge': 0.0,
                    'under_edge': 0.0,
                    'edge_quality': 'none',
                    'ev_per_dollar': 0.0,
                    'implied_probability': 0.5,
                    'model_probability': over_prob or 0.5,
                    'has_edge': False,
                    'bet_filter': bet_filter_result,
                    'bet_filter_passed': False,
                    'bet_filter_tier': 'no_bet',
                    'predicted_minutes': _pred_mins,
                }

            # Narrow-interval gate: only bet when the quantile model's 80% CI
            # width is less than 2x the edge. If uncertainty is too wide relative
            # to edge, the signal is drowned in noise.
            if pred_low is not None and pred_high is not None and predicted_value is not None:
                ci_width_80 = pred_high - pred_low  # 10th to 90th percentile
                edge_abs = abs(predicted_value - line)
                if edge_abs > 0 and ci_width_80 > 2.0 * edge_abs:
                    bet_recommendation = 'PASS'
                    suggested_bet_size = 0.0
                    bet_filter_result = {
                        'should_bet': False,
                        'reason': (
                            f'CI width {ci_width_80:.1f} > 2x edge {edge_abs:.1f} '
                            f'(narrow-interval gate)'
                        ),
                        'tier': 'no_bet',
                    }
                    # Skip evaluate_bet entirely — this is clearly noise
                    return {
                        'player_name': player_name,
                        'player_id': player_id,
                        'prop_type': prop_type,
                        'line': line,
                        'predicted_value': predicted_value,
                        'over_probability': over_prob,
                        'confidence_score': 50.0,
                        'edge_quality_tier': 'none',
                        'suggested_bet_size': 0.0,
                        'bet_recommendation': 'PASS',
                        'signal': 'PASS',
                        'edge': edge_abs,
                        'over_edge': 0.0,
                        'under_edge': 0.0,
                        'edge_quality': 'none',
                        'ev_per_dollar': 0.0,
                        'implied_probability': 0.5,
                        'model_probability': over_prob or 0.5,
                        'has_edge': False,
                        'bet_filter': bet_filter_result,
                        'bet_filter_passed': False,
                        'bet_filter_tier': 'no_bet',
                        'predicted_minutes': minutes_dist.get('p50') if minutes_dist else None,
                    }

            # over_prob comes from quantile-derived CDF or predict_over_probability
            # and is already calibrated — skip temperature scaling.
            bet_filter_result = _evaluate_bet(
                prop_type=prop_type.lower(),
                predicted=predicted_value,
                line=line,
                raw_confidence=over_prob,
                games_played=_season_games,
                bankroll=1000.0,
                pre_calibrated=True,
            )

            # Override bet recommendation when filter says no-bet
            if not bet_filter_result.get('should_bet', True):
                bet_recommendation = 'PASS'
                # Override suggested_bet_size to 0 when filter rejects
                suggested_bet_size = 0.0
            elif bet_filter_result.get('bet_size', 0) > 0:
                # Use pipeline bet size (% of $1000 bankroll, same as legacy)
                suggested_bet_size = (bet_filter_result['bet_size'] / 1000.0) * 100
        except Exception:
            logger.warning("Bet filter failed for %s %s — defaulting to PASS", player_name, prop_type, exc_info=True)
            bet_recommendation = 'PASS'
            suggested_bet_size = 0.0

    return {
        'player': player_name,
        'player_id': player_id,
        'stat': prop_type.upper(),
        'line': line,
        'over_prob': over_prob,
        'under_prob': 1.0 - over_prob,
        'edge': edge,
        'predicted_value': predicted_value,
        'ensemble_predicted_value': ensemble_predicted_value,
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
        # Sigma diagnostics
        'effective_sigma': effective_sigma,
        'quantile_sigma': quantile_sigma,
        # IMPROVEMENT 6: Bet filter metadata
        'bet_filter': bet_filter_result,
        'bet_filter_passed': bet_filter_result.get('should_bet', True),
        'bet_filter_tier': bet_filter_result.get('tier', 'unknown'),
        'bet_filter_reason': bet_filter_result.get('reason', ''),
        # Phase 3.3: Threes-specific metadata (empty for other prop types)
        'threes_streak_type': threes_streak_info.get('streak_type', 'neutral'),
        'threes_streak_fade': threes_streak_info.get('streak_fade', 0.0),
        'threes_streak_details': threes_streak_info.get('details', ''),
        # Data-quality signal: 'full' (≥120 features) or 'degraded' (<120) or
        # 'unknown' (no API features used). Downstream bet filters should
        # require 'full' for production-tier bet sizing.
        'feature_quality': locals().get('_feature_quality', 'unknown'),
        'feature_count': len(features) if features else 0,
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
    parser.add_argument(
        "--prop-source",
        choices=["odds-api", "balldontlie", "hybrid"],
        default="hybrid",
        help="Player prop line source: odds-api (FD/DK only), balldontlie (legacy), hybrid (try Odds API first, fallback to BDL)",
    )
    parser.add_argument(
        "--max-prop-games",
        type=int,
        default=0,
        help="Limit Odds API prop fetches to N games (0=all). Use 1-2 for testing.",
    )
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
        datetime.strptime(target_date, "%Y-%m-%d")
        current_injuries = fetch_current_injuries()

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

    # Lineup intelligence: starter confirmations + minutes-restriction multipliers.
    # injury_tracker_v3 gives binary OUT/DOUBTFUL/QUESTIONABLE — useful but coarse.
    # LineupIntelService adds:
    #   - starter_confidence (is_starter, useful for rookies and rotation moves)
    #   - availability_probability (continuous, e.g., 0.6 for "GTD lean Q")
    #   - minutes_multiplier (e.g., 0.65 for a player on a hard minutes cap
    #     returning from injury — predictions on raw season averages will be
    #     wildly wrong without this)
    # Keyed by player_name.lower() since BDL IDs aren't always available upstream.
    lineup_intel_lookup: dict[str, dict] = {}
    try:
        from lineup_intel import LineupIntelService  # noqa: PLC0415
        _lis = LineupIntelService()
        _team_abbrevs: set[str] = set()
        # Defer collection until games are loaded — bail gracefully if unavailable.
        # We only query intel per-player inside the prop_tasks loop to avoid
        # rate-limiting the underlying news/injury scrapers.
        lineup_intel_service = _lis
        print("  Lineup intel service loaded (queries deferred per player)")
    except ImportError:
        lineup_intel_service = None
        print("  Lineup intel service unavailable (module not installed)")
    except Exception as e:
        lineup_intel_service = None
        print(f"  Lineup intel service init failed: {e}")

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

    # Initialize PlayerPropFetcher for Odds API prop lines (FanDuel/DraftKings)
    prop_fetcher = None
    event_map = {}  # {bdl_game_id: {event_id, home_team, ...}}
    prop_source = getattr(args, 'prop_source', 'hybrid')
    max_prop_games = getattr(args, 'max_prop_games', 0)

    if prop_source in ("odds-api", "hybrid"):
        try:
            from odds_fetcher import PlayerPropFetcher
            prop_fetcher = PlayerPropFetcher()
            events = prop_fetcher.fetch_todays_events()
            if events:
                event_map = prop_fetcher.match_events_to_games(events, games)
                print(f"  Odds API: {len(events)} events, {len(event_map)} matched to games")
                if max_prop_games > 0:
                    # Limit to first N games
                    limited = dict(list(event_map.items())[:max_prop_games])
                    print(f"  Odds API: Limited to {len(limited)} games (--max-prop-games {max_prop_games})")
                    event_map = limited
                print(f"  Prop source: {prop_source} | API credits remaining: {prop_fetcher.remaining_requests}")
            else:
                print("  Odds API: No events returned (games may not have props yet)")
        except Exception as e:
            print(f"  Odds API init failed: {e}")
            if prop_source == "odds-api":
                print("  WARNING: --prop-source=odds-api but Odds API unavailable!")

    # TASK 4.1: Cache warmup - pre-fetch all team/player data in parallel
    # Also cache props data to avoid duplicate API calls
    props_cache = {}  # Cache props data for reuse in main loop
    props_source_cache = {}  # Track source per game: {game_id: 'odds-api'|'balldontlie'}

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
        # Uses hybrid fetching: Odds API (FD/DK) -> Balldontlie fallback
        print("\n  Collecting player IDs for cache warmup...", end='', flush=True)
        id_mapper_warmup = get_id_mapper()
        for game in games:
            game_id = game.get('id')
            if game_id:
                try:
                    props_data, source = get_player_props_hybrid(
                        game_id=game_id,
                        prop_source=prop_source,
                        api=api,
                        event_map=event_map,
                        prop_fetcher=prop_fetcher,
                        id_mapper=id_mapper_warmup,
                    )
                    if props_data:
                        # Cache props for later use in main loop
                        props_cache[game_id] = props_data
                        props_source_cache[game_id] = source

                        # Get players with significant lines (likely to be analyzed)
                        for pid, props in props_data.items():
                            if props.get('points_line', 0) >= 15:
                                player_ids_to_warm.append(pid)
                except Exception as e:
                    print(f" [game {game_id} error: {e}]", end='')

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
        if game_id and (api or prop_fetcher):
            home = analysis['home_team']
            away = analysis['away_team']
            print(f"\n  Analyzing {away}@{home} props...", end="", flush=True)

            # OPTIMIZATION: Use cached props if available (from warmup phase)
            props_data = props_cache.get(game_id)
            game_prop_source = props_source_cache.get(game_id, "unknown")
            if not props_data:
                # Fall back to hybrid fetch if not in cache
                props_data, game_prop_source = get_player_props_hybrid(
                    game_id=game_id,
                    prop_source=prop_source,
                    api=api,
                    event_map=event_map,
                    prop_fetcher=prop_fetcher,
                    id_mapper=get_id_mapper(),
                )

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

                    # Fallback: use Odds API player name or Balldontlie API
                    missing_ids = [pid for pid, _ in sorted_players if pid not in player_names]
                    if missing_ids:
                        for pid in missing_ids:
                            # First check if Odds API provided the name
                            odds_api_name = props_data.get(pid, {}).get('player_name_odds_api')
                            if odds_api_name:
                                player_names[pid] = odds_api_name
                                continue
                            # Fall back to Balldontlie API
                            try:
                                if api:
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

                    # ================================================================
                    # PHASE 2: Compute game-level schedule / travel context
                    # ================================================================
                    try:
                        from nba_data.transformers.travel_fatigue import TravelFatigueCalculator
                        _travel_calc = TravelFatigueCalculator()

                        # Compute season phase from target_date
                        _td_month = int(target_date[5:7])
                        if _td_month in (10, 11):
                            _season_phase = 0   # early season
                        elif _td_month in (12, 1):
                            _season_phase = 1   # mid season
                        elif _td_month in (2, 3):
                            _season_phase = 2   # late season (post All-Star)
                        else:
                            _season_phase = 3   # playoff push (Apr+)

                        # Fetch recent game schedules for both teams for rest/travel
                        def _get_recent_team_games(team_id, before_date, limit=7):
                            """Fetch team's recent games from API for fatigue calc."""
                            try:
                                if not api:
                                    return []
                                recent = api.get_games(
                                    team_ids=[team_id],
                                    per_page=limit,
                                )
                                # Filter completed games before today and format for TravelFatigueCalculator
                                result = []
                                for g in recent:
                                    gd = g.get('date', '') or ''
                                    if isinstance(gd, str) and 'T' in gd:
                                        gd = gd.split('T')[0]
                                    if gd and gd < before_date and g.get('status') == 'Final':
                                        ht = g.get('home_team', {}) or {}
                                        result.append({
                                            'date': gd,
                                            'home_team_id': ht.get('id', team_id),
                                        })
                                result.sort(key=lambda x: x['date'], reverse=True)
                                return result[:limit]
                            except Exception:
                                return []

                        _home_recent = _get_recent_team_games(home_team_id, target_date)
                        _away_recent = _get_recent_team_games(away_team_id, target_date)

                        _home_travel = _travel_calc.get_travel_features(
                            team_id=home_team_id,
                            game_date=target_date,
                            opponent_id=away_team_id,
                            is_home=True,
                            team_games=_home_recent,
                        )
                        _away_travel = _travel_calc.get_travel_features(
                            team_id=away_team_id,
                            game_date=target_date,
                            opponent_id=home_team_id,
                            is_home=False,
                            team_games=_away_recent,
                        )

                        # Compute opp_def_tier from analysis stats
                        _home_opp_stats = analysis.get('away_stats', {}) or {}
                        _away_opp_stats = analysis.get('home_stats', {}) or {}
                        _home_def_rating = _home_opp_stats.get('def_rating', 114)
                        _away_def_rating = _away_opp_stats.get('def_rating', 114)

                        # Simple tier classification using absolute thresholds
                        def _def_tier(dr):
                            if not dr:
                                return 2
                            if dr <= 111.0:
                                return 1  # elite
                            elif dr >= 117.0:
                                return 3  # weak
                            return 2

                        _home_def_tier = _def_tier(_away_def_rating)  # home team faces away's defense
                        _away_def_tier = _def_tier(_home_def_rating)  # away team faces home's defense

                    except Exception:
                        _home_travel = {'days_rest': 2, 'is_back_to_back': 0, 'travel_distance': 0, 'games_last_7_days': 3}
                        _away_travel = {'days_rest': 2, 'is_back_to_back': 0, 'travel_distance': 0, 'games_last_7_days': 3}
                        _home_def_tier = 2
                        _away_def_tier = 2
                        _season_phase = 1

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
                        # Fix 0.2: Enhanced DNP filter — skip OUT/DOUBTFUL players AND
                        # players with avg minutes < 15 to avoid junk predictions.
                        uncertainty_flag = None
                        if player_id in injury_lookup:
                            status = injury_lookup[player_id]
                            if status in [InjuryStatus.OUT, InjuryStatus.DOUBTFUL]:
                                print(f"    Skipping {player_name} ({status}) [DNP filter]")
                                continue
                            if status in [InjuryStatus.QUESTIONABLE, InjuryStatus.GTD]:
                                uncertainty_flag = "HIGH_UNCERTAINTY"

                        # Layered check: LineupIntelService provides finer-grained
                        # signal than injury_tracker — most importantly the
                        # minutes_multiplier for players on hard minutes caps
                        # returning from injury. Defaults: availability=1.0,
                        # mins_mult=1.0 (no restriction).
                        player_minutes_multiplier = 1.0
                        player_availability_prob = 1.0
                        if lineup_intel_service is not None and player_name:
                            try:
                                _li_team = (analysis['home_team']
                                            if player_team_id == home_team_id
                                            else analysis['away_team'])
                                _li = lineup_intel_service.get_player_intel(
                                    player_name=player_name,
                                    team=_li_team,
                                )
                                if _li is not None:
                                    player_minutes_multiplier = float(
                                        getattr(_li, 'minutes_multiplier', 1.0)
                                    )
                                    player_availability_prob = float(
                                        getattr(_li, 'availability_probability', 1.0)
                                    )
                                    # Hard gate: availability < 0.5 means lineup
                                    # intel believes the player is more likely to
                                    # not play than play. Skip even if injury
                                    # tracker said QUESTIONABLE — lineup intel
                                    # aggregates more sources.
                                    if player_availability_prob < 0.5:
                                        print(
                                            f"    Skipping {player_name} "
                                            f"(lineup intel availability "
                                            f"{player_availability_prob:.0%}) [DNP filter]"
                                        )
                                        continue
                                    if (player_minutes_multiplier < 0.85
                                            and uncertainty_flag is None):
                                        uncertainty_flag = "MINUTES_RESTRICTION"
                            except Exception:
                                # Lineup intel is best-effort. Never let it block
                                # a prediction — the rest of the pipeline still
                                # has injury_tracker + minutes_oracle to fall back on.
                                pass

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

                        # Fix 0.2: Skip low-minutes players (bench warmers)
                        if bdl_stats_id and api:
                            try:
                                _stats = api.get_season_averages(
                                    season=int(target_date[:4]),
                                    player_ids=[bdl_stats_id],
                                )
                                if _stats:
                                    _player_avg_min = _stats[0].get('min', 0) or 0
                                    if isinstance(_player_avg_min, str):
                                        _player_avg_min = float(str(_player_avg_min).split(':')[0])
                                    if 0 < _player_avg_min < 15:
                                        print(f"    Skipping {player_name} (avg {_player_avg_min:.0f} min) [low minutes]")
                                        continue
                            except Exception:
                                pass

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

                        # Determine player's team abbreviation
                        team_abbrev = analysis['home_team'] if player_team_id == home_team_id else analysis['away_team']

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
                                    'team_abbrev': team_abbrev,
                                    # Lineup intel signals (minutes_multiplier
                                    # captures hard minutes caps for returning
                                    # players that the season-average baseline
                                    # cannot otherwise pick up).
                                    'minutes_multiplier': player_minutes_multiplier,
                                    'availability_probability': player_availability_prob,
                                    'over_odds': props.get(f'{prop_type}_over_odds', -110),
                                    'under_odds': props.get(f'{prop_type}_under_odds', -110),
                                    'line_source': game_prop_source,
                                    'line_vendor': props.get(f'{prop_type}_vendor', 'unknown'),
                                    # Phase 4.2: Line shopping fields (populated when source=odds-api)
                                    'best_over_book': props.get(f'{prop_type}_best_over_book'),
                                    'best_over_odds': props.get(f'{prop_type}_best_over_odds'),
                                    'best_under_book': props.get(f'{prop_type}_best_under_book'),
                                    'best_under_odds': props.get(f'{prop_type}_best_under_odds'),
                                    'implied_prob_over': props.get(f'{prop_type}_implied_prob_over'),
                                    'per_book_odds': props.get(f'{prop_type}_per_book', []),
                                    'game_context': {
                                        'spread': odds.get('spread', 0),
                                        'total': odds.get('total', 220),
                                        'is_home': player_team_id == home_team_id,
                                        'opponent_team_id': opponent_id,
                                        # Phase 2.2: Player team schedule context
                                        'days_rest': (_home_travel if player_team_id == home_team_id else _away_travel).get('days_rest', 2),
                                        'is_b2b': bool((_home_travel if player_team_id == home_team_id else _away_travel).get('is_back_to_back', 0)),
                                        'travel_distance': (_home_travel if player_team_id == home_team_id else _away_travel).get('travel_distance', 0.0),
                                        'games_last_7_days': (_home_travel if player_team_id == home_team_id else _away_travel).get('games_last_7_days', 3),
                                        'season_phase': _season_phase,
                                        'is_b2b_home': bool((_home_travel if player_team_id == home_team_id else _away_travel).get('is_back_to_back', 0) and player_team_id == home_team_id),
                                        'is_b2b_away': bool((_home_travel if player_team_id == home_team_id else _away_travel).get('is_back_to_back', 0) and player_team_id != home_team_id),
                                        # Phase 2.1: Opponent schedule context
                                        'opp_days_rest': (_away_travel if player_team_id == home_team_id else _home_travel).get('days_rest', 2),
                                        'opp_is_back_to_back': bool((_away_travel if player_team_id == home_team_id else _home_travel).get('is_back_to_back', 0)),
                                        'opp_def_tier': _away_def_tier if player_team_id == home_team_id else _home_def_tier,
                                    },
                                })

                    # TASK 4.1: Execute prop predictions in parallel
                    if prop_tasks:
                        executor = get_executor(max_workers=5)

                        # Phase 4.3: Initialise line-movement tracker once per game loop
                        _prop_tracker = get_prop_tracker() if HAS_PROP_TRACKER else None

                        def process_prop_task(task, _prop_tracker=_prop_tracker):
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
                                american_odds=task.get('over_odds', -110),
                                under_odds=task.get('under_odds'),
                                minutes_multiplier=task.get('minutes_multiplier', 1.0),
                                availability_probability=task.get('availability_probability', 1.0),
                            )
                            # predict_player_prop returns None on early-exit paths
                            # (empty name, incomplete features, sub-15-min DNP, threes
                            # sample gate). The downstream consumer filters `if pred:`,
                            # so propagate None instead of mutating it.
                            if pred is None:
                                return None
                            # Fix 1: Pass through team abbreviation
                            pred['team_abbrev'] = task.get('team_abbrev', '')
                            # Fix 4: Pass through actual sportsbook odds
                            pred['over_odds'] = task.get('over_odds', -110)
                            pred['under_odds'] = task.get('under_odds')
                            # Track line source (odds-api vs balldontlie) and vendor
                            pred['line_source'] = task.get('line_source', 'unknown')
                            pred['line_vendor'] = task.get('line_vendor', 'unknown')

                            # Phase 4.2: Line shopping — best available odds across all books
                            pred['best_over_book'] = task.get('best_over_book')
                            pred['best_over_odds'] = task.get('best_over_odds')
                            pred['best_under_book'] = task.get('best_under_book')
                            pred['best_under_odds'] = task.get('best_under_odds')
                            # Implied probability from vig-free devigging (best available odds)
                            pred['implied_prob_over'] = task.get('implied_prob_over')
                            # Per-book EV list for dashboard line-shopping display
                            pred['per_book_odds'] = task.get('per_book_odds', [])

                            # Phase 4.3: Line movement signal — does smart money agree with our model?
                            pick = pred.get('pick', '-')
                            line_movement_signal = 'NEUTRAL'
                            if _prop_tracker and pick in ('OVER', 'UNDER'):
                                try:
                                    line_movement_signal = _prop_tracker.get_movement_signal(
                                        game_date=target_date,
                                        player_name=task['player_name'],
                                        prop_type=task['prop_type'],
                                        pick=pick,
                                    )
                                except Exception:
                                    pass
                            pred['line_movement_signal'] = line_movement_signal

                            # Derive the "best book" for the recommended pick side
                            if pick == 'OVER':
                                pred['best_book'] = pred.get('best_over_book') or task.get('line_vendor', 'unknown')
                                pred['best_odds'] = pred.get('best_over_odds') or task.get('over_odds', -110)
                            elif pick == 'UNDER':
                                pred['best_book'] = pred.get('best_under_book') or task.get('line_vendor', 'unknown')
                                pred['best_odds'] = pred.get('best_under_odds') or task.get('under_odds', -110)
                            else:
                                pred['best_book'] = task.get('line_vendor', 'unknown')
                                pred['best_odds'] = task.get('over_odds', -110)

                            # Fix 5: Expand uncertainty_flag beyond just injury status
                            if task['uncertainty_flag']:
                                pred['uncertainty_flag'] = task['uncertainty_flag']
                            elif pred.get('minutes_uncertainty') == 'high':
                                pred['uncertainty_flag'] = 'HIGH_MINUTES_UNCERTAINTY'
                            elif pred.get('confidence_score', 100) < 35:
                                pred['uncertainty_flag'] = 'LOW_CONFIDENCE'
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
                    edges = [p.get('edge', 0) for p in analysis['player_props'] if abs(p.get('edge', 0)) > 3]
                    print(f" {prop_count} props analyzed, {len(edges)} edges found")
            else:
                print(" no props available")

        all_analyses.append(analysis)
        print_game_analysis(analysis)

    # Portfolio-level exposure caps. Per-bet Kelly is correct for the marginal
    # bet but produces concentrated exposure when multiple high-edge bets land
    # on the same game/player/prop type — all of which are heavily correlated.
    # This pass admits bets greedily by edge until each bucket cap is hit and
    # marks the rest with cap_rejected_reason so they remain visible.
    if all_player_props:
        try:
            from nba_betting.exposure_caps import apply_exposure_caps
            _cap_summary = apply_exposure_caps(all_player_props)
            print(
                f"  Exposure caps: {_cap_summary['admitted']}/"
                f"{_cap_summary['eligible']} bets admitted "
                f"(total exposure {_cap_summary['total_exposure']:.1%}, "
                f"{_cap_summary['rejected']} rejected by caps)"
            )
            if _cap_summary['rejections_by_reason']:
                for reason, count in _cap_summary['rejections_by_reason'].items():
                    print(f"    - {reason}: {count}")
        except Exception as e:
            print(f"  Warning: exposure cap pass failed: {e}")

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
                        team=prop.get('team_abbrev', ''),
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

    # CLV health surface: print rolling 7-day CLV summary so the operator sees
    # whether the model is sharp on every run. Beating the closing line is the
    # only honest signal of model alpha — short-term win rate is mostly noise.
    # Failure modes this catches:
    #   - closing_odds_scheduler isn't running (settled_bets stays at 0)
    #   - model edge has decayed (avg_clv trends negative)
    try:
        from nba_betting.edge.clv_analyzer import CLVAnalyzer
        _clv_summary = CLVAnalyzer().get_clv_summary(days=7)
        if _clv_summary:
            _settled = _clv_summary.get('settled_bets', 0)
            _total = _clv_summary.get('total_bets', 0)
            _avg = _clv_summary.get('avg_clv', 0.0)
            _pos_rate = _clv_summary.get('positive_clv_rate', 0.0)
            print(
                f"  7-day CLV: avg {_avg:+.2f}% | positive rate {_pos_rate:.0%} | "
                f"{_settled}/{_total} settled"
            )
            if _total > 0 and _settled == 0:
                print(
                    "  WARN: bets recorded but none settled — "
                    "closing_odds_scheduler may not be running"
                )
            if _settled >= 50 and _avg < -1.0:
                print(
                    f"  WARN: 7-day CLV is {_avg:+.2f}% (significantly negative) — "
                    "model may have lost edge; investigate before increasing stakes"
                )
    except ImportError:
        # CLVAnalyzer module not installed in this environment — quiet skip.
        pass
    except Exception as _clv_exc:
        # CLV reporting is informational only — never let it break the run.
        # But surface the failure (DB schema mismatch, missing table, etc.) so
        # the operator knows CLV is silently broken instead of falsely assuming
        # it's healthy. Print AND log so it appears in both stdout and logs.
        print(f"  WARN: CLV summary failed: {type(_clv_exc).__name__}: {_clv_exc}")
        logger.warning("CLV summary failed: %s", _clv_exc, exc_info=True)

    # Phase 4.3: Store prop odds snapshots for line movement tracking
    if HAS_PROP_TRACKER and all_player_props:
        try:
            _snap_tracker = get_prop_tracker()
            snap_records = []
            for prop in all_player_props:
                player_name = prop.get('player', '')
                prop_type = prop.get('stat', '')
                line = prop.get('line', 0)
                over_odds_snap = prop.get('over_odds', -110)
                under_odds_snap = prop.get('under_odds', -110)
                book = prop.get('line_vendor', 'unknown')
                if player_name and prop_type and line:
                    snap_records.append({
                        'player_name': player_name,
                        'prop_type': prop_type,
                        'book_name': book or 'unknown',
                        'line': line,
                        'over_odds': over_odds_snap or -110,
                        'under_odds': under_odds_snap or -110,
                    })
                    # Also store per-book snapshots from line shopping
                    for pb in prop.get('per_book_odds', []):
                        if pb.get('book') and pb['book'] != book:
                            snap_records.append({
                                'player_name': player_name,
                                'prop_type': prop_type,
                                'book_name': pb['book'],
                                'line': pb.get('line', line),
                                'over_odds': pb.get('over_odds', -110),
                                'under_odds': pb.get('under_odds', -110),
                            })
            stored_count = _snap_tracker.store_snapshots_bulk(
                game_date=target_date,
                props=snap_records,
                is_opening=False,  # Daily predictions run at 9 AM; odds may have opened earlier
            )
            print(f"  Stored {stored_count} prop odds snapshots for line movement tracking")
        except Exception as e:
            print(f"  Warning: Prop snapshot storage failed: {e}")

    # Paper trading: log all predictions for forward validation
    try:
        from nba_betting.paper_trading import PaperTrader
        trader = PaperTrader()
        paper_preds = []
        for prop in all_player_props:
            paper_preds.append({
                "game_date": target_date,
                "game_id": prop.get("game", ""),
                "player_name": prop.get("player", ""),
                "prop_type": prop.get("stat", ""),
                "line": prop.get("line", 0),
                "direction": prop.get("pick", "over"),
                "predicted_value": prop.get("predicted_value"),
                "over_prob": prop.get("over_prob"),
                "edge": prop.get("edge"),
                "ev": prop.get("ev_per_dollar"),
                "should_bet": prop.get("bet_recommendation", prop.get("signal", "PASS")) == "BET",
                "bet_size": prop.get("suggested_bet_size") or 0,
                "over_odds": prop.get("over_odds", -110),
                "under_odds": prop.get("under_odds", -110),
                "confidence": prop.get("confidence_score"),
                "tier": prop.get("edge_quality_tier", ""),
            })
        if paper_preds:
            paper_count = trader.log_predictions_batch(paper_preds, target_date)
            print(f"  Paper trading: logged {paper_count} predictions for {target_date}")
    except Exception as e:
        print(f"  Warning: Paper trading log failed (non-blocking): {e}")

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

        # Spread edges — only when betting is enabled (Phase 3.1: disabled,
        # RMSE 14.2 > market 12-13; spread output is used as ML feature instead)
        if SPREAD_BETTING_ENABLED and a['spread']['edge_pct'] > 5:
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
                vendor = prop.get('line_vendor', 'unknown')
                source = prop.get('line_source', 'unknown')
                bettable = vendor in ('draftkings', 'fanduel')
                recommendations.append({
                    'game': f"{away}@{home}",
                    'bet': f"{prop['player']} {prop['stat']} {direction} {prop['line']}",
                    'prob': prob,
                    'edge': prop.get('edge', 0),
                    'signal': signal,
                    'line_vendor': vendor,
                    'line_source': source,
                    'bettable': bettable,
                })

    # Sort by edge
    recommendations.sort(key=lambda x: x['edge'], reverse=True)

    if recommendations:
        # When using odds-api or hybrid, prioritize bettable lines from FD/DK
        bettable_recs = [r for r in recommendations if r.get('bettable', True)]
        non_bettable = [r for r in recommendations if not r.get('bettable', True)]

        print()
        display_recs = bettable_recs[:10] if bettable_recs else recommendations[:10]
        for i, rec in enumerate(display_recs, 1):
            vendor_tag = f" [{rec.get('line_vendor', '')}]" if rec.get('line_vendor', 'unknown') != 'unknown' else ""
            print(f"  {i}. {rec['game']}: {rec['bet']} ({rec['prob']:.0%}, edge: {rec['edge']:+.1f}%){vendor_tag}")

        if non_bettable and prop_source in ('odds-api', 'hybrid'):
            print(f"\n  NOTE: {len(non_bettable)} additional edges from non-FD/DK sources (not shown)")
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
                game_str = prop.get('game', '')
                team = prop.get('team_abbrev', '')

                # Phase 4: Use pick directly from prediction output
                over_prob = prop.get('over_prob', 0.5)
                bet_rec = prop.get('bet_recommendation', 'PASS')
                pick = prop.get('pick', 'OVER' if over_prob > 0.5 else 'UNDER')

                # Use pick-appropriate odds (over_odds for OVER, under_odds for UNDER)
                if pick == 'OVER':
                    display_odds = prop.get('over_odds', prop.get('american_odds', -110))
                else:
                    display_odds = prop.get('under_odds', prop.get('american_odds', -110))

                # Phase 4.1: Compute EV in dollar terms ($100 stake) for display
                ev_per_dollar = prop.get('ev_per_dollar', 0) or 0
                ev_dollars_100 = round(ev_per_dollar * 100, 2)

                row = {
                    'date': target_date,
                    'game': game_str,
                    'player_name': prop.get('player', ''),
                    'team': team,
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
                    'american_odds': display_odds,
                    # Phase 4.1: Real odds integration
                    'implied_probability': prop.get('implied_probability') or prop.get('implied_prob_over'),
                    'ev_per_dollar': ev_per_dollar,
                    'ev_dollars_100': ev_dollars_100,
                    # Phase 4.2: Line shopping
                    'best_odds': prop.get('best_odds'),
                    'best_book': prop.get('best_book'),
                    'over_odds': prop.get('over_odds', -110),
                    'under_odds': prop.get('under_odds', -110),
                    # Phase 4.3: Line movement signal
                    'line_movement_signal': prop.get('line_movement_signal', 'NEUTRAL'),
                    'has_edge': prop.get('has_edge', False),
                    'uncertainty_flag': prop.get('uncertainty_flag', ''),
                    'injury_boost': prop.get('injury_boost', 1.0),
                    'line_source': prop.get('line_source', 'unknown'),
                    'line_vendor': prop.get('line_vendor', 'unknown'),
                    # Portfolio-cap admission flags. cap_admitted=True → eligible
                    # to stake at suggested_bet_size; False → rejected by an
                    # exposure cap, do NOT stake; None → not a BET-tier signal.
                    'cap_admitted': prop.get('cap_admitted'),
                    'cap_rejected_reason': prop.get('cap_rejected_reason', ''),
                    # Data-quality signal added by Audit 2026-05-15.
                    'feature_quality': prop.get('feature_quality', 'unknown'),
                    'feature_count': prop.get('feature_count', 0),
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
                            -- injury_boost is a numeric multiplier (e.g., 1.05
                            -- = +5% boost), NOT a boolean. The original schema
                            -- typed it as BOOLEAN by mistake — every daily run
                            -- since failed DB persistence with "column is of
                            -- type boolean but expression is of type numeric"
                            -- and silently fell back to CSV-only. The ALTER
                            -- TABLE in _new_cols below fixes existing tables.
                            injury_boost DOUBLE PRECISION,
                            line_source VARCHAR(20),
                            line_vendor VARCHAR(50),
                            created_at TIMESTAMP DEFAULT NOW(),
                            UNIQUE(date, player_name, prop_type)
                        )
                    """)
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions_history(date)")
                    # Add new columns if table already exists (idempotent)
                    # Migration 010 adds line_source/line_vendor; migration 012 adds Phase 4 fields.
                    _new_cols = [
                        ("line_source", "VARCHAR(50)"),
                        ("line_vendor", "VARCHAR(50)"),
                        ("implied_probability", "FLOAT"),
                        ("ev_per_dollar", "FLOAT"),
                        ("best_odds", "INTEGER"),
                        ("best_book", "VARCHAR(50)"),
                        ("line_movement_signal", "VARCHAR(20)"),
                        ("opening_line", "FLOAT"),
                        ("over_odds", "INTEGER"),
                        ("under_odds", "INTEGER"),
                    ]
                    for col, col_type in _new_cols:
                        try:
                            cursor.execute(f"ALTER TABLE predictions_history ADD COLUMN {col} {col_type}")
                        except Exception:
                            conn.rollback()  # Column already exists — safe to ignore

                    # One-shot migration: change injury_boost from BOOLEAN to
                    # DOUBLE PRECISION on existing tables. The USING clause
                    # coerces existing true/false rows to 1.0/0.0 so legacy
                    # rows survive the type change. Idempotent — succeeds even
                    # if the column is already DOUBLE PRECISION (becomes a
                    # no-op ALTER).
                    try:
                        cursor.execute(
                            "ALTER TABLE predictions_history "
                            "ALTER COLUMN injury_boost TYPE DOUBLE PRECISION "
                            "USING (CASE WHEN injury_boost IS NULL THEN NULL "
                            "WHEN injury_boost = TRUE THEN 1.0 "
                            "WHEN injury_boost = FALSE THEN 0.0 ELSE 1.0 END)"
                        )
                    except Exception:
                        # Either the column is already DOUBLE PRECISION (no-op
                        # already committed) or the alter genuinely failed; in
                        # both cases roll back so subsequent statements work.
                        conn.rollback()
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
                                pick, uncertainty_flag, injury_boost,
                                line_source, line_vendor,
                                implied_probability, ev_per_dollar,
                                best_odds, best_book, line_movement_signal,
                                over_odds, under_odds
                            ) VALUES (
                                %s, %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s,
                                %s, %s, %s,
                                %s, %s,
                                %s, %s,
                                %s, %s, %s,
                                %s, %s
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
                            safe_val(row.get('injury_boost')),
                            safe_val(row.get('line_source')),
                            safe_val(row.get('line_vendor')),
                            # Phase 4.1: Real odds integration fields
                            safe_val(row.get('implied_probability')),
                            safe_val(row.get('ev_per_dollar')),
                            # Phase 4.2: Line shopping fields
                            safe_val(row.get('best_odds')),
                            safe_val(row.get('best_book')),
                            # Phase 4.3: Line movement signal
                            safe_val(row.get('line_movement_signal')),
                            safe_val(row.get('over_odds')),
                            safe_val(row.get('under_odds')),
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
