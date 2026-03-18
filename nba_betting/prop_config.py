"""
Per-Prop Model Configuration Registry (Fix 6.1)

Central registry for per-prop-type model configuration. Replaces scattered
constants across constants.py, bet_filter.py, and prediction_pipeline.py
with a single dataclass-based config per prop type.

Every prop type has:
  - Feature list (from Fix 1.1 REDUCED_FEATURES)
  - Quality thresholds (R², RMSE improvement, bias)
  - Betting thresholds (min edge, min confidence, min EV)
  - Enable/disable flag (gated by baseline_comparison.py results)
  - Hyperparameters (LightGBM defaults, can be overridden by Optuna)

Usage:
    from nba_betting.prop_config import PROP_REGISTRY, get_prop_config

    cfg = get_prop_config('points')
    if cfg.enabled:
        features = cfg.features
        model = train_prop_model(X[features], y, **cfg.hyperparameters)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PropModelConfig:
    """Configuration for a single prop type model."""

    # Identity
    prop_type: str

    # Enable/disable — gated by baseline_comparison.py (Fix 5.2)
    enabled: bool = False

    # Feature list (Fix 1.1: reduced from 80+ to 15-20)
    features: list[str] = field(default_factory=list)

    # Target column in training data
    target_col: str = ''

    # Season average column for residual prediction (Fix 1.4)
    season_avg_col: str | None = None

    # Quality thresholds — model must meet ALL to be enabled
    min_r2: float = 0.02
    max_bias: float = 1.0
    min_rmse_improvement_pct: float = 1.0  # Must beat season avg by at least 1%

    # Betting thresholds
    min_edge: float = 3.0          # Minimum stat edge to consider
    min_confidence: float = 0.62   # Minimum calibrated probability
    min_ev: float = 0.03           # Minimum true EV when odds available

    # Standard deviation for Z-score computation
    std_dev: float = 5.0

    # LightGBM hyperparameters (defaults, overridable by Optuna)
    hyperparameters: dict = field(default_factory=lambda: {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'objective': 'mae',
    })


# ---------------------------------------------------------------------------
# Registry: one config per prop type
# ---------------------------------------------------------------------------

PROP_REGISTRY: dict[str, PropModelConfig] = {
    'points': PropModelConfig(
        prop_type='points',
        enabled=False,  # Fix 5.2: disabled until proven
        target_col='actual_pts',
        season_avg_col='season_pts_avg',
        std_dev=6.16,
        min_edge=3.0,
        features=[
            'season_pts_avg', 'last5_pts_avg', 'last3_pts_avg', 'recent_pts_avg',
            'season_min_avg', 'last5_min_avg', 'predicted_minutes',
            'opp_def_rating', 'opp_pts_allowed',
            'opp_pace', 'is_home', 'days_rest', 'usage_rate', 'ts_pct',
            'pts_trend', 'pts_recency_ratio', 'season_games',
            'prop_line_vs_recent',
        ],
    ),
    'rebounds': PropModelConfig(
        prop_type='rebounds',
        enabled=False,
        target_col='actual_reb',
        season_avg_col='season_reb_avg',
        std_dev=2.67,
        min_edge=2.0,
        features=[
            'season_reb_avg', 'last5_reb_avg', 'last3_reb_avg', 'recent_reb_avg',
            'season_min_avg', 'last5_min_avg', 'predicted_minutes',
            'opp_reb_factor',
            'is_center', 'is_forward', 'opp_pace', 'is_home', 'days_rest',
            'reb_trend', 'reb_recency_ratio', 'season_games',
            'prop_line_vs_recent',
        ],
    ),
    'assists': PropModelConfig(
        prop_type='assists',
        enabled=False,
        target_col='actual_ast',
        season_avg_col='season_ast_avg',
        std_dev=1.95,
        min_edge=2.0,
        features=[
            'season_ast_avg', 'last5_ast_avg', 'last3_ast_avg', 'recent_ast_avg',
            'season_min_avg', 'last5_min_avg', 'predicted_minutes',
            'opp_def_rating',
            'is_guard', 'is_ball_handler', 'opp_pace', 'is_home', 'days_rest',
            'ast_trend', 'ast_recency_ratio', 'season_games',
            'prop_line_vs_recent',
        ],
    ),
    'threes': PropModelConfig(
        prop_type='threes',
        enabled=False,
        target_col='actual_fg3m',
        season_avg_col='season_fg3m_avg',
        std_dev=1.36,
        min_edge=1.0,
        features=[
            'season_fg3m_avg', 'last5_fg3m_avg', 'last3_fg3m_avg', 'recent_fg3m_avg',
            'season_min_avg', 'predicted_minutes',
            'fg3a_avg', 'fg3_pct', 'regressed_fg3_pct',
            'fg3_rate', 'fg3_momentum', 'is_volume_shooter',
            'opp_pace', 'is_home', 'days_rest', 'season_games',
            'prop_line_vs_recent',
        ],
    ),
    'pra': PropModelConfig(
        prop_type='pra',
        enabled=False,
        target_col='actual_pra',
        season_avg_col=None,  # Computed as sum of pts+reb+ast avgs
        std_dev=7.97,
        min_edge=4.0,
        features=[
            'season_pts_avg', 'season_reb_avg', 'season_ast_avg',
            'last5_pts_avg', 'last5_reb_avg', 'last5_ast_avg',
            'pra_avg', 'last3_pra_avg', 'season_min_avg', 'last5_min_avg', 'predicted_minutes',
            'opp_def_rating', 'opp_pace', 'is_home', 'days_rest',
            'usage_rate', 'season_games',
            'prop_line_vs_recent',
        ],
    ),
    'spread': PropModelConfig(
        prop_type='spread',
        enabled=False,
        target_col='point_differential',
        season_avg_col=None,
        std_dev=12.0,
        min_edge=2.0,
        min_ev=0.04,
        features=[],  # Team model — uses team features, not player features
    ),
}


def get_prop_config(prop_type: str) -> PropModelConfig:
    """Get configuration for a prop type. Returns a disabled config for unknown types."""
    return PROP_REGISTRY.get(
        prop_type.lower(),
        PropModelConfig(prop_type=prop_type.lower(), enabled=False),
    )


def get_enabled_props() -> list[str]:
    """Return list of prop types that are currently enabled."""
    return [name for name, cfg in PROP_REGISTRY.items() if cfg.enabled]


def enable_prop(prop_type: str) -> None:
    """Enable a prop type (call after baseline_comparison.py validates it)."""
    if prop_type in PROP_REGISTRY:
        PROP_REGISTRY[prop_type].enabled = True


def disable_prop(prop_type: str) -> None:
    """Disable a prop type."""
    if prop_type in PROP_REGISTRY:
        PROP_REGISTRY[prop_type].enabled = False
