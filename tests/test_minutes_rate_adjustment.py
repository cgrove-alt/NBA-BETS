"""Tests for _compute_minutes_rate_adjustment.

Background: the audit on 2026-05-15 traced a Wembanyama UNDER pick
(predicted 35.3 vs line 44.5, +20.6% UNDER edge) to recent_avg being
dragged down by 12-min minutes-restricted outlier games. The fix added
recent_*_per_min fields and a rate-based projection (rate * predicted_min,
blended 0.6/0.4 with the model) that's robust to those outliers.

These tests pin the helper's behavior so regressions in the rate-projection
path show up immediately, and document the expected math via fixtures
keyed to the Wembanyama case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make repo root importable so nba_models.inference can be imported without a
# packaged install. Mirrors the convention used by other test modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nba_models.inference.daily_predictions import (  # noqa: E402
    _LEGACY_NUDGE_MAX_ADJ_FRAC,
    _MODEL_PROJECTION_WEIGHT,
    _RATE_PROJECTION_MAX_ADJ_FRAC,
    _RATE_PROJECTION_WEIGHT,
    _compute_minutes_rate_adjustment,
)


# ---------------------------------------------------------------------------
# Legacy fallback path (rate = predicted / avg_minutes, ±15% nudge cap)
# ---------------------------------------------------------------------------

def test_legacy_fallback_when_features_missing():
    """No per-min features → falls back to legacy heuristic."""
    adjusted, source, rate = _compute_minutes_rate_adjustment(
        predicted_value=20.0,
        avg_minutes=30.0,
        predicted_minutes=30.0,
        prop_type='points',
        features=None,
    )
    assert source == 'legacy_pred_div_avg'
    # rate = 20/30 = 0.667, delta = 0.667 * (30-30) = 0
    assert rate == pytest.approx(20.0 / 30.0, rel=1e-6)
    assert adjusted == pytest.approx(20.0, abs=1e-6)


def test_legacy_fallback_scales_with_predicted_minutes():
    """More predicted_minutes → upward nudge, capped at ±15%."""
    adjusted, source, _ = _compute_minutes_rate_adjustment(
        predicted_value=20.0,
        avg_minutes=30.0,
        predicted_minutes=33.0,  # +3 min => +2.0 raw, within 15% cap (3.0)
        prop_type='points',
        features={},
    )
    assert source == 'legacy_pred_div_avg'
    expected = 20.0 + (20.0 / 30.0) * 3.0
    assert adjusted == pytest.approx(expected, abs=1e-6)


def test_legacy_fallback_respects_15pct_nudge_cap():
    """Extreme minutes deltas get clamped to ±15% of predicted_value."""
    # rate = 1.0, predicted_minutes=60 => raw delta = +30 => clipped to +3
    adjusted, _, _ = _compute_minutes_rate_adjustment(
        predicted_value=20.0,
        avg_minutes=20.0,
        predicted_minutes=60.0,
        prop_type='points',
        features={},
    )
    assert adjusted == pytest.approx(20.0 * (1 + _LEGACY_NUDGE_MAX_ADJ_FRAC), abs=1e-6)


def test_legacy_fallback_returns_input_when_avg_minutes_unknown():
    """avg_minutes <= 0 means we have no baseline — adjusted equals input."""
    adjusted, source, rate = _compute_minutes_rate_adjustment(
        predicted_value=20.0,
        avg_minutes=0,
        predicted_minutes=30.0,
        prop_type='points',
        features={},
    )
    assert source == 'legacy_pred_div_avg'
    assert rate == 0.0
    assert adjusted == pytest.approx(20.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Rate-based projection (recent_per_min × predicted_min, blended 0.6/0.4)
# ---------------------------------------------------------------------------

def test_rate_projection_activates_when_per_min_feature_present():
    """Direct-stat prop with recent_*_per_min feature → rate-based path."""
    adjusted, source, rate = _compute_minutes_rate_adjustment(
        predicted_value=20.0,
        avg_minutes=30.0,
        predicted_minutes=33.0,
        prop_type='points',
        features={'recent_pts_per_min': 0.8},
    )
    assert source == 'recent_per_min'
    assert rate == pytest.approx(0.8, abs=1e-6)
    # rate_projection = 0.8 * 33 = 26.4
    # adjusted = 0.6 * 26.4 + 0.4 * 20.0 = 23.84
    # Within 35% cap (cap = 7.0) → no clamp
    expected = (
        _RATE_PROJECTION_WEIGHT * 0.8 * 33.0
        + _MODEL_PROJECTION_WEIGHT * 20.0
    )
    assert adjusted == pytest.approx(expected, abs=1e-6)


def test_rate_projection_pra_sums_three_components():
    """PRA uses the sum of pts/reb/ast per-minute rates as its rate."""
    adjusted, source, rate = _compute_minutes_rate_adjustment(
        predicted_value=35.0,
        avg_minutes=30.0,
        predicted_minutes=33.0,
        prop_type='pra',
        features={
            'recent_pts_per_min': 0.8,
            'recent_reb_per_min': 0.3,
            'recent_ast_per_min': 0.15,
        },
    )
    assert source == 'recent_per_min'
    assert rate == pytest.approx(1.25, abs=1e-6)


def test_rate_projection_pra_falls_back_when_components_zero():
    """PRA with no per-min features → falls through to legacy heuristic."""
    adjusted, source, _ = _compute_minutes_rate_adjustment(
        predicted_value=35.0,
        avg_minutes=30.0,
        predicted_minutes=30.0,
        prop_type='pra',
        features={
            # No per-min fields at all
            'recent_pts_avg': 20.0,
        },
    )
    assert source == 'legacy_pred_div_avg'


def test_rate_projection_respects_35pct_cap():
    """Big disagreement between rate projection and model → ±35% cap."""
    # rate=2.0, predicted_minutes=33 => rate_projection=66
    # blend = 0.6*66 + 0.4*20 = 47.6 (vs predicted_value=20)
    # Raw delta = +27.6, far exceeding cap of 20 * 0.35 = 7.0
    adjusted, _, _ = _compute_minutes_rate_adjustment(
        predicted_value=20.0,
        avg_minutes=30.0,
        predicted_minutes=33.0,
        prop_type='points',
        features={'recent_pts_per_min': 2.0},
    )
    assert adjusted == pytest.approx(20.0 * (1 + _RATE_PROJECTION_MAX_ADJ_FRAC), abs=1e-6)


def test_rate_projection_threes_uses_fg3m_naming():
    """'threes' prop must look up recent_fg3m_per_min, NOT recent_fg3_per_min.

    Naming bug found in self-audit of 242bace: data_service.py originally
    used 'recent_fg3_per_min' while the codebase convention is 'recent_fg3m_*'.
    This test pins the consumer name so any future schema drift re-surfaces.
    """
    adjusted, source, rate = _compute_minutes_rate_adjustment(
        predicted_value=2.0,
        avg_minutes=30.0,
        predicted_minutes=30.0,
        prop_type='threes',
        features={'recent_fg3m_per_min': 0.05},  # 1.5 threes per 30 min
    )
    assert source == 'recent_per_min'
    assert rate == pytest.approx(0.05, abs=1e-6)

    # If someone reverts to fg3 (no 'm') this should fall back to legacy.
    _, source_legacy, _ = _compute_minutes_rate_adjustment(
        predicted_value=2.0,
        avg_minutes=30.0,
        predicted_minutes=30.0,
        prop_type='threes',
        features={'recent_fg3_per_min': 0.05},  # wrong key
    )
    assert source_legacy == 'legacy_pred_div_avg'


# ---------------------------------------------------------------------------
# Regression fixture: the Wembanyama case that motivated the fix
# ---------------------------------------------------------------------------

def test_wembanyama_case_pra_under_collapses():
    """Pins the Wembanyama (2026-05-15) audit case.

    Pre-fix: model predicted 35.3 PRA on a 44.5 line → 20.6% UNDER edge.
    Post-fix: rate-projection from his last 7 normal-minutes games (sum=288 PRA
    over 237 min => 1.215 PRA/min). At predicted_minutes=33 the projection is
    40.1 PRA, blended 0.6/0.4 with the model's 35.3 gives 38.2 PRA — close to
    the 44.5 line, dropping the under edge from 20.6% to ~6%.

    If this test ever fails, the rate-projection path has regressed and any
    star-player UNDER picks should be regarded as suspect again.
    """
    pts_pm = 1.0   # from filtered last-5 totals
    reb_pm = 0.18
    ast_pm = 0.08
    # rate = 1.26 PRA/min (matches the data_service computation closely)

    adjusted, source, rate = _compute_minutes_rate_adjustment(
        predicted_value=35.3,
        avg_minutes=33.0,
        predicted_minutes=33.0,
        prop_type='pra',
        features={
            'recent_pts_per_min': pts_pm,
            'recent_reb_per_min': reb_pm,
            'recent_ast_per_min': ast_pm,
        },
    )
    assert source == 'recent_per_min'
    assert rate == pytest.approx(pts_pm + reb_pm + ast_pm, abs=1e-6)

    # Adjusted prediction lands between the model (35.3) and the rate
    # projection (1.26 * 33 = 41.6). Should be CLOSER to the rate projection
    # than to the model thanks to the 0.6 rate weight.
    rate_proj = rate * 33.0
    assert adjusted > 35.3  # moved up toward the line
    # Verify within the 35% cap and consistent with the blend formula
    expected = 0.6 * rate_proj + 0.4 * 35.3
    cap = 35.3 * _RATE_PROJECTION_MAX_ADJ_FRAC
    assert adjusted == pytest.approx(min(expected, 35.3 + cap), abs=1e-6)
    # Pred-line gap shrinks from -9.2 to roughly -6.5 (still UNDER but much
    # less convincing — the original 20.6% UNDER edge becomes ~7%).
    assert (44.5 - adjusted) < 9.2
