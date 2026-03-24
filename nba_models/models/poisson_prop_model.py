"""
Poisson Regression Model for Count-Based Player Props (Phase 3.3)

Three-pointers made (3PM) are discrete counts (0, 1, 2, 3, …), not continuous
values.  Standard linear/gradient-boosted regressors assume Gaussian residuals
and systematically under-estimate variance for count data.

PoissonPropModel wraps a GLM with a log link (i.e., a Poisson regression) to
produce a rate parameter λ (expected makes) from player and opponent features.
The Poisson CDF is then used to compute P(X > line) exactly:

    P(3PM > n.5) = 1 - CDF_Poisson(floor(n.5), λ)
                 = 1 - sum_{k=0}^{floor(n.5)} e^{-λ} * λ^k / k!

This avoids the Gaussian approximation used elsewhere in the pipeline and is
particularly accurate for low-count stats like threes (mean ≈ 2–3 makes/game).

Regression-to-Mean Detection
-----------------------------
Hot/cold streaks are identified and fade is applied:
  - HOT: recent 3P% >= 50% over last 3 games → predict regression (slight fade)
  - COLD: recent 3P% <= 20% over last 3 games → predict mean-reversion (slight boost)
  - NEITHER: use λ unchanged

The fade magnitude is calibrated conservatively (±5-10% of λ) because streaks
are partially real skill and partially noise.

Training Usage
--------------
    from nba_models.models.poisson_prop_model import PoissonPropModel

    model = PoissonPropModel(features=THREES_FEATURES, min_samples=15)
    model.fit(X_train, y_train)

    # Predict λ
    lam = model.predict(X_test)

    # Predict P(>= threshold)
    prob_over = model.predict_over_probability(X_test, line=2.5)

Inference Usage
---------------
    pred = model.predict_single(features_dict, line=2.5)
    # Returns {'lambda': float, 'over_prob': float, 'streak_type': str, ...}

Pickling
--------
The model can be pickled directly; it stores (estimator, feature_names,
scaler, min_samples).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default minimum 3PA attempts per game required to generate a prediction.
# Below this threshold the player's shooting sample is too small; we decline
# to generate a threes prediction to avoid noise bets.
DEFAULT_MIN_FG3A = 2.0  # average 3-point attempts per game

# Default minimum games required for a reliable season average.
DEFAULT_MIN_GAMES = 15

# Fade magnitude for hot/cold streak regression-to-mean
_HOT_STREAK_FADE = 0.08    # 8% downward adjustment to λ when player is hot
_COLD_STREAK_BOOST = 0.06  # 6% upward adjustment to λ when player is cold

# 3P% thresholds for streak classification
_HOT_PCT_THRESHOLD = 0.50   # >= 50% over last 3 games = hot streak
_COLD_PCT_THRESHOLD = 0.20  # <= 20% over last 3 games = cold streak


class PoissonPropModel:
    """
    Poisson GLM wrapper for count-based prop predictions (3PM, etc.).

    Parameters
    ----------
    features : list[str]
        Names of features used during training.
    min_fg3a : float
        Minimum average 3-point attempts per game required to generate a
        prediction.  Players below this threshold are skipped.
    min_games : int
        Minimum games played for a reliable season average.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        min_fg3a: float = DEFAULT_MIN_FG3A,
        min_games: int = DEFAULT_MIN_GAMES,
    ) -> None:
        self.features: List[str] = features or []
        self.min_fg3a = min_fg3a
        self.min_games = min_games

        self._estimator: Any = None   # sklearn Poisson GLM or compatible
        self._scaler: Any = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: 'np.ndarray', y: 'np.ndarray') -> 'PoissonPropModel':
        """
        Fit a Poisson GLM (log link) on the training data.

        Uses sklearn's TweedieRegressor with power=1 (Poisson) and log link.
        Falls back to GradientBoostingRegressor with Poisson loss if sklearn
        version < 0.23 (where TweedieRegressor is unavailable).
        """
        try:
            from sklearn.linear_model import TweedieRegressor
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            # Clip targets to >= 0 (Poisson requires non-negative counts)
            y_clipped = np.clip(y, 0, None)

            self._estimator = TweedieRegressor(
                power=1,           # Poisson distribution
                link='log',        # log link: λ = exp(Xβ)
                alpha=0.1,         # L2 regularisation
                max_iter=500,
            )
            self._estimator.fit(X_scaled, y_clipped)
            self._is_fitted = True
            logger.info(
                "PoissonPropModel fitted on %d samples (%d features)",
                len(y), X.shape[1] if len(X.shape) > 1 else 1,
            )
        except ImportError:
            # Fallback: use GBR with Poisson loss (sklearn >= 0.21)
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            y_clipped = np.clip(y, 0, None)

            self._estimator = GradientBoostingRegressor(
                loss='poisson',
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
            )
            self._estimator.fit(X_scaled, y_clipped)
            self._is_fitted = True
            logger.info("PoissonPropModel fitted (GBR Poisson fallback)")

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: 'np.ndarray') -> 'np.ndarray':
        """Predict λ (expected makes) for each row."""
        self._check_fitted()
        if self._scaler is not None:
            X = self._scaler.transform(X)
        lam = self._estimator.predict(X)
        return np.clip(lam, 0.01, None)  # λ must be positive

    def predict_over_probability(
        self, X: 'np.ndarray', line: float
    ) -> 'np.ndarray':
        """
        Predict P(X > line) using the Poisson CDF exactly.

        For a half-integer line (e.g., 2.5):
            P(X > 2.5) = P(X >= 3) = 1 - P(X <= 2) = 1 - CDF(2, λ)
        """
        lambdas = self.predict(X)
        probs = np.array([
            _poisson_over_prob(lam, line) for lam in lambdas
        ])
        return probs

    def predict_single(
        self,
        features_dict: Dict[str, float],
        line: float,
        streak_context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Predict from a raw feature dict.

        Parameters
        ----------
        features_dict
            Feature dict with the same keys as self.features.
        line
            Sportsbook line (e.g., 2.5).
        streak_context
            Optional dict with keys: 'fg3m_last3', 'fg3a_last3',
            'fg3a_avg', 'season_games'.  Used for regression-to-mean
            detection.

        Returns
        -------
        dict with keys:
            lambda : float         — predicted mean (λ)
            over_prob : float      — P(X > line) using Poisson CDF
            streak_type : str      — 'hot' | 'cold' | 'neutral'
            streak_fade : float    — adjustment applied to λ (+/-)
            eligible : bool        — False if sample size too small
            ineligible_reason : str
        """
        if not self._is_fitted:
            return self._not_fitted_result(line)

        fg3a_avg = features_dict.get('fg3a_avg', 0.0)
        season_games = features_dict.get('season_games', 0)

        if fg3a_avg < self.min_fg3a:
            return {
                'lambda': None,
                'over_prob': None,
                'streak_type': 'neutral',
                'streak_fade': 0.0,
                'eligible': False,
                'ineligible_reason': (
                    f'fg3a_avg {fg3a_avg:.1f} < min {self.min_fg3a:.1f}'
                ),
            }

        if season_games < self.min_games:
            return {
                'lambda': None,
                'over_prob': None,
                'streak_type': 'neutral',
                'streak_fade': 0.0,
                'eligible': False,
                'ineligible_reason': (
                    f'season_games {season_games} < min {self.min_games}'
                ),
            }

        import pandas as pd
        X = pd.DataFrame([{k: features_dict.get(k, np.nan) for k in self.features}])
        X = X.fillna(X.mean())
        if self._scaler is not None:
            X_arr = self._scaler.transform(X[self.features])
        else:
            X_arr = X[self.features].values
        lam = float(self._estimator.predict(X_arr)[0])
        lam = max(lam, 0.01)

        # Regression-to-mean detection
        streak_type, streak_fade = _detect_streak(streak_context or features_dict)
        adjusted_lam = lam * (1.0 + streak_fade)
        adjusted_lam = max(adjusted_lam, 0.01)

        over_prob = _poisson_over_prob(adjusted_lam, line)

        return {
            'lambda': round(adjusted_lam, 4),
            'over_prob': round(over_prob, 4),
            'streak_type': streak_type,
            'streak_fade': round(streak_fade, 4),
            'eligible': True,
            'ineligible_reason': '',
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("PoissonPropModel is not fitted yet.")

    @staticmethod
    def _not_fitted_result(line: float) -> Dict[str, Any]:
        return {
            'lambda': None,
            'over_prob': None,
            'streak_type': 'neutral',
            'streak_fade': 0.0,
            'eligible': False,
            'ineligible_reason': 'model not fitted',
        }


# ---------------------------------------------------------------------------
# Module-level helpers (used by inference path even without full model)
# ---------------------------------------------------------------------------

def compute_poisson_over_prob(lam: float, line: float) -> float:
    """
    Compute P(X > line) for X ~ Poisson(λ).

    Public entry point for the inference path when we have λ from any source.
    """
    return _poisson_over_prob(lam, line)


def detect_threes_streak(features_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Detect hot/cold streak in a player's recent three-point shooting.

    Exposed at module level so the inference pipeline can call it without
    instantiating a full PoissonPropModel.

    Parameters
    ----------
    features_dict
        Must contain: fg3m_last3 (makes), fg3a_last3 (attempts),
        fg3a_avg (season average), season_games.  If absent, defaults are used.

    Returns
    -------
    dict
        {'streak_type': str, 'streak_fade': float, 'details': str}
    """
    streak_type, fade = _detect_streak(features_dict)
    pct_last3 = _last3_pct(features_dict)
    details = (
        f"last3_pct={pct_last3:.1%} → {streak_type} (fade={fade:+.1%})"
        if pct_last3 is not None
        else "insufficient 3PA data"
    )
    return {'streak_type': streak_type, 'streak_fade': fade, 'details': details}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _poisson_over_prob(lam: float, line: float) -> float:
    """P(X > line) for X ~ Poisson(λ), using scipy if available."""
    lam = max(float(lam), 1e-9)
    k_floor = int(math.floor(line))  # largest integer <= line

    try:
        from scipy.stats import poisson as _poisson
        # P(X > line) = P(X >= k_floor + 1) = 1 - P(X <= k_floor)
        return float(1.0 - _poisson.cdf(k_floor, lam))
    except ImportError:
        # Manual CDF via log-factorial
        log_lam = math.log(lam)
        log_sum = -math.inf
        log_term = -lam  # k=0 term: e^{-λ} * λ^0 / 0! = e^{-λ}
        for k in range(k_floor + 1):
            if k > 0:
                log_term += log_lam - math.log(k)
            log_sum = _log_add_exp(log_sum, log_term)
        cdf = min(1.0, math.exp(log_sum))
        return float(1.0 - cdf)


def _log_add_exp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _last3_pct(features_dict: Dict[str, float]) -> Optional[float]:
    """Compute last-3 game 3P% from feature dict, or None if data absent."""
    fg3m = features_dict.get('fg3m_last3') or features_dict.get('last3_fg3m_avg')
    fg3a = features_dict.get('fg3a_last3') or features_dict.get('last3_fg3a_avg')
    if fg3a and fg3a > 0:
        return float(fg3m or 0.0) / float(fg3a)
    return None


def _detect_streak(features_dict: Dict[str, float]) -> tuple:
    """
    Returns (streak_type, fade_multiplier) where fade_multiplier is the
    fractional change to apply to λ.  Positive = boost, Negative = fade.
    """
    pct = _last3_pct(features_dict)
    if pct is None:
        return ('neutral', 0.0)

    fg3a_avg = features_dict.get('fg3a_avg', 0.0)

    # Only apply streak logic to players who regularly shoot threes
    if fg3a_avg < DEFAULT_MIN_FG3A:
        return ('neutral', 0.0)

    if pct >= _HOT_PCT_THRESHOLD:
        # Hot streak: regress toward mean → predict slightly fewer makes
        return ('hot', -_HOT_STREAK_FADE)
    elif pct <= _COLD_PCT_THRESHOLD:
        # Cold streak: mean reversion → predict slightly more makes
        return ('cold', +_COLD_STREAK_BOOST)
    else:
        return ('neutral', 0.0)
