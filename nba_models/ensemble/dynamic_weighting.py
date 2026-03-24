"""
Dynamic Ensemble Weighting (Phase 3.2)

Replaces the static np.mean() averaging of base-model predictions with:
  1. Performance-based weights — models with lower recent MAE get more weight.
  2. Diversity-adjusted weights — correlated model pairs have their combined
     weight reduced to avoid double-counting the same signal.
  3. Persistent state — performance history is written to JSON so weights
     improve across sessions without retraining.

Usage (inference)
-----------------
    from nba_models.ensemble.dynamic_weighting import DynamicEnsembleWeighter

    weighter = DynamicEnsembleWeighter.load(path)  # or DynamicEnsembleWeighter()

    # When you have base-model predictions for a single game/player:
    weights = weighter.get_weights(
        model_names=['xgboost', 'lightgbm', 'catboost', 'random_forest'],
        recent_predictions=None,   # pass dict of {name: pred} for diversity adj.
    )
    predicted = sum(weights[n] * preds[n] for n in preds)

    # After the actual result is known (e.g., next-day settlement):
    for name, pred in preds.items():
        weighter.record(name, prop_type='points', prediction=pred, actual=28.0)
    weighter.save(path)

Design notes
------------
* Window = last 100 settled predictions per (model, prop_type) pair.
* Weights are derived from inverse-MAE with Laplace smoothing so a model with
  zero history still gets a reasonable starting weight.
* Diversity penalty: for each pair (A, B) whose recent predictions correlate
  > 0.85, we multiply each one's weight by sqrt(1 - corr^2) to reduce the
  effective weight of the redundant signal.
* All weights are renormalized to sum to 1.0 before being returned.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Laplace smoothing: assume each unseen model starts with this pseudo-MAE
_DEFAULT_MAE = 3.0

# Diversity penalty threshold: correlation above this triggers weight reduction
_DIVERSITY_CORR_THRESHOLD = 0.85

# History window per (model, prop_type): keep this many settled predictions
_WINDOW = 100

# Exponential decay applied to older observations (per-prediction step)
# decay^1 = most recent, decay^N = oldest.  0.97 ≈ half-life of ~23 predictions.
_DECAY = 0.97


class DynamicEnsembleWeighter:
    """
    Tracks per-model recent performance and computes dynamic ensemble weights.

    Attributes
    ----------
    _history : dict
        {(model_name, prop_type): deque of (prediction, actual) tuples}
    """

    def __init__(self) -> None:
        # Key: (model_name, prop_type)  Value: deque[(prediction, actual)]
        self._history: Dict[tuple, deque] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record(
        self,
        model_name: str,
        prop_type: str,
        prediction: float,
        actual: float,
    ) -> None:
        """Record a settled prediction so performance history accumulates."""
        key = (model_name.lower(), prop_type.lower())
        if key not in self._history:
            self._history[key] = deque(maxlen=_WINDOW)
        self._history[key].append((float(prediction), float(actual)))

    def get_weights(
        self,
        model_names: Sequence[str],
        prop_type: str = 'points',
        recent_predictions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute normalised weights for the given set of models.

        Parameters
        ----------
        model_names
            Names of the base models in the ensemble.
        prop_type
            Stat category — weights are per (model, prop_type).
        recent_predictions
            Optional dict of {model_name: current_prediction_value}.  When
            provided, a diversity penalty is applied: highly correlated model
            pairs have their combined weight reduced.

        Returns
        -------
        dict
            {model_name: weight}  — values sum to 1.0.
        """
        if not model_names:
            return {}

        # Step 1: base weights from inverse-MAE
        raw_weights: Dict[str, float] = {}
        for name in model_names:
            mae = self._recent_mae(name, prop_type)
            raw_weights[name] = 1.0 / max(mae, 0.1)  # inverse-MAE

        # Step 2: diversity adjustment on current predictions
        if recent_predictions and len(recent_predictions) >= 2:
            raw_weights = self._apply_diversity_penalty(raw_weights, recent_predictions)

        # Step 3: normalise
        total = sum(raw_weights.values())
        if total <= 0:
            equal = 1.0 / len(model_names)
            return {n: equal for n in model_names}

        return {n: raw_weights[n] / total for n in model_names}

    def get_model_stats(
        self, model_name: str, prop_type: str
    ) -> Dict[str, float]:
        """Return accuracy metrics for a (model, prop_type) pair."""
        key = (model_name.lower(), prop_type.lower())
        history = list(self._history.get(key, []))
        if not history:
            return {'n': 0, 'mae': _DEFAULT_MAE, 'rmse': _DEFAULT_MAE, 'bias': 0.0}

        preds = np.array([p for p, _ in history])
        actuals = np.array([a for _, a in history])
        errors = preds - actuals
        return {
            'n': len(history),
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'bias': float(np.mean(errors)),
        }

    def log_all_stats(self, prop_type: str) -> None:
        """Log accuracy summary for all models on a given prop type."""
        model_names = {k[0] for k in self._history if k[1] == prop_type.lower()}
        if not model_names:
            logger.info("No performance history yet for prop_type=%s", prop_type)
            return
        logger.info("--- Ensemble model stats (prop=%s) ---", prop_type)
        for name in sorted(model_names):
            s = self.get_model_stats(name, prop_type)
            logger.info(
                "  %-20s  n=%3d  MAE=%.3f  RMSE=%.3f  bias=%+.3f",
                name, s['n'], s['mae'], s['rmse'], s['bias'],
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            f"{k[0]}::{k[1]}": list(v)
            for k, v in self._history.items()
        }
        with open(path, 'w') as f:
            json.dump(serialisable, f, indent=2)
        logger.debug("DynamicEnsembleWeighter saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> 'DynamicEnsembleWeighter':
        """Load a previously saved weighter, or return a fresh one if not found."""
        obj = cls()
        path = Path(path)
        if not path.exists():
            logger.debug("No weighter state found at %s — starting fresh", path)
            return obj
        try:
            with open(path) as f:
                data = json.load(f)
            for key_str, pairs in data.items():
                model_name, prop_type = key_str.split('::', 1)
                key = (model_name, prop_type)
                obj._history[key] = deque(
                    [(float(p), float(a)) for p, a in pairs],
                    maxlen=_WINDOW,
                )
            logger.info(
                "DynamicEnsembleWeighter loaded from %s (%d keys)",
                path, len(obj._history),
            )
        except Exception as exc:
            logger.warning("Failed to load weighter state: %s — starting fresh", exc)
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recent_mae(self, model_name: str, prop_type: str) -> float:
        """Return exponentially-decayed MAE for a model/prop pair."""
        key = (model_name.lower(), prop_type.lower())
        history = list(self._history.get(key, []))
        if not history:
            return _DEFAULT_MAE

        # Apply exponential decay: most recent observation has weight=1,
        # older observations have weight=decay^(distance from end).
        n = len(history)
        weights = np.array([_DECAY ** (n - 1 - i) for i in range(n)])
        errors = np.array([abs(p - a) for p, a in history])
        weighted_mae = float(np.dot(weights, errors) / weights.sum())
        return max(weighted_mae, 0.1)

    def _apply_diversity_penalty(
        self,
        weights: Dict[str, float],
        predictions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Reduce combined weight for model pairs with high prediction correlation.

        For each pair (A, B) where recent predicted values correlate strongly,
        both weights are multiplied by sqrt(1 - rho) to penalise redundancy.
        This prevents two nearly-identical models from dominating the ensemble.
        """
        names = [n for n in weights if n in predictions]
        if len(names) < 2:
            return weights

        adjusted = dict(weights)

        # Build list of recent predictions per model for correlation estimation
        recent_preds: Dict[str, List[float]] = {}
        for name in names:
            # Use history if available; fall back to just the current prediction
            history_data = []
            for prop_type in {k[1] for k in self._history if k[0] == name.lower()}:
                key = (name.lower(), prop_type)
                history_data.extend([p for p, _ in self._history.get(key, [])])
            if len(history_data) >= 5:
                recent_preds[name] = history_data[-20:]  # last 20 settled
            else:
                recent_preds[name] = [predictions[name]]

        # Pairwise correlation check
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                corr = self._pairwise_correlation(
                    recent_preds.get(a, [predictions[a]]),
                    recent_preds.get(b, [predictions[b]]),
                )
                if corr is not None and corr > _DIVERSITY_CORR_THRESHOLD:
                    penalty = math.sqrt(1.0 - corr ** 2)
                    logger.debug(
                        "Diversity penalty %.3f applied to (%s, %s) corr=%.3f",
                        penalty, a, b, corr,
                    )
                    adjusted[a] *= penalty
                    adjusted[b] *= penalty

        return adjusted

    @staticmethod
    def _pairwise_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
        """Pearson correlation; returns None if insufficient data."""
        n = min(len(xs), len(ys))
        if n < 5:
            return None
        xs_a = np.array(xs[-n:], dtype=float)
        ys_a = np.array(ys[-n:], dtype=float)
        if xs_a.std() < 1e-9 or ys_a.std() < 1e-9:
            return None
        return float(np.corrcoef(xs_a, ys_a)[0, 1])
