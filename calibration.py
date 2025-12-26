"""
Probability Calibration Module

Converts raw model outputs into well-calibrated probabilities.
A well-calibrated model means: when we predict 70% probability,
the event should occur approximately 70% of the time.

Methods implemented:
1. Platt Scaling (sigmoid calibration) - parametric
2. Isotonic Regression - non-parametric
3. Temperature Scaling - simple single-parameter method
4. Beta Calibration - flexible 3-parameter method

Calibration is CRITICAL for betting because:
- Raw model probabilities are often overconfident or underconfident
- Betting edge calculations rely on accurate probabilities
- Kelly criterion bet sizing requires well-calibrated probabilities
"""

import numpy as np
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from scipy.special import expit, logit
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float  # Brier score (lower is better)
    log_loss: float  # Log loss / cross-entropy
    reliability_diagram: Dict[str, List[float]]  # For plotting

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Calibration Metrics:\n"
            f"  ECE (Expected Calibration Error): {self.ece:.4f}\n"
            f"  MCE (Maximum Calibration Error): {self.mce:.4f}\n"
            f"  Brier Score: {self.brier_score:.4f}\n"
            f"  Log Loss: {self.log_loss:.4f}"
        )


class BaseCalibrator(ABC):
    """Abstract base class for probability calibrators."""

    @abstractmethod
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "BaseCalibrator":
        """
        Fit the calibrator to validation data.

        Args:
            y_prob: Uncalibrated probability predictions (0 to 1)
            y_true: True binary labels (0 or 1)

        Returns:
            self
        """
        pass

    @abstractmethod
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibrate probability predictions.

        Args:
            y_prob: Uncalibrated probability predictions

        Returns:
            Calibrated probabilities
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save calibrator to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> "BaseCalibrator":
        """Load calibrator from disk."""
        pass


class PlattScaling(BaseCalibrator):
    """
    Platt Scaling (Sigmoid Calibration)

    Fits a logistic regression to transform raw scores into probabilities.
    Works well when the uncalibrated probabilities are roughly sigmoid-shaped.

    P(y=1|f) = 1 / (1 + exp(A*f + B))

    Where f is the uncalibrated score and A, B are fitted parameters.
    """

    def __init__(self):
        self.A: float = 0.0
        self.B: float = 0.0
        self._fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "PlattScaling":
        """Fit Platt scaling parameters."""
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        # Convert probabilities to logits for fitting
        # Clip to avoid inf values
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
        logits = logit(y_prob_clipped)

        # Fit logistic regression
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(logits.reshape(-1, 1), y_true)

        self.A = float(lr.coef_[0][0])
        self.B = float(lr.intercept_[0])
        self._fitted = True

        logger.info(f"Platt Scaling fitted: A={self.A:.4f}, B={self.B:.4f}")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply Platt scaling calibration."""
        if not self._fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob)
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
        logits = logit(y_prob_clipped)

        calibrated = expit(self.A * logits + self.B)
        return np.clip(calibrated, 0.0, 1.0)

    def save(self, path: str) -> None:
        """Save to disk."""
        data = {"A": self.A, "B": self.B, "fitted": self._fitted}
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str) -> "PlattScaling":
        """Load from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.A = data["A"]
        self.B = data["B"]
        self._fitted = data["fitted"]
        return self


class IsotonicCalibration(BaseCalibrator):
    """
    Isotonic Regression Calibration

    Non-parametric method that fits a monotonically increasing function.
    More flexible than Platt scaling but can overfit with small datasets.

    Best when:
    - You have enough calibration data (>1000 samples)
    - The calibration function is not sigmoid-shaped
    """

    def __init__(self, out_of_bounds: str = 'clip'):
        """
        Args:
            out_of_bounds: How to handle predictions outside training range
                          'clip' (default) or 'nan'
        """
        self._isotonic = IsotonicRegression(
            out_of_bounds=out_of_bounds,
            y_min=0.0,
            y_max=1.0
        )
        self._fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibration":
        """Fit isotonic regression."""
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        self._isotonic.fit(y_prob, y_true)
        self._fitted = True

        logger.info("Isotonic calibration fitted")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        if not self._fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob)
        original_shape = y_prob.shape
        calibrated = self._isotonic.predict(y_prob.flatten())
        return np.clip(calibrated.reshape(original_shape), 0.0, 1.0)

    def save(self, path: str) -> None:
        """Save to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self._isotonic, f)

    def load(self, path: str) -> "IsotonicCalibration":
        """Load from disk."""
        with open(path, 'rb') as f:
            self._isotonic = pickle.load(f)
        self._fitted = True
        return self


class TemperatureScaling(BaseCalibrator):
    """
    Temperature Scaling

    Simple single-parameter calibration that divides logits by temperature T.
    Preserves accuracy while improving calibration.

    P(y=1|f) = sigmoid(logit(f) / T)

    Where T > 1 makes predictions less confident, T < 1 more confident.

    Advantages:
    - Very simple (single parameter)
    - Doesn't change model ranking (preserves accuracy)
    - Works well for neural networks
    """

    def __init__(self):
        self.temperature: float = 1.0
        self._fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        """Fit temperature parameter using NLL minimization."""
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
        logits = logit(y_prob_clipped)

        def nll_loss(T):
            """Negative log likelihood loss."""
            T = max(T[0], 0.01)  # Ensure positive temperature
            scaled_probs = expit(logits / T)
            scaled_probs = np.clip(scaled_probs, 1e-10, 1 - 1e-10)
            return -np.mean(
                y_true * np.log(scaled_probs) +
                (1 - y_true) * np.log(1 - scaled_probs)
            )

        # Optimize temperature
        result = minimize(nll_loss, [1.0], method='Nelder-Mead')
        self.temperature = max(float(result.x[0]), 0.01)
        self._fitted = True

        logger.info(f"Temperature Scaling fitted: T={self.temperature:.4f}")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        if not self._fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob)
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
        logits = logit(y_prob_clipped)

        calibrated = expit(logits / self.temperature)
        return np.clip(calibrated, 0.0, 1.0)

    def save(self, path: str) -> None:
        """Save to disk."""
        data = {"temperature": self.temperature, "fitted": self._fitted}
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str) -> "TemperatureScaling":
        """Load from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.temperature = data["temperature"]
        self._fitted = data["fitted"]
        return self


class BetaCalibration(BaseCalibrator):
    """
    Beta Calibration

    Three-parameter calibration using beta distribution family.
    More flexible than Platt scaling while avoiding overfitting of isotonic.

    P(y=1|f) = 1 / (1 + 1/exp(a * log(f/(1-f)) + b * log(f) + c))

    Advantages:
    - Handles asymmetric calibration curves
    - Good for betting where tails matter
    """

    def __init__(self):
        self.a: float = 1.0
        self.b: float = 0.0
        self.c: float = 0.0
        self._fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "BetaCalibration":
        """Fit beta calibration parameters."""
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)

        def nll_loss(params):
            """Negative log likelihood loss."""
            a, b, c = params
            # Beta calibration transformation
            log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped))
            log_prob = np.log(y_prob_clipped)

            cal_logits = a * log_odds + b * log_prob + c
            cal_probs = expit(cal_logits)
            cal_probs = np.clip(cal_probs, 1e-10, 1 - 1e-10)

            return -np.mean(
                y_true * np.log(cal_probs) +
                (1 - y_true) * np.log(1 - cal_probs)
            )

        # Optimize parameters
        result = minimize(nll_loss, [1.0, 0.0, 0.0], method='Nelder-Mead')
        self.a, self.b, self.c = result.x
        self._fitted = True

        logger.info(f"Beta Calibration fitted: a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply beta calibration."""
        if not self._fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob)
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)

        log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped))
        log_prob = np.log(y_prob_clipped)

        cal_logits = self.a * log_odds + self.b * log_prob + self.c
        calibrated = expit(cal_logits)

        return np.clip(calibrated, 0.0, 1.0)

    def save(self, path: str) -> None:
        """Save to disk."""
        data = {"a": self.a, "b": self.b, "c": self.c, "fitted": self._fitted}
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str) -> "BetaCalibration":
        """Load from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.a = data["a"]
        self.b = data["b"]
        self.c = data["c"]
        self._fitted = data["fitted"]
        return self


def apply_probability_shrinkage(raw_prob: float, shrinkage: float = 0.15) -> float:
    """
    Shrink probabilities toward 0.5 to reduce overconfidence.

    Research shows ML models are typically overconfident in their predictions.
    This simple adjustment improves calibration and betting performance by
    reducing extreme probability estimates.

    Formula: adjusted = 0.5 + (raw - 0.5) * (1 - shrinkage)

    Args:
        raw_prob: Raw probability prediction (0-1)
        shrinkage: Shrinkage factor (0-1). Default 0.15 based on sports betting research.
                   - 0.10 = light shrinkage (mildly overconfident models)
                   - 0.15 = moderate shrinkage (typical ML models)
                   - 0.20 = aggressive shrinkage (highly overconfident models)

    Returns:
        Adjusted probability closer to 0.5

    Example:
        >>> apply_probability_shrinkage(0.70, 0.15)
        0.67  # Pulled toward 0.5
        >>> apply_probability_shrinkage(0.30, 0.15)
        0.33  # Pulled toward 0.5
    """
    if not 0 <= shrinkage <= 1:
        raise ValueError(f"Shrinkage must be between 0 and 1, got {shrinkage}")

    raw_prob = np.clip(raw_prob, 0.0, 1.0)
    adjusted = 0.5 + (raw_prob - 0.5) * (1 - shrinkage)
    return float(np.clip(adjusted, 0.0, 1.0))


def apply_probability_shrinkage_array(raw_probs: np.ndarray, shrinkage: float = 0.15) -> np.ndarray:
    """
    Apply probability shrinkage to an array of probabilities.

    Args:
        raw_probs: Array of raw probability predictions (0-1)
        shrinkage: Shrinkage factor (0-1). Default 0.15.

    Returns:
        Array of adjusted probabilities
    """
    raw_probs = np.clip(np.asarray(raw_probs), 0.0, 1.0)
    adjusted = 0.5 + (raw_probs - 0.5) * (1 - shrinkage)
    return np.clip(adjusted, 0.0, 1.0)


class FavoriteLongshotCalibrator(BaseCalibrator):
    """
    Split Calibrator for Favorites vs Underdogs

    Sports betting research shows the "favorite-longshot bias" - favorites and underdogs
    have systematically different calibration patterns:
    - Favorites (prob > 0.5) tend to be underconfident in raw ML models
    - Underdogs (prob < 0.5) tend to be overconfident

    This calibrator fits separate calibration curves for each segment,
    which significantly improves betting edge calculation accuracy.

    Reference: "The Favorite-Longshot Bias in Parimutuel Betting" (Hausch & Ziemba)
    """

    def __init__(self, base_method: str = "beta"):
        """
        Args:
            base_method: Base calibration method to use for each segment
                        Options: 'platt', 'isotonic', 'temperature', 'beta'
        """
        self.base_method = base_method
        self.favorite_calibrator: BaseCalibrator = None
        self.underdog_calibrator: BaseCalibrator = None
        self._fitted = False
        self._fav_count = 0
        self._dog_count = 0

    def _create_calibrator(self) -> BaseCalibrator:
        """Create a new instance of the base calibrator."""
        if self.base_method == "platt":
            return PlattScaling()
        elif self.base_method == "isotonic":
            return IsotonicCalibration()
        elif self.base_method == "temperature":
            return TemperatureScaling()
        elif self.base_method == "beta":
            return BetaCalibration()
        else:
            raise ValueError(f"Unknown base method: {self.base_method}")

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray,
            is_favorite: np.ndarray = None) -> "FavoriteLongshotCalibrator":
        """
        Fit separate calibrators for favorites and underdogs.

        Args:
            y_prob: Uncalibrated probability predictions (0-1)
            y_true: True binary labels (0 or 1)
            is_favorite: Optional boolean array indicating favorites.
                        If None, determined by y_prob > 0.5

        Returns:
            self
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        if is_favorite is None:
            is_favorite = y_prob > 0.5
        else:
            is_favorite = np.asarray(is_favorite).flatten().astype(bool)

        # Split data
        fav_mask = is_favorite
        dog_mask = ~is_favorite

        self._fav_count = np.sum(fav_mask)
        self._dog_count = np.sum(dog_mask)

        # Fit favorite calibrator
        if self._fav_count >= 20:
            self.favorite_calibrator = self._create_calibrator()
            self.favorite_calibrator.fit(y_prob[fav_mask], y_true[fav_mask])
            logger.info(f"Favorite calibrator fitted on {self._fav_count} samples")
        else:
            logger.warning(f"Only {self._fav_count} favorite samples, using identity calibration")
            self.favorite_calibrator = None

        # Fit underdog calibrator
        if self._dog_count >= 20:
            self.underdog_calibrator = self._create_calibrator()
            self.underdog_calibrator.fit(y_prob[dog_mask], y_true[dog_mask])
            logger.info(f"Underdog calibrator fitted on {self._dog_count} samples")
        else:
            logger.warning(f"Only {self._dog_count} underdog samples, using identity calibration")
            self.underdog_calibrator = None

        self._fitted = True
        return self

    def calibrate(self, y_prob: np.ndarray, is_favorite: np.ndarray = None) -> np.ndarray:
        """
        Apply split calibration.

        Args:
            y_prob: Uncalibrated probability predictions
            is_favorite: Optional boolean array indicating favorites.
                        If None, determined by y_prob > 0.5

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob)
        original_shape = y_prob.shape
        y_prob = y_prob.flatten()

        if is_favorite is None:
            is_favorite = y_prob > 0.5
        else:
            is_favorite = np.asarray(is_favorite).flatten().astype(bool)

        calibrated = np.zeros_like(y_prob, dtype=float)

        fav_mask = is_favorite
        dog_mask = ~is_favorite

        # Calibrate favorites
        if np.any(fav_mask):
            if self.favorite_calibrator is not None:
                calibrated[fav_mask] = self.favorite_calibrator.calibrate(y_prob[fav_mask])
            else:
                calibrated[fav_mask] = y_prob[fav_mask]

        # Calibrate underdogs
        if np.any(dog_mask):
            if self.underdog_calibrator is not None:
                calibrated[dog_mask] = self.underdog_calibrator.calibrate(y_prob[dog_mask])
            else:
                calibrated[dog_mask] = y_prob[dog_mask]

        return np.clip(calibrated.reshape(original_shape), 0.0, 1.0)

    def save(self, path: str) -> None:
        """Save to disk."""
        data = {
            "base_method": self.base_method,
            "fitted": self._fitted,
            "fav_count": self._fav_count,
            "dog_count": self._dog_count
        }

        base_path = Path(path).with_suffix('')

        # Save metadata
        with open(f"{base_path}_metadata.json", 'w') as f:
            json.dump(data, f)

        # Save sub-calibrators
        if self.favorite_calibrator is not None:
            ext = "pkl" if self.base_method == "isotonic" else "json"
            self.favorite_calibrator.save(f"{base_path}_favorite.{ext}")

        if self.underdog_calibrator is not None:
            ext = "pkl" if self.base_method == "isotonic" else "json"
            self.underdog_calibrator.save(f"{base_path}_underdog.{ext}")

    def load(self, path: str) -> "FavoriteLongshotCalibrator":
        """Load from disk."""
        base_path = Path(path).with_suffix('')

        # Load metadata
        with open(f"{base_path}_metadata.json", 'r') as f:
            data = json.load(f)

        self.base_method = data["base_method"]
        self._fitted = data["fitted"]
        self._fav_count = data["fav_count"]
        self._dog_count = data["dog_count"]

        ext = "pkl" if self.base_method == "isotonic" else "json"

        # Load sub-calibrators
        fav_path = Path(f"{base_path}_favorite.{ext}")
        if fav_path.exists():
            self.favorite_calibrator = self._create_calibrator()
            self.favorite_calibrator.load(str(fav_path))

        dog_path = Path(f"{base_path}_underdog.{ext}")
        if dog_path.exists():
            self.underdog_calibrator = self._create_calibrator()
            self.underdog_calibrator.load(str(dog_path))

        return self


class ShrinkagePlusCalibrator(BaseCalibrator):
    """
    Combined Shrinkage + Calibration approach.

    Applies probability shrinkage BEFORE standard calibration.
    This two-step approach is often more effective than either alone:

    1. Shrinkage: Reduces extreme predictions toward 0.5
    2. Calibration: Fine-tunes the probability mapping

    Particularly effective for overconfident models (common in sports betting).
    """

    def __init__(self, shrinkage: float = 0.15, base_method: str = "beta"):
        """
        Args:
            shrinkage: Shrinkage factor applied before calibration
            base_method: Base calibration method ('platt', 'isotonic', 'temperature', 'beta')
        """
        self.shrinkage = shrinkage
        self.base_method = base_method
        self.calibrator: BaseCalibrator = None
        self._fitted = False

    def _create_calibrator(self) -> BaseCalibrator:
        """Create a new instance of the base calibrator."""
        if self.base_method == "platt":
            return PlattScaling()
        elif self.base_method == "isotonic":
            return IsotonicCalibration()
        elif self.base_method == "temperature":
            return TemperatureScaling()
        elif self.base_method == "beta":
            return BetaCalibration()
        else:
            raise ValueError(f"Unknown base method: {self.base_method}")

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "ShrinkagePlusCalibrator":
        """
        Fit calibrator on shrunk probabilities.

        Args:
            y_prob: Uncalibrated probability predictions (0-1)
            y_true: True binary labels (0 or 1)

        Returns:
            self
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        # Apply shrinkage first
        y_prob_shrunk = apply_probability_shrinkage_array(y_prob, self.shrinkage)

        # Fit base calibrator on shrunk probabilities
        self.calibrator = self._create_calibrator()
        self.calibrator.fit(y_prob_shrunk, y_true)

        self._fitted = True
        logger.info(f"ShrinkagePlus calibrator fitted (shrinkage={self.shrinkage}, method={self.base_method})")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply shrinkage then calibration.

        Args:
            y_prob: Uncalibrated probability predictions

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob)

        # Apply shrinkage first
        y_prob_shrunk = apply_probability_shrinkage_array(y_prob, self.shrinkage)

        # Then apply calibration
        calibrated = self.calibrator.calibrate(y_prob_shrunk)

        return np.clip(calibrated, 0.0, 1.0)

    def save(self, path: str) -> None:
        """Save to disk."""
        base_path = Path(path).with_suffix('')

        data = {
            "shrinkage": self.shrinkage,
            "base_method": self.base_method,
            "fitted": self._fitted
        }

        with open(f"{base_path}_metadata.json", 'w') as f:
            json.dump(data, f)

        if self.calibrator is not None:
            ext = "pkl" if self.base_method == "isotonic" else "json"
            self.calibrator.save(f"{base_path}_base.{ext}")

    def load(self, path: str) -> "ShrinkagePlusCalibrator":
        """Load from disk."""
        base_path = Path(path).with_suffix('')

        with open(f"{base_path}_metadata.json", 'r') as f:
            data = json.load(f)

        self.shrinkage = data["shrinkage"]
        self.base_method = data["base_method"]
        self._fitted = data["fitted"]

        ext = "pkl" if self.base_method == "isotonic" else "json"
        cal_path = Path(f"{base_path}_base.{ext}")

        if cal_path.exists():
            self.calibrator = self._create_calibrator()
            self.calibrator.load(str(cal_path))

        return self


class CalibrationMonitor:
    """
    Monitor calibration drift over time.

    Tracks predictions and outcomes to detect when model calibration
    has degraded, triggering recalibration alerts.

    Use this in production to ensure model stays well-calibrated.
    """

    def __init__(self, ece_threshold: float = 0.08, window_size: int = 200):
        """
        Args:
            ece_threshold: ECE threshold to trigger recalibration alert
            window_size: Number of recent predictions to evaluate
        """
        self.ece_threshold = ece_threshold
        self.window_size = window_size
        self.prediction_history: List[float] = []
        self.outcome_history: List[int] = []
        self.alert_triggered = False
        self.last_check_ece: float = 0.0

    def add_prediction(self, prob: float, actual: int) -> None:
        """
        Record a prediction and its outcome.

        Args:
            prob: Predicted probability
            actual: Actual outcome (0 or 1)
        """
        self.prediction_history.append(float(prob))
        self.outcome_history.append(int(actual))

        # Check calibration periodically
        if len(self.prediction_history) % 50 == 0:
            self._check_calibration()

    def _check_calibration(self) -> None:
        """Check if calibration has drifted."""
        if len(self.prediction_history) < self.window_size:
            return

        recent_probs = np.array(self.prediction_history[-self.window_size:])
        recent_outcomes = np.array(self.outcome_history[-self.window_size:])

        self.last_check_ece = CalibrationEvaluator.expected_calibration_error(
            recent_probs, recent_outcomes
        )

        if self.last_check_ece > self.ece_threshold:
            self.alert_triggered = True
            logger.warning(
                f"Calibration drift detected! ECE={self.last_check_ece:.4f} "
                f"exceeds threshold {self.ece_threshold}"
            )

    def should_recalibrate(self) -> bool:
        """Check if recalibration is recommended."""
        return self.alert_triggered

    def reset_alert(self) -> None:
        """Reset alert after recalibration."""
        self.alert_triggered = False

    def get_diagnostics(self) -> Dict:
        """Get calibration monitoring diagnostics."""
        return {
            "total_predictions": len(self.prediction_history),
            "window_size": self.window_size,
            "last_ece": self.last_check_ece,
            "ece_threshold": self.ece_threshold,
            "alert_triggered": self.alert_triggered,
            "recent_win_rate": np.mean(self.outcome_history[-self.window_size:]) if len(self.outcome_history) >= self.window_size else None,
            "recent_avg_prob": np.mean(self.prediction_history[-self.window_size:]) if len(self.prediction_history) >= self.window_size else None
        }

    def clear_history(self) -> None:
        """Clear prediction history."""
        self.prediction_history = []
        self.outcome_history = []
        self.alert_triggered = False


class CalibrationEvaluator:

    @staticmethod
    def expected_calibration_error(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE = sum(|bin_accuracy - bin_confidence| * bin_weight)

        Lower is better. 0 = perfectly calibrated.
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(y_true[mask])
                bin_confidence = np.mean(y_prob[mask])
                bin_weight = np.sum(mask) / len(y_prob)
                ece += np.abs(bin_accuracy - bin_confidence) * bin_weight

        return float(ece)

    @staticmethod
    def maximum_calibration_error(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Maximum Calibration Error (MCE).

        MCE = max(|bin_accuracy - bin_confidence|)

        Lower is better. Focuses on worst-case calibration.
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        bin_edges = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(y_true[mask])
                bin_confidence = np.mean(y_prob[mask])
                mce = max(mce, np.abs(bin_accuracy - bin_confidence))

        return float(mce)

    @staticmethod
    def brier_score(y_prob: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate Brier Score.

        Brier = mean((y_prob - y_true)^2)

        Lower is better. Range: 0 (perfect) to 1 (worst).
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()
        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def log_loss(y_prob: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate log loss (binary cross-entropy).

        Lower is better.
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()
        y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

        return float(-np.mean(
            y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
        ))

    @staticmethod
    def reliability_diagram_data(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, List[float]]:
        """
        Generate data for reliability diagram (calibration curve).

        Returns:
            Dictionary with 'mean_predicted_prob', 'fraction_of_positives',
            'bin_counts' for plotting
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        try:
            fraction_pos, mean_pred = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy='uniform'
            )
        except ValueError:
            # Fall back to manual calculation
            bin_edges = np.linspace(0, 1, n_bins + 1)
            fraction_pos = []
            mean_pred = []
            for i in range(n_bins):
                mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
                if np.sum(mask) > 0:
                    fraction_pos.append(np.mean(y_true[mask]))
                    mean_pred.append(np.mean(y_prob[mask]))

            fraction_pos = np.array(fraction_pos)
            mean_pred = np.array(mean_pred)

        return {
            "mean_predicted_prob": mean_pred.tolist(),
            "fraction_of_positives": fraction_pos.tolist(),
        }

    @classmethod
    def evaluate(
        cls,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> CalibrationMetrics:
        """
        Calculate all calibration metrics.

        Args:
            y_prob: Predicted probabilities
            y_true: True binary labels
            n_bins: Number of bins for ECE/MCE calculation

        Returns:
            CalibrationMetrics object
        """
        return CalibrationMetrics(
            ece=cls.expected_calibration_error(y_prob, y_true, n_bins),
            mce=cls.maximum_calibration_error(y_prob, y_true, n_bins),
            brier_score=cls.brier_score(y_prob, y_true),
            log_loss=cls.log_loss(y_prob, y_true),
            reliability_diagram=cls.reliability_diagram_data(y_prob, y_true, n_bins)
        )


class ModelCalibrator:
    """
    High-level interface for calibrating betting model predictions.

    Automatically selects best calibration method based on validation data.
    Includes new methods for sports betting:
    - shrinkage_beta: Probability shrinkage + Beta calibration (recommended for overconfident models)
    - favorite_longshot: Split calibration for favorites vs underdogs
    """

    def __init__(self, model_name: str = "default", include_advanced: bool = True):
        """
        Args:
            model_name: Name for saving/loading calibrator
            include_advanced: Whether to include advanced calibration methods
                            (shrinkage, favorite-longshot split)
        """
        self.model_name = model_name
        self.calibrators: Dict[str, BaseCalibrator] = {
            "platt": PlattScaling(),
            "isotonic": IsotonicCalibration(),
            "temperature": TemperatureScaling(),
            "beta": BetaCalibration(),
        }

        # Add advanced calibrators for sports betting
        if include_advanced:
            self.calibrators["shrinkage_beta"] = ShrinkagePlusCalibrator(shrinkage=0.15, base_method="beta")
            self.calibrators["shrinkage_platt"] = ShrinkagePlusCalibrator(shrinkage=0.15, base_method="platt")
            self.calibrators["favorite_longshot"] = FavoriteLongshotCalibrator(base_method="beta")

        self.best_method: str = "platt"
        self.metrics: Dict[str, CalibrationMetrics] = {}

    def fit(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        method: str = "auto"
    ) -> "ModelCalibrator":
        """
        Fit calibrator(s) to validation data.

        Args:
            y_prob: Uncalibrated probability predictions
            y_true: True binary labels
            method: Calibration method to use
                   'auto' = try all and pick best by ECE
                   'platt', 'isotonic', 'temperature', 'beta' = use specific method

        Returns:
            self
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        if len(y_prob) < 100:
            logger.warning("Limited calibration data (<100 samples). Results may be unreliable.")

        if method == "auto":
            # Fit all methods and select best
            best_ece = float('inf')

            for name, calibrator in self.calibrators.items():
                try:
                    calibrator.fit(y_prob, y_true)
                    calibrated = calibrator.calibrate(y_prob)
                    metrics = CalibrationEvaluator.evaluate(calibrated, y_true)
                    self.metrics[name] = metrics

                    logger.info(f"{name}: ECE={metrics.ece:.4f}, Brier={metrics.brier_score:.4f}")

                    if metrics.ece < best_ece:
                        best_ece = metrics.ece
                        self.best_method = name

                except Exception as e:
                    logger.warning(f"Failed to fit {name}: {e}")

            logger.info(f"Best calibration method: {self.best_method} (ECE={best_ece:.4f})")

        elif method in self.calibrators:
            self.calibrators[method].fit(y_prob, y_true)
            self.best_method = method
            calibrated = self.calibrators[method].calibrate(y_prob)
            self.metrics[method] = CalibrationEvaluator.evaluate(calibrated, y_true)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'auto', 'platt', 'isotonic', 'temperature', or 'beta'")

        return self

    def calibrate(
        self,
        y_prob: Union[float, np.ndarray],
        method: str = None
    ) -> Union[float, np.ndarray]:
        """
        Calibrate probability predictions.

        Args:
            y_prob: Uncalibrated probability prediction(s)
            method: Calibration method (default: use best from fit)

        Returns:
            Calibrated probability
        """
        method = method or self.best_method

        if method not in self.calibrators:
            raise ValueError(f"Unknown method: {method}")

        # Handle single value
        is_scalar = np.isscalar(y_prob)
        y_prob = np.atleast_1d(y_prob)

        calibrated = self.calibrators[method].calibrate(y_prob)

        if is_scalar:
            return float(calibrated[0])
        return calibrated

    def get_metrics(self, method: str = None) -> Optional[CalibrationMetrics]:
        """Get calibration metrics for a method."""
        method = method or self.best_method
        return self.metrics.get(method)

    def save(self, directory: str = "models/calibration") -> None:
        """Save all fitted calibrators."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        for name, calibrator in self.calibrators.items():
            try:
                ext = "pkl" if name == "isotonic" else "json"
                calibrator.save(str(path / f"{self.model_name}_{name}.{ext}"))
            except Exception as e:
                logger.warning(f"Failed to save {name}: {e}")

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "best_method": self.best_method,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "saved_at": datetime.now().isoformat()
        }
        with open(path / f"{self.model_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Calibrators saved to {directory}")

    def load(self, directory: str = "models/calibration") -> "ModelCalibrator":
        """Load fitted calibrators."""
        path = Path(directory)

        # Load metadata
        metadata_path = path / f"{self.model_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.best_method = metadata.get("best_method", "platt")

        # Load calibrators
        for name, calibrator in self.calibrators.items():
            try:
                ext = "pkl" if name == "isotonic" else "json"
                cal_path = path / f"{self.model_name}_{name}.{ext}"
                if cal_path.exists():
                    calibrator.load(str(cal_path))
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")

        logger.info(f"Calibrators loaded from {directory}")
        return self


# Convenience functions
def calibrate_moneyline_probability(
    raw_prob: float,
    calibrator: ModelCalibrator = None
) -> float:
    """
    Calibrate a moneyline probability prediction.

    Args:
        raw_prob: Raw model probability (0-1)
        calibrator: Pre-fitted calibrator (loads from disk if None)

    Returns:
        Calibrated probability (0-1)
    """
    if calibrator is None:
        calibrator = ModelCalibrator("moneyline")
        try:
            calibrator.load()
        except FileNotFoundError:
            logger.warning("No calibrator found. Returning raw probability.")
            return float(np.clip(raw_prob, 0.0, 1.0))

    return calibrator.calibrate(raw_prob)


def calibrate_spread_probability(
    raw_prob: float,
    calibrator: ModelCalibrator = None
) -> float:
    """
    Calibrate a spread cover probability prediction.

    Args:
        raw_prob: Raw model probability of covering spread (0-1)
        calibrator: Pre-fitted calibrator

    Returns:
        Calibrated probability (0-1)
    """
    if calibrator is None:
        calibrator = ModelCalibrator("spread")
        try:
            calibrator.load()
        except FileNotFoundError:
            logger.warning("No calibrator found. Returning raw probability.")
            return float(np.clip(raw_prob, 0.0, 1.0))

    return calibrator.calibrate(raw_prob)


def evaluate_calibration(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    print_results: bool = True
) -> CalibrationMetrics:
    """
    Quick evaluation of model calibration.

    Args:
        y_prob: Predicted probabilities
        y_true: True binary outcomes
        print_results: Whether to print summary

    Returns:
        CalibrationMetrics object
    """
    metrics = CalibrationEvaluator.evaluate(y_prob, y_true)

    if print_results:
        print(metrics.summary())
        print(f"\nInterpretation:")
        if metrics.ece < 0.05:
            print("  - ECE < 0.05: Well calibrated!")
        elif metrics.ece < 0.10:
            print("  - ECE 0.05-0.10: Reasonably calibrated")
        else:
            print("  - ECE > 0.10: Poorly calibrated - consider recalibration")

    return metrics


# =============================================================================
# STAT-TYPE SPECIFIC CALIBRATORS FOR PLAYER PROPS
# =============================================================================

class PropEdgeCalibrator:
    """
    Calibrator that maps model edge (predicted - line) to OVER probability.

    For player props, we predict a value (e.g., 24.5 points) against a line (e.g., 22.5).
    The edge is +2.0 points. But what's the actual probability of OVER hitting?

    This calibrator learns from historical predictions:
    - Input: edge_percentage = (predicted - line) / line
    - Output: calibrated probability of OVER hitting

    Critical for converting regression outputs to betting probabilities.
    """

    def __init__(self, prop_type: str):
        """
        Args:
            prop_type: The prop type ('points', 'rebounds', 'assists', 'threes', 'pra')
        """
        self.prop_type = prop_type
        self.calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        self._fitted = False
        self._n_samples = 0
        self._edge_range = (None, None)

    def fit(self, edges: np.ndarray, hit_over: np.ndarray) -> "PropEdgeCalibrator":
        """
        Fit the calibrator on historical prediction data.

        Args:
            edges: Array of edge percentages: (predicted - line) / line
            hit_over: Binary array: 1 if actual > line, 0 otherwise

        Returns:
            self
        """
        edges = np.asarray(edges).flatten()
        hit_over = np.asarray(hit_over).flatten().astype(float)

        # Filter out any NaN values
        valid_mask = ~(np.isnan(edges) | np.isnan(hit_over))
        edges = edges[valid_mask]
        hit_over = hit_over[valid_mask]

        if len(edges) < 50:
            logger.warning(f"Only {len(edges)} samples for {self.prop_type} - may be unreliable")

        # Sort by edge (required for isotonic regression)
        sort_idx = np.argsort(edges)
        edges_sorted = edges[sort_idx]
        hit_over_sorted = hit_over[sort_idx]

        self.calibrator.fit(edges_sorted, hit_over_sorted)

        self._fitted = True
        self._n_samples = len(edges)
        self._edge_range = (float(np.min(edges)), float(np.max(edges)))

        logger.info(f"PropEdgeCalibrator fitted for {self.prop_type} on {self._n_samples} samples")
        logger.info(f"  Edge range: [{self._edge_range[0]:.2%}, {self._edge_range[1]:.2%}]")

        return self

    def calibrate(self, edge: float) -> float:
        """
        Convert edge to calibrated OVER probability.

        Args:
            edge: Edge percentage (predicted - line) / line

        Returns:
            Calibrated probability of OVER hitting (0-1)
        """
        if not self._fitted:
            # Fallback: simple sigmoid mapping
            return 1 / (1 + np.exp(-edge * 10))

        return float(self.calibrator.predict([[edge]])[0])

    def calibrate_array(self, edges: np.ndarray) -> np.ndarray:
        """Calibrate an array of edges."""
        if not self._fitted:
            return 1 / (1 + np.exp(-np.asarray(edges) * 10))

        return self.calibrator.predict(np.asarray(edges).reshape(-1, 1))

    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        data = {
            'calibrator': self.calibrator,
            'prop_type': self.prop_type,
            'n_samples': self._n_samples,
            'edge_range': self._edge_range,
            'fitted': self._fitted,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "PropEdgeCalibrator":
        """Load calibrator from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        calibrator = cls(data['prop_type'])
        calibrator.calibrator = data['calibrator']
        calibrator._n_samples = data['n_samples']
        calibrator._edge_range = data['edge_range']
        calibrator._fitted = data['fitted']
        return calibrator


class StatTypeCalibrator:
    """
    Manages separate calibrators for each prop stat type.

    Different stats have different calibration patterns:
    - Points: Higher volume, more predictable
    - Rebounds: Position-dependent, moderate variance
    - Assists: Playmaker-dependent, team-context sensitive
    - 3PM: High variance, regression to mean important
    - PRA: Combined stat, smooths individual variance

    This class:
    1. Maintains separate PropEdgeCalibrators for each stat type
    2. Provides unified interface for calibration
    3. Supports auto-training from PropTracker data
    """

    STAT_TYPES = ['points', 'rebounds', 'assists', 'threes', 'pra']

    def __init__(self):
        self.calibrators: Dict[str, PropEdgeCalibrator] = {}
        self._global_fallback = None  # Used when stat-specific not available

    def fit(self, prop_data: Dict[str, Dict[str, np.ndarray]]) -> "StatTypeCalibrator":
        """
        Fit calibrators for each stat type.

        Args:
            prop_data: Dictionary mapping stat_type to {'edges': array, 'hit_over': array}

        Returns:
            self
        """
        # Collect all data for global fallback
        all_edges = []
        all_hits = []

        for stat_type in self.STAT_TYPES:
            if stat_type in prop_data:
                data = prop_data[stat_type]
                edges = np.asarray(data['edges'])
                hit_over = np.asarray(data['hit_over'])

                if len(edges) >= 30:
                    calibrator = PropEdgeCalibrator(stat_type)
                    calibrator.fit(edges, hit_over)
                    self.calibrators[stat_type] = calibrator
                    logger.info(f"Fitted calibrator for {stat_type}: {len(edges)} samples")
                else:
                    logger.warning(f"Insufficient data for {stat_type}: {len(edges)} samples")

                all_edges.extend(edges)
                all_hits.extend(hit_over)

        # Fit global fallback
        if len(all_edges) >= 50:
            self._global_fallback = PropEdgeCalibrator('global')
            self._global_fallback.fit(np.array(all_edges), np.array(all_hits))
            logger.info(f"Fitted global fallback calibrator: {len(all_edges)} samples")

        return self

    def calibrate(self, edge: float, stat_type: str) -> float:
        """
        Get calibrated OVER probability for a specific stat type.

        Args:
            edge: Edge percentage (predicted - line) / line
            stat_type: The prop type ('points', 'rebounds', etc.)

        Returns:
            Calibrated probability (0-1)
        """
        # Normalize stat type
        stat_type = stat_type.lower()
        if stat_type == '3pm' or stat_type == 'fg3m':
            stat_type = 'threes'

        # Try stat-specific calibrator first
        if stat_type in self.calibrators:
            return self.calibrators[stat_type].calibrate(edge)

        # Fall back to global calibrator
        if self._global_fallback is not None:
            return self._global_fallback.calibrate(edge)

        # Last resort: simple sigmoid
        return 1 / (1 + np.exp(-edge * 10))

    def save(self, directory: Path) -> None:
        """Save all calibrators to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for stat_type, calibrator in self.calibrators.items():
            calibrator.save(directory / f"prop_{stat_type}_calibrator.pkl")

        if self._global_fallback:
            self._global_fallback.save(directory / "prop_global_calibrator.pkl")

        logger.info(f"Saved {len(self.calibrators)} stat-specific calibrators to {directory}")

    @classmethod
    def load(cls, directory: Path) -> "StatTypeCalibrator":
        """Load all calibrators from directory."""
        directory = Path(directory)
        instance = cls()

        for stat_type in cls.STAT_TYPES:
            path = directory / f"prop_{stat_type}_calibrator.pkl"
            if path.exists():
                instance.calibrators[stat_type] = PropEdgeCalibrator.load(path)
                logger.info(f"Loaded calibrator for {stat_type}")

        global_path = directory / "prop_global_calibrator.pkl"
        if global_path.exists():
            instance._global_fallback = PropEdgeCalibrator.load(global_path)
            logger.info("Loaded global fallback calibrator")

        return instance


if __name__ == "__main__":
    # Example usage and demonstration
    print("=" * 60)
    print("Probability Calibration Demo")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Simulate overconfident model (common in ML)
    true_prob = np.random.beta(2, 2, n_samples)  # True probabilities
    y_true = (np.random.random(n_samples) < true_prob).astype(int)

    # Overconfident predictions (pushed toward 0 and 1)
    y_prob_uncalibrated = np.clip(true_prob ** 0.5 * 1.1 - 0.05, 0.01, 0.99)

    print("\nBefore Calibration:")
    metrics_before = evaluate_calibration(y_prob_uncalibrated, y_true)

    # Fit calibrator
    print("\n" + "=" * 60)
    print("Fitting calibrators...")
    print("=" * 60)

    calibrator = ModelCalibrator("demo")
    calibrator.fit(y_prob_uncalibrated, y_true, method="auto")

    # Apply calibration
    y_prob_calibrated = calibrator.calibrate(y_prob_uncalibrated)

    print("\nAfter Calibration:")
    metrics_after = evaluate_calibration(y_prob_calibrated, y_true)

    # Compare
    print("\n" + "=" * 60)
    print("Improvement Summary")
    print("=" * 60)
    print(f"ECE: {metrics_before.ece:.4f} -> {metrics_after.ece:.4f} ({(1 - metrics_after.ece/metrics_before.ece)*100:.1f}% improvement)")
    print(f"Brier: {metrics_before.brier_score:.4f} -> {metrics_after.brier_score:.4f}")

    # Single prediction example
    print("\n" + "=" * 60)
    print("Single Prediction Example")
    print("=" * 60)
    raw_pred = 0.75
    calibrated_pred = calibrator.calibrate(raw_pred)
    print(f"Raw prediction: {raw_pred:.3f}")
    print(f"Calibrated prediction: {calibrated_pred:.3f}")
