"""
Minutes Oracle Validation Module

Provides comprehensive validation metrics:
- Calibration: Are predictions properly calibrated at each quantile?
- Coverage: Do prediction intervals contain the expected % of actuals?
- RMSE by bucket: Performance in different game contexts
- Comparison to baselines
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .minutes_predictor import MinutesPredictor, MinutesDistribution


@dataclass
class ValidationResult:
    """Container for validation results."""
    # Overall metrics
    median_rmse: float
    median_mae: float
    baseline_rmse: float
    rmse_improvement_pct: float

    # Calibration (% of actuals <= predicted quantile)
    p10_calibration: float  # Target: 10%
    p25_calibration: float  # Target: 25%
    p50_calibration: float  # Target: 50%
    p75_calibration: float  # Target: 75%
    p90_calibration: float  # Target: 90%

    # Coverage
    p10_p90_coverage: float  # Target: 80%
    p25_p75_coverage: float  # Target: 50%

    # By game type
    rmse_close_games: float  # Spread < 5
    rmse_medium_games: float  # Spread 5-10
    rmse_blowout_games: float  # Spread > 10

    # By player type
    rmse_starters: float  # High-minute players
    rmse_rotation: float  # Medium-minute players
    rmse_bench: float  # Low-minute players

    # Sample counts
    n_total: int
    n_close: int
    n_medium: int
    n_blowout: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'overall': {
                'median_rmse': round(self.median_rmse, 2),
                'median_mae': round(self.median_mae, 2),
                'baseline_rmse': round(self.baseline_rmse, 2),
                'improvement_pct': round(self.rmse_improvement_pct * 100, 1),
            },
            'calibration': {
                'p10': round(self.p10_calibration * 100, 1),
                'p25': round(self.p25_calibration * 100, 1),
                'p50': round(self.p50_calibration * 100, 1),
                'p75': round(self.p75_calibration * 100, 1),
                'p90': round(self.p90_calibration * 100, 1),
            },
            'coverage': {
                'p10_p90': round(self.p10_p90_coverage * 100, 1),
                'p25_p75': round(self.p25_p75_coverage * 100, 1),
            },
            'by_game_type': {
                'close_rmse': round(self.rmse_close_games, 2),
                'medium_rmse': round(self.rmse_medium_games, 2),
                'blowout_rmse': round(self.rmse_blowout_games, 2),
            },
            'by_player_type': {
                'starters_rmse': round(self.rmse_starters, 2),
                'rotation_rmse': round(self.rmse_rotation, 2),
                'bench_rmse': round(self.rmse_bench, 2),
            },
            'sample_counts': {
                'total': self.n_total,
                'close': self.n_close,
                'medium': self.n_medium,
                'blowout': self.n_blowout,
            },
        }

    def print_report(self):
        """Print a formatted validation report."""
        print("\n" + "=" * 60)
        print("MINUTES ORACLE VALIDATION REPORT")
        print("=" * 60)

        print(f"\nOverall Performance ({self.n_total} samples):")
        print(f"  Median RMSE: {self.median_rmse:.2f} min")
        print(f"  Median MAE: {self.median_mae:.2f} min")
        print(f"  Baseline RMSE: {self.baseline_rmse:.2f} min")
        print(f"  Improvement: {self.rmse_improvement_pct:.1%}")

        print("\nCalibration (actual % <= predicted quantile):")
        print(f"  P10: {self.p10_calibration:.1%} (target: 10%)")
        print(f"  P25: {self.p25_calibration:.1%} (target: 25%)")
        print(f"  P50: {self.p50_calibration:.1%} (target: 50%)")
        print(f"  P75: {self.p75_calibration:.1%} (target: 75%)")
        print(f"  P90: {self.p90_calibration:.1%} (target: 90%)")

        print("\nInterval Coverage:")
        print(f"  P10-P90: {self.p10_p90_coverage:.1%} (target: 80%)")
        print(f"  P25-P75: {self.p25_p75_coverage:.1%} (target: 50%)")

        print("\nBy Game Type:")
        print(f"  Close (spread < 5):  {self.rmse_close_games:.2f} RMSE ({self.n_close} games)")
        print(f"  Medium (spread 5-10): {self.rmse_medium_games:.2f} RMSE ({self.n_medium} games)")
        print(f"  Blowout (spread > 10): {self.rmse_blowout_games:.2f} RMSE ({self.n_blowout} games)")

        print("\nBy Player Type:")
        print(f"  Starters (30+ min): {self.rmse_starters:.2f} RMSE")
        print(f"  Rotation (20-30 min): {self.rmse_rotation:.2f} RMSE")
        print(f"  Bench (10-20 min): {self.rmse_bench:.2f} RMSE")

        # Overall assessment
        print("\n" + "-" * 60)
        print("Assessment:")

        issues = []
        if abs(self.p50_calibration - 0.50) > 0.05:
            issues.append(f"P50 calibration off by {abs(self.p50_calibration - 0.50):.1%}")
        if self.p10_p90_coverage < 0.75:
            issues.append(f"P10-P90 coverage low ({self.p10_p90_coverage:.1%})")
        if self.rmse_improvement_pct < 0.05:
            issues.append(f"Limited improvement over baseline ({self.rmse_improvement_pct:.1%})")

        if not issues:
            print("  Model is well-calibrated and shows good improvement over baseline.")
        else:
            print("  Potential issues:")
            for issue in issues:
                print(f"    - {issue}")

        print("=" * 60)


class MinutesOracleValidator:
    """
    Validates Minutes Oracle predictions against actual outcomes.
    """

    def __init__(self, predictor: MinutesPredictor | None = None):
        """
        Initialize validator.

        Args:
            predictor: Trained MinutesPredictor instance
        """
        self.predictor = predictor

    def load_model(self, model_path: str):
        """Load a trained model for validation."""
        self.predictor = MinutesPredictor.load(model_path)

    def validate(self,
                 features: pd.DataFrame,
                 actuals: np.ndarray,
                 spreads: np.ndarray | None = None,
                 player_avg_mins: np.ndarray | None = None) -> ValidationResult:
        """
        Run full validation on test data.

        Args:
            features: Feature DataFrame
            actuals: Actual minutes played
            spreads: Optional absolute Vegas spreads for stratification
            player_avg_mins: Optional player season averages for baseline

        Returns:
            ValidationResult with all metrics
        """
        if self.predictor is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Get predictions
        predictions = self.predictor.predict_batch(features)

        # Extract prediction arrays
        p10 = np.array([p.p10 for p in predictions])
        p25 = np.array([p.p25 for p in predictions])
        p50 = np.array([p.p50 for p in predictions])
        p75 = np.array([p.p75 for p in predictions])
        p90 = np.array([p.p90 for p in predictions])

        n_total = len(actuals)

        # Overall metrics
        median_rmse = np.sqrt(np.mean((actuals - p50) ** 2))
        median_mae = np.mean(np.abs(actuals - p50))

        # Baseline (season average)
        if player_avg_mins is not None:
            baseline = player_avg_mins
        elif 'season_min_avg' in features.columns:
            baseline = features['season_min_avg'].values
        else:
            baseline = np.full_like(actuals, actuals.mean())

        baseline_rmse = np.sqrt(np.mean((actuals - baseline) ** 2))
        rmse_improvement = (baseline_rmse - median_rmse) / baseline_rmse if baseline_rmse > 0 else 0

        # Calibration
        p10_cal = np.mean(actuals <= p10)
        p25_cal = np.mean(actuals <= p25)
        p50_cal = np.mean(actuals <= p50)
        p75_cal = np.mean(actuals <= p75)
        p90_cal = np.mean(actuals <= p90)

        # Coverage
        p10_p90_cov = np.mean((actuals >= p10) & (actuals <= p90))
        p25_p75_cov = np.mean((actuals >= p25) & (actuals <= p75))

        # By game type (spread buckets)
        if spreads is not None:
            close_mask = spreads < 5
            medium_mask = (spreads >= 5) & (spreads < 10)
            blowout_mask = spreads >= 10
        elif 'vegas_spread_abs' in features.columns:
            spreads = features['vegas_spread_abs'].values
            close_mask = spreads < 5
            medium_mask = (spreads >= 5) & (spreads < 10)
            blowout_mask = spreads >= 10
        else:
            close_mask = np.ones(n_total, dtype=bool)
            medium_mask = np.zeros(n_total, dtype=bool)
            blowout_mask = np.zeros(n_total, dtype=bool)

        rmse_close = np.sqrt(np.mean((actuals[close_mask] - p50[close_mask]) ** 2)) if close_mask.sum() > 0 else 0
        rmse_medium = np.sqrt(np.mean((actuals[medium_mask] - p50[medium_mask]) ** 2)) if medium_mask.sum() > 0 else 0
        rmse_blowout = np.sqrt(np.mean((actuals[blowout_mask] - p50[blowout_mask]) ** 2)) if blowout_mask.sum() > 0 else 0

        # By player type (based on actual minutes)
        starter_mask = actuals >= 30
        rotation_mask = (actuals >= 20) & (actuals < 30)
        bench_mask = (actuals >= 10) & (actuals < 20)

        rmse_starters = np.sqrt(np.mean((actuals[starter_mask] - p50[starter_mask]) ** 2)) if starter_mask.sum() > 0 else 0
        rmse_rotation = np.sqrt(np.mean((actuals[rotation_mask] - p50[rotation_mask]) ** 2)) if rotation_mask.sum() > 0 else 0
        rmse_bench = np.sqrt(np.mean((actuals[bench_mask] - p50[bench_mask]) ** 2)) if bench_mask.sum() > 0 else 0

        return ValidationResult(
            median_rmse=median_rmse,
            median_mae=median_mae,
            baseline_rmse=baseline_rmse,
            rmse_improvement_pct=rmse_improvement,
            p10_calibration=p10_cal,
            p25_calibration=p25_cal,
            p50_calibration=p50_cal,
            p75_calibration=p75_cal,
            p90_calibration=p90_cal,
            p10_p90_coverage=p10_p90_cov,
            p25_p75_coverage=p25_p75_cov,
            rmse_close_games=rmse_close,
            rmse_medium_games=rmse_medium,
            rmse_blowout_games=rmse_blowout,
            rmse_starters=rmse_starters,
            rmse_rotation=rmse_rotation,
            rmse_bench=rmse_bench,
            n_total=n_total,
            n_close=int(close_mask.sum()),
            n_medium=int(medium_mask.sum()),
            n_blowout=int(blowout_mask.sum()),
        )

    def validate_calibration_curve(self,
                                    features: pd.DataFrame,
                                    actuals: np.ndarray,
                                    n_bins: int = 10) -> dict[str, list[float]]:
        """
        Calculate calibration curve data for plotting.

        Args:
            features: Feature DataFrame
            actuals: Actual minutes played
            n_bins: Number of bins for calibration

        Returns:
            Dict with 'predicted_quantiles' and 'actual_fractions'
        """
        if self.predictor is None:
            raise RuntimeError("No model loaded.")

        predictions = self.predictor.predict_batch(features)
        np.array([p.p50 for p in predictions])

        # Calculate quantile bins
        quantiles = np.linspace(0, 1, n_bins + 1)
        predicted_quantiles = []
        actual_fractions = []

        for i in range(n_bins):
            q_low, q_high = quantiles[i], quantiles[i + 1]
            q_mid = (q_low + q_high) / 2

            # Get prediction at this quantile
            if q_mid <= 0.10:
                preds = np.array([p.p10 for p in predictions])
            elif q_mid <= 0.25:
                # Interpolate between p10 and p25
                w = (q_mid - 0.10) / 0.15
                preds = np.array([(1-w) * p.p10 + w * p.p25 for p in predictions])
            elif q_mid <= 0.50:
                w = (q_mid - 0.25) / 0.25
                preds = np.array([(1-w) * p.p25 + w * p.p50 for p in predictions])
            elif q_mid <= 0.75:
                w = (q_mid - 0.50) / 0.25
                preds = np.array([(1-w) * p.p50 + w * p.p75 for p in predictions])
            elif q_mid <= 0.90:
                w = (q_mid - 0.75) / 0.15
                preds = np.array([(1-w) * p.p75 + w * p.p90 for p in predictions])
            else:
                preds = np.array([p.p90 for p in predictions])

            actual_frac = np.mean(actuals <= preds)
            predicted_quantiles.append(q_mid)
            actual_fractions.append(actual_frac)

        return {
            'predicted_quantiles': predicted_quantiles,
            'actual_fractions': actual_fractions,
        }


def run_quick_validation(model_path: str = 'models/minutes_oracle.pkl',
                         n_samples: int = 1000) -> ValidationResult:
    """
    Run a quick validation using synthetic data.

    Useful for testing the validation pipeline without loading full data.
    """
    from .minutes_features import MINUTES_FEATURE_NAMES

    print("Running quick validation with synthetic data...")

    # Load model
    validator = MinutesOracleValidator()
    validator.load_model(model_path)

    # Generate synthetic features
    np.random.seed(42)
    features = pd.DataFrame({
        name: np.random.randn(n_samples) for name in MINUTES_FEATURE_NAMES
    })

    # Set realistic ranges
    features['season_min_avg'] = np.random.uniform(15, 38, n_samples)
    features['recent_min_avg'] = features['season_min_avg'] + np.random.randn(n_samples) * 2
    features['vegas_spread_abs'] = np.random.uniform(0, 15, n_samples)
    features['is_home'] = np.random.choice([0, 1], n_samples)
    features['is_back_to_back'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    features['days_rest'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.5, 0.2, 0.1])

    # Generate synthetic actuals (correlated with season average)
    actuals = features['season_min_avg'].values + np.random.randn(n_samples) * 5
    actuals = np.clip(actuals, 10, 48)

    # Run validation
    result = validator.validate(features, actuals)
    result.print_report()

    return result


if __name__ == '__main__':
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/minutes_oracle.pkl'

    if Path(model_path).exists():
        run_quick_validation(model_path)
    else:
        print(f"Model not found at {model_path}")
        print("Train the model first with: python -m minutes_oracle.minutes_trainer")
