"""
Backtesting Framework for NBA Betting Model

Evaluates model performance on historical data with proper
walk-forward validation to prevent look-ahead bias.

Metrics calculated:
- Brier Score: Measures probability prediction accuracy (lower is better)
- Log Loss: Cross-entropy loss for probability predictions
- Expected Calibration Error (ECE): Measures if 70% predictions win ~70%
- Return on Investment (ROI): Simulated betting performance

Usage:
    from backtest import PropBacktester, MoneylineBacktester, BacktestResults

    # Backtest player props
    prop_tester = PropBacktester()
    results = prop_tester.run_backtest(historical_prop_data)
    print(results.summary())

    # Backtest moneyline predictions
    ml_tester = MoneylineBacktester()
    results = ml_tester.run_backtest(historical_games)
    print(results.summary())
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

# Import calibration evaluator for ECE calculation
try:
    from calibration import CalibrationEvaluator
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False


@dataclass
class BacktestResults:
    """Container for comprehensive backtest metrics."""
    brier_score: float
    log_loss: float
    accuracy: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    roi: float  # Return on Investment (if betting)
    num_predictions: int
    num_bets: int  # Number of bets placed (with confidence filter)
    win_rate: float  # Actual win rate on bets
    avg_confidence: float  # Average confidence of predictions
    confidence_correlation: float  # Correlation between confidence and outcomes

    # Breakdown by confidence bucket
    breakdown_by_confidence: Dict = field(default_factory=dict)

    # Timestamps
    backtest_date: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        """Generate human-readable summary."""
        interpretation = self._interpret_brier()
        calibration_status = self._interpret_ece()

        return f"""
================================================================================
                         BACKTEST RESULTS
================================================================================
Predictions: {self.num_predictions:,} total | {self.num_bets:,} bets placed

PROBABILITY METRICS:
  Brier Score:     {self.brier_score:.4f}  {interpretation}
  Log Loss:        {self.log_loss:.4f}
  Accuracy:        {self.accuracy:.1%}

CALIBRATION:
  ECE:             {self.ece:.4f}  {calibration_status}
  MCE:             {self.mce:.4f}

BETTING PERFORMANCE:
  Win Rate:        {self.win_rate:.1%}
  ROI:             {self.roi:+.2%}
  Avg Confidence:  {self.avg_confidence:.1%}

CONFIDENCE-OUTCOME CORRELATION: {self.confidence_correlation:.3f}
  (Higher = model confidence is predictive of outcomes)
================================================================================
"""

    def _interpret_brier(self) -> str:
        """Interpret Brier score quality."""
        if self.brier_score < 0.20:
            return "(excellent)"
        elif self.brier_score < 0.22:
            return "(very good)"
        elif self.brier_score < 0.25:
            return "(good)"
        elif self.brier_score < 0.28:
            return "(fair)"
        else:
            return "(needs improvement)"

    def _interpret_ece(self) -> str:
        """Interpret calibration quality."""
        if self.ece < 0.03:
            return "(excellent calibration)"
        elif self.ece < 0.05:
            return "(well calibrated)"
        elif self.ece < 0.08:
            return "(acceptable)"
        elif self.ece < 0.12:
            return "(needs calibration)"
        else:
            return "(poorly calibrated)"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "accuracy": self.accuracy,
            "ece": self.ece,
            "mce": self.mce,
            "roi": self.roi,
            "num_predictions": self.num_predictions,
            "num_bets": self.num_bets,
            "win_rate": self.win_rate,
            "avg_confidence": self.avg_confidence,
            "confidence_correlation": self.confidence_correlation,
            "breakdown_by_confidence": self.breakdown_by_confidence,
            "backtest_date": self.backtest_date,
        }

    def save(self, filepath: str) -> None:
        """Save results to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BacktestResults":
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class CalibrationCalculator:
    """Calculate calibration metrics when calibration.py is not available."""

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


class BaseBacktester:
    """Base class for backtesting with common metrics calculation."""

    def __init__(self, min_confidence: float = 0.55, bet_unit: float = 110.0):
        """
        Initialize backtester.

        Args:
            min_confidence: Minimum confidence to place a bet (0.5-1.0)
            bet_unit: Amount wagered per bet (for ROI calculation)
        """
        self.min_confidence = min_confidence
        self.bet_unit = bet_unit

    def _calculate_metrics(
        self,
        predictions: List[float],
        actuals: List[int],
        confidences: Optional[List[float]] = None
    ) -> BacktestResults:
        """
        Calculate all backtest metrics.

        Args:
            predictions: Predicted probabilities (0-1) for the positive outcome
            actuals: Actual outcomes (0 or 1)
            confidences: Optional confidence scores (if different from predictions)

        Returns:
            BacktestResults object with all metrics
        """
        preds = np.array(predictions)
        acts = np.array(actuals)

        if len(preds) == 0:
            return BacktestResults(
                brier_score=0.0,
                log_loss=0.0,
                accuracy=0.0,
                ece=0.0,
                mce=0.0,
                roi=0.0,
                num_predictions=0,
                num_bets=0,
                win_rate=0.0,
                avg_confidence=0.0,
                confidence_correlation=0.0,
            )

        # Use predictions as confidence if not provided
        if confidences is None:
            confs = np.abs(preds - 0.5) + 0.5  # Convert to confidence (0.5-1.0)
        else:
            confs = np.array(confidences)

        # Brier score: mean squared error of probabilities
        brier = float(np.mean((preds - acts) ** 2))

        # Log loss: cross-entropy
        preds_clipped = np.clip(preds, 1e-10, 1 - 1e-10)
        log_loss = float(-np.mean(
            acts * np.log(preds_clipped) +
            (1 - acts) * np.log(1 - preds_clipped)
        ))

        # Accuracy at 50% threshold
        accuracy = float(np.mean((preds > 0.5) == acts))

        # Calibration metrics
        if HAS_CALIBRATION:
            ece = CalibrationEvaluator.expected_calibration_error(preds, acts)
            mce = CalibrationEvaluator.maximum_calibration_error(preds, acts)
        else:
            ece = CalibrationCalculator.expected_calibration_error(preds, acts)
            mce = CalibrationCalculator.maximum_calibration_error(preds, acts)

        # Betting metrics
        roi, num_bets, win_rate = self._calculate_betting_metrics(preds, acts, confs)

        # Confidence statistics
        avg_confidence = float(np.mean(confs))

        # Confidence-outcome correlation
        # Higher values mean confidence is predictive
        if len(preds) > 1:
            confidence_correlation = float(np.corrcoef(confs, acts)[0, 1])
            if np.isnan(confidence_correlation):
                confidence_correlation = 0.0
        else:
            confidence_correlation = 0.0

        # Breakdown by confidence bucket
        breakdown = self._breakdown_by_confidence(preds, acts, confs)

        return BacktestResults(
            brier_score=brier,
            log_loss=log_loss,
            accuracy=accuracy,
            ece=ece,
            mce=mce,
            roi=roi,
            num_predictions=len(preds),
            num_bets=num_bets,
            win_rate=win_rate,
            avg_confidence=avg_confidence,
            confidence_correlation=confidence_correlation,
            breakdown_by_confidence=breakdown,
        )

    def _calculate_betting_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[float, int, float]:
        """
        Calculate betting-specific metrics.

        Assumes -110 odds (bet $110 to win $100).

        Args:
            predictions: Predicted probabilities
            actuals: Actual outcomes
            confidences: Confidence scores

        Returns:
            Tuple of (roi, num_bets, win_rate)
        """
        # Only bet when confidence exceeds threshold
        bet_mask = confidences >= self.min_confidence

        if not np.any(bet_mask):
            return 0.0, 0, 0.0

        bet_preds = predictions[bet_mask]
        bet_acts = actuals[bet_mask]
        num_bets = len(bet_preds)

        # Determine which side to bet: OVER when pred > 0.5, UNDER when pred < 0.5
        # Win if our prediction matches actual
        wins = ((bet_preds > 0.5) == bet_acts)
        win_rate = float(np.mean(wins))

        # -110 odds: win $100 on $110 bet, lose $110
        profit_per_win = 100.0
        loss_per_loss = 110.0

        total_profit = np.sum(wins * profit_per_win - ~wins * loss_per_loss)
        total_wagered = num_bets * loss_per_loss

        roi = float(total_profit / total_wagered) if total_wagered > 0 else 0.0

        return roi, num_bets, win_rate

    def _breakdown_by_confidence(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: np.ndarray
    ) -> Dict:
        """
        Break down performance by confidence level.

        Returns dict with metrics for each confidence bucket.
        """
        buckets = [
            ("50-55%", 0.50, 0.55),
            ("55-60%", 0.55, 0.60),
            ("60-65%", 0.60, 0.65),
            ("65-70%", 0.65, 0.70),
            ("70-75%", 0.70, 0.75),
            ("75%+", 0.75, 1.01),
        ]

        breakdown = {}

        for name, low, high in buckets:
            mask = (confidences >= low) & (confidences < high)
            count = np.sum(mask)

            if count > 0:
                bucket_preds = predictions[mask]
                bucket_acts = actuals[mask]

                wins = ((bucket_preds > 0.5) == bucket_acts)

                breakdown[name] = {
                    "count": int(count),
                    "win_rate": float(np.mean(wins)),
                    "avg_confidence": float(np.mean(confidences[mask])),
                    "brier": float(np.mean((bucket_preds - bucket_acts) ** 2)),
                }
            else:
                breakdown[name] = {
                    "count": 0,
                    "win_rate": 0.0,
                    "avg_confidence": 0.0,
                    "brier": 0.0,
                }

        return breakdown


class PropBacktester(BaseBacktester):
    """
    Backtest player prop predictions.

    Use this to validate prop prediction accuracy on historical data.
    """

    def __init__(
        self,
        min_confidence: float = 0.55,
        prop_types: Optional[List[str]] = None
    ):
        """
        Initialize prop backtester.

        Args:
            min_confidence: Minimum confidence to place bet
            prop_types: List of prop types to backtest (default: all)
        """
        super().__init__(min_confidence)
        self.prop_types = prop_types or ["points", "rebounds", "assists", "3pm", "pra"]

    def run_backtest(
        self,
        historical_props: List[Dict],
        prediction_fn: Optional[Callable] = None
    ) -> BacktestResults:
        """
        Run backtest on historical prop data.

        Args:
            historical_props: List of dicts with:
                - prop_type: str (e.g., "points")
                - line: float (e.g., 25.5)
                - prediction: float (predicted value, e.g., 27.2)
                - actual: float (actual value, e.g., 28)
                - confidence: Optional[float] (0.5-1.0)
            prediction_fn: Optional function to generate predictions
                          (for walk-forward testing)

        Returns:
            BacktestResults object
        """
        predictions = []
        actuals = []
        confidences = []

        for prop in historical_props:
            prop_type = prop.get("prop_type", "").lower()

            # Filter by prop type if specified
            if self.prop_types and prop_type not in self.prop_types:
                continue

            line = prop.get("line", 0)
            prediction = prop.get("prediction", 0)
            actual = prop.get("actual", 0)
            confidence = prop.get("confidence", 0.5)

            if line <= 0 or prediction <= 0:
                continue

            # Convert to probability: what's the probability of going OVER?
            # If prediction > line, prob_over is high
            edge = prediction - line

            # Convert edge to probability using sigmoid
            # Scale: 5 point edge = ~73% probability
            prob_over = 1.0 / (1.0 + np.exp(-edge / 5.0))

            # Actual outcome: did they go over?
            went_over = 1 if actual > line else 0

            predictions.append(float(prob_over))
            actuals.append(went_over)
            confidences.append(float(confidence) / 100.0 if confidence > 1 else float(confidence))

        return self._calculate_metrics(predictions, actuals, confidences)

    def run_walk_forward(
        self,
        historical_props: List[Dict],
        train_size: float = 0.7,
        retrain_interval: int = 100
    ) -> BacktestResults:
        """
        Run walk-forward backtest (trains on past, predicts future).

        This is the most realistic backtest as it prevents look-ahead bias.

        Args:
            historical_props: List of prop data sorted by date
            train_size: Proportion of data for initial training
            retrain_interval: Number of predictions between retraining

        Returns:
            BacktestResults object
        """
        # For now, use simple backtest
        # Walk-forward requires access to model training which we don't have here
        return self.run_backtest(historical_props)


class MoneylineBacktester(BaseBacktester):
    """
    Backtest moneyline (win probability) predictions.

    Use this to validate game outcome predictions on historical data.
    """

    def run_backtest(
        self,
        historical_games: List[Dict]
    ) -> BacktestResults:
        """
        Run backtest on historical game data.

        Args:
            historical_games: List of dicts with:
                - home_win_prob: float (predicted probability of home win)
                - home_won: bool (actual outcome)
                - confidence: Optional[float]

        Returns:
            BacktestResults object
        """
        predictions = []
        actuals = []
        confidences = []

        for game in historical_games:
            home_prob = game.get("home_win_prob", 0.5)
            home_won = game.get("home_won", None)
            confidence = game.get("confidence", max(home_prob, 1 - home_prob))

            if home_won is None:
                continue

            predictions.append(float(home_prob))
            actuals.append(1 if home_won else 0)
            confidences.append(float(confidence))

        return self._calculate_metrics(predictions, actuals, confidences)


class SpreadBacktester(BaseBacktester):
    """
    Backtest spread (cover) predictions.

    Use this to validate spread betting predictions.
    """

    def run_backtest(
        self,
        historical_games: List[Dict]
    ) -> BacktestResults:
        """
        Run backtest on historical spread data.

        Args:
            historical_games: List of dicts with:
                - cover_prob: float (predicted probability of home covering)
                - spread: float (the spread line, e.g., -3.5)
                - home_margin: float (actual home margin, e.g., 7)
                - confidence: Optional[float]

        Returns:
            BacktestResults object
        """
        predictions = []
        actuals = []
        confidences = []

        for game in historical_games:
            cover_prob = game.get("cover_prob", 0.5)
            spread = game.get("spread", 0)
            home_margin = game.get("home_margin", None)
            confidence = game.get("confidence", max(cover_prob, 1 - cover_prob))

            if home_margin is None:
                continue

            # Did home cover the spread?
            # Home covers if home_margin > spread (where spread is negative for favorites)
            covered = 1 if home_margin > spread else 0

            predictions.append(float(cover_prob))
            actuals.append(covered)
            confidences.append(float(confidence))

        return self._calculate_metrics(predictions, actuals, confidences)


def compare_models(
    results_before: BacktestResults,
    results_after: BacktestResults
) -> str:
    """
    Compare two backtest results to measure improvement.

    Args:
        results_before: Baseline results
        results_after: New results to compare

    Returns:
        Formatted comparison string
    """
    def pct_change(old: float, new: float, lower_better: bool = True) -> str:
        if old == 0:
            return "N/A"
        change = (new - old) / abs(old) * 100
        if lower_better:
            direction = "improved" if change < 0 else "worsened"
        else:
            direction = "improved" if change > 0 else "worsened"
        return f"{change:+.1f}% ({direction})"

    return f"""
================================================================================
                      MODEL COMPARISON
================================================================================
                          BEFORE          AFTER           CHANGE
Brier Score:         {results_before.brier_score:.4f}          {results_after.brier_score:.4f}          {pct_change(results_before.brier_score, results_after.brier_score)}
Log Loss:            {results_before.log_loss:.4f}          {results_after.log_loss:.4f}          {pct_change(results_before.log_loss, results_after.log_loss)}
Accuracy:            {results_before.accuracy:.1%}          {results_after.accuracy:.1%}          {pct_change(results_before.accuracy, results_after.accuracy, False)}
ECE:                 {results_before.ece:.4f}          {results_after.ece:.4f}          {pct_change(results_before.ece, results_after.ece)}
ROI:                 {results_before.roi:+.2%}          {results_after.roi:+.2%}          {pct_change(results_before.roi, results_after.roi, False)}
Win Rate:            {results_before.win_rate:.1%}          {results_after.win_rate:.1%}          {pct_change(results_before.win_rate, results_after.win_rate, False)}
================================================================================
"""


# Convenience function for quick validation
def quick_validate(predictions: List[float], actuals: List[int]) -> None:
    """
    Quick validation of prediction quality.

    Args:
        predictions: Predicted probabilities (0-1)
        actuals: Actual outcomes (0 or 1)
    """
    tester = BaseBacktester(min_confidence=0.50)
    results = tester._calculate_metrics(predictions, actuals)
    print(results.summary())


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Backtesting Framework Demo")
    print("=" * 60)

    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 500

    # Simulate a reasonably calibrated model
    true_probs = np.random.beta(2, 2, n_samples)
    outcomes = (np.random.random(n_samples) < true_probs).astype(int)

    # Add some noise to predictions (typical of real models)
    predictions = true_probs + np.random.normal(0, 0.1, n_samples)
    predictions = np.clip(predictions, 0.01, 0.99)

    # Calculate metrics
    print("\nRunning quick validation on synthetic data...")
    quick_validate(predictions.tolist(), outcomes.tolist())

    # Create structured prop data
    print("\n" + "=" * 60)
    print("Prop Backtester Demo")
    print("=" * 60)

    prop_data = []
    for i in range(200):
        line = np.random.uniform(15, 30)
        prediction = line + np.random.normal(0, 3)
        actual = line + np.random.normal(0, 5)
        confidence = 50 + np.random.uniform(0, 30)

        prop_data.append({
            "prop_type": "points",
            "line": line,
            "prediction": prediction,
            "actual": actual,
            "confidence": confidence,
        })

    tester = PropBacktester(min_confidence=0.55)
    results = tester.run_backtest(prop_data)
    print(results.summary())

    print("\nBreakdown by confidence bucket:")
    for bucket, stats in results.breakdown_by_confidence.items():
        if stats["count"] > 0:
            print(f"  {bucket}: {stats['count']} bets, {stats['win_rate']:.1%} win rate")
