"""
NBA Betting Model Trainer

Implements machine learning models for NBA betting predictions:
- Logistic Regression for moneyline (win probability)
- Support Vector Machines for spread predictions
- Random Forest for player props
- Parlay probability calculator
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
# DIVERSITY MODELS: Add non-tree-based models to reduce ensemble correlation
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings('ignore')

# PRODUCTION FIX: Smart feature defaults for predictions (not zeros)
PREDICTION_FEATURE_DEFAULTS = {
    # Player averages (conservative estimates)
    'season_pts_avg': 10.0, 'recent_pts_avg': 10.0,
    'season_reb_avg': 4.0, 'recent_reb_avg': 4.0,
    'season_ast_avg': 2.5, 'recent_ast_avg': 2.5,
    'season_fg3m_avg': 1.0, 'recent_fg3m_avg': 1.0,
    'season_min_avg': 20.0, 'recent_min_avg': 20.0,
    'pra_avg': 16.5,
    # Team stats (league average)
    'off_rating': 114.0, 'def_rating': 114.0, 'net_rating': 0.0, 'pace': 100.0,
    # Game context
    'days_rest': 2, 'is_home': 0.5, 'is_back_to_back': 0,
    # Elo ratings
    'elo_diff': 0.0, 'home_elo': 1500.0, 'away_elo': 1500.0,
}


def smart_fillna_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply smart defaults for missing features instead of zeros."""
    result = df.copy()
    for col in result.columns:
        if result[col].isna().any():
            if col in PREDICTION_FEATURE_DEFAULTS:
                default = PREDICTION_FEATURE_DEFAULTS[col]
            elif 'rating' in col.lower():
                default = 114.0
            elif 'elo' in col.lower():
                default = 0.0 if 'diff' in col.lower() else 1500.0
            elif 'pct' in col.lower() or 'rate' in col.lower():
                default = 0.5
            else:
                default = 0.0
            result[col] = result[col].fillna(default)
    return result


# Try to import XGBoost and LightGBM (optional but recommended)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Run: pip install xgboost")

# Import calibration module for probability calibration
try:
    from calibration import ModelCalibrator, calibrate_moneyline_probability
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError) as e:
    HAS_LIGHTGBM = False
    # OSError can occur if libomp is missing on macOS
    # print("LightGBM not available (may need libomp on macOS)")
    pass

# Model save directory
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Metrics directory
METRICS_DIR = Path("training_metrics")
METRICS_DIR.mkdir(exist_ok=True)

# Backtest results directory
BACKTEST_DIR = Path("backtest_results")
BACKTEST_DIR.mkdir(exist_ok=True)

# =============================================================================
# BACKTEST THRESHOLDS FOR MODEL PROMOTION
# =============================================================================
# Models must meet these thresholds in walk-forward backtesting before being
# promoted to production. This ensures we only deploy profitable models.

BACKTEST_THRESHOLDS = {
    "moneyline": {
        "min_roi": 0.0,           # Must be profitable (ROI >= 0%)
        "min_win_rate": 52.0,     # Above break-even at -110 odds
        "max_ece": 0.10,          # Expected Calibration Error < 10%
        "min_sharpe": 0.3,        # Risk-adjusted return threshold
    },
    "spread": {
        "min_roi": 0.0,
        "min_win_rate": 52.0,
        "max_ece": 0.10,
        "min_sharpe": 0.3,
    },
    "prop": {
        "min_roi": 0.0,
        "min_win_rate": 52.0,
        "max_ece": 0.15,          # Props can be noisier
        "min_sharpe": 0.2,
    }
}

# =============================================================================
# SANITY LIMITS - IMPOSSIBLE METRICS DETECTION (DATA LEAKAGE PREVENTION)
# =============================================================================
# If ANY metric exceeds these limits, it indicates DATA LEAKAGE or a bug.
# These are mathematically impossible for legitimate sports betting models.
# Professional sports bettors achieve 3-8% ROI, 54-57% win rates.

SANITY_LIMITS = {
    "max_roi": 30.0,              # ROI > 30% is impossible (data leakage)
    "max_win_rate": 62.0,         # Win rate > 62% is impossible at -110
    "max_sharpe": 3.5,            # Sharpe > 3.5 exceeds hedge funds
    "min_ece": 0.02,              # ECE < 0.02 is suspiciously perfect
    "max_training_roi": 50.0,     # Training ROI > 50% = testing on train data
}


class DataLeakageError(Exception):
    """Raised when impossible metrics indicate data leakage."""
    pass


def check_sanity_limits(
    results: Dict,
    context: str = "backtest",
    raise_on_failure: bool = True
) -> Tuple[bool, List[str]]:
    """
    Check if results exceed sanity limits indicating data leakage.

    This is a CRITICAL safety check. If metrics are "too good to be true",
    they indicate the model has seen future data (data leakage).

    Args:
        results: Results dictionary with metrics
        context: Where this check is being run (for error messages)
        raise_on_failure: If True, raise DataLeakageError on detection

    Returns:
        Tuple of (is_sane: bool, violations: List[str])

    Raises:
        DataLeakageError: If raise_on_failure=True and leakage detected
    """
    violations = []

    # Check ROI
    roi = results.get("overall_roi", results.get("roi", results.get("betting_roi_pct", 0)))
    if roi is not None and roi > SANITY_LIMITS["max_roi"]:
        violations.append(
            f"IMPOSSIBLE ROI: {roi:.1f}% > {SANITY_LIMITS['max_roi']}% limit. "
            f"This indicates DATA LEAKAGE - model is seeing future data!"
        )

    # Check win rate
    win_rate = results.get("overall_win_rate", results.get("win_rate", results.get("bet_win_rate", 0)))
    if win_rate is not None and win_rate > SANITY_LIMITS["max_win_rate"]:
        violations.append(
            f"IMPOSSIBLE WIN RATE: {win_rate:.1f}% > {SANITY_LIMITS['max_win_rate']}% limit. "
            f"No legitimate model achieves >62% at -110 odds!"
        )

    # Check Sharpe ratio
    sharpe = results.get("sharpe_ratio", results.get("sharpe", 0))
    if sharpe is not None and sharpe > SANITY_LIMITS["max_sharpe"]:
        violations.append(
            f"IMPOSSIBLE SHARPE: {sharpe:.2f} > {SANITY_LIMITS['max_sharpe']} limit. "
            f"This exceeds top hedge funds - indicates data leakage!"
        )

    # Check ECE (suspiciously perfect calibration)
    ece = results.get("ece", results.get("metrics", {}).get("ece"))
    if ece is not None and ece < SANITY_LIMITS["min_ece"]:
        violations.append(
            f"SUSPICIOUSLY PERFECT ECE: {ece:.4f} < {SANITY_LIMITS['min_ece']} limit. "
            f"Calibration may be overfitting to validation data!"
        )

    is_sane = len(violations) == 0

    if not is_sane:
        error_msg = (
            f"\n{'='*70}\n"
            f"  DATA LEAKAGE DETECTED in {context}!\n"
            f"{'='*70}\n"
            f"  The following metrics are IMPOSSIBLE for legitimate betting models:\n\n"
        )
        for v in violations:
            error_msg += f"  - {v}\n"
        error_msg += (
            f"\n  REQUIRED ACTIONS:\n"
            f"  1. Check train/test data separation - ensure NO overlap\n"
            f"  2. Verify features only use data BEFORE game date\n"
            f"  3. Ensure calibration uses held-out data only\n"
            f"  4. Check for schedule lookahead features\n"
            f"{'='*70}\n"
        )

        print(error_msg, flush=True)

        if raise_on_failure:
            raise DataLeakageError(error_msg)

    return is_sane, violations


def check_improvement_thresholds(
    results: Dict,
    bet_type: str,
    check_leakage: bool = True
) -> Tuple[bool, List[str]]:
    """
    Check if backtest results meet promotion thresholds.

    IMPORTANT: This function now checks for DATA LEAKAGE first.
    If metrics are impossibly good, it will raise DataLeakageError.

    Args:
        results: Backtest results dictionary with metrics
        bet_type: Type of bet ("moneyline", "spread", "prop")
        check_leakage: If True, run sanity checks first (default True)

    Returns:
        Tuple of (passed: bool, failures: List[str])

    Raises:
        DataLeakageError: If sanity limits are exceeded (data leakage detected)
    """
    # CRITICAL: Check for data leakage FIRST
    if check_leakage:
        check_sanity_limits(results, context=f"{bet_type} backtest", raise_on_failure=True)

    thresholds = BACKTEST_THRESHOLDS.get(bet_type, BACKTEST_THRESHOLDS["moneyline"])
    failures = []

    # Check ROI
    roi = results.get("overall_roi", results.get("roi", -100))
    if roi < thresholds["min_roi"]:
        failures.append(f"ROI {roi:.2f}% < {thresholds['min_roi']}%")

    # Check win rate
    win_rate = results.get("overall_win_rate", results.get("win_rate", 0))
    if win_rate < thresholds["min_win_rate"]:
        failures.append(f"Win rate {win_rate:.1f}% < {thresholds['min_win_rate']}%")

    # Check ECE (if available)
    ece = results.get("ece", results.get("metrics", {}).get("ece"))
    if ece is not None and ece > thresholds["max_ece"]:
        failures.append(f"ECE {ece:.3f} > {thresholds['max_ece']}")

    # Check Sharpe ratio
    sharpe = results.get("sharpe_ratio", results.get("sharpe", 0))
    if sharpe < thresholds["min_sharpe"]:
        failures.append(f"Sharpe {sharpe:.2f} < {thresholds['min_sharpe']}")

    return len(failures) == 0, failures


class BacktestReporter:
    """
    Run walk-forward backtests and save JSON reports.

    Integrates with ModelBacktester from backtesting.py to evaluate model
    performance using realistic betting simulation with walk-forward validation.

    Usage:
        reporter = BacktestReporter()
        results = reporter.run_moneyline_backtest(model, games_data)
        report_path = reporter.save_report(results, "moneyline", "moneyline")
        passed, failures = check_improvement_thresholds(results, "moneyline")
    """

    def __init__(self, output_dir: str = "backtest_results"):
        """Initialize BacktestReporter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_moneyline_backtest(
        self,
        model,
        games_data: List[Dict],
        initial_bankroll: float = 10000.0,
        min_edge: float = 0.02,
        holdout_fraction: float = 0.2,
    ) -> Dict:
        """
        Run backtest for moneyline model on HOLDOUT data only.

        CRITICAL FIX: Previous version tested on ALL data including training data,
        causing massive overfitting (20%+ ROI that isn't real).

        This version only evaluates on the last holdout_fraction of data,
        which mirrors the train/test split used during model training.

        Args:
            model: Trained moneyline model with predict() method
            games_data: Historical games with outcomes
            initial_bankroll: Starting bankroll for simulation
            min_edge: Minimum edge to place bet
            holdout_fraction: Fraction of data to use for testing (should match training split)

        Returns:
            Backtest results dictionary
        """
        try:
            from backtesting import ModelBacktester, BetType
        except ImportError:
            return {"error": "backtesting module not found"}

        backtester = ModelBacktester(
            model_name="moneyline_backtest",
            initial_bankroll=initial_bankroll,
            staking_strategy="kelly",
            min_edge=min_edge,
        )

        # Define predict function for backtester
        def model_predict_fn(features):
            """Predict home/away probabilities from features."""
            try:
                prediction = model.predict(features)
                home_prob = prediction.get("home_win_probability", 0.5)
                return home_prob, 1 - home_prob
            except Exception:
                return 0.5, 0.5

        # Format ALL games for sorting
        all_formatted_games = []
        skipped_no_date = 0
        for game in games_data:
            features = game.get("moneyline_features", {})
            if not features:
                continue

            # CRITICAL FIX: Require valid game_date, never fall back to datetime.now()
            # Using datetime.now() would leak future information into historical backtesting
            game_date_raw = game.get("game_date")
            if not game_date_raw:
                skipped_no_date += 1
                continue

            try:
                if isinstance(game_date_raw, str):
                    game_date = datetime.strptime(game_date_raw, "%Y-%m-%d")
                elif isinstance(game_date_raw, datetime):
                    game_date = game_date_raw
                else:
                    skipped_no_date += 1
                    continue
            except (ValueError, TypeError):
                skipped_no_date += 1
                continue

            formatted_game = {
                "game_id": f"{game.get('game_date', '')}_{game.get('home_team', '')}_{game.get('away_team', '')}",
                "date": game_date,
                "home_team": game.get("home_team", "HOM"),
                "away_team": game.get("away_team", "AWY"),
                "home_odds": game.get("home_odds", -110),
                "away_odds": game.get("away_odds", -110),
                "home_score": game.get("home_score", 0),
                "away_score": game.get("away_score", 0),
                "features": features,
            }
            all_formatted_games.append(formatted_game)

        if skipped_no_date > 0:
            print(f"  WARNING: Skipped {skipped_no_date} games with missing/invalid dates")

        if len(all_formatted_games) < 50:
            return {"error": f"Not enough games for backtest ({len(all_formatted_games)} < 50)"}

        # Sort by date to ensure chronological order
        all_formatted_games.sort(key=lambda x: x["date"])

        # CRITICAL: Only use HOLDOUT data (last portion not seen during training)
        n_total = len(all_formatted_games)
        n_holdout = int(n_total * holdout_fraction)
        holdout_games = all_formatted_games[-n_holdout:]

        print(f"  HOLDOUT BACKTEST: Testing on {n_holdout} games (last {holdout_fraction:.0%})")
        print(f"  Holdout period: {holdout_games[0]['date'].strftime('%Y-%m-%d')} to {holdout_games[-1]['date'].strftime('%Y-%m-%d')}")

        # Run backtest on holdout only
        result = backtester.backtest_moneyline(holdout_games, model_predict_fn)

        # Convert to dict with additional metrics
        result_dict = result.to_dict()
        result_dict["games_tested"] = len(holdout_games)
        result_dict["total_games"] = n_total
        result_dict["holdout_fraction"] = holdout_fraction
        result_dict["holdout_start_date"] = holdout_games[0]["date"].strftime("%Y-%m-%d")
        result_dict["holdout_end_date"] = holdout_games[-1]["date"].strftime("%Y-%m-%d")

        return result_dict

    def run_spread_backtest(
        self,
        model,
        games_data: List[Dict],
        initial_bankroll: float = 10000.0,
        min_edge: float = 0.02,
        holdout_fraction: float = 0.2,
    ) -> Dict:
        """
        Run backtest for spread model on HOLDOUT data only.

        CRITICAL FIX: Only evaluates on holdout data that wasn't seen during training.

        Args:
            model: Trained spread model with predict() method
            games_data: Historical games with outcomes and spread lines
            initial_bankroll: Starting bankroll for simulation
            min_edge: Minimum edge to place bet
            holdout_fraction: Fraction of data to use for testing

        Returns:
            Backtest results dictionary
        """
        try:
            from backtesting import ModelBacktester, BetType
        except ImportError:
            return {"error": "backtesting module not found"}

        backtester = ModelBacktester(
            model_name="spread_backtest",
            initial_bankroll=initial_bankroll,
            staking_strategy="kelly",
            min_edge=min_edge,
        )

        # Define predict function for backtester
        def model_predict_fn(features, spread_line):
            """Predict spread cover probability from features and spread_line."""
            try:
                # SpreadCoverClassifier.predict() requires both features and spread_line
                prediction = model.predict(features, spread_line)
                cover_prob = prediction.get("home_cover_probability", 0.5)
                # Return (predicted_spread, cover_prob) - spread is the line itself for classifiers
                return spread_line, cover_prob
            except Exception as e:
                print(f"  Warning: Spread prediction failed: {e}")
                return spread_line, 0.5

        # Format ALL games for sorting
        all_formatted_games = []
        skipped_no_date = 0
        for game in games_data:
            features = game.get("spread_features", game.get("moneyline_features", {}))
            if not features:
                continue

            # CRITICAL FIX: Require valid game_date, never fall back to datetime.now()
            game_date_raw = game.get("game_date")
            if not game_date_raw:
                skipped_no_date += 1
                continue

            try:
                if isinstance(game_date_raw, str):
                    game_date = datetime.strptime(game_date_raw, "%Y-%m-%d")
                elif isinstance(game_date_raw, datetime):
                    game_date = game_date_raw
                else:
                    skipped_no_date += 1
                    continue
            except (ValueError, TypeError):
                skipped_no_date += 1
                continue

            formatted_game = {
                "game_id": f"{game.get('game_date', '')}_{game.get('home_team', '')}_{game.get('away_team', '')}",
                "date": game_date,
                "home_team": game.get("home_team", "HOM"),
                "away_team": game.get("away_team", "AWY"),
                "spread_line": game.get("spread_line", 0),
                "home_odds_spread": game.get("home_odds_spread", -110),
                "home_score": game.get("home_score", 0),
                "away_score": game.get("away_score", 0),
                "features": features,
            }
            all_formatted_games.append(formatted_game)

        if skipped_no_date > 0:
            print(f"  WARNING: Skipped {skipped_no_date} games with missing/invalid dates")

        if len(all_formatted_games) < 50:
            return {"error": f"Not enough games for backtest ({len(all_formatted_games)} < 50)"}

        # Sort by date
        all_formatted_games.sort(key=lambda x: x["date"])

        # CRITICAL: Only use HOLDOUT data
        n_total = len(all_formatted_games)
        n_holdout = int(n_total * holdout_fraction)
        holdout_games = all_formatted_games[-n_holdout:]

        print(f"  HOLDOUT BACKTEST: Testing on {n_holdout} games (last {holdout_fraction:.0%})")
        print(f"  Holdout period: {holdout_games[0]['date'].strftime('%Y-%m-%d')} to {holdout_games[-1]['date'].strftime('%Y-%m-%d')}")

        # Run backtest on holdout only
        result = backtester.backtest_spread(holdout_games, model_predict_fn)

        # Convert to dict
        result_dict = result.to_dict()
        result_dict["games_tested"] = len(holdout_games)
        result_dict["total_games"] = n_total
        result_dict["holdout_fraction"] = holdout_fraction

        return result_dict

    def save_report(
        self,
        results: Dict,
        model_name: str,
        bet_type: str,
    ) -> Path:
        """
        Save backtest report to JSON file.

        Args:
            results: Backtest results dictionary
            model_name: Name of the model
            bet_type: Type of bet (moneyline, spread, prop)

        Returns:
            Path to saved report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{bet_type}_{timestamp}.json"
        filepath = self.output_dir / filename

        # Add metadata
        report = {
            "model_name": model_name,
            "bet_type": bet_type,
            "timestamp": timestamp,
            "report_date": datetime.now().isoformat(),
            "thresholds": BACKTEST_THRESHOLDS.get(bet_type, {}),
            "results": results,
        }

        # Check thresholds
        passed, failures = check_improvement_thresholds(results, bet_type)
        report["thresholds_passed"] = passed
        report["threshold_failures"] = failures

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return filepath

    def print_summary(self, results: Dict, bet_type: str) -> None:
        """Print backtest summary to console."""
        print(f"\n  {'='*50}")
        print(f"  BACKTEST RESULTS: {bet_type.upper()}")
        print(f"  {'='*50}")

        if "error" in results:
            print(f"  Error: {results['error']}")
            return

        print(f"  Total Bets: {results.get('total_bets', 0)}")
        print(f"  Win Rate: {results.get('overall_win_rate', 0):.1f}%")
        print(f"  ROI: {results.get('overall_roi', 0):+.2f}%")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {results.get('max_drawdown_pct', 0):.1f}%")

        # Check thresholds
        passed, failures = check_improvement_thresholds(results, bet_type)
        if passed:
            print(f"\n  ✓ Model PASSES promotion thresholds")
        else:
            print(f"\n  ✗ Model FAILS promotion thresholds:")
            for failure in failures:
                print(f"    - {failure}")
        print(f"  {'='*50}")


class TrainingMetricsLogger:
    """
    Logs and saves training metrics for model evaluation and tracking.

    Saves comprehensive metrics with timestamps to training_metrics/ directory,
    including:
    - Accuracy, precision, recall, F1 for classifiers
    - RMSE, MAE, R² for regressors
    - Brier score, log loss, ECE for calibrated probabilities
    - Betting ROI simulation results

    Usage:
        logger = TrainingMetricsLogger("moneyline")
        logger.log_classification_metrics(y_true, y_pred, y_prob)
        logger.log_calibration_metrics(y_prob, y_true)
        logger.log_betting_roi(predictions, outcomes, odds)
        logger.save()
    """

    def __init__(self, model_name: str, model_type: str = "classifier"):
        """
        Initialize metrics logger.

        Args:
            model_name: Name of the model (e.g., "moneyline", "spread", "prop_points")
            model_type: Type of model ("classifier" or "regressor")
        """
        self.model_name = model_name
        self.model_type = model_type
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {
            "model_name": model_name,
            "model_type": model_type,
            "timestamp": self.timestamp,
            "training_date": datetime.now().isoformat(),
        }

    def log_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict:
        """Log classification metrics."""
        self.metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        self.metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        self.metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        self.metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        if y_prob is not None:
            from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
            self.metrics["log_loss"] = float(log_loss(y_true, y_prob))
            self.metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
            try:
                self.metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                self.metrics["auc_roc"] = None

        return self.metrics

    def log_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Log regression metrics."""
        self.metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        self.metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        self.metrics["r2"] = float(r2_score(y_true, y_pred))
        self.metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        return self.metrics

    def log_calibration_metrics(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Log calibration metrics (ECE, MCE).

        Args:
            y_prob: Predicted probabilities
            y_true: True binary labels
            n_bins: Number of bins for calibration
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        # Expected Calibration Error
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(y_true[mask])
                bin_confidence = np.mean(y_prob[mask])
                bin_weight = np.sum(mask) / len(y_prob)
                ece += np.abs(bin_accuracy - bin_confidence) * bin_weight
                mce = max(mce, np.abs(bin_accuracy - bin_confidence))

        self.metrics["ece"] = float(ece)
        self.metrics["mce"] = float(mce)
        return self.metrics

    def log_betting_roi(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        odds: np.ndarray = None,
        min_edge: float = 0.03,
        kelly_fraction: float = 0.25
    ) -> Dict:
        """
        Simulate betting ROI with Kelly criterion.

        Args:
            predicted_probs: Model's predicted probabilities
            actual_outcomes: Actual binary outcomes
            odds: American odds for each bet (default -110)
            min_edge: Minimum edge to place bet
            kelly_fraction: Fractional Kelly for bet sizing
        """
        predicted_probs = np.asarray(predicted_probs).flatten()
        actual_outcomes = np.asarray(actual_outcomes).flatten()

        if odds is None:
            odds = np.full_like(predicted_probs, -110.0)

        # Convert American odds to decimal
        def american_to_decimal(american: float) -> float:
            if american > 0:
                return (american / 100) + 1
            else:
                return (100 / abs(american)) + 1

        # Simulate betting
        bankroll = 1000.0
        initial_bankroll = bankroll
        bets_placed = 0
        bets_won = 0

        for i, (prob, outcome, odd) in enumerate(zip(predicted_probs, actual_outcomes, odds)):
            # Calculate implied probability from odds
            decimal_odd = american_to_decimal(odd)
            implied_prob = 1 / decimal_odd

            # Calculate edge
            edge = prob - implied_prob

            if edge >= min_edge and prob >= 0.52:
                # Kelly bet sizing
                kelly = (prob * (decimal_odd - 1) - (1 - prob)) / (decimal_odd - 1)
                bet_size = max(0, min(kelly * kelly_fraction, 0.05)) * bankroll

                if bet_size > 0:
                    bets_placed += 1
                    if outcome == 1:
                        bets_won += 1
                        bankroll += bet_size * (decimal_odd - 1)
                    else:
                        bankroll -= bet_size

        roi = (bankroll - initial_bankroll) / initial_bankroll * 100
        win_rate = bets_won / bets_placed if bets_placed > 0 else 0

        self.metrics["betting_roi_pct"] = float(roi)
        self.metrics["bets_placed"] = int(bets_placed)
        self.metrics["bets_won"] = int(bets_won)
        self.metrics["bet_win_rate"] = float(win_rate)
        self.metrics["final_bankroll"] = float(bankroll)
        self.metrics["min_edge_threshold"] = float(min_edge)

        return self.metrics

    def log_time_series_split(
        self,
        n_splits: int,
        train_sizes: List[int],
        test_sizes: List[int],
        fold_metrics: List[Dict]
    ) -> Dict:
        """Log time-series cross-validation results."""
        self.metrics["cv_n_splits"] = n_splits
        self.metrics["cv_train_sizes"] = train_sizes
        self.metrics["cv_test_sizes"] = test_sizes
        self.metrics["cv_fold_metrics"] = fold_metrics

        # Calculate mean and std across folds
        for key in fold_metrics[0].keys():
            values = [f[key] for f in fold_metrics if key in f and f[key] is not None]
            if values:
                self.metrics[f"cv_{key}_mean"] = float(np.mean(values))
                self.metrics[f"cv_{key}_std"] = float(np.std(values))

        return self.metrics

    def add_custom_metric(self, name: str, value: Any) -> None:
        """Add a custom metric."""
        self.metrics[name] = value

    def log_clv_metrics(
        self,
        bet_odds: List[float],
        closing_odds: List[float],
        outcomes: List[int] = None
    ) -> Dict:
        """
        Log Closing Line Value metrics.

        CLV is the most reliable predictor of long-term betting edge.
        Sharp bettors consistently beat the closing line.

        Args:
            bet_odds: American odds at time of bet
            closing_odds: Closing odds before game start
            outcomes: Optional binary outcomes (1=win, 0=loss)
        """
        try:
            from feature_engineering import calculate_clv_metrics
            clv_metrics = calculate_clv_metrics(bet_odds, closing_odds, outcomes)
            self.metrics["clv"] = clv_metrics
            return clv_metrics
        except ImportError:
            # Calculate CLV inline if feature_engineering not available
            def american_to_prob(odds: float) -> float:
                if odds > 0:
                    return 100 / (odds + 100)
                else:
                    return abs(odds) / (abs(odds) + 100)

            clv_values = []
            for bet, closing in zip(bet_odds, closing_odds):
                bet_prob = american_to_prob(bet)
                closing_prob = american_to_prob(closing)
                clv = (closing_prob - bet_prob) * 100
                clv_values.append(clv)

            clv_array = np.array(clv_values)
            self.metrics["clv"] = {
                "avg_clv_pct": float(np.mean(clv_array)),
                "positive_clv_rate": float(np.mean(clv_array > 0)),
                "clv_roi_estimate": float(np.mean(clv_array) * 1.05),
                "total_bets": len(clv_array),
            }
            return self.metrics["clv"]

    def save(self, directory: Path = None) -> Path:
        """
        Save metrics to JSON file.

        IMPORTANT: This method now checks for impossible training metrics
        that indicate data leakage. Training metrics are expected to be
        optimistic, but extreme values (>50% ROI) indicate a bug.

        Args:
            directory: Directory to save to (default: training_metrics/)

        Returns:
            Path to saved file
        """
        if directory is None:
            directory = METRICS_DIR

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # CRITICAL: Check for impossible training metrics (data leakage indicator)
        training_roi = self.metrics.get("betting_roi_pct", 0)
        if training_roi > SANITY_LIMITS["max_training_roi"]:
            print(
                f"\n  ⚠️  WARNING: IMPOSSIBLE TRAINING ROI DETECTED!\n"
                f"      ROI: {training_roi:.1f}% > {SANITY_LIMITS['max_training_roi']}% limit\n"
                f"      This indicates the model is being evaluated on TRAINING data,\n"
                f"      not held-out test data. This is DATA LEAKAGE!\n"
                f"      Action: Ensure log_betting_roi() uses TEST data only.\n",
                flush=True
            )
            # Add warning flag to saved metrics
            self.metrics["_warning_possible_data_leakage"] = True
            self.metrics["_warning_message"] = (
                f"Training ROI of {training_roi:.1f}% exceeds sanity limit of "
                f"{SANITY_LIMITS['max_training_roi']}%. This indicates data leakage."
            )

        filename = f"{self.model_name}_{self.timestamp}.json"
        filepath = directory / filename

        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

        print(f"  Metrics saved to {filepath}")
        return filepath

    def get_summary(self) -> str:
        """Get a formatted summary of key metrics."""
        lines = [f"\n  {self.model_name} Training Metrics:"]

        if self.model_type == "classifier":
            if "accuracy" in self.metrics:
                lines.append(f"    Accuracy: {self.metrics['accuracy']:.2%}")
            if "brier_score" in self.metrics:
                lines.append(f"    Brier Score: {self.metrics['brier_score']:.4f}")
            if "auc_roc" in self.metrics and self.metrics["auc_roc"]:
                lines.append(f"    AUC-ROC: {self.metrics['auc_roc']:.4f}")
            if "ece" in self.metrics:
                lines.append(f"    ECE: {self.metrics['ece']:.4f}")
        else:
            if "rmse" in self.metrics:
                lines.append(f"    RMSE: {self.metrics['rmse']:.4f}")
            if "mae" in self.metrics:
                lines.append(f"    MAE: {self.metrics['mae']:.4f}")
            if "r2" in self.metrics:
                lines.append(f"    R²: {self.metrics['r2']:.4f}")

        if "betting_roi_pct" in self.metrics:
            lines.append(f"    Betting ROI: {self.metrics['betting_roi_pct']:+.2f}%")
            lines.append(f"    Bets: {self.metrics['bets_placed']} ({self.metrics['bet_win_rate']:.1%} win rate)")

        return "\n".join(lines)


class BaseModelTrainer:
    """Base class for model trainers with common functionality."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_metrics = {}

    def preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Preprocess features with scaling.

        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler

        Returns:
            Scaled numpy array
        """
        # Store feature names
        if fit:
            self.feature_names = list(X.columns)

        # Handle missing values with smart defaults (not zeros)
        X_clean = smart_fillna_features(X)

        # Scale features
        if fit:
            return self.scaler.fit_transform(X_clean)
        return self.scaler.transform(X_clean)

    def save_model(self, filepath: Optional[Path] = None):
        """Save model, scaler, and metadata to disk."""
        if filepath is None:
            filepath = MODEL_DIR / f"{self.model_name}.pkl"

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "model_name": self.model_name,
            "saved_at": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath: Optional[Path] = None):
        """Load model, scaler, and metadata from disk."""
        if filepath is None:
            filepath = MODEL_DIR / f"{self.model_name}.pkl"

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.training_metrics = model_data["training_metrics"]
        self.is_fitted = True

        print(f"Model loaded from {filepath}")
        return self

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        if not self.is_fitted:
            return {}

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_).flatten()
        else:
            return {}

        return dict(zip(self.feature_names, importance))


class PropModelWrapper:
    """
    Wrapper class for player prop models trained with train_complete_balldontlie.py.
    Compatible with the existing app.py model loading system.
    """

    def __init__(self, model=None, scaler=None, feature_names=None, training_metrics=None, prop_type="points"):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.training_metrics = training_metrics or {}
        self.prop_type = prop_type
        self.is_fitted = True
        self.model_name = f"player_{prop_type}"

    def predict(self, features: Dict, prop_line: float = None) -> Dict:
        """Make a prediction - compatible with app.py interface."""
        import numpy as np
        import pandas as pd

        X = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)

        predicted = float(self.model.predict(X_scaled)[0])

        result = {
            "predicted_value": predicted,
            "prop_type": self.prop_type,
        }

        if prop_line is not None:
            result["prop_line"] = prop_line
            result["prediction"] = "over" if predicted > prop_line else "under"
            result["edge"] = predicted - prop_line
            result["confidence"] = abs(predicted - prop_line) / max(prop_line, 1)

        return result


class EnsembleMoneylineWrapper:
    """
    Wrapper class that makes ensemble models compatible with
    the existing app.py model loading system.

    This class mimics the interface of BaseModelTrainer.
    """

    def __init__(self, models=None, model_weights=None, scaler=None, feature_names=None, training_metrics=None):
        self.models = models or {}
        self.model_weights = model_weights or {}
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.training_metrics = training_metrics or {}
        self.is_fitted = True
        self.model_name = "moneyline_ensemble"

    def predict(self, features: Dict) -> Dict:
        """Make a prediction - compatible with app.py interface."""
        import numpy as np
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

        home_prob = float(np.clip(probs[0, 1], 0.0, 1.0))
        away_prob = float(np.clip(probs[0, 0], 0.0, 1.0))

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(max(home_prob, away_prob)),
        }


class MoneylineModel(BaseModelTrainer):
    """
    Logistic Regression model for moneyline predictions.

    Predicts probability of home team winning.
    """

    def __init__(self):
        super().__init__("moneyline_logistic_regression")
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )

    def prepare_training_data(self, games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from historical games.

        Args:
            games_data: List of game dictionaries with features and outcomes

        Returns:
            Tuple of (features DataFrame, labels array) - SORTED CHRONOLOGICALLY
        """
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("moneyline_features", {})
            outcome = game.get("home_win", None)

            if features and outcome is not None:
                # Remove non-numeric and identifier fields
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
                }
                features_list.append(numeric_features)
                labels.append(1 if outcome else 0)
                game_dates.append(game.get("game_date", "1900-01-01"))

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        # TimeSeriesSplit REQUIRES chronological ordering to work correctly
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_time_series_cv: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the moneyline prediction model.

        Args:
            X: Feature DataFrame (MUST be sorted by date for time-series CV)
            y: Target labels (1 for home win, 0 for away win)
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            use_time_series_cv: Use time-series walk-forward validation (recommended)

        Returns:
            Dictionary with training metrics
        """
        if use_time_series_cv:
            # TIME-SERIES WALK-FORWARD VALIDATION
            # This is CRITICAL for sports betting - prevents look-ahead bias
            # Data must be sorted chronologically (oldest first)
            n_samples = len(X)
            test_samples = int(n_samples * test_size)

            # Use last test_size% as held-out test set
            X_train = X.iloc[:-test_samples]
            X_test = X.iloc[-test_samples:]
            y_train = y[:-test_samples]
            y_test = y[-test_samples:]

            print(f"\n  Using TIME-SERIES validation (walk-forward)")
            print(f"  Train: games 0-{len(X_train)-1}, Test: games {len(X_train)}-{n_samples-1}")
        else:
            # Legacy random split (NOT recommended for time-series data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            print(f"\n  Using RANDOM split (not recommended for time-series)")

        # Preprocess
        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        # Cross-validation with TimeSeriesSplit for proper evaluation
        if use_time_series_cv:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=tscv)
        else:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)

        # Train on full training set
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Predictions on held-out test set
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        self.training_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "validation_type": "time_series" if use_time_series_cv else "random",
        }

        print(f"\nMoneyline Model Training Results:")
        print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
        print(f"  Precision: {self.training_metrics['precision']:.4f}")
        print(f"  Recall: {self.training_metrics['recall']:.4f}")
        print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        print(f"  CV Score: {self.training_metrics['cv_mean']:.4f} (+/- {self.training_metrics['cv_std']:.4f})")

        return self.training_metrics

    def predict(self, features: Dict) -> Dict[str, float]:
        """
        Predict home team win probability.

        Args:
            features: Moneyline features dictionary

        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        # Prepare features
        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])

        # Ensure all expected features are present
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        # Scale and predict
        X_scaled = self.preprocess_features(X, fit=False)
        prob = self.model.predict_proba(X_scaled)[0]

        # CRITICAL: Ensure probabilities are valid (0.0 to 1.0)
        home_prob = float(np.clip(prob[1], 0.0, 1.0))
        away_prob = float(np.clip(prob[0], 0.0, 1.0))

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(np.clip(max(prob), 0.0, 1.0)),
        }


class SpreadModel(BaseModelTrainer):
    """
    Support Vector Machine model for spread predictions.

    Predicts point differential (positive = home team wins by that margin).
    """

    def __init__(self, use_classifier: bool = False):
        model_name = "spread_svm_classifier" if use_classifier else "spread_svm_regressor"
        super().__init__(model_name)
        self.use_classifier = use_classifier

        if use_classifier:
            # SVC for classifying if spread is covered
            self.model = SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            )
        else:
            # SVR for predicting actual point spread
            self.model = SVR(
                kernel="rbf",
                C=1.0,
                epsilon=0.1,
            )

    def prepare_training_data(
        self,
        games_data: List[Dict],
        spread_line: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from historical games.

        Args:
            games_data: List of game dictionaries with features and outcomes
            spread_line: If using classifier, the spread line to evaluate

        Returns:
            Tuple of (features DataFrame, labels array) - SORTED CHRONOLOGICALLY
        """
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("spread_features", {})
            actual_diff = game.get("point_differential", None)  # home - away

            if features and actual_diff is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in [
                        "home_team_id", "away_team_id", "injury_details"
                    ]
                }
                features_list.append(numeric_features)
                game_dates.append(game.get("game_date", "1900-01-01"))

                if self.use_classifier and spread_line is not None:
                    # 1 if home covers spread, 0 otherwise
                    labels.append(1 if actual_diff > spread_line else 0)
                else:
                    labels.append(actual_diff)

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_time_series_cv: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the spread prediction model.

        Args:
            X: Feature DataFrame (MUST be sorted by date for time-series CV)
            y: Target values (point diff or cover labels)
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            use_time_series_cv: Use time-series walk-forward validation (recommended)

        Returns:
            Dictionary with training metrics
        """
        # Split data using time-series or random approach
        if use_time_series_cv:
            # TIME-SERIES WALK-FORWARD VALIDATION
            n_samples = len(X)
            test_samples = int(n_samples * test_size)
            X_train = X.iloc[:-test_samples]
            X_test = X.iloc[-test_samples:]
            y_train = y[:-test_samples]
            y_test = y[-test_samples:]
            print(f"\n  Using TIME-SERIES validation (walk-forward)")
        elif self.use_classifier:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        # Preprocess
        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        # Cross-validation with TimeSeriesSplit or standard K-fold
        if use_time_series_cv:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=tscv)
        else:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)

        # Train
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Predictions
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        if self.use_classifier:
            self.training_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "validation_type": "time_series" if use_time_series_cv else "random",
            }
            print(f"\nSpread Classifier Training Results:")
            print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
            print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        else:
            self.training_metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "validation_type": "time_series" if use_time_series_cv else "random",
            }
            print(f"\nSpread Regressor Training Results:")
            print(f"  RMSE: {self.training_metrics['rmse']:.2f} points")
            print(f"  MAE: {self.training_metrics['mae']:.2f} points")
            print(f"  R2: {self.training_metrics['r2']:.4f}")

        print(f"  CV Score: {self.training_metrics['cv_mean']:.4f} (+/- {self.training_metrics['cv_std']:.4f})")

        return self.training_metrics

    def predict(self, features: Dict, spread_line: Optional[float] = None) -> Dict[str, Any]:
        """
        Predict spread outcome.

        Args:
            features: Spread features dictionary
            spread_line: The betting line to evaluate

        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in [
                "home_team_id", "away_team_id", "injury_details"
            ]
        }

        X = pd.DataFrame([numeric_features])

        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)

        if self.use_classifier:
            prob = self.model.predict_proba(X_scaled)[0]
            return {
                "cover_probability": float(np.clip(prob[1], 0.0, 1.0)),
                "no_cover_probability": float(np.clip(prob[0], 0.0, 1.0)),
                "prediction": "cover" if prob[1] > 0.5 else "no_cover",
                "confidence": float(np.clip(max(prob), 0.0, 1.0)),
            }
        else:
            predicted_diff = self.model.predict(X_scaled)[0]

            # CRITICAL FIX: Clip spread prediction to realistic NBA range
            # NBA games are never decided by more than ~50 points, typical range is -20 to +20
            predicted_diff = float(np.clip(predicted_diff, -30.0, 30.0))

            result = {
                "predicted_spread": predicted_diff,
                "predicted_winner": "home" if predicted_diff > 0 else "away",
                "predicted_margin": abs(predicted_diff),
            }

            if spread_line is not None:
                result["covers_spread"] = predicted_diff > spread_line
                result["spread_line"] = spread_line
                # Edge is capped to realistic values (-20 to +20)
                result["edge"] = float(np.clip(predicted_diff - spread_line, -20.0, 20.0))

            return result


class SpreadCoverClassifier(BaseModelTrainer):
    """
    Line-aware spread cover classifier that outputs P(Home Covers).

    CRITICAL FIX: The previous SpreadModel used SVR to predict point differential,
    but the backtester expected 'home_cover_probability'. This resulted in 0 bets.

    This classifier:
    1. Takes spread_line as an INPUT FEATURE (line-aware)
    2. Outputs P(home_covers) directly as a probability
    3. Uses XGBoost with regularization to prevent overfitting
    4. Conservative settings for 3-5% ROI target

    The key insight is that the spread line contains valuable market information
    and should be used as a feature, not just for evaluation.
    """

    def __init__(self):
        super().__init__("spread_cover_classifier")

        # Use XGBoost with conservative hyperparameters
        # Heavy regularization to prevent overfitting
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=4,              # Shallow trees
                learning_rate=0.03,       # Slow learning
                min_child_weight=30,      # High regularization
                subsample=0.7,            # Row sampling
                colsample_bytree=0.7,     # Column sampling
                reg_alpha=1.0,            # L1 regularization
                reg_lambda=5.0,           # L2 regularization
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
            )
            self._has_xgb = True
        except ImportError:
            # Fallback to GradientBoosting
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=20,
                random_state=42,
            )
            self._has_xgb = False

    def prepare_training_data(
        self,
        games_data: List[Dict],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data with spread_line as a feature.

        The spread line is included as a feature because it contains
        valuable market information about the expected margin.

        If spread_line is not available in training data, we estimate it from
        net_rating_diff: spread ≈ -net_rating_diff * 0.4 + 3 (home advantage)

        Returns:
            Tuple of (features DataFrame with spread_line, cover labels)
        """
        features_list = []
        labels = []
        game_dates = []

        for game in games_data:
            features = game.get("spread_features", {})
            actual_diff = game.get("point_differential", None)  # home - away
            spread_line = game.get("spread_line", None)

            if features and actual_diff is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in [
                        "home_team_id", "away_team_id", "injury_details"
                    ]
                }

                # If no market spread_line, estimate from net_rating_diff
                # Formula: spread ≈ -net_rating_diff * 0.4 + home_court_advantage
                # Negative spread means home is favored
                if spread_line is None:
                    net_rating_diff = features.get("net_rating_diff", 0)
                    # Home advantage typically ~3 points, net rating converts ~0.4 to spread
                    estimated_spread = -net_rating_diff * 0.4 - 3.0
                    spread_line = estimated_spread

                # CRITICAL: Add spread_line as a feature
                # This makes the model "line-aware"
                numeric_features['spread_line'] = spread_line

                features_list.append(numeric_features)
                game_dates.append(game.get("game_date", "1900-01-01"))

                # 1 if home covers spread (actual_diff > spread_line), 0 otherwise
                # Note: If spread is -5.5 and home wins by 6, they cover (6 > -5.5)
                labels.append(1 if actual_diff > spread_line else 0)

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # Sort chronologically for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Train the spread cover classifier with time-series validation.

        Returns metrics on held-out test data.
        """
        if len(X) == 0:
            return {"error": "No training data"}

        # Store feature names
        self.feature_names = list(X.columns)

        # Time-series split (use last 20% for testing)
        n_test = int(len(X) * test_size)
        X_train = X.iloc[:-n_test]
        y_train = y[:-n_test]
        X_test = X.iloc[-n_test:]
        y_test = y[-n_test:]

        # Fit scaler on training data only
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Evaluate on TEST data (honest metrics)
        y_prob_test = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = (y_prob_test > 0.5).astype(int)

        metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
            "test_precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "model_type": "spread_cover_classifier",
        }

        # Add AUC-ROC
        try:
            from sklearn.metrics import roc_auc_score
            metrics["test_auc_roc"] = float(roc_auc_score(y_test, y_prob_test))
        except ValueError:
            metrics["test_auc_roc"] = None

        print(f"  Spread Cover Classifier - Test Accuracy: {metrics['test_accuracy']:.1%}")
        print(f"  Spread Cover Classifier - Test AUC-ROC: {metrics.get('test_auc_roc', 'N/A')}")

        return metrics

    def predict(self, features: Dict, spread_line: float) -> Dict[str, Any]:
        """
        Predict probability that home team covers the spread.

        Args:
            features: Spread features dictionary
            spread_line: The betting line (e.g., -5.5 means home favored by 5.5)

        Returns:
            Dictionary with home_cover_probability for backtester compatibility
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in [
                "home_team_id", "away_team_id", "injury_details"
            ]
        }

        # Add spread_line as feature (line-aware prediction)
        numeric_features['spread_line'] = spread_line

        X = pd.DataFrame([numeric_features])

        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)
        prob = self.model.predict_proba(X_scaled)[0]

        # Clip probabilities to prevent extreme values
        home_cover_prob = float(np.clip(prob[1], 0.05, 0.95))

        return {
            "home_cover_probability": home_cover_prob,
            "away_cover_probability": 1.0 - home_cover_prob,
            "prediction": "home_covers" if home_cover_prob > 0.5 else "away_covers",
            "confidence": float(max(home_cover_prob, 1.0 - home_cover_prob)),
            "spread_line": spread_line,
        }

    def save_model(self, directory: Path = None) -> Path:
        """Save model to disk."""
        if directory is None:
            directory = MODEL_DIR
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        filepath = directory / f"{self.model_name}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "is_fitted": self.is_fitted,
            }, f)
        print(f"  Saved {self.model_name} to {filepath}")
        return filepath

    @classmethod
    def load_model(cls, filepath: Path = None) -> "SpreadCoverClassifier":
        """Load model from disk."""
        if filepath is None:
            filepath = MODEL_DIR / "spread_cover_classifier.pkl"
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.is_fitted = data["is_fitted"]
        return instance


class QuantilePropModel(BaseModelTrainer):
    """
    TIER 1 UPGRADE: Quantile Regression model for player prop predictions.

    Uses GradientBoostingRegressor with quantile loss to predict:
    - 0.45 quantile (lower bound)
    - 0.50 quantile (median)
    - 0.55 quantile (upper bound)

    This generates implied probabilities for Over/Under betting that are
    more accurate than simple mean prediction because it captures the
    asymmetric uncertainty around the prediction.
    """

    def __init__(self, prop_type: str = "points"):
        """
        Initialize quantile prop model.

        Args:
            prop_type: Type of prop ("points", "rebounds", "assists", "threes", "pra")
        """
        self.prop_type = prop_type
        model_name = f"player_{prop_type}_quantile"
        super().__init__(model_name)

        # Three quantile models for implied probability calculation
        self.quantile_models = {
            0.45: GradientBoostingRegressor(
                loss='quantile', alpha=0.45,
                n_estimators=100, max_depth=5,
                learning_rate=0.1, min_samples_split=10,
                random_state=42
            ),
            0.50: GradientBoostingRegressor(
                loss='quantile', alpha=0.50,
                n_estimators=100, max_depth=5,
                learning_rate=0.1, min_samples_split=10,
                random_state=42
            ),
            0.55: GradientBoostingRegressor(
                loss='quantile', alpha=0.55,
                n_estimators=100, max_depth=5,
                learning_rate=0.1, min_samples_split=10,
                random_state=42
            ),
        }
        self.model = self.quantile_models[0.50]  # Default median model for compatibility

    def prepare_training_data(
        self,
        player_data: List[Dict],
        prop_line: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data from historical player games."""
        prop_feature_map = {
            "points": "points_features",
            "rebounds": "rebounds_features",
            "assists": "assists_features",
            "threes": "threes_features",
            "pra": "pra_features",
        }
        stat_key_map = {
            "points": "pts",
            "rebounds": "reb",
            "assists": "ast",
            "threes": "fg3_made",
            "pra": "pra",
        }

        feature_key = prop_feature_map.get(self.prop_type, "points_features")
        stat_key = stat_key_map.get(self.prop_type, "pts")

        features_list = []
        labels = []
        game_dates = []

        for game in player_data:
            features = game.get(feature_key, {})
            actual_value = game.get("actual_stats", {}).get(stat_key, None)

            if self.prop_type == "pra" and actual_value is None:
                stats = game.get("actual_stats", {})
                pts = stats.get("pts", 0) or 0
                reb = stats.get("reb", 0) or 0
                ast = stats.get("ast", 0) or 0
                actual_value = pts + reb + ast

            if features and actual_value is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k != "player_id"
                }
                features_list.append(numeric_features)
                labels.append(actual_value)
                game_dates.append(game.get("game_date", "1900-01-01"))

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train all three quantile models."""
        # Time-series split
        n_samples = len(X)
        test_samples = int(n_samples * test_size)
        X_train = X.iloc[:-test_samples]
        X_test = X.iloc[-test_samples:]
        y_train = y[:-test_samples]
        y_test = y[-test_samples:]

        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        print(f"\n  Training Quantile Prop Model ({self.prop_type})...")
        print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Train all three quantile models
        predictions = {}
        for quantile, model in self.quantile_models.items():
            print(f"    Training quantile {quantile}...")
            model.fit(X_train_scaled, y_train)
            predictions[quantile] = model.predict(X_test_scaled)

        self.is_fitted = True

        # Calculate metrics using median prediction
        y_pred_median = predictions[0.50]

        # Check quantile crossing (lower should be <= median <= upper)
        crossings = np.sum(predictions[0.45] > predictions[0.55])

        self.training_metrics = {
            "mse": mean_squared_error(y_test, y_pred_median),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_median)),
            "mae": mean_absolute_error(y_test, y_pred_median),
            "r2": r2_score(y_test, y_pred_median),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "quantile_crossings": int(crossings),
            "quantiles_trained": list(self.quantile_models.keys()),
        }

        print(f"  Quantile {self.prop_type.title()} Model Results:")
        print(f"    RMSE: {self.training_metrics['rmse']:.2f}")
        print(f"    MAE: {self.training_metrics['mae']:.2f}")
        print(f"    R²: {self.training_metrics['r2']:.4f}")
        print(f"    Quantile crossings: {crossings} (should be 0)")

        return self.training_metrics

    def predict(self, features: Dict, prop_line: Optional[float] = None) -> Dict[str, Any]:
        """
        Predict with quantile-based implied probability.

        Uses the spread between quantiles to estimate Over/Under probability.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k != "player_id"
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        X_scaled = self.preprocess_features(X, fit=False)

        # Get predictions from all quantiles
        q45 = self.quantile_models[0.45].predict(X_scaled)[0]
        q50 = self.quantile_models[0.50].predict(X_scaled)[0]  # Median
        q55 = self.quantile_models[0.55].predict(X_scaled)[0]

        result = {
            "predicted_value": float(q50),  # Use median as main prediction
            "q45": float(q45),
            "q50": float(q50),
            "q55": float(q55),
            "prediction_spread": float(q55 - q45),  # Uncertainty width
            "prop_type": self.prop_type,
        }

        if prop_line is not None:
            result["prop_line"] = prop_line

            # Calculate implied probability using quantile positions
            # If line is below q45, strong over
            # If line is above q55, strong under
            # If line is between q45-q55, interpolate
            if prop_line <= q45:
                over_prob = 0.55 + 0.35 * (q45 - prop_line) / max(q50 - q45 + 1, 1)
            elif prop_line >= q55:
                over_prob = 0.45 - 0.35 * (prop_line - q55) / max(q55 - q50 + 1, 1)
            else:
                # Linear interpolation between q45 and q55
                range_width = q55 - q45
                if range_width > 0:
                    position = (prop_line - q45) / range_width
                    over_prob = 0.55 - 0.10 * position  # 0.55 at q45, 0.45 at q55
                else:
                    over_prob = 0.50

            # Clip to valid probability range
            over_prob = float(np.clip(over_prob, 0.05, 0.95))

            result["over_probability"] = over_prob
            result["under_probability"] = 1.0 - over_prob
            result["prediction"] = "over" if over_prob > 0.5 else "under"
            result["edge"] = q50 - prop_line
            result["confidence"] = abs(over_prob - 0.5) * 2  # 0 to 1 scale

        return result

    def save_model(self, filepath: Optional[Path] = None):
        """Save all quantile models, scaler, and metadata."""
        if filepath is None:
            filepath = MODEL_DIR / f"{self.model_name}.pkl"

        model_data = {
            "quantile_models": self.quantile_models,
            "model": self.quantile_models[0.50],  # For compatibility
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "model_name": self.model_name,
            "prop_type": self.prop_type,
            "saved_at": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Quantile model saved to {filepath}")
        return filepath

    def load_model(self, filepath: Optional[Path] = None):
        """Load quantile models from disk."""
        if filepath is None:
            filepath = MODEL_DIR / f"{self.model_name}.pkl"

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.quantile_models = model_data.get("quantile_models", {})
        self.model = model_data.get("model")
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.training_metrics = model_data["training_metrics"]
        self.is_fitted = True

        print(f"Quantile model loaded from {filepath}")
        return self


class LineAwarePropClassifier(BaseModelTrainer):
    """
    PRODUCTION UPGRADE: Line-aware prop classifier that predicts Over/Under probability.

    Unlike regression models that predict a value, this classifier takes the prop line
    as an INPUT FEATURE and directly outputs P(Over). This is more accurate for betting
    because:
    1. The line is known at prediction time and contains market information
    2. Different lines require different decision boundaries
    3. Outputs calibrated probabilities, not raw point predictions

    Training:
    - For each historical game, generates training samples at multiple prop lines
    - Labels are binary: 1 if actual > line, 0 if actual <= line
    - Line is included as a feature to learn the decision boundary

    Inference:
    - Given player features + prop line, outputs P(Over) directly
    - No need to convert predicted value to probability
    """

    def __init__(self, prop_type: str = "points"):
        """
        Initialize line-aware prop classifier.

        Args:
            prop_type: Type of prop ("points", "rebounds", "assists", "threes", "pra")
        """
        self.prop_type = prop_type
        model_name = f"player_{prop_type}_line_classifier"
        super().__init__(model_name)

        # Use gradient boosting for calibrated probabilities
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
            )
            self.use_xgboost = True
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_samples_leaf=10,
                random_state=42,
            )
            self.use_xgboost = False

        # For probability calibration
        self.calibrator = None
        self.line_stats = {}  # Store stats about training lines for validation

    def prepare_training_data(
        self,
        player_data: List[Dict],
        line_range: Tuple[float, float] = None,
        n_lines_per_game: int = 5,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data with prop line as a feature.

        For each game, generates multiple training samples at different prop lines.
        This teaches the model how the decision boundary changes with the line.

        Args:
            player_data: List of player game dictionaries
            line_range: (min, max) prop line range. If None, uses stat-specific defaults
            n_lines_per_game: Number of line samples per game

        Returns:
            Tuple of (features DataFrame with 'prop_line' column, binary labels)
        """
        stat_key_map = {
            "points": "pts",
            "rebounds": "reb",
            "assists": "ast",
            "threes": "fg3_made",
            "pra": "pra",
        }

        feature_key_map = {
            "points": "points_features",
            "rebounds": "rebounds_features",
            "assists": "assists_features",
            "threes": "threes_features",
            "pra": "pra_features",
        }

        # Default line ranges by prop type
        default_ranges = {
            "points": (5.5, 45.5),
            "rebounds": (2.5, 15.5),
            "assists": (1.5, 12.5),
            "threes": (0.5, 8.5),
            "pra": (10.5, 60.5),
        }

        stat_key = stat_key_map.get(self.prop_type, "pts")
        feature_key = feature_key_map.get(self.prop_type, "points_features")

        if line_range is None:
            line_range = default_ranges.get(self.prop_type, (5.5, 35.5))

        features_list = []
        labels = []
        game_dates = []

        for game in player_data:
            features = game.get(feature_key, {})
            actual_value = game.get("actual_stats", {}).get(stat_key)
            game_date = game.get("game_date", "1900-01-01")

            # Handle PRA calculation
            if self.prop_type == "pra" and actual_value is None:
                stats = game.get("actual_stats", {})
                pts = stats.get("pts", 0) or 0
                reb = stats.get("reb", 0) or 0
                ast = stats.get("ast", 0) or 0
                actual_value = pts + reb + ast

            if not features or actual_value is None:
                continue

            # Extract numeric features
            numeric_features = {
                k: v for k, v in features.items()
                if isinstance(v, (int, float)) and k != "player_id"
            }

            if not numeric_features:
                continue

            # Generate training samples at multiple lines around actual value
            # Focus lines around player's expected range for better learning
            player_avg = features.get(f'season_{stat_key}_avg', actual_value)
            if player_avg is None:
                player_avg = actual_value

            # Sample lines: some around the actual value, some around expected
            lines_to_sample = set()

            # Around actual value (±3 points for points, less for other stats)
            spread = 3.0 if self.prop_type == "points" else 1.5
            for offset in np.linspace(-spread, spread, 3):
                line = actual_value + offset + 0.5  # Standard half-point lines
                if line_range[0] <= line <= line_range[1]:
                    lines_to_sample.add(round(line * 2) / 2)  # Round to 0.5

            # Around player average
            for offset in np.linspace(-spread, spread, 3):
                line = player_avg + offset + 0.5
                if line_range[0] <= line <= line_range[1]:
                    lines_to_sample.add(round(line * 2) / 2)

            # Add some random lines in the range
            random_lines = np.random.uniform(line_range[0], line_range[1], n_lines_per_game)
            for line in random_lines:
                lines_to_sample.add(round(line * 2) / 2)

            # Create training sample for each line
            for prop_line in lines_to_sample:
                sample_features = numeric_features.copy()
                sample_features['prop_line'] = prop_line

                # Binary label: 1 if actual > line (over hit), 0 otherwise
                label = 1 if actual_value > prop_line else 0

                features_list.append(sample_features)
                labels.append(label)
                game_dates.append(game_date)

        # Create DataFrame and sort by date
        X = pd.DataFrame(features_list)
        y = np.array(labels)

        if game_dates:
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        # Store line statistics
        if 'prop_line' in X.columns:
            self.line_stats = {
                'min_line': X['prop_line'].min(),
                'max_line': X['prop_line'].max(),
                'mean_line': X['prop_line'].mean(),
                'n_samples': len(X),
                'over_rate': y.mean(),
            }

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        calibrate: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the line-aware classifier with temporal split.

        Args:
            X: Features including 'prop_line' column
            y: Binary labels (1=over, 0=under)
            test_size: Fraction for test set
            calibrate: Whether to apply isotonic calibration

        Returns:
            Training metrics dictionary
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        # Temporal split (data should be sorted by date)
        n_samples = len(X)
        split_idx = int(n_samples * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        # Preprocess
        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        print(f"\n  Training Line-Aware Prop Classifier ({self.prop_type})...")
        print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Over rate: {y.mean():.1%} (train: {y_train.mean():.1%}, test: {y_test.mean():.1%})")

        # Train base model
        self.model.fit(X_train_scaled, y_train)

        # Get probabilities
        y_prob_train = self.model.predict_proba(X_train_scaled)[:, 1]
        y_prob_test = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate pre-calibration metrics
        brier_uncalibrated = brier_score_loss(y_test, y_prob_test)

        # Apply calibration if requested
        if calibrate:
            try:
                from sklearn.isotonic import IsotonicRegression

                # Fit calibrator on training data
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(y_prob_train, y_train)

                # Calibrate test probabilities
                y_prob_test_cal = self.calibrator.predict(y_prob_test)
                brier_calibrated = brier_score_loss(y_test, y_prob_test_cal)

                # Only use calibrator if it improves Brier score
                if brier_calibrated < brier_uncalibrated:
                    print(f"  Calibration improved Brier: {brier_uncalibrated:.4f} -> {brier_calibrated:.4f}")
                    y_prob_final = y_prob_test_cal
                else:
                    print(f"  Calibration did not improve: {brier_uncalibrated:.4f} vs {brier_calibrated:.4f}")
                    self.calibrator = None
                    y_prob_final = y_prob_test
            except Exception as e:
                print(f"  Calibration failed: {e}")
                self.calibrator = None
                y_prob_final = y_prob_test
        else:
            y_prob_final = y_prob_test

        # Final predictions
        y_pred = (y_prob_final > 0.5).astype(int)

        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        try:
            auc = roc_auc_score(y_test, y_prob_final)
        except:
            auc = 0.5

        brier_final = brier_score_loss(y_test, y_prob_final)

        # Calculate ECE and MCE for calibration metrics
        from calibration import CalibrationEvaluator
        ece = CalibrationEvaluator.expected_calibration_error(y_prob_final, y_test)
        mce = CalibrationEvaluator.maximum_calibration_error(y_prob_final, y_test)

        # Store test data for external calibrator saving
        self.y_test_final = y_test
        self.y_prob_final = y_prob_final

        self.is_fitted = True
        self.training_metrics = {
            "accuracy": float(accuracy),
            "brier_score": float(brier_final),
            "auc_roc": float(auc),
            "ece": float(ece),
            "mce": float(mce),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "over_rate_test": float(y_test.mean()),
            "calibrated": self.calibrator is not None,
            "line_stats": self.line_stats,
        }

        print(f"  Line-Aware Classifier Results:")
        print(f"    Accuracy: {accuracy:.2%}")
        print(f"    Brier Score: {brier_final:.4f}")
        print(f"    ECE: {ece:.4f}")
        print(f"    AUC-ROC: {auc:.4f}")

        return self.training_metrics

    def predict(self, features: Dict, prop_line: float) -> Dict[str, Any]:
        """
        Predict Over probability for a given prop line.

        Args:
            features: Player/game features (without prop_line)
            prop_line: The betting line to evaluate

        Returns:
            Dictionary with over_probability, under_probability, prediction
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        # Add prop_line to features
        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k != "player_id"
        }
        numeric_features['prop_line'] = prop_line

        # Build feature array
        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)

        # Get probability
        prob_raw = self.model.predict_proba(X_scaled)[0, 1]

        # Apply calibration if available
        if self.calibrator is not None:
            over_prob = float(self.calibrator.predict([prob_raw])[0])
        else:
            over_prob = float(prob_raw)

        # Clip to valid range
        over_prob = np.clip(over_prob, 0.01, 0.99)

        return {
            "over_probability": over_prob,
            "under_probability": 1.0 - over_prob,
            "prediction": "over" if over_prob > 0.5 else "under",
            "prop_line": prop_line,
            "prop_type": self.prop_type,
            "confidence": abs(over_prob - 0.5) * 2,
            "raw_probability": float(prob_raw),
        }

    def predict_multiple_lines(
        self,
        features: Dict,
        lines: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Predict Over probabilities for multiple lines efficiently.

        Useful for finding the line where P(Over) = 0.5 (fair line).
        """
        results = []
        for line in lines:
            results.append(self.predict(features, line))
        return results

    def find_fair_line(
        self,
        features: Dict,
        search_range: Tuple[float, float] = None
    ) -> float:
        """
        Find the prop line where P(Over) = 50%.

        This is the model's estimate of the "fair" line.
        """
        if search_range is None:
            default_ranges = {
                "points": (5.5, 45.5),
                "rebounds": (2.5, 15.5),
                "assists": (1.5, 12.5),
                "threes": (0.5, 8.5),
                "pra": (10.5, 60.5),
            }
            search_range = default_ranges.get(self.prop_type, (5.5, 35.5))

        # Binary search for line where P(Over) = 0.5
        low, high = search_range
        while high - low > 0.5:
            mid = (low + high) / 2
            pred = self.predict(features, mid)
            if pred['over_probability'] > 0.5:
                low = mid
            else:
                high = mid

        return round((low + high) / 2 * 2) / 2  # Round to 0.5

    def save_model(self, filepath: Optional[Path] = None):
        """Save the line-aware classifier."""
        if filepath is None:
            filepath = MODEL_DIR / f"{self.model_name}.pkl"

        model_data = {
            "model": self.model,
            "calibrator": self.calibrator,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "model_name": self.model_name,
            "prop_type": self.prop_type,
            "line_stats": self.line_stats,
            "saved_at": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Line-aware classifier saved to {filepath}")
        return filepath

    def load_model(self, filepath: Optional[Path] = None):
        """Load a saved line-aware classifier."""
        if filepath is None:
            filepath = MODEL_DIR / f"{self.model_name}.pkl"

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.calibrator = model_data.get("calibrator")
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.training_metrics = model_data["training_metrics"]
        self.line_stats = model_data.get("line_stats", {})
        self.is_fitted = True

        print(f"Line-aware classifier loaded from {filepath}")
        return self


class PlayerPropModel(BaseModelTrainer):
    """
    Random Forest model for player prop predictions.

    Predicts various player statistics (points, rebounds, assists, etc.).
    """

    def __init__(self, prop_type: str = "points", use_classifier: bool = False):
        """
        Initialize player prop model.

        Args:
            prop_type: Type of prop ("points", "rebounds", "assists", "threes", "pra")
            use_classifier: If True, classifies over/under; if False, predicts value
        """
        self.prop_type = prop_type
        self.use_classifier = use_classifier

        model_name = f"player_{prop_type}_{'classifier' if use_classifier else 'regressor'}"
        super().__init__(model_name)

        if use_classifier:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )

    def prepare_training_data(
        self,
        player_data: List[Dict],
        prop_line: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from historical player games.

        Args:
            player_data: List of player game dictionaries with features and outcomes
            prop_line: If using classifier, the prop line to evaluate

        Returns:
            Tuple of (features DataFrame, labels array)
        """
        # Map prop type to feature key
        prop_feature_map = {
            "points": "points_features",
            "rebounds": "rebounds_features",
            "assists": "assists_features",
            "threes": "threes_features",
            "pra": "pra_features",
        }

        # Map prop type to actual stat key
        stat_key_map = {
            "points": "pts",
            "rebounds": "reb",
            "assists": "ast",
            "threes": "fg3_made",
            "pra": "pra",  # pts + reb + ast
        }

        feature_key = prop_feature_map.get(self.prop_type, "points_features")
        stat_key = stat_key_map.get(self.prop_type, "pts")

        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in player_data:
            features = game.get(feature_key, {})
            actual_value = game.get("actual_stats", {}).get(stat_key, None)

            # Handle PRA calculation
            if self.prop_type == "pra" and actual_value is None:
                stats = game.get("actual_stats", {})
                pts = stats.get("pts", 0) or 0
                reb = stats.get("reb", 0) or 0
                ast = stats.get("ast", 0) or 0
                actual_value = pts + reb + ast

            if features and actual_value is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k != "player_id"
                }
                features_list.append(numeric_features)
                game_dates.append(game.get("game_date", "1900-01-01"))

                if self.use_classifier and prop_line is not None:
                    labels.append(1 if actual_value > prop_line else 0)
                else:
                    labels.append(actual_value)

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
        tune_hyperparameters: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the player prop model.

        Args:
            X: Feature DataFrame
            y: Target values
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            tune_hyperparameters: Whether to perform grid search

        Returns:
            Dictionary with training metrics
        """
        # Split data
        if self.use_classifier:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        # Preprocess
        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        # Hyperparameter tuning
        if tune_hyperparameters:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            }
            grid_search = GridSearchCV(
                self.model, param_grid, cv=cv_folds, scoring="neg_mean_squared_error" if not self.use_classifier else "accuracy",
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)

            # Train
            self.model.fit(X_train_scaled, y_train)

        self.is_fitted = True

        # Predictions
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        if self.use_classifier:
            self.training_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "cv_mean": cv_scores.mean() if not tune_hyperparameters else 0,
                "cv_std": cv_scores.std() if not tune_hyperparameters else 0,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
            print(f"\n{self.prop_type.title()} Prop Classifier Training Results:")
            print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
            print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        else:
            self.training_metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "cv_mean": cv_scores.mean() if not tune_hyperparameters else 0,
                "cv_std": cv_scores.std() if not tune_hyperparameters else 0,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
            print(f"\n{self.prop_type.title()} Prop Regressor Training Results:")
            print(f"  RMSE: {self.training_metrics['rmse']:.2f}")
            print(f"  MAE: {self.training_metrics['mae']:.2f}")
            print(f"  R2: {self.training_metrics['r2']:.4f}")

        return self.training_metrics

    def predict(self, features: Dict, prop_line: Optional[float] = None) -> Dict[str, Any]:
        """
        Predict player prop outcome.

        Args:
            features: Player prop features dictionary
            prop_line: The betting line to evaluate

        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k != "player_id"
        }

        X = pd.DataFrame([numeric_features])

        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)

        if self.use_classifier:
            prob = self.model.predict_proba(X_scaled)[0]
            return {
                "over_probability": prob[1],
                "under_probability": prob[0],
                "prediction": "over" if prob[1] > 0.5 else "under",
                "confidence": max(prob),
                "prop_type": self.prop_type,
            }
        else:
            predicted_value = self.model.predict(X_scaled)[0]
            result = {
                "predicted_value": predicted_value,
                "prop_type": self.prop_type,
            }

            if prop_line is not None:
                result["prop_line"] = prop_line
                result["prediction"] = "over" if predicted_value > prop_line else "under"
                result["edge"] = predicted_value - prop_line

            return result


class XGBoostMoneylineModel(BaseModelTrainer):
    """
    XGBoost model for moneyline predictions.

    XGBoost typically outperforms logistic regression for complex patterns.
    Requires: pip install xgboost
    """

    def __init__(self):
        super().__init__("moneyline_xgboost")
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
        )

    def prepare_training_data(self, games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data (same as MoneylineModel) - SORTED CHRONOLOGICALLY."""
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("moneyline_features", {})
            outcome = game.get("home_win", None)

            if features and outcome is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
                }
                features_list.append(numeric_features)
                labels.append(1 if outcome else 0)
                game_dates.append(game.get("game_date", "1900-01-01"))

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_time_series_cv: bool = True,
    ) -> Dict[str, Any]:
        """Train the XGBoost moneyline model with time-series validation."""
        if use_time_series_cv:
            # TIME-SERIES WALK-FORWARD VALIDATION
            n_samples = len(X)
            test_samples = int(n_samples * test_size)
            X_train = X.iloc[:-test_samples]
            X_test = X.iloc[-test_samples:]
            y_train = y[:-test_samples]
            y_test = y[-test_samples:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        self.training_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        print(f"\nXGBoost Moneyline Model Training Results:")
        print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        print(f"  CV Score: {self.training_metrics['cv_mean']:.4f} (+/- {self.training_metrics['cv_std']:.4f})")

        return self.training_metrics

    def predict(self, features: Dict, calibrate: bool = True) -> Dict[str, float]:
        """
        Predict home team win probability.

        Args:
            features: Feature dictionary for the matchup
            calibrate: Whether to apply probability calibration (default: True)
                       Calibration improves betting accuracy by ensuring
                       predicted probabilities match actual win rates.

        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)
        prob = self.model.predict_proba(X_scaled)[0]

        home_prob = float(np.clip(prob[1], 0.0, 1.0))
        away_prob = float(np.clip(prob[0], 0.0, 1.0))

        # Apply probability calibration if available and requested
        if calibrate and HAS_CALIBRATION:
            try:
                home_prob = calibrate_moneyline_probability(home_prob)
                away_prob = 1.0 - home_prob  # Ensure probabilities sum to 1
            except Exception:
                pass  # Use uncalibrated if calibration fails

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(np.clip(max(home_prob, away_prob), 0.0, 1.0)),
            "is_calibrated": calibrate and HAS_CALIBRATION,
        }


class LightGBMSpreadModel(BaseModelTrainer):
    """
    LightGBM model for spread predictions.

    LightGBM is faster and often more accurate than traditional methods.
    Requires: pip install lightgbm
    """

    def __init__(self):
        super().__init__("spread_lightgbm")
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

    def prepare_training_data(self, games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data for spread prediction - SORTED CHRONOLOGICALLY."""
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("spread_features", {})
            actual_diff = game.get("point_differential", None)

            if features and actual_diff is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id", "injury_details"]
                }
                features_list.append(numeric_features)
                labels.append(actual_diff)
                game_dates.append(game.get("game_date", "1900-01-01"))

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train the LightGBM spread model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds,
                                     scoring='neg_mean_squared_error')

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test_scaled)

        self.training_metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "cv_mean_rmse": np.sqrt(-cv_scores.mean()),
            "cv_std_rmse": np.sqrt(cv_scores.std()),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        print(f"\nLightGBM Spread Model Training Results:")
        print(f"  RMSE: {self.training_metrics['rmse']:.2f} points")
        print(f"  MAE: {self.training_metrics['mae']:.2f} points")
        print(f"  R2: {self.training_metrics['r2']:.4f}")
        print(f"  CV RMSE: {self.training_metrics['cv_mean_rmse']:.2f}")

        return self.training_metrics

    def predict(self, features: Dict, spread_line: Optional[float] = None) -> Dict[str, Any]:
        """Predict spread outcome."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id", "injury_details"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)
        predicted_diff = self.model.predict(X_scaled)[0]

        # Clip to realistic NBA range
        predicted_diff = float(np.clip(predicted_diff, -30.0, 30.0))

        result = {
            "predicted_spread": predicted_diff,
            "predicted_winner": "home" if predicted_diff > 0 else "away",
            "predicted_margin": abs(predicted_diff),
        }

        if spread_line is not None:
            result["covers_spread"] = predicted_diff > spread_line
            result["spread_line"] = spread_line
            result["edge"] = float(np.clip(predicted_diff - spread_line, -20.0, 20.0))

        return result


class EnsembleMoneylineModel(BaseModelTrainer):
    """
    TIER 1 UPGRADE: Ensemble model with Neural Network for moneyline predictions.

    Uses stacking with:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - MLPClassifier (Neural Network) - NEW
    - XGBoost (if installed)
    - LightGBM (if installed)

    The MLPClassifier adds non-linear pattern recognition that tree-based
    models may miss, improving overall ensemble diversity and accuracy.
    """

    def __init__(self):
        super().__init__("moneyline_ensemble")

        # Base estimators - Now includes Neural Network + DIVERSE MODELS
        estimators = [
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
            # Neural Network for capturing complex non-linear patterns
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(64, 32),  # Two hidden layers
                activation='relu',
                solver='adam',
                alpha=0.0001,  # L2 regularization
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42,
            )),
            # DIVERSITY MODELS: Reduce ensemble correlation with non-tree models
            # Naive Bayes: Fast, different inductive bias (independence assumption)
            ('nb', GaussianNB()),
            # Quadratic Discriminant Analysis: Quadratic decision boundaries
            ('qda', QuadraticDiscriminantAnalysis(reg_param=0.1)),  # Regularized to avoid singularity
        ]

        if HAS_XGBOOST:
            estimators.append(('xgb', xgb.XGBClassifier(
                n_estimators=100, max_depth=6, random_state=42,
                use_label_encoder=False, eval_metric='logloss'
            )))

        if HAS_LIGHTGBM:
            estimators.append(('lgb', lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, random_state=42, verbose=-1
            )))

        # Stacking classifier with logistic regression as final estimator
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
            n_jobs=-1,
        )

    def prepare_training_data(self, games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data - SORTED CHRONOLOGICALLY."""
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("moneyline_features", {})
            outcome = game.get("home_win", None)

            if features and outcome is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
                }
                features_list.append(numeric_features)
                labels.append(1 if outcome else 0)
                game_dates.append(game.get("game_date", "1900-01-01"))

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_time_series_cv: bool = True,
    ) -> Dict[str, Any]:
        """Train the ensemble model with walk-forward validation."""
        if use_time_series_cv:
            # TIME-SERIES WALK-FORWARD VALIDATION
            n_samples = len(X)
            test_samples = int(n_samples * test_size)
            X_train = X.iloc[:-test_samples]
            X_test = X.iloc[-test_samples:]
            y_train = y[:-test_samples]
            y_test = y[-test_samples:]
            print(f"\n  Using TIME-SERIES validation (walk-forward)")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        print("Training ensemble model (this may take a few minutes)...")
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        self.training_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "num_base_estimators": len(self.model.estimators_),
            "validation_type": "time_series" if use_time_series_cv else "random",
        }

        print(f"\nEnsemble Moneyline Model Training Results:")
        print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        print(f"  Base Estimators: {self.training_metrics['num_base_estimators']}")

        return self.training_metrics

    def predict(self, features: Dict) -> Dict[str, float]:
        """Predict home team win probability."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)
        prob = self.model.predict_proba(X_scaled)[0]

        home_prob = float(np.clip(prob[1], 0.0, 1.0))
        away_prob = float(np.clip(prob[0], 0.0, 1.0))

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(np.clip(max(prob), 0.0, 1.0)),
        }


class TunedEnsembleMoneylineModel(BaseModelTrainer):
    """
    Ensemble model with hyperparameter tuning for moneyline predictions.

    Uses RandomizedSearchCV to find optimal hyperparameters for each base model
    before combining them in a stacking ensemble.
    """

    # Hyperparameter search spaces for each model type
    PARAM_GRIDS = {
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        'gb': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'min_samples_split': [2, 5, 10],
        },
        'xgb': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9, 1.0],
        },
        'lgb': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [15, 31, 63],
        },
    }

    def __init__(self, n_iter: int = 20, cv_folds: int = 3):
        """
        Initialize the tuned ensemble model.

        Args:
            n_iter: Number of parameter combinations to try in RandomizedSearchCV
            cv_folds: Number of cross-validation folds for tuning
        """
        super().__init__("moneyline_ensemble_tuned")
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.best_params = {}
        self.model = None

    def _tune_model(self, model, param_grid, X_train, y_train, model_name: str):
        """Tune a single model using RandomizedSearchCV with TimeSeriesSplit."""
        print(f"    Tuning {model_name}...")

        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=min(self.n_iter, 10),  # Limit iterations per model
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)

        print(f"      Best {model_name} score: {search.best_score_:.4f}")
        print(f"      Best params: {search.best_params_}")

        self.best_params[model_name] = search.best_params_
        return search.best_estimator_

    def prepare_training_data(self, games_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data - SORTED CHRONOLOGICALLY for time-series validation."""
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            features = game.get("moneyline_features", {})
            outcome = game.get("home_win", None)

            if features and outcome is not None:
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
                }
                features_list.append(numeric_features)
                labels.append(1 if outcome else 0)
                game_dates.append(game.get("game_date", "1900-01-01"))

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        tune_hyperparameters: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the tuned ensemble model.

        Args:
            X: Feature DataFrame (MUST be sorted by date)
            y: Target labels
            test_size: Proportion for test set
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Dictionary with training metrics
        """
        # TIME-SERIES SPLIT
        n_samples = len(X)
        test_samples = int(n_samples * test_size)
        X_train = X.iloc[:-test_samples]
        X_test = X.iloc[-test_samples:]
        y_train = y[:-test_samples]
        y_test = y[-test_samples:]

        print(f"\n  Training TunedEnsembleMoneylineModel")
        print(f"  Train: {len(X_train)} games, Test: {len(X_test)} games")

        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        # Build estimators (tuned if requested)
        estimators = []

        # Logistic Regression (minimal tuning needed)
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        estimators.append(('lr', lr))

        if tune_hyperparameters:
            print("  Tuning hyperparameters (this may take a while)...")

            # Tune Random Forest
            rf = RandomForestClassifier(random_state=42)
            tuned_rf = self._tune_model(rf, self.PARAM_GRIDS['rf'], X_train_scaled, y_train, 'rf')
            estimators.append(('rf', tuned_rf))

            # Tune Gradient Boosting
            gb = GradientBoostingClassifier(random_state=42)
            tuned_gb = self._tune_model(gb, self.PARAM_GRIDS['gb'], X_train_scaled, y_train, 'gb')
            estimators.append(('gb', tuned_gb))

            # Tune XGBoost if available
            if HAS_XGBOOST:
                xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
                tuned_xgb = self._tune_model(xgb_model, self.PARAM_GRIDS['xgb'], X_train_scaled, y_train, 'xgb')
                estimators.append(('xgb', tuned_xgb))

            # Tune LightGBM if available
            if HAS_LIGHTGBM:
                lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                tuned_lgb = self._tune_model(lgb_model, self.PARAM_GRIDS['lgb'], X_train_scaled, y_train, 'lgb')
                estimators.append(('lgb', tuned_lgb))

            # DIVERSITY MODELS: Add non-tree models (no tuning needed)
            estimators.append(('nb', GaussianNB()))
            estimators.append(('qda', QuadraticDiscriminantAnalysis(reg_param=0.1)))
        else:
            # Use default parameters
            estimators.append(('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)))
            estimators.append(('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)))
            if HAS_XGBOOST:
                estimators.append(('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')))
            if HAS_LIGHTGBM:
                estimators.append(('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)))
            # DIVERSITY MODELS: Add non-tree models
            estimators.append(('nb', GaussianNB()))
            estimators.append(('qda', QuadraticDiscriminantAnalysis(reg_param=0.1)))

        # Create stacking classifier with XGBoost meta-learner (better than LR)
        if HAS_XGBOOST:
            final_estimator = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
        else:
            final_estimator = LogisticRegression(max_iter=1000, C=0.1)

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,  # Standard CV for stacking (TimeSeriesSplit not compatible)
            n_jobs=-1,
        )

        print("  Training final stacked ensemble...")
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        self.training_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "num_base_estimators": len(estimators),
            "tuned": tune_hyperparameters,
            "best_params": self.best_params,
        }

        print(f"\n  Tuned Ensemble Training Results:")
        print(f"    Accuracy: {self.training_metrics['accuracy']:.4f}")
        print(f"    F1 Score: {self.training_metrics['f1']:.4f}")
        print(f"    Base Estimators: {len(estimators)}")

        return self.training_metrics

    def predict(self, features: Dict) -> Dict[str, float]:
        """Predict home team win probability."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)
        prob = self.model.predict_proba(X_scaled)[0]

        home_prob = float(np.clip(prob[1], 0.0, 1.0))
        away_prob = float(np.clip(prob[0], 0.0, 1.0))

        return {
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": float(np.clip(max(prob), 0.0, 1.0)),
        }


class TotalsModel(BaseModelTrainer):
    """
    Model for predicting game totals (over/under).

    Predicts total combined points and evaluates against betting lines.
    """

    def __init__(self, use_classifier: bool = False):
        model_name = "totals_classifier" if use_classifier else "totals_regressor"
        super().__init__(model_name)
        self.use_classifier = use_classifier

        if use_classifier:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
            )
        else:
            if HAS_LIGHTGBM:
                self.model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1,
                )
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                )

    def prepare_training_data(
        self,
        games_data: List[Dict],
        total_line: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data for totals prediction - SORTED CHRONOLOGICALLY.

        Args:
            games_data: List of game dictionaries with features and outcomes
            total_line: If using classifier, the total line to evaluate

        Returns:
            Tuple of (features DataFrame, labels array)
        """
        features_list = []
        labels = []
        game_dates = []  # Track dates for chronological sorting

        for game in games_data:
            # Get spread features and add pace-related features
            features = game.get("spread_features", {}).copy()

            # Add totals-specific features if available
            totals_features = game.get("totals_features", {})
            features.update(totals_features)

            home_score = game.get("home_score", None)
            away_score = game.get("away_score", None)

            if features and home_score is not None and away_score is not None:
                total_points = home_score + away_score

                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
                }
                features_list.append(numeric_features)
                game_dates.append(game.get("game_date", "1900-01-01"))

                if self.use_classifier and total_line is not None:
                    labels.append(1 if total_points > total_line else 0)
                else:
                    labels.append(total_points)

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        # CRITICAL: Sort by date (oldest first) for time-series validation
        if game_dates and len(game_dates) == len(X):
            date_series = pd.Series(game_dates)
            sort_indices = date_series.argsort().values
            X = X.iloc[sort_indices].reset_index(drop=True)
            y = y[sort_indices]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train the totals model."""
        if self.use_classifier:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_test_scaled = self.preprocess_features(X_test, fit=False)

        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test_scaled)

        if self.use_classifier:
            self.training_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
            print(f"\nTotals Classifier Training Results:")
            print(f"  Accuracy: {self.training_metrics['accuracy']:.4f}")
            print(f"  F1 Score: {self.training_metrics['f1']:.4f}")
        else:
            self.training_metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
            print(f"\nTotals Regressor Training Results:")
            print(f"  RMSE: {self.training_metrics['rmse']:.2f} points")
            print(f"  MAE: {self.training_metrics['mae']:.2f} points")
            print(f"  R2: {self.training_metrics['r2']:.4f}")

        print(f"  CV Score: {self.training_metrics['cv_mean']:.4f} (+/- {self.training_metrics['cv_std']:.4f})")
        return self.training_metrics

    def predict(self, features: Dict, total_line: Optional[float] = None) -> Dict[str, Any]:
        """
        Predict total points.

        Args:
            features: Game features dictionary
            total_line: The betting line to evaluate

        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Train or load a model first.")

        numeric_features = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float)) and k not in ["home_team_id", "away_team_id"]
        }

        X = pd.DataFrame([numeric_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.preprocess_features(X, fit=False)

        if self.use_classifier:
            prob = self.model.predict_proba(X_scaled)[0]
            return {
                "over_probability": float(np.clip(prob[1], 0.0, 1.0)),
                "under_probability": float(np.clip(prob[0], 0.0, 1.0)),
                "prediction": "over" if prob[1] > 0.5 else "under",
                "confidence": float(np.clip(max(prob), 0.0, 1.0)),
            }
        else:
            predicted_total = self.model.predict(X_scaled)[0]

            # Clip to realistic NBA total range (160-280)
            predicted_total = float(np.clip(predicted_total, 160.0, 280.0))

            result = {
                "predicted_total": predicted_total,
            }

            if total_line is not None:
                result["total_line"] = total_line
                result["prediction"] = "over" if predicted_total > total_line else "under"
                result["edge"] = float(np.clip(predicted_total - total_line, -30.0, 30.0))

                # Convert edge to probability estimate
                edge_in_points = predicted_total - total_line
                # Roughly: 10 points edge = ~70% probability
                prob = 1 / (1 + np.exp(-edge_in_points / 10.0))
                result["over_probability"] = float(np.clip(prob, 0.0, 1.0))
                result["under_probability"] = float(np.clip(1.0 - prob, 0.0, 1.0))

            return result


class ParlayCalculator:
    """
    Calculator for parlay probabilities and expected values.

    Combines multiple individual bet predictions into parlay analysis.
    """

    def __init__(self):
        self.models = {}

    def add_model(self, model_type: str, model: BaseModelTrainer):
        """Add a trained model for parlay calculations."""
        self.models[model_type] = model

    def calculate_parlay_probability(self, legs: List[Dict]) -> Dict[str, Any]:
        """
        Calculate combined probability for a parlay.

        Args:
            legs: List of parlay legs, each with:
                - type: "moneyline", "spread", or "prop"
                - features: Feature dictionary for prediction
                - selection: "home", "away", "over", "under", "cover"
                - line: Betting line (for spread/props)

        Returns:
            Dictionary with parlay analysis
        """
        individual_probs = []
        leg_details = []

        for leg in legs:
            leg_type = leg.get("type")
            features = leg.get("features", {})
            selection = leg.get("selection")
            line = leg.get("line")

            if leg_type == "moneyline" and "moneyline" in self.models:
                pred = self.models["moneyline"].predict(features)
                if selection == "home":
                    prob = pred["home_win_probability"]
                else:
                    prob = pred["away_win_probability"]

            elif leg_type == "spread" and "spread" in self.models:
                pred = self.models["spread"].predict(features, spread_line=line)
                if self.models["spread"].use_classifier:
                    prob = pred["cover_probability"] if selection == "cover" else pred["no_cover_probability"]
                else:
                    # Convert regression to probability estimate
                    edge = pred.get("edge", 0)
                    prob = self._edge_to_probability(edge)

            elif leg_type == "prop":
                prop_type = leg.get("prop_type", "points")
                model_key = f"prop_{prop_type}"
                if model_key in self.models:
                    pred = self.models[model_key].predict(features, prop_line=line)
                    if self.models[model_key].use_classifier:
                        prob = pred["over_probability"] if selection == "over" else pred["under_probability"]
                    else:
                        edge = pred.get("edge", 0)
                        prob = self._edge_to_probability(edge) if selection == "over" else self._edge_to_probability(-edge)
                else:
                    prob = 0.5  # Default if model not available

            else:
                prob = 0.5  # Default probability

            individual_probs.append(prob)
            leg_details.append({
                "type": leg_type,
                "selection": selection,
                "probability": prob,
                "line": line,
            })

        # Calculate combined probability (independent events)
        # Ensure all individual probs are valid before multiplication
        valid_probs = [float(np.clip(p, 0.0, 1.0)) for p in individual_probs]
        combined_prob = float(np.prod(valid_probs))
        # Combined probability must be between 0 and 1
        combined_prob = float(np.clip(combined_prob, 0.0, 1.0))

        return {
            "combined_probability": combined_prob,
            "individual_legs": leg_details,
            "num_legs": len(legs),
            "implied_odds": self._probability_to_american_odds(combined_prob),
        }

    def calculate_expected_value(
        self,
        parlay_prob: float,
        odds: float,
        stake: float = 100,
    ) -> Dict[str, float]:
        """
        Calculate expected value for a parlay.

        Args:
            parlay_prob: Combined probability of winning
            odds: American odds offered
            stake: Bet amount

        Returns:
            Dictionary with EV analysis
        """
        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

        # Calculate potential profit and EV
        potential_profit = stake * (decimal_odds - 1)
        ev = (parlay_prob * potential_profit) - ((1 - parlay_prob) * stake)
        ev_percentage = (ev / stake) * 100

        # Calculate implied probability from odds
        implied_prob = 1 / decimal_odds

        return {
            "expected_value": ev,
            "ev_percentage": ev_percentage,
            "potential_profit": potential_profit,
            "stake": stake,
            "model_probability": parlay_prob,
            "implied_probability": implied_prob,
            "edge": parlay_prob - implied_prob,
            "recommendation": "bet" if ev > 0 else "pass",
        }

    def _edge_to_probability(self, edge: float, scale: float = 5.0) -> float:
        """
        Convert point edge to probability using sigmoid.

        Args:
            edge: Point edge (positive = favorable)
            scale: Scaling factor (default 5.0 means ~73% prob for 5-point edge)

        Returns:
            Probability between 0.0 and 1.0
        """
        # Clip edge to prevent overflow in exp
        edge = float(np.clip(edge, -50.0, 50.0))
        prob = 1 / (1 + np.exp(-edge / scale))
        # Ensure output is valid probability
        return float(np.clip(prob, 0.0, 1.0))

    def _probability_to_american_odds(self, prob: float) -> float:
        """Convert probability to American odds."""
        if prob <= 0:
            return 0
        if prob >= 1:
            return -10000

        if prob >= 0.5:
            return -100 * prob / (1 - prob)
        else:
            return 100 * (1 - prob) / prob


class ModelTrainingPipeline:
    """
    Complete training pipeline for all NBA betting models.
    """

    def __init__(self, season: str = "2025-26"):
        self.season = season
        self.models = {}

    def train_all_models(
        self,
        games_data: List[Dict],
        player_data: Optional[List[Dict]] = None,
        save_models: bool = True,
        use_ensemble: bool = True,
        use_tuned_ensemble: bool = True,
        use_line_aware: bool = True,
        run_backtest: bool = False,
        backtest_min_games: int = 100,
    ) -> Dict[str, Dict]:
        """
        Train all models with provided data.

        Args:
            games_data: Historical game data with features and outcomes
            player_data: Historical player data with features and outcomes
            save_models: Whether to save models after training
            use_ensemble: If True, use Ensemble model for maximum accuracy (default: True)
            use_tuned_ensemble: If True with use_ensemble, use TunedEnsembleMoneylineModel
                               with GridSearchCV hyperparameter optimization (default: True)
            use_line_aware: If True, use LineAwarePropClassifier for props (default: True)
                           These classifiers take the prop line as input and output P(Over)
                           directly, which is better for betting than regression models.
            run_backtest: If True, run walk-forward backtest after training (default: False)
            backtest_min_games: Minimum games required for backtesting (default: 100)

        Returns:
            Dictionary with all training metrics
        """
        results = {}

        # Train Moneyline Model - Use Tuned Ensemble for best accuracy
        print("\n" + "=" * 50)
        if use_ensemble and use_tuned_ensemble and (HAS_XGBOOST or HAS_LIGHTGBM):
            print("Training Moneyline Model (TUNED ENSEMBLE - OPTIMIZED HYPERPARAMETERS)")
            print("  Components: LR + RF + GradientBoosting + XGBoost + LightGBM")
            print("  Using GridSearchCV for hyperparameter optimization")
            print("=" * 50)
            moneyline_model = TunedEnsembleMoneylineModel(n_iter=30, cv_folds=5)
        elif use_ensemble and (HAS_XGBOOST or HAS_LIGHTGBM):
            print("Training Moneyline Model (ENSEMBLE - MAXIMUM ACCURACY)")
            print("  Components: LR + RF + GradientBoosting + XGBoost + LightGBM")
            print("=" * 50)
            moneyline_model = EnsembleMoneylineModel()
        elif HAS_XGBOOST:
            print("Training Moneyline Model (XGBoost)")
            print("=" * 50)
            moneyline_model = XGBoostMoneylineModel()
        else:
            print("Training Moneyline Model (Logistic Regression)")
            print("=" * 50)
            moneyline_model = MoneylineModel()
        X_ml, y_ml = moneyline_model.prepare_training_data(games_data)
        if len(X_ml) > 0:
            # CRITICAL FIX: Split data BEFORE training to prevent data leakage!
            # Previously, model was trained on ALL data, then "test" set was created
            # from data the model had already seen = data leakage = fake 3500%+ ROI
            n_samples = len(X_ml)
            n_test = int(n_samples * 0.2)  # 20% for final test
            n_cal_val = int((n_samples - n_test) * 0.25)  # 25% of remaining for calibration

            train_end = n_samples - n_test - n_cal_val
            cal_val_end = n_samples - n_test

            # Data splits (chronological order preserved - oldest data for training)
            X_train_only = X_ml.iloc[:train_end]
            y_train_only = y_ml[:train_end]
            X_cal_val = X_ml.iloc[train_end:cal_val_end]
            y_cal_val = y_ml[train_end:cal_val_end]
            X_test = X_ml.iloc[cal_val_end:]
            y_test = y_ml[cal_val_end:]

            print(f"  Data split: {len(X_train_only)} train, {len(X_cal_val)} cal_val, {len(X_test)} test (HELD OUT)")

            # Train model on TRAINING DATA ONLY (test set never seen!)
            results["moneyline"] = moneyline_model.train(X_train_only, y_train_only)
            self.models["moneyline"] = moneyline_model
            if save_models:
                moneyline_model.save_model()

            # CALIBRATION: Fit calibrators on cal_val data
            if HAS_CALIBRATION:
                try:
                    print("\n  Fitting moneyline calibrator...")

                    # STEP 1: Get predictions on CALIBRATION VALIDATION set only
                    # Model was trained on X_train_only, so cal_val is unseen
                    y_prob_cal = np.array([
                        moneyline_model.predict(dict(zip(moneyline_model.feature_names, x)))["home_win_probability"]
                        for x in X_cal_val.values
                    ])

                    # STEP 3: Fit calibrator on calibration validation predictions ONLY
                    from calibration import ModelCalibrator
                    ml_calibrator = ModelCalibrator("moneyline", include_advanced=True)
                    ml_calibrator.fit(y_prob_cal, y_cal_val, method="auto")

                    if save_models:
                        calibration_dir = MODEL_DIR / "calibration"
                        calibration_dir.mkdir(exist_ok=True)
                        ml_calibrator.save(str(calibration_dir))

                    results["moneyline"]["calibration"] = {
                        "best_method": ml_calibrator.best_method,
                        "ece": ml_calibrator.metrics.get(ml_calibrator.best_method, {}).ece if ml_calibrator.metrics.get(ml_calibrator.best_method) else None,
                    }
                    print(f"  Moneyline calibrator saved (method: {ml_calibrator.best_method})")

                    # STEP 4: LOG METRICS on HELD-OUT TEST set (never seen by calibrator)
                    try:
                        logger = TrainingMetricsLogger("moneyline", model_type="classifier")

                        # Get predictions on truly held-out test set
                        y_prob_test = np.array([
                            moneyline_model.predict(dict(zip(moneyline_model.feature_names, x)))["home_win_probability"]
                            for x in X_test.values
                        ])

                        # Calibrate the test predictions
                        y_prob_test_calibrated = ml_calibrator.calibrate(y_prob_test)

                        # Log classification metrics on TEST data only (honest evaluation)
                        y_pred_test = (y_prob_test_calibrated > 0.5).astype(int)
                        logger.log_classification_metrics(y_test, y_pred_test, y_prob_test_calibrated)
                        logger.log_calibration_metrics(y_prob_test_calibrated, y_test)

                        # Log betting ROI on TEST data only
                        logger.log_betting_roi(y_prob_test, np.array(y_test))
                        logger.add_custom_metric("train_size", len(X_train_only))
                        logger.add_custom_metric("cal_val_size", len(X_cal_val))
                        logger.add_custom_metric("test_size", len(X_test))
                        logger.add_custom_metric("calibration_method", ml_calibrator.best_method)
                        logger.add_custom_metric("_leakage_free", True)  # Flag confirming proper split
                        if save_models:
                            logger.save()
                        print(logger.get_summary())
                    except Exception as e:
                        print(f"  Warning: Metrics logging failed: {e}")

                except Exception as e:
                    print(f"  Warning: Calibration failed: {e}")

            # BACKTEST: Run walk-forward backtest for moneyline model
            if run_backtest and len(games_data) >= backtest_min_games:
                print("\n  Running walk-forward backtest for moneyline...")
                try:
                    reporter = BacktestReporter()
                    backtest_results = reporter.run_moneyline_backtest(
                        model=moneyline_model,
                        games_data=games_data,
                        initial_bankroll=10000.0,
                        min_edge=0.02,
                    )

                    # Save report
                    report_path = reporter.save_report(backtest_results, "moneyline_ensemble", "moneyline")
                    print(f"  Backtest report saved to {report_path}")

                    # Print summary
                    reporter.print_summary(backtest_results, "moneyline")

                    # Store results
                    results["moneyline"]["backtest"] = backtest_results
                except Exception as e:
                    print(f"  Warning: Moneyline backtest failed: {e}")

        # Train Spread Cover Classifier (replaces SVR regressor)
        # CRITICAL FIX: Using classification (predict cover vs not) instead of regression
        # The classifier takes spread_line as input and directly outputs P(home_covers)
        # This is more accurate than predicting point differential then converting to probability
        print("\n" + "=" * 50)
        print("Training Spread Cover Classifier (XGBoost)")
        print("  - Line-aware: spread_line is an input feature")
        print("  - Outputs P(home_covers) directly")
        print("=" * 50)
        spread_model = SpreadCoverClassifier()
        X_sp, y_sp = spread_model.prepare_training_data(games_data)
        if len(X_sp) > 0:
            # CRITICAL FIX: Split data BEFORE training to prevent data leakage!
            # Previously, model was trained on ALL data, then "test" set was created
            # from data the model had already seen = data leakage = fake 91000%+ ROI
            n_samples = len(X_sp)
            n_test = int(n_samples * 0.2)  # 20% for final test
            n_cal_val = int((n_samples - n_test) * 0.25)  # 25% of remaining for calibration

            train_end = n_samples - n_test - n_cal_val
            cal_val_end = n_samples - n_test

            # Data splits (chronological order preserved - oldest data for training)
            X_train_only = X_sp.iloc[:train_end]
            y_train_only = y_sp[:train_end]
            X_cal_val = X_sp.iloc[train_end:cal_val_end]
            y_cal_val = y_sp[train_end:cal_val_end]
            X_test = X_sp.iloc[cal_val_end:]
            y_test = y_sp[cal_val_end:]

            print(f"  Data split: {len(X_train_only)} train, {len(X_cal_val)} cal_val, {len(X_test)} test (HELD OUT)")

            # Train model on TRAINING DATA ONLY (test set never seen!)
            results["spread"] = spread_model.train(X_train_only, y_train_only)
            self.models["spread"] = spread_model
            if save_models:
                spread_model.save_model()

            # CALIBRATION: Fit calibrators on cal_val data
            if HAS_CALIBRATION:
                try:
                    print("\n  Fitting spread cover calibrator...")

                    # STEP 1: Get predictions on CALIBRATION VALIDATION set only
                    # Model was trained on X_train_only, so cal_val is unseen
                    X_cal_scaled = spread_model.preprocess_features(X_cal_val, fit=False)
                    y_prob_cal = spread_model.model.predict_proba(X_cal_scaled)[:, 1]

                    # STEP 3: Fit calibrator on calibration validation predictions ONLY
                    from calibration import ModelCalibrator
                    sp_calibrator = ModelCalibrator("spread", include_advanced=True)
                    sp_calibrator.fit(y_prob_cal, y_cal_val, method="auto")

                    if save_models:
                        calibration_dir = MODEL_DIR / "calibration"
                        calibration_dir.mkdir(exist_ok=True)
                        sp_calibrator.save(str(calibration_dir))

                    results["spread"]["calibration"] = {
                        "best_method": sp_calibrator.best_method,
                        "ece": sp_calibrator.metrics.get(sp_calibrator.best_method, {}).ece if sp_calibrator.metrics.get(sp_calibrator.best_method) else None,
                    }
                    print(f"  Spread calibrator saved (method: {sp_calibrator.best_method})")

                    # STEP 4: LOG METRICS on HELD-OUT TEST set (never seen by calibrator)
                    try:
                        logger = TrainingMetricsLogger("spread", model_type="classifier")

                        # Get predictions on truly held-out test set
                        X_test_scaled = spread_model.preprocess_features(X_test, fit=False)
                        y_prob_test = spread_model.model.predict_proba(X_test_scaled)[:, 1]

                        # Calibrate the test predictions
                        y_prob_test_calibrated = sp_calibrator.calibrate(y_prob_test)

                        # Log classification metrics on TEST data only
                        y_pred_test = (y_prob_test_calibrated > 0.5).astype(int)
                        logger.log_classification_metrics(y_test, y_pred_test, y_prob_test_calibrated)
                        logger.log_calibration_metrics(y_prob_test_calibrated, y_test)

                        # Log betting ROI on TEST data only
                        logger.log_betting_roi(y_prob_test, y_test)
                        logger.add_custom_metric("train_size", len(X_train_only))
                        logger.add_custom_metric("cal_val_size", len(X_cal_val))
                        logger.add_custom_metric("test_size", len(X_test))
                        logger.add_custom_metric("calibration_method", sp_calibrator.best_method)
                        logger.add_custom_metric("_leakage_free", True)
                        if save_models:
                            logger.save()
                        print(logger.get_summary())
                    except Exception as e:
                        print(f"  Warning: Metrics logging failed: {e}")

                except Exception as e:
                    print(f"  Warning: Spread calibration failed: {e}")

            # BACKTEST: Run walk-forward backtest for spread model
            if run_backtest and len(games_data) >= backtest_min_games:
                print("\n  Running walk-forward backtest for spread...")
                try:
                    reporter = BacktestReporter()
                    backtest_results = reporter.run_spread_backtest(
                        model=spread_model,
                        games_data=games_data,
                        initial_bankroll=10000.0,
                        min_edge=0.02,
                    )

                    # Save report
                    report_path = reporter.save_report(backtest_results, "spread_svm", "spread")
                    print(f"  Backtest report saved to {report_path}")

                    # Print summary
                    reporter.print_summary(backtest_results, "spread")

                    # Store results
                    results["spread"]["backtest"] = backtest_results
                except Exception as e:
                    print(f"  Warning: Spread backtest failed: {e}")

        # Train Player Prop Models
        if player_data:
            prop_types = ["points", "rebounds", "assists", "threes", "pra"]
            for prop_type in prop_types:
                print("\n" + "=" * 50)

                if use_line_aware:
                    # LINE-AWARE CLASSIFIER: Takes prop line as input, outputs P(Over) directly
                    # This is better for betting because it directly predicts betting outcomes
                    print(f"Training {prop_type.title()} Prop Model (LINE-AWARE CLASSIFIER)")
                    print("  - Takes prop line as input feature")
                    print("  - Outputs calibrated P(Over) probability")
                    print("=" * 50)

                    line_classifier = LineAwarePropClassifier(prop_type=prop_type)
                    X_prop, y_prop = line_classifier.prepare_training_data(player_data)

                    if len(X_prop) > 0:
                        train_result = line_classifier.train(X_prop, y_prop)
                        results[f"prop_{prop_type}_line_aware"] = train_result

                        # Register under multiple keys for app.py compatibility
                        self.models[f"prop_{prop_type}_line_aware"] = line_classifier
                        self.models[f"prop_{prop_type}"] = line_classifier  # Primary key for app.py
                        self.models[f"player_{prop_type}_line_classifier"] = line_classifier

                        if save_models:
                            line_classifier.save_model()

                        # Log comprehensive metrics with TrainingMetricsLogger
                        prop_logger = TrainingMetricsLogger(f"prop_{prop_type}_line_aware", "classifier")
                        prop_logger.metrics["accuracy"] = train_result.get("accuracy", 0)
                        prop_logger.metrics["brier_score"] = train_result.get("brier_score", 0)
                        prop_logger.metrics["auc_roc"] = train_result.get("auc_roc", 0)
                        prop_logger.metrics["ece"] = train_result.get("ece", 0)
                        prop_logger.metrics["mce"] = train_result.get("mce", 0)
                        prop_logger.metrics["train_size"] = train_result.get("train_size", 0)
                        prop_logger.metrics["test_size"] = train_result.get("test_size", 0)
                        prop_logger.metrics["over_rate_test"] = train_result.get("over_rate_test", 0)
                        prop_logger.metrics["calibrated"] = train_result.get("calibrated", False)
                        prop_logger.metrics["line_stats"] = train_result.get("line_stats", {})
                        prop_logger.metrics["prop_type"] = prop_type
                        prop_logger.metrics["model_architecture"] = "LineAwarePropClassifier"
                        prop_logger.save()
                        print(f"  Metrics saved to training_metrics/prop_{prop_type}_line_aware_{prop_logger.timestamp}.json")

                        # Save separate prop calibrator for consistency with moneyline/spread
                        if hasattr(line_classifier, 'y_prob_final') and hasattr(line_classifier, 'y_test_final'):
                            try:
                                from calibration import ModelCalibrator
                                prop_calibrator = ModelCalibrator(f"prop_{prop_type}", include_advanced=False)
                                prop_calibrator.fit(
                                    line_classifier.y_prob_final,
                                    line_classifier.y_test_final,
                                    method="isotonic"
                                )
                                prop_calibrator.save(str(calibration_dir))
                                print(f"  Prop calibrator saved to models/calibration/prop_{prop_type}_*.pkl")
                            except Exception as e:
                                print(f"  Warning: Could not save prop calibrator: {e}")
                else:
                    # REGRESSION MODEL: Predicts stat value, requires conversion to probability
                    print(f"Training {prop_type.title()} Prop Model (Random Forest Regressor)")
                    print("=" * 50)

                    prop_model = PlayerPropModel(prop_type=prop_type, use_classifier=False)
                    X_prop, y_prop = prop_model.prepare_training_data(player_data)

                    if len(X_prop) > 0:
                        results[f"prop_{prop_type}"] = prop_model.train(X_prop, y_prop)
                        self.models[f"prop_{prop_type}"] = prop_model
                        if save_models:
                            prop_model.save_model()

        return results

    def load_all_models(self) -> Dict[str, BaseModelTrainer]:
        """Load all saved models."""
        model_files = list(MODEL_DIR.glob("*.pkl"))

        for filepath in model_files:
            model_name = filepath.stem

            try:
                # First, try direct loading to check for ensemble/wrapper or prop models
                with open(filepath, "rb") as f:
                    model_data = pickle.load(f)

                # Check if this is a prop model FIRST (before ensemble check)
                # Prop models have 'prop_type' key OR are named 'player_*'
                if "prop_type" in model_data or model_name.startswith("player_"):
                    prop_wrapper = PropModelWrapper(
                        model=model_data.get("model"),
                        scaler=model_data.get("scaler"),
                        feature_names=model_data.get("feature_names", []),
                        training_metrics=model_data.get("training_metrics", {}),
                        prop_type=model_data.get("prop_type", model_name.replace("player_", "")),
                    )
                    self.models[model_name] = prop_wrapper
                    continue

                # Check if this is an ensemble wrapper model (has predict method on the 'model' key)
                model_obj = model_data.get("model")
                if model_obj is not None and hasattr(model_obj, "predict") and hasattr(model_obj, "models"):
                    # This is our EnsembleMoneylineWrapper
                    wrapper = model_obj
                    wrapper.scaler = model_data.get("scaler", wrapper.scaler if hasattr(wrapper, 'scaler') else None)
                    wrapper.feature_names = model_data.get("feature_names", wrapper.feature_names if hasattr(wrapper, 'feature_names') else [])
                    wrapper.training_metrics = model_data.get("training_metrics", {})
                    wrapper.is_fitted = True
                    self.models[model_name] = wrapper
                    continue

            except Exception:
                pass  # Fall through to legacy loading

            if "moneyline" in model_name:
                # Try XGBoost first if available, fall back to Logistic Regression
                if "xgboost" in model_name.lower() and HAS_XGBOOST:
                    model = XGBoostMoneylineModel()
                elif HAS_XGBOOST:
                    # For generic moneyline models, try XGBoost
                    try:
                        model = XGBoostMoneylineModel()
                    except Exception:
                        model = MoneylineModel()
                else:
                    model = MoneylineModel()
            elif "spread" in model_name:
                # Use SpreadCoverClassifier for spread_cover_classifier.pkl
                if "spread_cover_classifier" in model_name:
                    # SpreadCoverClassifier.load_model is a classmethod that returns the loaded instance
                    try:
                        model = SpreadCoverClassifier.load_model(filepath)
                        self.models[model_name] = model
                        # Also register under "spread" key for app.py compatibility
                        self.models["spread"] = model
                        print(f"  Registered {model_name} as spread (classifier preferred)")
                        continue  # Skip the normal load_model call below
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                        continue
                else:
                    # Legacy SpreadModel (SVR regressor)
                    use_classifier = "classifier" in model_name
                    model = SpreadModel(use_classifier=use_classifier)
            elif "player_" in model_name:
                # Extract prop type
                parts = model_name.split("_")
                prop_type = parts[1] if len(parts) > 1 else "points"

                # Check if this is a line-aware classifier
                if "line_classifier" in model_name or "line_aware" in model_name:
                    model = LineAwarePropClassifier(prop_type=prop_type)
                else:
                    use_classifier = "classifier" in model_name
                    model = PlayerPropModel(prop_type=prop_type, use_classifier=use_classifier)
            else:
                continue

            try:
                model.load_model(filepath)
                self.models[model_name] = model

                # For line-aware prop classifiers, also register under prop_{type} key
                # This ensures app.py can find them using the standard prop_{type} lookup
                if "line_classifier" in model_name or "line_aware" in model_name:
                    parts = model_name.split("_")
                    if len(parts) > 1:
                        prop_type = parts[1]  # Extract from player_{type}_line_classifier
                        self.models[f"prop_{prop_type}"] = model
                        print(f"  Registered {model_name} as prop_{prop_type} (line-aware preferred)")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        return self.models

    def get_parlay_calculator(self) -> ParlayCalculator:
        """Get parlay calculator with loaded models."""
        calculator = ParlayCalculator()
        for model_name, model in self.models.items():
            calculator.add_model(model_name, model)
        return calculator


def create_sample_training_data() -> Tuple[List[Dict], List[Dict]]:
    """
    Create sample training data structure for demonstration.

    In production, this would be replaced with actual historical data.

    Returns:
        Tuple of (games_data, player_data)
    """
    # Sample structure for games data
    games_data_sample = {
        "moneyline_features": {
            "season_win_pct_diff": 0.1,
            "recent_win_pct_diff": 0.2,
            "net_rating_diff": 3.5,
            "off_rating_diff": 2.0,
            "def_rating_diff": -1.5,
            "home_streak": 3,
            "away_streak": -2,
            "h2h_home_win_pct": 0.6,
        },
        "spread_features": {
            "season_win_pct_diff": 0.1,
            "expected_point_diff": 5.0,
            "plus_minus_diff": 4.2,
            "net_rating_diff": 3.5,
        },
        "home_win": True,
        "point_differential": 8,
    }

    # Sample structure for player data
    player_data_sample = {
        "points_features": {
            "season_pts_avg": 25.3,
            "recent_pts_avg": 28.1,
            "pts_trend": 2.8,
            "opp_def_rating": 112.5,
        },
        "actual_stats": {
            "pts": 27,
            "reb": 8,
            "ast": 6,
        },
    }

    print("\nSample training data structure created.")
    print("In production, replace with actual historical data.")

    return [games_data_sample], [player_data_sample]


if __name__ == "__main__":
    print("NBA Betting Model Trainer")
    print("=" * 50)
    print("\nUsage:")
    print("""
# Initialize pipeline
pipeline = ModelTrainingPipeline(season="2025-26")

# Train all models with historical data
# games_data: List of dicts with moneyline_features, spread_features, outcomes
# player_data: List of dicts with prop features and actual stats
results = pipeline.train_all_models(games_data, player_data)

# Load saved models
models = pipeline.load_all_models()

# Make predictions
moneyline_pred = models["moneyline"].predict(features)
spread_pred = models["spread"].predict(features, spread_line=-3.5)

# Calculate parlay probabilities
calculator = pipeline.get_parlay_calculator()
parlay = calculator.calculate_parlay_probability([
    {"type": "moneyline", "features": ml_features, "selection": "home"},
    {"type": "spread", "features": sp_features, "selection": "cover", "line": -3.5},
])
ev = calculator.calculate_expected_value(parlay["combined_probability"], odds=+250)
""")

    # Create sample data structure
    games_sample, player_sample = create_sample_training_data()
    print("\nSample games data structure:")
    print(json.dumps(games_sample[0], indent=2, default=str))
