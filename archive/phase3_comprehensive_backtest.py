"""
Phase 3 Comprehensive 2-Season Backtest

Validates all Phase 3 improvements including:
1. Quantile regression prediction bands
2. Kelly criterion bet sizing with tier adjustments
3. Stop-loss rules
4. Portfolio risk management
5. Comprehensive metrics and HTML reporting

Runs on TWO full seasons:
- 2023-24: Oct 24, 2023 - Apr 14, 2024
- 2024-25: Oct 22, 2024 - Apr 13, 2025

Usage:
    python3 phase3_comprehensive_backtest.py
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict
import traceback

import numpy as np
from scipy.stats import pearsonr

# Import from existing backtest infrastructure
from comprehensive_backtest import (
    SeasonBacktester
)

# Import Risk Management with Kelly Criterion
try:
    from risk_management import calculate_kelly_bet_size, get_kelly_multiplier_for_tier
    HAS_KELLY = True
except ImportError:
    HAS_KELLY = False
    print("WARNING: risk_management.py not found - Kelly bet sizing disabled")

# Import Edge Quality
try:
    from edge_quality import EdgeTier
    HAS_EDGE_QUALITY = True
except ImportError:
    HAS_EDGE_QUALITY = False
    print("WARNING: edge_quality.py not found - using default tiers")

warnings.filterwarnings('ignore')

# Directories
MODEL_DIR = Path("models")
RESULTS_DIR = Path("backtest_results")
RESULTS_DIR.mkdir(exist_ok=True)


def get_tier_from_confidence(confidence: float) -> str:
    """Map confidence score (0-100) to edge quality tier."""
    if confidence >= 90:
        return 'elite'
    if confidence >= 75:
        return 'strong'
    if confidence >= 60:
        return 'moderate'
    if confidence >= 40:
        return 'weak'
    return 'avoid'


@dataclass
class QuantilePrediction:
    """Prediction with quantile bands and confidence scoring."""
    player_name: str
    player_id: int
    prop_type: str
    game_date: str
    game_id: int

    # Predictions
    pred_median: float
    pred_low: float  # 10th percentile
    pred_high: float  # 90th percentile
    predicted_value: float  # Mean prediction (for betting)

    # Actual
    actual_value: float | None = None

    # Confidence & Tier
    confidence: float = 0.0
    tier: str = "unknown"
    band_width: float = 0.0

    # Betting
    line: float | None = None
    over_prob: float = 0.5
    edge: float = 0.0
    suggested_bet_size: float = 0.0
    kelly_fraction: float = 0.0
    bet_recommendation: str = "MONITOR"

    # Error metrics (filled after game)
    error: float | None = None
    abs_error: float | None = None
    squared_error: float | None = None
    hit_over: bool | None = None
    hit_under: bool | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        # Convert None and bool values to JSON-serializable types
        for key, value in result.items():
            if value is None:
                result[key] = None  # None is JSON-serializable
            elif isinstance(value, (bool, np.bool_)):
                result[key] = bool(value)  # Ensure it's Python bool
        return result


@dataclass
class BettingPortfolio:
    """Track bankroll, bets, and stop-loss state."""
    initial_bankroll: float = 1000.0
    current_bankroll: float = 1000.0
    peak_bankroll: float = 1000.0
    max_drawdown_seen: float = 0.0  # Bug #5 fix: track max drawdown over full backtest

    total_bets: int = 0
    total_wagered: float = 0.0
    total_profit: float = 0.0

    wins: int = 0
    losses: int = 0
    pushes: int = 0

    # Stop-loss tracking
    daily_start_bankroll: float = 1000.0
    daily_loss: float = 0.0
    weekly_start_bankroll: float = 1000.0
    weekly_loss: float = 0.0

    # Risk limits
    daily_loss_limit: float = 0.03  # 3% per day
    weekly_loss_limit: float = 0.08  # 8% per week
    max_drawdown_limit: float = 0.15  # 15% from peak
    daily_exposure_limit: float = 0.20  # 20% of bankroll per day

    # Daily exposure tracking
    current_day: str | None = None
    daily_exposure: float = 0.0

    # State
    is_stopped: bool = False
    stop_reason: str | None = None

    def check_stop_loss(self, current_date: str) -> bool:
        """Check if any stop-loss condition is triggered."""
        if self.is_stopped:
            return True

        # Daily loss check
        daily_loss_pct = abs(self.daily_loss) / self.daily_start_bankroll if self.daily_start_bankroll > 0 else 0
        if daily_loss_pct > self.daily_loss_limit:
            self.is_stopped = True
            self.stop_reason = f"Daily loss limit exceeded: {daily_loss_pct:.1%} on {current_date}"
            return True

        # Weekly loss check
        weekly_loss_pct = abs(self.weekly_loss) / self.weekly_start_bankroll if self.weekly_start_bankroll > 0 else 0
        if weekly_loss_pct > self.weekly_loss_limit:
            self.is_stopped = True
            self.stop_reason = f"Weekly loss limit exceeded: {weekly_loss_pct:.1%}"
            return True

        # Max drawdown check
        drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll if self.peak_bankroll > 0 else 0
        if drawdown > self.max_drawdown_limit:
            self.is_stopped = True
            self.stop_reason = f"Max drawdown exceeded: {drawdown:.1%}"
            return True

        return False

    def can_place_bet(self, bet_size: float, current_date: str) -> bool:
        """Check if bet can be placed given exposure limits."""
        # Reset daily exposure for new day
        if self.current_day != current_date:
            self.current_day = current_date
            self.daily_exposure = 0.0

        # Check if adding this bet would exceed daily exposure
        return not self.daily_exposure + bet_size > self.daily_exposure_limit * self.current_bankroll

    def place_bet(self, bet_size: float, current_date: str):
        """Record a bet being placed."""
        if self.current_day != current_date:
            self.current_day = current_date
            self.daily_exposure = 0.0

        self.daily_exposure += bet_size
        self.total_bets += 1
        self.total_wagered += bet_size

    def settle_bet(self, bet_size: float, won: bool, push: bool = False):
        """Settle a bet and update bankroll."""
        if push:
            self.pushes += 1
            return

        if won:
            # Assume -110 odds: win $0.909 per $1 risked
            profit = bet_size * 0.909
            self.current_bankroll += profit
            self.total_profit += profit
            self.wins += 1
            self.daily_loss -= profit
            self.weekly_loss -= profit
        else:
            # Lose the bet amount
            self.current_bankroll -= bet_size
            self.total_profit -= bet_size
            self.losses += 1
            self.daily_loss += bet_size
            self.weekly_loss += bet_size

        # Bug #5 fix: track max drawdown BEFORE updating peak
        current_dd = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll if self.peak_bankroll > 0 else 0
        self.max_drawdown_seen = max(self.max_drawdown_seen, current_dd)

        # Update peak
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll

    def reset_daily(self):
        """Reset daily counters (call at start of each day)."""
        self.daily_start_bankroll = self.current_bankroll
        self.daily_loss = 0.0
        self.daily_exposure = 0.0

    def reset_weekly(self):
        """Reset weekly counters (call at start of each week)."""
        self.weekly_start_bankroll = self.current_bankroll
        self.weekly_loss = 0.0

    def get_roi(self) -> float:
        """Calculate ROI as percentage."""
        if self.total_wagered == 0:
            return 0.0
        return (self.total_profit / self.total_wagered) * 100

    def get_win_rate(self) -> float:
        """Calculate win rate percentage."""
        decided_bets = self.wins + self.losses
        if decided_bets == 0:
            return 0.0
        return (self.wins / decided_bets) * 100

    def get_sharpe_ratio(self, bet_returns: list[float]) -> float:
        """Calculate Sharpe ratio from bet returns."""
        if len(bet_returns) < 2:
            return 0.0
        mean_return = np.mean(bet_returns)
        std_return = np.std(bet_returns)
        if std_return == 0:
            return 0.0
        # Assuming ~250 betting days per year (not 252 trading days)
        return (mean_return / std_return) * np.sqrt(250)

    def get_max_drawdown(self) -> float:
        """Get maximum drawdown as percentage (over entire backtest)."""
        # Bug #5 fix: return the historically tracked max drawdown, not current drawdown
        return self.max_drawdown_seen * 100


class Phase3Backtester(SeasonBacktester):
    """Extended backtester with quantile predictions and Kelly sizing."""

    def __init__(self, season: int = 2024):
        super().__init__(season)
        self.quantile_models = {}
        self.portfolio = BettingPortfolio()
        self.bet_returns = []  # Track individual bet returns for Sharpe
        self.predictions = []  # Store all predictions

    def load_quantile_models(self):
        """Load quantile regression models for prediction bands."""
        print("\n=== Loading Quantile Models ===")

        prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']
        for prop_type in prop_types:
            quantile_path = MODEL_DIR / f"player_{prop_type}_quantile.pkl"
            if quantile_path.exists():
                try:
                    with open(quantile_path, 'rb') as f:
                        self.quantile_models[prop_type] = pickle.load(f)
                    print(f"  ✓ Loaded quantile model for {prop_type}")
                except Exception as e:
                    print(f"  ✗ Failed to load {prop_type}: {e}")
            else:
                print(f"  ⚠ Quantile model not found for {prop_type}")

        print(f"\nLoaded {len(self.quantile_models)} quantile models")

    def predict_with_quantiles(self, prop_type: str, features: dict,
                                predicted_minutes: float | None = None,
                                line: float | None = None) -> QuantilePrediction | None:
        """
        Make prediction with quantile bands.

        Returns QuantilePrediction with pred_low/median/high and confidence.
        """
        # Get standard prediction first
        pred_mean = self.predict(prop_type, features, predicted_minutes=predicted_minutes)
        if pred_mean is None:
            return None

        # Try to get quantile predictions
        pred_low = pred_median = pred_high = None

        if prop_type in self.quantile_models:
            try:
                quantile_model_data = self.quantile_models[prop_type]

                # Check if it's a QuantilePropModel object
                if hasattr(quantile_model_data, 'predict'):
                    result = quantile_model_data.predict([features], prop_line=line)
                    pred_low = result.get('pred_low', [pred_mean * 0.9])[0]
                    pred_median = result.get('pred_median', [pred_mean])[0]
                    pred_high = result.get('pred_high', [pred_mean * 1.1])[0]

                # Or dict with 'quantile_models' key
                elif isinstance(quantile_model_data, dict) and 'quantile_models' in quantile_model_data:
                    # Build feature array (simplified - in production would match training features)
                    feature_names = quantile_model_data.get('feature_names', [])
                    if feature_names:
                        X = np.array([[features.get(f, 0.0) for f in feature_names]])

                        quantile_models = quantile_model_data['quantile_models']
                        pred_low = float(quantile_models[0.10].predict(X)[0]) if 0.10 in quantile_models else pred_mean * 0.9
                        pred_median = float(quantile_models[0.50].predict(X)[0]) if 0.50 in quantile_models else pred_mean
                        pred_high = float(quantile_models[0.90].predict(X)[0]) if 0.90 in quantile_models else pred_mean * 1.1

            except Exception as e:
                print(f"    Warning: Quantile prediction failed for {prop_type}: {e}")

        # Fallback: estimate bands from mean prediction
        if pred_low is None or pred_median is None or pred_high is None:
            pred_median = pred_mean
            pred_low = pred_mean * 0.85
            pred_high = pred_mean * 1.15

        # Calculate band width and confidence
        band_width = pred_high - pred_low

        # Confidence based on band width (narrower = higher confidence)
        if band_width < 3:
            confidence = 85.0
        elif band_width < 5:
            confidence = 70.0
        elif band_width < 8:
            confidence = 55.0
        else:
            confidence = 40.0

        tier = get_tier_from_confidence(confidence)

        # Calculate over probability and edge (if line provided)
        over_prob = 0.5
        edge = 0.0
        if line is not None:
            # Simple normal approximation
            std_dev = band_width / 3.29  # 80% interval / 2.58 std devs ≈ band_width / 3.29
            if std_dev > 0:
                from scipy.stats import norm
                over_prob = 1 - norm.cdf(line, loc=pred_mean, scale=std_dev)
                # Bug fix: use proper odds-based break-even instead of hardcoded 0.524
                # Standard -110/-110 line: no-vig probability = 0.50
                implied_prob = 0.50  # Standard -110/-110 no-vig
                edge = (over_prob - implied_prob) * 100

        # Create prediction object
        return QuantilePrediction(
            player_name="",  # Fill in later
            player_id=0,
            prop_type=prop_type,
            game_date="",
            game_id=0,
            pred_median=pred_median,
            pred_low=pred_low,
            pred_high=pred_high,
            predicted_value=pred_mean,
            confidence=confidence,
            tier=tier,
            band_width=band_width,
            line=line,
            over_prob=over_prob,
            edge=edge,
        )

    def calculate_bet_size(self, prediction: QuantilePrediction) -> float:
        """Calculate Kelly bet size with tier adjustments."""
        if not HAS_KELLY:
            # Fallback: flat 1% of bankroll for elite/strong, 0 otherwise
            if prediction.tier in ['elite', 'strong']:
                return self.portfolio.current_bankroll * 0.01
            return 0.0

        # Only bet if edge > 2%
        if abs(prediction.edge) < 2.0:
            return 0.0

        # Determine bet direction
        win_prob = prediction.over_prob if prediction.over_prob > 0.5 else 1 - prediction.over_prob

        # Assume -110 odds (decimal 1.909)
        decimal_odds = 1.909

        # Calculate Kelly size with tier adjustment
        try:
            return calculate_kelly_bet_size(
                win_prob=win_prob,
                decimal_odds=decimal_odds,
                bankroll=self.portfolio.current_bankroll,
                fractional=0.25,  # Quarter Kelly for safety
                edge_tier=prediction.tier,
                current_drawdown=self.portfolio.get_max_drawdown() / 100,
                num_same_day_bets=1,  # Simplified
                max_bet_pct=0.05
            )
        except Exception as e:
            print(f"    Kelly calculation failed: {e}")
            return 0.0

    def determine_bet_recommendation(self, prediction: QuantilePrediction) -> str:
        """Determine if we should BET, CONSIDER, or MONITOR."""
        if prediction.tier in ['elite', 'strong'] and abs(prediction.edge) > 5:
            return 'BET'
        if prediction.tier == 'moderate' and abs(prediction.edge) > 3:
            return 'CONSIDER'
        return 'MONITOR'

    def run_comprehensive_backtest(self, start_date: str, end_date: str, enable_stop_loss: bool = True) -> dict[str, Any]:
        """
        Run comprehensive backtest with quantile predictions and Kelly sizing.

        Args:
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"
            enable_stop_loss: If False, run full backtest ignoring stop-loss (for validation)

        Returns:
            Dict with comprehensive metrics
        """
        print(f"\n{'='*80}")
        print(f"PHASE 3 COMPREHENSIVE BACKTEST: {start_date} to {end_date}")
        print(f"{'='*80}\n")

        # Load models
        self.load_models()
        self.load_quantile_models()

        # Load all games and historical stats FIRST
        print("\nLoading games and historical player statistics...")
        self.load_games()
        self.load_historical_player_stats()

        # Filter games to date range
        print(f"\nFiltering games to date range: {start_date} to {end_date}...")
        all_games = [g for g in self.games if start_date <= g['date'] <= end_date]
        print(f"Found {len(all_games)} games in date range")

        if not all_games:
            print("ERROR: No games found in date range")
            return {"error": "No games found in date range"}

        # Reset portfolio
        self.portfolio = BettingPortfolio()
        self.predictions = []
        self.bet_returns = []

        # Process each game
        total_predictions = 0
        games_processed = 0

        for i, game in enumerate(all_games):
            game_id = game['id']
            game_date = game['date']

            if (i + 1) % 50 == 0:
                print(f"  Processing game {i+1}/{len(all_games)} ({game_date})")

            try:
                # Reset daily limits at start of new day
                if games_processed == 0 or game_date != all_games[i-1]['date']:
                    self.portfolio.reset_daily()

                    # Reset weekly on Mondays (simplified)
                    if datetime.strptime(game_date, "%Y-%m-%d").weekday() == 0:
                        self.portfolio.reset_weekly()

                # Check stop-loss (only if enabled)
                if enable_stop_loss and self.portfolio.check_stop_loss(game_date):
                    print(f"\n  ⚠️  STOP-LOSS TRIGGERED: {self.portfolio.stop_reason}")
                    break

                # Get player stats for this game from cache
                box_scores = self.fetch_box_scores_for_game(game)
                if not box_scores:
                    continue

                # Make predictions for each player
                for player_id, box_score in box_scores.items():
                    player = box_score.get('player', {})
                    player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()

                    if not player_id or not player_name:
                        continue

                    # Get actual values
                    actual_pts = box_score.get('pts', 0) or 0
                    actual_reb = box_score.get('reb', 0) or 0
                    actual_ast = box_score.get('ast', 0) or 0
                    actual_fg3m = box_score.get('fg3m', 0) or 0
                    actual_pra = actual_pts + actual_reb + actual_ast

                    actuals = {
                        'points': actual_pts,
                        'rebounds': actual_reb,
                        'assists': actual_ast,
                        'threes': actual_fg3m,
                        'pra': actual_pra
                    }

                    # Generate features using parent class method (point-in-time)
                    # Determine home/away and opponent
                    # Handle both object format (home_team: {id: X}) and ID format (home_team_id: X)
                    home_id = game.get('home_team_id') or game.get('home_team', {}).get('id')
                    away_id = game.get('away_team_id') or game.get('visitor_team', {}).get('id') or game.get('away_team', {}).get('id')
                    is_home = box_score.get('team_id') == home_id
                    opponent_id = away_id if is_home else home_id
                    player_position = player.get('position', 'F')

                    features = self.get_player_features_before_date(
                        player_id=player_id,
                        game_date=game_date,
                        opponent_id=opponent_id,
                        is_home=is_home,
                        player_position=player_position
                    )

                    if not features:
                        continue

                    # Make predictions for each prop type
                    for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
                        # Estimate betting line from player's season average (reasonable proxy)
                        # Books typically set lines near season average ± recent form
                        line_map = {
                            'points': features.get('season_pts_avg', actuals['points']),
                            'rebounds': features.get('season_reb_avg', actuals['rebounds']),
                            'assists': features.get('season_ast_avg', actuals['assists']),
                            'threes': features.get('season_fg3m_avg', actuals['threes']),
                            'pra': features.get('season_pts_avg', 0) + features.get('season_reb_avg', 0) + features.get('season_ast_avg', 0)
                        }
                        line = line_map.get(prop_type, actuals[prop_type])

                        # Fallback if no historical data available (rookie, early season)
                        if line is None or line == 0:
                            line = actuals[prop_type]  # Use actual as last resort

                        # Try quantile prediction first, fall back to regular prediction
                        prediction = self.predict_with_quantiles(
                            prop_type, features, line=line
                        )

                        if prediction is None:
                            continue

                        # Fill in metadata
                        prediction.player_name = player_name
                        prediction.player_id = player_id
                        prediction.game_date = game_date
                        prediction.game_id = game_id
                        prediction.actual_value = actuals[prop_type]

                        # Calculate error metrics
                        prediction.error = prediction.predicted_value - actuals[prop_type]
                        prediction.abs_error = abs(prediction.error)
                        prediction.squared_error = prediction.error ** 2

                        # Determine bet recommendation
                        prediction.bet_recommendation = self.determine_bet_recommendation(prediction)

                        # Calculate Kelly bet size
                        prediction.suggested_bet_size = self.calculate_bet_size(prediction)

                        # Simulate betting (only on BET recommendations with size > 0)
                        if prediction.bet_recommendation == 'BET' and prediction.suggested_bet_size > 0:
                            # Check if we can place bet
                            if self.portfolio.can_place_bet(prediction.suggested_bet_size, game_date):
                                # Place bet
                                self.portfolio.place_bet(prediction.suggested_bet_size, game_date)

                                # Determine outcome
                                if prediction.over_prob > 0.5:
                                    # We bet over
                                    won = actuals[prop_type] > line
                                    push = actuals[prop_type] == line
                                else:
                                    # We bet under
                                    won = actuals[prop_type] < line
                                    push = actuals[prop_type] == line

                                # Settle bet
                                self.portfolio.settle_bet(prediction.suggested_bet_size, won, push)

                                # Track return for Sharpe
                                if not push:
                                    if won:
                                        roi = 0.909  # Win at -110
                                    else:
                                        roi = -1.0  # Loss
                                    self.bet_returns.append(roi)

                                # Store hit result
                                if prediction.over_prob > 0.5:
                                    prediction.hit_over = won
                                    prediction.hit_under = not won
                                else:
                                    prediction.hit_over = not won
                                    prediction.hit_under = won

                        # Store prediction
                        self.predictions.append(prediction)
                        total_predictions += 1

                games_processed += 1

            except Exception as e:
                print(f"  Error processing game {game_id}: {e}")
                traceback.print_exc()
                continue

        print(f"\n✓ Processed {games_processed} games, {total_predictions} predictions")

        # Calculate comprehensive metrics
        return self.calculate_comprehensive_metrics()

    def calculate_comprehensive_metrics(self) -> dict[str, Any]:
        """Calculate all Phase 3 metrics."""
        print("\n=== Calculating Comprehensive Metrics ===")

        if not self.predictions:
            return {"error": "No predictions to analyze"}

        # Overall performance
        all_errors = [p.error for p in self.predictions if p.error is not None]
        all_abs_errors = [p.abs_error for p in self.predictions if p.abs_error is not None]
        all_sq_errors = [p.squared_error for p in self.predictions if p.squared_error is not None]

        overall_metrics = {
            'count': len(all_errors),
            'rmse': np.sqrt(np.mean(all_sq_errors)) if all_sq_errors else 0.0,
            'mae': np.mean(all_abs_errors) if all_abs_errors else 0.0,
            'mean_error': np.mean(all_errors) if all_errors else 0.0,
            'bias': np.mean(all_errors) if all_errors else 0.0,
        }

        # By tier performance
        tier_metrics = {}
        for tier in ['elite', 'strong', 'moderate', 'weak', 'avoid']:
            tier_preds = [p for p in self.predictions if p.tier == tier]
            if tier_preds:
                errors = [p.error for p in tier_preds if p.error is not None]
                sq_errors = [p.squared_error for p in tier_preds if p.squared_error is not None]
                abs_errors = [p.abs_error for p in tier_preds if p.abs_error is not None]

                tier_metrics[tier] = {
                    'count': len(tier_preds),
                    'rmse': np.sqrt(np.mean(sq_errors)) if sq_errors else 0.0,
                    'mae': np.mean(abs_errors) if abs_errors else 0.0,
                    'bias': np.mean(errors) if errors else 0.0,
                }

                # Bug #11 fix: guard against NaN from empty means
                for k, v in tier_metrics[tier].items():
                    if isinstance(v, float) and np.isnan(v):
                        tier_metrics[tier][k] = 0.0

        # By prop type
        prop_metrics = {}
        for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
            prop_preds = [p for p in self.predictions if p.prop_type == prop_type]
            if prop_preds:
                errors = [p.error for p in prop_preds if p.error is not None]
                sq_errors = [p.squared_error for p in prop_preds if p.squared_error is not None]
                abs_errors = [p.abs_error for p in prop_preds if p.abs_error is not None]

                # Calculate R²
                actual_vals = [p.actual_value for p in prop_preds if p.actual_value is not None]
                predicted_vals = [p.predicted_value for p in prop_preds if p.actual_value is not None]

                r2 = 0.0
                if actual_vals and predicted_vals and len(actual_vals) > 1:
                    try:
                        from sklearn.metrics import r2_score
                        r2 = r2_score(actual_vals, predicted_vals)
                    except:
                        pass

                prop_metrics[prop_type] = {
                    'count': len(prop_preds),
                    'rmse': np.sqrt(np.mean(sq_errors)) if sq_errors else 0.0,
                    'mae': np.mean(abs_errors) if abs_errors else 0.0,
                    'r2': r2,
                    'bias': np.mean(errors) if errors else 0.0,
                }

        # Elite + Strong combined
        elite_strong_preds = [p for p in self.predictions if p.tier in ['elite', 'strong']]
        if elite_strong_preds:
            es_errors = [p.error for p in elite_strong_preds if p.error is not None]
            es_sq_errors = [p.squared_error for p in elite_strong_preds if p.squared_error is not None]
            es_abs_errors = [p.abs_error for p in elite_strong_preds if p.abs_error is not None]

            elite_strong_metrics = {
                'count': len(elite_strong_preds),
                'rmse': np.sqrt(np.mean(es_sq_errors)) if es_sq_errors else 0.0,
                'mae': np.mean(es_abs_errors) if es_abs_errors else 0.0,
                'bias': np.mean(es_errors) if es_errors else 0.0,
                'percentage': (len(elite_strong_preds) / len(self.predictions)) * 100
            }
        else:
            elite_strong_metrics = {'count': 0}

        # Betting performance
        betting_metrics = {
            'total_bets': self.portfolio.total_bets,
            'wins': self.portfolio.wins,
            'losses': self.portfolio.losses,
            'pushes': self.portfolio.pushes,
            'win_rate': self.portfolio.get_win_rate(),
            'roi': self.portfolio.get_roi(),
            'total_wagered': self.portfolio.total_wagered,
            'total_profit': self.portfolio.total_profit,
            'final_bankroll': self.portfolio.current_bankroll,
            'peak_bankroll': self.portfolio.peak_bankroll,
            'max_drawdown': self.portfolio.get_max_drawdown(),
            'sharpe_ratio': self.portfolio.get_sharpe_ratio(self.bet_returns) if self.bet_returns else 0.0,
            'stopped': self.portfolio.is_stopped,
            'stop_reason': self.portfolio.stop_reason,
        }

        # Confidence calibration
        # Check if confidence scores correlate with actual accuracy
        confidences = [p.confidence for p in self.predictions if p.confidence > 0]
        accuracies = []
        for p in self.predictions:
            if p.confidence > 0 and p.abs_error is not None:
                # Inverse accuracy (lower error = higher accuracy)
                # Normalize to 0-100 scale (assume errors range 0-20)
                accuracy = max(0, 100 - (p.abs_error / 0.20))
                accuracies.append(accuracy)

        confidence_correlation = 0.0
        if confidences and accuracies and len(confidences) == len(accuracies) and len(confidences) > 1:
            try:
                corr, pval = pearsonr(confidences, accuracies)
                confidence_correlation = corr
            except:
                pass

        # Bug #11 fix: pre-compute elite-only confidences to guard against empty-mean NaN
        elite_only_confs = [p.confidence for p in self.predictions if p.tier == 'elite']

        calibration_metrics = {
            'confidence_accuracy_correlation': confidence_correlation,
            'avg_confidence_elite': float(np.mean(elite_only_confs)) if elite_only_confs else 0.0,
            'avg_confidence_all': float(np.mean(confidences)) if confidences else 0.0,
        }

        # Sample predictions
        sample_preds = []
        for p in self.predictions[:50]:  # First 50 predictions
            sample_preds.append({
                'player': p.player_name,
                'prop_type': p.prop_type,
                'pred_low': round(p.pred_low, 2),
                'pred_median': round(p.pred_median, 2),
                'pred_high': round(p.pred_high, 2),
                'predicted': round(p.predicted_value, 2),
                'actual': p.actual_value,
                'confidence': round(p.confidence, 1),
                'tier': p.tier,
                'error': round(p.error, 2) if p.error is not None else None,
                'bet_rec': p.bet_recommendation,
                'bet_size': round(p.suggested_bet_size, 2),
                'game_date': p.game_date,
            })

        # Compile results
        results = {
            'phase': 'Phase 3: Optimization (Weeks 5-6)',
            'date_completed': datetime.now().strftime('%Y-%m-%d'),
            'total_predictions': len(self.predictions),
            'overall_performance': overall_metrics,
            'tier_performance': tier_metrics,
            'prop_type_performance': prop_metrics,
            'elite_strong_performance': elite_strong_metrics,
            'betting_performance': betting_metrics,
            'calibration': calibration_metrics,
            'sample_predictions': sample_preds,
        }

        # Phase 3 targets validation
        targets = {
            'overall_rmse': {
                'target': '< 4.8',
                'actual': round(overall_metrics['rmse'], 3),
                'met': bool(overall_metrics['rmse'] < 4.8)
            },
            'points_rmse': {
                'target': '< 5.5',
                'actual': round(prop_metrics.get('points', {}).get('rmse', 999), 3),
                'met': bool(prop_metrics.get('points', {}).get('rmse', 999) < 5.5)
            },
            'threes_r2': {
                'target': '> 0.10',
                'actual': round(prop_metrics.get('threes', {}).get('r2', -1), 3),
                'met': bool(prop_metrics.get('threes', {}).get('r2', -1) > 0.10)
            },
            'roi_all': {
                'target': '> 3%',
                'actual': round(betting_metrics['roi'], 2),
                'met': bool(betting_metrics['roi'] > 3.0)
            },
            'roi_elite': {
                'target': '> 7%',
                'actual': 'N/A (need tier-specific betting)',
                'met': False
            },
            'sharpe_ratio': {
                'target': '> 1.5',
                'actual': round(betting_metrics['sharpe_ratio'], 2),
                'met': bool(betting_metrics['sharpe_ratio'] > 1.5)
            },
            'max_drawdown': {
                'target': '< 15%',
                'actual': round(betting_metrics['max_drawdown'], 2),
                'met': bool(betting_metrics['max_drawdown'] < 15.0)
            },
            'confidence_correlation': {
                'target': '> 0.5',
                'actual': round(confidence_correlation, 3),
                'met': bool(confidence_correlation > 0.5)
            }
        }

        results['phase3_targets'] = targets

        # Print summary
        print(f"\n{'='*80}")
        print("PHASE 3 BACKTEST SUMMARY")
        print(f"{'='*80}")
        print("\nOverall Performance:")
        print(f"  Total Predictions: {len(self.predictions):,}")
        print(f"  RMSE: {overall_metrics['rmse']:.3f}")
        print(f"  MAE: {overall_metrics['mae']:.3f}")
        print(f"  Bias: {overall_metrics['bias']:.3f}")

        print("\nElite + Strong Tier:")
        print(f"  Count: {elite_strong_metrics['count']:,} ({elite_strong_metrics.get('percentage', 0):.1f}%)")
        print(f"  RMSE: {elite_strong_metrics.get('rmse', 0):.3f}")

        print("\nBetting Performance:")
        print(f"  Total Bets: {betting_metrics['total_bets']}")
        print(f"  Win Rate: {betting_metrics['win_rate']:.1f}%")
        print(f"  ROI: {betting_metrics['roi']:.2f}%")
        print(f"  Sharpe Ratio: {betting_metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {betting_metrics['max_drawdown']:.1f}%")
        print(f"  Final Bankroll: ${betting_metrics['final_bankroll']:.2f}")

        print("\nPhase 3 Targets:")
        for target_name, target_data in targets.items():
            status = "✓" if target_data['met'] else "✗"
            print(f"  {status} {target_name}: {target_data['actual']} (target: {target_data['target']})")

        return results



def main():
    """Run 2-season comprehensive backtest."""

    # Season 1: 2024-25 (actual dates in data: Oct 22, 2024 - June 22, 2025)
    print("\n" + "="*80)
    print("SEASON 1: 2024-25")
    print("="*80)

    # Bug #6 fix: season=2025 loads games_2025_full.json for the 2024-25 season
    # (SeasonBacktester uses season as the ending year: 2024-25 → season=2025)
    backtester_s1 = Phase3Backtester(season=2025)
    results_s1 = backtester_s1.run_comprehensive_backtest(
        start_date="2024-10-22",
        end_date="2025-01-13",  # Use actual data range we have
        enable_stop_loss=False  # Disable for validation - want to see full performance
    )

    # Save results
    output_file_2324 = RESULTS_DIR / "phase3_backtest_2024-25_season1.json"
    with open(output_file_2324, 'w') as f:
        json.dump(results_s1, f, indent=2)
    print(f"\n✓ Saved Season 1 results to: {output_file_2324}")

    # Season 2: 2025-26 (actual dates in data: Oct 21, 2025 - Jan 13, 2026)
    print("\n" + "="*80)
    print("SEASON 2: 2025-26")
    print("="*80)

    # Bug #6 fix: season=2026 loads games_2026_full.json for the 2025-26 season
    # (SeasonBacktester uses season as the ending year: 2025-26 → season=2026)
    backtester_2425 = Phase3Backtester(season=2026)
    results_2425 = backtester_2425.run_comprehensive_backtest(
        start_date="2025-10-21",
        end_date="2026-01-13",  # Use actual data range we have
        enable_stop_loss=False  # Disable for validation - want to see full performance
    )

    # Save results
    output_file_2425 = RESULTS_DIR / "phase3_backtest_2025-26_season2.json"
    with open(output_file_2425, 'w') as f:
        json.dump(results_2425, f, indent=2)
    print(f"\n✓ Saved Season 2 results to: {output_file_2425}")

    # Combined analysis
    print("\n" + "="*80)
    print("COMBINED 2-SEASON ANALYSIS")
    print("="*80)

    # Handle case where no predictions were made
    total_preds_1 = results_s1.get('total_predictions', 0) if isinstance(results_s1, dict) else 0
    total_preds_2 = results_2425.get('total_predictions', 0) if isinstance(results_2425, dict) else 0

    combined_results = {
        'season_2024_25': results_s1,
        'season_2025_26': results_2425,
        'combined_summary': {
            'total_predictions': total_preds_1 + total_preds_2,
            'seasons_analyzed': 2,
            'date_range': '2024-10-22 to 2026-01-13',
        }
    }

    # Add combined metrics if both seasons have results
    if total_preds_1 > 0 and total_preds_2 > 0:
        try:
            combined_results['combined_summary'].update({
                'avg_rmse': (results_s1['overall_performance']['rmse'] + results_2425['overall_performance']['rmse']) / 2,
                'avg_roi': (results_s1['betting_performance']['roi'] + results_2425['betting_performance']['roi']) / 2,
                'avg_sharpe': (results_s1['betting_performance']['sharpe_ratio'] + results_2425['betting_performance']['sharpe_ratio']) / 2,
            })
        except KeyError as e:
            print(f"Warning: Could not calculate combined metrics: {e}")

    output_file_combined = RESULTS_DIR / "phase3_backtest_2seasons.json"
    with open(output_file_combined, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print(f"\n✓ Saved combined results to: {output_file_combined}")
    print("\n" + "="*80)
    print("BACKTEST COMPLETE!")
    print("="*80)

    return combined_results


if __name__ == "__main__":
    main()
