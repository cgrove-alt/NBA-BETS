"""
NBA Betting Model Orchestrator

Main application that orchestrates the complete betting workflow:
1. Fetch today's NBA schedule and data
2. Engineer features with injury and matchup analysis
3. Load trained ML models
4. Generate predictions for all bet types
5. Output comprehensive daily bet slip with betting strategy
"""

import load_env  # noqa: F401  — load .env before any code reads os.environ
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict

# Import scipy for proper probability calculations
try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Using approximate probability calculations.")


# Constants for spread-to-probability conversion
NBA_SPREAD_VOLATILITY = 13.0  # Historical standard deviation of NBA game margins
MIN_FEATURES_REQUIRED = 3  # Minimum non-zero features needed for valid prediction


def spread_edge_to_cover_probability(spread_edge: float, volatility: float = NBA_SPREAD_VOLATILITY) -> float:
    """
    Convert spread edge (in points) to cover probability using normal CDF.

    This is the mathematically correct way to convert point spreads to probabilities.
    NBA games have ~13 point standard deviation in final margins.

    Args:
        spread_edge: Model spread - Market spread (positive = model favors home more)
        volatility: Standard deviation of score margins (~13 for NBA)

    Returns:
        Probability of covering the spread (0.0 to 1.0)

    Examples:
        spread_edge=0: 50% (coin flip)
        spread_edge=6.5: ~69% (half a standard deviation)
        spread_edge=13: ~84% (one full standard deviation)
    """
    if HAS_SCIPY:
        return float(norm.cdf(spread_edge / volatility))
    # Approximate using logistic function if scipy not available
    return 1.0 / (1.0 + math.exp(-spread_edge / (volatility * 0.6)))


def determine_spread_bet_side(
    model_spread: float,
    market_spread: float,
) -> tuple[str, float, float]:
    """
    Determine which side to bet based on model vs market spread.

    CRITICAL: This fixes the sign convention bug where model was recommending
    opposite bets!

    Args:
        model_spread: Model's predicted home margin (+ = home wins by X)
        market_spread: Market spread for home team (- = home is favorite)
                       e.g., -12 means home must win by 13+ to cover

    Returns:
        Tuple of (side_to_bet, edge_in_points, cover_probability)

    Examples:
        model_spread=4.6, market_spread=-12.0:
        -> Model says home wins by 4.6
        -> Market says home -12 (home must win by 13+)
        -> Home WON'T cover (4.6 < 12), so bet AWAY +12
        -> Edge: 12 - 4.6 = 7.4 points for away

        model_spread=15.0, market_spread=-12.0:
        -> Model says home wins by 15
        -> Market says home -12
        -> Home WILL cover (15 > 12), so bet HOME -12
        -> Edge: 15 - 12 = 3 points for home
    """
    # The spread from home's perspective:
    # model_spread: positive means home wins by that many (home favored)
    # market_spread: negative means home is favored (e.g., -12 = home -12)

    # To cover, home needs: actual_margin > abs(market_spread) when home is favorite
    # market_spread = -12 means home needs to win by more than 12

    # Convert market spread to "points home must win by to cover"
    # If market_spread = -12, home must win by 13+ to cover
    # If market_spread = +5, home can lose by up to 4 and still cover
    home_cover_threshold = -market_spread  # e.g., -(-12) = 12 points needed

    # Model says home wins by model_spread
    # If model_spread > home_cover_threshold: bet HOME (they'll cover)
    # If model_spread < home_cover_threshold: bet AWAY (home won't cover)

    if model_spread > home_cover_threshold:
        # Model predicts home covers
        side = "home"
        edge_points = model_spread - home_cover_threshold
        cover_prob = spread_edge_to_cover_probability(edge_points)
    else:
        # Model predicts away covers (home doesn't cover)
        side = "away"
        edge_points = home_cover_threshold - model_spread
        cover_prob = spread_edge_to_cover_probability(edge_points)

    return side, edge_points, cover_prob


def validate_features(features: dict, feature_type: str = "game") -> tuple[bool, str]:
    """
    Validate that features were actually generated (not all zeros/defaults).

    This prevents the model from making predictions based on failed feature
    generation, which was causing 6/9 games to have identical predictions.

    Args:
        features: Dictionary of features
        feature_type: "game" or "player" for appropriate validation

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    if not features:
        return False, "Empty features dictionary"

    # Count non-zero, non-default numeric features
    non_zero_count = 0
    key_features = []

    if feature_type == "game":
        key_features = [
            "net_rating_diff", "off_rating_diff", "def_rating_diff",
            "home_recent_win_pct", "away_recent_win_pct",
            "home_pts_avg", "away_pts_avg",
        ]
    else:  # player
        key_features = [
            "season_pts_avg", "recent_pts_avg", "min_avg",
            "season_reb_avg", "season_ast_avg",
        ]

    for key in key_features:
        val = features.get(key, 0)
        if isinstance(val, (int, float)) and abs(val) > 0.01:
            non_zero_count += 1

    if non_zero_count < MIN_FEATURES_REQUIRED:
        return False, f"Only {non_zero_count}/{len(key_features)} key features have values (need {MIN_FEATURES_REQUIRED}+)"

    return True, "Features validated"

from data_fetcher import (
    fetch_todays_schedule,
    parse_game_details,
)
from feature_engineering import (
    generate_game_features,
    generate_player_features,
    create_injury_report,
    InjuryReportManager,
)
from model_trainer import (
    ModelTrainingPipeline,
    SpreadCoverClassifier,
)

# Import real odds and injury integrations
try:
    from odds_fetcher import OddsFetcher, get_nba_odds, get_best_odds, find_value_bets, LineMovementTracker, CLVTracker
    HAS_ODDS_FETCHER = True
except ImportError:
    HAS_ODDS_FETCHER = False
    print("Note: odds_fetcher.py not available. Using default odds.")

try:
    from injury_fetcher import InjuryFetcher, get_injuries_for_game, get_spread_adjustment
    HAS_INJURY_FETCHER = True
except ImportError:
    HAS_INJURY_FETCHER = False
    print("Note: injury_fetcher.py not available. Skipping injury adjustments.")


try:
    from balldontlie_api import BalldontlieAPI, format_odds_for_model
    HAS_BALLDONTLIE = True
except ImportError:
    HAS_BALLDONTLIE = False

try:
    from calibration import ModelCalibrator
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False

try:
    from bet_tracker import BetTracker, TrackedBet, BetType as TrackerBetType
    HAS_BET_TRACKER = True
except ImportError:
    HAS_BET_TRACKER = False

try:
    from edge_quality import EdgeQualityScorer, EdgeQualityResult, EdgeTier, DynamicKellyCalculator
    HAS_EDGE_QUALITY = True
except ImportError:
    HAS_EDGE_QUALITY = False


@dataclass
class BetRecommendation:
    """Represents a single bet recommendation."""
    bet_type: str  # "moneyline", "spread", "total", "prop", "parlay"
    description: str
    selection: str
    line: float | None = None
    probability: float = 0.0
    confidence: str = "low"  # "low", "medium", "high"
    edge: float = 0.0
    expected_value: float = 0.0
    recommended_stake: float = 0.0
    reasoning: str = ""
    game_info: dict = field(default_factory=dict)
    # Real odds fields
    odds: float = -110  # American odds
    implied_probability: float = 0.524  # Implied from odds
    sportsbook: str = ""  # Where to place the bet
    closing_line_value: float | None = None  # CLV if available
    # Edge quality scoring
    edge_quality_score: float | None = None  # 0-100 score
    edge_quality_tier: str | None = None  # "elite", "strong", "moderate", "weak", "avoid"
    edge_quality_factors: list[str] = field(default_factory=list)  # Key factors


@dataclass
class GameAnalysis:
    """Complete analysis for a single game."""
    game_id: str
    home_team: str
    away_team: str
    game_time: str
    features: dict = field(default_factory=dict)
    features_valid: bool = False  # Whether features were successfully generated
    moneyline_prediction: dict = field(default_factory=dict)
    spread_prediction: dict = field(default_factory=dict)
    total_prediction: dict = field(default_factory=dict)
    player_props: list[dict] = field(default_factory=list)
    recommendations: list[BetRecommendation] = field(default_factory=list)
    # Real market data
    market_odds: dict = field(default_factory=dict)  # Real odds from sportsbooks
    injury_impact: dict = field(default_factory=dict)  # Injury adjustments
    best_odds: dict = field(default_factory=dict)  # Best available odds by market


@dataclass
class DailyBetSlip:
    """Complete daily betting recommendations."""
    date: str
    generated_at: str
    games_analyzed: int
    total_recommendations: int
    top_picks: list[BetRecommendation] = field(default_factory=list)
    game_analyses: list[GameAnalysis] = field(default_factory=list)
    parlay_recommendations: list[dict] = field(default_factory=list)
    bankroll_allocation: dict = field(default_factory=dict)


class BettingStrategy:
    """
    Sophisticated betting strategy for generating recommendations.

    Uses Kelly Criterion modified for sports betting with additional
    confidence filters and bankroll management.
    """

    # Minimum thresholds for recommendations
    MIN_EDGE = 0.03  # 3% minimum edge
    MIN_PROBABILITY = 0.52  # Minimum win probability

    # Confidence levels based on edge
    CONFIDENCE_THRESHOLDS = {
        "high": 0.08,    # 8%+ edge
        "medium": 0.05,  # 5-8% edge
        "low": 0.03,     # 3-5% edge
    }

    # Maximum stake percentages (of bankroll)
    MAX_STAKE_PCT = {
        "high": 0.05,    # 5% max for high confidence
        "medium": 0.03,  # 3% max for medium confidence
        "low": 0.01,     # 1% max for low confidence
    }

    # Bet type weights for diversification
    BET_TYPE_WEIGHTS = {
        "moneyline": 1.0,
        "spread": 0.9,
        "total": 0.8,
        "prop": 0.7,
        "parlay": 0.5,
    }

    def __init__(self, bankroll: float = 1000.0, risk_tolerance: str = "moderate"):
        """
        Initialize betting strategy.

        Args:
            bankroll: Total bankroll for betting
            risk_tolerance: "conservative", "moderate", or "aggressive"
        """
        self.bankroll = bankroll
        self.risk_tolerance = risk_tolerance

        # Adjust thresholds based on risk tolerance
        self._adjust_for_risk_tolerance()

    def _adjust_for_risk_tolerance(self):
        """Adjust strategy parameters based on risk tolerance."""
        if self.risk_tolerance == "conservative":
            self.MIN_EDGE = 0.05
            self.MIN_PROBABILITY = 0.55
            self.kelly_fraction = 0.25
        elif self.risk_tolerance == "aggressive":
            self.MIN_EDGE = 0.02
            self.MIN_PROBABILITY = 0.51
            self.kelly_fraction = 0.5
        else:  # moderate
            self.MIN_EDGE = 0.03
            self.MIN_PROBABILITY = 0.52
            self.kelly_fraction = 0.33

    def calculate_kelly_stake(
        self,
        probability: float,
        odds: float,
        confidence: str = "medium",
    ) -> float:
        """
        Calculate recommended stake using fractional Kelly Criterion.

        Args:
            probability: Model's win probability
            odds: American odds
            confidence: Confidence level

        Returns:
            Recommended stake amount
        """
        # Convert American odds to decimal
        decimal_odds = odds / 100 + 1 if odds > 0 else 100 / abs(odds) + 1

        # Kelly formula: (bp - q) / b
        # where b = decimal odds - 1, p = win probability, q = 1 - p
        b = decimal_odds - 1
        p = probability
        q = 1 - p

        kelly = (b * p - q) / b if b > 0 else 0

        # Apply fractional Kelly
        kelly = kelly * self.kelly_fraction

        # Cap by confidence level
        max_stake_pct = self.MAX_STAKE_PCT.get(confidence, 0.02)
        stake_pct = min(kelly, max_stake_pct)

        # Ensure non-negative
        stake_pct = max(0, stake_pct)

        return round(self.bankroll * stake_pct, 2)

    def evaluate_bet(
        self,
        probability: float,
        implied_probability: float,
        bet_type: str = "moneyline",
    ) -> dict[str, Any]:
        """
        Evaluate a potential bet.

        Args:
            probability: Model's predicted probability
            implied_probability: Probability implied by odds
            bet_type: Type of bet

        Returns:
            Dictionary with evaluation metrics
        """
        edge = probability - implied_probability

        # Determine confidence level
        if edge >= self.CONFIDENCE_THRESHOLDS["high"]:
            confidence = "high"
        elif edge >= self.CONFIDENCE_THRESHOLDS["medium"]:
            confidence = "medium"
        elif edge >= self.CONFIDENCE_THRESHOLDS["low"]:
            confidence = "low"
        else:
            confidence = "none"

        # Apply bet type weight
        weighted_edge = edge * self.BET_TYPE_WEIGHTS.get(bet_type, 1.0)

        # Calculate expected value (per $100 wagered)
        ev = (probability * 100) - ((1 - probability) * 100)

        # Recommendation
        is_recommended = (
            edge >= self.MIN_EDGE and
            probability >= self.MIN_PROBABILITY and
            confidence != "none"
        )

        return {
            "edge": edge,
            "weighted_edge": weighted_edge,
            "confidence": confidence,
            "expected_value": ev,
            "is_recommended": is_recommended,
            "probability": probability,
            "implied_probability": implied_probability,
        }

    def generate_parlay_strategy(
        self,
        legs: list[dict],
        max_legs: int = 4,
    ) -> list[dict]:
        """
        Generate optimal parlay combinations from available bets.

        Args:
            legs: List of potential parlay legs
            max_legs: Maximum legs per parlay

        Returns:
            List of recommended parlay combinations
        """
        parlays = []

        # Filter to only high-confidence legs
        quality_legs = [
            leg for leg in legs
            if leg.get("confidence") in ["high", "medium"]
            and leg.get("probability", 0) >= 0.55
        ]

        if len(quality_legs) < 2:
            return parlays

        # Sort by edge
        quality_legs.sort(key=lambda x: x.get("edge", 0), reverse=True)

        # Generate 2-leg parlays from top picks
        if len(quality_legs) >= 2:
            top_legs = quality_legs[:4]
            for i in range(len(top_legs)):
                for j in range(i + 1, len(top_legs)):
                    combined_prob = top_legs[i]["probability"] * top_legs[j]["probability"]
                    if combined_prob >= 0.30:  # Minimum 30% combined probability
                        parlays.append({
                            "legs": [top_legs[i], top_legs[j]],
                            "combined_probability": combined_prob,
                            "num_legs": 2,
                            "type": "2-leg parlay",
                        })

        # Generate 3-leg parlays from very high confidence legs
        high_conf_legs = [l for l in quality_legs if l.get("confidence") == "high"]
        if len(high_conf_legs) >= 3:
            combined_prob = 1.0
            for leg in high_conf_legs[:3]:
                combined_prob *= leg["probability"]

            if combined_prob >= 0.20:  # Minimum 20% for 3-leg
                parlays.append({
                    "legs": high_conf_legs[:3],
                    "combined_probability": combined_prob,
                    "num_legs": 3,
                    "type": "3-leg parlay",
                })

        return parlays

    def allocate_bankroll(
        self,
        recommendations: list[BetRecommendation],
    ) -> dict[str, Any]:
        """
        Allocate bankroll across recommendations.

        Args:
            recommendations: List of bet recommendations

        Returns:
            Bankroll allocation summary
        """
        total_stake = 0
        allocation = []

        # Sort by confidence and edge
        sorted_recs = sorted(
            recommendations,
            key=lambda x: (
                {"high": 3, "medium": 2, "low": 1}.get(x.confidence, 0),
                x.edge
            ),
            reverse=True
        )

        # Allocate stakes
        remaining_bankroll = self.bankroll * 0.2  # Max 20% of bankroll per day

        for rec in sorted_recs:
            if remaining_bankroll <= 0:
                break

            stake = min(rec.recommended_stake, remaining_bankroll)
            if stake > 0:
                allocation.append({
                    "bet": rec.description,
                    "stake": stake,
                    "confidence": rec.confidence,
                })
                total_stake += stake
                remaining_bankroll -= stake

        return {
            "total_stake": total_stake,
            "num_bets": len(allocation),
            "allocation": allocation,
            "remaining_daily_budget": remaining_bankroll,
            "bankroll_percentage_used": (total_stake / self.bankroll) * 100,
        }


class Orchestrator:
    """
    Main orchestrator for the NBA betting model workflow.

    Coordinates data fetching, feature engineering, model predictions,
    and bet slip generation.
    """

    def __init__(
        self,
        season: str = "2025-26",
        bankroll: float = 1000.0,
        risk_tolerance: str = "moderate",
    ):
        """
        Initialize the orchestrator.

        Args:
            season: NBA season
            bankroll: Betting bankroll
            risk_tolerance: Risk tolerance level
        """
        self.season = season
        self.bankroll = bankroll
        self.risk_tolerance = risk_tolerance

        # Initialize components
        self.strategy = BettingStrategy(bankroll, risk_tolerance)
        self.pipeline = ModelTrainingPipeline(season)
        self.injury_manager = InjuryReportManager(season)

        # Initialize real odds fetcher
        self.odds_fetcher = None
        if HAS_ODDS_FETCHER:
            try:
                self.odds_fetcher = OddsFetcher()
                print("Real odds integration enabled")
            except Exception as e:
                print(f"Could not initialize odds fetcher: {e}")

        # Initialize injury fetcher
        self.injury_fetcher = None
        if HAS_INJURY_FETCHER:
            try:
                self.injury_fetcher = InjuryFetcher()
                print("Real injury data integration enabled")
            except Exception as e:
                print(f"Could not initialize injury fetcher: {e}")

        # Initialize bet tracker
        self.bet_tracker = None
        if HAS_BET_TRACKER:
            try:
                self.bet_tracker = BetTracker()
                print("Bet tracking enabled")
            except Exception as e:
                print(f"Could not initialize bet tracker: {e}")

        # Initialize Balldontlie (preferred premium data provider)
        self.balldontlie = None
        if HAS_BALLDONTLIE:
            try:
                self.balldontlie = BalldontlieAPI()
                print("Balldontlie premium data enabled (odds, live stats, injuries)")
            except ValueError:
                # API key not set - that's OK, will use free sources
                pass
            except Exception as e:
                print(f"Could not initialize Balldontlie: {e}")

        # State
        self.models_loaded = False
        self.schedule = []
        self.game_analyses = []
        self.current_odds = {}  # Cache for real odds
        self.injuries_cache = []  # Balldontlie injuries cache

        # Line movement and CLV tracking
        self.line_tracker = None
        self.clv_tracker = None
        if HAS_ODDS_FETCHER:
            try:
                self.line_tracker = LineMovementTracker(storage_dir="odds_history")
                self.clv_tracker = CLVTracker(self.odds_fetcher)
                print("Line movement and CLV tracking enabled")
            except Exception as e:
                print(f"Could not initialize line tracking: {e}")

        # Probability calibrators for improved betting edge calculation
        self.moneyline_calibrator = None
        self.spread_calibrator = None
        self.prop_calibrators = {}  # Dict of prop_type -> ModelCalibrator

        # Edge quality scoring for bet filtering and stake sizing
        self.edge_quality_scorer = None
        self.dynamic_kelly = None
        if HAS_EDGE_QUALITY:
            try:
                self.edge_quality_scorer = EdgeQualityScorer(min_edge_threshold=0.02)
                self.dynamic_kelly = DynamicKellyCalculator(
                    base_kelly_fraction=0.25,  # Quarter Kelly as base
                    max_bet_pct=0.05,          # Max 5% of bankroll
                    min_bet_pct=0.005,         # Min 0.5% of bankroll
                )
                print("Edge quality scoring enabled")
            except Exception as e:
                print(f"Could not initialize edge quality scoring: {e}")

    def load_models(self) -> bool:
        """
        Load trained ML models and calibrators.

        Returns:
            True if models loaded successfully
        """
        try:
            self.pipeline.load_all_models()
            self.models_loaded = len(self.pipeline.models) > 0

            if self.models_loaded:
                print(f"Loaded {len(self.pipeline.models)} models:")
                for name in self.pipeline.models:
                    print(f"  - {name}")
            else:
                print("No trained models found. Using feature-based predictions.")

            # Load probability calibrators for improved edge calculation
            self._load_calibrators()

            return self.models_loaded
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Falling back to feature-based predictions.")
            return False

    def _load_calibrators(self):
        """Load probability calibrators for moneyline, spread, and prop models."""
        import logging

        try:
            from calibration import ModelCalibrator
            calibration_dir = Path("models/calibration")

            if calibration_dir.exists():
                # Load moneyline calibrator
                try:
                    self.moneyline_calibrator = ModelCalibrator("moneyline")
                    self.moneyline_calibrator.load(str(calibration_dir))
                    print(f"  - moneyline calibrator ({self.moneyline_calibrator.best_method})")
                except Exception as e:
                    logging.warning(f"Moneyline calibrator not loaded: {e}")
                    print("  - moneyline calibrator not available (using uncalibrated)")
                    self.moneyline_calibrator = None

                # Load spread calibrator
                try:
                    self.spread_calibrator = ModelCalibrator("spread")
                    self.spread_calibrator.load(str(calibration_dir))
                    print(f"  - spread calibrator ({self.spread_calibrator.best_method})")
                except Exception as e:
                    logging.warning(f"Spread calibrator not loaded: {e}")
                    print("  - spread calibrator not available (using uncalibrated)")
                    self.spread_calibrator = None

                # Load prop calibrators (per type)
                self.prop_calibrators = {}
                prop_types = ["points", "rebounds", "assists", "threes", "pra"]
                for prop_type in prop_types:
                    try:
                        cal = ModelCalibrator(f"prop_{prop_type}")
                        cal.load(str(calibration_dir))
                        self.prop_calibrators[prop_type] = cal
                        print(f"  - prop_{prop_type} calibrator ({cal.best_method})")
                    except Exception:
                        logging.warning(f"Prop {prop_type} calibrator not found - using uncalibrated")
                        # Don't print for each missing prop calibrator to reduce noise

                if not self.prop_calibrators:
                    print("  - prop calibrators not available (using uncalibrated)")
            else:
                logging.warning("No calibrators found (models/calibration/ not present)")
                print("  No calibrators found (models/calibration/ not present)")
                self.prop_calibrators = {}
        except ImportError:
            logging.warning("Calibration module not available")
            print("  Calibration module not available")
            self.prop_calibrators = {}
        except Exception as e:
            logging.warning(f"Error loading calibrators: {e}")
            print(f"  Error loading calibrators: {e}")
            self.prop_calibrators = {}

    def _calibrate_moneyline(self, prediction: dict) -> dict:
        """
        Apply calibration to moneyline probability prediction.

        Calibration improves betting edge calculation by correcting for
        model overconfidence/underconfidence.

        Args:
            prediction: Model prediction with home_win_probability

        Returns:
            Prediction with calibrated probability
        """
        if self.moneyline_calibrator is None:
            return prediction

        try:
            raw_prob = prediction.get("home_win_probability", 0.5)
            calibrated_prob = self.moneyline_calibrator.calibrate(raw_prob)

            # Update prediction with calibrated probability
            prediction["home_win_probability"] = float(calibrated_prob)
            prediction["away_win_probability"] = 1.0 - float(calibrated_prob)
            prediction["predicted_winner"] = "home" if calibrated_prob > 0.5 else "away"
            prediction["confidence"] = float(max(calibrated_prob, 1 - calibrated_prob))
            prediction["calibrated"] = True
            prediction["raw_probability"] = raw_prob
        except Exception:
            pass  # Return original prediction if calibration fails

        return prediction

    def _calibrate_spread(self, prediction: dict) -> dict:
        """
        Apply calibration to spread prediction.

        For spread models, we calibrate the cover probability derived from
        the predicted point differential.

        Args:
            prediction: Model prediction with predicted_spread and cover_probability

        Returns:
            Prediction with calibrated cover probability
        """
        if self.spread_calibrator is None:
            return prediction

        try:
            # Get cover probability if available, otherwise compute from predicted spread
            cover_prob = prediction.get("cover_probability")
            if cover_prob is None:
                # Estimate cover probability from predicted spread vs market spread
                predicted = prediction.get("predicted_spread", 0)
                spread_line = prediction.get("spread_line", 0)
                edge = predicted - spread_line
                # Simple sigmoid: edge -> probability
                import math
                cover_prob = 1 / (1 + math.exp(-edge / 5.0))

            calibrated_prob = self.spread_calibrator.calibrate(cover_prob)

            # Update prediction with calibrated probability
            prediction["cover_probability"] = float(calibrated_prob)
            prediction["calibrated"] = True
            prediction["raw_cover_probability"] = cover_prob
        except Exception:
            pass  # Return original prediction if calibration fails

        return prediction

    def set_injuries(self, injury_data: list[dict]):
        """
        Set injury report data.

        Args:
            injury_data: List of team injury reports
        """
        self.injury_manager = create_injury_report(injury_data, self.season)
        print(f"Injury report set for {len(injury_data)} teams")

    def fetch_schedule(self) -> list[dict]:
        """
        Fetch today's NBA schedule.

        Returns:
            List of scheduled games
        """
        try:
            games_data, date = fetch_todays_schedule()
            self.schedule = parse_game_details(games_data)
            print(f"Found {len(self.schedule)} games scheduled for {date}")
            return self.schedule
        except Exception as e:
            print(f"Error fetching schedule: {e}")
            return []

    def fetch_real_odds(self) -> dict[str, Any]:
        """
        Fetch real betting odds from sportsbooks.

        Returns:
            Dictionary with odds for all NBA games
        """
        if not self.odds_fetcher:
            print("Real odds not available. Using default -110 odds.")
            return {}

        try:
            print("Fetching real odds from sportsbooks...")
            odds_data = self.odds_fetcher.get_nba_odds()

            # Process and store
            self.current_odds = {}
            for game_odds in odds_data:
                # Create key from team names
                home = game_odds.get("home_team", "")
                away = game_odds.get("away_team", "")
                key = f"{away}@{home}"

                self.current_odds[key] = {
                    "moneyline": {
                        "home": game_odds.get("home_odds", -110),
                        "away": game_odds.get("away_odds", -110),
                    },
                    "spread": {
                        "home_line": game_odds.get("spread_home", -3.5),
                        "home_odds": game_odds.get("spread_home_odds", -110),
                        "away_line": game_odds.get("spread_away", 3.5),
                        "away_odds": game_odds.get("spread_away_odds", -110),
                    },
                    "total": {
                        "line": game_odds.get("total", 220.0),
                        "over_odds": game_odds.get("over_odds", -110),
                        "under_odds": game_odds.get("under_odds", -110),
                    },
                    "sportsbook": game_odds.get("sportsbook", "Unknown"),
                    "last_updated": game_odds.get("commence_time", ""),
                }

            print(f"Loaded odds for {len(self.current_odds)} games")
            return self.current_odds

        except Exception as e:
            print(f"Error fetching odds: {e}")
            return {}

    def _estimate_spread_odds(self, spread_line: float) -> tuple[int, int]:
        """
        Estimate realistic spread odds when real odds unavailable.

        In reality, larger spreads have worse odds for favorites and
        better odds for underdogs (vig adjustment).

        Args:
            spread_line: The home spread line (negative = home favored)

        Returns:
            Tuple of (favorite_odds, underdog_odds)
        """
        abs_spread = abs(spread_line)

        if abs_spread <= 3.0:
            # Close games: standard -110 / -110
            return -110, -110
        if abs_spread <= 6.0:
            # Medium spreads: slight adjustment
            return -115, -105
        if abs_spread <= 10.0:
            # Large spreads: bigger adjustment
            return -120, 100
        # Very large spreads: heavy adjustment
        return -130, 110

    def _evaluate_edge_quality(
        self,
        model_probability: float,
        implied_probability: float,
        bet_type: str,
        home_away: str = "home",
        analysis: Optional['GameAnalysis'] = None,
        line_movement: dict | None = None,
    ) -> Optional['EdgeQualityResult']:
        """
        Evaluate edge quality for a bet using the EdgeQualityScorer.

        Args:
            model_probability: Model's predicted probability
            implied_probability: Market implied probability
            bet_type: "moneyline", "spread", or "total"
            home_away: "home" or "away" for the side being bet
            analysis: GameAnalysis with additional context
            line_movement: Line movement data if available

        Returns:
            EdgeQualityResult or None if edge quality scoring unavailable
        """
        if not self.edge_quality_scorer:
            return None

        # Extract data from analysis if available
        injury_impact = 0.0
        games_played = 30  # Default
        is_back_to_back = False

        if analysis and hasattr(analysis, 'injury_impact'):
            injury_data = analysis.injury_impact or {}
            # Normalize injury impact to 0-1 scale
            spread_adj = abs(injury_data.get("spread_adjustment", 0))
            injury_impact = min(1.0, spread_adj / 10.0)  # 10pt adjustment = max impact

        # Extract line movement data
        opening_odds = None
        current_odds = None
        if line_movement:
            spread_move = line_movement.get("movements", {}).get(bet_type, {})
            if spread_move:
                opening_odds = spread_move.get("opening")
                current_odds = spread_move.get("current")

        try:
            return self.edge_quality_scorer.evaluate_edge(
                model_probability=model_probability,
                implied_probability=implied_probability,
                opening_odds=opening_odds,
                current_odds=current_odds,
                home_away=home_away,
                injury_impact_score=injury_impact,
                games_played=games_played,
                is_back_to_back=is_back_to_back,
                training_data_age_days=30.0,  # Default - could be computed from model metadata
                last_game_days_ago=2.0,       # Default
            )
        except Exception as e:
            print(f"Edge quality evaluation failed: {e}")
            return None

    def get_game_odds(self, home_abbrev: str, away_abbrev: str) -> dict[str, Any]:
        """
        Get real odds for a specific game.

        Args:
            home_abbrev: Home team abbreviation
            away_abbrev: Away team abbreviation

        Returns:
            Dictionary with odds data or default odds
        """
        # Default odds structure
        default_odds = {
            "moneyline": {"home": -110, "away": -110},
            "spread": {"home_line": -3.5, "home_odds": -110, "away_line": 3.5, "away_odds": -110},
            "total": {"line": 220.0, "over_odds": -110, "under_odds": -110},
            "sportsbook": "Default",
        }

        if not self.current_odds:
            return default_odds

        # Try different key formats
        keys_to_try = [
            f"{away_abbrev}@{home_abbrev}",
            f"{away_abbrev} @ {home_abbrev}",
            f"{away_abbrev.lower()}@{home_abbrev.lower()}",
        ]

        for key in keys_to_try:
            if key in self.current_odds:
                return self.current_odds[key]

        # Try fuzzy matching by team names
        for key, odds in self.current_odds.items():
            if home_abbrev.lower() in key.lower() and away_abbrev.lower() in key.lower():
                return odds

        return default_odds

    def _format_odds_for_tracker(
        self,
        game_odds: dict,
        home_abbrev: str,
        away_abbrev: str
    ) -> dict:
        """
        Format odds data for LineMovementTracker.

        Args:
            game_odds: Odds dictionary from get_game_odds()
            home_abbrev: Home team abbreviation
            away_abbrev: Away team abbreviation

        Returns:
            Formatted dictionary for line tracker
        """
        ml = game_odds.get("moneyline", {})
        spread = game_odds.get("spread", {})
        total = game_odds.get("total", {})

        return {
            "timestamp": datetime.now().isoformat(),
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "sportsbook": game_odds.get("sportsbook", "Unknown"),
            "moneyline": {
                "home": ml.get("home", -110),
                "away": ml.get("away", -110),
            },
            "spread": {
                "home_line": spread.get("home_line", 0),
                "home_odds": spread.get("home_odds", -110),
                "away_line": spread.get("away_line", 0),
                "away_odds": spread.get("away_odds", -110),
            },
            "total": {
                "line": total.get("line", 220),
                "over_odds": total.get("over_odds", -110),
                "under_odds": total.get("under_odds", -110),
            },
        }

    def fetch_balldontlie_odds(self, date: str = None) -> dict[str, Any]:
        """
        Fetch betting odds from Balldontlie API (GOAT tier required).

        This provides:
        - Real-time betting odds from multiple sportsbooks
        - Moneyline, spread, and total markets
        - Player prop odds

        Args:
            date: Date in YYYY-MM-DD format (defaults to today)

        Returns:
            Dictionary with odds data by game
        """
        if not self.balldontlie:
            return {"odds": [], "error": "Balldontlie not configured"}

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        result = {"date": date, "odds": [], "games_with_odds": 0}

        try:
            print(f"Fetching Balldontlie odds for {date}...")

            # First fetch games to get team abbreviations (odds only have game_id)
            games_data = self.balldontlie.get_games(dates=[date])
            game_lookup = {}
            for game in games_data:
                gid = game.get("id")
                home = game.get("home_team", {}).get("abbreviation", "")
                away = game.get("visitor_team", {}).get("abbreviation", "")
                game_lookup[gid] = {"home": home, "away": away}

            # Now fetch odds
            odds_data = self.balldontlie.get_betting_odds(date=date)
            result["odds"] = odds_data

            # Group odds by game_id (may have multiple sportsbooks)
            odds_by_game = {}
            for odds in odds_data:
                gid = odds.get("game_id")
                if gid not in odds_by_game:
                    odds_by_game[gid] = odds  # Use first sportsbook

            result["games_with_odds"] = len(odds_by_game)

            # Process and merge with current_odds
            for game_id, odds in odds_by_game.items():
                teams = game_lookup.get(game_id, {})
                home_team = teams.get("home", "")
                away_team = teams.get("away", "")

                if not home_team or not away_team:
                    continue

                key = f"{away_team}@{home_team}"

                # Balldontlie uses flat structure for odds
                self.current_odds[key] = {
                    "moneyline": {
                        "home": odds.get("moneyline_home_odds", -110),
                        "away": odds.get("moneyline_away_odds", -110),
                    },
                    "spread": {
                        "home_line": odds.get("spread_home_value", -3.5),
                        "home_odds": odds.get("spread_home_odds", -110),
                        "away_line": odds.get("spread_away_value", 3.5),
                        "away_odds": odds.get("spread_away_odds", -110),
                    },
                    "total": {
                        "line": odds.get("total_value", 220.0),
                        "over_odds": odds.get("total_over_odds", -110),
                        "under_odds": odds.get("total_under_odds", -110),
                    },
                    "sportsbook": odds.get("vendor", "Balldontlie"),
                }

            print(f"  Loaded odds for {result['games_with_odds']} games from Balldontlie")
            return result

        except Exception as e:
            print(f"Error fetching Balldontlie odds: {e}")
            result["error"] = str(e)
            return result

    def fetch_balldontlie_injuries(self) -> list[dict]:
        """
        Fetch current injury data from Balldontlie API (All-Star tier required).

        Returns:
            List of injury reports
        """
        if not self.balldontlie:
            return []

        try:
            print("Fetching Balldontlie injury data...")
            injuries = self.balldontlie.get_injuries()
            self.injuries_cache = injuries

            # Count injuries by status
            by_status = {}
            for inj in injuries:
                status = inj.get("status", "unknown")
                by_status[status] = by_status.get(status, 0) + 1

            print(f"  Found {len(injuries)} injuries:")
            for status, count in by_status.items():
                print(f"    {status}: {count}")

            return injuries

        except Exception as e:
            print(f"Error fetching Balldontlie injuries: {e}")
            return []

    def fetch_balldontlie_live_scores(self) -> list[dict]:
        """
        Fetch live box scores from Balldontlie API (GOAT tier required).

        Returns:
            List of live game box scores
        """
        if not self.balldontlie:
            return []

        try:
            print("Fetching live box scores...")
            live_scores = self.balldontlie.get_live_box_scores()
            print(f"  Found {len(live_scores)} live games")
            return live_scores

        except Exception as e:
            print(f"Error fetching live scores: {e}")
            return []

    def fetch_all_premium_data(self, date: str = None) -> dict[str, Any]:
        """
        Fetch all premium data from available sources.

        Prioritizes Balldontlie (if available with API key),
        falls back to The Odds API, then free sources.

        Args:
            date: Date in YYYY-MM-DD format (defaults to today)

        Returns:
            Dictionary with all premium data
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        result = {
            "date": date,
            "odds_source": None,
            "injuries_source": None,
            "data": {},
        }

        # Try Balldontlie first (preferred)
        if self.balldontlie:
            print("\nUsing Balldontlie for premium data...")

            # Fetch odds (GOAT tier)
            odds_result = self.fetch_balldontlie_odds(date)
            if odds_result.get("games_with_odds", 0) > 0:
                result["odds_source"] = "Balldontlie"
                result["data"]["odds"] = odds_result

            # Fetch injuries (All-Star tier)
            injuries = self.fetch_balldontlie_injuries()
            if injuries:
                result["injuries_source"] = "Balldontlie"
                result["data"]["injuries"] = injuries

        # Fall back to The Odds API
        if result["odds_source"] is None and self.odds_fetcher:
            print("\nFalling back to The Odds API for odds...")
            self.fetch_real_odds()
            if self.current_odds:
                result["odds_source"] = "TheOddsAPI"

        # Summary
        print("\nPremium data summary:")
        print(f"  Odds source: {result['odds_source'] or 'None (using defaults)'}")
        print(f"  Injuries source: {result['injuries_source'] or 'ESPN (free)'}")

        return result

    def get_injury_adjustment(self, home_abbrev: str, away_abbrev: str) -> dict[str, Any]:
        """
        Get injury-based spread adjustment for a game.

        Args:
            home_abbrev: Home team abbreviation
            away_abbrev: Away team abbreviation

        Returns:
            Injury impact data
        """
        if not self.injury_fetcher:
            return {"spread_adjustment": 0.0, "home_impact": 0.0, "away_impact": 0.0}

        try:
            summary = self.injury_fetcher.get_game_injury_summary(home_abbrev, away_abbrev)
            spread_adj = self.injury_fetcher.get_spread_adjustment(home_abbrev, away_abbrev)

            return {
                "spread_adjustment": spread_adj,
                "home_impact": summary.get("home_impact", {}).get("overall_impact", 0.0),
                "away_impact": summary.get("away_impact", {}).get("overall_impact", 0.0),
                "recommendation": summary.get("recommendation", ""),
                "home_players_out": summary.get("home_impact", {}).get("total_players_out", 0),
                "away_players_out": summary.get("away_impact", {}).get("total_players_out", 0),
            }
        except Exception as e:
            print(f"Error getting injury data: {e}")
            return {"spread_adjustment": 0.0}

    @staticmethod
    def american_to_implied_prob(odds: float) -> float:
        """Convert American odds to implied probability."""
        if odds >= 100:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    def analyze_game(self, game: dict) -> GameAnalysis:
        """
        Perform complete analysis on a single game.

        Args:
            game: Game dictionary from schedule

        Returns:
            Complete game analysis
        """
        home_team = game["home_team"]
        away_team = game["visitor_team"]

        home_abbrev = home_team.get("abbreviation", "")
        away_abbrev = away_team.get("abbreviation", "")

        print(f"\nAnalyzing: {away_abbrev} @ {home_abbrev}...")

        # Generate features with validation
        features = {}
        features_valid = False
        try:
            print("  Fetching team data and generating features...")
            features = generate_game_features(
                home_abbrev,
                away_abbrev,
                season=self.season,
                include_advanced=True,
                injury_manager=self.injury_manager,
            )

            # CRITICAL: Validate features were actually generated
            moneyline_features = features.get("moneyline_features", {})
            is_valid, validation_msg = validate_features(moneyline_features, "game")

            if is_valid:
                print("  Features generated and validated successfully.")
                features_valid = True
            else:
                print(f"  WARNING: Feature validation failed: {validation_msg}")
                print("  This game will have limited predictions due to incomplete data.")

        except Exception as e:
            print(f"  ERROR generating features: {e}")
            print("  This game will be skipped or have very limited predictions.")

        # Initialize analysis with feature validation status
        analysis = GameAnalysis(
            game_id=game.get("game_id", ""),
            home_team=f"{home_team.get('city', '')} {home_team.get('name', '')}".strip(),
            away_team=f"{away_team.get('city', '')} {away_team.get('name', '')}".strip(),
            game_time=game.get("game_time", ""),
            features=features,
            features_valid=features_valid,  # Track if features were successfully generated
        )

        # Generate predictions
        moneyline_features = features.get("moneyline_features", {})
        spread_features = features.get("spread_features", {})

        # Moneyline prediction
        if self.models_loaded and "moneyline" in self.pipeline.models:
            try:
                analysis.moneyline_prediction = self.pipeline.models["moneyline"].predict(moneyline_features)
                # Apply calibration for more accurate probabilities
                analysis.moneyline_prediction = self._calibrate_moneyline(analysis.moneyline_prediction)
            except Exception:
                analysis.moneyline_prediction = self._feature_based_moneyline(moneyline_features)
        else:
            analysis.moneyline_prediction = self._feature_based_moneyline(moneyline_features)

        # Spread prediction - need market spread_line for classifier
        # Fetch market odds early to get the spread line
        game_odds = self.get_game_odds(home_abbrev, away_abbrev)
        spread_odds = game_odds.get("spread", {"home_line": -3.5})
        market_spread_line = spread_odds.get("home_line", -3.5)
        if isinstance(market_spread_line, str):
            try:
                market_spread_line = float(market_spread_line)
            except ValueError:
                market_spread_line = -3.5

        if self.models_loaded and "spread" in self.pipeline.models:
            spread_model = self.pipeline.models["spread"]
            try:
                # Check if using SpreadCoverClassifier (line-aware) or SpreadModel (regressor)
                if isinstance(spread_model, SpreadCoverClassifier):
                    # Classifier takes spread_line as input and outputs P(home_covers) directly
                    analysis.spread_prediction = spread_model.predict(spread_features, spread_line=market_spread_line)
                    # Convert classifier output to expected format
                    cover_prob = analysis.spread_prediction.get("home_cover_probability", 0.5)
                    analysis.spread_prediction["cover_probability"] = cover_prob
                    # Apply calibration
                    analysis.spread_prediction = self._calibrate_spread(analysis.spread_prediction)
                else:
                    # Legacy regressor model
                    analysis.spread_prediction = spread_model.predict(spread_features)
                    analysis.spread_prediction = self._calibrate_spread(analysis.spread_prediction)
            except Exception:
                analysis.spread_prediction = self._feature_based_spread(spread_features)
        else:
            analysis.spread_prediction = self._feature_based_spread(spread_features)

        # Store fetched odds for later use in recommendations
        analysis.market_odds = game_odds

        # Generate recommendations
        analysis.recommendations = self._generate_game_recommendations(analysis, home_abbrev, away_abbrev)

        return analysis

    def _feature_based_moneyline(self, features: dict) -> dict:
        """Generate moneyline prediction from features without ML model."""
        # Use net rating differential as primary indicator
        net_rating_diff = features.get("net_rating_diff", 0)
        home_advantage = features.get("home_advantage_factor", 0.03)
        recent_form = features.get("combined_form", 0)
        h2h_factor = features.get("h2h_home_win_pct", 0.5) - 0.5
        injury_factor = features.get("injury_advantage", 0) * 0.01

        # Combine factors (weights tuned empirically)
        combined_score = (
            net_rating_diff * 0.03 +  # Each point of net rating ~3% win probability
            home_advantage * 0.5 +
            recent_form * 0.01 +
            h2h_factor * 0.1 +
            injury_factor
        )

        # Convert to probability (logistic function)
        import math
        home_prob = 1 / (1 + math.exp(-combined_score))

        return {
            "home_win_probability": home_prob,
            "away_win_probability": 1 - home_prob,
            "predicted_winner": "home" if home_prob > 0.5 else "away",
            "confidence": max(home_prob, 1 - home_prob),
        }

    def _feature_based_spread(self, features: dict) -> dict:
        """Generate spread prediction from features without ML model."""
        # Base prediction from expected point differential
        expected_diff = features.get("expected_point_diff", 0)
        plus_minus_diff = features.get("plus_minus_diff", 0)
        net_rating_diff = features.get("net_rating_diff", 0)
        h2h_spread = features.get("h2h_spread_prediction", 0)

        # Weighted average
        predicted_spread = (
            expected_diff * 0.3 +
            plus_minus_diff * 0.3 +
            net_rating_diff * 0.25 +
            h2h_spread * 0.15
        )

        # Add home court advantage (~3 points historically)
        predicted_spread += 3.0

        return {
            "predicted_spread": predicted_spread,
            "predicted_winner": "home" if predicted_spread > 0 else "away",
            "predicted_margin": abs(predicted_spread),
        }

    def _generate_game_recommendations(
        self,
        analysis: GameAnalysis,
        home_abbrev: str,
        away_abbrev: str,
    ) -> list[BetRecommendation]:
        """Generate betting recommendations for a game using REAL ODDS."""
        recommendations = []

        # Use pre-fetched odds from analyze_game (stored in analysis.market_odds)
        # This avoids redundant API calls
        game_odds = getattr(analysis, 'market_odds', None) or self.get_game_odds(home_abbrev, away_abbrev)
        ml_odds = game_odds.get("moneyline", {"home": -110, "away": -110})
        spread_odds = game_odds.get("spread", {"home_line": -3.5, "home_odds": -110, "away_line": 3.5, "away_odds": -110})
        total_odds = game_odds.get("total", {"line": 220.0, "over_odds": -110, "under_odds": -110})
        sportsbook = game_odds.get("sportsbook", "Default")

        # Get injury adjustments
        injury_data = self.get_injury_adjustment(home_abbrev, away_abbrev)
        analysis.injury_impact = injury_data

        # Get line movement data for edge quality
        line_movement = {}
        line_movement_note = ""
        if self.line_tracker:
            game_id = analysis.game_id if hasattr(analysis, 'game_id') else f"{away_abbrev}@{home_abbrev}"
            line_movement = self.line_tracker.calculate_line_movement(game_id) or {}
            if line_movement:
                spread_move = line_movement.get("movements", {}).get("spread", {})
                if spread_move:
                    pt_change = spread_move.get("point_change", 0)
                    direction = spread_move.get("direction", "")
                    if abs(pt_change) >= 0.5:
                        line_movement_note = f" | Line moved {pt_change:+.1f}pts ({direction})"

        # Moneyline recommendation
        ml = analysis.moneyline_prediction
        home_prob = ml.get("home_win_probability", 0.5)
        away_prob = ml.get("away_win_probability", 0.5)

        # Use REAL odds to calculate implied probability
        home_ml_odds = ml_odds.get("home", -110)
        away_ml_odds = ml_odds.get("away", -110)
        home_implied_prob = self.american_to_implied_prob(home_ml_odds)
        away_implied_prob = self.american_to_implied_prob(away_ml_odds)

        # Evaluate home moneyline with REAL odds
        home_eval = self.strategy.evaluate_bet(home_prob, home_implied_prob, "moneyline")
        if home_eval["is_recommended"]:
            # Evaluate edge quality
            home_edge_quality = self._evaluate_edge_quality(
                home_prob, home_implied_prob, "moneyline", "home", analysis, line_movement
            )

            # Skip AVOID tier bets
            if home_edge_quality and home_edge_quality.tier.value == "avoid":
                pass  # Don't add this bet
            else:
                stake = self.strategy.calculate_kelly_stake(home_prob, home_ml_odds, home_eval["confidence"])

                # Adjust stake by edge quality multiplier
                if home_edge_quality:
                    stake = round(stake * home_edge_quality.recommended_kelly_multiplier, 2)

                # Extract edge quality info
                eq_score = home_edge_quality.overall_score if home_edge_quality else None
                eq_tier = home_edge_quality.tier.value if home_edge_quality else None
                eq_factors = (home_edge_quality.positive_factors[:2] + home_edge_quality.risk_factors[:2]) if home_edge_quality else []

                recommendations.append(BetRecommendation(
                    bet_type="moneyline",
                    description=f"{analysis.home_team} ML",
                    selection="home",
                    probability=home_prob,
                    confidence=home_eval["confidence"],
                    edge=home_eval["edge"],
                    expected_value=home_eval["expected_value"],
                    recommended_stake=stake,
                    reasoning=f"Model: {home_prob:.1%} vs Market: {home_implied_prob:.1%} (odds: {home_ml_odds:+.0f})",
                    game_info={"home": home_abbrev, "away": away_abbrev},
                    odds=home_ml_odds,
                    implied_probability=home_implied_prob,
                    sportsbook=sportsbook,
                    edge_quality_score=eq_score,
                    edge_quality_tier=eq_tier,
                    edge_quality_factors=eq_factors,
                ))

        # Evaluate away moneyline with REAL odds
        away_eval = self.strategy.evaluate_bet(away_prob, away_implied_prob, "moneyline")
        if away_eval["is_recommended"]:
            # Evaluate edge quality
            away_edge_quality = self._evaluate_edge_quality(
                away_prob, away_implied_prob, "moneyline", "away", analysis, line_movement
            )

            # Skip AVOID tier bets
            if away_edge_quality and away_edge_quality.tier.value == "avoid":
                pass  # Don't add this bet
            else:
                stake = self.strategy.calculate_kelly_stake(away_prob, away_ml_odds, away_eval["confidence"])

                # Adjust stake by edge quality multiplier
                if away_edge_quality:
                    stake = round(stake * away_edge_quality.recommended_kelly_multiplier, 2)

                # Extract edge quality info
                eq_score = away_edge_quality.overall_score if away_edge_quality else None
                eq_tier = away_edge_quality.tier.value if away_edge_quality else None
                eq_factors = (away_edge_quality.positive_factors[:2] + away_edge_quality.risk_factors[:2]) if away_edge_quality else []

                recommendations.append(BetRecommendation(
                    bet_type="moneyline",
                    description=f"{analysis.away_team} ML",
                    selection="away",
                    probability=away_prob,
                    confidence=away_eval["confidence"],
                    edge=away_eval["edge"],
                    expected_value=away_eval["expected_value"],
                    recommended_stake=stake,
                    reasoning=f"Model: {away_prob:.1%} vs Market: {away_implied_prob:.1%} (odds: {away_ml_odds:+.0f})",
                    game_info={"home": home_abbrev, "away": away_abbrev},
                    odds=away_ml_odds,
                    implied_probability=away_implied_prob,
                    sportsbook=sportsbook,
                    edge_quality_score=eq_score,
                    edge_quality_tier=eq_tier,
                    edge_quality_factors=eq_factors,
                ))

        # Spread recommendation with REAL spread line
        sp = analysis.spread_prediction
        injury_adj = injury_data.get("spread_adjustment", 0.0)

        # Use REAL spread line from market
        real_spread_line = spread_odds.get("home_line", -3.5)
        if isinstance(real_spread_line, str):
            try:
                real_spread_line = float(real_spread_line)
            except ValueError:
                real_spread_line = -3.5

        # Get spread odds - use real odds if available, otherwise estimate based on spread size
        home_spread_odds = spread_odds.get("home_odds")
        away_spread_odds = spread_odds.get("away_odds")

        # If no real odds or both are default -110, estimate based on spread
        if home_spread_odds is None or away_spread_odds is None or (home_spread_odds == -110 and away_spread_odds == -110):
            fav_odds, dog_odds = self._estimate_spread_odds(real_spread_line)
            # If home is favored (negative line), home gets favorite odds
            if real_spread_line < 0:
                home_spread_odds = fav_odds
                away_spread_odds = dog_odds
            else:
                home_spread_odds = dog_odds
                away_spread_odds = fav_odds

        # Get cover probability - two sources:
        # 1. SpreadCoverClassifier outputs home_cover_probability directly
        # 2. Legacy regressor requires conversion via determine_spread_bet_side()
        home_cover_prob = sp.get("cover_probability") or sp.get("home_cover_probability")

        if home_cover_prob is not None:
            # Classifier path: We have direct P(home_covers)
            # Determine which side to bet based on probability
            if home_cover_prob > 0.5:
                side = "home"
                cover_prob = home_cover_prob
                bet_odds = home_spread_odds
                bet_line = real_spread_line
                description = f"{analysis.home_team} {real_spread_line}"
            else:
                side = "away"
                cover_prob = 1.0 - home_cover_prob  # P(away covers)
                bet_odds = away_spread_odds
                bet_line = -real_spread_line
                description = f"{analysis.away_team} {bet_line:+.1f}"

            implied_prob = self.american_to_implied_prob(bet_odds)
            true_edge = cover_prob - implied_prob
            reasoning_base = f"Model: {cover_prob:.1%} vs Market: {implied_prob:.1%}"
        else:
            # Legacy regressor path: convert predicted_spread to probability
            predicted_spread = sp.get("predicted_spread", 0)
            adjusted_spread = predicted_spread + injury_adj

            side, edge_points, cover_prob = determine_spread_bet_side(
                model_spread=adjusted_spread,
                market_spread=real_spread_line
            )

            if side == "home":
                bet_odds = home_spread_odds
                bet_line = real_spread_line
                description = f"{analysis.home_team} {real_spread_line}"
            else:
                bet_odds = away_spread_odds
                bet_line = -real_spread_line
                description = f"{analysis.away_team} {bet_line:+.1f}"

            implied_prob = self.american_to_implied_prob(bet_odds)
            true_edge = cover_prob - implied_prob
            reasoning_base = f"Model: {adjusted_spread:+.1f} vs Line: {real_spread_line} ({edge_points:.1f}pt edge)"

        # Minimum edge threshold for spread bets
        min_edge = 0.03  # 3% minimum edge
        if true_edge >= min_edge and analysis.features_valid:
            # Evaluate bet using probability edge
            spread_eval = self.strategy.evaluate_bet(cover_prob, implied_prob, "spread")

            if spread_eval["is_recommended"]:
                # Evaluate edge quality
                spread_edge_quality = self._evaluate_edge_quality(
                    cover_prob, implied_prob, "spread", side, analysis, line_movement
                )

                # Skip AVOID tier bets
                if spread_edge_quality and spread_edge_quality.tier.value == "avoid":
                    pass  # Don't add this bet
                else:
                    stake = self.strategy.calculate_kelly_stake(cover_prob, bet_odds, spread_eval["confidence"])
                    inj_note = f" (injury adj: {injury_adj:+.1f})" if abs(injury_adj) > 0.5 else ""

                    # Adjust stake by edge quality multiplier
                    if spread_edge_quality:
                        stake = round(stake * spread_edge_quality.recommended_kelly_multiplier, 2)

                    # Extract edge quality info
                    eq_score = spread_edge_quality.overall_score if spread_edge_quality else None
                    eq_tier = spread_edge_quality.tier.value if spread_edge_quality else None
                    eq_factors = (spread_edge_quality.positive_factors[:2] + spread_edge_quality.risk_factors[:2]) if spread_edge_quality else []

                    recommendations.append(BetRecommendation(
                        bet_type="spread",
                        description=description,
                        selection=side,
                        line=bet_line,
                        probability=cover_prob,
                        confidence=spread_eval["confidence"],
                        edge=true_edge,
                        expected_value=spread_eval["expected_value"],
                        recommended_stake=stake,
                        reasoning=f"{reasoning_base}{inj_note}{line_movement_note}",
                        game_info={"home": home_abbrev, "away": away_abbrev},
                        odds=bet_odds,
                        implied_probability=implied_prob,
                        sportsbook=sportsbook,
                        edge_quality_score=eq_score,
                        edge_quality_tier=eq_tier,
                        edge_quality_factors=eq_factors,
                    ))

        # Total recommendation (over/under) with REAL line - CORRECTED LOGIC
        # Totals have similar volatility to spreads (~13 points stddev)
        total_line = total_odds.get("line", 220.0)
        over_odds = total_odds.get("over_odds", -110)
        under_odds = total_odds.get("under_odds", -110)

        # Get totals prediction if available
        totals_pred = analysis.total_prediction
        if totals_pred and analysis.features_valid:  # Only if features validated
            predicted_total = totals_pred.get("predicted_total", total_line)
            total_edge = predicted_total - total_line  # Positive = favor over

            # Use proper normal CDF for probability (same volatility as spreads)
            total_prob = spread_edge_to_cover_probability(abs(total_edge))

            if abs(total_edge) >= 2.0:  # At least 2 point edge on totals
                if total_edge > 0:
                    # Favor over
                    over_implied = self.american_to_implied_prob(over_odds)
                    true_edge = total_prob - over_implied

                    total_eval = self.strategy.evaluate_bet(total_prob, over_implied, "total")
                    if total_eval["is_recommended"] and true_edge > 0.02:
                        # Evaluate edge quality
                        over_edge_quality = self._evaluate_edge_quality(
                            total_prob, over_implied, "total", "home", analysis, line_movement
                        )

                        # Skip AVOID tier bets
                        if over_edge_quality and over_edge_quality.tier.value == "avoid":
                            pass  # Don't add this bet
                        else:
                            stake = self.strategy.calculate_kelly_stake(total_prob, over_odds, total_eval["confidence"])

                            # Adjust stake by edge quality multiplier
                            if over_edge_quality:
                                stake = round(stake * over_edge_quality.recommended_kelly_multiplier, 2)

                            # Extract edge quality info
                            eq_score = over_edge_quality.overall_score if over_edge_quality else None
                            eq_tier = over_edge_quality.tier.value if over_edge_quality else None
                            eq_factors = (over_edge_quality.positive_factors[:2] + over_edge_quality.risk_factors[:2]) if over_edge_quality else []

                            recommendations.append(BetRecommendation(
                                bet_type="total",
                                description=f"OVER {total_line}",
                                selection="over",
                                line=total_line,
                                probability=total_prob,  # TRUE probability (no cap!)
                                confidence=total_eval["confidence"],
                                edge=true_edge,  # TRUE edge
                                expected_value=total_eval["expected_value"],
                                recommended_stake=stake,
                                reasoning=f"Model: {predicted_total:.1f} pts vs Line: {total_line} ({abs(total_edge):.1f}pt edge)",
                                game_info={"home": home_abbrev, "away": away_abbrev},
                                odds=over_odds,
                                implied_probability=over_implied,
                                sportsbook=sportsbook,
                                edge_quality_score=eq_score,
                                edge_quality_tier=eq_tier,
                                edge_quality_factors=eq_factors,
                            ))
                else:
                    # Favor under
                    under_implied = self.american_to_implied_prob(under_odds)
                    true_edge = total_prob - under_implied

                    total_eval = self.strategy.evaluate_bet(total_prob, under_implied, "total")
                    if total_eval["is_recommended"] and true_edge > 0.02:
                        # Evaluate edge quality
                        under_edge_quality = self._evaluate_edge_quality(
                            total_prob, under_implied, "total", "home", analysis, line_movement
                        )

                        # Skip AVOID tier bets
                        if under_edge_quality and under_edge_quality.tier.value == "avoid":
                            pass  # Don't add this bet
                        else:
                            stake = self.strategy.calculate_kelly_stake(total_prob, under_odds, total_eval["confidence"])

                            # Adjust stake by edge quality multiplier
                            if under_edge_quality:
                                stake = round(stake * under_edge_quality.recommended_kelly_multiplier, 2)

                            # Extract edge quality info
                            eq_score = under_edge_quality.overall_score if under_edge_quality else None
                            eq_tier = under_edge_quality.tier.value if under_edge_quality else None
                            eq_factors = (under_edge_quality.positive_factors[:2] + under_edge_quality.risk_factors[:2]) if under_edge_quality else []

                            recommendations.append(BetRecommendation(
                                bet_type="total",
                                description=f"UNDER {total_line}",
                                selection="under",
                                line=total_line,
                                probability=total_prob,  # TRUE probability (no cap!)
                                confidence=total_eval["confidence"],
                                edge=true_edge,  # TRUE edge
                                expected_value=total_eval["expected_value"],
                                recommended_stake=stake,
                                reasoning=f"Model: {predicted_total:.1f} pts vs Line: {total_line} ({abs(total_edge):.1f}pt edge)",
                                game_info={"home": home_abbrev, "away": away_abbrev},
                                odds=under_odds,
                                implied_probability=under_implied,
                                sportsbook=sportsbook,
                                edge_quality_score=eq_score,
                                edge_quality_tier=eq_tier,
                                edge_quality_factors=eq_factors,
                            ))

        return recommendations

    def analyze_player_props(
        self,
        player_name: str,
        opponent_team: str,
        prop_lines: dict[str, float],
    ) -> list[BetRecommendation]:
        """
        Analyze player props for a specific player.

        Args:
            player_name: Player name
            opponent_team: Opponent team abbreviation
            prop_lines: Dictionary of prop type to line (e.g., {"points": 24.5})

        Returns:
            List of prop recommendations
        """
        recommendations = []

        try:
            features = generate_player_features(player_name, opponent_team, self.season)
        except Exception as e:
            print(f"Error generating features for {player_name}: {e}")
            return recommendations

        prop_type_map = {
            "points": "points_features",
            "rebounds": "rebounds_features",
            "assists": "assists_features",
            "threes": "threes_features",
            "pra": "pra_features",
        }

        for prop_type, line in prop_lines.items():
            feature_key = prop_type_map.get(prop_type)
            if not feature_key:
                continue

            prop_features = features.get(feature_key, {})

            # Get predicted value - prefer line-aware classifiers
            prop_model = self._get_prop_model(prop_type)
            pred = None
            over_prob = None

            if prop_model is not None:
                try:
                    pred = prop_model.predict(prop_features, prop_line=line)
                    predicted_value = pred.get("predicted_value", line)
                    # Line-aware classifiers return over_probability directly
                    over_prob = pred.get("over_probability")
                except Exception as e:
                    print(f"  Warning: Prop model failed for {prop_type}: {e}, using fallback")
                    predicted_value = self._feature_based_prop(prop_features, prop_type)
            else:
                predicted_value = self._feature_based_prop(prop_features, prop_type)

            # Calculate edge and probability
            edge = predicted_value - line
            edge_pct = edge / line if line > 0 else 0

            # Use line-aware probability if available, otherwise convert from edge
            if over_prob is not None:
                # Line-aware classifier output - apply external calibration if available
                if prop_type in self.prop_calibrators:
                    try:
                        over_prob = float(self.prop_calibrators[prop_type].calibrate(over_prob))
                    except Exception:
                        pass  # Use uncalibrated probability

                if over_prob >= 0.5:
                    selection = "over"
                    prob = over_prob
                else:
                    selection = "under"
                    prob = 1 - over_prob

                # Check minimum edge for props (5%)
                if abs(over_prob - 0.5) < 0.05:
                    continue  # Skip if edge too small
            elif abs(edge_pct) >= 0.05:  # 5% edge minimum for props
                if edge > 0:
                    selection = "over"
                    prob = 0.5 + abs(edge_pct) * 2  # Rough conversion
                else:
                    selection = "under"
                    prob = 0.5 + abs(edge_pct) * 2

                prob = min(0.75, prob)  # Cap probability
            else:
                continue  # Skip if edge too small

            # Evaluate bet and add recommendation
            prop_eval = self.strategy.evaluate_bet(prob, 0.524, "prop")
            if prop_eval["is_recommended"]:
                stake = self.strategy.calculate_kelly_stake(prob, -110, prop_eval["confidence"])
                recommendations.append(BetRecommendation(
                    bet_type="prop",
                    description=f"{player_name} {selection.upper()} {line} {prop_type}",
                    selection=selection,
                    line=line,
                    probability=prob,
                    confidence=prop_eval["confidence"],
                    edge=prop_eval["edge"],
                    expected_value=prop_eval["expected_value"],
                    recommended_stake=stake,
                    reasoning=f"Model predicts {predicted_value:.1f} vs line {line}",
                    game_info={"player": player_name, "opponent": opponent_team},
                ))

        return recommendations

    def _get_prop_model(self, prop_type: str):
        """
        Get prop model, preferring line-aware classifiers.

        Priority order:
        1. player_{type}_line_classifier (line-aware, best for betting)
        2. player_{type}_line_aware
        3. prop_{type} (generic key)
        4. player_{type} (legacy regression)

        Returns:
            Model instance or None if not found
        """
        if not self.models_loaded:
            return None

        # Priority order: line-aware models first, then ensembles
        priority_keys = [
            f"player_{prop_type}_line_classifier",
            f"player_{prop_type}_line_aware",
            f"prop_{prop_type}_line_aware",
            f"player_{prop_type}_ensemble",       # PropEnsembleModel
            f"player_{prop_type}_position_aware", # Position-aware ensemble
            f"prop_{prop_type}",
            f"player_{prop_type}",
        ]

        for key in priority_keys:
            if key in self.pipeline.models:
                return self.pipeline.models[key]

        return None

    def _feature_based_prop(self, features: dict, prop_type: str) -> float:
        """Generate prop prediction from features without ML model."""
        # Use weighted average of season and recent
        season_key = f"season_{prop_type[:3]}_avg"
        recent_key = f"recent_{prop_type[:3]}_avg"
        vs_team_key = f"vs_team_{prop_type[:3]}_avg"

        season_avg = features.get(season_key, 0) or features.get("season_pts_avg", 0)
        recent_avg = features.get(recent_key, 0) or features.get("recent_pts_avg", 0)
        vs_team_avg = features.get(vs_team_key, 0)

        # Weight: 40% season, 40% recent, 20% vs team (if available)
        if vs_team_avg > 0:
            return season_avg * 0.35 + recent_avg * 0.45 + vs_team_avg * 0.20
        return season_avg * 0.45 + recent_avg * 0.55

    def generate_daily_bet_slip(
        self,
        include_props: bool = False,
        player_props: dict[str, dict[str, float]] | None = None,
    ) -> DailyBetSlip:
        """
        Generate comprehensive daily bet slip.

        Args:
            include_props: Whether to include player props
            player_props: Dictionary of player names to prop lines

        Returns:
            Complete daily bet slip
        """
        # Fetch today's games if not already done
        if not self.schedule:
            self.fetch_schedule()

        if not self.schedule:
            print("No games scheduled today.")
            return DailyBetSlip(
                date=datetime.now().strftime("%Y-%m-%d"),
                generated_at=datetime.now().isoformat(),
                games_analyzed=0,
                total_recommendations=0,
            )

        # Fetch premium data (odds, injuries) from best available source
        # Balldontlie > SportsDataIO > The Odds API > ESPN (free)
        self.fetch_all_premium_data()

        # Record opening odds for CLV tracking
        if self.line_tracker and self.current_odds:
            for game in self.schedule:
                home_team = game.get("home_team", {})
                away_team = game.get("visitor_team", {})
                home_abbrev = home_team.get("abbreviation", "")
                away_abbrev = away_team.get("abbreviation", "")
                game_id = str(game.get("id", f"{away_abbrev}@{home_abbrev}"))

                # Get current odds for this game
                game_odds = self.get_game_odds(home_abbrev, away_abbrev)

                # Format odds for tracker
                odds_data = self._format_odds_for_tracker(game_odds, home_abbrev, away_abbrev)

                # Record as opening odds
                self.line_tracker.record_odds_snapshot(
                    game_id=game_id,
                    home_team=home_abbrev,
                    away_team=away_abbrev,
                    odds_data=odds_data,
                    is_opening=True
                )
            print(f"Recorded opening odds for {len(self.schedule)} games")

        # Analyze each game
        all_recommendations = []
        self.game_analyses = []

        for game in self.schedule:
            # Skip games with invalid team data
            home_team = game.get("home_team")
            away_team = game.get("visitor_team")
            if not home_team or not away_team:
                print("  Skipping game with missing team data")
                continue
            if not home_team.get("abbreviation") or not away_team.get("abbreviation"):
                print("  Skipping game with missing team abbreviation")
                continue

            analysis = self.analyze_game(game)
            self.game_analyses.append(analysis)
            all_recommendations.extend(analysis.recommendations)

        # Add player props if requested
        prop_recommendations = []
        if include_props:
            print("\nAnalyzing player props...")

            # If player_props provided manually, use those
            if player_props:
                for player, lines in player_props.items():
                    opponent = self._find_player_opponent(player)
                    if opponent:
                        prop_recs = self.analyze_player_props(player, opponent, lines)
                        prop_recommendations.extend(prop_recs)

            # Otherwise, fetch props from API for each game
            elif HAS_BALLDONTLIE and self.balldontlie:
                from daily_predictions import get_player_props_for_game
                from id_mapping import get_id_mapper

                mapper = get_id_mapper()

                for game in self.schedule:
                    game_id = game.get("id")
                    if not game_id:
                        continue

                    home_team = game.get("home_team", {}).get("abbreviation", "")
                    away_team = game.get("visitor_team", {}).get("abbreviation", "")

                    try:
                        props_data = get_player_props_for_game(self.balldontlie, game_id)
                        if not props_data:
                            continue

                        # Filter to key players (points line >= 15)
                        key_players = {
                            pid: props for pid, props in props_data.items()
                            if props.get("points_line", 0) >= 15
                        }

                        # Process top 6 players per game
                        sorted_players = sorted(
                            key_players.items(),
                            key=lambda x: x[1].get("points_line", 0),
                            reverse=True
                        )[:6]

                        for player_id, props in sorted_players:
                            # Get player name from mapper
                            player_name = mapper.get_player_name(player_id)
                            if not player_name:
                                continue

                            # Get player's team to determine opponent
                            player_info = self.balldontlie.get_player(player_id)
                            if not player_info:
                                continue

                            player_team = player_info.get("team", {}).get("abbreviation", "")
                            if player_team == home_team:
                                opponent = away_team
                            elif player_team == away_team:
                                opponent = home_team
                            else:
                                continue

                            # Convert props to format expected by analyze_player_props
                            prop_lines = {
                                "points": props.get("points_line"),
                                "rebounds": props.get("rebounds_line"),
                                "assists": props.get("assists_line"),
                                "threes": props.get("threes_line"),
                            }
                            # Filter out None values
                            prop_lines = {k: v for k, v in prop_lines.items() if v is not None}

                            if prop_lines:
                                prop_recs = self.analyze_player_props(player_name, opponent, prop_lines)
                                prop_recommendations.extend(prop_recs)

                    except Exception as e:
                        print(f"  Warning: Failed to fetch props for game {game_id}: {e}")

            if prop_recommendations:
                print(f"  Found {len(prop_recommendations)} player prop recommendations")
                all_recommendations.extend(prop_recommendations)

        # Generate parlay recommendations
        parlay_legs = [
            {
                "type": r.bet_type,
                "description": r.description,
                "probability": r.probability,
                "confidence": r.confidence,
                "edge": r.edge,
            }
            for r in all_recommendations
        ]
        parlay_recs = self.strategy.generate_parlay_strategy(parlay_legs)

        # Sort recommendations by edge
        all_recommendations.sort(key=lambda x: x.edge, reverse=True)

        # Get top picks
        top_picks = [r for r in all_recommendations if r.confidence in ["high", "medium"]][:5]

        # Calculate bankroll allocation
        bankroll_allocation = self.strategy.allocate_bankroll(all_recommendations)

        # Create bet slip
        return DailyBetSlip(
            date=datetime.now().strftime("%Y-%m-%d"),
            generated_at=datetime.now().isoformat(),
            games_analyzed=len(self.game_analyses),
            total_recommendations=len(all_recommendations),
            top_picks=top_picks,
            game_analyses=self.game_analyses,
            parlay_recommendations=parlay_recs,
            bankroll_allocation=bankroll_allocation,
        )


    def _find_player_opponent(self, player_name: str) -> str | None:
        """Find a player's opponent from today's schedule."""
        try:
            if not HAS_BALLDONTLIE or not self.balldontlie:
                return None

            # Search for player by name in Balldontlie API
            # Note: API doesn't handle full names with spaces well
            # Try searching by first name, then last name if needed
            name_parts = player_name.split()
            players = []

            # Try first name (usually more unique)
            if name_parts:
                players = self.balldontlie.get_players(search=name_parts[0])

            # If no results or too many, try last name
            if (not players or len(players) > 50) and len(name_parts) > 1:
                players = self.balldontlie.get_players(search=name_parts[-1])

            if not players:
                return None

            # Find exact match by full name
            player_info = None
            target_name = player_name.lower()
            for p in players:
                full_name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                if full_name.lower() == target_name:
                    player_info = p
                    break

            # If no exact match, use first result
            if not player_info:
                player_info = players[0]

            player_team = player_info.get("team", {}).get("abbreviation")
            if not player_team:
                return None

            # Get today's games and find this team's game
            games = self.balldontlie.get_todays_games()

            for game in games:
                home = game.get("home_team", {}).get("abbreviation")
                away = game.get("visitor_team", {}).get("abbreviation")

                if player_team == home:
                    return away
                if player_team == away:
                    return home

            return None  # Player's team not playing today
        except Exception as e:
            print(f"Warning: Could not find opponent for {player_name}: {e}")
            return None

    def print_bet_slip(self, bet_slip: DailyBetSlip):
        """Print formatted bet slip to console."""
        print("\n" + "=" * 70)
        print("NBA DAILY BET SLIP")
        print("=" * 70)
        print(f"Date: {bet_slip.date}")
        print(f"Generated: {bet_slip.generated_at}")
        print(f"Games Analyzed: {bet_slip.games_analyzed}")
        print(f"Total Recommendations: {bet_slip.total_recommendations}")

        if bet_slip.top_picks:
            print("\n" + "-" * 40)
            print("TOP PICKS")
            print("-" * 40)
            for i, pick in enumerate(bet_slip.top_picks, 1):
                print(f"\n{i}. {pick.description}")
                print(f"   Confidence: {pick.confidence.upper()}")
                print(f"   Win Probability: {pick.probability:.1%}")
                print(f"   Edge: {pick.edge:.1%}")
                print(f"   Recommended Stake: ${pick.recommended_stake:.2f}")
                print(f"   Reasoning: {pick.reasoning}")

        if bet_slip.parlay_recommendations:
            print("\n" + "-" * 40)
            print("PARLAY RECOMMENDATIONS")
            print("-" * 40)
            for parlay in bet_slip.parlay_recommendations:
                print(f"\n{parlay['type']} ({parlay['combined_probability']:.1%} probability)")
                for leg in parlay['legs']:
                    print(f"  - {leg['description']}")

        if bet_slip.bankroll_allocation:
            alloc = bet_slip.bankroll_allocation
            print("\n" + "-" * 40)
            print("BANKROLL ALLOCATION")
            print("-" * 40)
            print(f"Total Stake: ${alloc['total_stake']:.2f}")
            print(f"Number of Bets: {alloc['num_bets']}")
            print(f"Bankroll Used: {alloc['bankroll_percentage_used']:.1f}%")

        print("\n" + "=" * 70)

    def save_bet_slip(self, bet_slip: DailyBetSlip, filepath: Path | None = None):
        """Save bet slip to JSON file."""
        if filepath is None:
            filepath = Path(f"bet_slip_{bet_slip.date}.json")

        # Convert to serializable format
        output = {
            "date": bet_slip.date,
            "generated_at": bet_slip.generated_at,
            "games_analyzed": bet_slip.games_analyzed,
            "total_recommendations": bet_slip.total_recommendations,
            "top_picks": [asdict(p) for p in bet_slip.top_picks],
            "parlay_recommendations": bet_slip.parlay_recommendations,
            "bankroll_allocation": bet_slip.bankroll_allocation,
            "game_analyses": [
                {
                    "game_id": g.game_id,
                    "home_team": g.home_team,
                    "away_team": g.away_team,
                    "moneyline_prediction": g.moneyline_prediction,
                    "spread_prediction": g.spread_prediction,
                    "recommendations": [asdict(r) for r in g.recommendations],
                }
                for g in bet_slip.game_analyses
            ],
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Bet slip saved to {filepath}")


def main():
    """Main entry point for the NBA betting model."""
    print("=" * 70)
    print("NBA BETTING MODEL - Daily Analysis")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize orchestrator
    orchestrator = Orchestrator(
        season="2025-26",
        bankroll=1000.0,
        risk_tolerance="moderate",
    )

    # Try to load models
    orchestrator.load_models()

    # Example injury report (would be populated from real data)
    # orchestrator.set_injuries([
    #     {
    #         "team": "LAL",
    #         "injuries": [
    #             {"player_name": "Anthony Davis", "player_id": 203076, "status": "questionable", "position": "PF"}
    #         ]
    #     }
    # ])

    # Generate daily bet slip (with player props enabled)
    bet_slip = orchestrator.generate_daily_bet_slip(include_props=True)

    # Print and save results
    orchestrator.print_bet_slip(bet_slip)
    orchestrator.save_bet_slip(bet_slip)

    return bet_slip


if __name__ == "__main__":
    main()
