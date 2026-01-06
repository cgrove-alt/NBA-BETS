"""
NBA Betting Model Orchestrator

Main application that orchestrates the complete betting workflow:
1. Fetch today's NBA schedule and data
2. Engineer features with injury and matchup analysis
3. Load trained ML models
4. Generate predictions for all bet types
5. Output comprehensive daily bet slip with betting strategy
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
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
    else:
        # Approximate using logistic function if scipy not available
        return 1.0 / (1.0 + math.exp(-spread_edge / (volatility * 0.6)))


def determine_spread_bet_side(
    model_spread: float,
    market_spread: float,
) -> Tuple[str, float, float]:
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


def validate_features(features: Dict, feature_type: str = "game") -> Tuple[bool, str]:
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
    fetch_team_roster,
    get_team_id,
    get_player_id,
)
from feature_engineering import (
    generate_game_features,
    generate_player_features,
    create_injury_report,
    InjuryReportManager,
    MatchupFeatureGenerator,
    PlayerPropFeatureGenerator,
)
from model_trainer import (
    ModelTrainingPipeline,
    MoneylineModel,
    SpreadModel,
    PlayerPropModel,
    ParlayCalculator,
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


@dataclass
class BetRecommendation:
    """Represents a single bet recommendation."""
    bet_type: str  # "moneyline", "spread", "total", "prop", "parlay"
    description: str
    selection: str
    line: Optional[float] = None
    probability: float = 0.0
    confidence: str = "low"  # "low", "medium", "high"
    edge: float = 0.0
    expected_value: float = 0.0
    recommended_stake: float = 0.0
    reasoning: str = ""
    game_info: Dict = field(default_factory=dict)
    # Real odds fields
    odds: float = -110  # American odds
    implied_probability: float = 0.524  # Implied from odds
    sportsbook: str = ""  # Where to place the bet
    closing_line_value: Optional[float] = None  # CLV if available


@dataclass
class GameAnalysis:
    """Complete analysis for a single game."""
    game_id: str
    home_team: str
    away_team: str
    game_time: str
    features: Dict = field(default_factory=dict)
    features_valid: bool = False  # Whether features were successfully generated
    moneyline_prediction: Dict = field(default_factory=dict)
    spread_prediction: Dict = field(default_factory=dict)
    total_prediction: Dict = field(default_factory=dict)
    player_props: List[Dict] = field(default_factory=list)
    recommendations: List[BetRecommendation] = field(default_factory=list)
    # Real market data
    market_odds: Dict = field(default_factory=dict)  # Real odds from sportsbooks
    injury_impact: Dict = field(default_factory=dict)  # Injury adjustments
    best_odds: Dict = field(default_factory=dict)  # Best available odds by market


@dataclass
class DailyBetSlip:
    """Complete daily betting recommendations."""
    date: str
    generated_at: str
    games_analyzed: int
    total_recommendations: int
    top_picks: List[BetRecommendation] = field(default_factory=list)
    game_analyses: List[GameAnalysis] = field(default_factory=list)
    parlay_recommendations: List[Dict] = field(default_factory=list)
    bankroll_allocation: Dict = field(default_factory=dict)


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
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

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
    ) -> Dict[str, Any]:
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
        legs: List[Dict],
        max_legs: int = 4,
    ) -> List[Dict]:
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
        recommendations: List[BetRecommendation],
    ) -> Dict[str, Any]:
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
                for name in self.pipeline.models.keys():
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
                    print(f"  - moneyline calibrator not available (using uncalibrated)")
                    self.moneyline_calibrator = None

                # Load spread calibrator
                try:
                    self.spread_calibrator = ModelCalibrator("spread")
                    self.spread_calibrator.load(str(calibration_dir))
                    print(f"  - spread calibrator ({self.spread_calibrator.best_method})")
                except Exception as e:
                    logging.warning(f"Spread calibrator not loaded: {e}")
                    print(f"  - spread calibrator not available (using uncalibrated)")
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
                    print(f"  - prop calibrators not available (using uncalibrated)")
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

    def _calibrate_moneyline(self, prediction: Dict) -> Dict:
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

    def _calibrate_spread(self, prediction: Dict) -> Dict:
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

    def set_injuries(self, injury_data: List[Dict]):
        """
        Set injury report data.

        Args:
            injury_data: List of team injury reports
        """
        self.injury_manager = create_injury_report(injury_data, self.season)
        print(f"Injury report set for {len(injury_data)} teams")

    def fetch_schedule(self) -> List[Dict]:
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

    def fetch_real_odds(self) -> Dict[str, Any]:
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

    def get_game_odds(self, home_abbrev: str, away_abbrev: str) -> Dict[str, Any]:
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
        game_odds: Dict,
        home_abbrev: str,
        away_abbrev: str
    ) -> Dict:
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

    def fetch_balldontlie_odds(self, date: str = None) -> Dict[str, Any]:
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

    def fetch_balldontlie_injuries(self) -> List[Dict]:
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

    def fetch_balldontlie_live_scores(self) -> List[Dict]:
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

    def fetch_all_premium_data(self, date: str = None) -> Dict[str, Any]:
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
        print(f"\nPremium data summary:")
        print(f"  Odds source: {result['odds_source'] or 'None (using defaults)'}")
        print(f"  Injuries source: {result['injuries_source'] or 'ESPN (free)'}")

        return result

    def get_injury_adjustment(self, home_abbrev: str, away_abbrev: str) -> Dict[str, Any]:
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
        else:
            return abs(odds) / (abs(odds) + 100)

    def analyze_game(self, game: Dict) -> GameAnalysis:
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
            print(f"  Fetching team data and generating features...")
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
                print(f"  Features generated and validated successfully.")
                features_valid = True
            else:
                print(f"  WARNING: Feature validation failed: {validation_msg}")
                print(f"  This game will have limited predictions due to incomplete data.")

        except Exception as e:
            print(f"  ERROR generating features: {e}")
            print(f"  This game will be skipped or have very limited predictions.")

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

        # Spread prediction
        if self.models_loaded and "spread" in self.pipeline.models:
            try:
                analysis.spread_prediction = self.pipeline.models["spread"].predict(spread_features)
                # Apply calibration for more accurate cover probabilities
                analysis.spread_prediction = self._calibrate_spread(analysis.spread_prediction)
            except Exception:
                analysis.spread_prediction = self._feature_based_spread(spread_features)
        else:
            analysis.spread_prediction = self._feature_based_spread(spread_features)

        # Generate recommendations
        analysis.recommendations = self._generate_game_recommendations(analysis, home_abbrev, away_abbrev)

        return analysis

    def _feature_based_moneyline(self, features: Dict) -> Dict:
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

    def _feature_based_spread(self, features: Dict) -> Dict:
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
    ) -> List[BetRecommendation]:
        """Generate betting recommendations for a game using REAL ODDS."""
        recommendations = []

        # Get REAL odds for this game
        game_odds = self.get_game_odds(home_abbrev, away_abbrev)
        ml_odds = game_odds.get("moneyline", {"home": -110, "away": -110})
        spread_odds = game_odds.get("spread", {"home_line": -3.5, "home_odds": -110, "away_line": 3.5, "away_odds": -110})
        total_odds = game_odds.get("total", {"line": 220.0, "over_odds": -110, "under_odds": -110})
        sportsbook = game_odds.get("sportsbook", "Default")

        # Store odds in analysis
        analysis.market_odds = game_odds

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
            stake = self.strategy.calculate_kelly_stake(home_prob, home_ml_odds, home_eval["confidence"])
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
            ))

        # Evaluate away moneyline with REAL odds
        away_eval = self.strategy.evaluate_bet(away_prob, away_implied_prob, "moneyline")
        if away_eval["is_recommended"]:
            stake = self.strategy.calculate_kelly_stake(away_prob, away_ml_odds, away_eval["confidence"])
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
            ))

        # Spread recommendation with REAL spread line - CORRECTED LOGIC
        # This fixes the sign convention bug that was causing wrong-direction bets
        sp = analysis.spread_prediction
        predicted_spread = sp.get("predicted_spread", 0)

        # Apply injury adjustment to predicted spread
        injury_adj = injury_data.get("spread_adjustment", 0.0)
        adjusted_spread = predicted_spread + injury_adj

        # Use REAL spread line from market
        real_spread_line = spread_odds.get("home_line", -3.5)
        # Ensure spread line is a float (API may return string)
        if isinstance(real_spread_line, str):
            try:
                real_spread_line = float(real_spread_line)
            except ValueError:
                real_spread_line = -3.5
        home_spread_odds = spread_odds.get("home_odds", -110)
        away_spread_odds = spread_odds.get("away_odds", -110)

        # CRITICAL FIX: Use correct spread-to-probability conversion and direction logic
        # determine_spread_bet_side() properly handles:
        # - Which side to bet based on model vs market
        # - Correct edge calculation in points
        # - Proper probability using normal CDF (not linear approximation)
        side, edge_points, cover_prob = determine_spread_bet_side(
            model_spread=adjusted_spread,
            market_spread=real_spread_line
        )

        # Only recommend if we have meaningful edge (at least 1 point)
        # and the features were validated successfully
        min_edge_points = 1.0
        if edge_points >= min_edge_points and analysis.features_valid:
            # Get the appropriate odds and implied probability for chosen side
            if side == "home":
                bet_odds = home_spread_odds
                bet_line = real_spread_line
                description = f"{analysis.home_team} {real_spread_line}"
                implied_prob = self.american_to_implied_prob(home_spread_odds)
            else:  # away
                bet_odds = away_spread_odds
                bet_line = -real_spread_line  # Flip sign for away
                description = f"{analysis.away_team} {bet_line:+.1f}"
                implied_prob = self.american_to_implied_prob(away_spread_odds)

            # Calculate true edge (probability difference, not point difference)
            true_edge = cover_prob - implied_prob

            # Evaluate bet using true probability edge
            spread_eval = self.strategy.evaluate_bet(cover_prob, implied_prob, "spread")

            if spread_eval["is_recommended"] and true_edge > 0.02:  # Min 2% true edge
                stake = self.strategy.calculate_kelly_stake(cover_prob, bet_odds, spread_eval["confidence"])
                inj_note = f" (injury adj: {injury_adj:+.1f})" if abs(injury_adj) > 0.5 else ""

                recommendations.append(BetRecommendation(
                    bet_type="spread",
                    description=description,
                    selection=side,
                    line=bet_line,
                    probability=cover_prob,  # TRUE probability (no 75% cap!)
                    confidence=spread_eval["confidence"],
                    edge=true_edge,  # TRUE edge (probability diff, not points)
                    expected_value=spread_eval["expected_value"],
                    recommended_stake=stake,
                    reasoning=f"Model: {adjusted_spread:+.1f} vs Line: {real_spread_line} ({edge_points:.1f}pt edge){inj_note}{line_movement_note}",
                    game_info={"home": home_abbrev, "away": away_abbrev},
                    odds=bet_odds,
                    implied_probability=implied_prob,
                    sportsbook=sportsbook,
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
                        stake = self.strategy.calculate_kelly_stake(total_prob, over_odds, total_eval["confidence"])
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
                        ))
                else:
                    # Favor under
                    under_implied = self.american_to_implied_prob(under_odds)
                    true_edge = total_prob - under_implied

                    total_eval = self.strategy.evaluate_bet(total_prob, under_implied, "total")
                    if total_eval["is_recommended"] and true_edge > 0.02:
                        stake = self.strategy.calculate_kelly_stake(total_prob, under_odds, total_eval["confidence"])
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
                        ))

        return recommendations

    def analyze_player_props(
        self,
        player_name: str,
        opponent_team: str,
        prop_lines: Dict[str, float],
    ) -> List[BetRecommendation]:
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
                except Exception:
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

        # Priority order: line-aware models first
        priority_keys = [
            f"player_{prop_type}_line_classifier",
            f"player_{prop_type}_line_aware",
            f"prop_{prop_type}_line_aware",
            f"prop_{prop_type}",
            f"player_{prop_type}",
        ]

        for key in priority_keys:
            if key in self.pipeline.models:
                return self.pipeline.models[key]

        return None

    def _feature_based_prop(self, features: Dict, prop_type: str) -> float:
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
        else:
            return season_avg * 0.45 + recent_avg * 0.55

    def generate_daily_bet_slip(
        self,
        include_props: bool = False,
        player_props: Optional[Dict[str, Dict[str, float]]] = None,
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
        premium_data = self.fetch_all_premium_data()

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
                print(f"  Skipping game with missing team data")
                continue
            if not home_team.get("abbreviation") or not away_team.get("abbreviation"):
                print(f"  Skipping game with missing team abbreviation")
                continue

            analysis = self.analyze_game(game)
            self.game_analyses.append(analysis)
            all_recommendations.extend(analysis.recommendations)

        # Add player props if requested
        if include_props and player_props:
            for player, lines in player_props.items():
                # Determine opponent from schedule
                opponent = self._find_player_opponent(player)
                if opponent:
                    prop_recs = self.analyze_player_props(player, opponent, lines)
                    all_recommendations.extend(prop_recs)

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
        bet_slip = DailyBetSlip(
            date=datetime.now().strftime("%Y-%m-%d"),
            generated_at=datetime.now().isoformat(),
            games_analyzed=len(self.game_analyses),
            total_recommendations=len(all_recommendations),
            top_picks=top_picks,
            game_analyses=self.game_analyses,
            parlay_recommendations=parlay_recs,
            bankroll_allocation=bankroll_allocation,
        )

        return bet_slip

    def _find_player_opponent(self, player_name: str) -> Optional[str]:
        """Find a player's opponent from today's schedule."""
        try:
            player_id = get_player_id(player_name)
            if not player_id:
                return None

            # This would require fetching player's team and matching to schedule
            # Simplified: return None to skip
            return None
        except Exception:
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

    def save_bet_slip(self, bet_slip: DailyBetSlip, filepath: Optional[Path] = None):
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

    # Generate daily bet slip
    bet_slip = orchestrator.generate_daily_bet_slip()

    # Print and save results
    orchestrator.print_bet_slip(bet_slip)
    orchestrator.save_bet_slip(bet_slip)

    return bet_slip


if __name__ == "__main__":
    main()
