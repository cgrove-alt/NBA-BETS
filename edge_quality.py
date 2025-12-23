"""
Edge Quality Scoring System for NBA Betting Model

This module provides a sophisticated scoring system to evaluate the quality
and reliability of betting edges, combining multiple factors:
- Ensemble model agreement
- Line movement alignment with model predictions
- Feature stability and consistency
- Historical edge performance in similar situations
- Closing Line Value (CLV) prediction

A higher edge quality score indicates greater confidence in the edge being real.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class EdgeTier(Enum):
    """Classification of edge quality."""
    ELITE = "elite"       # 85-100: High confidence, max Kelly
    STRONG = "strong"     # 70-84: Good confidence, 75% Kelly
    MODERATE = "moderate" # 55-69: Moderate confidence, 50% Kelly
    WEAK = "weak"         # 40-54: Low confidence, 25% Kelly
    AVOID = "avoid"       # 0-39: Not trustworthy, skip bet


@dataclass
class EdgeQualityResult:
    """Result of edge quality evaluation."""
    overall_score: float  # 0-100
    tier: EdgeTier
    ensemble_agreement_score: float
    line_movement_score: float
    feature_stability_score: float
    recency_score: float
    situational_score: float
    recommended_kelly_multiplier: float
    confidence_interval: Tuple[float, float]
    risk_factors: List[str]
    positive_factors: List[str]
    detailed_breakdown: Dict[str, float]


class EdgeQualityScorer:
    """
    Comprehensive edge quality scoring system.

    Evaluates betting edges across multiple dimensions to determine
    confidence level and appropriate stake sizing.
    """

    # Component weights (sum to 1.0)
    WEIGHTS = {
        'ensemble_agreement': 0.25,    # Model consensus
        'line_movement': 0.20,         # Sharp money alignment
        'feature_stability': 0.15,     # Recent performance consistency
        'recency': 0.15,               # Data freshness
        'situational': 0.15,           # Historical edge in similar spots
        'probability_confidence': 0.10, # How far from 50% the prediction is
    }

    # Kelly multipliers by tier
    KELLY_MULTIPLIERS = {
        EdgeTier.ELITE: 1.0,      # Full quarter-Kelly
        EdgeTier.STRONG: 0.75,    # 75% of base Kelly
        EdgeTier.MODERATE: 0.50,  # Half Kelly
        EdgeTier.WEAK: 0.25,      # Quarter of base Kelly
        EdgeTier.AVOID: 0.0,      # Don't bet
    }

    def __init__(
        self,
        historical_edges: Optional[List[Dict]] = None,
        min_edge_threshold: float = 0.02,
    ):
        """
        Initialize the scorer.

        Args:
            historical_edges: List of past edge evaluations for calibration
            min_edge_threshold: Minimum edge required to consider betting (2%)
        """
        self.historical_edges = historical_edges or []
        self.min_edge_threshold = min_edge_threshold

    def calculate_ensemble_agreement_score(
        self,
        individual_predictions: Dict[str, float],
        ensemble_prediction: float,
    ) -> Tuple[float, List[str]]:
        """
        Score how well individual models agree with each other.

        High agreement = models see the same signal = more reliable edge.

        Args:
            individual_predictions: Dict of model_name -> probability
            ensemble_prediction: Final ensemble probability

        Returns:
            (score 0-100, list of factors)
        """
        if not individual_predictions:
            return 50.0, ["No individual model data available"]

        probs = list(individual_predictions.values())
        factors = []

        # Calculate standard deviation of predictions
        std_dev = np.std(probs)
        mean_prob = np.mean(probs)

        # All models agree on direction (>50% or <50%)?
        all_same_direction = all(p > 0.5 for p in probs) or all(p < 0.5 for p in probs)

        # Calculate coefficient of variation for disagreement
        cv = std_dev / max(abs(mean_prob - 0.5), 0.01)

        # Base score from standard deviation
        # STD < 0.02 = excellent (95-100)
        # STD 0.02-0.05 = good (80-94)
        # STD 0.05-0.10 = moderate (60-79)
        # STD > 0.10 = poor (<60)
        if std_dev < 0.02:
            score = 95 + (0.02 - std_dev) * 250  # Up to 100
            factors.append(f"Excellent model agreement (σ={std_dev:.3f})")
        elif std_dev < 0.05:
            score = 80 + (0.05 - std_dev) * 500
            factors.append(f"Good model agreement (σ={std_dev:.3f})")
        elif std_dev < 0.10:
            score = 60 + (0.10 - std_dev) * 400
            factors.append(f"Moderate model disagreement (σ={std_dev:.3f})")
        else:
            score = max(30, 60 - (std_dev - 0.10) * 200)
            factors.append(f"High model disagreement (σ={std_dev:.3f})")

        # Bonus for unanimous direction
        if all_same_direction:
            score = min(100, score + 5)
            factors.append("All models agree on direction")
        else:
            score = max(0, score - 10)
            factors.append("Models disagree on direction (CAUTION)")

        # Check for outlier models
        for name, prob in individual_predictions.items():
            if abs(prob - mean_prob) > 2 * std_dev and std_dev > 0.02:
                factors.append(f"{name} is an outlier ({prob:.3f} vs mean {mean_prob:.3f})")
                score = max(0, score - 5)

        return min(100, max(0, score)), factors

    def calculate_line_movement_score(
        self,
        model_prediction: float,
        opening_odds: float,
        current_odds: float,
        is_home: bool,
        public_betting_pct: Optional[float] = None,
        is_reverse_line_movement: bool = False,
        is_steam_move: bool = False,
    ) -> Tuple[float, List[str]]:
        """
        Score alignment between model and sharp money.

        If line moves toward our prediction = sharps agree = better edge.
        Reverse line movement indicates sharp action.

        Args:
            model_prediction: Model's win probability for selected side
            opening_odds: Opening line (spread or ML)
            current_odds: Current line
            is_home: Whether we're betting on home team
            public_betting_pct: Percentage of public on our side (if available)
            is_reverse_line_movement: Did line move against public?
            is_steam_move: Was there a rapid sharp move?

        Returns:
            (score 0-100, list of factors)
        """
        factors = []
        score = 50.0  # Start neutral

        # Calculate line movement direction
        line_moved = current_odds - opening_odds

        # Determine if line moved in our favor
        # For home team: negative spread move = more favorable
        # For away team: positive spread move = more favorable
        if is_home:
            favorable_move = line_moved < 0  # Line shortened for home
        else:
            favorable_move = line_moved > 0  # Line shortened for away

        # Score based on movement magnitude and direction
        move_magnitude = abs(line_moved)

        if model_prediction > 0.55:  # Strong model lean
            if favorable_move:
                # Market moved toward us = sharps agree
                score += min(25, move_magnitude * 10)
                factors.append(f"Line moved in our favor by {move_magnitude:.1f}")
            else:
                # Market moved against us = sharps disagree
                score -= min(20, move_magnitude * 8)
                factors.append(f"Line moved against us by {move_magnitude:.1f} (CAUTION)")

        # Reverse line movement is a strong signal
        if is_reverse_line_movement:
            if favorable_move:
                score += 15
                factors.append("Reverse line movement supports our side (sharp money)")
            else:
                score -= 20
                factors.append("Reverse line movement against us (sharps disagree)")

        # Steam move indicates strong sharp action
        if is_steam_move:
            if favorable_move:
                score += 20
                factors.append("Steam move detected on our side (very sharp)")
            else:
                score -= 25
                factors.append("Steam move against us (STRONG CAUTION)")

        # Public betting percentage
        if public_betting_pct is not None:
            if public_betting_pct > 70:
                # Heavy public side - potential trap
                score -= 10
                factors.append(f"Heavy public side ({public_betting_pct:.0f}%) - potential trap")
            elif public_betting_pct < 30:
                # Going against public with sharp support
                if favorable_move:
                    score += 10
                    factors.append(f"Contrarian play ({public_betting_pct:.0f}% public) with sharp support")

        return min(100, max(0, score)), factors

    def calculate_feature_stability_score(
        self,
        recent_form: Dict[str, float],
        season_averages: Dict[str, float],
        games_played: int,
    ) -> Tuple[float, List[str]]:
        """
        Score the stability of key features used in prediction.

        Erratic recent performance = less reliable prediction.

        Args:
            recent_form: Recent stats (last 5-10 games)
            season_averages: Full season averages
            games_played: Number of games in sample

        Returns:
            (score 0-100, list of factors)
        """
        factors = []
        score = 70.0  # Start with moderate confidence

        if not recent_form or not season_averages:
            return 50.0, ["Insufficient data for stability analysis"]

        # Minimum games for reliable stats
        if games_played < 10:
            score -= 15
            factors.append(f"Small sample size ({games_played} games)")
        elif games_played < 20:
            score -= 5
            factors.append(f"Moderate sample size ({games_played} games)")
        else:
            score += 5
            factors.append(f"Good sample size ({games_played} games)")

        # Compare recent form to season averages
        deviations = []
        for stat, recent_val in recent_form.items():
            season_val = season_averages.get(stat)
            if season_val and season_val > 0:
                pct_deviation = abs(recent_val - season_val) / season_val
                deviations.append(pct_deviation)

                if pct_deviation > 0.20:
                    factors.append(f"{stat}: Recent form deviates {pct_deviation:.0%} from average")

        if deviations:
            avg_deviation = np.mean(deviations)
            if avg_deviation < 0.05:
                score += 15
                factors.append("Very stable recent performance")
            elif avg_deviation < 0.10:
                score += 5
                factors.append("Stable recent performance")
            elif avg_deviation > 0.20:
                score -= 15
                factors.append("Unstable recent performance (high variance)")

        return min(100, max(0, score)), factors

    def calculate_recency_score(
        self,
        training_data_age_days: float,
        last_game_days_ago: float,
        is_back_to_back: bool,
    ) -> Tuple[float, List[str]]:
        """
        Score based on data freshness and schedule factors.

        Args:
            training_data_age_days: Average age of training data
            last_game_days_ago: Days since last game for the team
            is_back_to_back: Is this a back-to-back game?

        Returns:
            (score 0-100, list of factors)
        """
        factors = []
        score = 80.0

        # Training data freshness
        if training_data_age_days < 30:
            score += 10
            factors.append("Training data is very fresh")
        elif training_data_age_days < 90:
            factors.append("Training data is reasonably current")
        else:
            score -= 15
            factors.append(f"Training data is {training_data_age_days:.0f} days old on average")

        # Last game recency
        if last_game_days_ago <= 2:
            factors.append("Recent game data available")
        elif last_game_days_ago <= 4:
            score -= 5
            factors.append(f"Last game was {last_game_days_ago:.0f} days ago")
        else:
            score -= 10
            factors.append(f"Last game was {last_game_days_ago:.0f} days ago (stale)")

        # Back-to-back factor
        if is_back_to_back:
            # B2B adds uncertainty to predictions
            score -= 5
            factors.append("Back-to-back game (increased uncertainty)")

        return min(100, max(0, score)), factors

    def calculate_situational_score(
        self,
        home_away: str,
        division_game: bool,
        playoff_implications: bool,
        injury_impact_score: float,
        travel_fatigue_score: float,
    ) -> Tuple[float, List[str]]:
        """
        Score based on situational factors that affect prediction reliability.

        Args:
            home_away: "home" or "away"
            division_game: Is this a division game?
            playoff_implications: Does game have playoff significance?
            injury_impact_score: 0-1 scale of injury impact
            travel_fatigue_score: 0-1 scale of travel fatigue

        Returns:
            (score 0-100, list of factors)
        """
        factors = []
        score = 70.0

        # Home advantage is well-modeled
        if home_away == "home":
            score += 5
            factors.append("Home game (well-modeled advantage)")

        # Division games can be more unpredictable
        if division_game:
            score -= 3
            factors.append("Division game (increased variance)")

        # Playoff implications add motivation uncertainty
        if playoff_implications:
            score -= 5
            factors.append("Playoff implications (motivation variance)")

        # High injury impact increases uncertainty
        if injury_impact_score > 0.5:
            score -= 15
            factors.append(f"Significant injury impact ({injury_impact_score:.0%})")
        elif injury_impact_score > 0.25:
            score -= 7
            factors.append(f"Moderate injury impact ({injury_impact_score:.0%})")

        # Travel fatigue
        if travel_fatigue_score > 0.7:
            score -= 10
            factors.append(f"High travel fatigue ({travel_fatigue_score:.0%})")
        elif travel_fatigue_score > 0.4:
            score -= 5
            factors.append(f"Moderate travel fatigue ({travel_fatigue_score:.0%})")

        return min(100, max(0, score)), factors

    def calculate_probability_confidence_score(
        self,
        raw_probability: float,
        calibrated_probability: float,
    ) -> Tuple[float, List[str]]:
        """
        Score based on how confident the probability prediction is.

        Predictions close to 50% are less reliable.

        Args:
            raw_probability: Pre-calibration probability
            calibrated_probability: Post-calibration probability

        Returns:
            (score 0-100, list of factors)
        """
        factors = []

        # Distance from 50%
        distance_from_even = abs(calibrated_probability - 0.5)

        if distance_from_even > 0.15:
            score = 90 + min(10, distance_from_even * 50)
            factors.append(f"Strong model conviction ({calibrated_probability:.1%})")
        elif distance_from_even > 0.10:
            score = 75 + (distance_from_even - 0.10) * 300
            factors.append(f"Good model conviction ({calibrated_probability:.1%})")
        elif distance_from_even > 0.05:
            score = 55 + (distance_from_even - 0.05) * 400
            factors.append(f"Moderate model conviction ({calibrated_probability:.1%})")
        else:
            score = max(30, 55 - (0.05 - distance_from_even) * 500)
            factors.append(f"Weak model conviction ({calibrated_probability:.1%})")

        # Check calibration shift
        calibration_shift = abs(raw_probability - calibrated_probability)
        if calibration_shift > 0.10:
            score -= 10
            factors.append(f"Large calibration adjustment ({calibration_shift:.1%})")

        return min(100, max(0, score)), factors

    def evaluate_edge(
        self,
        # Core prediction data
        model_probability: float,
        implied_probability: float,
        individual_model_predictions: Optional[Dict[str, float]] = None,
        raw_probability: Optional[float] = None,

        # Line movement data
        opening_odds: Optional[float] = None,
        current_odds: Optional[float] = None,
        public_betting_pct: Optional[float] = None,
        is_reverse_line_movement: bool = False,
        is_steam_move: bool = False,

        # Form/stability data
        recent_form: Optional[Dict[str, float]] = None,
        season_averages: Optional[Dict[str, float]] = None,
        games_played: int = 30,

        # Recency data
        training_data_age_days: float = 30.0,
        last_game_days_ago: float = 2.0,
        is_back_to_back: bool = False,

        # Situational data
        home_away: str = "home",
        division_game: bool = False,
        playoff_implications: bool = False,
        injury_impact_score: float = 0.0,
        travel_fatigue_score: float = 0.0,
    ) -> EdgeQualityResult:
        """
        Comprehensive edge quality evaluation.

        Args:
            model_probability: Calibrated model probability for our side
            implied_probability: Market implied probability
            individual_model_predictions: Dict of individual model probs
            raw_probability: Pre-calibration probability
            ... (see parameter descriptions above)

        Returns:
            EdgeQualityResult with comprehensive scoring
        """
        all_factors_positive = []
        all_factors_negative = []
        detailed_scores = {}

        # Calculate edge
        edge = model_probability - implied_probability

        # 1. Ensemble Agreement Score
        if individual_model_predictions:
            ensemble_score, ensemble_factors = self.calculate_ensemble_agreement_score(
                individual_model_predictions, model_probability
            )
        else:
            ensemble_score = 60.0
            ensemble_factors = ["No individual model data"]
        detailed_scores['ensemble_agreement'] = ensemble_score

        for f in ensemble_factors:
            if any(x in f.lower() for x in ['caution', 'disagree', 'outlier']):
                all_factors_negative.append(f)
            else:
                all_factors_positive.append(f)

        # 2. Line Movement Score
        if opening_odds is not None and current_odds is not None:
            line_score, line_factors = self.calculate_line_movement_score(
                model_probability, opening_odds, current_odds,
                home_away == "home", public_betting_pct,
                is_reverse_line_movement, is_steam_move
            )
        else:
            line_score = 50.0
            line_factors = ["No line movement data"]
        detailed_scores['line_movement'] = line_score

        for f in line_factors:
            if any(x in f.lower() for x in ['caution', 'against', 'trap']):
                all_factors_negative.append(f)
            else:
                all_factors_positive.append(f)

        # 3. Feature Stability Score
        stability_score, stability_factors = self.calculate_feature_stability_score(
            recent_form or {}, season_averages or {}, games_played
        )
        detailed_scores['feature_stability'] = stability_score

        for f in stability_factors:
            if any(x in f.lower() for x in ['small', 'unstable', 'deviate']):
                all_factors_negative.append(f)
            else:
                all_factors_positive.append(f)

        # 4. Recency Score
        recency_score, recency_factors = self.calculate_recency_score(
            training_data_age_days, last_game_days_ago, is_back_to_back
        )
        detailed_scores['recency'] = recency_score

        for f in recency_factors:
            if any(x in f.lower() for x in ['stale', 'old', 'uncertainty']):
                all_factors_negative.append(f)
            else:
                all_factors_positive.append(f)

        # 5. Situational Score
        situational_score, situational_factors = self.calculate_situational_score(
            home_away, division_game, playoff_implications,
            injury_impact_score, travel_fatigue_score
        )
        detailed_scores['situational'] = situational_score

        for f in situational_factors:
            if any(x in f.lower() for x in ['impact', 'fatigue', 'variance']):
                all_factors_negative.append(f)
            else:
                all_factors_positive.append(f)

        # 6. Probability Confidence Score
        prob_score, prob_factors = self.calculate_probability_confidence_score(
            raw_probability or model_probability, model_probability
        )
        detailed_scores['probability_confidence'] = prob_score

        for f in prob_factors:
            if any(x in f.lower() for x in ['weak', 'large']):
                all_factors_negative.append(f)
            else:
                all_factors_positive.append(f)

        # Calculate weighted overall score
        overall_score = sum(
            detailed_scores[k] * self.WEIGHTS[k]
            for k in self.WEIGHTS.keys()
        )

        # Apply edge magnitude bonus/penalty
        if edge > 0.10:
            overall_score = min(100, overall_score + 5)
            all_factors_positive.append(f"Large edge ({edge:.1%})")
        elif edge < 0.02:
            overall_score = max(0, overall_score - 10)
            all_factors_negative.append(f"Small edge ({edge:.1%})")

        # Determine tier
        if overall_score >= 85:
            tier = EdgeTier.ELITE
        elif overall_score >= 70:
            tier = EdgeTier.STRONG
        elif overall_score >= 55:
            tier = EdgeTier.MODERATE
        elif overall_score >= 40:
            tier = EdgeTier.WEAK
        else:
            tier = EdgeTier.AVOID

        # Calculate confidence interval for edge
        # Higher quality = tighter interval
        ci_width = 0.05 * (1 + (100 - overall_score) / 50)
        confidence_interval = (
            max(0, edge - ci_width),
            min(1, edge + ci_width)
        )

        return EdgeQualityResult(
            overall_score=overall_score,
            tier=tier,
            ensemble_agreement_score=ensemble_score,
            line_movement_score=line_score,
            feature_stability_score=stability_score,
            recency_score=recency_score,
            situational_score=situational_score,
            recommended_kelly_multiplier=self.KELLY_MULTIPLIERS[tier],
            confidence_interval=confidence_interval,
            risk_factors=all_factors_negative,
            positive_factors=all_factors_positive,
            detailed_breakdown=detailed_scores,
        )


class DynamicKellyCalculator:
    """
    Dynamic Kelly Criterion calculator that adjusts stake sizing based on:
    - Edge quality score
    - Current drawdown level
    - Win/loss streak
    - Bankroll volatility
    - Confidence in probability estimate
    """

    def __init__(
        self,
        base_kelly_fraction: float = 0.25,  # Quarter Kelly as base
        max_bet_pct: float = 0.05,          # Max 5% of bankroll
        min_bet_pct: float = 0.005,         # Min 0.5% of bankroll
    ):
        """
        Initialize the calculator.

        Args:
            base_kelly_fraction: Fraction of full Kelly to use as baseline
            max_bet_pct: Maximum bet as percentage of bankroll
            min_bet_pct: Minimum bet as percentage of bankroll
        """
        self.base_kelly_fraction = base_kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.min_bet_pct = min_bet_pct

    def calculate_full_kelly(
        self,
        probability: float,
        decimal_odds: float,
    ) -> float:
        """
        Calculate full Kelly criterion bet size.

        Kelly % = (bp - q) / b
        where:
            b = decimal odds - 1 (net odds)
            p = probability of winning
            q = probability of losing (1 - p)

        Args:
            probability: Estimated probability of winning
            decimal_odds: Decimal odds (e.g., 2.0 for even money)

        Returns:
            Kelly percentage (can be negative if -EV)
        """
        if decimal_odds <= 1:
            return 0.0

        b = decimal_odds - 1  # Net odds
        p = probability
        q = 1 - p

        kelly = (b * p - q) / b
        return kelly

    def calculate_dynamic_kelly(
        self,
        probability: float,
        decimal_odds: float,
        edge_quality: EdgeQualityResult,
        current_drawdown: float = 0.0,
        recent_win_rate: float = 0.5,
        consecutive_losses: int = 0,
        bankroll_volatility: float = 0.1,
    ) -> Dict:
        """
        Calculate dynamically adjusted Kelly bet size.

        Args:
            probability: Model's win probability
            decimal_odds: Decimal odds for the bet
            edge_quality: EdgeQualityResult from scorer
            current_drawdown: Current drawdown percentage (0-1)
            recent_win_rate: Win rate over last N bets
            consecutive_losses: Current losing streak
            bankroll_volatility: Recent bankroll volatility

        Returns:
            Dict with bet sizing details
        """
        # Calculate base Kelly
        full_kelly = self.calculate_full_kelly(probability, decimal_odds)

        if full_kelly <= 0:
            return {
                'recommended_bet_pct': 0.0,
                'full_kelly': full_kelly,
                'base_kelly': 0.0,
                'adjustments': {'reason': 'Negative expected value'},
                'should_bet': False,
            }

        # Start with fractional Kelly
        base_kelly = full_kelly * self.base_kelly_fraction

        # Track adjustments
        adjustments = {}
        adjusted_kelly = base_kelly

        # 1. Edge Quality Adjustment
        quality_mult = edge_quality.recommended_kelly_multiplier
        adjusted_kelly *= quality_mult
        adjustments['edge_quality'] = {
            'score': edge_quality.overall_score,
            'tier': edge_quality.tier.value,
            'multiplier': quality_mult,
        }

        # 2. Drawdown Adjustment (protect during drawdowns)
        if current_drawdown >= 0.30:
            dd_mult = 0.25  # Extreme caution
            adjustments['drawdown'] = f"Severe drawdown ({current_drawdown:.1%}), using 25%"
        elif current_drawdown >= 0.20:
            dd_mult = 0.50
            adjustments['drawdown'] = f"Significant drawdown ({current_drawdown:.1%}), using 50%"
        elif current_drawdown >= 0.10:
            dd_mult = 0.75
            adjustments['drawdown'] = f"Moderate drawdown ({current_drawdown:.1%}), using 75%"
        else:
            dd_mult = 1.0
            adjustments['drawdown'] = "No drawdown adjustment"
        adjusted_kelly *= dd_mult

        # 3. Losing Streak Adjustment
        if consecutive_losses >= 5:
            streak_mult = 0.50
            adjustments['losing_streak'] = f"{consecutive_losses} consecutive losses, using 50%"
        elif consecutive_losses >= 3:
            streak_mult = 0.75
            adjustments['losing_streak'] = f"{consecutive_losses} consecutive losses, using 75%"
        else:
            streak_mult = 1.0
            adjustments['losing_streak'] = "No streak adjustment"
        adjusted_kelly *= streak_mult

        # 4. Recent Performance Adjustment
        if recent_win_rate < 0.45:
            perf_mult = 0.75
            adjustments['recent_performance'] = f"Below expectation ({recent_win_rate:.1%}), using 75%"
        elif recent_win_rate > 0.60:
            perf_mult = 1.10  # Slight boost when running well
            adjustments['recent_performance'] = f"Running well ({recent_win_rate:.1%}), using 110%"
        else:
            perf_mult = 1.0
            adjustments['recent_performance'] = "Normal recent performance"
        adjusted_kelly *= perf_mult

        # 5. Volatility Adjustment
        if bankroll_volatility > 0.20:
            vol_mult = 0.75
            adjustments['volatility'] = f"High volatility ({bankroll_volatility:.1%}), using 75%"
        else:
            vol_mult = 1.0
            adjustments['volatility'] = "Normal volatility"
        adjusted_kelly *= vol_mult

        # Apply min/max constraints
        final_bet_pct = max(self.min_bet_pct, min(self.max_bet_pct, adjusted_kelly))

        # Determine if we should actually bet
        should_bet = (
            final_bet_pct >= self.min_bet_pct and
            edge_quality.tier != EdgeTier.AVOID and
            current_drawdown < 0.40  # Hard stop at 40% drawdown
        )

        return {
            'recommended_bet_pct': final_bet_pct if should_bet else 0.0,
            'full_kelly': full_kelly,
            'base_kelly': base_kelly,
            'adjusted_kelly': adjusted_kelly,
            'adjustments': adjustments,
            'should_bet': should_bet,
            'edge_quality_tier': edge_quality.tier.value,
            'confidence_interval': edge_quality.confidence_interval,
        }

    def calculate_bet_size(
        self,
        bankroll: float,
        probability: float,
        decimal_odds: float,
        edge_quality: EdgeQualityResult,
        **kwargs
    ) -> Dict:
        """
        Calculate actual dollar bet size.

        Args:
            bankroll: Current bankroll in dollars
            probability: Model's win probability
            decimal_odds: Decimal odds
            edge_quality: EdgeQualityResult
            **kwargs: Additional args for calculate_dynamic_kelly

        Returns:
            Dict with bet sizing including dollar amount
        """
        result = self.calculate_dynamic_kelly(
            probability, decimal_odds, edge_quality, **kwargs
        )

        if result['should_bet']:
            result['bet_amount'] = bankroll * result['recommended_bet_pct']
        else:
            result['bet_amount'] = 0.0

        return result


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / decimal_odds


# Example usage
if __name__ == "__main__":
    # Create scorer
    scorer = EdgeQualityScorer()

    # Example edge evaluation
    result = scorer.evaluate_edge(
        model_probability=0.58,
        implied_probability=0.52,
        individual_model_predictions={
            'lr': 0.56, 'rf': 0.59, 'gb': 0.58, 'xgb': 0.57, 'lgb': 0.59
        },
        raw_probability=0.60,
        opening_odds=-3.0,
        current_odds=-4.5,
        public_betting_pct=65,
        is_reverse_line_movement=True,
        games_played=40,
        training_data_age_days=15,
        last_game_days_ago=1,
        home_away="home",
    )

    print(f"\nEdge Quality Evaluation")
    print(f"=" * 50)
    print(f"Overall Score: {result.overall_score:.1f}/100")
    print(f"Tier: {result.tier.value.upper()}")
    print(f"Recommended Kelly Multiplier: {result.recommended_kelly_multiplier:.2f}")
    print(f"\nScore Breakdown:")
    for k, v in result.detailed_breakdown.items():
        print(f"  {k}: {v:.1f}")
    print(f"\nPositive Factors:")
    for f in result.positive_factors[:5]:
        print(f"  + {f}")
    print(f"\nRisk Factors:")
    for f in result.risk_factors[:5]:
        print(f"  - {f}")

    # Calculate bet size
    kelly_calc = DynamicKellyCalculator()
    bet = kelly_calc.calculate_bet_size(
        bankroll=10000,
        probability=0.58,
        decimal_odds=american_to_decimal(-110),
        edge_quality=result,
        current_drawdown=0.05,
        consecutive_losses=1,
    )

    print(f"\nBet Sizing")
    print(f"=" * 50)
    print(f"Should Bet: {bet['should_bet']}")
    print(f"Full Kelly: {bet['full_kelly']:.2%}")
    print(f"Recommended: {bet['recommended_bet_pct']:.2%}")
    print(f"Bet Amount: ${bet['bet_amount']:.2f}")
