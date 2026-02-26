"""
Bet Recommender - Final Recommendations with Reasoning

The decision layer that:
1. Pulls from all agents (Minutes Oracle, Lineup Intel, Calibration Tracker)
2. Calculates edge and optimal bet size
3. Generates human-readable recommendations with reasoning
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .edge_calculator import EdgeCalculator, EdgeResult
from .kelly_criterion import KellyCriterion, BetSize
from .bankroll_manager import BankrollManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceTier(Enum):
    """Confidence tier for recommendations."""
    STRONG = "strong"
    MODERATE = "moderate"
    MARGINAL = "marginal"
    PASS = "pass"


@dataclass
class BetRecommendation:
    """Complete bet recommendation with reasoning."""

    # Player and prop info
    player_name: str
    player_id: int | None
    team: str
    opponent: str
    prop_type: str
    line: float

    # Prediction
    prediction: float
    pick: str  # OVER or UNDER

    # Edge analysis
    edge_percentage: float
    ev_per_dollar: float
    model_probability: float

    # Recommendation
    recommendation: str  # BET, LEAN, PASS
    confidence_tier: ConfidenceTier
    suggested_units: float
    suggested_stake: float
    kelly_fraction: float

    # Optional fields with defaults
    odds: int = -110

    # Context
    minutes_predicted: float = 0.0
    minutes_uncertainty: str = "medium"
    injury_status: str = "healthy"
    is_starter: bool = True

    # Reasoning
    reasoning: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    game_date: str = ""

    def to_dict(self) -> dict:
        return {
            'player_name': self.player_name,
            'player_id': self.player_id,
            'team': self.team,
            'opponent': self.opponent,
            'prop_type': self.prop_type,
            'line': self.line,
            'odds': self.odds,
            'prediction': round(self.prediction, 2),
            'pick': self.pick,
            'edge_percentage': round(self.edge_percentage, 2),
            'ev_per_dollar': round(self.ev_per_dollar, 4),
            'model_probability': round(self.model_probability, 3),
            'recommendation': self.recommendation,
            'confidence_tier': self.confidence_tier.value,
            'suggested_units': round(self.suggested_units, 1),
            'suggested_stake': round(self.suggested_stake, 2),
            'kelly_fraction': round(self.kelly_fraction, 2),
            'minutes_predicted': round(self.minutes_predicted, 1),
            'minutes_uncertainty': self.minutes_uncertainty,
            'injury_status': self.injury_status,
            'is_starter': self.is_starter,
            'reasoning': self.reasoning,
            'risks': self.risks,
            'timestamp': self.timestamp,
            'game_date': self.game_date,
        }

    def format_summary(self) -> str:
        """Format a one-line summary."""
        return (
            f"{self.player_name} {self.pick} {self.line} {self.prop_type} | "
            f"{self.suggested_units:.1f}u | {self.edge_percentage:+.1f}% edge | "
            f"{self.confidence_tier.value.upper()}"
        )


class BetRecommender:
    """
    Generate bet recommendations from predictions.

    Integrates:
    - Edge calculation
    - Kelly criterion sizing
    - Minutes Oracle (uncertainty)
    - Lineup Intel (injury status)
    - Calibration Tracker (adjustments)
    """

    # Confidence tier settings
    TIER_SETTINGS = {
        ConfidenceTier.STRONG: {
            'min_edge': 0.05,
            'kelly_fraction': 0.50,
            'max_units': 3.0,
            'recommendation': 'BET',
        },
        ConfidenceTier.MODERATE: {
            'min_edge': 0.03,
            'kelly_fraction': 0.35,
            'max_units': 2.0,
            'recommendation': 'BET',
        },
        ConfidenceTier.MARGINAL: {
            'min_edge': 0.02,
            'kelly_fraction': 0.25,
            'max_units': 1.0,
            'recommendation': 'LEAN',
        },
        ConfidenceTier.PASS: {
            'min_edge': 0.0,
            'kelly_fraction': 0.0,
            'max_units': 0.0,
            'recommendation': 'PASS',
        },
    }

    # Risk factors
    RISK_FACTORS = {
        'high_spread': {
            'threshold': 9.0,
            'message': 'Large spread ({spread}) increases blowout risk',
        },
        'back_to_back': {
            'message': 'Back-to-back game may limit minutes',
        },
        'minutes_uncertainty': {
            'threshold': 'high',
            'message': 'High minutes uncertainty ({uncertainty})',
        },
        'injury_gtd': {
            'statuses': ['GTD', 'Questionable', 'Probable'],
            'message': 'Player injury status: {status}',
        },
        'bench_player': {
            'threshold': 20,
            'message': 'Bench player with inconsistent minutes',
        },
    }

    def __init__(
        self,
        bankroll: float = 1000,
        min_edge: float = 0.02,
        bankroll_manager: BankrollManager = None,
    ):
        """
        Initialize bet recommender.

        Args:
            bankroll: Current bankroll
            min_edge: Minimum edge to consider
            bankroll_manager: Optional BankrollManager instance
        """
        self.bankroll = bankroll
        self.min_edge = min_edge

        self.edge_calc = EdgeCalculator(min_edge_threshold=min_edge)
        self.kelly = KellyCriterion()
        self.bankroll_mgr = bankroll_manager or BankrollManager(bankroll)

        # Optional integrations (lazy loaded)
        self._minutes_oracle = None
        self._lineup_intel = None
        self._calibration = None

    def _load_integrations(self):
        """Lazy load optional integrations."""
        # Minutes Oracle
        if self._minutes_oracle is None:
            try:
                from minutes_oracle import MinutesPredictor
                # Load from models if available
                # self._minutes_oracle = MinutesPredictor.load('models/minutes_oracle.pkl')
            except ImportError:
                pass

        # Lineup Intel
        if self._lineup_intel is None:
            try:
                from lineup_intel import LineupIntelService
                self._lineup_intel = LineupIntelService()
            except ImportError:
                pass

        # Calibration Tracker
        if self._calibration is None:
            try:
                from calibration_tracker import CalibrationService
                self._calibration = CalibrationService()
            except ImportError:
                pass

    def _get_confidence_tier(self, edge: float) -> ConfidenceTier:
        """Determine confidence tier from edge."""
        if edge >= self.TIER_SETTINGS[ConfidenceTier.STRONG]['min_edge']:
            return ConfidenceTier.STRONG
        elif edge >= self.TIER_SETTINGS[ConfidenceTier.MODERATE]['min_edge']:
            return ConfidenceTier.MODERATE
        elif edge >= self.TIER_SETTINGS[ConfidenceTier.MARGINAL]['min_edge']:
            return ConfidenceTier.MARGINAL
        else:
            return ConfidenceTier.PASS

    def _build_reasoning(
        self,
        prediction: float,
        line: float,
        edge_result: EdgeResult,
        minutes_predicted: float,
        minutes_uncertainty: str,
        calibration_adj: float = 0,
        injury_status: str = "healthy",
    ) -> list[str]:
        """Build reasoning list for the recommendation."""
        reasoning = []

        # Prediction vs line
        diff = prediction - line
        if diff > 0:
            reasoning.append(
                f"Model predicts {prediction:.1f} vs line of {line:.1f} (+{diff:.1f} edge)"
            )
        else:
            reasoning.append(
                f"Model predicts {prediction:.1f} vs line of {line:.1f} ({diff:.1f} edge)"
            )

        # Edge quality
        reasoning.append(
            f"Edge: {edge_result.edge_percentage:+.1f}% "
            f"({edge_result.edge_quality} - EV ${edge_result.ev_per_dollar:+.3f}/dollar)"
        )

        # Minutes
        if minutes_predicted > 0:
            reasoning.append(
                f"Minutes projection: {minutes_predicted:.0f} min "
                f"(uncertainty: {minutes_uncertainty})"
            )

        # Calibration adjustment
        if abs(calibration_adj) > 0.1:
            direction = "up" if calibration_adj > 0 else "down"
            reasoning.append(
                f"Calibration adjustment: {calibration_adj:+.1f} (historical {direction}-prediction)"
            )

        # Injury status
        if injury_status != "healthy":
            reasoning.append(f"Injury status: {injury_status}")
        else:
            reasoning.append("No injury concerns, confirmed starter")

        return reasoning

    def _identify_risks(
        self,
        spread: float,
        is_b2b: bool,
        minutes_uncertainty: str,
        injury_status: str,
        minutes_predicted: float,
    ) -> list[str]:
        """Identify risk factors for the bet."""
        risks = []

        # High spread (blowout risk)
        if abs(spread) >= self.RISK_FACTORS['high_spread']['threshold']:
            risks.append(
                self.RISK_FACTORS['high_spread']['message'].format(spread=f"{spread:+.1f}")
            )

        # Back-to-back
        if is_b2b:
            risks.append(self.RISK_FACTORS['back_to_back']['message'])

        # Minutes uncertainty
        if minutes_uncertainty == 'high':
            risks.append(
                self.RISK_FACTORS['minutes_uncertainty']['message'].format(
                    uncertainty=minutes_uncertainty
                )
            )

        # Injury status
        if injury_status in self.RISK_FACTORS['injury_gtd']['statuses']:
            risks.append(
                self.RISK_FACTORS['injury_gtd']['message'].format(status=injury_status)
            )

        # Bench player
        if minutes_predicted < self.RISK_FACTORS['bench_player']['threshold']:
            risks.append(self.RISK_FACTORS['bench_player']['message'])

        return risks

    def analyze_prop(
        self,
        player_name: str,
        team: str,
        opponent: str,
        prop_type: str,
        line: float,
        prediction: float,
        odds: int = -110,
        model_confidence: float = None,
        player_id: int = None,
        game_date: str = None,
        game_context: dict = None,
    ) -> BetRecommendation:
        """
        Analyze a single prop and generate recommendation.

        Args:
            player_name: Player name
            team: Team abbreviation
            opponent: Opponent abbreviation
            prop_type: Prop type (points, rebounds, etc.)
            line: Betting line
            prediction: Model's prediction
            odds: American odds
            model_confidence: Model's confidence (0-100)
            player_id: Player ID
            game_date: Game date
            game_context: Additional game context

        Returns:
            BetRecommendation
        """
        game_date = game_date or datetime.now().strftime('%Y-%m-%d')
        game_context = game_context or {}

        # Load integrations
        self._load_integrations()

        # Default context values
        spread = game_context.get('spread', 0)
        is_b2b = game_context.get('is_b2b', False)
        minutes_predicted = game_context.get('minutes_predicted', 30)
        minutes_uncertainty = game_context.get('minutes_uncertainty', 'medium')
        injury_status = game_context.get('injury_status', 'healthy')
        is_starter = game_context.get('is_starter', True)
        calibration_adj = game_context.get('calibration_adjustment', 0)

        # Try to get lineup intel
        if self._lineup_intel:
            try:
                player_intel = self._lineup_intel.get_player_intel(player_name, team)
                if player_intel:
                    injury_status = player_intel.injury_status.value
                    is_starter = player_intel.is_starter
                    if player_intel.expected_minutes > 0:
                        minutes_predicted = player_intel.expected_minutes
                    minutes_uncertainty = player_intel.minutes_uncertainty
            except Exception as e:
                logger.debug(f"Could not get lineup intel: {e}")

        # Try to get calibration adjustment
        if self._calibration:
            try:
                calibrated = self._calibration.apply_adjustments(
                    predicted_value=prediction,
                    confidence=model_confidence or 50,
                    prop_type=prop_type,
                    position=game_context.get('position'),
                )
                calibration_adj = calibrated['total_value_adjustment']
                prediction = calibrated['adjusted_value']
            except Exception as e:
                logger.debug(f"Could not get calibration: {e}")

        # Determine pick (OVER or UNDER)
        pick = 'OVER' if prediction > line else 'UNDER'

        # Calculate edge
        # Convert prediction difference to probability
        diff = abs(prediction - line)
        prob_adjustment = diff * 0.04  # ~4% per point
        model_prob = 0.50 + prob_adjustment

        if model_confidence:
            # Blend with model confidence
            conf_prob = model_confidence / 100
            model_prob = 0.7 * model_prob + 0.3 * conf_prob

        model_prob = max(0.05, min(0.95, model_prob))

        edge_result = self.edge_calc.calculate_edge(
            model_probability=model_prob,
            american_odds=odds,
        )

        # Determine confidence tier
        tier = self._get_confidence_tier(edge_result.edge)
        tier_settings = self.TIER_SETTINGS[tier]

        # Calculate bet size
        bet_size = self.kelly.calculate_from_american(
            win_probability=model_prob,
            american_odds=odds,
            bankroll=self.bankroll,
            edge=edge_result.edge,
        )

        # Cap at tier max units
        max_units = tier_settings['max_units']
        if bet_size.bet_units > max_units:
            bet_size.bet_units = max_units
            bet_size.bet_amount = max_units * (self.bankroll * 0.01)

        # Build reasoning
        reasoning = self._build_reasoning(
            prediction=prediction,
            line=line,
            edge_result=edge_result,
            minutes_predicted=minutes_predicted,
            minutes_uncertainty=minutes_uncertainty,
            calibration_adj=calibration_adj,
            injury_status=injury_status,
        )

        # Identify risks
        risks = self._identify_risks(
            spread=spread,
            is_b2b=is_b2b,
            minutes_uncertainty=minutes_uncertainty,
            injury_status=injury_status,
            minutes_predicted=minutes_predicted,
        )

        # Create recommendation
        return BetRecommendation(
            player_name=player_name,
            player_id=player_id,
            team=team,
            opponent=opponent,
            prop_type=prop_type,
            line=line,
            odds=odds,
            prediction=prediction,
            pick=pick,
            edge_percentage=edge_result.edge_percentage,
            ev_per_dollar=edge_result.ev_per_dollar,
            model_probability=model_prob,
            recommendation=tier_settings['recommendation'],
            confidence_tier=tier,
            suggested_units=bet_size.bet_units if bet_size.should_bet else 0,
            suggested_stake=bet_size.bet_amount if bet_size.should_bet else 0,
            kelly_fraction=tier_settings['kelly_fraction'],
            minutes_predicted=minutes_predicted,
            minutes_uncertainty=minutes_uncertainty,
            injury_status=injury_status,
            is_starter=is_starter,
            reasoning=reasoning,
            risks=risks,
            game_date=game_date,
        )


    def analyze_props(
        self,
        props: list[dict],
        min_tier: ConfidenceTier = ConfidenceTier.MARGINAL,
    ) -> list[BetRecommendation]:
        """
        Analyze multiple props and return sorted recommendations.

        Args:
            props: List of prop dictionaries with required fields
            min_tier: Minimum tier to include in results

        Returns:
            List of BetRecommendation sorted by edge
        """
        recommendations = []

        for prop in props:
            try:
                rec = self.analyze_prop(
                    player_name=prop['player_name'],
                    team=prop.get('team', ''),
                    opponent=prop.get('opponent', ''),
                    prop_type=prop['prop_type'],
                    line=prop['line'],
                    prediction=prop['prediction'],
                    odds=prop.get('odds', -110),
                    model_confidence=prop.get('confidence'),
                    player_id=prop.get('player_id'),
                    game_date=prop.get('game_date'),
                    game_context=prop.get('game_context', {}),
                )

                # Filter by minimum tier
                tier_order = [
                    ConfidenceTier.PASS,
                    ConfidenceTier.MARGINAL,
                    ConfidenceTier.MODERATE,
                    ConfidenceTier.STRONG,
                ]
                if tier_order.index(rec.confidence_tier) >= tier_order.index(min_tier):
                    recommendations.append(rec)

            except Exception as e:
                logger.error(f"Error analyzing prop for {prop.get('player_name')}: {e}")
                continue

        # Sort by edge (descending)
        recommendations.sort(key=lambda x: x.edge_percentage, reverse=True)

        return recommendations

    def get_top_picks(
        self,
        props: list[dict],
        max_picks: int = 5,
        max_total_units: float = 10.0,
    ) -> tuple[list[BetRecommendation], dict]:
        """
        Get top picks respecting total exposure limit.

        Args:
            props: List of prop dictionaries
            max_picks: Maximum number of picks
            max_total_units: Maximum total units to recommend

        Returns:
            Tuple of (recommendations, summary)
        """
        all_recs = self.analyze_props(props, min_tier=ConfidenceTier.MARGINAL)

        # Filter to BET recommendations only
        bet_recs = [r for r in all_recs if r.recommendation == 'BET']

        # Select top picks while respecting exposure
        selected = []
        total_units = 0.0

        for rec in bet_recs:
            if len(selected) >= max_picks:
                break
            if total_units + rec.suggested_units > max_total_units:
                # Try to fit with reduced size
                remaining = max_total_units - total_units
                if remaining >= 0.5:
                    rec.suggested_units = remaining
                    rec.suggested_stake = remaining * (self.bankroll * 0.01)
                else:
                    continue

            selected.append(rec)
            total_units += rec.suggested_units

        # Calculate summary
        total_stake = sum(r.suggested_stake for r in selected)
        expected_value = sum(r.ev_per_dollar * r.suggested_stake for r in selected)

        summary = {
            'total_picks': len(selected),
            'total_analyzed': len(props),
            'total_units': total_units,
            'total_stake': total_stake,
            'expected_value': expected_value,
            'by_tier': {
                'strong': len([r for r in selected if r.confidence_tier == ConfidenceTier.STRONG]),
                'moderate': len([r for r in selected if r.confidence_tier == ConfidenceTier.MODERATE]),
                'marginal': len([r for r in selected if r.confidence_tier == ConfidenceTier.MARGINAL]),
            },
        }

        return selected, summary


def format_recommendations_table(
    recommendations: list[BetRecommendation],
    summary: dict = None,
) -> str:
    """
    Format recommendations as ASCII table for CLI.

    Args:
        recommendations: List of recommendations
        summary: Optional summary dict

    Returns:
        Formatted string
    """
    if not recommendations:
        return "No recommendations found.\n"

    lines = []

    # Header
    lines.append("╔" + "═" * 68 + "╗")
    title = f"  TODAY'S RECOMMENDED BETS ({len(recommendations)} picks)"
    lines.append("║" + title.ljust(68) + "║")
    lines.append("╠" + "═" * 68 + "╣")

    # Recommendations
    for i, rec in enumerate(recommendations, 1):
        # Truncate player name if needed
        player = rec.player_name[:18]
        pick_line = f"{rec.pick} {rec.line}"

        tier_short = rec.confidence_tier.value[:3].upper()

        line = f"  {i}. {player:18} {pick_line:12} │ {rec.suggested_units:.1f}u │ {rec.edge_percentage:+.1f}% │ {tier_short}"
        lines.append("║" + line.ljust(68) + "║")

    lines.append("╚" + "═" * 68 + "╝")

    # Summary
    if summary:
        lines.append("")
        lines.append(f"Total exposure: {summary['total_units']:.1f}u (${summary['total_stake']:.2f})")
        lines.append(f"Expected value: ${summary['expected_value']:+.2f}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test bet recommender
    recommender = BetRecommender(bankroll=1000)

    print("=" * 60)
    print("BET RECOMMENDER TEST")
    print("=" * 60)

    # Test single prop
    print("\nSingle Prop Analysis:")
    rec = recommender.analyze_prop(
        player_name="LeBron James",
        team="LAL",
        opponent="BOS",
        prop_type="points",
        line=26.5,
        prediction=28.2,
        odds=-110,
        model_confidence=65,
        game_context={
            'spread': -3.5,
            'is_b2b': False,
            'minutes_predicted': 35,
            'minutes_uncertainty': 'low',
        }
    )

    print(f"\n{rec.player_name} {rec.prop_type.upper()}")
    print(f"  Line: {rec.line}")
    print(f"  Prediction: {rec.prediction}")
    print(f"  Pick: {rec.pick}")
    print(f"  Edge: {rec.edge_percentage:+.1f}%")
    print(f"  EV/dollar: ${rec.ev_per_dollar:+.3f}")
    print(f"  Recommendation: {rec.recommendation}")
    print(f"  Tier: {rec.confidence_tier.value}")
    print(f"  Suggested: {rec.suggested_units:.1f}u (${rec.suggested_stake:.2f})")

    print("\n  Reasoning:")
    for r in rec.reasoning:
        print(f"    - {r}")

    if rec.risks:
        print("\n  Risks:")
        for r in rec.risks:
            print(f"    - {r}")

    # Test batch analysis
    print("\n" + "=" * 60)
    print("Batch Analysis:")

    test_props = [
        {'player_name': 'LeBron James', 'team': 'LAL', 'opponent': 'BOS', 'prop_type': 'points',
         'line': 26.5, 'prediction': 28.2, 'confidence': 65},
        {'player_name': 'Nikola Jokic', 'team': 'DEN', 'opponent': 'PHX', 'prop_type': 'rebounds',
         'line': 11.5, 'prediction': 13.5, 'confidence': 70},
        {'player_name': 'Tyrese Haliburton', 'team': 'IND', 'opponent': 'MIA', 'prop_type': 'assists',
         'line': 9.5, 'prediction': 8.2, 'confidence': 62},
        {'player_name': 'Stephen Curry', 'team': 'GSW', 'opponent': 'LAC', 'prop_type': 'threes',
         'line': 4.5, 'prediction': 5.1, 'confidence': 55},
        {'player_name': 'Giannis Antetokounmpo', 'team': 'MIL', 'opponent': 'PHI', 'prop_type': 'points',
         'line': 29.5, 'prediction': 31.8, 'confidence': 68},
    ]

    picks, summary = recommender.get_top_picks(test_props, max_picks=3)

    print(format_recommendations_table(picks, summary))
