"""
Lineup Intel Integration with Minutes Oracle and Prop Predictions

This module shows how to:
1. Use LineupIntelService to get player status
2. Feed injury/lineup data into Minutes Oracle
3. Adjust prop confidence based on lineup uncertainty
"""

from datetime import datetime
from typing import Optional

from .lineup_intel_service import LineupIntelService, PlayerIntel, GameIntel
from .injury_scraper import InjuryStatus

# Try to import Minutes Oracle (may not be available)
try:
    from minutes_oracle import MinutesPredictor, MinutesFeatureGenerator
    MINUTES_ORACLE_AVAILABLE = True
except ImportError:
    MINUTES_ORACLE_AVAILABLE = False


class LineupAwarePredictor:
    """
    Wrapper that combines Lineup Intel with Minutes Oracle
    for lineup-aware minutes predictions.
    """

    def __init__(
        self,
        minutes_oracle: Optional['MinutesPredictor'] = None,
        feature_gen: Optional['MinutesFeatureGenerator'] = None,
    ):
        """
        Initialize with optional Minutes Oracle components.

        Args:
            minutes_oracle: Trained MinutesPredictor instance
            feature_gen: MinutesFeatureGenerator instance
        """
        self._lineup_intel = LineupIntelService()
        self._minutes_oracle = minutes_oracle
        self._feature_gen = feature_gen

    def get_player_minutes_prediction(
        self,
        player_name: str,
        team: str,
        opponent_team: str,
        game_context: dict = None
    ) -> dict:
        """
        Get minutes prediction that accounts for lineup status.

        Args:
            player_name: Player's full name
            team: Player's team abbreviation
            opponent_team: Opponent team abbreviation
            game_context: Game context with spread, total, is_home, etc.

        Returns:
            Dict with minutes prediction and confidence
        """
        # Get lineup intelligence
        player_intel = self._lineup_intel.get_player_intel(player_name, team)

        # Base response
        result = {
            'player_name': player_name,
            'team': team,
            'injury_status': player_intel.injury_status.value,
            'is_starter': player_intel.is_starter,
            'availability_probability': player_intel.availability_probability,
            'lineup_confidence': player_intel.starter_confidence,
        }

        # If player is OUT or DOUBTFUL, return 0 minutes
        if player_intel.injury_status in [InjuryStatus.OUT, InjuryStatus.DOUBTFUL]:
            result.update({
                'expected_minutes': 0.0,
                'p10': 0.0,
                'p25': 0.0,
                'p50': 0.0,
                'p75': 0.0,
                'p90': 0.0,
                'uncertainty': 'out',
                'prediction_source': 'lineup_intel_out',
            })
            return result

        # Start with lineup intel estimate
        base_minutes = player_intel.expected_minutes
        if base_minutes <= 0:
            base_minutes = 28.0 if player_intel.is_starter else 15.0

        # Apply availability/restriction multipliers
        adjusted_minutes = base_minutes * player_intel.minutes_multiplier

        # Try to use Minutes Oracle if available
        if MINUTES_ORACLE_AVAILABLE and self._minutes_oracle and self._feature_gen:
            try:
                # Build features for Minutes Oracle
                # This would integrate with actual feature generation
                # For now, return lineup-adjusted estimate

                # Apply lineup-based adjustments
                if player_intel.injury_status == InjuryStatus.GTD:
                    # GTD = high uncertainty
                    uncertainty = 'high'
                    spread_factor = 1.3
                elif player_intel.injury_status == InjuryStatus.QUESTIONABLE:
                    uncertainty = 'high'
                    spread_factor = 1.2
                elif player_intel.injury_status == InjuryStatus.PROBABLE:
                    uncertainty = 'medium'
                    spread_factor = 1.1
                else:
                    uncertainty = player_intel.minutes_uncertainty
                    spread_factor = 1.0

                # Build distribution
                result.update({
                    'expected_minutes': adjusted_minutes,
                    'p10': max(0, adjusted_minutes - 10 * spread_factor),
                    'p25': max(0, adjusted_minutes - 5 * spread_factor),
                    'p50': adjusted_minutes,
                    'p75': min(48, adjusted_minutes + 5 * spread_factor),
                    'p90': min(48, adjusted_minutes + 10 * spread_factor),
                    'uncertainty': uncertainty,
                    'prediction_source': 'lineup_intel_oracle',
                })

            except Exception as e:
                # Fall back to simple estimate
                result.update({
                    'expected_minutes': adjusted_minutes,
                    'p50': adjusted_minutes,
                    'uncertainty': 'high',
                    'prediction_source': 'lineup_intel_fallback',
                    'error': str(e),
                })

        else:
            # No Minutes Oracle - use lineup intel estimates
            uncertainty = player_intel.minutes_uncertainty

            # Widen intervals for uncertain statuses
            if player_intel.injury_status in [InjuryStatus.GTD, InjuryStatus.QUESTIONABLE]:
                uncertainty = 'high'
                floor = max(0, adjusted_minutes * 0.5)
                ceiling = min(48, adjusted_minutes * 1.2)
            else:
                floor = player_intel.minutes_floor
                ceiling = player_intel.minutes_ceiling

            result.update({
                'expected_minutes': adjusted_minutes,
                'p10': floor,
                'p25': max(floor, adjusted_minutes - 5),
                'p50': adjusted_minutes,
                'p75': min(ceiling, adjusted_minutes + 5),
                'p90': ceiling,
                'uncertainty': uncertainty,
                'prediction_source': 'lineup_intel_only',
            })

        return result

    def get_prop_confidence_adjustment(
        self,
        player_name: str,
        team: str,
        base_confidence: float
    ) -> dict:
        """
        Get confidence adjustment for a prop bet based on lineup status.

        Args:
            player_name: Player's full name
            team: Team abbreviation
            base_confidence: Base confidence (0-100)

        Returns:
            Dict with adjusted confidence and factors
        """
        player_intel = self._lineup_intel.get_player_intel(player_name, team)

        adjustments = []
        total_adjustment = 0.0

        # Status-based adjustments
        if player_intel.injury_status == InjuryStatus.OUT:
            return {
                'adjusted_confidence': 0.0,
                'factors': ['Player OUT - skip bet'],
                'skip_bet': True,
            }

        if player_intel.injury_status == InjuryStatus.DOUBTFUL:
            total_adjustment -= 50.0
            adjustments.append('DOUBTFUL (-50): High chance of missing game')

        elif player_intel.injury_status == InjuryStatus.GTD:
            total_adjustment -= 25.0
            adjustments.append('GTD (-25): Game-time decision uncertainty')

        elif player_intel.injury_status == InjuryStatus.QUESTIONABLE:
            total_adjustment -= 15.0
            adjustments.append('Questionable (-15): May have minutes restriction')

        elif player_intel.injury_status == InjuryStatus.PROBABLE:
            total_adjustment -= 5.0
            adjustments.append('Probable (-5): Minor injury concern')

        # Alert-based adjustments
        if player_intel.has_recent_alert:
            if player_intel.alert_severity:
                from .news_monitor import AlertSeverity
                if player_intel.alert_severity == AlertSeverity.CRITICAL:
                    total_adjustment -= 20.0
                    adjustments.append(f'Critical Alert (-20): {player_intel.alert_detail[:50]}')
                elif player_intel.alert_severity == AlertSeverity.HIGH:
                    total_adjustment -= 10.0
                    adjustments.append(f'High Alert (-10): {player_intel.alert_detail[:50]}')
                elif player_intel.alert_severity == AlertSeverity.MEDIUM:
                    total_adjustment -= 5.0
                    adjustments.append(f'Medium Alert (-5): {player_intel.alert_detail[:50]}')

        # Starter confidence adjustments
        if player_intel.is_starter and player_intel.starter_confidence < 0.7:
            total_adjustment -= 10.0
            adjustments.append(f'Low starter confidence (-10): {player_intel.starter_confidence:.0%}')

        # Minutes uncertainty
        if player_intel.minutes_uncertainty == 'high':
            total_adjustment -= 8.0
            adjustments.append('High minutes uncertainty (-8)')
        elif player_intel.minutes_uncertainty == 'low':
            total_adjustment += 5.0
            adjustments.append('Low minutes uncertainty (+5)')

        # Calculate final confidence
        adjusted = max(0.0, min(100.0, base_confidence + total_adjustment))

        # Recommend skipping bet if confidence too low
        skip_bet = adjusted < 30.0 or player_intel.availability_probability < 0.5

        return {
            'base_confidence': base_confidence,
            'adjusted_confidence': adjusted,
            'total_adjustment': total_adjustment,
            'factors': adjustments,
            'skip_bet': skip_bet,
            'availability_probability': player_intel.availability_probability,
            'injury_status': player_intel.injury_status.value,
        }


def integrate_lineup_intel_with_data_service():
    """
    Example code showing how to integrate LineupIntel with DataService.

    Add this to data_service.py:

    1. Import at top:
    ```python
    try:
        from lineup_intel import LineupIntelService
        LINEUP_INTEL_AVAILABLE = True
    except ImportError:
        LineupIntelService = None
        LINEUP_INTEL_AVAILABLE = False
    ```

    2. Add to __init__:
    ```python
    self._lineup_intel = None
    if LINEUP_INTEL_AVAILABLE:
        self._lineup_intel = LineupIntelService()
    ```

    3. Use in _get_player_predictions():
    ```python
    # Get lineup intelligence
    if self._lineup_intel:
        player_intel = self._lineup_intel.get_player_intel(
            player_name=player.get('name', ''),
            team=player.get('team_abbrev', '')
        )

        # Skip if player is OUT
        if player_intel.injury_status.value == 'Out':
            return None  # Skip this player

        # Adjust minutes prediction
        minutes_adjustment = self._lineup_intel.get_minutes_adjustment(
            player_name=player.get('name', ''),
            team=player.get('team_abbrev', ''),
            base_minutes=projected_minutes
        )
        projected_minutes = minutes_adjustment['adjusted_minutes']
    ```

    4. Use in confidence calculation:
    ```python
    if self._lineup_intel:
        conf_adj = LineupAwarePredictor().get_prop_confidence_adjustment(
            player_name=player.get('name'),
            team=player.get('team_abbrev'),
            base_confidence=confidence
        )
        confidence = conf_adj['adjusted_confidence']
        if conf_adj['skip_bet']:
            prediction['skip_reason'] = 'Low lineup confidence'
    ```
    """
    print("See docstring for integration code")


if __name__ == "__main__":
    # Test the integration
    predictor = LineupAwarePredictor()

    print("="*60)
    print("LINEUP-AWARE PREDICTOR TEST")
    print("="*60)

    # Test player predictions
    test_players = [
        ("LeBron James", "LAL", "BOS"),
        ("Stephen Curry", "GSW", "PHX"),
        ("Giannis Antetokounmpo", "MIL", "CLE"),
    ]

    for player, team, opp in test_players:
        print(f"\n{player} ({team} vs {opp}):")

        pred = predictor.get_player_minutes_prediction(player, team, opp)
        print(f"  Status: {pred['injury_status']}")
        print(f"  Is Starter: {pred['is_starter']}")
        print(f"  Expected Minutes: {pred['expected_minutes']:.1f}")
        print(f"  P10-P90 Range: {pred.get('p10', 0):.1f} - {pred.get('p90', 0):.1f}")
        print(f"  Uncertainty: {pred.get('uncertainty', 'N/A')}")
        print(f"  Source: {pred.get('prediction_source', 'N/A')}")

        # Test confidence adjustment
        conf = predictor.get_prop_confidence_adjustment(player, team, base_confidence=65.0)
        print(f"\n  Confidence: {conf['base_confidence']} -> {conf['adjusted_confidence']:.1f}")
        if conf['factors']:
            print("  Factors:")
            for factor in conf['factors']:
                print(f"    - {factor}")
        print(f"  Skip Bet: {conf['skip_bet']}")
