"""
Calibration Adjuster - Generate and Apply Calibration Adjustments

Responsibilities:
- Store calibration adjustments in database
- Apply adjustments to new predictions
- Track adjustment effectiveness over time
- Decay adjustments as they age
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

from .database import CalibrationDatabase
from .bias_analyzer import BiasAnalyzer, BiasReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationAdjustment:
    """Single calibration adjustment."""

    dimension: str  # prop_type, position, minutes_bucket, etc.
    dimension_value: str  # points, guard, starter, etc.
    bias: float  # The observed bias
    adjustment: float  # The correction to apply (usually -bias)
    confidence_multiplier: float = 1.0  # Multiply confidence by this

    # Evidence
    sample_size: int = 0
    hit_rate: float | None = None
    avg_error: float | None = None
    std_error: float | None = None

    def to_dict(self) -> dict:
        return {
            'dimension': self.dimension,
            'dimension_value': self.dimension_value,
            'bias': self.bias,
            'adjustment': self.adjustment,
            'confidence_multiplier': self.confidence_multiplier,
            'sample_size': self.sample_size,
            'hit_rate': self.hit_rate,
            'avg_error': self.avg_error,
            'std_error': self.std_error,
        }


class CalibrationAdjuster:
    """
    Generate and apply calibration adjustments.
    """

    # Minimum sample size to create adjustment
    MIN_SAMPLE_SIZE = 50

    # Minimum bias magnitude to warrant adjustment
    MIN_BIAS_THRESHOLD = 0.5

    # Maximum adjustment magnitude (cap extreme adjustments)
    MAX_ADJUSTMENT = 5.0

    # Adjustment decay rate (per day)
    DECAY_RATE = 0.02  # 2% per day

    def __init__(self, db: CalibrationDatabase = None):
        """
        Initialize calibration adjuster.

        Args:
            db: CalibrationDatabase instance
        """
        self.db = db or CalibrationDatabase()
        self.analyzer = BiasAnalyzer(db)
        self._adjustment_cache: dict[str, CalibrationAdjustment] = {}
        self._cache_loaded_at: datetime | None = None

    def _load_adjustments(self, force: bool = False):
        """Load active adjustments from database."""
        # Cache for 5 minutes
        if not force and self._cache_loaded_at:
            if datetime.now() - self._cache_loaded_at < timedelta(minutes=5):
                return

        active = self.db.get_active_adjustments()
        self._adjustment_cache = {}

        for adj in active:
            key = f"{adj['dimension']}:{adj['dimension_value']}"
            self._adjustment_cache[key] = CalibrationAdjustment(
                dimension=adj['dimension'],
                dimension_value=adj['dimension_value'],
                bias=adj['bias'],
                adjustment=adj['adjustment'],
                confidence_multiplier=adj.get('confidence_multiplier', 1.0),
                sample_size=adj['sample_size'],
                hit_rate=adj.get('hit_rate'),
                avg_error=adj.get('avg_error'),
                std_error=adj.get('std_error'),
            )

        self._cache_loaded_at = datetime.now()
        logger.debug(f"Loaded {len(self._adjustment_cache)} active adjustments")

    def generate_adjustments(
        self,
        start_date: str = None,
        end_date: str = None,
        save_to_db: bool = True,
    ) -> list[CalibrationAdjustment]:
        """
        Generate new calibration adjustments from historical data.

        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            save_to_db: Whether to save adjustments to database

        Returns:
            List of generated adjustments
        """
        # Run bias analysis
        report = self.analyzer.analyze(start_date=start_date, end_date=end_date)

        adjustments = []

        # Generate adjustments for each dimension
        dimensions = [
            ('prop_type', report.by_prop_type),
            ('position', report.by_position),
            ('minutes_bucket', report.by_minutes_bucket),
            ('game_type', report.by_game_type),
            ('day_type', report.by_day_type),
            ('player_tier', report.by_player_tier),
        ]

        for dimension_name, dimension_data in dimensions:
            for value, analysis in dimension_data.items():
                # Skip if insufficient data
                if analysis.sample_size < self.MIN_SAMPLE_SIZE:
                    logger.debug(f"Skipping {dimension_name}:{value} - insufficient data ({analysis.sample_size})")
                    continue

                # Skip if bias is too small
                if abs(analysis.bias) < self.MIN_BIAS_THRESHOLD:
                    logger.debug(f"Skipping {dimension_name}:{value} - bias too small ({analysis.bias:.2f})")
                    continue

                # Calculate adjustment (negative of bias, capped)
                raw_adjustment = -analysis.bias
                adjustment = max(-self.MAX_ADJUSTMENT, min(self.MAX_ADJUSTMENT, raw_adjustment))

                # Calculate confidence multiplier
                # Boost confidence for strong edges, reduce for negative edges
                if analysis.edge_quality == 'strong':
                    conf_mult = 1.1
                elif analysis.edge_quality == 'negative':
                    conf_mult = 0.8
                else:
                    conf_mult = 1.0

                adj = CalibrationAdjustment(
                    dimension=dimension_name,
                    dimension_value=value,
                    bias=analysis.bias,
                    adjustment=adjustment,
                    confidence_multiplier=conf_mult,
                    sample_size=analysis.sample_size,
                    hit_rate=analysis.hit_rate,
                    avg_error=analysis.avg_error,
                    std_error=analysis.std_error,
                )

                adjustments.append(adj)

                if save_to_db:
                    self.db.insert_adjustment(adj.to_dict())
                    logger.info(
                        f"Created adjustment: {dimension_name}:{value} "
                        f"bias={analysis.bias:.2f} adj={adjustment:.2f}"
                    )

        # Refresh cache
        if save_to_db:
            self._load_adjustments(force=True)

        return adjustments

    def get_adjustment(self, dimension: str, value: str) -> CalibrationAdjustment | None:
        """
        Get adjustment for a specific dimension/value.

        Args:
            dimension: Dimension name (prop_type, position, etc.)
            value: Dimension value (points, guard, etc.)

        Returns:
            CalibrationAdjustment if found, None otherwise
        """
        self._load_adjustments()
        key = f"{dimension}:{value}"
        return self._adjustment_cache.get(key)

    def apply_adjustments(
        self,
        predicted_value: float,
        confidence: float,
        prop_type: str,
        position: str = None,
        minutes_bucket: str = None,
        game_type: str = None,
        is_back_to_back: bool = False,
        player_tier: str = None,
    ) -> dict:
        """
        Apply all relevant adjustments to a prediction.

        Args:
            predicted_value: Raw predicted value
            confidence: Raw confidence (0-100)
            prop_type: Prop type (points, rebounds, etc.)
            position: Player position (guard, forward, center)
            minutes_bucket: Minutes bucket (starter, rotation, bench)
            game_type: Game type (favorite, underdog, etc.)
            is_back_to_back: Whether it's a back-to-back game

        Returns:
            Dict with adjusted_value, adjusted_confidence, adjustments_applied
        """
        self._load_adjustments()

        adjusted_value = predicted_value
        adjusted_confidence = confidence
        applied = []

        # Apply prop type adjustment (primary)
        prop_adj = self.get_adjustment('prop_type', prop_type)
        if prop_adj:
            adjusted_value += prop_adj.adjustment
            adjusted_confidence *= prop_adj.confidence_multiplier
            applied.append({
                'dimension': 'prop_type',
                'value': prop_type,
                'value_adjustment': prop_adj.adjustment,
                'confidence_multiplier': prop_adj.confidence_multiplier,
            })

        # Apply position adjustment (secondary, half weight)
        if position:
            pos_adj = self.get_adjustment('position', position)
            if pos_adj:
                adjusted_value += pos_adj.adjustment * 0.5  # Half weight
                adjusted_confidence *= (pos_adj.confidence_multiplier - 1) * 0.5 + 1
                applied.append({
                    'dimension': 'position',
                    'value': position,
                    'value_adjustment': pos_adj.adjustment * 0.5,
                    'confidence_multiplier': (pos_adj.confidence_multiplier - 1) * 0.5 + 1,
                })

        # Apply minutes bucket adjustment
        if minutes_bucket:
            min_adj = self.get_adjustment('minutes_bucket', minutes_bucket)
            if min_adj:
                adjusted_value += min_adj.adjustment * 0.5
                adjusted_confidence *= (min_adj.confidence_multiplier - 1) * 0.5 + 1
                applied.append({
                    'dimension': 'minutes_bucket',
                    'value': minutes_bucket,
                    'value_adjustment': min_adj.adjustment * 0.5,
                    'confidence_multiplier': (min_adj.confidence_multiplier - 1) * 0.5 + 1,
                })

        # Apply player tier adjustment (0.4 weight)
        if player_tier:
            tier_adj = self.get_adjustment('player_tier', player_tier)
            if tier_adj:
                adjusted_value += tier_adj.adjustment * 0.4
                adjusted_confidence *= (tier_adj.confidence_multiplier - 1) * 0.4 + 1
                applied.append({
                    'dimension': 'player_tier',
                    'value': player_tier,
                    'value_adjustment': tier_adj.adjustment * 0.4,
                    'confidence_multiplier': (tier_adj.confidence_multiplier - 1) * 0.4 + 1,
                })

        # Apply game type adjustment
        if game_type:
            game_adj = self.get_adjustment('game_type', game_type)
            if game_adj:
                adjusted_value += game_adj.adjustment * 0.3  # Lower weight
                adjusted_confidence *= (game_adj.confidence_multiplier - 1) * 0.3 + 1
                applied.append({
                    'dimension': 'game_type',
                    'value': game_type,
                    'value_adjustment': game_adj.adjustment * 0.3,
                    'confidence_multiplier': (game_adj.confidence_multiplier - 1) * 0.3 + 1,
                })

        # Apply B2B adjustment
        day_type = 'back_to_back' if is_back_to_back else 'regular'
        day_adj = self.get_adjustment('day_type', day_type)
        if day_adj:
            adjusted_value += day_adj.adjustment * 0.3
            adjusted_confidence *= (day_adj.confidence_multiplier - 1) * 0.3 + 1
            applied.append({
                'dimension': 'day_type',
                'value': day_type,
                'value_adjustment': day_adj.adjustment * 0.3,
                'confidence_multiplier': (day_adj.confidence_multiplier - 1) * 0.3 + 1,
            })

        # Ensure confidence stays in bounds
        adjusted_confidence = max(0, min(100, adjusted_confidence))

        return {
            'original_value': predicted_value,
            'adjusted_value': round(adjusted_value, 2),
            'original_confidence': confidence,
            'adjusted_confidence': round(adjusted_confidence, 1),
            'total_value_adjustment': round(adjusted_value - predicted_value, 2),
            'adjustments_applied': applied,
        }

    def get_all_active_adjustments(self) -> list[CalibrationAdjustment]:
        """Get all currently active adjustments."""
        self._load_adjustments(force=True)
        return list(self._adjustment_cache.values())

    def get_adjustment_summary(self) -> dict:
        """Get summary of all active adjustments."""
        adjustments = self.get_all_active_adjustments()

        summary = {
            'total_adjustments': len(adjustments),
            'by_dimension': {},
        }

        for adj in adjustments:
            dim = adj.dimension
            if dim not in summary['by_dimension']:
                summary['by_dimension'][dim] = []

            summary['by_dimension'][dim].append({
                'value': adj.dimension_value,
                'bias': adj.bias,
                'adjustment': adj.adjustment,
                'sample_size': adj.sample_size,
                'hit_rate': adj.hit_rate,
            })

        return summary

    def should_skip_bet(
        self,
        prop_type: str,
        position: str = None,
        confidence: float = None,
    ) -> tuple[bool, str]:
        """
        Check if a bet should be skipped based on historical performance.

        Args:
            prop_type: Prop type
            position: Player position
            confidence: Prediction confidence

        Returns:
            Tuple of (should_skip, reason)
        """
        self._load_adjustments()

        # Check prop type edge
        prop_adj = self.get_adjustment('prop_type', prop_type)
        if prop_adj and prop_adj.hit_rate is not None:
            if prop_adj.hit_rate < 0.48 and prop_adj.sample_size >= 50:
                return True, f"{prop_type} has negative historical edge ({prop_adj.hit_rate:.1%})"

        # Check position edge
        if position:
            pos_adj = self.get_adjustment('position', position)
            if pos_adj and pos_adj.hit_rate is not None:
                if pos_adj.hit_rate < 0.46 and pos_adj.sample_size >= 50:
                    return True, f"{position} {prop_type} props underperforming ({pos_adj.hit_rate:.1%})"

        # Check combined prop+position
        # This would require more sophisticated tracking

        return False, ""


if __name__ == "__main__":
    # Test the calibration adjuster
    adjuster = CalibrationAdjuster()

    # Generate adjustments (from existing data)
    print("Generating calibration adjustments...")
    adjustments = adjuster.generate_adjustments(save_to_db=True)
    print(f"Generated {len(adjustments)} adjustments")

    for adj in adjustments:
        print(f"  {adj.dimension}:{adj.dimension_value} "
              f"bias={adj.bias:.2f} adj={adj.adjustment:.2f} "
              f"n={adj.sample_size}")

    # Test applying adjustments
    print("\nTesting adjustment application...")
    result = adjuster.apply_adjustments(
        predicted_value=25.5,
        confidence=65.0,
        prop_type='points',
        position='forward',
        minutes_bucket='starter',
        game_type='favorite',
        is_back_to_back=False,
    )

    print(f"\nOriginal: {result['original_value']} @ {result['original_confidence']}%")
    print(f"Adjusted: {result['adjusted_value']} @ {result['adjusted_confidence']}%")
    print(f"Total adjustment: {result['total_value_adjustment']:+.2f}")

    if result['adjustments_applied']:
        print("\nAdjustments applied:")
        for adj in result['adjustments_applied']:
            print(f"  {adj['dimension']}:{adj['value']} -> "
                  f"value {adj['value_adjustment']:+.2f}, "
                  f"conf x{adj['confidence_multiplier']:.2f}")

    # Test skip check
    print("\nTesting skip check...")
    should_skip, reason = adjuster.should_skip_bet('points', 'guard', 55.0)
    print(f"Should skip: {should_skip}")
    if reason:
        print(f"Reason: {reason}")
