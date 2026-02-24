"""
Bias Analyzer - Find Systematic Over/Under Predictions

Analyzes predictions across multiple dimensions to identify:
- Which prop types are we over/under predicting?
- Which positions are we best/worst at?
- How does game context affect accuracy?
- Where are our biggest edges and leaks?

Dimensions analyzed:
- Prop type (points, rebounds, assists, threes, pra)
- Position (guard, forward, center)
- Minutes bucket (bench, rotation, starter)
- Game type (favorite, underdog, close, blowout)
- Day type (regular, back-to-back)
- Confidence level (high, medium, low)
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .database import CalibrationDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DimensionAnalysis:
    """Analysis results for a single dimension value."""

    dimension: str
    value: str
    sample_size: int

    # Performance metrics
    hit_rate: float  # Percentage of correct predictions
    avg_error: float  # Average (predicted - actual)
    std_error: float  # Standard deviation of error
    mae: float  # Mean absolute error

    # Bias metrics
    bias: float  # Systematic bias (positive = overpredict)
    bias_significance: float  # Statistical significance (0-1)

    # Edge metrics
    clv_avg: float  # Average closing line value
    roi_estimate: float  # Estimated ROI

    # Classification
    edge_quality: str  # strong, moderate, neutral, negative

    def to_dict(self) -> dict:
        return {
            'dimension': self.dimension,
            'value': self.value,
            'sample_size': self.sample_size,
            'hit_rate': round(self.hit_rate, 3),
            'avg_error': round(self.avg_error, 2),
            'std_error': round(self.std_error, 2),
            'mae': round(self.mae, 2),
            'bias': round(self.bias, 2),
            'bias_significance': round(self.bias_significance, 2),
            'clv_avg': round(self.clv_avg, 3) if self.clv_avg else None,
            'roi_estimate': round(self.roi_estimate, 3),
            'edge_quality': self.edge_quality,
        }


@dataclass
class BiasReport:
    """Complete bias analysis report."""

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    analysis_period_start: str = ""
    analysis_period_end: str = ""

    # Overall stats
    total_predictions: int = 0
    matched_predictions: int = 0
    overall_hit_rate: float = 0.0
    overall_clv: float = 0.0
    overall_roi: float = 0.0

    # Dimension analyses
    by_prop_type: dict[str, DimensionAnalysis] = field(default_factory=dict)
    by_position: dict[str, DimensionAnalysis] = field(default_factory=dict)
    by_minutes_bucket: dict[str, DimensionAnalysis] = field(default_factory=dict)
    by_game_type: dict[str, DimensionAnalysis] = field(default_factory=dict)
    by_day_type: dict[str, DimensionAnalysis] = field(default_factory=dict)
    by_confidence: dict[str, DimensionAnalysis] = field(default_factory=dict)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    adjustments: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'generated_at': self.generated_at,
            'analysis_period': {
                'start': self.analysis_period_start,
                'end': self.analysis_period_end,
            },
            'overall': {
                'predictions': self.total_predictions,
                'matched': self.matched_predictions,
                'hit_rate': f"{self.overall_hit_rate:.1%}",
                'clv_avg': f"{self.overall_clv:+.2%}",
                'roi_estimate': f"{self.overall_roi:+.2%}",
            },
            'by_prop_type': {k: v.to_dict() for k, v in self.by_prop_type.items()},
            'by_position': {k: v.to_dict() for k, v in self.by_position.items()},
            'by_minutes_bucket': {k: v.to_dict() for k, v in self.by_minutes_bucket.items()},
            'by_game_type': {k: v.to_dict() for k, v in self.by_game_type.items()},
            'by_day_type': {k: v.to_dict() for k, v in self.by_day_type.items()},
            'by_confidence': {k: v.to_dict() for k, v in self.by_confidence.items()},
            'recommendations': self.recommendations,
            'adjustments': self.adjustments,
        }


class BiasAnalyzer:
    """
    Analyze prediction biases across multiple dimensions.
    """

    # Confidence buckets
    CONFIDENCE_BUCKETS = {
        'high': (70, 100),
        'medium': (55, 70),
        'low': (0, 55),
    }

    # Minutes buckets
    MINUTES_BUCKETS = {
        'starter': (30, 48),
        'rotation': (20, 30),
        'bench': (0, 20),
    }

    # Game type thresholds
    SPREAD_THRESHOLDS = {
        'heavy_favorite': (-float('inf'), -8),
        'favorite': (-8, -3),
        'close_game': (-3, 3),
        'underdog': (3, 8),
        'heavy_underdog': (8, float('inf')),
    }

    # Minimum sample size for analysis
    MIN_SAMPLE_SIZE = 30

    def __init__(self, db: CalibrationDatabase = None):
        """
        Initialize bias analyzer.

        Args:
            db: CalibrationDatabase instance
        """
        self.db = db or CalibrationDatabase()

    def _classify_confidence(self, confidence: float) -> str:
        """Classify confidence into bucket."""
        if confidence is None:
            return 'unknown'
        for bucket, (low, high) in self.CONFIDENCE_BUCKETS.items():
            if low <= confidence < high:
                return bucket
        return 'high' if confidence >= 70 else 'low'

    def _classify_minutes(self, minutes: float) -> str:
        """Classify minutes into bucket."""
        if minutes is None:
            return 'unknown'
        for bucket, (low, high) in self.MINUTES_BUCKETS.items():
            if low <= minutes < high:
                return bucket
        return 'starter' if minutes >= 30 else 'bench'

    def _classify_game_type(self, spread: float, is_home: bool) -> str:
        """Classify game type based on spread."""
        if spread is None:
            return 'unknown'

        # Adjust spread for home/away perspective
        team_spread = -spread if is_home else spread

        for game_type, (low, high) in self.SPREAD_THRESHOLDS.items():
            if low <= team_spread < high:
                return game_type
        return 'close_game'

    def _calculate_edge_quality(self, hit_rate: float, clv: float, sample_size: int) -> str:
        """
        Classify edge quality based on hit rate and CLV.

        Args:
            hit_rate: Win rate (0-1)
            clv: Closing line value
            sample_size: Number of samples

        Returns:
            Edge quality classification
        """
        # Break-even is ~52.4% at -110 odds
        breakeven = 0.524

        if sample_size < self.MIN_SAMPLE_SIZE:
            return 'insufficient_data'

        if hit_rate >= 0.56 and (clv or 0) >= 0:
            return 'strong'
        elif hit_rate >= breakeven and (clv or 0) >= 0:
            return 'moderate'
        elif hit_rate >= breakeven - 0.02:
            return 'neutral'
        else:
            return 'negative'

    def _analyze_dimension(
        self,
        records: list[dict],
        dimension: str,
        value: str
    ) -> DimensionAnalysis:
        """
        Analyze a single dimension value.

        Args:
            records: Prediction records for this dimension
            dimension: Dimension name
            value: Dimension value

        Returns:
            DimensionAnalysis object
        """
        if not records:
            return DimensionAnalysis(
                dimension=dimension,
                value=value,
                sample_size=0,
                hit_rate=0.0,
                avg_error=0.0,
                std_error=0.0,
                mae=0.0,
                bias=0.0,
                bias_significance=0.0,
                clv_avg=0.0,
                roi_estimate=0.0,
                edge_quality='insufficient_data',
            )

        sample_size = len(records)

        # Calculate metrics
        hits = [r['hit'] for r in records]
        errors = [r['error'] for r in records if r['error'] is not None]
        clvs = [r['clv'] for r in records if r['clv'] is not None]

        hit_rate = sum(hits) / len(hits) if hits else 0.0
        avg_error = np.mean(errors) if errors else 0.0
        std_error = np.std(errors) if errors else 0.0
        mae = np.mean(np.abs(errors)) if errors else 0.0
        clv_avg = np.mean(clvs) if clvs else 0.0

        # Bias is the average error (positive = overpredicting)
        bias = avg_error

        # Calculate bias significance using t-test approximation
        if errors and len(errors) >= 2 and std_error > 0:
            t_stat = abs(avg_error) / (std_error / np.sqrt(len(errors)))
            # Rough p-value approximation (1 - significance)
            bias_significance = min(1.0, t_stat / 3.0)  # Scale to 0-1
        else:
            bias_significance = 0.0

        # Estimate ROI (rough approximation)
        # Assuming -110 odds, each win returns 0.909, each loss costs 1.0
        win_return = 0.909
        if sample_size > 0:
            wins = sum(hits)
            losses = sample_size - wins
            roi_estimate = (wins * win_return - losses) / sample_size
        else:
            roi_estimate = 0.0

        edge_quality = self._calculate_edge_quality(hit_rate, clv_avg, sample_size)

        return DimensionAnalysis(
            dimension=dimension,
            value=value,
            sample_size=sample_size,
            hit_rate=hit_rate,
            avg_error=avg_error,
            std_error=std_error,
            mae=mae,
            bias=bias,
            bias_significance=bias_significance,
            clv_avg=clv_avg,
            roi_estimate=roi_estimate,
            edge_quality=edge_quality,
        )

    def _group_by_dimension(
        self,
        records: list[dict],
        dimension_key: str,
        classifier=None
    ) -> dict[str, list[dict]]:
        """
        Group records by a dimension.

        Args:
            records: List of prediction records
            dimension_key: Key in record to group by
            classifier: Optional function to transform value

        Returns:
            Dict mapping dimension value to list of records
        """
        groups: dict[str, list[dict]] = {}

        for record in records:
            value = record.get(dimension_key)
            if classifier:
                value = classifier(value, record)
            if value is None:
                value = 'unknown'
            value = str(value)

            if value not in groups:
                groups[value] = []
            groups[value].append(record)

        return groups

    def analyze(
        self,
        start_date: str = None,
        end_date: str = None,
        min_confidence: float = None,
    ) -> BiasReport:
        """
        Run comprehensive bias analysis.

        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            min_confidence: Minimum confidence threshold

        Returns:
            BiasReport with all analysis results
        """
        # Default to last 30 days
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        # Fetch predictions with outcomes
        records = self.db.get_predictions_with_outcomes(
            start_date=start_date,
            end_date=end_date,
            min_confidence=min_confidence,
        )

        logger.info(f"Analyzing {len(records)} predictions from {start_date} to {end_date}")

        # Initialize report
        report = BiasReport(
            analysis_period_start=start_date,
            analysis_period_end=end_date,
            total_predictions=len(records),
            matched_predictions=len(records),
        )

        if not records:
            report.recommendations.append("No predictions to analyze. Log more predictions.")
            return report

        # Calculate overall metrics
        overall = self._analyze_dimension(records, 'overall', 'all')
        report.overall_hit_rate = overall.hit_rate
        report.overall_clv = overall.clv_avg
        report.overall_roi = overall.roi_estimate

        # Analyze by prop type
        by_prop = self._group_by_dimension(records, 'prop_type')
        for value, group_records in by_prop.items():
            report.by_prop_type[value] = self._analyze_dimension(group_records, 'prop_type', value)

        # Analyze by position
        by_position = self._group_by_dimension(records, 'position')
        for value, group_records in by_position.items():
            report.by_position[value] = self._analyze_dimension(group_records, 'position', value)

        # Analyze by minutes bucket
        def minutes_classifier(minutes, record):
            return self._classify_minutes(record.get('minutes_predicted'))

        by_minutes = self._group_by_dimension(records, 'minutes_predicted', minutes_classifier)
        for value, group_records in by_minutes.items():
            report.by_minutes_bucket[value] = self._analyze_dimension(group_records, 'minutes_bucket', value)

        # Analyze by game type
        def game_type_classifier(spread, record):
            return self._classify_game_type(record.get('spread'), record.get('is_home'))

        by_game = self._group_by_dimension(records, 'spread', game_type_classifier)
        for value, group_records in by_game.items():
            report.by_game_type[value] = self._analyze_dimension(group_records, 'game_type', value)

        # Analyze by day type (B2B vs regular)
        def day_type_classifier(_, record):
            return 'back_to_back' if record.get('is_back_to_back') else 'regular'

        by_day = self._group_by_dimension(records, 'is_back_to_back', day_type_classifier)
        for value, group_records in by_day.items():
            report.by_day_type[value] = self._analyze_dimension(group_records, 'day_type', value)

        # Analyze by confidence level
        def confidence_classifier(_, record):
            return self._classify_confidence(record.get('confidence'))

        by_confidence = self._group_by_dimension(records, 'confidence', confidence_classifier)
        for value, group_records in by_confidence.items():
            report.by_confidence[value] = self._analyze_dimension(group_records, 'confidence', value)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        report.adjustments = self._generate_adjustments(report)

        return report

    def _generate_recommendations(self, report: BiasReport) -> list[str]:
        """
        Generate actionable recommendations from analysis.

        Args:
            report: BiasReport to analyze

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Overall performance
        if report.overall_hit_rate < 0.50:
            recommendations.append(
                f"WARNING: Overall hit rate ({report.overall_hit_rate:.1%}) below break-even. "
                "Review model calibration."
            )
        elif report.overall_hit_rate >= 0.55:
            recommendations.append(
                f"POSITIVE: Strong overall hit rate ({report.overall_hit_rate:.1%}). "
                "Model is well-calibrated."
            )

        # Prop type recommendations
        for prop_type, analysis in report.by_prop_type.items():
            if analysis.sample_size < self.MIN_SAMPLE_SIZE:
                continue

            if analysis.edge_quality == 'strong':
                recommendations.append(
                    f"STRENGTH: {prop_type.title()} props show strong edge "
                    f"({analysis.hit_rate:.1%} hit rate). Consider increasing bet sizing."
                )
            elif analysis.edge_quality == 'negative':
                recommendations.append(
                    f"AVOID: {prop_type.title()} props showing negative edge "
                    f"({analysis.hit_rate:.1%} hit rate). Consider reducing or avoiding."
                )

            if abs(analysis.bias) > 1.0:
                direction = "over" if analysis.bias > 0 else "under"
                recommendations.append(
                    f"BIAS: Model {direction}predicts {prop_type} by {abs(analysis.bias):.1f}. "
                    f"Apply {-analysis.bias:+.1f} adjustment."
                )

        # Position recommendations
        for position, analysis in report.by_position.items():
            if analysis.sample_size < self.MIN_SAMPLE_SIZE:
                continue

            if analysis.edge_quality == 'strong':
                recommendations.append(
                    f"STRENGTH: {position.title()} props are your strongest segment "
                    f"({analysis.hit_rate:.1%} hit rate)."
                )
            elif analysis.edge_quality == 'negative':
                recommendations.append(
                    f"WEAKNESS: {position.title()} props underperforming "
                    f"({analysis.hit_rate:.1%} hit rate). Review position-specific factors."
                )

        # Game type recommendations
        for game_type, analysis in report.by_game_type.items():
            if analysis.sample_size < self.MIN_SAMPLE_SIZE:
                continue

            if game_type in ['heavy_favorite', 'heavy_underdog'] and analysis.edge_quality == 'negative':
                recommendations.append(
                    f"CAUTION: Poor performance in {game_type.replace('_', ' ')} games "
                    f"({analysis.hit_rate:.1%}). Blowout risk may be mispriced."
                )

        # Confidence recommendations
        high_conf = report.by_confidence.get('high')
        low_conf = report.by_confidence.get('low')

        if high_conf and low_conf:
            if high_conf.hit_rate > low_conf.hit_rate + 0.05:
                recommendations.append(
                    f"VALIDATED: High confidence picks ({high_conf.hit_rate:.1%}) "
                    f"outperform low confidence ({low_conf.hit_rate:.1%}). "
                    "Confidence calibration is working."
                )
            elif low_conf.hit_rate > high_conf.hit_rate:
                recommendations.append(
                    f"WARNING: Low confidence picks ({low_conf.hit_rate:.1%}) "
                    f"outperforming high confidence ({high_conf.hit_rate:.1%}). "
                    "Review confidence calculation."
                )

        return recommendations

    def _generate_adjustments(self, report: BiasReport) -> dict[str, float]:
        """
        Generate calibration adjustments from analysis.

        Args:
            report: BiasReport to analyze

        Returns:
            Dict mapping dimension:value to adjustment
        """
        adjustments = {}

        # Prop type adjustments
        for prop_type, analysis in report.by_prop_type.items():
            if analysis.sample_size >= self.MIN_SAMPLE_SIZE and abs(analysis.bias) > 0.5:
                # Adjustment is negative of bias (if we overpredict, adjust down)
                adjustments[f'prop_type:{prop_type}'] = -analysis.bias

        # Position adjustments (smaller, secondary)
        for position, analysis in report.by_position.items():
            if analysis.sample_size >= self.MIN_SAMPLE_SIZE and abs(analysis.bias) > 1.0:
                adjustments[f'position:{position}'] = -analysis.bias * 0.5  # Half weight

        return adjustments


if __name__ == "__main__":
    # Test the bias analyzer
    analyzer = BiasAnalyzer()

    # Run analysis
    report = analyzer.analyze()

    print("="*60)
    print("BIAS ANALYSIS REPORT")
    print("="*60)

    print(f"\nOverall: {report.total_predictions} predictions")
    print(f"Hit Rate: {report.overall_hit_rate:.1%}")
    print(f"CLV: {report.overall_clv:+.2%}")
    print(f"ROI: {report.overall_roi:+.2%}")

    print("\nBy Prop Type:")
    for prop_type, analysis in report.by_prop_type.items():
        print(f"  {prop_type}: {analysis.hit_rate:.1%} hit, {analysis.bias:+.1f} bias ({analysis.edge_quality})")

    print("\nBy Position:")
    for position, analysis in report.by_position.items():
        print(f"  {position}: {analysis.hit_rate:.1%} hit, {analysis.bias:+.1f} bias ({analysis.edge_quality})")

    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")

    print("\nAdjustments:")
    for key, adj in report.adjustments.items():
        print(f"  {key}: {adj:+.2f}")
