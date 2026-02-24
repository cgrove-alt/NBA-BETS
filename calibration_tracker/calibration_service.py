"""
Calibration Service - Main Integration Point

Combines all calibration tracking components:
- PredictionLogger: Log predictions
- OutcomeTracker: Record results
- BiasAnalyzer: Find biases
- CalibrationAdjuster: Apply corrections

Also provides:
- Nightly processing job
- Daily report generation
- Integration helpers for the prediction pipeline
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from .database import CalibrationDatabase
from .prediction_logger import PredictionLogger, PredictionRecord
from .outcome_tracker import OutcomeTracker, OutcomeRecord
from .bias_analyzer import BiasAnalyzer, BiasReport
from .calibration_adjuster import CalibrationAdjuster, CalibrationAdjustment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationService:
    """
    Main service for prediction calibration tracking.

    Provides a unified interface for:
    - Logging predictions
    - Recording outcomes
    - Analyzing biases
    - Applying adjustments
    - Running nightly jobs
    """

    def __init__(self, db_path: str = "data/calibration.db"):
        """
        Initialize calibration service.

        Args:
            db_path: Path to SQLite database
        """
        self.db = CalibrationDatabase(db_path)
        self.prediction_logger = PredictionLogger(self.db)
        self.outcome_tracker = OutcomeTracker(self.db)
        self.bias_analyzer = BiasAnalyzer(self.db)
        self.calibration_adjuster = CalibrationAdjuster(self.db)

        logger.info("CalibrationService initialized")

    # ========== PREDICTION LOGGING ==========

    def log_prediction(self, **kwargs) -> int:
        """
        Log a prediction.

        Args:
            **kwargs: Prediction parameters (see PredictionLogger.log_prediction)

        Returns:
            Prediction ID
        """
        return self.prediction_logger.log_prediction(**kwargs)

    def log_batch_predictions(self, predictions: list[dict]) -> list[int]:
        """
        Log multiple predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            List of prediction IDs
        """
        return self.prediction_logger.log_batch(predictions)

    def log_from_model_output(self, player: dict, prediction: dict,
                               game_context: dict = None, model_version: str = None) -> int:
        """
        Log prediction from model output format.

        Args:
            player: Player data dict
            prediction: Model prediction dict
            game_context: Game context dict
            model_version: Model version string

        Returns:
            Prediction ID
        """
        return self.prediction_logger.log_from_model_output(
            player, prediction, game_context, model_version
        )

    # ========== OUTCOME RECORDING ==========

    def record_outcome(self, prediction_id: int, actual_value: float, **kwargs) -> int:
        """
        Record outcome for a prediction.

        Args:
            prediction_id: ID of the prediction
            actual_value: Actual stat value
            **kwargs: Additional outcome data

        Returns:
            Outcome ID
        """
        return self.outcome_tracker.record_outcome(
            prediction_id=prediction_id,
            actual_value=actual_value,
            **kwargs
        )

    def process_game_outcomes(self, game_date: str) -> dict:
        """
        Fetch and record outcomes for all predictions on a date.

        Args:
            game_date: Date to process (YYYY-MM-DD)

        Returns:
            Processing results summary
        """
        return self.outcome_tracker.fetch_and_record_outcomes(game_date)

    # ========== ANALYSIS ==========

    def analyze_biases(self, start_date: str = None, end_date: str = None) -> BiasReport:
        """
        Run bias analysis.

        Args:
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            BiasReport with analysis results
        """
        return self.bias_analyzer.analyze(start_date=start_date, end_date=end_date)

    def get_calibration_report(self, days: int = 30) -> dict:
        """
        Get a formatted calibration report.

        Args:
            days: Number of days to analyze

        Returns:
            Formatted report dictionary
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        report = self.bias_analyzer.analyze(start_date=start_date, end_date=end_date)
        return report.to_dict()

    # ========== ADJUSTMENTS ==========

    def generate_calibration_adjustments(self, save_to_db: bool = True) -> list[CalibrationAdjustment]:
        """
        Generate new calibration adjustments from historical data.

        Args:
            save_to_db: Whether to save to database

        Returns:
            List of generated adjustments
        """
        return self.calibration_adjuster.generate_adjustments(save_to_db=save_to_db)

    def apply_adjustments(self, predicted_value: float, confidence: float,
                          prop_type: str, **kwargs) -> dict:
        """
        Apply calibration adjustments to a prediction.

        Args:
            predicted_value: Raw predicted value
            confidence: Raw confidence
            prop_type: Prop type
            **kwargs: Additional context (position, minutes_bucket, etc.)

        Returns:
            Dict with adjusted values
        """
        return self.calibration_adjuster.apply_adjustments(
            predicted_value=predicted_value,
            confidence=confidence,
            prop_type=prop_type,
            **kwargs
        )

    def get_active_adjustments(self) -> list[CalibrationAdjustment]:
        """Get all active calibration adjustments."""
        return self.calibration_adjuster.get_all_active_adjustments()

    def should_skip_bet(self, prop_type: str, position: str = None,
                        confidence: float = None) -> tuple[bool, str]:
        """
        Check if a bet should be skipped based on historical edge.

        Returns:
            Tuple of (should_skip, reason)
        """
        return self.calibration_adjuster.should_skip_bet(prop_type, position, confidence)

    # ========== NIGHTLY JOB ==========

    def run_nightly_job(self, game_date: str = None) -> dict:
        """
        Run the complete nightly calibration job.

        This should be run after all games complete (~1am ET).

        Steps:
        1. Fetch actual stats for completed games
        2. Match predictions to outcomes
        3. Expire unmatched predictions
        4. Re-generate calibration adjustments
        5. Generate daily report

        Args:
            game_date: Date to process (defaults to yesterday)

        Returns:
            Job execution summary
        """
        if not game_date:
            # Default to yesterday
            game_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        logger.info(f"Running nightly calibration job for {game_date}")

        results = {
            'game_date': game_date,
            'started_at': datetime.now().isoformat(),
            'steps': {},
        }

        # Step 1: Fetch and record outcomes
        logger.info("Step 1: Processing game outcomes...")
        outcome_results = self.process_game_outcomes(game_date)
        results['steps']['outcomes'] = outcome_results

        # Step 2: Expire unmatched predictions
        logger.info("Step 2: Expiring unmatched predictions...")
        self.outcome_tracker.expire_unmatched_predictions(game_date)
        results['steps']['expired'] = True

        # Step 3: Re-generate calibration adjustments
        logger.info("Step 3: Generating calibration adjustments...")
        adjustments = self.generate_calibration_adjustments(save_to_db=True)
        results['steps']['adjustments_generated'] = len(adjustments)

        # Step 4: Generate daily report
        logger.info("Step 4: Generating daily report...")
        report = self.analyze_biases()
        report_dict = report.to_dict()

        # Save report to database
        self.db.insert_daily_report(game_date, {
            'total_predictions': report.total_predictions,
            'matched_predictions': report.matched_predictions,
            'overall_hit_rate': report.overall_hit_rate,
            'overall_clv': report.overall_clv,
            **report_dict
        })
        results['steps']['report_saved'] = True

        # Step 5: Log summary
        results['completed_at'] = datetime.now().isoformat()
        results['summary'] = {
            'predictions_matched': outcome_results.get('matched', 0),
            'predictions_not_found': outcome_results.get('not_found', 0),
            'predictions_dnp': outcome_results.get('dnp', 0),
            'adjustments_created': len(adjustments),
            'overall_hit_rate': report.overall_hit_rate,
        }

        logger.info(f"Nightly job complete: {results['summary']}")

        return results

    # ========== DAILY REPORT ==========

    def get_daily_report(self, report_date: str) -> Optional[dict]:
        """
        Get a daily calibration report.

        Args:
            report_date: Date of report (YYYY-MM-DD)

        Returns:
            Report dictionary or None
        """
        report = self.db.get_daily_report(report_date)
        if report:
            return report.get('report')
        return None

    def get_recent_reports(self, limit: int = 7) -> list[dict]:
        """
        Get recent daily reports.

        Args:
            limit: Number of reports to return

        Returns:
            List of report summaries
        """
        reports = self.db.get_recent_reports(limit)
        return [
            {
                'date': r['report_date'],
                'hit_rate': r.get('overall_hit_rate'),
                'clv': r.get('overall_clv'),
                'predictions': r.get('total_predictions'),
            }
            for r in reports
        ]

    # ========== INTEGRATION HELPERS ==========

    def create_calibrated_prediction(
        self,
        player: dict,
        prop_type: str,
        raw_prediction: float,
        prop_line: float,
        raw_confidence: float,
        game_context: dict = None,
        log_prediction: bool = True,
    ) -> dict:
        """
        Create a fully calibrated prediction.

        This is the main integration point for the prediction pipeline.
        It applies all calibration adjustments and optionally logs the prediction.

        Args:
            player: Player data dict with id, name, position, team
            prop_type: Prop type
            raw_prediction: Raw model prediction
            prop_line: Betting line
            raw_confidence: Raw model confidence
            game_context: Game context dict
            log_prediction: Whether to log this prediction

        Returns:
            Dict with calibrated prediction details
        """
        game_context = game_context or {}

        # Determine classification buckets
        position = player.get('position', 'forward')
        if position in ['PG', 'SG', 'G', 'G-F']:
            position_group = 'guard'
        elif position in ['C', 'C-F']:
            position_group = 'center'
        else:
            position_group = 'forward'

        projected_minutes = player.get('projected_minutes', 28)
        if projected_minutes >= 30:
            minutes_bucket = 'starter'
        elif projected_minutes >= 20:
            minutes_bucket = 'rotation'
        else:
            minutes_bucket = 'bench'

        spread = game_context.get('spread', 0)
        is_home = game_context.get('is_home', True)
        if spread is not None:
            team_spread = -spread if is_home else spread
            if team_spread < -8:
                game_type = 'heavy_favorite'
            elif team_spread < -3:
                game_type = 'favorite'
            elif team_spread > 8:
                game_type = 'heavy_underdog'
            elif team_spread > 3:
                game_type = 'underdog'
            else:
                game_type = 'close_game'
        else:
            game_type = 'unknown'

        is_b2b = game_context.get('is_back_to_back', False)

        # Classify player tier from projected minutes
        projected_minutes = player.get('projected_minutes', 28)
        if projected_minutes >= 32:
            player_tier = 'star'
        elif projected_minutes >= 24:
            player_tier = 'starter'
        else:
            player_tier = 'role_player'

        # Apply calibration adjustments
        calibrated = self.apply_adjustments(
            predicted_value=raw_prediction,
            confidence=raw_confidence,
            prop_type=prop_type,
            position=position_group,
            minutes_bucket=minutes_bucket,
            game_type=game_type,
            is_back_to_back=is_b2b,
            player_tier=player_tier,
        )

        # Check if should skip
        should_skip, skip_reason = self.should_skip_bet(
            prop_type=prop_type,
            position=position_group,
            confidence=calibrated['adjusted_confidence'],
        )

        # Calculate edge
        predicted_over = calibrated['adjusted_value'] > prop_line
        edge = abs(calibrated['adjusted_value'] - prop_line) / prop_line * 100 if prop_line else 0

        result = {
            'player_id': player.get('id') or player.get('player_id'),
            'player_name': player.get('name') or player.get('player_name'),
            'team': player.get('team') or player.get('team_abbrev'),
            'prop_type': prop_type,
            'prop_line': prop_line,
            'raw_prediction': raw_prediction,
            'calibrated_prediction': calibrated['adjusted_value'],
            'raw_confidence': raw_confidence,
            'calibrated_confidence': calibrated['adjusted_confidence'],
            'predicted_over': predicted_over,
            'edge': round(edge, 2),
            'should_skip': should_skip,
            'skip_reason': skip_reason,
            'adjustments_applied': calibrated['adjustments_applied'],
            'classification': {
                'position': position_group,
                'minutes_bucket': minutes_bucket,
                'game_type': game_type,
                'is_back_to_back': is_b2b,
            },
        }

        # Log prediction if requested
        if log_prediction and not should_skip:
            pred_id = self.log_prediction(
                player_id=result['player_id'],
                player_name=result['player_name'],
                team=result['team'],
                opponent=game_context.get('opponent', ''),
                game_date=game_context.get('game_date', datetime.now().strftime('%Y-%m-%d')),
                prop_type=prop_type,
                predicted_value=calibrated['adjusted_value'],
                prop_line=prop_line,
                predicted_over_prob=0.5 + (edge / 100) if predicted_over else 0.5 - (edge / 100),
                confidence=calibrated['adjusted_confidence'],
                edge=edge,
                position=position_group,
                minutes_predicted=projected_minutes,
                is_home=is_home,
                spread=spread,
                total=game_context.get('total'),
                is_back_to_back=is_b2b,
                days_rest=game_context.get('days_rest'),
            )
            result['prediction_id'] = pred_id

        return result


# Convenience function for quick access
def get_calibration_service() -> CalibrationService:
    """Get a CalibrationService instance."""
    return CalibrationService()


if __name__ == "__main__":
    # Test the calibration service
    service = CalibrationService()

    print("="*60)
    print("CALIBRATION SERVICE TEST")
    print("="*60)

    # Test creating a calibrated prediction
    print("\nCreating calibrated prediction...")
    result = service.create_calibrated_prediction(
        player={
            'id': 2544,
            'name': 'LeBron James',
            'position': 'F',
            'team': 'LAL',
            'projected_minutes': 35.0,
        },
        prop_type='points',
        raw_prediction=27.5,
        prop_line=26.5,
        raw_confidence=65.0,
        game_context={
            'opponent': 'BOS',
            'game_date': '2024-01-15',
            'is_home': True,
            'spread': -3.5,
            'total': 225.5,
            'is_back_to_back': False,
        },
        log_prediction=True,
    )

    print(f"\nPlayer: {result['player_name']}")
    print(f"Prop: {result['prop_type']} @ {result['prop_line']}")
    print(f"Raw prediction: {result['raw_prediction']}")
    print(f"Calibrated prediction: {result['calibrated_prediction']}")
    print(f"Raw confidence: {result['raw_confidence']}")
    print(f"Calibrated confidence: {result['calibrated_confidence']}")
    print(f"Predicted: {'OVER' if result['predicted_over'] else 'UNDER'}")
    print(f"Edge: {result['edge']}%")
    print(f"Should skip: {result['should_skip']}")
    if result.get('prediction_id'):
        print(f"Logged as prediction ID: {result['prediction_id']}")

    print(f"\nClassification:")
    for k, v in result['classification'].items():
        print(f"  {k}: {v}")

    if result['adjustments_applied']:
        print(f"\nAdjustments applied:")
        for adj in result['adjustments_applied']:
            print(f"  {adj['dimension']}:{adj['value']}")

    # Test getting calibration report
    print("\n" + "="*60)
    print("Getting calibration report...")
    report = service.get_calibration_report(days=30)

    print(f"\nOverall: {report['overall']['predictions']} predictions")
    print(f"Hit Rate: {report['overall']['hit_rate']}")
    print(f"CLV: {report['overall']['clv_avg']}")

    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations'][:5]:
            print(f"  - {rec}")
