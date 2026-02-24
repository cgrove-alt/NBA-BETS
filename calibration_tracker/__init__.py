"""
Calibration Tracker - Prediction Performance Tracking and Bias Analysis

This module provides:
1. PredictionLogger - Log every prediction with full context
2. OutcomeTracker - Record actual results after games complete
3. BiasAnalyzer - Find systematic over/under predictions
4. CalibrationAdjuster - Generate adjustments for future predictions
5. CalibrationService - Main service integrating all components

Usage:
    from calibration_tracker import CalibrationService

    service = CalibrationService()

    # Log a prediction
    service.log_prediction(
        player_id=123,
        player_name="LeBron James",
        prop_type="points",
        predicted_value=27.5,
        prop_line=26.5,
        predicted_over_prob=0.58,
        ...
    )

    # After game, record outcome
    service.record_outcome(
        prediction_id=1,
        actual_value=29,
        actual_minutes=35.2,
    )

    # Generate calibration report
    report = service.generate_calibration_report()
"""

from .database import CalibrationDatabase
from .prediction_logger import PredictionLogger, PredictionRecord
from .outcome_tracker import OutcomeTracker, OutcomeRecord
from .bias_analyzer import BiasAnalyzer, BiasReport
from .calibration_adjuster import CalibrationAdjuster, CalibrationAdjustment
from .calibration_service import CalibrationService
from .weekly_report import WeeklyReportGenerator

__all__ = [
    'CalibrationDatabase',
    'PredictionLogger',
    'PredictionRecord',
    'OutcomeTracker',
    'OutcomeRecord',
    'BiasAnalyzer',
    'BiasReport',
    'CalibrationAdjuster',
    'CalibrationAdjustment',
    'CalibrationService',
    'WeeklyReportGenerator',
]
