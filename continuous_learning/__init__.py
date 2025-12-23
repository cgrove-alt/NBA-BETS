"""
Continuous Learning Package for NBA Betting Model

This package provides automated continuous learning capabilities:
- Auto-settlement of predictions with actual game results
- Performance monitoring and drift detection
- Model versioning and rollback
- Automated retraining based on performance thresholds

Usage:
    from continuous_learning import ContinuousLearningOrchestrator

    # Create and start the orchestrator
    orchestrator = ContinuousLearningOrchestrator()
    orchestrator.start()

    # Or run operations manually
    orchestrator.run_settlement()
    orchestrator.run_drift_check()
    orchestrator.trigger_retraining()
"""

from .settlement_service import SettlementService
from .drift_detector import DriftDetector
from .model_registry import ModelRegistry
from .incremental_trainer import IncrementalTrainer
from .orchestrator import ContinuousLearningOrchestrator

__all__ = [
    'SettlementService',
    'DriftDetector',
    'ModelRegistry',
    'IncrementalTrainer',
    'ContinuousLearningOrchestrator',
]
