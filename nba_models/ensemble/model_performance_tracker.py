"""
Per-Model Performance Tracker (Phase 3.2)

Logs individual base-model predictions to JSONL and provides accuracy metrics
(MAE, RMSE, bias, Brier score) per (model_name, prop_type) over a rolling
window.  Distinct from DynamicEnsembleWeighter — this module focuses on
*storage and reporting*, while the weighter focuses on *weight computation*.

The log file grows by one line per settled prediction; it is never truncated
so full history is preserved for offline analysis.

Usage
-----
    from nba_models.ensemble.model_performance_tracker import ModelPerformanceTracker

    tracker = ModelPerformanceTracker(log_dir=Path("data/model_performance"))

    # At prediction time — record each base-model's raw output:
    tracker.log_predictions(
        date='2026-03-24',
        player='Stephen Curry',
        prop_type='threes',
        line=4.5,
        predictions={'xgboost': 5.1, 'lightgbm': 4.8, 'catboost': 5.3},
        ensemble_pred=5.07,
        american_odds=-110,
    )

    # After settlement — record the actual result:
    tracker.log_actual(
        date='2026-03-24',
        player='Stephen Curry',
        prop_type='threes',
        actual=6.0,
    )

    # Print accuracy table for recent games:
    tracker.print_summary(prop_type='threes', window=30)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[3] / "data" / "model_performance"


class ModelPerformanceTracker:
    """
    Append-only JSONL log of individual base-model predictions with
    per-model accuracy reporting.
    """

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self.log_dir = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._pred_log = self.log_dir / "predictions.jsonl"
        self._actual_log = self.log_dir / "actuals.jsonl"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_predictions(
        self,
        date: str,
        player: str,
        prop_type: str,
        line: float,
        predictions: Dict[str, float],
        ensemble_pred: Optional[float] = None,
        american_odds: int = -110,
    ) -> None:
        """Append a prediction record to the JSONL log."""
        record = {
            'ts': datetime.utcnow().isoformat(),
            'date': date,
            'player': player,
            'prop_type': prop_type.lower(),
            'line': line,
            'predictions': {k: round(float(v), 4) for k, v in predictions.items()},
            'ensemble_pred': round(float(ensemble_pred), 4) if ensemble_pred is not None else None,
            'american_odds': american_odds,
        }
        self._append(self._pred_log, record)

    def log_actual(
        self,
        date: str,
        player: str,
        prop_type: str,
        actual: float,
    ) -> None:
        """Append a settled result so prediction records can be matched later."""
        record = {
            'ts': datetime.utcnow().isoformat(),
            'date': date,
            'player': player,
            'prop_type': prop_type.lower(),
            'actual': float(actual),
        }
        self._append(self._actual_log, record)

    # ------------------------------------------------------------------
    # Accuracy metrics
    # ------------------------------------------------------------------

    def get_accuracy(
        self,
        prop_type: str,
        window: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-model accuracy metrics over the last *window* settled games.

        Returns
        -------
        dict
            {model_name: {'n': int, 'mae': float, 'rmse': float, 'bias': float}}
        """
        preds = self._load_jsonl(self._pred_log)
        actuals = self._load_jsonl(self._actual_log)

        # Build actual lookup: (date, player, prop_type) → actual
        actual_map: Dict[tuple, float] = {}
        for rec in actuals:
            key = (rec.get('date'), rec.get('player'), rec.get('prop_type'))
            actual_map[key] = rec.get('actual', 0.0)

        # Filter predictions for this prop type and join with actuals
        matched: List[Dict] = []
        for rec in preds:
            if rec.get('prop_type', '').lower() != prop_type.lower():
                continue
            key = (rec.get('date'), rec.get('player'), prop_type.lower())
            if key in actual_map:
                matched.append({**rec, 'actual': actual_map[key]})

        if not matched:
            return {}

        matched = matched[-window:]

        # Aggregate per-model errors
        errors_by_model: Dict[str, List[float]] = defaultdict(list)
        for rec in matched:
            actual = rec['actual']
            for model_name, pred_val in rec.get('predictions', {}).items():
                errors_by_model[model_name].append(pred_val - actual)
            # Track ensemble separately
            if rec.get('ensemble_pred') is not None:
                errors_by_model['ensemble'].append(rec['ensemble_pred'] - actual)

        results: Dict[str, Dict[str, float]] = {}
        for model_name, errs in errors_by_model.items():
            arr = np.array(errs)
            results[model_name] = {
                'n': len(arr),
                'mae': float(np.mean(np.abs(arr))),
                'rmse': float(np.sqrt(np.mean(arr ** 2))),
                'bias': float(np.mean(arr)),
            }

        return results

    def print_summary(self, prop_type: str, window: int = 50) -> None:
        """Print a human-readable accuracy table."""
        stats = self.get_accuracy(prop_type, window)
        if not stats:
            print(f"  No settled data for prop_type={prop_type}")
            return

        print(f"\n  Model performance — {prop_type} (last {window} games)")
        print(f"  {'Model':<22} {'N':>4}  {'MAE':>6}  {'RMSE':>6}  {'Bias':>7}")
        print("  " + "-" * 55)
        for name in sorted(stats, key=lambda n: stats[n].get('mae', 99)):
            s = stats[name]
            print(
                f"  {name:<22} {s['n']:>4}  {s['mae']:>6.3f}  "
                f"{s['rmse']:>6.3f}  {s['bias']:>+7.3f}"
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        try:
            with open(path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as exc:
            logger.warning("ModelPerformanceTracker: failed to write log: %s", exc)

    @staticmethod
    def _load_jsonl(path: Path) -> List[dict]:
        if not path.exists():
            return []
        records = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.warning("ModelPerformanceTracker: failed to read log: %s", exc)
        return records
