"""
Calibration Database - SQLite Schema and Operations

Tables:
- predictions: Every prediction made with full context
- outcomes: Actual results matched to predictions
- calibration_adjustments: Current calibration adjustments
- daily_reports: Historical daily calibration reports
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationDatabase:
    """
    SQLite database for prediction tracking and calibration.
    """

    def __init__(self, db_path: str = "data/calibration.db"):
        """
        Initialize the calibration database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # ========== PREDICTIONS TABLE ==========
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    game_id INTEGER,

                    -- Player info
                    player_id INTEGER NOT NULL,
                    player_name TEXT NOT NULL,
                    team TEXT NOT NULL,
                    opponent TEXT NOT NULL,
                    position TEXT,

                    -- Prediction details
                    prop_type TEXT NOT NULL,
                    predicted_value REAL NOT NULL,
                    prop_line REAL NOT NULL,
                    predicted_over_prob REAL,
                    confidence REAL,
                    edge REAL,

                    -- Minutes prediction
                    minutes_predicted REAL,
                    minutes_p10 REAL,
                    minutes_p90 REAL,
                    minutes_uncertainty TEXT,

                    -- Game context
                    is_home INTEGER,
                    spread REAL,
                    total REAL,
                    is_favorite INTEGER,
                    is_back_to_back INTEGER,
                    days_rest INTEGER,

                    -- Player context
                    season_avg REAL,
                    recent_avg REAL,
                    vs_opponent_avg REAL,

                    -- Model info
                    model_version TEXT,
                    features_hash TEXT,

                    -- Status
                    status TEXT DEFAULT 'pending',  -- pending, matched, expired

                    -- Indexes
                    UNIQUE(player_id, game_date, prop_type)
                )
            """)

            # ========== OUTCOMES TABLE ==========
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,

                    -- Actual results
                    actual_value REAL NOT NULL,
                    actual_minutes REAL,

                    -- Result classification
                    result TEXT NOT NULL,  -- over, under, push
                    hit INTEGER NOT NULL,  -- 1 if prediction correct, 0 otherwise
                    error REAL,  -- predicted - actual

                    -- Line movement
                    closing_line REAL,
                    clv REAL,  -- Closing Line Value (positive = beat the close)

                    -- Additional context
                    game_score_diff INTEGER,  -- Final margin (for blowout detection)
                    player_started INTEGER,

                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                )
            """)

            # ========== CALIBRATION ADJUSTMENTS TABLE ==========
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calibration_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,

                    -- Dimension
                    dimension TEXT NOT NULL,  -- overall, prop_type, position, etc.
                    dimension_value TEXT NOT NULL,  -- points, guards, etc.

                    -- Adjustment
                    bias REAL NOT NULL,
                    adjustment REAL NOT NULL,
                    confidence_multiplier REAL DEFAULT 1.0,

                    -- Evidence
                    sample_size INTEGER NOT NULL,
                    hit_rate REAL,
                    avg_error REAL,
                    std_error REAL,

                    -- Status
                    is_active INTEGER DEFAULT 1,

                    UNIQUE(dimension, dimension_value, valid_from)
                )
            """)

            # ========== DAILY REPORTS TABLE ==========
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date TEXT NOT NULL UNIQUE,
                    generated_at TEXT NOT NULL,

                    -- Summary stats
                    total_predictions INTEGER,
                    matched_predictions INTEGER,
                    overall_hit_rate REAL,
                    overall_clv REAL,

                    -- Detailed report (JSON)
                    report_json TEXT NOT NULL,

                    -- Status
                    status TEXT DEFAULT 'complete'
                )
            """)

            # ========== INDEXES ==========
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_game_date
                ON predictions(game_date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_player
                ON predictions(player_id, game_date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_prop_type
                ON predictions(prop_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_status
                ON predictions(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_prediction
                ON outcomes(prediction_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_adjustments_active
                ON calibration_adjustments(is_active, dimension)
            """)

            logger.info(f"Calibration database initialized at {self.db_path}")

    # ========== PREDICTION OPERATIONS ==========

    def insert_prediction(self, prediction: dict) -> int:
        """
        Insert a new prediction record.

        Args:
            prediction: Dictionary with prediction data

        Returns:
            Inserted prediction ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO predictions (
                    timestamp, game_date, game_id,
                    player_id, player_name, team, opponent, position,
                    prop_type, predicted_value, prop_line, predicted_over_prob,
                    confidence, edge,
                    minutes_predicted, minutes_p10, minutes_p90, minutes_uncertainty,
                    is_home, spread, total, is_favorite, is_back_to_back, days_rest,
                    season_avg, recent_avg, vs_opponent_avg,
                    model_version, features_hash, status
                ) VALUES (
                    ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
                )
            """, (
                prediction.get('timestamp', datetime.now().isoformat()),
                prediction['game_date'],
                prediction.get('game_id'),
                prediction['player_id'],
                prediction['player_name'],
                prediction['team'],
                prediction['opponent'],
                prediction.get('position'),
                prediction['prop_type'],
                prediction['predicted_value'],
                prediction['prop_line'],
                prediction.get('predicted_over_prob'),
                prediction.get('confidence'),
                prediction.get('edge'),
                prediction.get('minutes_predicted'),
                prediction.get('minutes_p10'),
                prediction.get('minutes_p90'),
                prediction.get('minutes_uncertainty'),
                prediction.get('is_home'),
                prediction.get('spread'),
                prediction.get('total'),
                prediction.get('is_favorite'),
                prediction.get('is_back_to_back'),
                prediction.get('days_rest'),
                prediction.get('season_avg'),
                prediction.get('recent_avg'),
                prediction.get('vs_opponent_avg'),
                prediction.get('model_version'),
                prediction.get('features_hash'),
                prediction.get('status', 'pending'),
            ))

            return cursor.lastrowid

    def get_prediction(self, prediction_id: int) -> Optional[dict]:
        """Get a prediction by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_pending_predictions(self, game_date: str) -> list[dict]:
        """Get all pending predictions for a game date."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM predictions
                WHERE game_date = ? AND status = 'pending'
            """, (game_date,))
            return [dict(row) for row in cursor.fetchall()]

    def update_prediction_status(self, prediction_id: int, status: str):
        """Update prediction status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE predictions SET status = ? WHERE id = ?
            """, (status, prediction_id))

    # ========== OUTCOME OPERATIONS ==========

    def insert_outcome(self, outcome: dict) -> int:
        """
        Insert an outcome record.

        Args:
            outcome: Dictionary with outcome data

        Returns:
            Inserted outcome ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO outcomes (
                    prediction_id, recorded_at,
                    actual_value, actual_minutes,
                    result, hit, error,
                    closing_line, clv,
                    game_score_diff, player_started
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome['prediction_id'],
                outcome.get('recorded_at', datetime.now().isoformat()),
                outcome['actual_value'],
                outcome.get('actual_minutes'),
                outcome['result'],
                outcome['hit'],
                outcome.get('error'),
                outcome.get('closing_line'),
                outcome.get('clv'),
                outcome.get('game_score_diff'),
                outcome.get('player_started'),
            ))

            # Update prediction status
            cursor.execute("""
                UPDATE predictions SET status = 'matched' WHERE id = ?
            """, (outcome['prediction_id'],))

            return cursor.lastrowid

    def get_outcome(self, prediction_id: int) -> Optional[dict]:
        """Get outcome for a prediction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM outcomes WHERE prediction_id = ?
            """, (prediction_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ========== ANALYSIS QUERIES ==========

    def get_predictions_with_outcomes(
        self,
        start_date: str = None,
        end_date: str = None,
        prop_type: str = None,
        min_confidence: float = None,
    ) -> list[dict]:
        """
        Get predictions joined with their outcomes.

        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            prop_type: Filter by prop type
            min_confidence: Minimum confidence threshold

        Returns:
            List of joined prediction/outcome records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    p.*,
                    o.actual_value,
                    o.actual_minutes,
                    o.result,
                    o.hit,
                    o.error,
                    o.closing_line,
                    o.clv,
                    o.game_score_diff,
                    o.player_started
                FROM predictions p
                JOIN outcomes o ON p.id = o.prediction_id
                WHERE 1=1
            """
            params = []

            if start_date:
                query += " AND p.game_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND p.game_date <= ?"
                params.append(end_date)
            if prop_type:
                query += " AND p.prop_type = ?"
                params.append(prop_type)
            if min_confidence:
                query += " AND p.confidence >= ?"
                params.append(min_confidence)

            query += " ORDER BY p.game_date DESC, p.timestamp DESC"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_summary_stats(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """Get summary statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    COUNT(*) as total,
                    SUM(o.hit) as hits,
                    AVG(o.hit) as hit_rate,
                    AVG(o.error) as avg_error,
                    AVG(o.clv) as avg_clv,
                    AVG(ABS(o.error)) as mae
                FROM predictions p
                JOIN outcomes o ON p.id = o.prediction_id
                WHERE 1=1
            """
            params = []

            if start_date:
                query += " AND p.game_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND p.game_date <= ?"
                params.append(end_date)

            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else {}

    # ========== CALIBRATION ADJUSTMENT OPERATIONS ==========

    def insert_adjustment(self, adjustment: dict) -> int:
        """Insert a calibration adjustment."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Deactivate existing adjustments for this dimension
            cursor.execute("""
                UPDATE calibration_adjustments
                SET is_active = 0, valid_until = ?
                WHERE dimension = ? AND dimension_value = ? AND is_active = 1
            """, (
                datetime.now().isoformat(),
                adjustment['dimension'],
                adjustment['dimension_value'],
            ))

            cursor.execute("""
                INSERT INTO calibration_adjustments (
                    created_at, valid_from, dimension, dimension_value,
                    bias, adjustment, confidence_multiplier,
                    sample_size, hit_rate, avg_error, std_error,
                    is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                adjustment['dimension'],
                adjustment['dimension_value'],
                adjustment['bias'],
                adjustment['adjustment'],
                adjustment.get('confidence_multiplier', 1.0),
                adjustment['sample_size'],
                adjustment.get('hit_rate'),
                adjustment.get('avg_error'),
                adjustment.get('std_error'),
            ))

            return cursor.lastrowid

    def get_active_adjustments(self) -> list[dict]:
        """Get all active calibration adjustments."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM calibration_adjustments WHERE is_active = 1
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_adjustment(self, dimension: str, dimension_value: str) -> Optional[dict]:
        """Get active adjustment for a specific dimension."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM calibration_adjustments
                WHERE dimension = ? AND dimension_value = ? AND is_active = 1
            """, (dimension, dimension_value))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ========== DAILY REPORT OPERATIONS ==========

    def insert_daily_report(self, report_date: str, report: dict) -> int:
        """Insert a daily calibration report."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO daily_reports (
                    report_date, generated_at,
                    total_predictions, matched_predictions,
                    overall_hit_rate, overall_clv,
                    report_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_date,
                datetime.now().isoformat(),
                report.get('total_predictions', 0),
                report.get('matched_predictions', 0),
                report.get('overall_hit_rate'),
                report.get('overall_clv'),
                json.dumps(report),
                'complete',
            ))

            return cursor.lastrowid

    def get_daily_report(self, report_date: str) -> Optional[dict]:
        """Get daily report by date."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM daily_reports WHERE report_date = ?
            """, (report_date,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['report'] = json.loads(result['report_json'])
                return result
            return None

    def get_recent_reports(self, limit: int = 30) -> list[dict]:
        """Get recent daily reports."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM daily_reports
                ORDER BY report_date DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]


if __name__ == "__main__":
    # Test database initialization
    db = CalibrationDatabase()
    print(f"Database initialized at: {db.db_path}")

    # Test insert prediction
    pred_id = db.insert_prediction({
        'game_date': '2024-01-15',
        'player_id': 123,
        'player_name': 'LeBron James',
        'team': 'LAL',
        'opponent': 'BOS',
        'position': 'F',
        'prop_type': 'points',
        'predicted_value': 27.5,
        'prop_line': 26.5,
        'predicted_over_prob': 0.58,
        'confidence': 65.0,
        'edge': 3.2,
        'is_home': 1,
        'spread': -3.5,
        'total': 225.5,
    })
    print(f"Inserted prediction ID: {pred_id}")

    # Test insert outcome
    out_id = db.insert_outcome({
        'prediction_id': pred_id,
        'actual_value': 29,
        'actual_minutes': 35.2,
        'result': 'over',
        'hit': 1,
        'error': -1.5,
        'clv': 0.5,
    })
    print(f"Inserted outcome ID: {out_id}")

    # Test get with outcomes
    results = db.get_predictions_with_outcomes()
    print(f"Predictions with outcomes: {len(results)}")
