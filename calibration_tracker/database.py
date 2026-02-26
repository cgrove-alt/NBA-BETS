"""
Calibration Database - PostgreSQL primary, SQLite fallback.

Tables:
- predictions: Every prediction made with full context
- outcomes: Actual results matched to predictions
- calibration_adjustments: Current calibration adjustments
- daily_reports: Historical daily calibration reports
- weekly_reports: Weekly performance summaries

Production uses PostgreSQL (via DATABASE_URL).
Local dev/tests fall back to SQLite.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from agents.core.connections import get_postgres_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationDatabase:
    """
    Calibration database with PostgreSQL-primary, SQLite-fallback pattern.
    """

    def __init__(self, db_path: str = "data/calibration.db", pg_conn=None):
        self._pg_conn = pg_conn
        self._use_postgres = pg_conn is not None

        if not self._use_postgres:
            self._pg_conn = get_postgres_connection()
            self._use_postgres = self._pg_conn is not None

        if not self._use_postgres:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
            logger.info(f"CalibrationDatabase using SQLite: {self.db_path}")
        else:
            self.db_path = None
            logger.info("CalibrationDatabase using PostgreSQL")

    @contextmanager
    def _get_sqlite_conn(self):
        """Context manager for SQLite connections."""
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
        """Initialize SQLite schema (PG uses migration 003)."""
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    game_id INTEGER,
                    player_id INTEGER NOT NULL,
                    player_name TEXT NOT NULL,
                    team TEXT NOT NULL,
                    opponent TEXT NOT NULL,
                    position TEXT,
                    prop_type TEXT NOT NULL,
                    predicted_value REAL NOT NULL,
                    prop_line REAL NOT NULL,
                    predicted_over_prob REAL,
                    confidence REAL,
                    edge REAL,
                    minutes_predicted REAL,
                    minutes_p10 REAL,
                    minutes_p90 REAL,
                    minutes_uncertainty TEXT,
                    is_home INTEGER,
                    spread REAL,
                    total REAL,
                    is_favorite INTEGER,
                    is_back_to_back INTEGER,
                    days_rest INTEGER,
                    season_avg REAL,
                    recent_avg REAL,
                    vs_opponent_avg REAL,
                    model_version TEXT,
                    features_hash TEXT,
                    status TEXT DEFAULT 'pending',
                    UNIQUE(player_id, game_date, prop_type)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    actual_value REAL NOT NULL,
                    actual_minutes REAL,
                    result TEXT NOT NULL,
                    hit INTEGER NOT NULL,
                    error REAL,
                    closing_line REAL,
                    clv REAL,
                    game_score_diff INTEGER,
                    player_started INTEGER,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calibration_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    dimension TEXT NOT NULL,
                    dimension_value TEXT NOT NULL,
                    bias REAL NOT NULL,
                    adjustment REAL NOT NULL,
                    confidence_multiplier REAL DEFAULT 1.0,
                    sample_size INTEGER NOT NULL,
                    hit_rate REAL,
                    avg_error REAL,
                    std_error REAL,
                    is_active INTEGER DEFAULT 1,
                    UNIQUE(dimension, dimension_value, valid_from)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date TEXT NOT NULL UNIQUE,
                    generated_at TEXT NOT NULL,
                    total_predictions INTEGER,
                    matched_predictions INTEGER,
                    overall_hit_rate REAL,
                    overall_clv REAL,
                    report_json TEXT NOT NULL,
                    status TEXT DEFAULT 'complete'
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weekly_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    week_ending TEXT NOT NULL UNIQUE,
                    generated_at TEXT NOT NULL,
                    total_predictions INTEGER,
                    matched_predictions INTEGER,
                    overall_hit_rate REAL,
                    overall_clv REAL,
                    overall_roi REAL,
                    ece REAL,
                    report_json TEXT NOT NULL,
                    status TEXT DEFAULT 'complete'
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_game_date ON predictions(game_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions(player_id, game_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_prop_type ON predictions(prop_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_prediction ON outcomes(prediction_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_adjustments_active ON calibration_adjustments(is_active, dimension)")

            logger.info(f"Calibration database initialized at {self.db_path}")

    # ========== PREDICTION OPERATIONS ==========

    def insert_prediction(self, prediction: dict) -> int:
        if self._use_postgres:
            return self._insert_prediction_pg(prediction)
        return self._insert_prediction_sqlite(prediction)

    def _insert_prediction_pg(self, prediction: dict) -> int:
        cur = self._pg_conn.cursor()
        cur.execute("""
            INSERT INTO calibration_predictions (
                timestamp, game_date, game_id,
                player_id, player_name, team, opponent, position,
                prop_type, predicted_value, prop_line, predicted_over_prob,
                confidence, edge,
                minutes_predicted, minutes_p10, minutes_p90, minutes_uncertainty,
                is_home, spread, total, is_favorite, is_back_to_back, days_rest,
                season_avg, recent_avg, vs_opponent_avg,
                model_version, features_hash, status
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (player_id, game_date, prop_type) DO UPDATE SET
                predicted_value = EXCLUDED.predicted_value,
                prop_line = EXCLUDED.prop_line,
                confidence = EXCLUDED.confidence,
                edge = EXCLUDED.edge,
                status = EXCLUDED.status
            RETURNING id
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
        row = cur.fetchone()
        cur.close()
        return row[0] if row else 0

    def _insert_prediction_sqlite(self, prediction: dict) -> int:
        with self._get_sqlite_conn() as conn:
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
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
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

    def get_prediction(self, prediction_id: int) -> dict | None:
        if self._use_postgres:
            return self._get_prediction_pg(prediction_id)
        return self._get_prediction_sqlite(prediction_id)

    def _get_prediction_pg(self, prediction_id: int) -> dict | None:
        cur = self._pg_conn.cursor()
        cur.execute("SELECT * FROM calibration_predictions WHERE id = %s", (prediction_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            return None
        cols = [desc[0] for desc in cur.description]
        cur.close()
        return dict(zip(cols, row, strict=False))

    def _get_prediction_sqlite(self, prediction_id: int) -> dict | None:
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_pending_predictions(self, game_date: str) -> list[dict]:
        if self._use_postgres:
            return self._get_pending_predictions_pg(game_date)
        return self._get_pending_predictions_sqlite(game_date)

    def _get_pending_predictions_pg(self, game_date: str) -> list[dict]:
        cur = self._pg_conn.cursor()
        cur.execute("SELECT * FROM calibration_predictions WHERE game_date = %s AND status = 'pending'", (game_date,))
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        return [dict(zip(cols, row, strict=False)) for row in rows]

    def _get_pending_predictions_sqlite(self, game_date: str) -> list[dict]:
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE game_date = ? AND status = 'pending'", (game_date,))
            return [dict(row) for row in cursor.fetchall()]

    def update_prediction_status(self, prediction_id: int, status: str):
        if self._use_postgres:
            cur = self._pg_conn.cursor()
            cur.execute("UPDATE calibration_predictions SET status = %s WHERE id = %s", (status, prediction_id))
            cur.close()
        else:
            with self._get_sqlite_conn() as conn:
                conn.execute("UPDATE predictions SET status = ? WHERE id = ?", (status, prediction_id))

    # ========== OUTCOME OPERATIONS ==========

    def insert_outcome(self, outcome: dict) -> int:
        if self._use_postgres:
            return self._insert_outcome_pg(outcome)
        return self._insert_outcome_sqlite(outcome)

    def _insert_outcome_pg(self, outcome: dict) -> int:
        cur = self._pg_conn.cursor()
        cur.execute("""
            INSERT INTO calibration_outcomes (
                prediction_id, recorded_at,
                actual_value, actual_minutes,
                result, hit, error,
                closing_line, clv,
                game_score_diff, player_started
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
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
        row = cur.fetchone()
        # Update prediction status
        cur.execute("UPDATE calibration_predictions SET status = 'matched' WHERE id = %s", (outcome['prediction_id'],))
        cur.close()
        return row[0] if row else 0

    def _insert_outcome_sqlite(self, outcome: dict) -> int:
        with self._get_sqlite_conn() as conn:
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
            cursor.execute("UPDATE predictions SET status = 'matched' WHERE id = ?", (outcome['prediction_id'],))
            return cursor.lastrowid

    def get_outcome(self, prediction_id: int) -> dict | None:
        if self._use_postgres:
            cur = self._pg_conn.cursor()
            cur.execute("SELECT * FROM calibration_outcomes WHERE prediction_id = %s", (prediction_id,))
            row = cur.fetchone()
            if not row:
                cur.close()
                return None
            cols = [desc[0] for desc in cur.description]
            cur.close()
            return dict(zip(cols, row, strict=False))
        else:
            with self._get_sqlite_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM outcomes WHERE prediction_id = ?", (prediction_id,))
                row = cursor.fetchone()
                return dict(row) if row else None

    # ========== ANALYSIS QUERIES ==========

    def get_predictions_with_outcomes(self, start_date=None, end_date=None, prop_type=None, min_confidence=None) -> list[dict]:
        if self._use_postgres:
            return self._get_pwo_pg(start_date, end_date, prop_type, min_confidence)
        return self._get_pwo_sqlite(start_date, end_date, prop_type, min_confidence)

    def _get_pwo_pg(self, start_date, end_date, prop_type, min_confidence) -> list[dict]:
        cur = self._pg_conn.cursor()
        query = """
            SELECT p.*, o.actual_value, o.actual_minutes, o.result, o.hit,
                   o.error, o.closing_line, o.clv, o.game_score_diff, o.player_started
            FROM calibration_predictions p
            JOIN calibration_outcomes o ON p.id = o.prediction_id
            WHERE 1=1
        """
        params = []
        if start_date:
            query += " AND p.game_date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND p.game_date <= %s"
            params.append(end_date)
        if prop_type:
            query += " AND p.prop_type = %s"
            params.append(prop_type)
        if min_confidence:
            query += " AND p.confidence >= %s"
            params.append(min_confidence)
        query += " ORDER BY p.game_date DESC, p.timestamp DESC"
        cur.execute(query, params)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        return [dict(zip(cols, row, strict=False)) for row in rows]

    def _get_pwo_sqlite(self, start_date, end_date, prop_type, min_confidence) -> list[dict]:
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            query = """
                SELECT p.*, o.actual_value, o.actual_minutes, o.result, o.hit,
                       o.error, o.closing_line, o.clv, o.game_score_diff, o.player_started
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

    def get_summary_stats(self, start_date=None, end_date=None) -> dict:
        if self._use_postgres:
            return self._get_summary_pg(start_date, end_date)
        return self._get_summary_sqlite(start_date, end_date)

    def _get_summary_pg(self, start_date, end_date) -> dict:
        cur = self._pg_conn.cursor()
        query = """
            SELECT COUNT(*) as total, SUM(o.hit::int) as hits, AVG(o.hit::int) as hit_rate,
                   AVG(o.error) as avg_error, AVG(o.clv) as avg_clv, AVG(ABS(o.error)) as mae
            FROM calibration_predictions p
            JOIN calibration_outcomes o ON p.id = o.prediction_id
            WHERE 1=1
        """
        params = []
        if start_date:
            query += " AND p.game_date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND p.game_date <= %s"
            params.append(end_date)
        cur.execute(query, params)
        row = cur.fetchone()
        cols = [desc[0] for desc in cur.description]
        cur.close()
        return dict(zip(cols, row, strict=False)) if row else {}

    def _get_summary_sqlite(self, start_date, end_date) -> dict:
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            query = """
                SELECT COUNT(*) as total, SUM(o.hit) as hits, AVG(o.hit) as hit_rate,
                       AVG(o.error) as avg_error, AVG(o.clv) as avg_clv, AVG(ABS(o.error)) as mae
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
        if self._use_postgres:
            return self._insert_adjustment_pg(adjustment)
        return self._insert_adjustment_sqlite(adjustment)

    def _insert_adjustment_pg(self, adjustment: dict) -> int:
        cur = self._pg_conn.cursor()
        now = datetime.now().isoformat()
        cur.execute("""
            UPDATE calibration_adjustments SET is_active = FALSE, valid_until = %s
            WHERE dimension = %s AND dimension_value = %s AND is_active = TRUE
        """, (now, adjustment['dimension'], adjustment['dimension_value']))
        cur.execute("""
            INSERT INTO calibration_adjustments (
                created_at, valid_from, dimension, dimension_value,
                bias, adjustment, confidence_multiplier,
                sample_size, hit_rate, avg_error, std_error, is_active
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
            RETURNING id
        """, (
            now, now,
            adjustment['dimension'], adjustment['dimension_value'],
            adjustment['bias'], adjustment['adjustment'],
            adjustment.get('confidence_multiplier', 1.0),
            adjustment['sample_size'],
            adjustment.get('hit_rate'), adjustment.get('avg_error'), adjustment.get('std_error'),
        ))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else 0

    def _insert_adjustment_sqlite(self, adjustment: dict) -> int:
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                UPDATE calibration_adjustments SET is_active = 0, valid_until = ?
                WHERE dimension = ? AND dimension_value = ? AND is_active = 1
            """, (now, adjustment['dimension'], adjustment['dimension_value']))
            cursor.execute("""
                INSERT INTO calibration_adjustments (
                    created_at, valid_from, dimension, dimension_value,
                    bias, adjustment, confidence_multiplier,
                    sample_size, hit_rate, avg_error, std_error, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                now, now,
                adjustment['dimension'], adjustment['dimension_value'],
                adjustment['bias'], adjustment['adjustment'],
                adjustment.get('confidence_multiplier', 1.0),
                adjustment['sample_size'],
                adjustment.get('hit_rate'), adjustment.get('avg_error'), adjustment.get('std_error'),
            ))
            return cursor.lastrowid

    def get_active_adjustments(self) -> list[dict]:
        if self._use_postgres:
            cur = self._pg_conn.cursor()
            cur.execute("SELECT * FROM calibration_adjustments WHERE is_active = TRUE")
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            cur.close()
            return [dict(zip(cols, row, strict=False)) for row in rows]
        else:
            with self._get_sqlite_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM calibration_adjustments WHERE is_active = 1")
                return [dict(row) for row in cursor.fetchall()]

    def get_adjustment(self, dimension: str, dimension_value: str) -> dict | None:
        if self._use_postgres:
            cur = self._pg_conn.cursor()
            cur.execute("""
                SELECT * FROM calibration_adjustments
                WHERE dimension = %s AND dimension_value = %s AND is_active = TRUE
            """, (dimension, dimension_value))
            row = cur.fetchone()
            if not row:
                cur.close()
                return None
            cols = [desc[0] for desc in cur.description]
            cur.close()
            return dict(zip(cols, row, strict=False))
        else:
            with self._get_sqlite_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM calibration_adjustments
                    WHERE dimension = ? AND dimension_value = ? AND is_active = 1
                """, (dimension, dimension_value))
                row = cursor.fetchone()
                return dict(row) if row else None

    # ========== DAILY REPORT OPERATIONS ==========

    def insert_daily_report(self, report_date: str, report: dict) -> int:
        if self._use_postgres:
            return self._insert_daily_report_pg(report_date, report)
        return self._insert_daily_report_sqlite(report_date, report)

    def _insert_daily_report_pg(self, report_date: str, report: dict) -> int:
        cur = self._pg_conn.cursor()
        cur.execute("""
            INSERT INTO calibration_daily_reports (
                report_date, generated_at, total_predictions, matched_predictions,
                overall_hit_rate, overall_clv, report_json, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (report_date) DO UPDATE SET
                generated_at = EXCLUDED.generated_at,
                total_predictions = EXCLUDED.total_predictions,
                matched_predictions = EXCLUDED.matched_predictions,
                overall_hit_rate = EXCLUDED.overall_hit_rate,
                overall_clv = EXCLUDED.overall_clv,
                report_json = EXCLUDED.report_json
            RETURNING id
        """, (
            report_date, datetime.now().isoformat(),
            report.get('total_predictions', 0), report.get('matched_predictions', 0),
            report.get('overall_hit_rate'), report.get('overall_clv'),
            json.dumps(report), 'complete',
        ))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else 0

    def _insert_daily_report_sqlite(self, report_date: str, report: dict) -> int:
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO daily_reports (
                    report_date, generated_at, total_predictions, matched_predictions,
                    overall_hit_rate, overall_clv, report_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_date, datetime.now().isoformat(),
                report.get('total_predictions', 0), report.get('matched_predictions', 0),
                report.get('overall_hit_rate'), report.get('overall_clv'),
                json.dumps(report), 'complete',
            ))
            return cursor.lastrowid

    def get_daily_report(self, report_date: str) -> dict | None:
        if self._use_postgres:
            cur = self._pg_conn.cursor()
            cur.execute("SELECT * FROM calibration_daily_reports WHERE report_date = %s", (report_date,))
            row = cur.fetchone()
            if not row:
                cur.close()
                return None
            cols = [desc[0] for desc in cur.description]
            result = dict(zip(cols, row, strict=False))
            cur.close()
            rj = result.get('report_json')
            result['report'] = json.loads(rj) if isinstance(rj, str) else rj
            return result
        else:
            with self._get_sqlite_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM daily_reports WHERE report_date = ?", (report_date,))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    result['report'] = json.loads(result['report_json'])
                    return result
                return None

    def get_recent_reports(self, limit: int = 30) -> list[dict]:
        if self._use_postgres:
            cur = self._pg_conn.cursor()
            cur.execute("SELECT * FROM calibration_daily_reports ORDER BY report_date DESC LIMIT %s", (limit,))
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            cur.close()
            return [dict(zip(cols, row, strict=False)) for row in rows]
        else:
            with self._get_sqlite_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM daily_reports ORDER BY report_date DESC LIMIT ?", (limit,))
                return [dict(row) for row in cursor.fetchall()]

    # ========== WEEKLY REPORT OPERATIONS ==========

    def insert_weekly_report(self, week_ending: str, report: dict) -> int:
        if self._use_postgres:
            return self._insert_weekly_report_pg(week_ending, report)
        return self._insert_weekly_report_sqlite(week_ending, report)

    def _insert_weekly_report_pg(self, week_ending: str, report: dict) -> int:
        cur = self._pg_conn.cursor()
        cur.execute("""
            INSERT INTO calibration_weekly_reports (
                week_ending, generated_at, total_predictions, matched_predictions,
                overall_hit_rate, overall_clv, overall_roi, ece, report_json, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (week_ending) DO UPDATE SET
                generated_at = EXCLUDED.generated_at,
                total_predictions = EXCLUDED.total_predictions,
                matched_predictions = EXCLUDED.matched_predictions,
                overall_hit_rate = EXCLUDED.overall_hit_rate,
                overall_clv = EXCLUDED.overall_clv,
                overall_roi = EXCLUDED.overall_roi,
                ece = EXCLUDED.ece,
                report_json = EXCLUDED.report_json
            RETURNING id
        """, (
            week_ending, datetime.now().isoformat(),
            report.get('total_predictions', 0), report.get('matched_predictions', 0),
            report.get('overall_hit_rate'), report.get('overall_clv'),
            report.get('overall_roi'), report.get('ece'),
            json.dumps(report), 'complete',
        ))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else 0

    def _insert_weekly_report_sqlite(self, week_ending: str, report: dict) -> int:
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO weekly_reports (
                    week_ending, generated_at, total_predictions, matched_predictions,
                    overall_hit_rate, overall_clv, overall_roi, ece, report_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                week_ending, datetime.now().isoformat(),
                report.get('total_predictions', 0), report.get('matched_predictions', 0),
                report.get('overall_hit_rate'), report.get('overall_clv'),
                report.get('overall_roi'), report.get('ece'),
                json.dumps(report), 'complete',
            ))
            return cursor.lastrowid

    def get_weekly_report(self, week_ending: str) -> dict | None:
        if self._use_postgres:
            cur = self._pg_conn.cursor()
            cur.execute("SELECT * FROM calibration_weekly_reports WHERE week_ending = %s", (week_ending,))
            row = cur.fetchone()
            if not row:
                cur.close()
                return None
            cols = [desc[0] for desc in cur.description]
            result = dict(zip(cols, row, strict=False))
            cur.close()
            rj = result.get('report_json')
            result['report'] = json.loads(rj) if isinstance(rj, str) else rj
            return result
        else:
            with self._get_sqlite_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM weekly_reports WHERE week_ending = ?", (week_ending,))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    result['report'] = json.loads(result['report_json'])
                    return result
                return None

    def get_recent_weekly_reports(self, limit: int = 12) -> list[dict]:
        if self._use_postgres:
            cur = self._pg_conn.cursor()
            cur.execute("SELECT * FROM calibration_weekly_reports ORDER BY week_ending DESC LIMIT %s", (limit,))
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            cur.close()
            return [dict(zip(cols, row, strict=False)) for row in rows]
        else:
            with self._get_sqlite_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM weekly_reports ORDER BY week_ending DESC LIMIT ?", (limit,))
                return [dict(row) for row in cursor.fetchall()]
