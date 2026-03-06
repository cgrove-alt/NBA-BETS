"""
Token budgets, execution limits, and circuit breaker.

Persists to PostgreSQL (production) or SQLite (local/tests).
"""

import os
import json
import sqlite3
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from .connections import get_postgres_connection

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Token budget for a single agent."""
    agent_name: str
    daily_limit: int = 50_000
    used_today: int = 0
    reset_date: str = ''

    def __post_init__(self):
        if not self.reset_date:
            self.reset_date = date.today().isoformat()

    def can_spend(self, tokens: int) -> bool:
        """Check if budget allows spending tokens, with daily auto-reset."""
        self._maybe_reset()
        return (self.used_today + tokens) <= self.daily_limit

    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage."""
        self._maybe_reset()
        self.used_today += (input_tokens + output_tokens)

    def _maybe_reset(self):
        """Reset budget if it's a new day."""
        today = date.today().isoformat()
        if self.reset_date != today:
            self.used_today = 0
            self.reset_date = today


class Guardrails:
    """
    Token budgets, circuit breaker, and run recording.

    Uses PostgreSQL when available, falls back to SQLite.
    """

    def __init__(self, pg_conn=None, sqlite_path: str = None):
        self._pg_conn = pg_conn
        self._use_postgres = pg_conn is not None

        if not self._use_postgres:
            # Try to connect to PostgreSQL
            self._pg_conn = get_postgres_connection()
            self._use_postgres = self._pg_conn is not None

        if not self._use_postgres:
            # Fall back to SQLite
            if sqlite_path is None:
                data_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    'data'
                )
                os.makedirs(data_dir, exist_ok=True)
                sqlite_path = os.path.join(data_dir, 'agent_guardrails.db')
            self._sqlite_path = sqlite_path
            self._init_sqlite()
            logger.info(f"Guardrails using SQLite: {sqlite_path}")
        else:
            self._sqlite_path = None
            logger.info("Guardrails using PostgreSQL")

    def _init_sqlite(self):
        """Initialize SQLite tables."""
        conn = sqlite3.connect(self._sqlite_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_token_budgets (
                agent_name TEXT PRIMARY KEY,
                daily_limit INTEGER DEFAULT 50000,
                used_today INTEGER DEFAULT 0,
                reset_date TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                run_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                success INTEGER,
                tokens_used INTEGER DEFAULT 0,
                execution_seconds REAL,
                messages_sent INTEGER DEFAULT 0,
                errors TEXT,
                payload TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()

    def _get_sqlite_conn(self):
        return sqlite3.connect(self._sqlite_path)

    def get_budget(self, agent_name: str, daily_limit: int = 50_000) -> TokenBudget:
        """Get or create token budget for an agent."""
        today = date.today().isoformat()

        if self._use_postgres:
            return self._get_budget_pg(agent_name, daily_limit, today)
        else:
            return self._get_budget_sqlite(agent_name, daily_limit, today)

    def _get_budget_pg(self, agent_name: str, daily_limit: int, today: str) -> TokenBudget:
        cur = self._pg_conn.cursor()
        try:
            cur.execute(
                "SELECT agent_name, daily_limit, used_today, reset_date FROM agent_token_budgets WHERE agent_name = %s",
                (agent_name,)
            )
            row = cur.fetchone()

            if row is None:
                cur.execute(
                    "INSERT INTO agent_token_budgets (agent_name, daily_limit, used_today, reset_date) VALUES (%s, %s, 0, %s)",
                    (agent_name, daily_limit, today)
                )
                self._pg_conn.commit()
                return TokenBudget(agent_name=agent_name, daily_limit=daily_limit, used_today=0, reset_date=today)

            budget = TokenBudget(
                agent_name=row[0],
                daily_limit=row[1],
                used_today=row[2],
                reset_date=str(row[3]),
            )

            # Auto-reset if new day
            if budget.reset_date != today:
                cur.execute(
                    "UPDATE agent_token_budgets SET used_today = 0, reset_date = %s WHERE agent_name = %s",
                    (today, agent_name)
                )
                self._pg_conn.commit()
                budget.used_today = 0
                budget.reset_date = today

            return budget
        finally:
            cur.close()

    def _get_budget_sqlite(self, agent_name: str, daily_limit: int, today: str) -> TokenBudget:
        conn = self._get_sqlite_conn()
        row = conn.execute(
            "SELECT agent_name, daily_limit, used_today, reset_date FROM agent_token_budgets WHERE agent_name = ?",
            (agent_name,)
        ).fetchone()

        if row is None:
            conn.execute(
                "INSERT INTO agent_token_budgets (agent_name, daily_limit, used_today, reset_date) VALUES (?, ?, 0, ?)",
                (agent_name, daily_limit, today)
            )
            conn.commit()
            conn.close()
            return TokenBudget(agent_name=agent_name, daily_limit=daily_limit, used_today=0, reset_date=today)

        budget = TokenBudget(agent_name=row[0], daily_limit=row[1], used_today=row[2], reset_date=row[3])

        if budget.reset_date != today:
            conn.execute(
                "UPDATE agent_token_budgets SET used_today = 0, reset_date = ? WHERE agent_name = ?",
                (today, agent_name)
            )
            conn.commit()
            budget.used_today = 0
            budget.reset_date = today

        conn.close()
        return budget

    def save_budget(self, budget: TokenBudget):
        """Persist updated budget."""
        if self._use_postgres:
            cur = self._pg_conn.cursor()
            try:
                cur.execute(
                    "UPDATE agent_token_budgets SET used_today = %s, reset_date = %s WHERE agent_name = %s",
                    (budget.used_today, budget.reset_date, budget.agent_name)
                )
                self._pg_conn.commit()
            finally:
                cur.close()
        else:
            conn = self._get_sqlite_conn()
            conn.execute(
                "UPDATE agent_token_budgets SET used_today = ?, reset_date = ? WHERE agent_name = ?",
                (budget.used_today, budget.reset_date, budget.agent_name)
            )
            conn.commit()
            conn.close()

    def check_circuit_breaker(self, agent_name: str, max_failures: int = 3) -> bool:
        """
        Check if circuit breaker should trip.

        Returns True if agent should be in COOLDOWN (too many consecutive failures).
        """
        if self._use_postgres:
            return self._check_cb_pg(agent_name, max_failures)
        else:
            return self._check_cb_sqlite(agent_name, max_failures)

    def _check_cb_pg(self, agent_name: str, max_failures: int) -> bool:
        cur = self._pg_conn.cursor()
        try:
            cur.execute("""
                SELECT success FROM agent_runs
                WHERE agent_name = %s
                ORDER BY started_at DESC
                LIMIT %s
            """, (agent_name, max_failures))
            rows = cur.fetchall()
        finally:
            cur.close()

        if len(rows) < max_failures:
            return False
        return all(row[0] is False or row[0] == 0 for row in rows)

    def _check_cb_sqlite(self, agent_name: str, max_failures: int) -> bool:
        conn = self._get_sqlite_conn()
        rows = conn.execute("""
            SELECT success FROM agent_runs
            WHERE agent_name = ?
            ORDER BY started_at DESC
            LIMIT ?
        """, (agent_name, max_failures)).fetchall()
        conn.close()

        if len(rows) < max_failures:
            return False
        return all(row[0] == 0 for row in rows)

    def record_run(
        self,
        agent_name: str,
        run_id: str,
        started_at: str,
        completed_at: str,
        status: str,
        success: bool,
        tokens_used: int = 0,
        execution_seconds: float = 0.0,
        messages_sent: int = 0,
        errors: list = None,
        payload: dict = None,
    ):
        """Record an agent run."""
        errors_json = json.dumps(errors) if errors else None
        payload_json = json.dumps(payload) if payload else None

        if self._use_postgres:
            cur = self._pg_conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO agent_runs
                        (agent_name, run_id, started_at, completed_at, status, success,
                         tokens_used, execution_seconds, messages_sent, errors, payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    agent_name, run_id, started_at, completed_at, status, success,
                    tokens_used, execution_seconds, messages_sent,
                    errors_json, payload_json,
                ))
                self._pg_conn.commit()
            finally:
                cur.close()
        else:
            conn = self._get_sqlite_conn()
            conn.execute("""
                INSERT INTO agent_runs
                    (agent_name, run_id, started_at, completed_at, status, success,
                     tokens_used, execution_seconds, messages_sent, errors, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_name, run_id, started_at, completed_at, status, int(success),
                tokens_used, execution_seconds, messages_sent,
                errors_json, payload_json,
            ))
            conn.commit()
            conn.close()

    def get_daily_cost_summary(self) -> dict:
        """Get summary of today's token usage across all agents."""
        today = date.today().isoformat()

        if self._use_postgres:
            cur = self._pg_conn.cursor()
            try:
                cur.execute(
                    "SELECT agent_name, daily_limit, used_today FROM agent_token_budgets WHERE reset_date = %s",
                    (today,)
                )
                rows = cur.fetchall()
            finally:
                cur.close()
        else:
            conn = self._get_sqlite_conn()
            rows = conn.execute(
                "SELECT agent_name, daily_limit, used_today FROM agent_token_budgets WHERE reset_date = ?",
                (today,)
            ).fetchall()
            conn.close()

        summary = {}
        for row in rows:
            summary[row[0]] = {
                'daily_limit': row[1],
                'used_today': row[2],
                'remaining': row[1] - row[2],
                'utilization_pct': round(row[2] / row[1] * 100, 1) if row[1] > 0 else 0,
            }
        return summary
