"""
Central registry tracking all agents.

Stores agent metadata, schedules, and status in PostgreSQL (or in-memory fallback).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from .connections import get_postgres_connection

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry of all agents with status tracking."""

    # In-memory registry (class-level, always available)
    _agents: dict = {}

    def __init__(self, pg_conn=None):
        self._pg_conn = pg_conn
        self._use_postgres = pg_conn is not None

        if not self._use_postgres:
            self._pg_conn = get_postgres_connection()
            self._use_postgres = self._pg_conn is not None

    def register(
        self,
        agent_name: str,
        agent_class: str,
        schedule: str = None,
        enabled: bool = True,
    ):
        """Register an agent."""
        AgentRegistry._agents[agent_name] = {
            'agent_name': agent_name,
            'agent_class': agent_class,
            'schedule': schedule,
            'enabled': enabled,
            'status': 'idle',
            'last_run_at': None,
            'last_result': None,
        }

        if self._use_postgres:
            try:
                cur = self._pg_conn.cursor()
                cur.execute("""
                    INSERT INTO agent_registry (agent_name, agent_class, schedule, enabled)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (agent_name) DO UPDATE SET
                        agent_class = EXCLUDED.agent_class,
                        schedule = EXCLUDED.schedule,
                        enabled = EXCLUDED.enabled
                """, (agent_name, agent_class, schedule, enabled))
                cur.close()
            except Exception as e:
                logger.warning(f"Failed to persist agent registration to PostgreSQL: {e}")

        logger.info(f"Agent registered: {agent_name} ({agent_class})")

    def get_agent(self, name: str) -> dict | None:
        """Get agent info by name."""
        # Try in-memory first
        if name in AgentRegistry._agents:
            return AgentRegistry._agents[name]

        # Try PostgreSQL
        if self._use_postgres:
            try:
                cur = self._pg_conn.cursor()
                cur.execute(
                    "SELECT agent_name, agent_class, schedule, enabled, status, last_run_at "
                    "FROM agent_registry WHERE agent_name = %s",
                    (name,)
                )
                row = cur.fetchone()
                cur.close()
                if row:
                    return {
                        'agent_name': row[0],
                        'agent_class': row[1],
                        'schedule': row[2],
                        'enabled': row[3],
                        'status': row[4],
                        'last_run_at': row[5],
                    }
            except Exception as e:
                logger.warning(f"Failed to query agent registry: {e}")

        return None

    def get_all_statuses(self) -> dict:
        """Get status of all registered agents."""
        statuses = {}
        for name, info in AgentRegistry._agents.items():
            statuses[name] = {
                'status': info.get('status', 'unknown'),
                'enabled': info.get('enabled', True),
                'schedule': info.get('schedule'),
                'last_run_at': info.get('last_run_at'),
            }
        return statuses

    def update_status(self, name: str, status: str, result: dict = None):
        """Update agent status after a run."""
        now = datetime.now(timezone.utc).isoformat()

        if name in AgentRegistry._agents:
            AgentRegistry._agents[name]['status'] = status
            AgentRegistry._agents[name]['last_run_at'] = now
            AgentRegistry._agents[name]['last_result'] = result

        if self._use_postgres:
            try:
                import json
                cur = self._pg_conn.cursor()
                cur.execute("""
                    UPDATE agent_registry
                    SET status = %s, last_run_at = %s, last_result = %s
                    WHERE agent_name = %s
                """, (status, now, json.dumps(result) if result else None, name))
                cur.close()
            except Exception as e:
                logger.warning(f"Failed to update agent status in PostgreSQL: {e}")
