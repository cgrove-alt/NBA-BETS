#!/usr/bin/env python3
"""
Read-only diagnostic for the NBA-BETS agent system.

Inspects existing state (scheduler status, agent registry, guardrails,
databases) WITHOUT running any agents or making LLM calls.
Safe for Railway production.

Usage:
    python scripts/agent_diagnostics.py          # CLI report
    python scripts/agent_diagnostics.py --json    # JSON output

API usage (from backend/api.py):
    from scripts.agent_diagnostics import run_diagnostics
    data = run_diagnostics(return_json=True)
"""

import os
import sys
import json
import sqlite3
import traceback
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

WIDTH = 72
AGENT_NAMES = [
    "pregame", "postgame", "odds_monitor",
    "orchestrator", "watchdog", "briefing",
]


# ============================================================================
# Section collectors — each returns a dict
# ============================================================================

def _collect_environment():
    """Section A — Environment + Dependencies."""
    data = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "project_dir": str(PROJECT_DIR),
        "cwd": os.getcwd(),
        "railway_environment": os.environ.get("RAILWAY_ENVIRONMENT"),
        "railway_static_url": os.environ.get("RAILWAY_STATIC_URL"),
        "is_railway": bool(os.environ.get("RAILWAY_ENVIRONMENT")),
    }

    # load_env
    try:
        import load_env  # noqa: F401
        data["load_env"] = "ok"
    except Exception as e:
        data["load_env"] = f"error: {e}"

    # Secrets (masked)
    db_url = os.environ.get("DATABASE_URL")
    redis_url = os.environ.get("REDIS_URL")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    data["database_url"] = _mask(db_url, 20)
    data["redis_url"] = _mask(redis_url, 20)
    data["gemini_api_key"] = "present" if gemini_key else "ABSENT"

    # APScheduler
    try:
        from apscheduler.triggers.cron import CronTrigger  # noqa: F401
        data["apscheduler"] = "ok"
    except ImportError:
        data["apscheduler"] = "NOT INSTALLED"

    return data


def _collect_scheduler(pg_conn=None):
    """Section B — Scheduler status from Postgres agent_runs, status file as fallback."""
    data = {
        "status_file_exists": False,
        "status_file_stale": False,
        "stale_hours": None,
        "start_time": None,
        "total_runs": 0,
        "total_failures": 0,
        "agents": {},
        "daemon_running": False,
        "daemon_pid": None,
        "daemon_message": "",
        "source": "none",
    }

    # Import schedule metadata (needed for cron_kwargs / descriptions)
    try:
        from agents.core.agent_scheduler import AGENT_SCHEDULES
    except Exception:
        AGENT_SCHEDULES = {}

    # ------------------------------------------------------------------
    # Primary source: Postgres agent_runs table
    # ------------------------------------------------------------------
    if pg_conn is not None:
        try:
            import psycopg2.extras
            cursor = pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Aggregate stats from last 24 hours
            cursor.execute("""
                SELECT
                    agent_name,
                    COUNT(*) as run_count,
                    MAX(started_at) as last_run,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successes
                FROM agent_runs
                WHERE started_at >= NOW() - INTERVAL '24 hours'
                GROUP BY agent_name
            """)
            recent_runs = cursor.fetchall()

            data["total_runs"] = sum(r["run_count"] for r in recent_runs)
            data["total_failures"] = sum(r["failures"] for r in recent_runs)

            # Any runs in last 2 hours => scheduler is alive
            cursor.execute("""
                SELECT COUNT(*) as recent_count
                FROM agent_runs
                WHERE started_at >= NOW() - INTERVAL '2 hours'
            """)
            recent_count = cursor.fetchone()["recent_count"]
            scheduler_running = recent_count > 0

            data["daemon_running"] = scheduler_running
            data["daemon_message"] = (
                f"Scheduler active (detected via Postgres, {recent_count} runs in last 2h)"
                if scheduler_running
                else "No recent activity in Postgres agent_runs"
            )
            data["source"] = "postgres"

            # Build per-agent stats
            for name in AGENT_NAMES:
                agent_data = next((r for r in recent_runs if r["agent_name"] == name), None)
                cron_kwargs, sched_desc = AGENT_SCHEDULES.get(name, ({}, "N/A"))

                if agent_data:
                    last_run_val = agent_data["last_run"]
                    data["agents"][name] = {
                        "last_run": last_run_val.isoformat() if last_run_val else None,
                        "last_status": "completed" if agent_data["successes"] > 0 else "failed",
                        "runs": agent_data["run_count"],
                        "failures": agent_data["failures"],
                        "schedule": sched_desc,
                        "cron_kwargs": cron_kwargs,
                    }
                else:
                    data["agents"][name] = {
                        "last_run": None,
                        "last_status": "",
                        "runs": 0,
                        "failures": 0,
                        "schedule": sched_desc,
                        "cron_kwargs": cron_kwargs,
                    }

            cursor.close()
            return data

        except Exception as e:
            data["postgres_error"] = str(e)
            # Fall through to status file fallback

    # ------------------------------------------------------------------
    # Fallback: local status file (only works on scheduler service)
    # ------------------------------------------------------------------
    try:
        from agents.core.agent_scheduler import get_status, STATUS_FILE
    except Exception as e:
        data["error"] = f"Import error: {e}"
        return data

    status_path = Path(STATUS_FILE) if not isinstance(STATUS_FILE, Path) else STATUS_FILE

    if status_path.exists():
        data["status_file_exists"] = True
        data["source"] = "status_file"

        try:
            mtime = status_path.stat().st_mtime
            age_hours = (datetime.now(timezone.utc).timestamp() - mtime) / 3600
            data["stale_hours"] = round(age_hours, 1)
            if age_hours > 2:
                data["status_file_stale"] = True
        except Exception:
            pass

        try:
            with open(status_path) as f:
                saved = json.load(f)
            data["start_time"] = saved.get("start_time")
            data["total_runs"] = saved.get("total_runs", 0)
            data["total_failures"] = saved.get("total_failures", 0)

            for name in AGENT_NAMES:
                a = saved.get("agents", {}).get(name, {})
                cron_kwargs, sched_desc = AGENT_SCHEDULES.get(name, ({}, "N/A"))
                data["agents"][name] = {
                    "last_run": a.get("last_run"),
                    "last_status": a.get("last_status", "pending"),
                    "runs": a.get("runs", 0),
                    "failures": a.get("failures", 0),
                    "schedule": sched_desc,
                    "cron_kwargs": cron_kwargs,
                }
        except Exception as e:
            data["parse_error"] = str(e)

    # Daemon PID check (only meaningful on scheduler service)
    status = get_status()
    if data["source"] != "postgres":
        data["daemon_running"] = status.get("running", False)
    data["daemon_pid"] = status.get("pid")
    if not data["daemon_message"]:
        data["daemon_message"] = status.get("message", "")

    return data


def _collect_registry_guardrails():
    """Section C — Agent Registry + Guardrails (read-only aside from idempotent register)."""
    data = {
        "infrastructure_ok": False,
        "agents": {},
        "circuit_breakers": {},
        "cost_summary": {},
    }

    try:
        from agents.core.agent_runner import _setup_infrastructure, AGENT_CATALOG
        from agents.core.agent_registry import AgentRegistry

        message_bus, guardrails, pg_conn = _setup_infrastructure()
        data["infrastructure_ok"] = True

        # Register all agents (idempotent ON CONFLICT)
        registry = AgentRegistry(pg_conn=pg_conn)
        for name, (mod, cls, schedule) in AGENT_CATALOG.items():
            registry.register(name, cls, schedule=schedule)

        statuses = registry.get_all_statuses()

        # Cost summary
        try:
            data["cost_summary"] = guardrails.get_daily_cost_summary()
        except Exception as e:
            data["cost_summary_error"] = str(e)

        # Circuit breaker per agent
        for name in AGENT_NAMES:
            s = statuses.get(name, {})
            try:
                tripped = guardrails.check_circuit_breaker(name, max_failures=3)
            except Exception:
                tripped = None

            data["agents"][name] = {
                "registry_status": s.get("status", "unknown"),
                "enabled": s.get("enabled", True),
                "schedule": s.get("schedule"),
                "last_run_at": str(s.get("last_run_at") or ""),
            }
            data["circuit_breakers"][name] = {
                "tripped": tripped,
                "state": "OPEN" if tripped is True else ("CLOSED" if tripped is False else "UNKNOWN"),
            }

    except Exception as e:
        data["error"] = str(e)
        data["traceback"] = traceback.format_exc()

    return data


def _collect_redis():
    """Section D — Redis connectivity (read-only, ping only)."""
    data = {"status": "UNKNOWN", "detail": ""}

    try:
        from agents.core.connections import get_redis_client
        redis_client = get_redis_client()

        if redis_client:
            try:
                pong = redis_client.ping()
                if pong:
                    data["status"] = "HEALTHY"
                    data["detail"] = "Redis PING successful"
                else:
                    data["status"] = "BROKEN"
                    data["detail"] = "Redis PING returned falsy"
            except Exception as e:
                data["status"] = "BROKEN"
                data["detail"] = f"Redis PING failed: {e}"
        else:
            data["status"] = "IN-MEMORY"
            data["detail"] = "IN-MEMORY ONLY — messages lost on Railway restart"
    except Exception as e:
        data["status"] = "BROKEN"
        data["detail"] = f"Import/connect error: {e}"

    return data


def _collect_databases():
    """Section E — Postgres + SQLite databases (all queries read-only)."""
    data = {"postgres": {}, "guardrails_db": {}, "calibration_db": {}, "bet_tracking_db": {}}

    # --- Postgres ---
    try:
        from agents.core.connections import get_postgres_connection
        pg_conn = get_postgres_connection()
        if pg_conn:
            data["postgres"]["connected"] = True
            cur = pg_conn.cursor()

            cur.execute("SHOW server_version")
            data["postgres"]["server_version"] = cur.fetchone()[0]

            for table in ["agent_runs", "agent_token_budgets", "agent_registry"]:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                    data["postgres"][f"{table}_count"] = cur.fetchone()[0]
                except Exception:
                    pg_conn.rollback() if hasattr(pg_conn, "rollback") else None
                    data["postgres"][f"{table}_count"] = "TABLE MISSING"

            cur.close()
            pg_conn.close()
        else:
            data["postgres"]["connected"] = False
            data["postgres"]["detail"] = "DATABASE_URL not set or connection failed"
    except Exception as e:
        data["postgres"]["connected"] = False
        data["postgres"]["error"] = str(e)

    # --- SQLite databases ---
    for db_key, db_name, primary_tables in [
        ("guardrails_db", "agent_guardrails.db", ["agent_runs", "agent_token_budgets"]),
        ("calibration_db", "calibration.db", ["predictions", "calibration_data", "prediction_outcomes"]),
        ("bet_tracking_db", "bet_tracking.db", ["tracked_bets", "bets", "clv_data"]),
    ]:
        db_path = PROJECT_DIR / "data" / db_name
        info = {"exists": db_path.exists(), "path": str(db_path)}

        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()]
                info["tables"] = tables

                # Check row counts + date ranges for primary tables
                info["table_details"] = {}
                for tbl in primary_tables:
                    if tbl not in tables:
                        continue
                    tbl_info = {}
                    try:
                        cnt = conn.execute(f"SELECT COUNT(*) FROM [{tbl}]").fetchone()[0]  # noqa: S608
                        tbl_info["rows"] = cnt

                        col_info = conn.execute(f"PRAGMA table_info([{tbl}])").fetchall()
                        date_cols = [c[1] for c in col_info if any(
                            kw in c[1].lower() for kw in ["date", "time", "created", "timestamp"]
                        )]
                        if date_cols:
                            col = date_cols[0]
                            row = conn.execute(
                                f"SELECT MIN([{col}]), MAX([{col}]) FROM [{tbl}]"  # noqa: S608
                            ).fetchone()
                            if row and row[0]:
                                tbl_info["min_date"] = str(row[0])
                                tbl_info["max_date"] = str(row[1])
                    except Exception as e:
                        tbl_info["error"] = str(e)
                    info["table_details"][tbl] = tbl_info

                # Staleness check for calibration
                if db_key == "calibration_db" and "predictions" in tables:
                    try:
                        col_info = conn.execute("PRAGMA table_info(predictions)").fetchall()
                        date_cols = [c[1] for c in col_info if "date" in c[1].lower()]
                        if date_cols:
                            max_date = conn.execute(
                                f"SELECT MAX([{date_cols[0]}]) FROM predictions"  # noqa: S608
                            ).fetchone()[0]
                            if max_date:
                                from datetime import timedelta
                                try:
                                    max_dt = datetime.strptime(str(max_date)[:10], "%Y-%m-%d")
                                    days_old = (datetime.now() - max_dt).days
                                    info["staleness_days"] = days_old
                                    if days_old > 7:
                                        info["stale_warning"] = f"Predictions data is {days_old} days old (>7 days)"
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # Recent runs for guardrails_db
                if db_key == "guardrails_db" and "agent_runs" in tables:
                    try:
                        rows = conn.execute(
                            "SELECT agent_name, run_id, started_at, status, tokens_used "
                            "FROM agent_runs ORDER BY started_at DESC LIMIT 5"
                        ).fetchall()
                        info["recent_runs"] = [
                            {"agent": r[0], "run_id": r[1], "started_at": r[2],
                             "status": r[3], "tokens": r[4] or 0}
                            for r in rows
                        ]
                    except Exception:
                        pass

                conn.close()
            except Exception as e:
                info["error"] = str(e)

        data[db_key] = info

    return data


def _collect_agent_health(scheduler_data, registry_data, pg_conn=None):
    """Section F — Agent health analysis from Postgres agent_runs (NO agent execution)."""
    agents = {}
    now = datetime.now(timezone.utc)

    # Schedule-specific staleness thresholds (in minutes)
    _HEALTHY_THRESHOLDS = {
        "odds_monitor": 30,     # runs every 15 min during 8AM-11PM
        "pregame": 360,         # runs twice daily (6h)
        "briefing": 360,        # runs twice daily (6h)
        "postgame": 1500,       # runs once daily (25h)
        "orchestrator": 1500,   # runs once daily (25h)
        "watchdog": 1500,       # runs once daily (25h)
    }

    # Pre-fetch per-agent data from Postgres if available
    pg_agent_data = {}
    if pg_conn is not None:
        try:
            import psycopg2.extras
            cursor = pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            for name in AGENT_NAMES:
                # Most recent run
                cursor.execute("""
                    SELECT status, started_at, completed_at, tokens_used,
                           messages_sent, errors
                    FROM agent_runs
                    WHERE agent_name = %s
                    ORDER BY started_at DESC
                    LIMIT 1
                """, (name,))
                last_run = cursor.fetchone()

                # Failures in last 24h
                cursor.execute("""
                    SELECT COUNT(*) as failure_count
                    FROM agent_runs
                    WHERE agent_name = %s
                      AND status = 'failed'
                      AND started_at >= NOW() - INTERVAL '24 hours'
                """, (name,))
                failures_24h = cursor.fetchone()["failure_count"]

                pg_agent_data[name] = {
                    "last_run": last_run,
                    "failures_24h": failures_24h,
                }

            cursor.close()
        except Exception:
            pg_agent_data = {}
            # Fall through — use scheduler_data below

    for name in AGENT_NAMES:
        agent_info = {
            "health": "UNKNOWN",
            "time_since_last_run_minutes": None,
            "next_scheduled_run": None,
            "details": {},
        }

        # Registry data (circuit breaker, token budget)
        reg_agent = registry_data.get("agents", {}).get(name, {})
        reg_status = reg_agent.get("registry_status", "unknown")
        cb_info = registry_data.get("circuit_breakers", {}).get(name, {})
        cb_state = cb_info.get("state", "UNKNOWN")
        cost_info = registry_data.get("cost_summary", {}).get(name, {})
        token_util = cost_info.get("utilization_pct", 0)

        # Schedule cron_kwargs for next_scheduled_run computation
        sched_agent = scheduler_data.get("agents", {}).get(name, {})
        cron_kwargs = sched_agent.get("cron_kwargs", {})

        # Compute next_scheduled_run via CronTrigger
        if cron_kwargs:
            try:
                from apscheduler.triggers.cron import CronTrigger
                trigger = CronTrigger(**cron_kwargs, timezone="America/New_York")
                next_fire = trigger.get_next_fire_time(None, now)
                if next_fire:
                    agent_info["next_scheduled_run"] = next_fire.isoformat()
            except Exception:
                pass

        # ----------------------------------------------------------
        # Determine health from Postgres (primary) or scheduler data (fallback)
        # ----------------------------------------------------------
        pg_data = pg_agent_data.get(name)
        last_run_row = pg_data["last_run"] if pg_data else None

        if last_run_row is not None:
            # Postgres-sourced health
            last_status = last_run_row["status"]
            started_at = last_run_row["started_at"]
            failures = pg_data["failures_24h"]

            if started_at is not None:
                # started_at is a datetime from psycopg2, ensure tz-aware
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=timezone.utc)
                time_diff = now - started_at
                time_since = int(time_diff.total_seconds() / 60)
                agent_info["time_since_last_run_minutes"] = time_since

                threshold = _HEALTHY_THRESHOLDS.get(name, 1500)
                if last_status == "failed" and failures >= 3:
                    health = "FAILED"
                elif last_status == "completed" and time_since < threshold:
                    health = "HEALTHY"
                elif last_status == "completed" or last_status == "failed":
                    health = "DEGRADED"
                else:
                    health = "DEGRADED"
            else:
                health = "NEVER_RUN"

            agent_info["health"] = health
            agent_info["details"] = {
                "last_status": last_status,
                "failures": failures,
                "registry_status": reg_status,
                "cb_state": cb_state,
                "token_utilization_pct": token_util,
                "last_run_at": started_at.isoformat() if started_at else None,
                "tokens_used": last_run_row["tokens_used"] or 0,
                "messages_sent": last_run_row["messages_sent"] or 0,
            }
        else:
            # Fallback: use scheduler_data (status file or empty)
            last_run_str = sched_agent.get("last_run")
            last_status = sched_agent.get("last_status", "")
            failures = sched_agent.get("failures", 0)

            # Parse last_run timestamp
            if last_run_str:
                try:
                    clean = str(last_run_str).replace("Z", "+00:00")
                    if "+" not in clean and len(clean) <= 19:
                        clean += "+00:00"
                    last_run_dt = datetime.fromisoformat(clean)
                    if last_run_dt.tzinfo is None:
                        last_run_dt = last_run_dt.replace(tzinfo=timezone.utc)
                    delta = now - last_run_dt
                    agent_info["time_since_last_run_minutes"] = round(delta.total_seconds() / 60, 1)
                except Exception:
                    pass

            # Health classification from scheduler data
            if not last_run_str and reg_status in ("idle", "unknown"):
                health = "NEVER_RUN"
            elif last_status in ("failed", "error"):
                health = "FAILED"
            elif cb_info.get("tripped") is True:
                health = "DEGRADED"
            elif last_status == "completed":
                health = "HEALTHY"
            elif last_status == "pending":
                health = "NEVER_RUN"
            else:
                health = "UNKNOWN"

            agent_info["health"] = health
            agent_info["details"] = {
                "last_status": last_status,
                "failures": failures,
                "registry_status": reg_status,
                "cb_state": cb_state,
                "token_utilization_pct": token_util,
            }

        agents[name] = agent_info

    return agents


def _collect_daemon_health(pg_conn=None):
    """Section G — Scheduler process health via Postgres activity or PID file."""
    data = {"status": "UNKNOWN", "detail": "", "pid": None, "pid_alive": False}

    # ------------------------------------------------------------------
    # Primary: detect scheduler activity via Postgres agent_runs
    # ------------------------------------------------------------------
    if pg_conn is not None:
        try:
            import psycopg2.extras
            cursor = pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT MAX(started_at) as most_recent
                FROM agent_runs
                WHERE started_at >= NOW() - INTERVAL '2 hours'
            """)
            result = cursor.fetchone()
            most_recent = result["most_recent"] if result else None
            cursor.close()

            if most_recent:
                if most_recent.tzinfo is None:
                    most_recent = most_recent.replace(tzinfo=timezone.utc)
                minutes_ago = int((datetime.now(timezone.utc) - most_recent).total_seconds() / 60)
                data["status"] = "OK"
                data["detail"] = f"Scheduler active — last run {minutes_ago} minutes ago"
                data["pid_alive"] = True
                return data
            else:
                data["status"] = "FAIL"
                data["detail"] = "No agent activity in last 2 hours — scheduler may be down"
                data["pid_alive"] = False
                return data

        except Exception as e:
            data["detail"] = f"Postgres query failed: {e}"
            # Fall through to PID file check

    # ------------------------------------------------------------------
    # Fallback: PID file (only works on scheduler service)
    # ------------------------------------------------------------------
    try:
        from agents.core.agent_scheduler import PID_FILE
    except Exception as e:
        data["detail"] = f"Import error: {e}"
        return data

    pid_path = Path(PID_FILE) if not isinstance(PID_FILE, Path) else PID_FILE
    is_railway = bool(os.environ.get("RAILWAY_ENVIRONMENT"))

    if not pid_path.exists():
        if is_railway:
            data["status"] = "FAIL"
            data["detail"] = "PID file missing on Railway — scheduler not running"
        else:
            data["status"] = "WARN"
            data["detail"] = "PID file missing (expected in local dev)"
        return data

    try:
        pid = int(pid_path.read_text().strip())
        data["pid"] = pid
        os.kill(pid, 0)
        data["pid_alive"] = True
        data["status"] = "OK"
        data["detail"] = f"Scheduler daemon alive (PID {pid})"
    except (OSError, ValueError):
        data["pid_alive"] = False
        if is_railway:
            data["status"] = "FAIL"
            data["detail"] = f"PID {data.get('pid', '?')} not alive on Railway — scheduler crashed"
        else:
            data["status"] = "WARN"
            data["detail"] = f"PID {data.get('pid', '?')} not alive (expected in local dev)"

    return data


def _collect_summary(env, scheduler, registry, redis, databases, agent_health, daemon):
    """Section H — Summary + suggested actions."""
    suggestions = []

    # Infrastructure status
    infra = {
        "scheduler": "OK" if scheduler.get("daemon_running") else "DOWN",
        "redis": redis.get("status", "UNKNOWN"),
        "postgres": "OK" if databases.get("postgres", {}).get("connected") else "DOWN",
        "guardrails_db": "OK" if databases.get("guardrails_db", {}).get("exists") else "MISSING",
        "calibration_db": "OK" if databases.get("calibration_db", {}).get("exists") else "MISSING",
        "bet_tracking_db": "OK" if databases.get("bet_tracking_db", {}).get("exists") else "MISSING",
    }

    # Per-agent one-liner
    agent_lines = {}
    for name in AGENT_NAMES:
        ah = agent_health.get(name, {})
        health = ah.get("health", "UNKNOWN")
        mins = ah.get("time_since_last_run_minutes")
        if mins is not None:
            time_str = f"last run {int(mins)}m ago"
        else:
            time_str = "never run"
        agent_lines[name] = {"health": health, "summary": f"{health.lower()}, {time_str}"}

    # Generate suggestions
    if env.get("gemini_api_key") == "ABSENT":
        suggestions.append("Set GEMINI_API_KEY in Railway env (or .env locally)")
    if env.get("apscheduler") == "NOT INSTALLED":
        suggestions.append("Install APScheduler: pip install apscheduler")
    if not scheduler.get("daemon_running") and not daemon.get("pid_alive"):
        suggestions.append("Start unified scheduler: python agents/core/agent_scheduler.py --daemon")
    if scheduler.get("status_file_stale"):
        suggestions.append(f"Scheduler status file is stale ({scheduler.get('stale_hours', '?')}h old)")
    if redis.get("status") == "IN-MEMORY":
        suggestions.append("Set REDIS_URL for persistent message bus across process restarts")
    if redis.get("status") == "BROKEN":
        suggestions.append("Fix Redis connection — check REDIS_URL configuration")
    if not databases.get("postgres", {}).get("connected"):
        suggestions.append("Set DATABASE_URL for PostgreSQL (agent registry + guardrails persistence)")
    if not databases.get("calibration_db", {}).get("exists"):
        suggestions.append("calibration.db missing — orchestrator may not have data")
    for db_key in ["calibration_db", "bet_tracking_db"]:
        warn_msg = databases.get(db_key, {}).get("stale_warning")
        if warn_msg:
            suggestions.append(warn_msg)
    for name in AGENT_NAMES:
        ah = agent_health.get(name, {})
        if ah.get("health") == "FAILED":
            suggestions.append(f"Check {name} logs — multiple recent failures")
        elif ah.get("health") == "DEGRADED":
            suggestions.append(f"{name} hasn't run recently — verify schedule")

    # Overall status based on agent health counts
    healths = [agent_health.get(n, {}).get("health", "UNKNOWN") for n in AGENT_NAMES]
    healthy_count = sum(1 for h in healths if h == "HEALTHY")
    failed_count = sum(1 for h in healths if h == "FAILED")
    degraded_count = sum(1 for h in healths if h in ("DEGRADED", "STALE"))
    never_run_count = sum(1 for h in healths if h == "NEVER_RUN")
    total_agents = len(healths)

    if never_run_count == total_agents:
        overall = "CRITICAL"
    elif healthy_count == total_agents:
        overall = "ALL_GREEN"
    elif failed_count > 0 or not daemon.get("pid_alive") or degraded_count > 0:
        overall = "DEGRADED"
    else:
        overall = "ALL_GREEN"

    return {
        "overall_status": overall,
        "infrastructure": infra,
        "agents": agent_lines,
        "suggestions": suggestions,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# Core collection
# ============================================================================

def _collect_all():
    """Collect all diagnostic data into a structured dict."""
    env = _collect_environment()

    # Get a shared Postgres connection for sections that query agent_runs
    pg_conn = None
    try:
        from agents.core.connections import get_postgres_connection
        pg_conn = get_postgres_connection()
    except Exception:
        pass

    scheduler = _collect_scheduler(pg_conn=pg_conn)
    registry = _collect_registry_guardrails()
    redis = _collect_redis()
    databases = _collect_databases()
    agent_health = _collect_agent_health(scheduler, registry, pg_conn=pg_conn)
    daemon = _collect_daemon_health(pg_conn=pg_conn)
    summary = _collect_summary(env, scheduler, registry, redis, databases, agent_health, daemon)

    # Clean up shared connection
    if pg_conn is not None:
        try:
            pg_conn.close()
        except Exception:
            pass

    return {
        "environment": env,
        "scheduler": scheduler,
        "registry_guardrails": registry,
        "redis": redis,
        "databases": databases,
        "agent_health": agent_health,
        "daemon": daemon,
        "summary": summary,
    }


# ============================================================================
# CLI rendering
# ============================================================================

def _render_cli(data):
    """Print a formatted CLI report from collected data."""
    def banner(title):
        print()
        print("=" * WIDTH)
        print(f"  {title}")
        print("=" * WIDTH)

    def sub(title):
        print()
        print(f"--- {title} " + "-" * max(0, WIDTH - len(title) - 5))

    def ok(msg):
        print(f"  [OK]   {msg}")

    def warn(msg):
        print(f"  [WARN] {msg}")

    def fail(msg):
        print(f"  [FAIL] {msg}")

    def info(msg):
        print(f"  [INFO] {msg}")

    # Header
    print()
    print("=" * WIDTH)
    print("  NBA-BETS AGENT SYSTEM — READ-ONLY DIAGNOSTIC REPORT")
    print(f"  Generated: {datetime.now(timezone.utc).isoformat()[:19]}Z")
    print("=" * WIDTH)

    # --- A: Environment ---
    env = data["environment"]
    banner("SECTION A — ENVIRONMENT + DEPENDENCIES")
    info(f"Python version: {env['python_version']} ({env['python_executable']})")
    info(f"PROJECT_DIR:    {env['project_dir']}")
    info(f"CWD:            {env['cwd']}")
    if env["is_railway"]:
        ok(f"Railway detected: {env['railway_environment']}")
    else:
        info("Running locally (no RAILWAY_ENVIRONMENT)")

    sub("Key environment variables")
    info(f"DATABASE_URL:   {env['database_url']}")
    info(f"REDIS_URL:      {env['redis_url']}")
    info(f"GEMINI_API_KEY: {env['gemini_api_key']}")

    sub("Dependencies")
    if env["apscheduler"] == "ok":
        ok("APScheduler importable")
    else:
        fail(f"APScheduler: {env['apscheduler']}")
    if env["load_env"] == "ok":
        ok("load_env imported")
    else:
        warn(f"load_env: {env['load_env']}")

    # --- B: Scheduler ---
    sched = data["scheduler"]
    banner("SECTION B — SCHEDULER STATUS")
    if sched.get("error"):
        fail(sched["error"])
    else:
        source = sched.get("source", "none")
        info(f"Data source:    {source}")
        if source == "postgres":
            ok("Reading scheduler activity from Postgres agent_runs")
        elif source == "status_file":
            info("Reading from local status file (scheduler service only)")
        else:
            warn("No scheduler data source available")

        if sched.get("postgres_error"):
            warn(f"Postgres query failed: {sched['postgres_error']}")

        info(f"total_runs(24h): {sched['total_runs']}")
        info(f"total_failures:  {sched['total_failures']}")

        if sched.get("start_time"):
            info(f"start_time:      {sched['start_time']}")
        if sched.get("status_file_stale"):
            warn(f"Status file is STALE ({sched['stale_hours']}h since last update)")

        for name in AGENT_NAMES:
            a = sched["agents"].get(name, {})
            info(f"  {name:<15} last_run={str(a.get('last_run', '-'))[:19]:<22} "
                 f"status={a.get('last_status', '-'):<10} "
                 f"runs={a.get('runs', 0):>3}  fails={a.get('failures', 0):>3}  "
                 f'schedule="{a.get("schedule", "N/A")}"')

    sub("Scheduler activity")
    if sched["daemon_running"]:
        ok(sched["daemon_message"])
    else:
        warn(sched["daemon_message"])

    # --- C: Registry + Guardrails ---
    reg = data["registry_guardrails"]
    banner("SECTION C — AGENT REGISTRY + GUARDRAILS STATUS")
    if reg.get("error"):
        fail(f"Infrastructure setup failed: {reg['error']}")
    else:
        ok("_setup_infrastructure() succeeded")

        sub("Agent Registry + Guardrails table")
        header = (f"  {'AGENT':<15} {'STATUS':<12} {'ENABLED':<8} "
                  f"{'LAST RUN':<22} {'CB_STATE':<8} {'TOKENS':>12}")
        print(header)
        print(f"  {'-'*13:<15} {'-'*10:<12} {'-'*6:<8} "
              f"{'-'*20:<22} {'-'*6:<8} {'-'*11:>12}")

        for name in AGENT_NAMES:
            a = reg["agents"].get(name, {})
            cb = reg["circuit_breakers"].get(name, {})
            tokens = reg.get("cost_summary", {}).get(name, {}).get("used_today", 0)
            print(f"  {name:<15} {a.get('registry_status', '?'):<12} "
                  f"{'yes' if a.get('enabled', True) else 'NO':<8} "
                  f"{a.get('last_run_at', '-')[:22]:<22} "
                  f"{cb.get('state', '?'):<8} {tokens:>12}")
        print()

    # --- D: Redis ---
    redis_data = data["redis"]
    banner("SECTION D — MESSAGE BUS / REDIS CONNECTIVITY")
    status_fn = {"HEALTHY": ok, "BROKEN": fail, "IN-MEMORY": warn}.get(redis_data["status"], info)
    status_fn(f"Redis: {redis_data['status']} — {redis_data['detail']}")

    # --- E: Databases ---
    dbs = data["databases"]
    banner("SECTION E — DATABASES")

    sub("PostgreSQL")
    pg = dbs["postgres"]
    if pg.get("connected"):
        ok(f"Connected — server v{pg.get('server_version', '?')}")
        for tbl in ["agent_runs", "agent_token_budgets", "agent_registry"]:
            cnt = pg.get(f"{tbl}_count", "?")
            info(f"  {tbl}: {cnt} rows")
    else:
        warn(f"PostgreSQL unavailable: {pg.get('detail', pg.get('error', ''))}")

    for db_key, label in [
        ("guardrails_db", "agent_guardrails.db"),
        ("calibration_db", "calibration.db"),
        ("bet_tracking_db", "bet_tracking.db"),
    ]:
        sub(label)
        db = dbs.get(db_key, {})
        if not db.get("exists"):
            warn(f"File missing: {db.get('path', '?')}")
            continue
        ok(f"File exists: {db.get('path', '?')}")
        if db.get("tables"):
            info(f"Tables: {', '.join(db['tables'])}")
        for tbl, details in db.get("table_details", {}).items():
            parts = [f"{details.get('rows', '?')} rows"]
            if details.get("min_date"):
                parts.append(f"dates {details['min_date']} to {details['max_date']}")
            info(f"  {tbl}: {', '.join(parts)}")
        if db.get("stale_warning"):
            warn(db["stale_warning"])
        if db.get("recent_runs"):
            info("Most recent runs:")
            for r in db["recent_runs"]:
                info(f"  {r['agent']:<15} {str(r['started_at'])[:19]:<22} "
                     f"status={r['status']:<10} tokens={r['tokens']}")

    # --- F: Agent Health ---
    ah = data["agent_health"]
    banner("SECTION F — AGENT HEALTH ANALYSIS (READ-ONLY)")
    print(f"  {'AGENT':<15} {'HEALTH':<12} {'LAST RUN':<18} {'NEXT RUN':<22} {'CB':>6} {'DETAILS'}")
    print(f"  {'-'*13:<15} {'-'*10:<12} {'-'*16:<18} {'-'*20:<22} {'-'*4:>6} {'-'*20}")
    for name in AGENT_NAMES:
        a = ah.get(name, {})
        health = a.get("health", "?")
        mins = a.get("time_since_last_run_minutes")
        last_str = f"{int(mins)}m ago" if mins is not None else "never"
        next_run = str(a.get("next_scheduled_run") or "-")[:19]
        cb = a.get("details", {}).get("cb_state", "?")
        detail_parts = []
        d = a.get("details", {})
        if d.get("last_status"):
            detail_parts.append(f"status={d['last_status']}")
        if d.get("failures"):
            detail_parts.append(f"fails={d['failures']}")
        detail_str = ", ".join(detail_parts)
        print(f"  {name:<15} {health:<12} {last_str:<18} {next_run:<22} {cb:>6} {detail_str}")
    print()

    # --- G: Daemon Health ---
    daemon = data["daemon"]
    banner("SECTION G — RAILWAY DAEMON PROCESS HEALTH")
    status_fn = {"OK": ok, "FAIL": fail, "WARN": warn}.get(daemon["status"], info)
    status_fn(daemon["detail"])

    # --- H: Summary ---
    summary = data["summary"]
    banner("SECTION H — SUMMARY + SUGGESTED ACTIONS")

    sub("Overall status")
    overall = summary["overall_status"]
    marker = {"ALL_GREEN": "[OK]  ", "DEGRADED": "[WARN]", "CRITICAL": "[FAIL]"}.get(overall, "      ")
    print(f"  {marker} {overall}")

    sub("Agent health")
    emoji_map = {"HEALTHY": "\u2705", "DEGRADED": "\u26a0\ufe0f ", "FAILED": "\u274c",
                 "STALE": "\u23f0", "NEVER_RUN": "\u2796", "UNKNOWN": "\u2753"}
    for name in AGENT_NAMES:
        al = summary["agents"].get(name, {})
        health = al.get("health", "UNKNOWN")
        emoji = emoji_map.get(health, "  ")
        print(f"  {emoji} {name} — {al.get('summary', '?')}")

    sub("Infrastructure")
    for key, status in summary["infrastructure"].items():
        marker = "[OK]  " if status == "OK" else ("[WARN]" if status in ("IN-MEMORY", "MISSING") else "[FAIL]")
        print(f"  {marker} {key:<20} {status}")

    sub("Suggested actions")
    if summary["suggestions"]:
        for i, action in enumerate(summary["suggestions"], 1):
            print(f"  {i}. {action}")
    else:
        ok("No actions needed — system is healthy!")

    print()
    print("=" * WIDTH)
    print(f"  DIAGNOSTICS COMPLETE — {summary['generated_at'][:19]}Z")
    print("=" * WIDTH)
    print()


# ============================================================================
# Helpers
# ============================================================================

def _mask(val, prefix_len=12):
    if not val:
        return "(not set)"
    if len(val) <= prefix_len:
        return val[:4] + "..."
    return val[:prefix_len] + "..."



# ============================================================================
# Public API
# ============================================================================

def run_diagnostics(return_json=False):
    """
    Run all diagnostics.

    Args:
        return_json: If True, return dict (for API). If False, print CLI report.

    Returns:
        dict if return_json=True, else None.
    """
    data = _collect_all()
    if return_json:
        return data
    _render_cli(data)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NBA-BETS agent system diagnostics (read-only)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of CLI report")
    args = parser.parse_args()

    if args.json:
        data = run_diagnostics(return_json=True)
        print(json.dumps(data, indent=2, default=str))
    else:
        run_diagnostics(return_json=False)


if __name__ == "__main__":
    main()
