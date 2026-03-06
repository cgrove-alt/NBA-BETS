#!/usr/bin/env python3
"""
Unified Agent Scheduler — runs all 6 agents from a single daemon.

Consolidates 6 separate Railway cron services into one always-on worker
using APScheduler with CronTrigger. Each agent runs on its own schedule
in its own thread; failures are isolated per-agent.

Usage:
    python3 agent_scheduler.py --start           # Start in foreground
    python3 agent_scheduler.py --daemon          # Run as background daemon (Railway)
    python3 agent_scheduler.py --status          # Show scheduler + agent status
    python3 agent_scheduler.py --stop            # Stop running daemon
    python3 agent_scheduler.py --trigger pregame # Run one agent immediately

For Railway deployment:
    Start Command: python agent_scheduler.py --daemon
"""

import os
import sys
import json
import time
import logging
import argparse
import signal
import atexit
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Ensure project root is on path
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import load_env  # noqa: F401  — load .env before any code reads os.environ

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False
    print("ERROR: APScheduler not installed. Install with: pip install apscheduler")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = PROJECT_DIR / "data"
LOGS_DIR = PROJECT_DIR / "logs"
PID_FILE = DATA_DIR / "agent_scheduler.pid"
STATUS_FILE = DATA_DIR / "agent_scheduler_status.json"

LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'agent_scheduler.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('agent_scheduler')

# Agent schedules — maps to AGENT_CATALOG in agent_runner.py
# Format: (cron_kwargs, description)
AGENT_SCHEDULES = {
    'pregame': (
        {'hour': '11,17', 'minute': '0'},
        '11 AM + 5 PM ET — injury/lineup intel',
    ),
    'postgame': (
        {'hour': '1', 'minute': '0'},
        '1 AM ET — post-game miss analysis',
    ),
    'odds_monitor': (
        {'hour': '8-23', 'minute': '*/15'},
        'Every 15 min (8 AM-11 PM ET) — line movements',
    ),
    'orchestrator': (
        [
            {'hour': '11', 'minute': '30'},
            {'hour': '17', 'minute': '15'},
        ],
        '11:30 AM + 5:15 PM ET — prediction pipeline',
    ),
    'watchdog': (
        {'hour': '1', 'minute': '30'},
        '1:30 AM ET — model health check',
    ),
    'briefing': (
        {'hour': '12,18', 'minute': '0'},
        'Noon + 6 PM ET — daily briefing',
    ),
}


# ============================================================================
# Job wrapper
# ============================================================================

# Runtime stats (in-memory, persisted to STATUS_FILE periodically)
_stats = {
    'start_time': None,
    'total_runs': 0,
    'total_failures': 0,
    'agents': {},
}


def _init_stats():
    """Initialize per-agent stats."""
    _stats['start_time'] = datetime.now().isoformat()
    for name in AGENT_SCHEDULES:
        _stats['agents'][name] = {
            'last_run': None,
            'last_status': 'pending',
            'last_duration_s': None,
            'last_tokens': None,
            'runs': 0,
            'failures': 0,
            'consecutive_failures': 0,
        }


def _persist_stats():
    """Write current stats to disk."""
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(_stats, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to persist stats: {e}")


# Agents whose failure/recovery should trigger push notifications
CRITICAL_AGENTS = {'briefing', 'orchestrator'}

# Max age (hours) for a missed job to still be recoverable
RECOVERY_WINDOW_HOURS = 4


def _send_scheduler_notification(title: str, message: str, priority: int = 0):
    """Send a Pushover notification for scheduler events. Never raises."""
    try:
        from agents.core.notifications import send_pushover
        send_pushover(title, message, priority=priority)
    except Exception as e:
        logger.warning(f"Scheduler notification failed: {e}")


def _recover_missed_jobs():
    """
    Check if any recoverable agents missed their scheduled run while
    the scheduler was down. If a critical agent was due within the last
    RECOVERY_WINDOW_HOURS, trigger it immediately.
    """
    if not STATUS_FILE.exists():
        logger.info("No previous status file — skipping missed job recovery")
        return

    try:
        with open(STATUS_FILE) as f:
            saved = json.load(f)
    except Exception:
        logger.warning("Could not read status file for recovery check")
        return

    last_start = saved.get('start_time')
    if not last_start:
        return

    try:
        last_alive = datetime.fromisoformat(last_start)
    except (ValueError, TypeError):
        return

    now = datetime.now()
    downtime = now - last_alive

    # Only check if downtime is between 10 minutes and RECOVERY_WINDOW_HOURS
    if downtime < timedelta(minutes=10) or downtime > timedelta(hours=RECOVERY_WINDOW_HOURS):
        if downtime > timedelta(hours=RECOVERY_WINDOW_HOURS):
            logger.info(f"Downtime ({downtime}) exceeds recovery window — skipping recovery")
        return

    logger.info(f"Scheduler was down for {downtime}. Checking for missed critical jobs...")

    recovered = []
    for agent_name in CRITICAL_AGENTS:
        agent_info = saved.get('agents', {}).get(agent_name, {})
        last_run_str = agent_info.get('last_run')

        if last_run_str:
            try:
                last_run = datetime.fromisoformat(last_run_str)
                time_since_last = now - last_run
                # If last run was more than 2 hours ago, it likely missed a cycle
                if time_since_last < timedelta(hours=2):
                    continue
            except (ValueError, TypeError):
                pass

        logger.info(f"[{agent_name}] Missed run detected — triggering recovery")
        try:
            run_agent_job(agent_name)
            recovered.append(agent_name)
        except Exception as e:
            logger.error(f"[{agent_name}] Recovery run failed: {e}")

    if recovered:
        msg = f"Recovered after {downtime}: {', '.join(recovered)}"
        logger.info(msg)
        _send_scheduler_notification("Scheduler Recovery", msg, priority=0)


def _scheduler_health_check():
    """
    Periodic health check — runs every 2 hours.

    Looks for agents with 3+ consecutive failures and alerts.
    """
    problem_agents = []

    for agent_name, info in _stats.get('agents', {}).items():
        consecutive = info.get('consecutive_failures', 0)
        runs = info.get('runs', 0)

        if consecutive >= 3:
            problem_agents.append(f"{agent_name}: {consecutive} consecutive failures in {runs} runs")

    if problem_agents:
        msg = "Agents with repeated failures:\n" + "\n".join(f"  - {p}" for p in problem_agents)
        logger.warning(f"Health check alert: {msg}")
        _send_scheduler_notification("Agent Health Alert", msg, priority=1)
    else:
        logger.info("Health check: all agents healthy")


def run_agent_job(agent_name: str):
    """
    Wrapper that runs a single agent via agent_runner.run_agent().

    Catches all exceptions so one agent failure never crashes the daemon.
    Logs outcome with duration and token count.
    """
    from agents.core.agent_runner import run_agent

    start = datetime.now()
    agent_stats = _stats['agents'].get(agent_name, {})
    agent_stats['last_run'] = start.isoformat()
    agent_stats['runs'] = agent_stats.get('runs', 0) + 1
    _stats['total_runs'] += 1

    try:
        logger.info(f"[{agent_name}] Starting scheduled run")
        exit_code = run_agent(agent_name)
        duration = (datetime.now() - start).total_seconds()

        if exit_code == 0:
            agent_stats['last_status'] = 'completed'
            agent_stats['consecutive_failures'] = 0
            logger.info(f"[{agent_name}] Completed in {duration:.1f}s")
        else:
            agent_stats['last_status'] = 'failed'
            agent_stats['failures'] = agent_stats.get('failures', 0) + 1
            agent_stats['consecutive_failures'] = agent_stats.get('consecutive_failures', 0) + 1
            _stats['total_failures'] += 1
            logger.error(f"[{agent_name}] Failed (exit code {exit_code}) after {duration:.1f}s")

        agent_stats['last_duration_s'] = round(duration, 1)

    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        agent_stats['last_status'] = 'error'
        agent_stats['last_duration_s'] = round(duration, 1)
        agent_stats['failures'] = agent_stats.get('failures', 0) + 1
        agent_stats['consecutive_failures'] = agent_stats.get('consecutive_failures', 0) + 1
        _stats['total_failures'] += 1
        logger.error(f"[{agent_name}] Exception after {duration:.1f}s: {e}", exc_info=True)

    _stats['agents'][agent_name] = agent_stats
    _persist_stats()


# ============================================================================
# Scheduler setup
# ============================================================================

def create_scheduler(daemon: bool = False):
    """Create APScheduler with all 6 agent jobs."""
    scheduler = BackgroundScheduler(
        timezone='America/New_York',
    ) if daemon else BlockingScheduler(
        timezone='America/New_York',
    )

    for agent_name, (cron_kwargs, description) in AGENT_SCHEDULES.items():
        # Support both single cron dict and list of cron dicts
        schedules = cron_kwargs if isinstance(cron_kwargs, list) else [cron_kwargs]
        for i, kwargs in enumerate(schedules):
            job_id = f'agent_{agent_name}' if len(schedules) == 1 else f'agent_{agent_name}_{i}'
            scheduler.add_job(
                run_agent_job,
                CronTrigger(**kwargs),
                args=[agent_name],
                id=job_id,
                name=f'{agent_name}: {description}',
                max_instances=1,
                coalesce=True,
                misfire_grace_time=600,  # 10 min grace
            )
        logger.info(f"Scheduled: {agent_name} — {description}")

    # Health check job — runs every 2 hours
    from apscheduler.triggers.interval import IntervalTrigger
    scheduler.add_job(
        _scheduler_health_check,
        IntervalTrigger(hours=2),
        id='scheduler_health_check',
        name='Scheduler health check (every 2h)',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    logger.info("Scheduled: health_check — every 2 hours")

    # Job event listener — alerts on critical agent failures
    def job_listener(event):
        job_id = event.job_id.replace('agent_', '')
        if event.code == EVENT_JOB_EXECUTED:
            logger.debug(f"Job '{job_id}' executed successfully")
        elif event.code == EVENT_JOB_ERROR:
            logger.error(f"Job '{job_id}' raised exception: {event.exception}")
            # Alert on critical agent failures
            agent_name = job_id.split('_')[0] if '_' in job_id else job_id
            if agent_name in CRITICAL_AGENTS:
                _send_scheduler_notification(
                    f"Critical Job Failed: {agent_name}",
                    f"Agent '{agent_name}' raised an exception:\n{str(event.exception)[:500]}",
                    priority=1,
                )

    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    return scheduler


# ============================================================================
# PID management
# ============================================================================

def _get_boot_id() -> str:
    """Return a string unique to this OS boot / container lifecycle."""
    # Linux: /proc/sys/kernel/random/boot_id is unique per boot
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    if boot_id_path.exists():
        try:
            return boot_id_path.read_text().strip()
        except Exception:
            pass
    # Fallback: process start time of PID 1 (changes on container restart)
    try:
        return str(Path("/proc/1").stat().st_mtime)
    except Exception:
        pass
    # macOS / other: use system boot time
    try:
        import subprocess
        r = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def save_pid():
    """Write PID + boot ID so we can detect stale files from previous containers."""
    boot_id = _get_boot_id()
    with open(PID_FILE, 'w') as f:
        f.write(f"{os.getpid()}\n{boot_id}")


def remove_pid():
    if PID_FILE.exists():
        PID_FILE.unlink()


def get_status() -> dict:
    """Get scheduler status (from PID file + persisted stats)."""
    result = {'running': False, 'pid': None, 'message': 'Scheduler not running'}

    if PID_FILE.exists():
        try:
            lines = PID_FILE.read_text().strip().split('\n')
            pid = int(lines[0])
            saved_boot_id = lines[1] if len(lines) > 1 else None
            current_boot_id = _get_boot_id()

            # No boot ID → old format, definitely stale (predates this fix)
            # Different boot ID → stale file from a previous container
            if saved_boot_id is None or saved_boot_id != current_boot_id:
                logger.info(f"Stale PID file detected (PID {pid}, boot_id={'missing' if saved_boot_id is None else 'changed'})")
                remove_pid()
                return result  # not running

            # Same boot — but if it's our own PID, we haven't started yet
            if pid == os.getpid():
                logger.info(f"Stale PID file from current process (PID {pid})")
                remove_pid()
                return result  # not running

            # Same boot, different PID — check if that process is alive
            os.kill(pid, 0)
            result = {'running': True, 'pid': pid, 'message': f'Running (PID {pid})'}
        except (OSError, ValueError):
            remove_pid()
            result['message'] = 'Stale PID file (cleaned)'

    # Merge persisted stats
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE) as f:
                saved = json.load(f)
            result['start_time'] = saved.get('start_time')
            result['total_runs'] = saved.get('total_runs', 0)
            result['total_failures'] = saved.get('total_failures', 0)
            result['agents'] = saved.get('agents', {})

            # Add schedule info
            for name, info in result.get('agents', {}).items():
                _, desc = AGENT_SCHEDULES.get(name, ({}, ''))
                info['schedule'] = desc
        except Exception:
            pass

    return result


def stop_scheduler():
    if not PID_FILE.exists():
        print("Scheduler not running (no PID file)")
        return False

    try:
        lines = PID_FILE.read_text().strip().split('\n')
        pid = int(lines[0])
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}")
        remove_pid()
        return True
    except (OSError, ValueError) as e:
        print(f"Failed to stop: {e}")
        remove_pid()
        return False


# ============================================================================
# CLI
# ============================================================================

def print_status():
    status = get_status()

    print("=" * 65)
    print("AGENT SCHEDULER STATUS")
    print("=" * 65)
    print(f"  Status:     {status['message']}")
    if status.get('start_time'):
        print(f"  Started:    {status['start_time']}")
    print(f"  Total runs: {status.get('total_runs', 0)}  "
          f"(failures: {status.get('total_failures', 0)})")
    print()

    agents = status.get('agents', {})
    if agents:
        print(f"  {'Agent':<15} {'Last Status':<12} {'Last Run':<22} {'Runs':>5} {'Fails':>5}")
        print(f"  {'-'*13:<15} {'-'*10:<12} {'-'*20:<22} {'-'*5:>5} {'-'*5:>5}")
        for name in AGENT_SCHEDULES:
            info = agents.get(name, {})
            last_run = info.get('last_run', '-')
            if last_run and last_run != '-':
                last_run = last_run[:19]  # trim microseconds
            print(f"  {name:<15} {info.get('last_status', '-'):<12} "
                  f"{last_run:<22} {info.get('runs', 0):>5} {info.get('failures', 0):>5}")
        print()
        print("  Schedules:")
        for name, (_, desc) in AGENT_SCHEDULES.items():
            print(f"    {name:<15} {desc}")
    print("=" * 65)


def trigger_agent(agent_name: str):
    """Run one agent immediately (manual override)."""
    if agent_name not in AGENT_SCHEDULES:
        print(f"Unknown agent: {agent_name}")
        print(f"Available: {', '.join(AGENT_SCHEDULES.keys())}")
        return 1

    print(f"Triggering {agent_name} manually...")
    run_agent_job(agent_name)

    info = _stats['agents'].get(agent_name, {})
    print(f"Result: {info.get('last_status', 'unknown')} "
          f"({info.get('last_duration_s', '?')}s)")
    return 0 if info.get('last_status') == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(
        description='Unified Agent Scheduler — runs all 6 agents from a single daemon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true', help='Start in foreground')
    group.add_argument('--daemon', action='store_true', help='Run as background daemon (Railway)')
    group.add_argument('--status', action='store_true', help='Show scheduler + agent status')
    group.add_argument('--stop', action='store_true', help='Stop running daemon')
    group.add_argument('--trigger', type=str, metavar='AGENT',
                       help='Run one agent immediately (e.g. --trigger pregame)')

    args = parser.parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    elif args.stop:
        success = stop_scheduler()
        sys.exit(0 if success else 1)

    elif args.trigger:
        _init_stats()
        sys.exit(trigger_agent(args.trigger))

    elif args.start or args.daemon:
        # Check if already running
        status = get_status()
        if status['running']:
            print(f"ERROR: {status['message']}")
            sys.exit(1)

        _init_stats()

        scheduler = create_scheduler(daemon=args.daemon)

        save_pid()
        atexit.register(remove_pid)

        shutdown_event = threading.Event()

        def signal_handler(signum, frame):
            logger.info("Received shutdown signal, flagging exit...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("=" * 60)
        logger.info("UNIFIED AGENT SCHEDULER STARTED")
        logger.info("=" * 60)
        logger.info(f"Mode: {'Daemon' if args.daemon else 'Foreground'}")
        logger.info(f"PID: {os.getpid()}")
        logger.info("Scheduled jobs:")
        for job in scheduler.get_jobs():
            logger.info(f"  - {job.name} | next: {job.trigger}")
        logger.info("=" * 60)

        # Recover any missed jobs from downtime
        _recover_missed_jobs()

        _persist_stats()

        # Startup notification
        job_count = len(scheduler.get_jobs())
        _send_scheduler_notification(
            "Scheduler Started",
            f"Agent scheduler online (PID {os.getpid()}, {job_count} jobs).",
        )

        try:
            scheduler.start()

            if args.daemon:
                while not shutdown_event.is_set():
                    shutdown_event.wait(timeout=60)
        except (KeyboardInterrupt, SystemExit):
            shutdown_event.set()

        # Single shutdown path — no matter how we got here
        logger.info("Shutting down scheduler...")
        _send_scheduler_notification("Scheduler Stopping", "Agent scheduler shutting down.")
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass  # already stopped, or never started
        _persist_stats()
        remove_pid()
        logger.info("Agent scheduler exited cleanly")
        sys.exit(0)


if __name__ == '__main__':
    main()
