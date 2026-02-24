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
from pathlib import Path
from datetime import datetime

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
        {'hour': '11', 'minute': '30'},
        '11:30 AM ET — prediction pipeline',
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
        }


def _persist_stats():
    """Write current stats to disk."""
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(_stats, f, indent=2, default=str)
    except Exception:
        pass  # non-critical


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
            logger.info(f"[{agent_name}] Completed in {duration:.1f}s")
        else:
            agent_stats['last_status'] = 'failed'
            agent_stats['failures'] = agent_stats.get('failures', 0) + 1
            _stats['total_failures'] += 1
            logger.error(f"[{agent_name}] Failed (exit code {exit_code}) after {duration:.1f}s")

        agent_stats['last_duration_s'] = round(duration, 1)

    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        agent_stats['last_status'] = 'error'
        agent_stats['last_duration_s'] = round(duration, 1)
        agent_stats['failures'] = agent_stats.get('failures', 0) + 1
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
        scheduler.add_job(
            run_agent_job,
            CronTrigger(**cron_kwargs),
            args=[agent_name],
            id=f'agent_{agent_name}',
            name=f'{agent_name}: {description}',
            max_instances=1,
            coalesce=True,
            misfire_grace_time=600,  # 10 min grace
        )
        logger.info(f"Scheduled: {agent_name} — {description}")

    # Job event listener
    def job_listener(event):
        job_id = event.job_id.replace('agent_', '')
        if event.code == EVENT_JOB_EXECUTED:
            logger.debug(f"Job '{job_id}' executed successfully")
        elif event.code == EVENT_JOB_ERROR:
            logger.error(f"Job '{job_id}' raised exception: {event.exception}")

    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    return scheduler


# ============================================================================
# PID management
# ============================================================================

def save_pid():
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))


def remove_pid():
    if PID_FILE.exists():
        PID_FILE.unlink()


def get_status() -> dict:
    """Get scheduler status (from PID file + persisted stats)."""
    result = {'running': False, 'pid': None, 'message': 'Scheduler not running'}

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)  # check if alive
            result = {'running': True, 'pid': pid, 'message': f'Running (PID {pid})'}
        except (OSError, ValueError):
            result = {'running': False, 'pid': None, 'message': 'Stale PID file'}

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
        pid = int(PID_FILE.read_text().strip())
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

        def signal_handler(signum, frame):
            logger.info("Received shutdown signal, stopping scheduler...")
            scheduler.shutdown(wait=True)
            _persist_stats()
            remove_pid()
            sys.exit(0)

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

        _persist_stats()

        try:
            scheduler.start()

            if args.daemon:
                while True:
                    time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down scheduler...")
            scheduler.shutdown(wait=True)
            _persist_stats()
            remove_pid()


if __name__ == '__main__':
    main()
