#!/usr/bin/env python3
"""
Automated Model Retraining Pipeline with APScheduler

This production-grade retraining system:
1. Runs full retraining every 7 days (Sundays at 2 AM)
2. Performs incremental meta-learner updates every 3 days
3. Detects drift and triggers emergency retraining
4. Validates new models before deployment
5. Sends alerts on failures or performance degradation

Usage:
    python3 scheduled_retraining.py --start          # Start scheduler (blocking)
    python3 scheduled_retraining.py --daemon         # Run as daemon (background)
    python3 scheduled_retraining.py --full           # Trigger full retrain now
    python3 scheduled_retraining.py --incremental    # Trigger incremental update now
    python3 scheduled_retraining.py --status         # Check scheduler status
    python3 scheduled_retraining.py --stop           # Stop daemon

Environment Variables:
    ALERT_EMAIL: Email for critical alerts (optional)
    SLACK_WEBHOOK: Slack webhook URL for alerts (optional)
    MAX_TRAINING_TIME: Max training time in seconds (default: 7200)

For Railway deployment, add to railway.toml:
    [[services]]
    name = "retraining-scheduler"
    command = "python3 scheduled_retraining.py --daemon"
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
import logging
import traceback
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import signal
import atexit

# Try to import APScheduler
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False
    print("ERROR: APScheduler not installed. Install with: pip install apscheduler")
    sys.exit(1)

# Configuration
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"
DATA_DIR = PROJECT_DIR / "data" / "balldontlie_cache"
BACKTEST_RESULTS = PROJECT_DIR / "backtest_results"
RETRAIN_LOG = LOGS_DIR / "retrain_history.json"
PID_FILE = LOGS_DIR / "scheduler.pid"

# Training scripts
FULL_TRAIN_SCRIPT = PROJECT_DIR / "train_complete_balldontlie.py"
INCREMENTAL_TRAIN_SCRIPT = PROJECT_DIR / "train_stacking_model.py"
BACKTEST_SCRIPT = PROJECT_DIR / "comprehensive_backtest.py"

# Thresholds
MAX_TRAINING_TIME = int(os.getenv('MAX_TRAINING_TIME', 7200))  # 2 hours
MIN_DAYS_BETWEEN_FULL_RETRAIN = 14
MIN_DAYS_BETWEEN_INCREMENTAL = 3
PERFORMANCE_DEGRADATION_THRESHOLD = 0.05  # 5% RMSE increase triggers alert
R2_CRITICAL_THRESHOLD = -0.5  # R² below this is critical

# Setup logging
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_retrain_history() -> list[dict]:
    """Load retraining history from JSON log."""
    if RETRAIN_LOG.exists():
        try:
            with open(RETRAIN_LOG) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load retrain history: {e}")
    return []


def save_retrain_record(retrain_type: str, success: bool,
                        duration_seconds: float, metrics: dict = None,
                        error_message: str = None):
    """Save a record of this retrain attempt."""
    record = {
        'timestamp': datetime.now().isoformat(),
        'type': retrain_type,  # 'full' or 'incremental'
        'success': success,
        'duration_seconds': round(duration_seconds, 2),
        'game_count': count_cached_games(),
    }

    if metrics:
        record['metrics'] = metrics

    if error_message:
        record['error'] = error_message

    # Load existing history
    history = get_retrain_history()
    history.append(record)

    # Keep last 100 records
    history = history[-100:]

    with open(RETRAIN_LOG, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Retrain record saved: {retrain_type}, success={success}")


def get_last_retrain_info(retrain_type: str = None) -> dict | None:
    """Get info about the last retrain of given type."""
    history = get_retrain_history()

    if not history:
        return None

    # Filter by type if specified
    if retrain_type:
        filtered = [r for r in history if r.get('type') == retrain_type and r.get('success')]
        return filtered[-1] if filtered else None

    return history[-1]


def count_cached_games() -> int:
    """Count total games in the cache."""
    game_files = list(DATA_DIR.glob("games_*.json"))
    total = 0
    for f in game_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    total += len(data)
                elif isinstance(data, dict) and 'data' in data:
                    total += len(data['data'])
        except Exception:
            pass
    return total


def get_latest_backtest_metrics() -> dict:
    """Get metrics from the most recent backtest."""
    if not BACKTEST_RESULTS.exists():
        return {}

    # Find most recent backtest JSON
    backtest_files = sorted(BACKTEST_RESULTS.glob("*.json"),
                           key=lambda x: x.stat().st_mtime, reverse=True)

    if not backtest_files:
        return {}

    try:
        with open(backtest_files[0]) as f:
            results = json.load(f)
            return {
                'overall_rmse': results.get('overall', {}).get('rmse', 0),
                'overall_r2': results.get('overall', {}).get('r2', 0),
                'roi': results.get('betting', {}).get('roi', 0),
                'win_rate': results.get('betting', {}).get('win_rate', 0),
            }
    except Exception as e:
        logger.warning(f"Failed to load backtest metrics: {e}")
        return {}


def send_alert(subject: str, message: str, severity: str = 'info'):
    """Send alert via shared notification module (email + Slack)."""
    try:
        from agents.core.notifications import send_alert as _send_alert
        _send_alert(subject, message, severity)
    except ImportError:
        # Fallback if agents module not on path (e.g. standalone execution)
        logger.log(
            logging.CRITICAL if severity == 'critical' else
            logging.ERROR if severity == 'error' else
            logging.WARNING if severity == 'warning' else logging.INFO,
            f"ALERT [{severity.upper()}]: {subject}\n{message}"
        )


def check_drift_status() -> dict:
    """Check if drift detector indicates retraining is needed."""
    try:
        from continuous_learning.drift_detector import DriftDetector

        detector = DriftDetector()
        recommendation = detector.should_retrain(lookback_days=7)

        return {
            'should_retrain': recommendation['should_retrain'],
            'urgency': recommendation['urgency'],
            'reasons': recommendation.get('reasons', []),
            'drift_score': recommendation.get('drift_score', 0),
        }
    except Exception as e:
        logger.warning(f"Drift detection failed: {e}")
        return {'should_retrain': False, 'urgency': 'none', 'reasons': [], 'drift_score': 0}


# ============================================================================
# CORE RETRAINING FUNCTIONS
# ============================================================================

def fetch_new_data() -> bool:
    """Fetch latest game data from Balldontlie API."""
    logger.info("Fetching new game data from Balldontlie...")

    try:
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from balldontlie_api import BalldontlieAPI
from datetime import datetime, timedelta

# Initialize API client (uses BALLDONTLIE_API_KEY env var)
api = BalldontlieAPI()

# Fetch games from last 14 days
end_date = datetime.now()
start_date = end_date - timedelta(days=14)

# Generate list of dates for the range
dates = []
current = start_date
while current <= end_date:
    dates.append(current.strftime('%Y-%m-%d'))
    current += timedelta(days=1)

# BalldontlieAPI.get_games() accepts dates parameter
games = api.get_games(dates=dates)
print(f"Fetched {len(games)} games from last 14 days")
"""],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=180
        )

        if result.returncode == 0:
            logger.info(result.stdout.strip() if result.stdout else "Data fetch complete")
            return True
        logger.error(f"Data fetch failed: {result.stderr[:200]}")
        return False

    except subprocess.TimeoutExpired:
        logger.error("Data fetch timed out after 3 minutes")
        return False
    except Exception as e:
        logger.error(f"Data fetch exception: {e}")
        return False


def full_retrain() -> bool:
    """Execute full model retraining.

    Returns:
        True if successful, False otherwise
    """
    logger.info("="*60)
    logger.info("STARTING FULL MODEL RETRAINING")
    logger.info("="*60)

    start_time = datetime.now()

    try:
        # Step 1: Fetch latest data
        if not fetch_new_data():
            logger.warning("Data fetch failed, continuing with cached data")

        # Step 2: Backup existing models
        backup_dir = MODELS_DIR / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(exist_ok=True)

        for model_file in MODELS_DIR.glob("*.pkl"):
            shutil.copy2(model_file, backup_dir / model_file.name)

        logger.info(f"Backed up {len(list(backup_dir.glob('*.pkl')))} models to {backup_dir}")

        # Step 3: Run training script
        logger.info(f"Running training script: {FULL_TRAIN_SCRIPT}")

        result = subprocess.run(
            [sys.executable, str(FULL_TRAIN_SCRIPT)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=MAX_TRAINING_TIME
        )

        if result.returncode != 0:
            logger.error(f"Training failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr[:500]}")

            # Restore backup
            logger.info("Restoring backup models...")
            for backup_file in backup_dir.glob("*.pkl"):
                shutil.copy2(backup_file, MODELS_DIR / backup_file.name)

            send_alert(
                "Full Retraining Failed",
                f"Training script failed. Models restored from backup.\nError: {result.stderr[:200]}",
                severity='error'
            )
            return False

        logger.info("Training completed successfully")

        # Step 4: Run validation backtest
        logger.info("Running validation backtest...")

        old_metrics = get_latest_backtest_metrics()

        subprocess.run(
            [sys.executable, str(BACKTEST_SCRIPT)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )

        new_metrics = get_latest_backtest_metrics()

        # Generate HTML report from backtest results
        try:
            logger.info("Generating HTML backtest report...")
            from report_generator import generate_html_report

            # Find most recent backtest JSON
            backtest_files = sorted(BACKTEST_RESULTS.glob("*.json"),
                                  key=lambda p: p.stat().st_mtime, reverse=True)
            if backtest_files:
                report_path = generate_html_report(str(backtest_files[0]))
                logger.info(f"Report generated: {report_path}")
            else:
                logger.warning("No backtest results found to generate report")
        except Exception as e:
            logger.warning(f"Failed to generate HTML report (non-critical): {e}")

        # Step 5: Validate performance
        if old_metrics and new_metrics:
            old_rmse = old_metrics.get('overall_rmse', 10)
            new_rmse = new_metrics.get('overall_rmse', 10)

            degradation = (new_rmse - old_rmse) / old_rmse if old_rmse > 0 else 0

            if degradation > PERFORMANCE_DEGRADATION_THRESHOLD:
                logger.error(f"Performance degraded by {degradation*100:.1f}%! RMSE: {old_rmse:.3f} → {new_rmse:.3f}")

                # Restore backup
                logger.info("Restoring backup models due to performance degradation...")
                for backup_file in backup_dir.glob("*.pkl"):
                    shutil.copy2(backup_file, MODELS_DIR / backup_file.name)

                send_alert(
                    "Model Performance Degradation Detected",
                    f"New models worse than old by {degradation*100:.1f}%.\n"
                    f"Old RMSE: {old_rmse:.3f}, New RMSE: {new_rmse:.3f}\n"
                    f"Models restored from backup.",
                    severity='error'
                )
                return False
            logger.info(f"Performance validated: RMSE {old_rmse:.3f} → {new_rmse:.3f}")

        # Step 6: Save training record
        duration = (datetime.now() - start_time).total_seconds()
        save_retrain_record('full', True, duration, new_metrics)

        logger.info("="*60)
        logger.info(f"FULL RETRAINING COMPLETE (took {duration/60:.1f} minutes)")
        logger.info("="*60)

        send_alert(
            "Full Retraining Successful",
            f"Models retrained and validated.\n"
            f"Duration: {duration/60:.1f} minutes\n"
            f"RMSE: {new_metrics.get('overall_rmse', 0):.3f}\n"
            f"R²: {new_metrics.get('overall_r2', 0):.3f}\n"
            f"ROI: {new_metrics.get('roi', 0):.2%}",
            severity='info'
        )

        return True

    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out after {MAX_TRAINING_TIME/60:.1f} minutes")
        send_alert("Training Timeout", f"Training exceeded {MAX_TRAINING_TIME/60:.1f} min limit", severity='error')
        return False

    except Exception as e:
        logger.error(f"Full retraining exception: {e}")
        logger.error(traceback.format_exc())
        send_alert("Retraining Exception", f"Exception: {str(e)}\n{traceback.format_exc()[:500]}", severity='error')
        return False


def incremental_update() -> bool:
    """Execute incremental meta-learner update.

    Only retrains the meta-learner with recent data,
    keeping base models unchanged. Much faster than full retrain.

    Returns:
        True if successful, False otherwise
    """
    logger.info("="*60)
    logger.info("STARTING INCREMENTAL META-LEARNER UPDATE")
    logger.info("="*60)

    start_time = datetime.now()

    try:
        # Step 1: Fetch recent data
        if not fetch_new_data():
            logger.warning("Data fetch failed, continuing with cached data")

        # Step 2: Backup meta-learner models
        meta_learner_files = list(MODELS_DIR.glob("*meta_learner*.pkl"))
        if meta_learner_files:
            backup_dir = MODELS_DIR / f"backup_incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(exist_ok=True)

            for model_file in meta_learner_files:
                shutil.copy2(model_file, backup_dir / model_file.name)

            logger.info(f"Backed up {len(meta_learner_files)} meta-learner models")

        # Step 3: Run incremental training (meta-learner only)
        logger.info("Running incremental update (meta-learner only)...")

        result = subprocess.run(
            [sys.executable, str(INCREMENTAL_TRAIN_SCRIPT), '--incremental'],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=900  # 15 minutes
        )

        if result.returncode != 0:
            logger.error(f"Incremental update failed: {result.stderr[:500]}")
            send_alert("Incremental Update Failed", f"Error: {result.stderr[:200]}", severity='warning')
            return False

        logger.info("Incremental update completed")

        # Step 4: Quick validation (last 30 days)
        logger.info("Running quick validation...")

        get_latest_backtest_metrics()

        # Run quick backtest (smaller dataset)
        subprocess.run(
            [sys.executable, str(BACKTEST_SCRIPT), '--quick'],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )

        new_metrics = get_latest_backtest_metrics()

        # Generate HTML report from backtest results
        try:
            logger.info("Generating HTML backtest report...")
            from report_generator import generate_html_report

            # Find most recent backtest JSON
            backtest_files = sorted(BACKTEST_RESULTS.glob("*.json"),
                                  key=lambda p: p.stat().st_mtime, reverse=True)
            if backtest_files:
                report_path = generate_html_report(str(backtest_files[0]))
                logger.info(f"Report generated: {report_path}")
            else:
                logger.warning("No backtest results found to generate report")
        except Exception as e:
            logger.warning(f"Failed to generate HTML report (non-critical): {e}")

        # Step 5: Save record
        duration = (datetime.now() - start_time).total_seconds()
        save_retrain_record('incremental', True, duration, new_metrics)

        logger.info("="*60)
        logger.info(f"INCREMENTAL UPDATE COMPLETE (took {duration/60:.1f} minutes)")
        logger.info("="*60)

        return True

    except subprocess.TimeoutExpired:
        logger.error("Incremental update timed out")
        send_alert("Incremental Update Timeout", "Update exceeded time limit", severity='warning')
        return False

    except Exception as e:
        logger.error(f"Incremental update exception: {e}")
        logger.error(traceback.format_exc())
        return False


def drift_triggered_retrain():
    """Check for drift and trigger retraining if needed."""
    logger.info("Checking for model drift...")

    drift_status = check_drift_status()

    if drift_status['should_retrain']:
        urgency = drift_status['urgency']
        reasons = drift_status['reasons']

        logger.warning(f"Drift detected (urgency: {urgency})")
        logger.warning(f"Reasons: {reasons}")

        if urgency == 'immediate':
            send_alert(
                "CRITICAL: Model Drift Detected",
                "Immediate retraining triggered!\nReasons:\n" + "\n".join(f"- {r}" for r in reasons),
                severity='critical'
            )
            full_retrain()
        else:
            send_alert(
                "Model Drift Detected",
                "Retraining recommended.\nReasons:\n" + "\n".join(f"- {r}" for r in reasons),
                severity='warning'
            )


# ============================================================================
# SCHEDULER SETUP
# ============================================================================

def job_listener(event):
    """Listen to job events for logging."""
    if event.code == EVENT_JOB_EXECUTED:
        logger.info(f"Job '{event.job_id}' executed successfully")
    elif event.code == EVENT_JOB_ERROR:
        logger.error(f"Job '{event.job_id}' failed: {event.exception}")


def create_scheduler(daemon: bool = False):
    """Create and configure the APScheduler.

    Args:
        daemon: If True, use BackgroundScheduler; else BlockingScheduler
    """
    tz = 'America/New_York'
    scheduler = BackgroundScheduler(timezone=tz) if daemon else BlockingScheduler(timezone=tz)

    # Job 1: Full retraining every 7 days (Sundays at 2 AM)
    scheduler.add_job(
        full_retrain,
        CronTrigger(day_of_week='sun', hour=2, minute=0),
        id='full_retrain',
        name='Full Model Retraining (Weekly)',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600  # 1 hour grace period
    )
    logger.info("Scheduled: Full retraining every Sunday at 2:00 AM")

    # Job 2: Incremental update every 3 days at 4 AM
    scheduler.add_job(
        incremental_update,
        IntervalTrigger(days=3, start_date=datetime.now().replace(hour=4, minute=0)),
        id='incremental_update',
        name='Incremental Meta-Learner Update',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=1800  # 30 min grace period
    )
    logger.info("Scheduled: Incremental update every 3 days at 4:00 AM")

    # Job 3: Drift check daily at 6 AM
    scheduler.add_job(
        drift_triggered_retrain,
        CronTrigger(hour=6, minute=0),
        id='drift_check',
        name='Drift Detection & Emergency Retrain',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600  # 1 hour grace period
    )
    logger.info("Scheduled: Drift check daily at 6:00 AM")

    # Add event listener
    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    return scheduler


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
    """Remove PID file."""
    if PID_FILE.exists():
        PID_FILE.unlink()


def get_scheduler_status() -> dict:
    """Get status of the scheduler."""
    if not PID_FILE.exists():
        return {'running': False, 'message': 'Scheduler not running'}

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
            return {'running': False, 'message': 'Scheduler not running'}

        # Same boot, same PID → we're checking our own stale file
        if pid == os.getpid():
            logger.info(f"Stale PID file from current process (PID {pid})")
            remove_pid()
            return {'running': False, 'message': 'Scheduler not running'}

        # Same boot, different PID — check if process is alive
        os.kill(pid, 0)
        return {
            'running': True,
            'pid': pid,
            'message': f'Scheduler running (PID: {pid})'
        }
    except (OSError, ValueError):
        remove_pid()
        return {
            'running': False,
            'message': 'Stale PID file (cleaned)'
        }


def stop_scheduler():
    """Stop the running scheduler."""
    status = get_scheduler_status()

    if not status['running']:
        print(status['message'])
        return False

    pid = status['pid']
    print(f"Stopping scheduler (PID: {pid})...")

    try:
        os.kill(pid, signal.SIGTERM)
        print("Scheduler stopped")
        remove_pid()
        return True
    except Exception as e:
        print(f"Failed to stop scheduler: {e}")
        return False


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated Model Retraining Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true', help='Start scheduler (blocking)')
    group.add_argument('--daemon', action='store_true', help='Run as daemon (background)')
    group.add_argument('--full', action='store_true', help='Trigger full retrain now')
    group.add_argument('--incremental', action='store_true', help='Trigger incremental update now')
    group.add_argument('--status', action='store_true', help='Check scheduler status')
    group.add_argument('--stop', action='store_true', help='Stop daemon')
    group.add_argument('--history', action='store_true', help='Show retraining history')

    args = parser.parse_args()

    if args.status:
        status = get_scheduler_status()
        print(json.dumps(status, indent=2))

        # Show recent retrain history
        history = get_retrain_history()
        if history:
            print("\nRecent retraining history:")
            for record in history[-5:]:
                status_icon = "✅" if record['success'] else "❌"
                print(f"  {status_icon} {record['timestamp'][:19]} | {record['type']:12} | "
                      f"{record['duration_seconds']/60:.1f}m | "
                      f"RMSE: {record.get('metrics', {}).get('overall_rmse', 0):.3f}")

        sys.exit(0 if status['running'] else 1)

    elif args.stop:
        success = stop_scheduler()
        sys.exit(0 if success else 1)

    elif args.full:
        logger.info("Manual full retraining triggered")
        success = full_retrain()
        sys.exit(0 if success else 1)

    elif args.incremental:
        logger.info("Manual incremental update triggered")
        success = incremental_update()
        sys.exit(0 if success else 1)

    elif args.history:
        history = get_retrain_history()
        print(json.dumps(history, indent=2))
        sys.exit(0)

    elif args.start or args.daemon:
        # Check if already running
        status = get_scheduler_status()
        if status['running']:
            print(f"ERROR: {status['message']}")
            sys.exit(1)

        # Create scheduler
        scheduler = create_scheduler(daemon=args.daemon)

        # Save PID
        save_pid()
        atexit.register(remove_pid)

        # Handle signals
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal, stopping scheduler...")
            scheduler.shutdown(wait=True)
            remove_pid()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start scheduler
        logger.info("="*60)
        logger.info("AUTOMATED RETRAINING PIPELINE STARTED")
        logger.info("="*60)
        logger.info(f"Mode: {'Daemon (background)' if args.daemon else 'Blocking (foreground)'}")
        logger.info(f"PID: {os.getpid()}")
        logger.info("Scheduled jobs:")
        for job in scheduler.get_jobs():
            logger.info(f"  - {job.name}: {job.trigger}")
        logger.info("="*60)

        send_alert(
            "Retraining Pipeline Started",
            f"Automated retraining scheduler is now running.\n"
            f"Mode: {'Daemon' if args.daemon else 'Blocking'}\n"
            f"PID: {os.getpid()}",
            severity='info'
        )

        try:
            scheduler.start()

            if args.daemon:
                # Keep daemon alive
                import time
                while True:
                    time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down scheduler...")
            scheduler.shutdown(wait=True)
            remove_pid()


if __name__ == "__main__":
    main()
