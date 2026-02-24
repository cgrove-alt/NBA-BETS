"""
NBA Odds Tracker Background Service

This service runs as a background job to fetch and store NBA odds at regular
intervals using APScheduler. It is designed to run during NBA game days to
capture line movements in real-time.

Features:
- Automatic odds fetching every 5 minutes during game hours (8 AM - 11 PM)
- NBA season awareness (only runs Oct-Jun)
- Error handling and retry logic
- Comprehensive logging to odds_tracker.log
- Health monitoring and uptime tracking

Usage:
    # Run as standalone service
    python odds_tracker_service.py

    # Or import and control programmatically
    from odds_tracker_service import OddsTrackerService
    service = OddsTrackerService()
    service.start()

Configuration:
- Set THE_ODDS_API_KEY environment variable
- Optionally set ODDS_DB_PATH for custom database location
- Logs written to odds_tracker.log

Dependencies:
- APScheduler 3.10+
- betting_market_features.py (OddsTracker class)
"""

import load_env  # noqa: F401  — load .env before any code reads os.environ
import os
import sys
import logging
import signal
from datetime import datetime, timedelta
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

try:
    from betting_market_features import OddsTracker
except ImportError:
    print("ERROR: betting_market_features.py not found. Please ensure it's in the same directory.")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# NBA Season months (October through June)
NBA_SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6]

# Operating hours (8 AM to 11 PM)
START_HOUR = 8
END_HOUR = 23

# Update interval in minutes
UPDATE_INTERVAL = 5

# Database path
DEFAULT_DB_PATH = "odds_history.db"

# Log file
LOG_FILE = "odds_tracker.log"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 60


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_file: str = LOG_FILE) -> logging.Logger:
    """
    Configure logging to file and console.

    Args:
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("OddsTrackerService")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler with rotation
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# ODDS TRACKER SERVICE
# =============================================================================

class OddsTrackerService:
    """
    Background service to fetch and store NBA odds at regular intervals.

    This service uses APScheduler to run the OddsTracker every 5 minutes
    during NBA season game hours (8 AM - 11 PM, Oct-Jun).
    """

    def __init__(self,
                 api_key: str | None = None,
                 update_interval: int = UPDATE_INTERVAL,
                 db_path: str = DEFAULT_DB_PATH,
                 log_file: str = LOG_FILE):
        """
        Initialize the odds tracker service.

        Args:
            api_key: The Odds API key (or set THE_ODDS_API_KEY env var)
            update_interval: Minutes between updates (default 5)
            db_path: Path to SQLite database
            log_file: Path to log file
        """
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
        self.update_interval = update_interval
        self.db_path = db_path
        self.log_file = log_file

        # Setup logging
        self.logger = setup_logging(log_file)

        # Initialize scheduler
        self.scheduler = BackgroundScheduler(
            timezone='America/New_York',  # NBA Eastern timezone
            job_defaults={
                'coalesce': True,  # Combine missed runs
                'max_instances': 1,  # Only one instance at a time
                'misfire_grace_time': 300  # 5 min grace for misfires
            }
        )

        # Initialize tracker
        self.tracker = OddsTracker(
            api_key=self.api_key,
            update_interval_minutes=update_interval,
            db_path=db_path
        )

        # Health monitoring
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        self.last_success: datetime | None = None
        self.last_failure: datetime | None = None
        self.service_start_time: datetime | None = None

        # Register event listeners
        self.scheduler.add_listener(self._job_executed_listener, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error_listener, EVENT_JOB_ERROR)

        # Validate setup
        self._validate_setup()

    def _validate_setup(self):
        """Validate API key and database connectivity."""
        if not self.api_key:
            self.logger.error("No API key provided. Set THE_ODDS_API_KEY environment variable.")
            raise ValueError("Missing THE_ODDS_API_KEY")

        self.logger.info("Odds Tracker Service initialized")
        self.logger.info(f"Database: {self.db_path}")
        self.logger.info(f"Update interval: {self.update_interval} minutes")
        self.logger.info(f"Operating hours: {START_HOUR}:00 - {END_HOUR}:00 EST")

    def is_nba_season(self) -> bool:
        """
        Check if current month is during NBA season.

        Returns:
            True if current month is Oct-Jun
        """
        current_month = datetime.now().month
        return current_month in NBA_SEASON_MONTHS

    def should_run_now(self) -> bool:
        """
        Check if service should be running now.

        Returns:
            True if it's NBA season and within operating hours
        """
        now = datetime.now()

        # Check season
        if not self.is_nba_season():
            return False

        # Check time of day (8 AM - 11 PM)
        return START_HOUR <= now.hour < END_HOUR

    def fetch_and_store_with_retry(self):
        """
        Fetch and store odds with retry logic.

        This is the main job function that runs on schedule.
        Implements retry logic with exponential backoff.
        """
        # Check if we should run
        if not self.should_run_now():
            self.logger.info("Outside operating hours or NBA season. Skipping fetch.")
            return

        self.total_runs += 1
        retry_count = 0

        while retry_count < MAX_RETRIES:
            try:
                # Fetch and store odds
                count = self.tracker.fetch_and_store_odds()

                # Log success
                self.successful_runs += 1
                self.last_success = datetime.now()

                self.logger.info(f"✓ Stored {count} odds snapshots (attempt {retry_count + 1})")

                # Break on success
                return

            except Exception as e:
                retry_count += 1

                if retry_count < MAX_RETRIES:
                    self.logger.warning(
                        f"Fetch failed (attempt {retry_count}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {RETRY_DELAY_SECONDS}s..."
                    )
                    import time
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    # Max retries reached
                    self.failed_runs += 1
                    self.last_failure = datetime.now()

                    self.logger.error(
                        f"✗ Failed after {MAX_RETRIES} attempts: {e}",
                        exc_info=True
                    )

    def _job_executed_listener(self, event):
        """Listener for successful job execution."""
        pass  # Already logged in fetch_and_store_with_retry

    def _job_error_listener(self, event):
        """Listener for job errors."""
        if event.exception:
            self.logger.error(f"Job error: {event.exception}", exc_info=True)

    def get_health_status(self) -> dict:
        """
        Get current health status of the service.

        Returns:
            Dictionary with health metrics
        """
        uptime = None
        if self.service_start_time:
            uptime = (datetime.now() - self.service_start_time).total_seconds()

        success_rate = 0.0
        if self.total_runs > 0:
            success_rate = (self.successful_runs / self.total_runs) * 100

        return {
            'status': 'running' if self.scheduler.running else 'stopped',
            'uptime_seconds': uptime,
            'total_runs': self.total_runs,
            'successful_runs': self.successful_runs,
            'failed_runs': self.failed_runs,
            'success_rate': f"{success_rate:.1f}%",
            'last_success': self.last_success.isoformat() if self.last_success else None,
            'last_failure': self.last_failure.isoformat() if self.last_failure else None,
            'is_nba_season': self.is_nba_season(),
            'should_run_now': self.should_run_now()
        }

    def print_status(self):
        """Print current health status to console."""
        status = self.get_health_status()

        print("\n" + "=" * 70)
        print("ODDS TRACKER SERVICE STATUS")
        print("=" * 70)

        print(f"Status: {status['status'].upper()}")

        if status['uptime_seconds']:
            uptime_str = str(timedelta(seconds=int(status['uptime_seconds'])))
            print(f"Uptime: {uptime_str}")

        print(f"\nRuns: {status['total_runs']} total "
              f"({status['successful_runs']} ✓, {status['failed_runs']} ✗)")
        print(f"Success Rate: {status['success_rate']}")

        if status['last_success']:
            print(f"Last Success: {status['last_success']}")
        if status['last_failure']:
            print(f"Last Failure: {status['last_failure']}")

        print(f"\nNBA Season: {'Yes' if status['is_nba_season'] else 'No'}")
        print(f"Should Run Now: {'Yes' if status['should_run_now'] else 'No'}")

        print("=" * 70 + "\n")

    def start(self):
        """
        Start the odds tracker service.

        Adds scheduled job and starts the scheduler.
        """
        self.logger.info("Starting Odds Tracker Service...")

        # Add job to run every N minutes during operating hours
        self.scheduler.add_job(
            func=self.fetch_and_store_with_retry,
            trigger=CronTrigger(
                minute=f'*/{self.update_interval}',  # Every N minutes
                hour=f'{START_HOUR}-{END_HOUR-1}',  # 8 AM to 10:59 PM
                month=','.join(map(str, NBA_SEASON_MONTHS))  # Oct-Jun
            ),
            id='odds_tracker_job',
            name='Fetch and Store NBA Odds',
            replace_existing=True
        )

        # Start scheduler
        self.scheduler.start()
        self.service_start_time = datetime.now()

        self.logger.info("✓ Odds Tracker Service started successfully")
        self.logger.info(f"Next run: {self.scheduler.get_jobs()[0].next_run_time}")

        # Run once immediately if we're in operating hours
        if self.should_run_now():
            self.logger.info("Running initial fetch...")
            self.fetch_and_store_with_retry()

    def stop(self):
        """
        Stop the odds tracker service.
        """
        self.logger.info("Stopping Odds Tracker Service...")

        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)

        self.logger.info("✓ Odds Tracker Service stopped")

    def run_blocking(self):
        """
        Start service and block until interrupted.

        Useful for running as a standalone daemon.
        """
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start service
        self.start()

        self.logger.info("Service running. Press Ctrl+C to stop.")
        print("\nOdds Tracker Service is running...")
        print(f"Logs: {Path(self.log_file).absolute()}")
        print("Press Ctrl+C to stop.\n")

        try:
            # Keep alive
            while True:
                import time
                time.sleep(60)  # Wake up every minute

                # Print status every 30 minutes
                if self.total_runs > 0 and self.total_runs % 6 == 0:
                    self.print_status()

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
            self.stop()

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        self.logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """
    Main entry point for running the service from command line.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="NBA Odds Tracker Background Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (5-minute intervals)
  python odds_tracker_service.py

  # Run with custom interval (10 minutes)
  python odds_tracker_service.py --interval 10

  # Use custom database path
  python odds_tracker_service.py --db-path /path/to/odds.db

  # Check status only
  python odds_tracker_service.py --status
        """
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=UPDATE_INTERVAL,
        help=f'Update interval in minutes (default: {UPDATE_INTERVAL})'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default=DEFAULT_DB_PATH,
        help=f'Path to SQLite database (default: {DEFAULT_DB_PATH})'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default=LOG_FILE,
        help=f'Path to log file (default: {LOG_FILE})'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Print status and exit (non-blocking)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run one fetch and exit (for testing)'
    )

    args = parser.parse_args()

    # Initialize service
    try:
        service = OddsTrackerService(
            update_interval=args.interval,
            db_path=args.db_path,
            log_file=args.log_file
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        print("\nPlease set THE_ODDS_API_KEY environment variable:")
        print("  export THE_ODDS_API_KEY='your_api_key_here'")
        sys.exit(1)

    # Handle commands
    if args.status:
        # Status check only
        service.print_status()
        sys.exit(0)

    elif args.test:
        # Test run
        print("Running test fetch...")
        service.fetch_and_store_with_retry()
        service.print_status()
        sys.exit(0)

    else:
        # Start service (blocking)
        service.run_blocking()


if __name__ == "__main__":
    main()
