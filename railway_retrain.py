#!/usr/bin/env python3
"""
Railway Cron Job entrypoint for model retraining.
Designed to run as a Railway Cron service (weekly).

Usage:
  1. Create a new Railway Cron service in your project dashboard
  2. Set the schedule: 0 8 * * 0 (Every Sunday at 8 AM UTC / 3 AM EST)
  3. Set the command: python railway_retrain.py
  4. Copy environment variables from the main web service
"""
import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path


def log(message: str):
    """Print with timestamp."""
    print(f"[{datetime.now().isoformat()}] {message}")


def ensure_logs_directory():
    """Create logs directory if it doesn't exist."""
    Path("logs").mkdir(exist_ok=True)


def save_retrain_result(success: bool, duration_seconds: float, error: str = None):
    """Save retrain result to history file."""
    history_file = Path("logs/retrain_history.json")

    # Load existing history
    history = []
    if history_file.exists():
        try:
            with open(history_file) as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    # Add new entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "duration_seconds": round(duration_seconds, 1),
        "trigger": "railway_cron",
    }
    if error:
        entry["error"] = error[:500]  # Truncate long errors

    history.append(entry)

    # Keep only last 50 entries
    history = history[-50:]

    # Save
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    log(f"Saved retrain history to {history_file}")


def main():
    log("Starting Railway scheduled retrain...")
    log(f"Python version: {sys.version}")
    log(f"Working directory: {os.getcwd()}")

    ensure_logs_directory()

    start_time = datetime.now()

    # Check if scheduled_retrain.py exists
    retrain_script = Path("scheduled_retrain.py")
    if not retrain_script.exists():
        log("ERROR: scheduled_retrain.py not found!")
        save_retrain_result(False, 0, "scheduled_retrain.py not found")
        sys.exit(1)

    # Run the scheduled retrain script
    # --quick: Skip backtesting to reduce runtime
    # --save-models: Ensure models are saved
    try:
        log("Executing: python scheduled_retrain.py --quick")
        result = subprocess.run(
            [sys.executable, "scheduled_retrain.py", "--quick"],
            capture_output=True,
            text=True,
            timeout=2700  # 45 minute timeout (training can take 30+ minutes)
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Print output
        if result.stdout:
            print("=== STDOUT ===")
            print(result.stdout)

        if result.stderr:
            print("=== STDERR ===", file=sys.stderr)
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            log(f"ERROR: Retrain failed with exit code {result.returncode}")
            save_retrain_result(False, duration, f"Exit code: {result.returncode}")
            sys.exit(1)

        log(f"Retrain completed successfully in {duration:.1f} seconds")
        save_retrain_result(True, duration)

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start_time).total_seconds()
        log("ERROR: Retrain timed out after 45 minutes")
        save_retrain_result(False, duration, "Timeout after 45 minutes")
        sys.exit(1)

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        log(f"ERROR: Unexpected error: {e}")
        save_retrain_result(False, duration, str(e))
        sys.exit(1)

    log("Railway retrain job complete!")


if __name__ == "__main__":
    main()
