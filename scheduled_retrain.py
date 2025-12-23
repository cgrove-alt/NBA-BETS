#!/usr/bin/env python3
"""
Scheduled Model Retraining Script for NBA Player Props Model

This script can be run manually or scheduled via cron/launchd to:
1. Fetch latest game data from Balldontlie API
2. Retrain all player prop models
3. Run a validation backtest
4. Log results and send notifications

Usage:
    python3 scheduled_retrain.py              # Full retrain
    python3 scheduled_retrain.py --quick      # Quick retrain (no backtest)
    python3 scheduled_retrain.py --check      # Check if retrain is needed

Schedule with cron (run weekly on Sunday at 3am):
    0 3 * * 0 cd /Users/sygrovefamily/NBA\ Betting\ Model && python3 scheduled_retrain.py >> logs/retrain.log 2>&1

Schedule with launchd (macOS):
    See the generated .plist file in this directory
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pickle

# Configuration
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"
DATA_DIR = PROJECT_DIR / "data" / "balldontlie_cache"
BACKTEST_RESULTS = PROJECT_DIR / "backtest_results_2025.json"
RETRAIN_LOG = LOGS_DIR / "retrain_history.json"

# Thresholds
MIN_DAYS_BETWEEN_RETRAINS = 3  # Don't retrain more often than this
MIN_NEW_GAMES_FOR_RETRAIN = 15  # Minimum new games to trigger retrain
R2_DEGRADATION_THRESHOLD = 0.05  # Trigger retrain if R2 drops by this much


def setup_logging():
    """Ensure logs directory exists."""
    LOGS_DIR.mkdir(exist_ok=True)


def log(message: str):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_model_age_days() -> float:
    """Get the age of the newest model in days."""
    model_files = list(MODELS_DIR.glob("player_*_ensemble.pkl"))
    if not model_files:
        return float('inf')

    newest_mtime = max(f.stat().st_mtime for f in model_files)
    age_seconds = datetime.now().timestamp() - newest_mtime
    return age_seconds / 86400  # Convert to days


def get_last_retrain_info() -> dict:
    """Get info about the last retrain."""
    if RETRAIN_LOG.exists():
        with open(RETRAIN_LOG, 'r') as f:
            history = json.load(f)
            if history:
                return history[-1]
    return {}


def count_cached_games() -> int:
    """Count total games in the cache."""
    game_files = list(DATA_DIR.glob("games_*.json"))
    total = 0
    for f in game_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    total += len(data)
                elif isinstance(data, dict) and 'data' in data:
                    total += len(data['data'])
        except:
            pass
    return total


def get_current_backtest_r2() -> float:
    """Get the current overall R2 from backtest results."""
    if BACKTEST_RESULTS.exists():
        with open(BACKTEST_RESULTS, 'r') as f:
            results = json.load(f)
            return results.get('overall', {}).get('r2', 0)
    return 0


def should_retrain(force: bool = False) -> tuple[bool, str]:
    """
    Determine if retraining is needed.

    Returns:
        (should_retrain, reason)
    """
    if force:
        return True, "Forced retrain requested"

    # Check model age
    model_age = get_model_age_days()
    if model_age > 7:
        return True, f"Models are {model_age:.1f} days old (>7 days)"

    # Check minimum time between retrains
    last_retrain = get_last_retrain_info()
    if last_retrain:
        last_date = datetime.fromisoformat(last_retrain.get('timestamp', '2000-01-01'))
        days_since = (datetime.now() - last_date).days
        if days_since < MIN_DAYS_BETWEEN_RETRAINS:
            return False, f"Only {days_since} days since last retrain (min: {MIN_DAYS_BETWEEN_RETRAINS})"

        # Check for new games
        last_game_count = last_retrain.get('game_count', 0)
        current_game_count = count_cached_games()
        new_games = current_game_count - last_game_count
        if new_games >= MIN_NEW_GAMES_FOR_RETRAIN:
            return True, f"{new_games} new games since last retrain"

    # Check for R2 degradation (would need live monitoring)
    # This is a placeholder - in production you'd track live accuracy

    return False, "No retrain needed"


def fetch_new_data():
    """Fetch latest game data from Balldontlie API."""
    log("Fetching new game data...")

    # Import and run the data fetcher
    try:
        # Run a quick fetch of recent games
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from balldontlie_client import BalldontlieClient
from datetime import datetime, timedelta

client = BalldontlieClient()

# Fetch games from last 7 days
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

games = client.get_games(
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d')
)
print(f"Fetched {len(games)} games from last 7 days")
"""],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )
        log(result.stdout.strip() if result.stdout else "Data fetch complete")
        if result.returncode != 0 and result.stderr:
            log(f"Warning: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        log("Warning: Data fetch timed out")
    except Exception as e:
        log(f"Warning: Data fetch failed: {e}")


def run_training():
    """Run the model training script."""
    log("Starting model training...")

    train_script = PROJECT_DIR / "train_complete_balldontlie.py"
    if not train_script.exists():
        log("ERROR: Training script not found!")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        # Check for success indicators in output
        if "Training complete" in result.stdout or result.returncode == 0:
            log("Training completed successfully")
            # Log last few lines of output
            lines = result.stdout.strip().split('\n')[-10:]
            for line in lines:
                if line.strip():
                    log(f"  {line}")
            return True
        else:
            log(f"Training may have failed. Return code: {result.returncode}")
            if result.stderr:
                log(f"Errors: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        log("ERROR: Training timed out after 30 minutes")
        return False
    except Exception as e:
        log(f"ERROR: Training failed: {e}")
        return False


def run_backtest():
    """Run backtest to validate new models."""
    log("Running validation backtest...")

    backtest_script = PROJECT_DIR / "comprehensive_backtest.py"
    if not backtest_script.exists():
        log("Warning: Backtest script not found, skipping validation")
        return None

    try:
        result = subprocess.run(
            [sys.executable, str(backtest_script)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Read results
        if BACKTEST_RESULTS.exists():
            with open(BACKTEST_RESULTS, 'r') as f:
                results = json.load(f)
                r2 = results.get('overall', {}).get('r2', 0)
                log(f"Backtest complete. Overall R2: {r2:.3f}")
                return results

    except subprocess.TimeoutExpired:
        log("Warning: Backtest timed out")
    except Exception as e:
        log(f"Warning: Backtest failed: {e}")

    return None


def save_retrain_record(success: bool, backtest_results: dict = None):
    """Save a record of this retrain attempt."""
    record = {
        'timestamp': datetime.now().isoformat(),
        'success': success,
        'game_count': count_cached_games(),
        'model_age_days': get_model_age_days(),
    }

    if backtest_results:
        record['backtest_r2'] = backtest_results.get('overall', {}).get('r2', 0)
        record['backtest_rmse'] = backtest_results.get('overall', {}).get('rmse', 0)

    # Load existing history
    history = []
    if RETRAIN_LOG.exists():
        with open(RETRAIN_LOG, 'r') as f:
            history = json.load(f)

    # Append and save (keep last 50 records)
    history.append(record)
    history = history[-50:]

    with open(RETRAIN_LOG, 'w') as f:
        json.dump(history, f, indent=2)

    log(f"Retrain record saved to {RETRAIN_LOG}")


def main():
    parser = argparse.ArgumentParser(description="NBA Model Scheduled Retraining")
    parser.add_argument('--quick', action='store_true', help='Skip backtest validation')
    parser.add_argument('--check', action='store_true', help='Only check if retrain is needed')
    parser.add_argument('--force', action='store_true', help='Force retrain regardless of schedule')
    args = parser.parse_args()

    setup_logging()

    log("=" * 60)
    log("NBA Player Props Model - Scheduled Retrain")
    log("=" * 60)

    # Check if retrain is needed
    should, reason = should_retrain(force=args.force)
    log(f"Retrain check: {reason}")

    if args.check:
        log(f"Retrain needed: {should}")
        sys.exit(0 if not should else 1)

    if not should and not args.force:
        log("Skipping retrain - not needed")
        sys.exit(0)

    # Fetch new data
    fetch_new_data()

    # Run training
    training_success = run_training()

    if not training_success:
        log("Training failed!")
        save_retrain_record(success=False)
        sys.exit(1)

    # Run backtest (unless --quick)
    backtest_results = None
    if not args.quick:
        backtest_results = run_backtest()

    # Save record
    save_retrain_record(success=True, backtest_results=backtest_results)

    log("=" * 60)
    log("Retrain complete!")
    log("=" * 60)


if __name__ == "__main__":
    main()
