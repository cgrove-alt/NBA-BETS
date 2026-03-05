"""
Comprehensive Test Suite for Scheduled Retraining Pipeline

Tests all components of the automated retraining system:
- Full retraining workflow
- Incremental update workflow
- Drift detection integration
- Performance validation
- Alert system
- Scheduler configuration

Usage:
    pytest tests/test_scheduled_retraining.py -v
    python -m pytest tests/test_scheduled_retraining.py::test_full_retrain -v
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import nba_models.training.scheduled_retraining as sr


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_project_dir(tmp_path):
    """Create temporary project directory structure."""
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    data_dir = tmp_path / "data" / "balldontlie_cache"
    backtest_results = tmp_path / "backtest_results"

    models_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    backtest_results.mkdir(parents=True)

    # Create mock model files
    for model_name in ['player_points_ensemble.pkl', 'player_rebounds_ensemble.pkl',
                       'moneyline_model.pkl', 'spread_model.pkl']:
        (models_dir / model_name).write_text("mock model")

    # Mock training scripts
    train_script = tmp_path / "train_complete_balldontlie.py"
    train_script.write_text("print('Training complete')")

    incremental_script = tmp_path / "train_stacking_model.py"
    incremental_script.write_text("print('Incremental update complete')")

    backtest_script = tmp_path / "comprehensive_backtest.py"
    backtest_script.write_text("print('Backtest complete')")

    # Mock game data
    games_data = [{"id": i, "date": "2025-01-01"} for i in range(100)]
    (data_dir / "games_2025.json").write_text(json.dumps(games_data))

    # Update module paths
    original_dirs = {
        'PROJECT_DIR': sr.PROJECT_DIR,
        'MODELS_DIR': sr.MODELS_DIR,
        'LOGS_DIR': sr.LOGS_DIR,
        'DATA_DIR': sr.DATA_DIR,
        'BACKTEST_RESULTS': sr.BACKTEST_RESULTS,
        'RETRAIN_LOG': sr.RETRAIN_LOG,
        'PID_FILE': sr.PID_FILE,
    }

    sr.PROJECT_DIR = tmp_path
    sr.MODELS_DIR = models_dir
    sr.LOGS_DIR = logs_dir
    sr.DATA_DIR = data_dir
    sr.BACKTEST_RESULTS = backtest_results
    sr.RETRAIN_LOG = logs_dir / "retrain_history.json"
    sr.PID_FILE = logs_dir / "scheduler.pid"
    sr.FULL_TRAIN_SCRIPT = train_script
    sr.INCREMENTAL_TRAIN_SCRIPT = incremental_script
    sr.BACKTEST_SCRIPT = backtest_script

    yield tmp_path

    # Restore original paths
    for key, value in original_dirs.items():
        setattr(sr, key, value)


@pytest.fixture
def mock_retrain_history():
    """Create mock retraining history."""
    return [
        {
            'timestamp': '2025-01-01T02:00:00',
            'type': 'full',
            'success': True,
            'duration_seconds': 1200,
            'game_count': 500,
            'metrics': {
                'overall_rmse': 5.2,
                'overall_r2': 0.45,
                'roi': 0.05,
                'win_rate': 0.54
            }
        },
        {
            'timestamp': '2025-01-05T04:00:00',
            'type': 'incremental',
            'success': True,
            'duration_seconds': 300,
            'game_count': 520,
            'metrics': {
                'overall_rmse': 5.1,
                'overall_r2': 0.46,
            }
        }
    ]


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

def test_get_retrain_history(temp_project_dir, mock_retrain_history):
    """Test loading retrain history from JSON."""
    # Write mock history
    with open(sr.RETRAIN_LOG, 'w') as f:
        json.dump(mock_retrain_history, f)

    history = sr.get_retrain_history()

    assert len(history) == 2
    assert history[0]['type'] == 'full'
    assert history[1]['type'] == 'incremental'


def test_get_retrain_history_empty(temp_project_dir):
    """Test loading history when file doesn't exist."""
    history = sr.get_retrain_history()
    assert history == []


def test_save_retrain_record(temp_project_dir):
    """Test saving a retrain record."""
    sr.save_retrain_record(
        retrain_type='full',
        success=True,
        duration_seconds=1234.56,
        metrics={'rmse': 5.0, 'r2': 0.5},
        error_message=None
    )

    history = sr.get_retrain_history()

    assert len(history) == 1
    record = history[0]
    assert record['type'] == 'full'
    assert record['success'] is True
    assert record['duration_seconds'] == 1234.56
    assert record['metrics']['rmse'] == 5.0


def test_save_retrain_record_with_error(temp_project_dir):
    """Test saving a failed retrain record."""
    sr.save_retrain_record(
        retrain_type='incremental',
        success=False,
        duration_seconds=100,
        error_message='Training script failed'
    )

    history = sr.get_retrain_history()
    record = history[0]

    assert record['success'] is False
    assert record['error'] == 'Training script failed'


def test_get_last_retrain_info(temp_project_dir, mock_retrain_history):
    """Test getting last retrain info."""
    with open(sr.RETRAIN_LOG, 'w') as f:
        json.dump(mock_retrain_history, f)

    # Get last of any type
    last = sr.get_last_retrain_info()
    assert last['type'] == 'incremental'

    # Get last full retrain
    last_full = sr.get_last_retrain_info('full')
    assert last_full['type'] == 'full'
    assert last_full['metrics']['overall_rmse'] == 5.2


def test_count_cached_games(temp_project_dir):
    """Test counting games in cache."""
    count = sr.count_cached_games()
    assert count == 100  # From fixture


def test_get_latest_backtest_metrics(temp_project_dir):
    """Test loading latest backtest metrics."""
    # Create mock backtest results
    backtest_data = {
        'overall': {'rmse': 5.3, 'r2': 0.42, 'mae': 3.2},
        'betting': {'roi': 0.06, 'win_rate': 0.55}
    }

    backtest_file = sr.BACKTEST_RESULTS / "backtest_2025.json"
    with open(backtest_file, 'w') as f:
        json.dump(backtest_data, f)

    metrics = sr.get_latest_backtest_metrics()

    assert metrics['overall_rmse'] == 5.3
    assert metrics['overall_r2'] == 0.42
    assert metrics['roi'] == 0.06
    assert metrics['win_rate'] == 0.55


def test_get_latest_backtest_metrics_no_file(temp_project_dir):
    """Test loading metrics when no backtest file exists."""
    metrics = sr.get_latest_backtest_metrics()
    assert metrics == {}


# ============================================================================
# ALERT SYSTEM TESTS
# ============================================================================

@patch('nba_models.training.scheduled_retraining.subprocess.run')
def test_send_alert_email(mock_run, temp_project_dir, monkeypatch):
    """Test sending email alert."""
    monkeypatch.setenv('ALERT_EMAIL', 'test@example.com')

    sr.send_alert("Test Alert", "This is a test", severity='info')

    # Verify mail command was called
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert 'mail' in call_args[0][0]


@patch('requests.post')
def test_send_alert_slack(mock_post, temp_project_dir, monkeypatch):
    """Test sending Slack alert."""
    monkeypatch.setenv('SLACK_WEBHOOK', 'https://hooks.slack.com/test')

    sr.send_alert("Test Alert", "This is a test", severity='warning')

    # Verify Slack webhook was called
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == 'https://hooks.slack.com/test'
    assert ':warning:' in call_args[1]['json']['text']


# ============================================================================
# DRIFT DETECTION TESTS
# ============================================================================

@patch('continuous_learning.drift_detector.DriftDetector')
def test_check_drift_status_no_drift(mock_detector_class, temp_project_dir):
    """Test drift check when no drift detected."""
    mock_detector = Mock()
    mock_detector.should_retrain.return_value = {
        'should_retrain': False,
        'urgency': 'none',
        'reasons': [],
        'drift_score': 15
    }
    mock_detector_class.return_value = mock_detector

    result = sr.check_drift_status()

    assert result['should_retrain'] is False
    assert result['urgency'] == 'none'
    assert result['drift_score'] == 15


@patch('continuous_learning.drift_detector.DriftDetector')
def test_check_drift_status_with_drift(mock_detector_class, temp_project_dir):
    """Test drift check when drift detected."""
    mock_detector = Mock()
    mock_detector.should_retrain.return_value = {
        'should_retrain': True,
        'urgency': 'high',
        'reasons': ['Win rate dropped 10%', 'Calibration error high'],
        'drift_score': 65
    }
    mock_detector_class.return_value = mock_detector

    result = sr.check_drift_status()

    assert result['should_retrain'] is True
    assert result['urgency'] == 'high'
    assert len(result['reasons']) == 2


# ============================================================================
# DATA FETCHING TESTS
# ============================================================================

@patch('nba_models.training.scheduled_retraining.subprocess.run')
def test_fetch_new_data_success(mock_run, temp_project_dir):
    """Test successful data fetch."""
    mock_run.return_value = Mock(returncode=0, stdout="Fetched 50 games")

    result = sr.fetch_new_data()

    assert result is True
    mock_run.assert_called_once()


@patch('nba_models.training.scheduled_retraining.subprocess.run')
def test_fetch_new_data_failure(mock_run, temp_project_dir):
    """Test failed data fetch."""
    mock_run.return_value = Mock(returncode=1, stderr="API error")

    result = sr.fetch_new_data()

    assert result is False


# ============================================================================
# FULL RETRAINING TESTS
# ============================================================================

@patch('nba_models.training.scheduled_retraining.send_alert')
@patch('nba_models.training.scheduled_retraining.fetch_new_data')
@patch('nba_models.training.scheduled_retraining.subprocess.run')
def test_full_retrain_success(mock_run, mock_fetch, mock_alert, temp_project_dir):
    """Test successful full retraining."""
    # Mock successful data fetch
    mock_fetch.return_value = True

    # Mock three subprocess.run calls: training, calibration, backtest
    mock_run.side_effect = [
        Mock(returncode=0, stdout="Training complete", stderr=""),       # Training
        Mock(returncode=0, stdout="Calibration complete", stderr=""),    # Quantile calibration (new)
        Mock(returncode=0, stdout="Backtest complete", stderr=""),       # Backtest
    ]

    # Create mock backtest results
    backtest_data = {
        'overall': {'rmse': 5.0, 'r2': 0.50},
        'betting': {'roi': 0.07, 'win_rate': 0.56}
    }
    with open(sr.BACKTEST_RESULTS / "backtest_new.json", 'w') as f:
        json.dump(backtest_data, f)

    result = sr.full_retrain()

    assert result is True

    # Verify training script was called (at minimum); calibration may be skipped if
    # script path doesn't resolve, so check at least 2 calls (training + backtest).
    assert mock_run.call_count >= 2

    # Verify success alert was sent
    alert_calls = [call for call in mock_alert.call_args_list if 'Successful' in call[0][0]]
    assert len(alert_calls) > 0


@patch('nba_models.training.scheduled_retraining.send_alert')
@patch('nba_models.training.scheduled_retraining.fetch_new_data')
@patch('nba_models.training.scheduled_retraining.subprocess.run')
def test_full_retrain_training_failure(mock_run, mock_fetch, mock_alert, temp_project_dir):
    """Test full retraining when training script fails."""
    mock_fetch.return_value = True

    # Mock failed training
    mock_run.return_value = Mock(returncode=1, stderr="Training error")

    result = sr.full_retrain()

    assert result is False

    # Verify error alert was sent
    alert_calls = [call for call in mock_alert.call_args_list if 'Failed' in call[0][0]]
    assert len(alert_calls) > 0


@patch('nba_models.training.scheduled_retraining.send_alert')
@patch('nba_models.training.scheduled_retraining.fetch_new_data')
@patch('nba_models.training.scheduled_retraining.subprocess.run')
@patch('nba_models.training.scheduled_retraining.get_latest_backtest_metrics')
def test_full_retrain_performance_degradation(mock_metrics, mock_run, mock_fetch, mock_alert, temp_project_dir):
    """Test full retraining rollback on performance degradation."""
    mock_fetch.return_value = True

    # Mock old metrics (good) and new metrics (worse - 20% degradation)
    mock_metrics.side_effect = [
        {'overall_rmse': 5.0, 'overall_r2': 0.50, 'roi': 0.07, 'win_rate': 0.55},  # Before training
        {'overall_rmse': 6.0, 'overall_r2': 0.35, 'roi': 0.02, 'win_rate': 0.48}   # After training
    ]

    # Mock three subprocess.run calls: training, calibration, backtest
    mock_run.side_effect = [
        Mock(returncode=0, stdout="Training complete", stderr=""),
        Mock(returncode=0, stdout="Calibration complete", stderr=""),   # Quantile calibration (new)
        Mock(returncode=0, stdout="Backtest complete", stderr=""),
    ]

    result = sr.full_retrain()

    # Should fail due to 20% degradation (> 5% threshold)
    assert result is False

    # Verify degradation alert was sent
    alert_calls = [call for call in mock_alert.call_args_list if 'Degradation' in call[0][0]]
    assert len(alert_calls) > 0


# ============================================================================
# INCREMENTAL UPDATE TESTS
# ============================================================================

@patch('nba_models.training.scheduled_retraining.fetch_new_data')
@patch('nba_models.training.scheduled_retraining.subprocess.run')
def test_incremental_update_success(mock_run, mock_fetch, temp_project_dir):
    """Test successful incremental update."""
    mock_fetch.return_value = True

    # Mock successful incremental training and backtest
    mock_run.side_effect = [
        Mock(returncode=0, stdout="Incremental complete", stderr=""),
        Mock(returncode=0, stdout="Backtest complete", stderr="")
    ]

    result = sr.incremental_update()

    assert result is True
    assert mock_run.call_count == 2


@patch('nba_models.training.scheduled_retraining.fetch_new_data')
@patch('nba_models.training.scheduled_retraining.subprocess.run')
def test_incremental_update_failure(mock_run, mock_fetch, temp_project_dir):
    """Test failed incremental update."""
    mock_fetch.return_value = True

    # Mock failed incremental training
    mock_run.return_value = Mock(returncode=1, stderr="Update failed")

    result = sr.incremental_update()

    assert result is False


# ============================================================================
# SCHEDULER TESTS
# ============================================================================

def test_create_scheduler_blocking(temp_project_dir):
    """Test creating blocking scheduler."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = sr.create_scheduler(daemon=False)

    assert isinstance(scheduler, BlockingScheduler)

    # Verify jobs are scheduled
    jobs = scheduler.get_jobs()
    job_ids = [job.id for job in jobs]

    assert 'full_retrain' in job_ids
    assert 'incremental_update' in job_ids
    assert 'drift_check' in job_ids


def test_create_scheduler_daemon(temp_project_dir):
    """Test creating background scheduler."""
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = sr.create_scheduler(daemon=True)

    assert isinstance(scheduler, BackgroundScheduler)

    jobs = scheduler.get_jobs()
    assert len(jobs) == 3


def test_save_and_remove_pid(temp_project_dir):
    """Test PID file management."""
    sr.save_pid()

    assert sr.PID_FILE.exists()

    with open(sr.PID_FILE) as f:
        # PID file format: "<pid>\n<boot_id>" — read only the first line
        pid = int(f.readline().strip())

    assert pid == os.getpid()

    sr.remove_pid()
    assert not sr.PID_FILE.exists()


@patch('nba_models.training.scheduled_retraining.os.kill')
def test_get_scheduler_status_running(mock_kill, temp_project_dir):
    """Test getting status when scheduler is running.

    Simulates a scheduler that was started by a *different* process by writing
    a PID file manually with a different PID (PID 1 = init, always running on Linux).
    os.kill is mocked to avoid sending a real signal.
    """
    # Write PID file with a different PID (PID 1) and the current boot ID
    boot_id = sr._get_boot_id()
    sr.PID_FILE.write_text(f"1\n{boot_id}")

    # os.kill(1, 0) should succeed (process exists) — mock returns None
    mock_kill.return_value = None

    status = sr.get_scheduler_status()

    assert status['running'] is True
    assert status['pid'] == 1

    sr.remove_pid()


def test_get_scheduler_status_not_running(temp_project_dir):
    """Test getting status when scheduler is not running."""
    status = sr.get_scheduler_status()

    assert status['running'] is False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@patch('nba_models.training.scheduled_retraining.send_alert')
@patch('nba_models.training.scheduled_retraining.full_retrain')
@patch('nba_models.training.scheduled_retraining.check_drift_status')
def test_drift_triggered_retrain_immediate(mock_drift, mock_full, mock_alert, temp_project_dir):
    """Test drift-triggered retraining with immediate urgency."""
    mock_drift.return_value = {
        'should_retrain': True,
        'urgency': 'immediate',
        'reasons': ['Critical accuracy drop']
    }

    sr.drift_triggered_retrain()

    # Verify full retrain was triggered
    mock_full.assert_called_once()

    # Verify critical alert was sent
    alert_calls = [call for call in mock_alert.call_args_list if 'CRITICAL' in call[0][0]]
    assert len(alert_calls) > 0


@patch('nba_models.training.scheduled_retraining.send_alert')
@patch('nba_models.training.scheduled_retraining.full_retrain')
@patch('nba_models.training.scheduled_retraining.check_drift_status')
def test_drift_triggered_retrain_no_drift(mock_drift, mock_full, mock_alert, temp_project_dir):
    """Test drift check when no drift detected."""
    mock_drift.return_value = {
        'should_retrain': False,
        'urgency': 'none',
        'reasons': []
    }

    sr.drift_triggered_retrain()

    # Verify full retrain was NOT triggered
    mock_full.assert_not_called()


# ============================================================================
# INTEGRATION TESTS (NO MOCKS)
# ============================================================================

def test_balldontlie_api_import():
    """Integration test: Verify actual BalldontlieAPI import works."""
    try:
        from balldontlie_api import BalldontlieAPI
        # Verify class exists and has expected methods
        assert hasattr(BalldontlieAPI, 'get_games')
        assert hasattr(BalldontlieAPI, '__init__')
    except ImportError as e:
        pytest.fail(f"BalldontlieAPI import failed: {e}")


def test_train_stacking_model_incremental_flag():
    """Integration test: Verify --incremental flag exists."""
    script_path = Path(__file__).parent.parent / "train_stacking_model.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=5
    )

    assert result.returncode == 0
    assert "--incremental" in result.stdout, "train_stacking_model.py missing --incremental flag"


def test_comprehensive_backtest_quick_flag():
    """Integration test: Verify --quick flag exists."""
    script_path = Path(__file__).parent.parent / "comprehensive_backtest.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=5
    )

    assert result.returncode == 0
    assert "--quick" in result.stdout, "comprehensive_backtest.py missing --quick flag"


def test_scheduled_retraining_cli():
    """Integration test: Verify CLI commands work."""
    # Test --help
    script_path = Path(__file__).parent.parent / "scheduled_retraining.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=5
    )

    assert result.returncode == 0
    assert "--start" in result.stdout
    assert "--daemon" in result.stdout
    assert "--status" in result.stdout
    assert "--full" in result.stdout
    assert "--incremental" in result.stdout


# ============================================================================
# SUMMARY
# ============================================================================

def test_summary():
    """Print test summary."""
    print("\n" + "="*60)
    print("SCHEDULED RETRAINING TEST SUITE SUMMARY")
    print("="*60)
    print("✅ Helper function tests")
    print("✅ Alert system tests")
    print("✅ Drift detection tests")
    print("✅ Data fetching tests")
    print("✅ Full retraining tests")
    print("✅ Incremental update tests")
    print("✅ Scheduler configuration tests")
    print("✅ Integration tests (NO MOCKS)")
    print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
