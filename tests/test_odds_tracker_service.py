"""
Unit Tests for Odds Tracker Service

Tests the APScheduler-based background service for fetching NBA odds.
"""

import os
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from odds_tracker_service import (
    OddsTrackerService,
    NBA_SEASON_MONTHS,
    START_HOUR,
    END_HOUR,
    setup_logging
)


class TestOddsTrackerService(unittest.TestCase):
    """Test suite for OddsTrackerService."""

    def setUp(self):
        """Set up test fixtures."""
        # Use test database
        self.test_db = "test_odds_tracker.db"
        self.test_log = "test_odds_tracker.log"

        # Mock API key
        os.environ['THE_ODDS_API_KEY'] = 'test_api_key_12345'

    def tearDown(self):
        """Clean up test artifacts."""
        # Remove test database
        if Path(self.test_db).exists():
            Path(self.test_db).unlink()

        # Remove test log
        if Path(self.test_log).exists():
            Path(self.test_log).unlink()

    @patch('odds_tracker_service.OddsTracker')
    def test_service_initialization(self, mock_tracker):
        """Test service initializes correctly."""
        service = OddsTrackerService(
            api_key='test_key',
            update_interval=5,
            db_path=self.test_db,
            log_file=self.test_log
        )

        # Verify initialization
        self.assertEqual(service.api_key, 'test_key')
        self.assertEqual(service.update_interval, 5)
        self.assertEqual(service.db_path, self.test_db)
        self.assertEqual(service.total_runs, 0)
        self.assertEqual(service.successful_runs, 0)
        self.assertEqual(service.failed_runs, 0)
        self.assertIsNone(service.service_start_time)

        # Verify scheduler created
        self.assertIsNotNone(service.scheduler)
        self.assertFalse(service.scheduler.running)

    def test_initialization_without_api_key(self):
        """Test service raises error without API key."""
        # Remove API key
        if 'THE_ODDS_API_KEY' in os.environ:
            del os.environ['THE_ODDS_API_KEY']

        with self.assertRaises(ValueError) as context:
            OddsTrackerService(
                db_path=self.test_db,
                log_file=self.test_log
            )

        self.assertIn("THE_ODDS_API_KEY", str(context.exception))

    @patch('odds_tracker_service.OddsTracker')
    def test_is_nba_season(self, mock_tracker):
        """Test NBA season detection."""
        service = OddsTrackerService(
            api_key='test_key',
            db_path=self.test_db,
            log_file=self.test_log
        )

        # Mock current month
        with patch('odds_tracker_service.datetime') as mock_datetime:
            # October (NBA season)
            mock_datetime.now.return_value = datetime(2024, 10, 15, 12, 0)
            self.assertTrue(service.is_nba_season())

            # January (NBA season)
            mock_datetime.now.return_value = datetime(2025, 1, 15, 12, 0)
            self.assertTrue(service.is_nba_season())

            # June (NBA season, playoffs)
            mock_datetime.now.return_value = datetime(2025, 6, 15, 12, 0)
            self.assertTrue(service.is_nba_season())

            # July (offseason)
            mock_datetime.now.return_value = datetime(2025, 7, 15, 12, 0)
            self.assertFalse(service.is_nba_season())

            # August (offseason)
            mock_datetime.now.return_value = datetime(2025, 8, 15, 12, 0)
            self.assertFalse(service.is_nba_season())

    @patch('odds_tracker_service.OddsTracker')
    def test_should_run_now(self, mock_tracker):
        """Test operating hours detection."""
        service = OddsTrackerService(
            api_key='test_key',
            db_path=self.test_db,
            log_file=self.test_log
        )

        with patch('odds_tracker_service.datetime') as mock_datetime:
            # During season, during operating hours (10 AM)
            mock_datetime.now.return_value = datetime(2024, 11, 15, 10, 0)
            self.assertTrue(service.should_run_now())

            # During season, during operating hours (8 PM)
            mock_datetime.now.return_value = datetime(2024, 11, 15, 20, 0)
            self.assertTrue(service.should_run_now())

            # During season, before operating hours (6 AM)
            mock_datetime.now.return_value = datetime(2024, 11, 15, 6, 0)
            self.assertFalse(service.should_run_now())

            # During season, after operating hours (11 PM)
            mock_datetime.now.return_value = datetime(2024, 11, 15, 23, 0)
            self.assertFalse(service.should_run_now())

            # Offseason, during operating hours
            mock_datetime.now.return_value = datetime(2024, 8, 15, 10, 0)
            self.assertFalse(service.should_run_now())

    @patch('odds_tracker_service.OddsTracker')
    def test_fetch_and_store_with_retry_success(self, mock_tracker_class):
        """Test successful fetch and store."""
        # Mock tracker instance
        mock_tracker_instance = Mock()
        mock_tracker_instance.fetch_and_store_odds.return_value = 150
        mock_tracker_class.return_value = mock_tracker_instance

        service = OddsTrackerService(
            api_key='test_key',
            db_path=self.test_db,
            log_file=self.test_log
        )

        # Mock should_run_now to return True
        service.should_run_now = Mock(return_value=True)

        # Call fetch
        service.fetch_and_store_with_retry()

        # Verify tracker was called
        mock_tracker_instance.fetch_and_store_odds.assert_called_once()

        # Verify metrics updated
        self.assertEqual(service.total_runs, 1)
        self.assertEqual(service.successful_runs, 1)
        self.assertEqual(service.failed_runs, 0)
        self.assertIsNotNone(service.last_success)
        self.assertIsNone(service.last_failure)

    @patch('odds_tracker_service.OddsTracker')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_fetch_and_store_with_retry_failure(self, mock_sleep, mock_tracker_class):
        """Test fetch failure with retries."""
        # Mock tracker to raise error
        mock_tracker_instance = Mock()
        mock_tracker_instance.fetch_and_store_odds.side_effect = Exception("API error")
        mock_tracker_class.return_value = mock_tracker_instance

        service = OddsTrackerService(
            api_key='test_key',
            db_path=self.test_db,
            log_file=self.test_log
        )

        # Mock should_run_now to return True
        service.should_run_now = Mock(return_value=True)

        # Call fetch (should retry 3 times)
        service.fetch_and_store_with_retry()

        # Verify retries
        self.assertEqual(mock_tracker_instance.fetch_and_store_odds.call_count, 3)

        # Verify metrics updated
        self.assertEqual(service.total_runs, 1)
        self.assertEqual(service.successful_runs, 0)
        self.assertEqual(service.failed_runs, 1)
        self.assertIsNone(service.last_success)
        self.assertIsNotNone(service.last_failure)

    @patch('odds_tracker_service.OddsTracker')
    @patch('time.sleep')
    def test_fetch_and_store_with_retry_eventual_success(self, mock_sleep, mock_tracker_class):
        """Test fetch succeeds after initial failures."""
        # Mock tracker to fail twice, then succeed
        mock_tracker_instance = Mock()
        mock_tracker_instance.fetch_and_store_odds.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            100  # Success on third try
        ]
        mock_tracker_class.return_value = mock_tracker_instance

        service = OddsTrackerService(
            api_key='test_key',
            db_path=self.test_db,
            log_file=self.test_log
        )

        # Mock should_run_now to return True
        service.should_run_now = Mock(return_value=True)

        # Call fetch
        service.fetch_and_store_with_retry()

        # Verify retries stopped after success
        self.assertEqual(mock_tracker_instance.fetch_and_store_odds.call_count, 3)

        # Verify metrics (should count as success)
        self.assertEqual(service.total_runs, 1)
        self.assertEqual(service.successful_runs, 1)
        self.assertEqual(service.failed_runs, 0)

    @patch('odds_tracker_service.OddsTracker')
    def test_fetch_outside_operating_hours(self, mock_tracker_class):
        """Test fetch is skipped outside operating hours."""
        mock_tracker_instance = Mock()
        mock_tracker_class.return_value = mock_tracker_instance

        service = OddsTrackerService(
            api_key='test_key',
            db_path=self.test_db,
            log_file=self.test_log
        )

        # Mock should_run_now to return False
        service.should_run_now = Mock(return_value=False)

        # Call fetch
        service.fetch_and_store_with_retry()

        # Verify tracker was NOT called
        mock_tracker_instance.fetch_and_store_odds.assert_not_called()

        # Verify metrics NOT updated
        self.assertEqual(service.total_runs, 0)

    @patch('odds_tracker_service.OddsTracker')
    def test_get_health_status(self, mock_tracker):
        """Test health status reporting."""
        service = OddsTrackerService(
            api_key='test_key',
            db_path=self.test_db,
            log_file=self.test_log
        )

        # Initial status
        status = service.get_health_status()
        self.assertEqual(status['status'], 'stopped')
        self.assertEqual(status['total_runs'], 0)
        self.assertEqual(status['successful_runs'], 0)
        self.assertEqual(status['failed_runs'], 0)
        self.assertEqual(status['success_rate'], '0.0%')

        # Simulate some runs
        service.total_runs = 10
        service.successful_runs = 9
        service.failed_runs = 1

        status = service.get_health_status()
        self.assertEqual(status['total_runs'], 10)
        self.assertEqual(status['successful_runs'], 9)
        self.assertEqual(status['failed_runs'], 1)
        self.assertEqual(status['success_rate'], '90.0%')

    @patch('odds_tracker_service.OddsTracker')
    def test_start_and_stop_service(self, mock_tracker):
        """Test starting and stopping the scheduler."""
        service = OddsTrackerService(
            api_key='test_key',
            update_interval=5,
            db_path=self.test_db,
            log_file=self.test_log
        )

        # Mock should_run_now to avoid immediate fetch
        service.should_run_now = Mock(return_value=False)

        # Start service
        service.start()

        # Verify scheduler is running
        self.assertTrue(service.scheduler.running)
        self.assertIsNotNone(service.service_start_time)

        # Verify job was added
        jobs = service.scheduler.get_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].id, 'odds_tracker_job')

        # Stop service
        service.stop()

        # Verify scheduler stopped
        self.assertFalse(service.scheduler.running)

    @patch('odds_tracker_service.OddsTracker')
    def test_logging_setup(self, mock_tracker):
        """Test logging configuration."""
        service = OddsTrackerService(
            api_key='test_key',
            db_path=self.test_db,
            log_file=self.test_log
        )

        # Verify logger created
        self.assertIsNotNone(service.logger)
        self.assertEqual(service.logger.name, "OddsTrackerService")

        # Note: Log file may not exist until first write
        # Just verify logger works without error
        service.logger.info("Test log message")

    @patch('odds_tracker_service.OddsTracker')
    def test_print_status(self, mock_tracker):
        """Test status printing (no assertions, just ensure no errors)."""
        service = OddsTrackerService(
            api_key='test_key',
            db_path=self.test_db,
            log_file=self.test_log
        )

        service.total_runs = 5
        service.successful_runs = 4
        service.failed_runs = 1
        service.service_start_time = datetime.now()

        # Should not raise error
        service.print_status()


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""

    def test_setup_logging(self):
        """Test logging setup."""
        test_log = "test_helper.log"

        logger = setup_logging(test_log)

        # Verify logger created
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "OddsTrackerService")

        # Write test message
        logger.info("Test message")

        # Clean up if file exists
        if Path(test_log).exists():
            Path(test_log).unlink()

    def test_nba_season_months(self):
        """Test NBA season months constant."""
        self.assertEqual(len(NBA_SEASON_MONTHS), 9)
        self.assertIn(10, NBA_SEASON_MONTHS)  # October
        self.assertIn(1, NBA_SEASON_MONTHS)   # January
        self.assertIn(6, NBA_SEASON_MONTHS)   # June
        self.assertNotIn(7, NBA_SEASON_MONTHS)  # July (offseason)
        self.assertNotIn(8, NBA_SEASON_MONTHS)  # August (offseason)
        self.assertNotIn(9, NBA_SEASON_MONTHS)  # September (offseason)

    def test_operating_hours(self):
        """Test operating hours constants."""
        self.assertEqual(START_HOUR, 8)
        self.assertEqual(END_HOUR, 23)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_db = "test_edge_cases.db"
        self.test_log = "test_edge_cases.log"
        os.environ['THE_ODDS_API_KEY'] = 'test_api_key'

    def tearDown(self):
        """Clean up test artifacts."""
        for file in [self.test_db, self.test_log]:
            if Path(file).exists():
                Path(file).unlink()

    @patch('odds_tracker_service.OddsTracker')
    def test_scheduler_graceful_shutdown(self, mock_tracker):
        """Test scheduler shuts down gracefully."""
        service = OddsTrackerService(
            api_key='test_key',
            db_path=self.test_db,
            log_file=self.test_log
        )

        service.should_run_now = Mock(return_value=False)
        service.start()

        # Should not raise error
        service.stop()
        self.assertFalse(service.scheduler.running)

    @patch('odds_tracker_service.OddsTracker')
    def test_multiple_start_calls(self, mock_tracker):
        """Test service handles multiple start calls."""
        service = OddsTrackerService(
            api_key='test_key',
            db_path=self.test_db,
            log_file=self.test_log
        )

        service.should_run_now = Mock(return_value=False)

        # First start should work
        service.start()

        # Should have 1 job
        jobs = service.scheduler.get_jobs()
        self.assertEqual(len(jobs), 1)

        # Second start should raise error (scheduler already running)
        # This is expected behavior - APScheduler doesn't allow double-start
        from apscheduler.schedulers import SchedulerAlreadyRunningError
        with self.assertRaises(SchedulerAlreadyRunningError):
            service.start()

        service.stop()


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestOddsTrackerService))
    suite.addTests(loader.loadTestsFromTestCase(TestHelperFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
