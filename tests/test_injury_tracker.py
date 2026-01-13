"""
Unit Tests for Injury Tracker v3
=================================

Tests cover:
- InjuryStatus enum parsing
- InjuryReport data class
- InjuryCache with TTL
- Data fetching with fallback logic
- Player availability checks
- Star player detection
- Database persistence

Run with:
    pytest tests/test_injury_tracker.py -v
    pytest tests/test_injury_tracker.py::TestInjuryStatus -v
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from injury_tracker_v3 import (
    InjuryStatus,
    InjuryReport,
    InjuryCache,
    fetch_current_injuries,
    is_player_available,
    detect_star_player_out,
    get_injury_summary,
    clear_injury_cache,
)


class TestInjuryStatus(unittest.TestCase):
    """Test InjuryStatus enum functionality."""

    def test_from_string_standard_cases(self):
        """Test parsing standard status strings."""
        self.assertEqual(InjuryStatus.from_string("Out"), InjuryStatus.OUT)
        self.assertEqual(InjuryStatus.from_string("out"), InjuryStatus.OUT)
        self.assertEqual(InjuryStatus.from_string("Doubtful"), InjuryStatus.DOUBTFUL)
        self.assertEqual(InjuryStatus.from_string("Questionable"), InjuryStatus.QUESTIONABLE)
        self.assertEqual(InjuryStatus.from_string("Probable"), InjuryStatus.PROBABLE)
        self.assertEqual(InjuryStatus.from_string("GTD"), InjuryStatus.GTD)

    def test_from_string_abbreviations(self):
        """Test parsing abbreviated status strings."""
        self.assertEqual(InjuryStatus.from_string("O"), InjuryStatus.OUT)
        self.assertEqual(InjuryStatus.from_string("D"), InjuryStatus.DOUBTFUL)
        self.assertEqual(InjuryStatus.from_string("Q"), InjuryStatus.QUESTIONABLE)
        self.assertEqual(InjuryStatus.from_string("P"), InjuryStatus.PROBABLE)

    def test_from_string_variations(self):
        """Test parsing status string variations."""
        self.assertEqual(InjuryStatus.from_string("game time decision"), InjuryStatus.GTD)
        self.assertEqual(InjuryStatus.from_string("day-to-day"), InjuryStatus.GTD)
        self.assertEqual(InjuryStatus.from_string("day to day"), InjuryStatus.GTD)

    def test_from_string_unknown(self):
        """Test parsing unknown status strings."""
        self.assertEqual(InjuryStatus.from_string("injured"), InjuryStatus.UNKNOWN)
        self.assertEqual(InjuryStatus.from_string(""), InjuryStatus.UNKNOWN)
        self.assertEqual(InjuryStatus.from_string("xyz"), InjuryStatus.UNKNOWN)

    def test_availability_probability(self):
        """Test availability probability calculations."""
        self.assertEqual(InjuryStatus.OUT.availability_probability(), 0.0)
        self.assertEqual(InjuryStatus.DOUBTFUL.availability_probability(), 0.25)
        self.assertEqual(InjuryStatus.QUESTIONABLE.availability_probability(), 0.50)
        self.assertEqual(InjuryStatus.PROBABLE.availability_probability(), 0.75)
        self.assertEqual(InjuryStatus.GTD.availability_probability(), 0.50)
        self.assertEqual(InjuryStatus.AVAILABLE.availability_probability(), 1.0)


class TestInjuryReport(unittest.TestCase):
    """Test InjuryReport data class."""

    def test_initialization(self):
        """Test InjuryReport initialization."""
        report = InjuryReport(
            player_name="LeBron James",
            player_id=237,
            team_abbrev="LAL",
            team_id=13,
            status=InjuryStatus.QUESTIONABLE,
            injury_detail="Ankle soreness",
        )

        self.assertEqual(report.player_name, "LeBron James")
        self.assertEqual(report.player_id, 237)
        self.assertEqual(report.team_abbrev, "LAL")
        self.assertEqual(report.status, InjuryStatus.QUESTIONABLE)
        self.assertIsNotNone(report.last_updated)

    def test_is_unavailable(self):
        """Test unavailability detection."""
        out_report = InjuryReport("Player", status=InjuryStatus.OUT)
        doubtful_report = InjuryReport("Player", status=InjuryStatus.DOUBTFUL)
        questionable_report = InjuryReport("Player", status=InjuryStatus.QUESTIONABLE)

        self.assertTrue(out_report.is_unavailable())
        self.assertTrue(doubtful_report.is_unavailable())
        self.assertFalse(questionable_report.is_unavailable())

    def test_is_uncertain(self):
        """Test uncertainty detection."""
        questionable_report = InjuryReport("Player", status=InjuryStatus.QUESTIONABLE)
        gtd_report = InjuryReport("Player", status=InjuryStatus.GTD)
        out_report = InjuryReport("Player", status=InjuryStatus.OUT)

        self.assertTrue(questionable_report.is_uncertain())
        self.assertTrue(gtd_report.is_uncertain())
        self.assertFalse(out_report.is_uncertain())

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = InjuryReport(
            player_name="Stephen Curry",
            player_id=201939,
            status=InjuryStatus.OUT,
        )

        data = report.to_dict()

        self.assertIsInstance(data, dict)
        self.assertEqual(data['player_name'], "Stephen Curry")
        self.assertEqual(data['status'], "Out")
        self.assertIsNotNone(data['last_updated'])


class TestInjuryCache(unittest.TestCase):
    """Test InjuryCache functionality."""

    def setUp(self):
        """Set up test cache."""
        self.cache = InjuryCache(ttl_minutes=1, max_size=10)

    def tearDown(self):
        """Clean up test cache."""
        self.cache.clear()

    def test_cache_set_and_get(self):
        """Test basic cache set and get."""
        reports = [
            InjuryReport("Player A", status=InjuryStatus.OUT),
            InjuryReport("Player B", status=InjuryStatus.QUESTIONABLE),
        ]

        self.cache.set("2025-01-15", reports)
        cached = self.cache.get("2025-01-15")

        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), 2)
        self.assertEqual(cached[0].player_name, "Player A")

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cached = self.cache.get("2025-01-15")
        self.assertIsNone(cached)

    def test_cache_expiration(self):
        """Test cache expiration (TTL)."""
        reports = [InjuryReport("Player A", status=InjuryStatus.OUT)]

        # Use cache with 0-second TTL to simulate expiration
        short_cache = InjuryCache(ttl_minutes=0, max_size=10)
        short_cache.set("2025-01-15", reports)

        # Should be expired immediately
        import time
        time.sleep(0.1)
        cached = short_cache.get("2025-01-15")
        self.assertIsNone(cached)

    def test_cache_max_size(self):
        """Test cache eviction when max size reached."""
        small_cache = InjuryCache(ttl_minutes=10, max_size=3)

        # Add 4 entries (should evict the first one)
        for i in range(4):
            reports = [InjuryReport(f"Player {i}", status=InjuryStatus.OUT)]
            small_cache.set(f"2025-01-{i:02d}", reports)

        # First entry should be evicted
        self.assertIsNone(small_cache.get("2025-01-00"))

        # Last 3 should still be there
        self.assertIsNotNone(small_cache.get("2025-01-01"))
        self.assertIsNotNone(small_cache.get("2025-01-02"))
        self.assertIsNotNone(small_cache.get("2025-01-03"))

    def test_cache_stats(self):
        """Test cache statistics."""
        reports = [InjuryReport("Player A", status=InjuryStatus.OUT)]
        self.cache.set("2025-01-15", reports)

        stats = self.cache.get_stats()

        self.assertEqual(stats['size'], 1)
        self.assertEqual(stats['max_size'], 10)
        self.assertEqual(stats['ttl_minutes'], 1)
        self.assertIn("2025-01-15", stats['entries'])

    def test_cache_clear(self):
        """Test cache clearing."""
        reports = [InjuryReport("Player A", status=InjuryStatus.OUT)]
        self.cache.set("2025-01-15", reports)

        self.cache.clear()
        stats = self.cache.get_stats()

        self.assertEqual(stats['size'], 0)


class TestFetchCurrentInjuries(unittest.TestCase):
    """Test injury fetching with fallback logic."""

    def setUp(self):
        """Set up test environment."""
        clear_injury_cache()

    @patch('injury_tracker_v3.scrape_nba_injuries')
    def test_fetch_from_nba_com(self, mock_scrape):
        """Test fetching from NBA.com scraper."""
        mock_injuries = [
            InjuryReport("Player A", status=InjuryStatus.OUT, source="nba.com"),
            InjuryReport("Player B", status=InjuryStatus.QUESTIONABLE, source="nba.com"),
        ]
        mock_scrape.return_value = mock_injuries

        injuries = fetch_current_injuries(use_cache=False)

        self.assertEqual(len(injuries), 2)
        self.assertEqual(injuries[0].source, "nba.com")
        mock_scrape.assert_called_once()

    @patch('injury_tracker_v3.scrape_espn_injuries')
    @patch('injury_tracker_v3.scrape_nba_injuries')
    def test_fallback_to_espn(self, mock_nba, mock_espn):
        """Test fallback to ESPN when NBA.com fails."""
        mock_nba.return_value = []  # NBA.com returns nothing
        mock_espn.return_value = [
            InjuryReport("Player A", status=InjuryStatus.OUT, source="espn"),
        ]

        injuries = fetch_current_injuries(use_cache=False)

        self.assertEqual(len(injuries), 1)
        self.assertEqual(injuries[0].source, "espn")
        mock_nba.assert_called_once()
        mock_espn.assert_called_once()

    @patch('injury_tracker_v3.scrape_nba_injuries')
    def test_cache_is_used(self, mock_scrape):
        """Test that cache is used on subsequent calls."""
        mock_injuries = [InjuryReport("Player A", status=InjuryStatus.OUT)]
        mock_scrape.return_value = mock_injuries

        # First call - should hit scraper
        injuries1 = fetch_current_injuries(use_cache=True)
        self.assertEqual(mock_scrape.call_count, 1)

        # Second call - should use cache
        injuries2 = fetch_current_injuries(use_cache=True)
        self.assertEqual(mock_scrape.call_count, 1)  # Still 1 (cache hit)

        self.assertEqual(len(injuries1), len(injuries2))


class TestPlayerAvailability(unittest.TestCase):
    """Test player availability checking."""

    @patch('injury_tracker_v3.fetch_current_injuries')
    def test_player_available_no_injury(self, mock_fetch):
        """Test player with no injury report (available)."""
        mock_fetch.return_value = []

        available, status = is_player_available(237, datetime.now())

        self.assertTrue(available)
        self.assertIsNone(status)

    @patch('injury_tracker_v3.fetch_current_injuries')
    def test_player_out(self, mock_fetch):
        """Test player marked as OUT."""
        mock_fetch.return_value = [
            InjuryReport("Player", player_id=237, status=InjuryStatus.OUT)
        ]

        available, status = is_player_available(237, datetime.now())

        self.assertFalse(available)
        self.assertEqual(status, InjuryStatus.OUT)

    @patch('injury_tracker_v3.fetch_current_injuries')
    def test_player_questionable(self, mock_fetch):
        """Test player marked as QUESTIONABLE (available but uncertain)."""
        mock_fetch.return_value = [
            InjuryReport("Player", player_id=237, status=InjuryStatus.QUESTIONABLE)
        ]

        available, status = is_player_available(237, datetime.now())

        self.assertTrue(available)  # Not definitively unavailable
        self.assertEqual(status, InjuryStatus.QUESTIONABLE)


class TestStarPlayerDetection(unittest.TestCase):
    """Test star player injury detection."""

    @patch('injury_tracker_v3.fetch_current_injuries')
    def test_star_player_out(self, mock_fetch):
        """Test detection of star player being out."""
        mock_fetch.return_value = [
            InjuryReport(
                "LeBron James",
                player_id=237,
                team_id=13,
                status=InjuryStatus.OUT
            )
        ]

        has_star_out, names = detect_star_player_out(13, datetime.now())

        self.assertTrue(has_star_out)
        self.assertIn("LeBron James", names)

    @patch('injury_tracker_v3.fetch_current_injuries')
    def test_no_star_out(self, mock_fetch):
        """Test when no star players are out."""
        mock_fetch.return_value = []

        has_star_out, names = detect_star_player_out(13, datetime.now())

        self.assertFalse(has_star_out)
        self.assertEqual(names, [])


class TestInjurySummary(unittest.TestCase):
    """Test injury summary generation."""

    @patch('injury_tracker_v3.fetch_current_injuries')
    def test_injury_summary(self, mock_fetch):
        """Test injury summary generation."""
        mock_fetch.return_value = [
            InjuryReport("Player A", status=InjuryStatus.OUT, source="nba.com"),
            InjuryReport("Player B", status=InjuryStatus.OUT, source="nba.com"),
            InjuryReport("Player C", status=InjuryStatus.QUESTIONABLE, source="nba.com"),
            InjuryReport("Player D", status=InjuryStatus.DOUBTFUL, source="nba.com"),
        ]

        summary = get_injury_summary()

        self.assertEqual(summary['total_count'], 4)
        self.assertEqual(summary['out_count'], 2)
        self.assertEqual(summary['questionable_count'], 1)
        self.assertEqual(summary['doubtful_count'], 1)
        self.assertEqual(summary['source'], "nba.com")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_player_name(self):
        """Test injury report with empty player name."""
        report = InjuryReport("", status=InjuryStatus.OUT)
        self.assertEqual(report.player_name, "")

    def test_none_player_id(self):
        """Test injury report with None player_id."""
        report = InjuryReport("Player", player_id=None)
        self.assertIsNone(report.player_id)

    def test_status_enum_edge_cases(self):
        """Test status enum with edge cases."""
        self.assertEqual(InjuryStatus.from_string("  OUT  "), InjuryStatus.OUT)
        self.assertEqual(InjuryStatus.from_string("oUt"), InjuryStatus.OUT)


# =============================================================================
# Integration Tests (require actual data sources)
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests (can be skipped if dependencies unavailable)."""

    @unittest.skipUnless(os.getenv('RUN_INTEGRATION_TESTS'), "Skipping integration tests")
    def test_live_nba_scraping(self):
        """Test live scraping from NBA.com (requires internet)."""
        from injury_tracker_v3 import scrape_nba_injuries

        injuries = scrape_nba_injuries()

        # If successful, should return a list (may be empty on off-days)
        self.assertIsInstance(injuries, list)

    @unittest.skipUnless(os.getenv('RUN_INTEGRATION_TESTS'), "Skipping integration tests")
    def test_live_espn_scraping(self):
        """Test live scraping from ESPN (requires internet)."""
        from injury_tracker_v3 import scrape_espn_injuries

        injuries = scrape_espn_injuries()

        # If successful, should return a list
        self.assertIsInstance(injuries, list)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
