"""
Unit tests for player_impact_fetcher.py with DARKO/EPM/RAPTOR enhancements.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from player_impact_fetcher import (
    PlayerImpactFetcher,
    get_star_player_impact,
    calculate_injury_adjustment,
    get_player_role,
    calculate_prop_injury_boost,
)


class TestPlayerImpactFetcher:
    """Test the enhanced PlayerImpactFetcher class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.fetcher = PlayerImpactFetcher()

    def test_standardize_metric_darko(self):
        """Test DARKO metric standardization."""
        # MVP-level DARKO (8.0) should map to 10.0
        assert self.fetcher._standardize_metric(8.0, 'darko') == 10.0

        # Average player (0.0) should map to 0.0
        assert self.fetcher._standardize_metric(0.0, 'darko') == 0.0

        # Below-average (-4.0) should map to -5.0
        assert self.fetcher._standardize_metric(-4.0, 'darko') == -5.0

        # Should cap at ±10
        assert self.fetcher._standardize_metric(20.0, 'darko') == 10.0
        assert self.fetcher._standardize_metric(-20.0, 'darko') == -10.0

    def test_standardize_metric_epm(self):
        """Test EPM metric standardization."""
        # Elite EPM (7.0) should map close to 10.0
        result = self.fetcher._standardize_metric(7.0, 'epm')
        assert 9.0 <= result <= 10.0

        # Average (0.0) should stay 0.0
        assert self.fetcher._standardize_metric(0.0, 'epm') == 0.0

        # Should cap at ±10
        assert self.fetcher._standardize_metric(15.0, 'epm') == 10.0
        assert self.fetcher._standardize_metric(-15.0, 'epm') == -10.0

    def test_standardize_metric_raptor(self):
        """Test RAPTOR metric standardization."""
        # Elite RAPTOR (8.0) should map to 10.0
        assert self.fetcher._standardize_metric(8.0, 'raptor') == 10.0

        # Average (0.0) should stay 0.0
        assert self.fetcher._standardize_metric(0.0, 'raptor') == 0.0

    def test_standardize_metric_plus_minus(self):
        """Test plus/minus metric standardization."""
        # Already on correct scale, should just cap
        assert self.fetcher._standardize_metric(5.0, 'plus_minus') == 5.0
        assert self.fetcher._standardize_metric(-8.0, 'plus_minus') == -8.0
        assert self.fetcher._standardize_metric(15.0, 'plus_minus') == 10.0
        assert self.fetcher._standardize_metric(-15.0, 'plus_minus') == -10.0

    def test_standardize_metric_unknown_type(self):
        """Test unknown metric type returns 0."""
        assert self.fetcher._standardize_metric(5.0, 'unknown') == 0.0

    @patch('player_impact_fetcher.requests.get')
    def test_fetch_darko_dpm_success(self, mock_get):
        """Test successful DARKO data fetching."""
        # Mock HTML response with a table
        mock_html = """
        <html>
            <table>
                <tr>
                    <th>Player Name</th>
                    <th>Team</th>
                    <th>DPM</th>
                </tr>
                <tr>
                    <td>Nikola Jokic</td>
                    <td>DEN</td>
                    <td>7.5</td>
                </tr>
                <tr>
                    <td>Luka Doncic</td>
                    <td>DAL</td>
                    <td>6.8</td>
                </tr>
            </table>
        </html>
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_html.encode('utf-8')
        mock_get.return_value = mock_response

        result = self.fetcher.fetch_darko_dpm("2024-25")

        assert len(result) == 2
        assert "Nikola Jokic" in result
        assert result["Nikola Jokic"]["source"] == "darko"
        assert result["Nikola Jokic"]["raw_dpm"] == 7.5
        assert result["Nikola Jokic"]["team"] == "DEN"
        # Check standardized impact is calculated
        assert "impact_metric" in result["Nikola Jokic"]

    @patch('player_impact_fetcher.requests.get')
    def test_fetch_darko_dpm_http_error(self, mock_get):
        """Test DARKO fetching handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = self.fetcher.fetch_darko_dpm("2024-25")

        assert result == {}

    @patch('player_impact_fetcher.requests.get')
    def test_fetch_darko_dpm_no_tables(self, mock_get):
        """Test DARKO fetching when no tables found."""
        mock_html = "<html><body>No tables here</body></html>"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_html.encode('utf-8')
        mock_get.return_value = mock_response

        result = self.fetcher.fetch_darko_dpm("2024-25")

        assert result == {}

    @patch('player_impact_fetcher.requests.get')
    def test_fetch_raptor_success(self, mock_get):
        """Test successful RAPTOR data fetching."""
        # Mock CSV response
        mock_csv = """player_name,season,team,raptor_total
Nikola Jokic,2024,DEN,8.2
Luka Doncic,2024,DAL,7.5
LeBron James,2023,LAL,5.0
"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_csv
        mock_get.return_value = mock_response

        result = self.fetcher.fetch_fivethirtyeight_raptor("2024-25")

        assert len(result) >= 2  # Should have 2024 players
        # Should filter to 2024 season
        if "Nikola Jokic" in result:
            assert result["Nikola Jokic"]["source"] == "raptor"
            assert "impact_metric" in result["Nikola Jokic"]

    @patch('player_impact_fetcher.requests.get')
    def test_fetch_raptor_http_error(self, mock_get):
        """Test RAPTOR fetching handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = self.fetcher.fetch_fivethirtyeight_raptor("2024-25")

        assert result == {}

    @patch('player_impact_fetcher.requests.get')
    def test_fetch_espn_epm(self, mock_get):
        """Test ESPN EPM fetching (expected to fail - requires JS)."""
        mock_html = "<html><body>Stats page</body></html>"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_html.encode('utf-8')
        mock_get.return_value = mock_response

        result = self.fetcher.fetch_espn_epm(2025)

        # Should return empty as ESPN requires JavaScript
        assert result == {}

    def test_get_player_impact_priority_darko(self):
        """Test get_player_impact prioritizes DARKO."""
        # Setup multiple cache entries for same player
        self.fetcher.darko_cache = {
            "Test Player": {"source": "darko", "impact_metric": 8.0}
        }
        self.fetcher.epm_cache = {
            "Test Player": {"source": "epm", "impact_metric": 7.0}
        }
        self.fetcher.raptor_cache = {
            "Test Player": {"source": "raptor", "impact_metric": 6.0}
        }

        result = self.fetcher.get_player_impact("Test Player")

        # Should return DARKO (highest priority)
        assert result["source"] == "darko"
        assert result["impact_metric"] == 8.0

    def test_get_player_impact_priority_epm(self):
        """Test get_player_impact falls back to EPM."""
        self.fetcher.epm_cache = {
            "Test Player": {"source": "epm", "impact_metric": 7.0}
        }
        self.fetcher.raptor_cache = {
            "Test Player": {"source": "raptor", "impact_metric": 6.0}
        }

        result = self.fetcher.get_player_impact("Test Player")

        # Should return EPM (DARKO not available)
        assert result["source"] == "epm"
        assert result["impact_metric"] == 7.0

    def test_get_player_impact_priority_raptor(self):
        """Test get_player_impact falls back to RAPTOR."""
        self.fetcher.raptor_cache = {
            "Test Player": {"source": "raptor", "impact_metric": 6.0}
        }
        self.fetcher.basic_stats_cache = {
            "Test Player": {"source": "nba_api", "impact_metric": 5.0}
        }

        result = self.fetcher.get_player_impact("Test Player")

        # Should return RAPTOR
        assert result["source"] == "raptor"
        assert result["impact_metric"] == 6.0

    def test_get_player_impact_priority_basic(self):
        """Test get_player_impact falls back to basic stats."""
        self.fetcher.basic_stats_cache = {
            "Test Player": {"source": "nba_api", "impact_metric": 5.0}
        }

        result = self.fetcher.get_player_impact("Test Player")

        # Should return basic stats
        assert result["source"] == "nba_api"
        assert result["impact_metric"] == 5.0

    def test_get_player_impact_not_found(self):
        """Test get_player_impact returns None when player not found."""
        # Empty caches, prevent auto-refresh
        self.fetcher.darko_cache = {"Other Player": {}}

        result = self.fetcher.get_player_impact("Unknown Player")

        assert result is None

    def test_get_player_impact_metric(self):
        """Test get_player_impact_metric extracts metric value."""
        self.fetcher.darko_cache = {
            "Test Player": {"impact_metric": 7.5}
        }

        result = self.fetcher.get_player_impact_metric("Test Player")

        assert result == 7.5

    def test_get_player_impact_metric_not_found(self):
        """Test get_player_impact_metric returns 0.0 for unknown player."""
        self.fetcher.darko_cache = {"Other Player": {}}

        result = self.fetcher.get_player_impact_metric("Unknown Player")

        assert result == 0.0

    def test_get_team_impact_when_player_on_court(self):
        """Test team impact calculation with player."""
        self.fetcher.darko_cache = {
            "Nikola Jokic": {"team": "DEN", "impact_metric": 8.5}
        }

        result = self.fetcher.get_team_impact_when_player_on_court("DEN", "Nikola Jokic")

        assert result == 8.5  # Should return player's impact

    def test_get_opponent_defensive_impact_vs_position(self):
        """Test opponent defensive impact calculation."""
        # Setup opponent team with defenders
        self.fetcher.darko_cache = {
            "Defender 1": {"team": "LAL", "impact_metric": 5.0},
            "Defender 2": {"team": "LAL", "impact_metric": 4.0},
            "Defender 3": {"team": "LAL", "impact_metric": 3.0},
            "Defender 4": {"team": "LAL", "impact_metric": 2.0},
        }

        result = self.fetcher.get_opponent_defensive_impact_vs_position("LAL", "G")

        # Should use top 3 defenders: avg = (5+4+3)/3 = 4.0
        # Inverted and scaled: -4.0 * 0.3 = -1.2
        assert result == pytest.approx(-1.2, abs=0.01)

    def test_calculate_team_rating_adjustment_single_player(self):
        """Test team rating adjustment for single injured player."""
        self.fetcher.darko_cache = {
            "LeBron James": {"team": "LAL", "impact_metric": 6.0, "minutes": 35}
        }

        result = self.fetcher.calculate_team_rating_adjustment(
            "LAL",
            injured_players=["LeBron James"]
        )

        # Impact: 6.0, minutes weight: 35/36 ≈ 0.97, scale: 0.5
        # Adjustment: -6.0 * 0.97 * 0.5 ≈ -2.91
        assert result < 0  # Should be negative
        assert -3.5 <= result <= -2.5

    def test_calculate_team_rating_adjustment_multiple_players(self):
        """Test team rating adjustment for multiple injured players."""
        self.fetcher.darko_cache = {
            "LeBron James": {"team": "LAL", "impact_metric": 6.0, "minutes": 35},
            "Anthony Davis": {"team": "LAL", "impact_metric": 5.5, "minutes": 34}
        }

        result = self.fetcher.calculate_team_rating_adjustment(
            "LAL",
            injured_players=["LeBron James", "Anthony Davis"]
        )

        # Should be more negative than single player
        assert result < -4.0

    def test_calculate_team_rating_adjustment_no_injuries(self):
        """Test team rating adjustment with no injuries."""
        result = self.fetcher.calculate_team_rating_adjustment("LAL")

        assert result == 0.0

    def test_get_team_roster_impacts(self):
        """Test getting team roster sorted by impact."""
        self.fetcher.darko_cache = {
            "Star Player": {"team": "BOS", "impact_metric": 8.0, "minutes": 35, "points": 28},
            "Role Player": {"team": "BOS", "impact_metric": 2.0, "minutes": 22, "points": 10},
        }
        self.fetcher.raptor_cache = {
            "Starter": {"team": "BOS", "impact_metric": 5.0, "minutes": 30, "points": 18},
        }

        result = self.fetcher.get_team_roster_impacts("BOS")

        assert len(result) == 3
        # Should be sorted by impact (descending)
        assert result[0]["name"] == "Star Player"
        assert result[0]["impact"] == 8.0
        assert result[1]["name"] == "Starter"
        assert result[1]["impact"] == 5.0
        assert result[2]["name"] == "Role Player"
        assert result[2]["impact"] == 2.0

    def test_get_team_roster_impacts_no_duplicates(self):
        """Test team roster doesn't include duplicate players."""
        # Same player in multiple caches
        self.fetcher.darko_cache = {
            "Player A": {"team": "LAL", "impact_metric": 7.0}
        }
        self.fetcher.raptor_cache = {
            "Player A": {"team": "LAL", "impact_metric": 6.5}
        }

        result = self.fetcher.get_team_roster_impacts("LAL")

        # Should only appear once (from highest priority cache)
        assert len(result) == 1
        assert result[0]["impact"] == 7.0  # DARKO value

    def test_cache_save_and_load_darko(self, tmp_path):
        """Test saving and loading DARKO cache."""
        # Override cache directory
        import player_impact_fetcher
        original_cache_dir = player_impact_fetcher.CACHE_DIR
        player_impact_fetcher.CACHE_DIR = tmp_path

        fetcher = PlayerImpactFetcher()
        fetcher.darko_cache = {
            "Test Player": {"impact_metric": 7.0, "team": "BOS"}
        }

        # Save cache
        fetcher._save_cache('darko')

        # Create new fetcher to load cache
        fetcher2 = PlayerImpactFetcher()

        assert "Test Player" in fetcher2.darko_cache
        assert fetcher2.darko_cache["Test Player"]["impact_metric"] == 7.0

        # Restore original cache dir
        player_impact_fetcher.CACHE_DIR = original_cache_dir

    def test_cache_expiry(self, tmp_path):
        """Test cache expiry after 24 hours."""
        import player_impact_fetcher
        original_cache_dir = player_impact_fetcher.CACHE_DIR
        player_impact_fetcher.CACHE_DIR = tmp_path

        # Create expired cache
        cache_file = tmp_path / "darko_cache.json"
        expired_time = datetime.now() - timedelta(hours=25)
        cache_data = {
            'timestamp': expired_time.isoformat(),
            'players': {"Old Player": {"impact_metric": 5.0}}
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

        # Load should skip expired cache
        fetcher = PlayerImpactFetcher()
        assert "Old Player" not in fetcher.darko_cache

        # Restore
        player_impact_fetcher.CACHE_DIR = original_cache_dir


class TestUtilityFunctions:
    """Test standalone utility functions."""

    def test_get_star_player_impact(self):
        """Test getting hardcoded star player impacts."""
        assert get_star_player_impact("Nikola Jokic") == 5.0
        assert get_star_player_impact("Luka Doncic") == 4.5
        assert get_star_player_impact("Unknown Player") == 0.0

    def test_calculate_injury_adjustment(self):
        """Test calculating injury adjustment from hardcoded impacts."""
        result = calculate_injury_adjustment(["Nikola Jokic", "Luka Doncic"])

        # Should be negative (team gets worse)
        # Jokic: -5.0, Luka: -4.5, Total: -9.5
        assert result == -9.5

    def test_calculate_injury_adjustment_empty(self):
        """Test injury adjustment with no injuries."""
        result = calculate_injury_adjustment([])
        assert result == 0.0

    def test_get_player_role(self):
        """Test getting player role info."""
        result = get_player_role("Jrue Holiday")

        assert result is not None
        assert result["position"] == "G"
        assert result["defensive_role"] == "perimeter"
        assert result["impact_score"] == 2.5

    def test_get_player_role_not_found(self):
        """Test getting role for unknown player."""
        result = get_player_role("Unknown Player")
        assert result is None

    def test_calculate_prop_injury_boost_perimeter_defender_out(self):
        """Test prop boost when perimeter defender is out."""
        result = calculate_prop_injury_boost(
            player_position="G",
            prop_type="points",
            opponent_injured=["Jrue Holiday"]
        )

        assert result["boost_factor"] > 1.0  # Should boost
        assert "Jrue Holiday" in result["reasons"][0]
        assert result["adjustment_pct"] > 0

    def test_calculate_prop_injury_boost_rim_protector_out(self):
        """Test prop boost when rim protector is out."""
        result = calculate_prop_injury_boost(
            player_position="C",
            prop_type="points",
            opponent_injured=["Rudy Gobert"]
        )

        assert result["boost_factor"] > 1.0  # Should boost
        assert "Rudy Gobert" in result["reasons"][0]

    def test_calculate_prop_injury_boost_primary_scorer_out(self):
        """Test prop boost when teammate primary scorer is out."""
        result = calculate_prop_injury_boost(
            player_position="G",
            prop_type="points",
            opponent_injured=[],
            teammate_injured=["Stephen Curry"]
        )

        assert result["boost_factor"] > 1.0  # Should boost usage
        assert "Stephen Curry" in result["reasons"][0]

    def test_calculate_prop_injury_boost_playmaker_out(self):
        """Test assist boost when teammate playmaker is out."""
        result = calculate_prop_injury_boost(
            player_position="G",
            prop_type="assists",
            opponent_injured=[],
            teammate_injured=["Luka Doncic"]
        )

        assert result["boost_factor"] > 1.0  # Should boost assists
        assert "Luka Doncic" in result["reasons"][0]

    def test_calculate_prop_injury_boost_capped(self):
        """Test prop boost is capped at ±15%."""
        # Create scenario with many injuries to test cap
        many_injured = [
            "Jrue Holiday", "Marcus Smart", "Alex Caruso",
            "Derrick White", "Lu Dort"
        ]

        result = calculate_prop_injury_boost(
            player_position="G",
            prop_type="points",
            opponent_injured=many_injured
        )

        # Should be capped at 1.15 (15% boost)
        assert result["boost_factor"] <= 1.15
        assert result["boost_factor"] >= 0.85

    def test_calculate_prop_injury_boost_no_injuries(self):
        """Test no boost when no injuries."""
        result = calculate_prop_injury_boost(
            player_position="G",
            prop_type="points",
            opponent_injured=[],
            teammate_injured=[]
        )

        assert result["boost_factor"] == 1.0
        assert len(result["reasons"]) == 0
        assert result["adjustment_pct"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
