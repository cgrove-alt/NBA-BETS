
import sys
import unittest
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(".")

from simulation_engine import GameSimulatorV3, PlayerStats, TeamStats, PlayerTrackingStats

class TestV3Integration(unittest.TestCase):
    def setUp(self):
        # Create basic teams
        self.home_players = [
            PlayerStats(id=1, name="Home 1", position="G", ppg=20),
            PlayerStats(id=2, name="Home 2", position="G", ppg=15),
            PlayerStats(id=3, name="Home 3", position="F", ppg=15),
            PlayerStats(id=4, name="Home 4", position="F", ppg=10),
            PlayerStats(id=5, name="Home 5", position="C", ppg=10),
        ]
        self.away_players = [
            PlayerStats(id=6, name="Away 1", position="G", ppg=20),
            PlayerStats(id=7, name="Away 2", position="G", ppg=15),
            PlayerStats(id=8, name="Away 3", position="F", ppg=15),
            PlayerStats(id=9, name="Away 4", position="F", ppg=10),
            PlayerStats(id=10, name="Away 5", position="C", ppg=10),
        ]

        self.home_team = TeamStats(id=101, name="Home", abbreviation="HOM", players=self.home_players)
        self.away_team = TeamStats(id=102, name="Away", abbreviation="AWY", players=self.away_players)

        self.sim = GameSimulatorV3(self.home_team, self.away_team)

    def test_tracking_data_loading(self):
        """Test that tracking data loads and upgrades players."""
        print("\nTesting Tracking Data Loading...")

        # Mock tracking components
        mock_atlas = MagicMock()
        mock_rotation = MagicMock()

        # Load mock data
        self.sim.load_tracking_data(mock_atlas, mock_rotation)

        # Check flag
        self.assertTrue(self.sim.use_tracking_data, "Tracking data flag should be True")
        print("✓ Flag set correctly")

        # Check player upgrade
        upgraded_count = 0
        for p in self.sim.home.players:
            if isinstance(p, PlayerTrackingStats):
                upgraded_count += 1

        self.assertEqual(upgraded_count, 5, "All players should be upgraded to PlayerTrackingStats")
        print(f"✓ {upgraded_count} players upgraded")

    def test_v3_simulation_flow(self):
        """Test that simulation runs with V3 logic."""
        print("\nTesting V3 Simulation Flow...")

        mock_atlas = MagicMock()
        mock_rotation = MagicMock()

        # Configure mocks to avoid errors during simulation
        # Mock RotationTracker.to_simulation_input
        mock_rotation.to_simulation_input.return_value = {'lineup_probabilities': {}}
        mock_rotation.lineup_spells = {101: [], 102: []}

        self.sim.load_tracking_data(mock_atlas, mock_rotation)

        # Mock PlayerTrackingStats methods that will be called
        for team in [self.sim.home, self.sim.away]:
            for p in team.players:
                # Mock select_shot_zone
                p.select_shot_zone = MagicMock(return_value="Restricted Area")
                # Mock get_zone_shot_probability
                p.get_zone_shot_probability = MagicMock(return_value=0.5)
                # Mock usage factor
                p.lineup_usage_factor = 1.0
                p.synergy_partners = {}

        # Run a short simulation
        print("Running single game simulation...")
        try:
            result = self.sim.simulate_game()
            print(f"✓ Simulation completed. Score: {result['home_score']}-{result['away_score']}")
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Simulation failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
