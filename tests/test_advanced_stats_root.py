
import unittest
from advanced_stats_v2 import FourFactorsCalculator

class TestAdvancedStats(unittest.TestCase):

    def setUp(self):
        self.calculator = FourFactorsCalculator()

    def test_four_factors_calculation(self):
        # Sample stats for a game
        # Team shoots 40/80 (50%), 10/20 3PT (50%)
        # FT: 15/20
        # TOV: 11
        # ORB: 10, Opp DRB: 30 (Total 40 reb opps)
        stats = {
            'fgm': 40, 'fga': 80,
            'fg3m': 10,
            'ftm': 15, 'fta': 20,
            'tov': 11,
            'orb': 10,
            'drb': 25, 'opp_orb': 8
        }
        opp_stats = {'drb': 30}

        # calculate_four_factors rounds to 4 decimal places
        factors = self.calculator.calculate_four_factors(stats, opp_stats=opp_stats)

        # 1. eFG% = (40 + 0.5*10) / 80 = 45 / 80 = 0.5625
        self.assertAlmostEqual(factors['efg_pct'], 0.5625, places=4)

        # 2. TOV% = 11 / (80 + 0.44*20 + 11) = 11 / 99.8 ≈ 0.1102
        self.assertAlmostEqual(factors['tov_pct'], 0.1102, places=4)

        # 3. ORB% = 10 / (10 + 30) = 0.25
        self.assertAlmostEqual(factors['orb_pct'], 0.25, places=4)

        # 4. FT Rate = FTA / FGA = 20 / 80 = 0.25
        self.assertAlmostEqual(factors['ft_rate'], 0.25, places=4)

    def test_four_factors_without_opponent_stats(self):
        """When opp_stats is None, ORB% uses estimated opponent DRB."""
        stats = {
            'fgm': 40, 'fga': 80,
            'fg3m': 10,
            'ftm': 15, 'fta': 20,
            'tov': 11,
            'orb': 10,
            'reb': 45,  # Used for opponent DRB estimate
        }

        factors = self.calculator.calculate_four_factors(stats)

        # All keys should be present
        self.assertIn('efg_pct', factors)
        self.assertIn('tov_pct', factors)
        self.assertIn('orb_pct', factors)
        self.assertIn('ft_rate', factors)

        # ORB% estimated: opp_drb = 45 * 0.74 = 33.3, ORB% = 10 / (10 + 33.3) ≈ 0.2310
        self.assertAlmostEqual(factors['orb_pct'], 10 / (10 + 45 * 0.74), places=4)

    def test_multiple_games_calculation(self):
        games = [
            {'fgm': 40, 'fga': 80, 'fg3m': 10, 'ftm': 15, 'fta': 20, 'tov': 11, 'orb': 10},
            {'fgm': 35, 'fga': 70, 'fg3m': 5, 'ftm': 10, 'fta': 10, 'tov': 10, 'orb': 5}
        ]

        # Verify calculate_four_factors works on each game and returns expected keys
        for game_stats in games:
            factors = self.calculator.calculate_four_factors(game_stats)
            self.assertIn('efg_pct', factors)
            self.assertIn('tov_pct', factors)
            self.assertIn('orb_pct', factors)
            self.assertIn('ft_rate', factors)

        # Verify second game values
        factors2 = self.calculator.calculate_four_factors(games[1])
        # eFG% = (35 + 0.5*5) / 70 = 37.5 / 70 = 0.5357...
        self.assertAlmostEqual(factors2['efg_pct'], 37.5 / 70, places=4)

if __name__ == '__main__':
    unittest.main()
