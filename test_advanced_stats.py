
import unittest
from advanced_stats_v2 import get_four_factors, get_rolling_four_factors

class TestAdvancedStats(unittest.TestCase):

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
            'orb': 10, 'opp_drb': 30,
            # Extra fields just to match dict structure
            'drb': 25, 'opp_orb': 8
        }

        factors = get_four_factors(stats)

        # 1. eFG% = (40 + 0.5*10) / 80 = 45 / 80 = 0.5625
        self.assertAlmostEqual(factors['efg_pct'], 0.5625)

        # 2. TOV% = 11 / (80 + 0.44*20 + 11) = 11 / (80 + 8.8 + 11) = 11 / 99.8 = 0.1102...
        expected_tov_pct = 11 / (80 + 0.44*20 + 11)
        self.assertAlmostEqual(factors['tov_pct'], expected_tov_pct)

        # 3. ORB% = 10 / (10 + 30) = 0.25
        self.assertAlmostEqual(factors['orb_pct'], 0.25)

        # 4. FT Rate = 15 / 80 = 0.1875
        self.assertAlmostEqual(factors['ft_rate'], 0.1875)

    def test_rolling_factors(self):
        games = [
            {'fgm': 40, 'fga': 80, 'fg3m': 10, 'ftm': 15, 'fta': 20, 'tov': 11, 'orb': 10, 'opp_drb': 30},
            {'fgm': 35, 'fga': 70, 'fg3m': 5, 'ftm': 10, 'fta': 10, 'tov': 10, 'orb': 5, 'opp_drb': 20}
        ]

        rolling = get_rolling_four_factors(games, window=2)
        # Check keys exist
        self.assertIn('efg_pct_avg', rolling)
        self.assertIn('tov_pct_avg', rolling)

if __name__ == '__main__':
    unittest.main()
