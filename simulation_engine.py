"""
NBA Monte Carlo Game Simulation Engine

Possession-level simulation for accurate probability distributions of:
- Final scores and margins
- Player statistics (points, rebounds, assists, threes)
- Correlated outcomes (parlays, same-game combinations)

This approach captures nonlinearities that regression models miss:
- Pace-dependent standard deviations
- Blowout scenarios (starters benched early)
- Game script effects (increased attempts when trailing)

=============================================================================
ARCHITECTURE
=============================================================================
State Machine: Each possession flows through states
    StartPossession -> ShotAttempt/Turnover -> Made/Missed -> Rebound/Transition

Transition Probabilities: Derived from player and team statistics
    - P(Shot | Player, Matchup) from shooting splits
    - P(Turnover | Player, Defense) from turnover rates
    - P(Rebound | Team, Position) from rebounding rates

Output: Distribution of 10,000+ simulated games providing:
    - Win probability (more accurate than logistic regression)
    - Score distributions (for totals/spreads)
    - Player stat distributions (for props)
    - Correlation matrices (for parlays)
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import random
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime


# =============================================================================
# POSSESSION STATE MACHINE
# =============================================================================

class PossessionOutcome(Enum):
    """Outcomes of a possession."""
    MADE_TWO = "made_2pt"
    MADE_THREE = "made_3pt"
    MISSED_TWO = "missed_2pt"
    MISSED_THREE = "missed_3pt"
    TURNOVER = "turnover"
    FREE_THROWS = "free_throws"
    OFFENSIVE_REBOUND = "off_rebound"


@dataclass
class PlayerStats:
    """
    Player representation with per-game statistics.

    All stats are per-game averages which get scaled by game context.
    """
    id: int
    name: str
    position: str  # G, F, C
    team_id: int = 0

    # Minutes and usage
    minutes: float = 24.0  # Minutes per game
    usage_rate: float = 0.18  # % of team possessions used

    # Scoring
    ppg: float = 10.0  # Points per game
    fga: float = 8.0  # Field goal attempts per game
    fgm: float = 3.5  # Field goals made per game
    fg_pct: float = 0.44  # Field goal percentage
    fg3a: float = 3.0  # 3-point attempts per game
    fg3m: float = 1.0  # 3-point makes per game
    fg3_pct: float = 0.35  # 3-point percentage
    fta: float = 2.5  # Free throw attempts per game
    ftm: float = 2.0  # Free throws made per game
    ft_pct: float = 0.78  # Free throw percentage

    # Rebounding
    orb: float = 0.6  # Offensive rebounds per game
    drb: float = 3.0  # Defensive rebounds per game
    reb: float = 3.6  # Total rebounds per game

    # Playmaking
    ast: float = 2.5  # Assists per game
    tov: float = 1.2  # Turnovers per game

    # Defense
    stl: float = 0.7  # Steals per game
    blk: float = 0.3  # Blocks per game

    # Status
    is_starter: bool = False
    availability: float = 1.0  # 1.0 = fully healthy

    @property
    def three_rate(self) -> float:
        """Percentage of FGA that are 3-pointers."""
        return self.fg3a / self.fga if self.fga > 0 else 0.35


@dataclass
class TeamStats:
    """Team-level statistics for simulation."""
    id: int
    name: str
    abbreviation: str

    # Team ratings
    pace: float = 100.0  # Possessions per 48 minutes
    off_rating: float = 112.0  # Points per 100 possessions
    def_rating: float = 112.0  # Points allowed per 100 possessions

    # Team shooting
    efg_pct: float = 0.52  # Effective FG%
    ts_pct: float = 0.56  # True shooting %

    # Rebounding
    orb_pct: float = 0.26  # Offensive rebound %
    drb_pct: float = 0.74  # Defensive rebound %

    # Ball handling
    tov_pct: float = 0.13  # Turnover %

    # Roster
    players: List[PlayerStats] = field(default_factory=list)


@dataclass
class GameState:
    """Current state during game simulation."""
    quarter: int = 1
    time_remaining: float = 720.0  # seconds in quarter
    home_score: int = 0
    away_score: int = 0
    possession_team: int = 0  # 0 = home, 1 = away

    # Player stats tracking
    home_player_stats: Dict[int, Dict[str, int]] = field(default_factory=dict)
    away_player_stats: Dict[int, Dict[str, int]] = field(default_factory=dict)

    def get_margin(self) -> int:
        """Home team margin (positive = home leading)."""
        return self.home_score - self.away_score


# =============================================================================
# GAME SIMULATOR
# =============================================================================

class GameSimulator:
    """
    Monte Carlo NBA game simulator.

    Simulates games possession-by-possession using player/team statistics
    to generate probability distributions for all outcomes.
    """

    # Game constants
    QUARTER_LENGTH = 720  # seconds
    AVG_POSSESSION_TIME = 14.0  # seconds per possession
    HOME_ADVANTAGE = 2.5  # Points of home court advantage

    # Possession outcome base rates (NBA averages)
    TURNOVER_RATE = 0.13  # ~13% of possessions end in turnover
    FT_RATE = 0.22  # ~22% of possessions involve free throws
    ORB_RATE = 0.26  # ~26% of missed shots result in offensive rebound

    def __init__(
        self,
        home_team: TeamStats,
        away_team: TeamStats,
        neutral_site: bool = False
    ):
        self.home = home_team
        self.away = away_team
        self.neutral_site = neutral_site

        # Calculate game pace (average of both teams)
        self.game_pace = (home_team.pace + away_team.pace) / 2.0

        # Expected possessions per team
        self.possessions_per_team = int(self.game_pace * 48 / 48)

        # Home court adjustment
        self.home_boost = 0.0 if neutral_site else self.HOME_ADVANTAGE / 100.0

        # Results storage
        self.results: List[Dict] = []

    def _init_player_stats(self, team: TeamStats) -> Dict[int, Dict[str, int]]:
        """Initialize stat tracking for all players."""
        return {
            i: {
                'pts': 0, 'fgm': 0, 'fga': 0, 'fg3m': 0, 'fg3a': 0,
                'ftm': 0, 'fta': 0, 'orb': 0, 'drb': 0, 'reb': 0,
                'ast': 0, 'stl': 0, 'blk': 0, 'tov': 0,
                'min_played': 0, 'possessions': 0
            }
            for i in range(len(team.players))
        }

    def _select_shooter(self, team: TeamStats, game_state: GameState) -> int:
        """
        Select which player takes the shot based on usage rates.

        Accounts for:
        - Player usage rates
        - Minutes allocation
        - Blowout scenarios (bench players in garbage time)
        """
        margin = game_state.get_margin()
        is_garbage_time = game_state.quarter == 4 and abs(margin) > 20 and game_state.time_remaining < 300

        weights = []
        for player in team.players:
            if player.availability < 0.5:
                weight = 0.0
            elif is_garbage_time and player.is_starter:
                weight = player.usage_rate * 0.3  # Reduce starter usage in blowouts
            else:
                weight = player.usage_rate * player.availability
            weights.append(weight)

        if sum(weights) == 0:
            return 0

        # Normalize
        total = sum(weights)
        probs = [w / total for w in weights]

        return np.random.choice(len(team.players), p=probs)

    def _simulate_shot(
        self,
        shooter: PlayerStats,
        is_three: bool,
        defense_rating: float
    ) -> Tuple[bool, int]:
        """
        Simulate a shot attempt.

        Returns:
            (made, points)
        """
        # Adjust for defense
        def_factor = defense_rating / 112.0  # 112 = league average

        if is_three:
            # Add some variance to three-point shooting
            variance = np.random.normal(0, 0.05)
            adjusted_pct = shooter.fg3_pct * (1.0 / def_factor) + variance
            adjusted_pct = max(0.15, min(0.55, adjusted_pct))
            made = random.random() < adjusted_pct
            return (made, 3 if made else 0)
        else:
            # Two-point shot
            two_pt_pct = shooter.fg_pct  # Simplified
            variance = np.random.normal(0, 0.03)
            adjusted_pct = two_pt_pct * (1.0 / def_factor) + variance
            adjusted_pct = max(0.30, min(0.70, adjusted_pct))
            made = random.random() < adjusted_pct
            return (made, 2 if made else 0)

    def _simulate_free_throws(self, shooter: PlayerStats, num_fts: int) -> int:
        """Simulate free throw attempts."""
        made = 0
        for _ in range(num_fts):
            # Add slight variance to FT shooting
            variance = np.random.normal(0, 0.03)
            ft_pct = min(0.98, max(0.40, shooter.ft_pct + variance))
            if random.random() < ft_pct:
                made += 1
        return made

    def _simulate_possession(
        self,
        offense: TeamStats,
        defense: TeamStats,
        game_state: GameState,
        is_home_offense: bool
    ) -> Tuple[int, Dict]:
        """
        Simulate a single possession.

        Returns:
            (points_scored, stat_updates)
        """
        stat_updates = {'offense': {}, 'defense': {}}
        points = 0

        # Select primary ball handler
        shooter_idx = self._select_shooter(offense, game_state)
        shooter = offense.players[shooter_idx]

        # Adjust ratings for home court
        off_rating = offense.off_rating * (1 + self.home_boost if is_home_offense else 1)
        def_rating = defense.def_rating * (1 - self.home_boost * 0.5 if is_home_offense else 1)

        # Check for turnover
        tov_rate = offense.tov_pct
        if random.random() < tov_rate:
            stat_updates['offense'][shooter_idx] = {'tov': 1}

            # Check for steal
            if random.random() < 0.55:
                steal_idx = random.randint(0, len(defense.players) - 1)
                stat_updates['defense'][steal_idx] = {'stl': 1}

            return 0, stat_updates

        # Determine shot type
        is_three = random.random() < shooter.three_rate

        # Check for and-one opportunity (foul on made shot)
        foul_on_shot = random.random() < 0.08

        # Check for block
        block_rate = sum(p.blk for p in defense.players) / 48.0 / 5.0
        if random.random() < block_rate:
            blocker_idx = random.randint(0, len(defense.players) - 1)
            stat_updates['defense'][blocker_idx] = {'blk': 1}
            stat_updates['offense'][shooter_idx] = {'fga': 1, 'fg3a': 1 if is_three else 0}

            # 50% of blocks go out of bounds
            if random.random() < 0.5:
                return 0, stat_updates
            # Otherwise, rebound opportunity
            if random.random() < offense.orb_pct:
                reb_idx = self._select_rebounder(offense, is_offensive=True)
                stat_updates['offense'][reb_idx] = stat_updates['offense'].get(reb_idx, {})
                stat_updates['offense'][reb_idx]['orb'] = 1
                stat_updates['offense'][reb_idx]['reb'] = 1
                # Second chance - simplified
                is_three_2 = random.random() < 0.15  # Putbacks usually 2pt
                made_2, pts_2 = self._simulate_shot(shooter, is_three_2, def_rating)
                if made_2:
                    stat_updates['offense'][shooter_idx] = {
                        'pts': pts_2, 'fga': 1, 'fgm': 1,
                        'fg3a': 1 if is_three_2 else 0,
                        'fg3m': 1 if is_three_2 and made_2 else 0
                    }
                    return pts_2, stat_updates
            return 0, stat_updates

        # Simulate shot
        made, pts = self._simulate_shot(shooter, is_three, def_rating)

        # Update shooter stats
        shooter_stats = {
            'fga': 1,
            'fg3a': 1 if is_three else 0,
        }

        if made:
            points = pts
            shooter_stats['fgm'] = 1
            shooter_stats['fg3m'] = 1 if is_three else 0
            shooter_stats['pts'] = pts

            # Check for assist
            if random.random() < 0.58:  # ~58% of made shots assisted
                ast_idx = self._select_assister(offense, shooter_idx)
                if ast_idx is not None:
                    stat_updates['offense'][ast_idx] = stat_updates['offense'].get(ast_idx, {})
                    stat_updates['offense'][ast_idx]['ast'] = 1

            # And-one opportunity
            if foul_on_shot:
                ft_made = self._simulate_free_throws(shooter, 1)
                shooter_stats['fta'] = 1
                shooter_stats['ftm'] = ft_made
                shooter_stats['pts'] = pts + ft_made
                points += ft_made
        else:
            # Missed shot - foul or rebound
            if foul_on_shot:
                # Shooting foul on miss
                num_fts = 3 if is_three else 2
                ft_made = self._simulate_free_throws(shooter, num_fts)
                shooter_stats['fta'] = num_fts
                shooter_stats['ftm'] = ft_made
                shooter_stats['pts'] = ft_made
                points = ft_made
            else:
                # Rebound
                if random.random() < offense.orb_pct:
                    reb_idx = self._select_rebounder(offense, is_offensive=True)
                    stat_updates['offense'][reb_idx] = stat_updates['offense'].get(reb_idx, {})
                    stat_updates['offense'][reb_idx]['orb'] = 1
                    stat_updates['offense'][reb_idx]['reb'] = 1
                else:
                    reb_idx = self._select_rebounder(defense, is_offensive=False)
                    stat_updates['defense'][reb_idx] = stat_updates['defense'].get(reb_idx, {})
                    stat_updates['defense'][reb_idx]['drb'] = 1
                    stat_updates['defense'][reb_idx]['reb'] = 1

        stat_updates['offense'][shooter_idx] = stat_updates['offense'].get(shooter_idx, {})
        for k, v in shooter_stats.items():
            stat_updates['offense'][shooter_idx][k] = stat_updates['offense'][shooter_idx].get(k, 0) + v

        return points, stat_updates

    def _select_rebounder(self, team: TeamStats, is_offensive: bool) -> int:
        """Select rebounder based on rebounding rates."""
        weights = []
        for player in team.players:
            reb_rate = player.orb if is_offensive else player.drb
            weight = reb_rate * player.availability
            weights.append(weight)

        if sum(weights) == 0:
            return 0

        probs = [w / sum(weights) for w in weights]
        return np.random.choice(len(team.players), p=probs)

    def _select_assister(self, team: TeamStats, shooter_idx: int) -> Optional[int]:
        """Select who made the assist."""
        weights = []
        for i, player in enumerate(team.players):
            if i == shooter_idx:
                weights.append(0)
            else:
                weights.append(player.ast * player.availability)

        if sum(weights) == 0:
            return None

        probs = [w / sum(weights) for w in weights]
        return np.random.choice(len(team.players), p=probs)

    def _apply_stat_updates(
        self,
        game_state: GameState,
        stat_updates: Dict,
        is_home_offense: bool
    ):
        """Apply stat updates to game state."""
        if is_home_offense:
            off_stats = game_state.home_player_stats
            def_stats = game_state.away_player_stats
        else:
            off_stats = game_state.away_player_stats
            def_stats = game_state.home_player_stats

        for player_idx, updates in stat_updates.get('offense', {}).items():
            for stat, value in updates.items():
                off_stats[player_idx][stat] = off_stats[player_idx].get(stat, 0) + value

        for player_idx, updates in stat_updates.get('defense', {}).items():
            for stat, value in updates.items():
                def_stats[player_idx][stat] = def_stats[player_idx].get(stat, 0) + value

    def simulate_game(self) -> Dict:
        """
        Simulate a single complete game.

        Returns:
            Dictionary with scores and player stats
        """
        game_state = GameState()
        game_state.home_player_stats = self._init_player_stats(self.home)
        game_state.away_player_stats = self._init_player_stats(self.away)

        # Simulate 4 quarters
        for quarter in range(1, 5):
            game_state.quarter = quarter
            game_state.time_remaining = self.QUARTER_LENGTH

            # Possessions per quarter
            poss_per_quarter = self.possessions_per_team // 4

            for poss in range(poss_per_quarter * 2):  # Both teams
                is_home_offense = poss % 2 == 0

                if is_home_offense:
                    offense, defense = self.home, self.away
                else:
                    offense, defense = self.away, self.home

                points, stat_updates = self._simulate_possession(
                    offense, defense, game_state, is_home_offense
                )

                if is_home_offense:
                    game_state.home_score += points
                else:
                    game_state.away_score += points

                self._apply_stat_updates(game_state, stat_updates, is_home_offense)

                game_state.time_remaining -= self.AVG_POSSESSION_TIME / 2

        # Handle overtime if tied
        while game_state.home_score == game_state.away_score:
            for poss in range(10):  # ~5 per team in OT
                is_home_offense = poss % 2 == 0

                if is_home_offense:
                    offense, defense = self.home, self.away
                else:
                    offense, defense = self.away, self.home

                points, stat_updates = self._simulate_possession(
                    offense, defense, game_state, is_home_offense
                )

                if is_home_offense:
                    game_state.home_score += points
                else:
                    game_state.away_score += points

                self._apply_stat_updates(game_state, stat_updates, is_home_offense)

        return {
            'home_score': game_state.home_score,
            'away_score': game_state.away_score,
            'home_win': game_state.home_score > game_state.away_score,
            'margin': game_state.home_score - game_state.away_score,
            'total': game_state.home_score + game_state.away_score,
            'home_player_stats': dict(game_state.home_player_stats),
            'away_player_stats': dict(game_state.away_player_stats),
        }

    def run_simulation(self, n_simulations: int = 10000) -> Dict:
        """
        Run Monte Carlo simulation.

        Args:
            n_simulations: Number of games to simulate

        Returns:
            Aggregated results with probability distributions
        """
        self.results = []

        for _ in range(n_simulations):
            result = self.simulate_game()
            self.results.append(result)

        return self._analyze_results()

    def _analyze_results(self) -> Dict:
        """Analyze simulation results."""
        if not self.results:
            return {}

        n = len(self.results)

        home_wins = sum(1 for r in self.results if r['home_win'])
        home_scores = [r['home_score'] for r in self.results]
        away_scores = [r['away_score'] for r in self.results]
        margins = [r['margin'] for r in self.results]
        totals = [r['total'] for r in self.results]

        return {
            'n_simulations': n,

            # Win probabilities
            'home_win_prob': home_wins / n,
            'away_win_prob': 1 - (home_wins / n),

            # Score projections
            'home_score_mean': np.mean(home_scores),
            'home_score_std': np.std(home_scores),
            'away_score_mean': np.mean(away_scores),
            'away_score_std': np.std(away_scores),

            # Spread analysis
            'margin_mean': np.mean(margins),
            'margin_std': np.std(margins),
            'margin_percentiles': {
                '5': np.percentile(margins, 5),
                '25': np.percentile(margins, 25),
                '50': np.percentile(margins, 50),
                '75': np.percentile(margins, 75),
                '95': np.percentile(margins, 95),
            },

            # Total analysis
            'total_mean': np.mean(totals),
            'total_std': np.std(totals),
            'total_percentiles': {
                '5': np.percentile(totals, 5),
                '25': np.percentile(totals, 25),
                '50': np.percentile(totals, 50),
                '75': np.percentile(totals, 75),
                '95': np.percentile(totals, 95),
            },

            # Player stats (aggregated)
            'home_players': self._aggregate_player_stats('home_player_stats'),
            'away_players': self._aggregate_player_stats('away_player_stats'),
        }

    def _aggregate_player_stats(self, team_key: str) -> Dict:
        """Aggregate player stats across simulations."""
        if not self.results:
            return {}

        stats_by_player = defaultdict(lambda: defaultdict(list))

        for result in self.results:
            for player_idx, stats in result[team_key].items():
                for stat, value in stats.items():
                    stats_by_player[player_idx][stat].append(value)

        aggregated = {}
        for player_idx, stats in stats_by_player.items():
            aggregated[player_idx] = {
                stat: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'p10': np.percentile(values, 10),
                    'p90': np.percentile(values, 90),
                }
                for stat, values in stats.items()
            }

        return aggregated

    # =========================================================================
    # BETTING PROBABILITY CALCULATORS
    # =========================================================================

    def calculate_spread_probability(self, spread: float) -> Dict:
        """
        Calculate probability of covering a spread.

        Args:
            spread: Line (negative = home favored)

        Example:
            spread=-3.5: Home team favored by 3.5
            Home covers if margin > 3.5
        """
        if not self.results:
            return {'home_cover': 0.5, 'away_cover': 0.5}

        # For spread, if home is -3.5, they must win by 4+
        # margin > -spread means home covers (margin - (-spread) > 0)
        home_covers = sum(1 for r in self.results if r['margin'] > -spread)
        pushes = sum(1 for r in self.results if r['margin'] == -spread)
        n = len(self.results)

        return {
            'home_cover_prob': home_covers / n,
            'away_cover_prob': (n - home_covers - pushes) / n,
            'push_prob': pushes / n,
            'spread': spread,
            'projected_margin': np.mean([r['margin'] for r in self.results]),
        }

    def calculate_total_probability(self, total_line: float) -> Dict:
        """Calculate probability of over/under total."""
        if not self.results:
            return {'over_prob': 0.5, 'under_prob': 0.5}

        totals = [r['total'] for r in self.results]
        over = sum(1 for t in totals if t > total_line)
        push = sum(1 for t in totals if t == total_line)
        n = len(totals)

        return {
            'over_prob': over / n,
            'under_prob': (n - over - push) / n,
            'push_prob': push / n,
            'total_line': total_line,
            'projected_total': np.mean(totals),
        }

    def calculate_moneyline_probability(self) -> Dict:
        """Calculate moneyline win probabilities."""
        if not self.results:
            return {'home_prob': 0.5, 'away_prob': 0.5}

        home_wins = sum(1 for r in self.results if r['home_win'])
        n = len(self.results)

        return {
            'home_prob': home_wins / n,
            'away_prob': (n - home_wins) / n,
        }

    def calculate_prop_probability(
        self,
        player_idx: int,
        stat: str,
        line: float,
        is_home: bool = True
    ) -> Dict:
        """
        Calculate probability of player prop hitting.

        Args:
            player_idx: Index in team's player list
            stat: Stat to check (pts, reb, ast, fg3m, etc.)
            line: The prop line
            is_home: Whether player is on home team
        """
        if not self.results:
            return {'over_prob': 0.5, 'under_prob': 0.5}

        team_key = 'home_player_stats' if is_home else 'away_player_stats'

        values = []
        for result in self.results:
            if player_idx in result[team_key]:
                values.append(result[team_key][player_idx].get(stat, 0))

        if not values:
            return {'over_prob': 0.5, 'under_prob': 0.5}

        over = sum(1 for v in values if v > line)
        push = sum(1 for v in values if v == line)
        n = len(values)

        return {
            'over_prob': over / n,
            'under_prob': (n - over - push) / n,
            'push_prob': push / n,
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'line': line,
        }

    def calculate_parlay_correlation(self, bet1: Dict, bet2: Dict) -> Dict:
        """
        Calculate correlation between two bets for same-game parlay analysis.

        Determines if bets are positively/negatively correlated.
        """
        if not self.results:
            return {'correlation': 0.0}

        outcomes1 = []
        outcomes2 = []

        for result in self.results:
            o1 = self._evaluate_bet(result, bet1)
            o2 = self._evaluate_bet(result, bet2)
            outcomes1.append(1 if o1 else 0)
            outcomes2.append(1 if o2 else 0)

        n = len(outcomes1)
        both_hit = sum(1 for o1, o2 in zip(outcomes1, outcomes2) if o1 and o2)
        p1 = sum(outcomes1) / n
        p2 = sum(outcomes2) / n

        # Calculate correlation
        if len(set(outcomes1)) < 2 or len(set(outcomes2)) < 2:
            corr = 0.0
        else:
            corr = np.corrcoef(outcomes1, outcomes2)[0, 1]

        return {
            'correlation': corr if not np.isnan(corr) else 0.0,
            'both_hit_prob': both_hit / n,
            'independent_prob': p1 * p2,
            'correlation_boost': (both_hit / n) / (p1 * p2) if p1 * p2 > 0 else 1.0,
        }

    def _evaluate_bet(self, result: Dict, bet: Dict) -> bool:
        """Evaluate if a bet wins in a simulated game."""
        bet_type = bet.get('type', 'prop')

        if bet_type == 'moneyline':
            side = bet.get('side', 'home')
            return result['home_win'] if side == 'home' else not result['home_win']

        elif bet_type == 'spread':
            spread = bet.get('line', 0)
            side = bet.get('side', 'home')
            if side == 'home':
                return result['margin'] > -spread
            else:
                return result['margin'] < -spread

        elif bet_type == 'total':
            line = bet.get('line', 220)
            side = bet.get('side', 'over')
            return result['total'] > line if side == 'over' else result['total'] < line

        elif bet_type == 'prop':
            is_home = bet.get('is_home', True)
            team_key = 'home_player_stats' if is_home else 'away_player_stats'
            player_idx = bet.get('player_idx', 0)
            stat = bet.get('stat', 'pts')
            line = bet.get('line', 20)
            side = bet.get('side', 'over')

            value = result[team_key].get(player_idx, {}).get(stat, 0)
            return value > line if side == 'over' else value < line

        return False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_player_from_dict(data: Dict, team_id: int = 0) -> PlayerStats:
    """Create PlayerStats from dictionary (e.g., from API response)."""

    # Normalize percentage values
    def norm_pct(val):
        if val is None:
            return 0.0
        return val / 100.0 if val > 1.0 else val

    return PlayerStats(
        id=data.get('player_id', data.get('id', 0)),
        name=data.get('player_name', data.get('name', 'Unknown')),
        position=data.get('position', 'G'),
        team_id=team_id,
        minutes=data.get('min', data.get('minutes', 20.0)) or 20.0,
        usage_rate=data.get('usage_rate', 0.18),
        ppg=data.get('pts', data.get('ppg', 10.0)) or 10.0,
        fga=data.get('fga', 8.0) or 8.0,
        fgm=data.get('fgm', 3.5) or 3.5,
        fg_pct=norm_pct(data.get('fg_pct', 0.44)),
        fg3a=data.get('fg3a', 3.0) or 3.0,
        fg3m=data.get('fg3m', 1.0) or 1.0,
        fg3_pct=norm_pct(data.get('fg3_pct', 0.35)),
        fta=data.get('fta', 2.5) or 2.5,
        ftm=data.get('ftm', 2.0) or 2.0,
        ft_pct=norm_pct(data.get('ft_pct', 0.78)),
        orb=data.get('oreb', data.get('orb', 0.6)) or 0.6,
        drb=data.get('dreb', data.get('drb', 3.0)) or 3.0,
        reb=data.get('reb', 3.6) or 3.6,
        ast=data.get('ast', 2.5) or 2.5,
        tov=data.get('turnover', data.get('tov', 1.2)) or 1.2,
        stl=data.get('stl', 0.7) or 0.7,
        blk=data.get('blk', 0.3) or 0.3,
        is_starter=data.get('min', 20) > 28,
        availability=data.get('availability', 1.0),
    )


def create_team_from_dict(data: Dict, players: List[Dict]) -> TeamStats:
    """Create TeamStats from dictionary."""
    player_objs = [create_player_from_dict(p, data.get('id', 0)) for p in players[:12]]

    return TeamStats(
        id=data.get('id', 0),
        name=data.get('full_name', data.get('name', 'Team')),
        abbreviation=data.get('abbreviation', 'TM'),
        pace=data.get('pace', 100.0),
        off_rating=data.get('offensive_rating', data.get('off_rating', 112.0)),
        def_rating=data.get('defensive_rating', data.get('def_rating', 112.0)),
        efg_pct=data.get('efg_pct', 0.52),
        ts_pct=data.get('ts_pct', 0.56),
        orb_pct=data.get('orb_pct', 0.26),
        drb_pct=data.get('drb_pct', 0.74),
        tov_pct=data.get('tov_pct', 0.13),
        players=player_objs,
    )


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_simulation():
    """Demonstrate simulation with realistic NBA data."""

    # Create Lakers players (realistic stats)
    lakers_players = [
        PlayerStats(id=1, name="LeBron James", position="F", minutes=35.0,
                   usage_rate=0.30, ppg=25.5, fga=18.0, fg_pct=0.52,
                   fg3a=5.5, fg3_pct=0.38, fta=6.0, ft_pct=0.73,
                   orb=1.0, drb=6.5, reb=7.5, ast=8.0, tov=3.5,
                   stl=1.2, blk=0.6, is_starter=True),
        PlayerStats(id=2, name="Anthony Davis", position="C", minutes=34.0,
                   usage_rate=0.28, ppg=24.0, fga=17.0, fg_pct=0.55,
                   fg3a=2.0, fg3_pct=0.28, fta=7.0, ft_pct=0.80,
                   orb=2.5, drb=9.5, reb=12.0, ast=3.0, tov=2.0,
                   stl=1.2, blk=2.2, is_starter=True),
        PlayerStats(id=3, name="Austin Reaves", position="G", minutes=32.0,
                   usage_rate=0.22, ppg=15.0, fga=11.0, fg_pct=0.48,
                   fg3a=5.0, fg3_pct=0.40, fta=3.0, ft_pct=0.85,
                   orb=0.5, drb=3.5, reb=4.0, ast=5.0, tov=2.0,
                   stl=0.8, blk=0.3, is_starter=True),
        PlayerStats(id=4, name="D'Angelo Russell", position="G", minutes=30.0,
                   usage_rate=0.24, ppg=14.5, fga=12.0, fg_pct=0.43,
                   fg3a=6.0, fg3_pct=0.36, fta=2.0, ft_pct=0.82,
                   orb=0.3, drb=2.5, reb=2.8, ast=6.0, tov=2.5,
                   stl=0.8, blk=0.2, is_starter=True),
        PlayerStats(id=5, name="Rui Hachimura", position="F", minutes=26.0,
                   usage_rate=0.20, ppg=12.0, fga=9.0, fg_pct=0.50,
                   fg3a=2.5, fg3_pct=0.35, fta=2.5, ft_pct=0.78,
                   orb=0.8, drb=4.0, reb=4.8, ast=1.5, tov=1.0,
                   stl=0.5, blk=0.4, is_starter=True),
        # Bench
        PlayerStats(id=6, name="Gabe Vincent", position="G", minutes=18.0,
                   usage_rate=0.18, ppg=8.0, fga=6.0, fg_pct=0.42,
                   fg3a=4.0, fg3_pct=0.38, fta=1.5, ft_pct=0.85),
        PlayerStats(id=7, name="Jarred Vanderbilt", position="F", minutes=20.0,
                   usage_rate=0.12, ppg=5.0, fga=4.0, fg_pct=0.55,
                   fg3a=0.5, fg3_pct=0.25, fta=1.0, ft_pct=0.60,
                   orb=2.0, drb=4.0, reb=6.0, ast=1.5, stl=1.0, blk=0.5),
        PlayerStats(id=8, name="Cam Reddish", position="F", minutes=15.0,
                   usage_rate=0.16, ppg=6.0, fga=5.0, fg_pct=0.42,
                   fg3a=2.0, fg3_pct=0.32, fta=1.0, ft_pct=0.80),
    ]

    # Create Celtics players
    celtics_players = [
        PlayerStats(id=10, name="Jayson Tatum", position="F", minutes=36.0,
                   usage_rate=0.30, ppg=27.0, fga=20.0, fg_pct=0.47,
                   fg3a=8.0, fg3_pct=0.37, fta=6.5, ft_pct=0.83,
                   orb=1.0, drb=7.5, reb=8.5, ast=4.5, tov=2.8,
                   stl=1.0, blk=0.7, is_starter=True),
        PlayerStats(id=11, name="Jaylen Brown", position="G", minutes=34.0,
                   usage_rate=0.28, ppg=23.0, fga=17.0, fg_pct=0.49,
                   fg3a=5.0, fg3_pct=0.35, fta=4.5, ft_pct=0.72,
                   orb=1.0, drb=4.5, reb=5.5, ast=3.5, tov=2.5,
                   stl=1.0, blk=0.5, is_starter=True),
        PlayerStats(id=12, name="Derrick White", position="G", minutes=32.0,
                   usage_rate=0.20, ppg=15.5, fga=11.0, fg_pct=0.46,
                   fg3a=5.5, fg3_pct=0.40, fta=2.5, ft_pct=0.88,
                   orb=0.5, drb=3.5, reb=4.0, ast=5.0, tov=1.5,
                   stl=0.8, blk=1.0, is_starter=True),
        PlayerStats(id=13, name="Kristaps Porzingis", position="C", minutes=30.0,
                   usage_rate=0.25, ppg=20.0, fga=14.0, fg_pct=0.52,
                   fg3a=4.0, fg3_pct=0.38, fta=3.5, ft_pct=0.85,
                   orb=1.5, drb=6.0, reb=7.5, ast=2.0, tov=1.5,
                   stl=0.5, blk=2.0, is_starter=True),
        PlayerStats(id=14, name="Jrue Holiday", position="G", minutes=32.0,
                   usage_rate=0.18, ppg=12.5, fga=10.0, fg_pct=0.47,
                   fg3a=4.0, fg3_pct=0.38, fta=2.0, ft_pct=0.80,
                   orb=0.5, drb=4.5, reb=5.0, ast=5.0, tov=1.5,
                   stl=1.0, blk=0.4, is_starter=True),
        # Bench
        PlayerStats(id=15, name="Al Horford", position="C", minutes=22.0,
                   usage_rate=0.14, ppg=8.0, fga=6.0, fg_pct=0.50,
                   fg3a=3.0, fg3_pct=0.38, fta=1.0, ft_pct=0.80,
                   orb=1.0, drb=5.0, reb=6.0, ast=3.0, blk=1.0),
        PlayerStats(id=16, name="Payton Pritchard", position="G", minutes=18.0,
                   usage_rate=0.20, ppg=10.0, fga=7.0, fg_pct=0.43,
                   fg3a=5.0, fg3_pct=0.40, fta=1.0, ft_pct=0.90,
                   ast=2.5),
        PlayerStats(id=17, name="Sam Hauser", position="F", minutes=15.0,
                   usage_rate=0.15, ppg=7.0, fga=5.0, fg_pct=0.45,
                   fg3a=4.0, fg3_pct=0.42, fta=0.5, ft_pct=0.85),
    ]

    # Create teams
    lakers = TeamStats(
        id=14, name="Los Angeles Lakers", abbreviation="LAL",
        pace=100.5, off_rating=114.0, def_rating=110.0,
        orb_pct=0.27, drb_pct=0.73, tov_pct=0.12,
        players=lakers_players
    )

    celtics = TeamStats(
        id=2, name="Boston Celtics", abbreviation="BOS",
        pace=99.0, off_rating=118.5, def_rating=106.5,
        orb_pct=0.25, drb_pct=0.75, tov_pct=0.11,
        players=celtics_players
    )

    # Run simulation
    print("=" * 70)
    print("NBA MONTE CARLO SIMULATION: Lakers @ Celtics")
    print("=" * 70)

    sim = GameSimulator(celtics, lakers)  # Celtics home
    results = sim.run_simulation(n_simulations=5000)

    print(f"\nSimulations: {results['n_simulations']}")
    print(f"\nWIN PROBABILITIES:")
    print(f"  Celtics (Home): {results['home_win_prob']:.1%}")
    print(f"  Lakers (Away):  {results['away_win_prob']:.1%}")

    print(f"\nSCORE PROJECTIONS:")
    print(f"  Celtics: {results['home_score_mean']:.1f} (+/- {results['home_score_std']:.1f})")
    print(f"  Lakers:  {results['away_score_mean']:.1f} (+/- {results['away_score_std']:.1f})")

    print(f"\nSPREAD ANALYSIS:")
    print(f"  Projected Margin: BOS {results['margin_mean']:+.1f}")
    print(f"  Margin Std Dev: {results['margin_std']:.1f}")

    print(f"\nTOTAL ANALYSIS:")
    print(f"  Projected Total: {results['total_mean']:.1f}")
    print(f"  Total Std Dev: {results['total_std']:.1f}")

    # Betting probabilities
    print(f"\nBETTING PROBABILITIES:")

    spread = sim.calculate_spread_probability(-6.5)  # Celtics -6.5
    print(f"  Celtics -6.5: {spread['home_cover_prob']:.1%}")

    total = sim.calculate_total_probability(224.5)
    print(f"  Over 224.5: {total['over_prob']:.1%}")

    # Player props
    print(f"\nPLAYER PROPS (from simulation):")

    tatum_pts = sim.calculate_prop_probability(0, 'pts', 26.5, is_home=True)
    print(f"  Tatum Over 26.5 pts: {tatum_pts['over_prob']:.1%} (proj: {tatum_pts['mean']:.1f})")

    lebron_pts = sim.calculate_prop_probability(0, 'pts', 24.5, is_home=False)
    print(f"  LeBron Over 24.5 pts: {lebron_pts['over_prob']:.1%} (proj: {lebron_pts['mean']:.1f})")

    ad_reb = sim.calculate_prop_probability(1, 'reb', 11.5, is_home=False)
    print(f"  AD Over 11.5 reb: {ad_reb['over_prob']:.1%} (proj: {ad_reb['mean']:.1f})")

    # Parlay correlation
    print(f"\nPARLAY CORRELATION:")
    bet1 = {'type': 'moneyline', 'side': 'home'}
    bet2 = {'type': 'prop', 'is_home': True, 'player_idx': 0, 'stat': 'pts', 'line': 26.5, 'side': 'over'}
    corr = sim.calculate_parlay_correlation(bet1, bet2)
    print(f"  Celtics ML + Tatum Over 26.5: Correlation = {corr['correlation']:.3f}")
    print(f"    Joint hit rate: {corr['both_hit_prob']:.1%} (independent would be {corr['independent_prob']:.1%})")

    return results


if __name__ == "__main__":
    demo_simulation()
