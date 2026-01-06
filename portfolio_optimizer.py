"""
NBA Portfolio Optimization

Covariance-aware bet sizing using Multivariate Kelly Criterion.

=============================================================================
THE PROBLEM
=============================================================================
Independent Kelly sizing ignores correlations between bets:
- "Lakers ML" and "LeBron Over" are highly correlated
- Betting both at full Kelly overexposes to Lakers performing well
- Uncorrelated bets (different games) should get larger allocations

SOLUTION: Multivariate Kelly with Covariance Matrix
=============================================================================

=============================================================================
KELLY CRITERION OVERVIEW
=============================================================================
Single bet Kelly: f* = (bp - q) / b
    where b = decimal odds - 1
          p = probability of winning
          q = 1 - p

Multi-bet Kelly: Maximize E[log(wealth)] subject to constraints
    - Requires solving quadratic optimization problem
    - Accounts for bet correlations via covariance matrix
=============================================================================
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Risk management parameters
MAX_SINGLE_BET = 0.05  # Maximum 5% of bankroll on single bet
MAX_TOTAL_EXPOSURE = 0.25  # Maximum 25% of bankroll at risk
MAX_CORRELATED_EXPOSURE = 0.15  # Maximum 15% on correlated bets
MIN_BET_SIZE = 0.005  # Minimum 0.5% to place bet
KELLY_FRACTION = 0.25  # Use quarter Kelly for safety

# Correlation classification
CORRELATION_SAME_GAME = 0.6  # Same game bets highly correlated
CORRELATION_SAME_TEAM = 0.4  # Same team different games
CORRELATION_SAME_PLAYER = 0.5  # Same player props correlated
CORRELATION_OPPOSITE_SIDES = -0.3  # Opposite sides negatively correlated


class BetType(Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"
    PARLAY = "parlay"


@dataclass
class Bet:
    """A single bet to be optimized."""
    id: str
    game_id: str
    bet_type: BetType
    selection: str  # e.g., "Lakers ML", "LeBron Over 25.5"
    odds: int  # American odds
    probability: float  # Model probability
    edge: float  # probability - implied_probability

    # Additional context for correlation calculation
    team: Optional[str] = None
    player: Optional[str] = None
    side: Optional[str] = None  # 'home', 'away', 'over', 'under'

    # Optimization results
    kelly_fraction: float = 0.0
    optimal_stake: float = 0.0
    final_stake: float = 0.0

    @property
    def decimal_odds(self) -> float:
        """Convert American odds to decimal."""
        if self.odds > 0:
            return 1 + self.odds / 100
        else:
            return 1 + 100 / abs(self.odds)

    @property
    def implied_probability(self) -> float:
        """Get implied probability from odds."""
        if self.odds > 0:
            return 100 / (self.odds + 100)
        else:
            return abs(self.odds) / (abs(self.odds) + 100)


@dataclass
class PortfolioResult:
    """Results from portfolio optimization."""
    bets: List[Bet]
    total_stake: float
    expected_return: float
    expected_variance: float
    sharpe_ratio: float
    max_drawdown_risk: float
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict:
        return {
            'bets': [
                {
                    'id': b.id,
                    'selection': b.selection,
                    'odds': b.odds,
                    'probability': b.probability,
                    'edge': b.edge,
                    'kelly_fraction': b.kelly_fraction,
                    'optimal_stake': b.optimal_stake,
                    'final_stake': b.final_stake,
                }
                for b in self.bets
            ],
            'total_stake': self.total_stake,
            'expected_return': self.expected_return,
            'expected_variance': self.expected_variance,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_risk': self.max_drawdown_risk,
        }


# =============================================================================
# COVARIANCE CALCULATOR
# =============================================================================

class CovarianceCalculator:
    """
    Calculate covariance matrix for a set of bets.

    Covariance determines how bet outcomes move together:
    - Positive: Bets tend to win/lose together
    - Negative: When one wins, the other tends to lose
    - Zero: Independent outcomes
    """

    # Historical correlation data (can be updated from backtesting)
    HISTORICAL_CORRELATIONS = {
        # Same game correlations
        ('moneyline', 'spread', 'same_game', 'same_side'): 0.85,
        ('moneyline', 'spread', 'same_game', 'opposite_side'): -0.85,
        ('moneyline', 'total', 'same_game', 'same_side'): 0.15,
        ('spread', 'total', 'same_game', 'same_side'): 0.10,

        # Player props with team outcomes
        ('moneyline', 'player_prop', 'same_game', 'same_team'): 0.45,
        ('moneyline', 'player_prop', 'same_game', 'opposite_team'): -0.25,
        ('spread', 'player_prop', 'same_game', 'same_team'): 0.35,

        # Player props with each other
        ('player_prop', 'player_prop', 'same_game', 'same_player'): 0.60,
        ('player_prop', 'player_prop', 'same_game', 'same_team'): 0.25,
        ('player_prop', 'player_prop', 'same_game', 'different_team'): -0.10,

        # Cross-game correlations (generally low)
        ('moneyline', 'moneyline', 'different_game', 'same_team'): 0.15,
        ('spread', 'spread', 'different_game', 'same_team'): 0.12,
        ('player_prop', 'player_prop', 'different_game', 'same_player'): 0.30,
    }

    def __init__(self):
        self.custom_correlations = {}

    def calculate_covariance(self, bets: List[Bet]) -> np.ndarray:
        """
        Calculate covariance matrix for a list of bets.

        Returns:
            n x n covariance matrix where n = len(bets)
        """
        n = len(bets)
        if n == 0:
            return np.array([])

        # Initialize with identity (uncorrelated)
        cov_matrix = np.eye(n)

        # Calculate pairwise correlations
        for i in range(n):
            for j in range(i + 1, n):
                corr = self._calculate_correlation(bets[i], bets[j])

                # Convert correlation to covariance
                # Cov(X,Y) = Corr(X,Y) * StdDev(X) * StdDev(Y)
                std_i = self._bet_std(bets[i])
                std_j = self._bet_std(bets[j])

                cov = corr * std_i * std_j
                cov_matrix[i, j] = cov
                cov_matrix[j, i] = cov

        # Ensure positive semi-definite
        cov_matrix = self._ensure_psd(cov_matrix)

        return cov_matrix

    def _calculate_correlation(self, bet1: Bet, bet2: Bet) -> float:
        """Calculate correlation between two bets."""

        # Same bet
        if bet1.id == bet2.id:
            return 1.0

        # Determine relationship
        same_game = bet1.game_id == bet2.game_id
        same_team = bet1.team and bet2.team and bet1.team == bet2.team
        same_player = bet1.player and bet2.player and bet1.player == bet2.player
        same_side = bet1.side == bet2.side
        opposite_side = (
            (bet1.side in ['home', 'over'] and bet2.side in ['away', 'under']) or
            (bet1.side in ['away', 'under'] and bet2.side in ['home', 'over'])
        )

        # Look up base correlation
        type1, type2 = bet1.bet_type.value, bet2.bet_type.value

        # Game context
        if same_game:
            game_context = 'same_game'
        else:
            game_context = 'different_game'

        # Side context
        if same_player:
            side_context = 'same_player'
        elif same_team:
            side_context = 'same_team'
        elif same_side:
            side_context = 'same_side'
        elif opposite_side:
            side_context = 'opposite_side'
        else:
            side_context = 'different_team'

        # Look up correlation
        key = (type1, type2, game_context, side_context)
        alt_key = (type2, type1, game_context, side_context)

        corr = self.HISTORICAL_CORRELATIONS.get(key)
        if corr is None:
            corr = self.HISTORICAL_CORRELATIONS.get(alt_key)
        if corr is None:
            # Default correlations based on game
            if same_game:
                if same_team:
                    corr = CORRELATION_SAME_TEAM
                elif same_player:
                    corr = CORRELATION_SAME_PLAYER
                else:
                    corr = CORRELATION_SAME_GAME * 0.3
            else:
                corr = 0.05  # Small positive correlation for NBA games

        return corr

    def _bet_std(self, bet: Bet) -> float:
        """Calculate standard deviation of bet outcome."""
        # Binary outcome variance: p(1-p)
        p = bet.probability
        variance = p * (1 - p)
        return np.sqrt(variance)

    def _ensure_psd(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive semi-definite."""
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Replace negative eigenvalues with small positive value
        eigenvalues = np.maximum(eigenvalues, 1e-8)

        # Reconstruct matrix
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def update_from_simulation(self, sim_results: Dict, bet1: Bet, bet2: Bet):
        """
        Update correlation estimate from simulation results.

        This can be called after running GameSimulator to get more
        accurate correlations for same-game bets.
        """
        # Extract joint outcomes from simulation
        # Would integrate with simulation_engine.py
        pass


# =============================================================================
# KELLY OPTIMIZER
# =============================================================================

class KellyOptimizer:
    """
    Multivariate Kelly Criterion optimization.

    Maximizes expected log growth rate while considering:
    - Bet correlations
    - Risk constraints
    - Drawdown limits
    """

    def __init__(
        self,
        max_single_bet: float = MAX_SINGLE_BET,
        max_total_exposure: float = MAX_TOTAL_EXPOSURE,
        kelly_fraction: float = KELLY_FRACTION,
        min_bet_size: float = MIN_BET_SIZE
    ):
        self.max_single_bet = max_single_bet
        self.max_total_exposure = max_total_exposure
        self.kelly_fraction = kelly_fraction
        self.min_bet_size = min_bet_size

        self.cov_calculator = CovarianceCalculator()

    def single_kelly(self, bet: Bet) -> float:
        """
        Calculate single-bet Kelly fraction.

        f* = (bp - q) / b
        where b = decimal_odds - 1, p = probability, q = 1-p
        """
        b = bet.decimal_odds - 1  # Net profit if win
        p = bet.probability
        q = 1 - p

        if b <= 0:
            return 0.0

        kelly = (b * p - q) / b

        # Apply fractional Kelly
        kelly *= self.kelly_fraction

        # Clamp to constraints
        kelly = max(0, min(kelly, self.max_single_bet))

        return kelly

    def optimize_portfolio(self, bets: List[Bet], bankroll: float = 1.0) -> PortfolioResult:
        """
        Optimize bet sizing for a portfolio of bets.

        Uses quadratic programming to maximize expected log growth
        subject to constraints.

        Args:
            bets: List of Bet objects
            bankroll: Total bankroll (default 1.0 for fractional allocation)

        Returns:
            PortfolioResult with optimized stakes
        """
        if not bets:
            return PortfolioResult(
                bets=[],
                total_stake=0,
                expected_return=0,
                expected_variance=0,
                sharpe_ratio=0,
                max_drawdown_risk=0,
            )

        n = len(bets)

        # Calculate single Kelly fractions first
        for bet in bets:
            bet.kelly_fraction = self.single_kelly(bet)

        # Filter out negative edge bets
        positive_edge_bets = [b for b in bets if b.kelly_fraction > 0]

        if not positive_edge_bets:
            return PortfolioResult(
                bets=bets,
                total_stake=0,
                expected_return=0,
                expected_variance=0,
                sharpe_ratio=0,
                max_drawdown_risk=0,
            )

        # For single bet, just use Kelly
        if len(positive_edge_bets) == 1:
            bet = positive_edge_bets[0]
            bet.optimal_stake = bet.kelly_fraction * bankroll
            bet.final_stake = bet.optimal_stake

            return PortfolioResult(
                bets=bets,
                total_stake=bet.final_stake,
                expected_return=bet.edge * bet.final_stake,
                expected_variance=bet.probability * (1 - bet.probability) * bet.final_stake ** 2,
                sharpe_ratio=bet.edge / np.sqrt(bet.probability * (1 - bet.probability)),
                max_drawdown_risk=bet.final_stake,
            )

        # Calculate covariance matrix
        cov_matrix = self.cov_calculator.calculate_covariance(positive_edge_bets)

        # Extract returns (edges)
        edges = np.array([b.edge for b in positive_edge_bets])

        # Solve optimization problem
        optimal_fractions = self._solve_optimization(
            edges, cov_matrix, positive_edge_bets
        )

        # Apply results
        total_stake = 0
        for i, bet in enumerate(positive_edge_bets):
            bet.optimal_stake = optimal_fractions[i] * bankroll
            bet.final_stake = bet.optimal_stake
            total_stake += bet.final_stake

        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_fractions, edges) * bankroll
        portfolio_variance = optimal_fractions @ cov_matrix @ optimal_fractions * bankroll ** 2
        sharpe = portfolio_return / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0

        # Estimate max drawdown risk (simplified)
        max_dd_risk = total_stake  # Worst case: all bets lose

        return PortfolioResult(
            bets=bets,
            total_stake=total_stake,
            expected_return=portfolio_return,
            expected_variance=portfolio_variance,
            sharpe_ratio=sharpe,
            max_drawdown_risk=max_dd_risk,
            correlation_matrix=cov_matrix,
        )

    def _solve_optimization(
        self,
        edges: np.ndarray,
        cov_matrix: np.ndarray,
        bets: List[Bet]
    ) -> np.ndarray:
        """
        Solve the portfolio optimization problem.

        Maximize: E[log(1 + r'f)] ≈ r'f - (1/2)f'Σf
        Subject to:
            - 0 <= f_i <= max_single_bet
            - sum(f) <= max_total_exposure
            - f_correlated <= max_correlated_exposure
        """
        n = len(edges)

        # Start with single Kelly fractions
        x0 = np.array([b.kelly_fraction for b in bets])

        # Objective: Maximize growth (equivalent to minimizing negative)
        def objective(f):
            # Approximate Kelly objective: edge - 0.5 * variance
            expected = np.dot(edges, f)
            variance = f @ cov_matrix @ f
            return -(expected - 0.5 * variance)

        # Constraints
        constraints = []

        # Total exposure constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda f: self.max_total_exposure - np.sum(f)
        })

        # Add correlated group constraints
        correlated_groups = self._find_correlated_groups(bets, cov_matrix)
        for group in correlated_groups:
            constraints.append({
                'type': 'ineq',
                'fun': lambda f, g=group: MAX_CORRELATED_EXPOSURE - np.sum(f[list(g)])
            })

        # Bounds: 0 to max_single_bet for each
        bounds = [(0, self.max_single_bet) for _ in range(n)]

        # Solve
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            if result.success:
                fractions = result.x
            else:
                # Fall back to scaled single Kelly
                fractions = x0 * self.max_total_exposure / np.sum(x0) if np.sum(x0) > 0 else x0
        except Exception:
            # Fall back to scaled single Kelly
            fractions = x0 * self.max_total_exposure / np.sum(x0) if np.sum(x0) > 0 else x0

        # Apply minimum bet size filter
        fractions[fractions < self.min_bet_size] = 0

        return fractions

    def _find_correlated_groups(
        self,
        bets: List[Bet],
        cov_matrix: np.ndarray,
        threshold: float = 0.3
    ) -> List[set]:
        """Find groups of highly correlated bets."""
        n = len(bets)
        groups = []

        # Convert covariance to correlation
        stds = np.sqrt(np.diag(cov_matrix))
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = cov_matrix / np.outer(stds, stds)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Find connected components of high correlation
        visited = set()

        for i in range(n):
            if i in visited:
                continue

            group = {i}
            queue = [i]

            while queue:
                node = queue.pop(0)
                for j in range(n):
                    if j not in visited and j not in group:
                        if abs(corr_matrix[node, j]) > threshold:
                            group.add(j)
                            queue.append(j)

            if len(group) > 1:
                groups.append(group)
                visited.update(group)

        return groups


# =============================================================================
# PORTFOLIO OPTIMIZER (Main Class)
# =============================================================================

class PortfolioOptimizer:
    """
    Main interface for portfolio optimization.

    Usage:
        optimizer = PortfolioOptimizer(bankroll=1000)

        # Add bets
        optimizer.add_bet(
            game_id="123",
            bet_type=BetType.MONEYLINE,
            selection="Lakers ML",
            odds=-150,
            probability=0.62,
            team="LAL"
        )

        # Optimize and get stakes
        result = optimizer.optimize()
        for bet in result.bets:
            print(f"{bet.selection}: ${bet.final_stake:.2f}")
    """

    def __init__(
        self,
        bankroll: float = 1000,
        max_single_bet: float = MAX_SINGLE_BET,
        max_total_exposure: float = MAX_TOTAL_EXPOSURE,
        kelly_fraction: float = KELLY_FRACTION
    ):
        self.bankroll = bankroll
        self.kelly_optimizer = KellyOptimizer(
            max_single_bet=max_single_bet,
            max_total_exposure=max_total_exposure,
            kelly_fraction=kelly_fraction
        )

        self.pending_bets: List[Bet] = []
        self._bet_counter = 0

    def add_bet(
        self,
        game_id: str,
        bet_type: BetType,
        selection: str,
        odds: int,
        probability: float,
        team: str = None,
        player: str = None,
        side: str = None
    ) -> Bet:
        """Add a bet to the portfolio for optimization."""
        self._bet_counter += 1
        bet_id = f"bet_{self._bet_counter}"

        # Calculate edge
        if odds > 0:
            implied = 100 / (odds + 100)
        else:
            implied = abs(odds) / (abs(odds) + 100)

        edge = probability - implied

        bet = Bet(
            id=bet_id,
            game_id=game_id,
            bet_type=bet_type,
            selection=selection,
            odds=odds,
            probability=probability,
            edge=edge,
            team=team,
            player=player,
            side=side,
        )

        self.pending_bets.append(bet)
        return bet

    def optimize(self) -> PortfolioResult:
        """Optimize the current portfolio."""
        result = self.kelly_optimizer.optimize_portfolio(
            self.pending_bets,
            bankroll=self.bankroll
        )
        return result

    def clear(self):
        """Clear pending bets."""
        self.pending_bets = []
        self._bet_counter = 0

    def get_bet_recommendation(self, bet: Bet) -> Dict:
        """Get recommendation for a single bet."""
        kelly = self.kelly_optimizer.single_kelly(bet)
        stake = kelly * self.bankroll

        return {
            'selection': bet.selection,
            'odds': bet.odds,
            'probability': bet.probability,
            'edge': bet.edge,
            'kelly_fraction': kelly,
            'recommended_stake': stake,
            'risk_adjusted_stake': stake * 0.5,  # Half Kelly for safety
            'max_stake': self.bankroll * MAX_SINGLE_BET,
        }


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def calculate_covariance(active_bets: List[Dict]) -> np.ndarray:
    """
    Calculate covariance matrix for active bets.

    This is the main integration point with bet_tracker.py.

    Args:
        active_bets: List of bet dictionaries with keys:
            - game_id, bet_type, team, player, side, probability

    Returns:
        Covariance matrix as numpy array
    """
    # Convert to Bet objects
    bets = []
    for i, b in enumerate(active_bets):
        bet_type = BetType(b.get('bet_type', 'moneyline'))
        bet = Bet(
            id=f"bet_{i}",
            game_id=str(b.get('game_id', '')),
            bet_type=bet_type,
            selection=b.get('selection', ''),
            odds=b.get('odds', -110),
            probability=b.get('probability', 0.5),
            edge=b.get('edge', 0.0),
            team=b.get('team'),
            player=b.get('player'),
            side=b.get('side'),
        )
        bets.append(bet)

    calculator = CovarianceCalculator()
    return calculator.calculate_covariance(bets)


def optimize_portfolio_kelly(
    bets: List[Dict],
    covariance: np.ndarray = None,
    bankroll: float = 1000
) -> Dict:
    """
    Optimize bet sizing using Kelly criterion.

    This is the main integration point with bet_tracker.py.

    Args:
        bets: List of bet dictionaries
        covariance: Pre-calculated covariance matrix (optional)
        bankroll: Total bankroll

    Returns:
        Dictionary with optimized stakes and metrics
    """
    optimizer = PortfolioOptimizer(bankroll=bankroll)

    for b in bets:
        bet_type = BetType(b.get('bet_type', 'moneyline'))
        optimizer.add_bet(
            game_id=str(b.get('game_id', '')),
            bet_type=bet_type,
            selection=b.get('selection', ''),
            odds=b.get('odds', -110),
            probability=b.get('probability', 0.5),
            team=b.get('team'),
            player=b.get('player'),
            side=b.get('side'),
        )

    result = optimizer.optimize()
    return result.to_dict()


# =============================================================================
# DEMO
# =============================================================================

def demo_portfolio_optimization():
    """Demonstrate portfolio optimization."""
    print("=" * 70)
    print("NBA PORTFOLIO OPTIMIZATION")
    print("=" * 70)

    bankroll = 1000
    optimizer = PortfolioOptimizer(bankroll=bankroll)

    # Add sample bets
    print("\n1. ADDING BETS")
    print("-" * 40)

    bets_to_add = [
        {
            'game_id': '12345',
            'bet_type': BetType.MONEYLINE,
            'selection': 'Lakers ML',
            'odds': -150,
            'probability': 0.65,  # Model says 65%, market implies 60%
            'team': 'LAL',
            'side': 'home',
        },
        {
            'game_id': '12345',
            'bet_type': BetType.PLAYER_PROP,
            'selection': 'LeBron Over 25.5 pts',
            'odds': -110,
            'probability': 0.58,  # 5.5% edge
            'team': 'LAL',
            'player': 'LeBron James',
            'side': 'over',
        },
        {
            'game_id': '12345',
            'bet_type': BetType.SPREAD,
            'selection': 'Lakers -3.5',
            'odds': -110,
            'probability': 0.55,  # Small edge
            'team': 'LAL',
            'side': 'home',
        },
        {
            'game_id': '67890',
            'bet_type': BetType.MONEYLINE,
            'selection': 'Warriors ML',
            'odds': +130,
            'probability': 0.48,  # Underdog edge
            'team': 'GSW',
            'side': 'away',
        },
        {
            'game_id': '67890',
            'bet_type': BetType.TOTAL,
            'selection': 'Over 225.5',
            'odds': -105,
            'probability': 0.56,  # Over edge
            'side': 'over',
        },
    ]

    for b in bets_to_add:
        bet = optimizer.add_bet(**b)
        print(f"  Added: {bet.selection} @ {bet.odds:+d} (Edge: {bet.edge:.1%})")

    # Optimize
    print("\n2. OPTIMIZATION RESULTS")
    print("-" * 40)

    result = optimizer.optimize()

    print(f"\n  PORTFOLIO SUMMARY:")
    print(f"    Total Stake: ${result.total_stake:.2f} / ${bankroll:.2f} ({result.total_stake/bankroll:.1%})")
    print(f"    Expected Return: ${result.expected_return:.2f}")
    print(f"    Expected Variance: ${result.expected_variance:.4f}")
    print(f"    Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"    Max Drawdown Risk: ${result.max_drawdown_risk:.2f}")

    print(f"\n  INDIVIDUAL BETS:")
    for bet in result.bets:
        if bet.final_stake > 0:
            print(f"    {bet.selection}:")
            print(f"      Odds: {bet.odds:+d}, Edge: {bet.edge:.1%}")
            print(f"      Kelly: {bet.kelly_fraction:.2%} -> Stake: ${bet.final_stake:.2f}")
        else:
            print(f"    {bet.selection}: NO BET (insufficient edge)")

    # Show correlation analysis
    print(f"\n3. CORRELATION ANALYSIS")
    print("-" * 40)

    if result.correlation_matrix.size > 0:
        corr = result.correlation_matrix
        positive_bets = [b for b in result.bets if b.final_stake > 0]

        print("\n  Correlation Matrix (top correlations):")
        for i in range(len(positive_bets)):
            for j in range(i + 1, len(positive_bets)):
                corr_val = corr[i, j]
                if abs(corr_val) > 0.1:
                    print(f"    {positive_bets[i].selection} <-> {positive_bets[j].selection}: {corr_val:.2f}")

    # Single bet Kelly comparison
    print(f"\n4. SINGLE VS PORTFOLIO KELLY")
    print("-" * 40)

    total_single_kelly = 0
    for bet in result.bets:
        single_kelly = optimizer.kelly_optimizer.single_kelly(bet)
        total_single_kelly += single_kelly
        if bet.final_stake > 0:
            print(f"    {bet.selection}:")
            print(f"      Single Kelly: {single_kelly:.2%} (${single_kelly * bankroll:.2f})")
            print(f"      Portfolio: {bet.kelly_fraction:.2%} (${bet.final_stake:.2f})")

    print(f"\n    Total Single Kelly: {total_single_kelly:.2%} (${total_single_kelly * bankroll:.2f})")
    print(f"    Total Portfolio: {result.total_stake / bankroll:.2%} (${result.total_stake:.2f})")
    print(f"    Reduction due to correlation: {(total_single_kelly - result.total_stake / bankroll) / total_single_kelly:.1%}")

    print("\nPortfolio optimizer ready!")
    return result


if __name__ == "__main__":
    demo_portfolio_optimization()
