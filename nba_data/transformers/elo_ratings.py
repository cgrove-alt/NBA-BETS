"""
Elo Rating System for NBA Teams

Provides dynamic team strength ratings that can be queried point-in-time,
making them suitable for backtesting without temporal leakage.

Key Features:
1. Elo ratings updated after each game
2. Home court advantage adjustment
3. Margin of victory modifier
4. Mean-reversion between seasons
5. Pythagorean expectation for expected win%
6. Point-in-time queries for historical predictions

Reference: FiveThirtyEight's NBA Elo methodology
"""

from __future__ import annotations

import math
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Elo configuration
DEFAULT_ELO = 1500  # Starting Elo for new teams
K_FACTOR = 20  # Base K-factor for rating updates
HOME_ADVANTAGE = 100  # Home team Elo boost (~3.5 points)
SEASON_REGRESSION = 0.33  # Regress 33% toward mean between seasons

# Margin of victory constants (for MOV multiplier)
# Based on FiveThirtyEight's formula
MOV_MULTIPLIER_BASE = 2.2  # Multiplier for a large victory

# Pythagorean exponent (basketball uses ~13.91, simplified to 14)
PYTHAGOREAN_EXPONENT = 13.91


@dataclass
class EloRating:
    """Represents a team's Elo rating at a point in time."""
    team_id: int
    team_name: str
    rating: float
    games_played: int
    wins: int
    losses: int
    points_for: int
    points_against: int
    last_updated: str  # ISO date string

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EloUpdate:
    """Record of a single Elo update after a game."""
    game_id: str
    game_date: str
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    home_score: int
    away_score: int
    home_elo_before: float
    away_elo_before: float
    home_elo_after: float
    away_elo_after: float
    home_expected_win_prob: float
    home_actual_win: bool
    elo_change: float  # Amount home team gained/lost

    def to_dict(self) -> dict:
        return asdict(self)


class EloRatingSystem:
    """
    NBA Elo Rating System with point-in-time queries.

    Usage:
        elo = EloRatingSystem()

        # Process historical games
        for game in historical_games:
            elo.update_from_game(game)

        # Query ratings at a specific date (for predictions)
        home_elo, away_elo = elo.get_ratings_at_date(
            home_team_id, away_team_id, game_date
        )

        # Calculate win probability
        home_prob = elo.predict_home_win_probability(home_elo, away_elo)
    """

    def __init__(
        self,
        k_factor: float = K_FACTOR,
        home_advantage: float = HOME_ADVANTAGE,
        default_elo: float = DEFAULT_ELO,
        use_mov_multiplier: bool = True,
        season_regression: float = SEASON_REGRESSION,
    ):
        """
        Initialize the Elo rating system.

        Args:
            k_factor: Controls how quickly ratings change
            home_advantage: Elo points added for home team
            default_elo: Starting rating for new teams
            use_mov_multiplier: Whether to adjust for margin of victory
            season_regression: How much to regress toward mean between seasons
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.default_elo = default_elo
        self.use_mov_multiplier = use_mov_multiplier
        self.season_regression = season_regression

        # Current ratings (most recent)
        self.ratings: dict[int, EloRating] = {}

        # Historical ratings: {team_id: [(date, rating), ...]}
        self.rating_history: dict[int, list[tuple[str, float]]] = defaultdict(list)

        # Game-by-game updates for auditing
        self.updates: list[EloUpdate] = []

        # Season tracking for regression
        self.current_season: str | None = None

    def get_team_rating(self, team_id: int, team_name: str = None) -> EloRating:
        """Get current rating for a team, creating if necessary."""
        if team_id not in self.ratings:
            self.ratings[team_id] = EloRating(
                team_id=team_id,
                team_name=team_name or f"Team_{team_id}",
                rating=self.default_elo,
                games_played=0,
                wins=0,
                losses=0,
                points_for=0,
                points_against=0,
                last_updated=datetime.now().strftime("%Y-%m-%d"),
            )
        return self.ratings[team_id]

    def get_ratings_at_date(
        self,
        home_team_id: int,
        away_team_id: int,
        date: str
    ) -> tuple[float, float]:
        """
        Get team ratings as of a specific date.

        TEMPORAL DISCIPLINE: This is safe for backtesting because it only
        uses ratings computed from games BEFORE the specified date.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            date: Date string (YYYY-MM-DD)

        Returns:
            Tuple of (home_elo, away_elo) as of that date
        """
        home_elo = self._get_rating_at_date(home_team_id, date)
        away_elo = self._get_rating_at_date(away_team_id, date)
        return home_elo, away_elo

    def _get_rating_at_date(self, team_id: int, date: str) -> float:
        """Get a team's rating as of a specific date."""
        if team_id not in self.rating_history:
            return self.default_elo

        history = self.rating_history[team_id]
        if not history:
            return self.default_elo

        # Find most recent rating before the specified date
        for hist_date, rating in reversed(history):
            if hist_date < date:
                return rating

        # No history before date - return default
        return self.default_elo

    def calculate_expected_win_prob(
        self,
        team_elo: float,
        opponent_elo: float,
        is_home: bool = False
    ) -> float:
        """
        Calculate expected win probability using Elo formula.

        Args:
            team_elo: Team's Elo rating
            opponent_elo: Opponent's Elo rating
            is_home: Whether team is home (adds home advantage)

        Returns:
            Probability of team winning (0-1)
        """
        effective_elo = team_elo
        if is_home:
            effective_elo += self.home_advantage

        elo_diff = effective_elo - opponent_elo
        return 1.0 / (1.0 + 10 ** (-elo_diff / 400))

    def predict_home_win_probability(
        self,
        home_elo: float,
        away_elo: float
    ) -> float:
        """
        Predict probability of home team winning.

        Args:
            home_elo: Home team's base Elo
            away_elo: Away team's base Elo

        Returns:
            Home win probability (0-1)
        """
        return self.calculate_expected_win_prob(home_elo, away_elo, is_home=True)

    def predict_spread(
        self,
        home_elo: float,
        away_elo: float
    ) -> float:
        """
        Predict the point spread from home team's perspective.

        Conversion: ~25 Elo points = 1 point of spread
        (Based on historical NBA data)

        Args:
            home_elo: Home team's Elo
            away_elo: Away team's Elo

        Returns:
            Predicted margin for home team (positive = home favored)
        """
        # Add home advantage to get effective ratings
        effective_home = home_elo + self.home_advantage
        elo_diff = effective_home - away_elo

        # Convert Elo difference to points (~25 Elo = 1 point)
        return elo_diff / 25.0

    def _calculate_mov_multiplier(
        self,
        winner_elo: float,
        loser_elo: float,
        mov: int
    ) -> float:
        """
        Calculate margin of victory multiplier.

        Based on FiveThirtyEight's formula - gives more credit for bigger
        wins but with diminishing returns, and adjusts for upset vs expected.
        """
        if not self.use_mov_multiplier:
            return 1.0

        # Elo difference (positive if favorite won)
        elo_diff = winner_elo - loser_elo

        # FiveThirtyEight formula
        multiplier = math.log(abs(mov) + 1) * (MOV_MULTIPLIER_BASE / (0.001 * elo_diff + MOV_MULTIPLIER_BASE))

        return max(0.5, min(multiplier, 3.0))  # Cap between 0.5 and 3.0

    def update_from_game(
        self,
        game_id: str,
        game_date: str,
        home_team_id: int,
        away_team_id: int,
        home_score: int,
        away_score: int,
        home_team_name: str = None,
        away_team_name: str = None,
        season: str = None
    ) -> EloUpdate:
        """
        Update ratings based on a completed game.

        Args:
            game_id: Unique game identifier
            game_date: Date of game (YYYY-MM-DD)
            home_team_id: Home team ID
            away_team_id: Away team ID
            home_score: Home team final score
            away_score: Away team final score
            home_team_name: Optional home team name
            away_team_name: Optional away team name
            season: Optional season string for regression tracking

        Returns:
            EloUpdate record
        """
        # Handle season change
        if season and season != self.current_season:
            if self.current_season is not None:
                self._apply_season_regression()
            self.current_season = season

        # Get current ratings
        home_rating = self.get_team_rating(home_team_id, home_team_name)
        away_rating = self.get_team_rating(away_team_id, away_team_name)

        home_elo_before = home_rating.rating
        away_elo_before = away_rating.rating

        # Calculate expected win probability
        home_expected = self.calculate_expected_win_prob(
            home_elo_before, away_elo_before, is_home=True
        )

        # Determine winner
        home_won = home_score > away_score
        home_actual = 1.0 if home_won else 0.0

        # Calculate margin of victory multiplier
        mov = abs(home_score - away_score)
        if home_won:
            mov_mult = self._calculate_mov_multiplier(
                home_elo_before + self.home_advantage, away_elo_before, mov
            )
        else:
            mov_mult = self._calculate_mov_multiplier(
                away_elo_before, home_elo_before + self.home_advantage, mov
            )

        # Calculate Elo change
        elo_change = self.k_factor * mov_mult * (home_actual - home_expected)

        # Update ratings
        home_rating.rating += elo_change
        away_rating.rating -= elo_change

        # Update stats
        home_rating.games_played += 1
        away_rating.games_played += 1
        home_rating.points_for += home_score
        home_rating.points_against += away_score
        away_rating.points_for += away_score
        away_rating.points_against += home_score

        if home_won:
            home_rating.wins += 1
            away_rating.losses += 1
        else:
            home_rating.losses += 1
            away_rating.wins += 1

        home_rating.last_updated = game_date
        away_rating.last_updated = game_date

        # Record history
        self.rating_history[home_team_id].append((game_date, home_rating.rating))
        self.rating_history[away_team_id].append((game_date, away_rating.rating))

        # Create update record
        update = EloUpdate(
            game_id=game_id,
            game_date=game_date,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_team_name=home_rating.team_name,
            away_team_name=away_rating.team_name,
            home_score=home_score,
            away_score=away_score,
            home_elo_before=home_elo_before,
            away_elo_before=away_elo_before,
            home_elo_after=home_rating.rating,
            away_elo_after=away_rating.rating,
            home_expected_win_prob=home_expected,
            home_actual_win=home_won,
            elo_change=elo_change,
        )

        self.updates.append(update)
        return update

    def _apply_season_regression(self) -> None:
        """Apply mean regression between seasons."""
        mean_elo = sum(r.rating for r in self.ratings.values()) / max(len(self.ratings), 1)

        for team_id, rating in self.ratings.items():
            old_rating = rating.rating
            new_rating = old_rating + self.season_regression * (mean_elo - old_rating)
            rating.rating = new_rating

            # Record the regression
            date = datetime.now().strftime("%Y-%m-%d")
            self.rating_history[team_id].append((f"{date}_regression", new_rating))

    def get_standings(self) -> list[EloRating]:
        """Get all teams sorted by Elo rating."""
        return sorted(self.ratings.values(), key=lambda x: x.rating, reverse=True)

    def get_elo_momentum(self, team_id: int, num_games: int = 10) -> float:
        """
        Calculate Elo momentum (change over last N games).

        Momentum captures recent form - teams on hot streaks will have
        positive momentum, cold streaks negative.

        Args:
            team_id: Team ID
            num_games: Number of recent games to consider

        Returns:
            Elo change over last N games (positive = improving)
        """
        if team_id not in self.rating_history:
            return 0.0

        history = self.rating_history[team_id]
        if len(history) < 2:
            return 0.0

        # Get last N+1 ratings (to calculate change over N games)
        recent = history[-(num_games + 1):]
        if len(recent) < 2:
            return 0.0

        # Filter out regression entries
        recent = [(d, r) for d, r in recent if '_regression' not in d]
        if len(recent) < 2:
            return 0.0

        old_rating = recent[0][1]
        new_rating = recent[-1][1]

        return new_rating - old_rating

    def get_elo_volatility(self, team_id: int, num_games: int = 10) -> float:
        """
        Calculate Elo volatility (standard deviation of recent changes).

        High volatility teams are less predictable.

        Args:
            team_id: Team ID
            num_games: Number of recent games to consider

        Returns:
            Standard deviation of Elo changes
        """
        if team_id not in self.rating_history:
            return 0.0

        history = self.rating_history[team_id]
        if len(history) < 3:
            return 0.0

        # Filter out regression entries
        recent = [(d, r) for d, r in history[-(num_games + 1):] if '_regression' not in d]
        if len(recent) < 3:
            return 0.0

        # Calculate game-to-game changes
        changes = []
        for i in range(1, len(recent)):
            change = recent[i][1] - recent[i - 1][1]
            changes.append(change)

        if not changes:
            return 0.0

        # Standard deviation
        mean = sum(changes) / len(changes)
        variance = sum((c - mean) ** 2 for c in changes) / len(changes)
        return math.sqrt(variance)

    def get_pythagorean_expectation(self, team_id: int) -> float:
        """
        Calculate expected win percentage using Pythagorean formula.

        More predictive of future performance than actual win%.

        Formula: PF^exp / (PF^exp + PA^exp)
        """
        if team_id not in self.ratings:
            return 0.5

        rating = self.ratings[team_id]
        if rating.points_for == 0 and rating.points_against == 0:
            return 0.5

        pf = max(rating.points_for, 1)
        pa = max(rating.points_against, 1)

        pf_exp = pf ** PYTHAGOREAN_EXPONENT
        pa_exp = pa ** PYTHAGOREAN_EXPONENT

        return pf_exp / (pf_exp + pa_exp)

    def save_to_file(self, filepath: str) -> None:
        """Save the entire Elo system state to JSON."""
        data = {
            "config": {
                "k_factor": self.k_factor,
                "home_advantage": self.home_advantage,
                "default_elo": self.default_elo,
                "use_mov_multiplier": self.use_mov_multiplier,
                "season_regression": self.season_regression,
            },
            "current_season": self.current_season,
            "ratings": {str(k): v.to_dict() for k, v in self.ratings.items()},
            "rating_history": {
                str(k): v for k, v in self.rating_history.items()
            },
            "updates_count": len(self.updates),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "EloRatingSystem":
        """Load Elo system from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        config = data.get("config", {})
        system = cls(
            k_factor=config.get("k_factor", K_FACTOR),
            home_advantage=config.get("home_advantage", HOME_ADVANTAGE),
            default_elo=config.get("default_elo", DEFAULT_ELO),
            use_mov_multiplier=config.get("use_mov_multiplier", True),
            season_regression=config.get("season_regression", SEASON_REGRESSION),
        )

        system.current_season = data.get("current_season")

        # Load ratings
        for team_id_str, rating_dict in data.get("ratings", {}).items():
            team_id = int(team_id_str)
            system.ratings[team_id] = EloRating(**rating_dict)

        # Load history
        for team_id_str, history in data.get("rating_history", {}).items():
            team_id = int(team_id_str)
            system.rating_history[team_id] = [(h[0], h[1]) for h in history]

        return system


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def elo_to_spread(elo_diff: float, home_advantage: float = HOME_ADVANTAGE) -> float:
    """Convert Elo difference to point spread."""
    return (elo_diff + home_advantage) / 25.0


def spread_to_elo(spread: float, home_advantage: float = HOME_ADVANTAGE) -> float:
    """Convert point spread to Elo difference."""
    return spread * 25.0 - home_advantage


def elo_to_moneyline(win_prob: float) -> int:
    """Convert win probability to American odds."""
    if win_prob <= 0 or win_prob >= 1:
        return 0

    if win_prob > 0.5:
        return int(-100 * win_prob / (1 - win_prob))
    return int(100 * (1 - win_prob) / win_prob)


def build_elo_from_games(games: list[dict]) -> EloRatingSystem:
    """
    Build an Elo rating system from a list of historical games.

    Args:
        games: List of game dicts with keys:
            - game_id, game_date, home_team_id, away_team_id,
            - home_score, away_score, home_team_name (optional), away_team_name (optional)

    Returns:
        Initialized EloRatingSystem
    """
    elo = EloRatingSystem()

    # Sort by date
    sorted_games = sorted(games, key=lambda g: g.get("game_date", "1900-01-01"))

    for game in sorted_games:
        elo.update_from_game(
            game_id=game["game_id"],
            game_date=game["game_date"],
            home_team_id=game["home_team_id"],
            away_team_id=game["away_team_id"],
            home_score=game["home_score"],
            away_score=game["away_score"],
            home_team_name=game.get("home_team_name"),
            away_team_name=game.get("away_team_name"),
        )

    return elo


# =============================================================================
# FEATURE GENERATION FOR ML MODELS
# =============================================================================

def generate_elo_features(
    elo_system: EloRatingSystem,
    home_team_id: int,
    away_team_id: int,
    game_date: str,
    market_spread: float | None = None,
    market_home_prob: float | None = None
) -> dict[str, float]:
    """
    Generate Elo-based features for ML models.

    TEMPORAL DISCIPLINE: Uses get_ratings_at_date to ensure no leakage.

    Args:
        elo_system: Initialized Elo system with historical data
        home_team_id: Home team ID
        away_team_id: Away team ID
        game_date: Date of prediction (YYYY-MM-DD)
        market_spread: Optional market spread (for disagreement features)
        market_home_prob: Optional market-implied home win probability

    Returns:
        Dict of features suitable for ML models
    """
    # Get point-in-time ratings
    home_elo, away_elo = elo_system.get_ratings_at_date(
        home_team_id, away_team_id, game_date
    )

    # Calculate probabilities
    home_win_prob = elo_system.predict_home_win_probability(home_elo, away_elo)
    predicted_spread = elo_system.predict_spread(home_elo, away_elo)

    # Pythagorean expectations (if available)
    home_pyth = elo_system.get_pythagorean_expectation(home_team_id)
    away_pyth = elo_system.get_pythagorean_expectation(away_team_id)

    # MOMENTUM FEATURES (Elo change over recent games)
    home_momentum_10 = elo_system.get_elo_momentum(home_team_id, 10)
    away_momentum_10 = elo_system.get_elo_momentum(away_team_id, 10)
    home_momentum_5 = elo_system.get_elo_momentum(home_team_id, 5)
    away_momentum_5 = elo_system.get_elo_momentum(away_team_id, 5)

    # VOLATILITY FEATURES (unpredictability)
    home_volatility = elo_system.get_elo_volatility(home_team_id, 10)
    away_volatility = elo_system.get_elo_volatility(away_team_id, 10)

    features = {
        # Raw Elo ratings
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "elo_diff_with_hca": home_elo - away_elo + elo_system.home_advantage,

        # Probabilities
        "elo_home_win_prob": home_win_prob,
        "elo_away_win_prob": 1 - home_win_prob,
        "elo_predicted_spread": predicted_spread,

        # Relative strength
        "home_elo_vs_avg": home_elo - DEFAULT_ELO,
        "away_elo_vs_avg": away_elo - DEFAULT_ELO,

        # Pythagorean
        "home_pythagorean": home_pyth,
        "away_pythagorean": away_pyth,
        "pythagorean_diff": home_pyth - away_pyth,

        # MOMENTUM: Recent form (positive = improving)
        "home_elo_momentum_10": home_momentum_10,
        "away_elo_momentum_10": away_momentum_10,
        "momentum_diff_10": home_momentum_10 - away_momentum_10,
        "home_elo_momentum_5": home_momentum_5,
        "away_elo_momentum_5": away_momentum_5,
        "momentum_diff_5": home_momentum_5 - away_momentum_5,

        # VOLATILITY: Unpredictability
        "home_elo_volatility": home_volatility,
        "away_elo_volatility": away_volatility,
        "volatility_diff": home_volatility - away_volatility,
        "total_volatility": home_volatility + away_volatility,
    }

    # MARKET DISAGREEMENT FEATURES (when market data is available)
    if market_spread is not None:
        spread_disagreement = predicted_spread - market_spread
        features["elo_vs_market_spread"] = spread_disagreement
        features["elo_market_spread_abs_diff"] = abs(spread_disagreement)
        # Positive = Elo likes home more than market does
        features["elo_has_home_edge"] = 1 if spread_disagreement > 1.5 else 0
        features["elo_has_away_edge"] = 1 if spread_disagreement < -1.5 else 0

    if market_home_prob is not None:
        prob_disagreement = home_win_prob - market_home_prob
        features["elo_vs_market_prob"] = prob_disagreement
        features["elo_market_prob_abs_diff"] = abs(prob_disagreement)
        # Significant disagreement (>5%) could indicate value
        features["elo_significant_disagreement"] = 1 if abs(prob_disagreement) > 0.05 else 0

    return features


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NBA Elo Rating System Demo")
    print("=" * 60)

    # Create system
    elo = EloRatingSystem()

    # Simulate some games
    demo_games = [
        {"game_id": "1", "game_date": "2024-10-22", "home_team_id": 1, "away_team_id": 2, "home_score": 110, "away_score": 105, "home_team_name": "Lakers", "away_team_name": "Warriors"},
        {"game_id": "2", "game_date": "2024-10-23", "home_team_id": 2, "away_team_id": 3, "home_score": 108, "away_score": 102, "home_team_name": "Warriors", "away_team_name": "Celtics"},
        {"game_id": "3", "game_date": "2024-10-24", "home_team_id": 3, "away_team_id": 1, "home_score": 115, "away_score": 120, "home_team_name": "Celtics", "away_team_name": "Lakers"},
        {"game_id": "4", "game_date": "2024-10-25", "home_team_id": 1, "away_team_id": 3, "home_score": 112, "away_score": 108, "home_team_name": "Lakers", "away_team_name": "Celtics"},
    ]

    print("\nProcessing games...")
    for game in demo_games:
        update = elo.update_from_game(**game)
        print(f"  {update.home_team_name} vs {update.away_team_name}: "
              f"{update.home_score}-{update.away_score}, "
              f"Elo change: {update.elo_change:+.1f}")

    print("\nCurrent Standings:")
    for i, team in enumerate(elo.get_standings(), 1):
        pyth = elo.get_pythagorean_expectation(team.team_id)
        print(f"  {i}. {team.team_name}: {team.rating:.0f} Elo, "
              f"{team.wins}-{team.losses}, Pyth: {pyth:.3f}")

    print("\nSample Prediction (Lakers @ Warriors on 2024-10-26):")
    features = generate_elo_features(elo, 2, 1, "2024-10-26")
    print(f"  Home Win Prob: {features['elo_home_win_prob']:.1%}")
    print(f"  Predicted Spread: {features['elo_predicted_spread']:+.1f}")
    print(f"  Elo Diff: {features['elo_diff']:.0f}")
