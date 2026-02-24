"""
Outcome Tracker - Record Actual Results After Games Complete

Responsibilities:
- Match predictions to actual box score results
- Calculate hit/miss and error metrics
- Track closing line value (CLV)
- Handle edge cases (DNP, blowouts, OT)
"""

import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from .database import CalibrationDatabase

# Try to import Balldontlie for fetching actual stats
try:
    from balldontlie_api import BalldontlieAPI
    BALLDONTLIE_AVAILABLE = True
except ImportError:
    BALLDONTLIE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OutcomeRecord:
    """Record of actual outcome for a prediction."""

    prediction_id: int
    actual_value: float
    actual_minutes: Optional[float] = None

    # Result classification
    result: str = ""  # over, under, push
    hit: int = 0  # 1 if correct, 0 if wrong

    # Error metrics
    error: Optional[float] = None  # predicted - actual

    # Line movement
    closing_line: Optional[float] = None
    clv: Optional[float] = None  # Closing line value

    # Game context
    game_score_diff: Optional[int] = None  # Final margin
    player_started: Optional[bool] = None

    # Timestamps
    recorded_at: str = ""

    def __post_init__(self):
        if not self.recorded_at:
            self.recorded_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


class OutcomeTracker:
    """
    Track actual outcomes and match to predictions.
    """

    # Prop type to stat field mapping
    PROP_TO_STAT = {
        'points': 'pts',
        'rebounds': 'reb',
        'assists': 'ast',
        'threes': 'fg3m',
        'steals': 'stl',
        'blocks': 'blk',
        'turnovers': 'turnover',
        'pra': ['pts', 'reb', 'ast'],  # Combined stat
        'pts_reb': ['pts', 'reb'],
        'pts_ast': ['pts', 'ast'],
        'reb_ast': ['reb', 'ast'],
        'stocks': ['stl', 'blk'],
    }

    def __init__(self, db: CalibrationDatabase = None):
        """
        Initialize outcome tracker.

        Args:
            db: CalibrationDatabase instance
        """
        self.db = db or CalibrationDatabase()

        # Initialize Balldontlie API for fetching stats
        self._api = None
        if BALLDONTLIE_AVAILABLE:
            try:
                self._api = BalldontlieAPI()
                logger.info("Balldontlie API available for outcome tracking")
            except Exception as e:
                logger.warning(f"Could not init Balldontlie API: {e}")

    def record_outcome(
        self,
        prediction_id: int,
        actual_value: float,
        actual_minutes: float = None,
        closing_line: float = None,
        game_score_diff: int = None,
        player_started: bool = None,
    ) -> int:
        """
        Record the outcome for a prediction.

        Args:
            prediction_id: ID of the prediction
            actual_value: Actual stat value achieved
            actual_minutes: Actual minutes played
            closing_line: Closing line (for CLV calculation)
            game_score_diff: Final score differential
            player_started: Whether player started

        Returns:
            Outcome record ID
        """
        # Get the original prediction
        prediction = self.db.get_prediction(prediction_id)
        if not prediction:
            raise ValueError(f"Prediction {prediction_id} not found")

        # Calculate result
        prop_line = prediction['prop_line']
        predicted_value = prediction['predicted_value']

        if actual_value > prop_line:
            result = 'over'
        elif actual_value < prop_line:
            result = 'under'
        else:
            result = 'push'

        # Determine if prediction was correct
        predicted_over = predicted_value > prop_line
        actual_over = actual_value > prop_line

        if result == 'push':
            hit = 0  # Pushes don't count as hits
        elif predicted_over == actual_over:
            hit = 1
        else:
            hit = 0

        # Calculate error
        error = predicted_value - actual_value

        # Calculate CLV if closing line provided
        clv = None
        if closing_line is not None:
            # CLV = (closing_line - opening_line) in direction of our bet
            if predicted_over:
                clv = closing_line - prop_line  # Higher close = positive CLV for over
            else:
                clv = prop_line - closing_line  # Lower close = positive CLV for under

        # Create outcome record
        outcome = OutcomeRecord(
            prediction_id=prediction_id,
            actual_value=actual_value,
            actual_minutes=actual_minutes,
            result=result,
            hit=hit,
            error=error,
            closing_line=closing_line,
            clv=clv,
            game_score_diff=game_score_diff,
            player_started=player_started,
        )

        # Insert to database
        outcome_id = self.db.insert_outcome(outcome.to_dict())

        logger.info(
            f"Recorded outcome for prediction {prediction_id}: "
            f"{prediction['player_name']} {prediction['prop_type']} "
            f"predicted={predicted_value:.1f} actual={actual_value:.1f} "
            f"result={result} hit={hit}"
        )

        return outcome_id

    def _get_stat_value(self, stats: dict, prop_type: str) -> float:
        """
        Extract stat value from box score stats.

        Args:
            stats: Player stats dictionary
            prop_type: Prop type

        Returns:
            Stat value
        """
        mapping = self.PROP_TO_STAT.get(prop_type, prop_type)

        if isinstance(mapping, list):
            # Combined stat
            total = 0.0
            for stat_key in mapping:
                total += float(stats.get(stat_key, 0) or 0)
            return total
        else:
            return float(stats.get(mapping, 0) or 0)

    def _parse_minutes(self, min_str) -> float:
        """Parse minutes from various formats."""
        if min_str is None:
            return 0.0
        if isinstance(min_str, (int, float)):
            return float(min_str)
        if isinstance(min_str, str):
            if ':' in min_str:
                parts = min_str.split(':')
                return float(parts[0]) + float(parts[1]) / 60
            try:
                return float(min_str)
            except ValueError:
                return 0.0
        return 0.0

    def fetch_and_record_outcomes(self, game_date: str) -> dict:
        """
        Fetch actual stats and record outcomes for a game date.

        Args:
            game_date: Date to process (YYYY-MM-DD)

        Returns:
            Summary of outcomes recorded
        """
        if not self._api:
            logger.error("Balldontlie API not available for fetching outcomes")
            return {'error': 'API not available'}

        # Get pending predictions for this date
        predictions = self.db.get_pending_predictions(game_date)
        if not predictions:
            logger.info(f"No pending predictions for {game_date}")
            return {'matched': 0, 'not_found': 0, 'errors': 0}

        logger.info(f"Processing {len(predictions)} predictions for {game_date}")

        # Fetch games for this date
        try:
            games = self._api.get_games(dates=[game_date])
        except Exception as e:
            logger.error(f"Error fetching games: {e}")
            return {'error': str(e)}

        # Build game ID lookup
        game_lookup = {}
        for game in games:
            home_id = game.get('home_team', {}).get('id')
            away_id = game.get('visitor_team', {}).get('id')
            game_id = game.get('id')
            if home_id and game_id:
                game_lookup[home_id] = game
            if away_id and game_id:
                game_lookup[away_id] = game

        # Fetch all player stats for these games
        game_ids = [g.get('id') for g in games if g.get('id')]
        if not game_ids:
            logger.info(f"No completed games found for {game_date}")
            return {'matched': 0, 'not_found': 0, 'errors': 0, 'no_games': True}

        try:
            all_stats = self._api.get_player_stats(game_ids=game_ids, per_page=500)
        except Exception as e:
            logger.error(f"Error fetching player stats: {e}")
            return {'error': str(e)}

        # Build player stats lookup (player_id -> stats)
        stats_lookup = {}
        for stat in all_stats:
            player = stat.get('player', {})
            player_id = player.get('id')
            if player_id:
                stats_lookup[player_id] = stat

        # Process each prediction
        results = {
            'matched': 0,
            'not_found': 0,
            'errors': 0,
            'dnp': 0,
        }

        for pred in predictions:
            try:
                player_id = pred['player_id']
                prop_type = pred['prop_type']

                # Find player stats
                player_stats = stats_lookup.get(player_id)
                if not player_stats:
                    logger.warning(f"Stats not found for player {pred['player_name']} (ID: {player_id})")
                    results['not_found'] += 1
                    continue

                # Get actual value
                actual_value = self._get_stat_value(player_stats, prop_type)
                actual_minutes = self._parse_minutes(player_stats.get('min'))

                # Handle DNP
                if actual_minutes == 0:
                    logger.info(f"DNP: {pred['player_name']} did not play")
                    results['dnp'] += 1
                    # Still record it as an outcome (with 0 value)
                    self.record_outcome(
                        prediction_id=pred['id'],
                        actual_value=0.0,
                        actual_minutes=0.0,
                        game_score_diff=None,
                        player_started=False,
                    )
                    continue

                # Get game score differential
                game = player_stats.get('game', {})
                home_score = game.get('home_team_score', 0)
                away_score = game.get('visitor_team_score', 0)
                score_diff = abs(home_score - away_score) if home_score and away_score else None

                # Record outcome
                self.record_outcome(
                    prediction_id=pred['id'],
                    actual_value=actual_value,
                    actual_minutes=actual_minutes,
                    game_score_diff=score_diff,
                    player_started=actual_minutes >= 25,  # Rough heuristic
                )

                results['matched'] += 1

            except Exception as e:
                logger.error(f"Error recording outcome for prediction {pred['id']}: {e}")
                results['errors'] += 1

        logger.info(f"Outcome processing complete: {results}")
        return results

    def get_outcome(self, prediction_id: int) -> Optional[dict]:
        """Get outcome for a prediction."""
        return self.db.get_outcome(prediction_id)

    def expire_unmatched_predictions(self, game_date: str):
        """
        Mark unmatched predictions as expired.

        Called after outcome processing to handle predictions that
        couldn't be matched (player DNP, game postponed, etc.)

        Args:
            game_date: Date to process
        """
        pending = self.db.get_pending_predictions(game_date)
        expired_count = 0

        for pred in pending:
            # Check if already has outcome
            outcome = self.db.get_outcome(pred['id'])
            if not outcome:
                self.db.update_prediction_status(pred['id'], 'expired')
                expired_count += 1

        if expired_count:
            logger.info(f"Expired {expired_count} unmatched predictions for {game_date}")


if __name__ == "__main__":
    # Test the outcome tracker
    from .prediction_logger import PredictionLogger

    # First log a prediction
    pred_logger = PredictionLogger()
    pred_id = pred_logger.log_prediction(
        player_id=2544,
        player_name="LeBron James",
        team="LAL",
        opponent="BOS",
        game_date="2024-01-15",
        prop_type="points",
        predicted_value=27.5,
        prop_line=26.5,
        predicted_over_prob=0.58,
        confidence=65.0,
    )
    print(f"Logged prediction ID: {pred_id}")

    # Now record outcome
    tracker = OutcomeTracker()
    outcome_id = tracker.record_outcome(
        prediction_id=pred_id,
        actual_value=29.0,
        actual_minutes=35.2,
        closing_line=27.0,
        game_score_diff=8,
        player_started=True,
    )
    print(f"Recorded outcome ID: {outcome_id}")

    # Get outcome
    outcome = tracker.get_outcome(pred_id)
    print(f"\nOutcome details:")
    print(f"  Actual: {outcome['actual_value']}")
    print(f"  Result: {outcome['result']}")
    print(f"  Hit: {outcome['hit']}")
    print(f"  Error: {outcome['error']}")
    print(f"  CLV: {outcome['clv']}")
