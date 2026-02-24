"""
Prediction Logger - Log Every Prediction with Full Context

Captures:
- Player and game information
- Predicted values and probabilities
- Minutes predictions with uncertainty
- Game context (spread, total, home/away, B2B)
- Model metadata for reproducibility
"""

import hashlib
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

from .database import CalibrationDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Complete prediction record with all context."""

    # Required fields
    player_id: int
    player_name: str
    team: str
    opponent: str
    game_date: str
    prop_type: str
    predicted_value: float
    prop_line: float

    # Prediction confidence
    predicted_over_prob: Optional[float] = None
    confidence: Optional[float] = None
    edge: Optional[float] = None

    # Minutes prediction
    minutes_predicted: Optional[float] = None
    minutes_p10: Optional[float] = None
    minutes_p90: Optional[float] = None
    minutes_uncertainty: Optional[str] = None  # low/medium/high

    # Player info
    position: Optional[str] = None
    season_avg: Optional[float] = None
    recent_avg: Optional[float] = None
    vs_opponent_avg: Optional[float] = None

    # Game context
    game_id: Optional[int] = None
    is_home: Optional[bool] = None
    spread: Optional[float] = None
    total: Optional[float] = None
    is_favorite: Optional[bool] = None
    is_back_to_back: Optional[bool] = None
    days_rest: Optional[int] = None

    # Model metadata
    model_version: Optional[str] = None
    features_hash: Optional[str] = None

    # Timestamps
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return asdict(self)

    def compute_edge(self) -> float:
        """Compute edge as difference from implied probability."""
        if self.predicted_over_prob is None:
            return 0.0
        # Assuming -110 odds (implied 52.4%)
        implied_prob = 0.524
        return (self.predicted_over_prob - implied_prob) * 100


class PredictionLogger:
    """
    Logger for tracking all predictions made by the model.

    Records predictions with full context for later analysis.
    """

    # Prop type normalization
    PROP_TYPE_ALIASES = {
        'pts': 'points',
        'reb': 'rebounds',
        'ast': 'assists',
        '3pm': 'threes',
        '3pt': 'threes',
        'fg3m': 'threes',
        'pra': 'pra',
        'pts+reb': 'pts_reb',
        'pts+ast': 'pts_ast',
        'reb+ast': 'reb_ast',
        'stl': 'steals',
        'blk': 'blocks',
        'stl+blk': 'stocks',
        'tov': 'turnovers',
    }

    # Position normalization
    POSITION_GROUPS = {
        'PG': 'guard',
        'SG': 'guard',
        'G': 'guard',
        'G-F': 'guard',
        'SF': 'forward',
        'PF': 'forward',
        'F': 'forward',
        'F-G': 'forward',
        'F-C': 'forward',
        'C': 'center',
        'C-F': 'center',
    }

    def __init__(self, db: CalibrationDatabase = None):
        """
        Initialize the prediction logger.

        Args:
            db: CalibrationDatabase instance (creates new if None)
        """
        self.db = db or CalibrationDatabase()
        self._batch: list[PredictionRecord] = []
        self._batch_size = 100

    def _normalize_prop_type(self, prop_type: str) -> str:
        """Normalize prop type to standard format."""
        normalized = prop_type.lower().strip()
        return self.PROP_TYPE_ALIASES.get(normalized, normalized)

    def _normalize_position(self, position: str) -> str:
        """Normalize position to group (guard/forward/center)."""
        if not position:
            return 'unknown'
        return self.POSITION_GROUPS.get(position.upper(), 'forward')

    def _compute_features_hash(self, prediction: PredictionRecord) -> str:
        """Compute hash of key prediction features for reproducibility."""
        key_features = {
            'player_id': prediction.player_id,
            'game_date': prediction.game_date,
            'prop_type': prediction.prop_type,
            'predicted_value': prediction.predicted_value,
            'minutes_predicted': prediction.minutes_predicted,
            'spread': prediction.spread,
            'total': prediction.total,
        }
        feature_str = json.dumps(key_features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()[:12]

    def log_prediction(
        self,
        player_id: int,
        player_name: str,
        team: str,
        opponent: str,
        game_date: str,
        prop_type: str,
        predicted_value: float,
        prop_line: float,
        **kwargs
    ) -> int:
        """
        Log a single prediction.

        Args:
            player_id: Player's ID
            player_name: Player's name
            team: Player's team abbreviation
            opponent: Opponent team abbreviation
            game_date: Game date (YYYY-MM-DD)
            prop_type: Type of prop (points, rebounds, etc.)
            predicted_value: Model's predicted value
            prop_line: Betting line
            **kwargs: Additional context (confidence, minutes, game context, etc.)

        Returns:
            Database ID of inserted prediction
        """
        # Normalize prop type
        prop_type = self._normalize_prop_type(prop_type)

        # Create prediction record
        prediction = PredictionRecord(
            player_id=player_id,
            player_name=player_name,
            team=team.upper(),
            opponent=opponent.upper(),
            game_date=game_date,
            prop_type=prop_type,
            predicted_value=predicted_value,
            prop_line=prop_line,
            predicted_over_prob=kwargs.get('predicted_over_prob'),
            confidence=kwargs.get('confidence'),
            edge=kwargs.get('edge'),
            minutes_predicted=kwargs.get('minutes_predicted'),
            minutes_p10=kwargs.get('minutes_p10'),
            minutes_p90=kwargs.get('minutes_p90'),
            minutes_uncertainty=kwargs.get('minutes_uncertainty'),
            position=kwargs.get('position'),
            season_avg=kwargs.get('season_avg'),
            recent_avg=kwargs.get('recent_avg'),
            vs_opponent_avg=kwargs.get('vs_opponent_avg'),
            game_id=kwargs.get('game_id'),
            is_home=kwargs.get('is_home'),
            spread=kwargs.get('spread'),
            total=kwargs.get('total'),
            is_favorite=kwargs.get('is_favorite'),
            is_back_to_back=kwargs.get('is_back_to_back'),
            days_rest=kwargs.get('days_rest'),
            model_version=kwargs.get('model_version', 'v1.0'),
        )

        # Compute edge if not provided
        if prediction.edge is None and prediction.predicted_over_prob:
            prediction.edge = prediction.compute_edge()

        # Compute features hash
        prediction.features_hash = self._compute_features_hash(prediction)

        # Insert to database
        pred_dict = prediction.to_dict()

        # Normalize position for analysis
        if pred_dict.get('position'):
            pred_dict['position'] = self._normalize_position(pred_dict['position'])

        prediction_id = self.db.insert_prediction(pred_dict)

        logger.debug(f"Logged prediction {prediction_id}: {player_name} {prop_type} {predicted_value}")

        return prediction_id

    def log_batch(self, predictions: list[dict]) -> list[int]:
        """
        Log multiple predictions at once.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            List of inserted prediction IDs
        """
        ids = []
        for pred in predictions:
            try:
                pred_id = self.log_prediction(**pred)
                ids.append(pred_id)
            except Exception as e:
                logger.error(f"Error logging prediction: {e}")
                continue
        return ids

    def log_from_model_output(
        self,
        player: dict,
        prediction: dict,
        game_context: dict = None,
        model_version: str = None
    ) -> int:
        """
        Log prediction from model output format.

        This adapter converts common model output formats to the logging format.

        Args:
            player: Player data dict
            prediction: Model prediction dict
            game_context: Game context dict
            model_version: Model version string

        Returns:
            Prediction ID
        """
        game_context = game_context or {}

        return self.log_prediction(
            player_id=player.get('id') or player.get('player_id', 0),
            player_name=player.get('name') or player.get('player_name', ''),
            team=player.get('team') or player.get('team_abbrev', ''),
            opponent=game_context.get('opponent', ''),
            game_date=game_context.get('game_date', datetime.now().strftime('%Y-%m-%d')),
            prop_type=prediction.get('prop_type', ''),
            predicted_value=prediction.get('predicted') or prediction.get('value', 0),
            prop_line=prediction.get('line', 0),
            predicted_over_prob=prediction.get('over_prob') or prediction.get('probability'),
            confidence=prediction.get('confidence'),
            edge=prediction.get('edge'),
            minutes_predicted=prediction.get('minutes') or player.get('projected_minutes'),
            minutes_p10=prediction.get('minutes_p10'),
            minutes_p90=prediction.get('minutes_p90'),
            minutes_uncertainty=prediction.get('minutes_uncertainty'),
            position=player.get('position'),
            season_avg=player.get('season_avg') or player.get('season_averages', {}).get(prediction.get('prop_type')),
            recent_avg=player.get('recent_avg') or player.get('recent_averages', {}).get(prediction.get('prop_type')),
            is_home=game_context.get('is_home'),
            spread=game_context.get('spread'),
            total=game_context.get('total'),
            is_favorite=game_context.get('is_favorite'),
            is_back_to_back=game_context.get('is_b2b') or game_context.get('is_back_to_back'),
            days_rest=game_context.get('days_rest'),
            model_version=model_version,
        )

    def get_pending_predictions(self, game_date: str) -> list[dict]:
        """
        Get all pending predictions for a game date.

        Args:
            game_date: Date to query (YYYY-MM-DD)

        Returns:
            List of pending prediction records
        """
        return self.db.get_pending_predictions(game_date)

    def get_prediction(self, prediction_id: int) -> Optional[dict]:
        """Get a specific prediction by ID."""
        return self.db.get_prediction(prediction_id)


# Convenience function for quick logging
def log_prediction(**kwargs) -> int:
    """Quick prediction logging without instantiating logger."""
    logger_instance = PredictionLogger()
    return logger_instance.log_prediction(**kwargs)


if __name__ == "__main__":
    # Test the prediction logger
    logger_instance = PredictionLogger()

    # Log a test prediction
    pred_id = logger_instance.log_prediction(
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
        minutes_predicted=35.0,
        minutes_uncertainty="low",
        position="F",
        is_home=True,
        spread=-3.5,
        total=225.5,
        is_favorite=True,
        is_back_to_back=False,
        days_rest=2,
        season_avg=25.8,
        recent_avg=28.2,
    )

    print(f"Logged prediction ID: {pred_id}")

    # Retrieve it
    pred = logger_instance.get_prediction(pred_id)
    print(f"\nRetrieved prediction:")
    print(f"  Player: {pred['player_name']}")
    print(f"  Prop: {pred['prop_type']} = {pred['predicted_value']}")
    print(f"  Line: {pred['prop_line']}")
    print(f"  Over Prob: {pred['predicted_over_prob']}")
    print(f"  Features Hash: {pred['features_hash']}")
