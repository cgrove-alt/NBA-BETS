"""
Minutes Oracle - NBA Player Minutes Distribution Prediction

A dedicated minutes prediction model that predicts a distribution of minutes
(not just a point estimate) using quantile regression.

Usage:
    from minutes_oracle import MinutesPredictor, MinutesFeatureGenerator

    # Load trained model
    predictor = MinutesPredictor.load('models/minutes_oracle.pkl')

    # Generate features and predict
    feature_gen = MinutesFeatureGenerator(api_client)
    features = feature_gen.generate_features(player_id, game_context)
    minutes_dist = predictor.predict(features)

    # Returns:
    # {
    #     'p10': 28.2,   # Floor (blowout scenario)
    #     'p25': 31.5,
    #     'p50': 34.1,   # Most likely (median)
    #     'p75': 36.8,
    #     'p90': 39.5,   # Ceiling (OT/close game)
    #     'expected': 33.8,  # Weighted mean
    #     'uncertainty': 'medium',  # low/medium/high
    # }
"""

from .minutes_predictor import MinutesPredictor
from .minutes_features import MinutesFeatureGenerator
from .coach_tendencies import COACH_TENDENCIES, CoachTendencyLearner

__all__ = [
    'MinutesPredictor',
    'MinutesFeatureGenerator',
    'COACH_TENDENCIES',
    'CoachTendencyLearner',
]

__version__ = '1.0.0'
