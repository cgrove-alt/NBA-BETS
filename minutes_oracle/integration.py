"""
Minutes Oracle Integration Module

Shows how to integrate the Minutes Oracle with existing prop prediction pipeline.

Usage:
    from minutes_oracle.integration import MinutesAwarePropPredictor

    predictor = MinutesAwarePropPredictor()
    predictor.load_models()

    # Predict with minutes awareness
    result = predictor.predict_player_props(
        player_id=203999,
        game_context={'vegas_spread': -5.5, 'vegas_total': 225.5, ...}
    )
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Any
from pathlib import Path
from dataclasses import dataclass

from .minutes_predictor import MinutesPredictor, MinutesDistribution
from .minutes_features import MinutesFeatureGenerator


@dataclass
class MinutesAwarePropPrediction:
    """Prop prediction with minutes distribution context."""
    prop_type: str
    predicted_value: float
    prop_line: float | None
    prediction: str  # 'over' or 'under'
    confidence: float
    edge: float

    # Minutes context
    minutes_distribution: MinutesDistribution
    minutes_adjusted_value: float
    minutes_uncertainty_penalty: float

    # Original prediction (without minutes adjustment)
    original_predicted_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            'prop_type': self.prop_type,
            'predicted_value': round(self.predicted_value, 1),
            'prop_line': self.prop_line,
            'prediction': self.prediction,
            'confidence': round(self.confidence, 3),
            'edge': round(self.edge, 2),
            'minutes': {
                'p50': round(self.minutes_distribution.p50, 1),
                'uncertainty': self.minutes_distribution.uncertainty,
                'spread': round(self.minutes_distribution.spread, 1),
            },
            'minutes_adjusted': round(self.minutes_adjusted_value, 1),
            'original_value': round(self.original_predicted_value, 1),
            'uncertainty_penalty': round(self.minutes_uncertainty_penalty, 3),
        }


class MinutesAwarePropPredictor:
    """
    Prop predictor that integrates Minutes Oracle for uncertainty-aware predictions.

    This class wraps existing prop models and adjusts predictions based on
    minutes distribution uncertainty.
    """

    # Confidence penalty factors by uncertainty level
    UNCERTAINTY_PENALTIES = {
        'low': 1.0,      # No penalty for stable players
        'medium': 0.9,   # 10% confidence reduction
        'high': 0.75,    # 25% confidence reduction
    }

    def __init__(self,
                 minutes_model_path: str = 'models/minutes_oracle.pkl',
                 prop_models_dir: str = 'models'):
        """
        Initialize the predictor.

        Args:
            minutes_model_path: Path to trained Minutes Oracle
            prop_models_dir: Directory containing prop models
        """
        self.minutes_model_path = minutes_model_path
        self.prop_models_dir = Path(prop_models_dir)

        self.minutes_predictor: MinutesPredictor | None = None
        self.prop_models: dict[str, Any] = {}
        self.feature_gen = MinutesFeatureGenerator()

        self._loaded = False

    def load_models(self):
        """Load all required models."""
        print("Loading models...")

        # Load Minutes Oracle
        minutes_path = Path(self.minutes_model_path)
        if minutes_path.exists():
            self.minutes_predictor = MinutesPredictor.load(minutes_path)
            print(f"  Minutes Oracle loaded from {minutes_path}")
        else:
            print(f"  Warning: Minutes Oracle not found at {minutes_path}")

        # Load prop models (use existing loader pattern)
        prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']
        for prop_type in prop_types:
            model_path = self.prop_models_dir / f'player_{prop_type}_ensemble.pkl'
            if model_path.exists():
                import pickle
                with open(model_path, 'rb') as f:
                    self.prop_models[prop_type] = pickle.load(f)
                print(f"  {prop_type.capitalize()} model loaded")

        self._loaded = True
        print("Models loaded successfully.")

    def predict_minutes(self,
                        player_id: int,
                        team_id: int,
                        opponent_team_id: int,
                        game_date: str,
                        game_context: dict,
                        player_game_logs: list[dict] | None = None) -> MinutesDistribution:
        """
        Predict minutes distribution for a player.

        Args:
            player_id: Player ID
            team_id: Player's team ID
            opponent_team_id: Opponent team ID
            game_date: Game date
            game_context: Pre-game context
            player_game_logs: Player's recent game logs

        Returns:
            MinutesDistribution
        """
        if not self.minutes_predictor:
            # Return default distribution if no model
            return MinutesDistribution(
                p10=25.0, p25=28.0, p50=32.0, p75=35.0, p90=38.0,
                expected=32.0, uncertainty='medium', spread=13.0,
                player_id=player_id
            )

        # Generate features
        features = self.feature_gen.generate_features(
            player_id=player_id,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            game_date=game_date,
            game_context=game_context,
            player_game_logs=player_game_logs,
        )

        # Predict
        return self.minutes_predictor.predict(features, player_id=player_id)

    def predict_player_props(self,
                              player_id: int,
                              team_id: int,
                              opponent_team_id: int,
                              game_date: str,
                              game_context: dict,
                              prop_lines: dict[str, float] | None = None,
                              player_game_logs: list[dict] | None = None,
                              prop_features: dict[str, dict] | None = None) -> dict[str, MinutesAwarePropPrediction]:
        """
        Generate prop predictions with minutes-aware adjustments.

        Args:
            player_id: Player ID
            team_id: Player's team ID
            opponent_team_id: Opponent team ID
            game_date: Game date
            game_context: Pre-game context
            prop_lines: Optional dict of prop_type -> line value
            player_game_logs: Player's recent game logs
            prop_features: Pre-computed prop features by type

        Returns:
            Dict of prop_type -> MinutesAwarePropPrediction
        """
        if not self._loaded:
            self.load_models()

        prop_lines = prop_lines or {}
        results = {}

        # Step 1: Predict minutes distribution
        minutes_dist = self.predict_minutes(
            player_id=player_id,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            game_date=game_date,
            game_context=game_context,
            player_game_logs=player_game_logs
        )

        # Step 2: Get uncertainty penalty
        uncertainty_penalty = self.UNCERTAINTY_PENALTIES.get(
            minutes_dist.uncertainty, 0.9
        )

        # Step 3: Generate predictions for each prop type
        for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
            if prop_type not in self.prop_models:
                continue

            model = self.prop_models[prop_type]
            prop_line = prop_lines.get(prop_type)

            # Get features for this prop
            if prop_features and prop_type in prop_features:
                features = prop_features[prop_type]
            else:
                # Would normally call feature_engineering.generate_*_prop_features()
                # For now, create a minimal feature set
                features = self._create_minimal_prop_features(
                    player_id, prop_type, game_context, player_game_logs
                )

            # Add minutes-related features
            features['predicted_minutes_p50'] = minutes_dist.p50
            features['predicted_minutes_uncertainty'] = minutes_dist.spread
            features['minutes_floor'] = minutes_dist.p10
            features['minutes_ceiling'] = minutes_dist.p90

            # Get original prediction
            try:
                if hasattr(model, 'predict'):
                    import pandas as pd
                    features_df = pd.DataFrame([features])
                    original_pred = model.predict(features_df)[0]
                else:
                    original_pred = 0.0
            except Exception:
                original_pred = self._get_stat_baseline(prop_type, player_game_logs)

            # Step 4: Adjust prediction based on minutes
            # If expected minutes differ from baseline, scale prediction
            baseline_mins = features.get('season_min_avg', 32.0)
            if baseline_mins > 0:
                mins_ratio = minutes_dist.p50 / baseline_mins
                adjusted_pred = original_pred * mins_ratio
            else:
                adjusted_pred = original_pred

            # Step 5: Calculate confidence with uncertainty penalty
            if prop_line is not None:
                edge = adjusted_pred - prop_line
                edge_pct = abs(edge) / max(prop_line, 1)
                base_confidence = 0.5 + (edge_pct * 0.5)
                base_confidence = min(0.85, max(0.5, base_confidence))
            else:
                edge = 0.0
                base_confidence = 0.6

            final_confidence = base_confidence * uncertainty_penalty

            # Determine over/under
            prediction = 'over' if adjusted_pred > (prop_line or adjusted_pred) else 'under'

            results[prop_type] = MinutesAwarePropPrediction(
                prop_type=prop_type,
                predicted_value=adjusted_pred,
                prop_line=prop_line,
                prediction=prediction,
                confidence=final_confidence,
                edge=edge,
                minutes_distribution=minutes_dist,
                minutes_adjusted_value=adjusted_pred,
                minutes_uncertainty_penalty=1.0 - uncertainty_penalty,
                original_predicted_value=original_pred,
            )

        return results

    def _create_minimal_prop_features(self,
                                       player_id: int,
                                       prop_type: str,
                                       game_context: dict,
                                       game_logs: list[dict] | None) -> dict:
        """Create minimal features for a prop prediction."""
        features = {
            'player_id': player_id,
            'is_home': 1 if game_context.get('is_home') else 0,
            'opp_def_rating': 114.0,
            'opp_pace': 100.0,
        }

        # Add season averages from game logs
        if game_logs:
            stat_map = {
                'points': 'pts',
                'rebounds': 'reb',
                'assists': 'ast',
                'threes': 'fg3m',
                'pra': None,  # Calculated
            }

            stat_key = stat_map.get(prop_type)
            if stat_key:
                values = [g.get(stat_key, 0) for g in game_logs[-10:] if g.get(stat_key) is not None]
                if values:
                    features[f'season_{prop_type[:3]}_avg'] = np.mean(values)
                    features[f'recent_{prop_type[:3]}_avg'] = np.mean(values[-5:]) if len(values) >= 5 else np.mean(values)
            elif prop_type == 'pra':
                pra_values = [
                    (g.get('pts', 0) or 0) + (g.get('reb', 0) or 0) + (g.get('ast', 0) or 0)
                    for g in game_logs[-10:]
                ]
                if pra_values:
                    features['season_pra_avg'] = np.mean(pra_values)
                    features['recent_pra_avg'] = np.mean(pra_values[-5:]) if len(pra_values) >= 5 else np.mean(pra_values)

            # Minutes
            mins = [self._parse_minutes(g.get('min')) for g in game_logs[-10:]]
            mins = [m for m in mins if m > 0]
            if mins:
                features['season_min_avg'] = np.mean(mins)

        return features

    def _get_stat_baseline(self, prop_type: str, game_logs: list[dict] | None) -> float:
        """Get baseline stat value from game logs."""
        if not game_logs:
            return {'points': 15.0, 'rebounds': 5.0, 'assists': 3.0, 'threes': 1.5, 'pra': 23.0}.get(prop_type, 10.0)

        stat_map = {'points': 'pts', 'rebounds': 'reb', 'assists': 'ast', 'threes': 'fg3m'}
        if prop_type in stat_map:
            values = [g.get(stat_map[prop_type], 0) for g in game_logs[-10:]]
            return np.mean(values) if values else 10.0
        elif prop_type == 'pra':
            values = [(g.get('pts', 0) or 0) + (g.get('reb', 0) or 0) + (g.get('ast', 0) or 0) for g in game_logs[-10:]]
            return np.mean(values) if values else 23.0
        return 10.0

    def _parse_minutes(self, value) -> float:
        """Parse minutes value."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            if ':' in value:
                parts = value.split(':')
                try:
                    return int(parts[0]) + int(parts[1]) / 60
                except (ValueError, IndexError):
                    return 0.0
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_integration():
    """
    Example showing how to use the Minutes Oracle in prop predictions.
    """
    print("=" * 60)
    print("MINUTES ORACLE INTEGRATION EXAMPLE")
    print("=" * 60)

    # Initialize predictor
    predictor = MinutesAwarePropPredictor()

    # Check if models exist
    if not Path('models/minutes_oracle.pkl').exists():
        print("\nMinutes Oracle not trained yet.")
        print("Run: python -m minutes_oracle.minutes_trainer")
        return

    predictor.load_models()

    # Example: Predict for a player
    player_id = 203999  # Nikola Jokic
    team_id = 1610612743  # Denver Nuggets
    opponent_team_id = 1610612747  # Lakers
    game_date = '2026-01-30'

    game_context = {
        'vegas_spread': -5.5,  # Nuggets favored by 5.5
        'vegas_total': 225.5,
        'is_home': True,
        'is_back_to_back': False,
        'days_rest': 2,
    }

    # Mock game logs (in practice, fetch from API)
    game_logs = [
        {'min': '35:42', 'pts': 28, 'reb': 12, 'ast': 9, 'fg3m': 2},
        {'min': '34:15', 'pts': 25, 'reb': 14, 'ast': 11, 'fg3m': 1},
        {'min': '36:30', 'pts': 31, 'reb': 10, 'ast': 8, 'fg3m': 3},
        {'min': '33:00', 'pts': 22, 'reb': 13, 'ast': 10, 'fg3m': 2},
        {'min': '35:00', 'pts': 27, 'reb': 11, 'ast': 12, 'fg3m': 1},
    ]

    prop_lines = {
        'points': 26.5,
        'rebounds': 11.5,
        'assists': 9.5,
        'pra': 47.5,
    }

    print(f"\nPredicting for Player ID {player_id}")
    print(f"Game: vs {opponent_team_id} on {game_date}")
    print(f"Context: Spread={game_context['vegas_spread']}, Total={game_context['vegas_total']}")

    # Get predictions
    results = predictor.predict_player_props(
        player_id=player_id,
        team_id=team_id,
        opponent_team_id=opponent_team_id,
        game_date=game_date,
        game_context=game_context,
        prop_lines=prop_lines,
        player_game_logs=game_logs,
    )

    # Print results
    print("\n" + "-" * 60)
    print("PREDICTIONS:")
    print("-" * 60)

    # First print minutes prediction
    if results:
        first_result = list(results.values())[0]
        mins = first_result.minutes_distribution
        print("\nMinutes Distribution:")
        print(f"  P10 (floor): {mins.p10:.1f}")
        print(f"  P50 (median): {mins.p50:.1f}")
        print(f"  P90 (ceiling): {mins.p90:.1f}")
        print(f"  Uncertainty: {mins.uncertainty}")
        print(f"  Spread (P90-P10): {mins.spread:.1f}")

    print("\nProp Predictions:")
    for prop_type, result in results.items():
        print(f"\n  {prop_type.upper()}:")
        print(f"    Line: {result.prop_line}")
        print(f"    Prediction: {result.prediction.upper()}")
        print(f"    Predicted Value: {result.predicted_value:.1f}")
        print(f"    Original (no mins adj): {result.original_predicted_value:.1f}")
        print(f"    Confidence: {result.confidence:.1%}")
        print(f"    Edge: {result.edge:+.1f}")
        if result.minutes_uncertainty_penalty > 0:
            print(f"    Minutes Uncertainty Penalty: -{result.minutes_uncertainty_penalty:.1%}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    example_integration()
