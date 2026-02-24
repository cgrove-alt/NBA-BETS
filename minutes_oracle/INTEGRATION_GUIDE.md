# Minutes Oracle Integration Guide

This document shows the specific code changes needed to integrate the Minutes Oracle into your existing prop prediction pipeline.

## Overview

The integration adds:
1. Minutes Oracle model loading in `_load_ml_models()`
2. New `_predict_minutes_distribution()` method
3. Minutes features added to prop predictions
4. Confidence adjustment based on minutes uncertainty

---

## Step 1: Add Import at Top of data_service.py

Add this near the other imports (around line 45):

```python
# Minutes Oracle for minutes distribution prediction
try:
    from minutes_oracle import MinutesPredictor, MinutesFeatureGenerator
    from minutes_oracle.minutes_features import MINUTES_FEATURE_NAMES
    MINUTES_ORACLE_AVAILABLE = True
except ImportError:
    MinutesPredictor = None
    MinutesFeatureGenerator = None
    MINUTES_ORACLE_AVAILABLE = False
    print("Warning: minutes_oracle not available. Using fallback minutes projection.")
```

---

## Step 2: Add Instance Variables in __init__

Add these to the `__init__` method of `DataService` class (around line 680):

```python
        # Minutes Oracle for distribution-based minutes prediction
        self._minutes_oracle = None
        self._minutes_feature_gen = None
```

---

## Step 3: Load Minutes Oracle in _load_ml_models()

Add this at the end of the `_load_ml_models()` method (around line 895):

```python
        # Load Minutes Oracle for minutes distribution prediction
        if MINUTES_ORACLE_AVAILABLE:
            try:
                minutes_oracle_path = model_dir / "minutes_oracle.pkl"
                if minutes_oracle_path.exists():
                    self._minutes_oracle = MinutesPredictor.load(minutes_oracle_path)
                    self._minutes_feature_gen = MinutesFeatureGenerator()
                    print(f"Minutes Oracle loaded (interval_scale={self._minutes_oracle.interval_scale})")
                else:
                    print("Minutes Oracle model not found at models/minutes_oracle.pkl")
            except Exception as e:
                print(f"Error loading Minutes Oracle: {e}")
                self._minutes_oracle = None
```

---

## Step 4: Add _predict_minutes_distribution() Method

Add this new method after `_project_player_minutes()` (around line 3567):

```python
    def _predict_minutes_distribution(self, player: dict, game_context: dict = None) -> dict:
        """Predict minutes distribution using the Minutes Oracle.

        Returns a distribution with percentiles and uncertainty classification.

        Args:
            player: Player data dictionary
            game_context: Game context with vegas_spread, vegas_total, is_home, etc.

        Returns:
            dict with p10, p25, p50, p75, p90, expected, uncertainty, spread
        """
        # Default fallback distribution based on season average
        season_min = player.get("avg_minutes", 0) or player.get("season_averages", {}).get("min_avg", 28)
        fallback = {
            'p10': max(10, season_min - 8),
            'p25': max(15, season_min - 4),
            'p50': season_min,
            'p75': min(42, season_min + 4),
            'p90': min(48, season_min + 8),
            'expected': season_min,
            'uncertainty': 'medium',
            'spread': 16.0,
        }

        if not self._minutes_oracle or not self._minutes_feature_gen:
            return fallback

        try:
            # Extract player info
            player_id = player.get('id') or player.get('player_id', 0)
            team_id = player.get('team_id', 0)
            opponent_team_id = game_context.get('opponent_team_id', 0) if game_context else 0

            # Build game context for feature generation
            ctx = {
                'vegas_spread': game_context.get('spread', 0) if game_context else 0,
                'vegas_total': game_context.get('total', 220) if game_context else 220,
                'is_home': game_context.get('is_home', True) if game_context else True,
                'is_back_to_back': game_context.get('is_b2b', False) if game_context else False,
                'days_rest': game_context.get('days_rest', 2) if game_context else 2,
            }

            # Build player game logs from recent stats
            game_logs = []
            recent_avg = player.get('recent_averages', {})
            season_avg = player.get('season_averages', {})

            # Create synthetic game logs from averages (for feature generation)
            min_avg = recent_avg.get('min_avg', 0) or season_avg.get('min_avg', 0) or season_min
            pts_avg = recent_avg.get('pts_avg', 0) or season_avg.get('pts_avg', 0) or 0
            reb_avg = recent_avg.get('reb_avg', 0) or season_avg.get('reb_avg', 0) or 0
            ast_avg = recent_avg.get('ast_avg', 0) or season_avg.get('ast_avg', 0) or 0

            # Create 5 synthetic logs at the average
            for _ in range(5):
                game_logs.append({
                    'min': min_avg,
                    'pts': pts_avg,
                    'reb': reb_avg,
                    'ast': ast_avg,
                    'pf': 2.5,  # Average fouls
                })

            # Generate features
            from datetime import datetime
            game_date = datetime.now().strftime('%Y-%m-%d')

            features = self._minutes_feature_gen.generate_features(
                player_id=player_id,
                team_id=team_id,
                opponent_team_id=opponent_team_id,
                game_date=game_date,
                game_context=ctx,
                player_game_logs=game_logs,
            )

            # Get prediction
            result = self._minutes_oracle.predict(features, player_id=player_id)

            return result.to_dict()

        except Exception as e:
            print(f"Minutes Oracle prediction error: {e}")
            return fallback
```

---

## Step 5: Modify _predict_with_ml_model() to Use Minutes Distribution

In the `_predict_with_ml_model()` method, add minutes features to the features dict.
Find the section around line 3390 where features are being built and add:

```python
            # ============ MINUTES ORACLE: Add predicted minutes distribution features ============
            # Get minutes distribution for uncertainty-aware predictions
            minutes_dist = self._predict_minutes_distribution(player_stats, {
                'spread': opp_stats.get('spread', 0) if opp_stats else 0,
                'total': opp_stats.get('total', 220) if opp_stats else 220,
                'is_home': is_home,
                'is_b2b': is_back_to_back,
                'days_rest': days_rest,
            })

            # Add minutes distribution features
            features['predicted_minutes_p50'] = minutes_dist.get('p50', min_avg)
            features['minutes_uncertainty'] = minutes_dist.get('spread', 12.0)
            features['minutes_floor'] = minutes_dist.get('p10', min_avg - 8)
            features['minutes_ceiling'] = minutes_dist.get('p90', min_avg + 8)
```

---

## Step 6: Adjust Confidence Based on Minutes Uncertainty

In `_calculate_prop_confidence()` method (around line 2814), add uncertainty penalty:

```python
    def _calculate_prop_confidence(self, prediction: float, line: float,
                                    prop_type: str = None, player_stats: dict = None,
                                    minutes_dist: dict = None) -> float:
        """Calculate confidence for a prop prediction.

        Now includes minutes uncertainty penalty.
        """
        # ... existing confidence calculation ...

        # Apply minutes uncertainty penalty
        if minutes_dist:
            uncertainty = minutes_dist.get('uncertainty', 'medium')
            if uncertainty == 'high':
                confidence *= 0.80  # 20% penalty for high uncertainty
            elif uncertainty == 'medium':
                confidence *= 0.92  # 8% penalty for medium uncertainty
            # 'low' uncertainty = no penalty

        return confidence
```

---

## Step 7: Update _get_player_predictions() to Use Minutes Distribution

In `_get_player_predictions()` method (around line 3599), get minutes distribution early and pass it through:

```python
    def _get_player_predictions(self, player: dict, opponent_abbrev: str,
                                 is_home: bool, game_context: dict = None) -> dict:
        """Generate predictions for a single player."""

        # Get minutes distribution FIRST (it affects all prop predictions)
        minutes_dist = self._predict_minutes_distribution(player, game_context)

        # Store for use in predictions
        predicted_minutes = minutes_dist.get('p50', player.get('avg_minutes', 28))
        minutes_uncertainty = minutes_dist.get('uncertainty', 'medium')

        # ... rest of method, passing minutes_dist to confidence calculation ...
```

---

## Usage Example

After integration, the flow becomes:

```python
# 1. Load models (happens automatically in DataService.__init__)
service = DataService()

# 2. When generating predictions for a player:
player = {...}  # Player data
game_context = {
    'spread': -5.5,
    'total': 225.5,
    'is_home': True,
    'is_b2b': False,
}

# 3. Get minutes distribution
minutes_dist = service._predict_minutes_distribution(player, game_context)
# Returns: {'p10': 28.2, 'p50': 34.1, 'p90': 39.5, 'uncertainty': 'medium', ...}

# 4. Use in prop predictions
# - predicted_minutes_p50 added to features
# - Confidence adjusted based on uncertainty
```

---

## Verification

After integration, you should see in the logs:
```
Minutes Oracle loaded (interval_scale=1.15)
```

And prop predictions will now include minutes-aware confidence adjustments.
