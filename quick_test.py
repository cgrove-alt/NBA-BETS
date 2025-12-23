#!/usr/bin/env python3
"""Quick test of the player prop models."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")

def load_model(name):
    """Load a model from disk."""
    path = MODEL_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def predict_with_model(model_data, features):
    """Make a prediction using the model structure from training."""
    if isinstance(model_data, dict) and 'models' in model_data and 'meta_model' in model_data:
        # New stacked ensemble format
        base_models = model_data['models']
        meta_model = model_data['meta_model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']

        # Build feature array in correct order
        X = pd.DataFrame([features])
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names].fillna(0)

        X_scaled = scaler.transform(X)

        # Get base model predictions
        base_preds = []
        for name, model in base_models.items():
            pred = model.predict(X_scaled)[0]
            base_preds.append(pred)

        # Use meta model for stacking
        if meta_model is not None:
            meta_features = np.array(base_preds).reshape(1, -1)
            return float(meta_model.predict(meta_features)[0])
        else:
            return float(np.mean(base_preds))

    elif hasattr(model_data, 'predict'):
        X = pd.DataFrame([features])
        return float(model_data.predict(X)[0])

    else:
        # Legacy format
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']

        X = pd.DataFrame([features])
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names].fillna(0)

        X_scaled = scaler.transform(X)
        return float(model.predict(X_scaled)[0])

def test_models():
    """Test all player prop models."""
    print("=" * 60)
    print("NBA PLAYER PROPS MODEL - Quick Test")
    print("=" * 60)

    # Load models
    models = {}
    for prop in ['points', 'rebounds', 'assists', 'threes', 'pra']:
        model = load_model(f'player_{prop}_ensemble')
        if model:
            models[prop] = model
            print(f"✅ Loaded {prop} model")
        else:
            print(f"❌ Failed to load {prop} model")

    minutes_model = load_model('player_minutes_model')
    if minutes_model:
        print(f"✅ Loaded minutes model")

    print()

    # Get feature names from one model
    if 'points' in models:
        model_data = models['points']
        if isinstance(model_data, dict):
            feature_names = model_data.get('feature_names', [])
            print(f"Model uses {len(feature_names)} features")

    print()

    # Define test player profiles
    test_players = [
        {
            "name": "Star Guard (like Curry/Tatum)",
            "features": {
                'season_games': 25, 'season_pts_avg': 28.5, 'season_reb_avg': 5.8,
                'season_ast_avg': 6.2, 'season_fg3m_avg': 4.5, 'season_min_avg': 35.0,
                'recent_pts_avg': 30.0, 'recent_pts_std': 7.0, 'recent_pts_min': 18,
                'recent_pts_max': 45, 'recent_reb_avg': 6.0, 'recent_reb_std': 2.0,
                'recent_ast_avg': 6.5, 'recent_ast_std': 2.5, 'recent_fg3m_avg': 4.8,
                'recent_fg3m_std': 2.0, 'recent_min_avg': 36.0, 'recent_min_std': 3.0,
                'pts_trend': 2.0, 'reb_trend': 0.2, 'ast_trend': 0.3, 'fg3m_trend': 0.3,
                'min_trend': 1.0, 'home_pts_avg': 29.0, 'away_pts_avg': 28.0,
                'home_reb_avg': 5.9, 'away_reb_avg': 5.7, 'home_ast_avg': 6.3,
                'away_ast_avg': 6.1, 'home_fg3m_avg': 4.6, 'away_fg3m_avg': 4.4,
                'min_consistency': 0.92, 'is_home': 1, 'days_rest': 2, 'is_back_to_back': 0,
                'opp_def_rating': 112.0, 'opp_pace': 100.5, 'opp_pts_allowed': 113.0,
                'opp_reb_rate': 49.5, 'opp_def_strength': 0.01,
                'is_guard': 1, 'is_forward': 0, 'is_center': 0, 'is_starter': 1, 'is_star': 1,
                'usage_rate': 0.32, 'ts_pct': 0.62, 'efg_pct': 0.58, 'fg3_rate': 0.45,
                'fta_rate': 0.28, 'pra_avg': 40.5, 'fg3a_per_min': 0.18, 'fg3a_avg': 9.5,
                'regressed_fg3_pct': 0.42, 'expected_fg3m': 4.0, 'is_volume_shooter': 1,
            }
        },
        {
            "name": "Starting Center (like Gobert/Adebayo)",
            "features": {
                'season_games': 25, 'season_pts_avg': 14.5, 'season_reb_avg': 12.5,
                'season_ast_avg': 2.8, 'season_fg3m_avg': 0.1, 'season_min_avg': 32.0,
                'recent_pts_avg': 15.0, 'recent_pts_std': 4.0, 'recent_pts_min': 8,
                'recent_pts_max': 22, 'recent_reb_avg': 13.0, 'recent_reb_std': 3.5,
                'recent_ast_avg': 3.0, 'recent_ast_std': 1.5, 'recent_fg3m_avg': 0.1,
                'recent_fg3m_std': 0.3, 'recent_min_avg': 33.0, 'recent_min_std': 4.0,
                'pts_trend': 0.5, 'reb_trend': 0.5, 'ast_trend': 0.2, 'fg3m_trend': 0.0,
                'min_trend': 1.0, 'home_pts_avg': 14.8, 'away_pts_avg': 14.2,
                'home_reb_avg': 12.8, 'away_reb_avg': 12.2, 'home_ast_avg': 2.9,
                'away_ast_avg': 2.7, 'home_fg3m_avg': 0.1, 'away_fg3m_avg': 0.1,
                'min_consistency': 0.85, 'is_home': 1, 'days_rest': 1, 'is_back_to_back': 0,
                'opp_def_rating': 115.0, 'opp_pace': 98.0, 'opp_pts_allowed': 116.0,
                'opp_reb_rate': 51.0, 'opp_def_strength': 0.03,
                'is_guard': 0, 'is_forward': 0, 'is_center': 1, 'is_starter': 1, 'is_star': 0,
                'usage_rate': 0.18, 'ts_pct': 0.65, 'efg_pct': 0.62, 'fg3_rate': 0.02,
                'fta_rate': 0.35, 'pra_avg': 29.8, 'fg3a_per_min': 0.01, 'fg3a_avg': 0.2,
                'regressed_fg3_pct': 0.30, 'expected_fg3m': 0.1, 'is_volume_shooter': 0,
            }
        },
        {
            "name": "Bench Guard (low minutes)",
            "features": {
                'season_games': 20, 'season_pts_avg': 8.5, 'season_reb_avg': 2.0,
                'season_ast_avg': 2.5, 'season_fg3m_avg': 1.2, 'season_min_avg': 18.0,
                'recent_pts_avg': 9.0, 'recent_pts_std': 5.0, 'recent_pts_min': 2,
                'recent_pts_max': 18, 'recent_reb_avg': 2.2, 'recent_reb_std': 1.5,
                'recent_ast_avg': 2.8, 'recent_ast_std': 1.8, 'recent_fg3m_avg': 1.4,
                'recent_fg3m_std': 1.2, 'recent_min_avg': 19.0, 'recent_min_std': 6.0,
                'pts_trend': 0.5, 'reb_trend': 0.2, 'ast_trend': 0.3, 'fg3m_trend': 0.2,
                'min_trend': 1.0, 'home_pts_avg': 8.8, 'away_pts_avg': 8.2,
                'home_reb_avg': 2.1, 'away_reb_avg': 1.9, 'home_ast_avg': 2.6,
                'away_ast_avg': 2.4, 'home_fg3m_avg': 1.3, 'away_fg3m_avg': 1.1,
                'min_consistency': 0.65, 'is_home': 0, 'days_rest': 3, 'is_back_to_back': 0,
                'opp_def_rating': 110.0, 'opp_pace': 102.0, 'opp_pts_allowed': 111.0,
                'opp_reb_rate': 49.0, 'opp_def_strength': -0.02,
                'is_guard': 1, 'is_forward': 0, 'is_center': 0, 'is_starter': 0, 'is_star': 0,
                'usage_rate': 0.20, 'ts_pct': 0.55, 'efg_pct': 0.50, 'fg3_rate': 0.40,
                'fta_rate': 0.20, 'pra_avg': 13.0, 'fg3a_per_min': 0.15, 'fg3a_avg': 3.0,
                'regressed_fg3_pct': 0.36, 'expected_fg3m': 1.1, 'is_volume_shooter': 0,
            }
        },
    ]

    for player in test_players:
        print(f"\n📊 {player['name']}")
        print("-" * 50)

        for prop, model_data in models.items():
            try:
                pred = predict_with_model(model_data, player['features'])
                # Clamp predictions to reasonable ranges
                if prop == 'threes':
                    pred = max(0, min(pred, 12))
                elif prop == 'rebounds':
                    pred = max(0, min(pred, 25))
                elif prop == 'assists':
                    pred = max(0, min(pred, 20))
                elif prop == 'points':
                    pred = max(0, min(pred, 60))
                elif prop == 'pra':
                    pred = max(0, min(pred, 80))

                print(f"  {prop.upper():10s}: {pred:5.1f}")
            except Exception as e:
                print(f"  {prop.upper():10s}: Error - {e}")

    print()
    print("=" * 60)
    print("Model test complete!")
    print("For real predictions, visit the dashboard at http://localhost:8050")
    print("=" * 60)

if __name__ == "__main__":
    test_models()
