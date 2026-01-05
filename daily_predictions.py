"""
Daily NBA Predictions - Comprehensive Betting Analysis

Generates predictions for all bet types:
- Moneyline (win probability)
- Spread (cover probability vs market line)
- Player Props (points, rebounds, assists, threes for all starters)

Uses Balldontlie API (GOAT tier) for real betting lines.

Usage:
    python3 daily_predictions.py              # Today's predictions
    python3 daily_predictions.py --date 2026-01-05  # Specific date
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import pickle
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Suppress logging noise
logging.disable(logging.WARNING)

# Import our modules
from balldontlie_api import BalldontlieAPI
from feature_engineering import generate_game_features, PlayerPropFeatureGenerator, InjuryReportManager
from scipy.stats import norm

# Global feature generator for player props (lazy loaded)
_prop_feature_gen = None
_player_feature_cache = {}  # Cache player features to avoid redundant API calls
_id_mapper = None  # IDMapper for Balldontlie player/team ID lookups
_injury_manager = None  # InjuryReportManager for injury data

def get_feature_engine(season: str = "2025-26") -> PlayerPropFeatureGenerator:
    """Get or create the feature generator for player prop features."""
    global _prop_feature_gen
    if _prop_feature_gen is None:
        _prop_feature_gen = PlayerPropFeatureGenerator(season=season)
    return _prop_feature_gen

def get_id_mapper():
    """Get or create the ID mapper for Balldontlie lookups."""
    global _id_mapper
    if _id_mapper is None:
        try:
            from id_mapping import IDMapper
            _id_mapper = IDMapper()
        except ImportError:
            return None
    return _id_mapper

def get_player_name_from_bdl_id(bdl_player_id: int) -> Optional[str]:
    """Get player name from Balldontlie player ID."""
    mapper = get_id_mapper()
    if mapper:
        return mapper.get_player_name(bdl_player_id)
    return None

def get_bdl_player_id(player_name: str) -> Optional[int]:
    """Get Balldontlie player ID from player name (fast, cached)."""
    mapper = get_id_mapper()
    if mapper:
        return mapper.get_player_id(player_name)
    return None


def get_injury_manager(season: str = "2025-26") -> InjuryReportManager:
    """Get or create the injury manager for injury data."""
    global _injury_manager
    if _injury_manager is None:
        _injury_manager = InjuryReportManager(season=season)
        try:
            _injury_manager.fetch_all_injuries()
        except Exception:
            pass  # Continue without injuries if fetch fails
    return _injury_manager


def apply_injury_adjustments(
    home_prob: float,
    away_prob: float,
    injury_features: Dict
) -> Tuple[float, float]:
    """
    Apply post-hoc injury adjustments to win probabilities.

    Since models weren't trained with injury data, we apply adjustments
    after prediction. Research shows star player injuries shift win prob 4-8%.
    """
    home_key_out = injury_features.get("home_key_players_out", 0)
    away_key_out = injury_features.get("away_key_players_out", 0)
    injury_advantage = injury_features.get("injury_advantage", 0)

    # Each key player out shifts probability ~4%
    adjustment = (away_key_out - home_key_out) * 0.04

    # Add injury advantage impact (scaled down since it's cumulative)
    if injury_advantage != 0:
        adjustment += injury_advantage * 0.002

    # Cap adjustment at +/- 15%
    adjustment = max(-0.15, min(0.15, adjustment))

    adjusted_home = home_prob + adjustment
    adjusted_home = max(0.1, min(0.9, adjusted_home))
    adjusted_away = 1 - adjusted_home

    return adjusted_home, adjusted_away

def get_cached_features(player_name: str, prop_type: str, opponent_id: int, bdl_player_id: int = None) -> dict:
    """
    Get cached features or generate new ones using Balldontlie data.

    Now uses Balldontlie IDs throughout for consistency and speed.
    Falls back to NBA API if Balldontlie data unavailable.
    """
    cache_key = f"{player_name}_{prop_type}_{opponent_id}"
    if cache_key in _player_feature_cache:
        return _player_feature_cache[cache_key]

    # Skip unknown players
    if not player_name or player_name.startswith("Player "):
        return None

    # Get Balldontlie player ID if not provided
    if bdl_player_id is None:
        bdl_player_id = get_bdl_player_id(player_name)

    fe = get_feature_engine()
    try:
        # Try to generate features using player name (works with new Balldontlie layer)
        if prop_type == 'points':
            features = fe.generate_points_prop_features(
                player_id=bdl_player_id,
                opponent_team_id=opponent_id,
            )
        elif prop_type == 'rebounds':
            features = fe.generate_rebounds_prop_features(
                player_id=bdl_player_id,
                opponent_team_id=opponent_id,
            )
        elif prop_type == 'assists':
            features = fe.generate_assists_prop_features(
                player_id=bdl_player_id,
                opponent_team_id=opponent_id,
            )
        else:
            features = fe.generate_points_prop_features(
                player_id=bdl_player_id,
                opponent_team_id=opponent_id,
            )

        if features:
            _player_feature_cache[cache_key] = features
            return features
    except Exception:
        pass

    return None

# Constants
NBA_SPREAD_VOLATILITY = 13.0  # Historical std dev of NBA margins
MODEL_DIR = Path("models")


def load_models() -> Dict:
    """Load all prediction models."""
    models = {}

    # Moneyline
    ml_path = MODEL_DIR / "moneyline_ensemble_tuned.pkl"
    if ml_path.exists():
        try:
            with open(ml_path, 'rb') as f:
                data = pickle.load(f)
                models['moneyline'] = data.get('model', data) if isinstance(data, dict) else data
        except Exception as e:
            print(f"    Warning: Could not load moneyline model: {e}")

    # Spread
    spread_path = MODEL_DIR / "spread_svm_regressor.pkl"
    if spread_path.exists():
        try:
            with open(spread_path, 'rb') as f:
                data = pickle.load(f)
                models['spread'] = data.get('model', data) if isinstance(data, dict) else data
        except Exception as e:
            print(f"    Warning: Could not load spread model: {e}")

    # Player prop models - load available models
    for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
        # Try different model files in order of preference
        model_paths = [
            MODEL_DIR / f"player_{prop_type}_line_classifier.pkl",
            MODEL_DIR / f"player_{prop_type}.pkl",  # Simple regressor
            MODEL_DIR / f"player_{prop_type}_quantile.pkl",
        ]

        for path in model_paths:
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)

                    # Handle dict format with model, scaler, feature_names
                    if isinstance(data, dict):
                        model = data.get('model')
                        scaler = data.get('scaler')
                        feature_names = data.get('feature_names', [])

                        if model and hasattr(model, 'predict'):
                            # Store as tuple for later unpacking
                            models[f'prop_{prop_type}'] = {
                                'model': model,
                                'scaler': scaler,
                                'feature_names': feature_names,
                                'prop_type': prop_type,
                            }
                            break
                    elif hasattr(data, 'predict'):
                        models[f'prop_{prop_type}'] = data
                        break
                except Exception as e:
                    continue

    return models


def get_spread_cover_probability(edge_points: float) -> float:
    """Convert point edge to cover probability using normal CDF."""
    return norm.cdf(edge_points / NBA_SPREAD_VOLATILITY)


def get_implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def predict_moneyline(features: Dict, models: Dict) -> Tuple[float, float]:
    """Predict moneyline probabilities."""
    model = models.get('moneyline')
    if not model:
        return 0.5, 0.5

    # Extract key features
    feature_cols = ['net_rating_diff', 'win_pct_diff', 'pace_diff', 'off_rating_diff',
                   'def_rating_diff', 'home_win_streak', 'away_win_streak']

    X = np.array([[features.get(col, 0) for col in feature_cols]])

    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[0]
            home_prob = probs[1] if len(probs) > 1 else probs[0]
        else:
            # Fallback to simple formula
            net_rating_diff = features.get('net_rating_diff', 0)
            home_prob = 0.5 + (net_rating_diff * 0.02)
    except Exception:
        net_rating_diff = features.get('net_rating_diff', 0)
        home_prob = 0.5 + (net_rating_diff * 0.02)

    home_prob = max(0.1, min(0.9, home_prob))
    return home_prob, 1 - home_prob


def predict_spread(features: Dict, models: Dict) -> float:
    """Predict point spread (positive = home favored)."""
    model = models.get('spread')

    # Simple formula based on net rating
    net_rating_diff = features.get('net_rating_diff', 0)
    home_advantage = 3.0

    if model:
        try:
            feature_cols = ['net_rating_diff', 'off_rating_diff', 'def_rating_diff',
                          'pace_diff', 'expected_point_diff']
            X = np.array([[features.get(col, 0) for col in feature_cols]])
            predicted = model.predict(X)[0]
            return predicted
        except Exception:
            pass

    # Fallback: net rating / 3 + home advantage
    return (net_rating_diff / 3.0) + home_advantage


def analyze_game(game: Dict, odds: Dict, models: Dict) -> Dict:
    """
    Analyze a single game with all bet types.

    Args:
        game: Game info from Balldontlie
        odds: Betting odds for this game
        models: Loaded prediction models

    Returns:
        Analysis dictionary with all predictions
    """
    home_team = game.get('home_team', {})
    away_team = game.get('visitor_team', {})

    home_abbrev = home_team.get('abbreviation', 'HOME')
    away_abbrev = away_team.get('abbreviation', 'AWAY')
    game_time = game.get('status', '')
    game_id = game.get('id')

    analysis = {
        'game_id': game_id,
        'home_team': home_abbrev,
        'away_team': away_abbrev,
        'game_time': game_time,
        'moneyline': {},
        'spread': {},
        'player_props': []
    }

    # Generate features with injury data
    injury_mgr = get_injury_manager()
    injury_features = {}
    injury_details = {'home': [], 'away': []}

    try:
        features = generate_game_features(
            home_abbrev, away_abbrev,
            season="2025-26",
            include_advanced=True,
            injury_manager=injury_mgr
        )
        ml_features = features.get('moneyline_features', {}) if features else {}

        # Extract injury features from moneyline_features (where they're stored)
        if ml_features:
            injury_features = {
                'home_injury_impact': ml_features.get('home_injury_impact', 0),
                'away_injury_impact': ml_features.get('away_injury_impact', 0),
                'home_key_players_out': ml_features.get('home_key_players_out', 0),
                'away_key_players_out': ml_features.get('away_key_players_out', 0),
                'injury_advantage': ml_features.get('injury_advantage', 0),
            }
            injury_details = ml_features.get('injury_details', {'home': [], 'away': []})
    except Exception as e:
        ml_features = {}

    if not ml_features:
        # Use basic defaults
        ml_features = {'net_rating_diff': 0, 'win_pct_diff': 0}

    # Store injury info in analysis
    analysis['injury_features'] = injury_features
    analysis['injury_details'] = injury_details

    # Moneyline prediction (with injury adjustments)
    home_prob, away_prob = predict_moneyline(ml_features, models)

    # Apply post-hoc injury adjustments
    if injury_features.get('home_key_players_out', 0) or injury_features.get('away_key_players_out', 0):
        home_prob, away_prob = apply_injury_adjustments(home_prob, away_prob, injury_features)

    # Get market odds
    home_ml_odds = odds.get('home_moneyline', -110)
    away_ml_odds = odds.get('away_moneyline', -110)
    home_implied = get_implied_probability(home_ml_odds)
    away_implied = get_implied_probability(away_ml_odds)

    analysis['moneyline'] = {
        'home_prob': home_prob,
        'away_prob': away_prob,
        'home_odds': home_ml_odds,
        'away_odds': away_ml_odds,
        'home_edge': (home_prob - home_implied) * 100,
        'away_edge': (away_prob - away_implied) * 100,
    }

    # Spread prediction
    predicted_spread = predict_spread(ml_features, models)
    market_spread = odds.get('spread', 0)  # Negative = home favored

    # Edge = model spread - market spread (if positive, home is undervalued)
    spread_edge_points = predicted_spread - market_spread
    cover_prob = get_spread_cover_probability(abs(spread_edge_points))

    # Determine which side to bet
    if spread_edge_points > 0:
        # Model thinks home covers more than market
        bet_side = f"{home_abbrev} {market_spread:+.1f}"
        edge_pct = (cover_prob - 0.524) * 100  # vs -110 implied
    else:
        # Model thinks away covers
        bet_side = f"{away_abbrev} {-market_spread:+.1f}"
        cover_prob = 1 - cover_prob
        edge_pct = (cover_prob - 0.524) * 100

    analysis['spread'] = {
        'predicted_spread': predicted_spread,
        'market_spread': market_spread,
        'spread_edge_points': spread_edge_points,
        'cover_prob': cover_prob,
        'edge_pct': edge_pct,
        'bet_side': bet_side,
    }

    return analysis


def print_game_analysis(analysis: Dict):
    """Print formatted game analysis."""
    home = analysis['home_team']
    away = analysis['away_team']
    time = analysis['game_time']

    print(f"\n{'='*65}")
    print(f"  {away} @ {home}  ({time})")
    print(f"{'='*65}")

    # Display injuries if any
    injury_details = analysis.get('injury_details', {})
    home_injured = injury_details.get('home', []) if isinstance(injury_details, dict) else []
    away_injured = injury_details.get('away', []) if isinstance(injury_details, dict) else []

    if home_injured or away_injured:
        print(f"\n  INJURIES:")
        if home_injured:
            print(f"    {home}:")
            for inj in home_injured[:5]:  # Limit to 5 per team
                player_name = inj.get('player_name', 'Unknown')
                status = inj.get('status', 'Unknown')
                print(f"      - {player_name} ({status})")
        if away_injured:
            print(f"    {away}:")
            for inj in away_injured[:5]:  # Limit to 5 per team
                player_name = inj.get('player_name', 'Unknown')
                status = inj.get('status', 'Unknown')
                print(f"      - {player_name} ({status})")

    # Moneyline
    ml = analysis['moneyline']
    home_prob = ml['home_prob']
    away_prob = ml['away_prob']
    home_edge = ml['home_edge']
    away_edge = ml['away_edge']

    ml_rec = ""
    if home_edge > 3:
        ml_rec = f">>> {home} ML"
    elif away_edge > 3:
        ml_rec = f">>> {away} ML"

    print(f"\n  MONEYLINE:")
    print(f"    {home}: {home_prob:.1%} (edge: {home_edge:+.1f}%)")
    print(f"    {away}: {away_prob:.1%} (edge: {away_edge:+.1f}%)")
    if ml_rec:
        print(f"    {ml_rec}")

    # Spread
    sp = analysis['spread']
    print(f"\n  SPREAD:")
    print(f"    Model: {home} {sp['predicted_spread']:+.1f}")
    print(f"    Market: {home} {sp['market_spread']:+.1f}")
    print(f"    Cover Prob: {sp['cover_prob']:.1%} | Edge: {sp['edge_pct']:+.1f}%")
    if abs(sp['edge_pct']) > 2:
        print(f"    >>> {sp['bet_side']}")

    # Player props (if any)
    props = analysis.get('player_props', [])
    if props:
        print(f"\n  PLAYER PROPS:")
        for prop in props:
            player = prop.get('player', 'Unknown')
            stat = prop.get('stat', '')
            line = prop.get('line', 0)
            over_prob = prop.get('over_prob', 0.5)
            edge = prop.get('edge', 0)
            predicted = prop.get('predicted_value')

            direction = "Over" if over_prob > 0.5 else "Under"
            prob = over_prob if over_prob > 0.5 else (1 - over_prob)

            marker = "**" if abs(edge) > 5 else ("*" if abs(edge) > 3 else "")

            if predicted is not None:
                print(f"    {player} {stat} {line} (pred: {predicted:.1f}): {direction} {prob:.0%} ({edge:+.1f}%) {marker}")
            else:
                print(f"    {player} {stat} {line}: {direction} {prob:.0%} ({edge:+.1f}%) {marker}")


def get_player_props_for_game(api: BalldontlieAPI, game_id: int) -> Dict[int, Dict]:
    """
    Get player props from Balldontlie API for a game.

    Returns dict indexed by player_id with prop lines.
    """
    props_by_player = {}

    try:
        props = api.get_player_props(game_id)
        if not props:
            return props_by_player

        # Group by player and prop type, preferring DraftKings
        for prop in props:
            player_id = prop.get('player_id')
            prop_type = prop.get('prop_type', '').lower()
            vendor = prop.get('vendor', '').lower()
            line = prop.get('line_value')

            if not player_id or not prop_type or not line:
                continue

            try:
                line = float(line)
            except:
                continue

            # Skip milestone props (like "18+ points" which have high odds)
            market = prop.get('market', {})
            if market.get('type') == 'milestone':
                continue

            if player_id not in props_by_player:
                props_by_player[player_id] = {'player_id': player_id}

            # Store team_id if available
            team_id = prop.get('team_id')
            if team_id:
                props_by_player[player_id]['team_id'] = team_id

            # Store if not exists or vendor is preferred
            key = f'{prop_type}_line'
            if key not in props_by_player[player_id] or vendor in ['draftkings', 'fanduel']:
                props_by_player[player_id][key] = line
                props_by_player[player_id][f'{prop_type}_vendor'] = vendor

    except Exception as e:
        print(f"    Error fetching props: {e}")

    return props_by_player


def predict_player_prop(
    player_name: str,
    player_id: int,
    prop_type: str,
    line: float,
    opponent: str,
    opponent_id: int,
    models: Dict,
    use_api_features: bool = False  # Disable by default for speed
) -> Dict:
    """
    Predict over/under probability for a player prop.

    Args:
        player_name: Player's name
        player_id: Player's ID
        prop_type: Type of prop (points, rebounds, assists, threes)
        line: The betting line
        opponent: Opponent team abbreviation
        opponent_id: Opponent team ID for matchup features
        models: Loaded models
        use_api_features: If True, fetch player stats from API (slow)

    Returns:
        Prediction dict with over_prob and edge
    """
    import pandas as pd

    over_prob = 0.5
    edge = 0.0
    predicted_value = None

    model_data = models.get(f'prop_{prop_type}')

    if model_data and use_api_features:
        try:
            # Get cached features (or generate new ones) - use Balldontlie ID for fast lookup
            features = get_cached_features(player_name, prop_type, opponent_id, bdl_player_id=player_id)

            if features:
                # Handle dict format (raw model with scaler)
                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                    scaler = model_data.get('scaler')
                    feature_names = model_data.get('feature_names', [])

                    # Build feature array matching training features
                    X = pd.DataFrame([{k: features.get(k, 0) for k in feature_names}])
                    for col in feature_names:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[feature_names].fillna(0)

                    # Scale if scaler available
                    if scaler is not None:
                        X_scaled = scaler.transform(X)
                    else:
                        X_scaled = X.values

                    # Predict (regression model predicts stat value)
                    predicted_value = float(model.predict(X_scaled)[0])

                    # Convert to probability using normal CDF
                    std = line * 0.20 if line > 0 else 5.0
                    z_score = (predicted_value - line) / max(std, 1)
                    over_prob = float(norm.cdf(z_score))
                    edge = (over_prob - 0.524) * 100

                # Handle model object with predict method
                elif hasattr(model_data, 'predict'):
                    result = model_data.predict(features, prop_line=line)

                    if 'over_probability' in result:
                        over_prob = result['over_probability']
                        edge = (over_prob - 0.524) * 100
                    elif 'predicted_value' in result:
                        predicted_value = result['predicted_value']
                        std = line * 0.20 if line > 0 else 5.0
                        z_score = (predicted_value - line) / max(std, 1)
                        over_prob = float(norm.cdf(z_score))
                        edge = (over_prob - 0.524) * 100

        except Exception as e:
            pass  # Fall through to return defaults

    # Without full feature generation, just show the prop lines
    # Edge calculation would need player stats
    return {
        'player': player_name,
        'player_id': player_id,
        'stat': prop_type.upper(),
        'line': line,
        'over_prob': over_prob,
        'edge': edge,
        'predicted_value': predicted_value,
    }


def get_starters_for_game(api: BalldontlieAPI, game: Dict) -> Dict[str, List[Dict]]:
    """Get starters for both teams from recent games."""
    home_team = game.get('home_team', {})
    away_team = game.get('visitor_team', {})

    starters = {
        'home': [],
        'away': []
    }

    # Try to get from season averages - top 5 by minutes
    try:
        for team_key, team in [('home', home_team), ('away', away_team)]:
            team_id = team.get('id')
            if team_id:
                # Get team roster and stats
                players = api.get_players(team_ids=[team_id])
                if players:
                    # Just take first 5 as approximation of starters
                    starters[team_key] = players[:5]
    except Exception:
        pass

    return starters


def main():
    """Main entry point for daily predictions."""
    import argparse

    parser = argparse.ArgumentParser(description="Daily NBA Predictions")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")

    print("=" * 65)
    print("  NBA BETTING MODEL - Daily Predictions")
    print("=" * 65)
    print(f"  Date: {target_date}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load models
    print("\n  Loading models...")
    models = load_models()
    print(f"  Loaded: {list(models.keys())}")

    # Initialize Balldontlie API
    try:
        api = BalldontlieAPI()
        print("  Balldontlie API: Connected (GOAT tier)")
    except Exception as e:
        print(f"  Balldontlie API: Error - {e}")
        api = None

    # Get today's games
    print("\n  Fetching games...")
    games = []
    odds_data = {}

    if api:
        try:
            games = api.get_games(dates=[target_date])
            print(f"  Found {len(games)} games")

            # Get betting odds (returns list of odds from multiple vendors)
            odds_list = api.get_betting_odds(date=target_date)
            print(f"  Fetched {len(odds_list)} odds entries")

            # Index by game_id, preferring DraftKings/FanDuel
            preferred_vendors = ['draftkings', 'fanduel', 'betmgm', 'caesars']
            for odds in odds_list:
                game_id = odds.get('game_id')
                vendor = odds.get('vendor', '').lower()
                if not game_id:
                    continue

                # Only store if no odds yet, or this is a preferred vendor
                if game_id not in odds_data or vendor in preferred_vendors:
                    odds_data[game_id] = odds
        except Exception as e:
            print(f"  Error fetching games: {e}")

    # Parse odds into game-indexed dict with best values
    if odds_data:
        parsed_odds = {}
        for odds in odds_data.values() if isinstance(odds_data, dict) else odds_data:
            game_id = odds.get('game_id')
            if not game_id:
                continue

            # Parse spread (convert string to float)
            spread_val = odds.get('spread_home_value', '0')
            try:
                spread = float(spread_val) if spread_val else 0
            except:
                spread = 0

            # Get moneylines
            home_ml = odds.get('moneyline_home_odds', -110)
            away_ml = odds.get('moneyline_away_odds', -110)

            # Store parsed odds
            if game_id not in parsed_odds:
                parsed_odds[game_id] = {
                    'spread': -spread,  # Convert to home perspective (negative = home favored)
                    'home_moneyline': home_ml,
                    'away_moneyline': away_ml,
                    'total': float(odds.get('total_value', '220') or '220'),
                    'vendor': odds.get('vendor', 'unknown')
                }

        odds_data = parsed_odds
        print(f"  Parsed odds for {len(parsed_odds)} games")

    if not games:
        # Fallback to NBA API
        print("  Falling back to NBA Live API...")
        try:
            from nba_api.live.nba.endpoints import scoreboard
            board = scoreboard.ScoreBoard()
            nba_games = board.games.get_dict()

            for g in nba_games:
                games.append({
                    'id': g.get('gameId'),
                    'home_team': {'abbreviation': g.get('homeTeam', {}).get('teamTricode', 'HOME')},
                    'visitor_team': {'abbreviation': g.get('awayTeam', {}).get('teamTricode', 'AWAY')},
                    'status': g.get('gameStatusText', ''),
                })
            print(f"  Found {len(games)} games via NBA API")
        except Exception as e:
            print(f"  Error: {e}")

    if not games:
        print("\n  No games found for today.")
        return

    # Analyze each game
    print("\n" + "=" * 65)
    print("  GAME PREDICTIONS")
    print("=" * 65)

    all_analyses = []
    all_player_props = []

    for game in games:
        game_id = game.get('id')
        odds = odds_data.get(game_id, {})

        # Set default odds if not available
        if not odds:
            odds = {
                'home_moneyline': -110,
                'away_moneyline': -110,
                'spread': -3.0,  # Default home favored by 3
            }

        analysis = analyze_game(game, odds, models)

        # Fetch player props for this game (limit to top players with points props)
        if api and game_id:
            home = analysis['home_team']
            away = analysis['away_team']
            print(f"\n  Analyzing {away}@{home} props...", end="", flush=True)
            props_data = get_player_props_for_game(api, game_id)

            if props_data:
                # Get player names from API
                player_ids = list(props_data.keys())

                # Filter to players with points lines > 15 (likely starters/key players)
                key_players = {pid: props for pid, props in props_data.items()
                              if props.get('points_line', 0) >= 15}

                # Limit to top 10 players by points line
                sorted_players = sorted(key_players.items(),
                                       key=lambda x: x[1].get('points_line', 0), reverse=True)[:10]

                if sorted_players:
                    # Build player name mapping using IDMapper (cached, fast)
                    player_names = {}
                    mapper = get_id_mapper()

                    # Get names for all players in props
                    for player_id, _ in sorted_players:
                        if mapper:
                            name = mapper.get_player_name(player_id)
                            if name:
                                player_names[player_id] = name

                    # Fallback: try Balldontlie API for any missing names
                    missing_ids = [pid for pid, _ in sorted_players if pid not in player_names]
                    if missing_ids:
                        try:
                            for pid in missing_ids:
                                p = api.get_player(pid)
                                if p:
                                    fname = p.get('first_name', '')
                                    lname = p.get('last_name', '')
                                    if fname or lname:
                                        player_names[pid] = f"{fname} {lname}".strip()
                        except:
                            pass

                    # Get team IDs for opponent context
                    home_team = game.get('home_team', {})
                    away_team = game.get('visitor_team', {})
                    home_team_id = home_team.get('id')
                    away_team_id = away_team.get('id')

                    # Get injured players to filter from props
                    injury_details = analysis.get('injury_details', {})
                    injured_players = set()
                    for inj in injury_details.get('home', []):
                        status = inj.get('status', '').upper()
                        if status in ('OUT', 'DOUBTFUL'):
                            injured_players.add(inj.get('player_name', '').lower())
                    for inj in injury_details.get('away', []):
                        status = inj.get('status', '').upper()
                        if status in ('OUT', 'DOUBTFUL'):
                            injured_players.add(inj.get('player_name', '').lower())

                    # Add props to analysis
                    for player_id, props in sorted_players:
                        player_name = player_names.get(player_id, f"Player {player_id}")
                        player_team_id = props.get('team_id')

                        # Skip injured players (OUT/DOUBTFUL)
                        if player_name.lower() in injured_players:
                            continue

                        # CRITICAL: Look up the correct Balldontlie ID for stats
                        # Props API uses different IDs than active players endpoint
                        bdl_stats_id = None
                        if player_name and not player_name.startswith("Player"):
                            bdl_stats_id = get_bdl_player_id(player_name)

                        # Determine opponent based on player's team
                        if player_team_id == home_team_id:
                            opponent_id = away_team_id
                            opponent_abbrev = analysis['away_team']
                        else:
                            opponent_id = home_team_id
                            opponent_abbrev = analysis['home_team']

                        for prop_type in ['points', 'rebounds', 'assists']:
                            line_key = f'{prop_type}_line'
                            if line_key in props:
                                line = props[line_key]
                                # Use bdl_stats_id for feature generation (correct ID for stats API)
                                pred = predict_player_prop(
                                    player_name, bdl_stats_id or player_id, prop_type, line,
                                    opponent_abbrev, opponent_id, models,
                                    use_api_features=True  # Enable real predictions
                                )
                                analysis['player_props'].append(pred)
                                all_player_props.append({
                                    'game': f"{analysis['away_team']}@{analysis['home_team']}",
                                    **pred
                                })

                    # Show completion
                    prop_count = len(analysis['player_props'])
                    edges = [p['edge'] for p in analysis['player_props'] if abs(p['edge']) > 3]
                    print(f" {prop_count} props analyzed, {len(edges)} edges found")

        all_analyses.append(analysis)
        print_game_analysis(analysis)

    # Summary of best bets
    print("\n" + "=" * 65)
    print("  TOP RECOMMENDATIONS (>5% edge)")
    print("=" * 65)

    recommendations = []

    for a in all_analyses:
        home = a['home_team']
        away = a['away_team']

        # Moneyline edges
        if a['moneyline']['home_edge'] > 5:
            recommendations.append({
                'game': f"{away}@{home}",
                'bet': f"{home} ML",
                'prob': a['moneyline']['home_prob'],
                'edge': a['moneyline']['home_edge']
            })
        if a['moneyline']['away_edge'] > 5:
            recommendations.append({
                'game': f"{away}@{home}",
                'bet': f"{away} ML",
                'prob': a['moneyline']['away_prob'],
                'edge': a['moneyline']['away_edge']
            })

        # Spread edges
        if a['spread']['edge_pct'] > 5:
            recommendations.append({
                'game': f"{away}@{home}",
                'bet': a['spread']['bet_side'],
                'prob': a['spread']['cover_prob'],
                'edge': a['spread']['edge_pct']
            })

        # Player prop edges
        for prop in a.get('player_props', []):
            if abs(prop['edge']) > 5:
                direction = "Over" if prop['over_prob'] > 0.5 else "Under"
                prob = prop['over_prob'] if prop['over_prob'] > 0.5 else (1 - prop['over_prob'])
                recommendations.append({
                    'game': f"{away}@{home}",
                    'bet': f"{prop['player']} {prop['stat']} {direction} {prop['line']}",
                    'prob': prob,
                    'edge': prop['edge']
                })

    # Sort by edge
    recommendations.sort(key=lambda x: x['edge'], reverse=True)

    if recommendations:
        print()
        for i, rec in enumerate(recommendations[:10], 1):
            print(f"  {i}. {rec['game']}: {rec['bet']} ({rec['prob']:.0%}, edge: {rec['edge']:+.1f}%)")
    else:
        print("\n  No strong edges found today.")

    print("\n" + "=" * 65)
    print("  Note: Always verify with actual sportsbook odds before betting.")
    print("=" * 65)


if __name__ == "__main__":
    main()
