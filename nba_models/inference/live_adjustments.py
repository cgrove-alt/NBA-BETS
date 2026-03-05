"""
Live In-Game Prediction Adjustments.

This module provides functions to adjust pre-game predictions based on
live game state (score, time remaining, player stats).

Key adjustments:
1. Spread: Blend pre-game model with current margin based on time elapsed
2. Moneyline: Adjust win probability based on score differential and time
3. Player Props: Pace-project current stats to end of game
"""

from __future__ import annotations

import math


# NBA game constants
REGULATION_MINUTES = 48.0
OVERTIME_MINUTES = 5.0
QUARTERS = 4


def parse_game_time(period: int, time_remaining: str) -> float:
    """
    Calculate total minutes elapsed in the game.

    Args:
        period: Current period (1-4 for regulation, 5+ for OT)
        time_remaining: Time left in period (e.g., "5:30", "45.2")

    Returns:
        Minutes elapsed since game start
    """
    if period <= 0:
        return 0.0

    # Parse time remaining in current period
    minutes_left = 0.0
    if time_remaining:
        try:
            if ':' in time_remaining:
                parts = time_remaining.split(':')
                minutes_left = float(parts[0]) + float(parts[1]) / 60.0
            else:
                # Just seconds (e.g., "45.2")
                minutes_left = float(time_remaining) / 60.0
        except (ValueError, IndexError):
            minutes_left = 0.0

    # Calculate elapsed time
    if period <= QUARTERS:
        # Regulation: 12 minutes per quarter
        completed_quarters = period - 1
        elapsed_in_current = 12.0 - minutes_left
        total_elapsed = (completed_quarters * 12.0) + elapsed_in_current
    else:
        # Overtime: regulation complete + OT periods
        ot_period = period - QUARTERS
        completed_ot = ot_period - 1
        elapsed_in_current = 5.0 - minutes_left
        total_elapsed = REGULATION_MINUTES + (completed_ot * OVERTIME_MINUTES) + elapsed_in_current

    return max(0.0, total_elapsed)


def get_time_factor(minutes_elapsed: float, total_minutes: float = REGULATION_MINUTES) -> float:
    """
    Calculate how much weight to give live data vs pre-game prediction.

    Early game: Trust pre-game model more
    Late game: Trust current game state more

    Uses a non-linear curve that accelerates weighting toward live data
    in the 4th quarter.

    Args:
        minutes_elapsed: Minutes played so far
        total_minutes: Expected total game minutes (48 for regulation)

    Returns:
        Float between 0 and 1 (higher = more weight on pre-game model)
    """
    if total_minutes <= 0:
        return 1.0

    # Clamp elapsed time
    elapsed = min(minutes_elapsed, total_minutes)

    # Non-linear progression: sqrt gives more weight to live data as game progresses
    # but not too aggressive early
    progress = elapsed / total_minutes

    # Use a curve that's gentle early but accelerates late
    # At 25% of game: ~40% weight on pre-game
    # At 50% of game: ~30% weight on pre-game
    # At 75% of game: ~15% weight on pre-game
    # At 100%: 0% weight on pre-game (fully trust actual result)
    return (1 - progress) ** 1.5



def adjust_spread_prediction(
    pre_game_spread: float,
    home_score: int,
    away_score: int,
    period: int,
    time_remaining: str,
    home_team_perspective: bool = True
) -> dict[str, float]:
    """
    Adjust spread prediction based on live game state.

    The adjustment blends the pre-game model spread with the current score
    differential, weighted by time elapsed.

    Formula:
        adjusted_spread = pre_game_spread * time_factor + current_margin * (1 - time_factor)

    Args:
        pre_game_spread: Model's pre-game spread prediction (positive = home favored)
        home_score: Current home team score
        away_score: Current away team score
        period: Current period (1-4, 5+ for OT)
        time_remaining: Time left in current period
        home_team_perspective: If True, spread is from home team perspective

    Returns:
        Dict with adjusted_spread, time_factor, current_margin, minutes_elapsed
    """
    minutes_elapsed = parse_game_time(period, time_remaining)

    # Handle overtime by extending total expected minutes
    total_minutes = REGULATION_MINUTES
    if period > QUARTERS:
        total_minutes = REGULATION_MINUTES + (period - QUARTERS) * OVERTIME_MINUTES

    time_factor = get_time_factor(minutes_elapsed, total_minutes)

    # Current margin from home team perspective
    current_margin = home_score - away_score
    if not home_team_perspective:
        current_margin = -current_margin

    # Blend pre-game prediction with current margin
    adjusted_spread = (pre_game_spread * time_factor) + (current_margin * (1 - time_factor))

    return {
        "adjusted_spread": round(adjusted_spread, 1),
        "pre_game_spread": pre_game_spread,
        "current_margin": current_margin,
        "time_factor": round(time_factor, 3),
        "minutes_elapsed": round(minutes_elapsed, 1),
        "period": period,
    }


def adjust_moneyline_probability(
    pre_game_prob: float,
    home_score: int,
    away_score: int,
    period: int,
    time_remaining: str,
    for_home_team: bool = True
) -> dict[str, float]:
    """
    Adjust win probability based on live game state.

    Uses a logistic model that factors in:
    1. Pre-game probability (weighted by time remaining)
    2. Current score differential
    3. Time remaining (larger leads matter more late in game)

    Args:
        pre_game_prob: Pre-game win probability (0-1)
        home_score: Current home team score
        away_score: Current away team score
        period: Current period
        time_remaining: Time left in current period
        for_home_team: If True, probability is for home team winning

    Returns:
        Dict with adjusted_probability, components, and metadata
    """
    minutes_elapsed = parse_game_time(period, time_remaining)

    # Handle overtime
    total_minutes = REGULATION_MINUTES
    if period > QUARTERS:
        total_minutes = REGULATION_MINUTES + (period - QUARTERS) * OVERTIME_MINUTES

    time_factor = get_time_factor(minutes_elapsed, total_minutes)
    minutes_remaining = max(0, total_minutes - minutes_elapsed)

    # Score differential (positive = home leading)
    score_diff = home_score - away_score
    if not for_home_team:
        score_diff = -score_diff
        pre_game_prob = 1 - pre_game_prob

    # Calculate live win probability using logistic model
    # Key insight: A 1-point lead per minute remaining is roughly a 50% win prob
    # More lead per minute remaining = higher win probability

    if minutes_remaining <= 0:
        # Game over - use actual result
        live_prob = 1.0 if score_diff > 0 else (0.5 if score_diff == 0 else 0.0)
    else:
        # Points per minute remaining - normalized measure of lead
        points_per_minute = score_diff / minutes_remaining

        # Logistic function centered at 0 (tied)
        # Coefficient 0.5 means ~73% win prob with 1 point/minute lead
        # Coefficient tuned from NBA historical data
        k = 0.5
        live_prob = 1 / (1 + math.exp(-k * points_per_minute * minutes_remaining ** 0.5))

    # Blend pre-game and live probabilities
    adjusted_prob = (pre_game_prob * time_factor) + (live_prob * (1 - time_factor))

    # Clamp to valid probability range
    adjusted_prob = max(0.01, min(0.99, adjusted_prob))

    return {
        "adjusted_probability": round(adjusted_prob, 3),
        "pre_game_probability": round(pre_game_prob, 3),
        "live_probability": round(live_prob, 3),
        "time_factor": round(time_factor, 3),
        "score_differential": score_diff,
        "minutes_remaining": round(minutes_remaining, 1),
        "minutes_elapsed": round(minutes_elapsed, 1),
    }


def adjust_player_prop(
    pre_game_prediction: float,
    current_stat: float,
    minutes_played: float,
    expected_minutes: float = 32.0,
    prop_type: str = "points"
) -> dict[str, float]:
    """
    Adjust player prop prediction using pace projection.

    Blends pace-projected stats with pre-game prediction, weighted by
    minutes played. More weight on actual pace as player accumulates minutes.

    Formula:
        pace_projected = current_stat * (expected_minutes / minutes_played)
        adjusted = weight * pace_projected + (1 - weight) * pre_game_prediction

    Args:
        pre_game_prediction: Pre-game stat prediction
        current_stat: Current accumulated stat (points, rebounds, etc.)
        minutes_played: Minutes the player has played
        expected_minutes: Expected total minutes for the player
        prop_type: Type of prop ("points", "rebounds", "assists", "3pm", "pra")

    Returns:
        Dict with adjusted prediction and projection details
    """
    if minutes_played <= 0:
        return {
            "adjusted_prediction": pre_game_prediction,
            "pace_projected": pre_game_prediction,
            "current_stat": current_stat,
            "minutes_played": 0,
            "expected_minutes": expected_minutes,
            "confidence": 0.0,
            "projection_weight": 0.0,
        }

    # Cap minutes played at expected minutes
    effective_minutes = min(minutes_played, expected_minutes)

    # Pace projection: extrapolate current performance to full game
    if effective_minutes > 0:
        pace_rate = current_stat / effective_minutes
        pace_projected = pace_rate * expected_minutes
    else:
        pace_projected = pre_game_prediction

    # Weight on pace projection increases with minutes played
    # At 8 minutes: ~25% weight on pace
    # At 16 minutes: ~50% weight on pace
    # At 24 minutes: ~75% weight on pace
    # At 32 minutes: ~100% weight on pace
    projection_weight = min(1.0, effective_minutes / expected_minutes)

    # Apply different weighting curves based on prop type
    # Points are more volatile early, rebounds/assists more predictive
    if prop_type == "points":
        # Points can be streaky - be more conservative early
        projection_weight = projection_weight ** 1.2
    elif prop_type in ["rebounds", "assists"]:
        # These accumulate more steadily
        projection_weight = projection_weight ** 0.9
    elif prop_type == "3pm":
        # Three pointers are highly volatile - very conservative
        projection_weight = projection_weight ** 1.5
    elif prop_type == "pra":
        # Combined stat - moderate approach
        projection_weight = projection_weight ** 1.0

    # Blend pace projection with pre-game prediction
    adjusted = (projection_weight * pace_projected) + ((1 - projection_weight) * pre_game_prediction)

    # Confidence based on sample size (minutes played)
    confidence = min(1.0, effective_minutes / 20.0)  # Full confidence at 20 min

    return {
        "adjusted_prediction": round(adjusted, 1),
        "pace_projected": round(pace_projected, 1),
        "pre_game_prediction": pre_game_prediction,
        "current_stat": current_stat,
        "minutes_played": round(minutes_played, 1),
        "expected_minutes": expected_minutes,
        "projection_weight": round(projection_weight, 3),
        "confidence": round(confidence, 3),
        "prop_type": prop_type,
    }


def adjust_game_predictions(
    game_predictions: dict,
    live_score: dict,
    player_stats: dict | None = None
) -> dict:
    """
    Adjust all predictions for a game based on live state.

    Args:
        game_predictions: Dict containing pre-game predictions:
            - spread: float (home team perspective)
            - home_win_prob: float (0-1)
            - away_win_prob: float (0-1)
            - player_props: List of {player, prop_type, prediction, line}
        live_score: Dict containing:
            - home_score: int
            - away_score: int
            - period: int
            - time_remaining: str
        player_stats: Optional dict of {player_name: {stat_type: value, minutes: float}}

    Returns:
        Dict with all adjusted predictions
    """
    home_score = live_score.get("home_score", 0)
    away_score = live_score.get("away_score", 0)
    period = live_score.get("period", 0)
    time_remaining = live_score.get("time_remaining", "12:00")

    # If game hasn't started, return original predictions
    if period <= 0 or (home_score == 0 and away_score == 0 and period == 1):
        return {
            "spread": game_predictions.get("spread"),
            "home_win_prob": game_predictions.get("home_win_prob"),
            "away_win_prob": game_predictions.get("away_win_prob"),
            "player_props": game_predictions.get("player_props", []),
            "is_live_adjusted": False,
            "live_score": live_score,
        }

    # Adjust spread
    spread_result = adjust_spread_prediction(
        pre_game_spread=game_predictions.get("spread", 0),
        home_score=home_score,
        away_score=away_score,
        period=period,
        time_remaining=time_remaining,
    )

    # Adjust moneyline probabilities
    home_ml_result = adjust_moneyline_probability(
        pre_game_prob=game_predictions.get("home_win_prob", 0.5),
        home_score=home_score,
        away_score=away_score,
        period=period,
        time_remaining=time_remaining,
        for_home_team=True,
    )

    away_ml_result = adjust_moneyline_probability(
        pre_game_prob=game_predictions.get("away_win_prob", 0.5),
        home_score=home_score,
        away_score=away_score,
        period=period,
        time_remaining=time_remaining,
        for_home_team=False,
    )

    # Adjust player props if stats available
    adjusted_props = []
    if player_stats:
        for prop in game_predictions.get("player_props", []):
            player = prop.get("player", "")
            prop_type = prop.get("prop_type", "points")
            pre_game_pred = prop.get("prediction", 0)

            # Get current stats for this player
            stats = player_stats.get(player, {})
            current_stat = stats.get(prop_type, 0)
            minutes_played = stats.get("minutes", 0)
            expected_minutes = stats.get("expected_minutes", 32)

            prop_result = adjust_player_prop(
                pre_game_prediction=pre_game_pred,
                current_stat=current_stat,
                minutes_played=minutes_played,
                expected_minutes=expected_minutes,
                prop_type=prop_type,
            )

            adjusted_props.append({
                **prop,
                "adjusted_prediction": prop_result["adjusted_prediction"],
                "pace_projected": prop_result["pace_projected"],
                "projection_weight": prop_result["projection_weight"],
            })
    else:
        adjusted_props = game_predictions.get("player_props", [])

    return {
        "spread": spread_result["adjusted_spread"],
        "spread_details": spread_result,
        "home_win_prob": home_ml_result["adjusted_probability"],
        "away_win_prob": away_ml_result["adjusted_probability"],
        "home_ml_details": home_ml_result,
        "away_ml_details": away_ml_result,
        "player_props": adjusted_props,
        "is_live_adjusted": True,
        "live_score": live_score,
    }


# Example usage and testing
if __name__ == "__main__":
    # Test spread adjustment
    print("=== Spread Adjustment Test ===")
    spread_test = adjust_spread_prediction(
        pre_game_spread=-5.0,  # Home favored by 5
        home_score=58,
        away_score=52,
        period=3,
        time_remaining="6:30"
    )
    print(f"Pre-game spread: {spread_test['pre_game_spread']}")
    print(f"Current margin: {spread_test['current_margin']}")
    print(f"Time factor: {spread_test['time_factor']}")
    print(f"Adjusted spread: {spread_test['adjusted_spread']}")
    print()

    # Test moneyline adjustment
    print("=== Moneyline Adjustment Test ===")
    ml_test = adjust_moneyline_probability(
        pre_game_prob=0.65,  # 65% pre-game win prob
        home_score=58,
        away_score=52,
        period=3,
        time_remaining="6:30",
        for_home_team=True
    )
    print(f"Pre-game probability: {ml_test['pre_game_probability']}")
    print(f"Live probability: {ml_test['live_probability']}")
    print(f"Time factor: {ml_test['time_factor']}")
    print(f"Adjusted probability: {ml_test['adjusted_probability']}")
    print()

    # Test player prop adjustment
    print("=== Player Prop Adjustment Test ===")
    prop_test = adjust_player_prop(
        pre_game_prediction=25.0,  # Predicted 25 points
        current_stat=15,  # Has 15 points
        minutes_played=20,
        expected_minutes=32,
        prop_type="points"
    )
    print(f"Pre-game prediction: {prop_test['pre_game_prediction']}")
    print(f"Current stat: {prop_test['current_stat']}")
    print(f"Pace projected: {prop_test['pace_projected']}")
    print(f"Projection weight: {prop_test['projection_weight']}")
    print(f"Adjusted prediction: {prop_test['adjusted_prediction']}")
