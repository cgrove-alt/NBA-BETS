"""
Callbacks for NBA Betting Dashboard.
"""

import time
from datetime import datetime, timezone
from dash import callback, Input, Output, State, html, no_update

# Import live adjustments for in-progress games
try:
    from live_adjustments import adjust_player_prop
    LIVE_ADJUSTMENTS_AVAILABLE = True
except ImportError:
    LIVE_ADJUSTMENTS_AVAILABLE = False


def parse_minutes_string(min_str: str) -> float:
    """Convert minutes string like '18:30' or '18.5' to float minutes."""
    if not min_str:
        return 0.0
    try:
        if ':' in str(min_str):
            parts = str(min_str).split(':')
            return float(parts[0]) + float(parts[1]) / 60.0
        return float(min_str)
    except (ValueError, IndexError):
        return 0.0


def format_game_time(game: dict) -> str:
    """Convert game time from UTC to local timezone and format for display."""
    game_time = game.get('game_time', '')
    status = game.get('status', '')

    # If game is Final or In Progress, show that instead of time
    if status in ['Final', 'In Progress']:
        return status

    # Try to parse ISO datetime (e.g., "2025-12-17T01:30:00Z")
    if game_time and 'T' in game_time:
        try:
            # Parse UTC time
            if game_time.endswith('Z'):
                dt = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
            else:
                dt = datetime.fromisoformat(game_time)
            # Convert to local timezone
            local_dt = dt.astimezone()
            return local_dt.strftime('%I:%M %p').lstrip('0')  # "7:30 PM"
        except ValueError:
            pass

    return ''  # No time available
import dash_bootstrap_components as dbc
from dashboard.data_service import get_data_service
from dashboard.layouts import create_prop_table


@callback(
    Output("game-selector", "options"),
    Output("game-selector", "value"),
    Output("last-update-time", "children"),
    Output("points-table-container", "children", allow_duplicate=True),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)
def load_games(n_clicks):
    """Load today's games into the dropdown - always fetch fresh data."""
    ds = get_data_service()
    games = ds.get_todays_games(force_refresh=True)  # Force fresh odds on every page refresh

    if not games:
        no_games_msg = dbc.Alert(
            "No NBA games scheduled for today. Check back later!",
            color="info",
            className="mt-3"
        )
        return [], None, f"Updated: {datetime.now().strftime('%I:%M %p')} - No games today", no_games_msg

    # Format games for dropdown
    options = []
    for game in games:
        home = game.get('home_team', {}).get('abbreviation', '?')
        away = game.get('visitor_team', {}).get('abbreviation', '?')
        game_id = str(game.get('game_id', game.get('id', '')))

        # Get formatted local time
        time_str = format_game_time(game)

        # Get live scores if available
        home_score = game.get('home_score', 0)
        away_score = game.get('away_score', 0)
        is_live = game.get('is_live', False)
        is_final = game.get('is_final', False)

        # Build label with scores for live/final games
        if is_live or is_final:
            # Show score: "CHI 85 @ CLE 92 (3rd Qtr)" or "CHI 98 @ CLE 105 (Final)"
            status = game.get('status', 'Final' if is_final else 'Live')
            label = f"{away} {away_score} @ {home} {home_score} ({status})"
        else:
            # Scheduled game: "CHI @ CLE - 7:30 PM"
            label = f"{away} @ {home}"
            if time_str:
                label += f" - {time_str}"

        options.append({"label": label, "value": game_id})

    # Auto-select first game
    default_value = options[0]["value"] if options else None

    update_time = datetime.now().strftime('%I:%M %p')
    return options, default_value, f"Updated: {update_time}", no_update


@callback(
    Output("props-store", "data"),
    Output("game-info-store", "data"),
    Input("game-selector", "value"),
    prevent_initial_call=True,
)
def fetch_props(game_id):
    """Fetch props for the selected game with fresh data."""
    if not game_id:
        return {}, {}

    ds = get_data_service()
    games = ds.get_todays_games(force_refresh=True)  # Get fresh game status

    # Find the selected game
    game_data = None
    for game in games:
        gid = str(game.get('game_id', game.get('id', '')))
        if gid == game_id:
            game_data = game
            break

    if not game_data:
        return {}, {}

    home = game_data.get('home_team', {}).get('abbreviation', '')
    away = game_data.get('visitor_team', {}).get('abbreviation', '')
    is_live = game_data.get('is_live', False)

    # Start fetching props
    ds.start_player_props_fetch(game_id, home, away)

    # Wait for completion (with timeout)
    props_data = {"home": [], "away": []}
    for _ in range(60):  # Max 60 seconds (props fetch can take 30-45s)
        status = ds.get_props_fetch_status(game_id)
        if status.get('status') in ['complete', 'ready', 'error']:
            # Add team info to each player
            home_props = status.get('home', [])
            away_props = status.get('away', [])

            for p in home_props:
                p['team'] = home
            for p in away_props:
                p['team'] = away

            props_data = {"home": home_props, "away": away_props}
            break
        time.sleep(1)

    # Apply live adjustments if game is in progress
    if is_live and LIVE_ADJUSTMENTS_AVAILABLE:
        try:
            # Get live player stats from Balldontlie
            live_stats = ds.get_live_player_stats(game_id)
            if live_stats:
                props_data = _apply_live_adjustments(props_data, live_stats)
        except Exception as e:
            print(f"[LIVE] Error applying live adjustments: {e}", flush=True)

    game_info = {
        "home": home,
        "away": away,
        "game_id": game_id,
        "is_live": is_live,
        "home_score": game_data.get('home_score', 0),
        "away_score": game_data.get('away_score', 0),
        "period": game_data.get('period', 0),
        "status": game_data.get('status', ''),
    }

    return props_data, game_info


def _apply_live_adjustments(props_data: dict, live_stats: dict) -> dict:
    """Apply pace projections to player props for in-progress games.

    Args:
        props_data: Dict with 'home' and 'away' lists of player props
        live_stats: Dict keyed by player_id with current stats

    Returns:
        Updated props_data with live-adjusted predictions
    """
    # Map prop types to stat keys in live_stats
    prop_to_stat = {
        'points': 'pts',
        'rebounds': 'reb',
        'assists': 'ast',
        '3pm': 'fg3m',
        'threes': 'fg3m',
        'pra': 'pra',
    }

    for team_key in ['home', 'away']:
        for player_prop in props_data.get(team_key, []):
            player_id = player_prop.get('player_id')
            if not player_id or player_id not in live_stats:
                continue

            stats = live_stats[player_id]
            minutes_played = parse_minutes_string(stats.get('min', '0:00'))

            # Skip if player hasn't played yet
            if minutes_played < 1:
                continue

            # Get expected minutes (use 32 as default or stored value)
            expected_minutes = player_prop.get('expected_minutes', 32)

            # Process each prop type this player has
            for prop in player_prop.get('props', []):
                prop_type = prop.get('prop_type', '')
                stat_key = prop_to_stat.get(prop_type.lower())
                if not stat_key:
                    continue

                current_stat = stats.get(stat_key, 0)
                pre_game_pred = prop.get('prediction', 0)

                if pre_game_pred > 0:
                    # Apply live adjustment using pace projection
                    result = adjust_player_prop(
                        pre_game_prediction=pre_game_pred,
                        current_stat=current_stat,
                        minutes_played=minutes_played,
                        expected_minutes=expected_minutes,
                        prop_type=prop_type.lower(),
                    )

                    # Store both original and adjusted prediction
                    prop['original_prediction'] = pre_game_pred
                    prop['prediction'] = result['adjusted_prediction']
                    prop['pace_projected'] = result['pace_projected']
                    prop['current_stat'] = current_stat
                    prop['minutes_played'] = minutes_played
                    prop['is_live_adjusted'] = True

    return props_data


@callback(
    Output("points-table-container", "children"),
    Output("rebounds-table-container", "children"),
    Output("assists-table-container", "children"),
    Output("threes-table-container", "children"),
    Output("pra-table-container", "children"),
    Input("props-store", "data"),
    State("game-info-store", "data"),
    prevent_initial_call=True,
)
def render_prop_tables(props_data, game_info):
    """Render the 5 prop type tables."""
    if not props_data or not game_info:
        empty = html.P("Select a game to view props", className="text-muted text-center")
        return empty, empty, empty, empty, empty

    home = game_info.get('home', '')
    away = game_info.get('away', '')

    # Combine home and away props
    all_props = props_data.get('home', []) + props_data.get('away', [])

    if not all_props:
        empty = html.P("No player props available for this game", className="text-muted text-center")
        return empty, empty, empty, empty, empty

    # Create a table for each prop type
    points_table = create_prop_table(all_props, 'points', home, away)
    rebounds_table = create_prop_table(all_props, 'rebounds', home, away)
    assists_table = create_prop_table(all_props, 'assists', home, away)
    threes_table = create_prop_table(all_props, '3pm', home, away)
    pra_table = create_prop_table(all_props, 'pra', home, away)

    return points_table, rebounds_table, assists_table, threes_table, pra_table
