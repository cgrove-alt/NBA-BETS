"""
Callbacks for NBA Betting Dashboard.
"""

import time
from datetime import datetime, timezone
from dash import callback, Input, Output, State, html, no_update


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
    """Load today's games into the dropdown."""
    ds = get_data_service()
    games = ds.get_todays_games()

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

        # Build label: "CHI @ CLE - 7:30 PM" or "CHI @ CLE (Final)"
        label = f"{away} @ {home}"
        if time_str:
            if time_str in ['Final', 'In Progress']:
                label += f" ({time_str})"
            else:
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
    """Fetch props for the selected game."""
    if not game_id:
        return {}, {}

    ds = get_data_service()
    games = ds.get_todays_games()

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

    game_info = {"home": home, "away": away, "game_id": game_id}

    return props_data, game_info


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
