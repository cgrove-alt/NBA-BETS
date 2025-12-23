"""
Simple layout helpers for NBA Betting Dashboard.
"""

from typing import Dict, List
from dash import html
import dash_bootstrap_components as dbc


# Prop type display names
PROP_LABELS = {
    'points': 'Points',
    'rebounds': 'Rebounds',
    'assists': 'Assists',
    '3pm': '3PM',
    'pra': 'PRA'
}


def create_prop_table(props: List[Dict], prop_type: str, home_abbrev: str, away_abbrev: str) -> dbc.Card:
    """Create a table for a single prop type.

    Args:
        props: List of player prop dictionaries (combined home + away)
        prop_type: One of 'points', 'rebounds', 'assists', '3pm', 'pra'
        home_abbrev: Home team abbreviation (e.g., "CLE")
        away_abbrev: Away team abbreviation (e.g., "CHI")

    Returns:
        A dbc.Card containing the prop table
    """
    rows = []

    for player in props:
        player_name = player.get('player_name', 'Unknown')
        team = player.get('team', '')

        # Get prop-specific fields
        pred = player.get(f'{prop_type}_pred', 0) or 0
        line = player.get(f'{prop_type}_line', 0) or 0
        pick = player.get(f'{prop_type}_pick', '-') or '-'
        conf = player.get(f'{prop_type}_confidence', 0) or 0

        # Only show rows with valid predictions
        if pred > 0:
            # Truncate long names
            display_name = player_name[:20] + '...' if len(player_name) > 20 else player_name

            # Color-code the pick
            if pick == 'OVER':
                pick_badge = dbc.Badge("OVER", color="success")
            elif pick == 'UNDER':
                pick_badge = dbc.Badge("UNDER", color="danger")
            else:
                pick_badge = dbc.Badge("-", color="secondary")

            rows.append(html.Tr([
                html.Td(display_name, style={"maxWidth": "150px"}),
                html.Td(team, className="text-center"),
                html.Td(f"{line:.1f}", className="text-center"),
                html.Td(f"{pred:.1f}", className="text-center"),
                html.Td(pick_badge, className="text-center"),
                html.Td(f"{conf:.0f}%", className="text-center"),
            ]))

    # Sort by confidence descending
    rows.sort(key=lambda r: float(r.children[5].children.replace('%', '') or 0), reverse=True)

    # If no rows, show message
    if not rows:
        rows.append(html.Tr([
            html.Td("No predictions available", colSpan=6, className="text-center text-muted")
        ]))

    prop_label = PROP_LABELS.get(prop_type, prop_type.upper())

    return dbc.Card([
        dbc.CardHeader(
            prop_label,
            className="fw-bold",
            style={
                "backgroundColor": "#21262d",
                "borderBottom": "1px solid #30363d",
                "color": "#e6edf3",
                "fontSize": "1rem"
            }
        ),
        dbc.CardBody([
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Player", style={"width": "30%"}),
                    html.Th("Team", className="text-center", style={"width": "10%"}),
                    html.Th("Line", className="text-center", style={"width": "12%"}),
                    html.Th("Pred", className="text-center", style={"width": "12%"}),
                    html.Th("Pick", className="text-center", style={"width": "15%"}),
                    html.Th("Conf", className="text-center", style={"width": "12%"}),
                ], className="text-muted small")),
                html.Tbody(rows)
            ], className="table table-sm mb-0", style={
                "color": "#e6edf3",
                "fontSize": "0.85rem"
            })
        ], style={"padding": "0.5rem"})
    ], style={
        "backgroundColor": "#161b22",
        "border": "1px solid #30363d",
    })


def create_game_card(game_data: Dict, props: List[Dict]) -> dbc.Card:
    """Create a simple card with game info and props table.

    Note: This is the old function kept for backwards compatibility.
    """
    home = game_data.get('home_team', {}).get('abbreviation', '?')
    away = game_data.get('visitor_team', {}).get('abbreviation', '?')

    # Build table rows from player props
    rows = []
    for player in props:
        player_name = player.get('player_name', 'Unknown')
        # Truncate long names
        display_name = player_name[:18] + '...' if len(player_name) > 18 else player_name

        for prop_type in ['points', 'rebounds', 'assists', '3pm', 'pra']:
            pred = player.get(f'{prop_type}_pred', 0) or 0
            line = player.get(f'{prop_type}_line', 0) or 0
            pick = player.get(f'{prop_type}_pick', '-') or '-'
            conf = player.get(f'{prop_type}_confidence', 0) or 0

            # Show all props (even with 0 prediction for debugging)
            if pred >= 0:
                # Color-code the pick
                if pick == 'OVER':
                    pick_badge = dbc.Badge("OVER", color="success", className="ms-1")
                elif pick == 'UNDER':
                    pick_badge = dbc.Badge("UNDER", color="danger", className="ms-1")
                else:
                    pick_badge = dbc.Badge("-", color="secondary", className="ms-1")

                # Prop type label
                prop_label = {
                    'points': 'PTS',
                    'rebounds': 'REB',
                    'assists': 'AST',
                    '3pm': '3PM',
                    'pra': 'PRA'
                }.get(prop_type, prop_type.upper()[:3])

                rows.append(html.Tr([
                    html.Td(display_name, className="text-truncate", style={"maxWidth": "120px"}),
                    html.Td(prop_label),
                    html.Td(f"{line:.1f}"),
                    html.Td(f"{pred:.1f}"),
                    html.Td(f"{conf:.0f}%"),
                    html.Td(pick_badge),
                ]))

    # If no props, show diagnostic message
    if not rows:
        if len(props) == 0:
            msg = "No player data received"
        else:
            # We have props but no predictions generated rows
            player_names = [p.get('player_name', 'Unknown') for p in props[:3]]
            msg = f"Received {len(props)} players ({', '.join(player_names)}...) but no predictions"
        rows.append(html.Tr([
            html.Td(msg, colSpan=6, className="text-center text-muted")
        ]))

    return dbc.Card([
        dbc.CardHeader(
            f"{away} @ {home}",
            className="fw-bold",
            style={
                "backgroundColor": "#21262d",
                "borderBottom": "1px solid #30363d",
                "color": "#e6edf3"
            }
        ),
        dbc.CardBody([
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Player", style={"width": "30%"}),
                    html.Th("Prop"),
                    html.Th("Line"),
                    html.Th("Pred"),
                    html.Th("Conf"),
                    html.Th("Pick"),
                ], className="text-muted small")),
                html.Tbody(rows)
            ], className="table table-sm mb-0", style={
                "color": "#e6edf3",
                "fontSize": "0.85rem"
            })
        ], style={"padding": "0.5rem"})
    ], className="mb-3", style={
        "backgroundColor": "#161b22",
        "border": "1px solid #30363d",
    })
