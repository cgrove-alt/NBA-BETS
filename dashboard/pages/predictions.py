"""
Predictions Page - Game selector with separate prop type tables.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime


def layout():
    """Create predictions page layout with game dropdown and prop tables."""
    return dbc.Container([
        # Header row
        dbc.Row([
            dbc.Col([
                html.H4("NBA Player Props", className="text-white mb-0"),
                html.P(
                    id="last-update-time",
                    children=f"Updated: {datetime.now().strftime('%I:%M %p')}",
                    className="text-muted small mb-0"
                ),
            ], width=8),
            dbc.Col([
                dbc.Button(
                    "Refresh",
                    id="refresh-btn",
                    color="primary",
                    size="sm",
                ),
            ], width=4, className="text-end"),
        ], className="py-3 mb-3"),

        # Game selector dropdown
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id="game-selector",
                    options=[],
                    placeholder="Select a game...",
                    clearable=False,
                    style={
                        "backgroundColor": "#21262d",
                        "color": "#e6edf3",
                    },
                    className="dash-dropdown-dark"
                ),
            ], width=12),
        ], className="mb-4"),

        # Hidden store for props data
        dcc.Store(id="props-store", data={}),
        dcc.Store(id="game-info-store", data={}),

        # Loading spinner wrapper
        dbc.Spinner(
            html.Div([
                # 5 prop type table containers
                html.Div(id="points-table-container", className="mb-3"),
                html.Div(id="rebounds-table-container", className="mb-3"),
                html.Div(id="assists-table-container", className="mb-3"),
                html.Div(id="threes-table-container", className="mb-3"),
                html.Div(id="pra-table-container", className="mb-3"),
            ]),
            color="primary",
            type="border",
            size="sm",
        ),
    ], fluid=True, style={"backgroundColor": "#0d1117", "minHeight": "100vh"})
