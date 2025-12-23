"""
Bet Tracker Page

Track betting history, active bets, and P&L.
"""

from dash import html, dcc, register_page, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Register this page
register_page(__name__, path="/tracker", name="Bet Tracker")

# Dark theme colors
COLORS = {
    "bg_primary": "#0d1117",
    "bg_secondary": "#161b22",
    "bg_tertiary": "#21262d",
    "bg_card": "#1c2128",
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "text_muted": "#6e7681",
    "accent_primary": "#58a6ff",
    "accent_success": "#3fb950",
    "accent_warning": "#d29922",
    "accent_danger": "#f85149",
    "border_color": "#30363d",
}


def create_stat_card(value: str, label: str, color: str = None, icon: str = None) -> dbc.Card:
    """Create a stat card for quick metrics."""
    value_color = color if color else COLORS["text_primary"]
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(value, style={
                    "fontSize": "2rem",
                    "fontWeight": "700",
                    "color": value_color,
                    "lineHeight": "1.2",
                }),
                html.Div(label, style={
                    "fontSize": "0.875rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                }),
            ], className="text-center"),
        ]),
    ], className="h-100")


def layout():
    """Create the bet tracker page layout."""
    return html.Div([
        # Page Header
        html.Div([
            html.H2("Bet Tracker", className="mb-1"),
            html.P("Track your picks and monitor performance", className="text-muted"),
        ], className="mb-4"),

        # Quick Stats Row
        dbc.Row([
            dbc.Col([
                create_stat_card("$0.00", "Today's P&L", COLORS["text_primary"]),
            ], md=3, sm=6, className="mb-3"),
            dbc.Col([
                create_stat_card("$0.00", "Week's P&L", COLORS["text_primary"]),
            ], md=3, sm=6, className="mb-3"),
            dbc.Col([
                create_stat_card("0", "Active Bets", COLORS["accent_primary"]),
            ], md=3, sm=6, className="mb-3"),
            dbc.Col([
                create_stat_card("0-0", "Win Streak", COLORS["accent_success"]),
            ], md=3, sm=6, className="mb-3"),
        ], className="mb-4"),

        # Active Bets Section
        dbc.Card([
            dbc.CardHeader([
                html.H5("Active Bets", className="mb-0 d-inline"),
                dbc.Badge("0", color="primary", className="ms-2"),
            ]),
            dbc.CardBody([
                html.Div(id="active-bets-container", children=[
                    html.P("No active bets. Select a game from the Predictions page to place a bet.",
                           className="text-muted text-center py-4"),
                ]),
            ]),
        ], className="mb-4"),

        # Bet History Section
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.H5("Bet History", className="mb-0 d-inline"),
                ], className="d-flex justify-content-between align-items-center"),
            ]),
            dbc.CardBody([
                # Filters Row
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Date Range", className="small text-muted"),
                        dbc.Select(
                            id="date-range-filter",
                            options=[
                                {"label": "Last 7 Days", "value": "7"},
                                {"label": "Last 30 Days", "value": "30"},
                                {"label": "Last 90 Days", "value": "90"},
                                {"label": "All Time", "value": "all"},
                            ],
                            value="30",
                            size="sm",
                        ),
                    ], md=3, sm=6, className="mb-3"),
                    dbc.Col([
                        dbc.Label("Bet Type", className="small text-muted"),
                        dbc.Select(
                            id="bet-type-filter",
                            options=[
                                {"label": "All Types", "value": "all"},
                                {"label": "Moneyline", "value": "moneyline"},
                                {"label": "Spread", "value": "spread"},
                                {"label": "Props", "value": "props"},
                            ],
                            value="all",
                            size="sm",
                        ),
                    ], md=3, sm=6, className="mb-3"),
                    dbc.Col([
                        dbc.Label("Result", className="small text-muted"),
                        dbc.Select(
                            id="result-filter",
                            options=[
                                {"label": "All Results", "value": "all"},
                                {"label": "Won", "value": "won"},
                                {"label": "Lost", "value": "lost"},
                                {"label": "Pending", "value": "pending"},
                            ],
                            value="all",
                            size="sm",
                        ),
                    ], md=3, sm=6, className="mb-3"),
                    dbc.Col([
                        dbc.Label("Export", className="small text-muted"),
                        dbc.Button("Export CSV", id="export-csv-btn", color="secondary",
                                 size="sm", outline=True, className="w-100"),
                    ], md=3, sm=6, className="mb-3"),
                ], className="mb-3"),

                # History Table
                html.Div(id="bet-history-container", children=[
                    dash_table.DataTable(
                        id="bet-history-table",
                        columns=[
                            {"name": "Date", "id": "date"},
                            {"name": "Game", "id": "game"},
                            {"name": "Type", "id": "bet_type"},
                            {"name": "Pick", "id": "pick"},
                            {"name": "Odds", "id": "odds"},
                            {"name": "Stake", "id": "stake"},
                            {"name": "Result", "id": "result"},
                            {"name": "P&L", "id": "pnl"},
                        ],
                        data=[],
                        sort_action="native",
                        sort_mode="single",
                        page_size=15,
                        style_cell={
                            "textAlign": "center",
                            "padding": "10px 8px",
                            "fontSize": "14px",
                            "backgroundColor": COLORS["bg_card"],
                            "color": COLORS["text_primary"],
                            "border": f"1px solid {COLORS['border_color']}",
                        },
                        style_header={
                            "fontWeight": "600",
                            "backgroundColor": COLORS["bg_tertiary"],
                            "color": COLORS["text_primary"],
                            "borderBottom": f"2px solid {COLORS['border_color']}",
                        },
                        style_data_conditional=[
                            {
                                "if": {"filter_query": '{result} eq "Won"'},
                                "backgroundColor": "rgba(63, 185, 80, 0.1)",
                            },
                            {
                                "if": {"filter_query": '{result} eq "Lost"'},
                                "backgroundColor": "rgba(248, 81, 73, 0.1)",
                            },
                            {
                                "if": {
                                    "column_id": "pnl",
                                    "filter_query": '{pnl} contains "+"'
                                },
                                "color": COLORS["accent_success"],
                                "fontWeight": "600",
                            },
                            {
                                "if": {
                                    "column_id": "pnl",
                                    "filter_query": '{pnl} contains "-"'
                                },
                                "color": COLORS["accent_danger"],
                                "fontWeight": "600",
                            },
                        ],
                        style_table={"overflowX": "auto"},
                    ),
                ]),

                # Empty state message
                html.Div(id="no-history-message", children=[
                    html.P("No betting history yet. Start tracking your bets!",
                           className="text-muted text-center py-4"),
                ]),
            ]),
        ]),

        # Hidden stores for bet data
        dcc.Store(id="bet-history-store", data=[]),
        dcc.Store(id="active-bets-store", data=[]),
    ])


# Callbacks for tracker page
@callback(
    Output("bet-history-table", "data"),
    Output("no-history-message", "style"),
    Input("bet-history-store", "data"),
    Input("date-range-filter", "value"),
    Input("bet-type-filter", "value"),
    Input("result-filter", "value"),
)
def update_history_table(history_data, date_range, bet_type, result_filter):
    """Filter and update bet history table."""
    if not history_data:
        return [], {"display": "block"}

    filtered = history_data

    # Apply filters (when data exists)
    # For now, return empty with the empty state hidden
    return filtered, {"display": "none"} if filtered else {"display": "block"}
