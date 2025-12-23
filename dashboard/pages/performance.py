"""
Performance Page

Historical performance charts and metrics.
Shows real data from bet tracker database.
"""

import sys
from pathlib import Path
from dash import html, dcc, register_page, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Register this page
register_page(__name__, path="/performance", name="Performance")

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


def get_performance_data():
    """Fetch real performance data from bet tracker database."""
    try:
        from bet_tracker import BetTracker
        tracker = BetTracker()

        # Get bets from last 90 days
        start_date = datetime.now() - timedelta(days=90)
        bets = tracker.get_bets_by_date(start_date)

        # Convert TrackedBet objects to dicts for easier handling
        bet_history = []
        for bet in bets:
            bet_dict = {
                "bet_id": bet.bet_id,
                "placed_at": bet.placed_at.isoformat() if bet.placed_at else "",
                "bet_type": bet.bet_type.value if hasattr(bet.bet_type, 'value') else str(bet.bet_type),
                "selection": bet.selection,
                "odds": bet.odds,
                "stake": bet.stake,
                "pnl": bet.pnl,
                "status": bet.status.value if hasattr(bet.status, 'value') else str(bet.status),
                "edge": bet.edge,
            }
            bet_history.append(bet_dict)

        # Get performance metrics
        metrics = tracker.calculate_performance(start_date=start_date)

        return {
            "has_data": len(bet_history) > 0,
            "metrics": metrics,
            "history": bet_history,
        }
    except Exception as e:
        print(f"Error loading performance data: {e}")
        return {
            "has_data": False,
            "metrics": {},
            "history": [],
        }


def create_empty_state() -> html.Div:
    """Create empty state message when no betting data exists."""
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="bi bi-bar-chart", style={
                        "fontSize": "4rem",
                        "color": COLORS["text_muted"],
                    }),
                    html.H4("No Performance Data Yet", className="mt-3 mb-2"),
                    html.P(
                        "Start tracking your bets to see performance analytics here.",
                        className="text-muted mb-4"
                    ),
                    html.P([
                        "Go to the ",
                        html.A("Predictions", href="/", className="text-primary"),
                        " page and select a game to make your first pick."
                    ], className="text-muted"),
                ], className="text-center py-5"),
            ]),
        ]),
    ])


def create_roi_chart(history: list) -> dcc.Graph:
    """Create ROI over time chart from real data."""
    if not history:
        # Empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS["text_muted"])
        )
    else:
        # Calculate cumulative ROI from history
        dates = []
        cumulative_roi = []
        running_pnl = 0
        total_wagered = 0

        for bet in sorted(history, key=lambda x: x.get("placed_at", "")):
            dates.append(bet.get("placed_at", "")[:10])
            running_pnl += bet.get("pnl", 0)
            total_wagered += bet.get("stake", 0)
            roi = (running_pnl / total_wagered * 100) if total_wagered > 0 else 0
            cumulative_roi.append(roi)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_roi,
            mode='lines+markers',
            name='ROI',
            line=dict(color=COLORS["accent_primary"], width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(88, 166, 255, 0.1)',
        ))

        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_muted"],
                      annotation_text="Break-even")

    fig.update_layout(
        title="Cumulative ROI Over Time",
        xaxis_title="Date",
        yaxis_title="ROI (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_primary"]),
        xaxis=dict(gridcolor=COLORS["border_color"], showgrid=True),
        yaxis=dict(gridcolor=COLORS["border_color"], showgrid=True, ticksuffix="%"),
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
        hovermode="x unified",
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_win_rate_chart(history: list) -> dcc.Graph:
    """Create win rate by bet type chart from real data."""
    if not history:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS["text_muted"])
        )
    else:
        # Calculate win rates by bet type
        bet_types = {}
        for bet in history:
            bt = bet.get("bet_type", "Other")
            if bt not in bet_types:
                bet_types[bt] = {"wins": 0, "total": 0}
            bet_types[bt]["total"] += 1
            if bet.get("status") == "won":
                bet_types[bt]["wins"] += 1

        labels = list(bet_types.keys())
        win_rates = [
            (bt["wins"] / bt["total"] * 100) if bt["total"] > 0 else 0
            for bt in bet_types.values()
        ]

        colors = [
            COLORS["accent_success"] if wr >= 52.4 else COLORS["accent_danger"]
            for wr in win_rates
        ]

        fig = go.Figure(go.Bar(
            x=labels,
            y=win_rates,
            marker_color=colors,
            text=[f"{wr:.1f}%" for wr in win_rates],
            textposition='outside',
            textfont=dict(color=COLORS["text_primary"]),
        ))

        fig.add_hline(y=52.4, line_dash="dash", line_color=COLORS["accent_warning"],
                      annotation_text="52.4% Break-even", annotation_position="right")

    fig.update_layout(
        title="Win Rate by Bet Type",
        xaxis_title="Bet Type",
        yaxis_title="Win Rate (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_primary"]),
        xaxis=dict(gridcolor=COLORS["border_color"]),
        yaxis=dict(gridcolor=COLORS["border_color"], range=[0, 70], ticksuffix="%"),
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_drawdown_chart(history: list) -> dcc.Graph:
    """Create drawdown chart from real data."""
    if not history:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS["text_muted"])
        )
    else:
        # Calculate drawdown from history
        dates = []
        drawdowns = []
        running_pnl = 0
        peak_pnl = 0

        for bet in sorted(history, key=lambda x: x.get("placed_at", "")):
            dates.append(bet.get("placed_at", "")[:10])
            running_pnl += bet.get("pnl", 0)
            peak_pnl = max(peak_pnl, running_pnl)
            dd = ((running_pnl - peak_pnl) / max(peak_pnl, 1)) * 100 if peak_pnl > 0 else 0
            drawdowns.append(min(0, dd))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdowns,
            fill='tozeroy',
            fillcolor='rgba(248, 81, 73, 0.3)',
            line=dict(color=COLORS["accent_danger"], width=1),
            name='Drawdown',
        ))

        if drawdowns:
            max_dd = min(drawdowns)
            fig.add_hline(y=max_dd, line_dash="dash", line_color=COLORS["accent_danger"],
                          annotation_text=f"Max DD: {max_dd:.1f}%")

    fig.update_layout(
        title="Drawdown History",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_primary"]),
        xaxis=dict(gridcolor=COLORS["border_color"]),
        yaxis=dict(gridcolor=COLORS["border_color"], ticksuffix="%"),
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_stat_card(value: str, label: str, sublabel: str = None,
                     color: str = None) -> dbc.Card:
    """Create a stat card for performance metrics."""
    value_color = color if color else COLORS["text_primary"]

    children = [
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
    ]

    if sublabel:
        children.append(html.Div(sublabel, style={
            "fontSize": "0.75rem",
            "color": COLORS["text_secondary"],
            "marginTop": "4px",
        }))

    return dbc.Card([
        dbc.CardBody([
            html.Div(children, className="text-center"),
        ]),
    ], className="h-100")


def layout():
    """Create the performance page layout."""
    # Fetch real data
    data = get_performance_data()

    if not data["has_data"]:
        return html.Div([
            html.Div([
                html.H2("Performance Analytics", className="mb-1"),
                html.P("Track model accuracy and betting performance", className="text-muted"),
            ], className="mb-4"),
            create_empty_state(),
        ])

    # Extract metrics
    metrics = data.get("metrics", {})
    history = data.get("history", [])

    total_bets = len(history)
    wins = len([b for b in history if b.get("status") == "won"])
    losses = len([b for b in history if b.get("status") == "lost"])
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0

    total_pnl = sum(b.get("pnl", 0) for b in history)
    total_wagered = sum(b.get("stake", 0) for b in history)
    roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0

    # Calculate max drawdown
    running_pnl = 0
    peak_pnl = 0
    max_drawdown = 0
    for bet in sorted(history, key=lambda x: x.get("placed_at", "")):
        running_pnl += bet.get("pnl", 0)
        peak_pnl = max(peak_pnl, running_pnl)
        dd = ((running_pnl - peak_pnl) / max(peak_pnl, 1)) * 100 if peak_pnl > 0 else 0
        max_drawdown = min(max_drawdown, dd)

    return html.Div([
        # Page Header
        html.Div([
            html.H2("Performance Analytics", className="mb-1"),
            html.P("Track model accuracy and betting performance", className="text-muted"),
        ], className="mb-4"),

        # Date Range Selector
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("7D", id="range-7d", color="secondary", outline=True, size="sm"),
                    dbc.Button("30D", id="range-30d", color="primary", size="sm"),
                    dbc.Button("90D", id="range-90d", color="secondary", outline=True, size="sm"),
                    dbc.Button("All", id="range-all", color="secondary", outline=True, size="sm"),
                ]),
            ], className="mb-4"),
        ]),

        # Key Metrics Row
        dbc.Row([
            dbc.Col([
                create_stat_card(
                    f"{roi:+.1f}%",
                    "Total ROI",
                    f"{total_bets} bets",
                    COLORS["accent_success"] if roi > 0 else COLORS["accent_danger"]
                ),
            ], md=3, sm=6, className="mb-3"),
            dbc.Col([
                create_stat_card(
                    f"{win_rate:.1f}%",
                    "Win Rate",
                    f"{wins} of {total_bets} bets",
                    COLORS["accent_success"] if win_rate >= 52.4 else COLORS["accent_danger"]
                ),
            ], md=3, sm=6, className="mb-3"),
            dbc.Col([
                create_stat_card(
                    f"{max_drawdown:.1f}%",
                    "Max Drawdown",
                    None,
                    COLORS["accent_danger"]
                ),
            ], md=3, sm=6, className="mb-3"),
            dbc.Col([
                create_stat_card(
                    f"${total_pnl:+.2f}",
                    "Total P&L",
                    f"${total_wagered:.2f} wagered",
                    COLORS["accent_success"] if total_pnl > 0 else COLORS["accent_danger"]
                ),
            ], md=3, sm=6, className="mb-3"),
        ], className="mb-4"),

        # Charts Row 1
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        create_roi_chart(history),
                    ]),
                ]),
            ], lg=8, className="mb-4"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        create_win_rate_chart(history),
                    ]),
                ]),
            ], lg=4, className="mb-4"),
        ]),

        # Charts Row 2
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        create_drawdown_chart(history),
                    ]),
                ]),
            ], lg=12, className="mb-4"),
        ]),

        # Detailed Statistics
        dbc.Card([
            dbc.CardHeader(html.H5("Detailed Statistics", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Strong("Avg Odds: ", className="text-muted"),
                            html.Span(f"{sum(b.get('odds', 0) for b in history) / max(len(history), 1):.0f}"),
                        ], className="mb-2"),
                        html.Div([
                            html.Strong("Total Wagered: ", className="text-muted"),
                            html.Span(f"${total_wagered:.2f}"),
                        ]),
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            html.Strong("Total Bets: ", className="text-muted"),
                            html.Span(f"{total_bets}"),
                        ], className="mb-2"),
                        html.Div([
                            html.Strong("Avg Stake: ", className="text-muted"),
                            html.Span(f"${total_wagered / max(total_bets, 1):.2f}"),
                        ]),
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            html.Strong("Wins: ", className="text-muted"),
                            html.Span(f"{wins}", className="text-success"),
                        ], className="mb-2"),
                        html.Div([
                            html.Strong("Losses: ", className="text-muted"),
                            html.Span(f"{losses}", className="text-danger"),
                        ]),
                    ], md=4),
                ]),
            ]),
        ]),

        # Hidden stores
        dcc.Store(id="performance-data-store", data={}),
    ])
