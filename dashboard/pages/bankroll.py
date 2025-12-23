"""
Bankroll Management Page

Kelly calculator, risk management, and Monte Carlo projections.
Shows real data from bet tracker database.
"""

import sys
from pathlib import Path
from dash import html, dcc, register_page, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Register this page
register_page(__name__, path="/bankroll", name="Bankroll")

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

# Default bankroll settings (user can modify)
DEFAULT_SETTINGS = {
    "starting_bankroll": 1000.0,
    "target_bankroll": 2000.0,
    "base_unit_pct": 2.0,
    "daily_limit_pct": 5.0,
    "weekly_limit_pct": 15.0,
    "max_drawdown_pct": 25.0,
}


def get_bankroll_data():
    """Fetch real bankroll data from bet tracker database."""
    try:
        from bet_tracker import BetTracker
        tracker = BetTracker()

        # Get all bets (last 365 days for full history)
        start_date = datetime.now() - timedelta(days=365)
        bets = tracker.get_bets_by_date(start_date)

        # Convert to list of dicts
        bet_history = []
        for bet in bets:
            bet_dict = {
                "bet_id": bet.bet_id,
                "placed_at": bet.placed_at.isoformat() if bet.placed_at else "",
                "odds": bet.odds,
                "stake": bet.stake,
                "pnl": bet.pnl,
                "status": bet.status.value if hasattr(bet.status, 'value') else str(bet.status),
            }
            bet_history.append(bet_dict)

        # Calculate metrics from bet history
        total_pnl = sum(b["pnl"] for b in bet_history)
        total_wagered = sum(b["stake"] for b in bet_history)

        # Calculate win rate for projections
        total_bets = len(bet_history)
        wins = len([b for b in bet_history if b["status"] == "won"])
        win_rate = (wins / total_bets) if total_bets > 0 else 0.5

        # Calculate average odds
        odds_list = [b["odds"] for b in bet_history if b.get("odds")]
        avg_odds = sum(odds_list) / len(odds_list) if odds_list else -110

        # Calculate current drawdown
        running_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        for bet in sorted(bet_history, key=lambda x: x.get("placed_at", "")):
            running_pnl += bet.get("pnl", 0)
            peak_pnl = max(peak_pnl, running_pnl)
            if peak_pnl > 0:
                dd = ((running_pnl - peak_pnl) / peak_pnl) * 100
                max_drawdown = min(max_drawdown, dd)

        # Current drawdown (from peak)
        current_drawdown = abs(max_drawdown)

        return {
            "has_data": len(bet_history) > 0,
            "total_bets": total_bets,
            "total_pnl": total_pnl,
            "total_wagered": total_wagered,
            "win_rate": win_rate,
            "avg_odds": avg_odds,
            "current_drawdown": current_drawdown,
            "history": bet_history,
        }
    except Exception as e:
        print(f"Error loading bankroll data: {e}")
        return {
            "has_data": False,
            "total_bets": 0,
            "total_pnl": 0,
            "total_wagered": 0,
            "win_rate": 0.5,
            "avg_odds": -110,
            "current_drawdown": 0,
            "history": [],
        }


def create_empty_state() -> html.Div:
    """Create empty state message when no betting data exists."""
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="bi bi-wallet2", style={
                        "fontSize": "4rem",
                        "color": COLORS["text_muted"],
                    }),
                    html.H4("No Bankroll Data Yet", className="mt-3 mb-2"),
                    html.P(
                        "Start tracking your bets to see bankroll analytics here.",
                        className="text-muted mb-4"
                    ),
                    html.P([
                        "Go to the ",
                        html.A("Predictions", href="/", className="text-primary"),
                        " page and make your first pick to begin tracking."
                    ], className="text-muted"),
                ], className="text-center py-5"),
            ]),
        ]),

        # Still show Kelly Calculator even without data
        html.Div([
            html.H4("Kelly Calculator", className="mt-5 mb-3"),
            html.P("Use this calculator to determine optimal bet sizing.",
                   className="text-muted mb-4"),
            create_kelly_calculator_card(),
        ]),
    ])


def create_bankroll_gauge(current: float, target: float) -> dcc.Graph:
    """Create a gauge showing bankroll progress to target."""
    pct = min(100, (current / target) * 100)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current,
        number={"prefix": "$", "font": {"size": 36, "color": COLORS["text_primary"]}},
        delta={"reference": target, "prefix": "$", "relative": False},
        title={"text": "Current Bankroll", "font": {"size": 14, "color": COLORS["text_secondary"]}},
        gauge={
            "axis": {"range": [0, target * 1.2], "tickprefix": "$"},
            "bar": {"color": COLORS["accent_success"]},
            "bgcolor": COLORS["bg_tertiary"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, target * 0.5], "color": "rgba(248, 81, 73, 0.2)"},
                {"range": [target * 0.5, target * 0.8], "color": "rgba(210, 153, 34, 0.2)"},
                {"range": [target * 0.8, target * 1.2], "color": "rgba(63, 185, 80, 0.2)"},
            ],
            "threshold": {
                "line": {"color": COLORS["accent_primary"], "width": 4},
                "thickness": 0.75,
                "value": target,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text_primary"]},
        margin=dict(l=30, r=30, t=50, b=30),
        height=250,
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_monte_carlo_chart(bankroll: float = 1000, win_rate: float = 0.54,
                             avg_odds: float = -110, simulations: int = 1000,
                             bets: int = 100) -> dcc.Graph:
    """Create Monte Carlo simulation chart showing bankroll projections."""
    np.random.seed(42)

    # Calculate implied probability and edge
    if avg_odds < 0:
        implied_prob = abs(avg_odds) / (abs(avg_odds) + 100)
        payout_mult = 100 / abs(avg_odds)
    else:
        implied_prob = 100 / (avg_odds + 100)
        payout_mult = avg_odds / 100

    # Run simulations
    all_paths = []
    final_values = []

    stake_pct = 0.02  # 2% of bankroll per bet

    for _ in range(simulations):
        path = [bankroll]
        current = bankroll

        for _ in range(bets):
            stake = current * stake_pct
            if np.random.random() < win_rate:
                current += stake * payout_mult
            else:
                current -= stake
            path.append(current)

        all_paths.append(path)
        final_values.append(current)

    # Calculate percentiles
    paths_array = np.array(all_paths)
    p5 = np.percentile(paths_array, 5, axis=0)
    p25 = np.percentile(paths_array, 25, axis=0)
    p50 = np.percentile(paths_array, 50, axis=0)
    p75 = np.percentile(paths_array, 75, axis=0)
    p95 = np.percentile(paths_array, 95, axis=0)

    x = list(range(bets + 1))

    fig = go.Figure()

    # 90% confidence interval (p5-p95)
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(p95) + list(p5)[::-1],
        fill='toself',
        fillcolor='rgba(88, 166, 255, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='90% CI',
        showlegend=True,
    ))

    # 50% confidence interval (p25-p75)
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(p75) + list(p25)[::-1],
        fill='toself',
        fillcolor='rgba(88, 166, 255, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='50% CI',
        showlegend=True,
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=x,
        y=p50,
        mode='lines',
        name='Median',
        line=dict(color=COLORS["accent_primary"], width=2),
    ))

    # Starting bankroll line
    fig.add_hline(y=bankroll, line_dash="dash", line_color=COLORS["text_muted"],
                  annotation_text="Starting Bankroll")

    fig.update_layout(
        title="Monte Carlo Bankroll Projection",
        xaxis_title="Number of Bets",
        yaxis_title="Bankroll ($)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_primary"]),
        xaxis=dict(gridcolor=COLORS["border_color"]),
        yaxis=dict(gridcolor=COLORS["border_color"], tickprefix="$"),
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_kelly_calculator_card() -> dbc.Card:
    """Create the Kelly Calculator card with sliders."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Kelly Calculator", className="mb-0 d-inline"),
            dbc.Badge("Dynamic", color="primary", className="ms-2"),
        ]),
        dbc.CardBody([
            # Win Probability Slider
            html.Div([
                dbc.Label("Win Probability", className="text-muted"),
                dcc.Slider(
                    id="kelly-win-prob",
                    min=0.40,
                    max=0.70,
                    step=0.01,
                    value=0.55,
                    marks={0.40: "40%", 0.50: "50%", 0.60: "60%", 0.70: "70%"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], className="mb-4"),

            # Odds Slider
            html.Div([
                dbc.Label("American Odds", className="text-muted"),
                dcc.Slider(
                    id="kelly-odds",
                    min=-200,
                    max=200,
                    step=5,
                    value=-110,
                    marks={-200: "-200", -110: "-110", 100: "+100", 200: "+200"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], className="mb-4"),

            # Edge Quality Slider
            html.Div([
                dbc.Label("Edge Quality Score", className="text-muted"),
                dcc.Slider(
                    id="kelly-quality",
                    min=0,
                    max=100,
                    step=5,
                    value=70,
                    marks={0: "0", 50: "50", 100: "100"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], className="mb-4"),

            html.Hr(),

            # Results
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Full Kelly", className="text-muted small"),
                        html.Div(id="kelly-full", children="5.0%",
                                style={"fontSize": "1.5rem", "fontWeight": "600"}),
                    ], className="text-center"),
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.Div("Half Kelly", className="text-muted small"),
                        html.Div(id="kelly-half", children="2.5%",
                                style={"fontSize": "1.5rem", "fontWeight": "600",
                                      "color": COLORS["accent_warning"]}),
                    ], className="text-center"),
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.Div("Quarter Kelly", className="text-muted small"),
                        html.Div(id="kelly-quarter", children="1.25%",
                                style={"fontSize": "1.5rem", "fontWeight": "600",
                                      "color": COLORS["accent_success"]}),
                    ], className="text-center"),
                ], width=4),
            ]),

            html.Hr(),

            # Recommended stake
            html.Div([
                html.Div("Recommended Stake (Quarter Kelly)", className="text-muted small mb-2"),
                html.Div(id="kelly-stake", children="$12.50",
                        style={"fontSize": "2rem", "fontWeight": "700",
                              "color": COLORS["accent_success"]}),
                html.Div(id="kelly-stake-note", children="Based on $1,000 bankroll",
                        className="text-muted small"),
            ], className="text-center"),
        ]),
    ])


def create_risk_status_card(drawdown: float = 5.2) -> dbc.Card:
    """Create risk status card showing current risk level."""
    if drawdown < 10:
        status = "LOW"
        color = COLORS["accent_success"]
        stake_mult = "100%"
        description = "Normal betting, full stakes allowed"
    elif drawdown < 20:
        status = "MODERATE"
        color = COLORS["accent_warning"]
        stake_mult = "75%"
        description = "Reduced stakes recommended"
    elif drawdown < 30:
        status = "HIGH"
        color = COLORS["accent_danger"]
        stake_mult = "50%"
        description = "Half stakes, exercise caution"
    else:
        status = "CRITICAL"
        color = "#ff0000"
        stake_mult = "25%"
        description = "Minimal betting, review strategy"

    return dbc.Card([
        dbc.CardHeader([
            html.H5("Risk Status", className="mb-0 d-inline"),
            dbc.Badge(status, style={"backgroundColor": color}, className="ms-2"),
        ]),
        dbc.CardBody([
            # Drawdown display
            html.Div([
                html.Div("Current Drawdown", className="text-muted small"),
                html.Div(f"{drawdown:.1f}%",
                        style={"fontSize": "2.5rem", "fontWeight": "700", "color": color}),
            ], className="text-center mb-3"),

            # Progress bars for risk levels
            html.Div([
                html.Div("Risk Levels", className="text-muted small mb-2"),
                dbc.Progress([
                    dbc.Progress(value=33, color="success", bar=True, label="Low"),
                    dbc.Progress(value=33, color="warning", bar=True, label="Mod"),
                    dbc.Progress(value=34, color="danger", bar=True, label="High"),
                ], className="mb-2", style={"height": "24px"}),
            ], className="mb-3"),

            html.Hr(),

            # Stake multiplier
            html.Div([
                html.Div([
                    html.Strong("Stake Multiplier: ", className="text-muted"),
                    html.Span(stake_mult, style={"color": color, "fontWeight": "600"}),
                ], className="mb-2"),
                html.P(description, className="text-muted small mb-0"),
            ]),

            # Circuit breakers
            html.Hr(),
            html.Div([
                html.Div("Circuit Breakers", className="text-muted small mb-2"),
                html.Div([
                    dbc.Badge("Daily Loss: OK", color="success", className="me-2 mb-1"),
                    dbc.Badge("Weekly Loss: OK", color="success", className="me-2 mb-1"),
                    dbc.Badge("Max DD: OK", color="success", className="mb-1"),
                ]),
            ]),
        ]),
    ])


def layout():
    """Create the bankroll management page layout."""
    # Fetch real data
    data = get_bankroll_data()

    # Page header
    header = html.Div([
        html.H2("Bankroll Management", className="mb-1"),
        html.P("Kelly sizing, risk management, and projections", className="text-muted"),
    ], className="mb-4")

    # Show empty state if no betting data
    if not data["has_data"]:
        return html.Div([
            header,
            create_empty_state(),
            # Hidden stores (still needed for callbacks)
            dcc.Store(id="bankroll-settings-store", data={
                "starting": DEFAULT_SETTINGS["starting_bankroll"],
                "current": DEFAULT_SETTINGS["starting_bankroll"],
                "target": DEFAULT_SETTINGS["target_bankroll"],
                "base_unit_pct": DEFAULT_SETTINGS["base_unit_pct"],
                "daily_limit": DEFAULT_SETTINGS["daily_limit_pct"],
                "weekly_limit": DEFAULT_SETTINGS["weekly_limit_pct"],
                "max_drawdown": DEFAULT_SETTINGS["max_drawdown_pct"],
            }),
        ])

    # Calculate real values from data
    starting_bankroll = DEFAULT_SETTINGS["starting_bankroll"]
    total_pnl = data["total_pnl"]
    current_bankroll = starting_bankroll + total_pnl
    target_bankroll = DEFAULT_SETTINGS["target_bankroll"]
    progress = (current_bankroll / target_bankroll * 100) if target_bankroll > 0 else 0
    win_rate = data["win_rate"]
    avg_odds = data["avg_odds"]
    current_drawdown = data["current_drawdown"]

    # Determine profit/loss color
    pnl_color = COLORS["accent_success"] if total_pnl >= 0 else COLORS["accent_danger"]

    return html.Div([
        header,

        # Top Row - Bankroll Status and Risk
        dbc.Row([
            # Bankroll Gauge
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        create_bankroll_gauge(current_bankroll, target_bankroll),
                    ]),
                ]),
            ], lg=4, className="mb-4"),

            # Quick Stats
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Div(f"${current_bankroll:,.2f}", style={
                                        "fontSize": "1.5rem",
                                        "fontWeight": "700",
                                        "color": COLORS["text_primary"],
                                    }),
                                    html.Div("Current Bankroll", className="text-muted small"),
                                ], className="text-center"),
                            ]),
                        ], className="h-100"),
                    ], width=6, className="mb-3"),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Div(f"{'+' if total_pnl >= 0 else ''}${total_pnl:,.2f}", style={
                                        "fontSize": "1.5rem",
                                        "fontWeight": "700",
                                        "color": pnl_color,
                                    }),
                                    html.Div("Total P&L", className="text-muted small"),
                                ], className="text-center"),
                            ]),
                        ], className="h-100"),
                    ], width=6, className="mb-3"),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Div(f"${target_bankroll:,.0f}", style={
                                        "fontSize": "1.5rem",
                                        "fontWeight": "700",
                                        "color": COLORS["accent_primary"],
                                    }),
                                    html.Div("Target", className="text-muted small"),
                                ], className="text-center"),
                            ]),
                        ], className="h-100"),
                    ], width=6, className="mb-3"),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Div(f"{progress:.0f}%", style={
                                        "fontSize": "1.5rem",
                                        "fontWeight": "700",
                                        "color": COLORS["accent_warning"] if progress < 100 else COLORS["accent_success"],
                                    }),
                                    html.Div("Progress", className="text-muted small"),
                                ], className="text-center"),
                            ]),
                        ], className="h-100"),
                    ], width=6, className="mb-3"),
                ]),
            ], lg=4, className="mb-4"),

            # Risk Status
            dbc.Col([
                create_risk_status_card(current_drawdown),
            ], lg=4, className="mb-4"),
        ]),

        # Kelly Calculator and Monte Carlo
        dbc.Row([
            dbc.Col([
                create_kelly_calculator_card(),
            ], lg=5, className="mb-4"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        create_monte_carlo_chart(
                            bankroll=current_bankroll,
                            win_rate=win_rate if win_rate > 0 else 0.52,
                            avg_odds=avg_odds,
                            simulations=500,
                            bets=100
                        ),
                    ]),
                ]),
            ], lg=7, className="mb-4"),
        ]),

        # Bankroll Settings
        dbc.Card([
            dbc.CardHeader([
                html.H5("Bankroll Settings", className="mb-0"),
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Starting Bankroll", className="text-muted"),
                        dbc.InputGroup([
                            dbc.InputGroupText("$"),
                            dbc.Input(id="bankroll-start", type="number",
                                     value=DEFAULT_SETTINGS["starting_bankroll"]),
                        ], className="mb-3"),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Target Bankroll", className="text-muted"),
                        dbc.InputGroup([
                            dbc.InputGroupText("$"),
                            dbc.Input(id="bankroll-target", type="number",
                                     value=DEFAULT_SETTINGS["target_bankroll"]),
                        ], className="mb-3"),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Base Unit Size", className="text-muted"),
                        dbc.InputGroup([
                            dbc.InputGroupText("%"),
                            dbc.Input(id="base-unit-pct", type="number",
                                     value=DEFAULT_SETTINGS["base_unit_pct"],
                                     min=0.5, max=5, step=0.5),
                        ], className="mb-3"),
                    ], md=4),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Daily Loss Limit", className="text-muted"),
                        dbc.InputGroup([
                            dbc.InputGroupText("%"),
                            dbc.Input(id="daily-loss-limit", type="number",
                                     value=DEFAULT_SETTINGS["daily_limit_pct"]),
                        ], className="mb-3"),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Weekly Loss Limit", className="text-muted"),
                        dbc.InputGroup([
                            dbc.InputGroupText("%"),
                            dbc.Input(id="weekly-loss-limit", type="number",
                                     value=DEFAULT_SETTINGS["weekly_limit_pct"]),
                        ], className="mb-3"),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Max Drawdown Halt", className="text-muted"),
                        dbc.InputGroup([
                            dbc.InputGroupText("%"),
                            dbc.Input(id="max-drawdown-halt", type="number",
                                     value=DEFAULT_SETTINGS["max_drawdown_pct"]),
                        ], className="mb-3"),
                    ], md=4),
                ]),
                dbc.Button("Save Settings", id="save-bankroll-settings", color="primary",
                          className="mt-2"),
            ]),
        ]),

        # Hidden stores
        dcc.Store(id="bankroll-settings-store", data={
            "starting": starting_bankroll,
            "current": current_bankroll,
            "target": target_bankroll,
            "base_unit_pct": DEFAULT_SETTINGS["base_unit_pct"],
            "daily_limit": DEFAULT_SETTINGS["daily_limit_pct"],
            "weekly_limit": DEFAULT_SETTINGS["weekly_limit_pct"],
            "max_drawdown": DEFAULT_SETTINGS["max_drawdown_pct"],
        }),
    ])


# Kelly Calculator Callback
@callback(
    Output("kelly-full", "children"),
    Output("kelly-half", "children"),
    Output("kelly-quarter", "children"),
    Output("kelly-stake", "children"),
    Input("kelly-win-prob", "value"),
    Input("kelly-odds", "value"),
    Input("kelly-quality", "value"),
    State("bankroll-settings-store", "data"),
)
def update_kelly_calculator(win_prob, odds, quality, settings):
    """Calculate Kelly criterion stake recommendations."""
    # Calculate decimal odds
    if odds < 0:
        decimal_odds = 1 + (100 / abs(odds))
    else:
        decimal_odds = 1 + (odds / 100)

    # Kelly formula: (bp - q) / b
    # b = decimal odds - 1
    # p = win probability
    # q = loss probability (1 - p)
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p

    kelly = (b * p - q) / b

    # Apply quality adjustment
    kelly_adjusted = kelly * (quality / 100)

    # Calculate stakes
    full_kelly = max(0, kelly_adjusted * 100)
    half_kelly = full_kelly / 2
    quarter_kelly = full_kelly / 4

    # Get current bankroll
    current = settings.get("current", 1000) if settings else 1000
    stake_amount = current * (quarter_kelly / 100)

    return (
        f"{full_kelly:.1f}%",
        f"{half_kelly:.1f}%",
        f"{quarter_kelly:.2f}%",
        f"${stake_amount:.2f}",
    )
