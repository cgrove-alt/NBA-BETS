"""
Navigation Bar Component

Provides the main navigation bar for the dashboard.
"""

from dash import html
import dash_bootstrap_components as dbc


def create_navbar(current_page: str = "predictions") -> dbc.Navbar:
    """Create the main navigation bar.

    Args:
        current_page: The current active page identifier

    Returns:
        dbc.Navbar component
    """
    nav_items = [
        {"label": "Predictions", "href": "/", "id": "predictions"},
        {"label": "Bet Tracker", "href": "/tracker", "id": "tracker"},
        {"label": "Performance", "href": "/performance", "id": "performance"},
        {"label": "Bankroll", "href": "/bankroll", "id": "bankroll"},
    ]

    nav_links = []
    for item in nav_items:
        is_active = item["id"] == current_page
        nav_links.append(
            dbc.NavItem(
                dbc.NavLink(
                    item["label"],
                    href=item["href"],
                    active=is_active,
                    className="nav-link" + (" active" if is_active else ""),
                )
            )
        )

    return dbc.Navbar(
        dbc.Container([
            # Brand/Logo
            dbc.NavbarBrand([
                html.Span("NBA", style={"fontWeight": "700", "color": "#58a6ff"}),
                html.Span(" Betting", style={"fontWeight": "400"}),
            ], href="/", className="me-auto"),

            # Mobile toggle
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),

            # Navigation links
            dbc.Collapse(
                dbc.Nav(nav_links, className="ms-auto", navbar=True),
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ], fluid=True),
        color="dark",
        dark=True,
        className="navbar mb-4",
        sticky="top",
    )


def create_simple_navbar() -> html.Div:
    """Create a simplified navigation bar without collapse (for multi-page apps)."""
    return html.Div([
        dbc.Container([
            dbc.Row([
                # Brand
                dbc.Col([
                    html.A([
                        html.Span("NBA", style={"fontWeight": "700", "color": "#58a6ff"}),
                        html.Span(" Betting", style={"fontWeight": "400", "color": "#e6edf3"}),
                    ], href="/", style={"textDecoration": "none", "fontSize": "1.25rem"}),
                ], width="auto"),

                # Navigation
                dbc.Col([
                    html.Div([
                        dbc.NavLink("Predictions", href="/", className="nav-link d-inline-block px-3"),
                        dbc.NavLink("Tracker", href="/tracker", className="nav-link d-inline-block px-3"),
                        dbc.NavLink("Performance", href="/performance", className="nav-link d-inline-block px-3"),
                        dbc.NavLink("Bankroll", href="/bankroll", className="nav-link d-inline-block px-3"),
                    ], className="text-end"),
                ]),
            ], align="center"),
        ], fluid=True),
    ], className="navbar py-3 mb-4", style={
        "backgroundColor": "#161b22",
        "borderBottom": "1px solid #30363d",
    })
