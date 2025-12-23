"""
NBA Betting Dashboard - Main Application

A simple, single-page Dash app for NBA player props.

Usage:
    python -m dashboard.app
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dash import Dash
import dash_bootstrap_components as dbc

from dashboard.data_service import get_data_service
from dashboard.pages import predictions

# Import callbacks to register them
from dashboard import callbacks  # noqa: F401


def create_app() -> Dash:
    """Create and configure the Dash application."""

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="NBA Props",
    )

    # Pre-initialize data service (loads models)
    print("Initializing data service...")
    get_data_service()
    print("Data service ready.")

    # Single page layout - no routing needed
    app.layout = predictions.layout()

    return app


# Create application instance
app = create_app()
server = app.server  # For WSGI deployment


if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("NBA Player Props Dashboard")
    print("=" * 40)
    print("Starting server at http://localhost:8050")
    print("=" * 40 + "\n")

    app.run(
        debug=False,
        port=8050,
        host="0.0.0.0",
        use_reloader=False,
    )
