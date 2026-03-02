"""
FastAPI Backend for NBA Props Dashboard

Wraps existing DataService (ML/prediction logic) with REST endpoints.
DO NOT modify the underlying data_service.py - this is a read-only wrapper.

Usage:
    uvicorn backend.api:app --reload --port 8000

Authentication (Optional):
    Set AUTH_ENABLED=true to enable JWT authentication
    Set JWT_SECRET_KEY for production security
    Set API_KEY for simple API key authentication

    Protected endpoints support:
        - Bearer token in Authorization header
        - X-API-Key header for API key auth

    Generate token: POST /api/auth/token
    Verify token: GET /api/auth/verify

New Endpoints (Task 4.4):
    - GET /api/predictions/{date} - Daily predictions with confidence & bet sizing
    - GET /api/injuries/{date} - Injury report for specific date
    - GET /api/line-movement/{game_id} - Odds history and movement analysis
    - GET /api/backtest/latest - Latest backtest results
    - POST /api/auth/token - Generate JWT token (if AUTH_ENABLED)
    - GET /api/auth/verify - Verify JWT token (if AUTH_ENABLED)
"""

import load_env  # noqa: F401  — load .env before any code reads os.environ
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Eastern Time for date-sensitive operations
ET = ZoneInfo('America/New_York')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.data_service import get_data_service, DataService
from backend.schemas import (
    HealthResponse,
    GamesResponse,
    Game,
    Team,
    PropsResponse,
    PlayerProp,
    PropPrediction,
    StartPropsRequest,
    AnalysisStatus,
    GameAnalysis,
    MoneylinePrediction,
    SpreadPrediction,
    OddsResponse,
    GameOdds,
    MoneylineOdds,
    SpreadOdds,
    TotalOdds,
    BestBetsResponse,
    BestBet,
    GameResults,
    PlayerResult,
    FinalScore,
    MoneylineResult,
    ResultsSummary,
    DailyPredictionsResponse,
    DailyPrediction,
    InjuryReportResponse,
    InjuryReport,
    LineMovementResponse,
    OddsSnapshot,
    LineMovement,
    LatestBacktestResponse,
    BacktestResults,
    BacktestMetrics,
    BacktestBettingMetrics,
    BacktestByProp,
    BankrollResponse,
    PerformanceResponse,
    DailyRecord,
    PropTypeStats,
    ConfidenceTierStats,
    CalibrationSummaryResponse,
    SystemHealthResponse,
    AgentStatus,
    ModelStatus,
    BriefingResponse,
    BriefingSections,
    SettingsResponse,
    SettingsUpdateRequest,
)

# Singleton data service instance
_data_service: DataService | None = None

# Cache for game team mappings (game_id -> {"home": abbrev, "away": abbrev})
_game_teams_cache: dict = {}


def get_service() -> DataService:
    """Get the DataService singleton instance."""
    global _data_service
    if _data_service is None:
        _data_service = get_data_service()
    return _data_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize data service on startup, run migrations if needed."""
    print("Initializing NBA Props API...")

    # Run database migrations on startup (idempotent — safe to run every time)
    try:
        from scripts.run_migrations import run_migrations
        ok = run_migrations()
        if ok:
            print("Database migrations: up to date.")
        else:
            print("WARNING: Database migration failed — some features may be unavailable.")
    except Exception as e:
        print(f"WARNING: Could not run migrations: {e} — continuing with existing schema.")

    service = get_service()
    print("Data service ready.")

    # Auto-generate props for today's games at startup so /api/best-bets
    # returns data immediately instead of waiting for a /api/games call.
    try:
        games = service.get_todays_games()
        triggered = 0
        for game in games:
            game_id = str(game.get("game_id", ""))
            home_abbrev = game.get("home_team", {}).get("abbreviation", "")
            away_abbrev = game.get("visitor_team", {}).get("abbreviation", "")
            status_data = service.get_props_fetch_status(game_id)
            if status_data.get("status") == "not_started" and home_abbrev and away_abbrev:
                _game_teams_cache[game_id] = {"home": home_abbrev, "away": away_abbrev}
                service.start_player_props_fetch(
                    game_id=game_id,
                    home_abbrev=home_abbrev,
                    away_abbrev=away_abbrev,
                    selected_props=None,
                )
                triggered += 1
        if triggered:
            print(f"Auto-generating props for {triggered} games...")
    except Exception as e:
        print(f"WARNING: Startup prop generation failed: {e}")

    yield
    print("Shutting down NBA Props API.")


# Create FastAPI app
app = FastAPI(
    title="NBA Props API",
    description="REST API for NBA player prop predictions",
    version="2.0.0",
    lifespan=lifespan,
)

# Initialize authentication endpoints (optional - controlled by AUTH_ENABLED env var)
try:
    from backend.auth import add_auth_endpoints
    add_auth_endpoints(app)
except ImportError:
    pass  # Auth module optional

# Configure CORS for React frontend
# Add your production Vercel URL here after deployment
import os
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]
# Add production URL from environment variable if set
if os.environ.get("FRONTEND_URL"):
    CORS_ORIGINS.append(os.environ.get("FRONTEND_URL"))
# Also allow any Vercel preview URLs
CORS_ORIGINS.append("https://*.vercel.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",  # Allow all Vercel subdomains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Set Cache-Control headers based on response path."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path

        if path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif path == "/sw.js":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        elif path == "/" or path.endswith(".html"):
            response.headers["Cache-Control"] = "no-cache"
        elif any(path.endswith(ext) for ext in (".json", ".png", ".svg", ".ico")):
            response.headers["Cache-Control"] = "public, max-age=3600, stale-while-revalidate=86400"

        return response


app.add_middleware(CacheControlMiddleware)


# ============== HEALTH CHECK ==============

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint — verifies actual system state."""
    from datetime import datetime
    from pathlib import Path

    checks: dict[str, Any] = {}

    # 1. PostgreSQL check
    db_connected = False
    try:
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            import psycopg2
            conn = psycopg2.connect(db_url)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()
            db_connected = True
            checks["database"] = "connected"
        else:
            checks["database"] = "DATABASE_URL not set (SQLite fallback)"
    except Exception as e:
        checks["database"] = f"error: {e}"

    # 2. Redis check (non-fatal — agents have in-memory fallback)
    redis_connected = False
    try:
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            import redis as redis_lib
            r = redis_lib.Redis.from_url(redis_url, decode_responses=True)
            r.ping()
            redis_connected = True
            checks["redis"] = "connected"
        else:
            checks["redis"] = "REDIS_URL not set (in-memory fallback)"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    # 3. Models check
    models_dir = Path("models")
    pkl_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []
    models_loaded = len(pkl_files) > 0
    checks["models"] = f"{len(pkl_files)} .pkl files found"

    # 4. Determine overall status
    if db_connected and models_loaded:
        status = "healthy"
    elif models_loaded and not db_connected and os.environ.get("DATABASE_URL") or not models_loaded:
        status = "unhealthy"
    else:
        # Redis down or DATABASE_URL not set (local dev) — degraded at worst
        status = "degraded" if (os.environ.get("REDIS_URL") and not redis_connected) else "healthy"

    from fastapi.responses import JSONResponse
    status_code = 503 if status == "unhealthy" else 200

    return JSONResponse(
        status_code=status_code,
        content=HealthResponse(
            status=status,
            service="nba-props-api",
            timestamp=datetime.now(ET).isoformat(),
            models_loaded=models_loaded,
            database_connected=db_connected,
            redis_connected=redis_connected,
            checks=checks,
        ).model_dump(),
    )


# ============== GAMES ENDPOINTS ==============

@app.get("/api/games", response_model=GamesResponse)
def get_games(
    date: str | None = Query(None, description="Date in YYYY-MM-DD format (defaults to today Eastern)"),
    force_refresh: bool = Query(False, description="Force refresh from API"),
    auto_generate_props: bool = Query(True, description="Automatically generate props for all games")
):
    """Get NBA games for a specific date.

    Args:
        date: Date string in YYYY-MM-DD format. Defaults to today (Eastern timezone).
        force_refresh: If True, bypass cache and fetch fresh data.
        auto_generate_props: If True, automatically trigger prop generation for all games (default: True).
    """
    global _game_teams_cache
    service = get_service()
    games_data = service.get_todays_games(force_refresh=force_refresh, date=date)

    games = []
    for g in games_data:
        home = g.get("home_team", {})
        visitor = g.get("visitor_team", {})
        game_id = str(g.get("game_id", ""))

        games.append(Game(
            game_id=game_id,
            home_team=Team(
                id=home.get("id", 0),
                abbreviation=home.get("abbreviation", ""),
                city=home.get("city"),
                name=home.get("name"),
            ),
            visitor_team=Team(
                id=visitor.get("id", 0),
                abbreviation=visitor.get("abbreviation", ""),
                city=visitor.get("city"),
                name=visitor.get("name"),
            ),
            game_time=g.get("game_time"),
            status=g.get("status"),
        ))

        # AUTO-GENERATION: Automatically trigger prop generation for each game
        # This ensures predictions are ready when frontend calls /api/best-bets
        if auto_generate_props:
            home_abbrev = home.get("abbreviation", "")
            away_abbrev = visitor.get("abbreviation", "")

            # Check if props already exist or are being generated
            status_data = service.get_props_fetch_status(game_id)
            if status_data.get("status") == "not_started":
                # Cache team abbreviations
                _game_teams_cache[game_id] = {"home": home_abbrev, "away": away_abbrev}

                # Start background prop generation (non-blocking)
                try:
                    service.start_player_props_fetch(
                        game_id=game_id,
                        home_abbrev=home_abbrev,
                        away_abbrev=away_abbrev,
                        selected_props=None,  # All prop types
                    )
                except Exception as e:
                    # Log error but don't fail the request
                    print(f"Warning: Could not auto-generate props for game {game_id}: {e}")

    return GamesResponse(games=games, count=len(games))


# ============== PROPS ENDPOINTS ==============

def _build_prop_prediction(player_data: dict, prop_key: str) -> PropPrediction | None:
    """Build PropPrediction from flattened player data keys.

    DataService returns flattened keys like:
        points_pred, points_line, points_pick, points_edge, points_confidence

    This function extracts them and builds a nested PropPrediction object.
    """
    pred_key = f"{prop_key}_pred"
    if pred_key not in player_data or player_data.get(pred_key) is None:
        return None

    # Get line - use None for missing/invalid lines instead of 0
    line = player_data.get(f"{prop_key}_line")
    if line is not None and line <= 0:
        line = None  # Treat 0 or negative as "no line available"

    prediction = player_data.get(pred_key, 0) or 0
    edge_pct_from_ds = player_data.get(f"{prop_key}_edge", 0) or 0  # Percentage from DataService
    confidence = player_data.get(f"{prop_key}_confidence", 50) or 50
    pick = player_data.get(f"{prop_key}_pick", "-") or "-"

    # Calculate raw edge (points) for frontend display
    # Frontend expects edge to be raw points (e.g., +2.5), not percentage
    raw_edge = prediction - line if line and line > 0 else 0

    return PropPrediction(
        prediction=prediction,
        confidence=confidence,
        edge=raw_edge,            # Raw points (e.g., +2.5)
        edge_pct=edge_pct_from_ds,  # Percentage (e.g., 10.2%)
        pick=pick,
        line=line,  # Can be None now
        implied_probability=None,
    )


def _convert_player_prop(player_data: dict, is_best_bet: bool = False, team_abbrev: str = None) -> PlayerProp:
    """Convert raw player prop dict to PlayerProp schema.

    DataService returns data with flattened keys (e.g., points_pred, rebounds_line).
    This function transforms them into nested PropPrediction objects.

    Args:
        player_data: Raw player data from DataService
        is_best_bet: Whether this player qualifies as a best bet
        team_abbrev: Team abbreviation to use (injected since DataService doesn't include it)
    """
    return PlayerProp(
        player_name=player_data.get("player_name", "Unknown"),
        player_id=player_data.get("player_id", 0),
        team=team_abbrev or player_data.get("team_abbrev"),
        position=player_data.get("position"),
        avg_minutes=player_data.get("avg_minutes"),
        Points=_build_prop_prediction(player_data, "points"),
        Rebounds=_build_prop_prediction(player_data, "rebounds"),
        Assists=_build_prop_prediction(player_data, "assists"),
        three_pm=_build_prop_prediction(player_data, "3pm"),
        PRA=_build_prop_prediction(player_data, "pra"),
        is_best_bet=is_best_bet,
    )


@app.get("/api/games/{game_id}/props", response_model=PropsResponse)
def get_props(game_id: str):
    """Get player props for a specific game."""
    global _game_teams_cache
    service = get_service()
    status_data = service.get_props_fetch_status(game_id)

    status = status_data.get("status", "not_started")
    error = status_data.get("error")
    home_props_raw = status_data.get("home", [])
    away_props_raw = status_data.get("away", [])

    # Handle locked status - game has started, predictions are frozen
    if status == "locked":
        return PropsResponse(
            game_id=game_id,
            status="locked",
            error=error or "Game has started - predictions locked for betting integrity",
            home_props=[],
            away_props=[],
            all_props=[],
            count=0,
        )

    # Get team abbreviations from cache (set when props fetch was started)
    cached_teams = _game_teams_cache.get(game_id, {})
    home_abbrev = cached_teams.get("home")
    away_abbrev = cached_teams.get("away")

    # Determine best bets (confidence >= 65, edge >= 2.0)
    # Lowered from 80%/2.5 because the model's confidence naturally caps at ~70%
    # The heuristic/quantile calculations produce 50-70% range realistically
    def is_best_bet(player: dict) -> bool:
        for prop_key in ["points", "rebounds", "assists", "3pm", "pra"]:
            conf = player.get(f"{prop_key}_confidence", 0) or 0
            edge = abs(player.get(f"{prop_key}_edge", 0) or 0)
            pick = player.get(f"{prop_key}_pick", "-")
            if pick != "-" and conf >= 65 and edge >= 2.0:
                return True
        return False

    # Inject team abbreviations when converting player props
    home_props = [_convert_player_prop(p, is_best_bet(p), team_abbrev=home_abbrev) for p in home_props_raw]
    away_props = [_convert_player_prop(p, is_best_bet(p), team_abbrev=away_abbrev) for p in away_props_raw]
    all_props = home_props + away_props

    # Get team names from cache
    home_team = home_abbrev
    away_team = away_abbrev

    return PropsResponse(
        game_id=game_id,
        status=status,
        home_team=home_team,
        away_team=away_team,
        home_props=home_props,
        away_props=away_props,
        all_props=all_props,
        count=len(all_props),
    )


@app.post("/api/games/{game_id}/props/start")
def start_props_fetch(
    game_id: str,
    home_abbrev: str = Query(..., description="Home team abbreviation"),
    away_abbrev: str = Query(..., description="Away team abbreviation"),
    request: StartPropsRequest | None = None,
):
    """Start background fetch of player props for a game."""
    global _game_teams_cache
    service = get_service()

    # Cache team abbreviations for this game
    _game_teams_cache[game_id] = {"home": home_abbrev, "away": away_abbrev}

    prop_types = None
    if request and request.prop_types:
        prop_types = request.prop_types

    service.start_player_props_fetch(
        game_id=game_id,
        home_abbrev=home_abbrev,
        away_abbrev=away_abbrev,
        selected_props=prop_types,
    )

    return {"message": "Props fetch started", "game_id": game_id}


@app.get("/api/games/{game_id}/live-stats")
def get_live_stats(game_id: str):
    """Get live player stats for an in-progress or completed game.

    Returns real-time player statistics during games (via Balldontlie GOAT tier).
    For completed games, returns final box score stats.
    """
    from datetime import datetime
    service = get_service()

    # First check if we have a cached game status
    games = service.get_todays_games()
    game_status = None
    for g in games:
        if str(g.get('id')) == str(game_id):
            game_status = g.get('status', '')
            break

    # Get stats based on game status
    if game_status == 'Final':
        # Use final box score for completed games
        stats = service.get_game_final_stats(game_id)
    else:
        # Use live box scores for in-progress games
        stats = service.get_live_player_stats(game_id)

    return {
        "game_id": game_id,
        "status": game_status or "unknown",
        "stats": stats,
        "timestamp": datetime.now().isoformat()
    }


# ============== ANALYSIS ENDPOINTS ==============

@app.get("/api/games/{game_id}/analysis/status", response_model=AnalysisStatus)
def get_analysis_status(game_id: str):
    """Check status of game analysis."""
    service = get_service()
    status_data = service.get_analysis_status(game_id)

    moneyline = None
    spread = None

    if status_data.get("status") == "ready":
        ml_data = status_data.get("moneyline")
        if ml_data and isinstance(ml_data, dict):
            moneyline = MoneylinePrediction(
                home_win_probability=ml_data.get("home_win_probability", 0.5),
                away_win_probability=ml_data.get("away_win_probability", 0.5),
                predicted_winner=ml_data.get("predicted_winner", "home"),
                confidence=ml_data.get("confidence", 0.5),
                calibrated=ml_data.get("calibrated", False),
            )

        sp_data = status_data.get("spread")
        if sp_data and isinstance(sp_data, dict):
            spread = SpreadPrediction(
                predicted_spread=sp_data.get("predicted_spread", 0.0),
                confidence=sp_data.get("confidence", 0.5),
            )

    return AnalysisStatus(
        game_id=game_id,
        status=status_data.get("status", "not_started"),
        moneyline=moneyline,
        spread=spread,
        error=status_data.get("error"),
    )


@app.post("/api/games/{game_id}/analysis/start")
def start_game_analysis(
    game_id: str,
    home_abbrev: str = Query(..., description="Home team abbreviation"),
    away_abbrev: str = Query(..., description="Away team abbreviation"),
):
    """Start background game analysis with ML models."""
    service = get_service()
    service.start_game_analysis(game_id, home_abbrev, away_abbrev)
    return {"message": "Analysis started", "game_id": game_id}


@app.get("/api/games/{game_id}/analysis", response_model=GameAnalysis)
def get_game_analysis(game_id: str):
    """Get complete analysis for a game."""
    service = get_service()
    analysis = service.get_game_analysis(game_id)

    if not analysis:
        raise HTTPException(status_code=404, detail=f"No analysis found for game {game_id}")

    # Convert moneyline prediction - only if valid predictions exist
    moneyline = None
    ml_data = analysis.get("moneyline_prediction")
    if ml_data and isinstance(ml_data, dict):
        # Check if this is a real prediction (has probability) or just a status
        if "home_win_probability" in ml_data and ml_data.get("status") != "unavailable":
            moneyline = MoneylinePrediction(
                home_win_probability=ml_data.get("home_win_probability", 0.5),
                away_win_probability=ml_data.get("away_win_probability", 0.5),
                predicted_winner=ml_data.get("predicted_winner", "home"),
                confidence=ml_data.get("confidence", 0.5),
                calibrated=ml_data.get("calibrated", False),
            )

    # Convert spread prediction - only if valid predictions exist
    spread = None
    sp_data = analysis.get("spread_prediction")
    if sp_data and isinstance(sp_data, dict):
        # Check if this is a real prediction or just a status
        if "predicted_spread" in sp_data and sp_data.get("status") != "unavailable":
            spread = SpreadPrediction(
                predicted_spread=sp_data.get("predicted_spread", 0.0),
                confidence=sp_data.get("confidence", 0.5),
            )

    return GameAnalysis(
        game_id=game_id,
        home_team=analysis.get("home_team", ""),
        home_abbrev=analysis.get("home_abbrev", ""),
        away_team=analysis.get("away_team", ""),
        away_abbrev=analysis.get("away_abbrev", ""),
        game_time=analysis.get("game_time"),
        status=analysis.get("status"),
        moneyline_prediction=moneyline,
        spread_prediction=spread,
        market_odds=analysis.get("market_odds"),
        recommendations=analysis.get("recommendations", []),
    )


# ============== GAME RESULTS ENDPOINT ==============

@app.get("/api/games/{game_id}/results", response_model=GameResults)
def get_game_results(game_id: str):
    """Get actual results for a completed game with prediction comparison."""
    service = get_service()

    # Find the game
    games = service.get_todays_games()
    game = next((g for g in games if str(g.get('game_id')) == game_id), None)

    if not game:
        return GameResults(
            game_id=game_id,
            status="error",
            message="Game not found"
        )

    if game.get('status') != 'Final':
        return GameResults(
            game_id=game_id,
            status="not_completed",
            message="Game not yet completed"
        )

    # Fetch player stats using get_player_stats (NOT get_box_score which uses non-existent endpoint)
    try:
        from balldontlie_api import BalldontlieAPI
        api = BalldontlieAPI()
        player_stats_list = api.get_player_stats(game_ids=[int(game_id)])
    except Exception as e:
        return GameResults(
            game_id=game_id,
            status="error",
            message=f"Could not fetch stats: {str(e)}"
        )

    if not player_stats_list:
        return GameResults(
            game_id=game_id,
            status="error",
            message="Player stats not available for this game"
        )

    # Extract scores from the first player's game data
    home_abbrev = game.get('home_team', {}).get('abbreviation', '')
    away_abbrev = game.get('visitor_team', {}).get('abbreviation', '')

    # Get game data from first player stat entry (all have same game info)
    game_data = player_stats_list[0].get('game', {})
    home_score = game_data.get('home_team_score', 0) or 0
    away_score = game_data.get('visitor_team_score', 0) or 0

    final_score = FinalScore(
        home_team=home_abbrev,
        home_score=home_score,
        away_team=away_abbrev,
        away_score=away_score
    )

    # Determine actual winner for moneyline result
    actual_winner = "home" if home_score > away_score else "away"

    # Get stored predictions from prop_tracker
    try:
        from prop_tracker import PropTracker
        tracker = PropTracker()
        predictions = tracker.get_predictions_for_game(game_id)
    except Exception:
        predictions = []

    # FALLBACK 1: If no stored predictions, try props cache (DataService)
    if not predictions:
        props_data = service.get_props_fetch_status(game_id)
        if props_data.get("status") == "ready":
            home_props = props_data.get("home", [])
            away_props = props_data.get("away", [])
            all_props = home_props + away_props

            # Convert cached props to prediction format
            for player in all_props:
                player_id = player.get("player_id")
                player_name = player.get("player_name", "Unknown")
                team = player.get("team_abbrev", "")

                for prop_key in ["points", "rebounds", "assists", "3pm", "pra"]:
                    pred_val = player.get(f"{prop_key}_pred")
                    if pred_val is not None:
                        predictions.append({
                            "player_id": player_id,
                            "player_name": player_name,
                            "team_abbrev": team,
                            "prop_type": prop_key,
                            "predicted_value": pred_val,
                            "market_line": player.get(f"{prop_key}_line"),
                            "pick": player.get(f"{prop_key}_pick", "-"),
                        })

    # Get moneyline prediction for comparison
    analysis = service.get_game_analysis(game_id)
    moneyline_result = None
    if analysis:
        ml_data = analysis.get("moneyline_prediction")
        if ml_data and isinstance(ml_data, dict) and "home_win_probability" in ml_data:
            predicted_winner = ml_data.get("predicted_winner", "home")
            moneyline_result = MoneylineResult(
                predicted_winner=predicted_winner,
                actual_winner=actual_winner,
                correct=(predicted_winner == actual_winner),
                home_win_probability=ml_data.get("home_win_probability"),
                away_win_probability=ml_data.get("away_win_probability"),
            )

    # Extract player stats from player_stats_list for comparison
    player_stats = {}
    for stat in player_stats_list:
        player_info = stat.get('player', {})
        player_id = player_info.get('id')
        if player_id:
            player_stats[player_id] = {
                'pts': stat.get('pts', 0) or 0,
                'reb': stat.get('reb', 0) or 0,
                'ast': stat.get('ast', 0) or 0,
                'fg3m': stat.get('fg3m', 0) or 0,
            }

    # Build player results comparing predictions vs actuals
    player_results = []
    total_picks = 0
    total_hits = 0

    # Map prop types to stat keys
    stat_map = {
        "points": "pts",
        "rebounds": "reb",
        "assists": "ast",
        "3pm": "fg3m",
    }

    for pred in predictions:
        player_id = pred.get('player_id')
        prop_type = pred.get('prop_type', '').lower()
        predicted = pred.get('predicted_value', 0) or 0
        line = pred.get('market_line')
        pick = pred.get('pick')

        # Get actual value
        stats = player_stats.get(player_id, {})
        if prop_type == "pra":
            actual = stats.get('pts', 0) + stats.get('reb', 0) + stats.get('ast', 0)
        else:
            stat_key = stat_map.get(prop_type)
            actual = stats.get(stat_key, 0) if stat_key else 0

        # Determine hit/miss
        hit = None
        if pick and pick != "-" and line is not None:
            total_picks += 1
            if pick == "OVER" and actual > line or pick == "UNDER" and actual < line:
                hit = True
                total_hits += 1
            else:
                hit = False

        # Format prop type for display
        display_prop = prop_type.capitalize()
        if prop_type == "3pm":
            display_prop = "3PM"
        elif prop_type == "pra":
            display_prop = "PRA"

        player_results.append(PlayerResult(
            player_id=player_id,
            player_name=pred.get('player_name', 'Unknown'),
            team=pred.get('team_abbrev', ''),
            prop_type=display_prop,
            predicted=predicted,
            actual=actual,
            line=line,
            pick=pick,
            hit=hit,
            difference=actual - predicted
        ))

    # Create summary
    summary = ResultsSummary(
        total_predictions=len(predictions),
        total_picks=total_picks,
        total_hits=total_hits,
        hit_rate=total_hits / total_picks if total_picks > 0 else 0.0
    )

    return GameResults(
        game_id=game_id,
        status="completed",
        final_score=final_score,
        moneyline_result=moneyline_result,
        player_results=player_results,
        summary=summary
    )


# ============== ODDS ENDPOINTS ==============

def _convert_game_odds(odds_data: dict, game_id: str = None) -> GameOdds:
    """Convert raw odds dict to GameOdds schema."""
    moneyline = None
    spread = None
    total = None

    ml_data = odds_data.get("moneyline")
    if ml_data and isinstance(ml_data, dict):
        moneyline = MoneylineOdds(
            home=ml_data.get("home", 0),
            away=ml_data.get("away", 0),
        )

    sp_data = odds_data.get("spread")
    if sp_data and isinstance(sp_data, dict):
        spread = SpreadOdds(
            home_line=sp_data.get("home_line", 0.0),
            home_odds=sp_data.get("home_odds", 0),
            away_line=sp_data.get("away_line", 0.0),
            away_odds=sp_data.get("away_odds", 0),
        )

    total_data = odds_data.get("total")
    if total_data and isinstance(total_data, dict):
        total = TotalOdds(
            line=total_data.get("line", 0.0),
            over_odds=total_data.get("over_odds", 0),
            under_odds=total_data.get("under_odds", 0),
        )

    return GameOdds(
        game_id=game_id,
        moneyline=moneyline,
        spread=spread,
        total=total,
        sportsbook=odds_data.get("sportsbook"),
        last_updated=odds_data.get("last_updated"),
    )


@app.get("/api/odds", response_model=OddsResponse)
def get_all_odds():
    """Get betting odds for all games."""
    service = get_service()
    odds_data = service.get_betting_odds()

    converted = {}
    if isinstance(odds_data, dict):
        for game_id, game_odds in odds_data.items():
            if isinstance(game_odds, dict):
                converted[game_id] = _convert_game_odds(game_odds, game_id)

    return OddsResponse(odds=converted)


@app.get("/api/odds/{game_id}", response_model=GameOdds)
def get_game_odds(game_id: str):
    """Get betting odds for a specific game."""
    service = get_service()
    odds_data = service.get_betting_odds(game_id)

    if not odds_data:
        raise HTTPException(status_code=404, detail=f"No odds found for game {game_id}")

    return _convert_game_odds(odds_data, game_id)


# ============== BEST BETS ENDPOINT ==============

@app.get("/api/best-bets", response_model=BestBetsResponse)
def get_best_bets(
    min_confidence: float = Query(55.0, ge=0, le=100, description="Minimum confidence threshold (model outputs 50-70%)"),
    min_edge: float = Query(4.0, ge=0, description="Minimum edge threshold (percentage)"),
    prop_types: str | None = Query(None, description="Comma-separated prop types to filter"),
    pick_type: str | None = Query(None, description="Filter by OVER or UNDER"),
    sort_by: str = Query("quality", description="Sort order: quality, confidence, or edge"),
):
    """Get best bets across all games based on confidence and edge thresholds.

    Returns ALL bets meeting quality standards, sorted by user-selected criteria.
    Sort options: quality (confidence * edge), confidence, or edge.
    """
    service = get_service()

    # Get all games
    games = service.get_todays_games()
    best_bets = []

    # Ensure prop generation is running — if no games have props,
    # trigger generation so subsequent calls return data.
    # This makes the endpoint self-sufficient instead of depending on
    # /api/games being called first.
    any_ready = False
    for game in games:
        gid = str(game.get("game_id", ""))
        st = service.get_props_fetch_status(gid).get("status", "not_started")
        if st == "ready":
            any_ready = True
            break
    if not any_ready:
        for game in games:
            gid = str(game.get("game_id", ""))
            status_data = service.get_props_fetch_status(gid)
            if status_data.get("status") == "not_started":
                home_abbrev = game.get("home_team", {}).get("abbreviation", "")
                away_abbrev = game.get("visitor_team", {}).get("abbreviation", "")
                if home_abbrev and away_abbrev:
                    _game_teams_cache[gid] = {"home": home_abbrev, "away": away_abbrev}
                    try:
                        service.start_player_props_fetch(
                            game_id=gid,
                            home_abbrev=home_abbrev,
                            away_abbrev=away_abbrev,
                            selected_props=None,
                        )
                    except Exception:
                        pass

    # Parse prop types filter
    prop_type_filter = None
    if prop_types:
        prop_type_filter = [p.strip() for p in prop_types.split(",")]

    for game in games:
        game_id = str(game.get("game_id", ""))
        status_data = service.get_props_fetch_status(game_id)

        if status_data.get("status") != "ready":
            continue

        all_players = status_data.get("home", []) + status_data.get("away", [])

        for player in all_players:
            player_name = player.get("player_name", "Unknown")
            player_id = player.get("player_id", 0)
            team = player.get("team", "")

            # Map display names to flattened key prefixes
            prop_key_map = {
                "Points": "points",
                "Rebounds": "rebounds",
                "Assists": "assists",
                "3PM": "3pm",
                "PRA": "pra",
            }

            for prop_type, prop_key in prop_key_map.items():
                # Apply prop type filter
                if prop_type_filter and prop_type not in prop_type_filter:
                    continue

                # Extract from flattened keys
                pred_key = f"{prop_key}_pred"
                if pred_key not in player or player.get(pred_key) is None:
                    continue

                prediction = player.get(pred_key, 0) or 0
                line = player.get(f"{prop_key}_line", 0) or 0
                edge_from_ds = player.get(f"{prop_key}_edge", 0) or 0  # This is ALREADY a percentage
                confidence = player.get(f"{prop_key}_confidence", 0) or 0
                pick = player.get(f"{prop_key}_pick", "-") or "-"

                # data_service.py returns edge as a PERCENTAGE (line 3608)
                # But BestBet schema expects TWO fields:
                #   - edge: raw points difference (e.g., 1.5 assists)
                #   - edge_pct: percentage (e.g., 300%)
                edge = prediction - line  # Raw points
                edge_pct = edge_from_ds  # Already a percentage from data_service

                # Apply filters
                if confidence < min_confidence:
                    continue
                # CRITICAL: Filter using edge_pct (percentage), not edge (raw points)
                # min_edge=4.0 means "4% edge minimum", not "4 points minimum"
                # This ensures low-line props (assists, threes) aren't filtered out
                if abs(edge_pct) < min_edge:
                    continue
                if pick == "-":
                    continue
                if pick_type and pick != pick_type.upper():
                    continue

                # --- Extract explanation data from player dict ---
                injury_notes = player.get(f"{prop_key}_injury_notes", []) or []
                matchup_notes = player.get(f"{prop_key}_matchup_notes", []) or []
                opp_def = player.get(f"{prop_key}_opp_def", None)
                used_real_line = player.get(f"{prop_key}_real_line", False)
                used_ml_model = player.get(f"{prop_key}_ml_model", False)

                # Get season and recent averages
                season_avgs = player.get("season_averages", {}) or {}
                recent_avgs = player.get("recent_averages", {}) or {}

                # Map prop_key to the average dict keys
                avg_key_map = {
                    "points": "pts_avg",
                    "rebounds": "reb_avg",
                    "assists": "ast_avg",
                    "3pm": "fg3_avg",
                    "pra": None,
                }
                avg_key = avg_key_map.get(prop_key)
                season_avg_val = None
                recent_avg_val = None
                if avg_key:
                    season_avg_val = season_avgs.get(avg_key)
                    recent_avg_val = recent_avgs.get(avg_key)
                elif prop_key == "pra":
                    s_pts = season_avgs.get("pts_avg", 0) or 0
                    s_reb = season_avgs.get("reb_avg", 0) or 0
                    s_ast = season_avgs.get("ast_avg", 0) or 0
                    season_avg_val = s_pts + s_reb + s_ast if (s_pts + s_reb + s_ast) > 0 else None
                    r_pts = recent_avgs.get("pts_avg", 0) or 0
                    r_reb = recent_avgs.get("reb_avg", 0) or 0
                    r_ast = recent_avgs.get("ast_avg", 0) or 0
                    recent_avg_val = r_pts + r_reb + r_ast if (r_pts + r_reb + r_ast) > 0 else None

                # Build signals list
                signals = []
                if used_ml_model:
                    signals.append("ML Model")
                if used_real_line:
                    signals.append("Real Line")
                for note in injury_notes:
                    signals.append(note)
                for note in matchup_notes:
                    signals.append(note)
                if opp_def is not None and opp_def > 112:
                    signals.append(f"Weak defense ({opp_def:.0f} DEF RTG)")
                elif opp_def is not None and opp_def < 108:
                    signals.append(f"Strong defense ({opp_def:.0f} DEF RTG)")

                # Build human-readable explanation
                explanation_parts = []
                explanation_parts.append(f"Model predicts {prediction:.1f} {prop_type.lower()} (line: {line})")
                if season_avg_val:
                    explanation_parts.append(f"Season avg: {season_avg_val:.1f}")
                if recent_avg_val and season_avg_val and abs(recent_avg_val - season_avg_val) > 0.5:
                    direction = "up" if recent_avg_val > season_avg_val else "down"
                    explanation_parts.append(f"Trending {direction} (last 5: {recent_avg_val:.1f})")
                for note in injury_notes:
                    explanation_parts.append(note)
                for note in matchup_notes:
                    explanation_parts.append(note)
                explanation = ". ".join(explanation_parts)

                best_bets.append(BestBet(
                    player_name=player_name,
                    player_id=player_id,
                    team=team,
                    game_id=game_id,
                    prop_type=prop_type,
                    prediction=prediction,
                    line=line,
                    edge=edge,
                    edge_pct=edge_pct,
                    pick=pick,
                    confidence=confidence,
                    season_avg=round(season_avg_val, 1) if season_avg_val else None,
                    recent_avg=round(recent_avg_val, 1) if recent_avg_val else None,
                    explanation=explanation,
                    signals=signals,
                    used_real_line=bool(used_real_line),
                    used_ml_model=bool(used_ml_model),
                ))

    # Sort based on user preference
    if sort_by == "confidence":
        best_bets.sort(key=lambda x: x.confidence, reverse=True)
    elif sort_by == "edge":
        best_bets.sort(key=lambda x: abs(x.edge_pct), reverse=True)
    else:  # Default "quality" - composite score
        best_bets.sort(key=lambda x: (x.confidence - 50) * abs(x.edge_pct), reverse=True)

    # Assign rank after sorting (1 = best)
    for i, bet in enumerate(best_bets):
        bet.rank = i + 1

    return BestBetsResponse(
        best_bets=best_bets,
        count=len(best_bets),
        filters={
            "min_confidence": min_confidence,
            "min_edge": min_edge,
            "prop_types": prop_type_filter,
            "pick_type": pick_type,
            "sort_by": sort_by,
        },
    )


# ============== RETRAIN STATUS ENDPOINT ==============

@app.get("/api/retrain/status")
def get_retrain_status():
    """Get status of last retrain and continuous learning system.

    Returns information about:
    - Last full model retrain (from Railway cron or manual)
    - Continuous learning system status (settlements, drift detection)
    - Model age in days
    """
    import json
    from datetime import datetime

    service = get_service()

    # Get retrain history from log file
    retrain_log = Path("logs/retrain_history.json")
    last_retrain = None
    retrain_history = []

    if retrain_log.exists():
        try:
            with open(retrain_log) as f:
                retrain_history = json.load(f)
                if retrain_history:
                    last_retrain = retrain_history[-1]
        except (OSError, json.JSONDecodeError):
            pass

    # Calculate model age (days since last successful retrain)
    model_age_days = None
    if last_retrain and last_retrain.get("success"):
        try:
            last_ts = datetime.fromisoformat(last_retrain["timestamp"])
            model_age_days = (datetime.now() - last_ts).days
        except (ValueError, KeyError):
            pass

    # Get continuous learning status if available
    cl_status = None
    if hasattr(service, 'get_continuous_learning_status'):
        cl_status = service.get_continuous_learning_status()
    elif hasattr(service, 'orchestrator') and service.orchestrator:
        cl_status = {
            "enabled": True,
            "message": "Continuous learning orchestrator active"
        }

    # Get model file ages
    models_dir = Path("models")
    model_files = {}
    if models_dir.exists():
        for pkl_file in models_dir.glob("*.pkl"):
            try:
                mtime = datetime.fromtimestamp(pkl_file.stat().st_mtime)
                model_files[pkl_file.name] = {
                    "last_modified": mtime.isoformat(),
                    "age_days": (datetime.now() - mtime).days
                }
            except OSError:
                pass

    return {
        "last_full_retrain": last_retrain,
        "retrain_count": len(retrain_history),
        "model_age_days": model_age_days,
        "continuous_learning": cl_status,
        "models": model_files,
        "timestamp": datetime.now().isoformat(),
    }


# ============== DAILY PREDICTIONS ENDPOINT ==============

@app.get("/api/predictions/{date}", response_model=DailyPredictionsResponse)
def get_daily_predictions(date: str):
    """Get daily predictions for a specific date.

    Args:
        date: Date string in YYYY-MM-DD format

    Returns:
        Daily predictions with confidence, bet sizing, and recommendations
    """
    import pandas as pd
    from pathlib import Path

    # Validate date format
    try:
        from datetime import datetime
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD"
        )

    # Check for predictions file
    csv_path = Path(f"predictions_{date}.csv")

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for {date}. Generate predictions first."
        )

    # Load predictions CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading predictions file: {str(e)}"
        )

    # Convert to prediction objects
    predictions = []
    for _, row in df.iterrows():
        # Handle NaN values for string fields - pandas reads empty cells as NaN
        team = row.get('team', '')
        team = str(team) if pd.notna(team) and team != '' else ''

        uncertainty_flag = row.get('uncertainty_flag')
        if pd.notna(uncertainty_flag) and uncertainty_flag != '':
            uncertainty_flag = str(uncertainty_flag)
        else:
            uncertainty_flag = None

        pick = row.get('pick')
        pick = str(pick) if pd.notna(pick) and pick != '' else None

        edge_quality_tier = row.get('edge_quality_tier')
        edge_quality_tier = str(edge_quality_tier) if pd.notna(edge_quality_tier) else None

        bet_recommendation = row.get('bet_recommendation')
        bet_recommendation = str(bet_recommendation) if pd.notna(bet_recommendation) else None

        predictions.append(DailyPrediction(
            player_name=row.get('player_name', 'Unknown'),
            team=team,
            prop_type=row.get('prop_type', ''),
            prediction=float(row.get('prediction', 0)),
            pred_low=float(row['pred_low']) if pd.notna(row.get('pred_low')) else None,
            pred_median=float(row['pred_median']) if pd.notna(row.get('pred_median')) else None,
            pred_high=float(row['pred_high']) if pd.notna(row.get('pred_high')) else None,
            line=float(row['line']) if pd.notna(row.get('line')) else None,
            confidence_score=float(row['confidence_score']) if pd.notna(row.get('confidence_score')) else None,
            edge_quality_tier=edge_quality_tier,
            suggested_bet_size=float(row['suggested_bet_size']) if pd.notna(row.get('suggested_bet_size')) else None,
            bet_recommendation=bet_recommendation,
            uncertainty_flag=uncertainty_flag,
            pick=pick,
            edge=float(row['edge']) if pd.notna(row.get('edge')) else None,
        ))

    return DailyPredictionsResponse(
        date=date,
        predictions=predictions,
        count=len(predictions),
        metadata={
            "file_path": str(csv_path),
            "total_elite_bets": len([p for p in predictions if p.edge_quality_tier == "elite"]),
            "total_strong_bets": len([p for p in predictions if p.edge_quality_tier == "strong"]),
        }
    )


# ============== INJURY REPORT ENDPOINT ==============

@app.get("/api/injuries/{date}", response_model=InjuryReportResponse)
def get_injury_report(date: str):
    """Get injury report for a specific date.

    Args:
        date: Date string in YYYY-MM-DD format

    Returns:
        List of injured players with status and team info
    """
    from datetime import datetime

    # Validate date format
    try:
        target_date = datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD"
        )

    # Import injury tracker
    try:
        from injury_tracker_v3 import fetch_current_injuries
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Injury tracker module not available"
        )

    # Fetch injuries (no date parameter — fetches all current injuries)
    try:
        injuries_data = fetch_current_injuries()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching injuries: {str(e)}"
        )

    # Convert to injury report objects
    # fetch_current_injuries returns InjuryReport objects with fields:
    #   player_id, player_name, status (str), team (str), injury_type, injury_detail, game_date
    injuries = []
    for injury_obj in injuries_data:
        injuries.append(InjuryReport(
            player_id=injury_obj.player_id or 0,
            player_name=injury_obj.player_name or 'Unknown',
            team_id=0,
            team_abbrev=injury_obj.team or '',
            status=str(injury_obj.status) if injury_obj.status else 'UNKNOWN',
            injury_type=injury_obj.injury_type or None,
            detected_at=datetime.now().isoformat(),
        ))

    return InjuryReportResponse(
        date=date,
        injuries=injuries,
        count=len(injuries),
        last_updated=datetime.now().isoformat(),
    )


# ============== LINE MOVEMENT ENDPOINT ==============

@app.get("/api/line-movement/{game_id}", response_model=LineMovementResponse)
def get_line_movement(
    game_id: str,
    market: str = Query("spread", description="Market type: moneyline, spread, total")
):
    """Get line movement history for a specific game.

    Args:
        game_id: Game ID
        market: Market type (moneyline, spread, total)

    Returns:
        Historical odds snapshots and movement analysis
    """
    # Import betting market features
    try:
        from betting_market_features import BettingMarketFeatures
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Betting market features module not available"
        )

    # Initialize features module
    try:
        bmf = BettingMarketFeatures()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing betting features: {str(e)}"
        )

    # Get odds history
    try:
        odds_history = bmf.db.get_odds_history(game_id, market, lookback_minutes=1440)  # 24 hours
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching odds history: {str(e)}"
        )

    # Convert to snapshots
    snapshots = []
    for odds in odds_history:
        snapshots.append(OddsSnapshot(
            timestamp=odds.get('timestamp', ''),
            book_name=odds.get('book_name', ''),
            market=odds.get('market', ''),
            home_odds=odds.get('home_odds'),
            away_odds=odds.get('away_odds'),
            home_line=odds.get('home_line'),
            away_line=odds.get('away_line'),
            total=odds.get('total'),
        ))

    # Calculate movement analysis
    movement = None
    if len(snapshots) >= 2:
        opening = snapshots[0]
        closing = snapshots[-1]

        opening_line = opening.home_line or 0
        closing_line = closing.home_line or 0
        line_move = closing_line - opening_line

        # Detect RLM and steam moves
        try:
            rlm_detected = bmf.detect_reverse_line_movement(game_id, market)
            steam_detected = bmf.detect_steam_move(game_id, market, lookback_minutes=15)
        except:
            rlm_detected = False
            steam_detected = False

        movement = LineMovement(
            opening_line=opening_line,
            closing_line=closing_line,
            movement=line_move,
            rlm_detected=rlm_detected,
            steam_move_detected=steam_detected,
        )

    return LineMovementResponse(
        game_id=game_id,
        market=market,
        odds_history=snapshots,
        movement_analysis=movement,
        count=len(snapshots),
    )


# ============== BACKTEST RESULTS ENDPOINT ==============

@app.get("/api/backtest/latest", response_model=LatestBacktestResponse)
def get_latest_backtest():
    """Get the latest backtest results.

    Returns:
        Most recent backtest results with performance metrics
    """
    import json
    from pathlib import Path
    from datetime import datetime

    # Find all backtest JSON files
    backtest_dir = Path("backtest_results")

    if not backtest_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Backtest results directory not found"
        )

    # Get all JSON files
    backtest_files = list(backtest_dir.glob("*.json"))

    if not backtest_files:
        raise HTTPException(
            status_code=404,
            detail="No backtest results found"
        )

    # Sort by modification time to get latest
    backtest_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Load the latest backtest
    latest_file = backtest_files[0]

    try:
        with open(latest_file) as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading backtest file: {str(e)}"
        )

    # Parse backtest results — handle both old keys (overall_performance, by_prop_type)
    # and new keys (overall_metrics, prop_type_metrics)
    try:
        # Extract overall metrics (try both key conventions)
        overall = data.get('overall_metrics', data.get('overall_performance', {}))
        overall_metrics = BacktestMetrics(
            rmse=overall.get('rmse'),
            mae=overall.get('mae'),
            r2=overall.get('r2'),
            bias=overall.get('bias'),
        )

        # Extract betting metrics
        betting = data.get('betting_performance', {})
        betting_metrics = None
        if betting:
            betting_metrics = BacktestBettingMetrics(
                total_bets=betting.get('total_bets', 0),
                wins=betting.get('wins', 0),
                losses=betting.get('losses', 0),
                pushes=betting.get('pushes', 0),
                win_rate=betting.get('win_rate', 0.0),
                roi=betting.get('roi', 0.0),
                total_wagered=betting.get('total_wagered', 0.0),
                total_profit=betting.get('total_profit', 0.0),
                sharpe_ratio=betting.get('sharpe_ratio'),
                max_drawdown=betting.get('max_drawdown'),
            )

        # Extract by prop type (try both key conventions)
        by_prop = []
        prop_metrics = data.get('prop_type_metrics', data.get('by_prop_type', {}))
        if isinstance(prop_metrics, dict):
            for prop_type, metrics in prop_metrics.items():
                if isinstance(metrics, dict):
                    by_prop.append(BacktestByProp(
                        prop_type=prop_type,
                        metrics=BacktestMetrics(
                            rmse=metrics.get('rmse'),
                            mae=metrics.get('mae'),
                            r2=metrics.get('r2'),
                            bias=metrics.get('bias'),
                        ),
                        count=metrics.get('count', 0),
                    ))

        # Extract elite+strong metrics
        elite_strong = data.get('elite_strong_tier', {})
        elite_strong_metrics = None
        if elite_strong:
            elite_strong_metrics = BacktestMetrics(
                rmse=elite_strong.get('rmse'),
                mae=elite_strong.get('mae'),
                r2=elite_strong.get('r2'),
                bias=elite_strong.get('bias'),
            )

        backtest_result = BacktestResults(
            backtest_id=latest_file.stem,
            date_range=data.get('date_range', 'unknown'),
            games_analyzed=data.get('games_analyzed', data.get('games_processed', 0)),
            total_predictions=data.get('total_predictions', 0),
            overall_metrics=overall_metrics,
            betting_metrics=betting_metrics,
            by_prop_type=by_prop if by_prop else None,
            elite_strong_metrics=elite_strong_metrics,
            confidence_correlation=data.get('confidence_correlation'),
            phase=data.get('phase'),
            timestamp=datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing backtest data: {str(e)}"
        )

    return LatestBacktestResponse(
        latest_backtest=backtest_result,
        available_backtests=[f.stem for f in backtest_files],
        count=len(backtest_files),
    )


# ============== BANKROLL & P&L ENDPOINT ==============

_settings_path = Path(__file__).parent.parent / "data" / "settings.json"

def _load_settings() -> dict:
    """Load settings from JSON file."""
    import json
    defaults = {
        "bankroll": 5000.0,
        "min_edge": 5.0,
        "min_confidence": 55.0,
        "kelly_fraction": 0.25,
        "max_exposure": 10.0,
        "default_bet_size": 100.0,
        "bet_size_type": "fixed",
        "max_bets_per_day": 10,
    }
    if _settings_path.exists():
        try:
            with open(_settings_path) as f:
                stored = json.load(f)
            defaults.update(stored)
        except (OSError, json.JSONDecodeError):
            pass
    return defaults


@app.get("/api/bankroll", response_model=BankrollResponse)
def get_bankroll():
    """Get bankroll state and P&L summary.

    Data sourced from bet_tracking.db (tracked_bets table) and settings.json.
    """
    import sqlite3
    from datetime import datetime, timedelta

    settings = _load_settings()
    initial_bankroll = settings["bankroll"]

    db_path = Path("data/bet_tracking.db")

    # Default empty response
    result = {
        "current_bankroll": initial_bankroll,
        "initial_bankroll": initial_bankroll,
        "daily_pnl": 0.0,
        "weekly_pnl": 0.0,
        "monthly_pnl": 0.0,
        "season_pnl": 0.0,
        "season_roi": 0.0,
        "total_exposure_today": 0.0,
        "total_bets": 0,
        "win_rate": 0.0,
        "active_bets": 0,
    }

    if not db_path.exists():
        return BankrollResponse(**result)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        now = datetime.now(ET)
        today = now.strftime("%Y-%m-%d")
        week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        month_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        # NBA season start (approximate)
        season_start = f"{now.year - (1 if now.month < 10 else 0)}-10-01"

        # The bet_tracker stores bets in a 'bets' table (SQLite)
        # Check table name
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        bet_table = "tracked_bets" if "tracked_bets" in tables else "bets"

        # Get settled bets with pnl info
        # The bet_tracker has: status, pnl, event_date, stake
        rows = conn.execute(f"""
            SELECT status, pnl, event_date, stake
            FROM {bet_table}
            WHERE status IN ('won', 'lost', 'push')
        """).fetchall()

        total_bets = len(rows)
        wins = sum(1 for r in rows if r["status"] == "won")
        season_pnl = sum(float(r["pnl"] or 0) for r in rows
                         if r["event_date"] and r["event_date"] >= season_start)
        monthly_pnl = sum(float(r["pnl"] or 0) for r in rows
                          if r["event_date"] and r["event_date"] >= month_ago)
        weekly_pnl = sum(float(r["pnl"] or 0) for r in rows
                         if r["event_date"] and r["event_date"] >= week_ago)
        daily_pnl = sum(float(r["pnl"] or 0) for r in rows
                        if r["event_date"] and r["event_date"] == today)

        # Active (pending) bets
        pending = conn.execute(f"SELECT COUNT(*) FROM {bet_table} WHERE status = 'pending'").fetchone()
        active_bets = pending[0] if pending else 0

        # Today's exposure
        exposure_row = conn.execute(f"""
            SELECT COALESCE(SUM(stake), 0) FROM {bet_table}
            WHERE status = 'pending' AND event_date = ?
        """, (today,)).fetchone()
        total_exposure = float(exposure_row[0]) if exposure_row else 0.0

        conn.close()

        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0.0
        current_bankroll = initial_bankroll + season_pnl
        season_roi = (season_pnl / initial_bankroll * 100) if initial_bankroll > 0 else 0.0

        result.update({
            "current_bankroll": round(current_bankroll, 2),
            "daily_pnl": round(daily_pnl, 2),
            "weekly_pnl": round(weekly_pnl, 2),
            "monthly_pnl": round(monthly_pnl, 2),
            "season_pnl": round(season_pnl, 2),
            "season_roi": round(season_roi, 1),
            "total_bets": total_bets,
            "win_rate": round(win_rate, 1),
            "active_bets": active_bets,
            "total_exposure_today": round(total_exposure, 2),
        })
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Bankroll query failed: {e}")

    return BankrollResponse(**result)


# ============== PERFORMANCE HISTORY ENDPOINT ==============

@app.get("/api/performance", response_model=PerformanceResponse)
def get_performance(days: int = Query(30, ge=1, le=365)):
    """Get performance history over a time range.

    Args:
        days: Number of days to look back (default 30)
    """
    import sqlite3
    from datetime import datetime, timedelta
    from collections import defaultdict

    now = datetime.now(ET)
    start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    daily_records = []
    by_prop_type: dict[str, dict] = defaultdict(lambda: {"total": 0, "wins": 0, "losses": 0})
    by_confidence: dict[str, dict] = defaultdict(lambda: {"total": 0, "wins": 0})
    total_bets = 0
    total_wins = 0
    total_losses = 0

    # Try calibration DB first (richer data)
    cal_db_path = Path("data/calibration.db")
    if cal_db_path.exists():
        try:
            conn = sqlite3.connect(str(cal_db_path))
            conn.row_factory = sqlite3.Row

            rows = conn.execute("""
                SELECT p.game_date, p.prop_type, p.confidence, p.edge,
                       o.hit, o.clv, o.error
                FROM predictions p
                JOIN outcomes o ON p.id = o.prediction_id
                WHERE p.game_date >= ?
                ORDER BY p.game_date DESC
            """, (start_date,)).fetchall()

            # Group by date
            date_groups: dict[str, list] = defaultdict(list)
            for r in rows:
                date_groups[r["game_date"]].append(r)

            for date_str in sorted(date_groups.keys(), reverse=True):
                group = date_groups[date_str]
                wins = sum(1 for r in group if r["hit"])
                losses = sum(1 for r in group if not r["hit"])
                clvs = [r["clv"] for r in group if r["clv"] is not None]
                profit = (wins * 91 - losses * 100)  # Approximate at -110
                daily_records.append(DailyRecord(
                    date=date_str,
                    wins=wins,
                    losses=losses,
                    pushes=0,
                    roi=round(profit / max(len(group) * 100, 1) * 100, 1),
                    clv_avg=round(sum(clvs) / len(clvs), 2) if clvs else None,
                    profit=round(profit, 2),
                ))

                total_bets += len(group)
                total_wins += wins
                total_losses += losses

                for r in group:
                    pt = r["prop_type"] or "Unknown"
                    by_prop_type[pt]["total"] += 1
                    if r["hit"]:
                        by_prop_type[pt]["wins"] += 1
                    else:
                        by_prop_type[pt]["losses"] += 1

                    conf = r["confidence"] or 0
                    if conf >= 60:
                        tier = "high"
                    elif conf >= 55:
                        tier = "medium"
                    else:
                        tier = "low"
                    by_confidence[tier]["total"] += 1
                    if r["hit"]:
                        by_confidence[tier]["wins"] += 1

            conn.close()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Calibration DB query failed: {e}")

    # Fall back to bet_tracking.db if no calibration data
    if total_bets == 0:
        bt_db_path = Path("data/bet_tracking.db")
        if bt_db_path.exists():
            try:
                conn = sqlite3.connect(str(bt_db_path))
                conn.row_factory = sqlite3.Row
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                bet_table = "tracked_bets" if "tracked_bets" in tables else "bets"

                rows = conn.execute(f"""
                    SELECT event_date, status, pnl, tags
                    FROM {bet_table}
                    WHERE status IN ('won', 'lost', 'push')
                    AND event_date >= ?
                    ORDER BY event_date DESC
                """, (start_date,)).fetchall()

                date_groups = defaultdict(list)
                for r in rows:
                    date_groups[r["event_date"]].append(r)

                for date_str in sorted(date_groups.keys(), reverse=True):
                    group = date_groups[date_str]
                    wins = sum(1 for r in group if r["status"] == "won")
                    losses = sum(1 for r in group if r["status"] == "lost")
                    pushes = sum(1 for r in group if r["status"] == "push")
                    profit = sum(float(r["pnl"] or 0) for r in group)
                    daily_records.append(DailyRecord(
                        date=date_str,
                        wins=wins,
                        losses=losses,
                        pushes=pushes,
                        roi=round(profit / max(len(group) * 100, 1) * 100, 1),
                        clv_avg=None,
                        profit=round(profit, 2),
                    ))
                    total_bets += len(group)
                    total_wins += wins
                    total_losses += losses

                conn.close()
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Bet tracking DB query failed: {e}")

    # Calibration summary from weekly reports
    calibration_summary = None
    if cal_db_path.exists():
        try:
            conn = sqlite3.connect(str(cal_db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM weekly_reports ORDER BY week_ending DESC LIMIT 1").fetchone()
            if row:
                calibration_summary = CalibrationSummaryResponse(
                    total_predictions=row["total_predictions"] or 0,
                    matched_predictions=row["matched_predictions"] or 0,
                    overall_hit_rate=row["overall_hit_rate"],
                    overall_clv=row["overall_clv"],
                    ece=row["ece"],
                )
            conn.close()
        except Exception:
            pass

    overall_hit_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0.0
    total_profit = sum(r.profit for r in daily_records)
    overall_roi = (total_profit / max(total_bets * 100, 1) * 100) if total_bets > 0 else 0.0

    return PerformanceResponse(
        daily_records=daily_records,
        by_prop_type={
            k: PropTypeStats(
                total=v["total"], wins=v["wins"], losses=v["losses"],
                hit_rate=round(v["wins"] / max(v["total"], 1) * 100, 1),
            ) for k, v in by_prop_type.items()
        },
        by_confidence_tier={
            k: ConfidenceTierStats(
                total=v["total"], wins=v["wins"],
                hit_rate=round(v["wins"] / max(v["total"], 1) * 100, 1),
            ) for k, v in by_confidence.items()
        },
        calibration_summary=calibration_summary,
        total_bets=total_bets,
        total_wins=total_wins,
        total_losses=total_losses,
        overall_hit_rate=round(overall_hit_rate, 1),
        overall_roi=round(overall_roi, 1),
    )


# ============== SYSTEM HEALTH ENDPOINT ==============


def _query_agent_status_pg(agent_names: list) -> dict:
    """Query agent run history from PostgreSQL (production).

    The Agent Scheduler writes run data to PostgreSQL via the Guardrails
    class. This function reads it back so the API can report agent status.

    Args:
        agent_names: List of agent name strings to query.

    Returns:
        Dict mapping agent_name to AgentStatus, or empty dict if
        PostgreSQL is unavailable.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return {}

    try:
        import psycopg2
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()

        # Verify the agent_runs table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'agent_runs'
            )
        """)
        if not cur.fetchone()[0]:
            cur.close()
            conn.close()
            return {}

        result: dict = {}
        for agent_name in agent_names:
            # Last run
            cur.execute("""
                SELECT started_at, status, success FROM agent_runs
                WHERE agent_name = %s
                ORDER BY started_at DESC LIMIT 1
            """, (agent_name,))
            row = cur.fetchone()

            # Consecutive failures
            cur.execute("""
                SELECT success FROM agent_runs
                WHERE agent_name = %s
                ORDER BY started_at DESC LIMIT 5
            """, (agent_name,))
            recent = cur.fetchall()

            consecutive_failures = 0
            for r in recent:
                if r[0] == 0:
                    consecutive_failures += 1
                else:
                    break

            # Token usage
            cur.execute("""
                SELECT used_today FROM agent_token_budgets
                WHERE agent_name = %s
            """, (agent_name,))
            budget_row = cur.fetchone()

            result[agent_name] = AgentStatus(
                last_run=str(row[0]) if row else None,
                last_status=row[1] if row else None,
                consecutive_failures=consecutive_failures,
                tokens_used_today=budget_row[0] if budget_row else 0,
            )

        cur.close()
        conn.close()
        return result

    except ImportError:
        return {}
    except Exception:
        return {}


@app.get("/api/system-health", response_model=SystemHealthResponse)
def get_system_health():
    """Get system health status including agent runs, model freshness, and data freshness."""
    import sqlite3
    from datetime import datetime
    import glob as glob_mod

    agents_status: dict[str, AgentStatus] = {}
    agent_names = ["pregame", "postgame", "odds_monitor", "orchestrator", "watchdog", "briefing"]

    # Query agent runs — try PostgreSQL first (production), fall back to SQLite (local)
    try:
        agents_status = _query_agent_status_pg(agent_names)
    except Exception:
        agents_status = {}

    if not agents_status:
        guardrails_path = Path("data/agent_guardrails.db")
        if guardrails_path.exists():
            try:
                conn = sqlite3.connect(str(guardrails_path))
                conn.row_factory = sqlite3.Row

                for agent_name in agent_names:
                    row = conn.execute("""
                        SELECT started_at, status, success FROM agent_runs
                        WHERE agent_name = ?
                        ORDER BY started_at DESC LIMIT 1
                    """, (agent_name,)).fetchone()

                    recent = conn.execute("""
                        SELECT success FROM agent_runs
                        WHERE agent_name = ?
                        ORDER BY started_at DESC LIMIT 5
                    """, (agent_name,)).fetchall()

                    consecutive_failures = 0
                    for r in recent:
                        if r["success"] == 0:
                            consecutive_failures += 1
                        else:
                            break

                    budget = conn.execute("""
                        SELECT used_today FROM agent_token_budgets
                        WHERE agent_name = ?
                    """, (agent_name,)).fetchone()

                    agents_status[agent_name] = AgentStatus(
                        last_run=row["started_at"] if row else None,
                        last_status=row["status"] if row else None,
                        consecutive_failures=consecutive_failures,
                        tokens_used_today=budget["used_today"] if budget else 0,
                    )

                conn.close()
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Guardrails DB query failed: {e}")

    # Fill in agents with no data
    for name in agent_names:
        if name not in agents_status:
            agents_status[name] = AgentStatus()

    # Model freshness
    models_list = []
    models_dir = Path("models")
    if models_dir.exists():
        for pkl_file in sorted(models_dir.glob("*.pkl")):
            try:
                mtime = datetime.fromtimestamp(pkl_file.stat().st_mtime)
                age_days = (datetime.now() - mtime).days
                models_list.append(ModelStatus(
                    filename=pkl_file.name,
                    last_modified=mtime.isoformat(),
                    age_days=age_days,
                ))
            except OSError:
                pass

    # Data freshness
    data_freshness: dict[str, str | None] = {
        "last_predictions": None,
        "last_odds_fetch": None,
        "last_bdl_call": None,
    }

    # Check latest prediction CSV
    pred_csvs = sorted(Path().glob("predictions_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pred_csvs:
        data_freshness["last_predictions"] = datetime.fromtimestamp(pred_csvs[0].stat().st_mtime).isoformat()

    # Check odds cache
    odds_cache_dir = Path(".odds_cache")
    if odds_cache_dir.exists():
        odds_files = sorted(odds_cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if odds_files:
            data_freshness["last_odds_fetch"] = datetime.fromtimestamp(odds_files[0].stat().st_mtime).isoformat()

    # Check player impact cache for BDL freshness
    bdl_cache = Path("player_impact_cache")
    if bdl_cache.exists():
        cache_files = sorted(bdl_cache.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cache_files:
            data_freshness["last_bdl_call"] = datetime.fromtimestamp(cache_files[0].stat().st_mtime).isoformat()

    # Determine overall status
    overall_status = "healthy"
    # Check for critical: any agent with 3+ consecutive failures
    if any(a.consecutive_failures >= 3 for a in agents_status.values()):
        overall_status = "critical"
    # Check for degraded: stale models (>30 days) or any agent failures
    elif (models_list and any(m.age_days > 30 for m in models_list)) or \
         any(a.consecutive_failures >= 1 for a in agents_status.values()):
        overall_status = "degraded"

    return SystemHealthResponse(
        agents=agents_status,
        models=models_list,
        data_freshness=data_freshness,
        overall_status=overall_status,
    )


# ============== DAILY BRIEFING ENDPOINT ==============

def _query_yesterday_record(yesterday_str: str) -> dict | None:
    """Query yesterday's prediction results from calibration.db or bet_tracking.db.

    Returns a structured dict with overall, by_bet_type, by_confidence,
    clv_summary, and date fields — or None if no data is available.
    """
    import sqlite3

    record: dict | None = None

    # --- Attempt 1: calibration.db (has predictions + outcomes) ---
    cal_path = Path("data/calibration.db")
    if cal_path.exists():
        try:
            conn = sqlite3.connect(str(cal_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT p.prop_type, p.confidence, o.hit, o.clv
                FROM predictions p
                JOIN outcomes o ON p.id = o.prediction_id
                WHERE p.game_date = ?
            """, (yesterday_str,)).fetchall()
            conn.close()

            if rows:
                record = _build_record_from_calibration(rows, yesterday_str)
        except Exception:
            pass

    if record is not None:
        return record

    # --- Attempt 2: bet_tracking.db ---
    bt_path = Path("data/bet_tracking.db")
    if bt_path.exists():
        try:
            conn = sqlite3.connect(str(bt_path))
            conn.row_factory = sqlite3.Row

            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            bet_table = "tracked_bets" if "tracked_bets" in tables else "bets"

            rows = conn.execute(f"""
                SELECT status, pnl, tags, bet_type
                FROM {bet_table}
                WHERE event_date LIKE ?
                  AND status IN ('won', 'lost', 'push')
            """, (f"{yesterday_str}%",)).fetchall()
            conn.close()

            if rows:
                record = _build_record_from_tracking(rows, yesterday_str)
        except Exception:
            pass

    return record


def _build_record_from_calibration(rows, yesterday_str: str) -> dict:
    """Build a yesterday_record dict from calibration.db prediction+outcome rows."""
    import json

    wins = losses = pushes = 0
    total_clv = 0.0
    clv_count = 0
    positive_clv = 0
    by_type: dict[str, dict] = {}
    by_conf: dict[str, dict] = {"high": {"wins": 0, "losses": 0, "total": 0},
                                 "medium": {"wins": 0, "losses": 0, "total": 0},
                                 "low": {"wins": 0, "losses": 0, "total": 0}}

    for row in rows:
        hit = row["hit"]
        prop_type = row["prop_type"] or "Unknown"
        confidence = row["confidence"] or 0
        clv = row["clv"]

        if hit is None:
            pushes += 1
            continue
        if hit:
            wins += 1
        else:
            losses += 1

        # By bet type
        if prop_type not in by_type:
            by_type[prop_type] = {"wins": 0, "losses": 0, "total": 0}
        by_type[prop_type]["total"] += 1
        if hit:
            by_type[prop_type]["wins"] += 1
        else:
            by_type[prop_type]["losses"] += 1

        # By confidence tier
        if confidence >= 60:
            tier = "high"
        elif confidence >= 55:
            tier = "medium"
        else:
            tier = "low"
        by_conf[tier]["total"] += 1
        if hit:
            by_conf[tier]["wins"] += 1
        else:
            by_conf[tier]["losses"] += 1

        # CLV
        if clv is not None:
            total_clv += clv
            clv_count += 1
            if clv > 0:
                positive_clv += 1

    total = wins + losses + pushes
    hit_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0

    # Add hit_rate to by_type / by_conf entries
    for v in by_type.values():
        denom = v["wins"] + v["losses"]
        v["hit_rate"] = round(v["wins"] / denom * 100, 1) if denom > 0 else 0.0
    for v in by_conf.values():
        denom = v["wins"] + v["losses"]
        v["hit_rate"] = round(v["wins"] / denom * 100, 1) if denom > 0 else 0.0

    clv_summary = None
    if clv_count > 0:
        clv_summary = {
            "avg_clv": round(total_clv / clv_count, 2),
            "positive_clv_rate": round(positive_clv / clv_count * 100, 1),
        }

    return {
        "date": yesterday_str,
        "overall": {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total": total,
            "hit_rate": hit_rate,
            "profit": 0.0,   # calibration.db doesn't track dollar P&L
            "roi": 0.0,
        },
        "by_bet_type": by_type,
        "by_confidence": by_conf,
        "clv_summary": clv_summary,
        "source": "calibration",
    }


def _build_record_from_tracking(rows, yesterday_str: str) -> dict:
    """Build a yesterday_record dict from bet_tracking.db rows."""
    import json

    wins = losses = pushes = 0
    total_pnl = 0.0
    total_stake = 0.0
    by_type: dict[str, dict] = {}

    for row in rows:
        status = row["status"]
        pnl = row["pnl"] or 0.0
        total_pnl += pnl

        # Try to extract prop type from tags
        tags_raw = row["tags"]
        bet_type_raw = row["bet_type"] or "Unknown"
        prop_type = bet_type_raw
        if tags_raw:
            try:
                tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
                if isinstance(tags, dict):
                    prop_type = tags.get("prop_type", bet_type_raw)
                elif isinstance(tags, list) and tags:
                    prop_type = tags[0]
            except Exception:
                pass

        if status == "won":
            wins += 1
        elif status == "lost":
            losses += 1
        elif status == "push":
            pushes += 1

        if prop_type not in by_type:
            by_type[prop_type] = {"wins": 0, "losses": 0, "total": 0}
        by_type[prop_type]["total"] += 1
        if status == "won":
            by_type[prop_type]["wins"] += 1
        elif status == "lost":
            by_type[prop_type]["losses"] += 1

    total = wins + losses + pushes
    hit_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0

    for v in by_type.values():
        denom = v["wins"] + v["losses"]
        v["hit_rate"] = round(v["wins"] / denom * 100, 1) if denom > 0 else 0.0

    return {
        "date": yesterday_str,
        "overall": {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total": total,
            "hit_rate": hit_rate,
            "profit": round(total_pnl, 2),
            "roi": 0.0,  # can't compute without stake data from this query
        },
        "by_bet_type": by_type,
        "by_confidence": {},  # bet_tracking.db doesn't store confidence tiers
        "clv_summary": None,
        "source": "bet_tracking",
    }


def _format_yesterday_text(record: dict) -> str:
    """Format yesterday_record dict into human-readable briefing text."""
    if not record:
        return "YESTERDAY'S RECORD\n  No games yesterday."

    o = record["overall"]
    date_str = record["date"]
    lines = [f"YESTERDAY'S RECORD ({date_str})"]
    profit_str = f" | ${o['profit']:+,.0f}" if o["profit"] else ""
    roi_str = f" | ROI: {o['roi']:+.1f}%" if o["roi"] else ""
    lines.append(f"  Overall: {o['wins']}-{o['losses']} ({o['hit_rate']}%){profit_str}{roi_str}")

    # By bet type
    if record.get("by_bet_type"):
        lines.append("")
        lines.append("  By Bet Type:")
        # Sort by total descending for readability
        sorted_types = sorted(record["by_bet_type"].items(), key=lambda x: x[1]["total"], reverse=True)
        max_name_len = max(len(name) for name, _ in sorted_types) if sorted_types else 0
        for name, stats in sorted_types:
            pad = " " * (max_name_len - len(name) + 1)
            lines.append(f"    {name}:{pad}{stats['wins']}-{stats['losses']} ({stats['hit_rate']}%)")

    # By confidence
    if record.get("by_confidence"):
        non_empty = {k: v for k, v in record["by_confidence"].items() if v.get("total", 0) > 0}
        if non_empty:
            lines.append("")
            lines.append("  By Confidence:")
            tier_labels = {
                "high": "High (\u226560)",
                "medium": "Medium (55-59)",
                "low": "Low (<55)",
            }
            for tier in ("high", "medium", "low"):
                if tier in non_empty:
                    s = non_empty[tier]
                    label = tier_labels[tier]
                    lines.append(f"    {label}: {s['wins']}-{s['losses']} ({s['hit_rate']}%)")

    # CLV
    if record.get("clv_summary"):
        cs = record["clv_summary"]
        lines.append("")
        lines.append(f"  CLV: {cs['avg_clv']:+.1f} avg | {cs['positive_clv_rate']:.0f}% positive CLV rate")

    return "\n".join(lines)


def _get_today_preview() -> dict | None:
    """Count actionable plays for today from in-memory props."""
    try:
        service = get_service()
        games = service.get_todays_games()
        if not games:
            return {"actionable_plays": 0, "games_count": 0}

        play_count = 0
        games_with_data = 0
        prop_keys = {"points_pred": "Points", "rebounds_pred": "Rebounds",
                     "assists_pred": "Assists", "3pm_pred": "3PM", "pra_pred": "PRA"}

        for game in games:
            game_id = str(game.get("game_id", ""))
            status_data = service.get_props_fetch_status(game_id)
            if status_data.get("status") != "ready":
                continue
            games_with_data += 1
            all_players = status_data.get("home", []) + status_data.get("away", [])
            for player in all_players:
                for pred_key in prop_keys:
                    pred_val = player.get(pred_key)
                    conf_key = pred_key.replace("_pred", "_confidence")
                    edge_key = pred_key.replace("_pred", "_edge_pct")
                    conf = player.get(conf_key, 0) or 0
                    edge_pct = abs(player.get(edge_key, 0) or 0)
                    if pred_val is not None and conf >= 55 and edge_pct >= 4:
                        play_count += 1

        return {
            "actionable_plays": play_count,
            "games_count": len(games),
            "games_analyzed": games_with_data,
        }
    except Exception:
        return None


@app.get("/api/briefing", response_model=BriefingResponse)
def get_briefing(date: str = Query(None, description="Date in YYYY-MM-DD format")):
    """Get the daily briefing for a specific date (defaults to today)."""
    import sqlite3
    import json
    from datetime import datetime, timedelta

    if date is None:
        date = datetime.now(ET).strftime("%Y-%m-%d")

    # Calculate yesterday in ET
    requested_dt = datetime.strptime(date, "%Y-%m-%d")
    yesterday_str = (requested_dt - timedelta(days=1)).strftime("%Y-%m-%d")

    # --- 1. Try to find a briefing from agent_runs ---
    briefing_text = ""
    generated_at = None
    sections = None

    guardrails_path = Path("data/agent_guardrails.db")
    if guardrails_path.exists():
        try:
            conn = sqlite3.connect(str(guardrails_path))
            conn.row_factory = sqlite3.Row

            row = conn.execute("""
                SELECT payload, completed_at FROM agent_runs
                WHERE agent_name = 'briefing'
                AND started_at LIKE ?
                AND success = 1
                ORDER BY started_at DESC LIMIT 1
            """, (f"{date}%",)).fetchone()

            if row and row["payload"]:
                payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
                briefing_text = payload.get("briefing_text", "")
                generated_at = row["completed_at"]

                if "sections" in payload:
                    sections = BriefingSections(**payload["sections"])

            conn.close()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Briefing query failed: {e}")

    # --- 2. Query yesterday's record directly from DB (always) ---
    yesterday_record = _query_yesterday_record(yesterday_str)

    # --- 3. Get today's preview ---
    today_preview = _get_today_preview()

    # --- 4. Build briefing text if no agent briefing exists ---
    if not briefing_text:
        try:
            bankroll = get_bankroll()
            briefing_lines = [
                f"Daily Briefing - {date}",
                "",
                f"Bankroll: ${bankroll.current_bankroll:,.0f}",
                f"Today's P&L: ${bankroll.daily_pnl:+,.0f}",
                f"Season ROI: {bankroll.season_roi:+.1f}%",
                f"Win Rate: {bankroll.win_rate:.1f}%",
                f"Active Bets: {bankroll.active_bets}",
            ]
            briefing_text = "\n".join(briefing_lines)
            generated_at = datetime.now(ET).isoformat()
        except Exception:
            briefing_text = f"Daily Briefing - {date}"
            generated_at = datetime.now(ET).isoformat()

    # Append yesterday's record section to briefing text
    briefing_text += "\n\n" + _format_yesterday_text(yesterday_record)

    # Append today's preview
    if today_preview:
        plays = today_preview["actionable_plays"]
        games = today_preview["games_count"]
        briefing_text += f"\n\nTODAY'S PREVIEW\n  {plays} actionable play{'s' if plays != 1 else ''} across {games} game{'s' if games != 1 else ''}"

    return BriefingResponse(
        date=date,
        briefing_text=briefing_text,
        generated_at=generated_at,
        sections=sections,
        yesterday_record=yesterday_record,
        today_preview=today_preview,
    )


# ============== SETTINGS ENDPOINTS ==============

@app.get("/api/settings", response_model=SettingsResponse)
def get_settings():
    """Get current application settings."""
    s = _load_settings()
    return SettingsResponse(**s)


@app.put("/api/settings", response_model=SettingsResponse)
def update_settings(req: SettingsUpdateRequest):
    """Update application settings."""
    import json

    current = _load_settings()

    # Merge only provided fields
    updates = req.model_dump(exclude_none=True)
    current.update(updates)

    # Persist
    _settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_settings_path, "w") as f:
        json.dump(current, f, indent=2)

    return SettingsResponse(**current)


# ============== CLV ANALYSIS ENDPOINT ==============


@app.get("/api/clv/summary")
def get_clv_summary(days: int = None):
    """Get CLV analysis summary.

    Args:
        days: Optional — limit to last N days.

    Returns:
        CLV summary dict with avg_clv, sharp_rating, etc.
    """
    try:
        from nba_betting.edge.clv_analyzer import CLVAnalyzer
        analyzer = CLVAnalyzer()
        return analyzer.get_clv_summary(days=days)
    except Exception as e:
        return {
            "total_bets": 0, "settled_bets": 0, "avg_clv": 0.0,
            "avg_clv_7d": 0.0, "avg_clv_30d": 0.0, "median_clv": 0.0,
            "positive_clv_rate": 0.0, "clv_by_prop_type": {},
            "clv_by_direction": {}, "win_rate_positive_clv": 0.0,
            "win_rate_negative_clv": 0.0, "sharp_rating": "insufficient_data",
            "error": str(e),
        }


# ============== PAPER TRADING ENDPOINTS ==============


@app.get("/api/paper-trading/summary")
def get_paper_trading_summary(days: int = None):
    """Get paper trading performance summary.

    Args:
        days: Optional — limit to last N days.
    """
    try:
        from nba_betting.paper_trading import PaperTrader
        trader = PaperTrader()
        return trader.get_summary(days=days)
    except Exception as e:
        return {
            "total_predictions": 0, "settled_predictions": 0,
            "unsettled_predictions": 0, "overall_accuracy": 0.0,
            "recommended_bets": 0, "recommended_accuracy": 0.0,
            "total_wagered": 0.0, "total_profit": 0.0, "roi": 0.0,
            "brier_score": 0.0, "by_prop_type": {},
            "by_confidence_tier": {}, "by_edge_bucket": {},
            "error": str(e),
        }


@app.get("/api/paper-trading/daily/{date}")
def get_paper_trading_daily(date: str):
    """Get paper trading results for a specific date."""
    try:
        from nba_betting.paper_trading import PaperTrader
        return PaperTrader().get_daily_report(date)
    except Exception as e:
        return {"date": date, "predictions": [], "error": str(e)}


@app.post("/api/paper-trading/log-game/{game_id}")
def log_game_predictions(game_id: str):
    """Persist in-memory props for a game to the paper trading database.

    Call this after props are generated (status=ready) but before games start,
    so predictions can later be settled against actual outcomes.
    """
    from nba_betting.paper_trading import PaperTrader
    from datetime import date as date_cls

    service = get_service()
    status_data = service.get_props_fetch_status(game_id)

    if status_data.get("status") not in ("ready", "locked"):
        return {"error": "Props not ready for this game", "status": status_data.get("status")}

    all_players = status_data.get("home", []) + status_data.get("away", [])
    if not all_players:
        return {"error": "No player props found for this game"}

    game_date = date_cls.today().isoformat()
    trader = PaperTrader()
    logged = 0

    prop_key_map = {
        "Points": "points", "Rebounds": "rebounds", "Assists": "assists",
        "3PM": "3pm", "PRA": "pra",
    }

    for player in all_players:
        player_name = player.get("player_name", "")
        team = player.get("team", "")
        for prop_type, prop_key in prop_key_map.items():
            pred_key = f"{prop_key}_pred"
            if pred_key not in player or player.get(pred_key) is None:
                continue

            prediction_val = player.get(pred_key, 0) or 0
            line = player.get(f"{prop_key}_line", 0) or 0
            confidence = player.get(f"{prop_key}_confidence", 0) or 0
            pick = player.get(f"{prop_key}_pick", "-") or "-"
            edge_pct = player.get(f"{prop_key}_edge", 0) or 0

            if pick == "-" or line == 0:
                continue

            edge_raw = prediction_val - line
            should_bet = confidence >= 55 and abs(edge_pct) >= 4.0

            try:
                trader.log_prediction({
                    "game_date": game_date,
                    "game_id": game_id,
                    "player_name": player_name,
                    "prop_type": prop_type,
                    "line": line,
                    "direction": pick.lower(),
                    "predicted_value": prediction_val,
                    "edge": edge_raw,
                    "confidence": confidence,
                    "should_bet": should_bet,
                    "bet_size": 10.0 if should_bet else 0.0,
                    "tier": "elite" if confidence >= 70 else "standard",
                })
                logged += 1
            except Exception as e:
                print(f"Failed to log {player_name} {prop_type}: {e}")

    return {"logged": logged, "game_id": game_id, "game_date": game_date}


@app.post("/api/paper-trading/settle/{date}")
def settle_paper_trades(date: str):
    """Settle all paper trades for a given date using actual box scores.

    Fetches actual player stats from BallDontLie API, then grades
    each unsettled prediction against the real outcomes.
    """
    try:
        from nba_betting.settle_trades import settle_date
        settled = settle_date(date)
        return {"settled": settled, "date": date}
    except Exception as e:
        return {"error": str(e), "date": date, "settled": 0}


@app.post("/api/paper-trading/generate-and-settle/{game_id}")
def generate_and_settle_game(
    game_id: str,
    home_abbrev: str = Query(..., description="Home team abbreviation"),
    away_abbrev: str = Query(..., description="Away team abbreviation"),
):
    """Generate predictions for a completed game, log them, and settle.

    Bypasses the game-lock to allow retroactive settlement of completed games.
    This is for tracking model accuracy — predictions are timestamped as post-hoc.
    """
    import threading
    from datetime import date as date_cls

    service = get_service()
    game_date = date_cls.today().isoformat()

    def _run():
        try:
            # Initialize status dict so _fetch_props_background can write to it
            with service._prop_status_lock:
                service._prop_fetch_status[game_id] = {
                    'status': 'pending', 'home': [], 'away': []
                }

            # Force-generate props bypassing the game-started lock
            service._fetch_props_background(
                game_id, home_abbrev, away_abbrev,
                ["Points", "Rebounds", "Assists", "3PM", "PRA"],
            )

            # Log predictions to paper trading DB
            status_data = service.get_props_fetch_status(game_id)
            all_players = status_data.get("home", []) + status_data.get("away", [])

            from nba_betting.paper_trading import PaperTrader
            trader = PaperTrader()
            logged = 0

            prop_key_map = {
                "Points": "points", "Rebounds": "rebounds",
                "Assists": "assists", "3PM": "3pm", "PRA": "pra",
            }

            for player in all_players:
                player_name = player.get("player_name", "")
                for prop_type, prop_key in prop_key_map.items():
                    pred_key = f"{prop_key}_pred"
                    if pred_key not in player or player.get(pred_key) is None:
                        continue

                    prediction_val = player.get(pred_key, 0) or 0
                    line_val = player.get(f"{prop_key}_line", 0) or 0
                    confidence = player.get(f"{prop_key}_confidence", 0) or 0
                    pick = player.get(f"{prop_key}_pick", "-") or "-"
                    edge_pct = player.get(f"{prop_key}_edge", 0) or 0

                    if pick == "-" or line_val == 0:
                        continue

                    should_bet = confidence >= 55 and abs(edge_pct) >= 4.0
                    try:
                        trader.log_prediction({
                            "game_date": game_date,
                            "game_id": game_id,
                            "player_name": player_name,
                            "prop_type": prop_type,
                            "line": line_val,
                            "direction": pick.lower(),
                            "predicted_value": prediction_val,
                            "edge": prediction_val - line_val,
                            "confidence": confidence,
                            "should_bet": should_bet,
                            "bet_size": 10.0 if should_bet else 0.0,
                            "tier": "elite" if confidence >= 70 else "standard",
                        })
                        logged += 1
                    except Exception:
                        pass

            # Now settle using actual stats
            from nba_betting.settle_trades import settle_date
            settled = settle_date(game_date)

            print(f"Game {game_id}: logged {logged} predictions, settled {settled}")

        except Exception as e:
            print(f"Generate-and-settle failed for {game_id}: {e}")
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "message": "Generate-and-settle started",
        "game_id": game_id,
        "game_date": game_date,
        "note": "Predictions are post-hoc (game already completed)",
    }


# ============== MODEL HEALTH DASHBOARD ==============


@app.get("/api/model-health")
def get_model_health():
    """Unified model health dashboard.

    Aggregates paper trading performance, CLV analysis, model metrics,
    and bet filter configuration into a single snapshot. Each section
    is independently error-handled so partial data is always returned.

    Returns:
        Dict with keys: paper_trading, clv, models, bet_filter, last_updated.
    """
    from datetime import datetime, timezone

    health: dict = {"last_updated": datetime.now(timezone.utc).isoformat()}

    # Paper trading section
    try:
        from nba_betting.paper_trading import PaperTrader
        pt = PaperTrader()
        summary = pt.get_summary()
        health["paper_trading"] = {
            "total_predictions": summary.get("total_predictions", 0),
            "recommended_bets": summary.get("recommended_bets", 0),
            "overall_accuracy": summary.get("overall_accuracy", 0.0),
            "recommended_accuracy": summary.get("recommended_accuracy", 0.0),
            "roi": summary.get("roi", 0.0),
            "brier_score": summary.get("brier_score", 0.0),
        }
    except Exception as e:
        health["paper_trading"] = {
            "total_predictions": 0, "recommended_bets": 0,
            "overall_accuracy": 0.0, "recommended_accuracy": 0.0,
            "roi": 0.0, "brier_score": 0.0, "error": str(e),
        }

    # CLV section
    try:
        from nba_betting.edge.clv_analyzer import CLVAnalyzer
        analyzer = CLVAnalyzer()
        clv_summary = analyzer.get_clv_summary()
        health["clv"] = {
            "avg_clv_7d": clv_summary.get("avg_clv_7d", 0.0),
            "avg_clv_30d": clv_summary.get("avg_clv_30d", 0.0),
            "avg_clv_all": clv_summary.get("avg_clv_all", 0.0),
            "positive_clv_rate": clv_summary.get("positive_clv_rate", 0.0),
            "sharp_rating": clv_summary.get("sharp_rating", "unknown"),
        }
    except Exception as e:
        health["clv"] = {
            "avg_clv_7d": 0.0, "avg_clv_30d": 0.0, "avg_clv_all": 0.0,
            "positive_clv_rate": 0.0, "sharp_rating": "unknown", "error": str(e),
        }

    # Models section — read from bet_filter thresholds and known model states
    try:
        from nba_betting.bet_filter import MIN_EDGE_THRESHOLDS, DISABLED_PROPS
        prop_types = ["spread", "points", "rebounds", "assists", "pra", "threes", "moneyline"]
        models_data: dict = {}
        for prop in prop_types:
            entry: dict = {
                "status": "disabled" if prop in DISABLED_PROPS else "enabled",
                "threshold": MIN_EDGE_THRESHOLDS.get(prop, 0),
            }
            if prop in DISABLED_PROPS:
                entry["reason"] = "no demonstrated edge"
            models_data[prop] = entry
        health["models"] = models_data
    except Exception as e:
        health["models"] = {"error": str(e)}

    # Bet filter configuration section
    try:
        from nba_betting.bet_filter import (
            MIN_EDGE_THRESHOLDS as thresholds,
            DISABLED_PROPS as disabled,
            MIN_CONFIDENCE,
        )
        kelly_fraction = 0.25
        try:
            from nba_betting.prediction_pipeline import KELLY_FRACTION
            kelly_fraction = KELLY_FRACTION
        except ImportError:
            pass
        health["bet_filter"] = {
            "min_confidence": MIN_CONFIDENCE,
            "disabled_props": list(disabled),
            "kelly_fraction": kelly_fraction,
            "thresholds": dict(thresholds),
        }
    except Exception as e:
        health["bet_filter"] = {"error": str(e)}

    return health


# ============== AGENT DIAGNOSTICS ==============


@app.get("/api/diagnostics/agents")
def get_agent_diagnostics():
    """Deep diagnostic of agent infrastructure on PostgreSQL.

    Checks which agent-related tables exist, row counts, recent runs,
    and the agent_registry status. Designed to debug why agents may
    not be executing on Railway.

    Returns:
        Dict with tables, agent_registry rows, agent_runs rows,
        and Python/environment info.
    """
    from datetime import datetime, timezone
    import sys as _sys

    diag: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": _sys.version,
        "database_url_set": bool(os.environ.get("DATABASE_URL")),
        "redis_url_set": bool(os.environ.get("REDIS_URL")),
        "gemini_api_key_set": bool(os.environ.get("GEMINI_API_KEY")),
    }

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        diag["error"] = "DATABASE_URL not set — cannot query PostgreSQL"
        return diag

    try:
        import psycopg2
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()

        # List all agent-related tables
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN (
                  'agent_runs', 'agent_registry', 'agent_token_budgets',
                  'agent_messages', 'paper_trades', 'bet_tracking'
              )
            ORDER BY table_name
        """)
        existing_tables = [row[0] for row in cur.fetchall()]
        diag["existing_tables"] = existing_tables

        # agent_registry — shows if scheduler ever started and registered agents
        if "agent_registry" in existing_tables:
            cur.execute("""
                SELECT agent_name, agent_class, status, enabled,
                       last_run_at, schedule
                FROM agent_registry
                ORDER BY agent_name
            """)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            diag["agent_registry"] = [dict(zip(cols, r, strict=False)) for r in rows]
            diag["agent_registry_count"] = len(rows)
        else:
            diag["agent_registry"] = "TABLE DOES NOT EXIST"

        # agent_runs — shows actual executions
        if "agent_runs" in existing_tables:
            cur.execute("SELECT COUNT(*) FROM agent_runs")
            total_runs = cur.fetchone()[0]
            diag["agent_runs_total"] = total_runs

            # Last 10 runs
            cur.execute("""
                SELECT agent_name, run_id, started_at, completed_at,
                       status, success, tokens_used, execution_seconds,
                       errors
                FROM agent_runs
                ORDER BY started_at DESC
                LIMIT 10
            """)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            diag["agent_runs_recent"] = [dict(zip(cols, r, strict=False)) for r in rows]
        else:
            diag["agent_runs"] = "TABLE DOES NOT EXIST"

        # agent_token_budgets
        if "agent_token_budgets" in existing_tables:
            cur.execute("SELECT * FROM agent_token_budgets ORDER BY agent_name")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            diag["agent_token_budgets"] = [dict(zip(cols, r, strict=False)) for r in rows]
        else:
            diag["agent_token_budgets"] = "TABLE DOES NOT EXIST"

        # Check if apscheduler is installed (needed by agent scheduler)
        try:
            import apscheduler
            diag["apscheduler_version"] = apscheduler.__version__
        except ImportError:
            diag["apscheduler_version"] = "NOT INSTALLED"
        except AttributeError:
            diag["apscheduler_version"] = "installed (no __version__)"

        cur.close()
        conn.close()

    except ImportError:
        diag["error"] = "psycopg2 not installed"
    except Exception as e:
        diag["error"] = f"PostgreSQL query failed: {e}"

    return diag


# ============== BACKTEST TRIGGER ==============


@app.post("/api/backtest/run-profitability")
def run_profitability_backtest():
    """Trigger the profitability backtest in a background thread.

    Returns immediately with status; poll GET /api/backtest/profitability-status
    for results.
    """
    import threading

    if getattr(app.state, "_backtest_running", False):
        raise HTTPException(409, "Backtest already running")

    def _run():
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "nba_models" / "training"))
        try:
            app.state._backtest_running = True
            from nba_models.backtesting.profitability_backtest import run_backtest
            import argparse
            args = argparse.Namespace(bankroll=1000.0, season="2023-24")
            results = run_backtest(args)
            app.state._backtest_results = results or {"error": "Backtest returned None"}
        except Exception as e:
            import traceback
            app.state._backtest_results = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1000:],
            }
        finally:
            app.state._backtest_running = False

    app.state._backtest_results = None
    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "message": "Backtest running in background. Poll GET /api/backtest/profitability-status for results."}


@app.get("/api/backtest/profitability-status")
def get_profitability_backtest_status():
    """Check status of the profitability backtest."""
    running = getattr(app.state, "_backtest_running", False)
    results = getattr(app.state, "_backtest_results", None)

    if running:
        # Try to get progress from the backtest module
        try:
            from nba_models.backtesting.profitability_backtest import _progress
            return {"status": "running", "progress": dict(_progress)}
        except Exception:
            return {"status": "running"}
    if results is not None:
        return {"status": "complete", "results": results}
    return {"status": "idle", "message": "No backtest has been triggered yet. POST /api/backtest/run-profitability to start."}


# ============== OOS BACKTEST ==============


@app.post("/api/backtest/run-oos")
def run_oos_backtest(
    train_seasons: list[str] | None = None,
    test_season: str = "2023-24",
    skip_retrain: bool = False,
):
    """Trigger an out-of-sample backtest in a background thread.

    Trains models on train_seasons (default: 2020-2022), then tests on
    test_season using those holdout models.

    Returns immediately; poll GET /api/backtest/oos-status for results.
    """
    import threading

    if getattr(app.state, "_oos_running", False):
        raise HTTPException(409, "OOS backtest already running")

    if train_seasons is None:
        train_seasons = ["2020-21", "2021-22", "2022-23"]

    def _run():
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "nba_models" / "training"))
        try:
            app.state._oos_running = True
            from nba_models.backtesting.oos_backtest import run_oos_backtest as _run_oos
            results = _run_oos(
                train_seasons=train_seasons,
                test_season=test_season,
                skip_retrain=skip_retrain,
            )
            app.state._oos_results = results or {"error": "OOS backtest returned None"}
        except Exception as e:
            import traceback
            app.state._oos_results = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1000:],
            }
        finally:
            app.state._oos_running = False

    app.state._oos_results = None
    threading.Thread(target=_run, daemon=True).start()
    return {
        "status": "started",
        "train_seasons": train_seasons,
        "test_season": test_season,
        "skip_retrain": skip_retrain,
        "message": "OOS backtest running in background. Poll GET /api/backtest/oos-status for results.",
    }


@app.get("/api/backtest/oos-status")
def get_oos_backtest_status():
    """Check status of the out-of-sample backtest."""
    running = getattr(app.state, "_oos_running", False)
    results = getattr(app.state, "_oos_results", None)

    if running:
        try:
            from nba_models.backtesting.oos_backtest import _oos_progress
            return {"status": "running", "progress": dict(_oos_progress)}
        except Exception:
            return {"status": "running"}
    if results is not None:
        return {"status": "complete", "results": results}
    return {
        "status": "idle",
        "message": "No OOS backtest triggered. POST /api/backtest/run-oos to start.",
    }


# ============== REAL-LINES BACKTEST ==============


@app.post("/api/backtest/run-real-lines")
def run_real_lines_backtest():
    """Trigger the real-lines profitability backtest in a background thread.

    Uses actual sportsbook lines from The Odds API and real player outcomes
    from BallDontLie for the 2024-25 season.

    Returns immediately; poll GET /api/backtest/real-lines-status for results.
    """
    import threading

    if getattr(app.state, "_real_lines_running", False):
        raise HTTPException(409, "Real-lines backtest already running")

    def _run():
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "nba_models" / "training"))
        try:
            app.state._real_lines_running = True
            from nba_models.backtesting.real_lines_backtest import run_backtest
            import argparse
            args = argparse.Namespace(bankroll=1000.0, model_dir=None)
            results = run_backtest(args)
            app.state._real_lines_results = results or {"error": "Real-lines backtest returned None"}
        except Exception as e:
            import traceback
            app.state._real_lines_results = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:],
            }
        finally:
            app.state._real_lines_running = False

    app.state._real_lines_results = None
    threading.Thread(target=_run, daemon=True).start()
    return {
        "status": "started",
        "message": "Real-lines backtest running in background. Poll GET /api/backtest/real-lines-status for results.",
    }


@app.get("/api/backtest/real-lines-status")
def get_real_lines_backtest_status():
    """Check status of the real-lines backtest."""
    running = getattr(app.state, "_real_lines_running", False)
    results = getattr(app.state, "_real_lines_results", None)

    if running:
        try:
            from nba_models.backtesting.real_lines_backtest import _progress
            return {"status": "running", "progress": dict(_progress)}
        except Exception:
            return {"status": "running"}
    if results is not None:
        return {"status": "complete", "results": results}
    return {
        "status": "idle",
        "message": "No real-lines backtest triggered. POST /api/backtest/run-real-lines to start.",
    }


# ============== SERVE FRONTEND STATIC FILES ==============

_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the React SPA for any non-API route."""
        file_path = _frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_frontend_dist / "index.html"), media_type="text/html")


# ============== RUN SERVER ==============

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 40)
    print("NBA Props API Server")
    print("=" * 40)
    print("Starting server at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    print("=" * 40 + "\n")

    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
