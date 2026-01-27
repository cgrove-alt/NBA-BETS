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

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

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
    """Initialize data service on startup."""
    print("Initializing NBA Props API...")
    get_service()
    print("Data service ready.")
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


# ============== HEALTH CHECK ==============

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    from datetime import datetime
    service = get_service()
    return HealthResponse(
        status="healthy",
        service="nba-props-api",
        timestamp=datetime.now().isoformat(),
        models_loaded=service.models_loaded if hasattr(service, 'models_loaded') else True,
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
                ))

    # Sort based on user preference
    if sort_by == "confidence":
        best_bets.sort(key=lambda x: x.confidence, reverse=True)
    elif sort_by == "edge":
        best_bets.sort(key=lambda x: abs(x.edge_pct), reverse=True)
    else:  # Default "quality" - composite score
        best_bets.sort(key=lambda x: (x.confidence - 50) * abs(x.edge_pct), reverse=True)

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

    # Fetch injuries for the date
    try:
        injuries_data = fetch_current_injuries(target_date)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching injuries: {str(e)}"
        )

    # Convert to injury report objects
    # fetch_current_injuries returns InjuryReport dataclass objects, not dicts
    injuries = []
    for injury_obj in injuries_data:
        injuries.append(InjuryReport(
            player_id=injury_obj.player_id or 0,
            player_name=injury_obj.player_name or 'Unknown',
            team_id=injury_obj.team_id or 0,
            team_abbrev=injury_obj.team_abbrev or '',
            status=str(injury_obj.status.value) if injury_obj.status else 'UNKNOWN',
            injury_type=injury_obj.injury_type or None,
            detected_at=injury_obj.last_updated.isoformat() if injury_obj.last_updated else datetime.now().isoformat(),
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

    # Parse backtest results
    try:
        # Extract overall metrics
        overall = data.get('overall_performance', {})
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

        # Extract by prop type
        by_prop = []
        prop_metrics = data.get('by_prop_type', {})
        for prop_type, metrics in prop_metrics.items():
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
            games_analyzed=data.get('games_analyzed', 0),
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
