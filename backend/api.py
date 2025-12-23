"""
FastAPI Backend for NBA Props Dashboard

Wraps existing DataService (ML/prediction logic) with REST endpoints.
DO NOT modify the underlying data_service.py - this is a read-only wrapper.

Usage:
    uvicorn backend.api:app --reload --port 8000
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

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
)

# Singleton data service instance
_data_service: Optional[DataService] = None

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
    version="1.0.0",
    lifespan=lifespan,
)

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
def get_games(force_refresh: bool = Query(False, description="Force refresh from API")):
    """Get today's NBA games."""
    service = get_service()
    games_data = service.get_todays_games(force_refresh=force_refresh)

    games = []
    for g in games_data:
        home = g.get("home_team", {})
        visitor = g.get("visitor_team", {})

        games.append(Game(
            game_id=str(g.get("game_id", "")),
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

    return GamesResponse(games=games, count=len(games))


# ============== PROPS ENDPOINTS ==============

def _build_prop_prediction(player_data: dict, prop_key: str) -> Optional[PropPrediction]:
    """Build PropPrediction from flattened player data keys.

    DataService returns flattened keys like:
        points_pred, points_line, points_pick, points_edge, points_confidence

    This function extracts them and builds a nested PropPrediction object.
    """
    pred_key = f"{prop_key}_pred"
    if pred_key not in player_data or player_data.get(pred_key) is None:
        return None

    line = player_data.get(f"{prop_key}_line", 0) or 0
    prediction = player_data.get(pred_key, 0) or 0
    edge = player_data.get(f"{prop_key}_edge", 0) or 0
    confidence = player_data.get(f"{prop_key}_confidence", 50) or 50
    pick = player_data.get(f"{prop_key}_pick", "-") or "-"

    # Calculate edge_pct
    edge_pct = (edge / line * 100) if line and line > 0 else 0

    return PropPrediction(
        prediction=prediction,
        confidence=confidence,
        edge=edge,
        edge_pct=edge_pct,
        pick=pick,
        line=line,
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
    home_props_raw = status_data.get("home", [])
    away_props_raw = status_data.get("away", [])

    # Get team abbreviations from cache (set when props fetch was started)
    cached_teams = _game_teams_cache.get(game_id, {})
    home_abbrev = cached_teams.get("home")
    away_abbrev = cached_teams.get("away")

    # Determine best bets (confidence >= 80, edge >= 2.5)
    def is_best_bet(player: dict) -> bool:
        for prop_key in ["points", "rebounds", "assists", "3pm", "pra"]:
            conf = player.get(f"{prop_key}_confidence", 0) or 0
            edge = abs(player.get(f"{prop_key}_edge", 0) or 0)
            pick = player.get(f"{prop_key}_pick", "-")
            if pick != "-" and conf >= 80 and edge >= 2.5:
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
    request: Optional[StartPropsRequest] = None,
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

    # Convert moneyline prediction
    moneyline = None
    ml_data = analysis.get("moneyline_prediction")
    if ml_data and isinstance(ml_data, dict):
        moneyline = MoneylinePrediction(
            home_win_probability=ml_data.get("home_win_probability", 0.5),
            away_win_probability=ml_data.get("away_win_probability", 0.5),
            predicted_winner=ml_data.get("predicted_winner", "home"),
            confidence=ml_data.get("confidence", 0.5),
            calibrated=ml_data.get("calibrated", False),
        )

    # Convert spread prediction
    spread = None
    sp_data = analysis.get("spread_prediction")
    if sp_data and isinstance(sp_data, dict):
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
    min_confidence: float = Query(80.0, ge=0, le=100, description="Minimum confidence threshold"),
    min_edge: float = Query(2.5, ge=0, description="Minimum edge threshold"),
    prop_types: Optional[str] = Query(None, description="Comma-separated prop types to filter"),
    pick_type: Optional[str] = Query(None, description="Filter by OVER or UNDER"),
):
    """Get best bets across all games based on confidence and edge thresholds."""
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
                edge = player.get(f"{prop_key}_edge", 0) or 0
                confidence = player.get(f"{prop_key}_confidence", 0) or 0
                pick = player.get(f"{prop_key}_pick", "-") or "-"

                # Calculate edge_pct
                edge_pct = (edge / line * 100) if line and line > 0 else 0

                # Apply filters
                if confidence < min_confidence:
                    continue
                if abs(edge) < min_edge:
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

    # Sort by confidence descending, then by edge descending
    best_bets.sort(key=lambda x: (x.confidence, abs(x.edge)), reverse=True)

    return BestBetsResponse(
        best_bets=best_bets,
        count=len(best_bets),
        filters={
            "min_confidence": min_confidence,
            "min_edge": min_edge,
            "prop_types": prop_type_filter,
            "pick_type": pick_type,
        },
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
