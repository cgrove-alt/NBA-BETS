"""
Pydantic schemas for NBA Props API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============== TEAM SCHEMAS ==============

class Team(BaseModel):
    id: int
    abbreviation: str
    city: Optional[str] = None
    name: Optional[str] = None


# ============== GAME SCHEMAS ==============

class Game(BaseModel):
    game_id: str
    home_team: Team
    visitor_team: Team
    game_time: Optional[str] = None
    status: Optional[str] = None


class GamesResponse(BaseModel):
    games: List[Game]
    count: int


# ============== PROP SCHEMAS ==============

class PropPrediction(BaseModel):
    prediction: float
    confidence: float = Field(ge=0, le=100)
    edge: float
    edge_pct: float
    pick: str  # "OVER", "UNDER", or "-"
    line: Optional[float] = None
    implied_probability: Optional[float] = None


class PlayerProp(BaseModel):
    player_name: str
    player_id: int
    team: Optional[str] = None
    position: Optional[str] = None
    avg_minutes: Optional[float] = None
    Points: Optional[PropPrediction] = None
    Rebounds: Optional[PropPrediction] = None
    Assists: Optional[PropPrediction] = None
    three_pm: Optional[PropPrediction] = Field(None, alias="3PM")
    PRA: Optional[PropPrediction] = None
    is_best_bet: bool = False

    class Config:
        populate_by_name = True


class PropsResponse(BaseModel):
    game_id: str
    status: str  # "pending", "ready", "error", "not_started", "locked"
    error: Optional[str] = None  # Error message for error/locked status
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    home_props: List[PlayerProp] = []
    away_props: List[PlayerProp] = []
    all_props: List[PlayerProp] = []
    count: int = 0


class StartPropsRequest(BaseModel):
    prop_types: Optional[List[str]] = None  # Default: ["Points", "Rebounds", "Assists", "3PM", "PRA"]


# ============== ANALYSIS SCHEMAS ==============

class MoneylinePrediction(BaseModel):
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str  # "home" or "away"
    confidence: float
    calibrated: bool = False


class SpreadPrediction(BaseModel):
    predicted_spread: float
    confidence: float


class AnalysisStatus(BaseModel):
    game_id: str
    status: str  # "not_started", "pending", "ready", "error"
    moneyline: Optional[MoneylinePrediction] = None
    spread: Optional[SpreadPrediction] = None
    error: Optional[str] = None


class GameAnalysis(BaseModel):
    game_id: str
    home_team: str
    home_abbrev: str
    away_team: str
    away_abbrev: str
    game_time: Optional[str] = None
    status: Optional[str] = None
    moneyline_prediction: Optional[MoneylinePrediction] = None
    spread_prediction: Optional[SpreadPrediction] = None
    market_odds: Optional[Dict[str, Any]] = None
    recommendations: List[Dict[str, Any]] = []


# ============== ODDS SCHEMAS ==============

class MoneylineOdds(BaseModel):
    home: int
    away: int


class SpreadOdds(BaseModel):
    home_line: float
    home_odds: int
    away_line: float
    away_odds: int


class TotalOdds(BaseModel):
    line: float
    over_odds: int
    under_odds: int


class GameOdds(BaseModel):
    game_id: Optional[str] = None
    moneyline: Optional[MoneylineOdds] = None
    spread: Optional[SpreadOdds] = None
    total: Optional[TotalOdds] = None
    sportsbook: Optional[str] = None
    last_updated: Optional[str] = None


class OddsResponse(BaseModel):
    odds: Dict[str, GameOdds]


# ============== BEST BETS SCHEMA ==============

class BestBet(BaseModel):
    player_name: str
    player_id: int
    team: str
    game_id: str
    prop_type: str
    prediction: float
    line: float
    edge: float
    edge_pct: float
    pick: str
    confidence: float


class BestBetsResponse(BaseModel):
    best_bets: List[BestBet]
    count: int
    filters: Dict[str, Any]


# ============== GAME RESULTS SCHEMAS ==============

class PlayerResult(BaseModel):
    player_id: int
    player_name: str
    team: str
    prop_type: str
    predicted: float
    actual: float
    line: Optional[float] = None
    pick: Optional[str] = None
    hit: Optional[bool] = None  # True = win, False = loss, None = no pick
    difference: float  # actual - predicted


class FinalScore(BaseModel):
    home_team: str
    home_score: int
    away_team: str
    away_score: int


class MoneylineResult(BaseModel):
    predicted_winner: str
    actual_winner: str
    correct: bool
    home_win_probability: Optional[float] = None
    away_win_probability: Optional[float] = None


class ResultsSummary(BaseModel):
    total_predictions: int
    total_picks: int  # Predictions where a pick was made
    total_hits: int
    hit_rate: float


class GameResults(BaseModel):
    game_id: str
    status: str  # "completed", "not_completed", "error"
    message: Optional[str] = None
    final_score: Optional[FinalScore] = None
    moneyline_result: Optional[MoneylineResult] = None
    player_results: List[PlayerResult] = []
    summary: Optional[ResultsSummary] = None


# ============== HEALTH CHECK ==============

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    models_loaded: bool = False
