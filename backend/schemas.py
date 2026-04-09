"""
Pydantic schemas for NBA Props API request/response validation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


# ============== TEAM SCHEMAS ==============

class Team(BaseModel):
    id: int
    abbreviation: str
    city: str | None = None
    name: str | None = None


# ============== GAME SCHEMAS ==============

class Game(BaseModel):
    game_id: str
    home_team: Team
    visitor_team: Team
    game_time: str | None = None
    status: str | None = None


class GamesResponse(BaseModel):
    games: list[Game]
    count: int


# ============== PROP SCHEMAS ==============

class PropPrediction(BaseModel):
    prediction: float
    confidence: float = Field(ge=0, le=100)
    edge: float
    edge_pct: float
    pick: str  # "OVER", "UNDER", or "-"
    line: float | None = None
    # Phase 4.1: Real Odds Integration
    implied_probability: float | None = None     # Vig-free implied prob for the recommended side
    ev_per_dollar: float | None = None           # Expected value per dollar staked (positive = value)
    # Phase 4.2: Line Shopping
    best_book: str | None = None                 # Sportsbook with best available odds
    best_odds: int | None = None                 # Best American odds across all books
    # Phase 4.3: Line Movement
    line_movement_signal: str | None = None      # "CONFIRMS_MODEL", "WARNS_MODEL", or "NEUTRAL"


class PlayerProp(BaseModel):
    player_name: str
    player_id: int
    team: str | None = None
    position: str | None = None
    avg_minutes: float | None = None
    Points: PropPrediction | None = None
    Rebounds: PropPrediction | None = None
    Assists: PropPrediction | None = None
    three_pm: PropPrediction | None = Field(None, alias="3PM")
    PRA: PropPrediction | None = None
    is_best_bet: bool = False

    class Config:
        populate_by_name = True


class PropsResponse(BaseModel):
    game_id: str
    status: str  # "pending", "ready", "error", "not_started", "locked"
    error: str | None = None  # Error message for error/locked status
    home_team: str | None = None
    away_team: str | None = None
    home_props: list[PlayerProp] = []
    away_props: list[PlayerProp] = []
    all_props: list[PlayerProp] = []
    count: int = 0


class StartPropsRequest(BaseModel):
    prop_types: list[str] | None = None  # Default: ["Points", "Rebounds", "Assists", "3PM", "PRA"]


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
    moneyline: MoneylinePrediction | None = None
    spread: SpreadPrediction | None = None
    error: str | None = None


class GameAnalysis(BaseModel):
    game_id: str
    home_team: str
    home_abbrev: str
    away_team: str
    away_abbrev: str
    game_time: str | None = None
    status: str | None = None
    moneyline_prediction: MoneylinePrediction | None = None
    spread_prediction: SpreadPrediction | None = None
    market_odds: dict[str, Any] | None = None
    recommendations: list[dict[str, Any]] = []


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
    game_id: str | None = None
    moneyline: MoneylineOdds | None = None
    spread: SpreadOdds | None = None
    total: TotalOdds | None = None
    sportsbook: str | None = None
    last_updated: str | None = None


class OddsResponse(BaseModel):
    odds: dict[str, GameOdds]


# ============== BEST BETS SCHEMA ==============

class BookOddsEntry(BaseModel):
    """Per-sportsbook odds entry for line shopping display."""
    book: str
    line: float | None = None
    over_odds: int | None = None
    under_odds: int | None = None
    implied_prob_over: float | None = None


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
    # Ranking and explanation fields
    rank: int = 0
    season_avg: float | None = None
    recent_avg: float | None = None
    explanation: str = ""
    signals: list[str] = []
    used_real_line: bool = False
    used_ml_model: bool = False
    line_vendor: str = "unknown"
    line_source: str = "unknown"
    bettable: bool = True
    # Phase 4.1: Real Odds Integration
    implied_probability: float | None = None     # Vig-free implied prob for the recommended side
    ev_per_dollar: float | None = None           # EV per $1 staked
    ev_dollars: float | None = None             # EV per $100 staked (= ev_per_dollar * 100)
    # Phase 4.2: Line Shopping
    best_book: str | None = None                 # Book with best available odds
    best_odds: int | None = None                 # Best American odds for the recommended side
    book_comparison: list[BookOddsEntry] = []    # Per-book breakdown for line shopping display
    # Phase 4.3: Line Movement
    line_movement_signal: str | None = None      # "CONFIRMS_MODEL", "WARNS_MODEL", or "NEUTRAL"


class BestBetsResponse(BaseModel):
    best_bets: list[BestBet]
    count: int
    filters: dict[str, Any]
    data_source: str = "realtime"  # "realtime", "precomputed", or "mixed"
    warnings: list[str] = []       # Non-fatal issues surfaced to frontend
    locked_games: list[str] = []   # Game IDs that were locked (games started)


# ============== GAME RESULTS SCHEMAS ==============

class PlayerResult(BaseModel):
    player_id: int
    player_name: str
    team: str
    prop_type: str
    predicted: float
    actual: float
    line: float | None = None
    pick: str | None = None
    hit: bool | None = None  # True = win, False = loss, None = no pick
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
    home_win_probability: float | None = None
    away_win_probability: float | None = None


class ResultsSummary(BaseModel):
    total_predictions: int
    total_picks: int  # Predictions where a pick was made
    total_hits: int
    hit_rate: float


class GameResults(BaseModel):
    game_id: str
    status: str  # "completed", "not_completed", "error"
    message: str | None = None
    final_score: FinalScore | None = None
    moneyline_result: MoneylineResult | None = None
    player_results: list[PlayerResult] = []
    summary: ResultsSummary | None = None


# ============== HEALTH CHECK ==============

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    models_loaded: bool = False
    database_connected: bool = False
    redis_connected: bool = False
    checks: dict[str, Any] = {}
    environment: dict[str, Any] = {}
    warnings: list[str] = []
    issues: list[str] = []


# ============== DAILY PREDICTIONS SCHEMAS ==============

class DailyPrediction(BaseModel):
    player_name: str
    team: str
    prop_type: str
    prediction: float
    pred_low: float | None = None
    pred_median: float | None = None
    pred_high: float | None = None
    line: float | None = None
    confidence_score: float | None = None
    edge_quality_tier: str | None = None
    suggested_bet_size: float | None = None
    bet_recommendation: str | None = None
    uncertainty_flag: str | None = None
    pick: str | None = None
    edge: float | None = None
    line_source: str | None = None
    line_vendor: str | None = None
    # Phase 4.1: Real Odds Integration
    implied_probability: float | None = None
    ev_per_dollar: float | None = None
    # Phase 4.2: Line Shopping
    best_odds: int | None = None
    best_book: str | None = None
    # Phase 4.3: Line Movement
    line_movement_signal: str | None = None


class DailyPredictionsResponse(BaseModel):
    date: str
    predictions: list[DailyPrediction]
    count: int
    metadata: dict[str, Any] | None = None


# ============== INJURY REPORT SCHEMAS ==============

class InjuryReport(BaseModel):
    player_id: int
    player_name: str
    team_id: int
    team_abbrev: str
    status: str  # OUT, DOUBTFUL, QUESTIONABLE, GTD
    injury_type: str | None = None
    detected_at: str


class InjuryReportResponse(BaseModel):
    date: str
    injuries: list[InjuryReport]
    count: int
    last_updated: str


# ============== LINE MOVEMENT SCHEMAS ==============

class OddsSnapshot(BaseModel):
    timestamp: str
    book_name: str
    market: str  # moneyline, spread, total, props
    home_odds: int | None = None
    away_odds: int | None = None
    home_line: float | None = None
    away_line: float | None = None
    total: float | None = None


class LineMovement(BaseModel):
    opening_line: float | None = None
    closing_line: float | None = None
    movement: float | None = None  # closing - opening
    rlm_detected: bool = False
    steam_move_detected: bool = False


class LineMovementResponse(BaseModel):
    game_id: str
    market: str
    odds_history: list[OddsSnapshot]
    movement_analysis: LineMovement | None = None
    count: int


# ============== PROP LINE MOVEMENT SCHEMAS (Phase 4.3) ==============

class PropOddsSnapshotItem(BaseModel):
    """Single point-in-time prop odds entry for a specific sportsbook."""
    timestamp: str
    book_name: str
    line: float
    over_odds: int | None = None
    under_odds: int | None = None
    implied_prob_over: float | None = None
    is_opening: bool = False


class PropLineMovement(BaseModel):
    """Line movement summary from opening to current for a player prop."""
    opening_line: float | None = None
    current_line: float | None = None
    movement: float | None = None           # current_line - opening_line
    movement_signal: str | None = None     # "CONFIRMS_MODEL", "WARNS_MODEL", "NEUTRAL"
    opening_timestamp: str | None = None
    current_timestamp: str | None = None
    num_snapshots: int = 0


class PropLineMovementResponse(BaseModel):
    """Response for the /api/prop-line-movement endpoint."""
    player_name: str
    prop_type: str
    game_date: str
    snapshots: list[PropOddsSnapshotItem] = []
    movement: PropLineMovement | None = None
    book_comparison: list[BookOddsEntry] = []   # Most recent per-book comparison
    count: int = 0


# ============== BACKTEST RESULTS SCHEMAS ==============

class BacktestMetrics(BaseModel):
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    bias: float | None = None


class BacktestBettingMetrics(BaseModel):
    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    roi: float
    total_wagered: float
    total_profit: float
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None


class BacktestByProp(BaseModel):
    prop_type: str
    metrics: BacktestMetrics
    count: int


class BacktestResults(BaseModel):
    backtest_id: str
    date_range: str
    games_analyzed: int
    total_predictions: int
    overall_metrics: BacktestMetrics
    betting_metrics: BacktestBettingMetrics | None = None
    by_prop_type: list[BacktestByProp] | None = None
    elite_strong_metrics: BacktestMetrics | None = None
    confidence_correlation: float | None = None
    phase: str | None = None
    timestamp: str


class LatestBacktestResponse(BaseModel):
    latest_backtest: BacktestResults | None = None
    available_backtests: list[str]
    count: int


# ============== BANKROLL SCHEMAS ==============

class BankrollResponse(BaseModel):
    current_bankroll: float
    initial_bankroll: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    season_pnl: float
    season_roi: float
    total_exposure_today: float
    total_bets: int
    win_rate: float
    active_bets: int


# ============== PERFORMANCE SCHEMAS ==============

class DailyRecord(BaseModel):
    date: str
    wins: int
    losses: int
    pushes: int
    roi: float
    clv_avg: float | None = None
    profit: float

class PropTypeStats(BaseModel):
    total: int
    wins: int
    losses: int
    hit_rate: float

class ConfidenceTierStats(BaseModel):
    total: int
    wins: int
    hit_rate: float

class CalibrationSummaryResponse(BaseModel):
    total_predictions: int
    matched_predictions: int
    overall_hit_rate: float | None = None
    overall_clv: float | None = None
    ece: float | None = None

class PerformanceResponse(BaseModel):
    daily_records: list[DailyRecord]
    by_prop_type: dict[str, PropTypeStats]
    by_confidence_tier: dict[str, ConfidenceTierStats]
    calibration_summary: CalibrationSummaryResponse | None = None
    total_bets: int
    total_wins: int
    total_losses: int
    overall_hit_rate: float
    overall_roi: float


# ============== SYSTEM HEALTH SCHEMAS ==============

class AgentStatus(BaseModel):
    last_run: str | None = None
    last_status: str | None = None
    consecutive_failures: int = 0
    tokens_used_today: int = 0

class ModelStatus(BaseModel):
    filename: str
    last_modified: str
    age_days: int

class SystemHealthResponse(BaseModel):
    agents: dict[str, AgentStatus]
    models: list[ModelStatus]
    data_freshness: dict[str, str | None]
    overall_status: str  # "healthy", "degraded", "critical"


# ============== BRIEFING SCHEMAS ==============

class BriefingSections(BaseModel):
    yesterday_results: dict | str | None = None
    today_plays: list | str | None = None
    bankroll: dict | str | None = None
    alerts: list | str | None = None
    market_intel: list | str | None = None

class BriefingResponse(BaseModel):
    date: str
    briefing_text: str
    generated_at: str | None = None
    sections: BriefingSections | None = None
    yesterday_record: dict | None = None
    today_preview: dict | None = None


# ============== SETTINGS SCHEMAS ==============

class SettingsResponse(BaseModel):
    bankroll: float
    min_edge: float
    min_confidence: float
    kelly_fraction: float
    max_exposure: float
    default_bet_size: float
    bet_size_type: str
    max_bets_per_day: int

class SettingsUpdateRequest(BaseModel):
    bankroll: float | None = None
    min_edge: float | None = None
    min_confidence: float | None = None
    kelly_fraction: float | None = None
    max_exposure: float | None = None
    default_bet_size: float | None = None
    bet_size_type: str | None = None
    max_bets_per_day: int | None = None
