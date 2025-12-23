// TypeScript interfaces for NBA Props API

export interface Team {
  id: number;
  abbreviation: string;
  city?: string;
  name?: string;
}

export interface Game {
  game_id: string;
  home_team: Team;
  visitor_team: Team;
  game_time?: string;
  status?: string;
}

export interface GamesResponse {
  games: Game[];
  count: number;
}

export interface PropPrediction {
  prediction: number;
  confidence: number;
  edge: number;
  edge_pct: number;
  pick: 'OVER' | 'UNDER' | '-';
  line?: number;
  implied_probability?: number;
}

export interface PlayerProp {
  player_name: string;
  player_id: number;
  team?: string;
  position?: string;
  avg_minutes?: number;
  Points?: PropPrediction;
  Rebounds?: PropPrediction;
  Assists?: PropPrediction;
  '3PM'?: PropPrediction;
  PRA?: PropPrediction;
  is_best_bet: boolean;
}

export interface PropsResponse {
  game_id: string;
  status: 'pending' | 'ready' | 'error' | 'not_started';
  home_team?: string;
  away_team?: string;
  home_props: PlayerProp[];
  away_props: PlayerProp[];
  all_props: PlayerProp[];
  count: number;
}

export interface MoneylinePrediction {
  home_win_probability: number;
  away_win_probability: number;
  predicted_winner: 'home' | 'away';
  confidence: number;
  calibrated: boolean;
}

export interface SpreadPrediction {
  predicted_spread: number;
  confidence: number;
}

export interface AnalysisStatus {
  game_id: string;
  status: 'not_started' | 'pending' | 'ready' | 'error';
  moneyline?: MoneylinePrediction;
  spread?: SpreadPrediction;
  error?: string;
}

export interface GameAnalysis {
  game_id: string;
  home_team: string;
  home_abbrev: string;
  away_team: string;
  away_abbrev: string;
  game_time?: string;
  status?: string;
  moneyline_prediction?: MoneylinePrediction;
  spread_prediction?: SpreadPrediction;
  market_odds?: Record<string, unknown>;
  recommendations: Record<string, unknown>[];
}

export interface MoneylineOdds {
  home: number;
  away: number;
}

export interface SpreadOdds {
  home_line: number;
  home_odds: number;
  away_line: number;
  away_odds: number;
}

export interface TotalOdds {
  line: number;
  over_odds: number;
  under_odds: number;
}

export interface GameOdds {
  game_id?: string;
  moneyline?: MoneylineOdds;
  spread?: SpreadOdds;
  total?: TotalOdds;
  sportsbook?: string;
  last_updated?: string;
}

export interface OddsResponse {
  odds: Record<string, GameOdds>;
}

export interface BestBet {
  player_name: string;
  player_id: number;
  team: string;
  game_id: string;
  prop_type: string;
  prediction: number;
  line: number;
  edge: number;
  edge_pct: number;
  pick: string;
  confidence: number;
}

export interface BestBetsResponse {
  best_bets: BestBet[];
  count: number;
  filters: {
    min_confidence: number;
    min_edge: number;
    prop_types?: string[];
    pick_type?: string;
  };
}

export interface HealthResponse {
  status: string;
  service: string;
  timestamp: string;
  models_loaded: boolean;
}

// Prop types constant
export const PROP_TYPES = ['Points', 'Rebounds', 'Assists', '3PM', 'PRA'] as const;
export type PropType = (typeof PROP_TYPES)[number];

// Filter state
export interface FilterState {
  minConfidence: number;
  minEdge: number;
  propTypes: PropType[];
  pickType: 'OVER' | 'UNDER' | null;
  sortBy: string;
  sortOrder: 'asc' | 'desc';
}
