// API client for NBA Props backend

import axios from 'axios';
import type {
  GamesResponse,
  PropsResponse,
  AnalysisStatus,
  GameAnalysis,
  OddsResponse,
  GameOdds,
  BestBetsResponse,
  HealthResponse,
  GameResults,
  LiveStatsResponse,
  BankrollResponse,
  PerformanceResponse,
  SystemHealthResponse,
  BriefingResponseData,
  SettingsData,
} from './types';

// API base URL - uses environment variable in production, proxy in development
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: 30000,
});

// Health check
export async function checkHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health');
  return data;
}

// Games
export async function getGames(date?: string, forceRefresh = false): Promise<GamesResponse> {
  const params: Record<string, string | boolean> = { force_refresh: forceRefresh };
  if (date) {
    params.date = date;
  }
  const { data } = await api.get<GamesResponse>('/games', { params });
  return data;
}

// Props
export async function getProps(gameId: string): Promise<PropsResponse> {
  const { data } = await api.get<PropsResponse>(`/games/${gameId}/props`);
  return data;
}

export async function startPropsFetch(
  gameId: string,
  homeAbbrev: string,
  awayAbbrev: string,
  propTypes?: string[]
): Promise<{ message: string; game_id: string }> {
  const { data } = await api.post(
    `/games/${gameId}/props/start`,
    propTypes ? { prop_types: propTypes } : undefined,
    { params: { home_abbrev: homeAbbrev, away_abbrev: awayAbbrev } }
  );
  return data;
}

// Analysis
export async function getAnalysisStatus(gameId: string): Promise<AnalysisStatus> {
  const { data } = await api.get<AnalysisStatus>(`/games/${gameId}/analysis/status`);
  return data;
}

export async function startAnalysis(
  gameId: string,
  homeAbbrev: string,
  awayAbbrev: string
): Promise<{ message: string; game_id: string }> {
  const { data } = await api.post(`/games/${gameId}/analysis/start`, null, {
    params: { home_abbrev: homeAbbrev, away_abbrev: awayAbbrev },
  });
  return data;
}

export async function getGameAnalysis(gameId: string): Promise<GameAnalysis> {
  const { data } = await api.get<GameAnalysis>(`/games/${gameId}/analysis`);
  return data;
}

// Odds
export async function getAllOdds(): Promise<OddsResponse> {
  const { data } = await api.get<OddsResponse>('/odds');
  return data;
}

export async function getGameOdds(gameId: string): Promise<GameOdds> {
  const { data } = await api.get<GameOdds>(`/odds/${gameId}`);
  return data;
}

// Best Bets
export async function getBestBets(params?: {
  minConfidence?: number;
  minEdge?: number;
  propTypes?: string[];
  pickType?: string;
  sortBy?: 'quality' | 'confidence' | 'edge';
}): Promise<BestBetsResponse> {
  const { data } = await api.get<BestBetsResponse>('/best-bets', {
    params: {
      min_confidence: params?.minConfidence,
      min_edge: params?.minEdge,
      prop_types: params?.propTypes?.join(','),
      pick_type: params?.pickType,
      sort_by: params?.sortBy,
    },
  });
  return data;
}

// Game Results
export async function getGameResults(gameId: string): Promise<GameResults> {
  const { data } = await api.get<GameResults>(`/games/${gameId}/results`);
  return data;
}

// Live Stats
export async function getLiveStats(gameId: string): Promise<LiveStatsResponse> {
  const { data } = await api.get<LiveStatsResponse>(`/games/${gameId}/live-stats`);
  return data;
}

// Bankroll
export async function fetchBankroll(): Promise<BankrollResponse> {
  const { data } = await api.get<BankrollResponse>('/bankroll');
  return data;
}

// Performance
export async function fetchPerformance(days: number = 30): Promise<PerformanceResponse> {
  const { data } = await api.get<PerformanceResponse>('/performance', { params: { days } });
  return data;
}

// System Health
export async function fetchSystemHealth(): Promise<SystemHealthResponse> {
  const { data } = await api.get<SystemHealthResponse>('/system-health');
  return data;
}

// Briefing
export async function fetchBriefing(date?: string): Promise<BriefingResponseData> {
  const { data } = await api.get<BriefingResponseData>('/briefing', { params: date ? { date } : {} });
  return data;
}

// Settings
export async function fetchSettings(): Promise<SettingsData> {
  const { data } = await api.get<SettingsData>('/settings');
  return data;
}

export async function updateSettings(settings: Partial<SettingsData>): Promise<SettingsData> {
  const { data } = await api.put<SettingsData>('/settings', settings);
  return data;
}
