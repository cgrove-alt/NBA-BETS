import { useQuery } from '@tanstack/react-query';
import { getLiveStats } from '../lib/api';
import type { LiveStatsResponse } from '../lib/types';

/**
 * Check if a game is currently in progress
 */
function isGameLive(status: string | undefined): boolean {
  if (!status) return false;
  if (status === 'In Progress') return true;
  if (status.includes('Qtr') || status.includes('Half') || status.includes('OT')) return true;
  return false;
}

/**
 * Check if a game has completed
 */
function isGameFinal(status: string | undefined): boolean {
  return status === 'Final';
}

/**
 * Hook to fetch live player stats during games
 *
 * - Polls every 15 seconds for in-progress games
 * - Fetches once for completed games (final stats)
 * - Does not fetch for games that haven't started
 */
export function useLiveStats(gameId: string | null, gameStatus: string | undefined) {
  const isLive = isGameLive(gameStatus);
  const isFinal = isGameFinal(gameStatus);
  const shouldFetch = isLive || isFinal;

  const query = useQuery<LiveStatsResponse>({
    queryKey: ['liveStats', gameId],
    queryFn: () => getLiveStats(gameId!),
    enabled: !!gameId && shouldFetch,
    refetchInterval: isLive ? 15000 : false,  // Poll every 15 seconds during live games
    staleTime: isLive ? 10000 : Infinity,     // Consider data stale after 10 seconds for live, never for final
    gcTime: 60000,                            // Keep in cache for 1 minute
  });

  return {
    liveStats: query.data?.stats || {},
    gameStatus: query.data?.status,
    timestamp: query.data?.timestamp,
    isLoading: query.isLoading,
    error: query.error,
    isLive,
    isFinal,
  };
}
