import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getProps, startPropsFetch } from '../lib/api';
import type { Game } from '../lib/types';

/**
 * Check if a game has started based on its status
 */
function isGameStarted(status: string | undefined): boolean {
  if (!status) return false;
  // Known in-progress/completed statuses
  if (status === 'Final' || status === 'In Progress') return true;
  if (status.includes('Qtr') || status.includes('Half') || status.includes('OT')) return true;
  // Check if ISO datetime and compare to now
  if (status.includes('T') && status.includes(':')) {
    try {
      const gameTime = new Date(status);
      return new Date() >= gameTime;
    } catch {
      return false;
    }
  }
  return false;
}

export function usePredictions(gameId: string | null, game: Game | null) {
  const queryClient = useQueryClient();

  // Query for props
  const propsQuery = useQuery({
    queryKey: ['props', gameId],
    queryFn: () => getProps(gameId!),
    enabled: !!gameId,
    staleTime: 0,  // Never use stale data - always fetch fresh when game changes
    gcTime: 0,     // Don't cache results - prevents showing wrong game's players
    refetchInterval: (query) => {
      // Poll every 2 seconds while pending
      // Stop polling if locked, ready, or error
      const status = query.state.data?.status;
      if (status === 'locked' || status === 'ready' || status === 'error') {
        return false;
      }
      return status === 'pending' ? 2000 : false;
    },
  });

  // Mutation to start fetch
  const startFetch = useMutation({
    mutationFn: () => {
      if (!gameId || !game) throw new Error('No game selected');
      // Guard: Don't allow starting fetch if game has already started
      if (isGameStarted(game.status)) {
        throw new Error('Game has started - predictions are locked');
      }
      return startPropsFetch(
        gameId,
        game.home_team.abbreviation,
        game.visitor_team.abbreviation
      );
    },
    onSuccess: () => {
      // Invalidate to trigger refetch
      queryClient.invalidateQueries({ queryKey: ['props', gameId] });
    },
  });

  // Check if game has started (for UI disabling)
  const gameStarted = isGameStarted(game?.status);

  return {
    props: propsQuery.data,
    isLoading: propsQuery.isLoading,
    isPending: propsQuery.data?.status === 'pending',
    isReady: propsQuery.data?.status === 'ready',
    isLocked: propsQuery.data?.status === 'locked' || gameStarted,
    lockedMessage: propsQuery.data?.error,
    error: propsQuery.error,
    startFetch: startFetch.mutate,
    isStarting: startFetch.isPending,
    gameStarted,
  };
}
