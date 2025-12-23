import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getProps, startPropsFetch } from '../lib/api';
import type { Game } from '../lib/types';

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
      const status = query.state.data?.status;
      return status === 'pending' ? 2000 : false;
    },
  });

  // Mutation to start fetch
  const startFetch = useMutation({
    mutationFn: () => {
      if (!gameId || !game) throw new Error('No game selected');
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

  return {
    props: propsQuery.data,
    isLoading: propsQuery.isLoading,
    isPending: propsQuery.data?.status === 'pending',
    isReady: propsQuery.data?.status === 'ready',
    error: propsQuery.error,
    startFetch: startFetch.mutate,
    isStarting: startFetch.isPending,
  };
}
