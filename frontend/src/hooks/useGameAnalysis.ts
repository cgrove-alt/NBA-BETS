import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getAnalysisStatus, startAnalysis, getGameAnalysis } from '../lib/api';
import type { Game } from '../lib/types';

export function useGameAnalysis(gameId: string | null, game: Game | null) {
  const queryClient = useQueryClient();

  // Query for analysis status
  const statusQuery = useQuery({
    queryKey: ['analysis-status', gameId],
    queryFn: () => getAnalysisStatus(gameId!),
    enabled: !!gameId,
    staleTime: 0,
    gcTime: 0,
    refetchInterval: (query) => {
      // Poll every 2 seconds while pending
      const status = query.state.data?.status;
      return status === 'pending' ? 2000 : false;
    },
  });

  // Query for full analysis (only when status is ready)
  const analysisQuery = useQuery({
    queryKey: ['analysis', gameId],
    queryFn: () => getGameAnalysis(gameId!),
    enabled: !!gameId && statusQuery.data?.status === 'ready',
    staleTime: 30000, // Cache for 30 seconds
  });

  // Mutation to start analysis
  const startAnalysisMutation = useMutation({
    mutationFn: () => {
      if (!gameId || !game) throw new Error('No game selected');
      return startAnalysis(
        gameId,
        game.home_team.abbreviation,
        game.visitor_team.abbreviation
      );
    },
    onSuccess: () => {
      // Invalidate to trigger refetch
      queryClient.invalidateQueries({ queryKey: ['analysis-status', gameId] });
    },
  });

  const status = statusQuery.data?.status || 'not_started';
  const isReady = status === 'ready';

  return {
    status,
    statusData: statusQuery.data,
    analysis: analysisQuery.data,
    isLoading: statusQuery.isLoading || (isReady && analysisQuery.isLoading),
    isPending: status === 'pending',
    isReady,
    error: statusQuery.error || analysisQuery.error,
    startAnalysis: startAnalysisMutation.mutate,
    isStarting: startAnalysisMutation.isPending,
  };
}
