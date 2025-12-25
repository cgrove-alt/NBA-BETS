import { useQuery } from '@tanstack/react-query';
import { getGameResults } from '../lib/api';

export function useGameResults(gameId: string | null) {
  const { data: results, isLoading, error } = useQuery({
    queryKey: ['gameResults', gameId],
    queryFn: () => getGameResults(gameId!),
    enabled: !!gameId,
    staleTime: 1000 * 60 * 5, // Cache for 5 minutes (results don't change)
  });

  return { results, isLoading, error };
}
