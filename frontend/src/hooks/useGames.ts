import { useQuery } from '@tanstack/react-query';
import { getGames } from '../lib/api';

export function useGames(date?: string) {
  return useQuery({
    queryKey: ['games', date],
    queryFn: () => getGames(date),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
  });
}
