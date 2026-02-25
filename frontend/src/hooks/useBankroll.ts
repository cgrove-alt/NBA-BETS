import { useQuery } from '@tanstack/react-query';
import { fetchBankroll } from '../lib/api';
import type { BankrollResponse } from '../lib/types';
import type { BankrollData } from '../components/v2/BankrollSummary';

/**
 * Shared hook for bankroll data used across Dashboard, AllPredictions, Performance, Settings.
 * Fetches from /api/bankroll and transforms to BankrollData format for components.
 */
export function useBankroll() {
  const query = useQuery<BankrollResponse>({
    queryKey: ['bankroll'],
    queryFn: fetchBankroll,
    staleTime: 60 * 1000, // 1 min
    refetchInterval: 5 * 60 * 1000, // Refresh every 5 min
  });

  const bankrollData: BankrollData = query.data
    ? {
        totalBankroll: query.data.current_bankroll,
        todayPnL: query.data.daily_pnl,
        weekPnL: query.data.weekly_pnl,
        monthPnL: query.data.monthly_pnl,
        allTimeROI: query.data.season_roi,
        winRate: query.data.win_rate,
        activeBets: query.data.active_bets,
        pendingBets: 0,
      }
    : {
        totalBankroll: 0,
        todayPnL: 0,
        weekPnL: 0,
        monthPnL: 0,
        allTimeROI: 0,
        winRate: 0,
        activeBets: 0,
        pendingBets: 0,
      };

  return {
    bankrollData,
    raw: query.data,
    isLoading: query.isLoading,
    error: query.error,
  };
}
