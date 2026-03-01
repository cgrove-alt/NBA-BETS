import { useState, useMemo, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Filter,
  Shield,
  Flame,
  Zap,
  Loader2,
  X,
  SlidersHorizontal,
} from 'lucide-react';
import { ResponsiveLayout } from '../../components/v2/ResponsiveLayout';
import { BetCard } from '../../components/v2/BetCard';
import type { BetCardData } from '../../components/v2/BetCard';
import { Card } from '../../components/v2/Card';
import { Button } from '../../components/v2/Button';
import { Badge } from '../../components/v2/Badge';
import { getBestBets, getGames } from '../../lib/api';
import type { Game } from '../../lib/types';
import { getTodayDate } from '../../components/game/DateSelector';
import { useBankroll } from '../../hooks/useBankroll';

/**
 * Check if a game has started based on its status
 * Games are locked once they start to prevent retroactive betting
 */
function isGameStarted(status: string | undefined): boolean {
  if (!status) return false;

  const startedPatterns = [
    'Qtr', 'Quarter', 'Half', 'OT', 'Final', 'In Progress', 'Live',
  ];

  return startedPatterns.some(pattern =>
    status.toLowerCase().includes(pattern.toLowerCase())
  );
}

// Filter presets
type FilterPreset = 'all' | 'safe' | 'high-reward' | 'whale';

interface FilterConfig {
  minConfidence: number;
  minEdge: number;
  label: string;
  icon: React.ReactNode;
  description: string;
}

// Note: Model confidence outputs range 50-70%, adjust thresholds accordingly
const FILTER_PRESETS: Record<FilterPreset, FilterConfig> = {
  all: {
    minConfidence: 50,
    minEdge: 0,
    label: 'All Picks',
    icon: <Filter className="w-4 h-4" />,
    description: 'All available predictions',
  },
  safe: {
    minConfidence: 58,
    minEdge: 3,
    label: 'Safe Bets',
    icon: <Shield className="w-4 h-4" />,
    description: 'Higher confidence picks',
  },
  'high-reward': {
    minConfidence: 52,
    minEdge: 8,
    label: 'High Reward',
    icon: <Flame className="w-4 h-4" />,
    description: 'High edge value opportunities',
  },
  whale: {
    minConfidence: 60,
    minEdge: 10,
    label: 'Whale Plays',
    icon: <Zap className="w-4 h-4" />,
    description: 'Best confidence + edge combo',
  },
};

/**
 * AllPredictions - Browse all betting predictions
 *
 * Features:
 * - Filter presets (Safe, High Reward, Whale Plays)
 * - Search and sort
 * - Grid/List view toggle
 * - Prop type filters
 */
export function AllPredictions() {
  const [selectedPreset, setSelectedPreset] = useState<FilterPreset>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [propTypeFilter, setPropTypeFilter] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);

  const selectedDate = getTodayDate();
  const filterConfig = FILTER_PRESETS[selectedPreset];

  // Fetch games for context
  const { data: gamesData } = useQuery({
    queryKey: ['games', selectedDate],
    queryFn: () => getGames(selectedDate),
    staleTime: 5 * 60 * 1000,
  });

  // Fetch best bets with current filter
  const { data: bestBetsData, isLoading } = useQuery({
    queryKey: ['bestBets', filterConfig.minConfidence, filterConfig.minEdge],
    queryFn: () =>
      getBestBets({
        minConfidence: filterConfig.minConfidence,
        minEdge: filterConfig.minEdge,
      }),
    staleTime: 2 * 60 * 1000,
  });

  const gamesMap = useMemo(() => {
    const games = gamesData?.games || [];
    const map = new Map<string, Game>();
    games.forEach((g) => map.set(g.game_id, g));
    return map;
  }, [gamesData]);

  // Transform and filter bets
  // CRITICAL: Lock bets for games that have already started (betting integrity)
  const bets: BetCardData[] = useMemo(() => {
    const bestBets = bestBetsData?.best_bets || [];

    // Apply prop type filter
    const filtered = propTypeFilter
      ? bestBets.filter((b) => b.prop_type === propTypeFilter)
      : bestBets;

    return filtered.map((bet) => {
      const game = gamesMap.get(bet.game_id);
      const gameStatus = game?.status;
      const isLocked = isGameStarted(gameStatus);

      const signals: BetCardData['signals'] = (bet.signals || []).map((s) => ({
        label: s,
        type: (s.includes('Weak') || s.includes('ML Model') || s.includes('Real Line'))
          ? 'positive' as const
          : s.includes('Strong defense')
            ? 'negative' as const
            : 'neutral' as const,
      }));
      if (signals.length === 0) {
        signals.push({ label: bet.prop_type, type: 'neutral' as const });
      }

      return {
        id: `${bet.game_id}-${bet.player_id}-${bet.prop_type}`,
        matchup: {
          homeTeam: game?.home_team?.abbreviation || 'HOME',
          homeAbbrev: game?.home_team?.abbreviation || '---',
          awayTeam: game?.visitor_team?.abbreviation || 'AWAY',
          awayAbbrev: game?.visitor_team?.abbreviation || '---',
          gameTime: game?.game_time || new Date().toISOString(),
          status: isLocked ? 'live' as const : 'upcoming' as const,
        },
        pick: {
          type: 'prop' as const,
          selection: `${bet.player_name} ${bet.pick} ${bet.line} ${bet.prop_type}`,
          odds: 0,
        },
        edge: bet.edge_pct,
        confidence: bet.confidence,
        signals,
        locked: isLocked,
        rank: bet.rank,
        explanation: bet.explanation,
        seasonAvg: bet.season_avg,
        recentAvg: bet.recent_avg,
      };
    });
  }, [bestBetsData, gamesMap, propTypeFilter]);

  // Get unique prop types for filter
  const propTypes = useMemo(() => {
    const types = new Set<string>();
    bestBetsData?.best_bets?.forEach((b) => types.add(b.prop_type));
    return Array.from(types);
  }, [bestBetsData]);

  // Real bankroll data from /api/bankroll
  const { bankrollData } = useBankroll();

  const [copiedBetId, setCopiedBetId] = useState<string | null>(null);

  const handleTakeBet = useCallback((bet: BetCardData) => {
    const text = `${bet.pick.selection} | Edge: ${bet.edge > 0 ? '+' : ''}${bet.edge.toFixed(1)}% | Conf: ${bet.confidence}%`;
    navigator.clipboard.writeText(text).catch(() => {});
    setCopiedBetId(bet.id);
    setTimeout(() => setCopiedBetId(null), 2000);
  }, []);

  const handleExpandBet = useCallback(() => {
    // No-op: card already shows details inline
  }, []);

  return (
    <ResponsiveLayout bankroll={bankrollData} activePage="predictions">
      <div className="space-y-6 pb-20 md:pb-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">Predictions</h1>
            <p className="text-sm text-text-muted mt-1">
              {bets.length} picks available
            </p>
          </div>
          <Button
            variant="secondary"
            size="sm"
            icon={<SlidersHorizontal className="w-4 h-4" />}
            onClick={() => setShowFilters(!showFilters)}
          >
            Filters
          </Button>
        </div>

        {/* Filter Presets - Scrollable on mobile */}
        <div className="flex gap-2 overflow-x-auto pb-2 -mx-4 px-4 md:mx-0 md:px-0 scrollbar-hide">
          {(Object.entries(FILTER_PRESETS) as [FilterPreset, FilterConfig][]).map(
            ([key, config]) => (
              <Button
                key={key}
                variant={selectedPreset === key ? 'primary' : 'ghost'}
                size="sm"
                icon={config.icon}
                onClick={() => setSelectedPreset(key)}
                className="whitespace-nowrap shrink-0"
              >
                {config.label}
              </Button>
            )
          )}
        </div>

        {/* Filter Description */}
        <Card variant="glass" className="p-3">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[rgba(0,212,255,0.1)]">
              {filterConfig.icon}
            </div>
            <div>
              <div className="font-semibold text-text-primary">{filterConfig.label}</div>
              <div className="text-sm text-text-muted">{filterConfig.description}</div>
            </div>
            <div className="ml-auto text-right hidden sm:block">
              <div className="text-xs text-text-muted">Confidence ≥ {filterConfig.minConfidence}%</div>
              <div className="text-xs text-text-muted">Edge ≥ {filterConfig.minEdge}%</div>
            </div>
          </div>
        </Card>

        {/* Advanced Filters (collapsible) */}
        {showFilters && (
          <Card className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-text-primary">Filter by Prop Type</h3>
              {propTypeFilter && (
                <Button
                  variant="ghost"
                  size="sm"
                  icon={<X className="w-3 h-3" />}
                  onClick={() => setPropTypeFilter(null)}
                >
                  Clear
                </Button>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {propTypes.map((type) => (
                <Badge
                  key={type}
                  variant={propTypeFilter === type ? 'success' : 'default'}
                  size="md"
                  className="cursor-pointer"
                  onClick={() =>
                    setPropTypeFilter(propTypeFilter === type ? null : type)
                  }
                >
                  {type}
                </Badge>
              ))}
            </div>
          </Card>
        )}

        {/* View Mode Toggle (desktop only) */}
        <div className="hidden md:flex justify-end gap-2">
          <Button
            variant={viewMode === 'grid' ? 'primary' : 'ghost'}
            size="sm"
            onClick={() => setViewMode('grid')}
          >
            Grid
          </Button>
          <Button
            variant={viewMode === 'list' ? 'primary' : 'ghost'}
            size="sm"
            onClick={() => setViewMode('list')}
          >
            List
          </Button>
        </div>

        {/* Bets Grid/List */}
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-text-muted" />
          </div>
        ) : bets.length > 0 ? (
          viewMode === 'grid' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {bets.map((bet) => (
                <BetCard
                  key={bet.id}
                  bet={{ ...bet, copied: copiedBetId === bet.id }}
                  variant="compact"
                  onTake={handleTakeBet}
                  onExpand={handleExpandBet}
                />
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              {bets.map((bet) => (
                <BetCard
                  key={bet.id}
                  bet={{ ...bet, copied: copiedBetId === bet.id }}
                  variant="list"
                  onTake={handleTakeBet}
                  onExpand={handleExpandBet}
                />
              ))}
            </div>
          )
        ) : (
          <Card className="p-12 text-center">
            <div className="text-text-muted mb-2">No predictions match your filters</div>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                setSelectedPreset('all');
                setPropTypeFilter(null);
              }}
            >
              Reset Filters
            </Button>
          </Card>
        )}
      </div>
    </ResponsiveLayout>
  );
}
