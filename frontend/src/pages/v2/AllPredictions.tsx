import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import {
  Filter,
  Shield,
  Flame,
  Zap,
  Loader2,
  X,
  SlidersHorizontal,
  Search,
  Clock,
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
import { useFilters } from '../../hooks/useFilters';
import { isGameStarted, classifySignals } from '../../lib/utils';

type QuickPreset = 'all' | 'safe' | 'high-reward' | 'whale' | 'custom';

interface PresetConfig {
  minConfidence: number;
  minEdge: number;
  label: string;
  icon: React.ReactNode;
}

const QUICK_PRESETS: Record<Exclude<QuickPreset, 'custom'>, PresetConfig> = {
  all: { minConfidence: 50, minEdge: 0, label: 'All Picks', icon: <Filter className="w-4 h-4" /> },
  safe: { minConfidence: 58, minEdge: 3, label: 'Safe Bets', icon: <Shield className="w-4 h-4" /> },
  'high-reward': { minConfidence: 52, minEdge: 8, label: 'High Reward', icon: <Flame className="w-4 h-4" /> },
  whale: { minConfidence: 60, minEdge: 10, label: 'Whale Plays', icon: <Zap className="w-4 h-4" /> },
};

const SORT_OPTIONS = [
  { value: 'quality' as const, label: 'Best Overall' },
  { value: 'confidence' as const, label: 'Highest Confidence' },
  { value: 'edge' as const, label: 'Highest Edge' },
];

export function AllPredictions() {
  const [searchParams] = useSearchParams();
  const initialGameId = searchParams.get('game');

  const { filters, updateFilters, resetFilters } = useFilters();

  const [selectedPreset, setSelectedPreset] = useState<QuickPreset>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [propTypeFilter, setPropTypeFilter] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(initialGameId);
  const [sortBy, setSortBy] = useState<'quality' | 'confidence' | 'edge'>(
    (filters.sortBy as 'quality' | 'confidence' | 'edge') || 'quality'
  );
  const [pickType, setPickType] = useState<'ALL' | 'OVER' | 'UNDER'>('ALL');
  const [searchTerm, setSearchTerm] = useState('');

  // Custom slider state (debounced)
  const [sliderConfidence, setSliderConfidence] = useState(filters.minConfidence);
  const [sliderEdge, setSliderEdge] = useState(filters.minEdge);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const emptyRetryCount = useRef(0);

  // For non-custom presets, derive the API values directly (no state needed)
  const presetConfidence = selectedPreset !== 'custom'
    ? QUICK_PRESETS[selectedPreset].minConfidence
    : null;
  const presetEdge = selectedPreset !== 'custom'
    ? QUICK_PRESETS[selectedPreset].minEdge
    : null;

  // Debounced API filter values for custom sliders only
  const [debouncedCustomConfidence, setDebouncedCustomConfidence] = useState(sliderConfidence);
  const [debouncedCustomEdge, setDebouncedCustomEdge] = useState(sliderEdge);

  useEffect(() => {
    if (selectedPreset !== 'custom') return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setDebouncedCustomConfidence(sliderConfidence);
      setDebouncedCustomEdge(sliderEdge);
      updateFilters({ minConfidence: sliderConfidence, minEdge: sliderEdge });
    }, 300);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [sliderConfidence, sliderEdge, selectedPreset, updateFilters]);

  // Final effective values for API calls
  const effectiveMinConfidence = presetConfidence ?? debouncedCustomConfidence;
  const effectiveMinEdge = presetEdge ?? debouncedCustomEdge;

  const selectedDate = getTodayDate();

  const { data: gamesData } = useQuery({
    queryKey: ['games', selectedDate],
    queryFn: () => getGames(selectedDate),
    staleTime: 5 * 60 * 1000,
  });

  const { data: bestBetsData, isLoading } = useQuery({
    queryKey: ['bestBets', effectiveMinConfidence, effectiveMinEdge, sortBy, pickType === 'ALL' ? undefined : pickType],
    queryFn: () =>
      getBestBets({
        minConfidence: effectiveMinConfidence,
        minEdge: effectiveMinEdge,
        sortBy,
        pickType: pickType === 'ALL' ? undefined : pickType,
      }),
    staleTime: 2 * 60 * 1000,
    // After a deploy, props generate in the background. Retry every 5s
    // up to 12 times (60s max) until real-time data arrives.
    refetchInterval: (query) => {
      const bets = query.state.data?.best_bets;
      const source = query.state.data?.data_source;
      const hasGames = (gamesData?.games?.length ?? 0) > 0;
      if (hasGames && (!bets || bets.length === 0 || source === 'precomputed')) {
        emptyRetryCount.current++;
        return emptyRetryCount.current <= 12 ? 5000 : false;
      }
      emptyRetryCount.current = 0;
      return false;
    },
  });

  const gamesMap = useMemo(() => {
    const games = gamesData?.games || [];
    const map = new Map<string, Game>();
    games.forEach((g) => map.set(g.game_id, g));
    return map;
  }, [gamesData]);

  // Build game chips data: game_id -> { label, count }
  const gameChips = useMemo(() => {
    const bestBets = bestBetsData?.best_bets || [];
    const countMap = new Map<string, number>();
    bestBets.forEach((b) => {
      countMap.set(b.game_id, (countMap.get(b.game_id) || 0) + 1);
    });
    const chips: { gameId: string; label: string; count: number; time: string }[] = [];
    for (const [gameId, count] of countMap) {
      const game = gamesMap.get(gameId);
      const away = game?.visitor_team?.abbreviation || '???';
      const home = game?.home_team?.abbreviation || '???';
      let time = '';
      if (game?.game_time) {
        try {
          time = new Date(game.game_time).toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            hour12: true,
          });
        } catch { /* ignore */ }
      }
      chips.push({ gameId, label: `${away} @ ${home}`, count, time });
    }
    return chips;
  }, [bestBetsData, gamesMap]);

  // Transform and filter bets
  const bets: BetCardData[] = useMemo(() => {
    let bestBets = bestBetsData?.best_bets || [];

    // Game filter
    if (selectedGameId) {
      bestBets = bestBets.filter((b) => b.game_id === selectedGameId);
    }

    // Prop type filter
    if (propTypeFilter) {
      bestBets = bestBets.filter((b) => b.prop_type === propTypeFilter);
    }

    // Player search (client-side)
    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase().trim();
      bestBets = bestBets.filter((b) => b.player_name.toLowerCase().includes(term));
    }

    return bestBets.map((bet) => {
      const game = gamesMap.get(bet.game_id);
      const gameStatus = game?.status;
      const isLocked = isGameStarted(gameStatus);

      const signals = classifySignals(bet.signals || [], bet.prop_type);

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
  }, [bestBetsData, gamesMap, propTypeFilter, selectedGameId, searchTerm]);

  // Get unique prop types for filter
  const propTypes = useMemo(() => {
    const types = new Set<string>();
    bestBetsData?.best_bets?.forEach((b) => types.add(b.prop_type));
    return Array.from(types);
  }, [bestBetsData]);

  const { bankrollData } = useBankroll();
  const [copiedBetId, setCopiedBetId] = useState<string | null>(null);

  const handleTakeBet = useCallback((bet: BetCardData) => {
    const text = `${bet.pick.selection} | Edge: ${bet.edge > 0 ? '+' : ''}${bet.edge.toFixed(1)}% | Conf: ${bet.confidence}%`;
    navigator.clipboard.writeText(text).catch(() => {});
    setCopiedBetId(bet.id);
    setTimeout(() => setCopiedBetId(null), 2000);
  }, []);

  const handleSelectPreset = (preset: QuickPreset) => {
    setSelectedPreset(preset);
    if (preset !== 'custom') {
      const config = QUICK_PRESETS[preset];
      setSliderConfidence(config.minConfidence);
      setSliderEdge(config.minEdge);
    }
  };

  // Count active filters for summary
  const activeFilterParts: string[] = [];
  if (selectedGameId) {
    const chip = gameChips.find(c => c.gameId === selectedGameId);
    if (chip) activeFilterParts.push(chip.label);
  }
  if (effectiveMinConfidence > 50) activeFilterParts.push(`Conf ≥ ${effectiveMinConfidence}%`);
  if (effectiveMinEdge > 0) activeFilterParts.push(`Edge ≥ ${effectiveMinEdge}%`);
  if (propTypeFilter) activeFilterParts.push(propTypeFilter);
  if (pickType !== 'ALL') activeFilterParts.push(pickType);
  if (searchTerm.trim()) activeFilterParts.push(`"${searchTerm.trim()}"`);

  const handleClearAll = () => {
    setSelectedPreset('all');
    setSelectedGameId(null);
    setPropTypeFilter(null);
    setPickType('ALL');
    setSearchTerm('');
    setSortBy('quality');
    setSliderConfidence(50);
    setSliderEdge(0);
    resetFilters();
  };

  return (
    <ResponsiveLayout bankroll={bankrollData} activePage="predictions">
      <div className="space-y-4 pb-20 md:pb-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">Predictions</h1>
            <p className="text-sm text-text-muted mt-1">
              {bets.length} pick{bets.length !== 1 ? 's' : ''} available
            </p>
          </div>
          <div className="flex items-center gap-2">
            {/* Sort dropdown */}
            <select
              value={sortBy}
              onChange={(e) => {
                const val = e.target.value as typeof sortBy;
                setSortBy(val);
                updateFilters({ sortBy: val });
              }}
              className="bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-[#00d4ff]"
            >
              {SORT_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            <Button
              variant={showFilters ? 'primary' : 'secondary'}
              size="sm"
              icon={<SlidersHorizontal className="w-4 h-4" />}
              onClick={() => setShowFilters(!showFilters)}
            >
              Filters
            </Button>
          </div>
        </div>

        {/* Game Selector Bar */}
        {gameChips.length > 0 && (
          <div className="flex gap-2 overflow-x-auto pb-1 -mx-4 px-4 md:mx-0 md:px-0 scrollbar-hide">
            <button
              onClick={() => setSelectedGameId(null)}
              className={`whitespace-nowrap shrink-0 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                selectedGameId === null
                  ? 'bg-[#00d4ff] text-black'
                  : 'bg-bg-tertiary text-text-muted hover:text-text-primary border border-border'
              }`}
            >
              All Games ({bestBetsData?.best_bets?.length || 0})
            </button>
            {gameChips.map(chip => (
              <button
                key={chip.gameId}
                onClick={() => setSelectedGameId(selectedGameId === chip.gameId ? null : chip.gameId)}
                className={`whitespace-nowrap shrink-0 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                  selectedGameId === chip.gameId
                    ? 'bg-[#00d4ff] text-black'
                    : 'bg-bg-tertiary text-text-muted hover:text-text-primary border border-border'
                }`}
              >
                {chip.label} {chip.time && <span className="text-xs opacity-75">{chip.time}</span>} ({chip.count})
              </button>
            ))}
          </div>
        )}

        {/* Search + Pick Type Row */}
        <div className="flex gap-2">
          {/* Player search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              placeholder="Search player..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-bg-tertiary border border-border rounded-lg pl-9 pr-8 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-[#00d4ff]"
            />
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-text-muted hover:text-text-primary"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {/* Pick type toggle */}
          <div className="flex rounded-lg border border-border overflow-hidden shrink-0">
            {(['ALL', 'OVER', 'UNDER'] as const).map(pt => (
              <button
                key={pt}
                onClick={() => setPickType(pt)}
                className={`px-3 py-2 text-xs font-semibold transition-colors ${
                  pickType === pt
                    ? 'bg-[#00d4ff] text-black'
                    : 'bg-bg-tertiary text-text-muted hover:text-text-primary'
                }`}
              >
                {pt}
              </button>
            ))}
          </div>
        </div>

        {/* Filter Presets */}
        <div className="flex gap-2 overflow-x-auto pb-1 -mx-4 px-4 md:mx-0 md:px-0 scrollbar-hide">
          {(Object.entries(QUICK_PRESETS) as [Exclude<QuickPreset, 'custom'>, PresetConfig][]).map(
            ([key, config]) => (
              <Button
                key={key}
                variant={selectedPreset === key ? 'primary' : 'ghost'}
                size="sm"
                icon={config.icon}
                onClick={() => handleSelectPreset(key)}
                className="whitespace-nowrap shrink-0"
              >
                {config.label}
              </Button>
            )
          )}
          <Button
            variant={selectedPreset === 'custom' ? 'primary' : 'ghost'}
            size="sm"
            icon={<SlidersHorizontal className="w-4 h-4" />}
            onClick={() => handleSelectPreset('custom')}
            className="whitespace-nowrap shrink-0"
          >
            Custom
          </Button>
        </div>

        {/* Custom Sliders (when Custom preset active) */}
        {selectedPreset === 'custom' && (
          <Card className="p-4 space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-text-primary">Min Confidence</label>
                <span className="text-sm font-semibold text-[#00d4ff]">{sliderConfidence}%</span>
              </div>
              <input
                type="range"
                min="50"
                max="70"
                step="1"
                value={sliderConfidence}
                onChange={(e) => setSliderConfidence(Number(e.target.value))}
                className="w-full accent-[#00d4ff]"
              />
              <div className="flex justify-between text-xs text-text-muted mt-1">
                <span>50%</span>
                <span>70%</span>
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-text-primary">Min Edge</label>
                <span className="text-sm font-semibold text-[#00d4ff]">{sliderEdge}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="20"
                step="1"
                value={sliderEdge}
                onChange={(e) => setSliderEdge(Number(e.target.value))}
                className="w-full accent-[#00d4ff]"
              />
              <div className="flex justify-between text-xs text-text-muted mt-1">
                <span>0%</span>
                <span>20%</span>
              </div>
            </div>
          </Card>
        )}

        {/* Advanced Filters (collapsible) — prop type filter */}
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

        {/* Active filter summary */}
        {activeFilterParts.length > 0 && (
          <div className="flex items-center gap-2 flex-wrap text-xs">
            <span className="text-text-muted">
              Showing {bets.length} pick{bets.length !== 1 ? 's' : ''} ·
            </span>
            {activeFilterParts.map((part, i) => (
              <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-bg-tertiary text-text-secondary border border-border">
                {part}
              </span>
            ))}
            <span className="text-text-muted">
              · Sorted by {SORT_OPTIONS.find(o => o.value === sortBy)?.label?.toLowerCase()}
            </span>
            <button onClick={handleClearAll} className="text-[#00d4ff] hover:underline ml-1">
              Clear all
            </button>
          </div>
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

        {/* Precomputed data banner */}
        {bestBetsData?.data_source === 'precomputed' && bets.length > 0 && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#1c2333] border border-[#30363d] text-sm text-text-muted">
            <Clock className="w-4 h-4 text-[#ff8800] animate-pulse" />
            <span>Predictions from earlier today. Live predictions updating...</span>
          </div>
        )}

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
              onClick={handleClearAll}
            >
              Reset Filters
            </Button>
          </Card>
        )}
      </div>
    </ResponsiveLayout>
  );
}
