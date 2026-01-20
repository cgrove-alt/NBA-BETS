import { useState, useEffect, useMemo, useRef } from 'react';
import { Loader2, Lock, Radio } from 'lucide-react';
import { GameSelector } from '../components/game/GameSelector';
import { DateSelector, getTodayDate } from '../components/game/DateSelector';
import { EnhancedFilterPanel } from '../components/predictions/EnhancedFilterPanel';
import { ActiveFiltersBar } from '../components/predictions/ActiveFiltersBar';
import { PropTable } from '../components/predictions/PropTable';
import { BestBets } from '../components/predictions/BestBets';
import { QuickPicks } from '../components/predictions/QuickPicks';
import { GamePredictions } from '../components/predictions/GamePredictions';
import { useGames } from '../hooks/useGames';
import { usePredictions } from '../hooks/usePredictions';
import { useFilters } from '../hooks/useFilters';
import { useLiveStats } from '../hooks/useLiveStats';
import { PROP_TYPES } from '../lib/types';
import type { PropPrediction } from '../lib/types';

export function Predictions() {
  const [selectedDate, setSelectedDate] = useState<string>(getTodayDate());
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const bestBetsRef = useRef<HTMLDivElement>(null);

  // Fetch games for selected date
  const { data: gamesData, isLoading: gamesLoading, error: gamesError } = useGames(selectedDate);
  const games = gamesData?.games || [];

  // Get selected game
  const selectedGame = games.find((g) => g.game_id === selectedGameId) || null;

  // Build game context string (e.g., "NYK @ DET")
  const gameContext = selectedGame
    ? `${selectedGame.visitor_team.abbreviation} @ ${selectedGame.home_team.abbreviation}`
    : undefined;

  // Fetch predictions
  const {
    props: propsData,
    isLoading: propsLoading,
    isPending,
    isReady,
    isLocked,
    lockedMessage,
    gameStarted,
    startFetch,
    isStarting,
  } = usePredictions(selectedGameId, selectedGame);

  // Filters with preset management
  const {
    filters,
    updateFilters,
    resetFilters,
    presets,
    savePreset,
    loadPreset,
    deletePreset,
  } = useFilters();

  // Live stats tracking
  const { liveStats, isLive, isFinal } = useLiveStats(selectedGameId, selectedGame?.status);

  // Auto-select first game
  useEffect(() => {
    if (games.length > 0 && !selectedGameId) {
      setSelectedGameId(games[0].game_id);
    }
  }, [games, selectedGameId]);

  // Auto-start fetch when game selected and not started (only if game hasn't started)
  useEffect(() => {
    if (selectedGameId && selectedGame && propsData?.status === 'not_started' && !gameStarted) {
      startFetch();
    }
  }, [selectedGameId, selectedGame, propsData?.status, startFetch, gameStarted]);

  // All players from both teams
  const allPlayers = useMemo(() => {
    if (!propsData) return [];
    return [...(propsData.home_props || []), ...(propsData.away_props || [])];
  }, [propsData]);

  // Count filtered results with new max filters
  const { filteredCount, totalCount } = useMemo(() => {
    if (allPlayers.length === 0) return { filteredCount: 0, totalCount: 0 };

    let total = 0;
    let filtered = 0;

    for (const player of allPlayers) {
      for (const propType of filters.propTypes) {
        const prop = propType === '3PM' ? player['3PM'] : player[propType as keyof typeof player];
        if (prop && typeof prop === 'object' && 'pick' in prop) {
          const p = prop as { pick: string; confidence: number; edge: number };
          if (p.pick === '-') continue;

          total++;

          // Apply all filters
          if (p.confidence < filters.minConfidence) continue;
          if (filters.maxConfidence && p.confidence > filters.maxConfidence) continue;

          const propPred = p as PropPrediction;
          const edgeValue = filters.edgeMode === 'percentage'
            ? propPred.edge_pct || Math.abs(propPred.edge)
            : Math.abs(propPred.edge);
          if (edgeValue < filters.minEdge) continue;
          if (filters.maxEdge && edgeValue > filters.maxEdge) continue;

          if (filters.pickType && p.pick !== filters.pickType) continue;

          filtered++;
        }
      }
    }
    return { filteredCount: filtered, totalCount: total };
  }, [allPlayers, filters]);

  // Handle filter chip removal
  const handleRemoveFilter = (filterKey: keyof typeof filters, value?: string) => {
    switch (filterKey) {
      case 'minConfidence':
        updateFilters({ minConfidence: 55, maxConfidence: undefined });
        break;
      case 'minEdge':
        updateFilters({ minEdge: 4, maxEdge: undefined });
        break;
      case 'pickType':
        updateFilters({ pickType: null });
        break;
      case 'propTypes':
        if (value) {
          const updated = filters.propTypes.filter((p) => p !== value);
          updateFilters({ propTypes: updated.length > 0 ? updated : [...PROP_TYPES] });
        }
        break;
    }
  };

  const handleGameSelect = (gameId: string) => {
    setSelectedGameId(gameId);
  };

  const scrollToBestBets = () => {
    bestBetsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Player Props</h1>
          <p className="text-sm text-text-secondary mt-1">
            ML-powered predictions for NBA games
          </p>
        </div>
      </div>

      {/* Date Selector */}
      <DateSelector
        selectedDate={selectedDate}
        onSelectDate={(date) => {
          setSelectedDate(date);
          setSelectedGameId(null);
        }}
      />

      {/* Game Selector */}
      <div className="max-w-md">
        <GameSelector
          games={games}
          selectedGameId={selectedGameId}
          onSelectGame={handleGameSelect}
          loading={gamesLoading}
        />
      </div>

      {/* Error state */}
      {gamesError && (
        <div className="bg-danger-light border border-accent-danger/30 rounded-lg p-4 text-accent-danger">
          Error loading games. Make sure the API server is running.
        </div>
      )}

      {/* Game Predictions (Spread/Moneyline) */}
      {selectedGameId && selectedGame && (
        <GamePredictions gameId={selectedGameId} game={selectedGame} />
      )}

      {/* Quick Picks Summary Bar */}
      {isReady && allPlayers.length > 0 && (
        <QuickPicks players={allPlayers} onScrollToBestBets={scrollToBestBets} />
      )}

      {/* Live tracking banner */}
      {isLive && selectedGameId && (
        <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4 flex items-center gap-3">
          <Radio className="text-blue-500 shrink-0 animate-pulse" size={20} />
          <div>
            <p className="text-blue-500 font-medium">Live Tracking Active</p>
            <p className="text-blue-500/80 text-sm">
              Tracking real-time stats against pre-game predictions. Stats update every 15 seconds.
            </p>
          </div>
        </div>
      )}

      {/* Locked state - Game has started but not live tracking */}
      {isLocked && !isLive && selectedGameId && (
        <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-lg p-4 flex items-center gap-3">
          <Lock className="text-yellow-500 shrink-0" size={20} />
          <div>
            <p className="text-yellow-500 font-medium">Predictions Locked</p>
            <p className="text-yellow-500/80 text-sm">
              {lockedMessage || 'Game has started - predictions are locked for betting integrity. Pre-game predictions are preserved below if available.'}
            </p>
          </div>
        </div>
      )}

      {/* Loading state */}
      {(propsLoading || isPending || isStarting) && selectedGameId && (
        <div className="flex items-center justify-center py-12">
          <div className="flex items-center gap-3 text-text-secondary">
            <Loader2 className="animate-spin" size={24} />
            <span>Loading predictions...</span>
          </div>
        </div>
      )}

      {/* Active Filters Bar */}
      {isReady && allPlayers.length > 0 && (
        <ActiveFiltersBar
          filters={filters}
          games={games}
          onRemoveFilter={handleRemoveFilter}
          onResetAll={resetFilters}
          totalCount={totalCount}
          filteredCount={filteredCount}
        />
      )}

      {/* Main content */}
      {isReady && allPlayers.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-4">
            <EnhancedFilterPanel
              filters={filters}
              onFilterChange={updateFilters}
              resultCount={filteredCount}
              presets={presets}
              onSavePreset={savePreset}
              onLoadPreset={loadPreset}
              onDeletePreset={deletePreset}
            />
          </div>

          {/* Main content */}
          <div className="lg:col-span-3 space-y-6">
            {/* Best Bets */}
            <div ref={bestBetsRef}>
              <BestBets players={allPlayers} gameContext={gameContext} />
            </div>

            {/* Prop Tables */}
            {filters.propTypes.map((propType) => (
              <PropTable
                key={propType}
                propType={propType}
                players={allPlayers}
                filters={filters}
                liveStats={liveStats}
                isLive={isLive}
                isFinal={isFinal}
              />
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {isReady && allPlayers.length === 0 && (
        <div className="text-center py-12">
          <p className="text-text-muted">No player props available for this game.</p>
        </div>
      )}

      {/* No game selected */}
      {!selectedGameId && games.length > 0 && (
        <div className="text-center py-12">
          <p className="text-text-muted">Select a game to view predictions.</p>
        </div>
      )}

      {/* No games for selected date */}
      {!gamesLoading && games.length === 0 && !gamesError && (
        <div className="text-center py-12">
          <p className="text-text-muted">No NBA games scheduled for this date.</p>
        </div>
      )}
    </div>
  );
}
