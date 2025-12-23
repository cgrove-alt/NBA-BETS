import { useState, useEffect, useMemo } from 'react';
import { Loader2 } from 'lucide-react';
import { GameSelector } from '../components/game/GameSelector';
import { FilterPanel } from '../components/predictions/FilterPanel';
import { PropTable } from '../components/predictions/PropTable';
import { BestBets } from '../components/predictions/BestBets';
import { GamePredictions } from '../components/predictions/GamePredictions';
import { useGames } from '../hooks/useGames';
import { usePredictions } from '../hooks/usePredictions';
import { useFilters } from '../hooks/useFilters';

export function Predictions() {
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);

  // Fetch games
  const { data: gamesData, isLoading: gamesLoading, error: gamesError } = useGames();
  const games = gamesData?.games || [];

  // Get selected game
  const selectedGame = games.find((g) => g.game_id === selectedGameId) || null;

  // Fetch predictions
  const {
    props: propsData,
    isLoading: propsLoading,
    isPending,
    isReady,
    startFetch,
    isStarting,
  } = usePredictions(selectedGameId, selectedGame);

  // Filters
  const { filters, updateFilters } = useFilters();

  // Auto-select first game
  useEffect(() => {
    if (games.length > 0 && !selectedGameId) {
      setSelectedGameId(games[0].game_id);
    }
  }, [games, selectedGameId]);

  // Auto-start fetch when game selected and not started
  useEffect(() => {
    if (selectedGameId && selectedGame && propsData?.status === 'not_started') {
      startFetch();
    }
  }, [selectedGameId, selectedGame, propsData?.status, startFetch]);

  // All players from both teams
  const allPlayers = useMemo(() => {
    if (!propsData) return [];
    return [...(propsData.home_props || []), ...(propsData.away_props || [])];
  }, [propsData]);

  // Count filtered results
  const filteredCount = useMemo(() => {
    if (allPlayers.length === 0) return 0;

    let count = 0;
    for (const player of allPlayers) {
      for (const propType of filters.propTypes) {
        const prop = propType === '3PM' ? player['3PM'] : player[propType as keyof typeof player];
        if (prop && typeof prop === 'object' && 'pick' in prop) {
          const p = prop as { pick: string; confidence: number; edge: number };
          if (p.pick === '-') continue;
          if (p.confidence < filters.minConfidence) continue;
          if (Math.abs(p.edge) < filters.minEdge) continue;
          if (filters.pickType && p.pick !== filters.pickType) continue;
          count++;
        }
      }
    }
    return count;
  }, [allPlayers, filters]);

  const handleGameSelect = (gameId: string) => {
    setSelectedGameId(gameId);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Player Props</h1>
          <p className="text-sm text-text-secondary mt-1">
            ML-powered predictions for today's games
          </p>
        </div>
      </div>

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

      {/* Loading state */}
      {(propsLoading || isPending || isStarting) && selectedGameId && (
        <div className="flex items-center justify-center py-12">
          <div className="flex items-center gap-3 text-text-secondary">
            <Loader2 className="animate-spin" size={24} />
            <span>Loading predictions...</span>
          </div>
        </div>
      )}

      {/* Main content */}
      {isReady && allPlayers.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-4">
            <FilterPanel
              filters={filters}
              onFilterChange={updateFilters}
              resultCount={filteredCount}
            />
          </div>

          {/* Main content */}
          <div className="lg:col-span-3 space-y-6">
            {/* Best Bets */}
            <BestBets players={allPlayers} />

            {/* Prop Tables */}
            {filters.propTypes.map((propType) => (
              <PropTable
                key={propType}
                propType={propType}
                players={allPlayers}
                filters={filters}
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

      {/* No games today */}
      {!gamesLoading && games.length === 0 && !gamesError && (
        <div className="text-center py-12">
          <p className="text-text-muted">No NBA games scheduled for today.</p>
        </div>
      )}
    </div>
  );
}
