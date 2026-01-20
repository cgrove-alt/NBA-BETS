import { useState } from 'react';
import { ChevronDown, ChevronUp, Percent } from 'lucide-react';
import { PROP_TYPES } from '../../lib/types';
import type { FilterState, PropType, Game } from '../../lib/types';
import { FilterPresets } from './FilterPresets';
import { formatMatchup } from '../../lib/utils';

interface EnhancedFilterPanelProps {
  filters: FilterState;
  games: Game[];
  onFilterChange: (filters: Partial<FilterState>) => void;
  resultCount: number;
  // Preset management
  presets: any[];
  onSavePreset: (name: string, description?: string) => void;
  onLoadPreset: (presetId: string) => void;
  onDeletePreset: (presetId: string) => void;
}

export function EnhancedFilterPanel({
  filters,
  games,
  onFilterChange,
  resultCount,
  presets,
  onSavePreset,
  onLoadPreset,
  onDeletePreset,
}: EnhancedFilterPanelProps) {
  const [expandedSections, setExpandedSections] = useState({
    games: true,
    confidence: true,
    edge: true,
    propTypes: true,
    pickType: true,
    sort: true,
    presets: false,
  });

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const handleConfidenceChange = (value: number, type: 'min' | 'max') => {
    if (type === 'min') {
      onFilterChange({ minConfidence: value });
    } else {
      onFilterChange({ maxConfidence: value === 80 ? undefined : value });
    }
  };

  const handleEdgeChange = (value: number, type: 'min' | 'max') => {
    if (type === 'min') {
      onFilterChange({ minEdge: value });
    } else {
      onFilterChange({ maxEdge: value === 10 ? undefined : value });
    }
  };

  const handlePropTypeToggle = (propType: PropType) => {
    const current = filters.propTypes;
    const updated = current.includes(propType)
      ? current.filter((p) => p !== propType)
      : [...current, propType];
    onFilterChange({ propTypes: updated.length > 0 ? updated : [...PROP_TYPES] });
  };

  const handlePickTypeChange = (pickType: 'OVER' | 'UNDER' | null) => {
    onFilterChange({ pickType: filters.pickType === pickType ? null : pickType });
  };

  const handleGameToggle = (gameId: string) => {
    const current = filters.gameIds;
    const updated = current.includes(gameId)
      ? current.filter((id) => id !== gameId)
      : [...current, gameId];
    onFilterChange({ gameIds: updated });
  };

  const handleSelectAllGames = () => {
    onFilterChange({ gameIds: [] }); // Empty array = all games
  };

  const handleDeselectAllGames = () => {
    onFilterChange({ gameIds: games.map((g) => g.game_id).slice(0, 1) }); // Select only first game
  };

  const toggleEdgeMode = () => {
    onFilterChange({
      edgeMode: filters.edgeMode === 'points' ? 'percentage' : 'points',
    });
  };

  const handleSortByChange = (sortBy: string) => {
    onFilterChange({ sortBy });
  };

  const currentFiltersActive =
    filters.minConfidence > 55 ||
    filters.maxConfidence !== undefined ||
    filters.minEdge > 4 ||
    filters.maxEdge !== undefined ||
    filters.pickType !== null ||
    filters.propTypes.length < 5 ||
    (filters.gameIds.length > 0 && filters.gameIds.length < games.length);

  return (
    <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <h3 className="text-sm font-semibold text-text-primary">Filters</h3>
        <span className="text-xs font-medium text-accent-primary">{resultCount} results</span>
      </div>

      <div className="p-4 space-y-4">
        {/* Game Filter */}
        {games.length > 1 && (
          <div>
            <button
              onClick={() => toggleSection('games')}
              className="flex items-center justify-between w-full mb-2"
            >
              <label className="text-xs text-text-secondary font-medium">Games</label>
              {expandedSections.games ? (
                <ChevronUp size={14} className="text-text-muted" />
              ) : (
                <ChevronDown size={14} className="text-text-muted" />
              )}
            </button>
            {expandedSections.games && (
              <div className="space-y-2">
                <div className="flex gap-2 text-xs mb-2">
                  <button
                    onClick={handleSelectAllGames}
                    className="px-2 py-1 text-accent-primary hover:bg-accent-primary/10 rounded transition-colors"
                  >
                    All
                  </button>
                  <button
                    onClick={handleDeselectAllGames}
                    className="px-2 py-1 text-text-muted hover:bg-bg-hover rounded transition-colors"
                  >
                    Clear
                  </button>
                </div>
                <div className="space-y-1.5 max-h-48 overflow-y-auto">
                  {games.map((game) => {
                    const isSelected =
                      filters.gameIds.length === 0 || filters.gameIds.includes(game.game_id);
                    const matchup = formatMatchup(
                      game.home_team.abbreviation,
                      game.visitor_team.abbreviation
                    );
                    return (
                      <label
                        key={game.game_id}
                        className="flex items-center gap-2 p-2 bg-bg-tertiary rounded cursor-pointer hover:bg-bg-hover transition-colors"
                      >
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => handleGameToggle(game.game_id)}
                          className="w-3.5 h-3.5 rounded border-border text-accent-primary focus:ring-accent-primary focus:ring-offset-0 focus:ring-1"
                        />
                        <span className="text-xs text-text-primary">{matchup}</span>
                        {game.status && (
                          <span className="ml-auto text-xs text-text-muted">{game.status}</span>
                        )}
                      </label>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Confidence Range */}
        <div>
          <button
            onClick={() => toggleSection('confidence')}
            className="flex items-center justify-between w-full mb-2"
          >
            <label className="text-xs text-text-secondary font-medium">Confidence</label>
            {expandedSections.confidence ? (
              <ChevronUp size={14} className="text-text-muted" />
            ) : (
              <ChevronDown size={14} className="text-text-muted" />
            )}
          </button>
          {expandedSections.confidence && (
            <div className="space-y-3">
              {/* Min Confidence */}
              <div>
                <label className="text-xs text-text-secondary block mb-2">
                  Min: {filters.minConfidence}%
                </label>
                <input
                  type="range"
                  min="50"
                  max="80"
                  value={filters.minConfidence}
                  onChange={(e) => handleConfidenceChange(Number(e.target.value), 'min')}
                  className="w-full h-2 bg-bg-tertiary rounded-lg appearance-none cursor-pointer accent-accent-primary"
                />
              </div>
              {/* Max Confidence */}
              <div>
                <label className="text-xs text-text-secondary block mb-2">
                  Max: {filters.maxConfidence ?? 80}%
                </label>
                <input
                  type="range"
                  min={filters.minConfidence}
                  max="80"
                  value={filters.maxConfidence ?? 80}
                  onChange={(e) => handleConfidenceChange(Number(e.target.value), 'max')}
                  className="w-full h-2 bg-bg-tertiary rounded-lg appearance-none cursor-pointer accent-accent-primary"
                />
              </div>
              <div className="flex justify-between text-xs text-text-muted">
                <span>50%</span>
                <span>80%</span>
              </div>
            </div>
          )}
        </div>

        {/* Edge Range */}
        <div>
          <button
            onClick={() => toggleSection('edge')}
            className="flex items-center justify-between w-full mb-2"
          >
            <label className="text-xs text-text-secondary font-medium">Edge</label>
            <div className="flex items-center gap-2">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  toggleEdgeMode();
                }}
                className="flex items-center gap-1 px-1.5 py-0.5 text-xs bg-bg-tertiary text-text-muted hover:bg-bg-hover rounded transition-colors"
                title={`Switch to ${filters.edgeMode === 'points' ? 'percentage' : 'points'}`}
              >
                <Percent size={10} />
                {filters.edgeMode === 'points' ? 'pts' : '%'}
              </button>
              {expandedSections.edge ? (
                <ChevronUp size={14} className="text-text-muted" />
              ) : (
                <ChevronDown size={14} className="text-text-muted" />
              )}
            </div>
          </button>
          {expandedSections.edge && (
            <div className="space-y-3">
              {/* Min Edge */}
              <div>
                <label className="text-xs text-text-secondary block mb-2">
                  Min: {filters.minEdge.toFixed(1)}
                  {filters.edgeMode === 'points' ? ' pts' : '%'}
                </label>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.5"
                  value={filters.minEdge}
                  onChange={(e) => handleEdgeChange(Number(e.target.value), 'min')}
                  className="w-full h-2 bg-bg-tertiary rounded-lg appearance-none cursor-pointer accent-accent-primary"
                />
              </div>
              {/* Max Edge */}
              <div>
                <label className="text-xs text-text-secondary block mb-2">
                  Max: {(filters.maxEdge ?? 10).toFixed(1)}
                  {filters.edgeMode === 'points' ? ' pts' : '%'}
                </label>
                <input
                  type="range"
                  min={filters.minEdge}
                  max="10"
                  step="0.5"
                  value={filters.maxEdge ?? 10}
                  onChange={(e) => handleEdgeChange(Number(e.target.value), 'max')}
                  className="w-full h-2 bg-bg-tertiary rounded-lg appearance-none cursor-pointer accent-accent-primary"
                />
              </div>
            </div>
          )}
        </div>

        {/* Prop Types */}
        <div>
          <button
            onClick={() => toggleSection('propTypes')}
            className="flex items-center justify-between w-full mb-2"
          >
            <label className="text-xs text-text-secondary font-medium">Prop Types</label>
            {expandedSections.propTypes ? (
              <ChevronUp size={14} className="text-text-muted" />
            ) : (
              <ChevronDown size={14} className="text-text-muted" />
            )}
          </button>
          {expandedSections.propTypes && (
            <div className="flex flex-wrap gap-2">
              {PROP_TYPES.map((propType) => (
                <button
                  key={propType}
                  onClick={() => handlePropTypeToggle(propType)}
                  className={`
                    px-3 py-1.5 text-xs font-medium rounded transition-colors
                    ${
                      filters.propTypes.includes(propType)
                        ? 'bg-accent-primary text-white shadow-sm'
                        : 'bg-bg-tertiary text-text-secondary hover:bg-bg-hover'
                    }
                  `}
                >
                  {propType}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Pick Type (Bet Type) */}
        <div>
          <button
            onClick={() => toggleSection('pickType')}
            className="flex items-center justify-between w-full mb-2"
          >
            <label className="text-xs text-text-secondary font-medium">Bet Type</label>
            {expandedSections.pickType ? (
              <ChevronUp size={14} className="text-text-muted" />
            ) : (
              <ChevronDown size={14} className="text-text-muted" />
            )}
          </button>
          {expandedSections.pickType && (
            <div className="flex gap-2">
              <button
                onClick={() => handlePickTypeChange('OVER')}
                className={`
                  flex-1 px-3 py-2 text-xs font-medium rounded transition-all
                  ${
                    filters.pickType === 'OVER'
                      ? 'bg-accent-success text-white shadow-sm'
                      : 'bg-bg-tertiary text-text-secondary hover:bg-bg-hover'
                  }
                `}
              >
                OVER
              </button>
              <button
                onClick={() => handlePickTypeChange('UNDER')}
                className={`
                  flex-1 px-3 py-2 text-xs font-medium rounded transition-all
                  ${
                    filters.pickType === 'UNDER'
                      ? 'bg-accent-danger text-white shadow-sm'
                      : 'bg-bg-tertiary text-text-secondary hover:bg-bg-hover'
                  }
                `}
              >
                UNDER
              </button>
            </div>
          )}
        </div>

        {/* Sort By */}
        <div>
          <button
            onClick={() => toggleSection('sort')}
            className="flex items-center justify-between w-full mb-2"
          >
            <label className="text-xs text-text-secondary font-medium">Sort By</label>
            {expandedSections.sort ? (
              <ChevronUp size={14} className="text-text-muted" />
            ) : (
              <ChevronDown size={14} className="text-text-muted" />
            )}
          </button>
          {expandedSections.sort && (
            <select
              value={filters.sortBy}
              onChange={(e) => handleSortByChange(e.target.value)}
              className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded-lg text-text-primary focus:outline-none focus:border-accent-primary"
            >
              <option value="quality">Smart Sort (Quality)</option>
              <option value="confidence">Highest Confidence</option>
              <option value="edge">Biggest Edge</option>
            </select>
          )}
        </div>

        {/* Filter Presets */}
        <div>
          <button
            onClick={() => toggleSection('presets')}
            className="flex items-center justify-between w-full mb-2"
          >
            <label className="text-xs text-text-secondary font-medium">Presets</label>
            {expandedSections.presets ? (
              <ChevronUp size={14} className="text-text-muted" />
            ) : (
              <ChevronDown size={14} className="text-text-muted" />
            )}
          </button>
          {expandedSections.presets && (
            <FilterPresets
              presets={presets}
              currentFiltersActive={currentFiltersActive}
              onSavePreset={onSavePreset}
              onLoadPreset={onLoadPreset}
              onDeletePreset={onDeletePreset}
            />
          )}
        </div>
      </div>
    </div>
  );
}
