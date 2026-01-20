import { useState, useMemo } from 'react';
import { ChevronDown, ChevronUp, Percent, Info } from 'lucide-react';
import { PROP_TYPES, POSITIONS } from '../../lib/types';
import type { FilterState, PropType, FilterPreset, Game, Position, PlayerProp } from '../../lib/types';
import { FilterPresets } from './FilterPresets';
import { Tooltip } from '../ui/Tooltip';
import { ConfidenceExplanation } from './ConfidenceExplanation';
import { formatMatchup } from '../../lib/utils';

interface EnhancedFilterPanelProps {
  filters: FilterState;
  onFilterChange: (filters: Partial<FilterState>) => void;
  resultCount: number;
  // Game selection
  games: Game[];
  selectedGameId: string | null;
  onGameSelect: (gameId: string) => void;
  // Player data (for extracting unique teams/positions)
  players?: PlayerProp[];
  // Preset management
  presets: FilterPreset[];
  onSavePreset: (name: string, description?: string) => void;
  onLoadPreset: (presetId: string) => void;
  onDeletePreset: (presetId: string) => void;
}

export function EnhancedFilterPanel({
  filters,
  onFilterChange,
  resultCount,
  games,
  selectedGameId,
  onGameSelect,
  players = [],
  presets,
  onSavePreset,
  onLoadPreset,
  onDeletePreset,
}: EnhancedFilterPanelProps) {
  const [expandedSections, setExpandedSections] = useState({
    game: false,
    team: false,
    position: false,
    confidence: true,
    edge: true,
    propTypes: true,
    pickType: true,
    sort: true,
    presets: false,
  });

  // Extract unique teams from player data
  const availableTeams = useMemo(() => {
    const teams = new Set<string>();
    players.forEach((p) => {
      if (p.team) teams.add(p.team);
    });
    return Array.from(teams).sort();
  }, [players]);

  // Extract unique positions from player data
  const availablePositions = useMemo(() => {
    const positions = new Set<Position>();
    players.forEach((p) => {
      if (p.position && POSITIONS.includes(p.position as Position)) {
        positions.add(p.position as Position);
      }
    });
    return Array.from(positions).sort();
  }, [players]);

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

  const toggleEdgeMode = () => {
    onFilterChange({
      edgeMode: filters.edgeMode === 'points' ? 'percentage' : 'points',
    });
  };

  const handleSortByChange = (sortBy: string) => {
    onFilterChange({ sortBy });
  };

  const handleTeamToggle = (team: string) => {
    const current = filters.teams || [];
    const updated = current.includes(team)
      ? current.filter((t) => t !== team)
      : [...current, team];
    onFilterChange({ teams: updated.length > 0 ? updated : undefined });
  };

  const handlePositionToggle = (position: Position) => {
    const current = filters.positions || [];
    const updated = current.includes(position)
      ? current.filter((p) => p !== position)
      : [...current, position];
    onFilterChange({ positions: updated.length > 0 ? updated : undefined });
  };

  const currentFiltersActive =
    filters.minConfidence > 55 ||
    filters.maxConfidence !== undefined ||
    filters.minEdge > 4 ||
    filters.maxEdge !== undefined ||
    filters.pickType !== null ||
    filters.propTypes.length < 5 ||
    (filters.teams && filters.teams.length > 0) ||
    (filters.positions && filters.positions.length > 0);

  return (
    <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <h3 className="text-sm font-semibold text-text-primary">Filters</h3>
        <span className="text-xs font-medium text-accent-primary">{resultCount} results</span>
      </div>

      <div className="p-4 space-y-4">
        {/* Game Selection */}
        {games.length > 1 && (
          <div>
            <button
              onClick={() => toggleSection('game')}
              className="flex items-center justify-between w-full mb-2"
            >
              <label className="text-xs text-text-secondary font-medium">Game</label>
              {expandedSections.game ? (
                <ChevronUp size={14} className="text-text-muted" />
              ) : (
                <ChevronDown size={14} className="text-text-muted" />
              )}
            </button>
            {expandedSections.game && (
              <select
                value={selectedGameId || ''}
                onChange={(e) => onGameSelect(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded-lg text-text-primary focus:outline-none focus:border-accent-primary"
              >
                {games.map((game) => {
                  const matchup = formatMatchup(
                    game.home_team.abbreviation,
                    game.visitor_team.abbreviation
                  );
                  return (
                    <option key={game.game_id} value={game.game_id}>
                      {matchup}
                      {game.status && game.status !== 'scheduled' ? ` - ${game.status}` : ''}
                    </option>
                  );
                })}
              </select>
            )}
          </div>
        )}

        {/* Team Filter */}
        {availableTeams.length > 0 && (
          <div>
            <button
              onClick={() => toggleSection('team')}
              className="flex items-center justify-between w-full mb-2"
            >
              <label className="text-xs text-text-secondary font-medium">
                Team {filters.teams && filters.teams.length > 0 && `(${filters.teams.length})`}
              </label>
              {expandedSections.team ? (
                <ChevronUp size={14} className="text-text-muted" />
              ) : (
                <ChevronDown size={14} className="text-text-muted" />
              )}
            </button>
            {expandedSections.team && (
              <div className="flex flex-wrap gap-2">
                {availableTeams.map((team) => (
                  <button
                    key={team}
                    onClick={() => handleTeamToggle(team)}
                    className={`
                      px-3 py-1.5 text-xs font-medium rounded transition-colors
                      ${
                        filters.teams?.includes(team)
                          ? 'bg-accent-primary text-white shadow-sm'
                          : 'bg-bg-tertiary text-text-secondary hover:bg-bg-hover'
                      }
                    `}
                  >
                    {team}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Position Filter */}
        {availablePositions.length > 0 && (
          <div>
            <button
              onClick={() => toggleSection('position')}
              className="flex items-center justify-between w-full mb-2"
            >
              <label className="text-xs text-text-secondary font-medium">
                Position {filters.positions && filters.positions.length > 0 && `(${filters.positions.length})`}
              </label>
              {expandedSections.position ? (
                <ChevronUp size={14} className="text-text-muted" />
              ) : (
                <ChevronDown size={14} className="text-text-muted" />
              )}
            </button>
            {expandedSections.position && (
              <div className="flex flex-wrap gap-2">
                {POSITIONS.map((position) => (
                  <button
                    key={position}
                    onClick={() => handlePositionToggle(position)}
                    disabled={!availablePositions.includes(position)}
                    className={`
                      px-3 py-1.5 text-xs font-medium rounded transition-colors
                      ${
                        filters.positions?.includes(position)
                          ? 'bg-accent-primary text-white shadow-sm'
                          : availablePositions.includes(position)
                          ? 'bg-bg-tertiary text-text-secondary hover:bg-bg-hover'
                          : 'bg-bg-tertiary text-text-muted opacity-40 cursor-not-allowed'
                      }
                    `}
                  >
                    {position}
                  </button>
                ))}
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
            <div className="flex items-center gap-1.5">
              <label className="text-xs text-text-secondary font-medium">Confidence</label>
              <Tooltip content={<ConfidenceExplanation />} side="right">
                <Info size={12} className="text-text-muted hover:text-accent-primary transition-colors" />
              </Tooltip>
            </div>
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
