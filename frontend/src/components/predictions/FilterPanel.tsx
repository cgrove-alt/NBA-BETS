import { PROP_TYPES } from '../../lib/types';
import type { FilterState, PropType } from '../../lib/types';

interface FilterPanelProps {
  filters: FilterState;
  onFilterChange: (filters: Partial<FilterState>) => void;
  resultCount: number;
}

export function FilterPanel({ filters, onFilterChange, resultCount }: FilterPanelProps) {
  const handleConfidenceChange = (value: number) => {
    onFilterChange({ minConfidence: value });
  };

  const handleEdgeChange = (value: number) => {
    onFilterChange({ minEdge: value });
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

  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-text-primary">Filters</h3>
        <span className="text-xs text-text-muted">{resultCount} results</span>
      </div>

      {/* Confidence slider */}
      <div>
        <label className="text-xs text-text-secondary block mb-2">
          Min Confidence: {filters.minConfidence}%
        </label>
        <input
          type="range"
          min="0"
          max="100"
          value={filters.minConfidence}
          onChange={(e) => handleConfidenceChange(Number(e.target.value))}
          className="w-full h-2 bg-bg-tertiary rounded-lg appearance-none cursor-pointer accent-accent-primary"
        />
      </div>

      {/* Edge slider */}
      <div>
        <label className="text-xs text-text-secondary block mb-2">
          Min Edge: {filters.minEdge.toFixed(1)}
        </label>
        <input
          type="range"
          min="0"
          max="10"
          step="0.5"
          value={filters.minEdge}
          onChange={(e) => handleEdgeChange(Number(e.target.value))}
          className="w-full h-2 bg-bg-tertiary rounded-lg appearance-none cursor-pointer accent-accent-primary"
        />
      </div>

      {/* Prop types */}
      <div>
        <label className="text-xs text-text-secondary block mb-2">Prop Types</label>
        <div className="flex flex-wrap gap-2">
          {PROP_TYPES.map((propType) => (
            <button
              key={propType}
              onClick={() => handlePropTypeToggle(propType)}
              className={`
                px-2 py-1 text-xs rounded transition-colors
                ${
                  filters.propTypes.includes(propType)
                    ? 'bg-accent-primary text-white'
                    : 'bg-bg-tertiary text-text-secondary hover:bg-bg-hover'
                }
              `}
            >
              {propType}
            </button>
          ))}
        </div>
      </div>

      {/* Pick type */}
      <div>
        <label className="text-xs text-text-secondary block mb-2">Pick Type</label>
        <div className="flex gap-2">
          <button
            onClick={() => handlePickTypeChange('OVER')}
            className={`
              flex-1 px-3 py-1.5 text-xs rounded transition-colors
              ${
                filters.pickType === 'OVER'
                  ? 'bg-accent-success text-white'
                  : 'bg-bg-tertiary text-text-secondary hover:bg-bg-hover'
              }
            `}
          >
            OVER
          </button>
          <button
            onClick={() => handlePickTypeChange('UNDER')}
            className={`
              flex-1 px-3 py-1.5 text-xs rounded transition-colors
              ${
                filters.pickType === 'UNDER'
                  ? 'bg-accent-danger text-white'
                  : 'bg-bg-tertiary text-text-secondary hover:bg-bg-hover'
              }
            `}
          >
            UNDER
          </button>
        </div>
      </div>
    </div>
  );
}
