import { X, RotateCcw } from 'lucide-react';
import type { FilterState } from '../../lib/types';

interface ActiveFiltersBarProps {
  filters: FilterState;
  onRemoveFilter: (filterKey: keyof FilterState, value?: string) => void;
  onResetAll: () => void;
  totalCount: number;
  filteredCount: number;
}

export function ActiveFiltersBar({
  filters,
  onRemoveFilter,
  onResetAll,
  totalCount,
  filteredCount,
}: ActiveFiltersBarProps) {
  const activeFilters: Array<{
    key: keyof FilterState;
    label: string;
    value?: string;
  }> = [];

  // Confidence filter
  if (filters.minConfidence > 55 || filters.maxConfidence) {
    const label = filters.maxConfidence
      ? `Confidence: ${filters.minConfidence}%-${filters.maxConfidence}%`
      : `Confidence ≥ ${filters.minConfidence}%`;
    activeFilters.push({ key: 'minConfidence', label });
  }

  // Edge filter
  if (filters.minEdge > 4 || filters.maxEdge) {
    const unit = filters.edgeMode === 'percentage' ? '%' : ' pts';
    const label = filters.maxEdge
      ? `Edge: ${filters.minEdge}${unit}-${filters.maxEdge}${unit}`
      : `Edge ≥ ${filters.minEdge}${unit}`;
    activeFilters.push({ key: 'minEdge', label });
  }

  // Pick type filter
  if (filters.pickType) {
    activeFilters.push({
      key: 'pickType',
      label: `${filters.pickType} only`,
    });
  }

  // Prop types filter (if not all selected)
  if (filters.propTypes.length < 5) {
    filters.propTypes.forEach((propType) => {
      activeFilters.push({
        key: 'propTypes',
        label: propType,
        value: propType,
      });
    });
  }

  // Don't render if no active filters
  if (activeFilters.length === 0) {
    return null;
  }

  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-text-secondary">Active Filters</span>
          <span className="text-xs text-text-muted">
            {filteredCount} of {totalCount} results
          </span>
        </div>
        <button
          onClick={onResetAll}
          className="flex items-center gap-1 px-2 py-1 text-xs text-text-muted hover:text-accent-primary transition-colors"
        >
          <RotateCcw size={12} />
          Reset All
        </button>
      </div>

      <div className="flex flex-wrap gap-2">
        {activeFilters.map((filter, index) => (
          <div
            key={`${filter.key}-${filter.value || index}`}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-accent-primary/10 border border-accent-primary/20 rounded-full text-xs text-accent-primary"
          >
            <span>{filter.label}</span>
            <button
              onClick={() => onRemoveFilter(filter.key, filter.value)}
              className="hover:bg-accent-primary/20 rounded-full p-0.5 transition-colors"
              aria-label={`Remove ${filter.label} filter`}
            >
              <X size={12} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
