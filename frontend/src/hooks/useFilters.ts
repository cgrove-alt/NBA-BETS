import { useState, useCallback } from 'react';
import { PROP_TYPES } from '../lib/types';
import type { FilterState } from '../lib/types';

const defaultFilters: FilterState = {
  minConfidence: 55,  // Match backend default
  minEdge: 4,         // Match backend default
  propTypes: [...PROP_TYPES],
  pickType: null,
  sortBy: 'quality',  // Match backend default (quality = confidence * edge)
  sortOrder: 'desc',
};

export function useFilters(initialFilters?: Partial<FilterState>) {
  const [filters, setFilters] = useState<FilterState>({
    ...defaultFilters,
    ...initialFilters,
  });

  const updateFilters = useCallback((updates: Partial<FilterState>) => {
    setFilters((prev) => ({ ...prev, ...updates }));
  }, []);

  const resetFilters = useCallback(() => {
    setFilters(defaultFilters);
  }, []);

  return {
    filters,
    updateFilters,
    resetFilters,
  };
}
