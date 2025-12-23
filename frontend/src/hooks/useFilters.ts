import { useState, useCallback } from 'react';
import { PROP_TYPES } from '../lib/types';
import type { FilterState } from '../lib/types';

const defaultFilters: FilterState = {
  minConfidence: 0,
  minEdge: 0,
  propTypes: [...PROP_TYPES],
  pickType: null,
  sortBy: 'confidence',
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
