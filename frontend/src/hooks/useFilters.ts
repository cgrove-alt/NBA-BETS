import { useState, useCallback, useEffect } from 'react';
import { PROP_TYPES } from '../lib/types';
import type { FilterState, FilterPreset } from '../lib/types';

const STORAGE_KEY = 'nba-props-filters';
const PRESETS_KEY = 'nba-props-filter-presets';

const defaultFilters: FilterState = {
  minConfidence: 55,  // Match backend default
  minEdge: 4,         // Match backend default
  maxConfidence: undefined,
  maxEdge: undefined,
  propTypes: [...PROP_TYPES],
  pickType: null,
  sortBy: 'quality',  // Match backend default (quality = confidence * edge)
  sortOrder: 'desc',
  edgeMode: 'points', // Default to points display
};

// Load filters from localStorage
function loadFilters(): FilterState {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return { ...defaultFilters, ...parsed };
    }
  } catch (error) {
    console.warn('Failed to load filters from localStorage:', error);
  }
  return defaultFilters;
}

// Save filters to localStorage
function saveFilters(filters: FilterState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filters));
  } catch (error) {
    console.warn('Failed to save filters to localStorage:', error);
  }
}

// Load presets from localStorage
function loadPresets(): FilterPreset[] {
  try {
    const stored = localStorage.getItem(PRESETS_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (error) {
    console.warn('Failed to load presets from localStorage:', error);
  }
  return [];
}

// Save presets to localStorage
function savePresets(presets: FilterPreset[]): void {
  try {
    localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
  } catch (error) {
    console.warn('Failed to save presets to localStorage:', error);
  }
}

export function useFilters(initialFilters?: Partial<FilterState>) {
  const [filters, setFilters] = useState<FilterState>(() => ({
    ...loadFilters(),
    ...initialFilters,
  }));
  const [presets, setPresets] = useState<FilterPreset[]>(loadPresets);

  // Save filters to localStorage whenever they change
  useEffect(() => {
    saveFilters(filters);
  }, [filters]);

  const updateFilters = useCallback((updates: Partial<FilterState>) => {
    setFilters((prev) => ({ ...prev, ...updates }));
  }, []);

  const resetFilters = useCallback(() => {
    setFilters(defaultFilters);
  }, []);

  // Preset management
  const savePreset = useCallback((name: string, description?: string) => {
    const newPreset: FilterPreset = {
      id: `preset-${Date.now()}`,
      name,
      description,
      filters: { ...filters },
      createdAt: new Date().toISOString(),
    };
    const updated = [...presets, newPreset];
    setPresets(updated);
    savePresets(updated);
    return newPreset;
  }, [filters, presets]);

  const loadPreset = useCallback((presetId: string) => {
    const preset = presets.find((p) => p.id === presetId);
    if (preset) {
      setFilters(preset.filters);
    }
  }, [presets]);

  const deletePreset = useCallback((presetId: string) => {
    const updated = presets.filter((p) => p.id !== presetId);
    setPresets(updated);
    savePresets(updated);
  }, [presets]);

  const updatePreset = useCallback((presetId: string, updates: Partial<Omit<FilterPreset, 'id' | 'createdAt'>>) => {
    const updated = presets.map((p) =>
      p.id === presetId ? { ...p, ...updates } : p
    );
    setPresets(updated);
    savePresets(updated);
  }, [presets]);

  return {
    filters,
    updateFilters,
    resetFilters,
    presets,
    savePreset,
    loadPreset,
    deletePreset,
    updatePreset,
  };
}
