// Utility functions for NBA Props Dashboard

import type { PropPrediction } from './types';

// Color thresholds - Adjusted for model's natural confidence range (50-70%)
const CONFIDENCE_FIRE = 70;    // Fire picks (top tier for this model)
const CONFIDENCE_STRONG = 65;  // Strong picks
const CONFIDENCE_GOOD = 60;    // Good picks
const CONFIDENCE_MODERATE = 55; // Moderate picks

// Get confidence color class - Enhanced with more tiers
export function getConfidenceColor(confidence: number): string {
  if (confidence >= CONFIDENCE_FIRE) return 'text-orange-400';
  if (confidence >= CONFIDENCE_STRONG) return 'text-yellow-400';
  if (confidence >= CONFIDENCE_GOOD) return 'text-green-400';
  if (confidence >= CONFIDENCE_MODERATE) return 'text-gray-400';
  return 'text-text-muted';
}

// Get confidence background class - Enhanced
export function getConfidenceBgClass(confidence: number): string {
  if (confidence >= CONFIDENCE_FIRE) return 'bg-orange-500/20';
  if (confidence >= CONFIDENCE_STRONG) return 'bg-yellow-500/20';
  if (confidence >= CONFIDENCE_GOOD) return 'bg-green-500/20';
  if (confidence >= CONFIDENCE_MODERATE) return 'bg-gray-500/20';
  return 'bg-bg-tertiary';
}

// Get confidence gradient class - Enhanced
export function getConfidenceGradient(confidence: number): string {
  if (confidence >= CONFIDENCE_FIRE) return 'bg-gradient-to-r from-orange-500 to-red-500';
  if (confidence >= CONFIDENCE_STRONG) return 'bg-gradient-to-r from-yellow-500 to-amber-500';
  if (confidence >= CONFIDENCE_GOOD) return 'bg-gradient-to-r from-green-500 to-emerald-500';
  return 'bg-gradient-to-r from-gray-500 to-gray-600';
}

// Get pick color class - Enhanced with brighter colors
export function getPickColor(pick: string): string {
  if (pick === 'OVER') return 'text-green-400';
  if (pick === 'UNDER') return 'text-red-400';
  return 'text-text-muted';
}

// Get pick background class - Enhanced
export function getPickBgClass(pick: string): string {
  if (pick === 'OVER') return 'bg-green-500/20';
  if (pick === 'UNDER') return 'bg-red-500/20';
  return 'transparent';
}

// Get pick border class - New helper
export function getPickBorderClass(pick: string): string {
  if (pick === 'OVER') return 'border-green-500/40';
  if (pick === 'UNDER') return 'border-red-500/40';
  return 'border-gray-500/20';
}

// Get edge color class
export function getEdgeColor(edge: number): string {
  if (edge >= 2.5) return 'text-accent-success';
  if (edge <= -2.5) return 'text-accent-danger';
  if (edge > 0) return 'text-accent-success/70';
  if (edge < 0) return 'text-accent-danger/70';
  return 'text-text-muted';
}

// Format edge with sign
export function formatEdge(edge: number): string {
  if (edge === 0) return '0.0';
  const sign = edge > 0 ? '+' : '';
  return `${sign}${edge.toFixed(1)}`;
}

// Format edge percentage
export function formatEdgePct(edgePct: number): string {
  if (edgePct === 0) return '0.0%';
  const sign = edgePct > 0 ? '+' : '';
  return `${sign}${edgePct.toFixed(1)}%`;
}

// Format confidence
export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence)}%`;
}

// Format prediction
export function formatPrediction(prediction: number): string {
  return prediction.toFixed(1);
}

// Format line - returns "-" for missing, null, or invalid lines
export function formatLine(line: number | undefined | null): string {
  if (line === undefined || line === null || line <= 0) return '-';
  return line.toFixed(1);
}

// Check if prop has strong edge
export function hasStrongEdge(prop: PropPrediction | undefined): boolean {
  if (!prop) return false;
  return Math.abs(prop.edge) >= 2.5;
}

// Check if prop is a best bet
// Thresholds lowered to match model's natural confidence range (50-70%)
export function isBestBet(prop: PropPrediction | undefined): boolean {
  if (!prop) return false;
  return prop.confidence >= 65 && Math.abs(prop.edge) >= 2.0;
}

// Format game time
export function formatGameTime(gameTime: string | undefined): string {
  if (!gameTime) return '';
  try {
    const date = new Date(gameTime);
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  } catch {
    return gameTime;
  }
}

// Format matchup string
export function formatMatchup(homeAbbrev: string, awayAbbrev: string): string {
  return `${awayAbbrev} @ ${homeAbbrev}`;
}

// Class name helper (like clsx but simpler)
export function cn(...classes: (string | boolean | undefined)[]): string {
  return classes.filter(Boolean).join(' ');
}
