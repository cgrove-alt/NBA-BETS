// Utility functions for NBA Props Dashboard

import type { PropPrediction } from './types';

// Color thresholds
const CONFIDENCE_HIGH = 70;
const CONFIDENCE_MEDIUM = 50;

// Get confidence color class
export function getConfidenceColor(confidence: number): string {
  if (confidence >= CONFIDENCE_HIGH) return 'text-accent-success';
  if (confidence >= CONFIDENCE_MEDIUM) return 'text-accent-warning';
  return 'text-text-muted';
}

// Get confidence background class
export function getConfidenceBgClass(confidence: number): string {
  if (confidence >= CONFIDENCE_HIGH) return 'bg-success-light';
  if (confidence >= CONFIDENCE_MEDIUM) return 'bg-warning-light';
  return 'bg-bg-tertiary';
}

// Get confidence gradient class
export function getConfidenceGradient(confidence: number): string {
  if (confidence >= CONFIDENCE_HIGH) return 'gradient-success';
  if (confidence >= CONFIDENCE_MEDIUM) return 'gradient-warning';
  return 'gradient-neutral';
}

// Get pick color class
export function getPickColor(pick: string): string {
  if (pick === 'OVER') return 'text-accent-success';
  if (pick === 'UNDER') return 'text-accent-danger';
  return 'text-text-muted';
}

// Get pick background class
export function getPickBgClass(pick: string): string {
  if (pick === 'OVER') return 'bg-success-light';
  if (pick === 'UNDER') return 'bg-danger-light';
  return 'transparent';
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

// Format line
export function formatLine(line: number | undefined): string {
  if (line === undefined) return '-';
  return line.toFixed(1);
}

// Check if prop has strong edge
export function hasStrongEdge(prop: PropPrediction | undefined): boolean {
  if (!prop) return false;
  return Math.abs(prop.edge) >= 2.5;
}

// Check if prop is a best bet
export function isBestBet(prop: PropPrediction | undefined): boolean {
  if (!prop) return false;
  return prop.confidence >= 80 && Math.abs(prop.edge) >= 2.5;
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
