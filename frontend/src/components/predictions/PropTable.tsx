import { useState, useMemo } from 'react';
import { ChevronUp, ChevronDown, CheckCircle, XCircle, ArrowUp, ArrowDown } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { ConfidenceBadge } from '../ui/ConfidenceTier';
import { cn, formatPrediction } from '../../lib/utils';
import type { PlayerProp, PropPrediction, PropType, FilterState, PlayerLiveStats } from '../../lib/types';

interface PropTableProps {
  propType: PropType;
  players: PlayerProp[];
  filters: FilterState;
  liveStats?: Record<number, PlayerLiveStats>;
  isLive?: boolean;
  isFinal?: boolean;
}

type SortField = 'bet' | 'player' | 'prediction' | 'confidence';

/**
 * Get the actual stat value from live stats for a given prop type
 */
function getActualStat(stats: PlayerLiveStats | undefined, propType: PropType): number | undefined {
  if (!stats) return undefined;
  switch (propType) {
    case 'Points': return stats.pts;
    case 'Rebounds': return stats.reb;
    case 'Assists': return stats.ast;
    case '3PM': return stats.fg3m;
    case 'PRA': return stats.pra;
    default: return undefined;
  }
}

/**
 * Determine if a pick is currently winning/won
 */
function isPickWinning(pick: string, actual: number, line: number | null | undefined): boolean | null {
  if (line === null || line === undefined) return null;
  if (pick === 'OVER') return actual > line;
  if (pick === 'UNDER') return actual < line;
  return null;
}

/**
 * Status indicator component for actual stats
 */
function ActualStatDisplay({
  actual,
  pick,
  line,
  isLive,
  isFinal,
}: {
  actual: number | undefined;
  pick: string;
  line: number | null | undefined;
  isLive: boolean;
  isFinal: boolean;
}) {
  // No stats yet (game hasn't started or no data)
  if (actual === undefined) {
    return <span className="text-text-muted">-</span>;
  }

  const winning = isPickWinning(pick, actual, line);

  // Live game - show pulsing indicator
  if (isLive) {
    return (
      <span className={cn(
        'inline-flex items-center gap-1.5 font-medium',
        winning === true ? 'text-green-500' : winning === false ? 'text-red-500' : 'text-text-primary'
      )}>
        <span className="relative flex h-2 w-2">
          <span className={cn(
            'animate-ping absolute inline-flex h-full w-full rounded-full opacity-75',
            winning === true ? 'bg-green-400' : winning === false ? 'bg-red-400' : 'bg-blue-400'
          )} />
          <span className={cn(
            'relative inline-flex rounded-full h-2 w-2',
            winning === true ? 'bg-green-500' : winning === false ? 'bg-red-500' : 'bg-blue-500'
          )} />
        </span>
        {actual}
      </span>
    );
  }

  // Final game - show HIT/MISS indicator
  if (isFinal) {
    if (winning === true) {
      return (
        <span className="inline-flex items-center gap-1 text-green-500 font-medium">
          <CheckCircle size={14} />
          {actual}
        </span>
      );
    } else if (winning === false) {
      return (
        <span className="inline-flex items-center gap-1 text-red-500 font-medium">
          <XCircle size={14} />
          {actual}
        </span>
      );
    }
  }

  // No pick made or can't determine
  return <span className="text-text-primary font-medium">{actual}</span>;
}

/**
 * The Bet Action Box - prominently shows what bet to make
 */
function BetActionBox({ prop, propType }: { prop: PropPrediction; propType: PropType }) {
  const isOver = prop.pick === 'OVER';
  const hasLine = prop.line !== null && prop.line !== undefined && prop.line > 0;
  const propLabel = propType === '3PM' ? '3PM' : propType.slice(0, 3);

  return (
    <div className={cn(
      'flex flex-col items-center justify-center px-3 py-2 rounded-lg border min-w-[80px]',
      isOver
        ? 'bg-green-500/10 border-green-500/40'
        : 'bg-red-500/10 border-red-500/40'
    )}>
      {/* Pick direction with icon */}
      <div className={cn(
        'flex items-center gap-1 text-sm font-black',
        isOver ? 'text-green-400' : 'text-red-400'
      )}>
        {isOver ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
        {prop.pick}
      </div>
      {/* Line value */}
      <div className="text-base font-bold text-text-primary">
        {hasLine ? prop.line!.toFixed(1) : '—'}
      </div>
      {/* Prop type label */}
      <div className="text-[10px] text-text-muted uppercase">
        {propLabel}
      </div>
    </div>
  );
}

/**
 * Prediction vs Line comparison cell
 */
function PredictionCell({ prop }: { prop: PropPrediction }) {
  const hasLine = prop.line !== null && prop.line !== undefined && prop.line > 0;
  const edge = hasLine ? prop.prediction - prop.line! : 0;
  const isOver = edge > 0;

  return (
    <div className="flex flex-col">
      {/* Prediction value */}
      <span className={cn(
        'font-bold text-base',
        isOver ? 'text-green-400' : 'text-red-400'
      )}>
        {formatPrediction(prop.prediction)}
      </span>
      {/* Edge vs line */}
      {hasLine && (
        <span className={cn(
          'text-xs font-medium',
          isOver ? 'text-green-400/70' : 'text-red-400/70'
        )}>
          {isOver ? '+' : ''}{edge.toFixed(1)} vs {prop.line!.toFixed(1)}
        </span>
      )}
    </div>
  );
}

export function PropTable({ propType, players, filters, liveStats, isLive = false, isFinal = false }: PropTableProps) {
  const [sortField, setSortField] = useState<SortField>('confidence');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Get prop data for each player
  const getProp = (player: PlayerProp): PropPrediction | undefined => {
    if (propType === '3PM') {
      return player['3PM'];
    }
    return player[propType as keyof PlayerProp] as PropPrediction | undefined;
  };

  // Filter and sort players with enhanced filters
  const filteredPlayers = useMemo(() => {
    return players
      .map((player) => ({ player, prop: getProp(player) }))
      .filter(({ prop }) => {
        if (!prop || prop.pick === '-') return false;

        // Confidence filters
        if (prop.confidence < filters.minConfidence) return false;
        if (filters.maxConfidence && prop.confidence > filters.maxConfidence) return false;

        // Edge filters (support both points and percentage mode)
        const edgeValue = filters.edgeMode === 'percentage'
          ? prop.edge_pct || Math.abs(prop.edge)
          : Math.abs(prop.edge);
        if (edgeValue < filters.minEdge) return false;
        if (filters.maxEdge && edgeValue > filters.maxEdge) return false;

        // Pick type filter
        if (filters.pickType && prop.pick !== filters.pickType) return false;

        return true;
      })
      .sort((a, b) => {
        const propA = a.prop!;
        const propB = b.prop!;
        let comparison = 0;

        switch (sortField) {
          case 'bet':
            // Sort by pick type, then by line
            comparison = propA.pick.localeCompare(propB.pick);
            if (comparison === 0) {
              comparison = (propA.line || 0) - (propB.line || 0);
            }
            break;
          case 'player':
            comparison = a.player.player_name.localeCompare(b.player.player_name);
            break;
          case 'prediction':
            comparison = propA.prediction - propB.prediction;
            break;
          case 'confidence':
          default:
            comparison = propA.confidence - propB.confidence;
            break;
        }

        return sortOrder === 'desc' ? -comparison : comparison;
      });
  }, [players, propType, filters, sortField, sortOrder]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortOrder === 'desc' ? (
      <ChevronDown size={14} className="inline" />
    ) : (
      <ChevronUp size={14} className="inline" />
    );
  };

  const headerClass = 'px-3 py-2 text-left text-xs font-medium text-text-secondary uppercase tracking-wider cursor-pointer hover:text-text-primary';

  // Format prop type for display
  const propLabel = propType === '3PM' ? '3-Pointers Made' : propType;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>{propLabel}</CardTitle>
          <span className="text-sm text-text-muted">
            {filteredPlayers.length} picks
          </span>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {filteredPlayers.length === 0 ? (
          <div className="p-8 text-center text-text-muted">
            No props match the current filters
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-bg-tertiary">
                <tr>
                  <th className={headerClass} onClick={() => handleSort('bet')}>
                    Bet <SortIcon field="bet" />
                  </th>
                  <th className={headerClass} onClick={() => handleSort('player')}>
                    Player <SortIcon field="player" />
                  </th>
                  <th className={headerClass} onClick={() => handleSort('prediction')}>
                    Prediction <SortIcon field="prediction" />
                  </th>
                  {(isLive || isFinal || liveStats) && (
                    <th className={headerClass}>
                      Actual {isLive && <span className="text-blue-400 text-[10px] ml-1">LIVE</span>}
                    </th>
                  )}
                  <th className={cn(headerClass, 'text-center')} onClick={() => handleSort('confidence')}>
                    Confidence <SortIcon field="confidence" />
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {filteredPlayers.map(({ player, prop }) => (
                  <tr
                    key={`${player.player_id}-${propType}`}
                    className="hover:bg-bg-hover transition-colors"
                  >
                    {/* THE BET - Action first */}
                    <td className="px-3 py-3">
                      <BetActionBox prop={prop!} propType={propType} />
                    </td>
                    {/* Player info */}
                    <td className="px-3 py-3">
                      <div className="flex flex-col">
                        <span className="font-medium text-text-primary">
                          {player.player_name}
                        </span>
                        <span className="text-xs text-text-muted">{player.team || '-'}</span>
                      </div>
                    </td>
                    {/* Prediction vs Line */}
                    <td className="px-3 py-3">
                      <PredictionCell prop={prop!} />
                    </td>
                    {/* Actual (Live/Final) */}
                    {(isLive || isFinal || liveStats) && (
                      <td className="px-3 py-3 text-sm">
                        <ActualStatDisplay
                          actual={getActualStat(liveStats?.[player.player_id], propType)}
                          pick={prop!.pick}
                          line={prop?.line}
                          isLive={isLive}
                          isFinal={isFinal}
                        />
                      </td>
                    )}
                    {/* Confidence */}
                    <td className="px-3 py-3">
                      <div className="flex justify-center">
                        <ConfidenceBadge confidence={prop!.confidence} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
