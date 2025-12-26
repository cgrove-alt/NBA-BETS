import { useState, useMemo } from 'react';
import { ChevronUp, ChevronDown, CheckCircle, XCircle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { ConfidenceBar } from './ConfidenceBar';
import { EdgeBadge } from './EdgeBadge';
import { cn, formatPrediction, formatLine, getPickColor, getPickBgClass } from '../../lib/utils';
import type { PlayerProp, PropPrediction, PropType, FilterState, PlayerLiveStats } from '../../lib/types';

interface PropTableProps {
  propType: PropType;
  players: PlayerProp[];
  filters: FilterState;
  liveStats?: Record<number, PlayerLiveStats>;
  isLive?: boolean;
  isFinal?: boolean;
}

type SortField = 'player' | 'team' | 'line' | 'prediction' | 'edge' | 'pick' | 'confidence';

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

  // Filter and sort players
  const filteredPlayers = useMemo(() => {
    return players
      .map((player) => ({ player, prop: getProp(player) }))
      .filter(({ prop }) => {
        if (!prop || prop.pick === '-') return false;
        if (prop.confidence < filters.minConfidence) return false;
        if (Math.abs(prop.edge) < filters.minEdge) return false;
        if (filters.pickType && prop.pick !== filters.pickType) return false;
        return true;
      })
      .sort((a, b) => {
        const propA = a.prop!;
        const propB = b.prop!;
        let comparison = 0;

        switch (sortField) {
          case 'player':
            comparison = a.player.player_name.localeCompare(b.player.player_name);
            break;
          case 'team':
            comparison = (a.player.team || '').localeCompare(b.player.team || '');
            break;
          case 'line':
            comparison = (propA.line || 0) - (propB.line || 0);
            break;
          case 'prediction':
            comparison = propA.prediction - propB.prediction;
            break;
          case 'edge':
            comparison = Math.abs(propA.edge) - Math.abs(propB.edge);
            break;
          case 'pick':
            comparison = propA.pick.localeCompare(propB.pick);
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

  return (
    <Card>
      <CardHeader>
        <CardTitle>{propType}</CardTitle>
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
                  <th className={headerClass} onClick={() => handleSort('player')}>
                    Player <SortIcon field="player" />
                  </th>
                  <th className={headerClass} onClick={() => handleSort('team')}>
                    Team <SortIcon field="team" />
                  </th>
                  <th className={headerClass} onClick={() => handleSort('line')}>
                    Line <SortIcon field="line" />
                  </th>
                  <th className={headerClass} onClick={() => handleSort('prediction')}>
                    Prediction <SortIcon field="prediction" />
                  </th>
                  <th className={headerClass} onClick={() => handleSort('edge')}>
                    Edge <SortIcon field="edge" />
                  </th>
                  <th className={headerClass} onClick={() => handleSort('pick')}>
                    Pick <SortIcon field="pick" />
                  </th>
                  {(isLive || isFinal || liveStats) && (
                    <th className={headerClass}>
                      Actual {isLive && <span className="text-blue-400 text-[10px] ml-1">LIVE</span>}
                    </th>
                  )}
                  <th className={cn(headerClass, 'w-32')} onClick={() => handleSort('confidence')}>
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
                    <td className="px-3 py-3 text-sm text-text-primary font-medium">
                      {player.player_name}
                    </td>
                    <td className="px-3 py-3 text-sm text-text-secondary">
                      {player.team || '-'}
                    </td>
                    <td className="px-3 py-3 text-sm text-text-secondary">
                      {formatLine(prop?.line)}
                    </td>
                    <td className="px-3 py-3 text-sm text-text-primary font-medium">
                      {formatPrediction(prop!.prediction)}
                    </td>
                    <td className="px-3 py-3">
                      <EdgeBadge prop={prop!} />
                    </td>
                    <td className="px-3 py-3">
                      <span
                        className={cn(
                          'inline-flex items-center px-2 py-0.5 rounded text-xs font-bold',
                          getPickColor(prop!.pick),
                          getPickBgClass(prop!.pick)
                        )}
                      >
                        {prop!.pick}
                      </span>
                    </td>
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
                    <td className="px-3 py-3 w-32">
                      <ConfidenceBar confidence={prop!.confidence} size="sm" />
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
