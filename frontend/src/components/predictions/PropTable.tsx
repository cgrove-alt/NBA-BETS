import { useState, useMemo } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { ConfidenceBar } from './ConfidenceBar';
import { EdgeBadge } from './EdgeBadge';
import { cn, formatPrediction, formatLine, getPickColor, getPickBgClass } from '../../lib/utils';
import type { PlayerProp, PropPrediction, PropType, FilterState } from '../../lib/types';

interface PropTableProps {
  propType: PropType;
  players: PlayerProp[];
  filters: FilterState;
}

type SortField = 'player' | 'team' | 'line' | 'prediction' | 'edge' | 'pick' | 'confidence';

export function PropTable({ propType, players, filters }: PropTableProps) {
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
