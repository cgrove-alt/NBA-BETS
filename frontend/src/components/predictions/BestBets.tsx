import { Star } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { ConfidenceBar } from './ConfidenceBar';
import { cn, formatPrediction, formatLine, formatEdge, getPickColor } from '../../lib/utils';
import type { PlayerProp, PropPrediction, PropType } from '../../lib/types';

interface BestBetsProps {
  players: PlayerProp[];
}

interface BestBetItem {
  player: PlayerProp;
  propType: PropType;
  prop: PropPrediction;
}

export function BestBets({ players }: BestBetsProps) {
  // Find all best bets (confidence >= 80, edge >= 2.5)
  const bestBets: BestBetItem[] = [];

  const propTypes: PropType[] = ['Points', 'Rebounds', 'Assists', '3PM', 'PRA'];

  for (const player of players) {
    for (const propType of propTypes) {
      const prop = propType === '3PM'
        ? player['3PM']
        : player[propType as keyof PlayerProp] as PropPrediction | undefined;

      if (prop && prop.pick !== '-' && prop.confidence >= 80 && Math.abs(prop.edge) >= 2.5) {
        bestBets.push({ player, propType, prop });
      }
    }
  }

  // Sort by confidence, then by edge
  bestBets.sort((a, b) => {
    if (b.prop.confidence !== a.prop.confidence) {
      return b.prop.confidence - a.prop.confidence;
    }
    return Math.abs(b.prop.edge) - Math.abs(a.prop.edge);
  });

  if (bestBets.length === 0) {
    return null;
  }

  return (
    <Card className="border-accent-primary/30">
      <CardHeader className="bg-primary-light">
        <div className="flex items-center gap-2">
          <Star className="text-accent-primary" size={18} />
          <CardTitle>Best Bets</CardTitle>
          <Badge variant="primary">{bestBets.length}</Badge>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="divide-y divide-border">
          {bestBets.slice(0, 10).map(({ player, propType, prop }, index) => (
            <div
              key={`${player.player_id}-${propType}-${index}`}
              className="px-4 py-3 hover:bg-bg-hover transition-colors"
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-text-primary">
                      {player.player_name}
                    </span>
                    <span className="text-xs text-text-muted">{player.team}</span>
                  </div>
                  <div className="flex items-center gap-3 mt-1">
                    <Badge variant="default">{propType}</Badge>
                    <span className="text-sm text-text-secondary">
                      Line: {formatLine(prop.line)}
                    </span>
                    <span className="text-sm font-medium text-text-primary">
                      Pred: {formatPrediction(prop.prediction)}
                    </span>
                    <span
                      className={cn(
                        'text-sm font-medium',
                        prop.edge > 0 ? 'text-accent-success' : 'text-accent-danger'
                      )}
                    >
                      Edge: {formatEdge(prop.edge)}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <span
                    className={cn(
                      'text-lg font-bold px-3 py-1 rounded',
                      getPickColor(prop.pick),
                      prop.pick === 'OVER' ? 'bg-success-light' : 'bg-danger-light'
                    )}
                  >
                    {prop.pick}
                  </span>
                  <div className="w-24">
                    <ConfidenceBar confidence={prop.confidence} size="sm" />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
