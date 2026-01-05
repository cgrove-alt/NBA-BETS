import { Flame, Star } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { BetCard } from './BetCard';
import type { PlayerProp, PropPrediction, PropType } from '../../lib/types';

interface BestBetsProps {
  players: PlayerProp[];
  gameContext?: string; // e.g., "NYK @ DET"
}

interface BestBetItem {
  player: PlayerProp;
  propType: PropType;
  prop: PropPrediction;
}

export function BestBets({ players, gameContext }: BestBetsProps) {
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

  // Sort by confidence (highest first), then by edge
  bestBets.sort((a, b) => {
    if (b.prop.confidence !== a.prop.confidence) {
      return b.prop.confidence - a.prop.confidence;
    }
    return Math.abs(b.prop.edge) - Math.abs(a.prop.edge);
  });

  // Separate fire picks (90%+) from strong picks
  const firePicks = bestBets.filter(b => b.prop.confidence >= 90);
  const strongPicks = bestBets.filter(b => b.prop.confidence >= 80 && b.prop.confidence < 90);

  if (bestBets.length === 0) {
    return null;
  }

  return (
    <Card className="border-accent-primary/30">
      <CardHeader className="bg-gradient-to-r from-orange-500/10 to-yellow-500/10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Flame className="text-orange-500" size={20} />
            <CardTitle>Best Bets</CardTitle>
            <Badge variant="primary">{bestBets.length}</Badge>
          </div>
          <div className="flex items-center gap-2 text-sm text-text-muted">
            {firePicks.length > 0 && (
              <span className="flex items-center gap-1">
                <span className="text-orange-400">🔥</span>
                <span>{firePicks.length} Fire</span>
              </span>
            )}
            {strongPicks.length > 0 && (
              <span className="flex items-center gap-1">
                <Star className="text-yellow-400" size={14} />
                <span>{strongPicks.length} Strong</span>
              </span>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4">
        {/* Fire Picks Section (90%+) */}
        {firePicks.length > 0 && (
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-lg">🔥</span>
              <h4 className="font-bold text-text-primary">Fire Picks</h4>
              <span className="text-xs text-text-muted bg-orange-500/20 px-2 py-0.5 rounded-full">
                90%+ Confidence
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {firePicks.slice(0, 6).map(({ player, propType, prop }, index) => (
                <BetCard
                  key={`${player.player_id}-${propType}-${index}`}
                  player={player}
                  propType={propType}
                  prop={prop}
                  gameContext={gameContext}
                />
              ))}
            </div>
          </div>
        )}

        {/* Strong Picks Section (80-89%) */}
        {strongPicks.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Star className="text-yellow-400" size={18} />
              <h4 className="font-bold text-text-primary">Strong Picks</h4>
              <span className="text-xs text-text-muted bg-yellow-500/20 px-2 py-0.5 rounded-full">
                80-89% Confidence
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {strongPicks.slice(0, 9).map(({ player, propType, prop }, index) => (
                <BetCard
                  key={`${player.player_id}-${propType}-${index}`}
                  player={player}
                  propType={propType}
                  prop={prop}
                  gameContext={gameContext}
                />
              ))}
            </div>
          </div>
        )}

        {/* Show more hint if there are more picks */}
        {bestBets.length > 15 && (
          <div className="mt-4 text-center text-sm text-text-muted">
            Showing top {Math.min(15, bestBets.length)} of {bestBets.length} best bets
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Summary stats component for the Best Bets section
export function BestBetsSummary({ players }: { players: PlayerProp[] }) {
  let totalPicks = 0;
  let firePicks = 0;
  let strongPicks = 0;

  const propTypes: PropType[] = ['Points', 'Rebounds', 'Assists', '3PM', 'PRA'];

  for (const player of players) {
    for (const propType of propTypes) {
      const prop = propType === '3PM'
        ? player['3PM']
        : player[propType as keyof PlayerProp] as PropPrediction | undefined;

      if (prop && prop.pick !== '-') {
        totalPicks++;
        if (prop.confidence >= 90 && Math.abs(prop.edge) >= 2.5) {
          firePicks++;
        } else if (prop.confidence >= 80 && Math.abs(prop.edge) >= 2.5) {
          strongPicks++;
        }
      }
    }
  }

  return {
    totalPicks,
    firePicks,
    strongPicks,
    bestBets: firePicks + strongPicks,
  };
}
