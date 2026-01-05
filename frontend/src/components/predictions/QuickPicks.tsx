import { Flame, Zap, Target, TrendingUp } from 'lucide-react';
import { cn } from '../../lib/utils';
import type { PlayerProp, PropPrediction, PropType } from '../../lib/types';

interface QuickPicksProps {
  players: PlayerProp[];
  onScrollToBestBets?: () => void;
}

interface PickStats {
  totalPicks: number;
  firePicks: number;      // 90%+ confidence
  strongPicks: number;    // 80-89% confidence
  goodPicks: number;      // 70-79% confidence
  overPicks: number;
  underPicks: number;
  avgConfidence: number;
  avgEdge: number;
}

function calculateStats(players: PlayerProp[]): PickStats {
  let totalPicks = 0;
  let firePicks = 0;
  let strongPicks = 0;
  let goodPicks = 0;
  let overPicks = 0;
  let underPicks = 0;
  let totalConfidence = 0;
  let totalEdge = 0;

  const propTypes: PropType[] = ['Points', 'Rebounds', 'Assists', '3PM', 'PRA'];

  for (const player of players) {
    for (const propType of propTypes) {
      const prop = propType === '3PM'
        ? player['3PM']
        : player[propType as keyof PlayerProp] as PropPrediction | undefined;

      if (prop && prop.pick !== '-' && prop.line && prop.line > 0) {
        totalPicks++;
        totalConfidence += prop.confidence;
        totalEdge += Math.abs(prop.edge);

        if (prop.pick === 'OVER') overPicks++;
        if (prop.pick === 'UNDER') underPicks++;

        // Only count as "best bet" if edge >= 2.5
        if (Math.abs(prop.edge) >= 2.5) {
          if (prop.confidence >= 90) firePicks++;
          else if (prop.confidence >= 80) strongPicks++;
          else if (prop.confidence >= 70) goodPicks++;
        }
      }
    }
  }

  return {
    totalPicks,
    firePicks,
    strongPicks,
    goodPicks,
    overPicks,
    underPicks,
    avgConfidence: totalPicks > 0 ? totalConfidence / totalPicks : 0,
    avgEdge: totalPicks > 0 ? totalEdge / totalPicks : 0,
  };
}

export function QuickPicks({ players, onScrollToBestBets }: QuickPicksProps) {
  const stats = calculateStats(players);
  const bestBets = stats.firePicks + stats.strongPicks;

  if (stats.totalPicks === 0) {
    return null;
  }

  return (
    <div className="bg-gradient-to-r from-bg-card via-bg-card to-bg-card border border-border rounded-lg overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-4 px-4 py-3">
        {/* Left side - Main stats */}
        <div className="flex items-center gap-6">
          {/* Best Bets (clickable if handler provided) */}
          <button
            onClick={onScrollToBestBets}
            disabled={!onScrollToBestBets || bestBets === 0}
            className={cn(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors',
              bestBets > 0
                ? 'bg-orange-500/20 hover:bg-orange-500/30 cursor-pointer'
                : 'bg-gray-500/10 cursor-default'
            )}
          >
            <Flame className={cn(
              'w-5 h-5',
              bestBets > 0 ? 'text-orange-500' : 'text-gray-500'
            )} />
            <div className="text-left">
              <div className={cn(
                'text-lg font-bold leading-none',
                bestBets > 0 ? 'text-orange-400' : 'text-gray-400'
              )}>
                {bestBets}
              </div>
              <div className="text-xs text-text-muted">Best Bets</div>
            </div>
          </button>

          {/* Fire picks breakdown */}
          {stats.firePicks > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-lg">🔥</span>
              <div className="text-left">
                <div className="text-lg font-bold text-orange-400 leading-none">
                  {stats.firePicks}
                </div>
                <div className="text-xs text-text-muted">Fire (90%+)</div>
              </div>
            </div>
          )}

          {/* Total plays */}
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-blue-400" />
            <div className="text-left">
              <div className="text-lg font-bold text-text-primary leading-none">
                {stats.totalPicks}
              </div>
              <div className="text-xs text-text-muted">Total Plays</div>
            </div>
          </div>
        </div>

        {/* Right side - Quick stats */}
        <div className="flex items-center gap-4">
          {/* Over/Under split */}
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-green-400" />
            <span className="text-sm">
              <span className="font-medium text-green-400">{stats.overPicks}</span>
              <span className="text-text-muted"> / </span>
              <span className="font-medium text-red-400">{stats.underPicks}</span>
            </span>
            <span className="text-xs text-text-muted">O/U</span>
          </div>

          {/* Average confidence */}
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4 text-text-muted" />
            <span className="text-sm">
              <span className="font-medium text-text-primary">
                {stats.avgConfidence.toFixed(0)}%
              </span>
            </span>
            <span className="text-xs text-text-muted">Avg Conf</span>
          </div>

          {/* Average edge */}
          <div className="hidden sm:flex items-center gap-2">
            <span className="text-sm">
              <span className="font-medium text-green-400">
                +{stats.avgEdge.toFixed(1)}
              </span>
            </span>
            <span className="text-xs text-text-muted">Avg Edge</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Compact version for smaller spaces
export function QuickPicksCompact({ players }: { players: PlayerProp[] }) {
  const stats = calculateStats(players);
  const bestBets = stats.firePicks + stats.strongPicks;

  return (
    <div className="flex items-center gap-4 text-sm">
      <span className="flex items-center gap-1">
        <Flame className={cn(
          'w-4 h-4',
          bestBets > 0 ? 'text-orange-500' : 'text-gray-500'
        )} />
        <span className={cn(
          'font-bold',
          bestBets > 0 ? 'text-orange-400' : 'text-gray-400'
        )}>
          {bestBets}
        </span>
        <span className="text-text-muted">Best</span>
      </span>
      <span className="text-text-muted">|</span>
      <span className="flex items-center gap-1">
        <Zap className="w-4 h-4 text-blue-400" />
        <span className="font-bold text-text-primary">{stats.totalPicks}</span>
        <span className="text-text-muted">Total</span>
      </span>
    </div>
  );
}
