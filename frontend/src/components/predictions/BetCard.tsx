import { ArrowUp, ArrowDown, TrendingUp } from 'lucide-react';
import { cn } from '../../lib/utils';
import { ConfidenceTier, getConfidenceLevel } from '../ui/ConfidenceTier';
import { LineComparison } from '../ui/LineComparison';
import type { PlayerProp, PropPrediction, PropType } from '../../lib/types';

interface BetCardProps {
  player: PlayerProp;
  propType: PropType;
  prop: PropPrediction;
  gameContext?: string; // e.g., "NYK @ DET"
}

export function BetCard({ player, propType, prop, gameContext }: BetCardProps) {
  const edge = prop.line ? prop.prediction - prop.line : 0;
  const isOver = prop.pick === 'OVER';
  const hasLine = prop.line !== null && prop.line !== undefined && prop.line > 0;

  const level = getConfidenceLevel(prop.confidence);

  // Format prop type for display
  const propLabel = propType === '3PM' ? '3-Pointers' : propType;

  return (
    <div
      className={cn(
        'relative bg-bg-card border rounded-xl overflow-hidden transition-all duration-200',
        'hover:shadow-lg hover:-translate-y-0.5',
        level === 'fire' && 'border-orange-500/50 shadow-[0_0_20px_rgba(249,115,22,0.15)]',
        level === 'strong' && 'border-yellow-500/50 shadow-[0_0_15px_rgba(234,179,8,0.1)]',
        level === 'good' && 'border-green-500/30',
        level === 'moderate' && 'border-gray-500/30',
        level === 'risky' && 'border-gray-600/30'
      )}
    >
      {/* Header with confidence and edge */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <ConfidenceTier confidence={prop.confidence} size="sm" />
        <div
          className={cn(
            'flex items-center gap-1 px-2 py-1 rounded-full text-sm font-bold',
            isOver ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
          )}
        >
          {hasLine && (
            <>
              {isOver ? <TrendingUp size={14} /> : <TrendingUp size={14} className="rotate-180" />}
              <span>{isOver ? '+' : ''}{edge.toFixed(1)} Edge</span>
            </>
          )}
        </div>
      </div>

      {/* Player info */}
      <div className="px-4 pt-3 pb-2">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-text-primary">{player.player_name}</h3>
            {gameContext && (
              <p className="text-sm text-text-muted">{gameContext}</p>
            )}
          </div>
          <span className="text-xl font-bold text-text-secondary">{player.team}</span>
        </div>
      </div>

      {/* THE BET - Main action box */}
      <div className="px-4 py-3">
        <div
          className={cn(
            'relative p-4 rounded-lg border-2 text-center',
            isOver
              ? 'bg-green-500/10 border-green-500/50'
              : 'bg-red-500/10 border-red-500/50'
          )}
        >
          {/* Arrow indicator */}
          <div className={cn(
            'absolute -top-3 left-1/2 -translate-x-1/2 w-6 h-6 rounded-full flex items-center justify-center',
            isOver ? 'bg-green-500' : 'bg-red-500'
          )}>
            {isOver ? (
              <ArrowUp size={16} className="text-white" />
            ) : (
              <ArrowDown size={16} className="text-white" />
            )}
          </div>

          <div className={cn(
            'text-2xl font-black tracking-wide',
            isOver ? 'text-green-400' : 'text-red-400'
          )}>
            {prop.pick}
          </div>
          <div className="text-xl font-bold text-text-primary mt-1">
            {hasLine ? prop.line!.toFixed(1) : '—'} {propLabel}
          </div>
        </div>
      </div>

      {/* Line comparison visualization */}
      {hasLine && (
        <div className="px-4 pb-4">
          <LineComparison
            line={prop.line}
            prediction={prop.prediction}
            propType={propLabel}
            size="sm"
          />
        </div>
      )}

      {/* No line fallback */}
      {!hasLine && (
        <div className="px-4 pb-4 text-center">
          <div className="text-sm text-text-muted">
            Model predicts{' '}
            <span className="font-bold text-text-primary">{prop.prediction.toFixed(1)} {propLabel.toLowerCase()}</span>
          </div>
          <div className="text-xs text-text-muted mt-1">(No sportsbook line available)</div>
        </div>
      )}
    </div>
  );
}

// Compact version for smaller displays
export function BetCardCompact({ player, propType, prop }: Omit<BetCardProps, 'gameContext'>) {
  const isOver = prop.pick === 'OVER';
  const hasLine = prop.line !== null && prop.line !== undefined && prop.line > 0;
  const edge = hasLine ? prop.prediction - prop.line! : 0;
  const propLabel = propType === '3PM' ? '3PM' : propType.slice(0, 3);

  return (
    <div className={cn(
      'flex items-center gap-3 p-3 rounded-lg border',
      'bg-bg-card hover:bg-bg-hover transition-colors',
      isOver ? 'border-green-500/30' : 'border-red-500/30'
    )}>
      {/* Bet action */}
      <div className={cn(
        'flex flex-col items-center justify-center w-16 py-2 rounded',
        isOver ? 'bg-green-500/20' : 'bg-red-500/20'
      )}>
        <span className={cn(
          'text-xs font-bold',
          isOver ? 'text-green-400' : 'text-red-400'
        )}>
          {prop.pick}
        </span>
        <span className="text-sm font-bold text-text-primary">
          {hasLine ? prop.line!.toFixed(1) : '—'}
        </span>
        <span className="text-[10px] text-text-muted uppercase">{propLabel}</span>
      </div>

      {/* Player info */}
      <div className="flex-1 min-w-0">
        <div className="font-medium text-text-primary truncate">{player.player_name}</div>
        <div className="text-xs text-text-muted">{player.team}</div>
      </div>

      {/* Prediction and confidence */}
      <div className="text-right">
        <div className={cn(
          'font-bold',
          isOver ? 'text-green-400' : 'text-red-400'
        )}>
          {prop.prediction.toFixed(1)}
        </div>
        {hasLine && (
          <div className={cn(
            'text-xs',
            isOver ? 'text-green-400/70' : 'text-red-400/70'
          )}>
            {isOver ? '+' : ''}{edge.toFixed(1)}
          </div>
        )}
      </div>

      {/* Confidence indicator - thresholds adjusted for model's range */}
      <div className="flex flex-col items-center">
        <div className={cn(
          'text-sm font-bold px-2 py-0.5 rounded',
          prop.confidence >= 70 ? 'bg-green-500/20 text-green-400' :
          prop.confidence >= 60 ? 'bg-yellow-500/20 text-yellow-400' :
          'bg-gray-500/20 text-gray-400'
        )}>
          {Math.round(prop.confidence)}%
        </div>
      </div>
    </div>
  );
}
