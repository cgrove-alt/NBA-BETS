import { useState } from 'react';
import { Card } from './Card';
import { Badge, EdgeBadge } from './Badge';
import { Button } from './Button';
import { ConfidenceMeter } from './ConfidenceMeter';
import {
  Clock,
  TrendingUp,
  Zap,
  ChevronDown,
  ChevronUp,
  Target,
  Lock,
  Radio,
  Check,
} from 'lucide-react';

// Types
export interface BetCardData {
  id: string;
  matchup: {
    homeTeam: string;
    homeAbbrev: string;
    awayTeam: string;
    awayAbbrev: string;
    gameTime: string;
    status?: 'upcoming' | 'live' | 'final';
  };
  pick: {
    type: 'spread' | 'moneyline' | 'total' | 'prop';
    selection: string;
    odds: number;
  };
  edge: number;
  confidence: number;
  sportsbook?: string;
  signals?: Array<{
    label: string;
    type: 'positive' | 'negative' | 'neutral';
  }>;
  locked?: boolean;
  rank?: number;
  explanation?: string;
  seasonAvg?: number | null;
  recentAvg?: number | null;
  copied?: boolean;
}

interface BetCardProps {
  bet: BetCardData;
  variant?: 'featured' | 'compact' | 'list';
  onTake?: (bet: BetCardData) => void;
  onExpand?: (bet: BetCardData) => void;
  className?: string;
}

export function BetCard({
  bet,
  variant = 'compact',
  onTake,
  onExpand,
  className = '',
}: BetCardProps) {
  const isPositiveEdge = bet.edge > 0;
  const isHighConfidence = bet.confidence >= 65;
  const isTopPick = bet.edge >= 15 && bet.confidence >= 60;
  const isLocked = bet.locked === true;

  const cardVariant = isLocked ? 'default' : isTopPick ? 'gold' : isPositiveEdge ? 'success' : 'default';

  if (variant === 'featured') {
    return (
      <FeaturedBetCard
        bet={bet}
        cardVariant={cardVariant}
        isTopPick={isTopPick}
        isLocked={isLocked}
        onTake={onTake}
        onExpand={onExpand}
        className={className}
      />
    );
  }

  if (variant === 'list') {
    return (
      <ListBetCard
        bet={bet}
        isPositiveEdge={isPositiveEdge}
        isLocked={isLocked}
        onTake={onTake}
        className={className}
      />
    );
  }

  return (
    <CompactBetCard
      bet={bet}
      cardVariant={cardVariant}
      isHighConfidence={isHighConfidence}
      isLocked={isLocked}
      onTake={onTake}
      className={className}
    />
  );
}

function FeaturedBetCard({
  bet,
  cardVariant,
  isTopPick,
  isLocked,
  onTake,
  onExpand,
  className,
}: {
  bet: BetCardData;
  cardVariant: 'default' | 'success' | 'gold';
  isTopPick: boolean;
  isLocked: boolean;
  onTake?: (bet: BetCardData) => void;
  onExpand?: (bet: BetCardData) => void;
  className?: string;
}) {
  return (
    <Card
      variant={cardVariant}
      glow={isTopPick && !isLocked}
      hover={false}
      className={`relative overflow-hidden ${isLocked ? 'opacity-75' : ''} ${className}`}
    >
      {isTopPick && !isLocked && (
        <div className="absolute top-0 left-0 right-0 h-1 gradient-gold" />
      )}
      {isLocked && (
        <div className="absolute top-0 left-0 right-0 h-1 bg-[#ff3355]" />
      )}

      <div className="p-6">
        {/* Header: Matchup + Time */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <TeamLogo abbrev={bet.matchup.awayAbbrev} size="lg" />
              <span className="text-text-muted text-lg">@</span>
              <TeamLogo abbrev={bet.matchup.homeAbbrev} size="lg" />
            </div>
            <div>
              <div className="text-lg font-semibold text-text-primary">
                {bet.matchup.awayTeam} @ {bet.matchup.homeTeam}
              </div>
              <div className="flex items-center gap-2 text-sm text-text-muted">
                <Clock className="w-4 h-4" />
                {formatGameTime(bet.matchup.gameTime)}
              </div>
            </div>
          </div>

          {isLocked ? (
            <Badge variant="danger" glow className="gap-1">
              <Radio className="w-3 h-3 animate-pulse" />
              LIVE
            </Badge>
          ) : isTopPick ? (
            <Badge variant="premium" glow className="gap-1">
              <Zap className="w-3 h-3" />
              TOP PICK
            </Badge>
          ) : null}
        </div>

        {/* Main Pick */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm text-text-muted uppercase tracking-wide">
              {getPickTypeLabel(bet.pick.type)}
            </span>
            {bet.rank != null && bet.rank <= 10 && (
              <Badge variant={bet.rank <= 3 ? 'premium' : 'default'} size="sm">
                #{bet.rank}
              </Badge>
            )}
          </div>
          <div className="text-4xl md:text-5xl font-bold text-text-primary mb-2">
            {bet.pick.selection}
          </div>
          {formatOdds(bet.pick.odds) && (
            <div className="flex items-center gap-4">
              <span className="text-2xl font-mono text-text-secondary">
                {formatOdds(bet.pick.odds)}
              </span>
              {bet.sportsbook && (
                <span className="text-sm text-text-muted">
                  via {bet.sportsbook}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Explanation */}
        {bet.explanation && (
          <div className="text-sm text-text-secondary leading-relaxed mb-4 bg-bg-tertiary rounded-lg p-3">
            {bet.explanation}
          </div>
        )}

        {/* Season vs Recent averages */}
        {(bet.seasonAvg != null || bet.recentAvg != null) && (
          <div className="flex items-center gap-4 mb-4 text-xs text-text-muted">
            {bet.seasonAvg != null && (
              <span>Season: <span className="text-text-primary font-semibold">{bet.seasonAvg}</span></span>
            )}
            {bet.recentAvg != null && (
              <span>Last 5: <span className="text-text-primary font-semibold">{bet.recentAvg}</span></span>
            )}
          </div>
        )}

        {/* Stats Row */}
        <div className="flex items-center gap-6 mb-6">
          <div className="flex items-center gap-3">
            <ConfidenceMeter value={bet.confidence} size="lg" />
            <div>
              <div className="text-xs text-text-muted uppercase">Confidence</div>
              <div className="text-lg font-bold text-text-primary">
                {bet.confidence}%
              </div>
            </div>
          </div>

          <div className="w-px h-12 bg-border" />

          <div>
            <div className="text-xs text-text-muted uppercase">Edge</div>
            <div className="flex items-center gap-2">
              <TrendingUp className={`w-5 h-5 ${bet.edge > 0 ? 'text-[#00ff88]' : 'text-[#ff3355]'}`} />
              <span className={`text-2xl font-mono font-bold ${bet.edge > 0 ? 'text-[#00ff88] text-glow-green' : 'text-[#ff3355]'}`}>
                {bet.edge > 0 ? '+' : ''}{bet.edge.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {/* Signals */}
        {bet.signals && bet.signals.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-6">
            {bet.signals.map((signal, i) => (
              <Badge
                key={i}
                variant={signal.type === 'positive' ? 'success' : signal.type === 'negative' ? 'danger' : 'default'}
                size="sm"
              >
                {signal.label}
              </Badge>
            ))}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-3">
          {isLocked ? (
            <Button
              variant="secondary"
              size="lg"
              fullWidth
              disabled
              icon={<Lock className="w-5 h-5" />}
              className="opacity-50 cursor-not-allowed"
            >
              GAME IN PROGRESS
            </Button>
          ) : (
            <Button
              variant={bet.copied ? 'success' : 'action'}
              size="lg"
              fullWidth
              onClick={() => onTake?.(bet)}
              icon={bet.copied ? <Check className="w-5 h-5" /> : <Target className="w-5 h-5" />}
            >
              {bet.copied ? 'COPIED TO CLIPBOARD' : 'TAKE THIS BET'}
            </Button>
          )}
          {onExpand && (
            <Button
              variant="secondary"
              size="lg"
              onClick={() => onExpand(bet)}
              icon={<ChevronDown className="w-5 h-5" />}
            >
              Details
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}

function CompactBetCard({
  bet,
  cardVariant,
  isHighConfidence,
  isLocked,
  onTake,
  className,
}: {
  bet: BetCardData;
  cardVariant: 'default' | 'success' | 'gold';
  isHighConfidence: boolean;
  isLocked: boolean;
  onTake?: (bet: BetCardData) => void;
  className?: string;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <Card
      variant={cardVariant}
      glow={isHighConfidence && !isLocked}
      className={`${isLocked ? 'opacity-75' : ''} ${className}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <TeamLogo abbrev={bet.matchup.awayAbbrev} size="sm" />
            <span className="text-text-muted text-xs">@</span>
            <TeamLogo abbrev={bet.matchup.homeAbbrev} size="sm" />
          </div>
          <div className="flex items-center gap-2">
            {isLocked ? (
              <Badge variant="danger" size="sm" glow>
                <Radio className="w-2.5 h-2.5 animate-pulse mr-1" />
                LIVE
              </Badge>
            ) : (
              <div className="flex items-center gap-1 text-xs text-text-muted">
                <Clock className="w-3 h-3" />
                {formatGameTime(bet.matchup.gameTime)}
              </div>
            )}
            {expanded ? (
              <ChevronUp className="w-4 h-4 text-text-muted" />
            ) : (
              <ChevronDown className="w-4 h-4 text-text-muted" />
            )}
          </div>
        </div>

        {/* Pick */}
        <div className="mb-3">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs text-text-muted uppercase tracking-wide">
              {getPickTypeLabel(bet.pick.type)}
            </span>
            {bet.rank != null && bet.rank <= 20 && (
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold ${
                bet.rank <= 3 ? 'bg-[rgba(255,215,0,0.2)] text-[#ffd700] border border-[rgba(255,215,0,0.4)]' : 'bg-bg-tertiary text-text-muted border border-border'
              }`}>
                #{bet.rank}
              </div>
            )}
          </div>
          <div className="text-xl font-bold text-text-primary truncate">
            {bet.pick.selection}
          </div>
          {formatOdds(bet.pick.odds) && (
            <div className="text-sm font-mono text-text-secondary">
              {formatOdds(bet.pick.odds)}
            </div>
          )}
          {!expanded && bet.explanation && (
            <div className="text-xs text-text-muted truncate mt-1">
              {bet.explanation}
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <ConfidenceMeter value={bet.confidence} size="sm" />
            <span className="text-sm font-semibold">{bet.confidence}%</span>
          </div>
          <EdgeBadge edge={bet.edge} size="sm" />
        </div>

        {/* Expanded Details */}
        {expanded && (
          <div className="border-t border-border mt-1 pt-3 space-y-2">
            {bet.seasonAvg != null && (
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">Season Avg</span>
                <span className="text-text-primary font-semibold">{bet.seasonAvg}</span>
              </div>
            )}
            {bet.recentAvg != null && (
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">Last 5 Avg</span>
                <span className="text-text-primary font-semibold">{bet.recentAvg}</span>
              </div>
            )}

            {bet.explanation && (
              <div className="text-xs text-text-secondary bg-bg-tertiary rounded p-2 mt-2">
                {bet.explanation}
              </div>
            )}

            {bet.signals && bet.signals.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {bet.signals.map((signal, i) => (
                  <Badge key={i} variant={signal.type === 'positive' ? 'success' : signal.type === 'negative' ? 'danger' : 'default'} size="sm">
                    {signal.label}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Action */}
        {isLocked ? (
          <Button
            variant="secondary"
            size="sm"
            fullWidth
            disabled
            className="opacity-50 cursor-not-allowed"
          >
            <Lock className="w-3 h-3 mr-1" />
            LOCKED
          </Button>
        ) : (
          <Button
            variant={bet.copied ? 'success' : 'action'}
            size="sm"
            fullWidth
            onClick={(e) => {
              e.stopPropagation();
              onTake?.(bet);
            }}
          >
            {bet.copied ? 'COPIED' : 'TAKE'}
          </Button>
        )}
      </div>
    </Card>
  );
}

function ListBetCard({
  bet,
  isPositiveEdge,
  isLocked,
  onTake,
  className,
}: {
  bet: BetCardData;
  isPositiveEdge: boolean;
  isLocked: boolean;
  onTake?: (bet: BetCardData) => void;
  className?: string;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`
        rounded-lg bg-bg-card border border-border
        hover:bg-bg-card-hover hover:border-[rgba(255,255,255,0.1)]
        transition-all duration-200 cursor-pointer
        ${isLocked ? 'opacity-75' : ''}
        ${className}
      `}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-4 p-3">
        {/* Rank */}
        {bet.rank != null && bet.rank <= 20 && (
          <span className={`text-xs font-bold w-6 text-center ${
            bet.rank <= 3 ? 'text-[#ffd700]' : 'text-text-muted'
          }`}>
            #{bet.rank}
          </span>
        )}

        {/* Teams */}
        <div className="flex items-center gap-1.5 min-w-[80px]">
          <TeamLogo abbrev={bet.matchup.awayAbbrev} size="xs" />
          <span className="text-text-muted text-xs">@</span>
          <TeamLogo abbrev={bet.matchup.homeAbbrev} size="xs" />
        </div>

        {/* Pick */}
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-text-primary truncate">
            {bet.pick.selection}
          </div>
          {!expanded && bet.explanation && (
            <div className="text-xs text-text-muted truncate">
              {bet.explanation}
            </div>
          )}
        </div>

        {isLocked && (
          <Badge variant="danger" size="sm" glow>
            <Radio className="w-2.5 h-2.5 animate-pulse mr-1" />
            LIVE
          </Badge>
        )}

        {/* Confidence */}
        <div className="flex items-center gap-2">
          <ConfidenceMeter value={bet.confidence} size="xs" />
          <span className="text-sm font-mono w-10">{bet.confidence}%</span>
        </div>

        {/* Edge */}
        <EdgeBadge edge={bet.edge} size="sm" />

        {/* Action */}
        {isLocked ? (
          <Button
            variant="secondary"
            size="sm"
            disabled
            className="opacity-50 cursor-not-allowed"
          >
            <Lock className="w-3 h-3" />
          </Button>
        ) : (
          <Button
            variant={bet.copied ? 'success' : isPositiveEdge ? 'success' : 'secondary'}
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onTake?.(bet);
            }}
          >
            {bet.copied ? 'COPIED' : 'TAKE'}
          </Button>
        )}

        {expanded ? (
          <ChevronUp className="w-4 h-4 text-text-muted shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 text-text-muted shrink-0" />
        )}
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="border-t border-border px-3 pb-3 pt-2 space-y-2">
          <div className="grid grid-cols-2 gap-2 text-sm">
            {bet.seasonAvg != null && (
              <div className="flex justify-between">
                <span className="text-text-muted">Season Avg</span>
                <span className="text-text-primary font-semibold">{bet.seasonAvg}</span>
              </div>
            )}
            {bet.recentAvg != null && (
              <div className="flex justify-between">
                <span className="text-text-muted">Last 5 Avg</span>
                <span className="text-text-primary font-semibold">{bet.recentAvg}</span>
              </div>
            )}
          </div>

          {bet.explanation && (
            <div className="text-xs text-text-secondary bg-bg-tertiary rounded p-2">
              {bet.explanation}
            </div>
          )}

          {bet.signals && bet.signals.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {bet.signals.map((signal, i) => (
                <Badge key={i} variant={signal.type === 'positive' ? 'success' : signal.type === 'negative' ? 'danger' : 'default'} size="sm">
                  {signal.label}
                </Badge>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TeamLogo({ abbrev, size = 'md' }: { abbrev: string; size?: 'xs' | 'sm' | 'md' | 'lg' }) {
  const sizeClasses = {
    xs: 'w-6 h-6 text-[8px]',
    sm: 'w-8 h-8 text-[10px]',
    md: 'w-10 h-10 text-xs',
    lg: 'w-14 h-14 text-sm',
  };

  return (
    <div
      className={`
        ${sizeClasses[size]}
        rounded-full bg-bg-tertiary border border-border
        flex items-center justify-center
        font-bold text-text-secondary
      `}
    >
      {abbrev}
    </div>
  );
}

function formatGameTime(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  } catch {
    return isoString;
  }
}

function formatOdds(odds: number): string {
  if (odds === 0) return '';
  return odds > 0 ? `+${odds}` : `${odds}`;
}

function getPickTypeLabel(type: string): string {
  const labels: Record<string, string> = {
    spread: 'Spread',
    moneyline: 'Moneyline',
    total: 'Total',
    prop: 'Player Prop',
  };
  return labels[type] || type;
}
