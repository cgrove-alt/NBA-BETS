import { Card } from './Card';
import { Badge, EdgeBadge } from './Badge';
import { Button } from './Button';
import { ConfidenceMeter } from './ConfidenceMeter';
import {
  Clock,
  TrendingUp,
  Zap,
  ChevronRight,
  Target,
  Lock,
  Radio,
} from 'lucide-react';

// Types
export interface BetCardData {
  id: string;
  matchup: {
    homeTeam: string;
    homeAbbrev: string;
    awayTeam: string;
    awayAbbrev: string;
    gameTime: string; // ISO string or formatted time
    status?: 'upcoming' | 'live' | 'final';
  };
  pick: {
    type: 'spread' | 'moneyline' | 'total' | 'prop';
    selection: string; // e.g., "LAL -5.5", "Over 224.5", "LeBron Over 25.5 PTS"
    odds: number; // American odds, e.g., -110
  };
  edge: number; // Percentage, e.g., 15.5 for +15.5%
  confidence: number; // 0-100
  sportsbook?: string;
  signals?: Array<{
    label: string;
    type: 'positive' | 'negative' | 'neutral';
  }>;
  /** If true, game has started - betting is locked for integrity */
  locked?: boolean;
}

interface BetCardProps {
  bet: BetCardData;
  variant?: 'featured' | 'compact' | 'list';
  onTake?: (bet: BetCardData) => void;
  onExpand?: (bet: BetCardData) => void;
  className?: string;
}

/**
 * BetCard - The Hero Component
 *
 * The primary UI element for displaying betting recommendations.
 * Designed to be visually striking with clear call-to-action.
 *
 * Variants:
 * - featured: Large, detailed card for top picks (hero section)
 * - compact: Medium card for carousel/grid display
 * - list: Slim row for dense lists
 */
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

  // Determine card variant based on edge/confidence (dimmed if locked)
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
        onExpand={onExpand}
        className={className}
      />
    );
  }

  // Default: Compact variant
  return (
    <CompactBetCard
      bet={bet}
      cardVariant={cardVariant}
      isHighConfidence={isHighConfidence}
      isLocked={isLocked}
      onTake={onTake}
      onExpand={onExpand}
      className={className}
    />
  );
}

/**
 * Featured BetCard - Large hero card for top picks
 */
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
      {/* Top Pick Banner */}
      {isTopPick && !isLocked && (
        <div className="absolute top-0 left-0 right-0 h-1 gradient-gold" />
      )}

      {/* Locked Banner for games in progress */}
      {isLocked && (
        <div className="absolute top-0 left-0 right-0 h-1 bg-[#ff3355]" />
      )}

      <div className="p-6">
        {/* Header: Matchup + Time */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            {/* Team Matchup */}
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

          {/* Live Badge (when locked) or Top Pick Badge */}
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

        {/* Main Pick - THE BIG TEXT */}
        <div className="mb-6">
          <div className="text-sm text-text-muted uppercase tracking-wide mb-2">
            {getPickTypeLabel(bet.pick.type)}
          </div>
          <div className="text-4xl md:text-5xl font-bold text-text-primary mb-2">
            {bet.pick.selection}
          </div>
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
        </div>

        {/* Stats Row */}
        <div className="flex items-center gap-6 mb-6">
          {/* Confidence Meter */}
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

          {/* Edge */}
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
              variant="action"
              size="lg"
              fullWidth
              onClick={() => onTake?.(bet)}
              icon={<Target className="w-5 h-5" />}
            >
              TAKE THIS BET
            </Button>
          )}
          {onExpand && (
            <Button
              variant="secondary"
              size="lg"
              onClick={() => onExpand(bet)}
              icon={<ChevronRight className="w-5 h-5" />}
            >
              Details
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}

/**
 * Compact BetCard - Medium card for grids/carousels
 */
function CompactBetCard({
  bet,
  cardVariant,
  isHighConfidence,
  isLocked,
  onTake,
  onExpand,
  className,
}: {
  bet: BetCardData;
  cardVariant: 'default' | 'success' | 'gold';
  isHighConfidence: boolean;
  isLocked: boolean;
  onTake?: (bet: BetCardData) => void;
  onExpand?: (bet: BetCardData) => void;
  className?: string;
}) {
  return (
    <Card
      variant={cardVariant}
      glow={isHighConfidence && !isLocked}
      className={`${isLocked ? 'opacity-75' : ''} ${className}`}
      onClick={() => onExpand?.(bet)}
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <TeamLogo abbrev={bet.matchup.awayAbbrev} size="sm" />
            <span className="text-text-muted text-xs">@</span>
            <TeamLogo abbrev={bet.matchup.homeAbbrev} size="sm" />
          </div>
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
        </div>

        {/* Pick */}
        <div className="mb-3">
          <div className="text-xs text-text-muted uppercase tracking-wide mb-1">
            {getPickTypeLabel(bet.pick.type)}
          </div>
          <div className="text-xl font-bold text-text-primary truncate">
            {bet.pick.selection}
          </div>
          <div className="text-sm font-mono text-text-secondary">
            {formatOdds(bet.pick.odds)}
          </div>
        </div>

        {/* Stats */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <ConfidenceMeter value={bet.confidence} size="sm" />
            <span className="text-sm font-semibold">{bet.confidence}%</span>
          </div>
          <EdgeBadge edge={bet.edge} size="sm" />
        </div>

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
            variant="action"
            size="sm"
            fullWidth
            onClick={(e) => {
              e.stopPropagation();
              onTake?.(bet);
            }}
          >
            TAKE
          </Button>
        )}
      </div>
    </Card>
  );
}

/**
 * List BetCard - Slim row for dense lists
 */
function ListBetCard({
  bet,
  isPositiveEdge,
  isLocked,
  onTake,
  onExpand,
  className,
}: {
  bet: BetCardData;
  isPositiveEdge: boolean;
  isLocked: boolean;
  onTake?: (bet: BetCardData) => void;
  onExpand?: (bet: BetCardData) => void;
  className?: string;
}) {
  return (
    <div
      className={`
        flex items-center gap-4 p-3 rounded-lg
        bg-bg-card border border-border
        hover:bg-bg-card-hover hover:border-[rgba(255,255,255,0.1)]
        transition-all duration-200 cursor-pointer
        ${isLocked ? 'opacity-75' : ''}
        ${className}
      `}
      onClick={() => onExpand?.(bet)}
    >
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
        <div className="text-xs text-text-muted">
          {formatOdds(bet.pick.odds)}
        </div>
      </div>

      {/* Live Badge (when locked) */}
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
          variant={isPositiveEdge ? 'success' : 'secondary'}
          size="sm"
          onClick={(e) => {
            e.stopPropagation();
            onTake?.(bet);
          }}
        >
          TAKE
        </Button>
      )}
    </div>
  );
}

/**
 * Team Logo Placeholder
 * In production, this would load actual team logos
 */
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

// Utility functions
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
