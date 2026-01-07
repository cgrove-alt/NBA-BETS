import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ChevronRight, Calendar, Flame, TrendingUp } from 'lucide-react';
import { ResponsiveLayout } from '../../components/v2/ResponsiveLayout';
import { BetCard } from '../../components/v2/BetCard';
import type { BetCardData } from '../../components/v2/BetCard';
import { BankrollSummary, PnLTicker } from '../../components/v2/BankrollSummary';
import type { BankrollData } from '../../components/v2/BankrollSummary';
import { Card } from '../../components/v2/Card';
import { Button } from '../../components/v2/Button';
import { Badge } from '../../components/v2/Badge';
import { BetCardSkeleton, GameCardSkeleton } from '../../components/v2/LoadingSkeleton';
import { getGames, getBestBets } from '../../lib/api';
import type { Game } from '../../lib/types';
import { getTodayDate } from '../../components/game/DateSelector';

/**
 * Check if a game has started based on its status
 * Games are locked once they start to prevent retroactive betting
 */
function isGameStarted(status: string | undefined): boolean {
  if (!status) return false;

  // Game statuses that indicate the game has started or ended
  const startedPatterns = [
    'Qtr',           // "1st Qtr", "2nd Qtr", etc.
    'Quarter',       // Alternative format
    'Half',          // "Halftime", "1st Half", "2nd Half"
    'OT',            // Overtime
    'Final',         // Game ended
    'In Progress',   // Generic in-progress
    'Live',          // Live game
  ];

  return startedPatterns.some(pattern =>
    status.toLowerCase().includes(pattern.toLowerCase())
  );
}

/**
 * Dashboard - The Oracle Home Page
 *
 * Features:
 * - Bankroll summary at the top
 * - "Top Picks of the Day" hero section
 * - Today's games quick view
 * - Performance snapshot
 */
export function Dashboard() {
  const [selectedDate] = useState<string>(getTodayDate());

  // Fetch today's games
  const { data: gamesData, isLoading: gamesLoading } = useQuery({
    queryKey: ['games', selectedDate],
    queryFn: () => getGames(selectedDate),
    staleTime: 5 * 60 * 1000,
  });

  // Fetch best bets across all games
  // Note: Model confidence outputs range 50-70%, so use lower thresholds
  const { data: bestBetsData, isLoading: bestBetsLoading } = useQuery({
    queryKey: ['bestBets'],
    queryFn: () => getBestBets({ minConfidence: 50, minEdge: 3 }),
    staleTime: 5 * 60 * 1000,
  });

  // Mock bankroll data (in production, this would come from a backend/local storage)
  const bankrollData: BankrollData = {
    totalBankroll: 5000,
    todayPnL: 245.50,
    weekPnL: 892.00,
    monthPnL: 2150.00,
    allTimeROI: 12.5,
    winRate: 58.3,
    activeBets: 3,
    pendingBets: 2,
  };

  const games = gamesData?.games || [];
  const bestBets = bestBetsData?.best_bets || [];

  // Create a map of game_id -> game data for quick lookup
  const gamesMap = useMemo(() => {
    const map = new Map<string, Game>();
    games.forEach((game) => map.set(game.game_id, game));
    return map;
  }, [games]);

  // Transform best bets to BetCardData format
  // CRITICAL: Lock bets for games that have already started (betting integrity)
  const topPicks: BetCardData[] = useMemo(() => {
    return bestBets.slice(0, 5).map((bet) => {
      const game = gamesMap.get(bet.game_id);
      const gameStatus = game?.status;
      const isLocked = isGameStarted(gameStatus);

      return {
        id: `${bet.game_id}-${bet.player_id}-${bet.prop_type}`,
        matchup: {
          homeTeam: game?.home_team?.name || 'Home',
          homeAbbrev: game?.home_team?.abbreviation || bet.team,
          awayTeam: game?.visitor_team?.name || 'Away',
          awayAbbrev: game?.visitor_team?.abbreviation || '---',
          gameTime: game?.game_time || new Date().toISOString(),
          status: isLocked ? 'live' as const : 'upcoming' as const,
        },
        pick: {
          type: 'prop' as const,
          selection: `${bet.player_name} ${bet.pick} ${bet.line} ${bet.prop_type}`,
          odds: -110, // Default odds
        },
        edge: bet.edge_pct,
        confidence: bet.confidence,
        signals: [
          { label: `${bet.prop_type}`, type: 'neutral' as const },
          bet.edge_pct > 10 ? { label: 'High Value', type: 'positive' as const } : null,
        ].filter(Boolean) as BetCardData['signals'],
        locked: isLocked, // Lock betting for games in progress
      };
    });
  }, [bestBets, gamesMap]);

  // Handle bet action
  const handleTakeBet = (bet: BetCardData) => {
    console.log('Taking bet:', bet);
    // In production: add to bet slip, track analytics
  };

  const handleExpandBet = (bet: BetCardData) => {
    console.log('Expanding bet:', bet);
    // In production: navigate to detailed view
  };

  return (
    <ResponsiveLayout bankroll={bankrollData} activePage="dashboard">
      <div className="space-y-6 pb-20 md:pb-6">
        {/* Hero Section - Today's Top Pick */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Flame className="w-5 h-5 text-[#ff8800]" />
              <h2 className="text-lg font-bold text-text-primary">Top Pick</h2>
            </div>
            <Badge variant="premium" glow>
              <TrendingUp className="w-3 h-3 mr-1" />
              HOT
            </Badge>
          </div>

          {bestBetsLoading ? (
            <BetCardSkeleton variant="featured" />
          ) : topPicks.length > 0 ? (
            <BetCard
              bet={topPicks[0]}
              variant="featured"
              onTake={handleTakeBet}
              onExpand={handleExpandBet}
            />
          ) : (
            <Card className="p-8 text-center">
              <p className="text-text-muted">No top picks available yet.</p>
              <p className="text-sm text-text-muted mt-2">
                Check back closer to game time for predictions.
              </p>
            </Card>
          )}
        </section>

        {/* Bankroll Overview (Full on mobile) */}
        <section className="md:hidden">
          <h2 className="text-lg font-bold text-text-primary mb-4">Your Bankroll</h2>
          <BankrollSummary data={bankrollData} variant="full" />
        </section>

        {/* More Top Picks */}
        {topPicks.length > 1 && (
          <section>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-text-primary">More Picks</h2>
              <Button variant="ghost" size="sm" icon={<ChevronRight className="w-4 h-4" />} iconPosition="right">
                View All
              </Button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {topPicks.slice(1, 4).map((pick) => (
                <BetCard
                  key={pick.id}
                  bet={pick}
                  variant="compact"
                  onTake={handleTakeBet}
                  onExpand={handleExpandBet}
                />
              ))}
            </div>
          </section>
        )}

        {/* Today's Games */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Calendar className="w-5 h-5 text-[#00d4ff]" />
              <h2 className="text-lg font-bold text-text-primary">Today's Games</h2>
            </div>
            <span className="text-sm text-text-muted">
              {games.length} game{games.length !== 1 ? 's' : ''}
            </span>
          </div>

          {gamesLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <GameCardSkeleton key={i} />
              ))}
            </div>
          ) : games.length > 0 ? (
            <div className="space-y-3">
              {games.map((game, i) => (
                <div key={game.game_id} className={`animate-stagger-${Math.min(i + 1, 5)}`}>
                  <GameCard game={game} />
                </div>
              ))}
            </div>
          ) : (
            <Card className="p-8 text-center">
              <p className="text-text-muted">No games scheduled for today.</p>
            </Card>
          )}
        </section>

        {/* Quick Stats */}
        <section>
          <h2 className="text-lg font-bold text-text-primary mb-4">Performance</h2>
          <div className="grid grid-cols-2 gap-4">
            <Card glow>
              <div className="p-4">
                <div className="text-xs text-text-muted uppercase tracking-wider mb-1">
                  This Week
                </div>
                <PnLTicker value={bankrollData.weekPnL} size="lg" />
              </div>
            </Card>
            <Card>
              <div className="p-4">
                <div className="text-xs text-text-muted uppercase tracking-wider mb-1">
                  Win Rate
                </div>
                <div className="text-2xl font-bold text-[#00ff88]">
                  {bankrollData.winRate.toFixed(1)}%
                </div>
              </div>
            </Card>
          </div>
        </section>
      </div>
    </ResponsiveLayout>
  );
}

/**
 * Game Card - Shows a single game matchup
 */
function GameCard({ game }: { game: Game }) {
  const gameTime = game.game_time
    ? new Date(game.game_time).toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
      })
    : game.status || 'TBD';

  const isLive = game.status?.includes('Qtr') || game.status === 'In Progress';
  const isFinal = game.status === 'Final';

  return (
    <Card hover className="cursor-pointer">
      <div className="p-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Teams */}
          <div className="flex items-center gap-2">
            <TeamBadge abbrev={game.visitor_team.abbreviation} />
            <span className="text-text-muted text-sm">@</span>
            <TeamBadge abbrev={game.home_team.abbreviation} />
          </div>

          {/* Team Names (hidden on mobile) */}
          <div className="hidden sm:block">
            <div className="text-sm text-text-primary font-medium">
              {game.visitor_team.abbreviation} @ {game.home_team.abbreviation}
            </div>
          </div>
        </div>

        {/* Time/Status */}
        <div className="flex items-center gap-2">
          {isLive && (
            <Badge variant="danger" size="sm" glow>
              LIVE
            </Badge>
          )}
          {isFinal && (
            <Badge variant="default" size="sm">
              FINAL
            </Badge>
          )}
          <span className={`text-sm ${isLive ? 'text-[#ff3355]' : 'text-text-muted'}`}>
            {gameTime}
          </span>
          <ChevronRight className="w-4 h-4 text-text-muted" />
        </div>
      </div>
    </Card>
  );
}

/**
 * Team Badge - Shows team abbreviation
 */
function TeamBadge({ abbrev }: { abbrev: string }) {
  return (
    <div className="w-10 h-10 rounded-full bg-bg-tertiary border border-border flex items-center justify-center text-sm font-bold text-text-secondary">
      {abbrev}
    </div>
  );
}
