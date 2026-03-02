import { useState, useMemo, useCallback, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { ChevronRight, Calendar, Flame, TrendingUp } from 'lucide-react';
import { ResponsiveLayout } from '../../components/v2/ResponsiveLayout';
import { BetCard } from '../../components/v2/BetCard';
import type { BetCardData } from '../../components/v2/BetCard';
import { BankrollSummary, PnLTicker } from '../../components/v2/BankrollSummary';
import { Card } from '../../components/v2/Card';
import { Button } from '../../components/v2/Button';
import { Badge } from '../../components/v2/Badge';
import { BetCardSkeleton, GameCardSkeleton } from '../../components/v2/LoadingSkeleton';
import { getGames, getBestBets } from '../../lib/api';
import type { Game } from '../../lib/types';
import { getTodayDate } from '../../components/game/DateSelector';
import { useBankroll } from '../../hooks/useBankroll';
import { isGameStarted, classifySignals } from '../../lib/utils';

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
  const navigate = useNavigate();
  const [selectedDate] = useState<string>(getTodayDate());

  // Retry counter must be declared before useQuery that references it
  const emptyRetryCount = useRef(0);

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
    // After a deploy, props generate in the background. Retry every 5s
    // up to 6 times (30s max) until data arrives.
    refetchInterval: (query) => {
      const bets = query.state.data?.best_bets;
      const hasGames = (gamesData?.games?.length ?? 0) > 0;
      if (hasGames && (!bets || bets.length === 0)) {
        emptyRetryCount.current++;
        return emptyRetryCount.current <= 6 ? 5000 : false;
      }
      emptyRetryCount.current = 0;
      return false;
    },
  });

  // Real bankroll data from /api/bankroll
  const { bankrollData } = useBankroll();

  const games = useMemo(() => gamesData?.games || [], [gamesData]);
  const gamesMap = useMemo(() => {
    const map = new Map<string, Game>();
    games.forEach((game) => map.set(game.game_id, game));
    return map;
  }, [games]);

  // Count picks per game for display on game cards
  const pickCountMap = useMemo(() => {
    const map = new Map<string, number>();
    const bets = bestBetsData?.best_bets || [];
    bets.forEach((b) => map.set(b.game_id, (map.get(b.game_id) || 0) + 1));
    return map;
  }, [bestBetsData]);

  // Transform best bets to BetCardData format
  // CRITICAL: Lock bets for games that have already started (betting integrity)
  const topPicks: BetCardData[] = useMemo(() => {
    const bestBets = bestBetsData?.best_bets || [];
    return bestBets.slice(0, 5).map((bet) => {
      const game = gamesMap.get(bet.game_id);
      const gameStatus = game?.status;
      const isLocked = isGameStarted(gameStatus);

      const signals = classifySignals(bet.signals || [], bet.prop_type);

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
          odds: 0,
        },
        edge: bet.edge_pct,
        confidence: bet.confidence,
        signals,
        locked: isLocked,
        rank: bet.rank,
        explanation: bet.explanation,
        seasonAvg: bet.season_avg,
        recentAvg: bet.recent_avg,
      };
    });
  }, [bestBetsData, gamesMap]);

  // Copy bet details to clipboard on TAKE
  const [copiedBetId, setCopiedBetId] = useState<string | null>(null);

  const handleTakeBet = useCallback((bet: BetCardData) => {
    const text = `${bet.pick.selection} | Edge: ${bet.edge > 0 ? '+' : ''}${bet.edge.toFixed(1)}% | Conf: ${bet.confidence}%`;
    navigator.clipboard.writeText(text).catch(() => {});
    setCopiedBetId(bet.id);
    setTimeout(() => setCopiedBetId(null), 2000);
  }, []);

  const handleExpandBet = useCallback((bet: BetCardData) => {
    // bet.id format is "gameId-playerId-propType" — extract the game_id
    const gameId = bet.id.split('-')[0];
    navigate(`/predictions?game=${gameId}`);
  }, [navigate]);

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
              bet={{ ...topPicks[0], copied: copiedBetId === topPicks[0].id }}
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
              <Button variant="ghost" size="sm" icon={<ChevronRight className="w-4 h-4" />} iconPosition="right" onClick={() => navigate('/predictions')}>
                View All
              </Button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {topPicks.slice(1, 4).map((pick) => (
                <BetCard
                  key={pick.id}
                  bet={{ ...pick, copied: copiedBetId === pick.id }}
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
                  <GameCard
                    game={game}
                    pickCount={pickCountMap.get(game.game_id) || 0}
                    onClick={() => navigate(`/predictions?game=${game.game_id}`)}
                  />
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
function GameCard({ game, pickCount, onClick }: { game: Game; pickCount?: number; onClick?: () => void }) {
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
    <Card hover className="cursor-pointer" onClick={onClick}>
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
            {pickCount != null && pickCount > 0 && (
              <div className="text-xs text-text-muted">{pickCount} pick{pickCount !== 1 ? 's' : ''}</div>
            )}
          </div>
        </div>

        {/* Time/Status */}
        <div className="flex items-center gap-2">
          {pickCount != null && pickCount > 0 && (
            <span className="text-xs text-[#00d4ff] font-medium sm:hidden">
              {pickCount} pick{pickCount !== 1 ? 's' : ''}
            </span>
          )}
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
