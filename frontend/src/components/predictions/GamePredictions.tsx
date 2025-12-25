import { useEffect } from 'react';
import { TrendingUp, Target, Loader2 } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { useGameAnalysis } from '../../hooks/useGameAnalysis';
import { useGameResults } from '../../hooks/useGameResults';
import { GameResultsDisplay } from './GameResults';
import { cn } from '../../lib/utils';
import type { Game } from '../../lib/types';

interface GamePredictionsProps {
  gameId: string;
  game: Game;
}

function ConfidenceIndicator({ confidence }: { confidence: number }) {
  const getColor = () => {
    if (confidence >= 75) return 'text-accent-success';
    if (confidence >= 60) return 'text-accent-warning';
    return 'text-accent-danger';
  };

  return (
    <span className={cn('text-sm font-medium', getColor())}>
      {confidence.toFixed(0)}% conf
    </span>
  );
}

export function GamePredictions({ gameId, game }: GamePredictionsProps) {
  const isCompleted = game.status === 'Final';

  const {
    status,
    statusData,
    analysis,
    isLoading,
    isPending,
    startAnalysis,
    isStarting,
  } = useGameAnalysis(gameId, game);

  // Fetch results for completed games
  const { results, isLoading: resultsLoading } = useGameResults(
    isCompleted ? gameId : null
  );

  // Auto-start analysis when component mounts and status is not_started
  // Only for non-completed games
  useEffect(() => {
    if (!isCompleted && status === 'not_started' && !isStarting) {
      startAnalysis();
    }
  }, [status, isStarting, startAnalysis, isCompleted]);

  const homeAbbrev = game.home_team.abbreviation;
  const awayAbbrev = game.visitor_team.abbreviation;

  // Use statusData for quick display while full analysis loads
  const moneyline = analysis?.moneyline_prediction || statusData?.moneyline;
  const spread = analysis?.spread_prediction || statusData?.spread;

  // For completed games, show results
  if (isCompleted) {
    if (resultsLoading) {
      return (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Target className="text-accent-primary" size={18} />
              <CardTitle>Game Results</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center py-8">
              <div className="flex items-center gap-3 text-text-secondary">
                <Loader2 className="animate-spin" size={20} />
                <span>Loading results...</span>
              </div>
            </div>
          </CardContent>
        </Card>
      );
    }

    if (results) {
      return <GameResultsDisplay results={results} />;
    }
  }

  // Loading state for in-progress games
  if (isLoading || isPending || isStarting) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Target className="text-accent-primary" size={18} />
            <CardTitle>Game Predictions</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="flex items-center gap-3 text-text-secondary">
              <Loader2 className="animate-spin" size={20} />
              <span>Analyzing matchup...</span>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // No predictions available (ML failed or game already completed)
  if (!moneyline && !spread) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Target className="text-accent-primary" size={18} />
            <CardTitle>Game Predictions</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6 text-text-muted">
            <p>ML predictions unavailable for this game.</p>
            <p className="text-sm mt-2">This may happen for completed games or when models are loading.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Target className="text-accent-primary" size={18} />
          <CardTitle>Game Predictions</CardTitle>
          <span className="text-sm text-text-muted ml-auto">
            {homeAbbrev} vs {awayAbbrev}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Moneyline Card */}
          {moneyline && (
            <div className="bg-bg-secondary rounded-lg p-4 border border-border">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp size={16} className="text-accent-primary" />
                <span className="text-sm font-medium text-text-secondary uppercase tracking-wide">
                  Moneyline
                </span>
              </div>

              <div className="space-y-3">
                {/* Home Team */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        'font-bold text-lg',
                        moneyline.predicted_winner === 'home'
                          ? 'text-accent-success'
                          : 'text-text-primary'
                      )}
                    >
                      {homeAbbrev}
                    </span>
                    {moneyline.predicted_winner === 'home' && (
                      <span className="text-xs bg-success-light text-accent-success px-2 py-0.5 rounded">
                        PICK
                      </span>
                    )}
                  </div>
                  <div className="text-right">
                    <span className="text-lg font-semibold text-text-primary">
                      {(moneyline.home_win_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Probability Bar */}
                <div className="h-2 bg-bg-tertiary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-accent-success rounded-full transition-all duration-300"
                    style={{ width: `${moneyline.home_win_probability * 100}%` }}
                  />
                </div>

                {/* Away Team */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        'font-bold text-lg',
                        moneyline.predicted_winner === 'away'
                          ? 'text-accent-success'
                          : 'text-text-primary'
                      )}
                    >
                      {awayAbbrev}
                    </span>
                    {moneyline.predicted_winner === 'away' && (
                      <span className="text-xs bg-success-light text-accent-success px-2 py-0.5 rounded">
                        PICK
                      </span>
                    )}
                  </div>
                  <div className="text-right">
                    <span className="text-lg font-semibold text-text-primary">
                      {(moneyline.away_win_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Confidence */}
                <div className="pt-2 border-t border-border flex items-center justify-between">
                  <span className="text-sm text-text-muted">Model Confidence</span>
                  <ConfidenceIndicator confidence={moneyline.confidence * 100} />
                </div>
              </div>
            </div>
          )}

          {/* Spread Card */}
          {spread && (
            <div className="bg-bg-secondary rounded-lg p-4 border border-border">
              <div className="flex items-center gap-2 mb-3">
                <Target size={16} className="text-accent-primary" />
                <span className="text-sm font-medium text-text-secondary uppercase tracking-wide">
                  Spread
                </span>
              </div>

              <div className="space-y-3">
                {/* Predicted Spread */}
                <div className="text-center py-2">
                  <div className="text-3xl font-bold text-text-primary">
                    {spread.predicted_spread > 0 ? '+' : ''}
                    {spread.predicted_spread.toFixed(1)}
                  </div>
                  <div className="text-sm text-text-muted mt-1">
                    {spread.predicted_spread < 0
                      ? `${homeAbbrev} by ${Math.abs(spread.predicted_spread).toFixed(1)}`
                      : spread.predicted_spread > 0
                      ? `${awayAbbrev} by ${spread.predicted_spread.toFixed(1)}`
                      : 'Even matchup'}
                  </div>
                </div>

                {/* Team indicators */}
                <div className="flex justify-between text-sm">
                  <div
                    className={cn(
                      'px-3 py-1.5 rounded',
                      spread.predicted_spread < 0
                        ? 'bg-success-light text-accent-success font-medium'
                        : 'bg-bg-tertiary text-text-secondary'
                    )}
                  >
                    {homeAbbrev} {spread.predicted_spread < 0 ? 'favored' : ''}
                  </div>
                  <div
                    className={cn(
                      'px-3 py-1.5 rounded',
                      spread.predicted_spread > 0
                        ? 'bg-success-light text-accent-success font-medium'
                        : 'bg-bg-tertiary text-text-secondary'
                    )}
                  >
                    {awayAbbrev} {spread.predicted_spread > 0 ? 'favored' : ''}
                  </div>
                </div>

                {/* Confidence */}
                <div className="pt-2 border-t border-border flex items-center justify-between">
                  <span className="text-sm text-text-muted">Model Confidence</span>
                  <ConfidenceIndicator confidence={spread.confidence * 100} />
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
