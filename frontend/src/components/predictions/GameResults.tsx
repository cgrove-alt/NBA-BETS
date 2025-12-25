import { Trophy, TrendingUp, TrendingDown, Minus, Check, X } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { cn } from '../../lib/utils';
import type { GameResults as GameResultsType, PlayerResult } from '../../lib/types';

interface GameResultsProps {
  results: GameResultsType;
}

export function GameResultsDisplay({ results }: GameResultsProps) {
  if (results.status !== 'completed') {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Trophy className="text-accent-warning" size={18} />
            <CardTitle>Game Results</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6 text-text-muted">
            <p>{results.message || 'Results not available'}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { final_score, moneyline_result, player_results, summary } = results;

  return (
    <div className="space-y-4">
      {/* Final Score Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Trophy className="text-accent-warning" size={18} />
            <CardTitle>Final Score</CardTitle>
            {summary && summary.total_picks > 0 && (
              <span className={cn(
                'ml-auto text-sm font-medium px-2 py-0.5 rounded',
                summary.hit_rate >= 0.55 ? 'bg-success-light text-accent-success' : 'bg-danger-light text-accent-danger'
              )}>
                {(summary.hit_rate * 100).toFixed(0)}% Hit Rate
              </span>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {final_score && (
            <div className="flex justify-center items-center gap-8 py-4">
              <div className="text-center">
                <div className="text-lg font-medium text-text-secondary">{final_score.away_team}</div>
                <div className={cn(
                  'text-4xl font-bold',
                  final_score.away_score > final_score.home_score ? 'text-accent-success' : 'text-text-primary'
                )}>
                  {final_score.away_score}
                </div>
              </div>
              <div className="text-text-muted text-lg">@</div>
              <div className="text-center">
                <div className="text-lg font-medium text-text-secondary">{final_score.home_team}</div>
                <div className={cn(
                  'text-4xl font-bold',
                  final_score.home_score > final_score.away_score ? 'text-accent-success' : 'text-text-primary'
                )}>
                  {final_score.home_score}
                </div>
              </div>
            </div>
          )}

          {/* Moneyline Result */}
          {moneyline_result && (
            <div className="mt-4 pt-4 border-t border-border">
              <div className="flex items-center justify-between text-sm">
                <span className="text-text-muted">Moneyline Prediction</span>
                <div className="flex items-center gap-2">
                  <span className="text-text-secondary">
                    Predicted: {moneyline_result.predicted_winner === 'home'
                      ? final_score?.home_team
                      : final_score?.away_team}
                  </span>
                  {moneyline_result.correct ? (
                    <span className="flex items-center gap-1 text-accent-success font-medium">
                      <Check size={16} /> Correct
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-accent-danger font-medium">
                      <X size={16} /> Wrong
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Player Props Results */}
      {player_results.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <TrendingUp className="text-accent-primary" size={18} />
              <CardTitle>Player Props vs Actual</CardTitle>
              {summary && (
                <span className="ml-auto text-sm text-text-muted">
                  {summary.total_hits}/{summary.total_picks} picks hit
                </span>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-text-muted">
                    <th className="text-left py-2 pr-4">Player</th>
                    <th className="text-center px-2">Prop</th>
                    <th className="text-center px-2">Line</th>
                    <th className="text-center px-2">Predicted</th>
                    <th className="text-center px-2">Actual</th>
                    <th className="text-center px-2">Pick</th>
                    <th className="text-center pl-2">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {player_results.map((result) => (
                    <ResultRow
                      key={`${result.player_id}-${result.prop_type}`}
                      result={result}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* No predictions message */}
      {player_results.length === 0 && (
        <Card>
          <CardContent>
            <div className="text-center py-6 text-text-muted">
              <p>No player prop predictions were recorded for this game.</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function ResultRow({ result }: { result: PlayerResult }) {
  const hitColor = result.hit === true
    ? 'text-accent-success'
    : result.hit === false
      ? 'text-accent-danger'
      : 'text-text-muted';

  return (
    <tr className="border-b border-border/50 hover:bg-bg-secondary/50">
      <td className="py-2 pr-4">
        <span className="font-medium">{result.player_name}</span>
        <span className="text-text-muted ml-2 text-xs">{result.team}</span>
      </td>
      <td className="text-center px-2">{result.prop_type}</td>
      <td className="text-center px-2">{result.line?.toFixed(1) ?? '-'}</td>
      <td className="text-center px-2">{result.predicted.toFixed(1)}</td>
      <td className="text-center px-2 font-bold">{result.actual.toFixed(1)}</td>
      <td className="text-center px-2">
        {result.pick === 'OVER' ? (
          <TrendingUp className="inline text-accent-success" size={16} />
        ) : result.pick === 'UNDER' ? (
          <TrendingDown className="inline text-accent-danger" size={16} />
        ) : (
          <Minus className="inline text-text-muted" size={16} />
        )}
      </td>
      <td className={cn('text-center pl-2 font-bold', hitColor)}>
        {result.hit === true ? (
          <span className="flex items-center justify-center gap-1">
            <Check size={14} /> HIT
          </span>
        ) : result.hit === false ? (
          <span className="flex items-center justify-center gap-1">
            <X size={14} /> MISS
          </span>
        ) : (
          '-'
        )}
      </td>
    </tr>
  );
}
