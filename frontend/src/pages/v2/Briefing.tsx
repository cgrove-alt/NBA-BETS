import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  FileText,
  Calendar,
  Loader2,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  Target,
  BarChart3,
  Zap,
} from 'lucide-react';
import { ResponsiveLayout } from '../../components/v2/ResponsiveLayout';
import { Card, CardHeader, CardBody, CardFooter } from '../../components/v2/Card';
import { Button } from '../../components/v2/Button';
import { Badge } from '../../components/v2/Badge';
import { StatCard } from '../../components/ui/StatCard';
import { fetchBriefing } from '../../lib/api';
import { useBankroll } from '../../hooks/useBankroll';
import type { YesterdayRecord } from '../../lib/types';

function getTodayET(): string {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}

/**
 * Briefing - Daily briefing page with date navigation
 */
export function Briefing() {
  const { bankrollData } = useBankroll();
  const [selectedDate, setSelectedDate] = useState<string>(getTodayET());

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['briefing', selectedDate],
    queryFn: () => fetchBriefing(selectedDate),
    staleTime: 2 * 60 * 1000,
    refetchInterval: selectedDate === getTodayET() ? 5 * 60 * 1000 : undefined,
  });

  const isToday = selectedDate === getTodayET();

  const navigateDate = (offset: number) => {
    const d = new Date(selectedDate + 'T12:00:00');
    d.setDate(d.getDate() + offset);
    setSelectedDate(d.toISOString().split('T')[0]);
  };

  return (
    <ResponsiveLayout bankroll={bankrollData} activePage="briefing">
      <div className="space-y-6 pb-20 md:pb-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">Daily Briefing</h1>
            <p className="text-sm text-text-muted mt-1">AI-generated daily summary</p>
          </div>
          {isToday && (
            <Button
              variant="ghost"
              size="sm"
              icon={<RefreshCw className="w-4 h-4" />}
              onClick={() => refetch()}
            >
              Refresh
            </Button>
          )}
        </div>

        {/* Date Navigation */}
        <div className="flex items-center justify-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            icon={<ChevronLeft className="w-4 h-4" />}
            onClick={() => navigateDate(-1)}
          >{''}</Button>
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-[#00d4ff]" />
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              max={getTodayET()}
              className="bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-text-primary text-sm focus:outline-none focus:border-[#00d4ff]"
            />
          </div>
          <Button
            variant="ghost"
            size="sm"
            icon={<ChevronRight className="w-4 h-4" />}
            onClick={() => navigateDate(1)}
            disabled={isToday}
          >{''}</Button>
        </div>

        {isToday && (
          <Badge variant="success" size="sm">
            Today's Briefing
          </Badge>
        )}

        {/* Briefing Content */}
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-text-muted" />
          </div>
        ) : error ? (
          <Card className="p-8 text-center">
            <p className="text-text-muted">Failed to load briefing for {selectedDate}</p>
          </Card>
        ) : data ? (
          <>
            {data.yesterday_record && <YesterdayRecordCard record={data.yesterday_record} />}
            {data.today_preview && <TodayPreviewCard preview={data.today_preview} />}

            {/* Formatted Briefing Text */}
            <Card>
              <div className="p-4 border-b border-border flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-[#00d4ff]" />
                  <h3 className="font-semibold text-text-primary">Briefing</h3>
                </div>
                {data.generated_at && (
                  <span className="text-xs text-text-muted">
                    Generated {new Date(data.generated_at).toLocaleTimeString('en-US', {
                      hour: 'numeric',
                      minute: '2-digit',
                      hour12: true,
                    })}
                  </span>
                )}
              </div>
              <div className="p-4">
                <pre className="whitespace-pre-wrap font-mono text-sm text-text-secondary leading-relaxed">
                  {data.briefing_text}
                </pre>
              </div>
            </Card>

            {/* Structured Sections (if available) */}
            {data.sections && (
              <div className="space-y-4">
                {data.sections.yesterday_results && (
                  <BriefingSection title="Yesterday's Results" content={data.sections.yesterday_results} />
                )}
                {data.sections.today_plays && (
                  <BriefingSection title="Today's Plays" content={data.sections.today_plays} />
                )}
                {data.sections.bankroll && (
                  <BriefingSection title="Bankroll" content={data.sections.bankroll} />
                )}
                {data.sections.alerts && (
                  <BriefingSection title="Alerts" content={data.sections.alerts} />
                )}
                {data.sections.market_intel && (
                  <BriefingSection title="Market Intel" content={data.sections.market_intel} />
                )}
              </div>
            )}
          </>
        ) : (
          <Card className="p-8 text-center">
            <p className="text-text-muted">No briefing available for {selectedDate}</p>
          </Card>
        )}
      </div>
    </ResponsiveLayout>
  );
}

function YesterdayRecordCard({ record }: { record: YesterdayRecord }) {
  const { overall, by_bet_type, by_confidence, clv_summary, date, source } = record;
  const hitRate = overall.hit_rate;
  const variant = hitRate >= 55 ? 'success' : 'danger';
  const hitRateVariant = hitRate >= 55 ? 'success' : hitRate >= 45 ? 'warning' : 'danger';
  const showProfit = source === 'bet_tracking' && overall.profit !== 0;

  const betTypeEntries = Object.entries(by_bet_type);
  const confidenceEntries = Object.entries(by_confidence);

  return (
    <Card variant={variant} glow>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-accent-success" />
            <h3 className="font-semibold text-text-primary">Yesterday's Record</h3>
            <span className="text-xs text-text-muted">{date}</span>
          </div>
          {source && (
            <Badge variant="default" size="sm">{source === 'bet_tracking' ? 'Tracked' : 'Predictions'}</Badge>
          )}
        </div>
      </CardHeader>

      <CardBody>
        {/* Headline Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard
            label="Record"
            value={`${overall.wins}-${overall.losses}-${overall.pushes}`}
          />
          <StatCard
            label="Hit Rate"
            value={`${hitRate.toFixed(1)}%`}
            variant={hitRateVariant}
          />
          <StatCard
            label="Profit"
            value={showProfit ? `$${overall.profit >= 0 ? '+' : ''}${overall.profit.toFixed(0)}` : 'N/A'}
            variant={showProfit ? (overall.profit >= 0 ? 'success' : 'danger') : 'default'}
          />
          <StatCard
            label="Total Bets"
            value={overall.total}
          />
        </div>

        {/* By Bet Type */}
        {betTypeEntries.length > 0 && (
          <div className="mt-4">
            <p className="text-xs text-text-muted uppercase tracking-wide mb-2 flex items-center gap-1">
              <BarChart3 className="w-3 h-3" /> By Bet Type
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {betTypeEntries.map(([type, stats]) => (
                <div key={type} className="bg-bg-tertiary rounded-lg p-2">
                  <p className="text-xs text-text-muted">{type}</p>
                  <p className="text-sm font-bold text-text-primary">
                    {stats.wins}-{stats.losses}
                  </p>
                  <p className={`text-xs ${stats.hit_rate >= 55 ? 'text-accent-success' : stats.hit_rate >= 45 ? 'text-accent-warning' : 'text-accent-danger'}`}>
                    {stats.hit_rate.toFixed(0)}%
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* By Confidence */}
        {confidenceEntries.length > 0 && (
          <div className="mt-4">
            <p className="text-xs text-text-muted uppercase tracking-wide mb-2 flex items-center gap-1">
              <Target className="w-3 h-3" /> By Confidence
            </p>
            <div className="grid grid-cols-3 gap-2">
              {confidenceEntries.map(([tier, stats]) => (
                <div key={tier} className="bg-bg-tertiary rounded-lg p-2">
                  <p className="text-xs text-text-muted capitalize">{tier}</p>
                  <p className="text-sm font-bold text-text-primary">
                    {stats.wins}-{stats.losses}
                  </p>
                  <p className={`text-xs ${stats.hit_rate >= 55 ? 'text-accent-success' : stats.hit_rate >= 45 ? 'text-accent-warning' : 'text-accent-danger'}`}>
                    {stats.hit_rate.toFixed(0)}%
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardBody>

      {/* CLV Summary */}
      {clv_summary && (
        <CardFooter>
          <p className="text-xs text-text-secondary">
            Avg CLV: <span className={clv_summary.avg_clv >= 0 ? 'text-accent-success' : 'text-accent-danger'}>{clv_summary.avg_clv >= 0 ? '+' : ''}{clv_summary.avg_clv.toFixed(1)}%</span>
            {' · '}
            Positive CLV: <span className="text-text-primary">{(clv_summary.positive_clv_rate * 100).toFixed(0)}%</span>
          </p>
        </CardFooter>
      )}
    </Card>
  );
}

function TodayPreviewCard({ preview }: { preview: { actionable_plays: number; games_count: number; games_analyzed?: number } }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-[#00d4ff]" />
          <h3 className="font-semibold text-text-primary">Today's Preview</h3>
        </div>
      </CardHeader>
      <CardBody>
        <div className="grid grid-cols-3 gap-3">
          <StatCard
            label="Actionable Plays"
            value={preview.actionable_plays}
            variant="default"
            className="[&>p:nth-child(2)]:text-[#00d4ff]"
          />
          <StatCard
            label="Games"
            value={preview.games_count}
          />
          {preview.games_analyzed != null && (
            <StatCard
              label="Analyzed"
              value={preview.games_analyzed}
            />
          )}
        </div>
      </CardBody>
    </Card>
  );
}

function BriefingSection({ title, content }: { title: string; content: string }) {
  return (
    <Card>
      <div className="p-4 border-b border-border">
        <h3 className="font-semibold text-text-primary">{title}</h3>
      </div>
      <div className="p-4">
        <pre className="whitespace-pre-wrap font-mono text-sm text-text-secondary leading-relaxed">
          {content}
        </pre>
      </div>
    </Card>
  );
}
