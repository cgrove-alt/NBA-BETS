import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  FileText,
  Calendar,
  Loader2,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { ResponsiveLayout } from '../../components/v2/ResponsiveLayout';
import { Card } from '../../components/v2/Card';
import { Button } from '../../components/v2/Button';
import { Badge } from '../../components/v2/Badge';
import { fetchBriefing } from '../../lib/api';
import { useBankroll } from '../../hooks/useBankroll';

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
