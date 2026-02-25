import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  TrendingUp,
  Calendar,
  Target,
  DollarSign,
  Activity,
  Loader2,
} from 'lucide-react';
import { ResponsiveLayout } from '../../components/v2/ResponsiveLayout';
import { Card } from '../../components/v2/Card';
import { Button } from '../../components/v2/Button';
import { Badge } from '../../components/v2/Badge';
import { ConfidenceBar } from '../../components/v2/ConfidenceMeter';
import { fetchPerformance } from '../../lib/api';
import { useBankroll } from '../../hooks/useBankroll';

// Time range options
type TimeRange = '7d' | '30d' | '90d' | 'all';

const DAYS_MAP: Record<TimeRange, number> = {
  '7d': 7,
  '30d': 30,
  '90d': 90,
  'all': 365,
};

/**
 * Performance - Track betting performance over time
 */
export function Performance() {
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const { bankrollData } = useBankroll();

  const { data: perfData, isLoading } = useQuery({
    queryKey: ['performance', timeRange],
    queryFn: () => fetchPerformance(DAYS_MAP[timeRange]),
    staleTime: 5 * 60 * 1000,
  });

  const dailyRecords = perfData?.daily_records || [];
  const byPropType = perfData?.by_prop_type || {};
  const totalPnL = dailyRecords.reduce((sum, d) => sum + d.profit, 0);
  const totalBets = perfData?.total_bets || 0;
  const winRate = perfData?.overall_hit_rate || 0;
  const roi = perfData?.overall_roi || 0;

  return (
    <ResponsiveLayout bankroll={bankrollData} activePage="performance">
      <div className="space-y-6 pb-20 md:pb-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">Performance</h1>
            <p className="text-sm text-text-muted mt-1">Track your betting results</p>
          </div>
        </div>

        {/* Time Range Selector */}
        <div className="flex gap-2 overflow-x-auto pb-2 -mx-4 px-4 md:mx-0 md:px-0">
          {(['7d', '30d', '90d', 'all'] as TimeRange[]).map((range) => (
            <Button
              key={range}
              variant={timeRange === range ? 'primary' : 'ghost'}
              size="sm"
              onClick={() => setTimeRange(range)}
              className="whitespace-nowrap"
            >
              {range === 'all' ? 'All Time' : range.replace('d', ' Days')}
            </Button>
          ))}
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-text-muted" />
          </div>
        ) : (
          <>
            {/* Key Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard
                label="Total P&L"
                value={`${totalPnL >= 0 ? '+' : ''}$${Math.abs(totalPnL).toFixed(0)}`}
                icon={<DollarSign className="w-5 h-5" />}
                variant={totalPnL >= 0 ? 'success' : 'danger'}
              />
              <MetricCard
                label="Win Rate"
                value={`${winRate.toFixed(1)}%`}
                icon={<Target className="w-5 h-5" />}
                variant={winRate >= 52.4 ? 'success' : 'default'}
              />
              <MetricCard
                label="ROI"
                value={`${roi >= 0 ? '+' : ''}${roi.toFixed(1)}%`}
                icon={<TrendingUp className="w-5 h-5" />}
                variant={roi >= 0 ? 'success' : 'danger'}
              />
              <MetricCard
                label="Total Bets"
                value={totalBets}
                icon={<Activity className="w-5 h-5" />}
                variant="default"
              />
            </div>

            {/* P&L Chart */}
            <Card>
              <div className="p-4 border-b border-border">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-text-primary">Profit & Loss</h3>
                  <Badge variant={totalPnL >= 0 ? 'success' : 'danger'}>
                    {totalPnL >= 0 ? '+' : ''}${Math.abs(totalPnL).toFixed(0)}
                  </Badge>
                </div>
              </div>
              <div className="p-4">
                {dailyRecords.length > 0 ? (
                  <PnLChart data={dailyRecords} />
                ) : (
                  <div className="h-48 flex items-center justify-center text-text-muted">
                    No data available for this period
                  </div>
                )}
              </div>
            </Card>

            {/* Win Rate by Prop Type */}
            <Card>
              <div className="p-4 border-b border-border">
                <h3 className="font-semibold text-text-primary">Win Rate by Prop Type</h3>
              </div>
              <div className="p-4 space-y-4">
                {Object.entries(byPropType).length > 0 ? (
                  Object.entries(byPropType).map(([type, stats]) => (
                    <div key={type}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-text-primary">{type}</span>
                        <span className="text-sm text-text-muted">{stats.total} bets</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="flex-1">
                          <ConfidenceBar value={stats.hit_rate} height={8} />
                        </div>
                        <span
                          className={`text-sm font-mono font-semibold w-14 text-right ${
                            stats.hit_rate >= 52.4 ? 'text-[#00ff88]' : 'text-[#ff8800]'
                          }`}
                        >
                          {stats.hit_rate.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-text-muted py-4">No prop type data yet</div>
                )}
              </div>
            </Card>

            {/* Recent Results */}
            <Card>
              <div className="p-4 border-b border-border">
                <h3 className="font-semibold text-text-primary">Recent Days</h3>
              </div>
              <div className="divide-y divide-border">
                {dailyRecords.slice(0, 7).map((day) => {
                  const dayTotal = day.wins + day.losses + day.pushes;
                  const dayWinRate = dayTotal > 0 ? ((day.wins / dayTotal) * 100).toFixed(0) : '0';
                  return (
                    <div key={day.date} className="p-4 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Calendar className="w-4 h-4 text-text-muted" />
                        <span className="text-sm text-text-primary">
                          {formatDate(day.date)}
                        </span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-sm text-text-muted">
                          {day.wins}/{dayTotal} ({dayWinRate}%)
                        </span>
                        <span
                          className={`text-sm font-mono font-semibold ${
                            day.profit >= 0 ? 'text-[#00ff88]' : 'text-[#ff3355]'
                          }`}
                        >
                          {day.profit >= 0 ? '+' : ''}${day.profit.toFixed(0)}
                        </span>
                      </div>
                    </div>
                  );
                })}
                {dailyRecords.length === 0 && (
                  <div className="p-4 text-center text-text-muted">No results yet</div>
                )}
              </div>
            </Card>

            {/* Calibration Summary */}
            {perfData?.calibration_summary && (
              <Card>
                <div className="p-4 border-b border-border">
                  <h3 className="font-semibold text-text-primary">Calibration</h3>
                </div>
                <div className="p-4 grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-text-muted uppercase tracking-wider mb-1">Hit Rate</div>
                    <div className="text-lg font-bold text-text-primary">
                      {perfData.calibration_summary.overall_hit_rate != null
                        ? `${(perfData.calibration_summary.overall_hit_rate * 100).toFixed(1)}%`
                        : 'N/A'}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-text-muted uppercase tracking-wider mb-1">CLV Avg</div>
                    <div className="text-lg font-bold text-text-primary">
                      {perfData.calibration_summary.overall_clv != null
                        ? `${perfData.calibration_summary.overall_clv >= 0 ? '+' : ''}${perfData.calibration_summary.overall_clv.toFixed(2)}`
                        : 'N/A'}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-text-muted uppercase tracking-wider mb-1">ECE</div>
                    <div className="text-lg font-bold text-text-primary">
                      {perfData.calibration_summary.ece != null
                        ? perfData.calibration_summary.ece.toFixed(3)
                        : 'N/A'}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-text-muted uppercase tracking-wider mb-1">Matched</div>
                    <div className="text-lg font-bold text-text-primary">
                      {perfData.calibration_summary.matched_predictions}/{perfData.calibration_summary.total_predictions}
                    </div>
                  </div>
                </div>
              </Card>
            )}
          </>
        )}
      </div>
    </ResponsiveLayout>
  );
}

/**
 * Metric Card
 */
function MetricCard({
  label,
  value,
  icon,
  variant = 'default',
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  variant?: 'default' | 'success' | 'danger';
}) {
  const variantStyles = {
    default: 'text-text-primary',
    success: 'text-[#00ff88]',
    danger: 'text-[#ff3355]',
  };

  return (
    <Card>
      <div className="p-4">
        <div className="flex items-center gap-2 text-text-muted mb-2">
          {icon}
          <span className="text-xs uppercase tracking-wider">{label}</span>
        </div>
        <div className={`text-2xl font-bold ${variantStyles[variant]}`}>{value}</div>
      </div>
    </Card>
  );
}

/**
 * P&L Chart - Simple bar chart showing daily P&L
 */
function PnLChart({ data }: { data: { date: string; profit: number }[] }) {
  const maxProfit = Math.max(...data.map((d) => Math.abs(d.profit)), 1);
  const chartData = data.slice(0, 14).reverse();

  return (
    <div className="h-48">
      <div className="flex items-end justify-between h-full gap-1">
        {chartData.map((day, i) => {
          const height = (Math.abs(day.profit) / maxProfit) * 100;
          const isPositive = day.profit >= 0;

          return (
            <div key={day.date} className="flex-1 flex flex-col items-center justify-end h-full">
              <div
                className={`w-full rounded-t transition-all duration-300 ${
                  isPositive ? 'bg-[#00ff88]' : 'bg-[#ff3355]'
                }`}
                style={{
                  height: `${Math.max(height, 5)}%`,
                  opacity: 0.8,
                  boxShadow: isPositive
                    ? '0 0 10px rgba(0, 255, 136, 0.3)'
                    : '0 0 10px rgba(255, 51, 85, 0.3)',
                }}
              />
              {(i % 3 === 0 || chartData.length <= 7) && (
                <div className="text-[10px] text-text-muted mt-1 truncate max-w-full">
                  {formatShortDate(day.date)}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr + 'T12:00:00');
  return date.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });
}

function formatShortDate(dateStr: string): string {
  const date = new Date(dateStr + 'T12:00:00');
  return date.toLocaleDateString('en-US', {
    month: 'numeric',
    day: 'numeric',
  });
}
