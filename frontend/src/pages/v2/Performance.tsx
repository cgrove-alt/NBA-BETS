import { useState } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Calendar,
  Target,
  DollarSign,
  Activity,
} from 'lucide-react';
import { ResponsiveLayout } from '../../components/v2/ResponsiveLayout';
import type { BankrollData } from '../../components/v2/BankrollSummary';
import { Card } from '../../components/v2/Card';
import { Button } from '../../components/v2/Button';
import { Badge } from '../../components/v2/Badge';
import { ConfidenceBar } from '../../components/v2/ConfidenceMeter';

// Time range options
type TimeRange = '7d' | '30d' | '90d' | 'all';

interface DailyResult {
  date: string;
  profit: number;
  bets: number;
  wins: number;
}

/**
 * Performance - Track betting performance over time
 *
 * Features:
 * - P&L Chart (line/bar)
 * - Win rate by prop type
 * - ROI metrics
 * - Streak tracking
 */
export function Performance() {
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');

  // Mock performance data
  const performanceData: DailyResult[] = generateMockData(timeRange);
  const metrics = calculateMetrics(performanceData);

  // Mock bankroll data
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

  // Win rate by prop type (mock data)
  const propTypeStats = [
    { type: 'Points', winRate: 62.5, bets: 48 },
    { type: 'Rebounds', winRate: 55.8, bets: 36 },
    { type: 'Assists', winRate: 58.3, bets: 42 },
    { type: '3PM', winRate: 51.2, bets: 28 },
    { type: 'PRA', winRate: 60.0, bets: 35 },
  ];

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

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="Total P&L"
            value={`$${metrics.totalPnL.toFixed(0)}`}
            change={metrics.pnlChange}
            changeLabel="vs prev period"
            icon={<DollarSign className="w-5 h-5" />}
            variant={metrics.totalPnL >= 0 ? 'success' : 'danger'}
          />
          <MetricCard
            label="Win Rate"
            value={`${metrics.winRate.toFixed(1)}%`}
            change={metrics.winRateChange}
            changeLabel="vs prev period"
            icon={<Target className="w-5 h-5" />}
            variant={metrics.winRate >= 52.4 ? 'success' : 'default'}
          />
          <MetricCard
            label="ROI"
            value={`${metrics.roi >= 0 ? '+' : ''}${metrics.roi.toFixed(1)}%`}
            icon={<TrendingUp className="w-5 h-5" />}
            variant={metrics.roi >= 0 ? 'success' : 'danger'}
          />
          <MetricCard
            label="Total Bets"
            value={metrics.totalBets}
            icon={<Activity className="w-5 h-5" />}
            variant="default"
          />
        </div>

        {/* P&L Chart */}
        <Card>
          <div className="p-4 border-b border-border">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-text-primary">Profit & Loss</h3>
              <Badge variant={metrics.totalPnL >= 0 ? 'success' : 'danger'}>
                {metrics.totalPnL >= 0 ? '+' : ''}${metrics.totalPnL.toFixed(0)}
              </Badge>
            </div>
          </div>
          <div className="p-4">
            <PnLChart data={performanceData} />
          </div>
        </Card>

        {/* Win Rate by Prop Type */}
        <Card>
          <div className="p-4 border-b border-border">
            <h3 className="font-semibold text-text-primary">Win Rate by Prop Type</h3>
          </div>
          <div className="p-4 space-y-4">
            {propTypeStats.map((stat) => (
              <div key={stat.type}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-text-primary">{stat.type}</span>
                  <span className="text-sm text-text-muted">{stat.bets} bets</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex-1">
                    <ConfidenceBar value={stat.winRate} height={8} />
                  </div>
                  <span
                    className={`text-sm font-mono font-semibold w-14 text-right ${
                      stat.winRate >= 52.4 ? 'text-[#00ff88]' : 'text-[#ff8800]'
                    }`}
                  >
                    {stat.winRate.toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Recent Results */}
        <Card>
          <div className="p-4 border-b border-border">
            <h3 className="font-semibold text-text-primary">Recent Days</h3>
          </div>
          <div className="divide-y divide-border">
            {performanceData.slice(0, 7).map((day) => (
              <div key={day.date} className="p-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Calendar className="w-4 h-4 text-text-muted" />
                  <span className="text-sm text-text-primary">
                    {formatDate(day.date)}
                  </span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-sm text-text-muted">
                    {day.wins}/{day.bets} ({((day.wins / day.bets) * 100).toFixed(0)}%)
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
            ))}
          </div>
        </Card>

        {/* Streak Info */}
        <div className="grid grid-cols-2 gap-4">
          <Card>
            <div className="p-4">
              <div className="flex items-center gap-2 text-text-muted mb-2">
                <TrendingUp className="w-4 h-4 text-[#00ff88]" />
                <span className="text-xs uppercase tracking-wider">Best Streak</span>
              </div>
              <div className="text-2xl font-bold text-[#00ff88]">7 Wins</div>
              <div className="text-xs text-text-muted mt-1">Dec 15-18</div>
            </div>
          </Card>
          <Card>
            <div className="p-4">
              <div className="flex items-center gap-2 text-text-muted mb-2">
                <TrendingDown className="w-4 h-4 text-[#ff3355]" />
                <span className="text-xs uppercase tracking-wider">Current</span>
              </div>
              <div className="text-2xl font-bold text-text-primary">3 Wins</div>
              <div className="text-xs text-text-muted mt-1">Active streak</div>
            </div>
          </Card>
        </div>
      </div>
    </ResponsiveLayout>
  );
}

/**
 * Metric Card - Displays a single performance metric
 */
function MetricCard({
  label,
  value,
  change,
  changeLabel,
  icon,
  variant = 'default',
}: {
  label: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
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
        {change !== undefined && (
          <div className="flex items-center gap-1 mt-1">
            {change >= 0 ? (
              <TrendingUp className="w-3 h-3 text-[#00ff88]" />
            ) : (
              <TrendingDown className="w-3 h-3 text-[#ff3355]" />
            )}
            <span
              className={`text-xs ${change >= 0 ? 'text-[#00ff88]' : 'text-[#ff3355]'}`}
            >
              {change >= 0 ? '+' : ''}
              {change.toFixed(1)}%
            </span>
            {changeLabel && (
              <span className="text-xs text-text-muted">{changeLabel}</span>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}

/**
 * P&L Chart - Simple bar chart showing daily P&L
 */
function PnLChart({ data }: { data: DailyResult[] }) {
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
              {/* Bar */}
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
              {/* Label (every few bars on mobile) */}
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

// Helper functions
function generateMockData(range: TimeRange): DailyResult[] {
  const days = range === '7d' ? 7 : range === '30d' ? 30 : range === '90d' ? 90 : 180;
  const results: DailyResult[] = [];

  for (let i = 0; i < days; i++) {
    const date = new Date();
    date.setDate(date.getDate() - i);

    const bets = Math.floor(Math.random() * 8) + 3;
    const winRate = 0.52 + Math.random() * 0.15;
    const wins = Math.round(bets * winRate);
    const profit = (wins - (bets - wins)) * (50 + Math.random() * 50) - 10 * bets;

    results.push({
      date: date.toISOString().split('T')[0],
      profit,
      bets,
      wins,
    });
  }

  return results;
}

function calculateMetrics(data: DailyResult[]) {
  const totalPnL = data.reduce((sum, d) => sum + d.profit, 0);
  const totalBets = data.reduce((sum, d) => sum + d.bets, 0);
  const totalWins = data.reduce((sum, d) => sum + d.wins, 0);
  const winRate = totalBets > 0 ? (totalWins / totalBets) * 100 : 0;
  const roi = totalBets > 0 ? (totalPnL / (totalBets * 100)) * 100 : 0;

  // Mock changes
  const pnlChange = Math.random() * 20 - 5;
  const winRateChange = Math.random() * 4 - 1;

  return {
    totalPnL,
    totalBets,
    totalWins,
    winRate,
    roi,
    pnlChange,
    winRateChange,
  };
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });
}

function formatShortDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', {
    month: 'numeric',
    day: 'numeric',
  });
}
