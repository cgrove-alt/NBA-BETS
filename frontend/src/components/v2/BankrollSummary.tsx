import { Card } from './Card';
import {
  TrendingUp,
  TrendingDown,
  Wallet,
  Target,
  BarChart3,
  Calendar,
} from 'lucide-react';

export interface BankrollData {
  totalBankroll: number;
  todayPnL: number;
  weekPnL: number;
  monthPnL: number;
  allTimeROI: number;
  winRate: number;
  activeBets: number;
  pendingBets: number;
}

interface BankrollSummaryProps {
  data: BankrollData;
  variant?: 'compact' | 'full' | 'minimal';
  className?: string;
}

/**
 * BankrollSummary - Displays bankroll and performance metrics
 *
 * Variants:
 * - compact: For mobile headers
 * - full: For dashboard hero section
 * - minimal: Just the numbers, no cards
 */
export function BankrollSummary({
  data,
  variant = 'compact',
  className = '',
}: BankrollSummaryProps) {
  if (variant === 'minimal') {
    return <MinimalSummary data={data} className={className} />;
  }

  if (variant === 'full') {
    return <FullSummary data={data} className={className} />;
  }

  return <CompactSummary data={data} className={className} />;
}

/**
 * Compact Summary - For headers
 */
function CompactSummary({
  data,
  className,
}: {
  data: BankrollData;
  className?: string;
}) {
  const isPnLPositive = data.todayPnL >= 0;

  return (
    <div className={`flex items-center gap-4 ${className}`}>
      {/* Today's P&L */}
      <div className="text-right">
        <div className="text-[10px] text-text-muted uppercase tracking-wider">
          Today
        </div>
        <div
          className={`text-sm font-bold ${
            isPnLPositive ? 'text-[#00ff88]' : 'text-[#ff3355]'
          }`}
        >
          {formatCurrency(data.todayPnL, true)}
        </div>
      </div>

      <div className="w-px h-8 bg-border" />

      {/* Bankroll */}
      <div className="text-right">
        <div className="text-[10px] text-text-muted uppercase tracking-wider">
          Bankroll
        </div>
        <div className="text-sm font-bold text-text-primary">
          {formatCurrency(data.totalBankroll)}
        </div>
      </div>
    </div>
  );
}

/**
 * Full Summary - For dashboard
 */
function FullSummary({
  data,
  className,
}: {
  data: BankrollData;
  className?: string;
}) {
  return (
    <div className={`grid grid-cols-2 md:grid-cols-4 gap-4 ${className}`}>
      {/* Total Bankroll */}
      <StatCard
        icon={<Wallet className="w-5 h-5" />}
        label="Bankroll"
        value={formatCurrency(data.totalBankroll)}
        variant="primary"
      />

      {/* Today's P&L */}
      <StatCard
        icon={<Calendar className="w-5 h-5" />}
        label="Today"
        value={formatCurrency(data.todayPnL, true)}
        variant={data.todayPnL >= 0 ? 'success' : 'danger'}
        glow={Math.abs(data.todayPnL) >= 100}
      />

      {/* Win Rate */}
      <StatCard
        icon={<Target className="w-5 h-5" />}
        label="Win Rate"
        value={`${data.winRate.toFixed(1)}%`}
        sublabel={`${data.activeBets} active`}
        variant={data.winRate >= 52.4 ? 'success' : 'default'}
      />

      {/* All-Time ROI */}
      <StatCard
        icon={<BarChart3 className="w-5 h-5" />}
        label="All-Time ROI"
        value={`${data.allTimeROI >= 0 ? '+' : ''}${data.allTimeROI.toFixed(1)}%`}
        variant={data.allTimeROI >= 0 ? 'success' : 'danger'}
      />
    </div>
  );
}

/**
 * Minimal Summary - Just numbers
 */
function MinimalSummary({
  data,
  className,
}: {
  data: BankrollData;
  className?: string;
}) {
  return (
    <div className={`flex items-center gap-6 ${className}`}>
      <Stat
        label="Bankroll"
        value={formatCurrency(data.totalBankroll)}
        size="lg"
      />
      <Stat
        label="Today"
        value={formatCurrency(data.todayPnL, true)}
        variant={data.todayPnL >= 0 ? 'success' : 'danger'}
        size="lg"
      />
      <Stat
        label="ROI"
        value={`${data.allTimeROI >= 0 ? '+' : ''}${data.allTimeROI.toFixed(1)}%`}
        variant={data.allTimeROI >= 0 ? 'success' : 'danger'}
        size="lg"
      />
    </div>
  );
}

/**
 * Stat Card - Individual stat display
 */
interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  sublabel?: string;
  variant?: 'default' | 'primary' | 'success' | 'danger';
  glow?: boolean;
}

function StatCard({
  icon,
  label,
  value,
  sublabel,
  variant = 'default',
  glow = false,
}: StatCardProps) {
  const variantStyles = {
    default: 'text-text-primary',
    primary: 'text-[#00d4ff]',
    success: 'text-[#00ff88]',
    danger: 'text-[#ff3355]',
  };

  const glowStyles = {
    default: '',
    primary: glow ? 'text-glow-cyan' : '',
    success: glow ? 'text-glow-green' : '',
    danger: glow ? 'text-glow-red' : '',
  };

  return (
    <Card variant={variant === 'success' ? 'success' : variant === 'danger' ? 'danger' : 'default'} glow={glow}>
      <div className="p-4">
        <div className="flex items-center gap-2 text-text-muted mb-2">
          {icon}
          <span className="text-xs uppercase tracking-wider">{label}</span>
        </div>
        <div className={`text-2xl font-bold ${variantStyles[variant]} ${glowStyles[variant]}`}>
          {value}
        </div>
        {sublabel && (
          <div className="text-xs text-text-muted mt-1">{sublabel}</div>
        )}
      </div>
    </Card>
  );
}

/**
 * Simple Stat - For inline display
 */
interface StatProps {
  label: string;
  value: string;
  variant?: 'default' | 'success' | 'danger';
  size?: 'sm' | 'md' | 'lg';
}

function Stat({
  label,
  value,
  variant = 'default',
  size = 'md',
}: StatProps) {
  const variantStyles = {
    default: 'text-text-primary',
    success: 'text-[#00ff88]',
    danger: 'text-[#ff3355]',
  };

  const sizeStyles = {
    sm: 'text-sm',
    md: 'text-lg',
    lg: 'text-xl',
  };

  return (
    <div>
      <div className="text-[10px] text-text-muted uppercase tracking-wider mb-0.5">
        {label}
      </div>
      <div className={`font-bold ${variantStyles[variant]} ${sizeStyles[size]}`}>
        {value}
      </div>
    </div>
  );
}

/**
 * PnL Ticker - Animated P&L display
 */
interface PnLTickerProps {
  value: number;
  showIcon?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function PnLTicker({
  value,
  showIcon = true,
  size = 'md',
  className = '',
}: PnLTickerProps) {
  const isPositive = value >= 0;

  const sizeStyles = {
    sm: 'text-sm',
    md: 'text-lg',
    lg: 'text-2xl',
  };

  const iconSize = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5',
  };

  return (
    <div
      className={`
        inline-flex items-center gap-1 font-bold font-mono
        ${isPositive ? 'text-[#00ff88]' : 'text-[#ff3355]'}
        ${isPositive ? 'text-glow-green' : 'text-glow-red'}
        ${sizeStyles[size]}
        ${className}
      `}
    >
      {showIcon &&
        (isPositive ? (
          <TrendingUp className={iconSize[size]} />
        ) : (
          <TrendingDown className={iconSize[size]} />
        ))}
      {formatCurrency(value, true)}
    </div>
  );
}

// Utility functions
function formatCurrency(value: number, showSign = false): string {
  const absValue = Math.abs(value);
  const formatted = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: absValue < 100 ? 2 : 0,
  }).format(absValue);

  if (showSign) {
    return value >= 0 ? `+${formatted}` : `-${formatted}`;
  }

  return value >= 0 ? formatted : `-${formatted}`;
}
