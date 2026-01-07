import type { ReactNode } from 'react';

interface BadgeProps {
  children: ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'premium';
  size?: 'sm' | 'md' | 'lg';
  glow?: boolean;
  pulse?: boolean;
  className?: string;
  onClick?: () => void;
}

/**
 * Premium Badge Component
 *
 * Used for status indicators, confidence levels, and edge values.
 * Supports glow effects and pulse animations.
 */
export function Badge({
  children,
  variant = 'default',
  size = 'md',
  glow = false,
  pulse = false,
  className = '',
  onClick,
}: BadgeProps) {
  const baseStyles = `
    inline-flex items-center justify-center
    font-semibold rounded-full
    border
  `;

  const variantStyles = {
    default: `
      bg-bg-tertiary
      text-text-secondary
      border-border
    `,
    success: `
      bg-[rgba(0,255,136,0.15)]
      text-[#00ff88]
      border-[rgba(0,255,136,0.3)]
    `,
    warning: `
      bg-[rgba(255,136,0,0.15)]
      text-[#ff8800]
      border-[rgba(255,136,0,0.3)]
    `,
    danger: `
      bg-[rgba(255,51,85,0.15)]
      text-[#ff3355]
      border-[rgba(255,51,85,0.3)]
    `,
    info: `
      bg-[rgba(0,212,255,0.15)]
      text-[#00d4ff]
      border-[rgba(0,212,255,0.3)]
    `,
    premium: `
      bg-gradient-to-r from-[rgba(255,215,0,0.2)] to-[rgba(255,215,0,0.1)]
      text-[#ffd700]
      border-[rgba(255,215,0,0.4)]
    `,
  };

  const sizeStyles = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base',
  };

  const glowStyles = glow
    ? {
        default: '',
        success: 'shadow-[0_0_10px_rgba(0,255,136,0.3)]',
        warning: 'shadow-[0_0_10px_rgba(255,136,0,0.3)]',
        danger: 'shadow-[0_0_10px_rgba(255,51,85,0.3)]',
        info: 'shadow-[0_0_10px_rgba(0,212,255,0.3)]',
        premium: 'shadow-[0_0_15px_rgba(255,215,0,0.4)]',
      }[variant]
    : '';

  const pulseStyles = pulse ? 'animate-pulse-glow' : '';

  const clickableStyles = onClick ? 'cursor-pointer hover:opacity-80 transition-opacity' : '';

  return (
    <span
      className={`${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${glowStyles} ${pulseStyles} ${clickableStyles} ${className}`}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => e.key === 'Enter' && onClick() : undefined}
    >
      {children}
    </span>
  );
}

/**
 * Edge Badge - Specifically for displaying betting edge values
 */
interface EdgeBadgeProps {
  edge: number; // Percentage value (e.g., 15.5 for +15.5%)
  size?: 'sm' | 'md' | 'lg';
  showSign?: boolean;
  className?: string;
}

export function EdgeBadge({
  edge,
  size = 'md',
  showSign = true,
  className = '',
}: EdgeBadgeProps) {
  const isPositive = edge > 0;
  const isStrong = Math.abs(edge) >= 10;

  const sign = showSign && isPositive ? '+' : '';
  const displayValue = `${sign}${edge.toFixed(1)}%`;

  return (
    <Badge
      variant={isPositive ? 'success' : 'danger'}
      size={size}
      glow={isStrong}
      className={`font-mono ${className}`}
    >
      {displayValue}
    </Badge>
  );
}

/**
 * Confidence Badge - For displaying model confidence
 */
interface ConfidenceBadgeProps {
  confidence: number; // 0-100
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function ConfidenceBadge({
  confidence,
  size = 'md',
  className = '',
}: ConfidenceBadgeProps) {
  const getVariant = () => {
    if (confidence >= 70) return 'success';
    if (confidence >= 55) return 'info';
    if (confidence >= 45) return 'warning';
    return 'danger';
  };

  return (
    <Badge
      variant={getVariant()}
      size={size}
      glow={confidence >= 65}
      className={`font-mono ${className}`}
    >
      {confidence.toFixed(0)}%
    </Badge>
  );
}

/**
 * Status Badge - For live/upcoming/final game states
 */
interface StatusBadgeProps {
  status: 'live' | 'upcoming' | 'final' | 'delayed';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function StatusBadge({ status, size = 'sm', className = '' }: StatusBadgeProps) {
  const config = {
    live: { label: 'LIVE', variant: 'danger' as const, pulse: true },
    upcoming: { label: 'UPCOMING', variant: 'info' as const, pulse: false },
    final: { label: 'FINAL', variant: 'default' as const, pulse: false },
    delayed: { label: 'DELAYED', variant: 'warning' as const, pulse: true },
  };

  const { label, variant, pulse } = config[status];

  return (
    <Badge variant={variant} size={size} pulse={pulse} className={`uppercase tracking-wider ${className}`}>
      {status === 'live' && (
        <span className="w-2 h-2 bg-current rounded-full mr-1.5 animate-pulse" />
      )}
      {label}
    </Badge>
  );
}

/**
 * Sentiment Chip - For model signals like "Sharp Money", "Model Convergence"
 */
interface SentimentChipProps {
  label: string;
  type: 'positive' | 'negative' | 'neutral';
  icon?: ReactNode;
  className?: string;
}

export function SentimentChip({
  label,
  type,
  icon,
  className = '',
}: SentimentChipProps) {
  const variantMap = {
    positive: 'success',
    negative: 'danger',
    neutral: 'default',
  } as const;

  return (
    <Badge variant={variantMap[type]} size="sm" className={`gap-1 ${className}`}>
      {icon}
      {label}
    </Badge>
  );
}
