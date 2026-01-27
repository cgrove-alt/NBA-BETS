import { cn } from '../../lib/utils';

export type ConfidenceLevel = 'fire' | 'strong' | 'good' | 'moderate' | 'risky';

interface ConfidenceTierProps {
  confidence: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  showPercentage?: boolean;
}

// Thresholds adjusted for model's natural confidence range (50-70%)
// eslint-disable-next-line react-refresh/only-export-components
export function getConfidenceLevel(confidence: number): ConfidenceLevel {
  if (confidence >= 70) return 'fire';    // Top tier for this model
  if (confidence >= 65) return 'strong';
  if (confidence >= 60) return 'good';
  if (confidence >= 55) return 'moderate';
  return 'risky';
}

// eslint-disable-next-line react-refresh/only-export-components
export function getConfidenceTierConfig(level: ConfidenceLevel) {
  const configs = {
    fire: {
      label: 'FIRE',
      icon: '🔥',
      bgClass: 'bg-gradient-to-r from-orange-500 to-red-500',
      textClass: 'text-white',
      borderClass: 'border-orange-400',
      glowClass: 'shadow-[0_0_15px_rgba(249,115,22,0.5)]',
    },
    strong: {
      label: 'STRONG',
      icon: '⭐',
      bgClass: 'bg-gradient-to-r from-yellow-500 to-amber-500',
      textClass: 'text-white',
      borderClass: 'border-yellow-400',
      glowClass: 'shadow-[0_0_10px_rgba(234,179,8,0.4)]',
    },
    good: {
      label: 'GOOD',
      icon: '✓',
      bgClass: 'bg-gradient-to-r from-green-500 to-emerald-500',
      textClass: 'text-white',
      borderClass: 'border-green-400',
      glowClass: '',
    },
    moderate: {
      label: 'MODERATE',
      icon: '⚠️',
      bgClass: 'bg-gradient-to-r from-gray-400 to-gray-500',
      textClass: 'text-white',
      borderClass: 'border-gray-400',
      glowClass: '',
    },
    risky: {
      label: 'RISKY',
      icon: '🔻',
      bgClass: 'bg-gray-600',
      textClass: 'text-gray-300',
      borderClass: 'border-gray-500',
      glowClass: '',
    },
  };
  return configs[level];
}

export function ConfidenceTier({
  confidence,
  size = 'md',
  showLabel = true,
  showPercentage = true
}: ConfidenceTierProps) {
  const level = getConfidenceLevel(confidence);
  const config = getConfidenceTierConfig(level);

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs gap-1',
    md: 'px-3 py-1 text-sm gap-1.5',
    lg: 'px-4 py-2 text-base gap-2',
  };

  return (
    <div
      className={cn(
        'inline-flex items-center rounded-full font-bold',
        config.bgClass,
        config.textClass,
        config.glowClass,
        sizeClasses[size]
      )}
    >
      {showPercentage && (
        <span className="font-bold">{Math.round(confidence)}%</span>
      )}
      {showLabel && (
        <>
          <span className="opacity-80">{config.icon}</span>
          <span className={size === 'sm' ? 'hidden sm:inline' : ''}>{config.label}</span>
        </>
      )}
    </div>
  );
}

// Compact badge version for tables
export function ConfidenceBadge({ confidence }: { confidence: number }) {
  const level = getConfidenceLevel(confidence);
  const config = getConfidenceTierConfig(level);

  return (
    <div className="flex flex-col items-center gap-1">
      <span
        className={cn(
          'inline-flex items-center justify-center w-12 h-6 rounded text-xs font-bold',
          config.bgClass,
          config.textClass
        )}
      >
        {Math.round(confidence)}%
      </span>
      <span className="text-[10px] text-text-muted uppercase tracking-wide">
        {config.label}
      </span>
    </div>
  );
}
