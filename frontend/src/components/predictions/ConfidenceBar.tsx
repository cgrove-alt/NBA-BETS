import { cn, getConfidenceGradient, formatConfidence } from '../../lib/utils';

interface ConfidenceBarProps {
  confidence: number;
  showLabel?: boolean;
  size?: 'sm' | 'md';
}

export function ConfidenceBar({ confidence, showLabel = true, size = 'md' }: ConfidenceBarProps) {
  const height = size === 'sm' ? 'h-1.5' : 'h-2';
  const clampedConfidence = Math.min(100, Math.max(0, confidence));

  return (
    <div className="flex items-center gap-2">
      <div className={cn('flex-1 bg-bg-tertiary rounded-full overflow-hidden', height)}>
        <div
          className={cn('h-full rounded-full transition-all duration-300', getConfidenceGradient(confidence))}
          style={{ width: `${clampedConfidence}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-xs text-text-secondary w-10 text-right">
          {formatConfidence(confidence)}
        </span>
      )}
    </div>
  );
}
