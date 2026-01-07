interface ConfidenceMeterProps {
  value: number; // 0-100
  size?: 'xs' | 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  animate?: boolean;
  className?: string;
}

/**
 * ConfidenceMeter - Visual confidence indicator
 *
 * A circular gauge that displays model confidence.
 * Color changes based on confidence level:
 * - Green (70%+): High confidence
 * - Cyan (55-69%): Medium confidence
 * - Orange (45-54%): Low confidence
 * - Red (<45%): Very low confidence
 */
export function ConfidenceMeter({
  value,
  size = 'md',
  showLabel = false,
  animate = true,
  className = '',
}: ConfidenceMeterProps) {
  // Clamp value between 0-100
  const clampedValue = Math.max(0, Math.min(100, value));

  // Determine color based on confidence
  const getColor = () => {
    if (clampedValue >= 70) return { main: '#00ff88', glow: 'rgba(0, 255, 136, 0.3)' };
    if (clampedValue >= 55) return { main: '#00d4ff', glow: 'rgba(0, 212, 255, 0.3)' };
    if (clampedValue >= 45) return { main: '#ff8800', glow: 'rgba(255, 136, 0, 0.3)' };
    return { main: '#ff3355', glow: 'rgba(255, 51, 85, 0.3)' };
  };

  const color = getColor();

  // Size configurations
  const sizeConfig = {
    xs: { outer: 24, stroke: 3, fontSize: 'text-[8px]' },
    sm: { outer: 32, stroke: 3, fontSize: 'text-xs' },
    md: { outer: 48, stroke: 4, fontSize: 'text-sm' },
    lg: { outer: 64, stroke: 5, fontSize: 'text-base' },
  };

  const config = sizeConfig[size];
  const radius = (config.outer - config.stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (clampedValue / 100) * circumference;

  return (
    <div
      className={`relative inline-flex items-center justify-center ${className}`}
      style={{ width: config.outer, height: config.outer }}
    >
      <svg
        width={config.outer}
        height={config.outer}
        className="transform -rotate-90"
      >
        {/* Background circle */}
        <circle
          cx={config.outer / 2}
          cy={config.outer / 2}
          r={radius}
          fill="none"
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth={config.stroke}
        />

        {/* Progress circle */}
        <circle
          cx={config.outer / 2}
          cy={config.outer / 2}
          r={radius}
          fill="none"
          stroke={color.main}
          strokeWidth={config.stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          style={{
            transition: animate ? 'stroke-dashoffset 0.5s ease-out' : 'none',
            filter: `drop-shadow(0 0 6px ${color.glow})`,
          }}
        />
      </svg>

      {/* Center label */}
      {showLabel && (
        <div
          className={`absolute inset-0 flex items-center justify-center ${config.fontSize} font-bold`}
          style={{ color: color.main }}
        >
          {clampedValue}
        </div>
      )}
    </div>
  );
}

/**
 * ConfidenceBar - Linear confidence indicator
 *
 * Alternative to the circular meter for horizontal layouts.
 */
interface ConfidenceBarProps {
  value: number;
  height?: number;
  showLabel?: boolean;
  className?: string;
}

export function ConfidenceBar({
  value,
  height = 6,
  showLabel = false,
  className = '',
}: ConfidenceBarProps) {
  const clampedValue = Math.max(0, Math.min(100, value));

  const getColor = () => {
    if (clampedValue >= 70) return '#00ff88';
    if (clampedValue >= 55) return '#00d4ff';
    if (clampedValue >= 45) return '#ff8800';
    return '#ff3355';
  };

  const color = getColor();

  return (
    <div className={`w-full ${className}`}>
      {showLabel && (
        <div className="flex justify-between text-xs mb-1">
          <span className="text-text-muted">Confidence</span>
          <span className="font-mono font-semibold" style={{ color }}>
            {clampedValue}%
          </span>
        </div>
      )}
      <div
        className="w-full rounded-full bg-[rgba(255,255,255,0.1)] overflow-hidden"
        style={{ height }}
      >
        <div
          className="h-full rounded-full transition-all duration-500 ease-out"
          style={{
            width: `${clampedValue}%`,
            backgroundColor: color,
            boxShadow: `0 0 10px ${color}40`,
          }}
        />
      </div>
    </div>
  );
}

/**
 * ConfidenceGauge - Large, detailed gauge for feature sections
 */
interface ConfidenceGaugeProps {
  value: number;
  label?: string;
  sublabel?: string;
  className?: string;
}

export function ConfidenceGauge({
  value,
  label = 'Confidence',
  sublabel,
  className = '',
}: ConfidenceGaugeProps) {
  const clampedValue = Math.max(0, Math.min(100, value));

  const getColor = () => {
    if (clampedValue >= 70) return { main: '#00ff88', bg: 'rgba(0, 255, 136, 0.1)' };
    if (clampedValue >= 55) return { main: '#00d4ff', bg: 'rgba(0, 212, 255, 0.1)' };
    if (clampedValue >= 45) return { main: '#ff8800', bg: 'rgba(255, 136, 0, 0.1)' };
    return { main: '#ff3355', bg: 'rgba(255, 51, 85, 0.1)' };
  };

  const color = getColor();
  const size = 120;
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (clampedValue / 100) * circumference;

  return (
    <div
      className={`flex flex-col items-center ${className}`}
      style={{ width: size }}
    >
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          {/* Background */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255, 255, 255, 0.08)"
            strokeWidth={strokeWidth}
          />

          {/* Progress */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color.main}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            style={{
              transition: 'stroke-dashoffset 0.8s ease-out',
              filter: `drop-shadow(0 0 10px ${color.main}60)`,
            }}
          />
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="text-3xl font-bold"
            style={{ color: color.main }}
          >
            {clampedValue}
          </span>
          <span className="text-xs text-text-muted">%</span>
        </div>
      </div>

      {/* Label */}
      <div className="mt-3 text-center">
        <div className="text-sm font-medium text-text-primary">{label}</div>
        {sublabel && (
          <div className="text-xs text-text-muted">{sublabel}</div>
        )}
      </div>
    </div>
  );
}
