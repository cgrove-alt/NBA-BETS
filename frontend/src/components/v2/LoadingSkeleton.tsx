/**
 * Loading Skeleton Components
 *
 * Placeholder content while data is loading.
 * Uses shimmer animation for visual feedback.
 */


interface SkeletonProps {
  className?: string;
}

/**
 * Base Skeleton - Customizable rectangle
 */
export function Skeleton({ className = '' }: SkeletonProps) {
  return (
    <div
      className={`animate-shimmer rounded-lg ${className}`}
      aria-hidden="true"
    />
  );
}

/**
 * Text Skeleton - Line of text
 */
export function SkeletonText({
  lines = 1,
  className = '',
}: {
  lines?: number;
  className?: string;
}) {
  return (
    <div className={`space-y-2 ${className}`}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="animate-shimmer h-4 rounded"
          style={{ width: i === lines - 1 && lines > 1 ? '70%' : '100%' }}
        />
      ))}
    </div>
  );
}

/**
 * BetCard Skeleton - Placeholder for BetCard
 */
export function BetCardSkeleton({ variant = 'compact' }: { variant?: 'featured' | 'compact' | 'list' }) {
  if (variant === 'featured') {
    return (
      <div className="bg-bg-card border border-border rounded-xl p-6 animate-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <Skeleton className="w-14 h-14 rounded-full" />
              <Skeleton className="w-4 h-4 rounded" />
              <Skeleton className="w-14 h-14 rounded-full" />
            </div>
            <div>
              <Skeleton className="w-48 h-5 mb-2" />
              <Skeleton className="w-24 h-4" />
            </div>
          </div>
          <Skeleton className="w-24 h-6 rounded-full" />
        </div>

        {/* Pick */}
        <div className="mb-6">
          <Skeleton className="w-20 h-4 mb-2" />
          <Skeleton className="w-64 h-12 mb-2" />
          <Skeleton className="w-16 h-6" />
        </div>

        {/* Stats */}
        <div className="flex gap-6 mb-6">
          <div className="flex items-center gap-3">
            <Skeleton className="w-16 h-16 rounded-full" />
            <div>
              <Skeleton className="w-16 h-3 mb-1" />
              <Skeleton className="w-12 h-5" />
            </div>
          </div>
          <Skeleton className="w-px h-12" />
          <div>
            <Skeleton className="w-10 h-3 mb-1" />
            <Skeleton className="w-16 h-6" />
          </div>
        </div>

        {/* Button */}
        <Skeleton className="w-full h-12 rounded-lg" />
      </div>
    );
  }

  if (variant === 'list') {
    return (
      <div className="flex items-center gap-4 p-3 bg-bg-card border border-border rounded-lg animate-fade-in">
        <div className="flex items-center gap-1.5 min-w-[80px]">
          <Skeleton className="w-6 h-6 rounded-full" />
          <Skeleton className="w-3 h-3 rounded" />
          <Skeleton className="w-6 h-6 rounded-full" />
        </div>
        <div className="flex-1">
          <Skeleton className="w-32 h-4 mb-1" />
          <Skeleton className="w-12 h-3" />
        </div>
        <Skeleton className="w-8 h-8 rounded-full" />
        <Skeleton className="w-10 h-4" />
        <Skeleton className="w-16 h-6 rounded-full" />
        <Skeleton className="w-16 h-8 rounded-lg" />
      </div>
    );
  }

  // Compact variant
  return (
    <div className="bg-bg-card border border-border rounded-xl p-4 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Skeleton className="w-8 h-8 rounded-full" />
          <Skeleton className="w-3 h-3 rounded" />
          <Skeleton className="w-8 h-8 rounded-full" />
        </div>
        <Skeleton className="w-16 h-4" />
      </div>

      {/* Pick */}
      <div className="mb-3">
        <Skeleton className="w-12 h-3 mb-1" />
        <Skeleton className="w-full h-6 mb-1" />
        <Skeleton className="w-10 h-4" />
      </div>

      {/* Stats */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Skeleton className="w-8 h-8 rounded-full" />
          <Skeleton className="w-8 h-4" />
        </div>
        <Skeleton className="w-14 h-6 rounded-full" />
      </div>

      {/* Button */}
      <Skeleton className="w-full h-9 rounded-lg" />
    </div>
  );
}

/**
 * Stat Card Skeleton - Placeholder for stat cards
 */
export function StatCardSkeleton() {
  return (
    <div className="bg-bg-card border border-border rounded-xl p-4 animate-fade-in">
      <div className="flex items-center gap-2 mb-2">
        <Skeleton className="w-5 h-5 rounded" />
        <Skeleton className="w-16 h-3" />
      </div>
      <Skeleton className="w-24 h-8" />
    </div>
  );
}

/**
 * Game Card Skeleton - Placeholder for game list items
 */
export function GameCardSkeleton() {
  return (
    <div className="bg-bg-card border border-border rounded-xl p-4 flex items-center justify-between animate-fade-in">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Skeleton className="w-10 h-10 rounded-full" />
          <Skeleton className="w-4 h-4 rounded" />
          <Skeleton className="w-10 h-10 rounded-full" />
        </div>
        <Skeleton className="w-24 h-4 hidden sm:block" />
      </div>
      <div className="flex items-center gap-2">
        <Skeleton className="w-12 h-4" />
        <Skeleton className="w-4 h-4 rounded" />
      </div>
    </div>
  );
}

// Generate random bar heights once at module level for ChartSkeleton
const CHART_BAR_HEIGHTS = Array.from({ length: 14 }, () => 20 + Math.random() * 60);

/**
 * Chart Skeleton - Placeholder for charts
 */
export function ChartSkeleton({ height = 'h-48' }: { height?: string }) {
  return (
    <div className={`${height} flex items-end justify-between gap-1 animate-fade-in`}>
      {CHART_BAR_HEIGHTS.map((barHeight, i) => (
        <div
          key={i}
          className="flex-1 animate-shimmer rounded-t"
          style={{ height: `${barHeight}%` }}
        />
      ))}
    </div>
  );
}
