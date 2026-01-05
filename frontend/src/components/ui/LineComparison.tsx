import { cn } from '../../lib/utils';

interface LineComparisonProps {
  line: number | null | undefined;
  prediction: number;
  propType: string;
  size?: 'sm' | 'md' | 'lg';
}

export function LineComparison({ line, prediction, propType, size = 'md' }: LineComparisonProps) {
  // Calculate edge
  const edge = line ? prediction - line : 0;
  const hasLine = line !== null && line !== undefined && line > 0;

  // Calculate position on visual scale (0-100%)
  // Scale from line-5 to line+5 for visualization
  const range = 10;
  const min = hasLine ? line - range / 2 : prediction - range / 2;
  const max = hasLine ? line + range / 2 : prediction + range / 2;

  const linePosition = hasLine ? ((line - min) / (max - min)) * 100 : 50;
  const predPosition = ((prediction - min) / (max - min)) * 100;

  // Clamp positions between 5% and 95%
  const clampedLinePos = Math.max(5, Math.min(95, linePosition));
  const clampedPredPos = Math.max(5, Math.min(95, predPosition));

  const isOver = edge > 0;

  const sizeConfig = {
    sm: {
      height: 'h-2',
      text: 'text-xs',
      spacing: 'mb-1',
    },
    md: {
      height: 'h-3',
      text: 'text-sm',
      spacing: 'mb-2',
    },
    lg: {
      height: 'h-4',
      text: 'text-base',
      spacing: 'mb-3',
    },
  };

  const config = sizeConfig[size];

  if (!hasLine) {
    return (
      <div className="text-text-muted text-sm">
        <span className="font-medium text-text-primary">{prediction.toFixed(1)}</span>
        <span className="ml-1 text-xs">(no line)</span>
      </div>
    );
  }

  return (
    <div className="w-full">
      {/* Labels row */}
      <div className={cn('flex justify-between items-center', config.spacing, config.text)}>
        <div className="text-text-secondary">
          <span className="text-text-muted">Line:</span>{' '}
          <span className="font-medium text-text-primary">{line.toFixed(1)}</span>
        </div>
        <div className="text-text-secondary">
          <span className="text-text-muted">Pred:</span>{' '}
          <span className={cn(
            'font-bold',
            isOver ? 'text-green-400' : 'text-red-400'
          )}>
            {prediction.toFixed(1)}
          </span>
          <span className={cn(
            'ml-1 font-medium',
            isOver ? 'text-green-400' : 'text-red-400'
          )}>
            ({isOver ? '+' : ''}{edge.toFixed(1)})
          </span>
        </div>
      </div>

      {/* Visual bar */}
      <div className="relative">
        <div className={cn(
          'w-full rounded-full bg-gray-700',
          config.height
        )}>
          {/* Fill from line to prediction */}
          <div
            className={cn(
              'absolute top-0 rounded-full',
              config.height,
              isOver ? 'bg-green-500/40' : 'bg-red-500/40'
            )}
            style={{
              left: `${Math.min(clampedLinePos, clampedPredPos)}%`,
              width: `${Math.abs(clampedPredPos - clampedLinePos)}%`,
            }}
          />
        </div>

        {/* Line marker */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-1 h-5 bg-gray-400 rounded"
          style={{ left: `${clampedLinePos}%`, marginLeft: '-2px' }}
        />

        {/* Prediction marker */}
        <div
          className={cn(
            'absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full border-2',
            isOver
              ? 'bg-green-500 border-green-300'
              : 'bg-red-500 border-red-300'
          )}
          style={{ left: `${clampedPredPos}%`, marginLeft: '-6px' }}
        />
      </div>

      {/* Direction indicator */}
      <div className={cn(
        'text-center mt-1',
        config.text
      )}>
        <span className={cn(
          'font-medium',
          isOver ? 'text-green-400' : 'text-red-400'
        )}>
          {isOver ? '▲' : '▼'} {Math.abs(edge).toFixed(1)} {propType.toLowerCase()} {isOver ? 'above' : 'below'} line
        </span>
      </div>
    </div>
  );
}

// Compact inline version for tables
export function LineComparisonInline({ line, prediction }: { line: number | null | undefined; prediction: number }) {
  const edge = line ? prediction - line : 0;
  const hasLine = line !== null && line !== undefined && line > 0;
  const isOver = edge > 0;

  return (
    <div className="flex flex-col">
      <span className={cn(
        'font-bold',
        isOver ? 'text-green-400' : 'text-red-400'
      )}>
        {prediction.toFixed(1)}
      </span>
      {hasLine && (
        <span className={cn(
          'text-xs font-medium',
          isOver ? 'text-green-400/80' : 'text-red-400/80'
        )}>
          {isOver ? '+' : ''}{edge.toFixed(1)} vs {line.toFixed(1)}
        </span>
      )}
    </div>
  );
}
