import { cn, formatEdge, getEdgeColor, hasStrongEdge } from '../../lib/utils';
import type { PropPrediction } from '../../lib/types';

interface EdgeBadgeProps {
  prop: PropPrediction;
  showPct?: boolean;
}

export function EdgeBadge({ prop, showPct = false }: EdgeBadgeProps) {
  const isStrong = hasStrongEdge(prop);
  const edgeValue = showPct ? `${formatEdge(prop.edge_pct)}%` : formatEdge(prop.edge);

  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium',
        getEdgeColor(prop.edge),
        isStrong && prop.edge > 0 && 'bg-success-light',
        isStrong && prop.edge < 0 && 'bg-danger-light'
      )}
    >
      {edgeValue}
    </span>
  );
}
