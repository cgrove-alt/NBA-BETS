import { cn } from '../../lib/utils';

interface StatCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  variant?: 'default' | 'success' | 'warning' | 'danger';
  className?: string;
}

const variantClasses = {
  default: 'text-text-primary',
  success: 'text-accent-success',
  warning: 'text-accent-warning',
  danger: 'text-accent-danger',
};

export function StatCard({ label, value, subValue, variant = 'default', className }: StatCardProps) {
  return (
    <div className={cn('bg-bg-tertiary rounded-lg p-3', className)}>
      <p className="text-xs text-text-muted uppercase tracking-wide mb-1">{label}</p>
      <p className={cn('text-xl font-bold', variantClasses[variant])}>{value}</p>
      {subValue && <p className="text-xs text-text-secondary mt-0.5">{subValue}</p>}
    </div>
  );
}
