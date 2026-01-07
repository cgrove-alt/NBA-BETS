import type { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  variant?: 'default' | 'glass' | 'elevated' | 'success' | 'danger' | 'gold';
  glow?: boolean;
  hover?: boolean;
  onClick?: () => void;
}

/**
 * Premium Card Component
 *
 * The foundation for all card-based UI elements.
 * Supports glassmorphism, glow effects, and hover states.
 */
export function Card({
  children,
  className = '',
  variant = 'default',
  glow = false,
  hover = true,
  onClick,
}: CardProps) {
  const baseStyles = 'rounded-xl border transition-all duration-200';

  const variantStyles = {
    default: 'bg-bg-card border-border',
    glass: 'glass-card',
    elevated: 'bg-bg-secondary border-border shadow-elevated',
    success: 'bg-bg-card border-[rgba(0,255,136,0.3)] gradient-card-success',
    danger: 'bg-bg-card border-[rgba(255,51,85,0.3)] gradient-card-danger',
    gold: 'bg-bg-card border-[rgba(255,215,0,0.3)] gradient-card-gold',
  };

  const glowStyles = {
    default: glow ? 'glow-primary' : '',
    glass: glow ? 'glow-primary' : '',
    elevated: glow ? 'glow-primary' : '',
    success: glow ? 'glow-success' : '',
    danger: glow ? 'glow-danger' : '',
    gold: glow ? 'glow-gold' : '',
  };

  const hoverStyles = hover
    ? 'hover:border-[rgba(255,255,255,0.1)] hover:shadow-card-hover hover:-translate-y-0.5'
    : '';

  const clickableStyles = onClick ? 'cursor-pointer' : '';

  return (
    <div
      className={`${baseStyles} ${variantStyles[variant]} ${glowStyles[variant]} ${hoverStyles} ${clickableStyles} ${className}`}
      onClick={onClick}
    >
      {children}
    </div>
  );
}

interface CardHeaderProps {
  children: ReactNode;
  className?: string;
}

export function CardHeader({ children, className = '' }: CardHeaderProps) {
  return (
    <div className={`px-4 py-3 border-b border-border ${className}`}>
      {children}
    </div>
  );
}

interface CardBodyProps {
  children: ReactNode;
  className?: string;
}

export function CardBody({ children, className = '' }: CardBodyProps) {
  return <div className={`p-4 ${className}`}>{children}</div>;
}

interface CardFooterProps {
  children: ReactNode;
  className?: string;
}

export function CardFooter({ children, className = '' }: CardFooterProps) {
  return (
    <div className={`px-4 py-3 border-t border-border ${className}`}>
      {children}
    </div>
  );
}
