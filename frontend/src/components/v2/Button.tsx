import type { ReactNode, ButtonHTMLAttributes } from 'react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'ghost' | 'action';
  size?: 'sm' | 'md' | 'lg';
  fullWidth?: boolean;
  loading?: boolean;
  icon?: ReactNode;
  iconPosition?: 'left' | 'right';
}

/**
 * Premium Button Component
 *
 * Supports multiple variants including the signature "BET THIS" action button.
 * Includes loading states, icons, and responsive sizing.
 */
export function Button({
  children,
  variant = 'primary',
  size = 'md',
  fullWidth = false,
  loading = false,
  icon,
  iconPosition = 'left',
  className = '',
  disabled,
  ...props
}: ButtonProps) {
  const baseStyles = `
    inline-flex items-center justify-center gap-2
    font-semibold rounded-lg
    transition-all duration-200
    touch-target btn-press btn-ripple
    disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
  `;

  const variantStyles = {
    primary: `
      bg-gradient-to-r from-[#0099cc] to-[#00d4ff]
      text-white
      hover:shadow-[0_0_20px_rgba(0,212,255,0.4)]
      hover:-translate-y-0.5
      active:translate-y-0
    `,
    secondary: `
      bg-bg-tertiary
      text-text-primary
      border border-border
      hover:bg-bg-hover
      hover:border-[rgba(255,255,255,0.1)]
    `,
    success: `
      bg-gradient-to-r from-[#00cc6a] to-[#00ff88]
      text-[#09090b]
      hover:shadow-[0_0_20px_rgba(0,255,136,0.4)]
      hover:-translate-y-0.5
      active:translate-y-0
    `,
    danger: `
      bg-gradient-to-r from-[#cc2244] to-[#ff3355]
      text-white
      hover:shadow-[0_0_20px_rgba(255,51,85,0.4)]
      hover:-translate-y-0.5
      active:translate-y-0
    `,
    ghost: `
      bg-transparent
      text-text-secondary
      hover:text-text-primary
      hover:bg-bg-hover
    `,
    action: `
      bg-gradient-to-r from-[#00cc6a] to-[#00ff88]
      text-[#09090b]
      font-bold
      uppercase
      tracking-wider
      hover:shadow-[0_0_30px_rgba(0,255,136,0.5)]
      hover:-translate-y-0.5
      active:translate-y-0
    `,
  };

  const sizeStyles = {
    sm: 'px-3 py-1.5 text-sm min-h-[36px]',
    md: 'px-4 py-2 text-base min-h-[44px]',
    lg: 'px-6 py-3 text-lg min-h-[52px]',
  };

  const widthStyles = fullWidth ? 'w-full' : '';

  return (
    <button
      className={`${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${widthStyles} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <>
          <LoadingSpinner />
          <span>Loading...</span>
        </>
      ) : (
        <>
          {icon && iconPosition === 'left' && icon}
          {children}
          {icon && iconPosition === 'right' && icon}
        </>
      )}
    </button>
  );
}

function LoadingSpinner() {
  return (
    <svg
      className="animate-spin h-4 w-4"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

/**
 * Icon Button - For toolbar actions
 */
interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  icon: ReactNode;
  label: string;
  variant?: 'default' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
}

export function IconButton({
  icon,
  label,
  variant = 'default',
  size = 'md',
  className = '',
  ...props
}: IconButtonProps) {
  const baseStyles = `
    inline-flex items-center justify-center
    rounded-lg transition-all duration-200
    touch-target
  `;

  const variantStyles = {
    default: 'bg-bg-tertiary text-text-secondary hover:text-text-primary hover:bg-bg-hover',
    ghost: 'bg-transparent text-text-secondary hover:text-text-primary hover:bg-bg-hover',
  };

  const sizeStyles = {
    sm: 'w-9 h-9',     // 36px - smaller but still tappable
    md: 'w-11 h-11',   // 44px - standard touch target
    lg: 'w-12 h-12',   // 48px - larger targets
  };

  return (
    <button
      className={`${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
      aria-label={label}
      {...props}
    >
      {icon}
    </button>
  );
}
