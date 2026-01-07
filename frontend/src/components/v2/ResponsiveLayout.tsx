import type { ReactNode } from 'react';
import { useState, useEffect } from 'react';
import { MobileLayout } from './MobileLayout';
import { DesktopLayout } from './DesktopLayout';
import type { BankrollData } from './BankrollSummary';

export interface ResponsiveLayoutProps {
  children: ReactNode;
  bankroll?: BankrollData;
  activePage?: 'dashboard' | 'predictions' | 'performance' | 'settings';
}

/**
 * Responsive Layout Wrapper
 *
 * Automatically switches between MobileLayout and DesktopLayout
 * based on viewport width.
 *
 * Breakpoint: 768px (md in Tailwind)
 */
export function ResponsiveLayout({ children, bankroll, activePage }: ResponsiveLayoutProps) {
  const [isMobile, setIsMobile] = useState(() => {
    if (typeof window === 'undefined') return true;
    return window.innerWidth < 768;
  });

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    // Set initial value
    handleResize();

    // Listen for resize events
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  if (isMobile) {
    return <MobileLayout bankroll={bankroll}>{children}</MobileLayout>;
  }

  return <DesktopLayout bankroll={bankroll} activePage={activePage}>{children}</DesktopLayout>;
}
