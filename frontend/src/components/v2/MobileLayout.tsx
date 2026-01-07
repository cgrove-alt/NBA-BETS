import type { ReactNode } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Home,
  TrendingUp,
  BarChart3,
  Settings,
  Zap,
} from 'lucide-react';
import type { BankrollData } from './BankrollSummary';

interface MobileLayoutProps {
  children: ReactNode;
  bankroll?: BankrollData;
}

// Format currency for display
function formatCurrency(value: number, showSign = false): string {
  const absValue = Math.abs(value);
  const formatted = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: absValue < 100 ? 2 : 0,
  }).format(absValue);

  if (showSign) {
    return value >= 0 ? `+${formatted}` : `-${formatted}`;
  }
  return value >= 0 ? formatted : `-${formatted}`;
}

/**
 * Mobile-First Layout with Bottom Navigation
 *
 * Features:
 * - Bottom tab bar for mobile (thumb-friendly)
 * - Fixed header with bankroll summary
 * - Safe area support for notched devices
 */
export function MobileLayout({ children, bankroll }: MobileLayoutProps) {
  return (
    <div className="min-h-screen bg-bg-primary flex flex-col">
      {/* Header */}
      <Header bankroll={bankroll} />

      {/* Main Content - with bottom padding for nav */}
      <main className="flex-1 pb-nav overflow-y-auto px-4 pt-4">{children}</main>

      {/* Bottom Navigation - Mobile Only */}
      <BottomNav />
    </div>
  );
}

function Header({ bankroll }: { bankroll?: BankrollData }) {
  const todayPnL = bankroll?.todayPnL ?? 0;
  const totalBankroll = bankroll?.totalBankroll ?? 0;
  const isPnLPositive = todayPnL >= 0;

  return (
    <header className="glass-strong sticky top-0 z-40 safe-top">
      <div className="flex items-center justify-between px-4 py-3">
        {/* Logo / Brand */}
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-lg text-text-primary">
            The Oracle
          </span>
        </div>

        {/* Quick Bankroll Display */}
        <div className="flex items-center gap-3">
          <div className="text-right">
            <div className="text-xs text-text-muted uppercase tracking-wide">
              Today
            </div>
            <div className={`text-sm font-semibold ${isPnLPositive ? 'text-[#00ff88]' : 'text-[#ff3355]'}`}>
              {formatCurrency(todayPnL, true)}
            </div>
          </div>
          <div className="w-px h-8 bg-border" />
          <div className="text-right">
            <div className="text-xs text-text-muted uppercase tracking-wide">
              Bankroll
            </div>
            <div className="text-sm font-semibold text-text-primary">
              {formatCurrency(totalBankroll)}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

const navItems = [
  { to: '/', icon: Home, label: 'Home' },
  { to: '/predictions', icon: TrendingUp, label: 'Picks' },
  { to: '/performance', icon: BarChart3, label: 'Stats' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

function BottomNav() {
  const location = useLocation();

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 glass-strong safe-bottom md:hidden">
      <div className="flex items-center justify-around px-2 py-2">
        {navItems.map(({ to, icon: Icon, label }) => {
          const isActive =
            to === '/'
              ? location.pathname === '/'
              : location.pathname.startsWith(to);

          return (
            <NavLink
              key={to}
              to={to}
              className={`
                flex flex-col items-center justify-center
                w-16 h-14 rounded-xl
                transition-all duration-200
                touch-target
                ${
                  isActive
                    ? 'text-[#00d4ff] bg-[rgba(0,212,255,0.1)]'
                    : 'text-text-muted hover:text-text-primary'
                }
              `}
            >
              <Icon
                className={`w-5 h-5 mb-1 ${isActive ? 'drop-shadow-[0_0_8px_rgba(0,212,255,0.5)]' : ''}`}
              />
              <span className="text-[10px] font-medium">{label}</span>
            </NavLink>
          );
        })}
      </div>
    </nav>
  );
}
