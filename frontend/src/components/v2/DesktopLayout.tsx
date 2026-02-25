import type { ReactNode } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Home,
  TrendingUp,
  BarChart3,
  Settings,
  Zap,
  ChevronRight,
  Activity,
  FileText,
} from 'lucide-react';
import type { BankrollData } from './BankrollSummary';

interface DesktopLayoutProps {
  children: ReactNode;
  bankroll?: BankrollData;
  activePage?: 'dashboard' | 'predictions' | 'performance' | 'health' | 'briefing' | 'settings';
}

// Format currency for display
function formatCurrency(value: number, showSign = false): string {
  const absValue = Math.abs(value);
  const formatted = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(absValue);

  if (showSign) {
    return value >= 0 ? `+${formatted}` : `-${formatted}`;
  }
  return value >= 0 ? formatted : `-${formatted}`;
}

// Page titles mapping
const PAGE_TITLES: Record<string, string> = {
  dashboard: 'Dashboard',
  predictions: 'Predictions',
  performance: 'Performance',
  health: 'System Health',
  briefing: 'Daily Briefing',
  settings: 'Settings',
};

/**
 * Desktop Layout with Slim Sidebar Rail
 *
 * Features:
 * - Slim vertical navigation rail
 * - Expandable on hover (optional)
 * - Top header with bankroll summary
 */
export function DesktopLayout({ children, bankroll, activePage = 'dashboard' }: DesktopLayoutProps) {
  return (
    <div className="min-h-screen bg-bg-primary flex">
      {/* Sidebar Rail */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header */}
        <TopHeader bankroll={bankroll} activePage={activePage} />

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}

const navItems = [
  { to: '/', icon: Home, label: 'Dashboard' },
  { to: '/predictions', icon: TrendingUp, label: 'Predictions' },
  { to: '/performance', icon: BarChart3, label: 'Performance' },
  { to: '/health', icon: Activity, label: 'System Health' },
  { to: '/briefing', icon: FileText, label: 'Briefing' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

function Sidebar() {
  const location = useLocation();

  return (
    <aside className="w-16 hover:w-56 transition-all duration-300 bg-bg-secondary border-r border-border flex flex-col group overflow-hidden">
      {/* Logo */}
      <div className="h-16 flex items-center px-4 border-b border-border">
        <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center flex-shrink-0">
          <Zap className="w-5 h-5 text-white" />
        </div>
        <span className="ml-3 font-bold text-lg text-text-primary whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          The Oracle
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4">
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
                flex items-center px-4 py-3 mx-2 rounded-lg
                transition-all duration-200
                ${
                  isActive
                    ? 'bg-[rgba(0,212,255,0.1)] text-[#00d4ff]'
                    : 'text-text-muted hover:text-text-primary hover:bg-bg-hover'
                }
              `}
            >
              <Icon
                className={`w-5 h-5 flex-shrink-0 ${
                  isActive ? 'drop-shadow-[0_0_8px_rgba(0,212,255,0.5)]' : ''
                }`}
              />
              <span className="ml-3 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                {label}
              </span>
              {isActive && (
                <ChevronRight className="w-4 h-4 ml-auto opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Expand hint */}
      <div className="p-4 border-t border-border">
        <div className="text-xs text-text-muted text-center opacity-0 group-hover:opacity-100 transition-opacity">
          Hover to expand
        </div>
      </div>
    </aside>
  );
}

function TopHeader({ bankroll, activePage }: { bankroll?: BankrollData; activePage: string }) {
  const todayPnL = bankroll?.todayPnL ?? 0;
  const totalBankroll = bankroll?.totalBankroll ?? 0;
  const allTimeROI = bankroll?.allTimeROI ?? 0;
  const isPnLPositive = todayPnL >= 0;
  const isROIPositive = allTimeROI >= 0;

  return (
    <header className="h-16 bg-bg-secondary border-b border-border flex items-center justify-between px-6">
      {/* Page Title - Dynamic based on activePage */}
      <div>
        <h1 className="text-xl font-semibold text-text-primary">
          {PAGE_TITLES[activePage] || 'Dashboard'}
        </h1>
        <p className="text-sm text-text-muted">
          {new Date().toLocaleDateString('en-US', {
            weekday: 'long',
            month: 'short',
            day: 'numeric',
          })}
        </p>
      </div>

      {/* Bankroll Summary */}
      <div className="flex items-center gap-6">
        {/* Today's P&L */}
        <div className="text-right">
          <div className="text-xs text-text-muted uppercase tracking-wide mb-0.5">
            Today's P&L
          </div>
          <div className={`text-lg font-bold ${isPnLPositive ? 'text-[#00ff88] text-glow-green' : 'text-[#ff3355]'}`}>
            {formatCurrency(todayPnL, true)}
          </div>
        </div>

        <div className="w-px h-10 bg-border" />

        {/* Total Bankroll */}
        <div className="text-right">
          <div className="text-xs text-text-muted uppercase tracking-wide mb-0.5">
            Bankroll
          </div>
          <div className="text-lg font-bold text-text-primary">
            {formatCurrency(totalBankroll)}
          </div>
        </div>

        <div className="w-px h-10 bg-border" />

        {/* ROI */}
        <div className="text-right">
          <div className="text-xs text-text-muted uppercase tracking-wide mb-0.5">
            All-Time ROI
          </div>
          <div className={`text-lg font-bold ${isROIPositive ? 'text-[#00ff88]' : 'text-[#ff3355]'}`}>
            {isROIPositive ? '+' : ''}{allTimeROI.toFixed(1)}%
          </div>
        </div>
      </div>
    </header>
  );
}
