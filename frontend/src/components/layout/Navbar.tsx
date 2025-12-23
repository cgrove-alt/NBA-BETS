import { Link, useLocation } from 'react-router-dom';
import { TrendingUp, Target, BarChart2, Wallet } from 'lucide-react';

const navItems = [
  { path: '/', label: 'Predictions', icon: TrendingUp },
  { path: '/tracker', label: 'Tracker', icon: Target },
  { path: '/performance', label: 'Performance', icon: BarChart2 },
  { path: '/bankroll', label: 'Bankroll', icon: Wallet },
];

export function Navbar() {
  const location = useLocation();

  return (
    <nav className="bg-bg-secondary border-b border-border">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-14">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2">
            <span className="text-xl font-bold text-text-primary">NBA Props</span>
          </Link>

          {/* Nav links */}
          <div className="flex items-center gap-1">
            {navItems.map(({ path, label, icon: Icon }) => {
              const isActive = location.pathname === path;
              return (
                <Link
                  key={path}
                  to={path}
                  className={`
                    flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
                    transition-colors duration-150
                    ${
                      isActive
                        ? 'bg-bg-tertiary text-accent-primary'
                        : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
                    }
                  `}
                >
                  <Icon size={16} />
                  <span className="hidden sm:inline">{label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
