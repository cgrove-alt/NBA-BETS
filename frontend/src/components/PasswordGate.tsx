import { useState, useEffect, type ReactNode } from 'react';
import { Lock } from 'lucide-react';

interface PasswordGateProps {
  children: ReactNode;
}

const STORAGE_KEY = 'nba-props-authenticated';

export function PasswordGate({ children }: PasswordGateProps) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user is already authenticated
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'true') {
      setIsAuthenticated(true);
    }
    setIsLoading(false);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    const correctPassword = import.meta.env.VITE_SITE_PASSWORD;

    if (!correctPassword) {
      // No password configured - allow access
      setIsAuthenticated(true);
      localStorage.setItem(STORAGE_KEY, 'true');
      return;
    }

    if (password === correctPassword) {
      setIsAuthenticated(true);
      localStorage.setItem(STORAGE_KEY, 'true');
    } else {
      setError('Incorrect password');
      setPassword('');
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-bg-primary flex items-center justify-center">
        <div className="text-text-secondary">Loading...</div>
      </div>
    );
  }

  if (isAuthenticated) {
    return <>{children}</>;
  }

  return (
    <div className="min-h-screen bg-bg-primary flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="bg-bg-card rounded-xl p-8 shadow-card border border-border">
          <div className="flex flex-col items-center mb-6">
            <div className="w-16 h-16 rounded-full bg-bg-tertiary flex items-center justify-center mb-4">
              <Lock className="w-8 h-8 text-accent-primary" />
            </div>
            <h1 className="text-2xl font-bold text-text-primary">NBA Props</h1>
            <p className="text-text-secondary mt-2 text-center">
              Enter password to access
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
                className="w-full px-4 py-3 rounded-lg bg-bg-secondary border border-border text-text-primary placeholder-text-muted focus:outline-none focus:border-accent-primary focus:ring-1 focus:ring-accent-primary transition-colors"
                autoFocus
              />
            </div>

            {error && (
              <div className="text-accent-danger text-sm text-center">
                {error}
              </div>
            )}

            <button
              type="submit"
              className="w-full py-3 px-4 rounded-lg gradient-primary text-white font-medium hover:opacity-90 transition-opacity"
            >
              Enter
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
