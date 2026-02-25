import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Dashboard, AllPredictions, Performance, Settings, SystemHealth, Briefing } from './pages/v2';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

/**
 * AppV2 - "The Oracle" Premium Betting Terminal
 *
 * New mobile-first design with:
 * - Bottom navigation on mobile
 * - Sidebar on desktop
 * - Premium cyberpunk/fintech aesthetic
 */
function AppV2() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Dashboard / Home */}
          <Route path="/" element={<Dashboard />} />

          {/* All Predictions with filtering */}
          <Route path="/predictions" element={<AllPredictions />} />

          {/* Performance tracking */}
          <Route path="/performance" element={<Performance />} />

          {/* System Health */}
          <Route path="/health" element={<SystemHealth />} />

          {/* Daily Briefing */}
          <Route path="/briefing" element={<Briefing />} />

          {/* Settings & Strategy */}
          <Route path="/settings" element={<Settings />} />

          {/* Fallback redirect */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default AppV2;
