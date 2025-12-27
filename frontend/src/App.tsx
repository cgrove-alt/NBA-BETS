import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/layout/Layout';
import { Predictions } from './pages/Predictions';
import { Tracker } from './pages/Tracker';
import { Performance } from './pages/Performance';
import { Bankroll } from './pages/Bankroll';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Predictions />} />
            <Route path="/tracker" element={<Tracker />} />
            <Route path="/performance" element={<Performance />} />
            <Route path="/bankroll" element={<Bankroll />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
