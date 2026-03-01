import { Component, StrictMode } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
// V2 Premium Design - "The Oracle"
import AppV2 from './AppV2.tsx'

declare global {
  interface Window {
    __oracleTimeout: ReturnType<typeof setTimeout> | undefined;
    __oracleNuke: () => void;
  }
}

// --- Error Boundary (inline to avoid module load failure) ---
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class AppErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[Oracle] React error boundary caught:', error, info);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  handleNuke = () => {
    window.__oracleNuke();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: '100vh', display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          background: '#09090b', color: '#fafafa',
          fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace",
          padding: '1rem',
        }}>
          <h1 style={{ color: '#00d4ff', fontSize: '1.5rem', letterSpacing: '0.3em', textTransform: 'uppercase' as const }}>
            The Oracle
          </h1>
          <p style={{ color: '#a1a1aa', marginBottom: '0.5rem' }}>Something went wrong.</p>
          <pre style={{
            background: '#18181b', color: '#ef4444', padding: '1rem', borderRadius: '0.5rem',
            maxWidth: '90vw', overflow: 'auto', fontSize: '0.75rem', marginBottom: '1.5rem',
          }}>
            {this.state.error?.message || 'Unknown error'}
          </pre>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button onClick={this.handleRetry} style={{
              background: '#00d4ff', color: '#09090b', border: 'none', padding: '0.75rem 1.5rem',
              borderRadius: '0.5rem', fontWeight: 600, cursor: 'pointer',
            }}>
              Try Again
            </button>
            <button onClick={this.handleNuke} style={{
              background: '#27272a', color: '#fafafa', border: '1px solid #3f3f46',
              padding: '0.75rem 1.5rem', borderRadius: '0.5rem', fontWeight: 600, cursor: 'pointer',
            }}>
              Clear Cache &amp; Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// --- Dismiss the HTML loader ---
function dismissLoader() {
  if (window.__oracleTimeout) {
    clearTimeout(window.__oracleTimeout);
    window.__oracleTimeout = undefined;
  }
  const loader = document.getElementById('oracle-loader');
  if (loader) {
    loader.classList.add('fade-out');
    setTimeout(() => loader.remove(), 400);
  }
}

// --- Mount app ---
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppErrorBoundary>
      <AppV2 />
    </AppErrorBoundary>
  </StrictMode>,
)

dismissLoader();

// --- Service worker registration with periodic update checks ---
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((reg) => {
        // Check for SW updates every 30 minutes
        setInterval(() => reg.update(), 30 * 60 * 1000);
      })
      .catch(() => {});
  });
}
