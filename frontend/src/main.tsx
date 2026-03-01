import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
// V2 Premium Design - "The Oracle"
import AppV2 from './AppV2.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppV2 />
  </StrictMode>,
)

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  });
}
