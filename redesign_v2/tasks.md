# Redesign V2 Tasks

## Phase 0: Critical Fixes (Data & API) - COMPLETE
- [x] **Data Pipeline Check**: Backend API verified - CORS already configured for localhost:5173
- [x] **Debug Empty State**: Fixed! Lowered default thresholds in `/api/best-bets`:
    - `min_confidence`: 80% → 50% (model outputs 50-70% range)
    - `min_edge`: 5% → 3%
- [x] **Enable CORS**: Already configured in `backend/api.py` for localhost:5173, localhost:3000, and Vercel
- [x] **Frontend Filter Thresholds**: Updated v2 pages to use realistic thresholds:
    - Dashboard: minConfidence 50%, minEdge 3%
    - AllPredictions presets adjusted for model's output range

## Phase 1: Foundation & Setup - COMPLETE
- [x] Initialize new Design System structure (colors, typography, shadows) in `index.css`
    - Cyberpunk/Fintech color palette with neon accents (green, cyan, purple, orange, red, gold)
    - Glassmorphism utilities (.glass, .glass-card, .glass-strong)
    - Neon glow effects (.glow-success, .glow-primary, .glow-gold, etc.)
    - Text glow effects for emphasis
    - Animation utilities (pulse-glow, float, shimmer)
    - Mobile-first utilities (touch-target, safe-bottom, pb-nav)
- [x] Create core UI components (Card, Button, Badge) with "Premium" aesthetic (Glassmorphism, Neon accents)
    - Card: default, glass, elevated, success, danger, gold variants with glow support
    - Button: primary, secondary, success, danger, ghost, action variants
    - Badge: EdgeBadge, ConfidenceBadge, StatusBadge, SentimentChip
- [x] Set up Mobile-First Layout container (Bottom Navigation for mobile, Sidebar for desktop)
    - MobileLayout: Bottom tab bar, fixed header with bankroll
    - DesktopLayout: Slim expandable sidebar rail, top header
    - ResponsiveLayout: Auto-switches at 768px breakpoint

## Phase 2: Core Components - COMPLETE
- [x] Build `BetCard` component (The Hero component)
    - Three variants: featured (large hero), compact (grid), list (dense rows)
    - Shows: Team logos, matchup, game time, pick type, selection, odds
    - Edge badge with glow for high values
    - Confidence meter integration
    - Signals/sentiment chips
    - "TAKE THIS BET" action button with glow effect
    - Gold variant for top picks (edge >= 15%, confidence >= 60%)
- [x] Build `ConfidenceMeter` visual component
    - ConfidenceMeter: Circular gauge with color-coded fill (green/cyan/orange/red)
    - ConfidenceBar: Linear progress bar alternative
    - ConfidenceGauge: Large detailed gauge for hero sections
    - Animated transitions, glow effects based on confidence level
- [x] Build `BankrollSummary` widget (for header/top view)
    - Three variants: compact (header), full (dashboard grid), minimal (inline)
    - StatCard component for individual metrics
    - PnLTicker animated component
    - Shows: Bankroll, Today's P&L, Win Rate, All-Time ROI

## Phase 3: Pages & Routing - COMPLETE
- [x] Implement `Dashboard` (Home)
    - Featured "Top Pick" hero section with BetCard
    - More Picks grid with compact BetCards
    - Today's Games list with game cards
    - Quick performance stats (Week P&L, Win Rate)
    - Full bankroll overview on mobile
- [x] Implement `AllPredictions` with filtering (Safe, High Reward, Whale Plays)
    - Filter presets: All, Safe Bets, High Reward, Whale Plays
    - Grid/List view toggle
    - Prop type filtering
    - Dynamic filter descriptions
- [x] Implement `Performance` view (Charts/Graphs)
    - Key metrics: Total P&L, Win Rate, ROI, Total Bets
    - Simple bar chart for daily P&L
    - Win rate breakdown by prop type
    - Recent results list
    - Streak tracking
- [x] Implement `Settings/Strategy` configuration
    - Bankroll management (amount, bet sizing strategy)
    - Prediction filters (min confidence, min edge)
    - Notification toggles
    - App info section
- [x] Set up routing with AppV2.tsx
    - Routes: /, /predictions, /performance, /settings
    - Updated layouts to accept bankroll and activePage props
    - Navigation items updated in MobileLayout and DesktopLayout

## Phase 4: Polish & Refinement - COMPLETE
- [x] Add Micro-animations (CSS transitions) for loading states and interactions
    - Page entrance animations (fade-in, slide-up, scale-in)
    - Staggered list animations for sequential items
    - Button press effect (btn-press) for tactile feedback
    - Button ripple effect (btn-ripple) for visual feedback
    - Card hover lift animation (card-lift)
    - Bounce in animation for popping elements
    - Number pop animation for value changes
    - Success flash animation for confirmations
    - Spin animation for loading spinners
- [x] Ensure Mobile Responsiveness (Touch targets, spacing)
    - All interactive elements meet 44px minimum touch target
    - Button min-heights: sm=36px, md=44px, lg=52px
    - IconButton sizes: sm=36px, md=44px, lg=48px
    - iOS tap highlight color customized
    - Text selection disabled on buttons
    - Touch scrolling optimization
    - Mobile-specific spacing utilities
    - Safe area support for notched devices
- [x] Add Loading Skeleton Components
    - Skeleton base component
    - SkeletonText for text placeholders
    - BetCardSkeleton (featured, compact, list variants)
    - StatCardSkeleton for stat cards
    - GameCardSkeleton for game list items
    - ChartSkeleton for chart placeholders
- [x] Final Review against "Appealing & Intuitive" mandate
    - Premium cyberpunk/fintech aesthetic achieved
    - Neon glow effects for visual hierarchy
    - Glassmorphism for depth and premium feel
    - Directive UX with clear "TAKE THIS BET" CTAs
    - Mobile-first responsive design
    - Intuitive bottom navigation on mobile
    - Expandable sidebar on desktop

---

## Review Summary

### What Was Built
A complete premium UI redesign for "The Oracle" NBA Betting Terminal with:

**Design System (Phase 1)**
- Custom color palette with neon accents (green, cyan, purple, orange, red, gold)
- Glassmorphism utilities for depth
- Glow effects for emphasis
- Mobile-first responsive breakpoints

**Core Components (Phase 2)**
- BetCard with 3 variants (featured, compact, list)
- ConfidenceMeter with circular and linear variants
- BankrollSummary with dynamic data display

**Pages (Phase 3)**
- Dashboard: Hero top pick, games list, performance snapshot
- AllPredictions: Filter presets, grid/list toggle
- Performance: P&L charts, win rate breakdown
- Settings: Strategy configuration, notifications

**Polish (Phase 4)**
- Loading skeletons with shimmer animations
- Page entrance/exit animations
- Touch-optimized targets (44px minimum)
- iOS-specific optimizations

### Key Files Created/Modified
- `frontend/src/components/v2/` - 12 new component files
- `frontend/src/pages/v2/` - 4 new page files
- `frontend/src/AppV2.tsx` - New app entry with v2 routing
- `frontend/src/index.css` - Complete design system rewrite
- `frontend/src/main.tsx` - Updated to use AppV2

### To Run
```bash
cd frontend
npm run dev
```

The app now loads with the v2 premium design at http://localhost:5173/
