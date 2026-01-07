# Implementation Plan: V2 Redesign

**Objective**: Rebuild the frontend to be mobile-optimized, visually stunning, and directive ("tell users exactly which bets to take").

## Tech Stack
- **Framework**: React 19 + Vite (Existing)
- **Styling**: Tailwind CSS 4 (Existing)
- **Icons**: Lucide React
- **Animations**: Framer Motion (Recommended to add) or plain CSS Transitions.

## Architecture Changes
1.  **Layout**: Switch from a dedicated sidebar (desktop-centric) to a tailored **Responsive Layout**.
    - Mobile: Bottom Tab Bar (Home, Bets, Portfolio, Account).
    - Desktop: Slim vertical rail.
2.  **Navigation**: Flatten the hierarchy. Home is the "Feed" of best bets.

## Component Strategy
### 1. The `BetCard`
This is the core atom of the new design.
- **Design logic**: High contrast.
- **Header**: Team Logos + Time.
- **Body**: The PICK (e.g., "Lakers -5.5") in large typography.
- **Footer**: Verification badges (Model Confidence %, EV %, Sportsbook Logo).
- **Background**: subtle gradient indicative of "Hotness".

### 2. The `ConfidenceGauge`
- A visual ring or bar component that turns Green/Fire when confidence is > 60%.

## Execution Steps (for Claude Code)

### Step 1: Clean Slate
- Rename/Move old components if necessary, or start fresh in `src/components/v2`.
- Update `index.css` with new CSS Variables for the "Premium Theme".

### Step 2: Foundation
- Create `src/components/layout/MobileLayout.tsx`.
- Create `src/components/layout/DesktopLayout.tsx`.
- Implement responsive switching in `App.tsx`.

### Step 3: The "Feed"
- Build the `Dashboard` page to consume predictions.
- **Crucial**: Sort predictions by `edge` or `confidence` descending.
- Display the Top 3 bets as "Featured Cards".
- Display the rest as a dense list.

### Step 4: Verification
- Verify the "TAKE" logic. Ensure the UI clearly distinguishes between a "Lean" and a "Bet".
- Test on Mobile viewport (Chrome DevTools).

## "Claude Code" Instructions
- Read `redesign_v2/tasks.md`.
- Mark tasks as `[x]` when done.
- Focus on *visual impact*. If it looks boring, iterate.
