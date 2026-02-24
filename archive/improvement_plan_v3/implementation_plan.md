# Implementation Plan V3

## Goal
Restore functionality to `https://nba-bets.vercel.app/` by fixing data generation, date synchronization, and UI interactivity.

## Proposed Changes

### 1. Backend: Relax Data Filters (`dashboard/data_service.py`)
**Problem**: Strict check for 'draftkings' causes 100% data loss when other providers (Rebet) are used.
**Change**:
```python
# Remove or Relax Strict Filter
# Old: if 'draftkings' not in book_name: continue
# New: 
valid_books = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'rebet', 'bovada']
if not any(book in book_name for book in valid_books):
    continue 
# OR just accept all:
# pass
```

### 2. Frontend: Fix Date Logic (`src/hooks/useGames.ts`)
**Problem**: Requesting 2026 dates.
**Change**:
- Verify `getToday()` utils.
- Ensure requests use `new Date().toISOString().split('T')[0]`.

### 3. Frontend: Fix Game Card Interaction (`src/components/v2/GameCard.tsx`)
**Problem**: Cards are not clickable.
**Change**:
- Add `onClick` prop or wrap in `<Link>`.
- Navigate to `/game/{gameId}`.

### 4. Deployment
- **Backend**: `railway up` (to push the data service fix).
- **Frontend**: `vercel --prod` (after verifying date fix).

## Verification Method
1. **Local**: Run `npm run dev` + `python -m uvicorn backend.api:app`.
2. **Check**: Verify "Top Pick" appears (no longer empty).
3. **Live**: Deploy and check Vercel site.
