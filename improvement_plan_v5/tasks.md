# Improvement Plan V5: Tasks

## Phase 1: Backend Repairs (The Data Pipeline)
- [x] **Unblock Data**: Edit `dashboard/data_service.py` to remove the strict "DraftKings" check. Allow `rebet`, `fanduel`, `betmgm`, etc.
- [ ] **Tune Filters**: Edit `backend/api.py` to set `min_confidence=55.0` and `min_edge=4.0` (Honest settings).
- [ ] **Deploy Backend**: Execute deployment to Railway (ensure changes are pushed).

## Phase 2: Frontend Repairs (The UI)
- [ ] **Make Cards Clickable**: Edit `src/components/v2/GameCard.tsx` to add `onClick` navigation to `/game/:id`.
- [ ] **Deploy Frontend**: Execute `vercel --prod` to push the UI fixes.

## Phase 3: Verification
- [ ] **Live Data Check**: Visit `https://nba-bets.vercel.app/` and confirm "Top Bets" list is populated.
- [ ] **Interaction Check**: Click a game card and verify it opens the game details page.
