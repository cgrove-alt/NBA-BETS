# Master Plan: Repairs & Features

## Phase 1: Backend Repairs & Features
- [ ] **Unblock Data**: Edit `dashboard/data_service.py` to remove the strict "DraftKings" filter. Accept ALL valid vendors.
- [ ] **Add Sorting**: Edit `backend/api.py` to add `sort_by` parameter to `get_best_bets` (implement sorting by 'quality', 'confidence', 'edge').
- [ ] **Deploy Backend**: Run `railway up` (or ensure git push) to deploy the Python backend.

## Phase 2: Frontend Repairs & Features
- [ ] **Fix Game Cards**: Edit `src/components/v2/GameCard.tsx` to make them clickable (navigate to `/game/:id`).
- [ ] **Update Filter State**: Edit `src/hooks/useFilters.ts` to add `minConfidence` and `sortBy`.
- [ ] **Update Filter UI**: Edit `src/components/predictions/FilterPanel.tsx` to add the Confidence Slider and Sort Dropdown.
- [ ] **Deploy Frontend**: Run `vercel --prod` to deploy the React updates.

## Phase 3: Verification
- [ ] **Data Check**: Verify live site shows picks.
- [ ] **Feature Check**: Verify changing the confidence slider updates the list in real-time.
