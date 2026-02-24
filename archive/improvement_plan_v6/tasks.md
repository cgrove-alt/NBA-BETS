# Improvement Plan V6: Tasks

## Phase 1: Backend API
- [ ] **Add Sort Parameter**: Update `backend/api.py` (`get_best_bets`) to accept `sort_by` query param and implement sorting logic for "confidence", "edge", and "quality".

## Phase 2: Frontend State
- [ ] **Update Hook**: Update `frontend/src/hooks/useFilters.ts` to include `minConfidence` (default 55) and `sortBy` (default 'quality').
- [ ] **Update API Client**: Ensure `frontend/src/lib/api.ts` passes these new params to the backend.

## Phase 3: Frontend UI
- [ ] **Enhance FilterPanel**: Edit `frontend/src/components/predictions/FilterPanel.tsx` to add:
    - A slider or number input for "Min Confidence %".
    - A dropdown/radio for "Sort By" (Quality vs Confidence vs Edge).

## Phase 4: Verification
- [ ] **Test Functionality**: Run the app locally, change the slider, and verify the list updates.
