# Master Implementation Plan: Repairs & Features

## Part A: Critical Repairs (Fix Broken Site)

### 1. Unblock Data (`dashboard/data_service.py`)
**Goal**: Stop discarding non-DraftKings data.
**Action**: modify `_get_players_from_props`.
```python
# REMOVE strict filter
# if 'draftkings' not in book_name... continue

# REPLACE with generic acceptance
valid_books = ['draftkings', 'fanduel', 'rebet', 'betmgm', 'caesars', 'bovada']
# Pass through if it matches ANY valid book, or just pass everything.
```

### 2. Fix Navigation (`frontend/src/components/v2/GameCard.tsx`)
**Goal**: Make game cards clickable.
**Action**: Add `useNavigate`.
```tsx
const navigate = useNavigate();
// ...
<div onClick={() => navigate(`/game/${game.game_id}`)} className="cursor-pointer ...">
```

## Part B: New Features (Confidence Control)

### 3. Backend Sorting (`backend/api.py`)
**Goal**: Allow sorting by confidence.
**Action**: Update `get_best_bets` signature.
```python
def get_best_bets(
    # ...
    sort_by: str = Query("quality", description="Sort order: quality, confidence, edge"),
):
    # Sort logic corresponding to selection
```

### 4. Frontend Filters (`frontend/src/components/predictions/FilterPanel.tsx`)
**Goal**: Add UI controls.
**Action**:
- Add **Slider** for `minConfidence` (Range: 50-85).
- Add **Select** for `sortBy` options.

## Part C: Deployment (Required)
**Goal**: Push changes to live server.
**Action**:
- `railway up` (Backend)
- `vercel --prod` (Frontend)
