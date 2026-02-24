# Improvement Plan V5: The Final Fix

## Goal
Force data to appear on the live site by removing all filters and fixing UI navigation.

## 1. Backend Data Fix (CRITICAL)
**File**: `dashboard/data_service.py`
**Action**: Modify `_get_players_from_props` to accept ALL vendors.

```python
# FIND THIS BLOCK:
# if 'draftkings' not in book_name and 'draftkings' not in vendor:
#    continue

# REPLACE WITH:
# Accept everything. We need data.
pass 
```

**File**: `backend/api.py`
**Action**: Update `get_best_bets` defaults to be "Honest" but functional.
- `min_confidence` = 55.0
- `min_edge` = 4.0

## 2. Frontend UI Fix
**File**: `frontend/src/components/v2/GameCard.tsx`
**Action**: Make the card clickable.

```tsx
import { useNavigate } from 'react-router-dom';

export function GameCard({ game }) {
  const navigate = useNavigate();
  return (
    <div 
      className="game-card ..." 
      onClick={() => navigate(`/game/${game.game_id}`)} // Use game_id matches API
      style={{ cursor: 'pointer' }}
    >
      {/* existing content */}
    </div>
  );
}
```

## 3. Deployment (The Missing Link)
We must ensure code reaches the servers.
1.  **Backend**: `railway up` (or git push if connected).
2.  **Frontend**: `vercel --prod`.

## 4. Verification Checklist
- [ ] Visit `https://WEB_PRODUCTION_URL/api/best-bets` -> Should show JSON data (not empty!).
- [ ] Visit `https://nba-bets.vercel.app/` -> Should show picks in the "Top Bets" list.
- [ ] Click a game card -> Should navigate to `/game/123...`.
