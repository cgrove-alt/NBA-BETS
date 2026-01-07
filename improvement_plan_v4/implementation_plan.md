# Improvement Plan V4: Best Bets Calibration

## Goal
Reduce "Best Bets" count from ~180 to a quality-focused list by boosting honest confidence signals and filtering noise.

## Proposed Changes

### 1. Update Defaults in `backend/api.py`
**File**: `backend/api.py`
**Change**: Update `get_best_bets` signature.
```python
def get_best_bets(
    min_confidence: float = Query(55.0, ...),  # Bump from 50.0 to ensure "Honesty"
    min_edge: float = Query(4.0, ...),         # Bump from 3.0
    # No limit parameter - we show EVERYTHING that qualifies
    # ...
)
```

### 2. Implement Intelligent Sorting
**File**: `backend/api.py` inside `get_best_bets`
**Logic**: Sort the results so the highest quality bets are at the top.
```python
# ... collection changes ...
best_bets = []

# Collect ALL potential candidates
candidates = []
for game in games:
    # ... check players ...
    # Calculate a score for sorting:
    score = (confidence - 50) * abs(edge_pct) 
    candidates.append({
        "bet": BestBet(...),
        "score": score
    })

# Sort by score descending (Best quality first)
candidates.sort(key=lambda x: x["score"], reverse=True)

# Return ALL candidates that passed the filter
best_bets = [c["bet"] for c in candidates]
```

### 3. Logic Upgrade (`dashboard/data_service.py`)
**Goal**: Make the model "confidently honest" by rewarding clear positive signals.
**Method**: Update `_calculate_prop_confidence`.

```python
# Add Boosting Logic
market_hit_rate = features.get('last_10_hit_rate', 0.5)
if market_hit_rate > 0.6: confidence += 5  # Proven winner

opponent_rank = features.get('opponent_rank', 15)
if opponent_rank >= 25: confidence += 5    # Easy matchup

# Cap at 85% to stay realistic (Honesty cap)
confidence = min(confidence, 85.0)
```

### 4. Frontend Update (Optional but Recommended)
**File**: `frontend/src/lib/api.ts`
**Change**: Ensure frontend calls match the new defaults or explicitly request stricter filters if needed.

## Verification
1. **API Check**: requests `http://localhost:8000/api/best-bets` should return a list sorted by quality.
2. **Quality Check**: Verify the returned bets have high confidence (>55%) and edge. Count may still be high (e.g. 50+), which is acceptable if they are good bets.
