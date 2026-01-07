# Improvement Plan V6: User-Controlled Confidence & Sorting

## Goal
Empower the user to adjust the "Confidence Threshold" and "Sort Order" directly in the UI, rather than relying on hardcoded backend defaults.

## 1. Backend Updates (Sorting Support)
**File**: `backend/api.py`
**Action**: Update `get_best_bets` to accept a `sort_by` parameter.

```python
def get_best_bets(
    # ... existing params ...
    sort_by: str = Query("quality", description="Sort order: quality, confidence, edge"),
):
    # ... filtering logic ...

    # Sorting Logic
    if sort_by == "confidence":
        best_bets.sort(key=lambda x: x.confidence, reverse=True)
    elif sort_by == "edge":
        best_bets.sort(key=lambda x: abs(x.edge_pct), reverse=True)
    else: # Default "quality"
        best_bets.sort(key=lambda x: (x.confidence - 50) * abs(x.edge_pct), reverse=True)
```

## 2. Frontend Hook Updates
**File**: `frontend/src/hooks/useFilters.ts`
**Action**: Add `minConfidence` and `sortBy` to the filter state.

```typescript
export interface FilterState {
  minConfidence: number;
  sortBy: 'quality' | 'confidence' | 'edge';
  // ... existing ...
}
```

## 3. Frontend UI Updates (The Filter Panel)
**File**: `frontend/src/components/predictions/FilterPanel.tsx`
**Action**: Add UI controls.
1.  **Confidence Slider**: Range input from 50% to 80%.
2.  **Sort Dropdown**: Select inputs for "Smart Sort (Quality)", "Highest Confidence", "Biggest Edge".

## 4. Verification
- [ ] Slider changes update the "Best Bets" list in real-time.
- [ ] Changing sort order reshuffles the list correctly.
