# Analysis of Excessive Best Bets (181 Bets)

## 1. Verified Symptoms
- **Volume**: The API returns 181 "Best Bets", which is practically unusable for a user.
- **Quality**: The vast majority likely have low confidence (50-55%) or marginal edges.

## 2. Root Cause Analysis

### Default Thresholds are Too Low
- **Code Location**: `backend/api.py` lines 730-731.
- **Current Defaults**:
    - `min_confidence`: **50.0%** (Coin flip)
    - `min_edge`: **3.0 points** (e.g., Line 20.5 vs Pred 23.5)
- **Problem**: The model's "Confidence" score is calibrated to be conservative, often hovering between 50% and 60%. A 50% threshold effectively means "Show me everything that isn't a guaranteed loss."
- **Result**: Almost every prediction with a positive edge > 3.0 passes the filter.

### Missing Quality Standards
- **Problem**: The current definition of "Best Bet" is too loose. A 50% confidence bet is effectively a "Lean", not a "Best Bet".
- **Impact**: The user is flooded with marginal plays, diluting the value of the true high-edge picks.

## 3. Recommended Fixes

### A. Tighten Defaults (Quality over Quantity)
- Increase `min_confidence` to **55.0%** or **56.0%**. This ensures we only show bets where the model has a genuine signal.
- Increase `min_edge` to **5.0%** (relative edge).
- **Result**: We won't hide any bets that meet these standards. If 50 bets are truly great, we show 50. But we won't show 130 "coin flips".

### B. Intelligent Sorting
- Ensure the API returns the bets sorted by `(Confidence * Edge)` so the absolute best options appear first, regardless of how many there are.

### C. Logic Upgrade (The "Honesty" Boost)
- **Problem**: Current confidence logic only looks at "Sample Size" and "Recent Form". It ignores "Matchup Difficulty" and "Home/Away Splits".
- **Fix**: Update `dashboard/data_service.py` (`_calculate_prop_confidence`) to reward positive external factors:
    - **Matchup**: If opponent is ranked 25th-30th vs Position -> **+5% Boost**.
    - **Consistency**: If player hits this line in >60% of games -> **+5% Boost**.
    - **Home Court**: If player performs 10% better at home -> **+3% Boost**.
- **Result**: Good bets get consistently higher scores (65-75%), separating them from the "coin flips" (50-55%).
