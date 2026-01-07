# Debugging Report

## Findings
1.  **Site Status:** Online and loading, but functionality is impaired.
2.  **"Best Bets" Issue:** The model is likely returning empty data arrays, or the `confidence` threshold in the frontend filters is too high (default > 60%).
3.  **"Nothing Happens" Issue:** The `Predictions` page is the home page. It uses a `GameSelector`. If clicking a game doesn't scroll or clearly show loading (logic is `setSelectedGameId`), and data is empty, the user sees no change.
4.  **Mobile Issue:** The layout is not mobile-optimized, making interactions difficult.

## Recommendations for Claude Code
- **Force Data Visibility:** Lower filter thresholds to ensure *some* bets always show, even if low confidence (label them appropriately).
- **Feedback Loop:** Add a clear "Loading..." overlay on the main content area when a game is clicked.
- **Interactive Cards:** Ensure the main game cards are clickable, not just the selector list.
- **Data Pipeline:** Verify `http://localhost:8000` (or prod URL) is returning data.
