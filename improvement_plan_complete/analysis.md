# Complete Analysis: Live Site Repairs & New Features

## 1. Why are there no picks? (Root Cause)
The live site is suffering from **"Data Starvation"**, not a crash.
- **The Issue**: The backend code filters player props strictly for "DraftKings".
- **The Reality**: The data provider is currently sending props from "Rebet" or other books.
- **The Result**: The backend receives the data, sees it's not "DraftKings", and discards it. The frontend requests data, and the backend honestly reports "I have 0 props".
- **The Fix**: We must update `data_service.py` to **accept all sportsbooks**.

## 2. Why can't I click games? (UI Bug)
- **The Issue**: The `GameCard` component on the new V2 site is visually designed but lacks interaction code.
- **The Fix**: We must add an `onClick` handler that navigates to `/game/{game_id}`.

## 3. Feature: Confidence Filtering & Sorting
- **The Need**: You want to control the risk/reward.
- **The Solution**:
    - **Backend**: Update the `/api/best-bets` endpoint to accept a `sort_by` parameter (Quality, Confidence, or Edge).
    - **Frontend**: Update the `FilterPanel` to include:
        - A **Slider** to set `min_confidence` (e.g., 50% to 80%).
        - A **Dropdown** to choose `Sort By`.

## 4. Execution Plan
This folder contains the **Master Plan**. It combines the "Critical Repairs" with the "New Features" into one workflow for Claude Code to execute.
