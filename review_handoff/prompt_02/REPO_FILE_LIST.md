# Repo File Listing — Prompt 02

**Generated:** 2026-03-17
**Commands run:**
```
pwd                         → /Users/colinai/Library/Mobile Documents/com~apple~CloudDocs/projects/NBA-BETS
git rev-parse --show-toplevel → /Users/colinai/Library/Mobile Documents/com~apple~CloudDocs/projects/NBA-BETS
git rev-parse HEAD          → 2b81b31b17b12926fcec5180b6af836f61a9f2ca
git status --porcelain      → (clean working tree, untracked: .playwright-mcp/, data/agent_scheduler_status.json, data/predictions/.gitkeep, test-results/)
find . -maxdepth 2 -type f | sort → 554 files
```

**Git HEAD:** `2b81b31b17b12926fcec5180b6af836f61a9f2ca`

See full file listing by running: `find . -maxdepth 2 -type f | sort`

Key directories confirmed present:
- `.git/` — valid git repo
- `nba_models/backtesting/` — backtest code
- `nba_models/inference/` — inference code
- `nba_betting/` — betting pipeline
- `edge_calculator/` — edge computation
- `data/historical_lines/` — 166 files (2024-10-22 through 2025-04-xx range)
- `models/` — trained model artifacts
- `tests/` — test suite
- `backend/` — FastAPI API
- `frontend/` — React frontend
