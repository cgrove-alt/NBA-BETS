#!/bin/bash
# Run daily predictions with API keys from Railway
# Usage: ./scripts/run_daily_predictions.sh [--date YYYY-MM-DD]

set -e

cd "$(dirname "$0")/.."

echo "Loading API keys from Railway..."
export BALLDONTLIE_API_KEY=$(railway variables --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('BALLDONTLIE_API_KEY',''))" 2>/dev/null)
export THE_ODDS_API_KEY=$(railway variables --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('THE_ODDS_API_KEY',''))" 2>/dev/null)

if [ -z "$BALLDONTLIE_API_KEY" ]; then
    echo "ERROR: Could not load BALLDONTLIE_API_KEY from Railway"
    exit 1
fi

echo "Keys loaded. Running predictions..."
PYTHONPATH=. python3 nba_models/inference/daily_predictions.py "$@"

echo ""
echo "Predictions saved to logs/predictions_$(date +%Y-%m-%d).log"
