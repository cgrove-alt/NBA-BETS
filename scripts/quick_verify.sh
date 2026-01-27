#!/bin/bash
# Quick verification script for Zenflow
# Runs fast checks after every agent turn (<60s limit)
# Assumes setup_script has already been run

set -e  # Exit on first error

echo "🔍 Running verification checks..."

# 1. Python syntax check (< 1 second)
echo "  → Checking Python syntax..."
python3 -m compileall backend/ -q

# 2. Ruff linting (< 5 seconds, catches bugs & code issues)
echo "  → Running Ruff linter..."
python3 -m ruff check . --quiet

# 3. Frontend linting (< 5 seconds)
echo "  → Linting frontend..."
(cd frontend && npm run lint --silent)

# 4. TypeScript type check & build (< 15 seconds)
echo "  → Building frontend..."
(cd frontend && npm run build)

echo "✅ Verification complete!"
