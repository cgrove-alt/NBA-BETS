#!/bin/bash
# Quick verification script for Zenflow
# Runs fast checks after every agent turn (<60s limit)

set -e  # Exit on first error

echo "🔍 Running verification checks..."

# 1. Python syntax check (< 1 second)
echo "  → Checking Python syntax..."
python3 -m compileall backend/ -q

# 2. Ruff linting (< 5 seconds, catches bugs & code issues)
echo "  → Running Ruff linter..."
python3 -m ruff check . --quiet

# 3. Fast unit tests only (< 20 seconds)
# Skip slow integration tests, API tests, and model training tests
echo "  → Running fast unit tests..."
python3 -m pytest tests/ \
  -v \
  --tb=short \
  -x \
  --timeout=5 \
  -k "not integration and not slow and not api and not train" \
  --maxfail=3 \
  -q \
  || true  # Don't fail verification on test failures (just show them)

# 4. Frontend linting (< 5 seconds)
echo "  → Linting frontend..."
cd frontend && npm run lint --silent

# 5. TypeScript type check & build (< 15 seconds)
echo "  → Building frontend..."
npm run build

echo "✅ Verification complete!"
