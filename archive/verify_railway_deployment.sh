#!/bin/bash
# NBA Prediction Model - Railway Deployment Verification Script
# This script helps verify that all Railway services are deployed and running correctly

echo "=============================================="
echo "NBA PREDICTION MODEL - DEPLOYMENT VERIFICATION"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Railway CLI is installed
echo "Step 1: Checking Railway CLI..."
if ! command_exists railway; then
    echo -e "${RED}✗ Railway CLI not installed${NC}"
    echo "Install with: npm install -g @railway/cli"
    exit 1
else
    echo -e "${GREEN}✓ Railway CLI installed${NC}"
    railway whoami
fi
echo ""

# Check if project is linked
echo "Step 2: Checking Railway project link..."
if ! railway status >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Project not linked locally${NC}"
    echo ""
    echo "To link this directory to your Railway project:"
    echo "  1. Get your project ID from Railway dashboard"
    echo "  2. Run: railway link [project-id]"
    echo "  OR"
    echo "  3. Run: railway link (and select from list interactively)"
    echo ""
    echo "Meanwhile, you can verify deployment from Railway web dashboard:"
    echo "  https://railway.app/dashboard"
    echo ""
    exit 1
else
    echo -e "${GREEN}✓ Project linked${NC}"
    railway status
fi
echo ""

# List all services
echo "Step 3: Listing deployed services..."
railway service list
echo ""

# Check each expected service
echo "Step 4: Checking individual services..."

services=("nba-betting-api" "nba-betting-predictions" "nba-betting-odds-tracker" "nba-betting-retraining")

for service in "${services[@]}"; do
    echo -n "  Checking $service... "
    if railway service info --service "$service" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Deployed${NC}"
    else
        echo -e "${RED}✗ Not found${NC}"
    fi
done
echo ""

# Check environment variables
echo "Step 5: Checking environment variables..."
required_vars=("BALLDONTLIE_API_KEY" "THE_ODDS_API_KEY" "DATABASE_URL")

for var in "${required_vars[@]}"; do
    echo -n "  Checking $var... "
    if railway variables | grep -q "$var"; then
        echo -e "${GREEN}✓ Set${NC}"
    else
        echo -e "${RED}✗ Missing${NC}"
    fi
done
echo ""

# Try to get API URL
echo "Step 6: Getting API endpoint..."
api_url=$(railway service url --service nba-betting-api 2>/dev/null)
if [ -n "$api_url" ]; then
    echo -e "${GREEN}✓ API URL: $api_url${NC}"

    echo ""
    echo "Step 7: Testing API health endpoint..."
    response=$(curl -s -o /dev/null -w "%{http_code}" "$api_url/api/health" 2>/dev/null)
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ API is responding (HTTP 200)${NC}"
        curl -s "$api_url/api/health" | python3 -m json.tool
    else
        echo -e "${RED}✗ API returned HTTP $response${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Could not get API URL (service may not be deployed)${NC}"
fi
echo ""

# Check recent logs
echo "Step 8: Checking recent logs (last 50 lines)..."
echo ""
echo "--- API Service Logs ---"
railway logs --service nba-betting-api --tail 50 2>/dev/null || echo "Could not fetch API logs"
echo ""
echo "--- Retraining Scheduler Logs ---"
railway logs --service nba-betting-retraining --tail 50 2>/dev/null || echo "Could not fetch scheduler logs"
echo ""

echo "=============================================="
echo "VERIFICATION COMPLETE"
echo "=============================================="
echo ""
echo "NEXT STEPS:"
echo "1. If project not linked: Run 'railway link' and select your project"
echo "2. Check Railway web dashboard: https://railway.app/dashboard"
echo "3. Look for any error messages in logs above"
echo "4. Verify all 4 services show 'Deployed' status"
echo ""
