# NBA Props API - Quick Reference Card

**Base URL**: `http://localhost:8000` (dev) | `https://your-app.up.railway.app` (prod)

## Core Endpoints

```bash
# Health Check
GET /api/health

# Daily Predictions (with confidence & bet sizing)
GET /api/predictions/{date}              # date: YYYY-MM-DD

# Injury Report (multi-source)
GET /api/injuries/{date}                 # date: YYYY-MM-DD

# Line Movement (with RLM detection)
GET /api/line-movement/{game_id}?market={spread|moneyline|total}

# Latest Backtest Results
GET /api/backtest/latest
```

## Authentication (Optional)

```bash
# Generate Token
POST /api/auth/token
Body: {"username": "user", "password": "pass"}

# Verify Token
GET /api/auth/verify
Header: Authorization: Bearer <token>

# Use in Requests
curl http://localhost:8000/api/predictions/2026-01-19 \
  -H "Authorization: Bearer <token>"
# OR
curl http://localhost:8000/api/predictions/2026-01-19 \
  -H "X-API-Key: <api-key>"
```

## Quick Test

```bash
# Test all endpoints
python3 test_task_4_4_endpoints.py

# Interactive docs
open http://localhost:8000/docs
```

## Start Server

```bash
uvicorn backend.api:app --reload --port 8000
```

## Environment Variables

```bash
AUTH_ENABLED=false              # Enable JWT auth
JWT_SECRET_KEY=secret           # Token signing key
API_KEY=your-key                # Simple API key
FRONTEND_URL=https://...        # Vercel URL for CORS
```

## Response Examples

### Predictions
```json
{
  "date": "2026-01-19",
  "predictions": [{
    "player_name": "LeBron James",
    "prop_type": "Points",
    "prediction": 26.5,
    "confidence_score": 78.5,
    "edge_quality_tier": "strong",
    "suggested_bet_size": 1.5,
    "bet_recommendation": "BET"
  }],
  "count": 150
}
```

### Injuries
```json
{
  "date": "2026-01-19",
  "injuries": [{
    "player_name": "Trae Young",
    "status": "Out",
    "injury_type": "Knee"
  }],
  "count": 99
}
```

### Line Movement
```json
{
  "game_id": "123456",
  "odds_history": [...],
  "movement_analysis": {
    "opening_line": -5.5,
    "closing_line": -4.5,
    "movement": 1.0,
    "rlm_detected": true
  }
}
```

## Error Codes

- `200` - Success
- `400` - Invalid input (bad date format)
- `401` - Unauthorized (auth required)
- `404` - Not found (no predictions/game)
- `429` - Rate limit exceeded
- `500` - Internal error
- `503` - Module unavailable

## Full Documentation

See `API_ENDPOINTS_README.md` for complete docs.
