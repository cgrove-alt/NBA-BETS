# Railway Deployment Guide - NBA Betting Model

**Complete step-by-step guide for deploying the NBA betting model to Railway with scheduled jobs**

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Deployment Steps](#deployment-steps)
5. [Service Configuration](#service-configuration)
6. [Database Setup](#database-setup)
7. [Environment Variables](#environment-variables)
8. [Scheduled Jobs](#scheduled-jobs)
9. [Monitoring & Alerts](#monitoring--alerts)
10. [Troubleshooting](#troubleshooting)
11. [Cost Estimation](#cost-estimation)

---

## Overview

This NBA betting model is deployed as a **multi-service architecture** on Railway with:

- **4 Railway Services** (API, Daily Predictions, Odds Tracker, Retraining)
- **1 PostgreSQL Database** (shared across all services)
- **Automatic scheduled jobs** (cron-based execution)
- **Health monitoring** and alerts
- **Zero-downtime deployments**

**Total Setup Time**: ~30 minutes
**Monthly Cost**: $20-40 (depending on usage)

---

## Prerequisites

Before starting, ensure you have:

1. ✅ **Railway Account** - Sign up at [railway.app](https://railway.app)
2. ✅ **GitHub Account** - Repository must be on GitHub
3. ✅ **Balldontlie API Key** - GOAT tier from [balldontlie.io](https://balldontlie.io)
4. ✅ **The Odds API Key** (optional) - From [the-odds-api.com](https://the-odds-api.com)
5. ✅ **Git Repository** - Code pushed to GitHub main branch
6. ✅ **Railway CLI** (optional) - For advanced management: `npm i -g @railway/cli`

---

## Architecture

### Multi-Service Design

```
┌─────────────────────────────────────────────────────────────┐
│                     Railway Project                         │
│                  nba-betting-production                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  API Service │  │ Predictions  │  │ Odds Tracker │     │
│  │   (FastAPI)  │  │   Service    │  │   Service    │     │
│  │   Port 8000  │  │  (Cron 9AM)  │  │ (Daemon 5min)│     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                  │             │
│         └─────────────────┼──────────────────┘             │
│                           │                                │
│                  ┌────────▼─────────┐                      │
│                  │   PostgreSQL     │                      │
│                  │   Database       │                      │
│                  │   (Shared)       │                      │
│                  └────────┬─────────┘                      │
│                           │                                │
│  ┌──────────────────────┐ │                                │
│  │ Retraining Scheduler │─┘                                │
│  │     (Daemon)         │                                  │
│  │ Full: 14 days        │                                  │
│  │ Incremental: 3 days  │                                  │
│  └──────────────────────┘                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why this architecture?**
- ✅ API stays responsive during retraining
- ✅ Independent scaling of each service
- ✅ Separate logs for easier debugging
- ✅ Automatic restarts on crashes
- ✅ Shared database ensures consistency

---

## Deployment Steps

### Step 1: Create Railway Project

1. Go to [railway.app](https://railway.app)
2. Click **New Project**
3. Select **Deploy from GitHub repo**
4. Authorize Railway to access your GitHub
5. Select repository: `your-username/nba-betting-model`
6. Click **Deploy Now**

### Step 2: Provision PostgreSQL Database

1. In your Railway project dashboard
2. Click **New** → **Database** → **PostgreSQL**
3. Railway will automatically provision a PostgreSQL instance
4. **DATABASE_URL** environment variable is auto-created
5. Wait for database to be ready (~2 minutes)

### Step 3: Run Database Migration

**Option A: Railway CLI** (recommended)
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login to Railway
railway login

# Link to your project
railway link

# Run migration
railway run psql $DATABASE_URL < migrations/001_initial_schema.sql
```

**Option B: Railway Dashboard**
1. Go to PostgreSQL service → **Data** tab
2. Click **Query** → paste contents of `migrations/001_initial_schema.sql`
3. Click **Execute**
4. Verify: should see "Schema migration 001 completed successfully!"

### Step 4: Configure API Service (Service #1)

1. In Railway dashboard, find your main service
2. Rename to: **nba-betting-api**
3. Go to **Settings** → **Start Command**
4. Set: `python -m uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
5. Go to **Settings** → **Healthcheck Path**
6. Set: `/api/health`
7. Click **Deploy**

### Step 5: Create Daily Predictions Service (Service #2)

1. Click **New** → **GitHub Repo**
2. Select same repository
3. Rename to: **nba-betting-predictions**
4. Go to **Settings** → **Start Command**
5. Set: `python daily_predictions.py`
6. Go to **Settings** → **Cron Schedule**
7. Set: `0 9 * * *` (9 AM daily, EST)
8. **Important**: Go to **Settings** → **Service Type** → Select **Cron Job**
9. Click **Deploy**

### Step 6: Create Odds Tracker Service (Service #3)

1. Click **New** → **GitHub Repo**
2. Select same repository
3. Rename to: **nba-betting-odds-tracker**
4. Go to **Settings** → **Start Command**
5. Set: `python odds_tracker_service.py --daemon`
6. **Important**: This runs as a **Worker** (always on)
7. Go to **Settings** → **Service Type** → Select **Worker**
8. Click **Deploy**

### Step 7: Create Retraining Service (Service #4)

1. Click **New** → **GitHub Repo**
2. Select same repository
3. Rename to: **nba-betting-retraining**
4. Go to **Settings** → **Start Command**
5. Set: `python scheduled_retraining.py --daemon`
6. **Important**: This runs as a **Worker** (always on)
7. Go to **Settings** → **Service Type** → Select **Worker**
8. Click **Deploy**

---

## Service Configuration

### Service #1: API Service

**Type**: Web Service
**Start Command**: `python -m uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
**Port**: $PORT (auto-assigned by Railway)
**Healthcheck**: `/api/health`
**Auto-Deploy**: ✅ Enabled (on git push to main)
**Restart Policy**: On failure

**Resources**:
- CPU: 1 vCPU (shared)
- RAM: 512 MB (scales automatically)
- Disk: 1 GB

**Endpoints**:
- `GET /api/health` - Health check
- `GET /api/predictions/{date}` - Get predictions
- `GET /api/injuries/{date}` - Get injury reports
- `GET /api/line-movement/{game_id}` - Get odds history
- `GET /api/backtest/latest` - Get backtest results

### Service #2: Daily Predictions Service

**Type**: Cron Job
**Start Command**: `python daily_predictions.py`
**Schedule**: `0 9 * * *` (9 AM daily, EST)
**Execution Time**: ~5 minutes
**Auto-Deploy**: ✅ Enabled

**What it does**:
1. Fetches today's NBA games
2. Generates predictions for all games
3. Calculates confidence scores and bet sizing
4. Saves to PostgreSQL `predictions_history` table
5. Exports to CSV (optional)

**Success Criteria**:
- Exit code 0
- Predictions saved to database
- Log shows "Predictions complete: X predictions generated"

### Service #3: Odds Tracker Service

**Type**: Worker (always-on daemon)
**Start Command**: `python odds_tracker_service.py --daemon`
**Schedule**: Every 5 minutes (8 AM - 11 PM EST)
**Season**: Oct-Jun only
**Auto-Deploy**: ✅ Enabled

**What it does**:
1. Runs as background scheduler
2. Fetches odds from The Odds API every 5 minutes
3. Stores in PostgreSQL `odds_history` table
4. Detects line movement and RLM
5. Logs all activity to `logs/odds_tracker.log`

**Health Check**:
- Uses APScheduler heartbeat
- Sends alert if no odds fetched for 30+ minutes
- Auto-restarts on crash

### Service #4: Retraining Scheduler Service

**Type**: Worker (always-on daemon)
**Start Command**: `python scheduled_retraining.py --daemon`
**Schedules**:
- Full retrain: Every 14 days (Sundays at 2 AM)
- Incremental: Every 3 days at 4 AM
- Drift check: Daily at 6 AM
**Auto-Deploy**: ✅ Enabled

**What it does**:
1. Runs as background scheduler
2. Monitors for model drift daily
3. Retrains models on schedule
4. Validates performance before deployment
5. Auto-rollback if performance degrades >5%
6. Sends alerts on completion/failures

**Expected Durations**:
- Full retrain: 30-120 minutes
- Incremental: 5-15 minutes
- Drift check: 1-2 minutes

---

## Database Setup

### PostgreSQL Configuration

**Version**: PostgreSQL 15+
**Storage**: 1 GB (starts), auto-scales to 10 GB
**Connection Limit**: 100 concurrent connections
**Backups**: Daily automatic backups (7-day retention)
**SSL**: ✅ Enabled (required)

### Database Tables

After migration, you'll have 10 tables:

1. **teams** - NBA teams master data
2. **players** - Players with current team
3. **games** - Game schedule and results
4. **player_game_stats** - Box score statistics
5. **injuries** - Injury reports (NBA.com, ESPN)
6. **odds_history** - Betting odds (5-min intervals)
7. **predictions_history** - Model predictions
8. **betting_history** - Actual bets and P&L
9. **model_metadata** - Trained model versions
10. **retraining_history** - Retraining attempts

### Verify Migration Success

**Option 1: Railway Dashboard**
```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
```
Should return 10 tables.

**Option 2: Railway CLI**
```bash
railway run psql $DATABASE_URL -c "\dt"
```

---

## Environment Variables

### Set for ALL Services

These environment variables must be set for **each of the 4 services**:

1. Go to service → **Variables** tab
2. Click **New Variable**
3. Add each variable below

### Required Variables

| Variable | Value | Where to Get |
|----------|-------|--------------|
| `BALLDONTLIE_API_KEY` | Your API key | https://balldontlie.io |
| `DATABASE_URL` | Auto-set by Railway | (Shared reference) |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `THE_ODDS_API_KEY` | None | For betting market features |
| `AUTH_ENABLED` | `false` | Enable JWT authentication |
| `JWT_SECRET_KEY` | None | JWT signing key (if auth enabled) |
| `API_KEY` | None | Simple API key alternative |
| `FRONTEND_URL` | None | CORS allowed origin (Vercel URL) |
| `ALERT_EMAIL` | None | Email for alerts |
| `SLACK_WEBHOOK` | None | Slack webhook for alerts |
| `MAX_TRAINING_TIME` | `7200` | Max training time (seconds) |
| `LOG_LEVEL` | `INFO` | Logging level |

### Reference Shared Variables

For `DATABASE_URL`, use Railway's reference feature:

1. Click **New Variable**
2. Select **Reference** tab
3. Select PostgreSQL service
4. Select `DATABASE_URL`
5. This ensures all services use the same database

---

## Scheduled Jobs

### Cron Schedule Reference

```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday = 0)
│ │ │ │ │
* * * * *
```

### Active Schedules

| Service | Schedule | Cron Expression | Description |
|---------|----------|-----------------|-------------|
| Daily Predictions | 9 AM daily | `0 9 * * *` | Generate predictions |
| Odds Tracker | Every 5 min | Built-in (8AM-11PM) | Fetch odds |
| Full Retrain | Sundays 2 AM | Built-in (14 days) | Retrain all models |
| Incremental | Every 3 days 4 AM | Built-in | Update meta-learner |
| Drift Check | Daily 6 AM | Built-in | Monitor model health |

**Note**: Odds Tracker and Retraining services manage their own schedules internally using APScheduler.

---

## Monitoring & Alerts

### Health Checks

**API Service**:
- Railway pings `/api/health` every 60 seconds
- If 3 consecutive failures → Auto-restart
- Check response: `{"status": "healthy", "timestamp": "..."}`

**Worker Services**:
- Internal heartbeat monitoring
- Logs to `logs/*.log` files
- Alert if no activity for 30+ minutes

### View Logs

**Railway Dashboard**:
1. Select service
2. Click **Deployments** tab
3. Click latest deployment
4. View real-time logs

**Railway CLI**:
```bash
# View API logs
railway logs --service nba-betting-api

# View predictions logs
railway logs --service nba-betting-predictions

# Follow logs in real-time
railway logs --service nba-betting-api --follow
```

### Alert Configuration

**Email Alerts** (if `ALERT_EMAIL` set):
- Retraining failures
- Model performance degradation
- API errors (500 responses)

**Slack Alerts** (if `SLACK_WEBHOOK` set):
- All email alerts
- Daily prediction completion
- Drift detection warnings

**Set up Slack webhook**:
1. Go to https://api.slack.com/messaging/webhooks
2. Create incoming webhook
3. Copy webhook URL
4. Set as `SLACK_WEBHOOK` environment variable

---

## Troubleshooting

### Common Issues

#### 1. API Returns 500 Error

**Symptoms**: `/api/health` returns 500
**Cause**: Models not loaded or database connection failed
**Fix**:
```bash
# Check logs
railway logs --service nba-betting-api --tail 100

# Verify DATABASE_URL is set
railway variables --service nba-betting-api

# Restart service
railway restart --service nba-betting-api
```

#### 2. Daily Predictions Not Running

**Symptoms**: No predictions in database after 9 AM
**Cause**: Cron job failed or wrong schedule
**Fix**:
```bash
# Check cron job logs
railway logs --service nba-betting-predictions

# Verify cron schedule in settings
# Should be: 0 9 * * *

# Manual trigger for testing
railway run python daily_predictions.py
```

#### 3. Odds Tracker Not Fetching

**Symptoms**: `odds_history` table empty
**Cause**: Missing `THE_ODDS_API_KEY` or service crashed
**Fix**:
```bash
# Check if service is running
railway ps

# Check logs for errors
railway logs --service nba-betting-odds-tracker --tail 50

# Verify THE_ODDS_API_KEY is set
railway variables --service nba-betting-odds-tracker

# Restart if needed
railway restart --service nba-betting-odds-tracker
```

#### 4. Database Migration Failed

**Symptoms**: Tables don't exist, API returns errors
**Cause**: Migration script not executed
**Fix**:
```bash
# Re-run migration
railway run psql $DATABASE_URL < migrations/001_initial_schema.sql

# Verify tables created
railway run psql $DATABASE_URL -c "\dt"
```

#### 5. Out of Memory Errors

**Symptoms**: Service crashes during training
**Cause**: Insufficient RAM for model training
**Fix**:
1. Go to service → **Settings** → **Resources**
2. Increase RAM allocation to 2 GB
3. Or split training into smaller batches

#### 6. High Database Costs

**Symptoms**: Railway bill is unexpectedly high
**Cause**: Database storage growth
**Fix**:
```sql
-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Archive old odds_history (keep last 30 days)
DELETE FROM odds_history
WHERE timestamp < NOW() - INTERVAL '30 days';

-- Vacuum to reclaim space
VACUUM FULL odds_history;
```

### Debug Checklist

- [ ] All 4 services are deployed and running
- [ ] PostgreSQL database is provisioned
- [ ] DATABASE_URL is set for all services
- [ ] BALLDONTLIE_API_KEY is set
- [ ] Database migration completed successfully
- [ ] `/api/health` returns 200
- [ ] Logs show no critical errors
- [ ] Cron jobs have correct schedules

---

## Cost Estimation

### Railway Pricing (as of 2026)

**Starter Plan**: $5/month
- Includes $5 in usage credits
- Pay only for what you use

### Expected Monthly Costs

| Resource | Usage | Cost |
|----------|-------|------|
| **API Service** | ~200 hours/month | $10 |
| **PostgreSQL** | 2 GB storage | $5 |
| **Worker Services (2)** | ~400 hours/month | $20 |
| **Bandwidth** | 10 GB | $1 |
| **Total** | | **~$36/month** |

### Cost Optimization Tips

1. **Use Cron Jobs Instead of Workers**:
   - If you can use Railway Cron instead of always-on workers
   - Savings: ~$15/month

2. **Archive Old Data**:
   - Delete `odds_history` older than 30 days
   - Savings: Reduces database costs

3. **Optimize Predictions Service**:
   - Run only on game days (not every day)
   - Savings: ~$3/month

4. **Use Hobby Plan**:
   - If traffic is low (<100k requests/month)
   - Railway Hobby: Free tier available

**Estimated Range**: $20-40/month depending on optimizations

---

## Deployment Verification

### Post-Deployment Checklist

#### 1. API Service ✅
```bash
# Test health endpoint
curl https://your-railway-url.railway.app/api/health

# Expected response:
# {"status":"healthy","timestamp":"2026-01-19T..."}

# Test predictions endpoint
curl https://your-railway-url.railway.app/api/predictions/2026-01-19

# Expected: JSON with predictions array
```

#### 2. Database ✅
```bash
# Check table count
railway run psql $DATABASE_URL -c "SELECT COUNT(*) FROM pg_tables WHERE schemaname='public';"

# Expected: 10

# Check sample data
railway run psql $DATABASE_URL -c "SELECT COUNT(*) FROM teams;"

# Expected: 30 (if teams data loaded)
```

#### 3. Scheduled Jobs ✅
```bash
# Check predictions service last run
railway logs --service nba-betting-predictions --tail 20

# Should show successful prediction generation

# Check odds tracker is running
railway logs --service nba-betting-odds-tracker --tail 20

# Should show "Scheduler started" or recent odds fetch
```

#### 4. Monitoring ✅
```bash
# Trigger test alert (if configured)
railway run python -c "from scheduled_retraining import send_alert; send_alert('Test alert from Railway deployment', 'info')"

# Check email or Slack for alert
```

### Success Criteria

- ✅ All 4 services show "Deployed" status in Railway dashboard
- ✅ API health check returns 200
- ✅ Database has 10 tables
- ✅ Logs show no critical errors
- ✅ Cron jobs executed at least once
- ✅ Alerts working (if configured)

---

## Next Steps

After successful deployment:

1. **Connect Frontend** (Vercel):
   - Update `FRONTEND_URL` with Vercel domain
   - Configure Vercel to use Railway API URL

2. **Load Historical Data**:
   - Run data collection scripts to populate database
   - Import past 2 seasons of games

3. **Train Initial Models**:
   - Trigger manual full retrain: `railway run python scheduled_retraining.py --full`
   - Wait for completion (~2 hours)
   - Verify models saved to `models/` directory

4. **Start Paper Trading**:
   - Enable predictions for next 7 days
   - Track performance in spreadsheet
   - Verify ROI > 3% before live betting

5. **Production Monitoring**:
   - Set up alerts (email + Slack)
   - Monitor logs daily for first week
   - Check database size weekly

---

## Support & Documentation

- **Railway Docs**: https://docs.railway.app
- **Project README**: See main `README.md`
- **API Docs**: https://your-railway-url.railway.app/docs
- **Issues**: Report at project GitHub repository

---

## Deployment Complete! 🚀

You now have a production-grade NBA betting model running on Railway with:

✅ High-availability API (auto-scaling)
✅ Automated daily predictions
✅ Real-time odds tracking
✅ Automatic model retraining
✅ Comprehensive monitoring
✅ Database backups
✅ Zero-downtime deployments

**Time to live betting? Let's go! 💰**
