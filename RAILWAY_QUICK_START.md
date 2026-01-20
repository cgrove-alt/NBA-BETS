# Railway Deployment - Quick Start Guide

**⚡ 30-Minute Deployment Checklist**

For detailed instructions, see [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md)

---

## Prerequisites ✅

- [ ] Railway account (https://railway.app)
- [ ] GitHub repository with code pushed to main
- [ ] Balldontlie API key (GOAT tier)
- [ ] The Odds API key (optional)

---

## Step 1: Create Railway Project (5 min)

1. Go to https://railway.app
2. Click **New Project** → **Deploy from GitHub repo**
3. Authorize Railway → Select your repo
4. Click **Deploy Now**

---

## Step 2: Provision PostgreSQL (2 min)

1. In Railway project dashboard
2. Click **New** → **Database** → **PostgreSQL**
3. Wait for database to provision (~2 min)
4. `DATABASE_URL` is auto-created ✅

---

## Step 3: Run Database Migration (3 min)

**Option A: Railway CLI** (recommended)
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and link project
railway login
railway link

# Run migration
railway run psql $DATABASE_URL < migrations/001_initial_schema.sql

# Verify (should show 10 tables)
railway run psql $DATABASE_URL -c "\dt"
```

**Option B: Railway Dashboard**
1. Go to PostgreSQL service → **Data** tab → **Query**
2. Paste contents of `migrations/001_initial_schema.sql`
3. Click **Execute**

---

## Step 4: Configure API Service (3 min)

1. Find main service in Railway dashboard
2. Rename to: **nba-betting-api**
3. **Settings** → **Start Command**: `python -m uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
4. **Settings** → **Healthcheck Path**: `/api/health`
5. Click **Deploy**

---

## Step 5: Set Environment Variables (5 min)

**For ALL 4 services** (API + 3 background services):

1. Go to service → **Variables** tab
2. Click **New Variable**
3. Add these **required** variables:
   - `BALLDONTLIE_API_KEY` = your_api_key
   - `DATABASE_URL` = (use **Reference** → PostgreSQL → DATABASE_URL)

4. Add these **optional** variables:
   - `THE_ODDS_API_KEY` = your_odds_api_key
   - `ALERT_EMAIL` = your_email@example.com
   - `SLACK_WEBHOOK` = https://hooks.slack.com/services/...
   - `AUTH_ENABLED` = false
   - `FRONTEND_URL` = https://your-vercel-app.vercel.app

---

## Step 6: Create Background Services (10 min)

### Service #2: Daily Predictions
1. **New** → **GitHub Repo** → Select same repo
2. Rename to: **nba-betting-predictions**
3. **Settings** → **Service Type** → **Cron Job**
4. **Settings** → **Start Command**: `python daily_predictions.py`
5. **Settings** → **Cron Schedule**: `0 9 * * *`
6. Set environment variables (same as API service)
7. **Deploy**

### Service #3: Odds Tracker
1. **New** → **GitHub Repo** → Select same repo
2. Rename to: **nba-betting-odds-tracker**
3. **Settings** → **Service Type** → **Worker**
4. **Settings** → **Start Command**: `python odds_tracker_service.py --daemon`
5. Set environment variables (same as API service)
6. **Deploy**

### Service #4: Retraining Scheduler
1. **New** → **GitHub Repo** → Select same repo
2. Rename to: **nba-betting-retraining**
3. **Settings** → **Service Type** → **Worker**
4. **Settings** → **Start Command**: `python scheduled_retraining.py --daemon`
5. Set environment variables (same as API service)
6. **Deploy**

---

## Step 7: Verify Deployment (2 min)

**Check Services Status**:
- [ ] All 4 services show "Deployed" in Railway dashboard
- [ ] PostgreSQL shows "Active"

**Test API**:
```bash
# Get your Railway API URL from dashboard
export RAILWAY_URL="https://your-app.railway.app"

# Test health endpoint
curl $RAILWAY_URL/api/health

# Should return: {"status":"healthy","timestamp":"..."}
```

**Or use verification script**:
```bash
python verify_deployment.py --url https://your-app.railway.app
```

---

## Common Issues 🔧

### Issue: API returns 500 error
**Fix**: Check logs for errors, verify `DATABASE_URL` is set
```bash
railway logs --service nba-betting-api --tail 50
```

### Issue: Database migration failed
**Fix**: Re-run migration script
```bash
railway run psql $DATABASE_URL < migrations/001_initial_schema.sql
```

### Issue: Cron job not running
**Fix**: Verify service type is "Cron Job" (not "Web" or "Worker")

### Issue: Environment variables not set
**Fix**: Use "Reference" feature for DATABASE_URL to share across services

---

## Success Checklist ✅

After deployment, verify:

- [ ] API health check returns 200: `GET /api/health`
- [ ] Database has 10 tables: `railway run psql $DATABASE_URL -c "\dt"`
- [ ] All 4 services show "Deployed" status
- [ ] Environment variables set for all services
- [ ] Logs show no critical errors
- [ ] Predictions service scheduled for 9 AM daily
- [ ] Odds tracker and retraining services running

---

## Next Steps 🚀

1. **Load Historical Data** - Populate database with past games
2. **Train Initial Models** - Run full retrain manually
3. **Start Paper Trading** - 7 days of validation
4. **Monitor Performance** - Check logs daily for first week
5. **Go Live** - Start live betting after paper trading success

---

## Cost Estimate 💰

**Expected Monthly Cost**: $20-40

- API Service: ~$10/month
- PostgreSQL: ~$5/month
- Worker Services (2): ~$20/month
- Bandwidth: ~$1/month

**Optimization Tips**:
- Archive old `odds_history` (delete >30 days)
- Run predictions only on game days (not every day)
- Use Railway Hobby plan for development

---

## Support & Resources 📚

- **Full Guide**: [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md)
- **Verification Script**: `python verify_deployment.py --help`
- **Railway Docs**: https://docs.railway.app
- **API Docs**: `https://your-app.railway.app/docs`

---

## Deployment Architecture

```
┌─────────────────────────────────────────────┐
│         Railway Project                     │
├─────────────────────────────────────────────┤
│                                             │
│  API Service (Web)                          │
│  ├── FastAPI (Port $PORT)                   │
│  └── Health check: /api/health              │
│                                             │
│  Daily Predictions (Cron)                   │
│  ├── Schedule: 0 9 * * *                    │
│  └── Command: python daily_predictions.py   │
│                                             │
│  Odds Tracker (Worker)                      │
│  ├── Daemon: Always on                      │
│  └── Fetches every 5 min (8 AM-11 PM)       │
│                                             │
│  Retraining Scheduler (Worker)              │
│  ├── Daemon: Always on                      │
│  ├── Full retrain: Every 14 days (2 AM)     │
│  ├── Incremental: Every 3 days (4 AM)       │
│  └── Drift check: Daily (6 AM)              │
│                                             │
│  PostgreSQL Database                        │
│  ├── 10 Tables (teams, players, games, ...) │
│  ├── 25+ Indexes                            │
│  └── Shared via DATABASE_URL                │
│                                             │
└─────────────────────────────────────────────┘
```

---

**⚡ Ready to deploy? Follow the steps above or see [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md) for detailed instructions.**

**No shortcuts. No excuses. Let's go! 🚀**
