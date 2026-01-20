# Railway Deployment Verification Guide

**Date**: 2026-01-20
**Status**: You said "Railway is deployed" - Let's verify it's actually working!

---

## QUICK VERIFICATION (5 Minutes)

### Option 1: Railway Web Dashboard (EASIEST)
1. Go to https://railway.app/dashboard
2. Look for your NBA prediction project
3. Verify you see **4 services**:
   - `nba-betting-api`
   - `nba-betting-predictions`
   - `nba-betting-odds-tracker`
   - `nba-betting-retraining`
4. Check each service shows **"Active"** status (green dot)

**Screenshot what you see and tell me if all 4 are green/active.**

---

### Option 2: Link Local Directory to Railway

Since your local directory isn't linked to the Railway project, run:

```bash
# Option A: Interactive selection
railway link

# Option B: Direct link if you know project ID
railway link [your-project-id]
```

Then run the verification script:
```bash
./verify_railway_deployment.sh
```

---

## CRITICAL CHECKS

### Check #1: Are Services Actually Running?

**Railway Dashboard**: Look for these 4 services with GREEN status:

1. **nba-betting-api**
   - Type: Web Service
   - Port: 8000 (or $PORT)
   - Should have a public URL (https://nba-betting-api-production.up.railway.app or similar)

2. **nba-betting-predictions**
   - Type: Cron Job
   - Schedule: 0 9 * * * (9 AM daily)
   - Last run: Should show today's date at 9 AM

3. **nba-betting-odds-tracker**
   - Type: Worker (daemon)
   - Should be continuously running
   - Check logs for "Fetched X odds" every 5 minutes

4. **nba-betting-retraining**
   - Type: Worker (daemon)
   - Should be continuously running
   - Check logs for "Automated retraining pipeline started"

---

### Check #2: Environment Variables Set?

In Railway dashboard, click on **each service** → **Variables** tab:

**Required Variables** (should be set on ALL 4 services):
- ✅ `BALLDONTLIE_API_KEY` = cc19b625-9176-4407-8623-f97ec32f4f3d
- ✅ `THE_ODDS_API_KEY` = (your key - you said it's set)
- ✅ `DATABASE_URL` = (PostgreSQL connection string - auto-set by Railway)

**Optional Variables**:
- `JWT_SECRET_KEY` (for API auth)
- `ALERT_EMAIL` (for monitoring alerts)
- `SLACK_WEBHOOK` (for Slack notifications)

**Screenshot** the variables tab and tell me which are set.

---

### Check #3: Database Provisioned?

In Railway dashboard:
1. Look for **PostgreSQL** service (database icon)
2. Click on it → **Data** tab
3. You should see **10 tables**:
   - teams
   - players
   - games
   - player_game_stats
   - injuries
   - odds_history
   - predictions_history
   - betting_history
   - model_metadata
   - retraining_history

**If tables don't exist**, you need to run the migration:
```bash
# After linking project
railway run psql $DATABASE_URL < migrations/001_initial_schema.sql
```

---

### Check #4: API is Responding?

**Get your API URL** from Railway dashboard:
- Click on `nba-betting-api` service
- Look for "Deployments" → "Domain" (e.g., https://nba-betting-api-production.up.railway.app)

**Test endpoints**:
```bash
# Replace with your actual URL
export API_URL="https://your-actual-url.up.railway.app"

# Health check
curl $API_URL/api/health

# Should return: {"status":"healthy","uptime":XXX,"models_loaded":true}

# Get today's predictions (if 9 AM job has run)
curl $API_URL/api/predictions/2026-01-20

# Get injuries
curl $API_URL/api/injuries/2026-01-20
```

**Tell me what the health check returns.**

---

### Check #5: Scheduled Jobs Running?

**Predictions Job** (9 AM daily):
1. Go to Railway dashboard → `nba-betting-predictions` service
2. Click "Deployments" → View logs
3. Look for today's run (should be around 9 AM)
4. Should see: "Generated 102 predictions" or similar

**Odds Tracker** (every 5 minutes):
1. Go to Railway dashboard → `nba-betting-odds-tracker` service
2. View logs
3. Should see recent entries like:
   ```
   INFO:OddsTrackerService: ✓ Stored 50 odds snapshots
   ```

**Retraining Scheduler**:
1. Go to Railway dashboard → `nba-betting-retraining` service
2. View logs
3. Should see:
   ```
   INFO:scheduled_retraining:Automated retraining pipeline started
   INFO:scheduled_retraining:Scheduled jobs:
     - Full Model Retraining: cron[day_of_week='sun', hour='2']
     - Incremental Meta-Learner Update: interval[3 days]
     - Drift Detection & Emergency Retrain: cron[hour='6']
   ```

**Screenshot the logs and tell me what you see.**

---

### Check #6: Database Has Data?

If you can connect to the database from Railway dashboard:

1. Click PostgreSQL service → "Data" tab
2. Run these queries:

```sql
-- Check if odds are being stored
SELECT COUNT(*) FROM odds_history;
-- Should be > 0 if odds tracker is running

-- Check if predictions exist
SELECT COUNT(*) FROM predictions_history;
-- Should be > 0 if 9 AM job has run

-- Check if injuries are being tracked
SELECT COUNT(*) FROM injuries;
-- Should be > 0

-- Check recent prediction
SELECT * FROM predictions_history
ORDER BY created_at DESC
LIMIT 5;
```

**Tell me the counts from each query.**

---

## COMMON ISSUES & FIXES

### Issue #1: Services Show "Building" or "Crashed"
**Symptom**: Services in Railway dashboard show red/yellow status
**Fix**:
1. Click on the service
2. View "Logs" tab
3. Look for error messages
4. **Tell me what the error says** (paste here)

Common errors:
- "Module not found" → Missing dependency in requirements.txt
- "Port already in use" → Change port in start command
- "Database connection failed" → DATABASE_URL not set

---

### Issue #2: Predictions Job Not Running
**Symptom**: No predictions CSV generated at 9 AM
**Fix**:
1. Check cron schedule is set: `0 9 * * *`
2. Check timezone (Railway uses UTC by default)
   - 9 AM EST = 14:00 UTC (2 PM UTC)
   - Update cron to: `0 14 * * *` for EST
3. Manually trigger: Click "Run Now" in Railway dashboard

---

### Issue #3: Odds Tracker Not Storing Data
**Symptom**: `odds_history` table is empty
**Fix**:
1. Check THE_ODDS_API_KEY is set
2. Check service logs for errors:
   ```
   ERROR: API key invalid
   ERROR: Rate limit exceeded
   ```
3. Verify API key at theoddsapi.com dashboard

---

### Issue #4: Database Tables Don't Exist
**Symptom**: API returns "relation does not exist" errors
**Fix**: Run migration script
```bash
# Link project first
railway link

# Run migration
railway run psql $DATABASE_URL < migrations/001_initial_schema.sql
```

---

## VALIDATION CHECKLIST

### Deployment Validated When:
- [ ] All 4 services show "Active" status in Railway dashboard
- [ ] PostgreSQL database has 10 tables
- [ ] Environment variables set (BALLDONTLIE_API_KEY, THE_ODDS_API_KEY, DATABASE_URL)
- [ ] API health check returns HTTP 200
- [ ] Predictions job ran today at 9 AM (check logs)
- [ ] Odds tracker shows recent logs (last 5 minutes)
- [ ] Retraining scheduler shows "pipeline started" log
- [ ] Database has data (odds_history, injuries, predictions_history)

### Production Ready When:
- [ ] All of above ✓
- [ ] No errors in any service logs
- [ ] 7 consecutive days of successful prediction runs
- [ ] Injury detection working (OUT players not in predictions)
- [ ] API responds in <1 second

---

## WHAT TO TELL ME

Since I can't access your Railway dashboard, **please provide**:

1. **Screenshot of Railway dashboard** showing all services
2. **Service status** for each of the 4 services (Active/Crashed/Building?)
3. **API URL** (from nba-betting-api service)
4. **Result of health check**: `curl YOUR_API_URL/api/health`
5. **Last 20 lines of logs** from each service (copy/paste)
6. **Database table count**: How many tables exist?
7. **Environment variables**: Which are set? (don't share the actual keys, just confirm they exist)

---

## QUICK DIAGNOSTIC SCRIPT

If you've linked the project locally, run:
```bash
# Link project first (interactive)
railway link

# Then run verification
./verify_railway_deployment.sh
```

This script will automatically check:
- ✅ Railway CLI installed
- ✅ Project linked
- ✅ All 4 services deployed
- ✅ Environment variables set
- ✅ API responding
- ✅ Recent logs from each service

**Paste the output here.**

---

## NEXT STEPS

### If Deployment is Working (All Green):
1. ✅ Mark deployment as verified
2. ✅ Monitor for 24 hours
3. ✅ Check predictions generate tomorrow at 9 AM
4. ✅ Run validation backtest
5. ✅ Start 7-day paper trading

### If Deployment Has Issues:
1. ❌ Identify which service is failing (tell me)
2. ❌ Share error logs (paste here)
3. ❌ We'll fix it together

**NO SHORTCUTS. NO EXCUSES.** Let's verify every service is actually running before declaring victory.

---

## YOUR ACTION ITEMS

**DO NOW** (5 minutes):
1. Go to https://railway.app/dashboard
2. Take screenshot of your project
3. Check status of all 4 services
4. Get API URL from nba-betting-api service
5. Test health endpoint: `curl YOUR_URL/api/health`
6. **Tell me the results**

Then we'll verify everything is working correctly!
