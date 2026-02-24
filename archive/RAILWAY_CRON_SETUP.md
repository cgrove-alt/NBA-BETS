# Railway Cron Setup - Daily Predictions Automation

## 🚨 CRITICAL ISSUE IDENTIFIED

**Problem:** Predictions are NOT running automatically on Railway because the **Daily Predictions Cron Service** is not deployed.

**Current State:**
- ✅ API Service is running (https://web-production-7b482.up.railway.app)
- ❌ Daily Predictions Service is NOT running (no automated predictions)
- ❌ Odds Tracker Service is NOT running (no line movement tracking)
- ❌ Retraining Scheduler is NOT running (no model updates)

**Impact:** Vercel frontend shows no predictions because Railway backend has no predictions file.

---

## 🎯 SOLUTION: Deploy 4 Separate Railway Services

Railway requires **4 independent services** in the same project:

### Service 1: API Service ✅ (Already Running)
- **Purpose:** REST API for frontend
- **Start Command:** `python -m uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
- **Type:** Web service (always on)
- **Status:** ✅ DEPLOYED

### Service 2: Daily Predictions ❌ (MISSING - THIS IS THE PROBLEM!)
- **Purpose:** Generate predictions every day at 9 AM EST
- **Start Command:** See Option A or B below
- **Type:** Cron job
- **Status:** ❌ NOT DEPLOYED

### Service 3: Odds Tracker ❌ (MISSING)
- **Purpose:** Fetch odds every 5 minutes
- **Start Command:** `python odds_tracker_service.py --daemon`
- **Type:** Worker (always on)
- **Status:** ❌ NOT DEPLOYED

### Service 4: Retraining Scheduler ❌ (MISSING)
- **Purpose:** Retrain models every 14 days
- **Start Command:** `python scheduled_retraining.py --daemon`
- **Type:** Worker (always on)
- **Status:** ❌ NOT DEPLOYED

---

## 🛠️ IMMEDIATE FIX: Deploy Daily Predictions Service

### Option A: Railway Cron (Recommended)

Railway supports cron jobs natively. Here's how to set it up:

#### Step 1: Create New Service in Railway Dashboard

1. Go to https://railway.app/dashboard
2. Open your NBA betting project
3. Click **"+ New Service"**
4. Select **"GitHub Repo"** → Same repo as API service
5. Name it: **"nba-daily-predictions"**

#### Step 2: Configure Service

**Service Settings:**
- **Service Name:** `nba-daily-predictions`
- **Root Directory:** Leave blank (same as API)
- **Build Command:** (leave default - nixpacks)
- **Start Command:**
  ```bash
  python daily_predictions.py --date $(date -u +%Y-%m-%d)
  ```

**Environment Variables:** (Copy from API service)
- `BALLDONTLIE_API_KEY` → (copy from Service 1)
- `DATABASE_URL` → (link to shared PostgreSQL)
- `THE_ODDS_API_KEY` → (copy from Service 1, optional)

**Cron Schedule:**
- Go to **Settings** → **Cron**
- **Schedule:** `0 13 * * *` (9 AM EST = 1 PM UTC)
- **Command:** `python daily_predictions.py --date $(date -u +%Y-%m-%d)`

#### Step 3: Test Manual Execution

Before setting up cron, test manually:

```bash
# In Railway dashboard, go to Settings → Variables
# Add temporary variable: RUN_NOW=true

# Trigger manual deploy
# Check logs to verify predictions generate successfully
```

#### Step 4: Verify Predictions are Saved

After running, check:
- **Railway Logs:** Should show "111 predictions generated"
- **API Endpoint:** `https://web-production-7b482.up.railway.app/api/predictions/2026-01-21`
- **Expected:** JSON with 111 predictions

---

### Option B: External Cron Service (If Railway Cron Unavailable)

Use **Cron-job.org** or **EasyCron** to trigger predictions remotely:

#### Step 1: Create API Trigger Endpoint

Add this endpoint to `backend/api.py`:

```python
@app.post("/api/admin/generate-predictions")
async def trigger_prediction_generation(
    date: str = Query(None),
    api_key: str = Query(...)
):
    """
    Manually trigger prediction generation (admin only).

    Args:
        date: Date in YYYY-MM-DD (defaults to today)
        api_key: Admin API key for authentication

    Returns:
        Status of prediction generation job
    """
    import os
    import subprocess
    from datetime import datetime

    # Verify API key
    if api_key != os.getenv("ADMIN_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Default to today
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')

    # Run predictions in background
    try:
        subprocess.Popen([
            "python", "daily_predictions.py",
            "--date", date
        ])
        return {
            "status": "started",
            "date": date,
            "message": "Prediction generation job started in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### Step 2: Set ADMIN_API_KEY in Railway

```bash
# In Railway dashboard → Settings → Variables
ADMIN_API_KEY=<generate-random-secure-key>
```

#### Step 3: Configure External Cron

At **cron-job.org**:
- **URL:** `https://web-production-7b482.up.railway.app/api/admin/generate-predictions?api_key=YOUR_KEY`
- **Method:** POST
- **Schedule:** Every day at 9:00 AM EST
- **Timezone:** America/New_York

---

## 📊 VERIFICATION CHECKLIST

After deploying the cron service:

### Test 1: Manual Execution
```bash
# SSH into Railway or use Railway CLI
railway run python daily_predictions.py --date 2026-01-21

# Expected output:
# ✓ 111 predictions generated
# ✓ CSV saved to predictions_2026-01-21.csv
```

### Test 2: API Endpoint
```bash
curl https://web-production-7b482.up.railway.app/api/predictions/2026-01-21

# Expected: JSON with 111 predictions
```

### Test 3: Vercel Frontend
```
Visit: https://your-vercel-site.vercel.app
Expected: Predictions visible in dashboard
```

### Test 4: Automatic Execution
```
Wait until 9 AM EST tomorrow
Check Railway logs for cron execution
Verify new predictions appear in API
```

---

## 🔧 TROUBLESHOOTING

### Issue: Predictions CSV not found
**Cause:** Railway filesystem is ephemeral - files don't persist between deployments

**Solution:** Store predictions in PostgreSQL instead of CSV

Add to `daily_predictions.py`:
```python
# After generating predictions, save to database
import psycopg2
import os

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cursor = conn.cursor()

# Insert predictions into database
for idx, row in df.iterrows():
    cursor.execute("""
        INSERT INTO predictions_history
        (date, player_name, prop_type, prediction, confidence, ...)
        VALUES (%s, %s, %s, %s, %s, ...)
    """, (date, row['player_name'], row['prop_type'], ...))

conn.commit()
conn.close()
```

Modify API endpoint to read from database:
```python
@app.get("/api/predictions/{date}")
def get_daily_predictions(date: str):
    # Read from PostgreSQL instead of CSV
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    df = pd.read_sql(
        "SELECT * FROM predictions_history WHERE date = %s",
        conn,
        params=(date,)
    )
    # Convert to JSON...
```

### Issue: Cron job fails silently
**Check Railway Logs:**
1. Go to Railway dashboard
2. Select "nba-daily-predictions" service
3. View Deployments → Latest → Logs
4. Search for errors

**Common Errors:**
- Missing API keys → Add to environment variables
- Import errors → Check requirements.txt is deployed
- Timeout → Increase Railway timeout limit

---

## 🚀 IMMEDIATE ACTION PLAN

**RIGHT NOW (5 minutes):**
1. Go to Railway dashboard
2. Create new service: "nba-daily-predictions"
3. Set start command: `python daily_predictions.py --date $(date -u +%Y-%m-%d)`
4. Copy environment variables from API service
5. Link to shared PostgreSQL database

**MANUAL TEST (10 minutes):**
1. Trigger manual deploy
2. Check logs: Should see "111 predictions generated"
3. Test API: `curl /api/predictions/2026-01-21`
4. Verify Vercel frontend shows predictions

**SETUP CRON (5 minutes):**
1. Go to Service Settings → Cron
2. Schedule: `0 13 * * *` (9 AM EST)
3. Command: `python daily_predictions.py`
4. Save and enable

**TOMORROW (Verify):**
1. Check Railway logs at 9 AM EST
2. Verify new predictions in API
3. Verify Vercel frontend updates automatically

---

## 📁 FILES NEEDED ON RAILWAY

Ensure these files are in the GitHub repo:

✅ `daily_predictions.py` (main script)
✅ `backend/api.py` (API with predictions endpoint)
✅ `requirements.txt` (all dependencies)
✅ `railway.toml` (deployment config)
✅ `models/*.pkl` (trained models - should be in repo or downloadable)
⚠️ `predictions_*.csv` (generated dynamically - DON'T commit)

---

## 💡 LONG-TERM SOLUTION

**Switch to Database Storage:**

Instead of CSV files, store predictions in PostgreSQL:

**Pros:**
- ✅ Persists across Railway deployments
- ✅ Queryable (filter by date, player, confidence)
- ✅ Supports historical analysis
- ✅ No file upload needed

**Cons:**
- ⚠️ Requires database migration
- ⚠️ More complex than CSV

**Migration Script:** (Already created in Task 4.5)
```sql
-- See migrations/001_initial_schema.sql
CREATE TABLE predictions_history (...);
```

---

## ✅ SUCCESS CRITERIA

Automated predictions are working when:

1. ✅ Railway cron job runs daily at 9 AM EST
2. ✅ API endpoint `/api/predictions/{date}` returns 100+ predictions
3. ✅ Vercel frontend displays predictions automatically
4. ✅ No manual intervention required
5. ✅ Zero DNP errors (injured players excluded)

---

## 🎯 NO SHORTCUTS. NO EXCUSES!

Deploy the cron service NOW to get predictions running automatically!
