# Railway Setup Guide

Everything on the code side is done and pushed. You need to do 3 things in Railway's dashboard, then create 10 services. Here's exactly what to do.

---

## Part 1: Add the 3 Missing Infrastructure Pieces

### Step 1: Add PostgreSQL Database

1. Open your Railway project at [railway.app](https://railway.app)
2. Click **"+ New"** (top right of your project canvas)
3. Select **"Database"**
4. Select **"PostgreSQL"**
5. Wait ~30 seconds for it to provision
6. Click on the new PostgreSQL service -> **"Variables"** tab
7. You'll see `DATABASE_URL` is automatically created -- **copy this value**
8. Now click on your **main app service** (the one running your API)
9. Go to its **"Variables"** tab
10. Click **"+ New Variable"**
11. Add: `DATABASE_URL` = paste the value you copied
12. **Repeat step 10-11 for every other service you create** (or use Railway's "Shared Variables" feature -- see tip below)

> **Tip:** Railway has a "Shared Variables" feature. Click on `DATABASE_URL` in your PostgreSQL service, then click "Share" to make it available to all services in the project automatically. This is much easier than copying it manually to each service.

### Step 2: Add Redis

1. Click **"+ New"** again (top right)
2. Select **"Database"**
3. Select **"Redis"**
4. Wait ~30 seconds
5. Same as above -- share the `REDIS_URL` variable with all services

### Step 3: Get a Gemini API Key (Free)

1. Go to **https://aistudio.google.com/apikey** in your browser
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key
5. Back in Railway, add `GEMINI_API_KEY` as a shared variable (or add it to each service manually)

### After Part 1 -- Your Variables Should Look Like:

| Variable | Where it comes from |
|----------|-------------------|
| `BALLDONTLIE_API_KEY` | Already set (you did this) |
| `THE_ODDS_API_KEY` | Already set (you did this) |
| `DATABASE_URL` | Auto-created by PostgreSQL service -- share it |
| `REDIS_URL` | Auto-created by Redis service -- share it |
| `GEMINI_API_KEY` | You just created this at Google AI Studio |

---

## Part 2: Create the 10 Services

Your project needs 12 items total. You already have 1 service (the API). You need to create 10 more services + the 2 databases from Part 1.

### How to Create Each Service

For **every** service below, do this:

1. Click **"+ New"** -> **"GitHub Repo"**
2. Select your repo: **`cgrove-alt/NBA-BETS`**
3. Railway will auto-detect settings from `railway.toml` -- **you need to override these**
4. Click on the new service -> **"Settings"** tab
5. Change the **service name** to match the table below
6. Change the **Start Command** to match the table below
7. If it's a **Cron Job**: Go to **"Settings"** -> find "Cron Schedule" -> enter the schedule
8. If it's a **Worker**: Set the service type to "Worker" (not "Web")
9. Make sure all 5 environment variables from Part 1 are available to this service

---

### Service 1: API Server (you already have this)

- **Name:** `nba-betting-api`
- **Type:** Web
- **Start Command:** `python -m uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
- **Health Check:** `/api/health`
- Already running -- no action needed
- **Note:** On first deploy after adding PostgreSQL, the API automatically runs database migrations on startup. Check logs for "Database migrations: up to date."

---

### Service 2: Daily Predictions

- **Name:** `nba-daily-predictions`
- **Type:** Cron Job
- **Start Command:** `python daily_predictions.py`
- **Cron Schedule:** `0 9 * * *`
- **What it does:** Runs at 9 AM ET every day. Loads models, fetches today's games, generates spread + prop predictions, saves to database.

---

### Service 3: Odds Tracker

- **Name:** `nba-odds-tracker`
- **Type:** Worker (always-on)
- **Start Command:** `python odds_tracker_service.py --daemon`
- **What it does:** Runs continuously. Fetches odds every 5 minutes from 8 AM to 11 PM ET during NBA season.

---

### Service 4: Retraining Scheduler

- **Name:** `nba-retraining`
- **Type:** Worker (always-on)
- **Start Command:** `python scheduled_retraining.py --daemon`
- **What it does:** Runs continuously. Full retrain every 2 weeks (Sundays 2 AM), incremental updates every 3 days (4 AM), drift detection daily (6 AM).

---

### Service 5: Pre-Game Intel Agent

- **Name:** `nba-agent-pregame`
- **Type:** Cron Job
- **Start Command:** `python -m agents.core.agent_runner --agent pregame`
- **Cron Schedule:** `0 11,17 * * *`
- **What it does:** Runs at 11 AM + 5 PM ET. Gathers injury reports, lineup predictions, schedule context. Synthesizes with AI reasoning.

---

### Service 6: Post-Game Analysis Agent

- **Name:** `nba-agent-postgame`
- **Type:** Cron Job
- **Start Command:** `python -m agents.core.agent_runner --agent postgame`
- **Cron Schedule:** `0 1 * * *`
- **What it does:** Runs at 1 AM ET. Reviews yesterday's predictions vs actual results. Identifies why predictions missed.

---

### Service 7: Odds Monitor Agent

- **Name:** `nba-agent-odds-monitor`
- **Type:** Cron Job
- **Start Command:** `python -m agents.core.agent_runner --agent odds_monitor`
- **Cron Schedule:** `*/15 8-23 * * *`
- **What it does:** Runs every 15 minutes (8 AM - 11 PM ET). Tracks line movements, detects sharp money, alerts on edge erosion.

---

### Service 8: Prediction Orchestrator Agent

- **Name:** `nba-agent-orchestrator`
- **Type:** Cron Job
- **Start Command:** `python -m agents.core.agent_runner --agent orchestrator`
- **Cron Schedule:** `30 11 * * *`
- **What it does:** Runs at 11:30 AM ET (after pregame intel). Coordinates the full prediction pipeline, adjusts confidence, generates final bet/pass/lean signals.

---

### Service 9: Model Watchdog Agent

- **Name:** `nba-agent-watchdog`
- **Type:** Cron Job
- **Start Command:** `python -m agents.core.agent_runner --agent watchdog`
- **Cron Schedule:** `30 1 * * *`
- **What it does:** Runs at 1:30 AM ET (after postgame). Monitors model health, detects drift, recommends retraining when needed.

---

### Service 10: Daily Briefing Agent

- **Name:** `nba-agent-briefing`
- **Type:** Cron Job
- **Start Command:** `python -m agents.core.agent_runner --agent briefing`
- **Cron Schedule:** `0 12,18 * * *`
- **What it does:** Runs at noon + 6 PM ET. Creates your daily briefing -- today's picks, yesterday's results, system health, bankroll status. Plain English, no jargon.

---

## Part 3: Verify Everything Works

After you've created all services:

1. **Check the API:** Visit your Railway API URL + `/api/health` in your browser. You should see `{"status": "healthy", ...}`
2. **Check migration logs:** Click on the API service -> "Logs" tab. Look for "Database migrations: up to date." on startup.
3. **Check other service logs:** Click on any service -> "Logs" tab. Look for successful starts, no red errors.
4. **Wait for first cron runs:** The agents will fire at their scheduled times. Check logs the next day to confirm they ran.

---

## Quick Reference Card

| # | Service | Type | Schedule | Start Command |
|---|---------|------|----------|---------------|
| 1 | API | Web | Always on | `python -m uvicorn backend.api:app --host 0.0.0.0 --port $PORT` |
| 2 | Daily Predictions | Cron | `0 9 * * *` | `python daily_predictions.py` |
| 3 | Odds Tracker | Worker | Always on | `python odds_tracker_service.py --daemon` |
| 4 | Retraining | Worker | Always on | `python scheduled_retraining.py --daemon` |
| 5 | PreGame Agent | Cron | `0 11,17 * * *` | `python -m agents.core.agent_runner --agent pregame` |
| 6 | PostGame Agent | Cron | `0 1 * * *` | `python -m agents.core.agent_runner --agent postgame` |
| 7 | Odds Monitor Agent | Cron | `*/15 8-23 * * *` | `python -m agents.core.agent_runner --agent odds_monitor` |
| 8 | Orchestrator Agent | Cron | `30 11 * * *` | `python -m agents.core.agent_runner --agent orchestrator` |
| 9 | Watchdog Agent | Cron | `30 1 * * *` | `python -m agents.core.agent_runner --agent watchdog` |
| 10 | Briefing Agent | Cron | `0 12,18 * * *` | `python -m agents.core.agent_runner --agent briefing` |
| 11 | PostgreSQL | Database | Always on | (auto-provisioned) |
| 12 | Redis | Database | Always on | (auto-provisioned) |

---

## Cost Estimate

- **Railway free tier:** $5/month credit
- **PostgreSQL + Redis:** ~$5-10/month each depending on usage
- **Cron jobs:** Billed per execution time (very cheap -- most run < 5 minutes)
- **Workers (Odds Tracker + Retraining):** ~$5-10/month each (always on)
- **Gemini API:** Free tier is sufficient for agent usage
- **Estimated total:** ~$20-40/month

## If Something Goes Wrong

- Check the service **Logs** tab in Railway -- errors will be there
- If a service won't start, verify all 5 environment variables are set for that service
- If the API health check fails, check that `DATABASE_URL` is properly shared
- If you see "migration failed" in API logs, the PostgreSQL database may not be linked -- check that `DATABASE_URL` is available to the API service
- You can always restart any service from Railway dashboard -> service -> Settings -> Restart
- You can run migrations manually: `python scripts/run_migrations.py` (requires DATABASE_URL)
