# NBA Odds Tracker Background Service

Automated background service for fetching and storing NBA betting odds at regular intervals using APScheduler.

## Features

- ✅ Automatic odds fetching every 5 minutes during game hours (8 AM - 11 PM EST)
- ✅ NBA season awareness (only runs Oct-Jun)
- ✅ Error handling with retry logic (3 attempts with 60s delays)
- ✅ Comprehensive logging to `odds_tracker.log`
- ✅ Health monitoring and uptime tracking
- ✅ Graceful shutdown handling
- ✅ Built-in test mode for validation

## Requirements

- Python 3.8+
- APScheduler 3.10+
- The Odds API key (100k subscription)
- `betting_market_features.py` module

Install dependencies:
```bash
pip install apscheduler
```

## Setup

1. **Set API Key**:
   ```bash
   export THE_ODDS_API_KEY='your_api_key_here'
   ```

2. **Verify Installation**:
   ```bash
   python3 odds_tracker_service.py --test
   ```

## Usage

### Run as Daemon (Recommended)

Start the service and let it run continuously:

```bash
python3 odds_tracker_service.py
```

The service will:
- Start immediately if during operating hours
- Schedule updates every 5 minutes
- Log all activity to `odds_tracker.log`
- Continue running until interrupted (Ctrl+C)

### Custom Configuration

```bash
# Use 10-minute intervals
python3 odds_tracker_service.py --interval 10

# Custom database path
python3 odds_tracker_service.py --db-path /path/to/odds.db

# Custom log file
python3 odds_tracker_service.py --log-file /path/to/tracker.log
```

### Check Status

```bash
python3 odds_tracker_service.py --status
```

Output:
```
======================================================================
ODDS TRACKER SERVICE STATUS
======================================================================
Status: RUNNING
Uptime: 2:45:30

Runs: 33 total (32 ✓, 1 ✗)
Success Rate: 97.0%

Last Success: 2025-01-15T14:30:00
Last Failure: 2025-01-15T12:15:00

NBA Season: Yes
Should Run Now: Yes
======================================================================
```

### Test Mode

Run one fetch and exit (for testing):

```bash
python3 odds_tracker_service.py --test
```

## Programmatic Usage

```python
from odds_tracker_service import OddsTrackerService

# Initialize service
service = OddsTrackerService(
    api_key='your_key',
    update_interval=5,  # minutes
    db_path='odds_history.db'
)

# Start in background
service.start()

# Check health
status = service.get_health_status()
print(f"Status: {status['status']}")
print(f"Success Rate: {status['success_rate']}")

# Manual fetch (outside schedule)
service.fetch_and_store_with_retry()

# Stop service
service.stop()
```

## Operating Schedule

- **Operating Hours**: 8 AM - 11 PM EST
- **Update Interval**: Every 5 minutes (configurable)
- **NBA Season Only**: Oct, Nov, Dec, Jan, Feb, Mar, Apr, May, Jun
- **Offseason Behavior**: Service runs but skips fetches (logs "Outside operating hours")

## Database Schema

The service stores odds in SQLite database `odds_history.db`:

### Tables

**games**
- `game_id` (PRIMARY KEY)
- `home_team`, `away_team`
- `commence_time`
- `created_at`

**odds_history**
- `id` (PRIMARY KEY)
- `game_id` (FOREIGN KEY)
- `timestamp`, `book_name`, `market`
- `home_odds`, `away_odds`, `home_line`, `away_line`
- `total`, `over_odds`, `under_odds`
- `is_opening`, `is_closing`

**line_movements**
- `id` (PRIMARY KEY)
- `game_id` (FOREIGN KEY)
- `market`, `opening_line`, `closing_line`, `movement`
- `rlm_detected`, `steam_detected`

## Logging

All activity logged to `odds_tracker.log`:

```
2025-01-15 14:30:00 - OddsTrackerService - INFO - ✓ Stored 150 odds snapshots (attempt 1)
2025-01-15 14:35:00 - OddsTrackerService - INFO - ✓ Stored 145 odds snapshots (attempt 1)
2025-01-15 14:40:00 - OddsTrackerService - WARNING - Fetch failed (attempt 1/3): Connection timeout. Retrying in 60s...
2025-01-15 14:41:00 - OddsTrackerService - INFO - ✓ Stored 148 odds snapshots (attempt 2)
```

## Error Handling

### Retry Logic
- Max 3 attempts per fetch
- 60-second delay between retries
- Exponential backoff (configurable)

### Failure Scenarios
1. **API Rate Limit**: Retries with backoff
2. **Network Error**: Retries 3 times, logs error
3. **Database Error**: Logged, reported in health status
4. **Invalid API Key**: Raises ValueError on initialization

## Monitoring

### Health Metrics
- `total_runs`: Total fetch attempts
- `successful_runs`: Successful fetches
- `failed_runs`: Failed fetches (after all retries)
- `success_rate`: Percentage of successful runs
- `last_success`: Timestamp of last successful fetch
- `last_failure`: Timestamp of last failure
- `uptime_seconds`: Total service uptime

### Alerts
- **API Failures**: Logged to file (set up external monitoring)
- **Success Rate < 95%**: Review logs for issues
- **No Success in 1 Hour**: Check API key, network, database

## Production Deployment

### Railway (Recommended)

1. **Add to `railway.toml`**:
   ```toml
   [build]
   builder = "nixpacks"

   [[services]]
   name = "odds-tracker"
   startCommand = "python3 odds_tracker_service.py"
   restartPolicyType = "on-failure"
   ```

2. **Set Environment Variables**:
   - `THE_ODDS_API_KEY=your_key`

3. **Deploy**:
   ```bash
   railway up
   ```

### systemd (Linux)

Create `/etc/systemd/system/odds-tracker.service`:

```ini
[Unit]
Description=NBA Odds Tracker Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/app
Environment="THE_ODDS_API_KEY=your_key"
ExecStart=/usr/bin/python3 odds_tracker_service.py
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable odds-tracker
sudo systemctl start odds-tracker
sudo systemctl status odds-tracker
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV THE_ODDS_API_KEY=""
CMD ["python3", "odds_tracker_service.py"]
```

Build and run:
```bash
docker build -t odds-tracker .
docker run -d --name odds-tracker \
  -e THE_ODDS_API_KEY=your_key \
  -v $(pwd)/odds_history.db:/app/odds_history.db \
  -v $(pwd)/odds_tracker.log:/app/odds_tracker.log \
  odds-tracker
```

## Testing

Run unit tests:
```bash
python3 tests/test_odds_tracker_service.py
```

Expected output:
```
Ran 17 tests in 0.009s
OK
```

## Troubleshooting

### Service Won't Start
- **Error**: `Missing THE_ODDS_API_KEY`
  - **Fix**: Set environment variable

- **Error**: `betting_market_features.py not found`
  - **Fix**: Ensure file is in same directory

### No Odds Being Fetched
- **Check**: Is it NBA season? (Oct-Jun only)
- **Check**: Is it during operating hours? (8 AM - 11 PM EST)
- **Check**: Run `--status` to see "Should Run Now"

### High Failure Rate
- **Check**: API key valid and quota remaining
- **Check**: Network connectivity
- **Check**: Database permissions (write access)

### Database Growing Too Large
- **Solution**: Add cleanup job to delete old odds (>30 days)
- **Example**:
  ```python
  # In betting_market_features.py
  def cleanup_old_odds(days=30):
      cutoff = datetime.now() - timedelta(days=days)
      db.execute("DELETE FROM odds_history WHERE timestamp < ?", (cutoff,))
  ```

## API Usage

The Odds API (100k subscription) allows:
- **500 requests/month** for free tier
- **10,000 requests/month** for $100 tier
- **100,000 requests/month** for $1,000 tier

With 5-minute intervals:
- **12 requests/hour** × **15 hours/day** = **180 requests/day**
- **180/day** × **30 days** = **5,400 requests/month**

**Recommendation**: Use $100 tier (10k/month) minimum

## Support

For issues or questions:
1. Check logs: `tail -f odds_tracker.log`
2. Run status check: `python3 odds_tracker_service.py --status`
3. Run test: `python3 odds_tracker_service.py --test`
4. Review unit tests: `python3 tests/test_odds_tracker_service.py`

## License

Part of the NBA Prediction Model project.
