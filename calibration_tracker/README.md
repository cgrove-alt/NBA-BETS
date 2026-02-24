# Calibration Tracker - Prediction Performance Tracking

Track every prediction vs actual outcome, identify systematic biases, and generate calibration adjustments.

## Components

### 1. Database (`database.py`)
SQLite schema with tables:
- `predictions` - Every prediction with full context
- `outcomes` - Actual results matched to predictions
- `calibration_adjustments` - Current calibration adjustments
- `daily_reports` - Historical daily reports

### 2. PredictionLogger (`prediction_logger.py`)
Log predictions with full context:
```python
from calibration_tracker import CalibrationService

service = CalibrationService()
pred_id = service.log_prediction(
    player_id=2544,
    player_name="LeBron James",
    team="LAL",
    opponent="BOS",
    game_date="2024-01-15",
    prop_type="points",
    predicted_value=27.5,
    prop_line=26.5,
    predicted_over_prob=0.58,
    confidence=65.0,
    minutes_predicted=35.0,
    position="forward",
    is_home=True,
    spread=-3.5,
    total=225.5,
)
```

### 3. OutcomeTracker (`outcome_tracker.py`)
Record actual results:
```python
service.record_outcome(
    prediction_id=pred_id,
    actual_value=29.0,
    actual_minutes=35.2,
    closing_line=27.0,
    game_score_diff=8,
)
```

### 4. BiasAnalyzer (`bias_analyzer.py`)
Analyze biases across dimensions:
- By prop type (points, rebounds, assists, threes, pra)
- By position (guard, forward, center)
- By minutes bucket (bench, rotation, starter)
- By game type (favorite, underdog, close, blowout)
- By day type (regular, back-to-back)
- By confidence level (high, medium, low)

### 5. CalibrationAdjuster (`calibration_adjuster.py`)
Generate and apply adjustments:
```python
# Generate adjustments from historical data
adjustments = service.generate_calibration_adjustments()

# Apply adjustments to new prediction
result = service.apply_adjustments(
    predicted_value=25.5,
    confidence=65.0,
    prop_type='points',
    position='forward',
    minutes_bucket='starter',
)
# result['adjusted_value'], result['adjusted_confidence']
```

## Usage

### Quick Start
```python
from calibration_tracker import CalibrationService

service = CalibrationService()

# Create calibrated prediction (logs automatically)
result = service.create_calibrated_prediction(
    player={'id': 2544, 'name': 'LeBron James', 'position': 'F', 'team': 'LAL', 'projected_minutes': 35},
    prop_type='points',
    raw_prediction=27.5,
    prop_line=26.5,
    raw_confidence=65.0,
    game_context={'opponent': 'BOS', 'game_date': '2024-01-15', 'is_home': True, 'spread': -3.5},
)

print(f"Calibrated: {result['calibrated_prediction']}")
print(f"Should skip: {result['should_skip']}")
```

### Nightly Job
Run after all games complete (~1am ET):
```bash
python -m calibration_tracker.nightly_job

# Or for a specific date:
python -m calibration_tracker.nightly_job --date 2024-01-15
```

Schedule with cron:
```
0 1 * * * cd /path/to/project && python -m calibration_tracker.nightly_job >> /var/log/calibration.log 2>&1
```

### Calibration Report
```python
report = service.get_calibration_report(days=30)

print(f"Hit Rate: {report['overall']['hit_rate']}")
print(f"CLV: {report['overall']['clv_avg']}")

# By prop type
for prop, analysis in report['by_prop_type'].items():
    print(f"{prop}: {analysis['hit_rate']:.1%} hit, {analysis['bias']:+.1f} bias")

# Recommendations
for rec in report['recommendations']:
    print(f"- {rec}")
```

## Output Format

### Calibration Report
```python
{
    'overall': {
        'predictions': 1000,
        'hit_rate': '52.3%',
        'clv_avg': '+1.2%',
        'roi_estimate': '+2.1%'
    },
    'by_prop_type': {
        'points': {'bias': -0.8, 'adjustment': +0.8, 'hit_rate': 0.54},
        'rebounds': {'bias': +1.2, 'adjustment': -1.2, 'hit_rate': 0.51},
    },
    'by_position': {
        'center': {'hit_rate': 0.58, 'edge_quality': 'strong'},
        'guard': {'hit_rate': 0.48, 'edge_quality': 'negative'},
    },
    'recommendations': [
        'STRENGTH: Center props show strong edge (58% hit rate)',
        'AVOID: Guard threes showing negative edge (46% hit rate)',
        'BIAS: Model overpredicts rebounds by 1.2. Apply -1.2 adjustment.',
    ]
}
```

## Integration with DataService

Add to `data_service.py`:

```python
# Import
from calibration_tracker import CalibrationService

# In __init__
self._calibration = CalibrationService()

# In _get_player_predictions
# Get calibrated prediction
calibrated = self._calibration.create_calibrated_prediction(
    player=player_data,
    prop_type=prop_type,
    raw_prediction=predicted_value,
    prop_line=prop_line,
    raw_confidence=confidence,
    game_context=game_context,
    log_prediction=True,
)

# Use calibrated values
predicted_value = calibrated['calibrated_prediction']
confidence = calibrated['calibrated_confidence']

# Check if should skip
if calibrated['should_skip']:
    continue  # Skip this prop
```

## Database Schema

### predictions
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| timestamp | TEXT | When prediction was made |
| game_date | TEXT | Game date |
| player_id | INTEGER | Player ID |
| player_name | TEXT | Player name |
| prop_type | TEXT | points, rebounds, etc. |
| predicted_value | REAL | Model prediction |
| prop_line | REAL | Betting line |
| confidence | REAL | Confidence (0-100) |
| minutes_predicted | REAL | Projected minutes |
| is_home | INTEGER | 1 if home, 0 if away |
| spread | REAL | Vegas spread |
| status | TEXT | pending, matched, expired |

### outcomes
| Column | Type | Description |
|--------|------|-------------|
| prediction_id | INTEGER | Links to predictions |
| actual_value | REAL | Actual stat value |
| result | TEXT | over, under, push |
| hit | INTEGER | 1 if correct, 0 if wrong |
| error | REAL | predicted - actual |
| clv | REAL | Closing line value |
