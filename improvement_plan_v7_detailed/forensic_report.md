# Forensic Analysis Report: January 7, 2026

Generated: 2026-01-08 09:13:05

## Executive Summary

- **Games Analyzed**: 12
- **Predictions Analyzed**: 1030
- **RMSE**: 5.988
- **MAE**: 4.069
- **Bias**: -0.738

## Error Categorization

### Data (46 errors)
Average Error: 15.2

Examples:
- Shai Gilgeous-Alexander points: pred=15.5, actual=46.0
- Shai Gilgeous-Alexander pra: pred=28.1, actual=58.0
- Keyonte George pra: pred=14.7, actual=43.0

### Features (176 errors)
Average Error: 7.0

Examples:
- Ryan Dunn pra: pred=14.9, actual=5.0
- Ivica Zubac pra: pred=25.1, actual=35.0
- Adem Bona points: pred=11.8, actual=2.0

### Variance (613 errors)
Average Error: 1.9

Examples:
- Day'Ron Sharpe assists: pred=0.0, actual=5.0
- Goga Bitadze rebounds: pred=5.0, actual=10.0
- Jamal Shead assists: pred=1.1, actual=6.0

### Model (34 errors)
Average Error: 13.9

Examples:
- Luka Doncic pra: pred=30.4, actual=58.0
- Paolo Banchero pra: pred=27.3, actual=50.0
- Giannis Antetokounmpo pra: pred=27.7, actual=49.0

### Injury (161 errors)
Average Error: 2.0

Examples:
- Jordan Hawkins pra: pred=14.4, actual=0.0
- Gradey Dick points: pred=11.7, actual=0.0
- Trendon Watford points: pred=11.7, actual=0.0

## Key Findings

- HIGH INJURY RATE: 161 players DNP unexpectedly
- PRA: High RMSE (9.4) - needs investigation

## Recommendations

1. IMPROVE DATA: Integrate real-time injury feeds (ESPN, Rotowire)
2. ENHANCE FEATURES: Add Four Factors (eFG%, TOV%, ORB%, FT/FGA)
3. UPGRADE MODEL: Implement stacked ensemble with meta-learner

## Worst Predictions

| Player | Prop | Predicted | Actual | Error |
|--------|------|-----------|--------|-------|
| Shai Gilgeous-Alexander | points | 15.5 | 46.0 | -30.5 |
| Shai Gilgeous-Alexander | pra | 28.1 | 58.0 | -29.9 |
| Keyonte George | pra | 14.7 | 43.0 | -28.3 |
| Luka Doncic | pra | 30.4 | 58.0 | -27.6 |
| Peyton Watson | pra | 11.6 | 38.0 | -26.4 |
| Jamal Murray | pra | 22.5 | 47.0 | -24.5 |
| Deni Avdija | points | 17.7 | 41.0 | -23.3 |
| Paolo Banchero | pra | 27.3 | 50.0 | -22.7 |
| Peyton Watson | points | 8.2 | 30.0 | -21.8 |
| Giannis Antetokounmpo | pra | 27.7 | 49.0 | -21.3 |
