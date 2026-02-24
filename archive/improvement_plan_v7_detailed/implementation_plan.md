# NBA Model Improvement Plan (v7)

## Goal Description
The current NBA prediction model has shown significant deviation from actual results, specifically "last night" (Jan 7, 2026). The goal is to perform a forensic analysis of these failures, establish a rigorous backtesting baseline, and implement a "State of the Art" (SOTA) betting model using advanced feature engineering and ensemble methods.

## User Review Required
> [!IMPORTANT]
> **Forensic Analysis First**: We will not write new model code until we understand *why* the predictions failed. The first task is to regenerate predictions for Jan 7th using point-in-time data and compare with actuals.

> [!WARNING]
> **Computationally Intensive**: The "Comprehensive Backtest" will replay the entire 2025-26 season. This may take significant time but is strictly necessary to prove improvements.

## Proposed Changes

### 1. Forensic Analysis & Baseline
We need to understand if the model failure was due to:
- **Data Latency**: Did we know Player X was out?
- **Feature Gap**: Did the model miss a matchup advantage?
- **Variance**: Was it just a weird night (unders/overs)?

#### [NEW] `analysis/forensic_jan7.py`
- Script to fetch data *exactly as it would have looked* on Jan 7th.
- Generate predictions.
- Compare with actual outcomes line-by-line.

### 2. Feature Engineering 2.0
The current model uses basic stats and some position defense. We will add:

#### [MODIFY] `feature_engineering.py`
- **Dean Oliver's Four Factors**: eFG%, TOV%, ORB%, FT/FGA. These are the "Holy Grail" of basketball analysis.
- **Star Player Impact**: Explicit features for "Is Top 3 Scorer Out?".
- **Momentum/Fatigue Interaction**: Not just "days rest", but "3rd game in 4 nights on the road".

#### [NEW] `advanced_stats_v2.py`
- Calculation logic for Four Factors.
- Rolling 5-game and 10-game Four Factor differentials.

### 3. Model Architecture Upgrade
Moving from simple ensembles to **Stacked Generalization**.

#### [MODIFY] `train_complete_balldontlie.py`
- Implement a **Meta-Learner** (Logistic Regression or simple Neural Net) that takes predictions from:
    1.  XGBoost (Trend based)
    2.  RandomForest (Stable)
    3.  Linear Regression (Baseline)
- The Meta-Learner decides which model to trust based on the specific game context.

### 4. Continuous Learning & Feedback
- Implement a "Self-Correction" loop where the model weighs recent performance higher.

## Verification Plan

### Automated Tests
- **Backtest Run**: Execute `comprehensive_backtest.py` before and after changes.
    - *Metric*: Must see increase in ROI (not just Accuracy).
    - *Metric*: Must see reduction in Brier Score (better probability calibration).

### Manual Verification
- **"Eye Test" on Variance**: Review the "worst misses" of the new model. Do they make sense? (e.g., star player got injured mid-game).
