# NBA Model Improvement Tasks (v7)

## Phase 1: Diagnosis & Baseline
- [ ] **Forensic Analysis of Jan 7th Failure**
    - [ ] Create `analysis/forensic_jan7.py` to replay the predictions from that night.
    - [ ] Compare predictions vs actuals to identify the root cause (Data vs Model).
    - [ ] Document findings in `improvement_plan_v7_detailed/forensic_report.md`.
- [ ] **Comprehensive Backtest Baseline**
    - [ ] Run `comprehensive_backtest.py` for the full 2025-26 season.
    - [ ] Save baseline metrics (ROI, RMSE, Brier Score) for comparison.

## Phase 2: Advanced Feature Engineering
- [ ] **Implement Four Factors**
    - [ ] Create `advanced_stats_v2.py`.
    - [ ] Implement `calculate_four_factors(games)`.
    - [ ] Add rolling 5/10 game differentials for Four Factors.
- [ ] **Enhanced Injury Impact**
    - [ ] Update `feature_engineering.py` to include `star_player_out` boolean features.
    - [ ] Calculate `usage_lost` metric (percentage of team usage missing).
- [ ] **Matchup Specifics**
    - [ ] Implement `pace_mismatch` features (Fast team vs Slow team dynamics).
    - [ ] Implement `style_clash` features (e.g., 3PT heavy vs Interior Defense weak).

## Phase 3: Model Architecture Upgrade (SOTA)
- [ ] **Prop Model Tuning**
    - [ ] Retrain `QuantilePropModel` with new features.
    - [ ] Verify calibration of the confidence intervals.
- [ ] **Moneyline/Spread Stacking**
    - [ ] Train Level 1 Models: XGBoost, LightGBM (if avail), RandomForest.
    - [ ] Train Level 2 Meta-Learner (Logistic Regression) on Level 1 outputs.
    - [ ] Verify improvement over single best model.

## Phase 4: Final Verification
- [ ] **Final Backtest**
    - [ ] Run `comprehensive_backtest.py` with new models.
    - [ ] Compare metrics against Phase 1 baseline.
    - [ ] Generate detailed `improvement_plan_v7_detailed/final_report.md`.
- [ ] **Production Deployment**
    - [ ] Update `model_classes.py` with the new architecture.
    - [ ] Ensure `daily_predictions.py` uses the new stack.
