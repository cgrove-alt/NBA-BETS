
# Remediation Plan: NBA Model Upgrade (v7-FIX)

## Analysis of Previous Attempt
The previous agent ("Claude Code") successfully created the Stacking Model architecture (`models/stacking_model.py`) and training script (`train_stacking_model.py`). However, the execution was incomplete:

1.  **Training Incomplete**: Only `moneyline_stacking.pkl` was generated. `spread_stacking.pkl` and prop models are missing.
2.  **Integration Missing**: `daily_predictions.py` was NOT updated to load the new `_stacking.pkl` models. It is still loading the old `_ensemble.pkl` models.
3.  **Verification Invalid**: Because of #2, any backtest run would have tested the *old* models, not the new ones.

## Execution Plan (Fixes)

### 1. Complete Model Training
You need to run the training script to generate the missing models.
- [ ] Run: `python3 train_stacking_model.py --model spread`
- [ ] Run: `python3 train_stacking_model.py --model props`
- [ ] Verify: Ensure `models/spread_stacking.pkl` and `models/player_*_stacking.pkl` exist.

### 2. Update Prediction Pipeline [CRITICAL]
You must Modify `daily_predictions.py` to check for and load the new Stacking models.

#### `daily_predictions.py` (load_models function)
```python
# Moneyline
ml_path = MODEL_DIR / "moneyline_stacking.pkl"  # CHANGED from moneyline_ensemble.pkl
if not ml_path.exists():
    ml_path = MODEL_DIR / "moneyline_ensemble.pkl" # Fallback

# Spread
spread_path = MODEL_DIR / "spread_stacking.pkl" # CHANGED from spread_ensemble.pkl
if not spread_path.exists():
    spread_path = MODEL_DIR / "spread_ensemble.pkl" # Fallback

# Props (inside the loop)
# Look for f"player_{prop_type}_stacking.pkl" first
```

### 3. Verify & Backtest
Once #1 and #2 are done, the pipeline is actually using the new models.
- [ ] Run: `python3 comprehensive_backtest.py`
- [ ] Check Output: Verify "Moneyline Accuracy" and "Spread MAE" reported at the end.

## Success Criteria
- All `_stacking.pkl` files exist in `models/`.
- `daily_predictions.py` contains string "stacking.pkl".
- Backtest runs and shows valid metrics.
