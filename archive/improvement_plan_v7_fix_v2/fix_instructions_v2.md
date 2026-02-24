
# Remediation Plan: NBA Model Upgrade (v7-FIX-v2)

## Analysis of Failure
The verification backtest **CRASHED** with `KeyError: 'model'`.

**Root Cause**:
- The new `StackingRegressor` saves its model data as a raw dictionary: `{ 'base_models': ..., 'meta_model': ... }`.
- The `comprehensive_backtest.py` script expects the old schema: `{ 'model': ... }`.
- When the script loaded `player_points_stacking.pkl`, it tried to access `['model']` and failed.

## Execution Plan (Fixes)

### 1. Update `comprehensive_backtest.py` Loader
You must modify the `predict` method (and potentially `load_models`) to handle the Stacking Model dictionary format.

**Locate `comprehensive_backtest.py` ~line 1230 (`predict` method):**
```python
# CURRENT BROKEN CODE:
# model = model_data['model']

# FIX:
if 'meta_model' in model_data and 'base_models' in model_data:
    # It's a Stacking Model!
    # You need to reconstruct it or handle prediction manually
    from models.stacking_model import create_stacking_model  # Ensure imported
    
    # Reconstruct
    if prop_type in ['moneyline']:
        model = create_stacking_model('classification')
    else:
        model = create_stacking_model('regression')
        
    model.base_models = model_data['base_models']
    model.meta_model = model_data['meta_model']
    model.scaler = model_data['scaler']
    model.feature_names = model_data['feature_names']
    model.is_fitted = True
    
    # Now model.predict() will work
else:
    # Legacy fallback
    model = model_data['model']
```

### 2. Verify `daily_predictions.py`
Check if `daily_predictions.py` has the same issue. It likely does if it blindly assumes `['model']`.
- **Action**: Apply the same fix logic to `load_models` or prediction verification in `daily_predictions.py`.

### 3. Run Verification
- [ ] Run `python3 comprehensive_backtest.py` again.
- [ ] **Success**: It prints results (RMSE, MAE) without crashing.
