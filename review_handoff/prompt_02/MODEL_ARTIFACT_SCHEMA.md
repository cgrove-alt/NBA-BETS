# Model Artifact Metadata Schema — NBA-BETS

**Version:** 1.0.0

---

## Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `artifact_version` | string | YES | Unique version ID (e.g., `20260317_083000`) |
| `git_sha` | string | YES | Git commit hash of the training code |
| `model_family` | string (enum) | YES | One of: `ensemble`, `quantile`, `position_aware`, `minutes_oracle`, `moneyline`, `spread` |
| `target_market_type` | string | YES | What the model predicts (e.g., `player_points`, `player_rebounds`) |
| `train_window_start` | string (date) | YES | First date in training data (inclusive) |
| `train_window_end` | string (date) | YES | Last date in training data (inclusive) |
| `feature_schema_version` | string | YES | Hash or version of the feature list used |
| `calibration_version` | string | NO | Version of calibration parameters applied |
| `data_snapshot_id` | string | NO | Identifier for the exact data snapshot used |
| `training_timestamp` | string (ISO8601) | YES | When training completed |
| `hyperparams_hash` | string | NO | SHA256 of sorted hyperparameter dict |
| `training_samples` | integer | YES | Number of training samples |
| `validation_samples` | integer | NO | Number of validation samples |
| `validation_metrics` | object | NO | Key metrics on validation set (RMSE, R², etc.) |
| `feature_names` | array[string] | YES | Ordered list of feature names |
| `scaler_type` | string | NO | Type of scaler used (e.g., `StandardScaler`) |
| `base_model_types` | array[string] | NO | For ensembles: list of base model types |
| `notes` | string | NO | Free-text notes |

---

## Example Instance

```json
{
  "artifact_version": "20260310_080000",
  "git_sha": "a1b2c3d4e5f6a7b8c9d0",
  "model_family": "ensemble",
  "target_market_type": "player_points",
  "train_window_start": "2023-10-22",
  "train_window_end": "2025-10-21",
  "feature_schema_version": "v3.1_150features",
  "calibration_version": "isotonic_20260310",
  "data_snapshot_id": "bdl_snapshot_20260310",
  "training_timestamp": "2026-03-10T08:00:00Z",
  "hyperparams_hash": "sha256:abc123def456",
  "training_samples": 45000,
  "validation_samples": 5000,
  "validation_metrics": {
    "rmse": 6.31,
    "r_squared": 0.42,
    "mae": 4.87
  },
  "feature_names": ["season_pts_avg", "recent_pts_avg", "opp_def_rating", "..."],
  "scaler_type": "StandardScaler",
  "base_model_types": ["xgboost", "lightgbm", "catboost", "random_forest"],
  "notes": "Walk-forward window 12/24"
}
```

---

## Current Repo Gap

**Current artifact format** (`nba_models/training/train_complete_balldontlie.py:L5763-5781`):
```python
pickle.dump({
    'model': wrapper,
    'scaler': scaler_ml,
    'feature_names': feature_names,
    'training_metrics': ml_metrics,
    'isotonic_calibrator': isotonic_calibrator,
    'calibration_enabled': use_calibration,
    'meta_learner': meta_learner,
    'stacking_enabled': use_stacking,
    'base_model_order': list(models.keys()),
    'saved_at': datetime.now().isoformat(),
}, f)
```

**Missing fields in current artifacts:**
- `git_sha` — not stored
- `train_window_start` / `train_window_end` — not stored
- `feature_schema_version` — not stored
- `artifact_version` — only `saved_at` timestamp
- `hyperparams_hash` — not stored
- `training_samples` — not stored
