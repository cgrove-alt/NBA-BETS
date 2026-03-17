# Artifact and Deployment Audit — NBA-BETS

## Model Training and Saving

**Training script:** `nba_models/training/train_complete_balldontlie.py`
**Output directory:** `models/`
**Format:** Python pickle (.pkl)

### Artifacts Created

| Artifact | Size | Contents |
|----------|------|----------|
| `moneyline_ensemble.pkl` | ~13 MB | wrapper, base_models, meta_learner, scaler, feature_names, isotonic_calibrator, saved_at |
| `spread_ensemble.pkl` | ~10 MB | spread models, weights, scaler |
| `player_{type}_ensemble.pkl` | ~7 MB each | 4-5 base models, meta_model, scaler, feature_names (points, rebounds, assists, threes, pra) |
| `player_{type}_quantile.pkl` | ~3 MB each | quantile_models dict (0.1-0.9), scaler, feature_names |
| `player_{type}_position_aware.pkl` | ~18 MB | Position-specific models (G/F/C) |
| `player_minutes_model.pkl` | ~0.4 MB | plays classifier + minutes regressor |
| `minutes_oracle.pkl` | ~3.4 MB | Minutes prediction model |
| `models/calibration/*.pkl` | Various | Calibration models |
| `models/probability_calibrators/*.pkl` | Various | Isotonic regression calibrators |
| `models/probability_calibrators/*.json` | Various | JSON lookup tables |
| `models/quantile_decompression.json` | Small | Per-prop slope/gap/mean_line |
| `models/selected_features.json` | Small | RFECV-selected feature names |
| `models/registry.json` | Small | Version registry metadata |

---

## Issues

### AF-01: No Model Quality Gate (CRITICAL)

**Files:** `.github/workflows/weekly-retrain.yml`
**Description:** The retrain workflow trains a model and commits it without any quality check. There is no automated test that verifies the new model performs at least as well as the previous one.
**Risk:** A bad retrain (data issue, API change, hyperparameter drift) produces a worse model that goes straight to production.
**Permanent fix:** After retrain, run quick out-of-sample validation on recent settled predictions. If accuracy degrades beyond threshold, reject model and keep previous one.
**Required tests:** Post-retrain validation script that compares new model vs old on held-out data.

### AF-02: Missing Code Version in Artifacts (HIGH)

**Description:** Model artifacts store `saved_at` timestamp but NOT git commit hash, training data hash, hyperparameters, or training data date range.
**Risk:** Cannot trace a prediction back to the exact training code + data that produced the model.
**Permanent fix:** Add `git_commit`, `training_data_hash`, `hyperparameters`, and `training_data_date_range` to every artifact.

### AF-03: Pickle Format Fragility (HIGH)

**Description:** All models use Python pickle, which is sensitive to module path changes, class definition changes, and Python version differences.
**Evidence:** `continuous_learning/model_registry.py` has custom `RegistryModelUnpickler` (line 41) for `__main__` remapping — a symptom of the problem.
**Risk:** Models saved by training script (classes in `__main__`) may fail to load in inference (different module context).
**Permanent fix:** Move all model classes to a dedicated module. Consider ONNX or joblib. Add load-test after every retrain.

### AF-04: Model Registry Underutilized (MEDIUM)

**Description:** `continuous_learning/model_registry.py` implements version tracking, rollback, and metadata via `models/registry.json`. But the main training script saves directly to `models/` bypassing the registry.
**Risk:** Registry out of sync with deployed models.
**Permanent fix:** Route all model saves through the registry.

### AF-05: Models Committed to Git (HIGH)

**Files:** `.github/workflows/weekly-retrain.yml`
**Description:** Retrain workflow commits .pkl files directly to repo. Binary files that don't diff, bloat the repo ~50-100 MB per retrain.
**Risk:** Repo size grows unboundedly. Git history unwieldy.
**Permanent fix:** Store models in cloud storage (S3, GCS) or Railway volume. Track metadata in git only.

### AF-06: No Artifact Integrity Verification (MEDIUM)

**Description:** No checksum or hash verification when loading. Corrupted .pkl files produce garbage predictions silently.
**Permanent fix:** Add SHA-256 hash to registry. Verify on load.

### AF-07: Workflow Naming Inconsistency (LOW)

**Description:** `weekly-retrain.yml` runs DAILY (`0 8 * * *`), not weekly.
**Permanent fix:** Rename to `daily-retrain.yml`.

---

## Deployment Architecture

### Railway Services (7 total)

| Service | Type | Health Check |
|---------|------|-------------|
| API | Web | `/api/health` |
| Daily Predictions | Cron (0 14 * * *) | N/A |
| Odds Tracker | Daemon | N/A |
| Retraining Scheduler | Daemon | N/A |
| Agent Scheduler | Daemon | N/A |
| PostgreSQL | Database | Auto |
| Redis | Cache | Auto |

### GitHub Actions Workflows

| Workflow | Schedule | Timeout |
|----------|----------|---------|
| `quality-checks.yml` | On push/PR | 15 min |
| `weekly-retrain.yml` | Daily 8 AM UTC | 240 min |
| `predict-daily.yml` | Daily 2 PM UTC | 30 min |

---

## Prediction Traceability

### Can a prediction be traced to exact training inputs and code version?

**Currently: NO.**

| Trace Element | Stored | Location |
|---------------|--------|----------|
| Prediction value | Yes | `data/predictions/`, PostgreSQL |
| Prediction timestamp | Yes | `created_at` in PostgreSQL |
| Model file used | No | Not stored |
| Model version/hash | No | Not stored |
| Git commit of code | No | Not stored |
| Training data date range | No | Not stored |
| Feature values at prediction time | No | Not stored |
| Line at prediction time | Yes | `line` column |
| Line source | Yes | `line_source`, `line_vendor` |
| Odds at prediction time | Partial | `american_odds` stored |
| Closing line | No | Not stored |

**Permanent fix:** Add `model_version`, `code_commit`. Store full feature vector with each prediction for audit trail. Capture closing line for CLV computation.
