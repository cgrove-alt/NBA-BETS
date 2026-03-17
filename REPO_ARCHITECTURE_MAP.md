# NBA-BETS Repository Architecture Map

## Training Entry Points

| File | Function | Status | Notes |
|------|----------|--------|-------|
| `nba_models/training/train_complete_balldontlie.py` | `main()`, `train_all_models()` | **Active, Canonical** | Primary training script. 6,385 lines. Trains all model types (moneyline, spread, player props, minutes). Uses Balldontlie API data. |
| `nba_models/training/train_models.py` | `train_models()`, `main()` | **Legacy** | Older training pipeline with integrated data fetching. 797 lines. |
| `minutes_oracle/minutes_trainer.py` | `train_minutes_oracle()` | **Active** | Specialized minutes prediction training. Separate from main pipeline. |
| `continuous_learning/incremental_trainer.py` | `IncrementalTrainer` class | **Active** | Online/incremental retraining on new prediction outcomes. |
| `train_models.py` (root) | N/A | **Shim** | Imports from `nba_models/training/train_models.py` |
| `train_enhanced_v2.py` (root) | N/A | **Legacy/Dead** | Old training variant |
| `train_from_csv.py` (root) | N/A | **Active** | Used by profitability backtest for data loading helpers |
| `train_balldontlie_final.py` (root) | N/A | **Legacy/Dead** | Superseded by `train_complete_balldontlie.py` |
| `train_complete_balldontlie.py` (root) | N/A | **Legacy/Dead** | Superseded by module version |
| `train_stacking_model.py` (root) | N/A | **Legacy/Dead** | Stacking training standalone |
| `train_with_balldontlie.py` (root) | N/A | **Legacy/Dead** | Early BDL training script |
| `railway_retrain.py` (root) | N/A | **Active** | Railway deployment retraining entry point |
| `scheduled_retrain.py` (root) | N/A | **Active** | Scheduled retraining wrapper |
| `scheduled_retraining.py` (root) | N/A | **Duplicate** | Duplicate of scheduled_retrain.py |

**RISK: 14 training-related files at root level, most are legacy/dead but still importable. Confusion risk is high.**

## Inference Entry Points

| File | Function | Status | Notes |
|------|----------|--------|-------|
| `nba_models/inference/daily_predictions.py` | `main()` | **Active, Canonical** | Primary inference engine. ~3,400 lines. Generates daily predictions. |
| `daily_predictions.py` (root) | N/A | **Shim** | Imports from `nba_models/inference/daily_predictions.py` |
| `backend/api.py` | FastAPI app | **Active** | REST API serving predictions via `/api/predictions/{date}` |
| `nba_models/inference/model_compat.py` | `prepare_loaded_model_artifact()` | **Active** | Model loading compatibility layer |

## Daily Prediction Entry Points

| File | Schedule | Status |
|------|----------|--------|
| `.github/workflows/predict-daily.yml` | Daily 2 PM UTC | **Active** | Runs settlement then prediction generation |
| `.github/workflows/weekly-retrain.yml` | Daily 8 AM UTC | **Active** | Full retrain despite name saying "weekly" |
| `agents/core/agent_scheduler.py` | Multiple schedules | **Active** | Orchestrates 6 agents (pregame, postgame, odds, orchestrator, watchdog, briefing) |

## Backtesting Entry Points

| File | Function | Status | Notes |
|------|----------|--------|-------|
| `nba_models/backtesting/profitability_backtest.py` | `run_backtest()` | **Active, Canonical** | Walk-forward P&L simulation on 2023-24 season. Uses `evaluate_bet()` pipeline. **IN-SAMPLE MODEL.** |
| `nba_models/backtesting/comprehensive_backtest.py` | `SeasonBacktester.run_backtest()` | **Active** | 2025-26 season accuracy metrics. Has post-hoc BIAS_CORRECTIONS. |
| `nba_models/backtesting/backtesting.py` | `BacktestSanityChecker` | **Active** | Sanity checks for backtest results |
| `nba_models/backtesting/report_generator.py` | `generate_html_report()` | **Active** | HTML report with Plotly charts |
| `backtesting.py` (root) | N/A | **Shim** | Re-exports from module |
| `backtest.py` (root) | N/A | **Legacy/Dead** | Old standalone backtest |
| `simple_backtest.py` (root) | N/A | **Legacy/Dead** | Simplified backtest |
| `comprehensive_backtest.py` (root) | N/A | **Legacy/Dead** | Old version of comprehensive |

## Feature Generation Modules

| File | Classes/Functions | Status |
|------|-------------------|--------|
| `nba_data/transformers/feature_engineering.py` | `TeamFeatureGenerator`, `PlayerPropFeatureGenerator`, `MatchupFeatureGenerator`, `FeatureSelector`, `LineupImpactCalculator`, `LineMovementFeatureGenerator`, `TravelFatigueFeatureGenerator`, `FourFactorsCalculator`, `ClutchPerformanceCalculator`, `MomentumCalculator` | **Active, Canonical** | 5,028 lines. Master feature engine. |
| `nba_data/transformers/feature_generator.py` | `PlayerFeatureGenerator` | **Active** | Player-specific feature generator (used by inference) |
| `nba_data/transformers/advanced_stats_v2.py` | `FourFactorsCalculator` | **Active** | Four Factors efficiency |
| `nba_data/transformers/elo_ratings.py` | `EloRating` | **Active** | Elo rating system |
| `nba_data/transformers/injury_impact_v2.py` | `InjuryImpactCalculator`, `PlayerUsageTracker` | **Active** | Injury impact features |
| `nba_data/transformers/travel_fatigue.py` | `TravelFatigueCalculator` | **Active** | Travel/rest features |
| `nba_betting/odds/betting_market_features.py` | `BettingMarketFeatures` | **Active** | Line movement, RLM, steam, CLV features |
| `minutes_oracle/minutes_features.py` | `MinutesFeatureGenerator` | **Active** | 38 minutes-specific features |
| `feature_engineering.py` (root) | N/A | **Legacy/Dead** | Old standalone version |
| `feature_generator.py` (root) | N/A | **Legacy/Dead** | Old standalone version |
| `advanced_stats.py` (root) | N/A | **Legacy** | Old version of advanced stats |
| `advanced_stats_v2.py` (root) | N/A | **Legacy** | Root-level copy |
| `betting_market_features.py` (root) | N/A | **Legacy** | Root-level copy |

**RISK: Feature generation code exists in 3+ locations (root, nba_data/transformers, within training script). Drift between them is highly likely.**

## Model Definition Modules

| File | Classes | Status |
|------|---------|--------|
| `nba_models/models/model_trainer.py` | `MoneylineModel`, `EnsembleMoneylineModel`, `TunedEnsembleMoneylineModel`, `SpreadModel`, `LightGBMSpreadModel`, `SpreadCoverClassifier`, `TotalsModel`, `PlayerPropModel`, `QuantilePropModel`, `LineAwarePropClassifier`, `PositionAwarePropEnsemble`, `ParlayCalculator` | **Active, Canonical** | 5,317 lines |
| `nba_models/training/train_complete_balldontlie.py` | `PropEnsembleModel`, `MinutesPredictionModel`, `QuantilePropModel` (inline) | **Active** | Model classes defined INSIDE training script |
| `model_classes.py` (root) | Various | **Active** | Portable model class definitions for unpickling |
| `stacked_model_v2.py` (root) | N/A | **Legacy** | Old stacking ensemble |
| `stacking_meta_learner.py` (root) | N/A | **Legacy** | Old meta-learner |
| `model_trainer.py` (root) | N/A | **Legacy** | Root-level copy |

**RISK: Model classes defined in both `model_trainer.py` and inside `train_complete_balldontlie.py`. Pickle deserialization depends on `__main__` module matching. Custom unpickler exists to handle this but is fragile.**

## Calibration-Related Code

| File | Classes/Functions | Status |
|------|-------------------|--------|
| `nba_models/models/calibration.py` | `PlattScaling`, `IsotonicCalibration`, `TemperatureScaling`, `BetaCalibration`, `FavoriteLongshotCalibrator`, `ShrinkagePlusCalibrator`, `ModelCalibrator`, `CalibrationEvaluator`, `PropEdgeCalibrator`, `StatTypeCalibrator` | **Active** | 1,595 lines. Comprehensive calibration library. |
| `nba_betting/prediction_pipeline.py` | `calibrate_probability()`, `apply_sample_size_confidence_shrink()` | **Active** | Temperature scaling (T=2.0) in pipeline |
| `calibration_tracker/calibration_service.py` | `CalibrationService` | **Active** | Runtime calibration tracking |
| `calibration_tracker/calibration_adjuster.py` | `CalibrationAdjuster` | **Active** | Dynamic calibration adjustments |
| `scripts/calibrate_quantile_decompression.py` | N/A | **Active** | Post-retrain calibration script |
| `scripts/build_probability_calibration.py` | N/A | **Active** | Builds isotonic probability calibrators |
| `calibration.py` (root) | N/A | **Legacy** | Root-level copy |

## Edge Calculation Logic

| File | Functions | Status |
|------|-----------|--------|
| `nba_models/inference/daily_predictions.py` | `_calculate_prop_edge()` | **Active, Primary** | Inference-time edge with devigging |
| `nba_betting/prediction_pipeline.py` | `evaluate_bet()` | **Active, Primary** | Pipeline edge = `abs(predicted - line)` |
| `edge_calculator/edge_calculator.py` | `EdgeCalculator.calculate_edge()` | **Active** | Standalone edge calculator |
| `nba_betting/bet_filter.py` | `should_bet()` | **Active** | Duplicate edge calculation |
| `edge_calculator/kelly_criterion.py` | `KellyCriterion.calculate()` | **Active** | Kelly fraction sizing |

**RISK: Edge is calculated in 4 different places with subtly different logic.**

## Bet Filtering / Sizing Logic

| File | Functions | Status |
|------|-----------|--------|
| `nba_betting/prediction_pipeline.py` | `evaluate_bet()` | **Active, Canonical** | Full pipeline: calibrate → filter → size |
| `nba_betting/bet_filter.py` | `should_bet()`, `calculate_bet_size()` | **Active** | Standalone filter (duplicates pipeline) |
| `nba_betting/constants.py` | Thresholds | **Active** | Single source of truth for constants |
| `edge_calculator/bankroll_manager.py` | N/A | **Active** | Bankroll management |
| `edge_calculator/bet_recommender.py` | N/A | **Active** | Bet recommendations |

## Artifact Save/Load Locations

| Path | Contents | Format |
|------|----------|--------|
| `models/*.pkl` | 39 trained model files | Pickle |
| `models/calibration/` | Calibration metadata | Pickle |
| `models/probability_calibrators/` | Isotonic calibrators per prop | Pickle + JSON |
| `models/quantile_decompression.json` | Decompression parameters | JSON |
| `models/selected_features.json` | Feature selection results | JSON |
| `models/registry.json` | Model version registry | JSON |
| `models_backup_20260226/` | Model backup | Pickle |
| `data/predictions/` | Daily predictions | CSV + JSON |
| `data/historical_lines/` | 170+ daily line snapshots | JSON |
| `data/balldontlie_cache/` | API response cache | JSON |
| `data/live_seasons/` | Season stats snapshots | CSV |

## Configuration Files

| File | Purpose | Status |
|------|---------|--------|
| `nba_betting/constants.py` | Canonical constants (std devs, thresholds, Kelly fractions) | **Active, Single Source of Truth** |
| `.env.example` | Environment variables template | **Active** |
| `railway.toml` | Railway deployment (7 services) | **Active** |
| `railway-cron.yml` | Railway cron jobs | **Active** |
| `requirements.txt` | Python dependencies | **Active** |
| `pytest.ini` | Test configuration | **Active** |
| `ruff.toml` | Linter config | **Active** |

## Report-Generation Files

| File | Purpose | Status |
|------|---------|--------|
| `nba_models/backtesting/report_generator.py` | HTML backtest reports with Plotly | **Active** |
| `report_generator.py` (root) | Root-level copy | **Legacy** |
| `generate_charts.py` (root) | Chart generation | **Legacy** |

## Tests

| Directory | Count | Coverage Focus |
|-----------|-------|----------------|
| `tests/` | 46 files | Agent framework, models, features, edge, calibration, risk, deployment |

## Archived / Legacy / Duplicate Paths

| Path | Status | Risk |
|------|--------|------|
| `archive/` | **Dead** | Contains 7+ improvement plan iterations. No runtime impact. |
| `models_backup_20260226/` | **Dead** | Old model backup. Should be removed. |
| `backtest_reports/` | **Dead** | Old reports |
| `backtest_results/` | **Dead** | 34 directories of old results |
| `training_data/` | **Dead** | Raw training datasets |
| `training_metrics/` | **Dead** | Historical metrics |
| `dashboard/` | **Legacy** | Old Dash dashboard, replaced by React frontend |
| `catboost_info/` | **Dead** | CatBoost training logs |
| 65 root-level Python files | **Mixed** | ~40 are legacy/dead shims. Creates massive confusion. |

## Data Flow Summary

```
Balldontlie API / NBA API / Odds API / RotoWire
        ↓
  Data Sources (nba_data/sources/)
        ↓
  Feature Engineering (nba_data/transformers/)
        ↓
  Training (nba_models/training/) → Models (models/*.pkl)
        ↓
  Inference (nba_models/inference/) ← Models loaded
        ↓
  Edge Calculation + Bet Filtering (nba_betting/)
        ↓
  Predictions (data/predictions/, PostgreSQL)
        ↓
  API (backend/api.py) → Frontend (frontend/)
        ↓
  Settlement (nba_betting/settle_trades.py)
```
