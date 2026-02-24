# Phase 1 — Full Codebase Audit

**Date:** 2026-02-23
**Auditor:** Claude Code
**Codebase Size:** ~111,842 lines of code (73,412 root Python + 11,687 dashboard + 8,938 tests + 2,063 backend + 3,843 minutes_oracle + 3,159 calibration_tracker + 2,824 edge_calculator + 2,516 lineup_intel + 2,151 continuous_learning + 1,122 scripts + 8,127 frontend TypeScript)

---

## Table of Contents

1. [Codebase Summary](#codebase-summary)
2. [File-by-File Audit (Root Python)](#file-by-file-audit-root-python)
3. [Subdirectory Modules](#subdirectory-modules)
4. [Frontend](#frontend)
5. [Configuration & Deployment](#configuration--deployment)
6. [Data Artifacts](#data-artifacts)
7. [Documentation & Planning Files](#documentation--planning-files)
8. [Environment Variables](#environment-variables)
9. [External Dependencies](#external-dependencies)
10. [External API Integrations](#external-api-integrations)
11. [Trained Models Inventory](#trained-models-inventory)
12. [Dead Code & Candidates for Removal](#dead-code--candidates-for-removal)
13. [Target Architecture Mapping](#target-architecture-mapping)
14. [Key Architectural Patterns](#key-architectural-patterns)
15. [Known Issues & Technical Debt](#known-issues--technical-debt)

---

## Codebase Summary

| Metric | Value |
|--------|-------|
| Root-level Python files | 75 |
| Python modules in subdirectories | 48 |
| Frontend TypeScript/TSX files | ~15 |
| Test files | 26 (13 in tests/, 13 at root) |
| Trained model files (.pkl) | 53 |
| Config files | 10 |
| Documentation/planning files | 40+ |
| Database files (.db) | 5 |
| Total Python lines of code | ~103,715 |
| Total TypeScript lines of code | ~8,127 |

---

## File-by-File Audit (Root Python)

### Core Prediction Pipeline

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `app.py` | ~2,400 | Main orchestrator — fetches schedule, runs feature engineering, loads models, generates moneyline/spread/prop predictions, outputs bet slips | `packages/prediction-engine/src/inference/` | **ACTIVE** — core entry point for batch predictions |
| `daily_predictions.py` | ~2,400 | Generates daily predictions for all bet types using Balldontlie API. Includes prop calibration, injury adjustment, Kelly sizing, risk management | `packages/prediction-engine/src/inference/` | **ACTIVE** — production daily prediction generator |
| `feature_engineering.py` | ~5,300 | Massive feature engineering module. Generates game features (team stats, H2H, home advantage, rest/fatigue), player prop features (points, rebounds, assists, threes, PRA), injury impact. Implements temporal discipline (`before_date` params) | `packages/data-pipeline/src/transformers/` | **ACTIVE** — core feature pipeline |
| `model_trainer.py` | ~5,500 | ML model definitions and training. Logistic Regression, SVM, Random Forest, Gradient Boosting, stacking classifiers. Smart feature defaults (not zeros). Uncertainty flags | `packages/prediction-engine/src/models/` | **ACTIVE** — model definitions |
| `train_complete_balldontlie.py` | ~6,600 | Complete training pipeline using Balldontlie API data. Trains moneyline ensemble, spread, and all player prop models. Optuna hyperparameter tuning. Probability calibration | `packages/prediction-engine/src/training/` | **ACTIVE** — primary training script |
| `train_models.py` | ~750 | Training orchestrator supporting multiple data sources (Kaggle CSV > Database > NBA API). Command-line args for data source selection | `packages/prediction-engine/src/training/` | **ACTIVE** — training entry point |
| `train_stacking_model.py` | ~1,100 | Trains stacking ensemble models (moneyline classifier, spread regressor, prop regressors). Optuna tuning | `packages/prediction-engine/src/training/` | **ACTIVE** — ensemble training |

### Data Fetching & Integration

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `data_fetcher.py` | ~2,000 | Fetches NBA data from Balldontlie (primary) and NBA API (fallback). Team stats, player stats, H2H, historical games. Temporal-safe variants (`_before_date`) for backtesting | `packages/data-pipeline/src/sources/` | **ACTIVE** — core data fetcher |
| `balldontlie_api.py` | ~850 | Full Balldontlie API client supporting Free/All-Star/GOAT tiers. Disk caching with TTLs (live: 1min, daily: 30min, stats: 1hr, historical: 24hr). Betting odds, live box scores, season averages | `packages/data-pipeline/src/sources/` | **ACTIVE** — primary API client |
| `fast_data_fetcher.py` | ~520 | Lightweight Balldontlie fetcher with minimal rate limiting (100ms). Static NBA team data. For quick lookups | `packages/data-pipeline/src/sources/` | **ACTIVE** — fast data access |
| `odds_fetcher.py` | ~1,400 | The Odds API integration. Real-time moneyline/spread/totals odds from 40+ sportsbooks. Line shopping, odds movement tracking, closing line storage | `packages/data-pipeline/src/sources/` | **ACTIVE** — odds integration |
| `historical_data_collector.py` | ~590 | Batch historical data collector. 3+ seasons via nba_api with resume capability. Rate limiting, SQLite storage | `packages/data-pipeline/src/sources/` | **ACTIVE** — historical backfill |
| `kaggle_data_loader.py` | ~530 | Loads 33,000+ games from 2010-2024 CSV files. No rate limits, fastest data source. Team abbreviation normalization | `packages/data-pipeline/src/sources/` | **ACTIVE** — CSV data loading |
| `live_season_fetcher.py` | ~280 | Fetches recent seasons (2023-26) using Balldontlie with nba_api fallback | `packages/data-pipeline/src/sources/` | **ACTIVE** — season data |
| `tracking_data.py` | ~1,400 | NBA play-by-play data, shot charts, tracking metrics. PBPParser, ShotAtlas, RotationTracker classes | `packages/data-pipeline/src/sources/` | **ACTIVE** — advanced tracking data |
| `injury_fetcher.py` | ~700 | Injury data from Balldontlie API with fallback database. InjuryStatus enum with availability probabilities | `packages/data-pipeline/src/sources/` | **ACTIVE** — injury data |
| `injury_tracker_v3.py` | ~430 | Multi-source injury tracking (RotoWire primary, NBA.com fallback). 100% DNP detection. SQLite cache, 15-min refresh | `packages/data-pipeline/src/sources/` | **ACTIVE** — real-time injury tracking |
| `player_impact_fetcher.py` | ~1,100 | Advanced player impact metrics (DARKO DPM, ESPN EPM, FiveThirtyEight RAPTOR, nba_api fallback). 24hr cache | `packages/data-pipeline/src/sources/` | **ACTIVE** — player impact data |
| `news_sentiment.py` | ~870 | Claude API-powered sentiment analysis of NBA news. NewsIngestor, SentimentAnalyzer, InjuryImpactCalculator | `packages/agents/src/pregame/` | **ACTIVE** — qualitative intelligence |
| `referee_data.py` | ~450 | Referee tendency analysis (50+ refs). Foul rates, home team win%, pace tendency. 24hr cache | `packages/data-pipeline/src/sources/` | **ACTIVE** — referee data |
| `id_mapping.py` | ~350 | Maps between Balldontlie API IDs and player/team names. TEAM_ABBREV_TO_BDL dictionary | `packages/data-pipeline/src/sources/` | **ACTIVE** — utility |

### Feature Engineering (Specialized)

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `feature_generator.py` | ~700 | Canonical single-source-of-truth for 150 player prop features. Ensures training/prediction consistency. Created 2026-01-14 to fix feature mismatch bug | `packages/data-pipeline/src/transformers/` | **ACTIVE** — canonical feature logic |
| `advanced_stats.py` | ~640 | Advanced basketball stats (PER, TS%, USG%, BPM, Win Shares, ORTG/DRTG, eFG%) | `packages/data-pipeline/src/transformers/` | **ACTIVE** — advanced stats |
| `advanced_stats_v2.py` | ~610 | Dean Oliver's Four Factors (eFG% 40%, TOV% 25%, ORB% 20%, FT rate 15%). Rolling differentials, matchup-specific factors, style clash indicators | `packages/data-pipeline/src/transformers/` | **ACTIVE** — Four Factors |
| `betting_market_features.py` | ~920 | Market intelligence features: opening/closing lines, line movement, RLM detection, steam moves, consensus odds, CLV | `packages/betting-engine/src/odds/` | **ACTIVE** — market features |
| `market_microstructure.py` | ~920 | Real-time line movement monitoring. SteamDetector, StaleLineFinder. Sharp vs soft book classification | `packages/betting-engine/src/odds/` | **ACTIVE** — microstructure analysis |
| `injury_impact_v2.py` | ~450 | Injury impact on team performance. PlayerUsageTracker, star player thresholds, usage redistribution | `packages/data-pipeline/src/transformers/` | **ACTIVE** — injury impact modeling |
| `elo_ratings.py` | ~670 | Dynamic Elo team strength ratings. Point-in-time queries, home court adjustment, margin of victory modifier | `packages/data-pipeline/src/transformers/` | **ACTIVE** — team ratings |
| `travel_fatigue.py` | ~450 | Travel fatigue calculation. Haversine distance, B2B detection, altitude adjustment, timezone crossing. All 30 arena coordinates | `packages/data-pipeline/src/transformers/` | **ACTIVE** — fatigue modeling |

### Model Architecture & Ensemble

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `stacking_meta_learner.py` | ~720 | Stacking ensemble with OOF predictions. TimeSeriesSplit validation. Meta-learner types: XGBoost, MLP, Ridge. Time-decay weighting, uncertainty quantification | `packages/prediction-engine/src/models/` | **ACTIVE** — ensemble core |
| `stacked_model_v2.py` | ~450 | Meta-learning ensemble (Level 0 base models, Level 1 meta-learner, optional calibration). Production stacking implementation | `packages/prediction-engine/src/models/` | **ACTIVE** — stacking v2 |
| `model_classes.py` | ~360 | Portable class definitions for unpickling trained models without heavy dependencies. Enables model loading on Railway | `packages/prediction-engine/src/models/` | **ACTIVE** — model deserialization |
| `calibration.py` | ~1,400 | Probability calibration (Platt Scaling, Isotonic Regression, Temperature Scaling, Beta Calibration). CalibrationMetrics, calibration curves | `packages/prediction-engine/src/models/` | **ACTIVE** — probability calibration |
| `simulation_engine.py` | ~1,380 | Monte Carlo game simulation. Possession-level state machine. 10,000+ simulations for score distributions, player stats, correlations | `packages/prediction-engine/src/models/` | **ACTIVE** — simulation engine |

### Betting & Edge Calculation

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `edge_quality.py` | ~840 | Edge quality scoring across 6 dimensions (ensemble agreement, line movement, feature stability, recency, situational, probability confidence). EdgeTier enum (ELITE through AVOID). Kelly multipliers per tier | `packages/betting-engine/src/edge/` | **ACTIVE** — edge quality |
| `risk_management.py` | ~1,240 | Capital preservation: drawdown protection, daily/weekly loss limits, dynamic position sizing, bankroll management, recovery mode, circuit breakers | `packages/betting-engine/src/bankroll/` | **ACTIVE** — risk management |
| `portfolio_optimizer.py` | ~720 | Multivariate Kelly Criterion with bet correlations. Quadratic optimization. Same-game/team/player correlation matrices. Max exposure limits | `packages/betting-engine/src/bankroll/` | **ACTIVE** — portfolio optimization |
| `bet_tracker.py` | ~990 | Comprehensive bet tracking (SQLite). BetStatus, BetType enums. ROI, P&L, CLV tracking. Performance by type/sportsbook | `packages/betting-engine/src/edge/` | **ACTIVE** — bet tracking |
| `monte_carlo.py` | ~490 | Monte Carlo simulation for betting performance analysis. ROI confidence intervals, probability of ruin, max drawdown percentiles | `packages/betting-engine/src/bankroll/` | **ACTIVE** — risk analysis |
| `prediction_optimizer.py` | ~420 | Performance optimization: parallel API calls, TTL-based caching, ThreadPoolExecutor. Cache TTLs: team_stats 6hr, player_stats 4hr, injury 15min, odds 5min | `packages/prediction-engine/src/inference/` | **ACTIVE** — performance |

### Backtesting

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `backtesting.py` | ~1,880 | Backtesting framework with walk-forward validation. BacktestSanityChecker (max 15% ROI, max 60% win rate). Betting simulation with juice/vig, CLV tracking | `packages/prediction-engine/src/backtesting/` | **ACTIVE** — backtesting core |
| `comprehensive_backtest.py` | ~2,080 | Point-in-time predictions for all completed games. SeasonBacktester class. Multi-season validation | `packages/prediction-engine/src/backtesting/` | **ACTIVE** — comprehensive backtest |
| `backtest.py` | ~610 | Walk-forward validation framework. PropBacktester, MoneylineBacktester. Brier Score, Log Loss, ECE, ROI metrics | `packages/prediction-engine/src/backtesting/` | **ACTIVE** — backtest utilities |
| `simple_backtest.py` | ~140 | Simplified backtest for meta-learner validation | `packages/prediction-engine/src/backtesting/` | **ACTIVE** — quick validation |

### Scheduling & Deployment

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `scheduled_retraining.py` | ~710 | APScheduler-based automated retraining. Full retrain every 7 days (Sun 2 AM), incremental every 3 days. Drift detection triggers. Pre-deployment validation | `packages/prediction-engine/src/training/` | **ACTIVE** — production scheduler |
| `scheduled_retrain.py` | ~260 | Simpler scheduled retraining with full/quick modes, --check flag. R-squared degradation thresholds | `packages/prediction-engine/src/training/` | **ACTIVE** — retraining entry point |
| `railway_retrain.py` | ~100 | Railway cron job entrypoint for weekly retraining. Saves retrain history | `packages/prediction-engine/src/training/` | **ACTIVE** — Railway cron |
| `odds_tracker_service.py` | ~280 | Background daemon fetching odds every 5 min during operating hours (8 AM - 11 PM EST). APScheduler | `packages/betting-engine/src/odds/` | **ACTIVE** — odds daemon |
| `closing_odds_scheduler.py` | ~360 | Captures closing odds 5 min before game start for CLV calculation. Daemon/single check modes | `packages/betting-engine/src/odds/` | **ACTIVE** — CLV capture |
| `upload_predictions_to_railway.py` | ~290 | Manual upload of predictions CSV to Railway PostgreSQL. Temporary workaround for cron deployment | `packages/api/src/services/` | **ACTIVE** — utility |
| `live_adjustments.py` | ~440 | Live in-game prediction adjustments. Blends pre-game model with current game state. Pace projection, spread/moneyline adjustment | `packages/prediction-engine/src/inference/` | **ACTIVE** — live adjustments |

### Reporting & Analysis

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `report_generator.py` | ~760 | Interactive HTML backtesting reports with Plotly. ROI curves, calibration plots, confidence buckets | `packages/prediction-engine/src/backtesting/` | **ACTIVE** — reporting |
| `prop_tracker.py` | ~610 | SQLite-based player prop prediction tracker. Records, settles, and analyzes prop predictions | `packages/betting-engine/src/edge/` | **ACTIVE** — prop tracking |
| `calculate_confidence_metrics.py` | ~360 | Confidence correlation, calibration curves, ECE from backtest results | `packages/prediction-engine/src/backtesting/` | **ACTIVE** — metrics |
| `database.py` | ~880 | SQLite database backend. Teams, players, games, stats, odds, injuries, bets. Point-in-time queries | `packages/data-pipeline/src/storage/` | **ACTIVE** — local database |
| `migrate_to_postgres.py` | ~230 | Migration script from SQLite to PostgreSQL | `packages/data-pipeline/src/storage/` | **ACTIVE** — migration utility |

### Training Variants & Enhanced Models

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `train_with_balldontlie.py` | ~690 | Training pipeline with Balldontlie data. BalldontlieDataCollector class. Point-in-time stats | `packages/prediction-engine/src/training/` | **ACTIVE** — training variant |
| `train_balldontlie_final.py` | ~470 | Standalone training using pure sklearn (avoids XGBoost compatibility issues) | `packages/prediction-engine/src/training/` | **SUPERSEDED** — use train_complete_balldontlie.py |
| `train_enhanced_v2.py` | ~450 | Enhanced training integrating Four Factors, injury impact, style clash. Post-forensic Jan 7 analysis | `packages/prediction-engine/src/training/` | **ACTIVE** — enhanced features |

### Phase-Specific Validation Scripts

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `phase2_backtest_with_confidence.py` | ~750 | Phase 2 backtest with confidence tier filtering | N/A (validation) | **HISTORICAL** |
| `phase2.5_bias_correction.py` | ~280 | Bias correction for Phase 2 underprediction | N/A (validation) | **HISTORICAL** |
| `phase2.5_missing_metrics.py` | ~410 | Phase 2 missing metrics calculation | N/A (validation) | **HISTORICAL** |
| `phase3_comprehensive_backtest.py` | ~1,000 | Phase 3 two-season backtest with quantile regression, Kelly sizing, stop-loss | N/A (validation) | **HISTORICAL** |
| `phase3_validation_backtest.py` | ~85 | Quick Phase 3 validation on January 2025 data | N/A (validation) | **HISTORICAL** |
| `generate_phase3_report.py` | ~350 | Phase 3 analysis report generator | N/A (validation) | **HISTORICAL** |
| `track_phase2_targets.py` | ~220 | Phase 2.5 target tracking | N/A (validation) | **HISTORICAL** |

### Fix & Diagnostic Scripts

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `fix2_recalculate_bias.py` | ~210 | Recalculates bias corrections after DNP fix | N/A (one-time fix) | **HISTORICAL** |
| `fix_api_predictions_endpoint.py` | ~160 | Temporary patch for PostgreSQL prediction reads | N/A (superseded) | **DEAD** |
| `apply_platt_scaling.py` | ~220 | Platt scaling calibration for confidence scores | N/A (validation) | **HISTORICAL** |
| `compare_backtest_results.py` | ~85 | Comparison of backtest results | N/A (validation) | **HISTORICAL** |
| `analyze_base_model_agreement.py` | ~210 | Diagnoses confidence mechanism via base model agreement | N/A (diagnostic) | **HISTORICAL** |
| `analyze_task_3.1_results.py` | ~260 | Task 3.1 results analysis | N/A (validation) | **HISTORICAL** |

### Validation Scripts

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `validate_fixes.py` | ~290 | Validates Phase 2.5 targets (RMSE, bias, confidence) | N/A (validation) | **HISTORICAL** |
| `validate_metalearner.py` | ~170 | Meta-learner vs weighted averaging A/B test | N/A (validation) | **HISTORICAL** |
| `validate_player_impact_integration.py` | ~210 | Player impact integration test | N/A (validation) | **HISTORICAL** |
| `validate_real_performance.py` | ~130 | End-to-end performance validation | N/A (validation) | **HISTORICAL** |
| `verify_deployment.py` | ~190 | Railway deployment verification | `scripts/` | **ACTIVE** |
| `verify_v3_integration.py` | ~100 | GameSimulatorV3 integration test | N/A (validation) | **HISTORICAL** |

### Utility & Misc

| File | Lines | Purpose | Target Package | Status |
|------|-------|---------|---------------|--------|
| `quick_test.py` | ~220 | Quick test of player prop models | N/A (development) | **DEVELOPMENT** |
| `fetch_props.py` | ~90 | Fetches/displays player prop predictions for specific games | N/A (utility) | **DEVELOPMENT** |
| `profile_daily_predictions.py` | ~100 | cProfile-based performance profiling | N/A (development) | **DEVELOPMENT** |

---

## Subdirectory Modules

### `backend/` — FastAPI REST API (2,063 lines)

| File | Lines | Purpose | Target Package |
|------|-------|---------|---------------|
| `api.py` | ~1,200 | FastAPI REST wrapper around DataService. CORS, auth, endpoints for predictions/injuries/line-movement/backtest. Health check at `/api/health` | `packages/api/src/routes/` |
| `schemas.py` | ~450 | Pydantic models for all API responses (GameResponse, PredictionResponse, InjuryResponse, etc.) | `packages/api/src/routes/` |
| `auth.py` | ~300 | JWT and API key authentication. Optional via `AUTH_ENABLED` env var | `packages/api/src/middleware/` |
| `requirements.txt` | — | Backend-specific dependencies | — |
| `bets.db` | — | SQLite database for bet tracking | — |

### `dashboard/` — Dash Dashboard (11,687 lines)

| File | Purpose | Target Package |
|------|---------|---------------|
| `app.py` | Dash application initialization and routing | `packages/dashboard/` |
| `data_service.py` | Central data service (predictions, games, odds, injuries, props). Largest file in dashboard | `packages/dashboard/src/hooks/` |
| `layouts.py` | Page layout definitions | `packages/dashboard/src/components/` |
| `callbacks.py` | Dash interactive callbacks | `packages/dashboard/src/components/` |
| `theme.py` | Dashboard theming/styling | `packages/dashboard/src/components/` |
| `components/navbar.py` | Navigation bar component | `packages/dashboard/src/components/` |
| `pages/predictions.py` | Daily predictions display page | `packages/dashboard/src/pages/` |
| `pages/performance.py` | Performance metrics and backtest results page | `packages/dashboard/src/pages/` |
| `pages/bankroll.py` | Bankroll management tracking page | `packages/dashboard/src/pages/` |
| `pages/tracker.py` | Prediction tracking and outcomes page | `packages/dashboard/src/pages/` |
| `assets/styles.css` | CSS styles | `packages/dashboard/` |

### `continuous_learning/` — Drift Detection & Incremental Training (2,151 lines)

| File | Purpose | Target Package |
|------|---------|---------------|
| `drift_detector.py` | Model performance drift detection | `packages/prediction-engine/src/training/` |
| `incremental_trainer.py` | Incremental model updates without full retraining | `packages/prediction-engine/src/training/` |
| `model_registry.py` | Model versioning and management | `packages/prediction-engine/src/training/` |
| `orchestrator.py` | Training orchestration | `packages/prediction-engine/src/training/` |
| `settlement_service.py` | Prediction settlement and outcome recording | `packages/betting-engine/src/edge/` |

### `minutes_oracle/` — Minutes Prediction Sub-Model (3,843 lines)

| File | Purpose | Target Package |
|------|---------|---------------|
| `minutes_predictor.py` | Quantile regression for minutes distribution (p10, p25, p50, p75, p90) | `packages/prediction-engine/src/models/` |
| `minutes_features.py` | 35 features across 5 categories (baseline, context, rotation, coach, situational) | `packages/prediction-engine/src/features/` |
| `minutes_trainer.py` | Training pipeline using 3+ seasons historical data | `packages/prediction-engine/src/training/` |
| `coach_tendencies.py` | All 30 NBA coaches' rotation patterns + CoachTendencyLearner | `packages/data-pipeline/src/transformers/` |
| `validation.py` | Calibration and coverage metrics | `packages/prediction-engine/src/backtesting/` |
| `integration.py` | Integration with existing prop predictor | `packages/prediction-engine/src/inference/` |
| `run_validation.py` | Validation runner | N/A (utility) |
| `INTEGRATION_GUIDE.md` | Usage documentation | `docs/` |

### `calibration_tracker/` — Calibration Feedback Loop (3,159 lines)

| File | Purpose | Target Package |
|------|---------|---------------|
| `calibration_service.py` | Core calibration service with bias correction | `packages/prediction-engine/src/models/` |
| `bias_analyzer.py` | Systematic bias analysis and correction | `packages/prediction-engine/src/models/` |
| `calibration_adjuster.py` | Adjustment mechanisms | `packages/prediction-engine/src/models/` |
| `prediction_logger.py` | Prediction logging for outcome tracking | `packages/betting-engine/src/edge/` |
| `outcome_tracker.py` | Outcome tracking and validation | `packages/betting-engine/src/edge/` |
| `database.py` | SQLite persistence for calibration data | `packages/data-pipeline/src/storage/` |
| `nightly_job.py` | Scheduled nightly calibration updates | `packages/prediction-engine/src/training/` |

### `edge_calculator/` — Edge & Kelly Calculation (2,824 lines)

| File | Purpose | Target Package |
|------|---------|---------------|
| `edge_calculator.py` | EV calculation for betting opportunities | `packages/betting-engine/src/edge/` |
| `kelly_criterion.py` | Kelly Criterion bet sizing (f* = (bp - q) / b) | `packages/betting-engine/src/bankroll/` |
| `bankroll_manager.py` | Bankroll management with stop-loss | `packages/betting-engine/src/bankroll/` |
| `bet_recommender.py` | Best bet filtering and recommendations | `packages/betting-engine/src/signals/` |
| `recommend.py` | Recommendation engine entry point | `packages/betting-engine/src/signals/` |

### `lineup_intel/` — Lineup & Injury Intelligence (2,516 lines)

| File | Purpose | Target Package |
|------|---------|---------------|
| `lineup_intel_service.py` | Injury and lineup impact analysis | `packages/agents/src/pregame/` |
| `lineup_tracker.py` | Real-time lineup tracking | `packages/data-pipeline/src/sources/` |
| `news_monitor.py` | News monitoring for lineup changes | `packages/agents/src/pregame/` |
| `injury_scraper.py` | Automated injury scraping | `packages/data-pipeline/src/sources/` |
| `integration.py` | Integration with prediction pipeline | `packages/prediction-engine/src/inference/` |

### `tests/` — Test Suite (13 files)

| File | Tests |
|------|-------|
| `test_advanced_stats.py` | Advanced stats validation |
| `test_betting_features.py` | Betting features and odds tracking |
| `test_best_bets_filter.py` | Best bets filter logic |
| `test_confidence_scoring.py` | Confidence scoring validation |
| `test_injury_tracker.py` | Injury tracking (IGNORED in pytest.ini) |
| `test_odds_tracker_service.py` | Odds tracker service |
| `test_player_impact.py` | Player impact metrics |
| `test_quantile_models.py` | Quantile model validation |
| `test_report_generator.py` | Report generation |
| `test_risk_management.py` | Risk management calculations |
| `test_scheduled_retraining.py` | Retraining scheduler |
| `test_stacking.py` | Stacking ensemble integration |
| `test_travel_fatigue.py` | Travel fatigue calculations |

### Root-level test files (13 files)

| File | Tests |
|------|-------|
| `test_advanced_stats.py` | Advanced stats unit tests |
| `test_api_direct.py` | Direct API testing |
| `test_confidence_formulas.py` | Confidence formula validation |
| `test_daily_predictions_integration.py` | Daily predictions integration |
| `test_darko_google_sheet.py` | DARKO data source testing |
| `test_deployment_config.py` | Railway deployment verification |
| `test_four_factors.py` | Four Factors calculations |
| `test_injury_integration.py` | Injury data integration |
| `test_optimization.py` | Optimization testing |
| `test_props.py` | Props prediction testing |
| `test_props_unlocked.py` | Unlocked props testing |
| `test_stacking_integration.py` | Stacking integration testing |
| `test_task_3_4_implementation.py` | Task 3.4 implementation |
| `test_task_4_4_endpoints.py` | API endpoint testing |

### `scripts/` — Utility Scripts

| File | Purpose |
|------|---------|
| `quick_verify.sh` | Fast verification script (syntax/linting) |
| `debug/list_missing_features.py` | Lists features expected by models but missing at prediction time |
| `debug/check_all_models.py` | Checks all trained model files for consistency |
| `debug/deep_dive_threes.py` | Deep analysis of threes model performance |
| `debug/diagnose_threes_model.py` | Diagnoses issues with threes prediction model |
| `debug/fetch_backtest_data.py` | Fetches data for backtesting |
| `debug/quick_model_validation.py` | Quick model validation checks |
| `debug/verify_feature_consistency.py` | Verifies feature consistency between training and prediction |

### `analysis/` — Forensic Analysis

| File | Purpose |
|------|---------|
| `forensic_jan7.py` | Forensic analysis of model failures on January 7, 2026 |

---

## Frontend

### `frontend/` — React 19 + TypeScript (8,127 lines)

**Tech Stack:** React 19.2, TypeScript 5.9, Vite 7.2, TailwindCSS 4.1, React Router 7.11, React Query 5.90

| File | Purpose |
|------|---------|
| `src/pages/Tracker.tsx` | Prediction tracking page |
| `src/pages/Performance.tsx` | Performance metrics page |
| `src/pages/Bankroll.tsx` | Bankroll management page |
| `src/components/layout/Navbar.tsx` | Navigation bar |
| `src/components/layout/Layout.tsx` | Page layout wrapper |
| `src/components/ui/Card.tsx` | Card component |
| `src/components/ui/Badge.tsx` | Badge component |
| `src/components/ui/StatCard.tsx` | Statistics card component |
| `src/components/predictions/ConfidenceBar.tsx` | Confidence level visualization |
| `src/components/predictions/EdgeBadge.tsx` | Edge quality badge |
| `vite.config.ts` | Vite build configuration |
| `eslint.config.js` | ESLint configuration |
| `tsconfig.json` | TypeScript configuration |
| `index.html` | Entry point HTML |

**Dependencies:**
- `@tanstack/react-query` ^5.90.12 — Data fetching and caching
- `axios` ^1.13.2 — HTTP client
- `lucide-react` ^0.562.0 — Icon library
- `react` ^19.2.0 — UI framework
- `react-dom` ^19.2.0 — DOM rendering
- `react-router-dom` ^7.11.0 — Client-side routing

---

## Configuration & Deployment

### Config Files

| File | Purpose |
|------|---------|
| `.env.example` | Template with all 25+ environment variables documented |
| `.env` | Actual secrets (NOT in repo, 218 bytes) |
| `.gitignore` | Ignores .env, __pycache__, node_modules, .db files, caches, logs, .claude/ |
| `Procfile` | Railway web process: `python -m uvicorn backend.api:app --host 0.0.0.0 --port $PORT` |
| `railway.toml` | Railway deployment config with multi-service strategy (API + 3 workers) |
| `railway-cron.yml` | Cron job documentation (Railway uses dashboard, not YAML) |
| `pytest.ini` | Pytest config: tests/ directory, ignores test_injury_tracker.py, strict markers |
| `ruff.toml` | Ruff linter config: line length 100, comprehensive rules (E, W, F, B, C4, UP, SIM, RET, PTH, PD, NPY) |
| `requirements.txt` | Python dependencies (31 packages) |
| `frontend/package.json` | Frontend Node.js dependencies |
| `com.nba-betting-model.retrain.plist` | macOS LaunchAgent for local scheduled retraining |
| `setup_scheduler.sh` | Sets up local macOS scheduler |
| `check_backtest_status.sh` | Shell script to check backtest progress |

### CI/CD (`.github/workflows/`)

| File | Purpose |
|------|---------|
| `quality-checks.yml` | Lint and test on pull requests |
| `weekly-retrain.yml` | Weekly model retraining workflow |

### Deployment Architecture

Railway runs 4 services:
1. **API Service** — FastAPI on Uvicorn (port $PORT)
2. **Daily Predictions** — Cron: 9 AM EST daily (`python daily_predictions.py`)
3. **Odds Tracker** — Daemon: every 5 min, 8 AM - 11 PM EST (`python odds_tracker_service.py --daemon`)
4. **Retraining Scheduler** — Daemon: full retrain Sun 2 AM, incremental every 3 days (`python scheduled_retraining.py --daemon`)

Vercel runs the React frontend.

---

## Data Artifacts

### Databases

| File | Type | Purpose | Size |
|------|------|---------|------|
| `nba_betting.db` | SQLite | Main database (teams, players, games, stats, odds) | 135 KB |
| `prop_predictions.db` | SQLite | Player prop prediction history | 1.7 MB |
| `predictions.db` | SQLite | Prediction storage (empty) | 0 bytes |
| `bets.db` | SQLite | Bet tracking | 24 KB |
| `backend/bets.db` | SQLite | Backend bet tracking | — |
| `data/calibration.db` | SQLite | Calibration metrics | — |
| `data/bankroll.db` | SQLite | Bankroll management | — |

### Cache Directories

| Directory | Purpose | In .gitignore? |
|-----------|---------|---------------|
| `.api_cache/` | General API response cache (715 entries) | YES |
| `.bdl_cache/` | Balldontlie API response cache | YES |
| `.news_cache/` | News article cache | NO |
| `.odds_cache/` | Odds data cache (empty) | NO |
| `.tracking_cache/` | Tracking data cache | NO |
| `player_impact_cache/` | Player impact metrics cache | NO |
| `catboost_info/` | CatBoost training artifacts | NO |

### Backtest Results (`backtest_results/`)

29 JSON files + 2 PNG files containing results from different phases and configurations. Key files:
- `phase2_backtest.json` — Phase 2 validation results
- `phase3_backtest_2seasons.json` — Phase 3 comprehensive backtest
- `calibration_curve.png`, `confidence_vs_error.png` — Visualization artifacts

### Training Metrics (`training_metrics/`)

46 files tracking training run metrics across model iterations.

### Bet Slips

| File | Date |
|------|------|
| `bet_slip_2025-12-12.json` | Dec 12, 2025 |
| `bet_slip_2025-12-18.json` | Dec 18, 2025 |
| `bet_slip_2025-12-27.json` | Dec 27, 2025 |
| `bet_slip_2026-01-05.json` | Jan 5, 2026 |
| `bet_slip_2026-01-06.json` | Jan 6, 2026 |

### Prediction CSVs

| File | Date |
|------|------|
| `predictions_2026-01-20.csv` | Jan 20, 2026 |
| `predictions_2026-01-21.csv` | Jan 21, 2026 |
| `predictions_2026-01-27.csv` | Jan 27, 2026 |

### Backtest Result Files (Large)

| File | Size |
|------|------|
| `backtest_results_2025.json` | 18.4 MB |
| `backtest_results_2025_quick.json` | 6.0 MB |

### Other Data Files

| File | Purpose |
|------|---------|
| `data/nba_data.zip` | Historical NBA data archive (likely Kaggle dataset) |
| `games_dump.json` | Games data dump |
| `nba_schedule_2025-12-13.json` | NBA schedule snapshot |
| `coverage.xml` | Test coverage report |

---

## Documentation & Planning Files

### Root-level Documentation

| File | Purpose |
|------|---------|
| `COMPLETE_DIAGNOSTIC_REPORT.md` | System component test results and root cause analysis |
| `QUALITY_SYSTEM.md` | 5-layer defense system for minimizing bugs |
| `RAILWAY_CRON_SETUP.md` | Railway deployment guide |
| `RAILWAY_CRON_FIX_COMPLETE.md` | Cron fix documentation |
| `WHY_NO_PREDICTIONS_ON_VERCEL.md` | Frontend debugging guide |
| `DAILY_PREDICTIONS_SUMMARY_2026-01-21.md` | Example predictions output |
| `CONFIDENCE_CALIBRATION_ANALYSIS.txt` | Confidence analysis results |
| `PATTERN_RISK_ANALYSIS.txt` | Pattern risk analysis |
| `todo.md` | Current project task tracking |

### Improvement Plans (7 iterations)

| Directory | Content |
|-----------|---------|
| `improvement_plan/` | Original plan (implementation_plan.md, tasks.md) |
| `improvement_plan_complete/` | Completed analysis (analysis.md, implementation_plan.md, tasks.md) |
| `improvement_plan_v3/` | V3 plan (analysis.md, implementation_plan.md, tasks.md) |
| `improvement_plan_v3_verification/` | V3 verification (verification_report.md, verify_v3_integration.py) |
| `improvement_plan_v4/` | V4 plan (analysis.md, implementation_plan.md, tasks.md) |
| `improvement_plan_v5/` | V5 plan (analysis.md, implementation_plan.md, tasks.md) |
| `improvement_plan_v6/` | V6 plan (implementation_plan.md, tasks.md) |
| `improvement_plan_v7_detailed/` | V7 detailed plan with forensic analysis |
| `improvement_plan_v7_fix/` | V7 fix instructions |
| `improvement_plan_v7_fix_v2/` | V7 fix v2 instructions |
| `redesign_v2/` | Redesign brief, implementation plan, debug report, deployment report |

### Backtest Output Files (Root)

| File | Purpose |
|------|---------|
| `backtest_baseline_output.txt` | Baseline backtest results |
| `backtest_fix1_output.txt` | Fix 1 backtest results |
| `backtest_fix2_final.txt` | Fix 2 final results |
| `backtest_fix2_output.txt` | Fix 2 raw output |
| `backtest_results_full.txt` | Full backtest results |
| `base_model_agreement_output.txt` | Base model agreement analysis |
| `feature_ablation_output.txt` | Feature ablation results |
| `feature_ablation_quick_test.txt` | Quick ablation test |
| `phase2_recalibrated_output.txt` | Phase 2 recalibration results |
| `phase2_solution1_final.txt` | Phase 2 solution 1 results |
| `phase2_tree_based_output.txt` | Phase 2 tree-based results |
| `task2.6_output.txt` | Task 2.6 results |

---

## Environment Variables

### Required

| Variable | Used By | Purpose |
|----------|---------|---------|
| `BALLDONTLIE_API_KEY` | data_fetcher.py, daily_predictions.py, balldontlie_api.py, fast_data_fetcher.py, train_complete_balldontlie.py | Balldontlie API authentication (GOAT tier recommended) |
| `DATABASE_URL` | daily_predictions.py, upload_predictions_to_railway.py, migrate_to_postgres.py, backend/api.py | PostgreSQL connection string (Railway auto-provisions) |

### Optional — API Keys

| Variable | Used By | Purpose |
|----------|---------|---------|
| `THE_ODDS_API_KEY` | odds_fetcher.py, betting_market_features.py | The Odds API authentication |
| `ROTOWIRE_API_KEY` | injury_tracker_v3.py | RotoWire injury data (fallback to NBA.com) |
| `ANTHROPIC_API_KEY` | news_sentiment.py | Claude API for sentiment analysis |

### Optional — Authentication

| Variable | Used By | Purpose |
|----------|---------|---------|
| `AUTH_ENABLED` | backend/auth.py | Enable JWT authentication (default: false) |
| `JWT_SECRET_KEY` | backend/auth.py | JWT secret for authentication |
| `JWT_ALGORITHM` | backend/auth.py | JWT algorithm (default: HS256) |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | backend/auth.py | Token expiry (default: 30 min) |
| `API_KEY` | backend/auth.py | API key for X-API-Key header auth |

### Optional — Configuration

| Variable | Used By | Purpose |
|----------|---------|---------|
| `FRONTEND_URL` | backend/api.py | CORS whitelist for frontend |
| `ALERT_EMAIL` | scheduled_retraining.py | Error alert email |
| `SLACK_WEBHOOK` | scheduled_retraining.py | Slack webhook for alerts |
| `MAX_TRAINING_TIME` | scheduled_retraining.py | Training timeout (default: 7200s) |
| `MIN_DAYS_BETWEEN_FULL_RETRAIN` | — | Days between full retrains (default: 14) |
| `MIN_DAYS_BETWEEN_INCREMENTAL` | — | Days between incremental updates (default: 3) |
| `PERFORMANCE_DEGRADATION_THRESHOLD` | — | Drift threshold (default: 0.05) |
| `ODDS_UPDATE_INTERVAL` | — | Odds update interval in minutes (default: 5) |
| `ODDS_TRACKER_START_HOUR` | — | Odds tracker start hour EST (default: 8) |
| `ODDS_TRACKER_END_HOUR` | — | Odds tracker end hour EST (default: 23) |
| `DAILY_PREDICTIONS_HOUR` | — | Daily predictions hour EST (default: 9) |
| `MIN_BET_CONFIDENCE` | — | Minimum confidence for bets (default: 75) |
| `LOG_LEVEL` | — | Logging level (default: INFO) |
| `LOG_DIR` | — | Log directory (default: logs) |
| `USE_V3_SIMULATION` | daily_predictions.py | Feature flag for V3 simulation (default: '1') |
| `USE_SQLITE` | — | Use local SQLite (default: false) |
| `SQLITE_PATH` | — | Local SQLite path (default: nba_betting.db) |
| `DEBUG` | — | Debug mode (default: false) |
| `SKIP_MODEL_LOADING` | — | Skip model loading on startup (default: false) |

---

## External Dependencies

### Python (from requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| **fastapi** | >=0.104.0 | REST API framework |
| **uvicorn** | >=0.24.0 | ASGI server |
| **pydantic** | >=2.5.0 | Data validation |
| **python-jose** | >=3.3.0 | JWT authentication |
| **passlib** | >=1.7.4 | Password hashing |
| **python-multipart** | >=0.0.6 | Form data parsing |
| **numpy** | >=1.24.0 | Numerical computation |
| **pandas** | >=2.0.0 | Data processing |
| **scikit-learn** | >=1.3.0 | ML models (base) |
| **lightgbm** | >=4.0.0 | Gradient boosting |
| **catboost** | >=1.2.0 | Gradient boosting |
| **xgboost** | >=2.0.0 | Gradient boosting |
| **scipy** | >=1.11.0 | Statistics & optimization |
| **requests** | >=2.31.0 | HTTP client |
| **apscheduler** | >=3.10.0 | Background job scheduling |
| **python-dotenv** | >=1.0.0 | Environment variable loading |
| **nba_api** | >=1.4.1 | NBA stats API wrapper |
| **dash** | >=2.14.0 | Dashboard framework |
| **dash-bootstrap-components** | >=1.5.0 | Dashboard styling |
| **plotly** | >=5.18.0 | Data visualization |
| **jinja2** | >=3.1.0 | HTML templating |
| **beautifulsoup4** | >=4.12.0 | Web scraping |
| **lxml** | >=5.0.0 | XML/HTML parsing |
| **psycopg2-binary** | >=2.9.0 | PostgreSQL driver |
| **pytest** | >=7.4.0 | Testing framework |
| **pytest-asyncio** | >=0.21.0 | Async test support |
| **pytest-cov** | >=4.1.0 | Test coverage |
| **pytest-timeout** | >=2.1.0 | Test timeouts |
| **ruff** | >=0.1.0 | Fast linter |

### Frontend (from package.json)

| Package | Version | Purpose |
|---------|---------|---------|
| **react** | ^19.2.0 | UI framework |
| **react-dom** | ^19.2.0 | DOM rendering |
| **react-router-dom** | ^7.11.0 | Client-side routing |
| **@tanstack/react-query** | ^5.90.12 | Data fetching/caching |
| **axios** | ^1.13.2 | HTTP client |
| **lucide-react** | ^0.562.0 | Icon library |
| **tailwindcss** | ^4.1.18 | CSS framework |
| **vite** | ^7.2.4 | Build tool |
| **typescript** | ~5.9.3 | Type system |

---

## External API Integrations

| Service | Module(s) | Purpose | Rate Limits |
|---------|-----------|---------|-------------|
| **Balldontlie.io** | balldontlie_api.py, data_fetcher.py, fast_data_fetcher.py | Game data, player stats, schedules, betting odds, live box scores | 600 req/min (GOAT tier) |
| **NBA API (nba_api)** | data_fetcher.py, tracking_data.py, historical_data_collector.py | Team stats, player stats, PBP, shot charts (fallback) | ~600ms between requests |
| **The Odds API** | odds_fetcher.py, betting_market_features.py | Real-time odds from 40+ sportsbooks | 500 req/month (free tier) |
| **RotoWire** | injury_tracker_v3.py | Real-time injury reports | Paid API |
| **DARKO (APAnalytics)** | player_impact_fetcher.py | Daily Plus-Minus player ratings | Web scraping |
| **ESPN EPM** | player_impact_fetcher.py | Estimated Plus-Minus | Web scraping |
| **FiveThirtyEight RAPTOR** | player_impact_fetcher.py | Player impact ratings | Web scraping |
| **Anthropic Claude** | news_sentiment.py | News sentiment analysis | API key required |
| **Railway PostgreSQL** | backend/api.py, daily_predictions.py | Production database | Auto-provisioned |

---

## Trained Models Inventory

### Active Production Models (`models/`)

| File | Type | Purpose |
|------|------|---------|
| `moneyline_ensemble.pkl` | Ensemble classifier | Moneyline prediction (primary) |
| `spread_ensemble.pkl` | Ensemble regressor | Spread prediction |
| `spread_svm_regressor.pkl` | SVM regressor | Spread prediction (legacy) |
| `spread_cover_classifier.pkl` | Classifier | Spread cover prediction |
| `spread_quantile.pkl` | Quantile regressor | Spread with uncertainty |
| `player_points_ensemble.pkl` | Ensemble regressor | Points prediction |
| `player_rebounds_ensemble.pkl` | Ensemble regressor | Rebounds prediction |
| `player_assists_ensemble.pkl` | Ensemble regressor | Assists prediction |
| `player_threes_ensemble.pkl` | Ensemble regressor | Three-pointers prediction |
| `player_pra_ensemble.pkl` | Ensemble regressor | PRA prediction |
| `player_points_quantile.pkl` | Quantile regressor | Points with uncertainty |
| `player_rebounds_quantile.pkl` | Quantile regressor | Rebounds with uncertainty |
| `player_assists_quantile.pkl` | Quantile regressor | Assists with uncertainty |
| `player_threes_quantile.pkl` | Quantile regressor | Threes with uncertainty |
| `player_pra_quantile.pkl` | Quantile regressor | PRA with uncertainty |
| `player_assists_position_aware.pkl` | Position-aware | Assists by position |
| `player_rebounds_position_aware.pkl` | Position-aware | Rebounds by position |
| `player_minutes_model.pkl` | Regressor | Minutes prediction |
| `moneyline_stacking.pkl` | Stacking ensemble | Moneyline stacking |
| `spread_stacking.pkl` | Stacking ensemble | Spread stacking |
| `moneyline_stacking_metalearner.pkl` | Meta-learner | Moneyline meta-learner |
| `spread_stacking_metalearner.pkl` | Meta-learner | Spread meta-learner |
| `moneyline_stacking_baseline.pkl` | Baseline | Moneyline baseline for comparison |
| `spread_stacking_baseline.pkl` | Baseline | Spread baseline for comparison |
| `minutes_oracle.pkl` | Quantile regressor | Minutes Oracle sub-model |
| `stacking_model.py` | Python | Stacking model definition |

### Calibration Models (`models/calibration/`)

32 calibration model versions for different prop types and calibration methods.

### Legacy/Backup Models

| File | Status |
|------|--------|
| `moneyline_logistic_regression.pkl` | Superseded by ensemble |
| `moneyline_gradient_boosting.pkl` | Superseded by ensemble |
| `moneyline_ensemble_tuned.pkl` | Superseded by current ensemble |
| `player_*_enhanced.pkl` (5 files) | Superseded by ensemble versions |
| `player_*.pkl` (5 base files) | Superseded by ensemble versions |
| `player_*_ensemble.pkl.with_travel` (4 files) | Backup before travel feature removal |
| `player_*_stacking_BROKEN_5features.pkl.backup` (4 files) | Known broken backups |
| `*_optuna_params.json` (5 files) | Hyperparameter search results |
| `enhanced_training_summary.json` | Training summary |

---

## Dead Code & Candidates for Removal

### Confirmed Dead Code

| File/Directory | Reason |
|---------------|--------|
| `fix_api_predictions_endpoint.py` | Temporary patch replaced by new database backend |
| `models/player_*_stacking_BROKEN_5features.pkl.backup` (4 files) | Explicitly marked as BROKEN |
| `predictions.db` | Empty file (0 bytes) |
| `backtest_baseline_output.txt` | Only 267 bytes, likely failed run |

### Historical/Superseded (Safe to Archive)

| File/Directory | Reason |
|---------------|--------|
| `improvement_plan/` through `improvement_plan_v7_fix_v2/` (10 dirs) | Historical planning documents — archive for reference |
| `redesign_v2/` | Historical redesign documentation |
| `phase2_*.py` (3 files) | Phase 2 validation scripts — work is complete |
| `phase3_*.py` (3 files) | Phase 3 validation scripts — work is complete |
| `fix2_recalculate_bias.py` | One-time fix script |
| `apply_platt_scaling.py` | One-time calibration script |
| `analyze_*.py` (2 files) | Diagnostic scripts from specific incidents |
| `validate_*.py` (4 files) | Validation scripts for completed phases |
| `verify_v3_integration.py` | Integration test for completed phase |
| `track_phase2_targets.py` | Phase 2 tracking — complete |
| `compare_backtest_results.py` | One-time comparison |
| `generate_phase3_report.py` | One-time report generation |
| `train_balldontlie_final.py` | Superseded by train_complete_balldontlie.py |
| `models/moneyline_logistic_regression.pkl` | Superseded by ensemble |
| `models/moneyline_gradient_boosting.pkl` | Superseded by ensemble |
| `models/moneyline_ensemble_tuned.pkl` | Superseded by current ensemble |
| `models/player_*_enhanced.pkl` (5 files) | Superseded by ensemble versions |
| `models/player_*.pkl` (5 base files) | Superseded by ensemble versions |
| `models/player_*_ensemble.pkl.with_travel` (4 files) | Pre-travel-removal backups |
| All root-level `.txt` output files (12 files) | Historical output artifacts |
| `bet_slip_*.json` (5 files) | Historical bet slip artifacts |
| `predictions_*.csv` (3 files) | Historical prediction artifacts |
| `backtest_results_2025.json` (18.4 MB) | Large historical backtest result |
| `backtest_results_2025_quick.json` (6.0 MB) | Large historical backtest result |
| `nba_schedule_2025-12-13.json` | Stale schedule snapshot |
| `games_dump.json` | One-time data dump |

### Potentially Redundant (Needs Investigation)

| Item | Question |
|------|----------|
| `advanced_stats.py` vs `advanced_stats_v2.py` | Do both coexist or does v2 supersede v1? |
| `injury_fetcher.py` vs `injury_tracker_v3.py` | Are both actively used or is one superseded? |
| `stacked_model_v2.py` vs `stacking_meta_learner.py` | Are both needed or redundant? |
| `train_enhanced_v2.py` vs `train_complete_balldontlie.py` | Does enhanced v2 feed into complete, or is it standalone? |
| `scheduled_retrain.py` vs `scheduled_retraining.py` vs `railway_retrain.py` | Three retraining scripts — which is canonical? |
| Root `test_*.py` files vs `tests/` directory | Duplicate test files at two locations |
| `database.py` (SQLite) vs PostgreSQL (DATABASE_URL) | Both active — SQLite for local, PostgreSQL for production? |
| `.news_cache/`, `.odds_cache/`, `.tracking_cache/` | Not in .gitignore — should they be? |
| `catboost_info/` | CatBoost training artifacts — should be gitignored |
| `player_impact_cache/` | Not in .gitignore — should be |

---

## Target Architecture Mapping

### `packages/data-pipeline/src/sources/` — Data Ingestion

| Current File | Migration Action |
|-------------|-----------------|
| `data_fetcher.py` | Move as-is |
| `balldontlie_api.py` | Move as-is |
| `fast_data_fetcher.py` | Move as-is |
| `odds_fetcher.py` | Move as-is |
| `historical_data_collector.py` | Move as-is |
| `kaggle_data_loader.py` | Move as-is |
| `live_season_fetcher.py` | Move as-is |
| `tracking_data.py` | Move as-is |
| `injury_fetcher.py` | Move as-is |
| `injury_tracker_v3.py` | Move as-is |
| `player_impact_fetcher.py` | Move as-is |
| `referee_data.py` | Move as-is |
| `id_mapping.py` | Move as-is |
| `lineup_intel/injury_scraper.py` | Move as-is |
| `lineup_intel/lineup_tracker.py` | Move as-is |

### `packages/data-pipeline/src/transformers/` — Feature Engineering

| Current File | Migration Action |
|-------------|-----------------|
| `feature_engineering.py` | Move as-is (5,300 lines — may split later) |
| `feature_generator.py` | Move as-is |
| `advanced_stats.py` | Move as-is |
| `advanced_stats_v2.py` | Move as-is |
| `injury_impact_v2.py` | Move as-is |
| `elo_ratings.py` | Move as-is |
| `travel_fatigue.py` | Move as-is |
| `minutes_oracle/coach_tendencies.py` | Move as-is |

### `packages/data-pipeline/src/validators/` — Data Quality

| Current File | Migration Action |
|-------------|-----------------|
| *None currently exists* | Create during Phase 2 (data validation layer) |

### `packages/data-pipeline/src/storage/` — Database

| Current File | Migration Action |
|-------------|-----------------|
| `database.py` | Move as-is |
| `migrate_to_postgres.py` | Move as-is |
| `calibration_tracker/database.py` | Move as-is |

### `packages/prediction-engine/src/models/` — Model Definitions

| Current File | Migration Action |
|-------------|-----------------|
| `model_trainer.py` | Move as-is |
| `model_classes.py` | Move as-is |
| `stacking_meta_learner.py` | Move as-is |
| `stacked_model_v2.py` | Move as-is |
| `calibration.py` | Move as-is |
| `simulation_engine.py` | Move as-is |
| `minutes_oracle/minutes_predictor.py` | Move as-is |
| `calibration_tracker/calibration_service.py` | Move as-is |
| `calibration_tracker/bias_analyzer.py` | Move as-is |
| `calibration_tracker/calibration_adjuster.py` | Move as-is |

### `packages/prediction-engine/src/features/` — Feature Selection

| Current File | Migration Action |
|-------------|-----------------|
| `feature_ablation_study.py` | Move as-is |
| `minutes_oracle/minutes_features.py` | Move as-is |

### `packages/prediction-engine/src/training/` — Model Training

| Current File | Migration Action |
|-------------|-----------------|
| `train_complete_balldontlie.py` | Move as-is (primary) |
| `train_models.py` | Move as-is (orchestrator) |
| `train_stacking_model.py` | Move as-is |
| `train_with_balldontlie.py` | Move as-is |
| `train_enhanced_v2.py` | Move as-is |
| `scheduled_retraining.py` | Move as-is |
| `scheduled_retrain.py` | Move as-is |
| `railway_retrain.py` | Move as-is |
| `minutes_oracle/minutes_trainer.py` | Move as-is |
| `calibration_tracker/nightly_job.py` | Move as-is |
| `continuous_learning/drift_detector.py` | Move as-is |
| `continuous_learning/incremental_trainer.py` | Move as-is |
| `continuous_learning/model_registry.py` | Move as-is |
| `continuous_learning/orchestrator.py` | Move as-is |

### `packages/prediction-engine/src/inference/` — Real-Time Prediction

| Current File | Migration Action |
|-------------|-----------------|
| `app.py` | Move as-is |
| `daily_predictions.py` | Move as-is |
| `live_adjustments.py` | Move as-is |
| `prediction_optimizer.py` | Move as-is |
| `minutes_oracle/integration.py` | Move as-is |
| `lineup_intel/integration.py` | Move as-is |

### `packages/prediction-engine/src/backtesting/` — Historical Validation

| Current File | Migration Action |
|-------------|-----------------|
| `backtesting.py` | Move as-is |
| `comprehensive_backtest.py` | Move as-is |
| `backtest.py` | Move as-is |
| `simple_backtest.py` | Move as-is |
| `report_generator.py` | Move as-is |
| `calculate_confidence_metrics.py` | Move as-is |
| `minutes_oracle/validation.py` | Move as-is |

### `packages/betting-engine/src/odds/` — Odds Analysis

| Current File | Migration Action |
|-------------|-----------------|
| `betting_market_features.py` | Move as-is |
| `market_microstructure.py` | Move as-is |
| `odds_tracker_service.py` | Move as-is |
| `closing_odds_scheduler.py` | Move as-is |

### `packages/betting-engine/src/edge/` — Edge Calculation

| Current File | Migration Action |
|-------------|-----------------|
| `edge_quality.py` | Move as-is |
| `bet_tracker.py` | Move as-is |
| `prop_tracker.py` | Move as-is |
| `edge_calculator/edge_calculator.py` | Move as-is |
| `calibration_tracker/prediction_logger.py` | Move as-is |
| `calibration_tracker/outcome_tracker.py` | Move as-is |
| `continuous_learning/settlement_service.py` | Move as-is |

### `packages/betting-engine/src/bankroll/` — Bankroll Management

| Current File | Migration Action |
|-------------|-----------------|
| `risk_management.py` | Move as-is |
| `portfolio_optimizer.py` | Move as-is |
| `monte_carlo.py` | Move as-is |
| `edge_calculator/kelly_criterion.py` | Move as-is |
| `edge_calculator/bankroll_manager.py` | Move as-is |

### `packages/betting-engine/src/signals/` — Signal Generation

| Current File | Migration Action |
|-------------|-----------------|
| `edge_calculator/bet_recommender.py` | Move as-is |
| `edge_calculator/recommend.py` | Move as-is |

### `packages/agents/src/pregame/` — Pre-Game Intelligence

| Current File | Migration Action |
|-------------|-----------------|
| `news_sentiment.py` | Move as-is (proto-agent) |
| `lineup_intel/lineup_intel_service.py` | Move as-is |
| `lineup_intel/news_monitor.py` | Move as-is |

### `packages/api/src/` — Backend API

| Current File | Migration Action |
|-------------|-----------------|
| `backend/api.py` | Move to `routes/` |
| `backend/schemas.py` | Move to `routes/` |
| `backend/auth.py` | Move to `middleware/` |
| `upload_predictions_to_railway.py` | Move to `services/` |

### `packages/dashboard/` — Dash Dashboard

| Current Directory | Migration Action |
|-------------------|-----------------|
| `dashboard/*` | Move entire directory |

---

## Key Architectural Patterns

### 1. Temporal Discipline (Anti-Look-Ahead Bias)
- `before_date` parameters throughout data fetchers and feature engineering
- `_before_date` variants of fetch functions
- Critical for backtesting integrity
- Implemented in: data_fetcher.py, feature_engineering.py, elo_ratings.py

### 2. Smart Feature Defaults
- `PREDICTION_FEATURE_DEFAULTS` dict uses NBA-realistic values (not zeros)
- e.g., player_pts_avg: 10.0, off_rating: 114.0
- Prevents model degradation when features are missing

### 3. Dual Data Source Strategy
- Primary: Balldontlie API (600 req/min, faster, GOAT tier recommended)
- Fallback: NBA API (rate-limited, slower)
- Auto-detection via `*_auto()` functions

### 4. Stacking Ensemble Architecture
- Level 0: Diverse base models (XGBoost, LightGBM, RF, Ridge, NB, QDA)
- Level 1: Meta-learner with out-of-fold predictions
- TimeSeriesSplit cross-validation (respects temporal order)
- Time-decay sample weighting (recent games weighted more)

### 5. Probability Calibration Pipeline
- Raw model outputs -> calibration (Platt/Isotonic/Temperature/Beta) -> calibrated probabilities
- Critical for Kelly criterion bet sizing

### 6. Risk Management Circuit Breakers
- BacktestSanityChecker: max 15% ROI, max 60% win rate (flags data leakage)
- Drawdown protection, daily/weekly loss limits
- Recovery mode with graduated return to full stakes

### 7. Multi-Tier Caching
- Live data: 1 minute TTL
- Daily data: 30 minute TTL
- Stats: 1 hour TTL
- Historical: 24 hour TTL
- Disk-based caching in `.api_cache/`, `.bdl_cache/`

---

## Known Issues & Technical Debt

### From CLAUDE.md "What's Broken"

1. **No dedicated minutes prediction model** — stat predictions assume average minutes, causing massive variance when actual minutes differ
   - **STATUS:** `minutes_oracle/` module exists but may not be fully integrated into production prediction pipeline

2. **Stale context at inference time** — model doesn't incorporate same-day injury reports, lineup confirmations, or late-breaking news
   - **STATUS:** `lineup_intel/` module exists; `injury_tracker_v3.py` has 15-min refresh; needs validation

3. **Predicting raw stat values instead of edges** — model outputs raw predictions but doesn't compare to market line for value
   - **STATUS:** `edge_calculator/` module exists; `edge_quality.py` is active; P(Over|Line) model not yet built

4. **No calibration feedback loop** — don't track whether 60% confidence predictions actually hit 60%
   - **STATUS:** `calibration_tracker/` module exists with bias_analyzer, outcome_tracker; needs validation

5. **Feature bloat** — hundreds of features with near-zero importance increasing overfitting risk
   - **STATUS:** `feature_ablation_study.py` exists; audit needed to verify pruning was done

6. **Edge calculation bug** — underdog spread direction not handled correctly
   - **STATUS:** `app.py` contains `determine_spread_bet_side()` — needs audit

### Structural Issues

- **Flat file structure** — 75 Python files at root level with no package organization
- **Duplicate code** — Multiple training scripts with overlapping logic
- **Test fragmentation** — Tests split between `tests/` and root level
- **Cache directories not fully gitignored** — `.news_cache/`, `.odds_cache/`, `.tracking_cache/`, `player_impact_cache/`, `catboost_info/` should be in .gitignore
- **Multiple SQLite databases** — 5+ .db files with overlapping purposes
- **No structured logging** — Still using print() and basic logging (not JSON structured)
- **No health check endpoint** — Procfile runs the API but railway.toml references `/api/health` (verify it exists)
- **Large files** — feature_engineering.py (5,300 lines), model_trainer.py (5,500 lines), train_complete_balldontlie.py (6,600 lines) need eventual decomposition

---

## Summary

This is a sophisticated, production-grade NBA betting prediction system with ~112,000 lines of code. The core prediction pipeline works: data fetching -> feature engineering -> model training -> inference -> edge calculation -> bet sizing. The system is deployed on Railway (backend) and Vercel (frontend).

**Strengths:**
- Robust temporal discipline preventing data leakage
- Comprehensive ensemble ML architecture
- Multiple data source redundancy
- Advanced risk management and Kelly criterion sizing
- Active backtesting framework with sanity checks

**Primary Migration Challenges:**
- 75 Python files at root level need reorganization into packages
- Multiple overlapping training scripts need consolidation
- New modules (minutes_oracle, calibration_tracker, edge_calculator, lineup_intel) are built but uncommitted and integration status unclear
- Test organization needs unification
- Structured logging needs implementation
- Health check endpoint needs verification

**Files to Move:** ~90 active Python modules
**Files to Archive:** ~30 historical/superseded files
**Files to Delete:** ~5 confirmed dead files
