# NBA-BETS Full System Audit Summary

**Audit Date:** 2026-03-17
**Auditor:** Principal Engineer / Quantitative Betting Systems Auditor
**Repo:** NBA-BETS
**Verdict:** Current profitability claims are unverifiable. The system has strong engineering foundations but critical leakage, parity, and market-realism issues invalidate all existing backtest results. The model may or may not have genuine edge — we cannot tell from current evidence.

---

## Repository Scale

| Metric | Count |
|--------|-------|
| Python files | 200+ |
| Root-level Python files (mostly dead) | 65 |
| Test files | 46 |
| Trained model artifacts | 39 |
| Active subsystems | 10+ |
| Lines of critical code | ~30,000 |
| Active workflows | 3 |
| Railway services | 7 |
| Archive/legacy directories | 13+ |

## Issue Summary

| Severity | Count | Categories |
|----------|-------|------------|
| CRITICAL | 7 | In-sample backtest, simulated lines, post-hoc bias corrections, bias corrections in production, calibration pipeline mismatch, no model quality gate, feature generator drift |
| HIGH | 12 | Availability asymmetry, CLV not tracked, stale-line blindness, low-minute threshold inconsistency, model loading divergence, edge computation difference, decompression parameters, empirical calibration, confidence multipliers, OT settlement, missing code version in artifacts, pickle fragility |
| MEDIUM | 10 | Feature selection enforcement, spread sign convention, temporal holdout CI, settlement timestamps, flat vs compounding Kelly, dual minutes paths, minutes oracle overlap, book-level data, execution simulation, missing scratch handling |
| LOW | 5 | OT-normalized stats, feature imputation, backup strategy, workflow naming, alternate lines |

## Cross-Reference to Audit Documents

| Document | Key Findings |
|----------|-------------|
| REPO_ARCHITECTURE_MAP.md | 65 root-level Python files (40+ dead), 14 training entry points, feature generation in 3+ locations, edge calculation in 4 places |
| TRAIN_INFER_BACKTEST_PARITY_AUDIT.md | 10 parity issues. Feature gen, calibration, edge calc, bias corrections, availability — ALL different across train/inference/backtest |
| LEAKAGE_AUDIT.md | 4 CRITICAL leakage issues: in-sample model, simulated lines, post-hoc bias corrections, production bias corrections. 3 HIGH issues. |
| MARKET_REALISM_AUDIT.md | 4 CRITICAL/HIGH issues: simulated lines, fixed -110 odds, no CLV tracking, no decision-time capture |
| AVAILABILITY_DNP_MINUTES_AUDIT.md | Production has good availability gating; backtests have none. Scratch handling missing everywhere. 3 different minute thresholds. |
| ARTIFACT_AND_DEPLOYMENT_AUDIT.md | No model quality gate, no provenance tracking, pickle fragility, models committed to git |
| TEST_COVERAGE_GAP_AUDIT.md | 7 critical test gaps. CI coverage only measures backend/. No train/inference parity test. No settlement test. |
| PRIORITIZED_EXECUTION_PLAN.md | 5 phases over 6 weeks. Phase 0: model quality gate + fix in-sample backtest. Phase 1: real lines + unified features. |

## What the System Does Well

1. **Point-in-time feature discipline** — Training and backtest feature generators use `*_before_date()` functions to prevent feature-level temporal leakage.
2. **Canonical constants** — `nba_betting/constants.py` serves as single source of truth for thresholds, std devs, Kelly fractions.
3. **Multi-layer calibration** — Isotonic regression, temperature scaling, sample-size shrinkage, quantile decompression all exist.
4. **Comprehensive agent architecture** — 6 specialized agents with guardrails, message bus, and scheduling.
5. **Production DNP filtering** — OUT/DOUBTFUL players are hard-filtered before prediction generation.
6. **Sanity checks** — `BacktestSanityChecker` flags unrealistic ROI/win rates as potential leakage.
7. **Disabled props** — System correctly disables prop types where it has no demonstrated edge.
8. **Quantile uncertainty** — Quantile regression provides prediction intervals, not just point estimates.
9. **Devigging infrastructure** — Proper multiplicative devig for computing true edges against no-vig probabilities.
10. **Railway deployment** — Well-structured 7-service deployment with cron scheduling.

## What Must Be Fixed Before Trusting Any Profitability Claim

1. **Train/test temporal separation** — Model must not train on test period data
2. **Real sportsbook lines** — Backtests must use real lines, not simulated from features
3. **Unified prediction pipeline** — Backtest must use identical code path as production
4. **Out-of-sample bias corrections** — Corrections must be fit on held-out data
5. **CLV tracking** — Primary model quality metric must be implemented
6. **Model quality gate** — No deploying untested models to production
