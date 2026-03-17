# Changed Files — Prompt 02

**No production code modified.**

## Copied Files (Prompt 1 → Prompt 2 handoff)

| From | To |
|------|----|
| `AUDIT_SUMMARY.md` | `review_handoff/prompt_02/prompt_01_outputs/AUDIT_SUMMARY.md` |
| `PRIORITIZED_EXECUTION_PLAN.md` | `review_handoff/prompt_02/prompt_01_outputs/PRIORITIZED_EXECUTION_PLAN.md` |
| `LEAKAGE_AUDIT.md` | `review_handoff/prompt_02/prompt_01_outputs/LEAKAGE_AUDIT.md` |
| `MARKET_REALISM_AUDIT.md` | `review_handoff/prompt_02/prompt_01_outputs/MARKET_REALISM_AUDIT.md` |
| `REPO_ARCHITECTURE_MAP.md` | `review_handoff/prompt_02/prompt_01_outputs/REPO_ARCHITECTURE_MAP.md` |
| `AVAILABILITY_DNP_MINUTES_AUDIT.md` | `review_handoff/prompt_02/prompt_01_outputs/AVAILABILITY_DNP_MINUTES_AUDIT.md` |
| `ARTIFACT_AND_DEPLOYMENT_AUDIT.md` | `review_handoff/prompt_02/prompt_01_outputs/ARTIFACT_AND_DEPLOYMENT_AUDIT.md` |
| `TEST_COVERAGE_GAP_AUDIT.md` | `review_handoff/prompt_02/prompt_01_outputs/TEST_COVERAGE_GAP_AUDIT.md` |
| `TRAIN_INFER_BACKTEST_PARITY_AUDIT.md` | `review_handoff/prompt_02/prompt_01_outputs/TRAIN_INFER_BACKTEST_PARITY_AUDIT.md` |

## Placeholder Files Created (Missing from Prompt 1)

| File | Reason |
|------|--------|
| `review_handoff/prompt_02/prompt_01_outputs/MISSING_CHANGED_FILES.md` | Prompt 1 did not produce CHANGED_FILES.md |
| `review_handoff/prompt_02/prompt_01_outputs/MISSING_TEST_RESULTS.md` | Prompt 1 did not produce TEST_RESULTS.md |
| `review_handoff/prompt_02/prompt_01_outputs/MISSING_RUN_COMMANDS.md` | Prompt 1 did not produce RUN_COMMANDS.md |
| `review_handoff/prompt_02/prompt_01_outputs/MISSING_REVIEWER_NOTES.md` | Prompt 1 did not produce REVIEWER_NOTES.md |

## New Files Created (Prompt 02 Spec Artifacts)

| File | Purpose |
|------|---------|
| `review_handoff/prompt_02/REPO_FILE_LIST.md` | Repo access confirmation |
| `review_handoff/prompt_02/CHANGED_FILES.md` | This file |
| `review_handoff/prompt_02/EVALUATION_SPEC.md` | Canonical evaluation specification |
| `review_handoff/prompt_02/DATA_SPLIT_POLICY.md` | Data split policy |
| `review_handoff/prompt_02/BET_RECORD_SCHEMA.md` | Per-bet record schema |
| `review_handoff/prompt_02/MODEL_ARTIFACT_SCHEMA.md` | Model artifact metadata schema |
| `review_handoff/prompt_02/REALISM_CHECKLIST.md` | 8 realism gates |
| `review_handoff/prompt_02/REPORTING_SCHEMA.md` | Report output schemas |
| `review_handoff/prompt_02/CURRENT_REPO_VS_SPEC_GAP_ANALYSIS.md` | Gap analysis |
| `review_handoff/prompt_02/MISSING_DATA.md` | BLOCKER: no closing lines |
| `review_handoff/prompt_02/TEST_PLAN.md` | Test inventory |
| `review_handoff/prompt_02/README.md` | Local run instructions |
| `review_handoff/prompt_02/FINAL_REPORT.md` | Verdict and next steps |
| `tests/eval_spec_tests/__init__.py` | Test package init |
| `tests/eval_spec_tests/test_gate_01_no_simulated_lines.py` | Gate 1 test |
| `tests/eval_spec_tests/test_gate_02_decision_time_line_present.py` | Gate 2 test |
| `tests/eval_spec_tests/test_gate_03_closing_line_present.py` | Gate 3 test |
| `tests/eval_spec_tests/test_gate_04_no_test_period_training.py` | Gate 4 test |
| `tests/eval_spec_tests/test_gate_05_real_odds_or_research.py` | Gate 5 test |
| `tests/eval_spec_tests/test_gate_06_settlement_void_dnp.py` | Gate 6 test |
| `tests/eval_spec_tests/test_gate_07_schema_completeness.py` | Gate 7 test |
| `tests/eval_spec_tests/test_gate_08_artifact_metadata.py` | Gate 8 test |
| `.github/workflows/eval-spec-check.yml` | CI workflow for gate tests |
