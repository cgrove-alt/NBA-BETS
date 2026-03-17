# Prompt 02 — Canonical Evaluation Framework

## Quick Start

```bash
# From repo root
cd /path/to/NBA-BETS

# Create virtual environment (if not exists)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# requirements.txt exists at repo root (verified: 649 bytes)

# Run evaluation spec gate tests
pytest -q tests/eval_spec_tests/

# Run with verbose output
pytest -v tests/eval_spec_tests/ --tb=short -ra
```

## Required Environment Variables

Found in code (`.env.example`):
- `BALLDONTLIE_API_KEY` — NOT required for gate tests (tests check static files/code only)
- `DATABASE_URL` — NOT required for gate tests
- `THE_ODDS_API_KEY` — NOT required for gate tests

**No environment variables are required to run the gate tests.**

## Expected Test Results

Most gate tests are **expected to FAIL** at this stage. They document spec violations that must be fixed before any evaluation can be labeled production-like. See TEST_PLAN.md for details.

## Deliverables

| File | Purpose |
|------|---------|
| `EVALUATION_SPEC.md` | Canonical evaluation specification (modes, rules, schemas) |
| `DATA_SPLIT_POLICY.md` | Train/val/test split rules with pseudocode |
| `BET_RECORD_SCHEMA.md` | Per-bet record schema with field definitions |
| `MODEL_ARTIFACT_SCHEMA.md` | Model artifact metadata schema |
| `REALISM_CHECKLIST.md` | 8 hard gates for realism labels |
| `REPORTING_SCHEMA.md` | Output report formats |
| `CURRENT_REPO_VS_SPEC_GAP_ANALYSIS.md` | Every spec requirement vs current repo |
| `MISSING_DATA.md` | BLOCKER: closing lines not available |
| `TEST_PLAN.md` | Test inventory and expected results |
| `FINAL_REPORT.md` | Executive summary and verdict |

## CI Workflow

`.github/workflows/eval-spec-check.yml` runs gate tests on PRs and workflow_dispatch.
Promotion job requires manual approval via GitHub environment protection rules.

Configure at: Settings > Environments > Create "promotion" > Add required reviewers.
