"""Gate 3: Closing line and odds present for CLV.

REALISM_CHECKLIST Gate 3:
  For CLV computation, closing_line and closing_odds must be captured
  after game start. At least 90% of accepted bets must have closing data.

EXPECTED: This test FAILS because closing lines are not captured anywhere
in the repo. See MISSING_DATA.md for details.
"""
import json
import os
import glob
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LINES_DIR = os.path.join(REPO_ROOT, "data", "historical_lines")


class TestGate03ClosingLine:

    def test_closing_line_field_exists_in_historical_data(self):
        """Historical line files must contain closing_line or closing_odds keys.

        EXPECTED FAIL: Current data has only single-snapshot lines.
        """
        if not os.path.isdir(LINES_DIR):
            pytest.fail(
                "Gate 3 BLOCKER: data/historical_lines/ missing. "
                "See MISSING_DATA.md"
            )

        json_files = sorted(glob.glob(os.path.join(LINES_DIR, "20*.json")))
        if not json_files:
            pytest.fail("Gate 3 BLOCKER: No historical line files found")

        # Check 3 representative files
        files_checked = json_files[:3]
        closing_found = False

        for jf in files_checked:
            with open(jf) as f:
                data = json.load(f)

            # Check game-level keys
            for g in data.get("games", []):
                if "closing_line" in g or "closing_odds" in g:
                    closing_found = True
                    break

                # Check prop-level keys
                for p in g.get("player_props", []):
                    if "closing_line" in p or "closing_odds" in p:
                        closing_found = True
                        break

            if closing_found:
                break

        if not closing_found:
            pytest.fail(
                "Gate 3 BLOCKER: No closing_line or closing_odds field found in "
                f"historical line files ({', '.join(os.path.basename(f) for f in files_checked)}). "
                "CLV computation is impossible. "
                "This blocks ALL production-like evaluation. "
                "See review_handoff/prompt_02/MISSING_DATA.md"
            )

    def test_no_clv_computation_exists(self):
        """Verify no CLV computation function exists in evaluation code.

        EXPECTED FAIL: Documents the gap.
        """
        eval_dir = os.path.join(REPO_ROOT, "nba_models", "evaluation")
        if os.path.isdir(eval_dir):
            for root, dirs, files in os.walk(eval_dir):
                for fn in files:
                    if fn.endswith(".py"):
                        with open(os.path.join(root, fn)) as f:
                            if "compute_clv" in f.read():
                                return  # Found — gate partially satisfied

        # Check existing modules
        for search_dir in ["nba_betting", "nba_models/backtesting"]:
            dirpath = os.path.join(REPO_ROOT, search_dir)
            if os.path.isdir(dirpath):
                for fn in os.listdir(dirpath):
                    if fn.endswith(".py"):
                        with open(os.path.join(dirpath, fn)) as f:
                            content = f.read()
                            if "def compute_clv" in content:
                                return  # Found

        pytest.fail(
            "Gate 3 VIOLATION: No compute_clv() function found in codebase. "
            "CLV computation must be implemented. "
            "See EVALUATION_SPEC.md C8 for function signature."
        )
