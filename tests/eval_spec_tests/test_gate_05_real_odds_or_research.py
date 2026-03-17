"""Gate 5: Real odds present OR run labeled research-only.

REALISM_CHECKLIST Gate 5:
  If decision_odds is a hardcoded constant (e.g., -110 for all bets),
  the run MUST be labeled RESEARCH-ONLY.
"""
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGate05RealOddsOrResearch:

    def test_profitability_backtest_uses_fixed_odds(self):
        """profitability_backtest.py uses STANDARD_ODDS = -110 for all bets.

        EXPECTED FAIL: Uses fixed odds without RESEARCH-ONLY label.
        """
        bt_path = os.path.join(REPO_ROOT, "nba_models", "backtesting", "profitability_backtest.py")
        if not os.path.exists(bt_path):
            pytest.skip("profitability_backtest.py not found")

        with open(bt_path) as f:
            source = f.read()

        has_fixed_odds = "STANDARD_ODDS = -110" in source
        has_research_label = "RESEARCH-ONLY" in source or "realism_level" in source.lower()

        if has_fixed_odds and not has_research_label:
            pytest.fail(
                "Gate 5 VIOLATION: profitability_backtest.py uses STANDARD_ODDS = -110 "
                "for all bets but does NOT carry a RESEARCH-ONLY label. "
                "See REALISM_CHECKLIST.md Gate 5."
            )

    def test_real_lines_backtest_uses_real_odds(self):
        """real_lines_backtest.py should use per-prop odds from data."""
        bt_path = os.path.join(REPO_ROOT, "nba_models", "backtesting", "real_lines_backtest.py")
        if not os.path.exists(bt_path):
            pytest.skip("real_lines_backtest.py not found")

        with open(bt_path) as f:
            source = f.read()

        assert "STANDARD_ODDS = -110" not in source, (
            "real_lines_backtest.py must not use hardcoded STANDARD_ODDS"
        )
