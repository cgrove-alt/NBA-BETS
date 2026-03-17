"""Gate 1: No simulated lines in production-like evaluation.

REALISM_CHECKLIST Gate 1:
  decision_line must come from an external sportsbook source, NOT from model
  features or player averages.

This test verifies that the profitability backtest's simulate_prop_line()
function exists (documenting the violation) and that real_lines_backtest.py
loads from historical data instead.
"""
import ast
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGate01NoSimulatedLines:

    def test_profitability_backtest_uses_simulated_lines(self):
        """EXPECTED FAIL: profitability_backtest.py uses simulate_prop_line().

        This test documents the Gate 1 violation. It will PASS when
        simulate_prop_line() is removed or the backtest is relabeled RESEARCH-ONLY.
        """
        bt_path = os.path.join(REPO_ROOT, "nba_models", "backtesting", "profitability_backtest.py")
        if not os.path.exists(bt_path):
            pytest.skip("profitability_backtest.py not found")

        with open(bt_path) as f:
            source = f.read()

        # Gate 1 requires: no simulated lines in production-like evaluation
        has_simulate = "simulate_prop_line" in source
        has_research_label = "RESEARCH-ONLY" in source or "REALISM_LEVEL" in source

        if has_simulate and not has_research_label:
            pytest.fail(
                "Gate 1 VIOLATION: profitability_backtest.py contains simulate_prop_line() "
                "but does NOT carry a RESEARCH-ONLY realism label. "
                "See REALISM_CHECKLIST.md Gate 1."
            )

    def test_real_lines_backtest_uses_historical_data(self):
        """Verify real_lines_backtest.py loads from data/historical_lines/."""
        bt_path = os.path.join(REPO_ROOT, "nba_models", "backtesting", "real_lines_backtest.py")
        if not os.path.exists(bt_path):
            pytest.skip("real_lines_backtest.py not found")

        with open(bt_path) as f:
            source = f.read()

        assert "historical_lines" in source, (
            "real_lines_backtest.py does not reference historical_lines directory"
        )
        assert "simulate_prop_line" not in source, (
            "real_lines_backtest.py must NOT use simulate_prop_line()"
        )
