"""Gate 6: Settlement supports VOID for DNP.

REALISM_CHECKLIST Gate 6:
  If a player has 0 minutes played, the bet result MUST be 'void' with PnL = 0.
"""
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGate06SettlementVoidDnp:

    def test_settle_trades_has_void_logic(self):
        """settle_trades.py must check for DNP and assign result='void'.

        EXPECTED FAIL: Current implementation has no void logic.
        """
        st_path = os.path.join(REPO_ROOT, "nba_betting", "settle_trades.py")
        if not os.path.exists(st_path):
            pytest.skip("settle_trades.py not found")

        with open(st_path) as f:
            source = f.read()

        has_void = "void" in source.lower() and ("minutes" in source.lower() or "dnp" in source.lower())

        if not has_void:
            pytest.fail(
                "Gate 6 VIOLATION: nba_betting/settle_trades.py does not contain "
                "void/DNP handling. Players with 0 minutes must have result='void'. "
                "See REALISM_CHECKLIST.md Gate 6."
            )

    def test_paper_trading_has_void_logic(self):
        """paper_trading.py settle_trades() must support void results.

        EXPECTED FAIL: No void logic in paper trading.
        """
        pt_path = os.path.join(REPO_ROOT, "nba_betting", "paper_trading.py")
        if not os.path.exists(pt_path):
            pytest.skip("paper_trading.py not found")

        with open(pt_path) as f:
            source = f.read()

        has_void = "void" in source.lower() and ("minutes" in source.lower() or "dnp" in source.lower())

        if not has_void:
            pytest.fail(
                "Gate 6 VIOLATION: nba_betting/paper_trading.py does not contain "
                "void/DNP handling. "
                "See REALISM_CHECKLIST.md Gate 6."
            )
