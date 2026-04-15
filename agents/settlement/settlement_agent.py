"""
Settlement Agent — nightly paper trade settlement.

Runs at 2 AM ET after games are complete. Fetches actual box-score stats
from the BallDontLie API and grades all unsettled paper trades for yesterday.
No LLM calls — pure data pipeline.
"""

import logging
from datetime import date, timedelta

from agents.core.agent_base import AgentBase

logger = logging.getLogger(__name__)


class SettlementAgent(AgentBase):
    """Nightly settlement: grade yesterday's paper trades against actual results."""

    AGENT_NAME = "settlement"
    DAILY_TOKEN_BUDGET = 0          # No LLM calls
    MAX_EXECUTION_SECONDS = 120     # Should complete well within 2 minutes

    def run(self) -> dict:
        """Settle paper trades for yesterday.

        Returns:
            Dict with date, trades_settled, and any errors.
        """
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        logger.info(f"[{self.AGENT_NAME}] Settling paper trades for {yesterday}")

        try:
            from nba_betting.settle_trades import settle_date
            trades_settled = settle_date(yesterday)
            logger.info(
                f"[{self.AGENT_NAME}] Settled {trades_settled} paper trades for {yesterday}"
            )
            return {
                "date": yesterday,
                "trades_settled": trades_settled,
                "status": "ok",
            }
        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] Settlement failed for {yesterday}: {e}")
            raise

    def report(self, run_output: dict):
        """No messages to send for settlement — just log."""
        logger.info(
            f"[{self.AGENT_NAME}] Settlement complete: "
            f"{run_output.get('trades_settled', 0)} trades for {run_output.get('date')}"
        )

    def cleanup(self):
        pass
