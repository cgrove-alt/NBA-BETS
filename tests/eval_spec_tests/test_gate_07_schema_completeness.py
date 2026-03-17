"""Gate 7: Bet record schema completeness.

REALISM_CHECKLIST Gate 7:
  Every bet record must have all REQUIRED fields non-null as defined
  in BET_RECORD_SCHEMA.md.
"""
import os
import csv
import glob
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Required fields from BET_RECORD_SCHEMA.md
REQUIRED_FIELDS = {
    "event_id", "game_id", "market_type", "side",
    "decision_timestamp", "snapshot_timestamp",
    "decision_line", "decision_odds", "book",
    "model_fair_probability", "market_implied_probability",
    "vig_adjusted_probability", "raw_edge", "vig_adjusted_edge",
    "accepted", "result", "PnL", "stake",
    "artifact_version", "realism_level",
}


class TestGate07SchemaCompleteness:

    def test_prediction_csv_has_required_fields(self):
        """Current prediction CSV output must contain all required bet record fields.

        EXPECTED FAIL: Current CSV has 37 columns but is missing many spec fields.
        """
        pred_dir = os.path.join(REPO_ROOT, "data", "predictions")
        if not os.path.isdir(pred_dir):
            pytest.skip("data/predictions/ not found")

        csv_files = sorted(glob.glob(os.path.join(pred_dir, "predictions_*.csv")))
        if not csv_files:
            pytest.skip("No prediction CSV files found")

        # Check the most recent file
        latest = csv_files[-1]
        with open(latest) as f:
            reader = csv.reader(f)
            headers = set(next(reader))

        missing = REQUIRED_FIELDS - headers
        if missing:
            pytest.fail(
                f"Gate 7 VIOLATION: Prediction CSV ({os.path.basename(latest)}) "
                f"missing {len(missing)} required fields: {sorted(missing)}. "
                f"Present columns: {sorted(headers)}. "
                "See BET_RECORD_SCHEMA.md for full specification."
            )

    def test_canonical_bet_log_format_exists(self):
        """A canonical JSONL bet log format should exist.

        EXPECTED FAIL: No JSONL bet log exists yet.
        """
        reports_dir = os.path.join(REPO_ROOT, "review_handoff", "prompt_02", "reports")
        jsonl_path = os.path.join(reports_dir, "per_bet_log.jsonl")

        if not os.path.exists(jsonl_path):
            pytest.fail(
                "Gate 7 INFO: No canonical per_bet_log.jsonl exists yet. "
                "This is expected — it will be created when the evaluation "
                "framework is implemented. See REPORTING_SCHEMA.md."
            )
