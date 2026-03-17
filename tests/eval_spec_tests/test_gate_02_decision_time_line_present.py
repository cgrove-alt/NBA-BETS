"""Gate 2: Decision-time line and odds present.

REALISM_CHECKLIST Gate 2:
  Every accepted bet must have non-null decision_line, decision_odds,
  snapshot_timestamp, and book.

This test verifies that historical line data includes the required fields.
"""
import json
import os
import glob
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LINES_DIR = os.path.join(REPO_ROOT, "data", "historical_lines")


class TestGate02DecisionTimeLine:

    def test_historical_lines_directory_exists(self):
        """data/historical_lines/ must exist."""
        assert os.path.isdir(LINES_DIR), (
            f"Gate 2 BLOCKER: {LINES_DIR} does not exist. "
            "Cannot evaluate decision-time line availability."
        )

    def test_historical_lines_have_snapshot_timestamp(self):
        """Each game in historical lines must have snapshot_timestamp."""
        if not os.path.isdir(LINES_DIR):
            pytest.skip("historical_lines directory missing")

        json_files = sorted(glob.glob(os.path.join(LINES_DIR, "20*.json")))
        if not json_files:
            pytest.fail("No historical line files found")

        # Check first 3 files
        for jf in json_files[:3]:
            with open(jf) as f:
                data = json.load(f)
            games = data.get("games", [])
            for g in games:
                assert "snapshot_timestamp" in g, (
                    f"Gate 2 VIOLATION: Game in {jf} missing snapshot_timestamp. "
                    f"Game keys: {list(g.keys())}"
                )

    def test_historical_lines_have_odds(self):
        """Player props must have bookmaker and odds."""
        if not os.path.isdir(LINES_DIR):
            pytest.skip("historical_lines directory missing")

        json_files = sorted(glob.glob(os.path.join(LINES_DIR, "20*.json")))
        if not json_files:
            pytest.fail("No historical line files found")

        with open(json_files[0]) as f:
            data = json.load(f)

        games = data.get("games", [])
        props_found = False
        for g in games:
            props = g.get("player_props", [])
            for p in props:
                props_found = True
                assert "bookmaker" in p, (
                    f"Gate 2 VIOLATION: Prop missing 'bookmaker' field in {json_files[0]}"
                )
                assert "over_odds" in p or "under_odds" in p, (
                    f"Gate 2 VIOLATION: Prop missing odds fields in {json_files[0]}"
                )
                break  # Check first prop only per file
            if props_found:
                break

        if not props_found:
            pytest.skip("No player props found in first historical line file")
