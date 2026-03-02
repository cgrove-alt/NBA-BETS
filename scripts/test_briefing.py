#!/usr/bin/env python3
"""
Briefing Pipeline End-to-End Test Script

Verifies the entire briefing pipeline: DB queries, agent execution, and API endpoint.
Run: PYTHONPATH=. python3 scripts/test_briefing.py

Each section prints PASS / FAIL / SKIP with details.
"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import load_env  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


# Track results globally: list of (name, "PASS" | "FAIL" | "SKIP")
results: list[tuple[str, str]] = []


def pass_msg(label: str, detail: str = ""):
    detail_str = f" — {detail}" if detail else ""
    print(f"  {Colors.GREEN}PASS{Colors.END}  {label}{detail_str}")


def fail_msg(label: str, detail: str = ""):
    detail_str = f" — {detail}" if detail else ""
    print(f"  {Colors.RED}FAIL{Colors.END}  {label}{detail_str}")


def skip_msg(label: str, detail: str = ""):
    detail_str = f" — {detail}" if detail else ""
    print(f"  {Colors.YELLOW}SKIP{Colors.END}  {label}{detail_str}")


# ---------------------------------------------------------------------------
# Section 1: Database Queries
# ---------------------------------------------------------------------------

def test_database_queries() -> str:
    """Check DB files and query_yesterday_record()."""
    print(f"\n{Colors.BOLD}1. Database Queries{Colors.END}")
    ok = True

    # calibration.db
    cal_path = Path(PROJECT_ROOT) / "data" / "calibration.db"
    if cal_path.exists():
        try:
            conn = sqlite3.connect(str(cal_path))
            pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            out_count = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            date_range = conn.execute(
                "SELECT MIN(game_date), MAX(game_date) FROM predictions"
            ).fetchone()
            conn.close()
            pass_msg("calibration.db", f"predictions={pred_count}, outcomes={out_count}, "
                     f"dates {date_range[0]} → {date_range[1]}")
        except Exception as e:
            fail_msg("calibration.db", f"query error: {e}")
            ok = False
    else:
        skip_msg("calibration.db", "file not found")

    # bet_tracking.db
    bt_path = Path(PROJECT_ROOT) / "data" / "bet_tracking.db"
    if bt_path.exists():
        try:
            conn = sqlite3.connect(str(bt_path))
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            bet_table = "tracked_bets" if "tracked_bets" in tables else "bets"
            row_count = conn.execute(f"SELECT COUNT(*) FROM {bet_table}").fetchone()[0]
            date_range = conn.execute(
                f"SELECT MIN(event_date), MAX(event_date) FROM {bet_table}"
            ).fetchone()
            conn.close()
            pass_msg("bet_tracking.db", f"{bet_table}={row_count}, "
                     f"dates {date_range[0]} → {date_range[1]}")
        except Exception as e:
            fail_msg("bet_tracking.db", f"query error: {e}")
            ok = False
    else:
        skip_msg("bet_tracking.db", "file not found")

    # query_yesterday_record
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        from agents.core.db_queries import query_yesterday_record
        record = query_yesterday_record(yesterday_str)
        if record is not None:
            overall = record.get("overall", {})
            pass_msg("query_yesterday_record()",
                     f"source={record.get('source')}, "
                     f"W{overall.get('wins',0)}-L{overall.get('losses',0)}-P{overall.get('pushes',0)}")
        else:
            pass_msg("query_yesterday_record()", f"returned None for {yesterday_str} (no data)")
    except Exception as e:
        fail_msg("query_yesterday_record()", str(e))
        ok = False

    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# Section 2: Briefing Agent
# ---------------------------------------------------------------------------

def test_briefing_agent() -> str:
    """Instantiate agent, patch LLM, call run(), validate return dict."""
    print(f"\n{Colors.BOLD}2. Briefing Agent (direct){Colors.END}")

    try:
        from agents.briefing.briefing_agent import DailyBriefingAgent
    except Exception as e:
        fail_msg("Import DailyBriefingAgent", str(e))
        return "FAIL"

    try:
        agent = DailyBriefingAgent()
        pass_msg("Instantiate DailyBriefingAgent")
    except Exception as e:
        fail_msg("Instantiate DailyBriefingAgent", str(e))
        return "FAIL"

    # Patch call_llm to return None → deterministic fallback path
    try:
        with patch.object(agent, "call_llm", return_value=None):
            result = agent.run()
    except Exception as e:
        fail_msg("agent.run()", str(e))
        return "FAIL"

    # Validate return dict keys
    expected_keys = {
        "briefing_date", "generated_at", "sections",
        "formatted_text", "yesterday_record", "data_sources", "reasoning",
    }
    actual_keys = set(result.keys())
    missing = expected_keys - actual_keys
    if missing:
        fail_msg("Return dict keys", f"missing: {missing}")
        return "FAIL"
    pass_msg("Return dict keys", f"all {len(expected_keys)} present")

    # Print formatted_text excerpt
    text = result.get("formatted_text", "")
    if text:
        excerpt = text[:200].replace("\n", " ")
        print(f"         formatted_text: {excerpt}...")
    else:
        print("         formatted_text: (empty)")

    # Print data_sources
    sources = result.get("data_sources", [])
    print(f"         data_sources: {sources}")

    # Print yesterday_record summary if present
    yr = result.get("yesterday_record")
    if yr:
        overall = yr.get("overall", {})
        print(f"         yesterday_record: W{overall.get('wins',0)}-L{overall.get('losses',0)}"
              f"-P{overall.get('pushes',0)} ({overall.get('hit_rate',0)}% hit rate)")
        by_type = yr.get("by_bet_type", {})
        if by_type:
            print(f"         by_bet_type: {list(by_type.keys())}")
    else:
        print("         yesterday_record: None")

    pass_msg("agent.run() completed")
    return "PASS"


# ---------------------------------------------------------------------------
# Section 3: API Endpoint
# ---------------------------------------------------------------------------

def test_api_endpoint() -> str:
    """Hit /api/briefing via requests."""
    print(f"\n{Colors.BOLD}3. API Endpoint (/api/briefing){Colors.END}")

    try:
        import requests
    except ImportError:
        skip_msg("requests not installed")
        return "SKIP"

    try:
        resp = requests.get("http://localhost:8000/api/briefing", timeout=10)
    except Exception:
        skip_msg("Server not running at localhost:8000")
        return "SKIP"

    if resp.status_code != 200:
        fail_msg("HTTP status", f"expected 200, got {resp.status_code}")
        return "FAIL"
    pass_msg("HTTP 200")

    try:
        data = resp.json()
    except Exception as e:
        fail_msg("JSON parse", str(e))
        return "FAIL"

    # Validate top-level fields (BriefingResponse schema)
    expected_fields = {"date", "briefing_text", "generated_at", "yesterday_record", "today_preview"}
    actual_fields = set(data.keys())
    missing = expected_fields - actual_fields
    if missing:
        fail_msg("Response fields", f"missing: {missing}")
        return "FAIL"
    pass_msg("Response fields", f"all expected fields present")

    print(f"         date: {data.get('date')}")
    print(f"         generated_at: {data.get('generated_at')}")
    briefing_excerpt = (data.get("briefing_text") or "")[:120].replace("\n", " ")
    print(f"         briefing_text: {briefing_excerpt}...")

    # Validate yesterday_record structure if present
    yr = data.get("yesterday_record")
    if yr:
        yr_keys = {"overall", "by_bet_type", "by_confidence"}
        yr_actual = set(yr.keys())
        yr_missing = yr_keys - yr_actual
        if yr_missing:
            fail_msg("yesterday_record keys", f"missing: {yr_missing}")
            return "FAIL"
        overall = yr.get("overall", {})
        pass_msg("yesterday_record",
                 f"W{overall.get('wins',0)}-L{overall.get('losses',0)}"
                 f"-P{overall.get('pushes',0)}")
    else:
        print("         yesterday_record: null")

    # Print today_preview summary
    tp = data.get("today_preview")
    if tp:
        print(f"         today_preview: {list(tp.keys()) if isinstance(tp, dict) else type(tp).__name__}")
    else:
        print("         today_preview: null")

    pass_msg("/api/briefing validated")
    return "PASS"


# ---------------------------------------------------------------------------
# Section 4: Summary
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  BRIEFING PIPELINE — END-TO-END TEST")
    print("=" * 60)

    tests = [
        ("Database Queries", test_database_queries),
        ("Briefing Agent", test_briefing_agent),
        ("API Endpoint", test_api_endpoint),
    ]

    for name, fn in tests:
        try:
            status = fn()
        except Exception as e:
            print(f"  {Colors.RED}ERROR{Colors.END}  {name} crashed: {e}")
            status = "FAIL"
        results.append((name, status))

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"  {Colors.BOLD}SUMMARY{Colors.END}")
    print(f"  {'Test':<25} {'Result'}")
    print(f"  {'-' * 25} {'-' * 10}")
    for name, status in results:
        if status == "PASS":
            color = Colors.GREEN
        elif status == "SKIP":
            color = Colors.YELLOW
        else:
            color = Colors.RED
        print(f"  {name:<25} {color}{status}{Colors.END}")

    passed = sum(1 for _, s in results if s == "PASS")
    skipped = sum(1 for _, s in results if s == "SKIP")
    failed = sum(1 for _, s in results if s == "FAIL")
    print(f"\n  {passed} passed, {skipped} skipped, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
