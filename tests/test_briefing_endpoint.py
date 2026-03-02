"""
Tests for the /api/briefing endpoint and its helper functions.

Covers:
  - _build_record_from_calibration
  - _build_record_from_tracking
  - _format_yesterday_text
  - _query_yesterday_record
  - GET /api/briefing (integration)
"""

import json
import os
import sqlite3
import sys

import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.core.db_queries import (
    _build_record_from_calibration,
    _build_record_from_tracking,
    query_yesterday_record as _query_yesterday_record,
)
from backend.api import _format_yesterday_text


# =============================================================================
# Helpers — fake sqlite3.Row-like dicts
# =============================================================================

class FakeRow(dict):
    """sqlite3.Row stand-in that supports both dict[key] and row["key"]."""
    def __getitem__(self, key):
        return super().__getitem__(key)


def _cal_row(prop_type="Points", confidence=62, hit=True, clv=0.5):
    return FakeRow(prop_type=prop_type, confidence=confidence, hit=hit, clv=clv)


def _track_row(status="won", pnl=10.0, tags=None, bet_type="Points"):
    return FakeRow(status=status, pnl=pnl, tags=tags, bet_type=bet_type)


# =============================================================================
# 1. _build_record_from_calibration
# =============================================================================

class TestBuildRecordFromCalibration:

    def test_basic_win_loss_push_tally(self):
        rows = [
            _cal_row(hit=True),
            _cal_row(hit=True),
            _cal_row(hit=False),
            _cal_row(hit=None),  # push
        ]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["overall"]["wins"] == 2
        assert rec["overall"]["losses"] == 1
        assert rec["overall"]["pushes"] == 1
        assert rec["overall"]["total"] == 4

    def test_hit_rate_excludes_pushes(self):
        rows = [
            _cal_row(hit=True),
            _cal_row(hit=False),
            _cal_row(hit=None),
        ]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        # 1W / (1W + 1L) = 50.0%
        assert rec["overall"]["hit_rate"] == 50.0

    def test_by_bet_type_bucketing(self):
        rows = [
            _cal_row(prop_type="Points", hit=True),
            _cal_row(prop_type="Points", hit=False),
            _cal_row(prop_type="Rebounds", hit=True),
        ]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["by_bet_type"]["Points"]["wins"] == 1
        assert rec["by_bet_type"]["Points"]["losses"] == 1
        assert rec["by_bet_type"]["Points"]["hit_rate"] == 50.0
        assert rec["by_bet_type"]["Rebounds"]["wins"] == 1
        assert rec["by_bet_type"]["Rebounds"]["losses"] == 0
        assert rec["by_bet_type"]["Rebounds"]["hit_rate"] == 100.0

    def test_by_confidence_tier_high(self):
        rows = [_cal_row(confidence=65, hit=True)]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["by_confidence"]["high"]["total"] == 1
        assert rec["by_confidence"]["high"]["wins"] == 1

    def test_by_confidence_tier_medium(self):
        rows = [_cal_row(confidence=57, hit=False)]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["by_confidence"]["medium"]["total"] == 1
        assert rec["by_confidence"]["medium"]["losses"] == 1

    def test_by_confidence_tier_low(self):
        rows = [_cal_row(confidence=50, hit=True)]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["by_confidence"]["low"]["total"] == 1
        assert rec["by_confidence"]["low"]["wins"] == 1

    def test_boundary_confidence_60_is_high(self):
        rows = [_cal_row(confidence=60, hit=True)]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["by_confidence"]["high"]["total"] == 1

    def test_boundary_confidence_55_is_medium(self):
        rows = [_cal_row(confidence=55, hit=True)]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["by_confidence"]["medium"]["total"] == 1

    def test_boundary_confidence_54_is_low(self):
        rows = [_cal_row(confidence=54, hit=True)]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["by_confidence"]["low"]["total"] == 1

    def test_clv_summary_computed(self):
        rows = [
            _cal_row(hit=True, clv=1.0),
            _cal_row(hit=True, clv=-0.5),
            _cal_row(hit=False, clv=0.2),
        ]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        cs = rec["clv_summary"]
        assert cs is not None
        # avg = (1.0 + -0.5 + 0.2) / 3 = 0.233...
        assert cs["avg_clv"] == pytest.approx(0.23, abs=0.01)
        # 2 of 3 have positive CLV
        assert cs["positive_clv_rate"] == pytest.approx(66.7, abs=0.1)

    def test_clv_none_rows_excluded(self):
        rows = [
            _cal_row(hit=True, clv=None),
            _cal_row(hit=True, clv=2.0),
        ]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        cs = rec["clv_summary"]
        assert cs is not None
        assert cs["avg_clv"] == 2.0
        assert cs["positive_clv_rate"] == 100.0

    def test_all_clv_none_returns_no_summary(self):
        rows = [_cal_row(hit=True, clv=None)]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["clv_summary"] is None

    def test_pushes_dont_affect_by_type_or_by_confidence(self):
        """Push rows (hit=None) should not be counted in by_bet_type or by_confidence."""
        rows = [_cal_row(hit=None, prop_type="Points", confidence=65)]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["by_bet_type"] == {}
        assert rec["by_confidence"]["high"]["total"] == 0

    def test_source_is_calibration(self):
        rows = [_cal_row(hit=True)]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["source"] == "calibration"

    def test_date_field_set(self):
        rows = [_cal_row(hit=True)]
        rec = _build_record_from_calibration(rows, "2026-03-01")
        assert rec["date"] == "2026-03-01"


# =============================================================================
# 2. _build_record_from_tracking
# =============================================================================

class TestBuildRecordFromTracking:

    def test_basic_win_loss_push(self):
        rows = [
            _track_row(status="won", pnl=10),
            _track_row(status="lost", pnl=-10),
            _track_row(status="push", pnl=0),
        ]
        rec = _build_record_from_tracking(rows, "2026-03-01")
        assert rec["overall"]["wins"] == 1
        assert rec["overall"]["losses"] == 1
        assert rec["overall"]["pushes"] == 1

    def test_pnl_summed(self):
        rows = [
            _track_row(status="won", pnl=25.50),
            _track_row(status="lost", pnl=-10.0),
        ]
        rec = _build_record_from_tracking(rows, "2026-03-01")
        assert rec["overall"]["profit"] == 15.50

    def test_prop_type_from_tags_dict(self):
        rows = [_track_row(tags=json.dumps({"prop_type": "Rebounds"}), bet_type="Points")]
        rec = _build_record_from_tracking(rows, "2026-03-01")
        assert "Rebounds" in rec["by_bet_type"]
        assert "Points" not in rec["by_bet_type"]

    def test_prop_type_from_tags_list(self):
        rows = [_track_row(tags=json.dumps(["Assists"]), bet_type="Points")]
        rec = _build_record_from_tracking(rows, "2026-03-01")
        assert "Assists" in rec["by_bet_type"]

    def test_fallback_to_bet_type_when_no_tags(self):
        rows = [_track_row(tags=None, bet_type="3PM")]
        rec = _build_record_from_tracking(rows, "2026-03-01")
        assert "3PM" in rec["by_bet_type"]

    def test_fallback_to_bet_type_when_tags_invalid_json(self):
        rows = [_track_row(tags="not-json{{{", bet_type="PRA")]
        rec = _build_record_from_tracking(rows, "2026-03-01")
        assert "PRA" in rec["by_bet_type"]

    def test_hit_rate_calculation(self):
        rows = [
            _track_row(status="won"),
            _track_row(status="won"),
            _track_row(status="lost"),
        ]
        rec = _build_record_from_tracking(rows, "2026-03-01")
        assert rec["overall"]["hit_rate"] == pytest.approx(66.7, abs=0.1)

    def test_source_is_bet_tracking(self):
        rows = [_track_row(status="won")]
        rec = _build_record_from_tracking(rows, "2026-03-01")
        assert rec["source"] == "bet_tracking"

    def test_by_confidence_empty(self):
        """bet_tracking source doesn't track confidence tiers."""
        rows = [_track_row(status="won")]
        rec = _build_record_from_tracking(rows, "2026-03-01")
        assert rec["by_confidence"] == {}

    def test_clv_summary_none(self):
        """bet_tracking source doesn't have CLV data."""
        rows = [_track_row(status="won")]
        rec = _build_record_from_tracking(rows, "2026-03-01")
        assert rec["clv_summary"] is None

    def test_pnl_none_treated_as_zero(self):
        rows = [_track_row(status="won", pnl=None)]
        rec = _build_record_from_tracking(rows, "2026-03-01")
        assert rec["overall"]["profit"] == 0.0


# =============================================================================
# 3. _format_yesterday_text
# =============================================================================

class TestFormatYesterdayText:

    def test_none_input(self):
        result = _format_yesterday_text(None)
        assert "No games yesterday" in result
        assert "YESTERDAY'S RECORD" in result

    def test_empty_dict_treated_as_falsy(self):
        result = _format_yesterday_text({})
        assert "No games yesterday" in result

    def test_full_record_formatted(self):
        record = {
            "date": "2026-03-01",
            "overall": {"wins": 5, "losses": 3, "pushes": 1, "hit_rate": 62.5, "profit": 120.0, "roi": 4.2},
            "by_bet_type": {
                "Points": {"wins": 3, "losses": 1, "total": 4, "hit_rate": 75.0},
                "Rebounds": {"wins": 2, "losses": 2, "total": 4, "hit_rate": 50.0},
            },
            "by_confidence": {
                "high": {"wins": 3, "losses": 1, "total": 4, "hit_rate": 75.0},
                "medium": {"wins": 1, "losses": 1, "total": 2, "hit_rate": 50.0},
                "low": {"wins": 0, "losses": 0, "total": 0, "hit_rate": 0.0},
            },
            "clv_summary": {"avg_clv": 0.8, "positive_clv_rate": 65.0},
        }
        result = _format_yesterday_text(record)
        assert "YESTERDAY'S RECORD (2026-03-01)" in result
        assert "5-3" in result
        assert "62.5%" in result
        assert "$+120" in result
        assert "By Bet Type:" in result
        assert "Points:" in result
        assert "Rebounds:" in result
        assert "By Confidence:" in result
        assert "High" in result
        assert "CLV:" in result
        assert "+0.8" in result
        assert "65%" in result

    def test_empty_by_confidence_omitted(self):
        record = {
            "date": "2026-03-01",
            "overall": {"wins": 2, "losses": 1, "pushes": 0, "hit_rate": 66.7, "profit": 0, "roi": 0},
            "by_bet_type": {},
            "by_confidence": {
                "high": {"wins": 0, "losses": 0, "total": 0, "hit_rate": 0.0},
                "medium": {"wins": 0, "losses": 0, "total": 0, "hit_rate": 0.0},
                "low": {"wins": 0, "losses": 0, "total": 0, "hit_rate": 0.0},
            },
            "clv_summary": None,
        }
        result = _format_yesterday_text(record)
        assert "By Confidence:" not in result

    def test_no_clv_omits_clv_line(self):
        record = {
            "date": "2026-03-01",
            "overall": {"wins": 1, "losses": 0, "pushes": 0, "hit_rate": 100.0, "profit": 0, "roi": 0},
            "by_bet_type": {},
            "by_confidence": {},
            "clv_summary": None,
        }
        result = _format_yesterday_text(record)
        assert "CLV:" not in result

    def test_zero_profit_omits_dollar_amount(self):
        record = {
            "date": "2026-03-01",
            "overall": {"wins": 1, "losses": 1, "pushes": 0, "hit_rate": 50.0, "profit": 0, "roi": 0},
            "by_bet_type": {},
            "by_confidence": {},
            "clv_summary": None,
        }
        result = _format_yesterday_text(record)
        assert "$" not in result


# =============================================================================
# 4. _query_yesterday_record
# =============================================================================

class TestQueryYesterdayRecord:

    def test_no_db_files_returns_none(self, tmp_path):
        """When neither calibration.db nor bet_tracking.db exist, returns None."""
        with patch("backend.api.Path") as mock_path:
            # Make both paths not exist
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance
            result = _query_yesterday_record("2026-03-01")
        assert result is None

    def test_calibration_db_with_data(self, tmp_path):
        """calibration.db with matching rows returns calibration-sourced record."""
        cal_db = tmp_path / "calibration.db"
        conn = sqlite3.connect(str(cal_db))
        conn.execute("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY,
                prop_type TEXT,
                confidence REAL,
                game_date TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE outcomes (
                prediction_id INTEGER,
                hit INTEGER,
                clv REAL
            )
        """)
        conn.execute("INSERT INTO predictions VALUES (1, 'Points', 65, '2026-03-01')")
        conn.execute("INSERT INTO outcomes VALUES (1, 1, 0.5)")
        conn.execute("INSERT INTO predictions VALUES (2, 'Rebounds', 55, '2026-03-01')")
        conn.execute("INSERT INTO outcomes VALUES (2, 0, -0.2)")
        conn.commit()
        conn.close()

        with patch("backend.api.Path") as mock_path:
            def path_side_effect(p):
                if "calibration.db" in str(p):
                    m = MagicMock()
                    m.exists.return_value = True
                    m.__str__ = lambda self: str(cal_db)
                    return m
                m = MagicMock()
                m.exists.return_value = False
                return m
            mock_path.side_effect = path_side_effect

            with patch("sqlite3.connect", return_value=sqlite3.connect(str(cal_db))) as mock_connect:
                mock_conn = sqlite3.connect(str(cal_db))
                mock_conn.row_factory = sqlite3.Row
                with patch("sqlite3.connect", return_value=mock_conn):
                    result = _query_yesterday_record("2026-03-01")

        assert result is not None
        assert result["source"] == "calibration"
        assert result["overall"]["wins"] == 1
        assert result["overall"]["losses"] == 1

    def test_calibration_empty_falls_back_to_tracking(self, tmp_path):
        """When calibration.db has no matching rows, falls back to bet_tracking.db."""
        # Create empty calibration DB
        cal_db = tmp_path / "calibration.db"
        conn = sqlite3.connect(str(cal_db))
        conn.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY, prop_type TEXT, confidence REAL, game_date TEXT)")
        conn.execute("CREATE TABLE outcomes (prediction_id INTEGER, hit INTEGER, clv REAL)")
        conn.commit()
        conn.close()

        # Create bet_tracking DB with data
        bt_db = tmp_path / "bet_tracking.db"
        conn = sqlite3.connect(str(bt_db))
        conn.execute("""
            CREATE TABLE tracked_bets (
                status TEXT, pnl REAL, tags TEXT, bet_type TEXT, event_date TEXT
            )
        """)
        conn.execute("INSERT INTO tracked_bets VALUES ('won', 15.0, NULL, 'Points', '2026-03-01')")
        conn.commit()
        conn.close()

        # Pre-create connections BEFORE patching sqlite3.connect to avoid recursion
        cal_conn = sqlite3.connect(str(cal_db))
        cal_conn.row_factory = sqlite3.Row
        bt_conn = sqlite3.connect(str(bt_db))
        bt_conn.row_factory = sqlite3.Row

        conns = iter([cal_conn, bt_conn])

        with patch("backend.api.Path") as mock_path:
            def path_side_effect(p):
                m = MagicMock()
                m.exists.return_value = True
                if "calibration.db" in str(p):
                    m.__str__ = lambda self: str(cal_db)
                else:
                    m.__str__ = lambda self: str(bt_db)
                return m
            mock_path.side_effect = path_side_effect

            with patch("sqlite3.connect", side_effect=lambda _: next(conns)):
                result = _query_yesterday_record("2026-03-01")

        assert result is not None
        assert result["source"] == "bet_tracking"
        assert result["overall"]["wins"] == 1

    def test_both_empty_returns_none(self, tmp_path):
        """When both DBs exist but have no matching rows, returns None."""
        cal_db = tmp_path / "calibration.db"
        conn = sqlite3.connect(str(cal_db))
        conn.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY, prop_type TEXT, confidence REAL, game_date TEXT)")
        conn.execute("CREATE TABLE outcomes (prediction_id INTEGER, hit INTEGER, clv REAL)")
        conn.commit()
        conn.close()

        bt_db = tmp_path / "bet_tracking.db"
        conn = sqlite3.connect(str(bt_db))
        conn.execute("CREATE TABLE tracked_bets (status TEXT, pnl REAL, tags TEXT, bet_type TEXT, event_date TEXT)")
        conn.commit()
        conn.close()

        # Pre-create connections BEFORE patching sqlite3.connect to avoid recursion
        cal_conn = sqlite3.connect(str(cal_db))
        cal_conn.row_factory = sqlite3.Row
        bt_conn = sqlite3.connect(str(bt_db))
        bt_conn.row_factory = sqlite3.Row

        conns = iter([cal_conn, bt_conn])

        with patch("backend.api.Path") as mock_path:
            def path_side_effect(p):
                m = MagicMock()
                m.exists.return_value = True
                if "calibration.db" in str(p):
                    m.__str__ = lambda self: str(cal_db)
                else:
                    m.__str__ = lambda self: str(bt_db)
                return m
            mock_path.side_effect = path_side_effect

            with patch("sqlite3.connect", side_effect=lambda _: next(conns)):
                result = _query_yesterday_record("2026-03-01")

        assert result is None


# =============================================================================
# 5. Integration test for GET /api/briefing
# =============================================================================

class TestBriefingEndpointIntegration:

    @pytest.fixture
    def client(self):
        """Create a test client with mocked dependencies."""
        from fastapi.testclient import TestClient

        # Mock the lifespan to avoid startup initialization
        with patch("backend.api.lifespan") as mock_lifespan:
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def noop_lifespan(app):
                yield

            mock_lifespan.side_effect = noop_lifespan

            # Re-import to pick up the mock — but the app is already created.
            # Instead, patch the functions called during the request.
            from backend.api import app
            with TestClient(app) as client:
                yield client

    def test_returns_200_with_valid_shape(self, client):
        """GET /api/briefing returns 200 with correct fields."""
        with patch("backend.api._query_yesterday_record", return_value=None), \
             patch("backend.api._get_today_preview", return_value=None), \
             patch("backend.api.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            resp = client.get("/api/briefing?date=2026-03-01")

        assert resp.status_code == 200
        data = resp.json()
        assert "date" in data
        assert "briefing_text" in data
        assert "yesterday_record" in data
        assert "today_preview" in data
        assert data["date"] == "2026-03-01"

    def test_yesterday_record_field_present(self, client):
        """yesterday_record field is present (may be None)."""
        with patch("backend.api._query_yesterday_record", return_value=None), \
             patch("backend.api._get_today_preview", return_value=None), \
             patch("backend.api.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            resp = client.get("/api/briefing?date=2026-03-01")

        data = resp.json()
        assert "yesterday_record" in data

    def test_today_preview_field_present(self, client):
        """today_preview field is present."""
        preview = {"actionable_plays": 3, "games_count": 5, "games_analyzed": 4}
        with patch("backend.api._query_yesterday_record", return_value=None), \
             patch("backend.api._get_today_preview", return_value=preview), \
             patch("backend.api.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            resp = client.get("/api/briefing?date=2026-03-01")

        data = resp.json()
        assert data["today_preview"] == preview

    def test_briefing_text_includes_yesterday_section(self, client):
        """briefing_text includes YESTERDAY'S RECORD section."""
        with patch("backend.api._query_yesterday_record", return_value=None), \
             patch("backend.api._get_today_preview", return_value=None), \
             patch("backend.api.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            resp = client.get("/api/briefing?date=2026-03-01")

        data = resp.json()
        assert "YESTERDAY'S RECORD" in data["briefing_text"]

    def test_briefing_text_includes_today_preview(self, client):
        """briefing_text includes TODAY'S PREVIEW when preview data exists."""
        preview = {"actionable_plays": 2, "games_count": 4, "games_analyzed": 3}
        with patch("backend.api._query_yesterday_record", return_value=None), \
             patch("backend.api._get_today_preview", return_value=preview), \
             patch("backend.api.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            resp = client.get("/api/briefing?date=2026-03-01")

        data = resp.json()
        assert "TODAY'S PREVIEW" in data["briefing_text"]
        assert "2 actionable plays" in data["briefing_text"]

    def test_with_yesterday_record_data(self, client):
        """When yesterday has data, it flows into both the field and text."""
        record = {
            "date": "2026-02-28",
            "overall": {"wins": 4, "losses": 2, "pushes": 0, "total": 6, "hit_rate": 66.7, "profit": 50.0, "roi": 3.5},
            "by_bet_type": {"Points": {"wins": 3, "losses": 1, "total": 4, "hit_rate": 75.0}},
            "by_confidence": {},
            "clv_summary": None,
            "source": "calibration",
        }
        with patch("backend.api._query_yesterday_record", return_value=record), \
             patch("backend.api._get_today_preview", return_value=None), \
             patch("backend.api.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            resp = client.get("/api/briefing?date=2026-03-01")

        data = resp.json()
        assert data["yesterday_record"] is not None
        assert data["yesterday_record"]["overall"]["wins"] == 4
        assert "4-2" in data["briefing_text"]
