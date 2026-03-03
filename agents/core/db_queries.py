"""
Shared DB query functions for yesterday's betting record.

Used by both the briefing agent (direct DB fallback) and backend/api.py.
Queries calibration.db and bet_tracking.db — no FastAPI dependencies.
"""

import json
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def query_yesterday_record(yesterday_str: str) -> dict | None:
    """Query yesterday's prediction results from PostgreSQL, calibration.db, or bet_tracking.db.

    Returns a structured dict with overall, by_bet_type, by_confidence,
    clv_summary, and date fields — or None if no data is available.
    """
    record: dict | None = None

    # --- Attempt 0: PostgreSQL paper_trades (production / Railway) ---
    if os.environ.get('DATABASE_URL'):
        try:
            from nba_betting.paper_trading import PaperTrader
            trader = PaperTrader()
            if trader._use_postgres:
                report = trader.get_daily_report(yesterday_str)
                if report and report.get('total', 0) > 0 and report.get('settled', 0) > 0:
                    preds = report.get('predictions', [])
                    wins = sum(1 for p in preds if p.get('result') == 'hit')
                    losses = sum(1 for p in preds if p.get('result') == 'miss')
                    pushes = sum(1 for p in preds if p.get('result') == 'push')
                    total = wins + losses + pushes
                    hit_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0
                    profit = report.get('profit_loss', 0.0)

                    by_type: dict[str, dict] = {}
                    by_conf: dict[str, dict] = {
                        "high": {"wins": 0, "losses": 0, "total": 0},
                        "medium": {"wins": 0, "losses": 0, "total": 0},
                        "low": {"wins": 0, "losses": 0, "total": 0},
                    }
                    for p in preds:
                        result = p.get('result')
                        if result not in ('hit', 'miss'):
                            continue
                        pt = p.get('prop_type', 'Unknown')
                        if pt not in by_type:
                            by_type[pt] = {"wins": 0, "losses": 0, "total": 0}
                        by_type[pt]["total"] += 1
                        if result == 'hit':
                            by_type[pt]["wins"] += 1
                        else:
                            by_type[pt]["losses"] += 1

                        conf = float(p.get('confidence') or 0)
                        if conf >= 60:
                            tier = "high"
                        elif conf >= 55:
                            tier = "medium"
                        else:
                            tier = "low"
                        by_conf[tier]["total"] += 1
                        if result == 'hit':
                            by_conf[tier]["wins"] += 1
                        else:
                            by_conf[tier]["losses"] += 1

                    for v in by_type.values():
                        denom = v["wins"] + v["losses"]
                        v["hit_rate"] = round(v["wins"] / denom * 100, 1) if denom > 0 else 0.0
                    for v in by_conf.values():
                        denom = v["wins"] + v["losses"]
                        v["hit_rate"] = round(v["wins"] / denom * 100, 1) if denom > 0 else 0.0

                    record = {
                        "date": yesterday_str,
                        "overall": {
                            "wins": wins,
                            "losses": losses,
                            "pushes": pushes,
                            "total": total,
                            "hit_rate": hit_rate,
                            "profit": round(profit, 2),
                            "roi": 0.0,
                        },
                        "by_bet_type": by_type,
                        "by_confidence": by_conf,
                        "clv_summary": None,
                        "source": "paper_trades_pg",
                    }
        except Exception as e:
            logger.warning(f"PostgreSQL paper_trades query failed: {e}")

    if record is not None:
        return record

    # --- Attempt 1: calibration.db (has predictions + outcomes) ---
    cal_path = Path("data/calibration.db")
    if cal_path.exists():
        try:
            conn = sqlite3.connect(str(cal_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT p.prop_type, p.confidence, o.hit, o.clv
                FROM predictions p
                JOIN outcomes o ON p.id = o.prediction_id
                WHERE p.game_date = ?
            """, (yesterday_str,)).fetchall()
            conn.close()

            if rows:
                record = _build_record_from_calibration(rows, yesterday_str)
        except Exception:
            pass

    if record is not None:
        return record

    # --- Attempt 2: bet_tracking.db ---
    bt_path = Path("data/bet_tracking.db")
    if bt_path.exists():
        try:
            conn = sqlite3.connect(str(bt_path))
            conn.row_factory = sqlite3.Row

            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            bet_table = "tracked_bets" if "tracked_bets" in tables else "bets"

            rows = conn.execute(f"""
                SELECT status, pnl, tags, bet_type
                FROM {bet_table}
                WHERE event_date LIKE ?
                  AND status IN ('won', 'lost', 'push')
            """, (f"{yesterday_str}%",)).fetchall()
            conn.close()

            if rows:
                record = _build_record_from_tracking(rows, yesterday_str)
        except Exception:
            pass

    return record


def _build_record_from_calibration(rows, yesterday_str: str) -> dict:
    """Build a yesterday_record dict from calibration.db prediction+outcome rows."""
    wins = losses = pushes = 0
    total_clv = 0.0
    clv_count = 0
    positive_clv = 0
    by_type: dict[str, dict] = {}
    by_conf: dict[str, dict] = {"high": {"wins": 0, "losses": 0, "total": 0},
                                 "medium": {"wins": 0, "losses": 0, "total": 0},
                                 "low": {"wins": 0, "losses": 0, "total": 0}}

    for row in rows:
        hit = row["hit"]
        prop_type = row["prop_type"] or "Unknown"
        confidence = row["confidence"] or 0
        clv = row["clv"]

        if hit is None:
            pushes += 1
            continue
        if hit:
            wins += 1
        else:
            losses += 1

        # By bet type
        if prop_type not in by_type:
            by_type[prop_type] = {"wins": 0, "losses": 0, "total": 0}
        by_type[prop_type]["total"] += 1
        if hit:
            by_type[prop_type]["wins"] += 1
        else:
            by_type[prop_type]["losses"] += 1

        # By confidence tier
        if confidence >= 60:
            tier = "high"
        elif confidence >= 55:
            tier = "medium"
        else:
            tier = "low"
        by_conf[tier]["total"] += 1
        if hit:
            by_conf[tier]["wins"] += 1
        else:
            by_conf[tier]["losses"] += 1

        # CLV
        if clv is not None:
            total_clv += clv
            clv_count += 1
            if clv > 0:
                positive_clv += 1

    total = wins + losses + pushes
    hit_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0

    # Add hit_rate to by_type / by_conf entries
    for v in by_type.values():
        denom = v["wins"] + v["losses"]
        v["hit_rate"] = round(v["wins"] / denom * 100, 1) if denom > 0 else 0.0
    for v in by_conf.values():
        denom = v["wins"] + v["losses"]
        v["hit_rate"] = round(v["wins"] / denom * 100, 1) if denom > 0 else 0.0

    clv_summary = None
    if clv_count > 0:
        clv_summary = {
            "avg_clv": round(total_clv / clv_count, 2),
            "positive_clv_rate": round(positive_clv / clv_count * 100, 1),
        }

    return {
        "date": yesterday_str,
        "overall": {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total": total,
            "hit_rate": hit_rate,
            "profit": 0.0,   # calibration.db doesn't track dollar P&L
            "roi": 0.0,
        },
        "by_bet_type": by_type,
        "by_confidence": by_conf,
        "clv_summary": clv_summary,
        "source": "calibration",
    }


def _build_record_from_tracking(rows, yesterday_str: str) -> dict:
    """Build a yesterday_record dict from bet_tracking.db rows."""
    wins = losses = pushes = 0
    total_pnl = 0.0
    by_type: dict[str, dict] = {}

    for row in rows:
        status = row["status"]
        pnl = row["pnl"] or 0.0
        total_pnl += pnl

        # Try to extract prop type from tags
        tags_raw = row["tags"]
        bet_type_raw = row["bet_type"] or "Unknown"
        prop_type = bet_type_raw
        if tags_raw:
            try:
                tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
                if isinstance(tags, dict):
                    prop_type = tags.get("prop_type", bet_type_raw)
                elif isinstance(tags, list) and tags:
                    prop_type = tags[0]
            except Exception:
                pass

        if status == "won":
            wins += 1
        elif status == "lost":
            losses += 1
        elif status == "push":
            pushes += 1

        if prop_type not in by_type:
            by_type[prop_type] = {"wins": 0, "losses": 0, "total": 0}
        by_type[prop_type]["total"] += 1
        if status == "won":
            by_type[prop_type]["wins"] += 1
        elif status == "lost":
            by_type[prop_type]["losses"] += 1

    total = wins + losses + pushes
    hit_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0

    for v in by_type.values():
        denom = v["wins"] + v["losses"]
        v["hit_rate"] = round(v["wins"] / denom * 100, 1) if denom > 0 else 0.0

    return {
        "date": yesterday_str,
        "overall": {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total": total,
            "hit_rate": hit_rate,
            "profit": round(total_pnl, 2),
            "roi": 0.0,  # can't compute without stake data from this query
        },
        "by_bet_type": by_type,
        "by_confidence": {},  # bet_tracking.db doesn't store confidence tiers
        "clv_summary": None,
        "source": "bet_tracking",
    }
