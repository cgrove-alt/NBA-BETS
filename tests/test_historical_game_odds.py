import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba_models.backtesting.fetch_historical_lines import (
    derive_game_market_history,
    summarize_game_market_snapshot,
)


def test_summarize_game_market_snapshot_builds_consensus_values():
    raw = {
        "data": {
            "home_team": "Boston Celtics",
            "away_team": "New York Knicks",
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -150},
                                {"name": "New York Knicks", "price": 130},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -110, "point": -5.5},
                                {"name": "New York Knicks", "price": -110, "point": 5.5},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -108, "point": 227.5},
                                {"name": "Under", "price": -112, "point": 227.5},
                            ],
                        },
                    ]
                },
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -160},
                                {"name": "New York Knicks", "price": 138},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -112, "point": -6.0},
                                {"name": "New York Knicks", "price": -108, "point": 6.0},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -105, "point": 228.0},
                                {"name": "Under", "price": -115, "point": 228.0},
                            ],
                        },
                    ]
                },
            ]
        }
    }

    summary = summarize_game_market_snapshot(raw)

    assert summary["book_count"] == 2
    assert summary["spread"]["home_line"] == pytest.approx(-5.75)
    assert summary["spread"]["away_line"] == pytest.approx(5.75)
    assert summary["moneyline"]["home_odds"] == pytest.approx(-155.0)
    assert summary["moneyline"]["away_odds"] == pytest.approx(134.0)
    assert summary["totals"]["line"] == pytest.approx(227.75)


def test_derive_game_market_history_computes_true_movement_flags():
    opening = {
        "spread": {"home_line": -4.5, "home_odds": -110},
        "moneyline": {"home_odds": -145},
        "totals": {"line": 225.5},
    }
    pregame = {
        "spread": {"home_line": -5.0, "home_odds": -111},
        "moneyline": {"home_odds": -152},
        "totals": {"line": 226.0},
    }
    closing = {
        "spread": {"home_line": -7.0, "home_odds": -114},
        "moneyline": {"home_odds": -170},
        "totals": {"line": 226.5},
    }

    derived = derive_game_market_history(opening, pregame, closing)

    assert derived["opening_line"] == -4.5
    assert derived["closing_line"] == -7.0
    assert derived["line_movement"] == pytest.approx(-2.5)
    assert derived["consensus_odds"] == -114
    assert derived["rlm_flag"] is True
    assert derived["steam_move_flag"] is True
    assert derived["moneyline_home_prob_movement"] > 0
