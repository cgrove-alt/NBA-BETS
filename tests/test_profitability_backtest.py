from pathlib import Path

import pytest

import nba_models.backtesting.profitability_backtest as pb


def test_candidate_rank_score_prefers_true_ev():
    assert pb._candidate_rank_score({"true_ev": 0.08, "confidence": 0.99}) == 0.08


def test_candidate_rank_score_falls_back_to_confidence():
    score = pb._candidate_rank_score({"confidence": 0.63})
    assert score == pytest.approx(0.13)


def test_generate_report_includes_simulation_settings(tmp_path):
    original_output_dir = pb.OUTPUT_DIR
    original_bankroll = pb.INITIAL_BANKROLL
    original_season = pb.TEST_SEASON
    try:
        pb.OUTPUT_DIR = str(tmp_path)
        pb.INITIAL_BANKROLL = 1000.0
        pb.TEST_SEASON = "2023-24"

        trades = [
            {
                "date": "2023-10-24",
                "player": "Player A",
                "prop_type": "points",
                "prop_line": 20.5,
                "predicted": 23.2,
                "actual": 25,
                "direction": "over",
                "edge": 2.7,
                "confidence": 0.63,
                "tier": "moderate",
                "bet_size": 15.0,
                "won": True,
                "pnl": 13.64,
                "bankroll": 1013.64,
            },
            {
                "date": "2023-10-25",
                "player": "Player B",
                "prop_type": "assists",
                "prop_line": 5.5,
                "predicted": 7.0,
                "actual": 4,
                "direction": "over",
                "edge": 1.5,
                "confidence": 0.60,
                "tier": "moderate",
                "bet_size": 15.0,
                "won": False,
                "pnl": -15.0,
                "bankroll": 998.64,
            },
        ]
        daily_bankroll = {
            "2023-10-24": 1013.64,
            "2023-10-25": 998.64,
        }

        result = pb.generate_report(
            trades=trades,
            daily_bankroll=daily_bankroll,
            max_bets_per_player_sample=2,
            progress_interval=100,
        )

        assert result["simulation_settings"]["max_bets_per_player_sample"] == 2
        assert result["simulation_settings"]["progress_interval"] == 100

        assert Path(tmp_path, "profitability_backtest_results.json").exists()
        assert Path(tmp_path, "profitability_backtest_report.txt").exists()
    finally:
        pb.OUTPUT_DIR = original_output_dir
        pb.INITIAL_BANKROLL = original_bankroll
        pb.TEST_SEASON = original_season
