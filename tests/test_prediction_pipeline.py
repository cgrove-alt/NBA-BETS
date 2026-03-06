import pytest

from nba_betting.prediction_pipeline import (
    apply_sample_size_confidence_shrink,
    evaluate_bet,
    evaluate_bets_batch,
)


def test_confidence_shrink_respects_sample_size():
    low_sample = apply_sample_size_confidence_shrink(0.80, games_played=5)
    high_sample = apply_sample_size_confidence_shrink(0.80, games_played=100)

    assert 0.5 < low_sample < high_sample < 0.95


def test_evaluate_bet_applies_reliability_shrink():
    low_games = evaluate_bet(
        prop_type="points",
        predicted=30.0,
        line=27.0,
        raw_confidence=0.75,
        games_played=10,
        bankroll=1000.0,
        over_odds=-110,
        under_odds=-110,
        pre_calibrated=True,
    )
    high_games = evaluate_bet(
        prop_type="points",
        predicted=30.0,
        line=27.0,
        raw_confidence=0.75,
        games_played=100,
        bankroll=1000.0,
        over_odds=-110,
        under_odds=-110,
        pre_calibrated=True,
    )

    assert low_games["confidence_reliability"] < high_games["confidence_reliability"]
    assert low_games["confidence"] < high_games["confidence"]


def test_evaluate_bets_batch_forwards_pre_calibrated_flag():
    predictions = [
        {
            "prop_type": "points",
            "predicted": 34.0,
            "line": 30.0,
            "raw_confidence": 0.99,
            "games_played": 100,
            "over_odds": -110,
            "under_odds": -110,
            "pre_calibrated": True,
        },
        {
            "prop_type": "points",
            "predicted": 34.0,
            "line": 30.0,
            "raw_confidence": 0.99,
            "games_played": 100,
            "over_odds": -110,
            "under_odds": -110,
            "pre_calibrated": False,
        },
    ]
    results = evaluate_bets_batch(predictions, bankroll=1000.0)

    assert len(results) == 2
    assert results[0]["confidence"] > results[1]["confidence"]
