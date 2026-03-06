import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba_models.training.train_stacking_model import TrainingDataLoader


def _seed_recent_team_games(loader: TrainingDataLoader, team_id: int, team_abbrev: str, opponent_abbrevs: list[str]):
    for idx, opp in enumerate(opponent_abbrevs):
        game_date = f"2025-01-0{idx + 5}"
        loader.team_history[team_id].append(
            {
                "date": game_date,
                "is_home": idx % 2 == 0,
                "pts_scored": 110 + idx,
                "pts_allowed": 102 - idx,
                "won": True,
                "point_diff": 8 + idx,
                "opponent_id": 900 + idx,
                "team_abbrev": team_abbrev,
                "opponent_abbrev": opp,
                "venue_abbrev": team_abbrev if idx % 2 == 0 else opp,
            }
        )


def test_extract_context_features_uses_real_availability_market_and_travel_signals():
    loader = TrainingDataLoader()

    home_id = 1
    away_id = 2
    home_abbrev = "ATL"
    away_abbrev = "BOS"

    _seed_recent_team_games(loader, home_id, home_abbrev, ["CHI", "MIA", "NYK"])
    _seed_recent_team_games(loader, away_id, away_abbrev, ["PHI", "NYK", "BKN"])

    # Home team recent rotation: player 11 is a star and goes missing on game day.
    for game_date in ["2025-01-05", "2025-01-06", "2025-01-07"]:
        loader.team_players_by_game[(game_date, home_abbrev)] = {
            11: {"min": 34.0, "pts": 24, "reb": 7, "ast": 5, "pra": 36},
            12: {"min": 28.0, "pts": 14, "reb": 4, "ast": 3, "pra": 21},
            13: {"min": 22.0, "pts": 10, "reb": 5, "ast": 2, "pra": 17},
        }

    for game_date in ["2025-01-05", "2025-01-06", "2025-01-07"]:
        loader.team_players_by_game[(game_date, away_abbrev)] = {
            21: {"min": 33.0, "pts": 21, "reb": 8, "ast": 4, "pra": 33},
            22: {"min": 29.0, "pts": 15, "reb": 6, "ast": 5, "pra": 26},
            23: {"min": 24.0, "pts": 11, "reb": 4, "ast": 3, "pra": 18},
        }

    loader.market_active_players[("2025-01-10", home_abbrev)] = {12, 13}
    loader.market_active_players[("2025-01-10", away_abbrev)] = {21, 22, 23}
    loader.market_context[("2025-01-10", home_abbrev, away_abbrev)] = {
        "market_strength_diff": 6.0,
        "snapshot_count": 2,
    }
    loader.game_market_context[("2025-01-10", home_abbrev, away_abbrev)] = {
        "opening_line": -4.5,
        "closing_line": -6.0,
        "line_movement": -1.5,
        "consensus_odds": -112,
        "rlm_flag": 1,
        "steam_move_flag": 0,
        "moneyline_home_prob_movement": 0.022,
    }

    game = {
        "date": "2025-01-10",
        "home_team": {"id": home_id, "abbreviation": home_abbrev},
        "visitor_team": {"id": away_id, "abbreviation": away_abbrev},
    }
    home_feats = {
        "recent_pts_avg": 118.0,
        "point_diff_avg": 6.0,
        "home_win_pct": 0.70,
    }
    away_feats = {
        "recent_pts_avg": 111.0,
        "point_diff_avg": 1.0,
        "away_win_pct": 0.45,
    }

    context = loader._extract_context_features(game, home_feats, away_feats)

    assert context["ctx_injury_count_home"] == 1
    assert context["ctx_star_player_out_home"] == 1
    assert context["ctx_injury_count_away"] == 0
    assert context["ctx_star_player_out_away"] == 0
    assert context["ctx_market_strength_diff"] == 6.0
    assert context["ctx_opening_line"] == -4.5
    assert context["ctx_closing_line"] == -6.0
    assert context["ctx_line_movement"] == -1.5
    assert context["ctx_rlm_flag"] == 1
    assert context["ctx_consensus_odds"] == -112
    assert context["ctx_steam_move_flag"] == 0
    assert context["ctx_moneyline_home_prob_movement"] == pytest.approx(0.022)
    assert context["ctx_rest_days_diff"] == 0
    assert context["ctx_avg_pace"] == 114.5
    assert context["ctx_home_advantage_factor"] == pytest.approx(0.25)
    assert context["ctx_away_is_b2b"] == 0
    assert context["ctx_prediction_variance"] > 0
    assert context["ctx_away_travel_distance"] > 0
