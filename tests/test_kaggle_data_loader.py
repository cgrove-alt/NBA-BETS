from __future__ import annotations

import pandas as pd

from nba_data.sources.kaggle_data_loader import (
    DATA_DIR,
    LIVE_SEASONS_DIR,
    load_live_season_data,
    process_games_to_matchups,
)


def test_loader_uses_real_workspace_data_directories():
    assert DATA_DIR.exists()
    assert LIVE_SEASONS_DIR.exists()


def test_load_live_season_data_reads_repo_cache():
    live_df = load_live_season_data()
    assert not live_df.empty
    assert "SEASON_YEAR" in live_df.columns


def test_process_games_to_matchups_builds_home_away_rows():
    df = pd.DataFrame(
        [
            {
                "GAME_ID": 1,
                "GAME_DATE": "2025-01-01",
                "TEAM_ABBREVIATION": "BOS",
                "MATCHUP": "BOS vs. NYK",
                "SEASON_YEAR": "2024-25",
                "PTS": 100,
                "WL": "W",
                "FG_PCT": 0.5,
                "FG3_PCT": 0.4,
                "FT_PCT": 0.8,
                "REB": 40,
                "AST": 25,
                "TOV": 10,
                "STL": 7,
                "BLK": 5,
                "PLUS_MINUS": 8,
            },
            {
                "GAME_ID": 1,
                "GAME_DATE": "2025-01-01",
                "TEAM_ABBREVIATION": "NYK",
                "MATCHUP": "NYK @ BOS",
                "SEASON_YEAR": "2024-25",
                "PTS": 92,
                "WL": "L",
                "FG_PCT": 0.45,
                "FG3_PCT": 0.35,
                "FT_PCT": 0.75,
                "REB": 38,
                "AST": 22,
                "TOV": 12,
                "STL": 6,
                "BLK": 4,
                "PLUS_MINUS": -8,
            },
        ]
    )

    games_df = process_games_to_matchups(df)

    assert len(games_df) == 1
    game = games_df.iloc[0]
    assert game["home_team"] == "BOS"
    assert game["away_team"] == "NYK"
    assert game["home_score"] == 100
    assert game["away_score"] == 92
