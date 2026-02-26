#!/usr/bin/env python3
"""Generate backtest cache files from CSV data for Phase 3 backtest."""
import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = '/home/user/workspace/NBA-BETS'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'nba_models', 'training'))
sys.path.insert(0, ROOT)

from train_from_csv import build_team_id_map, _build_team_metadata, _safe_int

CACHE_DIR = Path("data/balldontlie_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

team_id_map = build_team_id_map()
team_meta = _build_team_metadata()

# Load the live seasons CSV (has 2024-25 and 2025-26)
live = pd.read_csv('data/live_seasons/live_seasons_20251213.csv')
print(f"Live seasons CSV: {len(live)} rows")
print(f"Seasons: {live['SEASON_YEAR'].unique()}")

# For each season, build the games cache file
for bdl_season, csv_season_label in [(2025, '2024-25'), (2026, '2025-26')]:
    print(f"\n=== Building games_{bdl_season}_full.json for {csv_season_label} ===")

    season_rows = live[live['SEASON_YEAR'] == csv_season_label].copy()
    print(f"  Found {len(season_rows)} team-game rows")

    season_rows['GAME_DATE'] = season_rows['GAME_DATE'].astype(str).str[:10]

    games = []
    for game_id_raw, grp in season_rows.groupby('GAME_ID'):
        if len(grp) != 2:
            continue

        game_id = int(game_id_raw)
        rows = grp.to_dict('records')

        home_row = away_row = None
        for r in rows:
            matchup = str(r.get('MATCHUP', ''))
            if 'vs.' in matchup:
                home_row = r
            elif '@' in matchup:
                away_row = r

        if home_row is None or away_row is None:
            if home_row is not None:
                away_row = [r for r in rows if r is not home_row][0]
            elif away_row is not None:
                home_row = [r for r in rows if r is not away_row][0]
            else:
                continue

        home_nba_id = int(home_row['TEAM_ID'])
        away_nba_id = int(away_row['TEAM_ID'])
        home_cid = team_id_map.get(home_nba_id, home_nba_id % 30 + 1)
        away_cid = team_id_map.get(away_nba_id, away_nba_id % 30 + 1)

        home_info = team_meta.get(home_nba_id, {})
        away_info = team_meta.get(away_nba_id, {})

        game_date = str(home_row.get('GAME_DATE', ''))[:10]
        home_score = _safe_int(home_row.get('PTS', 0))
        away_score = _safe_int(away_row.get('PTS', 0))

        if not game_date or home_score == 0:
            continue

        game_dict = {
            'id': game_id,
            'date': game_date,
            'status': 'Final',
            'home_team': {
                'id': home_cid,
                'abbreviation': home_info.get('abbreviation', str(home_row.get('TEAM_ABBREVIATION',''))),
                'full_name': home_info.get('full_name', str(home_row.get('TEAM_NAME',''))),
                'name': home_info.get('name', ''),
                'city': home_info.get('city', ''),
            },
            'visitor_team': {
                'id': away_cid,
                'abbreviation': away_info.get('abbreviation', str(away_row.get('TEAM_ABBREVIATION',''))),
                'full_name': away_info.get('full_name', str(away_row.get('TEAM_NAME',''))),
                'name': away_info.get('name', ''),
                'city': away_info.get('city', ''),
            },
            'home_team_score': home_score,
            'visitor_team_score': away_score,
        }
        games.append(game_dict)

    games.sort(key=lambda g: g['date'])

    cache_data = {'games': games, 'complete': True, 'season': bdl_season}
    cache_file = CACHE_DIR / f"games_{bdl_season}_full.json"
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"  Built {len(games)} game dicts")
    if games:
        print(f"  Date range: {games[0]['date']} to {games[-1]['date']}")
    print(f"  Saved to: {cache_file}")

# Generate player stats cache from box scores
print("\n=== Building player stats cache ===")
BOX_PARTS = [
    os.path.join(ROOT, 'data', 'NBA-Data-2010-2024-main',
                 f'regular_season_box_scores_2010_2024_part_{i}.csv')
    for i in range(1, 4)
]

target_seasons = {'2023-24'}
parts = []
for path in BOX_PARTS:
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df['season_year'].isin(target_seasons)]
        parts.append(df)
        print(f"  Loaded {len(df)} rows from {Path(path).name}")

if parts:
    box = pd.concat(parts, ignore_index=True)
    print(f"  Total: {len(box)} player box score rows")

    games_saved = 0
    for game_id, group in box.groupby('gameId'):
        player_list = []
        game_date_str = ''
        for _, row in group.iterrows():
            full_name = str(row.get('personName', ''))
            name_parts = full_name.strip().split(' ', 1)
            first = name_parts[0] if name_parts else ''
            last = name_parts[1] if len(name_parts) > 1 else ''

            nba_team_id = int(row['teamId'])
            compact_tid = team_id_map.get(nba_team_id, nba_team_id % 30 + 1)

            raw_min = row.get('minutes', '')
            min_str = str(raw_min).strip() if pd.notna(raw_min) and str(raw_min).strip() else '0:00'
            game_date_str = str(row.get('game_date', ''))[:10]

            stat_dict = {
                'player': {
                    'id': int(row['personId']),
                    'first_name': first,
                    'last_name': last,
                    'position': str(row.get('position', '')) if pd.notna(row.get('position')) else '',
                },
                'team': {'id': compact_tid, 'abbreviation': str(row.get('teamTricode', ''))},
                'game': {'id': int(game_id), 'date': game_date_str},
                'min': min_str,
                'pts': _safe_int(row.get('points')),
                'reb': _safe_int(row.get('reboundsTotal')),
                'ast': _safe_int(row.get('assists')),
                'stl': _safe_int(row.get('steals')),
                'blk': _safe_int(row.get('blocks')),
                'turnover': _safe_int(row.get('turnovers')),
                'pf': _safe_int(row.get('foulsPersonal')),
                'fgm': _safe_int(row.get('fieldGoalsMade')),
                'fga': _safe_int(row.get('fieldGoalsAttempted')),
                'fg3m': _safe_int(row.get('threePointersMade')),
                'fg3a': _safe_int(row.get('threePointersAttempted')),
                'ftm': _safe_int(row.get('freeThrowsMade')),
                'fta': _safe_int(row.get('freeThrowsAttempted')),
                'oreb': _safe_int(row.get('reboundsOffensive')),
                'dreb': _safe_int(row.get('reboundsDefensive')),
            }
            player_list.append(stat_dict)

        cache_file = CACHE_DIR / f"player_stats_{game_id}.json"
        with open(cache_file, 'w') as f:
            json.dump(player_list, f)
        games_saved += 1

        if games_saved % 200 == 0:
            print(f"    Progress: {games_saved} games saved")

    print(f"  Saved player stats for {games_saved} games to cache")

print("\n✓ Cache generation complete!")
