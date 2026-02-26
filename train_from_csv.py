#!/usr/bin/env python3
"""
Train NBA models from local CSV data (no API needed).

Loads 14 years of team-level and player-level data from local CSVs and
converts them to the format expected by the existing training pipeline.

Usage:
    python3 train_from_csv.py
    python3 train_from_csv.py --seasons 2021 2022 2023 2024
    python3 train_from_csv.py --seasons 2022 2023 --use-optuna
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path setup — must happen before any project imports
# ---------------------------------------------------------------------------
ROOT = os.environ.get('NBA_BETS_ROOT', os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'nba_models', 'training'))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------
TOTALS_CSV     = os.path.join(ROOT, 'data', 'NBA-Data-2010-2024-main',
                               'regular_season_totals_2010_2024.csv')
LIVE_CSV       = os.path.join(ROOT, 'data', 'live_seasons',
                               'live_seasons_20251213.csv')
BOX_PARTS      = [
    os.path.join(ROOT, 'data', 'NBA-Data-2010-2024-main',
                 f'regular_season_box_scores_2010_2024_part_{i}.csv')
    for i in range(1, 4)
]

# ---------------------------------------------------------------------------
# Season-year helpers
# ---------------------------------------------------------------------------
# The CSVs use labels like "2021-22".  The --seasons flag accepts the *start*
# year as an integer (e.g. 2021 → "2021-22").

def int_to_season_label(year: int) -> str:
    """2021  →  '2021-22'"""
    return f"{year}-{str(year + 1)[-2:]}"


def season_label_to_int(label: str) -> int:
    """'2021-22'  →  2021"""
    return int(label.split('-')[0])


# ---------------------------------------------------------------------------
# Team ID mapping  (NBA.com 10-digit IDs → compact sequential 1-30)
# ---------------------------------------------------------------------------

# All 30 current/historical franchises in the CSVs, sorted by NBA.com ID.
# We assign sequential IDs 1-30 in sorted order so the mapping is stable
# across runs.  Teams that changed abbreviations (NJN→BKN, NOH→NOP, etc.)
# share the same NBA.com ID and therefore the same compact ID.
_KNOWN_NBA_IDS = [
    1610612737,  # ATL
    1610612738,  # BOS
    1610612739,  # CLE
    1610612740,  # NOP / NOH
    1610612741,  # CHI
    1610612742,  # DAL
    1610612743,  # DEN
    1610612744,  # GSW
    1610612745,  # HOU
    1610612746,  # LAC
    1610612747,  # LAL
    1610612748,  # MIA
    1610612749,  # MIL
    1610612750,  # MIN
    1610612751,  # BKN / NJN
    1610612752,  # NYK
    1610612753,  # ORL
    1610612754,  # IND
    1610612755,  # PHI
    1610612756,  # PHX
    1610612757,  # POR
    1610612758,  # SAC
    1610612759,  # SAS
    1610612760,  # OKC
    1610612761,  # TOR
    1610612762,  # UTA
    1610612763,  # MEM
    1610612764,  # WAS
    1610612765,  # DET
    1610612766,  # CHA / Bobcats
]


def build_team_id_map() -> dict[int, int]:
    """
    Return a dict mapping NBA.com team IDs (e.g. 1610612744) to compact
    sequential IDs (1–30).

    Any ID not in the pre-built list gets appended in sorted order so the
    function is forward-compatible.
    """
    known = sorted(set(_KNOWN_NBA_IDS))
    return {nba_id: idx + 1 for idx, nba_id in enumerate(known)}


# ---------------------------------------------------------------------------
# Team metadata cache  (NBA.com ID → {abbreviation, full_name, name, city})
# ---------------------------------------------------------------------------

def _build_team_metadata() -> dict[int, dict]:
    """
    Read both CSVs and build a per-NBA-ID metadata dict.
    Later entries override earlier ones, so modern abbreviations win.
    """
    meta: dict[int, dict] = {}

    def _parse_full_name(team_name: str, city_guess: str = '') -> tuple[str, str]:
        """
        Split "Golden State Warriors" into (city="Golden State", name="Warriors").
        Falls back gracefully if the name is unusual.
        """
        parts = team_name.rsplit(' ', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return city_guess, team_name

    # ---------- historical totals ----------
    totals = pd.read_csv(TOTALS_CSV, usecols=['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME'])
    for _, row in totals.drop_duplicates(subset='TEAM_ID').iterrows():
        tid = int(row['TEAM_ID'])
        city, name = _parse_full_name(str(row['TEAM_NAME']))
        meta[tid] = {
            'abbreviation': str(row['TEAM_ABBREVIATION']),
            'full_name':    str(row['TEAM_NAME']),
            'name':         name,
            'city':         city,
        }

    # ---------- live seasons (newer abbreviations take priority) ----------
    if os.path.exists(LIVE_CSV):
        live = pd.read_csv(LIVE_CSV, usecols=['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME'])
        for _, row in live.drop_duplicates(subset='TEAM_ID').iterrows():
            tid = int(row['TEAM_ID'])
            city, name = _parse_full_name(str(row['TEAM_NAME']))
            meta[tid] = {
                'abbreviation': str(row['TEAM_ABBREVIATION']),
                'full_name':    str(row['TEAM_NAME']),
                'name':         name,
                'city':         city,
            }

    return meta


# ---------------------------------------------------------------------------
# Safe numeric helper
# ---------------------------------------------------------------------------

def _safe_int(val, default: int = 0) -> int:
    try:
        f = float(val)
        return int(f) if not np.isnan(f) else default
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if not np.isnan(f) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Load team-level games
# ---------------------------------------------------------------------------

def load_team_games(seasons: list[str],
                    team_id_map: dict[int, int],
                    team_meta: dict[int, dict]) -> list[dict]:
    """
    Load game-level data from the team totals CSVs and convert to the
    format expected by process_games_for_training().

    Parameters
    ----------
    seasons : list of season labels to include, e.g. ['2021-22', '2022-23']
    team_id_map : NBA.com ID → compact sequential ID (1-30)
    team_meta   : NBA.com ID → {abbreviation, full_name, name, city}

    Returns
    -------
    List of game dicts in BallDontLie API format.
    """
    season_set = set(seasons)
    print(f"\n[load_team_games] Loading seasons: {sorted(season_set)}")

    # ------------------------------------------------------------------
    # 1. Historical totals  (up to and including 2023-24)
    # ------------------------------------------------------------------
    print(f"  Reading {TOTALS_CSV} ...")
    totals = pd.read_csv(TOTALS_CSV)
    totals = totals[totals['SEASON_YEAR'].isin(season_set)].copy()
    print(f"  → {len(totals):,} rows after season filter")

    # Normalise GAME_DATE: strip time component if present
    totals['GAME_DATE'] = totals['GAME_DATE'].astype(str).str[:10]

    # ------------------------------------------------------------------
    # 2. Live seasons  (2023-24, 2024-25, 2025-26)
    #    Only include seasons NOT already covered by the historical CSV
    #    to avoid duplicates (historical has player box scores too).
    # ------------------------------------------------------------------
    set(totals['SEASON_YEAR'].unique()) if len(totals) else set()
    live_seasons_needed = season_set - {'2023-24'}   # 2023-24 is in both; prefer historical
    # Keep any season in live CSV that the user requested and that ISN'T
    # fully covered by the historical totals
    live_extra = live_seasons_needed - set(
        pd.read_csv(TOTALS_CSV, usecols=['SEASON_YEAR'])['SEASON_YEAR'].unique()
    )

    live_rows = pd.DataFrame()
    if os.path.exists(LIVE_CSV) and live_extra:
        print(f"  Reading {LIVE_CSV} for seasons: {sorted(live_extra)} ...")
        live = pd.read_csv(LIVE_CSV)
        live = live[live['SEASON_YEAR'].isin(live_extra)].copy()
        # Align column names to the historical format where they differ
        live = live.rename(columns={'SEASON_ID': 'SEASON_ID_ORIG'})
        # live CSV already has SEASON_YEAR; GAME_DATE is already "YYYY-MM-DD"
        live_rows = live
        print(f"  → {len(live_rows):,} live rows added")

    # ------------------------------------------------------------------
    # 3. Combine
    # ------------------------------------------------------------------
    combined = pd.concat([totals, live_rows], ignore_index=True, sort=False)
    print(f"  Combined: {len(combined):,} team-game rows")

    # ------------------------------------------------------------------
    # 4. Build game dicts  (one row per team, two rows per game)
    # ------------------------------------------------------------------
    # Group by GAME_ID; each group should have exactly 2 rows.
    games: list[dict] = []
    skipped = 0

    for game_id_raw, grp in combined.groupby('GAME_ID'):
        if len(grp) != 2:
            skipped += 1
            continue

        game_id = int(game_id_raw)
        rows = grp.to_dict('records')

        # Determine home vs away from MATCHUP field
        # "XXX vs. YYY"  →  XXX is home
        # "XXX @ YYY"    →  XXX is away
        home_row = None
        away_row = None
        for r in rows:
            matchup = str(r.get('MATCHUP', ''))
            if 'vs.' in matchup:
                home_row = r
            elif '@' in matchup:
                away_row = r

        if home_row is None or away_row is None:
            # Fallback: try the other row if one was identified
            if home_row is not None and away_row is None:
                away_row = [r for r in rows if r is not home_row][0]
            elif away_row is not None and home_row is None:
                home_row = [r for r in rows if r is not away_row][0]
            else:
                skipped += 1
                continue

        # Resolve team metadata
        home_nba_id = int(home_row['TEAM_ID'])
        away_nba_id = int(away_row['TEAM_ID'])

        home_compact_id = team_id_map.get(home_nba_id, home_nba_id % 30 + 1)
        away_compact_id = team_id_map.get(away_nba_id, away_nba_id % 30 + 1)

        home_info = team_meta.get(home_nba_id, {
            'abbreviation': str(home_row.get('TEAM_ABBREVIATION', '')),
            'full_name': str(home_row.get('TEAM_NAME', '')),
            'name': str(home_row.get('TEAM_NAME', '')).split()[-1],
            'city': ' '.join(str(home_row.get('TEAM_NAME', '')).split()[:-1]),
        })
        away_info = team_meta.get(away_nba_id, {
            'abbreviation': str(away_row.get('TEAM_ABBREVIATION', '')),
            'full_name': str(away_row.get('TEAM_NAME', '')),
            'name': str(away_row.get('TEAM_NAME', '')).split()[-1],
            'city': ' '.join(str(away_row.get('TEAM_NAME', '')).split()[:-1]),
        })

        game_date = str(home_row.get('GAME_DATE', ''))[:10]
        home_score = _safe_int(home_row.get('PTS', 0))
        away_score = _safe_int(away_row.get('PTS', 0))

        if not game_date or home_score == 0:
            skipped += 1
            continue

        game_dict = {
            'id':               game_id,
            'date':             game_date,
            'status':           'Final',
            'home_team': {
                'id':           home_compact_id,
                'abbreviation': home_info['abbreviation'],
                'full_name':    home_info['full_name'],
                'name':         home_info['name'],
                'city':         home_info['city'],
            },
            'visitor_team': {
                'id':           away_compact_id,
                'abbreviation': away_info['abbreviation'],
                'full_name':    away_info['full_name'],
                'name':         away_info['name'],
                'city':         away_info['city'],
            },
            'home_team_score':    home_score,
            'visitor_team_score': away_score,
        }
        games.append(game_dict)

    print(f"  Built {len(games):,} game dicts  ({skipped} skipped)")
    return games


# ---------------------------------------------------------------------------
# Load player box scores
# ---------------------------------------------------------------------------

def load_player_stats(game_ids: set[int],
                      seasons: list[str],
                      team_id_map: dict[int, int]) -> dict[int, list[dict]]:
    """
    Load player box scores from all 3 CSV parts and convert to the format
    expected by process_games_for_training().

    Only includes rows whose gameId is in *game_ids* and whose season is in
    *seasons*.

    Returns
    -------
    dict mapping game_id (int) → list of player stat dicts
    """
    season_set = set(seasons)
    print("\n[load_player_stats] Reading box score parts ...")

    parts = []
    for path in BOX_PARTS:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found – skipping")
            continue
        df = pd.read_csv(path)
        df = df[df['season_year'].isin(season_set)].copy()
        parts.append(df)
        print(f"  {Path(path).name}: {len(df):,} rows after season filter")

    if not parts:
        print("  No box score data found!")
        return {}

    box = pd.concat(parts, ignore_index=True)

    # Filter to game_ids we actually have team data for
    box = box[box['gameId'].isin(game_ids)].copy()
    print(f"  After game_id filter: {len(box):,} rows")

    # ------------------------------------------------------------------
    # Parse player names  (personName = "First Last")
    # ------------------------------------------------------------------
    def _split_name(full: str) -> tuple[str, str]:
        parts_n = str(full).strip().split(' ', 1)
        if len(parts_n) == 2:
            return parts_n[0], parts_n[1]
        return parts_n[0], ''

    # ------------------------------------------------------------------
    # Build player_stats_by_game
    # ------------------------------------------------------------------
    player_stats_by_game: dict[int, list[dict]] = defaultdict(list)

    for _, row in box.iterrows():
        game_id  = int(row['gameId'])
        nba_team_id = int(row['teamId'])
        compact_tid = team_id_map.get(nba_team_id, nba_team_id % 30 + 1)

        person_id   = int(row['personId'])
        full_name   = str(row.get('personName', ''))
        first, last = _split_name(full_name)
        position    = str(row.get('position', '')) if pd.notna(row.get('position')) else ''

        # minutes field: keep as string "MM:SS"; handle NaN
        raw_min = row.get('minutes', '')
        if pd.isna(raw_min) or str(raw_min).strip() == '':
            min_str = '0:00'
        else:
            min_str = str(raw_min).strip()

        stat_dict = {
            'player': {
                'id':         person_id,
                'first_name': first,
                'last_name':  last,
                'position':   position,
            },
            'team': {
                'id':           compact_tid,
                'abbreviation': str(row.get('teamTricode', '')),
            },
            'game': {
                'id': game_id,
            },
            'min':      min_str,
            'pts':      _safe_int(row.get('points')),
            'reb':      _safe_int(row.get('reboundsTotal')),
            'ast':      _safe_int(row.get('assists')),
            'stl':      _safe_int(row.get('steals')),
            'blk':      _safe_int(row.get('blocks')),
            'turnover': _safe_int(row.get('turnovers')),   # NOTE: 'turnover' not 'turnovers'
            'pf':       _safe_int(row.get('foulsPersonal')),
            'fgm':      _safe_int(row.get('fieldGoalsMade')),
            'fga':      _safe_int(row.get('fieldGoalsAttempted')),
            'fg3m':     _safe_int(row.get('threePointersMade')),
            'fg3a':     _safe_int(row.get('threePointersAttempted')),
            'ftm':      _safe_int(row.get('freeThrowsMade')),
            'fta':      _safe_int(row.get('freeThrowsAttempted')),
            'oreb':     _safe_int(row.get('reboundsOffensive')),
            'dreb':     _safe_int(row.get('reboundsDefensive')),
            'fg_pct':   _safe_float(row.get('fieldGoalsPercentage')),
            'fg3_pct':  _safe_float(row.get('threePointersPercentage')),
            'ft_pct':   _safe_float(row.get('freeThrowsPercentage')),
        }
        player_stats_by_game[game_id].append(stat_dict)

    print(f"  Built player stats for {len(player_stats_by_game):,} games")
    total_records = sum(len(v) for v in player_stats_by_game.values())
    print(f"  Total player-game records: {total_records:,}")
    return dict(player_stats_by_game)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train NBA prediction models from local CSV data.'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        type=int,
        default=[2021, 2022, 2023, 2024],
        metavar='YEAR',
        help=(
            'Start years of seasons to include. '
            'E.g. --seasons 2021 2022 2023 2024 includes 2021-22 through 2024-25. '
            'Default: 2021 2022 2023 2024'
        ),
    )
    parser.add_argument(
        '--use-optuna',
        action='store_true',
        default=False,
        help='Enable Optuna hyperparameter tuning (slow). Default: off.',
    )
    parser.add_argument(
        '--optuna-trials',
        type=int,
        default=50,
        help='Number of Optuna trials per model when --use-optuna is set. Default: 50.',
    )
    parser.add_argument(
        '--time-decay-halflife',
        type=int,
        default=180,
        help='Half-life in days for time-decay sample weighting. Default: 180.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    season_labels = [int_to_season_label(y) for y in args.seasons]
    print("=" * 65)
    print("  NBA Model Training from Local CSV Data")
    print("=" * 65)
    print(f"  Seasons   : {season_labels}")
    print(f"  Optuna    : {args.use_optuna}")
    print(f"  Root dir  : {ROOT}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Build lookup structures
    # ------------------------------------------------------------------
    print("\n[Step 1/5] Building team ID maps...")
    team_id_map = build_team_id_map()
    team_meta   = _build_team_metadata()
    print(f"  Team ID map covers {len(team_id_map)} NBA.com IDs")
    print(f"  Team metadata covers {len(team_meta)} teams")

    # ------------------------------------------------------------------
    # 2. Load team-level games
    # ------------------------------------------------------------------
    print("\n[Step 2/5] Loading team game data...")
    games = load_team_games(season_labels, team_id_map, team_meta)
    if not games:
        print("ERROR: No games loaded. Check CSV paths and season filters.")
        sys.exit(1)

    game_ids = {g['id'] for g in games}
    print(f"  Total unique games: {len(game_ids):,}")

    # ------------------------------------------------------------------
    # 3. Load player box scores
    # ------------------------------------------------------------------
    print("\n[Step 3/5] Loading player box scores...")
    player_stats_by_game = load_player_stats(game_ids, season_labels, team_id_map)

    games_with_player_stats = len(set(player_stats_by_game.keys()) & game_ids)
    games_without = len(game_ids) - games_with_player_stats
    print(f"  Games with player stats : {games_with_player_stats:,}")
    print(f"  Games without (team only): {games_without:,}")

    # ------------------------------------------------------------------
    # 4. Import and call process_games_for_training
    # ------------------------------------------------------------------
    print("\n[Step 4/5] Importing training pipeline...")
    try:
        from train_complete_balldontlie import (
            process_games_for_training,
            train_all_models,
            initialize_league_averages,
        )
        print("  ✓ Imported process_games_for_training, train_all_models")
    except ImportError as exc:
        print(f"  ERROR importing training pipeline: {exc}")
        print("  Make sure you're running from the NBA-BETS root directory")
        print("  and all dependencies are installed.")
        sys.exit(1)

    # Initialise dynamic league averages (improves imputation quality)
    print("  Initialising dynamic league averages tracker...")
    # Build lightweight game dicts for the tracker
    tracker_games = [
        {
            'game_date': g['date'],
            'home_score': g['home_team_score'],
            'away_score': g['visitor_team_score'],
        }
        for g in games
    ]
    initialize_league_averages(tracker_games)

    print(f"\n  Processing {len(games):,} games into training samples...")
    team_data, player_data = process_games_for_training(games, player_stats_by_game)
    print(f"\n  Team training samples   : {len(team_data):,}")
    print(f"  Player training samples : {len(player_data):,}")

    if not team_data:
        print("ERROR: No team training samples generated. "
              "Check that seasons overlap with the CSV data.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Train all models
    # ------------------------------------------------------------------
    print("\n[Step 5/5] Training models...")
    print(f"  use_optuna={args.use_optuna}, "
          f"optuna_trials={args.optuna_trials}, "
          f"time_decay_halflife={args.time_decay_halflife}")

    results = train_all_models(
        team_data=team_data,
        player_data=player_data,
        use_time_decay=True,
        time_decay_half_life=args.time_decay_halflife,
        use_ensemble_props=True,
        use_optuna=args.use_optuna,
        optuna_trials=args.optuna_trials,
        tune_team_models=False,
        team_tune_trials=args.optuna_trials,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Training Complete")
    print("=" * 65)
    if results:
        for model_name, metrics in results.items():
            if isinstance(metrics, dict):
                acc = metrics.get('accuracy', metrics.get('test_accuracy'))
                mae = metrics.get('mae', metrics.get('test_mae'))
                if acc is not None:
                    print(f"  {model_name:<30} accuracy={acc:.4f}")
                elif mae is not None:
                    print(f"  {model_name:<30} MAE={mae:.4f}")
                else:
                    print(f"  {model_name}")
    print(f"\n  Models saved to: {os.path.join(ROOT, 'models')}")
    print("=" * 65)


if __name__ == '__main__':
    main()
