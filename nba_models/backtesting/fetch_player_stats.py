#!/usr/bin/env python3
"""Fetch and cache player box scores from BallDontLie API for backtesting.

Downloads player-level stats (points, rebounds, assists, etc.) for all games
in a season. Cached locally to avoid re-fetching.

The BDL /stats endpoint returns paginated player-game records.
We fetch all pages for the season and index by game_id for fast lookup.

Saves two files:
  - player_stats_{season}.json: game_id → [player stat dicts]
  - player_stats_{season}_meta.json: game_id → {date, home/away teams, scores}

Usage:
    export BALLDONTLIE_API_KEY=<key>
    python nba_models/backtesting/fetch_player_stats.py --season 2024
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as _time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "historical_lines"

logger = logging.getLogger(__name__)

BDL_BASE = "https://api.balldontlie.io/v1"


def fetch_season_stats(
    api_key: str, season: int
) -> tuple[dict[int, list[dict]], dict[int, dict]]:
    """Fetch all player box scores for a season from BallDontLie.

    Args:
        api_key: BallDontLie API key (GOAT tier).
        season: Season start year (e.g., 2024 for 2024-25).

    Returns:
        (player_stats_by_game, game_metadata) where:
          - player_stats_by_game: game_id → list of player stat dicts
          - game_metadata: game_id → {date, home_team_id, away_team_id, ...}
    """
    session = requests.Session()
    session.headers["Authorization"] = api_key

    all_stats: dict[int, list[dict]] = {}
    game_meta: dict[int, dict] = {}
    cursor = None
    page = 0
    total_records = 0

    while True:
        page += 1
        params: dict = {"seasons[]": season, "per_page": 100}
        if cursor:
            params["cursor"] = cursor

        resp = session.get(f"{BDL_BASE}/stats", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("data", [])
        if not records:
            break

        for stat in records:
            game_info = stat.get("game", {})
            game_id = game_info.get("id")
            if not game_id:
                continue

            # Save game metadata (once per game)
            if game_id not in game_meta:
                game_meta[game_id] = {
                    "date": game_info.get("date", ""),
                    "season": game_info.get("season"),
                    "home_team_id": game_info.get("home_team_id"),
                    "visitor_team_id": game_info.get("visitor_team_id"),
                    "home_team_score": game_info.get("home_team_score"),
                    "visitor_team_score": game_info.get("visitor_team_score"),
                    "status": game_info.get("status", ""),
                }

            player = stat.get("player", {})
            team = stat.get("team", {})
            entry = {
                "player_id": player.get("id"),
                "player_name": (
                    f"{player.get('first_name', '')} "
                    f"{player.get('last_name', '')}"
                ).strip(),
                "first_name": player.get("first_name", ""),
                "last_name": player.get("last_name", ""),
                "position": player.get("position", ""),
                "team_id": team.get("id"),
                "team_abbreviation": team.get("abbreviation", ""),
                "min": stat.get("min", "0"),
                "pts": stat.get("pts", 0),
                "reb": stat.get("reb", 0),
                "ast": stat.get("ast", 0),
                "stl": stat.get("stl", 0),
                "blk": stat.get("blk", 0),
                "turnover": stat.get("turnover", 0),
                "pf": stat.get("pf", 0),
                "fgm": stat.get("fgm", 0),
                "fga": stat.get("fga", 0),
                "fg3m": stat.get("fg3m", 0),
                "fg3a": stat.get("fg3a", 0),
                "ftm": stat.get("ftm", 0),
                "fta": stat.get("fta", 0),
                "oreb": stat.get("oreb", 0),
                "dreb": stat.get("dreb", 0),
                "fg_pct": stat.get("fg_pct", 0.0),
                "fg3_pct": stat.get("fg3_pct", 0.0),
                "ft_pct": stat.get("ft_pct", 0.0),
                "pra": (
                    (stat.get("pts", 0) or 0)
                    + (stat.get("reb", 0) or 0)
                    + (stat.get("ast", 0) or 0)
                ),
            }
            if game_id not in all_stats:
                all_stats[game_id] = []
            all_stats[game_id].append(entry)
            total_records += 1

        # Pagination
        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")

        if page % 50 == 0:
            logger.info(
                "  Page %d: %d records so far (%d games)",
                page,
                total_records,
                len(all_stats),
            )

        if not cursor:
            break

        _time.sleep(0.1)  # Rate limit

    logger.info(
        "Fetched %d player-game records across %d games",
        total_records,
        len(all_stats),
    )
    return all_stats, game_meta


def main() -> int:
    """Entry point for the player stats fetcher."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fetch player box scores from BDL API"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2024,
        help="Season start year (default: 2024)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if cache exists",
    )
    args = parser.parse_args()

    api_key = os.environ.get("BALLDONTLIE_API_KEY", "")
    if not api_key:
        print("ERROR: BALLDONTLIE_API_KEY not set.")
        return 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"player_stats_{args.season}.json"
    meta_path = CACHE_DIR / f"player_stats_{args.season}_meta.json"

    if cache_path.exists() and meta_path.exists() and not args.force:
        logger.info("Cache exists at %s — loading", cache_path)
        with open(cache_path) as f:
            stats = json.load(f)
        total = sum(len(v) for v in stats.values())
        print(f"Loaded {total} records across {len(stats)} games from cache")
        return 0

    logger.info(
        "Fetching season %d-%02d player stats...",
        args.season,
        (args.season + 1) % 100,
    )
    stats, game_meta = fetch_season_stats(api_key, args.season)

    # Save stats — convert int keys to strings for JSON
    serializable = {str(k): v for k, v in stats.items()}
    with open(cache_path, "w") as f:
        json.dump(serializable, f)
    logger.info("Saved stats to %s", cache_path)

    # Save game metadata
    meta_serializable = {str(k): v for k, v in game_meta.items()}
    with open(meta_path, "w") as f:
        json.dump(meta_serializable, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    total = sum(len(v) for v in stats.values())
    print(f"\nFetched {total} player-game records across {len(stats)} games")
    print(f"Game metadata for {len(game_meta)} games")
    return 0


if __name__ == "__main__":
    sys.exit(main())
