#!/usr/bin/env python3
"""Fetch historical player prop lines from The Odds API.

Downloads and caches sportsbook player prop lines for NBA games,
used for profitability backtesting with real market data.

The Odds API provides historical snapshots at 5-minute intervals.
For each game, we fetch the snapshot closest to tipoff minus 1 hour
to simulate what a bettor would have seen pre-game.

API Cost: ~41 credits per game (1 for events + 40 for 4 prop markets).
Full 2024-25 season (1,230 games): ~49,400 credits.

Usage:
    export THE_ODDS_API_KEY=<key>
    python nba_models/backtesting/fetch_historical_lines.py --season 2024-25 --max-games 5
    python nba_models/backtesting/fetch_historical_lines.py --season 2024-25 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "historical_lines"
LIVE_SEASONS_DIR = ROOT / "data" / "live_seasons"

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
MARKETS = "player_points,player_rebounds,player_assists,player_points_rebounds_assists"
REGION = "us"

# Mapping from The Odds API full team names to common abbreviations.
# The Odds API uses city+mascot; our CSV data uses 3-letter codes.
FULL_NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Market key → our prop type name
MARKET_TO_PROP = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_points_rebounds_assists": "pra",
}


def load_season_games(season: str) -> pd.DataFrame:
    """Load game data for a season from the live_seasons CSV.

    Args:
        season: Season label like '2024-25'.

    Returns:
        DataFrame with one row per unique game, including date, teams,
        game_id, and matchup info.
    """
    # Find the most recent live_seasons CSV
    csv_files = sorted(LIVE_SEASONS_DIR.glob("live_seasons_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No live_seasons CSV found in {LIVE_SEASONS_DIR}")

    df = pd.read_csv(csv_files[-1])
    season_df = df[df["SEASON_YEAR"] == season].copy()

    if season_df.empty:
        raise ValueError(f"No games found for season {season}")

    # Deduplicate: keep one row per game (the home team row)
    # Home games have "vs." in MATCHUP, away have "@"
    home_rows = season_df[season_df["MATCHUP"].str.contains(" vs. ")].copy()

    # Extract away team abbreviation from matchup
    home_rows["AWAY_ABBREV"] = home_rows["MATCHUP"].str.extract(r"vs\. (\w+)")

    # For games where home team row is missing, use away and flip
    seen_game_ids = set(home_rows["GAME_ID"])
    away_only = season_df[
        (~season_df["GAME_ID"].isin(seen_game_ids))
        & (season_df["MATCHUP"].str.contains(" @ "))
    ].copy()

    if not away_only.empty:
        away_only["AWAY_ABBREV"] = away_only["TEAM_ABBREVIATION"]
        away_only["TEAM_ABBREVIATION"] = away_only["MATCHUP"].str.extract(r"@ (\w+)")
        home_rows = pd.concat([home_rows, away_only], ignore_index=True)

    home_rows = home_rows.sort_values("GAME_DATE").reset_index(drop=True)
    logger.info(
        "Loaded %d games for %s (%s to %s)",
        len(home_rows),
        season,
        home_rows["GAME_DATE"].min(),
        home_rows["GAME_DATE"].max(),
    )
    return home_rows


def fetch_events_for_date(
    api_key: str, date_str: str, session: requests.Session
) -> tuple[list[dict], dict]:
    """Fetch historical event IDs from The Odds API for a specific date.

    Args:
        api_key: The Odds API key.
        date_str: Date in YYYY-MM-DD format.
        session: Requests session for connection reuse.

    Returns:
        (events_list, response_headers) where events_list contains event dicts
        with id, home_team, away_team, commence_time.
    """
    url = f"{ODDS_API_BASE}/historical/sports/{SPORT}/events"
    params = {
        "apiKey": api_key,
        "date": f"{date_str}T00:00:00Z",
    }
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", []), dict(resp.headers)


def fetch_player_props(
    api_key: str,
    event_id: str,
    snapshot_time: str,
    session: requests.Session,
) -> tuple[list[dict], dict]:
    """Fetch historical player prop odds for a specific event.

    Args:
        api_key: The Odds API key.
        event_id: The Odds API event ID.
        snapshot_time: ISO timestamp for historical snapshot (pre-game).
        session: Requests session.

    Returns:
        (player_props_list, response_headers) where each prop has player_name,
        prop_type, line, over_odds, under_odds, bookmaker.
    """
    url = f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "date": snapshot_time,
        "regions": REGION,
        "markets": MARKETS,
        "oddsFormat": "american",
    }
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    props = []
    bookmakers = raw.get("data", {}).get("bookmakers", [])
    for book in bookmakers:
        book_key = book["key"]
        for market in book.get("markets", []):
            prop_type = MARKET_TO_PROP.get(market["key"])
            if not prop_type:
                continue

            # Outcomes come in pairs: Over and Under for each player
            outcomes = market.get("outcomes", [])
            # Group by player name
            player_lines: dict[str, dict] = {}
            for outcome in outcomes:
                player = outcome.get("description", "")
                direction = outcome.get("name", "").lower()
                if not player or direction not in ("over", "under"):
                    continue

                if player not in player_lines:
                    player_lines[player] = {
                        "player_name": player,
                        "prop_type": prop_type,
                        "line": outcome.get("point", 0),
                        "bookmaker": book_key,
                    }
                if direction == "over":
                    player_lines[player]["over_odds"] = outcome.get("price", -110)
                    player_lines[player]["line"] = outcome.get("point", 0)
                else:
                    player_lines[player]["under_odds"] = outcome.get("price", -110)

            for pl in player_lines.values():
                if "over_odds" in pl and "under_odds" in pl:
                    props.append(pl)

    return props, dict(resp.headers)


def match_events_to_games(
    events: list[dict], games_on_date: pd.DataFrame
) -> list[dict]:
    """Match The Odds API events to our game records by team names.

    Args:
        events: Events from The Odds API with home_team/away_team full names.
        games_on_date: DataFrame of games on this date from CSV data.

    Returns:
        List of matched dicts with both game_id and event_id.
    """
    matched = []
    for event in events:
        home_abbrev = FULL_NAME_TO_ABBREV.get(event.get("home_team", ""))
        away_abbrev = FULL_NAME_TO_ABBREV.get(event.get("away_team", ""))

        if not home_abbrev or not away_abbrev:
            logger.debug(
                "Unknown team name: %s vs %s",
                event.get("home_team"),
                event.get("away_team"),
            )
            continue

        # Find matching game in our data
        match = games_on_date[
            (games_on_date["TEAM_ABBREVIATION"] == home_abbrev)
            & (games_on_date["AWAY_ABBREV"] == away_abbrev)
        ]

        if match.empty:
            # Try reverse (sometimes home/away is swapped in data)
            match = games_on_date[
                (games_on_date["TEAM_ABBREVIATION"] == away_abbrev)
                & (games_on_date["AWAY_ABBREV"] == home_abbrev)
            ]

        if not match.empty:
            row = match.iloc[0]
            matched.append(
                {
                    "bdl_game_id": int(row["GAME_ID"]),
                    "odds_api_event_id": event["id"],
                    "home_team": event["home_team"],
                    "away_team": event["away_team"],
                    "home_abbrev": home_abbrev,
                    "away_abbrev": away_abbrev,
                    "commence_time": event.get("commence_time", ""),
                }
            )
        else:
            logger.debug(
                "No CSV match for %s vs %s on this date",
                home_abbrev,
                away_abbrev,
            )

    return matched


def get_snapshot_time(commence_time: str) -> str:
    """Calculate the pre-game snapshot time (1 hour before tipoff).

    Args:
        commence_time: ISO timestamp of game start.

    Returns:
        ISO timestamp 1 hour before game start.
    """
    try:
        tip = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        # Fallback: use midnight UTC of the date
        return commence_time
    snapshot = tip - timedelta(hours=1)
    return snapshot.strftime("%Y-%m-%dT%H:%M:%SZ")


def dedupe_props_best_book(props: list[dict]) -> list[dict]:
    """Deduplicate props, keeping DraftKings > FanDuel > first available.

    When multiple bookmakers offer the same player+prop, keep the best
    (most liquid) book's line.

    Args:
        props: List of player prop dicts with bookmaker field.

    Returns:
        Deduplicated list with one entry per player+prop_type.
    """
    book_priority = {"draftkings": 0, "fanduel": 1, "williamhill_us": 2}
    best: dict[tuple[str, str], dict] = {}

    for p in props:
        key = (p["player_name"], p["prop_type"])
        existing = best.get(key)
        if existing is None:
            best[key] = p
        else:
            new_rank = book_priority.get(p["bookmaker"], 99)
            old_rank = book_priority.get(existing["bookmaker"], 99)
            if new_rank < old_rank:
                best[key] = p

    return list(best.values())


def fetch_season(
    api_key: str,
    season: str,
    max_games: int = 0,
    resume: bool = False,
) -> dict:
    """Fetch historical player prop lines for an entire season.

    Args:
        api_key: The Odds API key.
        season: Season label (e.g., '2024-25').
        max_games: If > 0, stop after this many games (for testing).
        resume: If True, skip dates that already have cached files.

    Returns:
        Summary dict with total games fetched, credits used, etc.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    games_df = load_season_games(season)

    unique_dates = sorted(games_df["GAME_DATE"].unique())
    logger.info("Season %s: %d unique game dates", season, len(unique_dates))

    session = requests.Session()
    total_credits = 0
    total_games_fetched = 0
    total_props = 0
    credits_remaining = None

    for date_idx, date_str in enumerate(unique_dates):
        cache_file = CACHE_DIR / f"{date_str}.json"

        if resume and cache_file.exists():
            logger.info("Skipping %s (cached)", date_str)
            with open(cache_file) as f:
                cached = json.load(f)
            total_games_fetched += len(cached.get("games", []))
            total_props += sum(
                len(g.get("player_props", [])) for g in cached.get("games", [])
            )
            continue

        if max_games > 0 and total_games_fetched >= max_games:
            logger.info("Reached max_games=%d, stopping", max_games)
            break

        games_on_date = games_df[games_df["GAME_DATE"] == date_str]
        logger.info(
            "[%d/%d] Fetching %s (%d games)...",
            date_idx + 1,
            len(unique_dates),
            date_str,
            len(games_on_date),
        )

        # Step 1: Get event IDs for this date
        try:
            events, headers = fetch_events_for_date(api_key, date_str, session)
            total_credits += 1
            credits_remaining = headers.get("x-requests-remaining")
        except requests.HTTPError as exc:
            logger.warning("Failed to fetch events for %s: %s", date_str, exc)
            _time.sleep(2)
            continue

        _time.sleep(0.5)

        # Step 2: Match events to our games
        matched = match_events_to_games(events, games_on_date)
        if not matched:
            logger.warning("No matched games for %s", date_str)
            # Save empty cache so resume skips this date
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "date": date_str,
                        "games": [],
                        "api_credits_used": 1,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    },
                    f,
                    indent=2,
                )
            continue

        # Step 3: Fetch player props for each matched game
        date_result = {
            "date": date_str,
            "games": [],
            "api_credits_used": 1,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        for game in matched:
            if max_games > 0 and total_games_fetched >= max_games:
                break

            snapshot = get_snapshot_time(game["commence_time"])

            try:
                props, prop_headers = fetch_player_props(
                    api_key, game["odds_api_event_id"], snapshot, session
                )
                total_credits += 40  # 4 markets × 10 credits
                credits_remaining = prop_headers.get("x-requests-remaining")
            except requests.HTTPError as exc:
                logger.warning(
                    "Failed to fetch props for %s vs %s: %s",
                    game["home_abbrev"],
                    game["away_abbrev"],
                    exc,
                )
                _time.sleep(2)
                continue

            # Deduplicate: keep best bookmaker per player+prop
            deduped = dedupe_props_best_book(props)

            game_entry = {
                "bdl_game_id": game["bdl_game_id"],
                "odds_api_event_id": game["odds_api_event_id"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_abbrev": game["home_abbrev"],
                "away_abbrev": game["away_abbrev"],
                "commence_time": game["commence_time"],
                "snapshot_timestamp": snapshot,
                "player_props": deduped,
            }
            date_result["games"].append(game_entry)
            date_result["api_credits_used"] += 40

            total_games_fetched += 1
            total_props += len(deduped)

            logger.info(
                "  %s vs %s: %d props (%d players)",
                game["home_abbrev"],
                game["away_abbrev"],
                len(deduped),
                len({p["player_name"] for p in deduped}),
            )

            _time.sleep(1)  # Rate limit between games

        # Save date cache
        with open(cache_file, "w") as f:
            json.dump(date_result, f, indent=2)

        logger.info(
            "  Credits used so far: %d | Remaining: %s",
            total_credits,
            credits_remaining,
        )

    summary = {
        "season": season,
        "total_games_fetched": total_games_fetched,
        "total_props": total_props,
        "total_credits_used": total_credits,
        "credits_remaining": credits_remaining,
        "dates_processed": len(unique_dates),
    }

    # Save summary
    summary_path = CACHE_DIR / f"fetch_summary_{season.replace('-', '_')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> int:
    """Entry point for the historical lines fetcher."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fetch historical player prop lines from The Odds API"
    )
    parser.add_argument(
        "--season", default="2024-25", help="Season to fetch (default: 2024-25)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Max games to fetch (0 = all). Use 5-10 for testing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip dates that already have cached files",
    )
    args = parser.parse_args()

    api_key = os.environ.get("THE_ODDS_API_KEY", "")
    if not api_key:
        print("ERROR: THE_ODDS_API_KEY not set. Run: export THE_ODDS_API_KEY=<key>")
        return 1

    summary = fetch_season(
        api_key=api_key,
        season=args.season,
        max_games=args.max_games,
        resume=args.resume,
    )

    print("\n=== Fetch Summary ===")
    print(f"Season:           {summary['season']}")
    print(f"Games fetched:    {summary['total_games_fetched']}")
    print(f"Total props:      {summary['total_props']}")
    print(f"Credits used:     {summary['total_credits_used']}")
    print(f"Credits remaining: {summary['credits_remaining']}")

    if summary["total_games_fetched"] > 0:
        avg_props = summary["total_props"] / summary["total_games_fetched"]
        est_full = 1230 * (summary["total_credits_used"] / summary["total_games_fetched"])
        print(f"\nAvg props/game:   {avg_props:.0f}")
        print(f"Est. full season: ~{est_full:,.0f} credits")

    return 0


if __name__ == "__main__":
    sys.exit(main())
