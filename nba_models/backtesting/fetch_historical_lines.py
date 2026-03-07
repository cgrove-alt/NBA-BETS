#!/usr/bin/env python3
"""Fetch historical player prop and game lines from The Odds API.

Downloads and caches sportsbook player prop lines plus core game markets
(moneyline/spread/totals) for NBA games, used for profitability backtesting
and train-time market-context features with real market data.

The Odds API provides historical snapshots at 5-minute intervals.
For each game, we fetch the snapshot closest to tipoff minus 1 hour
to simulate what a bettor would have seen pre-game.

API Cost:
- Props only: ~41 credits/game (1 for events + 40 for 4 prop markets)
- Game markets add a small extra per-snapshot cost depending on plan
  and regions/markets requested.

Usage:
    export THE_ODDS_API_KEY=<key>
    python nba_models/backtesting/fetch_historical_lines.py --season 2024-25 --max-games 5
    python nba_models/backtesting/fetch_historical_lines.py --season 2024-25 --resume
    python nba_models/backtesting/fetch_historical_lines.py --season 2024-25 --include-game-markets
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
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "historical_lines"
LIVE_SEASONS_DIR = ROOT / "data" / "live_seasons"

# Add project root to path for shared imports
sys.path.insert(0, str(ROOT / "nba_data" / "sources"))

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
REGION = "us"
GAME_MARKETS = "h2h,spreads,totals"

# Import shared constants from odds_fetcher (single source of truth)
try:
    from odds_fetcher import FULL_NAME_TO_ABBREV, MARKET_TO_PROP, PLAYER_PROP_MARKETS
    MARKETS = ",".join(PLAYER_PROP_MARKETS.values())
except ImportError:
    logger.warning("Could not import from odds_fetcher, using local constants")
    MARKETS = "player_points,player_rebounds,player_assists,player_points_rebounds_assists"
    FULL_NAME_TO_ABBREV = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
        "LA Clippers": "LAC", "LA Lakers": "LAL",
        "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    }
    MARKET_TO_PROP = {
        "player_points": "points", "player_rebounds": "rebounds",
        "player_assists": "assists", "player_points_rebounds_assists": "pra",
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


def american_to_implied_probability(odds: int | float | None) -> float | None:
    """Convert American odds to implied probability."""
    if odds in (None, 0):
        return None
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _median_or_none(values: list[float | int | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(np.median(valid))


def summarize_game_market_snapshot(raw: dict) -> dict:
    """Summarize a historical game-market API response across books.

    Produces consensus-style values for spread, moneyline, and totals so the
    resulting archive is compact and directly usable by train-time loaders.
    """
    bookmakers = raw.get("data", {}).get("bookmakers", [])
    home_team = raw.get("data", {}).get("home_team")

    spread_home_lines: list[float] = []
    spread_away_lines: list[float] = []
    spread_home_odds: list[float] = []
    spread_away_odds: list[float] = []
    ml_home_odds: list[float] = []
    ml_away_odds: list[float] = []
    totals_lines: list[float] = []
    totals_over_odds: list[float] = []
    totals_under_odds: list[float] = []

    for book in bookmakers:
        for market in book.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])

            if key == "h2h" and len(outcomes) >= 2:
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        ml_home_odds.append(outcome.get("price"))
                    else:
                        ml_away_odds.append(outcome.get("price"))

            elif key == "spreads" and len(outcomes) >= 2:
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        spread_home_lines.append(outcome.get("point"))
                        spread_home_odds.append(outcome.get("price"))
                    else:
                        spread_away_lines.append(outcome.get("point"))
                        spread_away_odds.append(outcome.get("price"))

            elif key == "totals" and len(outcomes) >= 2:
                totals_lines.extend(
                    [outcome.get("point") for outcome in outcomes if outcome.get("point") is not None]
                )
                for outcome in outcomes:
                    if outcome.get("name") == "Over":
                        totals_over_odds.append(outcome.get("price"))
                    elif outcome.get("name") == "Under":
                        totals_under_odds.append(outcome.get("price"))

    return {
        "book_count": len(bookmakers),
        "spread": {
            "home_line": _median_or_none(spread_home_lines),
            "away_line": _median_or_none(spread_away_lines),
            "home_odds": _median_or_none(spread_home_odds),
            "away_odds": _median_or_none(spread_away_odds),
        },
        "moneyline": {
            "home_odds": _median_or_none(ml_home_odds),
            "away_odds": _median_or_none(ml_away_odds),
        },
        "totals": {
            "line": _median_or_none(totals_lines),
            "over_odds": _median_or_none(totals_over_odds),
            "under_odds": _median_or_none(totals_under_odds),
        },
    }


def fetch_game_markets(
    api_key: str,
    event_id: str,
    snapshot_time: str,
    session: requests.Session,
) -> tuple[dict, dict]:
    """Fetch historical game markets (moneyline/spread/totals) for one event."""
    url = f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "date": snapshot_time,
        "regions": REGION,
        "markets": GAME_MARKETS,
        "oddsFormat": "american",
    }
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    summary = summarize_game_market_snapshot(raw)
    summary["snapshot_timestamp"] = snapshot_time
    return summary, dict(resp.headers)


def derive_game_market_history(
    opening: dict | None,
    pregame: dict | None,
    closing: dict | None,
) -> dict:
    """Derive train-time market features from archived game-market snapshots."""

    first = opening or pregame or closing or {}
    last = closing or pregame or opening or {}

    open_spread = (first.get("spread") or {}).get("home_line")
    close_spread = (last.get("spread") or {}).get("home_line")
    open_ml = (first.get("moneyline") or {}).get("home_odds")
    close_ml = (last.get("moneyline") or {}).get("home_odds")

    line_movement = 0.0
    if open_spread is not None and close_spread is not None:
        line_movement = float(close_spread - open_spread)

    open_prob = american_to_implied_probability(open_ml)
    close_prob = american_to_implied_probability(close_ml)
    moneyline_home_prob_movement = 0.0
    if open_prob is not None and close_prob is not None:
        moneyline_home_prob_movement = float(close_prob - open_prob)

    pregame_spread = (pregame or {}).get("spread", {})
    closing_spread = (closing or pregame or {}).get("spread", {})
    recent_movement = 0.0
    if pregame and closing:
        pregame_line = pregame_spread.get("home_line")
        closing_line = closing_spread.get("home_line")
        if pregame_line is not None and closing_line is not None:
            recent_movement = float(closing_line - pregame_line)

    consensus_odds = closing_spread.get("home_odds")
    if consensus_odds is None:
        consensus_odds = (pregame or {}).get("spread", {}).get("home_odds")
    if consensus_odds is None:
        consensus_odds = -110

    return {
        "opening_line": open_spread if open_spread is not None else 0.0,
        "closing_line": close_spread if close_spread is not None else 0.0,
        "line_movement": line_movement,
        "moneyline_home_prob_movement": moneyline_home_prob_movement,
        "consensus_odds": consensus_odds,
        "rlm_flag": bool(abs(line_movement) >= 2.0 or abs(moneyline_home_prob_movement) >= 0.05),
        "steam_move_flag": bool(abs(recent_movement) >= 1.5),
    }


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


def get_snapshot_time(
    commence_time: str,
    offset: timedelta | None = None,
) -> str:
    """Calculate a historical snapshot time relative to tipoff.

    Args:
        commence_time: ISO timestamp of game start.
        offset: Amount of time before tipoff to request. Defaults to 1 hour.

    Returns:
        ISO timestamp before game start.
    """
    offset = offset or timedelta(hours=1)
    try:
        tip = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        # Fallback: use midnight UTC of the date
        return commence_time
    snapshot = tip - offset
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
    include_game_markets: bool = False,
) -> dict:
    """Fetch historical player prop lines for an entire season.

    Args:
        api_key: The Odds API key.
        season: Season label (e.g., '2024-25').
        max_games: If > 0, stop after this many games (for testing).
        resume: If True, skip dates that already have cached files.
        include_game_markets: If True, also fetch historical spread/moneyline/totals
            snapshots (opening, pregame, closing proxies) for each game.

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
    total_game_market_snapshots = 0
    credits_remaining = None

    for date_idx, date_str in enumerate(unique_dates):
        cache_file = CACHE_DIR / f"{date_str}.json"
        cached = None
        cached_by_event: dict[str, dict] = {}
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
            for game in cached.get("games", []):
                event_key = str(game.get("odds_api_event_id") or game.get("bdl_game_id"))
                cached_by_event[event_key] = game

        if resume and cached:
            has_game_markets = all(
                isinstance(game.get("game_markets"), dict) and game["game_markets"].get("snapshots")
                for game in cached.get("games", [])
            )
            if not include_game_markets or has_game_markets:
                logger.info("Skipping %s (cached)", date_str)
                total_games_fetched += len(cached.get("games", []))
                total_props += sum(
                    len(g.get("player_props", [])) for g in cached.get("games", [])
                )
                total_game_market_snapshots += sum(
                    len((g.get("game_markets") or {}).get("snapshots", {}))
                    for g in cached.get("games", [])
                )
                continue
            logger.info("Augmenting %s with missing game markets", date_str)

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
        date_result = cached or {
            "date": date_str,
            "games": [],
            "api_credits_used": 1,
            "game_market_snapshots_fetched": 0,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        if "game_market_snapshots_fetched" not in date_result:
            date_result["game_market_snapshots_fetched"] = 0
        result_games_by_event = {
            str(game.get("odds_api_event_id") or game.get("bdl_game_id")): game
            for game in date_result.get("games", [])
        }

        for game in matched:
            event_key = str(game["odds_api_event_id"])
            if max_games > 0 and event_key not in result_games_by_event:
                if total_games_fetched + len(result_games_by_event) >= max_games:
                    break

            game_entry = result_games_by_event.get(event_key, {}).copy()
            if not game_entry:
                game_entry = {
                    "bdl_game_id": game["bdl_game_id"],
                    "odds_api_event_id": game["odds_api_event_id"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "home_abbrev": game["home_abbrev"],
                    "away_abbrev": game["away_abbrev"],
                    "commence_time": game["commence_time"],
                }

            if not game_entry.get("player_props"):
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

                deduped = dedupe_props_best_book(props)
                game_entry["snapshot_timestamp"] = snapshot
                game_entry["player_props"] = deduped
                date_result["api_credits_used"] = date_result.get("api_credits_used", 0) + 40

            if include_game_markets:
                existing_markets = (game_entry.get("game_markets") or {}).get("snapshots", {})
                snapshots = dict(existing_markets)
                snapshot_offsets = {
                    "opening": timedelta(hours=24),
                    "pregame": timedelta(hours=1),
                    "closing": timedelta(minutes=15),
                }
                for label, offset in snapshot_offsets.items():
                    if label in snapshots and resume:
                        continue
                    snapshot_time = get_snapshot_time(game["commence_time"], offset=offset)
                    try:
                        snapshot_summary, market_headers = fetch_game_markets(
                            api_key, game["odds_api_event_id"], snapshot_time, session
                        )
                        snapshots[label] = snapshot_summary
                        date_result["game_market_snapshots_fetched"] += 1
                        total_game_market_snapshots += 1
                        credits_remaining = market_headers.get("x-requests-remaining", credits_remaining)
                    except requests.HTTPError as exc:
                        logger.warning(
                            "Failed to fetch %s game markets for %s vs %s: %s",
                            label,
                            game["home_abbrev"],
                            game["away_abbrev"],
                            exc,
                        )
                        continue
                    _time.sleep(0.5)

                if snapshots:
                    game_entry["game_markets"] = {
                        "snapshots": snapshots,
                        "derived": derive_game_market_history(
                            snapshots.get("opening"),
                            snapshots.get("pregame"),
                            snapshots.get("closing"),
                        ),
                    }

            result_games_by_event[event_key] = game_entry

            logger.info(
                "  %s vs %s: %d props (%d players)",
                game["home_abbrev"],
                game["away_abbrev"],
                len(game_entry.get("player_props", [])),
                len({p["player_name"] for p in game_entry.get("player_props", [])}),
            )

            _time.sleep(1)  # Rate limit between games

        date_result["games"] = list(result_games_by_event.values())
        total_games_fetched += len(date_result["games"])
        total_props += sum(len(g.get("player_props", [])) for g in date_result["games"])

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
        "total_game_market_snapshots": total_game_market_snapshots,
        "game_markets_included": include_game_markets,
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
    parser.add_argument(
        "--include-game-markets",
        action="store_true",
        help="Also fetch historical moneyline/spread/totals snapshots for each game",
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
        include_game_markets=args.include_game_markets,
    )

    print("\n=== Fetch Summary ===")
    print(f"Season:           {summary['season']}")
    print(f"Games fetched:    {summary['total_games_fetched']}")
    print(f"Total props:      {summary['total_props']}")
    print(f"Game market snapshots: {summary['total_game_market_snapshots']}")
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
