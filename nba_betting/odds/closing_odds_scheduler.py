#!/usr/bin/env python3
from __future__ import annotations

r"""
Closing Odds Scheduler - Captures closing lines before game start.

This script monitors today's NBA games and captures closing odds ~5 minutes
before each game starts. Closing Line Value (CLV) is calculated by comparing
the odds when you placed your bet vs the closing odds.

Positive CLV over time = you're consistently getting better odds than the market
settles on, which is the best indicator of a sharp bettor.

Usage:
    python3 closing_odds_scheduler.py              # Daemon mode - runs continuously
    python3 closing_odds_scheduler.py --once       # Single check mode
    python3 closing_odds_scheduler.py --test       # Test mode - print upcoming games

Schedule via cron (recommended):
    # Run every 5 minutes from 6 PM to 11 PM ET on game days
    */5 18-23 * * * cd /path/to/NBA\ Betting\ Model && python3 closing_odds_scheduler.py --once
"""

import os
import sys
import time
import argparse
from datetime import datetime

import load_env  # noqa: F401  — load .env before any code reads os.environ

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from odds_fetcher import OddsFetcher, LineMovementTracker
    HAS_ODDS_FETCHER = True
except ImportError:
    HAS_ODDS_FETCHER = False
    print("Error: odds_fetcher.py not found")

try:
    from balldontlie_api import BalldontlieAPI
    HAS_BALLDONTLIE = True
except ImportError:
    HAS_BALLDONTLIE = False


def parse_game_time(status_str: str) -> datetime | None:
    """
    Parse game time from various status string formats.

    Args:
        status_str: Game status string (ISO format or "7:00 PM ET" format)

    Returns:
        datetime object or None if parsing fails
    """
    if not status_str:
        return None

    try:
        # Try ISO format first (2025-01-05T19:00:00Z)
        if 'T' in status_str:
            # Remove timezone indicator for parsing
            clean = status_str.replace('Z', '').replace('+00:00', '')
            return datetime.fromisoformat(clean)

        # Try "7:00 PM ET" format
        if 'PM' in status_str or 'AM' in status_str:
            today = datetime.now().date()
            time_str = status_str.replace(' ET', '').replace(' EST', '').replace(' EDT', '')
            parsed_time = datetime.strptime(time_str, "%I:%M %p").time()
            return datetime.combine(today, parsed_time)

    except Exception:
        pass

    return None


def get_todays_games() -> list[dict]:
    """
    Fetch today's NBA games from available sources.

    Returns:
        List of game dictionaries with id, home_team, visitor_team, status
    """
    games = []

    # Try Balldontlie first (premium)
    if HAS_BALLDONTLIE:
        try:
            api = BalldontlieAPI()
            today = datetime.now().strftime("%Y-%m-%d")
            bdl_games = api.get_games(dates=[today])

            for g in bdl_games:
                games.append({
                    'id': str(g.get('id', '')),
                    'home_team': g.get('home_team', {}).get('abbreviation', ''),
                    'away_team': g.get('visitor_team', {}).get('abbreviation', ''),
                    'status': g.get('status', ''),
                    'source': 'balldontlie'
                })

            if games:
                return games

        except Exception as e:
            print(f"Balldontlie error: {e}")

    # Fallback: try loading from cached schedule (if available)
    try:
        import json
        today = datetime.now().strftime("%Y-%m-%d")
        cache_file = f"cache/schedule_{today}.json"
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                games = json.load(f)
    except Exception:
        pass

    return games


def capture_closing_odds(
    game: dict,
    line_tracker: LineMovementTracker,
    odds_fetcher: OddsFetcher,
) -> bool:
    """
    Capture closing odds for a single game.

    Args:
        game: Game dictionary with id, home_team, away_team
        line_tracker: LineMovementTracker instance
        odds_fetcher: OddsFetcher instance

    Returns:
        True if closing odds were captured successfully
    """
    game_id = game.get('id', '')
    home_team = game.get('home_team', '')
    away_team = game.get('away_team', '')

    if not game_id or not home_team or not away_team:
        return False

    # Check if we already have closing odds for this game
    existing_closing = line_tracker.get_closing_odds(game_id)
    if existing_closing:
        print(f"  Already have closing odds for {away_team}@{home_team}")
        return True

    # Fetch current odds
    try:
        # Try Balldontlie first
        if HAS_BALLDONTLIE:
            api = BalldontlieAPI()
            odds_data = api.get_betting_odds(game_id=int(game_id))

            if odds_data and len(odds_data) > 0:
                odds = odds_data[0]
                formatted_odds = {
                    "timestamp": datetime.now().isoformat(),
                    "home_team": home_team,
                    "away_team": away_team,
                    "sportsbook": odds.get('sportsbook', 'Unknown'),
                    "moneyline": {
                        "home": odds.get('moneyline_home_odds', -110),
                        "away": odds.get('moneyline_away_odds', -110),
                    },
                    "spread": {
                        "home_line": odds.get('spread_home_line', 0),
                        "home_odds": odds.get('spread_home_odds', -110),
                        "away_line": odds.get('spread_away_line', 0),
                        "away_odds": odds.get('spread_away_odds', -110),
                    },
                    "total": {
                        "line": odds.get('total_line', 220),
                        "over_odds": odds.get('over_odds', -110),
                        "under_odds": odds.get('under_odds', -110),
                    },
                }

                # Record as closing odds
                line_tracker.record_odds_snapshot(
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    odds_data=formatted_odds,
                    is_closing=True
                )

                # Save to disk
                line_tracker.save_history(game_id)

                # Phase 4: Update CLV bridge with closing odds
                try:
                    from nba_betting.edge.clv_bridge import update_closing_odds as _update_clv
                    # Update any tracked prop bets for this game
                    # Prop closing odds default to the spread odds as approximation
                    _update_clv(game_id, formatted_odds.get('spread', {}).get('home_odds', -110))
                except ImportError:
                    pass
                except Exception:
                    pass

                print(f"  Captured closing odds for {away_team}@{home_team}")
                return True

        # Fallback to The Odds API
        if odds_fetcher:
            all_odds = odds_fetcher.get_nba_odds()

            # Find this game's odds
            for game_odds in all_odds:
                if (home_team.lower() in game_odds.get('home_team', '').lower() or
                    away_team.lower() in game_odds.get('away_team', '').lower()):

                    formatted_odds = {
                        "timestamp": datetime.now().isoformat(),
                        "home_team": home_team,
                        "away_team": away_team,
                        "sportsbook": game_odds.get('bookmaker', 'TheOddsAPI'),
                        "moneyline": {
                            "home": game_odds.get('home_odds', -110),
                            "away": game_odds.get('away_odds', -110),
                        },
                        "spread": {
                            "home_line": game_odds.get('home_spread', 0),
                            "home_odds": game_odds.get('home_spread_odds', -110),
                            "away_line": game_odds.get('away_spread', 0),
                            "away_odds": game_odds.get('away_spread_odds', -110),
                        },
                        "total": {
                            "line": game_odds.get('total', 220),
                            "over_odds": game_odds.get('over_odds', -110),
                            "under_odds": game_odds.get('under_odds', -110),
                        },
                    }

                    line_tracker.record_odds_snapshot(
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        odds_data=formatted_odds,
                        is_closing=True
                    )

                    line_tracker.save_history(game_id)
                    print(f"  Captured closing odds for {away_team}@{home_team} (TheOddsAPI)")
                    return True

    except Exception as e:
        print(f"  Error capturing odds for {away_team}@{home_team}: {e}")

    return False


def check_and_capture_closing_odds(
    minutes_before: int = 10,
    verbose: bool = True
) -> dict:
    """
    Check all games and capture closing odds for games starting soon.

    Args:
        minutes_before: Minutes before game to capture closing odds
        verbose: Print status messages

    Returns:
        Summary dictionary with counts
    """
    result = {
        "checked": 0,
        "captured": 0,
        "already_had": 0,
        "not_yet": 0,
        "errors": 0,
    }

    if not HAS_ODDS_FETCHER:
        print("Error: odds_fetcher module not available")
        return result

    # Initialize trackers
    line_tracker = LineMovementTracker(storage_dir="odds_history")
    odds_fetcher = OddsFetcher()

    # Get today's games
    games = get_todays_games()

    if not games:
        if verbose:
            print("No games found for today")
        return result

    if verbose:
        print(f"Checking {len(games)} games for closing odds capture...")

    now = datetime.now()

    for game in games:
        result["checked"] += 1

        game_id = game.get('id', '')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        status = game.get('status', '')

        # Parse game time
        game_start = parse_game_time(status)

        if not game_start:
            if verbose:
                print(f"  Could not parse time for {away_team}@{home_team}: {status}")
            result["errors"] += 1
            continue

        # Calculate time until game
        time_until_start = (game_start - now).total_seconds() / 60

        if verbose:
            print(f"  {away_team}@{home_team}: {time_until_start:.1f} min until start")

        # Only capture if game starts within the window
        if 0 < time_until_start <= minutes_before:
            # Check if we already have closing odds
            existing = line_tracker.get_closing_odds(game_id)
            if existing:
                result["already_had"] += 1
                if verbose:
                    print("    Already have closing odds")
            else:
                # Capture closing odds
                success = capture_closing_odds(game, line_tracker, odds_fetcher)
                if success:
                    result["captured"] += 1
                else:
                    result["errors"] += 1
        elif time_until_start > minutes_before:
            result["not_yet"] += 1
            if verbose:
                print(f"    Not yet (game in {time_until_start:.0f} min)")
        else:
            if verbose:
                print("    Game already started or finished")

    return result


def daemon_mode(
    check_interval: int = 300,  # 5 minutes
    minutes_before: int = 10,
):
    """
    Run in daemon mode - continuously check for games starting soon.

    Args:
        check_interval: Seconds between checks
        minutes_before: Minutes before game to capture closing odds
    """
    print(f"Starting closing odds daemon (checking every {check_interval}s)")
    print(f"Will capture closing odds {minutes_before} minutes before game start")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking games...")
            result = check_and_capture_closing_odds(
                minutes_before=minutes_before,
                verbose=True
            )

            print(f"\nSummary: {result['captured']} captured, {result['already_had']} already had, {result['not_yet']} not yet")
            print(f"Sleeping {check_interval}s until next check...")

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\nStopping daemon...")


def main():
    parser = argparse.ArgumentParser(
        description="Capture closing odds before NBA games start"
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help="Run once and exit (for cron jobs)"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help="Test mode - just print today's games"
    )
    parser.add_argument(
        '--minutes',
        type=int,
        default=10,
        help="Minutes before game to capture closing odds (default: 10)"
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help="Seconds between checks in daemon mode (default: 300)"
    )

    args = parser.parse_args()

    if args.test:
        print("Test mode - fetching today's games...\n")
        games = get_todays_games()

        if not games:
            print("No games found")
            return

        print(f"Found {len(games)} games:\n")
        for g in games:
            print(f"  {g.get('away_team')}@{g.get('home_team')} - {g.get('status')}")
        return

    if args.once:
        # Single check mode (for cron)
        result = check_and_capture_closing_odds(
            minutes_before=args.minutes,
            verbose=True
        )
        print(f"\nResult: {result}")
    else:
        # Daemon mode
        daemon_mode(
            check_interval=args.interval,
            minutes_before=args.minutes
        )


if __name__ == "__main__":
    main()
