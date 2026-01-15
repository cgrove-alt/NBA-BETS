"""
Fetch Backtest Data from Balldontlie API

This script fetches all necessary data for running comprehensive_backtest.py:
1. Games for 2025-26 season (completed games only)
2. Player stats (box scores) for each game
3. Caches data in data/balldontlie_cache/ directory

Usage:
    python3 fetch_backtest_data.py
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict

from balldontlie_api import BalldontlieAPI

# Directories
CACHE_DIR = Path("data/balldontlie_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Season configuration
SEASON = 2025  # 2025-26 season
SEASON_START = "2025-10-21"  # Season started Oct 21, 2025
SEASON_END = "2026-04-13"    # Regular season ends Apr 13, 2026

def fetch_season_games(api: BalldontlieAPI):
    """Fetch all games for the 2025-26 season."""
    print("="*70)
    print("FETCHING SEASON GAMES")
    print("="*70)

    print(f"\nFetching games for season {SEASON}...")
    print(f"  Start date: {SEASON_START}")
    print(f"  End date: {SEASON_END}")

    try:
        # Use get_games with season parameter
        all_games = api.get_games(seasons=[SEASON], per_page=100)

        print(f"  Got {len(all_games)} games")

    except Exception as e:
        print(f"  Error fetching games: {e}")
        return []

    # Filter to completed games only
    completed_games = [g for g in all_games if g.get("status") == "Final"]

    # Sort by date
    completed_games.sort(key=lambda g: g.get("date", ""))

    print(f"\n✓ Fetched {len(completed_games)} completed games")
    if completed_games:
        print(f"  Date range: {completed_games[0]['date']} to {completed_games[-1]['date']}")

    # Save to cache
    output_file = CACHE_DIR / f"games_{SEASON}_full.json"
    with open(output_file, "w") as f:
        json.dump({"games": completed_games}, f, indent=2)

    print(f"  Saved to: {output_file}")

    return completed_games

def fetch_player_stats_for_games(api: BalldontlieAPI, games: List[Dict]):
    """Fetch player stats (box scores) for all games."""
    print("\n" + "="*70)
    print("FETCHING PLAYER STATS")
    print("="*70)

    # Group games into batches
    game_ids = [g["id"] for g in games]
    batch_size = 25  # Fetch 25 games at a time

    all_stats = {}  # game_id -> list of player stats
    stats_count = 0

    for i in range(0, len(game_ids), batch_size):
        batch = game_ids[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(game_ids) + batch_size - 1) // batch_size

        print(f"\nBatch {batch_num}/{total_batches} (games {i+1}-{min(i+batch_size, len(game_ids))})")

        try:
            # Fetch stats for this batch
            stats = api.get_player_stats(game_ids=batch)

            # Group by game_id
            for stat in stats:
                game = stat.get("game", {})
                game_id = game.get("id")
                if game_id:
                    if game_id not in all_stats:
                        all_stats[game_id] = []
                    all_stats[game_id].append(stat)
                    stats_count += 1

            print(f"  Got {len(stats)} player stat records")

            # Save batch to cache
            batch_file = CACHE_DIR / f"player_stats_batch_{batch_num}.json"
            with open(batch_file, "w") as f:
                # Convert game_id keys to strings for JSON
                batch_data = {str(gid): all_stats[gid] for gid in batch if gid in all_stats}
                json.dump(batch_data, f, indent=2)

            time.sleep(1.0)  # Rate limiting

        except Exception as e:
            print(f"  Error fetching stats: {e}")
            continue

    print(f"\n✓ Fetched {stats_count} player stat records across {len(all_stats)} games")

    return all_stats

def fetch_historical_stats(api: BalldontlieAPI):
    """Fetch historical season stats for context."""
    print("\n" + "="*70)
    print("FETCHING HISTORICAL STATS")
    print("="*70)

    # Fetch stats for previous seasons to build player history
    historical_seasons = [2024, 2023]  # Last 2 seasons

    for season in historical_seasons:
        print(f"\nFetching season {season}...")

        output_file = CACHE_DIR / f"stats_{season}.json"

        # Skip if already exists
        if output_file.exists():
            print(f"  Already cached: {output_file}")
            continue

        try:
            # Use get_player_stats_paginated for large datasets
            print(f"  Fetching paginated data...")
            all_stats = api.get_player_stats_paginated(seasons=[season], per_page=100)

            print(f"  Got {len(all_stats)} records")

            # Save
            with open(output_file, "w") as f:
                json.dump(all_stats, f)

            print(f"  ✓ Saved to {output_file}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

def verify_data_quality(games: List[Dict], stats: Dict[int, List]):
    """Verify we have sufficient data for backtesting."""
    print("\n" + "="*70)
    print("DATA QUALITY VERIFICATION")
    print("="*70)

    # Check games
    print(f"\nGames:")
    print(f"  Total: {len(games)}")
    print(f"  Date range: {games[0]['date']} to {games[-1]['date']}")

    # Check coverage
    games_with_stats = sum(1 for g in games if g["id"] in stats)
    coverage = games_with_stats / len(games) * 100 if games else 0

    print(f"\nPlayer Stats Coverage:")
    print(f"  Games with stats: {games_with_stats}/{len(games)} ({coverage:.1f}%)")

    # Check average players per game
    total_players = sum(len(stats.get(g["id"], [])) for g in games)
    avg_players = total_players / len(games) if games else 0

    print(f"  Total player records: {total_players}")
    print(f"  Average players/game: {avg_players:.1f}")

    # Quality assessment
    print("\n" + "="*70)
    if coverage >= 95 and avg_players >= 20:
        print("✅ DATA QUALITY: EXCELLENT")
        print("   Ready for comprehensive backtest!")
    elif coverage >= 80 and avg_players >= 15:
        print("⚠️  DATA QUALITY: GOOD")
        print("   Some gaps but sufficient for backtest")
    else:
        print("❌ DATA QUALITY: INSUFFICIENT")
        print("   May need to re-fetch data")

    return coverage >= 80

def main():
    """Main data fetching routine."""
    print("="*70)
    print("BACKTEST DATA FETCHER")
    print("="*70)
    print(f"\nSeason: {SEASON}-{SEASON+1}")
    print(f"Cache directory: {CACHE_DIR}")

    # Initialize API
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if not api_key:
        print("\n❌ ERROR: BALLDONTLIE_API_KEY not found in environment")
        print("   Set it in .env file or export it")
        return

    print(f"API Key: {api_key[:8]}... (GOAT tier)")

    api = BalldontlieAPI(api_key=api_key)

    # Step 1: Fetch season games
    games = fetch_season_games(api)

    if not games:
        print("\n❌ No games fetched. Check API connection.")
        return

    # Step 2: Fetch player stats for games
    stats = fetch_player_stats_for_games(api, games)

    # Step 3: Fetch historical data (optional but recommended)
    try:
        fetch_historical_stats(api)
    except Exception as e:
        print(f"\n⚠️  Warning: Could not fetch historical stats: {e}")
        print("   Backtest will still work but with less context")

    # Step 4: Verify data quality
    quality_ok = verify_data_quality(games, stats)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ Games fetched: {len(games)}")
    print(f"✓ Player stats fetched: {sum(len(s) for s in stats.values())}")
    print(f"✓ Cache directory: {CACHE_DIR}")

    if quality_ok:
        print("\n🎯 READY FOR BACKTEST!")
        print("   Run: python3 comprehensive_backtest.py")
    else:
        print("\n⚠️  Data quality issues detected")
        print("   Review errors above and re-run if needed")

if __name__ == "__main__":
    main()
