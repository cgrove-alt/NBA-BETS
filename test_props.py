import asyncio
import sys
import os
from datetime import datetime
from dashboard.data_service import DataService

# Mock the asyncio run to test data service directly
async def test_props_generation():
    print("Initializing DataService...")
    service = DataService()
    
    # Check if we can get games
    games = service.get_todays_games(force_refresh=True)
    print(f"Found {len(games)} games for today.")
    
    if not games:
        print("No games found. Cannot test props.")
        return

    # Pick the first game
    game = games[0]
    game_id = str(game['game_id'])
    home_team = game['home_team']['abbreviation']
    away_team = game['visitor_team']['abbreviation']
    
    print(f"Testing props for Game {game_id}: {away_team} @ {home_team}")
    
    # Start fetch
    service.start_player_props_fetch(game_id, home_team, away_team)
    
    # Wait a bit for async thread
    print("Waiting for props generation (5s)...")
    await asyncio.sleep(5)
    
    # Check status
    status = service.get_props_fetch_status(game_id)
    print("Status:", status.get('status'))
    
    home_props = status.get('home', [])
    away_props = status.get('away', [])
    print(f"Props found: Home={len(home_props)}, Away={len(away_props)}")
    
    if len(home_props) == 0 and len(away_props) == 0:
        print("FAILURE: No props generated.")
        print("Error message:", status.get('error'))
    else:
        print("SUCCESS: Props generated.")
        print("Sample:", home_props[0] if home_props else away_props[0])

if __name__ == "__main__":
    asyncio.run(test_props_generation())
