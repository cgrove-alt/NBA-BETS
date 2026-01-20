import os
import time
from balldontlie_api import BalldontlieAPI

def test_api():
    print("Initializing BalldontlieAPI...")
    api = BalldontlieAPI()
    
    print("Fetching today's games to find a valid Game ID...")
    games = api.get_todays_games()
    
    if not games:
        print("No games found today. Trying recent games...")
        # Try to find a recent date
        from datetime import datetime, timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        games = api.get_games(dates=[yesterday])
        
    if not games:
        print("ERROR: Could not find any games to test.")
        return

    game = games[0]
    game_id = game['id']
    print(f"Testing with Game ID: {game_id} ({game['visitor_team']['abbreviation']} @ {game['home_team']['abbreviation']})")
    
    print("\n--- Testing get_player_props ---")
    start_time = time.time()
    props = api.get_player_props(game_id)
    duration = time.time() - start_time
    
    print(f"Request took {duration:.2f} seconds")
    print(f"Props returned: {len(props)}")
    
    if props:
        print("Sample Prop:", props[0])
        
        # Check for DraftKings specifically as logic depends on it
        dk_props = [p for p in props if 'draftkings' in str(p.get('sportsbook', '')).lower() or 'draftkings' in str(p.get('vendor', '')).lower()]
        print(f"DraftKings Props found: {len(dk_props)}")
        
        if len(dk_props) > 0:
            print("Sample DK Prop:", dk_props[0])
    else:
        print("WARNING: API returned empty list. This explains why the app has no data.")
        print("Possible reasons:")
        print("1. Game has no props yet (too early/late)")
        print("2. API Key doesn't have GOAT tier access to 'odds/player_props'")
        print("3. API endpoint changed")

if __name__ == "__main__":
    test_api()
