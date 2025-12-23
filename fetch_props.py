"""
Fetch and display player prop predictions for CLE @ CHI game.
"""
import sys
sys.path.insert(0, '.')

from dashboard.data_service import get_data_service

# Initialize data service
print("Initializing data service...")
ds = get_data_service()
print("Data service ready.\n")

# Get today's games
print("Fetching today's games...")
games = ds.get_todays_games()
print(f"Found {len(games)} games today.\n")

# Find CLE @ CHI game
target_game = None
for g in games:
    home = g.get('home_team', {}).get('abbreviation', '')
    away = g.get('visitor_team', {}).get('abbreviation', '')
    if 'CHI' in home and 'CLE' in away:
        target_game = g
        break
    elif 'CLE' in home and 'CHI' in away:
        target_game = g
        break

if not target_game:
    print("ERROR: CLE @ CHI game not found in today's games!")
    print("Available games:")
    for g in games:
        home = g.get('home_team', {}).get('abbreviation', '')
        away = g.get('visitor_team', {}).get('abbreviation', '')
        print(f"  {away} @ {home}")
    sys.exit(1)

home_abbrev = target_game.get('home_team', {}).get('abbreviation', '')
away_abbrev = target_game.get('visitor_team', {}).get('abbreviation', '')
# Use 'game_id' key which is set by data_service._format_balldontlie_games()
game_id = str(target_game.get('game_id', target_game.get('id', 'unknown')))

print(f"Found game: {away_abbrev} @ {home_abbrev} (ID: {game_id})\n")

# Start background fetch and wait for results
import time

print("Starting player props fetch...")
ds.start_player_props_fetch(game_id, home_abbrev, away_abbrev)

# Poll for completion
max_wait = 60  # seconds
start = time.time()
while time.time() - start < max_wait:
    status = ds.get_props_fetch_status(game_id)
    if status.get('status') == 'complete':
        break
    elif status.get('status') == 'error':
        print(f"ERROR: {status.get('error', 'Unknown error')}")
        sys.exit(1)
    print(f"  Status: {status.get('status', 'unknown')}...")
    time.sleep(2)

# Get results
status = ds.get_props_fetch_status(game_id)
print(f"\nFetch completed with status: {status.get('status')}\n")

home_props = status.get('home', [])
away_props = status.get('away', [])

print("=" * 70)
print(f"PLAYER PROP PREDICTIONS: {away_abbrev} @ {home_abbrev}")
print("=" * 70)

def print_team_props(props, team_name):
    print(f"\n{team_name} PLAYERS:")
    print("-" * 70)

    if not props:
        print("  No props available.")
        return

    # Props is a list of player dicts, each with predictions for all prop types
    prop_types = ["points", "rebounds", "assists", "3pm", "pra"]

    for player_data in props:
        name = player_data.get('player_name', 'Unknown')
        position = player_data.get('position', '')
        print(f"\n  {name} ({position}):")

        for pt in prop_types:
            pred = player_data.get(f'{pt}_pred', 0)
            line = player_data.get(f'{pt}_line', 0)
            pick = player_data.get(f'{pt}_pick', '-')
            confidence = player_data.get(f'{pt}_confidence', 0)
            edge = player_data.get(f'{pt}_edge', 0)

            if pred and pred > 0:
                pt_label = pt.upper() if pt != '3pm' else '3PM'
                print(f"    {pt_label:10s}: Pred {pred:5.1f}  Line {line:5.1f}  Pick: {pick:5s}  Confidence: {confidence:3.0f}%  Edge: {edge*100:+.1f}%")

print_team_props(away_props, away_abbrev)
print_team_props(home_props, home_abbrev)

print("\n" + "=" * 70)
print("Done!")
