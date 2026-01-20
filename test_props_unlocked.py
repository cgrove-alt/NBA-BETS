import sys
from unittest.mock import patch
from dashboard.data_service import DataService

# Original method to minimize side effects
original_is_game_started = DataService._is_game_started

def mock_is_game_started(self, game_status, game_datetime=None):
    print(f"DEBUG: Bypassing lock for status='{game_status}'")
    return False

# Patch the class method
DataService._is_game_started = mock_is_game_started

if __name__ == "__main__":
    # Import main test logic
    import test_props
    import asyncio
    
    print("RUNNING WITH UNLOCKED GAMES PATCH")
    asyncio.run(test_props.test_props_generation())
