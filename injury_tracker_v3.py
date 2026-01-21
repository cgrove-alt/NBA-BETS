#!/usr/bin/env python3
"""
Real-Time NBA Injury Detection System - V3
===========================================

Integrates with RotoWire/FantasyLabs or NBA.com/ESPN injury reports
to eliminate DNP (Did Not Play) prediction errors.

Requirements: FR-4 (P0 Critical)
- Update frequency: Every 15 minutes during game days
- Detection rate: 100% (zero DNP players in predictions)
- Handles OUT, DOUBTFUL, QUESTIONABLE, GTD statuses
- Caches injury data in SQLite for historical analysis
"""

import os
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class InjuryStatus:
    """Injury status constants."""
    OUT = "OUT"
    DOUBTFUL = "DOUBTFUL"
    QUESTIONABLE = "QUESTIONABLE"
    GTD = "GTD"  # Game-Time Decision
    PROBABLE = "PROBABLE"
    ACTIVE = "ACTIVE"


class InjuryTrackerV3:
    """
    Real-time injury tracking system with multiple data sources.

    Primary: RotoWire API (paid, real-time)
    Fallback: NBA.com injury reports
    Cache: SQLite database for historical analysis
    """

    def __init__(self, db_path: str = "data/injuries.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

        # API configuration
        self.rotowire_api_key = os.getenv("ROTOWIRE_API_KEY")
        self.nba_api_endpoint = "https://www.nba.com/stats/teams/injury-report"

        # Cache settings
        self.cache_ttl = 15 * 60  # 15 minutes in seconds
        self._cache: Dict[str, Dict] = {}
        self._cache_timestamp: Optional[datetime] = None

    def _init_database(self):
        """Initialize SQLite database for injury tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS injuries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                status TEXT NOT NULL,
                injury_type TEXT,
                injury_detail TEXT,
                game_date TEXT,
                last_update TIMESTAMP NOT NULL,
                source TEXT NOT NULL,
                UNIQUE(player_id, game_date)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_date
            ON injuries(player_id, game_date)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_game_date
            ON injuries(game_date)
        """)

        conn.commit()
        conn.close()

    def _is_cache_valid(self) -> bool:
        """Check if cached injury data is still valid."""
        if not self._cache_timestamp:
            return False

        age = (datetime.now() - self._cache_timestamp).total_seconds()
        return age < self.cache_ttl

    def fetch_rotowire_injuries(self) -> Dict[str, Dict]:
        """
        Fetch injury data from RotoWire API (primary source).

        Returns:
            Dict mapping player_id -> injury info
        """
        if not self.rotowire_api_key:
            return {}

        try:
            # RotoWire API endpoint (example - adjust based on actual API docs)
            url = "https://api.rotowire.com/v1/nba/injuries"
            headers = {"Authorization": f"Bearer {self.rotowire_api_key}"}

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Parse RotoWire response
            injuries = {}
            for injury in data.get("injuries", []):
                player_id = injury.get("player_id")
                if player_id:
                    injuries[str(player_id)] = {
                        "player_name": injury.get("player_name"),
                        "team": injury.get("team"),
                        "status": injury.get("status", "QUESTIONABLE"),
                        "injury_type": injury.get("injury_type"),
                        "injury_detail": injury.get("details"),
                        "last_update": datetime.now().isoformat(),
                        "source": "rotowire"
                    }

            return injuries

        except requests.RequestException as e:
            print(f"RotoWire API error: {e}")
            return {}

    def fetch_nba_com_injuries(self) -> Dict[str, Dict]:
        """
        Fetch injury data from NBA.com (fallback source).

        Returns:
            Dict mapping player_id -> injury info
        """
        try:
            # NBA.com injuries endpoint
            # Note: This is a simplified version - actual NBA.com scraping may require
            # more complex parsing depending on their current page structure
            url = "https://www.nba.com/stats/teams/injury-report"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse NBA.com response (structure depends on actual API)
            # This is a placeholder - actual implementation would parse HTML or JSON
            injuries = {}

            # TODO: Implement actual NBA.com parsing
            # For now, return empty dict to indicate fallback is not implemented

            return injuries

        except requests.RequestException as e:
            print(f"NBA.com fetch error: {e}")
            return {}

    def get_injuries(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Get current injury data from best available source.

        Args:
            force_refresh: Force API call even if cache is valid

        Returns:
            Dict mapping player_id -> injury info
        """
        # Return cached data if valid
        if not force_refresh and self._is_cache_valid():
            return self._cache

        # Try primary source (RotoWire)
        injuries = self.fetch_rotowire_injuries()

        # Fall back to NBA.com if RotoWire failed
        if not injuries:
            injuries = self.fetch_nba_com_injuries()

        # Update cache
        self._cache = injuries
        self._cache_timestamp = datetime.now()

        # Save to database
        self._save_to_database(injuries)

        return injuries

    def _save_to_database(self, injuries: Dict[str, Dict]):
        """Save injury data to SQLite for historical analysis."""
        if not injuries:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")

        for player_id, injury_data in injuries.items():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO injuries
                    (player_id, player_name, team, status, injury_type,
                     injury_detail, game_date, last_update, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(player_id),
                    injury_data.get("player_name"),
                    injury_data.get("team"),
                    injury_data.get("status"),
                    injury_data.get("injury_type"),
                    injury_data.get("injury_detail"),
                    today,
                    injury_data.get("last_update"),
                    injury_data.get("source")
                ))
            except Exception as e:
                print(f"Error saving injury for player {player_id}: {e}")

        conn.commit()
        conn.close()

    def is_player_available(self, player_id: int, player_name: str = None) -> Tuple[bool, str, str]:
        """
        Check if a player is available to play.

        Args:
            player_id: NBA player ID
            player_name: Player name (for lookup if ID not found)

        Returns:
            Tuple of (is_available, status, uncertainty_level)
            - is_available: False if OUT or DOUBTFUL
            - status: InjuryStatus constant
            - uncertainty_level: "LOW", "MEDIUM", "HIGH"
        """
        injuries = self.get_injuries()

        player_key = str(player_id)

        # Check by ID first
        if player_key in injuries:
            injury = injuries[player_key]
            status = injury["status"]

            # Determine availability
            if status == InjuryStatus.OUT:
                return False, status, "LOW"  # Definitely out
            elif status == InjuryStatus.DOUBTFUL:
                return False, status, "MEDIUM"  # Likely out
            elif status == InjuryStatus.GTD:
                return True, status, "HIGH"  # Uncertain
            elif status == InjuryStatus.QUESTIONABLE:
                return True, status, "MEDIUM"  # Probably plays
            elif status == InjuryStatus.PROBABLE:
                return True, status, "LOW"  # Likely plays
            else:
                return True, InjuryStatus.ACTIVE, "LOW"

        # If not in injury list, assume available
        return True, InjuryStatus.ACTIVE, "LOW"

    def get_team_injuries(self, team: str, game_date: str = None) -> List[Dict]:
        """
        Get all injuries for a specific team.

        Args:
            team: Team abbreviation (e.g., "LAL", "BOS")
            game_date: Optional date filter (YYYY-MM-DD)

        Returns:
            List of injury records for the team
        """
        injuries = self.get_injuries()

        team_injuries = []
        for player_id, injury in injuries.items():
            if injury.get("team") == team:
                team_injuries.append({
                    "player_id": player_id,
                    **injury
                })

        return team_injuries

    def get_historical_availability(self, player_id: int, start_date: str, end_date: str) -> List[Dict]:
        """
        Get historical injury data for a player.

        Args:
            player_id: NBA player ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of injury records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT game_date, status, injury_type, injury_detail, last_update, source
            FROM injuries
            WHERE player_id = ? AND game_date BETWEEN ? AND ?
            ORDER BY game_date DESC
        """, (player_id, start_date, end_date))

        records = []
        for row in cursor.fetchall():
            records.append({
                "game_date": row[0],
                "status": row[1],
                "injury_type": row[2],
                "injury_detail": row[3],
                "last_update": row[4],
                "source": row[5]
            })

        conn.close()
        return records

    def filter_predictions_by_availability(self, predictions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter predictions to remove unavailable players and flag uncertain ones.

        Args:
            predictions: List of prediction dicts with 'player_id' and 'player_name'

        Returns:
            Tuple of (valid_predictions, filtered_out)
        """
        valid = []
        filtered = []

        for pred in predictions:
            player_id = pred.get("player_id")
            player_name = pred.get("player_name")

            is_available, status, uncertainty = self.is_player_available(player_id, player_name)

            if not is_available:
                # Player is OUT or DOUBTFUL - filter out
                pred["filtered_reason"] = f"Player {status}"
                pred["injury_status"] = status
                filtered.append(pred)
            else:
                # Player available - add uncertainty flag
                pred["injury_status"] = status
                pred["uncertainty_level"] = uncertainty
                if uncertainty == "HIGH":
                    pred["warning"] = f"Player is {status} - use with caution"
                valid.append(pred)

        return valid, filtered


# Global instance
_injury_tracker = None


def get_injury_tracker() -> InjuryTrackerV3:
    """Get global injury tracker instance."""
    global _injury_tracker
    if _injury_tracker is None:
        _injury_tracker = InjuryTrackerV3()
    return _injury_tracker


def fetch_current_injuries():
    """Fetch current injuries from all sources (wrapper for global instance)."""
    tracker = get_injury_tracker()
    return tracker.fetch_all_injuries()


def is_player_available(player_id: int, date: str = None) -> bool:
    """Check if player is available (not OUT) on given date (wrapper for global instance)."""
    tracker = get_injury_tracker()
    status = tracker.get_player_status(player_id, date)
    return status != InjuryStatus.OUT


if __name__ == "__main__":
    # Test the injury tracker
    tracker = InjuryTrackerV3()

    print("Testing Injury Tracker V3")
    print("=" * 60)

    # Get current injuries
    injuries = tracker.get_injuries(force_refresh=True)
    print(f"\nFound {len(injuries)} injured players")

    if injuries:
        print("\nSample injuries:")
        for i, (player_id, injury) in enumerate(list(injuries.items())[:5]):
            print(f"  {injury['player_name']} ({injury['team']}): {injury['status']} - {injury.get('injury_type', 'N/A')}")

    # Test player availability check
    print("\nTesting player availability:")
    test_players = [
        (203507, "Giannis Antetokounmpo"),
        (203081, "Damian Lillard"),
        (2544, "LeBron James")
    ]

    for player_id, name in test_players:
        available, status, uncertainty = tracker.is_player_available(player_id, name)
        print(f"  {name}: {'Available' if available else 'OUT'} ({status}, uncertainty: {uncertainty})")
