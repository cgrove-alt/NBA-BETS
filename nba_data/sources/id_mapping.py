"""
ID Mapping Module for NBA Betting Model

Maps between Balldontlie API IDs and player/team names.
Uses caching to minimize API calls.

Usage:
    from id_mapping import IDMapper, TEAM_ABBREV_TO_BDL

    mapper = IDMapper()
    player_id = mapper.get_player_id("LeBron James")
    team_id = TEAM_ABBREV_TO_BDL["LAL"]
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from difflib import SequenceMatcher

# Balldontlie team IDs (verified from API)
TEAM_ABBREV_TO_BDL = {
    "ATL": 1,   # Atlanta Hawks
    "BOS": 2,   # Boston Celtics
    "BKN": 3,   # Brooklyn Nets
    "CHA": 4,   # Charlotte Hornets
    "CHI": 5,   # Chicago Bulls
    "CLE": 6,   # Cleveland Cavaliers
    "DAL": 7,   # Dallas Mavericks
    "DEN": 8,   # Denver Nuggets
    "DET": 9,   # Detroit Pistons
    "GSW": 10,  # Golden State Warriors
    "HOU": 11,  # Houston Rockets
    "IND": 12,  # Indiana Pacers
    "LAC": 13,  # LA Clippers
    "LAL": 14,  # Los Angeles Lakers
    "MEM": 15,  # Memphis Grizzlies
    "MIA": 16,  # Miami Heat
    "MIL": 17,  # Milwaukee Bucks
    "MIN": 18,  # Minnesota Timberwolves
    "NOP": 19,  # New Orleans Pelicans
    "NYK": 20,  # New York Knicks
    "OKC": 21,  # Oklahoma City Thunder
    "ORL": 22,  # Orlando Magic
    "PHI": 23,  # Philadelphia 76ers
    "PHX": 24,  # Phoenix Suns
    "POR": 25,  # Portland Trail Blazers
    "SAC": 26,  # Sacramento Kings
    "SAS": 27,  # San Antonio Spurs
    "TOR": 28,  # Toronto Raptors
    "UTA": 29,  # Utah Jazz
    "WAS": 30,  # Washington Wizards
}

# Reverse mapping
BDL_TO_TEAM_ABBREV = {v: k for k, v in TEAM_ABBREV_TO_BDL.items()}

# Common name variations
NAME_ALIASES = {
    # Common nicknames and variations
    "PJ Washington": "P.J. Washington",
    "CJ McCollum": "C.J. McCollum",
    "OG Anunoby": "O.G. Anunoby",
    "RJ Barrett": "R.J. Barrett",
    "TJ McConnell": "T.J. McConnell",
    "JJ Redick": "J.J. Redick",
    "KJ Martin": "Kenyon Martin Jr.",
    "Moe Wagner": "Moritz Wagner",
    "Nic Claxton": "Nicolas Claxton",
    "Herb Jones": "Herbert Jones",
    "Scottie Barnes": "Scottie Barnes",
    "Gary Trent Jr": "Gary Trent Jr.",
    "Tim Hardaway Jr": "Tim Hardaway Jr.",
    "Kelly Oubre Jr": "Kelly Oubre Jr.",
    "Jaren Jackson Jr": "Jaren Jackson Jr.",
    "Marcus Morris Sr": "Marcus Morris Sr.",
    "Wendell Carter Jr": "Wendell Carter Jr.",
    "Larry Nance Jr": "Larry Nance Jr.",
    "Derrick Jones Jr": "Derrick Jones Jr.",
    "Dennis Smith Jr": "Dennis Smith Jr.",
    "Troy Brown Jr": "Troy Brown Jr.",
    "Lonnie Walker IV": "Lonnie Walker IV",
    "Robert Williams III": "Robert Williams III",
    "Kevin Porter Jr": "Kevin Porter Jr.",
    "Otto Porter Jr": "Otto Porter Jr.",
    "Gary Payton II": "Gary Payton II",
    "Michael Porter Jr": "Michael Porter Jr.",
    "Jabari Smith Jr": "Jabari Smith Jr.",
}


class IDMapper:
    """
    Maps player names to Balldontlie IDs with caching.

    Attributes:
        cache_path: Path to the player cache file
        cache_ttl: Cache time-to-live in hours
    """

    def __init__(self, cache_dir: str = None, cache_ttl_hours: int = 24):
        """
        Initialize the ID mapper.

        Args:
            cache_dir: Directory for cache files (defaults to .bdl_cache/)
            cache_ttl_hours: How long to keep cached data
        """
        cache_dir = Path(__file__).parent / ".bdl_cache" if cache_dir is None else Path(cache_dir)

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        # In-memory caches
        self._player_cache: dict[str, int] = {}  # name -> id
        self._player_by_id: dict[int, dict] = {}  # id -> player data
        self._all_players: list[dict] = []

        # Load cache from disk
        self._load_cache()

    def _cache_file(self, name: str) -> Path:
        """Get path to a cache file."""
        return self.cache_dir / f"{name}.json"

    def _load_cache(self):
        """Load player cache from disk if valid."""
        cache_file = self._cache_file("players")

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)

                # Check if cache is still valid
                cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
                if datetime.now() - cached_at < self.cache_ttl:
                    self._all_players = data.get("players", [])
                    self._rebuild_lookup_tables()
                    return
            except (json.JSONDecodeError, ValueError):
                pass

        # Cache invalid or missing - will need to refresh
        self._all_players = []

    def _save_cache(self):
        """Save player cache to disk."""
        cache_file = self._cache_file("players")

        data = {
            "cached_at": datetime.now().isoformat(),
            "players": self._all_players,
        }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _rebuild_lookup_tables(self):
        """Rebuild in-memory lookup tables from player list."""
        self._player_cache = {}
        self._player_by_id = {}

        for player in self._all_players:
            pid = player.get("id")
            first = player.get("first_name", "")
            last = player.get("last_name", "")
            full_name = f"{first} {last}".strip()

            if pid and full_name:
                # Store by full name (lowercase for lookup)
                self._player_cache[full_name.lower()] = pid
                self._player_by_id[pid] = player

    def build_player_cache(self, api=None):
        """
        Build/refresh the player cache from Balldontlie API.

        Args:
            api: BalldontlieAPI instance (creates one if not provided)
        """
        if api is None:
            from balldontlie_api import BalldontlieAPI
            api = BalldontlieAPI()

        print("Building player cache from Balldontlie API...")

        # Fetch all active players
        self._all_players = api.get_all_active_players()

        if not self._all_players:
            # Fallback to paginated player fetch
            self._all_players = api.get_players_paginated()

        print(f"  Cached {len(self._all_players)} players")

        # Rebuild lookup tables
        self._rebuild_lookup_tables()

        # Save to disk
        self._save_cache()

    def get_player_id(
        self,
        player_name: str,
        fuzzy: bool = True,
        min_score: float = 0.8,
    ) -> int | None:
        """
        Get Balldontlie player ID from name.

        Args:
            player_name: Player name to look up
            fuzzy: Use fuzzy matching if exact match fails
            min_score: Minimum similarity score for fuzzy match (0-1)

        Returns:
            Player ID or None if not found
        """
        if not player_name:
            return None

        # Ensure cache is populated
        if not self._player_cache:
            self.build_player_cache()

        name_lower = player_name.lower().strip()

        # Check name aliases first
        if player_name in NAME_ALIASES:
            name_lower = NAME_ALIASES[player_name].lower()

        # Exact match
        if name_lower in self._player_cache:
            return self._player_cache[name_lower]

        # Try without Jr./Sr./II/III/IV suffixes
        name_simplified = self._simplify_name(name_lower)
        if name_simplified in self._player_cache:
            return self._player_cache[name_simplified]

        # Fuzzy matching
        if fuzzy:
            best_match, score = self._fuzzy_match(name_lower)
            if best_match and score >= min_score:
                return self._player_cache.get(best_match)

        return None

    def _simplify_name(self, name: str) -> str:
        """Remove suffixes and normalize name."""
        suffixes = [" jr.", " sr.", " iii", " ii", " iv", " jr", " sr"]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name.strip()

    def _fuzzy_match(self, name: str) -> tuple[str | None, float]:
        """
        Find best fuzzy match for a player name.

        Returns:
            Tuple of (matched_name, score) or (None, 0) if no match
        """
        best_match = None
        best_score = 0

        for cached_name in self._player_cache:
            score = SequenceMatcher(None, name, cached_name).ratio()
            if score > best_score:
                best_score = score
                best_match = cached_name

        return best_match, best_score

    def get_player_name(self, player_id: int) -> str | None:
        """
        Get player name from Balldontlie ID.

        Args:
            player_id: Balldontlie player ID

        Returns:
            Player full name or None
        """
        if not self._player_by_id:
            self.build_player_cache()

        player = self._player_by_id.get(player_id, {})
        first = player.get("first_name", "")
        last = player.get("last_name", "")
        return f"{first} {last}".strip() if first or last else None

    def get_player_info(self, player_id: int) -> dict | None:
        """
        Get full player info from Balldontlie ID.

        Args:
            player_id: Balldontlie player ID

        Returns:
            Player dictionary with all fields
        """
        if not self._player_by_id:
            self.build_player_cache()

        return self._player_by_id.get(player_id)

    def get_team_id(self, team_abbrev: str) -> int | None:
        """
        Get Balldontlie team ID from abbreviation.

        Args:
            team_abbrev: Team abbreviation (e.g., "LAL", "BOS")

        Returns:
            Team ID or None
        """
        return TEAM_ABBREV_TO_BDL.get(team_abbrev.upper())

    def get_team_abbrev(self, team_id: int) -> str | None:
        """
        Get team abbreviation from Balldontlie ID.

        Args:
            team_id: Balldontlie team ID

        Returns:
            Team abbreviation or None
        """
        return BDL_TO_TEAM_ABBREV.get(team_id)

    def search_players(
        self,
        query: str,
        limit: int = 5,
    ) -> list[tuple[str, int, float]]:
        """
        Search for players by name (fuzzy).

        Args:
            query: Search query
            limit: Max results to return

        Returns:
            List of (name, id, score) tuples sorted by score
        """
        if not self._player_cache:
            self.build_player_cache()

        query_lower = query.lower()
        results = []

        for name, pid in self._player_cache.items():
            score = SequenceMatcher(None, query_lower, name).ratio()
            results.append((name.title(), pid, score))

        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]


# Convenience function
def get_player_id(player_name: str) -> int | None:
    """Quick lookup of Balldontlie player ID."""
    mapper = IDMapper()
    return mapper.get_player_id(player_name)


def get_team_id(team_abbrev: str) -> int | None:
    """Quick lookup of Balldontlie team ID."""
    return TEAM_ABBREV_TO_BDL.get(team_abbrev.upper())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ID Mapping Utility")
    parser.add_argument("--build-cache", action="store_true", help="Build player cache")
    parser.add_argument("--lookup", type=str, help="Look up player ID by name")
    parser.add_argument("--search", type=str, help="Search for players")
    parser.add_argument("--teams", action="store_true", help="List all team IDs")

    args = parser.parse_args()

    mapper = IDMapper()

    if args.build_cache:
        mapper.build_player_cache()
        print("Cache built successfully!")

    elif args.lookup:
        pid = mapper.get_player_id(args.lookup)
        if pid:
            print(f"{args.lookup} -> ID: {pid}")
        else:
            print(f"Player not found: {args.lookup}")
            # Show similar matches
            matches = mapper.search_players(args.lookup)
            if matches:
                print("\nDid you mean:")
                for name, pid, score in matches:
                    print(f"  {name} (ID: {pid}, score: {score:.2f})")

    elif args.search:
        results = mapper.search_players(args.search, limit=10)
        print(f"Search results for '{args.search}':")
        for name, pid, score in results:
            print(f"  {name} (ID: {pid}, score: {score:.2f})")

    elif args.teams:
        print("Team ID Mapping:")
        for abbrev, tid in sorted(TEAM_ABBREV_TO_BDL.items()):
            print(f"  {abbrev}: {tid}")

    else:
        # Test basic functionality
        print("Testing ID Mapper...")

        test_players = [
            "LeBron James",
            "Stephen Curry",
            "Jayson Tatum",
            "Cade Cunningham",
            "Tim Hardaway Jr",
        ]

        for name in test_players:
            pid = mapper.get_player_id(name)
            print(f"  {name}: {pid}")

        print("\nTeam IDs:")
        for abbrev in ["LAL", "BOS", "GSW", "MIA", "NYK"]:
            print(f"  {abbrev}: {get_team_id(abbrev)}")
