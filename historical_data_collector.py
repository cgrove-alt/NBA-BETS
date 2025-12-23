"""
Historical NBA Data Collector

Batch fetches 3+ seasons of NBA games with point-in-time statistics.
Stores data in SQLite database for proper training data management.

Features:
- Resume capability for interrupted fetches
- Rate limiting for NBA API (0.6s delays)
- Point-in-time stats (stats known BEFORE each game)
- Progress tracking and logging

Usage:
    python historical_data_collector.py

    # Or programmatically:
    collector = HistoricalDataCollector()
    collector.collect_all_seasons()
"""

import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

try:
    from nba_api.stats.endpoints import (
        leaguegamefinder,
        teamdashboardbygeneralsplits,
        leaguedashteamstats,
    )
    from nba_api.stats.static import teams
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False
    print("Warning: nba_api not installed. Install with: pip install nba_api")

from database import DatabaseManager, initialize_teams, initialize_seasons
from fast_data_fetcher import NBA_TEAMS

# Rate limiting
API_DELAY = 0.6  # seconds between API calls
MAX_RETRIES = 3

# Seasons to collect
SEASONS_TO_COLLECT = [
    {"season": "2022-23", "start_date": "2022-10-18", "end_date": "2023-04-09"},
    {"season": "2023-24", "start_date": "2023-10-24", "end_date": "2024-04-14"},
    {"season": "2024-25", "start_date": "2024-10-22", "end_date": "2025-04-13"},
    {"season": "2025-26", "start_date": "2025-10-21", "end_date": None},  # Current season
]

# Progress file for resume capability
PROGRESS_FILE = Path("collection_progress.json")


class HistoricalDataCollector:
    """
    Collects historical NBA data and stores in SQLite database.

    Key principle: Only store stats that were known BEFORE each game
    to avoid look-ahead bias in training.
    """

    def __init__(self, db_path: str = "nba_betting.db"):
        self.db = DatabaseManager(Path(db_path))
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict:
        """Load progress from file for resume capability."""
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        return {"completed_seasons": [], "last_game_date": None, "games_collected": 0}

    def _save_progress(self):
        """Save progress for resume capability."""
        with open(PROGRESS_FILE, "w") as f:
            json.dump(self.progress, f, indent=2)

    def initialize_database(self):
        """Initialize database with schema, teams, and seasons."""
        print("Initializing database...")
        self.db.initialize()
        initialize_teams(self.db)
        initialize_seasons(self.db)
        print("Database initialized successfully.")

    def _api_call_with_retry(self, func, *args, **kwargs):
        """Make API call with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(API_DELAY)
                return func(*args, **kwargs)
            except Exception as e:
                print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(API_DELAY * 2)  # Extra delay on retry
                else:
                    raise
        return None

    def fetch_season_games(self, season: str) -> List[Dict]:
        """
        Fetch all games for a season.

        Args:
            season: NBA season string (e.g., "2024-25")

        Returns:
            List of game dictionaries
        """
        if not HAS_NBA_API:
            print("nba_api not available - cannot fetch games")
            return []

        print(f"\nFetching games for {season} season...")

        all_games = []
        seen_game_ids = set()

        # Fetch all teams' games
        for abbrev, team_data in NBA_TEAMS.items():
            team_id = team_data["id"]

            try:
                game_finder = self._api_call_with_retry(
                    leaguegamefinder.LeagueGameFinder,
                    team_id_nullable=team_id,
                    season_nullable=season,
                    season_type_nullable="Regular Season",
                )

                if game_finder is None:
                    continue

                games_dict = game_finder.get_normalized_dict()
                games = games_dict.get("LeagueGameFinderResults", [])

                for game in games:
                    game_id = game.get("GAME_ID")
                    if game_id and game_id not in seen_game_ids:
                        seen_game_ids.add(game_id)
                        all_games.append(game)

                print(f"  {abbrev}: {len(games)} game records")

            except Exception as e:
                print(f"  Error fetching {abbrev} games: {e}")
                continue

        print(f"Total unique games found: {len(all_games) // 2}")  # Each game appears twice
        return all_games

    def fetch_team_stats_at_date(
        self,
        team_id: int,
        season: str,
        as_of_date: str
    ) -> Optional[Dict]:
        """
        Fetch team stats as of a specific date (point-in-time).

        This avoids look-ahead bias by only using stats known before the game.

        Args:
            team_id: NBA team ID
            season: Season string (e.g., "2024-25")
            as_of_date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary of team stats or None
        """
        if not HAS_NBA_API:
            return None

        try:
            # Format date for API
            date_obj = datetime.strptime(as_of_date, "%Y-%m-%d")
            date_to = (date_obj - timedelta(days=1)).strftime("%m/%d/%Y")

            team_stats = self._api_call_with_retry(
                teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits,
                team_id=team_id,
                season=season,
                season_type_all_star="Regular Season",
                date_to_nullable=date_to,
            )

            if team_stats is None:
                return None

            stats_dict = team_stats.get_normalized_dict()
            overall = stats_dict.get("OverallTeamDashboard", [{}])

            if not overall or not overall[0]:
                return None

            stats = overall[0]

            return {
                "team_id": team_id,
                "games_played": stats.get("GP", 0),
                "wins": stats.get("W", 0),
                "losses": stats.get("L", 0),
                "pts_avg": stats.get("PTS", 0),
                "opp_pts_avg": stats.get("OPP_PTS", 0) if stats.get("OPP_PTS") else None,
                "fg_pct": stats.get("FG_PCT", 0),
                "fg3_pct": stats.get("FG3_PCT", 0),
                "ft_pct": stats.get("FT_PCT", 0),
                "reb_avg": stats.get("REB", 0),
                "ast_avg": stats.get("AST", 0),
                "stl_avg": stats.get("STL", 0),
                "blk_avg": stats.get("BLK", 0),
                "tov_avg": stats.get("TOV", 0),
                "off_rating": stats.get("OFF_RATING"),
                "def_rating": stats.get("DEF_RATING"),
                "net_rating": stats.get("NET_RATING"),
                "pace": stats.get("PACE"),
            }

        except Exception as e:
            print(f"    Error fetching team {team_id} stats: {e}")
            return None

    def process_game_pair(
        self,
        home_game: Dict,
        away_game: Dict,
        season_id: int
    ) -> Optional[Dict]:
        """
        Process a home/away game pair into a complete game record.

        Args:
            home_game: Home team's game record
            away_game: Away team's game record
            season_id: Database season ID

        Returns:
            Processed game dictionary or None
        """
        game_id = home_game.get("GAME_ID")
        game_date = home_game.get("GAME_DATE")

        home_team_id = home_game.get("TEAM_ID")
        away_team_id = away_game.get("TEAM_ID")
        home_score = home_game.get("PTS")
        away_score = away_game.get("PTS")

        if not all([game_id, game_date, home_team_id, away_team_id, home_score, away_score]):
            return None

        # Get database team IDs
        home_team = self.db.get_team_by_nba_id(home_team_id)
        away_team = self.db.get_team_by_nba_id(away_team_id)

        if not home_team or not away_team:
            return None

        return {
            "nba_game_id": game_id,
            "season_id": season_id,
            "game_date": game_date,
            "home_team_id": home_team["id"],
            "away_team_id": away_team["id"],
            "home_score": home_score,
            "away_score": away_score,
            "status": "completed",
            "home_nba_id": home_team_id,
            "away_nba_id": away_team_id,
        }

    def collect_season(self, season_info: Dict) -> int:
        """
        Collect all data for a single season.

        Args:
            season_info: Dictionary with season, start_date, end_date

        Returns:
            Number of games collected
        """
        season = season_info["season"]

        # Check if already completed
        if season in self.progress.get("completed_seasons", []):
            print(f"\nSeason {season} already collected. Skipping.")
            return 0

        print(f"\n{'='*60}")
        print(f"Collecting data for {season} season")
        print(f"{'='*60}")

        # Get season ID from database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM seasons WHERE season_string = ?",
                (season,)
            )
            row = cursor.fetchone()
            if not row:
                print(f"Season {season} not found in database")
                return 0
            season_id = row[0]

        # Fetch all games
        raw_games = self.fetch_season_games(season)

        if not raw_games:
            print(f"No games found for {season}")
            return 0

        # Group games by game_id (each game has home and away records)
        games_by_id = {}
        for game in raw_games:
            game_id = game.get("GAME_ID")
            matchup = game.get("MATCHUP", "")

            if game_id not in games_by_id:
                games_by_id[game_id] = {"home": None, "away": None}

            # Determine if home or away based on matchup string
            if "@" in matchup:
                games_by_id[game_id]["away"] = game
            else:
                games_by_id[game_id]["home"] = game

        # Process each game
        games_collected = 0
        total_games = len(games_by_id)

        print(f"\nProcessing {total_games} games...")

        for i, (game_id, game_pair) in enumerate(games_by_id.items()):
            home_game = game_pair["home"]
            away_game = game_pair["away"]

            if not home_game or not away_game:
                continue

            # Process game
            game_data = self.process_game_pair(home_game, away_game, season_id)

            if not game_data:
                continue

            try:
                # Store game in database
                self.db.upsert_game(game_data)
                games_collected += 1

                # Progress update every 50 games
                if games_collected % 50 == 0:
                    print(f"  Processed {games_collected}/{total_games} games...")
                    self.progress["games_collected"] = self.progress.get("games_collected", 0) + 50
                    self.progress["last_game_date"] = game_data["game_date"]
                    self._save_progress()

            except Exception as e:
                print(f"  Error storing game {game_id}: {e}")
                continue

        print(f"\nCompleted {season}: {games_collected} games stored")

        # Mark season as completed
        if "completed_seasons" not in self.progress:
            self.progress["completed_seasons"] = []
        self.progress["completed_seasons"].append(season)
        self._save_progress()

        return games_collected

    def collect_team_stats_snapshots(self, season: str, sample_dates: List[str] = None):
        """
        Collect team stats snapshots at regular intervals.

        For proper backtesting, we need point-in-time stats.
        This collects stats at weekly intervals throughout the season.

        Args:
            season: Season string (e.g., "2024-25")
            sample_dates: Optional list of dates to sample (YYYY-MM-DD)
        """
        print(f"\nCollecting team stats snapshots for {season}...")

        # Get season info
        season_info = None
        for s in SEASONS_TO_COLLECT:
            if s["season"] == season:
                season_info = s
                break

        if not season_info:
            print(f"Season {season} not found in configuration")
            return

        # Generate sample dates (every 2 weeks)
        if sample_dates is None:
            sample_dates = []
            start = datetime.strptime(season_info["start_date"], "%Y-%m-%d")
            end = datetime.strptime(season_info["end_date"], "%Y-%m-%d") if season_info["end_date"] else datetime.now()

            current = start + timedelta(days=14)  # Start 2 weeks into season
            while current < end:
                sample_dates.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=14)

        print(f"  Sampling stats at {len(sample_dates)} dates")

        # Get season ID
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM seasons WHERE season_string = ?",
                (season,)
            )
            row = cursor.fetchone()
            season_id = row[0] if row else None

        if not season_id:
            print(f"Season {season} not in database")
            return

        # Collect stats for each team at each date
        for date in sample_dates:
            print(f"  Collecting stats as of {date}...")

            for abbrev, team_data in NBA_TEAMS.items():
                nba_team_id = team_data["id"]

                # Get database team ID
                team = self.db.get_team_by_nba_id(nba_team_id)
                if not team:
                    continue

                # Fetch stats as of this date
                stats = self.fetch_team_stats_at_date(nba_team_id, season, date)

                if stats and stats.get("games_played", 0) > 0:
                    stats["team_id"] = team["id"]
                    stats["season_id"] = season_id
                    stats["snapshot_date"] = date

                    try:
                        self.db.upsert_team_stats(stats)
                    except Exception as e:
                        print(f"    Error storing {abbrev} stats: {e}")

            # Rate limiting between dates
            time.sleep(API_DELAY)

    def collect_all_seasons(self, include_stats_snapshots: bool = False):
        """
        Collect data for all configured seasons.

        Args:
            include_stats_snapshots: Whether to also collect point-in-time stats
        """
        print("\n" + "="*60)
        print("NBA Historical Data Collection")
        print("="*60)

        # Initialize database
        self.initialize_database()

        total_games = 0

        for season_info in SEASONS_TO_COLLECT:
            try:
                games = self.collect_season(season_info)
                total_games += games

                if include_stats_snapshots:
                    self.collect_team_stats_snapshots(season_info["season"])

            except Exception as e:
                print(f"\nError collecting {season_info['season']}: {e}")
                traceback.print_exc()
                continue

        print("\n" + "="*60)
        print(f"Collection Complete!")
        print(f"Total games collected: {total_games}")
        print(f"Database: {self.db.db_path}")
        print("="*60)

        # Clean up progress file
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()

    def get_collection_stats(self) -> Dict:
        """Get statistics about collected data."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Count games by season
            cursor.execute("""
                SELECT s.season_string, COUNT(g.id) as game_count
                FROM games g
                JOIN seasons s ON g.season_id = s.id
                GROUP BY s.season_string
            """)
            seasons = {row[0]: row[1] for row in cursor.fetchall()}

            # Count total games
            cursor.execute("SELECT COUNT(*) FROM games")
            total_games = cursor.fetchone()[0]

            # Count teams
            cursor.execute("SELECT COUNT(*) FROM teams")
            total_teams = cursor.fetchone()[0]

            # Count stats snapshots
            cursor.execute("SELECT COUNT(*) FROM team_stats_snapshot")
            total_snapshots = cursor.fetchone()[0]

            return {
                "total_games": total_games,
                "total_teams": total_teams,
                "total_snapshots": total_snapshots,
                "games_by_season": seasons,
            }


def generate_training_features_from_db(db: DatabaseManager) -> List[Dict]:
    """
    Generate training features from database games.

    This creates properly-scaled features using point-in-time stats.

    Args:
        db: DatabaseManager instance

    Returns:
        List of training examples with features and outcomes
    """
    training_data = []

    # Get all completed games
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT g.*,
                   ht.abbreviation as home_abbrev,
                   at.abbreviation as away_abbrev
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE g.status = 'completed' AND g.home_score IS NOT NULL
            ORDER BY g.game_date
        """)

        games = [dict(row) for row in cursor.fetchall()]

    print(f"Generating features for {len(games)} games...")

    for game in games:
        # Get point-in-time stats for both teams
        home_stats = db.get_team_stats_before_date(
            game["home_team_id"],
            game["game_date"]
        )
        away_stats = db.get_team_stats_before_date(
            game["away_team_id"],
            game["game_date"]
        )

        # Skip if no historical stats available
        if not home_stats or not away_stats:
            continue

        # Calculate features
        home_win = game["home_score"] > game["away_score"]
        point_diff = game["home_score"] - game["away_score"]

        # Win percentages
        home_win_pct = home_stats["wins"] / max(1, home_stats["games_played"])
        away_win_pct = away_stats["wins"] / max(1, away_stats["games_played"])

        features = {
            # Win percentage differential
            "season_win_pct_diff": home_win_pct - away_win_pct,

            # Scoring differentials
            "pts_avg_diff": (home_stats.get("pts_avg") or 110) - (away_stats.get("pts_avg") or 110),

            # Efficiency differentials
            "off_rating_diff": (home_stats.get("off_rating") or 110) - (away_stats.get("off_rating") or 110),
            "def_rating_diff": (home_stats.get("def_rating") or 110) - (away_stats.get("def_rating") or 110),
            "net_rating_diff": (home_stats.get("net_rating") or 0) - (away_stats.get("net_rating") or 0),

            # Individual stats
            "home_season_win_pct": home_win_pct,
            "away_season_win_pct": away_win_pct,
            "home_pts_avg": home_stats.get("pts_avg") or 110,
            "away_pts_avg": away_stats.get("pts_avg") or 110,
            "home_off_rating": home_stats.get("off_rating") or 110,
            "away_off_rating": away_stats.get("off_rating") or 110,
            "home_def_rating": home_stats.get("def_rating") or 110,
            "away_def_rating": away_stats.get("def_rating") or 110,
            "home_net_rating": home_stats.get("net_rating") or 0,
            "away_net_rating": away_stats.get("net_rating") or 0,

            # Pace
            "avg_pace": ((home_stats.get("pace") or 100) + (away_stats.get("pace") or 100)) / 2,
            "pace_diff": (home_stats.get("pace") or 100) - (away_stats.get("pace") or 100),

            # Shooting
            "fg_pct_diff": (home_stats.get("fg_pct") or 0.45) - (away_stats.get("fg_pct") or 0.45),
            "fg3_pct_diff": (home_stats.get("fg3_pct") or 0.36) - (away_stats.get("fg3_pct") or 0.36),

            # Home advantage
            "home_advantage_factor": 0.06,
        }

        training_data.append({
            "game_date": game["game_date"],
            "home_team": game["home_abbrev"],
            "away_team": game["away_abbrev"],
            "home_win": home_win,
            "point_differential": point_diff,
            "home_score": game["home_score"],
            "away_score": game["away_score"],
            "moneyline_features": features,
            "spread_features": {
                **features,
                "expected_home_pts": home_stats.get("pts_avg") or 110,
                "expected_away_pts": away_stats.get("pts_avg") or 110,
                "expected_point_diff": (home_stats.get("net_rating") or 0) - (away_stats.get("net_rating") or 0) + 3,  # Home advantage
            },
        })

    print(f"Generated {len(training_data)} training examples with features")
    return training_data


if __name__ == "__main__":
    print("NBA Historical Data Collector")
    print("="*60)

    if not HAS_NBA_API:
        print("\nERROR: nba_api is required but not installed.")
        print("Install it with: pip install nba_api")
        exit(1)

    collector = HistoricalDataCollector()

    # Option 1: Collect all seasons (takes ~30-45 minutes)
    print("\nThis will collect ~3,500 games across 3 seasons.")
    print("Estimated time: 30-45 minutes with rate limiting.\n")

    try:
        collector.collect_all_seasons(include_stats_snapshots=True)

        # Show collection stats
        stats = collector.get_collection_stats()
        print("\nCollection Statistics:")
        print(f"  Total games: {stats['total_games']}")
        print(f"  Total teams: {stats['total_teams']}")
        print(f"  Stats snapshots: {stats['total_snapshots']}")
        print(f"  Games by season: {stats['games_by_season']}")

    except KeyboardInterrupt:
        print("\n\nCollection interrupted. Progress saved.")
        print("Run again to resume from where you left off.")
