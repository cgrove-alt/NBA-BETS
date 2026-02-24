"""
NBA Betting Market Features Module

Consolidates betting market intelligence from odds movements, line shopping,
and market microstructure analysis to generate predictive features for the model.

This module integrates with existing odds_fetcher.py and market_microstructure.py
to provide a unified interface for betting market features.

Features:
- Opening and closing line tracking
- Line movement detection and quantification
- Reverse Line Movement (RLM) detection
- Consensus odds calculation across multiple sportsbooks
- Steam move detection (rapid sharp money movement)
- Real-time odds tracking and historical storage
- Closing Line Value (CLV) calculation

Key Concepts:
- RLM (Reverse Line Movement): Line moves opposite to public betting percentages
- Steam Move: Rapid line movement indicating sharp money (1.5+ points in <15 min)
- CLV (Closing Line Value): Difference between bet odds and closing line (positive = edge)
- Consensus Odds: Fair market odds calculated across 10+ sportsbooks

Usage:
    # Initialize with The Odds API key
    tracker = BettingMarketFeatures(api_key="your_key")

    # Fetch and store odds
    tracker.fetch_and_store_current_odds()

    # Get features for a game
    features = tracker.get_market_features(game_id, home_team, away_team)

    # Detect steam moves
    steam_alerts = tracker.detect_steam_moves(game_id)
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any
import numpy as np
from contextlib import contextmanager

# Import existing infrastructure
try:
    from odds_fetcher import OddsFetcher, LineMovementTracker, CLVTracker
    HAS_ODDS_FETCHER = True
except ImportError:
    HAS_ODDS_FETCHER = False
    print("Warning: odds_fetcher.py not found. Some features may be limited.")

try:
    from market_microstructure import (
        SteamDetector, StaleLineFinder, american_to_prob, prob_to_american,
        remove_vig, calculate_consensus as mm_calculate_consensus
    )
    HAS_MARKET_MICRO = True
except ImportError:
    HAS_MARKET_MICRO = False
    print("Warning: market_microstructure.py not found. Steam detection limited.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default database for odds history
DEFAULT_ODDS_DB = "odds_history.db"

# Sportsbook configuration (for consensus calculation)
SPORTSBOOKS = [
    "draftkings", "fanduel", "betmgm", "caesars", "pointsbet",
    "betrivers", "unibet_us", "wynnbet", "barstool", "foxbet"
]

# Detection thresholds
STEAM_THRESHOLD_POINTS = 1.5  # Points of spread/total movement
STEAM_THRESHOLD_ML = 0.03  # Moneyline probability movement (3%)
STEAM_TIME_WINDOW = 900  # 15 minutes in seconds
RLM_THRESHOLD = 0.02  # 2% probability movement opposite to public

# Rate limiting
UPDATE_INTERVAL_SECONDS = 300  # 5 minutes between updates


# =============================================================================
# DATABASE MANAGER FOR ODDS HISTORY
# =============================================================================

class OddsHistoryDB:
    """
    SQLite database for storing historical odds snapshots.

    Schema:
    - odds_history: Point-in-time odds snapshots from each sportsbook
    - games: Game metadata (home/away teams, commence time)
    - line_movements: Calculated line movements between snapshots
    """

    def __init__(self, db_path: str = DEFAULT_ODDS_DB):
        self.db_path = db_path
        self._initialize_schema()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create tables and indexes if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Games table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    commence_time TEXT NOT NULL,
                    sport TEXT DEFAULT 'basketball_nba',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Odds history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS odds_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    book_name TEXT NOT NULL,
                    market TEXT NOT NULL,
                    home_odds FLOAT,
                    away_odds FLOAT,
                    home_line FLOAT,
                    away_line FLOAT,
                    total FLOAT,
                    over_odds FLOAT,
                    under_odds FLOAT,
                    is_opening BOOLEAN DEFAULT 0,
                    is_closing BOOLEAN DEFAULT 0,
                    FOREIGN KEY (game_id) REFERENCES games(game_id),
                    UNIQUE(game_id, timestamp, book_name, market)
                )
            """)

            # Line movements summary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS line_movements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    market TEXT NOT NULL,
                    opening_line FLOAT,
                    closing_line FLOAT,
                    movement FLOAT,
                    opening_time TIMESTAMP,
                    closing_time TIMESTAMP,
                    num_moves INTEGER,
                    max_move FLOAT,
                    rlm_detected BOOLEAN DEFAULT 0,
                    steam_detected BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES games(game_id),
                    UNIQUE(game_id, market)
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_odds_game_market
                ON odds_history(game_id, market, timestamp DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_odds_timestamp
                ON odds_history(timestamp DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_movements_game
                ON line_movements(game_id)
            """)

    def upsert_game(self, game_id: str, home_team: str, away_team: str, commence_time: str):
        """Insert or update game metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO games (game_id, home_team, away_team, commence_time)
                VALUES (?, ?, ?, ?)
            """, (game_id, home_team, away_team, commence_time))

    def insert_odds_snapshot(self, game_id: str, book_name: str, market: str,
                            odds_data: dict, is_opening: bool = False, is_closing: bool = False):
        """Insert an odds snapshot."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            timestamp = datetime.now().isoformat()

            cursor.execute("""
                INSERT OR IGNORE INTO odds_history (
                    game_id, timestamp, book_name, market,
                    home_odds, away_odds, home_line, away_line,
                    total, over_odds, under_odds,
                    is_opening, is_closing
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, timestamp, book_name, market,
                odds_data.get('home_odds'),
                odds_data.get('away_odds'),
                odds_data.get('home_line'),
                odds_data.get('away_line'),
                odds_data.get('total'),
                odds_data.get('over_odds'),
                odds_data.get('under_odds'),
                is_opening, is_closing
            ))

    def get_opening_line(self, game_id: str, market: str) -> dict | None:
        """Fetch opening line for a game/market."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM odds_history
                WHERE game_id = ? AND market = ? AND is_opening = 1
                ORDER BY timestamp ASC
                LIMIT 1
            """, (game_id, market))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_closing_line(self, game_id: str, market: str) -> dict | None:
        """Fetch closing line for a game/market."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM odds_history
                WHERE game_id = ? AND market = ? AND is_closing = 1
                ORDER BY timestamp DESC
                LIMIT 1
            """, (game_id, market))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_odds_history(self, game_id: str, market: str,
                         lookback_minutes: int = 60) -> list[dict]:
        """Get odds history for a game/market within lookback window."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(minutes=lookback_minutes)).isoformat()

            cursor.execute("""
                SELECT * FROM odds_history
                WHERE game_id = ? AND market = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (game_id, market, cutoff))

            return [dict(row) for row in cursor.fetchall()]

    def upsert_line_movement(self, game_id: str, market: str, movement_data: dict):
        """Insert or update line movement summary."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO line_movements (
                    game_id, market, opening_line, closing_line, movement,
                    opening_time, closing_time, num_moves, max_move,
                    rlm_detected, steam_detected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, market,
                movement_data.get('opening_line'),
                movement_data.get('closing_line'),
                movement_data.get('movement'),
                movement_data.get('opening_time'),
                movement_data.get('closing_time'),
                movement_data.get('num_moves', 0),
                movement_data.get('max_move', 0),
                movement_data.get('rlm_detected', False),
                movement_data.get('steam_detected', False)
            ))

    def get_line_movement(self, game_id: str, market: str) -> dict | None:
        """Get line movement summary for a game/market."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM line_movements
                WHERE game_id = ? AND market = ?
            """, (game_id, market))
            row = cursor.fetchone()
            return dict(row) if row else None


# =============================================================================
# MAIN BETTING MARKET FEATURES CLASS
# =============================================================================

class BettingMarketFeatures:
    """
    Main class for generating betting market features.

    Provides unified interface to:
    - Fetch real-time odds from The Odds API
    - Store odds history in SQLite
    - Calculate line movements
    - Detect RLM and steam moves
    - Generate features for ML models
    """

    def __init__(self, api_key: str | None = None, db_path: str = DEFAULT_ODDS_DB):
        """
        Initialize betting market features tracker.

        Args:
            api_key: The Odds API key (or set THE_ODDS_API_KEY env var)
            db_path: Path to SQLite database for odds history
        """
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
        self.db = OddsHistoryDB(db_path)

        # Initialize odds fetcher if available
        if HAS_ODDS_FETCHER and self.api_key:
            self.odds_fetcher = OddsFetcher(self.api_key)
            self.line_tracker = LineMovementTracker()
            self.clv_tracker = CLVTracker(self.odds_fetcher)
        else:
            self.odds_fetcher = None
            self.line_tracker = None
            self.clv_tracker = None
            if not self.api_key:
                print("Warning: No API key provided. Set THE_ODDS_API_KEY or pass api_key parameter.")

        # Cache for current odds
        self._odds_cache: dict[str, dict] = {}
        self._cache_timestamp: datetime | None = None
        self._cache_ttl = timedelta(minutes=5)

    # =========================================================================
    # ODDS FETCHING AND STORAGE
    # =========================================================================

    def fetch_current_odds(self, force_refresh: bool = False) -> list[dict]:
        """
        Fetch current NBA odds from The Odds API.

        Args:
            force_refresh: Force API call even if cache is valid

        Returns:
            List of game odds dictionaries
        """
        # Check cache
        if not force_refresh and self._cache_timestamp:
            if datetime.now() - self._cache_timestamp < self._cache_ttl:
                return list(self._odds_cache.values())

        if not self.odds_fetcher:
            print("Error: OddsFetcher not initialized. Check API key.")
            return []

        try:
            odds_data = self.odds_fetcher.get_nba_odds()

            # Update cache
            self._odds_cache = {game['game_id']: game for game in odds_data}
            self._cache_timestamp = datetime.now()

            return odds_data
        except Exception as e:
            print(f"Error fetching odds: {e}")
            return []

    def fetch_and_store_odds(self, mark_as_opening: bool = False,
                            mark_as_closing: bool = False,
                            auto_detect_opening: bool = True,
                            auto_detect_closing: bool = True) -> int:
        """
        Fetch current odds and store in database.

        Args:
            mark_as_opening: Manually mark these odds as opening lines
            mark_as_closing: Manually mark these odds as closing lines
            auto_detect_opening: Auto-mark as opening if first odds for game (default True)
            auto_detect_closing: Auto-mark as closing if game starts in <15 min (default True)

        Returns:
            Number of snapshots stored
        """
        odds_data = self.fetch_current_odds(force_refresh=True)
        count = 0

        for game in odds_data:
            game_id = game.get('game_id')
            if not game_id:
                continue

            # Store game metadata
            commence_time_str = game.get('commence_time', '')
            self.db.upsert_game(
                game_id,
                game.get('home_team', ''),
                game.get('away_team', ''),
                commence_time_str
            )

            # Auto-detect opening/closing
            is_opening = mark_as_opening
            is_closing = mark_as_closing

            if auto_detect_opening and not mark_as_opening:
                # Mark as opening if no existing odds for this game
                existing = self.db.get_odds_history(game_id, 'spread', lookback_minutes=1440)  # 24 hours
                if not existing:
                    is_opening = True

            if auto_detect_closing and not mark_as_closing:
                # Mark as closing if game starts in next 15 minutes
                try:
                    commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                    now = datetime.now(commence_time.tzinfo)
                    minutes_until_game = (commence_time - now).total_seconds() / 60

                    if 0 < minutes_until_game <= 15:
                        is_closing = True
                except (ValueError, TypeError):
                    pass  # Invalid timestamp, skip auto-detection

            # Store odds from each bookmaker
            for bookmaker in game.get('bookmakers', []):
                book_name = bookmaker.get('key', bookmaker.get('title', 'unknown'))
                markets = bookmaker.get('markets', {})

                # Store moneyline
                if 'moneyline' in markets:
                    ml = markets['moneyline']
                    self.db.insert_odds_snapshot(
                        game_id, book_name, 'moneyline',
                        {'home_odds': ml.get('home'), 'away_odds': ml.get('away')},
                        is_opening, is_closing
                    )
                    count += 1

                # Store spread
                if 'spread' in markets:
                    sp = markets['spread']
                    self.db.insert_odds_snapshot(
                        game_id, book_name, 'spread',
                        {
                            'home_line': sp.get('home_line'),
                            'away_line': sp.get('away_line'),
                            'home_odds': sp.get('home'),
                            'away_odds': sp.get('away')
                        },
                        is_opening, is_closing
                    )
                    count += 1

                # Store totals
                if 'totals' in markets:
                    tot = markets['totals']
                    self.db.insert_odds_snapshot(
                        game_id, book_name, 'totals',
                        {
                            'total': tot.get('line'),
                            'over_odds': tot.get('over'),
                            'under_odds': tot.get('under')
                        },
                        is_opening, is_closing
                    )
                    count += 1

        return count

    # =========================================================================
    # LINE MOVEMENT CALCULATIONS
    # =========================================================================

    def calculate_line_movement(self, game_id: str, market: str) -> float | None:
        """
        Calculate line movement (closing - opening).

        Args:
            game_id: Unique game identifier
            market: 'moneyline', 'spread', or 'totals'

        Returns:
            Line movement in points (or None if insufficient data)
        """
        opening = self.db.get_opening_line(game_id, market)
        closing = self.db.get_closing_line(game_id, market)

        if not opening or not closing:
            return None

        if market == 'spread':
            # Home spread movement
            open_line = opening.get('home_line')
            close_line = closing.get('home_line')
            if open_line is not None and close_line is not None:
                return close_line - open_line

        elif market == 'totals':
            # Total line movement
            open_total = opening.get('total')
            close_total = closing.get('total')
            if open_total is not None and close_total is not None:
                return close_total - open_total

        elif market == 'moneyline':
            # Convert to probability movement
            open_odds = opening.get('home_odds')
            close_odds = closing.get('home_odds')
            if open_odds is not None and close_odds is not None:
                if HAS_MARKET_MICRO:
                    open_prob = american_to_prob(open_odds)
                    close_prob = american_to_prob(close_odds)
                    return close_prob - open_prob
                # Fallback probability calculation
                open_prob = self._american_to_prob(open_odds)
                close_prob = self._american_to_prob(close_odds)
                return close_prob - open_prob

        return None

    def detect_reverse_line_movement(self, game_id: str, market: str,
                                     public_betting_pct: float | None = None) -> bool:
        """
        Detect Reverse Line Movement (RLM).

        RLM occurs when the line moves opposite to public betting percentages,
        indicating sharp money on the other side.

        Args:
            game_id: Unique game identifier
            market: 'moneyline', 'spread', or 'totals'
            public_betting_pct: Percentage of public on home team/over (0-1)
                                If None, uses heuristics based on line movement

        Returns:
            True if RLM detected
        """
        movement = self.calculate_line_movement(game_id, market)
        if movement is None:
            return False

        # If public betting data available, use it
        if public_betting_pct is not None:
            if market == 'spread':
                # RLM: Public on home, but line moves toward away (negative movement)
                if public_betting_pct > 0.6 and movement < -RLM_THRESHOLD:
                    return True
                # RLM: Public on away, but line moves toward home (positive movement)
                if public_betting_pct < 0.4 and movement > RLM_THRESHOLD:
                    return True

            elif market == 'totals':
                # RLM: Public on over, but line moves down
                if public_betting_pct > 0.6 and movement < -RLM_THRESHOLD:
                    return True
                # RLM: Public on under, but line moves up
                if public_betting_pct < 0.4 and movement > RLM_THRESHOLD:
                    return True

        # Heuristic RLM detection without public data
        # Large movement (>2 points for spread, >1.5 for total) suggests sharp action
        if market == 'spread' and abs(movement) > 2.0:
            return True
        if market == 'totals' and abs(movement) > 1.5:
            return True
        if market == 'moneyline' and abs(movement) > 0.05:  # 5% probability shift
            return True

        return False

    def detect_steam_move(self, game_id: str, market: str,
                         lookback_minutes: int = 15) -> bool:
        """
        Detect steam moves (rapid sharp money action).

        A steam move is rapid line movement (>1.5 points in <15 minutes)
        indicating coordinated sharp action.

        Args:
            game_id: Unique game identifier
            market: 'moneyline', 'spread', or 'totals'
            lookback_minutes: Time window to check (default 15 min)

        Returns:
            True if steam move detected
        """
        history = self.db.get_odds_history(game_id, market, lookback_minutes)

        if len(history) < 2:
            return False

        # Get consensus line at start and end of window
        first_snapshots = [h for h in history if
                          datetime.fromisoformat(h['timestamp']) <=
                          datetime.fromisoformat(history[0]['timestamp']) + timedelta(seconds=60)]
        last_snapshots = [h for h in history if
                         datetime.fromisoformat(h['timestamp']) >=
                         datetime.fromisoformat(history[-1]['timestamp']) - timedelta(seconds=60)]

        if not first_snapshots or not last_snapshots:
            return False

        # Calculate consensus movement
        if market == 'spread':
            first_lines = [s['home_line'] for s in first_snapshots if s.get('home_line') is not None]
            last_lines = [s['home_line'] for s in last_snapshots if s.get('home_line') is not None]

            if first_lines and last_lines:
                first_consensus = np.median(first_lines)
                last_consensus = np.median(last_lines)
                movement = abs(last_consensus - first_consensus)

                return bool(movement >= STEAM_THRESHOLD_POINTS)

        elif market == 'totals':
            first_totals = [s['total'] for s in first_snapshots if s.get('total') is not None]
            last_totals = [s['total'] for s in last_snapshots if s.get('total') is not None]

            if first_totals and last_totals:
                first_consensus = np.median(first_totals)
                last_consensus = np.median(last_totals)
                movement = abs(last_consensus - first_consensus)

                return bool(movement >= STEAM_THRESHOLD_POINTS)

        elif market == 'moneyline':
            first_odds = [s['home_odds'] for s in first_snapshots if s.get('home_odds') is not None]
            last_odds = [s['home_odds'] for s in last_snapshots if s.get('home_odds') is not None]

            if first_odds and last_odds:
                if HAS_MARKET_MICRO:
                    first_probs = [american_to_prob(o) for o in first_odds]
                    last_probs = [american_to_prob(o) for o in last_odds]
                else:
                    first_probs = [self._american_to_prob(o) for o in first_odds]
                    last_probs = [self._american_to_prob(o) for o in last_odds]

                first_consensus = np.median(first_probs)
                last_consensus = np.median(last_probs)
                movement = abs(last_consensus - first_consensus)

                return bool(movement >= STEAM_THRESHOLD_ML)

        return False

    def calculate_consensus_odds(self, game_id: str, market: str) -> dict | None:
        """
        Calculate consensus (fair) odds across multiple sportsbooks.

        Args:
            game_id: Unique game identifier
            market: 'moneyline', 'spread', or 'totals'

        Returns:
            Dictionary with consensus odds/lines, or None if insufficient data
        """
        # Get recent odds (last 5 minutes)
        history = self.db.get_odds_history(game_id, market, lookback_minutes=5)

        if len(history) < 3:  # Need at least 3 books
            return None

        # Get latest snapshot from each book
        latest_by_book = {}
        for snap in history:
            book = snap['book_name']
            if book not in latest_by_book or snap['timestamp'] > latest_by_book[book]['timestamp']:
                latest_by_book[book] = snap

        if market == 'spread':
            lines = [s['home_line'] for s in latest_by_book.values() if s.get('home_line') is not None]
            home_odds = [s['home_odds'] for s in latest_by_book.values() if s.get('home_odds') is not None]

            if lines and home_odds:
                return {
                    'consensus_line': float(np.median(lines)),
                    'consensus_odds': int(np.median(home_odds)),
                    'num_books': len(lines)
                }

        elif market == 'totals':
            totals = [s['total'] for s in latest_by_book.values() if s.get('total') is not None]
            over_odds = [s['over_odds'] for s in latest_by_book.values() if s.get('over_odds') is not None]

            if totals and over_odds:
                return {
                    'consensus_total': float(np.median(totals)),
                    'consensus_over_odds': int(np.median(over_odds)),
                    'num_books': len(totals)
                }

        elif market == 'moneyline':
            home_odds = [s['home_odds'] for s in latest_by_book.values() if s.get('home_odds') is not None]
            away_odds = [s['away_odds'] for s in latest_by_book.values() if s.get('away_odds') is not None]

            if home_odds and away_odds:
                return {
                    'consensus_home_odds': int(np.median(home_odds)),
                    'consensus_away_odds': int(np.median(away_odds)),
                    'num_books': len(home_odds)
                }

        return None

    # =========================================================================
    # FEATURE GENERATION FOR ML MODELS
    # =========================================================================

    def get_market_features(self, game_id: str, home_team: str, away_team: str) -> dict[str, Any]:
        """
        Generate all betting market features for a game.

        This is the main method to call when generating features for predictions.

        Args:
            game_id: Unique game identifier
            home_team: Home team name
            away_team: Away team name

        Returns:
            Dictionary with 6 market features:
            - opening_line: Opening spread line (home perspective)
            - closing_line: Closing spread line (home perspective)
            - line_movement: Spread movement in points (closing - opening)
            - rlm_flag: Boolean, True if RLM detected
            - consensus_odds: Consensus spread odds (integer)
            - steam_move_flag: Boolean, True if steam move detected
        """
        features = {
            'opening_line': 0.0,
            'closing_line': 0.0,
            'line_movement': 0.0,
            'rlm_flag': False,
            'consensus_odds': -110,
            'steam_move_flag': False
        }

        # Opening and closing lines (spread market)
        opening = self.db.get_opening_line(game_id, 'spread')
        closing = self.db.get_closing_line(game_id, 'spread')

        if opening:
            features['opening_line'] = opening.get('home_line', 0.0) or 0.0

        if closing:
            features['closing_line'] = closing.get('home_line', 0.0) or 0.0

        # Line movement
        movement = self.calculate_line_movement(game_id, 'spread')
        if movement is not None:
            features['line_movement'] = movement

        # RLM detection
        features['rlm_flag'] = self.detect_reverse_line_movement(game_id, 'spread')

        # Consensus odds
        consensus = self.calculate_consensus_odds(game_id, 'spread')
        if consensus:
            features['consensus_odds'] = consensus.get('consensus_odds', -110)

        # Steam move detection
        features['steam_move_flag'] = self.detect_steam_move(game_id, 'spread', lookback_minutes=15)

        return features

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _american_to_prob(odds: int) -> float:
        """Convert American odds to implied probability (fallback method)."""
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    @staticmethod
    def _prob_to_american(prob: float) -> int:
        """Convert probability to American odds (fallback method)."""
        if prob <= 0 or prob >= 1:
            return -110
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        return int(100 * (1 - prob) / prob)


# =============================================================================
# ODDS TRACKER SERVICE (BACKGROUND JOB)
# =============================================================================

class OddsTracker:
    """
    Background service to fetch and store odds at regular intervals.

    Usage with APScheduler:
        tracker = OddsTracker(update_interval_minutes=5)
        tracker.fetch_and_store_odds()
    """

    def __init__(self, api_key: str | None = None,
                 update_interval_minutes: int = 5,
                 db_path: str = DEFAULT_ODDS_DB):
        """
        Initialize odds tracker.

        Args:
            api_key: The Odds API key
            update_interval_minutes: Minutes between updates
            db_path: Path to SQLite database
        """
        self.features = BettingMarketFeatures(api_key, db_path)
        self.update_interval = update_interval_minutes
        self.last_update: datetime | None = None

    def fetch_and_store_odds(self) -> int:
        """
        Fetch current odds and store in database.

        Returns:
            Number of odds snapshots stored
        """
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching odds...")

        count = self.features.fetch_and_store_odds()
        self.last_update = datetime.now()

        print(f"Stored {count} odds snapshots")
        return count

    def should_update(self) -> bool:
        """Check if enough time has passed since last update."""
        if not self.last_update:
            return True
        elapsed = (datetime.now() - self.last_update).total_seconds() / 60
        return elapsed >= self.update_interval


# =============================================================================
# CLI AND TESTING
# =============================================================================

def test_betting_market_features():
    """Test the betting market features module."""
    print("=" * 70)
    print("BETTING MARKET FEATURES TEST")
    print("=" * 70)

    # Check API key
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        print("\nWarning: No API key found. Set THE_ODDS_API_KEY environment variable.")
        print("Some tests will be skipped.\n")

    # Initialize
    tracker = BettingMarketFeatures(api_key)

    print("\n1. DATABASE INITIALIZATION")
    print("-" * 40)
    print(f"Database path: {tracker.db.db_path}")
    print("Schema created successfully")

    if api_key:
        print("\n2. FETCH CURRENT ODDS")
        print("-" * 40)
        odds = tracker.fetch_current_odds()
        print(f"Fetched odds for {len(odds)} games")

        if odds:
            game = odds[0]
            print(f"\nSample game: {game.get('away_team')} @ {game.get('home_team')}")
            print(f"Bookmakers: {len(game.get('bookmakers', []))}")

        print("\n3. STORE ODDS IN DATABASE")
        print("-" * 40)
        count = tracker.fetch_and_store_odds()
        print(f"Stored {count} odds snapshots")

        if odds and count > 0:
            game_id = odds[0].get('game_id')

            print("\n4. CALCULATE FEATURES")
            print("-" * 40)
            features = tracker.get_market_features(
                game_id,
                odds[0].get('home_team', ''),
                odds[0].get('away_team', '')
            )
            print(f"Features for game {game_id}:")
            for key, value in features.items():
                print(f"  {key}: {value}")

            print("\n5. LINE MOVEMENT DETECTION")
            print("-" * 40)
            movement = tracker.calculate_line_movement(game_id, 'spread')
            print(f"Spread movement: {movement}")

            rlm = tracker.detect_reverse_line_movement(game_id, 'spread')
            print(f"RLM detected: {rlm}")

            steam = tracker.detect_steam_move(game_id, 'spread')
            print(f"Steam move detected: {steam}")

    print("\n6. UTILITY FUNCTIONS")
    print("-" * 40)
    test_odds = [-150, +130, -110, +200]
    for odds in test_odds:
        prob = tracker._american_to_prob(odds)
        back = tracker._prob_to_american(prob)
        print(f"  {odds:+d} -> {prob:.1%} -> {back:+d}")

    print("\n" + "=" * 70)
    print("BETTING MARKET FEATURES MODULE READY")
    print("=" * 70)


if __name__ == "__main__":
    test_betting_market_features()
