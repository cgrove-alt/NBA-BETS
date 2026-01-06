"""
NBA Market Microstructure Analysis

Real-time monitoring of betting line movements to identify:
- Steam moves (sharp money moving lines rapidly)
- Stale lines (books slow to adjust)
- Consensus pricing (fair market odds)

=============================================================================
KEY CONCEPTS
=============================================================================
Steam Move: When sharp books (Pinnacle, Circa) move first and soft books
    (DraftKings, FanDuel) haven't caught up. This creates arbitrage-like
    opportunities at the laggard books.

Stale Line: A book offering significantly different odds than consensus,
    indicating they haven't incorporated recent information.

Leader vs Laggard Books:
    - Leaders: Pinnacle, Circa, Bookmaker (lowest vig, sharp action)
    - Laggards: DraftKings, FanDuel, BetMGM, Caesars (recreational books)

CLV (Closing Line Value): The difference between the odds you bet at and
    where the line closes. Positive CLV = beating the market.
=============================================================================
"""

import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import threading
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

# Book classification by sharpness
SHARP_BOOKS = ['pinnacle', 'circa', 'bookmaker', 'betcris', 'bet365']
SOFT_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet', 'barstool']
ALL_BOOKS = SHARP_BOOKS + SOFT_BOOKS

# Detection thresholds
STEAM_MOVE_THRESHOLD = 0.03  # 3% probability shift in short time
STEAM_TIME_WINDOW = 300  # 5 minutes
STALE_LINE_THRESHOLD = 0.025  # 2.5% diff from consensus
MIN_BOOKS_FOR_CONSENSUS = 3  # Minimum books to calculate consensus

# Rate limiting
MIN_POLL_INTERVAL = 60  # Minimum seconds between polls


class BookType(Enum):
    SHARP = "sharp"
    SOFT = "soft"
    UNKNOWN = "unknown"


@dataclass
class OddsSnapshot:
    """A snapshot of odds from a single book at a point in time."""
    timestamp: float
    book: str
    game_id: str

    # Moneyline
    home_ml: Optional[int] = None  # American odds
    away_ml: Optional[int] = None

    # Spread
    home_spread: Optional[float] = None
    home_spread_odds: Optional[int] = None
    away_spread: Optional[float] = None
    away_spread_odds: Optional[int] = None

    # Total
    total_line: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None

    def get_implied_prob(self, market: str, side: str) -> Optional[float]:
        """Convert American odds to implied probability."""
        odds = None

        if market == 'moneyline':
            odds = self.home_ml if side == 'home' else self.away_ml
        elif market == 'spread':
            odds = self.home_spread_odds if side == 'home' else self.away_spread_odds
        elif market == 'total':
            odds = self.over_odds if side == 'over' else self.under_odds

        if odds is None:
            return None

        return american_to_prob(odds)


@dataclass
class SteamAlert:
    """Alert for a detected steam move."""
    timestamp: float
    game_id: str
    market: str  # 'moneyline', 'spread', 'total'
    side: str  # 'home', 'away', 'over', 'under'
    direction: str  # 'up' or 'down' (probability movement)

    # Movement details
    leader_book: str
    leader_move: float  # Probability change at leader
    leader_current_prob: float

    # Laggard opportunities
    laggard_books: List[Dict] = field(default_factory=list)  # [{book, odds, edge}]

    # Confidence
    confidence: float = 0.0  # 0-1, how confident in the steam move

    def __str__(self):
        lag_str = ", ".join([f"{l['book']}({l['edge']:.1%})" for l in self.laggard_books[:3]])
        return (
            f"STEAM: {self.game_id} {self.market} {self.side} {self.direction} | "
            f"Leader: {self.leader_book} moved {self.leader_move:.1%} | "
            f"Laggards: {lag_str}"
        )


@dataclass
class StaleLine:
    """A stale line identified at a specific book."""
    timestamp: float
    game_id: str
    book: str
    market: str
    side: str

    book_odds: int
    book_implied_prob: float
    consensus_prob: float
    edge: float  # book_prob - consensus_prob (positive = +EV at book)

    def __str__(self):
        return (
            f"STALE: {self.book} on {self.game_id} {self.market} {self.side} | "
            f"Edge: {self.edge:.1%} | Book: {self.book_odds} vs Consensus: {self.consensus_prob:.1%}"
        )


# =============================================================================
# ODDS UTILITIES
# =============================================================================

def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability (with vig)."""
    if odds is None:
        return 0.5
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def prob_to_american(prob: float) -> int:
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return -110  # Default
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)


def remove_vig(prob1: float, prob2: float) -> Tuple[float, float]:
    """
    Remove vig from two-way market probabilities.

    Returns fair probabilities that sum to 1.
    """
    total = prob1 + prob2
    if total <= 0:
        return (0.5, 0.5)
    return (prob1 / total, prob2 / total)


def calculate_consensus(odds_list: List[Tuple[str, float]]) -> Tuple[float, float]:
    """
    Calculate consensus (fair) probability from multiple books.

    Uses inverse-vig weighted average, giving more weight to sharp books.
    """
    if not odds_list:
        return (0.5, 0.5)

    sharp_probs = []
    soft_probs = []

    for book, prob in odds_list:
        if book.lower() in SHARP_BOOKS:
            sharp_probs.append(prob)
        else:
            soft_probs.append(prob)

    # Weight sharp books 2x
    all_probs = sharp_probs * 2 + soft_probs

    if not all_probs:
        return (0.5, 0.5)

    avg_prob = sum(all_probs) / len(all_probs)
    return (avg_prob, 1 - avg_prob)


def calculate_edge(my_prob: float, market_prob: float) -> float:
    """
    Calculate edge as probability difference.

    Positive edge = my probability > market probability = +EV
    """
    return my_prob - market_prob


# =============================================================================
# ODDS FETCHER
# =============================================================================

class OddsFetcher:
    """
    Robust odds fetcher with rate limiting and caching.

    Integrates with Balldontlie API (GOAT tier) for live odds.
    """

    def __init__(self, api_client=None, cache_dir: str = ".odds_cache"):
        """
        Initialize the odds fetcher.

        Args:
            api_client: BalldontlieAPI instance (optional, will create if None)
            cache_dir: Directory for caching odds snapshots
        """
        self.api = api_client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.last_fetch_time = 0
        self.rate_limit_remaining = 600  # GOAT tier limit
        self.rate_limit_reset = 0

        # Historical snapshots for movement analysis
        self.snapshots: Dict[str, List[OddsSnapshot]] = defaultdict(list)
        self.max_snapshot_age = 3600 * 4  # Keep 4 hours of history

        # Lock for thread safety
        self._lock = threading.Lock()

    def _init_api(self):
        """Lazy initialization of API client."""
        if self.api is None:
            try:
                from balldontlie_api import BalldontlieAPI
                self.api = BalldontlieAPI()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize API: {e}")

    def fetch_odds(self, date: str = None, force: bool = False) -> List[Dict]:
        """
        Fetch current betting odds.

        Args:
            date: Date string (YYYY-MM-DD), defaults to today
            force: Force fetch even if recently fetched

        Returns:
            List of game odds dictionaries
        """
        self._init_api()

        # Rate limiting
        now = time.time()
        if not force and (now - self.last_fetch_time) < MIN_POLL_INTERVAL:
            # Return cached data
            return self._load_cache(date)

        try:
            odds_data = self.api.get_betting_odds(date=date)
            self.last_fetch_time = now

            # Store snapshots
            self._store_snapshots(odds_data)

            # Cache to disk
            self._save_cache(date, odds_data)

            return odds_data

        except Exception as e:
            print(f"Error fetching odds: {e}")
            return self._load_cache(date)

    def _store_snapshots(self, odds_data: List[Dict]):
        """Store odds snapshots for historical analysis."""
        now = time.time()

        with self._lock:
            for game in odds_data:
                game_id = str(game.get('game_id', game.get('id', '')))

                for book_data in game.get('odds', []):
                    snapshot = self._parse_book_odds(game_id, book_data, now)
                    if snapshot:
                        self.snapshots[game_id].append(snapshot)

            # Clean old snapshots
            cutoff = now - self.max_snapshot_age
            for game_id in list(self.snapshots.keys()):
                self.snapshots[game_id] = [
                    s for s in self.snapshots[game_id]
                    if s.timestamp > cutoff
                ]
                if not self.snapshots[game_id]:
                    del self.snapshots[game_id]

    def _parse_book_odds(
        self,
        game_id: str,
        book_data: Dict,
        timestamp: float
    ) -> Optional[OddsSnapshot]:
        """Parse book odds into OddsSnapshot."""
        book = book_data.get('book', book_data.get('sportsbook', '')).lower()
        if not book:
            return None

        return OddsSnapshot(
            timestamp=timestamp,
            book=book,
            game_id=game_id,
            home_ml=book_data.get('home_ml', book_data.get('home_moneyline')),
            away_ml=book_data.get('away_ml', book_data.get('away_moneyline')),
            home_spread=book_data.get('home_spread'),
            home_spread_odds=book_data.get('home_spread_odds'),
            away_spread=book_data.get('away_spread'),
            away_spread_odds=book_data.get('away_spread_odds'),
            total_line=book_data.get('total', book_data.get('over_under')),
            over_odds=book_data.get('over_odds'),
            under_odds=book_data.get('under_odds'),
        )

    def get_historical_snapshots(
        self,
        game_id: str,
        lookback_seconds: int = 3600
    ) -> List[OddsSnapshot]:
        """Get historical snapshots for a game."""
        cutoff = time.time() - lookback_seconds
        with self._lock:
            return [s for s in self.snapshots.get(game_id, []) if s.timestamp > cutoff]

    def _save_cache(self, date: str, data: List[Dict]):
        """Save odds to cache file."""
        date = date or datetime.now().strftime('%Y-%m-%d')
        cache_file = self.cache_dir / f"odds_{date}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'date': date,
                    'data': data
                }, f)
        except IOError:
            pass

    def _load_cache(self, date: str = None) -> List[Dict]:
        """Load odds from cache file."""
        date = date or datetime.now().strftime('%Y-%m-%d')
        cache_file = self.cache_dir / f"odds_{date}.json"
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    return cached.get('data', [])
        except (IOError, json.JSONDecodeError):
            pass
        return []


# =============================================================================
# STEAM DETECTOR
# =============================================================================

class SteamDetector:
    """
    Detects steam moves - rapid line movements at sharp books.

    Steam moves indicate sharp money moving the market. Betting at
    laggard books before they adjust can be profitable.
    """

    def __init__(
        self,
        odds_fetcher: OddsFetcher,
        move_threshold: float = STEAM_MOVE_THRESHOLD,
        time_window: int = STEAM_TIME_WINDOW
    ):
        """
        Initialize steam detector.

        Args:
            odds_fetcher: OddsFetcher instance
            move_threshold: Minimum probability move to trigger alert (default 3%)
            time_window: Time window in seconds to detect move (default 5 min)
        """
        self.fetcher = odds_fetcher
        self.move_threshold = move_threshold
        self.time_window = time_window

        # Track detected steam moves to avoid duplicates
        self.detected_moves: Dict[str, float] = {}  # key -> timestamp
        self.duplicate_window = 300  # 5 minutes

    def detect(self, game_id: str = None) -> List[SteamAlert]:
        """
        Detect steam moves.

        Args:
            game_id: Specific game to check, or None for all games

        Returns:
            List of SteamAlert objects
        """
        alerts = []
        now = time.time()

        if game_id:
            game_ids = [game_id]
        else:
            game_ids = list(self.fetcher.snapshots.keys())

        for gid in game_ids:
            snapshots = self.fetcher.get_historical_snapshots(gid, self.time_window * 2)
            if len(snapshots) < 2:
                continue

            # Group by book
            by_book: Dict[str, List[OddsSnapshot]] = defaultdict(list)
            for s in snapshots:
                by_book[s.book].append(s)

            # Check each market
            for market in ['moneyline', 'spread', 'total']:
                for side in self._get_sides(market):
                    alert = self._check_market(gid, by_book, market, side, now)
                    if alert:
                        alerts.append(alert)

        return alerts

    def _get_sides(self, market: str) -> List[str]:
        """Get sides for a market."""
        if market in ['moneyline', 'spread']:
            return ['home', 'away']
        else:
            return ['over', 'under']

    def _check_market(
        self,
        game_id: str,
        by_book: Dict[str, List[OddsSnapshot]],
        market: str,
        side: str,
        now: float
    ) -> Optional[SteamAlert]:
        """Check a specific market for steam moves."""

        # Find sharp book movements
        sharp_moves = []
        for book in SHARP_BOOKS:
            if book not in by_book:
                continue

            snaps = sorted(by_book[book], key=lambda x: x.timestamp)
            if len(snaps) < 2:
                continue

            # Check for recent movement
            recent = [s for s in snaps if s.timestamp > now - self.time_window]
            if len(recent) < 2:
                continue

            first_prob = recent[0].get_implied_prob(market, side)
            last_prob = recent[-1].get_implied_prob(market, side)

            if first_prob is None or last_prob is None:
                continue

            move = last_prob - first_prob
            if abs(move) >= self.move_threshold:
                sharp_moves.append({
                    'book': book,
                    'move': move,
                    'current_prob': last_prob,
                    'timestamp': recent[-1].timestamp
                })

        if not sharp_moves:
            return None

        # Find the largest sharp move
        leader = max(sharp_moves, key=lambda x: abs(x['move']))

        # Check for duplicate
        alert_key = f"{game_id}_{market}_{side}_{leader['book']}"
        if alert_key in self.detected_moves:
            if now - self.detected_moves[alert_key] < self.duplicate_window:
                return None
        self.detected_moves[alert_key] = now

        # Find laggard books with edge
        laggards = []
        direction = 'up' if leader['move'] > 0 else 'down'

        for book in SOFT_BOOKS:
            if book not in by_book:
                continue

            snaps = by_book[book]
            if not snaps:
                continue

            latest = max(snaps, key=lambda x: x.timestamp)
            prob = latest.get_implied_prob(market, side)

            if prob is None:
                continue

            # Calculate edge (difference from where sharp book moved to)
            if direction == 'up':
                # Sharp thinks probability is higher, so betting at lower prob is +EV
                edge = leader['current_prob'] - prob
            else:
                # Sharp thinks probability is lower, so betting against at higher prob is +EV
                edge = prob - leader['current_prob']

            if edge > 0.01:  # Minimum 1% edge
                # Get actual odds for display
                odds = self._get_odds_for_side(latest, market, side)
                laggards.append({
                    'book': book,
                    'odds': odds,
                    'prob': prob,
                    'edge': edge
                })

        if not laggards:
            return None

        # Sort by edge
        laggards.sort(key=lambda x: x['edge'], reverse=True)

        return SteamAlert(
            timestamp=now,
            game_id=game_id,
            market=market,
            side=side,
            direction=direction,
            leader_book=leader['book'],
            leader_move=leader['move'],
            leader_current_prob=leader['current_prob'],
            laggard_books=laggards,
            confidence=min(1.0, abs(leader['move']) / 0.05)  # Max confidence at 5% move
        )

    def _get_odds_for_side(
        self,
        snapshot: OddsSnapshot,
        market: str,
        side: str
    ) -> Optional[int]:
        """Get American odds for a specific market/side."""
        if market == 'moneyline':
            return snapshot.home_ml if side == 'home' else snapshot.away_ml
        elif market == 'spread':
            return snapshot.home_spread_odds if side == 'home' else snapshot.away_spread_odds
        elif market == 'total':
            return snapshot.over_odds if side == 'over' else snapshot.under_odds
        return None


# =============================================================================
# STALE LINE FINDER
# =============================================================================

class StaleLineFinder:
    """
    Identifies stale lines at individual sportsbooks.

    A stale line is one that differs significantly from the market
    consensus, indicating the book hasn't adjusted to recent info.
    """

    def __init__(
        self,
        odds_fetcher: OddsFetcher,
        threshold: float = STALE_LINE_THRESHOLD,
        min_books: int = MIN_BOOKS_FOR_CONSENSUS
    ):
        """
        Initialize stale line finder.

        Args:
            odds_fetcher: OddsFetcher instance
            threshold: Minimum edge to report (default 2.5%)
            min_books: Minimum books needed for consensus (default 3)
        """
        self.fetcher = odds_fetcher
        self.threshold = threshold
        self.min_books = min_books

    def find_stale_lines(self, game_id: str = None) -> List[StaleLine]:
        """
        Find stale lines across all games or a specific game.

        Returns:
            List of StaleLine objects
        """
        stale_lines = []
        now = time.time()

        # Get current odds
        odds_data = self.fetcher.fetch_odds()

        for game in odds_data:
            gid = str(game.get('game_id', game.get('id', '')))

            if game_id and gid != game_id:
                continue

            book_odds = game.get('odds', [])
            if len(book_odds) < self.min_books:
                continue

            # Check each market
            for market in ['moneyline', 'spread', 'total']:
                for side in ['home', 'away'] if market != 'total' else ['over', 'under']:
                    stale = self._check_for_stale(gid, book_odds, market, side, now)
                    stale_lines.extend(stale)

        return stale_lines

    def _check_for_stale(
        self,
        game_id: str,
        book_odds: List[Dict],
        market: str,
        side: str,
        now: float
    ) -> List[StaleLine]:
        """Check for stale lines in a specific market."""
        stale = []

        # Collect probabilities from all books
        probs = []
        book_probs = {}

        for book_data in book_odds:
            book = book_data.get('book', book_data.get('sportsbook', '')).lower()
            odds = self._get_odds(book_data, market, side)

            if odds is None:
                continue

            prob = american_to_prob(odds)
            probs.append((book, prob))
            book_probs[book] = {'prob': prob, 'odds': odds}

        if len(probs) < self.min_books:
            return stale

        # Calculate consensus
        consensus_prob, _ = calculate_consensus(probs)

        # Find outliers
        for book, data in book_probs.items():
            edge = data['prob'] - consensus_prob

            # For under/away, flip the edge interpretation
            if side in ['away', 'under']:
                edge = -edge

            if abs(edge) >= self.threshold:
                stale.append(StaleLine(
                    timestamp=now,
                    game_id=game_id,
                    book=book,
                    market=market,
                    side=side,
                    book_odds=data['odds'],
                    book_implied_prob=data['prob'],
                    consensus_prob=consensus_prob,
                    edge=abs(edge)
                ))

        return stale

    def _get_odds(self, book_data: Dict, market: str, side: str) -> Optional[int]:
        """Extract odds for a market/side from book data."""
        if market == 'moneyline':
            key = 'home_ml' if side == 'home' else 'away_ml'
            return book_data.get(key, book_data.get(f'{side}_moneyline'))
        elif market == 'spread':
            key = f'{side}_spread_odds'
            return book_data.get(key)
        elif market == 'total':
            key = f'{side}_odds'
            return book_data.get(key)
        return None


# =============================================================================
# MARKET MONITOR
# =============================================================================

class MarketMonitor:
    """
    Combined market monitoring system.

    Continuously monitors odds and triggers alerts for:
    - Steam moves
    - Stale lines
    - Significant line movements
    """

    def __init__(self, api_client=None, poll_interval: int = 60):
        """
        Initialize market monitor.

        Args:
            api_client: BalldontlieAPI instance (optional)
            poll_interval: Seconds between polls (default 60)
        """
        self.fetcher = OddsFetcher(api_client)
        self.steam_detector = SteamDetector(self.fetcher)
        self.stale_finder = StaleLineFinder(self.fetcher)

        self.poll_interval = max(MIN_POLL_INTERVAL, poll_interval)
        self.running = False
        self._thread = None

        # Alert callbacks
        self.steam_callbacks = []
        self.stale_callbacks = []

    def on_steam(self, callback):
        """Register callback for steam alerts."""
        self.steam_callbacks.append(callback)

    def on_stale(self, callback):
        """Register callback for stale line alerts."""
        self.stale_callbacks.append(callback)

    def check_once(self) -> Dict[str, List]:
        """
        Perform a single check for opportunities.

        Returns:
            Dict with 'steam' and 'stale' lists
        """
        # Fetch latest odds
        self.fetcher.fetch_odds(force=True)

        # Detect opportunities
        steam_alerts = self.steam_detector.detect()
        stale_lines = self.stale_finder.find_stale_lines()

        # Trigger callbacks
        for alert in steam_alerts:
            for cb in self.steam_callbacks:
                try:
                    cb(alert)
                except Exception as e:
                    print(f"Steam callback error: {e}")

        for stale in stale_lines:
            for cb in self.stale_callbacks:
                try:
                    cb(stale)
                except Exception as e:
                    print(f"Stale callback error: {e}")

        return {
            'steam': steam_alerts,
            'stale': stale_lines
        }

    def start(self):
        """Start continuous monitoring in background thread."""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self.check_once()
            except Exception as e:
                print(f"Monitor error: {e}")

            time.sleep(self.poll_interval)


# =============================================================================
# CONSENSUS CALCULATOR
# =============================================================================

class ConsensusCalculator:
    """
    Calculate fair (no-vig) consensus odds from multiple books.

    Useful for:
    - Setting model targets
    - Identifying edges
    - CLV analysis
    """

    def __init__(self, odds_fetcher: OddsFetcher):
        self.fetcher = odds_fetcher

    def calculate_game_consensus(self, game_id: str) -> Dict[str, Dict]:
        """
        Calculate consensus probabilities for a game.

        Returns:
            Dictionary with consensus for each market
        """
        snapshots = self.fetcher.get_historical_snapshots(game_id, lookback_seconds=300)
        if not snapshots:
            return {}

        # Get latest snapshot from each book
        latest_by_book = {}
        for s in snapshots:
            if s.book not in latest_by_book or s.timestamp > latest_by_book[s.book].timestamp:
                latest_by_book[s.book] = s

        result = {}

        # Moneyline
        ml_home_probs = []
        ml_away_probs = []
        for book, snap in latest_by_book.items():
            if snap.home_ml and snap.away_ml:
                h_prob = american_to_prob(snap.home_ml)
                a_prob = american_to_prob(snap.away_ml)
                # Remove vig
                h_fair, a_fair = remove_vig(h_prob, a_prob)
                ml_home_probs.append((book, h_fair))
                ml_away_probs.append((book, a_fair))

        if ml_home_probs:
            home_cons, away_cons = calculate_consensus(ml_home_probs)
            result['moneyline'] = {
                'home_prob': home_cons,
                'away_prob': away_cons,
                'home_fair_odds': prob_to_american(home_cons),
                'away_fair_odds': prob_to_american(away_cons),
            }

        # Spread
        spread_probs = []
        spread_line = None
        for book, snap in latest_by_book.items():
            if snap.home_spread is not None and snap.home_spread_odds:
                spread_line = snap.home_spread
                h_prob = american_to_prob(snap.home_spread_odds)
                a_prob = american_to_prob(snap.away_spread_odds) if snap.away_spread_odds else 1 - h_prob
                h_fair, a_fair = remove_vig(h_prob, a_prob)
                spread_probs.append((book, h_fair))

        if spread_probs:
            home_cons, _ = calculate_consensus(spread_probs)
            result['spread'] = {
                'line': spread_line,
                'home_cover_prob': home_cons,
                'away_cover_prob': 1 - home_cons,
                'home_fair_odds': prob_to_american(home_cons),
                'away_fair_odds': prob_to_american(1 - home_cons),
            }

        # Total
        total_probs = []
        total_line = None
        for book, snap in latest_by_book.items():
            if snap.total_line is not None and snap.over_odds:
                total_line = snap.total_line
                o_prob = american_to_prob(snap.over_odds)
                u_prob = american_to_prob(snap.under_odds) if snap.under_odds else 1 - o_prob
                o_fair, u_fair = remove_vig(o_prob, u_prob)
                total_probs.append((book, o_fair))

        if total_probs:
            over_cons, _ = calculate_consensus(total_probs)
            result['total'] = {
                'line': total_line,
                'over_prob': over_cons,
                'under_prob': 1 - over_cons,
                'over_fair_odds': prob_to_american(over_cons),
                'under_fair_odds': prob_to_american(1 - over_cons),
            }

        return result


# =============================================================================
# CLV TRACKER
# =============================================================================

class CLVTracker:
    """
    Track Closing Line Value for placed bets.

    CLV is the gold standard metric for bet quality.
    Consistently beating the closing line = long-term profitability.
    """

    def __init__(self, odds_fetcher: OddsFetcher):
        self.fetcher = odds_fetcher
        self.tracked_bets: List[Dict] = []

    def track_bet(
        self,
        game_id: str,
        market: str,
        side: str,
        odds_at_bet: int,
        bet_time: float = None
    ) -> str:
        """
        Start tracking a bet for CLV.

        Returns:
            Bet tracking ID
        """
        bet_id = hashlib.md5(f"{game_id}_{market}_{side}_{time.time()}".encode()).hexdigest()[:8]

        self.tracked_bets.append({
            'id': bet_id,
            'game_id': game_id,
            'market': market,
            'side': side,
            'odds_at_bet': odds_at_bet,
            'prob_at_bet': american_to_prob(odds_at_bet),
            'bet_time': bet_time or time.time(),
            'closing_odds': None,
            'closing_prob': None,
            'clv': None,
        })

        return bet_id

    def update_closing_lines(self):
        """Update closing lines for tracked bets."""
        consensus = ConsensusCalculator(self.fetcher)

        for bet in self.tracked_bets:
            if bet['clv'] is not None:
                continue  # Already calculated

            game_consensus = consensus.calculate_game_consensus(bet['game_id'])

            if bet['market'] not in game_consensus:
                continue

            market_data = game_consensus[bet['market']]

            if bet['market'] == 'moneyline':
                key = f"{bet['side']}_prob"
            elif bet['market'] == 'spread':
                key = f"{bet['side']}_cover_prob"
            elif bet['market'] == 'total':
                key = f"{bet['side']}_prob"
            else:
                continue

            if key not in market_data:
                continue

            bet['closing_prob'] = market_data[key]
            bet['closing_odds'] = prob_to_american(bet['closing_prob'])

            # CLV = probability we got - closing probability
            # Positive CLV = we got better odds than closing line
            bet['clv'] = bet['prob_at_bet'] - bet['closing_prob']

    def get_clv_summary(self) -> Dict:
        """Get summary of CLV across all tracked bets."""
        self.update_closing_lines()

        completed = [b for b in self.tracked_bets if b['clv'] is not None]

        if not completed:
            return {'avg_clv': 0, 'positive_rate': 0, 'n_bets': 0}

        clvs = [b['clv'] for b in completed]

        return {
            'avg_clv': sum(clvs) / len(clvs),
            'positive_rate': sum(1 for c in clvs if c > 0) / len(clvs),
            'n_bets': len(completed),
            'total_clv': sum(clvs),
        }


# =============================================================================
# DEMO
# =============================================================================

def demo_market_microstructure():
    """Demonstrate market microstructure analysis."""
    print("=" * 70)
    print("NBA MARKET MICROSTRUCTURE ANALYSIS")
    print("=" * 70)

    # Initialize (will use cached data if API not available)
    fetcher = OddsFetcher()

    print("\n1. ODDS UTILITIES DEMO")
    print("-" * 40)

    # Convert odds
    test_odds = [-150, +130, -110, +200]
    for odds in test_odds:
        prob = american_to_prob(odds)
        back = prob_to_american(prob)
        print(f"  {odds:+d} -> {prob:.1%} implied -> {back:+d} converted back")

    # Remove vig example
    home_prob = american_to_prob(-110)
    away_prob = american_to_prob(-110)
    fair_home, fair_away = remove_vig(home_prob, away_prob)
    print(f"\n  Vig removal: -110/-110 -> {home_prob:.1%}/{away_prob:.1%} -> {fair_home:.1%}/{fair_away:.1%} fair")

    print("\n2. STEAM DETECTOR CONFIG")
    print("-" * 40)
    print(f"  Move threshold: {STEAM_MOVE_THRESHOLD:.1%}")
    print(f"  Time window: {STEAM_TIME_WINDOW}s")
    print(f"  Sharp books: {', '.join(SHARP_BOOKS)}")
    print(f"  Soft books: {', '.join(SOFT_BOOKS)}")

    print("\n3. STALE LINE FINDER CONFIG")
    print("-" * 40)
    print(f"  Edge threshold: {STALE_LINE_THRESHOLD:.1%}")
    print(f"  Min books for consensus: {MIN_BOOKS_FOR_CONSENSUS}")

    print("\n4. EXAMPLE ALERT OBJECTS")
    print("-" * 40)

    # Example steam alert
    steam = SteamAlert(
        timestamp=time.time(),
        game_id="12345",
        market="spread",
        side="home",
        direction="up",
        leader_book="pinnacle",
        leader_move=0.04,
        leader_current_prob=0.55,
        laggard_books=[
            {'book': 'draftkings', 'odds': -105, 'edge': 0.03},
            {'book': 'fanduel', 'odds': -108, 'edge': 0.025},
        ],
        confidence=0.8
    )
    print(f"  {steam}")

    # Example stale line
    stale = StaleLine(
        timestamp=time.time(),
        game_id="12345",
        book="caesars",
        market="moneyline",
        side="home",
        book_odds=-120,
        book_implied_prob=0.545,
        consensus_prob=0.52,
        edge=0.025
    )
    print(f"  {stale}")

    print("\n5. MARKET MONITOR USAGE")
    print("-" * 40)
    print("""
    # Example usage:
    monitor = MarketMonitor()

    # Register callbacks
    monitor.on_steam(lambda alert: print(f"STEAM: {alert}"))
    monitor.on_stale(lambda stale: print(f"STALE: {stale}"))

    # Single check
    results = monitor.check_once()
    print(f"Found {len(results['steam'])} steam moves")
    print(f"Found {len(results['stale'])} stale lines")

    # Or continuous monitoring
    monitor.start()
    # ... let run ...
    monitor.stop()
    """)

    print("\nMarket microstructure module ready!")
    return True


if __name__ == "__main__":
    demo_market_microstructure()
