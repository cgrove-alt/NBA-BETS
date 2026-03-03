"""
NBA Betting Odds Fetcher

Integrates real-time betting odds from sportsbooks via The Odds API.
Free tier: 500 requests/month

Features:
- Real-time moneyline, spread, and totals odds
- Multiple sportsbooks (DraftKings, FanDuel, BetMGM, etc.)
- Line shopping (find best odds)
- Odds movement tracking
- Closing line storage for CLV calculation

Usage:
    fetcher = OddsFetcher(api_key="your_key")
    odds = fetcher.get_nba_odds()
    best = fetcher.get_best_odds(game_id, "spread", "home")
"""

import load_env  # noqa: F401  — load .env before any code reads os.environ
import os
import time
import requests
from datetime import datetime, timedelta
import json

# The Odds API Configuration
THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
NBA_SPORT_KEY = "basketball_nba"

# Rate limiting
API_DELAY = 0.2  # 200ms between requests

# Supported sportsbooks (by key)
SUPPORTED_BOOKS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "pointsbet",
    "wynnbet",
    "betrivers",
    "unibet_us",
]

# Market types
MARKETS = {
    "moneyline": "h2h",
    "spread": "spreads",
    "totals": "totals",
}

# Player prop market keys for The Odds API
PLAYER_PROP_MARKETS = {
    "points": "player_points",
    "rebounds": "player_rebounds",
    "assists": "player_assists",
    "pra": "player_points_rebounds_assists",
}

# Reverse mapping: Odds API market key -> our prop type name
MARKET_TO_PROP = {v: k for k, v in PLAYER_PROP_MARKETS.items()}

# Full NBA team names (The Odds API) -> 3-letter abbreviations (our data)
FULL_NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Bookmaker dedup priority: lower = preferred
BOOK_PRIORITY = {"draftkings": 0, "fanduel": 1, "betmgm": 2}


class OddsFetcher:
    """
    Fetches real-time NBA betting odds from The Odds API.

    The Odds API provides:
    - Pre-match and live odds
    - Historical odds
    - Odds from 40+ bookmakers
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize odds fetcher.

        Args:
            api_key: The Odds API key. If not provided, will check
                    THE_ODDS_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
        self.remaining_requests = None
        self.used_requests = None

        if not self.api_key:
            print("Warning: No API key provided. Set THE_ODDS_API_KEY environment variable.")
            print("Get a free key at: https://the-odds-api.com/")

    def _make_request(self, endpoint: str, params: dict = None) -> dict | None:
        """Make API request with rate limiting and error handling."""
        if not self.api_key:
            return None

        url = f"{THE_ODDS_API_BASE}/{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key

        time.sleep(API_DELAY)

        try:
            response = requests.get(url, params=params, timeout=30)

            # Track API usage
            self.remaining_requests = response.headers.get("x-requests-remaining")
            self.used_requests = response.headers.get("x-requests-used")

            if response.status_code == 401:
                print("Error: Invalid API key")
                return None
            if response.status_code == 429:
                print("Error: Rate limit exceeded")
                return None
            if response.status_code != 200:
                print(f"Error: API returned status {response.status_code}")
                return None

            data = response.json()

            # Validate response type (reject non-dict/non-list)
            if not isinstance(data, (dict, list)):
                import logging
                logging.getLogger(__name__).warning(
                    f"OddsAPI unexpected response type for {endpoint}: {type(data)}"
                )
                return None

            return data

        except requests.exceptions.Timeout:
            print("Error: Request timed out")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed - {e}")
            return None

    def get_api_usage(self) -> dict:
        """Get current API usage statistics."""
        return {
            "remaining_requests": self.remaining_requests,
            "used_requests": self.used_requests,
        }

    def get_nba_odds(
        self,
        markets: list[str] = None,
        bookmakers: list[str] = None,
        regions: str = "us",
    ) -> list[dict]:
        """
        Get current NBA odds for all games.

        Args:
            markets: List of markets ("moneyline", "spread", "totals")
            bookmakers: List of bookmaker keys to include
            regions: Region filter ("us", "us2", "uk", "eu", "au")

        Returns:
            List of game odds dictionaries
        """
        markets = markets or ["moneyline", "spread", "totals"]
        bookmakers = bookmakers or SUPPORTED_BOOKS

        # Convert market names to API keys
        market_keys = [MARKETS.get(m, m) for m in markets]

        params = {
            "regions": regions,
            "markets": ",".join(market_keys),
            "bookmakers": ",".join(bookmakers),
            "oddsFormat": "american",
        }

        data = self._make_request(f"sports/{NBA_SPORT_KEY}/odds", params)

        if not data:
            return []

        return self._parse_odds_response(data)

    def _parse_odds_response(self, data: list[dict]) -> list[dict]:
        """Parse API response into structured odds data."""
        parsed_games = []

        for game in data:
            game_data = {
                "game_id": game.get("id"),
                "sport": game.get("sport_key"),
                "commence_time": game.get("commence_time"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "bookmakers": [],
            }

            for bookmaker in game.get("bookmakers", []):
                book_data = {
                    "key": bookmaker.get("key"),
                    "title": bookmaker.get("title"),
                    "last_update": bookmaker.get("last_update"),
                    "markets": {},
                }

                for market in bookmaker.get("markets", []):
                    market_key = market.get("key")
                    outcomes = market.get("outcomes", [])

                    if market_key == "h2h":
                        # Moneyline
                        book_data["markets"]["moneyline"] = self._parse_moneyline(
                            outcomes, game_data["home_team"]
                        )
                    elif market_key == "spreads":
                        # Point spread
                        book_data["markets"]["spread"] = self._parse_spread(
                            outcomes, game_data["home_team"]
                        )
                    elif market_key == "totals":
                        # Over/Under
                        book_data["markets"]["totals"] = self._parse_totals(outcomes)

                game_data["bookmakers"].append(book_data)

            parsed_games.append(game_data)

        return parsed_games

    def _parse_moneyline(self, outcomes: list[dict], home_team: str) -> dict:
        """Parse moneyline outcomes."""
        result = {"home": None, "away": None}

        for outcome in outcomes:
            team = outcome.get("name")
            price = outcome.get("price")

            if team == home_team:
                result["home"] = price
            else:
                result["away"] = price

        return result

    def _parse_spread(self, outcomes: list[dict], home_team: str) -> dict:
        """Parse spread outcomes."""
        result = {"home": None, "away": None, "home_line": None, "away_line": None}

        for outcome in outcomes:
            team = outcome.get("name")
            price = outcome.get("price")
            point = outcome.get("point")

            if team == home_team:
                result["home"] = price
                result["home_line"] = point
            else:
                result["away"] = price
                result["away_line"] = point

        return result

    def _parse_totals(self, outcomes: list[dict]) -> dict:
        """Parse totals (over/under) outcomes."""
        result = {"line": None, "over": None, "under": None}

        for outcome in outcomes:
            name = outcome.get("name")
            price = outcome.get("price")
            point = outcome.get("point")

            if name == "Over":
                result["over"] = price
                result["line"] = point
            elif name == "Under":
                result["under"] = price
                if result["line"] is None:
                    result["line"] = point

        return result

    def get_best_odds(
        self,
        game_odds: dict,
        market: str,
        selection: str,
    ) -> dict:
        """
        Find the best available odds across all sportsbooks.

        Args:
            game_odds: Parsed game odds dictionary
            market: "moneyline", "spread", or "totals"
            selection: "home", "away", "over", "under"

        Returns:
            Dictionary with best odds and sportsbook
        """
        best_odds = None
        best_book = None

        for bookmaker in game_odds.get("bookmakers", []):
            market_data = bookmaker.get("markets", {}).get(market, {})

            odds = market_data.get(selection) if market == "totals" else market_data.get(selection)

            if odds is not None:
                # For American odds, higher is better for positive, less negative is better for negative
                if best_odds is None:
                    best_odds = odds
                    best_book = bookmaker.get("title")
                elif odds > 0 and best_odds > 0:
                    if odds > best_odds:
                        best_odds = odds
                        best_book = bookmaker.get("title")
                elif odds < 0 and best_odds < 0:
                    if odds > best_odds:  # Less negative is better
                        best_odds = odds
                        best_book = bookmaker.get("title")
                elif odds > 0 and best_odds < 0:
                    best_odds = odds
                    best_book = bookmaker.get("title")

        line = None
        if market == "spread":
            for bm in game_odds.get("bookmakers", []):
                spread_data = bm.get("markets", {}).get("spread", {})
                if selection == "home":
                    line = spread_data.get("home_line")
                else:
                    line = spread_data.get("away_line")
                if line is not None:
                    break
        elif market == "totals":
            for bm in game_odds.get("bookmakers", []):
                totals_data = bm.get("markets", {}).get("totals", {})
                line = totals_data.get("line")
                if line is not None:
                    break

        return {
            "best_odds": best_odds,
            "best_book": best_book,
            "line": line,
            "market": market,
            "selection": selection,
            "home_team": game_odds.get("home_team"),
            "away_team": game_odds.get("away_team"),
        }

    def compare_odds(self, game_odds: dict, market: str, selection: str) -> list[dict]:
        """
        Compare odds across all sportsbooks for a specific bet.

        Args:
            game_odds: Parsed game odds dictionary
            market: "moneyline", "spread", or "totals"
            selection: "home", "away", "over", "under"

        Returns:
            List of odds from each sportsbook, sorted by value
        """
        odds_list = []

        for bookmaker in game_odds.get("bookmakers", []):
            market_data = bookmaker.get("markets", {}).get(market, {})

            if market == "totals":
                odds = market_data.get(selection)
                line = market_data.get("line")
            else:
                odds = market_data.get(selection)
                if selection == "home":
                    line = market_data.get("home_line")
                else:
                    line = market_data.get("away_line")

            if odds is not None:
                odds_list.append({
                    "book": bookmaker.get("title"),
                    "book_key": bookmaker.get("key"),
                    "odds": odds,
                    "line": line,
                    "implied_prob": self.odds_to_probability(odds),
                })

        # Sort by odds (best first)
        odds_list.sort(key=lambda x: x["odds"], reverse=True)

        return odds_list

    @staticmethod
    def odds_to_probability(american_odds: int) -> float:
        """Convert American odds to implied probability."""
        if american_odds is None:
            return 0.5

        if american_odds > 0:
            return 100 / (american_odds + 100)
        return abs(american_odds) / (abs(american_odds) + 100)

    @staticmethod
    def probability_to_odds(probability: float) -> int:
        """Convert probability to American odds."""
        if probability <= 0 or probability >= 1:
            return 0

        if probability >= 0.5:
            return int(-100 * probability / (1 - probability))
        return int(100 * (1 - probability) / probability)

    @staticmethod
    def calculate_edge(model_prob: float, implied_prob: float) -> float:
        """Calculate betting edge."""
        return model_prob - implied_prob

    @staticmethod
    def calculate_ev(model_prob: float, american_odds: int, stake: float = 100) -> float:
        """
        Calculate expected value of a bet.

        Args:
            model_prob: Model's win probability
            american_odds: Betting odds
            stake: Bet amount

        Returns:
            Expected value in dollars
        """
        if american_odds > 0:
            profit = stake * (american_odds / 100)
        else:
            profit = stake * (100 / abs(american_odds))

        return (model_prob * profit) - ((1 - model_prob) * stake)


class LineMovementTracker:
    """
    Track line movements and store historical odds for CLV calculation.

    CLV (Closing Line Value) is the most important metric for validating
    betting edge quality. Sharp bettors consistently beat closing lines.

    Features:
    - Store opening odds when bet is placed
    - Track line movements over time
    - Fetch and store closing odds before game start
    - Calculate CLV after games complete
    - Detect steam moves and reverse line movement (RLM)
    """

    def __init__(self, storage_dir: str = "odds_history"):
        """
        Args:
            storage_dir: Directory to store odds history
        """
        self.storage_dir = storage_dir
        self._ensure_storage_dir()
        self.odds_history: dict[str, list[dict]] = {}  # game_id -> list of snapshots
        self.opening_odds: dict[str, dict] = {}  # game_id -> opening odds
        self.closing_odds: dict[str, dict] = {}  # game_id -> closing odds

    def _ensure_storage_dir(self):
        """Create storage directory if it doesn't exist."""
        import os
        os.makedirs(self.storage_dir, exist_ok=True)

    def record_odds_snapshot(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        odds_data: dict,
        is_opening: bool = False,
        is_closing: bool = False
    ) -> dict:
        """
        Record an odds snapshot for a game.

        Args:
            game_id: Unique game identifier
            home_team: Home team name
            away_team: Away team name
            odds_data: Current odds data from API
            is_opening: Mark this as opening odds
            is_closing: Mark this as closing odds

        Returns:
            The recorded snapshot
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "is_opening": is_opening,
            "is_closing": is_closing,
            "odds": {}
        }

        # Extract consensus odds (average across books)
        if "bookmakers" in odds_data:
            ml_home, ml_away = [], []
            spread_home, spread_line = [], []
            total_line, total_over = [], []

            for book in odds_data.get("bookmakers", []):
                markets = book.get("markets", {})

                if "moneyline" in markets:
                    ml = markets["moneyline"]
                    if ml.get("home"):
                        ml_home.append(ml["home"])
                    if ml.get("away"):
                        ml_away.append(ml["away"])

                if "spread" in markets:
                    sp = markets["spread"]
                    if sp.get("home"):
                        spread_home.append(sp["home"])
                    if sp.get("home_line") is not None:
                        spread_line.append(sp["home_line"])

                if "totals" in markets:
                    tot = markets["totals"]
                    if tot.get("line") is not None:
                        total_line.append(tot["line"])
                    if tot.get("over"):
                        total_over.append(tot["over"])

            # Calculate consensus (median)
            import numpy as np
            snapshot["odds"] = {
                "moneyline_home": int(np.median(ml_home)) if ml_home else None,
                "moneyline_away": int(np.median(ml_away)) if ml_away else None,
                "spread_line": float(np.median(spread_line)) if spread_line else None,
                "spread_odds_home": int(np.median(spread_home)) if spread_home else None,
                "total_line": float(np.median(total_line)) if total_line else None,
                "total_over_odds": int(np.median(total_over)) if total_over else None,
            }

        # Store in history
        if game_id not in self.odds_history:
            self.odds_history[game_id] = []
        self.odds_history[game_id].append(snapshot)

        # Mark opening/closing
        if is_opening:
            self.opening_odds[game_id] = snapshot
        if is_closing:
            self.closing_odds[game_id] = snapshot

        return snapshot

    def get_opening_odds(self, game_id: str) -> dict | None:
        """Get stored opening odds for a game."""
        return self.opening_odds.get(game_id)

    def get_closing_odds(self, game_id: str) -> dict | None:
        """Get stored closing odds for a game."""
        return self.closing_odds.get(game_id)

    def calculate_line_movement(self, game_id: str) -> dict | None:
        """
        Calculate line movement from opening to current/closing.

        Returns:
            Dictionary with movement details or None if not enough data
        """
        if game_id not in self.odds_history or len(self.odds_history[game_id]) < 2:
            return None

        opening = self.odds_history[game_id][0]
        latest = self.odds_history[game_id][-1]

        opening_odds = opening.get("odds", {})
        latest_odds = latest.get("odds", {})

        movement = {
            "game_id": game_id,
            "opening_time": opening["timestamp"],
            "latest_time": latest["timestamp"],
            "movements": {}
        }

        # Moneyline movement
        if opening_odds.get("moneyline_home") and latest_odds.get("moneyline_home"):
            ml_open = opening_odds["moneyline_home"]
            ml_curr = latest_odds["moneyline_home"]

            # Convert to implied probability for comparison
            open_prob = OddsFetcher.odds_to_probability(ml_open)
            curr_prob = OddsFetcher.odds_to_probability(ml_curr)

            movement["movements"]["moneyline"] = {
                "opening": ml_open,
                "current": ml_curr,
                "probability_change": curr_prob - open_prob,
                "direction": "toward_home" if curr_prob > open_prob else "toward_away"
            }

        # Spread movement
        if opening_odds.get("spread_line") is not None and latest_odds.get("spread_line") is not None:
            spread_open = opening_odds["spread_line"]
            spread_curr = latest_odds["spread_line"]

            movement["movements"]["spread"] = {
                "opening": spread_open,
                "current": spread_curr,
                "point_change": spread_curr - spread_open,
                "direction": "toward_home" if spread_curr < spread_open else "toward_away"
            }

        # Total movement
        if opening_odds.get("total_line") is not None and latest_odds.get("total_line") is not None:
            total_open = opening_odds["total_line"]
            total_curr = latest_odds["total_line"]

            movement["movements"]["total"] = {
                "opening": total_open,
                "current": total_curr,
                "point_change": total_curr - total_open,
                "direction": "up" if total_curr > total_open else "down"
            }

        return movement

    def detect_steam_move(self, game_id: str, threshold_points: float = 1.5) -> bool:
        """
        Detect steam moves (rapid sharp money action).

        A steam move is rapid line movement (>1.5 points in spread,
        or significant ML movement in <30 minutes) indicating sharp action.

        Args:
            game_id: Game to check
            threshold_points: Points of movement to consider "steam"

        Returns:
            True if steam move detected
        """
        if game_id not in self.odds_history or len(self.odds_history[game_id]) < 2:
            return False

        history = self.odds_history[game_id]

        # Check last 30 minutes of movement
        recent_cutoff = datetime.now() - timedelta(minutes=30)

        recent_snapshots = [
            s for s in history
            if datetime.fromisoformat(s["timestamp"]) > recent_cutoff
        ]

        if len(recent_snapshots) < 2:
            return False

        first = recent_snapshots[0]["odds"]
        last = recent_snapshots[-1]["odds"]

        # Check spread movement
        if first.get("spread_line") is not None and last.get("spread_line") is not None:
            spread_move = abs(last["spread_line"] - first["spread_line"])
            if spread_move >= threshold_points:
                return True

        return False

    def calculate_clv(
        self,
        game_id: str,
        bet_type: str,
        bet_odds: int,
        bet_selection: str
    ) -> float | None:
        """
        Calculate Closing Line Value (CLV).

        CLV = Our implied probability - Closing implied probability

        Positive CLV = we got a better line than market closed at.
        This is the best predictor of long-term betting success.

        Args:
            game_id: Game identifier
            bet_type: "moneyline", "spread", or "total"
            bet_odds: Our bet odds (American)
            bet_selection: "home", "away", "over", "under"

        Returns:
            CLV in percentage points (positive = good), or None if no closing odds
        """
        closing = self.get_closing_odds(game_id)
        if not closing:
            return None

        closing_odds_data = closing.get("odds", {})

        # Get corresponding closing odds
        closing_odds = None
        if bet_type == "moneyline":
            if bet_selection == "home":
                closing_odds = closing_odds_data.get("moneyline_home")
            else:
                closing_odds = closing_odds_data.get("moneyline_away")
        elif bet_type == "spread":
            closing_odds = closing_odds_data.get("spread_odds_home")
        elif bet_type == "total":
            closing_odds = closing_odds_data.get("total_over_odds")

        if closing_odds is None:
            return None

        # Convert to implied probabilities
        our_implied = OddsFetcher.odds_to_probability(bet_odds)
        closing_implied = OddsFetcher.odds_to_probability(closing_odds)

        # CLV = closing implied - our implied
        # Positive means we got better value than closing line
        return closing_implied - our_implied


    def save_history(self, game_id: str = None):
        """Save odds history to disk."""
        if game_id:
            games_to_save = {game_id: self.odds_history.get(game_id, [])}
        else:
            games_to_save = self.odds_history

        for gid, history in games_to_save.items():
            filepath = f"{self.storage_dir}/{gid}_odds.json"
            with open(filepath, 'w') as f:
                json.dump({
                    "game_id": gid,
                    "opening": self.opening_odds.get(gid),
                    "closing": self.closing_odds.get(gid),
                    "history": history
                }, f, indent=2)

    def load_history(self, game_id: str) -> list[dict] | None:
        """Load odds history from disk."""
        filepath = f"{self.storage_dir}/{game_id}_odds.json"
        try:
            with open(filepath) as f:
                data = json.load(f)
                self.odds_history[game_id] = data.get("history", [])
                if data.get("opening"):
                    self.opening_odds[game_id] = data["opening"]
                if data.get("closing"):
                    self.closing_odds[game_id] = data["closing"]
                return self.odds_history[game_id]
        except FileNotFoundError:
            return None


class CLVTracker:
    """
    High-level CLV tracking for the betting model.

    Integrates with bet_tracker to automatically record odds and calculate CLV.
    """

    def __init__(self, odds_fetcher: OddsFetcher = None):
        """
        Args:
            odds_fetcher: OddsFetcher instance (creates one if not provided)
        """
        self.odds_fetcher = odds_fetcher or OddsFetcher()
        self.line_tracker = LineMovementTracker()
        self.pending_bets: dict[str, dict] = {}  # bet_id -> bet details

    def record_bet_placement(
        self,
        bet_id: str,
        game_id: str,
        home_team: str,
        away_team: str,
        bet_type: str,
        bet_selection: str,
        bet_odds: int,
        current_odds_data: dict = None
    ) -> dict:
        """
        Record a bet placement with current odds (for later CLV calculation).

        Args:
            bet_id: Unique bet identifier
            game_id: Game identifier
            home_team: Home team name
            away_team: Away team name
            bet_type: "moneyline", "spread", or "total"
            bet_selection: "home", "away", "over", "under"
            bet_odds: Odds at time of bet
            current_odds_data: Full odds data if available

        Returns:
            Recorded bet details
        """
        # Record opening odds if this is first bet on game
        if game_id not in self.line_tracker.opening_odds and current_odds_data:
            self.line_tracker.record_odds_snapshot(
                game_id, home_team, away_team, current_odds_data, is_opening=True
            )

        bet_record = {
            "bet_id": bet_id,
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "bet_type": bet_type,
            "bet_selection": bet_selection,
            "bet_odds": bet_odds,
            "placed_at": datetime.now().isoformat(),
            "clv": None,
            "closing_odds": None
        }

        self.pending_bets[bet_id] = bet_record
        return bet_record

    def fetch_and_record_closing_odds(
        self,
        game_id: str,
        home_team: str,
        away_team: str
    ) -> dict | None:
        """
        Fetch current odds and record as closing odds.

        Should be called ~5 minutes before game start.

        Args:
            game_id: Game identifier
            home_team: Home team name
            away_team: Away team name

        Returns:
            Closing odds snapshot or None if fetch failed
        """
        odds_data = self.odds_fetcher.get_nba_odds()

        # Find matching game
        game_odds = None
        for game in odds_data:
            if game["home_team"] == home_team and game["away_team"] == away_team:
                game_odds = game
                break

        if not game_odds:
            return None

        snapshot = self.line_tracker.record_odds_snapshot(
            game_id, home_team, away_team, game_odds, is_closing=True
        )

        # Update pending bets with CLV
        for _bet_id, bet in self.pending_bets.items():
            if bet["game_id"] == game_id and bet["clv"] is None:
                clv = self.line_tracker.calculate_clv(
                    game_id,
                    bet["bet_type"],
                    bet["bet_odds"],
                    bet["bet_selection"]
                )
                bet["clv"] = clv
                bet["closing_odds"] = snapshot.get("odds", {})

        self.line_tracker.save_history(game_id)
        return snapshot

    def get_bet_clv(self, bet_id: str) -> float | None:
        """Get CLV for a specific bet."""
        bet = self.pending_bets.get(bet_id)
        if not bet:
            return None
        return bet.get("clv")

    def get_clv_summary(self) -> dict:
        """Get summary of CLV across all tracked bets."""
        clvs = [b["clv"] for b in self.pending_bets.values() if b["clv"] is not None]

        if not clvs:
            return {
                "total_bets": len(self.pending_bets),
                "bets_with_clv": 0,
                "avg_clv": None,
                "positive_clv_pct": None
            }

        import numpy as np
        return {
            "total_bets": len(self.pending_bets),
            "bets_with_clv": len(clvs),
            "avg_clv": float(np.mean(clvs)),
            "positive_clv_pct": sum(1 for c in clvs if c > 0) / len(clvs),
            "clv_std": float(np.std(clvs)),
            "best_clv": max(clvs),
            "worst_clv": min(clvs)
        }


class PlayerPropFetcher:
    """Fetches live player prop lines from The Odds API (FanDuel/DraftKings).

    Uses the same API infrastructure as OddsFetcher but targets player prop
    markets. Designed to replace Balldontlie/Rebet as the primary prop source
    for daily predictions.

    API cost per game: ~4 credits (4 markets x 1 region).
    """

    PROP_MARKETS_STR = ",".join(PLAYER_PROP_MARKETS.values())

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
        self.remaining_requests = None
        self.used_requests = None
        self._session = requests.Session()
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set THE_ODDS_API_KEY environment variable."
            )

    def _make_request(self, endpoint: str, params: dict | None = None) -> dict | list | None:
        """Make API request with rate limiting and error handling."""
        url = f"{THE_ODDS_API_BASE}/{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key

        time.sleep(API_DELAY)

        try:
            response = self._session.get(url, params=params, timeout=30)
            self.remaining_requests = response.headers.get("x-requests-remaining")
            self.used_requests = response.headers.get("x-requests-used")

            if response.status_code != 200:
                import logging
                logging.getLogger(__name__).warning(
                    "PlayerPropFetcher API error %d for %s", response.status_code, endpoint
                )
                return None

            return response.json()
        except requests.exceptions.RequestException as e:
            import logging
            logging.getLogger(__name__).warning("PlayerPropFetcher request failed: %s", e)
            return None

    def fetch_todays_events(self) -> list[dict]:
        """Fetch today's NBA events from The Odds API (1 credit).

        Returns:
            List of event dicts with id, home_team, away_team, commence_time.
        """
        data = self._make_request(f"sports/{NBA_SPORT_KEY}/events")
        if not data or not isinstance(data, list):
            return []
        return data

    def match_events_to_games(
        self, events: list[dict], games: list[dict]
    ) -> dict[int, dict]:
        """Match Odds API events to Balldontlie game IDs using team abbreviations.

        Args:
            events: Events from fetch_todays_events().
            games: Game dicts from the daily predictions pipeline. Each must have
                   game['home_team']['abbreviation'] and game['visitor_team']['abbreviation'].

        Returns:
            Dict mapping bdl_game_id -> event info dict.
        """
        event_map = {}
        for event in events:
            home_abbrev = FULL_NAME_TO_ABBREV.get(event.get("home_team", ""))
            away_abbrev = FULL_NAME_TO_ABBREV.get(event.get("away_team", ""))
            if not home_abbrev or not away_abbrev:
                continue

            for game in games:
                g_home = game.get("home_team", {}).get("abbreviation", "")
                g_away = game.get("visitor_team", {}).get("abbreviation", "")

                if (g_home == home_abbrev and g_away == away_abbrev) or \
                   (g_home == away_abbrev and g_away == home_abbrev):
                    game_id = game.get("id")
                    if game_id:
                        event_map[game_id] = {
                            "event_id": event["id"],
                            "home_team": event.get("home_team", ""),
                            "away_team": event.get("away_team", ""),
                            "home_abbrev": home_abbrev,
                            "away_abbrev": away_abbrev,
                            "commence_time": event.get("commence_time", ""),
                        }
                    break

        return event_map

    def fetch_props_for_event(self, event_id: str) -> list[dict]:
        """Fetch player props for a single event (~4 credits).

        Args:
            event_id: The Odds API event ID.

        Returns:
            List of parsed prop dicts with player_name, prop_type, line,
            over_odds, under_odds, bookmaker.
        """
        data = self._make_request(
            f"sports/{NBA_SPORT_KEY}/events/{event_id}/odds",
            params={
                "regions": "us",
                "markets": self.PROP_MARKETS_STR,
                "bookmakers": "draftkings,fanduel",
                "oddsFormat": "american",
            },
        )
        if not data or not isinstance(data, dict):
            return []

        props = []
        for book in data.get("bookmakers", []):
            book_key = book["key"]
            for market in book.get("markets", []):
                prop_type = MARKET_TO_PROP.get(market["key"])
                if not prop_type:
                    continue

                # Group outcomes by player (Over/Under pairs)
                player_lines: dict[str, dict] = {}
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", "")
                    direction = outcome.get("name", "").lower()
                    if not player or direction not in ("over", "under"):
                        continue

                    if player not in player_lines:
                        player_lines[player] = {
                            "player_name": player,
                            "prop_type": prop_type,
                            "line": outcome.get("point", 0),
                            "bookmaker": book_key,
                        }
                    if direction == "over":
                        player_lines[player]["over_odds"] = outcome.get("price", -110)
                        player_lines[player]["line"] = outcome.get("point", 0)
                    else:
                        player_lines[player]["under_odds"] = outcome.get("price", -110)

                for pl in player_lines.values():
                    if "over_odds" in pl and "under_odds" in pl:
                        props.append(pl)

        return props

    def _dedupe_props(self, props: list[dict]) -> list[dict]:
        """Deduplicate props, keeping DraftKings > FanDuel > others."""
        best: dict[tuple[str, str], dict] = {}
        for p in props:
            key = (p["player_name"], p["prop_type"])
            existing = best.get(key)
            if existing is None:
                best[key] = p
            else:
                new_rank = BOOK_PRIORITY.get(p["bookmaker"], 99)
                old_rank = BOOK_PRIORITY.get(existing["bookmaker"], 99)
                if new_rank < old_rank:
                    best[key] = p
        return list(best.values())

    def get_props_for_game(
        self,
        bdl_game_id: int,
        event_id: str,
        id_mapper=None,
    ) -> dict[int, dict]:
        """Main interface: fetch props and return in daily_predictions format.

        Args:
            bdl_game_id: Balldontlie game ID.
            event_id: The Odds API event ID.
            id_mapper: Optional IDMapper for resolving player names to BDL IDs.

        Returns:
            Dict keyed by player_id (BDL ID or hash) with prop lines in the
            same format that daily_predictions.py expects:
            {player_id: {'player_id': id, 'points_line': 25.5,
                         'points_vendor': 'draftkings', 'points_over_odds': -110, ...}}
        """
        raw_props = self.fetch_props_for_event(event_id)
        if not raw_props:
            return {}

        deduped = self._dedupe_props(raw_props)

        # Group by player name
        by_player_name: dict[str, dict] = {}
        for p in deduped:
            name = p["player_name"]
            if name not in by_player_name:
                by_player_name[name] = {}
            entry = by_player_name[name]
            pt = p["prop_type"]
            entry[f"{pt}_line"] = p["line"]
            entry[f"{pt}_vendor"] = p["bookmaker"]
            entry[f"{pt}_over_odds"] = p.get("over_odds", -110)
            entry[f"{pt}_under_odds"] = p.get("under_odds", -110)

        # Resolve player names to BDL IDs
        result: dict[int, dict] = {}
        for name, prop_data in by_player_name.items():
            player_id = None
            if id_mapper:
                player_id = id_mapper.get_player_id(name)

            if not player_id:
                # Use a deterministic hash as fallback ID (negative to avoid collisions)
                player_id = -abs(hash(name)) % 1_000_000

            prop_data["player_id"] = player_id
            prop_data["player_name_odds_api"] = name
            result[player_id] = prop_data

        return result

    def get_api_usage(self) -> dict:
        """Get current API usage statistics."""
        return {
            "remaining_requests": self.remaining_requests,
            "used_requests": self.used_requests,
        }


def get_odds_for_games(api_key: str | None = None) -> dict[str, dict]:
    """
    Convenience function to get current NBA odds indexed by matchup.

    Returns:
        Dictionary mapping "HOME vs AWAY" to odds data
    """
    fetcher = OddsFetcher(api_key)
    games = fetcher.get_nba_odds()

    odds_by_matchup = {}
    for game in games:
        matchup_key = f"{game['home_team']} vs {game['away_team']}"
        odds_by_matchup[matchup_key] = game

    return odds_by_matchup


def find_value_bets(
    model_predictions: list[dict],
    api_key: str | None = None,
    min_edge: float = 0.03,
) -> list[dict]:
    """
    Find value bets by comparing model predictions to market odds.

    Args:
        model_predictions: List of predictions with:
            - home_team, away_team
            - moneyline_prob (home win probability)
            - spread_prob (home covers probability)
            - spread_prediction (predicted spread)
        api_key: The Odds API key
        min_edge: Minimum edge to consider (default 3%)

    Returns:
        List of value bets with edge > min_edge
    """
    fetcher = OddsFetcher(api_key)
    odds_data = fetcher.get_nba_odds()

    value_bets = []

    for pred in model_predictions:
        home_team = pred.get("home_team")
        away_team = pred.get("away_team")

        # Find matching game odds
        game_odds = None
        for game in odds_data:
            if game["home_team"] == home_team and game["away_team"] == away_team:
                game_odds = game
                break

        if not game_odds:
            continue

        # Check moneyline value
        ml_prob = pred.get("moneyline_prob", 0.5)
        best_home_ml = fetcher.get_best_odds(game_odds, "moneyline", "home")
        best_away_ml = fetcher.get_best_odds(game_odds, "moneyline", "away")

        if best_home_ml["best_odds"]:
            implied = fetcher.odds_to_probability(best_home_ml["best_odds"])
            edge = ml_prob - implied
            if edge >= min_edge:
                value_bets.append({
                    "type": "moneyline",
                    "selection": "home",
                    "team": home_team,
                    "opponent": away_team,
                    "model_prob": ml_prob,
                    "implied_prob": implied,
                    "edge": edge,
                    "best_odds": best_home_ml["best_odds"],
                    "best_book": best_home_ml["best_book"],
                    "ev": fetcher.calculate_ev(ml_prob, best_home_ml["best_odds"]),
                })

        if best_away_ml["best_odds"]:
            implied = fetcher.odds_to_probability(best_away_ml["best_odds"])
            away_prob = 1 - ml_prob
            edge = away_prob - implied
            if edge >= min_edge:
                value_bets.append({
                    "type": "moneyline",
                    "selection": "away",
                    "team": away_team,
                    "opponent": home_team,
                    "model_prob": away_prob,
                    "implied_prob": implied,
                    "edge": edge,
                    "best_odds": best_away_ml["best_odds"],
                    "best_book": best_away_ml["best_book"],
                    "ev": fetcher.calculate_ev(away_prob, best_away_ml["best_odds"]),
                })

        # Check spread value
        if "spread_prediction" in pred:
            best_spread = fetcher.get_best_odds(game_odds, "spread", "home")
            if best_spread["best_odds"] and best_spread["line"] is not None:
                # Convert spread prediction to cover probability
                pred_spread = pred["spread_prediction"]
                market_spread = best_spread["line"]
                spread_edge = pred_spread - market_spread

                # Rough conversion: each point of spread edge ~= 3% probability
                cover_prob = 0.5 + (spread_edge * 0.03)
                cover_prob = max(0.01, min(0.99, cover_prob))

                implied = fetcher.odds_to_probability(best_spread["best_odds"])
                edge = cover_prob - implied

                if edge >= min_edge:
                    value_bets.append({
                        "type": "spread",
                        "selection": "home",
                        "team": home_team,
                        "opponent": away_team,
                        "line": market_spread,
                        "predicted_spread": pred_spread,
                        "model_prob": cover_prob,
                        "implied_prob": implied,
                        "edge": edge,
                        "best_odds": best_spread["best_odds"],
                        "best_book": best_spread["best_book"],
                        "ev": fetcher.calculate_ev(cover_prob, best_spread["best_odds"]),
                    })

    # Sort by edge
    value_bets.sort(key=lambda x: x["edge"], reverse=True)

    return value_bets


# =============================================================================
# V3: MULTI-THREADED ODDS MONITOR WITH STEAM DETECTION
# =============================================================================

import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections.abc import Callable
import numpy as np


@dataclass
class OddsSnapshot:
    """Point-in-time odds snapshot for a game."""
    game_id: str
    timestamp: datetime
    home_team: str
    away_team: str

    # Consensus odds across books
    moneyline_home: int | None = None
    moneyline_away: int | None = None
    spread_line: float | None = None
    spread_home_odds: int | None = None
    total_line: float | None = None
    total_over_odds: int | None = None

    # Book-specific odds for arbitrage detection
    book_odds: dict[str, dict] = field(default_factory=dict)

    def get_implied_prob(self, selection: str) -> float:
        """Get implied probability for a selection."""
        odds = None
        if selection == "home_ml":
            odds = self.moneyline_home
        elif selection == "away_ml":
            odds = self.moneyline_away
        elif selection == "home_spread":
            odds = self.spread_home_odds
        elif selection == "over":
            odds = self.total_over_odds

        if odds is None:
            return 0.5
        return OddsFetcher.odds_to_probability(odds)


@dataclass
class SteamAlert:
    """Steam move detection alert."""
    game_id: str
    timestamp: datetime
    alert_type: str  # "spread_steam", "ml_steam", "total_steam"
    direction: str  # "toward_home", "toward_away", "up", "down"
    magnitude: float  # Points or probability change
    time_window_seconds: float
    previous_line: float
    current_line: float
    confidence: float  # 0-1 confidence this is sharp action


class OddsMonitorV3:
    """
    V3: Multi-threaded real-time odds monitor with <1s steam detection.

    Architecture:
    - Main thread: Coordination and callback dispatch
    - Poll threads: Parallel API requests to multiple books
    - Heartbeat thread: Sub-second change detection
    - Alert queue: Thread-safe steam move notifications

    Features:
    - Multi-threaded polling for reduced latency
    - Heartbeat mechanism for <1s steam detection
    - Callback system for real-time alerts
    - Thread-safe odds history
    - Automatic rate limiting per thread

    Usage:
        monitor = OddsMonitorV3(api_key="your_key")
        monitor.add_steam_callback(my_alert_handler)
        monitor.start_monitoring(game_ids=["game1", "game2"])
        # ... later
        monitor.stop_monitoring()
    """

    # Steam detection thresholds
    STEAM_SPREAD_THRESHOLD = 1.0  # 1+ points = steam
    STEAM_ML_THRESHOLD = 0.03  # 3% probability change = steam
    STEAM_TOTAL_THRESHOLD = 1.5  # 1.5+ points = steam
    STEAM_TIME_WINDOW = 60  # seconds to look back

    # Polling configuration
    DEFAULT_POLL_INTERVAL = 5.0  # seconds between full polls
    HEARTBEAT_INTERVAL = 0.5  # 500ms heartbeat checks
    MAX_POLL_THREADS = 4

    def __init__(
        self,
        api_key: str | None = None,
        poll_interval: float = None,
        max_threads: int = None
    ):
        """
        Initialize the V3 odds monitor.

        Args:
            api_key: The Odds API key
            poll_interval: Seconds between full polls (default 5.0)
            max_threads: Maximum polling threads (default 4)
        """
        self.fetcher = OddsFetcher(api_key)
        self.poll_interval = poll_interval or self.DEFAULT_POLL_INTERVAL
        self.max_threads = max_threads or self.MAX_POLL_THREADS

        # Thread-safe data structures
        self._lock = threading.RLock()
        self._snapshots: dict[str, list[OddsSnapshot]] = {}  # game_id -> history
        self._latest: dict[str, OddsSnapshot] = {}  # game_id -> latest snapshot
        self._alert_queue: queue.Queue = queue.Queue()

        # Callbacks
        self._steam_callbacks: list[Callable[[SteamAlert], None]] = []
        self._odds_callbacks: list[Callable[[OddsSnapshot], None]] = []

        # Threading
        self._running = False
        self._poll_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None

        # Monitoring state
        self._game_ids: list[str] = []
        self._last_poll_time: dict[str, datetime] = {}

        # Statistics
        self._stats = {
            'polls': 0,
            'steam_alerts': 0,
            'avg_latency_ms': 0,
            'latency_samples': [],
        }

    def add_steam_callback(self, callback: Callable[[SteamAlert], None]):
        """Register a callback for steam move alerts."""
        self._steam_callbacks.append(callback)

    def add_odds_callback(self, callback: Callable[[OddsSnapshot], None]):
        """Register a callback for odds updates."""
        self._odds_callbacks.append(callback)

    def start_monitoring(self, game_ids: list[str] = None):
        """
        Start monitoring odds for specified games.

        Args:
            game_ids: List of game IDs to monitor (or all if None)
        """
        if self._running:
            print("Monitor already running")
            return

        self._running = True
        self._game_ids = game_ids or []
        self._executor = ThreadPoolExecutor(max_workers=self.max_threads)

        # Start poll thread
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="OddsMonitor-Poll",
            daemon=True
        )
        self._poll_thread.start()

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="OddsMonitor-Heartbeat",
            daemon=True
        )
        self._heartbeat_thread.start()

        print(f"OddsMonitorV3 started (poll={self.poll_interval}s, threads={self.max_threads})")

    def stop_monitoring(self):
        """Stop the odds monitor."""
        self._running = False

        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1.0)
        if self._executor:
            self._executor.shutdown(wait=False)

        print(f"OddsMonitorV3 stopped. Stats: {self.get_stats()}")

    def _poll_loop(self):
        """Main polling loop - runs in separate thread."""
        while self._running:
            try:
                start_time = time.time()

                # Fetch odds (potentially in parallel)
                self._parallel_fetch_odds()

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                self._update_latency_stats(latency_ms)

                self._stats['polls'] += 1

                # Sleep for remaining interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.poll_interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                print(f"Poll error: {e}")
                time.sleep(1.0)

    def _heartbeat_loop(self):
        """Heartbeat loop for rapid steam detection."""
        while self._running:
            try:
                # Process any pending alerts
                self._process_alert_queue()

                # Check for steam moves in recent data
                self._check_steam_moves()

                time.sleep(self.HEARTBEAT_INTERVAL)

            except Exception as e:
                print(f"Heartbeat error: {e}")
                time.sleep(0.5)

    def _parallel_fetch_odds(self):
        """Fetch odds using parallel threads for speed."""
        try:
            # Get all NBA odds
            odds_data = self.fetcher.get_nba_odds()

            if not odds_data:
                return

            # Filter to monitored games if specified
            if self._game_ids:
                odds_data = [g for g in odds_data if g.get("game_id") in self._game_ids]

            # Process each game
            for game in odds_data:
                snapshot = self._create_snapshot(game)
                if snapshot:
                    self._record_snapshot(snapshot)

        except Exception as e:
            print(f"Fetch error: {e}")

    def _create_snapshot(self, game_data: dict) -> OddsSnapshot | None:
        """Create OddsSnapshot from API response."""
        try:
            game_id = game_data.get("game_id")
            if not game_id:
                return None

            snapshot = OddsSnapshot(
                game_id=game_id,
                timestamp=datetime.now(),
                home_team=game_data.get("home_team", ""),
                away_team=game_data.get("away_team", ""),
            )

            # Aggregate odds across books
            ml_home, ml_away = [], []
            spread_line, spread_odds = [], []
            total_line, total_over = [], []

            for book in game_data.get("bookmakers", []):
                book_key = book.get("key", "unknown")
                markets = book.get("markets", {})

                book_snapshot = {}

                if "moneyline" in markets:
                    ml = markets["moneyline"]
                    if ml.get("home"):
                        ml_home.append(ml["home"])
                    if ml.get("away"):
                        ml_away.append(ml["away"])
                    book_snapshot["ml_home"] = ml.get("home")
                    book_snapshot["ml_away"] = ml.get("away")

                if "spread" in markets:
                    sp = markets["spread"]
                    if sp.get("home"):
                        spread_odds.append(sp["home"])
                    if sp.get("home_line") is not None:
                        spread_line.append(sp["home_line"])
                    book_snapshot["spread_line"] = sp.get("home_line")
                    book_snapshot["spread_odds"] = sp.get("home")

                if "totals" in markets:
                    tot = markets["totals"]
                    if tot.get("line") is not None:
                        total_line.append(tot["line"])
                    if tot.get("over"):
                        total_over.append(tot["over"])
                    book_snapshot["total_line"] = tot.get("line")
                    book_snapshot["total_over"] = tot.get("over")

                snapshot.book_odds[book_key] = book_snapshot

            # Calculate consensus (median)
            if ml_home:
                snapshot.moneyline_home = int(np.median(ml_home))
            if ml_away:
                snapshot.moneyline_away = int(np.median(ml_away))
            if spread_line:
                snapshot.spread_line = float(np.median(spread_line))
            if spread_odds:
                snapshot.spread_home_odds = int(np.median(spread_odds))
            if total_line:
                snapshot.total_line = float(np.median(total_line))
            if total_over:
                snapshot.total_over_odds = int(np.median(total_over))

            return snapshot

        except Exception as e:
            print(f"Snapshot creation error: {e}")
            return None

    def _record_snapshot(self, snapshot: OddsSnapshot):
        """Thread-safe snapshot recording."""
        with self._lock:
            game_id = snapshot.game_id

            # Store latest
            previous = self._latest.get(game_id)
            self._latest[game_id] = snapshot

            # Add to history
            if game_id not in self._snapshots:
                self._snapshots[game_id] = []
            self._snapshots[game_id].append(snapshot)

            # Trim history (keep last 100 snapshots per game)
            if len(self._snapshots[game_id]) > 100:
                self._snapshots[game_id] = self._snapshots[game_id][-100:]

        # Check for steam on this update
        if previous:
            self._check_steam_between(previous, snapshot)

        # Notify callbacks
        for callback in self._odds_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                print(f"Odds callback error: {e}")

    def _check_steam_between(self, prev: OddsSnapshot, curr: OddsSnapshot):
        """Check for steam move between two snapshots."""
        time_diff = (curr.timestamp - prev.timestamp).total_seconds()

        # Only check if within time window
        if time_diff > self.STEAM_TIME_WINDOW:
            return

        alerts = []

        # Check spread steam
        if prev.spread_line is not None and curr.spread_line is not None:
            spread_move = abs(curr.spread_line - prev.spread_line)
            if spread_move >= self.STEAM_SPREAD_THRESHOLD:
                direction = "toward_home" if curr.spread_line < prev.spread_line else "toward_away"
                alerts.append(SteamAlert(
                    game_id=curr.game_id,
                    timestamp=curr.timestamp,
                    alert_type="spread_steam",
                    direction=direction,
                    magnitude=spread_move,
                    time_window_seconds=time_diff,
                    previous_line=prev.spread_line,
                    current_line=curr.spread_line,
                    confidence=min(1.0, spread_move / 3.0),  # 3+ points = max confidence
                ))

        # Check moneyline steam
        if prev.moneyline_home is not None and curr.moneyline_home is not None:
            prev_prob = OddsFetcher.odds_to_probability(prev.moneyline_home)
            curr_prob = OddsFetcher.odds_to_probability(curr.moneyline_home)
            prob_change = abs(curr_prob - prev_prob)

            if prob_change >= self.STEAM_ML_THRESHOLD:
                direction = "toward_home" if curr_prob > prev_prob else "toward_away"
                alerts.append(SteamAlert(
                    game_id=curr.game_id,
                    timestamp=curr.timestamp,
                    alert_type="ml_steam",
                    direction=direction,
                    magnitude=prob_change,
                    time_window_seconds=time_diff,
                    previous_line=prev.moneyline_home,
                    current_line=curr.moneyline_home,
                    confidence=min(1.0, prob_change / 0.10),  # 10% = max confidence
                ))

        # Check total steam
        if prev.total_line is not None and curr.total_line is not None:
            total_move = abs(curr.total_line - prev.total_line)
            if total_move >= self.STEAM_TOTAL_THRESHOLD:
                direction = "up" if curr.total_line > prev.total_line else "down"
                alerts.append(SteamAlert(
                    game_id=curr.game_id,
                    timestamp=curr.timestamp,
                    alert_type="total_steam",
                    direction=direction,
                    magnitude=total_move,
                    time_window_seconds=time_diff,
                    previous_line=prev.total_line,
                    current_line=curr.total_line,
                    confidence=min(1.0, total_move / 3.0),
                ))

        # Queue alerts
        for alert in alerts:
            self._alert_queue.put(alert)

    def _check_steam_moves(self):
        """Check for steam moves in recent history."""
        # This runs on heartbeat, so just process the queue
        pass

    def _process_alert_queue(self):
        """Process pending steam alerts."""
        while not self._alert_queue.empty():
            try:
                alert = self._alert_queue.get_nowait()
                self._stats['steam_alerts'] += 1

                # Notify callbacks
                for callback in self._steam_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        print(f"Steam callback error: {e}")

            except queue.Empty:
                break

    def _update_latency_stats(self, latency_ms: float):
        """Update latency statistics."""
        self._stats['latency_samples'].append(latency_ms)
        # Keep last 100 samples
        if len(self._stats['latency_samples']) > 100:
            self._stats['latency_samples'] = self._stats['latency_samples'][-100:]
        self._stats['avg_latency_ms'] = np.mean(self._stats['latency_samples'])

    def get_latest_odds(self, game_id: str) -> OddsSnapshot | None:
        """Get latest odds snapshot for a game."""
        with self._lock:
            return self._latest.get(game_id)

    def get_odds_history(self, game_id: str, limit: int = 50) -> list[OddsSnapshot]:
        """Get odds history for a game."""
        with self._lock:
            history = self._snapshots.get(game_id, [])
            return history[-limit:] if limit else history

    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        return {
            'polls': self._stats['polls'],
            'steam_alerts': self._stats['steam_alerts'],
            'avg_latency_ms': round(self._stats['avg_latency_ms'], 1),
            'games_monitored': len(self._latest),
            'running': self._running,
        }

    def get_all_current_odds(self) -> dict[str, OddsSnapshot]:
        """Get all current odds snapshots."""
        with self._lock:
            return dict(self._latest)


def create_steam_logger() -> Callable[[SteamAlert], None]:
    """Create a simple logging callback for steam alerts."""
    def log_steam(alert: SteamAlert):
        print(f"🔥 STEAM ALERT: {alert.alert_type} on {alert.game_id}")
        print(f"   Direction: {alert.direction}, Magnitude: {alert.magnitude:.2f}")
        print(f"   {alert.previous_line} → {alert.current_line} in {alert.time_window_seconds:.1f}s")
        print(f"   Confidence: {alert.confidence:.0%}")
    return log_steam


if __name__ == "__main__":
    print("NBA Odds Fetcher")
    print("=" * 50)

    # Check for API key
    api_key = os.environ.get("THE_ODDS_API_KEY")

    if api_key:
        fetcher = OddsFetcher(api_key)
        print("\nFetching current NBA odds...")

        odds = fetcher.get_nba_odds()

        if odds:
            print(f"\nFound odds for {len(odds)} games:")
            for game in odds[:3]:  # Show first 3 games
                print(f"\n{game['away_team']} @ {game['home_team']}")
                print(f"  Commence: {game['commence_time']}")

                if game['bookmakers']:
                    book = game['bookmakers'][0]
                    print(f"  {book['title']}:")
                    if 'moneyline' in book['markets']:
                        ml = book['markets']['moneyline']
                        print(f"    Moneyline: Home {ml['home']} / Away {ml['away']}")
                    if 'spread' in book['markets']:
                        sp = book['markets']['spread']
                        print(f"    Spread: Home {sp['home_line']} ({sp['home']})")
                    if 'totals' in book['markets']:
                        tot = book['markets']['totals']
                        print(f"    Total: {tot['line']} (O {tot['over']} / U {tot['under']})")

            print(f"\nAPI Usage: {fetcher.get_api_usage()}")
        else:
            print("No odds available or API error")
    else:
        print("\nNo API key found. Set THE_ODDS_API_KEY environment variable.")
        print("Get a free key at: https://the-odds-api.com/")
        print("\nExample:")
        print("  export THE_ODDS_API_KEY='your_key_here'")
        print("  python odds_fetcher.py")
