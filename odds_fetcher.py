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

import os
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
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


class OddsFetcher:
    """
    Fetches real-time NBA betting odds from The Odds API.

    The Odds API provides:
    - Pre-match and live odds
    - Historical odds
    - Odds from 40+ bookmakers
    """

    def __init__(self, api_key: Optional[str] = None):
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

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
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
            elif response.status_code == 429:
                print("Error: Rate limit exceeded")
                return None
            elif response.status_code != 200:
                print(f"Error: API returned status {response.status_code}")
                return None

            return response.json()

        except requests.exceptions.Timeout:
            print("Error: Request timed out")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed - {e}")
            return None

    def get_api_usage(self) -> Dict:
        """Get current API usage statistics."""
        return {
            "remaining_requests": self.remaining_requests,
            "used_requests": self.used_requests,
        }

    def get_nba_odds(
        self,
        markets: List[str] = None,
        bookmakers: List[str] = None,
        regions: str = "us",
    ) -> List[Dict]:
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

    def _parse_odds_response(self, data: List[Dict]) -> List[Dict]:
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

    def _parse_moneyline(self, outcomes: List[Dict], home_team: str) -> Dict:
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

    def _parse_spread(self, outcomes: List[Dict], home_team: str) -> Dict:
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

    def _parse_totals(self, outcomes: List[Dict]) -> Dict:
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
        game_odds: Dict,
        market: str,
        selection: str,
    ) -> Dict:
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

            if market == "totals":
                odds = market_data.get(selection)
            else:
                odds = market_data.get(selection)

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

    def compare_odds(self, game_odds: Dict, market: str, selection: str) -> List[Dict]:
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
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    @staticmethod
    def probability_to_odds(probability: float) -> int:
        """Convert probability to American odds."""
        if probability <= 0 or probability >= 1:
            return 0

        if probability >= 0.5:
            return int(-100 * probability / (1 - probability))
        else:
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

        ev = (model_prob * profit) - ((1 - model_prob) * stake)
        return ev


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
        self.odds_history: Dict[str, List[Dict]] = {}  # game_id -> list of snapshots
        self.opening_odds: Dict[str, Dict] = {}  # game_id -> opening odds
        self.closing_odds: Dict[str, Dict] = {}  # game_id -> closing odds

    def _ensure_storage_dir(self):
        """Create storage directory if it doesn't exist."""
        import os
        os.makedirs(self.storage_dir, exist_ok=True)

    def record_odds_snapshot(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        odds_data: Dict,
        is_opening: bool = False,
        is_closing: bool = False
    ) -> Dict:
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

    def get_opening_odds(self, game_id: str) -> Optional[Dict]:
        """Get stored opening odds for a game."""
        return self.opening_odds.get(game_id)

    def get_closing_odds(self, game_id: str) -> Optional[Dict]:
        """Get stored closing odds for a game."""
        return self.closing_odds.get(game_id)

    def calculate_line_movement(self, game_id: str) -> Optional[Dict]:
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
    ) -> Optional[float]:
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
        clv = closing_implied - our_implied

        return clv

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

    def load_history(self, game_id: str) -> Optional[List[Dict]]:
        """Load odds history from disk."""
        filepath = f"{self.storage_dir}/{game_id}_odds.json"
        try:
            with open(filepath, 'r') as f:
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
        self.pending_bets: Dict[str, Dict] = {}  # bet_id -> bet details

    def record_bet_placement(
        self,
        bet_id: str,
        game_id: str,
        home_team: str,
        away_team: str,
        bet_type: str,
        bet_selection: str,
        bet_odds: int,
        current_odds_data: Dict = None
    ) -> Dict:
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
    ) -> Optional[Dict]:
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
        for bet_id, bet in self.pending_bets.items():
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

    def get_bet_clv(self, bet_id: str) -> Optional[float]:
        """Get CLV for a specific bet."""
        bet = self.pending_bets.get(bet_id)
        if not bet:
            return None
        return bet.get("clv")

    def get_clv_summary(self) -> Dict:
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


def get_odds_for_games(api_key: Optional[str] = None) -> Dict[str, Dict]:
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
    model_predictions: List[Dict],
    api_key: Optional[str] = None,
    min_edge: float = 0.03,
) -> List[Dict]:
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
