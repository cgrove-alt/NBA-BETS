"""
Settlement Service for Continuous Learning

Automatically settles predictions with actual game results by:
1. Fetching completed game box scores from Balldontlie API
2. Extracting player stats (pts, reb, ast, fg3m)
3. Settling each prediction as hit or miss
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prop_tracker import PropTracker


class SettlementService:
    """Settles predictions with actual game results."""

    def __init__(self, prop_tracker: PropTracker = None, balldontlie_api=None):
        """Initialize settlement service.

        Args:
            prop_tracker: PropTracker instance for accessing predictions
            balldontlie_api: BalldontlieAPI instance for fetching box scores
        """
        self.prop_tracker = prop_tracker or PropTracker()
        self.api = balldontlie_api

        # Initialize API if not provided
        if self.api is None:
            try:
                from balldontlie_api import BalldontlieAPI
                self.api = BalldontlieAPI()
            except ImportError:
                print("Warning: BalldontlieAPI not available. Settlement will fail.")

    def settle_unsettled_predictions(self, game_date: str = None) -> Dict:
        """Fetch actual results and settle pending predictions.

        Args:
            game_date: Optional date filter (YYYY-MM-DD). Defaults to yesterday.

        Returns:
            Dict with settlement results: settled_count, failed_games, skipped_games
        """
        if game_date is None:
            # Default to yesterday (games need to be completed)
            game_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        unsettled = self.prop_tracker.get_unsettled_predictions(game_date)

        if not unsettled:
            return {
                "settled_count": 0,
                "game_date": game_date,
                "message": "No unsettled predictions found"
            }

        # Group by game_id
        games = {}
        for pred in unsettled:
            game_id = pred.get('game_id')
            if game_id:
                games.setdefault(game_id, []).append(pred)

        settled_count = 0
        failed_games = []
        skipped_games = []

        for game_id, predictions in games.items():
            try:
                # Try to get numeric game_id for API call
                if game_id.isdigit():
                    numeric_game_id = int(game_id)
                else:
                    # Generated game_id format: TEAM_OPP_DATE - skip these
                    skipped_games.append(game_id)
                    continue

                # Fetch box score from API
                box_score = self._fetch_box_score(numeric_game_id)

                if not box_score:
                    failed_games.append({
                        "game_id": game_id,
                        "reason": "Could not fetch box score"
                    })
                    continue

                # Extract player stats
                player_stats = self._extract_player_stats(box_score)

                if not player_stats:
                    failed_games.append({
                        "game_id": game_id,
                        "reason": "No player stats in box score"
                    })
                    continue

                # Settle each prediction for this game
                game_settled = self.prop_tracker.settle_game_predictions(
                    game_id, player_stats
                )
                settled_count += game_settled
                print(f"Settled {game_settled} predictions for game {game_id}")

            except Exception as e:
                failed_games.append({
                    "game_id": game_id,
                    "reason": str(e)
                })
                print(f"Failed to settle game {game_id}: {e}")

        return {
            "settled_count": settled_count,
            "game_date": game_date,
            "total_games": len(games),
            "failed_games": failed_games,
            "skipped_games": skipped_games,
        }

    def _fetch_box_score(self, game_id: int) -> Optional[Dict]:
        """Fetch box score from Balldontlie API.

        Args:
            game_id: Numeric game ID

        Returns:
            Box score data or None if unavailable
        """
        if not self.api:
            return None

        try:
            return self.api.get_box_score(game_id)
        except Exception as e:
            print(f"Error fetching box score for game {game_id}: {e}")
            return None

    def _extract_player_stats(self, box_score: Dict) -> Dict[int, Dict]:
        """Extract player stats from box score response.

        Args:
            box_score: Box score data from API

        Returns:
            Dict mapping player_id to their stats {pts, reb, ast, fg3m}
        """
        stats = {}

        # Handle different response formats
        data = box_score.get('data', box_score)

        if isinstance(data, list):
            players = data
        elif isinstance(data, dict):
            players = data.get('players', [])
        else:
            return stats

        for player in players:
            # Try different key formats
            player_info = player.get('player', player)
            player_id = player_info.get('id') or player.get('player_id')

            if player_id:
                stats[player_id] = {
                    'pts': player.get('pts', 0) or 0,
                    'reb': player.get('reb', 0) or 0,
                    'ast': player.get('ast', 0) or 0,
                    'fg3m': player.get('fg3m', player.get('fg3_made', 0)) or 0,
                }

        return stats

    def settle_all_pending(self, days_back: int = 7) -> Dict:
        """Settle all pending predictions from the last N days.

        Args:
            days_back: Number of days to look back

        Returns:
            Summary of all settlement attempts
        """
        results = {
            "total_settled": 0,
            "days_processed": [],
            "errors": []
        }

        for i in range(1, days_back + 1):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                day_result = self.settle_unsettled_predictions(date)
                results["total_settled"] += day_result.get("settled_count", 0)
                results["days_processed"].append({
                    "date": date,
                    "settled": day_result.get("settled_count", 0),
                    "failed": len(day_result.get("failed_games", []))
                })
            except Exception as e:
                results["errors"].append({
                    "date": date,
                    "error": str(e)
                })

        return results

    def get_settlement_status(self) -> Dict:
        """Get current settlement status.

        Returns:
            Status dict with pending counts by date
        """
        unsettled = self.prop_tracker.get_unsettled_predictions()

        # Group by date
        by_date = {}
        for pred in unsettled:
            date = pred.get('game_date', 'unknown')
            by_date[date] = by_date.get(date, 0) + 1

        return {
            "total_pending": len(unsettled),
            "by_date": by_date,
            "oldest_pending": min(by_date.keys()) if by_date else None,
        }


def settle_yesterday():
    """Convenience function to settle yesterday's predictions."""
    service = SettlementService()
    return service.settle_unsettled_predictions()


if __name__ == "__main__":
    # CLI for manual settlement
    import argparse

    parser = argparse.ArgumentParser(description="Settle prop predictions")
    parser.add_argument("--date", help="Date to settle (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Settle all pending")
    parser.add_argument("--days", type=int, default=7, help="Days to look back (with --all)")
    parser.add_argument("--status", action="store_true", help="Show settlement status")

    args = parser.parse_args()

    service = SettlementService()

    if args.status:
        status = service.get_settlement_status()
        print(f"\nSettlement Status:")
        print(f"  Total pending: {status['total_pending']}")
        print(f"  By date:")
        for date, count in sorted(status['by_date'].items()):
            print(f"    {date}: {count} predictions")
    elif args.all:
        result = service.settle_all_pending(args.days)
        print(f"\nSettled {result['total_settled']} predictions over {len(result['days_processed'])} days")
    else:
        result = service.settle_unsettled_predictions(args.date)
        print(f"\nSettlement result for {result['game_date']}:")
        print(f"  Settled: {result['settled_count']}")
        if result.get('failed_games'):
            print(f"  Failed: {len(result['failed_games'])} games")
