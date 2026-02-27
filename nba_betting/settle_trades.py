"""Settlement task for paper trading forward validation.

Fetches actual player stats for yesterday's games and grades
all paper trade predictions against actual outcomes.

Designed to run daily via Railway scheduler, after games complete.
"""

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# BDL stat field → prop_type mapping used by paper trading
_PROP_STAT_MAP = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
}


def _fetch_actual_stats(game_date: str) -> dict:
    """Fetch actual player stats for all games on a given date.

    Uses the BallDontLie API to pull box-score stats, then maps
    each player's stats into the (player_name, prop_type) → value
    format expected by PaperTrader.settle_trades().

    Args:
        game_date: Date string in YYYY-MM-DD format.

    Returns:
        Dict mapping (player_name, prop_type) to the actual numeric value.
        Returns empty dict if the API is unavailable or no games were played.
    """
    try:
        from nba_data.sources.balldontlie_api import BalldontlieAPI
    except ImportError:
        logger.warning("BalldontlieAPI not available — cannot fetch stats")
        return {}

    api = BalldontlieAPI()

    # Fetch games on that date
    games = api.get_games(dates=[game_date])
    if not games:
        logger.info("No games found for %s", game_date)
        return {}

    # Only settle Final games
    game_ids = [
        g["id"] for g in games
        if str(g.get("status", "")).lower() == "final"
    ]
    if not game_ids:
        logger.info("No completed games for %s (%d games found)", game_date, len(games))
        return {}

    # Fetch box-score stats for every completed game
    stats = api.get_player_stats(game_ids=game_ids)
    if not stats:
        logger.warning("No player stats returned for %d games on %s", len(game_ids), game_date)
        return {}

    actual: dict = {}
    for stat_line in stats:
        player_data = stat_line.get("player", {})
        first = player_data.get("first_name", "")
        last = player_data.get("last_name", "")
        if not first or not last:
            continue
        player_name = f"{first} {last}"

        # Map each prop type to its stat value
        for prop_type, stat_key in _PROP_STAT_MAP.items():
            val = stat_line.get(stat_key)
            if val is not None:
                actual[(player_name, prop_type)] = float(val)

        # PRA (points + rebounds + assists) — a combined prop
        pts = stat_line.get("pts") or 0
        reb = stat_line.get("reb") or 0
        ast = stat_line.get("ast") or 0
        actual[(player_name, "pra")] = float(pts + reb + ast)

    logger.info(
        "Fetched stats for %d player-prop combos across %d games on %s",
        len(actual), len(game_ids), game_date,
    )
    return actual


def settle_date(game_date: str) -> int:
    """Settle paper trades for a specific date.

    Fetches actual player stats, then calls PaperTrader.settle_trades()
    to grade predictions and calculate P&L.

    Args:
        game_date: Date string in YYYY-MM-DD format.

    Returns:
        Number of trades settled.
    """
    actual_stats = _fetch_actual_stats(game_date)
    if not actual_stats:
        logger.info("No actual stats to settle for %s", game_date)
        return 0

    from nba_betting.paper_trading import PaperTrader

    trader = PaperTrader()
    settled = trader.settle_trades(game_date, actual_stats=actual_stats)
    logger.info("Settled %d paper trades for %s", settled, game_date)
    return settled


def settle_yesterday() -> int:
    """Settle paper trades for yesterday's games.

    Returns:
        Number of trades settled.
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    return settle_date(yesterday)


def main():
    """Entry point for Railway scheduler.

    Settles yesterday's paper trades and logs the result.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        count = settle_yesterday()
        logger.info("Settlement complete: %d trades settled", count)
    except Exception as e:
        logger.error("Settlement failed: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
