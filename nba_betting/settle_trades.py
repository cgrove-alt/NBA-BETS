"""Settlement task for paper trading forward validation.

Fetches actual player stats for yesterday's games and grades
all paper trade predictions against actual outcomes.

Designed to run daily via Railway scheduler, after games complete.
"""

import re
import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# prop_type (as stored in paper_trades) → BDL stat field name
# All aliases for the same stat collapse to the same canonical key so that
# predictions logged with "pts" and those logged with "points" both settle.
_PROP_STAT_MAP = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "3pm": "fg3m",   # Alternative name used by some logging paths
}

# Short-form aliases used by the model pipeline → canonical prop_type names
# used as keys in actual_stats.  Ensures "pts" in paper_trades matches the
# "points" key built from _PROP_STAT_MAP above.
_PROP_TYPE_ALIASES: dict[str, str] = {
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "fg3m": "threes",
    "3pm": "threes",
    # canonical names map to themselves (idempotent)
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "threes": "threes",
    "pra": "pra",
}


def _normalize_prop_type(prop_type: str) -> str:
    """Canonical prop_type for settlement lookups.

    Maps short-form aliases ("pts", "reb", "ast", "fg3m") to the canonical
    names used as keys in actual_stats ("points", "rebounds", etc.).
    Unknown prop_types are returned lowercased as-is.
    """
    return _PROP_TYPE_ALIASES.get((prop_type or "").lower(), (prop_type or "").lower())


# Tokens to strip when normalizing player names for matching
_SUFFIX_RE = re.compile(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)$', re.IGNORECASE)


def _normalize_player_name(name: str) -> str:
    """Canonical form of a player name for fuzzy matching during settlement.

    Steps:
    1. Lowercase + strip outer whitespace
    2. Remove Jr./Sr./II/III/IV/V suffixes (BDL sometimes includes, props APIs omit)
    3. Remove all non-alphanumeric-space chars (periods, apostrophes, hyphens)
    4. Collapse internal whitespace

    >>> _normalize_player_name("LeBron James")
    'lebron james'
    >>> _normalize_player_name("Marcus Morris Sr.")
    'marcus morris'
    >>> _normalize_player_name("Jaren Jackson Jr.")
    'jaren jackson'
    """
    name = name.lower().strip()
    name = _SUFFIX_RE.sub("", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = " ".join(name.split())
    return name


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

    # Only settle Final games — match "Final", "Final - OT", etc.
    game_ids = [
        g["id"] for g in games
        if "final" in str(g.get("status", "")).lower()
    ]
    if not game_ids:
        logger.info(
            "No completed games for %s (%d games found, statuses: %s)",
            game_date, len(games),
            [g.get("status") for g in games[:5]],
        )
        return {}

    # Fetch box-score stats for every completed game — paginate to get ALL players
    # (a typical NBA night with 5+ games has 130+ stat lines, exceeding the 100-row
    # default page size of get_player_stats).
    if hasattr(api, "get_player_stats_for_games"):
        stats = api.get_player_stats_for_games(game_ids=game_ids)
    else:
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
        # Normalized name used as the lookup key — handles Jr./Sr./punctuation
        # differences between BDL ("P.J. Washington") and the Odds API / paper_trades
        # ("PJ Washington").  _normalize_player_name strips periods, suffixes, and
        # collapses whitespace so both forms become "pj washington".
        norm_name = _normalize_player_name(player_name)

        # Map each prop type to its stat value.
        # Store under BOTH the canonical long-form name ("points") and the short-form
        # alias ("pts") so that paper_trades rows logged with either format settle
        # correctly without requiring a DB migration.
        for prop_type, stat_key in _PROP_STAT_MAP.items():
            val = stat_line.get(stat_key)
            if val is not None:
                fval = float(val)
                actual[(norm_name, prop_type)] = fval
                # Also index by short alias so "pts" in paper_trades finds "points" data
                alias = _PROP_TYPE_ALIASES.get(stat_key)
                if alias and alias != prop_type:
                    actual[(norm_name, alias)] = fval
                # And by the raw BDL field name (e.g., "pts") as a direct alias
                actual[(norm_name, stat_key)] = fval

        # PRA (points + rebounds + assists) — a combined prop
        pts = stat_line.get("pts") or 0
        reb = stat_line.get("reb") or 0
        ast = stat_line.get("ast") or 0
        actual[(norm_name, "pra")] = float(pts + reb + ast)

    unique_players = len({name for (name, _) in actual.keys()})
    logger.info(
        "Fetched stats for %d player-prop combos (%d players) across %d games on %s",
        len(actual), unique_players, len(game_ids), game_date,
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
