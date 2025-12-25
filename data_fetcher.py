"""
NBA Data Fetcher

Fetches NBA schedules, historical game data, team statistics, and player stats.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

try:
    from nba_api.stats.endpoints import (
        scoreboardv2,
        leaguegamefinder,
        teamdashboardbygeneralsplits,
        playergamelog,
        commonteamroster,
        teamgamelog,
        playerdashboardbygeneralsplits,
        leaguedashteamstats,
        commonplayerinfo,
    )
    from nba_api.stats.static import teams, players
except ImportError:
    print("Error: nba_api package not installed.")
    print("Install it with: pip install nba_api")
    exit(1)

# Rate limiting to avoid API throttling
API_DELAY = 0.4  # seconds between API calls (reduced from 0.6 for faster props loading)


class ThreadSafeRateLimiter:
    """Thread-safe rate limiter that allows concurrent requests while respecting API limits."""

    def __init__(self, min_interval: float = 0.4):
        """
        Initialize rate limiter.

        Args:
            min_interval: Minimum seconds between API calls (globally coordinated)
        """
        self._lock = threading.Lock()
        self._last_call_time = 0.0
        self._min_interval = min_interval

    def wait(self):
        """Wait until it's safe to make another API call."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                time.sleep(sleep_time)
            self._last_call_time = time.time()


# Global rate limiter instance
_rate_limiter = ThreadSafeRateLimiter(API_DELAY)


def fetch_todays_schedule():
    """Fetch today's NBA schedule from the NBA API."""
    today = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching NBA schedule for {today}...")

    scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
    games_data = scoreboard.get_normalized_dict()

    return games_data, today


def parse_game_details(games_data):
    """Parse the raw API response to extract relevant game details."""
    games = []

    game_header = games_data.get("GameHeader", [])
    line_score = games_data.get("LineScore", [])

    # Build team ID to info lookup from nba_api teams
    nba_teams = teams.get_teams()
    team_id_lookup = {team['id']: team for team in nba_teams}

    # Create a lookup for team scores by game ID (for live/completed games)
    team_scores = {}
    for team in line_score:
        game_id = team.get("GAME_ID")
        if game_id not in team_scores:
            team_scores[game_id] = []
        team_scores[game_id].append({
            "team_id": team.get("TEAM_ID"),
            "team_abbreviation": team.get("TEAM_ABBREVIATION"),
            "team_city": team.get("TEAM_CITY_NAME"),
            "team_name": team.get("TEAM_NAME"),
            "pts": team.get("PTS"),
        })

    for game in game_header:
        game_id = game.get("GAME_ID")
        game_status = game.get("GAME_STATUS_TEXT", "")
        game_time = game.get("GAME_DATE_EST", "")

        home_team_id = game.get("HOME_TEAM_ID")
        visitor_team_id = game.get("VISITOR_TEAM_ID")

        # Get team details from line score (for live/completed games)
        home_team = None
        visitor_team = None

        for team in team_scores.get(game_id, []):
            if team["team_id"] == home_team_id:
                home_team = team
            elif team["team_id"] == visitor_team_id:
                visitor_team = team

        # Fallback to nba_api team lookup if LineScore is empty (games not started)
        home_team_static = team_id_lookup.get(home_team_id, {})
        visitor_team_static = team_id_lookup.get(visitor_team_id, {})

        game_info = {
            "game_id": game_id,
            "status": game_status,
            "game_time": game_time,
            "arena": game.get("ARENA_NAME"),
            "home_team": {
                "id": home_team_id,
                "abbreviation": home_team["team_abbreviation"] if home_team else home_team_static.get("abbreviation"),
                "city": home_team["team_city"] if home_team else home_team_static.get("city"),
                "name": home_team["team_name"] if home_team else home_team_static.get("nickname"),
                "score": home_team["pts"] if home_team else None,
            },
            "visitor_team": {
                "id": visitor_team_id,
                "abbreviation": visitor_team["team_abbreviation"] if visitor_team else visitor_team_static.get("abbreviation"),
                "city": visitor_team["team_city"] if visitor_team else visitor_team_static.get("city"),
                "name": visitor_team["team_name"] if visitor_team else visitor_team_static.get("nickname"),
                "score": visitor_team["pts"] if visitor_team else None,
            },
            "live_period": game.get("LIVE_PERIOD"),
            "live_pc_time": game.get("LIVE_PC_TIME"),
            "natl_tv_broadcaster": game.get("NATL_TV_BROADCASTER_ABBREVIATION"),
        }

        games.append(game_info)

    return games


def get_team_id(team_name_or_abbrev):
    """Get team ID from team name or abbreviation."""
    nba_teams = teams.get_teams()
    for team in nba_teams:
        if (team_name_or_abbrev.upper() == team['abbreviation'] or
            team_name_or_abbrev.lower() in team['full_name'].lower() or
            team_name_or_abbrev.lower() == team['nickname'].lower()):
            return team['id']
    return None


def get_player_id(player_name):
    """Get player ID from player name."""
    nba_players = players.get_players()
    for player in nba_players:
        if player_name.lower() in player['full_name'].lower():
            return player['id']
    return None


def fetch_historical_games(team_id=None, season="2025-26", last_n_games=None, date_from=None, date_to=None):
    """
    Fetch historical game data for analysis.

    Args:
        team_id: Optional team ID to filter games
        season: NBA season (e.g., "2025-26")
        last_n_games: Limit to last N games
        date_from: Start date (MM/DD/YYYY format)
        date_to: End date (MM/DD/YYYY format)

    Returns:
        List of game dictionaries with detailed stats
    """
    _rate_limiter.wait()

    # Use LeagueGameFinder for all cases (more reliable)
    game_finder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable="Regular Season",
        date_from_nullable=date_from,
        date_to_nullable=date_to,
    )
    games_df = game_finder.get_normalized_dict()
    games = games_df.get("LeagueGameFinderResults", [])

    parsed_games = []
    for game in games:
        if last_n_games and len(parsed_games) >= last_n_games:
            break

        parsed_game = {
            "game_id": game.get("GAME_ID"),
            "game_date": game.get("GAME_DATE"),
            "matchup": game.get("MATCHUP"),
            "wl": game.get("WL"),
            "team_id": game.get("TEAM_ID"),
            "team_abbreviation": game.get("TEAM_ABBREVIATION"),
            "pts": game.get("PTS"),
            "fg_pct": game.get("FG_PCT"),
            "fg3_pct": game.get("FG3_PCT"),
            "ft_pct": game.get("FT_PCT"),
            "reb": game.get("REB"),
            "ast": game.get("AST"),
            "stl": game.get("STL"),
            "blk": game.get("BLK"),
            "tov": game.get("TOV"),
            "plus_minus": game.get("PLUS_MINUS"),
            "min": game.get("MIN"),
        }
        parsed_games.append(parsed_game)

    return parsed_games


def fetch_team_statistics(team_id, season="2025-26"):
    """
    Fetch comprehensive team statistics.

    Args:
        team_id: NBA team ID
        season: NBA season (e.g., "2025-26")

    Returns:
        Dictionary with team statistics
    """
    _rate_limiter.wait()

    # Fetch base stats
    team_stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
        team_id=team_id,
        season=season,
        season_type_all_star="Regular Season"
    )
    stats_dict = team_stats.get_normalized_dict()

    overall = stats_dict.get("OverallTeamDashboard", [{}])[0] if stats_dict.get("OverallTeamDashboard") else {}
    home_away = stats_dict.get("LocationTeamDashboard", [])

    # Fetch advanced stats for ratings (OFF_RATING, DEF_RATING, NET_RATING, PACE)
    _rate_limiter.wait()
    try:
        advanced_stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
            team_id=team_id,
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced"
        )
        advanced_dict = advanced_stats.get_normalized_dict()
        advanced_overall = advanced_dict.get("OverallTeamDashboard", [{}])[0] if advanced_dict.get("OverallTeamDashboard") else {}
    except Exception as e:
        print(f"Warning: Could not fetch advanced stats for team {team_id}: {e}")
        advanced_overall = {}

    home_stats = next((s for s in home_away if s.get("GROUP_VALUE") == "Home"), {})
    away_stats = next((s for s in home_away if s.get("GROUP_VALUE") == "Road"), {})

    # Calculate games played for dividing totals into averages
    gp = max(overall.get("GP") or 1, 1)  # Avoid division by zero

    return {
        "team_id": team_id,
        "season": season,
        "overall": {
            "games_played": overall.get("GP"),
            "wins": overall.get("W"),
            "losses": overall.get("L"),
            "win_pct": overall.get("W_PCT"),
            # NBA API returns season TOTALS, divide by GP for per-game averages
            "pts_avg": (overall.get("PTS") or 0) / gp,
            "reb_avg": (overall.get("REB") or 0) / gp,
            "ast_avg": (overall.get("AST") or 0) / gp,
            "stl_avg": (overall.get("STL") or 0) / gp,
            "blk_avg": (overall.get("BLK") or 0) / gp,
            "tov_avg": (overall.get("TOV") or 0) / gp,
            "fg_pct": overall.get("FG_PCT"),
            "fg3_pct": overall.get("FG3_PCT"),
            "ft_pct": overall.get("FT_PCT"),
            "plus_minus": (overall.get("PLUS_MINUS") or 0) / gp,
            "off_rating": advanced_overall.get("OFF_RATING"),
            "def_rating": advanced_overall.get("DEF_RATING"),
            "net_rating": advanced_overall.get("NET_RATING"),
            "pace": advanced_overall.get("PACE"),
        },
        "home": {
            "games_played": home_stats.get("GP"),
            "wins": home_stats.get("W"),
            "losses": home_stats.get("L"),
            "win_pct": home_stats.get("W_PCT"),
            # PTS is total points, divide by GP for average
            "pts_avg": (home_stats.get("PTS") or 0) / max(home_stats.get("GP") or 1, 1),
            "plus_minus": home_stats.get("PLUS_MINUS"),
        },
        "away": {
            "games_played": away_stats.get("GP"),
            "wins": away_stats.get("W"),
            "losses": away_stats.get("L"),
            "win_pct": away_stats.get("W_PCT"),
            # PTS is total points, divide by GP for average
            "pts_avg": (away_stats.get("PTS") or 0) / max(away_stats.get("GP") or 1, 1),
            "plus_minus": away_stats.get("PLUS_MINUS"),
        },
    }


def fetch_league_team_stats(season="2025-26"):
    """
    Fetch league-wide team statistics for ranking and comparison.

    Args:
        season: NBA season

    Returns:
        List of team stats dictionaries
    """
    _rate_limiter.wait()

    league_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame"
    )
    stats_dict = league_stats.get_normalized_dict()

    return stats_dict.get("LeagueDashTeamStats", [])


def fetch_player_stats(player_id, season="2025-26", last_n_games=None):
    """
    Fetch player statistics and game log.

    Args:
        player_id: NBA player ID
        season: NBA season (e.g., "2024-25")
        last_n_games: Optional limit to last N games

    Returns:
        Dictionary with player stats, game log, and last 5 game averages
    """
    _rate_limiter.wait()

    # Get player game log
    game_log = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season"
    )
    log_dict = game_log.get_normalized_dict()
    games = log_dict.get("PlayerGameLog", [])

    if last_n_games:
        games = games[:last_n_games]

    _rate_limiter.wait()

    # Get player dashboard stats (per-game averages)
    player_dashboard = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
        player_id=player_id,
        season=season,
        season_type_playoffs="Regular Season",
        per_mode_detailed="PerGame"
    )
    dashboard_dict = player_dashboard.get_normalized_dict()
    overall = dashboard_dict.get("OverallPlayerDashboard", [{}])[0] if dashboard_dict.get("OverallPlayerDashboard") else {}

    parsed_games = []
    for game in games:
        parsed_games.append({
            "game_id": game.get("Game_ID"),
            "game_date": game.get("GAME_DATE"),
            "matchup": game.get("MATCHUP"),
            "wl": game.get("WL"),
            "min": game.get("MIN"),
            "pts": game.get("PTS"),
            "reb": game.get("REB"),
            "ast": game.get("AST"),
            "stl": game.get("STL"),
            "blk": game.get("BLK"),
            "tov": game.get("TOV"),
            "fg_made": game.get("FGM"),
            "fg_att": game.get("FGA"),
            "fg_pct": game.get("FG_PCT"),
            "fg3_made": game.get("FG3M"),
            "fg3_att": game.get("FG3A"),
            "fg3_pct": game.get("FG3_PCT"),
            "ft_made": game.get("FTM"),
            "ft_att": game.get("FTA"),
            "ft_pct": game.get("FT_PCT"),
            "plus_minus": game.get("PLUS_MINUS"),
        })

    # Calculate last 5 games averages for recent form
    last_5_averages = {}
    if len(parsed_games) >= 1:
        recent_games = parsed_games[:5]  # Game log is already sorted most recent first
        num_games = len(recent_games)
        last_5_averages = {
            "games_count": num_games,
            "pts_avg": sum(g.get("pts", 0) or 0 for g in recent_games) / num_games,
            "reb_avg": sum(g.get("reb", 0) or 0 for g in recent_games) / num_games,
            "ast_avg": sum(g.get("ast", 0) or 0 for g in recent_games) / num_games,
            "fg3_avg": sum(g.get("fg3_made", 0) or 0 for g in recent_games) / num_games,
            "min_avg": sum(g.get("min", 0) or 0 for g in recent_games) / num_games,
            "stl_avg": sum(g.get("stl", 0) or 0 for g in recent_games) / num_games,
            "blk_avg": sum(g.get("blk", 0) or 0 for g in recent_games) / num_games,
        }

    return {
        "player_id": player_id,
        "season": season,
        "season_averages": {
            "games_played": overall.get("GP"),
            "min_avg": overall.get("MIN"),
            "pts_avg": overall.get("PTS"),
            "reb_avg": overall.get("REB"),
            "ast_avg": overall.get("AST"),
            "stl_avg": overall.get("STL"),
            "blk_avg": overall.get("BLK"),
            "tov_avg": overall.get("TOV"),
            "fg_pct": overall.get("FG_PCT"),
            "fg3_pct": overall.get("FG3_PCT"),
            "ft_pct": overall.get("FT_PCT"),
            "plus_minus": overall.get("PLUS_MINUS"),
        },
        "last_5_averages": last_5_averages,
        "game_log": parsed_games,
    }


def fetch_team_roster(team_id, season="2025-26"):
    """
    Fetch team roster with player IDs.

    Args:
        team_id: NBA team ID
        season: NBA season

    Returns:
        List of player dictionaries
    """
    _rate_limiter.wait()

    roster = commonteamroster.CommonTeamRoster(
        team_id=team_id,
        season=season
    )
    roster_dict = roster.get_normalized_dict()

    players_list = []
    for player in roster_dict.get("CommonTeamRoster", []):
        players_list.append({
            "player_id": player.get("PLAYER_ID"),
            "player_name": player.get("PLAYER"),
            "position": player.get("POSITION"),
            "height": player.get("HEIGHT"),
            "weight": player.get("WEIGHT"),
            "age": player.get("AGE"),
            "experience": player.get("EXP"),
        })

    return players_list


def fetch_player_info(player_id):
    """
    Fetch detailed player information.

    Args:
        player_id: NBA player ID

    Returns:
        Dictionary with player info
    """
    _rate_limiter.wait()

    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    info_dict = player_info.get_normalized_dict()

    info = info_dict.get("CommonPlayerInfo", [{}])[0] if info_dict.get("CommonPlayerInfo") else {}

    return {
        "player_id": player_id,
        "first_name": info.get("FIRST_NAME"),
        "last_name": info.get("LAST_NAME"),
        "full_name": f"{info.get('FIRST_NAME', '')} {info.get('LAST_NAME', '')}".strip(),
        "team_id": info.get("TEAM_ID"),
        "team_name": info.get("TEAM_NAME"),
        "team_abbreviation": info.get("TEAM_ABBREVIATION"),
        "position": info.get("POSITION"),
        "height": info.get("HEIGHT"),
        "weight": info.get("WEIGHT"),
        "birth_date": info.get("BIRTHDATE"),
        "experience": info.get("SEASON_EXP"),
        "jersey": info.get("JERSEY"),
        "draft_year": info.get("DRAFT_YEAR"),
        "draft_round": info.get("DRAFT_ROUND"),
        "draft_number": info.get("DRAFT_NUMBER"),
    }


def fetch_head_to_head(team1_id, team2_id, season="2025-26", last_n_games=10):
    """
    Fetch head-to-head game history between two teams.

    Args:
        team1_id: First team NBA ID
        team2_id: Second team NBA ID
        season: NBA season (can include multiple seasons like "2023-24,2025-26")
        last_n_games: Maximum number of games to return

    Returns:
        List of head-to-head game results
    """
    _rate_limiter.wait()

    # Get team1's games
    game_finder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team1_id,
        vs_team_id_nullable=team2_id,
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    games_dict = game_finder.get_normalized_dict()
    games = games_dict.get("LeagueGameFinderResults", [])

    if last_n_games:
        games = games[:last_n_games]

    h2h_games = []
    for game in games:
        h2h_games.append({
            "game_id": game.get("GAME_ID"),
            "game_date": game.get("GAME_DATE"),
            "matchup": game.get("MATCHUP"),
            "team_id": game.get("TEAM_ID"),
            "wl": game.get("WL"),
            "pts": game.get("PTS"),
            "fg_pct": game.get("FG_PCT"),
            "fg3_pct": game.get("FG3_PCT"),
            "reb": game.get("REB"),
            "ast": game.get("AST"),
            "plus_minus": game.get("PLUS_MINUS"),
        })

    return h2h_games


def fetch_player_vs_team(player_id, opponent_team_id, season="2025-26", last_n_games=10):
    """
    Fetch player's performance history against a specific team.

    Args:
        player_id: NBA player ID
        opponent_team_id: Opponent team NBA ID
        season: NBA season
        last_n_games: Maximum games to return

    Returns:
        List of player game logs against the opponent
    """
    _rate_limiter.wait()

    game_log = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season"
    )
    log_dict = game_log.get_normalized_dict()
    all_games = log_dict.get("PlayerGameLog", [])

    # Filter for games against the specific opponent
    # Get opponent abbreviation
    nba_teams = teams.get_teams()
    opp_abbrev = None
    for team in nba_teams:
        if team['id'] == opponent_team_id:
            opp_abbrev = team['abbreviation']
            break

    vs_games = []
    for game in all_games:
        matchup = game.get("MATCHUP", "")
        if opp_abbrev and opp_abbrev in matchup:
            vs_games.append({
                "game_id": game.get("Game_ID"),
                "game_date": game.get("GAME_DATE"),
                "matchup": matchup,
                "wl": game.get("WL"),
                "min": game.get("MIN"),
                "pts": game.get("PTS"),
                "reb": game.get("REB"),
                "ast": game.get("AST"),
                "stl": game.get("STL"),
                "blk": game.get("BLK"),
                "fg_pct": game.get("FG_PCT"),
                "fg3_made": game.get("FG3M"),
                "fg3_pct": game.get("FG3_PCT"),
                "plus_minus": game.get("PLUS_MINUS"),
            })
            if last_n_games and len(vs_games) >= last_n_games:
                break

    return vs_games


def save_schedule_to_json(schedule, date, output_dir="."):
    """Save the parsed schedule to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / f"nba_schedule_{date}.json"

    output_data = {
        "date": date,
        "fetched_at": datetime.now().isoformat(),
        "game_count": len(schedule),
        "games": schedule,
    }

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Schedule saved to {filename}")
    return filename


def main():
    """Main function to fetch, parse, and save NBA schedule."""
    # Fetch today's schedule
    games_data, date = fetch_todays_schedule()

    # Parse game details
    schedule = parse_game_details(games_data)

    if not schedule:
        print(f"No games scheduled for {date}")
    else:
        print(f"Found {len(schedule)} game(s) scheduled for {date}")
        for game in schedule:
            home = game["home_team"]
            visitor = game["visitor_team"]
            print(f"  {visitor['abbreviation']} @ {home['abbreviation']} - {game['status']}")

    # Save to JSON
    save_schedule_to_json(schedule, date)


if __name__ == "__main__":
    main()
