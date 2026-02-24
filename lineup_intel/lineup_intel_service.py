"""
Lineup Intel Service - Main Integration Point

Combines:
- InjuryScraper (injury status)
- LineupTracker (confirmed starters)
- NewsMonitor (breaking news)

Provides unified interface for:
1. Game intelligence (both teams' lineup status)
2. Player intelligence (individual player status)
3. Minutes impact estimates for prop betting
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

from .injury_scraper import InjuryScraper, InjuryStatus, PlayerInjury
from .lineup_tracker import LineupTracker, LineupConfirmation, StarterInfo
from .news_monitor import NewsMonitor, NewsAlert, AlertType, AlertSeverity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PlayerIntel:
    """Complete intelligence about a single player."""
    player_name: str
    team: str
    player_id: Optional[str] = None

    # Status
    injury_status: InjuryStatus = InjuryStatus.AVAILABLE
    injury_detail: str = ""
    is_starter: bool = False
    starter_confidence: float = 0.0

    # Minutes projection
    expected_minutes: float = 0.0
    minutes_floor: float = 0.0  # p10
    minutes_ceiling: float = 0.0  # p90
    minutes_uncertainty: str = "medium"  # low/medium/high

    # Availability
    availability_probability: float = 1.0
    minutes_multiplier: float = 1.0  # Even if plays, may be limited

    # Alerts
    has_recent_alert: bool = False
    alert_severity: Optional[AlertSeverity] = None
    alert_detail: str = ""

    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'player_name': self.player_name,
            'team': self.team,
            'player_id': self.player_id,
            'injury_status': self.injury_status.value,
            'injury_detail': self.injury_detail,
            'is_starter': self.is_starter,
            'starter_confidence': self.starter_confidence,
            'expected_minutes': self.expected_minutes,
            'minutes_floor': self.minutes_floor,
            'minutes_ceiling': self.minutes_ceiling,
            'minutes_uncertainty': self.minutes_uncertainty,
            'availability_probability': self.availability_probability,
            'minutes_multiplier': self.minutes_multiplier,
            'has_recent_alert': self.has_recent_alert,
            'alert_severity': self.alert_severity.value if self.alert_severity else None,
            'alert_detail': self.alert_detail,
            'last_updated': self.last_updated.isoformat(),
            'sources': self.sources,
        }


@dataclass
class GameIntel:
    """Complete lineup intelligence for a game."""
    home_team: str
    away_team: str
    game_date: str

    # Home team
    home_lineup: Optional[LineupConfirmation] = None
    home_injuries: list[PlayerInjury] = field(default_factory=list)
    home_alerts: list[NewsAlert] = field(default_factory=list)
    home_players: list[PlayerIntel] = field(default_factory=list)

    # Away team
    away_lineup: Optional[LineupConfirmation] = None
    away_injuries: list[PlayerInjury] = field(default_factory=list)
    away_alerts: list[NewsAlert] = field(default_factory=list)
    away_players: list[PlayerIntel] = field(default_factory=list)

    # Summary metrics
    home_star_out: bool = False
    away_star_out: bool = False
    home_injury_impact: float = 0.0  # Negative = more injuries
    away_injury_impact: float = 0.0
    injury_edge: str = ""  # "home", "away", or "neutral"

    # Confidence
    lineup_confidence: float = 0.0  # Overall confidence in lineup info
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            'home_team': self.home_team,
            'away_team': self.away_team,
            'game_date': self.game_date,
            'home_lineup': self.home_lineup.to_dict() if self.home_lineup else None,
            'away_lineup': self.away_lineup.to_dict() if self.away_lineup else None,
            'home_injuries': [i.to_dict() for i in self.home_injuries],
            'away_injuries': [i.to_dict() for i in self.away_injuries],
            'home_alerts': [a.to_dict() for a in self.home_alerts],
            'away_alerts': [a.to_dict() for a in self.away_alerts],
            'home_players': [p.to_dict() for p in self.home_players],
            'away_players': [p.to_dict() for p in self.away_players],
            'home_star_out': self.home_star_out,
            'away_star_out': self.away_star_out,
            'home_injury_impact': self.home_injury_impact,
            'away_injury_impact': self.away_injury_impact,
            'injury_edge': self.injury_edge,
            'lineup_confidence': self.lineup_confidence,
            'last_updated': self.last_updated.isoformat(),
        }


# Star players for impact calculation
STAR_PLAYERS = {
    "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
    "Nikola Jokic", "Luka Doncic", "Joel Embiid", "Jayson Tatum",
    "Anthony Davis", "Shai Gilgeous-Alexander", "Ja Morant", "Anthony Edwards",
    "Victor Wembanyama", "Damian Lillard", "Devin Booker", "Jaylen Brown",
    "Donovan Mitchell", "Trae Young", "Jimmy Butler", "Kawhi Leonard",
    "Paul George", "Tyrese Haliburton", "Jalen Brunson", "Karl-Anthony Towns",
}


class LineupIntelService:
    """
    Main service for lineup intelligence.

    Integrates injury scraper, lineup tracker, and news monitor
    to provide comprehensive game and player intelligence.
    """

    def __init__(self, cache_duration_minutes: int = 10):
        """
        Initialize the lineup intel service.

        Args:
            cache_duration_minutes: How long to cache combined intel
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self._game_cache: dict[str, tuple[datetime, GameIntel]] = {}
        self._player_cache: dict[str, tuple[datetime, PlayerIntel]] = {}

        # Initialize components
        self._injury_scraper = InjuryScraper(cache_duration_minutes=cache_duration_minutes)
        self._lineup_tracker = LineupTracker(cache_duration_minutes=cache_duration_minutes)
        self._news_monitor = NewsMonitor(lookback_hours=6)

        logger.info("LineupIntelService initialized")

    def _is_cache_valid(self, cache: dict, key: str) -> bool:
        """Check if cache entry is valid."""
        if key not in cache:
            return False
        cached_time, _ = cache[key]
        return datetime.now() - cached_time < self.cache_duration

    def _calculate_injury_impact(self, injuries: list[PlayerInjury]) -> float:
        """
        Calculate total injury impact for a team.

        Returns negative value (more injured = more negative).
        """
        impact = 0.0

        for injury in injuries:
            if injury.status == InjuryStatus.OUT:
                # Full impact
                if injury.player_name in STAR_PLAYERS:
                    impact -= 6.0  # Star player
                else:
                    impact -= 2.0  # Rotation player

            elif injury.status == InjuryStatus.DOUBTFUL:
                # 75% of full impact
                if injury.player_name in STAR_PLAYERS:
                    impact -= 4.5
                else:
                    impact -= 1.5

            elif injury.status in [InjuryStatus.QUESTIONABLE, InjuryStatus.GTD]:
                # 50% of full impact
                if injury.player_name in STAR_PLAYERS:
                    impact -= 3.0
                else:
                    impact -= 1.0

        return impact

    def _build_player_intel(
        self,
        player_name: str,
        team: str,
        injury: Optional[PlayerInjury],
        lineup: Optional[LineupConfirmation],
        alerts: list[NewsAlert]
    ) -> PlayerIntel:
        """
        Build complete player intelligence from components.
        """
        intel = PlayerIntel(
            player_name=player_name,
            team=team,
            sources=[]
        )

        # Injury information
        if injury:
            intel.injury_status = injury.status
            intel.injury_detail = injury.injury_detail
            intel.availability_probability = injury.availability_prob
            intel.minutes_multiplier = injury.minutes_multiplier
            intel.player_id = injury.player_id
            intel.sources.append(injury.source)

        # Lineup information
        if lineup:
            for starter in lineup.starters:
                if starter.player_name.lower() == player_name.lower():
                    intel.is_starter = True
                    intel.starter_confidence = starter.confidence
                    intel.expected_minutes = starter.avg_minutes
                    intel.sources.append(lineup.source)
                    break

        # If not starter, estimate bench minutes
        if not intel.is_starter and intel.expected_minutes == 0:
            intel.expected_minutes = 15.0  # Default bench minutes

        # Apply minutes multiplier for injuries
        intel.expected_minutes *= intel.minutes_multiplier

        # Minutes range (rough estimates)
        if intel.is_starter:
            intel.minutes_floor = max(0, intel.expected_minutes - 10)
            intel.minutes_ceiling = min(48, intel.expected_minutes + 8)
            intel.minutes_uncertainty = "medium"
        else:
            intel.minutes_floor = 0
            intel.minutes_ceiling = intel.expected_minutes + 15
            intel.minutes_uncertainty = "high"

        # Check for recent alerts
        player_alerts = [
            a for a in alerts
            if player_name.lower() in a.player_name.lower()
        ]
        if player_alerts:
            most_severe = max(player_alerts, key=lambda x: x.severity.value)
            intel.has_recent_alert = True
            intel.alert_severity = most_severe.severity
            intel.alert_detail = most_severe.headline
            intel.sources.append(most_severe.source)

            # Adjust minutes based on alert
            if most_severe.minutes_impact != 0:
                intel.expected_minutes = max(0, intel.expected_minutes + most_severe.minutes_impact)

        intel.last_updated = datetime.now()
        return intel

    def get_player_intel(
        self,
        player_name: str,
        team: str,
        force_refresh: bool = False
    ) -> PlayerIntel:
        """
        Get complete intelligence for a single player.

        Args:
            player_name: Player's full name
            team: Team abbreviation
            force_refresh: Force refresh even if cached

        Returns:
            PlayerIntel with all available information
        """
        cache_key = f"{player_name.lower()}_{team.upper()}"

        if not force_refresh and self._is_cache_valid(self._player_cache, cache_key):
            return self._player_cache[cache_key][1]

        # Get injury status
        injury = self._injury_scraper.get_player_injury(player_name)

        # Get lineup
        lineup = self._lineup_tracker.get_lineup(team)

        # Get alerts
        alerts = self._news_monitor.get_player_alerts(player_name)

        intel = self._build_player_intel(player_name, team, injury, lineup, alerts)

        self._player_cache[cache_key] = (datetime.now(), intel)
        return intel

    def get_game_intel(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> GameIntel:
        """
        Get complete lineup intelligence for a game.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Game date (YYYY-MM-DD), defaults to today
            force_refresh: Force refresh even if cached

        Returns:
            GameIntel with both teams' lineup info
        """
        game_date = game_date or datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{home_team}_{away_team}_{game_date}"

        if not force_refresh and self._is_cache_valid(self._game_cache, cache_key):
            return self._game_cache[cache_key][1]

        # Fetch all component data
        home_injuries = self._injury_scraper.get_team_injuries(home_team)
        away_injuries = self._injury_scraper.get_team_injuries(away_team)

        home_lineup = self._lineup_tracker.get_lineup(home_team, game_date, away_team)
        away_lineup = self._lineup_tracker.get_lineup(away_team, game_date, home_team)

        home_alerts = self._news_monitor.get_team_alerts(home_team)
        away_alerts = self._news_monitor.get_team_alerts(away_team)

        # Build player intel for key players
        home_players = []
        away_players = []

        # Process home team starters
        if home_lineup:
            for starter in home_lineup.starters:
                injury = next(
                    (i for i in home_injuries if i.player_name.lower() == starter.player_name.lower()),
                    None
                )
                player_intel = self._build_player_intel(
                    starter.player_name, home_team, injury, home_lineup, home_alerts
                )
                home_players.append(player_intel)

        # Process away team starters
        if away_lineup:
            for starter in away_lineup.starters:
                injury = next(
                    (i for i in away_injuries if i.player_name.lower() == starter.player_name.lower()),
                    None
                )
                player_intel = self._build_player_intel(
                    starter.player_name, away_team, injury, away_lineup, away_alerts
                )
                away_players.append(player_intel)

        # Calculate impact metrics
        home_impact = self._calculate_injury_impact(home_injuries)
        away_impact = self._calculate_injury_impact(away_injuries)

        # Check for star players out
        home_star_out = any(
            i.status == InjuryStatus.OUT and i.player_name in STAR_PLAYERS
            for i in home_injuries
        )
        away_star_out = any(
            i.status == InjuryStatus.OUT and i.player_name in STAR_PLAYERS
            for i in away_injuries
        )

        # Determine injury edge
        if abs(home_impact - away_impact) < 1.0:
            injury_edge = "neutral"
        elif home_impact > away_impact:
            injury_edge = "home"  # Home team healthier
        else:
            injury_edge = "away"  # Away team healthier

        # Calculate overall lineup confidence
        lineup_confidence = 0.5  # Base
        if home_lineup and home_lineup.is_confirmed:
            lineup_confidence += 0.25
        if away_lineup and away_lineup.is_confirmed:
            lineup_confidence += 0.25

        intel = GameIntel(
            home_team=home_team.upper(),
            away_team=away_team.upper(),
            game_date=game_date,
            home_lineup=home_lineup,
            away_lineup=away_lineup,
            home_injuries=home_injuries,
            away_injuries=away_injuries,
            home_alerts=home_alerts,
            away_alerts=away_alerts,
            home_players=home_players,
            away_players=away_players,
            home_star_out=home_star_out,
            away_star_out=away_star_out,
            home_injury_impact=home_impact,
            away_injury_impact=away_impact,
            injury_edge=injury_edge,
            lineup_confidence=lineup_confidence,
            last_updated=datetime.now(),
        )

        self._game_cache[cache_key] = (datetime.now(), intel)
        return intel

    def get_unavailable_players(self, team: str) -> list[PlayerInjury]:
        """
        Get list of players definitely OUT for a team.

        Args:
            team: Team abbreviation

        Returns:
            List of OUT/DOUBTFUL players
        """
        injuries = self._injury_scraper.get_team_injuries(team)
        return [
            i for i in injuries
            if i.status in [InjuryStatus.OUT, InjuryStatus.DOUBTFUL]
        ]

    def get_gtd_players(self, team: str) -> list[PlayerInjury]:
        """
        Get list of game-time decision players.

        Args:
            team: Team abbreviation

        Returns:
            List of GTD/Questionable players
        """
        injuries = self._injury_scraper.get_team_injuries(team)
        return [
            i for i in injuries
            if i.status in [InjuryStatus.GTD, InjuryStatus.QUESTIONABLE]
        ]

    def get_critical_alerts(self) -> list[NewsAlert]:
        """
        Get all critical (star player) alerts across the league.

        Returns:
            List of critical alerts
        """
        return self._news_monitor.get_critical_alerts()

    def refresh_all(self):
        """Force refresh all cached data."""
        self._game_cache.clear()
        self._player_cache.clear()
        self._injury_scraper.fetch_all_injuries(force_refresh=True)
        self._news_monitor.fetch_alerts(force_refresh=True)
        logger.info("All lineup intel caches refreshed")

    def get_minutes_adjustment(
        self,
        player_name: str,
        team: str,
        base_minutes: float
    ) -> dict:
        """
        Get minutes adjustment for a player based on current intel.

        Useful for Minutes Oracle integration.

        Args:
            player_name: Player's name
            team: Team abbreviation
            base_minutes: Baseline expected minutes

        Returns:
            Dict with adjusted minutes and uncertainty
        """
        intel = self.get_player_intel(player_name, team)

        # Start with base minutes
        adjusted = base_minutes

        # Apply availability probability
        if intel.availability_probability < 1.0:
            # Player may not play
            adjusted *= intel.availability_probability

        # Apply minutes restriction
        adjusted *= intel.minutes_multiplier

        # Adjust uncertainty based on alerts
        uncertainty = intel.minutes_uncertainty
        if intel.has_recent_alert:
            uncertainty = "high"

        return {
            'adjusted_minutes': adjusted,
            'availability_prob': intel.availability_probability,
            'minutes_multiplier': intel.minutes_multiplier,
            'is_starter': intel.is_starter,
            'uncertainty': uncertainty,
            'injury_status': intel.injury_status.value,
        }


# Convenience function for simple access
def get_game_intel(home_team: str, away_team: str) -> GameIntel:
    """
    Quick access to game intelligence.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation

    Returns:
        GameIntel object
    """
    service = LineupIntelService()
    return service.get_game_intel(home_team, away_team)


def get_player_status(player_name: str, team: str) -> PlayerIntel:
    """
    Quick access to player status.

    Args:
        player_name: Player's full name
        team: Team abbreviation

    Returns:
        PlayerIntel object
    """
    service = LineupIntelService()
    return service.get_player_intel(player_name, team)


if __name__ == "__main__":
    # Test the service
    service = LineupIntelService()

    print("="*60)
    print("LINEUP INTEL SERVICE TEST")
    print("="*60)

    # Test game intel
    print("\nFetching game intel: LAL vs BOS...")
    game = service.get_game_intel("LAL", "BOS")

    print(f"\nGame: {game.home_team} vs {game.away_team}")
    print(f"Date: {game.game_date}")
    print(f"Lineup Confidence: {game.lineup_confidence:.0%}")
    print(f"Injury Edge: {game.injury_edge}")

    print(f"\n{game.home_team} Injuries: {len(game.home_injuries)}")
    for inj in game.home_injuries[:3]:
        print(f"  {inj.player_name}: {inj.status.value}")

    print(f"\n{game.away_team} Injuries: {len(game.away_injuries)}")
    for inj in game.away_injuries[:3]:
        print(f"  {inj.player_name}: {inj.status.value}")

    print(f"\n{game.home_team} Expected Starters:")
    for player in game.home_players:
        print(f"  {player.player_name}: {player.expected_minutes:.1f} min ({player.injury_status.value})")

    print(f"\n{game.away_team} Expected Starters:")
    for player in game.away_players:
        print(f"  {player.player_name}: {player.expected_minutes:.1f} min ({player.injury_status.value})")

    # Test player intel
    print("\n" + "="*60)
    print("Testing player intel: LeBron James")
    lebron = service.get_player_intel("LeBron James", "LAL")
    print(f"  Status: {lebron.injury_status.value}")
    print(f"  Is Starter: {lebron.is_starter}")
    print(f"  Expected Minutes: {lebron.expected_minutes:.1f}")
    print(f"  Availability: {lebron.availability_probability:.0%}")

    # Test minutes adjustment
    print("\n" + "="*60)
    print("Testing minutes adjustment...")
    adj = service.get_minutes_adjustment("LeBron James", "LAL", base_minutes=35.0)
    print(f"  Base: 35.0 min")
    print(f"  Adjusted: {adj['adjusted_minutes']:.1f} min")
    print(f"  Uncertainty: {adj['uncertainty']}")
