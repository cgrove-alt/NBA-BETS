"""
News Monitor - Breaking NBA News and Lineup Change Detection

Monitors for:
1. Last-minute injury updates
2. Surprise scratches
3. Trade announcements affecting active players
4. Coach decisions (rest days, load management)

This module provides alerts when lineup-affecting news is detected.
"""

import re
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of lineup-affecting alerts."""
    INJURY_UPDATE = "injury_update"
    SURPRISE_SCRATCH = "surprise_scratch"
    RETURN_FROM_INJURY = "return_from_injury"
    LOAD_MANAGEMENT = "load_management"
    TRADE = "trade"
    SUSPENSION = "suspension"
    LINEUP_CHANGE = "lineup_change"
    MINUTES_RESTRICTION = "minutes_restriction"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Severity of alert for betting impact."""
    CRITICAL = "critical"  # Major star OUT, significant line movement expected
    HIGH = "high"  # Key rotation player OUT
    MEDIUM = "medium"  # Role player OUT or GTD
    LOW = "low"  # Minor impact or questionable


@dataclass
class NewsAlert:
    """Individual news alert about a player/team."""
    player_name: str
    team: str
    alert_type: AlertType
    severity: AlertSeverity
    headline: str
    detail: str
    source: str
    timestamp: datetime
    minutes_impact: float = 0.0  # Estimated minutes impact (-35 to +35)
    confidence: float = 0.0  # Confidence in this alert (0-1)

    def to_dict(self) -> dict:
        return {
            'player_name': self.player_name,
            'team': self.team,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'headline': self.headline,
            'detail': self.detail,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'minutes_impact': self.minutes_impact,
            'confidence': self.confidence,
        }


# Star player database for severity classification
STAR_PLAYERS = {
    # Tier 1 - Superstars (critical impact)
    "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
    "Nikola Jokic", "Luka Doncic", "Joel Embiid", "Jayson Tatum",
    "Anthony Davis", "Shai Gilgeous-Alexander", "Ja Morant", "Anthony Edwards",
    "Victor Wembanyama", "Damian Lillard", "Devin Booker",
    # Tier 2 - All-Stars (high impact)
    "Jaylen Brown", "Donovan Mitchell", "Trae Young", "Jimmy Butler",
    "Kawhi Leonard", "Paul George", "Tyrese Haliburton", "Zion Williamson",
    "De'Aaron Fox", "Domantas Sabonis", "Karl-Anthony Towns", "Jalen Brunson",
    "Bam Adebayo", "Chet Holmgren", "Paolo Banchero", "LaMelo Ball",
    "Kyrie Irving", "Jamal Murray", "Derrick White", "Brandon Ingram",
}

# Keywords for alert classification
INJURY_KEYWORDS = [
    "out", "ruled out", "will not play", "miss", "injured", "injury",
    "sidelined", "doubtful", "questionable", "day-to-day", "gtd",
    "game time decision", "limped", "sprained", "strained", "sore",
    "illness", "personal reasons", "rest", "load management"
]

RETURN_KEYWORDS = [
    "return", "cleared", "expected to play", "available", "upgraded",
    "no longer listed", "removed from injury report", "practicing",
    "back in lineup", "activated"
]

TRADE_KEYWORDS = [
    "traded", "trade", "waived", "released", "signed", "acquired",
    "buyout", "joining"
]


class NewsMonitor:
    """
    Monitor NBA news sources for lineup-affecting updates.

    Sources:
    1. ESPN NBA News
    2. Twitter/X NBA reporters (via web scraping)
    """

    ESPN_NBA_NEWS_URL = "https://www.espn.com/nba/"

    def __init__(self, lookback_hours: int = 6):
        """
        Initialize news monitor.

        Args:
            lookback_hours: How far back to look for news
        """
        self.lookback = timedelta(hours=lookback_hours)
        self._cache: list[NewsAlert] = []
        self._last_fetch: Optional[datetime] = None
        self._fetch_interval = timedelta(minutes=5)  # Minimum time between fetches

    def _classify_severity(self, player_name: str, alert_type: AlertType) -> AlertSeverity:
        """
        Classify alert severity based on player impact.

        Args:
            player_name: Player's name
            alert_type: Type of alert

        Returns:
            AlertSeverity level
        """
        is_star = player_name in STAR_PLAYERS

        if alert_type in [AlertType.INJURY_UPDATE, AlertType.SURPRISE_SCRATCH]:
            if is_star:
                return AlertSeverity.CRITICAL
            return AlertSeverity.HIGH

        if alert_type == AlertType.RETURN_FROM_INJURY:
            if is_star:
                return AlertSeverity.HIGH
            return AlertSeverity.MEDIUM

        if alert_type == AlertType.LOAD_MANAGEMENT:
            if is_star:
                return AlertSeverity.HIGH
            return AlertSeverity.MEDIUM

        if alert_type == AlertType.TRADE:
            return AlertSeverity.CRITICAL if is_star else AlertSeverity.HIGH

        return AlertSeverity.LOW

    def _estimate_minutes_impact(
        self,
        player_name: str,
        alert_type: AlertType
    ) -> float:
        """
        Estimate minutes impact from an alert.

        Negative = player losing minutes
        Positive = player gaining minutes (returning)

        Args:
            player_name: Player's name
            alert_type: Type of alert

        Returns:
            Estimated minutes impact
        """
        is_star = player_name in STAR_PLAYERS
        base_minutes = 35.0 if is_star else 25.0

        if alert_type in [AlertType.INJURY_UPDATE, AlertType.SURPRISE_SCRATCH]:
            return -base_minutes  # Full loss

        if alert_type == AlertType.RETURN_FROM_INJURY:
            return base_minutes * 0.8  # Often on minutes restriction

        if alert_type == AlertType.LOAD_MANAGEMENT:
            return -base_minutes  # Full rest day

        if alert_type == AlertType.MINUTES_RESTRICTION:
            return -base_minutes * 0.4  # Partial reduction

        if alert_type == AlertType.TRADE:
            return -base_minutes  # Removed from team

        return 0.0

    def _parse_alert_from_headline(
        self,
        headline: str,
        detail: str = "",
        source: str = ""
    ) -> Optional[NewsAlert]:
        """
        Parse a news headline into a structured alert.

        Args:
            headline: News headline
            detail: Additional detail text
            source: Source of the news

        Returns:
            NewsAlert if relevant, None otherwise
        """
        text = (headline + " " + detail).lower()

        # Try to extract player name (simplified - works for "Player Name" format)
        # In production, would use NER or player name database
        player_name = ""
        team = ""

        # Common pattern: "Player Name (Team)" or "Player Name is..."
        name_match = re.search(r'^([A-Z][a-z]+ [A-Z][a-z\-\']+)', headline)
        if name_match:
            player_name = name_match.group(1)

        # Extract team from parentheses
        team_match = re.search(r'\(([A-Z]{2,3})\)', headline)
        if team_match:
            team = team_match.group(1)

        if not player_name:
            return None

        # Classify alert type
        alert_type = AlertType.UNKNOWN

        for keyword in INJURY_KEYWORDS:
            if keyword in text:
                if "return" in text or "cleared" in text:
                    alert_type = AlertType.RETURN_FROM_INJURY
                elif "rest" in text or "load management" in text:
                    alert_type = AlertType.LOAD_MANAGEMENT
                else:
                    alert_type = AlertType.INJURY_UPDATE
                break

        if alert_type == AlertType.UNKNOWN:
            for keyword in RETURN_KEYWORDS:
                if keyword in text:
                    alert_type = AlertType.RETURN_FROM_INJURY
                    break

        if alert_type == AlertType.UNKNOWN:
            for keyword in TRADE_KEYWORDS:
                if keyword in text:
                    alert_type = AlertType.TRADE
                    break

        if alert_type == AlertType.UNKNOWN:
            return None  # Not a relevant alert

        severity = self._classify_severity(player_name, alert_type)
        minutes_impact = self._estimate_minutes_impact(player_name, alert_type)

        return NewsAlert(
            player_name=player_name,
            team=team,
            alert_type=alert_type,
            severity=severity,
            headline=headline,
            detail=detail,
            source=source,
            timestamp=datetime.now(),
            minutes_impact=minutes_impact,
            confidence=0.7,  # Moderate confidence from parsing
        )

    def fetch_espn_news(self) -> list[NewsAlert]:
        """
        Fetch and parse ESPN NBA news for alerts.

        Returns:
            List of NewsAlert objects
        """
        alerts = []

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            response = requests.get(self.ESPN_NBA_NEWS_URL, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find news headlines
            # ESPN structure varies, look for common article patterns
            articles = soup.find_all('article') + soup.find_all('div', class_='contentItem')

            for article in articles[:20]:  # Limit to recent articles
                headline_elem = article.find(['h1', 'h2', 'h3', 'a'])
                if headline_elem:
                    headline = headline_elem.get_text(strip=True)

                    # Get detail if available
                    detail_elem = article.find('p')
                    detail = detail_elem.get_text(strip=True) if detail_elem else ""

                    alert = self._parse_alert_from_headline(headline, detail, "ESPN")
                    if alert:
                        alerts.append(alert)

            logger.info(f"Found {len(alerts)} relevant alerts from ESPN")
            return alerts

        except requests.RequestException as e:
            logger.error(f"ESPN news fetch error: {e}")
            return []
        except Exception as e:
            logger.error(f"ESPN news parse error: {e}")
            return []

    def fetch_alerts(self, force_refresh: bool = False) -> list[NewsAlert]:
        """
        Fetch all alerts from all sources.

        Args:
            force_refresh: Force fetch even if recently updated

        Returns:
            List of all current alerts
        """
        # Rate limiting
        if not force_refresh and self._last_fetch:
            if datetime.now() - self._last_fetch < self._fetch_interval:
                return self._cache

        alerts = []

        # Fetch from ESPN
        espn_alerts = self.fetch_espn_news()
        alerts.extend(espn_alerts)

        # Deduplicate by player + type
        unique_alerts: dict[str, NewsAlert] = {}
        for alert in alerts:
            key = f"{alert.player_name}_{alert.alert_type.value}"
            if key not in unique_alerts:
                unique_alerts[key] = alert
            else:
                # Keep higher confidence
                if alert.confidence > unique_alerts[key].confidence:
                    unique_alerts[key] = alert

        # Filter to recent alerts
        cutoff = datetime.now() - self.lookback
        recent_alerts = [
            a for a in unique_alerts.values()
            if a.timestamp >= cutoff
        ]

        # Sort by severity and timestamp
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
        }
        recent_alerts.sort(
            key=lambda x: (severity_order[x.severity], -x.timestamp.timestamp())
        )

        self._cache = recent_alerts
        self._last_fetch = datetime.now()

        return recent_alerts

    def get_player_alerts(self, player_name: str) -> list[NewsAlert]:
        """
        Get alerts for a specific player.

        Args:
            player_name: Player's full name

        Returns:
            List of alerts for this player
        """
        all_alerts = self.fetch_alerts()
        player_lower = player_name.lower()

        return [
            a for a in all_alerts
            if player_lower in a.player_name.lower()
        ]

    def get_team_alerts(self, team: str) -> list[NewsAlert]:
        """
        Get alerts for a specific team.

        Args:
            team: Team abbreviation

        Returns:
            List of alerts for this team
        """
        all_alerts = self.fetch_alerts()
        team_upper = team.upper()

        return [a for a in all_alerts if a.team.upper() == team_upper]

    def get_critical_alerts(self) -> list[NewsAlert]:
        """
        Get only critical alerts (star players affected).

        Returns:
            List of critical alerts
        """
        all_alerts = self.fetch_alerts()
        return [a for a in all_alerts if a.severity == AlertSeverity.CRITICAL]

    def create_manual_alert(
        self,
        player_name: str,
        team: str,
        alert_type: AlertType,
        headline: str,
        detail: str = ""
    ) -> NewsAlert:
        """
        Create a manual alert (for testing or manual updates).

        Args:
            player_name: Player's name
            team: Team abbreviation
            alert_type: Type of alert
            headline: Alert headline
            detail: Additional detail

        Returns:
            Created NewsAlert
        """
        severity = self._classify_severity(player_name, alert_type)
        minutes_impact = self._estimate_minutes_impact(player_name, alert_type)

        alert = NewsAlert(
            player_name=player_name,
            team=team,
            alert_type=alert_type,
            severity=severity,
            headline=headline,
            detail=detail,
            source="Manual",
            timestamp=datetime.now(),
            minutes_impact=minutes_impact,
            confidence=1.0,  # Manual = full confidence
        )

        self._cache.append(alert)
        return alert


# Singleton instance
_monitor_instance: Optional[NewsMonitor] = None


def get_news_monitor() -> NewsMonitor:
    """Get global NewsMonitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = NewsMonitor()
    return _monitor_instance


if __name__ == "__main__":
    # Test the monitor
    monitor = NewsMonitor()

    print("Fetching NBA news alerts...")
    alerts = monitor.fetch_alerts()

    print(f"\nFound {len(alerts)} alerts")

    if alerts:
        print("\nRecent Alerts:")
        for alert in alerts[:10]:
            print(f"\n  [{alert.severity.value.upper()}] {alert.player_name} ({alert.team})")
            print(f"  Type: {alert.alert_type.value}")
            print(f"  Headline: {alert.headline[:80]}...")
            print(f"  Minutes Impact: {alert.minutes_impact:+.0f}")
    else:
        print("\nNo lineup-affecting alerts found")

    # Test manual alert
    print("\n" + "="*50)
    print("Testing manual alert...")
    manual = monitor.create_manual_alert(
        player_name="LeBron James",
        team="LAL",
        alert_type=AlertType.LOAD_MANAGEMENT,
        headline="LeBron James (rest) listed out for Thursday",
        detail="Lakers giving star rest on second night of back-to-back"
    )
    print(f"\nManual Alert: [{manual.severity.value}] {manual.headline}")
    print(f"Minutes Impact: {manual.minutes_impact:+.0f}")
