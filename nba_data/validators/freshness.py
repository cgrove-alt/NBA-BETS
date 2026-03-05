"""
Data freshness tracking for predictions.

Tracks when each data source was last fetched so predictions
can report how stale their input data is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DataFreshness:
    """Tracks data freshness timestamps for a prediction."""

    odds_fetched_at: datetime | None = None
    stats_fetched_at: datetime | None = None
    injuries_fetched_at: datetime | None = None
    schedule_fetched_at: datetime | None = None

    def record_odds_fetch(self) -> None:
        """Record that odds data was just fetched."""
        self.odds_fetched_at = datetime.now()

    def record_stats_fetch(self) -> None:
        """Record that stats data was just fetched."""
        self.stats_fetched_at = datetime.now()

    def record_injuries_fetch(self) -> None:
        """Record that injury data was just fetched."""
        self.injuries_fetched_at = datetime.now()

    def record_schedule_fetch(self) -> None:
        """Record that schedule data was just fetched."""
        self.schedule_fetched_at = datetime.now()

    def _age_seconds(self, ts: datetime | None) -> float | None:
        """Get age in seconds of a timestamp."""
        if ts is None:
            return None
        return (datetime.now() - ts).total_seconds()

    def is_stale(
        self,
        max_odds_age_sec: int = 300,
        max_stats_age_min: int = 60,
        max_injuries_age_min: int = 30,
    ) -> bool:
        """
        Check if any data source is stale.

        Args:
            max_odds_age_sec: Maximum age for odds data (default 5 min)
            max_stats_age_min: Maximum age for stats data (default 60 min)
            max_injuries_age_min: Maximum age for injury data (default 30 min)

        Returns:
            True if any source is stale or missing
        """
        odds_age = self._age_seconds(self.odds_fetched_at)
        stats_age = self._age_seconds(self.stats_fetched_at)
        injuries_age = self._age_seconds(self.injuries_fetched_at)

        if odds_age is not None and odds_age > max_odds_age_sec:
            return True
        if stats_age is not None and stats_age > max_stats_age_min * 60:
            return True
        if injuries_age is not None and injuries_age > max_injuries_age_min * 60:
            return True

        return False

    def stale_sources(
        self,
        max_odds_age_sec: int = 300,
        max_stats_age_min: int = 60,
        max_injuries_age_min: int = 30,
    ) -> list[str]:
        """Return list of stale source names."""
        stale = []
        odds_age = self._age_seconds(self.odds_fetched_at)
        stats_age = self._age_seconds(self.stats_fetched_at)
        injuries_age = self._age_seconds(self.injuries_fetched_at)

        if self.odds_fetched_at is None:
            stale.append('odds (never fetched)')
        elif odds_age and odds_age > max_odds_age_sec:
            stale.append(f'odds ({odds_age:.0f}s old)')

        if self.stats_fetched_at is None:
            stale.append('stats (never fetched)')
        elif stats_age and stats_age > max_stats_age_min * 60:
            stale.append(f'stats ({stats_age/60:.0f}min old)')

        if self.injuries_fetched_at is None:
            stale.append('injuries (never fetched)')
        elif injuries_age and injuries_age > max_injuries_age_min * 60:
            stale.append(f'injuries ({injuries_age/60:.0f}min old)')

        return stale

    def to_dict(self) -> dict:
        """Convert to dict for inclusion in prediction output."""
        now = datetime.now()
        return {
            'odds_fetched_at': self.odds_fetched_at.isoformat() if self.odds_fetched_at else None,
            'odds_age_seconds': round(self._age_seconds(self.odds_fetched_at) or -1, 1),
            'stats_fetched_at': self.stats_fetched_at.isoformat() if self.stats_fetched_at else None,
            'stats_age_seconds': round(self._age_seconds(self.stats_fetched_at) or -1, 1),
            'injuries_fetched_at': self.injuries_fetched_at.isoformat() if self.injuries_fetched_at else None,
            'injuries_age_seconds': round(self._age_seconds(self.injuries_fetched_at) or -1, 1),
            'is_stale': self.is_stale(),
            'checked_at': now.isoformat(),
        }
