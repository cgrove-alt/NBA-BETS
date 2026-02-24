"""
Lineup Intel - Real-Time NBA Lineup and Injury Intelligence

This module provides:
1. InjuryScraper - Fetches official NBA injury reports
2. LineupTracker - Tracks confirmed starting lineups
3. NewsMonitor - Monitors breaking news for last-minute changes
4. LineupIntelService - Main service integrating all components

Usage:
    from lineup_intel import LineupIntelService

    service = LineupIntelService()
    intel = service.get_game_intel("LAL", "BOS")

    # Result includes:
    # - Player statuses (OUT, GTD, QUESTIONABLE, etc.)
    # - Confirmed starters
    # - Minutes impact estimates
    # - Last update timestamps
"""

from .injury_scraper import InjuryScraper, InjuryStatus, PlayerInjury
from .lineup_tracker import LineupTracker, LineupConfirmation, StarterInfo
from .news_monitor import NewsMonitor, NewsAlert
from .lineup_intel_service import LineupIntelService, GameIntel, PlayerIntel

__all__ = [
    'InjuryScraper',
    'InjuryStatus',
    'PlayerInjury',
    'LineupTracker',
    'LineupConfirmation',
    'StarterInfo',
    'NewsMonitor',
    'NewsAlert',
    'LineupIntelService',
    'GameIntel',
    'PlayerIntel',
]
