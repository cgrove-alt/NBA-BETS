"""
Advanced NBA Statistics Calculator

Calculates advanced basketball statistics for enhanced predictions:
- PER (Player Efficiency Rating)
- TS% (True Shooting Percentage)
- USG% (Usage Rate)
- BPM (Box Plus/Minus)
- Win Shares
- ORTG/DRTG (Offensive/Defensive Rating)
- eFG% (Effective Field Goal Percentage)
- And more...

These stats provide deeper insights than traditional box score stats
and are crucial for accurate betting predictions.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# League average constants (2023-24 season approximations)
LEAGUE_AVERAGES = {
    "pace": 100.0,  # Possessions per 48 minutes
    "ortg": 114.0,  # Points per 100 possessions
    "drtg": 114.0,
    "ts_pct": 0.575,  # True Shooting %
    "ast_pct": 0.60,  # % of FGM that are assisted
    "orb_pct": 0.25,  # Offensive rebound %
    "tov_pct": 0.12,  # Turnover %
    "ftr": 0.28,  # Free throw rate (FTA/FGA)
    "efg_pct": 0.545,  # Effective FG%
    "fg_pct": 0.47,
    "fg3_pct": 0.365,
    "ft_pct": 0.78,
    "ppg": 114.0,  # Points per game (team)
    "minutes_per_game": 240.0,  # Team minutes per game
}


@dataclass
class PlayerStats:
    """Container for player box score statistics."""
    # Basic counting stats
    minutes: float = 0.0
    points: float = 0.0
    rebounds: float = 0.0
    offensive_rebounds: float = 0.0
    defensive_rebounds: float = 0.0
    assists: float = 0.0
    steals: float = 0.0
    blocks: float = 0.0
    turnovers: float = 0.0
    personal_fouls: float = 0.0

    # Shooting stats
    fgm: float = 0.0  # Field goals made
    fga: float = 0.0  # Field goals attempted
    fg3m: float = 0.0  # 3-pointers made
    fg3a: float = 0.0  # 3-pointers attempted
    ftm: float = 0.0  # Free throws made
    fta: float = 0.0  # Free throws attempted

    # Team context (needed for some calculations)
    team_minutes: float = 240.0
    team_fgm: float = 40.0
    team_fga: float = 88.0
    team_fg3m: float = 12.0
    team_fg3a: float = 35.0
    team_ftm: float = 18.0
    team_fta: float = 23.0
    team_orb: float = 10.0
    team_drb: float = 35.0
    team_ast: float = 25.0
    team_stl: float = 8.0
    team_blk: float = 5.0
    team_tov: float = 14.0
    team_pf: float = 20.0
    team_pts: float = 112.0

    # Opponent context
    opp_drb: float = 35.0
    opp_orb: float = 10.0
    opp_fgm: float = 40.0
    opp_fga: float = 88.0
    opp_ftm: float = 18.0
    opp_fta: float = 23.0
    opp_tov: float = 14.0
    opp_pts: float = 110.0


@dataclass
class TeamStats:
    """Container for team-level statistics."""
    # Basic stats
    minutes: float = 240.0
    points: float = 112.0
    rebounds: float = 45.0
    offensive_rebounds: float = 10.0
    defensive_rebounds: float = 35.0
    assists: float = 25.0
    steals: float = 8.0
    blocks: float = 5.0
    turnovers: float = 14.0
    personal_fouls: float = 20.0

    # Shooting
    fgm: float = 40.0
    fga: float = 88.0
    fg3m: float = 12.0
    fg3a: float = 35.0
    ftm: float = 18.0
    fta: float = 23.0

    # Opponent stats
    opp_points: float = 110.0
    opp_rebounds: float = 44.0
    opp_offensive_rebounds: float = 9.0
    opp_defensive_rebounds: float = 35.0
    opp_fgm: float = 39.0
    opp_fga: float = 86.0
    opp_fg3m: float = 11.0
    opp_fg3a: float = 33.0
    opp_ftm: float = 17.0
    opp_fta: float = 22.0
    opp_turnovers: float = 13.0


class AdvancedStatsCalculator:
    """
    Calculator for advanced NBA statistics.

    All formulas based on Basketball-Reference definitions:
    https://www.basketball-reference.com/about/glossary.html
    """

    def __init__(self, league_averages: Dict[str, float] = None):
        """
        Initialize calculator with league averages.

        Args:
            league_averages: Optional custom league averages
        """
        self.league_avg = league_averages or LEAGUE_AVERAGES.copy()

    # =========================================================================
    # SHOOTING EFFICIENCY STATS
    # =========================================================================

    def true_shooting_pct(
        self,
        points: float,
        fga: float,
        fta: float
    ) -> float:
        """
        True Shooting Percentage (TS%)

        Measures shooting efficiency accounting for 2PT, 3PT, and FT.
        TS% = PTS / (2 * (FGA + 0.44 * FTA))

        Args:
            points: Total points scored
            fga: Field goal attempts
            fta: Free throw attempts

        Returns:
            TS% (0.0 to 1.0, typically 0.45-0.70)
        """
        tsa = 2 * (fga + 0.44 * fta)  # True Shooting Attempts
        if tsa == 0:
            return 0.0
        ts_pct = points / tsa
        return float(np.clip(ts_pct, 0.0, 1.0))

    def effective_fg_pct(
        self,
        fgm: float,
        fg3m: float,
        fga: float
    ) -> float:
        """
        Effective Field Goal Percentage (eFG%)

        Adjusts FG% to account for 3-pointers being worth more.
        eFG% = (FGM + 0.5 * 3PM) / FGA

        Returns:
            eFG% (0.0 to 1.0, typically 0.45-0.60)
        """
        if fga == 0:
            return 0.0
        efg = (fgm + 0.5 * fg3m) / fga
        return float(np.clip(efg, 0.0, 1.0))

    # =========================================================================
    # USAGE AND RATE STATS
    # =========================================================================

    def usage_rate(
        self,
        fga: float,
        fta: float,
        tov: float,
        minutes: float,
        team_fga: float,
        team_fta: float,
        team_tov: float,
        team_minutes: float = 240.0
    ) -> float:
        """
        Usage Rate (USG%)

        Estimates percentage of team plays used by player while on floor.
        USG% = 100 * ((FGA + 0.44*FTA + TOV) * (Team_MP/5)) / (MP * (Team_FGA + 0.44*Team_FTA + Team_TOV))

        Returns:
            USG% (0.0 to 50.0, typically 15-35)
        """
        if minutes == 0 or team_minutes == 0:
            return 0.0

        player_possessions = fga + 0.44 * fta + tov
        team_possessions = team_fga + 0.44 * team_fta + team_tov

        if team_possessions == 0:
            return 0.0

        usg = 100 * (player_possessions * (team_minutes / 5)) / (minutes * team_possessions)
        return float(np.clip(usg, 0.0, 50.0))

    def assist_percentage(
        self,
        assists: float,
        minutes: float,
        team_fgm: float,
        fgm: float,
        team_minutes: float = 240.0
    ) -> float:
        """
        Assist Percentage (AST%)

        Percentage of teammate field goals player assisted while on floor.
        AST% = 100 * AST / (((MP / (Team_MP / 5)) * Team_FGM) - FGM)

        Returns:
            AST% (0.0 to 60.0, typically 5-40)
        """
        if minutes == 0 or team_minutes == 0:
            return 0.0

        teammate_fgm = ((minutes / (team_minutes / 5)) * team_fgm) - fgm
        if teammate_fgm <= 0:
            return 0.0

        ast_pct = 100 * assists / teammate_fgm
        return float(np.clip(ast_pct, 0.0, 60.0))

    def rebound_percentage(
        self,
        rebounds: float,
        minutes: float,
        team_rebounds: float,
        opp_rebounds: float,
        team_minutes: float = 240.0,
        is_offensive: bool = False
    ) -> float:
        """
        Rebound Percentage (TRB%, ORB%, DRB%)

        Percentage of available rebounds grabbed while on floor.

        Returns:
            REB% (0.0 to 30.0, typically 5-20)
        """
        if minutes == 0 or team_minutes == 0:
            return 0.0

        available = team_rebounds + opp_rebounds
        if available == 0:
            return 0.0

        reb_pct = 100 * (rebounds * (team_minutes / 5)) / (minutes * available)
        return float(np.clip(reb_pct, 0.0, 30.0))

    def turnover_percentage(
        self,
        turnovers: float,
        fga: float,
        fta: float
    ) -> float:
        """
        Turnover Percentage (TOV%)

        Turnovers per 100 plays.
        TOV% = 100 * TOV / (FGA + 0.44*FTA + TOV)

        Returns:
            TOV% (0.0 to 30.0, typically 5-20)
        """
        possessions = fga + 0.44 * fta + turnovers
        if possessions == 0:
            return 0.0

        tov_pct = 100 * turnovers / possessions
        return float(np.clip(tov_pct, 0.0, 30.0))

    def steal_percentage(
        self,
        steals: float,
        minutes: float,
        opp_possessions: float,
        team_minutes: float = 240.0
    ) -> float:
        """
        Steal Percentage (STL%)

        Percentage of opponent possessions ending in steal by player.

        Returns:
            STL% (0.0 to 5.0, typically 1-3)
        """
        if minutes == 0 or opp_possessions == 0 or team_minutes == 0:
            return 0.0

        stl_pct = 100 * (steals * (team_minutes / 5)) / (minutes * opp_possessions)
        return float(np.clip(stl_pct, 0.0, 5.0))

    def block_percentage(
        self,
        blocks: float,
        minutes: float,
        opp_fga: float,
        opp_fg3a: float,
        team_minutes: float = 240.0
    ) -> float:
        """
        Block Percentage (BLK%)

        Percentage of opponent 2-point attempts blocked by player.

        Returns:
            BLK% (0.0 to 15.0, typically 1-5)
        """
        if minutes == 0 or team_minutes == 0:
            return 0.0

        opp_2pa = opp_fga - opp_fg3a
        if opp_2pa == 0:
            return 0.0

        blk_pct = 100 * (blocks * (team_minutes / 5)) / (minutes * opp_2pa)
        return float(np.clip(blk_pct, 0.0, 15.0))

    # =========================================================================
    # TEAM STATS
    # =========================================================================

    def possessions(
        self,
        fga: float,
        orb: float,
        tov: float,
        fta: float,
        opp_drb: float = None,
        team_drb: float = None
    ) -> float:
        """
        Estimate team possessions.

        Basic formula: POSS = FGA - ORB + TOV + 0.44*FTA

        Returns:
            Estimated possessions per game (typically 95-105)
        """
        poss = fga - orb + tov + 0.44 * fta
        return float(np.clip(poss, 80, 120))

    def pace(
        self,
        team_poss: float,
        opp_poss: float,
        minutes: float = 240.0
    ) -> float:
        """
        Pace Factor

        Possessions per 48 minutes.
        Pace = 48 * ((Team_Poss + Opp_Poss) / (2 * (Team_MP / 5)))

        Returns:
            Pace (typically 95-105)
        """
        if minutes == 0:
            return self.league_avg["pace"]

        pace = 48 * ((team_poss + opp_poss) / (2 * (minutes / 5)))
        return float(np.clip(pace, 90, 110))

    def offensive_rating(
        self,
        points: float,
        possessions: float
    ) -> float:
        """
        Offensive Rating (ORtg)

        Points scored per 100 possessions.

        Returns:
            ORtg (typically 100-120)
        """
        if possessions == 0:
            return self.league_avg["ortg"]

        ortg = (points / possessions) * 100
        return float(np.clip(ortg, 80, 140))

    def defensive_rating(
        self,
        opp_points: float,
        possessions: float
    ) -> float:
        """
        Defensive Rating (DRtg)

        Points allowed per 100 possessions. Lower is better.

        Returns:
            DRtg (typically 100-120)
        """
        if possessions == 0:
            return self.league_avg["drtg"]

        drtg = (opp_points / possessions) * 100
        return float(np.clip(drtg, 80, 140))

    def net_rating(
        self,
        ortg: float,
        drtg: float
    ) -> float:
        """
        Net Rating

        Difference between offensive and defensive rating.
        Positive = team scores more than they allow per 100 poss.

        Returns:
            Net Rating (typically -15 to +15)
        """
        net = ortg - drtg
        return float(np.clip(net, -25, 25))

    # =========================================================================
    # PLAYER EFFICIENCY RATING (PER)
    # =========================================================================

    def player_efficiency_rating(self, stats: PlayerStats) -> float:
        """
        Player Efficiency Rating (PER)

        Comprehensive rating measuring per-minute production.
        League average is always 15.0.

        Simplified formula (full formula is very complex):
        This is an approximation of the Hollinger PER formula.

        Returns:
            PER (typically 5-35, league average = 15)
        """
        if stats.minutes == 0:
            return 0.0

        # Factor calculations
        factor = (2 / 3) - (0.5 * (stats.team_ast / stats.team_fgm)) / (2 * (stats.team_fgm / stats.team_fta))
        vop = stats.team_pts / (stats.team_fga - stats.team_orb + stats.team_tov + 0.44 * stats.team_fta)
        drbp = (stats.team_drb) / (stats.team_drb + stats.opp_orb)

        # Unadjusted PER calculation
        uper = (1 / stats.minutes) * (
            stats.fg3m +
            (2 / 3) * stats.assists +
            (2 - factor * (stats.team_ast / stats.team_fgm)) * stats.fgm +
            (stats.ftm * 0.5 * (1 + (1 - (stats.team_ast / stats.team_fgm)) + (2 / 3) * (stats.team_ast / stats.team_fgm))) -
            vop * stats.turnovers -
            vop * drbp * (stats.fga - stats.fgm) -
            vop * 0.44 * (0.44 + (0.56 * drbp)) * (stats.fta - stats.ftm) +
            vop * (1 - drbp) * (stats.rebounds - stats.offensive_rebounds) +
            vop * drbp * stats.offensive_rebounds +
            vop * stats.steals +
            vop * drbp * stats.blocks -
            stats.personal_fouls * ((self.league_avg["ftm"] / self.league_avg["minutes_per_game"]) - 0.44 * (self.league_avg["fta"] / self.league_avg["minutes_per_game"]) * vop)
        )

        # Pace adjustment (simplified)
        pace_adjustment = self.league_avg["pace"] / 100.0
        per = uper * pace_adjustment * 15.0 / self.league_avg["ppg"]

        return float(np.clip(per, 0, 40))

    def simplified_per(self, stats: PlayerStats) -> float:
        """
        Simplified PER calculation.

        Approximation that captures the essence of PER without
        full complexity. Useful when you don't have all the data.

        Returns:
            Approximate PER (5-35)
        """
        if stats.minutes == 0:
            return 0.0

        per_minute = (
            stats.points +
            0.4 * stats.fgm -
            0.7 * stats.fga +
            0.35 * stats.rebounds +
            0.7 * stats.assists +
            0.7 * stats.steals +
            0.7 * stats.blocks -
            0.4 * stats.personal_fouls -
            0.7 * stats.turnovers +
            0.35 * (stats.ftm - stats.fta)
        ) / stats.minutes

        # Scale to approximate PER (league avg = 15)
        per = per_minute * 15 / 0.8
        return float(np.clip(per, 0, 40))

    # =========================================================================
    # BOX PLUS/MINUS (BPM)
    # =========================================================================

    def box_plus_minus(self, stats: PlayerStats) -> float:
        """
        Box Plus/Minus (BPM)

        Estimates player's contribution in points per 100 possessions
        relative to league average (0.0).

        Simplified approximation of the BPM formula.

        Returns:
            BPM (typically -10 to +15)
        """
        if stats.minutes == 0:
            return 0.0

        # Per-minute rates
        mp = stats.minutes
        pts_per_min = stats.points / mp
        ast_per_min = stats.assists / mp
        reb_per_min = stats.rebounds / mp
        stl_per_min = stats.steals / mp
        blk_per_min = stats.blocks / mp
        tov_per_min = stats.turnovers / mp
        pf_per_min = stats.personal_fouls / mp

        # TS% relative to league
        ts = self.true_shooting_pct(stats.points, stats.fga, stats.fta)
        ts_diff = ts - self.league_avg["ts_pct"]

        # Simplified BPM formula
        bpm = (
            0.064 * (100 * pts_per_min - 7.5) +
            0.032 * (100 * ast_per_min - 3.5) +
            0.013 * (100 * reb_per_min - 6.0) +
            0.017 * (100 * stl_per_min - 1.2) +
            0.013 * (100 * blk_per_min - 0.8) -
            0.018 * (100 * tov_per_min - 2.0) -
            0.005 * (100 * pf_per_min - 3.0) +
            25 * ts_diff  # TS% premium
        )

        return float(np.clip(bpm, -15, 20))

    # =========================================================================
    # WIN SHARES
    # =========================================================================

    def win_shares(
        self,
        stats: PlayerStats,
        team_wins: float,
        games_played: int = 82
    ) -> float:
        """
        Win Shares (WS)

        Estimate of wins contributed by player.
        Full season for average starter: ~5-10 WS.
        MVP caliber: 15+ WS.

        Simplified calculation.

        Returns:
            Win Shares (0 to 20+)
        """
        if stats.minutes == 0:
            return 0.0

        # Marginal offense
        pts_produced = stats.points + 0.4 * stats.assists
        marginal_offense = pts_produced - (0.92 * self.league_avg["ppg"] * stats.minutes / stats.team_minutes)

        # Marginal defense (simplified)
        marginal_defense = (stats.steals * 1.0 + stats.blocks * 0.6 +
                           stats.defensive_rebounds * 0.3 -
                           stats.personal_fouls * 0.5)

        # Win shares
        ws = (marginal_offense + marginal_defense) / 30  # Rough scaling

        # Scale by team wins
        team_ws_available = team_wins
        ws = ws * (team_ws_available / 41)  # Half wins go to offense, half to defense

        return float(np.clip(ws, 0, 25))

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def calculate_all_player_stats(self, stats: PlayerStats) -> Dict[str, float]:
        """
        Calculate all advanced stats for a player.

        Args:
            stats: PlayerStats object with raw statistics

        Returns:
            Dictionary of all advanced stats
        """
        return {
            # Shooting
            "ts_pct": self.true_shooting_pct(stats.points, stats.fga, stats.fta),
            "efg_pct": self.effective_fg_pct(stats.fgm, stats.fg3m, stats.fga),

            # Usage & Rates
            "usg_pct": self.usage_rate(
                stats.fga, stats.fta, stats.turnovers, stats.minutes,
                stats.team_fga, stats.team_fta, stats.team_tov, stats.team_minutes
            ),
            "ast_pct": self.assist_percentage(
                stats.assists, stats.minutes, stats.team_fgm, stats.fgm, stats.team_minutes
            ),
            "trb_pct": self.rebound_percentage(
                stats.rebounds, stats.minutes,
                stats.team_drb + stats.team_orb, stats.opp_drb + stats.opp_orb,
                stats.team_minutes
            ),
            "tov_pct": self.turnover_percentage(stats.turnovers, stats.fga, stats.fta),

            # Efficiency
            "per": self.simplified_per(stats),
            "bpm": self.box_plus_minus(stats),
        }

    def calculate_all_team_stats(self, stats: TeamStats) -> Dict[str, float]:
        """
        Calculate all advanced stats for a team.

        Args:
            stats: TeamStats object with raw statistics

        Returns:
            Dictionary of all advanced stats
        """
        team_poss = self.possessions(
            stats.fga, stats.offensive_rebounds, stats.turnovers, stats.fta
        )
        opp_poss = self.possessions(
            stats.opp_fga, stats.opp_offensive_rebounds, stats.opp_turnovers, stats.opp_fta
        )

        ortg = self.offensive_rating(stats.points, team_poss)
        drtg = self.defensive_rating(stats.opp_points, opp_poss)

        return {
            "possessions": team_poss,
            "pace": self.pace(team_poss, opp_poss, stats.minutes),
            "offensive_rating": ortg,
            "defensive_rating": drtg,
            "net_rating": self.net_rating(ortg, drtg),
            "ts_pct": self.true_shooting_pct(
                stats.points, stats.fga, stats.fta
            ),
            "efg_pct": self.effective_fg_pct(
                stats.fgm, stats.fg3m, stats.fga
            ),
            "tov_pct": self.turnover_percentage(
                stats.turnovers, stats.fga, stats.fta
            ),
            "orb_pct": stats.offensive_rebounds / (stats.offensive_rebounds + stats.opp_defensive_rebounds) if (stats.offensive_rebounds + stats.opp_defensive_rebounds) > 0 else 0.25,
            "ftr": stats.fta / stats.fga if stats.fga > 0 else 0.0,
        }


# Convenience functions
def calculate_ts_pct(points: float, fga: float, fta: float) -> float:
    """Quick TS% calculation."""
    calc = AdvancedStatsCalculator()
    return calc.true_shooting_pct(points, fga, fta)


def calculate_efg_pct(fgm: float, fg3m: float, fga: float) -> float:
    """Quick eFG% calculation."""
    calc = AdvancedStatsCalculator()
    return calc.effective_fg_pct(fgm, fg3m, fga)


def calculate_usage_rate(
    fga: float, fta: float, tov: float, minutes: float,
    team_fga: float, team_fta: float, team_tov: float
) -> float:
    """Quick USG% calculation."""
    calc = AdvancedStatsCalculator()
    return calc.usage_rate(fga, fta, tov, minutes, team_fga, team_fta, team_tov)


def calculate_pace(team_poss: float, opp_poss: float, minutes: float = 240.0) -> float:
    """Quick Pace calculation."""
    calc = AdvancedStatsCalculator()
    return calc.pace(team_poss, opp_poss, minutes)


def calculate_ortg(points: float, possessions: float) -> float:
    """Quick ORtg calculation."""
    calc = AdvancedStatsCalculator()
    return calc.offensive_rating(points, possessions)


def calculate_drtg(opp_points: float, possessions: float) -> float:
    """Quick DRtg calculation."""
    calc = AdvancedStatsCalculator()
    return calc.defensive_rating(opp_points, possessions)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Advanced NBA Statistics Calculator Demo")
    print("=" * 60)

    calc = AdvancedStatsCalculator()

    # Example player stats (LeBron-like player)
    player = PlayerStats(
        minutes=35.0,
        points=25.0,
        rebounds=8.0,
        offensive_rebounds=1.0,
        defensive_rebounds=7.0,
        assists=8.0,
        steals=1.5,
        blocks=0.5,
        turnovers=3.5,
        personal_fouls=2.0,
        fgm=9.0,
        fga=18.0,
        fg3m=2.0,
        fg3a=5.0,
        ftm=5.0,
        fta=6.0,
        team_pts=115.0,
        team_fgm=42.0,
        team_fga=90.0,
        team_orb=10.0,
        team_drb=35.0,
        team_ast=27.0,
        team_tov=14.0,
    )

    print("\nPlayer Advanced Stats:")
    print("-" * 40)
    advanced = calc.calculate_all_player_stats(player)
    for stat, value in advanced.items():
        if "pct" in stat:
            print(f"  {stat.upper()}: {value:.1%}")
        else:
            print(f"  {stat.upper()}: {value:.1f}")

    # Example team stats
    team = TeamStats(
        points=115.0,
        rebounds=45.0,
        offensive_rebounds=10.0,
        defensive_rebounds=35.0,
        assists=27.0,
        steals=8.0,
        blocks=5.0,
        turnovers=14.0,
        fgm=42.0,
        fga=90.0,
        fg3m=13.0,
        fg3a=35.0,
        ftm=18.0,
        fta=23.0,
        opp_points=108.0,
        opp_fgm=40.0,
        opp_fga=88.0,
        opp_turnovers=15.0,
    )

    print("\nTeam Advanced Stats:")
    print("-" * 40)
    team_advanced = calc.calculate_all_team_stats(team)
    for stat, value in team_advanced.items():
        if "pct" in stat or "rating" in stat:
            print(f"  {stat}: {value:.1f}")
        else:
            print(f"  {stat}: {value:.1f}")

    print("\n" + "=" * 60)
    print("Key Stats Interpretation:")
    print("=" * 60)
    print(f"""
TS% (True Shooting): {advanced['ts_pct']:.1%}
  - League avg: ~57.5%
  - Elite: >62%

USG% (Usage Rate): {advanced['usg_pct']:.1f}%
  - Average: ~20%
  - High usage star: 30%+

PER (Player Efficiency): {advanced['per']:.1f}
  - League avg: 15.0
  - All-Star: 20+
  - MVP caliber: 25+

BPM (Box Plus/Minus): {advanced['bpm']:.1f}
  - League avg: 0.0
  - All-Star: +4 to +6
  - MVP caliber: +8+

Net Rating: {team_advanced['net_rating']:.1f}
  - +0 = average team
  - +5 = playoff team
  - +10 = championship contender
""")
