"""
Coach-specific minutes patterns for the Minutes Oracle.

Contains:
1. COACH_TENDENCIES - Manual lookup table for all 30 NBA coaches
2. CoachTendencyLearner - Class to learn/update tendencies from historical data
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class CoachTendency:
    """Data class for coach-specific minutes patterns."""
    name: str
    team_id: int
    starter_min_avg: float  # Average minutes for starters
    bench_min_avg: float  # Average minutes for bench players
    blowout_pull_lead: int  # Lead at which coach pulls starters
    blowout_pull_deficit: int  # Deficit at which coach pulls starters
    min_variance: str  # 'low', 'medium', 'high' - how variable are rotations
    trusts_young_players: bool  # Does coach give young players meaningful minutes
    load_management_tendency: str  # 'aggressive', 'moderate', 'minimal'
    overtime_usage: str  # 'ride_starters', 'rotate', 'matchup_dependent'
    back_to_back_reduction: float  # Minutes reduction on B2B (0.0 to 0.15)

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'team_id': self.team_id,
            'starter_min_avg': self.starter_min_avg,
            'bench_min_avg': self.bench_min_avg,
            'blowout_pull_lead': self.blowout_pull_lead,
            'blowout_pull_deficit': self.blowout_pull_deficit,
            'min_variance': self.min_variance,
            'trusts_young_players': self.trusts_young_players,
            'load_management_tendency': self.load_management_tendency,
            'overtime_usage': self.overtime_usage,
            'back_to_back_reduction': self.back_to_back_reduction,
        }


# Team IDs for reference (NBA API format)
TEAM_IDS = {
    'ATL': 1610612737, 'BOS': 1610612738, 'BKN': 1610612751, 'CHA': 1610612766,
    'CHI': 1610612741, 'CLE': 1610612739, 'DAL': 1610612742, 'DEN': 1610612743,
    'DET': 1610612765, 'GSW': 1610612744, 'HOU': 1610612745, 'IND': 1610612754,
    'LAC': 1610612746, 'LAL': 1610612747, 'MEM': 1610612763, 'MIA': 1610612748,
    'MIL': 1610612749, 'MIN': 1610612750, 'NOP': 1610612740, 'NYK': 1610612752,
    'OKC': 1610612760, 'ORL': 1610612753, 'PHI': 1610612755, 'PHX': 1610612756,
    'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'TOR': 1610612761,
    'UTA': 1610612762, 'WAS': 1610612764,
}

# Reverse lookup
TEAM_ID_TO_ABBREV = {v: k for k, v in TEAM_IDS.items()}


# Manual lookup table for all 30 NBA coaches (2025-26 season)
# Research-based estimates, will be updated from data
COACH_TENDENCIES: dict[str, CoachTendency] = {
    # Tom Thibodeau - Known for riding starters hard
    'Tom Thibodeau': CoachTendency(
        name='Tom Thibodeau',
        team_id=TEAM_IDS['NYK'],
        starter_min_avg=36.5,
        bench_min_avg=18.0,
        blowout_pull_lead=25,
        blowout_pull_deficit=25,
        min_variance='low',
        trusts_young_players=False,
        load_management_tendency='minimal',
        overtime_usage='ride_starters',
        back_to_back_reduction=0.03,
    ),

    # Steve Kerr - Load manages, trusts bench
    'Steve Kerr': CoachTendency(
        name='Steve Kerr',
        team_id=TEAM_IDS['GSW'],
        starter_min_avg=32.0,
        bench_min_avg=20.0,
        blowout_pull_lead=15,
        blowout_pull_deficit=18,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='aggressive',
        overtime_usage='matchup_dependent',
        back_to_back_reduction=0.08,
    ),

    # Erik Spoelstra - Balanced, matchup-driven
    'Erik Spoelstra': CoachTendency(
        name='Erik Spoelstra',
        team_id=TEAM_IDS['MIA'],
        starter_min_avg=33.5,
        bench_min_avg=19.5,
        blowout_pull_lead=20,
        blowout_pull_deficit=20,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='matchup_dependent',
        back_to_back_reduction=0.06,
    ),

    # Joe Mazzulla - Young coach, aggressive rotations
    'Joe Mazzulla': CoachTendency(
        name='Joe Mazzulla',
        team_id=TEAM_IDS['BOS'],
        starter_min_avg=33.0,
        bench_min_avg=19.0,
        blowout_pull_lead=18,
        blowout_pull_deficit=18,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='ride_starters',
        back_to_back_reduction=0.05,
    ),

    # Mike Budenholzer - Known for load management
    'Mike Budenholzer': CoachTendency(
        name='Mike Budenholzer',
        team_id=TEAM_IDS['PHX'],
        starter_min_avg=31.5,
        bench_min_avg=20.5,
        blowout_pull_lead=15,
        blowout_pull_deficit=15,
        min_variance='low',
        trusts_young_players=False,
        load_management_tendency='aggressive',
        overtime_usage='rotate',
        back_to_back_reduction=0.10,
    ),

    # Tyronn Lue - Rides his guys in playoffs, moderate regular season
    'Tyronn Lue': CoachTendency(
        name='Tyronn Lue',
        team_id=TEAM_IDS['LAC'],
        starter_min_avg=33.0,
        bench_min_avg=19.0,
        blowout_pull_lead=18,
        blowout_pull_deficit=18,
        min_variance='medium',
        trusts_young_players=False,
        load_management_tendency='aggressive',  # Clippers load manage a lot
        overtime_usage='ride_starters',
        back_to_back_reduction=0.08,
    ),

    # JJ Redick - New coach (2024-25), learning patterns
    'JJ Redick': CoachTendency(
        name='JJ Redick',
        team_id=TEAM_IDS['LAL'],
        starter_min_avg=34.0,
        bench_min_avg=18.5,
        blowout_pull_lead=18,
        blowout_pull_deficit=20,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='ride_starters',
        back_to_back_reduction=0.05,
    ),

    # Michael Malone - Trusts his core, consistent minutes
    'Michael Malone': CoachTendency(
        name='Michael Malone',
        team_id=TEAM_IDS['DEN'],
        starter_min_avg=34.5,
        bench_min_avg=17.5,
        blowout_pull_lead=20,
        blowout_pull_deficit=22,
        min_variance='low',
        trusts_young_players=False,
        load_management_tendency='minimal',
        overtime_usage='ride_starters',
        back_to_back_reduction=0.04,
    ),

    # Chris Finch - Balanced approach
    'Chris Finch': CoachTendency(
        name='Chris Finch',
        team_id=TEAM_IDS['MIN'],
        starter_min_avg=34.0,
        bench_min_avg=18.0,
        blowout_pull_lead=18,
        blowout_pull_deficit=18,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='ride_starters',
        back_to_back_reduction=0.05,
    ),

    # Mark Daigneault - Young team, developing players
    'Mark Daigneault': CoachTendency(
        name='Mark Daigneault',
        team_id=TEAM_IDS['OKC'],
        starter_min_avg=33.5,
        bench_min_avg=19.5,
        blowout_pull_lead=15,
        blowout_pull_deficit=15,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='rotate',
        back_to_back_reduction=0.06,
    ),

    # Doc Rivers - Known for short rotations
    'Doc Rivers': CoachTendency(
        name='Doc Rivers',
        team_id=TEAM_IDS['MIL'],
        starter_min_avg=35.0,
        bench_min_avg=17.0,
        blowout_pull_lead=22,
        blowout_pull_deficit=22,
        min_variance='low',
        trusts_young_players=False,
        load_management_tendency='minimal',
        overtime_usage='ride_starters',
        back_to_back_reduction=0.04,
    ),

    # Ime Udoka - Defensive minded, rides starters
    'Ime Udoka': CoachTendency(
        name='Ime Udoka',
        team_id=TEAM_IDS['HOU'],
        starter_min_avg=34.0,
        bench_min_avg=18.0,
        blowout_pull_lead=18,
        blowout_pull_deficit=18,
        min_variance='low',
        trusts_young_players=True,
        load_management_tendency='minimal',
        overtime_usage='ride_starters',
        back_to_back_reduction=0.04,
    ),

    # Taylor Jenkins - Player development focus
    'Taylor Jenkins': CoachTendency(
        name='Taylor Jenkins',
        team_id=TEAM_IDS['MEM'],
        starter_min_avg=33.0,
        bench_min_avg=19.0,
        blowout_pull_lead=18,
        blowout_pull_deficit=18,
        min_variance='high',  # Injuries cause high variance
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='matchup_dependent',
        back_to_back_reduction=0.06,
    ),

    # Rick Carlisle - Old school, trusts veterans
    'Rick Carlisle': CoachTendency(
        name='Rick Carlisle',
        team_id=TEAM_IDS['IND'],
        starter_min_avg=33.5,
        bench_min_avg=18.5,
        blowout_pull_lead=20,
        blowout_pull_deficit=20,
        min_variance='low',
        trusts_young_players=False,
        load_management_tendency='minimal',
        overtime_usage='ride_starters',
        back_to_back_reduction=0.04,
    ),

    # Jason Kidd - Balanced, playoff experienced
    'Jason Kidd': CoachTendency(
        name='Jason Kidd',
        team_id=TEAM_IDS['DAL'],
        starter_min_avg=34.0,
        bench_min_avg=18.0,
        blowout_pull_lead=18,
        blowout_pull_deficit=20,
        min_variance='medium',
        trusts_young_players=False,
        load_management_tendency='moderate',
        overtime_usage='ride_starters',
        back_to_back_reduction=0.05,
    ),

    # Kenny Atkinson - Motion offense, spreads minutes
    'Kenny Atkinson': CoachTendency(
        name='Kenny Atkinson',
        team_id=TEAM_IDS['CLE'],
        starter_min_avg=32.5,
        bench_min_avg=20.0,
        blowout_pull_lead=15,
        blowout_pull_deficit=15,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='rotate',
        back_to_back_reduction=0.06,
    ),

    # Willie Green - Development focused
    'Willie Green': CoachTendency(
        name='Willie Green',
        team_id=TEAM_IDS['NOP'],
        starter_min_avg=33.5,
        bench_min_avg=18.5,
        blowout_pull_lead=18,
        blowout_pull_deficit=18,
        min_variance='high',  # Injury-prone roster
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='matchup_dependent',
        back_to_back_reduction=0.06,
    ),

    # Nick Nurse - Analytics driven, varies rotations
    'Nick Nurse': CoachTendency(
        name='Nick Nurse',
        team_id=TEAM_IDS['PHI'],
        starter_min_avg=33.0,
        bench_min_avg=19.0,
        blowout_pull_lead=18,
        blowout_pull_deficit=18,
        min_variance='high',  # Known for unpredictable rotations
        trusts_young_players=True,
        load_management_tendency='aggressive',  # Embiid management
        overtime_usage='matchup_dependent',
        back_to_back_reduction=0.08,
    ),

    # Chauncey Billups - Balanced approach
    'Chauncey Billups': CoachTendency(
        name='Chauncey Billups',
        team_id=TEAM_IDS['POR'],
        starter_min_avg=32.0,
        bench_min_avg=19.5,
        blowout_pull_lead=15,
        blowout_pull_deficit=15,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='rotate',
        back_to_back_reduction=0.06,
    ),

    # Mike Brown - Defensive focus, consistent rotations
    'Mike Brown': CoachTendency(
        name='Mike Brown',
        team_id=TEAM_IDS['SAC'],
        starter_min_avg=33.5,
        bench_min_avg=18.5,
        blowout_pull_lead=18,
        blowout_pull_deficit=18,
        min_variance='low',
        trusts_young_players=True,
        load_management_tendency='minimal',
        overtime_usage='ride_starters',
        back_to_back_reduction=0.04,
    ),

    # Gregg Popovich - The GOAT, load management pioneer
    'Gregg Popovich': CoachTendency(
        name='Gregg Popovich',
        team_id=TEAM_IDS['SAS'],
        starter_min_avg=31.0,
        bench_min_avg=20.5,
        blowout_pull_lead=15,
        blowout_pull_deficit=15,
        min_variance='medium',
        trusts_young_players=True,  # Wemby development
        load_management_tendency='aggressive',
        overtime_usage='rotate',
        back_to_back_reduction=0.10,
    ),

    # Darko Rajakovic - Modern approach
    'Darko Rajakovic': CoachTendency(
        name='Darko Rajakovic',
        team_id=TEAM_IDS['TOR'],
        starter_min_avg=33.0,
        bench_min_avg=19.0,
        blowout_pull_lead=15,
        blowout_pull_deficit=15,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='rotate',
        back_to_back_reduction=0.06,
    ),

    # Will Hardy - Modern analytics
    'Will Hardy': CoachTendency(
        name='Will Hardy',
        team_id=TEAM_IDS['UTA'],
        starter_min_avg=32.0,
        bench_min_avg=20.0,
        blowout_pull_lead=15,
        blowout_pull_deficit=15,
        min_variance='high',  # Tank mode variance
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='rotate',
        back_to_back_reduction=0.06,
    ),

    # Brian Keefe - New coach
    'Brian Keefe': CoachTendency(
        name='Brian Keefe',
        team_id=TEAM_IDS['WAS'],
        starter_min_avg=32.0,
        bench_min_avg=19.5,
        blowout_pull_lead=15,
        blowout_pull_deficit=15,
        min_variance='high',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='rotate',
        back_to_back_reduction=0.06,
    ),

    # Charles Lee - New coach (2024)
    'Charles Lee': CoachTendency(
        name='Charles Lee',
        team_id=TEAM_IDS['CHA'],
        starter_min_avg=32.5,
        bench_min_avg=19.5,
        blowout_pull_lead=15,
        blowout_pull_deficit=15,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='rotate',
        back_to_back_reduction=0.06,
    ),

    # Billy Donovan - Balanced, player-friendly
    'Billy Donovan': CoachTendency(
        name='Billy Donovan',
        team_id=TEAM_IDS['CHI'],
        starter_min_avg=33.0,
        bench_min_avg=19.0,
        blowout_pull_lead=18,
        blowout_pull_deficit=18,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='matchup_dependent',
        back_to_back_reduction=0.05,
    ),

    # JB Bickerstaff - Development focus
    'JB Bickerstaff': CoachTendency(
        name='JB Bickerstaff',
        team_id=TEAM_IDS['DET'],
        starter_min_avg=32.0,
        bench_min_avg=19.5,
        blowout_pull_lead=15,
        blowout_pull_deficit=15,
        min_variance='high',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='rotate',
        back_to_back_reduction=0.06,
    ),

    # Quin Snyder - Analytics pioneer
    'Quin Snyder': CoachTendency(
        name='Quin Snyder',
        team_id=TEAM_IDS['ATL'],
        starter_min_avg=33.0,
        bench_min_avg=19.0,
        blowout_pull_lead=18,
        blowout_pull_deficit=18,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='matchup_dependent',
        back_to_back_reduction=0.06,
    ),

    # Jordi Fernandez - New coach (2024)
    'Jordi Fernandez': CoachTendency(
        name='Jordi Fernandez',
        team_id=TEAM_IDS['BKN'],
        starter_min_avg=32.0,
        bench_min_avg=19.5,
        blowout_pull_lead=15,
        blowout_pull_deficit=15,
        min_variance='high',  # Rebuilding team
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='rotate',
        back_to_back_reduction=0.06,
    ),

    # Jamahl Mosley - Player development
    'Jamahl Mosley': CoachTendency(
        name='Jamahl Mosley',
        team_id=TEAM_IDS['ORL'],
        starter_min_avg=33.0,
        bench_min_avg=19.0,
        blowout_pull_lead=18,
        blowout_pull_deficit=18,
        min_variance='medium',
        trusts_young_players=True,
        load_management_tendency='moderate',
        overtime_usage='ride_starters',
        back_to_back_reduction=0.05,
    ),
}

# Lookup by team ID
COACH_BY_TEAM_ID: dict[int, CoachTendency] = {
    coach.team_id: coach for coach in COACH_TENDENCIES.values()
}

# Default tendency for unknown coaches
DEFAULT_COACH_TENDENCY = CoachTendency(
    name='Unknown',
    team_id=0,
    starter_min_avg=33.0,
    bench_min_avg=19.0,
    blowout_pull_lead=18,
    blowout_pull_deficit=18,
    min_variance='medium',
    trusts_young_players=True,
    load_management_tendency='moderate',
    overtime_usage='matchup_dependent',
    back_to_back_reduction=0.06,
)


def get_coach_tendency(team_id: int | None = None,
                       coach_name: str | None = None) -> CoachTendency:
    """
    Get coach tendency by team ID or coach name.

    Args:
        team_id: NBA team ID
        coach_name: Coach's name

    Returns:
        CoachTendency object
    """
    if coach_name and coach_name in COACH_TENDENCIES:
        return COACH_TENDENCIES[coach_name]

    if team_id and team_id in COACH_BY_TEAM_ID:
        return COACH_BY_TEAM_ID[team_id]

    return DEFAULT_COACH_TENDENCY


class CoachTendencyLearner:
    """
    Learn coach tendencies from historical game data.

    Analyzes game logs to calculate:
    - Average starter minutes by coach
    - Blowout pull patterns (at what lead/deficit do starters sit)
    - Minutes variance by coach
    - Back-to-back reduction patterns
    """

    def __init__(self):
        self.coach_data: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.games_processed = 0

    def add_game(self,
                 coach_name: str,
                 team_id: int,
                 player_minutes: list[dict],
                 final_margin: int,
                 is_back_to_back: bool,
                 went_to_overtime: bool):
        """
        Add a game's data to learn from.

        Args:
            coach_name: Name of the coach
            team_id: Team ID
            player_minutes: List of {'player_id', 'minutes', 'is_starter'}
            final_margin: Final score margin (positive = win)
            is_back_to_back: Whether this was a B2B game
            went_to_overtime: Whether game went to OT
        """
        if not coach_name or not player_minutes:
            return

        key = f"{coach_name}_{team_id}"

        # Calculate starter and bench minutes
        starter_mins = [p['minutes'] for p in player_minutes
                       if p.get('is_starter', False) and p['minutes'] > 0]
        bench_mins = [p['minutes'] for p in player_minutes
                     if not p.get('is_starter', False) and p['minutes'] > 5]

        if starter_mins:
            avg_starter_min = np.mean(starter_mins)
            self.coach_data[key]['starter_mins'].append(avg_starter_min)

            # Track blowout patterns
            if abs(final_margin) >= 20:
                self.coach_data[key]['blowout_starter_mins'].append(avg_starter_min)
                self.coach_data[key]['blowout_margins'].append(abs(final_margin))

            # Track B2B patterns
            if is_back_to_back:
                self.coach_data[key]['b2b_starter_mins'].append(avg_starter_min)
            else:
                self.coach_data[key]['normal_starter_mins'].append(avg_starter_min)

            # Track OT patterns
            if went_to_overtime:
                self.coach_data[key]['ot_starter_mins'].append(avg_starter_min)

        if bench_mins:
            self.coach_data[key]['bench_mins'].append(np.mean(bench_mins))

        self.games_processed += 1

    def calculate_tendencies(self, min_games: int = 20) -> dict[str, CoachTendency]:
        """
        Calculate coach tendencies from accumulated data.

        Args:
            min_games: Minimum games required to calculate tendencies

        Returns:
            Dictionary of coach name -> CoachTendency
        """
        learned_tendencies = {}

        for key, data in self.coach_data.items():
            if len(data.get('starter_mins', [])) < min_games:
                continue

            coach_name, team_id = key.rsplit('_', 1)
            team_id = int(team_id)

            starter_mins = data['starter_mins']
            bench_mins = data.get('bench_mins', [])
            blowout_mins = data.get('blowout_starter_mins', [])
            normal_mins = data.get('normal_starter_mins', [])
            b2b_mins = data.get('b2b_starter_mins', [])

            # Calculate starter average
            starter_avg = np.mean(starter_mins)

            # Calculate bench average
            bench_avg = np.mean(bench_mins) if bench_mins else 19.0

            # Calculate minutes variance
            starter_std = np.std(starter_mins)
            if starter_std < 2.5:
                variance = 'low'
            elif starter_std < 4.0:
                variance = 'medium'
            else:
                variance = 'high'

            # Estimate blowout pull threshold
            if blowout_mins and normal_mins:
                blowout_reduction = np.mean(normal_mins) - np.mean(blowout_mins)
                # If starters play 3+ fewer minutes in blowouts, coach pulls early
                if blowout_reduction > 5:
                    blowout_pull = 15
                elif blowout_reduction > 3:
                    blowout_pull = 18
                else:
                    blowout_pull = 22
            else:
                blowout_pull = 18

            # Calculate B2B reduction
            if b2b_mins and normal_mins:
                b2b_reduction = (np.mean(normal_mins) - np.mean(b2b_mins)) / np.mean(normal_mins)
                b2b_reduction = max(0.0, min(0.15, b2b_reduction))
            else:
                b2b_reduction = 0.06

            # Determine load management tendency
            if starter_avg < 32:
                load_mgmt = 'aggressive'
            elif starter_avg < 34:
                load_mgmt = 'moderate'
            else:
                load_mgmt = 'minimal'

            learned_tendencies[coach_name] = CoachTendency(
                name=coach_name,
                team_id=team_id,
                starter_min_avg=round(starter_avg, 1),
                bench_min_avg=round(bench_avg, 1),
                blowout_pull_lead=blowout_pull,
                blowout_pull_deficit=blowout_pull,
                min_variance=variance,
                trusts_young_players=True,  # Hard to learn from data
                load_management_tendency=load_mgmt,
                overtime_usage='matchup_dependent',  # Hard to learn from limited OT data
                back_to_back_reduction=round(b2b_reduction, 2),
            )

        return learned_tendencies

    def update_global_tendencies(self, min_games: int = 20):
        """
        Update the global COACH_TENDENCIES dict with learned values.

        Only updates coaches where we have sufficient data.
        """
        learned = self.calculate_tendencies(min_games)

        for coach_name, tendency in learned.items():
            if coach_name in COACH_TENDENCIES:
                # Update existing entry with learned values
                existing = COACH_TENDENCIES[coach_name]
                COACH_TENDENCIES[coach_name] = CoachTendency(
                    name=coach_name,
                    team_id=tendency.team_id,
                    starter_min_avg=tendency.starter_min_avg,
                    bench_min_avg=tendency.bench_min_avg,
                    blowout_pull_lead=tendency.blowout_pull_lead,
                    blowout_pull_deficit=tendency.blowout_pull_deficit,
                    min_variance=tendency.min_variance,
                    trusts_young_players=existing.trusts_young_players,
                    load_management_tendency=tendency.load_management_tendency,
                    overtime_usage=existing.overtime_usage,
                    back_to_back_reduction=tendency.back_to_back_reduction,
                )
                # Update team lookup
                COACH_BY_TEAM_ID[tendency.team_id] = COACH_TENDENCIES[coach_name]
            else:
                # Add new coach
                COACH_TENDENCIES[coach_name] = tendency
                COACH_BY_TEAM_ID[tendency.team_id] = tendency

        return len(learned)

    def get_summary(self) -> str:
        """Get a summary of learned data."""
        lines = ["Coach Tendency Learner Summary:",
                 f"  Games processed: {self.games_processed}",
                 f"  Coaches tracked: {len(self.coach_data)}",
                 ""]

        for key, data in sorted(self.coach_data.items()):
            games = len(data.get('starter_mins', []))
            if games > 0:
                coach_name = key.rsplit('_', 1)[0]
                avg_mins = np.mean(data['starter_mins'])
                lines.append(f"  {coach_name}: {games} games, {avg_mins:.1f} avg starter mins")

        return "\n".join(lines)


# Convenience function to get minutes adjustment factors
def get_blowout_minutes_factor(team_id: int,
                                expected_margin: float,
                                is_winning: bool = True) -> float:
    """
    Get expected minutes adjustment factor for blowout scenarios.

    Args:
        team_id: Team ID
        expected_margin: Expected final margin (absolute value)
        is_winning: Whether team is expected to win

    Returns:
        Multiplier for expected minutes (e.g., 0.85 = 15% reduction)
    """
    coach = get_coach_tendency(team_id=team_id)
    threshold = coach.blowout_pull_lead if is_winning else coach.blowout_pull_deficit

    if expected_margin < threshold:
        return 1.0  # No reduction expected

    # Calculate reduction based on how much over threshold
    excess = expected_margin - threshold

    # Each 5 points over threshold = ~3% reduction
    reduction = min(0.20, excess * 0.006)

    return 1.0 - reduction


def get_b2b_minutes_factor(team_id: int) -> float:
    """
    Get expected minutes reduction factor for back-to-back games.

    Args:
        team_id: Team ID

    Returns:
        Multiplier for expected minutes (e.g., 0.94 = 6% reduction)
    """
    coach = get_coach_tendency(team_id=team_id)
    return 1.0 - coach.back_to_back_reduction
