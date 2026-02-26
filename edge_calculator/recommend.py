#!/usr/bin/env python3
"""
CLI for Bet Recommendations

Usage:
    python -m edge_calculator.recommend --date today --bankroll 1000
    python -m edge_calculator.recommend --date 2024-01-15 --bankroll 500 --max-picks 5

Output:
    ╔══════════════════════════════════════════════════════════════════╗
    ║  TODAY'S RECOMMENDED BETS (3 of 47 props analyzed)              ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  1. LeBron James      OVER 26.5 pts  │ 1.5u │ +4.8% edge │ STR  ║
    ║  2. Nikola Jokic      OVER 11.5 reb  │ 2.0u │ +6.2% edge │ STR  ║
    ║  3. Haliburton        UNDER 9.5 ast  │ 1.0u │ +3.1% edge │ MOD  ║
    ╚══════════════════════════════════════════════════════════════════╝
"""

import sys
import argparse
import json
import logging
from datetime import datetime, timedelta

from .bet_recommender import (
    BetRecommender,
    BetRecommendation,
    ConfidenceTier,
    format_recommendations_table,
)
from .bankroll_manager import BankrollManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> str:
    """Parse date string to YYYY-MM-DD format."""
    if date_str.lower() == 'today':
        return datetime.now().strftime('%Y-%m-%d')
    elif date_str.lower() == 'tomorrow':
        return (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    elif date_str.lower() == 'yesterday':
        return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        # Assume YYYY-MM-DD format
        return date_str


def load_props_from_predictions(date: str) -> list[dict]:
    """
    Load props from the prediction pipeline.

    This function should integrate with your existing prediction system.
    For now, returns sample data.
    """
    # Try to load from data service
    try:
        # This would integrate with your existing DataService
        # from dashboard.data_service import DataService
        # service = DataService()
        # predictions = service.get_predictions_for_date(date)
        # return predictions
        pass
    except ImportError:
        pass

    # Sample data for demonstration
    logger.warning("Using sample data - integrate with your prediction pipeline")

    return [
        {
            'player_name': 'LeBron James',
            'player_id': 2544,
            'team': 'LAL',
            'opponent': 'BOS',
            'prop_type': 'points',
            'line': 26.5,
            'prediction': 28.2,
            'confidence': 65,
            'game_date': date,
            'game_context': {
                'spread': -3.5,
                'total': 225.5,
                'is_home': True,
                'is_b2b': False,
                'minutes_predicted': 35,
                'minutes_uncertainty': 'low',
            }
        },
        {
            'player_name': 'Nikola Jokic',
            'player_id': 203999,
            'team': 'DEN',
            'opponent': 'PHX',
            'prop_type': 'rebounds',
            'line': 11.5,
            'prediction': 13.5,
            'confidence': 70,
            'game_date': date,
            'game_context': {
                'spread': -6.5,
                'is_b2b': False,
                'minutes_predicted': 34,
                'minutes_uncertainty': 'low',
            }
        },
        {
            'player_name': 'Tyrese Haliburton',
            'player_id': 1630169,
            'team': 'IND',
            'opponent': 'MIA',
            'prop_type': 'assists',
            'line': 9.5,
            'prediction': 8.2,
            'confidence': 62,
            'game_date': date,
            'game_context': {
                'spread': 2.5,
                'is_b2b': True,
                'minutes_predicted': 33,
                'minutes_uncertainty': 'medium',
            }
        },
        {
            'player_name': 'Stephen Curry',
            'player_id': 201939,
            'team': 'GSW',
            'opponent': 'LAC',
            'prop_type': 'threes',
            'line': 4.5,
            'prediction': 5.1,
            'confidence': 58,
            'game_date': date,
            'game_context': {
                'spread': -1.5,
                'is_b2b': False,
                'minutes_predicted': 32,
                'minutes_uncertainty': 'medium',
            }
        },
        {
            'player_name': 'Giannis Antetokounmpo',
            'player_id': 203507,
            'team': 'MIL',
            'opponent': 'PHI',
            'prop_type': 'points',
            'line': 29.5,
            'prediction': 31.8,
            'confidence': 68,
            'game_date': date,
            'game_context': {
                'spread': -5.0,
                'is_b2b': False,
                'minutes_predicted': 34,
                'minutes_uncertainty': 'low',
            }
        },
        {
            'player_name': 'Anthony Davis',
            'player_id': 203076,
            'team': 'LAL',
            'opponent': 'BOS',
            'prop_type': 'rebounds',
            'line': 11.5,
            'prediction': 12.3,
            'confidence': 55,
            'game_date': date,
            'game_context': {
                'spread': -3.5,
                'is_b2b': False,
                'minutes_predicted': 35,
                'minutes_uncertainty': 'medium',
            }
        },
        {
            'player_name': 'Jayson Tatum',
            'player_id': 1628369,
            'team': 'BOS',
            'opponent': 'LAL',
            'prop_type': 'points',
            'line': 27.5,
            'prediction': 25.8,
            'confidence': 52,
            'game_date': date,
            'game_context': {
                'spread': 3.5,
                'is_b2b': False,
                'minutes_predicted': 36,
                'minutes_uncertainty': 'low',
                'injury_status': 'Questionable',
            }
        },
    ]



def print_detailed_pick(rec: BetRecommendation):
    """Print detailed information for a single pick."""
    print()
    print("─" * 60)
    print(f"  {rec.player_name} ({rec.team}) vs {rec.opponent}")
    print("─" * 60)
    print(f"  Prop: {rec.prop_type.upper()} {rec.pick} {rec.line}")
    print(f"  Model Prediction: {rec.prediction:.1f}")
    print(f"  Edge: {rec.edge_percentage:+.1f}% (EV: ${rec.ev_per_dollar:+.3f}/dollar)")
    print(f"  Tier: {rec.confidence_tier.value.upper()}")
    print()
    print(f"  RECOMMENDATION: {rec.recommendation}")
    print(f"  Suggested Bet: {rec.suggested_units:.1f} units (${rec.suggested_stake:.2f})")
    print()
    print("  Reasoning:")
    for r in rec.reasoning:
        print(f"    • {r}")
    if rec.risks:
        print()
        print("  Risks:")
        for r in rec.risks:
            print(f"    ⚠ {r}")


def run_recommendations(
    date: str,
    bankroll: float,
    max_picks: int,
    max_units: float,
    verbose: bool = False,
    output_json: bool = False,
):
    """
    Run the recommendation engine.

    Args:
        date: Date to analyze
        bankroll: Current bankroll
        max_picks: Maximum number of picks
        max_units: Maximum total units
        verbose: Print detailed output
        output_json: Output as JSON
    """
    date = parse_date(date)

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  EDGE CALCULATOR - NBA PROP BET RECOMMENDATIONS".center(68) + "║")
    print("║" + f"  Date: {date}  |  Bankroll: ${bankroll:.2f}".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Initialize recommender
    recommender = BetRecommender(bankroll=bankroll)

    # Load props
    print("Loading predictions...")
    props = load_props_from_predictions(date)
    print(f"Found {len(props)} props to analyze")
    print()

    # Get recommendations
    print("Analyzing edge and sizing...")
    picks, summary = recommender.get_top_picks(
        props,
        max_picks=max_picks,
        max_total_units=max_units,
    )

    if output_json:
        output = {
            'date': date,
            'bankroll': bankroll,
            'summary': summary,
            'picks': [p.to_dict() for p in picks],
        }
        print(json.dumps(output, indent=2))
        return

    # Print table
    print(format_recommendations_table(picks, summary))

    # Print detailed picks if verbose
    if verbose and picks:
        print("\n" + "=" * 68)
        print("  DETAILED ANALYSIS")
        print("=" * 68)
        for rec in picks:
            print_detailed_pick(rec)

    # Print all analyzed props summary
    print()
    print("─" * 68)
    all_recs = recommender.analyze_props(props)
    by_tier = {
        'STRONG': [r for r in all_recs if r.confidence_tier == ConfidenceTier.STRONG],
        'MODERATE': [r for r in all_recs if r.confidence_tier == ConfidenceTier.MODERATE],
        'MARGINAL': [r for r in all_recs if r.confidence_tier == ConfidenceTier.MARGINAL],
        'PASS': [r for r in all_recs if r.confidence_tier == ConfidenceTier.PASS],
    }
    print(f"  All Props: {len(all_recs)} analyzed")
    print(f"    STRONG: {len(by_tier['STRONG'])} | MODERATE: {len(by_tier['MODERATE'])} | "
          f"MARGINAL: {len(by_tier['MARGINAL'])} | PASS: {len(by_tier['PASS'])}")
    print()

    # Show passed props with close to edge threshold
    close_misses = [r for r in all_recs
                    if r.confidence_tier == ConfidenceTier.PASS
                    and r.edge_percentage >= 1.5]
    if close_misses and verbose:
        print("  Near-Misses (1.5-2% edge):")
        for r in close_misses[:3]:
            print(f"    {r.player_name} {r.pick} {r.line} {r.prop_type}: {r.edge_percentage:+.1f}%")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate NBA prop bet recommendations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m edge_calculator.recommend --date today --bankroll 1000
  python -m edge_calculator.recommend --date tomorrow --bankroll 500 --max-picks 3
  python -m edge_calculator.recommend --date 2024-01-15 --verbose
  python -m edge_calculator.recommend --json > picks.json
        """
    )

    parser.add_argument(
        '--date', '-d',
        type=str,
        default='today',
        help='Date to analyze (today, tomorrow, or YYYY-MM-DD)'
    )
    parser.add_argument(
        '--bankroll', '-b',
        type=float,
        default=1000,
        help='Current bankroll (default: 1000)'
    )
    parser.add_argument(
        '--max-picks', '-p',
        type=int,
        default=5,
        help='Maximum number of picks (default: 5)'
    )
    parser.add_argument(
        '--max-units', '-u',
        type=float,
        default=10.0,
        help='Maximum total units (default: 10.0)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed analysis for each pick'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )

    args = parser.parse_args()

    try:
        run_recommendations(
            date=args.date,
            bankroll=args.bankroll,
            max_picks=args.max_picks,
            max_units=args.max_units,
            verbose=args.verbose,
            output_json=args.json,
        )
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
