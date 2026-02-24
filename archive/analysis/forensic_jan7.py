"""
Forensic Analysis: January 7th, 2026 Prediction Failures

This script performs a deep dive into the model's predictions on January 7th, 2026
to understand why predictions diverged from actual results.

Key Questions:
1. Was it DATA: Did we know about key injuries/rest?
2. Was it FEATURES: Did the model miss matchup advantages?
3. Was it VARIANCE: Were results just outliers (statistical noise)?

Usage:
    python3 analysis/forensic_jan7.py
"""

import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from balldontlie_api import BalldontlieAPI
from comprehensive_backtest import SeasonBacktester, PropPrediction, BacktestResults

# Directories
MODEL_DIR = Path(__file__).parent.parent / "models"
CACHE_DIR = Path(__file__).parent.parent / "data/balldontlie_cache"
OUTPUT_DIR = Path(__file__).parent.parent / "improvement_plan_v7_detailed"


@dataclass
class GameForensics:
    """Forensic analysis for a single game."""
    game_id: int
    game_date: str
    home_team: str
    away_team: str
    actual_home_score: int
    actual_away_score: int

    # Predictions
    predicted_spread: float | None = None
    actual_spread: int = 0

    # Data quality
    home_injuries_known: list[str] = field(default_factory=list)
    away_injuries_known: list[str] = field(default_factory=list)
    home_injuries_missed: list[str] = field(default_factory=list)
    away_injuries_missed: list[str] = field(default_factory=list)

    # Features
    home_off_rating: float = 114.0
    away_off_rating: float = 114.0
    home_def_rating: float = 114.0
    away_def_rating: float = 114.0
    home_pace: float = 100.0
    away_pace: float = 100.0

    # Analysis
    prediction_error: float = 0.0
    error_category: str = "unknown"  # data, features, variance, model
    key_factors: list[str] = field(default_factory=list)


@dataclass
class PropForensics:
    """Forensic analysis for a single player prop."""
    player_name: str
    player_id: int
    prop_type: str
    line: float
    predicted: float
    actual: float

    # Error analysis
    error: float = 0.0
    error_pct: float = 0.0

    # Context
    minutes_played: float = 0.0
    expected_minutes: float = 30.0
    was_injured: bool = False
    teammate_injured: list[str] = field(default_factory=list)
    opponent_key_player_out: list[str] = field(default_factory=list)

    # Root cause
    root_cause: str = "unknown"  # minutes, matchup, variance, model, injury


class ForensicAnalyzer:
    """
    Deep forensic analysis of prediction failures.

    Methodology:
    1. Replay predictions using ONLY data available before game time
    2. Compare features used vs features that would have been ideal
    3. Categorize errors by root cause
    """

    TARGET_DATE = "2026-01-07"

    def __init__(self):
        self.api = None
        self.backtester = None
        self.games = []
        self.prop_predictions = []
        self.game_forensics = []

        # Error categorization
        self.error_summary = {
            'data': [],      # Injury data we didn't have
            'features': [],  # Matchup features we missed
            'variance': [],  # Statistical outliers
            'model': [],     # Model architecture issues
            'minutes': [],   # Unexpected minutes changes
            'injury': [],    # Unexpected DNP (injury-related)
            'unknown': [],   # Unknown causes
        }

    def initialize(self):
        """Initialize API and backtester."""
        print("Initializing forensic analysis...")

        # Initialize Balldontlie API
        try:
            self.api = BalldontlieAPI()
            print("  Balldontlie API: Connected")
        except Exception as e:
            print(f"  Balldontlie API: Error - {e}")

        # Initialize backtester (for consistent feature generation)
        self.backtester = SeasonBacktester(season=2025)
        self.backtester.load_models()
        self.backtester.load_historical_player_stats()

    def fetch_jan7_games(self) -> list[dict]:
        """Fetch all games from January 7th, 2026."""
        print(f"\nFetching games for {self.TARGET_DATE}...")

        if not self.api:
            print("  ERROR: API not initialized")
            return []

        try:
            games = self.api.get_games(dates=[self.TARGET_DATE])
            self.games = [g for g in games if g.get('status') == 'Final']
            print(f"  Found {len(self.games)} completed games")
            return self.games
        except Exception as e:
            print(f"  Error fetching games: {e}")
            return []

    def analyze_game(self, game: dict) -> GameForensics:
        """Perform forensic analysis on a single game."""
        game_id = game['id']
        home_team = game.get('home_team', {})
        away_team = game.get('visitor_team', {})

        home_score = game.get('home_team_score', 0)
        away_score = game.get('visitor_team_score', 0)
        actual_spread = home_score - away_score

        forensics = GameForensics(
            game_id=game_id,
            game_date=self.TARGET_DATE,
            home_team=home_team.get('abbreviation', 'UNK'),
            away_team=away_team.get('abbreviation', 'UNK'),
            actual_home_score=home_score,
            actual_away_score=away_score,
            actual_spread=actual_spread,
        )

        # Fetch box scores for this game
        try:
            stats = self.api.get_player_stats(game_ids=[game_id])

            # Analyze each player's performance
            for stat in stats:
                player = stat.get('player', {})
                player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()

                # Parse minutes
                min_val = stat.get('min', '0')
                if isinstance(min_val, str) and ':' in min_val:
                    parts = min_val.split(':')
                    minutes = float(parts[0]) + float(parts[1]) / 60
                else:
                    minutes = float(min_val) if min_val else 0

                # Check for DNP (key player out)
                if minutes < 5:
                    team_id = stat.get('team', {}).get('id')
                    if team_id == home_team.get('id'):
                        forensics.home_injuries_missed.append(player_name)
                    elif team_id == away_team.get('id'):
                        forensics.away_injuries_missed.append(player_name)

        except Exception as e:
            print(f"  Error fetching box score for game {game_id}: {e}")

        return forensics

    def analyze_prop_prediction(self, pred: PropPrediction, game: dict) -> PropForensics:
        """Analyze a single prop prediction failure."""
        forensics = PropForensics(
            player_name=pred.player_name,
            player_id=pred.player_id,
            prop_type=pred.prop_type,
            line=0.0,  # We don't have the line from backtest
            predicted=pred.predicted,
            actual=pred.actual,
            error=pred.error,
            error_pct=abs(pred.error / max(pred.actual, 1)) * 100 if pred.actual > 0 else 0,
        )

        # Categorize error
        if pred.actual == 0:
            forensics.root_cause = "injury"
            forensics.was_injured = True
        elif abs(pred.error) > 10:
            # Large error - likely data or model issue
            if forensics.error_pct > 50:
                forensics.root_cause = "data"  # Likely missing injury info
            else:
                forensics.root_cause = "model"
        elif abs(pred.error) > 5:
            forensics.root_cause = "features"  # Matchup issue
        else:
            forensics.root_cause = "variance"  # Normal variance

        return forensics

    def run_forensic_analysis(self) -> dict:
        """Run the complete forensic analysis."""
        print("\n" + "=" * 60)
        print("FORENSIC ANALYSIS: JANUARY 7, 2026")
        print("=" * 60)

        # Initialize
        self.initialize()

        # Fetch games
        games = self.fetch_jan7_games()
        if not games:
            print("No games found for analysis")
            return {}

        # Run point-in-time backtest for this date
        print("\nRunning point-in-time predictions...")
        results = self._run_pit_backtest()

        # Analyze each game
        print("\nAnalyzing game-level predictions...")
        for game in games:
            forensics = self.analyze_game(game)
            self.game_forensics.append(forensics)

        # Analyze prop predictions
        print("\nAnalyzing prop predictions...")
        for pred in results.predictions:
            game = next((g for g in games if g['id'] == pred.game_id), None)
            if game:
                prop_forensics = self.analyze_prop_prediction(pred, game)

                # Categorize by root cause
                self.error_summary[prop_forensics.root_cause].append(prop_forensics)

        # Generate summary
        return self._generate_summary(results)

    def _run_pit_backtest(self) -> BacktestResults:
        """Run point-in-time backtest for Jan 7th only."""
        results = BacktestResults()
        results.start_date = self.TARGET_DATE
        results.end_date = self.TARGET_DATE

        if not self.games:
            print("  No games to backtest")
            return results

        print(f"  Backtesting {len(self.games)} games from API...")

        for game in self.games:
            game_id = game['id']
            game_date = game.get('date', self.TARGET_DATE)
            if 'T' in game_date:
                game_date = game_date.split('T')[0]

            home_team = game.get('home_team', {})
            away_team = game.get('visitor_team', {})

            # Get box scores for this game
            try:
                stats = self.api.get_player_stats(game_ids=[game_id])

                for stat in stats:
                    player = stat.get('player', {})
                    player_id = player.get('id')
                    player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
                    player_team_id = stat.get('team', {}).get('id')
                    is_home = player_team_id == home_team.get('id')

                    # Parse minutes
                    min_val = stat.get('min', '0')
                    if isinstance(min_val, str) and ':' in min_val:
                        parts = min_val.split(':')
                        minutes = float(parts[0]) + float(parts[1]) / 60
                    else:
                        minutes = float(min_val) if min_val else 0

                    # Skip players with very few minutes
                    if minutes < 5:
                        continue

                    # Get player features using point-in-time data
                    features = self.backtester.get_player_features_before_date(
                        player_id, game_date,
                        opponent_id=away_team.get('id') if is_home else home_team.get('id'),
                        is_home=is_home
                    )

                    if not features:
                        continue

                    # Make predictions
                    for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
                        pred_value = self.backtester.predict(prop_type, features)
                        if pred_value is None:
                            continue

                        # Get actual value
                        stat_key = {'points': 'pts', 'rebounds': 'reb', 'assists': 'ast',
                                   'threes': 'fg3m', 'pra': 'pra'}[prop_type]

                        if stat_key == 'pra':
                            actual_value = (stat.get('pts', 0) or 0) + \
                                          (stat.get('reb', 0) or 0) + \
                                          (stat.get('ast', 0) or 0)
                        else:
                            actual_value = stat.get(stat_key, 0) or 0

                        # Record prediction
                        pred = PropPrediction(
                            player_id=player_id,
                            player_name=player_name,
                            team=home_team.get('abbreviation', '?') if is_home else away_team.get('abbreviation', '?'),
                            prop_type=prop_type,
                            predicted=pred_value,
                            actual=actual_value,
                            game_id=game_id,
                            game_date=game_date,
                            is_home=is_home,
                            days_rest=features.get('days_rest', 2),
                        )
                        results.add(pred)

                results.games_processed += 1

            except Exception as e:
                print(f"  Error processing game {game_id}: {e}")
                results.games_with_errors += 1

        return results

    def _generate_summary(self, results: BacktestResults) -> dict:
        """Generate forensic analysis summary."""
        summary = {
            'date': self.TARGET_DATE,
            'games_analyzed': len(self.game_forensics),
            'predictions_analyzed': len(results.predictions),
            'overall_metrics': results.calculate_metrics(),
            'error_breakdown': {},
            'worst_predictions': [],
            'key_findings': [],
            'recommendations': [],
        }

        # Error breakdown
        for category, errors in self.error_summary.items():
            if errors:
                avg_error = np.mean([abs(e.error) for e in errors])
                summary['error_breakdown'][category] = {
                    'count': len(errors),
                    'avg_error': round(avg_error, 2),
                    'examples': [
                        f"{e.player_name} {e.prop_type}: pred={e.predicted:.1f}, actual={e.actual:.1f}"
                        for e in sorted(errors, key=lambda x: abs(x.error), reverse=True)[:3]
                    ]
                }

        # Worst predictions
        sorted_preds = sorted(results.predictions, key=lambda p: abs(p.error), reverse=True)[:10]
        for p in sorted_preds:
            summary['worst_predictions'].append({
                'player': p.player_name,
                'prop_type': p.prop_type,
                'predicted': round(p.predicted, 1),
                'actual': round(p.actual, 1),
                'error': round(p.error, 1),
            })

        # Key findings
        if len(self.error_summary['injury']) > 5:
            summary['key_findings'].append(
                f"HIGH INJURY RATE: {len(self.error_summary['injury'])} players DNP unexpectedly"
            )

        if len(self.error_summary['data']) > len(results.predictions) * 0.2:
            summary['key_findings'].append(
                "DATA LATENCY: >20% of errors likely due to missing injury data"
            )

        # By-prop-type analysis
        for prop_type in ['points', 'rebounds', 'assists', 'threes', 'pra']:
            prop_preds = results.get_by_prop_type(prop_type)
            if prop_preds:
                metrics = results.calculate_metrics(prop_preds)
                if metrics.get('rmse', 0) > 8:
                    summary['key_findings'].append(
                        f"{prop_type.upper()}: High RMSE ({metrics['rmse']:.1f}) - needs investigation"
                    )

        # Recommendations
        if self.error_summary['data']:
            summary['recommendations'].append(
                "IMPROVE DATA: Integrate real-time injury feeds (ESPN, Rotowire)"
            )
        if self.error_summary['features']:
            summary['recommendations'].append(
                "ENHANCE FEATURES: Add Four Factors (eFG%, TOV%, ORB%, FT/FGA)"
            )
        if self.error_summary['model']:
            summary['recommendations'].append(
                "UPGRADE MODEL: Implement stacked ensemble with meta-learner"
            )

        return summary

    def print_report(self, summary: dict):
        """Print formatted forensic report."""
        print("\n" + "=" * 60)
        print("FORENSIC ANALYSIS REPORT")
        print("=" * 60)

        print(f"\nDate: {summary['date']}")
        print(f"Games Analyzed: {summary['games_analyzed']}")
        print(f"Predictions Analyzed: {summary['predictions_analyzed']}")

        # Overall metrics
        metrics = summary.get('overall_metrics', {})
        print("\n--- OVERALL ACCURACY ---")
        print(f"RMSE: {metrics.get('rmse', 'N/A')}")
        print(f"MAE: {metrics.get('mae', 'N/A')}")
        print(f"Bias: {metrics.get('bias', 'N/A')}")

        # Error breakdown
        print("\n--- ERROR CATEGORIZATION ---")
        for category, data in summary.get('error_breakdown', {}).items():
            print(f"\n{category.upper()} ({data['count']} errors, avg={data['avg_error']:.1f}):")
            for example in data.get('examples', []):
                print(f"  - {example}")

        # Worst predictions
        print("\n--- TOP 10 WORST PREDICTIONS ---")
        for i, pred in enumerate(summary.get('worst_predictions', [])[:10], 1):
            print(f"{i:2}. {pred['player']:<20} {pred['prop_type']:<10} "
                  f"Pred={pred['predicted']:>6.1f} Actual={pred['actual']:>6.1f} "
                  f"Error={pred['error']:>+7.1f}")

        # Key findings
        print("\n--- KEY FINDINGS ---")
        for finding in summary.get('key_findings', []):
            print(f"  * {finding}")

        # Recommendations
        print("\n--- RECOMMENDATIONS ---")
        for rec in summary.get('recommendations', []):
            print(f"  -> {rec}")

        print("\n" + "=" * 60)

    def save_report(self, summary: dict):
        """Save forensic report to file."""
        OUTPUT_DIR.mkdir(exist_ok=True)

        output_file = OUTPUT_DIR / "forensic_report.md"

        with open(output_file, 'w') as f:
            f.write("# Forensic Analysis Report: January 7, 2026\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")
            f.write(f"- **Games Analyzed**: {summary['games_analyzed']}\n")
            f.write(f"- **Predictions Analyzed**: {summary['predictions_analyzed']}\n")

            metrics = summary.get('overall_metrics', {})
            f.write(f"- **RMSE**: {metrics.get('rmse', 'N/A')}\n")
            f.write(f"- **MAE**: {metrics.get('mae', 'N/A')}\n")
            f.write(f"- **Bias**: {metrics.get('bias', 'N/A')}\n\n")

            f.write("## Error Categorization\n\n")
            for category, data in summary.get('error_breakdown', {}).items():
                f.write(f"### {category.title()} ({data['count']} errors)\n")
                f.write(f"Average Error: {data['avg_error']:.1f}\n\n")
                f.write("Examples:\n")
                for example in data.get('examples', []):
                    f.write(f"- {example}\n")
                f.write("\n")

            f.write("## Key Findings\n\n")
            for finding in summary.get('key_findings', []):
                f.write(f"- {finding}\n")
            f.write("\n")

            f.write("## Recommendations\n\n")
            for i, rec in enumerate(summary.get('recommendations', []), 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")

            f.write("## Worst Predictions\n\n")
            f.write("| Player | Prop | Predicted | Actual | Error |\n")
            f.write("|--------|------|-----------|--------|-------|\n")
            for pred in summary.get('worst_predictions', []):
                f.write(f"| {pred['player']} | {pred['prop_type']} | "
                       f"{pred['predicted']:.1f} | {pred['actual']:.1f} | "
                       f"{pred['error']:+.1f} |\n")

        print(f"\nReport saved to: {output_file}")

        # Also save JSON for programmatic access
        json_file = OUTPUT_DIR / "forensic_jan7_results.json"
        with open(json_file, 'w') as f:
            # Convert to JSON-serializable format
            json_summary = {
                'date': summary['date'],
                'games_analyzed': summary['games_analyzed'],
                'predictions_analyzed': summary['predictions_analyzed'],
                'overall_metrics': summary['overall_metrics'],
                'error_breakdown': {
                    k: {'count': v['count'], 'avg_error': v['avg_error']}
                    for k, v in summary.get('error_breakdown', {}).items()
                },
                'key_findings': summary.get('key_findings', []),
                'recommendations': summary.get('recommendations', []),
            }
            json.dump(json_summary, f, indent=2)
        print(f"JSON saved to: {json_file}")


def main():
    """Main entry point."""
    analyzer = ForensicAnalyzer()

    try:
        summary = analyzer.run_forensic_analysis()
        analyzer.print_report(summary)
        analyzer.save_report(summary)
    except Exception as e:
        print(f"Error during forensic analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
