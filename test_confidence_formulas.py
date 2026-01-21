#!/usr/bin/env python3
"""
Test different confidence formulas against backtest data.

This script loads backtest results and simulates different confidence
formulas to find the optimal balance of bet count, win rate, and ROI.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class BettingResults:
    """Results from testing a confidence formula."""
    formula_name: str
    multiplier: float
    total_bets: int
    total_predictions: int
    bet_percentage: float
    wins: int
    losses: int
    win_rate: float
    total_profit: float
    roi: float
    avg_confidence: float
    avg_edge: float

    def __str__(self):
        return f"""
{self.formula_name} (multiplier={self.multiplier})
{'='*60}
Total Predictions: {self.total_predictions:,}
Bets Generated: {self.total_bets:,} ({self.bet_percentage:.1f}%)
Win Rate: {self.win_rate:.1%} ({self.wins}W-{self.losses}L)
ROI: {self.roi:.2%}
Total Profit: ${self.total_profit:,.2f}
Avg Confidence: {self.avg_confidence:.1f}%
Avg Edge: {self.avg_edge:.1f}%
"""


def calculate_confidence(band_width: float, multiplier: float) -> float:
    """Calculate confidence score using given multiplier."""
    return max(40.0, min(90.0, 90.0 - (band_width * multiplier)))


def simulate_betting(predictions: List[Dict], multiplier: float,
                     confidence_threshold: float = 65.0,
                     bet_size: float = 100.0) -> BettingResults:
    """
    Simulate betting with a given confidence formula.

    Args:
        predictions: List of prediction dicts from backtest
        multiplier: Band width multiplier for confidence formula
        confidence_threshold: Minimum confidence to bet (default 65%)
        bet_size: Base bet size in dollars (default $100)

    Returns:
        BettingResults with performance metrics
    """
    total_predictions = len(predictions)
    bets_made = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    confidence_sum = 0.0
    edge_sum = 0.0

    for pred in predictions:
        # Extract prediction data
        predicted = pred['predicted']
        actual = pred['actual']
        prop_type = pred['prop_type']

        # Skip if missing data
        if predicted is None or actual is None:
            continue

        # Calculate band width (simulate quantile predictions)
        # Use RMSE from backtest as proxy for band width
        # Points: RMSE=6.65, Rebounds: 2.68, Assists: 2.01, Threes: 1.39, PRA: 8.15
        rmse_map = {
            'points': 6.65,
            'rebounds': 2.68,
            'assists': 2.01,
            'threes': 1.39,
            'pra': 8.15
        }

        # Estimate band width as 2.5x RMSE (covers ~95% of predictions)
        band_width = rmse_map.get(prop_type, 5.0) * 2.5

        # Calculate confidence with this formula
        confidence = calculate_confidence(band_width, multiplier)

        # Skip if confidence below threshold
        if confidence < confidence_threshold:
            continue

        bets_made += 1
        confidence_sum += confidence

        # Calculate edge (prediction - actual)
        edge = abs(predicted - actual)
        edge_sum += edge

        # Determine win/loss
        # Use absolute error - smaller error = more likely to win
        # Based on backtest results: MAE=3.50, RMSE=5.27
        # Win if prediction is within 1 standard error (RMSE)

        abs_error = abs(predicted - actual)

        # Win threshold based on prop type RMSE
        # Points: 6.65, Rebounds: 2.68, Assists: 2.01, Threes: 1.39, PRA: 8.15
        win_threshold_map = {
            'points': 6.65,
            'rebounds': 2.68,
            'assists': 2.01,
            'threes': 1.39,
            'pra': 8.15
        }
        win_threshold = win_threshold_map.get(prop_type, 5.0)

        # Win if within 1 RMSE (approximately 68% of predictions)
        if abs_error <= win_threshold:
            wins += 1
            # Profit: bet_size * 0.91 (assuming -110 odds)
            total_profit += bet_size * 0.91
        else:
            losses += 1
            # Loss: -bet_size
            total_profit -= bet_size

    # Calculate metrics
    win_rate = wins / bets_made if bets_made > 0 else 0.0
    roi = (total_profit / (bets_made * bet_size)) if bets_made > 0 else 0.0
    bet_percentage = (bets_made / total_predictions * 100) if total_predictions > 0 else 0.0
    avg_confidence = confidence_sum / bets_made if bets_made > 0 else 0.0
    avg_edge = edge_sum / bets_made if bets_made > 0 else 0.0

    return BettingResults(
        formula_name=f"90 - (band_width × {multiplier})",
        multiplier=multiplier,
        total_bets=bets_made,
        total_predictions=total_predictions,
        bet_percentage=bet_percentage,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_profit=total_profit,
        roi=roi,
        avg_confidence=avg_confidence,
        avg_edge=avg_edge
    )


def main():
    """Test multiple confidence formulas and compare results."""
    print("="*60)
    print("CONFIDENCE FORMULA TESTING")
    print("="*60)

    # Load backtest results
    backtest_file = Path("backtest_results_2025_quick.json")

    if not backtest_file.exists():
        print(f"ERROR: {backtest_file} not found")
        print("Run: python3 comprehensive_backtest.py --quick")
        return

    print(f"\nLoading backtest data from {backtest_file}...")
    with open(backtest_file, 'r') as f:
        data = json.load(f)

    predictions = data.get('raw_predictions', [])

    if not predictions:
        print("ERROR: No predictions found in backtest results")
        return

    print(f"Loaded {len(predictions):,} predictions")
    print(f"Date range: {data.get('start_date')} to {data.get('end_date')}")
    print(f"Games: {data.get('games_processed')}")

    # Test different multipliers
    multipliers = [
        6.25,  # Current (conservative)
        5.50,  # Moderate
        5.00,  # Moderate-aggressive
        4.50,  # Aggressive
        4.00,  # Very aggressive
        3.50,  # Extremely aggressive
        3.00,  # Maximum aggressive (IMPLEMENTED)
    ]

    results = []

    print("\n" + "="*60)
    print("TESTING FORMULAS")
    print("="*60)

    for mult in multipliers:
        result = simulate_betting(predictions, mult)
        results.append(result)
        print(result)

    # Find best formula
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    print(f"\n{'Multiplier':<12} {'Bets':<8} {'Win%':<8} {'ROI':<10} {'Confidence':<12}")
    print("-"*60)

    for r in results:
        print(f"{r.multiplier:<12.2f} {r.total_bets:<8} {r.win_rate*100:<8.1f} {r.roi*100:<10.2f} {r.avg_confidence:<12.1f}")

    # Recommend best formula
    # Filter: min 50 bets, min 52% win rate
    viable = [r for r in results if r.total_bets >= 50 and r.win_rate >= 0.52]

    if viable:
        best = max(viable, key=lambda r: r.roi)
        print("\n" + "="*60)
        print("RECOMMENDED FORMULA")
        print("="*60)
        print(best)

        print("\nIMPLEMENTATION:")
        print(f"Update daily_predictions.py line ~1589:")
        print(f"  OLD: confidence_score = max(40.0, min(90.0, 90.0 - (band_width * 6.25)))")
        print(f"  NEW: confidence_score = max(40.0, min(90.0, 90.0 - (band_width * {best.multiplier})))")
    else:
        print("\n⚠️  WARNING: No viable formulas found (need ≥50 bets and ≥52% win rate)")
        print("Consider:")
        print("  1. Testing more aggressive multipliers (< 3.5)")
        print("  2. Lowering confidence threshold (< 65%)")
        print("  3. Improving model to reduce band width")


if __name__ == '__main__':
    main()
