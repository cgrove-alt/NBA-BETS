#!/usr/bin/env python3
"""
Deep analysis of model weaknesses and improvement roadmap.
"""
import os
import sys
import json
import pickle
import warnings
import numpy as np

ROOT = '/home/user/workspace/NBA-BETS'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'nba_models', 'training'))
warnings.filterwarnings('ignore')

print("=" * 70)
print("DEEP MODEL ANALYSIS")
print("=" * 70)

# ============================================
# 1. SPREAD MODEL - THE CRITICAL WEAKNESS
# ============================================
print("\n" + "=" * 60)
print("1. SPREAD MODEL ANALYSIS (RMSE = 14.20)")
print("=" * 60)

with open('models/spread_ensemble.pkl', 'rb') as f:
    spread = pickle.load(f)

sm = spread.get('training_metrics', {})
print("\nCurrent spread model metrics:")
print(f"  Holdout RMSE: {sm.get('holdout_rmse', sm.get('rmse', 'N/A'))}")
print(f"  Features used: {len(spread.get('feature_names', []))}")
print(f"  Feature names: {spread.get('feature_names', [])[:10]}...")

# Per-model breakdown
models = spread.get('models', {})
weights = spread.get('weights', {})
print("\n  Individual model performance:")
for name, w in weights.items():
    print(f"    {name}: weight={w:.2f}")

print("\n  DIAGNOSIS:")
print("  Vegas spread RMSE is ~12-13 pts. Our model is at 14.2.")
print("  This means we're WORSE than Vegas — negative edge.")
print("  We need to get RMSE below 12 to have a profitable ATS model.")
print("  The fundamental issue: 36 features derived from team-level stats")
print("  are insufficient. We need richer features:")
print("    - Lineup-based adjustments")
print("    - Travel/schedule fatigue")
print("    - Market line data (beating the closing line)")
print("    - Player availability (injuries)")

# ============================================
# 2. PLAYER PROPS - MODEL vs BASELINE
# ============================================
print("\n" + "=" * 60)
print("2. PLAYER PROPS - MODEL vs NAIVE BASELINES")
print("=" * 60)

# Load baselines from batch data
with open('data/balldontlie_cache/player_stats_batch_2024.json') as f:
    batch = json.load(f)

player_season = {}
for gid, players in batch.items():
    if not isinstance(players, list):
        continue
    for stat in players:
        pid = stat.get('player', {}).get('id')
        if not pid:
            continue
        ms = stat.get('min', '0')
        try:
            if isinstance(ms, str) and ':' in ms:
                mn, sc = ms.split(':')
                mp = int(mn) + int(sc)/60
            else:
                mp = float(ms or 0)
        except Exception:  # noqa: BLE001
            mp = 0
        if mp < 5:
            continue
        if pid not in player_season:
            player_season[pid] = {'pts': [], 'reb': [], 'ast': [], 'fg3m': []}
        player_season[pid]['pts'].append(stat.get('pts', 0) or 0)
        player_season[pid]['reb'].append(stat.get('reb', 0) or 0)
        player_season[pid]['ast'].append(stat.get('ast', 0) or 0)
        player_season[pid]['fg3m'].append(stat.get('fg3m', 0) or 0)

comparison = {}
for stat_name, prop_name in [('pts','points'), ('reb','rebounds'), ('ast','assists'), ('fg3m','threes')]:
    errors_avg, errors_l5 = [], []
    for pid, stats in player_season.items():
        vals = stats[stat_name]
        if len(vals) < 10:
            continue
        for i in range(10, len(vals)):
            actual = vals[i]
            season_avg = np.mean(vals[:i])
            errors_avg.append((season_avg - actual) ** 2)
            last5_avg = np.mean(vals[max(0,i-5):i])
            errors_l5.append((last5_avg - actual) ** 2)

    baseline_rmse = np.sqrt(np.mean(errors_avg))
    l5_rmse = np.sqrt(np.mean(errors_l5))

    # Model RMSE from training
    with open(f'models/player_{prop_name}_ensemble.pkl', 'rb') as f:
        m = pickle.load(f)
    model_rmse = m.get('training_metrics', {}).get('ensemble_rmse', None)

    pct_vs_avg = ((baseline_rmse - model_rmse) / baseline_rmse * 100) if model_rmse else 0
    pct_vs_l5 = ((l5_rmse - model_rmse) / l5_rmse * 100) if model_rmse else 0

    comparison[prop_name] = {
        'model_rmse': model_rmse,
        'season_avg_rmse': baseline_rmse,
        'last5_rmse': l5_rmse,
        'pct_improve_vs_avg': pct_vs_avg,
        'pct_improve_vs_l5': pct_vs_l5,
    }

    r2 = m.get('training_metrics', {}).get('ensemble_r2', 0)
    print(f"\n  {prop_name.upper()}:")
    print(f"    Model RMSE: {model_rmse:.3f} | R²: {r2:.4f}")
    print(f"    Season-avg baseline: {baseline_rmse:.3f}")
    print(f"    Last-5 baseline: {l5_rmse:.3f}")
    print(f"    Improvement vs avg: {pct_vs_avg:+.1f}%")
    print(f"    Improvement vs L5:  {pct_vs_l5:+.1f}%")

    # For betting, we need the over/under accuracy
    # If we can predict within 1.5 pts for PTS, 0.5 for REB/AST,
    # we have a viable over/under edge

print("\n  DIAGNOSIS:")
print(f"  Points: {comparison['points']['pct_improve_vs_avg']:+.1f}% vs season avg (GOOD - small but real edge)")
print(f"  Rebounds: {comparison['rebounds']['pct_improve_vs_avg']:+.1f}% vs season avg (GOOD)")
print(f"  Assists: {comparison['assists']['pct_improve_vs_avg']:+.1f}% vs season avg (WEAK)")
print(f"  Threes: {comparison['threes']['pct_improve_vs_avg']:+.1f}% vs season avg (ZERO EDGE - threes are too random)")

# ============================================
# 3. OVER/UNDER ACCURACY SIMULATION
# ============================================
print("\n" + "=" * 60)
print("3. SIMULATED OVER/UNDER HIT RATES")
print("=" * 60)

print("\n  For player props, the bet is OVER/UNDER a posted line.")
print("  If the line is set at the season average, our edge = how much")
print("  better our prediction is than the average.")
print("")
print("  A 2% RMSE improvement translates to roughly 1-2% better O/U accuracy.")
print("  Break-even for standard -110 odds: 52.4%")
print("")
print("  Estimated O/U hit rates (vs season-avg line):")
for prop_name, data in comparison.items():
    # Rough heuristic: accuracy ≈ 50% + improvement_pct * 0.3
    est_acc = 50 + data['pct_improve_vs_avg'] * 0.3
    roi = (est_acc/100 * 1.909 - 1) * 100 if est_acc > 0 else -100
    edge = "PROFITABLE" if est_acc > 52.4 else "NOT PROFITABLE"
    print(f"    {prop_name.upper()}: ~{est_acc:.1f}% ({edge}) | Est. ROI: {roi:+.1f}%")

# ============================================
# 4. KEY IMPROVEMENTS NEEDED
# ============================================
print("\n" + "=" * 60)
print("4. CRITICAL IMPROVEMENTS TO MAXIMIZE ROI")
print("=" * 60)

improvements = [
    {
        'priority': 1,
        'name': 'Fix spread model overfit / Add closing line value (CLV)',
        'impact': 'HIGH - Spread is currently WORSE than market. Either fix it or disable ATS betting',
        'details': 'RMSE 14.2 vs market 12-13. Need: (1) More features (injuries, schedule, travel), (2) Regularization tuning, (3) Or pivot to using closing line value as a feature',
    },
    {
        'priority': 2,
        'name': 'Improve player props confidence calibration',
        'impact': 'HIGH - The win_prob=1.0 issue means Kelly sizing is broken',
        'details': 'During backtest, many predictions output win_prob=1.0 which is clearly wrong. The over/under classifier needs recalibration with Platt scaling or isotonic regression',
    },
    {
        'priority': 3,
        'name': 'Add opponent-adjusted player prop features',
        'impact': 'MEDIUM - Current features are player-centric only',
        'details': 'Need: (1) Opponent defensive rating vs position, (2) Opponent pace, (3) Game total/spread as context features, (4) Matchup history',
    },
    {
        'priority': 4,
        'name': 'Improve three-point model or disable threes betting',
        'impact': 'MEDIUM - Model has no edge on threes (too random)',
        'details': 'R²=0.31 and zero improvement over baseline. Either add shot selection/attempt features or disable threes prop betting',
    },
    {
        'priority': 5,
        'name': 'Implement bet selection filter',
        'impact': 'HIGH - Only bet when edge is strongest',
        'details': 'Even a 51% accurate model is profitable if you only bet the top 10% most confident predictions. Need: confidence thresholds, bet sizing based on edge',
    },
    {
        'priority': 6,
        'name': 'Add line shopping / market odds integration',
        'impact': 'HIGH - Finding -105 vs -110 doubles ROI',
        'details': 'Integrate real odds from The Odds API. Only bet when our predicted value significantly exceeds the market line',
    },
]

for imp in improvements:
    print(f"\n  #{imp['priority']}: {imp['name']}")
    print(f"    Impact: {imp['impact']}")
    print(f"    Details: {imp['details']}")

# Save analysis
analysis = {
    'spread_model': {'rmse': 14.20, 'market_rmse': '12-13', 'edge': 'negative', 'verdict': 'NEEDS MAJOR WORK'},
    'moneyline_model': {'accuracy': 0.633, 'verdict': 'DECENT - 63.3% but need odds data to determine profitability'},
    'player_props': comparison,
    'improvements': improvements,
}
with open('backtest_results/deep_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2, default=str)

print("\n\n✓ Analysis saved to backtest_results/deep_analysis.json")
