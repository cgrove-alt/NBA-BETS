"""
Feature Ablation Study
======================

Tests if Phase 2 features are causing the extreme base model disagreement (CV 0.3-1.4).

Phase 2 Feature Groups:
1. Travel/Fatigue (10 features): miles_traveled, time_zones, back_to_back, etc.
2. Betting Markets (6 features): implied_totals, market_efficiency, etc.
3. Enhanced Injuries (4 features): star_player_out, usage_redistribution, etc.

Test Plan:
- Baseline: All Phase 2 features
- Test 1: Remove travel/fatigue features
- Test 2: Remove betting market features
- Test 3: Remove injury features
- Test 4: Remove ALL Phase 2 features (Phase 1 baseline)

For each test, measure:
- Mean CV across prop types
- Elite+Strong percentage
- Overall RMSE

This will identify which features harm model agreement.
"""

import pickle
import numpy as np
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict

from comprehensive_backtest import smart_fillna_prediction


# Phase 2 feature groups
PHASE2_FEATURE_GROUPS = {
    'travel_fatigue': [
        'miles_traveled_last_7d',
        'time_zones_crossed',
        'is_back_to_back',
        'rest_days_last_week',
        'games_in_last_7d',
        'avg_travel_distance',
        'consecutive_road_games',
        'home_stand_length',
        'fatigue_index',
        'schedule_density'
    ],
    'betting_markets': [
        'implied_team_total',
        'implied_opponent_total',
        'market_spread',
        'total_line',
        'market_efficiency_score',
        'sharp_money_indicator'
    ],
    'enhanced_injuries': [
        'star_player_out',
        'teammate_injury_impact',
        'usage_rate_adjustment',
        'minutes_redistribution'
    ]
}


def test_feature_configuration(prop_type, feature_config_name, features_to_exclude):
    """
    Test a specific feature configuration by excluding certain features.

    Returns CV statistics for this configuration.
    """
    model_dir = Path('models')
    model_file = model_dir / f'player_{prop_type}_ensemble.pkl'

    if not model_file.exists():
        return None

    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)

    base_models = model_data['models']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']

    # Filter out excluded features
    available_features = [f for f in feature_names if f not in features_to_exclude]

    # Generate test samples
    cv_values = []
    n_samples = 100

    for _i in range(n_samples):
        # Generate realistic feature values
        features = {
            'season_pts_avg': np.random.uniform(5, 30),
            'recent_pts_avg': np.random.uniform(5, 30),
            'season_reb_avg': np.random.uniform(2, 12),
            'recent_reb_avg': np.random.uniform(2, 12),
            'season_ast_avg': np.random.uniform(1, 10),
            'recent_ast_avg': np.random.uniform(1, 10),
            'season_fg3m_avg': np.random.uniform(0, 4),
            'recent_fg3m_avg': np.random.uniform(0, 4),
            'season_min_avg': np.random.uniform(15, 36),
            'recent_min_avg': np.random.uniform(15, 36),
            'is_home': np.random.choice([0, 1]),
            'days_rest': np.random.choice([0, 1, 2, 3]),
            'pace': np.random.uniform(95, 105),
            'off_rating': np.random.uniform(105, 120),
            'def_rating': np.random.uniform(105, 120),
        }

        # Add Phase 2 features (will be zeroed if excluded)
        for group_features in PHASE2_FEATURE_GROUPS.values():
            for feat in group_features:
                if feat in features_to_exclude:
                    features[feat] = 0  # Zero out excluded features
                else:
                    features[feat] = np.random.uniform(0, 1)  # Placeholder

        # Build feature array
        X = pd.DataFrame([features])
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0

        # Use only available features
        X[available_features].copy()

        # For features not in available list, set to 0 (effectively removing them)
        for col in feature_names:
            if col not in available_features and col in X.columns:
                X[col] = 0

        X = smart_fillna_prediction(X[feature_names])
        X_scaled = scaler.transform(X)

        # Get predictions from each base model
        base_preds = []
        for model in base_models.values():
            pred = model.predict(X_scaled)[0]
            base_preds.append(pred)

        # Calculate CV
        std_dev = np.std(base_preds)
        mean_pred = np.mean(base_preds)
        cv = std_dev / max(abs(mean_pred), 0.01)
        cv_values.append(cv)

    cv_array = np.array(cv_values)

    # Calculate tier distribution
    elite = sum(1 for cv in cv_array if cv < 0.05)
    strong = sum(1 for cv in cv_array if 0.05 <= cv < 0.10)
    moderate = sum(1 for cv in cv_array if 0.10 <= cv < 0.20)
    weak = sum(1 for cv in cv_array if 0.20 <= cv < 0.30)
    avoid = sum(1 for cv in cv_array if cv >= 0.30)

    elite_strong_pct = 100 * (elite + strong) / len(cv_array)

    return {
        'config_name': feature_config_name,
        'features_excluded': len(features_to_exclude),
        'mean_cv': float(np.mean(cv_array)),
        'median_cv': float(np.median(cv_array)),
        'std_cv': float(np.std(cv_array)),
        'min_cv': float(np.min(cv_array)),
        'max_cv': float(np.max(cv_array)),
        'elite_pct': float(100 * elite / len(cv_array)),
        'strong_pct': float(100 * strong / len(cv_array)),
        'moderate_pct': float(100 * moderate / len(cv_array)),
        'weak_pct': float(100 * weak / len(cv_array)),
        'avoid_pct': float(100 * avoid / len(cv_array)),
        'elite_strong_pct': float(elite_strong_pct),
        'n_samples': n_samples
    }


def run_ablation_study():
    """Run complete feature ablation study."""
    print('='*70)
    print('FEATURE ABLATION STUDY')
    print('='*70)
    print('\nTesting if Phase 2 features cause extreme model disagreement...')

    prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']

    # Define test configurations
    configurations = [
        ('Baseline (All Phase 2)', []),
        ('Without Travel/Fatigue', PHASE2_FEATURE_GROUPS['travel_fatigue']),
        ('Without Betting Markets', PHASE2_FEATURE_GROUPS['betting_markets']),
        ('Without Enhanced Injuries', PHASE2_FEATURE_GROUPS['enhanced_injuries']),
        ('Phase 1 (No Phase 2)',
         PHASE2_FEATURE_GROUPS['travel_fatigue'] +
         PHASE2_FEATURE_GROUPS['betting_markets'] +
         PHASE2_FEATURE_GROUPS['enhanced_injuries'])
    ]

    results = defaultdict(dict)

    for config_name, excluded_features in configurations:
        print(f'\n{"="*70}')
        print(f'CONFIGURATION: {config_name}')
        print(f'{"="*70}')
        print(f'Excluding {len(excluded_features)} features')

        config_results = {}

        for prop_type in prop_types:
            print(f'\n  Testing {prop_type}...', end=' ')

            result = test_feature_configuration(prop_type, config_name, excluded_features)

            if result:
                config_results[prop_type] = result
                print(f'CV={result['mean_cv']:.4f}  E+S={result['elite_strong_pct']:.1f}%')
            else:
                print('SKIPPED')

        results[config_name] = config_results

    # Summary comparison
    print('\n' + '='*70)
    print('SUMMARY: MEAN CV BY CONFIGURATION')
    print('='*70)

    print(f'\n{"Configuration":30s}', end='')
    for prop in prop_types:
        print(f' {prop:8s}', end='')
    print('  Overall')
    print('-'*70)

    for config_name, config_data in results.items():
        print(f'{config_name:30s}', end='')
        cvs = []
        for prop in prop_types:
            if prop in config_data:
                cv = config_data[prop]['mean_cv']
                cvs.append(cv)
                print(f' {cv:8.4f}', end='')
            else:
                print(f' {"N/A":8s}', end='')

        if cvs:
            overall_cv = np.mean(cvs)
            print(f'  {overall_cv:.4f}')
        else:
            print(f'  {"N/A"}')

    # Elite+Strong comparison
    print('\n' + '='*70)
    print('SUMMARY: ELITE+STRONG % BY CONFIGURATION')
    print('='*70)

    print(f'\n{"Configuration":30s}', end='')
    for prop in prop_types:
        print(f' {prop:8s}', end='')
    print('  Overall  Status')
    print('-'*70)

    for config_name, config_data in results.items():
        print(f'{config_name:30s}', end='')
        es_pcts = []
        for prop in prop_types:
            if prop in config_data:
                es = config_data[prop]['elite_strong_pct']
                es_pcts.append(es)
                print(f' {es:7.1f}%', end='')
            else:
                print(f' {"N/A":8s}', end='')

        if es_pcts:
            overall_es = np.mean(es_pcts)
            status = '✅' if overall_es >= 10 else '❌'
            print(f'  {overall_es:6.1f}%  {status}')
        else:
            print(f'  {"N/A"}')

    # Identify best configuration
    print('\n' + '='*70)
    print('ANALYSIS')
    print('='*70)

    # Calculate improvement for each configuration vs baseline
    baseline_name = 'Baseline (All Phase 2)'
    if baseline_name in results:
        baseline = results[baseline_name]
        baseline_cv = np.mean([baseline[p]['mean_cv'] for p in prop_types if p in baseline])
        baseline_es = np.mean([baseline[p]['elite_strong_pct'] for p in prop_types if p in baseline])

        print('\nBaseline:')
        print(f'  Mean CV: {baseline_cv:.4f}')
        print(f'  Elite+Strong: {baseline_es:.1f}%')

        improvements = []

        for config_name, config_data in results.items():
            if config_name == baseline_name:
                continue

            if not config_data:
                continue

            config_cv = np.mean([config_data[p]['mean_cv'] for p in prop_types if p in config_data])
            config_es = np.mean([config_data[p]['elite_strong_pct'] for p in prop_types if p in config_data])

            cv_improvement = baseline_cv - config_cv
            es_improvement = config_es - baseline_es

            improvements.append({
                'config': config_name,
                'cv_improvement': cv_improvement,
                'cv_improvement_pct': 100 * cv_improvement / baseline_cv if baseline_cv > 0 else 0,
                'es_improvement': es_improvement,
                'es_improvement_pct': 100 * es_improvement / baseline_es if baseline_es > 0 else 0,
                'final_cv': config_cv,
                'final_es': config_es
            })

        # Sort by ES improvement
        improvements.sort(key=lambda x: x['es_improvement'], reverse=True)

        print('\nImprovements vs Baseline:')
        print(f'{"Configuration":30s} {"CV Change":>12s} {"E+S Change":>12s} {"Final E+S":>12s}')
        print('-'*70)

        for imp in improvements:
            cv_arrow = '↓' if imp['cv_improvement'] > 0 else '↑'
            es_arrow = '↑' if imp['es_improvement'] > 0 else '↓'

            print(f'{imp['config']:30s} '
                  f'{cv_arrow}{abs(imp['cv_improvement']):.4f} ({imp['cv_improvement_pct']:+.1f}%)  '
                  f'{es_arrow}{abs(imp['es_improvement']):.1f}% ({imp['es_improvement_pct']:+.1f}%)  '
                  f'{imp['final_es']:.1f}%')

    # Save results
    output_file = Path('backtest_results/feature_ablation_results.json')
    with open(output_file, 'w') as f:
        json.dump(dict(results), f, indent=2)

    print(f'\n\nResults saved to: {output_file}')

    # Recommendations
    print('\n' + '='*70)
    print('RECOMMENDATIONS')
    print('='*70)

    if improvements:
        best = improvements[0]
        print(f'\nBest configuration: {best['config']}')
        print(f'  Elite+Strong: {best['final_es']:.1f}% ({best['es_improvement']:+.1f}% vs baseline)')
        print(f'  Mean CV: {best['final_cv']:.4f} ({best['cv_improvement']:+.4f} vs baseline)')

        if best['final_es'] >= 10:
            print(f'\n✅ TARGET ACHIEVED! Remove features from: {best['config']}')
        elif best['es_improvement'] > 0:
            print('\n⚠️  Improvement shown but target not met.')
            print('   Consider: Remove features + recalibrate thresholds')
        else:
            print('\n❌ No configuration improves Elite+Strong percentage.')
            print('   Phase 2 features are NOT the primary cause.')
            print('   Next steps:')
            print('   1. Recalibrate confidence thresholds')
            print('   2. Consider removing Ridge model')
            print('   3. Apply calibration methods')

    return results


if __name__ == '__main__':
    results = run_ablation_study()
