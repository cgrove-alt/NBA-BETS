"""
Analyze Base Model Agreement
============================

Investigates why confidence distribution is broken (0.2% Elite+Strong vs 10% target).

Key questions:
1. What is the typical CV (coefficient of variation) across base models?
2. Are base models highly disagreeing?
3. Which prop types have worst/best agreement?
4. Is Phase 2 causing instability?

This will help diagnose the root cause of confidence mechanism failure.
"""

import pickle
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Import from backtest script
from comprehensive_backtest import smart_fillna_prediction


def analyze_model_agreement_on_samples(n_samples=1000):
    """
    Analyze base model agreement on random feature samples.

    This tells us if the models inherently disagree or if specific
    situations cause disagreement.
    """
    print('='*70)
    print('ANALYZING BASE MODEL AGREEMENT')
    print('='*70)

    model_dir = Path('models')
    prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']

    results = {}

    for prop_type in prop_types:
        model_file = model_dir / f'player_{prop_type}_ensemble.pkl'

        if not model_file.exists():
            print(f'\n❌ {prop_type}: Model not found')
            continue

        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)

        base_models = model_data['models']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']

        print(f'\n{prop_type.upper()}:')
        print(f'  Base models: {list(base_models.keys())}')
        print(f'  Features: {len(feature_names)}')

        # Load some actual game data to get realistic feature distributions
        # For now, generate synthetic but realistic features
        cv_values = []
        predictions_by_model = defaultdict(list)

        # Generate n_samples random feature sets based on realistic ranges
        for i in range(min(n_samples, 200)):  # Limit to 200 for speed
            # Create realistic feature values
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
                'days_rest': np.random.choice([0, 1, 2, 3, 4]),
                'pace': np.random.uniform(95, 105),
                'off_rating': np.random.uniform(105, 120),
                'def_rating': np.random.uniform(105, 120),
            }

            # Add any missing features with defaults
            X = pd.DataFrame([features])
            for col in feature_names:
                if col not in X.columns:
                    X[col] = 0

            X = smart_fillna_prediction(X[feature_names])
            X_scaled = scaler.transform(X)

            # Get predictions from each base model
            base_preds = []
            for model_name, model in base_models.items():
                pred = model.predict(X_scaled)[0]
                base_preds.append(pred)
                predictions_by_model[model_name].append(pred)

            # Calculate CV for this sample
            std_dev = np.std(base_preds)
            mean_pred = np.mean(base_preds)
            cv = std_dev / max(abs(mean_pred), 0.01)
            cv_values.append(cv)

        # Analyze CV distribution
        cv_array = np.array(cv_values)

        print(f'\n  Coefficient of Variation (CV) Analysis:')
        print(f'    Mean CV: {np.mean(cv_array):.4f}')
        print(f'    Median CV: {np.median(cv_array):.4f}')
        print(f'    Std CV: {np.std(cv_array):.4f}')
        print(f'    Min CV: {np.min(cv_array):.4f}')
        print(f'    Max CV: {np.max(cv_array):.4f}')

        # Map to confidence tiers
        elite_count = sum(1 for cv in cv_array if cv < 0.05)
        strong_count = sum(1 for cv in cv_array if 0.05 <= cv < 0.10)
        moderate_count = sum(1 for cv in cv_array if 0.10 <= cv < 0.20)
        weak_count = sum(1 for cv in cv_array if 0.20 <= cv < 0.30)
        avoid_count = sum(1 for cv in cv_array if cv >= 0.30)

        print(f'\n  Expected Confidence Distribution (based on CV thresholds):')
        print(f'    Elite (CV <0.05):     {elite_count:3d} ({100*elite_count/len(cv_array):5.1f}%)')
        print(f'    Strong (0.05-0.10):   {strong_count:3d} ({100*strong_count/len(cv_array):5.1f}%)')
        print(f'    Moderate (0.10-0.20): {moderate_count:3d} ({100*moderate_count/len(cv_array):5.1f}%)')
        print(f'    Weak (0.20-0.30):     {weak_count:3d} ({100*weak_count/len(cv_array):5.1f}%)')
        print(f'    Avoid (CV >=0.30):    {avoid_count:3d} ({100*avoid_count/len(cv_array):5.1f}%)')

        elite_strong_pct = 100 * (elite_count + strong_count) / len(cv_array)
        print(f'\n  Elite + Strong: {elite_strong_pct:.2f}% (target: ≥10%)')

        # Analyze inter-model correlation
        print(f'\n  Inter-Model Correlations:')
        model_names = list(predictions_by_model.keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                corr = np.corrcoef(predictions_by_model[model1], predictions_by_model[model2])[0, 1]
                print(f'    {model1:15s} vs {model2:15s}: {corr:.4f}')

        results[prop_type] = {
            'mean_cv': float(np.mean(cv_array)),
            'median_cv': float(np.median(cv_array)),
            'elite_strong_pct': float(elite_strong_pct),
            'cv_percentiles': {
                '25th': float(np.percentile(cv_array, 25)),
                '50th': float(np.percentile(cv_array, 50)),
                '75th': float(np.percentile(cv_array, 75)),
                '90th': float(np.percentile(cv_array, 90)),
            }
        }

    # Overall summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)

    for prop_type, stats in results.items():
        status = '✅' if stats['elite_strong_pct'] >= 10 else '❌'
        print(f'\n{prop_type:8s}: Mean CV={stats[\"mean_cv\"]:.4f}  Elite+Strong={stats[\"elite_strong_pct\"]:5.2f}% {status}')

    # Save results
    output_file = Path('backtest_results/base_model_agreement.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n\nResults saved to: {output_file}')

    # Diagnosis
    print('\n' + '='*70)
    print('DIAGNOSIS')
    print('='*70)

    overall_elite_strong = np.mean([s['elite_strong_pct'] for s in results.values()])
    print(f'\nOverall Elite+Strong percentage: {overall_elite_strong:.2f}%')

    if overall_elite_strong < 5:
        print('\n❌ CRITICAL ISSUE: Base models have EXTREMELY high disagreement')
        print('   Root causes:')
        print('   - Models may be poorly calibrated')
        print('   - Training data may be too noisy')
        print('   - Phase 2 features may be causing instability')
        print('   - Model architectures may be too different')
        print('\n   Recommendations:')
        print('   1. Recalibrate confidence thresholds (loosen CV requirements)')
        print('   2. Run feature ablation to identify harmful features')
        print('   3. Consider ensemble methods that reduce disagreement')
        print('   4. Try isotonic regression or Platt scaling for calibration')
    elif overall_elite_strong < 10:
        print('\n⚠️  WARNING: Elite+Strong below target')
        print('   Consider recalibrating thresholds or improving model agreement')
    else:
        print('\n✅ Base model agreement is adequate')

    return results


if __name__ == '__main__':
    results = analyze_model_agreement_on_samples(n_samples=200)
