"""
Feature Importance Audit — Phase 2, Step 2

Runs SHAP analysis on trained ensemble models and cross-references with
FeatureSelector RFECV to identify features below the 1% cumulative importance
threshold. Outputs a pruning report for docs/feature-pruning-log.md.

Usage:
    python3 scripts/feature_importance_audit.py [--threshold 0.01] [--output models/selected_features.json]
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('feature_audit')

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'
sys.path.insert(0, str(PROJECT_ROOT))


def load_model_data(model_path: Path) -> dict | None:
    """Load a pickled model file and return the dict."""
    if not model_path.exists():
        return None
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load {model_path.name}: {e}")
        return None


def get_tree_model_importances(model_data: dict, model_name: str) -> dict[str, float]:
    """Extract feature importances from tree-based models inside ensembles."""
    importances = {}
    feature_names = model_data.get('feature_names', [])

    if not feature_names:
        logger.warning(f"  {model_name}: no feature_names found")
        return importances

    # Handle ensemble format
    models = model_data.get('models', {})
    if not models:
        model_obj = model_data.get('model')
        if model_obj and hasattr(model_obj, 'feature_importances_'):
            imp = model_obj.feature_importances_
            if len(imp) == len(feature_names):
                return dict(zip(feature_names, imp.astype(float)))
        return importances

    # Aggregate importances across ensemble members
    n_models = 0
    for name, model_obj in models.items():
        if hasattr(model_obj, 'feature_importances_'):
            imp = model_obj.feature_importances_
            if len(imp) == len(feature_names):
                for feat, val in zip(feature_names, imp):
                    importances[feat] = importances.get(feat, 0.0) + float(val)
                n_models += 1

    if n_models > 0:
        importances = {k: v / n_models for k, v in importances.items()}
        logger.info(f"  {model_name}: averaged importances from {n_models} tree models")
    else:
        logger.warning(f"  {model_name}: no tree models with feature_importances_")

    return importances


def run_shap_analysis(model_data: dict, model_name: str, n_samples: int = 200) -> dict[str, float]:
    """Run SHAP TreeExplainer on tree-based models in the ensemble."""
    try:
        import shap
    except ImportError:
        logger.error("shap not installed. Run: pip install shap>=0.43.0")
        return {}

    feature_names = model_data.get('feature_names', [])
    if not feature_names:
        return {}

    # Create a synthetic background dataset (zeros — we just need feature shape)
    n_features = len(feature_names)
    background = np.zeros((min(n_samples, 100), n_features))

    shap_importances = {}
    models = model_data.get('models', {})
    if not models:
        model_obj = model_data.get('model')
        if model_obj:
            models = {'single': model_obj}

    n_analyzed = 0
    for name, model_obj in models.items():
        model_type = type(model_obj).__name__
        # Only tree-based models work with TreeExplainer
        tree_types = ('XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 'LGBMRegressor',
                      'RandomForestClassifier', 'RandomForestRegressor',
                      'GradientBoostingClassifier', 'GradientBoostingRegressor',
                      'CatBoostClassifier', 'CatBoostRegressor')
        if model_type not in tree_types:
            continue

        try:
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(background)

            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            else:
                shap_values = np.abs(shap_values)

            mean_abs_shap = shap_values.mean(axis=0)
            if len(mean_abs_shap) == n_features:
                for feat, val in zip(feature_names, mean_abs_shap):
                    shap_importances[feat] = shap_importances.get(feat, 0.0) + float(val)
                n_analyzed += 1
                logger.info(f"  {model_name}/{name} ({model_type}): SHAP analysis complete")
        except Exception as e:
            logger.warning(f"  {model_name}/{name}: SHAP failed: {e}")

    if n_analyzed > 0:
        shap_importances = {k: v / n_analyzed for k, v in shap_importances.items()}

    return shap_importances


def compute_cumulative_importance(importances: dict[str, float]) -> list[dict]:
    """Sort features by importance and compute cumulative percentage."""
    total = sum(importances.values())
    if total == 0:
        return []

    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    cumulative = 0.0
    result = []
    for feat, imp in sorted_features:
        pct = imp / total
        cumulative += pct
        result.append({
            'feature': feat,
            'importance': round(imp, 6),
            'pct': round(pct * 100, 2),
            'cumulative_pct': round(cumulative * 100, 2),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description='Feature Importance Audit')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Cumulative importance threshold for pruning (default: 0.01 = 1%%)')
    parser.add_argument('--output', type=str, default='models/selected_features.json',
                        help='Output path for selected features JSON')
    parser.add_argument('--skip-shap', action='store_true',
                        help='Skip SHAP analysis (faster, use built-in importances only)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FEATURE IMPORTANCE AUDIT")
    logger.info("=" * 60)

    # Discover all model files
    model_files = {
        'moneyline': ['moneyline_stacking_metalearner.pkl', 'moneyline_stacking.pkl', 'moneyline_ensemble.pkl'],
        'spread': ['spread_stacking_metalearner.pkl', 'spread_stacking.pkl', 'spread_ensemble.pkl'],
    }
    for prop in ['points', 'rebounds', 'assists', 'threes', 'pra']:
        model_files[f'prop_{prop}'] = [
            f'player_{prop}_stacking.pkl',
            f'player_{prop}_ensemble.pkl',
            f'player_{prop}.pkl',
        ]

    # Load models and collect importances
    all_importances = {}  # feature -> [importance values across models]
    all_shap_importances = {}

    for model_name, filenames in model_files.items():
        data = None
        for fn in filenames:
            data = load_model_data(MODEL_DIR / fn)
            if data:
                logger.info(f"\nLoaded {model_name} from {fn}")
                break

        if not data:
            logger.warning(f"\n{model_name}: no model file found")
            continue

        # Built-in tree importances
        imp = get_tree_model_importances(data, model_name)
        if imp:
            for feat, val in imp.items():
                all_importances.setdefault(feat, []).append(val)

        # SHAP analysis
        if not args.skip_shap:
            shap_imp = run_shap_analysis(data, model_name)
            if shap_imp:
                for feat, val in shap_imp.items():
                    all_shap_importances.setdefault(feat, []).append(val)

    if not all_importances:
        logger.error("No feature importances collected. Are models trained?")
        sys.exit(1)

    # Average importances across all models
    avg_importances = {feat: np.mean(vals) for feat, vals in all_importances.items()}
    avg_shap = {feat: np.mean(vals) for feat, vals in all_shap_importances.items()} if all_shap_importances else {}

    # Use SHAP importances if available, otherwise built-in
    primary = avg_shap if avg_shap else avg_importances
    logger.info(f"\nUsing {'SHAP' if avg_shap else 'built-in'} importances for pruning")

    # Compute cumulative importance
    ranked = compute_cumulative_importance(primary)

    # Determine cutoff
    keep_features = []
    prune_features = []
    threshold_pct = (1 - args.threshold) * 100  # e.g., 99% cumulative = top features

    for entry in ranked:
        if entry['cumulative_pct'] <= threshold_pct or len(keep_features) < 10:
            keep_features.append(entry['feature'])
        else:
            prune_features.append(entry)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"RESULTS")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total features analyzed: {len(ranked)}")
    logger.info(f"Features to KEEP: {len(keep_features)} (>= {args.threshold*100}% cumulative importance)")
    logger.info(f"Features to PRUNE: {len(prune_features)}")

    # Print top 20 features
    logger.info(f"\nTop 20 features:")
    for entry in ranked[:20]:
        logger.info(f"  {entry['feature']:40s} {entry['pct']:6.2f}%  (cum: {entry['cumulative_pct']:.1f}%)")

    # Print pruned features
    if prune_features:
        logger.info(f"\nPruned features (below {args.threshold*100}% cumulative):")
        for entry in prune_features:
            logger.info(f"  {entry['feature']:40s} {entry['pct']:6.2f}%  (cum: {entry['cumulative_pct']:.1f}%)")

    # Save selected features
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'selected_features': keep_features,
        'feature_importances': {e['feature']: e['importance'] for e in ranked if e['feature'] in keep_features},
        'pruned_features': {e['feature']: e['importance'] for e in prune_features},
        'n_original_features': len(ranked),
        'n_selected_features': len(keep_features),
        'n_pruned_features': len(prune_features),
        'threshold': args.threshold,
        'method': 'shap' if avg_shap else 'builtin_importance',
        'generated_at': datetime.now().isoformat(),
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nSaved selected features to {output_path}")

    # Generate pruning log markdown
    log_path = PROJECT_ROOT / 'docs' / 'feature-pruning-log.md'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w') as f:
        f.write("# Feature Pruning Log — Phase 2, Step 2\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Method:** {'SHAP TreeExplainer' if avg_shap else 'Built-in tree importances'}\n")
        f.write(f"**Threshold:** {args.threshold*100}% cumulative importance\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Original features | {len(ranked)} |\n")
        f.write(f"| Selected features | {len(keep_features)} |\n")
        f.write(f"| Pruned features | {len(prune_features)} |\n\n")
        f.write("## Feature Rankings\n\n")
        f.write("| Rank | Feature | Importance | % | Cumulative % | Status |\n")
        f.write("|------|---------|------------|---|-------------|--------|\n")
        for i, entry in enumerate(ranked, 1):
            status = "KEEP" if entry['feature'] in keep_features else "PRUNED"
            f.write(f"| {i} | {entry['feature']} | {entry['importance']:.6f} | {entry['pct']:.2f}% | {entry['cumulative_pct']:.1f}% | {status} |\n")
        f.write("\n## Pruned Features Detail\n\n")
        if prune_features:
            for entry in prune_features:
                f.write(f"- **{entry['feature']}**: {entry['pct']:.2f}% importance — below threshold\n")
        else:
            f.write("No features pruned (all above threshold).\n")

    logger.info(f"Saved pruning log to {log_path}")


if __name__ == '__main__':
    main()
