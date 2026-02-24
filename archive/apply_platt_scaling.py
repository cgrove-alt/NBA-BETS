"""
Apply Platt Scaling Calibration to Confidence Scores
====================================================

Solution 3 (Phase 2.5): Improve confidence correlation using Platt scaling

Platt scaling fits a logistic regression to map raw confidence scores
to calibrated probabilities that better correlate with actual accuracy.

Process:
1. Load backtest results with confidence scores and actual outcomes
2. Compute binary accuracy (correct within threshold)
3. Fit logistic regression: P(correct) = 1/(1 + exp(A*confidence + B))
4. Apply calibration to all confidence scores
5. Validate improvement in confidence correlation

Target: Confidence correlation r > 0.5 (current: 0.1019)
"""

import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.stats import pearsonr


def load_backtest_results():
    """Load Phase 2 backtest results."""
    results_file = Path('backtest_results/phase2_backtest.json')

    if not results_file.exists():
        raise FileNotFoundError(f"Backtest results not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)


def compute_binary_accuracy(predictions, threshold=0.8):
    """
    Compute binary accuracy: was prediction within threshold of actual?

    For each prediction, compute:
    - correct = 1 if |predicted - actual| / actual < threshold
    - correct = 0 otherwise

    This creates a binary target for calibration.
    """
    binary_outcomes = []

    for pred in predictions:
        actual = pred['actual']
        predicted = pred['predicted']

        if actual == 0:
            # DNP case - check if predicted was close to 0
            correct = 1 if predicted < 2 else 0
        else:
            # Normal case - relative error
            relative_error = abs(predicted - actual) / actual
            correct = 1 if relative_error < threshold else 0

        binary_outcomes.append(correct)

    return np.array(binary_outcomes)


def fit_platt_scaling(confidence_scores, binary_outcomes):
    """
    Fit Platt scaling: logistic regression on confidence scores.

    Returns fitted LogisticRegression model.
    """
    X = np.array(confidence_scores).reshape(-1, 1)
    y = np.array(binary_outcomes)

    # Fit logistic regression
    platt_model = LogisticRegression()
    platt_model.fit(X, y)

    print("\nPlatt Scaling Parameters:")
    print(f"  Coefficient (A): {platt_model.coef_[0][0]:.4f}")
    print(f"  Intercept (B): {platt_model.intercept_[0]:.4f}")

    return platt_model


def fit_isotonic_regression(confidence_scores, binary_outcomes):
    """
    Fit isotonic regression (non-parametric calibration).

    More flexible than Platt scaling, doesn't assume sigmoid relationship.
    """
    X = np.array(confidence_scores)
    y = np.array(binary_outcomes)

    iso_model = IsotonicRegression(out_of_bounds='clip')
    iso_model.fit(X, y)

    return iso_model


def apply_calibration(confidence_scores, calibration_model):
    """Apply calibration model to confidence scores."""
    X = np.array(confidence_scores).reshape(-1, 1)

    if isinstance(calibration_model, LogisticRegression):
        # Platt scaling - predict probability
        calibrated = calibration_model.predict_proba(X)[:, 1]
    else:
        # Isotonic regression
        calibrated = calibration_model.predict(confidence_scores)

    # Scale back to 0-100 range
    return calibrated * 100



def evaluate_calibration(original_conf, calibrated_conf, binary_outcomes):
    """
    Evaluate calibration improvement.

    Compare Pearson correlation before/after calibration.
    """
    # Original correlation
    r_original, p_original = pearsonr(original_conf, binary_outcomes)

    # Calibrated correlation
    r_calibrated, p_calibrated = pearsonr(calibrated_conf, binary_outcomes)

    print(f"\n{'='*70}")
    print("CALIBRATION EVALUATION")
    print(f"{'='*70}")
    print("\nOriginal Confidence:")
    print(f"  Correlation with accuracy: {r_original:.4f} (p={p_original:.4e})")

    print("\nCalibrated Confidence:")
    print(f"  Correlation with accuracy: {r_calibrated:.4f} (p={p_calibrated:.4e})")

    improvement = r_calibrated - r_original
    improvement_pct = 100 * improvement / abs(r_original) if r_original != 0 else float('inf')

    print("\nImprovement:")
    print(f"  Absolute: {improvement:+.4f}")
    print(f"  Relative: {improvement_pct:+.1f}%")

    target_met = '✅' if r_calibrated > 0.5 else '❌'
    print(f"\nTarget (r > 0.5): {target_met}")

    return {
        'original_correlation': float(r_original),
        'calibrated_correlation': float(r_calibrated),
        'improvement': float(improvement),
        'improvement_pct': float(improvement_pct),
        'target_met': r_calibrated > 0.5
    }


def main():
    print("="*70)
    print("APPLYING PLATT SCALING CALIBRATION")
    print("="*70)

    # Load backtest results
    print("\nLoading backtest results...")
    results = load_backtest_results()

    # Extract all predictions with confidence scores
    all_predictions = []
    confidence_scores = []

    for game in results['games']:
        for pred in game['predictions']:
            if 'confidence' in pred and pred['confidence'] is not None:
                all_predictions.append(pred)
                confidence_scores.append(pred['confidence'])

    print(f"  Found {len(all_predictions)} predictions with confidence scores")

    # Compute binary accuracy
    print("\nComputing binary accuracy (threshold=0.8)...")
    binary_outcomes = compute_binary_accuracy(all_predictions, threshold=0.8)
    accuracy_rate = 100 * np.mean(binary_outcomes)
    print(f"  Overall accuracy rate: {accuracy_rate:.2f}%")

    # Fit Platt scaling
    print("\nFitting Platt scaling (logistic regression)...")
    platt_model = fit_platt_scaling(confidence_scores, binary_outcomes)

    # Fit isotonic regression (alternative)
    print("\nFitting Isotonic regression (non-parametric)...")
    iso_model = fit_isotonic_regression(confidence_scores, binary_outcomes)

    # Apply both calibrations
    print("\nApplying calibrations...")
    platt_calibrated = apply_calibration(confidence_scores, platt_model)
    iso_calibrated = apply_calibration(confidence_scores, iso_model)

    # Evaluate Platt scaling
    print("\n" + "="*70)
    print("PLATT SCALING RESULTS")
    print("="*70)
    platt_results = evaluate_calibration(confidence_scores, platt_calibrated, binary_outcomes)

    # Evaluate isotonic regression
    print("\n" + "="*70)
    print("ISOTONIC REGRESSION RESULTS")
    print("="*70)
    iso_results = evaluate_calibration(confidence_scores, iso_calibrated, binary_outcomes)

    # Choose best method
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    if platt_results['calibrated_correlation'] > iso_results['calibrated_correlation']:
        best_method = 'Platt Scaling'
        best_model = platt_model
        best_correlation = platt_results['calibrated_correlation']
    else:
        best_method = 'Isotonic Regression'
        best_model = iso_model
        best_correlation = iso_results['calibrated_correlation']

    print(f"\nBest method: {best_method}")
    print(f"  Correlation: {best_correlation:.4f}")

    if best_correlation > 0.5:
        print("\n✅ TARGET ACHIEVED! Confidence correlation > 0.5")

        # Save calibration model
        model_file = Path('models/confidence_calibration.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump({
                'method': best_method,
                'model': best_model,
                'correlation': best_correlation
            }, f)
        print(f"\nSaved calibration model to: {model_file}")
    else:
        print(f"\n⚠️  Target not met (r={best_correlation:.4f} < 0.5)")
        print("   Calibration provides improvement but insufficient.")
        print("   Consider:")
        print("   1. Recalibrate with different accuracy threshold")
        print("   2. Apply calibration to tree-based ensemble (after Ridge removal)")
        print("   3. Alternative confidence metrics (quantile regression)")

    # Save results
    output = {
        'platt_scaling': platt_results,
        'isotonic_regression': iso_results,
        'best_method': best_method,
        'best_correlation': float(best_correlation),
        'target_met': best_correlation > 0.5
    }

    output_file = Path('backtest_results/calibration_results.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
