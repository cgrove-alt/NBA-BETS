"""
Auto-calibrate quantile decompression constants after model retraining.

This script should be run after every model retrain to update the
QUANTILE_DECOMPRESSION constants used in daily_predictions.py.

It computes the empirical relationship between Q10/Q50/Q90 predictions
and actual over/under outcomes on held-out calibration data.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from scipy import stats


def calibrate_decompression_for_prop(
    model_path: str,
    calibration_data: list[dict],
    prop_type: str,
) -> dict:
    """
    Compute optimal slope and mean_gap for a quantile model.

    The decompression converts (Q90 - Q10) spread into a probability
    via: prob = norm.cdf((predicted - line) / (adjusted_std))
    where adjusted_std = (Q90 - Q10) * slope + mean_gap

    Args:
        model_path: Path to the quantile model pickle
        calibration_data: List of {features, actual_value, line} dicts
        prop_type: Type of prop (points, rebounds, etc.)

    Returns:
        Dict with 'slope' and 'mean_gap' values
    """
    if not calibration_data:
        return {'slope': 0.7, 'mean_gap': -2.0}  # Safe defaults

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Generate predictions on calibration data
    spreads = []  # Q90 - Q10
    residuals = []  # actual - predicted_median

    for sample in calibration_data:
        try:
            features = sample['features']
            actual = sample['actual_value']

            if hasattr(model, 'predict_distribution'):
                dist = model.predict_distribution(features)
                q10 = dist.get(0.10, 0)
                q50 = dist.get(0.50, 0)
                q90 = dist.get(0.90, 0)
            elif hasattr(model, 'quantile_models'):
                # Direct quantile model access
                import pandas as pd
                X = pd.DataFrame([features])
                for col in model.feature_names:
                    if col not in X.columns:
                        X[col] = 0
                X = X[model.feature_names].fillna(0)
                X_scaled = model.scaler.transform(X)

                q10 = float(model.quantile_models[0.10].predict(X_scaled)[0])
                q50 = float(model.quantile_models[0.50].predict(X_scaled)[0])
                q90 = float(model.quantile_models[0.90].predict(X_scaled)[0])
            else:
                continue

            spread = q90 - q10
            if spread > 0.1:  # Only use meaningful spreads
                spreads.append(spread)
                residuals.append(actual - q50)
        except Exception:
            continue

    if len(spreads) < 50:
        return {'slope': 0.7, 'mean_gap': -2.0}

    spreads = np.array(spreads)
    residuals = np.array(residuals)

    # Fit: actual_std ≈ slope * spread + mean_gap
    actual_std = np.std(residuals)
    mean_spread = np.mean(spreads)

    if mean_spread > 0:
        slope = actual_std / mean_spread
    else:
        slope = 0.7

    mean_gap = np.mean(residuals)  # Systematic bias

    return {
        'slope': round(float(slope), 4),
        'mean_gap': round(float(mean_gap), 4),
        'calibration_samples': len(spreads),
        'actual_std': round(float(actual_std), 4),
        'mean_spread': round(float(mean_spread), 4),
    }


def run_full_calibration():
    """Run calibration for all prop types and save results."""
    output = {}
    prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']

    for prop_type in prop_types:
        model_path = Path(f"models/player_{prop_type}_quantile.pkl")
        if not model_path.exists():
            print(f"  Skipping {prop_type} — no quantile model found")
            output[prop_type] = {'slope': 0.7, 'mean_gap': -2.0}
            continue

        # TODO: Load actual calibration data from backtest results
        # For now, use defaults
        print(f"  Calibrating {prop_type}...")
        output[prop_type] = {'slope': 0.7, 'mean_gap': -2.0}

    # Save to model directory
    output_path = Path("models/quantile_decompression.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved quantile decompression constants to {output_path}")
    return output


if __name__ == "__main__":
    run_full_calibration()
