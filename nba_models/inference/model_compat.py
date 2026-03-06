from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def prepare_loaded_model_artifact(data: Any) -> Any:
    """Normalize loaded pickle artifacts for inference.

    Historical artifacts in this repo use several incompatible layouts:
    - raw model instances
    - dict wrappers with {'model', 'feature_names', 'scaler'}
    - saved stacking dicts with {'base_models', 'meta_model', ...}

    This helper preserves legacy stacking dicts, but unwraps simple wrappers while
    copying feature metadata back onto the inner model so callers can treat it as a
    normal estimator.
    """
    if not isinstance(data, Mapping):
        return data

    if _is_saved_stacking_dict(data):
        return data

    model = data.get("model")
    if model is None:
        return data

    for attr in ("feature_names", "context_feature_names", "context_scaler"):
        if attr not in data:
            continue
        current = getattr(model, attr, None)
        if current in (None, [], ()):
            setattr(model, attr, data[attr])

    return model


def get_feature_names(model: Any) -> list[str]:
    """Extract feature names from a wrapped artifact or raw model."""
    if isinstance(model, Mapping):
        names = model.get("feature_names") or []
        return list(names)
    names = getattr(model, "feature_names", None) or []
    return list(names)


def get_context_feature_names(model: Any) -> list[str]:
    """Extract context feature names from a wrapped artifact or raw model."""
    if isinstance(model, Mapping):
        names = model.get("context_feature_names") or []
        return list(names)
    names = getattr(model, "context_feature_names", None) or []
    return list(names)


def predict_binary_probability(model: Any, features: dict[str, float | int]) -> float | None:
    """Predict positive-class probability from heterogeneous model artifacts."""
    return _predict_value(model, features, task="classification")


def predict_regression_value(model: Any, features: dict[str, float | int]) -> float | None:
    """Predict scalar regression output from heterogeneous model artifacts."""
    return _predict_value(model, features, task="regression")


def _predict_value(
    model: Any,
    features: dict[str, float | int],
    task: str,
) -> float | None:
    if model is None:
        return None

    if isinstance(model, Mapping):
        if _is_saved_stacking_dict(model):
            return _predict_saved_stacking_dict(model, features, task)
        wrapped_model = model.get("model")
        if wrapped_model is None:
            return None
        return _predict_wrapped_model(model, wrapped_model, features, task)

    if hasattr(model, "model_weights") and hasattr(model, "model_name"):
        result = model.predict(features)
        if task == "classification":
            return float(result.get("home_win_probability", 0.5))
        for key in ("predicted_spread", "predicted_value", "prediction"):
            if key in result:
                return float(result[key])
        return None

    feature_names = get_feature_names(model)
    if not feature_names:
        return None

    feature_frame = _build_feature_frame(features, feature_names)
    context_features = _build_context_array(model, features)

    if context_features is not None and _supports_context_features(model):
        try:
            raw = model.predict(feature_frame.values, context_features=context_features)
        except Exception:
            raw = model.predict(feature_frame, context_features=context_features)
        value = float(np.asarray(raw).reshape(-1)[0])
        if task == "classification":
            return float(np.clip(value, 0.0, 1.0))
        return value

    return _predict_estimator(model, feature_frame, task)


def _predict_wrapped_model(
    artifact: Mapping[str, Any],
    model: Any,
    features: dict[str, float | int],
    task: str,
) -> float | None:
    if hasattr(model, "model_weights") and hasattr(model, "model_name"):
        result = model.predict(features)
        if task == "classification":
            return float(result.get("home_win_probability", 0.5))
        for key in ("predicted_spread", "predicted_value", "prediction"):
            if key in result:
                return float(result[key])
        return None

    feature_names = list(artifact.get("feature_names") or get_feature_names(model))
    if not feature_names:
        return None

    feature_frame = _build_feature_frame(features, feature_names)
    model_input: Any = feature_frame
    scaler = artifact.get("scaler")
    if scaler is not None:
        model_input = scaler.transform(feature_frame)

    return _predict_estimator(model, model_input, task)


def _predict_saved_stacking_dict(
    artifact: Mapping[str, Any],
    features: dict[str, float | int],
    task: str,
) -> float | None:
    feature_names = list(artifact.get("feature_names") or [])
    if not feature_names:
        return None

    feature_frame = _build_feature_frame(features, feature_names)
    base_input: Any = feature_frame
    scaler = artifact.get("scaler")
    if scaler is not None:
        base_input = scaler.transform(feature_frame)

    base_models = artifact.get("base_models")
    if isinstance(base_models, Mapping):
        base_model_iter = base_models.values()
    else:
        base_model_iter = base_models or []

    base_predictions: list[float] = []
    for base_model in base_model_iter:
        pred = _predict_estimator(base_model, base_input, task)
        if pred is None:
            return None
        base_predictions.append(pred)

    if not base_predictions:
        return None

    meta_features = np.asarray(base_predictions, dtype=float).reshape(1, -1)

    context_names = list(artifact.get("context_feature_names") or [])
    if context_names:
        context_raw = np.asarray(
            [[float(features.get(name, 0.0)) for name in context_names]],
            dtype=float,
        )
        context_scaler = artifact.get("context_scaler")
        if context_scaler is not None:
            context_raw = context_scaler.transform(context_raw)
        meta_features = np.hstack([meta_features, context_raw])

    meta_model = artifact.get("meta_model")
    if meta_model is None:
        return None

    return _predict_estimator(meta_model, meta_features, task)


def _predict_estimator(model: Any, model_input: Any, task: str) -> float | None:
    if task == "classification":
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(model_input))
            if probs.ndim == 2:
                row = probs[0]
                return float(row[1] if len(row) > 1 else row[0])
            return float(probs.reshape(-1)[0])

        raw = np.asarray(model.predict(model_input)).reshape(-1)
        if raw.size == 0:
            return None
        value = float(raw[0])
        if 0.0 <= value <= 1.0:
            return value
        return float(1.0 / (1.0 + np.exp(-value)))

    raw = np.asarray(model.predict(model_input)).reshape(-1)
    if raw.size == 0:
        return None
    return float(raw[0])


def _build_feature_frame(
    features: Mapping[str, float | int],
    feature_names: list[str],
) -> pd.DataFrame:
    row = {name: features.get(name, 0.0) for name in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def _build_context_array(
    model: Any,
    features: Mapping[str, float | int],
) -> np.ndarray | None:
    context_names = getattr(model, "context_feature_names", None)
    if not context_names:
        return None

    aliases = {
        "travel_distance_away": "away_travel_distance",
        "back_to_back_away": "away_is_b2b",
        "days_rest_diff": "rest_days_diff",
        "pace_combined": "avg_pace",
        "home_advantage": "home_advantage_factor",
    }

    values: list[float] = []
    for name in context_names:
        candidates = [name]
        if name.startswith("ctx_"):
            stripped = name[4:]
            candidates.append(stripped)
            if stripped in aliases:
                candidates.append(aliases[stripped])
        elif name in aliases:
            candidates.append(aliases[name])

        value = 0.0
        for candidate in candidates:
            if candidate in features:
                value = float(features.get(candidate, 0.0))
                break
        values.append(value)

    return np.asarray(
        [values],
        dtype=float,
    )


def _supports_context_features(model: Any) -> bool:
    try:
        signature = inspect.signature(model.predict)
    except (TypeError, ValueError):
        return False
    return "context_features" in signature.parameters


def _is_saved_stacking_dict(data: Mapping[str, Any]) -> bool:
    required = {"base_models", "meta_model", "feature_names"}
    return required.issubset(data.keys())
