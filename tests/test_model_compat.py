import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba_models.inference.model_compat import (
    predict_binary_probability,
    predict_regression_value,
    prepare_loaded_model_artifact,
)


class RecordingBinaryModel:
    def __init__(self):
        self.feature_names = []
        self.last_columns = None
        self.last_row = None

    def predict_proba(self, X):
        self.last_columns = list(X.columns)
        self.last_row = X.iloc[0].to_dict()
        return np.array([[0.2, 0.8]])


class FixedBaseClassifier:
    def __init__(self, positive_prob):
        self.positive_prob = positive_prob

    def predict_proba(self, X):
        assert list(X.columns) == ["feat_b", "feat_a"]
        return np.array([[1 - self.positive_prob, self.positive_prob]])


class RecordingMetaClassifier:
    def __init__(self):
        self.last_input = None

    def predict_proba(self, X):
        self.last_input = np.asarray(X)
        return np.array([[0.1, 0.9]])


class ContextAwareRegressor:
    def __init__(self):
        self.feature_names = ["net_rating_diff", "pace"]
        self.context_feature_names = [
            "ctx_rest_days_diff",
            "ctx_line_movement",
            "ctx_away_travel_distance",
        ]
        self.last_X = None
        self.last_context = None

    def predict(self, X, context_features=None):
        self.last_X = np.asarray(X)
        self.last_context = np.asarray(context_features)
        return np.array([self.last_X[0, 0] + self.last_context[0, 1] + self.last_context[0, 2]])


def test_prepare_loaded_model_artifact_copies_metadata_to_inner_model():
    model = RecordingBinaryModel()
    artifact = {
        "model": model,
        "feature_names": ["feat_b", "feat_a"],
        "context_feature_names": ["ctx_days_rest_diff"],
    }

    prepared = prepare_loaded_model_artifact(artifact)

    assert prepared is model
    assert prepared.feature_names == ["feat_b", "feat_a"]
    assert prepared.context_feature_names == ["ctx_days_rest_diff"]


def test_predict_binary_probability_supports_wrapped_models():
    model = RecordingBinaryModel()
    artifact = {
        "model": model,
        "feature_names": ["feat_b", "feat_a"],
    }

    prob = predict_binary_probability(artifact, {"feat_a": 4.0, "feat_b": 9.0})

    assert prob == 0.8
    assert model.last_columns == ["feat_b", "feat_a"]
    assert model.last_row == {"feat_b": 9.0, "feat_a": 4.0}


def test_predict_binary_probability_supports_saved_stacking_dicts():
    meta_model = RecordingMetaClassifier()
    artifact = {
        "feature_names": ["feat_b", "feat_a"],
        "base_models": {
            "model_one": FixedBaseClassifier(0.7),
            "model_two": FixedBaseClassifier(0.4),
        },
        "meta_model": meta_model,
    }

    prob = predict_binary_probability(artifact, {"feat_a": 1.0, "feat_b": 2.0})

    assert prob == 0.9
    assert np.allclose(meta_model.last_input, np.array([[0.7, 0.4]]))


def test_predict_regression_value_passes_context_features_to_context_models():
    model = ContextAwareRegressor()

    value = predict_regression_value(
        model,
        {
            "net_rating_diff": 6.0,
            "pace": 99.0,
            "rest_days_diff": 1.0,
            "ctx_line_movement": -2.5,
            "away_travel_distance": 1200.0,
        },
    )

    assert value == 1203.5
    assert np.allclose(model.last_X, np.array([[6.0, 99.0]]))
    assert np.allclose(model.last_context, np.array([[1.0, -2.5, 1200.0]]))


def test_preserve_model_context_features_restores_filtered_context():
    from daily_predictions import preserve_model_context_features

    model = ContextAwareRegressor()
    filtered = {"net_rating_diff": 6.0}
    raw = {
        "net_rating_diff": 6.0,
        "rest_days_diff": 1.0,
        "ctx_line_movement": -2.5,
        "away_travel_distance": 1200.0,
    }

    preserved = preserve_model_context_features(filtered, raw, model)

    assert preserved["net_rating_diff"] == 6.0
    assert preserved["rest_days_diff"] == 1.0
    assert preserved["ctx_line_movement"] == -2.5
    assert preserved["away_travel_distance"] == 1200.0
