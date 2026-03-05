from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models.stacking_model import StackingClassifier
from nba_models.models import model_trainer
from nba_models.models.stacking_meta_learner import StackingMetaLearner


def _build_saved_stacking_classifier(path):
    X = pd.DataFrame(
        {
            "f1": [0.0, 0.0, 1.0, 1.0],
            "f2": [0.0, 1.0, 0.0, 1.0],
        }
    )
    y = np.array([0, 0, 1, 1])

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    base_model = LogisticRegression().fit(X_scaled, y)
    base_predictions = base_model.predict_proba(X_scaled)[:, 1].reshape(-1, 1)
    meta_model = LogisticRegression().fit(base_predictions, y)

    model = StackingClassifier()
    model.scaler = scaler
    model.base_models = {"logreg": base_model}
    model.meta_model = meta_model
    model.feature_names = list(X.columns)
    model.use_proba = True
    model.is_fitted = True
    model.save(str(path))


def test_load_all_models_handles_direct_model_objects(tmp_path, monkeypatch):
    artifact_path = tmp_path / "moneyline_stacking_metalearner.pkl"
    with artifact_path.open("wb") as f:
        pickle.dump(StackingMetaLearner(base_models=[]), f)

    monkeypatch.setattr(model_trainer, "MODEL_DIR", tmp_path)
    pipeline = model_trainer.ModelTrainingPipeline()

    loaded_models = pipeline.load_all_models()

    assert "moneyline_stacking_metalearner" in loaded_models
    assert hasattr(loaded_models["moneyline_stacking_metalearner"], "predict")


def test_load_all_models_handles_stacking_classifier_dict_artifacts(tmp_path, monkeypatch):
    artifact_path = tmp_path / "moneyline_stacking.pkl"
    _build_saved_stacking_classifier(artifact_path)

    monkeypatch.setattr(model_trainer, "MODEL_DIR", tmp_path)
    pipeline = model_trainer.ModelTrainingPipeline()

    loaded_models = pipeline.load_all_models()
    model = loaded_models["moneyline_stacking"]

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
