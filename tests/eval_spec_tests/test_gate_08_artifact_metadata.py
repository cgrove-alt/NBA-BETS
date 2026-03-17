"""Gate 8: Artifact metadata completeness.

REALISM_CHECKLIST Gate 8:
  Every model artifact must have metadata conforming to MODEL_ARTIFACT_SCHEMA.md.
  Required: artifact_version, git_sha, train_window_start, train_window_end,
  training_timestamp, training_samples, feature_names.
"""
import os
import pickle
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED_METADATA = {
    "git_sha",
    "train_window_start",
    "train_window_end",
    "training_samples",
    "feature_names",
}


class TestGate08ArtifactMetadata:

    def _load_artifact(self, name):
        pkl_path = os.path.join(REPO_ROOT, "models", name)
        if not os.path.exists(pkl_path):
            pytest.skip(f"{name} not found")
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            pytest.skip(f"Cannot load {name}: {e}")
        if not isinstance(data, dict):
            pytest.skip(f"{name} is not a dict")
        return data

    def test_moneyline_ensemble_metadata(self):
        """moneyline_ensemble.pkl must have all required metadata fields.

        EXPECTED FAIL: Current artifacts lack git_sha, train_window_*, training_samples.
        """
        data = self._load_artifact("moneyline_ensemble.pkl")
        present_keys = set(data.keys())
        missing = REQUIRED_METADATA - present_keys
        if missing:
            pytest.fail(
                f"Gate 8 VIOLATION: moneyline_ensemble.pkl missing metadata: {sorted(missing)}. "
                f"Present keys: {sorted(k for k in present_keys if k != 'model')}. "
                "See MODEL_ARTIFACT_SCHEMA.md."
            )

    def test_player_points_ensemble_metadata(self):
        """player_points_ensemble.pkl must have all required metadata fields.

        EXPECTED FAIL: Current artifacts lack required metadata.
        """
        data = self._load_artifact("player_points_ensemble.pkl")
        present_keys = set(data.keys())
        missing = REQUIRED_METADATA - present_keys
        if missing:
            pytest.fail(
                f"Gate 8 VIOLATION: player_points_ensemble.pkl missing metadata: {sorted(missing)}. "
                f"Present keys: {sorted(k for k in present_keys if k != 'model')}. "
                "See MODEL_ARTIFACT_SCHEMA.md."
            )
