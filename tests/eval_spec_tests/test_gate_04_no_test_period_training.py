"""Gate 4: No test-period data in training.

REALISM_CHECKLIST Gate 4:
  Model weights used for test-period predictions must be trained ONLY
  on data with game_date < test_start_date.

EXPECTED: This test FAILS because:
  1. profitability_backtest.py admits in-sample model (L11)
  2. Model artifacts lack train_window_end metadata
"""
import os
import pickle
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGate04NoTestPeriodTraining:

    def test_profitability_backtest_admits_in_sample(self):
        """profitability_backtest.py docstring must NOT admit in-sample testing
        without a RESEARCH-ONLY label.

        EXPECTED FAIL: Current docstring says 'in-sample model'.
        """
        bt_path = os.path.join(REPO_ROOT, "nba_models", "backtesting", "profitability_backtest.py")
        if not os.path.exists(bt_path):
            pytest.skip("profitability_backtest.py not found")

        with open(bt_path) as f:
            # Read first 20 lines (docstring area)
            header = "".join(f.readline() for _ in range(20))

        if "in-sample" in header.lower():
            has_label = "RESEARCH-ONLY" in header or "REALISM_LEVEL" in header
            if not has_label:
                pytest.fail(
                    "Gate 4 VIOLATION: profitability_backtest.py admits 'in-sample' model "
                    "but has no RESEARCH-ONLY realism label. "
                    "See REALISM_CHECKLIST.md Gate 4."
                )

    def test_model_artifacts_have_train_window(self):
        """Model pkl files must contain train_window_start and train_window_end.

        EXPECTED FAIL: Current artifacts lack these fields.
        """
        models_dir = os.path.join(REPO_ROOT, "models")
        if not os.path.isdir(models_dir):
            pytest.skip("models/ directory not found")

        # Check one representative model
        for candidate in ["player_points_ensemble.pkl", "moneyline_ensemble.pkl"]:
            pkl_path = os.path.join(models_dir, candidate)
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                except Exception:
                    pytest.skip(f"Cannot load {candidate} (pickle error)")

                if isinstance(data, dict):
                    has_window = (
                        "train_window_start" in data and "train_window_end" in data
                    )
                    if not has_window:
                        pytest.fail(
                            f"Gate 4 VIOLATION: {candidate} missing train_window_start "
                            f"and/or train_window_end. Keys present: "
                            f"{[k for k in data.keys() if k != 'model']}"
                        )
                    return

        pytest.skip("No suitable model artifact found to check")
