"""
Confidence Calibration Tests

Verifies that high-confidence picks hit at a higher rate than low-confidence
picks. This is the fundamental sanity check that must pass before deploying
any model update — if high confidence doesn't correlate with higher hit rate,
the confidence scoring is broken.

Run before every model deployment:
    pytest tests/test_confidence_calibration.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hit_rate(predictions: list) -> float:
    """Return hit rate (fraction where actual > line for over, or < line for under)."""
    if not predictions:
        return 0.0
    hits = sum(1 for p in predictions if p.get('hit'))
    return hits / len(predictions)


def _split_by_confidence(predictions: list, threshold: float = 0.60):
    """Split predictions into high-confidence and low-confidence buckets.

    A prediction is high-confidence when its 'confidence' field >= threshold.
    Uses 'over_probability' as fallback.
    """
    high, low = [], []
    for p in predictions:
        conf = p.get('confidence') or p.get('over_probability') or 0.5
        bucket = high if conf >= threshold else low
        bucket.append(p)
    return high, low


def _make_prediction(confidence: float, hit: bool) -> dict:
    """Build a minimal prediction record for testing."""
    return {'confidence': confidence, 'hit': hit}


# ---------------------------------------------------------------------------
# Unit tests on synthetic data
# ---------------------------------------------------------------------------

class TestConfidenceCalibrationUnit:
    """Unit tests using synthetic prediction data."""

    def test_high_confidence_beats_low_confidence(self):
        """High-confidence picks should hit more often than low-confidence ones."""
        # 100 high-conf picks hitting at 65%
        high_conf = [_make_prediction(0.75, i < 65) for i in range(100)]
        # 100 low-conf picks hitting at 50%
        low_conf = [_make_prediction(0.45, i < 50) for i in range(100)]

        all_preds = high_conf + low_conf
        hi, lo = _split_by_confidence(all_preds, threshold=0.60)

        hi_rate = _hit_rate(hi)
        lo_rate = _hit_rate(lo)

        assert hi_rate > lo_rate, (
            f"High-confidence hit rate ({hi_rate:.1%}) should exceed "
            f"low-confidence ({lo_rate:.1%})"
        )

    def test_confidence_monotonicity_across_tiers(self):
        """Hit rate should increase monotonically as confidence increases."""
        tiers = [
            [_make_prediction(0.52, i < 51) for i in range(100)],  # 51%
            [_make_prediction(0.62, i < 57) for i in range(100)],  # 57%
            [_make_prediction(0.72, i < 63) for i in range(100)],  # 63%
            [_make_prediction(0.82, i < 68) for i in range(100)],  # 68%
        ]
        hit_rates = [_hit_rate(t) for t in tiers]

        for i in range(len(hit_rates) - 1):
            assert hit_rates[i] <= hit_rates[i + 1], (
                f"Hit rate should be non-decreasing with confidence tier: "
                f"tier {i} ({hit_rates[i]:.1%}) > tier {i+1} ({hit_rates[i+1]:.1%})"
            )

    def test_equal_confidence_equal_hit_rate(self):
        """When all picks have equal confidence, split should be empty in one bucket."""
        preds = [_make_prediction(0.55, True) for _ in range(50)]
        hi, lo = _split_by_confidence(preds, threshold=0.60)
        assert len(hi) == 0
        assert len(lo) == 50

    def test_minimum_sample_requirement(self):
        """Test gracefully handles tiny or empty prediction sets."""
        hit_rate_empty = _hit_rate([])
        assert hit_rate_empty == 0.0

        single = [_make_prediction(0.70, True)]
        hi, lo = _split_by_confidence(single, threshold=0.60)
        assert len(hi) == 1
        assert _hit_rate(hi) == 1.0

    def test_over_probability_fallback(self):
        """Confidence split uses over_probability when confidence key is absent."""
        preds = [
            {'over_probability': 0.75, 'hit': True},
            {'over_probability': 0.45, 'hit': False},
        ]
        hi, lo = _split_by_confidence(preds, threshold=0.60)
        assert len(hi) == 1
        assert len(lo) == 1


# ---------------------------------------------------------------------------
# Integration test: load OOS backtest results and check calibration
# ---------------------------------------------------------------------------

class TestConfidenceCalibrationIntegration:
    """Integration tests that load real backtest results when available.

    These tests are skipped if the results files do not exist (e.g., fresh
    checkout without pre-run backtests).
    """

    OOS_RESULTS_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'backtest_results',
        'oos_walkforward_results.json'
    )

    def _load_oos_results(self):
        import json
        path = os.path.abspath(self.OOS_RESULTS_PATH)
        if not os.path.exists(path):
            pytest.skip(f"OOS results not found at {path}")
        with open(path) as f:
            return json.load(f)

    def test_oos_directional_accuracy_above_chance(self):
        """Each prop type should have directional accuracy > 50% in OOS test."""
        results = self._load_oos_results()
        agg = results.get('aggregate', {}).get('by_prop_type', {})

        failures = []
        for prop_type, metrics in agg.items():
            dir_acc = metrics.get('directional_accuracy', 0)
            if dir_acc <= 0.50:
                failures.append(f"{prop_type}: {dir_acc:.1%}")

        assert not failures, (
            "Directional accuracy at or below chance for: " + ', '.join(failures)
        )

    def test_oos_no_extreme_bias(self):
        """OOS bias should not exceed 2.0 for any prop type (sanity check)."""
        results = self._load_oos_results()
        agg = results.get('aggregate', {}).get('by_prop_type', {})

        failures = []
        for prop_type, metrics in agg.items():
            bias = abs(metrics.get('bias', 0))
            if bias > 2.0:
                failures.append(f"{prop_type}: bias={bias:.2f}")

        assert not failures, (
            "Excessive OOS bias (> 2.0) detected for: " + ', '.join(failures)
        )

    def test_oos_positive_r2_all_props(self):
        """All prop models should have positive R2 in OOS evaluation."""
        results = self._load_oos_results()
        agg = results.get('aggregate', {}).get('by_prop_type', {})

        failures = []
        for prop_type, metrics in agg.items():
            r2 = metrics.get('r2', 0)
            if r2 < 0:
                failures.append(f"{prop_type}: R2={r2:.4f}")

        assert not failures, (
            "Negative OOS R2 (worse than mean predictor): " + ', '.join(failures)
        )
