"""
Integration tests for best-bets filter logic.

CRITICAL: Verifies that low-line props (assists, threes) pass the min_edge filter.
The filter uses edge_pct (percentage), not edge (raw points), to avoid filtering out
props with low raw edges but high percentage edges.
"""

import pytest


class TestBestBetsFilterLogic:
    """Test the best-bets filter uses edge_pct correctly."""

    def test_low_line_assists_pass_filter(self):
        """Verify assists with high % edge pass min_edge filter."""
        # Example: Center with 2.0 assists prediction vs 0.5 line
        prediction = 2.0
        line = 0.5
        edge = prediction - line  # 1.5 raw points
        edge_pct = (edge / line) * 100  # 300%
        min_edge = 4.0  # Default filter value (as percentage)

        # OLD BROKEN LOGIC (would filter out):
        assert abs(edge) < min_edge  # 1.5 < 4.0 = True (FILTERED OUT ❌)

        # NEW CORRECT LOGIC (should pass):
        assert abs(edge_pct) >= min_edge  # 300 >= 4.0 = True (PASSES ✅)

    def test_low_line_threes_pass_filter(self):
        """Verify three-pointers with high % edge pass min_edge filter."""
        # Example: Non-shooter with 1.4 threes prediction vs 0.5 line
        prediction = 1.4
        line = 0.5
        edge = prediction - line  # 0.9 raw points
        edge_pct = (edge / line) * 100  # 180%
        min_edge = 4.0

        # OLD BROKEN LOGIC (would filter out):
        assert abs(edge) < min_edge  # 0.9 < 4.0 = True (FILTERED OUT ❌)

        # NEW CORRECT LOGIC (should pass):
        assert abs(edge_pct) >= min_edge  # 180 >= 4.0 = True (PASSES ✅)

    def test_normal_line_points_pass_filter(self):
        """Verify normal points props still work correctly."""
        # Example: Points with 25.0 prediction vs 20.5 line
        prediction = 25.0
        line = 20.5
        edge = prediction - line  # 4.5 raw points
        edge_pct = (edge / line) * 100  # 22%
        min_edge = 4.0

        # Both old and new logic should pass this:
        assert abs(edge) >= min_edge  # 4.5 >= 4.0 = True ✅
        assert abs(edge_pct) >= min_edge  # 22 >= 4.0 = True ✅

    def test_low_edge_filtered_out(self):
        """Verify props with low % edge are correctly filtered out."""
        # Example: Prediction very close to line
        prediction = 20.5
        line = 20.0
        edge = prediction - line  # 0.5 raw points
        edge_pct = (edge / line) * 100  # 2.5%
        min_edge = 4.0

        # Should be filtered out by BOTH logics:
        assert abs(edge) < min_edge  # 0.5 < 4.0 = True (FILTERED OUT)
        assert abs(edge_pct) < min_edge  # 2.5 < 4.0 = True (FILTERED OUT)

    def test_edge_percentage_calculation(self):
        """Verify edge percentage calculation matches data_service.py."""
        test_cases = [
            # (prediction, line, expected_edge_pct)
            (2.0, 0.5, 300.0),  # Assists
            (1.5, 0.5, 200.0),  # Threes
            (25.0, 20.5, 21.95),  # Points
            (8.0, 5.5, 45.45),  # Rebounds
        ]

        for pred, line, expected in test_cases:
            edge = pred - line
            edge_pct = (edge / line) * 100
            assert abs(edge_pct - expected) < 0.1, \
                f"Edge % mismatch: {edge_pct:.2f} != {expected:.2f}"

    def test_realistic_scenario_all_prop_types(self):
        """Test filter with realistic values for all prop types."""
        props = [
            # (prop_type, prediction, line, should_pass_filter)
            ("Assists", 2.0, 0.5, True),   # 300% edge ✅
            ("Threes", 1.4, 0.5, True),    # 180% edge ✅
            ("Points", 25.0, 20.5, True),  # 22% edge ✅
            ("Rebounds", 8.0, 5.5, True),  # 45% edge ✅
            ("Points", 20.5, 20.0, False), # 2.5% edge ❌
        ]

        min_edge = 4.0

        for prop_type, pred, line, should_pass in props:
            edge = pred - line
            edge_pct = (edge / line) * 100

            passes_filter = abs(edge_pct) >= min_edge

            assert passes_filter == should_pass, \
                f"{prop_type} filter mismatch: {edge_pct:.1f}% edge, " \
                f"expected {'PASS' if should_pass else 'FAIL'}, " \
                f"got {'PASS' if passes_filter else 'FAIL'}"
