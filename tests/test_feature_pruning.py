"""
Tests for feature selection/pruning infrastructure.

Covers:
- FeatureSelector save/load round-trip
- transform correctness
- filter_features utility
- Edge cases (empty selection, missing features)
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFeatureSelector:
    """Test the FeatureSelector class from feature_engineering.py."""

    def _get_selector_class(self):
        from nba_data.transformers.feature_engineering import FeatureSelector
        return FeatureSelector

    def test_save_load_roundtrip(self, tmp_path):
        """Selected features survive save/load cycle."""
        FeatureSelector = self._get_selector_class()
        selector = FeatureSelector()
        selector.selected_features = ['feat_a', 'feat_b', 'feat_c']
        selector.feature_importances = {'feat_a': 0.5, 'feat_b': 0.3, 'feat_c': 0.2}
        selector.is_fitted = True

        filepath = str(tmp_path / 'test_features.json')
        selector.save(filepath)

        loaded = FeatureSelector()
        loaded.load(filepath)

        assert loaded.selected_features == ['feat_a', 'feat_b', 'feat_c']
        assert loaded.feature_importances == {'feat_a': 0.5, 'feat_b': 0.3, 'feat_c': 0.2}
        assert loaded.is_fitted is True

    def test_save_creates_parent_dirs(self, tmp_path):
        """Save creates parent directories if they don't exist."""
        FeatureSelector = self._get_selector_class()
        selector = FeatureSelector()
        selector.selected_features = ['x']
        selector.feature_importances = {}
        selector.is_fitted = True

        filepath = str(tmp_path / 'nested' / 'dir' / 'features.json')
        selector.save(filepath)

        assert os.path.exists(filepath)

    def test_transform_filters_columns(self):
        """Transform should return only selected feature columns."""
        FeatureSelector = self._get_selector_class()
        selector = FeatureSelector()
        selector.selected_features = ['a', 'c']
        selector.is_fitted = True

        X = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        result = selector.transform(X)

        assert list(result.columns) == ['a', 'c']
        assert len(result) == 2

    def test_transform_handles_missing_features(self):
        """Transform handles features missing from X gracefully."""
        FeatureSelector = self._get_selector_class()
        selector = FeatureSelector()
        selector.selected_features = ['a', 'missing_feat', 'c']
        selector.is_fitted = True

        X = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        result = selector.transform(X)

        assert 'a' in result.columns
        assert 'c' in result.columns
        assert 'missing_feat' not in result.columns

    def test_transform_raises_if_not_fitted(self):
        """Transform raises ValueError if not fitted."""
        FeatureSelector = self._get_selector_class()
        selector = FeatureSelector()

        X = pd.DataFrame({'a': [1]})
        with pytest.raises(ValueError, match="not fitted"):
            selector.transform(X)

    def test_json_file_format(self, tmp_path):
        """Saved JSON has expected structure."""
        FeatureSelector = self._get_selector_class()
        selector = FeatureSelector()
        selector.selected_features = ['x', 'y']
        selector.feature_importances = {'x': 0.7, 'y': 0.3}
        selector.elimination_history = [{'n_features': 3, 'cv_score': 0.85}]
        selector.is_fitted = True

        filepath = str(tmp_path / 'features.json')
        selector.save(filepath)

        with open(filepath) as f:
            data = json.load(f)

        assert 'selected_features' in data
        assert 'feature_importances' in data
        assert 'saved_at' in data
        assert data['selected_features'] == ['x', 'y']


class TestFilterFeatures:
    """Test the filter_features utility function."""

    def test_filters_when_selection_provided(self):
        from nba_data.transformers.feature_engineering import filter_features
        features = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        selected = {'a', 'c'}
        result = filter_features(features, selected)
        assert result == {'a': 1, 'c': 3}

    def test_passes_through_when_no_selection(self):
        from nba_data.transformers.feature_engineering import filter_features
        features = {'a': 1, 'b': 2}
        result = filter_features(features, selected=None)
        assert result == {'a': 1, 'b': 2}

    def test_empty_selection_passes_through(self):
        """Empty selection set = no pruning applied (e.g. placeholder file)."""
        from nba_data.transformers.feature_engineering import filter_features
        features = {'a': 1, 'b': 2}
        result = filter_features(features, selected=set())
        assert result == {'a': 1, 'b': 2}

    def test_empty_features_returns_empty(self):
        from nba_data.transformers.feature_engineering import filter_features
        result = filter_features({}, selected={'a', 'b'})
        assert result == {}

    def test_no_overlap_returns_empty(self):
        from nba_data.transformers.feature_engineering import filter_features
        features = {'x': 1, 'y': 2}
        result = filter_features(features, selected={'a', 'b'})
        assert result == {}


class TestLoadSelectedFeatures:
    """Test the load_selected_features function."""

    def test_loads_valid_json(self, tmp_path):
        from nba_data.transformers.feature_engineering import load_selected_features
        filepath = tmp_path / 'sel.json'
        filepath.write_text(json.dumps({
            'selected_features': ['feat1', 'feat2'],
        }))
        result = load_selected_features(filepath)
        assert result == {'feat1', 'feat2'}

    def test_returns_none_for_missing_file(self, tmp_path):
        from nba_data.transformers.feature_engineering import load_selected_features
        result = load_selected_features(tmp_path / 'nonexistent.json')
        assert result is None

    def test_empty_selection_returns_empty_set(self, tmp_path):
        from nba_data.transformers.feature_engineering import load_selected_features
        filepath = tmp_path / 'empty.json'
        filepath.write_text(json.dumps({'selected_features': []}))
        result = load_selected_features(filepath)
        assert result == set()
