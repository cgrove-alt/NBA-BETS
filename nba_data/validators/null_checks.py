"""
Null/NaN checks for required prediction features.

Validates that critical features are present and non-null before
predictions are generated.
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


# Required features for team-level game predictions (spread, moneyline)
REQUIRED_GAME_FEATURES = [
    'net_rating_diff',
    'win_pct_diff',
    'off_rating_diff',
    'def_rating_diff',
    'pace_diff',
]

# Required features for player prop predictions
REQUIRED_PROP_FEATURES = [
    'avg_minutes',
    'avg_stat',  # Generic — the primary stat being predicted
]


def _is_missing(value: Any) -> bool:
    """Check if a value is missing (None, NaN, or inf)."""
    if value is None:
        return True
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return True
    return False


def validate_features(
    features: dict,
    required: list[str] | None = None,
    context: str = "",
) -> tuple[bool, list[str]]:
    """
    Validate that required features are present and non-null.

    Args:
        features: Feature dictionary
        required: List of required feature names (defaults to REQUIRED_GAME_FEATURES)
        context: Context string for logging (e.g., game matchup)

    Returns:
        (is_valid, issues) — is_valid is True if all required features are present
        and non-null. issues is a list of human-readable problem descriptions.
    """
    if required is None:
        required = REQUIRED_GAME_FEATURES

    issues = []

    for feat_name in required:
        if feat_name not in features:
            issues.append(f"missing '{feat_name}'")
        elif _is_missing(features[feat_name]):
            issues.append(f"null/NaN '{feat_name}'")

    is_valid = len(issues) == 0

    if not is_valid and context:
        logger.warning(
            f"Feature validation failed ({context}): {', '.join(issues)}"
        )

    return is_valid, issues


def count_valid_features(features: dict) -> tuple[int, int]:
    """
    Count how many features are valid (non-null, non-NaN) vs total.

    Returns:
        (valid_count, total_count)
    """
    total = len(features)
    valid = sum(1 for v in features.values() if not _is_missing(v))
    return valid, total
