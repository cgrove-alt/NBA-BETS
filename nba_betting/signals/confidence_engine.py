"""
Unified Confidence Scoring Engine

Replaces the fragmented confidence calculation with a single, coherent system.
Confidence should represent calibrated probability of the bet winning.

Key principle: Confidence = f(model_probability, edge_size, model_agreement, data_quality)
NOT independently computed from edge.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class ConfidenceEngine:
    """
    Compute a unified confidence score from multiple model signals.

    Score range: 0-100 where:
    - 0-30: Low confidence (PASS)
    - 30-50: Marginal confidence (LEAN)
    - 50-70: Moderate confidence (BET with reduced size)
    - 70-85: Strong confidence (BET with standard Kelly)
    - 85-100: Elite confidence (BET with aggressive Kelly)
    """

    # Weights for confidence components (must sum to 1.0)
    WEIGHTS = {
        'model_probability': 0.40,    # Primary signal
        'model_agreement': 0.20,       # Cross-model consistency
        'edge_magnitude': 0.15,        # Size of edge vs market
        'data_quality': 0.15,          # Completeness of input features
        'historical_accuracy': 0.10,   # Calibration track record for this type
    }

    def __init__(
        self,
        min_confidence_to_bet: float = 45.0,
        min_edge_to_bet: float = 0.03,
    ):
        self.min_confidence_to_bet = min_confidence_to_bet
        self.min_edge_to_bet = min_edge_to_bet

    def score(
        self,
        model_probability: float,
        true_edge: float,
        model_agreement: float = 1.0,
        data_quality_score: float = 1.0,
        historical_accuracy: float = 0.5,
        prop_type: str | None = None,
    ) -> dict:
        """
        Compute unified confidence score.

        Args:
            model_probability: Calibrated model probability (0-1)
            true_edge: Edge after devigging (can be negative)
            model_agreement: Agreement across ensemble models (0-1, 1=perfect)
            data_quality_score: Feature completeness (0-1, 1=all features present)
            historical_accuracy: Historical accuracy for this bet type (0-1)
            prop_type: Optional prop type for type-specific adjustments

        Returns:
            Dictionary with confidence score and components
        """
        # Component 1: Model probability strength (distance from 50%)
        prob_strength = abs(model_probability - 0.5) * 2  # 0 to 1 scale
        prob_score = np.clip(prob_strength * 100, 0, 100)

        # Component 2: Model agreement (ensemble consistency)
        agreement_score = np.clip(model_agreement * 100, 0, 100)

        # Component 3: Edge magnitude (larger edge = more confident)
        # Normalize: 0% edge = 0 score, 5% edge = 50, 15%+ edge = 100
        edge_score = np.clip(abs(true_edge) / 0.15 * 100, 0, 100)

        # Component 4: Data quality
        quality_score = np.clip(data_quality_score * 100, 0, 100)

        # Component 5: Historical accuracy for this type
        hist_score = np.clip(historical_accuracy * 100, 0, 100)

        # Weighted composite
        composite = (
            self.WEIGHTS['model_probability'] * prob_score +
            self.WEIGHTS['model_agreement'] * agreement_score +
            self.WEIGHTS['edge_magnitude'] * edge_score +
            self.WEIGHTS['data_quality'] * quality_score +
            self.WEIGHTS['historical_accuracy'] * hist_score
        )

        # Apply prop-type-specific adjustments
        if prop_type:
            composite = self._apply_type_adjustment(composite, prop_type)

        # Penalty for edge-confidence disagreement
        # High edge + low model probability = suspicious
        if true_edge > 0.10 and model_probability < 0.55:
            composite *= 0.85  # 15% penalty

        confidence = np.clip(composite, 0, 100)

        # Determine signal
        signal = self._get_signal(confidence, true_edge)
        tier = self._get_tier(confidence)

        return {
            'confidence': round(float(confidence), 1),
            'signal': signal,
            'tier': tier,
            'components': {
                'model_probability': round(prob_score, 1),
                'model_agreement': round(agreement_score, 1),
                'edge_magnitude': round(edge_score, 1),
                'data_quality': round(quality_score, 1),
                'historical_accuracy': round(hist_score, 1),
            },
            'should_bet': signal in ('BET', 'STRONG_BET'),
            'kelly_fraction': self._kelly_fraction(tier),
        }

    def _apply_type_adjustment(self, confidence: float, prop_type: str) -> float:
        """Apply prop-type-specific confidence adjustments."""
        # Based on historical calibration accuracy by type
        type_multipliers = {
            'points': 1.0,      # Well-calibrated
            'rebounds': 0.95,   # Slightly overconfident historically
            'assists': 1.02,    # Slightly underconfident historically
            'threes': 0.90,     # Significantly overconfident
            'pra': 0.98,        # Slightly overconfident
        }
        multiplier = type_multipliers.get(prop_type, 1.0)
        return confidence * multiplier

    def _get_signal(self, confidence: float, true_edge: float) -> str:
        """Map confidence + edge to a betting signal."""
        if true_edge < -self.min_edge_to_bet:
            return 'FADE'
        if confidence >= 70 and true_edge >= self.min_edge_to_bet:
            return 'STRONG_BET'
        if confidence >= self.min_confidence_to_bet and true_edge >= self.min_edge_to_bet:
            return 'BET'
        if confidence >= 30 and true_edge >= 0.02:
            return 'LEAN'
        return 'PASS'

    def _get_tier(self, confidence: float) -> str:
        """Map confidence to a tier label."""
        if confidence >= 85:
            return 'elite'
        if confidence >= 70:
            return 'strong'
        if confidence >= 50:
            return 'moderate'
        if confidence >= 30:
            return 'marginal'
        return 'avoid'

    def _kelly_fraction(self, tier: str) -> float:
        """Get Kelly fraction multiplier by confidence tier."""
        fractions = {
            'elite': 0.50,
            'strong': 0.35,
            'moderate': 0.25,
            'marginal': 0.15,
            'avoid': 0.0,
        }
        return fractions.get(tier, 0.0)
