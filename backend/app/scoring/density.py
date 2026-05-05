"""
Density Scorer — Computes manipulation density metrics.

Calculates:
- Techniques per 100 words
- Category breakdown (emotional, social, logical, framing, rhetorical)
- Overall density score (0-100)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CategoryBreakdown:
    """Count of detections per meta-category."""

    emotional: int = 0
    social: int = 0
    logical: int = 0
    framing: int = 0
    rhetorical: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "emotional": self.emotional,
            "social": self.social,
            "logical": self.logical,
            "framing": self.framing,
            "rhetorical": self.rhetorical,
        }


@dataclass
class DensityResult:
    """The density scoring result."""

    density_score: float  # 0-100
    techniques_per_100_words: float
    word_count: int
    total_techniques: int
    category_breakdown: CategoryBreakdown


class DensityScorer:
    """
    Computes manipulation density from aggregated detections.

    The density score is normalized to 0-100 where:
    - 0 = no detected techniques (e.g., encyclopedia text)
    - 50 = moderate manipulation (e.g., opinion column)
    - 100 = extreme manipulation density (e.g., propaganda)

    The scaling is calibrated against the reference corpus.
    """

    # Baseline: techniques per 100 words that maps to a score of 50
    # Calibrated against reference corpus — adjust after Phase 6
    BASELINE_DENSITY: float = 2.0  # 2 techniques per 100 words = score 50

    def score(self, detections: list[Any], word_count: int) -> DensityResult:
        """
        Compute the density score from a list of detections.

        Args:
            detections: List of UnifiedDetection objects from the aggregator.
            word_count: Total word count of the analyzed text.

        Returns:
            DensityResult with score, breakdown, and metrics.
        """
        if word_count == 0:
            return DensityResult(
                density_score=0.0,
                techniques_per_100_words=0.0,
                word_count=0,
                total_techniques=0,
                category_breakdown=CategoryBreakdown(),
            )

        total = len(detections)
        per_100 = (total / word_count) * 100

        # Compute category breakdown
        breakdown = CategoryBreakdown()
        for det in detections:
            category = getattr(det, "category", "uncategorized")
            if hasattr(breakdown, category):
                setattr(breakdown, category, getattr(breakdown, category) + 1)

        # Map density to 0-100 score using sigmoid-like scaling
        # density_ratio = per_100 / BASELINE_DENSITY
        # score = 100 * (1 - e^(-density_ratio)) — saturates at 100
        import math

        density_ratio = per_100 / self.BASELINE_DENSITY
        density_score = min(100.0, 100.0 * (1 - math.exp(-density_ratio)))

        return DensityResult(
            density_score=round(density_score, 1),
            techniques_per_100_words=round(per_100, 2),
            word_count=word_count,
            total_techniques=total,
            category_breakdown=breakdown,
        )
