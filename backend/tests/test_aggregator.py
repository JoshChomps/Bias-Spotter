"""
Aggregator Tests — Pipeline integration and deduplication.
"""

import pytest

from app.pipeline.aggregator import Aggregator, UnifiedDetection


@pytest.fixture
def aggregator() -> Aggregator:
    """Create an aggregator with default components."""
    return Aggregator()


class TestAggregator:
    """Test the aggregation and deduplication logic."""

    @pytest.mark.asyncio
    async def test_analyze_returns_result(self, aggregator: Aggregator) -> None:
        result = await aggregator.analyze("This is a test.")
        assert result.text_length > 0
        assert result.word_count > 0

    @pytest.mark.asyncio
    async def test_detections_are_sorted_by_position(self, aggregator: Aggregator) -> None:
        text = "Act now! This limited time offer is backed by experts who recommend it."
        result = await aggregator.analyze(text)
        for i in range(len(result.detections) - 1):
            assert result.detections[i].start <= result.detections[i + 1].start

    def test_deduplication_keeps_higher_confidence(self, aggregator: Aggregator) -> None:
        """When two detections overlap, the higher-confidence one should survive."""
        detections = [
            UnifiedDetection(
                start=0, end=10, text_span="act now!!!", technique="false_scarcity",
                category="framing", confidence=0.9, source_layer="lexicon",
                explanation="test",
            ),
            UnifiedDetection(
                start=0, end=10, text_span="act now!!!", technique="false_scarcity",
                category="framing", confidence=0.7, source_layer="classifier",
                explanation="test",
            ),
        ]
        result = aggregator._deduplicate(detections)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_non_overlapping_detections_both_kept(self, aggregator: Aggregator) -> None:
        """Non-overlapping detections should both be kept."""
        detections = [
            UnifiedDetection(
                start=0, end=10, text_span="first span", technique="a",
                category="emotional", confidence=0.8, source_layer="lexicon",
                explanation="test",
            ),
            UnifiedDetection(
                start=50, end=60, text_span="second span", technique="b",
                category="social", confidence=0.8, source_layer="lexicon",
                explanation="test",
            ),
        ]
        result = aggregator._deduplicate(detections)
        assert len(result) == 2
