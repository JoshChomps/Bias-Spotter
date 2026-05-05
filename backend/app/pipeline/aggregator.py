"""
Aggregator — Combines outputs from all three analysis layers.

Responsibilities:
- Run all three layers (lexicon, classifier, LLM)
- Merge results, deduplicating overlapping spans
- When lexicon and classifier both flag the same phrase, prefer the
  higher-confidence source
- Return a unified list of detections with source attribution

This is the single entry point for the analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.pipeline.layer1_lexicon import Layer1Matcher, LexiconMatch
from app.pipeline.layer2_classifier import Layer2Classifier, ClassifierPrediction
from app.pipeline.layer3_llm import Layer3LLM, LLMPrediction


@dataclass
class UnifiedDetection:
    """A single detected persuasion technique from any layer."""

    start: int
    end: int
    text_span: str
    technique: str
    category: str
    confidence: float
    source_layer: str  # "lexicon", "classifier", "llm"
    explanation: str


@dataclass
class AggregatedResult:
    """The full analysis result combining all layers."""

    detections: list[UnifiedDetection] = field(default_factory=list)
    text_length: int = 0
    word_count: int = 0


class Aggregator:
    """
    Combines outputs from all three layers into a unified analysis.

    Usage:
        aggregator = Aggregator()
        result = await aggregator.analyze("Some text to analyze...")
    """

    # Overlap threshold — if two spans overlap by more than this fraction,
    # they're considered duplicates
    OVERLAP_THRESHOLD: float = 0.5

    def __init__(
        self,
        lexicon_matcher: Layer1Matcher | None = None,
        classifier: Layer2Classifier | None = None,
        llm_layer: Layer3LLM | None = None,
    ) -> None:
        self.lexicon = lexicon_matcher or Layer1Matcher()
        self.classifier = classifier or Layer2Classifier()
        self.llm = llm_layer or Layer3LLM()

    async def analyze(self, text: str, deep_analysis: bool = False) -> AggregatedResult:
        """
        Run the full analysis pipeline on input text.

        1. Layer 1 (lexicon) — always runs, fast
        2. Layer 2 (classifier) — always runs if model is loaded
        3. Layer 3 (LLM) — runs selectively based on confidence / cues
        """
        all_detections: list[UnifiedDetection] = []

        # Layer 1: Lexicon matching
        lexicon_matches = self.lexicon.match(text)
        for match in lexicon_matches:
            all_detections.append(
                UnifiedDetection(
                    start=match.start,
                    end=match.end,
                    text_span=match.text_span,
                    technique=match.technique,
                    category=match.category,
                    confidence=match.confidence,
                    source_layer="lexicon",
                    explanation=match.explanation,
                )
            )

        # Layer 2: Classifier (if model is loaded)
        classifier_preds = self.classifier.predict(text)
        min_confidence = 1.0
        for pred in classifier_preds:
            all_detections.append(
                UnifiedDetection(
                    start=pred.start,
                    end=pred.end,
                    text_span=pred.text_span,
                    technique=pred.technique,
                    category=pred.category,
                    confidence=pred.confidence,
                    source_layer="classifier",
                    explanation=pred.explanation,
                )
            )
            min_confidence = min(min_confidence, pred.confidence)

        # Layer 3: LLM (selective)
        if deep_analysis or self.llm.should_invoke(text, min_confidence if classifier_preds else None):
            llm_preds = await self.llm.analyze(text, force=deep_analysis)
            for pred in llm_preds:
                all_detections.append(
                    UnifiedDetection(
                        start=pred.start,
                        end=pred.end,
                        text_span=pred.text_span,
                        technique=pred.technique,
                        category=pred.category,
                        confidence=pred.confidence,
                        source_layer="llm",
                        explanation=pred.explanation,
                    )
                )

        # Deduplicate overlapping spans
        deduped = self._deduplicate(all_detections)

        return AggregatedResult(
            detections=deduped,
            text_length=len(text),
            word_count=len(text.split()),
        )

    def _deduplicate(self, detections: list[UnifiedDetection]) -> list[UnifiedDetection]:
        """
        Remove overlapping detections, keeping the highest-confidence one.

        Two spans are considered overlapping if their intersection is more than
        OVERLAP_THRESHOLD of the smaller span's length.
        """
        if not detections:
            return []

        # Sort by confidence descending — keep highest first
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[UnifiedDetection] = []

        for det in sorted_dets:
            is_duplicate = False
            for existing in kept:
                if self._overlaps(det, existing):
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(det)

        # Re-sort by position
        kept.sort(key=lambda d: d.start)
        return kept

    def _overlaps(self, a: UnifiedDetection, b: UnifiedDetection) -> bool:
        """Check if two detections overlap beyond the threshold."""
        overlap_start = max(a.start, b.start)
        overlap_end = min(a.end, b.end)
        overlap_len = max(0, overlap_end - overlap_start)

        if overlap_len == 0:
            return False

        smaller_len = min(a.end - a.start, b.end - b.start)
        if smaller_len == 0:
            return False

        return (overlap_len / smaller_len) > self.OVERLAP_THRESHOLD
