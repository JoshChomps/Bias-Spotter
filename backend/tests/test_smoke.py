"""
Smoke Tests — Verify all imports work and the app starts.
"""

import pytest


def test_imports() -> None:
    """All core modules should be importable."""
    from app.pipeline.layer1_lexicon import Layer1Matcher, LexiconMatch
    from app.pipeline.layer2_classifier import Layer2Classifier, ClassifierPrediction
    from app.pipeline.layer3_llm import Layer3LLM, LLMPrediction
    from app.pipeline.aggregator import Aggregator, UnifiedDetection
    from app.scoring.density import DensityScorer, DensityResult
    from app.scoring.percentile import PercentileRanker
    from app.parser.document_parser import DocumentParser, ParsedDocument
    from app.cache.cache import AnalysisCache


def test_fastapi_app_creates() -> None:
    """The FastAPI app should be importable and have expected endpoints."""
    from app.main import app

    routes = [route.path for route in app.routes]
    assert "/health" in routes
    assert "/analyze/text" in routes
    assert "/analyze/document" in routes


def test_layer1_matcher_loads() -> None:
    """Layer 1 matcher should load the lexicon without errors."""
    from app.pipeline.layer1_lexicon import Layer1Matcher

    matcher = Layer1Matcher()
    assert matcher.pattern_count > 0, "Lexicon should have at least some patterns loaded."


def test_density_scorer_empty() -> None:
    """Density scorer should handle empty input gracefully."""
    from app.scoring.density import DensityScorer

    scorer = DensityScorer()
    result = scorer.score([], 0)
    assert result.density_score == 0.0
    assert result.techniques_per_100_words == 0.0
