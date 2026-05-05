"""
Layer 3 — LLM Reasoning (Selective).

Only invoked for:
- Low-confidence classifier predictions (below threshold)
- Text with structural cues suggesting subtle techniques
- User-requested deep analysis

Catches techniques the classifier can't handle:
- False dichotomy, motte-and-bailey, hidden premises, intent disambiguation

Pluggable backends:
- HuggingFace Inference API (free tier, default)
- Ollama (local, unlimited, free)
- Anthropic/OpenAI API (fallback for highest quality)

~2-5s per invocation. Cached by passage hash.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class LLMBackend(Enum):
    """Available LLM backends."""

    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class LLMPrediction:
    """A prediction from the LLM reasoning layer."""

    start: int
    end: int
    text_span: str
    technique: str
    category: str
    confidence: float
    explanation: str
    reasoning: str  # LLM's chain-of-thought reasoning


ANALYSIS_PROMPT_TEMPLATE = """Analyze the following text passage for subtle persuasion and manipulation techniques.

Focus ONLY on these specific techniques that require reasoning to detect:
- False dichotomy (presenting only two options when more exist)
- Motte-and-bailey (defending a weak claim by retreating to a stronger one)
- Hidden premises (unstated assumptions baked into the argument)
- Causal oversimplification (single cause for complex outcome)
- Whataboutism (deflecting by pointing to others' wrongdoing)
- Straw man (misrepresenting an opponent's argument)

For each technique found, return a JSON array with objects containing:
- "technique": the technique name
- "span": the exact text span
- "start": approximate character offset
- "end": approximate character offset
- "explanation": why this is manipulative
- "reasoning": your chain-of-thought analysis
- "confidence": 0.0-1.0

If no techniques are found, return an empty array: []

TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

Return ONLY valid JSON. No markdown, no explanation outside the JSON."""


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def analyze(self, text: str) -> list[LLMPrediction]:
        """Send text to the LLM and parse structured predictions."""
        ...


class HuggingFaceProvider(LLMProvider):
    """
    HuggingFace Inference API provider (free tier).

    STUB: Will be implemented in Phase 5.
    """

    def __init__(self, api_token: str | None = None, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3") -> None:
        self.api_token = api_token
        self.model_id = model_id

    async def analyze(self, text: str) -> list[LLMPrediction]:
        """TODO: Implement HuggingFace Inference API call."""
        return []


class OllamaProvider(LLMProvider):
    """
    Ollama local inference provider (free, unlimited).

    STUB: Will be implemented in Phase 5.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1") -> None:
        self.base_url = base_url
        self.model = model

    async def analyze(self, text: str) -> list[LLMPrediction]:
        """TODO: Implement Ollama API call."""
        return []


class Layer3LLM:
    """
    Selective LLM reasoning layer.

    Only activates when:
    - Classifier confidence is below threshold
    - Text contains structural cues (e.g., "either X or Y")
    - User explicitly requests deep analysis
    """

    # Confidence threshold — below this, invoke LLM
    CONFIDENCE_THRESHOLD: float = 0.6

    # Structural cues that suggest LLM analysis would help
    STRUCTURAL_CUES: list[str] = [
        "either",
        "or else",
        "the only option",
        "you must choose",
        "what about",
        "but they",
        "so you're saying",
    ]

    def __init__(
        self,
        backend: LLMBackend = LLMBackend.HUGGINGFACE,
        api_token: str | None = None,
    ) -> None:
        self.backend = backend
        self.provider = self._init_provider(backend, api_token)
        self._cache: dict[str, list[LLMPrediction]] = {}

    def _init_provider(self, backend: LLMBackend, api_token: str | None) -> LLMProvider:
        """Initialize the appropriate LLM provider."""
        if backend == LLMBackend.HUGGINGFACE:
            return HuggingFaceProvider(api_token=api_token)
        elif backend == LLMBackend.OLLAMA:
            return OllamaProvider()
        else:
            # Fallback to HuggingFace
            return HuggingFaceProvider(api_token=api_token)

    def should_invoke(self, text: str, classifier_confidence: float | None = None) -> bool:
        """
        Determine whether LLM analysis should be triggered.

        Returns True if:
        - Classifier confidence is below threshold
        - Text contains structural cues
        """
        if classifier_confidence is not None and classifier_confidence < self.CONFIDENCE_THRESHOLD:
            return True

        text_lower = text.lower()
        return any(cue in text_lower for cue in self.STRUCTURAL_CUES)

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate a cache key from the text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def analyze(self, text: str, force: bool = False) -> list[LLMPrediction]:
        """
        Run LLM analysis on the text.

        Uses cache to avoid re-analyzing the same content.

        STUB: Returns empty results until Phase 5 implementation.
        """
        cache_key = self._cache_key(text)

        if not force and cache_key in self._cache:
            return self._cache[cache_key]

        results = await self.provider.analyze(text)
        self._cache[cache_key] = results
        return results
