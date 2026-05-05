"""
Layer 2 — Fine-Tuned Classifier (The Hero).

This is the core of the system. A fine-tuned DeBERTa-v3 / RoBERTa classifier
trained on SemEval-2020 Task 11 data for:
  - Span identification (BIO tagging via token classification)
  - Technique classification (14 SemEval categories)

Runs locally via ONNX or transformers. ~100-300ms, $0.

STATUS: Stub — will be implemented after model training in Phases 2-3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClassifierPrediction:
    """A prediction from the fine-tuned classifier."""

    start: int
    end: int
    text_span: str
    technique: str
    category: str
    confidence: float
    explanation: str


class Layer2Classifier:
    """
    Fine-tuned transformer classifier for persuasion technique detection.

    This will load a trained model checkpoint and run inference locally.
    The model does span identification + technique classification in sequence.

    STUB: Returns empty results until the model is trained (Phases 2-3).
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self) -> None:
        """
        Load the trained model checkpoint.

        TODO (Phase 2-3):
        - Load DeBERTa-v3 checkpoint from self.model_path
        - Initialize tokenizer
        - Set self._loaded = True
        """
        pass

    def predict(self, text: str) -> list[ClassifierPrediction]:
        """
        Run the classifier on input text.

        Pipeline:
        1. Tokenize text with sliding window for long inputs
        2. Run span identification model → find persuasive spans
        3. Run technique classifier on each detected span
        4. Return predictions with confidence scores

        STUB: Returns empty list until model is trained.
        """
        if not self._loaded:
            return []

        # TODO: Implement actual inference
        return []

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        return self._loaded
