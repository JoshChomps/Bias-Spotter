"""
Layer 1 — Lexicon Matching.

Fast, local, $0. Runs curated regex patterns against input text to catch
obvious persuasion techniques. Returns matches with character offsets.

This is the simplest and fastest layer. It catches:
- Scarcity/urgency cues
- Social proof language
- Authority appeals
- Loaded language patterns
- Identity bait
- Thought-terminating clichés
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LexiconMatch:
    """A single match from the lexicon layer."""

    start: int
    end: int
    text_span: str
    technique: str
    category: str
    explanation: str
    confidence: float = 0.85  # Lexicon matches are high-confidence by default


@dataclass
class CompiledPattern:
    """A compiled regex pattern with metadata."""

    regex: re.Pattern[str]
    technique: str
    category: str
    explanation: str


class Layer1Matcher:
    """
    Lexicon-based persuasion technique detector.

    Loads curated patterns from a YAML file and runs them against input text.
    Designed to be fast (~50ms) with zero external calls.
    """

    def __init__(self, lexicon_path: Path | str | None = None) -> None:
        if lexicon_path is None:
            lexicon_path = Path(__file__).parent.parent / "lexicon" / "techniques.yaml"
        self.lexicon_path = Path(lexicon_path)
        self.patterns: list[CompiledPattern] = []
        self._load_lexicon()

    def _load_lexicon(self) -> None:
        """Load and compile regex patterns from the YAML lexicon."""
        if not self.lexicon_path.exists():
            return

        with open(self.lexicon_path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        for technique_key, technique_data in data.items():
            if not isinstance(technique_data, dict):
                continue
            patterns = technique_data.get("patterns", [])
            category = technique_data.get("category", "uncategorized")
            explanation = technique_data.get("description", "")

            for pattern_str in patterns:
                try:
                    compiled = re.compile(pattern_str, re.IGNORECASE)
                    self.patterns.append(
                        CompiledPattern(
                            regex=compiled,
                            technique=technique_key,
                            category=category,
                            explanation=explanation,
                        )
                    )
                except re.error:
                    # Skip invalid patterns — log in production
                    continue

    def match(self, text: str) -> list[LexiconMatch]:
        """
        Run all patterns against the input text.

        Returns a list of LexiconMatch objects with character offsets.
        """
        matches: list[LexiconMatch] = []

        for pattern in self.patterns:
            for m in pattern.regex.finditer(text):
                matches.append(
                    LexiconMatch(
                        start=m.start(),
                        end=m.end(),
                        text_span=m.group(),
                        technique=pattern.technique,
                        category=pattern.category,
                        explanation=pattern.explanation,
                    )
                )

        # Sort by start position
        matches.sort(key=lambda x: x.start)
        return matches

    @property
    def pattern_count(self) -> int:
        """Number of compiled patterns loaded."""
        return len(self.patterns)
