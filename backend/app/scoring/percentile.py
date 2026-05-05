"""
Percentile Ranker — Compares analysis against a reference corpus.

Pre-scores 100-500 reference documents across genres:
- Encyclopedia (low manipulation)
- News reporting (low-moderate)
- Opinion columns (moderate)
- Advertising (moderate-high)
- Political speeches (high)
- Propaganda (very high)

Stores reference scores in SQLite. Computes percentile of new input
against this corpus.

STATUS: Stub — will be built in Phase 6 after the pipeline is working.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PercentileResult:
    """The percentile ranking result."""

    percentile: float  # 0-100
    corpus_size: int
    genre_comparisons: dict[str, float]  # percentile within each genre


class PercentileRanker:
    """
    Ranks new text against a pre-scored reference corpus.

    STUB: Returns None until the reference corpus is built (Phase 6).
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path
        self._corpus_loaded = False

    def rank(self, density_score: float) -> PercentileResult | None:
        """
        Compute the percentile rank of a density score against the reference corpus.

        Returns None if the reference corpus hasn't been built yet.

        TODO (Phase 6):
        - Load reference scores from SQLite
        - Compute percentile using bisect
        - Return genre-specific comparisons
        """
        if not self._corpus_loaded:
            return None

        return None

    def build_corpus(self, documents: list[dict]) -> None:
        """
        Score a batch of reference documents and store results.

        TODO (Phase 6):
        - Accept list of {text, genre} dicts
        - Run each through the analysis pipeline
        - Store density scores in SQLite keyed by genre
        """
        pass
