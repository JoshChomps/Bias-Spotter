"""
Cache Layer — SQLite-backed caching by content hash.

Stores analysis results keyed by sha256(text) so:
- Re-analysis of the same content is instant
- Results persist across restarts
- No external dependencies (SQLite is built into Python)

STATUS: Stub — will be fully implemented in Phase 6.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AnalysisCache:
    """
    SQLite-backed cache for analysis results.

    Usage:
        cache = AnalysisCache()
        cache_id = cache.cache_key("some text")
        cached = cache.get(cache_id)
        if cached is None:
            result = run_analysis(...)
            cache.put(cache_id, result)
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "analysis_cache.db"
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create the cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    cache_id TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    text_length INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @staticmethod
    def cache_key(text: str) -> str:
        """Generate a cache key from text content using SHA-256."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, cache_id: str) -> dict | None:
        """
        Retrieve a cached analysis result.

        Returns None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT result_json FROM analysis_cache WHERE cache_id = ?",
                (cache_id,),
            ).fetchone()

        if row is None:
            return None

        return json.loads(row[0])

    def put(self, cache_id: str, result: dict, text_length: int = 0) -> None:
        """Store an analysis result in the cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_cache (cache_id, result_json, text_length)
                VALUES (?, ?, ?)
                """,
                (cache_id, json.dumps(result), text_length),
            )
            conn.commit()

    def exists(self, cache_id: str) -> bool:
        """Check if a cache entry exists."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM analysis_cache WHERE cache_id = ?",
                (cache_id,),
            ).fetchone()
        return row is not None

    def clear(self) -> int:
        """Clear all cached results. Returns the number of entries deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM analysis_cache")
            conn.commit()
            return cursor.rowcount
