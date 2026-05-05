"""
Document Parser — Handles PDF, DOCX, TXT, MD, and HTML input.

Extracts text while preserving structure (headings, paragraphs, page numbers).
Each parsed segment includes character offsets that map back to the original
document position, enabling accurate annotation rendering.

STATUS: Stub — will be fully implemented in Phase 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import BinaryIO


class DocumentFormat(Enum):
    """Supported document formats."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    HTML = "html"


@dataclass
class ParsedSegment:
    """A segment of parsed text with structural metadata."""

    text: str
    offset: int  # Character offset in the full extracted text
    segment_type: str  # "heading", "paragraph", "list_item", etc.
    page_number: int | None = None  # For PDFs
    heading_level: int | None = None  # For headings (1-6)


@dataclass
class ParsedDocument:
    """The result of parsing a document."""

    full_text: str
    segments: list[ParsedSegment] = field(default_factory=list)
    format: DocumentFormat = DocumentFormat.TXT
    filename: str = ""
    page_count: int | None = None
    metadata: dict = field(default_factory=dict)


class DocumentParser:
    """
    Multi-format document parser.

    Usage:
        parser = DocumentParser()
        result = parser.parse_file(Path("article.pdf"))
        # or
        result = parser.parse_bytes(file_bytes, "article.pdf")
    """

    # Map file extensions to formats
    FORMAT_MAP: dict[str, DocumentFormat] = {
        ".pdf": DocumentFormat.PDF,
        ".docx": DocumentFormat.DOCX,
        ".txt": DocumentFormat.TXT,
        ".md": DocumentFormat.MD,
        ".html": DocumentFormat.HTML,
        ".htm": DocumentFormat.HTML,
    }

    def detect_format(self, filename: str) -> DocumentFormat:
        """Detect document format from filename extension."""
        ext = Path(filename).suffix.lower()
        fmt = self.FORMAT_MAP.get(ext)
        if fmt is None:
            raise ValueError(f"Unsupported file format: {ext}")
        return fmt

    def parse_file(self, file_path: Path | str) -> ParsedDocument:
        """
        Parse a document from a file path.

        TODO (Phase 7): Implement format-specific handlers.
        """
        path = Path(file_path)
        fmt = self.detect_format(path.name)

        if fmt == DocumentFormat.TXT or fmt == DocumentFormat.MD:
            return self._parse_text(path.read_text(encoding="utf-8"), path.name, fmt)

        # PDF, DOCX, HTML stubs
        raise NotImplementedError(f"Parser for {fmt.value} not yet implemented (Phase 7).")

    def parse_bytes(self, data: bytes, filename: str) -> ParsedDocument:
        """
        Parse a document from raw bytes (e.g., from file upload).

        TODO (Phase 7): Implement format-specific handlers.
        """
        fmt = self.detect_format(filename)

        if fmt == DocumentFormat.TXT or fmt == DocumentFormat.MD:
            text = data.decode("utf-8")
            return self._parse_text(text, filename, fmt)

        raise NotImplementedError(f"Parser for {fmt.value} not yet implemented (Phase 7).")

    def _parse_text(self, text: str, filename: str, fmt: DocumentFormat) -> ParsedDocument:
        """Parse plain text / markdown into segments."""
        segments: list[ParsedSegment] = []
        offset = 0

        for line in text.split("\n\n"):
            line = line.strip()
            if not line:
                continue

            seg_type = "paragraph"
            heading_level = None

            # Basic markdown heading detection
            if fmt == DocumentFormat.MD and line.startswith("#"):
                seg_type = "heading"
                heading_level = len(line) - len(line.lstrip("#"))
                heading_level = min(heading_level, 6)

            segments.append(
                ParsedSegment(
                    text=line,
                    offset=offset,
                    segment_type=seg_type,
                    heading_level=heading_level,
                )
            )
            offset += len(line) + 2  # +2 for the double newline

        return ParsedDocument(
            full_text=text,
            segments=segments,
            format=fmt,
            filename=filename,
        )
