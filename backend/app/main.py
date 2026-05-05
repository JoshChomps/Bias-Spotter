"""
Cognitive Bias Spotter — FastAPI Application Entry Point.

Exposes the analysis pipeline as a REST API with endpoints for:
- Paste-text analysis
- Document upload analysis
- Cached result retrieval
- Reference corpus info
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Cognitive Bias Spotter",
    description="Detect persuasion techniques in text using a 3-layer analysis pipeline.",
    version="0.1.0",
)

# CORS — allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────────────


class TextAnalysisRequest(BaseModel):
    """Request body for paste-text analysis."""

    text: str = Field(..., min_length=1, description="The text to analyze for persuasion techniques.")
    deep_analysis: bool = Field(
        default=False,
        description="If True, always invoke LLM layer regardless of classifier confidence.",
    )


class DetectedTechnique(BaseModel):
    """A single detected persuasion technique with span information."""

    start: int
    end: int
    text_span: str
    technique: str
    category: str
    confidence: float
    source_layer: str  # "lexicon", "classifier", or "llm"
    explanation: str


class CategoryBreakdown(BaseModel):
    """Breakdown of detected techniques by meta-category."""

    emotional: int = 0
    social: int = 0
    logical: int = 0
    framing: int = 0
    rhetorical: int = 0


class AnalysisResponse(BaseModel):
    """Full analysis result for a piece of text."""

    cache_id: str
    text_length: int
    techniques: list[DetectedTechnique]
    density_score: float = Field(
        ..., ge=0, le=100, description="Manipulation density score (0-100)."
    )
    category_breakdown: CategoryBreakdown
    percentile_rank: float | None = Field(
        None, description="Percentile rank against reference corpus (0-100)."
    )
    word_count: int
    techniques_per_100_words: float


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest) -> AnalysisResponse:
    """
    Analyze pasted text for persuasion techniques.

    Runs the full 3-layer pipeline:
    1. Lexicon matching (instant)
    2. Fine-tuned classifier (fast, local)
    3. LLM reasoning (selective, only if needed)
    """
    # TODO: Wire up the actual pipeline (Phase 6+)
    # For now, return a stub response so the API shape is testable
    raise HTTPException(
        status_code=501,
        detail="Analysis pipeline not yet implemented. Complete Phases 2-6 first.",
    )


@app.post("/analyze/document", response_model=AnalysisResponse)
async def analyze_document(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Upload a document (PDF, DOCX, TXT, MD) and analyze for persuasion techniques.

    The document is parsed, then the text is run through the full analysis pipeline.
    """
    # TODO: Wire up document parser + pipeline (Phase 7+)
    raise HTTPException(
        status_code=501,
        detail="Document analysis not yet implemented. Complete Phases 2-7 first.",
    )


@app.get("/analyze/{cache_id}", response_model=AnalysisResponse)
async def get_cached_analysis(cache_id: str) -> AnalysisResponse:
    """Retrieve a previously cached analysis result by its content hash."""
    # TODO: Wire up SQLite cache retrieval
    raise HTTPException(
        status_code=501,
        detail="Cache retrieval not yet implemented.",
    )


@app.get("/reference-corpus/info")
async def reference_corpus_info() -> dict:
    """Return statistics about the reference corpus used for percentile ranking."""
    # TODO: Populate with actual corpus stats (Phase 6)
    return {
        "status": "not_built",
        "target_size": "100-500 documents",
        "genres": [
            "encyclopedia",
            "news",
            "opinion",
            "advertising",
            "political_speeches",
            "propaganda",
        ],
        "note": "Reference corpus will be built in Phase 6.",
    }
