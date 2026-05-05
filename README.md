# 🔍 Cognitive Bias Spotter

An end-to-end pipeline for detecting persuasion techniques and cognitive biases across **News, Social Media, and Political Rhetoric**.

Unlike simple GPT-wrappers, this system uses a fine-tuned **ModernBERT-large** model (8k context) to identify 14 specific manipulation techniques defined by the SemEval-2020/2023 research tasks.

### 🚀 Key Features
- **ModernBERT-large Architecture:** Optimized for long-range dependencies (detects repetition and whataboutism across 8,000 tokens).
- **Universal Scope:** Trained on a unified corpus of news, Reddit (GoEmotions), and political speeches.
- **3-Layer Pipeline:** High-speed Lexicon Matcher → Fine-tuned ML Classifier → LLM Reasoning (for edge cases).
- **Research-Grade:** Implements Focal Loss to handle class imbalance (common vs. rare techniques).

## Architecture — 3 Layers

```
Input Text (up to 8k tokens)
    ↓
Layer 1: Lexicon Matching (~50ms, $0)
    • Curated regex patterns for obvious manipulation
    ↓
Layer 2: ML Classifier (ModernBERT-large, ~200-500ms, $0)
    • Trained on News + Social + Politics unified corpus
    • Span identification + technique classification
    ↓
Layer 3: LLM Reasoning (selective, low cost)
    • Only invoked for low-confidence predictions
    • Catches: false dichotomy, motte-and-bailey, hidden premises
    ↓
Aggregator + Scorer
    • Merge, deduplicate, score, rank
```

## Techniques Detected (14 SemEval Categories + Extras)

| Category | Techniques |
|----------|-----------|
| Emotional | Loaded Language, Appeal to Fear, Name Calling |
| Social | Bandwagon, Appeal to Authority, Social Proof |
| Logical | Black-and-White Fallacy, Causal Oversimplification, Whataboutism |
| Framing | Exaggeration/Minimization, Doubt, Flag-Waving |
| Rhetorical | Slogans, Repetition, Thought-Terminating Clichés |
| Universal | False Scarcity, FOMO, Identity Bait |

## Project Structure

```
/Bias_Spotter
  /backend
    /app
      main.py                         # FastAPI app entry
      /pipeline
        layer1_lexicon.py             # Regex/lexicon matching
        layer2_classifier.py          # Fine-tuned transformer classifier
        layer3_llm.py                 # Selective LLM reasoning
        aggregator.py                 # Combines layer outputs
      /lexicon
        techniques.yaml               # Curated regex patterns
      /scoring
        density.py                    # Manipulation density calculation
        percentile.py                 # Percentile ranking vs. reference corpus
      /parser
        document_parser.py            # PDF/DOCX/TXT/MD handling
      /cache
        cache.py                      # SQLite caching by content hash
    /tests
      test_smoke.py                   # Import smoke tests
      test_layer1.py                  # Lexicon matching tests
      test_aggregator.py             # Aggregator tests
    pyproject.toml                    # Python dependencies (uv)
    README.md                         # Backend-specific README
    TAXONOMY.md                       # The 14 SemEval techniques documented
  /training
    /data                             # Downloaded datasets (gitignored)
    /notebooks
      data_exploration.ipynb          # Dataset exploration
    /scripts
      download_datasets.py            # Automated dataset download
      train_technique_classifier.py   # Phase 2-3 training
      train_span_identifier.py        # Span identification training
    /models                           # Trained checkpoints (gitignored)
  /frontend                           # Web UI (Phase 9)
    index.html                        # Minimal paste-text MVP
```

## Quick Start

```bash
# Create and activate virtual environment
cd backend
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements-dev.txt

# Run the API
uvicorn app.main:app --reload

# Run tests
pytest tests/
```

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data & Setup (Universal Corpus) | 🔧 In Progress |
| 2 | Baseline Classifier (ModernBERT-base) | ⏳ |
| 3 | Production Model (ModernBERT-large) | ⏳ |
| 4 | Lexicon (Layer 1) | ✅ |
| 5 | LLM Layer (Layer 3) | ⏳ |
| 6 | Aggregator & Scoring | ✅ |
| 7 | Document Parser (TXT/MD) | ✅ |
| 8 | Backend API (Stubs) | ✅ |
| 9 | Frontend (Standalone App) | ⏳ |
| 10 | Browser Extension | ⏳ |

## Key Design Decisions

- **Real ML model, not a GPT wrapper.** The fine-tuned classifier is the hero.
- **Local-first.** Most analysis runs on the user's machine after model download.
- **LLM is selective.** Only invoked for genuinely hard reasoning tasks.
- **The lexicon + model do 90% of the work. The LLM does 10%.**

## Datasets

- [SemEval-2020 Task 11](https://zenodo.org/records/3952415) — 14 propaganda techniques
- [SemEval-2023 Task 3](https://propaganda.math.unipd.it/semeval2023task3/) — Multilingual extension
- [GUS-Net](https://github.com/Ethical-Spectacle/fairly) — 69k+ bias annotations
- [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) — Political fact-checking
- [Hyperpartisan News](https://zenodo.org/records/1489920) — Political bias
- [BABE](https://github.com/Media-Bias-Group/MBIC) — Expert media bias annotations
