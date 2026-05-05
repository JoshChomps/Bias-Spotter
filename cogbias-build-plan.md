# Cognitive Bias Spotter — Complete Build Plan

A persuasion technique detection system built around a fine-tuned BERT/RoBERTa classifier, not an LLM wrapper. Designed to be a standalone app and document reader first, with a browser extension as an optional later layer.

---

## Project Vision

**Core product:** A desktop/web application where users paste or upload text (article, speech, PDF, document) and get a fully annotated analysis showing every persuasion technique used, with explanations, density scores, and a percentile ranking against a reference corpus.

**Always-available modes:**
1. **Paste-text analysis** — quick textarea input for one-off analysis
2. **Document reader** — upload PDFs, .docx, .txt, .md files; analyze in-context with the original document layout preserved
3. **Browser extension** (later) — analyze any webpage in real time

The document reader is a first-class feature, not an afterthought. Users should be able to upload a long-form document, have it analyzed in full, and read it with annotations inline.

**Design philosophy:**
- Real ML model, not a GPT wrapper. The fine-tuned classifier is the hero.
- Local-first. Most analysis runs on the user's machine after model is downloaded.
- LLM is only invoked for genuinely hard reasoning tasks the classifier can't handle.
- The lexicon and the model do 90% of the work. The LLM does 10%.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Input                                                 │
│  • Pasted text                                              │
│  • Uploaded document (PDF/DOCX/TXT/MD)                      │
│  • (Later) Webpage via browser extension                    │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Document Parser                                            │
│  • Extract text + preserve structure (headings, paragraphs) │
│  • Tokenize into sentences with offsets                     │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Lexicon Matching (~50ms, $0)                      │
│  • Curated regex patterns for obvious manipulation          │
│  • Catches scarcity, urgency, social proof, authority, etc. │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: FINE-TUNED CLASSIFIER (the hero, ~100-300ms, $0)  │
│  • Span identification (BIO tagging via token classifier)   │
│  • Technique classification (14 SemEval categories)         │
│  • Trained on SemEval-2020 Task 11 + GUS-Net                │
│  • Runs locally via ONNX or transformers in production      │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: LLM Reasoning (selective, ~2-5s, low cost)        │
│  • ONLY invoked for low-confidence classifier predictions   │
│  • Catches: motte-and-bailey, false dichotomy, hidden       │
│    premises, intent disambiguation                          │
│  • Hugging Face free API → Ollama local → paid fallback     │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Aggregator + Scorer                                        │
│  • Merge layer outputs, deduplicate overlapping spans       │
│  • Compute manipulation density score                       │
│  • Compare against reference corpus → percentile ranking    │
│  • Cache result by content hash (SQLite)                    │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Frontend                                                   │
│  • Annotated text view with hover explanations              │
│  • Density meter + category breakdown                       │
│  • Document reader with original layout preserved           │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data & Setup

**Goal:** Have working data and a trainable Python environment.

**Steps:**
1. Create a fresh repo: `cogbias-spotter`. Initialize with git
2. Set up Python 3.11+ environment with `uv` (faster than poetry)
3. Install core dependencies: `transformers`, `datasets`, `torch`, `scikit-learn`, `fastapi`, `uvicorn`, `spacy`, `pypdf`, `python-docx`
4. Download **SemEval-2020 Task 11** dataset from Zenodo: https://zenodo.org/records/3952415
5. Download **GUS-Net** dataset from its repo (https://github.com/Ethical-Spectacle/fairly) for additional bias annotations
6. Explore both datasets in a Jupyter notebook: count samples per category, identify class imbalance, check label distribution
7. Set up Weights & Biases (free tier) or TensorBoard for experiment tracking
8. Decide on training compute: Google Colab free tier (T4 GPU, 12hr sessions) or Kaggle (P100, longer sessions). Both free.

**Deliverable:** A clean repo with data downloaded, dependencies installed, and a notebook showing the dataset structure.

**Time estimate:** 3-4 hours.

---

## Phase 2: Baseline Classifier

**Goal:** Get a working fine-tuning loop end-to-end, even if performance is mediocre.

**Steps:**
1. Convert SemEval data into HuggingFace `datasets` format. Two tasks:
   - **Span identification** (token classification, BIO tags)
   - **Technique classification** (sequence classification, 14 classes)
2. Start with **technique classification** since it's simpler. Span ID comes next
3. Fine-tune `distilbert-base-uncased` on the technique classification task. Small, fast, gets the loop working
4. Use standard training setup: AdamW, learning rate 2e-5, batch size 16, 3 epochs
5. Evaluate on the test set using macro F1 (the SemEval metric). Expect F1 ~0.40-0.50 from this baseline
6. Save the model checkpoint locally
7. Write an `inference.py` that loads the checkpoint and classifies a text fragment

**Deliverable:** A working classifier that takes a text fragment and predicts which of 14 propaganda techniques it uses (or "none").

**Time estimate:** 4-6 hours including debugging.

---

## Phase 3: Better Model + Span Identification

**Goal:** Move from baseline to competitive performance and add span detection.

**Steps:**
1. Switch to `roberta-base`. RoBERTa beat BERT on this task in the original SemEval paper. Same training script, just swap the model name
2. Handle class imbalance — pick ONE approach:
   - Focal loss (drop-in replacement for CrossEntropy)
   - Class weights in the loss function (simpler)
   - Oversampling minority classes (more involved)
   - Recommended: focal loss, it's clean and works well
3. Tune hyperparameters: try learning rates 1e-5, 2e-5, 3e-5; batch sizes 16 and 32; 3 vs 5 epochs
4. Target F1: 0.55-0.62 (competitive with SemEval winners)
5. Now train **span identification** as a separate model. Same RoBERTa base, but with a token classification head. Use BIO tags
6. Save both models. The pipeline will use them in sequence: span model finds where, technique model classifies what
7. Optionally: experiment with a multi-task model that does span + technique simultaneously (harder but cleaner)

**Deliverable:** Two trained models (span detector + technique classifier) with documented F1 scores. Saved as a model card with metrics.

**Time estimate:** 6-10 hours including experiment iterations.

---

## Phase 4: Lexicon (Layer 1)

**Goal:** Build a curated regex/lexicon layer that catches the obvious stuff the model might miss on edge cases or that doesn't need ML at all.

**Steps:**
1. Create `lexicon/techniques.yaml` with patterns organized by category:
   ```yaml
   false_scarcity:
     patterns:
       - "limited time"
       - "only \\d+ left"
       - "while supplies last"
       - "act fast"
     description: "Manufactured urgency to bypass deliberation"
   social_proof:
     patterns:
       - "thousands of (users|customers|people)"
       - "everyone (is|loves|wants)"
       - "(millions|hundreds of thousands) (have|use|trust)"
     description: "Appeals to crowd behavior to legitimize action"
   ```
2. Cover all 14 SemEval categories plus a few extras (urgency, identity bait, fear of missing out)
3. Each pattern is regex-compiled and tagged with a category and explanation
4. Build a `Layer1Matcher` class that runs all patterns over input text and returns matches with offsets
5. Manual curation is critical here. Spend 2-3 hours actually reading ad copy, political speeches, and propaganda to build the lexicon. The quality of these patterns determines a lot of the system's performance

**Deliverable:** A YAML lexicon with 100+ curated patterns and a matcher class that runs them.

**Time estimate:** 3-4 hours of focused work.

---

## Phase 5: LLM Layer (Layer 3)

**Goal:** Add LLM reasoning for the subtle techniques the classifier can't handle, while keeping costs near zero.

**Steps:**
1. Identify which techniques genuinely need LLM reasoning vs. which the classifier handles. Generally:
   - **Classifier handles:** loaded language, name calling, exaggeration, fear appeals, authority appeals, bandwagon, slogans
   - **LLM needed:** false dichotomy, motte-and-bailey, hidden premises, intent disambiguation, missing context
2. Build `Layer3LLM` class with pluggable backends:
   - `HuggingFaceInference` — free tier, default
   - `OllamaLocal` — for unlimited free local inference
   - `AnthropicAPI` — fallback for highest quality
3. Define a strict prompt template that asks the LLM to identify ONLY specific reasoning techniques, returning structured JSON
4. **Selective invocation rule:** Layer 3 is only called when:
   - Classifier confidence is below a threshold (e.g., 0.6)
   - The text contains specific structural cues (e.g., "either X or Y" — possible false dichotomy)
   - User explicitly requests deep analysis
5. Cache LLM responses by passage hash so repeated analysis costs nothing

**Deliverable:** An LLM layer that activates selectively and costs near-zero in normal use.

**Time estimate:** 3-4 hours.

---

## Phase 6: Aggregator & Scoring

**Goal:** Combine all three layers into a coherent analysis result.

**Steps:**
1. Build `Aggregator` class that:
   - Runs all three layers
   - Merges results, deduplicating overlapping spans (when lexicon and classifier both flag the same phrase, prefer the higher-confidence source)
   - Returns a unified list of `(start, end, technique, confidence, source_layer, explanation)` tuples
2. Build `DensityScorer`:
   - Counts techniques per 100 words
   - Categorizes detections (emotional, statistical, framing, social, logical)
   - Returns category breakdown
3. Build `PercentileRanker`:
   - Pre-score 100-500 reference documents across genres (encyclopedia, news, opinion, ads, political speeches, propaganda)
   - Store reference scores in SQLite
   - Compute percentile of new input against this corpus
4. Build SQLite cache layer keyed by `sha256(text)` so re-analysis of the same content is instant

**Deliverable:** A single `analyze(text) -> AnalysisResult` function that produces the full annotated output.

**Time estimate:** 4-5 hours.

---

## Phase 7: Document Parser

**Goal:** Support uploading documents, not just pasted text.

**Steps:**
1. Build `DocumentParser` with format-specific handlers:
   - PDF: use `pypdf` or `pdfplumber` (better text extraction)
   - DOCX: use `python-docx`
   - TXT/MD: native handling
   - HTML: use `beautifulsoup4` (preparation for browser extension)
2. Preserve structure: extract text by paragraph with offsets that map back to the original document position
3. For PDFs, also extract page numbers so the UI can show "page 3, paragraph 2"
4. Handle edge cases: scanned PDFs (out of scope, just warn), encrypted PDFs (warn and skip), corrupt files (graceful failure)

**Deliverable:** Upload any common document format and get back parsed text with structural metadata.

**Time estimate:** 3-4 hours.

---

## Phase 8: Backend API

**Goal:** Expose the analysis pipeline as a REST API.

**Steps:**
1. Build a FastAPI app with these endpoints:
   - `POST /analyze/text` — paste text, get analysis
   - `POST /analyze/document` — upload file, get analysis
   - `GET /analyze/{cache_id}` — retrieve cached analysis
   - `GET /reference-corpus/info` — show corpus stats
2. Async endpoints — long PDFs should not block. Use FastAPI background tasks or Celery for very long docs
3. Streaming response for live-update experience (analysis fills in as each section completes)
4. Rate limiting (in case you ever expose it publicly)
5. Standard OpenAPI docs auto-generated by FastAPI

**Deliverable:** Self-hosted API that handles both paste-text and document analysis.

**Time estimate:** 3-4 hours.

---

## Phase 9: Frontend (Standalone App)

**Goal:** A clean, polished standalone app — desktop or web — that's a first-class experience.

**Steps:**
1. Choose stack:
   - **Recommended:** Next.js + Tailwind for a web app deployable anywhere
   - Alternative: Tauri + React for a true desktop app (smaller install than Electron, Rust-based)
2. Build three primary views:
   - **Paste view:** large textarea + "Analyze" button + result panel
   - **Document view:** upload zone + parsed document + inline annotations + side panel with summary
   - **Settings view:** model selection, LLM backend choice, lexicon customization
3. Annotation rendering: use a library like `react-mark-text` or roll your own. Each detected span gets a colored background by category, hover for explanation
4. Density meter: prominent visual showing the percentile ranking. Make it feel like a sleek dashboard
5. Category breakdown: small radar chart or stacked bar showing emotional vs structural vs logical manipulation
6. Export: let users export the annotated document as PDF or HTML

**Deliverable:** A polished standalone app where you can paste text, upload a document, or read an annotated PDF.

**Time estimate:** 8-12 hours.

---

## Phase 10: Browser Extension (Optional, After Core Works)

**Goal:** Wrap the analysis pipeline in a browser extension that analyzes any webpage.

**Steps:**
1. Use Plasmo framework (https://plasmo.com) — much easier than vanilla extension dev
2. Content script extracts visible text from the page
3. Send to your local backend (or a hosted version) for analysis
4. Inject overlays/highlights directly into the page DOM
5. Toggle button in the toolbar to enable/disable
6. Settings page to choose which categories to highlight

**Deliverable:** Chrome/Firefox extension that adds bias spotting to any webpage.

**Time estimate:** 6-10 hours.

---

## Phase 11: Polish & Differentiators

**Goal:** Add the features that make this stand out from generic "AI text analysis" tools.

**Optional features, pick what matters most:**
- **Side-by-side rewrites:** for each detected manipulation, generate a neutral version (LLM-driven). Show the contrast.
- **Personalized blind-spot quiz:** users take a 5-minute quiz, app prioritizes highlighting techniques they're most vulnerable to.
- **Longitudinal tracking:** ingest a writer/publication's archive, show how their manipulation density has changed over time.
- **Comparison mode:** drop two documents in, see them side-by-side with comparative metrics.
- **Annotation export:** export annotated documents as PDFs for sharing or research.
- **API for researchers:** open up the API for academic use; this is a real research tool.

---

## Honest Reality Check

**This is a 2-3 week project minimum, not a weekend.** Phases 1-3 alone (training the model) typically take longer than people expect on their first fine-tuning project. Budget accordingly.

**Performance expectations:**
- Technique classification F1: 0.55-0.65 is competitive
- Span identification F1: 0.40-0.50 is competitive
- Most users won't care about exact F1 — they'll care about whether the highlights feel right

**Cost in production:**
- Layer 1 + Layer 2: $0 (runs locally after one-time training)
- Layer 3: ~$0.001-0.01 per document analyzed if using API; $0 if using local Ollama
- Hosting: $5-20/month if you self-host the model on a small VPS

**What will actually be hardest:**
1. Class imbalance in SemEval data — some techniques are very rare
2. Building a good reference corpus for percentile ranking
3. Frontend UX for annotation rendering (gets complex with overlapping spans)
4. The lexicon curation work that everyone wants to skip

**What will be easier than expected:**
1. Fine-tuning itself — modern transformers + HuggingFace makes this routine
2. FastAPI backend — quick to build, well-documented
3. Document parsing — libraries handle most cases well

---

## Claude Code Kickoff Prompt

Copy this prompt into a fresh Claude Code session to begin Phase 1.

```
I'm building a project called "Cognitive Bias Spotter" — a tool that analyzes persuasive 
text (news articles, political speeches, ad copy, opinion pieces, uploaded documents) and 
highlights every persuasion technique used. It produces a manipulation density score, 
color-coded annotations, and a percentile ranking against a reference corpus.

KEY ARCHITECTURAL DECISION: This is a real ML project, not a GPT wrapper. The hero is a 
fine-tuned BERT/RoBERTa classifier trained on the SemEval-2020 Task 11 dataset. LLM calls 
are only used selectively for subtle reasoning the classifier can't handle. The lexicon 
and the trained model do 90% of the work.

PRODUCT FORMS:
- Standalone app first (paste-text + document reader for PDFs/DOCX/TXT/MD)
- Browser extension comes later as an optional layer
- Both should always be available

GOAL FOR THIS SESSION (Phase 1):
Set up the project skeleton and prepare the training data. We are NOT training yet this 
session — just getting the data and environment ready.

TASKS FOR THIS SESSION:
1. Initialize a fresh git repo called "cogbias-spotter"
2. Set up Python 3.11+ environment using uv (faster than poetry)
3. Install dependencies: transformers, datasets, torch, scikit-learn, fastapi, uvicorn, 
   spacy, pypdf, python-docx, pyyaml, pytest
4. Download SemEval-2020 Task 11 dataset from https://zenodo.org/records/3952415 — both 
   the Span Identification (SI) and Technique Classification (TC) subtasks
5. Download the GUS-Net dataset from https://github.com/Ethical-Spectacle/fairly for 
   additional bias annotations
6. Create a Jupyter notebook (data_exploration.ipynb) that loads both datasets, shows 
   their structure, counts samples per technique category, and visualizes class imbalance
7. Document the SemEval taxonomy (14 propaganda techniques) in a TAXONOMY.md file — these 
   are: Loaded Language, Name Calling/Labeling, Repetition, Exaggeration/Minimization, 
   Doubt, Appeal to Fear/Prejudice, Flag-Waving, Causal Oversimplification, Slogans, 
   Appeal to Authority, Black-and-White Fallacy, Thought-Terminating Cliches, 
   Bandwagon/Reductio ad Hitlerum, Whataboutism/Straw Men/Red Herring
8. Set up the project structure (see below)

PROJECT STRUCTURE:
/cogbias-spotter
  /backend
    /app
      main.py                    # FastAPI app entry (skeleton only this session)
      /pipeline
        layer1_lexicon.py        # stub for now
        layer2_classifier.py     # stub for now
        layer3_llm.py            # stub for now
        aggregator.py            # stub for now
      /lexicon
        techniques.yaml          # empty starter file
      /scoring
        density.py               # stub
        percentile.py            # stub
      /parser
        document_parser.py       # stub for PDF/DOCX/TXT/MD handling
      /cache
        cache.py                 # stub for SQLite caching
    /tests
      test_smoke.py              # confirms imports work
    pyproject.toml               # uv-managed
    README.md                    # project overview
    TAXONOMY.md                  # the 14 techniques documented
  /training
    /data                        # downloaded datasets (gitignored)
    /notebooks
      data_exploration.ipynb     # this session's deliverable
    /scripts
      download_datasets.py       # automated download
      train_technique_classifier.py  # stub for Phase 2
      train_span_identifier.py       # stub for Phase 3
    /models                      # trained model checkpoints (gitignored)
  /frontend                      # empty for now, scaffolded later
  .gitignore                     # ignore data/, models/, __pycache__, .env
  README.md                      # top-level project README

QUESTIONS BEFORE YOU START:
Please ask me these before doing anything:

1. Am I using a Mac, Linux, or Windows machine? (affects uv/path setup)
2. Do I have a GPU available locally, or will training happen on Colab/Kaggle?
3. Do I want to use Weights & Biases for experiment tracking? It's free and useful but 
   requires signup. Alternative is local TensorBoard.
4. Should the document parser support OCR for scanned PDFs in v1, or is that out of scope?
5. Any preferences on the eventual frontend stack? Default plan is Next.js + Tailwind for 
   the web version.

CONSTRAINTS:
- Don't burn tokens being verbose — get to working code fast
- Use uv for Python deps (faster than poetry, modern standard)
- Commit frequently with clear messages
- Add type hints throughout the Python code
- Write a one-page README explaining the project to someone new
- This session is ONLY data setup and project scaffolding. Do NOT start training yet — 
  that's Phase 2 in a separate session

REFERENCE MATERIAL:
- SemEval-2020 Task 11 paper: https://aclanthology.org/2020.semeval-1.186.pdf
- Dataset source: https://zenodo.org/records/3952415
- GUS-Net repo: https://github.com/Ethical-Spectacle/fairly
- Pre-trained propaganda detection models exist on Hugging Face — may be useful as 
  starting checkpoints later

After Phase 1 is complete, future sessions will be:
- Phase 2: Train baseline DistilBERT technique classifier
- Phase 3: Train production RoBERTa span identifier + technique classifier
- Phase 4: Build the lexicon
- Phase 5: Add LLM layer (selective invocation)
- Phase 6: Aggregator + scoring
- Phase 7: Document parser
- Phase 8: FastAPI backend
- Phase 9: Frontend (standalone app)
- Phase 10: Browser extension (later)

Let's go.
```

---

## Subsequent Phase Prompts (Save These)

You'll need a fresh prompt at the start of each phase since Claude Code sessions don't persist context perfectly. Brief versions:

**Phase 2 prompt seed:** "We're at Phase 2 of the cogbias-spotter project. Read README.md and TAXONOMY.md to refresh context. Train a baseline DistilBERT classifier on the SemEval-2020 Task 11 Technique Classification subtask. Goal: working end-to-end training loop with F1 ~0.40-0.50 baseline. Save checkpoint to /training/models/distilbert-baseline/. Write a model card with metrics."

**Phase 3 prompt seed:** "Phase 3 of cogbias-spotter. Read README.md. Switch from DistilBERT to roberta-base. Add focal loss for class imbalance. Tune learning rate and batch size. Target F1 0.55-0.65. Then train a separate token classification model for span identification using BIO tags. Save both to /training/models/. Document results in MODELS.md."

**Phase 4 prompt seed:** "Phase 4 of cogbias-spotter. Read TAXONOMY.md. Build the Layer 1 lexicon in /backend/app/lexicon/techniques.yaml. Need 100+ curated regex patterns covering all 14 SemEval techniques plus extras (urgency, identity bait, FOMO). Each pattern needs: regex, category, explanation. Write a Layer1Matcher class in layer1_lexicon.py that loads the YAML and returns matches with offsets. Write tests."

(And so on for each phase — keep the prompts narrow to one phase at a time so the agent doesn't drift.)

---

# Critical Analysis: Why SemEval-2020 Hit ~0.62 and How to Beat It in 2026

This section is the most important part of this document if your goal is to build something genuinely better than the 2020 state of the art. The SemEval-2020 Task 11 winners hit a macro F1 of ~0.62 on Technique Classification. The 2024-2025 state-of-the-art has pushed past 0.66. Your project, if built carefully with what's available now, can plausibly target **F1 of 0.70+** — meaningfully ahead of where research was in 2020 and competitive with current academic work.

This is not bragging. The reasons F1 was capped at ~0.62 are well-understood, and most of them have practical solutions that didn't exist or weren't standard practice in 2020.

## 1. Why SemEval-2020 Capped at ~0.62

### Flaw 1: Severe class imbalance, weakly addressed

The SemEval dataset has 14 classes with extreme imbalance. "Loaded Language" had thousands of samples; "Bandwagon, Reductio ad Hitlerum" had under 100. Standard cross-entropy loss treats all classes equally during training, so the model effectively learns to ignore rare classes — they don't move the loss enough to matter. Most 2020 teams handled this with basic techniques: oversampling, undersampling, simple class weights. The Inno team's paper acknowledges they used undersampling and cost-sensitive learning, and still hit only 0.58 F1.

The fundamental problem: with ~100 training samples for some classes, the model has nothing to learn from. Data scarcity beats clever loss functions.

### Flaw 2: Pre-trained models were weaker

In 2020, the available pretrained transformers were BERT-base (110M params, 2018), RoBERTa-base (125M params, 2019), and a few others. These models were pre-trained on smaller corpora than what's standard now, with simpler training objectives. BERT was trained on Wikipedia + BookCorpus (~16GB). Modern models train on text corpora that are 100-1000x larger.

The deeper issue: BERT-base and RoBERTa-base have limited reasoning capacity. They're good at surface patterns (loaded vocabulary, syntactic markers) but weak at the semantic/argumentative reasoning required for techniques like "Causal Oversimplification" or "Whataboutism."

### Flaw 3: Single-task training with no multi-task signal

Most 2020 teams trained a single classifier on the SemEval data alone. They didn't combine it with related tasks (sentiment, stance, hate speech, claim detection) that share representational structure. This left a lot of free signal on the table — a model that also learns "is this claim emotionally charged" will be better at "is this loaded language" because the underlying features overlap.

### Flaw 4: Limited context window usage

Many 2020 systems used truncated 120-token inputs (the Abdullah et al. RoBERTa paper that became state-of-the-art used 120 tokens). 120 tokens is barely enough to fit one sentence with surrounding context. Propaganda techniques are often only identifiable through context — "doubt" requires knowing what claim is being doubted; "whataboutism" requires knowing the original claim being deflected from. Truncating context destroys this signal.

### Flaw 5: No knowledge transfer from related datasets

Each team trained from scratch on SemEval-2020 alone. Datasets like the FLUTE corpus (figurative language), the LIAR dataset (political fact-checking), or the MultiFC dataset (claim verification) all contain related signal. None were used. In 2020, large-scale multi-task learning was uncommon practice.

### Flaw 6: Annotation noise and label ambiguity

Independent of model quality, there's a hard ceiling imposed by data quality. Multiple academic papers since SemEval-2020 have noted that **inter-annotator agreement on propaganda technique classification is itself only ~0.6-0.7**. Even human experts disagree about which technique a span uses — many spans use multiple techniques simultaneously, and the SemEval annotation forced single labels in early versions. If humans only agree at 0.65 F1, no model can systematically beat that on this exact dataset without cheating.

### Flaw 7: The taxonomy itself is messy

The 14 categories overlap. "Name Calling" and "Loaded Language" both involve emotionally charged words. "Straw Men," "Whataboutism," and "Red Herring" were collapsed into one super-category by the SemEval organizers because they couldn't reliably distinguish them. This creates fundamental confusion in the training signal.

## 2. What Has Changed Since 2020

A non-exhaustive list of what's improved, in roughly descending order of impact:

### 2.1 Models are dramatically more capable

**DeBERTa-v3** (Microsoft, 2021) consistently outperforms RoBERTa on classification benchmarks by 2-5 F1 points with the same training data. Disentangled attention and ELECTRA-style pre-training make it a drop-in upgrade. **Free, available on Hugging Face, same training code.**

**Larger pretrained encoders** like DeBERTa-v3-large (435M params) and ModernBERT (Dec 2024, 395M params) push further. ModernBERT has a 8192-token context window vs BERT's 512, directly fixing the truncation problem.

**Frontier LLMs** (GPT-4, Claude 3.5 Sonnet, Llama 3.1 70B) can do this task zero-shot at competitive levels, though research consistently shows fine-tuned smaller models still beat them on macro F1. Recent papers (2023-2024) found GPT-4 and Claude 3 Opus *did not* outperform fine-tuned BERT-based models on the SemEval task in zero-shot settings — fine-tuning still wins.

### 2.2 Training techniques are better

**Focal loss with proper gamma tuning** is now standard and gives ~2-3 F1 improvement on imbalanced classification with no other changes.

**Knowledge distillation from LLMs** is the technique that has changed the field most since 2023. Instead of (or in addition to) training only on the 6,000 SemEval examples, you can use a strong LLM to generate thousands of synthetic propaganda examples with technique labels, then train your smaller model on that augmented dataset. Recent papers (Lin et al. 2024, Lu et al. 2024) show this approach gives 5-8 F1 point improvements on related tasks. The 2025 hybrid annotation paper specifically demonstrates this on propaganda detection.

**Contrastive learning** as an auxiliary objective during fine-tuning. Pull representations of same-technique spans closer; push different-technique spans apart. Especially helpful for confusable categories.

**Multi-task learning** with related datasets — train the same encoder on SemEval propaganda, GUS-Net bias, FLUTE figurative language, and LIAR fact-checking simultaneously. The shared representation transfers, and the rare classes benefit from out-of-distribution signal.

### 2.3 More and better data

**SemEval-2023 Task 3** extended the propaganda detection task to 9 languages and refined the taxonomy. **SemEval-2024 Task 4** extended to memes and multimodal contexts. **PropaNews** dataset adds large-scale automatically-generated propaganda. Beyond direct propaganda data, related corpora that didn't exist or weren't widely used in 2020:

- **GUS-Net** (2024) — 69k+ bias annotations
- **HQP** (Maarouf et al. 2023) — high-quality propaganda annotations
- **PEACE** dataset for persuasion strategy detection
- **Hyperpartisan News Corpus** for biased writing
- The 2025 hybrid LLM-human annotation pipelines that generate cleaner labels

### 2.4 Hardware is cheaper and more accessible

Free Colab/Kaggle GPUs (T4, P100) in 2020 would let you fine-tune BERT-base in reasonable time. Now, free tiers handle DeBERTa-large fine-tuning. Beyond free tiers, $1/hour A100 access on platforms like RunPod and Lambda Labs makes serious training accessible.

### 2.5 Better tooling

The **transformers library**, **datasets library**, and **PEFT/LoRA** have matured. Training with parameter-efficient techniques means you can fine-tune larger models on less hardware. **Optuna** for automated hyperparameter search is now trivial to integrate. **Weights & Biases** for experiment tracking is standard practice.

## 3. The Plan to Beat 0.62

The combination of improvements above stacks. Each individually adds 1-3 F1 points; combined, they should comfortably push past 0.66 (current academic state-of-the-art) and toward 0.70-0.72 with care. The realistic ceiling is bounded by inter-annotator agreement (~0.65-0.70 on the original dataset), but with cleaner labels you can push beyond.

Here is the modified strategy, in order of expected impact per unit effort:

### 3.1 Switch the base model: RoBERTa → DeBERTa-v3 (essentially free win)

Replace `roberta-base` in Phase 3 with `microsoft/deberta-v3-base` (or `deberta-v3-large` if compute allows). Same training code. Expected gain: +2-4 F1 points.

If using a larger context: try **ModernBERT-large** for the 8192-token window. Expected gain: +1-2 additional F1 points specifically on context-dependent techniques (Whataboutism, Doubt).

### 3.2 Add LLM-generated synthetic training data (large gain, modest effort)

This is the highest-impact technique that wasn't standard in 2020. Steps:

1. Use Claude 3.5 Sonnet, GPT-4, or Llama 3.1-70B to generate ~5,000 synthetic propaganda examples covering rare techniques. Prompt the LLM with the SemEval taxonomy and ask it to generate diverse examples per technique, with explanations.
2. Filter the synthetic data: have the LLM verify its own labels on a separate pass; keep only examples with high confidence.
3. Combine synthetic + real SemEval data for training. The synthetic data fills in rare-class gaps; the real data anchors the model.
4. Critical step: hold out the original SemEval test set as the only ground truth for evaluation. Don't train or validate on it.

Expected gain: +3-6 F1 points, concentrated heavily in rare-class performance. This is the single most important improvement in this plan.

### 3.3 Multi-task training with related datasets

Train on SemEval propaganda + GUS-Net bias + LIAR fact-checking simultaneously, with task-specific output heads but shared encoder. The shared representation learns more general persuasion features.

Expected gain: +1-3 F1 points, mostly through better generalization on borderline cases.

### 3.4 Better loss function: Focal loss with hierarchical weighting

Don't just use CE or even simple focal loss. Implement **class-balanced focal loss** (Cui et al. 2019) which weights by effective number of samples per class, not raw count. For the most extreme rare classes, also add explicit oversampling at the data loader level.

Expected gain: +1-2 F1 points on rare classes specifically.

### 3.5 Contrastive auxiliary objective

During training, add a contrastive loss term: pull embeddings of same-technique spans together; push different-technique spans apart. Use a margin of 0.5 and weight 0.1 against the main classification loss.

Expected gain: +1 F1 point, especially on confusable categories (Loaded Language vs Name Calling).

### 3.6 Better context handling

For span identification, use a sliding window approach with overlap, then aggregate predictions. Don't truncate — process the full document. For technique classification, include the full surrounding paragraph as context, not just the target span.

Expected gain: +1-2 F1 points on context-dependent techniques.

### 3.7 Ensemble at inference time

Train 3-5 versions of the model with different random seeds, different hyperparameters, or different base models (DeBERTa, RoBERTa, ModernBERT). Average predictions at inference.

Expected gain: +0.5-1 F1 points consistently.

### 3.8 LLM verification of low-confidence predictions

When the classifier's top prediction has confidence below ~0.6, send the span to an LLM (Claude/GPT) for verification. Use the LLM's prediction as the final answer for these cases. This is your Layer 3 from the architecture, but specifically targeted at boundary cases.

Expected gain: +1-2 F1 points on the hardest cases without LLM cost on the easy ones.

### Expected total gain

Stacking all the above: the 2020 baseline of 0.62 → projected 0.68-0.72 with this approach. The exact number depends on careful implementation and how much you can stack without diminishing returns. **The realistic ceiling is the inter-annotator agreement of the dataset itself (~0.70), so without re-annotating data, this is roughly the best you can hope for.**

### What to do beyond the F1 ceiling

If you want to genuinely *exceed* the inter-annotator ceiling, you need cleaner data. Two paths:

1. **Re-annotate a subset.** Take ~500 SemEval examples, have you and 1-2 friends independently label them with current understanding, resolve disagreements through discussion, and use this as a higher-quality test set. Train on the noisy SemEval but evaluate on the clean subset. This is what you'd do for a research paper.

2. **Use the LLM-generated annotation pipeline from the 2025 hybrid paper.** Have an LLM pre-annotate, then verify with humans. Modern LLMs produce more consistent annotations than crowd workers, so this can give you cleaner labels at scale. The 2025 paper showed substantial improvements in inter-annotator agreement using this approach.

These are out of scope for a hackathon project but excellent next steps if you want to publish or extend.

## 4. Updated Phase 2 and Phase 3

The original plan said "fine-tune RoBERTa, target F1 0.55-0.65." Given the analysis above, here is the updated plan:

### Updated Phase 2 (Baseline)

**Goal:** Get a working fine-tuning loop end-to-end with a modern base model. Don't optimize yet.

1. Use **DeBERTa-v3-base** instead of DistilBERT/BERT
2. Standard cross-entropy loss, no fancy tricks yet
3. Target F1 ~0.55-0.60 (this is what a well-trained DeBERTa baseline gets)
4. The point of this phase is the loop, not the score

### Updated Phase 3 (Production)

**Goal:** Stack the improvements that beat the 2020 state of the art.

1. Switch to **DeBERTa-v3-large** (or ModernBERT-large if context length matters)
2. Generate **5,000 synthetic training examples** with an LLM, focused on rare classes
3. Implement **class-balanced focal loss**
4. Train with **multi-task setup** including GUS-Net bias data
5. Add **contrastive auxiliary loss**
6. Use **full paragraph context** for technique classification, not truncated 120 tokens
7. Train **3 model variants** for ensemble at inference
8. Target F1 **0.68-0.72** (meaningfully better than 2020 SOTA of 0.62, competitive with 2024-2025 work)
9. For the span identification task, train a separate **DeBERTa-v3 with token classification head** using BIO tagging on the full document with sliding windows

This is genuinely ambitious. If you hit the lower end (0.65), you're still ahead of 2020 SOTA. If you hit the upper end (0.72), you've matched or beaten current academic work in a hackathon project, which is a remarkable result.

### Updated Phase 5 (LLM Layer)

The LLM layer is now smarter — it's used for two purposes:

1. **Verification of low-confidence classifier predictions** (improves accuracy)
2. **Detection of techniques the SemEval taxonomy doesn't cover** (motte-and-bailey, hidden premises, etc.)

This is selective and cost-effective. Most documents will trigger zero LLM calls; the hard cases trigger them surgically.

## 5. The Honest Caveat

All of this is implementable but not trivial. The synthetic data generation alone is a significant subproject — you need to design good prompts, filter bad outputs, and avoid the LLM teaching itself its own biases. Multi-task learning requires careful loss balancing. Contrastive auxiliary objectives can hurt if the margin is wrong.

**Realistic budget: add 1-2 additional weeks to the original plan if you want to actually achieve F1 0.70+.** If you skip the synthetic data and multi-task learning and just upgrade to DeBERTa-v3 with focal loss, you'll get F1 ~0.65 with much less effort. That's still better than the 2020 SOTA — it's a perfectly defensible result for a portfolio project. The 0.70 target is the ambitious version.

The key intellectual point for your portfolio/demo: **you are not just reproducing 2020 results. You are explicitly identifying why they capped where they did, and applying 2024-2025 techniques to push past those limitations.** That narrative is far more impressive than "I fine-tuned BERT on SemEval." Whether you hit 0.65 or 0.72, the framing is the same: you've done a critical analysis of prior work and built something demonstrably better.

---

# Cross-Referencing Datasets: The Multi-Source Strategy

I undersold this in section 3.3 — multi-task training and dataset cross-referencing is actually the **single highest-leverage thing you can do** to push past 0.62, and it's underused in the existing literature. Here's the full picture of what's available and how to combine it.

## The Datasets — Concrete List

### Tier 1: Direct Propaganda/Persuasion Detection

These have the most directly transferable labels.

**SemEval-2020 Task 11** — 14 propaganda techniques in news articles. Span-level + sentence-level. ~6,000 training examples. **The anchor dataset.**

**SemEval-2023 Task 3** — Multilingual extension. 9 languages, refined taxonomy with ~23 persuasion techniques. Adds significantly more examples in English plus opens up cross-lingual training signal. https://propaganda.math.unipd.it/semeval2023task3/

**SemEval-2024 Task 4** — Persuasion techniques in memes. Multimodal (image+text), but the text-only sub-task is directly useful. Larger, more diverse data than 2020. Adds rhetorical and logical fallacy categories the original SemEval missed.

**PTC Corpus** (Propaganda Techniques Corpus) — The base corpus the SemEval datasets were drawn from, sometimes available with cleaner annotations. Worth checking for the original-source version of the data.

**HQP Dataset** (Maarouf et al. 2023) — High-quality propaganda annotations with focus on Russian disinformation campaigns. Different domain from SemEval news but same techniques.

**PropaNews** — Larger-scale propaganda training corpus, partially auto-generated. Good for data augmentation but lower per-example quality.

### Tier 2: Bias and Loaded Language

These tag related phenomena — they don't use the SemEval taxonomy but capture overlapping signal.

**GUS-Net** (2024) — 3,739 text snippets, 69,000+ token-level bias annotations across religion, race, gender, politics, nationality, age. The token-level annotations are gold for span identification training. https://github.com/Ethical-Spectacle/fairly

**Hyperpartisan News Corpus** — Labels articles as hyperpartisan or not. Useful for the encoder to learn what loaded political writing looks like vs. neutral reporting. Available via Zenodo.

**MBIC (Media Bias Identification Benchmark)** — Sentence-level bias annotations with multiple bias types. Smaller (~1,500 sentences) but high-quality.

**BABE (Bias Annotations By Experts)** — 3,700+ sentences annotated for media bias by trained experts. Published 2022, became a standard for media bias detection.

**RP3K (Rich Persuasive Propaganda)** — Recent (2024) dataset specifically for cross-referencing with SemEval. Uses similar but expanded taxonomy.

### Tier 3: Fact-Checking and Claim Verification

These don't directly tag persuasion techniques but provide complementary signal — propaganda often involves dubious claims.

**LIAR** — 12,800 manually labeled political statements from PolitiFact, 6 truthfulness categories (pants-fire, false, mostly-false, half-true, mostly-true, true). Adds signal about what false claims look like. https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

**LIAR-PLUS** — LIAR with extracted evidence sentences. Adds context.

**LIARS' BENCH** (Kretschmar et al., Nov 2025) — Modern benchmark built on LIAR with cleaner reproducibility.

**MultiFC** — Multi-source fact-checking with 36k claims. Different domains (medical, political, etc.) help generalization.

**FEVER** — 185k claims with evidence-based verification labels. Very large; useful as a pre-training corpus.

### Tier 4: Sentiment, Stance, and Emotion

These provide the encoder with general "is this text emotionally charged / taking a strong position" signal, which transfers well to detecting techniques like Loaded Language and Appeal to Fear.

**Sentiment140 / SemEval Sentiment tasks** — Standard sentiment data. Many available. Free signal that loaded vocabulary correlates with technique presence.

**SemEval-2016 Task 6 (Stance Detection)** — Whether a tweet supports/opposes a target. Stance detection signal helps with techniques like Whataboutism (which involves shifting stance focus).

**GoEmotions** (Google) — 58k Reddit comments tagged for 27 emotions. Helps the model develop fine-grained emotional vocabulary.

**SemEval-2025 Task 11** — Recent emotion detection across 32 languages. Adds modern, multi-lingual emotional signal.

### Tier 5: Hate Speech, Toxicity, Figurative Language

Looser connection but still useful for shared encoder representations.

**HateXplain** — Hate speech with rationale annotations. The rationale labels (which words trigger the hate label) are useful for span identification.

**Jigsaw Toxic Comment** — Large-scale toxicity classification.

**FLUTE** — Figurative language understanding (metaphors, irony, sarcasm). Helpful because many propaganda techniques rely on figurative language.

## How to Actually Use Multiple Datasets

There are three legitimate approaches, in increasing order of complexity and payoff.

### Approach 1: Sequential Fine-Tuning (Pre-training-style)

The simplest approach. Fine-tune your DeBERTa-v3 on a related but larger dataset *first* (e.g., GUS-Net for token classification, or Hyperpartisan for political language), *then* fine-tune again on SemEval.

The first stage gives the encoder relevant representations. The second stage specializes for the actual target task. Standard practice; easy to implement.

**Expected gain: +1-2 F1 points.** Cheap and reliable.

### Approach 2: Multi-Task Learning with Shared Encoder

Train a single DeBERTa-v3 encoder on multiple tasks simultaneously. Each task gets its own classification head, but the encoder is shared. The loss is a weighted sum across tasks.

```
DeBERTa-v3 encoder
        │
        ├── Head 1: SemEval propaganda (14 classes)
        ├── Head 2: GUS-Net bias (multi-label)
        ├── Head 3: LIAR truthfulness (6 classes)
        ├── Head 4: Sentiment (3 classes)
        └── Head 5: Hyperpartisan (binary)
```

The encoder learns a richer representation because it has to be useful for all five tasks at once. This regularizes against overfitting to SemEval's quirks and improves generalization.

**Implementation considerations:**
- Loss weighting matters. Don't weight all tasks equally — give SemEval (your target) ~50% of the loss weight, distribute the rest among auxiliary tasks
- Sample task batches in proportion to dataset size, but cap rare ones at a reasonable minimum
- Use Hugging Face's `Trainer` with custom compute_loss method, or PyTorch Lightning for cleaner multi-task code
- Train for more epochs than single-task — the model needs more time to learn the shared structure

**Expected gain: +2-4 F1 points.** Higher payoff than sequential fine-tuning, more implementation effort.

### Approach 3: Unified Taxonomy Mapping (Advanced)

This is where it gets interesting. The different datasets use different label schemes, but many labels overlap conceptually. You can map them into a unified taxonomy.

For example:
- SemEval "Loaded Language" ↔ GUS-Net "biased language" ↔ HateXplain "hateful term" ↔ BABE "biased framing"
- SemEval "Appeal to Fear" ↔ GoEmotions "fear" ↔ MBIC "fear-mongering"
- SemEval "Doubt" ↔ FEVER "refuted" claims often deploy doubt as a tactic

You build a unified label space — say, 30 fine-grained categories that subsume all the source datasets' labels — and convert every dataset into this unified space. Then train one model on the merged corpus.

The result: your training data jumps from 6,000 SemEval examples to potentially 100,000+ examples, all aligned to the same taxonomy. The data scarcity problem (the #1 cause of the 0.62 ceiling) substantially diminishes.

**Implementation considerations:**
- Building the taxonomy mapping is the hardest part. Spend real time on it. A good mapping is a small research contribution in its own right.
- Some labels won't map cleanly. That's fine — leave them as separate categories that only some datasets contribute to.
- Cross-reference the mapped data: examples that get the "loaded language" label from SemEval AND from GUS-Net are higher-quality positives; examples flagged by only one source are lower-quality and should be down-weighted.
- Document the mapping thoroughly. This becomes a key artifact of your project.

**Expected gain: +4-7 F1 points.** This is the biggest single lever. It's also the most work — budget a full week just for the taxonomy mapping and data conversion.

### Approach 4: Cross-Referencing for Quality Filtering (The Critical Move)

Independent of how you train, you can use cross-referencing to filter and improve your training data.

The key insight: **examples labeled with similar techniques across multiple datasets are higher-confidence training data.** A sentence that SemEval calls "Appeal to Fear" AND that has high "fear" sentiment in GoEmotions AND that's labeled "false" in LIAR — that's an extremely high-quality training example. A sentence that only one dataset flags is more questionable.

You can use this to:
- **Boost confidence-weighted samples** during training (multi-source agreement → higher weight)
- **Filter out noise** (single-source samples that don't agree with cross-references → exclude or down-weight)
- **Identify dataset bias** (if your model is doing well on multi-confirmed examples but poorly on single-source examples, the single-source label is probably wrong)
- **Find disagreements as research signal** (examples where datasets disagree are themselves interesting — they're the boundary cases worth thinking about)

This isn't really a separate approach — it's a layer on top of any of the above three. But it's where you start getting *quality* improvements rather than just *quantity* improvements.

**Expected gain: +1-3 F1 points** on top of whichever main approach you choose. More importantly, it gives you confidence intervals on your predictions you wouldn't otherwise have.

## The Recommended Strategy

Given your goal (push past 0.62, ambitious target 0.70+, hackathon-portfolio scope), here's what I'd actually do:

**Phase A (do this for sure):**
1. Sequential fine-tuning: pre-train DeBERTa-v3 on Hyperpartisan + GUS-Net first, then fine-tune on SemEval-2020. Cheap +1-2 F1.
2. Add SemEval-2023 Task 3 English data to your SemEval-2020 training set directly — same taxonomy, more data. Free +1-2 F1.

**Phase B (do this if you have time):**
3. Multi-task training across SemEval + GUS-Net + LIAR + Hyperpartisan with shared encoder. +2-3 F1 over Phase A.

**Phase C (do this for a research-grade result):**
4. Build a unified taxonomy mapping. Merge SemEval-2020 + SemEval-2023 + GUS-Net + BABE + HQP into one aligned dataset. Train on the merged corpus. +3-5 F1 over Phase B.
5. Cross-reference filtering — weight high-agreement samples higher, down-weight singletons. +1-2 F1.

Phase A alone takes you to ~0.65 (already past 2020 SOTA). Phase B takes you to ~0.68. Phase C with the unified taxonomy takes you to ~0.70-0.72 territory.

If the ambitious target matters, **Phase C is what gets you there**. The taxonomy mapping is the unsexy work that nobody does. That's exactly why doing it well is your competitive advantage. It's also the best portfolio narrative — "I unified five propaganda/bias datasets into a single training corpus" is a much more sophisticated story than "I fine-tuned BERT on SemEval."

## Updated Phase 2 and Phase 3 (Final Version)

Replacing the earlier updates with the dataset cross-referencing strategy folded in.

### Updated Phase 2 (Baseline + Initial Multi-Source)

**Goal:** Working training loop on a modern model with sequential fine-tuning across two datasets.

1. Use **DeBERTa-v3-base** as the encoder
2. Sequential fine-tuning: GUS-Net (token classification) first, then SemEval-2020 (technique classification)
3. Add SemEval-2023 Task 3 English data into the SemEval-2020 training mix
4. Standard training, focal loss for class imbalance
5. Target F1 ~0.62-0.66 — already at or past 2020 SOTA

### Updated Phase 3 (Production Multi-Task + Synthetic Data)

**Goal:** Stack everything that beats current academic state-of-the-art.

1. Switch to **DeBERTa-v3-large** (or ModernBERT-large for context length)
2. Generate **5,000 synthetic training examples** with an LLM, focused on rare classes
3. **Multi-task setup** with shared encoder: SemEval propaganda (target) + GUS-Net bias + LIAR truthfulness + Hyperpartisan (auxiliary)
4. **Class-balanced focal loss** with task-specific weighting
5. **Contrastive auxiliary loss** for confusable categories
6. **Full paragraph context** — no 120-token truncation
7. Cross-reference filtering: weight high-agreement samples 1.5x, down-weight singletons 0.7x
8. Train **3 model variants** for ensemble at inference
9. Target F1 **0.68-0.72**

### New Phase 3.5 (Ambitious — Unified Taxonomy)

**Goal:** Match or beat 2024-2025 academic state-of-the-art with a unified-taxonomy training corpus.

1. Build the unified taxonomy mapping in a `TAXONOMY_UNIFIED.md` document — 25-30 fine-grained categories covering SemEval-2020, SemEval-2023, GUS-Net, BABE, HQP
2. Write conversion scripts for each source dataset that map their labels into the unified taxonomy
3. Document edge cases and ambiguous mappings — these become research artifacts
4. Train DeBERTa-v3-large on the merged corpus (~50,000+ examples after merging)
5. Re-evaluate on the original SemEval-2020 test set as the canonical benchmark
6. Target F1 **0.70-0.75** — competitive with current academic state-of-the-art

This is the version of the project that's worth a blog post or research write-up. Everything before is a portfolio project. Phase 3.5 is a research contribution.

## Updated Phase 1 Tasks

Update the Phase 1 download list:

```
Datasets to download in Phase 1:
1. SemEval-2020 Task 11 (anchor): https://zenodo.org/records/3952415
2. SemEval-2023 Task 3 (more data, same domain): https://propaganda.math.unipd.it/semeval2023task3/
3. GUS-Net (token-level bias): https://github.com/Ethical-Spectacle/fairly  
4. LIAR (fact-checking): https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
5. Hyperpartisan News (political bias): https://zenodo.org/records/1489920
6. BABE (expert media bias): https://github.com/Media-Bias-Group/MBIC

Optional / nice-to-have:
7. PropaNews (large-scale, lower quality): https://github.com/CHENG-LIN-LI/PropaNews
8. HQP (high-quality propaganda): per the 2023 paper
9. GoEmotions (emotion baseline): https://github.com/google-research/google-research/tree/master/goemotions
```

The data exploration notebook should now compare these datasets — overlap in vocabulary, shared techniques, label distribution. The notebook itself is a research artifact.

## The Honest Assessment

Cross-referencing five datasets into a unified taxonomy is **at least 1 additional week of work**, on top of the 2-3 week base estimate for the project. So this version is realistically a 4-week project for an ambitious version, vs 2-3 weeks for the basic version.

But the upside is real: the basic version is "I fine-tuned BERT and got 0.65 on SemEval." The cross-referenced version is "I unified five propaganda/bias datasets into a 50,000-example aligned corpus and trained a model that pushes past current academic state-of-the-art on the SemEval benchmark." One of those is a portfolio piece. The other is a paper.

If you have the time, do Phase C. The unified taxonomy work is the differentiator and it's where the project becomes genuinely interesting rather than just well-executed.
