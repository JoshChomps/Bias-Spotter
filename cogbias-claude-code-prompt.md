# Claude Code Kickoff Prompt — Cognitive Bias Spotter

Copy and paste the prompt below into Claude Code (or any agentic coding tool) to start building the Cognitive Bias Spotter / Persuasion Density Analyzer.

---

## How to Use

1. Open Claude Code in a fresh empty directory: `claude`
2. Paste the prompt below as your first message
3. Let it scaffold the project, then review what it built
4. Iterate from there

The prompt is designed to:
- Get the agent to ask clarifying questions before diving in (saves rework)
- Establish the architecture clearly so it doesn't drift
- Use free/local resources by default (no token-heavy LLM calls in the MVP)
- Build something demo-able quickly with a clean upgrade path

---

## The Prompt

```
I'm building a project called "Cognitive Bias Spotter" — a tool that analyzes persuasive text 
(news articles, political speeches, ad copy, opinion pieces) and highlights every cognitive 
bias and persuasion technique being deployed. It produces a "manipulation density" score 
calibrated against a reference corpus, color-codes the text by technique category, and 
shows explanations on hover.

GOAL FOR THIS SESSION:
Set up the project skeleton, scaffolding, and a working MVP backend that can detect ~10 
common persuasion techniques in pasted text. Frontend can be minimal for now — focus on 
getting the analysis pipeline working end-to-end.

ARCHITECTURE (3 LAYERS):

Layer 1 — Lexicon Matching (local, fast, $0)
   - Curated regex/lexicon of marketing and rhetoric phrases
   - Catches scarcity ("limited time," "only X left"), urgency ("act now"), 
     social proof ("thousands agree"), authority appeals ("doctors recommend"),
     identity bait ("real Americans"), loaded language
   - Should run in ~50ms locally with zero external calls

Layer 2 — Structural NLP (local, $0) — DEFER for now, scaffold the interface only
   - Will use spaCy for passive voice detection, missing comparisons, framing analysis
   - Skip implementation for MVP, but design the pipeline to support it later

Layer 3 — LLM Reasoning (selective, low cost)
   - Only flagged passages from Layers 1-2 get sent to an LLM
   - Use Hugging Face Inference API (free tier) by default, configurable to swap in 
     Ollama (local) or a paid API later
   - Detects subtler stuff: false dichotomies, motte-and-bailey, hidden premises

OUTPUT FORMAT:
For each piece of analyzed text, return:
- A list of detected techniques with text spans (start, end, technique_name, explanation)
- An overall "manipulation density" score (0-100)
- A breakdown of techniques by category (emotional, statistical, framing, social, etc.)
- A percentile ranking against a reference corpus (placeholder for now)

DATASETS TO LEVERAGE:
- SemEval-2020 Task 11 (propaganda techniques in news articles, ~14 categories) — 
  available on Zenodo. Use this as the canonical taxonomy.
- GUS-Net dataset for bias-related tagging if needed.
- Pre-trained Hugging Face models exist for SemEval propaganda detection — use one of 
  these for Layer 3 instead of paying for LLM calls if quality is sufficient.

TECHNIQUES TO DETECT (MVP — start with these 10):
1. Loaded language (emotionally charged words)
2. Appeal to fear
3. Appeal to authority
4. Bandwagon / social proof
5. False scarcity / urgency
6. Name calling / labeling
7. Doubt (questioning credibility without evidence)
8. Exaggeration / minimization
9. Slogans / thought-terminating clichés
10. Black-and-white fallacy

TECH STACK:
- Backend: Python with FastAPI (the analysis pipeline + REST API)
- Layer 1: Pure Python regex + a JSON/YAML lexicon file
- Layer 3: huggingface_hub library calling free Inference API (model swappable via config)
- Caching: SQLite by content hash so re-analysis is free
- Frontend: Minimal HTML/JS or basic Next.js — defer fancy UI, focus on the API

PROJECT STRUCTURE I WANT:
/cogbias-spotter
  /backend
    /app
      main.py              # FastAPI app entry
      /pipeline
        layer1_lexicon.py  # regex matching
        layer2_structural.py  # stubs for now
        layer3_llm.py      # HF inference wrapper
        aggregator.py      # combines layer outputs into final result
      /lexicon
        techniques.yaml    # the curated patterns
      /cache
        cache.py           # SQLite by content hash
      /scoring
        density.py         # manipulation density calculation
        reference_corpus.py  # placeholder for percentile ranking
    /tests
      test_layer1.py
      test_aggregator.py
    pyproject.toml         # use uv or poetry, your choice
    README.md
  /frontend (minimal — defer)
    index.html             # paste-text input, calls API, renders highlights

QUESTIONS BEFORE YOU START:
Please ask me these first before writing any code:

1. Do I have a Hugging Face API token? If not, do I want to set one up now (free, 5 min) 
   or skip Layer 3 entirely for the MVP?
2. Do I want to use uv, poetry, or plain pip for Python dependency management?
3. Frontend now or later? If now, plain HTML or a framework?
4. Any specific text examples I want to use as the first end-to-end test (e.g., a 
   particular speech, ad, or article)?
5. Should the lexicon be opinionated and pre-populated by you, or should we collaboratively 
   build it together as we go?

ONCE YOU HAVE ANSWERS:
1. Set up the project structure
2. Implement Layer 1 with a starter lexicon covering the 10 MVP techniques
3. Stub Layer 2 (interface only)
4. Implement Layer 3 with Hugging Face Inference API
5. Build the aggregator that combines layer outputs
6. Set up SQLite caching
7. Expose a /analyze endpoint via FastAPI
8. Write basic tests for Layer 1 and the aggregator
9. Demo end-to-end with one of the test examples I provided

IMPORTANT CONSTRAINTS:
- Don't burn through tokens being verbose — get to working code fast
- Ask before installing big dependencies
- Commit early and often (git init, frequent commits with clear messages)
- Don't try to build the frontend in detail this session — backend first
- If Hugging Face inference fails or is too slow, fall back to a simple keyword-based 
  Layer 3 that returns "needs further analysis" rather than blocking
- Keep all code in clean, well-typed Python (use type hints throughout)

Let's go.
```

---

## After This Session

Once the agent has built the MVP, your follow-up sessions should focus on:

1. **Filling out the lexicon** — the quality of Layer 1 depends entirely on how good your patterns are. Spend real time curating these.
2. **Implementing Layer 2** — spaCy-based structural analysis for passive voice, missing comparisons, framing detection.
3. **Building the reference corpus** — score 1000-5000 documents across genres so percentile rankings are meaningful.
4. **Frontend** — clean text annotation UI with hover explanations and a density meter.
5. **Browser extension** — once the API is solid, wrap it in a Chrome/Firefox extension that analyzes any webpage.
6. **Personalized features** — blind spot quiz, longitudinal tracking, side-by-side rewrites.

Each of these can be its own focused Claude Code session with a similar kickoff prompt scoped to that feature.
