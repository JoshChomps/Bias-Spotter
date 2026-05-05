# Cognitive Bias Spotter — Backend

FastAPI-based analysis pipeline for detecting persuasion techniques in text.

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install core + dev dependencies
pip install -r requirements-dev.txt
```

## Running

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

## Testing

```bash
pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze/text` | Analyze pasted text |
| POST | `/analyze/document` | Analyze uploaded document |
| GET | `/analyze/{cache_id}` | Retrieve cached analysis |
| GET | `/reference-corpus/info` | Reference corpus stats |
| GET | `/health` | Health check |
