# Cognitive Bias Spotter — Backend

FastAPI-based analysis pipeline for detecting persuasion techniques in text.

## Setup

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e ".[dev]"
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
