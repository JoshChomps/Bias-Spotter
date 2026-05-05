"""
App Configuration — loads from .env file.

All environment variables are accessed through this module.
Import `settings` anywhere in the app instead of calling os.getenv() directly.

Usage:
    from app.config import settings
    token = settings.huggingface_token
"""

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from the backend directory
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


class Settings:
    """Central settings object populated from environment variables."""

    @property
    def huggingface_token(self) -> str | None:
        return os.getenv("HUGGINGFACE_TOKEN")

    @property
    def llm_backend(self) -> str:
        return os.getenv("LLM_BACKEND", "huggingface")

    @property
    def ollama_base_url(self) -> str:
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def ollama_model(self) -> str:
        return os.getenv("OLLAMA_MODEL", "llama3.1")

    @property
    def anthropic_api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")


settings = Settings()
