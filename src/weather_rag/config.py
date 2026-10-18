"""Settings read from environment variables (loaded from .env if present)."""

import os

from dotenv import load_dotenv

from weather_rag.errors import MissingApiKeyError

load_dotenv()


def get_weather_api_key() -> str:
    key = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
    if not key:
        raise MissingApiKeyError(
            "No weather API key is set. Copy .env.example to .env and add your "
            "VISUAL_CROSSING_API_KEY (see README)."
        )
    return key


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:12b"
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0


def get_ollama_host() -> str:
    """Ollama server URL: local by default, or https://ollama.com for direct cloud use."""
    return os.getenv("OLLAMA_HOST", "").strip().rstrip("/") or DEFAULT_OLLAMA_HOST


def get_ollama_model() -> str:
    """Model tag, e.g. gemma4:12b (local) or gemma4:cloud (cloud via a signed-in Ollama)."""
    return os.getenv("OLLAMA_MODEL", "").strip() or DEFAULT_OLLAMA_MODEL


def get_ollama_api_key() -> str | None:
    """Only needed when calling https://ollama.com directly; None for a local server."""
    return os.getenv("OLLAMA_API_KEY", "").strip() or None


def get_llm_timeout() -> float:
    """Seconds to wait for the model; falls back to the default if unset or invalid."""
    try:
        value = float(os.getenv("LLM_TIMEOUT_SECONDS", ""))
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_LLM_TIMEOUT_SECONDS


DEFAULT_CACHE_TTL_MINUTES = 30


def get_cache_ttl_minutes() -> int:
    """How long a fetched forecast is reused before calling the weather API again."""
    try:
        value = int(os.getenv("CACHE_TTL_MINUTES", ""))
    except ValueError:
        return DEFAULT_CACHE_TTL_MINUTES
    return value if value > 0 else DEFAULT_CACHE_TTL_MINUTES
