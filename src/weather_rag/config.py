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
