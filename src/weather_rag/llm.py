"""The single interface the app uses to talk to a language model.

``LLMProvider`` is a Protocol with one method, so providers are swappable: the app only
ever calls ``generate()``. ``OllamaProvider`` covers three setups with the same code,
because local Ollama and Ollama's cloud share the ``/api/chat`` endpoint:

- local model:                    OLLAMA_MODEL=gemma4:12b
- cloud model via local Ollama:   ``ollama signin``, then OLLAMA_MODEL=gemma4:cloud
- cloud model without Ollama:     OLLAMA_HOST=https://ollama.com, OLLAMA_API_KEY=...

``FakeProvider`` returns canned replies so tests never need a running model.
"""

from typing import Any, Protocol

import requests

from weather_rag import config


class LLMError(Exception):
    """Base class for anything that stops us getting a usable model reply."""


class LLMUnavailableError(LLMError):
    """The model could not be reached in time (not running, timeout, server error)."""


class LLMResponseError(LLMError):
    """The model service answered, but not with a reply we can read."""


class LLMProvider(Protocol):
    """Anything that can turn a system prompt and a user prompt into a reply string."""

    @property
    def name(self) -> str: ...

    def generate(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        """Return the model's raw reply, asked to follow the JSON ``schema``.

        Raises:
            LLMError: if no reply could be obtained.
        """
        ...


class OllamaProvider:
    """Calls an Ollama server (local or ollama.com) through its ``/api/chat`` endpoint."""

    def __init__(
        self, host: str, model: str, api_key: str | None = None, timeout: float = 30.0
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._api_key = api_key

    @property
    def name(self) -> str:
        return f"Ollama ({self.model})"

    def generate(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "format": schema,  # constrains the reply to our JSON schema
            "stream": False,
            "think": False,
            "options": {"temperature": 0},  # repeatable answers for the same data
        }
        try:
            response = requests.post(
                f"{self.host}/api/chat", json=payload, headers=headers, timeout=self.timeout
            )
        except requests.Timeout:
            raise LLMUnavailableError(
                f"The model took longer than {self.timeout:.0f}s to answer."
            ) from None
        except requests.RequestException:
            raise LLMUnavailableError(
                f"Couldn't reach the model at {self.host}. Is Ollama running?"
            ) from None

        if response.status_code == 404:
            raise LLMUnavailableError(
                f"Model '{self.model}' isn't available. Try `ollama pull {self.model}`."
            )
        if response.status_code in (401, 403):
            raise LLMUnavailableError("The model service rejected the credentials.")
        if not response.ok:
            raise LLMUnavailableError(
                f"The model service returned an error ({response.status_code})."
            )

        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError):
            raise LLMResponseError("The model service returned an unexpected response.") from None
        if not isinstance(content, str):
            raise LLMResponseError("The model service returned an unexpected response.")
        return content


class FakeProvider:
    """Test double: returns queued replies (or raises queued errors) and records prompts."""

    def __init__(self, *replies: str | LLMError) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "Fake"

    def generate(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        self.calls.append((system, prompt))
        if not self._replies:
            raise LLMUnavailableError("FakeProvider has no more replies.")
        reply = self._replies.pop(0)
        if isinstance(reply, LLMError):
            raise reply
        return reply


def get_provider() -> LLMProvider:
    """Build the provider from environment settings (see .env.example)."""
    return OllamaProvider(
        host=config.get_ollama_host(),
        model=config.get_ollama_model(),
        api_key=config.get_ollama_api_key(),
        timeout=config.get_llm_timeout(),
    )
