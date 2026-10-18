import json
from typing import Any

import pytest
import requests
import responses

from weather_rag.llm import (
    FakeProvider,
    LLMResponseError,
    LLMUnavailableError,
    OllamaProvider,
    get_provider,
)

HOST = "http://localhost:11434"
URL = f"{HOST}/api/chat"
SCHEMA: dict[str, Any] = {"type": "object"}


def reply(content: str) -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": content}, "done": True}


@responses.activate
def test_sends_chat_request_and_returns_content() -> None:
    responses.post(URL, json=reply('{"answer": "hi"}'))

    result = OllamaProvider(HOST, "gemma4:12b").generate("sys", "user", SCHEMA)

    assert result == '{"answer": "hi"}'
    body = json.loads(responses.calls[0].request.body or b"")
    assert body["model"] == "gemma4:12b"
    assert body["format"] == SCHEMA
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


@responses.activate
def test_local_server_gets_no_auth_header() -> None:
    responses.post(URL, json=reply("ok"))

    OllamaProvider(HOST, "gemma4:12b").generate("s", "u", SCHEMA)

    assert "Authorization" not in responses.calls[0].request.headers


@responses.activate
def test_cloud_key_is_sent_as_bearer_header_never_in_url() -> None:
    responses.post("https://ollama.com/api/chat", json=reply("ok"))

    OllamaProvider("https://ollama.com", "gemma4:31b", api_key="secret").generate("s", "u", SCHEMA)

    request = responses.calls[0].request
    assert request.headers["Authorization"] == "Bearer secret"
    assert "secret" not in (request.url or "")


@responses.activate
@pytest.mark.parametrize("error", [requests.Timeout(), requests.ConnectionError()])
def test_unreachable_model_raises_unavailable(error: Exception) -> None:
    responses.post(URL, body=error)

    with pytest.raises(LLMUnavailableError):
        OllamaProvider(HOST, "gemma4:12b").generate("s", "u", SCHEMA)


@responses.activate
@pytest.mark.parametrize(
    ("status", "hint"), [(404, "ollama pull"), (401, "credentials"), (500, "500")]
)
def test_error_status_raises_unavailable_with_hint(status: int, hint: str) -> None:
    responses.post(URL, status=status, json={"error": "x"})

    with pytest.raises(LLMUnavailableError, match=hint) as exc_info:
        OllamaProvider(HOST, "gemma4:12b", api_key="secret").generate("s", "u", SCHEMA)

    assert "secret" not in str(exc_info.value)


@responses.activate
@pytest.mark.parametrize("body", ['{"done": true}', "not json", '{"message": {"content": 5}}'])
def test_unexpected_body_raises_response_error(body: str) -> None:
    responses.post(URL, body=body, content_type="application/json")

    with pytest.raises(LLMResponseError):
        OllamaProvider(HOST, "gemma4:12b").generate("s", "u", SCHEMA)


def test_fake_provider_replays_replies_and_errors() -> None:
    fake = FakeProvider("first", LLMUnavailableError("down"))

    assert fake.generate("s", "u", SCHEMA) == "first"
    with pytest.raises(LLMUnavailableError):
        fake.generate("s", "u", SCHEMA)
    assert len(fake.calls) == 2


def test_get_provider_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.com/")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:cloud")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "not a number")

    provider = get_provider()

    assert isinstance(provider, OllamaProvider)
    assert provider.host == "https://ollama.com"
    assert provider.model == "gemma4:cloud"
    assert provider.timeout == 30.0
