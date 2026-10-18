"""Smoke tests for the Streamlit UI, run headless with Streamlit's AppTest.

The weather API and the model are replaced with the saved fixture and a FakeProvider,
so these tests need no network, no API key and no running model.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from weather_rag import fetch, llm
from weather_rag.errors import ForecastUnavailableError
from weather_rag.llm import FakeProvider, LLMUnavailableError

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
FIXTURE: dict[str, Any] = json.loads((ROOT / "tests" / "fixtures" / "valid.json").read_text())


@pytest.fixture(autouse=True)
def clean_cache() -> None:
    st.cache_data.clear()


def run_app(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider, fail: bool = False) -> AppTest:
    def fake_fetch(**_: Any) -> dict[str, Any]:
        if fail:
            raise ForecastUnavailableError("Couldn't reach the weather service.")
        return FIXTURE

    monkeypatch.setattr(fetch, "fetch_forecast", fake_fetch)
    monkeypatch.setattr(llm, "get_provider", lambda: provider)
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    return app


def test_forecast_view_renders_without_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    app = run_app(monkeypatch, FakeProvider())

    assert not app.exception
    assert app.title[0].value == "London · the next 7 days"
    assert len(app.tabs) == 2
    assert any("Updated" in caption.value for caption in app.caption)


def test_weather_service_failure_shows_error_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    app = run_app(monkeypatch, FakeProvider(), fail=True)

    assert not app.exception
    assert "Couldn't reach the weather service" in app.error[0].value
    assert any(button.label == "Try again" for button in app.button)


def ask(app: AppTest, question: str) -> AppTest:
    app.text_input(key="question").input(question)
    next(b for b in app.button if b.label == "Ask").click()
    app.run()
    return app


def test_grounded_answer_is_shown_with_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = json.dumps(
        {"answer": "Looks fine for a walk.", "cited_chunk_ids": ["2026-10-22-evening"]}
    )
    app = ask(run_app(monkeypatch, FakeProvider(reply)), "Thursday evening?")

    assert not app.exception
    assert app.success[0].value == "Looks fine for a walk."
    assert any("Cited:" in caption.value for caption in app.caption)


def test_model_failure_falls_back_to_forecast_data(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeProvider(LLMUnavailableError("Is Ollama running?"))
    app = ask(run_app(monkeypatch, provider), "Thursday evening?")

    assert not app.exception
    assert "Is Ollama running?" in app.warning[-1].value
    assert app.tabs  # the rest of the page is still there


def test_out_of_range_question_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeProvider()
    app = ask(run_app(monkeypatch, provider), "Will it rain on 30 November?")

    assert not app.exception
    assert "no forecast data" in app.info[0].value.lower()
    assert provider.calls == []
