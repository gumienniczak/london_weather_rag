"""Streamlit front end: presentation only. All logic lives in the weather_rag package.

Run with:  streamlit run app.py
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from weather_rag import config
from weather_rag.answer import AnswerResult, answer_question
from weather_rag.charts import (
    OUTING_PARTS,
    best_day,
    outing_chart,
    outing_scores,
    rain_heatmap,
    temperature_chart,
)
from weather_rag.chunks import Chunk, DayPart, forecast_chunks, guide_chunks
from weather_rag.errors import ForecastError, MissingApiKeyError
from weather_rag.fetch import fetch_forecast
from weather_rag.llm import LLMProvider, get_provider
from weather_rag.parse import DayForecast, Forecast, parse_forecast
from weather_rag.retrieve import Retriever
from weather_rag.units import UnitSystem, temperature, temperature_unit, wind, wind_unit

GUIDE_PATH = Path(__file__).parent / "corpus" / "what_to_wear.md"
EXAMPLE_QUESTIONS = [
    "Which evening this week is best for a run along the Thames?",
    "Do I need a coat on Thursday morning?",
    "Will it rain this weekend?",
]
ICONS = {
    "clear-day": "☀️",
    "clear-night": "🌙",
    "partly-cloudy-day": "⛅",
    "partly-cloudy-night": "☁️",
    "cloudy": "☁️",
    "fog": "🌫️",
    "wind": "💨",
    "rain": "🌧️",
    "showers-day": "🌦️",
    "showers-night": "🌧️",
    "thunder-rain": "⛈️",
    "thunder-showers-day": "⛈️",
    "thunder-showers-night": "⛈️",
    "snow": "❄️",
    "snow-showers-day": "🌨️",
    "snow-showers-night": "🌨️",
}

st.set_page_config(page_title="London Forecast", page_icon="🌦️", layout="wide")


# --- data loading ---------------------------------------------------------------------


@st.cache_data(ttl=config.get_cache_ttl_minutes() * 60, show_spinner=False)
def load_forecast_data() -> tuple[dict[str, Any], datetime]:
    """Fetch the raw forecast once per cache period; failures are not cached."""
    return fetch_forecast(), datetime.now()


def build_retriever(forecast: Forecast) -> Retriever:
    guides = guide_chunks(GUIDE_PATH.read_text(), "what-to-wear") if GUIDE_PATH.exists() else []
    return Retriever(forecast_chunks(forecast), guides)


def ask(
    question: str, forecast: Forecast, fetched_at: datetime, provider: LLMProvider
) -> AnswerResult:
    """Answer a question, reusing good answers for the same question and forecast."""
    cache: dict[tuple[str, str, str], AnswerResult] = st.session_state.setdefault("answers", {})
    key = (question.strip().lower(), fetched_at.isoformat(), provider.name)
    if key in cache:
        return cache[key]
    timeout = config.get_llm_timeout()
    with st.spinner(f"Asking {provider.name}… a local model can take up to {timeout:.0f} s"):
        result = answer_question(question, build_retriever(forecast), provider)
    if result.status == "answered":  # never cache failures: the model may be back next time
        cache[key] = result
    return result


# --- small rendering helpers ----------------------------------------------------------


def fmt_temp(celsius: float, units: UnitSystem) -> str:
    return f"{temperature(celsius, units):.0f}{temperature_unit(units)}"


def fmt_wind(mph: float, units: UnitSystem) -> str:
    return f"{wind(mph, units):.0f} {wind_unit(units)}"


def day_card(day: DayForecast, units: UnitSystem) -> None:
    with st.container(border=True):
        st.markdown(f"**{day.day:%a}** {day.day.day} {day.day:%b}")
        st.subheader(ICONS.get(day.icon or "", "🌡️"), anchor=False)
        st.markdown(f"**{fmt_temp(day.tempmax, units)}** / {fmt_temp(day.tempmin, units)}")
        st.caption(day.conditions)
        if day.precipprob is not None:
            st.caption(f"💧 {day.precipprob:.0f}% chance of rain")


def labels(chunks: list[Chunk]) -> str:
    return ", ".join(chunk.label for chunk in chunks) or "nothing"


def show_answer(question: str, result: AnswerResult) -> None:
    st.markdown(f"**You asked:** {question}")

    if result.status == "unanswerable":
        st.info(result.note or "That question can't be answered from this forecast.")
    elif result.status == "answered":
        st.success(result.answer or "")
        forecast_considered = [c for c in result.considered if c.kind == "forecast"]
        st.caption(
            f"**Cited:** {labels(result.sources)}  \n"
            f"**Considered:** {len(forecast_considered)} forecast period(s)"
        )
    else:
        st.warning(
            f"The AI answer isn't available: {result.reason} "
            "Here is the forecast data it would have used."
        )

    if result.note and result.status != "unanswerable":
        st.caption(result.note)

    shown = result.sources if result.status == "answered" else result.considered
    if shown:
        with st.expander(
            "Forecast data used" if result.status == "answered" else "Relevant forecast data",
            expanded=result.status == "fallback",
        ):
            for chunk in shown:
                st.markdown(f"**{chunk.label}**")
                st.text(chunk.text)

    if result.trace:
        with st.expander("How this answer was produced"):
            for step in result.trace:
                st.markdown(f"{'✅' if step.ok else '❌'} {step.text}")


# --- page -----------------------------------------------------------------------------

provider = get_provider()

with st.sidebar:
    st.header("Settings")
    units = st.radio("Units", list(UnitSystem), format_func=lambda u: u.value) or UnitSystem.UK
    if st.button("Refresh forecast", width="stretch"):
        load_forecast_data.clear()
        st.rerun()
    st.divider()
    st.caption(f"AI answers: {provider.name}")
    st.caption("Weather data: Visual Crossing")
    st.caption(f"Forecast reused for {config.get_cache_ttl_minutes()} min between fetches")

st.title("London · the next 7 days", anchor=False)

try:
    with st.spinner("Fetching the forecast…"):
        raw, fetched_at = load_forecast_data()
    forecast = parse_forecast(raw)
except MissingApiKeyError as exc:
    st.error(str(exc))
    st.stop()
except ForecastError as exc:
    st.error(f"Couldn't load the forecast. {exc}")
    if st.button("Try again"):
        load_forecast_data.clear()
        st.rerun()
    st.stop()

st.caption(f"Updated {fetched_at:%H:%M} · {forecast.location}")
if forecast.outlook:
    st.markdown(f"_{forecast.outlook}_")
for alert in forecast.alerts:
    st.warning(f"⚠️ {alert.headline or alert.event or 'Weather alert'}")
if forecast.warnings:
    with st.expander("Some forecast data was incomplete"):
        for warning in forecast.warnings:
            st.markdown(f"- {warning}")

for column, day in zip(st.columns(len(forecast.days)), forecast.days, strict=True):
    with column:
        day_card(day, units)

plan_tab, ask_tab = st.tabs(["📅 Plan your week", "💬 Ask the forecast"])

with plan_tab:
    st.altair_chart(rain_heatmap(forecast), width="stretch")
    st.caption("Each square is one hour. Find a light patch to stay dry.")

    st.markdown("**How good is each day for being outside?**")

    part = st.radio(
        "Part of the day",
        OUTING_PARTS,
        index=OUTING_PARTS.index(DayPart.EVENING),
        format_func=lambda p: p.value.capitalize(),
        horizontal=True,
    )
    scores = outing_scores(forecast, part)
    st.altair_chart(outing_chart(scores, part, units), width="stretch")
    if best := best_day(scores):
        st.caption(
            f"Best {part.value}: **{best}**. Calculated directly from the forecast, no AI: "
            f"rain chance, gusts above {fmt_wind(15, units)} and distance from a comfortable "
            f"{fmt_temp(15, units)} feels-like each lower the score."
        )

    st.altair_chart(temperature_chart(forecast, units), width="stretch")

with ask_tab:
    st.caption(
        "Answers come only from this forecast and are checked before you see them. "
        "They use °C and mph, matching the source data."
    )
    pending: str | None = None
    for column, example in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS, strict=True):
        if column.button(example, width="stretch"):
            st.session_state["question"] = example
            pending = example

    with st.form("ask", clear_on_submit=False):
        typed = st.text_input(
            "Your question",
            key="question",
            placeholder="e.g. Will it be windy on Saturday afternoon?",
        )
        submitted = st.form_submit_button("Ask")

    question = pending or (typed if submitted else None)
    if question and question.strip():
        st.session_state["last"] = (question, ask(question, forecast, fetched_at, provider))
    elif submitted:
        st.info("Please type a question first.")

    if "last" in st.session_state:
        last_question, last_result = st.session_state["last"]
        show_answer(last_question, last_result)
