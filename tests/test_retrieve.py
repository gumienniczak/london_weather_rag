from datetime import date

import pytest

from factories import START, make_forecast
from weather_rag.chunks import Chunk, DayPart, forecast_chunks, guide_chunks
from weather_rag.retrieve import Retriever, read_scope

GUIDE = """# What to wear
## Rain
Wear a waterproof jacket and shoes that cope with puddles.
## Wind
A windproof layer helps; umbrellas struggle in strong gusts.
## Running
Light layers and a reflective top for running after dark.
"""


@pytest.fixture
def retriever() -> Retriever:
    wet_thursday_evening = {(4, h): {"precipprob": 80.0} for h in range(17, 22)}
    forecast = make_forecast(hour_overrides=wet_thursday_evening)
    return Retriever(forecast_chunks(forecast), guide_chunks(GUIDE, source="what-to-wear"))


def ids(chunks: list[Chunk]) -> list[str]:
    return [c.id for c in chunks]


# --- reading the scope of a question ---------------------------------------------


@pytest.mark.parametrize(
    ("question", "days"),
    [
        ("Do I need a coat tomorrow?", [date(2026, 10, 19)]),
        ("What about Thursday?", [date(2026, 10, 22)]),
        ("Any rain on the 23rd?", [date(2026, 10, 23)]),
        ("How about 21st of October?", [date(2026, 10, 21)]),
        ("weather in 3 days", [date(2026, 10, 21)]),
        ("This weekend?", [date(2026, 10, 18), date(2026, 10, 24)]),
        ("next Sunday?", [date(2026, 10, 25)]),
        ("What was it like last Friday?", [date(2026, 10, 16)]),
    ],
)
def test_reads_days(question: str, days: list[date]) -> None:
    assert read_scope(question, today=START).days == days


def test_reads_parts_and_times() -> None:
    scope = read_scope("Thursday evening, around 6:30pm, or tonight?", today=START)

    assert DayPart.EVENING in scope.parts
    assert DayPart.NIGHT in scope.parts
    assert scope.hours == [18]  # 6:30pm is not also read as 06:30


def test_flags_far_future() -> None:
    assert read_scope("Will it snow next month?", today=START).far_future


# --- the coverage check ----------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "What was it like last Friday?",
        "Rain next Sunday?",
        "Any sun on 30 October?",
        "Snow next month?",
    ],
)
def test_out_of_range_questions_get_no_chunks_and_a_note(
    retriever: Retriever, question: str
) -> None:
    result = retriever.retrieve(question)

    assert not result.answerable
    assert result.chunks == []
    assert result.note is not None
    assert "The forecast covers Sun 18 Oct to Sat 24 Oct" in result.note


def test_partly_covered_question_keeps_what_exists_and_notes_the_rest(retriever: Retriever) -> None:
    result = retriever.retrieve("Compare Thursday with 30 October")

    assert result.answerable
    assert {c.day for c in result.chunks if c.day} == {date(2026, 10, 22)}
    assert result.note is not None and "Fri 30 Oct" in result.note


# --- selection and ranking -------------------------------------------------------


def test_day_and_part_select_exactly_that_chunk(retriever: Retriever) -> None:
    result = retriever.retrieve("Will it rain on Thursday evening?")

    assert ids([c for c in result.chunks if c.kind == "forecast"]) == ["2026-10-22-evening"]


def test_part_without_day_returns_every_evening_in_order(retriever: Retriever) -> None:
    result = retriever.retrieve("Which evening this week is best for a run along the Thames?")
    evenings = [c for c in result.chunks if c.kind == "forecast"]

    assert len(evenings) == 7
    assert all(c.part is DayPart.EVENING for c in evenings)
    assert ids(evenings) == sorted(ids(evenings))


def test_unscoped_question_is_ranked_and_capped(retriever: Retriever) -> None:
    result = retriever.retrieve("When is it wet?")
    forecast = [c for c in result.chunks if c.kind == "forecast"]

    assert len(forecast) == 8
    assert "2026-10-22-evening" in ids(forecast)


def test_time_near_boundary_adds_neighbouring_chunk(retriever: Retriever) -> None:
    result = retriever.retrieve("Will it be cold at 6am on Tuesday?")

    assert {"2026-10-19-night", "2026-10-20-morning"} <= set(ids(result.chunks))


def test_small_hours_map_to_previous_night(retriever: Retriever) -> None:
    result = retriever.retrieve("Rain at 3am on Wednesday?")

    assert "2026-10-20-night" in ids(result.chunks)


def test_matching_guide_sections_are_added(retriever: Retriever) -> None:
    result = retriever.retrieve("Do I need a waterproof jacket for running on Thursday?")
    guides = [c for c in result.chunks if c.kind == "guide"]

    assert guides and len(guides) <= 2
    assert "what-to-wear-rain" in ids(guides)


def test_no_forecast_data_is_not_answerable() -> None:
    result = Retriever([], []).retrieve("Anything?")

    assert not result.answerable
    assert result.note == "No forecast data is available."


def test_neighbour_is_added_only_across_the_boundary(retriever: Retriever) -> None:
    result = retriever.retrieve("Will it rain at 6am tomorrow?")

    forecast = [c for c in result.chunks if c.kind == "forecast"]
    assert ids(forecast) == ["2026-10-18-night", "2026-10-19-morning"]


def test_uninformative_question_falls_back_to_soonest_periods(retriever: Retriever) -> None:
    result = retriever.retrieve("What's the weather like?")

    assert ids(result.chunks) == ids(retriever.forecast_chunks[:8])


def test_single_section_guide_still_matches() -> None:
    guides = guide_chunks("## Rain\nTake a waterproof.\n", source="wear")
    result = Retriever(forecast_chunks(make_forecast()), guides).retrieve("Need a waterproof?")

    assert "wear-rain" in ids(result.chunks)


def test_guide_without_shared_words_is_not_added(retriever: Retriever) -> None:
    result = retriever.retrieve("Thursday evening?")

    assert [c for c in result.chunks if c.kind == "guide"] == []
