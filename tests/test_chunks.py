from datetime import date, timedelta
from typing import Any

from weather_rag.chunks import DayPart, chunk_label, forecast_chunks, guide_chunks
from weather_rag.parse import Forecast, parse_forecast

START = date(2026, 10, 18)  # a Sunday


def make_hour(hour: int, **overrides: Any) -> dict[str, Any]:
    return {
        "datetime": f"{hour:02d}:00:00",
        "temp": 12.0,
        "feelslike": 10.0,
        "precipprob": 10.0,
        "precip": 0.0,
        "windspeed": 8.0,
        "windgust": 15.0,
        "uvindex": 1.0,
        "conditions": "Partially cloudy",
        **overrides,
    }


def make_forecast(
    days: int = 7, hour_overrides: dict[tuple[int, int], dict[str, Any]] | None = None
) -> Forecast:
    """Build a forecast of ``days`` full days; override (day_index, hour) entries."""
    overrides = hour_overrides or {}
    raw_days = []
    for index in range(days):
        hours = [make_hour(h, **overrides.get((index, h), {})) for h in range(24)]
        raw_days.append(
            {
                "datetime": (START + timedelta(days=index)).isoformat(),
                "tempmax": 15.0,
                "tempmin": 8.0,
                "sunrise": "07:25:00",
                "sunset": "17:55:00",
                "hours": hours,
            }
        )
    return parse_forecast({"days": raw_days})


def by_id(forecast: Forecast) -> dict[str, str]:
    return {chunk.id: chunk.text for chunk in forecast_chunks(forecast)}


def test_seven_days_give_four_chunks_each_with_stable_ids() -> None:
    chunks = forecast_chunks(make_forecast())

    assert len(chunks) == 28
    assert len({c.id for c in chunks}) == 28
    assert chunks[0].id == "2026-10-18-morning"
    assert chunks[-1].id == "2026-10-24-night"
    assert [c.part for c in chunks[:4]] == list(DayPart)


def test_night_crosses_midnight_into_next_day() -> None:
    forecast = make_forecast(hour_overrides={(1, 3): {"precipprob": 90.0}})  # Mon 03:00

    sunday_night = by_id(forecast)["2026-10-18-night"]

    assert "(22:00–06:00)" in sunday_night
    assert "Rain chance up to 90%" in sunday_night
    assert "03:00 12°C 90%" in sunday_night


def test_last_night_is_marked_partial() -> None:
    last_night = by_id(make_forecast())["2026-10-24-night"]

    assert "Partial: forecast data only covers 22:00–00:00" in last_night


def test_summary_uses_max_rain_and_temperature_range() -> None:
    forecast = make_forecast(
        hour_overrides={
            (0, 18): {"temp": 9.0, "precipprob": 70.0, "preciptype": ["rain"]},
            (0, 19): {"temp": 14.4},
        }
    )

    evening = by_id(forecast)["2026-10-18-evening"]

    assert "temperature 9–14°C" in evening
    assert "Rain chance up to 70%" in evening
    assert "(rain)" in evening
    assert "Sunset 17:55." in evening
    assert "wet" in evening


def test_missing_values_say_not_available() -> None:
    missing = {(0, h): {"precipprob": None, "windgust": None} for h in range(6, 12)}

    morning = by_id(make_forecast(hour_overrides=missing))["2026-10-18-morning"]

    assert "Rain chance up to not available" in morning
    assert "gusts up to not available" in morning
    assert "rain n/a" in morning


def test_tags_reflect_thresholds() -> None:
    windy_cold = {(0, h): {"windgust": 35.0, "feelslike": 3.0} for h in range(12, 17)}

    afternoon = by_id(make_forecast(hour_overrides=windy_cold))["2026-10-18-afternoon"]

    assert "Overall: dry, windy, cold." in afternoon


def test_period_without_hours_is_skipped() -> None:
    forecast = parse_forecast(
        {"days": [{"datetime": START.isoformat(), "tempmax": 15, "tempmin": 8, "hours": []}]}
    )

    assert forecast_chunks(forecast) == []


def test_label_is_readable() -> None:
    assert chunk_label(date(2026, 10, 22), DayPart.EVENING) == "Thu 22 Oct, evening"
    assert chunk_label(date(2026, 10, 22), DayPart.NIGHT) == "Thu 22 Oct, night (into Fri)"


def test_guide_is_split_by_section() -> None:
    markdown = (
        "# What to wear\n\n## Rain\nTake a waterproof.\n\n## Wind\nA windproof layer helps.\n"
    )

    chunks = guide_chunks(markdown, source="what-to-wear")

    assert [c.id for c in chunks] == ["what-to-wear-rain", "what-to-wear-wind"]
    assert chunks[0].text == "Rain: Take a waterproof."
    assert all(c.kind == "guide" and c.day is None for c in chunks)
