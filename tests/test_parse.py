import json
from datetime import date, time
from pathlib import Path
from typing import Any

import pytest

from weather_rag.errors import ForecastParseError
from weather_rag.parse import parse_forecast

FIXTURES = Path(__file__).parent / "fixtures"


def make_hour(hour: int, **overrides: Any) -> dict[str, Any]:
    return {
        "datetime": f"{hour:02d}:00:00",
        "temp": 12.0,
        "feelslike": 10.5,
        "precipprob": 20.0,
        "windspeed": 9.0,
        "conditions": "Partially cloudy",
        **overrides,
    }


def make_day(day: str, **overrides: Any) -> dict[str, Any]:
    return {
        "datetime": day,
        "tempmax": 15.0,
        "tempmin": 8.0,
        "conditions": "Rain, Partially cloudy",
        "precipprob": 60.0,
        "preciptype": ["rain"],
        "sunrise": "07:21:30",
        "sunset": "18:05:12",
        "hours": [make_hour(h) for h in range(24)],
        **overrides,
    }


def make_response(*days: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return {
        "resolvedAddress": "London, England, United Kingdom",
        "description": "Cooler with rain midweek.",
        "days": list(days),
        **overrides,
    }


def test_parses_real_api_fixture() -> None:
    raw = json.loads((FIXTURES / "valid.json").read_text())

    forecast = parse_forecast(raw)

    assert len(forecast.days) == 7
    assert forecast.warnings == []
    assert all(day.tempmin <= day.tempmax for day in forecast.days)
    assert all(day.hours for day in forecast.days)


def test_parses_core_fields_and_types() -> None:
    forecast = parse_forecast(make_response(make_day("2026-10-18")))

    day = forecast.days[0]
    assert forecast.location == "London, England, United Kingdom"
    assert forecast.outlook == "Cooler with rain midweek."
    assert day.day == date(2026, 10, 18)
    assert (day.tempmin, day.tempmax) == (8.0, 15.0)
    assert day.sunset == time(18, 5, 12)
    assert len(day.hours) == 24
    assert day.hours[18].at == time(18, 0)


def test_day_missing_core_field_is_skipped_with_warning() -> None:
    broken = make_day("2026-10-19")
    del broken["tempmax"]

    forecast = parse_forecast(make_response(make_day("2026-10-18"), broken))

    assert [d.day for d in forecast.days] == [date(2026, 10, 18)]
    assert len(forecast.warnings) == 1
    assert "2026-10-19" in forecast.warnings[0]
    assert "tempmax" in forecast.warnings[0]


def test_unreadable_hours_are_dropped_not_fatal() -> None:
    hours: list[Any] = [make_hour(h) for h in range(24)]
    hours[5] = {"temp": 10.0}  # no time
    hours[6] = "garbage"

    forecast = parse_forecast(make_response(make_day("2026-10-18", hours=hours)))

    assert len(forecast.days[0].hours) == 22
    assert "skipped 2" in forecast.warnings[0]


def test_nulls_become_safe_defaults() -> None:
    day = make_day("2026-10-18", conditions=None, preciptype=None, windgust=None)

    parsed = parse_forecast(make_response(day)).days[0]

    assert parsed.conditions == "Unknown"
    assert parsed.preciptype == []
    assert parsed.windgust is None


def test_full_datetime_sunrise_is_accepted() -> None:
    day = make_day("2026-10-18", sunrise="2026-10-18T07:21:30")

    assert parse_forecast(make_response(day)).days[0].sunrise == time(7, 21, 30)


def test_days_are_sorted_by_date() -> None:
    forecast = parse_forecast(make_response(make_day("2026-10-20"), make_day("2026-10-18")))

    assert [d.day.isoformat() for d in forecast.days] == ["2026-10-18", "2026-10-20"]


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"days": []},
        {"days": "not a list"},
        {"days": [{"datetime": "2026-10-18"}]},  # only day is unusable
    ],
)
def test_unusable_response_raises(raw: dict[str, Any]) -> None:
    with pytest.raises(ForecastParseError):
        parse_forecast(raw)
