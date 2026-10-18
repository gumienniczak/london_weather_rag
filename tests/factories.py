"""Builders for realistic test forecasts, shared across test modules."""

from datetime import date, timedelta
from typing import Any

from weather_rag.parse import Forecast, parse_forecast

START = date(2026, 10, 18)  # a Sunday


def make_hour(hour: int, **overrides: Any) -> dict[str, Any]:
    """Return one raw API-style hour at ``hour``:00 with typical values;
    ``overrides`` replace fields."""
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
    """Build a forecast of ``days`` full days from START; override (day_index, hour) entries."""
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
