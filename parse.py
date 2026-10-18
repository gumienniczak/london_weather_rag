"""Turn the raw Visual Crossing Timeline response into typed forecast objects.

Design: be tolerant of partial data. A day missing its core fields (date, high, low)
is skipped and noted in ``Forecast.warnings``; optional fields simply become ``None``.
Only a response with no usable days at all is treated as an error.
"""

from datetime import date, time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from weather_rag.errors import ForecastParseError


class HourForecast(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    at: time = Field(alias="datetime")
    temp: float | None = None
    feelslike: float | None = None
    precipprob: float | None = None
    precip: float | None = None
    windspeed: float | None = None
    windgust: float | None = None
    uvindex: float | None = None
    preciptype: list[str] = Field(default_factory=list)
    conditions: str | None = None


class DayForecast(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    day: date = Field(alias="datetime")
    tempmax: float
    tempmin: float
    conditions: str = "Unknown"
    description: str | None = None
    icon: str | None = None
    feelslikemax: float | None = None
    feelslikemin: float | None = None
    precip: float | None = None
    precipprob: float | None = None
    preciptype: list[str] = Field(default_factory=list)
    windspeed: float | None = None
    windgust: float | None = None
    uvindex: float | None = None
    sunrise: time | None = None
    sunset: time | None = None
    hours: list[HourForecast] = Field(default_factory=list)

    @field_validator("conditions", mode="before")
    @classmethod
    def _default_conditions(cls, value: object) -> object:
        return "Unknown" if value in (None, "") else value

    @field_validator("preciptype", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: object) -> object:
        return [] if value is None else value

    @field_validator("sunrise", "sunset", mode="before")
    @classmethod
    def _keep_time_part(cls, value: object) -> object:
        # The API may send "07:21:30" or a full "2026-10-18T07:21:30".
        if isinstance(value, str) and "T" in value:
            return value.split("T", 1)[1]
        return value

    @field_validator("preciptype", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: object) -> object:
        return [] if value is None else value


class WeatherAlert(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    event: str | None = None
    headline: str | None = None
    description: str | None = None


class Forecast(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    location: str
    outlook: str | None = None
    days: list[DayForecast]
    alerts: list[WeatherAlert] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _invalid_fields(exc: ValidationError) -> str:
    """Return all validation errors collected by pydantic."""
    return ", ".join(sorted({str(err["loc"][0]) for err in exc.errors() if err["loc"]}))


def _parse_hours(raw_hours: object, label: str, warnings: list[str]) -> list[HourForecast]:
    """Return the readable hours of one day, sorted by time.

    A missing or non-list value yields an empty list. Unreadable hours are dropped
    and their count is appended to ``warnings``, using ``label`` to name the day.
    """
    if not isinstance(raw_hours, list):
        if raw_hours is not None:
            warnings.append(f"{label}: hourly data was not a list and was ignored")
        return []

    hours: list[HourForecast] = []
    skipped = 0
    for raw_hour in raw_hours:
        try:
            hours.append(HourForecast.model_validate(raw_hour))
        except ValidationError:
            skipped += 1
    if skipped:
        warnings.append(f"{label}: skipped {skipped} unreadable hour(s)")
    return sorted(hours, key=lambda h: h.at)


def _parse_day(raw_day: object, index: int, warnings: list[str]) -> DayForecast | None:
    """Parse one raw day, or return None if it lacks usable core fields.

    Hours are stripped before validation and parsed by ``_parse_hours``, so a bad
    hour never discards the whole day. Problems are appended to ``warnings``.
    """
    if not isinstance(raw_day, dict):
        warnings.append(f"Day {index + 1}: not an object, skipped")
        return None

    label = str(raw_day.get("datetime") or f"Day {index + 1}")
    fields = {key: value for key, value in raw_day.items() if key != "hours"}
    try:
        day = DayForecast.model_validate(fields)
    except ValidationError as exc:
        warnings.append(f"{label}: missing or invalid {_invalid_fields(exc)}, skipped")
        return None

    hours = _parse_hours(raw_day.get("hours"), label, warnings)
    return day.model_copy(update={"hours": hours})


def _parse_alerts(raw_alerts: object) -> list[WeatherAlert]:
    """Return the readable weather alerts; missing or malformed ones are dropped.

    Unlike days and hours, dropped alerts are not reported in warnings: alerts are
    an optional extra, so a bad one shouldn't clutter the forecast view.
    """
    if not isinstance(raw_alerts, list):
        return []
    alerts: list[WeatherAlert] = []
    for raw_alert in raw_alerts:
        try:
            alerts.append(WeatherAlert.model_validate(raw_alert))
        except ValidationError:
            continue
    return alerts


def parse_forecast(raw: dict[str, Any]) -> Forecast:
    """Validate a raw Timeline API response and return a ``Forecast``.

    Raises ``ForecastParseError`` if the response contains no usable days.
    """
    if not isinstance(raw, dict):
        raise ForecastParseError("Weather response was not a JSON object")

    raw_days = raw.get("days")
    if not isinstance(raw_days, list) or not raw_days:
        raise ForecastParseError("Weather response contained no daily forecast")

    warnings: list[str] = []
    days = [
        day
        for index, raw_day in enumerate(raw_days)
        if (day := _parse_day(raw_day, index, warnings)) is not None
    ]
    if not days:
        raise ForecastParseError("No day in the weather response could be read")

    outlook = raw.get("description")
    return Forecast(
        location=str(raw.get("resolvedAddress") or raw.get("address") or "London"),
        outlook=outlook if isinstance(outlook, str) else None,
        days=sorted(days, key=lambda d: d.day),
        alerts=_parse_alerts(raw.get("alerts")),
        warnings=warnings,
    )
