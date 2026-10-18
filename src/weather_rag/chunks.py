"""Turn a parsed ``Forecast`` into small, citable text chunks for retrieval.

Each forecast chunk covers one part of one day (morning, afternoon, evening, night).
Its text is a short summary computed from the hours in that period, followed by a
compact hourly line, so retrieval works on periods (how people ask) while the model
still sees hour-level detail. Night runs 22:00-06:00 and borrows the next day's early
hours; when the data stops early (the last night), the chunk says it is partial.

Missing values are written as "not available" rather than left out, so the model is
told what it doesn't know instead of being left to fill the gap.

Units follow the API's ``unitGroup=uk``: °C, mph and mm.
"""

from collections import Counter
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from weather_rag.parse import Forecast, HourForecast


class DayPart(StrEnum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


# (start hour, length in hours). Night crosses midnight into the next day.
PART_WINDOWS: dict[DayPart, tuple[int, int]] = {
    DayPart.MORNING: (6, 6),
    DayPart.AFTERNOON: (12, 5),
    DayPart.EVENING: (17, 5),
    DayPart.NIGHT: (22, 8),
}

# Thresholds for the plain-word tags that help keyword search ("wet", "windy"...).
WET_RAIN_CHANCE = 50
DRY_RAIN_CHANCE = 20
WINDY_GUST_MPH = 30
COLD_FEELSLIKE_C = 5
WARM_TEMP_C = 20

NOT_AVAILABLE = "not available"


class Chunk(BaseModel):
    """One retrievable unit of text, with the metadata needed to filter and cite it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: Literal["forecast", "guide"]
    label: str
    text: str
    day: date | None = None
    part: DayPart | None = None


def chunk_id(day: date, part: DayPart) -> str:
    """Return the stable ID for a forecast chunk, e.g. ``2026-10-23-evening``."""
    return f"{day.isoformat()}-{part.value}"


def chunk_label(day: date, part: DayPart) -> str:
    """Return a readable label for the UI, e.g. ``Thu 23 Oct, evening``."""
    label = f"{day:%a} {day.day} {day:%b}, {part.value}"
    if part is DayPart.NIGHT:
        label += f" (into {day + timedelta(days=1):%a})"
    return label


def _hours_by_datetime(forecast: Forecast) -> dict[datetime, HourForecast]:
    return {datetime.combine(day.day, hour.at): hour for day in forecast.days for hour in day.hours}


def _fmt_range(values: list[float], unit: str) -> str:
    if not values:
        return NOT_AVAILABLE
    low, high = round(min(values)), round(max(values))
    return f"{low}{unit}" if low == high else f"{low}–{high}{unit}"


def _fmt_max(values: list[float], unit: str) -> str:
    return f"{round(max(values))}{unit}" if values else NOT_AVAILABLE


def _present(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def _tags(rain: list[float], gusts: list[float], feels: list[float], temps: list[float]) -> str:
    tags: list[str] = []
    if rain and max(rain) >= WET_RAIN_CHANCE:
        tags.append("wet")
    elif rain and max(rain) < DRY_RAIN_CHANCE:
        tags.append("dry")
    if gusts and max(gusts) >= WINDY_GUST_MPH:
        tags.append("windy")
    if feels and min(feels) <= COLD_FEELSLIKE_C:
        tags.append("cold")
    if temps and max(temps) >= WARM_TEMP_C:
        tags.append("warm")
    return f" Overall: {', '.join(tags)}." if tags else ""


def _hourly_line(hours: list[tuple[datetime, HourForecast]]) -> str:
    entries = []
    for moment, hour in hours:
        temp = f"{round(hour.temp)}°C" if hour.temp is not None else "temp n/a"
        rain = f"{round(hour.precipprob)}%" if hour.precipprob is not None else "rain n/a"
        entry = f"{moment:%H:%M} {temp} {rain}"
        if hour.preciptype:
            entry += f" {'/'.join(hour.preciptype)}"
        entries.append(entry)
    return "Hourly: " + " · ".join(entries)


def _summarise(
    label: str,
    hours: list[tuple[datetime, HourForecast]],
    expected_hours: int,
    sun_note: str,
) -> str:
    values = [hour for _, hour in hours]
    temps = _present([h.temp for h in values])
    feels = _present([h.feelslike for h in values])
    rain = _present([h.precipprob for h in values])
    precip = _present([h.precip for h in values])
    winds = _present([h.windspeed for h in values])
    gusts = _present([h.windgust for h in values])
    uv = _present([h.uvindex for h in values])
    types = sorted({t for h in values for t in h.preciptype})
    conditions = Counter(h.conditions for h in values if h.conditions).most_common(1)

    start, end = hours[0][0], hours[-1][0] + timedelta(hours=1)
    amount = f", about {sum(precip):.1f} mm" if precip else ""
    type_note = f" ({', '.join(types)})" if types else ""
    partial = (
        f" Partial: forecast data only covers {start:%H:%M}–{end:%H:%M}."
        if len(hours) < expected_hours
        else ""
    )

    summary = (
        f"{label} ({start:%H:%M}–{end:%H:%M}): "
        f"temperature {_fmt_range(temps, '°C')}, feels like {_fmt_range(feels, '°C')}. "
        f"Rain chance up to {_fmt_max(rain, '%')}{amount}{type_note}. "
        f"Wind up to {_fmt_max(winds, ' mph')}, gusts up to {_fmt_max(gusts, ' mph')}. "
        f"UV index up to {_fmt_max(uv, '')}. "
        f"Mostly {conditions[0][0].lower() if conditions else NOT_AVAILABLE}."
        f"{sun_note}{_tags(rain, gusts, feels, temps)}{partial}"
    )
    return f"{summary}\n{_hourly_line(hours)}"


def forecast_chunks(forecast: Forecast) -> list[Chunk]:
    """Return one chunk per day part that has any hourly data, in time order.

    Night chunks include the next morning's hours (22:00-06:00). Periods with no
    hourly data are skipped; periods with some missing hours are marked partial.
    """
    by_time = _hours_by_datetime(forecast)
    chunks: list[Chunk] = []

    for day in forecast.days:
        for part, (start_hour, length) in PART_WINDOWS.items():
            start = datetime.combine(day.day, time(start_hour))
            window = [start + timedelta(hours=offset) for offset in range(length)]
            hours = [(moment, by_time[moment]) for moment in window if moment in by_time]
            if not hours:
                continue

            sun_note = ""
            if part is DayPart.MORNING and day.sunrise:
                sun_note = f" Sunrise {day.sunrise:%H:%M}."
            elif part is DayPart.EVENING and day.sunset:
                sun_note = f" Sunset {day.sunset:%H:%M}."

            label = chunk_label(day.day, part)
            chunks.append(
                Chunk(
                    id=chunk_id(day.day, part),
                    kind="forecast",
                    label=label,
                    text=_summarise(label, hours, length, sun_note),
                    day=day.day,
                    part=part,
                )
            )
    return chunks


def guide_chunks(markdown: str, source: str) -> list[Chunk]:
    """Split a small markdown guide into one chunk per ``## `` section.

    ``source`` names the guide and prefixes each ID, e.g. ``what-to-wear-rain``.
    """
    chunks: list[Chunk] = []
    for section in ("\n" + markdown).split("\n## ")[1:]:
        heading, _, body = section.partition("\n")
        heading, body = heading.strip(), body.strip()
        if not body:
            continue
        slug = "-".join(heading.lower().split())
        chunks.append(
            Chunk(
                id=f"{source}-{slug}",
                kind="guide",
                label=f"Guide: {heading}",
                text=f"{heading}: {body}",
            )
        )
    return chunks
