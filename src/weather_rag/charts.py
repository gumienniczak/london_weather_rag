"""Planning charts built from a parsed ``Forecast``. No Streamlit here: these are pure
functions (forecast -> table -> Altair chart), so they can be tested on their own.

Three charts, each answering a planning question:

- ``rain_heatmap``: when can I go out dry? Days x hours, coloured by rain chance.
- ``outing_chart``: which day is best for being outside in a given part of the day?
  A deterministic score from rain, wind and how far the feels-like temperature is from
  comfortable. It is a cross-check for the AI's "which evening is best?" style answers.
- ``temperature_chart``: do I need a coat? Actual vs feels-like temperature, with wind
  gusts in a separate panel on the same time axis (two charts, never two y-axes).
"""

from datetime import datetime

import altair as alt
import pandas as pd

from weather_rag.chunks import PART_WINDOWS, DayPart
from weather_rag.parse import Forecast
from weather_rag.units import UnitSystem, temperature, temperature_unit, wind, wind_unit

# Colours (validated categorical slots and a single-hue sequential ramp).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#008300"
LIGHT_GREEN = "#9ccf9c"
VIOLET = "#6250d6"
RAIN_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

# Outing score: 100 is ideal. Each penalty is in the source units (%, mph, °C).
RAIN_WEIGHT = 0.5  # points lost per % of the highest rain chance
GUST_ALLOWANCE_MPH = 15  # gusts up to this are fine
GUST_WEIGHT = 2.0  # points lost per mph above the allowance
COMFORT_C = 15  # feels-like temperature counted as comfortable
COMFORT_WEIGHT = 2.0  # points lost per °C away from comfortable
OUTING_PARTS = (DayPart.MORNING, DayPart.AFTERNOON, DayPart.EVENING)


def hourly_frame(forecast: Forecast, system: UnitSystem = UnitSystem.UK) -> pd.DataFrame:
    """One row per forecast hour, converted to ``system`` units for display."""
    rows = []
    for day in forecast.days:
        for hour in day.hours:
            rows.append(
                {
                    "time": datetime.combine(day.day, hour.at),
                    "day": f"{day.day:%a %d}",
                    "hour": hour.at.hour,
                    "rain": hour.precipprob,
                    "temp": None if hour.temp is None else temperature(hour.temp, system),
                    "feels": None
                    if hour.feelslike is None
                    else temperature(hour.feelslike, system),
                    "gust": None if hour.windgust is None else wind(hour.windgust, system),
                }
            )
    return pd.DataFrame(rows, columns=["time", "day", "hour", "rain", "temp", "feels", "gust"])


def outing_scores(forecast: Forecast, part: DayPart) -> pd.DataFrame:
    """Score each day's ``part`` from 0 (stay in) to 100 (ideal), with the inputs used.

    Missing values simply don't add a penalty; days with no hours in the period are left out.
    """
    start, length = PART_WINDOWS[part]
    rows = []
    for day in forecast.days:
        hours = [h for h in day.hours if start <= h.at.hour < start + length]
        if not hours:
            continue
        rain = max((h.precipprob for h in hours if h.precipprob is not None), default=None)
        gust = max((h.windgust for h in hours if h.windgust is not None), default=None)
        feels_values = [h.feelslike for h in hours if h.feelslike is not None]
        feels = sum(feels_values) / len(feels_values) if feels_values else None

        penalty = 0.0
        if rain is not None:
            penalty += rain * RAIN_WEIGHT
        if gust is not None:
            penalty += max(0.0, gust - GUST_ALLOWANCE_MPH) * GUST_WEIGHT
        if feels is not None:
            penalty += abs(feels - COMFORT_C) * COMFORT_WEIGHT
        rows.append(
            {
                "day": f"{day.day:%a %d}",
                "score": round(max(0.0, min(100.0, 100 - penalty))),
                "rain": rain,
                "gust": gust,
                "feels": None if feels is None else round(feels, 1),
            }
        )
    return pd.DataFrame(rows, columns=["day", "score", "rain", "gust", "feels"])


def best_day(scores: pd.DataFrame) -> str | None:
    """The day with the highest score (earliest wins a tie), or None if there are no rows."""
    if scores.empty:
        return None
    return str(scores.loc[scores["score"].idxmax(), "day"])


def _day_order(frame: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(frame["day"]))


def rain_heatmap(forecast: Forecast) -> alt.Chart:
    """Days (rows) x hours (columns), darker = higher chance of rain."""
    frame = hourly_frame(forecast)
    chart: alt.Chart = (
        alt.Chart(frame, title="Chance of rain by hour (darker = more likely)")
        .mark_rect(stroke="white", strokeWidth=2, cornerRadius=2)
        .encode(
            x=alt.X("hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("day:O", title=None, sort=_day_order(frame)),
            color=alt.Color(
                "rain:Q",
                title="Rain chance (%)",
                scale=alt.Scale(domain=[0, 100], range=RAIN_RAMP),
            ),
            tooltip=[
                alt.Tooltip("day:N", title="Day"),
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("rain:Q", title="Rain chance (%)", format=".0f"),
            ],
        )
        .properties(height=260)
    )
    return chart


def outing_chart(
    scores: pd.DataFrame, part: DayPart, system: UnitSystem = UnitSystem.UK
) -> alt.Chart:
    """One bar per day: how good ``part`` looks for being outside (0-100).

    Untitled: the app shows the title above its part-of-day selector. The score is always
    computed in source units; only the tooltip values are converted.
    """
    t_unit, w_unit = temperature_unit(system), wind_unit(system)

    best = best_day(scores)

    shown = scores.assign(
        rank=["Best" if day == best else "Other" for day in scores["day"]],
        gust=[None if pd.isna(g) else wind(g, system) for g in scores["gust"]],
        feels=[None if pd.isna(f) else temperature(f, system) for f in scores["feels"]],
    )
    chart: alt.Chart = (
        alt.Chart(shown)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("day:O", title=None, sort=_day_order(scores), axis=alt.Axis(labelAngle=0)),
            y=alt.Y("score:Q", title="Score (0-100)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "rank:N",
                scale=alt.Scale(domain=["Best", "Other"], range=[GREEN, LIGHT_GREEN]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("day:N", title="Day"),
                alt.Tooltip("score:Q", title="Score"),
                alt.Tooltip("rain:Q", title="Highest rain chance (%)", format=".0f"),
                alt.Tooltip("gust:Q", title=f"Strongest gust ({w_unit})", format=".0f"),
                alt.Tooltip("feels:Q", title=f"Average feels-like ({t_unit})", format=".1f"),
            ],
        )
        .properties(height=260)
    )
    return chart


def temperature_chart(forecast: Forecast, system: UnitSystem = UnitSystem.UK) -> alt.VConcatChart:
    """Actual vs feels-like temperature, with gusts in a second panel on the same time axis."""
    frame = hourly_frame(forecast, system)
    t_unit, w_unit = temperature_unit(system), wind_unit(system)
    long = frame.melt(
        id_vars=["time"], value_vars=["temp", "feels"], var_name="series", value_name="value"
    ).replace({"series": {"temp": "Actual", "feels": "Feels like"}})
    shared_x = alt.X("time:T", title=None, axis=alt.Axis(format="%a %H:%M"))

    temps = (
        alt.Chart(long, title=f"Temperature ({t_unit})")
        .mark_line(strokeWidth=2)
        .encode(
            x=shared_x,
            y=alt.Y("value:Q", title=t_unit),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(domain=["Actual", "Feels like"], range=[BLUE, ORANGE]),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%a %d %b, %H:%M"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title=t_unit, format=".1f"),
            ],
        )
        .properties(height=220)
    )
    gusts = (
        alt.Chart(frame, title=f"Wind gusts ({w_unit})")
        .mark_line(strokeWidth=2, color=VIOLET)
        .encode(
            x=shared_x,
            y=alt.Y("gust:Q", title=w_unit),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%a %d %b, %H:%M"),
                alt.Tooltip("gust:Q", title=f"Gust ({w_unit})", format=".0f"),
            ],
        )
        .properties(height=140)
    )
    combined: alt.VConcatChart = (
        alt.vconcat(temps, gusts).resolve_scale(x="shared").resolve_legend(color="independent")
    )
    return combined
