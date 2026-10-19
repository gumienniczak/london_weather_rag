from typing import Any

import pytest

from factories import make_forecast
from weather_rag.charts import (
    best_day,
    hourly_frame,
    outing_chart,
    outing_scores,
    rain_heatmap,
    temperature_chart,
)
from weather_rag.chunks import DayPart
from weather_rag.units import UnitSystem, temperature, wind


def test_unit_conversions() -> None:
    assert temperature(0, UnitSystem.US) == 32
    assert temperature(10, UnitSystem.METRIC) == 10
    assert wind(10, UnitSystem.METRIC) == pytest.approx(16.09, abs=0.01)
    assert wind(10, UnitSystem.US) == 10


def test_hourly_frame_has_one_row_per_hour_in_display_units() -> None:
    frame = hourly_frame(make_forecast(days=2), UnitSystem.US)

    assert len(frame) == 48
    assert frame["temp"].iloc[0] == pytest.approx(53.6)  # 12°C
    assert list(frame["day"].unique()) == ["Sun 18", "Mon 19"]


def test_outing_score_prefers_the_dry_calm_mild_day() -> None:
    wet_windy_monday = {(1, h): {"precipprob": 90.0, "windgust": 40.0} for h in range(17, 22)}
    scores = outing_scores(make_forecast(days=3, hour_overrides=wet_windy_monday), DayPart.EVENING)

    by_day = dict(zip(scores["day"], scores["score"], strict=True))
    assert by_day["Mon 19"] < by_day["Sun 18"]
    assert best_day(scores) == "Sun 18"  # tie with Tue goes to the earlier day


def test_outing_score_formula_is_as_documented() -> None:
    # 10% rain -> -5; gust 15 mph -> 0; feels 10°C (5 below comfortable) -> -10
    scores = outing_scores(make_forecast(days=1), DayPart.MORNING)

    assert scores["score"].tolist() == [85]


def test_missing_values_add_no_penalty() -> None:
    missing = {(0, h): {"precipprob": None, "windgust": None} for h in range(6, 12)}
    scores = outing_scores(make_forecast(days=1, hour_overrides=missing), DayPart.MORNING)

    assert scores["score"].tolist() == [90]


def test_best_day_of_empty_scores_is_none() -> None:
    empty = outing_scores(make_forecast(days=1), DayPart.EVENING).iloc[0:0]

    assert best_day(empty) is None


def spec(chart: Any) -> dict[str, Any]:
    result: dict[str, Any] = chart.to_dict()
    return result


def test_heatmap_maps_rain_to_colour_on_a_fixed_0_100_scale() -> None:
    encoding = spec(rain_heatmap(make_forecast()))["encoding"]

    assert encoding["color"]["field"] == "rain"
    assert encoding["color"]["scale"]["domain"] == [0, 100]
    assert encoding["tooltip"]


def test_outing_chart_has_tooltips_and_fixed_scale() -> None:
    scores = outing_scores(make_forecast(), DayPart.EVENING)
    encoding = spec(outing_chart(scores, DayPart.EVENING))["encoding"]

    assert encoding["y"]["scale"]["domain"] == [0, 100]
    assert {t["field"] for t in encoding["tooltip"]} >= {"score", "rain", "gust", "feels"}


def test_temperature_chart_uses_two_panels_not_two_axes() -> None:
    panels = spec(temperature_chart(make_forecast(), UnitSystem.US))["vconcat"]

    assert len(panels) == 2
    assert panels[0]["title"] == "Temperature (°F)"
    assert panels[1]["title"] == "Wind gusts (mph)"


def test_outing_tooltip_uses_display_units_but_score_does_not_change() -> None:
    scores = outing_scores(make_forecast(days=1), DayPart.MORNING)
    us = spec(outing_chart(scores, DayPart.MORNING, UnitSystem.US))
    metric = spec(outing_chart(scores, DayPart.MORNING, UnitSystem.METRIC))

    titles = {t["field"]: t["title"] for t in us["encoding"]["tooltip"]}
    assert titles["feels"] == "Average feels-like (°F)"
    assert {t["field"]: t["title"] for t in metric["encoding"]["tooltip"]}["gust"] == (
        "Strongest gust (km/h)"
    )
    row = next(iter(us["datasets"].values()))[0]
    assert row["feels"] == pytest.approx(50.0)  # 10°C shown as °F
    assert row["score"] == scores["score"].iloc[0]  # score unchanged by units
