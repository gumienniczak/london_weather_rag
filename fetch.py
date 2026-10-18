"""Fetch the raw 7-day London forecast from the Visual Crossing Timeline API.

This module only talks to the API: it builds the request, sends it, and returns the
raw JSON object or raises a typed ``ForecastError``. Interpreting the data is parse.py's job.

Security note: the API key travels as a query parameter, so request URLs contain it.
Error messages are therefore written by hand from the status code, and the original
``requests`` exception (whose message includes the URL) is never chained or shown.
"""

from datetime import date, timedelta
from typing import Any

import requests

from weather_rag.config import get_weather_api_key
from weather_rag.errors import ForecastApiError, ForecastParseError, ForecastUnavailableError

BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
LONDON = "51.5072,-0.1276"  # coordinates, so "London" can't resolve to London, Ontario
FORECAST_DAYS = 7
DEFAULT_TIMEOUT_SECONDS = 10.0
ELEMENTS = (
    "datetime,conditions,icon,description,tempmax,tempmin,temp,feelslike,feelslikemax,"
    "feelslikemin,windspeed,windgust,precip,precipprob,preciptype,sunrise,sunset,uvindex"
)

_STATUS_MESSAGES = {
    400: "The weather service rejected the request (400).",
    401: "The weather API key was rejected (401). Check VISUAL_CROSSING_API_KEY in .env.",
    429: "The weather service's daily limit has been reached (429). Try again later.",
}


def build_url(start: date) -> str:
    """Return the Timeline API URL for 7 days starting at ``start``.

    The API's end date is inclusive, so the range ends at start + 6 days.
    """
    end = start + timedelta(days=FORECAST_DAYS - 1)  # the API's end date is inclusive
    return f"{BASE_URL}/{LONDON}/{start.isoformat()}/{end.isoformat()}"


def _error_message(status_code: int) -> str:
    """Return error message matched with status_code."""
    if status_code in _STATUS_MESSAGES:
        return _STATUS_MESSAGES[status_code]
    if status_code >= 500:
        return f"The weather service is having problems ({status_code}). Try again later."
    return f"The weather service returned an unexpected error ({status_code})."


def fetch_forecast(
    *,
    api_key: str | None = None,
    today: date | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Return the raw Timeline API response for the next 7 days, starting today."""
    params = {
        "key": api_key or get_weather_api_key(),
        "unitGroup": "uk",
        "include": "days,hours,alerts",
        "elements": ELEMENTS,
    }
    url = build_url(today or date.today())

    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.Timeout:
        raise ForecastUnavailableError("The weather service took too long to respond.") from None
    except requests.RequestException:
        raise ForecastUnavailableError(
            "Couldn't reach the weather service. Check your internet connection."
        ) from None

    if not response.ok:
        raise ForecastApiError(_error_message(response.status_code), response.status_code)

    try:
        data = response.json()
    except requests.JSONDecodeError:
        raise ForecastParseError("The weather service returned data that isn't JSON.") from None

    if not isinstance(data, dict):
        raise ForecastParseError("The weather service returned data in an unexpected shape.")
    return data
