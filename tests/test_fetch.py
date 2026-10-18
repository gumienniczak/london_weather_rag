from datetime import date

import pytest
import requests
import responses
from responses import matchers

from weather_rag.errors import (
    ForecastApiError,
    ForecastError,
    ForecastParseError,
    ForecastUnavailableError,
    MissingApiKeyError,
)
from weather_rag.fetch import build_url, fetch_forecast

TODAY = date(2026, 10, 18)
KEY = "test-secret-key"
URL = build_url(TODAY)


def test_url_covers_exactly_seven_days() -> None:
    assert URL.endswith("/51.5072,-0.1276/2026-10-18/2026-10-24")


@responses.activate
def test_success_returns_raw_dict_and_sends_expected_params() -> None:
    responses.get(
        URL,
        json={"days": []},
        match=[
            matchers.query_param_matcher(
                {"key": KEY, "unitGroup": "uk", "include": "days,hours,alerts"},
                strict_match=False,
            )
        ],
    )

    assert fetch_forecast(api_key=KEY, today=TODAY) == {"days": []}


@responses.activate
@pytest.mark.parametrize(
    "error", [requests.Timeout(), requests.ConnectionError(f"failed: {URL}?key={KEY}")]
)
def test_network_problems_raise_unavailable(error: Exception) -> None:
    responses.get(URL, body=error)

    with pytest.raises(ForecastUnavailableError) as exc_info:
        fetch_forecast(api_key=KEY, today=TODAY)

    # The original requests error (whose message holds the URL + key) must not be chained.
    assert KEY not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__


@responses.activate
@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_error_status_raises_api_error_without_leaking_key(status: int) -> None:
    responses.get(URL, status=status, body="error")

    with pytest.raises(ForecastApiError) as exc_info:
        fetch_forecast(api_key=KEY, today=TODAY)

    assert exc_info.value.status_code == status
    assert KEY not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@responses.activate
@pytest.mark.parametrize("body", ["<html>not json</html>", "[1, 2, 3]"])
def test_unusable_body_raises_parse_error(body: str) -> None:
    responses.get(URL, body=body, content_type="application/json")

    with pytest.raises(ForecastParseError):
        fetch_forecast(api_key=KEY, today=TODAY)


def test_missing_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VISUAL_CROSSING_API_KEY", raising=False)

    with pytest.raises(MissingApiKeyError, match="README"):
        fetch_forecast(today=TODAY)


def test_all_fetch_errors_share_one_base_class() -> None:
    for error in (ForecastApiError, ForecastUnavailableError, MissingApiKeyError):
        assert issubclass(error, ForecastError)
