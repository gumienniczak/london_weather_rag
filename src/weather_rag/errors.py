"""Exceptions shared across the package."""


class ForecastError(Exception):
    """Base class for anything that stops us producing a forecast."""


class MissingApiKeyError(ForecastError):
    """The weather API key is not configured."""


class ForecastUnavailableError(ForecastError):
    """The weather service could not be reached (timeout, no connection)."""


class ForecastApiError(ForecastError):
    """The weather service answered with an error status."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class ForecastParseError(ForecastError):
    """The API response could not be turned into a usable forecast."""
