"""Display units. Forecast data is stored in UK units (°C, mph); conversion is display-only.

The Q&A path deliberately stays in °C and mph: its grounding checks compare numbers in
the model's answer with numbers in the forecast text, so both must use the same units.
"""

from enum import StrEnum


class UnitSystem(StrEnum):
    UK = "UK (°C, mph)"
    METRIC = "Metric (°C, km/h)"
    US = "US (°F, mph)"


KMH_PER_MPH = 1.609344


def temperature(celsius: float, system: UnitSystem) -> float:
    """Convert a °C value for display."""
    return celsius * 9 / 5 + 32 if system is UnitSystem.US else celsius


def wind(mph: float, system: UnitSystem) -> float:
    """Convert an mph value for display."""
    return mph * KMH_PER_MPH if system is UnitSystem.METRIC else mph


def temperature_unit(system: UnitSystem) -> str:
    return "°F" if system is UnitSystem.US else "°C"


def wind_unit(system: UnitSystem) -> str:
    return "km/h" if system is UnitSystem.METRIC else "mph"
