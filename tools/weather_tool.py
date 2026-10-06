"""Direct OpenWeather integration; no local process or MCP server is required."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from tools.http_client import ExternalAPIError, get_json


load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5"


def _params(city: str) -> dict[str, str]:
    if not city or not city.strip():
        raise ValueError("A destination is required to look up weather.")
    if not OPENWEATHER_API_KEY:
        raise ExternalAPIError("OpenWeather is not configured.")
    return {"q": city.strip(), "appid": OPENWEATHER_API_KEY, "units": "metric"}


def get_current_weather(city: str) -> dict[str, Any]:
    """Fetch a compact current-weather response from OpenWeather."""

    data = get_json(f"{OPENWEATHER_URL}/weather", params=_params(city))
    weather = data.get("weather") or [{}]
    main = data.get("main") or {}
    wind = data.get("wind") or {}
    return {
        "city": data.get("name", city),
        "temperature_c": main.get("temp", "N/A"),
        "feels_like_c": main.get("feels_like", "N/A"),
        "humidity": main.get("humidity", "N/A"),
        "condition": weather[0].get("description", "Unknown"),
        "wind_speed_mps": wind.get("speed", "N/A"),
    }


def get_forecast(city: str) -> dict[str, Any]:
    """Fetch the next five three-hour forecast points from OpenWeather."""

    data = get_json(f"{OPENWEATHER_URL}/forecast", params=_params(city))
    forecast = []
    for item in (data.get("list") or [])[:5]:
        weather = item.get("weather") or [{}]
        forecast.append(
            {
                "datetime": item.get("dt_txt", "Unknown"),
                "temperature_c": (item.get("main") or {}).get("temp", "N/A"),
                "condition": weather[0].get("description", "Unknown"),
            }
        )
    return {"city": (data.get("city") or {}).get("name", city), "forecast": forecast}
