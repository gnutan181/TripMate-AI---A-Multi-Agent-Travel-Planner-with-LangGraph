import unittest
from unittest.mock import Mock, patch

import backend
from langchain_core.messages import HumanMessage
from tools.http_client import ExternalAPIError
from tools import http_client
from tools.flight_tool import fetch_flights
from tools.tavily_tool import tavily_search
from tools.weather_tool import get_current_weather, get_forecast


class DirectAPIIntegrationTests(unittest.TestCase):
    def test_groq_invocation_binds_a_budgeted_output_limit(self):
        bound_llm = Mock()
        bound_llm.invoke.return_value = Mock(content="ok")

        with patch.object(type(backend.llm), "bind", return_value=bound_llm) as bind:
            response = backend._invoke_llm(
                [HumanMessage(content="Plan a trip to Tokyo")],
                preferred_output_tokens=99_999,
            )

        self.assertEqual(response.content, "ok")
        requested_output = bind.call_args.kwargs["max_tokens"]
        self.assertLessEqual(requested_output, backend.GROQ_MAX_TOKENS)
        self.assertLessEqual(requested_output, backend.GROQ_REQUEST_TOKEN_BUDGET)

    def test_http_client_configures_bounded_transient_failure_retries(self):
        response = Mock()
        response.json.return_value = {}
        session = Mock()
        session.request.return_value = response

        with patch("tools.http_client.requests.Session", return_value=session):
            self.assertEqual(http_client.get_json("https://example.test"), {})

        adapters = [call.args[1] for call in session.mount.call_args_list]
        self.assertTrue(all(adapter.max_retries.total == 3 for adapter in adapters))
        self.assertIn(429, adapters[0].max_retries.status_forcelist)
        session.close.assert_called_once()

    def test_weather_current_response_is_normalized(self):
        with patch("tools.weather_tool.get_json") as request:
            request.return_value = {
                "name": "Tokyo",
                "main": {"temp": 22.5, "feels_like": 23, "humidity": 61},
                "weather": [{"description": "clear sky"}],
                "wind": {"speed": 3.1},
            }
            with patch("tools.weather_tool.OPENWEATHER_API_KEY", "test-key"):
                result = get_current_weather("Tokyo")

        self.assertEqual(result["city"], "Tokyo")
        self.assertEqual(result["condition"], "clear sky")
        self.assertEqual(result["temperature_c"], 22.5)

    def test_weather_forecast_is_limited_to_five_entries(self):
        forecast_entries = [
            {
                "dt_txt": f"2026-01-01 {hour:02d}:00:00",
                "main": {"temp": hour},
                "weather": [{"description": "cloudy"}],
            }
            for hour in range(6)
        ]
        with patch("tools.weather_tool.get_json", return_value={"list": forecast_entries}):
            with patch("tools.weather_tool.OPENWEATHER_API_KEY", "test-key"):
                result = get_forecast("Tokyo")

        self.assertEqual(len(result["forecast"]), 5)

    def test_tavily_provider_failure_returns_a_safe_fallback(self):
        with patch("tools.tavily_tool.TAVILY_API_KEY", "test-key"):
            with patch(
                "tools.tavily_tool.get_json",
                side_effect=ExternalAPIError("provider unavailable"),
            ):
                result = tavily_search("Best hotels in Tokyo")

        self.assertIn("temporarily unavailable", result)

    def test_aviationstack_error_uses_a_safe_fallback(self):
        with patch("tools.flight_tool.AVIATIONSTACK_API_KEY", "test-key"):
            with patch(
                "tools.flight_tool.get_json",
                return_value={"error": {"code": "rate_limit", "message": "quota exceeded"}},
            ):
                with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
                    fetch_flights("DAC", "NRT")


if __name__ == "__main__":
    unittest.main()
