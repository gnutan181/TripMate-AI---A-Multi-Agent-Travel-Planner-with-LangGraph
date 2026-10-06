import io
import logging
import os
import sys
import types
import unittest
from unittest.mock import patch

from deployment_checks import DeploymentConfigurationError, require_executable
from tools.secure_logging import install_log_redaction, redact


class DeploymentTests(unittest.TestCase):
    def test_uvicorn_access_formatter_preserves_arguments_and_redacts_url(self):
        from uvicorn.logging import AccessFormatter
        install_log_redaction()
        logger = logging.getLogger("uvicorn.access")
        record = logger.makeRecord(
            logger.name, logging.INFO, __file__, 0,
            '%s - "%s %s HTTP/%s" %d',
            ("127.0.0.1:12345", "GET", "/?tavilyApiKey=placeholder-secret", "1.1", 200),
            None,
        )
        self.assertEqual(len(record.args), 5)
        self.assertEqual(record.args[-1], 200)
        formatter = AccessFormatter(
            '%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False,
        )
        formatted = formatter.format(record)
        self.assertIn("200 OK", formatted)
        self.assertIn("GET /?tavilyApiKey=[REDACTED] HTTP/1.1", formatted)
        self.assertNotIn("placeholder-secret", formatted)

    def test_mapping_arguments_preserve_numeric_formatting(self):
        install_log_redaction()
        record = logging.getLogger("httpx").makeRecord(
            "httpx", logging.INFO, __file__, 0, "%(url)s %(status)d",
            ({"url": "https://example.test/?api_key=placeholder-secret", "status": 200},), None,
        )
        self.assertEqual(record.getMessage(), "https://example.test/?api_key=[REDACTED] 200")

    def test_missing_executable_identifies_server(self):
        with patch("deployment_checks.shutil.which", return_value=None):
            with self.assertRaisesRegex(DeploymentConfigurationError, "aviationstack.*uvx.*PATH"):
                require_executable("aviationstack", "uvx")

    def test_url_credentials_redacted_without_environment(self):
        url = "https://mcp.tavily.com/mcp/?tavilyApiKey=placeholder-secret&x=1"
        self.assertEqual(redact(url), "https://mcp.tavily.com/mcp/?tavilyApiKey=[REDACTED]&x=1")
        self.assertNotIn("placeholder-secret", redact("https://example.test/?appid=placeholder-secret"))

    def test_httpx_and_exception_logs_redact_credentials(self):
        install_log_redaction()
        output = io.StringIO()
        logger = logging.getLogger("httpx")
        handler = logging.StreamHandler(output)
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            with patch.dict(os.environ, {"TAVILY_API_KEY": "placeholder-secret"}):
                logger.info('HTTP Request: GET %s "HTTP/1.1 200 OK"',
                            "https://mcp.tavily.com/?tavilyApiKey=placeholder-secret")
                try:
                    raise RuntimeError("Bearer placeholder-secret")
                except RuntimeError:
                    logger.exception("request failed")
            self.assertNotIn("placeholder-secret", output.getvalue())
            self.assertIn("[REDACTED]", output.getvalue())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

    def test_api_real_schema_with_mocked_backend(self):
        # No real DB, LLM or provider request: only the actual FastAPI handler.
        from fastapi.testclient import TestClient
        backend = types.ModuleType("backend")
        backend.run_travel_agent = lambda **kwargs: {"answer": "mock draft", "thread_id": kwargs["thread_id"]}
        backend.resume_travel_agent = lambda **kwargs: {}
        with patch.dict(sys.modules, {"backend": backend}):
            sys.modules.pop("app", None)
            import app
            client = TestClient(app.app)
            response = client.post("/api/travel", json={"message": "Plan hotels in Tokyo", "thread_id": "test-thread"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["answer"], "mock draft")
            with patch.object(app, "run_travel_agent", side_effect=DeploymentConfigurationError("Server 'aviationstack': executable 'uvx' missing")):
                response = client.post("/api/travel", json={"message": "Hotels in Tokyo"})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["code"], "deployment_configuration_error")
            self.assertEqual(client.post("/api/travel", json={"query": "Tokyo"}).status_code, 422)
            with patch.object(app, "run_travel_agent", side_effect=RuntimeError("placeholder-secret")):
                response = client.post("/api/travel", json={"message": "Tokyo"})
                self.assertEqual(response.status_code, 500)
                self.assertNotIn("placeholder-secret", response.text)
        sys.modules.pop("app", None)


if __name__ == "__main__":
    unittest.main()
