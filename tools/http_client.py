"""Small, production-safe HTTP helpers for third-party travel APIs."""

from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tools.secure_logging import install_log_redaction

install_log_redaction()


DEFAULT_TIMEOUT_SECONDS = 15


class ExternalAPIError(RuntimeError):
    """Raised when an external provider cannot return a usable response."""


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Make one retried JSON request without leaking provider details to users.

    Retries are limited to transient failures: connection/read errors, rate limits,
    and 5xx responses. Invalid requests and authentication failures fail quickly.
    """

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        response = session.request(
            method=method,
            url=url,
            params=params,
            json=json,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise ExternalAPIError("The provider timed out after retrying.") from exc
    except requests.RequestException as exc:
        raise ExternalAPIError("The provider could not be reached after retrying.") from exc
    except ValueError as exc:
        raise ExternalAPIError("The provider returned an invalid response.") from exc
    finally:
        session.close()

    if not isinstance(payload, dict):
        raise ExternalAPIError("The provider returned an unexpected response.")

    return payload
