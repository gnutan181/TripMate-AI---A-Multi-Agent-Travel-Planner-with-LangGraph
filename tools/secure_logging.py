"""Redact credentials before formatting application and third-party logs."""
import logging
import os
import re
from urllib.parse import quote, quote_plus

_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:tavilyApiKey|api[_-]?key|access[_-]?key|appid|token)=)[^&#\s\"']*"
)
_BEARER = re.compile(r"(?i)(Bearer\s+)[^\s\"']+")
_USERINFO = re.compile(r"(://)[^/\s:@]+:[^/@\s]+@")


def redact(value) -> str:
    text = str(value)
    for name, secret in os.environ.items():
        if secret and (name.endswith(("_KEY", "_TOKEN", "_PASSWORD")) or name == "DATABASE_URL"):
            for encoded in {secret, quote(secret, safe=""), quote_plus(secret)}:
                text = text.replace(encoded, "[REDACTED]")
    text = _QUERY_SECRET.sub(r"\1[REDACTED]", text)
    text = _BEARER.sub(r"\1[REDACTED]", text)
    return _USERINFO.sub(r"\1[REDACTED]@", text)


def install_log_redaction():
    # A record factory also covers third-party loggers with their own handlers.
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_tripmate_redaction", False):
        return

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        # Uvicorn's AccessFormatter unpacks the original five arguments.
        # Preserve lazy formatting and numeric values (including %d fields).
        record.msg = redact(record.msg)
        if isinstance(record.args, dict):
            record.args = {key: _redact_argument(value) for key, value in record.args.items()}
        else:
            record.args = tuple(_redact_argument(value) for value in record.args)
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return record

    factory._tripmate_redaction = True
    logging.setLogRecordFactory(factory)


def _redact_argument(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(value)
