"""Direct Tavily API integration with bounded retries and a safe fallback."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from tools.http_client import ExternalAPIError, get_json


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def tavily_search(query: str, max_results: int = 5) -> str:
    """Return concise web-search results without depending on an MCP gateway."""

    if not TAVILY_API_KEY:
        return "Hotel web research is unavailable because TAVILY_API_KEY is not configured."

    try:
        data = get_json(
            TAVILY_SEARCH_URL,
            method="POST",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max(1, min(max_results, 10)),
                "search_depth": "basic",
            },
        )
    except ExternalAPIError:
        return "Hotel web research is temporarily unavailable. Please verify options with a booking provider."

    results = data.get("results") or []
    if not results:
        return "No hotel research results were returned. Please verify options with a booking provider."

    formatted = []
    for index, result in enumerate(results, start=1):
        title = str(result.get("title") or "Untitled result")
        url = str(result.get("url") or "")
        snippet = str(result.get("content") or "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."
        formatted.append(f"{index}. {title}\n{url}\n{snippet}")

    return "\n\n".join(formatted)

