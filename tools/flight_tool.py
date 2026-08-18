
"""

Flow:

    User Query
        ↓
    Groq LLM
        ↓
    Structured RouteDecision
        ↓
    IATA validation using airportsdata
        ↓
    AviationStack API
        ↓
    Formatted flight results

"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import airportsdata
import certifi
import requests
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ============================================================================
# ENVIRONMENT
# ============================================================================

load_dotenv()

# SSL configuration
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


# ============================================================================
# CONFIGURATION
# ============================================================================

AVIATIONSTACK_API_KEY = os.getenv(
    "AVIATIONSTACK_API_KEY"
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

DEFAULT_ORIGIN_IATA = os.getenv(
    "DEFAULT_ORIGIN_IATA",
    "DAC",
).strip().upper()


AVIATIONSTACK_URL = (
    "https://api.aviationstack.com/v1/flights"
)

REQUEST_TIMEOUT = 30

DEFAULT_LIMIT = 10

MAX_LIMIT = 100

MAX_QUERY_LENGTH = 2000


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("aviation_service")


# ============================================================================
# AIRPORT DATABASE
# ============================================================================

AIRPORTS = airportsdata.load("IATA")


# ============================================================================
# GROQ CLIENT
# ============================================================================

groq_client: Optional[Groq] = None

if GROQ_API_KEY:
    groq_client = Groq(
        api_key=GROQ_API_KEY
    )


# ============================================================================
# STRUCTURED ROUTE MODEL
# ============================================================================

class RouteDecision(BaseModel):
    """
    Structured response expected from Groq.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    intent: str = Field(
        description=(
            "Flight search intent. "
            "Must be one of: "
            "route, origin_only, destination_only, "
            "global, unknown."
        )
    )

    origin: Optional[str] = Field(
        default=None,
        description=(
            "Natural language origin. "
            "Example: Bangladesh, Mumbai, DAC."
        ),
    )

    origin_iata: Optional[str] = Field(
        default=None,
        description=(
            "Three-letter IATA airport code for origin. "
            "Null if origin is absent."
        ),
    )

    destination: Optional[str] = Field(
        default=None,
        description=(
            "Natural language destination. "
            "Example: Japan, Tokyo, NRT."
        ),
    )

    destination_iata: Optional[str] = Field(
        default=None,
        description=(
            "Three-letter IATA airport code for destination. "
            "Null if destination is absent."
        ),
    )


# ============================================================================
# ROUTE RESULT
# ============================================================================

@dataclass(frozen=True)
class ResolvedRoute:
    origin: Optional[str]
    destination: Optional[str]


# ============================================================================
# GROQ SYSTEM PROMPT
# ============================================================================
 
ROUTE_SYSTEM_PROMPT = """
You are a flight-search route extraction engine.

Your ONLY job is to analyze the user's flight-search request.

Return:

1. intent
2. origin
3. origin_iata
4. destination
5. destination_iata

Do NOT answer the user.
Do NOT provide flight information.
Do NOT provide explanations.

INTENTS
=======

route
-----
Both origin and destination are specified.

origin_only
-----------
Only the departure/origin location is specified.

destination_only
----------------
Only the destination is specified.

global
------
The user explicitly requests:
- all flights
- worldwide flights
- global flights
- all live flights

unknown
-------
The request is not clear enough to determine a flight search.


LOCATION RULES
==============

The user may provide:

- country
- city
- airport
- IATA code
- natural language travel request

You must extract the actual location.

You may select a major international airport when
the user specifies an entire country.

Use practical major international airports.

Examples:

Bangladesh -> DAC
Japan -> NRT
India -> DEL
United Kingdom -> LHR
United Arab Emirates -> DXB
Singapore -> SIN
Thailand -> BKK
Nepal -> KTM
Qatar -> DOH
France -> CDG
Germany -> FRA
Australia -> SYD
Canada -> YYZ
Italy -> FCO
Spain -> MAD
Turkey -> IST

Examples of cities:

Dhaka -> DAC
Mumbai -> BOM
Delhi -> DEL
New Delhi -> DEL
Kolkata -> CCU
Chennai -> MAA
Bangalore -> BLR
Bengaluru -> BLR
Tokyo -> NRT
Osaka -> KIX
New York -> JFK
London -> LHR
Dubai -> DXB
Singapore -> SIN
Bangkok -> BKK
Doha -> DOH
Istanbul -> IST
Toronto -> YYZ
Sydney -> SYD
Paris -> CDG
Rome -> FCO
Madrid -> MAD
Frankfurt -> FRA


IMPORTANT
=========

If the user explicitly provides an IATA code,
preserve that code.

IATA codes must be exactly three letters.

Never invent an IATA code for an unknown location.

If a location does not exist or cannot reasonably be resolved,
return null for its IATA code.


EXAMPLES
========

User:
Plan a 7 days Japan trip from Bangladesh

Return:

{
    "intent": "route",
    "origin": "Bangladesh",
    "origin_iata": "DAC",
    "destination": "Japan",
    "destination_iata": "NRT"
}


User:
Flights from Mumbai to Tokyo

Return:

{
    "intent": "route",
    "origin": "Mumbai",
    "origin_iata": "BOM",
    "destination": "Tokyo",
    "destination_iata": "NRT"
}


User:
Show flights to Dubai

Return:

{
    "intent": "destination_only",
    "origin": null,
    "origin_iata": null,
    "destination": "Dubai",
    "destination_iata": "DXB"
}


User:
Show flights from Delhi

Return:

{
    "intent": "origin_only",
    "origin": "Delhi",
    "origin_iata": "DEL",
    "destination": null,
    "destination_iata": null
}


User:
Show all live flights worldwide

Return:

{
    "intent": "global",
    "origin": null,
    "origin_iata": null,
    "destination": null,
    "destination_iata": null
}


Return ONLY the structured JSON.
"""


# ============================================================================
# GROQ ROUTE EXTRACTION
# ============================================================================

def extract_route_with_llm(
    query: str,
) -> RouteDecision:
    """
    Extract origin/destination and IATA codes using Groq.
    """

    if not GROQ_API_KEY or groq_client is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add GROQ_API_KEY to your .env file."
        )

    if not query or not query.strip():
        raise ValueError(
            "Flight query cannot be empty."
        )

    query = query.strip()

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"Flight query is too long. "
            f"Maximum length is {MAX_QUERY_LENGTH} characters."
        )

    logger.info(
        "Extracting flight route using Groq"
    )

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": ROUTE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],

            temperature=0,

            max_tokens=300,

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "flight_route_decision",

                    "strict": True,

                    "schema": {
                        "type": "object",

                        "properties": {

                            "intent": {
                                "type": "string",
                                "enum": [
                                    "route",
                                    "origin_only",
                                    "destination_only",
                                    "global",
                                    "unknown",
                                ],
                            },

                            "origin": {
                                "type": [
                                    "string",
                                    "null",
                                ],
                            },

                            "origin_iata": {
                                "type": [
                                    "string",
                                    "null",
                                ],
                            },

                            "destination": {
                                "type": [
                                    "string",
                                    "null",
                                ],
                            },

                            "destination_iata": {
                                "type": [
                                    "string",
                                    "null",
                                ],
                            },
                        },

                        "required": [
                            "intent",
                            "origin",
                            "origin_iata",
                            "destination",
                            "destination_iata",
                        ],

                        "additionalProperties": False,
                    },
                },
            },
        )

    except Exception as exc:

        logger.exception(
            "Groq route extraction failed"
        )

        raise RuntimeError(
            "Unable to determine the flight route "
            "using the language model."
        ) from exc

    content = (
        response.choices[0]
        .message
        .content
    )

    if not content:
        raise RuntimeError(
            "Groq returned an empty route decision."
        )

    try:

        parsed = json.loads(
            content
        )

        decision = RouteDecision.model_validate(
            parsed
        )

    except (
        json.JSONDecodeError,
        ValidationError,
    ) as exc:

        logger.exception(
            "Invalid structured response from Groq: %s",
            content,
        )

        raise RuntimeError(
            "Groq returned an invalid route decision."
        ) from exc

    logger.info(
        (
            "LLM route decision: "
            "intent=%s "
            "origin=%s "
            "origin_iata=%s "
            "destination=%s "
            "destination_iata=%s"
        ),
        decision.intent,
        decision.origin,
        decision.origin_iata,
        decision.destination,
        decision.destination_iata,
    )

    return decision


# ============================================================================
# IATA VALIDATION
# ============================================================================

def validate_iata_code(
    iata: Optional[str],
) -> Optional[str]:
    """
    Validate an IATA code against the real airport database.

    The LLM is NOT trusted.

    Only an IATA code existing inside airportsdata
    is accepted.
    """

    if not iata:
        return None

    iata = iata.strip().upper()

    # Must be exactly 3 alphabetic characters.
    if not re.fullmatch(
        r"[A-Z]{3}",
        iata,
    ):
        return None

    # Must exist in airportsdata.
    if iata not in AIRPORTS:
        return None

    return iata


# ============================================================================
# ROUTE VALIDATION
# ============================================================================

def resolve_route(
    decision: RouteDecision,
) -> ResolvedRoute:
    """
    Validate the route selected by Groq.

    IMPORTANT:

    Groq determines the semantic route.

    airportsdata validates the final IATA codes.

    AviationStack is only called after validation.
    """

    intent = decision.intent

    # ------------------------------------------------------------------------
    # GLOBAL
    # ------------------------------------------------------------------------

    if intent == "global":

        return ResolvedRoute(
            origin=None,
            destination=None,
        )

    # ------------------------------------------------------------------------
    # UNKNOWN
    # ------------------------------------------------------------------------

    if intent == "unknown":

        raise ValueError(
            "I could not determine the flight route "
            "from your request."
        )

    # ------------------------------------------------------------------------
    # ROUTE
    # ------------------------------------------------------------------------

    if intent == "route":

        if not decision.origin:

            raise ValueError(
                "Origin is missing from the route."
            )

        if not decision.destination:

            raise ValueError(
                "Destination is missing from the route."
            )

        origin_iata = validate_iata_code(
            decision.origin_iata
        )

        if not origin_iata:

            raise ValueError(
                (
                    f"Could not validate origin "
                    f"'{decision.origin}' "
                    f"with IATA code "
                    f"'{decision.origin_iata}'."
                )
            )

        destination_iata = validate_iata_code(
            decision.destination_iata
        )

        if not destination_iata:

            raise ValueError(
                (
                    f"Could not validate destination "
                    f"'{decision.destination}' "
                    f"with IATA code "
                    f"'{decision.destination_iata}'."
                )
            )

        return ResolvedRoute(
            origin=origin_iata,
            destination=destination_iata,
        )

    # ------------------------------------------------------------------------
    # ORIGIN ONLY
    # ------------------------------------------------------------------------

    if intent == "origin_only":

        if not decision.origin:

            raise ValueError(
                "Origin is missing."
            )

        origin_iata = validate_iata_code(
            decision.origin_iata
        )

        if not origin_iata:

            raise ValueError(
                (
                    f"Could not validate origin "
                    f"'{decision.origin}' "
                    f"with IATA code "
                    f"'{decision.origin_iata}'."
                )
            )

        return ResolvedRoute(
            origin=origin_iata,
            destination=None,
        )

    # ------------------------------------------------------------------------
    # DESTINATION ONLY
    # ------------------------------------------------------------------------

    if intent == "destination_only":

        if not decision.destination:

            raise ValueError(
                "Destination is missing."
            )

        destination_iata = validate_iata_code(
            decision.destination_iata
        )

        if not destination_iata:

            raise ValueError(
                (
                    f"Could not validate destination "
                    f"'{decision.destination}' "
                    f"with IATA code "
                    f"'{decision.destination_iata}'."
                )
            )

        # Validate the configured default origin too.
        origin_iata = validate_iata_code(
            DEFAULT_ORIGIN_IATA
        )

        if not origin_iata:

            raise RuntimeError(
                (
                    f"DEFAULT_ORIGIN_IATA "
                    f"'{DEFAULT_ORIGIN_IATA}' "
                    f"is not a valid airport IATA code."
                )
            )

        return ResolvedRoute(
            origin=origin_iata,
            destination=destination_iata,
        )

    # ------------------------------------------------------------------------
    # INVALID INTENT
    # ------------------------------------------------------------------------

    raise ValueError(
        f"Unsupported flight-search intent: {intent}"
    )


# ============================================================================
# AVIATIONSTACK API
# ============================================================================

def fetch_flights(
    origin: Optional[str],
    destination: Optional[str],
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    """
    Call AviationStack with validated IATA parameters.
    """

    if not AVIATIONSTACK_API_KEY:

        raise RuntimeError(
            "AVIATIONSTACK_API_KEY is missing. "
            "Add it to your .env file."
        )

    # Protect API from unreasonable limits.
    limit = max(
        1,
        min(
            int(limit),
            MAX_LIMIT,
        ),
    )

    params = {
        "access_key": AVIATIONSTACK_API_KEY,
        "limit": limit,
    }

    if origin:
        params["dep_iata"] = origin

    if destination:
        params["arr_iata"] = destination

    logger.info(
        (
            "Calling AviationStack: "
            "origin=%s destination=%s limit=%s"
        ),
        origin,
        destination,
        limit,
    )

    try:

        response = requests.get(
            AVIATIONSTACK_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout as exc:

        logger.exception(
            "AviationStack request timed out."
        )

        raise RuntimeError(
            "Flight API request timed out."
        ) from exc

    except requests.exceptions.RequestException as exc:

        logger.exception(
            "AviationStack request failed."
        )

        raise RuntimeError(
            f"Flight API request failed: {exc}"
        ) from exc

    try:

        data = response.json()

    except ValueError as exc:

        raise RuntimeError(
            "AviationStack returned invalid JSON."
        ) from exc

    # AviationStack can return an error object
    # even when HTTP status is 200.
    if "error" in data:

        error = data.get(
            "error"
        ) or {}

        code = error.get(
            "code",
            "unknown",
        )

        message = error.get(
            "message",
            "Unknown AviationStack error.",
        )

        raise RuntimeError(
            (
                "AviationStack error "
                f"[{code}]: {message}"
            )
        )

    return data.get(
        "data",
        []
    ) or []


# ============================================================================
# FLIGHT FORMATTER
# ============================================================================

def format_flight(
    flight: dict,
) -> str:
    """
    Convert one AviationStack flight object
    into readable text.
    """

    airline = (
        flight.get(
            "airline",
            {}
        ).get("name")
        or "Unknown airline"
    )

    flight_number = (
        flight.get(
            "flight",
            {}
        ).get("iata")
        or "Unknown flight number"
    )

    status = (
        flight.get("flight_status")
        or "Unknown"
    )

    departure = (
        flight.get("departure")
        or {}
    )

    arrival = (
        flight.get("arrival")
        or {}
    )

    # ------------------------------------------------------------------------
    # DEPARTURE
    # ------------------------------------------------------------------------

    dep_airport = (
        departure.get("airport")
        or "Unknown departure airport"
    )

    dep_iata = (
        departure.get("iata")
        or "Unknown"
    )

    dep_terminal = (
        departure.get("terminal")
        or "N/A"
    )

    dep_gate = (
        departure.get("gate")
        or "N/A"
    )

    dep_scheduled = (
        departure.get("scheduled")
        or "Unknown"
    )

    dep_delay = departure.get(
        "delay"
    )

    dep_delay_text = (
        f"{dep_delay} minutes"
        if dep_delay is not None
        else "N/A"
    )

    # ------------------------------------------------------------------------
    # ARRIVAL
    # ------------------------------------------------------------------------

    arr_airport = (
        arrival.get("airport")
        or "Unknown arrival airport"
    )

    arr_iata = (
        arrival.get("iata")
        or "Unknown"
    )

    arr_terminal = (
        arrival.get("terminal")
        or "N/A"
    )

    arr_gate = (
        arrival.get("gate")
        or "N/A"
    )

    arr_scheduled = (
        arrival.get("scheduled")
        or "Unknown"
    )

    arr_delay = arrival.get(
        "delay"
    )

    arr_delay_text = (
        f"{arr_delay} minutes"
        if arr_delay is not None
        else "N/A"
    )

    return (
        f"Airline: {airline}\n"
        f"Flight: {flight_number}\n"
        f"Status: {status}\n\n"

        f"Departure:\n"
        f"- Airport: {dep_airport}\n"
        f"- IATA: {dep_iata}\n"
        f"- Terminal: {dep_terminal}\n"
        f"- Gate: {dep_gate}\n"
        f"- Scheduled: {dep_scheduled}\n"
        f"- Delay: {dep_delay_text}\n\n"

        f"Arrival:\n"
        f"- Airport: {arr_airport}\n"
        f"- IATA: {arr_iata}\n"
        f"- Terminal: {arr_terminal}\n"
        f"- Gate: {arr_gate}\n"
        f"- Scheduled: {arr_scheduled}\n"
        f"- Delay: {arr_delay_text}"
    )


# ============================================================================
# MAIN SEARCH FUNCTION
# ============================================================================

def search_flights(
    query: str,
    limit: int = DEFAULT_LIMIT,
) -> str:
    """
    Main flight-search entry point.

    Example:

        search_flights(
            "Plan a 7 days Japan trip from Bangladesh"
        )
    """

    if not query or not query.strip():

        return (
            "Please provide a flight-search query."
        )

    try:

        # ================================================================
        # 1. LLM ROUTE EXTRACTION
        # ================================================================

        decision = extract_route_with_llm(
            query
        )
        print(decision,"==============================================")
        
        # ================================================================
        # 2. ROUTE + IATA VALIDATION
        # ================================================================

        route = resolve_route(
            decision
        )
        print(route)
        logger.info(
            (
                "Resolved route: "
                "%s -> %s"
            ),
            route.origin,
            route.destination,
        )

        # ================================================================
        # 3. AVIATIONSTACK
        # ================================================================

        flights = fetch_flights(
            origin=route.origin,
            destination=route.destination,
            limit=limit,
        )

        # ================================================================
        # 4. NO RESULTS
        # ================================================================

        if not flights:

            if route.origin and route.destination:

                route_text = (
                    f"for route "
                    f"{route.origin} → "
                    f"{route.destination}"
                )

            elif route.origin:

                route_text = (
                    f"from {route.origin}"
                )

            elif route.destination:

                route_text = (
                    f"to {route.destination}"
                )

            else:

                route_text = (
                    "for the requested global search"
                )

            return (
                f"No live flight data found "
                f"{route_text}.\n\n"
                "AviationStack provides live/status "
                "flight data, not ticket prices."
            )

        # ================================================================
        # 5. ROUTE TITLE
        # ================================================================

        if route.origin and route.destination:

            route_info = (
                f"Live flights from "
                f"{route.origin} to "
                f"{route.destination}"
            )

        elif route.origin:

            route_info = (
                f"Live flights from "
                f"{route.origin}"
            )

        elif route.destination:

            route_info = (
                f"Live flights to "
                f"{route.destination}"
            )

        else:

            route_info = (
                "Global live flights"
            )

        # ================================================================
        # 6. FORMAT FLIGHTS
        # ================================================================

        formatted_flights = [
            format_flight(flight)
            for flight in flights[:limit]
        ]

        return (
            f"{route_info}\n\n"
            + "\n\n---\n\n".join(
                formatted_flights
            )
        )

    except ValueError as exc:

        logger.warning(
            "Flight query validation failed: %s",
            exc,
        )

        return (
            "Flight search validation error: "
            f"{exc}"
        )

    except RuntimeError as exc:

        logger.error(
            "Flight search failed: %s",
            exc,
        )

        return (
            "Flight search error: "
            f"{exc}"
        )

    except Exception:

        logger.exception(
            "Unexpected flight search error."
        )

        return (
            "An unexpected error occurred while "
            "searching for flights."
        )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":

    test_queries = [
        "Plan a 7 days Japan trip from Bangladesh",

        "Show me flights from Mumbai to Tokyo",

        "Find flights to Dubai",

        "Show flights from Delhi",

        "Show all live flights worldwide",
    ]

    for query in test_queries:

        print()
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        result = search_flights(
            query
        )

        print(result)

