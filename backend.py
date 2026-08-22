import os 
import certifi
from dotenv import load_dotenv

load_dotenv()
from langgraph.types import Command, interrupt

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] =certifi.where()

from typing import TypedDict,Annotated,Any

import operator
import uuid
import psycopg

from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END

from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langchain_groq import ChatGroq
from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search
from tools.weather_tool import get_current_weather, get_forecast
import json
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostGreSQL External Database URL to .env"
        )


    if "sslmode" not in database_url:
        separator ="&" if "?" in database_url else "?"
        database_url =f"{database_url}{separator}sslmode=require" 

    return database_url



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "2200"))
GROQ_REQUEST_TOKEN_BUDGET = int(os.getenv("GROQ_REQUEST_TOKEN_BUDGET", "7200"))
GROQ_MIN_OUTPUT_TOKENS = 256

if GROQ_MAX_TOKENS < GROQ_MIN_OUTPUT_TOKENS:
    raise ValueError("GROQ_MAX_TOKENS must be at least 256.")
if GROQ_REQUEST_TOKEN_BUDGET <= GROQ_MAX_TOKENS:
    raise ValueError("GROQ_REQUEST_TOKEN_BUDGET must exceed GROQ_MAX_TOKENS.")


def _create_planner_llm() -> ChatGroq:
    """Create an LLM whose output cap is calculated for each request."""

    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
    )


llm = _create_planner_llm()


class TravelState(TypedDict):
    messages : Annotated[list[AnyMessage],operator.add]
    user_query : str

    # Supervisor + guardrail state
    guardrail_allowed :bool
    guardrail_reason : str
    selected_agents :list[str]
    trip_constraints : dict[str, Any]
    supervisor_reasoning : str

    # original specialist results
    flight_results : str
    hotel_results :str
    itinerary : str
    weather_results:str

    # New budget + HITL state
    budget_results : str
    approval_request :str
    approved : bool
    human_feedback : str
    final_response : str

    llm_calls :int

################################
# SHARED HELPERS
###############################

KNOWN_AGENTS ={
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
}

AGENT_ORDER =[
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
]

def _llm_text(system_prompt : str, user_prompt : str)->str:
    response = _invoke_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=_compact_text(user_prompt, 4_000)),
        ],
        preferred_output_tokens=800,
    )
    return str(response.content)


def _compact_text(value: Any, max_tokens: int) -> str:
    """Keep the beginning and conclusion of long provider/agent results."""

    text = str(value or "").strip()
    token_ids = encoding.encode(text)
    if len(token_ids) <= max_tokens:
        return text

    head_tokens = int(max_tokens * 0.7)
    tail_tokens = max_tokens - head_tokens
    return (
        encoding.decode(token_ids[:head_tokens])
        + "\n\n[...context shortened to stay within the Groq request limit...]\n\n"
        + encoding.decode(token_ids[-tail_tokens:])
    )


def _invoke_llm(messages: list[AnyMessage], preferred_output_tokens: int):
    """Fit a request under the configured Groq TPM guardrail.

    Groq rejects a request when prompt tokens plus requested output tokens exceed
    the account's request allowance. Binding an explicit output cap prevents the
    provider default from turning a valid long prompt into a 413 response.
    """

    prompt_tokens = sum(count_tokens(str(message.content)) + 12 for message in messages)
    available_output_tokens = GROQ_REQUEST_TOKEN_BUDGET - prompt_tokens
    output_tokens = min(
        GROQ_MAX_TOKENS,
        preferred_output_tokens,
        available_output_tokens,
    )
    if output_tokens < GROQ_MIN_OUTPUT_TOKENS:
        raise ValueError(
            "The request is too large even after context compaction. "
            "Reduce the input size or increase GROQ_REQUEST_TOKEN_BUDGET."
        )

    return llm.bind(max_tokens=output_tokens).invoke(messages)


def _json_from_llm(text:str) -> dict[str, Any]:

    """Extract the first complete JSON object returned by the modal."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start :end +1])


def _empty_contraints()-> dict[str,Any]:
    return {
        "destination":"",
        "origin":"",
        "duration":"",
        "budget":"",
        "travel_style":"",
        "special_preferences":[]
    }


#############################################################################################################
# Supervisor Agent + Input Guardrail
#############################################################################################################

def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }

# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }    



            
    
# =====================================
# Flight Agent
# =====================================

def flight_agent(state: TravelState):
    # Calls AviationStack directly. The tool applies bounded retries and returns
    # a user-safe fallback if the provider cannot be reached.
    flight_data = search_flights(state["user_query"])

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =====================================
# hotel Agent
# ======================================
def hotel_agent(state:TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)


    return {
        "hotel_results" : hotel_results,
        "messages" :[
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls" :state.get("llm_calls",0) +1
    }

# =====================================
# Weather Agent
# ======================================

def _weather_destination(state: TravelState) -> str:
    """Use supervisor output first, then make a best-effort LLM extraction."""

    destination = str(
        state.get("trip_constraints", {}).get("destination") or ""
    ).strip()
    if destination:
        return destination

    try:
        destination = _llm_text(
            "Extract a destination for a weather lookup. Return only a city or country name.",
            state["user_query"],
        ).strip()
        return destination.splitlines()[0].strip(" .")
    except Exception:
        return ""


def weather_agent(state: TravelState):
    city = _weather_destination(state)

    # Avoid calling OpenWeather with the full user request when the supervisor
    # cannot determine a destination.
    if not city:
        unavailable = "Weather information is unavailable because no destination was identified."
        return {
            "weather_results": unavailable,
            "messages": [AIMessage(content="Weather lookup skipped: destination missing")],
        }

    try:
        weather_data = get_current_weather(city)
    except Exception:
        weather_data = "Current weather is temporarily unavailable after retrying."

    try:
        forecast_data = get_forecast(city)
    except Exception:
        forecast_data = "Forecast is temporarily unavailable after retrying."

    return {
        "weather_results" : f"""
         Current Weather:
         {weather_data}
         Forecast:
         {forecast_data}""",
         "messages" :[
             AIMessage(
                 content = "Weather information fetched"
             )
         ]
    }




# =========================
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    user_query = _compact_text(state["user_query"], 500)
    flight_results = _compact_text(state.get("flight_results", ""), 700)
    hotel_results = _compact_text(state.get("hotel_results", ""), 700)
    weather_results = _compact_text(state.get("weather_results", ""), 350)
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{user_query}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{flight_results}

Hotel Results:
{hotel_results}

Weather Results:
{weather_results}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = _invoke_llm(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ],
        preferred_output_tokens=1_200,
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =====================================
# Iitinery Agent
# ======================================

def itinerary_agent(state : TravelState):
    user_query = _compact_text(state["user_query"], 550)
    flight_results = _compact_text(state.get("flight_results", ""), 850)
    hotel_results = _compact_text(state.get("hotel_results", ""), 850)
    weather_results = _compact_text(state.get("weather_results", ""), 400)
    prompt =f"""
Create a complete travel itinerary.

User Query:
{user_query}

Flight Results:
{flight_results}

Hotel Results:
{hotel_results}

Weather Results:
{weather_results}

Make the itinerary practical, budget-aware, and easy to follow.
"""
    response = _invoke_llm([
        SystemMessage(content = "You are an expert travel planner."),
        HumanMessage(content=prompt)
    ], preferred_output_tokens=2_000)

    return {
        "itinerary":response.content,
        "messages":[response],
        "llm_calls":state.get("llm_calls",0) +1
    }



# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - original format kept, HITL feedback added
# =========================
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{_compact_text(review_instruction, 200)}

User Request:
{_compact_text(state['user_query'], 450)}

Supervisor Constraints:
{_compact_text(state.get('trip_constraints', {}), 250)}

Flights:
{_compact_text(state.get('flight_results', ''), 650)}

Hotels:
{_compact_text(state.get('hotel_results', ''), 650)}

Weather:
{_compact_text(state.get('weather_results', ''), 350)}

Budget Analysis:
{_compact_text(state.get('budget_results', ''), 550)}

Draft Itinerary:
{_compact_text(state.get('itinerary', ''), 1_400)}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
- Do not omit a required section. Include all useful concrete details from the
  specialist results; use "unavailable" only when that specialist has none.
"""

    response = _invoke_llm(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ],
        preferred_output_tokens=2_200,
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = get_database_url()
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }

def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }

def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)
