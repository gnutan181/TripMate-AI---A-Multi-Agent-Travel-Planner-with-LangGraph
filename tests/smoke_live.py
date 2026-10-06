"""Opt-in real API/DB/LLM smoke check; credentials come only from environment."""
import json
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.secure_logging import install_log_redaction, redact

install_log_redaction()


def main(base_url=None):
    if base_url:
        import httpx
        client = httpx.Client(base_url=base_url, timeout=180)
    else:
        from fastapi.testclient import TestClient
        from app import app
        client = TestClient(app)
    thread_id = "deployment_smoke_" + uuid.uuid4().hex
    response = client.post("/api/travel", json={
        "message": "Plan a 3-day trip to Tokyo including hotels and sightseeing.",
        "thread_id": thread_id,
    })
    data = response.json()
    print(json.dumps({
        "travel_status": response.status_code,
        "success": data.get("success"),
        "requires_approval": data.get("requires_approval"),
        "selected_agents": data.get("selected_agents"),
        "hotel_result_present": bool(data.get("hotel_results")),
        "hotel_research_returned_results": str(data.get("hotel_results", "")).startswith("1. "),
        "error": data.get("error"),
    }))
    if response.status_code != 200 or not data.get("requires_approval"):
        raise RuntimeError("Live travel did not reach the approval checkpoint")
    response = client.post("/api/travel/approve", json={
        "thread_id": thread_id, "approved": True, "feedback": "",
    })
    data = response.json()
    print(json.dumps({"approval_status": response.status_code,
                      "approved": data.get("approved"),
                      "answer_present": bool(data.get("answer")),
                      "error": data.get("error")}))
    if response.status_code != 200 or not data.get("approved"):
        raise RuntimeError("Live approval failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", help="Exercise a running server over real HTTP")
    args = parser.parse_args()
    try:
        main(args.base_url)
    except Exception as exc:
        print(redact(f"Live smoke blocked/failed: {type(exc).__name__}: {exc}"))
        sys.exit(1)
