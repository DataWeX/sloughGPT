import json
from fastapi.testclient import TestClient

# Import the FastAPI app defined in the server entrypoint
from apps.api.server.main import app


def _parse_sse_line(line: str) -> dict:
    """Extract the JSON payload from an SSE ``data: {json}`` line."""
    # FastAPI's StreamingResponse yields lines like "data: {...}\n\n"
    line = line.strip()
    if not line.startswith("data: "):
        return {}
    json_part = line[len("data: "):]
    try:
        return json.loads(json_part)
    except json.JSONDecodeError:
        return {}


def test_regenerate_uses_stored_context():
    client = TestClient(app)
    session_id = "unittest-session"
    # Store a simple conversation context
    messages = [{"role": "user", "content": "Hello"}]
    resp = client.post(f"/session/{session_id}/context", json={"messages": messages})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "stored"

    # Trigger regeneration – model is likely not loaded in the test env, so we expect a model‑not‑loaded error
    resp = client.post(f"/session/{session_id}/regenerate", json={})
    assert resp.status_code == 200

    # The response is a streaming SSE; collect the first few events
    events = []
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        ev = _parse_sse_line(raw_line if isinstance(raw_line, str) else raw_line.decode())
        if ev:
            events.append(ev)
        # Stop after we see a "complete" / "error" event to keep the test fast
        if ev.get("status") in ("error", "complete"):
            break

    # There should be at least one event with an error about the model not being loaded
    assert any(
        ev.get("status") == "error" and "Model not loaded" in ev.get("data", {}).get("error", "")
        for ev in events
    ), f"Expected a model-not-loaded error event, got: {events}"
