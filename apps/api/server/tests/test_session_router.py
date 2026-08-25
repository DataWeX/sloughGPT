"""Tests for the /session router — context, messages, inspector."""

from test_support import get_test_client, _data


def test_set_and_get_session_context():
    client = get_test_client()
    sid = "test_session_001"

    resp = client.post(f"/session/{sid}/context", json={
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
    })
    assert resp.status_code == 200
    body = _data(resp)
    assert body["message_count"] == 2

    resp2 = client.get(f"/session/{sid}/messages")
    assert resp2.status_code == 200
    body2 = _data(resp2)
    assert body2["session_id"] == sid
    assert len(body2["messages"]) == 2


def test_set_empty_context():
    client = get_test_client()
    sid = "test_session_empty"

    resp = client.post(f"/session/{sid}/context", json={"messages": []})
    assert resp.status_code == 200
    body = _data(resp)
    assert body["message_count"] == 0


def test_get_messages_empty_session():
    client = get_test_client()
    resp = client.get("/session/nonexistent_id/messages")
    assert resp.status_code == 200
    body = _data(resp)
    assert body["messages"] == []


def test_session_inspector():
    client = get_test_client()
    sid = "test_session_inspector"

    client.post(f"/session/{sid}/context", json={
        "messages": [{"role": "user", "content": "inspect me"}]
    })

    resp = client.get(f"/session/{sid}/inspector")
    assert resp.status_code == 200
    body = _data(resp)
    assert "session" in body
    assert "messages" in body["session"]
    assert "knowledge" in body
    assert "traits" in body
    assert "modes" in body
    assert "workspace" in body
    assert "elapsed_ms" in body
