"""Integration tests for the /memory router (stats, list, search, store, remember, config)."""

from test_support import get_test_client


def _cleanup():
    """Clear all stored memory via the knowledge singleton."""
    from domains.learner.knowledge import get_knowledge_memory
    get_knowledge_memory().clear_all()


def _d(resp):
    j = resp.json()
    return j.get("data", j)


def test_stats_enabled_flag():
    client = get_test_client()
    resp = client.get("/memory/stats")
    assert resp.status_code == 200
    body = _d(resp)
    assert isinstance(body, dict)
    assert "enabled" in body
    assert isinstance(body["enabled"], bool)


def test_list_empty_returns_empty_items():
    client = get_test_client()
    _cleanup()
    resp = client.get("/memory/list")
    assert resp.status_code == 200
    body = _d(resp)
    assert body["items"] == []
    assert body["total"] == 0


def test_store_then_list_returns_item():
    client = get_test_client()
    _cleanup()

    add = client.post("/memory/store", json={"content": "Memory fact alpha", "topic": "test", "source": "api"})
    assert add.status_code == 200
    assert _d(add)["stored"] is True

    resp = client.get("/memory/list")
    assert resp.status_code == 200
    items = _d(resp)["items"]
    assert len(items) == 1
    assert items[0]["content"] == "Memory fact alpha"


def test_store_empty_content_rejected():
    client = get_test_client()
    resp = client.post("/memory/store", json={"content": "   ", "topic": "test", "source": "api"})
    assert resp.status_code == 400


def test_search_returns_matching_results():
    client = get_test_client()
    _cleanup()
    client.post("/memory/store", json={"content": "The user prefers dark mode interfaces", "topic": "pref", "source": "api"})

    resp = client.get("/memory/search", params={"q": "dark mode"})
    assert resp.status_code == 200
    body = _d(resp)
    assert body["total"] >= 1
    assert any("dark mode" in r.get("content", "") for r in body["results"])


def test_search_missing_query_rejected():
    client = get_test_client()
    resp = client.get("/memory/search")
    assert resp.status_code == 422


def test_remember_persists_turn():
    client = get_test_client()
    _cleanup()

    resp = client.post(
        "/memory/remember",
        json={"user_message": "Tell me about machine learning", "assistant_response": "Machine learning learns patterns from data. Gradient descent is the optimizer."},
    )
    assert resp.status_code == 200
    assert _d(resp)["stored"] is True

    listed = _d(client.get("/memory/list"))["items"]
    assert len(listed) >= 1


def test_config_toggle_off_disables_store():
    client = get_test_client()
    _cleanup()

    off = client.post("/memory/config", json={"enabled": False})
    assert off.status_code == 200
    assert _d(off)["enabled"] is False

    stored = client.post("/memory/store", json={"content": "Should be skipped", "topic": "test", "source": "api"})
    assert stored.status_code == 200
    assert _d(stored)["stored"] is False

    listed = _d(client.get("/memory/list"))["items"]
    assert listed == []

    on = client.post("/memory/config", json={"enabled": True})
    assert on.status_code == 200
    assert _d(on)["enabled"] is True


def test_config_invalid_body_rejected():
    client = get_test_client()
    resp = client.post("/memory/config", json={"enabled": "not-a-bool"})
    assert resp.status_code == 422


def test_config_get_returns_snapshot():
    client = get_test_client()
    resp = client.get("/memory/config")
    assert resp.status_code == 200
    body = _d(resp)
    assert isinstance(body, dict)
    for key in ("enabled", "min_chars", "max_facts", "store_path", "sync_remember",
                "consolidation_threshold", "maintenance_interval_minutes", "archive_retention_days"):
        assert key in body
    assert isinstance(body["archive_retention_days"], (int, float))


def test_config_sets_archive_retention_days():
    client = get_test_client()
    resp = client.post("/memory/config", json={"archive_retention_days": 45})
    assert resp.status_code == 200
    assert _d(resp)["archive_retention_days"] == 45
    assert _d(client.get("/memory/config"))["archive_retention_days"] == 45


def test_config_clamps_negative_retention_to_zero():
    client = get_test_client()
    resp = client.post("/memory/config", json={"archive_retention_days": -10})
    assert resp.status_code == 200
    assert _d(resp)["archive_retention_days"] == 0


def test_delete_item_removes_entry():
    client = get_test_client()
    _cleanup()
    add = client.post("/memory/store", json={"content": "Deletable fact", "topic": "test", "source": "api"})
    item_id = _d(client.get("/memory/list"))["items"][0]["id"]

    resp = client.delete(f"/memory/{item_id}")
    assert resp.status_code == 200
    assert _d(resp)["deleted"] == 1

    assert _d(client.get("/memory/list"))["total"] == 0


def test_clear_empties_store():
    client = get_test_client()
    _cleanup()
    client.post("/memory/store", json={"content": "Fact one", "topic": "test", "source": "api"})
    client.post("/memory/store", json={"content": "Fact two", "topic": "test", "source": "api"})

    resp = client.post("/memory/clear")
    assert resp.status_code == 200
    assert _d(resp)["cleared"] >= 2

    assert _d(client.get("/memory/list"))["total"] == 0
