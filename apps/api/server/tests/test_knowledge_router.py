"""Integration tests for the /knowledge router (CRUD + search + batch)."""

from test_support import get_test_client


def _cleanup(client):
    """Clear all knowledge items via the singleton's clear_all (resets both vector store and _visited)."""
    from domains.learner.knowledge import get_knowledge_memory
    get_knowledge_memory().clear_all()


def test_list_knowledge_empty():
    client = get_test_client()
    _cleanup(client)
    resp = client.get("/knowledge")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 0


def test_add_and_list_knowledge():
    client = get_test_client()
    _cleanup(client)

    add = client.post("/knowledge", json={"content": "Test fact one", "topic": "test", "source": "manual"})
    assert add.status_code == 200
    assert add.json()["status"] == "stored"

    add2 = client.post("/knowledge", json={"content": "Test fact two", "topic": "code", "source": "manual"})
    assert add2.status_code == 200

    resp = client.get("/knowledge")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    topics = {i["topic"] for i in items}
    assert "test" in topics
    assert "code" in topics
    contents = {i["content"] for i in items}
    assert "Test fact one" in contents
    assert "Test fact two" in contents


def test_add_with_defaults():
    client = get_test_client()
    _cleanup(client)

    resp = client.post("/knowledge", json={"content": "Default fact"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "stored"
    assert data["content"] == "Default fact"


def test_add_empty_content_fails():
    client = get_test_client()
    resp = client.post("/knowledge", json={"content": ""})
    assert resp.status_code == 422


def test_delete_knowledge():
    client = get_test_client()
    _cleanup(client)

    client.post("/knowledge", json={"content": "Delete me", "topic": "test"})
    items = client.get("/knowledge").json()
    assert len(items) == 1
    item_id = items[0]["id"]

    delete = client.delete(f"/knowledge/{item_id}")
    assert delete.status_code == 200
    assert delete.json()["status"] == "deleted"

    assert len(client.get("/knowledge").json()) == 0


def test_delete_nonexistent_returns_404():
    client = get_test_client()
    resp = client.delete("/knowledge/nonexistent-id-12345")
    assert resp.status_code == 404
    body = resp.json()
    assert "not found" in body.get("error", body.get("detail", "")).lower()


def test_search_knowledge():
    client = get_test_client()
    _cleanup(client)

    client.post("/knowledge", json={"content": "Python is a programming language", "topic": "code"})
    client.post("/knowledge", json={"content": "Cats are furry animals", "topic": "pets"})

    resp = client.get("/knowledge/search?query=Python")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1
    assert any("Python" in r["content"] for r in body["results"])


def test_search_no_results():
    client = get_test_client()
    _cleanup(client)

    # With an empty store, search should return count: 0
    resp = client.get("/knowledge/search?query=anything")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_batch_ingest():
    client = get_test_client()
    _cleanup(client)

    resp = client.post("/knowledge/batch", json={
        "items": [
            {"content": "Batch item A", "source": "test"},
            {"content": "Batch item B"},
        ]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["stored"] == 2

    items = client.get("/knowledge").json()
    assert len(items) >= 2


def test_get_context():
    client = get_test_client()
    _cleanup(client)

    client.post("/knowledge", json={"content": "Context test fact", "topic": "test"})

    resp = client.get("/knowledge/context")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1
    assert "[KNOWN_FACTS]" in body["context"]
