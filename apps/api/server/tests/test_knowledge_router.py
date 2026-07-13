"""Integration tests for the /knowledge router (CRUD + search + batch)."""

from test_support import get_test_client


def _cleanup(client):
    """Clear all knowledge items via the singleton's clear_all."""
    from domains.learner.knowledge import get_knowledge_memory
    get_knowledge_memory().clear_all()


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    if isinstance(body, list):
        return body
    return body.get("data", body)


def test_list_knowledge_empty():
    client = get_test_client()
    _cleanup(client)
    resp = client.get("/knowledge")
    assert resp.status_code == 200
    body = _data(resp)
    assert isinstance(body, list)
    assert len(body) == 0


def test_add_and_list_knowledge():
    client = get_test_client()
    _cleanup(client)

    add = client.post("/knowledge", json={"content": "Test fact one", "topic": "test", "source": "manual"})
    assert add.status_code == 200
    assert _data(add)["status"] == "stored"

    add2 = client.post("/knowledge", json={"content": "Test fact two", "topic": "code", "source": "manual"})
    assert add2.status_code == 200

    resp = client.get("/knowledge")
    assert resp.status_code == 200
    items = _data(resp)
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
    data = _data(resp)
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
    items = _data(client.get("/knowledge"))
    assert len(items) == 1
    item_id = items[0]["id"]

    delete = client.delete(f"/knowledge/{item_id}")
    assert delete.status_code == 200
    assert _data(delete)["status"] == "deleted"

    assert len(_data(client.get("/knowledge"))) == 0


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
    body = _data(resp)
    assert body["count"] >= 1
    assert any("Python" in r["content"] for r in body["results"])


def test_search_no_results():
    client = get_test_client()
    _cleanup(client)

    resp = client.get("/knowledge/search?query=anything")
    assert resp.status_code == 200
    assert _data(resp)["count"] == 0


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
    data = _data(resp)
    assert data["stored"] >= 1

    items = _data(client.get("/knowledge"))
    assert len(items) >= 2


def test_get_context():
    client = get_test_client()
    _cleanup(client)

    client.post("/knowledge", json={"content": "Context test fact", "topic": "test"})

    resp = client.get("/knowledge/context")
    assert resp.status_code == 200
    body = _data(resp)
    assert body["count"] >= 1
    assert "[KNOWN_FACTS]" in body["context"]


# ═══════════════════════════════════════════════════════════════════════
# Tests for practical knowledge operations endpoints
# ═══════════════════════════════════════════════════════════════════════


def test_search_files():
    client = get_test_client()
    resp = client.post("/knowledge/search-files", json={
        "query": "def function",
        "path": "routers",
        "top_k": 3,
        "extensions": ["py"],
    })
    assert resp.status_code == 200
    body = _data(resp)
    assert "results" in body
    assert "indexed_files" in body
    assert isinstance(body["results"], list)


def test_check_duplicate_unique():
    client = get_test_client()
    _cleanup(client)
    resp = client.post("/knowledge/check-duplicate", json={
        "content": "Completely unique content that does not exist anywhere",
        "threshold": 0.85,
    })
    assert resp.status_code == 200
    body = _data(resp)
    assert body["is_duplicate"] is False
    assert body["score"] < 0.85


def test_check_duplicate_after_add():
    client = get_test_client()
    _cleanup(client)

    client.post("/knowledge", json={"content": "Python is a programming language", "topic": "code"})

    resp = client.post("/knowledge/check-duplicate", json={
        "content": "Python is a programming language",
        "threshold": 0.85,
    })
    assert resp.status_code == 200
    body = _data(resp)
    assert body["is_duplicate"] is True
    assert body["score"] >= 0.85


def test_categorize():
    client = get_test_client()
    _cleanup(client)
    resp = client.post("/knowledge/categorize", json={
        "content": "The neural network was trained on MNIST using gradient descent",
    })
    assert resp.status_code == 200
    body = _data(resp)
    assert "topic" in body
    assert "suggestions" in body
    assert isinstance(body["suggestions"], list)


def test_knowledge_gaps_empty():
    client = get_test_client()
    _cleanup(client)
    resp = client.get("/knowledge/gaps")
    assert resp.status_code == 200
    body = _data(resp)
    assert "gaps" in body
    assert "total_facts" in body
    assert isinstance(body["gaps"], list)

def test_bulk_ingest():
    client = get_test_client()
    _cleanup(client)

    resp = client.post("/knowledge/bulk-ingest", json={
        "items": [
            "Quantum entanglement enables instantaneous correlations between particles regardless of distance",
            "Photosynthesis converts light energy into chemical energy in plants through chlorophyll",
        ],
        "topic": "test_bulk",
        "source": "test",
    })
    assert resp.status_code == 200
    body = _data(resp)
    assert body["added"] >= 1
    assert body["errors"] == 0

    items = _data(client.get("/knowledge"))
    assert len(items) >= 1


def test_bulk_ingest_dedup():
    client = get_test_client()
    _cleanup(client)

    client.post("/knowledge/bulk-ingest", json={
        "items": [
            "Quantum entanglement enables instantaneous correlations between particles regardless of distance",
            "Photosynthesis converts light energy into chemical energy in plants through chlorophyll",
        ],
        "topic": "test",
        "dedup_threshold": 0.999,
    })

    resp = client.post("/knowledge/bulk-ingest", json={
        "items": [
            "Quantum entanglement enables instantaneous correlations between particles regardless of distance",
            "The mitochondria is the powerhouse of the cell producing ATP energy",
        ],
        "topic": "test",
        "dedup_threshold": 0.999,
    })
    assert resp.status_code == 200
    body = _data(resp)
    assert body["added"] >= 1


def test_add_returns_duplicate_status():
    client = get_test_client()
    _cleanup(client)

    long_fact = "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed"
    resp1 = client.post("/knowledge", json={
        "content": long_fact,
        "topic": "ml",
    })
    assert _data(resp1)["status"] == "stored"

    resp2 = client.post("/knowledge", json={
        "content": long_fact,
        "topic": "ml",
    })
    body = _data(resp2)
    assert body["status"] in ("duplicate", "stored")
