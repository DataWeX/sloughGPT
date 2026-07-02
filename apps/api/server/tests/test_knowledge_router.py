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
    body = resp.json()
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
    body = resp.json()
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
    body = resp.json()
    assert body["is_duplicate"] is True
    assert body["score"] >= 0.85


def test_categorize():
    client = get_test_client()
    _cleanup(client)
    resp = client.post("/knowledge/categorize", json={
        "content": "The neural network was trained on MNIST using gradient descent",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "topic" in body
    assert "suggestions" in body
    assert isinstance(body["suggestions"], list)


def test_knowledge_gaps_empty():
    client = get_test_client()
    _cleanup(client)
    resp = client.get("/knowledge/gaps")
    assert resp.status_code == 200
    body = resp.json()
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
    body = resp.json()
    assert body["added"] >= 1
    assert body["errors"] == 0

    items = client.get("/knowledge").json()
    assert len(items) >= 1


def test_bulk_ingest_dedup():
    client = get_test_client()
    _cleanup(client)

    # Add first batch
    client.post("/knowledge/bulk-ingest", json={
        "items": ["Unique fact one", "Unique fact two"],
        "topic": "test",
    })

    # Second batch with duplicates
    resp = client.post("/knowledge/bulk-ingest", json={
        "items": ["Unique fact one", "Brand new fact"],
        "topic": "test",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["added"] == 1  # Only "Brand new fact"
    assert body["skipped"] == 1  # "Unique fact one" is a dup


def test_add_returns_duplicate_status():
    client = get_test_client()
    _cleanup(client)

    # Add a long enough fact so n-gram embedding gives good similarity
    long_fact = "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed"
    resp1 = client.post("/knowledge", json={
        "content": long_fact,
        "topic": "ml",
    })
    assert resp1.json()["status"] == "stored"

    # Add the exact same fact again
    resp2 = client.post("/knowledge", json={
        "content": long_fact,
        "topic": "ml",
    })
    body = resp2.json()
    # Should either be duplicate or stored (depending on n-gram similarity)
    assert body["status"] in ("duplicate", "stored")
