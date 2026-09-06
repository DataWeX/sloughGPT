"""Tests for the /knowledge (kb) router — CRUD, search, batch, stats."""

from unittest.mock import patch

from test_support import _data, get_test_client


def _cleanup():
    from domains.learner.knowledge import get_knowledge_memory

    get_knowledge_memory().clear_all()


def _noop_truth_label(content):
    from dataclasses import dataclass

    @dataclass
    class _R:
        label: str = "neutral"
        confidence: float = 0.5

    return _R()


def _add_item(client, content="test fact alpha", topic="testing", source="test"):
    return client.post(
        "/knowledge",
        json={
            "content": content,
            "topic": topic,
            "source": source,
            "importance": 0.7,
        },
    )


# ── CRUD ──────────────────────────────────────────────────


@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_list_empty(_mock_tl):
    _cleanup()
    client = get_test_client()
    resp = client.get("/knowledge")
    assert resp.status_code == 200
    body = _data(resp)
    assert isinstance(body, list)


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_add_returns_stored(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    resp = _add_item(client)
    assert resp.status_code == 200
    body = _data(resp)
    assert body["status"] in ("stored", "duplicate")
    assert "id" in body


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_add_then_list_has_item(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    _add_item(client, content="alpha fact")
    resp = client.get("/knowledge")
    items = _data(resp)
    assert any("alpha fact" in it.get("content", "") for it in items)


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_add_duplicate_is_idempotent(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    r1 = _add_item(client, content="dup fact")
    r2 = _add_item(client, content="dup fact")
    assert _data(r1)["status"] == "stored"
    assert _data(r2)["status"] == "duplicate"


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_update_knowledge(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    add_resp = _add_item(client, content="original")
    item_id = _data(add_resp)["id"]

    resp = client.patch(f"/knowledge/{item_id}", json={"content": "updated"})
    assert resp.status_code == 200
    assert _data(resp)["status"] == "updated"


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_update_nonexistent_returns_404(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    resp = client.patch("/knowledge/fake_id_999", json={"content": "x"})
    assert resp.status_code == 404


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_delete_knowledge(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    add_resp = _add_item(client, content="delete me")
    item_id = _data(add_resp)["id"]

    resp = client.delete(f"/knowledge/{item_id}")
    assert resp.status_code == 200

    verify = client.get("/knowledge")
    items = _data(verify)
    assert not any(it.get("id") == item_id for it in items)


# ── Search ────────────────────────────────────────────────


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_search_returns_results(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    _add_item(client, content="Python is a programming language")
    resp = client.get("/knowledge/search", params={"query": "Python"})
    assert resp.status_code == 200
    body = _data(resp)
    assert "results" in body
    assert body["count"] >= 1


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_search_empty_query_returns_empty(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    _add_item(client, content="some fact")
    resp = client.get("/knowledge/search", params={"query": ""})
    assert resp.status_code == 200
    body = _data(resp)
    assert body["count"] == 0


# ── Stats ─────────────────────────────────────────────────


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_stats_returns_expected_keys(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    _add_item(client, topic="alpha")
    _add_item(client, content="second", topic="beta")
    resp = client.get("/knowledge/stats")
    assert resp.status_code == 200
    body = _data(resp)
    assert body["total_items"] >= 2
    assert "topics" in body
    assert "sources" in body
    assert "avg_importance" in body


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_list_topics(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    _add_item(client, topic="alpha")
    _add_item(client, content="second", topic="beta")
    resp = client.get("/knowledge/topics")
    assert resp.status_code == 200
    body = _data(resp)
    assert "topics" in body
    assert body["total"] >= 2


# ── Batch ─────────────────────────────────────────────────


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_batch_ingest(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    resp = client.post(
        "/knowledge/batch",
        json={
            "items": [
                {"content": "batch one", "topic": "batch_test"},
                {"content": "batch two", "topic": "batch_test"},
            ]
        },
    )
    assert resp.status_code == 200
    body = _data(resp)
    assert body.get("stored", body.get("count", 0)) >= 2


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_batch_delete(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    r1 = _add_item(client, content="del1", topic="del_test")
    r2 = _add_item(client, content="del2", topic="del_test")
    id1 = _data(r1)["id"]
    id2 = _data(r2)["id"]

    resp = client.post("/knowledge/batch-delete", json={"ids": [id1, id2]})
    assert resp.status_code == 200

    verify = client.get("/knowledge")
    items = _data(verify)
    remaining_ids = {it.get("id") for it in items}
    assert id1 not in remaining_ids
    assert id2 not in remaining_ids


# ── Duplicate Check ───────────────────────────────────────


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_check_duplicate_found(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    _add_item(client, content="unique fact xyz")
    resp = client.post("/knowledge/check-duplicate", json={"content": "unique fact xyz"})
    assert resp.status_code == 200
    body = _data(resp)
    assert body.get("is_duplicate") is True


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_check_duplicate_not_found(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    resp = client.post("/knowledge/check-duplicate", json={"content": "totally unique content"})
    assert resp.status_code == 200
    body = _data(resp)
    assert body.get("is_duplicate") is False


# ── Context ───────────────────────────────────────────────


@patch("domains.cognitive.rag_service.get_rag_service", side_effect=Exception("no rag"))
@patch(
    "domains.infrastructure.truth_labeler.get_truth_labeler", side_effect=Exception("no labeler")
)
def test_get_context(_mock_tl, _mock_rag):
    _cleanup()
    client = get_test_client()
    _add_item(client, content="context fact for retrieval")
    resp = client.get("/knowledge/context", params={"query": "context"})
    assert resp.status_code == 200
    body = _data(resp)
    assert isinstance(body, dict)


# ── Knowledge Gaps ────────────────────────────────────────


def test_knowledge_gaps():
    _cleanup()
    client = get_test_client()
    resp = client.get("/knowledge/gaps")
    assert resp.status_code == 200
    body = _data(resp)
    assert isinstance(body, dict)
