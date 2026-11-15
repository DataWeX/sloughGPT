"""Loadable memory card — save / load / restore lifecycle (TDD, card f74b9e75).

The card is a portable, checksummed snapshot of long-term memory facts
(``data/memory_cards/<name>.card.json``). Invariants under test:

- roundtrip fidelity (content/topic/source/url/timestamp/importance survive)
- checksum verified BEFORE any destructive step; corrupt cards never touch
  the store
- replace-load first autosaves current memory; if the autosave cannot be
  written the load aborts with the store untouched
- merge-load is idempotent (content-hash dedup)
- card names can never escape the cards directory
- ``ensure_active_card`` loads ``SLO_MEMORY_ACTIVE_CARD`` once, fail-closed
"""

from __future__ import annotations

import json

import pytest

from domain.memory import (
    KnowledgeFact,
    KnowledgeMemoryProvider,
    MemoryConfig,
    MemoryService,
)


@pytest.fixture
def iso_store(tmp_path, monkeypatch):
    """KnowledgeMemory isolated from the repo data dir (paths -> tmp)."""
    import domain.memory._internal.knowledge_store as ks
    from domain.inference._internal.vector_store import InMemoryVectorStore

    monkeypatch.setattr(ks, "ENTRIES_PATH", tmp_path / "entries.json")
    monkeypatch.setattr(ks, "VISITED_PATH", tmp_path / "visited.json")
    monkeypatch.setattr(ks, "_KNOWLEDGE_DB_PATH", str(tmp_path / "mogdb"))
    monkeypatch.setenv("SLO_MEMORY_CARDS_DIR", str(tmp_path / "cards"))
    return ks.KnowledgeMemory(vector_store=InMemoryVectorStore(dimension=384), load_persisted=False)


@pytest.fixture
def service(iso_store):
    return MemoryService(
        provider=KnowledgeMemoryProvider(store=iso_store),
        config=MemoryConfig(enabled=True),
    )


def _seed(store, content, topic="general", source="test", ts=123.5, imp=0.9, url=""):
    assert store.add_fact(
        KnowledgeFact(
            content=content,
            topic=topic,
            source=source,
            url=url,
            timestamp=ts,
            importance=imp,
        )
    )


# ── card file level ─────────────────────────────────────────────────────────


def test_save_then_load_roundtrip(service, iso_store):
    _seed(
        iso_store,
        "the model prefers violet accents",
        topic="ui",
        ts=42.25,
        imp=0.8,
        url="https://x",
    )
    assert iso_store.add_fact(
        KnowledgeFact(content="workspace scoped fact", topic="w", workspace_id="ws-1")
    )
    result = service.save_card(name="mem-1")
    assert result["ok"] is True, result
    assert result["name"] == "mem-1"
    assert result["facts_count"] == 2

    loaded = service.load_card("mem-1", mode="merge")
    assert loaded["ok"] is True, loaded
    assert loaded["imported"] == 0  # already present — dedup
    by_content = {f["content"]: f for f in iso_store.list_all(top_k=10)}
    item = by_content["the model prefers violet accents"]
    assert item["topic"] == "ui"
    assert item["source"] == "test"
    # workspace scope survives the card roundtrip
    assert by_content["workspace scoped fact"]["workspace_id"] == "ws-1"
    assert item["url"] == "https://x"
    assert item["timestamp"] == 42.25
    assert item["importance"] == 0.8


def test_save_roundtrip_into_fresh_store_preserves_facts(service, iso_store, tmp_path):
    _seed(iso_store, "fact alpha", topic="a")
    _seed(iso_store, "fact beta", topic="b", ts=7.0, imp=0.3)
    assert iso_store.add_fact(KnowledgeFact(content="fact gamma", topic="g", workspace_id="ws-9"))
    result = service.save_card(name="portable")
    assert result["ok"] and result["facts_count"] == 3

    iso_store.clear_all()
    assert iso_store.list_all(top_k=10) == []

    loaded = service.load_card("portable", mode="replace")
    assert loaded["ok"] and loaded["imported"] == 3, loaded
    by_content = {f["content"]: f for f in iso_store.list_all(top_k=10)}
    assert set(by_content) == {"fact alpha", "fact beta", "fact gamma"}
    assert by_content["fact beta"]["timestamp"] == 7.0
    assert by_content["fact beta"]["importance"] == 0.3
    assert by_content["fact beta"]["topic"] == "b"
    # workspace scope survives save -> clear -> load (through the card file)
    assert by_content["fact gamma"]["workspace_id"] == "ws-9"


def test_card_file_shape_and_checksum(service, iso_store, tmp_path):
    _seed(iso_store, "shape fact")
    service.save_card(name="shape")
    path = tmp_path / "cards" / "shape.card.json"
    assert path.exists()
    card = json.loads(path.read_text())
    assert card["format"] == "slo-memory-card/v1"
    assert card["name"] == "shape"
    assert card["facts_count"] == 1
    assert card["checksum"].startswith("sha256:")
    assert card["facts"][0]["content"] == "shape fact"
    # ids/scores are query artifacts and must NOT be part of a portable card
    assert "id" not in card["facts"][0]
    assert "score" not in card["facts"][0]


def test_duplicate_card_name_requires_overwrite(service, iso_store):
    _seed(iso_store, "x")
    assert service.save_card(name="dup")["ok"] is True
    again = service.save_card(name="dup")
    assert again["ok"] is False
    assert again["error_code"] == "exists"
    assert service.save_card(name="dup", overwrite=True)["ok"] is True


def test_automatic_card_name_is_unique(service, iso_store):
    _seed(iso_store, "x")
    first = service.save_card()
    second = service.save_card()
    assert first["ok"] and second["ok"]
    assert first["name"] != second["name"]


def test_load_missing_card_reports_not_found(service):
    result = service.load_card("no-such-card")
    assert result["ok"] is False
    assert result["error_code"] == "not_found"


def test_corrupt_card_rejected_before_touching_store(service, iso_store, tmp_path):
    _seed(iso_store, "precious current memory")
    service.save_card(name="good")
    _seed(iso_store, "another fact")

    path = tmp_path / "cards" / "good.card.json"
    card = json.loads(path.read_text())
    card["facts"][0]["content"] = "tampered content"  # checksum now mismatches
    path.write_text(json.dumps(card))

    result = service.load_card("good", mode="replace")
    assert result["ok"] is False
    assert result["error_code"] == "corrupt"
    contents = {f["content"] for f in iso_store.list_all(top_k=10)}
    assert contents == {"precious current memory", "another fact"}  # untouched


def test_truncated_card_json_rejected(service, iso_store, tmp_path):
    service.save_card(name="trunc")
    path = tmp_path / "cards" / "trunc.card.json"
    path.write_text(path.read_text()[:20])
    result = service.load_card("trunc")
    assert result["ok"] is False
    assert result["error_code"] == "corrupt"


def test_card_name_traversal_rejected(service):
    result = service.save_card(name="../evil")
    assert result["ok"] is False
    assert result["error_code"] == "invalid_name"
    result = service.load_card("..%2Fevil")
    assert result["ok"] is False
    assert result["error_code"] == "invalid_name"


# ── replace / merge semantics ───────────────────────────────────────────────


def test_replace_load_autosaves_current_memory(service, iso_store):
    _seed(iso_store, "card fact")
    service.save_card(name="src")

    iso_store.clear_all()
    _seed(iso_store, "current memory", topic="now")

    result = service.load_card("src", mode="replace")
    assert result["ok"] and result["mode"] == "replace", result
    assert result["backup"] and result["backup"].startswith("autosave-")
    assert {f["content"] for f in iso_store.list_all(top_k=10)} == {"card fact"}

    backup = service.load_card(result["backup"], mode="merge")
    assert backup["ok"]
    backup_contents = {f["content"] for f in iso_store.list_all(top_k=10)}
    assert "current memory" in backup_contents  # old memory recoverable


def test_replace_load_onto_empty_store_makes_no_autosave(service, iso_store):
    _seed(iso_store, "card fact")
    service.save_card(name="src")
    iso_store.clear_all()

    result = service.load_card("src", mode="replace")
    assert result["ok"] and result["backup"] is None
    cards = service.list_cards()
    assert all(not c["name"].startswith("autosave-") for c in cards["cards"])


def test_replace_load_aborts_when_autosave_fails(service, iso_store, monkeypatch):
    _seed(iso_store, "current memory")
    _seed(iso_store, "second current fact")
    service.save_card(name="src")
    iso_store.clear_all()
    _seed(iso_store, "fresh memory")

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    import domain.memory._internal.memory_card as mc

    monkeypatch.setattr(mc, "save_card", boom)
    result = service.load_card("src", mode="replace")
    assert result["ok"] is False
    assert result["error_code"] == "backup_failed"
    contents = {f["content"] for f in iso_store.list_all(top_k=10)}
    assert contents == {"fresh memory"}  # untouched — nothing was cleared


def test_merge_load_is_idempotent(service, iso_store):
    _seed(iso_store, "shared fact")
    _seed(iso_store, "card-only fact")
    service.save_card(name="src")

    iso_store.clear_all()
    _seed(iso_store, "shared fact")
    _seed(iso_store, "current-only fact")

    result = service.load_card("src", mode="merge")
    assert result["ok"] and result["mode"] == "merge"
    assert result["imported"] == 1 and result["skipped"] == 1
    contents = {f["content"] for f in iso_store.list_all(top_k=10)}
    assert contents == {"shared fact", "current-only fact", "card-only fact"}

    # second identical merge imports nothing — idempotent
    again = service.load_card("src", mode="merge")
    assert again["ok"] and again["imported"] == 0 and again["skipped"] == 2
    assert {f["content"] for f in iso_store.list_all(top_k=10)} == contents


def test_invalid_mode_rejected(service):
    result = service.load_card("anything", mode="overwrite")
    assert result["ok"] is False
    assert result["error_code"] == "invalid_mode"


# ── list / delete ───────────────────────────────────────────────────────────


def test_list_reports_validity(service, iso_store, tmp_path):
    _seed(iso_store, "x")
    service.save_card(name="valid-card")
    service.save_card(name="broken-card")

    path = tmp_path / "cards" / "broken-card.card.json"
    card = json.loads(path.read_text())
    card["facts"] = []  # tampered
    path.write_text(json.dumps(card))

    cards = {c["name"]: c for c in service.list_cards()["cards"]}
    assert cards["valid-card"]["valid"] is True
    assert cards["broken-card"]["valid"] is False
    assert cards["valid-card"]["facts_count"] == 1


def test_delete_card(service, iso_store):
    _seed(iso_store, "x")
    service.save_card(name="gone")
    assert service.delete_card("gone")["deleted"] is True
    assert service.delete_card("gone")["deleted"] is False
    assert service.delete_card("never-existed")["deleted"] is False


# ── disabled memory master switch ───────────────────────────────────────────


def test_disabled_memory_blocks_save_and_load(iso_store):
    disabled = MemoryService(
        provider=KnowledgeMemoryProvider(store=iso_store),
        config=MemoryConfig(enabled=False),
    )
    save = disabled.save_card(name="nope")
    assert save["ok"] is False and save["error_code"] == "disabled"
    load = disabled.load_card("nope")
    assert load["ok"] is False and load["error_code"] == "disabled"
    # listing stays available so cards remain manageable while disabled
    assert disabled.list_cards()["ok"] is True


# ── ensure_active_card (model-load integration) ─────────────────────────────


def test_ensure_active_card_env_unset_is_noop(service, monkeypatch):
    monkeypatch.delenv("SLO_MEMORY_ACTIVE_CARD", raising=False)
    assert service.ensure_active_card() is False


def test_ensure_active_card_loads_once(service, iso_store, monkeypatch):
    _seed(iso_store, "remembered fact")
    service.save_card(name="active")
    iso_store.clear_all()

    monkeypatch.setenv("SLO_MEMORY_ACTIVE_CARD", "active")
    assert service.ensure_active_card() is True
    assert {f["content"] for f in iso_store.list_all(top_k=10)} == {"remembered fact"}

    iso_store.clear_all()
    assert service.ensure_active_card() is False  # already loaded this session
    assert iso_store.list_all(top_k=10) == []  # no second load


def test_ensure_active_card_corrupt_fails_closed(service, iso_store, monkeypatch, tmp_path):
    service.save_card(name="broken", overwrite=True)
    path = tmp_path / "cards" / "broken.card.json"
    path.write_text("{}")
    monkeypatch.setenv("SLO_MEMORY_ACTIVE_CARD", "broken")

    assert service.ensure_active_card() is False  # no crash, store untouched
