"""Tests for SessionStore — MogDB-backed chat session persistence."""
from __future__ import annotations

import time
from typing import Any

import pytest


@pytest.fixture
def store(tmp_path):
    from routers.session_store import SessionStore
    return SessionStore(db_path=str(tmp_path / "mogdb"), sync_dir=str(tmp_path / "json"))


class TestSessionStore:
    def test_create_session(self, store):
        session = store.create(name="Test Chat", model="gpt2")
        assert session["id"]
        assert session["name"] == "Test Chat"
        assert session["model"] == "gpt2"
        assert session["messages"] == []
        assert session["created_at"]
        assert session["updated_at"]
        assert session.get("archived") is not True
        assert session.get("starred") is not True
        assert session.get("pinned") is not True

    def test_get_session(self, store):
        created = store.create(name="Test")
        found = store.get(created["id"])
        assert found is not None
        assert found["name"] == "Test"
        assert found["id"] == created["id"]

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_list_sessions(self, store):
        store.create(name="Chat 1")
        store.create(name="Chat 2")
        sessions = store.list()
        assert len(sessions) == 2
        names = {s["name"] for s in sessions}
        assert names == {"Chat 1", "Chat 2"}

    def test_list_excludes_archived(self, store):
        s1 = store.create(name="Active")
        s2 = store.create(name="Archived")
        store.upsert(s2["id"], archived=True)
        sessions = store.list(include_archived=False)
        assert len(sessions) == 1
        assert sessions[0]["name"] == "Active"

    def test_list_includes_archived(self, store):
        s1 = store.create(name="Active")
        s2 = store.create(name="Archived")
        store.upsert(s2["id"], archived=True)
        sessions = store.list(include_archived=True)
        assert len(sessions) == 2

    def test_upsert_fields(self, store):
        session = store.create(name="Test")
        store.upsert(session["id"], starred=True, pinned=True)
        found = store.get(session["id"])
        assert found["starred"] is True
        assert found["pinned"] is True

    def test_upsert_nonexistent_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.upsert("nonexistent", starred=True)

    def test_append_message(self, store):
        session = store.create(name="Test")
        store.append_message(session["id"], role="user", content="Hello")
        store.append_message(session["id"], role="assistant", content="Hi there")
        found = store.get(session["id"])
        assert len(found["messages"]) == 2
        assert found["messages"][0]["role"] == "user"
        assert found["messages"][0]["content"] == "Hello"
        assert found["messages"][1]["role"] == "assistant"
        assert found["messages"][1]["content"] == "Hi there"
        assert found["messages"][0]["timestamp"]

    def test_append_message_nonexistent_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.append_message("nonexistent", role="user", content="Hi")

    def test_delete_session(self, store):
        session = store.create(name="Test")
        store.delete(session["id"])
        assert store.get(session["id"]) is None

    def test_delete_nonexistent_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.delete("nonexistent")

    def test_search_by_name(self, store):
        store.create(name="Python training")
        store.create(name="JS debugging")
        store.create(name="Python review")
        results = store.search("Python")
        assert len(results) == 2

    def test_search_by_message_content(self, store):
        s = store.create(name="Chat")
        store.append_message(s["id"], role="user", content="How do I train a model?")
        results = store.search("train")
        assert len(results) == 1

    def test_search_no_results(self, store):
        store.create(name="Chat")
        results = store.search("nonexistent")
        assert len(results) == 0

    def test_count(self, store):
        assert store.count() == 0
        store.create(name="A")
        store.create(name="B")
        assert store.count() == 2

    def test_count_archived(self, store):
        s1 = store.create(name="Active")
        s2 = store.create(name="Archived")
        store.upsert(s2["id"], archived=True)
        assert store.count(include_archived=False) == 1
        assert store.count(include_archived=True) == 2

    def test_create_with_messages(self, store):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        session = store.create(name="Test", messages=messages)
        assert len(session["messages"]) == 2
        assert session["messages"][0]["content"] == "Hello"

    def test_persistence_across_instances(self, tmp_path):
        from routers.session_store import SessionStore

        store1 = SessionStore(db_path=str(tmp_path / "mogdb"), sync_dir=str(tmp_path / "json"))
        created = store1.create(name="Persistent")
        store1.append_message(created["id"], role="user", content="Saved")

        store2 = SessionStore(db_path=str(tmp_path / "mogdb"), sync_dir=str(tmp_path / "json"))
        found = store2.get(created["id"])
        assert found is not None
        assert found["name"] == "Persistent"
        assert len(found["messages"]) == 1

    def test_list_sorted_by_updated(self, store):
        s1 = store.create(name="First")
        time.sleep(0.01)
        s2 = store.create(name="Second")
        store.upsert(s1["id"], starred=True)  # Touch s1's updated_at
        sessions = store.list()
        assert sessions[0]["name"] == "First"
        assert sessions[1]["name"] == "Second"
