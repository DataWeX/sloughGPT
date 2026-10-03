"""Contract suite — the enforcement of "one search over every store".

Every adapter, regardless of its backing data, must satisfy the same
expectations: well-formed hits, case-insensitive matching, honored
limits, and empty-on-no-match instead of exceptions. A new store joins
system-wide search by passing this file.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from domain.search.adapters.datasets import DatasetsAdapter
from domain.search.adapters.files import FilesAdapter
from domain.search.adapters.kb import KBAdapter
from domain.search.adapters.knowledge import KnowledgeAdapter
from domain.search.adapters.members import MembersAdapter
from domain.search.adapters.memory import MemoryAdapter
from domain.search.adapters.sessions import SessionsAdapter
from domain.search.adapters.training import TrainingJobsAdapter
from domain.search.session_index import SessionSearchIndex
from domain.search.types import SearchContext

# ── Seed data ────────────────────────────────────────────────────────────────

_NONMATCH = "zzznonexistentquery"


def _member_adapter(tmp_path=None) -> MembersAdapter:
    member = SimpleNamespace(id="m1", user_id="user1", role=SimpleNamespace(value="admin"))
    user = SimpleNamespace(username="alice")
    return MembersAdapter(
        ws_repo=SimpleNamespace(list_members=lambda ws_id: [member]),
        user_repo=SimpleNamespace(get=lambda uid: user),
    )


def _training_adapter(tmp_path=None) -> TrainingJobsAdapter:
    # JobStore returns plain dicts — mirror the real row shape.
    job = {"id": "t1", "name": "Fine-tune LLaMA", "status": "completed"}
    return TrainingJobsAdapter(
        repo=SimpleNamespace(list_by_workspace=lambda ws_id: [job]),
    )


def _dataset_adapter(tmp_path=None) -> DatasetsAdapter:
    ds = SimpleNamespace(id="d1", name="training-data", format="jsonl", record_count=1000)
    return DatasetsAdapter(repo=SimpleNamespace(list=lambda: [ds]))


def _knowledge_adapter(tmp_path=None) -> KnowledgeAdapter:
    fact = SimpleNamespace(id="k1", content="Paris is the capital of France", topic="geography")
    return KnowledgeAdapter(repo=SimpleNamespace(list_facts=lambda: [fact]))


def _files_adapter(tmp_path) -> FilesAdapter:
    blob = tmp_path / "quarterly-report.pdf"
    blob.write_bytes(b"%PDF-1.4")
    docs = [
        {
            "file_id": "f1",
            "filename": "quarterly-report.pdf",
            "original_name": "quarterly-report.pdf",
            "extension": ".pdf",
            "tags": ["finance"],
        }
    ]
    return FilesAdapter(docs_factory=lambda: docs, uploads_dir=tmp_path)


class _FakeEngine:
    """Seed-faithful KnowledgeEngine stand-in: casefold substring, success envelope."""

    def __init__(self):
        self._seeded = [
            {
                "id": "k1",
                "content": "Paris is the capital of France",
                "topic": "geography",
                "score": 1.0,
            }
        ]

    def query(self, search: str, top_k: int = 10):
        needle = search.casefold()
        data = [
            e
            for e in self._seeded
            if needle in e["content"].casefold() or needle in e["topic"].casefold()
        ][:top_k]
        return SimpleNamespace(data=data, success=True)


def _kb_adapter(tmp_path=None) -> KBAdapter:
    return KBAdapter(engine=_FakeEngine())


class _FakeMemoryService:
    def __init__(self):
        self._seeded = [
            {"id": "m1", "content": "Remember the release checklist", "topic": "ops", "score": 1.0}
        ]

    def retrieve(self, query: str, limit: int | None = None):
        needle = query.casefold()
        return [
            i
            for i in self._seeded
            if needle in i["content"].casefold() or needle in i["topic"].casefold()
        ][: limit or 5]


def _memory_adapter(tmp_path=None) -> MemoryAdapter:
    return MemoryAdapter(service=_FakeMemoryService())


def _sessions_adapter(tmp_path) -> SessionsAdapter:
    d = tmp_path / "chat_sessions"
    d.mkdir()
    (d / "s1.json").write_text(
        json.dumps(
            {
                "id": "s1",
                "name": "Deployment playbook",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-02T00:00:00Z",
                "messages": [{"role": "user", "content": "walk me through the rollout"}],
            }
        )
    )
    index = SessionSearchIndex(max_age_seconds=0.0, data_dirs=[d])
    return SessionsAdapter(index=index)


CASES = [
    pytest.param(_member_adapter, "alice", id="members"),
    pytest.param(_training_adapter, "fine-tune", id="training_jobs"),
    pytest.param(_dataset_adapter, "training", id="datasets"),
    pytest.param(_knowledge_adapter, "paris", id="knowledge"),
    pytest.param(_files_adapter, "quarterly", id="files"),
    pytest.param(_kb_adapter, "capital", id="kb"),
    pytest.param(_memory_adapter, "release", id="memory"),
    pytest.param(_sessions_adapter, "deployment", id="sessions"),
]

CTX = SearchContext(workspace_id="ws1", user_id="user1", is_admin=False)


# ── Contract ─────────────────────────────────────────────────────────────────


async def _assert_contract(adapter, sample_q: str) -> None:
    # 1. Protocol conformance (capability metadata + shape).
    assert adapter.store
    assert adapter.capability in ("live", "indexed")
    assert isinstance(adapter.workspace_scoped, bool)

    # 2. A matching query returns well-formed hits.
    hits = await adapter.search(sample_q, 10, CTX)
    assert hits, f"{adapter.store}: expected a hit for {sample_q!r}"
    for h in hits:
        assert h.store == adapter.store, "hit.store must equal adapter.store"
        assert h.id, "hit.id must be non-empty"
        assert h.title, "hit.title must be non-empty"
        assert 0.0 < h.score <= 1.0, f"score out of range: {h.score}"
        assert h.locator, "hit.locator must be navigable"

    # 3. Matching is case-insensitive.
    upper = await adapter.search(sample_q.upper(), 10, CTX)
    assert len(upper) == len(hits), "case-insensitivity broken"

    # 4. Limits are honored.
    one = await adapter.search(sample_q, 1, CTX)
    assert len(one) <= 1, "limit not honored"

    # 5. No match => empty list, NOT an exception.
    none = await adapter.search(_NONMATCH, 10, CTX)
    assert none == [], "no-match must return [] rather than raise"


@pytest.mark.parametrize("factory,q", CASES)
async def test_adapter_contract(factory, q: str, tmp_path):
    await _assert_contract(factory(tmp_path), q)


async def test_registered_adapters_satisfy_the_protocol():
    """The registry is the source of truth — everything in it must be a
    Searchable with honest capability metadata. (Seeded behavior is
    enforced by the parametrized cases above; this checks the registry.)"""
    from domain.search import Searchable, get_registry

    registry = get_registry()
    names = set(registry.store_names())
    # DB/API stores + sessions + the repository corpus (docs, dev notes, kanban).
    assert names == {
        "members",
        "training_jobs",
        "datasets",
        "knowledge",
        "kb",
        "memory",
        "files",
        "sessions",
        "docs",
        "dev_notes",
        "kanban_cards",
    }
    for adapter in registry.adapters():
        assert isinstance(adapter, Searchable), f"{adapter} violates Searchable"
        assert adapter.capability in ("live", "indexed")
        assert isinstance(adapter.workspace_scoped, bool)
