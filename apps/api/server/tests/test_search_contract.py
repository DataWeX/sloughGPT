"""Contract suite — the enforcement of "one search over every store".

Every adapter, regardless of its backing data, must satisfy the same
expectations: well-formed hits, case-insensitive matching, honored
limits, and empty-on-no-match instead of exceptions. A new store joins
system-wide search by passing this file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from domain.search.adapters.datasets import DatasetsAdapter
from domain.search.adapters.knowledge import KnowledgeAdapter
from domain.search.adapters.members import MembersAdapter
from domain.search.adapters.training import TrainingJobsAdapter
from domain.search.types import SearchContext

# ── Seed data ────────────────────────────────────────────────────────────────

_NONMATCH = "zzznonexistentquery"


def _member_adapter() -> MembersAdapter:
    member = SimpleNamespace(id="m1", user_id="user1", role=SimpleNamespace(value="admin"))
    user = SimpleNamespace(username="alice")
    return MembersAdapter(
        ws_repo=SimpleNamespace(list_members=lambda ws_id: [member]),
        user_repo=SimpleNamespace(get=lambda uid: user),
    )


def _training_adapter() -> TrainingJobsAdapter:
    job = SimpleNamespace(id="t1", name="Fine-tune LLaMA", status="completed")
    return TrainingJobsAdapter(
        repo=SimpleNamespace(list_by_workspace=lambda ws_id: [job]),
    )


def _dataset_adapter() -> DatasetsAdapter:
    ds = SimpleNamespace(id="d1", name="training-data", format="jsonl", record_count=1000)
    return DatasetsAdapter(repo=SimpleNamespace(list=lambda: [ds]))


def _knowledge_adapter() -> KnowledgeAdapter:
    fact = SimpleNamespace(id="k1", content="Paris is the capital of France", topic="geography")
    return KnowledgeAdapter(repo=SimpleNamespace(list_facts=lambda: [fact]))


CASES = [
    pytest.param(_member_adapter, "alice", id="members"),
    pytest.param(_training_adapter, "fine-tune", id="training_jobs"),
    pytest.param(_dataset_adapter, "training", id="datasets"),
    pytest.param(_knowledge_adapter, "paris", id="knowledge"),
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
async def test_adapter_contract(factory, q: str):
    await _assert_contract(factory(), q)


async def test_registered_adapters_satisfy_the_protocol():
    """The registry is the source of truth — everything in it must be a
    Searchable with honest capability metadata. (Seeded behavior is
    enforced by the parametrized cases above; this checks the registry.)"""
    from domain.search import Searchable, get_registry

    registry = get_registry()
    names = set(registry.store_names())
    # DB-backed stores + the repository corpus (docs, dev notes, kanban).
    assert names == {
        "members",
        "training_jobs",
        "datasets",
        "knowledge",
        "docs",
        "dev_notes",
        "kanban_cards",
    }
    for adapter in registry.adapters():
        assert isinstance(adapter, Searchable), f"{adapter} violates Searchable"
        assert adapter.capability in ("live", "indexed")
        assert isinstance(adapter.workspace_scoped, bool)
