"""SearchService behavior — fan-out, limits, timeouts, visibility.

Fake adapters here; the contract suite covers real stores.
"""

from __future__ import annotations

import asyncio

import pytest

from domain.search.registry import SearchRegistry
from domain.search.service import SearchService
from domain.search.types import SearchContext, SearchHit

CTX = SearchContext(workspace_id="ws1", user_id="user1")


def _hit(store: str, title: str, score: float) -> SearchHit:
    return SearchHit(id=f"{store}:{title}", store=store, title=title, score=score)


class FakeAdapter:
    def __init__(self, store, hits=None, delay_s=0.0, fail=False, overshoot=False):
        self.store = store
        self.capability = "live"
        self.workspace_scoped = False
        self._hits = hits or []
        self._delay = delay_s
        self._fail = fail
        self._overshoot = overshoot

    async def search(self, q, limit, ctx):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError(f"{self.store} is down")
        if self._overshoot:
            return [_hit(self.store, f"extra-{i}", 0.5) for i in range(limit + 5)]
        return [h for h in self._hits if q.casefold() in h.title.casefold()][:limit]


class ScopedAdapter(FakeAdapter):
    def __init__(self, store):
        super().__init__(store, hits=[_hit(store, "scoped-thing", 0.9)])
        self.workspace_scoped = True
        self.seen_ctx = None

    async def search(self, q, limit, ctx):
        self.seen_ctx = ctx
        return await super().search(q, limit, ctx)


def _service(*adapters) -> SearchService:
    reg = SearchRegistry()
    for a in adapters:
        reg.register(a)
    return SearchService(reg)


# ── Merging & ranking ───────────────────────────────────────────────────────


async def test_merges_hits_across_stores_sorted_by_score():
    a = FakeAdapter("aaa", hits=[_hit("aaa", "alpha", 0.6)])
    b = FakeAdapter("bbb", hits=[_hit("bbb", "beta", 0.95)])
    result = await _service(a, b).search("a", CTX)
    assert [h.store for h in result.hits] == ["bbb", "aaa"]  # score desc
    assert result.partial == []
    assert result.skipped == []


async def test_empty_q_rejected():
    with pytest.raises(ValueError):
        await _service().search("   ", CTX)


async def test_unknown_store_rejected():
    with pytest.raises(KeyError):
        await _service(FakeAdapter("aaa")).search("a", CTX, stores=["nope"])


# ── Failure isolation ───────────────────────────────────────────────────────


async def test_failing_store_lands_in_partial_others_survive():
    good = FakeAdapter("good", hits=[_hit("good", "alpha", 0.8)])
    bad = FakeAdapter("bad", fail=True)
    result = await _service(good, bad).search("alpha", CTX)
    assert [h.store for h in result.hits] == ["good"]
    assert result.partial == ["bad"]


async def test_hung_store_times_out_into_partial():
    good = FakeAdapter("good", hits=[_hit("good", "alpha", 0.8)])
    hung = FakeAdapter("hung", delay_s=5.0)
    result = await _service(good, hung).search("alpha", CTX, timeout_s=0.05)
    assert [h.store for h in result.hits] == ["good"]
    assert result.partial == ["hung"]


# ── Per-store limits ────────────────────────────────────────────────────────


async def test_overshooting_store_is_trimmed_by_service():
    greedy = FakeAdapter("greedy", overshoot=True)
    result = await _service(greedy).search("extra", CTX, limit_per_store=3)
    assert len(result.hits) == 3


async def test_chatty_store_cannot_starve_others():
    chatty = FakeAdapter("chatty", overshoot=True)
    quiet = FakeAdapter(
        "quiet",
        hits=[_hit("quiet", "extra find", 0.9), _hit("quiet", "extra two", 0.7)],
    )
    result = await _service(chatty, quiet).search("extra", CTX, limit_per_store=2)
    assert len(result.hits) == 4  # 2 each
    assert {"chatty", "quiet"} == {h.store for h in result.hits}


# ── Visibility ──────────────────────────────────────────────────────────────


async def test_scoped_store_skipped_without_workspace():
    scoped = ScopedAdapter("scoped")
    result = await _service(scoped).search("scoped", SearchContext(user_id="u1"))
    assert result.hits == []
    assert result.skipped == ["scoped"]


async def test_scoped_store_receives_context_with_workspace():
    scoped = ScopedAdapter("scoped")
    result = await _service(scoped).search("scoped", CTX)
    assert len(result.hits) == 1
    assert result.skipped == []
    assert scoped.seen_ctx.workspace_id == "ws1"
