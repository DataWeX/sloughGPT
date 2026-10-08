"""File-backed adapters — docs/dev-notes TextFileAdapter, kanban JsonlRecordAdapter.

These carry the "development environment" half of system-wide search, so
they get the same contract treatment as DB stores plus file-specific
behaviors: mtime-keyed caching, one-hit-per-file, JSONL tolerance, and
absent corpus = empty (prod images without docs/ are fine).
"""

from __future__ import annotations

import json

from domain.search.adapters.filesources import JsonlRecordAdapter, TextFileAdapter
from domain.search.types import SearchContext

CTX = SearchContext(user_id="u1")
_NONMATCH = "zzznonexistentquery"


def _seed_docs(root):
    d = root / "docs"
    d.mkdir(parents=True)
    (d / "guide.md").write_text("# Setup Guide\nRun the dev stack first.\n")
    (d / "architecture.md").write_text("Layers: domain, routers, web.\n")
    return d


# ── TextFileAdapter contract ────────────────────────────────────────────────


async def test_contract_well_formed_hits(tmp_path):
    docs = _seed_docs(tmp_path)
    adapter = TextFileAdapter("docs", [docs])
    hits = await adapter.search("setup", 10, CTX)
    assert hits
    for h in hits:
        assert h.store == "docs"
        assert h.id and h.title
        assert 0.0 < h.score <= 1.0
        assert h.locator.startswith("file:")


async def test_contract_case_insensitive_limit_and_empty(tmp_path):
    docs = _seed_docs(tmp_path)
    adapter = TextFileAdapter("docs", [docs])
    base = await adapter.search("setup", 10, CTX)
    assert len(await adapter.search("SETUP", 10, CTX)) == len(base)
    assert len(await adapter.search("setup", 1, CTX)) <= 1
    assert await adapter.search(_NONMATCH, 10, CTX) == []


async def test_one_hit_per_file_first_matching_line(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "repeat.md").write_text("needle here\nneedle again\nneedle third\n")
    (docs / "other.md").write_text("nothing to see\n")
    adapter = TextFileAdapter("docs", [docs])
    hits = await adapter.search("needle", 10, CTX)
    assert len(hits) == 1
    assert hits[0].title.endswith("repeat.md")
    assert hits[0].locator == "file:repeat.md:1"
    assert "needle here" in hits[0].detail


async def test_absent_corpus_returns_empty_not_error(tmp_path):
    adapter = TextFileAdapter("docs", [tmp_path / "does-not-exist"])
    assert await adapter.search("anything", 10, CTX) == []


async def test_cache_refreshes_on_mtime_change(tmp_path):
    docs = _seed_docs(tmp_path)
    adapter = TextFileAdapter("docs", [docs])
    assert await adapter.search("second-wave", 10, CTX) == []
    target = docs / "guide.md"
    target.write_text("# Setup Guide\nA second-wave release lands here.\n")
    # Force a distinct mtime (some FS have coarse timestamps).
    import os

    stat = target.stat()
    os.utime(target, (stat.st_atime, stat.st_mtime + 5))
    hits = await adapter.search("second-wave", 10, CTX)
    assert len(hits) == 1, "stale mtime cache must not hide a changed file"


# ── JsonlRecordAdapter contract ─────────────────────────────────────────────


def _board(root, lines):
    path = root / "board.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


async def test_jsonl_contract(tmp_path):
    path = _board(
        tmp_path,
        [
            {"id": "c1", "title": "Fix search ranking", "description": "score ordering"},
            {"id": "c2", "title": "Polish docs", "description": "typos"},
        ],
    )
    adapter = JsonlRecordAdapter("kanban_cards", path, "title", "description", "id")
    hits = await adapter.search("ranking", 10, CTX)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.store == "kanban_cards"
    assert hit.id == "c1"
    assert hit.title == "Fix search ranking"
    assert hit.locator.startswith("file:")
    # Case-insensitive, empty-not-raise, limit.
    assert len(await adapter.search("RANKING", 10, CTX)) == 1
    assert await adapter.search(_NONMATCH, 10, CTX) == []
    assert len(await adapter.search("search", 1, CTX)) <= 1


async def test_jsonl_detail_field_match(tmp_path):
    path = _board(tmp_path, [{"id": "c1", "title": "Card", "description": "deep refactor"}])
    adapter = JsonlRecordAdapter("kanban_cards", path, "title", "description", "id")
    hits = await adapter.search("refactor", 10, CTX)
    assert len(hits) == 1
    assert hits[0].score < 1.0  # detail-only match scores below a title match


async def test_jsonl_tolerates_malformed_and_blank_lines(tmp_path):
    path = _board(tmp_path, [{"id": "c1", "title": "Good card", "description": ""}])
    with path.open("a") as f:
        f.write("not json at all\n\n")
    adapter = JsonlRecordAdapter("kanban_cards", path, "title", "description", "id")
    hits = await adapter.search("good", 10, CTX)
    assert len(hits) == 1
    assert await adapter.search("not json", 10, CTX) == []


async def test_jsonl_absent_file_returns_empty(tmp_path):
    adapter = JsonlRecordAdapter(
        "kanban_cards", tmp_path / "nope.jsonl", "title", "description", "id"
    )
    assert await adapter.search("anything", 10, CTX) == []


async def test_raw_json_keys_are_not_searchable(tmp_path):
    """Punctuation/keys must not match — records match on declared fields."""
    path = _board(tmp_path, [{"id": "c1", "title": "Quiet card", "description": ""}])
    adapter = JsonlRecordAdapter("kanban_cards", path, "title", "description", "id")
    # "description" is a JSON key present in the raw line but not a field value.
    assert await adapter.search("description", 10, CTX) == []
