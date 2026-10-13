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


# ── updated_at — recency contract ───────────────────────────────────────────


async def test_textfile_hits_carry_file_mtime_as_updated_at(tmp_path):
    docs = _seed_docs(tmp_path)
    adapter = TextFileAdapter("docs", [docs])
    hits = await adapter.search("setup", 10, CTX)
    assert hits
    for h in hits:
        assert h.updated_at == (docs / h.title).stat().st_mtime


async def test_jsonl_ts_field_numeric_and_absent(tmp_path):
    path = _board(
        tmp_path,
        [
            {"id": "old", "title": "match old", "description": "", "updated_at": 100.0},
            {"id": "new", "title": "match new", "description": "", "updated_at": 200.0},
            {"id": "undated", "title": "match undated", "description": ""},
        ],
    )
    adapter = JsonlRecordAdapter("c", path, "title", "description", "id", ts_field="updated_at")
    hits = {h.id: h for h in await adapter.search("match", 10, CTX)}
    assert hits["old"].updated_at == 100.0
    assert hits["new"].updated_at == 200.0
    assert hits["undated"].updated_at == 0.0  # missing field degrades, no raise


async def test_jsonl_ts_iso_string_parses(tmp_path):
    path = _board(
        tmp_path,
        [
            {
                "id": "c1",
                "title": "Iso card",
                "description": "",
                "updated_at": "2026-01-02T03:04:05+00:00",
            }
        ],
    )
    adapter = JsonlRecordAdapter("c", path, "title", "description", "id", ts_field="updated_at")
    hits = await adapter.search("iso", 10, CTX)
    assert hits and hits[0].updated_at > 0.0


async def test_jsonl_malformed_ts_degrades_to_zero(tmp_path):
    path = _board(
        tmp_path,
        [{"id": "c1", "title": "Bad ts card", "description": "", "updated_at": "not-a-date"}],
    )
    adapter = JsonlRecordAdapter("c", path, "title", "description", "id", ts_field="updated_at")
    hits = await adapter.search("bad ts", 10, CTX)
    assert hits and hits[0].updated_at == 0.0


async def test_jsonl_without_ts_field_ignores_the_column(tmp_path):
    """ts_field is a declaration — undeclared columns are never read."""
    path = _board(
        tmp_path,
        [{"id": "c1", "title": "Silent card", "description": "", "updated_at": 999.0}],
    )
    adapter = JsonlRecordAdapter("c", path, "title", "description", "id")
    hits = await adapter.search("silent", 10, CTX)
    assert hits and hits[0].updated_at == 0.0


# ── TextFileAdapter.sweep — two-pass staleness refresh ──────────────────────


def _touch(path, offset=5.0):
    """Force a distinct mtime (some FS have coarse timestamps)."""
    import os

    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + offset))


async def test_sweep_absorbs_new_upload_pass1(tmp_path):
    """Pass 1 (front, newest first): files that appeared since caching."""
    docs = _seed_docs(tmp_path)
    adapter = TextFileAdapter("docs", [docs])
    await adapter.search("setup", 10, CTX)  # cache the seeded pair
    fresh = docs / "fresh.md"
    fresh.write_text("brand new upload content\n")
    _touch(fresh)
    stats = adapter.sweep()
    assert stats["new"] == 1
    assert stats["refreshed"] == 0
    assert stats["evicted"] == 0
    assert fresh in adapter._cache, "pass 1 must index the latest upload"


async def test_sweep_brings_stale_doc_to_updated_state(tmp_path):
    """Pass 2 (back, oldest first): a modified doc nobody queried catches up."""
    docs = _seed_docs(tmp_path)
    target = docs / "architecture.md"
    adapter = TextFileAdapter("docs", [docs])
    await adapter.search("setup", 10, CTX)  # cache the old content
    assert not await adapter.search("zeta-version", 10, CTX)
    target.write_text("Layers: domain, routers, web. Now on zeta-version.\n")
    _touch(target)
    stats = adapter.sweep()
    assert stats["refreshed"] == 1
    assert stats["evicted"] == 0
    cached_lines = adapter._cache[target][1]
    assert any("zeta-version" in ln for ln in cached_lines), "sweep must re-read changed files"


async def test_sweep_evicts_deleted_docs(tmp_path):
    """No ghost entries: deleted files leave the cache."""
    docs = _seed_docs(tmp_path)
    adapter = TextFileAdapter("docs", [docs])
    await adapter.search("setup", 10, CTX)
    target = docs / "architecture.md"
    target.unlink()
    stats = adapter.sweep()
    assert stats["evicted"] == 1
    assert stats["refreshed"] == 0
    assert target not in adapter._cache


async def test_sweep_absent_root_is_all_zeroes(tmp_path):
    """Prod image without docs/ sweeps to zeroes, not an error."""
    adapter = TextFileAdapter("docs", [tmp_path / "does-not-exist"])
    assert adapter.sweep() == {"new": 0, "refreshed": 0, "unchanged": 0, "evicted": 0}


async def test_sweep_after_fresh_sweep_is_quiet(tmp_path):
    """Idempotence: a second sweep with no source change moves nothing."""
    docs = _seed_docs(tmp_path)
    adapter = TextFileAdapter("docs", [docs])
    adapter.sweep()  # first sweep indexes everything
    stats = adapter.sweep()
    assert stats == {"new": 0, "refreshed": 0, "unchanged": 2, "evicted": 0}
