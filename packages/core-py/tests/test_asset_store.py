"""
Tests for the AssetStore (unified file/image namespace: CAS blobs + metadata
+ event history).
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from domain.infrastructure.asset_store import ORIGINS, AssetMeta, AssetStore


def _store(tmp_path: Path) -> AssetStore:
    return AssetStore(tmp_path / "assets")


# ── put / get ──


def test_put_get_roundtrip(tmp_path):
    store = _store(tmp_path)
    data = b"hello asset store"
    meta = store.put(data, original_name="note.txt", origin="upload")

    assert isinstance(meta, AssetMeta)
    assert meta.sha256 == hashlib.sha256(data).hexdigest()
    assert meta.size_bytes == len(data)
    assert meta.media_type == "text/plain"
    assert meta.kind == "file"
    assert meta.origin == "upload"

    assert store.get_bytes(meta.id) == data
    assert store.get_meta(meta.id) == meta


def test_put_cas_layout_and_dedupe(tmp_path):
    store = _store(tmp_path)
    data = b"same bytes"
    first = store.put(data, original_name="a.txt", origin="upload")
    second = store.put(data, original_name="b.txt", origin="upload")

    # One blob for identical content, two logical assets.
    blobs = list((tmp_path / "assets" / "blobs").rglob("*"))
    blob_files = [p for p in blobs if p.is_file()]
    assert len(blob_files) == 1
    assert blob_files[0].name == first.sha256 == second.sha256
    assert first.id != second.id
    # Fan-out layout: blobs/<sha[:2]>/<sha[2:4]>/<sha>
    assert blob_files[0].parent.name == first.sha256[2:4]
    assert blob_files[0].parent.parent.name == first.sha256[:2]

    assert {m.id for m in store.find_by_sha(first.sha256)} == {first.id, second.id}


def test_put_image_kind_and_media_type(tmp_path):
    store = _store(tmp_path)
    meta = store.put(b"\x89PNG", original_name="pic.png", origin="generated")
    assert meta.kind == "image"
    assert meta.media_type == "image/png"
    assert [m.id for m in store.list(kind="image")] == [meta.id]
    assert [m.id for m in store.list(origin="generated")] == [meta.id]


def test_put_rejects_invalid_origin_and_kind(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="invalid origin"):
        store.put(b"x", original_name="x", origin="magic")
    with pytest.raises(ValueError, match="invalid kind"):
        store.put(b"x", original_name="x", origin="upload", kind="video")


def test_reserved_computer_use_origins_accepted(tmp_path):
    store = _store(tmp_path)
    for origin in ORIGINS:
        meta = store.put(b"data", original_name=f"{origin}.bin", origin=origin)
        assert meta.origin == origin


def test_metadata_survives_new_instance(tmp_path):
    store = _store(tmp_path)
    meta = store.put(b"durable", original_name="d.txt", origin="upload", tags=["keep"])
    reopened = _store(tmp_path)
    loaded = reopened.get_meta(meta.id)
    assert loaded is not None
    assert loaded.original_name == "d.txt"
    assert loaded.tags == ["keep"]


# ── delete ──


def test_delete_keeps_blob_by_default(tmp_path):
    store = _store(tmp_path)
    meta = store.put(b"keep me", original_name="k.txt", origin="upload")
    assert store.delete(meta.id) is True
    assert store.get_meta(meta.id) is None
    assert (
        tmp_path / "assets" / "blobs" / meta.sha256[:2] / meta.sha256[2:4] / meta.sha256
    ).exists()


def test_delete_purge_blob_when_unreferenced(tmp_path):
    store = _store(tmp_path)
    meta = store.put(b"purge me", original_name="p.txt", origin="upload")
    assert store.delete(meta.id, purge_blob=True) is True
    blob = tmp_path / "assets" / "blobs" / meta.sha256[:2] / meta.sha256[2:4] / meta.sha256
    assert not blob.exists()


def test_delete_purge_keeps_shared_blob(tmp_path):
    store = _store(tmp_path)
    a = store.put(b"shared", original_name="a.txt", origin="upload")
    b = store.put(b"shared", original_name="b.txt", origin="upload")
    assert store.delete(a.id, purge_blob=True) is True
    blob = tmp_path / "assets" / "blobs" / a.sha256[:2] / a.sha256[2:4] / a.sha256
    assert blob.exists()
    assert store.get_bytes(b.id) == b"shared"


def test_delete_missing_returns_false(tmp_path):
    store = _store(tmp_path)
    assert store.delete("nope") is False


# ── events / history ──


def test_events_recorded_for_put_and_delete(tmp_path):
    store = _store(tmp_path)
    meta = store.put(b"e", original_name="e.txt", origin="upload")
    store.record_event(
        "download", asset_id=meta.id, sha256=meta.sha256, original_name="e.txt", origin="upload"
    )
    store.delete(meta.id)

    actions = [e["action"] for e in store.history(asset_id=meta.id)]
    assert actions == ["delete", "download", "put"]  # newest first
    assert [e["action"] for e in store.history(action="put")] == ["put"]


def test_history_limit_and_corrupt_lines(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        store.put(f"payload-{i}".encode(), original_name=f"f{i}.txt", origin="upload")
    events_path = tmp_path / "assets" / "events.jsonl"
    with open(events_path, "a", encoding="utf-8") as f:
        f.write("{not json\n")
    assert len(store.history(limit=3)) == 3
    assert all(e["action"] == "put" for e in store.history())


# ── migration ──


def test_migrate_uploads_copies_not_moves(tmp_path):
    store = _store(tmp_path)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    legacy = uploads / "fid123.txt"
    legacy.write_text("legacy content")

    result = store.migrate_uploads(
        [(legacy, {"filename": "fid123.txt", "original_name": "报告.txt", "tags": ["legacy"]})]
    )
    assert result == {"scanned": 1, "migrated": 1, "skipped": 0}
    # Original untouched (file-safety), bytes copied into CAS.
    assert legacy.exists()
    assert legacy.read_text() == "legacy content"
    (asset,) = store.list(origin="upload")
    assert asset.legacy_path == str(legacy)
    assert asset.original_name == "报告.txt"
    assert asset.tags == ["legacy"]
    assert store.get_bytes(asset.id) == b"legacy content"
    # migrate event recorded
    assert any(e["action"] == "migrate" for e in store.history(asset_id=asset.id))


def test_migrate_uploads_idempotent(tmp_path):
    store = _store(tmp_path)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    legacy = uploads / "fid.txt"
    legacy.write_text("x")
    first = store.migrate_uploads([(legacy, {"filename": "fid.txt"})])
    second = store.migrate_uploads([(legacy, {"filename": "fid.txt"})])
    assert first["migrated"] == 1
    assert second == {"scanned": 1, "migrated": 0, "skipped": 1}
    assert len(store.list()) == 1


def test_migrate_gallery_generated_images(tmp_path):
    store = _store(tmp_path)
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    png = gallery / "generated_1717.png"
    png.write_bytes(b"\x89PNG-fake")
    result = store.migrate_gallery([png])
    assert result["migrated"] == 1
    (asset,) = store.list(origin="generated")
    assert asset.kind == "image"
    assert asset.media_type == "image/png"
    assert asset.legacy_path == str(png)


def test_migrate_skips_missing_paths(tmp_path):
    store = _store(tmp_path)
    result = store.migrate_uploads([(tmp_path / "ghost.txt", {"filename": "ghost.txt"})])
    assert result == {"scanned": 0, "migrated": 0, "skipped": 0}


# ── search / list ──


def test_search_matches_names_and_tags(tmp_path):
    store = _store(tmp_path)
    a = store.put(b"1", original_name="Report Final.pdf", origin="upload", tags=["finance"])
    store.put(b"2", original_name="photo.png", origin="generated")
    assert [m.id for m in store.search("report")] == [a.id]
    assert [m.id for m in store.search("FINANCE")] == [a.id]
    assert store.search("nothing") == []


def test_events_jsonl_is_valid_json_lines(tmp_path):
    store = _store(tmp_path)
    store.put(b"j", original_name="j.txt", origin="upload")
    events_path = tmp_path / "assets" / "events.jsonl"
    for line in events_path.read_text(encoding="utf-8").splitlines():
        json.loads(line)


# ── LRU cache ──


def _blob_path(tmp_path: Path, meta) -> Path:
    return tmp_path / "assets" / "blobs" / meta.sha256[:2] / meta.sha256[2:4] / meta.sha256


def test_lru_cache_hit_serves_cached_bytes(tmp_path):
    store = AssetStore(tmp_path / "assets", cache_bytes=1024 * 1024)
    meta = store.put(b"cache me", original_name="c.txt", origin="upload")
    assert store.get_bytes(meta.id) == b"cache me"  # populates the cache
    _blob_path(tmp_path, meta).write_bytes(b"tampered on disk")
    assert store.get_bytes(meta.id) == b"cache me"  # served from cache


def test_lru_evicts_oldest_at_budget(tmp_path):
    store = AssetStore(tmp_path / "assets", cache_bytes=10)
    a = store.put(b"12345", original_name="a.txt", origin="upload")
    b = store.put(b"67890", original_name="b.txt", origin="upload")
    c = store.put(b"abcde", original_name="c.txt", origin="upload")
    store.get_bytes(a.id)  # 5 B cached
    store.get_bytes(b.id)  # 10 B == budget
    store.get_bytes(c.id)  # 15 B > budget -> evict a (oldest)
    _blob_path(tmp_path, a).write_bytes(b"EVICT")  # 5 B: re-reading a evicts exactly b
    _blob_path(tmp_path, c).write_bytes(b"CHANG")
    assert store.get_bytes(a.id) == b"EVICT"  # evicted -> disk read
    assert store.get_bytes(c.id) == b"abcde"  # still cached


def test_delete_evicts_cache_entry(tmp_path):
    store = AssetStore(tmp_path / "assets", cache_bytes=1024)
    meta = store.put(b"gone", original_name="g.txt", origin="upload")
    assert store.get_bytes(meta.id) == b"gone"
    store.delete(meta.id, purge_blob=True)
    assert store.get_bytes(meta.id) is None


def test_cache_disabled_with_zero_budget(tmp_path):
    store = AssetStore(tmp_path / "assets", cache_bytes=0)
    meta = store.put(b"x", original_name="x.txt", origin="upload")
    assert store.get_bytes(meta.id) == b"x"
    _blob_path(tmp_path, meta).write_bytes(b"fresh")
    assert store.get_bytes(meta.id) == b"fresh"  # no caching at all


# ── operator surface ──


def test_verify_integrity_healthy(tmp_path):
    store = _store(tmp_path)
    store.put(b"a", original_name="a.txt", origin="upload")
    store.put(b"b", original_name="b.png", origin="generated")
    report = store.verify_integrity()
    assert report["assets"] == 2
    assert report["blobs_on_disk"] == 2
    assert report["missing_blobs"] == 0
    assert report["sha_mismatches"] == 0
    assert report["orphan_blobs"] == 0
    assert report["healthy"] is True


def test_verify_detects_missing_mismatch_and_orphan(tmp_path):
    store = _store(tmp_path)
    a = store.put(b"a-bytes", original_name="a.txt", origin="upload")
    b = store.put(b"b-bytes", original_name="b.txt", origin="upload")
    c = store.put(b"c-bytes", original_name="c.txt", origin="upload")
    _blob_path(tmp_path, b).write_bytes(b"tampered")  # sha mismatch
    _blob_path(tmp_path, c).unlink()  # missing blob
    store.delete(a.id)  # blob stays -> orphan
    report = store.verify_integrity()
    assert report["missing_blobs"] == 1
    assert report["sha_mismatches"] == 1
    assert report["orphan_blobs"] == 1
    assert report["healthy"] is False


def test_prune_removes_orphans_only(tmp_path):
    store = _store(tmp_path)
    a = store.put(b"aaa", original_name="a.txt", origin="upload")
    b = store.put(b"bbb", original_name="b.txt", origin="upload")
    store.delete(a.id)  # blob kept -> orphan
    result = store.prune_orphan_blobs()
    assert result == {"removed": 1, "freed_bytes": 3}
    assert not _blob_path(tmp_path, a).exists()
    assert store.get_bytes(b.id) == b"bbb"  # referenced blob untouched


def test_reindex_repairs_drift_from_blob(tmp_path):
    store = _store(tmp_path)
    meta = store.put(b"original", original_name="r.txt", origin="upload")
    _blob_path(tmp_path, meta).write_bytes(b"rewritten-content")  # drifts sha + size
    result = store.reindex()
    assert result["repaired"] == 1
    reloaded = store.get_meta(meta.id)
    assert reloaded.sha256 == hashlib.sha256(b"rewritten-content").hexdigest()
    assert reloaded.size_bytes == len(b"rewritten-content")
    assert store.verify_integrity()["healthy"] is True


def test_reindex_reports_missing_blob_without_deleting(tmp_path):
    store = _store(tmp_path)
    meta = store.put(b"soon gone", original_name="m.txt", origin="upload")
    _blob_path(tmp_path, meta).unlink()
    result = store.reindex()
    assert result["missing_blobs"] == [meta.id]
    assert store.get_meta(meta.id) is not None  # row preserved — human decides


# ── operator CLI (external utility tier) ──


def _run_cli(tmp_path: Path, *args: str):
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "SLO_DATA_DIR": str(tmp_path / "data")}
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "asset_store_ops.py"), *args],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
        timeout=60,
    )


def test_cli_verify_and_history(tmp_path):
    proc = _run_cli(tmp_path, "verify")
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report == {
        "assets": 0,
        "blobs_on_disk": 0,
        "missing_blobs": 0,
        "sha_mismatches": 0,
        "orphan_blobs": 0,
        "orphan_bytes": 0,
        "healthy": True,
    }
    proc = _run_cli(tmp_path, "history")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_cli_prune_and_reindex_on_seeded_store(tmp_path):
    store = AssetStore(tmp_path / "data" / "assets")
    a = store.put(b"seed-a", original_name="a.txt", origin="upload")
    store.put(b"seed-b", original_name="b.txt", origin="upload")
    store.delete(a.id)  # orphan blob
    proc = _run_cli(tmp_path, "prune")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"removed": 1, "freed_bytes": 6}
    proc = _run_cli(tmp_path, "reindex")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["repaired"] == 0
