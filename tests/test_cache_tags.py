"""Just-cache tags (``domain/training/_internal/cache_tags.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_classify_kind_adapters_system_media_dataset() -> None:
    from domain.training._internal.cache_tags import classify_kind

    assert classify_kind("user_adapters") == "adapter"
    assert classify_kind("knowledge_adapter") == "adapter"
    assert classify_kind("gallery") == "media"
    assert classify_kind("uploads") == "media"
    assert classify_kind("response_mogdb") == "system"
    assert classify_kind("errors_json") == "system"
    assert classify_kind("training_jobs.db") == "system"
    assert classify_kind("tinyshakespeare") == "dataset"
    assert classify_kind("uploads_json") == "dataset"


def test_classify_kind_npz_only_dir_is_adapter(tmp_path: Path) -> None:
    from domain.training._internal.cache_tags import classify_kind

    d = tmp_path / "lora_out"
    d.mkdir()
    (d / "adapter.npz").write_bytes(b"x")
    assert classify_kind("lora_out", d) == "adapter"


def test_guess_mime_known_extensions() -> None:
    from domain.training._internal.cache_tags import guess_mime

    assert guess_mime("corpus.jsonl") == "application/x-ndjson"
    assert guess_mime("input.txt") == "text/plain"
    assert guess_mime("data.csv") == "text/csv"
    assert guess_mime("adapter.npz") == "application/x-numpy"


def test_write_and_read_entry_meta_roundtrip(tmp_path: Path) -> None:
    from domain.training._internal.cache_tags import read_entry_meta, write_entry_meta

    d = tmp_path / "ds1"
    write_entry_meta(d, kind="dataset", tags=["dataset", "chat"], source="api")
    meta = read_entry_meta(d)
    assert meta["kind"] == "dataset"
    assert "chat" in meta["tags"]
    assert meta["source"] == "api"


def test_entry_tags_merges_stored_and_kind(tmp_path: Path) -> None:
    from domain.training._internal.cache_tags import entry_tags, write_entry_meta

    d = tmp_path / "ds1"
    (d).mkdir()
    (d / "corpus.jsonl").write_text('{"text": "hi"}\n')
    write_entry_meta(d, kind="dataset", tags=["chat"])
    tags = entry_tags(d, "dataset")
    assert "dataset" in tags
    assert "chat" in tags
    assert "jsonl" in tags


def test_resolve_in_cache_invalid_id_raises() -> None:
    import pytest

    from domain.training._internal.cache_tags import resolve_in_cache

    with pytest.raises(ValueError):
        resolve_in_cache("../escape")


def test_resolve_dataset_path_prefers_cache(tmp_path: Path, monkeypatch) -> None:
    from domain.training._internal import cache_tags
    from domain.training._internal.helpers import resolve_dataset_path

    cache = tmp_path / "cache"
    entry = cache / "mydata"
    entry.mkdir(parents=True)
    (entry / "input.txt").write_text("hello world, this is training text " * 20)
    monkeypatch.setattr(cache_tags, "get_cache_root", lambda: cache)
    assert resolve_dataset_path("mydata") == str(entry / "input.txt")


def test_find_corpus_file_priority(tmp_path: Path) -> None:
    from domain.training._internal.cache_tags import find_corpus_file

    d = tmp_path / "e"
    d.mkdir()
    (d / "train.txt").write_text("t")
    (d / "input.txt").write_text("i")
    assert find_corpus_file(d) == d / "input.txt"
