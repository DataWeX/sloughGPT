"""Tests for domain.files.FileStore — codec policy, round-trips, legacy.

Codec expectations are pinned to the proven policy (card fba1ce5e):
text/code → zstd, model weights → raw, unknown → zstd-if-smaller.
"""

from __future__ import annotations

import hashlib
import io
import os

import pytest

from domain.files import FileStore

TEXT = (b"def handler(event):\n    return {'ok': True}\n" * 500)
MODEL_BYTES = os.urandom(64 * 1024)  # genuinely incompressible


@pytest.fixture
def store(tmp_path):
    return FileStore(root=tmp_path / "uploads")


class TestCodecPolicy:
    def test_text_class_compresses(self, store):
        meta = store.put(TEXT, "handler.py")
        assert meta["codec"] == "zstd"
        assert meta["stored_size"] < meta["size_bytes"] / 3

    def test_model_class_passthrough(self, store):
        meta = store.put(MODEL_BYTES, "weights.gguf")
        assert meta["codec"] == "raw"
        assert meta["stored_size"] == meta["size_bytes"]

    def test_all_model_extensions_map_raw(self):
        for ext in (".gguf", ".safetensors", ".pt", ".pth", ".ckpt", ".onnx", ".bin", ".npz"):
            assert FileStore.codec_for(f"m{ext}") == "raw"

    def test_unknown_incompressible_prefers_raw(self, store):
        meta = store.put(MODEL_BYTES, "blob.unknownext")
        assert meta["codec"] == "raw"

    def test_unknown_compressible_prefers_zstd(self, store):
        meta = store.put(TEXT, "notes.weird")
        assert meta["codec"] == "zstd"

    def test_empty_file_stored_raw(self, store):
        meta = store.put(b"", "empty.txt")
        assert meta["codec"] == "raw"
        data, _ = store.get(meta["file_id"])
        assert data == b""


class TestRoundTrip:
    def test_get_returns_identical_bytes(self, store):
        meta = store.put(TEXT, "handler.py")
        data, m2 = store.get(meta["file_id"])
        assert data == TEXT
        assert m2["sha256"] == hashlib.sha256(TEXT).hexdigest()

    def test_put_stream_chunked_matches_put(self, store, tmp_path):
        big = TEXT * 40  # ~9 MiB, crosses multiple chunks
        m1 = store.put(big, "big.py")
        m2 = store.put_stream(io.BytesIO(big), "big2.py")
        assert store.get(m1["file_id"])[0] == store.get(m2["file_id"])[0]
        assert m2["size_bytes"] == len(big)

    def test_model_roundtrip(self, store):
        meta = store.put(MODEL_BYTES, "weights.safetensors")
        data, _ = store.get(meta["file_id"])
        assert data == MODEL_BYTES

    def test_max_bytes_enforced(self, store):
        with pytest.raises(ValueError, match="too large"):
            store.put_stream(io.BytesIO(TEXT), "huge.py", max_bytes=1024)

    def test_no_extension_rejected(self, store):
        with pytest.raises(ValueError, match="extension"):
            store.put(b"x", "noext")

    def test_iter_chunks_reassembles(self, store):
        meta = store.put(TEXT, "handler.py")
        joined = b"".join(store.iter_chunks(meta["file_id"], chunk_size=64))
        assert joined == TEXT


class TestCollectionOps:
    def test_list_and_delete(self, store):
        m1 = store.put(TEXT, "a.py")
        m2 = store.put(MODEL_BYTES, "b.gguf")
        listed = {m["file_id"] for m in store.list()}
        assert listed == {m1["file_id"], m2["file_id"]}
        store.delete(m1["file_id"])
        listed = {m["file_id"] for m in store.list()}
        assert listed == {m2["file_id"]}
        with pytest.raises(FileNotFoundError):
            store.get(m1["file_id"])

    def test_delete_removes_blob_from_disk(self, store):
        meta = store.put(TEXT, "a.py")
        stored, _ = store.open_stored(meta["file_id"])
        store.delete(meta["file_id"])
        assert not stored.exists()

    def test_tag_filter(self, store):
        store.put(TEXT, "a.py", tags=["keep"])
        store.put(TEXT, "b.py", tags=["drop"])
        assert len(store.list(tag="keep")) == 1

    def test_get_meta_missing_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.get_meta("nope")

    def test_ids_unique_same_second(self, store):
        ids = {store.put(TEXT, "same.py")["file_id"] for _ in range(5)}
        assert len(ids) == 5


class TestLegacyCompatibility:
    def test_legacy_entry_without_codec_reads_raw(self, store):
        """Pre-store uploads have no codec field — must read back as raw."""
        fid = "1700000000_legacy.txt"
        blob = b"legacy plain bytes"
        (store.root / f"{fid}.txt").write_bytes(blob)
        store._col().insert_one(
            {
                "file_id": fid,
                "filename": f"{fid}.txt",
                "original_name": "legacy.txt",
                "extension": ".txt",
                "size_bytes": len(blob),
                "chars": 0,
                "pages": 1,
                "uploaded_at": 0.0,
                "tags": [],
            }
        )
        data, meta = store.get(fid)
        assert data == blob
        assert "codec" not in meta

    def test_corrupt_zstd_blob_raises(self, store):
        meta = store.put(TEXT, "handler.py")
        stored, _ = store.open_stored(meta["file_id"])
        stored.write_bytes(b"not a zstd frame" * 10)
        with pytest.raises(Exception):
            store.get(meta["file_id"])
