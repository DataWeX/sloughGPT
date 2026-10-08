"""Tests for domain.infrastructure._internal.artifact_registry (Stage 1).

The registry is a read-only filesystem index. Tests redirect its roots
into tmp dirs by patching ``scan_roots`` — no real repo layout needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from domain.infrastructure._internal import artifact_registry as reg


def _roots(tmp: Path) -> dict[str, list[Path]]:
    models = tmp / "models"
    data = tmp / "data"
    ckpt = tmp / "ckpt"
    adapters = tmp / "adapters"
    cache = tmp / "cache"
    for d in (models, data, ckpt, adapters, cache):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "model": [models],
        "dataset": [data],
        "checkpoint": [ckpt],
        "adapter": [adapters],
        "weight": [models],
        "download": [cache],
        "file": [data, models],
    }


def _scan(tmp: Path):
    return patch.object(reg, "scan_roots", return_value=_roots(tmp))


class TestListArtifacts:
    def test_dataset_requires_corpus(self, tmp_path):
        ds = tmp_path / "data" / "good"
        ds.mkdir(parents=True)
        (ds / "corpus.jsonl").write_text('{"text": "hello"}\n')
        empty = tmp_path / "data" / "empty"
        empty.mkdir(parents=True)
        with _scan(tmp_path):
            ids = [a["id"] for a in reg.list_artifacts("dataset")]
        assert "good" in ids
        assert "empty" not in ids

    def test_checkpoint_adapter_model_weight(self, tmp_path):
        roots = _roots(tmp_path)
        (roots["checkpoint"][0] / "a.soul").write_bytes(b"\x00" * 10)
        (roots["adapter"][0] / "b.npz").write_bytes(b"\x00" * 10)
        (roots["model"][0] / "c.soul").write_bytes(b"\x00" * 10)
        (roots["model"][0] / "d.safetensors").write_bytes(b"\x00" * 10)
        with _scan(tmp_path):
            by_id = {a["id"]: a for a in reg.list_artifacts()}
        assert by_id["a.soul"]["kind"] == "checkpoint"
        assert by_id["b.npz"]["kind"] == "adapter"
        assert by_id["c.soul"]["kind"] == "model"
        assert by_id["d.safetensors"]["kind"] == "weight"
        for key in ("a.soul", "b.npz", "c.soul", "d.safetensors"):
            assert len(by_id[key]["meta"]["fingerprint"]) == 32
        assert by_id["c.soul"]["meta"]["format"] == "soul"

    def test_download_manifest(self, tmp_path):
        roots = _roots(tmp_path)
        entry = roots["download"][0] / "mymodel"
        entry.mkdir(parents=True)
        import hashlib
        import json

        digest = hashlib.sha256(b"\x00" * 3).hexdigest()
        (entry / ".manifest.json").write_text(
            json.dumps({"files": [{"path": "w.bin", "size": 3, "sha256": digest}]})
        )
        (entry / "w.bin").write_bytes(b"\x00" * 3)
        with _scan(tmp_path):
            found = [a for a in reg.list_artifacts("download") if a["id"] == "mymodel"]
        assert len(found) == 1
        assert found[0]["meta"]["files"] == [{"path": "w.bin", "size": 3, "sha256": digest}]

    def test_query_filter(self, tmp_path):
        ds = tmp_path / "data" / "tinyshakespeare"
        ds.mkdir(parents=True)
        (ds / "input.txt").write_text("hello world, this is a test corpus")
        with _scan(tmp_path):
            assert len(reg.list_artifacts("dataset", query="tiny")) == 1
            assert reg.list_artifacts("dataset", query="zzz") == []

    def test_unknown_kind_ignored(self, tmp_path):
        with _scan(tmp_path):
            assert reg.list_artifacts("nope") == []


class TestResolve:
    def test_resolve_dataset(self, tmp_path):
        ds = tmp_path / "data" / "myds"
        ds.mkdir(parents=True)
        corpus = ds / "corpus.jsonl"
        corpus.write_text('{"text": "x"}\n')
        with _scan(tmp_path):
            assert reg.resolve("dataset", "myds") == str(corpus)
            assert reg.resolve("dataset", "missing") is None

    def test_resolve_rejects_traversal(self, tmp_path):
        (tmp_path / "data").mkdir(parents=True)
        with _scan(tmp_path):
            assert reg.resolve("file", "../../etc/passwd") is None
            assert reg.resolve("file", "/etc/passwd") is None

    def test_resolve_unknown_kind(self, tmp_path):
        with _scan(tmp_path):
            assert reg.resolve("nope", "x") is None


class TestRegister:
    def test_register_file(self, tmp_path):
        fp = tmp_path / "w.safetensors"
        fp.write_bytes(b"\x00" * 100)
        rec = reg.register("weight", fp)
        assert rec["id"] == "w.safetensors"
        assert len(rec["meta"]["fingerprint"]) == 32
        assert rec["meta"]["format"] == "safetensors"

    def test_register_dataset_dir(self, tmp_path):
        ds = tmp_path / "myds"
        ds.mkdir()
        assert reg.register("dataset", ds)["id"] == "myds"

    def test_register_missing_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            reg.register("model", tmp_path / "nope.soul")

    def test_register_bad_kind(self, tmp_path):
        import pytest

        with pytest.raises(ValueError):
            reg.register("file", tmp_path)


class TestStats:
    def test_counts(self, tmp_path):
        roots = _roots(tmp_path)
        (roots["checkpoint"][0] / "a.soul").write_bytes(b"\x00" * 10)
        with _scan(tmp_path):
            stats = reg.stats()
        assert stats["by_kind"]["checkpoint"]["count"] == 1
        assert stats["total"] >= 1


class TestVerify:
    def test_ok_with_sidecar(self, tmp_path):
        import hashlib

        roots = _roots(tmp_path)
        fp = roots["checkpoint"][0] / "a.soul"
        fp.write_bytes(b"\x00" * 10)
        fp.with_suffix(".soul.sha256").write_text(hashlib.sha256(b"\x00" * 10).hexdigest())
        with _scan(tmp_path):
            result = reg.verify("checkpoint")
        assert result["summary"] == {"ok": 1, "mismatch": 0, "missing": 0, "unverified": 0}

    def test_mismatch_with_sidecar(self, tmp_path):
        roots = _roots(tmp_path)
        fp = roots["checkpoint"][0] / "a.soul"
        fp.write_bytes(b"\x00" * 10)
        fp.with_suffix(".soul.sha256").write_text("0" * 64)
        with _scan(tmp_path):
            result = reg.verify("checkpoint")
        assert result["summary"]["mismatch"] == 1
        assert result["results"][0]["verify"]["status"] == "mismatch"

    def test_unverified_without_baseline(self, tmp_path):
        roots = _roots(tmp_path)
        (roots["checkpoint"][0] / "a.soul").write_bytes(b"\x00" * 10)
        with _scan(tmp_path):
            result = reg.verify("checkpoint")
        assert result["summary"]["unverified"] == 1
        assert len(result["results"][0]["verify"]["sha256"]) == 64

    def test_download_hash_check(self, tmp_path):
        import hashlib
        import json

        roots = _roots(tmp_path)
        entry = roots["download"][0] / "mymodel"
        entry.mkdir(parents=True)
        digest = hashlib.sha256(b"\x00" * 3).hexdigest()
        (entry / ".manifest.json").write_text(
            json.dumps({"files": [{"path": "w.bin", "size": 3, "sha256": digest}]})
        )
        (entry / "w.bin").write_bytes(b"\x00" * 3)
        with _scan(tmp_path):
            result = reg.verify("download")
        assert result["summary"]["ok"] == 1

    def test_download_missing_file(self, tmp_path):
        import json

        roots = _roots(tmp_path)
        entry = roots["download"][0] / "mymodel"
        entry.mkdir(parents=True)
        (entry / ".manifest.json").write_text(json.dumps({"files": [{"path": "gone.bin"}]}))
        with _scan(tmp_path):
            result = reg.verify("download")
        assert result["summary"]["mismatch"] == 1
