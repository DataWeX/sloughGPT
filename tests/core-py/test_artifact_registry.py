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


class TestProvenance:
    def test_snapshot_records_commit_and_platform(self):
        snap = reg.provenance_snapshot()
        # commit may be None outside a checkout; platform/python always present.
        assert "commit" in snap and "dirty" in snap
        assert snap["python"]
        assert snap["platform"]
        assert "config_fingerprint" in snap

    def test_config_fingerprint_is_stable_and_env_sensitive(self, monkeypatch):
        monkeypatch.setenv("SLO_QUANT", "int8")
        first = reg.config_fingerprint()
        assert first == reg.config_fingerprint()
        monkeypatch.setenv("SLO_QUANT", "int4")
        assert reg.config_fingerprint() != first

    def test_write_then_read_roundtrip(self, tmp_path):
        artifact = tmp_path / "child.npz"
        artifact.write_bytes(b"\x00")
        record = reg.write_provenance(
            artifact, "adapter", "child.npz", parents=["adapter:a.npz"], extra={"k": 1}
        )
        assert record.parents == ["adapter:a.npz"]
        read = reg.read_provenance(artifact)
        assert read is not None
        assert read.artifact_id == "adapter:child.npz"
        assert read.parents == ["adapter:a.npz"]
        assert read.extra == {"k": 1}

    def test_read_missing_sidecar_is_none(self, tmp_path):
        artifact = tmp_path / "plain.npz"
        artifact.write_bytes(b"\x00")
        assert reg.read_provenance(artifact) is None

    def test_read_corrupt_sidecar_is_none_not_raise(self, tmp_path):
        artifact = tmp_path / "bad.npz"
        artifact.write_bytes(b"\x00")
        reg.provenance_path(artifact).write_text("{not json")
        assert reg.read_provenance(artifact) is None

    def test_write_provenance_never_raises_on_unwritable(self, tmp_path):
        # Point the sidecar at a directory so the write fails -> must not raise.
        artifact = tmp_path / "x.npz"
        artifact.write_bytes(b"\x00")
        blocked = reg.provenance_path(artifact)
        blocked.mkdir()  # a dir where a file is expected
        record = reg.write_provenance(artifact, "adapter", "x.npz")
        assert record.artifact_id == "adapter:x.npz"  # record still returned


class TestLineage:
    def test_walks_parents_to_roots(self, tmp_path):
        roots = _roots(tmp_path)
        adapters = roots["adapter"][0]
        # Two parent adapters (lineage roots) + one child merging them.
        for name in ("a.npz", "b.npz"):
            p = adapters / name
            p.write_bytes(b"\x00")
            reg.write_provenance(p, "adapter", name, parents=[])
        child = adapters / "child.npz"
        child.write_bytes(b"\x00")
        reg.write_provenance(
            child, "adapter", "child.npz", parents=["adapter:a.npz", "adapter:b.npz"]
        )

        with _scan(tmp_path):
            result = reg.lineage("adapter", "child.npz")
        assert result["root"] == "adapter:child.npz"
        assert set(result["roots"]) == {"adapter:a.npz", "adapter:b.npz"}
        assert set(result["nodes"]) == {
            "adapter:child.npz",
            "adapter:a.npz",
            "adapter:b.npz",
        }

    def test_no_sidecar_is_a_root(self, tmp_path):
        roots = _roots(tmp_path)
        (roots["adapter"][0] / "stamped.npz").write_bytes(b"\x00")
        with _scan(tmp_path):
            result = reg.lineage("adapter", "stamped.npz")
        assert result["roots"] == ["adapter:stamped.npz"]
        assert result["nodes"]["adapter:stamped.npz"]["parents"] == []

    def test_cycle_does_not_hang(self, tmp_path):
        roots = _roots(tmp_path)
        adapters = roots["adapter"][0]
        for name, parent in (("p.npz", "adapter:q.npz"), ("q.npz", "adapter:p.npz")):
            p = adapters / name
            p.write_bytes(b"\x00")
            reg.write_provenance(p, "adapter", name, parents=[parent])
        with _scan(tmp_path):
            result = reg.lineage("adapter", "p.npz")
        # Both visited exactly once; the cycle is cut by the seen-set.
        assert set(result["nodes"]) == {"adapter:p.npz", "adapter:q.npz"}

    def test_root_dir_resolves_outside_scan_roots(self, tmp_path):
        # A store NOT under the registry's scan roots (custom store_path).
        store = tmp_path / "custom_store"
        store.mkdir()
        child = store / "child.npz"
        child.write_bytes(b"\x00")
        reg.write_provenance(child, "adapter", "child.npz", parents=["adapter:parent.npz"])
        parent = store / "parent.npz"
        parent.write_bytes(b"\x00")
        reg.write_provenance(parent, "adapter", "parent.npz", parents=[])

        with _scan(tmp_path):
            result = reg.lineage("adapter", "child.npz", path=child, root_dir=store)
        assert result["roots"] == ["adapter:parent.npz"]
        assert "adapter:parent.npz" in result["nodes"]
