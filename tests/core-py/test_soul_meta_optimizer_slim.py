"""Sidecar slim: optimizer moments leave the JSON meta, live in gzip, hash-pinned.

The trained ``.soul.meta.json`` sidecar embedded Adam ``state`` as pretty JSON
floats (~162 MB per periodic checkpoint — twice the weights). The sidecar's
real consumers (checkpoint listing, artifact registry) read only tiny fields;
exact resume reads the *embedded* .soul header (untouched here). So the state
moves to a sibling ``<x>.optimizer.json.gz`` and the meta keeps a hash-pinned
``state_ref`` — see ``domain/inference/_internal/soul_meta_slim.py``.

Scope guards these tests enforce:
- embedded .soul header keeps the full inline state (phase 2 is out of scope),
- ``integrity_hash`` still describes the full logical content (identical in
  embedded and sidecar),
- old inline metas keep parsing (no migration required to read),
- a state_ref without a resolvable base_dir degrades to the existing
  fresh-momentum resume semantics instead of raising.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from domain.inference._internal.slo_format import SloProfile, load_soul, save_soul
from domain.training._internal.slonet import SloLinear, SloNet
from domain.training._internal.train_pipeline import _parse_training_state_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATE_SCRIPT = REPO_ROOT / "scripts" / "migrate_soul_meta_optimizer.py"

# ── fixtures ────────────────────────────────────────────────────────────────


def _fat_training_state() -> dict:
    """A realistic periodic-checkpoint training_state (moments inline)."""
    return {
        "step": 7,
        "epoch": 1,
        "completed_epochs": 1,
        "accumulation_step": 0,
        "optimizer": {
            "type": "SloAdamW",
            "hyperparameters": {"lr": 0.001, "betas": [0.9, 0.999]},
            "state": {
                "0": {
                    "exp_avg": [[0.25, -0.5], [0.125, 0.75]],
                    "exp_avg_sq": [[0.0625, 0.015625], [0.00390625, 0.5]],
                },
                "1": {
                    "exp_avg": [[0.125, 0.25]],
                    "exp_avg_sq": [[0.0078125, 0.0625]],
                },
            },
        },
        "scheduler": {"type": "cosine", "initial_lr": 0.001},
    }


def _saved_pair(tmpdir: str) -> tuple[Path, Path]:
    """Save a tiny net with fat metadata; return (sidecar path, gz path)."""
    net = SloNet(layers=[SloLinear(2, 2)], soul_name="SlimTest")
    profile = SloProfile(name="SlimTest")
    profile.metadata = {"training_state": _fat_training_state()}
    out = os.path.join(tmpdir, "tiny.soul")
    save_soul(net, out, soul_profile=profile)
    sidecar = Path(out + ".meta.json")
    gz = Path(out + ".optimizer.json.gz")
    return sidecar, gz


def _ref_of(meta: dict) -> dict:
    opt = meta["metadata"]["training_state"]["optimizer"]
    assert "state_ref" in opt, f"state_ref missing, optimizer keys: {list(opt)}"
    return opt["state_ref"]


# ── writer: save_soul externalizes the state ────────────────────────────────


class TestSaveSoulExternalizes:
    def test_sidecar_keeps_ref_not_moments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar, gz = _saved_pair(tmpdir)
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            opt = meta["metadata"]["training_state"]["optimizer"]

            # The whole point: no moment lists in the JSON sidecar.
            assert not opt.get("state"), "optimizer.state still inline in sidecar"
            ref = _ref_of(meta)
            assert ref["file"] == gz.name
            assert ref["format"] == "gzip-json-v1"
            assert isinstance(ref["sha256"], str) and len(ref["sha256"]) == 64

    def test_gz_holds_the_moments_and_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar, gz = _saved_pair(tmpdir)
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            ref = _ref_of(meta)
            assert gz.exists(), "state_ref points at a missing file"
            assert hashlib.sha256(gz.read_bytes()).hexdigest() == ref["sha256"]
            payload = json.loads(gzip.decompress(gz.read_bytes()))
            assert payload["state"] == _fat_training_state()["optimizer"]["state"]

    def test_sidecar_keeps_hyperparameters_inline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar, _ = _saved_pair(tmpdir)
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            opt = meta["metadata"]["training_state"]["optimizer"]
            assert opt["hyperparameters"] == {"lr": 0.001, "betas": [0.9, 0.999]}

    def test_embedded_header_untouched_by_externalization(self):
        """Phase-2 boundary: the .soul header still carries full inline state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar, gz = _saved_pair(tmpdir)
            soul_path = sidecar.name[: -len(".meta.json")]
            profile, _state = load_soul(os.path.join(tmpdir, soul_path))
            embedded = profile.metadata["training_state"]["optimizer"]
            assert embedded["state"] == _fat_training_state()["optimizer"]["state"]

    def test_integrity_hash_identical_across_embedded_and_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar, _ = _saved_pair(tmpdir)
            soul_path = sidecar.name[: -len(".meta.json")]
            profile, _state = load_soul(os.path.join(tmpdir, soul_path))
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            assert meta["integrity_hash"], "sidecar lost integrity_hash"
            assert meta["integrity_hash"] == profile.integrity_hash

    def test_save_without_optimizer_state_writes_no_ref(self):
        """Weights-only/final saves (no moments) stay a plain sidecar."""
        net = SloNet(layers=[SloLinear(2, 2)], soul_name="NoOpt")
        profile = SloProfile(name="NoOpt")
        profile.metadata = {"training_state": {"step": 3, "epoch": 0}}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "plain.soul")
            save_soul(net, out, soul_profile=profile)
            meta = json.loads(Path(out + ".meta.json").read_text(encoding="utf-8"))
            opt = meta["metadata"]["training_state"].get("optimizer")
            assert opt is None or "state_ref" not in opt
            assert not Path(out + ".optimizer.json.gz").exists()


# ── reader: _parse_training_state_metadata ──────────────────────────────────


class TestParseStateRef:
    def test_reads_ref_with_base_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar, _ = _saved_pair(tmpdir)
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            parsed = _parse_training_state_metadata(
                {"training_state": meta["metadata"]["training_state"]},
                base_dir=tmpdir,
            )
            assert parsed["step"] == 7
            state = parsed["optimizer"]["state"]
            np.testing.assert_allclose(state["0"]["exp_avg"], [[0.25, -0.5], [0.125, 0.75]])
            np.testing.assert_allclose(state["1"]["exp_avg_sq"], [[0.0078125, 0.0625]])
            assert state["0"]["exp_avg"].dtype == np.float64

    def test_without_base_dir_degrades_to_fresh_momentum(self):
        """No path context → hyperparameters survive, moments are dropped
        (identical semantics to include_optimizer_state=False)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar, _ = _saved_pair(tmpdir)
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            parsed = _parse_training_state_metadata(
                {"training_state": meta["metadata"]["training_state"]}
            )
            assert parsed["optimizer"]["hyperparameters"]["lr"] == 0.001
            assert not parsed["optimizer"].get("state")

    def test_inline_state_still_parses_without_ref(self):
        """Back-compat: old files (and the embedded header) keep working."""
        parsed = _parse_training_state_metadata({"training_state": _fat_training_state()})
        np.testing.assert_allclose(
            parsed["optimizer"]["state"]["0"]["exp_avg"], [[0.25, -0.5], [0.125, 0.75]]
        )
        assert parsed["optimizer"]["hyperparameters"]["lr"] == 0.001

    def test_corrupt_ref_without_base_dir_does_not_raise(self):
        state = _fat_training_state()
        state["optimizer"]["state_ref"] = {
            "file": "missing.optimizer.json.gz",
            "sha256": "0" * 64,
            "format": "gzip-json-v1",
        }
        del state["optimizer"]["state"]
        parsed = _parse_training_state_metadata({"training_state": state})
        assert parsed["optimizer"]["hyperparameters"]["lr"] == 0.001
        assert not parsed["optimizer"].get("state")


# ── export_to_sou round-trip stays slim ─────────────────────────────────────


class TestExportRoundtripSlim:
    def test_export_to_sou_externalizes_training_state(self):
        from domain.training._internal.slonet import export_to_sou

        net = SloNet(layers=[SloLinear(2, 2)], soul_name="RoundTrip")
        net.metadata = {"training_state": _fat_training_state()}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "rt.soul")
            export_to_sou(net, out, include_weights=True)
            sidecar = Path(out + ".meta.json")
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            opt = meta["metadata"]["training_state"]["optimizer"]
            assert not opt.get("state")
            gz = Path(out + ".optimizer.json.gz")
            ref = _ref_of(meta)
            assert ref["file"] == gz.name and gz.exists()


# ── migration tool ──────────────────────────────────────────────────────────


def _write_fat_sidecar(tmpdir: str) -> Path:
    models = Path(tmpdir)
    fat = {
        "name": "legacy",
        "integrity_hash": "abc123",
        "metadata": {"training_state": _fat_training_state(), "avg_loss": 1.5},
    }
    sidecar = models / "legacy.soul.meta.json"
    sidecar.write_text(json.dumps(fat, indent=2, default=str), encoding="utf-8")
    return sidecar


def _run_tool(*args: str) -> subprocess.CompletedProcess:
    assert MIGRATE_SCRIPT.exists(), "migration tool not built yet"
    return subprocess.run(
        [sys.executable, str(MIGRATE_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("migrate_soul_meta_optimizer", MIGRATE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMigrationTool:
    def test_dry_run_leaves_everything_alone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = _write_fat_sidecar(tmpdir)
            before = sidecar.read_bytes()
            proc = _run_tool("--root", tmpdir)
            assert proc.returncode == 0, proc.stderr
            assert sidecar.read_bytes() == before
            assert not (Path(tmpdir) / "legacy.soul.optimizer.json.gz").exists()
            assert not (Path(tmpdir) / ".meta-backups").exists()

    def test_apply_migrates_and_backs_up(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = _write_fat_sidecar(tmpdir)
            original = sidecar.read_bytes()
            proc = _run_tool("--root", tmpdir, "--apply")
            assert proc.returncode == 0, proc.stderr

            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            opt = meta["metadata"]["training_state"]["optimizer"]
            assert not opt.get("state")
            ref = opt["state_ref"]
            gz = Path(tmpdir) / ref["file"]
            assert gz.exists()
            assert hashlib.sha256(gz.read_bytes()).hexdigest() == ref["sha256"]
            payload = json.loads(gzip.decompress(gz.read_bytes()))
            assert payload["state"] == _fat_training_state()["optimizer"]["state"]

            # untouched facts survive the rewrite
            assert meta["integrity_hash"] == "abc123"
            assert meta["metadata"]["avg_loss"] == 1.5

            # original preserved byte-for-byte under .meta-backups
            backups = list((Path(tmpdir) / ".meta-backups").rglob("*.meta.json"))
            assert len(backups) == 1
            assert backups[0].read_bytes() == original

    def test_apply_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_fat_sidecar(tmpdir)
            assert _run_tool("--root", tmpdir, "--apply").returncode == 0
            second = _run_tool("--root", tmpdir, "--apply")
            assert second.returncode == 0, second.stderr
            assert "migrated: 0" in (second.stdout + second.stderr)

    def test_skips_metas_without_moments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lean = {"name": "lean", "metadata": {"training_state": {"step": 1}}}
            p = Path(tmpdir) / "lean.soul.meta.json"
            p.write_text(json.dumps(lean), encoding="utf-8")
            before = p.read_bytes()
            proc = _run_tool("--root", tmpdir, "--apply")
            assert proc.returncode == 0, proc.stderr
            assert p.read_bytes() == before


# ── delete keeps no orphans ─────────────────────────────────────────────────


class TestDeleteRemovesStateFile:
    @pytest.mark.asyncio
    async def test_delete_checkpoint_unlinks_state_gz(self, tmp_path, monkeypatch):
        from domain.training._internal import checkpoints as cp_mod

        soul = tmp_path / "gone.soul"
        soul.write_bytes(b"\x00" * 4096)
        meta = tmp_path / "gone.soul.meta.json"
        meta.write_text("{}", encoding="utf-8")
        gz = tmp_path / "gone.soul.optimizer.json.gz"
        gz.write_bytes(b"placeholder")

        monkeypatch.setattr(cp_mod, "CHECKPOINTS_DIR", tmp_path)
        monkeypatch.setattr(cp_mod, "TURBO_DIR", tmp_path)

        deleted = await cp_mod.delete_checkpoint("gone.soul")
        assert "gone.soul" in deleted
        assert not soul.exists()
        assert not meta.exists(), "pre-existing meta cleanup missing"
        assert not gz.exists(), "state file orphaned by delete"
