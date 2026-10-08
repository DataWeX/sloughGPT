"""Builds listing: a row addresses the file it was actually globbed from.

Two roots may hold one filename. The builds listing keyed its ``seen`` set by
name — so the second same-named build was silently hidden — and resolved rows
through ``load_soul(f.name)``, a name probe that walks the roots in order and
can describe a *different* root's twin instead of the file the glob found.
Both are the same defect the checkpoint listing fixed by keying on address:
the row must BE the file it points at.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import training.builds as builds_mod
from fastapi import FastAPI
from fastapi.testclient import TestClient
from training.builds import router as builds_router

import domain.training._internal.checkpoints as ckpt_mod

app = FastAPI()
app.include_router(builds_router)
client = TestClient(app)


def _soul(path: Path, meta: dict) -> None:
    """A >=4096-byte .soul (load_soul skips smaller files) plus its sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * 5000)
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta))


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Send every root the listing reads to tmp_path, in both modules.

    ``builds`` derives its glob directories from ``find_repo_root``; ``load_soul``
    resolves and confines through ``ckpt_roots()``. Both must point at the same
    tmp tree or a globbed file would fail the direct-child check and vanish.
    """
    roots = {
        "CHECKPOINTS_DIR": tmp_path / "models" / "auto-training",
        "TURBO_DIR": tmp_path / "models" / "turbo-trained",
        "LORA_DIR": tmp_path / "data" / "user_adapters",
        "TRAINED_DIR": tmp_path / "models",
    }
    for name, p in roots.items():
        p.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(ckpt_mod, name, p)
    monkeypatch.setattr(builds_mod, "find_repo_root", lambda _p: tmp_path)
    # Completed HF job rows are unrelated to file identity; keep them out so a
    # job another test registered cannot leak into these assertions.
    monkeypatch.setattr(builds_mod, "training_jobs", {})
    return roots


def _builds() -> list[dict]:
    resp = client.get("/training/builds")
    assert resp.status_code == 200
    return resp.json()["builds"]


class TestRowAddressedBuilds:
    def test_same_name_in_two_roots_lists_both(self, _isolated):
        roots = _isolated
        _soul(
            roots["CHECKPOINTS_DIR"] / "twin.soul",
            {"soul_name": "alpha-soul", "integrity_hash": "hash-alpha"},
        )
        _soul(
            roots["LORA_DIR"] / "twin.soul",
            {"soul_name": "beta-soul", "integrity_hash": "hash-beta"},
        )

        rows = [b for b in _builds() if b["name"] == "twin.soul"]

        assert len(rows) == 2, "a same-named build in another root must not be hidden"
        by_parent = {Path(r["path"]).resolve().parent: r for r in rows}
        assert set(by_parent) == {
            roots["CHECKPOINTS_DIR"].resolve(),
            roots["LORA_DIR"].resolve(),
        }
        # Identity follows the address: each row describes its own file.
        assert by_parent[roots["CHECKPOINTS_DIR"].resolve()]["integrity_hash"] == "hash-alpha"
        assert by_parent[roots["LORA_DIR"].resolve()]["integrity_hash"] == "hash-beta"

    def test_final_save_row_addresses_the_models_file_not_the_auto_twin(self, _isolated):
        roots = _isolated
        _soul(
            roots["CHECKPOINTS_DIR"] / "pre_trained.soul",
            {"soul_name": "auto-soul", "integrity_hash": "hash-auto"},
        )
        _soul(
            roots["TRAINED_DIR"] / "pre_trained.soul",
            {"soul_name": "final-soul", "integrity_hash": "hash-final"},
        )

        rows = [b for b in _builds() if b["name"] == "pre_trained.soul"]

        assert len(rows) == 2, "models/<stem>_trained.soul must list beside its auto-training twin"
        by_parent = {Path(r["path"]).resolve().parent: r for r in rows}
        trained = by_parent[roots["TRAINED_DIR"].resolve()]
        assert trained["build_type"] == "trained"
        assert trained["integrity_hash"] == "hash-final"
        auto = by_parent[roots["CHECKPOINTS_DIR"].resolve()]
        assert auto["build_type"] == "auto-train"
        assert auto["integrity_hash"] == "hash-auto"
