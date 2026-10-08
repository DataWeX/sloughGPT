"""Shared-core contract: an action lands on the row the user pointed at.

Two roots may hold one filename — ``models/`` and ``models/auto-training/``
are separate roots and nothing stops a job writing the same name into both.
``name`` alone therefore cannot identify a file, and every action in the
chain (load, info, download, delete) used to take a name:

* ``delete_checkpoint`` walked every root and unlinked *every* match, so
  clicking one row deleted the same-named file in the other root as well —
  while the listing hid that twin from view (``seen`` was keyed by name), so
  a file was deleted that the user was never shown.
* load / info / download returned the root-major *first* match, so the file
  a dialog described could be a different file than the row it was opened
  from.

The fix is the row's own ``path``: always present, unique by construction,
refused by the server when it falls outside a search root. This test pins
the whole trip over HTTP — the listing carries it, each action honours it,
and a path that fails confinement is rejected instead of quietly falling
back to a name sweep that would hit the wrong file.

It needs real files through the real route (not mocks) because every
failure mode here is silent: the API still answers 200, it just acts on the
other file, and no unit test below this seam can see which one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _redirect_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    import domain.training._internal.checkpoints as ckpt_mod

    roots: dict[str, Path] = {}
    for name in ("CHECKPOINTS_DIR", "TURBO_DIR", "LORA_DIR", "TRAINED_DIR"):
        root = tmp_path / name.lower()
        root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(ckpt_mod, name, root)
        roots[name] = root.resolve()
    return roots


def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from infrastructure.exception_handlers import register_app_error_handler
    from training.router import router

    app = FastAPI()
    register_app_error_handler(app)
    app.include_router(router)
    return TestClient(app)


def _write(fp: Path, payload: bytes, integrity_hash: str | None = None) -> None:
    fp.write_bytes(payload)
    if integrity_hash:
        Path(str(fp) + ".meta.json").write_text(
            json.dumps({"integrity_hash": integrity_hash}), encoding="utf-8"
        )


def test_a_same_named_row_is_the_file_it_points_at(tmp_path, monkeypatch) -> None:
    roots = _redirect_roots(tmp_path, monkeypatch)
    auto, trained = roots["CHECKPOINTS_DIR"], roots["TRAINED_DIR"]

    payload_a, payload_b = b"A" * 5000, b"B" * 5000
    hash_a, hash_b = "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"
    _write(auto / "twin_trained.soul", payload_a, hash_a)
    _write(trained / "twin_trained.soul", payload_b, hash_b)

    client = _client()

    # ── 1. The listing shows BOTH twins, each carrying its own address ──────
    resp = client.get("/training/checkpoints")
    assert resp.status_code == 200, resp.text
    rows = [r for r in resp.json()["data"] if r["name"] == "twin_trained.soul"]
    assert len(rows) == 2, f"name-keyed dedupe is still hiding a twin: {rows}"
    paths = {r["path"] for r in rows}
    assert len(paths) == 2, f"rows are addressed by the same path: {rows}"
    assert all(Path(p).name == "twin_trained.soul" for p in paths), rows

    def _expected_hash(path: str) -> str:
        parent = Path(path).resolve().parent
        if parent == auto:
            return hash_a
        assert parent == trained, f"path points at neither root: {path}"
        return hash_b

    # ── 2. info describes the file its OWN row points at, not the first ─────
    for path in paths:
        detail = client.get("/training/checkpoints/twin_trained.soul/info", params={"path": path})
        assert detail.status_code == 200, f"{path}: {detail.text}"
        info = detail.json()["data"]
        assert info.get("integrity_hash") == _expected_hash(path), (
            f"info answered about the other twin for {path}: {info}"
        )

    # ── 3. download hands back the bytes of that row ────────────────────────
    trained_path = str(trained / "twin_trained.soul")
    dl = client.get(
        "/training/checkpoints/twin_trained.soul/download", params={"path": trained_path}
    )
    assert dl.status_code == 200, dl.text
    assert dl.content == payload_b, "download served the other twin's bytes"

    # ── 4. delete removes the pointed-at file and NOTHING else ──────────────
    auto_path = str(auto / "twin_trained.soul")
    resp = client.delete("/training/checkpoints/twin_trained.soul", params={"path": auto_path})
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["deleted"] == ["twin_trained.soul"], resp.text
    assert not (auto / "twin_trained.soul").exists(), "pointed-at file survived"
    survivor = trained / "twin_trained.soul"
    assert survivor.read_bytes() == payload_b, "DELETE took the other root's file too"
    assert Path(str(survivor) + ".meta.json").exists(), "survivor's sidecar was swept"

    # ── 5. a path outside every root is REFUSED, never a name-sweep fallback ─
    # The sibling-name shape matters: `trained_dir_evil` shares the root's
    # string prefix, so a startswith-style check counts it as inside.
    stranger_root = tmp_path / f"{trained.name}_evil"
    stranger_root.mkdir(parents=True, exist_ok=True)
    stranger = stranger_root / "twin_trained.soul"
    _write(stranger, payload_a, hash_a)

    resp = client.delete("/training/checkpoints/twin_trained.soul", params={"path": str(stranger)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["deleted"] == [], "a path outside the search roots was acted on"
    assert stranger.exists(), "confined path fell through and was deleted anyway"
    assert survivor.exists(), "bad path fell back to the legacy name sweep"

    # ── 6. no path = the legacy behaviour, still working for name callers ───
    _write(auto / "solo.soul", b"C" * 5000)
    resp = client.delete("/training/checkpoints/solo.soul")
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["deleted"] == ["solo.soul"], resp.text
    assert not (auto / "solo.soul").exists(), "name-only delete stopped working"
