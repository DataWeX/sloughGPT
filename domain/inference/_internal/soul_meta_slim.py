"""Sidecar slim: Adam moments leave the JSON meta, live in a gzip sibling.

A periodic checkpoint's ``<x>.soul.meta.json`` embedded the optimizer's
``state`` (exp_avg / exp_avg_sq) as pretty-printed JSON floats — measured at
~162 MB per file, twice the weights, and ~2.2 s per ``json.loads``. The
sidecar's real consumers (checkpoint listing, artifact registry, soul index)
read only tiny profile fields, and exact resume reads the *embedded* .soul
header instead — so the sidecar copy is redundant weight.

Format: ``gzip-json-v1`` — the state dict, compact JSON, gzip, written next to
the sidecar as ``<x>.optimizer.json.gz`` (``<x>.meta.json`` minus its suffix),
sha256-pinned by a ``state_ref`` left inline:

    "optimizer": {"hyperparameters": {...},
                  "state_ref": {"file": ..., "sha256": ..., "format": ...}}

WAL order (mirrors the save paths): the gzip lands durably first; the sidecar
that references it is rename-atomic after. A crash between leaves a harmless
orphan gzip, never a ref pointing at nothing.

Why gzip-JSON and not .npz: ``*.npz`` in a model dir is a live token —
``_candidates_newest_first`` treats it as a resume candidate and
``artifact_registry`` registers it as a checkpoint artifact. ``*.gz`` matches
no glob in the tree, and JSON round-trips the float64 text losslessly.

Stdlib-only: loaded standalone by ``scripts/migrate_soul_meta_optimizer.py``
via importlib (no package ``__init__`` chain, no domain deps).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

FORMAT = "gzip-json-v1"
META_SUFFIX = ".meta.json"
STATE_SUFFIX = ".optimizer.json.gz"


def state_file_for(meta_path: str | os.PathLike[str]) -> Path:
    """``<x>.meta.json`` → ``<x>.optimizer.json.gz`` (sibling, same stem)."""
    p = Path(meta_path)
    name = p.name
    if name.endswith(META_SUFFIX):
        name = name[: -len(META_SUFFIX)]
    return p.with_name(name + STATE_SUFFIX)


def _optimizer_of(doc: Any) -> dict | None:
    """``doc["metadata"]["training_state"]["optimizer"]`` or None (any shape)."""
    if not isinstance(doc, dict):
        return None
    md = doc.get("metadata")
    if not isinstance(md, dict):
        return None
    ts = md.get("training_state")
    if not isinstance(ts, dict):
        return None
    opt = ts.get("optimizer")
    return opt if isinstance(opt, dict) else None


def externalize_optimizer_state(
    doc: dict, meta_path: str | os.PathLike[str], *, write: bool = True
) -> tuple[dict, bool]:
    """Return ``(sidecar_doc, moved?)`` with ``optimizer.state`` moved out.

    No-op returns the input doc untouched when there is nothing to move
    (already slim, weights-only save, or no optimizer at all). When ``write``
    is true the gzip sibling is written and fsynced before returning, so the
    caller can never publish a ``state_ref`` to a file that does not exist.
    """
    opt = _optimizer_of(doc)
    if opt is None:
        return doc, False
    state = opt.get("state")
    if not isinstance(state, dict) or not state:
        return doc, False

    payload = json.dumps(
        {"state": state}, separators=(",", ":"), default=str, allow_nan=False
    ).encode("utf-8")
    blob = gzip.compress(payload, compresslevel=6)
    sha = hashlib.sha256(blob).hexdigest()

    out = state_file_for(meta_path)
    if write:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)

    # Deep copy through the JSON domain (the doc is written as JSON anyway).
    slim = json.loads(json.dumps(doc, default=str))
    slim_opt = _optimizer_of(slim)
    slim_opt.pop("state", None)
    slim_opt["state_ref"] = {
        "file": out.name,
        "sha256": sha,
        "format": FORMAT,
    }
    return slim, True


def load_state_file(ref: Any, base_dir: str | os.PathLike[str]) -> dict | None:
    """Resolve a ``state_ref`` to the raw state dict (JSON lists), hash-pinned.

    Returns ``None`` for anything unresolvable (missing file, wrong format,
    hash mismatch, malformed payload) — callers degrade to fresh-momentum
    resume, which is the pre-existing semantics for a stripped state.
    """
    if not isinstance(ref, dict) or ref.get("format") != FORMAT:
        return None
    file = ref.get("file")
    if not isinstance(file, str):
        return None
    try:
        blob = (Path(base_dir) / file).read_bytes()
        if hashlib.sha256(blob).hexdigest() != ref.get("sha256"):
            return None
        doc = json.loads(gzip.decompress(blob))
    except Exception:  # noqa: BLE001 — degradation, never a hard failure
        return None
    state = doc.get("state") if isinstance(doc, dict) else None
    return state if isinstance(state, dict) and state else None
