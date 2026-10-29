"""Unified artifact registry — Stage 2 (read-only index + write-path hooks).

Single place to list and resolve every artifact kind the server manages:
``model``, ``dataset``, ``checkpoint``, ``adapter``, ``weight``,
``download``, ``file``.

Stage 1 is intentionally a read-only filesystem index over the existing
finders — no database, no behavior change for existing callers:

- download manifests: ``external_download.py`` (``.manifest.json`` sidecars)
- entry classification: ``domain.training._internal.cache_tags``
- checkpoint dirs: ``domain.training._internal.state``
- dataset corpus files: ``corpus.jsonl`` / ``input.txt`` / ``train.txt`` /
  ``text.txt`` / ``*.txt`` / ``*.jsonl`` priority (keep in sync with
  ``cache_tags._CORPUS_CANDIDATES`` and ``find_corpus_file``)

This module must not import ``domain.training`` (training depends on
infrastructure, not the other way round), so the small filename/id
conventions are repeated here with pointers instead of imports.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct
import zipfile
from pathlib import Path
from typing import Any

from domain.shared import repair_iso  # stdlib-only, no third-party imports

logger = logging.getLogger("slo.artifact_registry")

# Bytes hashed for the content fingerprint (magic + head chunk only, so
# multi-GB weight files stay cheap to identify).
_FINGERPRINT_HEAD_BYTES = 4096

# Keep in sync with domain.training._internal.cache_tags._CORPUS_CANDIDATES.
_CORPUS_CANDIDATES = ("corpus.jsonl", "input.txt", "train.txt", "text.txt")

# Keep in sync with domain.training._internal.cache_tags._VALID_DATASET_ID.
_VALID_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")

MODEL_SUFFIXES = (".soul", ".slnc")
WEIGHT_SUFFIXES = (".safetensors",)
ADAPTER_SUFFIXES = (".npz",)

ARTIFACT_KINDS = (
    "model",
    "dataset",
    "checkpoint",
    "adapter",
    "weight",
    "download",
    "file",
)


def repo_root() -> Path:
    """Repo root, resolved from this file (correct for any CWD)."""
    from domain.shared import find_repo_root

    return find_repo_root(Path(__file__).resolve())


def cache_root() -> Path:
    """Just-cache root for external downloads (may not exist)."""
    return Path(os.environ.get("SLO_CACHE_DIR", Path.home() / ".cache" / "sloughgpt")) / "external"


def scan_roots() -> dict[str, list[Path]]:
    """Filesystem roots scanned per kind (existing dirs only)."""
    root = repo_root()
    # Dataset order matters: just-cache wins over legacy dirs (same
    # convention as DatasetsController and the old resolve_dataset_path).
    candidates: dict[str, list[Path]] = {
        "model": [root / "models"],
        "dataset": [
            cache_root(),
            root / "datasets",
            root / "data",
            root / "data" / "datasets",
        ],
        "checkpoint": [],
        "adapter": [root / "data" / "user_adapters"],
        "weight": [root / "models"],
        "download": [cache_root()],
        "file": [root / "data", root / "models", cache_root()],
    }
    try:
        from domain.training._internal.state import CHECKPOINTS_DIR as _CKPT
        from domain.training._internal.state import TURBO_DIR as _TURBO

        candidates["checkpoint"] = [_CKPT, _TURBO]
    except Exception as exc:
        logger.debug("Checkpoint dirs unavailable: %s", exc)
    return {kind: [p for p in paths if p.exists()] for kind, paths in candidates.items()}


def _find_corpus(entry: Path) -> Path | None:
    for name in _CORPUS_CANDIDATES:
        candidate = entry / name
        if candidate.is_file():
            return candidate
    for pattern in ("*.txt", "*.jsonl", "*.csv"):
        hits = sorted(entry.glob(pattern))
        if hits:
            return hits[0]
    return None


def find_corpus_file(entry_dir: Path) -> Path | None:
    """Public corpus picker: shared priority for all dataset resolvers.

    Lets callers keep their own roots (and test seams) while sharing one
    priority implementation with the registry scan.
    """
    return _find_corpus(entry_dir)


def _sniff(path: Path) -> dict[str, Any]:
    """Cheap content classification: magic bytes + size, no full read."""
    info: dict[str, Any] = {"magic": "", "size_bytes": 0}
    try:
        if not path.is_file():
            return info
        info["size_bytes"] = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(16)
        info["magic"] = head.hex()
    except OSError as exc:
        logger.debug("Sniff failed for %s: %s", path, exc)
    return info


def _fingerprint(path: Path, magic: str, size: int) -> str:
    """Stable content identity from magic + size + head chunk.

    Identical files at different paths share a fingerprint, so duplicates
    (same checkpoint copied into two dirs) are detectable without hashing
    gigabytes.
    """
    h = hashlib.sha256()
    h.update(magic.encode())
    h.update(str(size).encode())
    try:
        with open(path, "rb") as f:
            h.update(f.read(_FINGERPRINT_HEAD_BYTES))
    except OSError:
        pass
    return h.hexdigest()[:32]


def _soul_sidecar_meta(path: Path) -> dict[str, Any]:
    """Metadata from a ``<file>.meta.json`` sidecar (stdlib only).

    Full soul-header parsing stays with the training callers
    (``checkpoints._load_soul_from_path``); the registry only needs the
    identity fields that make models unique across paths.
    """
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not sidecar.is_file():
        return {}
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Unreadable sidecar %s: %s", sidecar, exc)
        return {}
    if not isinstance(meta, dict):
        return {}
    keep = (
        "name",
        "soul_name",
        "born_at",
        "lineage",
        "base_model",
        "training_dataset",
        "model_type",
        "epochs_trained",
        "final_train_loss",
        "vocab_size",
    )
    out = {k: meta[k] for k in keep if meta.get(k) not in (None, "")}
    if isinstance(out.get("born_at"), str):
        # Legacy sidecars store "...+00:00Z"; repair it before it reaches a UI.
        out["born_at"] = repair_iso(out["born_at"])
    return out


def _safetensors_meta(path: Path) -> dict[str, Any]:
    """Tensor inventory from a safetensors header (no weight data read)."""
    try:
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            if header_len > 100_000_000:
                return {}
            header = json.loads(f.read(header_len))
        if not isinstance(header, dict):
            return {}
        header.pop("__metadata__", None)
        dtypes = sorted({t.get("dtype", "?") for t in header.values()})
        params = 0
        for tensor in header.values():
            shape = tensor.get("shape") or []
            count = 1
            for dim in shape:
                count *= int(dim or 0)
            params += count
        return {
            "tensors": len(header),
            "dtypes": dtypes,
            "tensor_names": sorted(header)[:50],
            "params_approx": params or None,
        }
    except Exception as exc:
        logger.debug("Safetensors header unreadable %s: %s", path, exc)
        return {}


def _npz_meta(path: Path) -> dict[str, Any]:
    """Array inventory of an .npz without loading array data."""
    try:
        names: list[str] = []
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                names = sorted(zf.namelist())[:100]
        return {"arrays": len(names), "array_names": names}
    except Exception as exc:
        logger.debug("Npz inventory failed %s: %s", path, exc)
        return {}


def _record(
    kind: str,
    artifact_id: str,
    path: Path,
    *,
    name: str | None = None,
    tags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
    sniff: bool = False,
) -> dict[str, Any]:
    try:
        size = path.stat().st_size if path.is_file() else 0
    except OSError:
        size = 0
    record: dict[str, Any] = {
        "artifact_id": f"{kind}:{artifact_id}",
        "kind": kind,
        "id": artifact_id,
        "name": name or artifact_id,
        "path": str(path),
        "size_bytes": size,
        "source": "cache" if "external" in path.parts else "local",
        "tags": tags or [],
        "meta": dict(meta or {}),
    }
    if sniff and path.is_file():
        info = _sniff(path)
        record["meta"]["magic"] = info["magic"]
        record["meta"]["fingerprint"] = _fingerprint(path, info["magic"], size)
        suffix = path.suffix.lower()
        if suffix in (".soul", ".slo"):
            record["meta"].update(_soul_sidecar_meta(path))
            record["meta"].setdefault("format", "soul")
        elif suffix == ".safetensors":
            record["meta"].update(_safetensors_meta(path))
            record["meta"].setdefault("format", "safetensors")
        elif suffix == ".npz":
            record["meta"].update(_npz_meta(path))
            record["meta"].setdefault("format", "numpy")
        elif suffix == ".slnc":
            record["meta"].setdefault("format", "slnc")
    return record


def _iter_models() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for root in scan_roots()["model"]:
        for fp in sorted(root.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in MODEL_SUFFIXES:
                out.append(_record("model", fp.name, fp, sniff=True))
    return out


def _iter_datasets() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in scan_roots()["dataset"]:
        try:
            entries = sorted(root.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or entry.name in seen:
                continue
            if entry.name.endswith(".db"):
                continue
            corpus = _find_corpus(entry)
            if corpus is None:
                continue
            seen.add(entry.name)
            try:
                samples = 0
                if corpus.suffix == ".jsonl" and corpus.stat().st_size < 1_000_000:
                    with open(corpus, encoding="utf-8", errors="replace") as f:
                        samples = sum(1 for _ in f)
            except OSError:
                samples = 0
            out.append(
                _record(
                    "dataset",
                    entry.name,
                    corpus,
                    name=entry.name.replace("_", " ").title(),
                    tags=["jsonl"] if corpus.suffix == ".jsonl" else [],
                    meta={"samples": samples, "entry": str(entry)},
                    sniff=True,
                )
            )
    return out


def _iter_checkpoints() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for root in scan_roots()["checkpoint"]:
        for fp in sorted(root.glob("*.soul")) + sorted(root.glob("*.npz")):
            out.append(_record("checkpoint", fp.name, fp, meta={"dir": str(root)}, sniff=True))
    return out


def _iter_adapters() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for root in scan_roots()["adapter"]:
        for fp in sorted(root.glob("*.npz")):
            out.append(_record("adapter", fp.name, fp, sniff=True))
    return out


def _iter_weights() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for root in scan_roots()["weight"]:
        for fp in sorted(root.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in WEIGHT_SUFFIXES:
                # Adapters are tracked under the adapter kind instead.
                if "user_adapters" in fp.parts:
                    continue
                out.append(_record("weight", fp.name, fp, sniff=True))
    return out


def _iter_downloads() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for root in scan_roots()["download"]:
        try:
            entries = sorted(root.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            manifest = entry / ".manifest.json"
            files: list[dict[str, Any]] = []
            if manifest.is_file():
                try:
                    for f in json.loads(manifest.read_text()).get("files", []):
                        files.append(
                            {
                                "path": f.get("path", ""),
                                "size": f.get("size"),
                                "sha256": f.get("sha256"),
                            }
                        )
                except Exception as exc:
                    logger.debug("Unreadable manifest %s: %s", manifest, exc)
            out.append(
                _record(
                    "download",
                    entry.name,
                    entry,
                    tags=["manifest"] if files else [],
                    meta={"files": files},
                )
            )
    return out


_ITERATORS = {
    "model": _iter_models,
    "dataset": _iter_datasets,
    "checkpoint": _iter_checkpoints,
    "adapter": _iter_adapters,
    "weight": _iter_weights,
    "download": _iter_downloads,
}


def list_artifacts(kind: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
    """List artifacts, optionally filtered by kind and substring query."""
    kinds = (kind,) if kind else ARTIFACT_KINDS[:-1]  # exclude generic "file"
    out: list[dict[str, Any]] = []
    for k in kinds:
        iterator = _ITERATORS.get(k)
        if iterator is None:
            continue
        try:
            out.extend(iterator())
        except Exception as exc:
            logger.debug("Artifact scan failed for kind %s: %s", k, exc)
    if query:
        q = query.lower()
        out = [a for a in out if q in a["id"].lower() or q in a["name"].lower()]
    return out


def resolve(kind: str, artifact_id: str) -> str | None:
    """Resolve ``(kind, id)`` to an absolute filesystem path, or None.

    - ``file``: guarded relative path under the file roots (no traversal).
    - other kinds: match by ``id`` (exact) or ``artifact_id``.
    """
    if kind == "file":
        for root in scan_roots()["file"]:
            candidate = (root / artifact_id).resolve()
            if (
                any(
                    str(candidate).startswith(str(r.resolve()))
                    for r in scan_roots()["file"]
                    if r.exists()
                )
                and candidate.exists()
            ):
                return str(candidate)
        return None
    if kind not in _ITERATORS:
        return None
    for artifact in list_artifacts(kind):
        if artifact["id"] == artifact_id or artifact["artifact_id"] == artifact_id:
            return artifact["path"]
    return None


def register(kind: str, path: str | Path, *, name: str | None = None) -> dict[str, Any]:
    """Register a newly written artifact: validate, fingerprint, return record.

    Stage 2 hook for write paths (download complete, import, checkpoint
    save). Read-only itself — the filesystem stays the source of truth;
    the record is what callers attach to completion responses.
    """
    if kind not in ARTIFACT_KINDS or kind == "file":
        raise ValueError(f"Cannot register kind {kind!r}")
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Cannot register missing path: {path}")
    if kind == "dataset" and fp.is_file():
        return _record(kind, fp.parent.name, fp, name=name, sniff=True)
    return _record(kind, fp.name, fp, name=name, sniff=True)


def try_register(kind: str, path: str | Path, *, name: str | None = None) -> dict[str, Any] | None:
    """Best-effort :func:`register` for write paths. Never raises.

    Returns the validated record, or ``None`` when the path is missing or
    the kind is not registerable (logged at debug). Write hooks call this
    so a registry failure can never break a download/import/checkpoint.
    """
    try:
        return register(kind, path, name=name)
    except Exception as exc:
        logger.debug("Artifact registration skipped (%s %s): %s", kind, path, exc)
        return None


def verify(kind: str | None = None, query: str | None = None) -> dict[str, Any]:
    """Verify artifacts by hash/size checks. Returns summary + per-item detail."""
    results = [verify_artifact(a) for a in list_artifacts(kind, query)]
    summary: dict[str, int] = {"ok": 0, "mismatch": 0, "missing": 0, "unverified": 0}
    for result in results:
        status = result["verify"]["status"]
        summary[status] = summary.get(status, 0) + 1
    return {"summary": summary, "total": len(results), "results": results}


def stats() -> dict[str, Any]:
    """Counts and bytes per kind (single scan)."""
    artifacts = list_artifacts()
    by_kind: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        entry = by_kind.setdefault(artifact["kind"], {"count": 0, "bytes": 0})
        entry["count"] += 1
        entry["bytes"] += artifact["size_bytes"]
    return {"total": len(artifacts), "by_kind": by_kind}


def _sha256_file(path: Path) -> str:
    """Streamed sha256 (constant memory, safe for GB weight files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_expected_hash(path: Path) -> str | None:
    """Expected hash from a ``<file>.sha256`` sidecar (``<hash>[  name]``)."""
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        return None
    try:
        return sidecar.read_text(encoding="utf-8").split()[0].strip() or None
    except Exception as exc:
        logger.debug("Unreadable hash sidecar %s: %s", sidecar, exc)
        return None


def verify_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Verify one artifact. Simple logic, three outcomes per check.

    - ``missing``: path does not exist (or is empty).
    - ``mismatch``: expected hash/size known and differs.
    - ``ok``: exists and matches every known baseline.
    - ``unverified``: exists, but no baseline (hash reported for the record).
    """
    path = Path(artifact["path"])
    checks: list[dict[str, str]] = []
    status = "ok"

    if not path.exists():
        return {
            **artifact,
            "verify": {
                "status": "missing",
                "checks": [{"file": artifact["path"], "result": "missing"}],
            },
        }

    if artifact["kind"] == "download":
        for entry in artifact.get("meta", {}).get("files", []):
            rel = entry.get("path", "") if isinstance(entry, dict) else entry
            fp = path / rel if path.is_dir() else path
            if not fp.is_file():
                checks.append({"file": rel, "result": "missing"})
                status = "mismatch"
                continue
            if fp.stat().st_size == 0:
                checks.append({"file": rel, "result": "empty"})
                status = "mismatch"
                continue
            expected = entry.get("sha256") if isinstance(entry, dict) else None
            if expected:
                actual = _sha256_file(fp)
                if actual != expected.lower():
                    checks.append({"file": rel, "result": "mismatch"})
                    status = "mismatch"
                    continue
            checks.append({"file": rel, "result": "ok"})
        if checks:
            return {**artifact, "verify": {"status": status, "checks": checks}}

    if path.is_file():
        if path.stat().st_size == 0:
            return {
                **artifact,
                "verify": {
                    "status": "mismatch",
                    "checks": [{"file": artifact["path"], "result": "empty"}],
                },
            }
        expected = _read_expected_hash(path)
        actual = _sha256_file(path)
        if expected is None:
            return {
                **artifact,
                "verify": {
                    "status": "unverified",
                    "checks": [{"file": artifact["path"], "result": "unverified"}],
                    "sha256": actual,
                },
            }
        ok = actual == expected.lower()
        return {
            **artifact,
            "verify": {
                "status": "ok" if ok else "mismatch",
                "checks": [{"file": artifact["path"], "result": "ok" if ok else "mismatch"}],
                "sha256": actual,
            },
        }

    return {
        **artifact,
        "verify": {
            "status": status,
            "checks": checks or [{"file": artifact["path"], "result": "ok"}],
        },
    }
