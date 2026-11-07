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
import platform
import re
import struct
import subprocess
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from domain.shared import repair_iso, utc_now_iso  # stdlib-only, no third-party imports

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


_corpus_memo: dict[str, tuple[int, Path | None]] = {}
_corpus_memo_lock = threading.Lock()
_CORPUS_MEMO_MAX = 8192


def find_corpus_file(entry_dir: Path) -> Path | None:
    """Public corpus picker: shared priority for all dataset resolvers.

    Lets callers keep their own roots (and test seams) while sharing one
    priority implementation with the registry scan.

    Memoized on the directory's own mtime: corpus selection only depends
    on which files are directly present in *entry_dir*, so a stat is a
    sound validity key (any add/remove/rename bumps the mtime, content
    edits don't change the result). Measured: the glob/sort fan-out here
    was 88% of the datasets list compute (0.3-0.5s on every cache miss).
    """
    try:
        mtime_ns = entry_dir.stat().st_mtime_ns
    except OSError:
        return _find_corpus(entry_dir)
    key = str(entry_dir)
    with _corpus_memo_lock:
        hit = _corpus_memo.get(key)
        if hit is not None and hit[0] == mtime_ns:
            return hit[1]
    result = _find_corpus(entry_dir)
    with _corpus_memo_lock:
        if len(_corpus_memo) >= _CORPUS_MEMO_MAX:
            _corpus_memo.clear()
        _corpus_memo[key] = (mtime_ns, result)
    return result


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


def list_artifacts(
    kind: str | None = None,
    query: str | None = None,
    *,
    with_provenance: bool = False,
) -> list[dict[str, Any]]:
    """List artifacts, optionally filtered by kind and substring query.

    ``with_provenance=True`` attaches a ``parents`` list per artifact — the
    union of its ``.prov.json`` sidecar parents (sidecar first) and parents
    derived from soul/export metadata, deduped. Default ``False`` keeps the
    hot path (``resolve``, ``verify``) free of per-artifact sidecar reads.
    """
    kinds = (kind,) if kind else ARTIFACT_KINDS[:-1]  # exclude generic "file"
    out: list[dict[str, Any]] = []
    for k in kinds:
        iterator = _ITERATORS.get(k)
        if iterator is None:
            continue
        try:
            out.extend(iterator())
        except Exception as exc:  # noqa: BLE001 — one bad kind must not hide the rest
            logger.debug("Artifact scan failed for kind %s: %s", k, exc)
    if with_provenance:
        for artifact in out:
            artifact["parents"] = _parents_for(artifact)
    if query:
        q = query.lower()
        out = [a for a in out if q in a["id"].lower() or q in a["name"].lower()]
    return out


def _parents_for(artifact: dict[str, Any]) -> list[str]:
    """Parents of one artifact record: sidecar union metadata, deduped.

    Total over junk: unreadable sidecars and non-dict metas each contribute
    nothing, and a record with neither resolves to an empty list (a root).
    """
    parents: list[str] = []
    try:
        record = read_provenance(artifact.get("path", ""))
    except Exception:  # noqa: BLE001 — provenance must never break a listing
        record = None
    if record is not None:
        parents.extend(record.parents)
    for parent in parents_from_metadata(artifact.get("meta")):
        if parent not in parents:
            parents.append(parent)
    return parents


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


# ── Provenance / lineage ────────────────────────────────────────────────
#
# Stage 3: attach *who/what produced this* to an artifact, and allow a
# reverse walk back to roots. Deliberately append-only and read-only over
# existing data: an artifact with no provenance sidecar is a normal, valid
# state (every adapter written before this landed has none). Nothing here
# gates or slows the model-weight path — provenance is a small JSON sidecar
# written once at artifact creation, never read during inference/training.
#
# The lineage model mirrors ``token_tree.decompose``: each artifact stores
# its direct ``parents`` (the artifact_ids it was derived from), and
# ``lineage()`` walks that graph backwards to roots. A merge records its
# inputs; the inputs themselves may have their own parents, so the walk is
# recursive and cycle-guarded.

PROVENANCE_SUFFIX = ".prov.json"

# Config keys folded into the fingerprint — the knobs that change an
# artifact's *meaning* (which model, which quantization, which adapter
# rank). Extended deliberately, not exhaustively: this is a cheap identity
# hint, not a full environment capture.
_PROVENANCE_CONFIG_KEYS = (
    "SLO_MODEL",
    "SLO_QUANT",
    "SLO_ADAPTER_RANK",
    "SLO_ADAPTER_ALPHA",
    "SLO_CACHE_DIR",
    "SLO_BENCH_DATA_DIR",
)

_git_memo: dict[str, tuple[str | None, bool | None]] = {}
_git_memo_lock = threading.Lock()


def _git_state() -> tuple[str | None, bool | None]:
    """Short commit + dirty flag for the repo root, memoized per process.

    Memoized because provenance is stamped on artifact *creation* (cold,
    off the hot path) but a long-lived server would otherwise shell out to
    git on every record. Returns ``(None, None)`` outside a git checkout.
    """
    with _git_memo_lock:
        hit = _git_memo.get("root")
        if hit is not None:
            return hit
    commit: str | None = None
    dirty: bool | None = None
    try:
        root = str(repo_root())
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            commit = out.stdout.strip()
            # A commit hash is only meaningful together with cleanliness:
            # a dirty tree means the artifact was NOT built from that hash.
            st = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if st.returncode == 0:
                dirty = bool(st.stdout.strip())
    except Exception as exc:  # noqa: BLE001 — provenance must never raise
        logger.debug("git state unavailable: %s", exc)
    with _git_memo_lock:
        _git_memo["root"] = (commit, dirty)
    return commit, dirty


def config_fingerprint() -> str:
    """Stable hash of the provenance-relevant config (env knobs).

    Identical config -> identical fingerprint, so two artifacts can be
    compared for "were you built under the same settings?" without diffing
    whole environments.
    """
    try:
        payload = {k: os.environ.get(k, "") for k in _PROVENANCE_CONFIG_KEYS}
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
    except Exception as exc:  # noqa: BLE001
        logger.debug("config fingerprint unavailable: %s", exc)
        return ""


def provenance_snapshot() -> dict[str, Any]:
    """Capture the current code+platform provenance.

    Returns commit (or ``None``), dirty (``None`` when unknown), python
    version, platform machine/system, hostname, and the config fingerprint.
    Cheap and side-effect free; safe to call from any write path.
    """
    commit, dirty = _git_state()
    try:
        hostname = platform.node()
    except Exception:  # noqa: BLE001
        hostname = ""
    return {
        "commit": commit,
        "dirty": dirty,
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "hostname": hostname,
        "config_fingerprint": config_fingerprint(),
    }


@dataclass
class ProvenanceRecord:
    """Immutable-ish provenance + lineage for one artifact.

    ``parents`` holds artifact_ids (``kind:id``) this artifact derives from.
    ``empty`` when there is no sidecar — the normal state for pre-existing
    artifacts, which is why every accessor here is total.
    """

    artifact_id: str
    kind: str
    parents: list[str] = field(default_factory=list)
    created_at: str = ""
    commit: str | None = None
    dirty: bool | None = None
    python: str = ""
    platform: str = ""
    hostname: str = ""
    config_fingerprint: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "parents": list(self.parents),
            "created_at": self.created_at,
            "commit": self.commit,
            "dirty": self.dirty,
            "python": self.python,
            "platform": self.platform,
            "hostname": self.hostname,
            "config_fingerprint": self.config_fingerprint,
        }
        if self.extra:
            out["extra"] = dict(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceRecord:
        parents = data.get("parents") or []
        return cls(
            artifact_id=str(data.get("artifact_id", "")),
            kind=str(data.get("kind", "")),
            parents=[str(p) for p in parents],
            created_at=str(data.get("created_at", "")),
            commit=data.get("commit"),
            dirty=data.get("dirty"),
            python=str(data.get("python", "")),
            platform=str(data.get("platform", "")),
            hostname=str(data.get("hostname", "")),
            config_fingerprint=str(data.get("config_fingerprint", "")),
            extra=dict(data.get("extra") or {}),
        )


def provenance_path(artifact_path: str | Path) -> Path:
    """Sidecar path for an artifact: ``<file>.prov.json``."""
    p = Path(artifact_path)
    return p.with_name(p.name + PROVENANCE_SUFFIX)


def write_provenance(
    artifact_path: str | Path,
    kind: str,
    artifact_id: str,
    *,
    parents: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> ProvenanceRecord:
    """Stamp an artifact with a fresh provenance sidecar.

    Captures the *current* git/platform state at write time. Overwrites any
    existing sidecar (last writer wins) — provenance describes the most
    recent derivation of the artifact at that path.
    """
    snap = provenance_snapshot()
    record = ProvenanceRecord(
        artifact_id=f"{kind}:{artifact_id}",
        kind=kind,
        parents=list(parents or []),
        created_at=utc_now_iso(),
        commit=snap["commit"],
        dirty=snap["dirty"],
        python=snap["python"],
        platform=snap["platform"],
        hostname=snap["hostname"],
        config_fingerprint=snap["config_fingerprint"],
        extra=dict(extra or {}),
    )
    sidecar = provenance_path(artifact_path)
    try:
        sidecar.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — provenance must never break a save
        logger.debug("Provenance write skipped for %s: %s", artifact_path, exc)
    return record


def read_provenance(artifact_path: str | Path) -> ProvenanceRecord | None:
    """Read an artifact's provenance sidecar, or ``None`` when absent.

    Total over missing/corrupt sidecars: a broken sidecar degrades to "no
    provenance recorded" rather than raising, so a lineage walk over a
    partially-stamped store still returns what it can.
    """
    sidecar = provenance_path(artifact_path)
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unreadable provenance %s: %s", sidecar, exc)
        return None
    if not isinstance(data, dict):
        return None
    return ProvenanceRecord.from_dict(data)


# Export/training metadata fields that name a parent artifact, and the kind
# each maps to. Checked flat AND under the caller's ``"metadata"`` sub-dict —
# ``slonet.export_to_sou`` nests caller metadata there when it merges.
_PROVENANCE_PARENT_FIELDS = (
    ("training_dataset", "dataset"),
    ("base_model", "model"),
    ("parent_model", "model"),
    ("teacher_model", "model"),
)


def parents_from_metadata(metadata: Any) -> list[str]:
    """Derive lineage parent artifact_ids from export/training metadata.

    Total over junk: non-dict input, missing keys, blank or non-string
    values each simply contribute no parent.
    """
    if not isinstance(metadata, dict):
        return []
    layers = [metadata]
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        layers.append(nested)
    parents: list[str] = []
    for layer in layers:
        for key, kind in _PROVENANCE_PARENT_FIELDS:
            value = layer.get(key)
            if isinstance(value, str) and value.strip():
                artifact_id = f"{kind}:{value.strip()}"
                if artifact_id not in parents:
                    parents.append(artifact_id)
    return parents


def stamp_export(
    path: str | Path,
    kind: str,
    metadata: Any = None,
    *,
    extra: dict[str, Any] | None = None,
) -> ProvenanceRecord | None:
    """Best-effort provenance stamp for a freshly written artifact.

    The single write-path hook for exports: derives ``parents`` from
    training metadata (which dataset/model produced this file) and writes
    the ``.prov.json`` sidecar, so :func:`lineage` can walk the result.
    Never raises — provenance must never break an export — and returns
    ``None`` when the artifact file does not exist. Last writer wins, so a
    checkpoint saver may re-stamp a file an export already stamped.
    """
    try:
        fp = Path(path)
        if not fp.is_file():
            return None
        return write_provenance(
            fp,
            kind,
            fp.name,
            parents=parents_from_metadata(metadata),
            extra=extra,
        )
    except Exception as exc:  # noqa: BLE001 — provenance must never break a save
        logger.debug("Export provenance stamp skipped (%s): %s", path, exc)
        return None


def lineage(
    kind: str,
    artifact_id: str,
    *,
    path: str | Path | None = None,
    root_dir: str | Path | None = None,
    max_depth: int = 32,
) -> dict[str, Any]:
    """Walk an artifact's parent graph back to its roots.

    Returns a dict with the starting ``root`` artifact_id, every visited
    ``nodes`` (artifact_id -> provenance dict), and ``roots`` (the artifact
    ids with no further parents). Mirrors ``token_tree.decompose``: recursive
    descent over recorded parents, cycle-guarded by a visited set and capped
    at ``max_depth`` so a malformed graph cannot hang the caller.

    ``path`` short-circuits resolution for the *starting* artifact when the
    caller already knows its location. Without it, artifacts are found via
    the registry's fixed scan roots — fine for default stores, wrong for a
    ``PerUserLoRAStore`` constructed with a custom ``store_path``, which
    would otherwise resolve to ``None`` and hide a real parent graph.

    ``root_dir`` is the fallback base for resolving parents: when an
    artifact is not under a registry scan root, ``root_dir/<id>`` is tried.
    Pass the store directory so a lineage walk stays inside one store.

    Artifacts with no provenance sidecar resolve to a node with empty
    parents (a root), so the walk is total over pre-existing data.
    """
    start = f"{kind}:{artifact_id}"
    visited: dict[str, dict[str, Any]] = {}
    roots: list[str] = []
    seen: set[str] = set()

    def _resolve_path(aid: str, *, is_start: bool = False) -> Path | None:
        if is_start and path is not None:
            return Path(path)
        if ":" not in aid:
            return None
        k, _, i = aid.partition(":")
        p = resolve(k, i)
        return Path(p) if p else None

    def _walk(aid: str, depth: int, *, is_start: bool = False) -> None:
        if aid in seen or depth > max_depth:
            return
        seen.add(aid)
        resolved = _resolve_path(aid, is_start=is_start)
        if resolved is None and root_dir is not None and ":" in aid:
            # Store-relative fallback: a custom store_path (e.g. a test tmp
            # dir or a per-tenant store) is not a registry scan root, so
            # resolve() cannot see it. Fall back to root_dir/<id>.
            candidate = Path(root_dir) / aid.partition(":")[2]
            if candidate.is_file():
                resolved = candidate
        record = read_provenance(resolved) if resolved else None
        if record is None:
            # No sidecar (or unresolvable): treat as a root — total, not fatal.
            visited[aid] = {
                "artifact_id": aid,
                "parents": [],
                "resolved": resolved is not None,
            }
            roots.append(aid)
            return
        node = record.to_dict()
        node["resolved"] = resolved is not None
        visited[aid] = node
        parents = record.parents
        if not parents:
            roots.append(aid)
            return
        for parent in parents:
            _walk(parent, depth + 1)

    _walk(start, 0, is_start=True)
    # De-dup roots while preserving discovery order.
    uniq_roots = list(dict.fromkeys(roots))
    return {"root": start, "nodes": visited, "roots": uniq_roots, "depth": len(seen)}
