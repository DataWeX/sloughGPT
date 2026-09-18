"""Just-cache tags — single source of truth for cache location, entry
classification (kind/tags), and per-file mime.

The cache root is the existing external-download root
(``SLO_CACHE_DIR/external``); the root logic lives in
``domain.infrastructure._internal.external_download`` and is reused here,
not duplicated. Tag/mime metadata is stored additively inside each entry's
``.manifest.json`` (plus the legacy ``.metadata.json`` sidecar when present),
so old readers keep working.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("slo.cache_tags")

_VALID_DATASET_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")

# Folder names that are adapters, not training corpora.
ADAPTER_NAMES = {"user_adapters", "knowledge_adapter"}

# Folder names that are media/raw uploads, not training corpora.
MEDIA_NAMES = {"gallery", "uploads", "logged_responses"}

# Suffixes / names that mark internal system stores, never datasets.
SYSTEM_SUFFIXES = ("_mogdb", ".db")
SYSTEM_NAMES = {
    "agents",
    "agent_runs",
    "auth_mogdb",
    "backups",
    "benchmark_results",
    "chat_sessions",
    "companion_json",
    "companion_mogdb",
    "consciousness",
    "data_filter_mogdb",
    "datasets",
    "docstore",
    "error_log",
    "errors_json",
    "errors_mogdb",
    "eval_results",
    "experiments",
    "exports",
    "feedback",
    "kg_pipeline",
    "knowledge",
    "knowledge_graph_json",
    "knowledge_json",
    "knowledge_mogdb",
    "learner",
    "mobile_training",
    "model_catalog",
    "model_catalog_json",
    "model_health_json",
    "rag_store",
    "response_logs",
    "response_mogdb",
    "shell_permissions_mogdb",
    "token_trees",
    "trait_snapshots",
    "trait_weights_json",
    "trait_weights_mogdb",
    "training_exports",
}

_CORPUS_CANDIDATES = ("corpus.jsonl", "input.txt", "train.txt", "text.txt")

_EXTENSION_MIME = {
    ".jsonl": "application/x-ndjson",
    ".ndjson": "application/x-ndjson",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".npz": "application/x-numpy",
    ".soul": "application/x-sloughgpt-soul",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}


def get_cache_root() -> Path:
    """Return the just-cache root (existing external-download root)."""
    from domain.infrastructure._internal.external_download import _get_cache_root

    return _get_cache_root()


def classify_kind(entry_name: str, entry_dir: Path | None = None) -> str:
    """Classify a cache/data entry as dataset/adapter/system/media.

    Adapters and system stores must never be offered as training datasets.
    """
    name = entry_name
    if name in ADAPTER_NAMES:
        return "adapter"
    if name in MEDIA_NAMES:
        return "media"
    if name in SYSTEM_NAMES or name.endswith(SYSTEM_SUFFIXES):
        return "system"
    if entry_dir is not None and entry_dir.is_dir():
        if list(entry_dir.glob("*.npz")) and not any(
            (entry_dir / c).exists() for c in _CORPUS_CANDIDATES
        ):
            return "adapter"
    return "dataset"


def guess_mime(path: str | Path) -> str:
    """Best-effort mime for a cached file (extension map, then stdlib)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in _EXTENSION_MIME:
        return _EXTENSION_MIME[ext]
    mimetypes.init()
    mime, _ = mimetypes.guess_type(str(p))
    return mime or "application/octet-stream"


def find_corpus_file(entry_dir: Path) -> Path | None:
    """Pick the training corpus file inside an entry (priority order)."""
    for name in _CORPUS_CANDIDATES:
        candidate = entry_dir / name
        if candidate.is_file():
            return candidate
    txt_files = sorted(entry_dir.glob("*.txt"))
    if txt_files:
        return txt_files[0]
    jsonl_files = sorted(entry_dir.glob("*.jsonl"))
    if jsonl_files:
        return jsonl_files[0]
    return None


def read_entry_meta(entry_dir: Path) -> dict[str, Any]:
    """Merge tag metadata from .manifest.json + .metadata.json sidecars."""
    meta: dict[str, Any] = {}
    for sidecar in (".manifest.json", ".metadata.json"):
        p = entry_dir / sidecar
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", p, exc)
            continue
        if isinstance(data, dict):
            meta.update(data)
    return meta


def write_entry_meta(
    entry_dir: Path,
    *,
    kind: str,
    tags: list[str] | None = None,
    source: str | None = None,
    mime: str | None = None,
) -> None:
    """Write tag metadata additively into the entry .manifest.json."""
    entry_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = entry_dir / ".manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", manifest_path, exc)
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing["kind"] = kind
    if tags is not None:
        merged = list(dict.fromkeys([*(existing.get("tags") or []), *tags]))
        existing["tags"] = merged
    if source is not None:
        existing.setdefault("source", source)
    if mime is not None:
        existing.setdefault("mime", mime)
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def entry_tags(entry_dir: Path, kind: str | None = None) -> list[str]:
    """Effective tag list for an entry: stored tags + inferred kind tag."""
    kind = kind or classify_kind(entry_dir.name, entry_dir)
    stored = read_entry_meta(entry_dir).get("tags") or []
    tags = list(dict.fromkeys([*(stored if isinstance(stored, list) else []), kind]))
    corpus = find_corpus_file(entry_dir)
    if corpus is not None and corpus.suffix.lower() == ".jsonl":
        tags.append("jsonl")
    return tags


def iter_cache_entries() -> list[Path]:
    """List entry directories currently in the just-cache root."""
    root = get_cache_root()
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def resolve_in_cache(dataset_id: str) -> str:
    """Resolve a dataset id to its corpus file inside the just-cache root.

    Returns "" when the id is not cached (caller falls back to legacy dirs).
    Raises ValueError on invalid ids, matching resolve_dataset_path.
    """
    if not _VALID_DATASET_ID.match(dataset_id):
        raise ValueError(f"Invalid dataset ID: {dataset_id!r}")
    safe = dataset_id.replace("/", "__")
    candidate = get_cache_root() / safe
    if not candidate.is_dir():
        return ""
    corpus = find_corpus_file(candidate)
    return str(corpus) if corpus is not None else ""
