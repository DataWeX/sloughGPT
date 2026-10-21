"""AssetStore — one namespace for user-facing files and images (the VFS seam).

Content-addressed blob storage (sha256) + dataclass metadata persisted through
:class:`FileRepository` (one JSON record per asset, atomic writes, migrations)
+ an append-only JSONL event collector that feeds asset history.

Design notes
------------
- **Blobs are content-addressed** under ``<root>/blobs/ab/cd/<sha>``: identical
  bytes are stored once (dedupe is inherent), and the blob path never changes
  when metadata (name, tags, origin) does.
- **Metadata is per-asset** (not a shared sidecar that is rewritten whole-file
  per upload): ``FileRepository`` gives atomic per-key writes + TTL cache.
- **Every mutation is an event** (put/download/delete/migrate): the JSONL log
  is the durable history; the in-memory dashboard ``EventBuffer`` is a
  projection, not the record.
- **URL stability**: migrated legacy assets keep ``legacy_path`` (their
  pre-migration location) in metadata; bytes are COPIED into the CAS, never
  moved or deleted (file-safety rule — originals are user data).
- **Computer-use seam**: ``origin`` is a closed enum
  (upload | generated | screenshot | session | agent-artifact) so future
  computer-use surfaces (screenshots, session captures, agent artifacts) land
  in the same namespace with provenance, without a new storage root.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from domain.infrastructure._internal.atomic_write import atomic_write
from domain.infrastructure._internal.repository import FileRepository

#: Closed set of provenance kinds. Computer-use adds consumers, never new roots.
ORIGINS = ("upload", "generated", "screenshot", "session", "agent-artifact")

#: Asset kinds — an image is a file with an image media type; the view differs,
#: the store does not.
KINDS = ("file", "image")


@dataclass
class AssetMeta:
    """Per-asset metadata record (JSON-safe dataclass)."""

    id: str
    sha256: str
    original_name: str
    media_type: str
    size_bytes: int
    origin: str
    kind: str
    created_at: float
    #: Blob location, POSIX-relative to the store root ("blobs/ab/cd/<sha>").
    storage_path: str
    #: Pre-migration location ("" for assets born in the store). Provenance
    #: only — reads always resolve through ``storage_path``.
    legacy_path: str = ""
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sha256": self.sha256,
            "original_name": self.original_name,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "origin": self.origin,
            "kind": self.kind,
            "created_at": self.created_at,
            "storage_path": self.storage_path,
            "legacy_path": self.legacy_path,
            "tags": list(self.tags),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssetMeta:
        return cls(
            id=data["id"],
            sha256=data["sha256"],
            original_name=data.get("original_name", ""),
            media_type=data.get("media_type", "application/octet-stream"),
            size_bytes=int(data.get("size_bytes", 0)),
            origin=data.get("origin", "upload"),
            kind=data.get("kind", "file"),
            created_at=float(data.get("created_at", 0.0)),
            storage_path=data.get("storage_path", ""),
            legacy_path=data.get("legacy_path", ""),
            tags=list(data.get("tags", [])),
            extra=dict(data.get("extra", {})),
        )


class AssetStore:
    """Filesystem-backed asset store: CAS blobs + per-asset metadata + events.

    Sync by design (see AGENTS.md execution philosophy): callers on the async
    host go through ``asyncio.to_thread`` at the seam.
    """

    def __init__(self, root: str | Path, *, cache_bytes: int = 64 * 1024 * 1024):
        self._root = Path(root)
        self._blobs = self._root / "blobs"
        self._blobs.mkdir(parents=True, exist_ok=True)
        self._meta: FileRepository[AssetMeta] = FileRepository(
            self._root / "meta",
            serializer=AssetMeta,
        )
        self._events_path = self._root / "events.jsonl"
        # LRU of asset_id -> bytes, bounded by total cached bytes. Per-asset
        # entries: two assets sharing one blob each hold their own copy.
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_bytes = 0
        self._cache_budget = cache_bytes
        self._cache_lock = Lock()

    # ── Paths ──

    def _blob_path(self, sha256: str) -> Path:
        return self._blobs / sha256[:2] / sha256[2:4] / sha256

    # ── Writes ──

    def put(
        self,
        data: bytes,
        *,
        original_name: str,
        origin: str,
        media_type: str | None = None,
        kind: str | None = None,
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
        legacy_path: str = "",
    ) -> AssetMeta:
        """Store bytes + metadata and record a ``put`` event.

        Identical content dedupes at the blob layer (one CAS object, one
        metadata record per logical asset).
        """
        if origin not in ORIGINS:
            raise ValueError(f"invalid origin {origin!r}; expected one of {ORIGINS}")
        if media_type is None:
            media_type = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
        if kind is None:
            kind = "image" if media_type.startswith("image/") else "file"
        if kind not in KINDS:
            raise ValueError(f"invalid kind {kind!r}; expected one of {KINDS}")

        sha = hashlib.sha256(data).hexdigest()
        blob = self._blob_path(sha)
        if not blob.exists():
            atomic_write(blob, data)

        meta = AssetMeta(
            id=uuid4().hex,
            sha256=sha,
            original_name=original_name,
            media_type=media_type,
            size_bytes=len(data),
            origin=origin,
            kind=kind,
            created_at=time.time(),
            storage_path=blob.relative_to(self._root).as_posix(),
            legacy_path=legacy_path,
            tags=list(tags or []),
            extra=dict(extra or {}),
        )
        if not self._meta.save(meta.id, meta):
            raise RuntimeError(f"failed to persist asset metadata for {meta.id}")
        self._record_event(
            "put",
            asset_id=meta.id,
            sha256=sha,
            original_name=original_name,
            origin=origin,
            detail={"size_bytes": meta.size_bytes},
        )
        return meta

    def delete(self, asset_id: str, *, purge_blob: bool = False) -> bool:
        """Remove metadata (and the event records a ``delete``).

        The blob is kept unless ``purge_blob`` and no other metadata record
        references the same sha256 (content may be shared across assets).
        """
        meta = self._meta.get(asset_id)
        if meta is None:
            return False
        with self._cache_lock:
            evicted = self._cache.pop(asset_id, None)
            if evicted is not None:
                self._cache_bytes -= len(evicted)
        if purge_blob:
            shared = any(m.sha256 == meta.sha256 for m in self._meta.list() if m.id != asset_id)
            if not shared:
                blob = self._root / meta.storage_path
                if blob.exists():
                    blob.unlink()
        self._meta.delete(asset_id)
        self._record_event(
            "delete",
            asset_id=asset_id,
            sha256=meta.sha256,
            original_name=meta.original_name,
            origin=meta.origin,
        )
        return True

    # ── Reads ──

    def get_meta(self, asset_id: str) -> AssetMeta | None:
        return self._meta.get(asset_id)

    def get_bytes(self, asset_id: str) -> bytes | None:
        if self._cache_budget:
            with self._cache_lock:
                cached = self._cache.get(asset_id)
                if cached is not None:
                    self._cache.move_to_end(asset_id)
                    return cached
        meta = self._meta.get(asset_id)
        if meta is None:
            return None
        blob = self._root / meta.storage_path
        if not blob.exists():
            return None
        data = blob.read_bytes()
        if self._cache_budget:
            with self._cache_lock:
                self._cache[asset_id] = data
                self._cache_bytes += len(data)
                self._cache.move_to_end(asset_id)
                while self._cache_bytes > self._cache_budget and self._cache:
                    _k, evicted = self._cache.popitem(last=False)
                    self._cache_bytes -= len(evicted)
        return data

    def list(self, *, kind: str | None = None, origin: str | None = None) -> list[AssetMeta]:
        items = self._meta.list()
        if kind is not None:
            items = [m for m in items if m.kind == kind]
        if origin is not None:
            items = [m for m in items if m.origin == origin]
        return sorted(items, key=lambda m: m.created_at, reverse=True)

    def search(self, query: str) -> list[AssetMeta]:
        """Substring search over original names and tags (case-insensitive)."""
        q = query.lower()
        return [
            m
            for m in self._meta.list()
            if q in m.original_name.lower() or any(q in t.lower() for t in m.tags)
        ]

    def find_by_sha(self, sha256: str) -> list[AssetMeta]:
        return [m for m in self._meta.list() if m.sha256 == sha256]

    # ── Operator surface (external utility tier — never a chat/agent tool) ──

    def verify_integrity(self) -> dict[str, Any]:
        """Read-only integrity report: metadata rows vs CAS blobs.

        Reports missing blobs (meta without bytes), sha mismatches (bytes
        changed under metadata), and orphan blobs (CAS objects no row
        references). Mutates nothing — the probe-style check an operator runs
        before deciding anything.
        """
        metas = self._meta.list()
        referenced: set[str] = set()
        missing_blobs = 0
        sha_mismatches = 0
        for m in metas:
            referenced.add(Path(m.storage_path).name)
            blob = self._root / m.storage_path
            if not blob.exists():
                missing_blobs += 1
                continue
            if hashlib.sha256(blob.read_bytes()).hexdigest() != m.sha256:
                sha_mismatches += 1
        blobs_on_disk = 0
        orphan_blobs = 0
        orphan_bytes = 0
        for blob in self._blobs.rglob("*"):
            if not blob.is_file():
                continue
            blobs_on_disk += 1
            if blob.name not in referenced:
                orphan_blobs += 1
                orphan_bytes += blob.stat().st_size
        return {
            "assets": len(metas),
            "blobs_on_disk": blobs_on_disk,
            "missing_blobs": missing_blobs,
            "sha_mismatches": sha_mismatches,
            "orphan_blobs": orphan_blobs,
            "orphan_bytes": orphan_bytes,
            "healthy": missing_blobs == 0 and sha_mismatches == 0,
        }

    def prune_orphan_blobs(self) -> dict[str, Any]:
        """Remove CAS blobs no metadata row references.

        Only unreferenced blob files under ``<root>/blobs`` are in scope —
        metadata, events, and legacy directories (user data) are never
        touched (file-safety).
        """
        referenced = {Path(m.storage_path).name for m in self._meta.list()}
        removed = 0
        freed = 0
        for blob in self._blobs.rglob("*"):
            if blob.is_file() and blob.name not in referenced:
                size = blob.stat().st_size
                blob.unlink(missing_ok=True)
                removed += 1
                freed += size
        return {"removed": removed, "freed_bytes": freed}

    def reindex(self) -> dict[str, Any]:
        """Reconcile metadata drift from blob content (the blob wins).

        Repairs ``sha256``/``size_bytes`` when the stored bytes disagree with
        the record; rows whose blob is gone are reported, never deleted —
        losing the last reference to user data is a human decision.
        """
        metas = self._meta.list()
        repaired = 0
        missing_blobs: list[str] = []
        for m in metas:
            blob = self._root / m.storage_path
            if not blob.exists():
                missing_blobs.append(m.id)
                continue
            data = blob.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if sha != m.sha256 or len(data) != m.size_bytes:
                m.sha256 = sha
                m.size_bytes = len(data)
                if not self._meta.save(m.id, m):
                    raise RuntimeError(f"failed to reindex asset metadata for {m.id}")
                repaired += 1
        return {"assets": len(metas), "repaired": repaired, "missing_blobs": missing_blobs}

    # ── Events / history ──

    def record_event(
        self,
        action: str,
        *,
        asset_id: str = "",
        sha256: str = "",
        original_name: str = "",
        origin: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append one event line to the durable history log (e.g. downloads,
        which routers record without going through ``put``)."""
        self._record_event(
            action,
            asset_id=asset_id,
            sha256=sha256,
            original_name=original_name,
            origin=origin,
            detail=detail,
        )

    def _record_event(
        self,
        action: str,
        *,
        asset_id: str = "",
        sha256: str = "",
        original_name: str = "",
        origin: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "ts": time.time(),
            "action": action,
            "asset_id": asset_id,
            "sha256": sha256,
            "original_name": original_name,
            "origin": origin,
            "detail": detail or {},
        }
        self._root.mkdir(parents=True, exist_ok=True)
        with open(self._events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def history(
        self,
        *,
        asset_id: str = "",
        action: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Newest-first event history (optionally filtered by asset/action)."""
        if not self._events_path.exists():
            return []
        lines = self._events_path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if asset_id and event.get("asset_id") != asset_id:
                continue
            if action and event.get("action") != action:
                continue
            out.append(event)
            if len(out) >= limit:
                break
        return out

    # ── Legacy migration (uploads + gallery → store) ──

    def migrate_uploads(self, entries: list[tuple[Path, dict[str, Any]]]) -> dict[str, int]:
        """Adopt legacy ``data/uploads`` files: ``(path, legacy_meta)`` pairs.

        Bytes are COPIED into the CAS (originals stay in place — file-safety);
        ``legacy_path`` records provenance. Idempotent: entries whose path was
        already migrated are skipped.
        """
        return self._migrate(
            [
                (
                    path,
                    {
                        "original_name": legacy.get("original_name")
                        or legacy.get("filename")
                        or path.name,
                        "origin": "upload",
                        "tags": list(legacy.get("tags", [])),
                    },
                )
                for path, legacy in entries
                if path.exists()
            ]
        )

    def migrate_gallery(self, paths: list[Path]) -> dict[str, int]:
        """Adopt legacy ``data/gallery`` images (generated_*)."""
        return self._migrate(
            [
                (path, {"original_name": path.name, "origin": "generated", "tags": []})
                for path in paths
                if path.exists()
            ]
        )

    def _migrate(self, entries: list[tuple[Path, dict[str, Any]]]) -> dict[str, int]:
        existing = {m.legacy_path for m in self._meta.list() if m.legacy_path}
        scanned = migrated = skipped = 0
        for path, spec in entries:
            scanned += 1
            key = str(path)
            if key in existing:
                skipped += 1
                continue
            meta = self.put(
                path.read_bytes(),
                original_name=spec["original_name"],
                origin=spec["origin"],
                tags=spec.get("tags", []),
                extra={"migrated_from": key},
                legacy_path=key,
            )
            self._record_event(
                "migrate",
                asset_id=meta.id,
                sha256=meta.sha256,
                original_name=meta.original_name,
                origin=meta.origin,
                detail={"from": key},
            )
            migrated += 1
        return {"scanned": scanned, "migrated": migrated, "skipped": skipped}


_default_store: AssetStore | None = None


def get_asset_store() -> AssetStore:
    """Process-wide store rooted at ``$SLO_DATA_DIR/assets`` or ``<repo>/data/assets``."""
    global _default_store
    if _default_store is None:
        data_dir = os.environ.get("SLO_DATA_DIR")
        root = (
            Path(data_dir) / "assets"
            if data_dir
            else Path(__file__).resolve().parents[3] / "data" / "assets"
        )
        _default_store = AssetStore(root)
    return _default_store
