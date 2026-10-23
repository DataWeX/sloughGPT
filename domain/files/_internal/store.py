"""FileStore — compressed blob store core (sync, standalone-runnable).

Codec policy is *proven*, not guessed — benchmarked by
``scripts/benchmark_downcraft_compression.py`` (card fba1ce5e, 2026-10-23):

* **text/code class** → zstd level 3 compress-at-rest (~7x on real source;
  the sweet spot — level 19 gains 5 ratio points for 130x slower encode).
  The stored zstd frame doubles as the wire format: a client that sends
  ``Accept-Encoding: zstd`` gets the stored bytes with zero transcoding.
* **model-weight class** → raw passthrough. GGUF/safetensors/float tensors
  are proven incompressible (~1.0x); compressing would burn CPU for nothing.
* **unknown class** → zstd-if-smaller probe (the downcraft auto-skip rule,
  inverted): stream both, keep whichever wins, capped at
  ``UNKNOWN_PROBE_MAX`` so a multi-GB unknown blob never pays double disk.

Grandfathering: blobs live in the same ``data/uploads`` directory the files
router always used, and the MogDB ``files`` collection keeps its schema —
``codec``/``sha256``/``stored_size`` are additive fields. Legacy entries
have no ``codec`` field and are read back as raw bytes, exactly as before.

This module is sync and import-safe from scripts/CLI/tests with no API
layer — same seam rule as PGQ (the async host calls it via
``asyncio.to_thread``; the core never imports asyncio).
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO

import zstandard as zstd

logger = logging.getLogger("slo.files.store")

CHUNK = 1024 * 1024
ZSTD_LEVEL = 3
UNKNOWN_PROBE_MAX = 64 * 1024 * 1024

#: Proven-incompressible artifacts — served/stored raw, never transcoded.
MODEL_EXTENSIONS = frozenset(
    {
        ".gguf",
        ".safetensors",
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".bin",
        ".npz",
        ".tflite",
        ".engine",
    }
)

#: Code/text — zstd compress-at-rest (proven ~7x on real source corpora).
TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".log",
        ".toml",
        ".ini",
        ".cfg",
        ".rst",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".html",
        ".css",
        ".sh",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".rs",
        ".go",
        ".java",
        ".sql",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = _REPO_ROOT / "data" / "uploads"


class FileStore:
    """Compressed blob store: put/get/list/delete with a per-class codec.

    Metadata lives in a MogDB collection named ``files`` (the same
    collection the files router already used); blobs live under *root*.
    Pass ``db`` to share a pooled MogDB instance; without one the store
    opens its own at ``$SLO_FILESTORE_DB`` or ``<root.parent>/filestore_mogdb``.
    """

    def __init__(self, root: Path | None = None, db: Any = None):
        self.root = Path(root) if root else DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = db
        self._own_db: Any = None
        self._lock = threading.Lock()

    # ── metadata seam ──────────────────────────────────────────────

    def _col(self) -> Any:
        if self._db is not None:
            return self._db.collection("files")
        if self._own_db is None:
            from mogdb import MogDB

            db_path = os.environ.get("SLO_FILESTORE_DB") or str(
                self.root.parent / "filestore_mogdb"
            )
            self._own_db = MogDB(db_path)
        return self._own_db.collection("files")

    # ── codec policy ───────────────────────────────────────────────

    @staticmethod
    def codec_for(filename: str) -> str:
        """Return ``"raw"`` | ``"zstd"`` | ``"unknown"`` for *filename*."""
        ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
        if ext in MODEL_EXTENSIONS:
            return "raw"
        return "zstd" if ext in TEXT_EXTENSIONS else "unknown"

    # ── write path ─────────────────────────────────────────────────

    def put(self, data: bytes, filename: str, tags: list[str] | None = None) -> dict:
        """Store bytes; thin wrapper over :meth:`put_stream`."""
        return self.put_stream(io.BytesIO(data), filename, tags)

    def put_stream(
        self,
        fileobj: BinaryIO,
        filename: str,
        tags: list[str] | None = None,
        max_bytes: int | None = None,
    ) -> dict:
        """Stream *fileobj* to a blob, applying the codec policy.

        Memory-safe for multi-GB uploads: reads/writes in ``CHUNK`` pieces,
        never materializes the whole payload. Raises ``ValueError`` when
        *max_bytes* is exceeded (caller maps to HTTP 413) or when the file
        has no extension (contract shared with the legacy router).
        """
        if not filename or "." not in filename.rsplit("/", 1)[-1]:
            raise ValueError("File must have an extension")
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        policy = self.codec_for(filename)

        fid = self._reserve_id(filename)
        final_stem = f"{fid}{ext}"
        tmp_raw = self.root / f".tmp-{fid}.raw"
        tmp_zst = self.root / f".tmp-{fid}.zst"

        sha = hashlib.sha256()
        raw_size = 0
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        zwriter = None
        if policy in ("zstd", "unknown"):
            zwriter = cctx.stream_writer(tmp_zst.open("wb"))
        over_max = False

        try:
            with tmp_raw.open("wb") as raw_f:
                while True:
                    chunk = fileobj.read(CHUNK)
                    if not chunk:
                        break
                    raw_size += len(chunk)
                    if max_bytes is not None and raw_size > max_bytes:
                        over_max = True
                        break
                    sha.update(chunk)
                    raw_f.write(chunk)
                    if zwriter is not None:
                        zwriter.write(chunk)
                        if policy == "unknown" and raw_size > UNKNOWN_PROBE_MAX:
                            # Probe cap reached — stop paying double disk.
                            zwriter.close()
                            zwriter = None
                            tmp_zst.unlink(missing_ok=True)
        finally:
            if zwriter is not None:
                zwriter.close()
        if over_max:
            tmp_raw.unlink(missing_ok=True)
            tmp_zst.unlink(missing_ok=True)
            raise ValueError(f"File too large (max {max_bytes} bytes)")

        # ── pick the winner ──
        codec = "raw"
        if policy == "zstd" and raw_size > 0:
            codec = "zstd"
        elif policy == "unknown" and tmp_zst.exists() and raw_size > 0:
            codec = "zstd" if tmp_zst.stat().st_size < tmp_raw.stat().st_size else "raw"
        elif policy == "unknown":
            tmp_zst.unlink(missing_ok=True)

        if codec == "zstd" and raw_size == 0:
            # Empty payload: a zstd frame of an empty file is still valid, but
            # keep the layout trivial and store raw.
            codec = "raw"
            tmp_zst.unlink(missing_ok=True)

        stored_name = final_stem + (".zst" if codec == "zstd" else "")
        src = tmp_zst if codec == "zstd" else tmp_raw
        os.replace(src, self.root / stored_name)
        tmp_raw.unlink(missing_ok=True)
        tmp_zst.unlink(missing_ok=True)

        meta = {
            "file_id": fid,
            "filename": stored_name,
            "original_name": filename,
            "extension": ext,
            "size_bytes": raw_size,
            "stored_size": (self.root / stored_name).stat().st_size,
            "codec": codec,
            "sha256": sha.hexdigest(),
            "chars": 0,
            "pages": 1,
            "uploaded_at": time.time(),
            "tags": tags or [],
        }
        self._upsert(meta)
        return meta

    def _reserve_id(self, filename: str) -> str:
        """Timestamped id (legacy convention) + uuid tail on collision."""
        safe = re.sub(r"[^\w\-]", "_", filename)
        col = self._col()
        for _ in range(8):
            fid = f"{int(time.time())}_{safe}_{uuid.uuid4().hex[:6]}"
            if col.find_one({"file_id": fid}) is None:
                return fid
        return f"{int(time.time())}_{safe}_{uuid.uuid4().hex[:12]}"

    def _upsert(self, meta: dict) -> None:
        col = self._col()
        existing = col.find_one({"file_id": meta["file_id"]})
        if existing:
            col.update_one({"file_id": meta["file_id"]}, {"$set": meta})
        else:
            col.insert_one(dict(meta))

    def update_meta(self, meta: dict) -> None:
        """Public metadata write (e.g. chars count after ingest)."""
        self._upsert(meta)

    # ── read path ──────────────────────────────────────────────────

    def get_meta(self, file_id: str) -> dict:
        """Metadata for *file_id*; raises ``FileNotFoundError`` if absent."""
        meta = self._col().find_one({"file_id": file_id})
        if meta is None:
            raise FileNotFoundError(file_id)
        path = self.root / meta["filename"]
        if not path.is_file():
            raise FileNotFoundError(meta["filename"])
        return meta

    def get(self, file_id: str) -> tuple[bytes, dict]:
        """Return ``(raw_bytes, meta)`` — decompressing on the way out."""
        meta = self.get_meta(file_id)
        path = self.root / meta["filename"]
        if meta.get("codec", "raw") == "zstd":
            with path.open("rb") as f:
                data = zstd.ZstdDecompressor().stream_reader(f).read()
        else:
            data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != meta.get("sha256", hashlib.sha256(data).hexdigest()):
            raise ValueError(f"Corrupt blob: sha256 mismatch for {file_id}")
        return data, meta

    def iter_chunks(self, file_id: str, chunk_size: int = CHUNK):
        """Yield raw bytes in *chunk_size* pieces (streaming download)."""
        meta = self.get_meta(file_id)
        path = self.root / meta["filename"]
        if meta.get("codec", "raw") == "zstd":
            with path.open("rb") as f, zstd.ZstdDecompressor().stream_reader(f) as reader:
                while True:
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        else:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

    def open_stored(self, file_id: str) -> tuple[Path, dict]:
        """Path of the *stored* blob + meta — for Content-Encoding passthrough.

        Only meaningful when ``meta["codec"] == "zstd"``: the stored frame is
        standard zstd, so it can go on the wire untouched.
        """
        meta = self.get_meta(file_id)
        return self.root / meta["filename"], meta

    # ── collection ops ─────────────────────────────────────────────

    def list(self, tag: str | None = None) -> list[dict]:
        """All metadata entries whose blob still exists (optionally by tag)."""
        out = []
        for meta in self._col().find():
            fid = meta.get("file_id", "")
            if not fid:
                continue
            if (self.root / meta["filename"]).is_file() and (tag is None or tag in meta.get("tags", [])):
                out.append(meta)
        return out

    def delete(self, file_id: str) -> None:
        """Remove blob + metadata; ``FileNotFoundError`` when absent."""
        meta = self.get_meta(file_id)
        (self.root / meta["filename"]).unlink(missing_ok=True)
        self._col().delete_one({"file_id": file_id})
