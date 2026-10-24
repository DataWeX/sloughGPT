"""FileHub — pull-through cache core (sync, standalone-runnable).

Answers "I asked a model hub like HuggingFace for a file — it is already in
my hub, stored per codec policy, and served from the LAN without re-fetch."

* Pull path is downcraft's ``download_file`` (resume + checksums) — the
  proven client, not a reimplementation.
* Storage is ``FileStore`` — same codec policy: model artifacts land raw
  (they arrive quantization-compressed, byte-incompressible — proven),
  text/code lands zstd at rest.
* Dedup by sha256: re-pulling an identical artifact returns the existing
  blob instead of storing twice.

Sync by design (same seam rule as PGQ): the wire host calls it from its
own threads/loops; the core never imports asyncio or wire.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from domain.files._internal.store import CHUNK, FileStore

logger = logging.getLogger("slo.files.hub")


class FileHub:
    """Pull-through cache over a :class:`FileStore`, with token auth."""

    def __init__(self, store: FileStore | None = None, token: str | None = None):
        self.store = store if store is not None else FileStore()
        # Empty token = open access (loopback/dev). LAN deployments MUST
        # set SLO_FILEHUB_TOKEN — the hub binds its own interface.
        self.token = token if token is not None else os.environ.get("SLO_FILEHUB_TOKEN", "")

    # ── auth ───────────────────────────────────────────────────────

    def check_token(self, provided: str | None) -> bool:
        """Constant-time token comparison; empty configured token = open."""
        if not self.token:
            return True
        return hmac.compare_digest((provided or "").encode(), self.token.encode())

    # ── pull-through cache ─────────────────────────────────────────

    def pull(
        self,
        url: str,
        filename: str | None = None,
        tags: list[str] | None = None,
        *,
        quantize: bool = False,
    ) -> dict:
        """Fetch *url* into the store (download once, serve forever).

        Returns the stored metadata with an extra ``deduped`` flag: True
        when an identical blob (same sha256) was already present.

        ``quantize=True`` (opt-in, lossy) routes float safetensors through
        :meth:`quantize_file` — int8 at store time, ~4x smaller. Artifacts
        that are not eligible (GGUF/NF4/GPTQ, non-float tensors) raise
        ``UnsupportedForQuantization``; the caller decides, never silently.
        """
        from downcraft.download.http import download_file

        name = filename or unquote(Path(urlparse(url).path).name)
        if not name or "." not in name.rsplit("/", 1)[-1]:
            raise ValueError("Cannot derive a filename with an extension; pass filename=")

        with tempfile.TemporaryDirectory(prefix="filehub-pull-") as td:
            dest = Path(td) / name
            download_file(url, dest)
            sha = _sha256_file(dest)
            existing = self.store.find_by_sha(sha)
            if existing is not None:
                logger.info("pull dedup: %s == %s", url, existing["file_id"])
                return {**existing, "deduped": True}
            if quantize:
                return self.quantize_file(dest, name, tags)
            with dest.open("rb") as f:
                meta = self.store.put_stream(f, name, tags)
        return {**meta, "deduped": False}

    def quantize_file(self, src_path: Path, filename: str, tags: list[str] | None = None) -> dict:
        """Opt-in quantize-on-ingest (card 019464d8) with quantized-bytes dedup.

        Quantization is deterministic (same input bytes → same int8 bytes),
        so dedup happens *after* the store: if an entry already holds these
        quantized bytes, the fresh one is dropped and the existing returned.
        ``source_sha256`` (the pre-quantization digest) rides in the
        ``quantization`` provenance block — identity for humans, while the
        blob identity stays the bytes actually stored.
        """
        sha = _sha256_file(src_path)
        meta = self.store.put_quantized(src_path, filename, tags)
        quant = {**meta.get("quantization", {}), "source_sha256": sha}
        meta = {**meta, "quantization": quant}
        self.store.update_meta(meta)  # persist source_sha256 provenance
        dup = self.store.find_by_sha(meta["sha256"], exclude_id=meta["file_id"])
        if dup is not None:
            self.store.delete(meta["file_id"])
            logger.info("quantize dedup: %s == %s", filename, dup["file_id"])
            return {**dup, "deduped": True}
        return {**meta, "deduped": False}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
