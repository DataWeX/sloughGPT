"""Content-addressed token cache for ``prepare_data``.

Training re-tokenizes the full corpus on every run — the single largest
fixed cost before step 0 for any data_path that resolves to a local file
(char loop over every character, or BPE over every word). This module
caches the resulting token-id array so a warm start opens a memmap instead
of re-reading and re-encoding.

Layout (under ``runs/tokencache/``, gitignored):

    <src_sha16>_<mode>_<tok_fp8>.bin    uint16/uint32 token ids (memmap)
    <src_sha16>_<mode>_<tok_fp8>.meta.json

Cache key is content-addressed — two independent axes that invalidate:

* ``src_sha16``  — first 16 hex of sha256 over the source file bytes,
  computed **chunked before** any full-text read (an edited corpus is a
  different key; no stale-token risk).
* ``tok_fp8``    — first 8 hex of sha256 over the tokenizer's sorted
  ``stoi`` mapping (mode="bpe"), or the constant ``nochar`` (mode="char",
  vocab is derived from the text itself and is fully described by
  src_sha16). Swapping/retraining the tokenizer is a different key.

dtype: uint16 when vocab <= 65535, else uint32 — halves (or quarters) the
on-disk and resident footprint vs the int64 the trainer used to carry.
``get_batch`` casts each sliced batch back to int64 at the seam (the only
place the model sees ids), so the compact dtype never leaks into math.

Writes are atomic: tmp file + ``os.replace`` for both .bin and .meta.json;
a half-written cache is never visible under the final name, and a crash
mid-write leaves only an inert ``*.tmp`` for the next run to overwrite.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path("runs") / "tokencache"

_CHUNK = 1 << 20  # 1 MiB — hash and write in chunks, never hold full file
_MODE_CHAR = "char"
_MODE_BPE = "bpe"
_FP_NOCHAR = "nochar"


def file_sha256(path: Path) -> str:
    """Streaming sha256 of *path*'s bytes — decided before any text read."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Stable fingerprint of a tokenizer's mapping state.

    Hashes the sorted ``(token, id)`` pairs of ``stoi`` — retraining,
    resizing, or swapping the tokenizer changes the fingerprint (and
    therefore the cache key). Falls back to ``str(tokenizer)`` when the
    object exposes no ``stoi`` (should not happen for SloBPE-compatible
    tokenizers, but keeps the hook total).
    """
    stoi = getattr(tokenizer, "stoi", None)
    if isinstance(stoi, dict) and stoi:
        payload = json.dumps(sorted((str(k), int(v)) for k, v in stoi.items()))
    else:
        payload = repr(tokenizer)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_paths(
    source_path: Path,
    mode: str,
    tok_fp: str,
    cache_root: Path | None = None,
) -> tuple[Path, Path]:
    """(bin, meta) paths for one cache key. ``tok_fp`` is the full hex;
    the filename keeps its first 8 chars for readability."""
    root = Path(cache_root) if cache_root else DEFAULT_CACHE_ROOT
    src_fp = file_sha256(source_path)[:16]
    stem = f"{src_fp}_{mode}_{tok_fp[:8]}"
    return root / f"{stem}.bin", root / f"{stem}.meta.json"


def _dtype_for(vocab: int) -> np.dtype:
    return np.dtype(np.uint16 if vocab <= 65535 else np.uint32)


def load_cache(
    bin_path: Path,
    meta_path: Path,
) -> tuple[np.ndarray, int, dict[str, int], dict[int, str]] | None:
    """Open a cached token array, or ``None`` on any miss/mismatch.

    Returns ``(data, vocab_size, stoi, itos)`` mirroring ``prepare_data``.
    ``data`` is a read-only memmap in the compact dtype (uint16/uint32) —
    ``get_batch`` casts slices to int64 at the seam. Every guard is a
    miss, never an exception: corrupt/partial files behave as a cold start.
    """
    if not bin_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dtype = np.dtype(meta["dtype"])
        n = int(meta["n_tokens"])
        vocab = int(meta["vocab"])
        if bin_path.stat().st_size != n * dtype.itemsize:
            logger.warning("token cache size mismatch, ignoring: %s", bin_path)
            return None
        data = np.memmap(bin_path, dtype=dtype, mode="r", shape=(n,))
        stoi_raw = meta.get("stoi") or {}
        stoi: dict[str, int] = {k: int(v) for k, v in stoi_raw.items()}
        itos: dict[int, str] = {int(k): v for k, v in (meta.get("itos") or {}).items()}
        return data, vocab, stoi, itos
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("token cache unreadable, ignoring %s: %s", meta_path, exc)
        return None


def store_cache(
    source_path: Path,
    data: np.ndarray,
    vocab: int,
    stoi: dict[str, int],
    itos: dict[int, str],
    mode: str,
    tokenizer: Any = None,
    cache_root: Path | None = None,
) -> tuple[Path, Path] | None:
    """Atomically persist *data* under the content-addressed key.

    Returns ``(bin, meta)`` on success, ``None`` on failure (a failed cache
    write must never break training — the caller just runs cold next time).
    ``stoi``/``itos`` are recorded only for char mode; bpe mode reloads them
    from the tokenizer object, which is already loaded on a hit path.
    """
    try:
        tok_fp = (
            _FP_NOCHAR
            if mode == _MODE_CHAR
            else (tokenizer_fingerprint(tokenizer) if tokenizer is not None else _FP_NOCHAR)
        )
        bin_path, meta_path = cache_paths(source_path, mode, tok_fp, cache_root)
        bin_path.parent.mkdir(parents=True, exist_ok=True)

        dtype = _dtype_for(vocab)
        compact = np.asarray(data, dtype=dtype)

        tmp_bin = bin_path.with_suffix(bin_path.suffix + ".tmp")
        compact.tofile(tmp_bin)
        os.replace(tmp_bin, bin_path)

        meta: dict[str, Any] = {
            "dtype": dtype.name,
            "n_tokens": int(compact.shape[0]),
            "vocab": int(vocab),
            "mode": mode,
            "src_sha256": file_sha256(source_path),
            "tokenizer_fp": tok_fp,
        }
        if mode == _MODE_CHAR:
            meta["stoi"] = {str(k): int(v) for k, v in stoi.items()}
            meta["itos"] = {str(k): str(v) for k, v in itos.items()}

        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp_meta, meta_path)
        return bin_path, meta_path
    except (OSError, ValueError) as exc:
        logger.warning("token cache write failed (continuing cold): %s", exc)
        return None


def try_load(
    source_path: Path,
    mode: str,
    tokenizer: Any = None,
    cache_root: Path | None = None,
) -> tuple[np.ndarray, int, dict[str, int], dict[int, str]] | None:
    """Lookup by content key; ``None`` on miss (absent, stale, or corrupt)."""
    tok_fp = (
        _FP_NOCHAR
        if mode == _MODE_CHAR
        else (tokenizer_fingerprint(tokenizer) if tokenizer is not None else _FP_NOCHAR)
    )
    bin_path, meta_path = cache_paths(source_path, mode, tok_fp, cache_root)
    hit = load_cache(bin_path, meta_path)
    if hit is not None:
        logger.info(
            "token cache hit: %s (%d tokens, vocab %d)",
            bin_path.name,
            hit[0].shape[0],
            hit[1],
            extra={"tag": "TRAIN"},
        )
    return hit
