"""Sidecar slim: Adam moments leave the JSON meta, live in a binary sibling.

A periodic checkpoint's ``<x>.soul.meta.json`` embedded the optimizer's
``state`` (exp_avg / exp_avg_sq) as pretty-printed JSON floats — measured at
~162 MB per file, twice the weights, and ~2.2 s per ``json.loads``. Worse,
the save path walked every float three times (``_make_json_safe`` →
``_soul_json_sanitize`` → ``json.dumps``): one periodic ``save_checkpoint``
measured 16.5 s wall, ~37 % of a training run's time on serialization alone.

Format: ``bin-v1`` — raw little-endian numpy bytes, written next to the
sidecar as ``<x>.optimizer.dat`` (``<x>.meta.json`` minus its suffix),
sha256-pinned by a ``state_ref`` left inline:

    "optimizer": {"hyperparameters": {...},
                  "state_ref": {"file": ..., "sha256": ..., "format": ...}}

File layout (little-endian throughout):

    magic b"SOPT" | u8 version=1 | u32 entry count
    per entry: u16 name_len + name utf-8 | u16 leaf_len + leaf utf-8
               | u8 dtype_str_len + dtype str ("<f8", "<f4", ...)
               | u8 ndim | u64 shape[ndim] | u64 nbytes | data bytes

``leaf`` is empty for flat buffers (``{name: buffer}`` SGD style) and the
per-moment key (``m`` / ``v`` / ``exp_avg`` ...) otherwise, so the decoded
dict mirrors the original state structure exactly.

The .soul header embeds the same bytes as base64 manifests via
``to_state_manifest`` (``{"__blob__": ..., "dtype": ..., "shape": ...}``) —
moments are bytes end-to-end: no ``tolist()`` of millions of floats, no
40 MB of JSON float text per save. Resume from the embedded header stays
self-contained (``_load_soul_checkpoint`` has no sidecar base_dir).

WAL order (mirrors the save paths): the state file lands durably first; the
sidecar that references it is rename-atomic after. A crash between leaves a
harmless orphan, never a ref pointing at nothing.

Why .dat and not .npz / .bin: ``*.npz`` in a model dir is a live token —
``_candidates_newest_first`` treats it as a resume candidate and
``artifact_registry`` registers it as a checkpoint artifact; ``*.bin`` is
globbed as HF weight files (``model_protector`` / ``model_size`` / ``hf_hub``).
``*.dat`` matches no glob in the tree.

Legacy: pre-binary sidecars keep loading — ``load_state_file`` dispatches on
``ref["format"]`` and still reads ``gzip-json-v1``. Exotic (ragged or
non-numeric) state that the binary codec cannot express falls back to the
gzip-JSON write path, so a save never fails on serialization.

Stdlib-only at import time: loaded standalone by
``scripts/migrate_soul_meta_optimizer.py`` via importlib (no package
``__init__`` chain, no domain deps); numpy is imported lazily inside the
encode/decode functions.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

FORMAT = "gzip-json-v1"  # legacy — still read, still the exotic-state fallback
FORMAT_BIN = "bin-v1"
META_SUFFIX = ".meta.json"
STATE_SUFFIX = ".optimizer.json.gz"  # legacy sibling name
STATE_SUFFIX_BIN = ".optimizer.dat"

_MAGIC = b"SOPT"
_VERSION = 1
_BLOB_KEY = "__blob__"


def state_file_for(meta_path: str | os.PathLike[str], *, format: str = FORMAT_BIN) -> Path:
    """``<x>.meta.json`` → sibling state file (``.optimizer.dat`` by default)."""
    p = Path(meta_path)
    name = p.name
    if name.endswith(META_SUFFIX):
        name = name[: -len(META_SUFFIX)]
    suffix = STATE_SUFFIX_BIN if format == FORMAT_BIN else STATE_SUFFIX
    return p.with_name(name + suffix)


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


# ── leaf ↔ array ────────────────────────────────────────────────────────────


def _leaf_to_array(value: Any):
    """Numeric leaf → C-contiguous little-endian ndarray, else None.

    Accepts ndarrays, rectangular numeric lists/tuples, and ``bin-v1``
    manifests (header-embedded form). Everything else — scalars, strings,
    ragged/object containers — returns None so callers can pass it through
    or fall back to the gzip-JSON path.
    """
    import numpy as np  # lazy: module import stays stdlib (migrate script)

    if isinstance(value, dict) and _BLOB_KEY in value:
        try:
            blob = base64.b64decode(value[_BLOB_KEY], validate=True)
            arr = np.frombuffer(blob, dtype=np.dtype(value["dtype"]))
            return arr.reshape(tuple(value["shape"]))
        except Exception:  # noqa: BLE001 — corrupt manifest is simply not encodable
            return None
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value)
        except (ValueError, TypeError):
            return None
    else:
        return None
    if arr.dtype.kind not in "fiub":
        return None
    if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and sys.byteorder == "big"):
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def to_state_manifest(state: Any) -> Any:
    """Encode optimizer ``state`` leaves as base64 manifests for JSON embedding.

    Applied to ``optimizer.state`` only (hyperparameters keep their plain
    lists). Already-encoded manifests pass through unchanged (idempotent);
    scalars and strings pass through for the caller's JSON sanitizer to
    handle. Recurses dicts, never walks float-by-float.
    """
    if isinstance(state, dict):
        if _BLOB_KEY in state:
            return state
        return {k: to_state_manifest(v) for k, v in state.items()}
    arr = _leaf_to_array(state)
    if arr is None:
        return state
    return {
        _BLOB_KEY: base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
    }


def from_state_manifest(state: Any) -> Any:
    """Inverse of ``to_state_manifest``: manifests → ndarrays (identity elsewhere).

    Returns ``None`` when any manifest is corrupt — callers degrade to
    fresh-momentum resume rather than raising (same semantics as an
    unresolvable ``state_ref``).
    """
    import numpy as np  # lazy: module import stays stdlib (migrate script)

    def _dec(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, dict) and _BLOB_KEY in v:
                    blob = base64.b64decode(v[_BLOB_KEY], validate=True)
                    arr = np.frombuffer(blob, dtype=np.dtype(v["dtype"]))
                    out[k] = arr.reshape(tuple(v["shape"])).copy()
                else:
                    out[k] = _dec(v)
            return out
        return obj

    try:
        return _dec(state)
    except Exception:  # noqa: BLE001 — corruption degrades, never raises
        return None


# ── bin-v1 file codec ───────────────────────────────────────────────────────


def encode_state_file(state: Any) -> bytes | None:
    """Serialize ``state`` to ``bin-v1`` bytes; None when not encodable.

    Encodable = a dict of ``{name: {leaf: numeric} | numeric}`` (≤ 2 levels,
    rectangular numeric leaves). Anything else returns None so the caller
    can fall back to gzip-JSON — a save never fails on serialization.
    """
    if not isinstance(state, dict) or not state:
        return None
    entries: list[tuple[str, str, Any]] = []
    for name, entry in state.items():
        if isinstance(entry, dict):
            for leaf, value in entry.items():
                if isinstance(value, dict) and _BLOB_KEY not in value:
                    return None  # >2 nesting levels: exotic, fall back
                arr = _leaf_to_array(value)
                if arr is None:
                    return None
                entries.append((str(name), str(leaf), arr))
        else:
            arr = _leaf_to_array(entry)
            if arr is None:
                return None
            entries.append((str(name), "", arr))
    if not entries:
        return None

    out = bytearray(_MAGIC)
    out += struct.pack("<BI", _VERSION, len(entries))
    for name, leaf, arr in entries:
        nb = name.encode("utf-8")
        lb = leaf.encode("utf-8")
        dt = arr.dtype.str.encode("ascii")
        if len(nb) > 0xFFFF or len(lb) > 0xFFFF or len(dt) > 0xFF:
            return None
        out += struct.pack("<H", len(nb)) + nb
        out += struct.pack("<H", len(lb)) + lb
        out += struct.pack("<B", len(dt)) + dt
        out += struct.pack("<B", arr.ndim)
        if arr.ndim:
            out += struct.pack(f"<{arr.ndim}Q", *arr.shape)
        data = arr.tobytes()
        out += struct.pack("<Q", len(data)) + data
    return bytes(out)


def decode_state_file_bytes(blob: bytes) -> dict | None:
    """Parse ``bin-v1`` bytes → state dict of writable ndarrays, else None."""
    import numpy as np  # lazy: module import stays stdlib (migrate script)

    try:
        if blob[:4] != _MAGIC:
            return None
        off = 4
        version = blob[off]
        off += 1
        if version != _VERSION:
            return None
        (count,) = struct.unpack_from("<I", blob, off)
        off += 4
        view = memoryview(blob)
        state: dict = {}
        for _ in range(count):
            (nlen,) = struct.unpack_from("<H", blob, off)
            off += 2
            name = bytes(view[off : off + nlen]).decode("utf-8")
            off += nlen
            (llen,) = struct.unpack_from("<H", blob, off)
            off += 2
            leaf = bytes(view[off : off + llen]).decode("utf-8")
            off += llen
            (dlen,) = struct.unpack_from("<B", blob, off)
            off += 1
            dtype = np.dtype(bytes(view[off : off + dlen]).decode("ascii"))
            off += dlen
            (ndim,) = struct.unpack_from("<B", blob, off)
            off += 1
            shape = struct.unpack_from(f"<{ndim}Q", blob, off) if ndim else ()
            off += 8 * ndim
            (nbytes,) = struct.unpack_from("<Q", blob, off)
            off += 8
            arr = np.frombuffer(view[off : off + nbytes], dtype=dtype).reshape(shape)
            off += nbytes
            # writable copy: the optimizer mutates moments in place on resume
            arr = arr.copy()
            if leaf:
                state.setdefault(name, {})[leaf] = arr
            else:
                state[name] = arr
        return state if state else None
    except Exception:  # noqa: BLE001 — malformed payload degrades, never raises
        return None


# ── sidecar externalization / loading ───────────────────────────────────────


def externalize_optimizer_state(
    doc: dict, meta_path: str | os.PathLike[str], *, write: bool = True
) -> tuple[dict, bool]:
    """Return ``(sidecar_doc, moved?)`` with ``optimizer.state`` moved out.

    No-op returns the input doc untouched when there is nothing to move
    (already slim, weights-only save, or no optimizer at all). When ``write``
    is true the state sibling is written and fsynced before returning, so the
    caller can never publish a ``state_ref`` to a file that does not exist.

    Encodable state goes to ``bin-v1`` raw bytes; exotic (ragged /
    non-numeric) state falls back to the legacy gzip-JSON codec — a save
    never fails because of serialization.
    """
    opt = _optimizer_of(doc)
    if opt is None:
        return doc, False
    state = opt.get("state")
    if not isinstance(state, dict) or not state:
        return doc, False

    blob = encode_state_file(state)
    fmt = FORMAT_BIN
    if blob is None:
        payload = json.dumps(
            {"state": state}, separators=(",", ":"), default=str, allow_nan=False
        ).encode("utf-8")
        blob = gzip.compress(payload, compresslevel=6)
        fmt = FORMAT
    sha = hashlib.sha256(blob).hexdigest()

    out = state_file_for(meta_path, format=fmt)
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
        "format": fmt,
    }
    return slim, True


def load_state_file(ref: Any, base_dir: str | os.PathLike[str]) -> dict | None:
    """Resolve a ``state_ref`` to the state dict, hash-pinned.

    ``bin-v1`` decodes to numpy arrays; legacy ``gzip-json-v1`` yields the
    original JSON lists. Returns ``None`` for anything unresolvable (missing
    file, unknown format, hash mismatch, malformed payload) — callers
    degrade to fresh-momentum resume, which is the pre-existing semantics
    for a stripped state.
    """
    if not isinstance(ref, dict):
        return None
    fmt = ref.get("format")
    if fmt not in (FORMAT, FORMAT_BIN):
        return None
    file = ref.get("file")
    if not isinstance(file, str):
        return None
    try:
        blob = (Path(base_dir) / file).read_bytes()
        if hashlib.sha256(blob).hexdigest() != ref.get("sha256"):
            return None
        if fmt == FORMAT_BIN:
            state = decode_state_file_bytes(blob)
        else:
            doc = json.loads(gzip.decompress(blob))
            state = doc.get("state") if isinstance(doc, dict) else None
    except Exception:  # noqa: BLE001 — degradation, never a hard failure
        return None
    return state if isinstance(state, dict) and state else None
