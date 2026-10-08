"""Checkpoint operations — load, list, scan, describe, download."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import struct
import time
from pathlib import Path
from typing import Any

from domain.inference._internal.slo_format import (
    SOUL_TIER_POLICY,
    read_sidecar,
    soul_read_candidates,
)
from domain.shared import is_valid_iso, repair_iso

from .helpers import (
    describe_checkpoint,
    read_slo_json_header,
)
from .state import CHECKPOINTS_DIR, LORA_DIR, REPO_ROOT, TURBO_DIR, VALID_CKPT_NAME

# Final job saves: trainer.save(f"models/<stem>_trained.soul") — a third root,
# outside auto-training/turbo-trained, and the exact path job records point at.
TRAINED_DIR = REPO_ROOT / "models"


# Every root a checkpoint may live in — declared ONCE, read at call time.
#
# Five operations need this set: the list that reports what exists, plus the
# lookup, resolution, download and delete that act on it. They used to carry
# five hand-written tuples, and LORA_DIR (data/user_adapters) sat in the list
# but not in load_soul — so a checkpoint could be listed, even downloaded,
# and still 404 the instant it was opened. One declaration makes that drift
# impossible rather than merely unlikely.
#
# A function, not a tuple: a module-level tuple would bind the Paths at import
# time, so patching these globals afterwards (which is how test isolation
# redirects the whole module — see _isolated_ckpt_dirs in test_checkpoints.py)
# would leave this set pointing at the real repo tree.
def ckpt_roots() -> tuple[Path, ...]:
    return (CHECKPOINTS_DIR, TURBO_DIR, LORA_DIR, TRAINED_DIR)


logger = logging.getLogger("slo.training")

# The two spellings a final job save may legitimately carry. The legacy
# double-append is not hypothetical — four such checkpoints (~100MB) sit in
# models/ right now.
TRAINED_SUFFIXES = ("_trained.soul", "_trained.soul.soul")


def is_trained_checkpoint(name: str) -> bool:
    """True when *name* is a final job save, under either legal spelling.

    ``models/`` holds far more than final saves (benchmarks, ad-hoc exports,
    native runs), so list, delete and download all narrow it to ``*_trained``.
    That decision used to be three separate ``endswith("_trained.soul")``
    checks, and every one of them rejected ``<stem>_trained.soul.soul`` —
    leaving four checkpoints invisible to the list, refused by download and
    undeletable, even though :func:`domain.inference.load_soul` reads them
    without complaint. One predicate owns the decision now.

    Args:
        name: bare file name; directory components are not inspected.

    Returns:
        Whether this is a final job save.
    """
    return name.endswith(TRAINED_SUFFIXES)


def _probe_paths(base: Path, name: str) -> list[Path]:
    """Every file *name* could mean inside *base*, in probe order.

    The ordering — the spelling the caller NAMED first, then its sibling —
    belongs to :func:`domain.inference._internal.slo_format.soul_read_candidates`,
    the single owner of .soul filename grammar. This only walks that list
    across one root, so callers keep their root-major search order.

    Before this, ``find_checkpoint`` and ``load_soul`` each hand-appended
    ``.soul``/``.slo`` themselves and never probed the legacy double-append
    sibling: a canonical name for a file that exists only at ``x.soul.soul``
    came back not-found while ``domain.inference.load_soul`` read it fine.
    Two owners, two answers.
    """
    return [Path(c) for c in soul_read_candidates(str(base / name))]


def _in_root(resolved: Path, base: Path) -> bool:
    """*resolved* is a direct child of search root *base*.

    Direct-child, never a string prefix: the roots are flat and ``models/``
    literally contains ``models/auto-training/``, so a file has to answer to
    exactly one root's rules — ``models/`` lists only ``*_trained`` while
    auto-training lists everything. ``str.startswith`` cannot say which root
    owns a nested file, and it counts ``<root>-evil/…`` as inside.
    """
    try:
        return resolved.parent == base.resolve()
    except OSError:  # a root that vanished must not take a lookup down with it
        return False


def _owning_root(resolved: Path) -> Path | None:
    """The one root *resolved* belongs to, or None when it belongs to none."""
    return next((b for b in ckpt_roots() if _in_root(resolved, b)), None)


def _resolve_ref_path(path: str) -> Path | None:
    """Resolve the ``path`` a listing handed back, confined to one root.

    Absolute or repo-relative both land here; being a direct child of a
    search root is the whole check. ``None`` means "not a checkpoint we may
    touch", and callers must NOT fall back to name order on it — falling
    back would act on a different file than the row the user pointed at,
    which is the very bug this parameter exists to prevent.
    """
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    try:
        resolved = p.resolve()
    except OSError:
        return None
    return resolved if _owning_root(resolved) is not None else None


def _path_key(fp: Path) -> str:
    """The address a row carries, so two same-named rows stay distinguishable.

    Repo-relative keeps the server's absolute layout out of the payload; the
    tmp roots tests redirect live outside the repo, so those fall back to an
    absolute path that still resolves back to the same file.
    """
    try:
        return fp.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return fp.resolve().as_posix()


def find_checkpoint(name: str, path: str | None = None) -> Path | None:
    # Callers hand back either a bare file name or the job record's
    # "models/<stem>_trained.soul" path style. Every search root is a flat
    # directory, so stripping directory components is safe — and it is what
    # keeps `{name}` route segments and traversal attempts working.
    name = Path(name).name
    if not name:
        return None
    if path is not None:
        # The row's own address wins: two roots may hold this exact name, so
        # the name alone cannot say which file the caller means.
        resolved = _resolve_ref_path(path)
        return resolved if resolved is not None and resolved.exists() else None
    for base in ckpt_roots():
        for candidate in _probe_paths(base, name):
            # resolve() first: a symlink, or a `..` that lands outside the
            # root, fails the direct-child check instead of being handed back.
            resolved = candidate.resolve()
            if resolved.exists() and _in_root(resolved, base):
                return resolved
    return None


def load_soul(name: str, path: str | None = None) -> dict | None:
    if path is not None:
        resolved = _resolve_ref_path(path)
        batches = [[resolved]] if resolved is not None else []
    else:
        batches = [_probe_paths(d, name) for d in ckpt_roots()]
    for candidates in batches:
        for fp in candidates:
            if not fp.exists():
                continue
            try:
                st = fp.stat()
                if fp.suffix == ".soul" and st.st_size < 4096:
                    continue
                return _load_soul_from_path(fp, st)
            except Exception as exc:
                logger.debug("Failed to load checkpoint %s: %s", fp.name, exc)
                continue
    return None


def _load_soul_from_path(fp: Path, st=None) -> dict | None:
    try:
        if st is None:
            st = fp.stat()
        size_mb = round(st.st_size / (1024 * 1024), 2)

        # The shared sidecar reader, not a bare json.load. The difference is
        # 12.2s over one models/ directory: `metadata.training_state` (resume
        # state, read by train_pipeline and nowhere in this row) reaches 40MB,
        # and every checkpoint row was paying to parse it.
        meta = read_sidecar(str(fp)) or None

        if meta is None and fp.suffix == ".soul":
            meta = read_slo_json_header(fp)

        if meta is None and fp.suffix == ".slo":
            try:
                from domain.inference._internal.slo_format import SouParser

                profile = SouParser.parse(fp.read_text(encoding="utf-8"))
                meta = {
                    "soul_name": profile.name,
                    "tagline": profile.tagline,
                    "description": profile.description,
                    "born_at": profile.born_at,
                    "lineage": profile.lineage,
                    "base_model": profile.base_model,
                    "training_dataset": profile.training_dataset,
                    "final_train_loss": profile.final_train_loss,
                    "system_prompt": profile.system_prompt,
                    "tags": profile.tags,
                    "epochs_trained": profile.epochs_trained,
                    "personality_traits": dict(profile.personality.to_dict().items()),
                    "metadata": dict(profile.metadata),
                }
            except Exception:
                logger.debug("Failed to serialize profile for %s", fp, exc_info=True)

        if meta:
            m = meta.get("metadata", {})
            raw_soul = meta.get("soul_name") or meta.get("soul") or meta.get("name") or "unknown"
            soul = raw_soul.replace("-soul", "")
            if fp.suffix == ".soul" and (soul == fp.stem or soul == fp.name):
                soul = "unknown"
            # The two declared identity axes. Both are stamped by save_soul at
            # write time — the only moment they are knowable — so an absent
            # value means the file predates the field and is reported unknown.
            # tier alone falls back to the suffix policy, because container
            # tier really is a property of the spelling; provenance never
            # falls back, since no name can tell you who produced a file.
            provenance = str(meta.get("provenance") or "")
            tier = str(meta.get("tier") or "") or SOUL_TIER_POLICY.get(fp.suffix, "")
            row = {
                "name": fp.name,
                # The address this row IS. Two roots may hold one filename, so
                # `name` alone cannot tell two rows apart — actions are keyed
                # by this, and React lists key on it for the same reason.
                "path": _path_key(fp),
                "soul": soul,
                # Content-derived unique key. Every sidecar already carries it;
                # nothing read it until now, which is why two DIFFERENT
                # checkpoints could share one name and look identical in a list.
                "integrity_hash": str(meta.get("integrity_hash") or ""),
                "tier": tier,
                "provenance": provenance,
                "loss": m.get("avg_loss") or meta.get("final_train_loss"),
                "steps": m.get("steps", 0),
                "epochs": m.get("step") or meta.get("epochs_trained", 0),
                "traits": meta.get("personality_traits", meta.get("traits", {})),
                "lineage": meta.get("lineage", "slonet"),
                "model_type": meta.get("model_type", "slonet"),
                "size_mb": size_mb,
                "tokenizer_type": m.get("tokenizer_type", "char"),
                "vocab_size": m.get("vocab_size") or meta.get("vocab_size", 0),
                "avg_quality": meta.get("avg_quality"),
                "created_at": meta.get("created_at", ""),
                "model_path": str(fp),
                "source": "auto-train",
                **{
                    k: meta[k]
                    for k in (
                        "tagline",
                        "description",
                        "born_at",
                        "epochs_trained",
                        "final_train_loss",
                        "final_val_loss",
                        "system_prompt",
                        "tags",
                        "base_model",
                        "training_dataset",
                        "personality",
                        "training_duration_s",
                    )
                    if k in meta and meta[k]
                },
            }
            # Repair legacy timestamps (e.g. "...+00:00Z") so the UI never
            # renders "Invalid Date"; values that stay unparseable are dropped.
            for _key in ("born_at", "created_at"):
                if _key in row:
                    row[_key] = repair_iso(row[_key])
                    if not is_valid_iso(row[_key]):
                        del row[_key]
            return row

        # No sidecar: the file still EXISTS, so it stays inspectable. Carry the
        # path so a caller can classify identity from the bytes themselves —
        # "unnamed" is not "absent".
        return {
            "name": fp.name,
            "path": _path_key(fp),
            "soul": "unknown",
            "size_mb": size_mb,
            "model_path": str(fp),
        }
    except Exception as e:
        logger.debug("Failed to read soul header %s: %s", fp.name, e)
        return None


def load_lora_soul(name: str) -> dict | None:
    for ext in (".soul", ".slo"):
        fp = LORA_DIR / name if name.endswith((ext,)) else LORA_DIR / (name + ext)
        if fp.exists():
            try:
                st = fp.stat()
                return _load_soul_from_path(fp, st)
            except Exception as exc:
                logger.debug("Failed to load LoRA soul %s: %s", fp.name, exc)
                continue
    return None


def _scan_all_checkpoints() -> list[dict]:
    checkpoints = []
    # Keyed by path, not by name: two roots may hold one filename, and
    # name-keyed dedupe hid the second one from the list while name-based
    # delete still walked every root — a file the user could not see being
    # deleted alongside the row they clicked. Rows carry their own `path`
    # so both are listable and individually addressable.
    seen = set()

    def _stat_key(p: Path):
        try:
            return p.stat().st_mtime, p
        except OSError:
            return (0, p)

    for ext in ("*.soul", "*.slo"):
        for f in sorted(CHECKPOINTS_DIR.glob(ext), key=_stat_key, reverse=True):
            if str(f) in seen:
                continue
            seen.add(str(f))
            try:
                st = f.stat()
            except OSError:
                continue
            if f.suffix == ".soul" and st.st_size < 4096:
                continue
            info = _load_soul_from_path(f, st)
            if info:
                checkpoints.append(info)

    for f in sorted(TURBO_DIR.glob("*.soul"), key=_stat_key, reverse=True):
        if str(f) in seen:
            continue
        seen.add(str(f))
        try:
            st = f.stat()
        except OSError:
            continue
        info = _load_soul_from_path(f, st)
        if info:
            info["source"] = "turbo"
            checkpoints.append(info)

    # Final job saves (models/<stem>_trained.soul) — what job records and the
    # results card hand back for "Load for chat". The predicate rather than a
    # literal "*_trained.soul" glob: the legacy double-append spelling must not
    # be silently skipped.
    trained_saves = [p for p in TRAINED_DIR.glob("*.soul") if is_trained_checkpoint(p.name)]
    for f in sorted(trained_saves, key=_stat_key, reverse=True):
        if str(f) in seen:
            continue
        seen.add(str(f))
        try:
            st = f.stat()
        except OSError:
            continue
        if st.st_size < 4096:
            continue
        info = _load_soul_from_path(f, st)
        if info:
            info["source"] = "trained"
            checkpoints.append(info)

    for npz in sorted(LORA_DIR.glob("*.soul"), key=_stat_key, reverse=True):
        if str(npz) not in seen:
            seen.add(str(npz))
            try:
                st = npz.stat()
            except OSError:
                continue
            info = _load_soul_from_path(npz, st)
            if info:
                info["source"] = "lora"
                checkpoints.append(info)

    for ckpt in checkpoints:
        ckpt["description"] = describe_checkpoint(ckpt)

    return checkpoints


async def list_checkpoints() -> list[dict]:
    return await asyncio.to_thread(_scan_all_checkpoints)


async def delete_checkpoint(name: str, path: str | None = None) -> list[str]:
    if not re.match(r"^[\w\-]+(\.\w+)*$", name):
        raise ValueError(f"Invalid checkpoint name: {name!r}")

    deleted = []

    def _unlink(resolved: Path) -> bool:
        root = _owning_root(resolved)
        # models/ root holds more than final saves — only ever delete what
        # the scan lists from it.
        if root is TRAINED_DIR and not is_trained_checkpoint(resolved.name):
            return False
        resolved.unlink()
        meta = Path(str(resolved) + ".meta.json")
        if meta.exists():
            meta.unlink()
        return True

    def _delete():
        if path is not None:
            # Exactly one file — the one the caller pointed at. No root sweep:
            # with a path in hand, walking every root would unlink same-named
            # siblings the user never selected.
            resolved = _resolve_ref_path(path)
            if resolved is not None and resolved.exists() and _unlink(resolved):
                deleted.append(resolved.name)
            return
        for base in ckpt_roots():
            for ext in (".soul", ".slo"):
                if name.endswith(ext):
                    candidates = [base / name]
                else:
                    candidates = [base / (name + ext)]
                for candidate in candidates:
                    resolved = candidate.resolve()
                    if resolved.exists() and _in_root(resolved, base) and _unlink(resolved):
                        deleted.append(candidate.name)

    await asyncio.to_thread(_delete)
    return deleted


async def build_checkpoint_provider(name: str, path: str | None = None) -> tuple[Any, dict]:
    """Load a checkpoint's weights into a provider WITHOUT registering it.

    Registration decides which model the server serves, so inspection and
    compare paths must never touch it.
    """
    from domain.models._internal.provider import SloTransformerProvider
    from domain.training._internal.slonet import import_from_sou

    cp = await asyncio.to_thread(find_checkpoint, name, path)
    if cp is None:
        raise FileNotFoundError(f"Checkpoint not found: {name}")

    def _load_meta():
        soul_net = import_from_sou(str(cp))
        with open(str(cp), "rb") as f:
            raw = f.read(12)
            json_len = struct.unpack("<I", raw[8:12])[0]
            meta_bytes = f.read(json_len).rstrip(b"\x00")
        md = json.loads(meta_bytes.decode())
        return soul_net, md

    soul_net, md = await asyncio.to_thread(_load_meta)
    soul_meta = soul_net.soul_signature()

    stoi = md.get("stoi") or md.get("metadata", {}).get("stoi")
    itos = md.get("itos") or md.get("metadata", {}).get("itos")
    if stoi is None or itos is None:
        raise ValueError("Checkpoint has no stoi/itos vocab - retrain to include vocab.")

    provider = SloTransformerProvider(
        model=soul_net,
        stoi=stoi,
        itos=itos,
        model_id_str=cp.stem,
    )

    logger.info(
        "Loaded checkpoint %s (vocab=%d, params=%d)", cp.name, len(stoi), soul_net.num_parameters()
    )

    info = {
        "name": cp.name,
        "soul": soul_meta.get("soul_name", soul_net.soul_name),
        "loss": md.get("final_train_loss"),
        "steps": md.get("total_steps", 0),
        "traits": soul_meta.get("soul_traits", {}),
        "lineage": soul_net.lineage,
        "vocab_size": len(stoi),
        "params": soul_net.num_parameters(),
        "provider": "slonet",
    }
    return provider, info


async def load_checkpoint(name: str, path: str | None = None) -> dict:
    from domain.models._internal.provider import register_provider

    provider, info = await build_checkpoint_provider(name, path)
    register_provider("slonet", provider)
    register_provider("default", provider)
    return info


# Identical sampling for both sides of a comparison — differing params would
# make the A/B a test of the sampler, not of the checkpoints.
_COMPARE_SAMPLING = {"temperature": 0.7, "top_p": 0.85, "top_k": 40, "repetition_penalty": 1.15}


async def compare_checkpoints(
    name_a: str, name_b: str, prompt: str, max_new_tokens: int = 128
) -> dict:
    """Same prompt against two checkpoints, side by side.

    Neither checkpoint is registered: the served model is untouched and each
    provider is released as soon as its answer is in.
    """
    results: dict[str, dict] = {}
    for key, name in (("a", name_a), ("b", name_b)):
        provider, info = await build_checkpoint_provider(name)
        try:
            text = await provider.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                **_COMPARE_SAMPLING,
            )
            results[key] = {"name": info["name"], "text": text}
        finally:
            del provider
    return results


async def download_checkpoint_path(name: str, path: str | None = None) -> str | None:
    if not VALID_CKPT_NAME.match(name) or ".." in name:
        raise ValueError("Invalid checkpoint name")

    def _find():
        if path is not None:
            # The row's address, so the file served is the file shown.
            resolved = _resolve_ref_path(path)
            if resolved is None or not resolved.exists():
                return None
            if _owning_root(resolved) is TRAINED_DIR and not is_trained_checkpoint(resolved.name):
                return None
            return str(resolved) if resolved.suffix in (".soul", ".slo") else None
        for d in ckpt_roots():
            fp = (d / name).resolve()
            if d is TRAINED_DIR and not is_trained_checkpoint(fp.name):
                continue
            if fp.exists() and fp.suffix in (".soul", ".slo") and _in_root(fp, d):
                return str(fp)
        return None

    return await asyncio.to_thread(_find)


def _describe_identity(info: dict) -> dict:
    """Fill identity gaps in a detail row from the bytes themselves.

    Detail view only — one file per request, so the probe is affordable;
    running it per row would undo the listing speed-up.

    Never overwrites what the sidecar already declared: only *missing*
    fields are supplied. That is what lets a stray ``evil.soul`` report
    ``not-soul``, and lets a file whose sidecar predates the identity
    fields still name its container, instead of the dialog omitting the
    Identity block entirely — a silent empty state that reads as "nothing
    to show" when the truth is "not recorded".

    classify_soul never raises, but a failure must not mask the row.
    """
    path = info.get("model_path")
    if not path:
        return info
    try:
        from domain.inference import classify_soul

        ident = classify_soul(path)
    except Exception as e:  # never let identity enrichment mask the row
        logger.debug("classify_soul failed for %s: %s", path, e)
        return info
    for field, value in (
        ("format", ident.format),
        ("tier", ident.tier),
        ("provenance", ident.provenance),
        ("integrity_hash", ident.integrity_hash),
        ("born_at", ident.born_at),
    ):
        if value and not info.get(field):
            info[field] = value
    return info


async def checkpoint_info(name: str, path: str | None = None) -> dict:
    if not VALID_CKPT_NAME.match(name) or ".." in name:
        raise ValueError("Invalid checkpoint name")
    info = await asyncio.to_thread(load_soul, name, path)
    if not info:
        # Absent row — the file is genuinely not there.
        raise FileNotFoundError(f"Checkpoint not found: {name}")
    # Unnamed, legacy or fully declared — all three describe themselves from
    # their bytes. Gated before on soul == "unknown", which meant a file with
    # a stub sidecar got no format at all and the dialog dropped the whole
    # Identity block: a silent empty state where the honest answer was
    # "not recorded". One file per request, so the probe is cheap here.
    return await asyncio.to_thread(_describe_identity, info)


async def get_all_checkpoint_data() -> list[dict]:
    return await asyncio.to_thread(_scan_all_checkpoints)


async def export_all_metrics() -> dict:
    checkpoints = await asyncio.to_thread(_scan_all_checkpoints)
    return {
        "exported_at": time.time(),
        "total_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
    }


async def export_checkpoint_mobile(name: str) -> dict:
    import base64

    import numpy as np

    from domain.training._internal.slonet import import_from_sou

    def _find_ckpt():
        for d in ckpt_roots():
            fp = d / name
            # models/ root holds more than final saves — only export what the
            # list would have offered (same rule delete/download apply).
            if d is TRAINED_DIR and not is_trained_checkpoint(fp.name):
                continue
            if fp.exists() and fp.suffix == ".soul":
                return str(fp)
        return None

    fp_str = await asyncio.to_thread(_find_ckpt)
    if not fp_str:
        raise FileNotFoundError(f"Checkpoint not found: {name}")
    fp = Path(fp_str)

    net = import_from_sou(str(fp))
    sd = net.state_dict()
    n_embed = net.n_embed
    n_layer = net.n_layer
    n_head = net.n_head
    vocab_size = net.vocab_size
    block_size = getattr(net, "block_size", 64)

    weights = []

    def _push(n):
        arr = sd.get(n)
        if arr is not None:
            weights.append(arr.astype(np.float32).ravel())

    _push("tok_emb.weight")
    for i in range(n_layer):
        _push(f"blocks.{i}.attn_norm.weight")
        _push(f"blocks.{i}.attn.q_proj.weight")
        _push(f"blocks.{i}.attn.k_proj.weight")
        _push(f"blocks.{i}.attn.v_proj.weight")
        _push(f"blocks.{i}.attn.o_proj.weight")
        _push(f"blocks.{i}.ff_norm.weight")
        _push(f"blocks.{i}.ff.w1.weight")
        _push(f"blocks.{i}.ff.w2.weight")
        _push(f"blocks.{i}.ff.w3.weight")
    _push("norm.weight")
    _push("lm_head.weight")

    flat = np.concatenate(weights) if weights else np.array([], dtype=np.float32)
    weights_b64 = base64.b64encode(flat.tobytes()).decode()

    return {
        "config": {
            "vocab_size": vocab_size,
            "n_embed": n_embed,
            "n_layer": n_layer,
            "n_head": n_head,
            "block_size": block_size,
            "num_weights": len(weights),
        },
        "weights_b64": weights_b64,
    }
