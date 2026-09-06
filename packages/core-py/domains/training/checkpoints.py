"""Checkpoint operations — load, list, scan, describe, download."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import struct
import time
from pathlib import Path

from .state import CHECKPOINTS_DIR, TURBO_DIR, LORA_DIR, VALID_CKPT_NAME, SOU_MAGIC
from .helpers import (
    _finite_payload,
    read_slo_json_header,
    describe_checkpoint,
)

logger = logging.getLogger("slo.training")


def find_checkpoint(name: str) -> Path | None:
    if name.endswith((".soul", ".slo")):
        for base in (CHECKPOINTS_DIR, TURBO_DIR):
            candidate = (base / name).resolve()
            if candidate.exists() and str(candidate).startswith(str(base.resolve())):
                return candidate
        return None
    for ext in (".soul", ".slo"):
        for base in (CHECKPOINTS_DIR, TURBO_DIR):
            candidate = (base / (name + ext)).resolve()
            if candidate.exists() and str(candidate).startswith(str(base.resolve())):
                return candidate
    return None


def load_soul(name: str) -> dict | None:
    for d in (CHECKPOINTS_DIR, TURBO_DIR):
        for ext in (".soul", ".slo"):
            fp = d / name if name.endswith((ext,)) else d / (name + ext)
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

        meta = None
        meta_file = fp.with_suffix(fp.suffix + ".meta.json")
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                meta = None

        if meta is None and fp.suffix == ".soul":
            meta = read_slo_json_header(fp)

        if meta is None and fp.suffix == ".slo":
            try:
                from domains.inference.slo_format import SouParser
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
                    "personality_traits": {k: v for k, v in profile.personality.to_dict().items()},
                    "metadata": dict(profile.metadata),
                }
            except Exception:
                logger.debug("Failed to serialize profile for %s", fp, exc_info=True)

        if meta:
            m = meta.get("metadata", {})
            raw_soul = (meta.get("soul_name") or meta.get("soul") or meta.get("name") or "unknown")
            soul = raw_soul.replace("-soul", "")
            if fp.suffix == ".soul" and (soul == fp.stem or soul == fp.name):
                soul = "unknown"
            return {
                "name": fp.name,
                "soul": soul,
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
                **{k: meta[k] for k in ("tagline", "description", "born_at", "epochs_trained",
                   "final_train_loss", "final_val_loss", "system_prompt",
                   "tags", "base_model", "training_dataset", "personality",
                   "training_duration_s")
                   if k in meta and meta[k]},
            }

        return {"name": fp.name, "soul": "unknown", "size_mb": size_mb}
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
    seen = set()

    def _stat_key(p: Path):
        try:
            return p.stat().st_mtime, p
        except OSError:
            return (0, p)

    for ext in ("*.soul", "*.slo"):
        for f in sorted(CHECKPOINTS_DIR.glob(ext), key=_stat_key, reverse=True):
            if f.name in seen:
                continue
            seen.add(f.name)
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
        if f.name in seen:
            continue
        seen.add(f.name)
        try:
            st = f.stat()
        except OSError:
            continue
        info = _load_soul_from_path(f, st)
        if info:
            info["source"] = "turbo"
            checkpoints.append(info)

    for npz in sorted(LORA_DIR.glob("*.soul"), key=_stat_key, reverse=True):
        if npz.name not in seen:
            seen.add(npz.name)
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


async def delete_checkpoint(name: str) -> list[str]:
    if not re.match(r'^[\w\-]+(\.\w+)*$', name):
        raise ValueError(f"Invalid checkpoint name: {name!r}")

    deleted = []

    def _delete():
        for base in (CHECKPOINTS_DIR, TURBO_DIR):
            for ext in (".soul", ".slo"):
                if name.endswith(ext):
                    candidates = [base / name]
                else:
                    candidates = [base / (name + ext)]
                for candidate in candidates:
                    resolved = candidate.resolve()
                    if resolved.exists() and str(resolved).startswith(str(base.resolve())):
                        resolved.unlink()
                        deleted.append(candidate.name)
                    meta = Path(str(resolved) + ".meta.json")
                    if meta.exists():
                        meta.unlink()

    await asyncio.to_thread(_delete)
    return deleted


async def load_checkpoint(name: str) -> dict:
    from domains.training.slonet import import_from_sou
    from domains.models.provider import SloTransformerProvider, register_provider

    cp = await asyncio.to_thread(find_checkpoint, name)
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
    register_provider("slonet", provider)
    register_provider("default", provider)

    logger.info("Loaded checkpoint %s (vocab=%d, params=%d)", cp.name, len(stoi), soul_net.num_parameters())

    return {
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


async def download_checkpoint_path(name: str) -> str | None:
    if not VALID_CKPT_NAME.match(name) or '..' in name:
        raise ValueError("Invalid checkpoint name")

    def _find():
        for d in (CHECKPOINTS_DIR, TURBO_DIR, LORA_DIR):
            fp = (d / name).resolve()
            if fp.exists() and fp.suffix in (".soul", ".slo") and str(fp).startswith(str(d.resolve())):
                return str(fp)
        return None

    return await asyncio.to_thread(_find)


async def checkpoint_info(name: str) -> dict:
    if not VALID_CKPT_NAME.match(name) or '..' in name:
        raise ValueError("Invalid checkpoint name")
    info = await asyncio.to_thread(load_soul, name)
    if not info or info.get("soul") == "unknown":
        raise FileNotFoundError(f"Checkpoint not found: {name}")
    return info


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
    import numpy as np
    from domains.training.slonet import import_from_sou
    import base64

    def _find_ckpt():
        for d in (CHECKPOINTS_DIR, TURBO_DIR, LORA_DIR):
            fp = d / name
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
    block_size = getattr(net, 'block_size', 64)

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
