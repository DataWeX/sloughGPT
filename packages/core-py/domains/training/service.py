"""Training Service — single source of truth for training business logic.

Lives in core domain (packages/core-py). Zero HTTP/framework dependencies.
Raises plain exceptions. Returns raw dicts/lists. API layer wraps responses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import struct
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("slo.training")

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "models" / "auto-training"
LORA_DIR = REPO_ROOT / "data" / "user_adapters"
TURBO_DIR = REPO_ROOT / "models" / "turbo-trained"
MAX_CHECKPOINT_DISK_MB = 500
VALID_CKPT_NAME = re.compile(r'^[a-zA-Z0-9_\-\.]+$')

SOU_MAGIC = b"SOUL"

for d in (CHECKPOINTS_DIR, LORA_DIR, TURBO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── State ─────────────────────────────────────────────────────────────────────


@dataclass
class TrainingState:
    running: bool = False
    config: dict = None
    student_net: object | None = None
    student_tokenizer: object | None = None
    complete_enqueued: bool = False

    def __post_init__(self):
        if self.config is None:
            self.config = {}


_state = TrainingState()

_turbo_lock = threading.Lock()
_turbo_cancel_event = threading.Event()
_turbo_pause_event = threading.Event()
_turbo_state: dict[str, Any] = {
    "status": "idle",
    "job_id": None,
    "global_step": 0,
    "total_steps": 0,
    "progress": 0.0,
    "loss": None,
    "learning_rate": None,
    "steps_per_sec": None,
    "eta_s": None,
    "elapsed_s": None,
    "avg_quality": None,
    "result": None,
    "error": None,
    "paused": False,
    "last_heartbeat": 0.0,
}

_auto_train_cancel_event: threading.Event | None = None
_auto_train_pause_event: threading.Event | None = None

try:
    from domains.infrastructure.pugqeep import PGQ
    _auto_train_pgq = PGQ(
        name="auto-train",
        storage_dir=REPO_ROOT / "models" / "auto-training" / ".pgq",
    )
except Exception as exc:
    logger.warning("PGQ queue init failed: %s", exc)
    _auto_train_pgq = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _finite_payload(o: Any) -> Any:
    if is_dataclass(o):
        return _finite_payload(asdict(o))
    if isinstance(o, dict):
        return {k: _finite_payload(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite_payload(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ── Pure helpers (moved from auto_train.py) ─────────────────────────────────


def log_experiment_metric(experiment_id: str, metric: str, value: float, step: int = 0) -> None:
    """Append a metric entry to an experiment's metrics JSONL file."""
    try:
        from datetime import datetime, timezone
        exp_dir = REPO_ROOT / "data" / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = exp_dir / f"{experiment_id}_metrics.jsonl"
        entry = {
            "experiment_id": experiment_id,
            "metric": metric,
            "value": value,
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Failed to log experiment metric %s for %s: %s", metric, experiment_id, e)


def log_experiment_param(experiment_id: str, param_name: str, value: Any) -> None:
    """Append a param entry to an experiment's params JSONL file."""
    try:
        from datetime import datetime, timezone
        exp_dir = REPO_ROOT / "data" / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        params_file = exp_dir / f"{experiment_id}_params.jsonl"
        entry = {
            "experiment_id": experiment_id,
            "param": param_name,
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(params_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Failed to log experiment param %s for %s: %s", param_name, experiment_id, e)


def parse_subtitle_text(text: str) -> list:
    """Parse subtitle text (SRT/VTT) or plain text into a list of lines."""
    lines = []
    srt_pattern = re.compile(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}')
    vtt_pattern = re.compile(r'\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}\.\d{3}')

    if srt_pattern.search(text) or vtt_pattern.search(text):
        for line in text.split('\n'):
            line = line.strip()
            if srt_pattern.match(line) or vtt_pattern.match(line):
                continue
            if re.match(r'^\d+$', line):
                continue
            if line.startswith('WEBVTT'):
                continue
            if '-->' in line:
                continue
            if line and not line.startswith('['):
                lines.append(line)
    else:
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 2:
                lines.append(line)

    return lines


_VALID_DATASET_ID = re.compile(r'^[a-zA-Z0-9_\-]+$')


def resolve_dataset_path(dataset_id: str) -> str:
    """Resolve a dataset ID to a data file path. Raises ValueError on invalid input."""
    if not _VALID_DATASET_ID.match(dataset_id):
        raise ValueError(f"Invalid dataset ID: {dataset_id!r}")
    # Search multiple locations for the dataset
    for base_name in ("datasets", "data/datasets", "data"):
        ds_candidate = (REPO_ROOT / base_name / dataset_id).resolve()
        allowed_base = (REPO_ROOT / base_name).resolve()
        if not str(ds_candidate).startswith(str(allowed_base)):
            continue
        if not ds_candidate.exists():
            continue
        for name in ("corpus.jsonl", "input.txt", "train.txt", "text.txt"):
            candidate = ds_candidate / name
            if candidate.exists():
                return str(candidate)
        txt_files = list(ds_candidate.glob("*.txt"))
        if txt_files:
            return str(txt_files[0])
    return ""


def build_soul_prompt(soul_name: str) -> str:
    """Build a system prompt for a given soul personality name."""
    prompts = {
        "assistant": "You are a helpful assistant. Be clear and friendly.",
        "creative": "You are a creative thinker. Be imaginative and playful.",
        "analyst": "You are a precise analyst. Be methodical and thorough.",
        "coder": "You are an expert coder. Write clean, efficient code.",
        "teacher": "You are a patient teacher. Explain step by step.",
    }
    return prompts.get(soul_name, prompts["assistant"])


def get_soul_name(soul) -> str:
    """Extract the soul name from a soul object."""
    if hasattr(soul, 'name') and soul.name:
        return soul.name
    return getattr(soul, 'soul_name', 'unknown')


def get_soul_traits(soul) -> dict:
    """Extract personality traits from a soul object."""
    raw = getattr(soul, 'soul_traits', None)
    if raw:
        return raw
    if hasattr(soul, 'personality'):
        p = soul.personality
        if isinstance(p, dict):
            return p
        if hasattr(p, 'to_dict'):
            return p.to_dict()
        if hasattr(p, '__dict__'):
            return vars(p)
        return dict(p)
    return {}


def read_slo_json_header(path: Path) -> dict:
    """Read the JSON header from a .soul binary file."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != SOU_MAGIC:
                return {}
            f.seek(8)
            json_len = struct.unpack("<I", f.read(4))[0]
            header = f.read(json_len)
            return json.loads(header.decode())
    except Exception:
        return {}


def describe_checkpoint(ckpt: dict) -> str:
    """Generate a human-readable description of a checkpoint."""
    parts = []
    soul = ckpt.get("soul", "")
    loss = ckpt.get("loss")
    epochs = ckpt.get("epochs") or ckpt.get("epochs_trained")
    steps = ckpt.get("steps", 0)
    dataset = ckpt.get("training_dataset", "")
    traits = ckpt.get("traits", {})
    model_type = ckpt.get("model_type", "")

    if dataset:
        parts.append(f"Trained on {dataset}")
    elif soul and soul != "unknown":
        parts.append(f"Soul: {soul}")
    else:
        parts.append("A trained model")

    if epochs:
        parts.append(f"for {epochs} epoch{'s' if epochs != 1 else ''}")
    elif steps:
        parts.append(f"for {steps} steps")

    if loss is not None:
        if loss < 1.5:
            parts.append(f"(loss {loss:.2f} - learned well)")
        elif loss < 3.0:
            parts.append(f"(loss {loss:.2f} - moderate)")
        else:
            parts.append(f"(loss {loss:.2f} - needs more training)")

    if traits:
        trait_names = list(traits.keys())[:3]
        if trait_names:
            parts.append(f"Personality: {', '.join(trait_names)}")

    if model_type and model_type not in ("slonet", "unknown"):
        parts.append(f"[{model_type}]")

    return " ".join(parts) + "."


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
            except Exception:
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
            except Exception:
                continue
    return None


def _enforce_checkpoint_budget():
    try:
        files = []
        for ext in ("*.soul", "*.slo"):
            for f in CHECKPOINTS_DIR.glob(ext):
                files.append(f)
        if not files:
            return
        total = 0
        alive_files = []
        for f in files:
            try:
                total += f.stat().st_size
                alive_files.append(f)
            except FileNotFoundError:
                continue
        files = alive_files
        budget_bytes = MAX_CHECKPOINT_DISK_MB * 1024 * 1024
        if total <= budget_bytes:
            return
        files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
        if len(files) > 1:
            files_to_prune = files[:-1]
        else:
            return
        freed = 0
        for f in files_to_prune:
            if total - freed <= budget_bytes:
                break
            try:
                sz = f.stat().st_size
                f.unlink()
                freed += sz
                logger.info("Pruned checkpoint %s (%d bytes)", f.name, sz)
            except Exception:
                continue
    except Exception as e:
        logger.warning("Checkpoint budget enforcement failed: %s", e)


# ── Public API ────────────────────────────────────────────────────────────────


def get_state() -> TrainingState:
    return _state


def get_turbo_state() -> dict:
    return _turbo_state


def get_turbo_lock() -> threading.Lock:
    return _turbo_lock


def get_turbo_pause_event() -> threading.Event:
    return _turbo_pause_event


def get_cancel_event() -> threading.Event | None:
    return _auto_train_cancel_event


def set_cancel_event(event: threading.Event):
    global _auto_train_cancel_event
    _auto_train_cancel_event = event


def get_pause_event() -> threading.Event | None:
    return _auto_train_pause_event


def set_pause_event(event: threading.Event):
    global _auto_train_pause_event
    _auto_train_pause_event = event


def get_pgq():
    return _auto_train_pgq


# ── Checkpoint operations (pure business logic, no HTTP) ──────────────────────


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


def _scan_all_checkpoints() -> list[dict]:
    """Scan all checkpoint directories and return enriched metadata for each.

    Used by both list_checkpoints and export_all_metrics.
    """
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


async def export_all_metrics() -> dict:
    """Scan all checkpoints and return metrics export dict.

    Returns:
        Dict with ``exported_at``, ``total_checkpoints``, and ``checkpoints`` list.
    """
    checkpoints = await asyncio.to_thread(_scan_all_checkpoints)
    return {
        "exported_at": time.time(),
        "total_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
    }


async def get_log() -> list[str]:
    from domains.infrastructure.output_buffer import get_server_buffer
    return [line.text for line in get_server_buffer().tail(200)]


def get_turbo_status() -> dict:
    with _turbo_lock:
        state = dict(_turbo_state)
        if (state.get("status") == "running"
                and state.get("last_heartbeat", 0) > 0
                and (time.time() - state["last_heartbeat"]) > 30):
            state["status"] = "error"
            state["error"] = "Training process lost — no progress for 30 seconds"
            state["paused"] = False
            _turbo_state["status"] = "error"
            _turbo_state["error"] = state["error"]
            _turbo_state["paused"] = False
            _turbo_pause_event.clear()
    return state


# ── Turbo training (business logic extracted from auto_train router) ──────────


def start_turbo_training(config: dict) -> dict:
    """Validate turbo training input, resolve paths, init turbo state, register job.

    Args:
        config: Flat dict with keys matching TurboStartRequest fields.

    Returns:
        Dict with ``job_id`` and ``output_dir`` for the caller to launch the worker.

    Raises:
        ValueError: Already training, no data path, or checkpoint not found.
        FileNotFoundError: Data file or checkpoint file does not exist.
    """
    with _turbo_lock:
        if _turbo_state.get("status") == "running":
            raise RuntimeError("A turbo training job is already running")

    data_path = config.get("data_path", "")
    dataset_id = config.get("dataset_id")
    if not data_path and dataset_id:
        data_path = resolve_dataset_path(dataset_id)

    if not data_path:
        raise ValueError("No data_path or dataset_id provided")

    # Store resolved data_path back so run_turbo_worker can find it
    config["data_path"] = data_path

    resume = False
    resume_path = ""
    checkpoint_name = config.get("checkpoint_name")
    if checkpoint_name:
        ckpt_soul = CHECKPOINTS_DIR / f"{checkpoint_name}.soul"
        if ckpt_soul.exists():
            resume = True
            resume_path = str(ckpt_soul)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

    dp = Path(data_path).resolve()
    allowed_bases = [REPO_ROOT / "datasets", REPO_ROOT / "data"]
    if not dp.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not any(str(dp).startswith(str(b.resolve())) for b in allowed_bases if b.exists()):
        raise ValueError(
            f"Data path must be under datasets/ or data/ directories, got: {data_path}"
        )

    job_id = f"turbo_{int(time.time())}"
    cancel_event = threading.Event()
    pause_event = threading.Event()

    with _turbo_lock:
        _turbo_state.update({
            "status": "running",
            "job_id": job_id,
            "global_step": 0,
            "total_steps": 0,
            "progress": 0.0,
            "loss": None,
            "learning_rate": None,
            "steps_per_sec": None,
            "eta_s": None,
            "elapsed_s": None,
            "avg_quality": None,
            "result": None,
            "error": None,
            "paused": False,
            "last_heartbeat": time.time(),
        })

    try:
        from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
        _mgr = get_cancel_manager()
        _cm_op_id = _mgr.register(
            op_type=OpType.TRAINING,
            label=f"auto-turbo:{config.get('method', 'slonet')}",
            cancel_fn=lambda: cancel_event.set(),
        )
        _mgr.start(_cm_op_id)
        _turbo_state["_cm_op_id"] = _cm_op_id
    except Exception as e:
        logger.warning("CancelManager registration failed (turbo training may be unkillable): %s", e)

    output_dir = TURBO_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    method = config.get("method", "slonet")
    experiment_id = config.get("experiment_id")
    runtime_job = {
        "id": job_id,
        "name": config.get("name", None) or "turbo",
        "model": method or "sloughgpt",
        "dataset": config.get("dataset_id") or data_path,
        "data_path": data_path,
        "status": "running",
        "progress": 0.0,
        "epochs": config.get("epochs"),
        "current_epoch": 0,
        "global_step": 0,
        "loss": None,
        "train_loss": None,
        "checkpoint": None,
        "checkpoint_dir": str(output_dir),
        "error": None,
        "experiment_id": experiment_id,
    }
    from domains.training.runtime_protocol import get_training_runtime
    get_training_runtime().register(job_id, runtime_job, cancel_event, config)

    return {
        "job_id": job_id,
        "output_dir": str(output_dir),
        "data_path": data_path,
        "resume": resume,
        "resume_path": resume_path,
        "cancel_event": cancel_event,
        "pause_event": pause_event,
    }


def run_turbo_worker(config: dict) -> None:
    """Execute SloughGPTTrainer on the calling thread.

    Reads turbo state / lock / events from module-level. Updates ``_turbo_state``
    on every progress tick. Called from a background thread or PGQ executor.
    """
    from domains.training.train_pipeline import SloughGPTTrainer
    from domains.training.runtime_protocol import get_training_runtime

    job_id = _turbo_state.get("job_id", "")
    data_path = config.get("data_path", "")
    output_dir = config.get("output_dir", str(TURBO_DIR))
    resume = config.get("resume", False)
    resume_path = config.get("resume_path", "")
    experiment_id = config.get("experiment_id")

    def _finish_cm(status: str, error: str = "") -> None:
        op_id = _turbo_state.get("_cm_op_id")
        if op_id:
            try:
                from domains.infrastructure.cancel_manager import get_cancel_manager
                get_cancel_manager().finish(op_id, error=error if status != "completed" else "")
            except Exception as exc:
                logger.debug("CancelManager.finish failed: %s", exc)

    def on_progress(info: dict[str, Any]) -> None:
        with _turbo_lock:
            _turbo_state["global_step"] = int(info.get("global_step", _turbo_state["global_step"]))
            _turbo_state["total_steps"] = int(info.get("total_steps", _turbo_state["total_steps"]))
            _turbo_state["progress"] = float(info.get("progress_percent", 0))
            _turbo_state["loss"] = info.get("train_loss", _turbo_state["loss"])
            _turbo_state["learning_rate"] = info.get("learning_rate", _turbo_state["learning_rate"])
            _turbo_state["steps_per_sec"] = info.get("steps_per_sec", _turbo_state["steps_per_sec"])
            _turbo_state["eta_s"] = info.get("eta_s", _turbo_state["eta_s"])
            _turbo_state["elapsed_s"] = info.get("elapsed_s", _turbo_state["elapsed_s"])
            _turbo_state["avg_quality"] = info.get("avg_quality", _turbo_state.get("avg_quality"))
            _turbo_state["last_heartbeat"] = time.time()
        job = get_training_runtime().get(job_id)
        if job is not None:
            with _turbo_lock:
                job["progress"] = float(_turbo_state["progress"])
                job["global_step"] = int(_turbo_state["global_step"])
                job["train_loss"] = _turbo_state["loss"]
                job["loss"] = _turbo_state["loss"]
            get_training_runtime().sync(job_id)
        if experiment_id and _turbo_state["loss"] is not None:
            log_experiment_metric(experiment_id, "train_loss", float(_turbo_state["loss"]), int(_turbo_state["global_step"]))

    try:
        n_layer = config.get("n_layer") or config.get("n_decoder_layers") or 3
        block_size = config.get("block_size") or config.get("max_tgt_len") or 128

        trainer = SloughGPTTrainer(
            data_path=data_path,
            vocab_size=config.get("vocab_size", 500),
            n_embed=config.get("n_embed", 128),
            n_layer=n_layer,
            n_head=config.get("n_head", 4),
            block_size=block_size,
            dropout=config.get("dropout", 0.1),
            batch_size=config.get("batch_size", 4),
            epochs=config.get("epochs", 3),
            lr=config.get("learning_rate", 3e-4),
            checkpoint_dir=output_dir,
        )

        logger.info(
            "Starting SloughGPTTrainer with method=%s data=%s resume=%s",
            config.get("method"), data_path, resume,
        )
        _train_t0 = time.monotonic()
        result = trainer.train(
            on_progress=on_progress,
            cancel_event=_turbo_cancel_event,
            pause_event=_turbo_pause_event,
            resume=resume,
            resume_path=resume_path or None,
        )
        _train_elapsed_ms = (time.monotonic() - _train_t0) * 1000
        logger.info("SloughGPTTrainer result: %s (elapsed=%dms)", result, _train_elapsed_ms)

        if _turbo_cancel_event.is_set():
            _turbo_pause_event.clear()
            with _turbo_lock:
                _turbo_state["status"] = "error"
                _turbo_state["error"] = "Training cancelled"
                _turbo_state["paused"] = False
            job = get_training_runtime().get(job_id)
            if job is not None:
                job["status"] = "cancelled"
                job["error"] = "Training cancelled"
                get_training_runtime().sync(job_id)
            _finish_cm("cancelled", "Training cancelled")
            return

        if isinstance(result, dict) and result.get("status") == "error":
            _turbo_pause_event.clear()
            with _turbo_lock:
                _turbo_state["status"] = "error"
                _turbo_state["error"] = result.get("message") or "Training failed"
                _turbo_state["paused"] = False
            job = get_training_runtime().get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = result.get("message") or "Training failed"
                get_training_runtime().sync(job_id)
            _finish_cm("failed", result.get("message") or "Training failed")
            return

        _turbo_pause_event.clear()
        with _turbo_lock:
            _turbo_state["status"] = "complete"
            _turbo_state["result"] = _finite_payload(result)
            _turbo_state["progress"] = 100.0
            _turbo_state["paused"] = False
        job = get_training_runtime().get(job_id)
        if job is not None:
            job["status"] = "completed"
            job["progress"] = 100.0
            soul_files = sorted(Path(output_dir).glob("*.soul"))
            if soul_files:
                job["checkpoint"] = str(soul_files[-1])
            get_training_runtime().sync(job_id)
        _finish_cm("completed")
    except Exception as e:
        logger.error("SloughGPTTrainer failed: %s", e)
        _turbo_pause_event.clear()
        with _turbo_lock:
            _turbo_state["status"] = "error"
            _turbo_state["error"] = str(e)
            _turbo_state["paused"] = False
        job = get_training_runtime().get(job_id)
        if job is not None:
            job["status"] = "failed"
            job["error"] = str(e)
            get_training_runtime().sync(job_id)
        _finish_cm("failed", str(e))
        logger.warning("Turbo training failed: %s", e)
    finally:
        _state.running = False


# ── From-sessions training (business logic extracted from auto_train router) ──


def start_from_sessions_training(state: TrainingState, config: dict) -> dict:
    """Validate no training running, build config, mark running.

    Args:
        state: The shared ``TrainingState`` instance.
        config: Flat dict with from-sessions parameters.

    Returns:
        The built config dict (also stored in ``state.config``).

    Raises:
        RuntimeError: Training already in progress.
    """
    if state.running:
        raise RuntimeError("Training already in progress")

    state.running = True
    state.config = {
        "method": "from-sessions",
        "epochs": config.get("epochs", 5),
        "learning_rate": config.get("learning_rate", 3e-4),
        "batch_size": config.get("batch_size", 8),
        "n_embed": config.get("n_embed", 128),
        "n_layer": config.get("n_layer", 4),
        "n_head": config.get("n_head", 4),
        "block_size": config.get("block_size", 128),
        "dropout": config.get("dropout", 0.1),
        "soul_name": config.get("soul_name", "chat-trained"),
        "min_pair_quality": config.get("min_pair_quality", 2.0),
        "max_pairs": config.get("max_pairs", 500),
        "checkpoint_name": config.get("checkpoint_name"),
        "session_ids": config.get("session_ids"),
        "experiment_id": config.get("experiment_id"),
        "started_at": time.time(),
    }
    return state.config


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


# ── SSE Stream Business Logic ───────────────────────────────────────
# The StreamingResponse + request.is_disconnected() stay in the router.
# These functions own the business decisions: what to do when training
# completes, how to update runtime, when to log experiments.


def process_training_completion(
    ev: dict,
    task_id: str,
    config: dict,
    checkpoints_dir: Path,
    finish_cm_fn,
) -> None:
    """Handle a completed/errored training event — update runtime, log experiments."""
    from domains.training.runtime_protocol import get_training_runtime

    job = get_training_runtime().get(task_id)
    if job is None:
        return

    if ev.get("status") == "complete":
        job["status"] = "completed"
    else:
        job["status"] = "failed"
        job["error"] = str(ev.get("message") or ev.get("data") or "training failed")

    # Attach latest checkpoint path
    try:
        soul_files = sorted(checkpoints_dir.glob("*.soul"))
        if soul_files:
            job["checkpoint"] = str(soul_files[-1])
    except Exception:
        logger.debug("Failed to discover checkpoints in %s", checkpoints_dir, exc_info=True)

    get_training_runtime().sync(task_id)

    # Log experiment metrics on completion
    experiment_id = config.get("experiment_id")
    if experiment_id and ev.get("status") == "complete":
        final_loss = job.get("train_loss") or job.get("loss")
        if final_loss is not None:
            log_experiment_metric(experiment_id, "final_train_loss", float(final_loss), int(job.get("global_step", 0)))
        log_experiment_param(experiment_id, "epochs", config.get("epochs", 0))
        log_experiment_param(experiment_id, "learning_rate", config.get("learning_rate", 0))

    finish_cm_fn(
        "complete" if ev.get("status") == "complete" else "failed",
        str(ev.get("message") or ev.get("data") or "") if ev.get("status") != "complete" else "",
    )


def cleanup_stream_state(
    task_id: str,
    config: dict,
    state: dict,
    finish_cm_fn,
    status: str = "interrupted",
    error: str = "",
) -> None:
    """Reset state after SSE stream ends — mark runtime as interrupted if needed."""
    from domains.training.runtime_protocol import get_training_runtime

    state["running"] = False
    job = get_training_runtime().get(task_id)
    if job is not None and job.get("status") not in ("completed", "failed", "cancelled"):
        job["status"] = status
        job["error"] = job.get("error") or error or "Training stream ended before completion"
        get_training_runtime().sync(task_id)
