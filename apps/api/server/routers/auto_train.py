"""
Auto-Train Router — SloNet LM Training Pipeline

Trains a SloNet LSTM as a next-token-prediction language model on user-provided
text (source_text, dataset, or file).  Pure NumPy student training.
Exports checkpoints as .soul (binary float32 format).

Phase sequence: TRAINING -> COMPLETE | FAILED

Encapsulates router state in ``AutoTrainRouter`` class rather than module-level
mutable globals.
"""

from dataclasses import dataclass, field, is_dataclass, asdict
import asyncio
import threading
from typing import Any, AsyncGenerator
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import Request
from pydantic import BaseModel, Field
import json
import logging
import math
import re
import time

from schemas.common import success_response, raise_error, classify_and_raise, safe_audit_log

from training.runtime import get_training_runtime

try:
    from domains.api.sse_envelope import sse_event, sse_error
except ImportError:
    def sse_event(stream, phase, status, data=None, meta=None, message="") -> dict:
        """sse_event."""
        import json
        return "data: " + json.dumps({
            "stream": stream, "phase": phase, "status": status,
            "data": data or {}, "meta": meta or {}, "message": message
        }) + "\n\n"
    def sse_error(stream, phase, error, meta=None, code=None, http_status=None) -> dict:
        """sse_error."""
        data = {"error": error}
        if code is not None:
            data["code"] = code
        if http_status is not None:
            data["http_status"] = http_status
        return sse_event(stream, phase, "error", data, meta or {}, f"Error: {error}")
    def sse_complete(stream, phase="COMPLETE", data=None, meta=None, message="Done") -> dict:
        """sse_complete."""
        return sse_event(stream, phase, "complete", data or {}, meta or {}, message)


def _log_experiment_metric(experiment_id: str, metric: str, value: float, step: int = 0) -> None:
    """Append a metric entry to an experiment's metrics JSONL file."""
    try:
        from datetime import datetime, timezone
        exp_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "experiments"
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
    except Exception:
        pass


def _log_experiment_param(experiment_id: str, param_name: str, value: Any) -> None:
    """Append a param entry to an experiment's params JSONL file."""
    try:
        from datetime import datetime, timezone
        exp_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "experiments"
        params_file = exp_dir / f"{experiment_id}_params.jsonl"
        entry = {
            "experiment_id": experiment_id,
            "param": param_name,
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(params_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


@dataclass
class AutoTrainState:
    running: bool = False
    config: dict = field(default_factory=dict)
    student_net: object | None = None
    student_tokenizer: object | None = None
    complete_enqueued: bool = False


_auto_train_cancel_event: threading.Event | None = None
_auto_train_pause_event: threading.Event | None = None

_turbo_lock = threading.Lock()
_turbo_cancel_event = threading.Event()
_turbo_pause_event = threading.Event()
_turbo_state: dict[str, Any] = {
    "status": "idle",  # idle | running | complete | error
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


def _finite_payload(o: Any) -> Any:
    """Recursively convert a value into a JSON-safe plain structure.

    Dataclasses (e.g. ``TrainResult``) become dicts, and non-finite floats
    (``inf``/``nan`` — produced when a trainer has no eval run) become ``None``
    so that ``json.dumps(allow_nan=False)`` never fails on the status endpoint.

    Args:
        o: Arbitrary value (dict, list, tuple, dataclass, scalar).

    Returns:
        A JSON-serializable copy of ``o`` with non-finite floats replaced by ``None``.

    Side effects:
        None — the input is not mutated.
    """
    if is_dataclass(o):
        return _finite_payload(asdict(o))
    if isinstance(o, dict):
        return {k: _finite_payload(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite_payload(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


class _AutoTrainCancelled(Exception):
    pass


REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "models" / "auto-training"
LORA_DIR = REPO_ROOT / "data" / "user_adapters"

autotrain_logger = logging.getLogger("slo.autotrain")
autotrain_logger.setLevel(logging.INFO)

try:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
except Exception as exc:
    autotrain_logger.warning("Failed to create auto-train directories: %s", exc)

try:
    from domains.infrastructure.pugqeep import PGQ
    _auto_train_pgq = PGQ(
        name="auto-train",
        storage_dir=REPO_ROOT / "models" / "auto-training" / ".pgq",
    )
except Exception as exc:
    autotrain_logger.warning("PGQ queue init failed (queue-based training disabled): %s", exc)
    _auto_train_pgq = None

MAX_CHECKPOINT_DISK_MB = 500

SOU_MAGIC = b"SOUL"

_VALID_CKPT_NAME = re.compile(r'^[a-zA-Z0-9_\-\.]+$')


def _enforce_checkpoint_budget():
    try:
        files = []
        for ext in ("*.soul", "*.slo"):
            for f in CHECKPOINTS_DIR.glob(ext):
                files.append(f)
        if not files:
            return
        total = sum(f.stat().st_size for f in files)
        budget_bytes = MAX_CHECKPOINT_DISK_MB * 1024 * 1024
        if total <= budget_bytes:
            return
        files.sort(key=lambda p: p.stat().st_mtime)
        if len(files) > 1:
            files_to_prune = files[:-1]
        else:
            return
        freed = 0
        for f in files_to_prune:
            if total - freed <= budget_bytes:
                break
            size = f.stat().st_size
            f.unlink()
            freed += size
            meta = Path(str(f) + ".meta.json")
            if meta.exists():
                meta.unlink()
            autotrain_logger.info("Pruned old checkpoint: %s (%.1f KB)", f.name, size / 1024, extra={"tag": "TRAIN"})
        autotrain_logger.info(
            "Checkpoint budget enforced: freed %.1f MB (budget %d MB)",
            freed / 1024 / 1024, MAX_CHECKPOINT_DISK_MB,
            extra={"tag": "TRAIN"},
        )
    except Exception as e:
        autotrain_logger.warning("Checkpoint budget enforcement failed: %s", e, extra={"tag": "TRAIN"})


def _parse_subtitle_text(text: str) -> list:
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


class StartRequest(BaseModel):
    teacher_model: str = Field(default="gpt2", description="Teacher model ID registered in the model server")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    soul_name: str = Field(default="assistant", min_length=1, max_length=200)
    epochs: int = Field(default=20, ge=1, le=1000)
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1.0)
    batch_size: int = Field(default=16, ge=1, le=1024, description="Chunk size for training")
    source_text: str | None = Field(default=None, description="Custom training text (SRT, plain, or lines). If provided, train on this instead of generating from teacher.")
    checkpoint_name: str | None = Field(default=None, description="Load existing checkpoint and continue training")
    dataset_id: str | None = Field(default=None, description="Dataset ID from /datasets to train on")
    early_stopping_patience: int = Field(default=5, ge=0, le=100, description="Stop if no eval improvement for N evaluations (0 = disabled)")
    # Native training architecture params (used when method=native)
    n_embed: int = Field(default=128, ge=16, le=1024, description="Embedding dimension for native training")
    n_layer: int = Field(default=4, ge=1, le=24, description="Number of transformer layers for native training")
    n_head: int = Field(default=4, ge=1, le=64, description="Number of attention heads for native training")
    block_size: int = Field(default=128, ge=8, le=2048, description="Context window size for native training")
    dropout: float = Field(default=0.1, ge=0.0, le=0.9, description="Dropout rate for native training")
    checkpoint_dir: str | None = Field(default=None, description="Override checkpoint output directory (default: models/auto-training)")
    experiment_id: str | None = Field(default=None, description="Link to an experiment — auto-logs loss/accuracy metrics")


class TurboStartRequest(BaseModel):
    method: str = Field(default="slonet", description="Training method: 'slonet', 'transformer', 'nanogpt', 'hf'")
    data_path: str = Field(default="", description="Path to training data file")
    dataset_id: str | None = Field(default=None, description="Dataset ID to train on")
    checkpoint_name: str | None = Field(default=None, description="Resume from existing checkpoint")
    epochs: int = Field(default=3, ge=1, le=1000)
    batch_size: int = Field(default=4, ge=1, le=256)
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1.0)
    vocab_size: int = Field(default=500, ge=50, le=50000)
    n_embed: int = Field(default=128, ge=16, le=1024)
    n_head: int = Field(default=4, ge=1, le=64)
    n_layer: int = Field(default=3, ge=1, le=24)
    block_size: int = Field(default=128, ge=8, le=2048)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    n_decoder_layers: int | None = Field(default=None, description="Deprecated: use n_layer")
    max_tgt_len: int | None = Field(default=None, description="Deprecated: use block_size")
    experiment_id: str | None = Field(default=None, description="Link to an experiment — auto-logs loss/accuracy metrics")


class FromSessionsRequest(BaseModel):
    epochs: int = Field(default=5, ge=1, le=100)
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1.0)
    batch_size: int = Field(default=8, ge=1, le=128)
    n_embed: int = Field(default=128, ge=16, le=512)
    n_layer: int = Field(default=4, ge=1, le=12)
    n_head: int = Field(default=4, ge=1, le=16)
    block_size: int = Field(default=128, ge=16, le=512)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    soul_name: str = Field(default="chat-trained")
    min_pair_quality: float = Field(default=2.0, ge=0.0, le=5.0)
    max_pairs: int = Field(default=500, ge=10, le=10000)
    checkpoint_name: str | None = Field(default=None, description="Resume from existing checkpoint")
    session_ids: list | None = Field(default=None, description="If provided, only train from these session IDs")
    experiment_id: str | None = Field(default=None, description="Link to an experiment — auto-logs loss/accuracy metrics")


_VALID_DATASET_ID = re.compile(r'^[a-zA-Z0-9_\-]+$')

def _resolve_dataset_path(dataset_id: str) -> str:
    if not _VALID_DATASET_ID.match(dataset_id):
        raise_error(f"Invalid dataset ID: {dataset_id!r}", code="E_BAD_REQUEST", status_code=400)
    ds_candidate = (REPO_ROOT / "datasets" / dataset_id).resolve()
    allowed_base = (REPO_ROOT / "datasets").resolve()
    if not str(ds_candidate).startswith(str(allowed_base)):
        raise_error("Path traversal detected", code="E_BAD_REQUEST", status_code=400)
    if not ds_candidate.exists():
        return ""
    for name in ("corpus.jsonl", "input.txt", "train.txt", "text.txt"):
        candidate = ds_candidate / name
        if candidate.exists():
            return str(candidate)
    txt_files = list(ds_candidate.glob("*.txt"))
    if txt_files:
        return str(txt_files[0])
    return ""


def _build_soul_prompt(soul_name: str) -> str:
    prompts = {
        "assistant": "You are a helpful assistant. Be clear and friendly.",
        "creative": "You are a creative thinker. Be imaginative and playful.",
        "analyst": "You are a precise analyst. Be methodical and thorough.",
        "coder": "You are an expert coder. Write clean, efficient code.",
        "teacher": "You are a patient teacher. Explain step by step.",
    }
    return prompts.get(soul_name, prompts["assistant"])


def _get_soul_name(soul) -> str:
    if hasattr(soul, 'name') and soul.name:
        return soul.name
    return getattr(soul, 'soul_name', 'unknown')


def _get_soul_traits(soul) -> dict:
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


def _read_slo_json_header(path: Path) -> dict:
    try:
        import struct
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


def _load_soul_meta(ckpt_file: Path) -> dict:
    meta_file = ckpt_file.with_suffix(ckpt_file.suffix + ".meta.json")
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text())
        except Exception:
            autotrain_logger.debug("Failed to parse %s", meta_file, exc_info=True)
    if ckpt_file.suffix == ".soul":
        return _read_slo_json_header(ckpt_file)
    if ckpt_file.suffix == ".slo":
        try:
            from domains.inference.slo_format import SouParser
            profile = SouParser.parse(ckpt_file.read_text(encoding="utf-8"))
            return {
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
            return {}
    return {}


# Per-file metadata cache: keyed by (file_path, mtime) to avoid re-reading
# .meta.json / binary headers for unchanged files on repeated scans.
_meta_cache: dict[str, tuple[float, dict]] = {}


def _load_soul_meta_cached(ckpt_file: Path) -> dict:
    """Wrapper around _load_soul_meta with a per-file mtime cache."""
    try:
        mtime = ckpt_file.stat().st_mtime
    except OSError:
        return _load_soul_meta(ckpt_file)
    cache_key = str(ckpt_file)
    cached = _meta_cache.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]
    result = _load_soul_meta(ckpt_file)
    _meta_cache[cache_key] = (mtime, result)
    return result


def _describe_checkpoint(ckpt: dict) -> str:
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




class AutoTrainRouter:
    def __init__(self):
        self.state = AutoTrainState()
        self.router = APIRouter(prefix="/auto-train", tags=["training"])
        self.REPO_ROOT = REPO_ROOT
        self.CHECKPOINTS_DIR = CHECKPOINTS_DIR
        self.TURBO_DIR = REPO_ROOT / "models" / "turbo-trained"
        self.TURBO_DIR.mkdir(parents=True, exist_ok=True)
        self.LORA_DIR = LORA_DIR
        self._checkpoints_cache: tuple | None = None
        self._checkpoints_cache_ts: float = 0
        self._register_routes()

    def _find_checkpoint(self, name: str) -> Path | None:
        """Locate a checkpoint file by name across all checkpoint directories.

        Searches ``CHECKPOINTS_DIR`` then ``TURBO_DIR``, appending each known
        extension (``.soul``/``.slo``) when ``name`` has none.

        Args:
            name: Checkpoint file name or bare stem.

        Returns:
            Absolute path to the checkpoint, or ``None`` if not found.
        """
        if name.endswith((".soul", ".slo")):
            for base in (self.CHECKPOINTS_DIR, self.TURBO_DIR):
                candidate = (base / name).resolve()
                if candidate.exists() and str(candidate).startswith(str(base.resolve())):
                    return candidate
            return None
        for ext in (".soul", ".slo"):
            for base in (self.CHECKPOINTS_DIR, self.TURBO_DIR):
                candidate = (base / (name + ext)).resolve()
                if candidate.exists() and str(candidate).startswith(str(base.resolve())):
                    return candidate
        return None

    def _register_routes(self):
        self.router.add_api_route("/start", self.start, methods=["POST"])
        self.router.add_api_route("/start-turbo", self.start_turbo, methods=["POST"])
        self.router.add_api_route("/turbo/status", self.turbo_status, methods=["GET"])
        self.router.add_api_route("/stop", self.stop, methods=["POST"])
        self.router.add_api_route("/pause", self.pause, methods=["POST"])
        self.router.add_api_route("/resume", self.resume, methods=["POST"])
        self.router.add_api_route("/status", self.status, methods=["GET"])
        self.router.add_api_route("/stream", self.stream, methods=["GET"])
        self.router.add_api_route("/checkpoints", self.list_checkpoints, methods=["GET"])
        self.router.add_api_route("/checkpoints/{name}", self.delete_checkpoint, methods=["DELETE"])
        self.router.add_api_route("/checkpoints/{name}/load", self.load_checkpoint, methods=["POST"])
        self.router.add_api_route("/checkpoints/{name}/download", self.download_checkpoint, methods=["GET"])
        self.router.add_api_route("/checkpoints/{name}/info", self.checkpoint_info, methods=["GET"])
        self.router.add_api_route("/checkpoints/{name}/export-mobile", self.export_checkpoint_mobile, methods=["GET"])
        self.router.add_api_route("/log", self.auto_train_log, methods=["GET"])
        self.router.add_api_route("/from-sessions/start", self.start_from_sessions, methods=["POST"])
        self.router.add_api_route("/from-sessions/stream", self.stream_from_sessions, methods=["GET"])
        self.router.add_api_route("/from-sessions/cancel", self.cancel_from_sessions, methods=["GET"])
        self.router.add_api_route("/metrics/export", self.export_metrics, methods=["GET"])

    async def start(self, req: StartRequest) -> dict:
        try:
            """start."""
            if not req.source_text and not req.dataset_id and not req.checkpoint_name:
                raise_error("Provide source_text, dataset_id, or checkpoint_name", code="E_VAL_REQUEST")

            data_path = ""
            if req.source_text:
                source_lines = _parse_subtitle_text(req.source_text)
                if source_lines:

                    def _write_src():
                        tmp = self.REPO_ROOT / ".opencode" / "tmp" / f"autotrain_source_{int(time.time())}.txt"
                        tmp.parent.mkdir(parents=True, exist_ok=True)
                        tmp.write_text("\n".join(source_lines), encoding="utf-8")
                        return tmp

                    tmp = await asyncio.to_thread(_write_src)
                    data_path = str(tmp)
                    autotrain_logger.info("Wrote %d source lines to %s", len(source_lines), tmp, extra={"tag": "TRAIN", "context": {"source_lines": len(source_lines), "path": str(tmp)}})
            elif req.dataset_id:
                data_path = _resolve_dataset_path(req.dataset_id)

            resume = getattr(req, "resume", False)
            resume_path = getattr(req, "resume_path", "")
            method = "slonet"

            if not resume and req.checkpoint_name:
                ckpt_soul = self.CHECKPOINTS_DIR / f"{req.checkpoint_name}.soul"
                if await asyncio.to_thread(ckpt_soul.exists):
                    resume = True
                    resume_path = str(ckpt_soul)
                    method = "chat-trained"
                    autotrain_logger.info("Auto-resume from %s (SloNet)", resume_path, extra={"tag": "TRAIN", "context": {"checkpoint": resume_path, "method": method}})

            self.state.config = {
                "epochs": req.epochs,
                "learning_rate": req.learning_rate,
                "batch_size": req.batch_size,
                "data_path": data_path,
                "checkpoint_name": req.checkpoint_name or "",
                "soul_name": req.soul_name,
                "resume": resume,
                "resume_path": resume_path,
                "method": method,
                "early_stopping_patience": req.early_stopping_patience,
                "n_embed": req.n_embed,
                "n_layer": req.n_layer,
                "n_head": req.n_head,
                "block_size": req.block_size,
                "dropout": req.dropout,
                "checkpoint_dir": req.checkpoint_dir,
                "experiment_id": req.experiment_id,
            }
            self.state.running = True
            import state as _srv_state
            _srv_state.training_active = True
            autotrain_logger.info("Auto-train configured: data_path=%s epochs=%d", data_path, req.epochs, extra={"tag": "TRAIN"})
            safe_audit_log("training.start", resource=req.dataset_id or req.checkpoint_name or (req.soul_name if req.source_text else "inline"), detail="resume" if resume else "fresh", method=method, epochs=req.epochs, dataset_id=req.dataset_id or "", checkpoint_name=req.checkpoint_name or "")
            return success_response(data={"status": "ready", "data_path": data_path, "epochs": req.epochs, "config": self.state.config})

        except Exception as e:
            classify_and_raise(e, source="autotrain.start")
    async def start_turbo(self, req: TurboStartRequest) -> dict:
        try:
            """start_turbo."""
            global _turbo_state, _turbo_cancel_event

            with _turbo_lock:
                if _turbo_state.get("status") == "running":
                    raise_error("A turbo training job is already running", code="E_INFRA_BUSY")

            data_path = req.data_path
            if not data_path and req.dataset_id:
                data_path = _resolve_dataset_path(req.dataset_id)

            if not data_path:
                raise_error("No data_path or dataset_id provided", code="E_VAL_REQUEST")

            resume = False
            resume_path = ""
            if req.checkpoint_name:
                ckpt_soul = self.CHECKPOINTS_DIR / f"{req.checkpoint_name}.soul"
                if await asyncio.to_thread(ckpt_soul.exists):
                    resume = True
                    resume_path = str(ckpt_soul)
                    autotrain_logger.info("Turbo resume from %s", resume_path, extra={"tag": "TRAIN"})
                else:
                    raise_error(f"Checkpoint not found: {req.checkpoint_name}", code="E_NOT_FOUND")

            # Validate data_path exists and is under allowed directories
            from pathlib import Path as _P
            _dp = _P(data_path).resolve()
            _allowed_bases = [REPO_ROOT / "datasets", REPO_ROOT / "data"]
            if not _dp.exists():
                raise_error(f"Data file not found: {data_path}", code="E_NOT_FOUND")
            if not any(str(_dp).startswith(str(b.resolve())) for b in _allowed_bases if b.exists()):
                raise_error(
                    f"Data path must be under datasets/ or data/ directories, got: {data_path}",
                    code="E_BAD_REQUEST", status_code=400,
                )

            with _turbo_lock:
                _turbo_state = {
                    "status": "running",
                    "job_id": f"turbo_{int(time.time())}",
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
                }
                _turbo_cancel_event = threading.Event()
                _turbo_pause_event = threading.Event()

            # Register with CancelManager
            try:
                from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
                _mgr = get_cancel_manager()
                _cm_op_id = _mgr.register(
                    op_type=OpType.TRAINING,
                    label=f"auto-turbo:{req.method}",
                    cancel_fn=lambda: _turbo_cancel_event.set(),
                )
                _mgr.start(_cm_op_id)
                _turbo_state["_cm_op_id"] = _cm_op_id
            except Exception as e:
                autotrain_logger.warning("CancelManager registration failed (turbo training may be unkillable): %s", e, extra={"tag": "TRAIN"})

            output_dir = Path(self.REPO_ROOT / "models" / "turbo-trained")
            await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

            job_id = _turbo_state["job_id"]
            runtime_job = {
                "id": job_id,
                "name": getattr(req, "name", None) or "turbo",
                "model": req.method or "sloughgpt",
                "dataset": req.dataset_id or data_path,
                "data_path": data_path,
                "status": "running",
                "progress": 0.0,
                "epochs": req.epochs,
                "current_epoch": 0,
                "global_step": 0,
                "loss": None,
                "train_loss": None,
                "checkpoint": None,
                "checkpoint_dir": str(output_dir),
                "error": None,
                "experiment_id": req.experiment_id,
            }
            from training.runtime import get_training_runtime
            get_training_runtime().register(
                job_id, runtime_job, _turbo_cancel_event, req.model_dump()
            )

            if _auto_train_pgq is not None:
                _auto_train_pgq.submit_training(
                    self._run_turbo_pgq,
                    job_id,
                    tree_id="auto-train",
                    req=req,
                    data_path=data_path,
                    output_dir=str(output_dir),
                    resume=resume,
                    resume_path=resume_path,
                )
            else:
                threading.Thread(
                    target=self._run_turbo,
                    args=(req, data_path, str(output_dir), job_id),
                    kwargs={"resume": resume, "resume_path": resume_path},
                    name=f"turbo-train-{job_id}",
                    daemon=True,
                ).start()

            autotrain_logger.info(
                "Turbo training started in background: job_id=%s data=%s",
                job_id, data_path, extra={"tag": "TRAIN"},
            )
            return success_response(data={"status": "started", "job_id": job_id, "message": "Turbo training started in background"})

        except Exception as e:
            classify_and_raise(e, source="autotrain.start_turbo")
    def _run_turbo(self, req: TurboStartRequest, data_path: str, output_dir: str, job_id: str, *, resume: bool = False, resume_path: str = "") -> None:
        """Run SloughGPTTrainer on a daemon thread, publishing progress to _turbo_state."""
        from domains.training.train_pipeline import SloughGPTTrainer

        def _finish_cm(status: str, error: str = "") -> None:
            op_id = _turbo_state.get("_cm_op_id")
            if op_id:
                try:
                    from domains.infrastructure.cancel_manager import get_cancel_manager
                    get_cancel_manager().finish(op_id, error=error if status != "completed" else "")
                except Exception as exc:
                    autotrain_logger.debug("CancelManager.finish failed: %s", exc)

        def on_progress(info: dict[str, Any]) -> None:
            """on_progress."""
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
            eid = req.experiment_id
            if eid and _turbo_state["loss"] is not None:
                _log_experiment_metric(eid, "train_loss", float(_turbo_state["loss"]), int(_turbo_state["global_step"]))

        try:
            n_layer = req.n_layer or req.n_decoder_layers or 3
            block_size = req.block_size or req.max_tgt_len or 128

            trainer = SloughGPTTrainer(
                data_path=data_path,
                vocab_size=req.vocab_size,
                n_embed=req.n_embed,
                n_layer=n_layer,
                n_head=req.n_head,
                block_size=block_size,
                dropout=req.dropout,
                batch_size=req.batch_size,
                epochs=req.epochs,
                lr=req.learning_rate,
                checkpoint_dir=output_dir,
            )

            autotrain_logger.info(
                "Starting SloughGPTTrainer with method=%s data=%s resume=%s",
                req.method, data_path, resume, extra={"tag": "TRAIN"},
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
            autotrain_logger.info("SloughGPTTrainer result: %s (elapsed=%dms)", result, _train_elapsed_ms, extra={"tag": "TRAIN"})

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

            safe_audit_log("training.complete", resource=data_path or req.dataset_id or "turbo", detail=f"turbo elapsed={_train_elapsed_ms:.0f}ms", method=req.method or "", epochs=req.epochs)
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
            autotrain_logger.error("SloughGPTTrainer failed: %s", e, extra={"tag": "TRAIN"})
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
            autotrain_logger.warning("Turbo training failed: %s", e, extra={"tag": "TRAIN"})

    def _run_turbo_pgq(
        self, job_id: str, tree_id: str, point_library: Any,
        is_cancelled: Any, *, req: Any, data_path: str, output_dir: str,
        resume: bool = False, resume_path: str = "",
    ) -> dict:
        """PGQ-compatible training entry point.

        Matches the ``PGQ.submit_training`` function signature:
        ``fn(job_id, tree_id, point_library, is_cancelled, **kwargs)``.

        Delegates to the existing ``_run_turbo`` logic.  Returns the
        trainer result dict so ``TrainingExecutor`` can auto-compress
        weights into Points.
        """
        self._run_turbo(req, data_path, output_dir, job_id, resume=resume, resume_path=resume_path)
        return {}

    async def turbo_status(self) -> dict:
        try:
            """Return the current turbo training job progress."""
            with _turbo_lock:
                state = dict(_turbo_state)
                # Detect stale training: running but no heartbeat for 30s
                if (state.get("status") == "running"
                        and state.get("last_heartbeat", 0) > 0
                        and (time.time() - state["last_heartbeat"]) > 30):
                    autotrain_logger.warning(
                        "Turbo training stale (no heartbeat for %.0fs) — marking failed",
                        time.time() - state["last_heartbeat"], extra={"tag": "TRAIN"},
                    )
                    state["status"] = "error"
                    state["error"] = "Training process lost — no progress for 30 seconds"
                    state["paused"] = False
                    _turbo_state["status"] = "error"
                    _turbo_state["error"] = state["error"]
                    _turbo_state["paused"] = False
                    _turbo_pause_event.clear()
                return success_response(data=_finite_payload(state))

        except Exception as e:
            classify_and_raise(e, source="autotrain.turbo_status")
    async def stop(self) -> dict:
        try:
            """stop."""
            self.state.running = False
            if _auto_train_cancel_event is not None:
                _auto_train_cancel_event.set()
            _turbo_cancel_event.set()
            _turbo_pause_event.clear()
            if _auto_train_pgq is not None:
                job_id = _turbo_state.get("job_id")
                if job_id:
                    try:
                        _auto_train_pgq.cancel_training(job_id)
                    except Exception as exc:
                        autotrain_logger.warning("Failed to cancel turbo training job %s: %s", job_id, exc)
            try:
                from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
                get_cancel_manager().cancel_all(op_type=OpType.TRAINING)
            except Exception as exc:
                autotrain_logger.warning("CancelManager.cancel_all failed: %s", exc)
            if _auto_train_cancel_event is not None:
                safe_audit_log("training.stop", resource=(self.state.config or {}).get("soul_name", ""), detail="cancelling")
                return success_response(data={"status": "cancelling", "message": "Cancelling auto-training"})
            safe_audit_log("training.stop", detail="not_running")
            return success_response(data={"status": "stopped"})

        except Exception as e:
            classify_and_raise(e, source="autotrain.stop")
    async def pause(self) -> dict:
        try:
            """pause."""
            if _turbo_state.get("status") == "running":
                if _turbo_pause_event.is_set():
                    return success_response(data={"success": False, "message": "Training is already paused"})
                _turbo_pause_event.set()
                with _turbo_lock:
                    _turbo_state["paused"] = True
                safe_audit_log("training.pause", resource=(self.state.config or {}).get("soul_name", ""), detail="paused")
                return success_response(data={"success": True, "message": "Training paused"})
            if _auto_train_pause_event is None:
                return success_response(data={"success": False, "message": "No active training to pause"})
            if _auto_train_pause_event.is_set():
                return success_response(data={"success": False, "message": "Training is already paused"})
            _auto_train_pause_event.set()
            safe_audit_log("training.pause", resource=(self.state.config or {}).get("soul_name", ""), detail="paused")
            return success_response(data={"success": True, "message": "Training paused"})

        except Exception as e:
            classify_and_raise(e, source="autotrain.pause")
    async def resume(self) -> dict:
        try:
            """resume."""
            if _turbo_state.get("status") == "running":
                if not _turbo_pause_event.is_set():
                    return success_response(data={"success": False, "message": "Training is not paused"})
                _turbo_pause_event.clear()
                with _turbo_lock:
                    _turbo_state["paused"] = False
                safe_audit_log("training.resume", resource=(self.state.config or {}).get("soul_name", ""), detail="resumed")
                return success_response(data={"success": True, "message": "Training resumed"})
            if _auto_train_pause_event is None:
                return success_response(data={"success": False, "message": "No active training to resume"})
            if not _auto_train_pause_event.is_set():
                return success_response(data={"success": False, "message": "Training is not paused"})
            _auto_train_pause_event.clear()
            safe_audit_log("training.resume", resource=(self.state.config or {}).get("soul_name", ""), detail="resumed")
            return success_response(data={"success": True, "message": "Training resumed"})

        except Exception as e:
            classify_and_raise(e, source="autotrain.resume")
    async def status(self) -> dict:
        try:
            """status."""
            return success_response(data={"running": self.state.running, "config": self.state.config})

        except Exception as e:
            classify_and_raise(e, source="autotrain.status")
    async def stream(self, request: Request) -> AsyncGenerator[str, None]:
        try:
            """stream."""
            if not self.state.config:
                return StreamingResponse(
                    iter([sse_error("auto-train", "IDLE", "No training state - call /auto-train/start first", code="E_STATE_IDLE", http_status=409)]),
                    media_type="text/event-stream",
                )

            import asyncio as _asyncio
            from domains.infrastructure.task_queue import Task, Priority, get_task_queue

            queue: _asyncio.Queue[str] = _asyncio.Queue()
            loop = _asyncio.get_running_loop()

            def _enqueue(event_str: str) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, event_str)

            method = self.state.config.get("method", "slonet")
            task_type = "training-sessions" if method == "chat-trained" else "training"

            tq = get_task_queue()
            await tq.start()
            task = Task(
                name="auto-train",
                task_type=task_type,
                priority=Priority.HIGH,
                payload={**self.state.config, "checkpoint_dir": str(CHECKPOINTS_DIR)},
                timeout=3600,
            )
            task.metadata["sse_queue"] = queue
            task.metadata["enqueue"] = _enqueue

            _auto_train_cancel_event = threading.Event()
            _auto_train_pause_event = threading.Event()
            self.state.complete_enqueued = False
            self.state.running = True

            # Register with CancelManager
            _stream_cm_op_id: str | None = None
            try:
                from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
                _mgr = get_cancel_manager()
                _stream_cm_op_id = _mgr.register(
                    op_type=OpType.TRAINING,
                    label="auto-train",
                    cancel_fn=lambda: _auto_train_cancel_event.set(),
                )
                _mgr.start(_stream_cm_op_id)
            except Exception as e:
                autotrain_logger.warning("CancelManager registration failed (training may be unkillable): %s", e, extra={"tag": "TRAIN"})

            task_id = await tq.enqueue(task)
            autotrain_logger.info("Training task enqueued via task queue: %s", task_id, extra={"tag": "TRAIN"})

            from training.runtime import get_training_runtime
            runtime_job = {
                "id": task_id,
                "name": "auto-train",
                "model": method,
                "dataset": self.state.config.get("data_path") or "",
                "data_path": self.state.config.get("data_path") or "",
                "data_source": "auto-train",
                "status": "running",
                "progress": 0.0,
                "epochs": self.state.config.get("epochs"),
                "current_epoch": 0,
                "global_step": 0,
                "loss": None,
                "train_loss": None,
                "checkpoint": None,
                "checkpoint_dir": str(CHECKPOINTS_DIR),
                "error": None,
                "experiment_id": self.state.config.get("experiment_id"),
            }
            get_training_runtime().register(task_id, runtime_job, None, dict(self.state.config))

            async def event_generator() -> AsyncGenerator[str, None]:
                """event_generator."""
                global _auto_train_cancel_event, _auto_train_pause_event

                def _finish_cm(status: str, error: str = "") -> None:
                    if _stream_cm_op_id:
                        try:
                            from domains.infrastructure.cancel_manager import get_cancel_manager
                            get_cancel_manager().finish(_stream_cm_op_id, error=error if status != "complete" else "")
                        except Exception as exc:
                            autotrain_logger.debug("CancelManager.finish failed: %s", exc)

                deadline = time.time() + 3600
                heartbeat_interval = 10.0
                last_yield = time.time()
                try:
                    while True:
                        if time.time() > deadline:
                            autotrain_logger.error("Auto-train SSE timed out after 1 hour - no completion event received", extra={"tag": "TRAIN"})
                            yield sse_error("auto-train", "TIMEOUT", "Training SSE stream timed out", code="E_TIMEOUT", http_status=408)
                            _finish_cm("failed", "SSE timeout")
                            return
                        if await request.is_disconnected():
                            await tq.cancel(task_id)
                            _auto_train_cancel_event.set()
                            self.state.running = False
                            _finish_cm("cancelled", "client disconnected")
                            autotrain_logger.info("Client disconnected from auto-train stream", extra={"tag": "TRAIN"})
                            return
                        remaining = heartbeat_interval - (time.time() - last_yield)
                        if remaining <= 0:
                            remaining = heartbeat_interval
                        try:
                            event = await _asyncio.wait_for(queue.get(), timeout=remaining)
                        except _asyncio.TimeoutError:
                            yield ": heartbeat\n\n"
                            last_yield = time.time()
                            continue
                        yield event
                        last_yield = time.time()
                        if event.startswith("data: "):
                            try:
                                ev = json.loads(event[6:])
                                if ev.get("status") in ("complete", "error"):
                                    from training.runtime import get_training_runtime
                                    job = get_training_runtime().get(task_id)
                                    if job is not None:
                                        if ev.get("status") == "complete":
                                            job["status"] = "completed"
                                        else:
                                            job["status"] = "failed"
                                            job["error"] = str(ev.get("message") or ev.get("data") or "auto-train failed")
                                        try:
                                            def _glob_soul():
                                                return sorted(CHECKPOINTS_DIR.glob("*.soul"))
                                            soul_files = await asyncio.to_thread(_glob_soul)
                                            if soul_files:
                                                job["checkpoint"] = str(soul_files[-1])
                                        except Exception as exc:
                                            autotrain_logger.debug("Checkpoint glob failed: %s", exc)
                                        get_training_runtime().sync(task_id)
                                        _eid = self.state.config.get("experiment_id")
                                        if _eid and ev.get("status") == "complete":
                                            _fl = job.get("train_loss") or job.get("loss")
                                            if _fl is not None:
                                                _log_experiment_metric(_eid, "final_train_loss", float(_fl), int(job.get("global_step", 0)))
                                            _log_experiment_param(_eid, "epochs", self.state.config.get("epochs", 0))
                                            _log_experiment_param(_eid, "learning_rate", self.state.config.get("learning_rate", 0))
                                    _finish_cm(
                                        "complete" if ev.get("status") == "complete" else "failed",
                                        str(ev.get("message") or ev.get("data") or "") if ev.get("status") != "complete" else "",
                                    )
                                    break
                            except json.JSONDecodeError:
                                pass

                    while not queue.empty():
                        try:
                            extra = queue.get_nowait()
                            yield extra
                        except _asyncio.QueueEmpty:
                            break

                except TimeoutError:
                    autotrain_logger.error("Auto-train SSE queue timed out - no event for 60s", extra={"tag": "TRAIN"})
                    _finish_cm("failed", "SSE queue timeout")
                    yield sse_error("auto-train", "TIMEOUT", "No training progress for 60 seconds", code="E_TIMEOUT", http_status=408)
                except Exception as e:
                    autotrain_logger.error("Auto-train SSE stream error: %s", e, extra={"tag": "TRAIN"})
                    _finish_cm("failed", str(e))
                    if not self.state.complete_enqueued:
                        yield sse_error("auto-train", "FAILED", str(e), code="E_INFRA_GENERATION", http_status=500)
                finally:
                    _auto_train_cancel_event = None
                    _auto_train_pause_event = None
                    self.state.running = False
                    try:
                        import state as _srv_state
                        _srv_state.training_active = False
                    except Exception as exc:
                        autotrain_logger.debug("Failed to reset training_active: %s", exc)
                    from training.runtime import get_training_runtime
                    job = get_training_runtime().get(task_id)
                    if job is not None and job.get("status") not in ("completed", "failed", "cancelled"):
                        job["status"] = "interrupted"
                        job["error"] = job.get("error") or "Training stream ended before completion"
                        get_training_runtime().sync(task_id)

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        except Exception as e:
            classify_and_raise(e, source="autotrain.stream")
    async def list_checkpoints(self) -> dict:
        try:
            """list_checkpoints."""
            now = time.monotonic()
            if self._checkpoints_cache and (now - self._checkpoints_cache_ts) < 30:
                return success_response(data=self._checkpoints_cache[0])

            def _scan():
                def _stat_key(p: Path):
                    try:
                        return p.stat().st_mtime, p
                    except OSError:
                        return (0, p)

                checkpoints = []
                seen = set()

                for ext in ("*.soul", "*.slo"):
                    for f in sorted(self.CHECKPOINTS_DIR.glob(ext), key=_stat_key, reverse=True):
                        if f.name in seen:
                            continue
                        seen.add(f.name)
                        try:
                            st = f.stat()
                        except OSError:
                            continue
                        if f.suffix == ".soul" and st.st_size < 4096:
                            autotrain_logger.debug("Skipping corrupt header-only checkpoint: %s", f.name)
                            continue
                        info = self._load_soul_from_path(f, st)
                        if info:
                            checkpoints.append(info)

                for f in sorted(self.TURBO_DIR.glob("*.soul"), key=_stat_key, reverse=True):
                    if f.name in seen:
                        continue
                    seen.add(f.name)
                    try:
                        st = f.stat()
                    except OSError:
                        continue
                    info = self._load_soul_from_path(f, st)
                    if info:
                        info["source"] = "turbo"
                        checkpoints.append(info)

                for npz in sorted(self.LORA_DIR.glob("*.soul"), key=_stat_key, reverse=True):
                    if npz.name not in seen:
                        seen.add(npz.name)
                        info = self._load_lora_soul(npz.name)
                        if info:
                            checkpoints.append(info)

                for ckpt in checkpoints:
                    ckpt["description"] = _describe_checkpoint(ckpt)

                return checkpoints

            checkpoints = await asyncio.to_thread(_scan)

            self._checkpoints_cache = (checkpoints,)
            self._checkpoints_cache_ts = now

            return success_response(data=checkpoints)

        except Exception as e:
            classify_and_raise(e, source="autotrain.list_checkpoints")
    async def delete_checkpoint(self, name: str) -> dict:
        try:
            """delete_checkpoint."""
            if not re.match(r'^[\w\-]+(\.\w+)*$', name):
                raise_error(f"Invalid checkpoint name: {name!r}", code="E_VAL_REQUEST", details={"name": name})

            deleted = []

            def _delete():
                for base in (self.CHECKPOINTS_DIR, self.TURBO_DIR):
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

            if deleted:
                self._checkpoints_cache = None
                safe_audit_log("training.checkpoint.delete", resource=name, detail="deleted")
                return success_response(data={"name": deleted}, message="deleted")
            return success_response(data={"name": name}, message="not_found")

        except Exception as e:
            classify_and_raise(e, source="autotrain.delete_checkpoint")
    async def load_checkpoint(self, name: str) -> dict:
        """load_checkpoint."""
        from domains.training.slonet import import_from_sou
        from domains.models.provider import SloTransformerProvider, register_provider

        cp = await asyncio.to_thread(self._find_checkpoint, name)
        if cp is None:
            raise_error(f"Checkpoint not found: {name}", code="E_NOT_FOUND", details={"name": name})

        try:
            import asyncio as _aio

            def _load_meta():
                soul_net = import_from_sou(str(cp))
                import struct as _struct
                with open(str(cp), "rb") as f:
                    raw = f.read(12)
                    json_len = _struct.unpack("<I", raw[8:12])[0]
                    meta_bytes = f.read(json_len).rstrip(b"\x00")
                md = json.loads(meta_bytes.decode())
                return soul_net, md

            soul_net, md = await _aio.to_thread(_load_meta)
            soul_meta = soul_net.soul_signature()

            stoi = md.get("stoi") or md.get("metadata", {}).get("stoi")
            itos = md.get("itos") or md.get("metadata", {}).get("itos")
            if stoi is None or itos is None:
                raise_error(
                    "Checkpoint has no stoi/itos vocab - retrain to include vocab.",
                    code="E_VAL_FIELD",
                    details={"name": cp.name},
                )

            provider = SloTransformerProvider(
                model=soul_net,
                stoi=stoi,
                itos=itos,
                model_id_str=cp.stem,
            )
            register_provider("slonet", provider)
            register_provider("default", provider)

            autotrain_logger.info(
                "Loaded checkpoint %s as provider (vocab=%d, params=%d)",
                cp.name, len(stoi), soul_net.num_parameters(),
                extra={"tag": "TRAIN", "context": {"checkpoint": cp.name, "vocab_size": len(stoi), "params": soul_net.num_parameters()}},
            )

            safe_audit_log("training.checkpoint.load", resource=cp.name, detail=f"vocab={len(stoi)} params={soul_net.num_parameters()}")

            self._checkpoints_cache = None
            return success_response(data={
                "name": cp.name,
                "soul": soul_meta.get("soul_name", soul_net.soul_name),
                "loss": md.get("final_train_loss"),
                "steps": md.get("total_steps", 0),
                "traits": soul_meta.get("soul_traits", {}),
                "lineage": soul_net.lineage,
                "vocab_size": len(stoi),
                "params": soul_net.num_parameters(),
                "provider": "slonet",
            }, message="loaded")
        except Exception as e:
            import traceback
            autotrain_logger.error("Failed to load checkpoint %s: %s", cp.name, e, extra={"tag": "TRAIN", "context": {"checkpoint": cp.name, "error": str(e), "traceback": traceback.format_exc()}})
            classify_and_raise(e, source="load_checkpoint")

    async def download_checkpoint(self, name: str) -> dict:
        try:
            """download_checkpoint."""
            if not _VALID_CKPT_NAME.match(name) or '..' in name:
                raise_error("Invalid checkpoint name", "E_BAD_REQUEST", status_code=400)

            def _find():
                for d in (self.CHECKPOINTS_DIR, self.TURBO_DIR, self.LORA_DIR):
                    fp = (d / name).resolve()
                    if fp.exists() and fp.suffix in (".soul", ".slo") and str(fp).startswith(str(d.resolve())):
                        return str(fp)
                return None

            fp_str = await asyncio.to_thread(_find)
            if fp_str:
                return FileResponse(fp_str, media_type="application/octet-stream", filename=name)
            raise_error("Checkpoint not found", "E_NOT_FOUND", status_code=404)

        except Exception as e:
            classify_and_raise(e, source="autotrain.download_checkpoint")
    async def checkpoint_info(self, name: str) -> dict:
        try:
            """Read-only checkpoint metadata — does NOT load the model."""
            if not _VALID_CKPT_NAME.match(name) or '..' in name:
                raise_error("Invalid checkpoint name", "E_BAD_REQUEST", status_code=400)
            info = await asyncio.to_thread(self._load_soul, name)
            if not info or info.get("soul") == "unknown":
                raise_error("Checkpoint not found", "E_NOT_FOUND", status_code=404)
            return success_response(data=info)

        except Exception as e:
            classify_and_raise(e, source="autotrain.checkpoint_info")
    async def export_metrics(self) -> dict:
        try:
            """Export all checkpoint metrics as a downloadable JSON file."""
            from fastapi.responses import Response

            def _scan():
                checkpoints = []
                seen = set()

                for ext in ("*.soul", "*.slo"):
                    for f in sorted(self.CHECKPOINTS_DIR.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True):
                        if f.name in seen:
                            continue
                        seen.add(f.name)
                        if f.suffix == ".soul" and f.stat().st_size < 4096:
                            continue
                        info = self._load_soul(f.name)
                        if info:
                            checkpoints.append(info)

                for f in sorted(self.TURBO_DIR.glob("*.soul"), key=lambda p: p.stat().st_mtime, reverse=True):
                    if f.name in seen:
                        continue
                    seen.add(f.name)
                    info = self._load_soul(f.name)
                    if info:
                        checkpoints.append(info)

                for npz in sorted(self.LORA_DIR.glob("*.soul"), key=lambda p: p.stat().st_mtime, reverse=True):
                    if npz.name not in seen:
                        seen.add(npz.name)
                        info = self._load_lora_soul(npz.name)
                        if info:
                            checkpoints.append(info)

                return checkpoints

            checkpoints = await asyncio.to_thread(_scan)

            export = {
                "exported_at": time.time(),
                "total_checkpoints": len(checkpoints),
                "checkpoints": checkpoints,
            }

            content = json.dumps(export, indent=2, default=str)
            return Response(
                content=content,
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=training-metrics.json"},
            )

        except Exception as e:
            classify_and_raise(e, source="autotrain.export_metrics")
    async def export_checkpoint_mobile(self, name: str) -> dict:
        """export_checkpoint_mobile."""
        import numpy as np
        from domains.training.slonet import import_from_sou
        import base64

        def _find_ckpt():
            for d in (self.CHECKPOINTS_DIR, self.TURBO_DIR, self.LORA_DIR):
                fp = d / name
                if fp.exists() and fp.suffix == ".soul":
                    return str(fp)
            return None

        fp_str = await asyncio.to_thread(_find_ckpt)
        if not fp_str:
            raise_error("Checkpoint not found", "E_NOT_FOUND", status_code=404)
        fp = Path(fp_str)

        try:
            net = import_from_sou(str(fp))
        except Exception as e:
            autotrain_logger.warning("Export checkpoint mobile failed: %s", e, extra={"tag": "TRAIN"})
            classify_and_raise(e, source="export_checkpoint_mobile")

        sd = net.state_dict()
        n_embed = net.n_embed
        n_layer = net.n_layer
        n_head = net.n_head
        vocab_size = net.vocab_size
        block_size = getattr(net, 'block_size', 64)
        dim_ff = n_embed * 8 // 3
        dim_ff = ((dim_ff + 63) // 64) * 64

        weights = []
        def _push(name):
            arr = sd.get(name)
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

        config = {
            "vocab_size": vocab_size,
            "n_embed": n_embed,
            "n_layer": n_layer,
            "n_head": n_head,
            "block_size": block_size,
            "num_weights": len(weights),
        }

        return success_response(data={"config": config, "weights_b64": weights_b64})

    async def auto_train_log(self) -> dict:
        try:
            """auto_train_log."""
            from domains.infrastructure.output_buffer import get_server_buffer
            lines = [line.text for line in get_server_buffer().tail(200)]
            return success_response(data={"lines": lines, "total": len(lines)})

        except Exception as e:
            classify_and_raise(e, source="autotrain.auto_train_log")
    async def start_from_sessions(self, req: FromSessionsRequest) -> dict:
        try:
            """start_from_sessions."""
            if self.state.running:
                raise_error("Training already in progress", code="E_INFRA_BUSY")

            self.state.running = True
            self.state.config = {
                "method": "from-sessions",
                "epochs": req.epochs,
                "learning_rate": req.learning_rate,
                "batch_size": req.batch_size,
                "n_embed": req.n_embed,
                "n_layer": req.n_layer,
                "n_head": req.n_head,
                "block_size": req.block_size,
                "dropout": req.dropout,
                "soul_name": req.soul_name,
                "min_pair_quality": req.min_pair_quality,
                "max_pairs": req.max_pairs,
                "checkpoint_name": req.checkpoint_name,
                "session_ids": req.session_ids,
                "experiment_id": req.experiment_id,
                "started_at": time.time(),
            }
            safe_audit_log("training.start", resource=req.soul_name or "from-sessions", detail="from-sessions", session_ids=len(req.session_ids) if req.session_ids else 0, epochs=req.epochs)
            return success_response(data=self.state.config, message="Training started")

        except Exception as e:
            classify_and_raise(e, source="autotrain.start_from_sessions")
    async def stream_from_sessions(self, request: Request) -> AsyncGenerator[str, None]:
        try:
            """stream_from_sessions."""
            if not self.state.config or self.state.config.get("method") != "from-sessions":
                return StreamingResponse(
                    iter([sse_error("auto-train", "IDLE", "No training state - call /auto-train/from-sessions/start first", code="E_STATE_IDLE", http_status=409)]),
                    media_type="text/event-stream",
                )

            import asyncio as _asyncio
            from domains.infrastructure.task_queue import Task, Priority, get_task_queue

            queue: _asyncio.Queue[str] = _asyncio.Queue()
            loop = _asyncio.get_running_loop()

            def _enqueue(event_str: str) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, event_str)

            tq = get_task_queue()
            await tq.start()
            task = Task(
                name="auto-train-sessions",
                task_type="training-sessions",
                priority=Priority.HIGH,
                payload={**self.state.config, "checkpoint_dir": str(CHECKPOINTS_DIR)},
                timeout=3600,
            )
            task.metadata["sse_queue"] = queue
            task.metadata["enqueue"] = _enqueue

            global _auto_train_cancel_event
            if _auto_train_cancel_event is None:
                _auto_train_cancel_event = threading.Event()
            global _auto_train_pause_event
            if _auto_train_pause_event is None:
                _auto_train_pause_event = threading.Event()
            self.state.complete_enqueued = False
            self.state.running = True

            task_id = await tq.enqueue(task)
            autotrain_logger.info("From-sessions training task enqueued: %s", task_id, extra={"tag": "TRAIN"})

            # Register with CancelManager
            _sess_cm_op_id: str | None = None
            try:
                from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
                _mgr = get_cancel_manager()
                _sess_cm_op_id = _mgr.register(
                    op_type=OpType.TRAINING,
                    label="auto-train-sessions",
                    cancel_fn=lambda: _auto_train_cancel_event.set() if _auto_train_cancel_event else None,
                )
                _mgr.start(_sess_cm_op_id)
            except Exception as exc:
                autotrain_logger.warning("CancelManager registration failed for from-sessions: %s", exc)

            from training.runtime import get_training_runtime
            runtime_job = {
                "id": task_id,
                "name": "auto-train-sessions",
                "model": "sloughgpt",
                "dataset": "",
                "data_path": "",
                "data_source": "from-sessions",
                "status": "running",
                "progress": 0.0,
                "epochs": self.state.config.get("epochs"),
                "current_epoch": 0,
                "global_step": 0,
                "loss": None,
                "train_loss": None,
                "checkpoint": None,
                "checkpoint_dir": str(CHECKPOINTS_DIR),
                "error": None,
                "experiment_id": self.state.config.get("experiment_id"),
            }
            get_training_runtime().register(task_id, runtime_job, None, dict(self.state.config))

            async def event_generator() -> AsyncGenerator[str, None]:
                """event_generator."""
                global _auto_train_cancel_event

                def _finish_cm(status: str, error: str = "") -> None:
                    if _sess_cm_op_id:
                        try:
                            from domains.infrastructure.cancel_manager import get_cancel_manager
                            get_cancel_manager().finish(_sess_cm_op_id, error=error if status != "complete" else "")
                        except Exception as exc:
                            autotrain_logger.debug("CancelManager.finish failed: %s", exc)

                deadline = time.time() + 3600
                heartbeat_interval = 10.0
                last_yield = time.time()
                try:
                    while True:
                        if time.time() > deadline:
                            _finish_cm("failed", "SSE timeout")
                            yield sse_error("auto-train", "TIMEOUT", "Training SSE stream timed out", code="E_TIMEOUT", http_status=408)
                            return
                        if await request.is_disconnected():
                            await tq.cancel(task_id)
                            _auto_train_cancel_event.set()
                            self.state.running = False
                            _finish_cm("cancelled", "client disconnected")
                            autotrain_logger.info("Client disconnected from from-sessions stream", extra={"tag": "TRAIN"})
                            return
                        remaining = heartbeat_interval - (time.time() - last_yield)
                        if remaining <= 0:
                            remaining = heartbeat_interval
                        try:
                            event = await _asyncio.wait_for(queue.get(), timeout=remaining)
                        except _asyncio.TimeoutError:
                            yield ": heartbeat\n\n"
                            last_yield = time.time()
                            continue
                        yield event
                        last_yield = time.time()
                        if event.startswith("data: "):
                            try:
                                ev = json.loads(event[6:])
                                if ev.get("status") in ("complete", "error"):
                                    from training.runtime import get_training_runtime
                                    job = get_training_runtime().get(task_id)
                                    if job is not None:
                                        if ev.get("status") == "complete":
                                            job["status"] = "completed"
                                        else:
                                            job["status"] = "failed"
                                            job["error"] = str(ev.get("message") or ev.get("data") or "auto-train failed")
                                        try:
                                            def _glob_soul():
                                                return sorted(CHECKPOINTS_DIR.glob("*.soul"))
                                            soul_files = await asyncio.to_thread(_glob_soul)
                                            if soul_files:
                                                job["checkpoint"] = str(soul_files[-1])
                                        except Exception as exc:
                                            autotrain_logger.debug("Checkpoint glob failed: %s", exc)
                                        get_training_runtime().sync(task_id)
                                        _eid = self.state.config.get("experiment_id")
                                        if _eid and ev.get("status") == "complete":
                                            _fl = job.get("train_loss") or job.get("loss")
                                            if _fl is not None:
                                                _log_experiment_metric(_eid, "final_train_loss", float(_fl), int(job.get("global_step", 0)))
                                            _log_experiment_param(_eid, "epochs", self.state.config.get("epochs", 0))
                                            _log_experiment_param(_eid, "learning_rate", self.state.config.get("learning_rate", 0))
                                    _finish_cm(
                                        "complete" if ev.get("status") == "complete" else "failed",
                                        str(ev.get("message") or ev.get("data") or "") if ev.get("status") != "complete" else "",
                                    )
                                    break
                            except json.JSONDecodeError:
                                pass

                    while not queue.empty():
                        try:
                            extra = queue.get_nowait()
                            yield extra
                        except _asyncio.QueueEmpty:
                            break

                except TimeoutError:
                    _finish_cm("failed", "SSE queue timeout")
                    yield sse_error("auto-train", "TIMEOUT", "No training progress for 60 seconds", code="E_TIMEOUT", http_status=408)
                except Exception as e:
                    autotrain_logger.error("From-sessions SSE stream error: %s", e, extra={"tag": "TRAIN"})
                    _finish_cm("failed", str(e))
                    if not self.state.complete_enqueued:
                        yield sse_error("auto-train", "FAILED", str(e), code="E_INFRA_GENERATION", http_status=500)
                finally:
                    _auto_train_cancel_event = None
                    self.state.running = False
                    from training.runtime import get_training_runtime
                    job = get_training_runtime().get(task_id)
                    if job is not None and job.get("status") not in ("completed", "failed", "cancelled"):
                        job["status"] = "interrupted"
                        job["error"] = job.get("error") or "Training stream ended before completion"
                        get_training_runtime().sync(task_id)

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        except Exception as e:
            classify_and_raise(e, source="autotrain.stream_from_sessions")
    async def cancel_from_sessions(self) -> dict:
        try:
            """cancel_from_sessions."""
            if _auto_train_cancel_event is not None:
                _auto_train_cancel_event.set()
            self.state.running = False
            try:
                from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
                get_cancel_manager().cancel_all(op_type=OpType.TRAINING)
            except Exception as exc:
                autotrain_logger.warning("CancelManager.cancel_all failed for from-sessions: %s", exc)
            safe_audit_log("training.stop", resource=(self.state.config or {}).get("soul_name", ""), detail="cancelled")
            return success_response(message="Cancel signal sent")

        except Exception as e:
            classify_and_raise(e, source="autotrain.cancel_from_sessions")
    def _load_soul_from_path(self, ckpt_file: Path, st: Any = None) -> dict:
        if st is None:
            st = ckpt_file.stat()
        size_mb = round(st.st_size / (1024 * 1024), 2)
        meta = _load_soul_meta_cached(ckpt_file)

        if meta:
            m = meta.get("metadata", {})
            raw_soul = (meta.get("soul_name") or meta.get("soul") or meta.get("name") or "unknown")
            soul = raw_soul.replace("-soul", "")
            if ckpt_file.suffix == ".soul" and (soul == ckpt_file.stem or soul == ckpt_file.name):
                soul = "unknown"
            return {
                "name": ckpt_file.name,
                "download_url": f"/auto-train/checkpoints/{ckpt_file.name}/download",
                "soul": soul,
                "loss": m.get("avg_loss", meta.get("final_train_loss")),
                "steps": m.get("steps", 0),
                "epochs": m.get("step", meta.get("epochs_trained", 0)),
                "traits": meta.get("personality_traits", meta.get("traits", {})),
                "lineage": meta.get("lineage", "slonet"),
                "model_type": meta.get("model_type", "slonet"),
                "size_mb": size_mb,
                "tokenizer_type": m.get("tokenizer_type", "char"),
                "vocab_size": m.get("vocab_size", meta.get("vocab_size", 0)),
                "avg_quality": meta.get("avg_quality"),
                **{k: meta[k] for k in ("tagline", "description", "born_at", "epochs_trained",
                   "final_train_loss", "final_val_loss", "system_prompt",
                   "tags", "base_model", "training_dataset", "personality",
                   "training_duration_s")
                   if k in meta and meta[k]},
            }

        return {"name": ckpt_file.name, "soul": "unknown", "size_mb": size_mb}

    def _load_soul(self, name: str) -> dict:
        ckpt_file = self._find_checkpoint(name)
        if ckpt_file is None:
            return {"name": name, "soul": "unknown"}
        return self._load_soul_from_path(ckpt_file)

    def _load_lora_soul(self, name: str) -> dict | None:
        candidate = None
        for lora_dir in (self.LORA_DIR, self.CHECKPOINTS_DIR):
            for ext in ("", ".soul"):
                if not name.endswith(".soul"):
                    path = lora_dir / (name + ext)
                else:
                    path = lora_dir / name
                if path.exists():
                    candidate = path
                    break
            if candidate is not None:
                break

        if candidate is None:
            return None

        size_mb = round(candidate.stat().st_size / (1024 * 1024), 2)
        meta = _load_soul_meta(candidate)

        if meta:
            m = meta.get("metadata", {})
            raw_soul = (meta.get("soul_name") or meta.get("soul") or meta.get("name") or "unknown")
            soul_name = raw_soul.replace("-soul", "")
            return {
                "name": candidate.name,
                "download_url": f"/auto-train/checkpoints/{candidate.name}/download",
                "soul": soul_name,
                "loss": m.get("avg_loss", meta.get("final_train_loss")),
                "steps": m.get("steps", 0),
                "epochs": 0,
                "traits": meta.get("personality_traits", meta.get("traits", {})),
                "lineage": meta.get("lineage", "lora-feedback"),
                "model_type": "lora",
                "verdict": meta.get("eval_verdict"),
                "perplexity_delta": meta.get("perplexity_delta"),
                "bleu_delta": meta.get("bleu_delta"),
                "size_mb": size_mb,
                "avg_quality": meta.get("avg_quality"),
                **{k: meta[k] for k in ("tagline", "description", "born_at", "system_prompt", "tags")
                   if k in meta and meta[k]},
            }

        return {"name": candidate.name, "soul": "unknown", "model_type": "lora", "size_mb": size_mb}


_auto_train_instance = AutoTrainRouter()
router = _auto_train_instance.router
state = _auto_train_instance.state

def stream(request) -> dict:
    """stream."""
    return _auto_train_instance.stream(request)

def start_from_sessions(req) -> dict:
    """start_from_sessions."""
    return _auto_train_instance.start_from_sessions(req)

def cancel_from_sessions() -> dict:
    """cancel_from_sessions."""
    return _auto_train_instance.cancel_from_sessions()

def _load_soul(name):
    return _auto_train_instance._load_soul(name)

def _load_lora_soul(name):
    return _auto_train_instance._load_lora_soul(name)

def stream_from_sessions(request) -> dict:
    """stream_from_sessions."""
    return _auto_train_instance.stream_from_sessions(request)