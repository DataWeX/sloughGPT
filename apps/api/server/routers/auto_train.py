"""
Auto-Train Router — SloNet LM Training Pipeline (DEPRECATED)

.. deprecated::
    All ``/auto-train/*`` routes are deprecated. Use ``/training/*`` instead.
    This module exists for backward compatibility only and will be removed in a
    future release. Frontend, CLI, SDK, and mobile clients have been migrated.

    New endpoints live in ``training/execution.py``, ``training/jobs_api.py``,
    ``training/control.py``, and ``training/router.py``.

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
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, FileResponse

from infrastructure.auth import require_auth_if_enabled
from fastapi import Request
from pydantic import BaseModel, Field
import json
import logging
import math
import re
import time

from schemas.common import success_response, raise_error, classify_and_raise, safe_audit_log

from training.runtime import get_training_runtime

from domains.training.service import (
    log_experiment_metric as _log_experiment_metric,
    log_experiment_param as _log_experiment_param,
    parse_subtitle_text as _parse_subtitle_text,
    resolve_dataset_path as _resolve_dataset_path,
    build_soul_prompt as _build_soul_prompt,
    get_soul_name as _get_soul_name,
    get_soul_traits as _get_soul_traits,
    read_slo_json_header as _read_slo_json_header,
    describe_checkpoint as _describe_checkpoint,
    find_checkpoint as _service_find_checkpoint,
    list_checkpoints as _service_list_checkpoints,
    delete_checkpoint as _service_delete_checkpoint,
    load_checkpoint as _service_load_checkpoint,
    download_checkpoint_path as _service_download_checkpoint_path,
    checkpoint_info as _service_checkpoint_info,
    get_turbo_status as _service_get_turbo_status,
    get_log as _service_get_log,
    export_checkpoint_mobile as _service_export_checkpoint_mobile,
    start_turbo_training as _service_start_turbo_training,
    run_turbo_worker as _service_run_turbo_worker,
    start_from_sessions_training as _service_start_from_sessions_training,
    export_all_metrics as _service_export_all_metrics,
    process_training_completion as _service_process_completion,
    cleanup_stream_state as _service_cleanup_stream,
)

from infrastructure.sse_fallback import sse_event, sse_error, sse_complete


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
                size = f.stat().st_size
                f.unlink()
            except FileNotFoundError:
                continue
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
        return _service_find_checkpoint(name)

    def _register_routes(self):
        self.router.add_api_route("/start", self.start, methods=["POST"])
        self.router.add_api_route("/start-turbo", self.start_turbo, methods=["POST"])
        self.router.add_api_route("/turbo/status", self.turbo_status, methods=["GET"])
        self.router.add_api_route("/stop", self.stop, methods=["POST"])
        self.router.add_api_route("/pause", self.pause, methods=["POST"])
        self.router.add_api_route("/resume", self.resume, methods=["POST"])
        self.router.add_api_route("/status", self.status, methods=["GET"])
        self.router.add_api_route("/stream", self.stream, methods=["GET"], response_model=None)
        self.router.add_api_route("/checkpoints", self.list_checkpoints, methods=["GET"])
        self.router.add_api_route("/checkpoints/{name}", self.delete_checkpoint, methods=["DELETE"])
        self.router.add_api_route("/checkpoints/{name}/load", self.load_checkpoint, methods=["POST"])
        self.router.add_api_route("/checkpoints/{name}/download", self.download_checkpoint, methods=["GET"])
        self.router.add_api_route("/checkpoints/{name}/info", self.checkpoint_info, methods=["GET"])
        self.router.add_api_route("/checkpoints/{name}/export-mobile", self.export_checkpoint_mobile, methods=["GET"])
        self.router.add_api_route("/log", self.auto_train_log, methods=["GET"])
        self.router.add_api_route("/from-sessions/start", self.start_from_sessions, methods=["POST"])
        self.router.add_api_route("/from-sessions/stream", self.stream_from_sessions, methods=["GET"], response_model=None)
        self.router.add_api_route("/from-sessions/cancel", self.cancel_from_sessions, methods=["GET"])
        self.router.add_api_route("/metrics/export", self.export_metrics, methods=["GET"])

    async def start(self, req: StartRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
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
    async def start_turbo(self, req: TurboStartRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """start_turbo."""
            global _turbo_state, _turbo_cancel_event

            config = req.model_dump()
            job_info = await asyncio.to_thread(_service_start_turbo_training, config)

            _turbo_cancel_event = job_info["cancel_event"]
            _turbo_pause_event = job_info["pause_event"]

            if _auto_train_pgq is not None:
                _auto_train_pgq.submit_training(
                    self._run_turbo_pgq,
                    job_info["job_id"],
                    tree_id="auto-train",
                    config=config,
                )
            else:
                threading.Thread(
                    target=self._run_turbo,
                    args=(config,),
                    name=f"turbo-train-{job_info['job_id']}",
                    daemon=True,
                ).start()

            autotrain_logger.info(
                "Turbo training started in background: job_id=%s data=%s",
                job_info["job_id"], job_info["data_path"], extra={"tag": "TRAIN"},
            )
            return success_response(data={"status": "started", "job_id": job_info["job_id"], "message": "Turbo training started in background"})

        except Exception as e:
            classify_and_raise(e, source="autotrain.start_turbo")
    def _run_turbo(self, config: dict) -> None:
        """Run SloughGPTTrainer on a daemon thread via service layer."""
        _service_run_turbo_worker(config)

    def _run_turbo_pgq(
        self, job_id: str, tree_id: str, point_library: Any,
        is_cancelled: Any, *, config: dict,
    ) -> dict:
        """PGQ-compatible training entry point.

        Matches the ``PGQ.submit_training`` function signature:
        ``fn(job_id, tree_id, point_library, is_cancelled, **kwargs)``.

        Delegates to the service ``run_turbo_worker``.  Returns empty dict
        so ``TrainingExecutor`` can auto-compress weights into Points.
        """
        config["output_dir"] = str(self.TURBO_DIR)
        _service_run_turbo_worker(config)
        return {}

    async def turbo_status(self) -> dict:
        try:
            state = _service_get_turbo_status()
            return success_response(data=_finite_payload(state))
        except Exception as e:
            classify_and_raise(e, source="autotrain.turbo_status")
    async def stop(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
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
    async def pause(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
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
    async def resume(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
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
                                    _service_process_completion(ev, task_id, self.state.config, CHECKPOINTS_DIR, _finish_cm)
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
                    _service_cleanup_stream(task_id, self.state.config, self.state.__dict__, _finish_cm)
                    try:
                        import state as _srv_state
                        _srv_state.training_active = False
                    except Exception as exc:
                        autotrain_logger.debug("Failed to reset training_active: %s", exc)

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        except Exception as e:
            classify_and_raise(e, source="autotrain.stream")
    async def list_checkpoints(self) -> dict:
        try:
            now = time.monotonic()
            if self._checkpoints_cache and (now - self._checkpoints_cache_ts) < 30:
                return success_response(data=self._checkpoints_cache[0])

            checkpoints = await _service_list_checkpoints()

            self._checkpoints_cache = (checkpoints,)
            self._checkpoints_cache_ts = now

            return success_response(data=checkpoints)

        except Exception as e:
            classify_and_raise(e, source="autotrain.list_checkpoints")
    async def delete_checkpoint(self, name: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            deleted = await _service_delete_checkpoint(name)

            if deleted:
                self._checkpoints_cache = None
                safe_audit_log("training.checkpoint.delete", resource=name, detail="deleted")
                return success_response(data={"name": deleted}, message="deleted")
            return success_response(data={"name": name}, message="not_found")

        except Exception as e:
            classify_and_raise(e, source="autotrain.delete_checkpoint")
    async def load_checkpoint(self, name: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            data = await _service_load_checkpoint(name)

            autotrain_logger.info(
                "Loaded checkpoint %s as provider (vocab=%d, params=%d)",
                data["name"], data["vocab_size"], data["params"],
                extra={"tag": "TRAIN", "context": {"checkpoint": data["name"], "vocab_size": data["vocab_size"], "params": data["params"]}},
            )

            safe_audit_log("training.checkpoint.load", resource=data["name"], detail=f"vocab={data['vocab_size']} params={data['params']}")

            self._checkpoints_cache = None
            return success_response(data=data, message="loaded")
        except Exception as e:
            import traceback
            autotrain_logger.error("Failed to load checkpoint %s: %s", name, e, extra={"tag": "TRAIN", "context": {"checkpoint": name, "error": str(e), "traceback": traceback.format_exc()}})
            classify_and_raise(e, source="load_checkpoint")

    async def download_checkpoint(self, name: str) -> dict:
        try:
            fp_str = await _service_download_checkpoint_path(name)
            if fp_str:
                return FileResponse(fp_str, media_type="application/octet-stream", filename=name)
            raise_error("Checkpoint not found", "E_NOT_FOUND", status_code=404)

        except Exception as e:
            classify_and_raise(e, source="autotrain.download_checkpoint")
    async def checkpoint_info(self, name: str) -> dict:
        try:
            info = await _service_checkpoint_info(name)
            return success_response(data=info)

        except Exception as e:
            classify_and_raise(e, source="autotrain.checkpoint_info")
    async def export_metrics(self) -> dict:
        try:
            from fastapi.responses import Response

            export = await _service_export_all_metrics()

            content = json.dumps(export, indent=2, default=str)
            return Response(
                content=content,
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=training-metrics.json"},
            )

        except Exception as e:
            classify_and_raise(e, source="autotrain.export_metrics")
    async def export_checkpoint_mobile(self, name: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            data = await _service_export_checkpoint_mobile(name)
            return success_response(data=data)
        except Exception as e:
            classify_and_raise(e, source="autotrain.export_checkpoint_mobile")

    async def auto_train_log(self) -> dict:
        try:
            lines = await _service_get_log()
            return success_response(data={"lines": lines, "total": len(lines)})
        except Exception as e:
            classify_and_raise(e, source="autotrain.auto_train_log")
    async def start_from_sessions(self, req: FromSessionsRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """start_from_sessions."""
            config = {
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
            }
            built_config = _service_start_from_sessions_training(self.state, config)
            safe_audit_log("training.start", resource=req.soul_name or "from-sessions", detail="from-sessions", session_ids=len(req.session_ids) if req.session_ids else 0, epochs=req.epochs)
            return success_response(data=built_config, message="Training started")

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
                                    _service_process_completion(ev, task_id, self.state.config, CHECKPOINTS_DIR, _finish_cm)
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
                    _service_cleanup_stream(task_id, self.state.config, self.state.__dict__, _finish_cm)

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        except Exception as e:
            classify_and_raise(e, source="autotrain.stream_from_sessions")
    async def cancel_from_sessions(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
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

def stream_from_sessions(request) -> dict:
    """stream_from_sessions."""
    return _auto_train_instance.stream_from_sessions(request)