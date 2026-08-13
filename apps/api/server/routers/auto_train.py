"""
Auto-Train Router — SloNet LM Training Pipeline

Trains a SloNet LSTM as a next-token-prediction language model on user-provided
text (source_text, dataset, or file).  Pure NumPy — no PyTorch dependency for
student training.  Exports checkpoints as .soul (binary float32 format).

Phase sequence: TRAINING -> COMPLETE | FAILED

Encapsulates router state in ``AutoTrainRouter`` class rather than module-level
mutable globals.
"""

from dataclasses import dataclass, field
import threading
from typing import Any, Dict, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import Request
from pydantic import BaseModel, Field
import json
import logging
import re
import time

from schemas.common import success_response, error_response

try:
    from domains.api.sse_envelope import sse_event, sse_error, sse_complete
except ImportError:
    def sse_event(stream, phase, status, data=None, meta=None, message=""):
        import json
        return "data: " + json.dumps({
            "stream": stream, "phase": phase, "status": status,
            "data": data or {}, "meta": meta or {}, "message": message
        }) + "\n\n"
    def sse_error(stream, phase, error, meta=None):
        return sse_event(stream, phase, "error", {"error": error}, meta or {}, f"Error: {error}")
    def sse_complete(stream, phase="COMPLETE", data=None, meta=None, message="Done"):
        return sse_event(stream, phase, "complete", data or {}, meta or {}, message)


@dataclass
class AutoTrainState:
    running: bool = False
    config: dict = field(default_factory=dict)
    student_net: Optional[object] = None
    student_tokenizer: Optional[object] = None


_auto_train_cancel_event: Optional[threading.Event] = None
_auto_train_pause_event: Optional[threading.Event] = None
_complete_enqueued = [False]

_turbo_lock = threading.Lock()
_turbo_cancel_event = threading.Event()
_turbo_state: Dict[str, Any] = {
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
    "result": None,
    "error": None,
}


class _AutoTrainCancelled(Exception):
    pass


REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "models" / "auto-training"
LORA_DIR = REPO_ROOT / "data" / "user_adapters"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHECKPOINT_DISK_MB = 500

autotrain_logger = logging.getLogger("slo.autotrain")
autotrain_logger.setLevel(logging.INFO)

SOU_MAGIC = b"SOUL"

_VALID_CKPT_NAME = re.compile(r'^[a-zA-Z0-9_\-\.]+$')


def _enforce_checkpoint_budget():
    try:
        files = []
        for ext in ("*.soul", "*.pt", "*.slo"):
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
    soul_name: str = "assistant"
    epochs: int = Field(default=20, ge=1, le=1000)
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1.0)
    batch_size: int = Field(default=16, ge=1, le=1024, description="Chunk size for training")
    source_text: Optional[str] = Field(default=None, description="Custom training text (SRT, plain, or lines). If provided, train on this instead of generating from teacher.")
    checkpoint_name: Optional[str] = Field(default=None, description="Load existing checkpoint and continue training")
    dataset_id: Optional[str] = Field(default=None, description="Dataset ID from /datasets to train on")
    early_stopping_patience: int = Field(default=5, ge=0, le=100, description="Stop if no eval improvement for N evaluations (0 = disabled)")
    # Native training architecture params (used when method=native)
    n_embed: int = Field(default=128, ge=16, le=1024, description="Embedding dimension for native training")
    n_layer: int = Field(default=4, ge=1, le=24, description="Number of transformer layers for native training")
    n_head: int = Field(default=4, ge=1, le=64, description="Number of attention heads for native training")
    block_size: int = Field(default=128, ge=8, le=2048, description="Context window size for native training")
    dropout: float = Field(default=0.1, ge=0.0, le=0.9, description="Dropout rate for native training")
    checkpoint_dir: Optional[str] = Field(default=None, description="Override checkpoint output directory (default: models/auto-training)")


class TurboStartRequest(BaseModel):
    method: str = Field(default="slonet", description="Training method: 'slonet', 'transformer', 'nanogpt', 'hf'")
    data_path: str = Field(default="", description="Path to training data file")
    dataset_id: Optional[str] = Field(default=None, description="Dataset ID to train on")
    epochs: int = Field(default=3, ge=1, le=1000)
    batch_size: int = Field(default=4, ge=1, le=256)
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1.0)
    vocab_size: int = Field(default=500, ge=50, le=50000)
    n_embed: int = Field(default=128, ge=16, le=1024)
    n_head: int = Field(default=4, ge=1, le=64)
    n_layer: int = Field(default=3, ge=1, le=24)
    block_size: int = Field(default=128, ge=8, le=2048)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    n_decoder_layers: Optional[int] = Field(default=None, description="Deprecated: use n_layer")
    max_tgt_len: Optional[int] = Field(default=None, description="Deprecated: use block_size")


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
    checkpoint_name: Optional[str] = Field(default=None, description="Resume from existing checkpoint")
    session_ids: Optional[list] = Field(default=None, description="If provided, only train from these session IDs")


def _resolve_dataset_path(dataset_id: str) -> str:
    ds_candidate = REPO_ROOT / "datasets" / dataset_id
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
        raw = path.read_bytes()
        if raw[:4] != SOU_MAGIC:
            return {}
        import struct
        json_len = struct.unpack("<I", raw[8:12])[0]
        return json.loads(raw[12:12+json_len].decode())
    except Exception:
        return {}


def _load_soul_meta(ckpt_file: Path) -> dict:
    meta_file = ckpt_file.with_suffix(ckpt_file.suffix + ".meta.json")
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text())
        except Exception:
            pass
    if ckpt_file.suffix == ".soul":
        return _read_slo_json_header(ckpt_file)
    if ckpt_file.suffix == ".pt":
        pt_meta = ckpt_file.with_suffix(".pt.meta.json")
        if pt_meta.exists():
            try:
                return json.loads(pt_meta.read_text())
            except Exception:
                pass
    return {}


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


def _train_chat_trained(cfg_state, cancel_event, enqueue):
    from domains.training.chat_trainer import ChatTrainConfig, train_from_sessions

    resume_path = cfg_state.config.get("resume_path", "")
    soul_name = cfg_state.config.get("soul_name", "chat-trained")

    train_config = ChatTrainConfig(
        n_embed=cfg_state.config.get("n_embed", 128),
        n_layer=cfg_state.config.get("n_layer", 4),
        n_head=cfg_state.config.get("n_head", 4),
        block_size=cfg_state.config.get("block_size", 128),
        dropout=cfg_state.config.get("dropout", 0.1),
        epochs=cfg_state.config.get("epochs", 5),
        lr=cfg_state.config.get("learning_rate", 3e-4),
        batch_size=cfg_state.config.get("batch_size", 8),
        soul_name=soul_name,
        checkpoint_dir=str(CHECKPOINTS_DIR),
        session_ids=cfg_state.config.get("session_ids"),
        resume_checkpoint=resume_path if resume_path and resume_path.endswith(".soul") else None,
    )

    def _on_step(step, loss, epoch):
        if cancel_event is not None and cancel_event.is_set():
            raise _AutoTrainCancelled("Training cancelled by user")
        enqueue(sse_event(
            "auto-train", "TRAIN", "working",
            data={"step": step, "loss": loss, "done": False},
            meta={"epoch": epoch, "total_epochs": cfg_state.config.get("epochs", 5)},
        ))

    try:
        enqueue(sse_event(
            "auto-train", "PAIRS", "working",
            message="Extracting chat pairs from sessions...",
        ))

        model, metadata = train_from_sessions(
            config=train_config,
            on_step=_on_step,
            cancel_event=cancel_event,
        )

        _complete_enqueued[0] = True
        _enforce_checkpoint_budget()

        enqueue(sse_complete(
            "auto-train",
            data={
                "checkpoint": metadata.get("checkpoint", ""),
                "final_loss": metadata.get("final_loss"),
                "num_pairs": metadata.get("num_pairs", 0),
                "total_pairs": metadata.get("total_pairs", 0),
                "epochs": metadata.get("epochs_completed", 0),
                "vocab_size": metadata.get("vocab_size", 0),
                "perplexity": metadata.get("perplexity"),
                "samples": metadata.get("samples", []),
                "avg_response_len": metadata.get("avg_response_len", 0),
            },
            message=f"Training complete - {metadata.get('num_pairs', 0)} pairs, loss={metadata.get('final_loss')}",
        ))

    except _AutoTrainCancelled:
        enqueue(sse_complete("auto-train", data={"cancelled": True}, message="Training cancelled"))
    except Exception as e:
        autotrain_logger.error("Chat-trained worker error: %s", e, extra={"tag": "TRAIN"})
        enqueue(sse_error("auto-train", "FAILED", str(e)))
    finally:
        cfg_state.running = False


class AutoTrainRouter:
    def __init__(self):
        self.state = AutoTrainState()
        self.router = APIRouter(prefix="/auto-train", tags=["training"])
        self.REPO_ROOT = REPO_ROOT
        self.CHECKPOINTS_DIR = CHECKPOINTS_DIR
        self.LORA_DIR = LORA_DIR
        self._register_routes()

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

    async def start(self, req: StartRequest):
        if not req.source_text and not req.dataset_id and not req.checkpoint_name:
            return {"status": "error", "message": "Provide source_text, dataset_id, or checkpoint_name"}

        data_path = ""
        if req.source_text:
            source_lines = _parse_subtitle_text(req.source_text)
            if source_lines:
                tmp = self.REPO_ROOT / ".opencode" / "tmp" / f"autotrain_source_{int(time.time())}.txt"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text("\n".join(source_lines), encoding="utf-8")
                data_path = str(tmp)
                autotrain_logger.info("Wrote %d source lines to %s", len(source_lines), tmp, extra={"tag": "TRAIN", "context": {"source_lines": len(source_lines), "path": str(tmp)}})
        elif req.dataset_id:
            data_path = _resolve_dataset_path(req.dataset_id)

        resume = getattr(req, "resume", False)
        resume_path = getattr(req, "resume_path", "")
        method = "slonet"

        if not resume and req.checkpoint_name:
            ckpt_soul = self.CHECKPOINTS_DIR / f"{req.checkpoint_name}.soul"
            ckpt_pt = self.CHECKPOINTS_DIR / f"{req.checkpoint_name}.pt"
            if ckpt_soul.exists():
                resume = True
                resume_path = str(ckpt_soul)
                method = "chat-trained"
                autotrain_logger.info("Auto-resume from %s (SloNet)", resume_path, extra={"tag": "TRAIN", "context": {"checkpoint": resume_path, "method": method}})
            elif ckpt_pt.exists():
                resume = True
                resume_path = str(ckpt_pt)
                if not data_path:
                    method = "chat-trained"
                    autotrain_logger.info("Checkpoint-only resume (no data): routing to chat_trainer", extra={"tag": "TRAIN", "context": {"checkpoint": resume_path}})
                else:
                    autotrain_logger.info("Auto-resume from %s", resume_path, extra={"tag": "TRAIN", "context": {"checkpoint": resume_path}})

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
        }
        self.state.running = True
        import state as _srv_state
        _srv_state.training_active = True
        autotrain_logger.info("Auto-train configured: data_path=%s epochs=%d", data_path, req.epochs, extra={"tag": "TRAIN"})
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log(
                "training.start",
                resource=req.dataset_id or req.checkpoint_name or (req.soul_name if req.source_text else "inline"),
                detail="resume" if resume else "fresh",
                extra={
                    "method": method,
                    "epochs": req.epochs,
                    "dataset_id": req.dataset_id or "",
                    "checkpoint_name": req.checkpoint_name or "",
                },
            )
        except Exception:
            pass
        return {"status": "ready", "data_path": data_path, "epochs": req.epochs, "config": self.state.config}

    async def start_turbo(self, req: TurboStartRequest):
        global _turbo_state, _turbo_cancel_event

        data_path = req.data_path
        if not data_path and req.dataset_id:
            data_path = _resolve_dataset_path(req.dataset_id)

        if not data_path:
            return {"status": "error", "message": "No data_path or dataset_id provided"}

        with _turbo_lock:
            if _turbo_state.get("status") == "running":
                return {"status": "error", "message": "A turbo training job is already running"}
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
                "result": None,
                "error": None,
            }
            _turbo_cancel_event = threading.Event()

        output_dir = Path(self.REPO_ROOT / "models" / "turbo-trained")
        output_dir.mkdir(parents=True, exist_ok=True)

        job_id = _turbo_state["job_id"]
        threading.Thread(
            target=self._run_turbo,
            args=(req, data_path, str(output_dir), job_id),
            name=f"turbo-train-{job_id}",
            daemon=True,
        ).start()

        autotrain_logger.info(
            "Turbo training started in background: job_id=%s data=%s",
            job_id, data_path, extra={"tag": "TRAIN"},
        )
        return {"status": "started", "job_id": job_id, "message": "Turbo training started in background"}

    def _run_turbo(self, req: TurboStartRequest, data_path: str, output_dir: str, job_id: str) -> None:
        """Run SloughGPTTrainer on a daemon thread, publishing progress to _turbo_state."""
        from domains.training.train_pipeline import SloughGPTTrainer

        def on_progress(info: Dict[str, Any]) -> None:
            with _turbo_lock:
                _turbo_state["global_step"] = int(info.get("global_step", _turbo_state["global_step"]))
                _turbo_state["total_steps"] = int(info.get("total_steps", _turbo_state["total_steps"]))
                _turbo_state["progress"] = float(info.get("progress_percent", 0))
                _turbo_state["loss"] = info.get("train_loss", _turbo_state["loss"])
                _turbo_state["learning_rate"] = info.get("learning_rate", _turbo_state["learning_rate"])
                _turbo_state["steps_per_sec"] = info.get("steps_per_sec", _turbo_state["steps_per_sec"])
                _turbo_state["eta_s"] = info.get("eta_s", _turbo_state["eta_s"])
                _turbo_state["elapsed_s"] = info.get("elapsed_s", _turbo_state["elapsed_s"])

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
                "Starting SloughGPTTrainer with method=%s data=%s",
                req.method, data_path, extra={"tag": "TRAIN"},
            )
            result = trainer.train(on_progress=on_progress, cancel_event=_turbo_cancel_event)
            autotrain_logger.info("SloughGPTTrainer result: %s", result, extra={"tag": "TRAIN"})

            if _turbo_cancel_event.is_set():
                with _turbo_lock:
                    _turbo_state["status"] = "error"
                    _turbo_state["error"] = "Training cancelled"
                return

            if isinstance(result, dict) and result.get("status") == "error":
                with _turbo_lock:
                    _turbo_state["status"] = "error"
                    _turbo_state["error"] = result.get("message") or "Training failed"
                return

            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log(
                    "training.start",
                    resource=data_path or req.dataset_id or "turbo",
                    detail="turbo",
                    extra={"method": req.method or "", "epochs": req.epochs},
                )
            except Exception:
                pass
            with _turbo_lock:
                _turbo_state["status"] = "complete"
                _turbo_state["result"] = result
                _turbo_state["progress"] = 100.0
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="auto_train_turbo")
            autotrain_logger.error("SloughGPTTrainer failed: %s", e, extra={"tag": "TRAIN"})
            with _turbo_lock:
                _turbo_state["status"] = "error"
                _turbo_state["error"] = str(e)

    async def turbo_status(self):
        """Return the current turbo training job progress."""
        with _turbo_lock:
            return dict(_turbo_state)

    async def stop(self):
        global _auto_train_cancel_event
        self.state.running = False
        if _auto_train_cancel_event is not None:
            _auto_train_cancel_event.set()
        _turbo_cancel_event.set()
        if _auto_train_cancel_event is not None:
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log(
                    "training.stop",
                    resource=(self.state.config or {}).get("soul_name") or "auto-train",
                    detail="cancelling",
                    extra={"method": (self.state.config or {}).get("method", "")},
                )
            except Exception:
                pass
            return {"status": "cancelling", "message": "Cancelling auto-training"}
        return {"status": "stopped"}

    async def pause(self):
        if _auto_train_pause_event is None:
            return {"success": False, "message": "No active training to pause"}
        if _auto_train_pause_event.is_set():
            return {"success": False, "message": "Training is already paused"}
        _auto_train_pause_event.set()
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log(
                "training.pause",
                resource=(self.state.config or {}).get("soul_name") or "auto-train",
            )
        except Exception:
            pass
        return {"success": True, "message": "Training paused"}

    async def resume(self):
        if _auto_train_pause_event is None:
            return {"success": False, "message": "No active training to resume"}
        if not _auto_train_pause_event.is_set():
            return {"success": False, "message": "Training is not paused"}
        _auto_train_pause_event.clear()
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log(
                "training.resume",
                resource=(self.state.config or {}).get("soul_name") or "auto-train",
            )
        except Exception:
            pass
        return {"success": True, "message": "Training resumed"}

    async def status(self):
        return {"running": self.state.running, "config": self.state.config}

    async def stream(self, request: Request):
        if not self.state.config:
            return StreamingResponse(
                iter([sse_error("auto-train", "IDLE", "No training state - call /auto-train/start first")]),
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
        _complete_enqueued[0] = False
        self.state.running = True

        task_id = await tq.enqueue(task)
        autotrain_logger.info("Training task enqueued via task queue: %s", task_id, extra={"tag": "TRAIN"})

        async def event_generator():
            global _auto_train_cancel_event, _auto_train_pause_event
            deadline = time.time() + 3600
            heartbeat_interval = 10.0
            last_yield = time.time()
            try:
                while True:
                    if time.time() > deadline:
                        autotrain_logger.error("Auto-train SSE timed out after 1 hour - no completion event received", extra={"tag": "TRAIN"})
                        yield sse_error("auto-train", "TIMEOUT", "Training SSE stream timed out")
                        return
                    if await request.is_disconnected():
                        await tq.cancel(task_id)
                        _auto_train_cancel_event.set()
                        self.state.running = False
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
                yield sse_error("auto-train", "TIMEOUT", "No training progress for 60 seconds")
            except Exception as e:
                from domains.infrastructure.errors import classify_exception, emit_error_event
                err = classify_exception(e)
                emit_error_event(err, source="auto_train_stream")
                autotrain_logger.error("Auto-train SSE stream error: %s", e, extra={"tag": "TRAIN"})
                if not _complete_enqueued[0]:
                    yield sse_error("auto-train", "FAILED", str(e))
            finally:
                _auto_train_cancel_event = None
                _auto_train_pause_event = None
                self.state.running = False
                try:
                    import state as _srv_state
                    _srv_state.training_active = False
                except Exception:
                    pass

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    async def list_checkpoints(self):
        checkpoints = []
        seen = set()

        for ext in ("*.soul", "*.pt", "*.slo"):
            for f in sorted(self.CHECKPOINTS_DIR.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True):
                if f.name in seen:
                    continue
                seen.add(f.name)
                if f.suffix == ".soul" and f.stat().st_size < 4096:
                    autotrain_logger.debug("Skipping corrupt header-only checkpoint: %s", f.name)
                    continue
                info = self._load_soul(f.name)
                if info:
                    checkpoints.append(info)

        for npz in sorted(self.LORA_DIR.glob("*.soul"), key=lambda p: p.stat().st_mtime, reverse=True):
            if npz.name not in seen:
                seen.add(npz.name)
                info = self._load_lora_soul(npz.name)
                if info:
                    checkpoints.append(info)

        for ckpt in checkpoints:
            ckpt["description"] = _describe_checkpoint(ckpt)

        return success_response(data=checkpoints)

    async def delete_checkpoint(self, name: str):
        deleted = []
        for ext in (".soul", ".pt", ".slo"):
            if name.endswith(ext):
                candidates = [self.CHECKPOINTS_DIR / name]
            else:
                candidates = [self.CHECKPOINTS_DIR / (name + ext)]
            for candidate in candidates:
                if candidate.exists():
                    candidate.unlink()
                    deleted.append(candidate.name)
                meta = Path(str(candidate) + ".meta.json")
                if meta.exists():
                    meta.unlink()

        if deleted:
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log(
                    "training.checkpoint.delete",
                    resource=name,
                    detail=",".join(deleted),
                )
            except Exception:
                pass
            return success_response(data={"name": deleted}, message="deleted")
        return success_response(data={"name": name}, message="not_found")

    async def load_checkpoint(self, name: str):
        from domains.training.slonet import import_from_sou
        from domains.models.provider import SloTransformerProvider, register_provider

        cp = self.CHECKPOINTS_DIR / name
        if not cp.exists():
            for ext in (".soul", ".pt"):
                candidate = self.CHECKPOINTS_DIR / (name + ext)
                if candidate.exists():
                    cp = candidate
                    break
        if not cp.exists():
            return error_response(message=f"Checkpoint not found: {name}", data={"name": name})

        try:
            soul_net = import_from_sou(str(cp))
            soul_meta = soul_net.soul_signature()

            import struct as _struct
            with open(str(cp), "rb") as f:
                raw = f.read(12)
                json_len = _struct.unpack("<I", raw[8:12])[0]
                meta_bytes = f.read(json_len).rstrip(b"\x00")
            md = json.loads(meta_bytes.decode())

            stoi = md.get("stoi") or md.get("metadata", {}).get("stoi")
            itos = md.get("itos") or md.get("metadata", {}).get("itos")
            if stoi is None or itos is None:
                return error_response(
                    message="Checkpoint has no stoi/itos vocab - retrain to include vocab.",
                    data={"name": cp.name},
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

            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log(
                    "training.checkpoint.load",
                    resource=cp.name,
                    detail="vocab=%d params=%d" % (len(stoi), soul_net.num_parameters()),
                )
            except Exception:
                pass

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
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="load_checkpoint")
            import traceback
            autotrain_logger.error("Failed to load checkpoint %s: %s", cp.name, e, extra={"tag": "TRAIN", "context": {"checkpoint": cp.name, "error": str(e), "traceback": traceback.format_exc()}})
            return error_response(message=str(e), data={"name": cp.name})

    async def download_checkpoint(self, name: str):
        if not _VALID_CKPT_NAME.match(name) or '..' in name:
            raise HTTPException(status_code=400, detail="Invalid checkpoint name")
        for d in (self.CHECKPOINTS_DIR, self.LORA_DIR):
            fp = (d / name).resolve()
            if fp.exists() and fp.suffix in (".soul", ".pt") and str(fp).startswith(str(d.resolve())):
                return FileResponse(str(fp), media_type="application/octet-stream", filename=name)
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    async def checkpoint_info(self, name: str):
        """Read-only checkpoint metadata — does NOT load the model."""
        if not _VALID_CKPT_NAME.match(name) or '..' in name:
            raise HTTPException(status_code=400, detail="Invalid checkpoint name")
        info = self._load_soul(name)
        if not info or info.get("soul") == "unknown":
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return success_response(data=info)

    async def export_metrics(self):
        """Export all checkpoint metrics as a downloadable JSON file."""
        import json
        from fastapi.responses import Response

        checkpoints = []
        seen = set()

        for ext in ("*.soul", "*.pt", "*.slo"):
            for f in sorted(self.CHECKPOINTS_DIR.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True):
                if f.name in seen:
                    continue
                seen.add(f.name)
                if f.suffix == ".soul" and f.stat().st_size < 4096:
                    continue
                info = self._load_soul(f.name)
                if info:
                    checkpoints.append(info)

        for npz in sorted(self.LORA_DIR.glob("*.soul"), key=lambda p: p.stat().st_mtime, reverse=True):
            if npz.name not in seen:
                seen.add(npz.name)
                info = self._load_lora_soul(npz.name)
                if info:
                    checkpoints.append(info)

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

    async def export_checkpoint_mobile(self, name: str):
        import numpy as np
        from domains.training.slonet import import_from_sou
        import base64
        import io
        import struct

        for d in (self.CHECKPOINTS_DIR, self.LORA_DIR):
            fp = d / name
            if fp.exists() and fp.suffix in (".soul", ".slo"):
                break
        else:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        try:
            net = import_from_sou(str(fp))
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="export_checkpoint_mobile")
            raise HTTPException(status_code=400, detail=f"Failed to load checkpoint: {e}")

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

        return {"config": config, "weights_b64": weights_b64}

    async def auto_train_log(self):
        from domains.infrastructure.output_buffer import get_server_buffer
        lines = [line.text for line in get_server_buffer().tail(200)]
        return {"lines": lines, "total": len(lines)}

    async def start_from_sessions(self, req: FromSessionsRequest):
        if self.state.running:
            return error_response("Training already in progress")

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
            "started_at": time.time(),
        }
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log(
                "training.start",
                resource=req.soul_name or "from-sessions",
                detail="from-sessions",
                extra={"session_ids": len(req.session_ids), "epochs": req.epochs},
            )
        except Exception:
            pass
        return success_response(data=self.state.config, message="Training started")

    async def stream_from_sessions(self, request: Request):
        if not self.state.config or self.state.config.get("method") != "from-sessions":
            return StreamingResponse(
                iter([sse_error("auto-train", "IDLE", "No training state - call /auto-train/from-sessions/start first")]),
                media_type="text/event-stream",
            )

        import asyncio as _asyncio
        from domains.infrastructure.task_queue import Task, Priority, get_task_queue

        queue: _asyncio.Queue[str] = _asyncio.Queue()
        loop = _asyncio.get_running_loop()

        def _enqueue(event_str: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event_str)

        tq = get_task_queue()
        task = Task(
            name="auto-train-sessions",
            task_type="training-sessions",
            priority=Priority.HIGH,
            payload={**self.state.config, "checkpoint_dir": str(CHECKPOINTS_DIR)},
            timeout=3600,
        )
        task.metadata["sse_queue"] = queue
        task.metadata["enqueue"] = _enqueue

        _auto_train_cancel_event = threading.Event()
        _complete_enqueued[0] = False
        self.state.running = True

        task_id = await tq.enqueue(task)
        autotrain_logger.info("From-sessions training task enqueued: %s", task_id, extra={"tag": "TRAIN"})

        async def event_generator():
            global _auto_train_cancel_event
            deadline = time.time() + 3600
            heartbeat_interval = 10.0
            last_yield = time.time()
            try:
                while True:
                    if time.time() > deadline:
                        yield sse_error("auto-train", "TIMEOUT", "Training SSE stream timed out")
                        return
                    if await request.is_disconnected():
                        await tq.cancel(task_id)
                        _auto_train_cancel_event.set()
                        self.state.running = False
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
                yield sse_error("auto-train", "TIMEOUT", "No training progress for 60 seconds")
            except Exception as e:
                from domains.infrastructure.errors import classify_exception, emit_error_event
                err = classify_exception(e)
                emit_error_event(err, source="auto_train_stream_turbo")
                if not _complete_enqueued[0]:
                    yield sse_error("auto-train", "FAILED", str(e))
            finally:
                _auto_train_cancel_event = None
                self.state.running = False

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    async def cancel_from_sessions(self):
        global _auto_train_cancel_event
        if _auto_train_cancel_event is not None:
            _auto_train_cancel_event.set()
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log(
                "training.stop",
                resource=(self.state.config or {}).get("soul_name") or "from-sessions",
                detail="cancelled",
            )
        except Exception:
            pass
        return success_response(message="Cancel signal sent")

    def _load_soul(self, name: str) -> dict:
        ckpt_file = self.CHECKPOINTS_DIR / name
        if not ckpt_file.exists():
            if name.endswith(".soul"):
                ckpt_file = self.CHECKPOINTS_DIR / name
            elif name.endswith(".pt"):
                ckpt_file = self.CHECKPOINTS_DIR / name
            else:
                for ext in (".soul", ".pt"):
                    candidate = self.CHECKPOINTS_DIR / (name + ext)
                    if candidate.exists():
                        ckpt_file = candidate
                        break
            if not ckpt_file.exists():
                return {"name": name, "soul": "unknown"}

        size_mb = round(ckpt_file.stat().st_size / (1024 * 1024), 2)
        meta = _load_soul_meta(ckpt_file)

        if meta:
            m = meta.get("metadata", {})
            raw_soul = (meta.get("soul_name") or meta.get("soul") or meta.get("name") or "unknown")
            soul = raw_soul.replace("-soul", "")
            if soul == ckpt_file.stem or soul == ckpt_file.name:
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
                **{k: meta[k] for k in ("tagline", "description", "born_at", "epochs_trained",
                   "final_train_loss", "final_val_loss", "system_prompt",
                   "tags", "base_model", "training_dataset", "personality",
                   "training_duration_s")
                   if k in meta and meta[k]},
            }

        try:
            if ckpt_file.suffix == ".soul":
                from domains.inference import load_soul
                soul, _ = load_soul(str(ckpt_file))
                soul_meta = getattr(soul, 'metadata', {})
                return {
                    "name": ckpt_file.name,
                    "download_url": f"/auto-train/checkpoints/{ckpt_file.name}/download",
                    "soul": _get_soul_name(soul),
                    "loss": getattr(soul, "final_train_loss", None) or soul_meta.get("avg_loss"),
                    "steps": soul_meta.get("steps", 0),
                    "epochs": getattr(soul, "epochs_trained", 0) or soul_meta.get("step", 0),
                    "traits": _get_soul_traits(soul),
                    "lineage": getattr(soul, "lineage", "slonet"),
                    "model_type": "slonet",
                    "size_mb": size_mb,
                    "tokenizer_type": soul_meta.get("tokenizer_type", "char"),
                    "vocab_size": soul_meta.get("vocab_size", 0),
                }
            elif ckpt_file.suffix == ".pt":
                import torch
                ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                result = {
                    "name": ckpt_file.name,
                    "download_url": f"/auto-train/checkpoints/{ckpt_file.name}/download",
                    "soul": ckpt.get("soul_name", "unknown"),
                    "loss": ckpt.get("train_loss"),
                    "steps": ckpt.get("steps", 0),
                    "epochs": ckpt.get("epochs", 0),
                    "traits": ckpt.get("personality_traits", {}),
                    "lineage": "legacy-pt",
                    "model_type": "lstm",
                    "size_mb": size_mb,
                    "tokenizer_type": "char",
                    "vocab_size": 0,
                }
                try:
                    meta_path = ckpt_file.with_suffix(".pt.meta.json")
                    meta_path.write_text(json.dumps(result, indent=2, default=str))
                except Exception:
                    pass
                return result
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="load_soul")
            autotrain_logger.warning("Failed to load %s: %s", ckpt_file, e, extra={"tag": "TRAIN", "context": {"checkpoint": str(ckpt_file), "error": str(e)}})

        return {"name": ckpt_file.name, "soul": "unknown", "size_mb": size_mb}

    def _load_lora_soul(self, name: str) -> Optional[dict]:
        from domains.inference import load_soul

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

        try:
            soul, _ = load_soul(str(candidate))
            soul_meta = getattr(soul, 'metadata', {})
            meta = _load_soul_meta(candidate)
            size_mb = round(candidate.stat().st_size / (1024 * 1024), 2)
            result = {
                "name": candidate.name,
                "download_url": f"/auto-train/checkpoints/{candidate.name}/download",
                "soul": _get_soul_name(soul),
                "loss": soul_meta.get("avg_loss"),
                "steps": soul_meta.get("steps", 0),
                "epochs": 0,
                "traits": _get_soul_traits(soul),
                "lineage": getattr(soul, 'lineage', None) or "lora-feedback",
                "model_type": "lora",
                "verdict": soul_meta.get("eval_verdict"),
                "perplexity_delta": soul_meta.get("perplexity_delta"),
                "bleu_delta": soul_meta.get("bleu_delta"),
                "size_mb": size_mb,
            }
            if meta:
                for k in ("tagline", "description", "born_at", "system_prompt", "tags"):
                    v = meta.get(k)
                    if v is not None and v != "":
                        result[k] = v
            return result
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="load_lora_soul")
            autotrain_logger.warning("Failed to load LoRA soul %s: %s", candidate, e, extra={"tag": "TRAIN", "context": {"candidate": str(candidate), "error": str(e)}})

        return None


_auto_train_instance = AutoTrainRouter()
router = _auto_train_instance.router
state = _auto_train_instance.state

def stream(request):
    return _auto_train_instance.stream(request)

def start_from_sessions(req):
    return _auto_train_instance.start_from_sessions(req)

def cancel_from_sessions():
    return _auto_train_instance.cancel_from_sessions()

def _load_soul(name):
    return _auto_train_instance._load_soul(name)

def _load_lora_soul(name):
    return _auto_train_instance._load_lora_soul(name)

def stream_from_sessions(request):
    return _auto_train_instance.stream_from_sessions(request)
