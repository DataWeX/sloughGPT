"""
Auto-Train Router — SloNet LM Training Pipeline

Trains a SloNet LSTM as a next-token-prediction language model on user-provided
text (source_text, dataset, or file).  Pure NumPy — no PyTorch dependency for
student training.  Exports checkpoints as .soul (binary float32 format).

Phase sequence: TRAINING → COMPLETE | FAILED

Encapsulates router state in ``AutoTrainState`` dataclass rather than module-level
mutable globals.
"""

from dataclasses import dataclass, field
import threading
from typing import Optional
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
    """Encapsulated mutable state for the auto-train router."""
    running: bool = False
    config: dict = field(default_factory=dict)


state = AutoTrainState()
_auto_train_cancel_event: Optional[threading.Event] = None
_auto_train_pause_event: Optional[threading.Event] = None
_complete_enqueued = [False]  # track if pipeline already sent a complete event

class _AutoTrainCancelled(Exception):
    """Raised inside the auto-train worker thread when user requests cancel."""

router = APIRouter(prefix="/auto-train", tags=["training"])

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "models" / "auto-training"
LORA_DIR = REPO_ROOT / "data" / "user_adapters"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHECKPOINT_DISK_MB = 500  # Global checkpoint disk budget

autotrain_logger = logging.getLogger("slo.autotrain")
autotrain_logger.setLevel(logging.INFO)
# Log handler is now the unified ServerOutputBuffer (wired via init_server_io)


def _enforce_checkpoint_budget():
    """Delete oldest checkpoints when total disk usage exceeds MAX_CHECKPOINT_DISK_MB.

    Exempts the most recently modified checkpoint and any loaded checkpoint.
    Operates on CHECKPOINTS_DIR only (not LORA_DIR).

    Side effects:
        - Deletes oldest .soul, .pt, .slo files and their .meta.json sidecars
    """
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
        # Sort oldest first
        files.sort(key=lambda p: p.stat().st_mtime)
        # Keep the most recent file
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
            # Delete sidecar
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
    """Parse SRT, VTT, or plain text into training lines.

    SRT format:
        1
        00:00:01,000 --> 00:00:04,000
        Text here

    VTT format:
        00:01.000 --> 00:04.000
        Text here

    Plain text: one line per training example.
    """
    lines = []

    # Try SRT format (look for timestamp pattern)
    srt_pattern = re.compile(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}')

    # Try VTT format
    vtt_pattern = re.compile(r'\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}\.\d{3}')

    if srt_pattern.search(text) or vtt_pattern.search(text):
        # Parse as subtitles
        for line in text.split('\n'):
            line = line.strip()
            # Skip timestamps and numbers
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
        # Plain text - treat each non-empty line as training example
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 2:
                lines.append(line)

    return lines


class StartRequest(BaseModel):
    teacher_model: str = Field(default="gpt2", deprecated="no longer used — kept for backward compat")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    soul_name: str = "assistant"
    epochs: int = Field(default=10, ge=1, le=1000)
    learning_rate: float = Field(default=0.001, ge=1e-5, le=1.0)
    batch_size: int = Field(default=64, ge=1, le=1024, description="Chunk size for training")
    source_text: Optional[str] = Field(default=None, description="Custom training text (SRT, plain, or lines). If provided, train on this instead of generating from teacher.")
    checkpoint_name: Optional[str] = Field(default=None, description="Load existing checkpoint and continue training")
    dataset_id: Optional[str] = Field(default=None, description="Dataset ID from /datasets to train on")
    early_stopping_patience: int = Field(default=0, ge=0, le=100, description="Stop if no eval improvement for N evaluations (0 = disabled)")


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
    # Legacy fields (ignored, kept for backward compat)
    n_encoder_layers: Optional[int] = Field(default=None, description="Deprecated: use n_layer")
    n_decoder_layers: Optional[int] = Field(default=None, description="Deprecated: use n_layer")
    dim_feedforward: Optional[int] = Field(default=None, description="Deprecated: ignored")
    max_src_len: Optional[int] = Field(default=None, description="Deprecated: ignored")
    max_tgt_len: Optional[int] = Field(default=None, description="Deprecated: use block_size")


def _resolve_dataset_path(dataset_id: str) -> str:
    """Resolve a dataset ID to a file path, checking common file names.

    Args:
        dataset_id: Dataset directory name under REPO_ROOT/datasets/

    Returns:
        File path string if found, empty string otherwise.
    """
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
    """Get canonical name from a SloProfile or SloNet-like object."""
    if hasattr(soul, 'name') and soul.name:
        return soul.name
    return getattr(soul, 'soul_name', 'unknown')


def _get_soul_traits(soul) -> dict:
    """Get personality traits from a SloProfile or SloNet-like object."""
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


SOU_MAGIC = b"SOUL"

def _read_slo_json_header(path: Path) -> dict:
    """Read only the JSON metadata header from a .soul file without loading full weights."""
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
    """Read checkpoint metadata from .soul.meta.json, falling back to .soul JSON header."""
    meta_file = ckpt_file.with_suffix(ckpt_file.suffix + ".meta.json")
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text())
        except Exception:
            pass
    if ckpt_file.suffix == ".soul":
        return _read_slo_json_header(ckpt_file)
    if ckpt_file.suffix == ".pt":
        # Try the .pt.meta.json sidecar written on previous load
        pt_meta = ckpt_file.with_suffix(".pt.meta.json")
        if pt_meta.exists():
            try:
                return json.loads(pt_meta.read_text())
            except Exception:
                pass
    return {}


def _describe_checkpoint(ckpt: dict) -> str:
    """Generate a plain-language description of a checkpoint.

    Translates raw metadata into sentences Alex can understand:
    "Trained on shakespeare for 5 epochs. Loss: 1.23 (good). Personality: warm."
    """
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
            parts.append(f"(loss {loss:.2f} — learned well)")
        elif loss < 3.0:
            parts.append(f"(loss {loss:.2f} — moderate)")
        else:
            parts.append(f"(loss {loss:.2f} — needs more training)")

    if traits:
        trait_names = list(traits.keys())[:3]
        if trait_names:
            parts.append(f"Personality: {', '.join(trait_names)}")

    if model_type and model_type not in ("slonet", "unknown"):
        parts.append(f"[{model_type}]")

    return " ".join(parts) + "."


def _load_soul(name: str) -> dict:
    """Load soul metadata from a .soul or .pt file. Prefers .soul.meta.json to avoid reading 80MB .soul files."""
    ckpt_file = CHECKPOINTS_DIR / name
    if not ckpt_file.exists():
        if name.endswith(".soul"):
            ckpt_file = CHECKPOINTS_DIR / name
        elif name.endswith(".pt"):
            ckpt_file = CHECKPOINTS_DIR / name
        else:
            for ext in (".soul", ".pt"):
                candidate = CHECKPOINTS_DIR / (name + ext)
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
               "tags", "base_model", "training_dataset", "personality")
               if k in meta and meta[k]},
        }

    # No .meta.json — fall back to loading the full .soul (slow, rare)
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
            # Cache metadata sidecar so future listings skip torch.load
            try:
                meta_path = ckpt_file.with_suffix(".pt.meta.json")
                meta_path.write_text(json.dumps(result, indent=2, default=str))
            except Exception:
                pass
            return result
    except Exception as e:
        autotrain_logger.warning("Failed to load %s: %s", ckpt_file, e, extra={"tag": "TRAIN", "context": {"checkpoint": str(ckpt_file), "error": str(e)}})

    return {"name": ckpt_file.name, "soul": "unknown", "size_mb": size_mb}


def _load_lora_soul(name: str) -> Optional[dict]:
    """Load soul metadata from a LoRA .soul checkpoint in the user_adapters directory."""
    from domains.inference import load_soul

    candidate = None
    for lora_dir in (LORA_DIR, CHECKPOINTS_DIR):
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
        autotrain_logger.warning("Failed to load LoRA soul %s: %s", candidate, e, extra={"tag": "TRAIN", "context": {"candidate": str(candidate), "error": str(e)}})

    return None


@router.post("/start")
async def start(req: StartRequest):
    """
    Configure auto-training — stores config for /auto-train/stream to consume.

    Delegates training to SloughGPTTrainer.

    Args:
        req: source_text or dataset_id, epochs, learning_rate, etc.

    Returns:
        dict with status and config summary

    Side effects:
        - Stores config in AutoTrainState for streaming worker
        - Writes source_text to temp file if provided
    """
    if not req.source_text and not req.dataset_id and not req.checkpoint_name:
        return {"status": "error", "message": "Provide source_text, dataset_id, or checkpoint_name"}

    data_path = ""
    if req.source_text:
        source_lines = _parse_subtitle_text(req.source_text)
        if source_lines:
            tmp = REPO_ROOT / ".opencode" / "tmp" / f"autotrain_source_{int(time.time())}.txt"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text("\n".join(source_lines), encoding="utf-8")
            data_path = str(tmp)
            autotrain_logger.info("Wrote %d source lines to %s", len(source_lines), tmp, extra={"tag": "TRAIN", "context": {"source_lines": len(source_lines), "path": str(tmp)}})
    elif req.dataset_id:
        data_path = _resolve_dataset_path(req.dataset_id)

    resume = getattr(req, "resume", False)
    resume_path = getattr(req, "resume_path", "")
    method = "slonet"  # default: SloughGPTTrainer

    if not resume and req.checkpoint_name:
        ckpt_soul = CHECKPOINTS_DIR / f"{req.checkpoint_name}.soul"
        ckpt_pt = CHECKPOINTS_DIR / f"{req.checkpoint_name}.pt"
        if ckpt_pt.exists():
            resume = True
            resume_path = str(ckpt_pt)
            autotrain_logger.info("Auto-resume from %s", resume_path, extra={"tag": "TRAIN", "context": {"checkpoint": resume_path}})
        elif ckpt_soul.exists():
            resume = True
            resume_path = str(ckpt_soul)
            method = "chat-trained"  # SloNet checkpoint — use chat_trainer
            autotrain_logger.info("Auto-resume from %s (SloNet)", resume_path, extra={"tag": "TRAIN", "context": {"checkpoint": resume_path, "method": method}})

    state.config = {
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
    }
    state.running = True
    import state as _srv_state
    _srv_state.training_active = True
    autotrain_logger.info("Auto-train configured: data_path=%s epochs=%d", data_path, req.epochs, extra={"tag": "TRAIN"})
    return {"status": "ready", "data_path": data_path, "epochs": req.epochs, "config": state.config}


@router.post("/start-turbo")
async def start_turbo(req: TurboStartRequest):
    """
    Start training using SloughGPTTrainer (decoder-only transformer).

    Args:
        req: TurboStartRequest with model architecture and training params

    Returns:
        dict with training result (status, model_path, final_loss, total_steps, epochs)
    """
    try:
        from domains.training.train_pipeline import SloughGPTTrainer

        data_path = req.data_path
        if not data_path and req.dataset_id:
            data_path = _resolve_dataset_path(req.dataset_id)

        if not data_path:
            return {"status": "error", "message": "No data_path or dataset_id provided"}

        # Map legacy fields: n_decoder_layers → n_layer, max_tgt_len → block_size
        n_layer = req.n_layer or req.n_decoder_layers or 3
        block_size = req.block_size or req.max_tgt_len or 128

        output_dir = Path(REPO_ROOT / "models" / "turbo-trained")
        output_dir.mkdir(parents=True, exist_ok=True)

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
            checkpoint_dir=str(output_dir),
        )

        autotrain_logger.info("Starting SloughGPTTrainer with method=%s data=%s", req.method, data_path, extra={"tag": "TRAIN"})
        result = trainer.train()
        autotrain_logger.info("SloughGPTTrainer result: %s", result, extra={"tag": "TRAIN"})

        return result
    except Exception as e:
        autotrain_logger.error("SloughGPTTrainer failed: %s", e, extra={"tag": "TRAIN"})
        return {"status": "error", "message": str(e)}


@router.post("/stop")
async def stop():
    global _auto_train_cancel_event
    state.running = False
    if _auto_train_cancel_event is not None:
        _auto_train_cancel_event.set()
        return {"status": "cancelling", "message": "Cancelling auto-training"}
    return {"status": "stopped"}


@router.post("/pause")
async def pause():
    """Pause the current auto-training run.

    Sets a pause_event that the training loop checks each step, sleeping
    until the event is cleared by a subsequent ``/resume`` call.
    """
    if _auto_train_pause_event is None:
        return {"success": False, "message": "No active training to pause"}
    if _auto_train_pause_event.is_set():
        return {"success": False, "message": "Training is already paused"}
    _auto_train_pause_event.set()
    return {"success": True, "message": "Training paused"}


@router.post("/resume")
async def resume():
    """Resume a paused auto-training run.

    Clears the pause_event so the training loop continues.
    """
    if _auto_train_pause_event is None:
        return {"success": False, "message": "No active training to resume"}
    if not _auto_train_pause_event.is_set():
        return {"success": False, "message": "Training is not paused"}
    _auto_train_pause_event.clear()
    return {"success": True, "message": "Training resumed"}


@router.get("/status")
async def status():
    """Get training status."""
    return {"running": state.running, "config": state.config}


def _train_chat_trained(cfg_state, cancel_event, enqueue):
    """Train using chat_trainer for .soul checkpoints.

    Resumes from an existing SloNet checkpoint and trains on session data.
    """
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
            message=f"Training complete — {metadata.get('num_pairs', 0)} pairs, loss={metadata.get('final_loss')}",
        ))

    except _AutoTrainCancelled:
        enqueue(sse_complete("auto-train", data={"cancelled": True}, message="Training cancelled"))
    except Exception as e:
        autotrain_logger.error("Chat-trained worker error: %s", e, extra={"tag": "TRAIN"})
        enqueue(sse_error("auto-train", "FAILED", str(e)))
    finally:
        cfg_state.running = False


@router.get("/stream")
async def stream(request: Request):
    """
    Stream auto-training as SSE via SloughGPTTrainer.

    Delegates model creation, training, and checkpoint export to the trainer.
    Phases: TRAIN → COMPLETE
    """
    if not state.config:
        return StreamingResponse(
            iter([sse_error("auto-train", "IDLE", "No training state — call /auto-train/start first")]),
            media_type="text/event-stream",
        )

    import asyncio as _asyncio
    queue: _asyncio.Queue[str] = _asyncio.Queue()
    loop = _asyncio.get_running_loop()

    def _enqueue(event_str: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event_str)

    def _training_worker():
        """Run SloughGPTTrainer in executor thread."""
        global _auto_train_cancel_event, _auto_train_pause_event
        from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig

        _auto_train_cancel_event = threading.Event()
        _auto_train_pause_event = threading.Event()
        _complete_enqueued[0] = False

        method = state.config.get("method", "slonet")

        if method == "chat-trained":
            _train_chat_trained(state, _auto_train_cancel_event, _enqueue)
            return

        data_path = state.config.get("data_path", "")
        output_dir = Path(CHECKPOINTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer_config = TrainerConfig(
            vocab_size=0,  # let prepare_data() determine from actual charset
            n_embed=state.config.get("n_embed", 64),
            n_layer=state.config.get("n_layer", 2),
            n_head=state.config.get("n_head", 4),
            block_size=state.config.get("block_size", 64),
            dropout=state.config.get("dropout", 0.1),
            batch_size=state.config.get("batch_size", 32),
            epochs=state.config.get("epochs", 10),
            learning_rate=state.config.get("learning_rate", 0.001),
            checkpoint_dir=str(output_dir),
            early_stopping_patience=state.config.get("early_stopping_patience", 0),
        )

        trainer = SloughGPTTrainer(
            data_path=data_path,
            config=trainer_config,
        )

        _auto_train_cancel_event = threading.Event()
        _auto_train_pause_event = threading.Event()
        _complete_enqueued[0] = False

        def _on_progress(info):
            if _auto_train_cancel_event is not None and _auto_train_cancel_event.is_set():
                raise _AutoTrainCancelled("Training cancelled by user")
            loss = info.get("train_loss")
            _enqueue(sse_event(
                "auto-train", "TRAIN", "working",
                data={
                    "progress": info.get("progress_percent", 0),
                    "loss": loss,
                    "eval_loss": info.get("eval_loss"),
                    "step": info.get("global_step", 0),
                    "learning_rate": info.get("learning_rate", 0),
                    "done": info.get("done", False),
                    "done_reason": info.get("done_reason"),
                },
                meta={
                    "epoch": info.get("epoch", 0),
                    "total_epochs": info.get("epochs", 0),
                },
            ))

        try:
            state.running = True
            result = trainer.train(
                on_progress=_on_progress,
                cancel_event=_auto_train_cancel_event,
                pause_event=_auto_train_pause_event,
                resume=state.config.get("resume", False),
                resume_path=state.config.get("resume_path", ""),
            )

            _complete_enqueued[0] = True
            if result.get("status") == "completed":
                ckpt = result.get("checkpoint", "")
                fl = result.get("final_loss")
                ts = result.get("total_steps", 0)
                best_path = result.get("model_path", "")
                epochs = result.get("epochs_completed", 0)
                autotrain_logger.info(
                    "Auto-train complete: checkpoint=%s final_loss=%s steps=%d epochs=%d best=%s",
                    ckpt, fl, ts, epochs, best_path, extra={"tag": "TRAIN", "context": {"checkpoint": ckpt, "final_loss": fl, "steps": ts, "epochs": epochs, "best_model_path": best_path}},
                )
                _enforce_checkpoint_budget()
                _enqueue(sse_complete("auto-train",
                    data={"checkpoint": ckpt, "final_loss": fl, "total_steps": ts, "epochs": epochs, "best_model_path": best_path},
                    message=f"Training complete — checkpoint={ckpt} loss={fl} steps={ts}"))
            else:
                autotrain_logger.warning("Auto-train result: %s", result.get("status"), extra={"tag": "TRAIN"})
                _enqueue(sse_error("auto-train", result.get("status", "FAILED"), result.get("error", "Unknown error")))

        except _AutoTrainCancelled:
            autotrain_logger.info("Auto-train cancelled by user", extra={"tag": "TRAIN", "context": {"action": "cancel"}})
            _enqueue(sse_complete("auto-train", data={"cancelled": True}, message="Training cancelled"))
        except Exception as e:
            autotrain_logger.error("Training worker error: %s", e, extra={"tag": "TRAIN", "context": {"error": str(e)}})
            _enqueue(sse_error("auto-train", "FAILED", str(e)))
        finally:
            _auto_train_cancel_event = None
            _auto_train_pause_event = None
            state.running = False
            try:
                import state as _srv_state
                _srv_state.training_active = False
            except Exception:
                pass

    async def event_generator():
        worker_task = loop.run_in_executor(None, _training_worker)
        deadline = time.time() + 3600  # 1-hour safety timeout
        heartbeat_interval = 10.0  # Send SSE comment every 10s to keep connection alive
        last_yield = time.time()
        try:
            while True:
                if time.time() > deadline:
                    autotrain_logger.error("Auto-train SSE timed out after 1 hour — no completion event received", extra={"tag": "TRAIN"})
                    yield sse_error("auto-train", "TIMEOUT", "Training SSE stream timed out")
                    return
                # Check for disconnect before each queue wait
                if await request.is_disconnected():
                    if _auto_train_cancel_event is not None:
                        _auto_train_cancel_event.set()
                    if _auto_train_pause_event is not None and _auto_train_pause_event.is_set():
                        _auto_train_pause_event.clear()
                    worker_task.cancel()
                    state.running = False
                    autotrain_logger.info("Client disconnected from auto-train stream", extra={"tag": "TRAIN"})
                    return
                # Wait for event or heartbeat timeout
                remaining = heartbeat_interval - (time.time() - last_yield)
                if remaining <= 0:
                    remaining = heartbeat_interval
                try:
                    event = await _asyncio.wait_for(queue.get(), timeout=remaining)
                except _asyncio.TimeoutError:
                    # No event within heartbeat interval — send SSE comment to keep alive
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

            # Drain any remaining events the worker may have enqueued
            while not queue.empty():
                try:
                    extra = queue.get_nowait()
                    yield extra
                except _asyncio.QueueEmpty:
                    break

            await worker_task
        except TimeoutError:
            autotrain_logger.error("Auto-train SSE queue timed out — no event for 60s", extra={"tag": "TRAIN"})
            yield sse_error("auto-train", "TIMEOUT", "No training progress for 60 seconds")
        except Exception as e:
            autotrain_logger.error("Auto-train SSE stream error: %s", e, extra={"tag": "TRAIN"})
            if not _complete_enqueued[0]:
                yield sse_error("auto-train", "FAILED", str(e))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/checkpoints")
async def list_checkpoints():
    """List all saved .soul checkpoints and LoRA adapters with soul metadata."""
    checkpoints = []
    seen = set()

    for ext in ("*.soul", "*.pt", "*.slo"):
        for f in sorted(CHECKPOINTS_DIR.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.name in seen:
                continue
            seen.add(f.name)
            # Skip truncated/corrupt header-only files (< 4 KB = header + no weights)
            if f.suffix == ".soul" and f.stat().st_size < 4096:
                autotrain_logger.debug("Skipping corrupt header-only checkpoint: %s", f.name)
                continue
            info = _load_soul(f.name)
            if info:
                checkpoints.append(info)

    for npz in sorted(LORA_DIR.glob("*.soul"), key=lambda p: p.stat().st_mtime, reverse=True):
        if npz.name not in seen:
            seen.add(npz.name)
            info = _load_lora_soul(npz.name)
            if info:
                checkpoints.append(info)

    # Add plain-language description to each checkpoint
    for ckpt in checkpoints:
        ckpt["description"] = _describe_checkpoint(ckpt)

    return success_response(data=checkpoints)


@router.delete("/checkpoints/{name}")
async def delete_checkpoint(name: str):
    """Delete a checkpoint file and its .meta.json sidecar.

    Accepts bare names (e.g., ``assistant_1781507107``) or names with extension
    (e.g., ``assistant_1781507107.soul``).  Searches all known extensions
    (``.soul``, ``.pt``, ``.slo``) and removes matching files + sidecars.
    """
    deleted = []
    for ext in (".soul", ".pt", ".slo"):
        if name.endswith(ext):
            candidates = [CHECKPOINTS_DIR / name]
        else:
            candidates = [CHECKPOINTS_DIR / (name + ext)]
        for candidate in candidates:
            if candidate.exists():
                candidate.unlink()
                deleted.append(candidate.name)
            meta = Path(str(candidate) + ".meta.json")
            if meta.exists():
                meta.unlink()

    if deleted:
        return success_response(data={"name": deleted}, message="deleted")
    return success_response(data={"name": name}, message="not_found")


@router.post("/checkpoints/{name}/load")
async def load_checkpoint(name: str):
    """Load a .soul checkpoint into the provider pipeline for chat.

    Loads the SloTransformer model, extracts stoi/itos vocab from metadata,
    creates a SloTransformerProvider, and registers it as the default text provider.

    Returns:
        dict with status, soul name, traits, and provider info
    """
    from domains.training.slonet import import_from_sou
    from domains.models.provider import SloTransformerProvider, register_provider

    # Find the checkpoint file
    cp = CHECKPOINTS_DIR / name
    if not cp.exists():
        # Try with/without extension
        for ext in (".soul", ".pt"):
            candidate = CHECKPOINTS_DIR / (name + ext)
            if candidate.exists():
                cp = candidate
                break
    if not cp.exists():
        return error_response(message=f"Checkpoint not found: {name}", data={"name": name})

    try:
        # Load the SloTransformer model from .soul file
        soul_net = import_from_sou(str(cp))
        soul_meta = soul_net.soul_signature()

        # Read raw JSON metadata from .soul header (stoi/itos stored at top level)
        import struct as _struct
        with open(str(cp), "rb") as f:
            raw = f.read(12)
            json_len = _struct.unpack("<I", raw[8:12])[0]
            meta_bytes = f.read(json_len).rstrip(b"\x00")
        md = json.loads(meta_bytes.decode())

        # Extract vocab from metadata — check top level and nested metadata dict
        stoi = md.get("stoi") or md.get("metadata", {}).get("stoi")
        itos = md.get("itos") or md.get("metadata", {}).get("itos")
        if stoi is None or itos is None:
            return error_response(
                message="Checkpoint has no stoi/itos vocab — retrain to include vocab.",
                data={"name": cp.name},
            )

        # Create provider and register as default
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
        return error_response(message=str(e), data={"name": cp.name})


@router.get("/log")
async def auto_train_log():
    """Return the auto-training log content from the shared server buffer."""
    from domains.infrastructure.output_buffer import get_server_buffer
    lines = [line.text for line in get_server_buffer().tail(200)]
    return {"lines": lines, "total": len(lines)}


@router.get("/checkpoints/{name}/export-mobile")
async def export_checkpoint_mobile(name: str):
    """Export a checkpoint as flat binary for on-device inference.

    Returns a JSON object with ``config`` (architecture) and ``weights_b64``
    (Base64-encoded float32 flat array).
    """
    import numpy as np
    from domains.training.slonet import import_from_sou
    import base64
    import io
    import struct

    for d in (CHECKPOINTS_DIR, LORA_DIR):
        fp = d / name
        if fp.exists() and fp.suffix in (".soul", ".slo"):
            break
    else:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    try:
        net = import_from_sou(str(fp))
    except Exception as e:
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


class FromSessionsRequest(BaseModel):
    """Configuration for on-device training from chat sessions."""
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


@router.post("/from-sessions/start")
async def start_from_sessions(req: FromSessionsRequest):
    """Start on-device training from chat sessions.

    Extracts (user, assistant) pairs from local session files, trains a small
    SloTransformer via next-token prediction, and exports a .soul checkpoint.

    Use ``GET /auto-train/from-sessions/stream`` to receive SSE progress events.
    """
    if state.running:
        return error_response("Training already in progress")

    state.running = True
    state.config = {
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
        "started_at": time.time(),
    }
    return success_response(data=state.config, message="Training started")


@router.get("/from-sessions/stream")
async def stream_from_sessions(request: Request):
    """Stream on-device chat training as SSE.

    Phases: PAIRS → TRAIN → COMPLETE | FAILED
    """
    if not state.config or state.config.get("method") != "from-sessions":
        return StreamingResponse(
            iter([sse_error("auto-train", "IDLE", "No training state — call /auto-train/from-sessions/start first")]),
            media_type="text/event-stream",
        )

    import asyncio as _asyncio
    queue: _asyncio.Queue[str] = _asyncio.Queue()
    loop = _asyncio.get_running_loop()

    def _enqueue(event_str: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event_str)

    def _training_worker():
        global _auto_train_cancel_event
        from domains.training.chat_trainer import ChatTrainConfig, train_from_sessions

        cfg = state.config
        train_config = ChatTrainConfig(
            n_embed=cfg.get("n_embed", 128),
            n_layer=cfg.get("n_layer", 4),
            n_head=cfg.get("n_head", 4),
            block_size=cfg.get("block_size", 128),
            dropout=cfg.get("dropout", 0.1),
            epochs=cfg.get("epochs", 5),
            lr=cfg.get("learning_rate", 3e-4),
            batch_size=cfg.get("batch_size", 8),
            min_pair_quality=cfg.get("min_pair_quality", 2.0),
            max_pairs=cfg.get("max_pairs", 500),
            soul_name=cfg.get("soul_name", "chat-trained"),
            checkpoint_dir=str(CHECKPOINTS_DIR),
            resume_checkpoint=cfg.get("checkpoint_name"),
        )

        _auto_train_cancel_event = threading.Event()
        _complete_enqueued[0] = False

        def _on_step(step: int, loss: float, epoch: int) -> None:
            if _auto_train_cancel_event is not None and _auto_train_cancel_event.is_set():
                raise _AutoTrainCancelled("Training cancelled by user")
            _enqueue(sse_event(
                "auto-train", "TRAIN", "working",
                data={"step": step, "loss": loss, "done": False},
                meta={"epoch": epoch, "total_epochs": cfg.get("epochs", 5)},
            ))

        try:
            _enqueue(sse_event(
                "auto-train", "PAIRS", "working",
                message="Extracting chat pairs from sessions...",
            ))

            model, metadata = train_from_sessions(
                config=train_config,
                on_step=_on_step,
                cancel_event=_auto_train_cancel_event,
            )

            _complete_enqueued[0] = True
            _enforce_checkpoint_budget()

            ckpt = metadata.get("checkpoint", "")
            fl = metadata.get("final_loss")
            pairs = metadata.get("num_pairs", 0)
            epochs_done = metadata.get("epochs_completed", 0)

            autotrain_logger.info(
                "From-sessions train complete: checkpoint=%s final_loss=%s pairs=%d epochs=%d",
                ckpt, fl, pairs, epochs_done,
                extra={"tag": "TRAIN"},
            )
            _enqueue(sse_complete(
                "auto-train",
                data={
                    "checkpoint": ckpt,
                    "final_loss": fl,
                    "num_pairs": pairs,
                    "total_pairs": metadata.get("total_pairs", 0),
                    "epochs": epochs_done,
                    "vocab_size": metadata.get("vocab_size", 0),
                    "train_losses": metadata.get("train_losses", []),
                    "val_losses": metadata.get("val_losses", []),
                    "perplexity": metadata.get("perplexity"),
                    "samples": metadata.get("samples", []),
                    "avg_response_len": metadata.get("avg_response_len", 0),
                },
                message=f"Training complete — {pairs} pairs, loss={fl}",
            ))

        except _AutoTrainCancelled:
            autotrain_logger.info("From-sessions training cancelled", extra={"tag": "TRAIN"})
            _enqueue(sse_complete("auto-train", data={"cancelled": True}, message="Training cancelled"))
        except Exception as e:
            autotrain_logger.error("From-sessions worker error: %s", e, extra={"tag": "TRAIN"})
            _enqueue(sse_error("auto-train", "FAILED", str(e)))
        finally:
            _auto_train_cancel_event = None
            state.running = False

    async def event_generator():
        worker_task = loop.run_in_executor(None, _training_worker)
        deadline = time.time() + 3600
        heartbeat_interval = 10.0
        last_yield = time.time()
        try:
            while True:
                if time.time() > deadline:
                    yield sse_error("auto-train", "TIMEOUT", "Training SSE stream timed out")
                    return
                if await request.is_disconnected():
                    if _auto_train_cancel_event is not None:
                        _auto_train_cancel_event.set()
                    worker_task.cancel()
                    state.running = False
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

            await worker_task
        except TimeoutError:
            yield sse_error("auto-train", "TIMEOUT", "No training progress for 60 seconds")
        except Exception as e:
            if not _complete_enqueued[0]:
                yield sse_error("auto-train", "FAILED", str(e))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/from-sessions/cancel")
async def cancel_from_sessions():
    """Cancel on-device training."""
    global _auto_train_cancel_event
    if _auto_train_cancel_event is not None:
        _auto_train_cancel_event.set()
    return success_response(message="Cancel signal sent")


@router.get("/checkpoints/{name}/download")
async def download_checkpoint(name: str):
    """Download a checkpoint .soul file for local (WebGPU) inference."""
    for d in (CHECKPOINTS_DIR, LORA_DIR):
        fp = d / name
        if fp.exists() and fp.suffix in (".soul", ".pt"):
            return FileResponse(str(fp), media_type="application/octet-stream", filename=name)
    raise HTTPException(status_code=404, detail="Checkpoint not found")
