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
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import json
import logging
import re
import time

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
from pathlib import Path

class _AutoTrainCancelled(Exception):
    """Raised inside the auto-train worker thread when user requests cancel."""

router = APIRouter(prefix="/auto-train", tags=["training"])

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "models" / "auto-training"
LORA_DIR = REPO_ROOT / "data" / "user_adapters"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)

autotrain_logger = logging.getLogger("autotrain")
autotrain_logger.setLevel(logging.INFO)


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
    algo: str = Field(default="bpe", description="Tokenization algorithm: 'bpe' (SloBPE) or 'unigram' (SloUnigram)")


class TurboStartRequest(BaseModel):
    method: str = Field(default="transformer", description="Training method: 'transformer', 'nanogpt', 'hf', 'slonet'")
    data_path: str = Field(default="", description="Path to training data file")
    dataset_id: Optional[str] = Field(default=None, description="Dataset ID to train on")
    epochs: int = Field(default=3, ge=1, le=1000)
    batch_size: int = Field(default=4, ge=1, le=256)
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1.0)
    vocab_size: int = Field(default=500, ge=50, le=50000)
    n_embed: int = Field(default=128, ge=16, le=1024)
    n_head: int = Field(default=4, ge=1, le=64)
    n_encoder_layers: int = Field(default=3, ge=1, le=24)
    n_decoder_layers: int = Field(default=3, ge=1, le=24)
    dim_feedforward: int = Field(default=256, ge=32, le=8192)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    max_src_len: int = Field(default=128, ge=8, le=2048)
    max_tgt_len: int = Field(default=128, ge=8, le=2048)


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


SOU_MAGIC = b"\x00SL\x0E"

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
            return {
                "name": ckpt_file.name,
                "download_url": f"/auto-train/checkpoints/{ckpt_file.name}/download",
                "soul": _get_soul_name(soul),
                "loss": getattr(soul, "final_train_loss", None) or soul.metadata.get("avg_loss"),
                "steps": soul.metadata.get("steps", 0),
                "epochs": getattr(soul, "epochs_trained", 0) or soul.metadata.get("step", 0),
                "traits": _get_soul_traits(soul),
                "lineage": soul.lineage,
                "model_type": "slonet",
                "size_mb": size_mb,
                "tokenizer_type": soul.metadata.get("tokenizer_type", "char"),
                "vocab_size": soul.metadata.get("vocab_size", 0),
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
        autotrain_logger.warning(f"Failed to load {ckpt_file}: {e}")

    return {"name": ckpt_file.name, "soul": "unknown", "size_mb": size_mb}


def _load_lora_soul(name: str) -> Optional[dict]:
    """Load soul metadata from a LoRA .soul checkpoint in the user_adapters directory."""
    from domains.inference import load_soul

    for lora_dir in (LORA_DIR, CHECKPOINTS_DIR):
        for ext in ("", ".soul"):
            if not name.endswith(".soul"):
                candidate = lora_dir / (name + ext)
            else:
                candidate = lora_dir / name

            if not candidate.exists():
                continue

            try:
                soul, _ = load_soul(str(candidate))
                meta = _load_soul_meta(candidate)
                size_mb = round(candidate.stat().st_size / (1024 * 1024), 2)
                result = {
                    "name": candidate.name,
                    "download_url": f"/auto-train/checkpoints/{candidate.name}/download",
                    "soul": _get_soul_name(soul),
                    "loss": soul.metadata.get("avg_loss"),
                    "steps": soul.metadata.get("steps", 0),
                    "epochs": 0,
                    "traits": _get_soul_traits(soul),
                    "lineage": soul.lineage or "lora-feedback",
                    "model_type": "lora",
                    "verdict": soul.metadata.get("eval_verdict"),
                    "perplexity_delta": soul.metadata.get("perplexity_delta"),
                    "bleu_delta": soul.metadata.get("bleu_delta"),
                    "size_mb": size_mb,
                }
                if meta:
                    for k in ("tagline", "description", "born_at", "system_prompt", "tags"):
                        v = meta.get(k)
                        if v is not None and v != "":
                            result[k] = v
                return result
            except Exception as e:
                autotrain_logger.warning(f"Failed to load LoRA soul {candidate}: {e}")

    return None


@router.post("/start")
async def start(req: StartRequest):
    """
    Configure auto-training — stores config for /auto-train/stream to consume.

    Delegates training to UnifiedTrainingPipeline (method='slonet').

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
            autotrain_logger.info(f"Wrote {len(source_lines)} source lines to {tmp}")
    elif req.dataset_id:
        ds_candidate = REPO_ROOT / "datasets" / req.dataset_id
        if ds_candidate.exists():
            # Try common dataset file names
            for name in ("corpus.jsonl", "input.txt", "train.txt", "text.txt"):
                candidate = ds_candidate / name
                if candidate.exists():
                    data_path = str(candidate)
                    break

    state.config = {
        "epochs": req.epochs,
        "learning_rate": req.learning_rate,
        "batch_size": req.batch_size,
        "data_path": data_path,
        "checkpoint_name": req.checkpoint_name or "",
        "algo": req.algo,
        "soul_name": req.soul_name,
    }
    state.running = True
    import state as _srv_state
    _srv_state.training_active = True
    autotrain_logger.info("Auto-train configured: data_path=%s epochs=%d", data_path, req.epochs)
    return {"status": "ready", "data_path": data_path, "epochs": req.epochs, "config": state.config}


@router.post("/start-turbo")
async def start_turbo(req: TurboStartRequest):
    """
    Start training using TurboTrainer (encoder-decoder Transformer via torch shim).

    Args:
        req: TurboStartRequest with model architecture and training params

    Returns:
        dict with training result (status, model_path, final_loss, total_steps, epochs)
    """
    try:
        from domains.training.turbo_trainer import TurboTrainer, TurboConfig

        data_path = req.data_path
        if not data_path and req.dataset_id:
            ds_candidate = REPO_ROOT / "datasets" / req.dataset_id
            if ds_candidate.exists():
                corp = ds_candidate / "corpus.jsonl"
                if corp.exists():
                    data_path = str(corp)
                else:
                    txt_files = list(ds_candidate.glob("*.txt"))
                    if txt_files:
                        data_path = str(txt_files[0])

        if not data_path:
            return {"status": "error", "message": "No data_path or dataset_id provided"}

        config = TurboConfig(
            data_path=data_path,
            vocab_size=req.vocab_size,
            n_embed=req.n_embed,
            n_head=req.n_head,
            n_encoder_layers=req.n_encoder_layers,
            n_decoder_layers=req.n_decoder_layers,
            dim_feedforward=req.dim_feedforward,
            dropout=req.dropout,
            batch_size=req.batch_size,
            epochs=req.epochs,
            learning_rate=req.learning_rate,
            max_src_len=req.max_src_len,
            max_tgt_len=req.max_tgt_len,
        )

        trainer = TurboTrainer(config)
        output_dir = Path(REPO_ROOT / "models" / "turbo-trained")
        config.output_dir = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        autotrain_logger.info("Starting TurboTrainer with method=%s data=%s", req.method, data_path)
        result = trainer.train()
        autotrain_logger.info("TurboTrainer result: %s", result)

        return result
    except Exception as e:
        autotrain_logger.error("TurboTrainer failed: %s", e)
        return {"status": "error", "message": str(e)}


@router.post("/stop")
async def stop():
    global _auto_train_cancel_event
    state.running = False
    if _auto_train_cancel_event is not None:
        _auto_train_cancel_event.set()
        return {"status": "cancelling", "message": "Cancelling auto-training"}
    return {"status": "stopped"}


@router.get("/status")
async def status():
    """Get training status."""
    return {"running": state.running, "config": state.config}


@router.get("/stream")
async def stream():
    """
    Stream auto-training as SSE via UnifiedTrainingPipeline (method='slonet').

    Delegates model creation, training, and .soul export to the pipeline.
    Phases: GENERATE_DATA → DISTILL → TRAIN → EVALUATE → DEPLOY → COMPLETE
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
        """Run UnifiedTrainingPipeline in executor thread."""
        global _auto_train_cancel_event
        from domains.training.unified_pipeline import (
            UnifiedTrainingPipeline, UnifiedTrainingConfig,
        )

        cfg = UnifiedTrainingConfig(
            method="slonet",
            data_path=state.config.get("data_path", ""),
            epochs=state.config.get("epochs", 10),
            learning_rate=state.config.get("learning_rate", 0.001),
            batch_size=state.config.get("batch_size", 32),
            checkpoint_dir=str(CHECKPOINTS_DIR),
            output_dir=str(CHECKPOINTS_DIR),
            vocab_size=0,
            n_embed=64,
            n_layer=2,
            n_head=4,
            block_size=64,
            soul_name=state.config.get("soul_name", "assistant"),
            system_prompt=_build_soul_prompt(state.config.get("soul_name", "assistant")),
            skip_generate=True,
            skip_distill=True,
            skip_evaluate=True,
            skip_deploy=True,
        )

        pipeline = UnifiedTrainingPipeline(cfg)
        _auto_train_cancel_event = threading.Event()

        def _on_progress(progress):
            if _auto_train_cancel_event is not None and _auto_train_cancel_event.is_set():
                raise _AutoTrainCancelled("Training cancelled by user")
            sse = progress.to_sse_event(stream_name="auto-train")
            sse_str = "data: " + json.dumps(sse) + "\n\n"
            _enqueue(sse_str)

        try:
            state.running = True
            result = pipeline.run(on_progress=_on_progress, cancel_event=_auto_train_cancel_event)

            if result.get("status") == "completed":
                ckpt = result.get("checkpoint", "")
                fl = result.get("final_loss")
                ts = result.get("total_steps", 0)
                autotrain_logger.info(
                    "Auto-train complete: checkpoint=%s final_loss=%s steps=%d",
                    ckpt, fl, ts,
                )
            elif result.get("cancelled"):
                autotrain_logger.info("Auto-train cancelled by user")
                _enqueue(sse_complete("auto-train", data={"cancelled": True}, message="Training cancelled"))
            else:
                autotrain_logger.warning("Auto-train result: %s", result.get("status"))

        except _AutoTrainCancelled:
            autotrain_logger.info("Auto-train cancelled by user")
            _enqueue(sse_complete("auto-train", data={"cancelled": True}, message="Training cancelled"))
        except Exception as e:
            autotrain_logger.error("Training worker error: %s", e)
            _enqueue(sse_error("auto-train", "FAILED", str(e)))
        finally:
            _auto_train_cancel_event = None
            state.running = False
            try:
                import state as _srv_state
                _srv_state.training_active = False
            except Exception:
                pass

    async def event_generator():
        worker_task = loop.run_in_executor(None, _training_worker)
        try:
            while True:
                event = await queue.get()
                yield event
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
        except Exception:
            pass

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

    return {"checkpoints": checkpoints}


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
        return {"status": "deleted", "name": deleted}
    return {"status": "not_found"}


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
        return {"status": "not_found", "name": name}

    try:
        # Load the SloTransformer model from .soul file
        soul_net = import_from_sou(str(cp))
        soul_meta = soul_net.soul_signature()
        md = soul_net.metadata or {}

        # Extract vocab from metadata
        stoi = md.get("stoi")
        itos = md.get("itos")
        if stoi is None or itos is None:
            return {
                "status": "error",
                "name": cp.name,
                "error": "Checkpoint has no stoi/itos vocab — retrain to include vocab.",
            }

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
        )

        return {
            "status": "loaded",
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
    except Exception as e:
        import traceback
        autotrain_logger.error("Failed to load checkpoint %s: %s\n%s", cp.name, e, traceback.format_exc())
        return {"status": "error", "name": cp.name, "error": str(e)}


@router.get("/checkpoints/{name}/download")
async def download_checkpoint(name: str):
    """Download a checkpoint .soul file for local (WebGPU) inference."""
    for d in (CHECKPOINTS_DIR, LORA_DIR):
        fp = d / name
        if fp.exists() and fp.suffix in (".soul", ".pt"):
            return FileResponse(str(fp), media_type="application/octet-stream", filename=name)
    raise HTTPException(status_code=404, detail="Checkpoint not found")





