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
from typing import Optional, List, Any
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
    student_net: Any = None
    student_tokenizer: Any = None
    source_lines: List[str] = field(default_factory=list)


state = AutoTrainState()
from pathlib import Path

router = APIRouter(prefix="/auto-train", tags=["training"])

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "models" / "auto-training"
LORA_DIR = REPO_ROOT / "data" / "user_adapters"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)

autotrain_logger = logging.getLogger("autotrain")
autotrain_logger.setLevel(logging.INFO)


def _parse_subtitle_text(text: str) -> List[str]:
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
                "loss": soul.metadata.get("avg_loss"),
                "steps": soul.metadata.get("steps", 0),
                "epochs": soul.metadata.get("step", 0),
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


def _load_checkpoint_into_model(name: str):
    from domains.training.slonet import import_from_sou
    
    autotrain_logger.info(f"Looking for checkpoint: {name}")
    autotrain_logger.info(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
    autotrain_logger.info(f"Files in dir: {list(CHECKPOINTS_DIR.glob('*.soul'))[:3]}")
    
    for ext in ("", ".soul", ".pt"):
        candidate = CHECKPOINTS_DIR / name
        if not (str(name).endswith(".soul") or str(name).endswith(".pt")):
            candidate = CHECKPOINTS_DIR / (name + ext)
        autotrain_logger.info(f"Trying candidate: {candidate}, exists: {candidate.exists()}")
        if candidate.exists():
            try:
                imported = import_from_sou(str(candidate))
                weights = imported._get_weights_dict()

                import numpy as np
                param_idx = 0
                loaded = 0
                skipped = 0
                for p in state.student_net.parameters():
                    key = f"p{param_idx}"
                    if key in weights:
                        w = np.array(weights[key], dtype=np.float32)
                        if w.shape == p.data.shape:
                            p.data[:] = w
                            loaded += 1
                        else:
                            autotrain_logger.warning(
                                f"Shape mismatch p{param_idx}: "
                                f"checkpoint {w.shape} != model {p.data.shape} — skipping"
                            )
                            skipped += 1
                    param_idx += 1

                autotrain_logger.info(
                    f"Loaded .soul weights into SloNet: {candidate.name} "
                    f"({loaded} ok, {skipped} skipped)"
                )
                return imported
            except Exception as e:
                autotrain_logger.error(f"SloNet weight load error: {e}")

            try:
                import torch
                ckpt = torch.load(candidate, map_location="cpu", weights_only=False)
                if "model_state" in ckpt and state.student_net is not None:
                    sd = ckpt["model_state"]
                    param_idx = 0
                    for p in state.student_net.parameters():
                        key = f"p{param_idx}"
                        if key in sd:
                            import numpy as np
                            p.data[:] = sd[key].numpy()
                        param_idx += 1
                    autotrain_logger.info(f"Loaded .pt weights into SloNet: {candidate.name}")
                    return ckpt
            except Exception as e:
                autotrain_logger.error(f"Legacy .pt load error: {e}")

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
            corp = ds_candidate / "corpus.jsonl"
            if corp.exists():
                data_path = str(corp)

    state.config = {
        "epochs": req.epochs,
        "learning_rate": req.learning_rate,
        "batch_size": req.batch_size,
        "data_path": data_path,
        "checkpoint_name": req.checkpoint_name or "",
        "algo": req.algo,
    }
    state.running = True
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
    state.running = False
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
        from domains.training.unified_pipeline import (
            UnifiedTrainingPipeline, UnifiedTrainingConfig,
        )

        cfg = UnifiedTrainingConfig(
            method="slonet",
            data_path=state.config.get("data_path", ""),
            epochs=state.config.get("epochs", 10),
            learning_rate=state.config.get("learning_rate", 0.001),
            batch_size=state.config.get("batch_size", 64),
            checkpoint_dir=str(CHECKPOINTS_DIR),
            output_dir=str(CHECKPOINTS_DIR),
            vocab_size=256,
            n_embed=256,
            n_layer=6,
            n_head=8,
            block_size=128,
            soul_name=state.config.get("soul_name", "assistant"),
            system_prompt=_build_soul_prompt(state.config.get("soul_name", "assistant")),
            skip_generate=True,
            skip_distill=True,
            skip_evaluate=True,
            skip_deploy=True,
        )

        pipeline = UnifiedTrainingPipeline(cfg)

        def _on_progress(progress):
            sse = progress.to_sse_event(stream_name="auto-train")
            sse_str = "data: " + json.dumps(sse) + "\n\n"
            _enqueue(sse_str)

        try:
            state.running = True
            result = pipeline.run(on_progress=_on_progress)

            if result.get("status") == "completed":
                ckpt = result.get("checkpoint", "")
                fl = result.get("final_loss")
                ts = result.get("total_steps", 0)
                autotrain_logger.info(
                    "Auto-train complete: checkpoint=%s final_loss=%s steps=%d",
                    ckpt, fl, ts,
                )
            else:
                autotrain_logger.warning("Auto-train result: %s", result.get("status"))

        except Exception as e:
            autotrain_logger.error("Training worker error: %s", e)
            _enqueue(sse_error("auto-train", "FAILED", str(e)))

    async def event_generator():
        worker_task = loop.run_in_executor(None, _training_worker)

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

        try:
            await worker_task
        except Exception:
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/checkpoints")
async def list_checkpoints():
    """List all saved .soul checkpoints and LoRA adapters with soul metadata."""
    checkpoints = []
    seen = set()

    for ext in ("*.soul", "*.pt"):
        for f in sorted(CHECKPOINTS_DIR.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.name in seen:
                continue
            seen.add(f.name)
            info = _load_soul(f.name)
            if info:
                checkpoints.append(info)

    for npz in sorted(LORA_DIR.glob("*.soul"), key=lambda p: p.stat().st_mtime, reverse=True):
        if npz.name not in seen:
            seen.add(npz.name)
            info = _load_lora_soul(npz.name)
            if info:
                checkpoints.append(info)

    return {"checkpoints": checkpoints}


@router.delete("/checkpoints/{name}")
async def delete_checkpoint(name: str):
    """Delete a checkpoint (.soul or .pt)."""
    base = CHECKPOINTS_DIR / name

    deleted = []
    for candidate in [base, base.with_suffix(".soul" if not name.endswith(".soul") else ".pt")]:
        if candidate.exists():
            candidate.unlink()
            deleted.append(candidate.name)
        meta = candidate.with_suffix(candidate.suffix + ".meta.json")
        if meta.exists():
            meta.unlink()

    if deleted:
        return {"status": "deleted", "name": deleted[0]}
    return {"status": "not_found"}


@router.post("/checkpoints/{name}/load")
async def load_checkpoint(name: str):
    """Load checkpoint weights into student SloNet model."""
    if state.student_net is None:
        return {"status": "error", "message": "No student model. Call /auto-train/start first."}

    imported = _load_checkpoint_into_model(name)

    if imported is not None:
        from domains.inference import load_soul
        soul_name = name
        if not (name.endswith(".soul") or name.endswith(".pt")):
            for candidate in [CHECKPOINTS_DIR / (name + ".soul"), CHECKPOINTS_DIR / (name + ".pt")]:
                if candidate.exists():
                    soul_name = candidate.name
                    break
        elif name.endswith(".pt"):
            soul_name = name.replace(".pt", ".soul")
        else:
            soul_name = name

        for ext in ("", ".soul", ".pt"):
            cp = CHECKPOINTS_DIR / name
            if not (str(name).endswith(".soul") or str(name).endswith(".pt")):
                cp = CHECKPOINTS_DIR / (name + ext)
            if cp.exists():
                try:
                    soul, _ = load_soul(str(cp))
                    return {
                        "status": "loaded",
                        "name": cp.name,
                        "soul": soul.soul_name,
                        "loss": soul.metadata.get("avg_loss"),
                        "steps": soul.metadata.get("steps", 0),
                        "traits": soul.soul_traits,
                        "lineage": soul.lineage,
                    }
                except Exception:
                    pass
                try:
                    import torch
                    ckpt = torch.load(cp, map_location="cpu", weights_only=False)
                    return {
                        "status": "loaded",
                        "name": cp.name,
                        "soul": ckpt.get("soul_name", "unknown"),
                        "loss": ckpt.get("train_loss"),
                        "steps": ckpt.get("steps", 0),
                        "traits": ckpt.get("personality_traits", {}),
                    }
                except Exception:
                    pass

    return {"status": "not_found", "name": name}


@router.get("/checkpoints/{name}/download")
async def download_checkpoint(name: str):
    """Download a checkpoint .soul file for local (WebGPU) inference."""
    for d in (CHECKPOINTS_DIR, LORA_DIR):
        fp = d / name
        if fp.exists() and fp.suffix in (".soul", ".pt"):
            return FileResponse(str(fp), media_type="application/octet-stream", filename=name)
    raise HTTPException(status_code=404, detail="Checkpoint not found")





