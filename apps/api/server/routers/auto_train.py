"""
Auto-Train Router - Unified Teacher-Student Training Pipeline

Follows TrainingSequence: GENERATE_DATA → DISTILL → TRAIN → EVALUATE → DEPLOY → COMPLETE

Uses GPT2 as teacher to generate training pairs, SloNet (NumPy) as student to learn.
Exports checkpoints as .soul (SloughGPT Soul Unit) — self-contained model + identity.
No PyTorch dependency for student training — pure NumPy via SloNet.

Encapsulates router state in ``AutoTrainState`` dataclass rather than 7 module-level
mutable globals (``_running``, ``_config``, ``_teacher_model``, etc.).
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

def to_python(val):
    """Convert numpy types to Python for JSON serialization"""
    if hasattr(val, 'item'):
        return val.item()
    return val

def to_list(vals):
    """Convert list of values"""
    return [to_python(v) for v in vals]


@dataclass
class AutoTrainState:
    """Encapsulated mutable state for the auto-train router.

    Replaces 7 bare module-level globals (``_running``, ``_config``,
    ``_teacher_model``, ``_teacher_tokenizer``, ``_student_net``,
    ``_student_tokenizer``, ``_source_lines``) with typed attributes.
    """
    running: bool = False
    config: dict = field(default_factory=dict)
    teacher_model: Any = None
    teacher_tokenizer: Any = None
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
    teacher_model: str = "gpt2"
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    soul_name: str = "assistant"
    epochs: int = Field(default=10, ge=1, le=1000)
    learning_rate: float = Field(default=0.001, ge=1e-5, le=1.0)
    source_text: Optional[str] = Field(default=None, description="Custom training text (SRT, plain, or lines). If provided, train on this instead of generating from teacher.")
    checkpoint_name: Optional[str] = Field(default=None, description="Load existing checkpoint and continue training")


class TrainOnConversationsRequest(BaseModel):
    """Train on collected conversation data."""
    min_rating: int = Field(default=1, description="Minimum rating to include (1=thumbs up)")
    epochs: int = Field(default=5, ge=1, le=100)
    learning_rate: float = Field(default=0.001)
    personality: str = Field(default="warm", description="warm, curious, playful, balanced")


def _build_soul_prompt(soul_name: str) -> str:
    prompts = {
        "assistant": "You are a helpful assistant. Be clear and friendly.",
        "creative": "You are a creative thinker. Be imaginative and playful.",
        "analyst": "You are a precise analyst. Be methodical and thorough.",
        "coder": "You are an expert coder. Write clean, efficient code.",
        "teacher": "You are a patient teacher. Explain step by step.",
    }
    return prompts.get(soul_name, prompts["assistant"])





def _load_soul_profile(soul_name: str) -> dict:
    """Load personality traits from existing .soul file."""
    try:
        for sou_path in (REPO_ROOT / "models").glob("*.soul"):
            if soul_name.lower() in sou_path.stem.lower():
                from domains.inference import load_soul
                soul_obj, _ = load_soul(str(sou_path))
                if soul_obj and soul_obj.personality:
                    return {
                        "warmth": soul_obj.personality.warmth,
                        "creativity": soul_obj.personality.creativity,
                        "curiosity": soul_obj.personality.curiosity,
                        "confidence": soul_obj.personality.confidence,
                        "empathy": soul_obj.personality.empathy,
                        "formality": soul_obj.personality.formality,
                    }
    except Exception:
        pass
    return {
        "warmth": 0.5, "creativity": 0.5,
        "curiosity": 0.5, "confidence": 0.5,
        "empathy": 0.5, "formality": 0.5,
    }


def _export_soul(
    student_net,
    student_tokenizer,
    soul_name: str,
    system_prompt: str,
    total_loss: float,
    step: int,
    epochs: int,
) -> tuple[str, dict]:
    """
    Export trained SloNet as a .soul file with full soul profile.

    Args:
        student_tokenizer: SloBPE tokenizer (or legacy dict)

    Returns:
        (checkpoint_name, soul_profile_dict)
    """
    import datetime
    from domains.inference import (
        SloProfile, PersonalityCore, GenerationParams,
        BehavioralTraits, CognitiveSignature, EmotionalRange,
        save_soul,
    )

    traits = _load_soul_profile(soul_name)
    avg_loss = total_loss / max(step, 1)

    # Get vocab size (handles both SloBPE and legacy dict tokenizer)
    if hasattr(student_tokenizer, 'vocab_size'):
        vocab_size = student_tokenizer.vocab_size
        tokenizer_type = "soulbpe"
    else:
        vocab_size = len(student_tokenizer.get("stoi", {}))
        tokenizer_type = "char"

    soul_profile = SloProfile(
        name=f"{soul_name}-soul",
        version="1.0.0",
        tagline=f"AI Slo trained via teacher-student distillation",
        description=(
            f"{'BPE' if tokenizer_type == 'soulbpe' else 'Char-level'} SloNet trained by GPT2 teacher "
            f"in {step} steps. Slo personality: {soul_name}."
        ),
        lineage="teacher-student-distillation",
        base_model="slonet-lstm",
        training_dataset="gpt2-generated",
        epochs_trained=epochs,
        final_train_loss=round(avg_loss, 6),
        final_val_loss=round(avg_loss, 6),
        personality=PersonalityCore(
            warmth=traits.get("warmth", 0.5),
            creativity=traits.get("creativity", 0.5),
            curiosity=traits.get("curiosity", 0.5),
            confidence=traits.get("confidence", 0.5),
            empathy=traits.get("empathy", 0.5),
            formality=traits.get("formality", 0.5),
        ),
        behavior=BehavioralTraits(
            speaking_style="conversational",
            explanation_depth="moderate",
        ),
        cognition=CognitiveSignature(
            pattern_recognition=0.5,
            abstract_reasoning=0.5,
            factual_precision=0.5,
        ),
        emotion=EmotionalRange(
            empathy_depth=0.5,
            mood_responsiveness=0.5,
        ),
        generation=GenerationParams(
            temperature=0.8,
            top_p=0.9,
            top_k=40,
            max_tokens=256,
        ),
        system_prompt=system_prompt,
        tags=[soul_name, "slonet", tokenizer_type, "gpt2-teacher"],
        metadata={
            "steps": step,
            "total_loss": round(total_loss, 6),
            "avg_loss": round(avg_loss, 6),
            "teacher": "gpt2",
            "student": "slonet-lstm",
            "vocab_size": vocab_size,
            "tokenizer_type": tokenizer_type,
            "tokenizer_config": student_tokenizer.to_dict() if hasattr(student_tokenizer, 'to_dict') else {},
            "exported_at": datetime.datetime.utcnow().isoformat() + "Z",
        },
    )

    ckpt_name = f"{soul_name}_{int(time.time())}.soul"
    ckpt_path = CHECKPOINTS_DIR / ckpt_name

    slonet_path = student_net._sou_path if hasattr(student_net, "_sou_path") and student_net._sou_path else None
    SloNetExport = student_net
    SloNetExport.system_prompt = system_prompt
    SloNetExport.metadata["avg_loss"] = round(avg_loss, 6)
    SloNetExport.metadata["steps"] = step
    SloNetExport.metadata["exported_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    SloNetExport.metadata["soul_profile"] = soul_profile.to_dict()
    SloNetExport.metadata["lstm_dropout"] = student_net.layers[1].dropout if len(student_net.layers) > 1 and hasattr(student_net.layers[1], 'dropout') else 0.0
    if hasattr(student_tokenizer, 'to_dict'):
        SloNetExport.metadata["tokenizer_config"] = student_tokenizer.to_dict()
        SloNetExport.metadata["tokenizer_type"] = "soulbpe"

    save_soul(student_net, str(ckpt_path), soul_profile=soul_profile)

    meta_path = ckpt_path.with_suffix(".soul.meta.json")
    with open(meta_path, "w") as f:
        json.dump(soul_profile.to_dict(), f, indent=2, default=str)

    # Keep only the best checkpoint per soul — delete all worse ones.
    all_soul_ckpts = sorted(CHECKPOINTS_DIR.glob(f"{soul_name}_*.soul"), key=lambda p: p.stat().st_mtime)
    if len(all_soul_ckpts) > 1:
        best_loss = float("inf")
        best_ckpt = None
        for c in all_soul_ckpts:
            m = _load_soul_meta(c)
            l = m.get("metadata", {}).get("avg_loss") or m.get("final_train_loss")
            if l is not None and l < best_loss:
                best_loss = l
                best_ckpt = c
        for c in all_soul_ckpts:
            if c.name != best_ckpt.name:
                c.unlink(missing_ok=True)
                meta = c.with_suffix(".soul.meta.json")
                meta.unlink(missing_ok=True)
                autotrain_logger.info(f"Pruned checkpoint: {c.name}")
        # If the new checkpoint was pruned, report the kept one instead
        if best_ckpt and not ckpt_path.exists():
            ckpt_name = best_ckpt.name
            autotrain_logger.info(f"New checkpoint was worse than existing; keeping {ckpt_name}")

    autotrain_logger.info(f"Slo exported: {ckpt_name}")
    return ckpt_name, soul_profile.to_dict()


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
    Configure and start a new auto-training session.

    Args:
        req: teacher_model (default gpt2), temperature, soul_name, epochs, learning_rate

    Returns:
        dict with status and config

    Side effects:
        - loads GPT2 teacher model
        - creates SloNet student model with BPE tokenizer
        - sets AutoTrainState
    """
    state.running = True
    state.config = {
        "teacher_model": req.teacher_model,
        "temperature": req.temperature,
        "soul_name": req.soul_name,
        "epochs": req.epochs,
        "learning_rate": req.learning_rate,
        "source_text": req.source_text,
    }

    # Parse custom source text if provided (supports SRT, VTT, or plain text)
    state.source_lines = []

    # Load existing checkpoint for continued training if requested
    if req.checkpoint_name:
        autotrain_logger.info(f"Loading checkpoint for continued training: {req.checkpoint_name}")
        try:
            from domains.training.slonet import import_from_sou
            from domains.training.tokenizer import SloBPE
            ckpt_path = CHECKPOINTS_DIR / req.checkpoint_name
            if not ckpt_path.exists():
                ckpt_path = CHECKPOINTS_DIR / (req.checkpoint_name + ".soul")
            if ckpt_path.exists():
                loaded = import_from_sou(str(ckpt_path))
                state.student_net = loaded
                # Try to load BPE tokenizer from checkpoint metadata
                meta = getattr(loaded, 'metadata', {}) or {}
                tok_config = meta.get('tokenizer_config') if isinstance(meta, dict) else None
                if tok_config:
                    state.student_tokenizer = SloBPE.from_dict(tok_config)
                    autotrain_logger.info(f"BPE tokenizer loaded from checkpoint (vocab={state.student_tokenizer.vocab_size})")
                if hasattr(loaded, 'soul_name'):
                    state.config["soul_name"] = loaded.soul_name
                autotrain_logger.info(f"Checkpoint loaded: {ckpt_path.name}")
        except Exception as e:
            autotrain_logger.error(f"Failed to load checkpoint: {e}")

    if req.source_text:
        state.source_lines = _parse_subtitle_text(req.source_text)
        autotrain_logger.info(f"Parsed {len(state.source_lines)} lines from custom source text")
        if state.source_lines:
            # Delegate to SloEngine for model + tokenizer creation
            from domains.core.soul import SloEngine
            engine = SloEngine(device="cpu")
            result = engine.learn(
                texts=state.source_lines,
                soul_name=req.soul_name,
                epochs=1,  # minimal — stream() will continue training
                learning_rate=req.learning_rate,
                vocab_size=512,
            )
            state.student_tokenizer = engine._tokenizer
            state.student_net = engine._model
            state.engine = engine
            autotrain_logger.info(f"SloEngine created SloNet: {result}")
            state.running = True
            return {"status": "ready", "source_lines": len(state.source_lines), "config": state.config}

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import os
        
        # Check for HuggingFace token (supports private models)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        autotrain_logger.info(f"Loading teacher: {req.teacher_model}")
        state.teacher_tokenizer = AutoTokenizer.from_pretrained(req.teacher_model, token=hf_token)
        state.teacher_model = AutoModelForCausalLM.from_pretrained(req.teacher_model, token=hf_token)
        state.teacher_model.eval()
        state.teacher_tokenizer.pad_token = state.teacher_tokenizer.eos_token
    except Exception as e:
        autotrain_logger.error(f"Teacher load failed: {e}")
        state.running = False
        return {"status": "error", "message": str(e)}

    # Train BPE on seed topic examples (used when no source_text)
    seed_texts = [
        "What would happen if the sun suddenly disappeared?",
        "Why do some animals sleep more than others?",
        "How do memories form in the brain?",
        "What is dark matter?",
        "How does machine learning work?",
        "What is the meaning of life?",
        "Tell me about artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    from domains.core.soul import SloEngine
    engine = SloEngine(device="cpu")
    result = engine.learn(
        texts=seed_texts,
        soul_name=req.soul_name,
        epochs=1,
        learning_rate=req.learning_rate,
        vocab_size=512,
    )
    state.student_tokenizer = engine._tokenizer
    state.student_net = engine._model
    state.engine = engine
    autotrain_logger.info(f"SloEngine created SloNet: {result}")

    return {
        "status": "started",
        "teacher": req.teacher_model,
        "student": "slonet-bpe-lstm",
        "soul": req.soul_name,
        "epochs": req.epochs,
    }


@router.post("/stop")
async def stop():
    state.running = False
    return {"status": "stopped"}


@router.get("/status")
async def status():
    """Get training status including BPE tokenizer stats."""
    result = {"running": state.running, "config": state.config}
    tok = state.student_tokenizer
    if tok is not None and hasattr(tok, 'vocab_size'):
        result["bpe"] = {
            "vocab_size": tok.vocab_size,
            "merges": len(tok.merges),
            "base_chars": tok.vocab_stats()["base_chars"],
            "subwords": tok.vocab_stats()["merged_subwords"],
        }
    return result


@router.get("/stream")
async def stream():
    """
    Stream training following TrainingSequence:
    GENERATE_DATA → DISTILL → TRAIN → EVALUATE → DEPLOY → COMPLETE

    Teacher (GPT2/PyTorch) generates training pairs.
    Student (SloNet/NumPy) learns via cross-entropy distillation.
    """
    if not state.config or state.teacher_model is None or state.student_net is None:
        return StreamingResponse(
            iter([sse_error("auto-train", "IDLE", "Call /auto-train/start first")]),
            media_type="text/event-stream",
        )

    async def event_generator():
        from domains.training.slonet import SloAdam, cross_entropy, tensor

        teacher = state.teacher_model
        tokenizer = state.teacher_tokenizer
        temp = state.config.get("temperature", 0.8)
        epochs = state.config.get("epochs", 10)
        soul_name = state.config.get("soul_name", "assistant")
        lr = state.config.get("learning_rate", 0.001)
        system_prompt = _build_soul_prompt(soul_name)
        
        # Check if we have custom source text
        source_text = state.config.get("source_text")
        use_custom = source_text and state.source_lines
        
        if use_custom:
            yield sse_event(
                stream="auto-train",
                phase="GENERATE_DATA",
                status="working",
                data={},
                meta={"epoch": 1, "total_epochs": epochs},
                message=f"Training on {len(state.source_lines)} custom lines | Slo: {soul_name}",
            )
        
        # Initialize training
        optimizer = SloAdam(lr=lr)
        _train_start = time.perf_counter()
        unk_idx = state.student_tokenizer.pad_id

        total_loss = 0.0
        step = 0
        loss_history = []

        if not use_custom:
            yield sse_event(
                stream="auto-train",
                phase="GENERATE_DATA",
                status="working",
                data={},
                meta={"epoch": 1, "total_epochs": epochs},
                message=f"Teacher: GPT2 | Student: SloNet (BPE NumPy) | Slo: {soul_name}",
            )

        topic_examples = [
            "What would happen if the sun suddenly disappeared?",
            "Why do some animals sleep more than others?",
            "How do memories form in the brain?",
            "What is dark matter?",
        ]

        for epoch in range(epochs):
            if not state.running:
                break

            yield sse_event(
                stream="auto-train",
                phase="TRAIN",
                status="working",
                data={},
                meta={
                    "epoch": to_python(epoch) + 1,
                    "total_epochs": to_python(epochs),
                    "step": to_python(step),
                    "loss": None,
                },
                message=f"Epoch {to_python(epoch) + 1}/{to_python(epochs)}",
            )

            # Get training data - custom or generated
            training_pairs = []
            
            if use_custom:
                # Use custom source text - each line is a training example
                for line_idx, line in enumerate(state.source_lines):
                    if state.running:
                        training_pairs.append((line, line))
            else:
                # Generate from teacher model (original logic)
                for topic in topic_examples:
                    if not state.running:
                        break
                    try:
                        import torch
                        with torch.no_grad():
                            topic_in = tokenizer(
                                f"Question: {topic}",
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=80,
                            )
                            topic_out = teacher.generate(
                                **topic_in,
                                max_new_tokens=20,
                                temperature=0.9,
                                top_k=50,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                            question = tokenizer.decode(topic_out[0], skip_special_tokens=True)
                            question = question.replace(f"Question: {topic}", "").strip() or topic

                        yield sse_event(
                            stream="auto-train",
                            phase="GENERATE_DATA",
                            status="working",
                            data={"type": "question", "question": question[:80]},
                            meta={"step": step},
                            message=f"Q: {question[:80]}",
                        )

                        with torch.no_grad():
                            ans_in = tokenizer(
                                f"Answer: {question}",
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=80,
                            )
                            ans_out = teacher.generate(
                                **ans_in,
                                max_new_tokens=60,
                                temperature=temp,
                                top_k=40,
                                top_p=0.9,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                            answer = tokenizer.decode(ans_out[0], skip_special_tokens=True)
                            if "Answer:" in answer:
                                answer = answer.split("Answer:")[-1].strip()
                            answer = " ".join(answer.split())
                            if len(answer) < 10:
                                answer = f"The answer to {question} involves fundamental principles."

                        yield sse_event(
                            stream="auto-train",
                            phase="DISTILL",
                            status="working",
                            data={"type": "teacher", "answer": answer[:100]},
                            meta={"step": step},
                            message=f"Teacher: {answer[:100]}",
                        )
                        training_pairs.append((question, answer))
                    except Exception as e:
                        autotrain_logger.warning(f"Training pair error: {e}")
            
            # Train on the collected pairs (both custom and generated)
            for pair_idx, (question, answer) in enumerate(training_pairs):
                if not state.running:
                    break
                    
                try:
                    input_ids = state.student_tokenizer.encode(answer[:64])
                    if len(input_ids) < 2:
                        continue

                    seq_len = min(len(input_ids) - 1, 32)
                    if seq_len < 1:
                        continue

                    chunk_size = 8
                    for i in range(0, seq_len, chunk_size):
                        x_chunk = input_ids[i : i + chunk_size]
                        y_chunk = input_ids[i + 1 : i + chunk_size + 1]
                        while len(x_chunk) < chunk_size:
                            x_chunk.append(unk_idx)
                        while len(y_chunk) < chunk_size:
                            y_chunk.append(unk_idx)

                        x = tensor([[x_chunk]], requires_grad=True)
                        y = tensor([[y_chunk]])

                        lstm_layer = state.student_net.layers[1]
                        hidden = lstm_layer.init_hidden()
                        logits, _ = lstm_layer.forward(x, hidden)
                        loss = cross_entropy(logits, y.reshape(-1))

                        loss.backward()
                        optimizer.step(state.student_net.parameters())

                        step += 1
                        total_loss += loss.data[()]
                        avg_loss = total_loss / step
                        loss_history.append(loss.data[()])

                        progress = min(int((step / (epochs * max(len(topic_examples), len(state.source_lines) if use_custom else 1))) * 100), 99)

                        yield sse_event(
                            stream="auto-train",
                            phase="TRAIN",
                            status="working",
                            data={"loss": round(avg_loss, 4), "progress": progress},
                            meta={
                                "step": step,
                                "epoch": epoch + 1,
                                "total_epochs": epochs,
                                "elapsed_ms": int((time.perf_counter() - _train_start) * 1000),
                            },
                            message=f"loss={avg_loss:.4f}",
                        )

                except Exception as e:
                    autotrain_logger.error(f"Step error: {e}")
                    yield sse_event(
                        stream="auto-train",
                        phase="TRAIN",
                        status="error",
                        data={"error": str(e)},
                        meta={"step": step},
                        message=f"Error: {e}",
                    )

            yield sse_event(
                stream="auto-train",
                phase="EVALUATE",
                status="success",
                data={"avg_loss": round(to_python(total_loss) / max(to_python(step), 1), 4)},
                meta={
                    "epoch": to_python(epoch) + 1,
                    "total_epochs": to_python(epochs),
                    "step": to_python(step),
                },
                message=f"Epoch {to_python(epoch) + 1} complete",
            )

            yield sse_event(
                stream="auto-train",
                phase="DEPLOY",
                status="working",
                data={},
                meta={"step": to_python(step)},
                message="Exporting SloNet to .soul format...",
            )

            state.student_net.system_prompt = system_prompt
            state.student_net.metadata["avg_loss"] = round(total_loss / max(step, 1), 6)
            state.student_net.metadata["steps"] = step

            ckpt_name, soul_dict = _export_soul(
                state.student_net,
                state.student_tokenizer,
                soul_name,
                system_prompt,
                total_loss,
                step,
                epochs,
            )

            yield sse_complete(
                stream="auto-train",
                phase="COMPLETE",
                data={
                    "checkpoint": ckpt_name,
                    "final_loss": round(to_python(total_loss) / max(to_python(step), 1), 4),
                    "epochs": to_python(epochs),
                    "loss_history": to_list(loss_history[-20:]),
                    "traits": soul_dict.get("personality", {}) if soul_dict else {},
                },
                meta={
                    "steps": to_python(step),
                    "total_epochs": to_python(epochs),
                },
                message=f"Training complete. Saved {ckpt_name}",
            )

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


@router.post("/train-conversations")
async def train_on_conversations(req: TrainOnConversationsRequest):
    """
    Train on collected conversation data.
    
    Uses high-rated conversations to train a natural-sounding model.
    """
    try:
        from domains.training_data import get_collector
        from domains.companion import create_companion
        
        collector = get_collector()
        
        # Get high-quality pairs
        pairs = collector.get_high_quality_pairs(min_rating=req.min_rating)
        
        if len(pairs) < 5:
            return {
                "status": "insufficient_data",
                "pairs": len(pairs),
                "message": f"Need at least 5 pairs, got {len(pairs)}. Chat more!",
            }
        
        # Build training data from conversations
        training_texts = [f"User: {p.user}\nAssistant: {p.assistant}" for p in pairs]
        
        # Get personality prompt
        companion = create_companion(name="Trained", personality=req.personality)
        system_prompt = companion.get_system_prompt()
        
        return {
            "status": "ready",
            "pairs": len(pairs),
            "personality": req.personality,
            "system_prompt": system_prompt[:200] + "...",
            "message": f"Ready to train on {len(pairs)} conversation pairs",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-pt/{name}")
async def load_pt_checkpoint(name: str):
    """Load a .pt checkpoint directly for chat."""
    from pathlib import Path
    import torch
    
    REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
    CHECKPOINTS_DIR = REPO_ROOT / "models" / "auto-training"
    
    # Find file
    pt_file = CHECKPOINTS_DIR / name
    if not pt_file.exists():
        # Try with .pt suffix
        pt_file = CHECKPOINTS_DIR / f"{name}.pt"
    
    if not pt_file.exists():
        return {"error": f"Not found: {name}"}
    
    try:
        ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
        
        return {
            "status": "loaded",
            "name": name,
            "steps": ckpt.get("total_steps", 0),
            "vocab_size": len(ckpt.get("tokenizer", {}).get("stoi", {})),
            "train_loss": ckpt.get("training_log", [None])[-1],
            "file_size": pt_file.stat().st_size,
        }
    except Exception as e:
        return {"error": str(e)}


