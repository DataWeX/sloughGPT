"""
Slos Router - Personality/soul management endpoints

Encapsulates router state in ``SloRouterState`` dataclass rather than module-level
mutable globals. Actual soul state lives in ``SloManager`` singleton.
"""
from dataclasses import dataclass
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from typing import Optional, Any, Dict
from pydantic import BaseModel
import json, asyncio, numpy as np, logging

try:
    from domains.api.sse_envelope import sse_event, sse_token, sse_error, sse_complete
except ImportError:
    def sse_event(stream, phase, status, data=None, meta=None, message=""):
        return "data: " + json.dumps({
            "stream": stream, "phase": phase, "status": status,
            "data": data or {}, "meta": meta or {}, "message": message
        }) + "\n\n"
    def sse_token(stream, token, meta=None):
        return sse_event(stream, "STREAMING", "working", {"token": token}, meta or {})
    def sse_error(stream, phase, error, meta=None):
        return sse_event(stream, phase, "error", {"error": error}, meta or {}, f"Error: {error}")
    def sse_complete(stream, phase="COMPLETE", data=None, meta=None, message="Done"):
        return sse_event(stream, phase, "complete", data or {}, meta or {}, message)

try:
    from domains.models import SloughGPTModel
except ImportError:
    SloughGPTModel = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/souls", tags=["souls"])


@dataclass
class SloRouterState:
    """Encapsulated state for the souls router.

    Current soul tracking and model references. The real soul state
    lives in SloManager; this is a thin cache for switch operations.
    """
    current_soul: Any = None
    main_model: Any = None
    main_tokenizer: Any = None


state = SloRouterState()


def _load_slough_model(checkpoint_path, tie_weights=True):
    """Load a SloughGPTModel from a .soul file using SloNet import.

    Auto-detects config (vocab, hidden, n_blocks) from state dict keys.

    Returns:
        SloughGPTModel with weights loaded.

    Side effects:
        - Reads .soul file from disk
    """
    if SloughGPTModel is None:
        raise RuntimeError("SloughGPTModel not available — PyTorch model module not loaded")
    from domains.inference import load_soul

    soul, sd = load_soul(checkpoint_path)
    if isinstance(sd, dict) and "tok_emb.weight" not in sd:
        sd = sd.get("weights", sd)
        if not isinstance(sd, dict):
            sd = sd.state_dict() if hasattr(sd, 'state_dict') else {}

    vocab = sd["tok_emb.weight"].shape[0]
    hidden = sd["tok_emb.weight"].shape[1]
    n_blocks = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1

    n_head = 8
    q_w = sd.get("blocks.0.attn.q_proj.weight")
    if q_w is None:
        q_w = sd.get("blocks.0.q_proj.weight")
    if q_w is not None:
        head_dim = hidden // 8
        if head_dim > 0:
            detected_heads = q_w.shape[0] // head_dim
            if detected_heads >= 1:
                n_head = detected_heads

    intermediate_size = 4 * hidden // 2
    for key in sd:
        if "mlp.w1.weight" in key:
            w1_shape = sd[key].shape
            if len(w1_shape) >= 2:
                intermediate_size = w1_shape[0]
            break

    model = SloughGPTModel(
        vocab_size=vocab,
        n_embed=hidden,
        n_layer=n_blocks,
        n_head=n_head,
        dropout=0.0,
        tie_weights=tie_weights,
        block_size=128,
        intermediate_size=intermediate_size,
    )

    model.load_state_dict(sd, strict=False)
    return model


class SwitchRequest(BaseModel):
    name: str
    checkpoint_name: Optional[str] = None


class SloChatRequest(BaseModel):
    checkpoint_name: str
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9


@router.post("/chat")
async def soul_chat(req: SloChatRequest, request: Request):
    """Chat using a SloughGPTModel checkpoint (PyTorch-trained transformer).

    Loads the .soul file (PyTorch ZIP format), creates a SloughGPTModel with matching
    config, loads weights, and streams generated tokens autoregressively.

    Falls back to SloNet (NumPy) for checkpoints that don't match SloughGPTModel.
    """
    try:
        from domains.slolib.gpu import get_accelerator

        acc = get_accelerator()
        repo_root = _get_repo_root()
        checkpoint_file = repo_root / "models" / "auto-training" / (req.checkpoint_name + ".soul")
        if not checkpoint_file.exists():
            checkpoint_file = repo_root / "models" / (req.checkpoint_name + ".soul")

        if not checkpoint_file.exists():
            return {"error": f"Checkpoint not found: {req.checkpoint_name}"}

        # Load checkpoint into SloughGPTModel
        model = _load_slough_model(checkpoint_file)

        chars = list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?'-")
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}

        def encode(t):
            return [stoi.get(c.lower(), 0) for c in t]

        def decode(ids):
            return "".join(itos.get(i, "?") for i in ids)

        enc_prompt = encode(req.prompt)
        if len(enc_prompt) > 128:
            enc_prompt = enc_prompt[:128]

        async def stream():
            temperature = max(0.1, req.temperature)
            top_p = max(0.1, min(1.0, req.top_p))

            generated = list(enc_prompt)

            for _ in range(req.max_new_tokens):
                if await request.is_disconnected():
                    return

                seq = np.array([generated[-128:]], dtype=np.int64)

                logits_arr, _ = model.forward(seq, targets=None)
                logit_row = logits_arr[0, -1, :] / temperature

                if top_p < 1.0:
                    sorted_idx = np.argsort(-logit_row)
                    sorted_vals = logit_row[sorted_idx]
                    probs_sorted = np.exp(sorted_vals - sorted_vals.max())
                    probs_sorted = probs_sorted / probs_sorted.sum()
                    cumsum = np.cumsum(probs_sorted)
                    mask = cumsum > top_p
                    mask[1:] = mask[:-1]
                    mask[0] = False
                    indices_to_remove = sorted_idx[mask]
                    logit_row[indices_to_remove] = -1e9

                max_val = logit_row.max()
                probs = np.exp(logit_row - max_val)
                probs = probs / (probs.sum() + 1e-10)
                probs = np.where(np.isfinite(probs), probs, np.ones_like(probs) / probs.size)
                probs = probs / (probs.sum() + 1e-10)
                next_tok = int(np.random.choice(len(probs), p=probs))

                generated.append(next_tok)
                token_text = decode([next_tok])

                if token_text.strip():
                    yield sse_token("souls-chat", token_text)

                if next_tok == 0:
                    break

                await asyncio.sleep(0)

            yield sse_complete("souls-chat", data={"response": decode(generated)})

        return StreamingResponse(stream(), media_type="text/event-stream")

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


def _soul_tensor(data):
    """Create a SloNet Tensor from numpy array."""
    from domains.training.slonet import Tensor
    if hasattr(data, 'data') and isinstance(getattr(data, 'data', None), np.ndarray):
        return data
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    elif data.dtype != np.float32:
        data = np.array(data, dtype=np.float32)
    else:
        data = np.array(data, copy=True)
    return Tensor(data)


def _get_repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[4]


@router.post("/switch")
async def switch_soul(
    req: SwitchRequest,
    checkpoint_name: Optional[str] = None,
):
    """Switch to a different soul and update ContextCore system prompt.

    If checkpoint_name is provided, also loads the corresponding checkpoint
    weights into the main chat model (baby model → inference engine).
    """
    try:
        from domains.inference.slo_manager import get_slo_manager

        manager = get_slo_manager()
        result = manager.switch_soul(req.name)

        # Sync soul to ContextCore system prompt (best-effort)
        soul_info = manager.get_soul(req.name)
        if soul_info:
            try:
                from domains.infrastructure.context_core import get_context_core
                ctx_core = get_context_core()
                if ctx_core:
                    soul_prompt = _build_soul_system_prompt(soul_info)
                    ctx_core.set_system_prompt(soul_prompt)
            except Exception as e:
                logger.warning("Failed to update context core system prompt on soul switch: %s", e)

        # Load checkpoint into main model if requested
        if req.checkpoint_name:
            loaded = _load_checkpoint_into_model(req.checkpoint_name)
            result["checkpoint_loaded"] = loaded
            try:
                from domains.infrastructure.server_state import get_server_state
                get_server_state().record_model_event(
                    "load", req.name, f"checkpoint={req.checkpoint_name}"
                )
            except Exception as e:
                logger.debug("Failed to record checkpoint load event: %s", e)

        if result.get("success") and soul_info and soul_info.path:
            try:
                from domains.core.soul import SloEngine
                engine = SloEngine(device="cpu")
                soul = engine.load_soul(soul_info.path)
                import state as server_state
                server_state.current_soul = soul
                server_state.soul_engine = engine
            except Exception as exc:
                logger.warning("Failed to set soul engine: %s", exc)

        try:
            from domains.infrastructure.server_state import get_server_state
            get_server_state().record_model_event("switch", req.name)
        except Exception:
            pass

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _load_checkpoint_into_model(checkpoint_name: str) -> dict:
    """Load an auto-train checkpoint's weights into the main model/global state."""
    try:
        import sys
        main_mod = sys.modules.get("__main__")
        if main_mod is None:
            return {"status": "no_main_module"}

        checkpoints_dir = getattr(main_mod, "_REPO_ROOT", None) / "models" / "auto-training"
        if checkpoints_dir is None:
            return {"status": "no_repo_root"}

        checkpoint_file = checkpoints_dir / checkpoint_name
        if not checkpoint_file.exists():
            return {"status": "not_found", "path": str(checkpoint_file)}

        # Use SloNet import for .soul files
        from domains.training.slonet import import_from_sou
        soul_net = import_from_sou(str(checkpoint_file))
        soul_meta = soul_net.soul_signature()
        model_state = soul_net.state_dict()

        # Try to load into baby model (auto-train model)
        baby = getattr(main_mod, "_auto_train_baby_model", None)
        if baby is not None:
            baby.load_state_dict(model_state, strict=False)
            return {
                "status": "loaded_into_baby",
                "name": checkpoint_name,
                "soul": soul_meta.get("soul_name", "unknown"),
                "steps": soul_meta.get("step", 0),
            }

        # Try to load into main model
        main_m = getattr(main_mod, "model", None)
        if main_m is not None and isinstance(main_m, dict):
            main_m["model"].load_state_dict(model_state, strict=False)
            return {"status": "loaded_into_main", "name": checkpoint_name}

        return {"status": "no_target_model"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _build_soul_system_prompt(soul_info) -> str:
    """Build system prompt from soul personality traits."""
    traits = soul_info.description or soul_info.name
    warmth = soul_info.personality.get("warmth", 0.5)
    creativity = soul_info.personality.get("creativity", 0.5)
    curiosity = soul_info.personality.get("curiosity", 0.5)
    confidence = soul_info.personality.get("confidence", 0.5)
    soul_traits = ", ".join(soul_info.traits) if soul_info.traits else "balanced"

    return f"""You are {soul_info.name}. {traits}

Personality: warmth={warmth:.1f}, creativity={creativity:.1f}, curiosity={curiosity:.1f}, confidence={confidence:.1f}
Reasoning approach: {soul_traits}
Be yourself — let your personality shape how you respond."""


@router.get("")
async def list_souls():
    """
    List all available souls with name, description, and traits.

    Returns:
        dict with ``souls`` (list) and ``current_soul`` (name string or None)

    Side effects:
        - calls SloManager.list_souls() and get_current_soul()
    """
    try:
        from domains.inference.slo_manager import get_slo_manager
        manager = get_slo_manager()
        souls = manager.list_souls()
        current = manager.get_current_soul()
        return {
            "souls": [
                {
                    "name": s.name,
                    "path": s.path,
                    "description": s.description,
                    "personality": getattr(s, "personality", {}),
                    "traits": getattr(s, "traits", []),
                }
                for s in souls
            ],
            "current_soul": current.name if current else None,
        }
    except Exception as e:
        return {"souls": [], "current_soul": None, "error": str(e)}


@router.get("/weights")
async def get_trait_weights():
    """
    Get the current trait weight attributes from the active model checkpoint.

    Returns a stat-card of personality, cognition, and emotion trait values
    (each 0.0–1.0), like a player's attributes in a sports game. These
    weights are read from the soul/checkpoint file and overlaid with any
    LoRA-adapted adjustments from feedback.

    Returns:
        dict with ``personality``, ``cognition``, ``emotion`` trait groups,
        or empty dicts if no soul is active.

    Side effects:
        - calls SloManager.get_trait_weights()
    """
    try:
        from domains.inference.slo_manager import get_slo_manager
        manager = get_slo_manager()
        weights = manager.get_trait_weights()
        return weights
    except Exception as e:
        return {"error": str(e)}


class SaveWeightsRequest(BaseModel):
    """Request body for saving trait weights."""
    personality: Optional[Dict[str, float]] = None
    cognition: Optional[Dict[str, float]] = None
    emotion: Optional[Dict[str, float]] = None


@router.post("/weights")
async def save_trait_weights(body: SaveWeightsRequest):
    """
    Save trait weights to the persistent config. Accepts a dict of trait
    groups (personality, cognition, emotion) with trait name → 0.0–1.0 values.

    Flattens the grouped structure and writes to TraitWeightsConfig. Returns
    the merged current state after save.

    Side effects:
        - overwrites selected trait weights in persistent config
        - does not modify soul files — only the live config overlay
    """
    try:
        from domains.context.managers import get_trait_config
        config = get_trait_config()
        flat: Dict[str, float] = {}
        for group in ("personality", "cognition", "emotion"):
            traits = getattr(body, group, None)
            if traits:
                for k, v in traits.items():
                    flat[k] = float(v)
        config.set_many(flat)
        return {"status": "saved"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/weights/modes")
async def get_trait_modes():
    """
    Return the active manager mode for each of the 4 context managers
    (Personality, Memory, Style, Task) derived from current trait weights.

    Each mode is a label + confidence score computed from weighted trait
    composites, treating trait values as config for engineered context
    steering — NOT direct model parameter modification.

    Returns:
        dict with keys ``personality``, ``memory``, ``style``, ``task``,
        each containing ``label``, ``confidence``, ``scores``, and
        manager-specific fields (e.g. ``capacity`` for memory).

    Side effects:
        - reads current TraitWeightsConfig
    """
    try:
        from domains.context.managers import (
            get_trait_config, PersonalityManager, MemoryManager,
            StyleManager, TaskManager,
        )
        config = get_trait_config()
        return {
            "personality": PersonalityManager(config).get_mode(),
            "memory": MemoryManager(config).get_mode(),
            "style": StyleManager(config).get_mode(),
            "task": TaskManager(config).get_mode(),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/current")
async def get_current_soul():
    """
    Get the currently active soul's name, path, description, and traits.

    Returns:
        dict with name, path, description, traits

    Side effects:
        - calls SloManager.get_current_soul()
    """
    try:
        from domains.inference.slo_manager import get_slo_manager
        manager = get_slo_manager()
        current = manager.get_current_soul()
        if current:
            return {"name": current.name, "path": current.path, "description": current.description,
                    "personality": getattr(current, "personality", {}),
                    "traits": getattr(current, "traits", [])}
        return {"name": None}
    except Exception as e:
        return {"error": str(e)}


@router.get("/weights/snapshots")
async def list_weight_snapshots():
    """
    List saved trait weight snapshots.

    Returns:
        list of snapshot names (sorted)

    Side effects:
        - calls TraitWeightsConfig.list_snapshots()
    """
    try:
        from domains.context.managers import get_trait_config
        config = get_trait_config()
        return {"snapshots": config.list_snapshots()}
    except Exception as e:
        return {"snapshots": [], "error": str(e)}


@router.post("/weights/snapshot/{name}")
async def save_weight_snapshot(name: str):
    """
    Save current trait weights as a named snapshot.

    Args:
        name: snapshot label

    Returns:
        dict with ``path`` to saved snapshot

    Side effects:
        - writes snapshot JSON to disk
    """
    try:
        from domains.context.managers import get_trait_config
        config = get_trait_config()
        path = config.save_snapshot(name)
        return {"status": "saved", "path": path}
    except Exception as e:
        return {"error": str(e)}


@router.post("/weights/snapshot/{name}/load")
async def load_weight_snapshot(name: str):
    """
    Load trait weights from a named snapshot.

    Args:
        name: snapshot label

    Returns:
        dict with ``traits_loaded`` count

    Side effects:
        - overwrites current trait weights with snapshot values
    """
    try:
        from domains.context.managers import get_trait_config
        config = get_trait_config()
        count = config.load_snapshot(name)
        return {"status": "loaded", "traits_loaded": count}
    except Exception as e:
        return {"error": str(e)}


@router.delete("/weights/snapshot/{name}")
async def delete_weight_snapshot(name: str):
    """
    Delete a trait weight snapshot.

    Args:
        name: snapshot label

    Returns:
        dict with ``deleted`` bool

    Side effects:
        - removes snapshot file from disk
    """
    try:
        from domains.context.managers import get_trait_config
        config = get_trait_config()
        ok = config.delete_snapshot(name)
        return {"deleted": ok}
    except Exception as e:
        return {"error": str(e)}


@router.get("/stats")
async def get_soul_stats():
    """
    Get soul manager statistics (counts, last switch time, etc.).

    Returns:
        dict with soul manager stats

    Side effects:
        - calls SloManager.get_stats()
    """
    try:
        from domains.inference.slo_manager import get_slo_manager
        return get_slo_manager().get_stats()
    except Exception as e:
        return {"error": str(e)}
