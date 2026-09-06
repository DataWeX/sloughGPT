"""
Slos Router - Personality/soul management endpoints

Encapsulates router state in ``SloRouterState`` dataclass rather than module-level
mutable globals. Actual soul state lives in ``SloManager`` singleton.
"""

import asyncio
import logging
import re
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from domains.infrastructure.errors import AppError
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from infrastructure.sse_fallback import sse_complete, sse_token
from pydantic import BaseModel, Field
from schemas.common import classify_and_raise, raise_error, safe_audit_log, success_response

# Response cache for list_souls: avoids FS glob + per-soul metadata parse.
_list_souls_cache: tuple[float, dict] | None = None
_LIST_SOULS_CACHE_TTL = 30.0
_list_souls_lock = threading.Lock()

try:
    from domains.models import SloughGPTModel
except ImportError:
    SloughGPTModel = None

logger = logging.getLogger(__name__)


@dataclass
class SloRouterState:
    """Encapsulated state for the souls router.

    Current soul tracking and model references. The real soul state
    lives in SloManager; this is a thin cache for switch operations.
    """

    current_soul: Any = None
    main_model: Any = None
    main_tokenizer: Any = None


class SwitchRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    checkpoint_name: str | None = Field(default=None, max_length=200)


class SloChatRequest(BaseModel):
    checkpoint_name: str = Field(..., min_length=1, max_length=100)
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_new_tokens: int = Field(default=100, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)


class SaveWeightsRequest(BaseModel):
    """Request body for saving trait weights."""

    personality: dict[str, float] | None = None
    cognition: dict[str, float] | None = None
    emotion: dict[str, float] | None = None


class SoulsRouter:
    """OOP router for soul personality/soul management endpoints.

    Wraps all endpoints and mutable state (``SloRouterState``) into a single
    class. Route handlers are instance methods registered via
    ``self.router.add_api_route(...)``.
    """

    _VALID_CKPT_NAME = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

    def __init__(self) -> None:
        self.state = SloRouterState()
        self.router = APIRouter(prefix="/souls", tags=["souls"])

        self.router.add_api_route("/chat", self.soul_chat, methods=["POST"], response_model=None)
        self.router.add_api_route("/switch", self.switch_soul, methods=["POST"])
        self.router.add_api_route("", self.list_souls, methods=["GET"])
        self.router.add_api_route("/weights", self.get_trait_weights, methods=["GET"])
        self.router.add_api_route("/weights", self.save_trait_weights, methods=["POST"])
        self.router.add_api_route("/weights/modes", self.get_trait_modes, methods=["GET"])
        self.router.add_api_route("/current", self.get_current_soul, methods=["GET"])
        self.router.add_api_route("/weights/snapshots", self.list_weight_snapshots, methods=["GET"])
        self.router.add_api_route(
            "/weights/snapshot/{name}", self.save_weight_snapshot, methods=["POST"]
        )
        self.router.add_api_route(
            "/weights/snapshot/{name}/load", self.load_weight_snapshot, methods=["POST"]
        )
        self.router.add_api_route(
            "/weights/snapshot/{name}", self.delete_weight_snapshot, methods=["DELETE"]
        )
        self.router.add_api_route("/stats", self.get_soul_stats, methods=["GET"])
        self.router.add_api_route("/{soul_name}", self.get_soul, methods=["GET"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_repo_root():
        """Return the repository root (4 levels up from this file)."""
        from pathlib import Path

        return Path(__file__).resolve().parents[4]

    def _load_slough_model(self, checkpoint_path, tie_weights=True):
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
        from domains.infrastructure.weight_loader import infer_arch_from_state_dict

        soul, sd = load_soul(checkpoint_path)
        if isinstance(sd, dict) and "tok_emb.weight" not in sd:
            sd = sd.get("weights", sd)
            if not isinstance(sd, dict):
                sd = sd.state_dict() if hasattr(sd, "state_dict") else {}

        arch = infer_arch_from_state_dict(sd)
        tie_weights = "lm_head.weight" not in sd

        model = SloughGPTModel(
            vocab_size=arch["vocab_size"],
            n_embed=arch["n_embed"],
            n_layer=arch["n_layer"],
            n_head=arch["n_head"],
            dropout=0.0,
            tie_weights=tie_weights,
            block_size=128,
            intermediate_size=arch["intermediate_size"],
        )

        model.load_state_dict(sd, strict=False)
        return model

    def _load_checkpoint_into_model(self, checkpoint_name: str) -> dict:
        """Load an auto-train checkpoint's weights into the main model/global state."""
        try:
            import sys

            main_mod = sys.modules.get("__main__")
            if main_mod is None:
                return {"status": "no_main_module"}

            checkpoints_dir = getattr(main_mod, "_REPO_ROOT", None) / "models" / "auto-training"
            if checkpoints_dir is None:
                return {"status": "no_repo_root"}

            if not self._VALID_CKPT_NAME.match(checkpoint_name) or ".." in checkpoint_name:
                return {"status": "invalid_name"}
            checkpoint_file = (checkpoints_dir / checkpoint_name).resolve()
            if not str(checkpoint_file).startswith(str(checkpoints_dir.resolve())):
                return {"status": "not_found", "path": str(checkpoint_file)}

            # Use SloNet import for .soul files
            from domains.training.slonet import import_from_sou

            try:
                soul_net = import_from_sou(str(checkpoint_file))
            except FileNotFoundError:
                return {"status": "not_found", "path": str(checkpoint_file)}
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
            logger.warning("Load checkpoint into model failed: %s", e)
            classify_and_raise(e, source="load_checkpoint_into_model")

    def _build_soul_system_prompt(self, soul_info) -> str:
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

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    async def soul_chat(
        self,
        req: SloChatRequest,
        request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> StreamingResponse:
        """Chat using a SloughGPTModel checkpoint (PyTorch-trained transformer).

        Loads the .soul file (PyTorch ZIP format), creates a SloughGPTModel with matching
        config, loads weights, and streams generated tokens autoregressively.

        Falls back to SloNet (NumPy) for checkpoints that don't match SloughGPTModel.
        """
        try:
            repo_root = self._get_repo_root()
            if not self._VALID_CKPT_NAME.match(req.checkpoint_name) or ".." in req.checkpoint_name:
                raise_error("Invalid checkpoint name", code="E_VAL_REQUEST")
            checkpoint_file = (
                repo_root / "models" / "auto-training" / (req.checkpoint_name + ".soul")
            ).resolve()
            if not checkpoint_file.exists():
                checkpoint_file = (repo_root / "models" / (req.checkpoint_name + ".soul")).resolve()

            if not str(checkpoint_file).startswith(str((repo_root / "models").resolve())):
                raise_error(f"Checkpoint not found: {req.checkpoint_name}", code="E_NOT_FOUND")

            # Load checkpoint into SloughGPTModel
            try:
                model = self._load_slough_model(checkpoint_file)
            except FileNotFoundError:
                raise_error(f"Checkpoint not found: {req.checkpoint_name}", code="E_NOT_FOUND")

            chars = list(
                " abcdefghijklmnopqrstuvwxyz0123456789.,!?':;-/\"@#$%^&*()[]{}<>~`|\\+=_\n\r\t"
            )
            # Extend with uppercase and remaining ASCII if model vocab is larger
            model_vocab = getattr(model, "vocab_size", len(chars))
            if model_vocab > len(chars):
                # Pad with remaining printable ASCII characters
                all_chars = [chr(i) for i in range(32, 127)]  # printable ASCII
                # Add common Unicode letters (basic Latin supplement)
                all_chars += [chr(i) for i in range(192, 256)]
                chars = all_chars[:model_vocab]
            stoi = {c: i for i, c in enumerate(chars)}
            itos = {i: c for i, c in enumerate(chars)}

            def encode(t) -> dict:
                """encode."""
                return [stoi.get(c.lower(), 0) for c in t]

            def decode(ids) -> dict:
                """decode."""
                return "".join(itos.get(i, "?") for i in ids)

            enc_prompt = encode(req.prompt)
            if len(enc_prompt) > 128:
                enc_prompt = enc_prompt[:128]

            async def stream() -> AsyncGenerator[str, None]:
                """stream."""
                temperature = max(0.1, req.temperature)
                top_p = max(0.1, min(1.0, req.top_p))

                generated = list(enc_prompt)

                for _ in range(req.max_new_tokens):
                    if await request.is_disconnected():
                        return

                    import numpy as np

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

                safe_audit_log(
                    "soul.chat",
                    resource=req.checkpoint_name,
                    detail=f"chars={len(decode(generated))}",
                )
                yield sse_complete("souls-chat", data={"response": decode(generated)})

            return StreamingResponse(stream(), media_type="text/event-stream")

        except Exception as e:
            logger.warning("Soul chat failed: %s", e)
            classify_and_raise(e, source="soul_chat")

    async def switch_soul(
        self,
        req: SwitchRequest,
        checkpoint_name: str | None = None,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Switch to a different soul and update ContextCore system prompt.

        If checkpoint_name is provided, also loads the corresponding checkpoint
        weights into the main chat model (baby model → inference engine).
        """
        try:
            import time as _time

            _switch_t0 = _time.monotonic()
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
                        soul_prompt = self._build_soul_system_prompt(soul_info)
                        ctx_core.set_system_prompt(soul_prompt)
                except Exception as e:
                    logger.warning(
                        "Failed to update context core system prompt on soul switch: %s",
                        e,
                        extra={"tag": "SOUL"},
                    )

                # Update PersonalityProcessor with soul traits
                try:
                    from domains.models.provider import update_personality_traits

                    personality = getattr(soul_info, "personality", {})
                    if personality:
                        update_personality_traits(personality)
                except Exception as e:
                    logger.debug(
                        "Failed to update personality processor: %s", e, extra={"tag": "SOUL"}
                    )

            # Load checkpoint into main model if requested
            if req.checkpoint_name:
                loaded = await asyncio.to_thread(
                    self._load_checkpoint_into_model, req.checkpoint_name
                )
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
                    logger.warning("Failed to set soul engine: %s", exc, extra={"tag": "SOUL"})

            try:
                from domains.infrastructure.server_state import get_server_state

                get_server_state().record_model_event("switch", req.name)
            except Exception:
                logger.debug("Failed to record model event", exc_info=True)

            # Record dashboard event
            try:
                from domains.infrastructure.event_buffer import get_event_buffer

                detail = f" checkpoint={req.checkpoint_name}" if req.checkpoint_name else ""
                get_event_buffer().record("SOUL", f"switched to {req.name}{detail}")
            except Exception as e:
                logger.debug("Failed to record dashboard event: %s", e)

            _switch_elapsed_ms = (_time.monotonic() - _switch_t0) * 1000
            safe_audit_log(
                "soul.switch",
                resource=req.name,
                detail=f"elapsed={_switch_elapsed_ms:.0f}ms checkpoint_loaded"
                if req.checkpoint_name
                else f"elapsed={_switch_elapsed_ms:.0f}ms",
                checkpoint_name=req.checkpoint_name or "",
            )

            return success_response(data=result)
        except Exception as e:
            logger.warning("Switch soul failed: %s", e)
            classify_and_raise(e, source="switch_soul")

    async def list_souls(self) -> dict:
        """
        List all available souls with name, description, and traits.

        Returns:
            dict with ``souls`` (list) and ``current_soul`` (name string or None)

        Side effects:
            - calls SloManager.list_souls() and get_current_soul()
        """
        global _list_souls_cache
        now = time.monotonic()
        with _list_souls_lock:
            if _list_souls_cache and (now - _list_souls_cache[0]) < _LIST_SOULS_CACHE_TTL:
                return _list_souls_cache[1]
        try:
            import asyncio

            from domains.inference.slo_manager import get_slo_manager

            manager = get_slo_manager()
            souls = await asyncio.to_thread(manager.list_souls)
            current = manager.get_current_soul()
            result = success_response(
                data=[
                    {
                        "name": s.name,
                        "path": s.path,
                        "description": s.description,
                        "personality": getattr(s, "personality", {}),
                        "traits": getattr(s, "traits", []),
                        "born_at": getattr(s, "born_at", ""),
                        "training_dataset": getattr(s, "training_dataset", ""),
                        "epochs_trained": getattr(s, "epochs_trained", 0),
                        "final_train_loss": getattr(s, "final_train_loss", None),
                        "final_val_loss": getattr(s, "final_val_loss", None),
                        "lineage": getattr(s, "lineage", ""),
                        "base_model": getattr(s, "base_model", ""),
                        "version": getattr(s, "version", ""),
                        "size_mb": getattr(s, "size_mb", 0.0),
                        "behavior": getattr(s, "behavior", {}),
                        "cognition": getattr(s, "cognition", {}),
                        "emotion": getattr(s, "emotion", {}),
                        "generation_params": getattr(s, "generation_params", {}),
                    }
                    for s in souls
                ],
                meta={"current_soul": current.name if current else None},
            )
            with _list_souls_lock:
                _list_souls_cache = (now, result)
            return result
        except Exception as e:
            classify_and_raise(e, source="list_souls")

    async def get_soul(self, soul_name: str) -> dict:
        """Get details for a specific soul by name."""
        try:
            import asyncio

            from domains.inference.slo_manager import get_slo_manager

            manager = get_slo_manager()
            souls = await asyncio.to_thread(manager.list_souls)
            for s in souls:
                if s.name == soul_name:
                    return success_response(
                        data={
                            "name": s.name,
                            "path": s.path,
                            "description": s.description,
                            "personality": getattr(s, "personality", {}),
                            "traits": getattr(s, "traits", []),
                            "born_at": getattr(s, "born_at", ""),
                            "training_dataset": getattr(s, "training_dataset", ""),
                            "epochs_trained": getattr(s, "epochs_trained", 0),
                            "final_train_loss": getattr(s, "final_train_loss", None),
                            "final_val_loss": getattr(s, "final_val_loss", None),
                            "lineage": getattr(s, "lineage", ""),
                            "base_model": getattr(s, "base_model", ""),
                            "version": getattr(s, "version", ""),
                            "size_mb": getattr(s, "size_mb", 0.0),
                            "behavior": getattr(s, "behavior", {}),
                            "cognition": getattr(s, "cognition", {}),
                            "emotion": getattr(s, "emotion", {}),
                            "generation_params": getattr(s, "generation_params", {}),
                        }
                    )
            raise_error(f"Soul '{soul_name}' not found", "E_NOT_FOUND", status_code=404)
        except AppError as e:
            classify_and_raise(e, source="souls.get_soul")
        except Exception as e:
            logger.warning("Get soul failed: %s", e)
            classify_and_raise(e, source="get_soul")

    async def get_trait_weights(self) -> dict:
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
            return success_response(data=weights)
        except Exception as e:
            logger.warning("Get trait weights failed: %s", e)
            classify_and_raise(e, source="get_trait_weights")

    async def save_trait_weights(
        self, body: SaveWeightsRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
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
            flat: dict[str, float] = {}
            for group in ("personality", "cognition", "emotion"):
                traits = getattr(body, group, None)
                if traits:
                    for k, v in traits.items():
                        flat[k] = float(v)
            config.set_many(flat)
            safe_audit_log(
                "soul.weights.save",
                resource="traits",
                detail=f"traits_saved={len(flat)}",
                groups=[
                    g for g in ("personality", "cognition", "emotion") if getattr(body, g, None)
                ],
            )
            return success_response(message="saved")
        except Exception as e:
            logger.warning("Save trait weights failed: %s", e)
            classify_and_raise(e, source="save_trait_weights")

    async def get_trait_modes(self) -> dict:
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
                MemoryManager,
                PersonalityManager,
                StyleManager,
                TaskManager,
                get_trait_config,
            )

            config = get_trait_config()
            return success_response(
                data={
                    "personality": PersonalityManager(config).get_mode(),
                    "memory": MemoryManager(config).get_mode(),
                    "style": StyleManager(config).get_mode(),
                    "task": TaskManager(config).get_mode(),
                }
            )
        except Exception as e:
            logger.warning("Get trait modes failed: %s", e)
            classify_and_raise(e, source="get_trait_modes")

    async def get_current_soul(self) -> dict:
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
                return success_response(
                    data={
                        "name": current.name,
                        "path": current.path,
                        "description": current.description,
                        "personality": getattr(current, "personality", {}),
                        "traits": getattr(current, "traits", []),
                    }
                )
            return success_response(data={"name": None})
        except Exception as e:
            logger.warning("Get current soul failed: %s", e)
            classify_and_raise(e, source="get_current_soul")

    async def list_weight_snapshots(self) -> dict:
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
            return success_response(data=config.list_snapshots())
        except Exception as e:
            classify_and_raise(e, source="list_weight_snapshots")

    async def save_weight_snapshot(
        self, name: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
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
            safe_audit_log("weights.snapshot.save", resource=name)
            return success_response(data={"path": path}, message="saved")
        except Exception as e:
            logger.warning("Save weight snapshot failed: %s", e)
            classify_and_raise(e, source="save_weight_snapshot")

    async def load_weight_snapshot(
        self, name: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
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
            safe_audit_log("weights.snapshot.load", resource=name, detail=f"traits_loaded={count}")
            return success_response(data={"traits_loaded": count}, message="loaded")
        except Exception as e:
            logger.warning("Load weight snapshot failed: %s", e)
            classify_and_raise(e, source="load_weight_snapshot")

    async def delete_weight_snapshot(
        self, name: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
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
            safe_audit_log("weights.snapshot.delete", resource=name, detail=f"deleted={ok}")
            return success_response(data={"deleted": ok})
        except Exception as e:
            logger.warning("Delete weight snapshot failed: %s", e)
            classify_and_raise(e, source="delete_weight_snapshot")

    async def get_soul_stats(self) -> dict:
        """
        Get soul manager statistics (counts, last switch time, etc.).

        Returns:
            dict with soul manager stats

        Side effects:
            - calls SloManager.get_stats()
        """
        try:
            from domains.inference.slo_manager import get_slo_manager

            return success_response(data=get_slo_manager().get_stats())
        except Exception as e:
            logger.warning("Get soul stats failed: %s", e)
            classify_and_raise(e, source="get_soul_stats")


router = SoulsRouter().router
