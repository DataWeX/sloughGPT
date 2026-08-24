"""
Models Controller - Business logic for model management
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import logging
import time
import threading
import os

logger = logging.getLogger(__name__)

# ── HF Hub API cache (avoids repeated unreachable API calls) ─────────
_HF_CACHE_TTL = 300  # 5 minutes
_hf_models_cache: Optional[List[Dict[str, Any]]] = None
_hf_cache_timestamp: float = 0.0
_hf_cache_lock = threading.Lock()


class ModelsController:
    """Controller for model management"""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.models_dir = repo_root / "models"
        self._current_model: Optional[str] = None
        self._current_device: Optional[str] = None
        self._loaded_at: Optional[datetime] = None
        self._model_instance: Optional[Any] = None
        self._hf_model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._process_guard: Optional[Any] = None
        self._inference_count: int = 0
        self._total_tokens_generated: int = 0
        self._last_inference_time: Optional[float] = None
        self._is_inferencing: bool = False

    def _resolve_device(self, device: str) -> str:
        """Resolve device string for model placement: mps > cuda > cpu.

        Validates explicit ``cuda``/``mps`` requests against actual hardware
        availability and falls back to ``cpu`` (with a warning) when the
        requested backend is not present. A requested accelerator must never
        be reported as the active device when no such device exists.

        Args:
            device: Requested device (``auto``, ``cpu``, ``mps``, ``cuda``, or None).

        Returns:
            The resolved, actually-usable device string.
        """
        if device is None or device == "auto":
            try:
                from domains.infrastructure.ml_types import auto_device
                return auto_device()
            except Exception:
                return "cpu"
        try:
            from domains.infrastructure.ml_types import _cuda_available, _mps_available
        except Exception:
            _cuda_available = _mps_available = None
        if device == "cuda" and (_cuda_available is None or not _cuda_available()):
            logger.warning("device='cuda' requested but CUDA unavailable — falling back to cpu", extra={"tag": "MODEL"})
            return "cpu"
        if device == "mps" and (_mps_available is None or not _mps_available()):
            logger.warning("device='mps' requested but MPS unavailable — falling back to cpu", extra={"tag": "MODEL"})
            return "cpu"
        return device

    def _find_model_path(self, model_id: str) -> Optional[Path]:
        """Find model file by ID"""
        model_path = self.models_dir / f"{model_id}.gguf"
        if model_path.exists():
            return model_path

        for f in self.models_dir.glob("*.gguf"):
            if f.stem == model_id:
                return f

        return None

    def _infer_config(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Infer model config from state dict"""
        config = {}

        for key, value in state_dict.items():
            if "tok_emb.weight" in key:
                config["vocab_size"] = value.shape[0]
                config["n_embed"] = value.shape[1]
                config["block_size"] = value.shape[1]
            elif "blocks.0.attn.q_proj.weight" in key:
                config["n_embed"] = value.shape[1]
                config["n_layer"] = len([k for k in state_dict.keys() if k.startswith("blocks.") and ".norm1.weight" in k])

        return config

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        models = []

        local_dir = self.models_dir
        if local_dir.exists():
            for f in local_dir.glob("*.gguf"):
                models.append({
                    "model_id": f.stem,
                    "path": str(f),
                    "type": "gguf",
                    "size_mb": f.stat().st_size / (1024 * 1024),
                })

        return models

    def _load_hf_model(self, model_id: str, device: str) -> Dict[str, Any]:
        """Load a HuggingFace model via SloNet (pure NumPy inference).

        Converts safetensors → .slnc on first load, then loads via mmap.
        """
        if model_id.endswith('.gguf'):
            return self._load_gguf_model(model_id, device)

        import state as server_state

        logger.info("Loading %s into SloTransformer (pure NumPy)...", model_id, extra={"tag": "MODEL"})
        try:
            from domains.models.provider import setup_providers
            from config import ServerConfig
            cfg = ServerConfig.from_env()

            # Lazy guard-backed path: defer parent weight load when a
            # ProcessGuard + .slnc are available. The guard worker materializes
            # weights; the parent only loads on guard death (lazy _get_model).
            try:
                from config import get_process_guard_enabled
                from domains.infrastructure.safetensors_loader import _get_model_dir
                _slnc = _get_model_dir(model_id) / "model.slnc"
                use_lazy = (
                    cfg.lazy_guard_autoload
                    and get_process_guard_enabled()
                    and _slnc.exists()
                )
            except Exception:
                use_lazy = False
            if use_lazy:
                process_guard = self._build_process_guard(model_id)
                if process_guard is None:
                    raise RuntimeError(
                        f"Lazy load requested for {model_id} but ProcessGuard "
                        "could not be started"
                    )
                from domains.inference.slonet_provider import SloNetChatProvider
                lazy_provider = SloNetChatProvider.lazy_from_slnc(
                    str(_slnc),
                    model_id=model_id,
                    quantize=cfg.quantize_slonet,
                    quant_bits=cfg.quant_bits,
                    quant_mode=cfg.quant_mode,
                    quant_clip=cfg.quant_clip,
                )
                setup_providers(
                    slonet_provider=lazy_provider,
                    process_guard=process_guard,
                    quantize=cfg.quantize_slonet,
                    quant_bits=cfg.quant_bits,
                    quant_mode=cfg.quant_mode,
                )
                # Publish to server_state — provider set, model deliberately
                # None so the parent advertises no resident weights.
                server_state.provider = lazy_provider
                server_state.model_type = model_id
                server_state.model = None
                logger.info("SloNet provider registered lazily (guard-backed): %s", model_id,
                            extra={"tag": "MODEL"})
                return {
                    "model_id": model_id,
                    "type": "slonet",
                    "device": "cpu",
                    "total_parameters": 0,
                    "tokenizer_type": "SloNetChatProvider",
                    "lazy": True,
                }

            process_guard = self._build_process_guard(model_id)
            setup_providers(
                slonet_hf_id=model_id,
                process_guard=process_guard,
                quantize=cfg.quantize_slonet,
                quant_bits=cfg.quant_bits,
                quant_mode=cfg.quant_mode,
            )
            logger.info("SloNet provider registered: %s (quant=%s)",
                        model_id, f"int{cfg.quant_bits}" if cfg.quantize_slonet else "none", extra={"tag": "MODEL"})

            # Auto-select precision on GPU (fp16 benchmark)
            try:
                from domains.slolib.gpu import set_accelerator_precision
                active = set_accelerator_precision("auto")
                if active == "fp16":
                    logger.info("GPU precision set to fp16 (auto-selected via benchmark)",
                                extra={"tag": "MODEL"})
            except Exception:
                pass
        except Exception as e:
            logger.error("Failed to register SloNet provider for %s: %s", model_id, e, extra={"tag": "MODEL"})
            raise

        # Publish the SloNet provider to server_state so the inference/chat
        # readiness guards (``state.model is not None``) accept the loaded
        # model. Mirrors the autoload path in startup._autoload_model.
        try:
            from domains.models.provider import get_provider
            slonet_provider = get_provider("slonet-native") or get_provider("slonet")
            # setup_providers() logs-and-continues when the requested model fails
            # to load (e.g. missing .slnc), leaving a stale provider registered
            # from a previous load. Verify the provider actually belongs to the
            # requested model before publishing it to server_state.
            if slonet_provider is None or getattr(slonet_provider, "_model_id", None) != model_id:
                raise RuntimeError(
                    f"SloNet provider for {model_id} not registered after setup_providers "
                    f"(found: {getattr(slonet_provider, '_model_id', None)})"
                )
            server_state.model = slonet_provider
            server_state.provider = slonet_provider
            server_state.model_type = model_id

            # Mirror to the core ServerState singleton — the source for
            # get_health_score() — so /health/detailed reports a loaded model
            # consistently across both model slots (Bug D).
            from domains.infrastructure.server_state import get_server_state
            core = get_server_state()
            core.model.set(slonet_provider)
            core.model_type.set(model_id)
            logger.info("SloNet provider published to server_state and ServerState: %s", model_id, extra={"tag": "MODEL"})
        except Exception as e:
            logger.error("Failed to publish SloNet provider for %s to server_state: %s", model_id, e, extra={"tag": "MODEL"})
            raise

        return {
            "model_id": model_id,
            "type": "slonet",
            "device": "cpu",
            "total_parameters": 0,
            "tokenizer_type": "SloNetChatProvider",
        }

    def _build_process_guard(self, model_id: str) -> Optional[Any]:
        """Build and start a ``ProcessGuard`` for the given model, if enabled.

        Reads the runtime ProcessGuard toggle; skips when the model has no
        compiled ``.slnc`` file. Any existing guard is stopped first to avoid
        orphan worker subprocesses on reload.

        Args:
            model_id: HuggingFace model ID to guard

        Returns:
            Started ``ProcessGuard``, or None when disabled/unavailable.
        """
        if self._process_guard is not None:
            try:
                self._process_guard.stop()
            except Exception:
                pass
            self._process_guard = None

        from config import get_process_guard_enabled
        if not get_process_guard_enabled():
            return None

        try:
            from domains.infrastructure.process_guard import ProcessGuard, resolve_memory_limit_mb
            from domains.infrastructure.safetensors_loader import _get_model_dir
            from domains.models.provider import attach_process_guard_to_provider
            from config import ServerConfig
            cfg = ServerConfig.from_env()

            slnc_path = str(_get_model_dir(model_id) / "model.slnc")
            if not os.path.exists(slnc_path):
                logger.info("ProcessGuard skipped: no .slnc file at %s", slnc_path, extra={"tag": "MODEL"})
                return None

            guard = ProcessGuard(
                slnc_path=slnc_path,
                model_id=model_id,
                worker_id=f"slo-{model_id.split('/')[-1]}",
                max_restarts=3,
                restart_delay=2.0,
                generate_timeout=cfg.generate_timeout,
                memory_limit_mb=resolve_memory_limit_mb(slnc_path, cfg.process_guard_memory_limit_mb),
                quantize=cfg.quantize_slonet,
                quant_bits=cfg.quant_bits,
                quant_mode=cfg.quant_mode,
                quant_clip=cfg.quant_clip,
            )
            guard.start()
            self._process_guard = guard
            attach_process_guard_to_provider(guard)
            logger.info("ProcessGuard started for %s", model_id, extra={"tag": "MODEL"})
            return guard
        except Exception as e:
            logger.warning("ProcessGuard creation failed: %s", e, extra={"tag": "MODEL"})
            return None

    def _load_gguf_model(self, model_path: str, device: str) -> Dict[str, Any]:
        """Load a GGUF model using llama.cpp"""
        try:
            from llama_cpp import Llama

            logger.info(f"Loading GGUF model: {model_path}", extra={"tag": "MODEL"})

            # Determine device for llama.cpp
            if device == "mps" or device == "cuda":
                n_gpu_layers = 999 if device == "cuda" else 1
            else:
                n_gpu_layers = 0

            self._gguf_model = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=4096,
                verbose=False,
            )

            self._model_type = "gguf"

            return {
                "model_id": model_path,
                "type": "gguf",
                "device": device,
                "n_ctx": 4096,
            }
        except Exception as e:
            logger.error(f"Failed to load GGUF model {model_path}: {e}", extra={"tag": "MODEL"})
            raise

    def load_model(self, model_id: str, device: str = "auto", quantize: Optional[str] = None,
                    **kwargs) -> Dict[str, Any]:
        """Load a model into memory via SloNet (pure NumPy inference).

        Converts safetensors → .slnc on first load, then loads via mmap.
        """
        resolved_device = self._resolve_device(device)

        try:
            self._load_hf_model(model_id, resolved_device)
            self._current_model = model_id
            self._current_device = resolved_device
            self._loaded_at = datetime.now()
            return {
                "status": "loaded",
                "model_id": model_id,
                "type": "slonet",
                "device": resolved_device,
                "loaded_at": self._loaded_at.isoformat(),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    def _stop_process_guard(self) -> None:
        """Stop and drop any active ProcessGuard (orphan worker cleanup)."""
        if self._process_guard is not None:
            try:
                self._process_guard.stop()
            except Exception as e:
                logger.debug("ProcessGuard stop failed: %s", e)
            self._process_guard = None

    def _resolve_active_model_id(self) -> Optional[str]:
        """Resolve the currently active model id across load paths.

        Order: controller state, then the ModelRegistry default (the autoload
        path registers directly with the registry, bypassing the controller),
        then ``server_state.model_type`` as a final fallback.

        Returns:
            Active model id, or None when no model is loaded.
        """
        if self._current_model:
            return self._current_model
        try:
            from domains.infrastructure.model_registry import get_model_registry
            mid = get_model_registry().default_id
            if mid:
                return mid
        except Exception:
            pass
        try:
            import state as server_state
            if getattr(server_state, "model_type", None):
                return server_state.model_type
        except Exception:
            pass
        return None

    def adopt_process_guard(self, guard: Any, model_id: Optional[str] = None) -> None:
        """Adopt a ProcessGuard created outside this controller (autoload path).

        Any previously held guard is stopped first so a manual reload replaces
        it cleanly. When the controller has no loaded model of its own (models
        autoloaded via the registry bypass the controller), ``_current_model``
        is set from ``model_id`` so status reporting and unload work.

        Args:
            guard: A started ProcessGuard instance.
            model_id: The model id the guard protects.
        """
        self._stop_process_guard()
        self._process_guard = guard
        try:
            from domains.models.provider import attach_process_guard_to_provider
            attach_process_guard_to_provider(guard)
        except Exception as e:
            logger.debug("ProcessGuard provider attach failed: %s", e)
        if model_id and self._current_model is None:
            self._current_model = model_id
            self._current_device = getattr(guard, "device", "cpu") or "cpu"
            if self._loaded_at is None:
                from datetime import datetime
                self._loaded_at = datetime.now()
        logger.info("ProcessGuard adopted for %s", model_id or guard.worker_id, extra={"tag": "MODEL"})

    def get_process_guard_status(self) -> Dict[str, Any]:
        """Return current ProcessGuard state.

        Returns:
            Dict with ``enabled`` (runtime toggle), ``active`` (guard running),
            ``model_id`` (guarded model, if any), and ``health`` (guard health snapshot).
        """
        from config import get_process_guard_enabled
        active = self._process_guard is not None and getattr(self._process_guard, "alive", False)
        health = None
        if self._process_guard is not None:
            try:
                health = self._process_guard.health()
            except Exception:
                health = {"alive": False}
        return {
            "enabled": get_process_guard_enabled(),
            "active": active,
            "model_id": self._current_model or self._resolve_active_model_id(),
            "health": health,
        }

    def set_process_guard_enabled(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable ProcessGuard at runtime.

        When disabling, stops any active guard. When enabling and a model is
        loaded with a .slnc file, starts the guard immediately.

        Args:
            enabled: True to enable, False to disable

        Returns:
            Status dict with new enabled state.
        """
        from config import set_process_guard_enabled
        set_process_guard_enabled(enabled)

        if not enabled:
            self._stop_process_guard()
            logger.info("ProcessGuard disabled at runtime", extra={"tag": "MODEL"})
        else:
            logger.info("ProcessGuard enabled at runtime", extra={"tag": "MODEL"})
            # Try to start guard for the active model (manual or autoload path)
            active_model = self._resolve_active_model_id()
            if active_model and self._process_guard is None:
                try:
                    self._build_process_guard(active_model)
                except Exception as e:
                    logger.debug("ProcessGuard startup on enable failed: %s", e)

        return self.get_process_guard_status()

    def _resolve_base_model_id(self, target: Path) -> Optional[str]:
        """Derive the base HuggingFace model id from a fine-tuned config.json.

        Prefers ``_name_or_path`` (recorded by transformers during fine-tuning),
        falling back to ``_hf_text_config._name_or_path``. Returns None when the
        base id cannot be determined.

        Args:
            target: Local fine-tuned model directory

        Returns:
            Base HF model id string, or None
        """
        try:
            cfg_path = target / "config.json"
            if not cfg_path.exists():
                return None
            import json as _json
            cfg = _json.loads(cfg_path.read_text())
            name = cfg.get("_name_or_path")
            if name:
                return str(name)
            name = (cfg.get("_hf_text_config") or {}).get("_name_or_path")
            if name:
                return str(name)
        except Exception as e:
            logger.debug("Could not resolve base model id from %s: %s", target, e)
        return None

    def load_model_path(self, model_path: str, device: str = "cpu",
                        base_model_id: Optional[str] = None,
                        identity: Optional[str] = None) -> Dict[str, Any]:
        """Load a local fine-tuned model directory into chat via SloNet.

        Compiles ``config.json`` + ``model.safetensors`` to ``model.slnc`` on
        first load (mmap-backed, pure NumPy), then registers it as the
        ``slonet-native`` provider. The base tokenizer is resolved from
        ``base_model_id``, else from the ``_name_or_path`` recorded in the
        fine-tuned config.

        Args:
            model_path: Local directory containing a fine-tuned HF model
            device: Device hint (SloNet inference always runs on CPU)
            base_model_id: Base HuggingFace model ID for the tokenizer
            identity: Reported model identity. Defaults to the base model id,
                but callers loading a specific fine-tuned directory should pass
                the directory name so health/UI can distinguish the variant
                from the plain base model

        Returns:
            Dict with status/type/device/loaded_at (status ``"loaded"``) or
            status ``"error"`` with a message on failure
        """
        target = Path(model_path)
        if not target.is_dir():
            return {"status": "error", "error": f"Not a directory: {model_path}"}

        try:
            from config import ServerConfig
            cfg = ServerConfig.from_env()

            from domains.infrastructure.slnc.compiler import SLNCCompiler
            slnc_path = target / "model.slnc"
            if not slnc_path.exists():
                logger.info("Compiling fine-tuned model %s to .slnc ...",
                            model_path, extra={"tag": "MODEL"})
                SLNCCompiler().compile_from_directory(str(target), output=str(slnc_path))

            if base_model_id is None:
                base_model_id = self._resolve_base_model_id(target)
            tokenizer_model_id = base_model_id or target.name
            model_id = identity or tokenizer_model_id

            import state as server_state
            server_state.model_type = model_id

            process_guard = self._build_process_guard_for_path(slnc_path, model_id)

            from domains.models.provider import setup_providers
            setup_providers(
                slonet_hf_id=tokenizer_model_id,
                slonet_path=str(slnc_path),
                process_guard=process_guard,
                quantize=cfg.quantize_slonet,
                quant_bits=cfg.quant_bits,
                quant_mode=cfg.quant_mode,
            )
            logger.info("SloNet provider registered from local .slnc: %s (quant=%s)",
                        model_id, f"int{cfg.quant_bits}" if cfg.quantize_slonet else "none",
                        extra={"tag": "MODEL"})

            self._current_model = model_id
            self._current_device = "cpu"
            self._loaded_at = datetime.now()

            # The SloNet fine-tuned provider is served by the controller, not the
            # ModelRegistry. Drop any previously autoloaded/registered HF model
            # from the registry so the health endpoint (registry-first) reflects
            # this model instead of a stale default.
            try:
                from domains.infrastructure.model_registry import get_model_registry
                registry = get_model_registry()
                stale = registry.default_id
                if stale is not None and stale != model_id:
                    registry.unregister(stale)
                    logger.info("Unregistered stale registry model %s after fine-tuned load",
                                stale, extra={"tag": "MODEL"})
            except Exception as e:
                logger.debug("Registry stale-model cleanup failed: %s", e)

            return {
                "status": "loaded",
                "model_id": model_id,
                "type": "slonet",
                "device": "cpu",
                "loaded_at": self._loaded_at.isoformat(),
                "model_path": str(target),
                "slnc_path": str(slnc_path),
            }
        except Exception as e:
            logger.error("Failed to load fine-tuned model %s: %s", model_path, e,
                         extra={"tag": "MODEL"})
            return {"status": "error", "error": str(e)}

    def _build_process_guard_for_path(self, slnc_path: Path, model_id: str) -> Optional[Any]:
        """Build and start a ProcessGuard for an explicit .slnc path (if enabled).

        Unlike ``_build_process_guard`` (which resolves the .slnc from the HF
        cache), this uses the given file path directly, so it works for locally
        compiled fine-tuned models.

        Args:
            slnc_path: Path to the .slnc file to guard
            model_id: Model id used for the worker name

        Returns:
            Started ProcessGuard, or None when disabled/unavailable
        """
        self._stop_process_guard()

        from config import get_process_guard_enabled
        if not get_process_guard_enabled():
            return None
        try:
            from domains.infrastructure.process_guard import ProcessGuard, resolve_memory_limit_mb
            from config import ServerConfig
            cfg = ServerConfig.from_env()
            guard = ProcessGuard(
                slnc_path=str(slnc_path),
                model_id=model_id,
                worker_id=f"slo-{Path(model_id).name}-finetuned",
                max_restarts=3,
                restart_delay=2.0,
                generate_timeout=cfg.generate_timeout,
                memory_limit_mb=resolve_memory_limit_mb(str(slnc_path), cfg.process_guard_memory_limit_mb),
                quantize=cfg.quantize_slonet,
                quant_bits=cfg.quant_bits,
                quant_mode=cfg.quant_mode,
                quant_clip=cfg.quant_clip,
            )
            guard.start()
            self._process_guard = guard
            logger.info("ProcessGuard started for fine-tuned %s", model_id, extra={"tag": "MODEL"})
            return guard
        except Exception as e:
            logger.warning("ProcessGuard creation failed: %s", e, extra={"tag": "MODEL"})
            return None

    def unload_model(self) -> Dict[str, Any]:
        """Unload current model and clean up ModelRegistry entry.

        Resolves the active model id from the registry (the authoritative
        source for autoloaded models that bypass the controller), then:
            - stops the process guard
            - unregisters from the registry (tears down the ModelServer)
            - clears all registered providers so chat fails fast until reload
            - resets ``state.model`` / ``tokenizer`` / ``model_type``

        Returns:
            Dict with the unloaded ``model_id`` and ``status: "unloaded"``.
        """
        # Resolve the active model: controller state > registry default.
        model_id = self._current_model
        try:
            from domains.infrastructure.model_registry import get_model_registry
            registry = get_model_registry()
            model_id = model_id or registry.default_id
        except Exception:
            registry = None

        if self._process_guard is not None:
            try:
                self._process_guard.stop()
            except Exception as e:
                logger.debug("ProcessGuard stop failed: %s", e)
            self._process_guard = None

        if self._model_instance is not None:
            del self._model_instance
            self._model_instance = None

        # Unregister from ModelRegistry (tears down ModelServer, releases semaphore, etc.)
        if registry is not None and model_id:
            try:
                registry.unregister(model_id)
            except Exception as e:
                logger.debug("Registry unregister failed: %s", e)

        # Drop cross-turn KV states — keys/values from the unloaded model are invalid
        try:
            from domains.models.provider import get_provider
            provider = get_provider("slonet-native")
            if provider is None:
                provider = get_provider("slonet")
            if provider is not None and hasattr(provider, "clear_all_sessions"):
                provider.clear_all_sessions()
        except Exception as e:
            logger.debug("Cross-turn KV state clear failed: %s", e)

        # Clear all providers so chat/generation fail fast until a model reloads
        try:
            from domains.models.provider import clear_providers
            clear_providers()
        except Exception as e:
            logger.debug("Provider clear failed: %s", e)

        # Reset shared server state
        try:
            import state as server_state
            server_state.model = None
            server_state.tokenizer = None
            server_state.model_type = None
            server_state.provider = None
        except Exception as e:
            logger.debug("Server state reset failed: %s", e)

        # Reset the core ServerState singleton to match
        try:
            from domains.infrastructure.server_state import get_server_state
            core = get_server_state()
            core.model.set(None)
            core.tokenizer.set(None)
            core.model_type.set(None)
        except Exception as e:
            logger.debug("ServerState singleton reset failed: %s", e)

        if self._hf_model is not None:
            del self._hf_model
            self._hf_model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        try:
            from domains.training.slonet import _get_accelerator
            acc = _get_accelerator()
            if acc is not None and hasattr(acc, 'empty_cache'):
                acc.empty_cache()
        except Exception as e:
            logger.debug("Accelerator cache clear failed: %s", e)

        import gc
        gc.collect()

        result = {
            "model_id": model_id,
            "status": "unloaded",
        }
        self._current_model = None
        self._current_device = None
        self._loaded_at = None
        return result

    def get_current_model(self) -> Optional[Dict[str, Any]]:
        """Get current loaded model info"""
        if not self._current_model or not self._current_device:
            return None

        return {
            "model_id": self._current_model,
            "status": "loaded",
            "device": self._current_device or "cpu",
            "loaded_at": self._loaded_at.isoformat() if self._loaded_at else None,
        }

    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        return {
            "inference_count": self._inference_count,
            "total_tokens_generated": self._total_tokens_generated,
            "is_inferencing": self._is_inferencing,
            "last_inference_time": self._last_inference_time,
        }

    def record_inference_start(self):
        """Record inference start"""
        self._is_inferencing = True
        self._inference_count += 1
        self._last_inference_time = time.time()

    def record_inference_end(self, tokens_generated: int = 0):
        """Record inference end"""
        self._is_inferencing = False
        self._total_tokens_generated += tokens_generated

    def list_hf_models(self, q: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search HuggingFace Hub for causal LM models.

        Returns list of dicts with model_id, parameters (approx), vocab_size.
        Falls back to curated list if the API is unreachable.
        Results are cached for 5 minutes to avoid repeated API calls.
        Thread-safe: concurrent callers block until the first fetch completes.
        """
        global _hf_models_cache, _hf_cache_timestamp

        # Return cached result if fresh (no query filter — cache is for full list)
        if q is None:
            with _hf_cache_lock:
                if _hf_models_cache is not None and (time.monotonic() - _hf_cache_timestamp) < _HF_CACHE_TTL:
                    return _hf_models_cache

        # Hold lock during fetch to prevent thundering herd (3+ parallel calls)
        with _hf_cache_lock:
            # Double-check after acquiring lock (another thread may have populated)
            if q is None and _hf_models_cache is not None and (time.monotonic() - _hf_cache_timestamp) < _HF_CACHE_TTL:
                return _hf_models_cache

            import os
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            search = q or ""
            try:
                import requests
                url = "https://huggingface.co/api/models"
                params = {
                    "search": search,
                    "task": "text-generation",
                    "sort": "downloads",
                    "direction": -1,
                    "limit": 50,
                }
                resp = requests.get(url, params=params, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    models = []
                    for m in data:
                        if not isinstance(m, dict) or not m.get("id"):
                            continue
                        pid = m.get("id")
                        if m.get("pipeline_tag") not in ("text-generation", "text2text-generation"):
                            continue
                        config = m.get("config") or {}
                        params = int(m.get("num_parameters", 0) or self._estimate_params(pid))
                        vocab_raw = config.get("vocab_size") if isinstance(config, dict) else None
                        vocab_size = int(vocab_raw or 0)
                        models.append({
                            "model_id": pid,
                            "parameters": params,
                            "vocab_size": vocab_size,
                        })
                    if models and q is None:
                        _hf_models_cache = models
                        _hf_cache_timestamp = time.monotonic()
                    return models
            except Exception:
                logger.debug("HF Hub API unreachable, using cached or curated model list")

            # Return cached if available (even if stale)
            if q is None and _hf_models_cache is not None:
                return _hf_models_cache

        # Fallback: curated list sorted by capability (chat-tuned first)
        curated = [
            ("microsoft/Phi-3.5-mini-instruct", 3820000000, 32064),
            ("Qwen/Qwen2.5-1.5B-Instruct", 1540000000, 151936),
            ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1100000000, 32000),
            ("microsoft/phi-2", 2700000000, 51200),
            ("microsoft/Phi-3-mini-128k-instruct", 3820000000, 32064),
            ("Qwen/Qwen2-0.5B-Instruct", 465000000, 151936),
            ("gpt2-xl", 1558000000, 50257),
            ("gpt2-large", 774000000, 50257),
            ("gpt2-medium", 355000000, 50257),
            ("gpt2", 124000000, 50257),
            ("distilgpt2", 82000000, 50257),
        ]
        if q:
            return [{"model_id": m, "parameters": p, "vocab_size": v} for m, p, v in curated if q.lower() in m.lower()]
        result = [{"model_id": m, "parameters": p, "vocab_size": v} for m, p, v in curated]
        if q is None:
            with _hf_cache_lock:
                _hf_models_cache = result
                _hf_cache_timestamp = time.monotonic()
        return result

    @staticmethod
    def _estimate_params(model_id: str) -> int:
        """Estimate parameter count from model ID string."""
        m = model_id.lower()
        if any(x in m for x in ("13b", "12b")):
            return 13000000000
        if "7b" in m:
            return 7000000000
        if "3b" in m:
            return 3000000000
        if any(x in m for x in ("2.7b", "2b8")):
            return 2700000000
        if "1.5b" in m:
            return 1500000000
        if "1b" in m:
            return 1000000000
        if any(x in m for x in ("0.5b", "500m")):
            return 500000000
        if "350m" in m:
            return 350000000
        if "125m" in m:
            return 125000000
        return 0


_models_controller: Optional[ModelsController] = None


def get_models_controller() -> ModelsController:
    """Get models controller instance"""
    global _models_controller
    if _models_controller is None:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        _models_controller = ModelsController(repo_root)
    return _models_controller
