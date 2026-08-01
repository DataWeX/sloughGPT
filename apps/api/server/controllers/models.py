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
import numpy as np

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
        """Resolve device string for PyTorch model placement: mps > cuda > cpu."""
        if device is None or device == "auto":
            try:
                from domains.infrastructure.ml_types import auto_device
                return auto_device()
            except Exception:
                return "cpu"
        return device

    def _find_model_path(self, model_id: str) -> Optional[Path]:
        """Find model file by ID"""
        model_path = self.models_dir / f"{model_id}.pt"
        if model_path.exists():
            return model_path

        model_path = self.models_dir / f"{model_id}.gguf"
        if model_path.exists():
            return model_path

        for f in self.models_dir.glob("*.pt"):
            if f.stem == model_id:
                return f
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
            for f in local_dir.glob("*.pt"):
                models.append({
                    "model_id": f.stem,
                    "path": str(f),
                    "type": "pt",
                    "size_mb": f.stat().st_size / (1024 * 1024),
                })
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
        No PyTorch dependency at inference time.
        """
        if model_id.endswith('.gguf'):
            return self._load_gguf_model(model_id, device)

        import state as server_state
        server_state.model_type = model_id

        logger.info("Loading %s into SloTransformer (pure NumPy)...", model_id, extra={"tag": "MODEL"})
        try:
            from domains.models.provider import setup_providers
            from domains.infrastructure.config import get_config
            cfg = get_config()
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

        return {
            "model_id": model_id,
            "type": "slonet",
            "device": "cpu",
            "total_parameters": 0,
            "tokenizer_type": "SloNetChatProvider",
        }

    def _build_process_guard(self, model_id: str) -> Optional[Any]:
        """Build and start a ``ProcessGuard`` for the given model, if enabled.

        Reads ``SLO_ENABLE_PROCESS_GUARD`` via ``ServerConfig``; skips when the
        model has no compiled ``.slnc`` file. Any existing guard is stopped
        first to avoid orphan worker subprocesses on reload.

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

        try:
            from config import ServerConfig
            server_cfg = ServerConfig.from_env()
        except Exception:
            server_cfg = None
        enabled = bool(
            server_cfg.enable_process_guard
            if server_cfg is not None
            else os.getenv("SLO_ENABLE_PROCESS_GUARD", "").lower() in ("1", "true", "yes")
        )
        if not enabled:
            return None

        try:
            from domains.infrastructure.process_guard import ProcessGuard
            from domains.infrastructure.safetensors_loader import _get_model_dir

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
                generate_timeout=120.0,
                quantize=getattr(server_cfg, "quantize_slonet", False),
                quant_bits=getattr(server_cfg, "quant_bits", 8),
                quant_mode=getattr(server_cfg, "quant_mode", "symmetric"),
                quant_clip=getattr(server_cfg, "quant_clip", 0.999),
            )
            guard.start()
            self._process_guard = guard
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
        No PyTorch dependency at inference time.
        """
        resolved_device = self._resolve_device(device)

        try:
            result = self._load_hf_model(model_id, resolved_device)
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

    def unload_model(self) -> Dict[str, Any]:
        """Unload current model and clean up ModelRegistry entry."""
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
        if self._current_model:
            try:
                from domains.infrastructure.model_registry import get_model_registry
                registry = get_model_registry()
                registry.unregister(self._current_model)
            except Exception as e:
                logger.debug("Registry unregister failed: %s", e)

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
            "model_id": self._current_model,
            "status": "unloaded",
        }
        self._current_model = None
        self._current_device = None
        self._loaded_at = None
        return result

    def get_current_model(self) -> Optional[Dict[str, Any]]:
        """Get current loaded model info"""
        if not self._current_model:
            return None

        return {
            "model_id": self._current_model,
            "status": "loaded",
            "device": self._current_device,
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
                        params = m.get("num_parameters", 0) or self._estimate_params(pid)
                        vocab_size = (config.get("vocab_size") if isinstance(config, dict) else None) or 0
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
