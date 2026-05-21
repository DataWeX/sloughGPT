"""
Models Controller - Business logic for model management
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


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
        self._inference_count: int = 0
        self._total_tokens_generated: int = 0
        self._last_inference_time: Optional[float] = None
        self._is_inferencing: bool = False
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string for PyTorch model placement.

        Always returns "cpu" for reliability. MPS (Apple Silicon) is avoided
        because KV cache accumulates across requests, leading to OOM crashes
        after ~10 sequential inferences on 8 GB systems.
        """
        return "cpu"
    
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
        """Load a HuggingFace model (delegates to infrastructure model_loader for MPS/BFloat16 safety)"""
        if model_id.endswith('.gguf') or model_id.startswith('/'):
            return self._load_gguf_model(model_id, device)

        try:
            from domains.infrastructure.model_loader import load_hf_model as safe_load

            resolved_device = self._resolve_device(device)
            model, tokenizer, actual_device = safe_load(model_id, resolved_device)

            self._tokenizer = tokenizer
            self._hf_model = model

            total_params = sum(p.numel() for p in model.parameters())

            # Update server_state so health endpoints reflect the active model
            import state as server_state
            server_state.model = model
            server_state.tokenizer = tokenizer
            server_state.model_type = model_id

            # Create InferenceEngine for the new model so the inference-engine
            # provider (which is preferred over hf-default) works with current weights.
            inference_engine = None
            try:
                from domains.inference.engine import InferenceEngine
                inference_engine = InferenceEngine(
                    model=model,
                    tokenizer=tokenizer,
                    device=actual_device,
                )
            except Exception as e:
                logger.warning("Failed to create InferenceEngine: %s", e)

            # Register all providers via setup_providers (handles priority,
            # hf-default registration, and default router wiring).
            try:
                from domains.models.provider import setup_providers
                setup_providers(
                    model, tokenizer,
                    hf_model_id=model_id,
                    inference_engine=inference_engine,
                )
                logger.info("Updated providers for: %s (text=inference-engine)", model_id)
            except Exception as e:
                logger.warning("Failed to set up model providers: %s", e)

            # Push the new engine into ChatDomain so streaming chat uses it.
            try:
                from domains.chat.domain import get_chat_domain
                cd = get_chat_domain()
                if inference_engine is not None:
                    cd.set_engine(inference_engine)
                else:
                    cd.set_engine(None)
            except Exception as e:
                logger.warning("Failed to inject engine into ChatDomain: %s", e)

            return {
                "model_id": model_id,
                "type": "huggingface",
                "device": actual_device,
                "total_parameters": total_params,
                "tokenizer_type": type(tokenizer).__name__,
            }
        except Exception as e:
            logger.error(f"Failed to load HF model {model_id}: {e}")
            raise
    
    def _load_gguf_model(self, model_path: str, device: str) -> Dict[str, Any]:
        """Load a GGUF model using llama.cpp"""
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading GGUF model: {model_path}")
            
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
            logger.error(f"Failed to load GGUF model {model_path}: {e}")
            raise
    
    def load_model(self, model_id: str, device: str = "auto", quantize: Optional[str] = None) -> Dict[str, Any]:
        """Load a model into memory (local or HuggingFace)"""
        resolved_device = self._resolve_device(device)
        
        # Treat any model_id as a HuggingFace model first.
        # Falls back to local file if HuggingFace loading fails.
        try:
            result = self._load_hf_model(model_id, resolved_device)
            self._current_model = model_id
            self._current_device = resolved_device
            self._loaded_at = datetime.now()
            return {
                "status": "loaded",
                "model_id": model_id,
                "type": "huggingface",
                "device": resolved_device,
                "loaded_at": self._loaded_at.isoformat(),
            }
        except Exception:
            pass

        # Try local model
        logger.info("HF load failed or model not found; trying local path: %s", model_id)
        model_path = self._find_model_path(model_id)
        if model_path is None:
            return {
                "status": "error",
                "error": f"Model not found: {model_id}",
            }
        
        try:
            if model_path.suffix == ".pt":
                import torch
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                
                if isinstance(checkpoint, dict):
                    raw_state_dict = checkpoint.get("model", checkpoint)
                    config = checkpoint.get("config", {})
                else:
                    raw_state_dict = checkpoint
                    config = {}
                
                state_dict = {}
                for k, v in raw_state_dict.items():
                    if hasattr(v, 'cpu'):
                        state_dict[k] = v.cpu().numpy().astype(np.float32)
                    else:
                        state_dict[k] = np.array(v, dtype=np.float32)
                
                if not config:
                    config = self._infer_config(state_dict)
                
                model_type = config.get("model_type", "sloughgpt")
                
                from domains.models import SloughGPTModel
                
                model = SloughGPTModel(
                    vocab_size=config.get("vocab_size", 256),
                    n_embed=config.get("n_embed", 256),
                    n_layer=config.get("n_layer", 6),
                    n_head=config.get("n_head", 8),
                    block_size=config.get("block_size", 128),
                    max_seq_len=config.get("max_seq_len", 2048),
                )
                model.load_state_dict(state_dict, strict=False)
                model = model.to(resolved_device)
                model.eval()
                
                self._model_instance = model
            else:
                return {
                    "status": "error",
                    "error": "GGUF loading not implemented in controller",
                }
            
            self._current_model = model_id
            self._current_device = resolved_device
            self._loaded_at = datetime.now()
            
            return {
                "status": "loaded",
                "model_id": model_id,
                "device": resolved_device,
                "quantize": quantize,
                "path": str(model_path),
                "loaded_at": self._loaded_at.isoformat(),
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
    
    def unload_model(self) -> Dict[str, Any]:
        """Unload current model"""
        if self._model_instance is not None:
            del self._model_instance
            self._model_instance = None
        
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
        except Exception:
            pass
        
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
    
    def list_hf_models(self, q: Optional[str] = None) -> List[str]:
        """Search HuggingFace Hub for causal LM models.

        Queries the HuggingFace Hub API (``api-inference.huggingface.co``) for
        ``text-generation`` models sorted by monthly downloads. Falls back to a
        curated list if the API is unreachable.
        """
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
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    m["id"] for m in data
                    if isinstance(m, dict) and m.get("id")
                    and m.get("pipeline_tag") in ("text-generation", "text2text-generation")
                ]
                if models:
                    return models
        except Exception:
            logger.warning("HF Hub API unreachable, using curated model list")

        # Fallback: curated list sorted by capability (chat-tuned first)
        curated = [
            "microsoft/Phi-3.5-mini-instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/phi-2",
            "microsoft/Phi-3-mini-128k-instruct",
            "Qwen/Qwen2-0.5B-Instruct",
            "gpt2-xl",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
            "distilgpt2",
        ]
        if q:
            return [m for m in curated if q.lower() in m.lower()]
        return curated


_models_controller: Optional[ModelsController] = None


def get_models_controller() -> ModelsController:
    """Get models controller instance"""
    global _models_controller
    if _models_controller is None:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        _models_controller = ModelsController(repo_root)
    return _models_controller