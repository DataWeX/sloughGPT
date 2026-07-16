"""
Unified Model Loader — single interface for SloNet and HuggingFace models.

Provides a common `ModelLoader` class that handles:
  - Model detection (SloNet .slnc vs HuggingFace safetensors)
  - Verification (test inference on load)
  - Provider registration (SloNet or HF provider)
  - Quantization (via walk_slo_linears or walk_hf_linears)

Usage:
    from domains.infrastructure.model_loader import ModelLoader

    loader = ModelLoader()
    result = loader.load("gpt2", device="cpu")
    if result.success:
        print(f"Loaded {result.model_type} with {result.provider}")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

logger = logging.getLogger("slo.infrastructure.model_loader")


@dataclass
class LoadResult:
    """Standardized result from model loading."""
    success: bool
    model_id: str
    model_type: str  # "slonet" or "huggingface"
    provider: Any = None
    model: Any = None
    tokenizer: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class ModelLoader:
    """Unified loader for SloNet and HuggingFace models.

    Detects model format and routes to the appropriate loader.
    Provides common verification and registration logic.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path("models")

    def load(
        self,
        model_id: str,
        device: str = "auto",
        quantize: bool = False,
        quant_bits: int = 8,
        quant_mode: str = "symmetric",
        verify: bool = True,
    ) -> LoadResult:
        """Load a model by ID, detecting format automatically.

        Args:
            model_id: Model identifier (e.g., "gpt2", "meta-llama/Llama-3-8B")
            device: Target device ("auto", "cpu", "mps", "cuda")
            quantize: Apply quantization after loading
            quant_bits: Bits for quantization (8 or 4)
            quant_mode: "symmetric" or "asymmetric"
            verify: Run test inference to verify model works

        Returns:
            LoadResult with provider, model, tokenizer, and metrics
        """
        # Try SloNet first (.slnc file)
        slnc_result = self._try_load_slnc(model_id, device, quantize, quant_bits, quant_mode)
        if slnc_result is not None:
            if verify and slnc_result.success:
                self._verify_model(slnc_result)
            return slnc_result

        # Fall back to HuggingFace
        hf_result = self._try_load_hf(model_id, device, quantize, quant_bits, quant_mode)
        if verify and hf_result.success:
            self._verify_model(hf_result)
        return hf_result

    def _resolve_attn_kwargs(self) -> dict:
        """Resolve attention implementation kwargs based on config.

        Returns a dict with ``attn_implementation`` when flash attention is
        requested and available, otherwise an empty dict.  Flash attention 2
        requires:
          - transformers >= 4.35
          - torch >= 2.0
          - CUDA-capable GPU (A100, V100, RTX 30xx+) or compatible hardware
        On CPU / MPS the kwarg is silently omitted and the default (eager /
        SDPA math backend) is used.
        """
        try:
            import torch
            from domains.infrastructure.config import get_config
            cfg = get_config()
            if not cfg.model.use_flash_attention:
                return {}
            if not torch.cuda.is_available():
                logger.info("Flash attention requested but no CUDA device — skipping", extra={"tag": "MODEL"})
                return {}
            if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                logger.info("Flash attention requested but torch SDPA not available — skipping", extra={"tag": "MODEL"})
                return {}
            logger.info("Using flash attention 2 for %s", cfg.model.name, extra={"tag": "MODEL"})
            return {"attn_implementation": "flash_attention_2"}
        except Exception as e:
            logger.debug("Flash attention detection failed: %s", e)
            return {}

    def _try_load_slnc(
        self,
        model_id: str,
        device: str,
        quantize: bool,
        quant_bits: int,
        quant_mode: str,
    ) -> Optional[LoadResult]:
        """Try to load a SloNet model from .slnc file.

        Returns None if no .slnc file found, otherwise LoadResult.
        """
        try:
            from domains.infrastructure.safetensors_loader import _get_model_dir
            cache_dir = _get_model_dir(model_id)
        except Exception:
            return None

        slnc_path = cache_dir / "model.slnc"
        if not slnc_path.exists():
            # Try auto-conversion from safetensors
            slnc_path = self._try_convert_to_slnc(cache_dir, model_id)
            if slnc_path is None:
                return None

        logger.info("Loading SloNet model from %s", slnc_path, extra={"tag": "MODEL"})
        try:
            from domains.inference.slonet_provider import SloNetChatProvider

            provider = SloNetChatProvider.from_slnc(
                str(slnc_path),
                model_id=model_id,
                quantize=quantize,
                quant_bits=quant_bits,
                quant_mode=quant_mode,
            )

            return LoadResult(
                success=True,
                model_id=model_id,
                model_type="slonet",
                provider=provider,
                model=provider._model,
                tokenizer=getattr(provider, "_tokenizer", None),
                metrics={
                    "slnc_path": str(slnc_path),
                    "quantized": quantize,
                    "quant_bits": quant_bits if quantize else None,
                },
            )
        except Exception as e:
            logger.warning("SloNet load failed: %s", e, extra={"tag": "MODEL"})
            return LoadResult(
                success=False,
                model_id=model_id,
                model_type="slonet",
                error=str(e),
            )

    def _try_convert_to_slnc(self, cache_dir: Path, model_id: str) -> Optional[Path]:
        """Try to convert safetensors to .slnc format.

        Returns path to new .slnc file, or None if conversion fails.
        Handles bfloat16 weights by reading raw bytes and converting to float32.
        """
        try:
            from domains.infrastructure.safetensors_loader import (
                _find_safetensors,
                load_model_config,
            )
            import json as _json
            import struct

            st_path = _find_safetensors(cache_dir)
            if st_path is None:
                return None

            slnc_path = cache_dir / "model.slnc"
            logger.info("Converting %s to .slnc", st_path.name, extra={"tag": "MODEL"})

            # Load config
            config = load_model_config(model_id)

            # Load weights — handle bfloat16 via raw byte reading
            weights = {}
            with open(str(st_path), "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = _json.loads(f.read(header_len))
                for key, info in header.items():
                    if key.startswith("__"):
                        continue
                    dtype_str = info["dtype"]
                    offsets = info["data_offsets"]
                    f.seek(8 + header_len + offsets[0])
                    raw = f.read(offsets[1] - offsets[0])

                    if dtype_str == "BF16":
                        # bfloat16 → float32: view as uint16, shift left 16 bits
                        arr = np.frombuffer(raw, dtype=np.uint16)
                        f32 = np.zeros(len(arr), dtype=np.float32)
                        f32.view(np.uint32)[:][:] = arr.astype(np.uint32) << 16
                        weights[key] = f32.reshape(info["shape"])
                    elif dtype_str == "F32":
                        weights[key] = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])
                    elif dtype_str == "F16":
                        weights[key] = np.frombuffer(raw, dtype=np.float16).reshape(info["shape"]).astype(np.float32)
                    elif dtype_str == "I64":
                        weights[key] = np.frombuffer(raw, dtype=np.int64).reshape(info["shape"])
                    elif dtype_str == "I32":
                        weights[key] = np.frombuffer(raw, dtype=np.int32).reshape(info["shape"])
                    else:
                        # Try numpy directly
                        weights[key] = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])

            from domains.infrastructure.slnc.compiler import SLNCCompiler
            compiler = SLNCCompiler()
            compiler.compile_from_dict(config, weights, str(slnc_path))

            logger.info("Converted to .slnc: %s", slnc_path, extra={"tag": "MODEL"})
            return slnc_path
        except Exception as e:
            logger.warning("SloNet conversion failed: %s", e, extra={"tag": "MODEL"})
            return None

    def _try_load_hf(
        self,
        model_id: str,
        device: str,
        quantize: bool,
        quant_bits: int,
        quant_mode: str,
    ) -> LoadResult:
        """Load a HuggingFace model via transformers.

        Returns LoadResult with HFModelProvider.
        """
        logger.info("Loading HuggingFace model: %s", model_id, extra={"tag": "MODEL"})
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Resolve device
            if device == "auto":
                device = self._resolve_device()

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Add a distinct pad token to avoid the "pad_token == eos_token"
            # warning.  The model never generates this token — it only appears
            # when padding sequences to equal length for batched generation.
            if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

            # Resolve attention implementation
            attn_kwargs = self._resolve_attn_kwargs()

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
                device_map=None,
                **attn_kwargs,
            )

            # Resize embeddings if we added a new pad token
            if len(tokenizer) != model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
                model.config.pad_token_id = tokenizer.pad_token_id
            # Sync pad token to generation config (used by model.generate)
            model.generation_config.pad_token_id = tokenizer.pad_token_id

            model = model.to(device)
            model.eval()

            # Create provider
            from domains.models.provider import HFModelProvider
            provider = HFModelProvider(model, tokenizer, model_id_str=model_id)

            # Apply quantization if requested
            if quantize:
                self._quantize_hf_model(provider, quant_bits, quant_mode)

            return LoadResult(
                success=True,
                model_id=model_id,
                model_type="huggingface",
                provider=provider,
                model=model,
                tokenizer=tokenizer,
                metrics={
                    "device": device,
                    "quantized": quantize,
                    "quant_bits": quant_bits if quantize else None,
                    "parameters": sum(p.numel() for p in model.parameters()),
                },
            )
        except Exception as e:
            logger.warning("HuggingFace load failed: %s", e, extra={"tag": "MODEL"})
            return LoadResult(
                success=False,
                model_id=model_id,
                model_type="huggingface",
                error=str(e),
            )

    def _resolve_device(self) -> str:
        """Resolve best available device: mps (arm64 only) > cuda > cpu.

        Intel Macs skip MPS — PyTorch can report it available but it
        crashes at runtime during actual inference on x86_64.
        """
        try:
            import platform
            is_apple_silicon = platform.machine() in ("arm64", "aarch64")
            import torch
            if is_apple_silicon and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _verify_model(self, result: LoadResult) -> bool:
        """Run lightweight verification to confirm model loads correctly.

        Uses a single forward pass (not autoregressive generation) to verify
        the model processes input without errors. ~1-2s vs ~12s for generate().
        """
        try:
            if result.model_type == "slonet":
                # SloNet: single forward pass on raw model (skip tokenizer/generate)
                import numpy as _np
                test_ids = _np.array([[1, 2, 3]], dtype=_np.int64)
                output = result.model.forward(test_ids)
            else:
                # HF model: single forward pass (no generation loop)
                import torch
                test_input = torch.tensor([[1, 2, 3]], dtype=torch.long)
                if hasattr(result.model, "device"):
                    test_input = test_input.to(result.model.device)
                with torch.no_grad():
                    output = result.model(test_input)

            if output is None:
                result.success = False
                result.error = "Model produced empty output"
                return False

            result.metrics["verified"] = True
            logger.info("Model verification passed: %s", result.model_id, extra={"tag": "MODEL"})
            return True
        except Exception as e:
            logger.warning("Model verification failed: %s", e, extra={"tag": "MODEL"})
            result.metrics["verified"] = False
            return False

    def _quantize_hf_model(self, provider: Any, bits: int, mode: str) -> None:
        """Quantize all nn.Linear layers in a HuggingFace model.

        Uses the same quantization engine as SloNet, but targets
        HuggingFace's nn.Linear layers instead of SloLinear.
        """
        try:
            from domains.infrastructure.quantization import QuantEngine

            model = provider._model
            engine = QuantEngine(bits=bits, mode=mode)

            # Find all nn.Linear layers
            linear_layers = {}
            for name, module in model.named_modules():
                if module.__class__.__name__ == "Linear":
                    linear_layers[name] = module

            # Quantize each layer
            quantized_count = 0
            for name, module in linear_layers.items():
                if hasattr(module, "weight") and module.weight is not None:
                    weight = module.weight.data.cpu().numpy()
                    info = engine.quantize(f"{name}.weight", weight)
                    if info.is_quantized:
                        # Store quantized weight
                        module._quant_info = info
                        quantized_count += 1

            provider._quant_engine = engine
            logger.info("Quantized %d/%d layers in %s",
                       quantized_count, len(linear_layers), provider._model_id,
                       extra={"tag": "MODEL"})
        except Exception as e:
            logger.warning("HF quantization failed: %s", e, extra={"tag": "MODEL"})

    def walk_hf_linears(self, model: Any) -> Dict[str, Any]:
        """Find all nn.Linear layers in a HuggingFace model.

        Similar to walk_slo_linears() but for HuggingFace models.
        Returns dict of {name: nn.Linear_module}.

        Usage:
            from domains.infrastructure.model_loader import ModelLoader
            loader = ModelLoader()
            layers = loader.walk_hf_linears(hf_model)
            for name, module in layers.items():
                print(f"{name}: {module.weight.shape}")
        """
        layers = {}
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Linear":
                layers[name] = module
        return layers


# Singleton for convenience
_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the global ModelLoader instance."""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader


def load_model(
    model_id: str,
    device: str = "auto",
    quantize: bool = False,
    quant_bits: int = 8,
    quant_mode: str = "symmetric",
    verify: bool = True,
) -> LoadResult:
    """Convenience function to load a model.

    Args:
        model_id: Model identifier (e.g., "gpt2", "meta-llama/Llama-3-8B")
        device: Target device ("auto", "cpu", "mps", "cuda")
        quantize: Apply quantization after loading
        quant_bits: Bits for quantization (8 or 4)
        quant_mode: "symmetric" or "asymmetric"
        verify: Run test inference to verify model works

    Returns:
        LoadResult with provider, model, tokenizer, and metrics
    """
    return get_model_loader().load(
        model_id=model_id,
        device=device,
        quantize=quantize,
        quant_bits=quant_bits,
        quant_mode=quant_mode,
        verify=verify,
    )
