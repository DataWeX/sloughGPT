"""
SloNet Model Loader — torch-free model loading via .slnc format.

Provides a `ModelLoader` class that handles:
  - Model detection (.slnc file, auto-converts from safetensors)
  - Verification (forward-pass smoke test on load)
  - Quantization (via walk_slo_linears)

Usage:
    from domains.infrastructure.model_loader import ModelLoader

    loader = ModelLoader()
    result = loader.load("gpt2")
    if result.success:
        print(f"Loaded {result.model_type} with {result.provider}")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("slo.infrastructure.model_loader")


@dataclass
class LoadResult:
    """Standardized result from model loading."""
    success: bool
    model_id: str
    model_type: str  # "slonet"
    provider: Any = None
    model: Any = None
    tokenizer: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class ModelLoader:
    """SloNet model loader — detects .slnc format and routes accordingly.

    Falls back to auto-conversion from safetensors when no .slnc file exists.
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
        """Load a model by ID via SloNet (.slnc format).

        Args:
            model_id: Model identifier (e.g., "gpt2")
            device: Target device (kept for API compat, not used by SloNet)
            quantize: Apply quantization after loading
            quant_bits: Bits for quantization (8 or 4)
            quant_mode: "symmetric" or "asymmetric"
            verify: Run test inference to verify model works

        Returns:
            LoadResult with provider, model, tokenizer, and metrics
        """
        result = self._try_load_slnc(model_id, device, quantize, quant_bits, quant_mode)
        if result is not None:
            if verify and result.success:
                self._verify_model(result)
            return result

        return LoadResult(
            success=False,
            model_id=model_id,
            model_type="slonet",
            error=f"No .slnc file found for {model_id}",
        )

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

            config = load_model_config(model_id)

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
                        weights[key] = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])

            from domains.infrastructure.slnc.compiler import SLNCCompiler
            compiler = SLNCCompiler()
            compiler.compile_from_dict(config, weights, str(slnc_path))

            logger.info("Converted to .slnc: %s", slnc_path, extra={"tag": "MODEL"})
            return slnc_path
        except Exception as e:
            logger.warning("SloNet conversion failed: %s", e, extra={"tag": "MODEL"})
            return None

    def _verify_model(self, result: LoadResult) -> bool:
        """Run lightweight verification to confirm model loads correctly.

        Single forward pass (~1-2s) instead of full generation (~12s).
        """
        try:
            test_ids = np.array([[1, 2, 3]], dtype=np.int64)
            output = result.model.forward(test_ids)

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
    """Convenience function to load a model via SloNet.

    Args:
        model_id: Model identifier (e.g., "gpt2")
        device: Target device (kept for API compat, not used by SloNet)
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
