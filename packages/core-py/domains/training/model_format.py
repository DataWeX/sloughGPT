"""Model format manager for SloughGPT - unified handling of standard formats."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum


class ModelFormat(str, Enum):
    """Supported model formats."""

    SAFETENSORS = "safetensors"
    SAFETENSORS_BF16 = "safetensors_bf16"
    GGUF_Q4_0 = "gguf_q4_0"
    GGUF_Q4_1 = "gguf_q4_1"
    GGUF_Q5_0 = "gguf_q5_0"
    GGUF_Q5_1 = "gguf_q5_1"
    GGUF_Q8_0 = "gguf_q8_0"
    GGUF_F16 = "gguf_f16"
    GGUF_F32 = "gguf_f32"
    ONNX = "onnx"
    TORCH = "torch"
    PYTORCH = "pytorch"
    GGUF = "gguf"
    AUTO = "auto"


@dataclass
class FormatInfo:
    """Information about a model format."""

    extension: str
    description: str
    supports_quantization: bool
    security: str
    loading_speed: str
    requires_extra_deps: bool


FORMAT_INFO: Dict[str, FormatInfo] = {
    "safetensors": FormatInfo(
        extension=".safetensors",
        description="HuggingFace SafeTensors format - recommended default",
        supports_quantization=False,
        security="secure (no pickle)",
        loading_speed="fast",
        requires_extra_deps=True,
    ),
    "safetensors_bf16": FormatInfo(
        extension=".safetensors",
        description="SafeTensors with BF16 precision",
        supports_quantization=False,
        security="secure (no pickle)",
        loading_speed="fast",
        requires_extra_deps=True,
    ),
    "gguf_q4_0": FormatInfo(
        extension="-Q4_0.gguf",
        description="GGUF 4-bit quantization (smallest, fastest)",
        supports_quantization=True,
        security="secure",
        loading_speed="fastest",
        requires_extra_deps=True,
    ),
    "gguf_q4_1": FormatInfo(
        extension="-Q4_1.gguf",
        description="GGUF 4-bit quantization with improved quality",
        supports_quantization=True,
        security="secure",
        loading_speed="fastest",
        requires_extra_deps=True,
    ),
    "gguf_q5_0": FormatInfo(
        extension="-Q5_0.gguf",
        description="GGUF 5-bit quantization",
        supports_quantization=True,
        security="secure",
        loading_speed="very fast",
        requires_extra_deps=True,
    ),
    "gguf_q5_1": FormatInfo(
        extension="-Q5_1.gguf",
        description="GGUF 5-bit quantization with improved quality",
        supports_quantization=True,
        security="secure",
        loading_speed="very fast",
        requires_extra_deps=True,
    ),
    "gguf_q8_0": FormatInfo(
        extension="-Q8_0.gguf",
        description="GGUF 8-bit quantization (balanced)",
        supports_quantization=True,
        security="secure",
        loading_speed="fast",
        requires_extra_deps=True,
    ),
    "gguf_f16": FormatInfo(
        extension="-F16.gguf",
        description="GGUF 16-bit float (half precision)",
        supports_quantization=True,
        security="secure",
        loading_speed="medium",
        requires_extra_deps=True,
    ),
    "gguf_f32": FormatInfo(
        extension="-F32.gguf",
        description="GGUF 32-bit float (full precision)",
        supports_quantization=True,
        security="secure",
        loading_speed="slow",
        requires_extra_deps=True,
    ),
    "onnx": FormatInfo(
        extension=".onnx",
        description="ONNX format for cross-platform inference",
        supports_quantization=False,
        security="secure",
        loading_speed="medium",
        requires_extra_deps=True,
    ),
    "torch": FormatInfo(
        extension=".pt",
        description="PyTorch checkpoint (uses pickle - legacy)",
        supports_quantization=False,
        security="warning: contains pickle",
        loading_speed="medium",
        requires_extra_deps=False,
    ),
    "pytorch": FormatInfo(
        extension=".pt",
        description="PyTorch checkpoint (legacy format)",
        supports_quantization=False,
        security="warning: contains pickle",
        loading_speed="medium",
        requires_extra_deps=False,
    ),
}


@dataclass
class ModelMetadata:
    """Metadata stored with the model."""

    format: str
    vocab_size: int
    n_embed: int
    n_layer: int
    n_head: int
    block_size: int
    model_type: str = "nanogpt"
    training_config: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


def save_with_safetensors(
    model,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    dtype: Optional[str] = None,
) -> str:
    """Save model in SafeTensors format with metadata."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError(
            "safetensors not installed. Run: pip install safetensors"
        )

    state_dict = model.state_dict()

    if dtype == "bf16":
        state_dict = {k: v.to(dtype="bfloat16") for k, v in state_dict.items()}

    metadata_dict = metadata or {}
    metadata_dict["format_version"] = "1.0"
    metadata_dict["format"] = "safetensors"

    save_file(state_dict, output_path, metadata=metadata_dict)

    meta_path = output_path.replace(".safetensors", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    return output_path


def save_with_metadata(
    model,
    output_path: str,
    format: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save model in specified format with standardized metadata."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    meta = ModelMetadata(
        format=format,
        vocab_size=getattr(model, "vocab_size", 0),
        n_embed=getattr(model, "n_embed", 0),
        n_layer=getattr(model, "n_layer", 0),
        n_head=getattr(model, "n_head", 0),
        block_size=getattr(model, "block_size", 0),
        model_type=getattr(model, "model_type", "nanogpt"),
        extra=metadata,
    )

    if format in ("safetensors", "safetensors_bf16"):
        dtype = "bf16" if format == "safetensors_bf16" else None
        return save_with_safetensors(
            model, output_path, metadata=asdict(meta), dtype=dtype
        )

    elif format == "torch":
        import torch

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": asdict(meta),
        }
        torch.save(checkpoint, output_path)
        return output_path

    else:
        raise ValueError(f"Unsupported format: {format}")


def load_model_weights(
    path: str,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load model weights from any supported format.

    Returns (state_dict, metadata).
    """
    path = Path(path)
    stem = path.stem
    suffix = path.suffix.lower()

    if suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("pip install safetensors")

        state_dict = {}
        metadata = None
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict, metadata

    elif suffix == ".gguf":
        return _load_gguf(path)

    elif suffix == ".onnx":
        raise ValueError("ONNX models cannot be loaded as state_dicts. Use onnxruntime.")

    elif suffix in (".pt", ".pth", ".pkl"):
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        metadata = checkpoint.get("metadata") or checkpoint.get("config")
        return state_dict, metadata

    else:
        raise ValueError(f"Unknown model format: {suffix}")


def _load_gguf(path: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load model from GGUF file."""
    try:
        from gguf import GGUFReader
    except ImportError:
        raise ImportError(
            "gguf not installed. Run: pip install gguf"
        )

    reader = GGUFReader(str(path), "r")

    state_dict = {}
    for key in reader.tensors:
        tensor = reader.tensors[key]
        state_dict[key] = tensor.data

    metadata = {k: v for k, v in reader.metadata.items()}
    return state_dict, metadata


def get_output_path(base: str, format: str) -> str:
    """Get output path with correct extension for format."""
    if format == "safetensors":
        return base.replace(".pt", ".safetensors")
    elif format == "safetensors_bf16":
        return base.replace(".pt", "-bf16.safetensors")
    elif format.startswith("gguf"):
        quant = format.replace("gguf_", "").upper()
        base_stem = Path(base).stem
        return f"{Path(base).parent}/{base_stem}-{quant}.gguf"
    elif format == "onnx":
        return base.replace(".pt", ".onnx")
    elif format in ("torch", "pytorch"):
        return base.replace(".safetensors", ".pt")
    return base


def list_supported_formats() -> Dict[str, FormatInfo]:
    """List all supported model formats."""
    return FORMAT_INFO


def get_recommended_format(quantized: bool = False) -> str:
    """Get recommended format for training."""
    if quantized:
        return "gguf_q4_0"
    return "safetensors"


def convert_format(
    input_path: str,
    output_format: str,
    output_path: Optional[str] = None,
) -> str:
    """Convert model from one format to another."""
    state_dict, metadata = load_model_weights(input_path)

    if output_path is None:
        output_path = get_output_path(input_path, output_format)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if output_format in ("safetensors", "safetensors_bf16"):
        dtype = "bf16" if output_format == "safetensors_bf16" else None
        try:
            import torch

            model = _state_dict_to_dummy_model(state_dict)
            return save_with_safetensors(
                model, output_path, metadata=metadata, dtype=dtype
            )
        except ImportError:
            try:
                from safetensors.torch import save_file

                state = state_dict
                if dtype == "bf16":
                    state = {k: v.to(dtype="bfloat16") for k, v in state.items()}
                save_file(state, output_path)
                return output_path
            except ImportError:
                raise ImportError("pip install safetensors torch")

    elif output_format in ("torch", "pytorch"):
        import torch

        checkpoint = {"model_state_dict": state_dict, "metadata": metadata}
        torch.save(checkpoint, output_path)
        return output_path

    else:
        raise ValueError(f"Conversion to {output_format} not yet supported")


def _state_dict_to_dummy_model(state_dict: Dict[str, Any]) -> Any:
    """Create a dummy model wrapper for saving state_dict."""

    class DummyModel:
        def __init__(self, sd):
            self._sd = sd
            for k, v in sd.items():
                setattr(self, k, v)

        def state_dict(self):
            return self._sd

    return DummyModel(state_dict)


__all__ = [
    "ModelFormat",
    "FormatInfo",
    "ModelMetadata",
    "FORMAT_INFO",
    "save_with_safetensors",
    "save_with_metadata",
    "load_model_weights",
    "get_output_path",
    "list_supported_formats",
    "get_recommended_format",
    "convert_format",
]
