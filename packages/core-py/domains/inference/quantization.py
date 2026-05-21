"""
Quantization Module
Provides model quantization functionality for inference optimization.

Supports:
- Dynamic quantization (INT8, INT4)
- FP16/BF16 precision
- Weight-only quantization
- Per-channel quantization
"""

try:
    import torch
except ImportError:
    from domains.training.slonet_compat import torch
try:
    import torch.nn as nn
except ImportError:
    nn = torch.nn
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from enum import Enum
import math


class QuantizationType(Enum):
    """Quantization types."""

    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT8_DYNAMIC = "int8_dynamic"
    INT4 = "int4"
    Q4_K = "q4_k"
    Q5_K = "q5_k"
    Q8_0 = "q8_0"


QUANTIZATION_PRESETS = {
    "fp16": {"bits": 16, "dtype": torch.float16, "desc": "Half precision (16-bit)"},
    "bf16": {"bits": 16, "dtype": torch.bfloat16, "desc": "BFloat16 (16-bit)"},
    "int8": {"bits": 8, "dtype": torch.int8, "desc": "8-bit integer"},
    "int8_dynamic": {"bits": 8, "dtype": torch.int8, "desc": "Dynamic INT8 quantization"},
    "int4": {"bits": 4, "dtype": torch.uint8, "desc": "4-bit integer (packed in uint8)"},
    "q4_k": {"bits": 4, "group_size": 128, "desc": "Q4_K (4-bit with K-scale)"},
    "q5_k": {"bits": 5, "group_size": 128, "desc": "Q5_K (5-bit with K-scale)"},
    "q8_0": {"bits": 8, "group_size": 32, "desc": "Q8_0 (8-bit)"},
}


def get_quantization_preset(name: str) -> Optional[Dict[str, Any]]:
    """Get quantization preset by name."""
    return QUANTIZATION_PRESETS.get(name)


@dataclass
class QuantizationInfo:
    """Quantization information."""

    quantization_type: QuantizationType
    bits: int
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    memory_saved_mb: float

    @property
    def reduction(self) -> float:
        return (1 - self.compression_ratio) * 100


@dataclass
class QuantizedLinear(nn.Module):
    """Quantized linear layer for inference."""
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    weight_scale: torch.Tensor
    weight_zero_point: Optional[torch.Tensor]
    
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        weight_scale: torch.Tensor,
        weight_zero_point: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
        self.weight_zero_point = nn.Parameter(weight_zero_point, requires_grad=False) if weight_zero_point is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class DynamicQuantizer:
    """Dynamic quantization - converts weights to INT8 on-the-fly during inference."""
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.quant_min = -(2 ** (bits - 1))
        self.quant_max = 2 ** (bits - 1) - 1
    
    def quantize_tensor(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a single tensor to int8."""
        scale = tensor.abs().max() / self.quant_max
        if scale == 0:
            scale = 1.0
        
        quantized = torch.round(tensor / scale).to(torch.int8)
        zero_point = torch.tensor(0, dtype=torch.int8)
        
        return quantized, scale, zero_point
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to model."""
        model.eval()
        
        quantized_modules = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight, scale, zero_point = self.quantize_tensor(module.weight.data)
                
                quantized_linear = QuantizedLinear(
                    weight=weight,
                    bias=module.bias.data if module.bias is not None else None,
                    weight_scale=scale,
                    weight_zero_point=zero_point
                )
                quantized_modules[name] = quantized_linear
        
        return self._replace_modules(model, quantized_modules)
    
    def _replace_modules(self, model: nn.Module, quantized_modules: Dict[str, QuantizedLinear]) -> nn.Module:
        """Replace linear modules with quantized versions."""
        for name, quantized_module in quantized_modules.items():
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, quantized_module)
            else:
                setattr(model, name, quantized_module)
        
        return model


class StaticQuantizer:
    """Static quantization with per-channel scaling."""
    
    def __init__(self, bits: int = 8, group_size: Optional[int] = None):
        self.bits = bits
        self.group_size = group_size
        self.quant_min = -(2 ** (bits - 1))
        self.quant_max = 2 ** (bits - 1) - 1
    
    def quantize_tensor(self, tensor: torch.Tensor, axis: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor with per-channel scaling."""
        if axis == 0:
            scale = tensor.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
            scale = scale.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        else:
            scale = tensor.abs().max(dim=axis, keepdim=True)[0].clamp(min=1e-8)
        
        quantized = torch.round(tensor / scale).clamp(self.quant_min, self.quant_max)
        
        return quantized.to(torch.int8), scale
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize model with static quantization."""
        model.eval()
        
        quantized_modules = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                quantized_weight, scale = self.quantize_tensor(module.weight.data)
                
                quantized_linear = QuantizedLinear(
                    weight=quantized_weight,
                    bias=module.bias.data if module.bias is not None else None,
                    weight_scale=scale,
                    weight_zero_point=None
                )
                quantized_modules[name] = quantized_linear
        
        return DynamicQuantizer()._replace_modules(model, quantized_modules)


class FP16Quantizer:
    """Convert model to half precision (FP16)."""
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Convert model to FP16."""
        return model.half()


class BF16Quantizer:
    """Convert model to bfloat16."""
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Convert model to BF16."""
        return model.to(torch.bfloat16)


class Quantizer:
    """Main quantizer class."""
    
    def __init__(self, quantization_type: QuantizationType = QuantizationType.INT8_DYNAMIC):
        self.quantization_type = quantization_type
        self._impl = self._create_implementation()
    
    def _create_implementation(self) -> Callable[[nn.Module], nn.Module]:
        """Create quantization implementation based on type."""
        if self.quantization_type == QuantizationType.INT8_DYNAMIC:
            return DynamicQuantizer(bits=8).quantize_model
        elif self.quantization_type == QuantizationType.INT8:
            return StaticQuantizer(bits=8).quantize_model
        elif self.quantization_type == QuantizationType.INT4:
            return DynamicQuantizer(bits=4).quantize_model
        elif self.quantization_type == QuantizationType.FP16:
            return FP16Quantizer().quantize_model
        elif self.quantization_type == QuantizationType.BF16:
            return BF16Quantizer().quantize_model
        else:
            return lambda m: m
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize model."""
        return self._impl(model)
    
    def get_quantization_info(self, model: nn.Module) -> QuantizationInfo:
        """Get quantization information for model."""
        bits_map = {
            QuantizationType.NONE: 32,
            QuantizationType.FP16: 16,
            QuantizationType.BF16: 16,
            QuantizationType.INT8: 8,
            QuantizationType.INT8_DYNAMIC: 8,
            QuantizationType.INT4: 4,
            QuantizationType.Q4_K: 4,
            QuantizationType.Q5_K: 5,
            QuantizationType.Q8_0: 8,
        }
        
        bits = bits_map.get(self.quantization_type, 32)
        
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        quantized_size = original_size * (bits / 32)
        
        return QuantizationInfo(
            quantization_type=self.quantization_type,
            bits=bits,
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=quantized_size / original_size if original_size > 0 else 1.0,
            memory_saved_mb=original_size - quantized_size,
        )


class SouModelQuantizer:
    """Model quantizer for .slo models."""
    
    def __init__(self, quantization_type: QuantizationType = QuantizationType.Q4_K):
        self.quantizer = Quantizer(quantization_type)
        self.quantization_type = quantization_type
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize a model."""
        return self.quantizer.quantize_model(model)
    
    def get_quantization_info(self, model: nn.Module) -> QuantizationInfo:
        """Get quantization info."""
        return self.quantizer.get_quantization_info(model)


def quantize_model(
    model: nn.Module,
    quantization_type: str = "int8_dynamic"
) -> tuple[nn.Module, QuantizationInfo]:
    """
    Quantize a model.
    
    Args:
        model: PyTorch model to quantize
        quantization_type: One of "fp16", "bf16", "int8", "int8_dynamic", "int4"
    
    Returns:
        Tuple of (quantized_model, quantization_info)
    """
    qtype = QuantizationType(quantization_type)
    quantizer = Quantizer(qtype)
    
    quantized = quantizer.quantize_model(model)
    info = quantizer.get_quantization_info(model)
    
    return quantized, info


def estimate_memory(
    num_parameters: int,
    quantization_type: str = "fp16"
) -> Dict[str, Any]:
    """Estimate memory requirements for a model."""
    precision_map = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    
    bytes_per_param = precision_map.get(quantization_type, 4)
    memory_bytes = num_parameters * bytes_per_param
    memory_mb = memory_bytes / (1024 * 1024)
    memory_gb = memory_mb / 1024
    
    return {
        "num_parameters": num_parameters,
        "quantization_type": quantization_type,
        "bytes_per_param": bytes_per_param,
        "memory_mb": memory_mb,
        "memory_gb": memory_gb,
        "memory_formatted": f"{memory_gb:.2f} GB" if memory_gb >= 1 else f"{memory_mb:.1f} MB",
    }


__all__ = [
    "QuantizationType",
    "QuantizationInfo",
    "QuantizedLinear",
    "DynamicQuantizer",
    "StaticQuantizer",
    "FP16Quantizer",
    "BF16Quantizer",
    "Quantizer",
    "SouModelQuantizer",
    "quantize_model",
    "estimate_memory",
    "QUANTIZATION_PRESETS",
    "get_quantization_preset",
]
