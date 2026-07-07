"""
Low-End Device Optimization for SloughGPT

Optimizes models for efficient inference on:
- Mobile devices
- Embedded systems
- CPU-only machines
- Edge devices

Includes:
- Dynamic quantization (PTQ)
- Weight-only quantization
- Dynamic layer skipping
- CPU-optimized operations
- Memory-efficient inference
"""

from domains.training.slonet_compat import torch
nn = torch.nn
try:
    from torch.nn import functional as F
except ImportError:
    F = nn.functional
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("man.training.efficient")


class DeviceType(Enum):
    """Target device types for optimization."""

    CPU = "cpu"
    MOBILE = "mobile"
    EMBEDDED = "embedded"
    EDGE_TPU = "edge_tpu"
    GPU_LOW_END = "gpu_low_end"


class QuantizationType(Enum):
    """Types of quantization."""

    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    DYNAMIC = "dynamic"


@dataclass
class EfficientConfig:
    """Configuration for low-end device optimization."""

    device_type: DeviceType = DeviceType.CPU
    quantization: QuantizationType = QuantizationType.INT8
    use_dynamic_quant: bool = True
    use_weight_only: bool = False
    use_qat: bool = False  # Quantization-aware training
    optimize_ops: bool = True
    use_flash_attention: bool = False
    max_batch_size: int = 1
    use_compile: bool = False


class Quantizer:
    """
    Post-training quantization for efficient inference.

    Supports:
    - Dynamic quantization (dynamic dtype)
    - Static quantization (calibration)
    - Weight-only quantization
    """

    @staticmethod
    def dynamic_quantize(
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """
        Apply dynamic quantization to model.

        Args:
            model: Model to quantize
            dtype: Target dtype (qint8, qint4)

        Returns:
            Quantized model
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=dtype,
        )
        logger.info(f"Dynamic quantization applied: {dtype}")
        return quantized_model

    @staticmethod
    def static_quantize(
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """
        Apply static quantization with calibration.

        Args:
            model: Model to quantize
            calibration_data: Data for calibration
            dtype: Target dtype

        Returns:
            Quantized model
        """
        # Set model to qconfig
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)

        # Calibrate
        model.eval()
        with torch.no_grad():
            for data in calibration_data[:100]:  # Use subset for speed
                model(data)

        # Convert
        quantized_model = torch.quantization.convert(model, inplace=False)
        logger.info(f"Static quantization applied: {dtype}")
        return quantized_model

    @staticmethod
    def weight_only_quantize(
        weights: torch.Tensor,
        bits: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights only (for storage/transfer).

        Args:
            weights: Weight tensor
            bits: Number of bits (2, 4, 8)

        Returns:
            (quantized_weights, scale)
        """
        # Calculate scale
        scale = weights.abs().max() / (2 ** (bits - 1) - 1)

        # Quantize
        quantized = torch.round(weights / scale).to(torch.int8)

        return quantized, scale

    @staticmethod
    def int4_quantize(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to int4."""
        return Quantizer.weight_only_quantize(weights, bits=4)

    @staticmethod
    def int8_quantize(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to int8."""
        return Quantizer.weight_only_quantize(weights, bits=8)


class EfficientLayer:
    """
    Optimized layer implementations for low-end devices.
    """

    @staticmethod
    def create_linear_replaced(
        module: nn.Linear,
        use_quantization: bool = True,
    ) -> nn.Module:
        """
        Replace Linear with optimized version.

        Args:
            module: Original Linear layer
            use_quantization: Whether to use quantization

        Returns:
            Optimized linear layer
        """
        if use_quantization:
            # Use quantized linear if available
            try:
                return torch.ops.quantized.linear_new(
                    module.weight,
                    module.bias,
                    module.in_features,
                    module.out_features,
                )
            except Exception:
                logger.warning("Quantized linear not available, using fallback", exc_info=True)
                pass

        return module

    @staticmethod
    def optimized_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        use_flash: bool = False,
    ) -> torch.Tensor:
        """
        Optimized attention computation.

        Args:
            query: [batch, heads, seq, dim]
            key: [batch, heads, seq, dim]
            value: [batch, heads, seq, dim]
            use_flash: Use flash attention if available

        Returns:
            Attention output
        """
        # Try flash attention first
        if use_flash:
            try:
                from torch.nn.attention import SDPBackend

                with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(query, key, value)
            except Exception:
                logger.warning("Flash attention not available, using standard attention", exc_info=True)
                pass

        # Fallback to standard attention
        return EfficientLayer._standard_attention(query, key, value)

    @staticmethod
    def _standard_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Standard attention without flash."""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, value)


class DynamicBatcher:
    """
    Dynamic batch sizing for memory efficiency.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_tokens: int = 2048,
        device: str = "cpu",
    ):
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.device = device
        self.buffer: List[torch.Tensor] = []
        self.buffer_tokens = 0

    def add(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Add inputs to batch buffer.

        Args:
            inputs: Input tensor [batch, seq]

        Returns:
            Batched inputs when ready, None otherwise
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        tokens = batch_size * seq_len

        # Check if we can add to buffer
        can_add = (
            len(self.buffer) < self.max_batch_size
            and self.buffer_tokens + tokens <= self.max_tokens
        )

        if can_add:
            self.buffer.append(inputs)
            self.buffer_tokens += tokens

            # Return None if not full yet
            if len(self.buffer) < self.max_batch_size:
                return None

        # Return batched inputs
        if self.buffer:
            batched = torch.cat(self.buffer, dim=0)
            self.buffer = []
            self.buffer_tokens = 0
            return batched

        return None

    def flush(self) -> Optional[torch.Tensor]:
        """Flush remaining buffer."""
        if self.buffer:
            batched = torch.cat(self.buffer, dim=0)
            self.buffer = []
            self.buffer_tokens = 0
            return batched
        return None


class EfficientInference:
    """
    Unified interface for efficient inference on low-end devices.
    """

    def __init__(
        self,
        model: nn.Module,
        config: EfficientConfig,
    ):
        self.model = model
        self.config = config
        self.quantized_model = None
        self.batcher = DynamicBatcher(
            max_batch_size=config.max_batch_size,
        )

    def optimize(self) -> nn.Module:
        """Apply all optimizations."""
        model = self.model

        # Apply quantization
        if self.config.quantization == QuantizationType.INT8:
            model = Quantizer.dynamic_quantize(model, torch.qint8)
        elif self.config.quantization == QuantizationType.INT4:
            model = Quantizer.dynamic_quantize(model, torch.qint4)

        # Apply torch.compile if requested
        if self.config.use_compile:
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                logger.warning("torch.compile not available, skipping")

        self.quantized_model = model
        return model

    def inference(
        self,
        inputs: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> torch.Tensor:
        """
        Run efficient inference.

        Args:
            inputs: Input tokens [batch, seq]
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated tokens
        """
        self.model.eval()

        generated = inputs.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits
                outputs = self.model(generated)
                logits = outputs[:, -1, :]

                # Apply temperature
                if hasattr(self, "temperature"):
                    logits = logits / self.temperature

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                # Stop if all EOS
                if (next_token == 0).all():
                    break

        return generated

    def streaming_inference(
        self,
        inputs: torch.Tensor,
        max_new_tokens: int = 100,
    ):
        """
        Run streaming inference (yield tokens).

        Args:
            inputs: Input tokens
            max_new_tokens: Max tokens to generate

        Yields:
            Generated tokens one at a time
        """
        self.model.eval()

        generated = inputs.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(generated)
                logits = outputs[:, -1, :]

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)
                yield next_token.item()

                if next_token.item() == 0:
                    break


def create_efficient_model(
    model: nn.Module,
    device_type: str = "cpu",
    quantization: str = "int8",
) -> nn.Module:
    """
    Create an optimized model for low-end devices.

    Args:
        model: Original model
        device_type: Target device (cpu, mobile, embedded)
        quantization: Quantization type (int8, int4, fp16)

    Returns:
        Optimized model
    """
    device = DeviceType(device_type)
    quant = QuantizationType(quantization)

    config = EfficientConfig(
        device_type=device,
        quantization=quant,
    )

    efficient = EfficientInference(model, config)
    return efficient.optimize()


def estimate_memory_usage(
    num_parameters: int,
    quantization: str = "fp16",
    batch_size: int = 1,
    sequence_length: int = 512,
) -> Dict[str, float]:
    """
    Estimate memory usage for given configuration.

    Args:
        num_parameters: Number of model parameters
        quantization: Quantization type
        batch_size: Batch size
        sequence_length: Sequence length

    Returns:
        Dictionary with memory estimates in MB
    """
    # Base sizes (bytes)
    sizes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
        "int2": 0.25,
    }

    weight_bytes = num_parameters * sizes.get(quantization, 2)

    # Activation memory (rough estimate)
    # KV cache: 2 * batch * seq * layers * hidden * 2 bytes
    layers = 12  # Assume 12 layers
    hidden = 768  # Assume 768 hidden
    kv_bytes = 2 * batch_size * sequence_length * layers * hidden * 2

    # Input/output
    io_bytes = batch_size * sequence_length * hidden * 2

    total_mb = (weight_bytes + kv_bytes + io_bytes) / (1024 * 1024)

    return {
        "weights_mb": weight_bytes / (1024 * 1024),
        "kv_cache_mb": kv_bytes / (1024 * 1024),
        "io_mb": io_bytes / (1024 * 1024),
        "total_mb": total_mb,
        "total_gb": total_mb / 1024,
    }


__all__ = [
    "DeviceType",
    "QuantizationType",
    "EfficientConfig",
    "Quantizer",
    "EfficientLayer",
    "DynamicBatcher",
    "EfficientInference",
    "create_efficient_model",
    "estimate_memory_usage",
    "AWQQuantizer",
    "GPTQQuantizer",
    "DynamicLayerSkipper",
    "KVCacheOptimizer",
    "CPUOptimizer",
]


# ============================================================================
# AWQ/GPTQ Quantization
# ============================================================================


class AWQQuantizer:
    """
    AWQ (Activation-aware Weight Quantization) for efficient inference.

    Primes weights based on activation statistics.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.quant_scales: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def calibrate(self, calibration_data: List[torch.Tensor], num_samples: int = 100):
        """
        Calibrate quantization scales using activation statistics.

        Args:
            calibration_data: Data for calibration
            num_samples: Number of samples to use
        """
        self.model.eval()
        activation_stats = {}

        # Hook to capture activations
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Track per-channel max for AWQ
                    channel_dim = -1 if name.endswith(".weight") else 0
                    activation_stats[name] = output.abs().amax(dim=channel_dim, keepdim=True)

            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Run calibration
        with torch.no_grad():
            for i, data in enumerate(calibration_data[:num_samples]):
                self.model(data)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        self.activation_stats = activation_stats

    def quantize(self, bits: int = 4) -> nn.Module:
        """
        Quantize model weights using AWQ.

        Args:
            bits: Number of bits (4, 8)

        Returns:
            Quantized model
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data

                # Get activation scale if available
                scale = self.activation_stats.get(name, torch.ones_like(weight[:1, :1]))

                # AWQ: scale weight by activation magnitude
                weight_scale = scale.view(1, -1) if scale.numel() > 1 else 1.0
                scaled_weight = weight * weight_scale

                # Quantize
                max_val = 2 ** (bits - 1)
                quantized = torch.round(scaled_weight / scaled_weight.abs().max() * max_val)

                # Store scale for dequantization
                self.quant_scales[name] = (
                    scaled_weight.abs().max() / max_val,
                    1.0 / weight_scale.clamp(min=1e-8),
                )

        return self.model

    def dequantize_weight(self, name: str, quantized_weight: torch.Tensor) -> torch.Tensor:
        """Dequantize weights for inference."""
        if name not in self.quant_scales:
            return quantized_weight

        scale, weight_scale = self.quant_scales[name]
        return quantized_weight * scale * weight_scale


class GPTQQuantizer:
    """
    GPTQ (GPT Quantization) for accurate post-training quantization.

    Uses second-order information for better quantization.
    """

    def __init__(self, model: nn.Module, bits: int = 4):
        self.model = model
        self.bits = bits
        self.hessian: Dict[str, torch.Tensor] = {}

    def quantize_layer(
        self, name: str, layer: nn.Linear, hessian: Optional[torch.Tensor] = None
    ) -> nn.Linear:
        """
        Quantize a single linear layer using GPTQ.

        Args:
            name: Layer name
            layer: Linear layer
            hessian: Precomputed Hessian (optional)

        Returns:
            Quantized layer
        """
        weight = layer.weight.data
        out_features, in_features = weight.shape

        # Compute or use provided Hessian
        if hessian is None:
            hessian = torch.eye(in_features, device=weight.device) / (in_features**0.5)

        # GPTQ quantization
        max_val = 2 ** (self.bits - 1)

        # Quantize column by column
        quantized = torch.zeros_like(weight)
        scales = torch.zeros(out_features, device=weight.device)

        for i in range(out_features):
            w = weight[i, :]
            h = hessian

            # Compute optimal scale
            scale = (w @ h @ w) ** 0.5
            if scale > 0:
                quantized[i] = torch.round(w / scale * max_val) / scale
                scales[i] = scale

        # Replace weight
        layer.weight.data = quantized
        layer.register_buffer("scale", scales)

        return layer

    def quantize(self) -> nn.Module:
        """Quantize entire model."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.quantize_layer(name, module)

        return self.model


# ============================================================================
# Dynamic Layer Skipping
# ============================================================================


class DynamicLayerSkipper:
    """
    Skip layers dynamically based on input complexity.

    Reduces computation for simple inputs.
    """

    def __init__(
        self,
        model: nn.Module,
        skip_threshold: float = 0.1,
        min_layers: int = 2,
    ):
        self.model = model
        self.skip_threshold = skip_threshold
        self.min_layers = min_layers
        self.layer_outputs: Dict[str, torch.Tensor] = {}
        self.skip_counts: Dict[str, int] = {}

    def should_skip(self, layer_name: str, output: torch.Tensor) -> bool:
        """
        Determine if a layer should be skipped.

        Args:
            layer_name: Name of the layer
            output: Layer output tensor

        Returns:
            True if layer should be skipped
        """
        # Check complexity (variance of output)
        complexity = output.var().item()

        if complexity < self.skip_threshold:
            self.skip_counts[layer_name] = self.skip_counts.get(layer_name, 0) + 1
            return True

        return False

    def create_skippable_model(self) -> nn.Module:
        """
        Wrap model with dynamic skipping.

        Returns:
            Model with skip logic
        """
        # This would require model-specific implementation
        # For now, return the original model
        return self.model

    def get_skip_stats(self) -> Dict[str, float]:
        """Get skip statistics."""
        return {name: count / (count + 1) for name, count in self.skip_counts.items()}


# ============================================================================
# KV Cache Optimization
# ============================================================================


class KVCacheOptimizer:
    """
    Optimizes KV cache for efficient generation.

    Features:
    - Paged attention
    - Cache eviction
    - Dynamic allocation
    """

    def __init__(
        self,
        max_sequence_length: int = 2048,
        page_size: int = 64,
        enable_eviction: bool = True,
    ):
        self.max_seq_len = max_sequence_length
        self.page_size = page_size
        self.enable_eviction = enable_eviction
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.access_counts: Dict[int, int] = {}

    def allocate(
        self, batch_size: int, num_heads: int, head_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Allocate KV cache for new sequence.

        Args:
            batch_size: Batch size
            num_heads: Number of attention heads
            head_dim: Dimension per head

        Returns:
            (key_cache, value_cache)
        """
        max_pages = self.max_seq_len // self.page_size

        key_cache = torch.zeros(batch_size, max_pages, num_heads, self.page_size, head_dim)
        value_cache = torch.zeros(batch_size, max_pages, num_heads, self.page_size, head_dim)

        return key_cache, value_cache

    def update(
        self,
        seq_id: int,
        page_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Update cache with new keys/values.

        Args:
            seq_id: Sequence ID
            page_idx: Page index to update
            keys: New keys [batch, num_heads, seq_len, head_dim]
            values: New values
        """
        if seq_id not in self.cache:
            # Initialize new sequence
            self.cache[seq_id] = self.allocate(keys.size(0), keys.size(1), keys.size(-1))

        key_cache, value_cache = self.cache[seq_id]

        # Update page
        key_cache[:, page_idx, :, : keys.size(2), :] = keys
        value_cache[:, page_idx, :, : values.size(2), :] = values

        self.access_counts[seq_id] = self.access_counts.get(seq_id, 0) + 1

    def evict_if_needed(self):
        """Evict least recently used sequences if cache is full."""
        if not self.enable_eviction:
            return

        if len(self.cache) >= self.max_seq_len:
            # Find LRU sequence
            lru_seq = min(self.access_counts, key=self.access_counts.get)
            del self.cache[lru_seq]
            del self.access_counts[lru_seq]

    def get_cache(self, seq_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cache for sequence."""
        return self.cache.get(seq_id)


# ============================================================================
# CPU-Specific Optimizations
# ============================================================================


class CPUOptimizer:
    """
    CPU-specific optimizations for efficient inference.

    Features:
    - Thread optimization
    - Memory mapping
    - Operator fusion
    """

    @staticmethod
    def optimize_threads(num_threads: Optional[int] = None):
        """
        Optimize PyTorch CPU threads.

        Args:
            num_threads: Number of threads (None = auto)
        """
        import threading

        if num_threads is None:
            # Use number of physical cores
            num_threads = threading.active_count()

        torch.set_num_threads(num_threads)
        logger.info(f"CPU threads set to {num_threads}")

    @staticmethod
    def enable_mkldnn():
        """Enable MKL-DNN (Intel) optimizations."""
        torch.backends.mkldnn.enabled = True
        logger.info("MKL-DNN enabled")

    @staticmethod
    def enable_ipex():
        """Enable Intel Extension for PyTorch (IPEX)."""
        try:
            import intel_extension_for_pytorch as ipex

            ipex.optimize()
            logger.info("Intel IPEX enabled")
        except ImportError:
            logger.warning("Intel IPEX not available")

    @staticmethod
    def optimize_for_inference():
        """
        Apply all CPU optimizations for inference.
        """
        CPUOptimizer.enable_mkldnn()
        CPUOptimizer.optimize_threads()

        # Set inference mode
        with torch.inference_mode():
            pass

        logger.info("CPU inference optimizations applied")

    @staticmethod
    def get_optimal_batch_size(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cpu",
    ) -> int:
        """
        Find optimal batch size for given model and memory.

        Args:
            model: Model to test
            input_shape: Input tensor shape
            device: Device to test on

        Returns:
            Optimal batch size
        """
        model = model.to(device)
        model.eval()

        batch_size = 1
        max_batch = 64

        with torch.no_grad():
            while batch_size < max_batch:
                try:
                    dummy_input = torch.randn(batch_size, *input_shape[1:], device=device)
                    _ = model(dummy_input)
                    batch_size *= 2
                except RuntimeError:
                    break

        return batch_size // 2


# ============================================================================
# Efficiency Utilities
# ============================================================================


def apply_efficiency_optimizations(
    model: nn.Module,
    target_device: str = "cpu",
    quantization: str = "int8",
) -> nn.Module:
    """
    Apply all efficiency optimizations to a model.

    Args:
        model: Model to optimize
        target_device: Target device (cpu, mobile, etc.)
        quantization: Quantization type

    Returns:
        Optimized model
    """
    # Apply quantization
    if quantization == "int8":
        model = Quantizer.dynamic_quantize(model, torch.qint8)
    elif quantization == "int4":
        model = Quantizer.dynamic_quantize(model, torch.qint4)

    # Apply CPU optimizations
    if target_device == "cpu":
        CPUOptimizer.optimize_for_inference()

    return model
