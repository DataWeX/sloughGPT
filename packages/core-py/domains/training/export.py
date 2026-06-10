"""Model export utilities for SloughGPT.

This module provides comprehensive model export functionality following industry
best practices for deploying trained PyTorch models to various formats.

Char-LM **perplexity parity** with training uses native ``SloughGPTTrainer`` ``step_*.pt``
(``stoi`` / ``itos`` / ``chars``); exported formats encode vocabulary differently — see
``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).

Export Formats
==============

SafeTensors (Recommended Default)
---------------------------------
- Format: .safetensors
- Pros: Safe deserialization, memory-mapped loading, zero-copy
- Use case: Production model storage, HuggingFace integration
- Best for: General purpose, safe deployment

ONNX (Cross-Platform)
---------------------
- Format: .onnx
- Pros: Framework-agnostic, ONNX Runtime optimization, broad support
- Use case: Server deployment, web inference (via TF.js), mobile (via OnnxRuntime)
- Requirements: torch>=2.0, onnx>=1.15.0, onnxruntime>=1.17.0

GGUF (Mobile/Embedded)
---------------------
- Format: .gguf
- Pros: Quantization support, llama.cpp optimized, excellent mobile performance
- Use case: React Native (llama.rn), iOS/Android, embedded devices
- Requirements: gguf>=0.10.0
- Quantizations: Q4_K_M (recommended), Q5_K_M, Q8_0, F16, F32

TorchScript (PyTorch Native)
----------------------------
- Format: .torchscript.pt
- Pros: PyTorch-native, optimized C++ inference
- Use case: Server deployment with TorchServe, C++ applications
- Requirements: torch>=2.0

Examples
========

Basic export::

    from domains.training.export import export_model, ExportConfig

    config = ExportConfig(
        input_path="model.pt",
        output_path="exported_model",
        format="safetensors",
    )
    results = export_model(config, model=my_model)

Export for mobile (llama.rn)::

    config = ExportConfig(
        input_path="model.pt",
        output_path="model_mobile",
        format="gguf_q4_k_m",
    )
    results = export_model(config, model=my_model)

Export all formats::

    config = ExportConfig(
        input_path="model.pt",
        output_path="model_all",
        format="all",
    )
    results = export_model(config, model=my_model)

CLI Usage::

    python3 cli.py export model.pt -f safetensors
    python3 cli.py export model.pt -f onnx --seq-len 128
    python3 cli.py export model.pt -f gguf_q4_k_m
    python3 cli.py export model.pt -f all

Tags
====

Model export supports tagging for model registry and metadata:

- ``model_type``: sloughgpt, nanogpt, custom
- ``training_dataset``: Dataset used for training
- ``epochs_trained``: Number of training epochs
- ``final_train_loss``: Final training loss
- ``final_val_loss``: Final validation loss
- ``quantization``: Quantization method used (Q4_K_M, etc.)

See Also
========

- :class:`ExportConfig`: Configuration dataclass
- :func:`export_model`: Main export dispatcher
- :func:`list_export_formats`: List all supported formats
- :mod:`domains.training.onnx_export`: ONNX-specific export
- :mod:`domains.training.gguf_export`: GGUF-specific export

"""

import logging
import os
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch.nn as Module

from dataclasses import dataclass, field, asdict

logger = logging.getLogger("man.export")


@dataclass
class ModelMetadata:
    """Comprehensive metadata for model training core compatibility.

    This class captures all information needed by training core logic to:
    - Load and understand the model architecture
    - Continue training from checkpoint
    - Reproduce training results
    - Validate model compatibility

    Attributes:
        name: Model name/identifier
        model_type: Architecture type (sloughgpt, nanogpt, custom)
        version: Model version string

        # Architecture (required for loading)
        vocab_size: Vocabulary size
        n_embed: Embedding dimension
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_kv_head: Number of key-value heads (for GQA)
        block_size: Maximum sequence length
        max_seq_len: Maximum supported sequence length

        # Training configuration
        training_dataset: Path or name of training dataset
        validation_dataset: Path or name of validation dataset
        epochs_trained: Number of epochs completed
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay value
        warmup_steps: Learning rate warmup steps
        grad_clip: Gradient clipping value

        # Training metrics
        final_train_loss: Final training loss
        final_val_loss: Final validation loss
        best_val_loss: Best validation loss achieved
        train_samples: Number of training samples
        val_samples: Number of validation samples
        steps_trained: Total training steps
        last_step: Last checkpoint step

        # Lineage and provenance
        lineage: Model lineage (parent model chain)
        base_model: Base model used for fine-tuning
        trained_from: Checkpoint path if continued training
        created_at: ISO timestamp of model creation
        trained_at: ISO timestamp of training completion
        exported_at: ISO timestamp of export

        # Personality (SloughGPT specific)
        soul_name: Name of the model's soul
        soul_hash: Slo integrity hash
        personality: Personality traits dict
        behavior: Behavior patterns dict
        cognition: Cognitive style dict
        emotion: Emotional signature dict

        # Technical metadata
        precision: Model precision (fp32, fp16, bf16)
        quantization: Quantization type if applicable
        export_format: Export format used
        export_version: Export format version
        sloughgpt_version: SloughGPT version
        torch_version: PyTorch version used
        architecture: Architecture description

        # Custom tags
        tags: List of arbitrary tags
        notes: Additional notes
        config: Additional configuration dict

    Example::

        metadata = ModelMetadata(
            name="sloughgpt-finetuned",
            model_type="sloughgpt",
            vocab_size=256,
            n_embed=256,
            n_layer=6,
            n_head=8,
            training_dataset="my_dataset.jsonl",
            epochs_trained=10,
            final_train_loss=0.05,
            final_val_loss=0.08,
            lineage="sloughgpt-base",
        )
    """

    # Identification
    name: str = "sloughgpt"
    model_type: str = "sloughgpt"
    version: str = "1.0"

    # Architecture
    vocab_size: int = 256
    n_embed: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: Optional[int] = None
    block_size: int = 128
    max_seq_len: int = 2048

    # Training config
    training_dataset: str = ""
    validation_dataset: str = ""
    epochs_trained: int = 0
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Training metrics
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = 0.0
    train_samples: int = 0
    val_samples: int = 0
    steps_trained: int = 0
    last_step: int = 0

    # Lineage
    lineage: str = ""
    base_model: str = ""
    trained_from: str = ""

    # Timestamps
    created_at: str = ""
    trained_at: str = ""
    exported_at: str = ""

    # Slo (SloughGPT)
    soul_name: str = ""
    soul_hash: str = ""
    personality: Dict[str, Any] = field(default_factory=dict)
    behavior: Dict[str, Any] = field(default_factory=dict)
    cognition: Dict[str, Any] = field(default_factory=dict)
    emotion: Dict[str, Any] = field(default_factory=dict)

    # Technical
    precision: str = "fp32"
    quantization: str = ""
    export_format: str = ""
    export_version: str = "1.0"
    sloughgpt_version: str = "1.0"
    torch_version: str = ""
    architecture: str = ""

    # Custom
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    @classmethod
    def from_model(cls, model: "Module", name: str = "sloughgpt") -> "ModelMetadata":
        """Extract metadata from a model instance.

        Args:
            model: PyTorch model to extract metadata from
            name: Model name

        Returns:
            ModelMetadata instance
        """
        metadata = cls(name=name)

        # Extract from model._config if available
        if hasattr(model, "_config") and model._config:
            config = model._config
            for field in ["vocab_size", "n_embed", "n_layer", "n_head", "n_kv_head", "block_size"]:
                if field in config:
                    setattr(metadata, field, config[field])

        # Extract from model attributes
        for field in ["vocab_size", "n_embed", "n_layer", "n_head", "block_size"]:
            if hasattr(model, field):
                val = getattr(model, field)
                if isinstance(val, (int, str)):
                    setattr(metadata, field, val)

        # Set timestamps
        metadata.created_at = datetime.datetime.utcnow().isoformat() + "Z"

        # Get PyTorch version
        try:
            import torch
            metadata.torch_version = torch.__version__
        except ImportError:
            pass

        return metadata

    def add_training_info(
        self,
        dataset: str = "",
        epochs: int = 0,
        train_loss: float = 0.0,
        val_loss: float = 0.0,
        steps: int = 0,
    ) -> "ModelMetadata":
        """Add training information to metadata.

        Args:
            dataset: Training dataset name/path
            epochs: Number of epochs trained
            train_loss: Final training loss
            val_loss: Final validation loss
            steps: Total training steps

        Returns:
            Self for chaining
        """
        self.training_dataset = dataset
        self.epochs_trained = epochs
        self.final_train_loss = train_loss
        self.final_val_loss = val_loss
        self.steps_trained = steps
        self.last_step = steps
        self.trained_at = datetime.datetime.utcnow().isoformat() + "Z"

        if val_loss > 0 and (self.best_val_loss == 0 or val_loss < self.best_val_loss):
            self.best_val_loss = val_loss

        return self

    def add_soul_info(
        self,
        soul_name: str = "",
        personality: Optional[Dict] = None,
        soul_hash: str = "",
    ) -> "ModelMetadata":
        """Add soul/personality information.

        Args:
            soul_name: Name of the soul
            personality: Personality traits dict
            soul_hash: Slo integrity hash

        Returns:
            Self for chaining
        """
        self.soul_name = soul_name
        self.soul_hash = soul_hash
        if personality:
            self.personality = personality
        return self

    def validate(self) -> List[str]:
        """Validate metadata completeness.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        # Required fields
        if self.vocab_size <= 0:
            issues.append("vocab_size must be positive")
        if self.n_embed <= 0:
            issues.append("n_embed must be positive")
        if self.n_layer <= 0:
            issues.append("n_layer must be positive")
        if self.n_head <= 0:
            issues.append("n_head must be positive")

        # Warnings
        if not self.training_dataset:
            issues.append("warning: training_dataset not set")
        if self.epochs_trained == 0:
            issues.append("warning: epochs_trained is 0")
        if not self.lineage:
            issues.append("warning: lineage not set")

        return issues


def create_model_metadata(
    model: "Module",
    name: str = "sloughgpt",
    training_info: Optional[Dict[str, Any]] = None,
    soul_info: Optional[Dict[str, Any]] = None,
) -> ModelMetadata:
    """Create comprehensive model metadata.

    Args:
        model: Model to create metadata for
        name: Model name
        training_info: Optional training information dict
        soul_info: Optional soul/personality information

    Returns:
        ModelMetadata instance

    Example::

        metadata = create_model_metadata(
            model=my_model,
            name="sloughgpt-finetuned",
            training_info={
                "dataset": "custom_data.jsonl",
                "epochs": 10,
                "train_loss": 0.05,
                "val_loss": 0.08,
            },
            soul_info={
                "soul_name": "Assistant",
                "personality": {"helpfulness": 0.9},
            },
        )
    """
    metadata = ModelMetadata.from_model(model, name)

    if training_info:
        metadata.add_training_info(**training_info)

    if soul_info:
        metadata.add_soul_info(**soul_info)

    metadata.exported_at = datetime.datetime.utcnow().isoformat() + "Z"

    return metadata


@dataclass
class ExportConfig:
    """Configuration for model export.

    Attributes:
        input_path: Path to input model file (.pt, .safetensors, etc.)
        output_path: Path for exported model (extension added automatically)
        format: Export format. Options:
            - "safetensors" (default, recommended)
            - "safetensors_bf16" (full precision storage)
            - "onnx" (cross-platform)
            - "gguf_q4_k_m" (mobile/llama.rn)
            - "gguf_fp16" (for separate quantization)
            - "gguf_q5_k_m" (better quality mobile)
            - "gguf_q8_0" (high quality)
            - "torch" (PyTorch checkpoint)
            - "torchscript" (PyTorch C++ inference)
            - "sou" (SloughGPT soul + personality)
            - "all" (export all formats)
        quantization: GGUF quantization type (Q4_K_M, Q5_K_M, Q8_0, F16, F32)
        include_tokenizer: Whether to export tokenizer alongside model
        metadata: Optional metadata dictionary for model tagging
        seq_len: Sequence length for ONNX export (default: 128)
        opset_version: ONNX opset version (default: 17)
        n_ctx: Context length for GGUF export (default: 2048)

    Example::

        config = ExportConfig(
            input_path="models/sloughgpt.pt",
            output_path="exports/sloughgpt",
            format="onnx",
            seq_len=128,
            opset_version=17,
            metadata={
                "model_type": "sloughgpt",
                "training_dataset": "custom_dataset",
                "epochs_trained": 10,
            }
        )

    Tags:
        The following tags are automatically added to exported models:
        - format: Export format used
        - format_version: Format version
        - exported_at: ISO timestamp of export
        - sloughgpt_version: SloughGPT version
    """

    input_path: str = ""
    output_path: str = ""
    format: str = "safetensors"
    quantization: Optional[str] = None
    include_tokenizer: bool = True
    metadata: Optional[Dict[str, Any]] = None
    seq_len: int = 128
    opset_version: int = 17
    n_ctx: int = 2048


@dataclass
class ONNXExportOptions:
    """Advanced ONNX export options.

    Attributes:
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic dimension mappings
        opset_version: ONNX opset version (default: 17)
        optimize: Whether to optimize the model
        verbose: Verbose export output
        external_data: Use external data for large models (>2GB)
        dynamo_export: Use PyTorch 2.x dynamo-based export (default: True)

    Dynamic Axes Example::

        {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        }

    Supported Opsets:
        - 17 (latest, recommended)
        - 16, 15, 14, 13, 12, 11

    Note:
        PyTorch 2.6+ uses dynamo-based export by default which handles
        dynamic shapes better. Falls back to TorchScript export for
        older PyTorch versions.
    """

    input_names: List[str] = field(default_factory=lambda: ["input_ids"])
    output_names: List[str] = field(default_factory=lambda: ["logits"])
    dynamic_axes: Optional[Dict[str, Any]] = None
    opset_version: int = 17
    optimize: bool = True
    verbose: bool = False
    external_data: bool = True
    dynamo_export: bool = True


@dataclass
class GGUFExportOptions:
    """Advanced GGUF export options for mobile deployment.

    Attributes:
        model_name: Name for the model in GGUF metadata
        model_version: Version string
        quantization: Quantization type
        n_ctx: Context length (default: 2048)
        rope_freq_base: RoPE frequency base (default: 10000.0)
        rope_freq_scale: RoPE frequency scale (default: 1.0)
        use_gpu: Whether to use GPU layers (for compatible hardware)

    Quantization Types:
        ================  ========  =================================
        Type               Size     Use Case
        ================  ========  =================================
        Q4_K_M (REC)      ~4.5bpw  Best balance for mobile
        Q5_K_M            ~5.5bpw  Better quality, slightly larger
        Q8_0              ~8bpw    High quality, larger file
        F16               16-bit   Full precision, no quantization
        F32               32-bit    Full precision, largest file
        ================  ========  =================================

    Memory Estimation (Q4_K_M):
        - Model memory: ~0.45 bytes per parameter
        - KV cache: 2 * n_layers * n_embed * 2 * n_ctx bytes
        - Example: 1M params + 2048 ctx ≈ 5MB total

    llama.rn Integration:
        The exported GGUF is compatible with llama.rn for React Native.
        See: https://github.com/mybigday/llama.rn

    Example::

        options = GGUFExportOptions(
            model_name="sloughgpt",
            quantization="Q4_K_M",
            n_ctx=2048,
            rope_freq_base=10000.0,
        )
    """

    model_name: str = "sloughgpt"
    model_version: str = "1.0"
    quantization: str = "Q4_K_M"
    n_ctx: int = 2048
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0
    use_gpu: bool = False


def export_to_torchscript(
    model: "Module",
    output_path: str,
    example_input: Optional[Any] = None,
) -> str:
    """Export model to TorchScript format.

    TorchScript provides optimized C++ inference without Python dependency.

    Args:
        model: PyTorch model to export
        output_path: Path for output TorchScript file
        example_input: Example input for tracing (required for static graphs)

    Returns:
        Path to exported file

    Raises:
        RuntimeError: If tracing fails

    Example::

        traced = torch.jit.trace(model, example_input)
        traced.save("model.torchscript.pt")

    Note:
        TorchScript export uses either tracing (with example input) or
        scripting (without example input). Tracing captures actual
        execution flow but requires static shapes. Scripting preserves
        control flow but may not capture all dynamic behavior.
    """
    import torch

    model.eval()

    if example_input is not None:
        traced = torch.jit.trace(model, example_input)
    else:
        traced = torch.jit.script(model)

    traced.save(output_path)
    logger.info(f"Exported TorchScript: {output_path}")
    return output_path


def export_to_onnx(
    model: "Module",
    output_path: str,
    example_input: Optional[Any] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Any]] = None,
    seq_len: int = 128,
    opset_version: int = 17,
) -> str:
    """Export model to ONNX format.

    ONNX provides cross-platform inference with optimized runtimes.

    Args:
        model: SloughGPT model to export
        output_path: Path for output ONNX file
        example_input: Example input tensor [batch, seq_len]
        input_names: Names for input tensors (default: ["input_ids"])
        output_names: Names for output tensors (default: ["logits"])
        dynamic_axes: Dynamic dimension mappings for variable batch/seq
        seq_len: Maximum sequence length (default: 128)
        opset_version: ONNX opset version (default: 17)

    Returns:
        Path to exported ONNX file

    Raises:
        ImportError: If onnx or onnxruntime not installed
        RuntimeError: If ONNX export fails

    Supported Custom Operations:
        - RoPE (Rotary Position Embeddings)
        - RMSNorm (Root Mean Square Layer Normalization)
        - SwiGLU (Swish + Gated Linear Unit)
        - Standard attention with causal masking

    Deployment Targets:
        ===================  ======================================
        Target              Runtime
        ===================  ======================================
        Server/CPU          ONNX Runtime
        Web                 ONNX.js or TF.js conversion
        Mobile              OnnxRuntime Mobile
        NVIDIA GPU          TensorRT via ONNX
        Intel CPU           OpenVINO via ONNX
        ===================  ======================================

    Example::

        from domains.training.onnx_export import export_sloughgpt_to_onnx

        result = export_sloughgpt_to_onnx(
            model=my_model,
            output_path="model.onnx",
            example_input=torch.zeros(1, 128, dtype=torch.long),
            seq_len=128,
        )

    Note:
        The ONNX export uses PyTorch 2.x dynamo-based export when
        available, which handles dynamic shapes better than the
        legacy TorchScript-based exporter.
    """
    import torch

    from domains.training.onnx_export import export_sloughgpt_to_onnx as onnx_export, ONNXExportConfig

    if input_names is None:
        input_names = ["input_ids"]
    if output_names is None:
        output_names = ["logits"]

    config = ONNXExportConfig(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )

    try:
        result = onnx_export(
            model=model,
            output_path=output_path,
            example_input=example_input,
            config=config,
            seq_len=seq_len,
        )
        logger.info(f"Exported ONNX: {output_path}")
        return result
    except Exception as e:
        logger.warning(f"Advanced ONNX export failed: {e}, using basic export")

        if example_input is None:
            example_input = torch.zeros(1, seq_len, dtype=torch.long)

        model.eval()
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or {},
            opset_version=opset_version,
            do_constant_folding=True,
        )
        logger.info(f"Exported ONNX (basic): {output_path}")
        return output_path


def export_to_safetensors(
    model: "Module",
    output_path: str,
    metadata: Optional[Dict] = None,
    dtype: Optional[str] = None,
) -> str:
    """Export model weights to SafeTensors format.

    SafeTensors is the recommended default format for production model storage.

    Args:
        model: PyTorch model to export
        output_path: Path for output file (.safetensors)
        metadata: Optional metadata dictionary
        dtype: Optional dtype conversion ("bf16" for bfloat16)

    Returns:
        Path to exported file

    Raises:
        ImportError: If safetensors not installed

    Advantages:
        - Safe deserialization (no arbitrary code execution)
        - Memory-mapped loading (zero-copy, fast loading)
        - Lazy loading (load only needed tensors)
        - Metadata support (tags, version, etc.)

    Metadata Example::

        {
            "format": "safetensors",
            "format_version": "1.0",
            "model_type": "sloughgpt",
            "vocab_size": 256,
            "n_embed": 256,
            "n_layer": 6,
            "exported_at": "2026-03-22T10:00:00Z",
        }

    Training Core Compatibility:
        The exported metadata includes all fields needed by training core:
        - Architecture: vocab_size, n_embed, n_layer, n_head, block_size
        - Training: epochs_trained, final_train_loss, final_val_loss
        - Lineage: lineage, base_model, trained_from
        - Slo: soul_name, personality, behavior (if applicable)

    Note:
        A .meta.json file is also created alongside the .safetensors
        file containing the metadata in human-readable format.
    """
    from safetensors.torch import save_file
    import json
    import torch
    import numpy as np

    state_dict = model.state_dict()
    state_dict = {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in state_dict.items()
        if not isinstance(v, dict)
    }
    if dtype == "bf16":
        state_dict = {k: v.to(dtype="bfloat16") for k, v in state_dict.items()}
        precision = "bf16"
    else:
        precision = "fp16" if dtype == "fp16" else "fp32"

    # Create comprehensive metadata
    if isinstance(metadata, ModelMetadata):
        model_meta = metadata
        model_meta.precision = precision
        model_meta.export_format = "safetensors"
        model_meta.exported_at = datetime.datetime.utcnow().isoformat() + "Z"
        meta = model_meta.to_dict()
    else:
        meta = metadata.copy() if metadata else {}
        meta["format"] = "safetensors"
        meta["format_version"] = "1.0"
        meta["precision"] = precision
        meta["exported_at"] = datetime.datetime.utcnow().isoformat() + "Z"

        # Extract architecture from model if not present
        for field in ["vocab_size", "n_embed", "n_layer", "n_head", "n_kv_head", "block_size"]:
            if field not in meta and hasattr(model, field):
                meta[field] = getattr(model, field)

        # Extract from model._config
        if hasattr(model, "_config") and model._config:
            for field in ["vocab_size", "n_embed", "n_layer", "n_head", "n_kv_head", "block_size"]:
                if field not in meta and field in model._config:
                    meta[field] = model._config[field]

    # Convert all non-string values to strings for safetensors compatibility
    def _to_str(v):
        if isinstance(v, str):
            return v
        elif isinstance(v, (int, float, bool)):
            return str(v)
        elif isinstance(v, (list, tuple)):
            return str(v)
        elif isinstance(v, dict):
            return json.dumps(v, default=str)
        else:
            return str(v)

    meta_str = {k: _to_str(v) for k, v in meta.items()}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output_path, metadata=meta_str)

    # Create human-readable .meta.json
    meta_path = output_path.replace(".safetensors", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Exported SafeTensors: {output_path}")
    logger.info(f"  Metadata: {meta_path}")
    return output_path


def export_to_safetensors_bf16(
    model: "Module",
    output_path: str,
    metadata: Optional[Dict] = None,
) -> str:
    """Export model weights to SafeTensors with BF16 precision.

    BF16 provides full float range with half the memory of FP32.

    Args:
        model: PyTorch model to export
        output_path: Path for output file (-bf16.safetensors)
        metadata: Optional metadata dictionary

    Returns:
        Path to exported file

    Note:
        BFloat16 (BF16) has the same range as float32 but half the
        precision. It's more stable for training than FP16 and
        commonly used in cloud TPUs and modern accelerators.
    """
    return export_to_safetensors(model, output_path, metadata, dtype="bf16")


def export_to_gguf(
    model: "Module",
    output_path: str,
    quantization: str = "Q4_K_M",
    tokenizer: Any = None,
) -> str:
    """Export model to GGUF format for mobile deployment.

    GGUF is optimized for llama.cpp and compatible with llama.rn for React Native.

    Args:
        model: SloughGPT model to export
        output_path: Path for output file
        quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, F16, F32)
        tokenizer: Optional tokenizer for vocabulary export

    Returns:
        Path to exported GGUF file

    Raises:
        ImportError: If gguf not installed

    Quantization Comparison:
        =============  ========  ========  ================
        Type          Size      Quality   Recommended
        =============  ========  ========  ================
        Q4_K_M        ~4.5bpw   Good      Yes (mobile)
        Q5_K_M        ~5.5bpw   Better    Yes (quality)
        Q8_0          ~8bpw     High      No (too large)
        F16           16-bit     Full      For quantization
        F32           32-bit     Full      No (too large)
        =============  ========  ========  ================

    llama.rn Integration::

        import { initLlama } from 'llama.rn';

        const context = await initLlama({
            model: 'file:///path/to/model-Q4_K_M.gguf',
            n_ctx: 2048,
            n_gpu_layers: 99,  // Metal on iOS
        });

        const result = await complete(context, {
            prompt: 'Hello, how are you?',
        });

    Memory Requirements (Q4_K_M):
        - Model: ~0.45 bytes per parameter
        - KV Cache: ~0.5MB per 1024 context tokens
        - Example: 1M params + 2048 ctx ≈ 5MB

    See Also:
        :mod:`domains.training.gguf_export`: Advanced GGUF options
        https://github.com/mybigday/llama.rn
        https://github.com/ggerganov/llama.cpp
    """
    from domains.training.gguf_export import export_to_gguf as gguf_export, GGUFExportConfig

    config = GGUFExportConfig(quantization=quantization)

    result = gguf_export(
        model=model,
        output_path=output_path,
        tokenizer=tokenizer,
        config=config,
    )
    logger.info(f"Exported GGUF: {output_path} ({quantization})")
    return result


def export_to_gguf_fp16(
    model: "Module",
    output_path: str,
    tokenizer: Any = None,
) -> str:
    """Export model to GGUF FP16 (no quantization).

    Use this to create a base model for separate quantization with llama.cpp.

    Args:
        model: SloughGPT model to export
        output_path: Path for output file (-F16.gguf)
        tokenizer: Optional tokenizer

    Returns:
        Path to exported file

    Quantization Command::

        llama-quantize model-F16.gguf model-Q4_K_M.gguf Q4_K_M

    See Also:
        llama.cpp quantize tool: https://github.com/ggerganov/llama.cpp
    """
    from domains.training.gguf_export import export_to_gguf_fp16 as gguf_fp16_export
    result = gguf_fp16_export(model, output_path, tokenizer)
    logger.info(f"Exported GGUF FP16: {output_path}")
    return result


def export_to_gguf_q4_k_m(
    model: "Module",
    output_path: str,
    tokenizer: Any = None,
) -> str:
    """Export model to GGUF Q4_K_M (recommended for mobile).

    Q4_K_M provides the best balance of size and quality for mobile devices.

    Args:
        model: SloughGPT model to export
        output_path: Path for output file (-Q4_K_M.gguf)
        tokenizer: Optional tokenizer

    Returns:
        Path to exported file

    Note:
        Q4_K_M uses 4-bit quantization with medium quality. It maintains
        good model quality while significantly reducing model size and
        memory requirements.
    """
    from domains.training.gguf_export import export_to_gguf_q4_k_m as gguf_q4_k_m_export
    result = gguf_q4_k_m_export(model, output_path, tokenizer)
    logger.info(f"Exported GGUF Q4_K_M: {output_path}")
    return result


def export_to_torch(
    model: "Module",
    output_path: str,
    metadata: Optional[Dict] = None,
) -> str:
    """Export model to PyTorch format (legacy).

    This is the traditional PyTorch checkpoint format. For production,
    prefer SafeTensors which is safer and faster to load.

    Args:
        model: PyTorch model to export
        output_path: Path for output file (.pt)
        metadata: Optional metadata dictionary

    Returns:
        Path to exported file

    Warning:
        .pt files can contain arbitrary Python code and are not
        safe for untrusted sources. Use SafeTensors for production.
    """
    import torch

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {"format": "torch"},
    }
    torch.save(checkpoint, output_path)
    logger.info(f"Exported PyTorch: {output_path}")
    return output_path


def export_to_sou(
    model: "Module",
    output_path: str,
    soul_profile: Any = None,
    weights_only: bool = False,
) -> str:
    """Export model to .soul Slo Unit format.

    Self-contained model with living soul personality and characteristics.

    Args:
        model: SloughGPT model to export
        output_path: Path for output file (.soul)
        soul_profile: Optional soul profile with personality traits
        weights_only: Export weights only (no soul data)

    Returns:
        Path to exported file

    See Also:
        :mod:`domains.inference.slo_format`: Slo format details
    """
    from domains.inference import save_soul as sou_export

    sou_export(
        model=model,
        output_path=output_path,
        soul_profile=soul_profile,
        weights_only=weights_only,
    )
    logger.info(f"Exported Slo Unit: {output_path}")
    return output_path


def _replace_ext(path: str, new_ext: str) -> str:
    """Replace file extension, handling various input extensions."""
    p = Path(path)
    return str(p.parent / f"{p.stem}{new_ext}")


def _gguf_path(output_path: str, quantization: Optional[str]) -> str:
    """Get GGUF output path with quantization suffix."""
    q = (quantization or "Q4_K_M").upper()
    stem = Path(output_path).stem
    parent = Path(output_path).parent
    return str(parent / f"{stem}-{q}.gguf")


def export_all_formats(
    config: ExportConfig,
    model: "Module",
    tokenizer: Any,
    example_input: Any,
    results: Dict[str, str],
) -> None:
    """Export model to all supported formats.

    Exports to:
        - SafeTensors (.safetensors) - recommended default
        - PyTorch (.torch) - legacy format
        - ONNX (.onnx) - cross-platform
        - GGUF Q4_K_M (.gguf) - mobile (llama.rn)
        - Slo Unit (.soul) - SloughGPT self-contained + personality

    Args:
        config: Export configuration
        model: Model to export
        tokenizer: Optional tokenizer
        example_input: Example input for ONNX export
        results: Dictionary to store output paths

    Note:
        SafeTensors is the recommended format as it's safe, fast,
        and widely supported. The .torch format is included for
        backward compatibility. The .soul format includes personality
        data and living soul metadata.
    """
    output = _replace_ext(config.output_path, ".safetensors")
    export_to_safetensors(model, output, config.metadata)
    results["safetensors"] = output

    output = _replace_ext(config.output_path, ".torch")
    export_to_torch(model, output, config.metadata)
    results["torch"] = output

    if example_input is not None:
        output = _replace_ext(config.output_path, ".onnx")
        export_to_onnx(model, output, example_input, seq_len=config.seq_len)
        results["onnx"] = output

    try:
        output = _replace_ext(config.output_path, "-Q4_K_M.gguf")
        export_to_gguf(model, output, "Q4_K_M", tokenizer)
        results["gguf_q4_k_m"] = output
    except ImportError as e:
        logger.warning(f"GGUF export skipped: {e}")

    output = _replace_ext(config.output_path, ".soul")
    export_to_sou(model, output, soul_profile=config.metadata)
    results["sou"] = output


def export_model(
    config: ExportConfig,
    model: "Module" = None,
    tokenizer: Any = None,
    example_input: Any = None,
) -> Dict[str, str]:
    """Export a model to various formats.

    This is the main export dispatcher that handles all supported formats.

    Args:
        config: ExportConfig with format and options
        model: Trained model to export
        tokenizer: Optional tokenizer for vocab export
        example_input: Example input for ONNX export

    Returns:
        Dictionary mapping format names to output file paths

    Raises:
        ValueError: If model is None or format is invalid
        ImportError: If required export package not installed

    Export Formats:

        ============  ====================================================
        Format        Description
        ============  ====================================================
        safetensors   SafeTensors weights (recommended default)
        safetensors_bf16  SafeTensors with BF16 precision
        onnx          ONNX model for cross-platform deployment
        gguf          GGUF quantized model
        gguf_q4_k_m   GGUF Q4_K_M (recommended for mobile)
        gguf_fp16     GGUF FP16 (for separate quantization)
        gguf_q5_k_m   GGUF Q5_K_M (better quality mobile)
        gguf_q8_0     GGUF Q8_0 (high quality)
        torch         PyTorch checkpoint (legacy)
        torchscript   TorchScript for C++ inference
        sou           Slo Unit with personality
        all           Export all formats at once
        ============  ====================================================

    Example::

        from domains.training.export import export_model, ExportConfig

        config = ExportConfig(
            input_path="model.pt",
            output_path="exports/model",
            format="all",
            metadata={"model_type": "sloughgpt"},
        )
        results = export_model(config, model=my_model)

        # results = {
        #     "safetensors": "exports/model.safetensors",
        #     "torch": "exports/model.torch",
        #     "onnx": "exports/model.onnx",
        #     "gguf_q4_k_m": "exports/model-Q4_K_M.gguf",
        # }

    Tags:
        The following metadata tags are automatically added:
        - format: Export format
        - format_version: Format version
        - exported_at: ISO timestamp

    See Also:
        :class:`ExportConfig`: Configuration options
        :func:`list_export_formats`: List all formats with descriptions
    """
    results = {}
    formats = config.format.split(",") if "," in config.format else [config.format]

    for fmt in formats:
        fmt = fmt.strip().lower()
        output = None

        try:
            if fmt == "safetensors":
                output = _replace_ext(config.output_path, ".safetensors")
                export_to_safetensors(model, output, config.metadata)
                results["safetensors"] = output

            elif fmt == "safetensors_bf16":
                output = _replace_ext(config.output_path, "-bf16.safetensors")
                export_to_safetensors_bf16(model, output, config.metadata)
                results["safetensors_bf16"] = output

            elif fmt == "gguf":
                output = _gguf_path(config.output_path, config.quantization or "Q4_K_M")
                export_to_gguf(model, output, config.quantization or "Q4_K_M", tokenizer)
                results["gguf"] = output

            elif fmt == "gguf_q4_k_m":
                output = _replace_ext(config.output_path, "-Q4_K_M.gguf")
                export_to_gguf(model, output, "Q4_K_M", tokenizer)
                results["gguf_q4_k_m"] = output

            elif fmt == "gguf_fp16":
                output = _replace_ext(config.output_path, "-F16.gguf")
                export_to_gguf_fp16(model, output, tokenizer)
                results["gguf_fp16"] = output

            elif fmt == "gguf_q8_0":
                output = _replace_ext(config.output_path, "-Q8_0.gguf")
                export_to_gguf(model, output, "Q8_0", tokenizer)
                results["gguf_q8_0"] = output

            elif fmt == "gguf_q5_k_m":
                output = _replace_ext(config.output_path, "-Q5_K_M.gguf")
                export_to_gguf(model, output, "Q5_K_M", tokenizer)
                results["gguf_q5_k_m"] = output

            elif fmt in ["gguf_f16", "gguf_f32"]:
                q_map = {"gguf_f16": "F16", "gguf_f32": "F32"}
                output = _replace_ext(config.output_path, f"-{q_map[fmt]}.gguf")
                export_to_gguf(model, output, q_map[fmt], tokenizer)
                results[fmt] = output

            elif fmt == "torch" or fmt == "pytorch":
                output = _replace_ext(config.output_path, ".pt")
                export_to_torch(model, output, config.metadata)
                results["torch"] = output

            elif fmt == "torchscript":
                output = _replace_ext(config.output_path, ".torchscript.pt")
                if example_input is None:
                    logger.warning("example_input required for TorchScript export")
                    continue
                export_to_torchscript(model, output, example_input)
                results["torchscript"] = output

            elif fmt == "onnx":
                output = _replace_ext(config.output_path, ".onnx")
                export_to_onnx(
                    model, output,
                    example_input=example_input,
                    seq_len=config.seq_len,
                    opset_version=config.opset_version,
                )
                results["onnx"] = output

            elif fmt == "sou":
                from domains.inference import create_soul_profile

                soul = create_soul_profile(
                    name=config.metadata.get("name", Path(config.output_path).stem) if config.metadata else Path(config.output_path).stem,
                    base_model="sloughgpt",
                    training_dataset=config.metadata.get("training_dataset", "") if config.metadata else "",
                    epochs_trained=config.metadata.get("epochs_trained", 0) if config.metadata else 0,
                    final_train_loss=config.metadata.get("final_train_loss", 0.0) if config.metadata else 0.0,
                    final_val_loss=config.metadata.get("final_val_loss", 0.0) if config.metadata else 0.0,
                    lineage="sloughgpt",
                    **({"lineage": config.metadata["lineage"]} if config.metadata and "lineage" in config.metadata else {}),
                )
                output = _replace_ext(config.output_path, ".soul")
                export_to_sou(model, output, soul_profile=soul)
                results["sou"] = output

            elif fmt == "all":
                export_all_formats(config, model, tokenizer, example_input, results)

        except Exception as e:
            logger.error(f"Export failed for format '{fmt}': {e}")

    if tokenizer and config.include_tokenizer:
        tokenizer_path = Path(config.output_path).parent / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_path))
        results["tokenizer"] = str(tokenizer_path)

    return results


def list_export_formats() -> Dict[str, str]:
    """List supported export formats with descriptions.

    Returns:
        Dictionary mapping format names to descriptions

    Format Categories:
        1. Recommended (default): SafeTensors
        2. Cross-platform: ONNX
        3. Mobile: GGUF Q4_K_M
        4. Legacy: PyTorch, TorchScript

    See Also:
        :class:`ExportConfig`: Configuration options
        :func:`export_model`: Main export function
    """
    return {
        "safetensors": "SafeTensors (.safetensors) - RECOMMENDED default (safe, fast)",
        "safetensors_bf16": "SafeTensors BF16 (-bf16.safetensors) - full precision storage",
        "onnx": "ONNX (.onnx) - cross-platform (server, web, TF.js)",
        "gguf_q4_k_m": "GGUF Q4_K_M (.gguf) - RECOMMENDED for mobile (llama.rn)",
        "gguf_fp16": "GGUF FP16 (.gguf) - for separate quantization",
        "gguf_q5_k_m": "GGUF Q5_K_M (.gguf) - better quality mobile",
        "gguf_q8_0": "GGUF Q8_0 (.gguf) - high quality, larger size",
        "torch": "PyTorch (.pt) - training checkpoint",
        "torchscript": "TorchScript (.torchscript.pt) - PyTorch C++ inference",
        "sou": "Slo Unit (.soul) - SloughGPT self-contained + personality",
        "all": "Export all formats at once",
    }


__all__ = [
    # Configuration
    "ExportConfig",
    "ONNXExportOptions",
    "GGUFExportOptions",
    # Metadata (training core compatibility)
    "ModelMetadata",
    "create_model_metadata",
    # Export functions
    "export_model",
    "export_to_torch",
    "export_to_safetensors",
    "export_to_safetensors_bf16",
    "export_to_gguf",
    "export_to_gguf_fp16",
    "export_to_gguf_q4_k_m",
    "export_to_torchscript",
    "export_to_onnx",
    "export_to_sou",
    "export_all_formats",
    "list_export_formats",
]
