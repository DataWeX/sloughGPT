"""Model export utilities for SloughGPT.

This module provides model export functionality for deploying trained SloNet
models to GGUF and SOU formats. ONNX and SafeTensors export were removed
(PyTorch is not supported in this project).

Export Formats
==============

GGUF (Mobile/Embedded)
---------------------
- Format: .gguf
- Pros: Quantization support, llama.cpp optimized, excellent mobile performance
- Use case: React Native (llama.rn), iOS/Android, embedded devices
- Requirements: gguf>=0.10.0
- Quantizations: Q4_K_M (recommended), Q5_K_M, Q8_0, F16, F32

Soul (.sou)
-----------
- Format: .soul
- Pros: Includes soul metadata (traits, system prompt), compact binary
- Use case: Loading models with personality into SloNet inference

Examples
========

Export to GGUF::

    from domains.training.export import export_to_gguf
    export_to_gguf(model, "model.gguf", quantization="Q4_K_M")

Export to Soul::

    from domains.training.export import export_to_sou
    export_to_sou(model, "model.soul")

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
- :func:`list_export_formats`: List all supported formats
- :mod:`domains.training.gguf_export`: GGUF-specific export

"""

import logging
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from dataclasses import dataclass, field, asdict

logger = logging.getLogger("slo.export")


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
    def from_model(cls, model: "Any", name: str = "sloughgpt") -> "ModelMetadata":
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
        metadata.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"

        metadata.torch_version = ""

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
        self.trained_at = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"

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
    model: "Any",
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

    metadata.exported_at = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"

    return metadata


@dataclass
class ExportConfig:
    """Configuration for model export.

    Attributes:
        input_path: Path to input model file (.soul, .safetensors, etc.)
        output_path: Path for exported model (extension added automatically)
        format: Export format. Options:
            - "safetensors" (default, recommended)
            - "safetensors_bf16" (full precision storage)
            - "onnx" (cross-platform)
            - "gguf_q4_k_m" (mobile/llama.rn)
            - "gguf_fp16" (for separate quantization)
            - "gguf_q5_k_m" (better quality mobile)
            - "gguf_q8_0" (high quality)
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
            input_path="models/sloughgpt.soul",
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


def export_to_gguf(
    model: "Any",
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
    logger.info(f"Exported GGUF: {output_path} ({quantization})",
        extra={"tag": "TRAIN"},)
    return result


def export_to_gguf_fp16(
    model: "Any",
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
    logger.info(f"Exported GGUF FP16: {output_path}",
        extra={"tag": "TRAIN"},)
    return result


def export_to_gguf_q4_k_m(
    model: "Any",
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
    logger.info(f"Exported GGUF Q4_K_M: {output_path}",
        extra={"tag": "TRAIN"},)
    return result


def export_to_sou(
    model: "Any",
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
    logger.info(f"Exported Slo Unit: {output_path}",
        extra={"tag": "TRAIN"},)
    return output_path


def export_model(config: ExportConfig, model: Any, tokenizer: Any) -> list:
    """Export a model using the given config.

    Dispatches to format-specific exporters and returns a list of
    exported file paths.

    Args:
        config: ExportConfig with output_path, format, etc.
        model: The model object to export.
        tokenizer: The tokenizer object (used for format-specific exports).

    Returns:
        List of file paths that were created.
    """
    results = []
    output_path = config.output_path
    fmt = config.format

    if fmt == "sou" or fmt == "all":
        path = export_to_sou(model, output_path, tokenizer)
        results.append(path)

    if fmt == "gguf_q4_k_m" or fmt == "all":
        path = export_to_gguf_q4_k_m(model, output_path, n_ctx=config.n_ctx)
        results.append(path)

    if fmt == "gguf_fp16" or fmt == "all":
        path = export_to_gguf_fp16(model, output_path, n_ctx=config.n_ctx)
        results.append(path)

    if not results:
        path = export_to_sou(model, output_path, tokenizer)
        results.append(path)

    return results


def list_export_formats() -> Dict[str, str]:
    """List supported export formats with descriptions.

    Returns:
        Dictionary mapping format names to descriptions

    Format Categories:
        1. Self-contained: Slo Unit
        2. Mobile: GGUF Q4_K_M

    See Also:
        :class:`ExportConfig`: Configuration options
    """
    return {
        "gguf_q4_k_m": "GGUF Q4_K_M (.gguf) - RECOMMENDED for mobile (llama.rn)",
        "gguf_fp16": "GGUF FP16 (.gguf) - for separate quantization",
        "gguf_q5_k_m": "GGUF Q5_K_M (.gguf) - better quality mobile",
        "gguf_q8_0": "GGUF Q8_0 (.gguf) - high quality, larger size",
        "sou": "Slo Unit (.soul) - SloughGPT self-contained + personality",
    }


__all__ = [
    # Configuration
    "ExportConfig",
    "GGUFExportOptions",
    # Metadata (training core compatibility)
    "ModelMetadata",
    "create_model_metadata",
    # Export functions
    "export_to_gguf",
    "export_to_gguf_fp16",
    "export_to_gguf_q4_k_m",
    "export_to_sou",
    "list_export_formats",
]
