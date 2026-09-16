"""training — Unified training capabilities (datasets, checkpoints, LoRA).

Public API:
    DatasetType, DataFormat, DatasetConfig, DatasetManager
    detect_dataset_type, PreprocessingStepType, PipelineStageType
    PipelineConfig, TrainingPipeline
    ModelType, ModelArchitecture, ModelConfig, ModelManager, DataPreprocessor
    find_checkpoint, load_soul, load_lora_soul
    LoRAType, LoRAConfig, LoRALinear, LoRAEmbedding
    apply_lora_to_model, get_lora_parameters
"""

from domain.training._internal.checkpoints import (
    find_checkpoint,
    load_lora_soul,
    load_soul,
)
from domain.training._internal.training import (
    DataFormat,
    DataPreprocessor,
    DatasetConfig,
    DatasetManager,
    DatasetType,
    ModelArchitecture,
    ModelConfig,
    ModelManager,
    ModelType,
    PipelineConfig,
    PipelineStageType,
    PreprocessingStepType,
    TrainingPipeline,
    detect_dataset_type,
)

_LAZY_IMPORTS = {
    "LoRAType": ("._internal.lora", "LoRAType"),
    "LoRAConfig": ("._internal.lora", "LoRAConfig"),
    "LoRALinear": ("._internal.lora", "LoRALinear"),
    "LoRAEmbedding": ("._internal.lora", "LoRAEmbedding"),
    "apply_lora_to_model": ("._internal.lora", "apply_lora_to_model"),
    "get_lora_parameters": ("._internal.lora", "get_lora_parameters"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatasetType",
    "DataFormat",
    "DatasetConfig",
    "DatasetManager",
    "detect_dataset_type",
    "PreprocessingStepType",
    "PipelineStageType",
    "PipelineConfig",
    "TrainingPipeline",
    "ModelType",
    "ModelArchitecture",
    "ModelConfig",
    "ModelManager",
    "DataPreprocessor",
    "find_checkpoint",
    "load_soul",
    "load_lora_soul",
    "LoRAType",
    "LoRAConfig",
    "LoRALinear",
    "LoRAEmbedding",
    "apply_lora_to_model",
    "get_lora_parameters",
]
