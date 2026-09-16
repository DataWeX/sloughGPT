"""Backward-compatibility shim — imports from the new ``domain.training`` package."""

# Allow submodule access (domains.X.Y) for test mocking
import importlib as _importlib

from domain.training import (
    DataFormat,
    DataPreprocessor,
    DatasetConfig,
    DatasetManager,
    DatasetType,
    LoRAConfig,
    LoRAEmbedding,
    LoRALinear,
    LoRAType,
    ModelArchitecture,
    ModelConfig,
    ModelManager,
    ModelType,
    PipelineConfig,
    PipelineStageType,
    PreprocessingStepType,
    TrainingPipeline,
    apply_lora_to_model,
    detect_dataset_type,
    find_checkpoint,
    get_lora_parameters,
    load_lora_soul,
    load_soul,
)
from domain.training._internal import executor, slonet, state, train_pipeline  # noqa: F401


def __getattr__(name):
    try:
        return _importlib.import_module(f"domain.{name}")
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        return _importlib.import_module(f"domain.training._internal.{name}")
    except (ImportError, ModuleNotFoundError):
        pass
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
    "slonet",
    "executor",
    "state",
    "train_pipeline",
]
