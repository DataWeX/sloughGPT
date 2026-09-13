"""Backward-compatibility shim — imports from the new ``domain.training`` package."""

from domain.training import (
    DatasetType,
    DataFormat,
    DatasetConfig,
    DatasetManager,
    detect_dataset_type,
    PreprocessingStepType,
    PipelineStageType,
    PipelineConfig,
    TrainingPipeline,
    ModelType,
    ModelArchitecture,
    ModelConfig,
    ModelManager,
    DataPreprocessor,
    find_checkpoint,
    load_soul,
    load_lora_soul,
    LoRAType,
    LoRAConfig,
    LoRALinear,
    LoRAEmbedding,
    apply_lora_to_model,
    get_lora_parameters,
)
from domain.training._internal import slonet, executor, state, train_pipeline  # noqa: F401

# Allow submodule access (domains.X.Y) for test mocking
import importlib as _importlib
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
