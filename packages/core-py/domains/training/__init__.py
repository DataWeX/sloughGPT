"""
Training Domain - Simplified

This domain provides unified training capabilities.
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


# ============== Dataset Types ==============


class DatasetType(Enum):
    TEXT = "text"
    CODE = "code"
    CONVERSATION = "conversation"
    INSTRUCTION = "instruction"


class DataFormat(Enum):
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"


@dataclass
class DatasetConfig:
    name: str
    dataset_type: DatasetType
    data_format: DataFormat
    path: str
    max_samples: Optional[int] = None


class DatasetManager:
    """Unified dataset manager for multiple dataset types."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("sloughgpt.training.datasets")
        self.datasets: Dict[str, DatasetConfig] = {}

    def register_dataset(self, config: DatasetConfig) -> None:
        self.datasets[config.name] = config
        self.logger.info(f"Registered: {config.name}")

    def load_dataset(self, name: str) -> List[Dict[str, Any]]:
        config = self.datasets.get(name)
        if not config:
            raise ValueError(f"Dataset not found: {name}")

        records = []
        with open(config.path, "r") as f:
            for i, line in enumerate(f):
                if config.max_samples and i >= config.max_samples:
                    break
                if line.strip():
                    records.append(json.loads(line))
        return records

    def stream_dataset(self, name: str) -> Iterator[Dict[str, Any]]:
        config = self.datasets.get(name)
        if not config:
            raise ValueError(f"Dataset not found: {name}")

        with open(config.path, "r") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


# ============== Preprocessing Types ==============


class PreprocessingStepType(Enum):
    CLEAN = "clean"
    TOKENIZE = "tokenize"
    FILTER = "filter"


class DataPreprocessor:
    """Unified preprocessing pipeline."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("sloughgpt.training.preprocessing")
        self.steps: List[Dict[str, Any]] = []

    def add_cleaning(self, text_field: str = "text", lowercase: bool = True) -> "DataPreprocessor":
        self.steps.append(
            {"type": PreprocessingStepType.CLEAN, "field": text_field, "lowercase": lowercase}
        )
        return self

    def add_filter(self, text_field: str = "text", min_length: int = 10) -> "DataPreprocessor":
        self.steps.append(
            {"type": PreprocessingStepType.FILTER, "field": text_field, "min_length": min_length}
        )
        return self

    def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for step in self.steps:
            if step["type"] == PreprocessingStepType.CLEAN:
                text = record.get(step["field"], "")
                if step.get("lowercase"):
                    text = text.lower()
                text = re.sub(r"\s+", " ", text).strip()
                record[step["field"]] = text

            elif step["type"] == PreprocessingStepType.FILTER:
                text = record.get(step["field"], "")
                if len(text) < step.get("min_length", 0):
                    return None
        return record

    def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for record in records:
            processed = self.process_record(record)
            if processed:
                results.append(processed)
        return results


# ============== Pipeline Types ==============


class PipelineStageType(Enum):
    PREPROCESS = "preprocess"
    TRAIN = "train"
    VALIDATE = "validate"
    SAVE = "save"


@dataclass
class PipelineConfig:
    name: str
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 1e-4


class TrainingPipeline:
    """Unified training pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.logger = logging.getLogger("sloughgpt.training.pipelines")
        self.config = config
        self.stages: List[Dict[str, Any]] = []

    def add_stage(
        self, name: str, stage_type: PipelineStageType, handler: Any
    ) -> "TrainingPipeline":
        self.stages.append({"name": name, "type": stage_type, "handler": handler})
        return self

    async def run(self, train_data: Iterator[Any]) -> Dict[str, Any]:
        self.logger.info(f"Running pipeline: {self.config.name}")
        results = {"epochs": 0, "stages": []}

        for epoch in range(self.config.epochs):
            for stage in self.stages:
                self.logger.debug(f"Stage: {stage['name']}")
                results["stages"].append(stage["name"])
            results["epochs"] = epoch + 1

        return results


# ============== Model Types ==============


class ModelType(Enum):
    LANGUAGE_MODEL = "language_model"
    CHAT_MODEL = "chat_model"


class ModelArchitecture(Enum):
    GPT = "gpt"
    BERT = "bert"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    name: str
    model_type: ModelType
    architecture: ModelArchitecture
    hidden_size: int = 768
    num_layers: int = 12


class ModelManager:
    """Unified model manager."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("sloughgpt.training.models")
        self.models: Dict[str, ModelConfig] = {}

    def register_model(self, config: ModelConfig) -> None:
        self.models[config.name] = config
        self.logger.info(f"Registered model: {config.name}")

    def create_model(self, name: str) -> Dict[str, Any]:
        config = self.models.get(name)
        if not config:
            raise ValueError(f"Model not found: {name}")
        return {"name": name, "config": config, "ready": True}


# Export all classes
__all__ = [
    # Core training
    "DatasetManager",
    "DatasetType",
    "DatasetConfig",
    # "DataFormat",  # Moved to data_loader module
    "DataPreprocessor",
    "PreprocessingStepType",
    "TrainingPipeline",
    "PipelineConfig",
    "PipelineStageType",
    "ModelManager",
    "ModelConfig",
    "ModelType",
    "ModelArchitecture",
    # NanoGPT model
    "NanoGPT",
    "Block",
    "CausalSelfAttention",
    "MLP",
    # Unified training (NEW)
    "TrainingConfig",
    "DataLoader",
    "UniversalDataLoader",
    "ModelWrapper",
    "TorchModelWrapper",
    "Trainer",
    "train",
    # Dataset management
    "DatasetRegistry",
    "DatasetMixer",
    "DatasetInfo",
    # Distributed training
    "DistributedTrainer",
    "DistributedConfig",
    # HuggingFace
    "HuggingFaceManager",
    "HuggingFaceDatasetManager",
    # Dataset creation
    "DatasetCreator",
    "create_dataset",
    # Batch processing
    "BatchProcessor",
    "JobScheduler",
    # Validation
    "DatasetValidator",
    "DatasetVersion",
    "DatasetQualityScorer",
    # Universal data loading and training
    "load_dataset",
    "UniversalTrainer",
    # Dataset preparation
    "TextCleaner",
    "Tokenizer",
    "DatasetPreparer",
    "DatasetStats",
    "prepare_dataset",
    # Data import
    "DataImporter",
    "RepoImporter",
    "HuggingFaceImporter",
    "URLImporter",
    "ImportResult",
    "import_data",
    # Export modules (NEW)
    "export_model",
    "ExportConfig",
    "ModelMetadata",
    "create_model_metadata",
    "ONNXExportOptions",
    "GGUFExportOptions",
    "export_to_safetensors",
    "export_to_onnx",
    "export_to_gguf",
    "export_to_gguf_q4_k_m",
    "export_to_gguf_fp16",
    "export_to_torchscript",
    "export_to_sou",
    "export_all_formats",
    "list_export_formats",
    # ONNX export
    "SloughGPTONNXExport",
    "ONNXExportConfig",
    # GGUF export
    "GGUFExportConfig",
    "estimate_memory_requirements",
    "quantize_gguf",
    # Unified pipeline
    "UnifiedTrainingConfig",
    "UnifiedTrainingPipeline",
    "TrainingProgress",
    "TrainingStage",
    "FederatedRLTrainer",
]


# Lazy imports for optional dependencies
def get_nanogpt():
    """Get NanoGPT model (requires torch)."""
    try:
        from .models.nanogpt import NanoGPT

        return NanoGPT
    except ImportError:
        return None


def get_trainer():
    """Get trainer function and config (requires torch)."""
    try:
        from .unified_training import train, TrainingConfig

        return train, TrainingConfig
    except ImportError:
        return None, None


# Lazy imports - avoid importing torch-dependent modules at package load time
_TRAINING_EXTRA_AVAILABLE = None


def __getattr__(name):
    """Lazy import of torch-dependent modules."""
    global _TRAINING_EXTRA_AVAILABLE

    lazy_imports = {
        "NanoGPT": ".models.nanogpt",
        "Block": ".models.nanogpt",
        "CausalSelfAttention": ".models.nanogpt",
        "MLP": ".models.nanogpt",
        "TrainingConfig": ".unified_training",
        "DataLoader": ".unified_training",
        "UniversalDataLoader": ".unified_training",
        "ModelWrapper": ".unified_training",
        "TorchModelWrapper": ".unified_training",
        "train": ".unified_training",
        "DatasetRegistry": ".dataset_manager",
        "DatasetMixer": ".dataset_manager",
        "DatasetInfo": ".dataset_manager",
        "DistributedTrainer": ".distributed",
        "DistributedConfig": ".distributed",
        "HuggingFaceManager": ".huggingface",
        "HuggingFaceDatasetManager": ".huggingface",
        "DatasetCreator": ".dataset_creator",
        "create_dataset": ".dataset_creator",
        "BatchProcessor": ".batch_processor",
        "JobScheduler": ".batch_processor",
        "DatasetValidator": ".validator",
        "DatasetVersion": ".validator",
        "DatasetQualityScorer": ".quality_scorer",
        "TextCleaner": ".dataset_prep",
        "Tokenizer": ".dataset_prep",
        "DatasetPreparer": ".dataset_prep",
        "DatasetStats": ".dataset_prep",
        "prepare_dataset": ".dataset_prep",
        "DataImporter": ".data_import",
        "RepoImporter": ".data_import",
        "HuggingFaceImporter": ".data_import",
        "URLImporter": ".data_import",
        "ImportResult": ".data_import",
        "import_data": ".data_import",
        # Export modules
        "export_model": ".export",
        "ExportConfig": ".export",
        "ModelMetadata": ".export",
        "create_model_metadata": ".export",
        "export_to_safetensors": ".export",
        "export_to_onnx": ".export",
        "export_to_gguf": ".export",
        "export_to_gguf_q4_k_m": ".export",
        "export_to_gguf_fp16": ".export",
        "export_to_torchscript": ".export",
        "export_to_sou": ".export",
        "export_all_formats": ".export",
        "list_export_formats": ".export",
        # Unified pipeline
        "UnifiedTrainingConfig": ".unified_pipeline",
        "UnifiedTrainingPipeline": ".unified_pipeline",
        "TrainingProgress": ".unified_pipeline",
        "TrainingStage": ".unified_pipeline",
        "FederatedRLTrainer": ".unified_pipeline",
    }

    if name in lazy_imports:
        import importlib

        module = importlib.import_module(lazy_imports[name], package=__name__)
        obj = getattr(module, name)
        globals()[name] = obj  # Cache for future access
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
