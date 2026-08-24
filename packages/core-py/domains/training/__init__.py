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
    AUDIO_TEXT = "audio_text"
    IMAGE_TEXT = "image_text"
    VIDEO_TEXT = "video_text"
    MULTIMODAL = "multimodal"


def detect_dataset_type(path: str) -> DatasetType:
    """Auto-detect dataset type by sampling file contents.

    Scans first 10 lines of each file to determine modality.

    Args:
        path: Path to dataset file or directory

    Returns:
        Detected DatasetType
    """
    from pathlib import Path

    p = Path(path)
    if p.is_dir():
        files = list(p.glob("*.jsonl")) + list(p.glob("*.json")) + list(p.glob("*.txt"))
        if not files:
            return DatasetType.TEXT
        return detect_dataset_type(str(files[0]))

    text = p.read_text(encoding="utf-8", errors="replace")[:2000]

    # Check if JSON/JSONL
    if p.suffix in (".jsonl", ".json"):
        lines = text.strip().split("\n")
        for line in lines[:10]:
            try:
                record = json.loads(line)
                keys = set(record.keys())
                if "audio" in keys or "speech" in keys or "wav" in keys:
                    return DatasetType.AUDIO_TEXT
                if "image" in keys or "jpg" in keys or "png" in keys:
                    return DatasetType.IMAGE_TEXT
                if "instruction" in keys and "response" in keys:
                    return DatasetType.INSTRUCTION
                if "conversation" in keys or "messages" in keys:
                    return DatasetType.CONVERSATION
            except (json.JSONDecodeError, ValueError):
                continue

    # Check for code (contains common programming keywords)
    code_patterns = ["def ", "class ", "import ", "function ", "const ", "fn "]
    code_lines = sum(1 for pat in code_patterns if pat in text)
    if code_lines >= 3:
        return DatasetType.CODE

    return DatasetType.TEXT


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
    """Unified dataset manager for multiple dataset types.

    Auto-categorizes datasets by modality so you only load what's needed.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("slo.training.datasets")
        self.datasets: Dict[str, DatasetConfig] = {}

    def register_dataset(self, config: DatasetConfig) -> None:
        self.datasets[config.name] = config
        self.logger.info("Registered: %s (%s)", config.name, config.dataset_type.value,
            extra={"tag": "TRAIN"},)

    def list_by_type(self, dtype: DatasetType) -> List[DatasetConfig]:
        """List all datasets of a given type."""
        return [d for d in self.datasets.values() if d.dataset_type == dtype]

    def load_dataset(self, name: str) -> List[Dict[str, Any]]:
        config = self.datasets.get(name)
        if not config:
            raise ValueError(f"Dataset not found: {name}")

        records = []
        with open(config.path, "r") as f:
            for i, line in enumerate(f):
                if config.max_samples and i >= config.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                if config.data_format == DataFormat.JSONL or config.data_format == DataFormat.JSON:
                    records.append(json.loads(line))
                elif config.data_format == DataFormat.CSV:
                    import csv, io
                    reader = csv.DictReader(io.StringIO(line))
                    for row in reader:
                        records.append(row)
        return records

    def stream_dataset(self, name: str) -> Iterator[Dict[str, Any]]:
        config = self.datasets.get(name)
        if not config:
            raise ValueError(f"Dataset not found: {name}")

        with open(config.path, "r") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def scan_directory(self, directory: str = "datasets") -> int:
        """Auto-discover and register all datasets in a directory.

        Scans each subdirectory for .txt, .jsonl, .json files and auto-detects
        the dataset type by sampling content. Skips already-registered datasets.

        Args:
            directory: Root datasets directory

        Returns:
            Number of newly registered datasets
        """
        from pathlib import Path

        base = Path(directory)
        if not base.exists():
            self.logger.warning("Directory not found: %s", directory,
                extra={"tag": "TRAIN"},)
            return 0

        count = 0
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith("_"):
                continue

            if entry.name in self.datasets:
                continue

            # Find data files
            files = list(entry.glob("*.txt")) + list(entry.glob("*.jsonl")) + list(entry.glob("*.json"))
            if not files:
                continue

            path = str(files[0])
            dtype = detect_dataset_type(path)
            fmt = DataFormat.JSONL if path.endswith(".jsonl") else DataFormat.JSON if path.endswith(".json") else DataFormat.CSV

            config = DatasetConfig(
                name=entry.name,
                dataset_type=dtype,
                data_format=fmt,
                path=path,
            )
            self.datasets[entry.name] = config
            count += 1
            dtype_label = f"{dtype.value:>12}"
            self.logger.info("  [%s] %s (%s)", dtype_label, entry.name, files[0].name,
                extra={"tag": "TRAIN"},)

        return count

    def summarize(self) -> Dict[str, List[str]]:
        """Get a modality-grouped summary of all registered datasets."""
        summary = {}
        for name, cfg in self.datasets.items():
            t = cfg.dataset_type.value
            if t not in summary:
                summary[t] = []
            summary[t].append(name)
        return summary


# ============== Preprocessing Types ==============


class PreprocessingStepType(Enum):
    CLEAN = "clean"
    TOKENIZE = "tokenize"
    FILTER = "filter"


class DataPreprocessor:
    """Unified preprocessing pipeline."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("slo.training.preprocessing")
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
        self.logger = logging.getLogger("slo.training.pipelines")
        self.config = config
        self.stages: List[Dict[str, Any]] = []

    def add_stage(
        self, name: str, stage_type: PipelineStageType, handler: Any
    ) -> "TrainingPipeline":
        self.stages.append({"name": name, "type": stage_type, "handler": handler})
        return self

    async def run(self, train_data: Iterator[Any]) -> Dict[str, Any]:
        self.logger.info("Running pipeline: %s", self.config.name,
            extra={"tag": "TRAIN"},)
        results = {"epochs": 0, "stages": []}

        for epoch in range(self.config.epochs):
            for stage in self.stages:
                self.logger.debug("Stage: %s", stage['name'])
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
        self.logger = logging.getLogger("slo.training.models")
        self.models: Dict[str, ModelConfig] = {}

    def register_model(self, config: ModelConfig) -> None:
        self.models[config.name] = config
        self.logger.info("Registered model: %s", config.name,
            extra={"tag": "TRAIN"},)

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
    # Trainer protocol
    "TrainerProtocol",
    "TrainResult",
    # Export modules
    "ExportConfig",
    "ModelMetadata",
    "create_model_metadata",
    "export_to_gguf",
    "export_to_gguf_q4_k_m",
    "export_to_gguf_fp16",
    "export_to_sou",
    "list_export_formats",
    # GGUF export
    "GGUFExportOptions",
    "GGUFExportConfig",
    "estimate_memory_requirements",
    "quantize_gguf",
]


# Lazy imports for optional dependencies
def get_nanogpt():
    """Get NanoGPT model (requires torch)."""
    return None  # NanoGPT not available in this codebase


# Lazy imports - avoid importing torch-dependent modules at package load time
_TRAINING_EXTRA_AVAILABLE = None


def __getattr__(name):
    """Lazy import of torch-dependent modules."""
    global _TRAINING_EXTRA_AVAILABLE

    lazy_imports = {
        "DataImporter": ".data_import",
        "RepoImporter": ".data_import",
        "HuggingFaceImporter": ".data_import",
        "URLImporter": ".data_import",
        "ImportResult": ".data_import",
        "import_data": ".data_import",
        "TrainerProtocol": ".trainer_protocol",
        "TrainResult": ".trainer_protocol",
        # Export modules
        "ExportConfig": ".export",
        "ModelMetadata": ".export",
        "create_model_metadata": ".export",
        "export_to_gguf": ".export",
        "export_to_gguf_q4_k_m": ".export",
        "export_to_gguf_fp16": ".export",
        "export_to_sou": ".export",
        "list_export_formats": ".export",
        # GGUF export options (pure config, no torch)
        "GGUFExportOptions": ".export",
        # GGUF export
        "GGUFExportConfig": ".gguf_export",
        "estimate_memory_requirements": ".gguf_export",
        "quantize_gguf": ".gguf_export",
    }

    if name in lazy_imports:
        import importlib

        module = importlib.import_module(lazy_imports[name], package=__name__)
        obj = getattr(module, name)
        globals()[name] = obj  # Cache for future access
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
