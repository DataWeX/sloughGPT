"""Training Domain.

This domain provides unified training capabilities.
All consumers should import from specific submodules:

    from domains.training.service import ...
    from domains.training.checkpoints import ...
    from domains.training.turbo import ...
    from domains.training.data_import import ...
    from domains.training.export import ...
"""

from __future__ import annotations

import csv
import json
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator
from pathlib import Path


class DatasetType(Enum):
    """Types of training datasets."""
    TEXT = "text"
    CODE = "code"
    CONVERSATION = "conversation"
    INSTRUCTION = "instruction"
    AUDIO_TEXT = "audio_text"
    IMAGE_TEXT = "image_text"
    VIDEO_TEXT = "video_text"
    MULTIMODAL = "multimodal"


class DataFormat(Enum):
    """Supported data formats for training datasets."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"


@dataclass
class DatasetConfig:
    """Configuration for a training dataset."""
    name: str
    dataset_type: DatasetType
    data_format: DataFormat
    path: str
    max_samples: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetManager:
    """Manages registered datasets."""
    
    def __init__(self):
        self._datasets: Dict[str, DatasetConfig] = {}
    
    @property
    def datasets(self) -> Dict[str, DatasetConfig]:
        return self._datasets
    
    def register_dataset(self, config: DatasetConfig) -> None:
        self._datasets[config.name] = config
    
    def get_dataset(self, name: str) -> Optional[DatasetConfig]:
        return self._datasets.get(name)
    
    def list_datasets(self) -> List[DatasetConfig]:
        return list(self._datasets.values())
    
    def list_by_type(self, dataset_type: DatasetType) -> List[DatasetConfig]:
        """List datasets filtered by type."""
        return [cfg for cfg in self._datasets.values() if cfg.dataset_type == dataset_type]
    
    def remove_dataset(self, name: str) -> bool:
        if name in self._datasets:
            del self._datasets[name]
            return True
        return False
    
    def load_dataset(self, name: str) -> List[Dict[str, Any]]:
        """Load all records from a dataset."""
        config = self._datasets.get(name)
        if config is None:
            raise ValueError("Dataset not found")
        
        path = Path(config.path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {config.path}")
        
        records = []
        max_samples = config.max_samples
        
        if config.data_format == DataFormat.JSONL:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if max_samples and len(records) >= max_samples:
                        break
        elif config.data_format == DataFormat.JSON:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if max_samples and len(records) >= max_samples:
                        break
        elif config.data_format == DataFormat.CSV:
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))
                    if max_samples and len(records) >= max_samples:
                        break
        
        return records
    
    def stream_dataset(self, name: str) -> Iterator[Dict[str, Any]]:
        """Stream records from a dataset."""
        config = self._datasets.get(name)
        if config is None:
            raise ValueError("Dataset not found")
        
        path = Path(config.path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {config.path}")
        
        max_samples = config.max_samples
        count = 0
        
        if config.data_format == DataFormat.JSONL:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        continue
                    if max_samples and count >= max_samples:
                        break
        elif config.data_format == DataFormat.JSON:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        continue
                    if max_samples and count >= max_samples:
                        break
        elif config.data_format == DataFormat.CSV:
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield dict(row)
                    count += 1
                    if max_samples and count >= max_samples:
                        break
    
    def scan_directory(self, directory: str) -> int:
        """Scan a directory for datasets and register them."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0
        
        count = 0
        for item in dir_path.iterdir():
            if item.is_dir():
                if item.name.startswith("_"):
                    continue
                # Check if directory has data files
                data_files = list(item.glob("*.jsonl")) + list(item.glob("*.json")) + list(item.glob("*.txt"))
                if not data_files:
                    continue
                # Register if not already registered
                if item.name not in self._datasets:
                    dataset_type = detect_dataset_type(str(item))
                    self.register_dataset(DatasetConfig(
                        name=item.name,
                        dataset_type=dataset_type,
                        data_format=DataFormat.JSONL,
                        path=str(item),
                    ))
                    count += 1
        return count
    
    def summarize(self) -> Dict[str, List[str]]:
        """Summarize registered datasets by type."""
        summary: Dict[str, List[str]] = {}
        for name, config in self._datasets.items():
            type_name = config.dataset_type.value
            if type_name not in summary:
                summary[type_name] = []
            summary[type_name].append(name)
        return summary


def detect_dataset_type(path: str) -> DatasetType:
    """Detect dataset type from file extension and content."""
    p = Path(path)
    
    # Handle directories
    if p.is_dir():
        # Look for data files in the directory
        for child in p.iterdir():
            if child.is_file() and child.suffix in (".jsonl", ".json", ".txt"):
                return detect_dataset_type(str(child))
        return DatasetType.TEXT
    
    suffix = p.suffix.lower()
    
    # Code file extensions (only common ones, others detected by content)
    code_extensions = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rb", ".php"}
    if suffix in code_extensions:
        return DatasetType.CODE
    
    # Files that need content check for code detection
    content_check_extensions = {".rs", ".jsx", ".tsx", ".txt"}
    if suffix in content_check_extensions:
        try:
            with open(path) as f:
                content = f.read(1000)
                code_patterns = ["def ", "class ", "import ", "function ", "const ", "let ", "var "]
                matches = sum(1 for p in code_patterns if p in content)
                if matches >= 2:
                    return DatasetType.CODE
        except Exception:
            pass
    
    if suffix == ".jsonl":
        try:
            with open(path) as f:
                first_line = f.readline()
                data = json.loads(first_line)
                if "messages" in data or "conversation" in data:
                    return DatasetType.CONVERSATION
                elif "instruction" in data:
                    return DatasetType.INSTRUCTION
                elif "audio" in data or "speech" in data or "wav" in data:
                    return DatasetType.AUDIO_TEXT
                elif "image" in data or "jpg" in data or "png" in data:
                    return DatasetType.IMAGE_TEXT
                elif "video" in data:
                    return DatasetType.VIDEO_TEXT
        except (json.JSONDecodeError, KeyError, Exception):
            pass
        return DatasetType.TEXT
    
    if suffix == ".json":
        try:
            with open(path) as f:
                data = json.load(f)
                if "messages" in data or "conversation" in data:
                    return DatasetType.CONVERSATION
        except (json.JSONDecodeError, Exception):
            pass
        return DatasetType.TEXT
    
    return DatasetType.TEXT


class PreprocessingStepType(Enum):
    """Types of preprocessing steps."""
    CLEAN = "clean"
    TOKENIZE = "tokenize"
    FILTER = "filter"


class PipelineStageType(Enum):
    """Types of pipeline stages."""
    PREPROCESS = "preprocess"
    TRAIN = "train"
    VALIDATE = "validate"
    SAVE = "save"


@dataclass
class PipelineConfig:
    """Configuration for a training pipeline."""
    name: str
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 1e-4
    stages: List[PipelineStageType] = field(default_factory=list)
    preprocessing_steps: List[PreprocessingStepType] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainingPipeline:
    """Training pipeline manager."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stages: List[Dict[str, Any]] = []
    
    def add_stage(self, name: str, stage_type: PipelineStageType, handler: Any = None) -> "TrainingPipeline":
        """Add a stage to the pipeline."""
        self.stages.append({
            "name": name,
            "type": stage_type,
            "handler": handler,
        })
        return self
    
    async def run(self, data_iter: Any) -> Dict[str, Any]:
        """Run the pipeline for the configured number of epochs."""
        epochs = self.config.epochs
        stages_run = []
        
        for epoch in range(epochs):
            for stage in self.stages:
                stages_run.append(stage["name"])
        
        return {
            "epochs": epochs,
            "stages": stages_run,
        }


class ModelType(Enum):
    """Types of models."""
    LANGUAGE_MODEL = "language_model"
    CHAT_MODEL = "chat_model"


class ModelArchitecture(Enum):
    """Model architectures."""
    GPT = "gpt"
    BERT = "bert"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    model_type: ModelType
    architecture: ModelArchitecture
    hidden_size: int = 768
    num_layers: int = 12
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelManager:
    """Manages registered models."""
    
    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
    
    @property
    def models(self) -> Dict[str, ModelConfig]:
        return self._models
    
    def register_model(self, config: ModelConfig) -> None:
        self._models[config.name] = config
    
    def get_model(self, name: str) -> Optional[ModelConfig]:
        return self._models.get(name)
    
    def list_models(self) -> List[ModelConfig]:
        return list(self._models.values())
    
    def create_model(self, name: str) -> Dict[str, Any]:
        """Create a model instance from registered config."""
        config = self._models.get(name)
        if config is None:
            raise ValueError(f"Model '{name}' not found")
        return {
            "name": config.name,
            "model_type": config.model_type.value,
            "architecture": config.architecture.value,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "ready": True,
            "config": config,
        }


class DataPreprocessor:
    """Data preprocessing pipeline."""
    
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
    
    def add_cleaning(self, text_field: str = "text", lowercase: bool = True) -> "DataPreprocessor":
        """Add a cleaning step for a field."""
        self.steps.append({
            "type": PreprocessingStepType.CLEAN,
            "field": text_field,
            "lowercase": lowercase,
        })
        return self
    
    def add_filter(self, text_field: str = "text", min_length: int = 0) -> "DataPreprocessor":
        """Add a filter step for a field."""
        self.steps.append({
            "type": PreprocessingStepType.FILTER,
            "field": text_field,
            "min_length": min_length,
        })
        return self
    
    def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single record through all steps."""
        result = dict(record)
        for step in self.steps:
            step_type = step["type"]
            field_name = step["field"]
            
            if step_type == PreprocessingStepType.CLEAN:
                value = result.get(field_name, "")
                if not isinstance(value, str):
                    value = str(value) if value is not None else ""
                # Normalize whitespace (replace tabs, newlines with spaces, collapse multiple spaces)
                value = re.sub(r'\s+', ' ', value).strip()
                if step.get("lowercase", True):
                    value = value.lower()
                result[field_name] = value
            
            elif step_type == PreprocessingStepType.FILTER:
                value = result.get(field_name)
                if value is None:
                    return None
                if isinstance(value, str) and len(value) < step.get("min_length", 0):
                    return None
        
        return result
    
    def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of records, filtering out None results."""
        results = []
        for record in records:
            processed = self.process_record(record)
            if processed is not None:
                results.append(processed)
        return results
