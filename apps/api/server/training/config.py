"""Training engine config — pure dataclasses, no pydantic/fastapi deps.

CLI-safe: can be imported standalone without pulling in the API server.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class TrainingMode(Enum):
    DIRECT = "direct"
    DISTILL = "distill"


class DataSourceType(Enum):
    DATASET = "dataset"
    MANIFEST = "manifest"
    SESSION = "session"
    SUBPROCESS = "subprocess"


# ── Training configs ─────────────────────────────────────────────────


@dataclass
class DirectConfig:
    mode: TrainingMode = TrainingMode.DIRECT
    epochs: int = 3
    batch_size: int = 32
    n_embed: int = 128
    use_lora: bool = False
    stream: bool = False
    name: str = ""


@dataclass
class DistillConfig:
    mode: TrainingMode = TrainingMode.DISTILL
    teacher_model: str = "gpt2"
    temperature: float = 4.0
    alpha: float = 0.5
    beta: float = 0.5


@dataclass
class LoRAConfig:
    model_path: str = ""
    rank: int = 8
    alpha: float = 16.0
    output_dir: str = "models"


@dataclass
class VisualConfig:
    embed_dim: int = 256
    hidden_dim: int = 512
    n_vision_layers: int = 3
    max_frames: int = 8


TrainingConfig = Union[DirectConfig, DistillConfig, LoRAConfig, VisualConfig]


# ── Data sources ─────────────────────────────────────────────────────


@dataclass
class DatasetSource:
    type: DataSourceType = DataSourceType.DATASET
    name: str = ""
    path: str = None


@dataclass
class ManifestSource:
    type: DataSourceType = DataSourceType.MANIFEST
    manifest_uri: str = ""
    dataset_id: str = None


@dataclass
class SessionSource:
    type: DataSourceType = DataSourceType.SESSION
    min_pair_quality: float = 2.0
    max_pairs: int = 500


@dataclass
class SubprocessSource:
    type: DataSourceType = DataSourceType.SUBPROCESS
    script: str = None
    script_args: list = field(default_factory=list)


DataSource = Union[DatasetSource, ManifestSource, SessionSource, SubprocessSource]


# ── Resume ───────────────────────────────────────────────────────────


@dataclass
class ResumeConfig:
    checkpoint_name: str = None
    resume_path: str = None


# ── Job request ──────────────────────────────────────────────────────


@dataclass
class TrainingJobRequest:
    config: TrainingConfig = field(default_factory=DirectConfig)
    source: DataSource = field(default_factory=DatasetSource)
    resume: ResumeConfig = None
