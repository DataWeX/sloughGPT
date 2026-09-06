"""
Training Schemas - Data models for training
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    CANCELLED = "cancelled"


class TrainingJob(BaseModel):
    id: str
    name: str = ""
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    model: str = ""
    dataset: str = ""
    method: str = ""
    epochs: int = 0
    current_epoch: int = 0
    global_step: int = 0
    loss: float = 0.0
    train_loss: float = 0.0
    eval_loss: float = 0.0
    checkpoint: str = ""
    data_source: str = ""
    batch_size: int = 32
    learning_rate: float = 1e-4
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str = ""
    explanation: str = ""
    loss_history: list[float] = []
    reward_history: list[float] = []
    epochs_completed: int = 0
    status_message: str = ""


class JobCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    model: str = "sloughgpt"
    dataset: str = "shakespeare"
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-1)


class JobResponse(BaseModel):
    id: str
    name: str
    status: str
    progress: int = 0


class CheckpointInfo(BaseModel):
    name: str = ""
    path: str = ""
    size_mb: float = 0
    created_at: str | None = None


class DatasetInfo(BaseModel):
    name: str
    path: str
    file_count: int
