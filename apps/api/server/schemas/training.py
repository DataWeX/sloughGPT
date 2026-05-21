"""
Training Schemas - Data models for training
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingJob(BaseModel):
    id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    model: str = "sloughgpt"
    dataset: str = "shakespeare"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4
    progress: int = 0
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


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
    name: str
    path: str
    size_mb: float
    created: str


class DatasetInfo(BaseModel):
    name: str
    path: str
    file_count: int