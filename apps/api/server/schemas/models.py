"""
Model Schemas - Data models for model management
"""

from enum import Enum

from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADED = "loaded"
    LOADING = "loading"
    ERROR = "error"
    AVAILABLE = "available"


class Device(str, Enum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    AUTO = "auto"


class LoadModelRequest(BaseModel):
    model_id: str = Field(..., description="HuggingFace model ID or local path")
    device: Device = Device.AUTO
    mode: str | None = Field(None, description="Load mode: local or api")
    quantize: str | None = Field(None, description="Quantization type: q4, q8, f16")


class ModelInfo(BaseModel):
    model_id: str
    status: ModelStatus
    device: str
    parameters: int = 0
    vocab_size: int = 0
    loaded_at: str | None = None
    description: str = ""


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
    current_model: str | None = None


class LoadModelResponse(BaseModel):
    status: str
    model_id: str | None = None
    device: str | None = None
    parameters: int = 0
    error: str | None = None
    type: str | None = None
    loaded_at: str | None = None
