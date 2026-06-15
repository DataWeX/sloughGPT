"""
Model Schemas - Data models for model management
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


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
    mode: Optional[str] = Field(None, description="Load mode: local or api")
    quantize: Optional[str] = Field(None, description="Quantization type: q4, q8, f16")


class ModelInfo(BaseModel):
    model_id: str
    status: ModelStatus
    device: str
    parameters: int = 0
    vocab_size: int = 0
    loaded_at: Optional[str] = None
    description: str = ""


class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    current_model: Optional[str] = None


class LoadModelResponse(BaseModel):
    status: str
    model_id: Optional[str] = None
    device: Optional[str] = None
    parameters: int = 0
    error: Optional[str] = None
    type: Optional[str] = None
    loaded_at: Optional[str] = None