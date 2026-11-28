"""
Model Schemas - Data models for model management
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class ModelStatus(StrEnum):
    UNLOADED = "unloaded"
    LOADED = "loaded"
    LOADING = "loading"
    ERROR = "error"
    AVAILABLE = "available"


class Device(StrEnum):
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
    # Weights present on disk (verified, not assumed). False means the id is
    # only a known Hub repo — selecting it starts a download, it does not
    # load. Without this the list read "available" for models that were
    # never downloaded, which is exactly the lie this flag removes.
    downloaded: bool = False


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
