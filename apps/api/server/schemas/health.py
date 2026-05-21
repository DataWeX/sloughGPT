"""
Health Schemas - Data models for health endpoints
"""
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    timestamp: str
    model_loaded: bool = False
    model_type: Optional[str] = None
    is_inferencing: bool = False
    inference_count: int = 0


class DetailedHealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    uptime_seconds: float
    timestamp: str
    system: Dict[str, Any]
    gpu: Dict[str, Any] = {}
    model_loaded: bool = False
    model_type: Optional[str] = None
    inference: Dict[str, Any] = {}


class LivenessResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str