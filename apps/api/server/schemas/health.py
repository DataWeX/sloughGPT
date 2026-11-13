"""
Health Schemas - Data models for health endpoints
"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    timestamp: str
    model_loaded: bool = False
    model_type: str | None = None
    is_inferencing: bool = False
    inference_count: int = 0
    summary: str = ""


class DetailedHealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    uptime_seconds: float
    timestamp: str
    system: dict[str, Any]
    gpu: dict[str, Any] = {}
    model_loaded: bool = False
    model_type: str | None = None
    inference: dict[str, Any] = {}


class LivenessResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
