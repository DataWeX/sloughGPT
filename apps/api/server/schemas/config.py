"""
Config Schemas - Data models for configuration
"""
from pydantic import BaseModel, Field
from typing import Optional


class GenerationConfig(BaseModel):
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=200)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    max_new_tokens: int = Field(default=200, ge=1, le=4096)
    max_context_length: int = Field(default=1024, ge=64, le=8192)


class ConfigUpdate(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None