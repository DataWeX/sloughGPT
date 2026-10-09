"""
Config Schemas - Data models for configuration
"""

from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=200)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)
    max_new_tokens: int = Field(default=300, ge=1, le=4096)
    max_context_length: int = Field(default=1024, ge=64, le=8192)


class ConfigUpdate(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    max_context_length: int | None = None
