"""
Feedback Schemas - Data models for feedback
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class FeedbackRequest(BaseModel):
    message_id: str
    rating: str = Field(..., description="thumbs_up or thumbs_down")
    session_id: Optional[str] = None
    message_content: Optional[str] = None
    context: Optional[str] = None
    user_message: Optional[str] = None
    assistant_response: Optional[str] = None

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: str) -> str:
        if v not in ("thumbs_up", "thumbs_down"):
            raise ValueError(f"rating must be 'thumbs_up' or 'thumbs_down', got {v!r}")
        return v


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    message_id: str
    rating: str
    timestamp: str


class FeedbackStats(BaseModel):
    thumbs_up: int = 0
    thumbs_down: int = 0
    total: int = 0
    up_ratio: float = 0.0


class ConversationCreate(BaseModel):
    name: str
    session_id: Optional[str] = None


class ConversationUpdate(BaseModel):
    name: Optional[str] = None
    pinned: Optional[bool] = None
    starred: Optional[bool] = None


class ConversationResponse(BaseModel):
    id: str
    name: str
    session_id: str
    created_at: str
    updated_at: str
    pinned: bool = False
    starred: bool = False
    message_count: int = 0
