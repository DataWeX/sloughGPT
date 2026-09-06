"""
Feedback Schemas - Data models for feedback
"""

from pydantic import BaseModel, Field, field_validator


class FeedbackRequest(BaseModel):
    message_id: str
    rating: str = Field(..., description="thumbs_up or thumbs_down")
    session_id: str | None = None
    message_content: str | None = None
    context: str | None = None
    user_message: str | None = None
    assistant_response: str | None = None

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
    session_id: str | None = None


class ConversationUpdate(BaseModel):
    name: str | None = None
    pinned: bool | None = None
    starred: bool | None = None


class ConversationResponse(BaseModel):
    id: str
    name: str
    session_id: str
    created_at: str
    updated_at: str
    pinned: bool = False
    starred: bool = False
    message_count: int = 0
