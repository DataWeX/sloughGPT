"""
Domain data models for the Data Repository layer.

Each dataclass represents a persisted entity. These replace ad-hoc
dict shapes scattered across routers and controllers with a single
schema definition per domain.

Usage:
    from domains.infrastructure.data_models import SessionData, KnowledgeFactData
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Session ──


@dataclass
class SessionData:
    """A chat session with its message history."""

    id: str
    messages: list[dict] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Conversation ──


@dataclass
class ConversationData:
    """A persisted conversation (client-facing list item)."""

    id: str
    session_id: str
    name: str
    created_at: str = ""
    updated_at: str = ""
    pinned: bool = False
    starred: bool = False
    message_count: int = 0


# ── Feedback ──


@dataclass
class FeedbackData:
    """User feedback on a model response."""

    id: str
    message_id: str
    rating: str  # "thumbs_up" | "thumbs_down"
    session_id: str | None = None
    context: str | None = None
    quality_score: float | None = None
    created_at: str | None = None


@dataclass
class FeedbackStatsData:
    """Aggregated feedback statistics."""

    thumbs_up: int = 0
    thumbs_down: int = 0
    total: int = 0
    up_ratio: float = 0.0

    def __post_init__(self):
        if self.total == 0 and (self.thumbs_up or self.thumbs_down):
            self.total = self.thumbs_up + self.thumbs_down
        if self.total > 0 and self.up_ratio == 0.0:
            self.up_ratio = self.thumbs_up / self.total


# ── Knowledge ──


@dataclass
class KnowledgeFactData:
    """A single knowledge fact stored in the knowledge base."""

    content: str
    topic: str = "general"
    source: str = "manual"
    url: str = ""
    timestamp: float = 0.0
    importance: float = 0.5
    score: float = 0.0
    tags: list[str] = field(default_factory=list)
    id: str = ""


@dataclass
class KnowledgeEntryData:
    """A knowledge entry with vector embedding metadata for persistence."""

    id: str
    text: str
    vector: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return self.text


# ── Dataset ──


@dataclass
class DatasetData:
    """Metadata about an imported or created dataset."""

    id: str
    name: str
    path: str = ""
    type: str = "text"  # "text" | "corpus" | "visual"
    size_bytes: int = 0
    num_samples: int = 0
    description: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Agent ──


@dataclass
class AgentData:
    """An AI agent definition."""

    id: str
    name: str
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    tools: list[str] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


# ── Training Job ──


@dataclass
class TrainingJobData:
    """A training job record."""

    id: str
    name: str
    status: str  # "pending" | "running" | "completed" | "failed"
    model: str = ""
    dataset: str = ""
    epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    progress: int = 0
    loss: float | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


# ── Session Context ──


@dataclass
class SessionContextData:
    """Stored context for session regeneration."""

    session_id: str
    system_prompt: str = ""
    knowledge: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    updated_at: str = ""


__all__ = [
    "SessionData",
    "ConversationData",
    "FeedbackData",
    "FeedbackStatsData",
    "KnowledgeFactData",
    "KnowledgeEntryData",
    "DatasetData",
    "AgentData",
    "TrainingJobData",
    "SessionContextData",
]
