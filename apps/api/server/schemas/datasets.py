"""
Datasets Schemas - Data models for datasets
"""

from typing import Any

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    id: str
    name: str
    path: str
    type: str = "text"
    size_bytes: int = 0
    size_formatted: str = "Empty"
    num_samples: int = 0
    visual_metadata: dict[str, Any] | None = None


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str | None = None
    source: str | None = None


class DatasetUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class DatasetDataRequest(BaseModel):
    data: list[str]


class DatasetStats(BaseModel):
    format: str = "text"
    samples: int = 0
    chars: int = 0
    avg_length: float = 0.0
    has_messages: bool = False
    sample_preview: list[str] = []
    lines: int = 0
    suggested_method: str = "unknown"
    file_type: str = "txt"


class DatasetListResponse(BaseModel):
    datasets: list[DatasetInfo]
    count: int


class SearchResponse(BaseModel):
    results: list[str]
    count: int


class GitHubImportRequest(BaseModel):
    url: str = Field(..., pattern=r"^https?://", max_length=2000)
    name: str = Field(..., min_length=1, max_length=200)
    extensions: list[str] | None = None
    max_files: int | None = Field(default=None, ge=1, le=10000)


class HuggingFaceImportRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1, max_length=200)
    name: str | None = Field(default=None, max_length=200)


class URLImportRequest(BaseModel):
    url: str = Field(..., pattern=r"^https?://", max_length=2000)
    name: str = Field(..., min_length=1, max_length=200)


class LocalImportRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=1000)
    name: str = Field(..., min_length=1, max_length=200)
    extensions: list[str] | None = None


class KaggleImportRequest(BaseModel):
    dataset: str = Field(..., min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9_\-/]+$")
    name: str | None = Field(default=None, max_length=200)


class CSVImportRequest(BaseModel):
    url: str = Field(..., pattern=r"^https?://", max_length=2000)
    name: str = Field(..., min_length=1, max_length=200)
    delimiter: str | None = Field(default=", max_length=1")
    encoding: str | None = Field(default="utf-8", max_length=50)


class BatchImportSource(BaseModel):
    type: str
    name: str | None = None
    url: str | None = None
    path: str | None = None
    dataset_id: str | None = None
    extensions: list[str] | None = None


class BatchImportRequest(BaseModel):
    sources: list[BatchImportSource]


class ISBNImportRequest(BaseModel):
    isbn: str = Field(min_length=10, max_length=13)
    name: str | None = None


class ImportResponse(BaseModel):
    success: bool = True
    dataset_id: str
    name: str | None = None
    message: str = ""
    files_imported: int = 0
    total_chars: int = 0
    output_path: str | None = None
    status: str = "imported"


class VersionCreateResponse(BaseModel):
    """Response for creating a new dataset version (snapshot)."""

    timestamp: str
    message: str


class VersionListResponse(BaseModel):
    """List of version timestamps for a dataset."""

    versions: list[str]
    count: int


class VersionRestoreResponse(BaseModel):
    """Result of a version restore operation."""

    success: bool
    message: str


class ChatMessage(BaseModel):
    """A single chat message for dataset import."""

    role: str = Field(pattern=r"^(user|assistant|system)$")
    content: str


class DatasetExportRequest(BaseModel):
    format: str = Field(default="jsonl", pattern="^(jsonl|csv)$")


class FromChatRequest(BaseModel):
    """Request body for POST /datasets/from-chat."""

    messages: list[ChatMessage]
    name: str = Field(default="chat-export", max_length=100)
