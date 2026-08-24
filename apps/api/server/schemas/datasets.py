"""
Datasets Schemas - Data models for datasets
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class DatasetInfo(BaseModel):
    id: str
    name: str
    path: str
    type: str = "text"
    size_bytes: int = 0
    size_formatted: str = "Empty"
    num_samples: int = 0
    visual_metadata: Optional[Dict[str, Any]] = None


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = None
    source: Optional[str] = None


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class DatasetDataRequest(BaseModel):
    data: list[str]


class DatasetStats(BaseModel):
    format: str = "text"
    samples: int = 0
    chars: int = 0
    avg_length: float = 0.0
    has_messages: bool = False
    sample_preview: List[str] = []
    lines: int = 0
    suggested_method: str = "unknown"
    file_type: str = "txt"


class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]
    count: int


class SearchResponse(BaseModel):
    results: List[str]
    count: int


class GitHubImportRequest(BaseModel):
    url: str = Field(..., pattern=r'^https?://', max_length=2000)
    name: str = Field(..., min_length=1, max_length=200)
    extensions: Optional[List[str]] = None
    max_files: Optional[int] = Field(default=None, ge=1, le=10000)


class HuggingFaceImportRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1, max_length=200)
    name: Optional[str] = Field(default=None, max_length=200)


class URLImportRequest(BaseModel):
    url: str = Field(..., pattern=r'^https?://', max_length=2000)
    name: str = Field(..., min_length=1, max_length=200)


class LocalImportRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=1000)
    name: str = Field(..., min_length=1, max_length=200)
    extensions: Optional[List[str]] = None


class KaggleImportRequest(BaseModel):
    dataset: str = Field(..., min_length=1, max_length=200)
    name: Optional[str] = Field(default=None, max_length=200)


class CSVImportRequest(BaseModel):
    url: str = Field(..., pattern=r'^https?://', max_length=2000)
    name: str = Field(..., min_length=1, max_length=200)
    delimiter: Optional[str] = Field(default=", max_length=1")
    encoding: Optional[str] = Field(default="utf-8", max_length=50)


class BatchImportSource(BaseModel):
    type: str
    name: Optional[str] = None
    url: Optional[str] = None
    path: Optional[str] = None
    dataset_id: Optional[str] = None
    extensions: Optional[List[str]] = None


class BatchImportRequest(BaseModel):
    sources: List[BatchImportSource]


class ISBNImportRequest(BaseModel):
    isbn: str = Field(min_length=10, max_length=13)
    name: Optional[str] = None


class ImportResponse(BaseModel):
    success: bool = True
    dataset_id: str
    name: Optional[str] = None
    message: str = ""
    files_imported: int = 0
    total_chars: int = 0
    output_path: Optional[str] = None
    status: str = "imported"


class VersionCreateResponse(BaseModel):
    """Response for creating a new dataset version (snapshot)."""
    timestamp: str
    message: str


class VersionListResponse(BaseModel):
    """List of version timestamps for a dataset."""
    versions: List[str]
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
    messages: List[ChatMessage]
    name: str = Field(default="chat-export", max_length=100)
