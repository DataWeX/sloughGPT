from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceConfig:
    type: str = "file"
    path: str = ""
    url: str = ""
    name: str = ""
    timeout: int = 30
    poll_interval: float = 60.0
    headers: dict = field(default_factory=dict)


@dataclass
class StoreConfig:
    type: str = "file"
    path: str = ""
    name: str = ""
    max_size: int = 10000


@dataclass
class FilterConfig:
    type: str = "length"
    min_length: int = 10
    max_length: int = 100000
    keywords: list[str] = field(default_factory=list)
    mode: str = "include"
    pattern: str = ""
    allowed_chars_ratio: float = 0.8


@dataclass
class PipelineConfig:
    name: str = ""
    source: SourceConfig = field(default_factory=SourceConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    filters: list[FilterConfig] = field(default_factory=list)
    collect_interval: float = 0.0
    max_rounds: int | None = None
